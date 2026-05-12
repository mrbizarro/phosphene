#!/usr/bin/env python3.11
"""Scenema audio client — character-aware TTS for Phosphene video renders.

Scenema (https://github.com/ScenemaAI/scenema-audio) provides zero-shot
voice cloning + emotional-direction TTS extracted from the LTX-2.3
audiovisual model. We use it to replace the LTX video pipeline's audio
output entirely when rendering with a Train-Character LoRA: silent
video + voice-cloned TTS using the character's reference clip.

This module is deployment-agnostic: same client works against a RunPod
self-hosted GPU pod, a future native MLX port, or anything else that
implements POST /generate per the upstream server.py contract.

API: pass SCENEMA_URL via env var or constructor; default is the local
MLX port placeholder.

Wire format for the request (from scenema-audio/src/server.py:138-167):
    POST /generate JSON
    {
      "prompt": "<speak voice='...' gender='...'><action>...</action>text</speak>",
      "reference_voice_url": "http://...",       # HTTP-fetchable URL only
      "mode": "generate",
      "background_sfx": false,
      "validate": true,
      "seed": -1,
      "pace": 1.5,
      "min_match_ratio": 0.90,
      "skip_vc": false,
      "vc_steps": 25,
      "vc_cfg_rate": 0.5
    }
    -> { "status": "succeeded", "audio": "<base64 WAV>", ... }

Reference URL gotcha: server.py:_download_reference uses httpx to GET the
URL, so it MUST be HTTP-fetchable from the Scenema server's network
namespace. For a remote pod this means: either upload the ref to a
public URL (CDN, S3 presigned), OR tunnel back via reverse-ssh.
The serve_reference() helper below handles the local case by spinning
up a one-shot HTTP server on the LAN/Tailscale interface.
"""

from __future__ import annotations

import base64
import contextlib
import http.server
import json
import os
import pathlib
import shlex
import socket
import socketserver
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request


SCENEMA_URL = os.environ.get("SCENEMA_URL", "http://127.0.0.1:8000/generate")
SCENEMA_TIMEOUT_SEC = int(os.environ.get("SCENEMA_TIMEOUT_SEC", "600"))

# Pinokio bundles ffmpeg under its own prefix; use that path explicitly so
# we don't depend on PATH ordering (the panel often has a sanitised PATH).
FFMPEG_BIN = os.environ.get(
    "PHOSPHENE_FFMPEG",
    "/Users/salo/pinokio/bin/ffmpeg-env/bin/ffmpeg",
)


# ---------------------------------------------------------------------------
# Reference-URL plumbing
# ---------------------------------------------------------------------------

def _lan_addr() -> str:
    """Pick an address Scenema can reach back on. For a RunPod deployment,
    set SCENEMA_REF_BASE_URL to a public host (e.g. an ngrok / Tailscale
    funnel / S3 presign). Local self-host: 127.0.0.1 is fine."""
    explicit = os.environ.get("SCENEMA_REF_BASE_URL", "").strip()
    if explicit:
        return explicit.rstrip("/")
    return "http://127.0.0.1"


@contextlib.contextmanager
def serve_reference(local_path: str):
    """Yield a URL that Scenema can fetch the local reference clip from.

    Spins up a tiny single-file HTTP server scoped to the file's parent
    directory. The server shuts down when the context exits, so refs
    don't leak as long-running listeners. Picks an ephemeral port."""
    p = pathlib.Path(local_path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"reference clip not found: {local_path}")

    parent = str(p.parent)

    class _Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=parent, **k)

        def log_message(self, fmt, *args):  # silence default access log
            return

    # Bind on 0.0.0.0 so a remote Scenema pod can reach us if the user has
    # tunnelled. For 127.0.0.1-only deployments it works the same way.
    srv = socketserver.TCPServer(("0.0.0.0", 0), _Handler)
    srv.allow_reuse_address = True
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        port = srv.server_address[1]
        base = _lan_addr()
        # If SCENEMA_REF_BASE_URL already includes a port, trust it as-is.
        if ":" in base.split("//", 1)[-1]:
            url = f"{base}/{p.name}"
        else:
            url = f"{base}:{port}/{p.name}"
        yield url
    finally:
        srv.shutdown()
        srv.server_close()


# ---------------------------------------------------------------------------
# Scenema request
# ---------------------------------------------------------------------------

def _build_speak_prompt(text: str,
                        voice_desc: str = "Adult speaking voice.",
                        gender: str = "male",
                        scene: str | None = None,
                        actions: list[tuple[int, str]] | None = None) -> str:
    """Build a Scenema <speak> XML prompt from plain text + voice metadata.

    `actions` is an optional list of (insert-after-char-index, direction)
    tuples for `<action>...</action>` tags. If omitted, a single neutral
    action prefix is used."""
    # Minimal escape for & and < in body text; Scenema's parser is lenient
    # but invalid XML can confuse the validator.
    safe = (text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;"))
    open_attrs = [f'voice="{voice_desc}"', f'gender="{gender}"']
    if scene:
        open_attrs.append(f'scene="{scene}"')
    prefix = "<action>Natural conversational tone.</action>\n"
    return f"<speak {' '.join(open_attrs)}>\n{prefix}{safe}\n</speak>"


def generate_audio(speech_text: str,
                   reference_voice_path: str,
                   *,
                   voice_desc: str = "Adult speaking voice.",
                   gender: str = "male",
                   seed: int = -1,
                   pace: float = 1.5,
                   skip_vc: bool = False,
                   vc_steps: int = 25,
                   background_sfx: bool = False,
                   url: str | None = None,
                   timeout_sec: int | None = None) -> bytes:
    """Generate voice-cloned audio from a reference clip. Returns raw WAV bytes.

    Raises RuntimeError on Scenema-side failure (status != "succeeded").
    Raises urllib.error.URLError on network failure."""
    endpoint = url or SCENEMA_URL
    timeout = timeout_sec or SCENEMA_TIMEOUT_SEC

    with serve_reference(reference_voice_path) as ref_url:
        body = {
            "prompt": _build_speak_prompt(speech_text, voice_desc, gender),
            "reference_voice_url": ref_url,
            "mode": "generate",
            "background_sfx": background_sfx,
            "validate": False,    # validator is whisper-small re-check; slows things, keep off for first integration
            "seed": seed,
            "pace": pace,
            "skip_vc": skip_vc,
            "vc_steps": vc_steps,
            "vc_cfg_rate": 0.5,
        }
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))

    status = data.get("status")
    if status != "succeeded":
        raise RuntimeError(
            f"Scenema generate failed: status={status!r} "
            f"error={data.get('error') or data.get('message') or '(no message)'}"
        )
    audio_b64 = data.get("audio")
    if not audio_b64:
        raise RuntimeError("Scenema response missing 'audio' field")
    return base64.b64decode(audio_b64)


# ---------------------------------------------------------------------------
# Video muxing
# ---------------------------------------------------------------------------

def _mux_video_audio(video_path: str, audio_path: str, out_path: str) -> None:
    """ffmpeg mux: copy video stream, encode audio to AAC, lock duration
    to the shorter of the two so we don't get a trailing audio overhang
    on short clips."""
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-i", video_path,
        "-i", audio_path,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg mux failed (exit {result.returncode}):\n"
            f"cmd: {' '.join(shlex.quote(c) for c in cmd)}\n"
            f"stderr tail:\n{result.stderr[-1500:]}"
        )


def voice_clone_overlay(video_path: str,
                        reference_voice_path: str,
                        speech_text: str,
                        out_path: str,
                        **scenema_kwargs) -> str:
    """End-to-end: silent (or LTX-audio) input mp4 → muxed mp4 with the
    LTX audio replaced by Scenema voice-cloned TTS reading `speech_text`
    in the voice of `reference_voice_path`.

    Returns the output path on success. Raises on Scenema or ffmpeg failure.
    The intermediate audio WAV is written to a tempfile and cleaned up."""
    audio_bytes = generate_audio(speech_text, reference_voice_path,
                                 **scenema_kwargs)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp_audio = f.name
    try:
        _mux_video_audio(video_path, tmp_audio, out_path)
    finally:
        try:
            os.unlink(tmp_audio)
        except OSError:
            pass
    return out_path


# ---------------------------------------------------------------------------
# Self-test / CLI
# ---------------------------------------------------------------------------

def _ping(url: str | None = None, timeout: int = 5) -> tuple[bool, str]:
    """Lightweight reachability probe — doesn't run a model job."""
    endpoint = (url or SCENEMA_URL).replace("/generate", "/health")
    try:
        with urllib.request.urlopen(endpoint, timeout=timeout) as r:
            return r.status == 200, r.read().decode("utf-8", errors="ignore")[:200]
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}"
    except urllib.error.URLError as e:
        return False, f"unreachable: {e.reason}"
    except socket.timeout:
        return False, "timeout"


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Scenema voice-clone client smoke test")
    ap.add_argument("--ping", action="store_true",
                    help="Probe SCENEMA_URL/health and exit")
    ap.add_argument("--video", help="Input mp4 (silent or audio-bearing)")
    ap.add_argument("--ref", help="Reference voice WAV/MP3 path")
    ap.add_argument("--text", help="Speech text to synthesise")
    ap.add_argument("--voice-desc", default="Adult speaking voice.",
                    help='Scenema voice description, e.g. "Middle-aged man, warm but weathered."')
    ap.add_argument("--gender", default="male", choices=["male", "female", "neutral"])
    ap.add_argument("--out", default=None,
                    help="Output mp4 (default: <video>.voice.mp4)")
    ap.add_argument("--url", default=None, help="Override SCENEMA_URL for this run")
    args = ap.parse_args()

    if args.ping:
        ok, msg = _ping(args.url)
        print(f"reachable={ok} body={msg!r}")
        raise SystemExit(0 if ok else 1)

    for required in ("video", "ref", "text"):
        if not getattr(args, required):
            ap.error(f"--{required} required (or use --ping)")

    out = args.out or args.video.rsplit(".", 1)[0] + ".voice.mp4"
    t0 = time.time()
    print(f"[scenema] generating audio for {len(args.text)} chars...")
    voice_clone_overlay(args.video, args.ref, args.text, out,
                        voice_desc=args.voice_desc, gender=args.gender,
                        url=args.url)
    print(f"[scenema] done in {time.time() - t0:.1f}s → {out}")
