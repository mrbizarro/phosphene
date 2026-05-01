#!/usr/bin/env python3
"""LTX MLX Studio — local control panel for LTX 2.3 video generation on Apple Silicon.

Features:
- Persistent batch queue with crash-resume (panel_queue.json)
- Warm helper subprocess holding MLX pipelines (mlx_warm_helper.py)
- Image cover-crop resize for I2V (handled inside the helper)
- Extend mode: chain clips for 15s+ via the dev-transformer ExtendPipeline
- Aspect / duration / frames driven by a single Duration field
- Hide outputs without deleting from disk (panel_hidden.json)
- caffeinate -i on while queue runs so the Mac doesn't idle-sleep
"""
from __future__ import annotations

import atexit
import cgi
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

# --- Paths -------------------------------------------------------------------
# Everything below is overridable via env vars so the panel can be cloned and
# run from any directory without source edits. Defaults assume the repo layout:
#
#   <repo>/                      <- LTX_STUDIO_ROOT (defaults to script's dir)
#     mlx_ltx_panel.py
#     mlx_warm_helper.py
#     ltx-2-mlx/                 <- LTX_MLX_PATH (sibling clone)
#       .venv/bin/python3.11     <- LTX_HELPER_PYTHON
#     mlx_models/                <- LTX_MODELS_DIR
#       gemma-3-12b-it-4bit/     <- LTX_GEMMA_PATH
#     mlx_outputs/               <- LTX_OUTPUT_DIR (created on first run)
#     panel_uploads/             <- LTX_UPLOADS_DIR (created on first run)
ROOT = Path(os.environ.get("LTX_STUDIO_ROOT", str(Path(__file__).resolve().parent)))
MLX = Path(os.environ.get("LTX_MLX_PATH", str(ROOT / "ltx-2-mlx")))
MODELS_DIR = Path(os.environ.get("LTX_MODELS_DIR", str(ROOT / "mlx_models")))
GEMMA = Path(os.environ.get("LTX_GEMMA_PATH", str(MODELS_DIR / "gemma-3-12b-it-4bit")))
OUTPUT = Path(os.environ.get("LTX_OUTPUT_DIR", str(ROOT / "mlx_outputs")))
UPLOADS = Path(os.environ.get("LTX_UPLOADS_DIR", str(ROOT / "panel_uploads")))
AUDIO_DEFAULT = Path(os.environ.get("LTX_DEFAULT_AUDIO", str(ROOT / "audio_inputs/default.wav")))
REFERENCE = Path(os.environ.get("LTX_DEFAULT_IMAGE", str(ROOT / "examples/reference.png")))
HELPER_PYTHON = Path(os.environ.get("LTX_HELPER_PYTHON", str(MLX / ".venv/bin/python3.11")))
HELPER_SCRIPT = Path(os.environ.get("LTX_HELPER_SCRIPT", str(ROOT / "mlx_warm_helper.py")))
# ffmpeg: env var → PATH → Pinokio bundled → Homebrew → /usr/local. First match wins.
def _resolve_ffmpeg() -> Path:
    candidates = [
        os.environ.get("LTX_FFMPEG"),
        shutil.which("ffmpeg"),
        str(Path.home() / "pinokio/bin/ffmpeg-env/bin/ffmpeg"),
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return Path(c)
    return Path("/usr/local/bin/ffmpeg")  # last-resort default; will fail at runtime if missing


FFMPEG = _resolve_ffmpeg()
FFMPEG_BIN = FFMPEG.parent
FFPROBE = FFMPEG.parent / "ffprobe"  # ships next to ffmpeg in every distribution we support

MODEL_ID = os.environ.get("LTX_MODEL", "dgrauet/ltx-2.3-mlx-q4")
MODEL_ID_HQ = os.environ.get("LTX_MODEL_HQ", "dgrauet/ltx-2.3-mlx-q8")
# Q8 model is detected on disk so the High quality tier can be conditionally enabled.
Q8_LOCAL_PATH = Path(os.environ.get("LTX_Q8_LOCAL", str(ROOT / "mlx_models/ltx-2.3-mlx-q8")))
COMFY_PATTERN = os.environ.get("LTX_COMFY_PATTERN", "pinokio/api/comfy.git.*main\\.py")
QUEUE_FILE = ROOT / "panel_queue.json"
HIDDEN_FILE = ROOT / "panel_hidden.json"
HELPER_IDLE_TIMEOUT = int(os.environ.get("LTX_HELPER_IDLE_TIMEOUT", "1800"))
HELPER_LOW_MEMORY = os.environ.get("LTX_HELPER_LOW_MEMORY", "true")
FPS = 24
PORT = int(os.environ.get("LTX_PORT", "8198"))
HISTORY_LIMIT = 200
HISTORY_PERSIST_LIMIT = 100  # how many history entries to write to panel_queue.json
HISTORY_API_LIMIT = 50       # how many to expose via /status
LOG_LIMIT = 1000

# ---- single source of truth: required files for "installed" ----------------
# Loaded from required_files.json so pinokio.js, this panel, and install.js
# all check the SAME list. When upstream model layout changes, edit ONE file.
def _load_required_files() -> dict:
    """Read required_files.json. On any failure, return a sane fallback so a
    corrupt/missing JSON doesn't take the panel down — log it loudly though."""
    fallback = {"repos": [], "env": {"marker_paths": []}, "min_size_bytes": 1024}
    try:
        with open(ROOT / "required_files.json", "r") as fh:
            data = json.load(fh)
        for key in ("repos", "env", "min_size_bytes"):
            if key not in data:
                raise ValueError(f"required_files.json missing key: {key}")
        return data
    except Exception as exc:
        sys.stderr.write(f"WARN: required_files.json unreadable ({exc}); "
                         f"completeness checks will under-report. Reinstall to fix.\n")
        return fallback


_REQUIRED = _load_required_files()
_MIN_FILE_BYTES = int(_REQUIRED.get("min_size_bytes", 1024))


# ---- panel settings: user-controllable preferences -------------------------
# Persisted to panel_settings.json so the user's choice survives restarts.
# Read at startup, exposed via /settings GET, mutated via /settings POST.
# Currently covers output codec (the cause of the original "X rejects my
# upload" complaint and the "videos are huge" complaint — lossless yuv444p
# crf 0 is great for archives but wrong as a default for social workflows).
#
# Schema:
#   {
#     "version": 1,
#     "output_preset": "standard" | "archival" | "web" | "custom",
#     "output_pix_fmt": "yuv420p" | "yuv444p" | <other valid ffmpeg pix_fmt>,
#     "output_crf": "0" .. "30" (string, since we pass it via env vars)
#   }
#
# `output_preset` is metadata so the UI can show which preset is active even
# after a reload. The actual ffmpeg behavior is driven by `output_pix_fmt`
# and `output_crf` which get exported into the helper subprocess's env.

# Preset table — single source of truth for the UI and the install default.
# Sizes are rough estimates for a 5s 1280x704 H.264 clip (yuv420p crf 18 is
# the long-standing "visually lossless" default for web video).
OUTPUT_PRESETS: dict[str, dict[str, str]] = {
    "archival": {
        # Internal key stays "archival" so any settings files already on
        # disk keep working. The user-facing label is "Video production"
        # because that's who actually picks this — colorists, editors,
        # VFX folks who'll grade or composite the clip downstream.
        "pix_fmt": "yuv444p",
        "crf": "0",
        "label": "Video production (lossless)",
        "blurb": "Mathematically lossless — full 4:4:4 chroma, no "
                 "compression artifacts. ~50 MB per 5s clip. For pro "
                 "workflows: color grading, compositing, VFX, anywhere "
                 "you'll re-encode downstream and need every frame intact.",
    },
    "standard": {
        "pix_fmt": "yuv420p",
        "crf": "18",
        "label": "Standard",
        "blurb": "Visually lossless to ~95% of viewers. ~7 MB per 5s clip. "
                 "Plays everywhere including X / Instagram / Discord. The "
                 "default for new installs.",
    },
    "web": {
        "pix_fmt": "yuv420p",
        "crf": "23",
        "label": "Web / social",
        "blurb": "Smallest files. ~3 MB per 5s clip. For embedding, mobile, "
                 "or quick previews where bandwidth matters more than peak "
                 "fidelity.",
    },
}

# Default preset for fresh installs. Switched from "archival" to "standard"
# after multiple users reported X upload failures (X rejects yuv444p) and
# disk-fill complaints about ~50 MB clips. Existing installs keep whatever
# is in their panel_settings.json; new installs and panels with no settings
# file yet get this default.
DEFAULT_OUTPUT_PRESET = "standard"

SETTINGS_FILE = ROOT / "panel_settings.json"
_SETTINGS_LOCK = threading.Lock()


def _settings_defaults() -> dict:
    preset = OUTPUT_PRESETS[DEFAULT_OUTPUT_PRESET]
    return {
        "version": 2,                                # bumped: added secrets
        "output_preset": DEFAULT_OUTPUT_PRESET,
        "output_pix_fmt": preset["pix_fmt"],
        "output_crf": preset["crf"],
        # Secrets — empty means "not configured". Never exposed by GET
        # /settings; the API surfaces only `has_civitai_key` /
        # `has_hf_token` booleans so the modal can render status pills
        # without leaking the values back to the frontend (defense
        # against the panel ever being exposed beyond loopback).
        "civitai_api_key": "",
        "hf_token": "",
        # When true, the inline "models ready" / "Q8 not installed" card
        # at the top of the form is suppressed across reloads. Set when
        # the user clicks × on that card. The card auto-clears this flag
        # when the model state regresses (e.g. a download disappears) so
        # a real new problem still surfaces. UI-only; no security impact.
        "models_card_dismissed": False,
    }


def _load_settings() -> dict:
    """Read panel_settings.json. Missing file → return + write the default
    so first-run users get the sensible Standard preset. Corrupt file →
    fall back to defaults but DON'T overwrite (preserves the bad file for
    forensic inspection if it was edited by hand)."""
    if not SETTINGS_FILE.exists():
        defaults = _settings_defaults()
        try:
            SETTINGS_FILE.write_text(json.dumps(defaults, indent=2))
        except Exception as exc:
            sys.stderr.write(f"WARN: could not write {SETTINGS_FILE} ({exc})\n")
        return defaults
    try:
        with SETTINGS_FILE.open("r") as fh:
            data = json.load(fh)
        # Backfill missing keys against defaults so older settings files
        # don't trip over fields added in later versions.
        defaults = _settings_defaults()
        for k, v in defaults.items():
            data.setdefault(k, v)
        return data
    except Exception as exc:
        sys.stderr.write(f"WARN: panel_settings.json unreadable ({exc}); "
                         f"using defaults until manually fixed\n")
        return _settings_defaults()


def _save_settings(settings: dict) -> None:
    """Atomic write so a Pinokio kill mid-write can't leave a half-file."""
    import tempfile, os as _os
    fd, tmp = tempfile.mkstemp(prefix="panel_settings.", dir=str(ROOT))
    try:
        with _os.fdopen(fd, "w") as fh:
            fh.write(json.dumps(settings, indent=2))
            fh.flush()
            _os.fsync(fh.fileno())
        _os.replace(tmp, SETTINGS_FILE)
    except Exception:
        try: _os.unlink(tmp)
        except OSError: pass
        raise


def _validate_settings_patch(patch: dict) -> tuple[dict, str | None]:
    """Validate user-submitted partial settings, return (clean, error_or_None).
    Whitelist what we accept — never trust the form payload to be safe to
    pass to ffmpeg / shell."""
    out: dict = {}

    if "output_preset" in patch:
        preset = str(patch["output_preset"]).strip().lower()
        if preset not in OUTPUT_PRESETS and preset != "custom":
            return {}, f"unknown output_preset: {preset}"
        out["output_preset"] = preset
        if preset != "custom":
            # Preset overrides any pix_fmt/crf in the same patch — picking
            # "Standard" should always give Standard's values. Note: we
            # FALL THROUGH to the rest of validation (don't early-return)
            # because the same patch may also carry token fields that
            # the JS Apply path always sends together with the preset.
            # An earlier version of this function had `return out, None`
            # here, which silently dropped civitai_api_key + hf_token
            # whenever the user clicked Apply on a non-custom preset —
            # i.e. always, in practice. Reproduced as: panel reports
            # has_civitai_key=False after a successful Apply with a
            # valid key in the form.
            out["output_pix_fmt"] = OUTPUT_PRESETS[preset]["pix_fmt"]
            out["output_crf"] = OUTPUT_PRESETS[preset]["crf"]
            # Skip the per-field pix_fmt/crf validation below; the
            # preset already populated them. But continue to the token
            # checks further down.

    # Per-field pix_fmt + crf validation (custom path, or when no
    # preset was supplied). Skipped automatically when the preset
    # branch above filled them in.
    if "output_pix_fmt" in patch and "output_pix_fmt" not in out:
        pf = str(patch["output_pix_fmt"]).strip().lower()
        # ffmpeg has many pix_fmts; whitelist the ones LTX 2.3 actually
        # produces correctly. Anything else is a footgun (color shifts,
        # encoder errors).
        ALLOWED_PIX_FMTS = {"yuv420p", "yuv422p", "yuv444p", "yuv420p10le",
                            "yuv422p10le", "yuv444p10le"}
        if pf not in ALLOWED_PIX_FMTS:
            return {}, (f"output_pix_fmt must be one of: "
                        f"{sorted(ALLOWED_PIX_FMTS)}")
        out["output_pix_fmt"] = pf

    if "output_crf" in patch and "output_crf" not in out:
        try:
            crf_i = int(str(patch["output_crf"]))
        except (TypeError, ValueError):
            return {}, "output_crf must be an integer 0-30"
        if not 0 <= crf_i <= 30:
            return {}, "output_crf must be between 0 (lossless) and 30 (low)"
        out["output_crf"] = str(crf_i)

    # Tokens. Sanity-check shape (no whitespace, reasonable length) but
    # don't hard-validate against the upstream API here — that adds
    # latency and the upstream services are the authoritative oracle
    # anyway. The Settings UI shows a connectivity-test button if the
    # user wants to verify before using.
    if "civitai_api_key" in patch:
        key = str(patch["civitai_api_key"]).strip()
        if key and not (8 <= len(key) <= 256):
            return {}, "civitai_api_key length looks wrong (expected 8–256 chars)"
        if any(c.isspace() for c in key):
            return {}, "civitai_api_key cannot contain whitespace"
        out["civitai_api_key"] = key

    if "hf_token" in patch:
        token = str(patch["hf_token"]).strip()
        # HF tokens start with "hf_" and are typically ~40 chars total.
        if token and not (token.startswith("hf_") and 20 <= len(token) <= 256):
            return {}, "hf_token must start with 'hf_' (get one at https://huggingface.co/settings/tokens)"
        if any(c.isspace() for c in token):
            return {}, "hf_token cannot contain whitespace"
        out["hf_token"] = token

    if "models_card_dismissed" in patch:
        # Accept the urlencoded "true"/"false"/"1"/"0" forms as well as a
        # bool — JS posts via URLSearchParams which stringifies bools.
        v = patch["models_card_dismissed"]
        if isinstance(v, bool):
            out["models_card_dismissed"] = v
        else:
            out["models_card_dismissed"] = str(v).strip().lower() in ("1", "true", "yes", "on")

    return out, None


def _active_civitai_key() -> str:
    """Resolve the CivitAI key in this priority:
       1. Saved panel setting (UI). Most users.
       2. CIVITAI_API_KEY env var. Power users with shell config.
    Returns empty string when neither is set."""
    saved = get_settings().get("civitai_api_key", "").strip()
    if saved:
        return saved
    return os.environ.get("CIVITAI_API_KEY", "").strip()


def _active_hf_token() -> str:
    """Resolve the HuggingFace token. Settings > HF_TOKEN env var.
    Note: huggingface_hub also reads ~/.cache/huggingface/token
    (the file `hf auth login` writes), and we don't override that —
    if neither settings nor the env var has a token, the library
    falls back to that cached file."""
    saved = get_settings().get("hf_token", "").strip()
    if saved:
        return saved
    return os.environ.get("HF_TOKEN", "").strip()


def get_settings_public() -> dict:
    """Settings shape safe to expose via GET /settings. Tokens are never
    returned — only booleans indicating whether each is configured. This
    matters because the panel binds to loopback only, but if a user ever
    proxies it (Pinokio's tunnel feature, ngrok, etc.) the secrets would
    leak with every status poll otherwise."""
    s = get_settings()
    return {
        "version": s.get("version", 1),
        "output_preset": s.get("output_preset"),
        "output_pix_fmt": s.get("output_pix_fmt"),
        "output_crf": s.get("output_crf"),
        "has_civitai_key": bool(s.get("civitai_api_key", "").strip()),
        "has_hf_token": bool(s.get("hf_token", "").strip()),
        "models_card_dismissed": bool(s.get("models_card_dismissed", False)),
    }


_SETTINGS = _load_settings()


def get_settings() -> dict:
    with _SETTINGS_LOCK:
        return dict(_SETTINGS)


def update_settings(patch: dict) -> tuple[dict, str | None]:
    """Apply a validated patch + persist + return (current, error_or_None).
    Caller is responsible for triggering a helper restart if codec settings
    changed (the ffmpeg call inside the helper reads env vars at job time,
    and the helper inherits env at spawn — so a running helper will keep
    using whatever was active when it spawned)."""
    clean, err = _validate_settings_patch(patch)
    if err:
        return get_settings(), err
    with _SETTINGS_LOCK:
        _SETTINGS.update(clean)
        try:
            _save_settings(_SETTINGS)
        except Exception as exc:
            return get_settings(), f"could not persist settings: {exc}"
        return dict(_SETTINGS), None


# ---- LoRA discovery + curated registry --------------------------------------
# LTX 2.3 supports LoRAs via apply_loras() in ltx_core_mlx.loader.fuse_loras.
# The pipeline's _pending_loras attribute is the integration hook — set it
# before the pipeline's load() and the deltas get fused into the transformer
# at quantization time. Path can be a local file (preferred for user-installed
# LoRAs) or a HuggingFace repo ID (used for the curated Lightricks officials).
#
# Two sources of LoRAs surface in the UI:
#
# 1. **Curated** — official Lightricks LoRAs we know about. Pinned by repo
#    ID so they always work without a separate download step (the helper
#    snapshot_downloads them on first use). The HDR LoRA is exposed as a
#    plain "HDR" toggle in the form; the rest live in the LoRA picker.
#
# 2. **User-installed** — anything dropped into mlx_models/loras/ as a
#    .safetensors file. Optional sidecar JSON next to the .safetensors
#    carries metadata (display name, trigger words, recommended strength,
#    preview thumbnail, source). The CivitAI browser writes these sidecars
#    automatically when it downloads a LoRA.

LORAS_DIR = ROOT / "mlx_models" / "loras"

# Curated Lightricks LoRAs. Keyed by short id; the UI exposes these next to
# user-installed LoRAs in the picker, with `is_curated: true` so the front
# end can render them differently (e.g. a "Lightricks" badge).
CURATED_LORAS: dict[str, dict] = {
    "hdr": {
        "id": "hdr",
        "name": "HDR",
        "description": "Lightricks' official HDR LoRA. Boosts dynamic range "
                       "and color depth. Exposed as a plain 'HDR' toggle in "
                       "the form, not buried under LoRAs — most users won't "
                       "want to know what's under the hood.",
        "repo_id": "Lightricks/LTX-2.3-22b-IC-LoRA-HDR",
        "default_strength": 1.0,
        "trigger_words": [],          # HDR is conditioning-style, no trigger
        "is_curated": True,
        "is_hdr_toggle": True,         # treat specially in the UI
    },
    "motion-track": {
        "id": "motion-track",
        "name": "Motion Track Control",
        "description": "Lightricks IC-LoRA for motion-tracked control. "
                       "Pairs with a video conditioning input for motion "
                       "transfer-style workflows.",
        "repo_id": "Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control",
        "default_strength": 1.0,
        "trigger_words": [],
        "is_curated": True,
        "is_hdr_toggle": False,
    },
    "union-control": {
        "id": "union-control",
        "name": "Union Control",
        "description": "Lightricks IC-LoRA combining multiple control signals "
                       "(depth, edges, pose) into one network.",
        "repo_id": "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control",
        "default_strength": 1.0,
        "trigger_words": [],
        "is_curated": True,
        "is_hdr_toggle": False,
    },
}


def _safe_loras_dir() -> Path:
    """Ensure mlx_models/loras/ exists. Idempotent."""
    LORAS_DIR.mkdir(parents=True, exist_ok=True)
    return LORAS_DIR


def _read_lora_sidecar(safetensors_path: Path) -> dict:
    """Read sidecar JSON next to a .safetensors LoRA. Falls back to bare
    filename + zero metadata when no sidecar is present, so users can
    drop in a raw .safetensors and get a usable picker entry without
    writing any metadata themselves.

    Sidecar schema (inspired by CivitAI's model JSON, kept compatible
    so a CivitAI download just writes its native shape):

      {
        "name": "Cinematic Post-Apocalyptic",
        "description": "...",
        "trigger_words": ["postapoc", "wasteland"],   // optional
        "recommended_strength": 0.8,                   // 0.0..1.5 typical
        "civitai_id": 2563394,                         // optional
        "civitai_version_id": 2880457,                 // optional
        "preview_url": "https://...",                  // optional
        "base_model": "LTXV 2.3",
        "downloaded_at": "2026-05-01T..."
      }
    """
    sidecar = safetensors_path.with_suffix(".json")
    meta = {
        "name": safetensors_path.stem.replace("_", " ").replace("-", " "),
        "description": "",
        "trigger_words": [],
        "recommended_strength": 1.0,
        "preview_url": None,
        "preview_type": None,        # "image" | "video" | None
        "base_model": None,
        "civitai_id": None,
        "civitai_version_id": None,
        "civitai_url": None,         # link back to source page (for "read instructions")
        "downloaded_at": None,
    }
    if sidecar.exists():
        try:
            with sidecar.open("r") as fh:
                user = json.load(fh) or {}
            for k in meta:
                if k in user and user[k] is not None:
                    meta[k] = user[k]
        except Exception as exc:
            sys.stderr.write(f"WARN: bad sidecar at {sidecar} ({exc}); "
                             f"falling back to filename\n")
    return meta


def list_user_loras() -> list[dict]:
    """Scan mlx_models/loras/ and return one entry per .safetensors found.
    Filenames are matched case-insensitive on the extension; subdirectories
    are not recursed (keeps the picker flat, no organizational ambiguity)."""
    out: list[dict] = []
    if not LORAS_DIR.exists():
        return out
    for path in sorted(LORAS_DIR.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".safetensors":
            continue
        try:
            size_bytes = path.stat().st_size
        except OSError:
            continue
        meta = _read_lora_sidecar(path)
        # Heuristic: if preview_type wasn't recorded in the sidecar (older
        # downloads pre-this-field), infer from the URL extension so the UI
        # picks <video> vs <img> correctly.
        preview_url = meta.get("preview_url")
        preview_type = meta.get("preview_type")
        if preview_url and not preview_type:
            preview_type = "video" if preview_url.lower().split("?")[0].endswith(".mp4") else "image"
        # Backfill civitai_url from civitai_id when the sidecar predates the
        # explicit civitai_url field.
        civitai_url = meta.get("civitai_url")
        if not civitai_url and meta.get("civitai_id"):
            civitai_url = f"https://civitai.com/models/{meta.get('civitai_id')}"
        out.append({
            "id": f"user:{path.name}",
            "name": meta["name"],
            "description": meta["description"],
            "path": str(path),
            "filename": path.name,
            "size_bytes": size_bytes,
            "trigger_words": list(meta.get("trigger_words") or []),
            "recommended_strength": float(meta.get("recommended_strength") or 1.0),
            "preview_url": preview_url,
            "preview_type": preview_type,
            "base_model": meta.get("base_model"),
            "civitai_id": meta.get("civitai_id"),
            "civitai_version_id": meta.get("civitai_version_id"),
            "civitai_url": civitai_url,
            "downloaded_at": meta.get("downloaded_at"),
            "is_curated": False,
        })
    return out


def list_curated_loras() -> list[dict]:
    """Curated entries surfaced in the UI even though the actual weights
    are pulled lazily from HuggingFace at first use. We hide the HDR one
    from the picker because it's exposed as a plain toggle elsewhere
    (see is_hdr_toggle filtering on the JS side)."""
    return [dict(v) for v in CURATED_LORAS.values()]


def parse_loras_from_form(form: dict) -> list[dict]:
    """Parse the loras=<JSON> form field that the UI submits with each job.
    Shape on the wire is a JSON array of {id, path, strength}. We trust
    `id` only for routing in this layer; what gets sent to the helper is
    just the (path, strength) pair the fuser needs."""
    raw = form.get("loras", [""])
    if isinstance(raw, list):
        raw = raw[0] if raw else ""
    raw = (raw or "").strip()
    if not raw:
        return []
    try:
        items = json.loads(raw)
    except json.JSONDecodeError:
        return []
    out: list[dict] = []
    if not isinstance(items, list):
        return out
    for item in items:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "").strip()
        if not path:
            continue
        try:
            strength = float(item.get("strength", 1.0))
        except (TypeError, ValueError):
            strength = 1.0
        # Clamp; LoRA strengths beyond ±2 are usually nonsense and risk
        # numerical issues during fusion.
        strength = max(-2.0, min(2.0, strength))
        out.append({"path": path, "strength": strength})
    return out


def _repos() -> list[dict]:
    """All repo entries from required_files.json (list, in order)."""
    return list(_REQUIRED.get("repos", []))


def _repo_missing(repo: dict) -> list[str]:
    """Files missing or zero-byte for one repo, expressed as bare filenames
    (relative to the repo's local_dir). Bare names are friendlier in the UI
    than full paths."""
    base = ROOT / repo["local_dir"]
    missing = []
    for fname in repo.get("files", []):
        p = base / fname
        try:
            if not p.exists() or p.stat().st_size < _MIN_FILE_BYTES:
                missing.append(fname)
        except OSError:
            missing.append(fname)
    return missing


def _repo_complete(repo: dict) -> bool:
    return not _repo_missing(repo)


# ---- HF cache fallback -------------------------------------------------------
# Background:
#   - Pinokio installs use `hf download --local-dir mlx_models/<repo>/`. Files
#     live at the canonical local_dir path the manifest declares.
#   - Manual / dev installs that pass an HF repo id as LTX_MODEL get the
#     weights resolved via huggingface_hub's cache at ~/.cache/huggingface/.
#     Same files, different location.
#
# Without this fallback, dev-env users whose models were already pulled into
# the HF cache (e.g. by an earlier `huggingface-cli download` or by the helper
# itself on first run) saw the panel report Q4 as "MISSING" even though every
# render was working perfectly. That was the confusing "honest note" — fixing
# it here so the modal honestly reflects "available, just not in mlx_models/".
def _hf_cache_root() -> Path:
    """Resolve HF cache root the same way huggingface_hub does — env vars
    in the order the library checks, falling back to ~/.cache/huggingface/hub."""
    explicit = (
        os.environ.get("HF_HUB_CACHE")
        or os.environ.get("HUGGINGFACE_HUB_CACHE")
    )
    if explicit:
        return Path(explicit)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    return Path.home() / ".cache/huggingface/hub"


def _repo_hf_cache_dir(repo_id: str) -> Path | None:
    """Return the most recent snapshot dir for repo_id in the HF cache, or
    None if not present. Layout: <cache>/models--<owner>--<repo>/snapshots/<rev>/"""
    safe = repo_id.replace("/", "--")
    base = _hf_cache_root() / f"models--{safe}" / "snapshots"
    if not base.is_dir():
        return None
    revs = []
    try:
        for d in base.iterdir():
            if d.is_dir():
                revs.append((d, d.stat().st_mtime))
    except OSError:
        return None
    if not revs:
        return None
    revs.sort(key=lambda t: t[1], reverse=True)
    return revs[0][0]


def _repo_missing_in_cache(repo: dict) -> list[str] | None:
    """Files missing from the HF cache snapshot for this repo, or None if
    the repo isn't in the cache at all (caller treats None as "fall back to
    local-dir reporting"). HF cache stores symlinks to ../blobs/<hash>; we
    follow them and check size on the actual blob."""
    cache_dir = _repo_hf_cache_dir(repo["repo_id"])
    if cache_dir is None:
        return None
    missing = []
    for fname in repo.get("files", []):
        p = cache_dir / fname
        try:
            actual = p.resolve()
            if not actual.exists() or actual.stat().st_size < _MIN_FILE_BYTES:
                missing.append(fname)
        except OSError:
            missing.append(fname)
    return missing


def base_missing() -> list[str]:
    """Aggregate of all base-kind repos' missing files (each prefixed with
    its local_dir for backwards-compat with existing /status consumers).

    Only checks the canonical Pinokio layout. If LTX_MODEL is set to an HF
    repo id rather than a local path, the helper resolves weights via the
    HF cache and the local-dir check would misreport — return [] there so
    the panel doesn't false-positive against a working manual install."""
    if not str(MODEL_ID).startswith("/"):
        return []
    out = []
    for r in _repos():
        if r.get("kind") != "base":
            continue
        for fname in _repo_missing(r):
            out.append(f"{r['local_dir']}/{fname}")
    return out


def q8_missing_files() -> list[str]:
    """Files missing for the Q8 repo (bare filenames). Honors $LTX_Q8_LOCAL.

    When LTX_Q8_LOCAL is set, it overrides the Q8 repo's local_dir — we
    re-route the check there. Other overrides (e.g. moving Q4 elsewhere)
    aren't supported via env var; users with custom layouts should adjust
    required_files.json directly."""
    for r in _repos():
        if r.get("key") == "q8":
            override = r.copy()
            override["local_dir"] = str(Q8_LOCAL_PATH.relative_to(ROOT)) \
                if Q8_LOCAL_PATH.is_relative_to(ROOT) else str(Q8_LOCAL_PATH)
            # _repo_missing uses ROOT-relative; if Q8_LOCAL_PATH is absolute
            # outside ROOT, build a temporary repo with absolute base.
            if Path(override["local_dir"]).is_absolute():
                missing = []
                for fname in r.get("files", []):
                    p = Path(override["local_dir"]) / fname
                    try:
                        if not p.exists() or p.stat().st_size < _MIN_FILE_BYTES:
                            missing.append(fname)
                    except OSError:
                        missing.append(fname)
                return missing
            return _repo_missing(override)
    return []


def repo_status_list() -> list[dict]:
    """Per-repo status snapshot for the /models endpoint and the UI panel.

    Resolution order per repo:
      1. Check the canonical local_dir (mlx_models/<repo>/...) — what
         Pinokio installs always populate.
      2. If incomplete there, check HF cache (~/.cache/huggingface/...) —
         what manual installs / dev environments use.
      3. Whichever is more complete wins; the `where` field tells the UI
         which storage layer the files are in (so we can show "Cached"
         instead of "Installed" for cache-backed installs).

    This is the fix for the previously-reported false "MISSING" against
    HF-cache installs — the panel now sees them and reports honestly."""
    out = []
    for r in _repos():
        local_dir = r["local_dir"]
        if r.get("key") == "q8":
            local_dir = str(Q8_LOCAL_PATH)

        local_missing = q8_missing_files() if r.get("key") == "q8" else _repo_missing(r)
        total = len(r.get("files", []))
        local_present = total - len(local_missing)

        cache_missing = _repo_missing_in_cache(r)
        cache_present = (total - len(cache_missing)) if cache_missing is not None else 0

        # Pick the layer with more files present — ties go to local_dir
        # because that's what Pinokio installs use and what the menu
        # consults.
        if cache_missing is not None and cache_present > local_present:
            where = "hf_cache"
            missing = cache_missing
            present = cache_present
            location = str(_repo_hf_cache_dir(r["repo_id"]) or _hf_cache_root())
        else:
            where = "local_dir"
            missing = local_missing
            present = local_present
            location = local_dir

        out.append({
            "key": r["key"],
            "kind": r.get("kind", "base"),
            "name": r["name"],
            "blurb": r.get("blurb", ""),
            "repo_id": r["repo_id"],
            "local_dir": local_dir,
            "where": where,                 # 'local_dir' | 'hf_cache'
            "location": location,           # actual on-disk path the files are in
            "size_gb": r.get("size_gb"),
            "total_files": total,
            "present_files": present,
            "missing_files": missing,
            "complete": not missing,
        })
    return out


# ---- HF download (in-panel model fetcher) ------------------------------------
# `hf` is the v1+ huggingface_hub CLI. We resolve it from (in order):
#   1. $LTX_HF env var
#   2. ltx-2-mlx/.venv/bin/hf       (manual install)
#   3. ltx-2-mlx/env/bin/hf         (Pinokio)
#   4. shutil.which("hf")           (system PATH fallback)
def _resolve_hf() -> Path | None:
    candidates = [
        os.environ.get("LTX_HF"),
        str(MLX / ".venv/bin/hf"),
        str(MLX / "env/bin/hf"),
        shutil.which("hf"),
    ]
    for c in candidates:
        if c and Path(c).exists():
            return Path(c)
    return None


HF_BIN = _resolve_hf()

# Single global download slot — concurrent hf downloads compete for bandwidth
# anyway, and serializing them keeps the log readable. State protected by
# DOWNLOAD_LOCK; UI polls via /status to render progress.
DOWNLOAD_LOCK = threading.Lock()
DOWNLOAD: dict = {
    "active": False,
    "key": None,           # which repo (q4/gemma/q8)
    "repo_id": None,
    "started_ts": None,
    "last_line": "",       # most recent hf output line for UI display
    "proc": None,
    "pgid": None,
}


def _download_thread(repo: dict) -> None:
    """Run `hf download <repo_id> --local-dir <repo.local_dir>` and stream
    stdout/stderr line-by-line into STATE['log']. Sets DOWNLOAD["active"]
    back to False on exit (success or fail). Files land at repo['local_dir']
    relative to ROOT, which is exactly where the completeness checks look —
    so the moment hf finishes, /status flips to complete + the UI updates."""
    repo_id = repo["repo_id"]
    target = ROOT / repo["local_dir"]
    target.mkdir(parents=True, exist_ok=True)
    cmd = [str(HF_BIN), "download", repo_id, "--local-dir", str(target)]
    push(f"[hf] {repo_id} → {target} (~{repo.get('size_gb','?')} GB) — resumable")
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        with DOWNLOAD_LOCK:
            DOWNLOAD["proc"] = proc
            try:
                DOWNLOAD["pgid"] = os.getpgid(proc.pid)
            except ProcessLookupError:
                DOWNLOAD["pgid"] = None
        # Stream every line. hf emits a tqdm progress bar with carriage
        # returns; we split on \r as well as \n so progress updates show
        # up live in the panel log instead of being buffered until done.
        buf = ""
        while True:
            ch = proc.stdout.read(1)
            if not ch:
                break
            if ch in ("\n", "\r"):
                line = buf.strip()
                buf = ""
                if line:
                    with DOWNLOAD_LOCK:
                        DOWNLOAD["last_line"] = line[:200]
                    push(f"[hf:{repo['key']}] {line[:300]}")
            else:
                buf += ch
        if buf.strip():
            push(f"[hf:{repo['key']}] {buf.strip()[:300]}")
        rc = proc.wait()
        if rc == 0:
            push(f"[hf] {repo_id} downloaded successfully.")
        else:
            push(f"[hf] {repo_id} FAILED — exit {rc}. Click Download again to retry/resume.")
    except Exception as exc:
        push(f"[hf] {repo_id} crashed: {exc}")
    finally:
        with DOWNLOAD_LOCK:
            DOWNLOAD["active"] = False
            DOWNLOAD["key"] = None
            DOWNLOAD["repo_id"] = None
            DOWNLOAD["started_ts"] = None
            DOWNLOAD["last_line"] = ""
            DOWNLOAD["proc"] = None
            DOWNLOAD["pgid"] = None


def _kill_active_download() -> None:
    """Best-effort kill the running hf process group. Called by atexit and
    by the /models/cancel endpoint."""
    with DOWNLOAD_LOCK:
        pgid = DOWNLOAD.get("pgid")
        proc = DOWNLOAD.get("proc")
    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    elif proc is not None:
        try:
            proc.terminate()
        except Exception:
            pass


atexit.register(_kill_active_download)


# ---- CivitAI bridge ---------------------------------------------------------
# CivitAI is the de-facto LoRA distribution hub. We hit its public REST API
# (no auth required for SFW; an optional CIVITAI_API_KEY env var unlocks
# higher rate limits + NSFW listings tied to an account) and download via
# urllib so we don't need a third-party HTTP dependency.
#
# Filtering: baseModels=LTXV%202.3 + types=LORA narrows to LTX-2.3 LoRAs.
# CivitAI's response shape is documented at
# https://github.com/civitai/civitai/wiki/REST-API-Reference but we trim
# the response server-side so the panel only ships what the UI needs.

CIVITAI_API_BASE = "https://civitai.com/api/v1"
CIVITAI_USER_AGENT = "phosphene/1.0 (https://github.com/mrbizarro/phosphene)"


def _civitai_request(path: str, params: dict | None = None,
                     timeout: float = 20.0) -> dict:
    """Minimal stdlib HTTP client for the CivitAI API. No third-party
    deps so we don't pin requests / httpx into the install. Returns the
    decoded JSON body or raises RuntimeError on non-2xx."""
    import urllib.parse, urllib.request
    url = f"{CIVITAI_API_BASE}{path}"
    if params:
        # urlencode handles the URL-quoting (spaces → %20 etc.) so
        # baseModels=LTXV 2.3 makes it through unmangled.
        url = url + "?" + urllib.parse.urlencode(params, doseq=True)
    req = urllib.request.Request(url, headers={
        "Accept": "application/json",
        "User-Agent": CIVITAI_USER_AGENT,
    })
    api_key = _active_civitai_key()
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "replace")
            return json.loads(body)
    except Exception as exc:
        raise RuntimeError(f"{type(exc).__name__}: {exc}") from exc


def _civitai_search(query: str = "", nsfw: bool = False,
                    cursor: str = "", limit: int = 20) -> dict:
    """Search LTX-2.3 LoRAs. Returns the trimmed shape:
        { "items": [{
            "id", "name", "creator", "description", "tags", "downloads",
            "rating", "nsfw", "preview_url", "version_id",
            "download_url", "trigger_words", "recommended_strength",
            "size_kb", "base_model"
          }, ...],
          "total_pages", "current_page" }
    Each item carries everything the front end needs to render a card
    AND everything the download endpoint needs to write a sidecar — no
    second round-trip required to install."""
    # CivitAI's /models endpoint uses cursor-style pagination
    # (`nextCursor` in the response → pass back as `cursor` on the next
    # request). Page numbers are deprecated for this endpoint. We expose
    # the same cursor flow to the panel UI so it can render a "Load more"
    # affordance without faking pagination on top of cursors.
    params: dict[str, object] = {
        "types": "LORA",
        "baseModels": "LTXV 2.3",
        "limit": limit,
        "sort": "Most Downloaded",
        # nsfw=True lets the API include NSFW results; the user's CivitAI
        # account-level filter (if any) is unaffected. We surface the
        # boolean on each item so the front end can render a warning.
        "nsfw": "true" if nsfw else "false",
    }
    if query.strip():
        params["query"] = query.strip()
    if cursor:
        params["cursor"] = cursor
    raw = _civitai_request("/models", params=params)
    items = []
    for m in (raw.get("items") or []):
        versions = m.get("modelVersions") or []
        if not versions:
            continue
        v = versions[0]
        files = v.get("files") or []
        # Pick the first .safetensors file. CivitAI sometimes hosts
        # multiple file variants (pruned, full, etc.); prefer the one
        # marked `primary: true` if present, otherwise fall back to the
        # first .safetensors.
        primary = next((f for f in files if f.get("primary") and
                        f.get("name", "").lower().endswith(".safetensors")),
                       None) or next((f for f in files if
                       f.get("name", "").lower().endswith(".safetensors")),
                       None)
        if not primary:
            continue
        images = v.get("images") or []
        # Prefer the first image-type preview if any exist (some LTX LoRAs
        # ship both video MP4s AND still images). Fall back to the first
        # entry of any type. We pass the type back to the client so the
        # renderer can pick <img> vs <video> appropriately — for LTX
        # video LoRAs the previews are usually MP4s, which would just
        # 404 inside an <img> tag.
        preview_obj = (
            next((img for img in images if img.get("type") == "image" and img.get("url")), None)
            or next((img for img in images if img.get("url")), None)
        )
        preview = preview_obj.get("url") if preview_obj else None
        preview_type = preview_obj.get("type") if preview_obj else None
        creator = (m.get("creator") or {}).get("username") or "unknown"
        # CivitAI puts trigger words on the version (`trainedWords`).
        trigger = list(v.get("trainedWords") or [])
        items.append({
            "id": m.get("id"),
            "version_id": v.get("id"),
            "name": m.get("name") or f"model-{m.get('id')}",
            "creator": creator,
            "description": (m.get("description") or "")[:600],
            "tags": list(m.get("tags") or []),
            "downloads": (m.get("stats") or {}).get("downloadCount"),
            "rating": (m.get("stats") or {}).get("rating"),
            "nsfw": bool(m.get("nsfw")),
            "preview_url": preview,
            "preview_type": preview_type,        # "image" | "video" | None
            "filename": primary.get("name"),
            "size_kb": primary.get("sizeKB"),
            "download_url": primary.get("downloadUrl"),
            "trigger_words": trigger,
            "recommended_strength": 1.0,    # CivitAI doesn't expose this; default
            "base_model": v.get("baseModel"),
            "civitai_url": f"https://civitai.com/models/{m.get('id')}",
        })
    metadata = raw.get("metadata") or {}
    return {
        "items": items,
        "next_cursor": metadata.get("nextCursor"),
        "has_more": bool(metadata.get("nextCursor")),
    }


def _civitai_download(download_url: str, meta: dict) -> dict:
    """Download a CivitAI .safetensors into mlx_models/loras/ and write
    a sidecar JSON. Returns { name, path, sidecar_path, size_bytes }.
    Streams to a .partial file then renames atomically — a Pinokio kill
    mid-download leaves nothing the next scan would mistake for a
    complete LoRA."""
    import urllib.parse, urllib.request
    if not download_url:
        raise RuntimeError("download_url required")
    # Restrict to civitai.com to prevent the endpoint being used as a
    # generic HTTP fetcher. Subdomains (e.g. images.civitai.com) are
    # also OK because CivitAI sometimes redirects there.
    parsed = urllib.parse.urlparse(download_url)
    if parsed.scheme != "https" or not parsed.netloc.endswith("civitai.com"):
        raise RuntimeError(f"refusing to download from {parsed.netloc}; "
                           f"only civitai.com is allowed")
    # CivitAI's download endpoint started requiring auth for many models
    # in 2025+. The Authorization header sometimes works, but the
    # canonical path is `?token=<key>` on the URL — and many CDN hops
    # only honour the token in the URL. Try the URL form first when a
    # key is available.
    api_key_env = _active_civitai_key()
    if api_key_env and "token=" not in (parsed.query or ""):
        sep = "&" if parsed.query else "?"
        download_url = download_url + sep + "token=" + urllib.parse.quote(api_key_env)
    loras_dir = _safe_loras_dir()
    # Filename from meta (preferred — preserves CivitAI's name) or fall
    # back to the URL's last path segment.
    fname = (meta.get("filename") or
             os.path.basename(parsed.path) or
             f"civitai_{meta.get('id', 'unknown')}.safetensors")
    if not fname.lower().endswith(".safetensors"):
        fname += ".safetensors"
    # Sanitize aggressively — CivitAI names can contain spaces and
    # weirder characters; the helper later passes this filename through
    # the safetensors loader which is filesystem-only and fine with
    # spaces, but the picker UI is calmer with snake_case.
    safe_fname = re.sub(r"[^A-Za-z0-9._-]+", "_", fname).strip("_")
    target = loras_dir / safe_fname
    if target.exists():
        # Don't silently re-download. Surface the conflict so the user
        # can decide (the UI will offer a choice; for now just skip).
        return {
            "name": meta.get("name") or target.stem,
            "path": str(target),
            "size_bytes": target.stat().st_size,
            "skipped": True,
            "reason": "already exists",
        }
    tmp = target.with_suffix(target.suffix + ".partial")
    push(f"[civitai] downloading {meta.get('name') or safe_fname}")
    api_key = _active_civitai_key()
    headers = {"User-Agent": CIVITAI_USER_AGENT}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(download_url, headers=headers)
    bytes_written = 0
    last_log = 0.0
    try:
        try:
            resp_ctx = urllib.request.urlopen(req, timeout=60)
        except urllib.request.HTTPError as he:
            # Surface the 401 case with a clear remediation. CivitAI now
            # requires API tokens for most LoRA downloads, even SFW ones,
            # but the API key for the search endpoint is optional — so a
            # user can browse without setting up auth and only hit this
            # wall on first install.
            if he.code == 401:
                raise RuntimeError(
                    "CivitAI returned 401 Unauthorized. CivitAI requires an "
                    "API token for LoRA downloads. Get one at "
                    "https://civitai.com/user/account, then set the "
                    "CIVITAI_API_KEY environment variable before launching "
                    "the panel (or in Pinokio's Phosphene start script). "
                    "Restart Pinokio's Phosphene to pick it up."
                )
            raise
        with resp_ctx as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            with tmp.open("wb") as fh:
                while True:
                    chunk = resp.read(1024 * 256)
                    if not chunk:
                        break
                    fh.write(chunk)
                    bytes_written += len(chunk)
                    now = time.time()
                    if now - last_log > 1.0:
                        if total:
                            pct = int(100 * bytes_written / total)
                            push(f"[civitai] {pct}% · "
                                 f"{bytes_written // (1024*1024)}/"
                                 f"{total // (1024*1024)} MB")
                        else:
                            push(f"[civitai] {bytes_written // (1024*1024)} MB")
                        last_log = now
        os.replace(tmp, target)
    except Exception:
        try: tmp.unlink()
        except OSError: pass
        raise
    push(f"[civitai] saved {target.name} ({bytes_written // (1024*1024)} MB)")
    # Write the sidecar — kept tolerant of meta gaps so a partial
    # payload still produces a usable picker entry.
    sidecar = target.with_suffix(".json")
    sidecar_data = {
        "name": meta.get("name") or target.stem,
        "description": meta.get("description") or "",
        "trigger_words": list(meta.get("trigger_words") or []),
        "recommended_strength": float(meta.get("recommended_strength") or 1.0),
        "preview_url": meta.get("preview_url"),
        "base_model": meta.get("base_model") or "LTXV 2.3",
        "civitai_id": meta.get("id"),
        "civitai_version_id": meta.get("version_id"),
        "civitai_url": meta.get("civitai_url"),
        "downloaded_at": iso_now(),
        "downloaded_size_bytes": bytes_written,
    }
    try:
        with sidecar.open("w") as fh:
            json.dump(sidecar_data, fh, indent=2)
    except Exception as exc:
        push(f"[civitai] WARN: could not write sidecar ({exc})")
    return {
        "name": sidecar_data["name"],
        "path": str(target),
        "sidecar_path": str(sidecar),
        "size_bytes": bytes_written,
        "skipped": False,
    }


# Aspect presets — width × height (model-safe, multiples of 32)
ASPECTS = {
    "landscape": {"label": "Landscape 16:9 (1280×704 → 720)", "w": 1280, "h": 704},
    "vertical":  {"label": "Vertical 9:16 (704×1280 → 720)",  "w": 704,  "h": 1280},
    "square":    {"label": "Square (768×768)",                 "w": 768,  "h": 768},
    "test":      {"label": "Quick test (512×288)",             "w": 512,  "h": 288},
    "wide":      {"label": "Ultra-wide 21:9 (1408×608)",       "w": 1408, "h": 608},
    "portrait":  {"label": "Mobile portrait (576×1024)",       "w": 576,  "h": 1024},
}

# One-click presets (fill aspect + duration in one button)
PRESETS = [
    {"key": "test_1s",    "label": "Quick test",    "sub": "512×288 · 1s",   "aspect": "test",      "dur": 1,  "steps": 8, "stop_comfy": True},
    {"key": "land_5s",    "label": "Landscape 5s",  "sub": "1280×704 · 5s",  "aspect": "landscape", "dur": 5,  "steps": 8, "stop_comfy": True},
    {"key": "land_10s",   "label": "Landscape 10s", "sub": "1280×704 · 10s", "aspect": "landscape", "dur": 10, "steps": 8, "stop_comfy": True},
    {"key": "vert_5s",    "label": "Vertical 5s",   "sub": "704×1280 · 5s",  "aspect": "vertical",  "dur": 5,  "steps": 8, "stop_comfy": True},
    {"key": "vert_10s",   "label": "Vertical 10s",  "sub": "704×1280 · 10s", "aspect": "vertical",  "dur": 10, "steps": 8, "stop_comfy": True},
    {"key": "square_5s",  "label": "Square 5s",     "sub": "768×768 · 5s",   "aspect": "square",    "dur": 5,  "steps": 8, "stop_comfy": True},
]


# ---- shared state ------------------------------------------------------------

STATE: dict = {
    "queue": [], "current": None, "history": [],
    "paused": False, "log": [],
    "running": False, "pid": None, "pgid": None,
}
LOCK = threading.RLock()
QUEUE_COND = threading.Condition(LOCK)

CAFFEINATE_PROC: subprocess.Popen | None = None
HIDDEN_PATHS: set[str] = set()


def push(line: str) -> None:
    line = line.rstrip()
    if not line:
        return
    stamped = f"[{time.strftime('%H:%M:%S')}] {line}"
    with LOCK:
        STATE["log"].append(stamped)
        STATE["log"] = STATE["log"][-LOG_LIMIT:]


def iso_now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def atomic_write_text(path: Path, text: str) -> None:
    """Write text to `path` via temp file + fsync + os.replace.

    Plain Path.write_text() can leave a half-written file if macOS sleeps,
    runs out of disk, or the panel crashes mid-write — corrupted queue or
    sidecar files would lose the user's work-in-progress. Atomic replace
    guarantees the file is either pre-write or fully post-write, never torn.
    """
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ---- caffeinate --------------------------------------------------------------

def caffeinate_on() -> None:
    global CAFFEINATE_PROC
    if CAFFEINATE_PROC and CAFFEINATE_PROC.poll() is None:
        return
    try:
        CAFFEINATE_PROC = subprocess.Popen(["caffeinate", "-i"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        push("caffeinate active — Mac won't idle-sleep while queue is running")
    except Exception as exc:
        push(f"caffeinate failed: {exc}")


def caffeinate_off() -> None:
    global CAFFEINATE_PROC
    if CAFFEINATE_PROC and CAFFEINATE_PROC.poll() is None:
        try:
            CAFFEINATE_PROC.terminate()
        except Exception:
            pass
    CAFFEINATE_PROC = None


# ---- system probes -----------------------------------------------------------

def find_comfy_pids() -> list[int]:
    try:
        out = subprocess.run(["pgrep", "-f", COMFY_PATTERN],
            capture_output=True, text=True, timeout=2).stdout
    except Exception:
        return []
    return [int(line) for line in out.splitlines() if line.strip().isdigit()]


def kill_comfy() -> int:
    pids = find_comfy_pids()
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
            push(f"Sent SIGTERM to Comfy PID {pid}")
        except ProcessLookupError:
            pass
        except PermissionError as exc:
            push(f"Could not signal {pid}: {exc}")
    if not pids:
        push("Comfy was not running.")
    return len(pids)


def open_pinokio() -> None:
    try:
        subprocess.run(["open", "-a", "Pinokio"], check=False, timeout=5)
        push("Opened Pinokio — start Comfy from the dashboard if you need it.")
    except Exception as exc:
        push(f"Could not open Pinokio: {exc}")


# ---- hardware tier + capability matrix ---------------------------------------
# Detected once at startup (RAM doesn't change at runtime). The /status
# endpoint exposes this so the UI can grey out features the box can't run
# instead of letting the user submit a job that's going to swap thrash for
# 2 hours and then fail.
#
# Empirically-derived limits from our own testing on a 64 GB M4 Studio:
#
#   T2V/I2V Q4 standard at 1280×704: ~38 GB peak, fine on 32 GB w/ a bit of swap
#   T2V/I2V Q4 standard at 768×432:  ~22 GB peak, fine anywhere
#   T2V/I2V Q8 HQ        at 1280×704: ~50 GB peak, fits 64 GB cleanly
#   FFLF (Q8 dev)        at 1280×704: > 64 GB peak, OOMs into swap
#   FFLF (Q8 dev)        at  768×416: ~48 GB peak, fits 64 GB
#   Extend (Q4 dev)      at 1280×704: ~50 GB peak + 12 GB swap (240s/step)
#   Extend (Q4 dev)      at  768×416: ~46 GB peak, no swap (54s/step)
#
# `t2v_max_dim` / `i2v_max_dim` clamp T2V/I2V resolution. `keyframe_max_dim`
# and `extend_max_dim` clamp those modes' resolutions specifically (since
# they use the dev transformer + guided denoise which is heavier). 0 means
# no clamp. `allows_q8` gates HQ + FFLF entirely.
# Time estimates are for a typical 5-second render (121 frames @ 24fps)
# on the hardware that's most common at each tier:
#   compact     M2 / M-Pro base — slowest of the bunch
#   comfortable M4 base Mac Studio (the rig this panel was built on)
#   roomy       M-Max
#   studio      M-Ultra (~3x the comfortable tier)
# These are *estimates* shown in the UI to help users decide whether to
# wait — wall-clock varies with prompt length, current memory pressure,
# and what else is running. We label them with "~" so users don't expect
# stopwatch precision.
CAPABILITIES: dict[str, dict] = {
    "base": {
        # < 48 GB. M2 8/16/24 GB, M-Pro 18/36 GB, base M-Max 36 GB.
        # Q8 won't fit; even Q4 at 720p is borderline.
        "label": "Compact",
        "ram_label": "Under 48 GB",
        "tagline": "Q4 base model · small renders",
        "t2v_max_dim": 768,
        "i2v_max_dim": 768,
        "keyframe_max_dim": 0,    # disabled
        "extend_max_dim": 0,      # disabled
        "allows_q8": False,
        "allows_keyframe": False,
        "allows_extend": False,
        "blurb": (
            "This Mac has under 48 GB of unified memory. The basic "
            "modes work — text-to-video and image-to-video — but only "
            "at smaller sizes (up to 768 pixels on the longer side). "
            "The bigger modes (High quality, first-last-frame, extend "
            "an existing clip) need more memory than this Mac has, so "
            "they're turned off."
        ),
        # Per-mode time estimates for a typical 5s render (121 frames @ 24fps).
        # Times are wall-clock on the typical hardware at this tier.
        "times": {
            "t2v_draft":     "about 2 min",
            "t2v_standard":  "about 5 min",
            "i2v_standard":  "about 5 min",
            "high":          None,  # disabled
            "keyframe":      None,  # disabled
            "extend":        None,  # disabled
        },
    },
    "standard": {
        # 48–79 GB. 64 GB M-Studio is the canonical hardware here.
        "label": "Comfortable",
        "ram_label": "48–79 GB",
        "tagline": "Every mode works · larger modes capped at 768 px",
        "t2v_max_dim": 0,         # no clamp
        "i2v_max_dim": 0,
        "keyframe_max_dim": 768,
        "extend_max_dim": 768,
        "allows_q8": True,
        "allows_keyframe": True,
        "allows_extend": True,
        "blurb": (
            "This is the 64 GB tier — the panel was built and tuned on "
            "exactly this hardware. Every mode works. Text-to-video and "
            "image-to-video run at the full 1280×704. The two biggest "
            "modes (first-last-frame interpolation and extending an "
            "existing clip) cap their video size at 768 pixels on the "
            "longer side because the bigger model behind them runs out "
            "of memory above that. 768 is a sweet-spot working size; "
            "you can run a separate upscaler afterwards if you need 720p+."
        ),
        "times": {
            "t2v_draft":     "about 2 min",
            "t2v_standard":  "about 7 min",
            "i2v_standard":  "about 7 min",
            "high":          "about 12 min",
            "keyframe":      "about 5 min (at 768 px)",
            "extend":        "about 11 min (at 768 px)",
        },
    },
    "high": {
        # 80–119 GB.
        "label": "Roomy",
        "ram_label": "80–119 GB",
        "tagline": "Most modes at full size · larger modes up to 1024 px",
        "t2v_max_dim": 0,
        "i2v_max_dim": 0,
        "keyframe_max_dim": 1024,
        "extend_max_dim": 1024,
        "allows_q8": True,
        "allows_keyframe": True,
        "allows_extend": True,
        "blurb": (
            "This Mac has 96 GB-class memory. Text-to-video, "
            "image-to-video, and high-quality renders all run at the "
            "full 1280×704. First-last-frame and extend can go up to "
            "1024 pixels (a real bump in detail over the 768 cap) "
            "without falling into swap."
        ),
        "times": {
            "t2v_draft":     "about 1 min",
            "t2v_standard":  "about 4 min",
            "i2v_standard":  "about 4 min",
            "high":          "about 7 min",
            "keyframe":      "about 6 min (at 1024 px)",
            "extend":        "about 9 min (at 1024 px)",
        },
    },
    "pro": {
        # 128+ GB. M-Ultra Mac Studio 192/256 GB.
        "label": "Studio",
        "ram_label": "120 GB or more",
        "tagline": "No size limits anywhere",
        "t2v_max_dim": 0,
        "i2v_max_dim": 0,
        "keyframe_max_dim": 0,
        "extend_max_dim": 0,
        "allows_q8": True,
        "allows_keyframe": True,
        "allows_extend": True,
        "blurb": (
            "This Mac has 120 GB or more of unified memory. There are "
            "no size limits on any mode — render at whatever resolution "
            "and length the model supports. Bigger renders take longer, "
            "but nothing's capped artificially."
        ),
        "times": {
            "t2v_draft":     "under a minute",
            "t2v_standard":  "about 2 min",
            "i2v_standard":  "about 2 min",
            "high":          "about 4 min",
            "keyframe":      "about 3 min (full size)",
            "extend":        "about 5 min (full size)",
        },
    },
}


def _detect_tier() -> str:
    """Pick a tier based on `hw.memsize`. Cached at module load — RAM is
    fixed at boot. LTX_TIER_OVERRIDE env var lets advanced users force a
    tier (useful for demos / reproducing other-machine bugs)."""
    override = os.environ.get("LTX_TIER_OVERRIDE", "").strip().lower()
    if override in CAPABILITIES:
        return override
    try:
        out = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=1,
        ).stdout.strip()
        gb = int(out) / 1024**3
    except Exception:
        return "standard"   # safe default
    if gb < 48:  return "base"
    if gb < 80:  return "standard"
    if gb < 120: return "high"
    return "pro"


SYSTEM_TIER = _detect_tier()
SYSTEM_CAPS = CAPABILITIES[SYSTEM_TIER]


def tier_max_dim(kind: str) -> int:
    """Return the max-dim clamp for a given pipeline kind on this tier.
    0 means "no clamp". Caller decides how to interpret 0 (skip downscale,
    pass user's W/H through, etc.)."""
    return int(SYSTEM_CAPS.get(f"{kind}_max_dim", 0))


def get_memory() -> dict:
    info = {"total_gb": 0.0, "used_gb": 0.0, "pressure_pct": 0, "swap_gb": 0.0}
    try:
        total = int(subprocess.run(["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=1).stdout.strip())
        info["total_gb"] = total / 1024**3
        vm = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=1).stdout
        m = re.search(r"page size of (\d+)", vm)
        page_size = int(m.group(1)) if m else 16384

        def pages(name: str) -> int:
            mm = re.search(rf"{re.escape(name)}:\s+(\d+)", vm)
            return int(mm.group(1)) if mm else 0

        used_bytes = (pages("Pages active") + pages("Pages wired down")
                      + pages("Pages occupied by compressor")) * page_size
        info["used_gb"] = used_bytes / 1024**3
        info["pressure_pct"] = round(used_bytes / total * 100) if total else 0

        swap = subprocess.run(["sysctl", "-n", "vm.swapusage"],
            capture_output=True, text=True, timeout=1).stdout
        m = re.search(r"used\s*=\s*([\d.]+)([KMG])", swap)
        if m:
            v = float(m.group(1))
            mult = {"K": 1 / 1024 / 1024, "M": 1 / 1024, "G": 1.0}[m.group(2)]
            info["swap_gb"] = v * mult
    except Exception:
        pass
    return info


# ---- hidden / output state ---------------------------------------------------

def load_hidden() -> None:
    global HIDDEN_PATHS
    if not HIDDEN_FILE.exists():
        return
    try:
        HIDDEN_PATHS = set(json.loads(HIDDEN_FILE.read_text()))
    except Exception as exc:
        print(f"hidden load failed: {exc}", flush=True)


def persist_hidden() -> None:
    try:
        with LOCK:
            snapshot = sorted(HIDDEN_PATHS)
        atomic_write_text(HIDDEN_FILE, json.dumps(snapshot, indent=2))
    except Exception as exc:
        push(f"hidden persist failed: {exc}")


def set_hidden(path: str, hidden: bool) -> None:
    with LOCK:
        if hidden:
            HIDDEN_PATHS.add(path)
        else:
            HIDDEN_PATHS.discard(path)
    persist_hidden()


def _probe_video_dims(path: str) -> tuple[int, int]:
    """Return (width, height) of a video via ffprobe. Returns (0, 0) on
    any failure — caller treats that as "unknown, skip resolution check"."""
    try:
        out = subprocess.run(
            [str(FFPROBE), "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", path],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        if "x" in out:
            w, h = out.split("x", 1)
            return int(w), int(h)
    except Exception:
        pass
    return 0, 0


def _ensure_downscaled(src: Path, max_dim: int = 768, align: int = 32) -> Path:
    """If `src` has its longer side > max_dim, write a downscaled lossless
    copy alongside it and return that. Cached on disk by target dimensions
    so repeated extends on the same source don't re-encode every time.

    Used by Extend: the upstream pipeline runs the dev transformer at the
    source's native resolution. On 64 GB Macs, 1280×704 + dev transformer
    + CFG-style guided denoising peaks past ~50 GB resident and pushes
    8-12 GB into swap, making each step take 4 minutes instead of 25
    seconds. Pre-downscaling to ≤768 max-side fits cleanly in RAM."""
    src_w, src_h = _probe_video_dims(str(src))
    if not src_w or not src_h or max(src_w, src_h) <= max_dim:
        return src
    scale = max_dim / max(src_w, src_h)
    new_w = max(align, (int(round(src_w * scale)) // align) * align)
    new_h = max(align, (int(round(src_h * scale)) // align) * align)
    cached = src.parent / f"{src.stem}_dn{new_w}x{new_h}.mp4"
    if cached.exists() and cached.stat().st_size > 1024:
        return cached
    cmd = [str(FFMPEG), "-y", "-i", str(src),
           "-vf", f"scale={new_w}:{new_h}",
           "-c:v", "libx264", "-pix_fmt", "yuv444p", "-crf", "0", "-preset", "veryfast",
           "-c:a", "copy",   # don't re-encode audio — extend doesn't need it transformed
           str(cached)]
    subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    return cached


def list_uploads(limit: int = 40) -> list[dict]:
    """Return recent panel_uploads/* images sorted by mtime descending.

    Powers the "Recent uploads — click to use" strip in the FFLF / I2V
    pickers. Image-only filter so user-dropped junk doesn't pollute the
    strip; we trust filename extensions because uploads are local-only
    and the worst outcome of a misnamed file is a broken thumbnail."""
    if not UPLOADS.exists():
        return []
    exts = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
    files = []
    for p in UPLOADS.iterdir():
        try:
            if not p.is_file() or p.suffix.lower() not in exts:
                continue
            files.append((p, p.stat().st_mtime, p.stat().st_size))
        except OSError:
            continue
    files.sort(key=lambda t: t[1], reverse=True)
    out = []
    for p, mtime, size in files[:limit]:
        # Strip the millisecond-prefix the upload handler adds, so the
        # display name is what the user actually picked. Keep the full
        # path as the value so the form submits the unique copy.
        display = p.name
        m = re.match(r"^\d+_(.*)$", display)
        if m: display = m.group(1)
        out.append({
            "name": display,
            "path": str(p),
            "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime)),
            "size_kb": int(size / 1024),
            "url": f"/image?path={quote(str(p))}",
        })
    return out


def list_outputs(include_hidden: bool = False) -> list[dict]:
    files = sorted(OUTPUT.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)[:120]
    # Snapshot HIDDEN_PATHS under LOCK so concurrent hide/unhide from other
    # threads can't tear the read.
    with LOCK:
        hidden_snap = set(HIDDEN_PATHS)
    out = []
    for p in files:
        path_s = str(p)
        is_hidden = path_s in hidden_snap
        if is_hidden and not include_hidden:
            continue
        out.append({
            "name": p.name,
            "path": path_s,
            "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime)),
            "size_mb": p.stat().st_size / 1024 / 1024,
            "url": f"/file?path={quote(path_s)}",
            "has_sidecar": p.with_suffix(p.suffix + ".json").exists(),
            "hidden": is_hidden,
        })
        if len(out) >= 60:
            break
    return out


def write_sidecar(path: Path, payload: dict) -> None:
    try:
        atomic_write_text(path, json.dumps(payload, indent=2))
    except Exception as exc:
        push(f"Sidecar write failed: {exc}")


# ---- queue persistence -------------------------------------------------------

def persist_queue() -> None:
    with LOCK:
        snapshot = {
            "queue": [_strip_for_disk(j) for j in STATE["queue"]],
            "current": _strip_for_disk(STATE["current"]) if STATE["current"] else None,
            "history": [_strip_for_disk(j) for j in STATE["history"][:HISTORY_PERSIST_LIMIT]],
            "paused": STATE["paused"],
        }
    try:
        atomic_write_text(QUEUE_FILE, json.dumps(snapshot, indent=2))
    except Exception as exc:
        push(f"Queue persist failed: {exc}")


def _strip_for_disk(job: dict) -> dict:
    if not job:
        return job
    return {k: v for k, v in job.items() if k not in ("started_ts",)}


def load_queue() -> None:
    if not QUEUE_FILE.exists():
        return
    try:
        data = json.loads(QUEUE_FILE.read_text())
    except Exception as exc:
        print(f"queue load failed: {exc}", flush=True)
        return
    with LOCK:
        STATE["queue"] = data.get("queue", []) or []
        STATE["history"] = data.get("history", []) or []
        STATE["paused"] = bool(data.get("paused", False))
        if data.get("current"):
            stale = data["current"]
            stale["status"] = "queued"
            stale["started_at"] = None
            stale["error"] = None
            STATE["queue"].insert(0, stale)


# ---- warm helper -------------------------------------------------------------

class WarmHelper:
    def __init__(self):
        self.proc: subprocess.Popen | None = None
        self.lock = threading.Lock()

    def _ensure(self) -> None:
        with self.lock:
            if self.proc is not None and self.proc.poll() is None:
                return
            env = os.environ.copy()
            env["PATH"] = f"{FFMPEG_BIN}:{env.get('PATH', '')}"
            env["LTX_MODEL"] = MODEL_ID
            env["LTX_GEMMA"] = str(GEMMA)
            env["LTX_IDLE_TIMEOUT"] = str(HELPER_IDLE_TIMEOUT)
            env["LTX_LOW_MEMORY"] = HELPER_LOW_MEMORY
            # Output codec env vars sourced from panel settings. The patched
            # ffmpeg call inside ltx_core_mlx reads these at job time, so
            # they need to be in the helper's env at spawn. When the user
            # changes settings via /settings POST we kill the helper; the
            # next job spawns it fresh and picks up the new values here.
            _s = get_settings()
            env["LTX_OUTPUT_PIX_FMT"] = _s.get("output_pix_fmt", "yuv420p")
            env["LTX_OUTPUT_CRF"] = _s.get("output_crf", "18")
            # Tokens — source-of-truth from settings (with env-var
            # fallback for power users). These end up driving:
            #   HF_TOKEN  → huggingface_hub.snapshot_download() picks it
            #     up automatically when the helper resolves a gated LoRA
            #     (Lightricks HDR, etc.).
            #   CIVITAI_API_KEY → not used by the helper currently
            #     (CivitAI downloads happen panel-side), but threaded
            #     through anyway so future helper-side CivitAI code
            #     inherits without ceremony.
            _hf = _active_hf_token()
            if _hf:
                env["HF_TOKEN"] = _hf
                # huggingface_hub also reads HUGGING_FACE_HUB_TOKEN —
                # set both so older lib versions work too.
                env["HUGGING_FACE_HUB_TOKEN"] = _hf
            _civ = _active_civitai_key()
            if _civ:
                env["CIVITAI_API_KEY"] = _civ
            push(f"Spawning warm helper (low_memory={HELPER_LOW_MEMORY}, idle_timeout={HELPER_IDLE_TIMEOUT}s)")
            self.proc = subprocess.Popen(
                [str(HELPER_PYTHON), str(HELPER_SCRIPT)],
                cwd=str(MLX), env=env,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, start_new_session=True,
            )
            ready = self._read_until(["ready", "error", "exit"], timeout=120)
            if not ready or ready.get("event") != "ready":
                raise RuntimeError(f"helper failed to start: {ready}")
            push(f"helper ready · model={ready.get('model')} · low_memory={ready.get('low_memory')}")

    def _read_until(self, target_events: list[str], timeout: float | None = None) -> dict | None:
        if not self.proc or not self.proc.stdout:
            return None
        deadline = time.time() + timeout if timeout else None
        while True:
            if deadline and time.time() > deadline:
                return None
            line = self.proc.stdout.readline()
            if not line:
                return None
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                push(line)
                continue
            ev_type = ev.get("event")
            if ev_type == "log":
                push(ev.get("line", ""))
            elif ev_type in target_events:
                return ev
            else:
                push(f"helper {ev_type}: {json.dumps(ev)[:200]}")

    def run(self, job_spec: dict) -> dict:
        self._ensure()
        with self.lock:
            assert self.proc is not None and self.proc.stdin is not None
            try:
                self.proc.stdin.write(json.dumps(job_spec) + "\n")
                self.proc.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise RuntimeError(f"helper stdin closed: {exc}")
        ev = self._read_until(["done", "error", "exit"])
        if ev is None:
            # Pipe closed without an event. peanut review correction: don't
            # claim "SIGKILL" with certainty — could be SIGKILL (OOM jetsam),
            # SIGSEGV (Metal/MLX fault), SIGABRT (assertion), or any other
            # native-level abort. Inspect proc.returncode to actually name
            # the signal so users have a real datapoint to share.
            proc = self.proc
            rc = proc.poll() if proc else None
            if rc is not None and rc < 0:
                sig_num = -rc
                try:
                    sig_name = signal.Signals(sig_num).name
                except (ValueError, AttributeError):
                    sig_name = f"signal{sig_num}"
                # Translate the common ones into intent — what the OS was
                # likely telling us when it sent that signal.
                hint = {
                    "SIGKILL": "out of memory (jetsam) or external kill — close memory-heavy apps and retry",
                    "SIGSEGV": "native segfault inside MLX/Metal — share the crashlog at ~/Library/Logs/DiagnosticReports/python3.11_*.crash",
                    "SIGABRT": "C-level assertion failed — share the crashlog as above",
                    "SIGBUS":  "memory access fault — could indicate a Metal driver issue or a bad weight file",
                }.get(sig_name, "external kill")
                raise RuntimeError(
                    f"helper exited from {sig_name} ({hint}); returncode={rc}"
                )
            # rc is None (still running but pipe closed?) or rc >= 0
            # (graceful exit but missed the event). Both are unusual.
            raise RuntimeError(
                f"helper pipe closed without an event; returncode={rc}. "
                "Check the panel log for the last step:* breadcrumb to see "
                "which phase died, and the crashlog at "
                "~/Library/Logs/DiagnosticReports/python3.11_*.crash if any."
            )
        if ev.get("event") == "error":
            raise RuntimeError(ev.get("error", "helper error"))
        if ev.get("event") == "exit":
            raise RuntimeError(f"helper exited mid-job: {ev.get('reason')}")
        return ev

    def kill(self) -> None:
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:
                self.proc = None
                return
            try:
                pgid = os.getpgid(self.proc.pid)
                os.killpg(pgid, signal.SIGTERM)
                push(f"helper SIGTERM sent (pgid {pgid})")
            except ProcessLookupError:
                pgid = None
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                if pgid is not None:
                    try:
                        os.killpg(pgid, signal.SIGKILL)
                    except Exception:
                        pass
            self.proc = None

    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def pid(self) -> int | None:
        return self.proc.pid if self.is_alive() else None


HELPER = WarmHelper()
atexit.register(HELPER.kill)
atexit.register(caffeinate_off)


# ---- generation pipeline -----------------------------------------------------

def compute_pad(w: int, h: int) -> tuple[int, int, str | None]:
    target_w = 720 if w == 704 and h % 16 == 0 else w
    target_h = 720 if h == 704 and w % 16 == 0 else h
    if target_w == w and target_h == h:
        return target_w, target_h, None
    pad_x = (target_w - w) // 2
    pad_y = (target_h - h) // 2
    return target_w, target_h, f"pad={target_w}:{target_h}:{pad_x}:{pad_y}:color=black"


def video_duration(frames: int) -> float:
    return round(frames / FPS, 3)


def stop_current_job(timeout: float = 5.0) -> None:
    """Kill the warm helper (and any in-flight mux subprocess). Worker advances."""
    with LOCK:
        cur = STATE["current"]
        mux_pgid = STATE.get("mux_pgid")
    if cur is not None:
        cur["cancel_requested"] = True
    push("Stop requested — killing helper + mux to abort current job.")
    HELPER.kill()
    # Mux runs in its own process group outside the helper's. Kill it too,
    # otherwise ffmpeg keeps writing the (now-orphaned) output file.
    if mux_pgid:
        try:
            os.killpg(mux_pgid, signal.SIGTERM)
            push(f"SIGTERM sent to mux pgid {mux_pgid}")
        except ProcessLookupError:
            pass


_JOB_COUNTER = 0
_JOB_COUNTER_LOCK = threading.Lock()


def _new_job_id() -> str:
    global _JOB_COUNTER
    with _JOB_COUNTER_LOCK:
        _JOB_COUNTER += 1
        n = _JOB_COUNTER
    return f"j-{int(time.time()*1000):x}-{n:03d}"


def make_job(form: dict[str, list[str]] | dict[str, str], *,
             override_prompt: str | None = None) -> dict:
    def f(name: str, default: str = "") -> str:
        val = form.get(name, default)
        if isinstance(val, list):
            val = val[0] if val else default
        return (val or "").strip() or default

    prompt = override_prompt if override_prompt is not None else f("prompt", "")
    if not prompt:
        prompt = "A cinematic atmospheric scene"

    return {
        "id": _new_job_id(),
        "status": "queued",
        "queued_at": iso_now(),
        "started_at": None,
        "started_ts": None,
        "finished_at": None,
        "elapsed_sec": None,
        "params": {
            "mode": f("mode", "t2v"),
            "prompt": prompt,
            "width": max(32, int(f("width", "1280") or 1280)),
            "height": max(32, int(f("height", "704") or 704)),
            "frames": max(1, int(f("frames", "121") or 121)),
            "steps": max(1, int(f("steps", "8") or 8)),
            "seed": f("seed", "-1") or "-1",
            "image": f("image", str(REFERENCE)),
            "audio": f("audio", str(AUDIO_DEFAULT)),
            # extend mode params
            "video_path": f("video_path", ""),
            "extend_frames": max(1, int(f("extend_frames", "5") or 5)),
            "extend_direction": f("extend_direction", "after"),
            "extend_steps": max(1, int(f("extend_steps", "12") or 12)),
            "extend_cfg": float(f("extend_cfg", "1.0") or 1.0),
            # keyframe (FFLF) mode params
            "start_image": f("start_image", ""),
            "end_image": f("end_image", ""),
            "enhance": f("enhance", "off") == "on",
            "stop_comfy": f("stop_comfy", "off") == "on",
            "open_when_done": f("open_when_done", "off") == "on",
            "label": f("preset_label", "") or None,
            "quality": f("quality", "standard"),  # draft / standard / high
            # LoRAs the user has enabled for this job. The UI submits a
            # JSON-encoded array via the `loras` form field; we parse +
            # validate here so the worker / helper layers receive a clean
            # list of {path, strength}. HDR toggle gets injected here too
            # (see make_job_postprocess below).
            "loras": parse_loras_from_form(form),
            # HDR toggle. Stored separately from `loras` so the UI can
            # render it as a plain checkbox without the user thinking of
            # it as a LoRA. The worker resolves it to the curated HDR
            # repo before submitting to the helper.
            "hdr": f("hdr", "off") == "on",
        },
        "command": None,
        "raw_path": None,
        "output_path": None,
        "error": None,
    }


def run_job_inner(job: dict) -> None:
    p = job["params"]
    mode = p["mode"]
    quality = p.get("quality", "standard")

    # Guard: Q4 distilled hardcoded 9-sigma schedule needs the full walk to
    # sigma=0. Truncating below 8 steps leaves the image partially denoised
    # (sigma=0.725 at 6 steps, sigma=0.975 at 4 steps) — i.e. literal noise.
    # Block before the user wastes 7+ minutes producing static.
    # Modes that don't use the distilled `steps` field skip this check:
    #   - extend / keyframe use stage1_steps + stage2_steps via two-stage path
    #   - high quality uses two-stage HQ with its own schedule
    if mode not in ("extend", "keyframe") and quality != "high" and int(p.get("steps", 8)) < 8:
        raise RuntimeError(
            f"steps={p.get('steps')} is below the 8-step minimum for the Q4 distilled "
            "schedule. Fewer steps truncates the sigma walk and leaves >70% noise in "
            "the output (this is what you saw last run). Use steps=8 for standard "
            "renders, or pick Quality=Draft for a faster smaller-resolution render at "
            "the same 8 steps."
        )

    if p["stop_comfy"]:
        kill_comfy()
        time.sleep(1)

    stamp = time.strftime("%Y%m%d_%H%M%S")

    if mode == "extend":
        # Extend: input video → longer video
        if not SYSTEM_CAPS["allows_extend"]:
            raise RuntimeError(
                f"Extend isn't supported on the {SYSTEM_CAPS['label']} hardware "
                f"tier — the dev transformer needs more headroom than this Mac "
                f"has. Bump to 64+ GB or render the longer clip in one shot."
            )
        src = p["video_path"]
        if not src or not Path(src).exists():
            raise RuntimeError(f"source video for extend not found: {src}")
        # Resolution clamp — same fix shape as FFLF. The upstream extend
        # pipeline runs the dev transformer in CFG-guided mode (line 294
        # of extend.py — guided_denoise_loop is unconditional; cfg_scale=1.0
        # changes the math but not the activation memory). max_dim is
        # tier-derived: 768 on standard (64 GB), 1024 on high (96 GB),
        # unclamped on pro (128+ GB). Cached by target dimensions so
        # repeated extends on the same source skip the re-encode.
        original_src = Path(src)
        ext_max = tier_max_dim("extend")
        if ext_max:
            downscaled_src = _ensure_downscaled(original_src, max_dim=ext_max)
            if downscaled_src != original_src:
                push(f"Extend: source {original_src.name} downscaled to "
                     f"{downscaled_src.name} (≤{ext_max} max-side, "
                     f"{SYSTEM_CAPS['label']} tier).")
                src = str(downscaled_src)
        else:
            push(f"Extend: no resolution clamp ({SYSTEM_CAPS['label']} tier).")
        out_name = Path(src).stem + f"_ext{p['extend_frames']}_{stamp}.mp4"
        final_out = OUTPUT / out_name
        job["raw_path"] = str(final_out)

        # Extend memory profile: pipe loads the ~10–12 GB dev transformer
        # (Q4-quantized) and does CFG-guided denoising over the source's
        # native resolution. At 1280×704 + CFG 3.0 we OOM into swap on 64 GB
        # Macs (peak ~47 GB resident + 12 GB swap → 240s/step). Default to
        # cfg_scale=1.0 (no CFG, ~half the activation memory, fits cleanly)
        # and 12 steps. Form exposes a "Quality" toggle that flips to
        # cfg=3.0 + steps=30 for users with the headroom.
        cfg_scale = float(p.get("extend_cfg") or 1.0)
        steps = int(p["extend_steps"]) if p.get("extend_steps") else 12
        job_spec = {
            "action": "extend",
            "id": job["id"],
            "params": {
                "prompt": p["prompt"],
                "video_path": src,
                "extend_frames": p["extend_frames"],
                "direction": p["extend_direction"],
                "output_path": str(final_out),
                "seed": p["seed"],
                "steps": steps,
                "cfg_scale": cfg_scale,
            },
        }
        push(f"Extend via helper: id={job['id']} src={Path(src).name} +{p['extend_frames']}f · "
             f"steps={steps} cfg={cfg_scale}")
        result = HELPER.run(job_spec)
        if "seed_used" in result:
            push(f"seed used: {result['seed_used']}")
            p["seed_used"] = result["seed_used"]

        sidecar = {
            "output": str(final_out), "raw_output": str(final_out),
            "params": {**p, "command": "extend"},
            "started": job.get("started_at"),
            "elapsed_sec": round(time.time() - job["started_ts"], 2) if job.get("started_ts") else None,
            "fps": FPS, "model": MODEL_ID, "queue_id": job["id"],
            "helper_elapsed_sec": result.get("elapsed_sec"),
        }
        write_sidecar(final_out.with_suffix(final_out.suffix + ".json"), sidecar)
        job["output_path"] = str(final_out)
        push(f"Extend done in {sidecar['elapsed_sec']}s → {final_out.name}")
        if p.get("open_when_done"):
            subprocess.run(["open", str(final_out)], check=False)
        return

    # Keyframe (FFLF) — two images bookend the clip, model interpolates.
    # Always uses Q8 dev transformer (the pipeline inherits two-stage), so we
    # require Q8 on disk and route to generate_keyframe regardless of quality
    # tier (which doesn't really apply — there's no "Q4 keyframe" path).
    if mode == "keyframe":
        if not SYSTEM_CAPS["allows_keyframe"]:
            raise RuntimeError(
                f"FFLF (keyframe interpolation) isn't supported on the "
                f"{SYSTEM_CAPS['label']} hardware tier — Q8 + the dev "
                f"transformer's two-stage memory peak doesn't fit. "
                f"Bump to 64+ GB."
            )
        # Check the SAME file list the menu + /status report on. The old
        # "directory exists and non-empty" check let half-downloaded Q8
        # installs through, only to crash later mid-render with a
        # `load_safetensors` error referencing a specific missing file.
        kf_missing = q8_missing_files()
        if kf_missing:
            raise RuntimeError(
                f"Keyframe mode requires the full Q8 model at {Q8_LOCAL_PATH}. "
                f"Missing {len(kf_missing)} file(s): {', '.join(kf_missing[:3])}"
                f"{' …' if len(kf_missing) > 3 else ''}. "
                f"Run: hf download {MODEL_ID_HQ} --local-dir {Q8_LOCAL_PATH}"
            )
        if not p.get("start_image") or not Path(p["start_image"]).exists():
            raise RuntimeError(f"start_image not found: {p.get('start_image')}")
        if not p.get("end_image") or not Path(p["end_image"]).exists():
            raise RuntimeError(f"end_image not found: {p.get('end_image')}")
        # Clamp keyframe-mode resolution to the tier's max-dim. The Q8
        # KeyframeInterpolationPipeline OOMs on 64 GB Macs at the stage-1 →
        # stage-2 transition for 1280×704 (peak memory hits the ceiling
        # during the upscale + VAE-encoder reload). On standard tier we
        # clamp to 768; on high tier we go to 1024; pro tier has no clamp.
        kf_max = tier_max_dim("keyframe")
        KF_ALIGN = 32
        req_w, req_h = p["width"], p["height"]
        if kf_max and max(req_w, req_h) > kf_max:
            scale = kf_max / max(req_w, req_h)
            width = max(KF_ALIGN, int(round(req_w * scale / KF_ALIGN)) * KF_ALIGN)
            height = max(KF_ALIGN, int(round(req_h * scale / KF_ALIGN)) * KF_ALIGN)
            push(
                f"Keyframe: clamping {req_w}×{req_h} → {width}×{height} "
                f"({SYSTEM_CAPS['label']} tier — {kf_max} max-side keeps "
                f"the dev DiT + upscaler + 4×-volume upscaled tensor in RAM)."
            )
            # Persist back to params so the sidecar / history reflect the
            # actual rendered resolution, not the form input.
            p["width"], p["height"] = width, height
        else:
            width, height = req_w, req_h
        frames = p["frames"]
        out_path = OUTPUT / f"mlx_keyframe_{width}x{height}_{frames}f_{stamp}.mp4"
        job["raw_path"] = str(out_path)
        job_spec = {
            "action": "generate_keyframe",
            "id": job["id"],
            "params": {
                "model_dir": str(Q8_LOCAL_PATH),
                "prompt": p["prompt"],
                "output_path": str(out_path),
                "start_image": p["start_image"],
                "end_image": p["end_image"],
                "height": height,
                "width": width,
                "frames": frames,
                "seed": p["seed"],
                # Upstream reference for the dev-model keyframe path is 20
                # stage-1 steps (`s1_steps = stage1_steps or 20` in
                # keyframe_interpolation.py). We were using 15 — slightly
                # under-cooked. Bumping to match upstream reference quality;
                # ~33% more wall time for noticeably cleaner motion.
                "stage1_steps": 20,
                "stage2_steps": 3,
                "cfg_scale": 3.0,
            },
        }
        push(f"Run KEYFRAME via helper: id={job['id']} {width}x{height} {frames}f · Q8 two-stage (stage1=20)")
        result = HELPER.run(job_spec)
        if "seed_used" in result:
            push(f"seed used: {result['seed_used']}")
            p["seed_used"] = result["seed_used"]
        sidecar = {
            "output": str(out_path), "raw_output": str(out_path),
            "params": {**p, "command": "keyframe"},
            "started": job.get("started_at"),
            "elapsed_sec": round(time.time() - job["started_ts"], 2) if job.get("started_ts") else None,
            "video_duration_sec": video_duration(frames),
            "fps": FPS, "model": MODEL_ID_HQ, "queue_id": job["id"],
            "helper_elapsed_sec": result.get("elapsed_sec"),
        }
        write_sidecar(out_path.with_suffix(out_path.suffix + ".json"), sidecar)
        job["output_path"] = str(out_path)
        push(f"Keyframe done in {sidecar['elapsed_sec']}s → {out_path.name}")
        if p.get("open_when_done"):
            subprocess.run(["open", str(out_path)], check=False)
        return

    # T2V / I2V / I2V+clean_audio
    width, height = p["width"], p["height"]
    frames = p["frames"]
    quality = p.get("quality", "standard")

    # T2V/I2V resolution clamp — only applies on the base tier (< 48 GB).
    # Standard / high / pro tiers pass full user-requested W×H through.
    t2v_max = tier_max_dim("t2v" if mode == "t2v" else "i2v")
    if t2v_max and max(width, height) > t2v_max:
        scale = t2v_max / max(width, height)
        new_w = max(32, (int(round(width * scale)) // 32) * 32)
        new_h = max(32, (int(round(height * scale)) // 32) * 32)
        push(
            f"{mode.upper()}: clamping {width}×{height} → {new_w}×{new_h} "
            f"({SYSTEM_CAPS['label']} tier — keeps you out of swap)."
        )
        width, height = new_w, new_h
        p["width"], p["height"] = width, height

    pad_w, pad_h, pad_filter = compute_pad(width, height)
    suffix = f"{pad_w}x{pad_h}" if mode == "i2v_clean_audio" and pad_filter else f"{width}x{height}"
    tag = f"{mode}_hq" if quality == "high" else mode
    # Only `i2v_clean_audio` runs a panel-side mux (raw → final). For T2V / I2V /
    # HQ the upstream-patched encode writes the lossless yuv444p crf 0 + AAC
    # file directly, so the "raw" is the final — keeping the `_raw` suffix in
    # that case made the filenames look half-finished and confused later
    # tooling that pattern-matched on `_raw`.
    needs_mux = mode == "i2v_clean_audio"
    if needs_mux:
        raw_out = OUTPUT / f"mlx_{tag}_{width}x{height}_{frames}f_{stamp}_raw.mp4"
        final_out = OUTPUT / f"mlx_{tag}_{suffix}_{frames}f_{stamp}.mp4"
    else:
        # Single file — name it as the final, no `_raw` suffix.
        raw_out = OUTPUT / f"mlx_{tag}_{width}x{height}_{frames}f_{stamp}.mp4"
        final_out = raw_out
    job["raw_path"] = str(raw_out)

    if quality == "high":
        if not SYSTEM_CAPS["allows_q8"]:
            raise RuntimeError(
                f"High quality (Q8 two-stage) isn't supported on the "
                f"{SYSTEM_CAPS['label']} hardware tier — Q8 dev transformer "
                f"(~19 GB) plus the upscaler stage doesn't fit. "
                f"Use Standard or Draft instead, or upgrade to 64+ GB."
            )
        # Route to TwoStageHQPipeline (Q8 dev model + res_2s sampler + CFG anchor + TeaCache).
        # Defaults from ltx-2-mlx CLAUDE.md LTX_2_3_PARAMS.
        # Same completeness check as keyframe — see comment there.
        hq_missing = q8_missing_files()
        if hq_missing:
            raise RuntimeError(
                f"High quality requires the full Q8 model at {Q8_LOCAL_PATH}. "
                f"Missing {len(hq_missing)} file(s): {', '.join(hq_missing[:3])}"
                f"{' …' if len(hq_missing) > 3 else ''}. "
                f"Run: hf download {MODEL_ID_HQ} --local-dir {Q8_LOCAL_PATH}"
            )
        job_spec = {
            "action": "generate_hq",
            "id": job["id"],
            "params": {
                "model_dir": str(Q8_LOCAL_PATH),
                "prompt": p["prompt"],
                "output_path": str(raw_out),
                "height": height,
                "width": width,
                "frames": frames,
                "seed": p["seed"],
                "image": p["image"] if mode != "t2v" else None,
                "stage1_steps": 15,
                "stage2_steps": 3,
                "cfg_scale": 3.0,
                # Upstream HQ params (`LTX_2_3_HQ_PARAMS`) and the
                # TwoStageHQPipeline signature both default stg_scale=0.0.
                # HQ uses res_2s sampler with `stg_blocks=[]`, so STG is
                # meant to be off — passing 1.0 burns one extra forward
                # pass per outer step (~33% slower) for nothing. Was 1.0
                # by mistake (copy from standard params); fixed here.
                "stg_scale": 0.0,
                "enable_teacache": True,
                "teacache_thresh": 1.0,
            },
        }
        push(f"Run HIGH via helper: id={job['id']} mode={mode} {width}x{height} {frames}f · Q8 two-stage HQ + TeaCache")
    else:
        # Draft / Standard — Q4 one-stage with steps from form.
        # Resolve LoRAs: user-picked entries from p["loras"] plus the
        # HDR shortcut if enabled (the HDR LoRA is a curated Lightricks
        # repo we know about, kept hidden from the picker because the
        # user shouldn't have to think about it as a LoRA).
        loras = list(p.get("loras") or [])
        if p.get("hdr"):
            loras.append({
                "path": CURATED_LORAS["hdr"]["repo_id"],
                "strength": float(CURATED_LORAS["hdr"]["default_strength"]),
            })
        job_spec = {
            "action": "generate",
            "id": job["id"],
            "params": {
                "mode": mode,
                "prompt": p["prompt"],
                "output_path": str(raw_out),
                "height": height,
                "width": width,
                "frames": frames,
                "steps": p["steps"],
                "seed": p["seed"],
                "image": p["image"] if mode != "t2v" else None,
                "loras": loras,
            },
        }
        if loras:
            push(f"Run via helper: id={job['id']} mode={mode} quality={quality} "
                 f"{width}x{height} {frames}f · {len(loras)} LoRA"
                 f"{'s' if len(loras) != 1 else ''}"
                 f"{' (incl. HDR)' if p.get('hdr') else ''}")
        else:
            push(f"Run via helper: id={job['id']} mode={mode} quality={quality} "
                 f"{width}x{height} {frames}f")

    result = HELPER.run(job_spec)
    if "seed_used" in result:
        push(f"seed used: {result['seed_used']}")
        p["seed_used"] = result["seed_used"]

    final_target = raw_out
    if mode == "i2v_clean_audio":
        audio = p["audio"]
        if not Path(audio).exists():
            raise RuntimeError(f"audio file not found: {audio}")
        duration = video_duration(frames)
        env = os.environ.copy()
        env["PATH"] = f"{FFMPEG_BIN}:{env.get('PATH', '')}"
        # Match the helper's lossless codec defaults (overridable via the
        # same env vars the upstream patch script reads). Previously this
        # mux step hardcoded yuv420p crf 18, undoing the codec patch for
        # i2v_clean_audio mode and re-introducing chroma-block artifacts.
        mux_pix_fmt = os.environ.get("LTX_OUTPUT_PIX_FMT", "yuv444p")
        mux_crf = os.environ.get("LTX_OUTPUT_CRF", "0")
        mux_cmd = [str(FFMPEG), "-y", "-i", str(raw_out), "-i", audio,
                   "-map", "0:v:0", "-map", "1:a:0"]
        if pad_filter:
            mux_cmd += ["-vf", pad_filter]
        mux_cmd += [
            "-af", f"apad,atrim=0:{duration},asetpts=PTS-STARTPTS",
            "-c:v", "libx264", "-pix_fmt", mux_pix_fmt, "-crf", mux_crf, "-preset", "medium",
            "-c:a", "aac", "-b:a", "192k",
            "-t", f"{duration}",
            str(final_out),
        ]
        push("Mux: " + " ".join(shlex.quote(c) for c in mux_cmd))
        # Run mux in its own process group so /stop can SIGTERM the whole
        # ffmpeg pipeline (it lives outside the helper's pgid). Track pgid
        # in STATE for stop_current_job() to find.
        mux_proc = subprocess.Popen(
            mux_cmd, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True,
        )
        with LOCK:
            STATE["mux_pgid"] = os.getpgid(mux_proc.pid)
        try:
            stdout, stderr = mux_proc.communicate()
        finally:
            with LOCK:
                STATE["mux_pgid"] = None
        if mux_proc.returncode != 0:
            push((stderr or "").strip())
            raise RuntimeError(f"mux exited with code {mux_proc.returncode}")
        final_target = final_out

    sidecar = {
        "output": str(final_target),
        "raw_output": str(raw_out),
        "params": {
            **p,
            "pad_w": pad_w, "pad_h": pad_h,
            "image": p["image"] if mode != "t2v" else None,
            "audio": p["audio"] if mode == "i2v_clean_audio" else None,
        },
        "command": "helper",
        "started": job.get("started_at"),
        "elapsed_sec": round(time.time() - job["started_ts"], 2) if job.get("started_ts") else None,
        "video_duration_sec": video_duration(frames),
        "fps": FPS, "model": MODEL_ID, "queue_id": job["id"],
        "helper_elapsed_sec": result.get("elapsed_sec"),
    }
    write_sidecar(final_target.with_suffix(final_target.suffix + ".json"), sidecar)
    job["output_path"] = str(final_target)
    push(f"Done in {sidecar['elapsed_sec']}s → {final_target.name}")
    if p.get("open_when_done"):
        subprocess.run(["open", str(final_target)], check=False)


# ---- worker thread -----------------------------------------------------------

def worker_loop() -> None:
    while True:
        with QUEUE_COND:
            while not STATE["queue"] or STATE["paused"]:
                if not STATE["queue"] and not STATE["current"]:
                    caffeinate_off()
                QUEUE_COND.wait(timeout=2)
            job = STATE["queue"].pop(0)
            job["status"] = "running"
            job["started_at"] = iso_now()
            job["started_ts"] = time.time()
            STATE["current"] = job
            STATE["running"] = True
            STATE["log"] = []
            caffeinate_on()
        persist_queue()

        try:
            run_job_inner(job)
            job["status"] = "done"
        except Exception as exc:
            if job.get("cancel_requested"):
                job["status"] = "cancelled"
                push("Job cancelled.")
            else:
                job["status"] = "failed"
                job["error"] = str(exc)
                push(f"ERROR: {exc}")
        finally:
            job["finished_at"] = iso_now()
            if job.get("started_ts"):
                job["elapsed_sec"] = round(time.time() - job["started_ts"], 2)
            with LOCK:
                STATE["history"].insert(0, job)
                STATE["history"] = STATE["history"][:HISTORY_LIMIT]
                STATE["current"] = None
                STATE["running"] = False
                STATE["pid"] = None
                STATE["pgid"] = None
            persist_queue()


# ---- HTTP --------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002
        return

    def _ok(self, body: bytes, content_type: str = "text/html; charset=utf-8") -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_video_with_range(self, path: Path) -> None:
        """Serve an mp4 with HTTP byte-range support so the browser <video>
        tag can seek without redownloading and the gallery's `preload="metadata"`
        thumbnails only fetch the moov atom range instead of full clips.

        Without this every preview pulled the whole 50–80 MB file at page load.
        Spec: RFC 7233. We support a single `bytes=start-end` form (the only
        one Chrome / Safari send) and ignore multi-range requests."""
        size = path.stat().st_size
        rng = self.headers.get("Range", "")
        if rng.startswith("bytes="):
            try:
                spec = rng.split("=", 1)[1]
                # We don't honor multi-range; first one wins.
                spec = spec.split(",", 1)[0].strip()
                start_s, end_s = spec.split("-", 1)
                # RFC 7233 § 2.1 has three forms:
                #   bytes=N-M   first..last (closed interval)
                #   bytes=N-    N..end-of-file
                #   bytes=-N    last N bytes (start_s empty, end_s is the count)
                # The third form is what browsers send for the mp4 moov atom
                # at the tail of the file. Initial implementation treated it
                # as bytes=0-N which is wrong.
                if not start_s and end_s:
                    suffix_len = int(end_s)
                    if suffix_len <= 0:
                        raise ValueError("zero-length suffix range")
                    start = max(0, size - suffix_len)
                    end = size - 1
                else:
                    start = int(start_s) if start_s else 0
                    end = int(end_s) if end_s else size - 1
                if start < 0 or end >= size or start > end:
                    raise ValueError("range out of bounds")
            except (ValueError, IndexError):
                # Malformed range header → 416 per RFC 7233.
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            length = end - start + 1
            self.send_response(206)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(length))
            self.end_headers()
            with path.open("rb") as fh:
                fh.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = fh.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        # Browser hung up mid-stream (user scrubbed or closed
                        # tab). Not an error — exit cleanly.
                        return
                    remaining -= len(chunk)
            return

        # No Range header — full file with Accept-Ranges advertised so the
        # browser knows it CAN range-request next time.
        self.send_response(200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Content-Length", str(size))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        with path.open("rb") as fh:
            while chunk := fh.read(1024 * 1024):
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._ok(page().encode())
            return
        if parsed.path == "/status":
            qs = parse_qs(parsed.query)
            include_hidden = qs.get("include_hidden", ["0"])[0] == "1"
            # Deep-snapshot STATE under lock — payload built with refs only
            # is racy when JSON serialization happens after lock release
            # (worker thread could mutate current/queue mid-encode and we'd
            # ship torn state to the browser).
            import copy as _copy
            with LOCK:
                avg = _avg_elapsed()
                payload = _copy.deepcopy({
                    "running": STATE["running"], "paused": STATE["paused"],
                    "current": STATE["current"], "queue": STATE["queue"],
                    "history": STATE["history"][:HISTORY_API_LIMIT], "log": STATE["log"],
                    "pid": STATE["pid"], "pgid": STATE["pgid"],
                })
                hidden_count = len(HIDDEN_PATHS)
            payload["outputs"] = list_outputs(include_hidden=include_hidden)
            payload["hidden_count"] = hidden_count
            payload["memory"] = get_memory()
            payload["comfy_pids"] = find_comfy_pids()
            payload["server_now"] = time.time()
            payload["avg_elapsed_sec"] = avg
            payload["eta_sec"] = (avg or 420) * len(payload["queue"])
            payload["helper"] = {
                "alive": HELPER.is_alive(), "pid": HELPER.pid(),
                "low_memory": HELPER_LOW_MEMORY == "true",
                "idle_timeout_sec": HELPER_IDLE_TIMEOUT,
            }
            # Completeness checks come from the shared required_files.json so
            # the menu, the UI, and the run-time job validator all agree on
            # what counts as "installed". Single source of truth — see the
            # _load_required_files() helper near the top of this file.
            _q8_missing = q8_missing_files()
            _base_missing = base_missing()
            payload["q8_available"] = not _q8_missing
            payload["q8_missing"] = _q8_missing
            payload["q8_path"] = str(Q8_LOCAL_PATH)
            payload["base_available"] = not _base_missing
            payload["base_missing"] = _base_missing
            # Repo-level counts for the header pill — granular view that
            # matches the modal's per-repo rows (Q4 + Gemma + Q8 = 3 in the
            # default manifest). Avoids the pill claiming "2/2 ready" while
            # the modal shows three rows.
            _repo_snap = repo_status_list()
            payload["repos_total"] = len(_repo_snap)
            payload["repos_ready"] = sum(1 for r in _repo_snap if r.get("complete"))
            # Hardware tier — UI uses this to disable mode pills / quality
            # buttons / show a helpful banner explaining what this Mac can
            # and can't do. Detected once at startup; the override env
            # var lets users force a tier for testing.
            payload["tier"] = {
                "key": SYSTEM_TIER,
                "label": SYSTEM_CAPS["label"],
                "ram_label": SYSTEM_CAPS["ram_label"],
                "tagline": SYSTEM_CAPS["tagline"],
                "blurb": SYSTEM_CAPS["blurb"],
                "allows_q8": SYSTEM_CAPS["allows_q8"],
                "allows_keyframe": SYSTEM_CAPS["allows_keyframe"],
                "allows_extend": SYSTEM_CAPS["allows_extend"],
                "t2v_max_dim": SYSTEM_CAPS["t2v_max_dim"],
                "i2v_max_dim": SYSTEM_CAPS["i2v_max_dim"],
                "keyframe_max_dim": SYSTEM_CAPS["keyframe_max_dim"],
                "extend_max_dim": SYSTEM_CAPS["extend_max_dim"],
                "times": SYSTEM_CAPS.get("times", {}),
            }
            # Active model-download status — UI shows a progress strip when
            # this is set. last_line is the most recent hf output line so the
            # user gets live feedback even before opening the log panel.
            with DOWNLOAD_LOCK:
                if DOWNLOAD["active"]:
                    payload["download"] = {
                        "active": True,
                        "key": DOWNLOAD["key"],
                        "repo_id": DOWNLOAD["repo_id"],
                        "started_ts": DOWNLOAD["started_ts"],
                        "last_line": DOWNLOAD["last_line"],
                    }
                else:
                    payload["download"] = {"active": False}
            payload["hf_available"] = HF_BIN is not None
            # Settings snapshot — only needs the public-safe view (booleans
            # for token presence, no secret values). The UI reads
            # `settings.models_card_dismissed` on each /status tick to know
            # whether to keep the inline models card hidden.
            payload["settings"] = get_settings_public()
            self._json(payload)
            return
        if parsed.path == "/uploads":
            # Recent panel_uploads/* images for the picker's "click to reuse"
            # strip. Limit defaults to 40; client can paginate later if needed.
            try:
                limit = int(parse_qs(parsed.query).get("limit", ["40"])[0])
            except (TypeError, ValueError):
                limit = 40
            self._json({"uploads": list_uploads(limit=max(1, min(200, limit)))})
            return
        if parsed.path == "/models":
            # Per-repo status snapshot for the Models modal in the UI.
            # Same data the menu/install rely on, just shaped per-repo so
            # the front-end can render rows without re-aggregating.
            payload = {
                "repos": repo_status_list(),
                "hf_available": HF_BIN is not None,
                "hf_path": str(HF_BIN) if HF_BIN else None,
            }
            with DOWNLOAD_LOCK:
                payload["active_download"] = (
                    {"key": DOWNLOAD["key"], "repo_id": DOWNLOAD["repo_id"],
                     "started_ts": DOWNLOAD["started_ts"], "last_line": DOWNLOAD["last_line"]}
                    if DOWNLOAD["active"] else None
                )
            self._json(payload)
            return
        if parsed.path == "/settings":
            # Return current panel settings + the preset table so the UI
            # can render preset pills with labels and blurbs without
            # hardcoding any of it on the client side. Secrets are
            # surfaced as has_X booleans only — actual key values
            # never leave the panel process.
            self._json({
                "settings": get_settings_public(),
                "presets": OUTPUT_PRESETS,
                "default_preset": DEFAULT_OUTPUT_PRESET,
            })
            return
        if parsed.path == "/civitai/test":
            # Sanity-check the saved CivitAI key by hitting an
            # auth-required endpoint and reporting back the upstream
            # status. Lets the Settings UI render a green/red dot
            # without users having to risk a 300 MB download just to
            # discover the key is malformed.
            key = _active_civitai_key()
            if not key:
                self._json({"ok": False, "error": "No CivitAI key configured."}, 400)
                return
            try:
                # /api/v1/me requires auth; success returns the user
                # profile, failure returns 401. We never echo the
                # username back — just enough info to tell the user
                # the key works.
                _civitai_request("/me", timeout=10)
                self._json({"ok": True, "message": "CivitAI auth works."})
            except Exception as exc:
                msg = str(exc)
                if "401" in msg or "403" in msg:
                    self._json({
                        "ok": False,
                        "error": "Key rejected by CivitAI (401/403). "
                                 "Re-paste the key and try again, or generate a new one.",
                    }, 401)
                else:
                    self._json({
                        "ok": False,
                        "error": f"Network error reaching CivitAI: {msg[:200]}",
                    }, 502)
            return
        if parsed.path == "/hf/test":
            # Same idea for Hugging Face — call /api/whoami-v2 which
            # is auth-required.
            token = _active_hf_token()
            if not token:
                self._json({"ok": False, "error": "No Hugging Face token configured."}, 400)
                return
            try:
                import urllib.request
                req = urllib.request.Request(
                    "https://huggingface.co/api/whoami-v2",
                    headers={"Authorization": f"Bearer {token}",
                             "User-Agent": CIVITAI_USER_AGENT},
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    body = resp.read().decode("utf-8", "replace")
                # Don't echo the username — just confirm.
                self._json({"ok": True, "message": "Hugging Face auth works."})
            except urllib.request.HTTPError as he:
                if he.code in (401, 403):
                    self._json({
                        "ok": False,
                        "error": "Token rejected by Hugging Face (401/403). "
                                 "Re-paste the token, or generate a new one with read access.",
                    }, 401)
                else:
                    self._json({
                        "ok": False,
                        "error": f"HTTP {he.code} reaching Hugging Face.",
                    }, 502)
            except Exception as exc:
                self._json({
                    "ok": False,
                    "error": f"Network error reaching Hugging Face: {str(exc)[:200]}",
                }, 502)
            return
        if parsed.path == "/loras":
            # Returns: { user: [user-installed], curated: [Lightricks
            # officials minus hdr_toggle entries], loras_dir: <abs path>,
            # civitai_auth: bool }.
            # The HDR-toggle special-case is filtered out of `curated`
            # because the UI exposes it as a plain checkbox elsewhere —
            # showing it in the picker would just confuse users.
            curated = [c for c in list_curated_loras()
                       if not c.get("is_hdr_toggle")]
            self._json({
                "user": list_user_loras(),
                "curated": curated,
                "loras_dir": str(_safe_loras_dir()),
                # True iff a CivitAI key is configured. Source of truth:
                # the saved panel settings first, falling back to the
                # env var if a power user prefers shell-level config.
                "civitai_auth": bool(_active_civitai_key()),
                # Same pattern for HF — used for gated repo downloads
                # (HDR LoRA, etc.).
                "hf_auth": bool(_active_hf_token()),
            })
            return
        if parsed.path == "/civitai/search":
            # Proxy CivitAI's API. Filtering down to LTX-Video LoRAs by
            # baseModel ("LTXV 2.3" is the canonical string used on
            # civitai.com for LTX-2.3 LoRAs as of 2026-05). Returns the
            # subset of fields the panel cares about, plus a flat
            # download_url that points at the .safetensors directly.
            qs = parse_qs(urlparse(self.path).query)
            query = qs.get("query", [""])[0]
            nsfw = (qs.get("nsfw", ["false"])[0] or "false").lower() == "true"
            cursor = qs.get("cursor", [""])[0]
            limit = max(1, min(50, int(qs.get("limit", ["20"])[0] or "20")))
            try:
                results = _civitai_search(query=query, nsfw=nsfw,
                                         cursor=cursor, limit=limit)
                self._json(results)
            except Exception as exc:
                self._json({"error": f"civitai search failed: {exc}",
                            "items": []}, 502)
            return
        if parsed.path == "/file":
            qs = parse_qs(parsed.query)
            try:
                path = Path(qs.get("path", [""])[0]).resolve()
            except Exception:
                self.send_error(400); return
            # Strict containment via Path.is_relative_to — str.startswith
            # would let "mlx_outputs_evil/" slip past since the prefix string
            # match is true even though the directory is a sibling, not a child.
            try:
                _ = path.relative_to(OUTPUT.resolve())
            except ValueError:
                self.send_error(404); return
            if not path.exists():
                self.send_error(404); return
            self._serve_video_with_range(path)
            return
        if parsed.path.startswith("/assets/"):
            # Serve files from <ROOT>/assets/ (creator avatar, future static).
            # Path-bound to that directory only — no traversal.
            rel = parsed.path[len("/assets/"):]
            assets_dir = (ROOT / "assets").resolve()
            try:
                path = (assets_dir / rel).resolve()
            except Exception:
                self.send_error(400); return
            if not path.is_relative_to(assets_dir) or not path.is_file():
                self.send_error(404); return
            ext = path.suffix.lower()
            ctype = {
                ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".webp": "image/webp", ".gif": "image/gif", ".svg": "image/svg+xml",
            }.get(ext, "application/octet-stream")
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(path.stat().st_size))
            self.send_header("Cache-Control", "public, max-age=86400")
            self.end_headers()
            self.wfile.write(path.read_bytes())
            return
        if parsed.path == "/image":
            qs = parse_qs(parsed.query)
            try:
                path = Path(qs.get("path", [""])[0]).resolve()
            except Exception:
                self.send_error(400); return
            # Allow any image file that exists locally (this is a local-only panel)
            if not path.exists() or not path.is_file():
                self.send_error(404); return
            ext = path.suffix.lower()
            ctype = {
                ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".webp": "image/webp", ".gif": "image/gif",
            }.get(ext, "application/octet-stream")
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(path.stat().st_size))
            self.end_headers()
            with path.open("rb") as fh:
                self.wfile.write(fh.read())
            return
        if parsed.path == "/sidecar":
            qs = parse_qs(parsed.query)
            try:
                path = Path(qs.get("path", [""])[0]).resolve()
            except Exception:
                self.send_error(400); return
            try:
                _ = path.relative_to(OUTPUT.resolve())
            except ValueError:
                self.send_error(404); return
            sidecar = path.with_suffix(path.suffix + ".json")
            if not sidecar.exists():
                self.send_error(404); return
            self._ok(sidecar.read_bytes(), "application/json")
            return
        self.send_error(404)

    def do_POST(self) -> None:
        path = self.path.split("?")[0]
        qs = parse_qs(urlparse(self.path).query)
        ctype = self.headers.get("Content-Type", "")

        # Multipart upload
        if path == "/upload" and ctype.startswith("multipart/form-data"):
            try:
                form = cgi.FieldStorage(
                    fp=self.rfile, headers=self.headers,
                    environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": ctype},
                )
                if "image" not in form:
                    self._json({"error": "no field 'image'"}, 400); return
                fld = form["image"]
                if not getattr(fld, "filename", None):
                    self._json({"error": "no filename"}, 400); return
                UPLOADS.mkdir(parents=True, exist_ok=True)
                safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", fld.filename)
                dest = UPLOADS / f"{int(time.time()*1000)}_{safe_name}"
                dest.write_bytes(fld.file.read())
                self._json({"ok": True, "path": str(dest)})
            except Exception as exc:
                self._json({"error": f"upload failed: {exc}"}, 500)
            return

        # All other POST endpoints expect urlencoded body
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode() if length else ""
        form = parse_qs(body)

        if path in ("/run", "/queue/add"):
            job = make_job(form)
            with QUEUE_COND:
                STATE["queue"].append(job)
                QUEUE_COND.notify_all()
            persist_queue()
            self._json({"ok": True, "id": job["id"]})
            return

        if path == "/queue/batch":
            raw = (form.get("prompts", [""])[0] or "").strip()
            if not raw:
                self._json({"error": "no prompts"}, 400); return
            chunks = [c.strip() for c in re.split(r"^\s*---\s*$", raw, flags=re.MULTILINE)]
            chunks = [c for c in chunks if c]
            if not chunks:
                self._json({"error": "no prompts after split"}, 400); return
            ids = []
            with QUEUE_COND:
                for prompt in chunks:
                    job = make_job(form, override_prompt=prompt)
                    job["params"]["open_when_done"] = False
                    STATE["queue"].append(job)
                    ids.append(job["id"])
                QUEUE_COND.notify_all()
            persist_queue()
            self._json({"ok": True, "added": len(ids), "ids": ids})
            return

        if path == "/queue/remove":
            job_id = qs.get("id", [""])[0] or form.get("id", [""])[0]
            removed = False
            with LOCK:
                for i, j in enumerate(STATE["queue"]):
                    if j["id"] == job_id:
                        STATE["queue"].pop(i)
                        removed = True
                        break
            persist_queue()
            self._json({"removed": removed}); return

        if path == "/queue/clear":
            with LOCK:
                count = len(STATE["queue"])
                STATE["queue"] = []
            persist_queue()
            self._json({"cleared": count}); return

        if path == "/queue/pause":
            with QUEUE_COND:
                STATE["paused"] = True
                QUEUE_COND.notify_all()
            persist_queue()
            self._json({"paused": True}); return

        if path == "/queue/resume":
            with QUEUE_COND:
                STATE["paused"] = False
                QUEUE_COND.notify_all()
            persist_queue()
            self._json({"paused": False}); return

        if path == "/output/hide":
            target = qs.get("path", [""])[0] or form.get("path", [""])[0]
            if target:
                set_hidden(target, True); self._json({"hidden": target})
            else:
                self._json({"error": "missing path"}, 400)
            return

        if path == "/output/show":
            target = qs.get("path", [""])[0] or form.get("path", [""])[0]
            if target:
                set_hidden(target, False); self._json({"shown": target})
            else:
                self._json({"error": "missing path"}, 400)
            return

        if path == "/output/show_all":
            with LOCK:
                count = len(HIDDEN_PATHS)
                HIDDEN_PATHS.clear()
            persist_hidden()
            self._json({"unhidden_count": count}); return

        if path == "/helper/restart":
            HELPER.kill()
            self._json({"ok": True}); return

        if path == "/loras/refresh":
            # Rescan mlx_models/loras/. The result is whatever
            # list_user_loras returns — filesystem is the source of
            # truth, no caching layer to invalidate.
            self._json({
                "ok": True,
                "user": list_user_loras(),
                "loras_dir": str(_safe_loras_dir()),
            })
            return

        if path == "/loras/delete":
            # Remove a user-installed LoRA (the .safetensors file plus
            # its sidecar JSON if present). Path must be inside the
            # loras dir — we resolve and bound-check to prevent
            # path-traversal mischief from a hostile form payload.
            target = form.get("path", [""])[0] or form.get("path", "")
            if isinstance(target, list): target = target[0] if target else ""
            try:
                p = Path(target).resolve()
                base = _safe_loras_dir().resolve()
                if not str(p).startswith(str(base)) or not p.is_file():
                    raise RuntimeError("path not inside loras dir")
                if p.suffix.lower() != ".safetensors":
                    raise RuntimeError("not a safetensors file")
                p.unlink()
                sidecar = p.with_suffix(".json")
                if sidecar.exists():
                    sidecar.unlink()
                self._json({"ok": True, "removed": str(p)})
            except Exception as exc:
                self._json({"ok": False, "error": str(exc)}, 400)
            return

        if path == "/civitai/download":
            # Triggers a download of a CivitAI LoRA into mlx_models/loras/.
            # Streams progress through STATE['log'] like the model
            # downloads do. Validates the requested URL points at
            # civitai.com to prevent the endpoint being weaponized as
            # a generic HTTP fetcher.
            url = form.get("download_url", [""])[0] or form.get("download_url", "")
            if isinstance(url, list): url = url[0] if url else ""
            try:
                meta_raw = form.get("meta", [""])[0] or form.get("meta", "")
                if isinstance(meta_raw, list): meta_raw = meta_raw[0] if meta_raw else ""
                meta = json.loads(meta_raw) if meta_raw else {}
            except json.JSONDecodeError:
                meta = {}
            try:
                result = _civitai_download(url, meta)
                self._json({"ok": True, **result})
            except Exception as exc:
                self._json({"ok": False, "error": str(exc)}, 400)
            return

        if path == "/settings":
            # Accept partial-patch updates: only the fields the user
            # actually changed need to be present. Validation lives in
            # _validate_settings_patch — never trust the form payload.
            #
            # Re-parse with keep_blank_values=True so that an explicit
            # empty value (e.g. `civitai_api_key=`) is treated as "clear
            # this field" rather than dropped silently. The default form
            # parser at the top of do_POST drops empty values, which
            # would otherwise turn the Clear button into a no-op.
            settings_form = parse_qs(body, keep_blank_values=True)
            payload: dict = {}
            for k, v in settings_form.items():
                payload[k] = v[0] if isinstance(v, list) else v
            prev = get_settings()
            current, err = update_settings(payload)
            if err:
                # Public-safe view on errors too — never echo a saved
                # secret back to the client even when validation fails
                # on a different field.
                self._json({"ok": False, "error": err,
                            "settings": get_settings_public()}, 400)
                return
            # Codec + token env vars are read at helper SPAWN time. If
            # the user changed any of them, kill the helper so the next
            # job respawns it with the new env. Job in flight finishes
            # with the OLD values (we're not interrupting a render).
            codec_changed = (
                prev.get("output_pix_fmt") != current.get("output_pix_fmt") or
                prev.get("output_crf") != current.get("output_crf")
            )
            tokens_changed = (
                prev.get("civitai_api_key", "") != current.get("civitai_api_key", "") or
                prev.get("hf_token", "") != current.get("hf_token", "")
            )
            if codec_changed:
                push(
                    f"settings: output codec → {current['output_pix_fmt']} "
                    f"crf {current['output_crf']} ({current['output_preset']}). "
                    f"Helper restarted; takes effect on next job."
                )
            if tokens_changed:
                # Don't log token values themselves, just the action.
                push("settings: API tokens updated. Helper restarted; "
                     "takes effect on next job.")
            if codec_changed or tokens_changed:
                HELPER.kill()
            # Return only the public-safe view — never echo the saved
            # key back to the client even on success.
            self._json({
                "ok": True,
                "settings": get_settings_public(),
                "helper_restarted": codec_changed or tokens_changed,
            })
            return

        if path == "/stop":
            stop_current_job()
            self._json({"ok": True}); return

        if path == "/stop_comfy":
            killed = kill_comfy()
            self._json({"killed": killed}); return

        if path == "/open_pinokio":
            open_pinokio()
            self._json({"ok": True}); return

        if path == "/models/download":
            # POST { repo_key: "q4" | "gemma" | "q8" }
            # Validates the key against required_files.json (so the user
            # can't trick the panel into running `hf download` on an
            # arbitrary repo by faking the form). One slot — return 409
            # if a download is already in progress.
            key = (form.get("repo_key", [""])[0] or "").strip()
            repo = next((r for r in _repos() if r.get("key") == key), None)
            if not repo:
                self._json({"error": f"unknown repo key: {key!r}. Valid keys: "
                                     f"{[r['key'] for r in _repos()]}"}, 400); return
            if HF_BIN is None:
                self._json({"error": "hf binary not found. Reinstall Phosphene "
                                     "or install huggingface_hub>=1.0 in the venv."}, 500); return
            with DOWNLOAD_LOCK:
                if DOWNLOAD["active"]:
                    self._json({"error": f"another download is in progress: "
                                         f"{DOWNLOAD['repo_id']}. Wait for it to finish "
                                         f"(or click Cancel)."}, 409); return
                DOWNLOAD["active"] = True
                DOWNLOAD["key"] = key
                DOWNLOAD["repo_id"] = repo["repo_id"]
                DOWNLOAD["started_ts"] = time.time()
                DOWNLOAD["last_line"] = "starting…"
            threading.Thread(target=_download_thread, args=(repo,), daemon=True).start()
            self._json({"ok": True, "key": key, "repo_id": repo["repo_id"]}); return

        if path == "/models/cancel":
            # Best-effort kill — the next status poll will see active=False.
            with DOWNLOAD_LOCK:
                was_active = DOWNLOAD["active"]
                rid = DOWNLOAD.get("repo_id")
            if not was_active:
                self._json({"error": "no active download"}, 404); return
            _kill_active_download()
            push(f"[hf] cancel requested for {rid}")
            self._json({"ok": True}); return

        if path == "/prompt/enhance":
            # Gemma-driven prompt enhancement, routed through the warm
            # helper subprocess. First call after panel start eats a
            # ~10-15s Gemma load; cached afterwards (subsequent enhances
            # ~3-5s). Helper's release_pipelines frees Gemma when a real
            # render comes in, so memory doesn't accumulate on top of
            # the dev transformer.
            user_prompt = (form.get("prompt", [""])[0] or "").strip()
            mode = (form.get("mode", ["t2v"])[0] or "t2v").lower()
            if mode not in ("t2v", "i2v"):
                mode = "t2v"
            if not user_prompt:
                self._json({"error": "no prompt provided"}, 400); return
            push(f"[enhance] {mode}: {user_prompt[:80]}…")
            try:
                result = HELPER.run({
                    "action": "enhance_prompt",
                    "id": f"enh-{int(time.time()*1000)}",
                    "params": {"prompt": user_prompt, "mode": mode, "seed": 10},
                })
            except Exception as exc:
                push(f"[enhance] failed: {exc}")
                self._json({"error": str(exc)}, 500); return
            enhanced = result.get("enhanced", "").strip()
            if not enhanced:
                self._json({"error": "Gemma returned empty result"}, 500); return
            push(f"[enhance] → {enhanced[:120]}… ({result.get('elapsed_sec','?')}s)")
            self._json({
                "ok": True,
                "original": user_prompt,
                "enhanced": enhanced,
                "mode": mode,
                "elapsed_sec": result.get("elapsed_sec"),
            })
            return

        self.send_error(404)


def _avg_elapsed() -> float | None:
    with LOCK:
        recent = [j["elapsed_sec"] for j in STATE["history"][:10]
                  if j.get("status") == "done" and j.get("elapsed_sec")]
    if not recent:
        return None
    return round(sum(recent) / len(recent), 1)


# ---- HTML --------------------------------------------------------------------

def page() -> str:
    bootstrap = json.dumps({
        "presets": PRESETS, "aspects": ASPECTS,
        "default_image": str(REFERENCE),
        "default_audio": str(AUDIO_DEFAULT),
        "fps": FPS, "model": MODEL_ID,
    })
    return HTML.replace("__BOOTSTRAP__", bootstrap)


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Phosphene</title>
  <link rel="icon" type="image/png" sizes="64x64" href="/assets/favicon-64.png">
  <link rel="icon" type="image/png" sizes="256x256" href="/assets/favicon.png">
  <style>
    :root {
      /* Phosphene void = #00061a — the canonical dark-navy backdrop the
         brand artwork was rendered against. Body uses it as-is; elevated
         panels lift slightly toward blue-violet so the brand color story
         carries from the logo through every surface. */
      --bg: #00061a; --bg-2: #050b22; --panel: #0c1330; --panel-2: #141a3a;
      --border: #1f2547; --border-strong: #2e3658; --text: #e6edf3; --muted: #8b949e;
      --accent: #2f81f7; --accent-bright: #58a6ff; --accent-dim: rgba(47,129,247,0.18);
      --success: #3fb950; --warning: #d29922; --danger: #f85149;
      --radius: 10px;
    }
    * { box-sizing: border-box; }
    html, body { margin: 0; height: 100%; }
    body {
      background: var(--bg); color: var(--text);
      font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
      display: flex; flex-direction: column; min-height: 100vh;
    }

    /* ===== HEADER =====
       Background uses var(--bg) which is the brand-void (#00061a) — same
       color baked into the logo PNG so it blends seamlessly into the
       header. Logo height 72px so the wordmark reads. */
    header {
      display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
      padding: 12px 18px; border-bottom: 1px solid var(--border);
      background: var(--bg);
    }
    header h1 {
      margin: 0; font-size: 15px; font-weight: 700; letter-spacing: -0.01em;
      display: inline-flex; align-items: center; gap: 8px;
    }
    .brand { display: inline-flex; align-items: center; flex-shrink: 0; }
    .brand img { height: 104px; width: auto; display: block; }
    .tag { color: var(--muted); font-size: 11px; }
    .pill {
      padding: 4px 10px; border-radius: 999px; font-size: 11px; font-weight: 500;
      background: var(--panel-2); border: 1px solid var(--border); color: var(--muted);
      white-space: nowrap;
    }
    .pill .dot { display: inline-block; width: 6px; height: 6px; border-radius: 999px; margin-right: 5px; background: currentColor; vertical-align: middle; }
    .pill-good { color: var(--success); border-color: rgba(63,185,80,0.4); }
    .pill-warn { color: var(--warning); border-color: rgba(210,153,34,0.5); }
    .pill-danger { color: var(--danger); border-color: rgba(248,81,73,0.5); }
    .pill-running { color: var(--accent-bright); border-color: var(--accent); animation: pulse 1.6s ease-in-out infinite; }
    @keyframes pulse { 50% { opacity: 0.7; } }
    .spacer { flex: 1; }
    .ghost-btn {
      /* width:auto overrides the global `button { width:100% }` rule above
         so ghost buttons sit at intrinsic width inside flex rows. Without
         this, the Enhance button stretches full-width and forces the
         No-music pill to wrap to a second line. */
      width: auto;
      background: transparent; border: 1px solid var(--border); color: var(--text);
      padding: 5px 10px; border-radius: 6px; font-size: 11px; cursor: pointer;
    }
    .ghost-btn:hover { border-color: var(--accent); color: var(--accent-bright); }

    /* Toggle pill — used for binary on/off controls (e.g. "No music")
       living next to ghost-btn actions. Same height + radius family as
       ghost-btn so they line up cleanly, but visually distinct: when ON,
       fills with the accent color so it reads as an active filter. */
    .toggle-pill {
      display: inline-flex; align-items: center; gap: 6px;
      background: transparent; border: 1px solid var(--border); color: var(--muted);
      padding: 5px 10px; border-radius: 6px; font-size: 11px; cursor: pointer;
      user-select: none; transition: border-color 120ms ease, color 120ms ease, background 120ms ease;
      white-space: nowrap;
    }
    .toggle-pill:hover { border-color: var(--accent); color: var(--text); }
    .toggle-pill input[type="checkbox"] {
      position: absolute; opacity: 0; pointer-events: none; width: 0; height: 0;
    }
    .toggle-pill .toggle-dot {
      width: 8px; height: 8px; border-radius: 50%;
      background: var(--border-strong); border: 1px solid transparent;
      transition: background 120ms ease, box-shadow 120ms ease;
      flex-shrink: 0;
    }
    .toggle-pill.on {
      background: var(--accent-dim); border-color: var(--accent); color: var(--accent-bright);
    }
    .toggle-pill.on .toggle-dot {
      background: var(--accent-bright);
      box-shadow: 0 0 0 2px rgba(47,129,247,0.25);
    }
    /* "by Bizarro" link in the top-right of the header. Bumped from a
       muted 11px chip to something with actual presence — accent-color
       border ring + bold text + subtle lift on hover. The avatar is a
       hand-drawn portrait so deserves to read at a glance. */
    .creator-link {
      display: inline-flex; align-items: center; gap: 9px;
      color: var(--fg, #d8e0ee); font-size: 12px; font-weight: 500;
      text-decoration: none;
      padding: 4px 12px 4px 4px; border-radius: 999px;
      border: 1.5px solid var(--accent, #5a7cff);
      background: linear-gradient(135deg, rgba(90,124,255,0.10), rgba(90,124,255,0.02));
      transition: transform 0.15s ease, border-color 0.15s ease, background 0.15s ease, box-shadow 0.15s ease;
      box-shadow: 0 0 0 1px rgba(90,124,255,0.18), 0 2px 8px rgba(90,124,255,0.10);
    }
    .creator-link:hover {
      color: var(--accent-bright, #93a8ff);
      border-color: var(--accent-bright, #93a8ff);
      transform: translateY(-1px);
      box-shadow: 0 0 0 1px rgba(147,168,255,0.30), 0 4px 14px rgba(90,124,255,0.25);
    }
    .creator-avatar {
      width: 28px; height: 28px; border-radius: 50%;
      object-fit: cover; display: block;
      box-shadow: 0 0 0 2px var(--accent, #5a7cff), 0 0 0 3px var(--bg, #0d1017);
    }
    .creator-link .x-icon {
      font-size: 10px; opacity: 0.6; margin-left: -2px;
      letter-spacing: 0.05em;
    }

    /* Inline models card — sits at the top of the form, where users
       actually look. Four states drawn from /status:
         missing-base    big red CTA, blocks generation
         missing-current big amber CTA when the SELECTED mode needs Q8
         downloading     green progress bar with last hf line
         hidden          (everything ready, no urgency — link in header)
       See updateModelsCard() in the JS. */
    .models-inline {
      border-radius: 10px; padding: 12px 14px; margin-bottom: 14px;
      border: 1.5px solid var(--border, #2a3140);
      background: var(--panel-2, #131724);
      transition: border-color 150ms ease, background 150ms ease;
      position: relative;
    }
    /* Dismiss × — only rendered (visible) in dismissible states (warn /
       partial-ok). The download-active and base-missing states omit it
       so users can't accidentally hide a hard blocker. */
    .models-inline-dismiss {
      position: absolute; top: 6px; right: 8px;
      width: 22px; height: 22px; border-radius: 4px;
      background: transparent; border: 1px solid transparent;
      color: var(--muted); font-size: 14px; line-height: 1;
      cursor: pointer; padding: 0;
      display: none;
    }
    .models-inline-dismiss:hover { color: var(--text); border-color: var(--border); }
    .models-inline.dismissible .models-inline-dismiss { display: inline-block; }
    .models-inline.state-missing  { border-color: rgba(220,80,80,0.55); background: rgba(220,80,80,0.06); }
    .models-inline.state-warn     { border-color: rgba(210,153,34,0.55); background: rgba(210,153,34,0.06); }
    .models-inline.state-downloading {
      border-color: var(--accent, #5a7cff);
      background: rgba(90,124,255,0.07);
      animation: keyframes-modeldl 1.6s ease-in-out infinite;
    }
    .models-inline-body {
      display: grid;
      grid-template-columns: 28px 1fr auto;
      gap: 12px; align-items: center;
    }
    .models-inline-icon { font-size: 22px; line-height: 1; text-align: center; }
    .models-inline-text { min-width: 0; }
    .models-inline-text .ttl { font-weight: 600; font-size: 13px; color: var(--fg, #d8e0ee); }
    .models-inline-text .sub { font-size: 11px; color: var(--muted); margin-top: 2px; }
    .models-inline-progress { margin-top: 8px; }
    .models-inline-progress .bar {
      height: 6px; border-radius: 3px; background: rgba(255,255,255,0.06);
      overflow: hidden;
    }
    .models-inline-progress .bar .fill {
      height: 100%; background: var(--accent, #5a7cff);
      transition: width 250ms ease;
    }
    .models-inline-progress .last {
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 10px; color: var(--muted); margin-top: 4px;
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .models-inline-actions button {
      padding: 7px 14px; border-radius: 7px;
      background: var(--accent, #5a7cff); color: white; border: none;
      font-size: 12px; font-weight: 600; cursor: pointer;
      white-space: nowrap;
    }
    .models-inline-actions button:hover { background: var(--accent-bright, #7e98ff); }
    .models-inline-actions button.danger { background: rgba(220,80,80,0.7); }
    .models-inline-actions button.ghost {
      background: transparent; color: var(--muted);
      border: 1px solid var(--border, #2a3140); font-weight: 500;
    }
    .models-inline-link {
      display: block; margin-top: 10px;
      font-size: 11px; color: var(--muted); text-align: right;
      cursor: pointer; text-decoration: none;
    }
    .models-inline-link:hover { color: var(--accent-bright, #93a8ff); }

    /* ===== MAIN LAYOUT ===== */
    .layout {
      flex: 1 1 auto; display: grid; grid-template-columns: 440px 1fr;
      gap: 14px; padding: 14px; min-height: 0;
    }
    .form-pane, .stage-pane {
      background: var(--panel); border: 1px solid var(--border);
      border-radius: var(--radius); overflow: hidden;
      display: flex; flex-direction: column; min-height: 0;
    }
    .form-pane { padding: 16px; overflow-y: auto; }
    .stage-pane { padding: 0; }

    /* ===== FORM ===== */
    h2 {
      font-size: 10px; margin: 14px 0 8px; color: var(--muted);
      text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600;
    }
    h2:first-child { margin-top: 0; }
    label.lbl {
      display: block; margin: 10px 0 4px; color: var(--muted);
      font-size: 10px; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600;
    }
    input, textarea, select, button {
      width: 100%; padding: 8px 11px; font: inherit; color: inherit;
      background: var(--panel-2); border: 1px solid var(--border); border-radius: 6px;
    }
    input:focus, textarea:focus, select:focus { outline: none; border-color: var(--accent); background: var(--bg-2); }
    textarea { min-height: 84px; resize: vertical; font-family: inherit; }

    /* Pill button groups (mode/quality/aspect) */
    .pill-group {
      display: grid; gap: 6px; margin-bottom: 6px;
    }
    .pill-group.cols-2 { grid-template-columns: 1fr 1fr; }
    .pill-group.cols-3 { grid-template-columns: 1fr 1fr 1fr; }
    .pill-group.cols-4 { grid-template-columns: 1fr 1fr 1fr 1fr; }
    .pill-btn {
      width: 100%; padding: 9px 8px; border-radius: 8px;
      background: var(--panel-2); border: 1px solid var(--border); color: var(--muted);
      cursor: pointer; transition: 0.12s; text-align: center;
      font-size: 12px; font-weight: 500;
      display: flex; flex-direction: column; align-items: center; gap: 2px;
    }
    .pill-btn:hover { border-color: var(--accent); color: var(--text); }
    .pill-btn.active {
      background: var(--accent-dim); border-color: var(--accent);
      color: var(--accent-bright); font-weight: 600;
    }
    .pill-btn .ico { font-size: 16px; line-height: 1; }
    .pill-btn .sub { font-size: 10px; color: var(--muted); margin-top: 1px; }
    .pill-btn:disabled, .pill-btn.disabled {
      opacity: 0.45; cursor: not-allowed; pointer-events: none;
    }

    /* Mode-specific blocks (image/audio/extend sections) */
    .mode-only { display: none; }
    .mode-only.show { display: block; }

    /* Image preview */
    .img-row { display: flex; gap: 6px; align-items: center; margin-top: 6px; }
    .img-row input[type="file"] { display: none; }
    .img-preview {
      display: none; margin-top: 8px; max-width: 100%;
      border-radius: 8px; border: 1px solid var(--border);
    }
    .img-preview.show { display: block; }

    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .row3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
    .check { display: flex; align-items: center; gap: 8px; margin: 10px 0; cursor: pointer; color: var(--text); font-size: 12px; }
    .check input { width: auto; margin: 0; }
    .hint { color: var(--muted); font-size: 10px; margin-top: 4px; }
    .derived {
      margin-top: 8px; padding: 9px 11px; border-radius: 6px;
      background: var(--panel-2); border: 1px solid var(--border);
      font-size: 12px; color: var(--muted);
    }
    .derived strong { color: var(--accent-bright); font-weight: 600; }
    .warn-banner {
      background: rgba(210,153,34,0.08); border: 1px solid rgba(210,153,34,0.4);
      color: var(--warning); padding: 8px 11px; border-radius: 6px; margin: 8px 0;
      font-size: 11px; display: none;
    }
    .warn-banner.show { display: block; }

    button.primary {
      background: var(--accent); border-color: var(--accent); color: white;
      padding: 11px; font-size: 14px; font-weight: 600; cursor: pointer;
    }
    button.primary:hover { background: var(--accent-bright); border-color: var(--accent-bright); }
    button.primary:disabled { opacity: 0.55; cursor: not-allowed; }
    button.danger {
      color: var(--danger); border-color: rgba(248,81,73,0.4); background: transparent;
      padding: 11px; font-weight: 600; cursor: pointer;
    }
    button.danger:hover { background: rgba(248,81,73,0.1); }
    .actions { display: grid; grid-template-columns: 2fr 1fr; gap: 8px; margin-top: 14px; }
    button.small {
      width: auto; padding: 6px 10px; font-size: 11px;
      background: var(--panel-2); font-weight: 500; cursor: pointer;
    }
    button.small:hover { border-color: var(--accent); }

    details summary { cursor: pointer; color: var(--muted); font-size: 11px; padding: 4px 0; }
    details summary:hover { color: var(--text); }
    details[open] summary { margin-bottom: 6px; }

    /* ===== STAGE (PLAYER + CAROUSEL) ===== */
    .stage-pane {
      background: linear-gradient(180deg, #0a0d12 0%, var(--panel) 100%);
    }
    .player-wrap {
      flex: 1 1 auto; display: flex; align-items: center; justify-content: center;
      background: black; min-height: 0; position: relative;
      overflow: hidden;
    }
    .player-wrap video {
      max-width: 100%; max-height: 100%; width: auto; height: auto;
      display: block;
    }
    .player-wrap.empty {
      background: var(--panel-2); color: var(--muted); font-size: 13px;
      flex-direction: column; gap: 6px; text-align: center;
    }
    .player-wrap.empty .dim-icon { font-size: 36px; opacity: 0.4; }
    .player-meta {
      flex: 0 0 auto; padding: 10px 16px;
      background: var(--panel); border-top: 1px solid var(--border);
      display: flex; justify-content: space-between; align-items: center; gap: 10px;
      font-size: 12px; color: var(--muted);
    }
    .player-meta .name { color: var(--text); font-weight: 500; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .player-meta .actions-bar { display: flex; gap: 6px; flex-shrink: 0; }
    .player-meta button { width: auto; padding: 5px 10px; font-size: 11px; background: transparent; border: 1px solid var(--border); color: var(--muted); cursor: pointer; border-radius: 6px; }
    .player-meta button:hover { color: var(--text); border-color: var(--accent); }

    /* Carousel */
    .carousel-wrap {
      flex: 0 0 auto; padding: 10px 14px 14px;
      background: var(--panel); border-top: 1px solid var(--border);
    }
    .carousel-head {
      display: flex; justify-content: space-between; align-items: center;
      margin-bottom: 8px; gap: 10px;
    }
    .carousel-head h3 {
      margin: 0; font-size: 11px; color: var(--muted);
      text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600;
    }
    .carousel-head .seg {
      display: inline-flex; border: 1px solid var(--border); border-radius: 6px; overflow: hidden;
    }
    .carousel-head .seg button {
      width: auto; padding: 4px 10px; font-size: 11px; background: transparent;
      border: 0; border-right: 1px solid var(--border); border-radius: 0;
      color: var(--muted); font-weight: 500; cursor: pointer;
    }
    .carousel-head .seg button:last-child { border-right: 0; }
    .carousel-head .seg button.active { background: var(--accent); color: white; }
    .carousel {
      display: flex; gap: 8px; overflow-x: auto; padding-bottom: 4px;
      scroll-snap-type: x proximity;
    }
    .carousel::-webkit-scrollbar { height: 8px; }
    .carousel::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 4px; }
    .car-card {
      flex: 0 0 168px; scroll-snap-align: start;
      border: 1px solid var(--border); border-radius: 8px; overflow: hidden;
      background: var(--panel-2); cursor: pointer; transition: 0.12s;
      display: flex; flex-direction: column;
    }
    .car-card:hover { border-color: var(--accent); transform: translateY(-1px); }
    .car-card.active { border-color: var(--accent-bright); box-shadow: 0 0 0 1px var(--accent-bright); }
    .car-card.hidden-card { opacity: 0.4; }
    .car-card video { width: 100%; aspect-ratio: 16/9; object-fit: cover; background: black; display: block; }
    .car-card .info { padding: 6px 8px; font-size: 10px; }
    .car-card .name { color: var(--text); font-weight: 500; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .car-card .sub { color: var(--muted); margin-top: 2px; }
    .car-card .row-btns { display: flex; gap: 4px; padding: 0 6px 6px; }
    .car-card .row-btns button {
      flex: 1; padding: 3px 4px; font-size: 9px; background: transparent;
      border: 1px solid var(--border); color: var(--muted); cursor: pointer; border-radius: 4px;
    }
    .car-card .row-btns button:hover { color: var(--text); border-color: var(--accent); }
    .empty-msg { color: var(--muted); font-size: 12px; padding: 12px; text-align: center; width: 100%; }

    /* ===== BOTTOM TABBED PANE ===== */
    .bottom-pane {
      flex: 0 0 auto; max-height: 280px;
      background: var(--panel); border-top: 1px solid var(--border);
      display: flex; flex-direction: column; min-height: 0;
    }
    .bottom-pane.collapsed .bottom-body { display: none; }
    .bottom-pane.collapsed { max-height: 38px; }
    .tabs {
      display: flex; align-items: center; gap: 0; padding: 0 12px;
      border-bottom: 1px solid var(--border); flex-shrink: 0;
    }
    .tabs button {
      width: auto; padding: 9px 14px; font-size: 12px; background: transparent;
      border: 0; border-bottom: 2px solid transparent; border-radius: 0;
      color: var(--muted); font-weight: 500; cursor: pointer;
      display: inline-flex; align-items: center; gap: 6px;
    }
    .tabs button:hover { color: var(--text); }
    .tabs button.active { color: var(--accent-bright); border-bottom-color: var(--accent); }
    .tabs button .badge {
      background: var(--accent-dim); color: var(--accent-bright);
      padding: 1px 6px; border-radius: 999px; font-size: 10px; font-weight: 600;
    }
    .tabs .spacer { flex: 1; }
    .tabs .model-credit {
      font-size: 11px; color: var(--muted); padding: 0 12px;
      text-decoration: none; border-right: 1px solid var(--border);
      align-self: stretch; display: inline-flex; align-items: center;
      transition: 0.12s;
    }
    .tabs .model-credit:hover { color: var(--accent-bright); }
    .tabs .model-credit::after { content: " ↗"; opacity: 0.6; margin-left: 4px; }
    .tabs .tab-collapse {
      width: auto; padding: 4px 8px; font-size: 11px; background: transparent;
      border: 1px solid var(--border); color: var(--muted); cursor: pointer; border-radius: 6px;
      align-self: center;
    }
    .bottom-body {
      flex: 1 1 auto; overflow-y: auto; padding: 12px 16px; min-height: 0;
    }
    .tab-content { display: none; }
    .tab-content.show { display: block; }

    /* Now panel */
    .now-card {
      padding: 10px 12px; border-radius: 8px; background: var(--panel-2);
      border: 1px solid var(--border);
    }
    .now-card.idle { opacity: 0.7; }
    /* Failed-job state — kept loud instead of letting the panel drift back
       to a sleepy "Idle". Border + background tint match the danger color
       so the eye lands on it immediately. Stays visible until the user
       submits a new job (next render flips us out of failed state). */
    .now-card.failed {
      border-color: rgba(248,81,73,0.55);
      background: rgba(248,81,73,0.06);
      opacity: 1;
    }
    .now-card .ttl { font-weight: 600; font-size: 13px; }
    .now-card .meta { margin-top: 6px; font-size: 11px; color: var(--muted); }
    .progress-bar { height: 5px; background: var(--border); border-radius: 3px; overflow: hidden; margin: 7px 0; }
    .progress-bar .fill { height: 100%; background: var(--accent); transition: width 0.3s; }

    /* Queue/recent lists */
    .row-list { list-style: none; padding: 0; margin: 0; }
    .row-list li {
      display: grid; grid-template-columns: auto 1fr auto auto; gap: 10px;
      align-items: center; padding: 7px 9px; border-radius: 6px;
      border: 1px solid var(--border); background: var(--panel-2);
      margin-bottom: 5px; font-size: 11px;
    }
    .row-list li .pos { color: var(--muted); font-weight: 600; min-width: 22px; }
    .row-list li .ttl, .row-list li .params {
      overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }
    .row-list li .params { color: var(--muted); font-size: 10px; }
    /* Inline error text in failed history rows. Shown in the title slot
       so users see the cause without reading the log. */
    .row-list li .err-inline { color: var(--danger, #f85149); font-weight: 500; }
    .row-list li .badge {
      font-size: 9px; text-transform: uppercase; letter-spacing: 0.08em;
      padding: 2px 7px; border-radius: 999px; border: 1px solid currentColor;
      font-weight: 600;
    }
    .row-list li.done .badge { color: var(--success); }
    .row-list li.failed .badge { color: var(--danger); }
    .row-list li.cancelled .badge { color: var(--muted); }
    .row-list li button {
      width: auto; background: transparent; border: 0; color: var(--muted);
      cursor: pointer; padding: 2px 6px; font-size: 14px; line-height: 1;
    }
    .row-list li button:hover { color: var(--danger); }
    .row-actions { display: flex; gap: 6px; margin-top: 8px; flex-wrap: wrap; }

    /* Logs */
    pre.log {
      white-space: pre-wrap; word-break: break-all;
      background: #06080c; border: 1px solid var(--border); border-radius: 6px;
      padding: 10px; font: 11px/1.5 ui-monospace, "SF Mono", Menlo, monospace;
      color: #b0b8c4; margin: 0; max-height: 220px; overflow-y: auto;
    }

    /* Modal */
    .modal-bg {
      position: fixed; inset: 0; background: rgba(0,0,0,0.6); z-index: 100;
      display: none; align-items: center; justify-content: center;
    }
    .modal-bg.show { display: flex; }
    .modal {
      background: var(--panel); border: 1px solid var(--border-strong);
      border-radius: var(--radius); padding: 18px; width: min(640px, 92vw);
      max-height: 80vh; overflow: hidden; display: flex; flex-direction: column;
    }
    .modal h3 { margin: 0 0 12px; font-size: 14px; }
    .modal textarea.batch {
      flex: 1 1 auto; min-height: 280px; font: 11px/1.5 ui-monospace, "SF Mono", Menlo, monospace;
    }
    .modal-actions { display: flex; gap: 8px; justify-content: flex-end; margin-top: 12px; }

    /* Empty state */
    .empty-state { color: var(--muted); font-size: 12px; padding: 14px 0; text-align: center; }

    /* ----- Image picker (FFLF + I2V) -----
       Replaces the old `<input>` with a path placeholder that confused
       non-technical users. Tile is the entire interaction surface: click
       opens the file dialog, drop receives the file, and below the tile
       is a thumbnail strip of recent uploads for one-click reuse. */
    .picker { margin-bottom: 14px; }
    .picker-drop {
      position: relative;
      border: 1.5px dashed var(--border, #2a3140);
      border-radius: 10px;
      min-height: 130px;
      display: flex; align-items: center; justify-content: center;
      cursor: pointer;
      background: rgba(255,255,255,0.015);
      transition: border-color 120ms ease, background 120ms ease;
      overflow: hidden;
    }
    .picker-drop:hover { border-color: var(--accent, #5a7cff); background: rgba(90,124,255,0.04); }
    .picker-drop.dragover {
      border-color: var(--accent-bright, #7e98ff);
      background: rgba(90,124,255,0.10);
      border-style: solid;
    }
    .picker-drop.has-image { border-style: solid; cursor: zoom-in; min-height: 0; }
    .picker-empty {
      text-align: center; padding: 18px 14px; color: var(--muted);
      pointer-events: none;
    }
    .picker-icon { font-size: 28px; margin-bottom: 6px; line-height: 1; }
    .picker-cta { font-size: 13px; color: var(--fg, #d8e0ee); margin-bottom: 4px; }
    .picker-empty .hint { font-size: 11px; color: var(--muted); }
    .picker-preview {
      max-width: 100%; max-height: 240px;
      display: block; margin: 0 auto;
      border-radius: 8px;
    }
    .picker-clear {
      position: absolute; top: 6px; right: 6px;
      width: 28px; height: 28px; border-radius: 50%;
      border: 1px solid var(--border, #2a3140);
      background: rgba(0,0,0,0.6); color: var(--fg, #d8e0ee);
      cursor: pointer; font-size: 16px; line-height: 1;
      display: flex; align-items: center; justify-content: center;
    }
    .picker-clear:hover { background: rgba(220,80,80,0.45); border-color: rgba(220,80,80,0.6); }
    .picker-uploading {
      position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
      background: rgba(0,0,0,0.55); border-radius: 8px;
      color: var(--fg, #d8e0ee); font-size: 12px;
    }
    .picker-recent { margin-top: 8px; }
    .picker-recent-label {
      font-size: 11px; color: var(--muted); margin-bottom: 4px;
    }
    .picker-recent-strip {
      display: flex; gap: 6px; overflow-x: auto; padding-bottom: 4px;
      scrollbar-width: thin;
    }
    .picker-recent-strip::-webkit-scrollbar { height: 6px; }
    .picker-recent-strip::-webkit-scrollbar-thumb { background: var(--border, #2a3140); border-radius: 3px; }
    .picker-recent-thumb {
      flex: 0 0 auto; width: 56px; height: 56px;
      border-radius: 6px; border: 2px solid transparent;
      object-fit: cover; cursor: pointer;
      background: rgba(255,255,255,0.04);
      transition: border-color 120ms ease, transform 120ms ease;
    }
    .picker-recent-thumb:hover { border-color: var(--accent, #5a7cff); transform: translateY(-1px); }
    .picker-recent-thumb.selected { border-color: var(--success, #3fb950); }

    /* LoRA picker — collapsible <details>. Earlier version was a tight
       4-column grid (checkbox | name | slider | × ); user feedback was
       "this layout sucks balls" — too cramped, no preview, trigger words
       hard to spot, no obvious link back to the source. Rebuilt as a
       card stack: each LoRA gets a video/image preview, name, clickable
       trigger word chips that append to the prompt, a full-width
       strength slider, and a clear Active toggle. */
    .loras-summary {
      cursor: pointer; user-select: none; font-size: 12px;
      font-weight: 600; color: var(--text);
      display: flex; align-items: center; gap: 8px;
      padding: 6px 0;
    }
    .loras-summary .hint { font-weight: 400; }
    .loras-list {
      display: flex; flex-direction: column; gap: 10px; margin-top: 8px;
    }
    .lora-card {
      position: relative;
      border-radius: 9px;
      border: 1px solid var(--border); background: var(--panel-2);
      overflow: hidden;
      transition: border-color 120ms ease, background 120ms ease;
    }
    .lora-card.active {
      border-color: var(--accent);
      background: var(--accent-dim, rgba(47,129,247,0.07));
    }
    .lora-card .lora-thumb-wrap {
      position: relative;
      width: 100%; aspect-ratio: 16/9; background: var(--bg-2, #0a0c14);
      overflow: hidden;
    }
    .lora-card .lora-thumb {
      width: 100%; height: 100%; object-fit: cover; display: block;
    }
    .lora-card .lora-thumb-empty {
      width: 100%; height: 100%;
      display: flex; align-items: center; justify-content: center;
      color: var(--muted); font-size: 11px;
      background: linear-gradient(135deg, rgba(255,255,255,0.02), rgba(255,255,255,0.04));
    }
    .lora-card .lora-corner-actions {
      position: absolute; top: 6px; right: 6px;
      display: flex; gap: 4px; z-index: 2;
    }
    .lora-card .lora-corner-btn {
      width: 28px; height: 28px; padding: 0;
      border-radius: 6px; border: 1px solid rgba(0,0,0,0.4);
      background: rgba(15,18,28,0.78); backdrop-filter: blur(4px);
      color: rgba(255,255,255,0.85); font-size: 13px; line-height: 1;
      cursor: pointer; display: inline-flex; align-items: center;
      justify-content: center; text-decoration: none;
    }
    .lora-card .lora-corner-btn:hover { background: rgba(20,25,40,0.9); color: #fff; }
    .lora-card .lora-corner-btn.danger:hover {
      color: #ff8a8a; border-color: rgba(220,80,80,0.5);
    }
    .lora-card .lora-body { padding: 10px 12px 12px; }
    .lora-card .lora-name {
      font-size: 13px; font-weight: 600; color: var(--text);
      overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
      margin-bottom: 6px;
    }
    .lora-card .lora-name .badge {
      display: inline-block; font-size: 9px; font-weight: 600;
      letter-spacing: 0.05em; text-transform: uppercase;
      padding: 1px 6px; border-radius: 999px; margin-left: 6px;
      border: 1px solid var(--accent); color: var(--accent-bright);
      vertical-align: middle;
    }
    .lora-card .trigger-chips {
      display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 8px;
    }
    .lora-card .trigger-chip {
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 10.5px; padding: 3px 8px; border-radius: 999px;
      background: rgba(255,255,255,0.05);
      border: 1px solid var(--border);
      color: var(--text);
      cursor: pointer; user-select: none;
      transition: background 100ms ease, border-color 100ms ease;
    }
    .lora-card .trigger-chip:hover {
      background: rgba(90,124,255,0.15);
      border-color: var(--accent);
    }
    .lora-card .trigger-chip.empty {
      color: var(--muted); font-style: italic;
      cursor: default; background: transparent; border: none;
      padding: 0; font-family: inherit;
    }
    .lora-card .lora-strength-row {
      display: flex; align-items: center; gap: 8px;
      margin-bottom: 8px;
    }
    .lora-card .lora-strength-row label {
      font-size: 10px; text-transform: uppercase; letter-spacing: 0.05em;
      color: var(--muted); width: 56px; flex: none;
    }
    .lora-card .lora-strength-row input[type="range"] {
      flex: 1; min-width: 0; accent-color: var(--accent);
    }
    .lora-card .lora-strength-row input[type="number"] {
      width: 56px; padding: 3px 6px; font-size: 11px;
      text-align: right;
    }
    .lora-card .lora-toggle-row {
      display: flex; align-items: center; gap: 8px;
    }
    .lora-card .lora-toggle {
      display: inline-flex; align-items: center; gap: 6px;
      padding: 5px 10px; border-radius: 6px;
      border: 1px solid var(--border); background: rgba(255,255,255,0.02);
      font-size: 11px; font-weight: 600; color: var(--muted);
      cursor: pointer; user-select: none;
    }
    .lora-card.active .lora-toggle {
      background: var(--accent, #5a7cff); color: #fff;
      border-color: var(--accent, #5a7cff);
    }
    .lora-card .lora-toggle input { display: none; }
    .lora-card .lora-toggle .dot {
      width: 8px; height: 8px; border-radius: 50%;
      background: var(--muted); transition: background 100ms;
    }
    .lora-card.active .lora-toggle .dot { background: #fff; }
    .lora-card .lora-meta-link {
      font-size: 11px; color: var(--muted); margin-left: auto;
      text-decoration: none;
    }
    .lora-card .lora-meta-link:hover { color: var(--accent-bright, #93a8ff); }
    /* Brief flash on the prompt textarea when a trigger chip is clicked
       but the word is already present — visual ack that the click did
       fire (we just chose not to duplicate). */
    textarea.flash-ok {
      box-shadow: 0 0 0 2px rgba(80,200,120,0.45);
      transition: box-shadow 240ms ease;
    }

    /* CivitAI browser modal — grid of cards with thumbnail, name,
       creator, downloads, NSFW indicator, and an Install button per
       card. Layered on the .models-modal scaffold. */
    .civitai-search-bar {
      display: flex; gap: 8px; margin-bottom: 12px; align-items: center;
    }
    .civitai-search-bar input[type="text"] { flex: 1; }
    .civitai-grid {
      display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
      gap: 12px; margin-top: 10px;
    }
    .civitai-card {
      border: 1px solid var(--border); border-radius: 8px;
      background: var(--panel-2); overflow: hidden;
      display: flex; flex-direction: column;
    }
    .civitai-card .preview {
      width: 100%; aspect-ratio: 16/10; background: var(--bg-2);
      object-fit: cover; display: block;
    }
    .civitai-card .preview-empty {
      width: 100%; aspect-ratio: 16/10; background: var(--bg-2);
      display: flex; align-items: center; justify-content: center;
      color: var(--muted); font-size: 11px;
    }
    .civitai-card .body { padding: 10px 12px; flex: 1;
      display: flex; flex-direction: column; gap: 4px; }
    .civitai-card .ttl {
      font-size: 13px; font-weight: 600; color: var(--text);
      overflow: hidden; text-overflow: ellipsis;
      display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
    }
    .civitai-card .meta {
      font-size: 10px; color: var(--muted);
      display: flex; gap: 10px; flex-wrap: wrap;
    }
    .civitai-card .nsfw-badge {
      display: inline-block; font-size: 9px; font-weight: 700;
      letter-spacing: 0.05em; padding: 1px 6px; border-radius: 999px;
      background: rgba(248,81,73,0.15); border: 1px solid var(--danger, #f85149);
      color: var(--danger, #f85149);
    }
    .civitai-card .civitai-source-link {
      color: var(--accent-bright, #7e98ff);
      text-decoration: none; font-size: 11px;
    }
    .civitai-card .civitai-source-link:hover { text-decoration: underline; }
    .civitai-card .actions { padding: 0 12px 12px; }
    .civitai-card .actions button { width: 100%; }
    .civitai-status-line { color: var(--muted); font-size: 11px; margin-top: 12px; }
    .civitai-status-line.err { color: var(--danger, #f85149); }
    .civitai-status-line.ok { color: var(--success, #3fb950); }

    /* Header icon button — same height/feel as ghost-btn but icon-only.
       Used for the settings cog. width:auto overrides the global
       button{width:100%} rule. */
    .icon-btn {
      width: auto;
      background: transparent; border: 1px solid var(--border);
      color: var(--muted); padding: 4px 7px; border-radius: 6px;
      cursor: pointer; display: inline-flex; align-items: center;
      transition: border-color 120ms ease, color 120ms ease;
    }
    .icon-btn:hover { border-color: var(--accent); color: var(--accent-bright); }
    .icon-btn svg { display: block; }

    /* Settings modal — preset pills, advanced controls, Apply button.
       Reuses the .models-modal frame so spacing/elevation match. */
    .preset-grid {
      display: grid; grid-template-columns: 1fr; gap: 8px;
      margin: 12px 0;
    }
    .preset-card {
      display: flex; align-items: flex-start; gap: 10px;
      border: 1.5px solid var(--border); border-radius: 8px;
      padding: 10px 12px; cursor: pointer; user-select: none;
      transition: border-color 120ms ease, background 120ms ease;
      background: var(--panel-2);
    }
    .preset-card:hover { border-color: var(--accent); }
    .preset-card.active {
      border-color: var(--accent);
      background: var(--accent-dim, rgba(47,129,247,0.18));
    }
    .preset-card input[type="radio"] {
      width: auto; margin: 4px 0 0 0; flex-shrink: 0;
      accent-color: var(--accent);
    }
    .preset-card .preset-text { flex: 1; min-width: 0; }
    .preset-card .preset-label {
      font-size: 13px; font-weight: 600; color: var(--text);
      margin: 0 0 4px 0;
    }
    .preset-card .preset-blurb {
      font-size: 11px; color: var(--muted); line-height: 1.4;
    }
    .preset-card .preset-spec {
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 10px; color: var(--accent-bright);
      margin-top: 4px; letter-spacing: 0.02em;
    }
    .settings-section { margin-top: 14px; }
    .settings-section h3 {
      font-size: 11px; font-weight: 600; text-transform: uppercase;
      letter-spacing: 0.08em; color: var(--muted); margin: 0 0 8px 0;
    }
    .settings-row { display: flex; gap: 10px; align-items: center; }
    .settings-row label {
      font-size: 11px; color: var(--muted); min-width: 70px;
    }
    /* Token rows in the Settings modal. The status pill on the right
       of the label tells the user at a glance whether a key is
       configured (green ✓) or missing (muted —). The input is masked
       by default; a 'show' button reveals so the user can confirm a
       paste landed correctly. Cleared by a separate button so accidental
       Apply with empty input doesn't wipe a saved key (we treat empty
       input as 'don't change' — actual removal is explicit). */
    .token-row { margin-bottom: 14px; }
    .token-row .token-label {
      display: flex; align-items: center; gap: 8px;
      font-size: 12px; font-weight: 500; color: var(--text);
      margin-bottom: 4px;
    }
    .token-row .token-status {
      font-size: 10px; font-weight: 600;
      padding: 1px 7px; border-radius: 999px;
      border: 1px solid var(--border); color: var(--muted);
      letter-spacing: 0.05em; text-transform: uppercase;
    }
    .token-row .token-status.set {
      color: var(--success, #3fb950);
      border-color: var(--success, #3fb950);
    }
    .token-row .token-status.dirty {
      color: var(--warning, #d29922);
      border-color: var(--warning, #d29922);
    }
    .token-row .token-row-input {
      display: flex; gap: 6px; align-items: center;
    }
    .token-row .token-row-input input {
      flex: 1; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 12px; padding: 5px 9px;
    }
    .token-row .token-row-input button { flex-shrink: 0; }

    .settings-foot {
      display: flex; gap: 10px; justify-content: flex-end;
      margin-top: 18px; padding-top: 14px;
      border-top: 1px solid var(--border);
    }
    .settings-status { color: var(--muted); font-size: 11px; flex: 1; align-self: center; }
    .settings-status.ok { color: var(--success, #3fb950); }
    .settings-status.err { color: var(--danger, #f85149); }
    button.primary-btn {
      width: auto;
      background: var(--accent); color: white; border: 1px solid var(--accent);
      padding: 6px 14px; border-radius: 6px; font-size: 12px; font-weight: 500;
      cursor: pointer;
    }
    button.primary-btn:hover { background: var(--accent-bright); }
    button.primary-btn:disabled { opacity: 0.5; cursor: not-allowed; }

    /* Models modal — opened by the header `models` pill. Layered on top
       of everything; the form/log keep working underneath while a
       download streams in (via the existing log panel at the bottom). */
    .models-modal {
      position: fixed; inset: 0; background: rgba(0,0,0,0.55); z-index: 100;
      display: flex; align-items: center; justify-content: center;
    }
    .models-card {
      background: var(--bg-elevated, #1a1f29); color: var(--fg, #d8e0ee);
      border: 1px solid var(--border, #2a3140); border-radius: 12px;
      width: min(640px, 92vw); max-height: 86vh; overflow-y: auto;
      box-shadow: 0 20px 60px rgba(0,0,0,0.6);
      padding: 22px 24px;
    }
    .models-head {
      display: flex; align-items: center; justify-content: space-between;
      margin-bottom: 6px;
    }
    .models-head h2 { margin: 0; font-size: 18px; font-weight: 600; }
    .models-hint { font-size: 12px; color: var(--muted); margin-bottom: 14px; }
    .models-list { list-style: none; padding: 0; margin: 0; display: flex; flex-direction: column; gap: 10px; }
    .models-list li {
      display: grid; grid-template-columns: 24px 1fr auto; gap: 10px;
      align-items: center;
      background: rgba(255,255,255,0.02); border: 1px solid var(--border, #2a3140);
      border-radius: 8px; padding: 10px 12px;
    }
    .models-list li .icon { font-size: 18px; line-height: 1; text-align: center; }
    .models-list li .meta { display: flex; flex-direction: column; gap: 2px; min-width: 0; }
    .models-list li .meta .ttl { font-weight: 500; font-size: 13px; }
    .models-list li .meta .sub { font-size: 11px; color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .models-list li .progress {
      font-size: 11px; color: var(--muted); margin-top: 4px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      max-width: 100%; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .models-list li.ready    { border-color: rgba(63,185,80,0.35); }
    .models-list li.partial  { border-color: rgba(210,153,34,0.45); }
    .models-list li.missing  { border-color: rgba(220,80,80,0.35); }
    .models-list li.downloading { border-color: var(--accent, #5a7cff); animation: keyframes-modeldl 1.6s ease-in-out infinite; }
    @keyframes keyframes-modeldl { 50% { box-shadow: 0 0 0 3px rgba(90,124,255,0.18); } }
    .models-list li button {
      padding: 6px 12px; font-size: 12px; border-radius: 6px;
      background: var(--accent, #5a7cff); color: white; border: none; cursor: pointer;
    }
    .models-list li button[disabled] { opacity: 0.55; cursor: not-allowed; }
    .models-list li button.ghost {
      background: transparent; color: var(--muted); border: 1px solid var(--border, #2a3140);
    }
    .models-foot { font-size: 11px; color: var(--muted); margin-top: 14px; line-height: 1.4; }
  </style>
</head>
<body>

<header>
  <a href="/" class="brand"><img src="/assets/logo-header.png" alt="Phosphene"></a>
  <span class="spacer"></span>
  <!-- Hardware tier badge — clickable, opens a dialog explaining what
       this Mac's RAM tier allows. Modes / qualities the tier doesn't
       support are visibly disabled in the form below. -->
  <span id="tierPill" class="pill" style="cursor:pointer" onclick="openTierModal()" title="Click to see what this Mac can do">click for system info</span>
  <span id="memPill" class="pill">memory…</span>
  <span id="comfyPill" class="pill" style="display:none">comfy…</span>
  <span id="helperPill" class="pill">helper…</span>
  <!-- Models pill: shows roll-up status (X / Y models ready) and click-opens
       a modal with per-repo download buttons. Color reflects readiness:
       pill-good = all models present, pill-warn = base ready but Q8 missing,
       pill-bad = base incomplete (panel can't render). -->
  <span id="modelsPill" class="pill" style="cursor:pointer" onclick="openModelsModal()" title="View / download model status">models…</span>
  <span id="queuePill" class="pill">queue 0</span>
  <span id="jobPill" class="pill">idle</span>
  <!-- Settings cog: opens the modal that lets users pick output codec
       presets (Standard / Video production / Web / Custom). Sits between the
       runtime pills and Stop Comfy so it's findable but not loud. -->
  <button id="settingsBtn" class="icon-btn" onclick="openSettingsModal()" title="Output settings — codec, file size">
    <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <circle cx="12" cy="12" r="3"></circle>
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
    </svg>
  </button>
  <button id="stopComfyBtn" class="ghost-btn" style="display:none" onclick="api('/stop_comfy', 'POST').then(poll)">Stop Comfy</button>
  <a class="creator-link" href="https://x.com/AIBizarrothe" target="_blank" rel="noopener" title="Follow Mr. Bizarro on X (the panel's creator)">
    <img src="/assets/bizarro-avatar.jpg" class="creator-avatar" alt="">
    <span>by Bizarro</span>
    <span class="x-icon">↗ X</span>
  </a>
</header>

<main class="layout">

  <!-- ============== FORM PANE ============== -->
  <aside class="form-pane">
    <form id="genForm">
      <input type="hidden" name="preset_label" id="preset_label" value="">

      <!-- Inline models card. Sits ABOVE the mode picker because for many
           users the very first thing they need to do is download base
           weights — burying that in a header modal hides the whole point
           of the panel. The card has four visual states it cycles through
           depending on /status data; see updateModelsCard() in the JS. -->
      <div id="modelsInline" class="models-inline" style="display:none">
        <button type="button" class="models-inline-dismiss" id="modelsInlineDismiss"
                title="Hide this card (won't show again until model state changes)"
                onclick="dismissModelsCard()">×</button>
        <div class="models-inline-body">
          <div class="models-inline-icon" id="modelsInlineIcon">⬇</div>
          <div class="models-inline-text">
            <div class="ttl" id="modelsInlineTitle">Download required</div>
            <div class="sub" id="modelsInlineSub">Click to start.</div>
            <div class="models-inline-progress" id="modelsInlineProgress" style="display:none">
              <div class="bar"><div class="fill" id="modelsInlineFill" style="width:0%"></div></div>
              <div class="last" id="modelsInlineLast"></div>
            </div>
          </div>
          <div class="models-inline-actions" id="modelsInlineActions"></div>
        </div>
        <a class="models-inline-link" onclick="openModelsModal()">Manage all models →</a>
      </div>

      <h2>Mode</h2>
      <div class="pill-group cols-2" id="modeGroup" style="grid-template-columns: 1fr 1fr 1fr 1fr;">
        <button type="button" class="pill-btn" data-mode="t2v"><span>Text</span><span class="sub">prompt → video</span></button>
        <button type="button" class="pill-btn" data-mode="i2v"><span>Image</span><span class="sub">image + prompt</span></button>
        <button type="button" class="pill-btn" data-mode="keyframe"><span>FFLF</span><span class="sub">first + last frame</span></button>
        <button type="button" class="pill-btn" data-mode="extend"><span>Extend</span><span class="sub">continue a clip</span></button>
      </div>
      <input type="hidden" name="mode" id="mode" value="t2v">

      <h2>Quality</h2>
      <div class="pill-group cols-3" id="qualityGroup">
        <button type="button" class="pill-btn" data-quality="draft"><span>Draft</span><span class="sub">half size · ~2 min</span></button>
        <button type="button" class="pill-btn active" data-quality="standard"><span>Standard</span><span class="sub">full size · ~7 min</span></button>
        <button type="button" class="pill-btn disabled" data-quality="high" id="qualityHigh"><span>High</span><span class="sub" id="highSub">Q8 not installed</span></button>
      </div>
      <input type="hidden" name="quality" id="quality" value="standard">

      <div id="warnBanner" class="warn-banner"></div>

      <!-- Mode-specific: image (I2V).
           New picker: a clickable / drag-drop tile + a "Recent uploads"
           strip below. The raw path input is gone — paths are still set
           via the hidden field for form submission, but never typed by the
           user. Same component is reused for FFLF Start / End below. -->
      <div class="mode-only" id="imageSection">
        <h2>Reference image</h2>
        <div class="picker" data-key="image">
          <div class="picker-drop" id="picker_drop_image">
            <div class="picker-empty">
              <div class="picker-icon">🖼</div>
              <div class="picker-cta">Drop image here, or <strong>click to browse</strong></div>
              <div class="hint">PNG / JPG / WEBP — auto cover-crop to model size</div>
            </div>
            <img class="picker-preview" id="picker_preview_image" alt="" style="display:none">
            <button type="button" class="picker-clear" id="picker_clear_image" title="Clear" style="display:none">×</button>
          </div>
          <input type="file" id="picker_file_image" accept="image/*" style="display:none">
          <input type="hidden" name="image" id="image" value="">
          <div class="picker-recent" id="picker_recent_image_wrap" style="display:none">
            <div class="picker-recent-label">Recent uploads · click to use</div>
            <div class="picker-recent-strip" id="picker_recent_image"></div>
          </div>
        </div>
      </div>

      <!-- Mode-specific: keyframe (FFLF). Two pickers, same component. -->
      <div class="mode-only" id="keyframeSection">
        <h2>Start frame (frame 0)</h2>
        <div class="picker" data-key="start_image">
          <div class="picker-drop" id="picker_drop_start_image">
            <div class="picker-empty">
              <div class="picker-icon">🎬</div>
              <div class="picker-cta">Drop the <strong>first frame</strong>, or <strong>click to browse</strong></div>
              <div class="hint">This image opens the clip — its aspect picks the output dimensions.</div>
            </div>
            <img class="picker-preview" id="picker_preview_start_image" alt="" style="display:none">
            <button type="button" class="picker-clear" id="picker_clear_start_image" title="Clear" style="display:none">×</button>
          </div>
          <input type="file" id="picker_file_start_image" accept="image/*" style="display:none">
          <input type="hidden" name="start_image" id="start_image" value="">
          <div class="picker-recent" id="picker_recent_start_image_wrap" style="display:none">
            <div class="picker-recent-label">Recent uploads · click to use</div>
            <div class="picker-recent-strip" id="picker_recent_start_image"></div>
          </div>
        </div>

        <h2>End frame (last frame)</h2>
        <div class="picker" data-key="end_image">
          <div class="picker-drop" id="picker_drop_end_image">
            <div class="picker-empty">
              <div class="picker-icon">🎯</div>
              <div class="picker-cta">Drop the <strong>last frame</strong>, or <strong>click to browse</strong></div>
              <div class="hint">A close-up here anchors face identity through the clip.</div>
            </div>
            <img class="picker-preview" id="picker_preview_end_image" alt="" style="display:none">
            <button type="button" class="picker-clear" id="picker_clear_end_image" title="Clear" style="display:none">×</button>
          </div>
          <input type="file" id="picker_file_end_image" accept="image/*" style="display:none">
          <input type="hidden" name="end_image" id="end_image" value="">
          <div class="picker-recent" id="picker_recent_end_image_wrap" style="display:none">
            <div class="picker-recent-label">Recent uploads · click to use</div>
            <div class="picker-recent-strip" id="picker_recent_end_image"></div>
          </div>
        </div>

        <div class="hint">FFLF needs Q8 (auto-selects High quality). The model interpolates between the two frames you provide.</div>
      </div>

      <!-- Mode-specific: extend -->
      <div class="mode-only" id="extendSection">
        <h2>Source video</h2>
        <select id="extendSrcSelect" onchange="document.getElementById('video_path').value=this.value"></select>
        <input name="video_path" id="video_path" placeholder="/path/to/source.mp4" style="margin-top:6px">
        <div class="row" style="margin-top:8px">
          <div>
            <label class="lbl">Extend by (seconds)</label>
            <!-- min must align to step from value=2: with min=0.4 step=0.5 the
                 valid sequence is 0.4, 0.9, 1.4, 1.9, 2.4… — `2` is OFF the
                 grid. Chrome blocks the entire form submission silently when
                 ANY input fails validation, even hidden ones in inactive
                 modes. Generate appeared dead until cocktailpeanut diagnosed
                 it. With min=0.5 the sequence is 0.5, 1.0, 1.5, 2.0… and
                 value=2 is valid. -->
            <input id="extend_seconds" type="number" value="2" min="0.5" max="10" step="0.5">
            <input type="hidden" name="extend_frames" id="extend_frames" value="6">
            <div class="hint" id="extendDurationHint" style="margin-top:4px">≈ 2.0 s of new content (6 latent frames × 8 video frames at 24 fps)</div>
          </div>
          <div>
            <label class="lbl">Direction</label>
            <select name="extend_direction" id="extend_direction">
              <option value="after" selected>After</option>
              <option value="before">Before</option>
            </select>
          </div>
        </div>

        <!-- Speed/quality preset for extend.
             Fast: cfg_scale=1.0 (no CFG, ~half the activation memory) +
                   12 steps. Fits comfortably on 64 GB at 1280×704. ~3-5 min.
             Quality: cfg_scale=3.0 + 30 steps. The upstream defaults. Will
                   swap on 64 GB at 1280×704 (peak ~47 GB resident + 12 GB
                   swap) — only pick this on a 96+ GB machine or below 768
                   max-side. Pinokio 64 GB users should leave this on Fast. -->
        <label class="lbl" style="margin-top:10px">Extend mode</label>
        <div class="pill-group cols-2" id="extendModeGroup">
          <button type="button" class="pill-btn active" data-extend-mode="fast"><span>Fast</span><span class="sub">12 steps · no CFG · 64 GB safe</span></button>
          <button type="button" class="pill-btn" data-extend-mode="quality"><span>Quality</span><span class="sub">30 steps · CFG 3.0 · 96+ GB</span></button>
        </div>
        <input type="hidden" name="extend_steps" id="extend_steps" value="12">
        <input type="hidden" name="extend_cfg"   id="extend_cfg"   value="1.0">
        <div class="hint">Each latent ≈ 8 frames (~0.33s). Quality mode runs the upstream defaults but pushes 1280×704 into swap on 64 GB Macs (~2 hr/render). Stick with Fast unless you've got more RAM.</div>
      </div>

      <h2>Prompt</h2>
      <!-- Audio-guidance hint in the placeholder. LTX 2.3 generates audio
           jointly but is CONDITIONED on prompt cues — visual-only prompts
           produce near-silent room tone (peaks at -37 dB on test runs).
           Most users assume "no sound" = bug. Hint nudges them to describe
           the soundscape too. Documented in the LTX 2.3 paper but unobvious. -->
      <textarea name="prompt" id="prompt" placeholder="Describe the scene AND the sound: e.g. 'wizard in a forest clearing, fireflies spiraling up — low whispered chant, ember crackle, distant owl'. Audio is generated jointly with video; without sound cues the model outputs near-silent ambient."></textarea>
      <!-- Gemma-driven prompt enhancement (upstream's `ltx-2-mlx enhance`).
           Rewrites your prompt with the structure/keywords LTX 2.3 trained
           on. ~12-15s on cold start (Gemma needs to load), ~5s warm.

           "No music" checkbox: appends an audio constraint to the prompt
           on submit so users can keep voice + ambient and skip the model's
           default soundtrack tendency. Music is annoying in editing because
           it can't be cleanly removed without affecting the dialogue track.
           Recommended for clips you plan to score yourself in post. -->
      <!-- Three controls on one row: Enhance button on the left, then
           HDR + No-music toggles pushed to the right. HDR is exposed as
           a plain toggle even though it's implemented as a curated
           Lightricks LoRA — most users don't care that it's a LoRA, they
           just want HDR or not. -->
      <div class="row-actions" style="margin-top:8px; display: flex; gap: 8px; align-items: center; flex-wrap: wrap">
        <button type="button" class="ghost-btn" id="enhanceBtn" onclick="enhancePrompt()" title="Use Gemma to rewrite your prompt in the style LTX 2.3 was trained on">✨ Enhance with Gemma</button>
        <label class="toggle-pill" id="hdrPill" style="margin-left:auto"
               title="Boost dynamic range and color depth. Implemented as the official Lightricks HDR LoRA fused into the transformer. First HDR job spawns a one-time download (~120 MB) of the LoRA weights from Hugging Face, then renders share the cache.">
          <input type="checkbox" id="hdr" name="hdr">
          <span class="toggle-dot"></span>
          <span>HDR</span>
        </label>
        <label class="toggle-pill" id="noMusicPill"
               title="When on, the prompt is augmented with: 'Audio: voice and ambient sounds only, no music, no soundtrack, no score.' Useful for clips you'll score yourself in post — music can't be cleanly removed afterwards.">
          <input type="checkbox" id="noMusic" name="no_music">
          <span class="toggle-dot"></span>
          <span>No music</span>
        </label>
      </div>

      <!-- LoRA picker. Collapsible because most users won't touch it on
           a given session, and inline because hiding it behind a modal
           makes it easy to forget. Loaded from /loras on open; the
           "Browse CivitAI" affordance opens the search modal. -->
      <details id="lorasDetails" style="margin-top:14px">
        <summary class="loras-summary">
          <span>LoRAs</span>
          <span class="hint" id="lorasSummaryCount">none active</span>
        </summary>
        <div class="loras-body" id="lorasBody">
          <div class="hint" id="lorasEmpty" style="margin-top:8px">
            Drop <code>.safetensors</code> files into <code id="lorasDir">mlx_models/loras/</code>
            to use them, or browse CivitAI below. Each LoRA picks up an
            optional sidecar <code>.json</code> with name + trigger words +
            recommended strength.
          </div>
          <div class="loras-list" id="lorasList"></div>
          <div class="loras-actions" style="margin-top:10px; display:flex; gap:8px">
            <button type="button" class="ghost-btn" onclick="refreshLoras()">↻ Rescan folder</button>
            <button type="button" class="ghost-btn" onclick="openCivitaiModal()">🔍 Browse CivitAI</button>
          </div>
        </div>
      </details>
      <!-- Hidden field; updated by the LoRA picker JS to a JSON-encoded
           array of {path, strength} that make_job parses. -->
      <input type="hidden" id="lorasJson" name="loras" value="">

      <!-- Advanced — power-user options. We trimmed two things in cleanup:
           1. Removed the "Enhance prompt" checkbox: it was labeled "CLI only,
              ignored by helper" — actual dead code. The Enhance button next
              to the prompt textarea is the real thing.
           2. The I2V audio mode (mux external audio over LTX-generated video)
              only applies in I2V mode; the panel auto-hides it elsewhere via
              `.mode-only` on the wrapper. -->
      <details>
        <summary>Advanced</summary>
        <div class="mode-only" id="i2vAudioModeSection">
          <label class="lbl">I2V audio source</label>
          <select id="i2vMode">
            <option value="i2v" selected>Joint audio (LTX generates audio synced with the visual)</option>
            <option value="i2v_clean_audio">Use external audio file (mux it onto LTX video)</option>
          </select>
          <div class="mode-only" id="audioSection">
            <label class="lbl">Audio file path</label>
            <input name="audio" id="audio" placeholder="/path/to/your/track.wav">
          </div>
        </div>
        <label class="check" style="margin-top:6px">
          <input type="checkbox" name="open_when_done" id="open_when_done"> Open file when done
        </label>
      </details>

      <!-- Sizing for non-extend modes -->
      <div class="mode-only" id="sizingSection">
        <h2>Aspect</h2>
        <div class="pill-group cols-2" id="aspectGroup">
          <button type="button" class="pill-btn active" data-aspect="landscape"><span>16 : 9</span><span class="sub">horizontal</span></button>
          <button type="button" class="pill-btn" data-aspect="vertical"><span>9 : 16</span><span class="sub">vertical</span></button>
        </div>
        <input type="hidden" id="aspect" value="landscape">

        <!-- Manual W/H fields only shown in T2V (where they're the primary
             control). In image flows they're auto-set from aspect + quality
             so the input image drives the framing without surprise crops. -->
        <div id="dimsRow" class="row" style="margin-top:10px">
          <div><label class="lbl">Width</label><input name="width" id="width" value="1280" type="number" min="32" step="32"></div>
          <div><label class="lbl">Height</label><input name="height" id="height" value="704" type="number" min="32" step="32"></div>
        </div>

        <div class="row3" style="margin-top:10px">
          <div><label class="lbl">Duration (s)</label><input id="duration" value="5" type="number" min="1" max="20" step="1"></div>
          <div><label class="lbl">Frames (8k+1)</label><input name="frames" id="frames" value="121" type="number" min="1"></div>
          <div><label class="lbl">Seed (-1 random)</label><input name="seed" id="seed" value="-1"></div>
        </div>

        <input type="hidden" name="steps" id="steps" value="8">

        <div class="derived" id="derived"></div>
      </div>

      <!-- Comfy-kill toggle. Hidden in DOM by default; shown only when the
           panel detects ComfyUI actually running on this Mac (see updateUI
           in the poll handler). Default-on when surfaced because Comfy idle
           costs ~27 GB and on a 64 GB Mac that pushes 720p+ renders into
           swap. The reviewer flagged the always-on hidden field as too
           surprising for a public install — this version is opt-in only
           when there's something to opt into. -->
      <div id="comfyKillRow" class="comfy-row" style="display:none">
        <label class="lbl" style="display:flex; align-items:center; gap:8px; cursor:pointer">
          <input type="checkbox" name="stop_comfy" id="stop_comfy" value="on" checked>
          <span>Stop ComfyUI before render <span style="color:var(--muted)">(detected · frees ~27 GB)</span></span>
        </label>
      </div>

      <div class="actions">
        <button type="submit" class="primary" id="genBtn">Generate</button>
        <button type="button" class="danger" onclick="api('/stop', 'POST').then(poll)">Stop</button>
      </div>
      <div class="row-actions" style="margin-top:8px">
        <button type="button" class="small" onclick="openBatch()">Batch paste</button>
        <button type="button" class="small" id="pauseBtn" onclick="togglePause()">Pause queue</button>
        <button type="button" class="small" onclick="api('/queue/clear','POST').then(poll)">Clear queue</button>
      </div>
    </form>
  </aside>

  <!-- ============== STAGE PANE: PLAYER + CAROUSEL ============== -->
  <section class="stage-pane">
    <div class="player-wrap empty" id="playerWrap">
      <div>No outputs yet</div>
      <div style="font-size:11px;opacity:0.6">generate something on the left to begin</div>
    </div>
    <div class="player-meta" id="playerMeta" style="display:none">
      <div class="name" id="playerName"></div>
      <div class="actions-bar">
        <button id="loadParamsBtn" onclick="loadParams()" disabled>Load params</button>
        <button onclick="useAsExtendSource()">Use as Extend</button>
        <button onclick="hideActive()">Hide</button>
      </div>
    </div>
    <div class="carousel-wrap">
      <div class="carousel-head">
        <h3 id="carouselTitle">Outputs</h3>
        <div class="seg">
          <button id="filterAll" onclick="setFilter('visible')" class="active">Visible</button>
          <button id="filterHidden" onclick="setFilter('hidden')">Hidden</button>
        </div>
      </div>
      <div class="carousel" id="carousel"></div>
    </div>
  </section>
</main>

<!-- ============== CIVITAI MODAL ============== -->
<!-- LoRA discovery + install modal. Hits /civitai/search to populate
     the grid and /civitai/download to fetch a selected LoRA into
     mlx_models/loras/ with a sidecar JSON capturing the metadata. -->
<div id="civitaiModal" class="models-modal" style="display:none"
     onclick="if(event.target===this) closeCivitaiModal()">
  <div class="models-card" style="width: min(960px, 96vw)">
    <div class="models-head">
      <h2>Browse CivitAI for LTX 2.3 LoRAs</h2>
      <button class="ghost-btn" onclick="closeCivitaiModal()">Close</button>
    </div>
    <div class="models-hint">
      LoRAs land in <code id="civitaiTargetDir">mlx_models/loras/</code>
      with a sidecar JSON that carries the trigger words and recommended
      strength. The CivitAI page link stays in the sidecar for attribution.
    </div>
    <!-- Auth hint shown only when the panel can't see CIVITAI_API_KEY.
         CivitAI now requires a token for most LoRA downloads (even SFW).
         Hidden by default; populated in openCivitaiModal() based on the
         /loras response. -->
    <div id="civitaiAuthHint" class="models-hint"
         style="display:none; border-left:3px solid var(--warning, #d29922); padding-left:10px; margin-top:8px">
      <strong>Heads up:</strong> CivitAI now requires an API token to
      download LoRAs. Get one at
      <a href="https://civitai.com/user/account" target="_blank" rel="noopener">civitai.com/user/account</a>,
      then set <code>CIVITAI_API_KEY</code> in your environment and restart
      the panel. Browsing works without it; Install will fail with a 401
      until the token is set.
    </div>
    <div class="civitai-search-bar">
      <input type="text" id="civitaiQuery" placeholder="Search by name, style, creator…"
             oninput="if(this._t) clearTimeout(this._t); this._t = setTimeout(civitaiSearch, 350)"
             onkeydown="if(event.key==='Enter'){ event.preventDefault(); civitaiSearch(); }">
      <label class="toggle-pill" id="civitaiNsfwPill">
        <input type="checkbox" id="civitaiNsfw">
        <span class="toggle-dot"></span>
        <span>Show NSFW</span>
      </label>
    </div>
    <div class="civitai-grid" id="civitaiGrid">
      <div class="hint">Loading…</div>
    </div>
    <div class="civitai-status-line" id="civitaiStatus"></div>
    <div style="display:flex; justify-content:center; margin-top:14px">
      <button type="button" class="ghost-btn" id="civitaiLoadMore"
              style="display:none" onclick="civitaiLoadMore()">Load more</button>
    </div>
  </div>
</div>

<!-- ============== SETTINGS MODAL ============== -->
<!-- Opened by the gear icon in the header. Lets users pick output codec
     presets (Standard / Video production / Web) or set custom pix_fmt + crf.
     Persisted to panel_settings.json; the helper subprocess restarts on
     codec change so the new ffmpeg args take effect on next job. -->
<div id="settingsModal" class="models-modal" style="display:none"
     onclick="if(event.target===this) closeSettingsModal()">
  <div class="models-card">
    <div class="models-head">
      <h2>Settings</h2>
      <button class="ghost-btn" onclick="closeSettingsModal()">Close</button>
    </div>
    <div class="models-hint">
      Pick how rendered clips are encoded. Settings are saved to
      <code>panel_settings.json</code> and apply to every new render.
      Files already in the gallery are not re-encoded.
    </div>

    <div class="settings-section">
      <h3>Output format</h3>
      <div class="preset-grid" id="settingsPresets">
        <!-- populated by openSettingsModal() from /settings -->
      </div>
    </div>

    <div class="settings-section">
      <h3>API tokens</h3>
      <div class="hint" style="margin-bottom:8px">
        Saved locally in <code>panel_settings.json</code>. Never sent
        anywhere except as auth headers to civitai.com / huggingface.co.
        Power users can override with <code>CIVITAI_API_KEY</code> /
        <code>HF_TOKEN</code> env vars; the saved value here wins when
        both are set.
      </div>

      <!-- CivitAI token row -->
      <div class="token-row">
        <div class="token-label">
          <span>CivitAI API key</span>
          <span class="token-status" id="civitaiKeyStatus">—</span>
        </div>
        <div class="token-row-input">
          <input type="password" id="civitaiKeyInput" autocomplete="off"
                 placeholder="paste key here…"
                 oninput="onTokenInput('civitai')">
          <button type="button" class="ghost-btn" id="civitaiKeyToggle"
                  onclick="toggleTokenVisibility('civitaiKeyInput', this)">show</button>
          <button type="button" class="ghost-btn"
                  onclick="testToken('civitai')">test</button>
          <button type="button" class="ghost-btn" id="civitaiKeyClear"
                  onclick="clearToken('civitai')" style="display:none">clear</button>
        </div>
        <div class="hint" id="civitaiTestResult">
          Required for installing CivitAI LoRAs.
          Get one at <a href="https://civitai.com/user/account" target="_blank" rel="noopener">civitai.com/user/account</a>
          (Account → API Keys → Add).
        </div>
      </div>

      <!-- HF token row -->
      <div class="token-row">
        <div class="token-label">
          <span>Hugging Face token</span>
          <span class="token-status" id="hfTokenStatus">—</span>
        </div>
        <div class="token-row-input">
          <input type="password" id="hfTokenInput" autocomplete="off"
                 placeholder="hf_…"
                 oninput="onTokenInput('hf')">
          <button type="button" class="ghost-btn" id="hfTokenToggle"
                  onclick="toggleTokenVisibility('hfTokenInput', this)">show</button>
          <button type="button" class="ghost-btn"
                  onclick="testToken('hf')">test</button>
          <button type="button" class="ghost-btn" id="hfTokenClear"
                  onclick="clearToken('hf')" style="display:none">clear</button>
        </div>
        <div class="hint" id="hfTestResult">
          Required for gated LoRAs (Lightricks HDR + Control LoRAs).
          Get one at <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noopener">huggingface.co/settings/tokens</a>
          — read access is enough.
        </div>
      </div>
    </div>

    <div class="settings-section" id="settingsCustomSection" style="display:none">
      <h3>Custom (advanced)</h3>
      <div class="settings-row" style="margin-bottom:10px">
        <label>pix_fmt</label>
        <select id="settingsPixFmt" style="flex:1">
          <option value="yuv420p">yuv420p — web standard, broad support</option>
          <option value="yuv422p">yuv422p — pro, more chroma than 420p</option>
          <option value="yuv444p">yuv444p — full chroma, no subsampling</option>
          <option value="yuv420p10le">yuv420p10le — 10-bit, HDR-ready</option>
          <option value="yuv422p10le">yuv422p10le — 10-bit pro</option>
          <option value="yuv444p10le">yuv444p10le — 10-bit, full chroma</option>
        </select>
      </div>
      <div class="settings-row">
        <label>CRF</label>
        <input type="range" id="settingsCrfRange" min="0" max="30" step="1"
               style="flex:1; width:auto" oninput="document.getElementById('settingsCrfNum').value = this.value">
        <input type="number" id="settingsCrfNum" min="0" max="30" step="1"
               style="width:60px"
               oninput="document.getElementById('settingsCrfRange').value = this.value">
      </div>
      <div class="hint" style="margin-top:6px">
        0 = mathematically lossless · 18 = visually lossless ·
        23 = web default · 28+ = lossy
      </div>
    </div>

    <div class="settings-foot">
      <span class="settings-status" id="settingsStatus"></span>
      <button class="ghost-btn" onclick="closeSettingsModal()">Cancel</button>
      <button class="primary-btn" id="settingsApplyBtn" onclick="applySettings()">Apply</button>
    </div>
  </div>
</div>

<!-- ============== TIER MODAL ============== -->
<!-- Opened by the "tier" pill in the header. Renders the detected
     hardware tier + what it allows. Helps users understand WHY some
     options are disabled instead of just hitting a wall mid-flow. -->
<div id="tierModal" class="models-modal" style="display:none" onclick="if(event.target===this) closeTierModal()">
  <div class="models-card">
    <div class="models-head">
      <h2 id="tierModalTitle">Hardware tier</h2>
      <button class="ghost-btn" onclick="closeTierModal()">Close</button>
    </div>
    <div class="models-hint" id="tierModalBlurb"></div>
    <ul class="models-list" id="tierCapsList"></ul>
    <div class="models-foot">
      Set <code>LTX_TIER_OVERRIDE=base|standard|high|pro</code> in the env to
      force a tier (useful for testing what other users see).
    </div>
  </div>
</div>

<!-- ============== MODELS MODAL ============== -->
<!-- Opened by the "models" pill in the header. Shows per-repo download
     status from /models, with a Download button per row. Active downloads
     stream into the existing log at the bottom of the page. -->
<div id="modelsModal" class="models-modal" style="display:none" onclick="if(event.target===this) closeModelsModal()">
  <div class="models-card">
    <div class="models-head">
      <h2>Models</h2>
      <button class="ghost-btn" onclick="closeModelsModal()">Close</button>
    </div>
    <div class="models-hint" id="modelsHint">Loading…</div>
    <ul class="models-list" id="modelsList"></ul>
    <div class="models-foot" id="modelsFoot"></div>
  </div>
</div>

<!-- ============== BOTTOM TABBED PANE ============== -->
<aside class="bottom-pane" id="bottomPane">
  <nav class="tabs">
    <button data-tab="now" class="active">Now</button>
    <button data-tab="queue">Queue <span class="badge" id="queueBadge" style="display:none">0</span></button>
    <button data-tab="recent">Recent</button>
    <button data-tab="logs">Logs</button>
    <span class="spacer"></span>
    <a class="model-credit" id="modelTag" href="https://github.com/dgrauet/ltx-2-mlx" target="_blank" rel="noopener" title="MLX port by @dgrauet"></a>
    <button class="tab-collapse" onclick="document.getElementById('bottomPane').classList.toggle('collapsed')">Collapse</button>
  </nav>
  <div class="bottom-body">
    <div class="tab-content show" id="tab-now">
      <div class="now-card idle" id="nowCard">
        <div class="ttl">Idle</div>
        <div class="progress-bar"><div class="fill" id="progressFill" style="width:0%"></div></div>
        <div class="meta" id="nowDetail">No job running</div>
      </div>
    </div>
    <div class="tab-content" id="tab-queue">
      <ul class="row-list" id="queueList"></ul>
    </div>
    <div class="tab-content" id="tab-recent">
      <ul class="row-list" id="historyList"></ul>
    </div>
    <div class="tab-content" id="tab-logs">
      <pre class="log" id="log">No log yet.</pre>
    </div>
  </div>
</aside>

<!-- ============== BATCH MODAL ============== -->
<div class="modal-bg" id="batchModal" onclick="if(event.target===this)closeBatch()">
  <div class="modal">
    <h3>Batch paste — split prompts with <code>---</code> on its own line</h3>
    <textarea class="batch" id="batchPrompts" placeholder="First prompt here.

---

Second prompt.

---

Third prompt."></textarea>
    <div class="hint" style="margin-top:6px">Each chunk between <code>---</code> lines becomes a queued job using the current form settings. Auto-open is forced off for batches.</div>
    <div class="modal-actions">
      <button class="small" onclick="closeBatch()">Cancel</button>
      <button class="small primary" style="padding:6px 14px" onclick="queueBatch().then(closeBatch)">Queue all</button>
    </div>
  </div>
</div>

<script>
const BOOT = __BOOTSTRAP__;
const ASPECTS = BOOT.aspects;
const FPS = BOOT.fps;

let filterMode = 'visible';
let activePath = null;
let currentOutputs = [];
let currentMode = 't2v';

// Model tag in the bottom-pane nav links to dgrauet's repo. Strip an
// absolute filesystem path back to the HF repo id form for display
// (the panel sets LTX_MODEL to a local path in Pinokio installs).
(function () {
  const m = String(BOOT.model || '');
  let label = m;
  const idx = m.indexOf('mlx_models/');
  if (idx >= 0) label = m.slice(idx + 'mlx_models/'.length);
  if (label.startsWith('/')) label = label.split('/').slice(-2).join('/');
  document.getElementById('modelTag').textContent = label;
})();
// `audio` is still a free-text input (advanced section); `image` is now a
// picker — leave the picker empty by default and let the user pick or
// drop. Pre-filling examples/reference.png surprised users into rendering
// the demo image when they meant to leave it blank.
document.getElementById('audio').value = BOOT.default_audio;

// ====== Pill-button group helpers ======
function setMode(mode) {
  currentMode = mode;
  document.getElementById('mode').value = mode;
  document.querySelectorAll('#modeGroup .pill-btn').forEach(b => b.classList.toggle('active', b.dataset.mode === mode));
  // For i2v, switch the actual mode based on the i2vMode select
  if (mode === 'i2v') {
    document.getElementById('mode').value = document.getElementById('i2vMode').value;
  }
  // Keyframe REQUIRES Q8 (uses dev transformer); force quality=high.
  // If Q8 isn't available the High pill stays disabled and the user gets the
  // same "Q8 not installed" hint as elsewhere.
  if (mode === 'keyframe') {
    setQuality('high');
  }
  updateDerived();
  // Refresh the inline models card immediately — switching to FFLF when
  // Q8 is missing should surface the Download Q8 CTA without waiting for
  // the next 1.5s poll tick.
  if (LAST_STATUS) updateModelsCard(LAST_STATUS);
}
function setQuality(q) {
  document.getElementById('quality').value = q;
  document.querySelectorAll('#qualityGroup .pill-btn').forEach(b => b.classList.toggle('active', b.dataset.quality === q));
  applyQuality();
  if (LAST_STATUS) updateModelsCard(LAST_STATUS);
}
function setAspect(a) {
  document.getElementById('aspect').value = a;
  document.querySelectorAll('#aspectGroup .pill-btn').forEach(b => b.classList.toggle('active', b.dataset.aspect === a));
  applyAspect(a);
}
function setExtendMode(m) {
  // Fast = no-CFG path, fits in 64 GB at 1280×704. Quality = upstream
  // defaults, requires headroom. Both are exposed on the form via hidden
  // inputs; this just flips the values + active pill.
  const steps = m === 'quality' ? 30  : 12;
  const cfg   = m === 'quality' ? 3.0 : 1.0;
  document.getElementById('extend_steps').value = String(steps);
  document.getElementById('extend_cfg').value   = String(cfg);
  document.querySelectorAll('#extendModeGroup .pill-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.extendMode === m));
}

document.querySelectorAll('#modeGroup .pill-btn').forEach(b => b.onclick = () => setMode(b.dataset.mode));
document.querySelectorAll('#qualityGroup .pill-btn').forEach(b => b.onclick = () => { if (!b.classList.contains('disabled')) setQuality(b.dataset.quality); });
document.querySelectorAll('#aspectGroup .pill-btn').forEach(b => b.onclick = () => setAspect(b.dataset.aspect));
document.querySelectorAll('#extendModeGroup .pill-btn').forEach(b => b.onclick = () => setExtendMode(b.dataset.extendMode));

// Prompt enhancement via Gemma — wraps the upstream CLI's `enhance`
// subcommand. Cold start ~12-15s (Gemma load), warm ~5s. Blocks the UI
// during the request (just the button — rest of the form stays usable).
async function enhancePrompt() {
  const ta = document.getElementById('prompt');
  const original = ta.value.trim();
  if (!original) { alert('Type a prompt before enhancing it.'); return; }
  const mode = (currentMode === 'i2v' || currentMode === 'keyframe' || currentMode === 'extend') ? 'i2v' : 't2v';
  const btn = document.getElementById('enhanceBtn');
  const originalLabel = btn.textContent;
  btn.disabled = true;
  btn.textContent = '✨ Loading Gemma… (~15s on cold start)';
  let res;
  try {
    const fd = new URLSearchParams({ prompt: original, mode });
    const r = await fetch('/prompt/enhance', { method: 'POST', body: fd });
    res = await r.json();
  } catch (e) {
    alert('Enhance request failed: ' + (e.message || e));
    btn.disabled = false; btn.textContent = originalLabel;
    return;
  }
  btn.disabled = false; btn.textContent = originalLabel;
  if (res.error) { alert('Enhance failed: ' + res.error); return; }
  // Show diff in a confirm so the user can decide whether to accept.
  const accept = confirm(
    `Original:\n${res.original}\n\nEnhanced:\n${res.enhanced}\n\nReplace your prompt with the enhanced version?`
  );
  if (accept) {
    ta.value = res.enhanced;
    ta.dispatchEvent(new Event('input', { bubbles: true }));
  }
}

// Extend duration: user types seconds, we convert to latent frames behind
// the scenes. Each latent = 8 video frames; at 24 fps that's 0.333 s.
// Round-up so the user gets at least the seconds they asked for.
//   seconds → latents: ceil(seconds * 24 / 8)
//   latents → actual seconds: latents * 8 / 24
// Hint line shows both numbers so the conversion isn't a black box.
function syncExtendDuration() {
  const secInput = document.getElementById('extend_seconds');
  const hidden = document.getElementById('extend_frames');
  const hint = document.getElementById('extendDurationHint');
  if (!secInput || !hidden || !hint) return;
  const seconds = parseFloat(secInput.value) || 0;
  const latents = Math.max(1, Math.ceil(seconds * 24 / 8));
  const actualSec = (latents * 8 / 24);
  hidden.value = String(latents);
  hint.textContent = `≈ ${actualSec.toFixed(2)} s of new content (${latents} latent frames × 8 video frames at 24 fps)`;
}
document.getElementById('extend_seconds').addEventListener('input', syncExtendDuration);
syncExtendDuration();   // initialize on load
document.getElementById('i2vMode').addEventListener('change', () => {
  document.getElementById('audioSection').classList.toggle('show', document.getElementById('i2vMode').value === 'i2v_clean_audio');
  if (currentMode === 'i2v') document.getElementById('mode').value = document.getElementById('i2vMode').value;
});

function applyAspect(key) {
  if (!ASPECTS[key]) return;
  document.getElementById('aspect').value = key;
  // Defer the actual sizing to applyQuality — it reads aspect + quality and
  // picks half-size (Draft) or full-size (Standard/High) of the chosen aspect.
  applyQuality();
}

function applyQuality() {
  const q = document.getElementById('quality').value;
  if (q === 'draft' || q === 'standard') {
    document.getElementById('steps').value = 8;
  } else if (q === 'high') {
    document.getElementById('steps').value = 18;
  }
  // Draft and Standard both keep the same aspect ratio. Draft halves each
  // dimension (1280×704 → 640×352 for 16:9; 704×1280 → 352×640 for 9:16).
  // Same ratio, ~25% the pixel count, ~3× faster — a real preview of the
  // final shot at draft cost. Standard restores full size.
  const aspect = document.getElementById('aspect').value || 'landscape';
  const a = ASPECTS[aspect] || ASPECTS.landscape;
  if (q === 'draft') {
    document.getElementById('width').value = Math.round(a.w / 2 / 32) * 32;
    document.getElementById('height').value = Math.round(a.h / 2 / 32) * 32;
  } else if (q === 'standard' || q === 'high') {
    document.getElementById('width').value = a.w;
    document.getElementById('height').value = a.h;
  }
  updateDerived();
}

function durationToFrames(s) {
  const k = Math.max(0, Math.round(s * FPS / 8));
  return k * 8 + 1;
}
function framesToDuration(f) { return ((f - 1) / FPS).toFixed(2); }

// LTX 2.3 requires frame counts in the form 1 + 8k (one keyframe + N
// VAE-temporal blocks of 8 frames each). Typing "100" or "240" wastes
// compute on partially-filled trailing latents — the pipeline rounds
// up internally but charges for the empty slots. Snap on blur to the
// nearest valid value below + 1 (so we never silently render *more*
// than the user asked for, only less or equal).
function snapFramesTo8kPlus1() {
  const el = document.getElementById('frames');
  if (!el) return;
  const v = parseInt(el.value) || 0;
  if (v < 1) { el.value = 9; return; }
  // Nearest 8k+1: round (v-1)/8 to nearest int, multiply back, +1.
  const k = Math.max(1, Math.round((v - 1) / 8));
  const snapped = k * 8 + 1;
  if (snapped !== v) {
    el.value = snapped;
    // Reflect the change in duration too, since they're bound.
    document.getElementById('duration').value = framesToDuration(snapped);
  }
}

function updateDerived() {
  const mode = document.getElementById('mode').value;
  const w = parseInt(document.getElementById('width').value || 0);
  const h = parseInt(document.getElementById('height').value || 0);
  const f = parseInt(document.getElementById('frames').value || 0);
  const dur = (f / FPS).toFixed(2);

  let pw = w, ph = h;
  if (w === 704 && h % 16 === 0) pw = 720;
  if (h === 704 && w % 16 === 0) ph = 720;
  const padded = (pw !== w || ph !== h) && mode === 'i2v_clean_audio';
  const finalRes = padded ? `${w}×${h} → <strong>${pw}×${ph}</strong>` : `<strong>${w}×${h}</strong>`;

  document.getElementById('derived').innerHTML = `Duration <strong>${dur}s</strong> @ ${FPS}fps · ${finalRes} · Steps ${document.getElementById('steps').value}`;

  const warns = [];
  if (w % 32 !== 0) warns.push(`Width ${w} isn't a multiple of 32 (closest ${Math.round(w/32)*32})`);
  if (h % 32 !== 0) warns.push(`Height ${h} isn't a multiple of 32 (closest ${Math.round(h/32)*32})`);
  if (f > 1 && (f - 1) % 8 !== 0) {
    const closest = Math.max(1, Math.round((f - 1) / 8) * 8 + 1);
    warns.push(`Frames work best as 8k+1 (closest ${closest})`);
  }
  const banner = document.getElementById('warnBanner');
  if (warns.length) { banner.innerHTML = '⚠ ' + warns.join(' · '); banner.classList.add('show'); }
  else banner.classList.remove('show');

  // Mode-aware visibility
  const inI2V = mode === 'i2v' || mode === 'i2v_clean_audio';
  const inImageFlow = inI2V || currentMode === 'keyframe';
  document.getElementById('imageSection').classList.toggle('show', inI2V && currentMode !== 'keyframe');
  document.getElementById('extendSection').classList.toggle('show', currentMode === 'extend');
  document.getElementById('keyframeSection').classList.toggle('show', currentMode === 'keyframe');
  document.getElementById('sizingSection').classList.toggle('show', currentMode !== 'extend');
  document.getElementById('audioSection').classList.toggle('show', mode === 'i2v_clean_audio');
  // I2V audio source picker (Advanced) — only relevant in I2V flow.
  // In T2V/Extend/FFLF the model generates audio jointly; there's nothing
  // to swap out, so the dropdown is just noise.
  const i2vAudioSec = document.getElementById('i2vAudioModeSection');
  if (i2vAudioSec) i2vAudioSec.classList.toggle('show', inI2V);
  // In image flows the aspect picker is the only sizing control. Width/height
  // auto-derive from aspect+quality so the source image drives the framing
  // and we don't accidentally cover-crop a 16:9 photo into 9:16.
  const dimsRow = document.getElementById('dimsRow');
  if (dimsRow) dimsRow.style.display = inImageFlow ? 'none' : '';

  // Image previews are now part of the picker component itself — the
  // preview <img> + clear button live inside .picker-drop and are toggled
  // by pickerSetImage(). No per-mode preview management here anymore;
  // the old imagePreview / startImagePreview / endImagePreview elements
  // are gone.
}

['width','height','frames','duration'].forEach(id => {
  const el = document.getElementById(id);
  if (id === 'duration') {
    el.addEventListener('input', e => { document.getElementById('frames').value = durationToFrames(parseFloat(e.target.value) || 0); updateDerived(); });
  } else if (id === 'frames') {
    el.addEventListener('input', e => { document.getElementById('duration').value = framesToDuration(parseInt(e.target.value) || 0); updateDerived(); });
    el.addEventListener('blur', () => { snapFramesTo8kPlus1(); updateDerived(); });
  } else {
    el.addEventListener('input', updateDerived);
  }
});
// Picker hidden inputs no longer take user input — their value changes
// via pickerSetImage(), which already calls updateDerived(). No per-input
// listeners needed.

// Auto-snap the aspect picker based on an image's actual dimensions.
// Avoids the 16:9-source-cropped-to-9:16-strip footgun.
function snapAspectToImage(path) {
  const probe = new Image();
  probe.onload = () => {
    const r = probe.naturalWidth / probe.naturalHeight;
    const target = r >= 1 ? 'landscape' : 'vertical';
    if (document.getElementById('aspect').value !== target) setAspect(target);
  };
  probe.src = '/image?path=' + encodeURIComponent(path);
}

// uploadImage() / uploadKeyframe() were replaced by the unified picker
// component (pickerUploadFile + refreshUploadsStrip). The /upload endpoint
// still drives the actual transfer; the only change is which JS calls it.

// ====== Image picker component ======
// One implementation, three call sites: I2V image, FFLF start_image,
// FFLF end_image. Each picker carries a `key` (the hidden field's name);
// every DOM element it owns is suffixed with `_<key>` so we can wire
// listeners by lookup instead of a per-instance closure.
const PICKERS = ['image', 'start_image', 'end_image'];

function pickerEls(key) {
  return {
    drop:    document.getElementById(`picker_drop_${key}`),
    file:    document.getElementById(`picker_file_${key}`),
    hidden:  document.getElementById(key),
    preview: document.getElementById(`picker_preview_${key}`),
    clear:   document.getElementById(`picker_clear_${key}`),
    empty:   document.querySelector(`#picker_drop_${key} .picker-empty`),
    recentWrap:  document.getElementById(`picker_recent_${key}_wrap`),
    recentStrip: document.getElementById(`picker_recent_${key}`),
  };
}

function pickerSetImage(key, path, opts = {}) {
  const els = pickerEls(key);
  if (!els.hidden) return;
  els.hidden.value = path;
  if (path) {
    els.preview.src = `/image?path=${encodeURIComponent(path)}`;
    els.preview.style.display = 'block';
    els.empty.style.display = 'none';
    els.clear.style.display = 'flex';
    els.drop.classList.add('has-image');
    // Highlight the matching thumbnail in the recent strip if visible.
    if (els.recentStrip) {
      els.recentStrip.querySelectorAll('img').forEach(img => {
        img.classList.toggle('selected', img.dataset.path === path);
      });
    }
    // FFLF anchors framing on the start frame; I2V anchors on its single
    // image. End frame doesn't drive aspect (would override the start frame).
    if (key !== 'end_image' && opts.snapAspect !== false) {
      snapAspectToImage(path);
    }
  } else {
    els.preview.removeAttribute('src');
    els.preview.style.display = 'none';
    els.empty.style.display = '';
    els.clear.style.display = 'none';
    els.drop.classList.remove('has-image');
    if (els.recentStrip) {
      els.recentStrip.querySelectorAll('img').forEach(img => img.classList.remove('selected'));
    }
  }
  updateDerived();
}

async function pickerUploadFile(key, file) {
  const els = pickerEls(key);
  if (!file || !els.drop) return;
  // Inline progress overlay on the drop tile while the upload runs.
  let busy = els.drop.querySelector('.picker-uploading');
  if (!busy) {
    busy = document.createElement('div');
    busy.className = 'picker-uploading';
    busy.textContent = `Uploading ${file.name}…`;
    els.drop.appendChild(busy);
  }
  try {
    const fd = new FormData(); fd.append('image', file);
    const r = await fetch('/upload', { method: 'POST', body: fd });
    const data = await r.json();
    if (!data.ok) throw new Error(data.error || 'upload failed');
    pickerSetImage(key, data.path);
    // Refresh the "Recent uploads" strip so the just-uploaded file shows
    // up immediately for the other slots too.
    refreshUploadsStrip();
  } catch (e) {
    alert(`Upload failed: ${e.message || e}`);
  } finally {
    busy.remove();
  }
}

function pickerWire(key) {
  const els = pickerEls(key);
  if (!els.drop) return;
  // Click → file dialog. Skip when the click came from the clear button.
  els.drop.addEventListener('click', (e) => {
    if (e.target.closest('.picker-clear')) return;
    els.file.click();
  });
  els.file.addEventListener('change', () => {
    if (els.file.files[0]) pickerUploadFile(key, els.file.files[0]);
    els.file.value = '';   // allow re-uploading the same file
  });
  els.clear.addEventListener('click', (e) => { e.stopPropagation(); pickerSetImage(key, ''); });
  // Drag-drop. preventDefault on dragover is what enables drop.
  els.drop.addEventListener('dragover', (e) => {
    e.preventDefault();
    els.drop.classList.add('dragover');
  });
  els.drop.addEventListener('dragleave', () => els.drop.classList.remove('dragover'));
  els.drop.addEventListener('drop', (e) => {
    e.preventDefault();
    els.drop.classList.remove('dragover');
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) pickerUploadFile(key, f);
  });
}

let _uploadsCache = [];   // last fetched list, kept module-level so all
                          //   three pickers render the same source data.
async function refreshUploadsStrip() {
  let data;
  try { data = await api('/uploads?limit=24'); }
  catch (e) { return; }
  _uploadsCache = data.uploads || [];
  PICKERS.forEach(key => {
    const els = pickerEls(key);
    if (!els.recentStrip) return;
    if (!_uploadsCache.length) {
      els.recentWrap.style.display = 'none';
      return;
    }
    els.recentWrap.style.display = '';
    const currentPath = els.hidden.value;
    els.recentStrip.innerHTML = _uploadsCache.map(u => `
      <img class="picker-recent-thumb${u.path === currentPath ? ' selected' : ''}"
           src="${escapeHtml(u.url)}"
           data-path="${escapeHtml(u.path)}"
           title="${escapeHtml(u.name)} · ${u.size_kb} KB · ${escapeHtml(u.mtime)}"
           alt="">
    `).join('');
    els.recentStrip.querySelectorAll('img').forEach(img => {
      img.addEventListener('click', () => pickerSetImage(key, img.dataset.path));
    });
  });
}

// ====== Format helpers ======
function fmtMem(m) { return `${m.used_gb.toFixed(1)} / ${m.total_gb.toFixed(0)} GB · swap ${m.swap_gb.toFixed(1)}`; }
function fmtMin(s) { if (!s || s < 0) return '—'; const m = Math.floor(s/60); const sec = Math.round(s%60); return m > 0 ? `${m}m ${sec}s` : `${sec}s`; }
function snippet(s, n = 70) { if (!s) return ''; s = s.replace(/\s+/g,' ').trim(); return s.length > n ? s.slice(0, n-1)+'…' : s; }
function escapeHtml(s) { if (!s) return ''; return s.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }

async function api(path, method = 'GET', body = null) {
  const opts = { method };
  if (body) {
    opts.body = body instanceof FormData ? new URLSearchParams(body) : body;
    opts.headers = { 'Content-Type': 'application/x-www-form-urlencoded' };
  }
  const r = await fetch(path, opts);
  if (!r.ok && r.status !== 409) throw new Error(`${path}: ${r.status}`);
  return r.status === 409 ? { error: 'busy' } : r.json().catch(() => ({}));
}

// ====== Poll ======
// Cache of the latest /status response so non-poll callers (setMode,
// setQuality) can refresh tier-gated UI without waiting for the next tick.
let LAST_STATUS = null;

async function poll() {
  let s;
  const url = '/status' + (filterMode === 'hidden' ? '?include_hidden=1' : '');
  try { s = await (await fetch(url)).json(); } catch (e) { return; }
  LAST_STATUS = s;

  // Memory
  const m = s.memory;
  const memPill = document.getElementById('memPill');
  memPill.innerHTML = `<span class="dot"></span>${fmtMem(m)}`;
  let memCls = 'pill-good';
  if (m.swap_gb > 8 || m.pressure_pct > 90) memCls = 'pill-danger';
  else if (m.swap_gb > 4 || m.pressure_pct > 75) memCls = 'pill-warn';
  memPill.className = 'pill ' + memCls;

  // Comfy (hidden when not running). Drives three things in lockstep —
  // the status pill, the global Stop Comfy button, and the per-render
  // "Stop ComfyUI before render" checkbox in the form. The checkbox row
  // stays hidden when Comfy isn't running so users who don't have Comfy
  // installed never see a cryptic toggle.
  const cp = document.getElementById('comfyPill');
  const stopBtn = document.getElementById('stopComfyBtn');
  const comfyRow = document.getElementById('comfyKillRow');
  const comfyToggle = document.getElementById('stop_comfy');
  if (s.comfy_pids.length) {
    cp.innerHTML = `<span class="dot"></span>Comfy ${s.comfy_pids.join(', ')}`;
    cp.className = 'pill pill-warn'; cp.style.display = '';
    stopBtn.style.display = '';
    if (comfyRow) comfyRow.style.display = '';
  } else {
    cp.style.display = 'none';
    stopBtn.style.display = 'none';
    if (comfyRow) comfyRow.style.display = 'none';
    // When Comfy isn't running, also force the form value off so the
    // submission doesn't carry a meaningless `stop_comfy=on` server-side.
    if (comfyToggle) comfyToggle.checked = false;
  }

  // Helper
  const hp = document.getElementById('helperPill');
  if (s.helper && s.helper.alive) {
    hp.innerHTML = `<span class="dot"></span>helper warm`;
    hp.className = 'pill pill-good';
    hp.title = 'Helper subprocess is loaded with pipelines and ready.';
  } else {
    // Helper auto-respawns on the next job (see WarmHelper._ensure). "Cold"
    // is normal after the idle timeout, not an error — first job after a
    // cold start eats a ~30s pipeline-load cost.
    hp.innerHTML = `<span class="dot"></span>helper idle`;
    hp.className = 'pill';
    hp.title = 'Helper is idle (auto-exited after the idle timeout). The next queued job will respawn it; expect a one-time ~30s pipeline-load delay.';
  }

  // Tier pill — what this Mac's RAM tier allows. Click to open the
  // explanation modal. Color is informational, not warning: the tier is
  // what it is, not "wrong".
  const tp = document.getElementById('tierPill');
  if (s.tier) {
    const t = s.tier;
    const cls = t.key === 'base' ? 'pill-warn'
              : (t.key === 'pro' ? 'pill-good' : '');
    // Show the friendly label ("Compact" / "Comfortable" / "Roomy" /
    // "Studio") not the internal key. Click opens the explanation modal.
    tp.innerHTML = `<span class="dot"></span>${escapeHtml(t.label || t.key)}`;
    tp.className = 'pill ' + cls;
    tp.title = `${t.label} (${t.ram_label}) · ${t.tagline} · click for details`;
    // Apply tier-driven enabled/disabled state to mode + quality pills.
    // Done here in poll() so a tier override (env var) flips state on
    // panel restart without needing to also change a separate setMode call.
    applyTierGates(t);
  }

  // Models pill — roll-up status: base ready / Q8 ready, plus active download.
  // Renders as one of:
  //   "models ↓ Q4 12%"   while a download streams (live progress, last hf line)
  //   "models 3/3"        all on disk
  //   "models 2/3"        base ready, Q8 missing → warn color
  //   "models 0/3"        base incomplete → bad color
  const mp = document.getElementById('modelsPill');
  const dl = s.download && s.download.active ? s.download : null;
  if (dl) {
    const elapsed = Math.max(0, Math.round(s.server_now - (dl.started_ts || s.server_now)));
    mp.innerHTML = `<span class="dot"></span>↓ ${dl.key} · ${elapsed}s`;
    mp.className = 'pill pill-running';
    mp.title = `Downloading ${dl.repo_id} — ${dl.last_line || 'starting…'}`;
  } else {
    // Per-repo ready/total counts, matches what the modal shows (3 rows by
    // default: Q4 + Gemma + Q8). base_available is a roll-up bool that
    // honors the HF-id env-var short-circuit; we use it for the color
    // hint, not the count itself.
    const baseOk = s.base_available;
    const q8Ok = s.q8_available;
    const ready = s.repos_ready ?? 0;
    const total = s.repos_total ?? 0;
    mp.innerHTML = `<span class="dot"></span>models ${ready}/${total}`;
    mp.className = 'pill ' + (!baseOk ? 'pill-warn' : (q8Ok ? 'pill-good' : ''));
    mp.title = !baseOk
      ? 'Base models incomplete — click to download'
      : (q8Ok ? 'All models on disk' : 'Q8 not installed (optional — needed for High quality + FFLF)');
  }
  // If the modal is open, refresh its rows on each poll so progress updates.
  if (document.getElementById('modelsModal').style.display !== 'none') {
    refreshModelsModal({ silent: true });
  }
  // Inline models card — top-of-form, big, can't miss it. State logic
  // lives in updateModelsCard so we don't bloat poll() further.
  updateModelsCard(s);

  // Queue pill + tab badge
  const qp = document.getElementById('queuePill');
  qp.innerHTML = `<span class="dot"></span>queue ${s.queue.length}${s.paused ? ' · paused' : ''}`;
  qp.className = 'pill ' + (s.paused ? 'pill-warn' : (s.queue.length ? 'pill-running' : ''));
  const qb = document.getElementById('queueBadge');
  if (s.queue.length) { qb.textContent = s.queue.length; qb.style.display = ''; } else { qb.style.display = 'none'; }

  // Job pill
  const jp = document.getElementById('jobPill');
  if (s.running && s.current) {
    const elapsed = Math.max(0, Math.round(s.server_now - s.current.started_ts));
    jp.innerHTML = `<span class="dot"></span>${s.current.params.label || s.current.params.mode} · ${elapsed}s`;
    jp.className = 'pill pill-running';
  } else {
    jp.innerHTML = `<span class="dot"></span>idle`;
    jp.className = 'pill';
  }

  document.getElementById('pauseBtn').textContent = s.paused ? 'Resume queue' : 'Pause queue';

  // Q8 / High enable
  const highBtn = document.getElementById('qualityHigh');
  const highSub = document.getElementById('highSub');
  if (s.q8_available) {
    highBtn.classList.remove('disabled');
    highSub.textContent = 'Q8 + TeaCache';
  } else {
    highBtn.classList.add('disabled');
    const missing = (s.q8_missing || []).length;
    highSub.textContent = missing > 0 && missing < 6 ? `Q8 downloading · ${missing} files left` : 'Q8 not installed';
    if (document.getElementById('quality').value === 'high') setQuality('standard');
  }

  // Keyframe (FFLF) requires Q8 — server enforces it (see run_job_inner). The
  // UI was previously silently downgrading the user to Standard when they
  // picked keyframe with Q8 missing, then the server would 500 on submit.
  // Disable Generate + show a clear reason while in that state.
  const genBtn = document.getElementById('genBtn');
  if (currentMode === 'keyframe' && !s.q8_available) {
    genBtn.disabled = true;
    const left = (s.q8_missing || []).length;
    genBtn.title = left > 0 && left < 6
      ? `Keyframe (FFLF) needs Q8 — ${left} file(s) still downloading.`
      : 'Keyframe (FFLF) needs the Q8 model. Click "Download Q8 (~25 GB)" in Pinokio.';
    genBtn.textContent = 'Generate · Q8 required';
  } else if (genBtn.disabled && genBtn.textContent.startsWith('Generate · Q8')) {
    // Restore — only do so if WE were the ones who disabled it, otherwise
    // some future code path that disables Generate for a different reason
    // would get clobbered here.
    genBtn.disabled = false;
    genBtn.title = '';
    genBtn.textContent = 'Generate';
  }

  // Now card
  const nowCard = document.getElementById('nowCard');
  const fill = document.getElementById('progressFill');
  if (s.running && s.current) {
    nowCard.classList.remove('idle', 'failed');
    const elapsed = Math.max(0, s.server_now - s.current.started_ts);
    const avg = s.avg_elapsed_sec || 420;
    const pct = Math.min(99, Math.round(elapsed / avg * 100));
    fill.style.width = pct + '%';
    const lastLog = s.log.slice(-1)[0] || '';
    nowCard.querySelector('.ttl').textContent = snippet(s.current.params.label || s.current.params.prompt, 80);
    nowCard.querySelector('.meta').innerHTML =
      `${s.current.params.mode} · ${s.current.params.width}×${s.current.params.height} · ${s.current.params.frames}f · <strong>${fmtMin(elapsed)}</strong> elapsed${avg ? ' / ~'+fmtMin(avg)+' avg' : ''}` +
      (lastLog ? `<br><span style="color:var(--muted)">${escapeHtml(lastLog.split(']').slice(1).join(']').trim().slice(0,100))}</span>` : '');
  } else {
    // Idle state. If the LAST history entry was a failure (helper crash,
    // OOM, etc.) surface it loud-and-clear here — otherwise users like
    // cocktailpeanut just see "Idle" and assume "the panel did nothing."
    // We hold the failure visible until the user starts a new job.
    fill.style.width = '0%';
    const last = (s.history || [])[0];
    const showFailure = last && last.status === 'failed' && !s.queue.length;
    if (showFailure) {
      nowCard.classList.remove('idle');
      nowCard.classList.add('failed');
      // Translate cryptic engine errors into actionable user guidance.
      // "helper died mid-job (no event)" is the SIGKILL-by-jetsam
      // signature on memory-pressured Macs — the helper subprocess gets
      // killed by the OS for using too much RAM and we never get an
      // event back. Tell the user how to recover instead of leaving them
      // with the engine wording.
      const raw = (last.error || 'unknown error');
      const rawLower = raw.toLowerCase();
      let friendly, hint;
      if (rawLower.includes('sigkill')) {
        friendly = 'Helper killed by the OS — out of memory (jetsam).';
        hint = 'Close memory-heavy apps (Chrome, Slack, iOS Simulator) and try again, ' +
               'or switch Quality to Draft (about half the RAM).';
      } else if (rawLower.includes('sigsegv') || rawLower.includes('sigbus')) {
        friendly = 'Helper crashed at the native level (MLX/Metal fault).';
        hint = 'Share the crashlog at ~/Library/Logs/DiagnosticReports/python3.11_*.crash ' +
               'on github.com/mrbizarro/phosphene/issues so we can fix it.';
      } else if (rawLower.includes('sigabrt')) {
        friendly = 'Helper hit a C-level assertion and aborted.';
        hint = 'Share the crashlog at ~/Library/Logs/DiagnosticReports/python3.11_*.crash ' +
               'on github.com/mrbizarro/phosphene/issues.';
      } else if (rawLower.includes('helper exited from') || rawLower.includes('helper pipe closed') ||
                 rawLower.includes('helper died') || rawLower.includes('helper exited')) {
        friendly = 'Helper exited unexpectedly.';
        hint = 'Check the log for the last "step:*" breadcrumb (tells us which ' +
               'phase died). If memory-pressured, close other apps and retry.';
      } else if (rawLower.includes('q8') || rawLower.includes('keyframe')) {
        friendly = 'This mode needs the Q8 model.';
        hint = raw;
      } else {
        friendly = 'Job failed.';
        hint = raw;
      }
      nowCard.querySelector('.ttl').innerHTML =
        `<span style="color: var(--danger, #f85149)">⚠ ${escapeHtml(friendly)}</span>`;
      nowCard.querySelector('.meta').innerHTML =
        `<span style="color: var(--muted)">${escapeHtml(snippet(last.params.label || last.params.prompt, 80))}</span>` +
        ` <span style="color: var(--muted)">· ${escapeHtml(last.params.mode)} · ${last.params.width}×${last.params.height}</span>` +
        `<br><span style="color: var(--text)">${escapeHtml(hint)}</span>`;
    } else {
      nowCard.classList.add('idle');
      nowCard.classList.remove('failed');
      nowCard.querySelector('.ttl').textContent = s.paused ? 'Paused' : 'Idle';
      nowCard.querySelector('.meta').textContent = s.paused
        ? 'Worker paused — current job (if any) finishes, queue waits for resume.'
        : (s.queue.length ? 'Worker about to pick up next queued job.' : 'No jobs queued. Generate something on the left.');
    }
  }

  // Logs
  const log = document.getElementById('log');
  const wasNearBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 60;
  log.textContent = s.log.length ? s.log.join('\n') : 'No log yet.';
  if (wasNearBottom) log.scrollTop = log.scrollHeight;

  // Queue list
  const ql = document.getElementById('queueList');
  if (!s.queue.length) ql.innerHTML = '<li class="empty-state"><span></span><span>Queue empty</span><span></span><span></span></li>';
  else ql.innerHTML = s.queue.map((j, i) => `
    <li>
      <span class="pos">#${i+1}</span>
      <span class="ttl" title="${escapeHtml(j.params.prompt)}">${escapeHtml(j.params.label || snippet(j.params.prompt, 60))}</span>
      <span class="params">${j.params.mode} · ${j.params.width}×${j.params.height} · ${j.params.frames}f</span>
      <button title="Remove" onclick="removeJob('${j.id}')">×</button>
    </li>`).join('');

  // History — failed jobs show the error inline in the title slot, so
  // users can see WHY without having to scroll the log to find it.
  const hl = document.getElementById('historyList');
  if (!s.history.length) hl.innerHTML = '<li class="empty-state"><span></span><span>No history yet</span><span></span><span></span></li>';
  else hl.innerHTML = s.history.slice(0, 20).map(j => {
    const titleText = escapeHtml(j.params.label || snippet(j.params.prompt, 60));
    const titleAttr = escapeHtml(j.params.prompt || '');
    let titleHtml;
    if (j.status === 'failed' && j.error) {
      titleHtml = `${titleText} ` +
        `<span class="err-inline" title="${escapeHtml(j.error)}">— ${escapeHtml(snippet(j.error, 70))}</span>`;
    } else {
      titleHtml = titleText;
    }
    return `
    <li class="${j.status}">
      <span class="badge">${j.status}</span>
      <span class="ttl" title="${titleAttr}">${titleHtml}</span>
      <span class="params">${fmtMin(j.elapsed_sec)} · ${j.finished_at ? j.finished_at.slice(11) : ''}</span>
      <span></span>
    </li>`;
  }).join('');

  // Outputs / carousel
  if (JSON.stringify(currentOutputs) !== JSON.stringify(s.outputs)) {
    currentOutputs = s.outputs;
    renderCarousel();
    if (!activePath && currentOutputs.length) selectOutput(currentOutputs[0].path);
    const sel = document.getElementById('extendSrcSelect');
    sel.innerHTML = '<option value="">— pick an output below or paste a path —</option>' +
      currentOutputs.slice(0, 40).map(o => `<option value="${escapeHtml(o.path)}">${escapeHtml(o.name)}</option>`).join('');
  }
  document.getElementById('filterHidden').textContent = `Hidden${s.hidden_count ? ' ('+s.hidden_count+')' : ''}`;
  document.getElementById('carouselTitle').textContent = filterMode === 'hidden' ? 'Hidden outputs' : `Outputs · ${currentOutputs.length}`;
}

function setFilter(mode) {
  filterMode = mode;
  document.getElementById('filterAll').classList.toggle('active', mode === 'visible');
  document.getElementById('filterHidden').classList.toggle('active', mode === 'hidden');
  poll();
}

function renderCarousel() {
  const el = document.getElementById('carousel');
  if (!currentOutputs.length) { el.innerHTML = '<div class="empty-msg">No outputs in this view yet.</div>'; return; }
  el.innerHTML = currentOutputs.map(o => `
    <div class="car-card${o.hidden ? ' hidden-card' : ''}${o.path === activePath ? ' active' : ''}"
         data-path="${escapeHtml(o.path)}" onclick="selectOutput('${escapeHtml(o.path)}')">
      <video src="/file?path=${encodeURIComponent(o.path)}#t=0.5" preload="metadata" muted></video>
      <div class="info">
        <div class="name" title="${escapeHtml(o.name)}">${escapeHtml(o.name)}</div>
        <div class="sub">${o.mtime.slice(11,16)} · ${o.size_mb.toFixed(1)} MB</div>
      </div>
      <div class="row-btns">
        <button onclick="event.stopPropagation(); ${o.hidden ? 'unhide' : 'hide'}('${escapeHtml(o.path)}')">${o.hidden ? 'Show' : 'Hide'}</button>
        <button onclick="event.stopPropagation(); useAsExtendSourcePath('${escapeHtml(o.path)}')">Extend</button>
      </div>
    </div>`).join('');
}

function selectOutput(path) {
  activePath = path;
  document.querySelectorAll('.car-card').forEach(el => el.classList.toggle('active', el.dataset.path === path));
  const wrap = document.getElementById('playerWrap');
  wrap.classList.remove('empty');
  wrap.innerHTML = `<video controls autoplay src="/file?path=${encodeURIComponent(path)}"></video>`;
  const o = currentOutputs.find(x => x.path === path);
  document.getElementById('playerMeta').style.display = '';
  document.getElementById('playerName').innerHTML = o ? `<strong>${escapeHtml(o.name)}</strong> · ${o.mtime} · ${o.size_mb.toFixed(1)} MB` : '';
  document.getElementById('loadParamsBtn').disabled = !(o && o.has_sidecar);
}

async function hide(path) { await fetch('/output/hide?path='+encodeURIComponent(path),{method:'POST'}); currentOutputs = []; poll(); }
async function unhide(path) { await fetch('/output/show?path='+encodeURIComponent(path),{method:'POST'}); currentOutputs = []; poll(); }
function hideActive() { if (activePath) hide(activePath); }

function useAsExtendSourcePath(path) {
  setMode('extend');
  document.getElementById('video_path').value = path;
  document.getElementById('extendSrcSelect').value = path;
  updateDerived();
  document.querySelector('aside.form-pane').scrollTop = 0;
}
function useAsExtendSource() { if (!activePath) return alert('Pick an output first.'); useAsExtendSourcePath(activePath); }

async function loadParams() {
  if (!activePath) return;
  const r = await fetch('/sidecar?path='+encodeURIComponent(activePath));
  if (!r.ok) return;
  const data = await r.json();
  const p = data.params;
  if (p.mode === 'extend') setMode('extend');
  else if (p.mode === 'keyframe') setMode('keyframe');
  else if (p.mode === 'i2v_clean_audio' || p.mode === 'i2v') { setMode('i2v'); document.getElementById('i2vMode').value = p.mode; document.getElementById('mode').value = p.mode; }
  else setMode('t2v');
  document.getElementById('prompt').value = p.prompt || '';
  if (p.width) document.getElementById('width').value = p.width;
  if (p.height) document.getElementById('height').value = p.height;
  if (p.frames) { document.getElementById('frames').value = p.frames; document.getElementById('duration').value = framesToDuration(p.frames); }
  if (p.steps) document.getElementById('steps').value = p.steps;
  if (p.seed != null) document.getElementById('seed').value = p.seed;
  // Image / start / end go through pickerSetImage so the preview tile
  // and recent-strip selection state update along with the hidden input.
  if (p.image)       pickerSetImage('image', p.image, { snapAspect: false });
  if (p.start_image) pickerSetImage('start_image', p.start_image, { snapAspect: false });
  if (p.end_image)   pickerSetImage('end_image', p.end_image, { snapAspect: false });
  if (p.audio) document.getElementById('audio').value = p.audio;
  // Extend-specific: restore source video path
  if (p.video_path) document.getElementById('video_path').value = p.video_path;
  if (p.label) document.getElementById('preset_label').value = p.label;
  for (const [k, a] of Object.entries(ASPECTS)) {
    if (a.w === p.width && a.h === p.height) { setAspect(k); break; }
  }
  updateDerived();
}

async function removeJob(id) { await fetch('/queue/remove?id='+encodeURIComponent(id),{method:'POST'}); poll(); }
async function togglePause() {
  const s = await (await fetch('/status')).json();
  await api(s.paused ? '/queue/resume' : '/queue/pause', 'POST');
  poll();
}

// ====== Tabs ======
document.querySelectorAll('.tabs button[data-tab]').forEach(b => b.onclick = () => {
  document.querySelectorAll('.tabs button[data-tab]').forEach(x => x.classList.toggle('active', x === b));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.toggle('show', t.id === 'tab-'+b.dataset.tab));
});

// ====== Batch modal ======
function openBatch() { document.getElementById('batchModal').classList.add('show'); }
function closeBatch() { document.getElementById('batchModal').classList.remove('show'); }
async function queueBatch() {
  const fd = new FormData(document.getElementById('genForm'));
  fd.set('prompts', document.getElementById('batchPrompts').value);
  const r = await api('/queue/batch','POST',fd);
  if (r && r.error) { alert('Batch error: '+r.error); return; }
  if (r && r.added) { document.getElementById('batchPrompts').value = ''; poll(); }
}

// ====== "No music" toggle pill ======
//
// Custom pill replacing the default checkbox. Click anywhere on the pill
// to flip the hidden checkbox + reflect state in the UI (.on class drives
// the accent fill from the toggle-pill CSS). Backed by a real <input
// type=checkbox> inside the label, so FormData still picks it up the
// normal way and screen readers / keyboard nav still work.
(function () {
  const pill = document.getElementById('noMusicPill');
  const cb = document.getElementById('noMusic');
  if (!pill || !cb) return;
  const sync = () => pill.classList.toggle('on', cb.checked);
  cb.addEventListener('change', sync);
  pill.addEventListener('click', e => {
    // <label> already toggles the checkbox; we just need to refresh the
    // visual state on the next tick AFTER the native toggle has fired.
    setTimeout(sync, 0);
  });
  sync();
})();

// ====== Form submit ======
//
// "No music" toggle: appends a clear audio constraint to the prompt
// before submission so the LTX 2.3 vocoder skips the soundtrack/score it
// otherwise tends to add. Music is hard to remove cleanly from a stem
// after the fact (it shares spectral space with dialogue), so users who
// plan to score the clip themselves want voice + ambient only.
//
// We modify the FormData copy, not the textarea value — so the user's
// original prompt stays untouched in the UI.
document.getElementById('genForm').addEventListener('submit', async e => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const noMusic = document.getElementById('noMusic');
  if (noMusic && noMusic.checked) {
    const original = fd.get('prompt') || '';
    const constraint = ' Audio: voice and ambient sounds only, no music, no soundtrack, no score, no melody.';
    if (!original.toLowerCase().includes('no music')) {
      fd.set('prompt', original.trim() + constraint);
    }
  }
  await api('/queue/add','POST',fd);
  poll();
});

// ====== Inline models card (top of form) ======
// Displays what the current install needs RIGHT WHERE the user is about
// to act. Beats burying the download CTA in a header modal. State picked
// from /status: base missing → red blocker, current-mode-needs-Q8 →
// amber prompt, downloading → animated progress, all-good → hidden.
function updateModelsCard(s) {
  const card     = document.getElementById('modelsInline');
  const icon     = document.getElementById('modelsInlineIcon');
  const title    = document.getElementById('modelsInlineTitle');
  const sub      = document.getElementById('modelsInlineSub');
  const progress = document.getElementById('modelsInlineProgress');
  const fill     = document.getElementById('modelsInlineFill');
  const last     = document.getElementById('modelsInlineLast');
  const actions  = document.getElementById('modelsInlineActions');
  if (!card) return;

  const baseOk = !!s.base_available;
  const q8Ok   = !!s.q8_available;
  const dl     = s.download && s.download.active ? s.download : null;
  const tier   = s.tier || {};
  const dismissed = !!(s.settings && s.settings.models_card_dismissed);

  // Reset state classes — we set the right one below.
  card.classList.remove('state-missing', 'state-warn', 'state-downloading', 'dismissible');
  progress.style.display = 'none';

  // ----- Active download takes precedence over everything ------------------
  if (dl) {
    card.style.display = '';
    card.classList.add('state-downloading');
    icon.textContent = '↓';
    const labelByKey = { q4: 'Q4 base model', gemma: 'Gemma text encoder', q8: 'Q8 high-quality model' };
    title.textContent = `Downloading ${labelByKey[dl.key] || dl.repo_id}`;
    const elapsed = Math.max(0, Math.round((Date.now()/1000) - (dl.started_ts || 0)));
    sub.textContent = `${elapsed}s elapsed · resumable if interrupted`;
    progress.style.display = '';
    // Try to extract a percent from the last hf line (tqdm format).
    const m = (dl.last_line || '').match(/\b(\d{1,3})%/);
    fill.style.width = m ? `${Math.min(100, parseInt(m[1]))}%` : '15%';
    last.textContent = dl.last_line || 'starting…';
    actions.innerHTML = `<button class="danger" onclick="cancelDownload()">Cancel</button>`;
    return;
  }

  // ----- Base missing — hard block, the panel can't render anything --------
  if (!baseOk) {
    card.style.display = '';
    card.classList.add('state-missing');
    icon.textContent = '⚠';
    title.textContent = 'Base models needed before you can render';
    const missing = (s.base_missing || []).length;
    sub.innerHTML = `Q4 (~25 GB) and Gemma (~6 GB) are required. Click below — downloads resume if interrupted.${
      missing ? ` <span style="color:var(--muted)">(${missing} files left)</span>` : ''
    }`;
    actions.innerHTML = (s.hf_available ?? true)
      ? `<button onclick="startDownload('q4')">Download Q4 (25 GB)</button>`
      : `<button disabled title="hf binary not found — reinstall via Pinokio">hf missing</button>`;
    return;
  }

  // ----- User picked a mode that needs Q8, but Q8 isn't there --------------
  // FFLF + Extend + High quality all need Q8. Surface the CTA *only* when
  // the user is about to do one of those — no point nagging a T2V user
  // about Q8 if they'll never use it.
  // Dismissible: a user who deliberately doesn't want Q8 (storage budget,
  // they only do T2V Draft/Standard) can × this away and we'll respect it
  // until either model state changes or they re-summon the modal.
  const needsQ8 = (currentMode === 'keyframe')
                || (document.getElementById('quality').value === 'high');
  if (needsQ8 && !q8Ok && tier.allows_q8 !== false) {
    if (dismissed) { card.style.display = 'none'; return; }
    card.style.display = '';
    card.classList.add('state-warn', 'dismissible');
    icon.textContent = '⬇';
    const reason = currentMode === 'keyframe' ? 'FFLF needs the Q8 model'
                                              : 'High quality needs the Q8 model';
    title.textContent = reason;
    const missing = (s.q8_missing || []).length;
    sub.innerHTML = `Q8 (~25 GB) is a separate one-time download. Resumable.${
      missing && missing < 8 ? ` <span style="color:var(--muted)">(${missing} files left — partial install detected)</span>` : ''
    }`;
    actions.innerHTML = (s.hf_available ?? true)
      ? `<button onclick="startDownload('q8')">Download Q8 (25 GB)</button>`
      : `<button disabled>hf missing</button>`;
    return;
  }

  // ----- All good — hide the card completely -------------------------------
  // Per user feedback: the "Models ready · 3/3" status was visual noise once
  // everything was downloaded. Hide the card on full readiness; the header
  // models pill still gives a way to reopen the modal if the user wants to
  // manage repos. If state regresses (a file gets deleted, partial download
  // appears), one of the branches above re-shows it automatically.
  const allReady = baseOk && q8Ok;
  if (allReady) {
    card.style.display = 'none';
    actions.innerHTML = '';
    return;
  }
  // ----- Partial-OK quiet state ---------------------------------------------
  // Base OK but Q8 missing on a tier that supports it AND the user hasn't
  // picked a Q8-needing mode — gentle nudge in neutral colours, dismissible.
  if (dismissed) { card.style.display = 'none'; return; }
  card.style.display = '';
  card.classList.add('dismissible');
  icon.textContent = '✓';
  const ready = s.repos_ready ?? 0;
  const total = s.repos_total ?? 0;
  title.textContent = `Models ready · ${ready}/${total}`;
  const partialNote = (q8Ok && baseOk) ? '' : ` · ${total - ready} optional missing`;
  sub.innerHTML =
    `All installed weights detected${partialNote}. ` +
    `<a style="color:var(--accent-bright,#7e98ff); cursor:pointer; text-decoration:underline" onclick="openModelsModal()">Manage models →</a>`;
  actions.innerHTML = '';
}

// Persist the "user dismissed the models card" flag. POSTs to /settings
// and re-runs updateModelsCard with the latest status so the card hides
// immediately (not after the next /status poll cycle, ~5s away).
async function dismissModelsCard() {
  try {
    const fd = new URLSearchParams();
    fd.set('models_card_dismissed', 'true');
    await fetch('/settings', {
      method: 'POST',
      headers: {'Content-Type': 'application/x-www-form-urlencoded'},
      body: fd,
    });
  } catch (e) { /* best effort — UI still hides locally on next poll */ }
  // Optimistically hide right now without waiting for the poll round-trip.
  const card = document.getElementById('modelsInline');
  if (card) card.style.display = 'none';
  // Patch LAST_STATUS so subsequent updateModelsCard calls before the next
  // /status fetch agree with the on-disk state.
  if (LAST_STATUS && LAST_STATUS.settings) {
    LAST_STATUS.settings.models_card_dismissed = true;
  }
}

// ====== Tier gating ======
// Disables the FFLF / Extend mode pills and the High quality pill when
// the detected hardware tier doesn't support them. Visual state +
// tooltip + intercepted clicks. Run from the poll handler so an env
// override flips state on restart.
function applyTierGates(tier) {
  // Mode pills
  document.querySelectorAll('#modeGroup .pill-btn').forEach(b => {
    const m = b.dataset.mode;
    const allowed = (m === 'keyframe') ? tier.allows_keyframe
                  : (m === 'extend')   ? tier.allows_extend
                  : true;
    b.classList.toggle('disabled', !allowed);
    if (!allowed) {
      const need = m === 'keyframe'
        ? 'first/last-frame interpolation needs more memory than this Mac has — try Image → Video instead'
        : 'extending an existing clip needs more memory than this Mac has — try Image → Video instead';
      b.title = `Off on the ${tier.label} tier · ${need}`;
    } else {
      b.title = '';
    }
  });
  // Quality: High requires Q8. We already disable based on q8_available
  // for the no-download case; this layer enforces the RAM tier on top.
  // Both layers can disable — we OR them together via a class.
  const highBtn = document.getElementById('qualityHigh');
  if (highBtn) {
    if (!tier.allows_q8) {
      highBtn.classList.add('disabled');
      highBtn.title = `Off on the ${tier.label} tier · the high-quality model needs more memory than this Mac has`;
    } else {
      // Don't unconditionally clear .disabled — the Q8-not-installed code
      // path also sets it. Only clear if the tier is the only reason.
      // The poll() code that checks q8_available re-applies that state
      // every cycle, so this branch is safe to unset.
      highBtn.title = '';
    }
  }
}
// Intercept clicks on disabled mode pills so users get a helpful message
// instead of a broken-feeling no-op.
document.addEventListener('click', (e) => {
  const btn = e.target.closest('#modeGroup .pill-btn.disabled');
  if (btn) {
    e.stopPropagation();
    e.preventDefault();
    alert(btn.title || 'This mode is not supported on this hardware tier.');
  }
}, true);

// ====== Tier modal ======
function openTierModal() {
  const modal = document.getElementById('tierModal');
  modal.style.display = 'flex';
  // Defensive: show "loading" state immediately so the modal never appears
  // with the body completely blank (which is what happens if the panel
  // process is dead and fetch fails — looked like a "buggy bug" to a user
  // who was just kicked off a stale browser view).
  document.getElementById('tierModalTitle').textContent = 'Hardware tier';
  document.getElementById('tierModalBlurb').innerHTML = '<em>Loading…</em>';
  document.getElementById('tierCapsList').innerHTML = '';
  // Tier doesn't change at runtime — RAM is fixed at boot — so a single
  // fetch on open is plenty. No need for live polling here.
  fetch('/status').then(r => r.json()).then(s => {
    const t = s.tier || {};
    const tt = t.times || {};
    // Helper: a row is "available" if it's allowed; "max" is the friendly
    // size limit (or "Any size" / "—" when there is no limit / disabled).
    const sizeLine = (on, maxDim, fallback) => {
      if (!on) return fallback || 'Not available on this Mac';
      if (!maxDim) return 'Any size';
      return `Up to ${maxDim} pixels on the longer side`;
    };
    document.getElementById('tierModalTitle').textContent = `What this Mac can do · ${t.label || 'unknown'}`;
    document.getElementById('tierModalBlurb').innerHTML = `
      <div style="margin-bottom: 6px"><strong>${escapeHtml(t.label || '')}</strong> · ${escapeHtml(t.ram_label || '')} of memory</div>
      <div>${escapeHtml(t.blurb || '')}</div>`;
    // One row per mode/option, with three pieces of info each:
    //   - is it available? (✓ / ✗)
    //   - what's the size limit? (plain English)
    //   - how long does a typical 5-second render take? (rough estimate)
    const items = [
      {
        title: 'Text → video',
        desc: 'Type a prompt, get a clip. The default mode.',
        on: true,
        size: sizeLine(true, t.t2v_max_dim),
        time: tt.t2v_standard,
      },
      {
        title: 'Image → video',
        desc: 'Drop in a still, get it animated. Same speed as text → video.',
        on: true,
        size: sizeLine(true, t.i2v_max_dim),
        time: tt.i2v_standard,
      },
      {
        title: 'Draft (faster, smaller)',
        desc: 'Half-resolution preview to scout prompts and seeds before a full render.',
        on: true,
        size: 'Always smaller than Standard',
        time: tt.t2v_draft,
      },
      {
        title: 'High quality',
        desc: 'Bigger model, two-stage denoising, sharper faces. Needs the optional Q8 download.',
        on: !!t.allows_q8,
        size: sizeLine(!!t.allows_q8, 0, 'Needs more memory than this Mac has'),
        time: tt.high,
      },
      {
        title: 'First / last frame (FFLF)',
        desc: 'Pick a start image and an end image, the model fills the motion between.',
        on: !!t.allows_keyframe,
        size: sizeLine(!!t.allows_keyframe, t.keyframe_max_dim, 'Needs more memory than this Mac has'),
        time: tt.keyframe,
      },
      {
        title: 'Extend an existing clip',
        desc: 'Pick a video you already rendered, the model adds more time onto either end.',
        on: !!t.allows_extend,
        size: sizeLine(!!t.allows_extend, t.extend_max_dim, 'Needs more memory than this Mac has'),
        time: tt.extend,
      },
    ];
    document.getElementById('tierCapsList').innerHTML = items.map(it => `
      <li class="${it.on ? 'ready' : 'missing'}">
        <span class="icon">${it.on ? '✓' : '✗'}</span>
        <div class="meta">
          <span class="ttl">${escapeHtml(it.title)}</span>
          <span class="sub">${escapeHtml(it.desc)}</span>
          <span class="sub" style="margin-top:2px">
            <span style="color:var(--fg,#d8e0ee)">${escapeHtml(it.size)}</span>${
              it.time ? ` · <span style="color:var(--accent-bright,#7e98ff)">~ ${escapeHtml(it.time)} for a 5-second clip</span>` : ''
            }
          </span>
        </div>
        <span></span>
      </li>`).join('');
  }).catch(err => {
    // Panel might be dead, status endpoint unreachable, or response not JSON.
    // Replace the loading state with a visible error so the modal doesn't
    // look "broken" with empty content.
    document.getElementById('tierModalBlurb').innerHTML =
      '<div style="color: var(--danger, #f85149)">Could not load tier info — the panel server may have stopped responding. Check the Pinokio terminal and restart the panel if needed.</div>';
    document.getElementById('tierCapsList').innerHTML = '';
    console.error('tier modal fetch failed:', err);
  });
}
function closeTierModal() { document.getElementById('tierModal').style.display = 'none'; }

// ====== Settings modal ======
// Single-shot fetch on open (settings change rarely). The modal hydrates
// preset cards from the /settings response so the labels and blurbs
// match the server-side OUTPUT_PRESETS table — no preset content
// duplicated in JS.
let _settingsCache = null;

async function openSettingsModal() {
  const modal = document.getElementById('settingsModal');
  modal.style.display = 'flex';
  document.getElementById('settingsStatus').textContent = '';
  document.getElementById('settingsStatus').className = 'settings-status';
  try {
    const r = await fetch('/settings');
    _settingsCache = await r.json();
  } catch (e) {
    document.getElementById('settingsStatus').textContent = 'Could not load settings.';
    document.getElementById('settingsStatus').className = 'settings-status err';
    return;
  }
  const cur = _settingsCache.settings;
  const presets = _settingsCache.presets;
  // Render preset cards (Standard, Video production, Web, Custom).
  // Display order matches the typical user journey: most users want
  // Standard, video pros pick Video production, web preview folks pick Web.
  const order = ['standard', 'archival', 'web'];
  const grid = document.getElementById('settingsPresets');
  grid.innerHTML = '';
  for (const key of order) {
    const p = presets[key];
    const active = cur.output_preset === key ? 'active' : '';
    const card = document.createElement('label');
    card.className = `preset-card ${active}`;
    card.dataset.preset = key;
    card.innerHTML = `
      <input type="radio" name="settingsPreset" value="${key}" ${cur.output_preset === key ? 'checked' : ''}>
      <div class="preset-text">
        <div class="preset-label">${escapeHtml(p.label)}</div>
        <div class="preset-blurb">${escapeHtml(p.blurb)}</div>
        <div class="preset-spec">pix_fmt=${p.pix_fmt} · crf=${p.crf}</div>
      </div>`;
    card.addEventListener('click', () => selectPreset(key));
    grid.appendChild(card);
  }
  // Custom row.
  const customActive = cur.output_preset === 'custom' ? 'active' : '';
  const custom = document.createElement('label');
  custom.className = `preset-card ${customActive}`;
  custom.dataset.preset = 'custom';
  custom.innerHTML = `
    <input type="radio" name="settingsPreset" value="custom" ${cur.output_preset === 'custom' ? 'checked' : ''}>
    <div class="preset-text">
      <div class="preset-label">Custom</div>
      <div class="preset-blurb">Set pix_fmt and CRF manually. For unusual workflows: 10-bit HDR, format-specific delivery, or non-standard CRF for video production work.</div>
      <div class="preset-spec">pix_fmt=${cur.output_pix_fmt} · crf=${cur.output_crf}</div>
    </div>`;
  custom.addEventListener('click', () => selectPreset('custom'));
  grid.appendChild(custom);
  // Pre-fill custom inputs with current values
  document.getElementById('settingsPixFmt').value = cur.output_pix_fmt;
  document.getElementById('settingsCrfRange').value = cur.output_crf;
  document.getElementById('settingsCrfNum').value = cur.output_crf;
  document.getElementById('settingsCustomSection').style.display =
    cur.output_preset === 'custom' ? 'block' : 'none';

  // Token rows. We never receive the actual key from the server (the
  // /settings GET returns has_X booleans only), so we display either
  // "set" with an empty placeholder input, or "—" with the placeholder.
  // Inputs start empty on every modal open; user pastes when they want
  // to change.
  setTokenStatus('civitaiKey', cur.has_civitai_key);
  setTokenStatus('hfToken', cur.has_hf_token);
  document.getElementById('civitaiKeyInput').value = '';
  document.getElementById('hfTokenInput').value = '';
  document.getElementById('civitaiKeyClear').style.display = cur.has_civitai_key ? '' : 'none';
  document.getElementById('hfTokenClear').style.display = cur.has_hf_token ? '' : 'none';
}

function setTokenStatus(prefix, isSet, dirty) {
  const el = document.getElementById(prefix + 'Status');
  if (!el) return;
  el.classList.remove('set', 'dirty');
  if (dirty) {
    el.textContent = '✎ unsaved';
    el.classList.add('dirty');
  } else if (isSet) {
    el.textContent = '✓ saved';
    el.classList.add('set');
  } else {
    el.textContent = 'not set';
  }
}

function onTokenInput(which) {
  const prefix = which === 'civitai' ? 'civitaiKey' : 'hfToken';
  const inp = document.getElementById(prefix + 'Input');
  setTokenStatus(prefix, false, !!inp.value);
}

function toggleTokenVisibility(inputId, btn) {
  const inp = document.getElementById(inputId);
  if (!inp) return;
  if (inp.type === 'password') {
    inp.type = 'text';
    btn.textContent = 'hide';
  } else {
    inp.type = 'password';
    btn.textContent = 'show';
  }
}

// Hits /civitai/test or /hf/test which makes an auth-required upstream
// request with the saved key. Lets the user verify their token works
// without paying for a 300 MB download to find out it doesn't.
async function testToken(which) {
  const path = which === 'civitai' ? '/civitai/test' : '/hf/test';
  const resultId = which === 'civitai' ? 'civitaiTestResult' : 'hfTestResult';
  const result = document.getElementById(resultId);
  if (!result) return;
  const original = result.innerHTML;
  result.textContent = 'Testing…';
  result.style.color = 'var(--muted)';
  try {
    const r = await fetch(path);
    const data = await r.json();
    if (data.ok) {
      result.innerHTML = `<strong style="color: var(--success, #3fb950)">✓</strong> ${escapeHtml(data.message)}`;
    } else {
      result.innerHTML = `<strong style="color: var(--danger, #f85149)">✗</strong> ${escapeHtml(data.error)}`;
    }
  } catch (e) {
    result.innerHTML = `<strong style="color: var(--danger, #f85149)">✗</strong> Network error: ${escapeHtml(e.message || String(e))}`;
  }
}

async function clearToken(which) {
  const fd = new FormData();
  if (which === 'civitai') fd.set('civitai_api_key', '');
  if (which === 'hf')      fd.set('hf_token', '');
  try {
    // urlencoded body — see applySettings for why.
    const r = await fetch('/settings', {
      method: 'POST',
      headers: {'Content-Type': 'application/x-www-form-urlencoded'},
      body: new URLSearchParams(fd),
    });
    const data = await r.json();
    if (!r.ok || data.error) {
      alert('Could not clear: ' + (data.error || `HTTP ${r.status}`));
      return;
    }
    // Refresh the modal so the status flips back to "not set".
    openSettingsModal();
  } catch (e) {
    alert('Network error: ' + (e.message || e));
  }
}

function closeSettingsModal() {
  document.getElementById('settingsModal').style.display = 'none';
}

function selectPreset(key) {
  document.querySelectorAll('#settingsPresets .preset-card').forEach(c => {
    c.classList.toggle('active', c.dataset.preset === key);
    const r = c.querySelector('input[type="radio"]');
    if (r) r.checked = (c.dataset.preset === key);
  });
  document.getElementById('settingsCustomSection').style.display =
    key === 'custom' ? 'block' : 'none';
  // Clear status so it doesn't claim "saved" after a fresh selection.
  document.getElementById('settingsStatus').textContent = '';
  document.getElementById('settingsStatus').className = 'settings-status';
}

async function applySettings() {
  const status = document.getElementById('settingsStatus');
  const btn = document.getElementById('settingsApplyBtn');
  status.textContent = 'Saving…';
  status.className = 'settings-status';
  btn.disabled = true;
  // Read which preset is selected. Custom path also sends pix_fmt + crf.
  const checked = document.querySelector('#settingsPresets input[type="radio"]:checked');
  const preset = checked ? checked.value : 'standard';
  const fd = new FormData();
  fd.set('output_preset', preset);
  if (preset === 'custom') {
    fd.set('output_pix_fmt', document.getElementById('settingsPixFmt').value);
    fd.set('output_crf', document.getElementById('settingsCrfNum').value);
  }
  // Tokens — only send a key when the input has a value. Empty input
  // means "leave as-is" (clearing is explicit via the Clear button).
  // This protects against accidentally wiping a saved key by clicking
  // Apply on an unchanged form.
  const civInput = document.getElementById('civitaiKeyInput').value.trim();
  if (civInput) fd.set('civitai_api_key', civInput);
  const hfInput = document.getElementById('hfTokenInput').value.trim();
  if (hfInput)  fd.set('hf_token', hfInput);
  try {
    // Convert FormData → URLSearchParams so the body is sent as
    // x-www-form-urlencoded — the panel's parse_qs only understands
    // that wire format, NOT the multipart/form-data fetch sends by
    // default with FormData. This bug silently turned every settings
    // save into a no-op (server saw empty payload) until caught.
    const r = await fetch('/settings', {
      method: 'POST',
      headers: {'Content-Type': 'application/x-www-form-urlencoded'},
      body: new URLSearchParams(fd),
    });
    const data = await r.json();
    if (!r.ok || data.error) {
      status.textContent = data.error || `HTTP ${r.status}`;
      status.className = 'settings-status err';
      btn.disabled = false;
      return;
    }
    status.textContent = data.helper_restarted
      ? 'Saved. Helper restarted — takes effect on the next render.'
      : 'Saved.';
    status.className = 'settings-status ok';
    btn.disabled = false;
    // Refresh cache so a re-open shows the new values without a stale flash.
    _settingsCache = { ...(_settingsCache || {}), settings: data.settings };
  } catch (e) {
    status.textContent = 'Network error: ' + (e.message || e);
    status.className = 'settings-status err';
    btn.disabled = false;
  }
}

// ====== HDR toggle pill (header pill behavior, same as No-music) ======
(function () {
  const pill = document.getElementById('hdrPill');
  const cb = document.getElementById('hdr');
  if (!pill || !cb) return;
  const sync = () => pill.classList.toggle('on', cb.checked);
  cb.addEventListener('change', sync);
  pill.addEventListener('click', () => setTimeout(sync, 0));
  sync();
})();

// ====== CivitAI NSFW toggle pill (mirrors HDR toggle UX) ======
(function () {
  const pill = document.getElementById('civitaiNsfwPill');
  const cb = document.getElementById('civitaiNsfw');
  if (!pill || !cb) return;
  const sync = () => pill.classList.toggle('on', cb.checked);
  cb.addEventListener('change', sync);
  pill.addEventListener('click', () => setTimeout(sync, 0));
  sync();
})();

// ====== LoRA picker ======
//
// State model: an in-memory list of LoRA entries the user has added.
// Adding can come from "Use" on a CivitAI install or from clicking a
// row in the local list. Each entry:
//   { path, name, strength, trigger_words, civitai_url }
// On every change we mirror the list into the hidden #lorasJson field
// so make_job's parse_loras_from_form picks them up at submit time.

let _activeLoras = [];   // [{path, name, strength, trigger_words, ...}]
let _knownUserLoras = []; // last list_user_loras() snapshot, for the picker

function _serializeLoras() {
  // What the helper actually needs is path + strength. Keep the rest in
  // the in-memory list for UI rendering, drop it on the wire.
  const slim = _activeLoras.map(l => ({ path: l.path, strength: l.strength }));
  document.getElementById('lorasJson').value = JSON.stringify(slim);
  // Update summary count
  const summary = document.getElementById('lorasSummaryCount');
  if (summary) {
    summary.textContent = _activeLoras.length === 0
      ? 'none active'
      : `${_activeLoras.length} active`;
  }
}

function addLoraToActive(entry) {
  // Idempotent: same path twice = update strength only.
  const existing = _activeLoras.find(l => l.path === entry.path);
  if (existing) {
    existing.strength = entry.strength;
  } else {
    _activeLoras.push(entry);
  }
  renderLorasList();
  _serializeLoras();
}

function removeLoraFromActive(path) {
  _activeLoras = _activeLoras.filter(l => l.path !== path);
  renderLorasList();
  _serializeLoras();
}

function setLoraStrength(path, strength) {
  const e = _activeLoras.find(l => l.path === path);
  if (!e) return;
  e.strength = Math.max(-2, Math.min(2, parseFloat(strength) || 0));
  _serializeLoras();
}

async function refreshLoras() {
  let data;
  try {
    data = await (await fetch('/loras')).json();
  } catch (e) {
    return;
  }
  _knownUserLoras = data.user || [];
  // Update displayed loras dir
  if (data.loras_dir) {
    const dirEl = document.getElementById('lorasDir');
    if (dirEl) dirEl.textContent = data.loras_dir;
  }
  // If a row was previously active but the file is gone (deleted on
  // disk), drop it from the active set so we don't submit a stale path.
  const knownPaths = new Set(_knownUserLoras.map(l => l.path));
  _activeLoras = _activeLoras.filter(l =>
    knownPaths.has(l.path) || l.path.includes('/'));   // keep HF ids (no dir slash)
  renderLorasList();
  _serializeLoras();
}

function renderLorasList() {
  const wrap = document.getElementById('lorasList');
  const empty = document.getElementById('lorasEmpty');
  if (!wrap) return;
  // Combine: user-installed LoRAs (from /loras) plus any active LoRAs
  // that aren't user-installed (HF repo paths, e.g. from the HDR toggle).
  const rows = [];
  const seen = new Set();
  for (const ul of _knownUserLoras) {
    const active = _activeLoras.find(a => a.path === ul.path);
    seen.add(ul.path);
    rows.push({
      path: ul.path,
      name: ul.name,
      trigger_words: ul.trigger_words || [],
      recommended_strength: ul.recommended_strength || 1.0,
      filename: ul.filename,
      preview_url: ul.preview_url,
      preview_type: ul.preview_type,
      civitai_url: ul.civitai_url,
      size_bytes: ul.size_bytes,
      active: !!active,
      strength: active ? active.strength : (ul.recommended_strength || 1.0),
      kind: 'user',
    });
  }
  for (const a of _activeLoras) {
    if (seen.has(a.path)) continue;
    rows.push({
      path: a.path,
      name: a.name || a.path,
      trigger_words: a.trigger_words || [],
      recommended_strength: 1.0,
      filename: null,
      preview_url: null,
      preview_type: null,
      civitai_url: null,
      size_bytes: null,
      active: true,
      strength: a.strength,
      kind: 'remote',
    });
  }

  if (rows.length === 0) {
    wrap.innerHTML = '';
    if (empty) empty.style.display = '';
    return;
  }
  if (empty) empty.style.display = 'none';

  wrap.innerHTML = rows.map(r => loraCardHtml(r)).join('');
}

// Build a single LoRA card. Pulled out of renderLorasList so the markup
// stays scannable. The card shows (top to bottom):
//   1. Preview thumbnail (16:9 video or image; "no preview" placeholder).
//   2. Corner buttons: open on CivitAI ↗, delete ×.
//   3. Title row (filename ellipsis-truncated, HF badge for remote).
//   4. Trigger word chips. Click → append to the prompt textarea so the
//      user doesn't have to remember the exact spelling. Most LTX LoRAs
//      need their trigger word in the prompt or they barely activate.
//   5. Strength row: range slider + number input (-2..2, 0.05 step).
//   6. Active toggle pill.
function loraCardHtml(r) {
  const pathHtml = escapeHtml(r.path);
  const nameHtml = escapeHtml(r.name);
  const nameAttr = JSON.stringify(r.name).replace(/"/g, '&quot;');
  // Preview: <video> for .mp4 (autoplay/muted/loop = animated GIF feel),
  // <img> otherwise, "no preview" placeholder when missing.
  let thumbHtml;
  if (!r.preview_url) {
    thumbHtml = `<div class="lora-thumb-empty">no preview</div>`;
  } else if (r.preview_type === 'video' || /\.mp4($|\?)/i.test(r.preview_url)) {
    thumbHtml = `<video class="lora-thumb" src="${escapeHtml(r.preview_url)}"
                        autoplay muted loop playsinline preload="metadata"></video>`;
  } else {
    thumbHtml = `<img class="lora-thumb" src="${escapeHtml(r.preview_url)}" alt="" loading="lazy">`;
  }
  // Corner actions: open on CivitAI when we know the page, then delete/×.
  const cornerLinks = [];
  if (r.civitai_url) {
    cornerLinks.push(`<a class="lora-corner-btn" href="${escapeHtml(r.civitai_url)}" target="_blank" rel="noopener" title="Open on CivitAI to read instructions / examples">↗</a>`);
  }
  if (r.kind === 'user') {
    cornerLinks.push(`<button class="lora-corner-btn danger" type="button" title="Delete from disk"
                              onclick="deleteLora('${pathHtml}', '${escapeHtml(r.name)}')">×</button>`);
  } else {
    cornerLinks.push(`<button class="lora-corner-btn" type="button" title="Remove from active set"
                              onclick="removeLoraFromActive('${pathHtml}')">×</button>`);
  }
  // Trigger chips. If a LoRA has no triggers (e.g. style-only LoRAs that
  // activate purely from style transfer) say so explicitly so the user
  // doesn't think the metadata is missing.
  const trigs = (r.trigger_words || []).slice(0, 8);
  const chipsHtml = trigs.length
    ? trigs.map(w => {
        const wAttr = JSON.stringify(w).replace(/"/g, '&quot;');
        return `<span class="trigger-chip" title="Click to append to prompt"
                       onclick="appendTriggerToPrompt(${wAttr})">${escapeHtml(w)}</span>`;
      }).join('')
    : `<span class="trigger-chip empty">no trigger word — applies as a style</span>`;

  return `
    <div class="lora-card ${r.active ? 'active' : ''}" data-path="${pathHtml}">
      <div class="lora-thumb-wrap">
        ${thumbHtml}
        <div class="lora-corner-actions">${cornerLinks.join('')}</div>
      </div>
      <div class="lora-body">
        <div class="lora-name" title="${pathHtml}">
          ${nameHtml}
          ${r.kind === 'remote' ? '<span class="badge">HF</span>' : ''}
        </div>
        <div class="trigger-chips">${chipsHtml}</div>
        <div class="lora-strength-row">
          <label>strength</label>
          <input type="range" min="-2" max="2" step="0.05" value="${r.strength}"
                 ${r.active ? '' : 'disabled'}
                 oninput="this.nextElementSibling.value = this.value; setLoraStrength('${pathHtml}', this.value)">
          <input type="number" min="-2" max="2" step="0.05" value="${r.strength}"
                 ${r.active ? '' : 'disabled'}
                 oninput="this.previousElementSibling.value = this.value; setLoraStrength('${pathHtml}', this.value)">
        </div>
        <div class="lora-toggle-row">
          <label class="lora-toggle">
            <input type="checkbox" ${r.active ? 'checked' : ''}
                   onchange="toggleLora('${pathHtml}', this.checked, ${r.recommended_strength}, ${nameAttr})">
            <span class="dot"></span>
            <span>${r.active ? 'Active' : 'Inactive'}</span>
          </label>
          ${r.civitai_url
            ? `<a class="lora-meta-link" href="${escapeHtml(r.civitai_url)}" target="_blank" rel="noopener">read on CivitAI →</a>`
            : ''}
        </div>
      </div>
    </div>`;
}

function toggleLora(path, on, recommended, name) {
  if (on) {
    addLoraToActive({ path, strength: recommended, name });
  } else {
    removeLoraFromActive(path);
  }
}

// Append a LoRA's trigger word to the prompt textarea. Most LTX LoRAs
// only fully activate when their trigger word is somewhere in the prompt,
// and asking users to remember + type a string like "DISPSTYLE" exactly
// is friction. Click the chip → it goes in. Idempotent: if the word is
// already present (case-insensitive substring), do nothing so users can
// click freely without piling duplicates.
function appendTriggerToPrompt(word) {
  const ta = document.getElementById('prompt');
  if (!ta) return;
  const cur = ta.value || '';
  if (cur.toLowerCase().includes(String(word).toLowerCase())) {
    // Brief visual ping so the click feels acknowledged even though we
    // didn't change anything — otherwise users repeat-click thinking
    // it's broken.
    ta.classList.add('flash-ok');
    setTimeout(() => ta.classList.remove('flash-ok'), 250);
    return;
  }
  // If the user has typed nothing, drop the trigger in alone. Otherwise
  // prepend to the existing prompt: many LoRA authors put the trigger
  // FIRST in their examples, and quality often degrades when the trigger
  // is buried at the end past 20+ tokens of unrelated context.
  if (cur.trim() === '') {
    ta.value = String(word);
  } else {
    ta.value = String(word) + ', ' + cur;
  }
  ta.focus();
  ta.dispatchEvent(new Event('input', { bubbles: true }));
}

async function deleteLora(path, name) {
  if (!confirm(`Delete the LoRA file for "${name}" from disk? This is permanent.`)) {
    return;
  }
  const fd = new FormData();
  fd.set('path', path);
  try {
    const r = await fetch('/loras/delete', {
      method: 'POST',
      headers: {'Content-Type': 'application/x-www-form-urlencoded'},
      body: new URLSearchParams(fd),
    });
    const data = await r.json();
    if (!r.ok || !data.ok) {
      alert('Delete failed: ' + (data.error || `HTTP ${r.status}`));
      return;
    }
    removeLoraFromActive(path);
    refreshLoras();
  } catch (e) {
    alert('Delete failed: ' + (e.message || e));
  }
}

// ====== CivitAI modal ======

let _civitaiCursor = '';
let _civitaiSearching = false;

function openCivitaiModal() {
  document.getElementById('civitaiModal').style.display = 'flex';
  // Pull /loras to populate the dir text + auth-status warning. Cheap
  // call (no I/O beyond the loras dir scan + an env-var read).
  fetch('/loras').then(r => r.json()).then(d => {
    const dirEl = document.getElementById('civitaiTargetDir');
    if (dirEl && d.loras_dir) dirEl.textContent = d.loras_dir;
    const hint = document.getElementById('civitaiAuthHint');
    if (hint) hint.style.display = d.civitai_auth ? 'none' : 'block';
  }).catch(() => {});
  document.getElementById('civitaiQuery').value = '';
  _civitaiCursor = '';
  civitaiSearch();
}

function closeCivitaiModal() {
  document.getElementById('civitaiModal').style.display = 'none';
}

async function civitaiSearch() {
  if (_civitaiSearching) return;
  _civitaiSearching = true;
  const grid = document.getElementById('civitaiGrid');
  const status = document.getElementById('civitaiStatus');
  const loadMore = document.getElementById('civitaiLoadMore');
  status.textContent = '';
  status.className = 'civitai-status-line';
  grid.innerHTML = '<div class="hint">Loading…</div>';
  loadMore.style.display = 'none';
  _civitaiCursor = '';
  try {
    const params = new URLSearchParams();
    const q = document.getElementById('civitaiQuery').value.trim();
    if (q) params.set('query', q);
    if (document.getElementById('civitaiNsfw').checked) params.set('nsfw', 'true');
    params.set('limit', '24');
    const r = await fetch('/civitai/search?' + params.toString());
    const data = await r.json();
    if (data.error) {
      grid.innerHTML = '';
      status.textContent = data.error;
      status.className = 'civitai-status-line err';
      return;
    }
    renderCivitaiGrid(data.items, /* append */ false);
    _civitaiCursor = data.next_cursor || '';
    if (data.has_more) loadMore.style.display = '';
    if ((data.items || []).length === 0) {
      grid.innerHTML = `<div class="hint">No LTX 2.3 LoRAs match "${escapeHtml(q || '')}"${document.getElementById('civitaiNsfw').checked ? '' : ' (try Show NSFW for more)'}.</div>`;
    }
  } catch (e) {
    status.textContent = 'Network error: ' + (e.message || e);
    status.className = 'civitai-status-line err';
  } finally {
    _civitaiSearching = false;
  }
}

async function civitaiLoadMore() {
  if (_civitaiSearching || !_civitaiCursor) return;
  _civitaiSearching = true;
  const loadMore = document.getElementById('civitaiLoadMore');
  loadMore.disabled = true;
  loadMore.textContent = 'Loading…';
  try {
    const params = new URLSearchParams();
    const q = document.getElementById('civitaiQuery').value.trim();
    if (q) params.set('query', q);
    if (document.getElementById('civitaiNsfw').checked) params.set('nsfw', 'true');
    params.set('limit', '24');
    params.set('cursor', _civitaiCursor);
    const r = await fetch('/civitai/search?' + params.toString());
    const data = await r.json();
    if (data.error) {
      document.getElementById('civitaiStatus').textContent = data.error;
      document.getElementById('civitaiStatus').className = 'civitai-status-line err';
      return;
    }
    renderCivitaiGrid(data.items, /* append */ true);
    _civitaiCursor = data.next_cursor || '';
    loadMore.style.display = data.has_more ? '' : 'none';
  } catch (e) {
    document.getElementById('civitaiStatus').textContent = 'Network error: ' + (e.message || e);
    document.getElementById('civitaiStatus').className = 'civitai-status-line err';
  } finally {
    _civitaiSearching = false;
    loadMore.disabled = false;
    loadMore.textContent = 'Load more';
  }
}

function renderCivitaiGrid(items, append) {
  const grid = document.getElementById('civitaiGrid');
  if (!append) grid.innerHTML = '';
  if (!items || items.length === 0) return;
  const frag = document.createDocumentFragment();
  for (const it of items) {
    const card = document.createElement('div');
    card.className = 'civitai-card';
    const sizeMb = it.size_kb ? (it.size_kb / 1024).toFixed(1) : '?';
    const dl = it.downloads ? new Intl.NumberFormat().format(it.downloads) : '?';
    const triggers = (it.trigger_words || []).slice(0, 3).join(', ');
    // LTX is a video model so most LoRAs ship animated previews. Render
    // <video> for videos (autoplay muted loop = looks like an animated
    // GIF, no user interaction needed) and <img> for stills. Both share
    // the .preview class so the card height is stable while images
    // load. CivitAI's CDN sets `Access-Control-Allow-Origin: *` so
    // cross-origin loads work without a panel-side proxy.
    let previewHtml;
    if (!it.preview_url) {
      previewHtml = `<div class="preview-empty">no preview</div>`;
    } else if (it.preview_type === 'video' || /\.mp4($|\?)/i.test(it.preview_url)) {
      previewHtml = `<video class="preview" src="${escapeHtml(it.preview_url)}"
                            autoplay muted loop playsinline preload="metadata"></video>`;
    } else {
      previewHtml = `<img class="preview" src="${escapeHtml(it.preview_url)}" alt="" loading="lazy">`;
    }
    card.innerHTML = `
      ${previewHtml}
      <div class="body">
        <div class="ttl" title="${escapeHtml(it.name)}">${escapeHtml(it.name)}</div>
        <div class="meta">
          <span>by ${escapeHtml(it.creator)}</span>
          <span>↓ ${dl}</span>
          <span>${sizeMb} MB</span>
          ${it.nsfw ? '<span class="nsfw-badge">NSFW</span>' : ''}
        </div>
        ${triggers ? `<div class="meta"><span title="trigger words">trigger: ${escapeHtml(triggers)}</span></div>` : ''}
        ${it.civitai_url
          ? `<div class="meta"><a class="civitai-source-link" href="${escapeHtml(it.civitai_url)}" target="_blank" rel="noopener" title="Open the original CivitAI page — usage notes, examples, comments">Read instructions on CivitAI ↗</a></div>`
          : ''}
      </div>
      <div class="actions">
        <button type="button" class="primary-btn" data-id="${it.id}">Install</button>
      </div>`;
    const btn = card.querySelector('button[data-id]');
    btn.addEventListener('click', () => civitaiInstall(btn, it));
    frag.appendChild(card);
  }
  grid.appendChild(frag);
}

async function civitaiInstall(btn, item) {
  btn.disabled = true;
  const origLabel = btn.textContent;
  btn.textContent = 'Downloading…';
  const fd = new FormData();
  fd.set('download_url', item.download_url);
  fd.set('meta', JSON.stringify(item));
  try {
    const r = await fetch('/civitai/download', {
      method: 'POST',
      headers: {'Content-Type': 'application/x-www-form-urlencoded'},
      body: new URLSearchParams(fd),
    });
    const data = await r.json();
    if (!r.ok || !data.ok) {
      const status = document.getElementById('civitaiStatus');
      status.textContent = `Download failed: ${data.error || 'HTTP ' + r.status}`;
      status.className = 'civitai-status-line err';
      btn.disabled = false;
      btn.textContent = origLabel;
      return;
    }
    btn.textContent = data.skipped ? 'Already installed ✓' : 'Installed ✓';
    const status = document.getElementById('civitaiStatus');
    status.textContent = data.skipped
      ? `Already in ${data.path} — auto-enabled below.`
      : `Saved to ${data.path}. Auto-enabled below.`;
    status.className = 'civitai-status-line ok';
    // Refresh the local picker so the new LoRA appears, then auto-enable.
    // Auto-enable applies on BOTH the fresh-download AND the
    // already-installed paths — clicking Install on a CivitAI card
    // should always result in "this LoRA is now usable in the next
    // render," regardless of whether it was already on disk. Earlier
    // build only auto-enabled fresh downloads, leaving repeat clicks
    // looking like a no-op even though the file was sitting right
    // there in the picker.
    await refreshLoras();
    addLoraToActive({
      path: data.path,
      name: data.name || item.name,
      strength: item.recommended_strength || 1.0,
      trigger_words: item.trigger_words || [],
    });
    // Open the LoRAs disclosure so the user sees the entry without
    // hunting for it after the modal closes.
    const det = document.getElementById('lorasDetails');
    if (det) det.open = true;
  } catch (e) {
    document.getElementById('civitaiStatus').textContent = 'Network error: ' + (e.message || e);
    document.getElementById('civitaiStatus').className = 'civitai-status-line err';
    btn.disabled = false;
    btn.textContent = origLabel;
  }
}

// Boot: load the local LoRA list on page load so the picker isn't empty
// when the user expands it for the first time.
document.addEventListener('DOMContentLoaded', () => { refreshLoras(); });

// ====== Models modal ======
// Opens to /models snapshot. While open, the main poll() refreshes the
// list every cycle so download progress appears live. Each row shows:
//   ✓ ready (green)             — all repo files present
//   ◐ partial (amber)           — some files there, some missing (e.g. interrupted)
//   ⊘ missing (red)             — nothing on disk
//   ↻ downloading (blue, anim)  — hf is currently fetching this repo
function openModelsModal() {
  document.getElementById('modelsModal').style.display = 'flex';
  refreshModelsModal();
}
function closeModelsModal() {
  document.getElementById('modelsModal').style.display = 'none';
}
async function refreshModelsModal({ silent = false } = {}) {
  const list = document.getElementById('modelsList');
  const hint = document.getElementById('modelsHint');
  const foot = document.getElementById('modelsFoot');
  let data;
  try { data = await api('/models'); }
  catch (e) {
    if (!silent) hint.textContent = 'Failed to load models. Panel might be restarting — try again.';
    return;
  }
  const repos = data.repos || [];
  const active = data.active_download;
  hint.innerHTML = data.hf_available
    ? `Each row shows what's on disk. Click <b>Download</b> to fetch missing files via <code>hf download</code>; progress streams to the log at the bottom of the page.`
    : `<span style="color:var(--warning,#d29922)">⚠ <code>hf</code> not found</span> — this Pinokio install doesn't have <code>huggingface_hub&gt;=1.0</code> in the venv. Run Update from Pinokio, then come back.`;
  const rows = repos.map(r => {
    let cls, icon, statusText, btnHtml;
    if (active && active.key === r.key) {
      cls = 'downloading';
      icon = '↻';
      const elapsed = Math.max(0, Math.round((Date.now()/1000) - (active.started_ts || 0)));
      const last = active.last_line ? `<div class="progress">${escapeHtml(active.last_line)}</div>` : '';
      statusText = `Downloading · ${elapsed}s${last}`;
      btnHtml = `<button class="ghost" onclick="cancelDownload()">Cancel</button>`;
    } else if (r.complete) {
      cls = 'ready'; icon = '✓';
      // `where: 'hf_cache'` means the files were resolved from
      // ~/.cache/huggingface/ rather than the canonical mlx_models/
      // dir. Common on manual / dev installs that pre-existed Pinokio
      // and pulled the model via `huggingface-cli` or first-run helper.
      const tag = r.where === 'hf_cache' ? 'HF cache' : 'local';
      statusText = `Ready · ${r.total_files} files · ~${r.size_gb || '?'} GB · ${tag}`;
      btnHtml = `<button class="ghost" disabled>Installed</button>`;
    } else if (r.present_files > 0) {
      cls = 'partial'; icon = '◐';
      const left = r.total_files - r.present_files;
      statusText = `Partial · ${r.present_files}/${r.total_files} files · ${left} missing — resume to finish`;
      btnHtml = data.hf_available
        ? `<button onclick="startDownload('${escapeHtml(r.key)}')" ${active ? 'disabled' : ''}>Resume</button>`
        : `<button disabled>Resume</button>`;
    } else {
      cls = 'missing'; icon = '⊘';
      statusText = `Not installed · ~${r.size_gb || '?'} GB`;
      btnHtml = data.hf_available
        ? `<button onclick="startDownload('${escapeHtml(r.key)}')" ${active ? 'disabled' : ''}>Download</button>`
        : `<button disabled>Download</button>`;
    }
    const kindBadge = r.kind === 'optional'
      ? `<span style="color:var(--muted)">optional</span>`
      : `<span style="color:var(--success,#3fb950)">required</span>`;
    return `
      <li class="${cls}">
        <span class="icon">${icon}</span>
        <div class="meta">
          <span class="ttl">${escapeHtml(r.name)} · ${kindBadge}</span>
          <span class="sub">${escapeHtml(r.repo_id)} → ${escapeHtml(r.local_dir)}</span>
          <span class="sub">${statusText}${r.blurb ? ' · ' + escapeHtml(r.blurb) : ''}</span>
        </div>
        ${btnHtml}
      </li>`;
  }).join('');
  list.innerHTML = rows || `<li class="empty-state">No model manifest found — required_files.json is missing or unreadable.</li>`;
  // Footer summarises required vs optional counts.
  const reqRepos = repos.filter(r => r.kind !== 'optional');
  const optRepos = repos.filter(r => r.kind === 'optional');
  const reqReady = reqRepos.filter(r => r.complete).length;
  const optReady = optRepos.filter(r => r.complete).length;
  foot.innerHTML = `
    <div>Required: ${reqReady}/${reqRepos.length} ready &nbsp;·&nbsp; Optional: ${optReady}/${optRepos.length} ready</div>
    <div style="margin-top:4px">Tip: downloads resume on retry — closing this dialog mid-download keeps it running in the background.</div>`;
}
async function startDownload(key) {
  let res;
  try {
    res = await api('/models/download', 'POST', `repo_key=${encodeURIComponent(key)}`);
  } catch (e) {
    alert('Download failed to start: ' + (e?.message || e));
    return;
  }
  // The api() helper coerces 409 (busy) to { error: 'busy' } — surface that
  // to the user instead of silently no-op'ing the click.
  if (res && res.error) {
    alert(`Can't start download: ${res.error}`);
  }
  refreshModelsModal();
  poll();
}
async function cancelDownload() {
  if (!confirm('Cancel the active download? Partial files stay on disk; clicking Download/Resume later picks up where you left off.')) return;
  try { await api('/models/cancel', 'POST'); } catch (e) {}
  refreshModelsModal();
}

// ====== Init ======
setInterval(poll, 1500);
poll();
setMode('t2v');
setQuality('standard');
setAspect('landscape');
updateDerived();

// Wire the picker components (I2V image + FFLF start/end) and seed the
// "Recent uploads" strip. The strip is shared across all three pickers,
// so dropping a new image in one slot makes it instantly clickable in
// the other two.
PICKERS.forEach(pickerWire);
refreshUploadsStrip();
// Refresh the strip whenever a render finishes (queue/history changes
// don't fire here), and whenever the user opens FFLF — covers the case
// where they uploaded something via I2V, then switched to FFLF.
document.querySelectorAll('#modeGroup .pill-btn').forEach(b => b.addEventListener('click', refreshUploadsStrip));
</script>
</body>
</html>

"""


if __name__ == "__main__":
    OUTPUT.mkdir(parents=True, exist_ok=True)
    UPLOADS.mkdir(parents=True, exist_ok=True)
    load_hidden()
    load_queue()
    threading.Thread(target=worker_loop, daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"LTX MLX Studio: http://127.0.0.1:{PORT}", flush=True)
    print(f"queue: {len(STATE['queue'])} pending, hidden: {len(HIDDEN_PATHS)}", flush=True)
    try:
        server.serve_forever()
    finally:
        HELPER.kill()
        caffeinate_off()
