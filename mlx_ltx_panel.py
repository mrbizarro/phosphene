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
import importlib.util
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
import urllib.error
import urllib.parse
import urllib.request
from urllib.parse import parse_qs, quote, urlparse

# Agentic Flows runtime — chat-driven shot planner + queue submitter.
# The agent module is self-contained (pure stdlib, imports nothing from
# this panel); we wire it up via a PanelOps callback object built at the
# end of this module just before the HTTP handler.
from agent import engine as agent_engine
from agent import image_engine as agent_image_engine
from agent import local_server as agent_local_server
from agent import prompts as agent_prompts
from agent import runtime as agent_runtime
from agent import tools as agent_tools

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
def _resolve_helper_python() -> Path:
    """Find the helper-subprocess Python interpreter.

    Order: explicit env var → manual-install convention (`.venv/`) →
    Pinokio convention (`env/`) → last-resort fallback so the panel
    boots even if neither exists (the helper will fail loudly when the
    first job tries to spawn it). Auto-detection means non-Pinokio
    launches (manual `python3.11 mlx_ltx_panel.py` for testing) work
    without needing LTX_HELPER_PYTHON set.
    """
    explicit = os.environ.get("LTX_HELPER_PYTHON")
    if explicit and Path(explicit).is_file():
        return Path(explicit)
    for sub in (".venv/bin/python3.11", "env/bin/python3.11"):
        p = MLX / sub
        if p.is_file():
            return p
    # No working interpreter found — return the manual-install default.
    # Job submission will surface a clear error referring users to the
    # install instructions.
    return MLX / ".venv/bin/python3.11"


HELPER_PYTHON = _resolve_helper_python()
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
# State files (queue + hidden + settings) live in <ROOT>/state/, which
# Pinokio's fs.link maps to a virtual drive that survives Reset. The
# directory is auto-created on first run; existing pre-Y1.004 installs
# get a one-time migration of root-level files into state/ at startup
# (see _migrate_state_dir below) so users don't lose their queue.
STATE_DIR = Path(os.environ.get("LTX_STATE_DIR", str(ROOT / "state")))
QUEUE_FILE = STATE_DIR / "panel_queue.json"
HIDDEN_FILE = STATE_DIR / "panel_hidden.json"
HELPER_IDLE_TIMEOUT = int(os.environ.get("LTX_HELPER_IDLE_TIMEOUT", "1800"))
HELPER_LOW_MEMORY = os.environ.get("LTX_HELPER_LOW_MEMORY", "true")
FPS = 24


def _optional_bool_env(name: str) -> bool | None:
    v = os.environ.get(name, "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return None

# ---- Profile (production vs dev) ---------------------------------------------
# Y1.015: support running a local "dev" Pinokio panel side-by-side with the
# production install. The dev panel checks out the `dev` branch of the same
# repo and binds to a different port, so commits can be tested in Pinokio
# before merging to main. Detection is automatic from the git branch — no
# config to maintain. PHOSPHENE_PROFILE env var overrides for testing.
def _detect_profile() -> str:
    forced = os.environ.get("PHOSPHENE_PROFILE", "").strip().lower()
    if forced in ("dev", "production"):
        return forced
    try:
        out = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL, timeout=2,
        ).decode("utf-8", "replace").strip()
        if out == "dev":
            return "dev"
    except Exception:
        pass
    return "production"


PROFILE = _detect_profile()
# Sharp/LTX latent upscale is an unsafe lab experiment, not a release feature.
# The official LTX pipelines use the latent upsampler only as the middle of a
# two-stage flow: upscale latent -> denoise/refine at full resolution -> decode.
# Running upsampler -> decode directly distorted faces and produced bad tail
# frames in release tests, so keep it hidden unless explicitly opted in.
MODEL_UPSCALE_ENABLED = _optional_bool_env("LTX_ENABLE_MODEL_UPSCALE") is True
PIPERSR_UPSCALE_ENABLED = _optional_bool_env("LTX_ENABLE_PIPERSR") is not False and importlib.util.find_spec("pipersr") is not None
VERSION_CHECK_ENABLED = _optional_bool_env("PHOSPHENE_DISABLE_VERSION_CHECK") is not True
# Default port: 8198 production, 8199 dev — so both panels can run side by
# side. LTX_PORT env var still overrides if the user wants something else.
DEFAULT_PORT = 8199 if PROFILE == "dev" else 8198
PORT = int(os.environ.get("LTX_PORT", str(DEFAULT_PORT)))
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

SETTINGS_FILE = STATE_DIR / "panel_settings.json"
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
        # Spicy mode — gates NSFW LoRA visibility. Default OFF (kid-safe).
        # When OFF: the CivitAI browser hides its "Show NSFW" toggle, the
        # server forces nsfw=false on CivitAI requests, and any incoming
        # NSFW-flagged items are filtered out as a defense-in-depth pass.
        # When ON: existing per-session "Show NSFW" toggle is exposed and
        # works as before. The Settings UI requires explicit confirmation
        # to flip OFF→ON so kids / casual visitors can't enable it by a
        # stray click.
        "spicy_mode": False,
    }


def _ensure_state_dir() -> None:
    """Create <ROOT>/state/ if it doesn't exist. Idempotent. Called at
    startup before any state-file load so loaders can assume the dir is
    there. When fs.link is wired in install.js, this directory is a
    symlink into the persistent drive — Pinokio Reset deletes the
    panel install but leaves the linked target intact."""
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        sys.stderr.write(f"WARN: could not create {STATE_DIR} ({exc})\n")


def _migrate_state_dir() -> None:
    """One-time migration: move pre-Y1.004 root-level state files into
    <ROOT>/state/. Existing users who Update from Y1.003- to Y1.004+
    have panel_settings.json + panel_queue.json + panel_hidden.json at
    the repo root; we move them so the new SETTINGS_FILE / QUEUE_FILE /
    HIDDEN_FILE constants find them. Idempotent — only moves files that
    exist at the source AND don't exist at the destination, so a user
    who already has Y1.004 state in place doesn't get clobbered."""
    _ensure_state_dir()
    pairs = [
        (ROOT / "panel_settings.json", SETTINGS_FILE),
        (ROOT / "panel_queue.json", QUEUE_FILE),
        (ROOT / "panel_hidden.json", HIDDEN_FILE),
    ]
    for src, dst in pairs:
        try:
            if src.exists() and src.is_file() and not dst.exists():
                # Move (not copy) so old root-level files don't linger
                # and re-confuse a second migration pass.
                shutil.move(str(src), str(dst))
                sys.stderr.write(f"NOTE: migrated {src.name} → state/\n")
        except OSError as exc:
            sys.stderr.write(f"WARN: could not migrate {src.name} ({exc})\n")


# Run the migration once at module import, before any settings load. Cheap
# (just a few existence checks in the no-op case) and side-effect-safe.
_migrate_state_dir()


def _load_settings() -> dict:
    """Read panel_settings.json. Missing file → return + write the default
    so first-run users get the sensible Standard preset. Corrupt file →
    fall back to defaults but DON'T overwrite (preserves the bad file for
    forensic inspection if it was edited by hand)."""
    if not SETTINGS_FILE.exists():
        defaults = _settings_defaults()
        try:
            _save_settings(defaults)
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
    fd, tmp = tempfile.mkstemp(prefix="panel_settings.", dir=str(STATE_DIR))
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

    if "spicy_mode" in patch:
        # Same urlencoded-bool coercion as models_card_dismissed.
        v = patch["spicy_mode"]
        if isinstance(v, bool):
            out["spicy_mode"] = v
        else:
            out["spicy_mode"] = str(v).strip().lower() in ("1", "true", "yes", "on")

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
        "spicy_mode": bool(s.get("spicy_mode", False)),
    }


_SETTINGS = _load_settings()


def get_settings() -> dict:
    with _SETTINGS_LOCK:
        return dict(_SETTINGS)


def output_codec_settings() -> dict[str, str]:
    """Current render codec settings, normalized for panel-side ffmpeg passes."""
    defaults = OUTPUT_PRESETS[DEFAULT_OUTPUT_PRESET]
    s = get_settings()
    return {
        "preset": str(s.get("output_preset") or DEFAULT_OUTPUT_PRESET),
        "pix_fmt": str(s.get("output_pix_fmt") or defaults["pix_fmt"]),
        "crf": str(s.get("output_crf") or defaults["crf"]),
    }


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


# ---- Version check / update notifier ----------------------------------------
# Phosphene ships fixes often (multiple commits a day in active periods) and
# users keep telling us "I clicked Update but I don't see anything new" — by
# the time their feedback lands, we've usually pushed three more commits.
# This module surfaces a small "Update available" pill in the header when the
# local install is behind origin/main, so users notice mismatch immediately
# instead of finding out later via a friend's tweet.
#
# How it works:
#   1. At startup we read the local commit SHA via `git rev-parse HEAD` and
#      branch name via `git rev-parse --abbrev-ref HEAD`. Stored once.
#   2. A daemon thread polls the GitHub commits API every 30 minutes (no
#      auth, public repo, well within unauthenticated rate limit).
#   3. We list `origin/main` commits newest-first and find the index of the
#      local SHA. If found at index N, we are N commits behind. If not found
#      in the first 30 commits, we report "30+ behind".
#   4. The /version endpoint returns the snapshot; the UI renders a pill
#      that links to the version modal listing each unseen commit.
#
# Suppressed cases (no pill rendered):
#   - User is on a non-main branch (local dev / fork)
#   - User has uncommitted local changes (in-progress work; we'd be wrong
#     to call them "behind")
#   - We can't reach the GitHub API (offline; suppress silently — we don't
#     bug users with red error toasts every 30 minutes)

_VERSION_LOCK = threading.Lock()
_VERSION_REPO_OWNER = "mrbizarro"
_VERSION_REPO_NAME = "phosphene"
_VERSION_POLL_INTERVAL_SEC = 30 * 60          # 30 minutes between checks
_VERSION_STARTUP_DELAY_SEC = 30                # don't compete with boot
_VERSION_STATE: dict = {
    "local_sha": None,            # 40-char hex of HEAD at panel start
    "local_short": None,           # 7-char display form
    "local_version": None,         # human label from /VERSION file (e.g. "Y1.001")
    "local_branch": None,          # branch name (e.g. "main") or "(detached)"
    "local_dirty": False,          # uncommitted changes in the working tree
    "remote_sha": None,            # latest sha on origin/main per GitHub API
    "remote_short": None,
    "remote_version": None,        # /VERSION file at origin/main (raw fetch)
    "behind_by": 0,                # number of commits between local and remote (0 = current)
    "behind_more_than": False,     # True when behind > 30 (we cap the API response)
    "commits_ahead": [],           # list of {sha, short, message, author, date} for the dropdown
    "checked_ts": None,            # epoch seconds of last successful poll
    "error": None,                 # last failure reason (None when last check OK)
    "suppress_reason": None,       # set when we deliberately skip checking (dev mode, dirty, etc.)
    # Pull state — populated only when the user clicks the magic button on
    # the pill while behind. Kept on _VERSION_STATE so the UI can render a
    # "restart needed" pill without a separate state store.
    "pull_state": "idle",          # idle | pulling | pulled | error
    "pull_message": None,          # human-readable result line (last git output line, or error)
    "pull_pulled_to_short": None,  # SHA we ended up at after the pull
    "pull_pulled_to_version": None,# VERSION label after the pull
    "pull_requires_full_update": False,  # True if the diff touched deps/patches
}


def _git_capture(args: list[str], cwd: Path = ROOT) -> str | None:
    """Run a git subcommand and return stripped stdout (or None on any failure).
    No exceptions ever escape — we never want a missing git binary to crash
    the panel boot."""
    try:
        out = subprocess.check_output(
            ["git"] + args, cwd=str(cwd), stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8", "replace").strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def _read_local_version() -> str | None:
    """Return the contents of <repo>/VERSION as a single stripped string,
    or None if the file is missing / unreadable. We use this to surface a
    human-friendly label like 'Y1.001' instead of raw 7-char SHAs in the
    header pill. Older checkouts that predate this file fall back to the
    short SHA in the UI without breaking."""
    try:
        v = (ROOT / "VERSION").read_text().strip()
        return v or None
    except Exception:
        return None


def _fetch_raw_text(url: str, timeout: int = 10) -> str | None:
    """GET a public URL and return its body as a stripped string, or None
    on any error. Used for the raw VERSION-file fetch from origin/main on
    GitHub — `https://raw.githubusercontent.com/.../main/VERSION`. No
    auth, no rate-limit headers worth honouring at this volume (one
    request per 30-min poll)."""
    import urllib.request as _urlreq
    try:
        req = _urlreq.Request(url, headers={"User-Agent": "phosphene-panel-version-check"})
        with _urlreq.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", "replace").strip() or None
    except Exception:
        return None


def _detect_local_install_state() -> None:
    """Populate _VERSION_STATE.local_* fields. Called once at startup; result
    doesn't change while the panel is running (no auto-pull on the user's
    behalf). Pinokio's Update step kills + restarts the panel, so a fresh
    detect runs after every Update."""
    sha = _git_capture(["rev-parse", "HEAD"])
    branch = _git_capture(["rev-parse", "--abbrev-ref", "HEAD"]) or "(unknown)"
    # `git status --porcelain` empty = clean tree.
    porcelain = _git_capture(["status", "--porcelain"])
    dirty = bool(porcelain and porcelain.strip())
    local_version = _read_local_version()
    with _VERSION_LOCK:
        _VERSION_STATE["local_sha"] = sha
        _VERSION_STATE["local_short"] = sha[:7] if sha else None
        _VERSION_STATE["local_version"] = local_version
        _VERSION_STATE["local_branch"] = branch
        _VERSION_STATE["local_dirty"] = dirty
        # Suppress remote checks when the user is clearly running their own
        # variant — we'd be wrong to nag them about being "behind."
        if not VERSION_CHECK_ENABLED:
            _VERSION_STATE["suppress_reason"] = "disabled by PHOSPHENE_DISABLE_VERSION_CHECK"
        elif not sha:
            _VERSION_STATE["suppress_reason"] = "not a git checkout"
        elif branch != "main":
            _VERSION_STATE["suppress_reason"] = f"on branch '{branch}', not main"
        elif dirty:
            _VERSION_STATE["suppress_reason"] = "local changes uncommitted"
        else:
            _VERSION_STATE["suppress_reason"] = None


def _fetch_remote_commits(limit: int = 30) -> list[dict] | None:
    """GET the latest N commits on origin/main from the public GitHub API.
    Returns a list of raw commit dicts (newest first), or None on any error.
    Caller is expected to handle None gracefully — we don't show errors to
    the user every 30 minutes when their wifi is flaky."""
    import urllib.request as _urlreq
    import urllib.error as _urlerr
    url = (f"https://api.github.com/repos/{_VERSION_REPO_OWNER}/"
           f"{_VERSION_REPO_NAME}/commits?sha=main&per_page={limit}")
    req = _urlreq.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "phosphene-panel-version-check",
    })
    try:
        with _urlreq.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if not isinstance(data, list):
            return None
        return data
    except (_urlerr.URLError, _urlerr.HTTPError, OSError, ValueError, TimeoutError):
        return None


def _check_remote_once() -> None:
    """Run one remote check, mutate _VERSION_STATE accordingly. Idempotent.
    Safe to call from a background thread or from a /version/check endpoint.
    Re-detects local state (HEAD, branch, dirty) at the top of every call so
    a tree that was dirty when the panel started but is clean now (e.g. user
    just committed) gets picked up without needing a panel restart. The
    earlier build only ran _detect_local_install_state once at startup,
    which left the suppress_reason stale across the panel's lifetime."""
    _detect_local_install_state()
    with _VERSION_LOCK:
        suppress = _VERSION_STATE["suppress_reason"]
        local_sha = _VERSION_STATE["local_sha"]
    if suppress:
        return                             # never poll in suppressed states
    commits = _fetch_remote_commits(limit=30)
    if commits is None:
        with _VERSION_LOCK:
            _VERSION_STATE["error"] = "could not reach github.com"
        return
    if not commits:
        with _VERSION_LOCK:
            _VERSION_STATE["error"] = "github returned no commits"
        return
    remote_sha = commits[0].get("sha")
    if not remote_sha:
        with _VERSION_LOCK:
            _VERSION_STATE["error"] = "github response missing sha"
        return

    # Find local SHA in the list. If it's at index N, we are N commits behind.
    behind_by = None
    for i, c in enumerate(commits):
        if c.get("sha") == local_sha:
            behind_by = i
            break
    if behind_by is None:
        # Local is older than what fits in the window OR has been rebased
        # away. Either way, "30+" is a fine UX answer — the modal will say
        # "many updates available, please run Pinokio Update."
        behind_by = len(commits)
        more = True
    else:
        more = False

    # Slim each commit dict down to what the modal renders. Keep only the
    # commits the user is missing.
    ahead: list[dict] = []
    for c in commits[:behind_by]:
        commit = c.get("commit") or {}
        author = (commit.get("author") or {}).get("name") or "unknown"
        date = (commit.get("author") or {}).get("date") or ""
        msg = (commit.get("message") or "").splitlines()[0]   # first line only
        sha = c.get("sha") or ""
        ahead.append({
            "sha": sha,
            "short": sha[:7],
            "message": msg,
            "author": author,
            "date": date,
        })

    # Fetch the upstream VERSION file in parallel with the commits API so
    # the pill can show the human-friendly label (Y1.NNN). This is a raw
    # GET on raw.githubusercontent.com — no rate limit concern at our
    # 30-min cadence. Tolerate missing-file silently: older origin/main
    # commits predate VERSION, and we still have the SHA fallback.
    remote_version = _fetch_raw_text(
        f"https://raw.githubusercontent.com/{_VERSION_REPO_OWNER}/"
        f"{_VERSION_REPO_NAME}/main/VERSION"
    )
    with _VERSION_LOCK:
        _VERSION_STATE["remote_sha"] = remote_sha
        _VERSION_STATE["remote_short"] = remote_sha[:7]
        _VERSION_STATE["remote_version"] = remote_version
        _VERSION_STATE["behind_by"] = behind_by
        _VERSION_STATE["behind_more_than"] = more
        _VERSION_STATE["commits_ahead"] = ahead
        _VERSION_STATE["checked_ts"] = time.time()
        _VERSION_STATE["error"] = None


def version_check_loop() -> None:
    """Daemon thread entry: detect local state once, then poll the remote
    every _VERSION_POLL_INTERVAL_SEC. First poll happens after a 30s
    delay so we don't compete with boot-time work."""
    _detect_local_install_state()
    time.sleep(_VERSION_STARTUP_DELAY_SEC)
    while True:
        try:
            _check_remote_once()
        except Exception as exc:                          # belt + braces
            sys.stderr.write(f"WARN: version check raised: {exc}\n")
        time.sleep(_VERSION_POLL_INTERVAL_SEC)


def get_version_state() -> dict:
    """Snapshot of _VERSION_STATE for the /version endpoint. Returns a copy
    so the caller can't mutate the live state under the lock."""
    with _VERSION_LOCK:
        return dict(_VERSION_STATE)


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
    so the moment hf finishes, /status flips to complete + the UI updates.

    Y1.022 — enables hf_transfer (Rust accelerator, ~5-10× faster on big
    repos) via env var, threads the user's HF token through if set
    (anonymous HF is throttled to ~50 KB/s and stalls Q8), and wraps the
    download in a 3-attempt retry loop with exponential backoff. hf is
    resumable, so retries pick up from where the previous attempt
    stopped. User cancellations (DOWNLOAD["active"] flipping to False)
    break the loop early."""
    repo_id = repo["repo_id"]
    target = ROOT / repo["local_dir"]
    target.mkdir(parents=True, exist_ok=True)
    cmd = [str(HF_BIN), "download", repo_id, "--local-dir", str(target)]
    # Y1.024 — apply --include filter when the repo entry declares one. Without
    # it `hf download` grabs every file in the upstream repo. dgrauet's LTX
    # repos host duplicate transformer variants (-distilled, -distilled-1.1,
    # -dev), duplicate distilled LoRAs, and unused upscalers — turning a
    # declared 25 GB Q8 into 82 GB on disk and a 25 GB Q4 into 56 GB.
    # `download_include` is the explicit allowlist of patterns we ship to
    # the user; everything else stays on the Hub. See required_files.json
    # for the comment block on this.
    include_patterns = repo.get("download_include") or []
    for pat in include_patterns:
        cmd.extend(["--include", pat])
    if include_patterns:
        push(f"[hf] filtering {repo_id} download to {len(include_patterns)} pattern(s) — only the files the panel actually loads.")
    push(f"[hf] {repo_id} → {target} (~{repo.get('size_gb','?')} GB) — resumable")

    # Build the env once. HF_HUB_ENABLE_HF_TRANSFER=1 turns on the Rust
    # downloader; if hf_transfer isn't installed the hf CLI logs a warning
    # and falls back to the Python downloader (slower but still works).
    # HF_TOKEN unlocks faster throughput for users who configured a token
    # in Settings — anonymous HF is throttled hard.
    env = os.environ.copy()
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    hf_token = _active_hf_token()
    if hf_token:
        env["HF_TOKEN"] = hf_token
        env["HUGGING_FACE_HUB_TOKEN"] = hf_token
        push(f"[hf] using authenticated download (HF token configured) — ~10× faster than anonymous.")
    else:
        push(f"[hf] no HF token configured — downloads run on the throttled anonymous tier. Set one in Settings → Hugging Face token for ~10× faster downloads.")

    max_attempts = 3
    backoff_sec = 5
    try:
        for attempt in range(1, max_attempts + 1):
            with DOWNLOAD_LOCK:
                if not DOWNLOAD["active"]:
                    push(f"[hf] {repo_id} cancelled before attempt {attempt}.")
                    break
            if attempt > 1:
                push(f"[hf] {repo_id} attempt {attempt}/{max_attempts} — resuming from where the previous attempt stopped.")
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=env,
                    start_new_session=True,
                )
            except Exception as exc:
                push(f"[hf] failed to spawn hf: {exc}")
                break
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
                break
            with DOWNLOAD_LOCK:
                still_active = DOWNLOAD["active"]
            if not still_active:
                push(f"[hf] {repo_id} cancelled (exit {rc}).")
                break
            if attempt < max_attempts:
                push(f"[hf] {repo_id} attempt {attempt} failed (exit {rc}). Retrying in {backoff_sec}s — hf will resume from the last completed file.")
                # Sleep in 1-second slices so a cancel during backoff is responsive.
                for _ in range(backoff_sec):
                    time.sleep(1)
                    with DOWNLOAD_LOCK:
                        if not DOWNLOAD["active"]:
                            break
                backoff_sec = min(backoff_sec * 2, 60)   # 5 → 10 → 20s cap by attempt 3
            else:
                push(f"[hf] {repo_id} FAILED after {max_attempts} attempts (last exit {rc}). Click Download again to keep retrying — hf will resume.")
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
    # Spicy mode gate (v2.0 — server-side authoritative). When the user
    # hasn't enabled Spicy mode in Settings, force nsfw=False regardless
    # of what the client sent. This keeps NSFW results out of casual /
    # kid-accessible installs even if someone fiddles the client param.
    spicy_on = bool(get_settings().get("spicy_mode", False))
    if not spicy_on:
        nsfw = False
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
    # Defense-in-depth: when Spicy mode is off, also drop any items the
    # API returned with nsfw=true. CivitAI's `nsfw=false` request param
    # filters by image-tier (Soft/Mature/X) but model cards have a
    # separate boolean we re-check here so nothing slips through.
    if not spicy_on:
        items = [it for it in items if not it.get("nsfw")]
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
        atomic_write_text(sidecar, json.dumps(sidecar_data, indent=2))
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


def atomic_write_text(path: Path, text: str, *, mode: int = 0o600) -> None:
    """Write text to `path` via temp file + fsync + os.replace.

    Plain Path.write_text() can leave a half-written file if macOS sleeps,
    runs out of disk, or the panel crashes mid-write — corrupted queue or
    sidecar files would lose the user's work-in-progress. Atomic replace
    guarantees the file is either pre-write or fully post-write, never torn.
    """
    # Multiple request/worker threads can persist state close together. A
    # process-wide temp name lets one writer replace/unlink another writer's
    # temp file, producing ENOENT in the logs and risking a missed persist.
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.tmp")
    try:
        with tmp.open("w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        # Default 0o600 ensures secrets in state files (API keys in
        # agent_config.json / agent_image_config.json) aren't world/group
        # readable. umask alone leaves them at 0644 on most systems.
        try:
            os.chmod(path, mode)
        except OSError:
            pass
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


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
#   Extend (Q8 dev)      at 1280×704: > 64 GB peak, OOMs into swap
#   Extend (Q8 dev)      at  768×416: ~48 GB peak, fits 64 GB
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
        # Per-mode time estimates for a typical 5 s render (121 frames @ 24 fps),
        # measured at Exact (no Boost/Turbo). The Comfortable tier is the
        # measured baseline (M4 Studio 64 GB); other tiers are scaled relative
        # to it using crude but defensible multipliers calibrated against
        # community reports (Compact ≈ 1.6× slower from swap pressure; Roomy ≈
        # 0.8× from headroom; Studio M-Ultra ≈ 0.55× from extra GPU cores).
        # The `quality_times` block is what the Quality pills show; the legacy
        # `times` block is what the Tier modal already uses.
        "times": {
            "t2v_draft":     "about 3 min",
            "t2v_standard":  "about 12 min",
            "i2v_standard":  "about 12 min",
            "high":          None,  # disabled
            "keyframe":      None,  # disabled
            "extend":        None,  # disabled
        },
        "quality_times": {
            "quick":    "~3 min",
            "balanced": "~8 min",
            "standard": "~12 min",
            "high":     None,    # Q8 disabled at this tier
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
        # Times are measured wall clocks for a 5 s render (121 frames @ 24 fps)
        # at Exact speed, on the canonical Comfortable hardware (M-Studio /
        # M-Max 64 GB). Y1.034 + Y1.035 added a temporal-streaming VAE decode
        # patch that adds ~30 s of decode work on a 5 s clip in exchange for
        # not melting on long ones; that bump is reflected in `standard` going
        # from ~7 to ~8 min vs. pre-Y1.034 baselines (median 459 s observed
        # across 26 Y1.013-Y1.024 runs; Y1.035 observed 493 s).
        # Boost/Turbo Speed pills shave ~23%/~34% off Standard respectively.
        "times": {
            "t2v_draft":     "about 2 min",
            "t2v_standard":  "about 8 min",
            "i2v_standard":  "about 8 min",
            "high":          "about 12 min",
            "keyframe":      "about 6 min (at 768 px)",
            # Extend on Comfortable measured 16 min for +3 s at 768 px on
            # M-Max 64 GB (Q8 dev transformer, CFG=1.0, 12 steps). The
            # earlier "about 11 min" estimate predates the Y1.036 Q8 routing
            # — pre-Y1.024 Extend ran on Q4-distilled-by-accident (faster
            # weights, but technically loading the wrong model).
            "extend":        "about 16 min (at 768 px, +3 s)",
        },
        "quality_times": {
            "quick":    "~2 min",
            "balanced": "~5 min",
            "standard": "~8 min",
            "high":     "~12 min",
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
        "quality_times": {
            "quick":    "~1 min",
            "balanced": "~3 min",
            "standard": "~4 min",
            "high":     "~7 min",
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
        "quality_times": {
            "quick":    "<1 min",
            "balanced": "~2 min",
            "standard": "~3 min",
            "high":     "~4 min",
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
    # Y1.039 — skip files that ffmpeg is still writing.
    #
    # Pre-Y1.039 a fresh render appeared in the gallery the moment its mp4
    # file existed, even before ffmpeg had finished writing the moov atom.
    # The <video> tag in the carousel fetched the truncated body, decoded a
    # black frame, and the browser then cached that broken byte stream
    # under the file URL — so the card stayed black for 2–3 minutes until
    # the HTTP cache expired and a refresh re-pulled the now-complete file.
    #
    # Two layers of protection:
    #   1. If a job is running, skip its known target paths (raw_path /
    #      output_path / native_path / upscaled_path tracked on the job
    #      dict). This catches the common case cleanly.
    #   2. As a belt-and-braces, also skip any file whose mtime is within
    #      the last 2 seconds — covers cancelled jobs that left a partial,
    #      and any path the worker forgot to record on the job dict.
    in_flight_paths: set[str] = set()
    with LOCK:
        hidden_snap = set(HIDDEN_PATHS)
        cur = STATE.get("current")
        if cur:
            for k in ("raw_path", "output_path", "native_path", "upscaled_path"):
                v = cur.get(k)
                if v:
                    in_flight_paths.add(str(v))
    inflight_mtime_cutoff = time.time() - 2.0
    out = []
    for p in files:
        path_s = str(p)
        is_hidden = path_s in hidden_snap
        if is_hidden and not include_hidden:
            continue
        # In-flight protection — see the comment block above.
        if path_s in in_flight_paths:
            continue
        try:
            mt = p.stat().st_mtime
        except OSError:
            continue
        if mt > inflight_mtime_cutoff:
            continue
        # Pull elapsed_sec from the sidecar so the gallery card can show
        # "how long this took to render" instead of a wall-clock timestamp
        # (the timestamp tells you nothing useful when scanning a list of
        # 60 outputs — duration is what users actually compare). Sidecar
        # read is cheap, but tolerate a missing/corrupt sidecar by leaving
        # elapsed_sec null and letting the UI fall back to the file mtime.
        elapsed_sec = None
        sidecar = p.with_suffix(p.suffix + ".json")
        has_sidecar = sidecar.exists()
        if has_sidecar:
            try:
                meta = json.loads(sidecar.read_text())
                v = meta.get("elapsed_sec")
                if isinstance(v, (int, float)):
                    elapsed_sec = float(v)
            except Exception:
                pass
        # Y1.039 cache-bust — append the file's mtime as a version param so
        # the browser treats post-replace files as a new URL. Without this,
        # if a file was overwritten in place (e.g. an upscale rewriting the
        # native mp4), the browser would keep serving its cached old copy.
        # `mt` was captured above for the in-flight skip; reuse it here.
        out.append({
            "name": p.name,
            "path": path_s,
            "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mt)),
            "size_mb": p.stat().st_size / 1024 / 1024,
            "elapsed_sec": elapsed_sec,
            "url": f"/file?path={quote(path_s)}&v={int(mt)}",
            "has_sidecar": has_sidecar,
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
            env["LTX_ENABLE_MODEL_UPSCALE"] = "1" if MODEL_UPSCALE_ENABLED else "0"
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


def compute_upscale_plan(w: int, h: int, mode: str | None,
                          helper_did_model_upscale: bool = False) -> dict | None:
    """Plan a panel-side ffmpeg upscale pass. Returns None when no further
    work is needed (helper already produced the target dims, or target is off).

    `helper_did_model_upscale=True` means the helper already ran the LTX latent
    x2 upscaler; the file on disk is at 2× the requested W/H. We use this to
    avoid double-upscaling and to plan a downscale-only pass for the
    fit_720p target on a model-upscaled source."""
    mode = (mode or "off").strip().lower()
    if mode in ("", "off", "native"):
        return None
    # Effective dims of the file the helper actually wrote. The model-based
    # upscale (Sharper) doubles them inside the helper before VAE decode.
    eff_w = w * 2 if helper_did_model_upscale else w
    eff_h = h * 2 if helper_did_model_upscale else h
    if mode == "fit_720p":
        # No crop, no distortion: fit inside standard 720p canvas and pad any
        # remainder. 1024×576 fills 1280×720 exactly; 1280×704 becomes 1280×704
        # with 8px bars top/bottom; 704×1280 becomes 704×1280 with side bars.
        if eff_w >= eff_h:
            target_w, target_h, tag = 1280, 720, "720p"
        else:
            target_w, target_h, tag = 720, 1280, "v720p"
        # If the helper already produced the exact target size, skip the pass.
        if eff_w == target_w and eff_h == target_h:
            return None
        vf = (
            f"scale={target_w}:{target_h}:"
            "force_original_aspect_ratio=decrease:flags=lanczos,"
            f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:color=black"
        )
        method = "ffmpeg_lanczos_downscale" if helper_did_model_upscale else "ffmpeg_lanczos"
    elif mode == "x2":
        # If the helper already did model x2, the file is already at the
        # x2 target — no further panel-side work needed.
        if helper_did_model_upscale:
            return None
        target_w, target_h, tag = w * 2, h * 2, "up2x"
        vf = f"scale={target_w}:{target_h}:flags=lanczos"
        method = "ffmpeg_lanczos"
    else:
        raise RuntimeError(f"unknown upscale mode: {mode}")
    return {
        "mode": mode,
        "method": method,
        "target_w": target_w,
        "target_h": target_h,
        "tag": tag,
        "vf": vf,
    }


def run_postprocess_tracked(cmd: list[str], label: str) -> tuple[str, str]:
    """Run a post-process in its own process group so /stop can kill it."""
    push(f"{label}: " + " ".join(shlex.quote(c) for c in cmd))
    env = os.environ.copy()
    env["PATH"] = f"{FFMPEG_BIN}:{env.get('PATH', '')}"
    proc = subprocess.Popen(
        cmd, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True,
    )
    with LOCK:
        STATE["mux_pgid"] = os.getpgid(proc.pid)
    try:
        stdout, stderr = proc.communicate()
    finally:
        with LOCK:
            STATE["mux_pgid"] = None
    if proc.returncode != 0:
        push((stderr or stdout or "").strip())
        raise RuntimeError(f"{label.lower()} exited with code {proc.returncode}")
    return stdout or "", stderr or ""


def run_ffmpeg_tracked(cmd: list[str], label: str) -> tuple[str, str]:
    """Run an ffmpeg process in its own process group so /stop can kill it."""
    return run_postprocess_tracked(cmd, label)


def run_pipersr_tracked(source: Path, output: Path, mode: str, crf: str,
                        pix_fmt: str, preset: str) -> tuple[str, str]:
    script = ROOT / "scripts" / "upscale_compare_pipersr.py"
    if not script.exists():
        raise RuntimeError("PiperSR upscale script is missing")
    if not PIPERSR_UPSCALE_ENABLED:
        raise RuntimeError("PiperSR is not installed in this environment")
    return run_postprocess_tracked([
        sys.executable, str(script), str(source),
        "--pipersr-only",
        "--mode", mode,
        "--output", str(output),
        "--crf", crf,
        "--pix-fmt", pix_fmt,
        "--preset", preset,
    ], "Sharp upscale")


def video_duration(frames: int) -> float:
    # LTX 2.x uses frame counts shaped like 8k+1. The encoded file contains
    # the intervals between those endpoints, so 121 means exactly 5.0s at
    # 24fps, not 5.0417s.
    return round(max(0.0, (max(1, int(frames)) - 1) / FPS), 3)


def _frames_to_model_duration(frames: int, fps: float = FPS) -> float:
    """LTX frame counts are 8k+1; the interval duration is (frames - 1) / fps."""
    return max(0.0, (max(1, int(frames)) - 1) / float(fps))


def _duration_to_8k_frames(duration_sec: float, fps: float) -> int:
    """Nearest LTX-compatible frame count for a duration/fps pair."""
    k = max(1, int(round(max(0.0, duration_sec) * float(fps) / 8.0)))
    return k * 8 + 1


def stop_current_job(timeout: float = 5.0) -> None:
    """Kill the warm helper (and any in-flight ffmpeg mux/upscale). Worker advances."""
    with LOCK:
        cur = STATE["current"]
        mux_pgid = STATE.get("mux_pgid")
    if cur is not None:
        cur["cancel_requested"] = True
    push("Stop requested — killing helper + ffmpeg post-process to abort current job.")
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
    quality = f("quality", "balanced")
    if quality == "quick":
        default_w, default_h = 640, 480
    elif quality in ("standard", "high"):
        default_w, default_h = 1280, 704
    else:
        default_w, default_h = 1024, 576
    upscale = f("upscale", "fit_720p" if quality == "balanced" else "off")
    requested_upscale_method = (f("upscale_method", "lanczos") or "lanczos").strip().lower()
    if requested_upscale_method == "model":
        requested_upscale_method = "pipersr"
    if requested_upscale_method == "pipersr" and not PIPERSR_UPSCALE_ENABLED:
        requested_upscale_method = "lanczos"
    if requested_upscale_method not in ("lanczos", "pipersr"):
        requested_upscale_method = "lanczos"
    temporal_mode = f("temporal_mode", "native").strip().lower()
    if temporal_mode not in ("native", "fps12_interp24"):
        temporal_mode = "native"

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
            "negative_prompt": f("negative_prompt", ""),
            "width": max(32, int(f("width", str(default_w)) or default_w)),
            "height": max(32, int(f("height", str(default_h)) or default_h)),
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
            # keyframe (FFLF) mode params.
            # Two-keyframe path (start_image + end_image) is the legacy panel
            # contract still used by the manual UI. The agent SDK can pass
            # `keyframes_json` — a JSON-encoded list of {image_path, frame_index}
            # — to use the engine's full multi-keyframe support (Layer 2 of
            # the keyframe SDK; see docs/SDK_KEYFRAME_INTERPOLATION.md).
            "start_image": f("start_image", ""),
            "end_image": f("end_image", ""),
            "keyframes_json": f("keyframes_json", ""),
            "keyframes_total_frames": f("keyframes_total_frames", ""),
            # Optional session tag — agent runs stamp this so the manifest
            # writer can later filter all jobs that came from one session.
            "session_tag": f("session_tag", ""),
            "enhance": f("enhance", "off") == "on",
            "stop_comfy": f("stop_comfy", "off") == "on",
            "open_when_done": f("open_when_done", "off") == "on",
            "label": f("preset_label", "") or None,
            "quality": quality,                    # quick / balanced / standard / high
            "accel": f("accel", "off"),            # off / boost / turbo
            "temporal_mode": temporal_mode,         # native / fps12_interp24
            "upscale": upscale,                     # off / fit_720p / x2
            "upscale_method": requested_upscale_method,   # lanczos / pipersr
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
    if p.get("accel") not in ("off", "boost", "turbo"):
        p["accel"] = "off"
    if quality == "high" or mode in ("extend", "keyframe"):
        p["accel"] = "off"

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
            "renders, or pick Quality=Quick for a faster smaller-resolution render at "
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
        # Y1.036 — Extend always loads the Q8 `transformer-dev.safetensors`
        # weights (CFG-guided dev transformer, not the 8-step distilled).
        # Pre-Y1.024 the Q4 dir incidentally shipped a copy of that file as
        # download bloat, so Extend silently worked on Q4-only installs. The
        # Y1.024 download filter pruned the dupe and exposed that Extend is
        # structurally Q8-class. Gate it the same way Keyframe does and route
        # the helper to the Q8 dir.
        ext_missing = q8_missing_files()
        if ext_missing:
            raise RuntimeError(
                f"Extend requires the full Q8 model at {Q8_LOCAL_PATH}. "
                f"Missing {len(ext_missing)} file(s): {', '.join(ext_missing[:3])}"
                f"{' …' if len(ext_missing) > 3 else ''}. "
                f"Click \"Download Q8\" in Pinokio to install it (~37 GB)."
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
                # Y1.036: explicit model_dir → Q8. Helper's get_pipe("extend")
                # caches per-model-dir so this rebuilds the pipe only when
                # the dir actually flips.
                "model_dir": str(Q8_LOCAL_PATH),
                "prompt": p["prompt"],
                "negative_prompt": p.get("negative_prompt", ""),
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
             f"steps={steps} cfg={cfg_scale} (Q8 dev transformer)")
        result = HELPER.run(job_spec)
        if "seed_used" in result:
            push(f"seed used: {result['seed_used']}")
            p["seed_used"] = result["seed_used"]

        sidecar = {
            "output": str(final_out), "raw_output": str(final_out),
            "params": {**p, "command": "extend"},
            "started": job.get("started_at"),
            "elapsed_sec": round(time.time() - job["started_ts"], 2) if job.get("started_ts") else None,
            # Y1.036 — Extend always loads Q8 `transformer-dev` regardless of
            # the panel's nominal MODEL_ID (which tracks the Q4 distilled
            # used by T2V/I2V/Standard). Record the actual path so the
            # /info modal and historical analyzers don't misreport.
            "fps": FPS, "model": str(Q8_LOCAL_PATH), "queue_id": job["id"],
            "helper_elapsed_sec": result.get("elapsed_sec"),
            "output_codec": output_codec_settings(),
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
        # Multi-keyframe path: agent submits a JSON-encoded list of
        # {image_path, frame_index} pairs. Layer 1 of the SDK shipped the
        # helper-side acceptance; this branch is Layer 2 (panel form).
        # Backward compat: when keyframes_json is empty, fall back to the
        # legacy start_image + end_image two-keyframe shape.
        kf_list_raw = (p.get("keyframes_json") or "").strip()
        kf_list: list[dict] = []
        if kf_list_raw:
            try:
                parsed = json.loads(kf_list_raw)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"keyframes_json must be valid JSON: {e}") from e
            if not isinstance(parsed, list) or len(parsed) < 2:
                raise RuntimeError("keyframes_json must be a list with >=2 items")
            last_idx = -1
            for kf in parsed:
                img = (kf.get("image_path") or "").strip()
                idx = int(kf.get("frame_index", 0))
                if not img or not Path(img).exists():
                    raise RuntimeError(f"keyframe image not found: {img}")
                if idx <= last_idx:
                    raise RuntimeError("keyframe frame_index values must strictly increase")
                last_idx = idx
                kf_list.append({"image_path": img, "frame_index": idx})

        if kf_list:
            # Sync start_image / end_image with the first/last entries so the
            # rest of the pipeline (sidecar, gallery preview) shows something
            # sensible even though the helper is using the full list.
            p["start_image"] = kf_list[0]["image_path"]
            p["end_image"] = kf_list[-1]["image_path"]
        else:
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
        # Filename: derive from the user's label or first words of the prompt.
        # Technical metadata (dimensions, frames, timestamp) lives in the
        # sidecar at <out_path>.json, surfaced by the gallery info button.
        kf_stem = _descriptive_filename(p.get("label") or "", p.get("prompt") or "",
                                        fallback="keyframe")
        out_path = _unique_output_path(OUTPUT, kf_stem)
        job["raw_path"] = str(out_path)
        # When the agent submitted a multi-keyframe list, hand the helper
        # the full keyframe_images + keyframe_indices arrays (its Layer 1
        # contract from 2026-05-06). Otherwise stick with the legacy two-
        # keyframe call shape.
        if kf_list:
            kf_imgs = [k["image_path"] for k in kf_list]
            kf_idxs = [int(k["frame_index"]) for k in kf_list]
            # Validate every index is reachable in the requested clip length.
            if any(i < 0 or i >= frames for i in kf_idxs):
                raise RuntimeError(
                    f"keyframe frame_index out of range [0, {frames-1}]: {kf_idxs}"
                )
            kf_helper_params = {
                "keyframe_images": kf_imgs,
                "keyframe_indices": kf_idxs,
            }
        else:
            kf_helper_params = {
                "start_image": p["start_image"],
                "end_image": p["end_image"],
            }
        job_spec = {
            "action": "generate_keyframe",
            "id": job["id"],
            "params": {
                "model_dir": str(Q8_LOCAL_PATH),
                "prompt": p["prompt"],
                "negative_prompt": p.get("negative_prompt", ""),
                "output_path": str(out_path),
                **kf_helper_params,
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
            "output_codec": output_codec_settings(),
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
    temporal_mode = (p.get("temporal_mode") or "native").strip().lower()
    if temporal_mode not in ("native", "fps12_interp24"):
        temporal_mode = "native"
    temporal_supported = mode in ("t2v", "i2v") and quality != "high"
    if temporal_mode != "native" and not temporal_supported:
        push("Long Clip Boost is only available for Q4 Text/Image renders; using native 24fps.")
        temporal_mode = "native"
    p["temporal_mode"] = temporal_mode

    delivery_fps = float(FPS)
    model_fps = delivery_fps
    model_frames = frames
    temporal_plan = None
    if temporal_mode == "fps12_interp24":
        model_fps = 12.0
        requested_duration = _frames_to_model_duration(frames, delivery_fps)
        model_frames = _duration_to_8k_frames(requested_duration, model_fps)
        temporal_plan = {
            "mode": "fps12_interp24",
            "label": "12 → 24 fps",
            "model_fps": model_fps,
            "delivery_fps": delivery_fps,
            "source_frames": model_frames,
            "delivery_frames": frames,
            "requested_duration_sec": round(requested_duration, 3),
            "method": "ffmpeg_minterpolate_mci",
        }

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
    # Filename: derive a descriptive stem from label / prompt. Technical
    # detail (mode/quality/dimensions/frames/timestamp) is preserved in
    # the sidecar at <final_out>.json — the gallery's ⓘ button surfaces it.
    desc_stem = _descriptive_filename(p.get("label") or "", p.get("prompt") or "",
                                      fallback=tag)
    needs_mux = mode == "i2v_clean_audio"
    if needs_mux:
        raw_out = _unique_output_path(OUTPUT, desc_stem + "_raw")
        # Strip the _raw suffix from the same stem for the muxed final.
        final_out = _unique_output_path(OUTPUT, desc_stem)
    elif temporal_plan:
        raw_out = _unique_output_path(OUTPUT, desc_stem + "_12fps")
        final_out = _unique_output_path(OUTPUT, desc_stem)
    else:
        # Single file — raw == final, no _raw suffix.
        raw_out = _unique_output_path(OUTPUT, desc_stem)
        final_out = raw_out
    job["raw_path"] = str(raw_out)

    if quality == "high":
        if not SYSTEM_CAPS["allows_q8"]:
            raise RuntimeError(
                f"High quality (Q8 two-stage) isn't supported on the "
                f"{SYSTEM_CAPS['label']} hardware tier — Q8 dev transformer "
                f"(~19 GB) plus the upscaler stage doesn't fit. "
                f"Use Standard or Quick instead, or upgrade to 64+ GB."
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
                "negative_prompt": p.get("negative_prompt", ""),
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
        # Quick / Standard — Q4 one-stage with steps from form.
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
                "negative_prompt": p.get("negative_prompt", ""),
                "output_path": str(raw_out),
                "height": height,
                "width": width,
                "frames": model_frames,
                "frame_rate": model_fps,
                "steps": p["steps"],
                "seed": p["seed"],
                "image": p["image"] if mode != "t2v" else None,
                "loras": loras,
                "accel": p.get("accel", "off"),
                # Sharp/PiperSR is a panel-side post-render pass. Do not pass it
                # through to the helper; the helper's old "model" path is the
                # hidden LTX latent upscaler experiment that distorted faces.
                "upscale": p.get("upscale", "off"),
                "upscale_method": "lanczos",
            },
        }
        temporal_suffix = ""
        if temporal_plan:
            temporal_suffix = (
                f" · Long Clip Boost {model_frames}f@{model_fps:g}fps"
                f" → {frames}f@{delivery_fps:g}fps"
            )
        if loras:
            push(f"Run via helper: id={job['id']} mode={mode} quality={quality} accel={p.get('accel', 'off')} "
                 f"{width}x{height} {model_frames}f · {len(loras)} LoRA"
                 f"{'s' if len(loras) != 1 else ''}"
                 f"{' (incl. HDR)' if p.get('hdr') else ''}{temporal_suffix}")
        else:
            push(f"Run via helper: id={job['id']} mode={mode} quality={quality} accel={p.get('accel', 'off')} "
                 f"{width}x{height} {model_frames}f{temporal_suffix}")

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
        # Match the selected panel output preset. The helper subprocess gets
        # these values through env vars at spawn time; panel-side ffmpeg passes
        # must read panel_settings.json directly or they fall back to lossless
        # yuv444p/crf0 and produce huge files.
        codec = output_codec_settings()
        mux_pix_fmt = codec["pix_fmt"]
        mux_crf = codec["crf"]
        mux_cmd = [str(FFMPEG), "-y", "-i", str(raw_out), "-i", audio,
                   "-map", "0:v:0", "-map", "1:a:0"]
        if pad_filter:
            mux_cmd += ["-vf", pad_filter]
        mux_cmd += [
            "-af", f"apad,atrim=0:{duration},asetpts=PTS-STARTPTS",
            "-c:v", "libx264", "-pix_fmt", mux_pix_fmt, "-crf", mux_crf, "-preset", "medium",
            "-movflags", "+faststart",
            "-c:a", "aac", "-b:a", "192k",
            "-t", f"{duration}",
            str(final_out),
        ]
        run_ffmpeg_tracked(mux_cmd, "Mux")
        final_target = final_out

    if temporal_plan:
        codec = output_codec_settings()
        mux_pix_fmt = codec["pix_fmt"]
        mux_crf = codec["crf"]
        temporal_preset = os.environ.get("LTX_TEMPORAL_PRESET", "medium")
        temporal_cmd = [
            str(FFMPEG), "-y", "-i", str(final_target),
            "-vf", f"minterpolate=fps={FPS}:mi_mode=mci:mc_mode=aobmc:vsbmc=1",
            "-c:v", "libx264", "-pix_fmt", mux_pix_fmt, "-crf", mux_crf,
            "-preset", temporal_preset,
            "-movflags", "+faststart",
            "-c:a", "copy",
            "-t", f"{video_duration(frames)}",
            str(final_out),
        ]
        run_ffmpeg_tracked(temporal_cmd, "Long Clip Boost")
        temporal_plan |= {
            "source": str(final_target),
            "output": str(final_out),
            "codec": output_codec_settings(),
            "encode_preset": temporal_preset,
        }
        set_hidden(str(final_target), True)
        push(
            f"Long Clip Boost done → {final_out.name} "
            f"({model_frames}f @ {model_fps:g}fps interpolated to {frames}f @ {FPS}fps)"
        )
        final_target = final_out

    native_target = final_target
    helper_did_model_upscale = (result.get("upscale_applied") == "model_x2")
    if helper_did_model_upscale:
        push(f"Helper produced model-upscaled output ({width*2}×{height*2}, sharper).")
    upscale_plan = compute_upscale_plan(
        width, height, p.get("upscale", "off"),
        helper_did_model_upscale=helper_did_model_upscale,
    )
    if upscale_plan:
        codec = output_codec_settings()
        mux_pix_fmt = codec["pix_fmt"]
        mux_crf = codec["crf"]
        upscale_preset = os.environ.get("LTX_UPSCALE_PRESET", "medium")
        upscale_method = (p.get("upscale_method", "lanczos") or "lanczos").strip().lower()
        if upscale_method == "model":
            upscale_method = "pipersr"
        upscaled_out = OUTPUT / (
            f"{final_target.stem}_{upscale_plan['tag']}{final_target.suffix}"
        )
        if upscale_method == "pipersr":
            push("Sharp upscale: PiperSR/CoreML 2× pass, then ffmpeg fit/export.")
            run_pipersr_tracked(
                final_target, upscaled_out, p.get("upscale", "fit_720p"),
                mux_crf, mux_pix_fmt, upscale_preset,
            )
            upscale_plan["method"] = "pipersr_coreml"
            upscale_plan["pre_pass"] = "pipersr_x2"
        else:
            upscale_cmd = [
                str(FFMPEG), "-y", "-i", str(final_target),
                "-vf", upscale_plan["vf"],
                "-c:v", "libx264", "-pix_fmt", mux_pix_fmt, "-crf", mux_crf,
                "-preset", upscale_preset,
                "-movflags", "+faststart",
                "-c:a", "copy",
                str(upscaled_out),
            ]
            run_ffmpeg_tracked(upscale_cmd, "Upscale")
            upscale_plan["method"] = "ffmpeg_lanczos"
        final_target = upscaled_out
        push(
            f"Upscale done → {upscaled_out.name} "
            f"({upscale_plan['target_w']}×{upscale_plan['target_h']}, no crop, "
            f"{upscale_plan['method']}, {mux_pix_fmt} crf {mux_crf}, preset={upscale_preset})"
        )
        set_hidden(str(native_target), True)
        push(f"Native source kept but hidden from gallery → {native_target.name}")

    sidecar = {
        "output": str(final_target),
        "raw_output": str(raw_out),
        "native_output": str(native_target),
        "params": {
            **p,
            "pad_w": pad_w, "pad_h": pad_h,
            "model_frames": model_frames if temporal_plan else frames,
            "model_fps": model_fps if temporal_plan else FPS,
            "delivery_fps": delivery_fps if temporal_plan else FPS,
            "image": p["image"] if mode != "t2v" else None,
            "audio": p["audio"] if mode == "i2v_clean_audio" else None,
        },
        "command": "helper",
        "started": job.get("started_at"),
        "elapsed_sec": round(time.time() - job["started_ts"], 2) if job.get("started_ts") else None,
        "video_duration_sec": video_duration(frames),
        "fps": FPS, "model": MODEL_ID, "queue_id": job["id"],
        "helper_elapsed_sec": result.get("elapsed_sec"),
        "output_codec": output_codec_settings(),
    }
    if result.get("accel_metrics"):
        sidecar["accel_metrics"] = result["accel_metrics"]
    if temporal_plan:
        sidecar["temporal"] = temporal_plan
    if upscale_plan:
        sidecar["upscale"] = {
            k: v for k, v in upscale_plan.items()
            if k not in ("vf",)
        } | {"source": str(native_target), "codec": output_codec_settings()}
        if helper_did_model_upscale:
            sidecar["upscale"]["pre_pass"] = "ltx_latent_x2"
    elif helper_did_model_upscale:
        # Helper produced the final 2× output already; record that for the
        # info modal so users see "Sharper (model x2)" on the card.
        sidecar["upscale"] = {
            "mode": p.get("upscale", "x2"),
            "method": "ltx_latent_x2",
            "target_w": width * 2,
            "target_h": height * 2,
            "source": str(native_target),
            "codec": output_codec_settings(),
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


# ---- Agentic Flows integration -----------------------------------------------
#
# Wires the `agent/` module into this panel. The agent runs *inside* the panel
# process (no new microservice — see CLAUDE.md §9 hard constraints), submitting
# jobs through the same FIFO queue + helper that the manual UI uses.
#
# State lives in two places:
#   - state/agent_config.json  — engine config (which LLM, which URL, key)
#   - state/agent_sessions/    — one JSON per chat session (atomic-replaced)
#
# Both survive Pinokio Reset → Reinstall via the existing fs.link symlink on
# state/ (see install.js / CLAUDE.md §7 "fs.link"). No extra plumbing needed.

AGENT_CONFIG_PATH = STATE_DIR / "agent_config.json"
AGENT_IMAGE_CONFIG_PATH = STATE_DIR / "agent_image_config.json"
AGENT_LOCK = threading.RLock()
_AGENT_CONFIG_CACHE: agent_engine.EngineConfig | None = None
_AGENT_IMAGE_CONFIG_CACHE: agent_image_engine.ImageEngineConfig | None = None


def _default_agent_engine_config() -> agent_engine.EngineConfig:
    """Default engine config: bundled Gemma 3 12B IT via mlx-lm.server.

    The Gemma weights ship with Phosphene as the LTX text encoder. They're
    a perfectly capable instruction-tuned chat model — zero extra download
    required to use the agent on a fresh Phosphene install. Users who want
    a stronger agent can drop Qwen 3 Coder 30B-A3B (or similar MLX 4-bit
    chat model) into mlx_models/ and switch via the engine settings drawer.
    """
    return agent_engine.EngineConfig(
        kind="phosphene_local",
        base_url=f"http://127.0.0.1:{agent_local_server._PORT}/v1",
        model="gemma-3-12b-it-4bit",
        local_model_path=str(GEMMA),
        api_key="",
        temperature=0.4,
        max_tokens=3072,
    )


def _load_agent_config() -> agent_engine.EngineConfig:
    """Read state/agent_config.json (creating defaults on first call).

    Cached after the first read; updates go through _save_agent_config()
    which atomically replaces the file and updates the cache.
    """
    global _AGENT_CONFIG_CACHE
    with AGENT_LOCK:
        if _AGENT_CONFIG_CACHE is not None:
            return _AGENT_CONFIG_CACHE
        if AGENT_CONFIG_PATH.is_file():
            try:
                data = json.loads(AGENT_CONFIG_PATH.read_text(encoding="utf-8"))
                cfg = agent_engine.EngineConfig(**data)
            except (OSError, json.JSONDecodeError, TypeError) as e:
                push(f"agent: invalid agent_config.json ({e}); using defaults")
                cfg = _default_agent_engine_config()
        else:
            cfg = _default_agent_engine_config()
        _AGENT_CONFIG_CACHE = cfg
        return cfg


_ALLOWED_REMOTE_HOSTS = {
    # OpenAI-compatible providers we've intentionally tested. The kind="custom"
    # branch lets the user point at any HTTPS endpoint (their own gateway,
    # OpenRouter, etc.); the allow-list below is purely informational and not
    # used to block — see _validate_engine_base_url for the actual policy.
}


def _validate_engine_base_url(kind: str, base_url: str) -> tuple[bool, str]:
    """Block configs that would leak the user's API key to a hostile target.

    The agent posts the configured api_key as a Bearer token to whatever
    base_url we save. If the user is talked into pasting `http://evil.com/v1`
    the next message exfiltrates the key in plaintext. Two rules:

      * Any non-loopback URL must be HTTPS. Plain http:// outside loopback
        is rejected — even if the user "trusts" the host, an upstream MITM
        could still capture the bearer token.
      * The URL must parse to a real http/https scheme with a host.
    """
    from urllib.parse import urlparse as _u
    url = (base_url or "").strip()
    if not url:
        return False, "base_url is empty"
    try:
        p = _u(url)
    except Exception as e:
        return False, f"unparseable base_url: {e}"
    if p.scheme not in ("http", "https"):
        return False, "base_url must be http:// or https://"
    host = (p.hostname or "").lower()
    if not host:
        return False, "base_url has no host"
    is_loopback = host in {"127.0.0.1", "::1", "localhost"}
    if p.scheme == "http" and not is_loopback:
        return False, "remote base_url must use https:// (refusing to send API key over plaintext)"
    return True, ""


def _save_agent_config(updates: dict) -> agent_engine.EngineConfig:
    """Merge `updates` into the current engine config and persist.

    Empty-string values are treated as 'leave as-is' for `api_key` so the
    masked /agent/config GET → echo POST round-trip doesn't accidentally
    erase the saved key (mirrors the panel_settings.json convention).
    """
    global _AGENT_CONFIG_CACHE
    with AGENT_LOCK:
        cur = _load_agent_config()
        merged = {**cur.__dict__}
        for k, v in (updates or {}).items():
            if k not in merged:
                continue            # ignore unknown fields
            if k == "api_key" and v == "":
                continue            # treat blank as no-change for secrets
            merged[k] = v
        cfg = agent_engine.EngineConfig(**merged)
        ok, why = _validate_engine_base_url(cfg.kind, cfg.base_url)
        if not ok:
            raise ValueError(why)
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        atomic_write_text(AGENT_CONFIG_PATH, json.dumps(cfg.__dict__, indent=2))
        _AGENT_CONFIG_CACHE = cfg
        return cfg


def _load_agent_image_config() -> agent_image_engine.ImageEngineConfig:
    """Read state/agent_image_config.json (creating defaults on first call)."""
    global _AGENT_IMAGE_CONFIG_CACHE
    with AGENT_LOCK:
        if _AGENT_IMAGE_CONFIG_CACHE is not None:
            return _AGENT_IMAGE_CONFIG_CACHE
        if AGENT_IMAGE_CONFIG_PATH.is_file():
            try:
                data = json.loads(AGENT_IMAGE_CONFIG_PATH.read_text(encoding="utf-8"))
                cfg = agent_image_engine.ImageEngineConfig(**data)
            except (OSError, json.JSONDecodeError, TypeError) as e:
                push(f"agent: invalid agent_image_config.json ({e}); using mock defaults")
                cfg = agent_image_engine.ImageEngineConfig()
        else:
            cfg = agent_image_engine.ImageEngineConfig()
        _AGENT_IMAGE_CONFIG_CACHE = cfg
        return cfg


def _save_agent_image_config(updates: dict) -> agent_image_engine.ImageEngineConfig:
    global _AGENT_IMAGE_CONFIG_CACHE
    with AGENT_LOCK:
        cur = _load_agent_image_config()
        merged = {**cur.__dict__}
        for k, v in (updates or {}).items():
            if k not in merged:
                continue
            if k == "bfl_api_key" and v == "":
                continue                        # blank means "leave as-is"
            merged[k] = v
        cfg = agent_image_engine.ImageEngineConfig(**merged)
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        atomic_write_text(AGENT_IMAGE_CONFIG_PATH, json.dumps(cfg.__dict__, indent=2))
        _AGENT_IMAGE_CONFIG_CACHE = cfg
        return cfg


def _agent_capabilities(*, include_secrets: bool = False) -> dict:
    """Hardware-tier snapshot the agent uses to clamp its plan.

    Mirrors the SYSTEM_CAPS table in tier-friendly form so the system
    prompt can show the user what this Mac can actually render. Also
    carries the current image-engine config so the `generate_shot_images`
    tool can dispatch without circular imports.

    `include_secrets` controls whether the api_key fields (BFL key etc.)
    are returned. The HTTP-facing endpoint (`GET /agent/config`) calls
    this with the default (False) → masked dict. The PanelOps construction
    used by tools.py inside the agent loop calls with True so the actual
    BFL request can authenticate.
    """
    img_cfg = _load_agent_image_config()
    return {
        "tier": SYSTEM_TIER,
        "tier_label": SYSTEM_CAPS.get("label", SYSTEM_TIER),
        "max_dim_t2v": tier_max_dim("t2v"),
        "max_dim_i2v": tier_max_dim("i2v"),
        "max_dim_kf": tier_max_dim("keyframe"),
        "max_dim_extend": tier_max_dim("extend"),
        "allows_q8": SYSTEM_CAPS.get("allows_q8", False),
        "allows_keyframe": SYSTEM_CAPS.get("allows_keyframe", False),
        "allows_extend": SYSTEM_CAPS.get("allows_extend", False),
        "image_engine_config": (
            dict(img_cfg.__dict__) if include_secrets
            else img_cfg.to_public_dict()
        ),
    }


def _descriptive_filename(label: str, prompt: str, *, fallback: str) -> str:
    """Build a descriptive output stem from the user-set label or prompt.

    Salo: 'Additional videos should have the descriptive names instead of
    being mlx text to video resolution all that you can see on the
    information button.' The technical info (mode/quality/dimensions/
    frames/timestamp) lives in the sidecar JSON next to every mp4 and is
    surfaced by the gallery's ⓘ button — it doesn't need to be in the
    filename too.

    Order of preference: label → first 6 words of prompt → fallback.
    Sanitized to lowercase ASCII alnum + underscore, capped at 50 chars.
    Caller appends a uniqueness suffix and the extension.
    """
    src = (label or "").strip()
    if not src and prompt:
        # Take the first ~6 words of the prompt; strip dialogue ('...').
        words = re.findall(r"[A-Za-z0-9]+", prompt)[:6]
        src = " ".join(words) if words else ""
    if not src:
        src = fallback
    safe = re.sub(r"[^a-z0-9]+", "_", src.lower()).strip("_")
    if not safe:
        safe = fallback
    return safe[:50].rstrip("_")


def _unique_output_path(base: Path, stem: str, ext: str = ".mp4") -> Path:
    """Resolve `<base>/<stem><ext>` to a non-colliding path.

    First call gets the bare name; subsequent collisions get `_2`, `_3`, ...
    suffixes. The full technical metadata is in the sidecar so users can
    still tell renders apart.
    """
    p = base / f"{stem}{ext}"
    if not p.exists():
        return p
    i = 2
    while True:
        p = base / f"{stem}_{i}{ext}"
        if not p.exists():
            return p
        i += 1


def _agent_submit_job(form: dict) -> dict:
    """PanelOps.submit_job: build a job and append to the live FIFO queue.

    Goes through the exact same path /run / /queue/add use, so jobs
    submitted by the agent are indistinguishable from manual ones once
    queued (same params, same worker, same history bucket).

    Pre-flight: verify the helper subprocess can actually run before
    queueing. Catches the 'agent says success, render fails 30 s later
    with [Errno 2] No such file or directory' class of bug — the queue
    accepts anything, but the worker hits a missing helper python and
    the agent never knows.
    """
    if not HELPER_PYTHON.is_file():
        raise RuntimeError(
            f"helper python not found at {HELPER_PYTHON}. "
            f"Renders will fail. Set LTX_HELPER_PYTHON env var to a valid "
            f"interpreter or reinstall via Pinokio (which sets it for you). "
            f"Tried: {MLX}/.venv/bin/python3.11, {MLX}/env/bin/python3.11."
        )
    if not HELPER_SCRIPT.is_file():
        raise RuntimeError(
            f"helper script not found at {HELPER_SCRIPT}. "
            f"The mlx_warm_helper.py file is missing — reinstall the panel."
        )
    job = make_job(form)
    with QUEUE_COND:
        STATE["queue"].append(job)
        QUEUE_COND.notify_all()
    persist_queue()
    push(f"agent: queued job {job['id']} mode={job['params'].get('mode')} "
         f"label={job['params'].get('label') or '-'}")
    return job


def _agent_queue_snapshot() -> dict:
    with LOCK:
        return {
            "running": STATE.get("running", False),
            "current": dict(STATE["current"]) if STATE.get("current") else None,
            "queue": [dict(j) for j in STATE.get("queue", [])],
            "history": [dict(j) for j in STATE.get("history", [])][:30],
        }


def _agent_find_job(job_id: str) -> dict | None:
    with LOCK:
        cur = STATE.get("current")
        if cur and cur.get("id") == job_id:
            return dict(cur)
        for j in STATE.get("queue", []):
            if j.get("id") == job_id:
                return dict(j)
        for j in STATE.get("history", []):
            if j.get("id") == job_id:
                return dict(j)
    return None


def _build_panel_ops() -> agent_tools.PanelOps:
    # The PanelOps surface is in-process only — passed to tool dispatch.
    # The capabilities dict here MUST include the bfl_api_key / etc. so
    # `generate_shot_images` can authenticate. The HTTP-facing
    # `/agent/config` endpoint calls _agent_capabilities() WITHOUT
    # include_secrets, masking the keys.
    return agent_tools.PanelOps(
        submit_job=_agent_submit_job,
        queue_snapshot=_agent_queue_snapshot,
        find_job=_agent_find_job,
        outputs_dir=OUTPUT,
        uploads_dir=UPLOADS,
        capabilities=_agent_capabilities(include_secrets=True),
        state_dir=STATE_DIR,
    )


def _agent_log_sink(line: str) -> None:
    """Forward mlx-lm.server stdout into the panel's Logs tab."""
    push(line)


_ATTACHMENTS_RE = re.compile(
    r"^<attachments>\s*(.*?)\s*</attachments>\s*", re.DOTALL
)


def _split_user_attachments(content: str) -> tuple[str, list[dict]]:
    """Pull a leading `<attachments>JSON</attachments>` block off a user
    message, returning (visible_text, attachments_list).

    The block is the wire convention for image/PDF/text attachments — see
    the /message handler. Anything that fails to parse is left in place
    so the user can still read what they typed.
    """
    m = _ATTACHMENTS_RE.match(content or "")
    if not m:
        return content, []
    try:
        atts = json.loads(m.group(1))
        if not isinstance(atts, list):
            return content, []
    except (json.JSONDecodeError, ValueError):
        return content, []
    return content[m.end():], atts


def _render_session_messages(sess: agent_runtime.Session) -> list[dict]:
    """Format session.messages for the chat UI.

    The raw message list contains:
      - the system prompt (skip in UI)
      - user messages (show as user bubbles, optionally with attachments)
      - assistant messages (may contain a fenced action block — strip it
        for display, surface the parsed tool name as a chip)
      - tool result wrapper user messages (parse, show as a tool-result
        chip rather than another user bubble)

    The UI renders each entry's `kind`: "user" | "assistant" | "tool_call"
    | "tool_result" | "system_note".
    """
    out = []
    for m in sess.messages:
        role = m.get("role", "")
        content = m.get("content", "") or ""
        if role == "system":
            continue                # never shown to the user
        if role == "user" and content.startswith("<tool_result"):
            # Synthetic tool-result message inserted by run_turn
            try:
                inner = content.split(">", 1)[1].rsplit("</tool_result>", 1)[0].strip()
                parsed = json.loads(inner) if inner.startswith("{") else {"raw": inner}
            except Exception:                       # noqa: BLE001
                parsed = {"raw": content}
            out.append({"kind": "tool_result", "result": parsed})
            continue
        if role == "user":
            visible, atts = _split_user_attachments(content)
            entry: dict = {"kind": "user", "content": visible}
            if atts:
                entry["attachments"] = atts
            out.append(entry)
            continue
        if role == "assistant":
            action = agent_tools.parse_action_block(content)
            display = agent_tools.strip_action_block(content)
            entry = {"kind": "assistant", "content": display}
            if action:
                entry["tool_call"] = {
                    "tool": action.get("tool"),
                    "args": action.get("args", {}),
                }
            out.append(entry)
            continue
        out.append({"kind": role, "content": content})
    return out


def _agent_ollama_status(timeout: float = 2.0) -> dict:
    """Probe a local Ollama server. Returns base_url + installed-model list.

    Ollama's native API:
      GET http://127.0.0.1:11434/api/tags  -> {"models":[{"name", "size", ...}]}
    Its OpenAI-compat surface lives at /v1/* (chat/completions, models, ...)
    so the agent pointing at base_url + /v1 just works once Ollama is
    running.
    """
    base_url = os.environ.get("LTX_OLLAMA_URL", "http://127.0.0.1:11434")
    tags_url = base_url.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(tags_url, timeout=timeout) as r:
            data = json.loads(r.read())
    except urllib.error.URLError as e:
        return {"running": False, "base_url": base_url,
                "openai_url": base_url.rstrip("/") + "/v1",
                "models": [], "error": f"unreachable: {getattr(e, 'reason', e)}"}
    except Exception as e:                  # noqa: BLE001
        return {"running": False, "base_url": base_url,
                "openai_url": base_url.rstrip("/") + "/v1",
                "models": [], "error": str(e)}
    models = []
    for m in (data.get("models") or []):
        size = int(m.get("size") or 0)
        details = m.get("details") or {}
        models.append({
            "name": m.get("name") or m.get("model"),
            "size_bytes": size,
            "size_gb": round(size / (1024 ** 3), 2) if size else None,
            "modified_at": m.get("modified_at"),
            "family": details.get("family"),
            "parameter_size": details.get("parameter_size"),
            "quantization": details.get("quantization_level"),
        })
    return {
        "running": True,
        "base_url": base_url,
        "openai_url": base_url.rstrip("/") + "/v1",
        "models": models,
    }


# ---- Hugging Face model browser ------------------------------------------
# `hf` CLI is already installed (used for LoRA + LTX downloads). We piggy-
# back on the same binary for chat-model installs. Search talks directly
# to https://huggingface.co/api which doesn't require auth for public
# models. Install streams `hf download` to the panel log; cancellation
# uses SIGTERM on the spawned process group.
HF_MODEL_DOWNLOAD_LOCK = threading.Lock()
HF_MODEL_DOWNLOAD: dict = {
    "active": False,
    "repo_id": None,
    "started_ts": None,
    "lines": [],            # ring of recent lines for the UI to display
    "proc": None,
    "pgid": None,
    "error": None,
    "done": False,
    "target_dir": None,
}
_HF_MODEL_LINES_LIMIT = 200


def _hf_model_search(query: str = "", *, abliterated: bool = False,
                     limit: int = 30, library: str = "mlx",
                     pipeline_tag: str = "text-generation") -> list[dict]:
    """Hit the public HF model search endpoint and return chat-capable
    MLX models matching `query`. When `abliterated` is True, swap the
    search term to 'abliterated' so the user gets uncensored variants
    (which are a substring convention in repo ids — `huihui-ai/...`,
    `mlx-community/...-abliterated-...`).
    """
    params = [
        ("library", library),
        ("pipeline_tag", pipeline_tag),
        ("sort", "downloads"),
        ("direction", "-1"),
        ("limit", str(limit)),
    ]
    if abliterated:
        # Combine the user's query with 'abliterated' so they can still
        # narrow ('qwen abliterated' -> qwen variants only).
        q = (query or "").strip()
        params.append(("search", (q + " abliterated").strip()))
    elif query:
        params.append(("search", query.strip()))

    url = "https://huggingface.co/api/models?" + urllib.parse.urlencode(params)
    headers = {"Accept": "application/json"}
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as r:
            items = json.loads(r.read())
    except urllib.error.URLError as e:
        raise RuntimeError(f"HF search failed: {getattr(e, 'reason', e)}") from e
    out = []
    for it in (items or []):
        out.append({
            "repo_id": it.get("id") or it.get("modelId"),
            "author": it.get("author"),
            "downloads": it.get("downloads", 0),
            "likes": it.get("likes", 0),
            "last_modified": it.get("lastModified"),
            "tags": it.get("tags") or [],
            "library_name": it.get("library_name") or library,
            "pipeline_tag": it.get("pipeline_tag") or pipeline_tag,
            "gated": bool(it.get("gated")),
            "private": bool(it.get("private")),
        })
    return out


def _hf_model_files(repo_id: str) -> dict:
    """Get the file list + total size for a repo. Used for 'this is a
    NN GB download' before clicking Install."""
    url = f"https://huggingface.co/api/models/{repo_id}/tree/main?recursive=true"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            items = json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code == 401 or e.code == 403:
            return {"repo_id": repo_id, "gated": True,
                    "error": f"HTTP {e.code} — model is gated. Open the model card on HF and accept terms first."}
        raise RuntimeError(f"HF tree failed (HTTP {e.code})") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"HF tree failed: {getattr(e, 'reason', e)}") from e
    files = []
    total = 0
    for f in items:
        if f.get("type") != "file":
            continue
        size = int((f.get("lfs") or {}).get("size") or f.get("size") or 0)
        files.append({"path": f.get("path"), "size_bytes": size})
        total += size
    return {
        "repo_id": repo_id,
        "files": files,
        "total_size_bytes": total,
        "total_size_gb": round(total / (1024 ** 3), 2) if total else 0,
        "file_count": len(files),
    }


def _hf_model_install_async(repo_id: str) -> dict:
    """Spawn `hf download {repo_id} --local-dir mlx_models/{name}` in a
    background thread. Status streams via /agent/models/install/status."""
    with HF_MODEL_DOWNLOAD_LOCK:
        if HF_MODEL_DOWNLOAD["active"]:
            return {"ok": False,
                    "error": f"download already active for {HF_MODEL_DOWNLOAD['repo_id']}"}
        if not HF_BIN or not Path(HF_BIN).is_file():
            return {"ok": False, "error": "hf CLI not found in this venv"}

        # Pick a sensible local dir name — last segment of the repo id.
        # Avoid clobbering existing models by appending _N if needed.
        leaf = repo_id.split("/")[-1]
        target = MODELS_DIR / leaf
        if target.exists():
            i = 2
            while (MODELS_DIR / f"{leaf}_{i}").exists():
                i += 1
            target = MODELS_DIR / f"{leaf}_{i}"
        target.parent.mkdir(parents=True, exist_ok=True)

        HF_MODEL_DOWNLOAD["active"] = True
        HF_MODEL_DOWNLOAD["repo_id"] = repo_id
        HF_MODEL_DOWNLOAD["started_ts"] = time.time()
        HF_MODEL_DOWNLOAD["lines"] = []
        HF_MODEL_DOWNLOAD["error"] = None
        HF_MODEL_DOWNLOAD["done"] = False
        HF_MODEL_DOWNLOAD["target_dir"] = str(target)

    def _run():
        try:
            cmd = [str(HF_BIN), "download", repo_id, "--local-dir", str(target)]
            env = os.environ.copy()
            env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
            tok = _active_hf_token() if "_active_hf_token" in globals() else None
            if tok:
                env["HF_TOKEN"] = tok
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                env=env, start_new_session=True,
            )
            with HF_MODEL_DOWNLOAD_LOCK:
                HF_MODEL_DOWNLOAD["proc"] = proc
                HF_MODEL_DOWNLOAD["pgid"] = os.getpgid(proc.pid)
            assert proc.stdout is not None
            for raw in proc.stdout:
                line = raw.decode("utf-8", errors="replace").rstrip()
                if not line:
                    continue
                with HF_MODEL_DOWNLOAD_LOCK:
                    HF_MODEL_DOWNLOAD["lines"].append(line)
                    HF_MODEL_DOWNLOAD["lines"] = HF_MODEL_DOWNLOAD["lines"][-_HF_MODEL_LINES_LIMIT:]
                push(f"hf download[{repo_id}]: {line}")
            rc = proc.wait()
            with HF_MODEL_DOWNLOAD_LOCK:
                HF_MODEL_DOWNLOAD["proc"] = None
                HF_MODEL_DOWNLOAD["pgid"] = None
                if rc != 0:
                    HF_MODEL_DOWNLOAD["error"] = f"hf download exited {rc}"
                else:
                    HF_MODEL_DOWNLOAD["done"] = True
                HF_MODEL_DOWNLOAD["active"] = False
        except Exception as e:                          # noqa: BLE001
            with HF_MODEL_DOWNLOAD_LOCK:
                HF_MODEL_DOWNLOAD["error"] = f"{type(e).__name__}: {e}"
                HF_MODEL_DOWNLOAD["active"] = False
                HF_MODEL_DOWNLOAD["proc"] = None
                HF_MODEL_DOWNLOAD["pgid"] = None
            push(f"hf download[{repo_id}] crashed: {e}")

    threading.Thread(target=_run, daemon=True, name=f"hf-dl-{leaf}").start()
    return {"ok": True, "repo_id": repo_id, "target_dir": str(target)}


def _hf_model_install_status() -> dict:
    with HF_MODEL_DOWNLOAD_LOCK:
        return {
            "active": HF_MODEL_DOWNLOAD["active"],
            "repo_id": HF_MODEL_DOWNLOAD["repo_id"],
            "started_ts": HF_MODEL_DOWNLOAD["started_ts"],
            "elapsed_s": (time.time() - HF_MODEL_DOWNLOAD["started_ts"]) if HF_MODEL_DOWNLOAD["started_ts"] else None,
            "lines": list(HF_MODEL_DOWNLOAD["lines"][-30:]),  # tail for display
            "last_line": HF_MODEL_DOWNLOAD["lines"][-1] if HF_MODEL_DOWNLOAD["lines"] else "",
            "error": HF_MODEL_DOWNLOAD["error"],
            "done": HF_MODEL_DOWNLOAD["done"],
            "target_dir": HF_MODEL_DOWNLOAD["target_dir"],
        }


def _hf_model_install_cancel() -> dict:
    with HF_MODEL_DOWNLOAD_LOCK:
        proc = HF_MODEL_DOWNLOAD.get("proc")
        pgid = HF_MODEL_DOWNLOAD.get("pgid")
    if not proc:
        return {"ok": False, "error": "no active download"}
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    return {"ok": True}


def _agent_local_start(model_path: str | None = None) -> dict:
    """Spawn the bundled mlx-lm.server against the current config's model.

    `model_path` overrides the saved `local_model_path` for one-off boots
    (e.g. switching models without persisting the change).
    """
    cfg = _load_agent_config()
    target = model_path or cfg.local_model_path or str(GEMMA)
    venv_py = str(MLX / "env/bin/python3.11")
    return agent_local_server.start(target, venv_python=venv_py,
                                    log_sink=_agent_log_sink)


# ---- HTTP --------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002
        return

    def _is_local_request(self) -> bool:
        """Reject DNS-rebinding attacks.

        We bind to 127.0.0.1, but a malicious page on the open internet can
        still reach us if the user's resolver returns 127.0.0.1 for an
        attacker-controlled hostname (classic DNS rebinding). The browser
        sends the rebound hostname in the Host header and the page's own
        origin in Origin/Referer — both must point at localhost for the
        request to be considered legitimate.

        Rules:
          - Host header must be loopback (127.0.0.1, [::1], localhost) on
            our PORT, or empty (some local tooling omits it).
          - If Origin is present it must also be loopback.
          - Referer is only checked as a last-resort hint.
        """
        host = (self.headers.get("Host") or "").strip().lower()
        # Host can include the port; strip it.
        host_name = host.rsplit(":", 1)[0] if host else ""
        if host_name.startswith("[") and host_name.endswith("]"):
            host_name = host_name[1:-1]
        allowed = {"127.0.0.1", "::1", "localhost", ""}
        if host_name not in allowed:
            return False
        origin = (self.headers.get("Origin") or "").strip().lower()
        if origin and origin != "null":
            try:
                from urllib.parse import urlparse as _u
                ohost = (_u(origin).hostname or "").lower()
                if ohost.startswith("[") and ohost.endswith("]"):
                    ohost = ohost[1:-1]
                if ohost not in {"127.0.0.1", "::1", "localhost"}:
                    return False
            except Exception:
                return False
        return True

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
            # Y1.039 — `no-cache` forces revalidation on every request even
            # when the URL is reused. Combined with the v=<mtime> URL bust
            # in list_outputs, this means a refreshed file gets re-fetched
            # cleanly instead of the browser serving stale partial bytes.
            self.send_header("Cache-Control", "no-cache")
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
        self.send_header("Cache-Control", "no-cache")  # Y1.039 — see above
        self.end_headers()
        with path.open("rb") as fh:
            while chunk := fh.read(1024 * 1024):
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return

    def do_GET(self) -> None:
        if not self._is_local_request():
            self.send_error(403, "non-local request rejected")
            return
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
            # Y1.039 — per-job progress for the Now-card. Phase-aware,
            # config-bucketed ETA, denoise-step extrapolation. Replaces the
            # old elapsed/global-avg ratio that mis-paced Quick/High renders.
            if payload.get("current"):
                payload["current"]["progress"] = _compute_progress(
                    payload["current"], payload.get("log") or [],
                )
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
        if parsed.path == "/version":
            # Snapshot of the version-check state. Cheap (just a dict copy
            # under a lock); the UI polls this every ~5 minutes to render
            # the "Update available" pill in the header.
            self._json(get_version_state())
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
        if parsed.path.startswith("/webapp/"):
            # Serve the extracted static frontend from <ROOT>/webapp/.
            # Mirrors the /assets/ pattern: resolve and require the final
            # path to live inside the webapp dir (no traversal). Same
            # is_relative_to() check the assets handler uses.
            rel = parsed.path[len("/webapp/"):]
            webapp_dir = (ROOT / "webapp").resolve()
            try:
                path = (webapp_dir / rel).resolve()
            except Exception:
                self.send_error(400); return
            if not path.is_relative_to(webapp_dir) or not path.is_file():
                self.send_error(404); return
            ext = path.suffix.lower()
            ctype = {
                ".html": "text/html; charset=utf-8",
                ".css": "text/css; charset=utf-8",
                ".js": "application/javascript; charset=utf-8",
                ".json": "application/json; charset=utf-8",
                ".svg": "image/svg+xml",
                ".png": "image/png",
                ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".webp": "image/webp",
                ".woff2": "font/woff2",
                ".txt": "text/plain; charset=utf-8",
            }.get(ext, "application/octet-stream")
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(path.stat().st_size))
            # Short cache — dev panel reloads frequently while we iterate.
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(path.read_bytes())
            return
        if parsed.path == "/api/page-config":
            # Bootstrap JSON consumed by webapp/index.html's <script> stub.
            # Used to be inlined into the HTML constant as the __BOOTSTRAP__
            # token; moved here so the HTML can be served as a literal file.
            cfg = _page_config()
            payload = json.dumps(cfg).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(payload)
            return
        if parsed.path == "/image":
            qs = parse_qs(parsed.query)
            try:
                path = Path(qs.get("path", [""])[0]).resolve()
            except Exception:
                self.send_error(400); return
            # Loopback alone isn't enough — a malicious page or extension on
            # the local machine could request /image?path=/etc/shadow. Resolve
            # both sides and require the requested path to live under our
            # OUTPUT, UPLOADS, or STATE_DIR. is_relative_to() already handles
            # the symlink/.. tricks since we resolved both ends.
            try:
                roots = [OUTPUT.resolve(), UPLOADS.resolve(), STATE_DIR.resolve()]
            except Exception:
                roots = []
            if not any(path.is_relative_to(r) for r in roots):
                self.send_error(403); return
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

        # ---- Agentic Flows GETs --------------------------------------------
        if parsed.path == "/agent/config":
            cfg = _load_agent_config()
            local = agent_local_server.status()
            self._json({
                "engine": cfg.to_public_dict(),
                "capabilities": _agent_capabilities(),
                "local_server": local,
                "available_models": agent_local_server.discover_local_models(ROOT),
            })
            return

        if parsed.path == "/agent/sessions":
            self._json({"sessions": agent_runtime.list_sessions(STATE_DIR)})
            return

        if parsed.path == "/agent/notes":
            try:
                from agent import project as _project
                content = _project.read_notes(STATE_DIR)
            except Exception as e:                  # noqa: BLE001
                self._json({"error": str(e)}, 500); return
            self._json({"content": content, "bytes": len(content.encode("utf-8"))})
            return

        if parsed.path.startswith("/agent/sessions/"):
            sid = parsed.path[len("/agent/sessions/"):]
            sid = sid.split("/", 1)[0]
            if not agent_runtime.is_valid_session_id(sid):
                self.send_error(404); return
            sess = agent_runtime.load_session(sid, STATE_DIR)
            if sess is None:
                self.send_error(404); return
            self._json({
                "session": sess.to_dict(),
                "rendered_messages": _render_session_messages(sess),
            })
            return

        if parsed.path == "/agent/local/status":
            self._json({
                "local_server": agent_local_server.status(),
                "available_models": agent_local_server.discover_local_models(ROOT),
            })
            return

        if parsed.path == "/agent/ollama/status":
            self._json(_agent_ollama_status())
            return

        if parsed.path == "/agent/models/search":
            qs = parse_qs(parsed.query)
            q = (qs.get("q", [""])[0] or "").strip()
            abliterated = qs.get("abliterated", ["0"])[0] in ("1", "true", "yes", "on")
            limit = int(qs.get("limit", ["30"])[0] or "30")
            try:
                results = _hf_model_search(q, abliterated=abliterated, limit=limit)
            except RuntimeError as e:
                self._json({"error": str(e)}, 502); return
            self._json({"results": results, "abliterated": abliterated, "query": q})
            return

        if parsed.path == "/agent/models/info":
            qs = parse_qs(parsed.query)
            repo_id = (qs.get("repo_id", [""])[0] or "").strip()
            if not repo_id:
                self._json({"error": "repo_id is required"}, 400); return
            try:
                info = _hf_model_files(repo_id)
            except RuntimeError as e:
                self._json({"error": str(e)}, 502); return
            self._json(info)
            return

        if parsed.path == "/agent/models/install/status":
            self._json(_hf_model_install_status())
            return

        if parsed.path == "/agent/image/config":
            cfg = _load_agent_image_config()
            ok, msg = agent_image_engine.health_check(cfg)
            self._json({
                "image_engine": cfg.to_public_dict(),
                "ok": ok,
                "message": msg,
            })
            return

        self.send_error(404)

    def do_POST(self) -> None:
        if not self._is_local_request():
            self.send_error(403, "non-local request rejected")
            return
        path = self.path.split("?")[0]
        qs = parse_qs(urlparse(self.path).query)
        ctype = self.headers.get("Content-Type", "")

        # Multipart upload
        if path == "/upload" and ctype.startswith("multipart/form-data"):
            # Hard cap on body size so a misbehaving / malicious caller can't
            # spool a multi-GB file into memory via cgi.FieldStorage. 64 MB
            # comfortably covers any reasonable still-image reference; multipart
            # framing adds a small overhead so we read the declared length.
            MAX_UPLOAD_BYTES = 64 * 1024 * 1024
            try:
                clen = int(self.headers.get("Content-Length") or "0")
            except ValueError:
                clen = 0
            if clen <= 0:
                self._json({"error": "Content-Length required"}, 411); return
            if clen > MAX_UPLOAD_BYTES:
                self._json({"error": f"upload too large (max {MAX_UPLOAD_BYTES} bytes)"}, 413)
                return
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

        if path == "/version/check":
            # Force an immediate remote check (UI button on the version
            # pill). Runs synchronously so the client gets the fresh
            # state in the response — at most a 10s round-trip to GitHub.
            try:
                _check_remote_once()
                self._json({"ok": True, "state": get_version_state()})
            except Exception as exc:
                self._json({"ok": False, "error": str(exc)}, 500)
            return

        if path == "/version/pull":
            # The "magic button" path — when the pill is in the behind
            # state and the user clicks it, this endpoint runs git pull
            # on the panel repo and reports back. The user still has to
            # restart phosphene in Pinokio to load the new code (we can't
            # restart ourselves from inside our own process), but we
            # surface that clearly via the pull_state field.
            #
            # If the pulled diff touches dependency manifests / patch
            # scripts (anything that update.js does in addition to
            # `git pull`), we set pull_requires_full_update=True and the
            # UI nudges the user toward Pinokio's full Update flow
            # instead of just Stop+Start.
            with _VERSION_LOCK:
                _VERSION_STATE["pull_state"] = "pulling"
                _VERSION_STATE["pull_message"] = None
                _VERSION_STATE["pull_pulled_to_short"] = None
                _VERSION_STATE["pull_pulled_to_version"] = None
                _VERSION_STATE["pull_requires_full_update"] = False
            try:
                # Capture HEAD before the pull so we can diff afterwards.
                pre_sha = _git_capture(["rev-parse", "HEAD"]) or ""
                # Step 1: fetch — populates origin/main without touching HEAD.
                fetch_proc = subprocess.run(
                    ["git", "-C", str(ROOT), "fetch", "origin"],
                    capture_output=True, timeout=60,
                )
                if fetch_proc.returncode != 0:
                    raise RuntimeError(
                        (fetch_proc.stdout + fetch_proc.stderr).decode("utf-8", "replace").strip()
                        or f"git fetch exited {fetch_proc.returncode}")
                # Step 2: try a fast-forward pull. Happy path for fresh installs
                # whose local history lines up with origin/main.
                pull_proc = subprocess.run(
                    ["git", "-C", str(ROOT), "pull", "--ff-only", "origin", "main"],
                    capture_output=True, timeout=60,
                )
                pull_out = (pull_proc.stdout + pull_proc.stderr).decode("utf-8", "replace").strip()
                # Step 3: if the fast-forward refused (history diverged from
                # origin — e.g. because of a past force-push that scrubbed
                # commit identities), fall back to a hard reset onto
                # origin/main. A Pinokio-installed panel is not a place
                # users keep local commits, so this is the Right Thing —
                # it's what they meant by clicking Update.
                if pull_proc.returncode != 0:
                    reset_proc = subprocess.run(
                        ["git", "-C", str(ROOT), "reset", "--hard", "origin/main"],
                        capture_output=True, timeout=30,
                    )
                    reset_out = (reset_proc.stdout + reset_proc.stderr).decode("utf-8", "replace").strip()
                    if reset_proc.returncode != 0:
                        raise RuntimeError(
                            f"fast-forward refused and reset --hard failed.\n"
                            f"pull: {pull_out}\nreset: {reset_out}")
                    pull_out = (
                        f"history diverged from origin (likely a past force-push); "
                        f"recovered via reset --hard origin/main\n{reset_out}"
                    )
                # Refresh local fields (HEAD, version) before computing the diff.
                _detect_local_install_state()
                post_sha = _git_capture(["rev-parse", "HEAD"]) or ""

                # Did the pull touch anything that needs the heavier Pinokio
                # Update.js (pip reinstalls + patch reapply)? If so, flag it.
                deps_touched = False
                if pre_sha and post_sha and pre_sha != post_sha:
                    diff_out = _git_capture(
                        ["diff", "--name-only", f"{pre_sha}..{post_sha}"]
                    ) or ""
                    deps_signals = (
                        "install.js", "update.js", "pinokio.js", "download_q8.js",
                        "patch_ltx_codec.py", "required_files.json",
                        "requirements.txt", "pyproject.toml", "setup.py",
                    )
                    for line in diff_out.splitlines():
                        if line in deps_signals or line.startswith("ltx-2-mlx/"):
                            deps_touched = True
                            break

                with _VERSION_LOCK:
                    _VERSION_STATE["pull_state"] = "pulled"
                    _VERSION_STATE["pull_message"] = (pull_out.splitlines() or ["pulled"])[-1]
                    _VERSION_STATE["pull_pulled_to_short"] = _VERSION_STATE["local_short"]
                    _VERSION_STATE["pull_pulled_to_version"] = _VERSION_STATE["local_version"]
                    _VERSION_STATE["pull_requires_full_update"] = deps_touched

                # Re-run the remote check so behind_by recalculates to 0
                # (normally) or to whatever new commits landed in the
                # window since we pulled.
                try:
                    _check_remote_once()
                except Exception:
                    pass

                self._json({"ok": True, "state": get_version_state()})
            except subprocess.TimeoutExpired:
                with _VERSION_LOCK:
                    _VERSION_STATE["pull_state"] = "error"
                    _VERSION_STATE["pull_message"] = "git pull timed out (60s)"
                self._json({"ok": False, "error": "git pull timed out", "state": get_version_state()}, 504)
            except Exception as exc:
                with _VERSION_LOCK:
                    _VERSION_STATE["pull_state"] = "error"
                    _VERSION_STATE["pull_message"] = str(exc)
                self._json({"ok": False, "error": str(exc), "state": get_version_state()}, 500)
            return

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
                if not p.is_relative_to(base) or not p.is_file():
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

        # ---- Agentic Flows POSTs -------------------------------------------
        if path == "/agent/config":
            # JSON or urlencoded body. JSON is what the chat UI sends; form
            # is here so curl-from-the-terminal still works.
            try:
                if ctype.startswith("application/json"):
                    payload = json.loads(body or "{}")
                else:
                    payload = {k: v[0] if v else "" for k, v in form.items()}
            except json.JSONDecodeError as e:
                self._json({"error": f"bad JSON: {e}"}, 400); return
            try:
                cfg = _save_agent_config(payload)
            except ValueError as e:
                self._json({"error": str(e)}, 400); return
            push(f"agent: engine config updated ({cfg.kind} → {cfg.model})")
            self._json({"ok": True, "engine": cfg.to_public_dict()})
            return

        if path == "/agent/local/start":
            try:
                if ctype.startswith("application/json"):
                    payload = json.loads(body or "{}")
                else:
                    payload = {k: v[0] if v else "" for k, v in form.items()}
            except json.JSONDecodeError as e:
                self._json({"error": f"bad JSON: {e}"}, 400); return
            override = (payload.get("model_path") or "").strip() or None
            status = _agent_local_start(override)
            self._json({"ok": status.get("running"), "local_server": status})
            return

        if path == "/agent/local/stop":
            status = agent_local_server.stop("manual stop via panel")
            self._json({"ok": True, "local_server": status})
            return

        if path == "/agent/sessions/new":
            try:
                payload = (json.loads(body or "{}")
                           if ctype.startswith("application/json")
                           else {k: v[0] if v else "" for k, v in form.items()})
            except json.JSONDecodeError as e:
                self._json({"error": f"bad JSON: {e}"}, 400); return
            title = (payload.get("title") or "").strip() or "Untitled"
            cfg = _load_agent_config()
            sess = agent_runtime.new_session(title=title, engine_config=cfg)
            agent_runtime.save_session(sess, STATE_DIR)
            push(f"agent: new session {sess.session_id} ({sess.title!r})")
            self._json({"ok": True, "session": sess.to_dict()})
            return

        if path.startswith("/agent/sessions/") and path.endswith("/message"):
            sid = path[len("/agent/sessions/"):-len("/message")]
            if not agent_runtime.is_valid_session_id(sid):
                self._json({"error": "session not found"}, 404); return
            sess = agent_runtime.load_session(sid, STATE_DIR)
            if sess is None:
                self._json({"error": "session not found"}, 404); return
            try:
                payload = (json.loads(body or "{}")
                           if ctype.startswith("application/json")
                           else {k: v[0] if v else "" for k, v in form.items()})
            except json.JSONDecodeError as e:
                self._json({"error": f"bad JSON: {e}"}, 400); return
            user_msg = payload.get("message") or payload.get("content") or ""
            user_msg = user_msg.strip()

            # Validate + normalize attachments. Each path must live under
            # UPLOADS — the client can lie about the path even though it
            # came from a /upload response. Without this check, a malicious
            # page could submit /etc/passwd as an attachment and the agent's
            # tools would happily inspect it.
            raw_atts = payload.get("attachments") or []
            attachments: list[dict] = []
            uploads_root = UPLOADS.resolve()
            for a in raw_atts:
                if not isinstance(a, dict):
                    continue
                p_str = (a.get("path") or "").strip()
                if not p_str:
                    continue
                try:
                    p = Path(p_str).resolve()
                except Exception:
                    continue
                if not p.is_relative_to(uploads_root) or not p.is_file():
                    self._json({"error": f"attachment outside uploads dir or missing: {p_str}"}, 400)
                    return
                attachments.append({
                    "path": str(p),
                    "name": (a.get("name") or p.name)[:200],
                    "mime": (a.get("mime") or "")[:80],
                    "size": int(a.get("size") or 0),
                })
            if not user_msg and not attachments:
                self._json({"error": "message or attachment is required"}, 400); return

            if attachments:
                # Embed an <attachments> JSON block in front of the user text
                # so the agent sees the file references inline. The renderer
                # strips this block back out for display and exposes the
                # parsed list as m.attachments.
                user_msg = (
                    "<attachments>\n"
                    + json.dumps(attachments)
                    + "\n</attachments>\n"
                    + user_msg
                )
            # Always re-load the engine config in case the user just changed
            # it via the settings drawer — the session keeps its own copy
            # for record-keeping but we honor the live config for actual calls.
            sess.engine_config = _load_agent_config()

            # Auto-start the local engine on first message so the user
            # doesn't dead-end on a connection-refused error. mlx_lm.server
            # boots in ~1-2s; weights load lazily on the first
            # /chat/completions call, which our run_turn timeout (300 s)
            # accommodates. Saves a trip to Settings → Start.
            if (sess.engine_config.kind == "phosphene_local"
                    and not agent_local_server.is_running()):
                # Memory guard — refuse to spawn mlx-lm when the Mac is
                # already in swap or close to OOM. Loading a 22 GB Qwen
                # 35B on top of a 64 GB system that's at 90%+ pressure +
                # swap is what put Salo in a force-quit dialog last time.
                # Bail early with an actionable error rather than silently
                # making the system thrash.
                mem = get_memory()
                pressure = mem.get("pressure_pct") or 0
                swap_gb = mem.get("swap_gb") or 0.0
                if pressure >= 92 or swap_gb >= 8:
                    push(f"agent: refusing to auto-start local engine "
                         f"(memory pressure {pressure}%, swap {swap_gb:.1f} GB)")
                    self._json({
                        "error": (
                            f"Memory too high to start the local engine "
                            f"({mem.get('used_gb', 0):.1f}/"
                            f"{mem.get('total_gb', 0):.0f} GB used, "
                            f"swap {swap_gb:.1f} GB). "
                            "Loading a 22 GB chat model on top of this "
                            "would force the Mac into heavy swap. "
                            "Either: (a) wait for current renders to "
                            "finish, (b) switch to a smaller model in "
                            "Settings (Gemma 12B is ~7.5 GB), or (c) "
                            "quit other heavy apps (Claude.app, Chrome) "
                            "and try again."
                        ),
                    }, 503)
                    return
                push("agent: auto-starting local engine for first message")
                _agent_local_start()
                # Wait briefly for /v1/models to respond before we proceed —
                # avoids the very first chat call seeing a half-bound port.
                import urllib.error
                base = sess.engine_config.base_url.rstrip("/") + "/models"
                deadline = time.time() + 30
                ready = False
                while time.time() < deadline:
                    try:
                        with urllib.request.urlopen(base, timeout=2):
                            ready = True
                            break
                    except (urllib.error.URLError, OSError):
                        time.sleep(0.4)
                if not ready:
                    push("agent: local engine spawn timed out; surfacing error")
                    last = agent_local_server.status().get("last_error") or ""
                    self._json({
                        "error": (
                            "Local engine spawned but isn't responding on "
                            f"{base}. Check the Logs tab for mlx-lm output. "
                            + (f" Last status: {last}" if last else "")
                        ),
                    }, 500)
                    return

            ops = _build_panel_ops()
            tools_doc = agent_runtime.render_tools_doc()
            events: list[dict] = []

            # Persist after every event so a client polling
            # /agent/sessions/<id> mid-loop sees the agent's incremental
            # progress (plan, tool call, tool result, next message, ...)
            # rather than a 5-minute silence followed by a single batch.
            # mid-loop disk writes are atomic via runtime.save_session.
            def _on_event(ev: agent_runtime.TurnEvent) -> None:
                try:
                    agent_runtime.save_session(sess, STATE_DIR)
                except Exception as e:                  # noqa: BLE001
                    push(f"agent: mid-loop save failed: {e}")

            try:
                for ev in agent_runtime.run_turn(
                    sess, user_msg, ops, tools_doc=tools_doc,
                    on_event=_on_event,
                ):
                    events.append({"kind": ev.kind, "payload": ev.payload})
            except Exception as e:                       # noqa: BLE001
                push(f"agent: run_turn errored: {e}")
                self._json({"error": str(e), "events": events}, 500); return
            agent_runtime.save_session(sess, STATE_DIR)

            # Plan-and-sleep mode: the agent has just called `finish` which
            # means the plan is queued and the agent is done thinking for
            # this batch. Drop the local chat model's memory NOW so the
            # LTX renderer has the full RAM budget for the overnight run.
            # On a 64 GB Mac, leaving Qwen 35B (22 GB) resident alongside
            # active renders pushes the system into swap — exactly the
            # scenario the user wants to avoid.
            engine_stopped = False
            if (sess.finished
                    and sess.engine_config.kind == "phosphene_local"
                    and getattr(sess.engine_config, "mode", "plan_sleep") == "plan_sleep"
                    and agent_local_server.is_running()):
                try:
                    agent_local_server.stop("plan-and-sleep auto-stop after finish")
                    engine_stopped = True
                    push("agent: plan-and-sleep — stopped local engine to free RAM for renders")
                except Exception as e:                  # noqa: BLE001
                    push(f"agent: failed to auto-stop local engine: {e}")

            self._json({
                "ok": True,
                "events": events,
                "session": sess.to_dict(),
                "rendered_messages": _render_session_messages(sess),
                "engine_stopped": engine_stopped,
            })
            return

        if path.startswith("/agent/sessions/") and path.endswith("/delete"):
            sid = path[len("/agent/sessions/"):-len("/delete")]
            ok = agent_runtime.delete_session(sid, STATE_DIR)
            self._json({"ok": ok})
            return

        if path == "/agent/notes":
            # Manual edit from the Project Notes modal. We overwrite the
            # whole file, so the user gets full control. Empty content
            # collapses to a deletion (file persists with 0 bytes — the
            # excerpt helper handles that).
            try:
                payload = (json.loads(body or "{}")
                           if ctype.startswith("application/json")
                           else {k: v[0] if v else "" for k, v in form.items()})
            except json.JSONDecodeError as e:
                self._json({"error": f"bad JSON: {e}"}, 400); return
            content = payload.get("content")
            if content is None:
                self._json({"error": "content is required"}, 400); return
            if len(content.encode("utf-8")) > 4 * 1024 * 1024:
                self._json({"error": "notes file too large (max 4 MB)"}, 413); return
            from agent import project as _project
            STATE_DIR.mkdir(parents=True, exist_ok=True)
            atomic_write_text(_project.notes_path(STATE_DIR), content)
            self._json({"ok": True, "bytes": len(content.encode("utf-8"))})
            return

        if path.startswith("/agent/sessions/") and path.endswith("/rename"):
            sid = path[len("/agent/sessions/"):-len("/rename")]
            try:
                payload = (json.loads(body or "{}")
                           if ctype.startswith("application/json")
                           else {k: v[0] if v else "" for k, v in form.items()})
            except json.JSONDecodeError as e:
                self._json({"error": f"bad JSON: {e}"}, 400); return
            new_title = (payload.get("title") or "").strip()
            if not new_title:
                self._json({"error": "title is required"}, 400); return
            ok = agent_runtime.rename_session(sid, new_title, STATE_DIR)
            self._json({"ok": ok})
            return

        if path.startswith("/agent/sessions/") and path.endswith("/pin"):
            sid = path[len("/agent/sessions/"):-len("/pin")]
            try:
                payload = (json.loads(body or "{}")
                           if ctype.startswith("application/json")
                           else {k: v[0] if v else "" for k, v in form.items()})
            except json.JSONDecodeError as e:
                self._json({"error": f"bad JSON: {e}"}, 400); return
            pinned = bool(payload.get("pinned", True))
            ok = agent_runtime.set_pinned(sid, pinned, STATE_DIR)
            self._json({"ok": ok, "pinned": pinned})
            return

        # Image-engine config (pluggable: mock | bfl).
        if path == "/agent/image/config":
            try:
                payload = (json.loads(body or "{}")
                           if ctype.startswith("application/json")
                           else {k: v[0] if v else "" for k, v in form.items()})
            except json.JSONDecodeError as e:
                self._json({"error": f"bad JSON: {e}"}, 400); return
            cfg = _save_agent_image_config(payload)
            push(f"agent: image engine updated to {cfg.kind}"
                 + (f" ({cfg.bfl_model})" if cfg.kind == "bfl" else ""))
            ok, msg = agent_image_engine.health_check(cfg)
            self._json({"ok": True, "image_engine": cfg.to_public_dict(),
                        "health_ok": ok, "health_message": msg})
            return

        # Anchor selection — user clicked a thumbnail in the chat.
        # Records {shot_label: candidate_dict} in the session's tool_state.
        if path == "/agent/models/install":
            try:
                payload = (json.loads(body or "{}")
                           if ctype.startswith("application/json")
                           else {k: v[0] if v else "" for k, v in form.items()})
            except json.JSONDecodeError as e:
                self._json({"error": f"bad JSON: {e}"}, 400); return
            repo_id = (payload.get("repo_id") or "").strip()
            if not repo_id:
                self._json({"error": "repo_id is required"}, 400); return
            res = _hf_model_install_async(repo_id)
            if not res.get("ok"):
                self._json(res, 409 if "already" in (res.get("error") or "") else 500)
                return
            self._json(res)
            return

        if path == "/agent/models/install/cancel":
            self._json(_hf_model_install_cancel())
            return

        if path.startswith("/agent/sessions/") and path.endswith("/anchors/select"):
            sid = path[len("/agent/sessions/"):-len("/anchors/select")]
            if not agent_runtime.is_valid_session_id(sid):
                self._json({"error": "session not found"}, 404); return
            sess = agent_runtime.load_session(sid, STATE_DIR)
            if sess is None:
                self._json({"error": "session not found"}, 404); return
            try:
                payload = (json.loads(body or "{}")
                           if ctype.startswith("application/json")
                           else {k: v[0] if v else "" for k, v in form.items()})
            except json.JSONDecodeError as e:
                self._json({"error": f"bad JSON: {e}"}, 400); return
            label = (payload.get("shot_label") or "").strip()
            png_path = (payload.get("png_path") or "").strip()
            if not label or not png_path:
                self._json({"error": "shot_label and png_path are required"}, 400); return
            # Validate path is real and inside our uploads dir (no traversal).
            try:
                p = Path(png_path).resolve()
                _ = p.relative_to(UPLOADS.resolve())
                if not p.is_file():
                    raise ValueError("not a file")
            except (ValueError, OSError) as e:
                self._json({"error": f"invalid png_path: {e}"}, 400); return
            # Find the matching candidate by path (so we keep seed/engine info)
            candidates = ((sess.tool_state.get("anchor_candidates") or {})
                          .get(label, {}).get("candidates")) or []
            picked = next((c for c in candidates if c.get("png_path") == str(p)), None)
            if picked is None:
                # Path didn't match a known candidate — accept anyway with a
                # synthetic record. Useful when the user uploads their own.
                picked = {"png_path": str(p), "engine": "user", "seed": -1}
            sess.tool_state.setdefault("selected_anchors", {})[label] = picked
            agent_runtime.save_session(sess, STATE_DIR)
            push(f"agent: anchor picked for {label}: {p.name}")
            self._json({
                "ok": True,
                "shot_label": label,
                "selected": picked,
                "all_selected": sess.tool_state.get("selected_anchors", {}),
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


# ---- Per-job progress (Y1.039) ----------------------------------------------
# The Now-card progress bar used to be `elapsed / global_avg_of_last_10`,
# which mis-paced Quick renders (crawling because the avg includes Standard +
# High) and High renders (blasting past 99% before decode). The bar also had
# zero phase awareness — denoise-step events the helper emits as tqdm output
# went unused.
#
# Y1.039 fixes both at once:
#   1. ETA is computed per-config from history matching (mode, quality, accel,
#      frames) instead of a global average. Quick uses Quick history.
#   2. Phase weights split each render into setup / denoise / decode / post
#      bands. Bar advances proportional to actual phase progress, not a
#      monotone elapsed/eta ratio.
#   3. Once the helper has reported one denoise step's per-it cost
#      ("30.89s/it"), remaining time = steps_left * per_it + decode_budget.
#      That becomes the dominant signal once denoise begins, so the bar
#      jumps when Boost/Turbo skip a cached step (helper emits the next K/N
#      near-instantly) and stays honest if a step is unusually slow.
#
# Returned `progress` dict is rendered by the Now-card JS — no client-side
# math. Single source of truth for what the user sees.

# Captures step counter in tqdm `Denoising: 12%|...| 1/8 [00:30<03:36, 30.89s/it]`.
_DENOISE_RE = re.compile(r"Denoising[^|]*\|[^|]*\|\s*(\d+)\s*/\s*(\d+)")
_PER_IT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*s/it")


def _parse_progress_signals(log_lines: list[str]) -> dict:
    """Walk the helper log tail (newest-to-oldest) and surface the latest
    phase-defining markers. The panel clears STATE['log'] when a new job
    becomes current (worker_loop), so any log lines passed in are already
    scoped to the running job — no cross-job filtering needed here."""
    denoise_step = denoise_total = None
    last_per_it = None
    decode_started = decode_done = generate_done = upscale_done = pipe_loaded = False
    for ln in reversed(log_lines or []):
        if not isinstance(ln, str) or not ln:
            continue
        if denoise_step is None:
            m = _DENOISE_RE.search(ln)
            if m:
                denoise_step = int(m.group(1))
                denoise_total = int(m.group(2))
                pit = _PER_IT_RE.search(ln)
                if pit:
                    last_per_it = float(pit.group(1))
        if not decode_started and "step:decode_and_save start" in ln:
            decode_started = True
        if not decode_done and "step:decode_and_save done" in ln:
            decode_done = True
        if not generate_done and "step:generate done" in ln:
            generate_done = True
        if not upscale_done and ("Upscale done" in ln or "Sharp upscale" in ln and "done" in ln):
            upscale_done = True
        if not pipe_loaded and "step:get_pipe done" in ln:
            pipe_loaded = True
    return {
        "denoise_step": denoise_step,
        "denoise_total": denoise_total,
        "last_per_it_sec": last_per_it,
        "decode_started": decode_started,
        "decode_done": decode_done,
        "generate_done": generate_done,
        "upscale_done": upscale_done,
        "pipe_loaded": pipe_loaded,
    }


def _bucket_eta(params: dict, lookback: int = 8) -> float | None:
    """Median elapsed_sec of the last `lookback` done jobs whose
    (mode, quality, accel, frames) matches the current job. When upscale is
    on, also requires the same upscale_method (Sharp adds ~26s).

    Falls back to None if no matches — caller should drop to the tier-based
    estimate from CAPABILITIES.times. Width/height intentionally NOT in the
    bucket key — wall time scales near-linearly with pixel count and most
    users render at the tier defaults anyway."""
    mode = params.get("mode")
    quality = params.get("quality")
    accel = params.get("accel")
    frames = params.get("frames")
    upscale = params.get("upscale", "off")
    upscale_method = params.get("upscale_method", "lanczos") if upscale != "off" else "off"
    matches: list[float] = []
    with LOCK:
        for j in STATE["history"]:
            if j.get("status") != "done" or not j.get("elapsed_sec"):
                continue
            p = j.get("params") or {}
            if not (p.get("mode") == mode and p.get("quality") == quality
                    and p.get("accel") == accel and p.get("frames") == frames):
                continue
            if upscale != "off":
                if p.get("upscale", "off") == "off":
                    continue
                if p.get("upscale_method", "lanczos") != upscale_method:
                    continue
            matches.append(j["elapsed_sec"])
            if len(matches) >= lookback:
                break
    if not matches:
        return None
    matches.sort()
    return matches[len(matches) // 2]


def _phase_weights(params: dict) -> dict[str, int]:
    """Phase budget percentages. Sum to 100. Calibrated against today's
    measured timings on Comfortable tier (M-Max 64 GB):
      Standard 121f exact: ~458 s = ~5 s pipe + ~424 s denoise + ~30 s decode
      High Q8 121f:        ~711 s = ~5 s pipe + ~620 s two-stage + ~85 s decode
    Sharp/PiperSR adds ~26 s post-pass which we carve out as `post`."""
    upscale = params.get("upscale", "off")
    upscale_method = params.get("upscale_method", "lanczos")
    quality = params.get("quality", "standard")
    has_post = (upscale != "off" and upscale_method == "pipersr")
    if quality == "high":
        setup, denoise, decode = 2, 86, 12
    elif quality == "quick":
        setup, denoise, decode = 4, 75, 21
    else:  # standard / balanced
        setup, denoise, decode = 3, 82, 15
    post = 0
    if has_post:
        # Steal from decode for the post-pass; keep total at 100.
        post = 5
        decode = max(8, decode - 5)
        setup = max(1, 100 - denoise - decode - post)
    return {"setup": setup, "denoise": denoise, "decode": decode, "post": post}


def _compute_progress(current: dict | None, log_lines: list[str]) -> dict | None:
    if not current:
        return None
    p = current.get("params") or {}
    started = current.get("started_ts") or 0
    if not started:
        return None
    elapsed = max(0.0, time.time() - started)
    eta = _bucket_eta(p)
    signals = _parse_progress_signals(log_lines[-200:] if log_lines else [])
    weights = _phase_weights(p)

    # Phase pick — newest-first markers win. Order: post > decode > denoise > setup.
    if signals["upscale_done"]:
        phase, phase_label = "post", "Finalizing"
    elif signals["decode_done"]:
        phase, phase_label = "post", "Encoding output"
    elif signals["decode_started"]:
        phase, phase_label = "decode", "VAE decode + audio mux"
    elif signals["denoise_step"] is not None and signals["denoise_step"] > 0:
        # tqdm emits a 0/N line at the very start before any step has
        # completed — that's still "preparing" UX-wise. Switch to the
        # "Denoising step K / N" label only once a step is actually done.
        phase = "denoise"
        ds = signals["denoise_step"]; dt = signals["denoise_total"] or 1
        phase_label = f"Denoising · step {ds} / {dt}"
    else:
        phase = "setup"
        phase_label = "Loading pipeline" if not signals["pipe_loaded"] else "Preparing"

    setup_w = weights["setup"]; den_w = weights["denoise"]
    dec_w = weights["decode"]; post_w = weights["post"]

    if phase == "setup":
        # Setup is short — usually <10 s on a warm helper, ~60 s cold.
        # Creep to most of the setup band; don't pin at 0.
        setup_budget = 60.0 if not signals["pipe_loaded"] else 10.0
        within = min(0.92, elapsed / setup_budget) if setup_budget else 0
        pct = within * setup_w
    elif phase == "denoise":
        ds = signals["denoise_step"] or 0
        dt = signals["denoise_total"] or 1
        within = ds / dt if dt else 0
        pct = setup_w + within * den_w
    elif phase == "decode":
        # Within decode: linear creep from 0 → 95% of the band. We don't get
        # progress events from the decoder, so just creep with elapsed.
        # If eta exists, anchor decode budget; else assume ~30 s.
        decode_budget = max(15.0, (eta or 60) * dec_w / 100)
        # Time spent inside decode: rough — use elapsed past the denoise band.
        denoise_done_at = (eta or 60) * (setup_w + den_w) / 100 if eta else None
        if denoise_done_at:
            in_decode = max(0.0, elapsed - denoise_done_at)
            within = min(0.95, in_decode / decode_budget) if decode_budget else 0
        else:
            within = 0.5
        pct = setup_w + den_w + within * dec_w
    else:  # post
        pct = setup_w + den_w + dec_w + 0.6 * post_w
        if signals["upscale_done"]:
            pct = setup_w + den_w + dec_w + post_w

    pct_int = int(round(min(99, max(0, pct))))

    # Remaining time. Per-step extrapolation is the most accurate during
    # denoise; otherwise fall back to bucket eta minus elapsed.
    remaining = None
    if (signals["denoise_step"] is not None and signals["last_per_it_sec"]
            and signals["denoise_total"]):
        ds = signals["denoise_step"]; dt = signals["denoise_total"]
        per_it = signals["last_per_it_sec"]
        steps_left = max(0, dt - ds)
        denoise_left = steps_left * per_it
        # Tail = decode + post. Use eta share if we have eta; else 30 s.
        if eta:
            tail = eta * (dec_w + post_w) / 100
        else:
            tail = 30.0
        remaining = denoise_left + tail
    elif eta:
        remaining = max(0.0, eta - elapsed)

    return {
        "phase": phase,
        "phase_label": phase_label,
        "pct": pct_int,
        "elapsed_sec": round(elapsed, 1),
        "eta_sec": round(eta, 1) if eta else None,
        "remaining_sec": round(remaining, 1) if remaining is not None else None,
        "denoise_step": signals["denoise_step"],
        "denoise_total": signals["denoise_total"],
    }


# ---- HTML --------------------------------------------------------------------
# Frontend extraction (2026-05-06): the HTML/CSS/JS that used to be inlined
# as a giant ~10k-line `HTML = r"""..."""` constant now lives in webapp/
# and is served by the /webapp/* static handler. page() returns
# webapp/index.html verbatim; the bootstrap config (formerly substituted
# into __BOOTSTRAP__) is delivered by GET /api/page-config; the dev profile
# badge (formerly __PROFILE_BADGE__) is a static <span hidden> that the
# bootstrap stub un-hides when profile === "dev".


def _page_config() -> dict:
    """Bootstrap blob the frontend reads from /api/page-config on load."""
    return {
        "presets": PRESETS, "aspects": ASPECTS,
        "default_image": str(REFERENCE),
        "default_audio": str(AUDIO_DEFAULT),
        "fps": FPS, "model": MODEL_ID,
        "profile": PROFILE,
        "port": PORT,
        "model_upscale_enabled": MODEL_UPSCALE_ENABLED,
        "pipersr_upscale_enabled": PIPERSR_UPSCALE_ENABLED,
        # Hardware-aware time estimates for the Quality pills. The pill HTML
        # ships with the Comfortable-tier defaults; on boot we rewrite the
        # subtext using the active tier's quality_times. Compact users see
        # honest "~12 min" instead of the M-Studio's "~7 min" optimism.
        "tier": {
            "key": SYSTEM_TIER,
            "label": SYSTEM_CAPS["label"],
        },
        "quality_times": SYSTEM_CAPS.get("quality_times", {}),
    }


# Cache the read so we're not paying a stat+read on every request. The
# dev workflow is: edit webapp/, refresh — that's a hard refresh, no
# server reload, so we DO want re-reads. Set INDEX_HTML_CACHE = None to
# force re-read on next call.
_INDEX_HTML_CACHE: str | None = None


def page() -> str:
    """Serve webapp/index.html. Always re-reads while PROFILE == 'dev' so
    the dev panel hot-reloads on disk edits without a Python restart."""
    global _INDEX_HTML_CACHE
    if PROFILE == "dev" or _INDEX_HTML_CACHE is None:
        _INDEX_HTML_CACHE = (ROOT / "webapp" / "index.html").read_text()
    return _INDEX_HTML_CACHE


def _diagnose_port_busy(port: int) -> str:
    """Best-effort diagnosis of WHO holds the port, for the pre-flight error
    message. Returns a single human-readable line. Never raises — if any
    detection step fails we just return a generic hint, since the goal is
    a friendlier error than a bare OSError, not a forensic report."""
    # Step 1: ask the existing listener if it's a phosphene panel. If it is,
    # the user almost certainly clicked Start when one was already running
    # (e.g. closed the Pinokio window without Stop). Tell them how to recover.
    try:
        import urllib.request as _urlreq
        with _urlreq.urlopen(f"http://127.0.0.1:{port}/version", timeout=1.5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        v = data.get("local_version") or data.get("local_short") or "?"
        return (f"another phosphene panel is already running on port {port} "
                f"(version {v}). Open it at http://127.0.0.1:{port}, or click "
                f"Stop in Pinokio before clicking Start.")
    except Exception:
        pass
    # Step 2: try lsof to identify whatever process IS listening.
    try:
        out = subprocess.check_output(
            ["lsof", "-nPi", f":{port}"], stderr=subprocess.DEVNULL, timeout=2,
        ).decode("utf-8", "replace")
        # First column is COMMAND, second is PID. Skip the header line.
        lines = [l for l in out.splitlines()[1:] if l.strip()]
        if lines:
            cols = lines[0].split()
            cmd = cols[0] if cols else "?"
            pid = cols[1] if len(cols) > 1 else "?"
            return (f"port {port} is held by another process: {cmd} (pid {pid}). "
                    f"Close it (or run `kill {pid}`) and click Start again.")
    except Exception:
        pass
    return (f"port {port} is already in use by another process. Close whatever "
            f"is on it and click Start again.")


if __name__ == "__main__":
    OUTPUT.mkdir(parents=True, exist_ok=True)
    UPLOADS.mkdir(parents=True, exist_ok=True)
    load_hidden()
    load_queue()
    threading.Thread(target=worker_loop, daemon=True).start()
    if VERSION_CHECK_ENABLED:
        threading.Thread(target=version_check_loop, daemon=True).start()
    else:
        _detect_local_install_state()
    # Pre-flight: bind in a try/except so a busy port surfaces an actionable
    # one-liner instead of a 6-frame Python traceback. The bare OSError
    # ("[Errno 48] Address already in use") was confusing users who'd closed
    # Pinokio without Stop and then clicked Start — they'd see a wall of
    # stack trace pointing at socketserver.py with no hint that the fix was
    # "kill the other panel" or "click Stop first."
    try:
        server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    except OSError as exc:
        if getattr(exc, "errno", None) == 48 or "Address already in use" in str(exc):
            print("─" * 64, flush=True)
            print(f"Phosphene can't start: {_diagnose_port_busy(PORT)}", flush=True)
            print("─" * 64, flush=True)
            sys.exit(1)
        raise
    print(f"LTX MLX Studio: http://127.0.0.1:{PORT}", flush=True)
    print(f"queue: {len(STATE['queue'])} pending, hidden: {len(HIDDEN_PATHS)}", flush=True)
    try:
        server.serve_forever()
    finally:
        HELPER.kill()
        caffeinate_off()
