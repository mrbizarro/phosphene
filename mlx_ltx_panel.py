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


def atomic_write_text(path: Path, text: str) -> None:
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


def _agent_capabilities() -> dict:
    """Hardware-tier snapshot the agent uses to clamp its plan.

    Mirrors the SYSTEM_CAPS table in tier-friendly form so the system
    prompt can show the user what this Mac can actually render. Also
    carries the current image-engine config so the `generate_shot_images`
    tool can dispatch without circular imports.
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
        "image_engine_config": dict(img_cfg.__dict__),
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
    return agent_tools.PanelOps(
        submit_job=_agent_submit_job,
        queue_snapshot=_agent_queue_snapshot,
        find_job=_agent_find_job,
        outputs_dir=OUTPUT,
        uploads_dir=UPLOADS,
        capabilities=_agent_capabilities(),
    )


def _agent_log_sink(line: str) -> None:
    """Forward mlx-lm.server stdout into the panel's Logs tab."""
    push(line)


def _render_session_messages(sess: agent_runtime.Session) -> list[dict]:
    """Format session.messages for the chat UI.

    The raw message list contains:
      - the system prompt (skip in UI)
      - user messages (show as user bubbles)
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
            out.append({"kind": "user", "content": content})
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

        if parsed.path.startswith("/agent/sessions/"):
            sid = parsed.path[len("/agent/sessions/"):]
            sid = sid.split("/", 1)[0]
            if not sid:
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
            cfg = _save_agent_config(payload)
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
            if not user_msg:
                self._json({"error": "message is required"}, 400); return
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
            self._json({
                "ok": True,
                "events": events,
                "session": sess.to_dict(),
                "rendered_messages": _render_session_messages(sess),
            })
            return

        if path.startswith("/agent/sessions/") and path.endswith("/delete"):
            sid = path[len("/agent/sessions/"):-len("/delete")]
            ok = agent_runtime.delete_session(sid, STATE_DIR)
            self._json({"ok": ok})
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

def page() -> str:
    bootstrap = json.dumps({
        "presets": PRESETS, "aspects": ASPECTS,
        "default_image": str(REFERENCE),
        "default_audio": str(AUDIO_DEFAULT),
        "fps": FPS, "model": MODEL_ID,
        "profile": PROFILE,
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
    })
    # Profile badge — only visible in the dev panel. Lets Salo tell at a
    # glance which install he's looking at when both panels are open.
    profile_badge = (
        '<span class="profile-badge" title="Dev panel · pulls from `dev` branch · port '
        + str(PORT) + '">DEV</span>'
        if PROFILE == "dev" else ""
    )
    return (HTML
            .replace("__BOOTSTRAP__", bootstrap)
            .replace("__PROFILE_BADGE__", profile_badge))


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
    /* DEV badge — visible only on the dev panel (PROFILE == 'dev'). Sits
       next to the Phosphene wordmark so it's the first thing you notice
       when comparing dev vs production tabs. */
    .profile-badge {
      display: inline-flex; align-items: center;
      margin-left: 12px;
      padding: 3px 9px;
      border-radius: 5px;
      background: rgba(240,185,64,0.18);
      color: var(--warning, #f0b940);
      border: 1px solid rgba(240,185,64,0.55);
      font-size: 10px; font-weight: 700;
      letter-spacing: 0.12em;
      cursor: help;
    }
    /* Phosphene 2.0 release badge — sits next to the wordmark logo so
       users immediately know they're on the major version bump. Brand
       violet vs the DEV badge's amber so they don't visually clash when
       both are shown on the dev panel. */
    .version-badge {
      display: inline-flex; align-items: center;
      margin-left: 10px;
      padding: 3px 10px;
      border-radius: 5px;
      background: rgba(140,120,255,0.16);
      color: #b6a4ff;
      border: 1px solid rgba(140,120,255,0.45);
      font-size: 11px; font-weight: 700;
      letter-spacing: 0.06em;
    }
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
    /* Version pill states. Always rendered so the spot is part of the
       user's mental map — when state changes, the colour shift draws
       the eye. Only `pill-update` glows; the other states are quiet. */
    .pill-update {
      color: var(--warning, #f0b940);
      border-color: rgba(240,185,64,0.55);
      background: rgba(240,185,64,0.08);
      font-weight: 600;
      animation: glow-update 2.4s ease-in-out infinite;
    }
    .pill-update:hover { background: rgba(240,185,64,0.18); }
    @keyframes glow-update {
      0%,100% { box-shadow: 0 0 0 0 rgba(240,185,64,0.0); }
      50%     { box-shadow: 0 0 0 3px rgba(240,185,64,0.18); }
    }
    /* Current — install is on origin/main HEAD. Subtle success tint so
       it reads as "all good" without competing for attention. */
    .pill-current {
      color: var(--success, #8ec07c);
      border-color: rgba(63,185,80,0.35);
    }
    .pill-current:hover { background: rgba(63,185,80,0.06); }
    /* Dev — non-main branch / dirty tree / no git. We don't know if the
       user is "behind"; surfacing the SHA in muted text just helps them
       confirm what's running. */
    .pill-dev {
      color: var(--muted);
      border-color: var(--border);
    }
    .pill-dev:hover { color: var(--text); }
    /* Checking — first /version poll hasn't completed yet. Same look as
       dev (muted) so the pill doesn't flash colours in the first 30s of
       boot before the GitHub fetch lands. */
    .pill-checking {
      color: var(--muted);
      border-color: var(--border);
    }
    /* Restart — set after a successful /version/pull. The user needs to
       click Stop+Start in Pinokio for the new code to take effect; this
       state nudges them to do that. Accent-blue so it reads as an
       actionable next-step, not a problem. */
    .pill-restart {
      color: var(--accent-bright, #7e98ff);
      border-color: rgba(126,152,255,0.55);
      background: rgba(126,152,255,0.08);
      font-weight: 600;
      animation: glow-update 2.4s ease-in-out infinite;
    }
    .pill-restart:hover { background: rgba(126,152,255,0.18); }
    /* Busy — temporary in-flight state during /version/check and
       /version/pull round-trips. Faded so the user knows the click
       registered without us flashing alarming colours. */
    .pill-busy {
      opacity: 0.7; pointer-events: none;
      animation: none !important;
    }
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

    /* Agentic Flows is a chat — it deserves more horizontal real estate
       than the manual form. When data-workflow="agent" is on the body,
       the form-pane grows to ~55% and the agent-stage-pane on the right
       takes over for showing live queue / renders / outputs. */
    body[data-workflow="agent"] .layout {
      grid-template-columns: minmax(560px, 1.2fr) minmax(440px, 1fr);
    }
    body[data-workflow="agent"] .form-pane { padding: 0; }
    body[data-workflow="agent"] .stage-pane { display: none; }
    body[data-workflow="agent"] .agent-stage-pane { display: flex; }

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
    textarea.avoid-textarea { min-height: 54px; }

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

    /* Quality pills (Y1.013) — richer than the standard pill-btn so each
       button can carry: name, dimensions+time, model+tier requirement.
       Same .pill-btn skeleton; just looser padding and explicit children
       for the three lines of content. */
    .quality-row .pill-quality {
      padding: 12px 8px 10px;
      gap: 4px;
      min-height: 76px;
      justify-content: flex-start;
    }
    .pill-quality .ql-name {
      font-size: 13px; font-weight: 600;
      color: inherit;            /* picks up muted/active colour from .pill-btn */
    }
    .pill-quality .ql-spec {
      font-size: 10.5px;
      color: var(--text);
      opacity: 0.85;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      letter-spacing: 0;
    }
    .pill-quality .ql-tier {
      font-size: 9.5px;
      color: var(--muted);
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }
    .pill-quality.active .ql-name { color: var(--accent-bright); }
    .pill-quality.active .ql-spec { opacity: 1; }
    .pill-quality.active .ql-tier { color: var(--accent-bright); opacity: 0.7; }

    /* Customize disclosure inside the form — sub-tier UI, lighter than
       a top-level <details>. Subtle border, indented body, distinct
       chevron so it doesn't compete with the LoRAs section header. */
    .customize-section {
      margin-top: 8px;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: rgba(255,255,255,0.012);
      overflow: hidden;
    }
    .customize-section > summary { list-style: none; cursor: pointer; }
    .customize-section > summary::-webkit-details-marker { display: none; }
    .cz-summary {
      display: flex; align-items: center; gap: 10px;
      padding: 8px 12px;
      font-size: 12px;
      color: var(--muted);
      user-select: none;
      transition: background 100ms;
    }
    .cz-summary:hover { background: rgba(255,255,255,0.02); color: var(--text); }
    .customize-section[open] .cz-summary { color: var(--text); }
    .cz-summary .cz-chevron {
      display: inline-block; width: 12px;
      font-size: 10px;
      transform: rotate(-90deg);
      transition: transform 140ms ease;
      color: var(--muted);
    }
    .customize-section[open] .cz-summary .cz-chevron {
      transform: rotate(0deg);
      color: var(--accent-bright, #58a6ff);
    }
    .cz-summary .cz-title {
      font-weight: 600;
      font-size: 12px;
      letter-spacing: 0;
    }
    .cz-summary .cz-meta {
      margin-left: auto;
      font-size: 11px; font-weight: 400;
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }
    .cz-body {
      padding: 12px 14px 14px;
      border-top: 1px solid var(--border);
      display: flex; flex-direction: column; gap: 14px;
    }
    .cz-control { display: block; }
    .cz-label {
      font-size: 11px; font-weight: 600;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 6px;
      display: flex; align-items: baseline; gap: 8px;
    }
    .cz-label .cz-label-hint {
      font-weight: 400;
      letter-spacing: 0;
      text-transform: none;
      font-size: 10.5px;
      color: var(--muted);
      opacity: 0.85;
    }
    .cz-control .row { gap: 8px; }

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
    /* ⓘ info button overlaid on the card thumbnail. Subtle until hover so
       it doesn't compete with the video preview itself. Click → opens
       outputInfoModal with the full sidecar data. */
    .car-card { position: relative; }
    .car-card .car-info-btn {
      position: absolute; top: 6px; right: 6px;
      width: 24px; height: 24px; padding: 0;
      border-radius: 6px;
      border: 1px solid rgba(0,0,0,0.5);
      background: rgba(15,18,28,0.7); backdrop-filter: blur(4px);
      color: rgba(255,255,255,0.85);
      font-size: 14px; line-height: 1;
      display: inline-flex; align-items: center; justify-content: center;
      cursor: pointer; opacity: 0; transition: opacity 100ms, background 100ms;
      z-index: 2;
    }
    .car-card:hover .car-info-btn,
    .car-card.active .car-info-btn { opacity: 1; }
    .car-card .car-info-btn:hover { background: rgba(20,25,40,0.92); color: #fff; }

    /* Output info modal — clean detail-pane styling. Reuses
       .models-modal scaffolding for the dim backdrop and centred card,
       but the body is intentionally chrome-light: no nested boxes, no
       coloured uppercase headers. Sections are separated by spacing
       and a thin top-border on the section title — closer to a Linear
       detail pane than a "developer console" dump. */
    .output-info-body {
      display: flex; flex-direction: column;
      gap: 22px;
      max-height: 70vh; overflow-y: auto;
      padding: 4px 6px 4px 0;       /* breathing room around the scrollbar */
    }
    .oi-section { /* no border, no background — separation is purely typographic */ }
    .oi-section-title {
      font-size: 11px;
      font-weight: 600;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin: 0 0 12px;
      padding-bottom: 6px;
      border-bottom: 1px solid var(--border);
      display: flex; align-items: center; justify-content: space-between;
      gap: 10px;
    }
    .oi-section-title .oi-count {
      font-weight: 400; letter-spacing: 0;
      text-transform: none; color: var(--muted);
    }
    .oi-grid {
      display: grid; grid-template-columns: 96px 1fr;
      row-gap: 9px; column-gap: 16px;
      margin: 0;
    }
    .oi-grid dt {
      color: var(--muted);
      font-size: 12px;
      font-weight: 400;
    }
    .oi-grid dd {
      margin: 0;
      color: var(--text);
      font-size: 13px;
      display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
      min-width: 0;
    }
    .oi-grid dd code {
      background: rgba(255,255,255,0.04);
      padding: 1px 6px; border-radius: 3px;
      font-size: 11.5px;
      color: var(--text);
    }
    .oi-prompt {
      font-size: 12.5px; line-height: 1.55; color: var(--text);
      background: rgba(255,255,255,0.025);
      border-radius: 5px;
      padding: 12px 14px;
      max-height: 240px; overflow-y: auto;
      white-space: pre-wrap; word-break: break-word;
    }
    .oi-copy {
      font-size: 11px;
      padding: 3px 10px;
      border-radius: 4px;
      border: 1px solid var(--border);
      background: transparent;
      color: var(--muted);
      cursor: pointer; font-weight: 500;
      letter-spacing: 0;
      transition: color 100ms, border-color 100ms, background 100ms;
    }
    .oi-copy:hover {
      color: var(--text); border-color: var(--accent);
      background: rgba(47,129,247,0.07);
    }
    /* LoRA list — flat rows, separated by hairlines instead of each
       row being its own card. Reads as a list, not a stack of cards. */
    .oi-lora-list {
      display: flex; flex-direction: column;
    }
    .oi-lora-row {
      display: flex; align-items: center; gap: 14px;
      padding: 9px 0; font-size: 13px;
    }
    .oi-lora-row + .oi-lora-row { border-top: 1px solid var(--border); }
    .oi-lora-row .oi-lora-name {
      flex: 1; color: var(--text); font-weight: 500;
      overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
      min-width: 0;
    }
    .oi-lora-row .oi-lora-strength {
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      color: var(--muted); font-size: 12px; flex: none;
    }
    .oi-actions {
      display: flex; gap: 8px; justify-content: flex-end;
      margin-top: 4px;
      padding-top: 14px;
      border-top: 1px solid var(--border);
    }
    .oi-actions .oi-primary {
      padding: 7px 14px;
      border-radius: 6px;
      background: var(--accent); color: white;
      border: 1px solid var(--accent);
      font-size: 12px; font-weight: 600;
      cursor: pointer;
      transition: background 100ms;
    }
    .oi-actions .oi-primary:hover { background: var(--accent-bright, #58a6ff); }
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

    /* LoRA picker — collapsible <details>. Compact list rows (Y1.007).
       Distinct visual section so users notice it (Y1.008): wrapped in a
       bordered container, separator dividers above/below, custom chevron
       in the summary header so it's obviously expandable.
       Each LoRA = one ~36px row. Active rows expand to show a strength
       slider inline. Filter input appears at 5+ LoRAs. */

    /* Thin horizontal separator used both above and below the LoRAs
       section. Reused in case we want it elsewhere in the form. */
    .form-divider {
      height: 1px; margin: 16px 0 14px;
      background: linear-gradient(to right,
        transparent, var(--border) 18%, var(--border) 82%, transparent);
    }
    /* Bordered container wrapping the whole <details>. Subtle accent
       background so the section pulls the eye but doesn't shout. */
    .loras-section {
      border: 1px solid var(--border);
      border-radius: 10px;
      background: rgba(255,255,255,0.015);
      overflow: hidden;     /* keeps the summary's hover bg inside the radius */
    }
    /* Hide the native disclosure marker; we render our own chevron
       inline so it can be sized + animated consistently. */
    .loras-section > summary { list-style: none; }
    .loras-section > summary::-webkit-details-marker { display: none; }
    .loras-summary {
      cursor: pointer; user-select: none;
      display: flex; align-items: center; gap: 10px;
      padding: 11px 14px;
      font-size: 13px; font-weight: 600; color: var(--text);
      transition: background 100ms;
    }
    .loras-summary:hover { background: rgba(255,255,255,0.02); }
    .loras-section[open] .loras-summary {
      border-bottom: 1px solid var(--border);
      background: rgba(255,255,255,0.025);
    }
    .loras-summary .loras-chevron {
      display: inline-block; width: 14px;
      font-size: 11px; color: var(--muted);
      transform: rotate(-90deg);    /* points right when collapsed */
      transition: transform 140ms ease;
      text-align: center;
    }
    .loras-section[open] .loras-summary .loras-chevron {
      transform: rotate(0deg);      /* points down when open */
      color: var(--accent-bright, #93a8ff);
    }
    .loras-summary .loras-title {
      font-size: 13px; font-weight: 600; letter-spacing: 0.01em;
    }
    .loras-summary .loras-meta {
      margin-left: auto; font-size: 11px; font-weight: 400;
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }
    /* Action buttons in the LoRAs section header. Rescan is icon-only,
       Browse CivitAI is the primary CTA. Both stop propagation so they
       don't toggle the <details>. */
    .loras-summary .loras-header-actions {
      display: inline-flex; gap: 6px; align-items: center;
      margin-left: 10px;
    }
    .loras-summary .loras-icon-btn {
      width: 28px; height: 28px; padding: 0;
      border-radius: 6px;
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.02);
      color: var(--muted);
      font-size: 14px; line-height: 1;
      cursor: pointer; display: inline-flex; align-items: center;
      justify-content: center;
      transition: color 100ms, border-color 100ms, background 100ms;
    }
    .loras-summary .loras-icon-btn:hover {
      color: var(--text); border-color: var(--accent);
      background: rgba(47,129,247,0.08);
    }
    .loras-summary .loras-browse-btn {
      padding: 6px 12px;
      border-radius: 6px;
      border: 1px solid var(--accent);
      background: var(--accent);
      color: white;
      font-size: 11.5px; font-weight: 600;
      cursor: pointer;
      letter-spacing: 0.01em;
      white-space: nowrap;
      transition: background 100ms, transform 80ms;
    }
    .loras-summary .loras-browse-btn:hover {
      background: var(--accent-bright, #58a6ff);
    }
    .loras-summary .loras-browse-btn:active { transform: translateY(1px); }
    /* Body padding + internal layout. All children get consistent
       spacing without each one declaring its own margin. */
    .loras-body {
      display: flex; flex-direction: column;
      gap: 10px; padding: 12px 14px 14px;
    }
    .lora-filter {
      width: 100%; padding: 6px 9px; border-radius: 6px;
      border: 1px solid var(--border); background: var(--bg-2, #0a0c14);
      color: var(--text); font-size: 12px;
      box-sizing: border-box;
    }
    .lora-filter:focus { outline: 1px solid var(--accent); }
    .loras-list {
      display: flex; flex-direction: column; gap: 4px; margin-top: 6px;
      max-height: 340px; overflow-y: auto;
      padding-right: 2px;        /* avoid scrollbar overlap on row content */
    }
    .lora-row {
      position: relative;
      border-radius: 6px;
      border: 1px solid var(--border);
      background: var(--panel-2);
      transition: border-color 100ms, background 100ms;
    }
    .lora-row.active {
      border-color: var(--accent);
      background: var(--accent-dim, rgba(47,129,247,0.06));
    }
    .lora-row .lora-row-main {
      display: grid;
      grid-template-columns: 18px 1fr auto auto;
      gap: 8px; align-items: center;
      padding: 7px 10px;
      cursor: pointer; user-select: none;
    }
    .lora-row .lora-toggle-dot {
      width: 14px; height: 14px; border-radius: 50%;
      border: 1.5px solid var(--muted);
      background: transparent;
      transition: background 100ms, border-color 100ms;
      box-sizing: border-box;
    }
    .lora-row.active .lora-toggle-dot {
      background: var(--accent); border-color: var(--accent);
      box-shadow: inset 0 0 0 3px var(--bg, #0a0c14);
    }
    .lora-row .lora-name {
      font-size: 12px; font-weight: 500; color: var(--text);
      overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
      min-width: 0;
    }
    .lora-row .lora-name .badge {
      display: inline-block; font-size: 9px; font-weight: 600;
      letter-spacing: 0.04em; text-transform: uppercase;
      padding: 1px 5px; border-radius: 999px; margin-left: 6px;
      border: 1px solid var(--accent); color: var(--accent-bright);
      vertical-align: middle;
    }
    .lora-row .lora-name-meta {
      font-size: 10px; color: var(--muted); margin-top: 1px;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }
    .lora-row .lora-text {
      min-width: 0; display: flex; flex-direction: column;
    }
    .lora-row .lora-row-actions {
      display: flex; gap: 4px; align-items: center;
      opacity: 0.4; transition: opacity 100ms;
    }
    .lora-row:hover .lora-row-actions,
    .lora-row.active .lora-row-actions { opacity: 1; }
    .lora-row .lora-icon-btn {
      width: 22px; height: 22px; padding: 0;
      border-radius: 4px; border: 1px solid transparent;
      background: transparent;
      color: var(--muted); font-size: 12px; line-height: 1;
      cursor: pointer; display: inline-flex; align-items: center;
      justify-content: center; text-decoration: none;
    }
    .lora-row .lora-icon-btn:hover {
      color: var(--text); border-color: var(--border);
    }
    .lora-row .lora-icon-btn.danger:hover {
      color: #ff8a8a; border-color: rgba(220,80,80,0.5);
    }
    /* Expanded section (visible only on active rows) — strength slider
       + trigger chips. Shows below the main row. */
    .lora-row .lora-row-extra { display: none; }
    .lora-row.active .lora-row-extra {
      display: block;
      padding: 0 10px 9px;
      border-top: 1px dashed var(--border);
      margin-top: 0;
    }
    .lora-row .lora-strength-row {
      display: flex; align-items: center; gap: 8px;
      padding-top: 7px;
    }
    .lora-row .lora-strength-row label {
      font-size: 10px; text-transform: uppercase; letter-spacing: 0.04em;
      color: var(--muted); width: 52px; flex: none;
    }
    .lora-row .lora-strength-row input[type="range"] {
      flex: 1; min-width: 0; accent-color: var(--accent);
    }
    .lora-row .lora-strength-row input[type="number"] {
      width: 54px; padding: 2px 5px; font-size: 11px; text-align: right;
    }
    .lora-row .trigger-chips {
      display: flex; flex-wrap: wrap; gap: 3px; margin-top: 7px;
    }
    .lora-row .trigger-chip {
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 10px; padding: 2px 7px; border-radius: 999px;
      background: rgba(255,255,255,0.04);
      border: 1px solid var(--border); color: var(--text);
      cursor: pointer; user-select: none;
      transition: background 80ms, border-color 80ms;
    }
    .lora-row .trigger-chip:hover {
      background: rgba(90,124,255,0.18); border-color: var(--accent);
    }
    .lora-row .trigger-chip.empty {
      color: var(--muted); font-style: italic;
      cursor: default; background: transparent; border: none; padding: 0;
      font-family: inherit;
    }
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

    /* Inline CivitAI auth banner. Lives at the top of the browser modal,
       above the search bar. Three states: missing (amber, has input),
       set (green check, dismissable into a "change" link), or testing/error. */
    .civitai-auth {
      margin: 4px 0 12px;
      padding: 10px 12px;
      border-radius: 8px;
      font-size: 12px; line-height: 1.45;
      display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
    }
    .civitai-auth.missing {
      background: rgba(240,185,64,0.07);
      border: 1px solid rgba(240,185,64,0.4);
      color: var(--text);
    }
    .civitai-auth.missing strong { color: var(--warning, #f0b940); }
    .civitai-auth.set {
      background: rgba(63,185,80,0.06);
      border: 1px solid rgba(63,185,80,0.35);
      color: var(--text);
    }
    .civitai-auth.err {
      background: rgba(220,80,80,0.06);
      border: 1px solid rgba(220,80,80,0.4);
      color: var(--text);
    }
    .civitai-auth .grow { flex: 1; min-width: 200px; }
    .civitai-auth input[type="password"],
    .civitai-auth input[type="text"] {
      flex: 1; min-width: 180px;
      padding: 6px 8px; border-radius: 5px;
      border: 1px solid var(--border); background: var(--bg-2, #0a0c14);
      color: var(--text); font-size: 12px;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }
    .civitai-auth button {
      padding: 6px 12px; border-radius: 5px;
      background: var(--accent); color: white; border: none;
      font-size: 12px; font-weight: 600; cursor: pointer;
    }
    .civitai-auth button:disabled { opacity: 0.6; cursor: default; }
    .civitai-auth a.changekey {
      color: var(--accent-bright, #7e98ff);
      cursor: pointer; text-decoration: none; font-size: 11px;
    }
    .civitai-auth a.changekey:hover { text-decoration: underline; }

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
    /* Spicy mode toggle row — explicit state badge + button.
       OFF: muted neutral chip, button reads "Enable Spicy mode"
       ARMED (mid-confirm): amber chip, button reads "Click again to confirm"
       ON: amber chip, button reads "Disable" (single click off — easy to
       turn off, harder to turn on, intentional per user spec). */
    .spicy-row {
      display: flex; align-items: center; gap: 12px;
      padding: 10px 12px;
      background: var(--panel-2);
      border: 1px solid var(--border);
      border-radius: 8px;
    }
    .spicy-state {
      display: inline-flex; align-items: center;
      padding: 3px 10px; border-radius: 999px;
      font-size: 11px; font-weight: 700;
      letter-spacing: 0.08em;
      border: 1px solid var(--border);
      color: var(--muted);
      background: rgba(255,255,255,0.03);
    }
    .spicy-state.on {
      color: var(--warning, #f0b940);
      border-color: rgba(240,185,64,0.55);
      background: rgba(240,185,64,0.16);
    }
    .spicy-state.armed {
      color: var(--warning, #f0b940);
      border-color: rgba(240,185,64,0.85);
      background: rgba(240,185,64,0.26);
      animation: spicyArm 1.2s ease-in-out infinite;
    }
    @keyframes spicyArm {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.5; }
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
    /* The primary action on each token row. Sized small to fit inline,
       but visually weighted as the main button so users naturally click
       it instead of leaving the modal without saving. */
    .token-row .token-savetest {
      padding: 5px 11px;
      font-size: 11.5px;
      font-weight: 600;
    }

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

    /* ============== AGENTIC FLOWS — chat UI =====================
       Aim: feel like Claude.ai — generous typography, soft surfaces,
       avatars, expandable tool cards, markdown rendering. The form-pane
       gets a top strip switching between the manual form and the chat;
       the chat itself runs full-height with header / scroll / composer.
    */
    .workflow-tabs {
      display: flex; gap: 4px;
      padding: 5px; margin: 0 0 14px 0;
      background: var(--panel); border: 1px solid var(--border);
      border-radius: 12px;
    }
    .workflow-tabs button {
      flex: 1; padding: 9px 14px;
      background: transparent; color: var(--muted);
      border: 1px solid transparent; border-radius: 8px;
      font-size: 13px; font-weight: 600; cursor: pointer;
      letter-spacing: 0.2px;
      transition: background 0.15s, color 0.15s, border-color 0.15s;
    }
    .workflow-tabs button:hover { color: var(--text); }
    .workflow-tabs button.active {
      background: var(--accent-dim); color: var(--accent-bright);
      border-color: var(--accent);
    }
    .workflow-tabs .new-badge {
      display: inline-block; margin-left: 6px;
      font-size: 9px; font-weight: 700; letter-spacing: 0.5px;
      padding: 2px 6px; border-radius: 999px;
      background: var(--accent); color: white;
      vertical-align: middle;
    }

    /* ---- Pane wrapper ---- */
    .agent-pane {
      display: flex; flex-direction: column;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      overflow: hidden;
      --agent-text: 14px;
      --agent-line: 1.6;
      --agent-bubble-pad: 14px 16px;
      --agent-radius: 12px;
      --avatar: 28px;
    }

    /* ---- Header ---- */
    .agent-header {
      display: flex; align-items: center; gap: 10px;
      padding: 12px 16px;
      border-bottom: 1px solid rgba(255,255,255,0.04);
      background: linear-gradient(180deg, var(--panel-2), var(--panel));
    }
    .agent-header .engine-pill {
      display: inline-flex; align-items: center; gap: 8px;
      padding: 5px 11px 5px 9px;
      background: var(--bg-2); color: var(--text);
      border: 1px solid var(--border); border-radius: 999px;
      font-size: 12px; font-weight: 500;
      cursor: pointer;
      transition: border-color 0.15s, background 0.15s;
      max-width: 320px;
      overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }
    .agent-header .engine-pill:hover { border-color: var(--accent); }
    .agent-header .engine-pill .dot {
      width: 7px; height: 7px; border-radius: 50%;
      background: var(--muted); flex-shrink: 0;
    }
    .agent-header .engine-pill .dot.live {
      background: #2ea043;
      box-shadow: 0 0 0 3px rgba(46,160,67,0.18);
    }
    .agent-header .engine-pill .dot.warn { background: #d29922; }
    .agent-header .engine-pill .dot.bad { background: #cf222e; }
    .agent-header .session-title {
      flex: 1; min-width: 0;
      font-size: 13px; font-weight: 500; color: var(--text);
      overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }
    .agent-header .session-title .meta {
      font-size: 11px; color: var(--muted); margin-left: 8px;
    }
    .agent-header .icon-btn {
      width: 30px; height: 30px;
      background: transparent; color: var(--muted);
      border: 1px solid var(--border); border-radius: 8px;
      cursor: pointer;
      display: inline-flex; align-items: center; justify-content: center;
      font-size: 14px;
      transition: color 0.15s, border-color 0.15s;
    }
    .agent-header .icon-btn:hover {
      color: var(--accent-bright); border-color: var(--accent);
    }

    /* ---- Sessions dropdown ---- */
    .agent-sessions-pop {
      position: absolute;
      top: 52px; right: 16px;
      width: min(380px, calc(100% - 32px));
      max-height: 320px; overflow-y: auto;
      background: var(--panel);
      border: 1px solid var(--border-strong);
      border-radius: 12px;
      box-shadow: 0 14px 40px rgba(0,0,0,0.5);
      z-index: 50;
      display: none;
      padding: 6px;
    }
    .agent-sessions-pop.open { display: block; }
    .agent-sessions-pop .item {
      display: block; padding: 9px 12px;
      border-radius: 8px;
      cursor: pointer; color: var(--text);
      font-size: 13px;
      transition: background 0.12s;
    }
    .agent-sessions-pop .item:hover { background: var(--bg-2); }
    .agent-sessions-pop .item.active {
      background: var(--accent-dim); color: var(--accent-bright);
    }
    .agent-sessions-pop .item .meta {
      font-size: 11px; color: var(--muted);
      margin-top: 3px;
      display: flex; gap: 10px;
    }
    .agent-sessions-pop .empty {
      padding: 14px; text-align: center;
      font-size: 12px; color: var(--muted); font-style: italic;
    }

    /* ---- Chat scroll area ---- */
    .agent-chat {
      flex: 1;
      min-height: 360px; max-height: 60vh;
      overflow-y: auto; overflow-x: hidden;
      padding: 24px 20px 8px;
      scroll-behavior: smooth;
    }
    .agent-chat::-webkit-scrollbar { width: 8px; }
    .agent-chat::-webkit-scrollbar-track { background: transparent; }
    .agent-chat::-webkit-scrollbar-thumb {
      background: var(--border-strong); border-radius: 4px;
    }
    .agent-chat::-webkit-scrollbar-thumb:hover { background: var(--accent); }

    /* ---- Empty state ---- */
    .agent-empty {
      max-width: 540px; margin: 24px auto 12px;
      text-align: center; color: var(--muted);
      animation: agent-fade-in 0.4s ease;
    }
    .agent-empty .badge {
      display: inline-flex; align-items: center; gap: 8px;
      padding: 6px 12px; border-radius: 999px;
      background: var(--accent-dim); color: var(--accent-bright);
      border: 1px solid var(--accent);
      font-size: 11px; font-weight: 600; letter-spacing: 0.5px;
      text-transform: uppercase;
      margin-bottom: 16px;
    }
    .agent-empty h3 {
      font-size: 18px; font-weight: 600; color: var(--text);
      margin: 0 0 10px;
    }
    .agent-empty p {
      font-size: 13px; line-height: 1.55; margin: 6px 0;
    }
    .agent-empty .examples {
      display: flex; flex-direction: column; gap: 8px;
      max-width: 420px; margin: 24px auto 0;
    }
    .agent-empty .example {
      text-align: left;
      padding: 12px 14px;
      background: var(--bg-2);
      border: 1px solid var(--border); border-radius: 10px;
      cursor: pointer;
      color: var(--text); font-size: 13px;
      transition: border-color 0.15s, background 0.15s;
      display: flex; align-items: center; gap: 10px;
    }
    .agent-empty .example:hover {
      border-color: var(--accent);
      background: var(--panel);
    }
    .agent-empty .example .arrow {
      color: var(--muted); flex-shrink: 0;
      transition: transform 0.15s, color 0.15s;
    }
    .agent-empty .example:hover .arrow {
      color: var(--accent-bright);
      transform: translateX(2px);
    }

    /* ---- Message rows ---- */
    .agent-msg-row {
      display: flex; gap: 12px;
      align-items: flex-start;
      margin-bottom: 22px;
      animation: agent-fade-in 0.3s ease;
    }
    @keyframes agent-fade-in {
      from { opacity: 0; transform: translateY(6px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .agent-avatar {
      width: var(--avatar); height: var(--avatar);
      border-radius: 50%;
      display: inline-flex; align-items: center; justify-content: center;
      font-size: 12px; font-weight: 700;
      flex-shrink: 0;
      letter-spacing: 0.3px;
      user-select: none;
    }
    .agent-avatar.claude {
      background: linear-gradient(135deg, #cc7a3a, #d18b4f);
      color: white;
      box-shadow: 0 0 0 1px rgba(204,122,58,0.3);
    }
    .agent-avatar.user {
      background: var(--panel-2);
      color: var(--text);
      border: 1px solid var(--border-strong);
    }
    .agent-msg-body { flex: 1; min-width: 0; }
    .agent-msg-name {
      font-size: 12px; font-weight: 600;
      color: var(--text);
      margin-bottom: 4px;
      letter-spacing: 0.2px;
    }
    .agent-msg-content {
      font-size: var(--agent-text);
      line-height: var(--agent-line);
      color: var(--text);
      word-wrap: break-word; overflow-wrap: break-word;
    }

    /* ---- Markdown rendering inside .agent-md ---- */
    .agent-md > *:first-child { margin-top: 0; }
    .agent-md > *:last-child { margin-bottom: 0; }
    .agent-md p { margin: 8px 0; }
    .agent-md h1, .agent-md h2, .agent-md h3, .agent-md h4 {
      font-weight: 600; color: var(--text); margin: 16px 0 8px;
      line-height: 1.3;
    }
    .agent-md h1 { font-size: 17px; }
    .agent-md h2 { font-size: 15px; }
    .agent-md h3 { font-size: 14px; color: var(--accent-bright); }
    .agent-md h4 { font-size: 13px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.4px; }
    .agent-md ul, .agent-md ol { margin: 8px 0; padding-left: 22px; }
    .agent-md li { margin: 4px 0; }
    .agent-md li > p { margin: 2px 0; }
    .agent-md strong { font-weight: 600; color: var(--text); }
    .agent-md em { font-style: italic; color: var(--muted); }
    .agent-md a { color: var(--accent-bright); text-decoration: underline; }
    .agent-md code {
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 0.92em;
      padding: 2px 6px;
      background: rgba(255,255,255,0.05);
      border: 1px solid var(--border);
      border-radius: 4px;
      color: var(--accent-bright);
    }
    .agent-md pre {
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px 14px;
      margin: 10px 0;
      overflow-x: auto;
      font-size: 12px;
      line-height: 1.5;
    }
    .agent-md pre code {
      background: transparent; border: none; padding: 0;
      color: var(--text);
      font-size: inherit;
    }
    .agent-md blockquote {
      margin: 10px 0;
      padding: 6px 14px;
      border-left: 3px solid var(--accent);
      color: var(--muted);
      background: var(--accent-dim);
      border-radius: 0 6px 6px 0;
    }
    .agent-md table {
      border-collapse: collapse;
      margin: 10px 0;
      font-size: 12px;
      width: 100%;
    }
    .agent-md th, .agent-md td {
      border: 1px solid var(--border);
      padding: 7px 10px;
      text-align: left;
      vertical-align: top;
    }
    .agent-md th {
      background: var(--bg-2);
      font-weight: 600;
      color: var(--text);
    }
    .agent-md hr {
      border: none; border-top: 1px solid var(--border);
      margin: 14px 0;
    }

    /* ---- Tool cards (calls + results, expandable) ---- */
    .agent-tool-card {
      margin: 10px 0;
      background: var(--bg-2);
      border: 1px solid var(--border);
      border-left: 3px solid var(--accent);
      border-radius: 10px;
      overflow: hidden;
      transition: border-color 0.15s;
      animation: agent-fade-in 0.25s ease;
    }
    .agent-tool-card:hover { border-color: var(--accent); }
    .agent-tool-card.success { border-left-color: #2ea043; }
    .agent-tool-card.error { border-left-color: #cf222e; }
    .agent-tool-card.pending { border-left-color: var(--accent); }
    .agent-tool-card .head {
      padding: 10px 14px;
      display: flex; align-items: center; gap: 10px;
      cursor: pointer;
      user-select: none;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 12px;
    }
    .agent-tool-card .head:hover { background: var(--panel); }
    .agent-tool-card .head .icon {
      font-size: 13px; flex-shrink: 0; line-height: 1;
    }
    .agent-tool-card .head .icon.success { color: #2ea043; }
    .agent-tool-card .head .icon.error { color: #cf222e; }
    .agent-tool-card .head .name {
      font-weight: 700; color: var(--accent-bright);
      flex-shrink: 0;
    }
    .agent-tool-card.success .head .name { color: #9be7a4; }
    .agent-tool-card.error .head .name { color: #f49a9e; }
    .agent-tool-card .head .summary {
      color: var(--muted); flex: 1; min-width: 0;
      overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }
    .agent-tool-card .head .chevron {
      color: var(--muted); flex-shrink: 0;
      transition: transform 0.2s ease;
      font-size: 11px;
    }
    .agent-tool-card.open .head .chevron { transform: rotate(90deg); }
    .agent-tool-card .body {
      display: none;
      padding: 0 14px 12px;
      border-top: 1px solid var(--border);
      margin-top: 0;
    }
    .agent-tool-card.open .body {
      display: block;
      padding-top: 10px;
    }
    .agent-tool-card .body pre {
      margin: 0;
      padding: 10px 12px;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 6px;
      font-size: 11px;
      line-height: 1.45;
      max-height: 280px;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      color: var(--text);
    }

    /* ---- Anchor candidate grid (Phase B of the director workflow) ---- */
    .anchor-grid-wrap {
      padding: 12px 14px 14px;
      border-top: 1px solid var(--border);
    }
    .anchor-grid-meta {
      font-size: 11px; color: var(--muted);
      margin-bottom: 8px;
      display: flex; align-items: center; gap: 8px;
    }
    .anchor-grid-meta .label-pill {
      padding: 2px 8px; border-radius: 999px;
      background: var(--accent-dim); color: var(--accent-bright);
      font-weight: 600; font-size: 10px;
      letter-spacing: 0.4px; text-transform: uppercase;
    }
    .anchor-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
      gap: 10px;
    }
    .anchor-cell {
      position: relative;
      aspect-ratio: 16/9;
      border-radius: 8px;
      overflow: hidden;
      cursor: pointer;
      border: 2px solid transparent;
      background: var(--bg);
      padding: 0;
      transition: border-color 0.15s, transform 0.15s;
    }
    .anchor-cell:hover {
      border-color: var(--accent);
      transform: scale(1.02);
    }
    .anchor-cell.selected {
      border-color: #2ea043;
      box-shadow: 0 0 0 3px rgba(46,160,67,0.25);
    }
    .anchor-cell img {
      width: 100%; height: 100%;
      object-fit: cover;
      display: block;
    }
    .anchor-cell .check {
      position: absolute; top: 6px; right: 6px;
      width: 22px; height: 22px;
      border-radius: 50%;
      background: var(--bg-2);
      color: var(--muted);
      display: flex; align-items: center; justify-content: center;
      font-size: 12px;
      font-weight: 800;
      border: 1px solid var(--border);
      transition: background 0.15s, color 0.15s;
    }
    .anchor-cell.selected .check {
      background: #2ea043;
      color: white;
      border-color: #2ea043;
    }
    .anchor-cell .seed {
      position: absolute; bottom: 4px; left: 4px;
      font-size: 9px;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      padding: 2px 6px;
      background: rgba(0,0,0,0.55);
      color: white;
      border-radius: 4px;
      letter-spacing: 0.3px;
    }
    .anchor-cell .engine-tag {
      position: absolute; bottom: 4px; right: 4px;
      font-size: 9px;
      padding: 2px 5px;
      background: rgba(0,0,0,0.55);
      color: white;
      border-radius: 4px;
      text-transform: uppercase;
      letter-spacing: 0.4px;
    }
    .anchor-prompt {
      font-size: 11px; color: var(--muted);
      line-height: 1.5;
      margin-bottom: 10px;
      padding: 8px 10px;
      background: var(--bg);
      border-radius: 6px;
      border: 1px solid var(--border);
      font-style: italic;
      max-height: 80px; overflow-y: auto;
    }

    /* ============================================================
       AGENT STAGE PANE — the right-side "live canvas" / code-interpreter
       ============================================================
       Shows what the agent is actually doing with the renderer in real
       time: current job + progress, queue, generated anchors waiting on
       a pick, finished mp4s playing inline. Polls /status every ~1.5s.

       Hidden by default; appears when body[data-workflow="agent"] OR
       body.agent-fullscreen. Cards are layered (top: now, middle:
       activity feed, bottom: gallery of session outputs) and scroll
       independently. */
    .agent-stage-pane {
      display: none;
      flex-direction: column;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      overflow: hidden;
      min-height: 0;
    }
    .agent-stage-head {
      padding: 12px 16px;
      border-bottom: 1px solid rgba(255,255,255,0.05);
      display: flex; align-items: center; gap: 10px;
      flex-shrink: 0;
      background: linear-gradient(180deg, rgba(255,255,255,0.02), transparent);
    }
    .agent-stage-head .label {
      font-size: 10px; font-weight: 700;
      text-transform: uppercase; letter-spacing: 0.5px;
      color: var(--muted);
    }
    .agent-stage-head .live-dot {
      width: 8px; height: 8px; border-radius: 50%;
      background: var(--muted);
    }
    .agent-stage-head .live-dot.live {
      background: #2ea043;
      box-shadow: 0 0 0 3px rgba(46,160,67,0.18);
      animation: stage-pulse 1.6s ease-in-out infinite;
    }
    @keyframes stage-pulse {
      0%, 100% { box-shadow: 0 0 0 3px rgba(46,160,67,0.18); }
      50% { box-shadow: 0 0 0 6px rgba(46,160,67,0.05); }
    }
    .agent-stage-head .session-pill {
      font-size: 10px; padding: 3px 8px;
      background: var(--bg-2); border: 1px solid var(--border);
      border-radius: 999px; color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      flex: 0 0 auto;
    }
    .agent-stage-head .spacer { flex: 1; }
    .agent-stage-head .stage-tab {
      background: transparent; color: var(--muted);
      border: 1px solid transparent; border-radius: 7px;
      padding: 5px 10px; font-size: 11px; cursor: pointer;
      width: auto;
    }
    .agent-stage-head .stage-tab.active {
      background: var(--accent-dim); color: var(--accent-bright);
      border-color: var(--accent);
    }
    .agent-stage-head .stage-tab:hover:not(.active) {
      color: var(--text);
    }
    .agent-stage-body {
      flex: 1 1 auto;
      overflow-y: auto; overflow-x: hidden;
      padding: 14px 16px 18px;
      display: flex; flex-direction: column; gap: 14px;
      min-height: 0;
    }
    .agent-stage-body::-webkit-scrollbar { width: 8px; }
    .agent-stage-body::-webkit-scrollbar-thumb {
      background: rgba(255,255,255,0.06); border-radius: 4px;
    }
    .agent-stage-body::-webkit-scrollbar-thumb:hover {
      background: rgba(255,255,255,0.14);
    }
    .agent-stage-section {
      background: var(--bg-2);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px 14px;
    }
    .agent-stage-section h4 {
      margin: 0 0 8px; font-size: 10px;
      text-transform: uppercase; letter-spacing: 0.5px;
      color: var(--muted); font-weight: 700;
      display: flex; align-items: center; gap: 8px;
    }
    .agent-stage-section h4 .count {
      padding: 1px 7px; border-radius: 999px;
      background: var(--accent-dim); color: var(--accent-bright);
      font-weight: 700;
    }

    /* "Now rendering" card */
    .stage-now-card {
      padding: 14px 14px 12px;
    }
    .stage-now-card.idle {
      text-align: center; color: var(--muted);
      font-size: 12px; padding: 24px 14px;
      font-style: italic;
    }
    .stage-now-label {
      font-size: 13px; font-weight: 600;
      color: var(--text); margin-bottom: 4px;
      overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }
    .stage-now-meta {
      font-size: 11px; color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      margin-bottom: 10px;
    }
    .stage-progress-bar {
      height: 6px; border-radius: 3px;
      background: rgba(255,255,255,0.06);
      overflow: hidden;
      margin-bottom: 4px;
    }
    .stage-progress-fill {
      height: 100%;
      background: linear-gradient(90deg, var(--accent), var(--accent-bright));
      transition: width 600ms ease;
      box-shadow: 0 0 12px rgba(47,129,247,0.4);
    }
    .stage-progress-text {
      font-size: 10px; color: var(--muted);
      display: flex; justify-content: space-between;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }

    /* Session outputs gallery */
    .stage-outputs {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
      gap: 8px;
    }
    .stage-output-cell {
      position: relative;
      aspect-ratio: 16/9;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
      cursor: pointer;
      transition: border-color 0.15s, transform 0.15s;
    }
    .stage-output-cell:hover {
      border-color: var(--accent);
      transform: translateY(-1px);
    }
    .stage-output-cell.failed {
      border-color: rgba(207,34,46,0.5);
      background: rgba(207,34,46,0.04);
    }
    .stage-output-cell .vid {
      width: 100%; height: 100%; object-fit: cover; display: block;
    }
    .stage-output-cell .label {
      position: absolute; bottom: 0; left: 0; right: 0;
      padding: 4px 8px;
      background: linear-gradient(180deg, transparent, rgba(0,0,0,0.7));
      color: white;
      font-size: 10px;
      overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }
    .stage-output-cell .badge {
      position: absolute; top: 4px; left: 4px;
      font-size: 9px; padding: 2px 6px;
      background: rgba(0,0,0,0.6); color: white;
      border-radius: 4px;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      letter-spacing: 0.3px;
    }
    .stage-output-cell.failed .badge {
      background: rgba(207,34,46,0.85);
    }
    /* Refine button — small circular icon top-right of every output cell.
       Distinct from the main click area (which opens the lightbox to
       play the clip). Click → drops a reference into the chat composer
       so the user can ask the agent for a variation. */
    .stage-output-cell .refine-btn {
      position: absolute; top: 4px; right: 4px;
      width: 22px; height: 22px;
      border-radius: 50%;
      background: rgba(0,0,0,0.6);
      color: white;
      border: 1px solid rgba(255,255,255,0.15);
      font-size: 11px;
      display: flex; align-items: center; justify-content: center;
      cursor: pointer;
      opacity: 0;
      transition: opacity 0.15s, background 0.15s;
      z-index: 1;
      width: 22px;
    }
    .stage-output-cell:hover .refine-btn { opacity: 1; }
    .stage-output-cell .refine-btn:hover {
      background: var(--accent);
      border-color: var(--accent);
    }
    .stage-empty {
      text-align: center; color: var(--muted);
      font-size: 12px; padding: 16px;
      font-style: italic;
    }

    /* Activity feed (recent agent actions) */
    .stage-activity {
      display: flex; flex-direction: column; gap: 6px;
      max-height: 220px; overflow-y: auto;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 11px;
    }
    .stage-activity-row {
      display: flex; align-items: baseline; gap: 8px;
      padding: 4px 0;
      border-bottom: 1px solid rgba(255,255,255,0.03);
    }
    .stage-activity-row:last-child { border-bottom: none; }
    .stage-activity-row .icon {
      width: 14px; flex-shrink: 0;
      font-size: 11px;
    }
    .stage-activity-row.ok .icon { color: #2ea043; }
    .stage-activity-row.fail .icon { color: #cf222e; }
    .stage-activity-row.run .icon { color: var(--accent-bright); }
    .stage-activity-row .text {
      flex: 1; color: var(--text);
      overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }
    .stage-activity-row .when {
      color: var(--muted); font-size: 10px;
      flex-shrink: 0;
    }

    /* ============== HF MODEL BROWSER MODAL ============== */
    .model-browser-modal {
      position: fixed; inset: 0;
      background: rgba(0,2,12,0.7); backdrop-filter: blur(6px);
      z-index: 220;
      display: none;
      align-items: center; justify-content: center;
      animation: agent-fade-in 0.2s ease;
    }
    .model-browser-modal.open { display: flex; }
    .model-browser-card {
      width: min(820px, 96vw); max-height: 88vh;
      display: flex; flex-direction: column;
      background: var(--panel);
      border: 1px solid var(--border-strong);
      border-radius: 14px;
      box-shadow: 0 24px 60px rgba(0,0,0,0.6);
      overflow: hidden;
    }
    .model-browser-head {
      padding: 18px 22px 14px;
      border-bottom: 1px solid var(--border);
      display: flex; align-items: center; gap: 10px;
    }
    .model-browser-head h2 {
      margin: 0; font-size: 17px; font-weight: 600;
      text-transform: none; letter-spacing: -0.1px;
      color: var(--text);
      flex: 1;
    }
    .model-browser-head .subtitle {
      font-size: 11px; color: var(--muted);
      letter-spacing: 0.3px; text-transform: uppercase;
      flex: 0 0 auto;
    }
    .model-browser-head .close-btn {
      width: auto; padding: 5px 12px;
      background: transparent; color: var(--muted);
      border: 1px solid var(--border);
      border-radius: 8px; font-size: 11px; font-weight: 600;
      cursor: pointer;
    }
    .model-browser-head .close-btn:hover {
      color: var(--text); border-color: var(--accent);
    }
    .model-browser-controls {
      padding: 14px 22px;
      display: flex; align-items: center; gap: 10px;
      border-bottom: 1px solid rgba(255,255,255,0.04);
      background: var(--bg-2);
    }
    .model-browser-controls input[type="text"] {
      flex: 1; padding: 9px 12px;
      background: var(--panel); color: var(--text);
      border: 1px solid var(--border);
      border-radius: 8px; font-size: 13px;
      width: auto;
    }
    .model-browser-controls input[type="text"]:focus {
      outline: none; border-color: var(--accent);
      box-shadow: 0 0 0 3px var(--accent-dim);
    }
    .model-browser-controls .checkbox-label {
      display: flex; align-items: center; gap: 6px;
      font-size: 12px; color: var(--muted);
      cursor: pointer; user-select: none;
      flex: 0 0 auto;
    }
    .model-browser-controls .checkbox-label input {
      width: auto; margin: 0;
    }
    .model-browser-controls button {
      padding: 9px 16px; border-radius: 8px;
      border: none; background: var(--accent); color: white;
      font-size: 12px; font-weight: 600; cursor: pointer;
      width: auto;
    }
    .model-browser-controls button:hover { background: var(--accent-bright); }
    .model-browser-controls button[disabled] { opacity: 0.5; cursor: not-allowed; }
    .model-browser-results {
      flex: 1; overflow-y: auto;
      padding: 8px 14px 18px;
    }
    .model-result {
      padding: 12px 14px; margin: 6px 0;
      background: var(--bg-2); border: 1px solid var(--border);
      border-radius: 10px;
      display: grid; grid-template-columns: 1fr auto;
      gap: 14px;
      transition: border-color 0.15s;
    }
    .model-result:hover { border-color: var(--accent); }
    .model-result .info { min-width: 0; }
    .model-result .name {
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 13px; color: var(--text);
      font-weight: 600;
      overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }
    .model-result .meta {
      display: flex; flex-wrap: wrap; gap: 10px;
      margin-top: 4px;
      font-size: 11px; color: var(--muted);
    }
    .model-result .meta .tag {
      padding: 1px 7px; border-radius: 999px;
      background: rgba(255,255,255,0.04);
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 10px;
    }
    .model-result .meta .tag.gated {
      background: rgba(207,34,46,0.12); color: #f49a9e;
      border: 1px solid rgba(207,34,46,0.3);
    }
    .model-result .meta .tag.abliterated {
      background: rgba(204,122,58,0.12); color: #ffb37a;
      border: 1px solid rgba(204,122,58,0.3);
    }
    .model-result .actions {
      display: flex; align-items: center; gap: 8px;
      flex-shrink: 0;
    }
    .model-result .install-btn {
      padding: 7px 14px; border-radius: 7px;
      background: var(--accent-dim); color: var(--accent-bright);
      border: 1px solid var(--accent);
      font-size: 11px; font-weight: 600; cursor: pointer;
      width: auto;
      transition: background 0.15s, color 0.15s;
    }
    .model-result .install-btn:hover {
      background: var(--accent); color: white;
    }
    .model-result .install-btn[disabled] { opacity: 0.5; cursor: not-allowed; }
    .model-result .info-btn {
      padding: 7px 10px; border-radius: 7px;
      background: transparent; color: var(--muted);
      border: 1px solid var(--border);
      font-size: 11px; cursor: pointer; width: auto;
    }
    .model-result .info-btn:hover {
      color: var(--text); border-color: var(--accent);
    }
    .model-browser-status {
      padding: 12px 22px;
      border-top: 1px solid var(--border);
      background: var(--bg-2);
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 11px; color: var(--muted);
      display: none;
    }
    .model-browser-status.visible { display: block; }
    .model-browser-status .label {
      color: var(--accent-bright); font-weight: 700;
      margin-right: 8px;
    }
    .model-browser-status .last-line {
      max-height: 60px; overflow-y: auto;
      white-space: pre-wrap; word-break: break-all;
      margin-top: 6px;
      color: var(--text);
    }
    .model-browser-empty {
      text-align: center; padding: 60px 20px;
      color: var(--muted); font-size: 13px;
    }

    /* Lightbox: clicking a stage-output-cell plays the video full-size */
    .stage-lightbox {
      position: fixed; inset: 0;
      background: rgba(0,2,12,0.85);
      backdrop-filter: blur(8px);
      z-index: 250;
      display: none;
      align-items: center; justify-content: center;
      padding: 40px;
    }
    .stage-lightbox.open { display: flex; }
    .stage-lightbox video {
      max-width: 90vw; max-height: 80vh;
      border-radius: 12px;
      box-shadow: 0 24px 80px rgba(0,0,0,0.6);
    }
    .stage-lightbox .close-btn {
      position: absolute;
      top: 16px; right: 20px;
      background: rgba(255,255,255,0.08);
      color: white;
      border: 1px solid rgba(255,255,255,0.1);
      border-radius: 8px;
      padding: 8px 14px;
      font-size: 12px;
      cursor: pointer;
      width: auto;
    }
    .stage-lightbox .close-btn:hover {
      background: rgba(255,255,255,0.14);
    }
    .stage-lightbox .refine-btn-large {
      position: absolute;
      top: 16px; right: 110px;
      background: var(--accent);
      color: white;
      border: 1px solid var(--accent);
      border-radius: 8px;
      padding: 8px 16px;
      font-size: 12px; font-weight: 600;
      cursor: pointer; width: auto;
      box-shadow: 0 4px 14px rgba(47,129,247,0.4);
    }
    .stage-lightbox .refine-btn-large:hover {
      background: var(--accent-bright);
    }

    /* ---- Composer reference chip ----
       When the user clicks Refine on a clip, a chip appears just above
       the textarea with the clip's label + a × to clear it. The chip's
       data is sent verbatim as a prefix to the user's next message so
       the agent picks up the reference and calls inspect_clip first. */
    .agent-ref-chip {
      display: none;
      align-items: center; gap: 8px;
      padding: 6px 10px 6px 12px;
      margin: 0 0 6px 0;
      background: var(--accent-dim);
      border: 1px solid var(--accent);
      border-radius: 999px;
      font-size: 12px;
      color: var(--accent-bright);
      max-width: fit-content;
      animation: agent-fade-in 0.2s ease;
    }
    .agent-ref-chip.visible { display: inline-flex; }
    .agent-ref-chip .ref-icon {
      font-size: 11px;
    }
    .agent-ref-chip .ref-label {
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 11px;
      max-width: 320px;
      overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }
    .agent-ref-chip .clear {
      background: transparent;
      color: var(--accent-bright);
      border: none;
      width: 18px; height: 18px;
      border-radius: 50%;
      cursor: pointer;
      font-size: 14px;
      line-height: 1;
      display: flex; align-items: center; justify-content: center;
    }
    .agent-ref-chip .clear:hover {
      background: rgba(255,255,255,0.1);
    }

    /* =========================================================
       FULLSCREEN / FOCUS MODE — pro-app polish
       =========================================================
       Toggle via .agent-fullscreen on <body>. Hides every chrome
       element, paints a soft radial-gradient backdrop, centers
       the conversation in a comfortable reading column, and
       floats the composer with a glass-blur effect so the chat
       fades under it.
       Esc exits — bound in JS. */

    /* Wipe all the regular panel chrome.
       NOTE: `body.agent-fullscreen > header` (direct child only) — without
       the > combinator the rule also matches the agent's own
       <header class="agent-header"> inside the chat, which is exactly
       what we DON'T want to hide.
       Stage pane (right column) STAYS visible — it becomes the agent's
       live canvas / code-interpreter view. */
    body.agent-fullscreen > header,
    body.agent-fullscreen .bottom-pane,
    body.agent-fullscreen .form-pane > :not(.agent-pane),
    body.agent-fullscreen .workflow-tabs,
    body.agent-fullscreen .stage-pane {
      display: none !important;
    }
    body.agent-fullscreen { overflow: hidden; }
    /* Fullscreen layout: chat on the left, stage pane on the right.
       Like Claude.ai with Artifacts open — conversation + live canvas. */
    body.agent-fullscreen .layout {
      max-width: none; padding: 0; margin: 0; gap: 0;
      height: 100vh;
      display: grid;
      grid-template-columns: 1fr minmax(420px, 480px);
    }
    body.agent-fullscreen .form-pane {
      max-width: none; padding: 0;
      width: 100%; height: 100vh;
      border: none; border-radius: 0;
      background: transparent;
    }
    body.agent-fullscreen .agent-stage-pane {
      display: flex !important;
      border: none; border-radius: 0;
      border-left: 1px solid rgba(255,255,255,0.05);
      background: rgba(0, 6, 26, 0.3);
      backdrop-filter: blur(8px);
      height: 100vh; width: 100%;
    }

    /* Soft body backdrop — subtle radial wash gives depth without
       fighting the conversation. */
    body.agent-fullscreen {
      background:
        radial-gradient(ellipse 80% 60% at 50% -10%, rgba(47,129,247,0.06), transparent 70%),
        radial-gradient(ellipse 70% 50% at 50% 110%, rgba(204,122,58,0.04), transparent 70%),
        var(--bg);
      background-attachment: fixed;
    }

    /* The pane fills the screen, no borders, no card */
    body.agent-fullscreen .agent-pane {
      width: 100vw; height: 100vh; max-height: 100vh;
      border: none; border-radius: 0;
      background: transparent;
      overflow: hidden;
      display: flex; flex-direction: column;
    }

    /* Header: minimal, transparent, generous padding, no bottom rule */
    body.agent-fullscreen .agent-header {
      background: transparent;
      border-bottom: none;
      padding: 16px 28px;
      max-width: 1080px; width: 100%;
      margin: 0 auto;
      box-sizing: border-box;
    }
    body.agent-fullscreen .agent-header .engine-pill {
      background: rgba(255,255,255,0.04);
      border-color: rgba(255,255,255,0.06);
    }
    body.agent-fullscreen .agent-header .engine-pill:hover {
      background: rgba(47,129,247,0.08);
      border-color: var(--accent);
    }
    body.agent-fullscreen .agent-header .icon-btn {
      background: transparent;
      border: 1px solid transparent;
    }
    body.agent-fullscreen .agent-header .icon-btn:hover {
      background: rgba(255,255,255,0.05);
      border-color: rgba(255,255,255,0.08);
      color: var(--accent-bright);
    }
    body.agent-fullscreen .agent-header .session-title {
      font-size: 14px;
      letter-spacing: -0.1px;
    }
    body.agent-fullscreen .agent-header .session-title .meta {
      font-size: 11px;
      opacity: 0.55;
    }

    /* Chat: takes the rest, content column centered at reading width */
    body.agent-fullscreen .agent-chat {
      max-height: none;
      flex: 1;
      padding: 8px 28px 0;
      scroll-padding-bottom: 140px;
    }
    body.agent-fullscreen .agent-chat::-webkit-scrollbar { width: 10px; }
    body.agent-fullscreen .agent-chat::-webkit-scrollbar-thumb {
      background: rgba(255,255,255,0.06);
      border-radius: 4px;
      border: 2px solid transparent;
      background-clip: padding-box;
    }
    body.agent-fullscreen .agent-chat::-webkit-scrollbar-thumb:hover {
      background: rgba(255,255,255,0.14);
      background-clip: padding-box;
    }

    /* Center every direct child of the chat at reading width */
    body.agent-fullscreen .agent-chat > * {
      max-width: 720px;
      margin-left: auto; margin-right: auto;
      width: 100%;
    }
    body.agent-fullscreen .agent-empty {
      max-width: 580px; margin: 80px auto 24px;
    }
    body.agent-fullscreen .agent-msg-row {
      margin-bottom: 26px;
    }
    body.agent-fullscreen .agent-avatar {
      width: 30px; height: 30px;
      box-shadow: 0 0 0 1px rgba(255,255,255,0.04);
    }
    body.agent-fullscreen .agent-avatar.claude {
      box-shadow: 0 0 0 1px rgba(204,122,58,0.4),
                  0 4px 16px rgba(204,122,58,0.18);
    }

    /* Composer: floating glass card with soft shadow + fading
       gradient above so chat appears to dissolve under it. */
    body.agent-fullscreen .agent-composer {
      background: transparent;
      border-top: none;
      padding: 0 28px 24px;
      position: relative;
      z-index: 5;
    }
    body.agent-fullscreen .agent-composer::before {
      content: '';
      position: absolute; inset: -56px 0 0 0;
      pointer-events: none;
      background: linear-gradient(
        180deg,
        transparent 0%,
        rgba(0,6,26,0.5) 30%,
        rgba(0,6,26,0.85) 65%,
        var(--bg) 100%
      );
    }
    body.agent-fullscreen .agent-composer-wrap {
      max-width: 720px; width: 100%;
      margin: 0 auto;
      position: relative; z-index: 1;
    }
    body.agent-fullscreen .agent-composer textarea {
      background: rgba(20, 26, 58, 0.85);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      border: 1px solid rgba(255,255,255,0.07);
      border-radius: 18px;
      padding: 14px 60px 14px 18px;          /* room for the inline send button */
      box-shadow:
        0 12px 36px rgba(0,0,0,0.45),
        0 0 0 1px rgba(47,129,247,0.0);
      font-size: 14.5px;
      transition: border-color 0.2s, box-shadow 0.2s, background 0.2s;
    }
    body.agent-fullscreen .agent-composer textarea:focus {
      background: rgba(28, 36, 72, 0.92);
      border-color: rgba(47,129,247,0.5);
      box-shadow:
        0 12px 36px rgba(0,0,0,0.5),
        0 0 0 4px rgba(47,129,247,0.12);
    }
    body.agent-fullscreen .agent-composer .hint {
      bottom: -22px; left: 0;
      width: 100%; text-align: center;
      opacity: 0.5;
    }

    /* Send button polish for fullscreen — slightly larger, deeper shadow */
    body.agent-fullscreen .agent-composer .send-btn {
      width: 38px; height: 38px;
      right: 8px; bottom: 8px;
      border-radius: 11px;
      z-index: 2;
      box-shadow: 0 6px 20px rgba(47,129,247,0.45);
    }
    body.agent-fullscreen .agent-composer .send-btn:hover:not(:disabled) {
      box-shadow: 0 8px 28px rgba(47,129,247,0.55);
    }
    body.agent-fullscreen .agent-composer .send-btn:disabled {
      background: rgba(255,255,255,0.05);
      box-shadow: none;
    }
    body.agent-fullscreen .agent-composer .send-btn svg {
      width: 17px; height: 17px;
    }

    /* Tool cards: tighten + soften in fullscreen */
    body.agent-fullscreen .agent-tool-card {
      background: rgba(255,255,255,0.025);
      border-color: rgba(255,255,255,0.06);
    }
    body.agent-fullscreen .agent-tool-card:hover {
      border-color: rgba(47,129,247,0.4);
    }
    body.agent-fullscreen .agent-tool-card .head {
      padding: 9px 14px;
      font-size: 12px;
    }
    body.agent-fullscreen .agent-tool-card .body {
      padding: 0 14px 12px;
    }

    /* Anchor grid in fullscreen: roomier cells, no card border around them */
    body.agent-fullscreen .anchor-grid {
      grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
      gap: 12px;
    }
    body.agent-fullscreen .anchor-cell {
      border-width: 1px;
      border-color: rgba(255,255,255,0.06);
      box-shadow: 0 4px 14px rgba(0,0,0,0.25);
    }
    body.agent-fullscreen .anchor-cell:hover {
      transform: translateY(-2px) scale(1.01);
      box-shadow: 0 8px 24px rgba(0,0,0,0.35);
    }

    /* Empty state in fullscreen — magazine-feel layout */
    body.agent-fullscreen .agent-empty {
      padding: 40px 0 60px;
    }
    body.agent-fullscreen .agent-empty h3 {
      font-size: 26px; font-weight: 600;
      letter-spacing: -0.4px;
      margin: 14px 0 12px;
    }
    body.agent-fullscreen .agent-empty p {
      font-size: 14px; line-height: 1.65;
      max-width: 460px; margin: 8px auto;
    }
    body.agent-fullscreen .agent-empty .examples {
      gap: 10px; margin-top: 32px;
    }
    body.agent-fullscreen .agent-empty .example {
      padding: 14px 16px;
      background: rgba(255,255,255,0.025);
      border-color: rgba(255,255,255,0.06);
      border-radius: 12px;
    }
    body.agent-fullscreen .agent-empty .example:hover {
      background: rgba(47,129,247,0.06);
      border-color: rgba(47,129,247,0.4);
    }

    /* Sessions popover anchored to the header — keep it usable in fullscreen */
    body.agent-fullscreen .agent-sessions-pop {
      top: 60px;
      right: 28px;
      box-shadow: 0 16px 48px rgba(0,0,0,0.55);
    }

    /* Fullscreen entrance */
    body.agent-fullscreen .agent-pane {
      animation: agent-fullscreen-in 0.28s cubic-bezier(0.2, 0.7, 0.2, 1);
    }
    @keyframes agent-fullscreen-in {
      from { opacity: 0.6; transform: scale(0.99); }
      to   { opacity: 1; transform: scale(1); }
    }

    /* ---- Typing indicator ---- */
    .agent-typing-row {
      display: flex; gap: 12px;
      align-items: flex-start;
      margin-bottom: 22px;
      animation: agent-fade-in 0.2s ease;
    }
    .agent-typing-bubble {
      padding: 10px 14px;
      background: var(--bg-2);
      border: 1px solid var(--border);
      border-radius: 12px;
      font-size: 12px;
      color: var(--muted);
      display: flex; align-items: center; gap: 10px;
    }
    .agent-typing-dots {
      display: inline-flex; gap: 4px;
    }
    .agent-typing-dot {
      width: 6px; height: 6px;
      border-radius: 50%;
      background: var(--accent-bright);
      animation: agent-typing-pulse 1.4s ease-in-out infinite;
    }
    .agent-typing-dot:nth-child(2) { animation-delay: 0.2s; }
    .agent-typing-dot:nth-child(3) { animation-delay: 0.4s; }
    @keyframes agent-typing-pulse {
      0%, 60%, 100% { opacity: 0.25; transform: scale(0.85); }
      30% { opacity: 1; transform: scale(1); }
    }

    /* ---- Composer ---- */
    /* Button sits INSIDE the textarea (absolute) so it feels like one
       integrated input — the modern chat-app convention. Textarea has
       padding-right to keep typed text clear of the button. */
    .agent-composer {
      border-top: 1px solid var(--border);
      padding: 14px 16px 18px;
      background: var(--bg-2);
    }
    .agent-composer-wrap {
      position: relative;
    }
    .agent-composer textarea {
      width: 100%;
      min-height: 48px; max-height: 220px;
      padding: 13px 56px 13px 16px;
      background: var(--panel);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 14px;
      font-family: inherit;
      font-size: var(--agent-text);
      line-height: 1.55;
      resize: none;
      transition: border-color 0.18s, box-shadow 0.18s, background 0.18s;
      box-sizing: border-box;
    }
    .agent-composer textarea:focus {
      outline: none;
      border-color: var(--accent);
      box-shadow: 0 0 0 3px var(--accent-dim);
    }
    .agent-composer textarea::placeholder {
      color: var(--muted);
      opacity: 0.7;
    }
    .agent-composer .hint {
      position: absolute;
      bottom: -16px; right: 4px;
      font-size: 10px;
      color: var(--muted);
      letter-spacing: 0.2px;
      pointer-events: none;
      opacity: 0.55;
    }
    .agent-composer .send-btn {
      position: absolute;
      right: 7px; bottom: 7px;
      width: 34px; height: 34px;
      border-radius: 10px;
      background: var(--accent);
      color: white;
      border: none;
      cursor: pointer;
      display: inline-flex;
      align-items: center; justify-content: center;
      transition: background 0.18s, transform 0.18s, box-shadow 0.18s;
      box-shadow: 0 2px 8px rgba(47,129,247,0.3);
    }
    .agent-composer .send-btn:hover:not(:disabled) {
      background: var(--accent-bright);
      transform: translateY(-1px);
      box-shadow: 0 4px 14px rgba(47,129,247,0.45);
    }
    .agent-composer .send-btn:active:not(:disabled) {
      transform: translateY(0);
    }
    .agent-composer .send-btn:disabled {
      background: rgba(255,255,255,0.06);
      color: rgba(255,255,255,0.3);
      cursor: not-allowed;
      box-shadow: none;
    }
    .agent-composer .send-btn svg {
      width: 16px; height: 16px;
    }

    /* ---- Settings drawer ---- */
    .agent-settings-modal {
      position: fixed; inset: 0;
      background: rgba(0,2,12,0.7);
      backdrop-filter: blur(6px);
      z-index: 200;
      display: none;
      align-items: center; justify-content: center;
      animation: agent-fade-in 0.2s ease;
    }
    .agent-settings-modal.open { display: flex; }
    .agent-settings-card {
      width: min(580px, 96vw); max-height: 90vh;
      overflow-y: auto;
      padding: 24px;
      border-radius: 14px;
      background: var(--panel);
      border: 1px solid var(--border-strong);
      box-shadow: 0 24px 60px rgba(0,0,0,0.6);
    }
    .agent-settings-card h2 {
      margin: 0 0 6px;
      font-size: 17px; font-weight: 600;
      /* Override the panel-wide h2 rule (line ~5322) which sets uppercase
         + 0.1em letter-spacing. The settings-card titles are sentence-
         case, like a real settings sheet, not eyebrow labels. */
      text-transform: none;
      letter-spacing: -0.1px;
      color: var(--text);
      display: flex; align-items: center; justify-content: space-between;
      gap: 12px;
    }
    .agent-settings-card h2 button,
    .agent-settings-card h2 .close-btn {
      flex-shrink: 0;          /* don't let the close button grow huge */
    }
    /* Inline 'Close' button next to the title. The button has no class,
       just sits inside h2 — keep it small and ghosty. */
    .agent-settings-card h2 > button {
      background: transparent;
      color: var(--muted);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 5px 12px;
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 0.2px;
      text-transform: none;
      cursor: pointer;
      transition: border-color 0.15s, color 0.15s;
    }
    .agent-settings-card h2 > button:hover {
      color: var(--text);
      border-color: var(--accent);
    }

    /* The panel has a global `input, textarea, select, button { width: 100% }`
       rule (see ~line 5331). It makes EVERY button full-width by default,
       which destroys the inline button layout in the modal: the Close
       button stretches across the whole card, the Start button next to
       the engine status pill takes 350px, etc. Override for the agent
       modal — buttons here are inline elements managed by their flex
       containers. */
    .agent-settings-card h2 > button,
    .agent-engine-row > button,
    .agent-settings-card .actions > button {
      width: auto;
    }
    .agent-settings-card .subtitle {
      font-size: 12px; color: var(--muted);
      margin-bottom: 18px;
    }
    .agent-settings-card .close-btn {
      background: transparent; color: var(--muted);
      border: 1px solid var(--border); border-radius: 8px;
      padding: 5px 11px; font-size: 11px; cursor: pointer;
    }
    .agent-settings-card .close-btn:hover {
      color: var(--text); border-color: var(--accent);
    }
    .agent-settings-card .field { margin-bottom: 14px; }
    .agent-settings-card label {
      display: block; margin-bottom: 6px;
      font-size: 11px;
      color: var(--muted); font-weight: 600;
      letter-spacing: 0.4px; text-transform: uppercase;
    }
    .agent-settings-card input,
    .agent-settings-card select,
    .agent-settings-card textarea {
      width: 100%;
      padding: 9px 12px;
      background: var(--bg-2); color: var(--text);
      border: 1px solid var(--border); border-radius: 8px;
      font-family: inherit; font-size: 13px;
      box-sizing: border-box;
      transition: border-color 0.15s;
    }
    .agent-settings-card input:focus,
    .agent-settings-card select:focus,
    .agent-settings-card textarea:focus {
      outline: none; border-color: var(--accent);
      box-shadow: 0 0 0 3px var(--accent-dim);
    }
    .agent-settings-card .row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .agent-settings-card .actions {
      display: flex; gap: 8px; margin-top: 18px;
      justify-content: flex-end;
    }
    .agent-settings-card .actions button {
      padding: 9px 18px; border-radius: 8px;
      border: none; cursor: pointer;
      font-size: 13px; font-weight: 600;
      transition: background 0.15s;
    }
    .agent-settings-card .actions .save {
      background: var(--accent); color: white;
    }
    .agent-settings-card .actions .save:hover { background: var(--accent-bright); }
    .agent-settings-card .actions .ghost {
      background: transparent; color: var(--muted);
      border: 1px solid var(--border);
    }
    .agent-settings-card .actions .ghost:hover {
      color: var(--text); border-color: var(--accent);
    }
    .agent-settings-card .hint {
      font-size: 12px; color: var(--muted); line-height: 1.55;
      padding: 12px 14px; border-radius: 8px;
      background: var(--bg-2); border: 1px solid var(--border);
      margin-bottom: 16px;
    }
    .agent-settings-card .hint code {
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 11px;
      padding: 1px 5px;
      background: rgba(255,255,255,0.06);
      border-radius: 3px;
      color: var(--accent-bright);
    }
    .agent-engine-row {
      display: flex; align-items: center; gap: 12px;
      padding: 10px 14px; border-radius: 8px;
      background: var(--bg-2); border: 1px solid var(--border);
      margin-bottom: 12px;
      font-size: 12px;
    }
    .agent-engine-row .pill {
      font-size: 10px; padding: 3px 9px; border-radius: 999px;
      background: var(--panel-2); color: var(--muted);
      border: 1px solid var(--border);
      font-weight: 600; letter-spacing: 0.3px; text-transform: uppercase;
    }
    .agent-engine-row .pill.live {
      color: #9be7a4; border-color: rgba(46,160,67,0.5);
      background: rgba(46,160,67,0.08);
    }
    .agent-engine-row .pill.bad {
      color: #f49a9e; border-color: rgba(207,34,46,0.5);
      background: rgba(207,34,46,0.08);
    }
    .agent-engine-row .row-detail {
      flex: 1; color: var(--muted); min-width: 0;
      overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }
    .agent-engine-row button {
      padding: 6px 12px;
      border-radius: 7px;
      border: 1px solid var(--border);
      background: var(--panel);
      color: var(--text);
      font-size: 11px; font-weight: 600;
      cursor: pointer;
      transition: border-color 0.15s;
    }
    .agent-engine-row button:hover {
      border-color: var(--accent);
      color: var(--accent-bright);
    }
  </style>
</head>
<body>

<header>
  <a href="/" class="brand"><img src="/assets/logo-header.png" alt="Phosphene"></a>
  <span class="version-badge" title="Phosphene 2.0">2.0</span>
  __PROFILE_BADGE__
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
  <!-- Update-available pill. Hidden by default; renders only when the
       /version poller finds the install behind origin/main. Click → modal
       listing the unseen commits. We do this because users keep telling
       us "I clicked Update but I don't see anything" — by the time they
       do, we've usually pushed three more fixes. -->
  <!-- Version pill — the "magic button". Always visible so users learn
       where to look. Clicking does the right thing for the state:
         pill-current   Y1.001          click → re-check GitHub now
         pill-update    ↑ Y1.002        click → git pull (then restart)
         pill-restart   ↻ restart       click → just a reminder hint
         pill-dev       Y1.001 ⚙        click → toast with reason
         pill-checking  Y1.001          (idle, while first poll runs)
       No modal — confirm() + small toast for everything. -->
  <span id="versionPill" class="pill pill-checking" style="cursor:pointer"
        onclick="versionPillClick()" title="Phosphene version">phosphene · …</span>
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

    <!-- Workflow tabs: switch the form-pane between the manual form and
         the chat-driven planner. Default = manual; toggling persists in
         localStorage so the user lands on the same tab next session. -->
    <nav class="workflow-tabs" id="workflowTabs">
      <button data-workflow="manual" class="active">Manual</button>
      <button data-workflow="agent">Agentic Flows<span class="new-badge">NEW</span></button>
    </nav>

    <!-- ============== AGENTIC FLOWS PANE ============== -->
    <!-- Chat-driven shot planner. The agent loop lives in the panel's
         Python backend (agent/runtime.py); this UI just sends user
         messages, renders the message log with markdown + expandable
         tool cards, and exposes engine settings via a drawer. -->
    <section class="agent-pane" id="agentPane" hidden>
      <header class="agent-header" style="position:relative">
        <button type="button" class="engine-pill" id="agentEnginePill"
                onclick="openAgentSettings()" title="Click to configure the agent engine">
          <span class="dot" id="agentEngineDot"></span>
          <span id="agentEngineLabel">engine…</span>
        </button>
        <div class="session-title" id="agentSessionTitle"
             onclick="agentToggleSessionsPop()"
             style="cursor:pointer"
             title="Click to switch sessions">
          New chat
        </div>
        <button type="button" class="icon-btn" onclick="agentNewSession()"
                title="Start a fresh session">
          <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
            <line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>
          </svg>
        </button>
        <a class="icon-btn" id="agentPopOutBtn"
           href="/" target="_blank" rel="noopener"
           onclick="agentPopOutFlagsBeforeNavigate()"
           title="Open in your default browser (true OS fullscreen — escapes Pinokio's sidebar)">
          <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>
            <polyline points="15 3 21 3 21 9"/>
            <line x1="10" y1="14" x2="21" y2="3"/>
          </svg>
        </a>
        <button type="button" class="icon-btn" id="agentFullscreenBtn"
                onclick="agentToggleFullscreen()" title="Expand to fullscreen (Esc to exit)">
          <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" id="agentFullscreenIconExpand">
            <polyline points="15 3 21 3 21 9"/>
            <polyline points="9 21 3 21 3 15"/>
            <line x1="21" y1="3" x2="14" y2="10"/>
            <line x1="3" y1="21" x2="10" y2="14"/>
          </svg>
          <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" id="agentFullscreenIconCollapse" style="display:none">
            <polyline points="4 14 10 14 10 20"/>
            <polyline points="20 10 14 10 14 4"/>
            <line x1="14" y1="10" x2="21" y2="3"/>
            <line x1="3" y1="21" x2="10" y2="14"/>
          </svg>
        </button>
        <button type="button" class="icon-btn" onclick="openAgentSettings()" title="Settings">
          <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="3"></circle>
            <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
          </svg>
        </button>
        <div class="agent-sessions-pop" id="agentSessionsPop"></div>
      </header>

      <div class="agent-chat" id="agentChat"></div>

      <div class="agent-composer">
        <!-- Reference chip: appears when the user clicks Refine on an
             existing clip. Carries the clip's job_id; on Send, we
             prepend "Refine <job_id>: " to the user's message so the
             agent calls inspect_clip and treats the request as a
             variation. Clear with the × button. -->
        <div class="agent-ref-chip" id="agentRefChip" style="margin-left:auto;margin-right:auto;max-width:720px">
          <span class="ref-icon">↻</span>
          <span style="color:var(--muted);font-size:11px">refine</span>
          <span class="ref-label" id="agentRefChipLabel">…</span>
          <button type="button" class="clear" onclick="agentClearRefine()" title="Cancel refine">×</button>
        </div>
        <div class="agent-composer-wrap">
          <textarea id="agentInput"
                    placeholder="Paste a script, or describe a piece. The agent will plan, estimate the wall time, and queue overnight."
                    rows="1"
                    autocomplete="off"></textarea>
          <button type="button" id="agentSendBtn" class="send-btn" onclick="agentSend()" title="Send (Cmd/Ctrl+Enter)" disabled>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
              <line x1="12" y1="19" x2="12" y2="5"/>
              <polyline points="5 12 12 5 19 12"/>
            </svg>
          </button>
          <span class="hint">Cmd/Ctrl + Enter to send</span>
        </div>
      </div>
    </section>

    <!-- The original manual form is unchanged; it sits below in the DOM
         and is shown/hidden by JS as the workflow tab toggles. -->
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

      <!-- Quality picker (Y1.013): one decision instead of four. Each
           button bundles dimensions + model + step count + tier
           recommendation, with the actual specs visible on the pill so
           power users can read what they're getting. Aspect, custom
           W/H, and the experimental Speed setting moved into the
           Customize disclosure below. Beginners pick a button; power
           users open Customize. -->
      <h2>Quality</h2>
      <div class="pill-group cols-2 quality-row" id="qualityGroup">
        <button type="button" class="pill-btn pill-quality" data-quality="quick">
          <span class="ql-name">Quick</span>
          <span class="sub ql-spec">640×480 · ~2 min</span>
          <span class="ql-tier">Q4 · any Mac</span>
        </button>
        <button type="button" class="pill-btn pill-quality active" data-quality="balanced">
          <span class="ql-name">Balanced</span>
          <span class="sub ql-spec">1024×576 → 720p</span>
          <span class="ql-tier">Q4 · ~5 min · no crop</span>
        </button>
        <button type="button" class="pill-btn pill-quality" data-quality="standard">
          <span class="ql-name">Standard</span>
          <span class="sub ql-spec">1280×704 · ~7 min</span>
          <span class="ql-tier">Q4 · standard tier+</span>
        </button>
        <button type="button" class="pill-btn pill-quality disabled" data-quality="high" id="qualityHigh">
          <span class="ql-name">High</span>
          <span class="sub ql-spec" id="highSpec">1280×704 · ~12 min</span>
          <span class="ql-tier" id="highSub">Q8 not installed</span>
        </button>
      </div>
      <input type="hidden" name="quality" id="quality" value="balanced">
      <input type="hidden" name="accel" id="accel" value="off">
      <input type="hidden" name="temporal_mode" id="temporal_mode" value="native">
      <input type="hidden" name="upscale" id="upscale" value="fit_720p">

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
      <label class="lbl" for="negative_prompt">Avoid</label>
      <textarea class="avoid-textarea" name="negative_prompt" id="negative_prompt" placeholder="Optional: blurry hands, distorted fingers, extra fingers, smeared face, warped text"></textarea>
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
      <!-- LoRAs section — visually distinct from the surrounding form so
           users notice "oh, there's a whole controllable area here." A
           thin separator above + bordered container + a clear chevron
           on the summary. Default open so first-time users see what's
           inside without hunting for the disclosure triangle. -->
      <div class="form-divider"></div>
      <details id="lorasDetails" open class="loras-section">
        <summary class="loras-summary">
          <span class="loras-chevron" aria-hidden="true">▾</span>
          <span class="loras-title">LoRAs</span>
          <span class="loras-meta" id="lorasSummaryCount">none active</span>
          <!-- Action buttons live in the header so they're visible
               regardless of how far the user has scrolled inside the
               LoRA list. Rescan is icon-only (the ↻ glyph reads as
               refresh universally); Browse CivitAI is the primary CTA
               for adding new LoRAs so it's a coloured button.
               event.stopPropagation() keeps clicks from toggling the
               <details> open/closed state. -->
          <span class="loras-header-actions">
            <button type="button" class="loras-icon-btn"
                    title="Rescan mlx_models/loras/ for new files"
                    onclick="event.stopPropagation(); event.preventDefault(); refreshLoras()">↻</button>
            <button type="button" class="loras-browse-btn"
                    onclick="event.stopPropagation(); event.preventDefault(); openCivitaiModal()">🔍 Browse CivitAI</button>
          </span>
        </summary>
        <div class="loras-body" id="lorasBody">
          <!-- Filter/search box. Shows up only when 5+ LoRAs are
               installed; below that it's just visual noise. Filters by
               name AND trigger words (case-insensitive substring). -->
          <div id="lorasFilterRow" style="display:none;">
            <input type="text" id="lorasFilter" class="lora-filter"
                   placeholder="Filter LoRAs… (name or trigger word)"
                   oninput="renderLorasList()">
          </div>
          <div class="hint" id="lorasEmpty">
            Drop <code>.safetensors</code> files into <code id="lorasDir">mlx_models/loras/</code>
            to use them, or click <strong>Browse CivitAI</strong> above.
            Each LoRA picks up an optional sidecar <code>.json</code> with
            name + trigger words + recommended strength.
          </div>
          <div class="loras-list" id="lorasList"></div>
        </div>
      </details>
      <div class="form-divider"></div>
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

      <!-- Sizing for non-extend modes. The headline Quality picker above
           sets sensible defaults; this disclosure lets power users override
           aspect, exact dimensions, and the experimental sampler speed. -->
      <div class="mode-only" id="sizingSection">
        <details id="customizeDetails" class="customize-section">
          <summary class="cz-summary">
            <span class="cz-chevron" aria-hidden="true">▾</span>
            <span class="cz-title">Customize</span>
            <span class="cz-meta" id="customizeSummary">16:9 · default speed</span>
          </summary>
          <div class="cz-body">
            <!-- Aspect — only relevant when the active Quality preset has
                 multiple aspects (Standard / High at 1280×704 vs 704×1280).
                 Hidden when Quick is active (640×480 is 4:3 only). -->
            <div id="aspectRow" class="cz-control">
              <div class="cz-label">Aspect ratio</div>
              <div class="pill-group cols-2" id="aspectGroup">
                <button type="button" class="pill-btn active" data-aspect="landscape"><span>16 : 9</span><span class="sub">horizontal</span></button>
                <button type="button" class="pill-btn" data-aspect="vertical"><span>9 : 16</span><span class="sub">vertical</span></button>
              </div>
              <input type="hidden" id="aspect" value="landscape">
            </div>

            <!-- Width × height. Setting custom dimensions makes the form
                 leave the active preset (the Quality pill stays highlighted
                 but the Customize summary shows "custom" so the user knows
                 they've deviated). -->
            <div id="dimsRow" class="cz-control">
              <div class="cz-label">Width × height</div>
              <div class="row">
                <div><input name="width" id="width" value="1024" type="number" min="32" step="32" aria-label="Width"></div>
                <div><input name="height" id="height" value="576" type="number" min="32" step="32" aria-label="Height"></div>
              </div>
            </div>

            <!-- Speed — experimental sampler acceleration. Boost/Turbo
                 skip 2-3 stable middle denoise calls. Off at High quality
                 (the two-stage HQ sampler doesn't support skipping). -->
            <div class="cz-control">
              <div class="cz-label">Speed
                <span class="cz-label-hint">experimental — skip stable denoise steps</span>
              </div>
              <div class="pill-group cols-3" id="accelGroup">
                <button type="button" class="pill-btn active" data-accel="off"><span>Exact</span><span class="sub">full sampler</span></button>
                <button type="button" class="pill-btn" data-accel="boost"><span>Boost</span><span class="sub">~10-15% faster</span></button>
                <button type="button" class="pill-btn" data-accel="turbo"><span>Turbo</span><span class="sub">~20-25% faster</span></button>
              </div>
            </div>

            <!-- Long Clip Boost — asks LTX 2.3 to generate fewer semantic
                 frames at 12fps, then exports a normal 24fps file with
                 motion interpolation. This is explicit because dialogue,
                 lip-sync, and fast motion need user judgment. -->
            <div class="cz-control" id="temporalRow">
              <div class="cz-label">Long clips
                <span class="cz-label-hint">experimental temporal interpolation</span>
              </div>
              <div class="pill-group cols-2" id="temporalGroup">
                <button type="button" class="pill-btn active" data-temporal="native"><span>Native</span><span class="sub">24fps from LTX</span></button>
                <button type="button" class="pill-btn" data-temporal="fps12_interp24"><span>12 → 24fps</span><span class="sub">half frames · interpolated</span></button>
              </div>
            </div>

            <!-- Export upscale — target dims after render. "720p fit"
                 preserves aspect ratio (no crop, no distortion). 2× doubles
                 each side. Native skips the upscale entirely. The "Method"
                 row below picks how the upscaling is done. -->
            <div class="cz-control">
              <div class="cz-label">Export
                <span class="cz-label-hint">post-render, no crop</span>
              </div>
              <div class="pill-group cols-3" id="upscaleGroup">
                <button type="button" class="pill-btn active" data-upscale="off"><span>Native</span><span class="sub">as generated</span></button>
                <button type="button" class="pill-btn" data-upscale="fit_720p"><span>720p fit</span><span class="sub">scale + pad</span></button>
                <button type="button" class="pill-btn" data-upscale="x2"><span>2×</span><span class="sub">same ratio</span></button>
              </div>
            </div>

            <!-- Upscale method. Hidden when Export = Native. Fast = ffmpeg
                 Lanczos resize, no detail recovery, near-instant. Sharp =
                 PiperSR/CoreML 2× post-upscale on Apple Neural Engine, then
                 ffmpeg fit/export. The old LTX latent upscaler path remains
                 hidden behind LTX_ENABLE_MODEL_UPSCALE for lab archaeology. -->
            <div class="cz-control" id="upscaleMethodRow" style="display:none;">
              <div class="cz-label">Method
                <span class="cz-label-hint">how the upscale is done</span>
              </div>
              <div class="pill-group cols-2" id="upscaleMethodGroup">
                <button type="button" class="pill-btn active" data-method="lanczos"><span>Fast</span><span class="sub">ffmpeg Lanczos · instant</span></button>
                <button type="button" class="pill-btn" data-method="pipersr"><span>Sharp</span><span class="sub">PiperSR ANE · +30-90 s</span></button>
              </div>
              <input type="hidden" name="upscale_method" id="upscale_method" value="lanczos">
            </div>
          </div>
        </details>

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
           in the poll handler). Default-off for public installs: killing a
           neighboring Pinokio app should be an explicit choice, even when it
           can free a lot of memory on 64 GB machines. -->
      <div id="comfyKillRow" class="comfy-row" style="display:none">
        <label class="lbl" style="display:flex; align-items:center; gap:8px; cursor:pointer">
          <input type="checkbox" name="stop_comfy" id="stop_comfy" value="on">
          <span>Stop ComfyUI before render <span style="color:var(--muted)">(optional · can free ~27 GB)</span></span>
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

  <!-- ============== AGENT STAGE PANE ============== -->
  <!-- Right-side live canvas for the agent: code-interpreter feel.
       Hidden by default; shown via CSS when body[data-workflow="agent"]
       OR body.agent-fullscreen. Polls /status every ~1.5 s and re-renders
       the current job, queue depth, and recent session outputs. Click an
       output to play it inline (lightbox). -->
  <aside class="agent-stage-pane">
    <header class="agent-stage-head">
      <span class="live-dot" id="agentStageDot"></span>
      <span class="label">Stage</span>
      <span class="session-pill" id="agentStageSession">no session</span>
      <span class="spacer"></span>
    </header>
    <div class="agent-stage-body" id="agentStageBody">
      <!-- Now Rendering section -->
      <div class="agent-stage-section">
        <h4>Now rendering</h4>
        <div class="stage-now-card idle" id="agentStageNow">
          <div>Idle — submit a shot to see it render here.</div>
        </div>
      </div>
      <!-- Activity feed -->
      <div class="agent-stage-section">
        <h4>Recent activity <span class="count" id="agentStageActivityCount">0</span></h4>
        <div class="stage-activity" id="agentStageActivity">
          <div class="stage-empty">No tool calls yet.</div>
        </div>
      </div>
      <!-- Session outputs -->
      <div class="agent-stage-section">
        <h4>Session outputs <span class="count" id="agentStageOutputsCount">0</span></h4>
        <div class="stage-outputs" id="agentStageOutputs">
          <div class="stage-empty">No mp4s rendered yet.</div>
        </div>
      </div>
    </div>
  </aside>

  <!-- Lightbox for stage outputs — plays a finished mp4 full-size on click.
       Refine button references the playing clip in the composer so the
       user can ask the agent for a variation while watching. -->
  <div class="stage-lightbox" id="agentStageLightbox" onclick="if(event.target===this)agentStageLightboxClose()">
    <button class="close-btn" onclick="agentStageLightboxClose()">Close · Esc</button>
    <button class="refine-btn-large" id="agentStageLightboxRefine" onclick="agentStageLightboxRefine()">↻ Refine this clip</button>
    <video id="agentStageLightboxVideo" controls preload="metadata"></video>
  </div>
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
    <!-- Inline API-key banner. CivitAI requires a token to download LoRAs
         (even SFW ones). Three states, all rendered in the same slot by
         renderCivitaiAuthBanner():
           - missing: amber, has an input + Save button
           - set: green checkmark with a "change" link that re-shows the input
           - error: red, shown after a failed test, with re-input + retry
         The actual key is POSTed to /settings (same endpoint the cog
         menu uses) so there's a single source of truth on disk. -->
    <div id="civitaiAuthBanner" class="civitai-auth" style="display:none"></div>
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
          <button type="button" class="primary-btn token-savetest"
                  onclick="testToken('civitai')" title="Save the pasted key and verify it with CivitAI">save & test</button>
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
          <button type="button" class="primary-btn token-savetest"
                  onclick="testToken('hf')" title="Save the pasted token and verify it with Hugging Face">save & test</button>
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

    <!-- Spicy mode — gates NSFW LoRA visibility in the CivitAI browser.
         Default OFF. Turning ON requires an explicit confirm-click pattern
         so a casual visitor / kid can't toggle it by a stray click. The
         server enforces the gate independently in _civitai_search, so a
         tampered client can't bypass this. -->
    <div class="settings-section">
      <h3>Spicy mode <span class="hint" style="font-weight:400">(adult content)</span></h3>
      <div class="hint" style="margin-bottom:10px">
        When OFF, the CivitAI LoRA browser hides NSFW models entirely —
        the "Show NSFW" toggle disappears and any NSFW result the API
        returns is filtered out. When ON, you can choose per-search whether
        to include NSFW results. The server enforces this gate even if the
        client is tampered with.
      </div>
      <div class="spicy-row">
        <span class="spicy-state" id="spicyStateBadge">OFF</span>
        <button type="button" class="ghost-btn" id="spicyToggleBtn"
                onclick="toggleSpicyMode()" style="margin-left:auto">
          Enable Spicy mode
        </button>
      </div>
      <div class="hint" id="spicyHint" style="margin-top:8px; display:none"></div>
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

<!-- (Version modal removed in the magic-button rewrite. Pill itself is the
     full UI now: click while behind → confirm + git pull; click while
     current → live re-check. See versionPillClick() in the JS section.) -->

<!-- ============== OUTPUT INFO MODAL ============== -->
<!-- Opened by the ⓘ button on each gallery card. Shows everything we
     wrote into the .mp4.json sidecar at render time: prompt, seed,
     mode/quality, frames + dimensions, LoRAs used (with name + strength),
     elapsed time, queue id, model. Plus copy-buttons for prompt + seed
     so users can easily re-use them. -->
<div id="outputInfoModal" class="models-modal" style="display:none"
     onclick="if(event.target===this) closeOutputInfoModal()">
  <div class="models-card" style="width: min(720px, 96vw)">
    <div class="models-head">
      <h2 id="outputInfoTitle">Generation info</h2>
      <button class="ghost-btn" onclick="closeOutputInfoModal()">Close</button>
    </div>
    <div id="outputInfoBody" class="output-info-body">
      <div class="hint">Loading…</div>
    </div>
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

<!-- ============== HF MODEL BROWSER MODAL ============== -->
<!-- Search + install MLX-format chat models from Hugging Face. Hits
     huggingface.co/api/models with library=mlx + pipeline_tag=text-generation
     filters. Click Install → backend spawns `hf download`, status streams
     in the bottom strip until done. Newly-installed models appear in the
     local-engine model picker on next refresh. -->
<div class="model-browser-modal" id="modelBrowserModal" onclick="if(event.target===this)closeModelBrowser()">
  <div class="model-browser-card">
    <div class="model-browser-head">
      <h2>Browse + install models</h2>
      <span class="subtitle">Hugging Face · MLX</span>
      <button type="button" class="close-btn" onclick="closeModelBrowser()">Close</button>
    </div>
    <div class="model-browser-controls">
      <input type="text" id="modelBrowserQuery" placeholder="Search HF (e.g. qwen coder, devstral, llama 3.3, granite)" autocomplete="off" />
      <label class="checkbox-label" title="Filter to abliterated / uncensored variants — the huihui-ai conventions on HF.">
        <input type="checkbox" id="modelBrowserAbliterated">
        Abliterated only
      </label>
      <button type="button" id="modelBrowserSearchBtn" onclick="modelBrowserSearch()">Search</button>
    </div>
    <div class="model-browser-results" id="modelBrowserResults">
      <div class="model-browser-empty">Type a query and hit Search. Try "qwen3 coder", "devstral", "abliterated", "32b".</div>
    </div>
    <div class="model-browser-status" id="modelBrowserStatus">
      <div><span class="label" id="modelBrowserStatusLabel">…</span><span id="modelBrowserStatusSummary"></span></div>
      <div class="last-line" id="modelBrowserStatusLine"></div>
    </div>
  </div>
</div>

<!-- ============== AGENT SETTINGS MODAL ============== -->
<div class="agent-settings-modal" id="agentSettingsModal" onclick="if(event.target===this)closeAgentSettings()">
  <div class="agent-settings-card">
    <h2>Agent engine
      <button type="button" onclick="closeAgentSettings()">Close</button>
    </h2>

    <div class="hint">
      The agent uses any OpenAI-compatible chat endpoint. Default is
      <strong>Phosphene Local</strong> — a small mlx-lm.server pointed at the
      bundled Gemma 3 12B IT (the same weights LTX uses as its text encoder,
      doubling as a chat model). Stronger options: drop a Qwen 3 Coder 30B
      MLX model into <code>mlx_models/</code> and pick it below; or point at
      a cloud endpoint (Anthropic compat, OpenAI, OpenRouter, your LM Studio
      box on the LAN).
    </div>

    <div class="agent-engine-row" id="agentLocalRow" style="display:none">
      <span class="pill" id="agentLocalPill">stopped</span>
      <span style="flex:1; font-size:12px; color:var(--muted)" id="agentLocalDetail">mlx-lm.server</span>
      <button type="button" onclick="agentLocalToggle()" id="agentLocalToggleBtn"
              style="padding:5px 10px;border-radius:6px;border:1px solid var(--border);background:transparent;color:var(--text);font-size:11px;cursor:pointer">Start</button>
    </div>

    <div class="field">
      <label>Engine kind</label>
      <select id="agentKind" onchange="agentKindChanged()">
        <option value="phosphene_local">Phosphene Local — bundled mlx-lm.server</option>
        <option value="ollama">Ollama bridge — talks to your `ollama serve` on port 11434</option>
        <option value="custom">Custom — any OpenAI-compatible URL (Anthropic compat, OpenAI, OpenRouter, LM Studio…)</option>
      </select>
    </div>

    <div class="field" id="agentLocalModelField">
      <label>
        Local model (chat-capable, MLX 4-bit recommended)
        <button type="button" onclick="openModelBrowser()" style="margin-left:8px;font-size:10px;padding:3px 9px;background:transparent;color:var(--accent-bright);border:1px solid var(--accent);border-radius:6px;cursor:pointer;width:auto">Browse + install…</button>
      </label>
      <select id="agentLocalModel"></select>
    </div>

    <!-- Ollama: shown only when engine kind == 'ollama'. Status is fetched
         live from /agent/ollama/status; if Ollama isn't running we surface
         a clear hint instead of a useless empty dropdown. -->
    <div class="field" id="agentOllamaField" style="display:none">
      <label>
        Ollama installed model
        <button type="button" onclick="agentOllamaRefresh()" style="margin-left:8px;font-size:10px;padding:3px 9px;background:transparent;color:var(--muted);border:1px solid var(--border);border-radius:6px;cursor:pointer;width:auto">Refresh</button>
      </label>
      <select id="agentOllamaModel"></select>
      <div id="agentOllamaHint" style="margin-top:6px;font-size:11px;color:var(--muted)">
        Manage Ollama models from the terminal: <code>ollama pull qwen2.5-coder:32b</code>, <code>ollama pull huihui_ai/llama3.3-abliterated:70b</code>, etc. Phosphene only talks to Ollama's <code>/v1/chat/completions</code> endpoint — the model has to already be installed in Ollama.
      </div>
    </div>

    <div class="field" id="agentBaseUrlField" style="display:none">
      <label>Base URL</label>
      <input type="text" id="agentBaseUrl" placeholder="https://api.openai.com/v1">
    </div>

    <div class="field" id="agentApiKeyField" style="display:none">
      <label>API key (saved locally; only sent as Authorization header)</label>
      <input type="password" id="agentApiKey" placeholder="(leave blank to keep saved key)">
    </div>

    <div class="field" id="agentRemoteModelField" style="display:none">
      <label>Model identifier</label>
      <input type="text" id="agentRemoteModel" placeholder="e.g. claude-sonnet-4-6 or gpt-5">
    </div>

    <div class="row">
      <div class="field">
        <label>Temperature</label>
        <input type="number" id="agentTemp" min="0" max="2" step="0.05" value="0.4">
      </div>
      <div class="field">
        <label>Max tokens</label>
        <input type="number" id="agentMaxTokens" min="256" max="32768" step="256" value="3072">
      </div>
    </div>

    <hr style="border:none;border-top:1px solid var(--border);margin:18px 0 12px">

    <h2 style="font-size:14px">Image generation
      <span class="pill" id="agentImagePill" style="font-size:10px;padding:3px 8px;border-radius:999px;background:var(--bg-2);color:var(--muted);border:1px solid var(--border)">…</span>
    </h2>
    <div class="subtitle" style="margin-bottom:12px">
      For the director-collaboration workflow: agent generates 4 candidate stills per
      shot, you pick the best one, then i2v anchors lock the look before each render.
    </div>

    <div class="field">
      <label>Backend</label>
      <select id="agentImageKind" onchange="agentImageKindChanged()">
        <option value="mock">Mock — PIL placeholders, free, instant (for testing the flow)</option>
        <option value="mflux">Phosphene Local Flux (mflux + Flux Krea Dev) — fully on-Mac, free, ~25–60 s/img</option>
        <option value="bfl">Black Forest Labs API — cloud Flux, ~$0.025/img on flux-dev, ~10–15 s/img</option>
      </select>
    </div>

    <!-- mflux fields -->
    <div class="field" id="agentMfluxModelField" style="display:none">
      <label>mflux model</label>
      <select id="agentMfluxModel" onchange="agentMfluxModelChanged()">
        <option value="krea-dev">Flux Krea Dev — recommended (best photorealism)</option>
        <option value="dev">Flux.1 Dev — vanilla 28-step</option>
        <option value="schnell">Flux.1 Schnell — 4-step, fastest</option>
        <option value="__custom__">Custom HF id / local path…</option>
      </select>
    </div>
    <div class="field" id="agentMfluxCustomField" style="display:none">
      <label>Custom model (HF repo id or local path)</label>
      <input type="text" id="agentMfluxCustomPath" placeholder="filipstrand/FLUX.1-Krea-dev-mflux-4bit">
      <div style="margin-top:4px;font-size:11px;color:var(--muted)">
        Required when using a non-named model. mflux will download to <code>~/.cache/huggingface</code> on first use (~6 GB for Krea Dev 4-bit).
      </div>
    </div>
    <div class="field" id="agentMfluxBaseField" style="display:none">
      <label>Base model (when using custom path)</label>
      <select id="agentMfluxBaseModel">
        <option value="krea-dev">krea-dev</option>
        <option value="dev">dev</option>
        <option value="schnell">schnell</option>
      </select>
    </div>
    <div class="row" id="agentMfluxParamsField" style="display:none">
      <div class="field">
        <label>Steps</label>
        <input type="number" id="agentMfluxSteps" min="1" max="50" value="25">
      </div>
      <div class="field">
        <label>Quantize (bits)</label>
        <select id="agentMfluxQuantize">
          <option value="4">4-bit (~6 GB, recommended)</option>
          <option value="8">8-bit (~12 GB, sharper)</option>
        </select>
      </div>
    </div>
    <div class="hint" id="agentMfluxInstallHint" style="display:none">
      <strong>First-time setup:</strong> install mflux into the panel's venv with
      <code>ltx-2-mlx/env/bin/pip install mflux</code> (one-time, ~50 MB of code; weights download separately on first generate).
      Verify the install with <code>ltx-2-mlx/env/bin/mflux-generate --help</code>.
    </div>

    <!-- BFL fields -->
    <div class="field" id="agentBflModelField" style="display:none">
      <label>BFL model</label>
      <select id="agentBflModel">
        <option value="flux-dev">flux-dev — 25 steps, ~$0.025 per image (recommended)</option>
        <option value="flux-pro">flux-pro — premium quality, ~$0.05 per image</option>
        <option value="flux-pro-1.1">flux-pro-1.1 — newer pro, ~$0.04 per image</option>
        <option value="flux-schnell">flux-schnell — 4 steps, ~$0.003 per image (fastest)</option>
      </select>
    </div>

    <div class="field" id="agentBflKeyField" style="display:none">
      <label>BFL API key</label>
      <input type="password" id="agentBflKey" placeholder="(leave blank to keep saved)">
      <div style="margin-top:6px;font-size:11px;color:var(--muted)">
        Get one at <code>https://api.bfl.ml</code> · stored in <code>state/agent_image_config.json</code>, only sent as the <code>X-Key</code> header.
      </div>
    </div>

    <div class="actions">
      <button type="button" class="ghost" onclick="closeAgentSettings()">Cancel</button>
      <button type="button" class="save" onclick="agentSaveSettings()">Save settings</button>
    </div>
  </div>
</div>

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
const MODEL_UPSCALE_ENABLED = !!BOOT.model_upscale_enabled;
const PIPERSR_UPSCALE_ENABLED = !!BOOT.pipersr_upscale_enabled;

// Apply tier-aware time estimates to the Quality pill subtitles. The HTML
// ships with the Comfortable-tier (M4 Studio 64 GB) numbers as defaults;
// users on Compact / Roomy / Studio tiers see realistic estimates instead
// of the optimistic baseline. Runs once on boot, plus when the tier modal
// reports new info (rare — tier is fixed for a given Mac).
function applyTierTimes() {
  const qt = (BOOT.quality_times || {});
  document.querySelectorAll('#qualityGroup .pill-quality').forEach(btn => {
    const key = btn.dataset.quality;
    const time = qt[key];
    const spec = btn.querySelector('.ql-spec');
    if (!spec) return;
    const dimsMatch = spec.textContent.match(/^([0-9]+×[0-9]+(\s+→\s+[0-9p]+)?)/);
    const dims = dimsMatch ? dimsMatch[1] : '';
    if (time && dims) {
      spec.textContent = `${dims} · ${time}`;
    } else if (time) {
      spec.textContent = time;
    }
  });
}

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
  updateAccelAvailability();
  updateTemporalAvailability();
  updateDerived();
  // Refresh the inline models card immediately — switching to FFLF when
  // Q8 is missing should surface the Download Q8 CTA without waiting for
  // the next 1.5s poll tick.
  if (LAST_STATUS) updateModelsCard(LAST_STATUS);
}
// Quality presets (Y1.013) — each one bundles the backend quality value
// (which selects the model + sampler) with the canonical dimensions.
// Backend still routes only on `quality == 'high'` vs anything else, so
  // 'quick', 'balanced', and 'standard' all run Q4 distilled — they differ in
// pixel count. The richer label is preserved in the sidecar so the
// info modal can show "Quick" / "Standard" / "High" verbatim.
const QUALITY_PRESETS = {
  quick:    { w: 640,  h: 480, upscale: 'off' },        // 4:3, fastest sanity check
  balanced: { w: 1024, h: 576, upscale: 'fit_720p' },   // exact 16:9 → 1280×720
  standard: { w: 1280, h: 704, upscale: 'off' },        // LTX-wide canonical render
  high:     { w: 1280, h: 704, upscale: 'off' },        // same dims, different model (Q8)
};

function setQuality(q) {
  // Tolerate legacy values from old sidecars: 'draft' → 'standard'.
  if (q === 'draft' || !QUALITY_PRESETS[q]) q = 'standard';
  document.getElementById('quality').value = q;
  document.querySelectorAll('#qualityGroup .pill-btn').forEach(b => b.classList.toggle('active', b.dataset.quality === q));
  // Set canonical dimensions for the preset, respecting the current
  // aspect choice. Quick is 4:3 only — landscape orientation only.
  const preset = QUALITY_PRESETS[q];
  const aspect = document.getElementById('aspect').value || 'landscape';
  const vertical = (aspect === 'vertical' && q !== 'quick');
  document.getElementById('width').value  = vertical ? preset.h : preset.w;
  document.getElementById('height').value = vertical ? preset.w : preset.h;
  setUpscale(preset.upscale || 'off');
  // Hide the Aspect row when Quick is active (only 4:3 supported); show
  // it for Standard/High where 16:9 vs 9:16 is a real choice.
  const aspectRow = document.getElementById('aspectRow');
  if (aspectRow) aspectRow.style.display = (q === 'quick') ? 'none' : '';
  applyQuality();
  updateAccelAvailability();
  updateTemporalAvailability();
  updateCustomizeSummary();
  if (LAST_STATUS) updateModelsCard(LAST_STATUS);
}
function setAccel(a) {
  const allowed = document.getElementById('quality').value !== 'high' && currentMode !== 'extend' && currentMode !== 'keyframe';
  const v = allowed ? a : 'off';
  document.getElementById('accel').value = v;
  document.querySelectorAll('#accelGroup .pill-btn').forEach(b => b.classList.toggle('active', b.dataset.accel === v));
  updateCustomizeSummary();
  updateDerived();
}
function temporalModeAllowed() {
  const q = document.getElementById('quality').value;
  const mode = document.getElementById('mode').value;
  return q !== 'high' && currentMode !== 'extend' && currentMode !== 'keyframe' && (mode === 't2v' || mode === 'i2v');
}
function setTemporalMode(t) {
  const allowed = temporalModeAllowed();
  const v = (allowed && t === 'fps12_interp24') ? 'fps12_interp24' : 'native';
  document.getElementById('temporal_mode').value = v;
  document.querySelectorAll('#temporalGroup .pill-btn').forEach(b => b.classList.toggle('active', b.dataset.temporal === v));
  updateCustomizeSummary();
  updateDerived();
}
function setUpscale(u) {
  const v = ['off', 'fit_720p', 'x2'].includes(u) ? u : 'off';
  document.getElementById('upscale').value = v;
  document.querySelectorAll('#upscaleGroup .pill-btn').forEach(b => b.classList.toggle('active', b.dataset.upscale === v));
  // Show / hide the Method pill row — only relevant when an upscale is
  // actually being applied. When toggled to "off", revert method to Fast
  // so a later toggle back to fit_720p starts from the safe default.
  const methodRow = document.getElementById('upscaleMethodRow');
  if (methodRow) methodRow.style.display = (v === 'off' || !PIPERSR_UPSCALE_ENABLED) ? 'none' : '';
  if (v === 'off' || !PIPERSR_UPSCALE_ENABLED) setUpscaleMethod('lanczos');
  updateCustomizeSummary();
  updateDerived();
}
function setUpscaleMethod(m) {
  if (m === 'model') m = 'pipersr'; // legacy sidecars from the retired LTX Sharp path
  const v = (PIPERSR_UPSCALE_ENABLED && m === 'pipersr') ? 'pipersr' : 'lanczos';
  document.getElementById('upscale_method').value = v;
  document.querySelectorAll('#upscaleMethodGroup .pill-btn').forEach(b => b.classList.toggle('active', b.dataset.method === v));
  updateCustomizeSummary();
  updateDerived();
}
function updateAccelAvailability() {
  const allowed = document.getElementById('quality').value !== 'high' && currentMode !== 'extend' && currentMode !== 'keyframe';
  document.querySelectorAll('#accelGroup .pill-btn').forEach(b => {
    const disabled = !allowed && b.dataset.accel !== 'off';
    b.classList.toggle('disabled', disabled);
  });
  if (!allowed && document.getElementById('accel').value !== 'off') setAccel('off');
}
function updateTemporalAvailability() {
  const allowed = temporalModeAllowed();
  document.querySelectorAll('#temporalGroup .pill-btn').forEach(b => {
    const disabled = !allowed && b.dataset.temporal !== 'native';
    b.classList.toggle('disabled', disabled);
    if (b.dataset.temporal === 'fps12_interp24') {
      b.title = allowed
        ? 'Generate at 12fps, then interpolate to a normal 24fps export.'
        : 'Available for Q4 Text/Image renders. Off for High, FFLF, Extend, and external-audio I2V.';
    }
  });
  if (!allowed && document.getElementById('temporal_mode').value !== 'native') setTemporalMode('native');
}
function setAspect(a) {
  if (!ASPECTS[a]) return;
  document.getElementById('aspect').value = a;
  document.querySelectorAll('#aspectGroup .pill-btn').forEach(b => b.classList.toggle('active', b.dataset.aspect === a));
  applyAspect(a);
}

// Compose the right-aligned line in the Customize summary. Reflects the
// current effective state: aspect, custom-dims callout, speed setting.
function updateCustomizeSummary() {
  const el = document.getElementById('customizeSummary');
  if (!el) return;
  const q = document.getElementById('quality').value;
  const w = parseInt(document.getElementById('width').value || 0);
  const h = parseInt(document.getElementById('height').value || 0);
  const aspect = document.getElementById('aspect').value || 'landscape';
  const accel = document.getElementById('accel').value || 'off';
  const upscale = document.getElementById('upscale').value || 'off';
  const parts = [];
  // Aspect (Quick is fixed 4:3, no choice; Standard/High show landscape/vertical).
  if (q === 'quick') parts.push('4:3 · 640×480');
  else parts.push(aspect === 'vertical' ? '9:16' : '16:9');
  // Flag custom dims if they don't match the preset.
  const preset = QUALITY_PRESETS[q] || QUALITY_PRESETS['standard'];
  const vertical = (aspect === 'vertical' && q !== 'quick');
  const expectedW = vertical ? preset.h : preset.w;
  const expectedH = vertical ? preset.w : preset.h;
  if (q !== 'quick' && (w !== expectedW || h !== expectedH)) {
    parts.push(`${w}×${h} custom`);
  }
  // Speed
  parts.push(accel === 'off' ? 'exact speed' : (accel === 'boost' ? 'boost' : 'turbo'));
  if ((document.getElementById('temporal_mode')?.value || 'native') === 'fps12_interp24') {
    parts.push('12→24fps long clip');
  }
  const method = (document.getElementById('upscale_method')?.value || 'lanczos');
  const methodTag = method === 'pipersr' || method === 'model' ? ' sharp' : '';
  if (upscale === 'fit_720p') parts.push('720p export' + methodTag);
  else if (upscale === 'x2') parts.push('2× export' + methodTag);
  el.textContent = parts.join(' · ');
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
document.querySelectorAll('#accelGroup .pill-btn').forEach(b => b.onclick = () => { if (!b.classList.contains('disabled')) setAccel(b.dataset.accel); });
document.querySelectorAll('#temporalGroup .pill-btn').forEach(b => b.onclick = () => { if (!b.classList.contains('disabled')) setTemporalMode(b.dataset.temporal); });
document.querySelectorAll('#upscaleGroup .pill-btn').forEach(b => b.onclick = () => { if (!b.classList.contains('disabled')) setUpscale(b.dataset.upscale); });
document.querySelectorAll('#upscaleMethodGroup .pill-btn').forEach(b => b.onclick = () => { if (!b.classList.contains('disabled')) setUpscaleMethod(b.dataset.method); });
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
  updateTemporalAvailability();
  updateCustomizeSummary();
  updateDerived();
});

function applyAspect(key) {
  if (!ASPECTS[key]) return;
  document.getElementById('aspect').value = key;
  // Aspect controls dimensions only when the active preset has a choice
  // (Standard / High at 1280×704 vs 704×1280). Quick is fixed 4:3 and
  // ignores the aspect picker (the row is hidden in that state, so this
  // path normally won't fire — defensive in case of programmatic calls).
  const q = document.getElementById('quality').value;
  if (q === 'quick') return;
  const preset = QUALITY_PRESETS[q] || QUALITY_PRESETS['standard'];
  const vertical = (key === 'vertical');
  document.getElementById('width').value  = vertical ? preset.h : preset.w;
  document.getElementById('height').value = vertical ? preset.w : preset.h;
  updateCustomizeSummary();
  updateDerived();
}

// applyQuality is kept as a tiny shim — old call sites (mode switching,
// etc.) call it expecting "set steps for the active quality." The
// dimensions are now owned by setQuality / applyAspect.
function applyQuality() {
  const q = document.getElementById('quality').value;
  if (q === 'high') {
    document.getElementById('steps').value = 18;
  } else {
    document.getElementById('steps').value = 8;       // quick + balanced + standard
  }
  updateCustomizeSummary();
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
  const dur = framesToDuration(f);

  const upscale = document.getElementById('upscale')?.value || 'off';
  let finalRes = `<strong>${w}×${h}</strong>`;
  if (upscale === 'fit_720p') {
    const tw = w >= h ? 1280 : 720;
    const th = w >= h ? 720 : 1280;
    finalRes = `${w}×${h} → <strong>${tw}×${th}</strong> fit`;
  } else if (upscale === 'x2') {
    finalRes = `${w}×${h} → <strong>${w * 2}×${h * 2}</strong>`;
  } else {
    let pw = w, ph = h;
    if (w === 704 && h % 16 === 0) pw = 720;
    if (h === 704 && w % 16 === 0) ph = 720;
    const padded = (pw !== w || ph !== h) && mode === 'i2v_clean_audio';
    finalRes = padded ? `${w}×${h} → <strong>${pw}×${ph}</strong>` : `<strong>${w}×${h}</strong>`;
  }
  const accel = document.getElementById('accel')?.value || 'off';
  const accelText = accel === 'off' ? '' : ` · ${accel === 'turbo' ? 'Turbo' : 'Boost'}`;
  const temporal = document.getElementById('temporal_mode')?.value || 'native';
  let temporalText = '';
  if (temporal === 'fps12_interp24') {
    const intervalSec = Math.max(0, (f - 1) / FPS);
    const sourceFrames = Math.max(1, Math.round(intervalSec * 12 / 8)) * 8 + 1;
    temporalText = ` · LTX ${sourceFrames}f @ 12fps → ${FPS}fps`;
  }

  document.getElementById('derived').innerHTML = `Duration <strong>${dur}s</strong> @ ${FPS}fps${temporalText} · ${finalRes} · Steps ${document.getElementById('steps').value}${accelText}`;

  const warns = [];
  if (w % 32 !== 0) warns.push(`Width ${w} isn't a multiple of 32 (closest ${Math.round(w/32)*32})`);
  if (h % 32 !== 0) warns.push(`Height ${h} isn't a multiple of 32 (closest ${Math.round(h/32)*32})`);
  if (f > 1 && (f - 1) % 8 !== 0) {
    const closest = Math.max(1, Math.round((f - 1) / 8) * 8 + 1);
    warns.push(`Frames work best as 8k+1 (closest ${closest})`);
  }
  if (temporal === 'fps12_interp24') {
    warns.push('12→24fps is experimental; check dialogue lip-sync and fast motion');
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
    // width / height: also refresh the Customize summary so "custom" flags
    // appear/disappear as the user types away from the preset values.
    el.addEventListener('input', () => { updateCustomizeSummary(); updateDerived(); });
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

  // Keyframe (FFLF) and Extend both require Q8 — server enforces it (see
  // run_job_inner). The UI was previously silently downgrading the user to
  // Standard when they picked keyframe with Q8 missing, then the server
  // would 500 on submit. Disable Generate + show a clear reason while in
  // that state. Y1.036 added Extend to the same gate after the Y1.024
  // download trim exposed that Extend is structurally Q8-class.
  const genBtn = document.getElementById('genBtn');
  const q8GatedMode = (currentMode === 'keyframe' || currentMode === 'extend');
  if (q8GatedMode && !s.q8_available) {
    genBtn.disabled = true;
    const modeName = currentMode === 'keyframe' ? 'Keyframe (FFLF)' : 'Extend';
    const left = (s.q8_missing || []).length;
    genBtn.title = left > 0 && left < 6
      ? `${modeName} needs Q8 — ${left} file(s) still downloading.`
      : `${modeName} needs the Q8 model. Click "Download Q8 (~37 GB)" in Pinokio.`;
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
  // Y1.039 — bar + meta line driven by server-computed progress (phase-aware,
  // config-bucketed ETA, denoise per-step extrapolation). Falls back to the
  // old elapsed/global-avg behavior if the server didn't ship a progress
  // block (e.g. mid-deploy where the server is older than the JS).
  const nowCard = document.getElementById('nowCard');
  const fill = document.getElementById('progressFill');
  if (s.running && s.current) {
    nowCard.classList.remove('idle', 'failed');
    const prog = s.current.progress || null;
    const elapsedFallback = Math.max(0, s.server_now - s.current.started_ts);
    let pct, elapsed, phaseLabel, timing;
    if (prog) {
      pct = Math.min(99, Math.max(0, prog.pct ?? 0));
      elapsed = prog.elapsed_sec ?? elapsedFallback;
      phaseLabel = prog.phase_label || 'Working';
      if (prog.remaining_sec != null && prog.remaining_sec > 0) {
        timing = `<strong>${fmtMin(elapsed)}</strong> in · ~${fmtMin(prog.remaining_sec)} left`;
      } else if (prog.eta_sec) {
        timing = `<strong>${fmtMin(elapsed)}</strong> / ~${fmtMin(prog.eta_sec)}`;
      } else {
        timing = `<strong>${fmtMin(elapsed)}</strong> elapsed`;
      }
    } else {
      // Legacy fallback path
      const avg = s.avg_elapsed_sec || 420;
      pct = Math.min(99, Math.round(elapsedFallback / avg * 100));
      elapsed = elapsedFallback;
      phaseLabel = '';
      timing = `<strong>${fmtMin(elapsed)}</strong> elapsed${avg ? ' / ~'+fmtMin(avg)+' avg' : ''}`;
    }
    fill.style.width = pct + '%';
    nowCard.querySelector('.ttl').textContent = snippet(s.current.params.label || s.current.params.prompt, 80);
    const baseMeta = `${s.current.params.mode} · ${s.current.params.width}×${s.current.params.height} · ${s.current.params.frames}f · ${timing}`;
    nowCard.querySelector('.meta').innerHTML = phaseLabel
      ? `${baseMeta}<br><span style="color:var(--muted)">${escapeHtml(phaseLabel)}</span>`
      : baseMeta;
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
               'or switch Quality to Quick (about half the RAM).';
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

// Format render duration for the gallery card sub-line. Falls back to
// the time-of-day when the sidecar is missing (older outputs that
// pre-date the elapsed_sec field, or outputs whose sidecar got
// deleted) so the slot is never empty.
function _outputDurationLabel(o) {
  const s = (o && typeof o.elapsed_sec === 'number') ? o.elapsed_sec : null;
  if (s == null) {
    // Fallback: show time-of-day from mtime so empty cards aren't worse.
    return o.mtime ? o.mtime.slice(11, 16) : '—';
  }
  if (s < 60)    return `${Math.round(s)} s`;
  if (s < 3600)  return `${Math.floor(s / 60)} m ${Math.round(s % 60)} s`;
  return `${Math.floor(s / 3600)} h ${Math.round((s % 3600) / 60)} m`;
}

function renderCarousel() {
  const el = document.getElementById('carousel');
  if (!currentOutputs.length) { el.innerHTML = '<div class="empty-msg">No outputs in this view yet.</div>'; return; }
  el.innerHTML = currentOutputs.map(o => {
    const pathAttr = JSON.stringify(o.path).replace(/"/g, '&quot;');
    // Thumbnail seek point: 2.5s is the midpoint of an LTX 5s clip (121
    // frames at 24fps ≈ 5.04s). The first half-second of LTX renders is
    // often dark/static (model fades into the scene), so #t=0.5 produced
    // "black thumbnail" complaints — the video was fine, the seek point
    // was the darkest moment. Mid-clip is reliably the visual peak.
    return `
    <div class="car-card${o.hidden ? ' hidden-card' : ''}${o.path === activePath ? ' active' : ''}"
         data-path="${escapeHtml(o.path)}" onclick="selectOutput(${pathAttr})">
      <video src="${o.url}#t=2.5" preload="metadata" muted></video>
      ${o.has_sidecar
        ? `<button class="car-info-btn" type="button" title="Show generation info"
                   onclick="event.stopPropagation(); openOutputInfoModal(${pathAttr})">ⓘ</button>`
        : ''}
      <div class="info">
        <div class="name" title="${escapeHtml(o.name)}">${escapeHtml(o.name)}</div>
        <div class="sub" title="Render time · file size">
          ${_outputDurationLabel(o)} · ${o.size_mb.toFixed(1)} MB
        </div>
      </div>
      <div class="row-btns">
        <button onclick="event.stopPropagation(); ${o.hidden ? 'unhide' : 'hide'}(${pathAttr})">${o.hidden ? 'Show' : 'Hide'}</button>
        <button onclick="event.stopPropagation(); useAsExtendSourcePath(${pathAttr})">Extend</button>
      </div>
    </div>`;
  }).join('');
}

function selectOutput(path) {
  activePath = path;
  document.querySelectorAll('.car-card').forEach(el => el.classList.toggle('active', el.dataset.path === path));
  const wrap = document.getElementById('playerWrap');
  wrap.classList.remove('empty');
  // Y1.039 — use the server-provided URL (which includes the mtime
  // cache-bust v=N param) instead of reconstructing from path. Otherwise
  // the player ends up on the cached stale-bytes URL and re-shows black
  // until the browser cache expires.
  const o = currentOutputs.find(x => x.path === path);
  const playerSrc = o ? o.url : `/file?path=${encodeURIComponent(path)}`;
  wrap.innerHTML = `<video controls autoplay src="${playerSrc}"></video>`;
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
  // Apply quality + aspect FIRST (these stomp on width/height), then
  // override with explicit sidecar values so any custom dims survive.
  if (p.quality) setQuality(p.quality);
  // Snap aspect from the sidecar's recorded dims; only call when quality
  // isn't 'quick' (Quick has no aspect choice and the row is hidden).
  if (p.quality !== 'quick' && p.width && p.height) {
    for (const [k, a] of Object.entries(ASPECTS)) {
      if ((a.w === p.width && a.h === p.height) ||
          (a.h === p.width && a.w === p.height)) { setAspect(k); break; }
    }
  }
  // Now load explicit dims — overrides whatever the preset/aspect set.
  if (p.width) document.getElementById('width').value = p.width;
  if (p.height) document.getElementById('height').value = p.height;
  if (p.accel) setAccel(p.accel);
  if (p.temporal_mode) setTemporalMode(p.temporal_mode);
  if (p.upscale) setUpscale(p.upscale);
  if (p.upscale_method) setUpscaleMethod(p.upscale_method);
  document.getElementById('prompt').value = p.prompt || '';
  document.getElementById('negative_prompt').value = p.negative_prompt || '';
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
  updateCustomizeSummary();
  updateDerived();
}

// ====== Output info modal ======
//
// Opened by the ⓘ button on each gallery card. Shows the full sidecar
// (.mp4.json) we wrote at render time: prompt, seed, mode, dimensions,
// frames, steps, LoRAs used (with display names + strengths), elapsed
// time, queue id, model. Plus per-field copy buttons for the things
// users actually want to reuse (prompt + seed).
//
// Why a modal and not inline detail-on-hover: the prompt alone can be
// 1000+ chars; trying to render it inline next to the thumbnail would
// blow up the gallery layout. Modal lets us scroll comfortably.

let _outputInfoLastPath = null;

async function openOutputInfoModal(path) {
  _outputInfoLastPath = path;
  const modal = document.getElementById('outputInfoModal');
  const body = document.getElementById('outputInfoBody');
  const title = document.getElementById('outputInfoTitle');
  modal.style.display = 'flex';
  body.innerHTML = '<div class="hint">Loading…</div>';
  // Display the filename in the modal title for quick orientation.
  const fname = path.split('/').pop();
  if (title) title.textContent = `Generation info · ${fname}`;
  let data;
  try {
    const r = await fetch('/sidecar?path=' + encodeURIComponent(path));
    if (!r.ok) {
      body.innerHTML = `<div class="hint">No sidecar metadata for this output (older generation, or sidecar was deleted).</div>`;
      return;
    }
    data = await r.json();
  } catch (e) {
    body.innerHTML = `<div class="hint">Couldn't load info: ${escapeHtml(e.message || String(e))}</div>`;
    return;
  }
  body.innerHTML = renderOutputInfoBody(path, data);
}

function closeOutputInfoModal() {
  document.getElementById('outputInfoModal').style.display = 'none';
}

function _copyToClipboard(text, btn) {
  // Best-effort copy with visual feedback. Falls back silently when the
  // clipboard API is blocked (e.g. iframe sandboxes without permissions).
  try {
    navigator.clipboard.writeText(text);
    if (btn) {
      const orig = btn.textContent;
      btn.textContent = 'Copied!';
      setTimeout(() => { btn.textContent = orig; }, 1200);
    }
  } catch (e) { /* swallow */ }
}

function _humanSize(b) {
  if (b == null) return '';
  if (b < 1024) return `${b} B`;
  if (b < 1024*1024) return `${(b/1024).toFixed(1)} KB`;
  if (b < 1024*1024*1024) return `${(b/1024/1024).toFixed(1)} MB`;
  return `${(b/1024/1024/1024).toFixed(2)} GB`;
}

function _humanDuration(s) {
  if (s == null) return '';
  if (s < 60) return `${s.toFixed(1)} s`;
  const m = Math.floor(s / 60); const r = (s - m*60).toFixed(0);
  return `${m} min ${r} s`;
}

function renderOutputInfoBody(path, data) {
  const p = (data && data.params) || {};
  const loras = Array.isArray(p.loras) ? p.loras : [];

  // Look up each LoRA's display name from the installed-LoRAs cache so
  // the modal shows "Claymation Style" instead of the raw safetensors
  // path. Falls back gracefully when a LoRA was deleted or is an HF id.
  const lookupLoraName = (loraPath) => {
    if (!loraPath) return '?';
    const known = (_knownUserLoras || []).find(l => l.path === loraPath);
    if (known) return known.name;
    if (loraPath.includes('/') && !loraPath.endsWith('.safetensors')) return loraPath;
    return loraPath.split('/').pop().replace(/\.safetensors$/, '');
  };

  const promptText = p.prompt || '';
  const promptAttr = JSON.stringify(promptText).replace(/"/g, '&quot;');
  const seedVal = String(p.seed_used != null ? p.seed_used : p.seed || '');
  const seedAttr = JSON.stringify(seedVal).replace(/"/g, '&quot;');
  const pathAttr = JSON.stringify(path).replace(/"/g, '&quot;');
  const accelMetrics = (data && data.accel_metrics) || null;
  const modeLabel = ({
    t2v: 'Text → Video',
    i2v: 'Image → Video',
    i2v_clean_audio: 'Image → Video (clean audio)',
    keyframe: 'FFLF (first + last frame)',
    extend: 'Extend',
  })[p.mode] || (p.mode || '—');

  // Compose the dimensions + duration into a single "Format" line — fewer
  // grid rows, easier to scan. We separate technical metadata (Format,
  // Frames) from generation parameters (Mode, Quality, Seed, Steps).
  const formatBits = [];
  if (p.width && p.height) formatBits.push(`${p.width} × ${p.height}`);
  if (data.video_duration_sec != null) formatBits.push(`${data.video_duration_sec.toFixed(2)} s @ ${data.fps || 24} fps`);

  let html = '';

  // ---- Output (technical) ----
  html += `<div class="oi-section">
    <div class="oi-section-title"><span>Output</span></div>
    <dl class="oi-grid">
      ${formatBits.length ? `<dt>Format</dt><dd>${formatBits.join('  ·  ')}</dd>` : ''}
      ${p.frames != null ? `<dt>Frames</dt><dd>${p.frames}</dd>` : ''}
    </dl>
  </div>`;

  // ---- Generation parameters ----
  const genRows = [];
  genRows.push(`<dt>Mode</dt><dd>${escapeHtml(modeLabel)}</dd>`);
  genRows.push(`<dt>Quality</dt><dd>${escapeHtml((p.quality || 'standard').replace(/^./, c => c.toUpperCase()))}</dd>`);
  if (p.accel && p.accel !== 'off') {
    genRows.push(`<dt>Speed</dt><dd>${escapeHtml(p.accel.replace(/^./, c => c.toUpperCase()))}</dd>`);
  }
  if (accelMetrics && p.accel && p.accel !== 'off') {
    const cachedCount = accelMetrics.cached_steps_count || 0;
    const totalSteps = accelMetrics.total_steps || p.steps || 0;
    const savings = accelMetrics.estimated_denoise_call_savings_pct;
    const cachedList = Array.isArray(accelMetrics.cached_steps) && accelMetrics.cached_steps.length
      ? ` · cached steps ${escapeHtml(accelMetrics.cached_steps.join(', '))}`
      : '';
    const savingsText = savings != null ? ` · ~${escapeHtml(String(savings))}% denoise calls saved` : '';
    genRows.push(`<dt>Accel metrics</dt><dd>${cachedCount}/${totalSteps} cached${savingsText}${cachedList}</dd>`);
  }
  if (p.temporal_mode === 'fps12_interp24' || data.temporal) {
    const t = data.temporal || {};
    const sourceFrames = t.source_frames || p.model_frames || '—';
    const deliveryFrames = t.delivery_frames || p.frames || '—';
    const sourceFps = t.model_fps || p.model_fps || 12;
    const deliveryFps = t.delivery_fps || p.delivery_fps || 24;
    genRows.push(`<dt>Long clips</dt><dd>12 → 24fps · LTX ${escapeHtml(String(sourceFrames))}f @ ${escapeHtml(String(sourceFps))}fps → ${escapeHtml(String(deliveryFrames))}f @ ${escapeHtml(String(deliveryFps))}fps</dd>`);
  }
  if (p.upscale && p.upscale !== 'off') {
    const up = data.upscale || {};
    const target = up.target_w && up.target_h ? ` → ${up.target_w} × ${up.target_h}` : '';
    const isSharp = p.upscale_method === 'pipersr' || p.upscale_method === 'model' || (data.upscale && (data.upscale.method === 'pipersr_coreml' || data.upscale.pre_pass === 'pipersr_x2' || data.upscale.method === 'ltx_latent_x2' || data.upscale.pre_pass === 'ltx_latent_x2'));
    const baseLabel = p.upscale === 'fit_720p' ? '720p fit (no crop)' : (p.upscale === 'x2' ? '2×' : p.upscale);
    const label = isSharp ? `${baseLabel} · Sharp (PiperSR)` : `${baseLabel} · Fast (Lanczos)`;
    genRows.push(`<dt>Upscale</dt><dd>${escapeHtml(label + target)}</dd>`);
  }
  const codec = data.output_codec || (data.upscale && data.upscale.codec);
  if (codec && codec.pix_fmt && codec.crf != null) {
    const preset = codec.preset ? ` · ${codec.preset}` : '';
    genRows.push(`<dt>Output codec</dt><dd>${escapeHtml(codec.pix_fmt)} · CRF ${escapeHtml(String(codec.crf))}${escapeHtml(preset)}</dd>`);
  }
  if (p.negative_prompt) {
    genRows.push(`<dt>Avoid</dt><dd>${escapeHtml(snippet(p.negative_prompt, 90))}</dd>`);
  }
  if (seedVal) {
    genRows.push(`<dt>Seed</dt><dd>
      <code>${escapeHtml(seedVal)}</code>
      <button class="oi-copy" type="button" onclick="_copyToClipboard(${seedAttr}, this)">Copy</button>
    </dd>`);
  }
  if (p.steps != null) genRows.push(`<dt>Steps</dt><dd>${p.steps}</dd>`);
  if (p.hdr) genRows.push(`<dt>HDR</dt><dd>On</dd>`);
  if (p.label) genRows.push(`<dt>Label</dt><dd>${escapeHtml(p.label)}</dd>`);

  html += `<div class="oi-section">
    <div class="oi-section-title"><span>Generation</span></div>
    <dl class="oi-grid">${genRows.join('')}</dl>
  </div>`;

  // ---- Prompt ----
  if (promptText) {
    html += `<div class="oi-section">
      <div class="oi-section-title">
        <span>Prompt</span>
        <button class="oi-copy" type="button" onclick="_copyToClipboard(${promptAttr}, this)">Copy</button>
      </div>
      <div class="oi-prompt">${escapeHtml(promptText)}</div>
    </div>`;
  }

  // ---- LoRAs (flat list, hairline-separated) ----
  if (loras.length) {
    const rows = loras.map(l => {
      const name = lookupLoraName(l.path);
      const strength = (l.strength != null ? l.strength : 1).toFixed(2);
      return `<div class="oi-lora-row">
        <span class="oi-lora-name" title="${escapeHtml(l.path || '')}">${escapeHtml(name)}</span>
        <span class="oi-lora-strength">strength ${strength}</span>
      </div>`;
    }).join('');
    html += `<div class="oi-section">
      <div class="oi-section-title">
        <span>LoRAs used</span>
        <span class="oi-count">${loras.length}</span>
      </div>
      <div class="oi-lora-list">${rows}</div>
    </div>`;
  }

  // ---- Timing + provenance ----
  const timingRows = [];
  if (data.started) timingRows.push(`<dt>Started</dt><dd>${escapeHtml(data.started)}</dd>`);
  if (data.elapsed_sec != null) timingRows.push(`<dt>Elapsed</dt><dd>${_humanDuration(data.elapsed_sec)}</dd>`);
  if (data.queue_id) timingRows.push(`<dt>Queue ID</dt><dd><code>${escapeHtml(data.queue_id)}</code></dd>`);
  if (data.model) timingRows.push(`<dt>Model</dt><dd><code>${escapeHtml(data.model.split('/').pop())}</code></dd>`);
  if (timingRows.length) {
    html += `<div class="oi-section">
      <div class="oi-section-title"><span>Timing</span></div>
      <dl class="oi-grid">${timingRows.join('')}</dl>
    </div>`;
  }

  // ---- Action row ----
  html += `<div class="oi-actions">
    <button class="ghost-btn" type="button" onclick="closeOutputInfoModal()">Close</button>
    <button class="oi-primary" type="button"
            onclick="closeOutputInfoModal(); selectOutput(${pathAttr}); loadParams()">
      Load params into form
    </button>
  </div>`;

  return html;
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
    sub.innerHTML = `Q4 (~20 GB) and Gemma (~6 GB) are required. Click below — downloads resume if interrupted.${
      missing ? ` <span style="color:var(--muted)">(${missing} files left)</span>` : ''
    }`;
    actions.innerHTML = (s.hf_available ?? true)
      ? `<button onclick="startDownload('q4')">Download Q4 (20 GB)</button>`
      : `<button disabled title="hf binary not found — reinstall via Pinokio">hf missing</button>`;
    return;
  }

  // ----- User picked a mode that needs Q8, but Q8 isn't there --------------
  // FFLF + Extend + High quality all need Q8. Surface the CTA *only* when
  // the user is about to do one of those — no point nagging a T2V user
  // about Q8 if they'll never use it.
  // Dismissible: a user who deliberately doesn't want Q8 (storage budget,
  // they only do T2V Quick/Standard) can × this away and we'll respect it
  // until either model state changes or they re-summon the modal.
  // Y1.036 — Extend joins FFLF and High in needing Q8. The Extend pipeline
  // loads `transformer-dev.safetensors` for CFG-guided denoise; Q4 doesn't
  // ship it after the Y1.024 download trim, so surface the same CTA here.
  const needsQ8 = (currentMode === 'keyframe')
                || (currentMode === 'extend')
                || (document.getElementById('quality').value === 'high');
  if (needsQ8 && !q8Ok && tier.allows_q8 !== false) {
    if (dismissed) { card.style.display = 'none'; return; }
    card.style.display = '';
    card.classList.add('state-warn', 'dismissible');
    icon.textContent = '⬇';
    const reason = currentMode === 'keyframe' ? 'FFLF needs the Q8 model'
                : currentMode === 'extend'    ? 'Extend needs the Q8 model'
                                              : 'High quality needs the Q8 model';
    title.textContent = reason;
    const missing = (s.q8_missing || []).length;
    sub.innerHTML = `Q8 (~37 GB) is a separate one-time download. Resumable.${
      missing && missing < 8 ? ` <span style="color:var(--muted)">(${missing} files left — partial install detected)</span>` : ''
    }`;
    actions.innerHTML = (s.hf_available ?? true)
      ? `<button onclick="startDownload('q8')">Download Q8 (37 GB)</button>`
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
        title: 'Quick (640×480)',
        desc: 'Smaller preview to scout prompts and seeds before a full-size render.',
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
  // Placeholders reflect the saved state so an empty input doesn't read
  // as "no token here" when there is one. The asterisks make it clear
  // something's persisted; the hint reminds users they paste to replace.
  const civInput = document.getElementById('civitaiKeyInput');
  const hfInput = document.getElementById('hfTokenInput');
  civInput.value = '';
  hfInput.value = '';
  civInput.placeholder = cur.has_civitai_key
    ? '•••••••••• saved — paste new to replace'
    : '32-char API key';
  hfInput.placeholder = cur.has_hf_token
    ? '•••••••••• saved — paste new to replace'
    : 'hf_…';
  document.getElementById('civitaiKeyClear').style.display = cur.has_civitai_key ? '' : 'none';
  document.getElementById('hfTokenClear').style.display = cur.has_hf_token ? '' : 'none';

  // Spicy mode — render current state. _spicyArmed is the mid-confirm
  // state (clicked once, waiting for the second click). It lives only
  // on the JS side; only ON/OFF gets persisted.
  _spicyArmed = false;
  renderSpicyState(!!cur.spicy_mode);
}

let _spicyArmed = false;

function renderSpicyState(isOn) {
  const badge = document.getElementById('spicyStateBadge');
  const btn = document.getElementById('spicyToggleBtn');
  const hint = document.getElementById('spicyHint');
  if (!badge || !btn) return;
  badge.classList.remove('on', 'armed');
  if (_spicyArmed) {
    badge.textContent = 'ARMED';
    badge.classList.add('armed');
    btn.textContent = 'Click again to confirm';
    btn.classList.remove('ghost-btn');
    btn.classList.add('primary-btn');
    hint.style.display = '';
    hint.textContent = 'Confirms turning Spicy mode ON. NSFW LoRAs will be available in the CivitAI browser. Cancel by closing the modal.';
  } else if (isOn) {
    badge.textContent = 'ON';
    badge.classList.add('on');
    btn.textContent = 'Disable';
    btn.classList.remove('primary-btn');
    btn.classList.add('ghost-btn');
    hint.style.display = '';
    hint.textContent = 'Spicy mode is ON. NSFW LoRAs are visible in the CivitAI browser when you tick "Show NSFW".';
  } else {
    badge.textContent = 'OFF';
    btn.textContent = 'Enable Spicy mode';
    btn.classList.remove('primary-btn');
    btn.classList.add('ghost-btn');
    hint.style.display = 'none';
    hint.textContent = '';
  }
}

async function toggleSpicyMode() {
  // Two-click confirm to turn ON, single-click to turn OFF.
  // Easy to disable, deliberate to enable — matches the user spec
  // ("don't want people to turn it on by mistake, or kids").
  const cur = (_settingsCache && _settingsCache.settings) || {};
  const isOn = !!cur.spicy_mode;
  if (isOn) {
    // Single-click off, no confirm.
    await _persistSpicyMode(false);
    return;
  }
  if (!_spicyArmed) {
    _spicyArmed = true;
    renderSpicyState(false);
    // Auto-disarm after 6 s if the user doesn't confirm — prevents
    // the "click again" state lingering across an unrelated tab return.
    setTimeout(() => {
      if (_spicyArmed) {
        _spicyArmed = false;
        renderSpicyState(!!(_settingsCache?.settings?.spicy_mode));
      }
    }, 6000);
    return;
  }
  // Second click — actually persist.
  _spicyArmed = false;
  await _persistSpicyMode(true);
}

async function _persistSpicyMode(target) {
  const status = document.getElementById('settingsStatus');
  try {
    const fd = new URLSearchParams();
    fd.set('spicy_mode', target ? 'true' : 'false');
    const r = await fetch('/settings', { method: 'POST', body: fd });
    const j = await r.json();
    if (j.error) throw new Error(j.error);
    if (_settingsCache && _settingsCache.settings) {
      _settingsCache.settings.spicy_mode = !!target;
    }
    renderSpicyState(!!target);
    if (status) {
      status.textContent = target ? 'Spicy mode ON · NSFW LoRAs unlocked' : 'Spicy mode OFF · NSFW LoRAs hidden';
      status.className = 'settings-status ok';
    }
    // Refresh the CivitAI panel so the "Show NSFW" toggle appears /
    // disappears immediately without a full page reload.
    if (typeof refreshCivitaiAccessUI === 'function') refreshCivitaiAccessUI();
  } catch (e) {
    if (status) {
      status.textContent = 'Could not change Spicy mode: ' + (e.message || e);
      status.className = 'settings-status err';
    }
  }
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

// Save-then-test in one click. The /civitai/test and /hf/test endpoints
// use the saved key, not the current input field. Pre-Y1.023 the user
// had to: paste → click Apply → click Test. That left a footgun: users
// pasted, clicked Test, saw it fail (because nothing was saved yet),
// closed the modal thinking the panel was broken. The token never
// landed in panel_settings.json, gated downloads kept failing.
//
// Now Test does Save first when the input has a value, so a single
// click works. If the save fails (validator rejects malformed token),
// we surface the error inline next to the field instead of just at
// the bottom of the modal.
async function testToken(which) {
  const path = which === 'civitai' ? '/civitai/test' : '/hf/test';
  const resultId = which === 'civitai' ? 'civitaiTestResult' : 'hfTestResult';
  const inputId = which === 'civitai' ? 'civitaiKeyInput' : 'hfTokenInput';
  const fieldName = which === 'civitai' ? 'civitai_api_key' : 'hf_token';
  const statusPrefix = which === 'civitai' ? 'civitaiKey' : 'hfToken';
  const clearBtnId = which === 'civitai' ? 'civitaiKeyClear' : 'hfTokenClear';
  const result = document.getElementById(resultId);
  if (!result) return;
  result.textContent = 'Testing…';
  result.style.color = 'var(--muted)';

  // If the input has content, save it first. Empty input means "test
  // the already-saved token" — the legitimate use after the panel is
  // configured.
  const inputEl = document.getElementById(inputId);
  const inputValue = inputEl ? inputEl.value.trim() : '';
  if (inputValue) {
    try {
      const fd = new URLSearchParams();
      fd.set(fieldName, inputValue);
      const saveResp = await fetch('/settings', {
        method: 'POST',
        headers: {'Content-Type': 'application/x-www-form-urlencoded'},
        body: fd,
      });
      const saveData = await saveResp.json();
      if (!saveResp.ok || saveData.error) {
        result.innerHTML = `<strong style="color: var(--danger, #f85149)">✗</strong> ${escapeHtml(saveData.error || `HTTP ${saveResp.status}`)}`;
        return;
      }
      // Save succeeded — reflect the persisted state in the UI.
      inputEl.value = '';
      _settingsCache = { ...(_settingsCache || {}), settings: saveData.settings };
      setTokenStatus(statusPrefix, true);
      const clearBtn = document.getElementById(clearBtnId);
      if (clearBtn) clearBtn.style.display = '';
    } catch (e) {
      result.innerHTML = `<strong style="color: var(--danger, #f85149)">✗</strong> Save failed: ${escapeHtml(e.message || String(e))}`;
      return;
    }
  }

  // Now hit the test endpoint, which reads the freshly-saved token.
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
  // the in-memory list for UI rendering, drop it on the wire. Summary
  // count is updated by renderLorasList() which has fuller state — we
  // don't touch it here to avoid two functions stomping each other.
  const slim = _activeLoras.map(l => ({ path: l.path, strength: l.strength }));
  document.getElementById('lorasJson').value = JSON.stringify(slim);
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
  const filterRow = document.getElementById('lorasFilterRow');
  const filterInput = document.getElementById('lorasFilter');
  if (!wrap) return;
  // Combine: user-installed LoRAs (from /loras) plus any active LoRAs
  // that aren't user-installed (HF repo paths, e.g. from the HDR toggle).
  const allRows = [];
  const seen = new Set();
  for (const ul of _knownUserLoras) {
    const active = _activeLoras.find(a => a.path === ul.path);
    seen.add(ul.path);
    allRows.push({
      path: ul.path,
      name: ul.name,
      trigger_words: ul.trigger_words || [],
      recommended_strength: ul.recommended_strength || 1.0,
      filename: ul.filename,
      civitai_url: ul.civitai_url,
      active: !!active,
      strength: active ? active.strength : (ul.recommended_strength || 1.0),
      kind: 'user',
    });
  }
  for (const a of _activeLoras) {
    if (seen.has(a.path)) continue;
    allRows.push({
      path: a.path,
      name: a.name || a.path,
      trigger_words: a.trigger_words || [],
      recommended_strength: 1.0,
      filename: null,
      civitai_url: null,
      active: true,
      strength: a.strength,
      kind: 'remote',
    });
  }

  // Empty state — collapse the filter box too.
  if (allRows.length === 0) {
    wrap.innerHTML = '';
    if (empty) empty.style.display = '';
    if (filterRow) filterRow.style.display = 'none';
    return;
  }
  if (empty) empty.style.display = 'none';
  // Surface the filter input only when 5+ LoRAs are installed; below that
  // the box is just visual noise.
  if (filterRow) filterRow.style.display = (allRows.length >= 5) ? '' : 'none';

  // Apply filter (case-insensitive substring on name + trigger words).
  let rows = allRows;
  const q = (filterInput && filterInput.value || '').trim().toLowerCase();
  if (q) {
    rows = allRows.filter(r => {
      if (r.name && r.name.toLowerCase().includes(q)) return true;
      for (const t of (r.trigger_words || [])) {
        if (String(t).toLowerCase().includes(q)) return true;
      }
      return false;
    });
  }
  // Sort: active rows first (so the user's selection floats to the top),
  // then alphabetical by name. Stable enough for a UI list.
  rows.sort((a, b) => {
    if (a.active !== b.active) return a.active ? -1 : 1;
    return (a.name || '').localeCompare(b.name || '');
  });

  // Update header summary.
  const summary = document.getElementById('lorasSummaryCount');
  if (summary) {
    const total = allRows.length;
    const active = allRows.filter(r => r.active).length;
    summary.textContent = `${total} installed · ${active} active${q ? ` · ${rows.length} match` : ''}`;
  }

  if (rows.length === 0) {
    wrap.innerHTML = `<div class="hint" style="padding:8px 0;">No LoRAs match "${escapeHtml(q)}".</div>`;
    return;
  }
  wrap.innerHTML = rows.map(r => loraRowHtml(r)).join('');
}

// Build a single compact LoRA row. Inactive rows are ~36px tall (just
// name + meta + corner actions). Active rows expand inline with the
// strength slider and trigger chips. Click anywhere on the main row to
// toggle activation.
function loraRowHtml(r) {
  const pathHtml = escapeHtml(r.path);
  const pathAttr = JSON.stringify(r.path).replace(/"/g, '&quot;');
  const nameHtml = escapeHtml(r.name);
  const nameAttr = JSON.stringify(r.name).replace(/"/g, '&quot;');
  // Trigger summary line under the name (when not expanded). Truncated.
  const trigs = r.trigger_words || [];
  const trigSummary = trigs.length
    ? trigs.slice(0, 4).join(' · ') + (trigs.length > 4 ? ` +${trigs.length - 4}` : '')
    : 'no trigger word';
  // Corner actions — link to civitai page + delete (or remove for HF/remote).
  const corner = [];
  if (r.civitai_url) {
    corner.push(`<a class="lora-icon-btn" href="${escapeHtml(r.civitai_url)}" target="_blank" rel="noopener" title="Open on CivitAI" onclick="event.stopPropagation()">↗</a>`);
  }
  if (r.kind === 'user') {
    corner.push(`<button class="lora-icon-btn danger" type="button" title="Delete from disk"
                         onclick="event.stopPropagation(); deleteLora(${pathAttr}, ${nameAttr})">×</button>`);
  } else {
    corner.push(`<button class="lora-icon-btn" type="button" title="Remove from active set"
                         onclick="event.stopPropagation(); removeLoraFromActive(${pathAttr})">×</button>`);
  }
  // Trigger chips for the expanded section. Same click-to-append behavior
  // as before — chips prepend the trigger to the prompt textarea.
  const chipsHtml = trigs.length
    ? trigs.slice(0, 12).map(w => {
        const wAttr = JSON.stringify(w).replace(/"/g, '&quot;');
        return `<span class="trigger-chip" title="Click to add to prompt"
                       onclick="event.stopPropagation(); appendTriggerToPrompt(${wAttr})">${escapeHtml(w)}</span>`;
      }).join('')
    : `<span class="trigger-chip empty">style-only LoRA — no trigger word needed</span>`;

  return `
    <div class="lora-row ${r.active ? 'active' : ''}" data-path="${pathHtml}">
      <div class="lora-row-main"
           onclick="toggleLora(${pathAttr}, ${!r.active}, ${r.recommended_strength}, ${nameAttr})">
        <div class="lora-toggle-dot"></div>
        <div class="lora-text">
          <div class="lora-name" title="${pathHtml}">
            ${nameHtml}${r.kind === 'remote' ? '<span class="badge">HF</span>' : ''}
          </div>
          <div class="lora-name-meta" title="${escapeHtml(trigs.join(', '))}">${escapeHtml(trigSummary)}</div>
        </div>
        <div class="lora-row-actions">${corner.join('')}</div>
      </div>
      <div class="lora-row-extra">
        <div class="lora-strength-row">
          <label>strength</label>
          <input type="range" min="-2" max="2" step="0.05" value="${r.strength}"
                 onclick="event.stopPropagation()"
                 oninput="this.nextElementSibling.value = this.value; setLoraStrength(${pathAttr}, this.value)">
          <input type="number" min="-2" max="2" step="0.05" value="${r.strength}"
                 onclick="event.stopPropagation()"
                 oninput="this.previousElementSibling.value = this.value; setLoraStrength(${pathAttr}, this.value)">
        </div>
        <div class="trigger-chips">${chipsHtml}</div>
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
  // Pull /loras to populate the dir text and the auth-banner state.
  fetch('/loras').then(r => r.json()).then(d => {
    const dirEl = document.getElementById('civitaiTargetDir');
    if (dirEl && d.loras_dir) dirEl.textContent = d.loras_dir;
    renderCivitaiAuthBanner(!!d.civitai_auth);
  }).catch(() => { renderCivitaiAuthBanner(false); });
  document.getElementById('civitaiQuery').value = '';
  _civitaiCursor = '';
  // Pull current Spicy mode state so the "Show NSFW" toggle hides when off.
  refreshCivitaiAccessUI();
  civitaiSearch();
}

// Hide / show the "Show NSFW" toggle in the CivitAI browser based on the
// Spicy mode setting. Called on modal open and after toggleSpicyMode flips
// the value, so the UI tracks the gate without a page reload.
async function refreshCivitaiAccessUI() {
  let spicy = false;
  try {
    const r = await fetch('/settings');
    const j = await r.json();
    spicy = !!(j && j.settings && j.settings.spicy_mode);
  } catch (_) { /* default off */ }
  const pill = document.getElementById('civitaiNsfwPill');
  const cb = document.getElementById('civitaiNsfw');
  if (pill) pill.style.display = spicy ? '' : 'none';
  if (!spicy && cb) cb.checked = false;  // force off when spicy mode is off
}

// Render the inline API-key banner at the top of the CivitAI browser.
// Three states: set (✓ small green), missing (amber, prompts for key),
// editing (input visible while user is changing/setting the key). The
// banner is the primary surface for the key now — Settings still has
// the field but most users won't need to dig there.
function renderCivitaiAuthBanner(haveKey, mode) {
  const box = document.getElementById('civitaiAuthBanner');
  if (!box) return;
  // Three visual modes: 'view' (default), 'edit' (showing input), 'err' (last save failed).
  const m = mode || (haveKey ? 'view' : 'edit');
  box.style.display = '';
  box.classList.remove('missing','set','err');
  if (m === 'view' && haveKey) {
    box.classList.add('set');
    box.innerHTML = `
      <span><strong style="color:var(--success,#3fb950)">✓</strong> CivitAI API key set —
      LoRA downloads will work.</span>
      <span class="grow"></span>
      <a class="changekey" onclick="renderCivitaiAuthBanner(true,'edit')">change key</a>`;
    return;
  }
  // edit / missing mode — render input + Save.
  box.classList.add(m === 'err' ? 'err' : 'missing');
  const intro = m === 'err'
    ? `<strong>That key didn't work.</strong> Double-check it from <a href="https://civitai.com/user/account" target="_blank" rel="noopener">civitai.com/user/account</a> and try again.`
    : haveKey
      ? `Replace your CivitAI API key. The current one stays active until you save a new one.`
      : `<strong>CivitAI requires an API key</strong> to download LoRAs. Get one at <a href="https://civitai.com/user/account" target="_blank" rel="noopener">civitai.com/user/account</a> and paste it here:`;
  box.innerHTML = `
    <div class="grow" style="flex-basis:100%; margin-bottom:6px;">${intro}</div>
    <input type="password" id="civitaiAuthInput" placeholder="paste API key — usually 32 hex chars"
           autocomplete="off" spellcheck="false">
    <button type="button" id="civitaiAuthSave" onclick="civitaiAuthSave()">Save & test</button>
    ${haveKey ? '<a class="changekey" onclick="renderCivitaiAuthBanner(true,\'view\')">cancel</a>' : ''}`;
  // Pressing Enter inside the input triggers save.
  const inp = document.getElementById('civitaiAuthInput');
  if (inp) inp.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); civitaiAuthSave(); }
  });
}

async function civitaiAuthSave() {
  const inp = document.getElementById('civitaiAuthInput');
  const btn = document.getElementById('civitaiAuthSave');
  if (!inp) return;
  const key = (inp.value || '').trim();
  if (!key) { inp.focus(); return; }
  if (btn) { btn.disabled = true; btn.textContent = 'Saving…'; }
  // Save via /settings (single source of truth for tokens). After save
  // we hit /civitai/test to verify before flipping the banner to "set" —
  // that catches the most common error (typo'd key) right at the moment
  // the user pasted it, instead of failing later on the first download.
  try {
    const fd = new URLSearchParams();
    fd.set('civitai_api_key', key);
    const r = await fetch('/settings', { method: 'POST',
      headers: {'Content-Type':'application/x-www-form-urlencoded'}, body: fd });
    const data = await r.json();
    if (!r.ok || !data.ok) throw new Error(data.error || `HTTP ${r.status}`);
    // Verify
    if (btn) btn.textContent = 'Testing…';
    const t = await fetch('/civitai/test');
    const td = await t.json();
    if (!td.ok) {
      renderCivitaiAuthBanner(false, 'err');
      return;
    }
    renderCivitaiAuthBanner(true, 'view');
    // Re-run the search so any 401-blocked thumbnails reload as authed.
    civitaiSearch();
  } catch (e) {
    if (btn) { btn.disabled = false; btn.textContent = 'Save & test'; }
    renderCivitaiAuthBanner(false, 'err');
  }
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

// ====== Version pill (the "magic button") ======
//
// One always-visible pill in the header that changes content + colour
// based on /version state. Clicking it does the right thing for the
// current state — no modal, no nested click flows.
//
// Backend: a daemon thread polls GitHub every 30 minutes (commits API
// for the SHA + raw VERSION file for the human-friendly Y1.NNN label)
// and exposes the result at /version. The JS polls /version every 5
// minutes (cheap; pre-computed dict read). When the user clicks while
// behind, /version/pull does the actual `git pull` server-side; the
// user still has to Stop+Start phosphene in Pinokio to apply.
//
// Rationale: users keep telling us "I clicked Update but I don't see
// the new features" — by the time the feedback reaches us we've usually
// pushed three more commits. The pill turns the loop from
// "hope-they-noticed" into a literal one-click action.

let _versionState = null;
let _versionRestartPending = false;   // set after a successful /version/pull;
                                      // pill turns into a "restart" reminder.

async function refreshVersionPill() {
  try {
    const r = await fetch('/version');
    _versionState = await r.json();
  } catch (e) {
    return;             // network blip; don't blow away last good state
  }
  renderVersionPill();
}

function _versionDisplayLabel(s) {
  // Prefer the human Y1.NNN VERSION file label. Fall back to the short
  // SHA for older checkouts that predate the VERSION file. Last-resort
  // ellipsis when nothing's known yet.
  return s.local_version || s.local_short || '…';
}

function _versionRemoteLabel(s) {
  return s.remote_version || s.remote_short || 'latest';
}

function renderVersionPill() {
  const pill = document.getElementById('versionPill');
  if (!pill) return;
  const s = _versionState || {};
  const local = _versionDisplayLabel(s);
  const remote = _versionRemoteLabel(s);
  // Reset every state class; exactly one is added below.
  pill.classList.remove('pill-update','pill-current','pill-dev','pill-checking','pill-restart','pill-busy');
  pill.style.display = '';

  // Pill text leads with the MEANING of the state, not the version code.
  // Earlier build showed bare "Y1.005" which read as a label rather than
  // a status — users didn't realize they could click it. Now every state
  // uses plain English so a user glancing at the header understands at
  // a glance whether they're current, behind, or need to restart.

  // Highest-priority state: a pull just happened and the panel needs a
  // restart to load the new code.
  if (_versionRestartPending) {
    pill.classList.add('pill-restart');
    pill.textContent = '↻ Restart Phosphene';
    const v = s.pull_pulled_to_version || s.pull_pulled_to_short || 'the new code';
    pill.title = s.pull_requires_full_update
      ? `Pulled ${v}. This update touched dependencies — use Pinokio's Update button (not just Stop+Start).`
      : `Pulled ${v}. Click Stop → Start in Pinokio to apply.`;
    return;
  }
  // Suppressed (dev branch / dirty tree / no git).
  if (s.suppress_reason) {
    pill.classList.add('pill-dev');
    pill.textContent = `${local} · dev`;
    pill.title = `Update check paused: ${s.suppress_reason}.`;
    return;
  }
  // Behind origin/main — eye-catching action prompt.
  if (!s.error && s.checked_ts && (s.behind_by | 0) > 0) {
    pill.classList.add('pill-update');
    pill.textContent = `↑ Update to ${remote}`;
    pill.title = `You're on ${local}; latest is ${remote}. Click to pull the update.`;
    return;
  }
  // Last check errored (offline).
  if (s.error) {
    pill.classList.add('pill-dev');
    pill.textContent = `${local} · offline`;
    pill.title = `Couldn't reach github.com (${s.error}). Click to retry.`;
    return;
  }
  // Current with origin/main.
  if (s.checked_ts && (s.behind_by | 0) === 0) {
    pill.classList.add('pill-current');
    pill.textContent = `✓ Up to date · ${local}`;
    pill.title = `You're on ${local}, the latest version. Click to re-check now.`;
    return;
  }
  // First poll hasn't landed yet.
  pill.classList.add('pill-checking');
  pill.textContent = `Checking · ${local}`;
  pill.title = 'Checking for updates…';
}

// One click — does the right thing for the current state. Magic button.
async function versionPillClick() {
  if (_versionRestartPending) {
    // Educational click: tell the user what's needed.
    const s = _versionState || {};
    const tip = s.pull_requires_full_update
      ? "Pulled. Because this update touched Python deps / patches, use Pinokio's Update button (it reinstalls + reapplies patches). After that click Start."
      : "Pulled. Click Stop, then Start in Pinokio to apply (your queue and settings are preserved).";
    alert(tip);
    return;
  }
  const s = _versionState || {};
  if (s.suppress_reason) {
    alert(`Update check is paused: ${s.suppress_reason}.\n\n` +
          `Phosphene only checks GitHub when you're on a clean main branch. ` +
          `Commit your local changes (or switch back to main) to re-enable updates.`);
    return;
  }
  // Behind: pull the update.
  if (!s.error && s.checked_ts && (s.behind_by | 0) > 0) {
    await versionDoPull();
    return;
  }
  // Current OR error OR pre-first-poll: re-check now.
  await versionDoRefresh();
}

async function versionDoRefresh() {
  const pill = document.getElementById('versionPill');
  pill.classList.add('pill-busy');
  const origText = pill.textContent;
  pill.textContent = '⟳ checking…';
  try {
    const r = await fetch('/version/check', { method: 'POST' });
    const data = await r.json();
    if (data && data.state) _versionState = data.state;
  } catch (e) {
    // Leave _versionState as-is so the pill returns to the prior render
    // instead of flashing to "unknown".
  }
  pill.classList.remove('pill-busy');
  renderVersionPill();
}

async function versionDoPull() {
  const s = _versionState || {};
  const target = _versionRemoteLabel(s);
  const local = _versionDisplayLabel(s);
  const ok = confirm(
    `Pull update from ${local} → ${target}?\n\n` +
    `This runs git pull on your phosphene install. After it succeeds, ` +
    `you'll need to click Stop, then Start in Pinokio to load the new code. ` +
    `Your queue and settings are preserved across restarts.`
  );
  if (!ok) return;
  const pill = document.getElementById('versionPill');
  pill.classList.add('pill-busy');
  pill.textContent = '⟳ pulling…';
  try {
    const r = await fetch('/version/pull', { method: 'POST' });
    const data = await r.json();
    if (data && data.state) _versionState = data.state;
    if (!r.ok || !data.ok) {
      pill.classList.remove('pill-busy');
      renderVersionPill();
      alert(`Pull failed:\n\n${(data && data.error) || 'unknown error'}\n\n` +
            `Tip: try the full Pinokio Update button instead — it also handles ` +
            `cases where you have local changes that block a fast-forward.`);
      return;
    }
    _versionRestartPending = true;
    pill.classList.remove('pill-busy');
    renderVersionPill();
    const newVersion = (data.state && (data.state.pull_pulled_to_version || data.state.pull_pulled_to_short)) || 'new version';
    const fullUpdateNote = data.state && data.state.pull_requires_full_update
      ? `\n\n⚠ This update touched Python dependencies / patches. Use ` +
        `Pinokio's Update button (not just Stop+Start) so deps reinstall.`
      : '';
    alert(`Pulled to ${newVersion}.\n\nClick Stop, then Start in Pinokio to apply.${fullUpdateNote}`);
  } catch (e) {
    pill.classList.remove('pill-busy');
    renderVersionPill();
    alert(`Pull failed: ${e.message || e}`);
  }
}

// Boot: first /version read happens 2 seconds after DOM ready (gives the
// panel's startup-delay thread time to complete its first remote check),
// then every 5 minutes thereafter.
setTimeout(refreshVersionPill, 2000);
setInterval(refreshVersionPill, 5 * 60 * 1000);

// ====== Init ======
setInterval(poll, 1500);
poll();
setMode('t2v');
setAspect('landscape');         // sets aspect first so the default preset orients correctly
setQuality('balanced');         // bundles quality + dims; respects current aspect
applyTierTimes();               // rewrite Quality pill subtitles to match this Mac
updateCustomizeSummary();
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


// ============================================================================
// AGENTIC FLOWS — chat UI (rich version)
// ============================================================================
//
// Pieces:
//   - Workflow tab switch (Manual / Agentic Flows).
//   - Chat with avatar bubbles, markdown rendering, expandable tool cards.
//   - Composer with auto-resize textarea + circular send button (disabled
//     when empty, Cmd/Ctrl+Enter to fire).
//   - Sessions dropdown (from the header session-title click).
//   - Engine settings drawer (modal).
//
// Server-authoritative: after every user message we re-render the whole
// thread from /agent/sessions/<id>'s `rendered_messages` payload. The
// optimistic user bubble + typing row are inserted immediately for snap.

window.AGENT = {
  sessionId: null,
  config: null,
  busy: false,
  models: [],
  sessions: [],            // cached list for the dropdown
  selectedAnchors: {},     // {shot_label: candidate_obj} — synced from session.tool_state
  imageConfig: null,       // {kind, has_bfl_api_key, ...}
};

function workflowSwitch(name) {
  document.querySelectorAll('#workflowTabs button[data-workflow]')
    .forEach(b => b.classList.toggle('active', b.dataset.workflow === name));
  const manual = document.getElementById('genForm');
  const agent = document.getElementById('agentPane');
  // Set body data attribute so CSS can switch the layout (wider form-pane,
  // show agent-stage-pane on the right).
  document.body.setAttribute('data-workflow', name);
  if (name === 'agent') {
    if (manual) manual.style.display = 'none';
    if (agent) agent.hidden = false;
    agentRefreshConfig();
    if (!window.AGENT.sessionId) {
      const stored = localStorage.getItem('phos_agent_session');
      if (stored) agentLoadSession(stored);
      else agentLoadMostRecent();
    } else {
      agentLoadSession(window.AGENT.sessionId);
    }
    agentStageStart();
    setTimeout(() => {
      const ta = document.getElementById('agentInput');
      if (ta) { agentAutoResize(ta); ta.focus(); }
    }, 50);
  } else {
    if (manual) manual.style.display = '';
    if (agent) agent.hidden = true;
    agentStageStop();
  }
  try { localStorage.setItem('phos_workflow', name); } catch(e) {}
}

document.querySelectorAll('#workflowTabs button[data-workflow]').forEach(b => {
  b.addEventListener('click', () => workflowSwitch(b.dataset.workflow));
});

// ---- Engine status (in the header pill) -----------------------------------
async function agentRefreshConfig() {
  try {
    const r = await fetch('/agent/config');
    const j = await r.json();
    window.AGENT.config = j;
    window.AGENT.models = j.available_models || [];
    const eng = j.engine || {};
    const local = j.local_server || {};
    const dot = document.getElementById('agentEngineDot');
    const label = document.getElementById('agentEngineLabel');
    let live = false;
    let summary = '';
    if (eng.kind === 'phosphene_local') {
      live = !!local.running;
      const modelName = (eng.model || '').replace('-it-4bit', '').replace(/^gemma-3-/, 'Gemma 3 ').replace(/(\d+)b/, '$1B');
      summary = live
        ? `${modelName || 'Local'} · live`
        : `${modelName || 'Local'} · click to start`;
    } else {
      const u = (eng.base_url || '').replace(/^https?:\/\//, '').replace(/\/v1$/, '');
      summary = `${eng.model || 'remote'} · ${u}`;
      live = !!eng.has_api_key;
    }
    if (dot) {
      dot.classList.remove('live', 'warn', 'bad');
      dot.classList.add(live ? 'live' : 'warn');
    }
    if (label) label.textContent = summary;
  } catch (e) {
    const label = document.getElementById('agentEngineLabel');
    if (label) label.textContent = 'engine unavailable';
  }
}

// ---- Sessions -------------------------------------------------------------
async function agentNewSession(initialMessage) {
  const title = (initialMessage || '').slice(0, 60) || 'New chat';
  const r = await fetch('/agent/sessions/new', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({title}),
  });
  const j = await r.json();
  if (!j.ok) {
    alert('Could not create session: ' + (j.error || 'unknown'));
    return null;
  }
  window.AGENT.sessionId = j.session.session_id;
  try { localStorage.setItem('phos_agent_session', j.session.session_id); } catch(e) {}
  agentSetSessionTitle(j.session.title || 'New chat', j.session.session_id);
  agentRender([]);
  return j.session;
}

async function agentLoadMostRecent() {
  try {
    const r = await fetch('/agent/sessions');
    const j = await r.json();
    const sessions = j.sessions || [];
    window.AGENT.sessions = sessions;
    if (sessions.length === 0) {
      agentRender([]);
      return;
    }
    const mostRecent = sessions[0];
    await agentLoadSession(mostRecent.session_id);
    try { localStorage.setItem('phos_agent_session', mostRecent.session_id); } catch(e) {}
  } catch (e) { console.warn('agentLoadMostRecent', e); }
}

async function agentLoadSession(sid) {
  try {
    const r = await fetch('/agent/sessions/' + encodeURIComponent(sid));
    if (!r.ok) {
      try { localStorage.removeItem('phos_agent_session'); } catch(e) {}
      window.AGENT.sessionId = null;
      window.AGENT.selectedAnchors = {};
      agentSetSessionTitle('New chat', null);
      agentRender([]);
      return;
    }
    const j = await r.json();
    window.AGENT.sessionId = sid;
    const sess = j.session || {};
    window.AGENT.selectedAnchors = (sess.tool_state || {}).selected_anchors || {};
    agentSetSessionTitle(sess.title || 'Untitled', sid);
    agentRender(j.rendered_messages || []);
  } catch (e) { console.error('agentLoadSession', e); }
}

function agentSetSessionTitle(title, sid) {
  const el = document.getElementById('agentSessionTitle');
  if (!el) return;
  el.innerHTML = '';
  const t = document.createElement('span');
  t.textContent = title;
  el.appendChild(t);
  if (sid) {
    const m = document.createElement('span');
    m.className = 'meta';
    m.textContent = '· ' + sid.slice(0, 8);
    el.appendChild(m);
  }
}

async function agentToggleSessionsPop() {
  const pop = document.getElementById('agentSessionsPop');
  if (!pop) return;
  if (pop.classList.contains('open')) { pop.classList.remove('open'); return; }
  // Refresh list from server
  try {
    const r = await fetch('/agent/sessions');
    const j = await r.json();
    window.AGENT.sessions = j.sessions || [];
  } catch(e) {}
  agentRenderSessionsPop();
  pop.classList.add('open');
  // Click-away closes
  setTimeout(() => {
    const closer = (e) => {
      if (!pop.contains(e.target) &&
          !document.getElementById('agentSessionTitle').contains(e.target)) {
        pop.classList.remove('open');
        document.removeEventListener('click', closer);
      }
    };
    document.addEventListener('click', closer);
  }, 0);
}

function agentRenderSessionsPop() {
  const pop = document.getElementById('agentSessionsPop');
  if (!pop) return;
  pop.innerHTML = '';
  const list = window.AGENT.sessions || [];
  if (list.length === 0) {
    const e = document.createElement('div');
    e.className = 'empty';
    e.textContent = 'No saved sessions yet.';
    pop.appendChild(e);
    return;
  }
  for (const s of list) {
    const item = document.createElement('div');
    item.className = 'item' + (s.session_id === window.AGENT.sessionId ? ' active' : '');
    const title = document.createElement('div');
    title.textContent = s.title || 'Untitled';
    title.style.fontWeight = '500';
    title.style.overflow = 'hidden';
    title.style.textOverflow = 'ellipsis';
    title.style.whiteSpace = 'nowrap';
    const meta = document.createElement('div');
    meta.className = 'meta';
    const when = s.updated_at ? new Date(s.updated_at * 1000).toLocaleString(undefined, {month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit'}) : '';
    meta.innerHTML = `<span>${when}</span><span>${s.messages || 0} msg · ${s.shots_submitted || 0} shots</span>`;
    item.appendChild(title);
    item.appendChild(meta);
    item.addEventListener('click', () => {
      pop.classList.remove('open');
      agentLoadSession(s.session_id);
    });
    pop.appendChild(item);
  }
}

// ---- Markdown rendering ---------------------------------------------------
// Tight subset: headers, bold, italic, inline code, code blocks, lists,
// tables, blockquotes, hr, paragraphs. Escapes HTML first; processes
// markdown on already-safe text. Code blocks are pulled out of the way
// before other regexes run, then restored.
function mdToHtml(src) {
  if (!src) return '';
  let s = String(src);
  // Strip the fenced ```action ...``` blocks we use for tool calls — they're
  // already represented as tool-call cards, no need to show the JSON twice.
  s = s.replace(/```(?:action|tool|json action|action_json)\s*\n[\s\S]*?\n```/gi, '').trim();
  // Pull out code blocks first
  const blocks = [];
  s = s.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
    blocks.push(`<pre><code>${escapeHtml(code)}</code></pre>`);
    return `CB${blocks.length - 1}`;
  });
  // Escape rest
  s = escapeHtml(s);
  // Inline code (must come before bold/italic so backticks don't get eaten)
  s = s.replace(/`([^`\n]+)`/g, (_, c) => `<code>${c}</code>`);
  // Headers (longest first)
  s = s.replace(/^#### (.+)$/gm, '<h4>$1</h4>');
  s = s.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  s = s.replace(/^## (.+)$/gm, '<h2>$1</h2>');
  s = s.replace(/^# (.+)$/gm, '<h1>$1</h1>');
  // Horizontal rule
  s = s.replace(/^---+$/gm, '<hr>');
  // Bold then italic
  s = s.replace(/\*\*([^\n*]+)\*\*/g, '<strong>$1</strong>');
  s = s.replace(/(^|[^*])\*([^\n*]+)\*(?!\*)/g, '$1<em>$2</em>');
  // Tables — block of lines starting with | ... |
  s = s.replace(/((?:^\|[^\n]*\|\n?)+)/gm, (block) => {
    const rows = block.trim().split('\n').filter(l => !/^\s*\|[\s\-:|]+\|\s*$/.test(l));
    if (rows.length < 1) return block;
    const html = rows.map((row, i) => {
      const cells = row.replace(/^\|/, '').replace(/\|$/, '').split('|').map(c => c.trim());
      const tag = i === 0 ? 'th' : 'td';
      return '<tr>' + cells.map(c => `<${tag}>${c}</${tag}>`).join('') + '</tr>';
    }).join('');
    return `<table>${html}</table>`;
  });
  // Blockquotes
  s = s.replace(/((?:^&gt; .+\n?)+)/gm, (block) => {
    const inner = block.split('\n').map(l => l.replace(/^&gt; ?/, '')).join(' ').trim();
    return `<blockquote>${inner}</blockquote>`;
  });
  // Unordered lists
  s = s.replace(/((?:^[-*] .+\n?)+)/gm, (block) => {
    const items = block.trim().split('\n').map(l => l.replace(/^[-*] /, ''));
    return '<ul>' + items.map(i => `<li>${i}</li>`).join('') + '</ul>';
  });
  // Ordered lists
  s = s.replace(/((?:^\d+\. .+\n?)+)/gm, (block) => {
    const items = block.trim().split('\n').map(l => l.replace(/^\d+\. /, ''));
    return '<ol>' + items.map(i => `<li>${i}</li>`).join('') + '</ol>';
  });
  // Wrap remaining text into paragraphs (split on blank lines)
  s = s.split(/\n\n+/).map(p => {
    p = p.trim();
    if (!p) return '';
    if (/^<(h[1-6]|ul|ol|table|pre|blockquote|hr)/.test(p)) return p;
    return `<p>${p.replace(/\n/g, '<br>')}</p>`;
  }).join('\n');
  // Restore code blocks
  s = s.replace(/CB(\d+)/g, (_, i) => blocks[parseInt(i, 10)] || '');
  return s;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({
    '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'
  }[c]));
}

// ---- Chat rendering -------------------------------------------------------
function agentRender(messages) {
  const chat = document.getElementById('agentChat');
  if (!chat) return;
  chat.innerHTML = '';
  if (!messages || messages.length === 0) {
    chat.appendChild(renderEmpty());
    return;
  }
  for (const m of messages) chat.appendChild(renderMessage(m));
  // Hand-off the scroll on next animation frame so freshly-inserted nodes
  // have measured heights.
  requestAnimationFrame(() => { chat.scrollTop = chat.scrollHeight; });
}

function renderEmpty() {
  const wrap = document.createElement('div');
  wrap.className = 'agent-empty';
  wrap.innerHTML = `
    <div class="badge">Beta · Phosphene Agentic Flows</div>
    <h3>Plan a film overnight</h3>
    <p>Paste a script or describe a piece. I'll plan the shots, estimate the wall time, and queue the renders.</p>
    <p>You wake up to mp4s and a manifest.json.</p>
    <div class="examples">
      <div class="example" data-prompt="I want to make a short movie. Help me plan it shot by shot, then we'll generate it together.">
        <span>I want to make a short movie</span>
        <span class="arrow">→</span>
      </div>
      <div class="example" data-prompt="I want to make a 30-minute video. Walk me through how to break it into renderable shots given LTX 2.3's per-clip limits.">
        <span>I want to make a 30-minute video</span>
        <span class="arrow">→</span>
      </div>
      <div class="example" data-prompt="I want to make clips for my existing project. Ask me about the project first, then we'll plan the next batch of shots together.">
        <span>I want to make clips for my existing project</span>
        <span class="arrow">→</span>
      </div>
    </div>
  `;
  wrap.querySelectorAll('.example').forEach(b => {
    b.addEventListener('click', () => {
      const ta = document.getElementById('agentInput');
      if (!ta) return;
      ta.value = b.dataset.prompt;
      agentAutoResize(ta);
      agentUpdateSendState();
      ta.focus();
    });
  });
  return wrap;
}

function renderMessage(m) {
  if (m.kind === 'tool_result') return renderToolResultCard(m.result || {});
  if (m.kind === 'system_note') return renderSystemNote(m.content || '');

  const row = document.createElement('div');
  row.className = 'agent-msg-row';

  const av = document.createElement('div');
  av.className = `agent-avatar ${m.kind === 'user' ? 'user' : 'claude'}`;
  av.textContent = m.kind === 'user' ? 'U' : 'C';

  const body = document.createElement('div');
  body.className = 'agent-msg-body';

  const name = document.createElement('div');
  name.className = 'agent-msg-name';
  name.textContent = m.kind === 'user' ? 'You' : 'Claude';

  const content = document.createElement('div');
  content.className = 'agent-msg-content agent-md';
  content.innerHTML = mdToHtml(m.content || '');

  body.appendChild(name);
  body.appendChild(content);

  if (m.tool_call) body.appendChild(renderToolCallCard(m.tool_call));

  row.appendChild(av);
  row.appendChild(body);
  return row;
}

function renderSystemNote(text) {
  const div = document.createElement('div');
  div.style.cssText = 'text-align:center; padding:8px; font-size:11px; color:var(--muted); font-style:italic;';
  div.textContent = text;
  return div;
}

function renderToolCallCard(call) {
  const card = document.createElement('div');
  card.className = 'agent-tool-card pending';
  const head = document.createElement('div');
  head.className = 'head';
  const summary = summarizeToolCall(call);
  head.innerHTML = `
    <span class="icon">⚙</span>
    <span class="name">${escapeHtml(call.tool || '?')}</span>
    <span class="summary">${escapeHtml(summary)}</span>
    <span class="chevron">›</span>
  `;
  const body = document.createElement('div');
  body.className = 'body';
  const pre = document.createElement('pre');
  try { pre.textContent = JSON.stringify(call.args || {}, null, 2); }
  catch(e) { pre.textContent = String(call.args); }
  body.appendChild(pre);
  card.appendChild(head);
  card.appendChild(body);
  head.addEventListener('click', () => card.classList.toggle('open'));
  return card;
}

function renderToolResultCard(result) {
  const card = document.createElement('div');
  const ok = result.ok !== false && !result.error;
  card.className = 'agent-tool-card ' + (ok ? 'success' : 'error');
  const head = document.createElement('div');
  head.className = 'head';
  const inner = result.result;
  const summary = ok ? summarizeToolResult(inner) : (result.error || 'failed');
  head.innerHTML = `
    <span class="icon ${ok ? 'success' : 'error'}">${ok ? '✓' : '✗'}</span>
    <span class="name">${ok ? 'result' : 'error'}</span>
    <span class="summary">${escapeHtml(summary)}</span>
    <span class="chevron">›</span>
  `;

  const body = document.createElement('div');
  body.className = 'body';
  const pre = document.createElement('pre');
  try { pre.textContent = JSON.stringify(ok ? inner : result, null, 2); }
  catch(e) { pre.textContent = String(result); }
  body.appendChild(pre);

  card.appendChild(head);
  card.appendChild(body);
  head.addEventListener('click', () => card.classList.toggle('open'));

  // Phase B of the director workflow: when the result carries
  // `candidates`, render an interactive thumbnail grid below the head.
  // The card stays expanded by default so the user can immediately pick.
  if (ok && inner && Array.isArray(inner.candidates) && inner.candidates.length > 0) {
    card.classList.add('open');                    // open by default
    const grid = renderAnchorGrid(inner);
    card.appendChild(grid);
  }
  return card;
}

function renderAnchorGrid(payload) {
  const wrap = document.createElement('div');
  wrap.className = 'anchor-grid-wrap';
  const label = payload.shot_label || 'shot';
  const prompt = payload.prompt || '';
  const engine = payload.engine || '';

  const meta = document.createElement('div');
  meta.className = 'anchor-grid-meta';
  meta.innerHTML = `
    <span class="label-pill">${escapeHtml(label)}</span>
    <span>${payload.candidates.length} candidates · ${escapeHtml(engine)}</span>
    <span style="flex:1"></span>
    <span style="font-size:10px">click to pick</span>
  `;
  wrap.appendChild(meta);

  if (prompt) {
    const p = document.createElement('div');
    p.className = 'anchor-prompt';
    p.textContent = prompt;
    wrap.appendChild(p);
  }

  const grid = document.createElement('div');
  grid.className = 'anchor-grid';

  const selected = window.AGENT.selectedAnchors || {};
  const currentPick = (selected[label] || {}).png_path;

  for (const cand of payload.candidates) {
    const cell = document.createElement('button');
    cell.type = 'button';
    cell.className = 'anchor-cell' + (cand.png_path === currentPick ? ' selected' : '');
    cell.dataset.shotLabel = label;
    cell.dataset.pngPath = cand.png_path;

    const img = document.createElement('img');
    img.src = '/image?path=' + encodeURIComponent(cand.png_path);
    img.alt = label + ' candidate';
    img.loading = 'lazy';
    cell.appendChild(img);

    const check = document.createElement('span');
    check.className = 'check';
    check.textContent = '✓';
    cell.appendChild(check);

    if (typeof cand.seed === 'number' && cand.seed >= 0) {
      const s = document.createElement('span');
      s.className = 'seed';
      s.textContent = 'seed ' + cand.seed;
      cell.appendChild(s);
    }
    if (cand.engine) {
      const e = document.createElement('span');
      e.className = 'engine-tag';
      e.textContent = cand.engine;
      cell.appendChild(e);
    }

    cell.addEventListener('click', () => agentPickAnchor(label, cand, grid));
    grid.appendChild(cell);
  }

  wrap.appendChild(grid);
  return wrap;
}

async function agentPickAnchor(label, cand, gridEl) {
  if (!window.AGENT.sessionId) return;
  // Optimistic UI: mark this cell selected, deselect siblings
  if (gridEl) {
    gridEl.querySelectorAll('.anchor-cell').forEach(c => c.classList.remove('selected'));
    const me = gridEl.querySelector(`[data-png-path="${CSS.escape(cand.png_path)}"]`);
    if (me) me.classList.add('selected');
  }
  window.AGENT.selectedAnchors[label] = cand;

  try {
    const r = await fetch(
      '/agent/sessions/' + encodeURIComponent(window.AGENT.sessionId) + '/anchors/select',
      {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({shot_label: label, png_path: cand.png_path}),
      }
    );
    const j = await r.json();
    if (!r.ok || j.error) {
      alert('Could not save selection: ' + (j.error || ('HTTP ' + r.status)));
      return;
    }
    window.AGENT.selectedAnchors = j.all_selected || window.AGENT.selectedAnchors;
  } catch (e) {
    console.error('agentPickAnchor', e);
  }
}

function summarizeToolCall(call) {
  const t = call.tool || '';
  const a = call.args || {};
  if (t === 'submit_shot') {
    return `${a.label || a.preset_label || 'unnamed'} — ${a.duration_seconds || '?'}s ${a.mode || 't2v'} ${a.quality || 'balanced'}`;
  }
  if (t === 'estimate_shot') {
    return `${a.duration_seconds || '?'}s ${a.mode || 't2v'} ${a.quality || 'balanced'} ${a.accel || ''}`;
  }
  if (t === 'extract_frame') {
    return `${a.which || 'last'} of ${(a.job_id || '').slice(0, 12)}…`;
  }
  if (t === 'wait_for_shot') {
    return (a.job_id || '').slice(0, 14) + '…';
  }
  if (t === 'get_queue_status') return 'queue snapshot';
  if (t === 'write_session_manifest') {
    return a.title || 'manifest';
  }
  if (t === 'finish') return a.summary ? a.summary.slice(0, 80) : 'done';
  if (t === 'upload_image') return (a.attachment_id || '').split('/').pop() || '';
  return Object.keys(a).slice(0, 3).join(', ');
}

function summarizeToolResult(inner) {
  if (typeof inner !== 'object' || inner === null) return String(inner ?? '').slice(0, 100);
  if ('job_id' in inner && 'estimated_wall_human' in inner) {
    return `queued ${inner.job_id} · ETA ${inner.estimated_wall_human}`;
  }
  if ('manifest_path' in inner) {
    return `manifest written · ${inner.shot_count || '?'} shots`;
  }
  if ('estimate_wall_human' in inner) return `ETA ${inner.estimate_wall_human}`;
  if ('png_path' in inner) {
    return `frame ${inner.frame_index} → ${(inner.png_path || '').split('/').pop()}`;
  }
  if ('summary' in inner) return inner.summary.slice(0, 100);
  if ('queue_depth' in inner) {
    return `queue ${inner.queue_depth}, total ${inner.total_estimated_wall_human || '?'}`;
  }
  if ('status' in inner && 'output_path' in inner) {
    return `${inner.status} · ${(inner.output_path || '').split('/').pop() || '-'}`;
  }
  if ('absolute_path' in inner) {
    return (inner.name || inner.absolute_path);
  }
  // Fallback: show top 2 keys
  return Object.entries(inner).slice(0, 2)
    .map(([k, v]) => `${k}=${typeof v === 'object' ? '...' : String(v).slice(0, 40)}`).join(', ');
}

function renderTypingRow(msg) {
  const row = document.createElement('div');
  row.className = 'agent-typing-row';
  row.id = 'agentTypingRow';
  const av = document.createElement('div');
  av.className = 'agent-avatar claude';
  av.textContent = 'C';
  const bubble = document.createElement('div');
  bubble.className = 'agent-typing-bubble';
  bubble.innerHTML = `
    <span class="agent-typing-dots">
      <span class="agent-typing-dot"></span>
      <span class="agent-typing-dot"></span>
      <span class="agent-typing-dot"></span>
    </span>
    <span id="agentTypingText">${escapeHtml(msg || 'Thinking')}</span>
  `;
  row.appendChild(av);
  row.appendChild(bubble);
  return row;
}

// ---- Send -----------------------------------------------------------------
async function agentSend() {
  if (window.AGENT.busy) return;
  const input = document.getElementById('agentInput');
  const btn = document.getElementById('agentSendBtn');
  const text = (input.value || '').trim();
  if (!text) return;

  if (!window.AGENT.sessionId) {
    const sess = await agentNewSession(text);
    if (!sess) return;
  }

  // If a Refine reference is set, prepend "Refine <jobid> (<label>): " to
  // the user's message so the agent picks it up as a variation request.
  // Clear the chip after so the next message is a normal one.
  let outgoing = text;
  if (window.AGENT_REFINE) {
    const r = window.AGENT_REFINE;
    const ref = r.jobId || r.clipPath;
    const lbl = r.label ? ` (${r.label})` : '';
    outgoing = `Refine ${ref}${lbl}: ${text}`;
    agentClearRefine();
  }

  const chat = document.getElementById('agentChat');
  // Clear empty-state if present, then append user bubble + typing
  const empty = chat.querySelector('.agent-empty');
  if (empty) empty.remove();
  chat.appendChild(renderMessage({kind: 'user', content: outgoing}));
  chat.appendChild(renderTypingRow('Drafting plan'));
  chat.scrollTop = chat.scrollHeight;

  input.value = '';
  agentAutoResize(input);
  agentUpdateSendState();
  window.AGENT.busy = true;
  btn.disabled = true;

  // Streaming-feel: while the message round-trip is in flight (which can
  // take minutes when the agent makes many tool calls on a local model),
  // poll the server-side session every 2 s and re-render as new messages
  // land. The typing indicator stays visible until the round-trip
  // completes (since `busy` doesn't flip until then).
  const sid = window.AGENT.sessionId;
  let poller = setInterval(async () => {
    if (!window.AGENT.busy) return;
    try {
      const sr = await fetch('/agent/sessions/' + encodeURIComponent(sid));
      if (!sr.ok) return;
      const sj = await sr.json();
      const msgs = sj.rendered_messages || [];
      // Only re-render if the message count actually grew — avoids
      // flickery rebuilds during lulls.
      const cur = chat.querySelectorAll('.agent-msg-row, .agent-tool-card').length;
      if (msgs.length > cur) {
        // Snapshot the typing row's text to preserve any contextual update,
        // then rebuild and re-append it so the indicator stays at the
        // bottom under the latest content.
        const typingTextEl = document.getElementById('agentTypingText');
        const phase = typingTextEl ? typingTextEl.textContent : 'Working';
        agentRender(msgs);
        chat.appendChild(renderTypingRow(_phaseFor(msgs, phase)));
        chat.scrollTop = chat.scrollHeight;
      }
    } catch(e) {}
  }, 2000);

  try {
    const r = await fetch(
      '/agent/sessions/' + encodeURIComponent(sid) + '/message',
      {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({message: outgoing}),
      }
    );
    const j = await r.json();
    if (!r.ok || j.error) {
      const typing = document.getElementById('agentTypingRow');
      if (typing) {
        typing.querySelector('.agent-typing-bubble').innerHTML =
          `<span style="color:#f49a9e">⚠ Error: ${escapeHtml(j.error || ('HTTP ' + r.status))}</span>`;
      }
    } else {
      agentRender(j.rendered_messages || []);
    }
  } catch (e) {
    const typing = document.getElementById('agentTypingRow');
    if (typing) {
      typing.querySelector('.agent-typing-bubble').innerHTML =
        `<span style="color:#f49a9e">⚠ Network error: ${escapeHtml(String(e))}</span>`;
    }
  } finally {
    clearInterval(poller);
    window.AGENT.busy = false;
    btn.disabled = false;
    // The auto-start backend logic may have just spawned the local
    // engine — the engine pill in the header was 'click to start'
    // when the user hit Send. Now it should read 'live'. Refresh
    // the config view so the pill catches up.
    agentRefreshConfig();
    requestAnimationFrame(() => { chat.scrollTop = chat.scrollHeight; });
  }
}

// Pick a contextual typing-indicator phrase based on what just happened.
function _phaseFor(messages, fallback) {
  if (!messages || messages.length === 0) return fallback || 'Drafting plan';
  const last = messages[messages.length - 1];
  if (last.kind === 'tool_result') {
    const r = last.result || {};
    const inner = r.result || {};
    if (typeof inner === 'object' && 'job_id' in inner) {
      return `Queueing next shot…`;
    }
    return 'Reading tool result…';
  }
  if (last.kind === 'assistant') {
    if (last.tool_call) return `Calling ${last.tool_call.tool}…`;
    return 'Drafting next step…';
  }
  return fallback || 'Working';
}

// ---- Composer plumbing ----------------------------------------------------
function agentAutoResize(ta) {
  if (!ta) return;
  ta.style.height = 'auto';
  ta.style.height = Math.min(220, Math.max(48, ta.scrollHeight)) + 'px';
}

function agentUpdateSendState() {
  const input = document.getElementById('agentInput');
  const btn = document.getElementById('agentSendBtn');
  if (!input || !btn) return;
  btn.disabled = !input.value.trim() || window.AGENT.busy;
}

document.addEventListener('DOMContentLoaded', () => {
  const ta = document.getElementById('agentInput');
  if (ta) {
    ta.addEventListener('input', () => {
      agentAutoResize(ta);
      agentUpdateSendState();
    });
    ta.addEventListener('keydown', e => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
        e.preventDefault();
        agentSend();
      }
    });
    agentAutoResize(ta);
    agentUpdateSendState();
  }
});

// ---- Engine settings drawer -----------------------------------------------
async function agentRefreshImageConfig() {
  try {
    const r = await fetch('/agent/image/config');
    const j = await r.json();
    window.AGENT.imageConfig = j;
    return j;
  } catch (e) {
    return null;
  }
}

function openAgentSettings() {
  Promise.all([agentRefreshConfig(), agentRefreshImageConfig()]).then(() => {
    const modal = document.getElementById('agentSettingsModal');
    if (!modal) return;
    const cfg = (window.AGENT.config && window.AGENT.config.engine) || {};
    const local = (window.AGENT.config && window.AGENT.config.local_server) || {};
    const imgCfg = (window.AGENT.imageConfig && window.AGENT.imageConfig.image_engine) || {};
    document.getElementById('agentKind').value = cfg.kind || 'phosphene_local';
    document.getElementById('agentBaseUrl').value = cfg.base_url || '';
    document.getElementById('agentRemoteModel').value = cfg.kind === 'custom' ? (cfg.model || '') : '';
    if ((cfg.kind || 'phosphene_local') === 'ollama') agentOllamaRefresh();
    document.getElementById('agentApiKey').value = '';
    document.getElementById('agentApiKey').placeholder =
      cfg.has_api_key ? '(saved key — leave blank to keep)' : 'Paste API key';
    document.getElementById('agentTemp').value = cfg.temperature ?? 0.4;
    document.getElementById('agentMaxTokens').value = cfg.max_tokens ?? 3072;

    // Image-engine fields
    document.getElementById('agentImageKind').value = imgCfg.kind || 'mock';
    // mflux
    const namedMfluxModels = ['krea-dev', 'dev', 'schnell'];
    const mfModel = imgCfg.mflux_model || 'krea-dev';
    if (namedMfluxModels.includes(mfModel)) {
      document.getElementById('agentMfluxModel').value = mfModel;
      document.getElementById('agentMfluxCustomPath').value = '';
    } else {
      document.getElementById('agentMfluxModel').value = '__custom__';
      document.getElementById('agentMfluxCustomPath').value = mfModel;
    }
    document.getElementById('agentMfluxBaseModel').value = imgCfg.mflux_base_model || 'krea-dev';
    document.getElementById('agentMfluxSteps').value = imgCfg.mflux_steps || 25;
    document.getElementById('agentMfluxQuantize').value = String(imgCfg.mflux_quantize || 4);
    // BFL
    document.getElementById('agentBflModel').value = imgCfg.bfl_model || 'flux-dev';
    document.getElementById('agentBflKey').value = '';
    document.getElementById('agentBflKey').placeholder =
      imgCfg.has_bfl_api_key ? '(saved key — leave blank to keep)' : 'Paste BFL API key';
    const imgPill = document.getElementById('agentImagePill');
    if (imgPill) {
      const okMsg = window.AGENT.imageConfig || {};
      imgPill.textContent = okMsg.ok === false ? 'needs config' : (imgCfg.kind || 'mock');
      imgPill.style.color = okMsg.ok === false ? '#f49a9e' : '#9be7a4';
      imgPill.style.borderColor = okMsg.ok === false ? 'rgba(207,34,46,0.5)' : 'rgba(46,160,67,0.5)';
      imgPill.title = okMsg.message || '';
    }
    agentImageKindChanged();

    const sel = document.getElementById('agentLocalModel');
    sel.innerHTML = '';
    const models = (window.AGENT.config && window.AGENT.config.available_models) || [];
    if (models.length === 0) {
      const opt = document.createElement('option');
      opt.value = '';
      opt.textContent = 'No chat-capable models found in mlx_models/';
      sel.appendChild(opt);
    } else {
      for (const m of models) {
        const opt = document.createElement('option');
        opt.value = m.path;
        opt.textContent = `${m.name} · ${m.size_gb} GB`;
        if ((cfg.local_model_path || '') === m.path) opt.selected = true;
        sel.appendChild(opt);
      }
    }
    agentKindChanged();
    agentLocalRefreshRow(local);
    modal.classList.add('open');
  });
}

function closeAgentSettings() {
  document.getElementById('agentSettingsModal').classList.remove('open');
}

function agentKindChanged() {
  const kind = document.getElementById('agentKind').value;
  document.getElementById('agentLocalModelField').style.display = kind === 'phosphene_local' ? '' : 'none';
  document.getElementById('agentLocalRow').style.display = kind === 'phosphene_local' ? '' : 'none';
  document.getElementById('agentOllamaField').style.display = kind === 'ollama' ? '' : 'none';
  document.getElementById('agentBaseUrlField').style.display = kind === 'custom' ? '' : 'none';
  document.getElementById('agentApiKeyField').style.display = kind === 'custom' ? '' : 'none';
  document.getElementById('agentRemoteModelField').style.display = kind === 'custom' ? '' : 'none';
  if (kind === 'ollama') agentOllamaRefresh();
}

async function agentOllamaRefresh() {
  const sel = document.getElementById('agentOllamaModel');
  const hint = document.getElementById('agentOllamaHint');
  if (!sel) return;
  sel.innerHTML = '<option>Probing Ollama at 127.0.0.1:11434…</option>';
  try {
    const r = await fetch('/agent/ollama/status');
    const j = await r.json();
    sel.innerHTML = '';
    if (!j.running) {
      const opt = document.createElement('option');
      opt.value = '';
      opt.textContent = 'Ollama not running on 127.0.0.1:11434';
      sel.appendChild(opt);
      if (hint) hint.innerHTML = `<strong>Ollama is not running.</strong> Start it with <code>ollama serve</code>, then click Refresh. Install models with <code>ollama pull qwen2.5-coder:32b</code>.`;
      return;
    }
    const cfg = (window.AGENT.config && window.AGENT.config.engine) || {};
    if ((j.models || []).length === 0) {
      const opt = document.createElement('option');
      opt.value = '';
      opt.textContent = 'No Ollama models installed';
      sel.appendChild(opt);
    }
    for (const m of (j.models || [])) {
      const opt = document.createElement('option');
      opt.value = m.name;
      opt.textContent = `${m.name}${m.size_gb ? ' · ' + m.size_gb + ' GB' : ''}${m.parameter_size ? ' · ' + m.parameter_size : ''}${m.quantization ? ' · ' + m.quantization : ''}`;
      if (cfg.kind === 'ollama' && cfg.model === m.name) opt.selected = true;
      sel.appendChild(opt);
    }
    if (hint) hint.innerHTML = `Talks to <code>${escapeHtml(j.openai_url || j.base_url + '/v1')}/chat/completions</code>. Tool calling works on models whose Modelfile declares it (Qwen3 Coder, Llama 3.x, Devstral, Mistral, Granite). Run <code>ollama show &lt;model&gt;</code> to verify.`;
  } catch(e) {
    sel.innerHTML = '<option>Failed to probe — see console</option>';
  }
}

function agentImageKindChanged() {
  const kind = document.getElementById('agentImageKind').value;
  // mflux fields
  const isMflux = kind === 'mflux';
  document.getElementById('agentMfluxModelField').style.display = isMflux ? '' : 'none';
  document.getElementById('agentMfluxParamsField').style.display = isMflux ? '' : 'none';
  document.getElementById('agentMfluxInstallHint').style.display = isMflux ? '' : 'none';
  if (isMflux) agentMfluxModelChanged();
  else {
    document.getElementById('agentMfluxCustomField').style.display = 'none';
    document.getElementById('agentMfluxBaseField').style.display = 'none';
  }
  // BFL fields
  document.getElementById('agentBflModelField').style.display = kind === 'bfl' ? '' : 'none';
  document.getElementById('agentBflKeyField').style.display = kind === 'bfl' ? '' : 'none';
}

function agentMfluxModelChanged() {
  const v = document.getElementById('agentMfluxModel').value;
  const isCustom = v === '__custom__';
  document.getElementById('agentMfluxCustomField').style.display = isCustom ? '' : 'none';
  document.getElementById('agentMfluxBaseField').style.display = isCustom ? '' : 'none';
  // For schnell, drop the recommended steps to 4
  const stepsInput = document.getElementById('agentMfluxSteps');
  if (stepsInput && !stepsInput.dataset.userTouched) {
    if (v === 'schnell') stepsInput.value = 4;
    else if (v === 'krea-dev' || v === 'dev') stepsInput.value = 25;
  }
}

function agentLocalRefreshRow(local) {
  const pill = document.getElementById('agentLocalPill');
  const detail = document.getElementById('agentLocalDetail');
  const btn = document.getElementById('agentLocalToggleBtn');
  if (!pill || !detail || !btn) return;
  if (local.running) {
    pill.textContent = 'live';
    pill.classList.add('live'); pill.classList.remove('bad');
    detail.textContent = `mlx-lm.server pid ${local.pid} on :${local.port}`;
    btn.textContent = 'Stop';
  } else {
    pill.textContent = 'stopped';
    pill.classList.remove('live');
    if (local.last_error) pill.classList.add('bad');
    detail.textContent = local.last_error || 'mlx-lm.server (will spawn on Start)';
    btn.textContent = 'Start';
  }
}

async function agentLocalToggle() {
  const local = (window.AGENT.config && window.AGENT.config.local_server) || {};
  if (local.running) {
    await fetch('/agent/local/stop', {method: 'POST'});
  } else {
    const sel = document.getElementById('agentLocalModel');
    const modelPath = sel ? sel.value : '';
    await fetch('/agent/local/start', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({model_path: modelPath || undefined}),
    });
  }
  await new Promise(r => setTimeout(r, 300));
  await agentRefreshConfig();
  const local2 = (window.AGENT.config && window.AGENT.config.local_server) || {};
  agentLocalRefreshRow(local2);
}

async function agentSaveSettings() {
  const kind = document.getElementById('agentKind').value;
  const payload = {
    kind,
    temperature: parseFloat(document.getElementById('agentTemp').value || '0.4'),
    max_tokens: parseInt(document.getElementById('agentMaxTokens').value || '3072', 10),
  };
  if (kind === 'phosphene_local') {
    const sel = document.getElementById('agentLocalModel');
    if (sel && sel.value) {
      payload.local_model_path = sel.value;
      const parts = sel.value.split('/');
      payload.model = parts[parts.length - 1] || sel.value;
    }
  } else if (kind === 'ollama') {
    // Ollama bridge: same OpenAI-compat shape, just talks to 127.0.0.1:11434/v1.
    // No api_key. The model field is the Ollama tag (e.g. "qwen2.5-coder:32b").
    const sel = document.getElementById('agentOllamaModel');
    payload.base_url = 'http://127.0.0.1:11434/v1';
    payload.model = sel ? sel.value : '';
    payload.local_model_path = '';
  } else {
    payload.base_url = (document.getElementById('agentBaseUrl').value || '').trim();
    payload.model = (document.getElementById('agentRemoteModel').value || '').trim();
    const ak = (document.getElementById('agentApiKey').value || '').trim();
    if (ak) payload.api_key = ak;
  }
  const r = await fetch('/agent/config', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload),
  });
  const j = await r.json();
  if (!r.ok || j.error) {
    alert('Could not save: ' + (j.error || ('HTTP ' + r.status)));
    return;
  }

  // Save image-engine config too (separate file).
  const imgKind = document.getElementById('agentImageKind').value;
  const imgPayload = {
    kind: imgKind,
    bfl_model: document.getElementById('agentBflModel').value,
  };
  const bk = (document.getElementById('agentBflKey').value || '').trim();
  if (bk) imgPayload.bfl_api_key = bk;
  // mflux fields (only meaningful when kind === 'mflux', but we save them
  // either way so the form retains the user's previous setup when they
  // toggle backends back and forth).
  const mfSel = document.getElementById('agentMfluxModel').value;
  if (mfSel === '__custom__') {
    const cp = (document.getElementById('agentMfluxCustomPath').value || '').trim();
    if (cp) {
      imgPayload.mflux_model = cp;
      imgPayload.mflux_base_model = document.getElementById('agentMfluxBaseModel').value;
    }
  } else {
    imgPayload.mflux_model = mfSel;
    imgPayload.mflux_base_model = '';
  }
  imgPayload.mflux_steps = parseInt(document.getElementById('agentMfluxSteps').value || '25', 10);
  imgPayload.mflux_quantize = parseInt(document.getElementById('agentMfluxQuantize').value || '4', 10);
  try {
    const ir = await fetch('/agent/image/config', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(imgPayload),
    });
    const ij = await ir.json();
    if (!ir.ok || ij.error) {
      alert('Image config could not be saved: ' + (ij.error || ('HTTP ' + ir.status)));
    }
  } catch(e) {
    alert('Image config save failed: ' + e);
  }

  closeAgentSettings();
  await agentRefreshConfig();
  await agentRefreshImageConfig();
}

// ============================================================================
// HF MODEL BROWSER — search + install MLX chat models
// ============================================================================
window.MODEL_BROWSER = {
  pollerId: null,
  lastResults: [],
};

function openModelBrowser() {
  const m = document.getElementById('modelBrowserModal');
  if (!m) return;
  m.classList.add('open');
  // Auto-focus the query input on open.
  setTimeout(() => {
    const q = document.getElementById('modelBrowserQuery');
    if (q) q.focus();
  }, 50);
  // If we already have results from a previous search, leave them; else
  // run a default search ("qwen") so the user sees something useful.
  if (!window.MODEL_BROWSER.lastResults.length) {
    document.getElementById('modelBrowserQuery').value = 'qwen';
    modelBrowserSearch();
  }
  // Start polling install status in case a download is already in flight.
  modelBrowserStartPolling();
}

function closeModelBrowser() {
  document.getElementById('modelBrowserModal').classList.remove('open');
  modelBrowserStopPolling();
}

async function modelBrowserSearch() {
  const q = (document.getElementById('modelBrowserQuery').value || '').trim();
  const abliterated = document.getElementById('modelBrowserAbliterated').checked;
  const results = document.getElementById('modelBrowserResults');
  const btn = document.getElementById('modelBrowserSearchBtn');
  results.innerHTML = '<div class="model-browser-empty">Searching…</div>';
  btn.disabled = true;
  try {
    const url = '/agent/models/search?q=' + encodeURIComponent(q)
              + '&abliterated=' + (abliterated ? '1' : '0')
              + '&limit=40';
    const r = await fetch(url);
    const j = await r.json();
    if (!r.ok || j.error) {
      results.innerHTML = '<div class="model-browser-empty" style="color:#f49a9e">Error: ' + escapeHtml(j.error || ('HTTP ' + r.status)) + '</div>';
      return;
    }
    window.MODEL_BROWSER.lastResults = j.results || [];
    modelBrowserRender(j.results || []);
  } catch(e) {
    results.innerHTML = '<div class="model-browser-empty" style="color:#f49a9e">Error: ' + escapeHtml(String(e)) + '</div>';
  } finally {
    btn.disabled = false;
  }
}

function modelBrowserRender(results) {
  const wrap = document.getElementById('modelBrowserResults');
  if (!results.length) {
    wrap.innerHTML = '<div class="model-browser-empty">No matches. Try a different query.</div>';
    return;
  }
  wrap.innerHTML = '';
  for (const m of results) {
    const row = document.createElement('div');
    row.className = 'model-result';
    const isAbliterated = (m.repo_id || '').toLowerCase().includes('abliterated')
                       || (m.repo_id || '').toLowerCase().startsWith('huihui-ai/');
    const dl = (m.downloads || 0).toLocaleString();
    const lk = (m.likes || 0).toLocaleString();
    const tags = [];
    if (m.gated) tags.push('<span class="tag gated">gated</span>');
    if (isAbliterated) tags.push('<span class="tag abliterated">abliterated</span>');
    if (m.library_name) tags.push(`<span class="tag">${escapeHtml(m.library_name)}</span>`);
    if (m.pipeline_tag) tags.push(`<span class="tag">${escapeHtml(m.pipeline_tag)}</span>`);
    row.innerHTML = `
      <div class="info">
        <div class="name">${escapeHtml(m.repo_id)}</div>
        <div class="meta">
          <span>↓ ${dl}</span>
          <span>♥ ${lk}</span>
          ${tags.join('')}
        </div>
      </div>
      <div class="actions">
        <button class="info-btn" onclick="modelBrowserInfo('${escapeHtml(m.repo_id)}')">Info</button>
        <button class="install-btn" onclick="modelBrowserInstall('${escapeHtml(m.repo_id)}', this)">Install</button>
      </div>
    `;
    wrap.appendChild(row);
  }
}

async function modelBrowserInfo(repoId) {
  // Lightweight inline info — pop a confirm with size + file count + gated state.
  try {
    const r = await fetch('/agent/models/info?repo_id=' + encodeURIComponent(repoId));
    const j = await r.json();
    if (j.gated || j.error) {
      alert('Repo info:\n' + (j.error || ('Gated. Open https://huggingface.co/' + repoId + ' and accept the terms first.')));
      return;
    }
    alert(`${repoId}\n\nFiles: ${j.file_count}\nTotal: ${j.total_size_gb} GB\n\nClick Install to download into mlx_models/.`);
  } catch(e) {
    alert('Could not fetch info: ' + e);
  }
}

async function modelBrowserInstall(repoId, btn) {
  if (!confirm(`Install ${repoId}?\n\nDownloads to mlx_models/. Files are large — first run can take 5-30 min depending on size and network.`)) {
    return;
  }
  btn.disabled = true;
  btn.textContent = 'Queuing…';
  try {
    const r = await fetch('/agent/models/install', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({repo_id: repoId}),
    });
    const j = await r.json();
    if (!r.ok || j.error) {
      alert('Install failed to start: ' + (j.error || ('HTTP ' + r.status)));
      btn.disabled = false;
      btn.textContent = 'Install';
      return;
    }
    btn.textContent = 'Downloading…';
    modelBrowserStartPolling();
  } catch(e) {
    alert('Install error: ' + e);
    btn.disabled = false;
    btn.textContent = 'Install';
  }
}

function modelBrowserStartPolling() {
  modelBrowserStopPolling();
  modelBrowserPollOnce();
  window.MODEL_BROWSER.pollerId = setInterval(modelBrowserPollOnce, 1500);
}

function modelBrowserStopPolling() {
  if (window.MODEL_BROWSER.pollerId) {
    clearInterval(window.MODEL_BROWSER.pollerId);
    window.MODEL_BROWSER.pollerId = null;
  }
}

async function modelBrowserPollOnce() {
  try {
    const r = await fetch('/agent/models/install/status');
    const j = await r.json();
    const status = document.getElementById('modelBrowserStatus');
    const lbl = document.getElementById('modelBrowserStatusLabel');
    const summ = document.getElementById('modelBrowserStatusSummary');
    const line = document.getElementById('modelBrowserStatusLine');
    if (!status || !lbl) return;
    if (j.active) {
      status.classList.add('visible');
      lbl.textContent = 'Downloading';
      lbl.style.color = '';
      const elapsed = j.elapsed_s ? ` · ${Math.round(j.elapsed_s)}s` : '';
      summ.textContent = `${j.repo_id || ''}${elapsed}`;
      line.textContent = j.last_line || '';
    } else if (j.done) {
      status.classList.add('visible');
      lbl.textContent = 'Installed';
      lbl.style.color = '#9be7a4';
      summ.textContent = `${j.repo_id || ''} → ${j.target_dir || ''}`;
      line.textContent = '';
      modelBrowserStopPolling();
      // Refresh the local-model picker in the background so the user
      // sees the new model on next settings-modal open.
      try { agentRefreshConfig(); } catch(e) {}
    } else if (j.error) {
      status.classList.add('visible');
      lbl.textContent = 'Failed';
      lbl.style.color = '#f49a9e';
      summ.textContent = j.repo_id || '';
      line.textContent = j.error;
      modelBrowserStopPolling();
    } else {
      status.classList.remove('visible');
      modelBrowserStopPolling();
    }
  } catch(e) {
    /* swallow */
  }
}

// Esc closes the model browser, before falling through to settings/fullscreen.
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    const m = document.getElementById('modelBrowserModal');
    if (m && m.classList.contains('open')) {
      closeModelBrowser();
      e.stopPropagation();
    }
  }
}, true);

// Cmd/Ctrl+Enter in the search box runs the search.
document.addEventListener('DOMContentLoaded', () => {
  const q = document.getElementById('modelBrowserQuery');
  if (q) {
    q.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { e.preventDefault(); modelBrowserSearch(); }
    });
  }
});

// ============================================================================
// AGENT STAGE PANE — live canvas on the right side
// ============================================================================
// Shows the agent's video work as it happens: currently rendering job,
// session outputs (with click-to-play lightbox), and a recent-activity
// feed of tool calls. Polls /status + the active session every ~1.5 s
// while the agent workflow is selected.

window.AGENT_STAGE = {
  pollerId: null,
  lastSessionShotIds: [],   // submitted_shots from session.tool_state, ordered
  lastEventCount: 0,        // for animating new entries in the activity feed
};

function agentStageStart() {
  agentStageStop();
  // Tick once immediately, then every 1500 ms.
  agentStageTick();
  window.AGENT_STAGE.pollerId = setInterval(agentStageTick, 1500);
}

function agentStageStop() {
  if (window.AGENT_STAGE.pollerId) {
    clearInterval(window.AGENT_STAGE.pollerId);
    window.AGENT_STAGE.pollerId = null;
  }
}

async function agentStageTick() {
  const pane = document.querySelector('.agent-stage-pane');
  // Don't bother polling when the pane is hidden.
  if (!pane || getComputedStyle(pane).display === 'none') return;
  try {
    const [statusResp, sessResp] = await Promise.all([
      fetch('/status').then(r => r.ok ? r.json() : null).catch(() => null),
      window.AGENT.sessionId
        ? fetch('/agent/sessions/' + encodeURIComponent(window.AGENT.sessionId))
            .then(r => r.ok ? r.json() : null).catch(() => null)
        : null,
    ]);
    agentStageRender(statusResp, sessResp);
  } catch(e) {
    /* swallow — next tick retries */
  }
}

function agentStageRender(status, sess) {
  const dot = document.getElementById('agentStageDot');
  const sessionPill = document.getElementById('agentStageSession');
  const nowEl = document.getElementById('agentStageNow');
  const outputsEl = document.getElementById('agentStageOutputs');
  const outputsCountEl = document.getElementById('agentStageOutputsCount');
  const activityEl = document.getElementById('agentStageActivity');
  const activityCountEl = document.getElementById('agentStageActivityCount');
  if (!dot || !nowEl || !outputsEl || !activityEl) return;

  const running = !!(status && status.running);
  dot.classList.toggle('live', running);
  if (sessionPill) {
    if (window.AGENT.sessionId) {
      sessionPill.textContent = window.AGENT.sessionId.slice(0, 10);
    } else {
      sessionPill.textContent = 'no session';
    }
  }

  // Now rendering
  const cur = status && status.current;
  if (running && cur) {
    const p = cur.params || {};
    const label = p.label || (p.preset_label || cur.id || 'render');
    const progress = (cur.progress != null ? cur.progress : (status.progress || 0));
    const pct = Math.max(0, Math.min(1, Number(progress) || 0)) * 100;
    const eta = cur.eta_seconds || cur.eta || null;
    const phase = cur.phase || (cur.status === 'running' ? 'rendering' : '');
    nowEl.classList.remove('idle');
    nowEl.innerHTML = `
      <div class="stage-now-label">${escapeHtml(label)}</div>
      <div class="stage-now-meta">${escapeHtml(p.mode || 't2v')} · ${escapeHtml(p.quality || 'balanced')} · ${escapeHtml(p.frames || '?')}f</div>
      <div class="stage-progress-bar">
        <div class="stage-progress-fill" style="width:${pct.toFixed(1)}%"></div>
      </div>
      <div class="stage-progress-text">
        <span>${escapeHtml(phase || 'rendering')}</span>
        <span>${pct.toFixed(0)}%${eta ? ' · ETA ' + agentFmtDur(eta) : ''}</span>
      </div>
    `;
  } else {
    nowEl.className = 'stage-now-card idle';
    nowEl.innerHTML = `<div>Idle. Ask the agent to plan a shot to see it render here.</div>`;
  }

  // Session outputs — pull from session.tool_state.submitted_shots (ordered)
  // and look up each in /status.history for current state + output_path.
  const tool = (sess && sess.session && sess.session.tool_state) || {};
  const submitted = tool.submitted_shots || [];
  const allJobs = []
    .concat(status && status.queue ? status.queue : [])
    .concat(cur ? [cur] : [])
    .concat(status && status.history ? status.history : []);
  const byId = new Map(allJobs.map(j => [j.id, j]));
  const outputs = submitted.map(s => {
    const j = byId.get(s.job_id) || s;
    const p = (j.params || {});
    return {
      id: s.job_id,
      label: s.label || p.label || s.job_id,
      status: j.status || 'unknown',
      output_path: j.output_path || null,
      mode: p.mode || s.mode || 't2v',
      duration: s.duration_seconds || null,
    };
  });
  outputsCountEl.textContent = outputs.length;
  if (outputs.length === 0) {
    outputsEl.innerHTML = `<div class="stage-empty">No mp4s rendered yet. Submit a shot from the chat.</div>`;
  } else {
    outputsEl.innerHTML = '';
    for (const o of outputs.slice(0, 24)) {
      const cell = document.createElement('div');
      const failed = o.status === 'error' || o.status === 'failed' || o.status === 'cancelled';
      cell.className = 'stage-output-cell' + (failed ? ' failed' : '');
      cell.title = o.label + ' · ' + o.status;
      // If we have a finished output_path, show the video as a thumbnail.
      // Otherwise show a status badge.
      if (o.output_path && o.status === 'done') {
        const v = document.createElement('video');
        v.className = 'vid';
        v.src = '/file?path=' + encodeURIComponent(o.output_path);
        v.preload = 'metadata';
        v.muted = true;
        cell.appendChild(v);
        cell.addEventListener('click', () => agentStageLightboxOpen(o.output_path, o.label, o.id));
        // Refine button (overlay top-right): "give me a variation of this clip"
        const refine = document.createElement('button');
        refine.type = 'button';
        refine.className = 'refine-btn';
        refine.title = 'Refine this clip — start a variation in the chat';
        refine.textContent = '↻';
        refine.addEventListener('click', (e) => {
          e.stopPropagation();
          agentSetRefine({jobId: o.id, label: o.label, clipPath: o.output_path});
        });
        cell.appendChild(refine);
      } else {
        cell.style.display = 'flex';
        cell.style.alignItems = 'center';
        cell.style.justifyContent = 'center';
        const span = document.createElement('span');
        span.style.cssText = 'color:var(--muted); font-size:11px; font-style:italic;';
        span.textContent = failed ? 'failed' : (o.status === 'running' ? 'rendering…' : 'queued');
        cell.appendChild(span);
      }
      const badge = document.createElement('span');
      badge.className = 'badge';
      badge.textContent = failed ? 'fail' : (o.status === 'done' ? '✓' : o.status.slice(0, 4));
      cell.appendChild(badge);
      const lbl = document.createElement('span');
      lbl.className = 'label';
      lbl.textContent = o.label;
      cell.appendChild(lbl);
      outputsEl.appendChild(cell);
    }
  }

  // Activity feed — derive from rendered_messages: each tool_call/tool_result
  // becomes one row with an icon. Newest at the top.
  const rendered = (sess && sess.rendered_messages) || [];
  const events = [];
  for (let i = rendered.length - 1; i >= 0 && events.length < 40; i--) {
    const m = rendered[i];
    if (m.kind === 'tool_result') {
      const r = m.result || {};
      const ok = r.ok !== false && !r.error;
      const inner = r.result || {};
      let txt;
      if (typeof inner === 'object' && 'job_id' in inner) {
        txt = `→ queued ${inner.job_id} · ${inner.estimated_wall_human || '?'}`;
      } else if (typeof inner === 'object' && 'manifest_path' in inner) {
        txt = `→ manifest written (${inner.shot_count} shots)`;
      } else if (typeof inner === 'object' && 'png_path' in inner) {
        txt = `→ frame extracted (${inner.frame_index})`;
      } else if (typeof inner === 'object' && 'candidates' in inner) {
        txt = `→ ${inner.candidates.length} candidates ready`;
      } else if (!ok) {
        txt = `✗ ${(r.error || 'failed').slice(0, 80)}`;
      } else {
        txt = '→ result';
      }
      events.push({ kind: ok ? 'ok' : 'fail', text: txt });
    } else if (m.kind === 'assistant' && m.tool_call) {
      events.push({ kind: 'run', text: '⚙ ' + m.tool_call.tool });
    }
  }
  activityCountEl.textContent = events.length;
  if (events.length === 0) {
    activityEl.innerHTML = `<div class="stage-empty">No tool calls yet.</div>`;
  } else {
    activityEl.innerHTML = '';
    for (const ev of events) {
      const row = document.createElement('div');
      row.className = 'stage-activity-row ' + ev.kind;
      const icon = (ev.kind === 'ok') ? '✓' : (ev.kind === 'fail') ? '✗' : '⚙';
      row.innerHTML = `
        <span class="icon">${icon}</span>
        <span class="text">${escapeHtml(ev.text)}</span>
      `;
      activityEl.appendChild(row);
    }
  }
}

function agentFmtDur(seconds) {
  const s = Math.max(0, Math.round(Number(seconds) || 0));
  if (s < 60) return s + 's';
  const m = Math.floor(s / 60), rem = s % 60;
  if (m < 60) return m + 'm ' + (rem < 10 ? '0' : '') + rem + 's';
  const h = Math.floor(m / 60), mr = m % 60;
  return h + 'h ' + (mr < 10 ? '0' : '') + mr + 'm';
}

// Track which clip is currently in the lightbox so the Refine button
// has something to reference when clicked.
window.AGENT_STAGE.lightboxCurrent = null;

function agentStageLightboxOpen(path, label, jobId) {
  const lb = document.getElementById('agentStageLightbox');
  const v = document.getElementById('agentStageLightboxVideo');
  if (!lb || !v) return;
  v.src = '/file?path=' + encodeURIComponent(path);
  v.title = label || '';
  window.AGENT_STAGE.lightboxCurrent = {jobId: jobId || null, label: label || '', clipPath: path};
  lb.classList.add('open');
  v.play().catch(() => {});
}

function agentStageLightboxClose() {
  const lb = document.getElementById('agentStageLightbox');
  const v = document.getElementById('agentStageLightboxVideo');
  if (!lb || !v) return;
  v.pause();
  v.src = '';
  window.AGENT_STAGE.lightboxCurrent = null;
  lb.classList.remove('open');
}

function agentStageLightboxRefine() {
  const cur = window.AGENT_STAGE.lightboxCurrent;
  if (!cur) return;
  agentStageLightboxClose();
  agentSetRefine(cur);
}

// ---- Composer reference chip — "Refine this clip" -----------------------
// When the user clicks ↻ on a stage output (or the Refine button in the
// lightbox), we set a refine reference. The chip shows above the textarea;
// on next Send, the user's message is prepended with "Refine <job_id>: "
// so the agent calls inspect_clip and treats the rest as the requested
// modification. Clear with × on the chip.
window.AGENT_REFINE = null;          // {jobId, label, clipPath}

function agentSetRefine(ref) {
  window.AGENT_REFINE = ref;
  const chip = document.getElementById('agentRefChip');
  const lbl = document.getElementById('agentRefChipLabel');
  if (!chip || !lbl) return;
  lbl.textContent = ref.label || ref.jobId || ref.clipPath || 'clip';
  lbl.title = ref.jobId ? `${ref.label || ''} · ${ref.jobId}` : ref.clipPath;
  chip.classList.add('visible');
  // Bring focus to the composer so the user can type their refinement.
  const ta = document.getElementById('agentInput');
  if (ta) {
    ta.placeholder = 'How should this clip be different? (e.g. "more pause", "warmer light", "longer take")';
    setTimeout(() => ta.focus(), 50);
  }
}

function agentClearRefine() {
  window.AGENT_REFINE = null;
  const chip = document.getElementById('agentRefChip');
  if (chip) chip.classList.remove('visible');
  const ta = document.getElementById('agentInput');
  if (ta) ta.placeholder = 'Paste a script, or describe a piece. The agent will plan, estimate the wall time, and queue overnight.';
}

// Esc closes the stage lightbox first (before falling through to the
// settings modal / fullscreen exit handlers).
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    const lb = document.getElementById('agentStageLightbox');
    if (lb && lb.classList.contains('open')) {
      agentStageLightboxClose();
      e.stopPropagation();
    }
  }
}, true);

// ---- Pop out to system browser -------------------------------------------
// The pop-out is a real <a target="_blank"> link (set up during boot via
// agentInitPopOut). The browser handles the actual navigation as a user
// gesture so popup-blockers don't kick in. This handler just sets the
// localStorage flags so the new tab boots straight into Agentic Flows +
// fullscreen.
function agentPopOutFlagsBeforeNavigate() {
  try {
    localStorage.setItem('phos_workflow', 'agent');
    localStorage.setItem('phos_agent_fullscreen', '1');
  } catch(e) {}
  // The <a href> handles navigation. Don't preventDefault.
}

function agentInitPopOut() {
  const a = document.getElementById('agentPopOutBtn');
  if (!a) return;
  a.href = window.location.origin || ('http://127.0.0.1:' + window.location.port);
}
// Run on every page load; cheap and idempotent.
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', agentInitPopOut);
} else {
  agentInitPopOut();
}

// ---- Fullscreen / focus mode ---------------------------------------------
function agentToggleFullscreen(force) {
  // `force === true` to enter, `false` to exit, undefined to toggle
  const cur = document.body.classList.contains('agent-fullscreen');
  const next = (typeof force === 'boolean') ? force : !cur;
  document.body.classList.toggle('agent-fullscreen', next);
  const btn = document.getElementById('agentFullscreenBtn');
  if (btn) btn.title = next ? 'Exit fullscreen (Esc)' : 'Expand to fullscreen';
  const ix = document.getElementById('agentFullscreenIconExpand');
  const iy = document.getElementById('agentFullscreenIconCollapse');
  if (ix) ix.style.display = next ? 'none' : '';
  if (iy) iy.style.display = next ? '' : 'none';
  try { localStorage.setItem('phos_agent_fullscreen', next ? '1' : ''); } catch(e) {}
  // After collapse animation settles, scroll chat to bottom so the
  // user keeps their place.
  requestAnimationFrame(() => {
    const chat = document.getElementById('agentChat');
    if (chat) chat.scrollTop = chat.scrollHeight;
  });
}

// Esc handler: priority is modal-close > fullscreen-exit. So Esc to
// dismiss the settings drawer doesn't drop the user out of fullscreen
// at the same time.
document.addEventListener('keydown', (e) => {
  if (e.key !== 'Escape') return;
  const modal = document.getElementById('agentSettingsModal');
  if (modal && modal.classList.contains('open')) {
    closeAgentSettings();
    return;
  }
  if (document.body.classList.contains('agent-fullscreen')) {
    agentToggleFullscreen(false);
  }
});

// Initial workflow tab restore from localStorage.
try {
  const saved = localStorage.getItem('phos_workflow');
  if (saved === 'agent') workflowSwitch('agent');
  // Restore fullscreen state, but ONLY when the agent tab is the
  // active workflow — otherwise we'd hide the manual form too.
  if (localStorage.getItem('phos_agent_fullscreen') && saved === 'agent') {
    agentToggleFullscreen(true);
  }
} catch(e) {}
</script>
</body>
</html>

"""


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
