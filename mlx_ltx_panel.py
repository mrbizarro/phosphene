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
            raise RuntimeError("helper died mid-job (no event)")
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
                "stage1_steps": 15,
                "stage2_steps": 3,
                "cfg_scale": 3.0,
            },
        }
        push(f"Run KEYFRAME via helper: id={job['id']} {width}x{height} {frames}f · Q8 two-stage")
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
            },
        }
        push(f"Run via helper: id={job['id']} mode={mode} quality={quality} {width}x{height} {frames}f")

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
                self._json({"error": "hf binary not found. Reinstall LTX23MLX "
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
  <title>LTX23MLX Studio</title>
  <link rel="icon" type="image/png" sizes="64x64" href="/assets/favicon-64.png">
  <link rel="icon" type="image/png" sizes="256x256" href="/assets/favicon.png">
  <style>
    :root {
      --bg: #0b0e13; --bg-2: #0d1117; --panel: #161b22; --panel-2: #1c2230;
      --border: #2a3038; --border-strong: #3a424d; --text: #e6edf3; --muted: #8b949e;
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

    /* ===== HEADER ===== */
    header {
      display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
      padding: 10px 18px; border-bottom: 1px solid var(--border);
      background: var(--panel);
    }
    header h1 {
      margin: 0; font-size: 15px; font-weight: 700; letter-spacing: -0.01em;
      display: inline-flex; align-items: center; gap: 8px;
    }
    .brand { display: inline-flex; align-items: center; flex-shrink: 0; }
    .brand img { height: 40px; width: auto; display: block; }
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
      background: transparent; border: 1px solid var(--border); color: var(--text);
      padding: 5px 10px; border-radius: 6px; font-size: 11px; cursor: pointer;
    }
    .ghost-btn:hover { border-color: var(--accent); color: var(--accent-bright); }
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
    }
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
  <a href="/" class="brand"><img src="/assets/logo-header.png" alt="LTX23MLX"></a>
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
            <input id="extend_seconds" type="number" value="2" min="0.4" max="10" step="0.5">
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
      <textarea name="prompt" id="prompt" placeholder="What should happen in the video..."></textarea>

      <!-- Mode-specific: audio (i2v_clean_audio only — accessed via Advanced) -->
      <details>
        <summary>Advanced</summary>
        <label class="lbl">I2V audio mode</label>
        <select id="i2vMode">
          <option value="i2v" selected>Joint audio (LTX generates audio synced with visual)</option>
          <option value="i2v_clean_audio">Replace LTX audio with external file (mux)</option>
        </select>
        <div class="mode-only" id="audioSection">
          <label class="lbl">Audio file</label>
          <input name="audio" id="audio">
        </div>
        <label class="check">
          <input type="checkbox" name="enhance" id="enhance"> Enhance prompt (Gemma rewrite — CLI only, ignored by helper)
        </label>
        <label class="check">
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
    nowCard.classList.remove('idle');
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
    nowCard.classList.add('idle');
    fill.style.width = '0%';
    nowCard.querySelector('.ttl').textContent = s.paused ? 'Paused' : 'Idle';
    nowCard.querySelector('.meta').textContent = s.paused
      ? 'Worker paused — current job (if any) finishes, queue waits for resume.'
      : (s.queue.length ? 'Worker about to pick up next queued job.' : 'No jobs queued. Generate something on the left.');
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

  // History
  const hl = document.getElementById('historyList');
  if (!s.history.length) hl.innerHTML = '<li class="empty-state"><span></span><span>No history yet</span><span></span><span></span></li>';
  else hl.innerHTML = s.history.slice(0, 20).map(j => `
    <li class="${j.status}">
      <span class="badge">${j.status}</span>
      <span class="ttl" title="${escapeHtml(j.params.prompt)}">${escapeHtml(j.params.label || snippet(j.params.prompt, 60))}</span>
      <span class="params">${fmtMin(j.elapsed_sec)} · ${j.finished_at ? j.finished_at.slice(11) : ''}</span>
      <span></span>
    </li>`).join('');

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

// ====== Form submit ======
document.getElementById('genForm').addEventListener('submit', async e => {
  e.preventDefault();
  const fd = new FormData(e.target);
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

  // Reset state classes — we set the right one below.
  card.classList.remove('state-missing', 'state-warn', 'state-downloading');
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
  const needsQ8 = (currentMode === 'keyframe')
                || (document.getElementById('quality').value === 'high');
  if (needsQ8 && !q8Ok && tier.allows_q8 !== false) {
    card.style.display = '';
    card.classList.add('state-warn');
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

  // ----- All good — keep the card visible but quiet ------------------------
  // Earlier version hid the card entirely when nothing was actionable, but
  // that made users on healthy installs think "the download UI doesn't
  // exist." The card now stays visible with a small ✓ status, telling
  // them what's installed and giving them a path to the per-repo manager.
  // Default border / background = muted neutral, no warning colour.
  card.style.display = '';
  card.classList.remove('state-missing', 'state-warn', 'state-downloading');
  icon.textContent = '✓';
  const ready = s.repos_ready ?? 0;
  const total = s.repos_total ?? 0;
  title.textContent = `Models ready · ${ready}/${total}`;
  const partialNote = (q8Ok && baseOk) ? '' : ` · ${total - ready} optional missing`;
  sub.innerHTML =
    `All installed weights detected${partialNote}. ` +
    `<a style="color:var(--accent-bright,#7e98ff); cursor:pointer; text-decoration:underline" onclick="openModelsModal()">Manage models →</a>`;
  actions.innerHTML = '';   // no big button in the ready state
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
  });
}
function closeTierModal() { document.getElementById('tierModal').style.display = 'none'; }

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
