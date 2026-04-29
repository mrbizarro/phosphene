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
HISTORY_LIMIT = 60
LOG_LIMIT = 1000

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
        HIDDEN_FILE.write_text(json.dumps(sorted(HIDDEN_PATHS), indent=2))
    except Exception as exc:
        push(f"hidden persist failed: {exc}")


def set_hidden(path: str, hidden: bool) -> None:
    if hidden:
        HIDDEN_PATHS.add(path)
    else:
        HIDDEN_PATHS.discard(path)
    persist_hidden()


def list_outputs(include_hidden: bool = False) -> list[dict]:
    files = sorted(OUTPUT.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)[:120]
    out = []
    for p in files:
        path_s = str(p)
        is_hidden = path_s in HIDDEN_PATHS
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
        path.write_text(json.dumps(payload, indent=2))
    except Exception as exc:
        push(f"Sidecar write failed: {exc}")


# ---- queue persistence -------------------------------------------------------

def persist_queue() -> None:
    with LOCK:
        snapshot = {
            "queue": [_strip_for_disk(j) for j in STATE["queue"]],
            "current": _strip_for_disk(STATE["current"]) if STATE["current"] else None,
            "history": [_strip_for_disk(j) for j in STATE["history"][:30]],
            "paused": STATE["paused"],
        }
    try:
        QUEUE_FILE.write_text(json.dumps(snapshot, indent=2))
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
    with LOCK:
        cur = STATE["current"]
    if cur is not None:
        cur["cancel_requested"] = True
    push("Stop requested — killing helper to abort current job.")
    HELPER.kill()


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
            "extend_steps": max(1, int(f("extend_steps", "30") or 30)),
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
        src = p["video_path"]
        if not src or not Path(src).exists():
            raise RuntimeError(f"source video for extend not found: {src}")
        out_name = Path(src).stem + f"_ext{p['extend_frames']}_{stamp}.mp4"
        final_out = OUTPUT / out_name
        job["raw_path"] = str(final_out)

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
                "steps": p["extend_steps"],
            },
        }
        push(f"Extend via helper: id={job['id']} src={Path(src).name} +{p['extend_frames']}f")
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
        if not Q8_LOCAL_PATH.exists() or not any(Q8_LOCAL_PATH.iterdir() if Q8_LOCAL_PATH.is_dir() else []):
            raise RuntimeError(
                f"Keyframe mode requires Q8 model at {Q8_LOCAL_PATH}. "
                f"Run: huggingface-cli download {MODEL_ID_HQ} --local-dir {Q8_LOCAL_PATH}"
            )
        if not p.get("start_image") or not Path(p["start_image"]).exists():
            raise RuntimeError(f"start_image not found: {p.get('start_image')}")
        if not p.get("end_image") or not Path(p["end_image"]).exists():
            raise RuntimeError(f"end_image not found: {p.get('end_image')}")
        width, height = p["width"], p["height"]
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
    pad_w, pad_h, pad_filter = compute_pad(width, height)
    suffix = f"{pad_w}x{pad_h}" if mode == "i2v_clean_audio" and pad_filter else f"{width}x{height}"
    tag = f"{mode}_hq" if quality == "high" else mode
    raw_out = OUTPUT / f"mlx_{tag}_{width}x{height}_{frames}f_{stamp}_raw.mp4"
    final_out = OUTPUT / f"mlx_{tag}_{suffix}_{frames}f_{stamp}.mp4"
    job["raw_path"] = str(raw_out)

    if quality == "high":
        # Route to TwoStageHQPipeline (Q8 dev model + res_2s sampler + CFG anchor + TeaCache).
        # Defaults from ltx-2-mlx CLAUDE.md LTX_2_3_PARAMS.
        if not Q8_LOCAL_PATH.exists() or not any(Q8_LOCAL_PATH.iterdir() if Q8_LOCAL_PATH.is_dir() else []):
            raise RuntimeError(
                f"High quality requires Q8 model at {Q8_LOCAL_PATH}. "
                f"Run: huggingface-cli download {MODEL_ID_HQ} --local-dir {Q8_LOCAL_PATH}"
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
                "stg_scale": 1.0,
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
        mux_cmd = [str(FFMPEG), "-y", "-i", str(raw_out), "-i", audio,
                   "-map", "0:v:0", "-map", "1:a:0"]
        if pad_filter:
            mux_cmd += ["-vf", pad_filter]
        mux_cmd += [
            "-af", f"apad,atrim=0:{duration},asetpts=PTS-STARTPTS",
            "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-t", f"{duration}",
            str(final_out),
        ]
        push("Mux: " + " ".join(shlex.quote(c) for c in mux_cmd))
        mux = subprocess.run(mux_cmd, env=env, text=True, capture_output=True)
        if mux.returncode != 0:
            push((mux.stderr or "").strip())
            raise RuntimeError(f"mux exited with code {mux.returncode}")
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

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._ok(page().encode())
            return
        if parsed.path == "/status":
            qs = parse_qs(parsed.query)
            include_hidden = qs.get("include_hidden", ["0"])[0] == "1"
            with LOCK:
                avg = _avg_elapsed()
                payload = {
                    "running": STATE["running"], "paused": STATE["paused"],
                    "current": STATE["current"], "queue": STATE["queue"],
                    "history": STATE["history"][:30], "log": STATE["log"],
                    "pid": STATE["pid"], "pgid": STATE["pgid"],
                }
            payload["outputs"] = list_outputs(include_hidden=include_hidden)
            payload["hidden_count"] = len(HIDDEN_PATHS)
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
            # Q8 is "available" only when ALL critical safetensors are on disk.
            # HQ + Keyframe pipelines need dev transformer + connector +
            # distilled LoRA (used for stage-2 refine) + VAE + audio. A
            # partially-downloaded model passes basic exists() checks but
            # fails at load_safetensors mid-run, which is worse than
            # reporting False.
            _Q8_REQUIRED = (
                "connector.safetensors",
                "transformer-dev.safetensors",
                "ltx-2.3-22b-distilled-lora-384.safetensors",  # stage-2 refine
                "vae_decoder.safetensors",
                "vae_encoder.safetensors",
                "audio_vae.safetensors",
                "vocoder.safetensors",
                "spatial_upscaler_x2_v1_1.safetensors",  # used by two-stage upscale
            )
            _q8_missing = []
            if Q8_LOCAL_PATH.exists() and Q8_LOCAL_PATH.is_dir():
                for fname in _Q8_REQUIRED:
                    fpath = Q8_LOCAL_PATH / fname
                    if not fpath.exists() or fpath.stat().st_size < 1024:
                        _q8_missing.append(fname)
            else:
                _q8_missing = list(_Q8_REQUIRED)
            payload["q8_available"] = not _q8_missing
            payload["q8_missing"] = _q8_missing
            payload["q8_path"] = str(Q8_LOCAL_PATH)
            self._json(payload)
            return
        if parsed.path == "/file":
            qs = parse_qs(parsed.query)
            try:
                path = Path(qs.get("path", [""])[0]).resolve()
            except Exception:
                self.send_error(400); return
            if not str(path).startswith(str(OUTPUT.resolve())) or not path.exists():
                self.send_error(404); return
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(path.stat().st_size))
            self.send_header("Accept-Ranges", "none")
            self.end_headers()
            with path.open("rb") as fh:
                while chunk := fh.read(1024 * 1024):
                    self.wfile.write(chunk)
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
            if not str(path).startswith(str(OUTPUT.resolve())):
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
    .creator-link {
      display: inline-flex; align-items: center; gap: 7px;
      color: var(--muted); font-size: 11px; text-decoration: none;
      padding: 3px 8px 3px 3px; border-radius: 999px;
      border: 1px solid var(--border); background: var(--panel-2);
      transition: 0.12s;
    }
    .creator-link:hover { color: var(--accent-bright); border-color: var(--accent); }
    .creator-avatar {
      width: 22px; height: 22px; border-radius: 50%;
      object-fit: cover; display: block;
      box-shadow: 0 0 0 1px var(--border);
    }

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
  </style>
</head>
<body>

<header>
  <a href="/" class="brand"><img src="/assets/logo-header.png" alt="LTX23MLX"></a>
  <span class="tag" id="modelTag"></span>
  <span class="spacer"></span>
  <span id="memPill" class="pill">memory…</span>
  <span id="comfyPill" class="pill" style="display:none">comfy…</span>
  <span id="helperPill" class="pill">helper…</span>
  <span id="queuePill" class="pill">queue 0</span>
  <span id="jobPill" class="pill">idle</span>
  <button id="stopComfyBtn" class="ghost-btn" style="display:none" onclick="api('/stop_comfy', 'POST').then(poll)">Stop Comfy</button>
  <a class="creator-link" href="https://x.com/AIBizarrothe" target="_blank" rel="noopener" title="Mr. Bizarro on X">
    <img src="/assets/bizarro-avatar.jpg" class="creator-avatar" alt="">
    <span>by Bizarro</span>
  </a>
</header>

<main class="layout">

  <!-- ============== FORM PANE ============== -->
  <aside class="form-pane">
    <form id="genForm">
      <input type="hidden" name="preset_label" id="preset_label" value="">

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

      <!-- Mode-specific: image -->
      <div class="mode-only" id="imageSection">
        <h2>Reference image</h2>
        <input name="image" id="image" placeholder="path or click Upload">
        <div class="img-row">
          <button type="button" class="small" onclick="document.getElementById('imageFile').click()">Upload…</button>
          <input type="file" id="imageFile" accept="image/*" onchange="uploadImage()">
          <span class="hint" id="imgHint">PIL cover-crop applied automatically to W×H</span>
        </div>
        <img id="imagePreview" class="img-preview" alt="">
      </div>

      <!-- Mode-specific: keyframe (FFLF) -->
      <div class="mode-only" id="keyframeSection">
        <h2>Start frame (frame 0)</h2>
        <input name="start_image" id="start_image" placeholder="path or click Upload">
        <div class="img-row">
          <button type="button" class="small" onclick="document.getElementById('startImageFile').click()">Upload start…</button>
          <input type="file" id="startImageFile" accept="image/*" onchange="uploadKeyframe('start')">
        </div>
        <img id="startImagePreview" class="img-preview" alt="">

        <h2>End frame (last frame)</h2>
        <input name="end_image" id="end_image" placeholder="path or click Upload">
        <div class="img-row">
          <button type="button" class="small" onclick="document.getElementById('endImageFile').click()">Upload end…</button>
          <input type="file" id="endImageFile" accept="image/*" onchange="uploadKeyframe('end')">
        </div>
        <img id="endImagePreview" class="img-preview" alt="">

        <div class="hint">FFLF needs Q8 (auto-selects High quality). Closeup as the end frame anchors face identity through the clip.</div>
      </div>

      <!-- Mode-specific: extend -->
      <div class="mode-only" id="extendSection">
        <h2>Source video</h2>
        <select id="extendSrcSelect" onchange="document.getElementById('video_path').value=this.value"></select>
        <input name="video_path" id="video_path" placeholder="/path/to/source.mp4" style="margin-top:6px">
        <div class="row" style="margin-top:8px">
          <div>
            <label class="lbl">Extend by (latent frames)</label>
            <input name="extend_frames" id="extend_frames" type="number" value="5" min="1" max="32">
          </div>
          <div>
            <label class="lbl">Direction</label>
            <select name="extend_direction" id="extend_direction">
              <option value="after" selected>After</option>
              <option value="before">Before</option>
            </select>
          </div>
        </div>
        <label class="lbl">Stage-1 steps</label>
        <input name="extend_steps" id="extend_steps" type="number" value="30" min="4" max="60">
        <div class="hint">Each latent ≈ 8 frames (~0.33s). Q8 weights recommended for two-stage extend.</div>
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

        <div class="row3" style="margin-top:10px">
          <div><label class="lbl">Width</label><input name="width" id="width" value="1280" type="number" min="32" step="32"></div>
          <div><label class="lbl">Height</label><input name="height" id="height" value="704" type="number" min="32" step="32"></div>
          <div><label class="lbl">Duration (s)</label><input id="duration" value="5" type="number" min="1" max="20" step="1"></div>
        </div>

        <div class="row" style="margin-top:6px">
          <div><label class="lbl">Frames (8k+1)</label><input name="frames" id="frames" value="121" type="number" min="1"></div>
          <div><label class="lbl">Seed (-1 random)</label><input name="seed" id="seed" value="-1"></div>
        </div>

        <input type="hidden" name="steps" id="steps" value="8">

        <div class="derived" id="derived"></div>
      </div>

      <input type="hidden" name="stop_comfy" id="stop_comfy" value="on">

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

<!-- ============== BOTTOM TABBED PANE ============== -->
<aside class="bottom-pane" id="bottomPane">
  <nav class="tabs">
    <button data-tab="now" class="active">Now</button>
    <button data-tab="queue">Queue <span class="badge" id="queueBadge" style="display:none">0</span></button>
    <button data-tab="recent">Recent</button>
    <button data-tab="logs">Logs</button>
    <span class="spacer"></span>
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

document.getElementById('modelTag').textContent = BOOT.model;
document.getElementById('image').value = BOOT.default_image;
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
}
function setQuality(q) {
  document.getElementById('quality').value = q;
  document.querySelectorAll('#qualityGroup .pill-btn').forEach(b => b.classList.toggle('active', b.dataset.quality === q));
  applyQuality();
}
function setAspect(a) {
  document.getElementById('aspect').value = a;
  document.querySelectorAll('#aspectGroup .pill-btn').forEach(b => b.classList.toggle('active', b.dataset.aspect === a));
  applyAspect(a);
}

document.querySelectorAll('#modeGroup .pill-btn').forEach(b => b.onclick = () => setMode(b.dataset.mode));
document.querySelectorAll('#qualityGroup .pill-btn').forEach(b => b.onclick = () => { if (!b.classList.contains('disabled')) setQuality(b.dataset.quality); });
document.querySelectorAll('#aspectGroup .pill-btn').forEach(b => b.onclick = () => setAspect(b.dataset.aspect));
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
  document.getElementById('imageSection').classList.toggle('show', inI2V && currentMode !== 'keyframe');
  document.getElementById('extendSection').classList.toggle('show', currentMode === 'extend');
  document.getElementById('keyframeSection').classList.toggle('show', currentMode === 'keyframe');
  document.getElementById('sizingSection').classList.toggle('show', currentMode !== 'extend');
  document.getElementById('audioSection').classList.toggle('show', mode === 'i2v_clean_audio');

  // Image preview (single image — i2v modes)
  const imgPath = document.getElementById('image').value.trim();
  const preview = document.getElementById('imagePreview');
  if (inI2V && currentMode !== 'keyframe' && imgPath) {
    preview.src = '/image?path=' + encodeURIComponent(imgPath);
    preview.classList.add('show');
  } else preview.classList.remove('show');

  // Keyframe previews (start + end)
  for (const which of ['start', 'end']) {
    const path = document.getElementById(which + '_image').value.trim();
    const p = document.getElementById(which + 'ImagePreview');
    if (currentMode === 'keyframe' && path) {
      p.src = '/image?path=' + encodeURIComponent(path);
      p.classList.add('show');
    } else p.classList.remove('show');
  }
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
document.getElementById('image').addEventListener('input', updateDerived);
['start_image', 'end_image'].forEach(id => document.getElementById(id).addEventListener('input', updateDerived));

async function uploadImage() {
  const f = document.getElementById('imageFile').files[0];
  if (!f) return;
  const fd = new FormData(); fd.append('image', f);
  const r = await fetch('/upload', { method: 'POST', body: fd });
  const data = await r.json();
  if (data.ok) {
    document.getElementById('image').value = data.path;
    document.getElementById('imgHint').textContent = `Uploaded: ${f.name} (${(f.size/1024).toFixed(0)} KB)`;
    updateDerived();
  } else alert('Upload failed: ' + (data.error || '?'));
}

async function uploadKeyframe(which) {
  const f = document.getElementById(which + 'ImageFile').files[0];
  if (!f) return;
  const fd = new FormData(); fd.append('image', f);
  const r = await fetch('/upload', { method: 'POST', body: fd });
  const data = await r.json();
  if (data.ok) {
    document.getElementById(which + '_image').value = data.path;
    updateDerived();
  } else alert('Upload failed: ' + (data.error || '?'));
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
async function poll() {
  let s;
  const url = '/status' + (filterMode === 'hidden' ? '?include_hidden=1' : '');
  try { s = await (await fetch(url)).json(); } catch (e) { return; }

  // Memory
  const m = s.memory;
  const memPill = document.getElementById('memPill');
  memPill.innerHTML = `<span class="dot"></span>${fmtMem(m)}`;
  let memCls = 'pill-good';
  if (m.swap_gb > 8 || m.pressure_pct > 90) memCls = 'pill-danger';
  else if (m.swap_gb > 4 || m.pressure_pct > 75) memCls = 'pill-warn';
  memPill.className = 'pill ' + memCls;

  // Comfy (hidden when not running)
  const cp = document.getElementById('comfyPill');
  const stopBtn = document.getElementById('stopComfyBtn');
  if (s.comfy_pids.length) {
    cp.innerHTML = `<span class="dot"></span>Comfy ${s.comfy_pids.join(', ')}`;
    cp.className = 'pill pill-warn'; cp.style.display = '';
    stopBtn.style.display = '';
  } else {
    cp.style.display = 'none';
    stopBtn.style.display = 'none';
  }

  // Helper
  const hp = document.getElementById('helperPill');
  if (s.helper && s.helper.alive) {
    hp.innerHTML = `<span class="dot"></span>helper warm`;
    hp.className = 'pill pill-good';
  } else {
    hp.innerHTML = `<span class="dot"></span>helper cold`;
    hp.className = 'pill';
  }

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
      currentOutputs.slice(0, 40).map(o => `<option value="${o.path}">${o.name}</option>`).join('');
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
  else if (p.mode === 'i2v_clean_audio' || p.mode === 'i2v') { setMode('i2v'); document.getElementById('i2vMode').value = p.mode; document.getElementById('mode').value = p.mode; }
  else setMode('t2v');
  document.getElementById('prompt').value = p.prompt || '';
  if (p.width) document.getElementById('width').value = p.width;
  if (p.height) document.getElementById('height').value = p.height;
  if (p.frames) { document.getElementById('frames').value = p.frames; document.getElementById('duration').value = framesToDuration(p.frames); }
  if (p.steps) document.getElementById('steps').value = p.steps;
  if (p.seed != null) document.getElementById('seed').value = p.seed;
  if (p.image) document.getElementById('image').value = p.image;
  if (p.audio) document.getElementById('audio').value = p.audio;
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

// ====== Init ======
setInterval(poll, 1500);
poll();
setMode('t2v');
setQuality('standard');
setAspect('landscape');
updateDerived();
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
