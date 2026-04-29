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
            # The HQ pipeline needs the dev transformer + connector + distilled
            # LoRA + VAE + audio. A partially-downloaded model passes basic
            # exists() checks but fails at load_safetensors mid-run, which is
            # worse than reporting False. List the files actually consumed by
            # ti2vid_two_stages_hq's loader.
            _Q8_REQUIRED = (
                "connector.safetensors",
                "transformer-dev.safetensors",
                "vae_decoder.safetensors",
                "vae_encoder.safetensors",
                "audio_vae.safetensors",
                "vocoder.safetensors",
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
  <title>LTX MLX Studio</title>
  <style>
    :root {
      --bg: #0d1117; --panel: #161b22; --panel-2: #1c2230;
      --border: #30363d; --text: #c9d1d9; --muted: #8b949e;
      --accent: #2f81f7; --accent-bright: #58a6ff;
      --success: #3fb950; --warning: #d29922; --danger: #f85149;
    }
    * { box-sizing: border-box; }
    html, body { margin: 0; }
    body {
      min-height: 100vh; background: var(--bg); color: var(--text);
      font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    }
    header {
      display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
      padding: 11px 18px; border-bottom: 1px solid var(--border);
      background: var(--panel); position: sticky; top: 0; z-index: 10;
    }
    header h1 { margin: 0; font-size: 16px; font-weight: 700; letter-spacing: -0.01em; }
    .tag { color: var(--muted); font-size: 12px; }
    .pill {
      padding: 4px 10px; border-radius: 999px; font-size: 12px; font-weight: 500;
      background: var(--panel-2); border: 1px solid var(--border); color: var(--muted);
      white-space: nowrap;
    }
    .pill .dot {
      display: inline-block; width: 7px; height: 7px; border-radius: 999px;
      margin-right: 5px; background: currentColor; vertical-align: middle;
    }
    .pill-good { color: var(--success); border-color: rgba(63, 185, 80, 0.4); }
    .pill-warn { color: var(--warning); border-color: rgba(210, 153, 34, 0.5); }
    .pill-danger { color: var(--danger); border-color: rgba(248, 81, 73, 0.5); }
    .pill-running {
      color: var(--accent-bright); border-color: var(--accent);
      animation: pulse 1.6s ease-in-out infinite;
    }
    @keyframes pulse { 50% { opacity: 0.7; } }
    .spacer { flex: 1; }
    .ghost-btn {
      background: transparent; border: 1px solid var(--border); color: var(--text);
      padding: 5px 10px; border-radius: 6px; font-size: 12px; cursor: pointer;
    }
    .ghost-btn:hover { border-color: var(--accent); color: var(--accent-bright); }
    main {
      display: grid; grid-template-columns: 460px 1fr; gap: 16px;
      padding: 16px; max-width: 1700px; margin: 0 auto;
    }
    section {
      border: 1px solid var(--border); border-radius: 10px; padding: 18px;
      background: var(--panel); margin-bottom: 16px;
    }
    .right-col > section:last-child { margin-bottom: 0; }
    h2 {
      font-size: 11px; margin: 0 0 14px; color: var(--muted);
      text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600;
      display: flex; justify-content: space-between; align-items: center; gap: 12px;
    }
    h2 .h2-meta { font-size: 11px; color: var(--muted); font-weight: 400; text-transform: none; letter-spacing: 0; }
    h2 .h2-actions { display: flex; gap: 6px; }
    h2.spaced { margin-top: 22px; }
    label {
      display: block; margin: 12px 0 4px; color: var(--muted);
      font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 500;
    }
    input, textarea, select, button {
      width: 100%; padding: 9px 11px; font: inherit; color: inherit;
      background: var(--panel-2); border: 1px solid var(--border); border-radius: 6px;
    }
    input:focus, textarea:focus, select:focus {
      outline: none; border-color: var(--accent); background: #161b22;
    }
    textarea { min-height: 84px; resize: vertical; font-family: -apple-system, system-ui, sans-serif; }
    textarea.batch { min-height: 200px; font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 12px; }
    button { cursor: pointer; font-weight: 600; transition: 0.1s; }
    button.primary {
      background: var(--accent); border-color: var(--accent); color: white;
      padding: 11px; font-size: 14px;
    }
    button.primary:hover { background: var(--accent-bright); border-color: var(--accent-bright); }
    button.primary:disabled { opacity: 0.6; cursor: not-allowed; }
    button.danger { color: var(--danger); border-color: rgba(248, 81, 73, 0.4); background: transparent; }
    button.danger:hover { background: rgba(248, 81, 73, 0.1); }
    button.small {
      width: auto; padding: 6px 10px; font-size: 12px;
      background: var(--panel-2); font-weight: 500;
    }
    button.small:hover { border-color: var(--accent); }
    button.tiny {
      width: auto; padding: 3px 8px; font-size: 11px;
      background: transparent; border: 1px solid var(--border); color: var(--muted);
      font-weight: 500;
    }
    button.tiny:hover { color: var(--text); border-color: var(--accent); }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .row3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
    .preset-grid {
      display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 6px; margin-bottom: 6px;
    }
    .preset {
      padding: 8px 10px; border: 1px solid var(--border); border-radius: 8px;
      background: var(--panel-2); cursor: pointer; text-align: left; transition: 0.15s;
    }
    .preset:hover { border-color: var(--accent); transform: translateY(-1px); }
    .preset .ttl { font-weight: 600; font-size: 12px; color: var(--text); }
    .preset .sub { font-size: 10px; color: var(--muted); margin-top: 2px; }
    .check {
      display: flex; align-items: center; gap: 8px; margin: 12px 0; cursor: pointer;
      color: var(--text); font-size: 13px;
    }
    .check input { width: auto; margin: 0; }
    .actions { display: grid; grid-template-columns: 2fr 1fr; gap: 8px; margin-top: 16px; }
    .actions button { padding: 11px; }
    .meta { display: flex; gap: 12px; flex-wrap: wrap; color: var(--muted); font-size: 12px; }
    .derived {
      margin-top: 8px; padding: 10px 12px; border-radius: 6px;
      background: var(--panel-2); border: 1px solid var(--border);
      font-size: 12px; color: var(--muted);
    }
    .derived strong { color: var(--accent-bright); font-weight: 600; }
    .warn-banner {
      background: rgba(210, 153, 34, 0.08);
      border: 1px solid rgba(210, 153, 34, 0.4);
      color: var(--warning);
      padding: 10px 12px; border-radius: 6px; margin-bottom: 12px;
      font-size: 12px; display: none;
    }
    .warn-banner.show { display: block; }
    pre.log {
      white-space: pre-wrap; word-break: break-all;
      min-height: 180px; max-height: 380px; overflow: auto;
      background: #0a0e14; border: 1px solid var(--border); border-radius: 8px;
      padding: 12px; font: 12px/1.5 ui-monospace, "SF Mono", Menlo, monospace;
      color: #b0b8c4; margin: 0;
    }
    video.player {
      width: 100%; max-height: 56vh; background: black; border-radius: 8px;
      margin-top: 8px;
    }
    .now-card {
      padding: 12px; border-radius: 8px; background: var(--panel-2);
      border: 1px solid var(--border); margin-bottom: 12px;
    }
    .now-card.idle { opacity: 0.7; }
    .now-card .ttl { font-weight: 600; font-size: 13px; }
    .now-card .meta { margin-top: 6px; font-size: 12px; color: var(--muted); }
    .progress-bar {
      height: 6px; background: var(--border); border-radius: 3px; overflow: hidden;
      margin: 8px 0;
    }
    .progress-bar .fill { height: 100%; background: var(--accent); transition: width 0.3s; }
    .queue-list { list-style: none; padding: 0; margin: 0; }
    .queue-list li, .history-list li {
      display: grid; grid-template-columns: auto 1fr auto auto; gap: 10px;
      align-items: center; padding: 9px 10px; border-radius: 6px;
      border: 1px solid var(--border); background: var(--panel-2);
      margin-bottom: 6px; font-size: 12px;
    }
    .queue-list li .pos { color: var(--muted); font-weight: 600; min-width: 22px; }
    .queue-list li .ttl, .history-list li .ttl {
      overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }
    .queue-list li .params, .history-list li .params {
      color: var(--muted); font-size: 11px; white-space: nowrap;
    }
    .queue-list li button, .history-list li button {
      width: auto; background: transparent; border: 0; color: var(--muted);
      cursor: pointer; padding: 2px 6px; font-size: 14px; line-height: 1;
    }
    .queue-list li button:hover { color: var(--danger); }
    .history-list li.done .badge { color: var(--success); }
    .history-list li.failed .badge { color: var(--danger); }
    .history-list li.cancelled .badge { color: var(--muted); }
    .badge {
      font-size: 10px; text-transform: uppercase; letter-spacing: 0.08em;
      padding: 2px 8px; border-radius: 999px; border: 1px solid currentColor;
      font-weight: 600;
    }
    .queue-actions { display: flex; gap: 8px; margin-top: 10px; flex-wrap: wrap; }
    .empty { color: var(--muted); font-size: 12px; padding: 8px 0; text-align: center; }
    details summary { cursor: pointer; color: var(--muted); font-size: 12px; padding: 4px 0; }
    details summary:hover { color: var(--text); }
    details[open] summary { margin-bottom: 8px; }
    .hint { color: var(--muted); font-size: 11px; margin-top: 4px; }
    .out-toolbar {
      display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-bottom: 12px;
    }
    .out-toolbar .seg { display: inline-flex; border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }
    .out-toolbar .seg button {
      width: auto; padding: 6px 12px; font-size: 12px; background: transparent;
      border: 0; border-right: 1px solid var(--border); border-radius: 0; color: var(--muted); font-weight: 500;
    }
    .out-toolbar .seg button:last-child { border-right: 0; }
    .out-toolbar .seg button.active { background: var(--accent); color: white; }
    .out-toolbar .seg button:hover:not(.active) { color: var(--text); background: rgba(255,255,255,0.04); }
    .out-grid {
      display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr));
      gap: 10px; margin-bottom: 12px;
    }
    .out-card {
      border: 1px solid var(--border); border-radius: 8px; overflow: hidden;
      background: var(--panel-2); cursor: pointer; transition: 0.12s;
      display: flex; flex-direction: column;
    }
    .out-card:hover { border-color: var(--accent); transform: translateY(-1px); }
    .out-card.active { border-color: var(--accent-bright); box-shadow: 0 0 0 1px var(--accent-bright); }
    .out-card video {
      width: 100%; aspect-ratio: 16/9; object-fit: cover; background: black; display: block;
    }
    .out-card .info { padding: 8px 10px; }
    .out-card .name {
      font-size: 11px; font-weight: 500; overflow: hidden; text-overflow: ellipsis;
      white-space: nowrap; color: var(--text);
    }
    .out-card .sub { font-size: 10px; color: var(--muted); margin-top: 3px; }
    .out-card .actions-row {
      display: flex; justify-content: space-between; gap: 4px; padding: 0 8px 8px;
    }
    .out-card .actions-row button {
      width: auto; padding: 3px 7px; font-size: 10px; background: transparent;
      border: 1px solid var(--border); color: var(--muted); font-weight: 500;
    }
    .out-card .actions-row button:hover { color: var(--text); border-color: var(--accent); }
    .out-card.hidden-card { opacity: 0.45; }
    .out-card.hidden-card .name::after { content: " · hidden"; color: var(--muted); }
    /* Image preview */
    .img-preview {
      display: none; margin-top: 8px; max-width: 100%;
      border-radius: 8px; border: 1px solid var(--border);
    }
    .img-preview.show { display: block; }
    .img-row { display: flex; gap: 6px; align-items: center; margin-top: 6px; }
    .img-row label { margin: 0; flex: 0 0 auto; }
    .img-row input[type="file"] { display: none; }
    /* Mode-aware sections */
    .mode-only { display: none; }
    .mode-only.show { display: block; }
  </style>
</head>
<body>
<header>
  <h1>⚡ LTX MLX Studio</h1>
  <span class="tag" id="modelTag"></span>
  <span class="spacer"></span>
  <span id="memPill" class="pill">memory…</span>
  <span id="comfyPill" class="pill" style="display:none">comfy…</span>
  <span id="helperPill" class="pill">helper…</span>
  <span id="queuePill" class="pill">queue 0</span>
  <span id="jobPill" class="pill">idle</span>
  <button id="stopComfyBtn" class="ghost-btn" style="display:none" onclick="api('/stop_comfy', 'POST').then(poll)">Stop Comfy</button>
  <button class="ghost-btn" onclick="api('/helper/restart', 'POST').then(()=>setTimeout(poll,500))">Restart helper</button>
  <button class="ghost-btn" onclick="api('/open_pinokio', 'POST').then(poll)">Open Pinokio</button>
</header>

<main>
  <section>
    <h2>Quick presets</h2>
    <div class="preset-grid" id="presets"></div>

    <div id="warnBanner" class="warn-banner"></div>

    <h2 class="spaced">Generate</h2>
    <form id="genForm">
      <input type="hidden" name="preset_label" id="preset_label" value="">

      <label>Mode</label>
      <select name="mode" id="mode">
        <option value="t2v" selected>Text → video</option>
        <option value="i2v">Image → video</option>
        <option value="i2v_clean_audio">Image → video + clean audio mux</option>
        <option value="extend">Extend (chain a clip — uses dev model, slower)</option>
      </select>

      <label>Prompt</label>
      <textarea name="prompt" id="prompt"></textarea>

      <!-- Image input + upload + preview (i2v modes only) -->
      <div class="mode-only" id="imageSection">
        <label>Image (auto cover-cropped to width × height)</label>
        <input name="image" id="image">
        <div class="img-row">
          <button type="button" class="small" onclick="document.getElementById('imageFile').click()">Upload…</button>
          <input type="file" id="imageFile" accept="image/*" onchange="uploadImage()">
          <span class="hint" id="imgHint">PNG/JPG · any size · cover-crop applied</span>
        </div>
        <img id="imagePreview" class="img-preview" alt="">
      </div>

      <!-- Audio input (i2v_clean_audio only) -->
      <div class="mode-only" id="audioSection">
        <label>Audio (muxed in for clean audio mode)</label>
        <input name="audio" id="audio">
      </div>

      <!-- Extend section (extend mode only) -->
      <div class="mode-only" id="extendSection">
        <label>Source video</label>
        <select id="extendSrcSelect" onchange="document.getElementById('video_path').value=this.value"></select>
        <input name="video_path" id="video_path" placeholder="/path/to/source.mp4">
        <div class="row">
          <div>
            <label>Extend by (latent frames)</label>
            <input name="extend_frames" id="extend_frames" type="number" value="5" min="1" max="32">
          </div>
          <div>
            <label>Direction</label>
            <select name="extend_direction" id="extend_direction">
              <option value="after" selected>After (continue motion)</option>
              <option value="before">Before (prepend)</option>
            </select>
          </div>
        </div>
        <div><label>Stage-1 steps</label><input name="extend_steps" id="extend_steps" type="number" value="30" min="4" max="60"></div>
        <div class="hint">Each latent ≈ 8 actual frames ≈ 0.33s. Try 5 for ~1.7s extension. Q4 model may not support extend — needs q8 weights for two-stage.</div>
      </div>

      <!-- T2V/I2V sizing -->
      <div class="mode-only" id="sizingSection">
        <label>Aspect</label>
        <select id="aspect" onchange="applyAspect(this.value)">
          <option value="custom">Custom</option>
        </select>

        <div class="row">
          <div><label>Width</label><input name="width" id="width" value="1280" type="number" min="32" step="32"></div>
          <div><label>Height</label><input name="height" id="height" value="704" type="number" min="32" step="32"></div>
        </div>

        <div class="row">
          <div><label>Duration (s)</label><input id="duration" value="5" type="number" min="1" max="20" step="1"></div>
          <div><label>Frames</label><input name="frames" id="frames" value="121" type="number" min="1"></div>
        </div>

        <label>Quality</label>
        <select name="quality" id="quality" onchange="applyQuality()">
          <option value="draft">Draft — Q4 · 4 steps · ~3 min for 5s</option>
          <option value="standard" selected>Standard — Q4 · 8 steps · ~7 min for 5s</option>
          <option value="high" id="qualityHigh" disabled>High — Q8 two-stage + TeaCache · ~12 min for 5s (Q8 not installed)</option>
        </select>

        <div class="row">
          <div><label>Steps</label><input name="steps" id="steps" value="8" type="number" min="1" max="60"></div>
          <div><label>Seed (-1 random)</label><input name="seed" id="seed" value="-1"></div>
        </div>

        <div class="derived" id="derived"></div>
      </div>

      <details>
        <summary>Advanced</summary>
        <label class="check">
          <input type="checkbox" name="enhance" id="enhance"> Enhance prompt — currently CLI-only, ignored by warm helper
        </label>
        <label class="check">
          <input type="checkbox" name="open_when_done" id="open_when_done"> Open file when done (off for batches)
        </label>
      </details>

      <label class="check" id="stopComfyRow" style="display:none">
        <input type="checkbox" name="stop_comfy" id="stop_comfy" checked> Stop Comfy before render <span style="color:var(--muted);font-size:11px;">(Comfy detected on this machine — kill it to free RAM for 720p+ renders)</span>
      </label>

      <div class="actions">
        <button type="submit" class="primary" id="genBtn">▶ Add to queue</button>
        <button type="button" class="danger" onclick="api('/stop', 'POST').then(poll)">⏹ Stop current</button>
      </div>
    </form>

    <details style="margin-top: 16px" class="mode-only" id="batchDetails">
      <summary>Batch paste — split prompts with <code>---</code> on its own line</summary>
      <textarea class="batch" id="batchPrompts" placeholder="First prompt here.

---

Second prompt.

---

Third prompt."></textarea>
      <div class="hint">Each chunk between <code>---</code> lines becomes a queued job using the settings above. Auto-open is off for batches.</div>
      <button class="small" style="margin-top: 8px" onclick="queueBatch()">Queue all</button>
    </details>
  </section>

  <div class="right-col">
    <section>
      <h2>Now <span class="h2-meta" id="nowMeta"></span></h2>
      <div class="now-card idle" id="nowCard">
        <div class="ttl">Idle</div>
        <div class="progress-bar"><div class="fill" id="progressFill" style="width: 0%"></div></div>
        <div class="meta" id="nowDetail">No job running</div>
      </div>
      <pre class="log" id="log">No log yet.</pre>
    </section>

    <section>
      <h2>Queue <span class="h2-meta" id="queueMeta"></span></h2>
      <ul class="queue-list" id="queueList"></ul>
      <div class="queue-actions">
        <button class="small" id="pauseBtn" onclick="togglePause()">⏸ Pause</button>
        <button class="small danger" onclick="if(confirm('Clear all queued jobs?'))api('/queue/clear','POST').then(poll)">Clear queued</button>
      </div>
    </section>

    <section>
      <h2>Recent jobs <span class="h2-meta" id="historyMeta"></span></h2>
      <ul class="history-list" id="historyList"></ul>
    </section>

    <section>
      <h2>
        Outputs
        <span class="h2-actions">
          <div class="seg">
            <button id="filterAll" onclick="setFilter('visible')" class="active">Visible</button>
            <button id="filterHidden" onclick="setFilter('hidden')">Hidden</button>
          </div>
          <button class="tiny" onclick="if(confirm('Unhide ALL hidden outputs?'))api('/output/show_all','POST').then(poll)">Unhide all</button>
        </span>
      </h2>
      <div class="out-grid" id="outGrid"></div>
      <video id="player" class="player" controls></video>
      <div class="meta" id="outputMeta" style="margin-top:8px"></div>
      <div style="margin-top:8px; display:flex; gap:8px;">
        <button class="small" id="loadParamsBtn" onclick="loadParams()" disabled>↩ Load params from selected</button>
        <button class="small" id="useAsExtendBtn" onclick="useAsExtendSource()">⏭ Use as Extend source</button>
      </div>
    </section>
  </div>
</main>

<script>
const BOOT = __BOOTSTRAP__;
const PRESETS = BOOT.presets;
const ASPECTS = BOOT.aspects;
const FPS = BOOT.fps;

let filterMode = 'visible';
let activePath = null;
let currentOutputs = [];

document.getElementById('modelTag').textContent = BOOT.model;
document.getElementById('image').value = BOOT.default_image;
document.getElementById('audio').value = BOOT.default_audio;

// Build presets
const presetEl = document.getElementById('presets');
for (const p of PRESETS) {
  const btn = document.createElement('button');
  btn.type = 'button'; btn.className = 'preset';
  btn.innerHTML = `<div class="ttl">${p.label}</div><div class="sub">${p.sub}</div>`;
  btn.onclick = () => applyPreset(p);
  presetEl.appendChild(btn);
}

// Build aspect dropdown
const aspectEl = document.getElementById('aspect');
for (const [key, a] of Object.entries(ASPECTS)) {
  const opt = document.createElement('option');
  opt.value = key;
  opt.textContent = a.label;
  aspectEl.appendChild(opt);
}
aspectEl.value = 'landscape';
applyAspect('landscape', /* don't trigger derived yet */ true);

function applyPreset(p) {
  applyAspect(p.aspect, true);
  setDuration(p.dur);
  document.getElementById('steps').value = p.steps;
  document.getElementById('stop_comfy').checked = p.stop_comfy;
  document.getElementById('preset_label').value = p.label;
  updateDerived();
}

function applyAspect(key, suppressDerived) {
  if (key === 'custom') return;
  const a = ASPECTS[key];
  if (!a) return;
  document.getElementById('width').value = a.w;
  document.getElementById('height').value = a.h;
  document.getElementById('aspect').value = key;
  if (!suppressDerived) updateDerived();
}

function durationToFrames(s) {
  // 8k+1 snap
  const k = Math.max(0, Math.round(s * FPS / 8));
  return k * 8 + 1;
}
function framesToDuration(f) {
  return ((f - 1) / FPS).toFixed(2);
}
function setDuration(s) {
  document.getElementById('duration').value = s;
  document.getElementById('frames').value = durationToFrames(s);
}

function applyQuality() {
  const q = document.getElementById('quality').value;
  if (q === 'draft')         document.getElementById('steps').value = 4;
  else if (q === 'standard') document.getElementById('steps').value = 8;
  else if (q === 'high') {
    // High = Q8 two-stage HQ (stage1=15, stage2=3 internally). Steps field
    // is informational only — the helper routes to a different action when
    // quality=high (TBD, requires Q8 + helper update).
    document.getElementById('steps').value = 18; // 15+3 for display purposes
  }
  updateDerived();
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

  document.getElementById('derived').innerHTML =
    `Duration <strong>${dur}s</strong> @ ${FPS}fps · Output ${finalRes}`;

  const warns = [];
  if (w % 32 !== 0) warns.push(`Width ${w} isn't a multiple of 32 (closest ${Math.round(w/32)*32})`);
  if (h % 32 !== 0) warns.push(`Height ${h} isn't a multiple of 32 (closest ${Math.round(h/32)*32})`);
  if (f > 1 && (f - 1) % 8 !== 0) {
    const closest = Math.max(1, Math.round((f - 1) / 8) * 8 + 1);
    warns.push(`Frames work best as 8k+1 (closest ${closest})`);
  }
  const banner = document.getElementById('warnBanner');
  if (warns.length) {
    banner.innerHTML = '⚠ ' + warns.join(' · ');
    banner.classList.add('show');
  } else {
    banner.classList.remove('show');
  }

  // Mode-aware section visibility
  document.getElementById('imageSection').classList.toggle('show', mode === 'i2v' || mode === 'i2v_clean_audio');
  document.getElementById('audioSection').classList.toggle('show', mode === 'i2v_clean_audio');
  document.getElementById('extendSection').classList.toggle('show', mode === 'extend');
  document.getElementById('sizingSection').classList.toggle('show', mode !== 'extend');
  document.getElementById('batchDetails').classList.toggle('show', mode !== 'extend');

  // Image preview
  const imgPath = document.getElementById('image').value.trim();
  const preview = document.getElementById('imagePreview');
  if ((mode === 'i2v' || mode === 'i2v_clean_audio') && imgPath) {
    preview.src = '/image?path=' + encodeURIComponent(imgPath);
    preview.classList.add('show');
  } else {
    preview.classList.remove('show');
  }
}

['width', 'height', 'mode'].forEach(id => {
  document.getElementById(id).addEventListener('input', () => {
    document.getElementById('preset_label').value = '';
    updateDerived();
  });
});
document.getElementById('aspect').addEventListener('change', e => applyAspect(e.target.value));
document.getElementById('duration').addEventListener('input', e => {
  document.getElementById('frames').value = durationToFrames(parseFloat(e.target.value) || 0);
  document.getElementById('preset_label').value = '';
  updateDerived();
});
document.getElementById('frames').addEventListener('input', e => {
  document.getElementById('duration').value = framesToDuration(parseInt(e.target.value) || 0);
  document.getElementById('preset_label').value = '';
  updateDerived();
});
document.getElementById('image').addEventListener('input', updateDerived);

async function uploadImage() {
  const f = document.getElementById('imageFile').files[0];
  if (!f) return;
  const fd = new FormData();
  fd.append('image', f);
  const r = await fetch('/upload', { method: 'POST', body: fd });
  const data = await r.json();
  if (data.ok) {
    document.getElementById('image').value = data.path;
    document.getElementById('imgHint').textContent = `Uploaded: ${f.name} (${(f.size/1024).toFixed(0)} KB) → will be cover-cropped to ${document.getElementById('width').value}×${document.getElementById('height').value}`;
    updateDerived();
  } else {
    alert('Upload failed: ' + (data.error || 'unknown'));
  }
}

function fmtMem(m) { return `${m.used_gb.toFixed(1)} / ${m.total_gb.toFixed(0)} GB · swap ${m.swap_gb.toFixed(1)} GB`; }
function fmtMin(s) {
  if (!s || s < 0) return '—';
  const m = Math.floor(s / 60);
  const sec = Math.round(s % 60);
  return m > 0 ? `${m}m ${sec}s` : `${sec}s`;
}
function snippet(s, n = 70) {
  if (!s) return '';
  s = s.replace(/\s+/g, ' ').trim();
  return s.length > n ? s.slice(0, n - 1) + '…' : s;
}
function escapeHtml(s) {
  if (!s) return '';
  return s.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

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

async function poll() {
  let s;
  const url = '/status' + (filterMode === 'hidden' ? '?include_hidden=1' : '');
  try { s = await (await fetch(url)).json(); } catch (e) { return; }

  const m = s.memory;
  const memPill = document.getElementById('memPill');
  memPill.innerHTML = `<span class="dot"></span>${fmtMem(m)}`;
  let memCls = 'pill-good';
  if (m.swap_gb > 8 || m.pressure_pct > 90) memCls = 'pill-danger';
  else if (m.swap_gb > 4 || m.pressure_pct > 75) memCls = 'pill-warn';
  memPill.className = 'pill ' + memCls;

  // Comfy UI is hidden by default. It only appears when a Comfy process is
  // detected on this machine — i.e. for the small subset of users who run
  // ComfyUI alongside this panel and need to kill it before heavy renders.
  // Fresh Pinokio installs of this panel never see this UI.
  const cp = document.getElementById('comfyPill');
  const stopBtn = document.getElementById('stopComfyBtn');
  const stopRow = document.getElementById('stopComfyRow');
  if (s.comfy_pids.length) {
    cp.innerHTML = `<span class="dot"></span>Comfy PID ${s.comfy_pids.join(', ')}`;
    cp.className = 'pill pill-warn';
    cp.style.display = '';
    stopBtn.style.display = '';
    stopRow.style.display = '';
  } else {
    cp.style.display = 'none';
    stopBtn.style.display = 'none';
    stopRow.style.display = 'none';
  }

  const hp = document.getElementById('helperPill');
  if (s.helper && s.helper.alive) {
    hp.innerHTML = `<span class="dot"></span>helper warm · PID ${s.helper.pid}`;
    hp.className = 'pill pill-good';
  } else {
    hp.innerHTML = `<span class="dot"></span>helper cold`;
    hp.className = 'pill';
  }

  const qp = document.getElementById('queuePill');
  qp.innerHTML = `<span class="dot"></span>queue ${s.queue.length}${s.paused ? ' · paused' : ''}`;
  qp.className = 'pill ' + (s.paused ? 'pill-warn' : (s.queue.length ? 'pill-running' : ''));

  const jp = document.getElementById('jobPill');
  if (s.running && s.current) {
    const elapsed = Math.max(0, Math.round(s.server_now - s.current.started_ts));
    jp.innerHTML = `<span class="dot"></span>${s.current.params.label || s.current.params.mode} · ${elapsed}s`;
    jp.className = 'pill pill-running';
  } else {
    jp.innerHTML = `<span class="dot"></span>idle`;
    jp.className = 'pill';
  }

  document.getElementById('pauseBtn').textContent = s.paused ? '▶ Resume' : '⏸ Pause';

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
      `${s.current.params.mode} · ${s.current.params.width}×${s.current.params.height} · ${s.current.params.frames}f · ${s.current.params.steps} steps · <strong>${fmtMin(elapsed)}</strong> elapsed` +
      (avg ? ` / ~${fmtMin(avg)} avg` : '') +
      (lastLog ? `<br><span style="color:var(--muted)">${escapeHtml(lastLog.split(']').slice(1).join(']').trim().slice(0, 100))}</span>` : '');
  } else {
    nowCard.classList.add('idle');
    fill.style.width = '0%';
    nowCard.querySelector('.ttl').textContent = s.paused ? 'Paused' : 'Idle';
    nowCard.querySelector('.meta').textContent = s.paused
      ? 'Worker paused — current job (if any) finishes, queue waits for resume.'
      : (s.queue.length ? 'Worker about to pick up next queued job.' : 'No jobs queued.');
  }
  document.getElementById('nowMeta').textContent = s.avg_elapsed_sec ? `avg ${fmtMin(s.avg_elapsed_sec)} per job` : '';

  const log = document.getElementById('log');
  const wasNearBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 60;
  log.textContent = s.log.length ? s.log.join('\n') : 'No log yet.';
  if (wasNearBottom) log.scrollTop = log.scrollHeight;

  const ql = document.getElementById('queueList');
  if (!s.queue.length) {
    ql.innerHTML = '<li class="empty"><span></span><span>Queue empty</span><span></span><span></span></li>';
  } else {
    ql.innerHTML = s.queue.map((j, i) => `
      <li>
        <span class="pos">#${i + 1}</span>
        <span class="ttl" title="${escapeHtml(j.params.prompt)}">${escapeHtml(j.params.label || snippet(j.params.prompt, 60))}</span>
        <span class="params">${j.params.mode} · ${j.params.width}×${j.params.height} · ${j.params.frames}f</span>
        <button title="Remove" onclick="removeJob('${j.id}')">×</button>
      </li>`).join('');
  }
  document.getElementById('queueMeta').textContent = s.queue.length ? `${s.queue.length} pending · ETA ${fmtMin(s.eta_sec)}` : '';

  const hl = document.getElementById('historyList');
  if (!s.history.length) {
    hl.innerHTML = '<li class="empty"><span></span><span>No history yet</span><span></span><span></span></li>';
  } else {
    hl.innerHTML = s.history.slice(0, 12).map(j => `
      <li class="${j.status}">
        <span class="badge">${j.status}</span>
        <span class="ttl" title="${escapeHtml(j.params.prompt)}">${escapeHtml(j.params.label || snippet(j.params.prompt, 60))}</span>
        <span class="params">${fmtMin(j.elapsed_sec)} · ${j.finished_at ? j.finished_at.slice(11) : ''}</span>
        <span></span>
      </li>`).join('');
  }
  document.getElementById('historyMeta').textContent = s.history.length ? `last ${Math.min(12, s.history.length)}` : '';

  if (JSON.stringify(currentOutputs) !== JSON.stringify(s.outputs)) {
    currentOutputs = s.outputs;
    renderOutputs();
    if (!activePath && currentOutputs.length) selectOutput(currentOutputs[0].path);
    // Also refresh extend source dropdown
    const sel = document.getElementById('extendSrcSelect');
    sel.innerHTML = '<option value="">— pick from outputs below or paste a path —</option>' +
      currentOutputs.slice(0, 40).map(o =>
        `<option value="${o.path}">${o.name}</option>`).join('');
  }
  document.getElementById('filterHidden').textContent = `Hidden${s.hidden_count ? ' ('+s.hidden_count+')' : ''}`;

  // Q8 availability — enables/disables the High quality tier
  const highOpt = document.getElementById('qualityHigh');
  if (s.q8_available) {
    highOpt.disabled = false;
    highOpt.textContent = 'High — Q8 two-stage + TeaCache · ~12 min for 5s (best face fidelity)';
  } else {
    highOpt.disabled = true;
    const missing = (s.q8_missing || []).length;
    if (missing > 0 && missing < 6) {
      highOpt.textContent = `High — Q8 downloading · ${missing} file${missing > 1 ? 's' : ''} still missing`;
    } else {
      highOpt.textContent = `High — Q8 not installed (need ~25 GB at ${s.q8_path || ''})`;
    }
    if (document.getElementById('quality').value === 'high') {
      document.getElementById('quality').value = 'standard';
      applyQuality();
    }
  }
}

function setFilter(mode) {
  filterMode = mode;
  document.getElementById('filterAll').classList.toggle('active', mode === 'visible');
  document.getElementById('filterHidden').classList.toggle('active', mode === 'hidden');
  poll();
}

function renderOutputs() {
  const grid = document.getElementById('outGrid');
  if (!currentOutputs.length) {
    grid.innerHTML = '<div class="empty" style="grid-column: 1/-1">No outputs in this view.</div>';
    return;
  }
  grid.innerHTML = currentOutputs.map(o => `
    <div class="out-card${o.hidden ? ' hidden-card' : ''}${o.path === activePath ? ' active' : ''}"
         data-path="${escapeHtml(o.path)}"
         onclick="selectOutput('${escapeHtml(o.path)}')">
      <video src="/file?path=${encodeURIComponent(o.path)}#t=0.5" preload="metadata" muted></video>
      <div class="info">
        <div class="name" title="${escapeHtml(o.name)}">${escapeHtml(o.name)}</div>
        <div class="sub">${o.mtime.slice(11,16)} · ${o.size_mb.toFixed(1)} MB${o.has_sidecar ? ' · ↩' : ''}</div>
      </div>
      <div class="actions-row">
        <button onclick="event.stopPropagation(); ${o.hidden ? 'unhide' : 'hide'}('${escapeHtml(o.path)}')">${o.hidden ? '👁 Show' : '⊘ Hide'}</button>
        <button onclick="event.stopPropagation(); useAsExtendSourcePath('${escapeHtml(o.path)}')">⏭ Extend</button>
      </div>
    </div>
  `).join('');
}

function selectOutput(path) {
  activePath = path;
  document.querySelectorAll('.out-card').forEach(el => el.classList.toggle('active', el.dataset.path === path));
  document.getElementById('player').src = '/file?path=' + encodeURIComponent(path);
  const o = currentOutputs.find(x => x.path === path);
  document.getElementById('loadParamsBtn').disabled = !(o && o.has_sidecar);
  document.getElementById('outputMeta').innerHTML = o ? `<strong>${escapeHtml(o.name)}</strong> · ${o.mtime} · ${o.size_mb.toFixed(1)} MB` : '';
}

async function hide(path) { await fetch('/output/hide?path=' + encodeURIComponent(path), {method: 'POST'}); currentOutputs = []; poll(); }
async function unhide(path) { await fetch('/output/show?path=' + encodeURIComponent(path), {method: 'POST'}); currentOutputs = []; poll(); }

function useAsExtendSourcePath(path) {
  document.getElementById('mode').value = 'extend';
  document.getElementById('video_path').value = path;
  document.getElementById('extendSrcSelect').value = path;
  updateDerived();
  window.scrollTo(0, 0);
}
function useAsExtendSource() {
  if (!activePath) { alert('Pick an output first.'); return; }
  useAsExtendSourcePath(activePath);
}

async function loadParams() {
  if (!activePath) return;
  const r = await fetch('/sidecar?path=' + encodeURIComponent(activePath));
  if (!r.ok) return;
  const data = await r.json();
  const p = data.params;
  document.getElementById('mode').value = p.mode || 't2v';
  document.getElementById('prompt').value = p.prompt;
  if (p.width) document.getElementById('width').value = p.width;
  if (p.height) document.getElementById('height').value = p.height;
  if (p.frames) {
    document.getElementById('frames').value = p.frames;
    document.getElementById('duration').value = framesToDuration(p.frames);
  }
  if (p.steps) document.getElementById('steps').value = p.steps;
  if (p.seed) document.getElementById('seed').value = p.seed;
  if (p.image) document.getElementById('image').value = p.image;
  if (p.audio) document.getElementById('audio').value = p.audio;
  document.getElementById('enhance').checked = !!p.enhance;
  document.getElementById('stop_comfy').checked = !!p.stop_comfy;
  document.getElementById('preset_label').value = p.label || '';
  // Try to pick aspect that matches
  for (const [k, a] of Object.entries(ASPECTS)) {
    if (a.w === p.width && a.h === p.height) { document.getElementById('aspect').value = k; break; }
  }
  updateDerived();
}

async function removeJob(id) { await fetch('/queue/remove?id=' + encodeURIComponent(id), {method: 'POST'}); poll(); }

async function togglePause() {
  const r = await fetch('/status'); const s = await r.json();
  await api(s.paused ? '/queue/resume' : '/queue/pause', 'POST'); poll();
}

async function queueBatch() {
  const fd = new FormData(document.getElementById('genForm'));
  fd.set('prompts', document.getElementById('batchPrompts').value);
  const r = await api('/queue/batch', 'POST', fd);
  if (r && r.error) { alert('Batch error: ' + r.error); return; }
  if (r && r.added) { document.getElementById('batchPrompts').value = ''; poll(); }
}

document.getElementById('genForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  await api('/queue/add', 'POST', fd);
  poll();
});

setInterval(poll, 1500);
poll();
updateDerived();
setDuration(5); // initial duration
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
