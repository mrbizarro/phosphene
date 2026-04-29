#!/usr/bin/env python3.11
"""LTX warm helper — long-lived process holding MLX pipelines in memory.

Reads JSON-line jobs from stdin, emits JSON-line events to stdout.

Actions:
  generate  — T2V or I2V (auto-resizes images to target dims via PIL cover-crop)
  extend    — chain a clip by N latent frames (uses ExtendPipeline)
  ping      — returns pong
  exit      — graceful shutdown

Auto-exits after LTX_IDLE_TIMEOUT seconds idle.
"""
from __future__ import annotations

import io
import json
import os
import random
import sys
import threading
import time
import traceback

# ---- config ------------------------------------------------------------------
# All paths come from env vars set by the panel. If LTX_GEMMA isn't set, the
# pipeline falls back to downloading the HF model id, which works first-run.
MODEL_ID = os.environ.get("LTX_MODEL", "dgrauet/ltx-2.3-mlx-q4")
GEMMA_PATH = os.environ.get("LTX_GEMMA", "mlx-community/gemma-3-12b-it-4bit")
IDLE_TIMEOUT = int(os.environ.get("LTX_IDLE_TIMEOUT", "1800"))
LOW_MEMORY = os.environ.get("LTX_LOW_MEMORY", "true").lower() in ("true", "1", "yes")

_real_stdout = sys.stdout
_emit_lock = threading.Lock()


def emit(event: dict) -> None:
    try:
        with _emit_lock:
            _real_stdout.write(json.dumps(event) + "\n")
            _real_stdout.flush()
    except Exception:
        pass


class LineEmitter(io.TextIOBase):
    def __init__(self):
        self.buf = ""
        self.lock = threading.Lock()

    def writable(self):
        return True

    def write(self, s):
        if not s:
            return 0
        with self.lock:
            self.buf += s
            while True:
                idx_n = self.buf.find("\n")
                idx_r = self.buf.find("\r")
                idxs = [i for i in (idx_n, idx_r) if i != -1]
                if not idxs:
                    break
                idx = min(idxs)
                line = self.buf[:idx].strip()
                self.buf = self.buf[idx + 1:]
                if line:
                    emit({"event": "log", "line": line})
        return len(s)

    def flush(self):
        pass


sys.stdout = LineEmitter()
sys.stderr = LineEmitter()

# ---- idle reaper -------------------------------------------------------------
_last_activity = time.time()
_is_busy = False  # set during active generation; reaper skips while True


def idle_reaper():
    while True:
        time.sleep(15)
        if _is_busy:
            continue
        if time.time() - _last_activity > IDLE_TIMEOUT:
            emit({"event": "exit", "reason": "idle"})
            os._exit(0)


threading.Thread(target=idle_reaper, daemon=True).start()

# ---- pipelines (lazy) --------------------------------------------------------
_t2v_pipe = None
_i2v_pipe = None
_extend_pipe = None
_hq_pipe = None          # TwoStageHQPipeline (Q8, res_2s + CFG, optional TeaCache)
_hq_model_dir = None     # remember which model the HQ pipe was built against
_pipe_lock = threading.Lock()


def release_pipelines(keep_kind=None):
    """Free every loaded pipeline except the one matching keep_kind.

    Each pipeline holds ~22 GB (Q4) or ~30 GB (Q8 dev) of weights. Holding
    two or more simultaneously on a 64 GB Mac OOMs the helper. Only one
    family stays resident at a time — switching mode reloads, but renders
    actually finish instead of getting SIGKILL'd by macOS.

    keep_kind ∈ {'t2v', 'i2v', 'extend', 'hq', 'keyframe'} or None (free all).
    Caller must hold _pipe_lock.
    """
    global _t2v_pipe, _i2v_pipe, _extend_pipe, _hq_pipe, _kf_pipe, _hq_model_dir, _kf_model_dir
    try:
        from ltx_core_mlx.utils.memory import aggressive_cleanup
    except Exception:
        aggressive_cleanup = lambda: None

    freed = []
    if keep_kind != "t2v" and _t2v_pipe is not None:
        _t2v_pipe = None; freed.append("T2V")
    if keep_kind != "i2v" and _i2v_pipe is not None:
        _i2v_pipe = None; freed.append("I2V")
    if keep_kind != "extend" and _extend_pipe is not None:
        _extend_pipe = None; freed.append("Extend")
    if keep_kind != "hq" and _hq_pipe is not None:
        _hq_pipe = None; _hq_model_dir = None; freed.append("HQ")
    if keep_kind != "keyframe" and _kf_pipe is not None:
        _kf_pipe = None; _kf_model_dir = None; freed.append("Keyframe")
    if freed:
        aggressive_cleanup()
        emit({"event": "log", "line": f"Released pipelines: {', '.join(freed)} (freeing RAM before next load)"})


def get_pipe(kind: str):
    """kind in {'t2v','i2v','extend'}"""
    global _t2v_pipe, _i2v_pipe, _extend_pipe
    from ltx_pipelines_mlx import TextToVideoPipeline, ImageToVideoPipeline, ExtendPipeline

    with _pipe_lock:
        # Free any other pipelines before loading a new one — strict
        # one-pipeline-at-a-time policy keeps memory bounded.
        release_pipelines(keep_kind=kind)
        if kind == "i2v":
            if _i2v_pipe is None:
                emit({"event": "log", "line": "Loading I2V pipeline (first job is the slow one)..."})
                _i2v_pipe = ImageToVideoPipeline(
                    model_dir=MODEL_ID, gemma_model_id=GEMMA_PATH, low_memory=LOW_MEMORY,
                )
            return _i2v_pipe
        if kind == "extend":
            if _extend_pipe is None:
                emit({"event": "log", "line": "Loading Extend pipeline (heavier — uses dev transformer)..."})
                _extend_pipe = ExtendPipeline(
                    model_dir=MODEL_ID, gemma_model_id=GEMMA_PATH, low_memory=LOW_MEMORY,
                )
            return _extend_pipe
        # t2v
        if _t2v_pipe is None:
            emit({"event": "log", "line": "Loading T2V pipeline (first job is the slow one)..."})
            _t2v_pipe = TextToVideoPipeline(
                model_dir=MODEL_ID, gemma_model_id=GEMMA_PATH, low_memory=LOW_MEMORY,
            )
        return _t2v_pipe


def get_hq_pipe(model_dir: str):
    """Returns the TwoStageHQPipeline lazily — Q8 model, res_2s sampler, CFG anchor.

    Same class handles both T2V (image=None) and I2V via the `image` kwarg of
    `generate_and_save`. We rebuild the pipe if the requested model_dir changes
    (e.g. user swapped Q8 for a different quant).
    """
    global _hq_pipe, _hq_model_dir
    from ltx_pipelines_mlx.ti2vid_two_stages_hq import TwoStageHQPipeline

    with _pipe_lock:
        release_pipelines(keep_kind="hq")
        if _hq_pipe is None or _hq_model_dir != model_dir:
            emit({"event": "log", "line": f"Loading HQ pipeline (Q8 dev model — {model_dir})..."})
            _hq_pipe = TwoStageHQPipeline(
                model_dir=model_dir, gemma_model_id=GEMMA_PATH, low_memory=LOW_MEMORY,
            )
            _hq_model_dir = model_dir
        return _hq_pipe


# Keyframe (FFLF) pipeline — two images locked at frame 0 + frame N-1, model
# interpolates between. Uses Q8 dev transformer + distilled LoRA stage 2.
_kf_pipe = None
_kf_model_dir = None


def get_kf_pipe(model_dir: str):
    """Returns the KeyframeInterpolationPipeline lazily.

    Keyframe REQUIRES explicit dev_transformer + distilled_lora at init time.
    The distilled-only path "hallucinates unrelated content during
    interpolation" — pipeline raises if you skip these. Names match the files
    inside dgrauet/ltx-2.3-mlx-q8.
    """
    global _kf_pipe, _kf_model_dir
    from ltx_pipelines_mlx.keyframe_interpolation import KeyframeInterpolationPipeline

    with _pipe_lock:
        release_pipelines(keep_kind="keyframe")
        if _kf_pipe is None or _kf_model_dir != model_dir:
            emit({"event": "log", "line": f"Loading Keyframe pipeline (Q8 dev model — {model_dir})..."})
            _kf_pipe = KeyframeInterpolationPipeline(
                model_dir=model_dir,
                gemma_model_id=GEMMA_PATH,
                low_memory=LOW_MEMORY,
                dev_transformer="transformer-dev.safetensors",
                distilled_lora="ltx-2.3-22b-distilled-lora-384.safetensors",
                distilled_lora_strength=1.0,
            )
            _kf_model_dir = model_dir
        return _kf_pipe


# ---- image preprocessing -----------------------------------------------------

def cover_crop_to_size(src_path: str, w: int, h: int) -> str:
    """Cover-crop and resize to exactly w×h. Saves PNG and returns its path."""
    from PIL import Image
    out_path = f"/tmp/ltx_helper_image_{os.getpid()}_{int(time.time()*1000)}.png"
    img = Image.open(src_path).convert("RGB")
    src_w, src_h = img.size
    if (src_w, src_h) == (w, h):
        img.save(out_path)
        return out_path
    src_ratio = src_w / src_h
    dst_ratio = w / h
    if src_ratio > dst_ratio:
        new_w = int(round(src_h * dst_ratio))
        left = (src_w - new_w) // 2
        img = img.crop((left, 0, left + new_w, src_h))
    elif src_ratio < dst_ratio:
        new_h = int(round(src_w / dst_ratio))
        top = (src_h - new_h) // 2
        img = img.crop((0, top, src_w, top + new_h))
    if img.size != (w, h):
        img = img.resize((w, h), Image.LANCZOS)
    img.save(out_path)
    emit({"event": "log", "line": f"Resized image {src_w}x{src_h} → {w}x{h} (cover-crop)"})
    return out_path


# ---- ready -------------------------------------------------------------------
emit({
    "event": "ready",
    "model": MODEL_ID,
    "gemma": GEMMA_PATH,
    "low_memory": LOW_MEMORY,
    "idle_timeout_sec": IDLE_TIMEOUT,
})


# ---- main loop ---------------------------------------------------------------
for line in sys.__stdin__:
    line = line.strip()
    if not line:
        continue
    try:
        msg = json.loads(line)
    except Exception as exc:
        emit({"event": "error", "error": f"bad json: {exc}"})
        continue

    _last_activity = time.time()
    action = msg.get("action")

    if action == "exit":
        emit({"event": "exit", "reason": "shutdown"})
        os._exit(0)

    if action == "ping":
        emit({"event": "pong"})
        continue

    if action == "generate":
        job_id = msg.get("id", "?")
        p = msg.get("params", {}) or {}
        mode = p.get("mode", "t2v")
        if mode not in ("t2v", "i2v", "i2v_clean_audio"):
            emit({"event": "error", "id": job_id, "error": f"unsupported mode: {mode}"})
            continue
        needs_image = mode != "t2v"
        seed = int(p.get("seed", -1))
        if seed == -1:
            seed = random.randint(0, 2**31 - 1)

        _is_busy = True
        try:
            t0 = time.time()
            pipe = get_pipe("i2v" if needs_image else "t2v")

            kwargs = dict(
                prompt=p["prompt"],
                output_path=p["output_path"],
                height=int(p["height"]),
                width=int(p["width"]),
                num_frames=int(p["frames"]),
                seed=seed,
                num_steps=int(p.get("steps", 8)),
            )
            if needs_image:
                src_image = p.get("image")
                if src_image:
                    if not os.path.exists(src_image):
                        raise RuntimeError(f"image not found: {src_image}")
                    # Pass the source path straight through. The pipeline's
                    # prepare_image_for_encoding does its own cover-crop + LANCZOS
                    # at the target W×H. Our previous pre-resize round-tripped
                    # through PNG and added quality loss for zero benefit.
                    kwargs["image"] = src_image
                    try:
                        from PIL import Image as _Image
                        _w, _h = _Image.open(src_image).size
                        emit({"event": "log", "line": f"Image {_w}x{_h} → pipeline will cover-crop to {kwargs['width']}x{kwargs['height']}"})
                    except Exception:
                        pass
                else:
                    kwargs["image"] = None

            out_path = pipe.generate_and_save(**kwargs)
            elapsed = round(time.time() - t0, 2)
            _last_activity = time.time()
            emit({
                "event": "done", "id": job_id,
                "output": str(out_path), "elapsed_sec": elapsed,
                "seed_used": seed,
            })
        except Exception as exc:
            _last_activity = time.time()
            emit({"event": "error", "id": job_id, "error": str(exc), "trace": traceback.format_exc()})
        finally:
            _is_busy = False
        continue

    if action == "extend":
        job_id = msg.get("id", "?")
        p = msg.get("params", {}) or {}
        seed = int(p.get("seed", -1))
        if seed == -1:
            seed = random.randint(0, 2**31 - 1)
        _is_busy = True
        try:
            t0 = time.time()
            pipe = get_pipe("extend")
            video_path = p["video_path"]
            if not os.path.exists(video_path):
                raise RuntimeError(f"source video not found: {video_path}")
            video_lat, audio_lat = pipe.extend_from_video(
                prompt=p["prompt"],
                video_path=video_path,
                extend_frames=int(p.get("extend_frames", 5)),
                direction=p.get("direction", "after"),
                seed=seed,
                num_steps=int(p.get("steps", 30)),
            )
            # Decode + save (mirrors the CLI _decode_and_save)
            from ltx_core_mlx.utils.memory import aggressive_cleanup
            if pipe.low_memory:
                pipe.dit = None
                pipe.text_encoder = None
                pipe.feature_extractor = None
                pipe._loaded = False
                aggressive_cleanup()
            pipe._load_decoders()
            pipe._decode_and_save_video(video_lat, audio_lat, p["output_path"])
            elapsed = round(time.time() - t0, 2)
            _last_activity = time.time()
            emit({
                "event": "done", "id": job_id,
                "output": p["output_path"], "elapsed_sec": elapsed,
                "seed_used": seed,
            })
        except Exception as exc:
            _last_activity = time.time()
            emit({"event": "error", "id": job_id, "error": str(exc), "trace": traceback.format_exc()})
        finally:
            _is_busy = False
        continue

    if action == "generate_hq":
        # Q8 two-stage HQ + optional TeaCache. Same TwoStageHQPipeline handles
        # T2V (image=None) and I2V via the `image` kwarg.
        job_id = msg.get("id", "?")
        p = msg.get("params", {}) or {}
        model_dir = p.get("model_dir") or MODEL_ID  # fallback if user forgot
        seed = int(p.get("seed", -1))
        if seed == -1:
            seed = random.randint(0, 2**31 - 1)
        _is_busy = True
        try:
            t0 = time.time()
            pipe = get_hq_pipe(model_dir)
            kwargs = dict(
                prompt=p["prompt"],
                output_path=p["output_path"],
                height=int(p["height"]),
                width=int(p["width"]),
                num_frames=int(p["frames"]),
                seed=seed,
                stage1_steps=int(p.get("stage1_steps", 15)),
                stage2_steps=int(p.get("stage2_steps", 3)),
                cfg_scale=float(p.get("cfg_scale", 3.0)),
                stg_scale=float(p.get("stg_scale", 1.0)),
                enable_teacache=bool(p.get("enable_teacache", True)),
                teacache_thresh=float(p.get("teacache_thresh", 1.0)),
            )
            img = p.get("image")
            if img:
                if not os.path.exists(img):
                    raise RuntimeError(f"image not found: {img}")
                kwargs["image"] = img
                emit({"event": "log", "line": f"HQ I2V — pipeline will cover-crop image to {kwargs['width']}x{kwargs['height']}"})
            out_path = pipe.generate_and_save(**kwargs)
            elapsed = round(time.time() - t0, 2)
            _last_activity = time.time()
            emit({
                "event": "done", "id": job_id,
                "output": str(out_path), "elapsed_sec": elapsed,
                "seed_used": seed,
            })
        except Exception as exc:
            _last_activity = time.time()
            emit({"event": "error", "id": job_id, "error": str(exc), "trace": traceback.format_exc()})
        finally:
            _is_busy = False
        continue

    if action == "generate_keyframe":
        # FFLF — two images anchored at frame 0 and frame N-1, model interpolates.
        # Like HQ this uses the Q8 dev transformer + two-stage refine.
        job_id = msg.get("id", "?")
        p = msg.get("params", {}) or {}
        model_dir = p.get("model_dir") or MODEL_ID
        seed = int(p.get("seed", -1))
        if seed == -1:
            seed = random.randint(0, 2**31 - 1)
        _is_busy = True
        try:
            t0 = time.time()
            for k in ("start_image", "end_image"):
                img = p.get(k)
                if not img or not os.path.exists(img):
                    raise RuntimeError(f"{k} not found: {img}")
            pipe = get_kf_pipe(model_dir)
            num_frames = int(p["frames"])
            kwargs = dict(
                prompt=p["prompt"],
                output_path=p["output_path"],
                keyframe_images=[p["start_image"], p["end_image"]],
                keyframe_indices=[0, num_frames - 1],
                height=int(p["height"]),
                width=int(p["width"]),
                num_frames=num_frames,
                fps=24,
                seed=seed,
                stage1_steps=int(p.get("stage1_steps", 15)),
                stage2_steps=int(p.get("stage2_steps", 3)),
                cfg_scale=float(p.get("cfg_scale", 3.0)),
            )
            emit({"event": "log", "line": f"Keyframe FFLF — frames=[0, {num_frames-1}], pipeline cover-crops both to {kwargs['width']}x{kwargs['height']}"})
            out_path = pipe.generate_and_save(**kwargs)
            elapsed = round(time.time() - t0, 2)
            _last_activity = time.time()
            emit({
                "event": "done", "id": job_id,
                "output": str(out_path), "elapsed_sec": elapsed,
                "seed_used": seed,
            })
        except Exception as exc:
            _last_activity = time.time()
            emit({"event": "error", "id": job_id, "error": str(exc), "trace": traceback.format_exc()})
        finally:
            _is_busy = False
        continue

    emit({"event": "error", "error": f"unknown action: {action}"})

emit({"event": "exit", "reason": "stdin_closed"})
