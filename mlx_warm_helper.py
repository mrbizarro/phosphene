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
import math
import os
import random
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

# ---- config ------------------------------------------------------------------
# All paths come from env vars set by the panel. If LTX_GEMMA isn't set, the
# pipeline falls back to downloading the HF model id, which works first-run.
MODEL_ID = os.environ.get("LTX_MODEL", "dgrauet/ltx-2.3-mlx-q4")
GEMMA_PATH = os.environ.get("LTX_GEMMA", "mlx-community/gemma-3-12b-it-4bit")
IDLE_TIMEOUT = int(os.environ.get("LTX_IDLE_TIMEOUT", "1800"))
LOW_MEMORY = os.environ.get("LTX_LOW_MEMORY", "true").lower() in ("true", "1", "yes")
MODEL_UPSCALE_ENABLED = os.environ.get("LTX_ENABLE_MODEL_UPSCALE", "").lower() in ("1", "true", "yes", "on")

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

# ---- exit / signal tracing ---------------------------------------------------
# When users hit "no error, just stopped" silent crashes (cocktailpeanut on
# I2V, ~10s in), we currently can't tell if the helper exited cleanly, was
# SIGTERM'd by the panel, was SIGKILL'd by jetsam (macOS OOM), or hit a
# C-level fault in MLX/Metal. atexit fires on graceful Python exit (which
# also runs after our own emit({"exit"})), and the SIGTERM handler emits
# before the raise/exit. Neither catches SIGKILL — that's the diagnostic
# fingerprint. If the panel sees no exit/sigterm event AND the pipe closes,
# we KNOW it was SIGKILL (jetsam OOM) or a segfault.
import atexit, signal
_exit_emitted = False

def _emit_exit(reason: str) -> None:
    global _exit_emitted
    if _exit_emitted:
        return
    _exit_emitted = True
    try:
        emit({"event": "exit", "reason": reason})
    except Exception:
        pass

atexit.register(lambda: _emit_exit("python_normal_exit"))

def _sigterm_handler(signum, frame):
    _emit_exit(f"sigterm({signum})")
    sys.exit(0)

# SIGTERM is what the panel sends on /helper/restart and at panel shutdown.
# SIGINT is Ctrl+C from a user running the helper directly. Both exit cleanly.
# SIGKILL can't be caught — that's by design, and is the OOM fingerprint.
for _sig in (signal.SIGTERM, signal.SIGINT):
    try:
        signal.signal(_sig, _sigterm_handler)
    except (ValueError, OSError):
        pass

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
    global _gemma_lm
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
    # Always free Gemma LanguageModel when releasing for any pipeline —
    # ~6 GB persistent that competes with the dev transformer's headroom.
    # Re-loaded on demand by the next enhance call (one-time ~10s cost).
    if keep_kind != "gemma_lm" and _gemma_lm is not None:
        _gemma_lm = None; freed.append("GemmaLM")
    if freed:
        aggressive_cleanup()
        emit({"event": "log", "line": f"Released pipelines: {', '.join(freed)} (freeing RAM before next load)"})


# Track which LoRA set is fused into each cached pipeline. LoRAs are
# fused INTO the model weights at load time (apply_loras in
# ltx_core_mlx.loader.fuse_loras), so changing the LoRA set requires
# reloading the pipeline. We invalidate the cache by LoRA-set fingerprint.
_t2v_lora_key: tuple | None = None
_i2v_lora_key: tuple | None = None
_extend_lora_key: tuple | None = None


def _lora_fingerprint(loras: list[dict] | None) -> tuple:
    """Stable hashable representation of a LoRA list. Order-insensitive
    so [{a,1},{b,2}] and [{b,2},{a,1}] hash to the same set — fusing
    is commutative."""
    if not loras:
        return ()
    return tuple(sorted(
        (str(l.get("path", "")), float(l.get("strength", 1.0)))
        for l in loras
    ))


def _resolve_lora_path(path: str) -> str:
    """Resolve a LoRA path to a local .safetensors file.

    The upstream `_pending_loras` hook calls SafetensorsStateDictLoader
    which calls `mx.load(path)` — that only accepts a local filesystem
    path, not a HuggingFace repo id. So when the panel sends a path that
    looks like an HF id (`<org>/<repo>` without a file extension), we
    resolve it here via `snapshot_download` and pick the largest
    .safetensors in the resulting directory (the LoRA weights file).

    Cached files land in ~/.cache/huggingface/, so the second job using
    the same LoRA hits a no-op verify pass instead of a re-download."""
    p = str(path)
    # Already a local file
    if os.path.isfile(p):
        return p
    # Looks like a filesystem path that didn't resolve. Filesystem paths
    # are absolute (start with /) OR explicitly have a `.safetensors`
    # extension. Bail with a clear error so the user knows the file
    # they pointed at isn't on disk.
    if p.startswith("/") or p.lower().endswith(".safetensors"):
        raise FileNotFoundError(f"LoRA file not found: {p}")
    # Looks like an HF repo id (`<org>/<repo>` form). Must contain exactly
    # one forward slash and no path-traversal chars.
    if p.count("/") != 1 or ".." in p:
        raise FileNotFoundError(f"LoRA path neither a file nor an HF id: {p}")
    emit({"event": "log",
          "line": f"  resolving HF LoRA: {p} (snapshot_download …)"})
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
    except ImportError as exc:
        raise RuntimeError(
            f"need huggingface_hub to resolve HF LoRA {p}: {exc}"
        ) from exc
    try:
        repo_dir = snapshot_download(repo_id=p, allow_patterns=["*.safetensors"])
    except GatedRepoError as exc:
        # Most Lightricks LoRAs are gated — they require accepting a
        # license on the model page AND an HF token authenticated with
        # an account that has accepted. Translate the upstream traceback
        # into something the user can act on.
        raise RuntimeError(
            f"This LoRA is gated on Hugging Face. To use it: "
            f"(1) visit https://huggingface.co/{p} and click 'Agree and "
            f"access repository' to accept the license. "
            f"(2) get a token at https://huggingface.co/settings/tokens "
            f"with read access. "
            f"(3) run `hf auth login` in Terminal and paste the token. "
            f"(4) restart the panel."
        ) from None
    except RepositoryNotFoundError:
        raise RuntimeError(
            f"Hugging Face repo not found: {p}. Check the repo id."
        ) from None
    except Exception as exc:
        # Catch the generic 401 too — `snapshot_download` raises a
        # different exception class for "not authenticated" (no token at
        # all) than for "authenticated but not approved for this gated
        # repo" (GatedRepoError). The string-match keeps both paths
        # consistent for the user.
        msg = str(exc)
        if "401" in msg or "gated" in msg.lower():
            raise RuntimeError(
                f"Could not access HF LoRA {p} (401 Unauthorized). "
                f"Accept the license at https://huggingface.co/{p} and "
                f"run `hf auth login` in Terminal to set up your token, "
                f"then restart the panel."
            ) from None
        raise
    candidates = []
    for name in os.listdir(repo_dir):
        if name.lower().endswith(".safetensors"):
            full = os.path.join(repo_dir, name)
            try:
                size = os.path.getsize(full)
            except OSError:
                size = 0
            candidates.append((size, full))
    if not candidates:
        raise FileNotFoundError(
            f"no .safetensors files found in HF repo {p}"
        )
    # Heuristic: pick the LARGEST .safetensors. LoRA repos sometimes
    # ship smaller "auxiliary" files (e.g. scene embeddings) alongside
    # the main weights — the main file is always biggest.
    candidates.sort(reverse=True)
    chosen = candidates[0][1]
    emit({"event": "log",
          "line": f"  resolved {p} -> {os.path.basename(chosen)} "
                  f"({candidates[0][0] // (1024*1024)} MB)"})
    return chosen


def _attach_loras(pipe, loras: list[dict] | None) -> None:
    """Set _pending_loras on a freshly-constructed pipeline. The upstream
    base class checks this attribute inside load() and fuses the LoRA
    deltas into the transformer weights before quantization. Path on the
    wire may be a local file OR an HF repo id; we resolve HF ids to a
    local .safetensors here so the loader (mx.load) sees an absolute
    path it can actually open."""
    if not loras:
        return
    pairs = []
    for l in loras:
        path = _resolve_lora_path(str(l["path"]))
        strength = float(l.get("strength", 1.0))
        pairs.append((path, strength))
        emit({"event": "log",
              "line": f"  + LoRA queued: {os.path.basename(path)} "
                      f"(strength {strength:.2f})"})
    pipe._pending_loras = pairs


def get_pipe(kind: str, loras: list[dict] | None = None):
    """kind in {'t2v','i2v','extend'}; loras is an optional list of
    {path, strength} dicts. When the requested LoRA set differs from
    the cached pipeline's, the pipeline is rebuilt — LoRA fusion is a
    one-shot weight transformation, not a runtime toggle."""
    global _t2v_pipe, _i2v_pipe, _extend_pipe
    global _t2v_lora_key, _i2v_lora_key, _extend_lora_key
    from ltx_pipelines_mlx import TextToVideoPipeline, ImageToVideoPipeline, ExtendPipeline

    fp = _lora_fingerprint(loras)

    with _pipe_lock:
        # Free any other pipelines before loading a new one — strict
        # one-pipeline-at-a-time policy keeps memory bounded.
        release_pipelines(keep_kind=kind)
        if kind == "i2v":
            if _i2v_pipe is None or _i2v_lora_key != fp:
                if _i2v_pipe is not None and _i2v_lora_key != fp:
                    emit({"event": "log",
                          "line": f"LoRA set changed; reloading I2V pipeline."})
                    _i2v_pipe = None
                emit({"event": "log",
                      "line": "Loading I2V pipeline (first job is the slow one)..."})
                pipe = ImageToVideoPipeline(
                    model_dir=MODEL_ID, gemma_model_id=GEMMA_PATH, low_memory=LOW_MEMORY,
                )
                _attach_loras(pipe, loras)
                _i2v_pipe = pipe
                _i2v_lora_key = fp
            return _i2v_pipe
        if kind == "extend":
            if _extend_pipe is None or _extend_lora_key != fp:
                if _extend_pipe is not None and _extend_lora_key != fp:
                    emit({"event": "log",
                          "line": f"LoRA set changed; reloading Extend pipeline."})
                    _extend_pipe = None
                emit({"event": "log",
                      "line": "Loading Extend pipeline (heavier — uses dev transformer)..."})
                pipe = ExtendPipeline(
                    model_dir=MODEL_ID, gemma_model_id=GEMMA_PATH, low_memory=LOW_MEMORY,
                )
                _attach_loras(pipe, loras)
                _extend_pipe = pipe
                _extend_lora_key = fp
            return _extend_pipe
        # t2v
        if _t2v_pipe is None or _t2v_lora_key != fp:
            if _t2v_pipe is not None and _t2v_lora_key != fp:
                emit({"event": "log",
                      "line": f"LoRA set changed; reloading T2V pipeline."})
                _t2v_pipe = None
            emit({"event": "log",
                  "line": "Loading T2V pipeline (first job is the slow one)..."})
            pipe = TextToVideoPipeline(
                model_dir=MODEL_ID, gemma_model_id=GEMMA_PATH, low_memory=LOW_MEMORY,
            )
            _attach_loras(pipe, loras)
            _t2v_pipe = pipe
            _t2v_lora_key = fp
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


# ---- prompt enhancement (Gemma language model) ------------------------------
# Separate from the pipeline's TextEncoder wrapper — same weights file, but
# the LanguageModel class supports `.enhance_t2v(prompt, seed)` /
# `.enhance_i2v(prompt, seed)` for prompt rewriting. Loaded lazily on first
# enhance request. Held warm across calls; freed by `release_pipelines`
# when a render starts to keep memory below the 64 GB ceiling.
_gemma_lm = None


def get_gemma_lm():
    global _gemma_lm
    if _gemma_lm is None:
        from ltx_core_mlx.text_encoders.gemma.encoders.base_encoder import GemmaLanguageModel
        emit({"event": "log", "line": "Loading Gemma language model for prompt enhancement (~10-15s)…"})
        with _pipe_lock:
            # Free any active pipeline first — Gemma is ~6 GB, the dev
            # transformer is ~12-19 GB, having both resident risks pushing
            # us past 64 GB on standard tier.
            release_pipelines(keep_kind=None)
            _gemma_lm = GemmaLanguageModel()
            _gemma_lm.load(GEMMA_PATH)
        emit({"event": "log", "line": "Gemma loaded — subsequent enhances will be fast."})
    return _gemma_lm


def free_gemma_lm():
    global _gemma_lm
    if _gemma_lm is not None:
        _gemma_lm = None
        try:
            from ltx_core_mlx.utils.memory import aggressive_cleanup
            aggressive_cleanup()
        except Exception:
            pass


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


def _free_pipe_for_decode(pipe):
    """Release generation-only modules before VAE/audio decode.

    Upstream TextToVideoPipeline.generate_and_save() already does this, but
    ImageToVideoPipeline.generate_and_save() currently skips the cleanup before
    _decode_and_save_video(). On 10s 1280x704 I2V that leaves the 10.5GB DiT
    resident during VAE decode and can turn the apparent "last step" into a
    multi-minute memory-pressure stall. Keep the policy here so T2V/I2V behave
    identically from the panel.
    """
    if getattr(pipe, "low_memory", False):
        pipe.dit = None
        pipe.text_encoder = None
        pipe.feature_extractor = None
        pipe._loaded = False
        try:
            from ltx_core_mlx.utils.memory import aggressive_cleanup
            aggressive_cleanup()
        except Exception:
            pass


def _generate_latents(pipe, *, needs_image: bool, kwargs: dict):
    if needs_image:
        return pipe.generate_from_image(
            prompt=kwargs["prompt"],
            image=kwargs.get("image"),
            height=kwargs["height"],
            width=kwargs["width"],
            num_frames=kwargs["num_frames"],
            seed=kwargs["seed"],
            num_steps=kwargs["num_steps"],
        )
    return pipe.generate(
        prompt=kwargs["prompt"],
        height=kwargs["height"],
        width=kwargs["width"],
        num_frames=kwargs["num_frames"],
        seed=kwargs["seed"],
        num_steps=kwargs["num_steps"],
    )


# ---- LTX 2.3 spatial latent upscaler (Y1.021+) ------------------------------
# Optional model-based ×2 upscale that runs on the video latent BEFORE the VAE
# decode, giving real detail recovery instead of the ffmpeg Lanczos resize that
# the panel's lightweight export path uses. The model file is a 1 GB
# safetensors under mlx_models/ltx-2.3-mlx-q8/. This path is intentionally
# disabled in public builds unless LTX_ENABLE_MODEL_UPSCALE=1 because the
# doubled latent + VAE decode peak can freeze 64 GB Macs under pressure.
# We hand-roll the loader rather than instantiating the HQ pipeline because
# we only want the upsampler — not the Q8 dev transformer that costs ~25 GB.
_UPSCALER_CACHE = None
_UPSCALER_CACHE_DIR = None


def _upscaler_dir() -> Path:
    # The upscaler weights live in the Q8 weights folder (HF repo organization),
    # but downloading them is independent of the full Q8 bundle — install.js
    # pulls the single safetensors. LTX_Q8_LOCAL is set by the panel.
    explicit = os.environ.get("LTX_Q8_LOCAL")
    if explicit:
        return Path(explicit)
    # Fall back to MODEL_ID's sibling dir (LTX_MODEL points at the Q4 dir
    # under mlx_models/ at install time, so swap the trailing folder).
    q4 = Path(MODEL_ID)
    if q4.is_dir():
        return q4.parent / "ltx-2.3-mlx-q8"
    return Path("mlx_models/ltx-2.3-mlx-q8")


def upscaler_available() -> bool:
    return (_upscaler_dir() / "spatial_upscaler_x2_v1_1.safetensors").exists()


def _load_upscaler():
    """Lazy-load + cache the spatial latent upscaler. Returns the model
    instance, or None if the weights aren't on disk (caller falls back to
    ffmpeg Lanczos)."""
    global _UPSCALER_CACHE, _UPSCALER_CACHE_DIR
    model_dir = _upscaler_dir()
    if _UPSCALER_CACHE is not None and _UPSCALER_CACHE_DIR == str(model_dir):
        return _UPSCALER_CACHE
    weights_path = model_dir / "spatial_upscaler_x2_v1_1.safetensors"
    if not weights_path.exists():
        return None
    from ltx_core_mlx.model.upsampler import LatentUpsampler
    from ltx_core_mlx.utils.weights import load_split_safetensors
    config_path = model_dir / "spatial_upscaler_x2_v1_1_config.json"
    if config_path.exists():
        cfg = json.loads(config_path.read_text()).get("config", {})
        ups = LatentUpsampler.from_config(cfg)
    else:
        ups = LatentUpsampler()
    sd = load_split_safetensors(weights_path, prefix="spatial_upscaler_x2_v1_1.")
    ups.load_weights(list(sd.items()))
    _UPSCALER_CACHE = ups
    _UPSCALER_CACHE_DIR = str(model_dir)
    return ups


def _free_upscaler():
    global _UPSCALER_CACHE, _UPSCALER_CACHE_DIR
    _UPSCALER_CACHE = None
    _UPSCALER_CACHE_DIR = None
    try:
        from ltx_core_mlx.utils.memory import aggressive_cleanup
        aggressive_cleanup()
    except Exception:
        pass


def _model_upscale_video_latent(pipe, video_latent):
    """Run the loaded latent x2 upscaler against the post-denoise video latent.
    Mirrors the dance from upstream's TwoStageHQPipeline so the upsampler
    gets the same denormalized input it was trained on. Returns the upscaled
    latent in the same (B, C, F, H, W) layout."""
    import mlx.core as mx
    upsampler = _load_upscaler()
    if upsampler is None:
        raise RuntimeError("LTX spatial upscaler weights not on disk")
    if pipe.vae_encoder is None:
        pipe._load_vae_encoder()
    # (B, C, F, H, W) -> (B, F, H, W, C) for denormalize_latent
    video_mlx = video_latent.transpose(0, 2, 3, 4, 1)
    video_denorm = pipe.vae_encoder.denormalize_latent(video_mlx)
    # back to (B, C, F, H, W) for the upsampler
    video_denorm = video_denorm.transpose(0, 4, 1, 2, 3)
    video_upscaled = upsampler(video_denorm)
    # renormalize for the VAE decoder
    video_up_mlx = video_upscaled.transpose(0, 2, 3, 4, 1)
    video_upscaled = pipe.vae_encoder.normalize_latent(video_up_mlx)
    video_upscaled = video_upscaled.transpose(0, 4, 1, 2, 3)
    mx.eval(video_upscaled)
    return video_upscaled


# ---- one-stage acceleration --------------------------------------------------
# Experimental but useful: standard Q4 T2V/I2V spends most wall time inside the
# denoise loop's X0Model call. The "boost" and "turbo" modes below reuse the
# previous x0 prediction for 2 or 3 locally-stable middle steps. This is opt-in
# per job; "off" restores the upstream sampler exactly.
_ORIGINAL_DENOISE_LOOP = None
_CURRENT_ACCEL_MODE = None
_LAST_ACCEL_STATS = None


def _clean_text(value) -> str:
    return str(value or "").strip()


def _prompt_with_soft_negative(prompt: str, negative_prompt: str) -> str:
    """Fold avoid terms into Q4 one-stage prompts where CFG is disabled."""
    neg = _clean_text(negative_prompt)
    if not neg:
        return prompt
    lower = prompt.lower()
    if "avoid:" in lower or "negative prompt:" in lower:
        return prompt
    return f"{prompt}\nAvoid: {neg}"


@contextmanager
def _override_default_negative_prompt(negative_prompt: str):
    """Temporarily extend upstream's CFG negative prompt for this one job."""
    neg = _clean_text(negative_prompt)
    if not neg:
        yield False
        return

    import ltx_pipelines_mlx.ti2vid_one_stage as one_stage

    previous = one_stage.DEFAULT_NEGATIVE_PROMPT
    one_stage.DEFAULT_NEGATIVE_PROMPT = f"{previous}, {neg}"
    try:
        yield True
    finally:
        one_stage.DEFAULT_NEGATIVE_PROMPT = previous


def _scalar(x) -> float:
    import mlx.core as mx

    mx.eval(x)
    return float(x)


def _relative_mae(mx, current, previous) -> float:
    if current is None or previous is None:
        return 999.0
    diff = mx.mean(mx.abs(current - previous))
    base = mx.maximum(mx.mean(mx.abs(previous)), 1e-6)
    return _scalar(diff / base)


def _build_adaptive_x0_loop(mode_name: str, max_skips: int, video_thresh: float, audio_thresh: float):
    import mlx.core as mx
    import ltx_pipelines_mlx.utils.samplers as samplers

    def denoise_loop_adaptive_x0(
        model,
        video_state,
        audio_state,
        video_text_embeds,
        audio_text_embeds,
        sigmas=None,
        video_positions=None,
        audio_positions=None,
        video_attention_mask=None,
        audio_attention_mask=None,
        show_progress=True,
    ):
        if sigmas is None:
            sigmas = samplers.DISTILLED_SIGMAS

        if video_positions is None and video_state.positions is not None:
            video_positions = video_state.positions
        if audio_positions is None and audio_state.positions is not None:
            audio_positions = audio_state.positions
        if video_attention_mask is None and video_state.attention_mask is not None:
            video_attention_mask = video_state.attention_mask
        if audio_attention_mask is None and audio_state.attention_mask is not None:
            audio_attention_mask = audio_state.attention_mask

        video_x = video_state.latent
        audio_x = audio_state.latent
        steps = list(zip(sigmas[:-1], sigmas[1:]))
        iterator = samplers.tqdm(steps, desc="Denoising", disable=not show_progress)
        protected_head = min(2, len(steps))
        protected_tail = min(len(steps), max(2, math.ceil(len(steps) / 3))) if steps else 0

        global _LAST_ACCEL_STATS
        stats = {
            "mode": mode_name,
            "max_skips": max_skips,
            "video_thresh": video_thresh,
            "audio_thresh": audio_thresh,
            "protected_head": protected_head,
            "protected_tail": protected_tail,
            "total_steps": len(steps),
            "steps": [],
            "cached_steps": [],
            "full_steps": [],
            "cached_steps_count": 0,
            "full_steps_count": 0,
            "estimated_denoise_call_savings_pct": 0.0,
        }
        _LAST_ACCEL_STATS = stats

        video_uniform = samplers._is_uniform_mask(video_state.denoise_mask)
        audio_uniform = samplers._is_uniform_mask(audio_state.denoise_mask)
        last_video_latent = None
        last_audio_latent = None
        last_video_x0 = None
        last_audio_x0 = None
        skip_count = 0

        for idx, (sigma, sigma_next) in enumerate(iterator):
            step_t0 = time.perf_counter()
            sigma_arr = mx.array([sigma], dtype=mx.bfloat16)
            batch = video_x.shape[0]
            # Keep early structure and late detail refinement exact. With the
            # standard 8-step schedule this protects steps 0, 1 and 5, 6, 7;
            # Turbo can only cache stable middle steps where artifacts are much
            # less likely to show up as blurry hands/faces/type.
            protected = idx < protected_head or idx >= len(steps) - protected_tail
            v_delta = _relative_mae(mx, video_x, last_video_latent)
            a_delta = _relative_mae(mx, audio_x, last_audio_latent)
            can_skip = (
                not protected
                and skip_count < max_skips
                and last_video_x0 is not None
                and last_audio_x0 is not None
                and v_delta <= video_thresh
                and a_delta <= audio_thresh
            )

            if can_skip:
                skip_count += 1
                video_x0, audio_x0 = last_video_x0, last_audio_x0
                decision = "cached"
            else:
                call_kwargs = dict(
                    video_latent=video_x,
                    audio_latent=audio_x,
                    sigma=mx.broadcast_to(sigma_arr, (batch,)),
                    video_text_embeds=video_text_embeds,
                    audio_text_embeds=audio_text_embeds,
                    video_positions=video_positions,
                    audio_positions=audio_positions,
                    video_attention_mask=video_attention_mask,
                    audio_attention_mask=audio_attention_mask,
                )
                if not video_uniform:
                    call_kwargs["video_timesteps"] = samplers._compute_per_token_timesteps(
                        sigma,
                        video_state.denoise_mask,
                    )
                if not audio_uniform:
                    call_kwargs["audio_timesteps"] = samplers._compute_per_token_timesteps(
                        sigma,
                        audio_state.denoise_mask,
                    )
                video_x0, audio_x0 = model(**call_kwargs)
                last_video_x0, last_audio_x0 = video_x0, audio_x0
                # Use only full-model steps as the comparison anchor. Updating
                # this on cached steps would make back-to-back skips too easy.
                last_video_latent, last_audio_latent = video_x, audio_x
                decision = "full"

            video_x0 = samplers.apply_denoise_mask(video_x0, video_state.clean_latent, video_state.denoise_mask)
            audio_x0 = samplers.apply_denoise_mask(audio_x0, audio_state.clean_latent, audio_state.denoise_mask)
            video_x = samplers.euler_step(video_x, video_x0, sigma, sigma_next)
            audio_x = samplers.euler_step(audio_x, audio_x0, sigma, sigma_next)
            mx.async_eval(video_x, audio_x)
            step_sec = round(time.perf_counter() - step_t0, 3)
            step_stats = {
                "step": idx,
                "decision": decision,
                "protected": protected,
                "v_delta": round(v_delta, 6),
                "a_delta": round(a_delta, 6),
                "wall_sec": step_sec,
            }
            stats["steps"].append(step_stats)
            if decision == "cached":
                stats["cached_steps"].append(idx)
            else:
                stats["full_steps"].append(idx)
            stats["cached_steps_count"] = len(stats["cached_steps"])
            stats["full_steps_count"] = len(stats["full_steps"])
            stats["estimated_denoise_call_savings_pct"] = (
                round(100.0 * stats["cached_steps_count"] / len(steps), 1)
                if steps else 0.0
            )
            emit({
                "event": "log",
                "line": (
                    "accel:adaptive_x0 "
                    f"step={idx} decision={decision} "
                    f"protected={int(protected)} "
                    f"v_delta={v_delta:.5f} a_delta={a_delta:.5f} "
                    f"skips={skip_count}/{max_skips} wall={step_sec:.2f}s"
                ),
            })

        samplers.aggressive_cleanup()
        return samplers.DenoiseOutput(video_latent=video_x, audio_latent=audio_x)

    return denoise_loop_adaptive_x0


def configure_acceleration(mode: str) -> str:
    """Configure the one-stage sampler acceleration mode for this helper.

    mode: off | boost | turbo
    boost: skip at most 2 stable middle X0Model calls.
    turbo: skip at most 3 stable middle X0Model calls.
    """
    global _ORIGINAL_DENOISE_LOOP, _CURRENT_ACCEL_MODE, _LAST_ACCEL_STATS

    requested = (mode or "off").strip().lower()
    if requested not in {"off", "boost", "turbo"}:
        requested = "off"

    import ltx_pipelines_mlx.ti2vid_one_stage as ti2vid
    import ltx_pipelines_mlx.utils.samplers as samplers

    if _ORIGINAL_DENOISE_LOOP is None:
        _ORIGINAL_DENOISE_LOOP = samplers.denoise_loop

    if requested == _CURRENT_ACCEL_MODE:
        if requested != "off":
            _LAST_ACCEL_STATS = None
        return requested

    _LAST_ACCEL_STATS = None
    if requested == "off":
        samplers.denoise_loop = _ORIGINAL_DENOISE_LOOP
        ti2vid.denoise_loop = _ORIGINAL_DENOISE_LOOP
    elif requested == "boost":
        loop = _build_adaptive_x0_loop("boost", max_skips=2, video_thresh=0.02, audio_thresh=0.02)
        samplers.denoise_loop = loop
        ti2vid.denoise_loop = loop
    else:
        loop = _build_adaptive_x0_loop("turbo", max_skips=3, video_thresh=0.03, audio_thresh=0.03)
        samplers.denoise_loop = loop
        ti2vid.denoise_loop = loop

    _CURRENT_ACCEL_MODE = requested
    emit({"event": "log", "line": f"accel:mode {requested}"})
    return requested


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
            # Granular breadcrumbs so a silent helper death is traceable:
            # if the panel's last log line is "step:get_pipe" then we died
            # during pipeline init (likely OOM or weight-load issue). If
            # it reaches "step:generate" / "step:decode_and_save", the
            # failure is inside denoising or VAE/audio decode respectively.
            # Without this the
            # last visible line was the original "Loading I2V pipeline..."
            # message and users had no idea where to look.
            # LoRAs (optional). Each is {"path": str, "strength": float}.
            # Path may be a local file or a HuggingFace repo id; the
            # safetensors loader handles both transparently. Empty list
            # behaves identically to the no-LoRA path (cache key matches
            # the unloaded baseline).
            loras = p.get("loras") or []
            if loras:
                emit({"event": "log",
                      "line": f"step:get_pipe kind={('i2v' if needs_image else 't2v')} loras={len(loras)}"})
            else:
                emit({"event": "log",
                      "line": f"step:get_pipe kind={('i2v' if needs_image else 't2v')}"})
            pipe = get_pipe("i2v" if needs_image else "t2v", loras=loras)
            emit({"event": "log", "line": "step:get_pipe done"})
            accel_mode = configure_acceleration(p.get("accel", "off"))
            negative_prompt = _clean_text(p.get("negative_prompt"))
            effective_prompt = _prompt_with_soft_negative(p["prompt"], negative_prompt)
            if negative_prompt:
                emit({
                    "event": "log",
                    "line": "Avoid terms active (Q4 path folds them into the positive prompt; CFG paths use native negative conditioning).",
                })

            kwargs = dict(
                prompt=effective_prompt,
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

            # Y1.021: model-based latent upscale path. When the user picks
            # the "Sharper" method on a non-Native target, we run the
            # spatial latent upscaler between denoise and VAE decode so the
            # decoder hallucinates real detail at 2× — vs. the cheaper
            # ffmpeg Lanczos path (which the panel applies after the helper
            # returns). Only fires when the upscaler weights are on disk;
            # otherwise we fall back silently to the normal path.
            upscale_method = (p.get("upscale_method") or "lanczos").strip().lower()
            upscale_target = (p.get("upscale") or "off").strip().lower()
            use_model_upscale = (
                MODEL_UPSCALE_ENABLED
                and upscale_method == "model"
                and upscale_target in ("fit_720p", "x2")
                and upscaler_available()
            )
            if upscale_method == "model" and upscale_target in ("fit_720p", "x2"):
                if not MODEL_UPSCALE_ENABLED:
                    emit({"event": "log", "line": "Sharper upscale is lab-only in this build — falling back to Lanczos."})
                elif not upscaler_available():
                    emit({"event": "log", "line": "Sharper upscale requested but model weights missing — falling back to Lanczos."})

            if use_model_upscale:
                emit({"event": "log", "line": f"step:generate mode={mode} {kwargs['width']}x{kwargs['height']} {kwargs['num_frames']}f steps={kwargs['num_steps']} accel={accel_mode} upscale=model"})
                # Step 1: generate latents (no save)
                video_latent, audio_latent = _generate_latents(pipe, needs_image=needs_image, kwargs=kwargs)
                emit({"event": "log", "line": "step:generate done"})
                # Free DiT + text encoder before the upscale + VAE decode peak.
                emit({"event": "log", "line": "step:free_generation_modules start"})
                _free_pipe_for_decode(pipe)
                emit({"event": "log", "line": "step:free_generation_modules done"})
                # Step 2: latent x2 upscale
                emit({"event": "log", "line": "step:latent_upscale_x2 start"})
                video_latent = _model_upscale_video_latent(pipe, video_latent)
                emit({"event": "log", "line": f"step:latent_upscale_x2 done — latent {video_latent.shape[-2]}×{video_latent.shape[-1]}"})
                # Free the upscaler before VAE decode (can be ~2-3 GB peak).
                _free_upscaler()
                # Step 3: VAE decode + save (decoder loads inside _decode_and_save_video).
                out_path = pipe._decode_and_save_video(video_latent, audio_latent, kwargs["output_path"])
                emit({"event": "log", "line": "step:decode_and_save done"})
            else:
                emit({"event": "log", "line": f"step:generate mode={mode} {kwargs['width']}x{kwargs['height']} {kwargs['num_frames']}f steps={kwargs['num_steps']} accel={accel_mode}"})
                video_latent, audio_latent = _generate_latents(pipe, needs_image=needs_image, kwargs=kwargs)
                emit({"event": "log", "line": "step:generate done"})
                emit({"event": "log", "line": "step:free_generation_modules start"})
                _free_pipe_for_decode(pipe)
                emit({"event": "log", "line": "step:free_generation_modules done"})
                emit({"event": "log", "line": "step:decode_and_save start"})
                out_path = pipe._decode_and_save_video(video_latent, audio_latent, kwargs["output_path"])
                emit({"event": "log", "line": "step:decode_and_save done"})
            elapsed = round(time.time() - t0, 2)
            _last_activity = time.time()
            done_event = {
                "event": "done", "id": job_id,
                "output": str(out_path), "elapsed_sec": elapsed,
                "seed_used": seed,
                "upscale_applied": "model_x2" if use_model_upscale else None,
            }
            if accel_mode != "off" and _LAST_ACCEL_STATS:
                done_event["accel_metrics"] = _LAST_ACCEL_STATS
            emit(done_event)
        except Exception as exc:
            _last_activity = time.time()
            emit({"event": "error", "id": job_id, "error": str(exc), "trace": traceback.format_exc()})
        finally:
            try:
                configure_acceleration("off")
            except Exception:
                pass
            try:
                _free_upscaler()
            except Exception:
                pass
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
            configure_acceleration("off")
            # Extend supports LoRAs via the same _pending_loras hook;
            # the dev transformer picks them up at load time just like T2V/I2V.
            loras = p.get("loras") or []
            pipe = get_pipe("extend", loras=loras)
            video_path = p["video_path"]
            if not os.path.exists(video_path):
                raise RuntimeError(f"source video not found: {video_path}")
            # cfg_scale defaults to 1.0 (no classifier-free guidance) on 64 GB
            # Macs: CFG runs both conditional + unconditional through the dev
            # transformer, doubling activation memory and pushing 1280×704
            # extends into swap (240s/step instead of ~25s/step). The panel
            # exposes a "Fast" / "Quality" toggle that flips this to 3.0.
            cfg_scale = float(p.get("cfg_scale", 1.0))
            with _override_default_negative_prompt(p.get("negative_prompt")) as neg_active:
                if neg_active:
                    emit({"event": "log", "line": "Avoid terms active via native CFG negative prompt."})
                video_lat, audio_lat = pipe.extend_from_video(
                    prompt=p["prompt"],
                    video_path=video_path,
                    extend_frames=int(p.get("extend_frames", 5)),
                    direction=p.get("direction", "after"),
                    seed=seed,
                    num_steps=int(p.get("steps", 12)),
                    cfg_scale=cfg_scale,
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
            configure_acceleration("off")
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
                # Default 0.0 — upstream HQ (TwoStageHQPipeline) uses empty
                # stg_blocks, so any nonzero stg_scale just runs an extra
                # forward pass per step that's then discarded.
                stg_scale=float(p.get("stg_scale", 0.0)),
                enable_teacache=bool(p.get("enable_teacache", True)),
                teacache_thresh=float(p.get("teacache_thresh", 1.0)),
            )
            img = p.get("image")
            if img:
                if not os.path.exists(img):
                    raise RuntimeError(f"image not found: {img}")
                kwargs["image"] = img
                emit({"event": "log", "line": f"HQ I2V — pipeline will cover-crop image to {kwargs['width']}x{kwargs['height']}"})
            with _override_default_negative_prompt(p.get("negative_prompt")) as neg_active:
                if neg_active:
                    emit({"event": "log", "line": "Avoid terms active via native CFG negative prompt."})
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
            configure_acceleration("off")
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
            with _override_default_negative_prompt(p.get("negative_prompt")) as neg_active:
                if neg_active:
                    emit({"event": "log", "line": "Avoid terms active via native CFG negative prompt."})
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

    if action == "enhance_prompt":
        # Gemma-driven prompt rewriting. Same model file as the pipeline's
        # text encoder, but loaded as a `GemmaLanguageModel` (the wrapper
        # that knows how to do `enhance_t2v` / `enhance_i2v`). First call
        # eats a ~10-15s Gemma load; cached afterwards. release_pipelines
        # frees Gemma when a real render comes in, so memory doesn't pile up.
        job_id = msg.get("id", "?")
        p = msg.get("params", {}) or {}
        user_prompt = (p.get("prompt") or "").strip()
        mode = (p.get("mode") or "t2v").lower()
        if mode not in ("t2v", "i2v"):
            mode = "t2v"
        seed = int(p.get("seed", 10))
        if not user_prompt:
            emit({"event": "error", "id": job_id, "error": "empty prompt"})
            continue
        _is_busy = True
        try:
            t0 = time.time()
            lm = get_gemma_lm()
            if mode == "t2v":
                enhanced = lm.enhance_t2v(user_prompt, seed=seed)
            else:
                enhanced = lm.enhance_i2v(user_prompt, seed=seed)
            elapsed = round(time.time() - t0, 2)
            _last_activity = time.time()
            emit({
                "event": "done", "id": job_id,
                "enhanced": enhanced,
                "original": user_prompt,
                "mode": mode,
                "elapsed_sec": elapsed,
            })
        except Exception as exc:
            _last_activity = time.time()
            emit({"event": "error", "id": job_id, "error": str(exc), "trace": traceback.format_exc()})
        finally:
            _is_busy = False
        continue

    emit({"event": "error", "error": f"unknown action: {action}"})

emit({"event": "exit", "reason": "stdin_closed"})
