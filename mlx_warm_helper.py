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

# Y1.037 — VAE temporal-streaming decision.
#
# Y1.035 patched the upstream `decode_and_stream` to actually stream temporal
# tiles (it had been pretending to). That fixed the "frozen final step" bug
# on long / 720p clips, but adds ~30 s of overlap-tile compute on a 5-second
# Standard render where the old full-volume decode fit in memory just fine
# (M-Max 64 GB measured: 459 s pre-Y1.035 → 493 s on Y1.035, +7.4%).
#
# This module captures whatever LTX_VAE_STREAMING was at process start. If the
# user explicitly set it (any value), we respect it. Otherwise the per-job
# helper code flips the env var per-render based on num_frames vs the
# threshold (default 200 frames ≈ 8 s @ 24 fps), letting the patched decoder's
# auto-pick the streaming or full-decode path. Power users can set
# LTX_VAE_STREAMING_THRESHOLD=N to override the cutoff.
_USER_VAE_STREAMING_OVERRIDE = os.environ.get("LTX_VAE_STREAMING")


def _apply_vae_streaming_decision(num_frames: int) -> None:
    """Set/unset os.environ['LTX_VAE_STREAMING'] for the upcoming decode.
    No-op if the user pinned a value at helper start time. Threshold reads
    LTX_VAE_STREAMING_AUTO_MAX_FRAMES (default 121, matching the patched
    decoder's auto-mode cutoff in patch_ltx_codec.py)."""
    if _USER_VAE_STREAMING_OVERRIDE is not None:
        return  # respect explicit override
    threshold = int(os.environ.get("LTX_VAE_STREAMING_AUTO_MAX_FRAMES", "121"))
    if num_frames <= threshold:
        os.environ["LTX_VAE_STREAMING"] = "0"
    else:
        # Long clip: let the patched decoder default ("auto") pick streaming.
        os.environ.pop("LTX_VAE_STREAMING", None)

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


def _filter_unsupported_kwargs(fn, kwargs: dict) -> dict:
    """Return `kwargs` with any keys the target callable doesn't accept removed.

    Phosphene's helper passes a superset of kwargs to `generate_and_save`
    so newer pipeline features (bongmath_max_iter, stage2_image_conditioning,
    etc.) work transparently. But stock upstream releases sometimes ship a
    strict signature without **kwargs — calling them blows up with
    `unexpected keyword argument 'X'` even if the feature would just be a
    no-op when missing. Introspect once, drop unsupported keys, log what we
    dropped. If the target has a VAR_KEYWORD parameter (**kwargs in any
    form), pass everything through untouched."""
    try:
        import inspect as _inspect
        sig = _inspect.signature(fn)
        has_var_kw = any(
            pp.kind == _inspect.Parameter.VAR_KEYWORD
            for pp in sig.parameters.values()
        )
        if has_var_kw:
            return kwargs
        accepted = set(sig.parameters.keys())
        dropped = sorted(k for k in kwargs if k not in accepted)
        if dropped:
            emit({"event": "log",
                  "line": f"pipeline.generate_and_save doesn't accept {dropped}; dropping."})
        return {k: v for k, v in kwargs.items() if k in accepted}
    except Exception:
        # Introspection itself shouldn't block a render — fall through
        # and let the original call surface whatever the real error is.
        return kwargs


_LORA_PATCH_INSTALLED = False
_VIDEO_DECODER_PATCH_INSTALLED = False


def _install_video_decoder_patch() -> None:
    """Translate fps/frame_rate kwargs across upstream layers.

    Upstream regression (observed 2026-05-13/14): `utils.blocks.VideoDecoder.
    decode_and_stream` accepts `fps=`, but `utils._orchestration.
    decode_and_save_video` calls it with `frame_rate=`, raising TypeError.
    Similarly the inner decoder in `ltx_core_mlx.model.video_vae.VideoDecoder`
    uses `frame_rate=` while the wrapper used `fps=` for the inner call.
    This patch wraps the wrapper to accept either kwarg and tries both
    when invoking the inner decoder. Idempotent.
    """
    global _VIDEO_DECODER_PATCH_INSTALLED
    if _VIDEO_DECODER_PATCH_INSTALLED:
        return
    import ltx_pipelines_mlx.utils.blocks as _blocks
    _orig = _blocks.VideoDecoder.decode_and_stream

    def _wrapped(self, video_latent, output_path, fps=24.0,
                 frame_rate=None, audio_path=None):
        if frame_rate is not None:
            fps = frame_rate
        decoder = self.load()
        try:
            decoder.decode_and_stream(
                video_latent, output_path, frame_rate=fps, audio_path=audio_path
            )
        except TypeError:
            decoder.decode_and_stream(
                video_latent, output_path, fps=fps, audio_path=audio_path
            )
        return output_path

    _blocks.VideoDecoder.decode_and_stream = _wrapped
    _VIDEO_DECODER_PATCH_INSTALLED = True


def _install_lora_fusion_patches() -> None:
    """Make subclass pipelines actually fuse _pending_loras during load().

    Upstream `BasePipeline.load()` in `_base.py` checks `_pending_loras` and
    fuses LoRA deltas into transformer weights before quantization. But the
    subclasses we use here — `DistilledPipeline`, `TI2VidTwoStagesPipeline`,
    `TI2VidOneStagePipeline`, `TI2VidTwoStagesHQPipeline` — each override
    `load()` entirely and load the DiT via
    `_load_transformer_with_optional_streaming` / `_load_dev_transformer`,
    bypassing the fusion path. Without this patch, every panel render with
    an attached LoRA silently produced LoRA-free output ("face is not him"
    bug).

    Fix: wrap each subclass's `load()` so that when `_pending_loras` is set
    and `self.dit is None`, we pre-load+fuse+quantize the transformer
    ourselves, set `self.dit`, then call the original `load()` which
    short-circuits the DiT step (because `self.dit is not None`) and
    proceeds to VAE encoder / upsampler / decoders as normal.

    Idempotent — sets `_phosphene_lora_fix=True` on each class and a
    module-level flag so repeated calls (e.g. on every `get_pipe`) are a
    no-op. Installed lazily from `get_pipe` because the pipeline import
    strategy is decided there (post-refactor vs. pre-refactor name
    fallback). `TI2VidTwoStagesHQPipeline` is patched too even though
    `get_hq_pipe` doesn't currently call `_attach_loras` — the patch is
    inert when `_pending_loras` is absent, and this is forward-proof if
    HQ ever gets a user-LoRA path."""
    global _LORA_PATCH_INSTALLED
    if _LORA_PATCH_INSTALLED:
        return

    classes = []
    for name in ("DistilledPipeline", "TI2VidTwoStagesPipeline",
                 "TI2VidOneStagePipeline", "TI2VidTwoStagesHQPipeline"):
        try:
            mod = __import__("ltx_pipelines_mlx", fromlist=[name])
            cls = getattr(mod, name, None)
        except ImportError:
            cls = None
        if cls is not None:
            classes.append(cls)

    if not classes:
        return  # very old install — nothing to patch

    from ltx_core_mlx.model.transformer.model import LTXModel
    from ltx_core_mlx.utils.memory import aggressive_cleanup
    from ltx_core_mlx.utils.weights import apply_quantization, load_split_safetensors

    distilled_cls = next(
        (c for c in classes if c.__name__ == "DistilledPipeline"), None
    )

    def _resolve_tx_path(pipe):
        # CRITICAL: this must return the SAME file the pipeline's own load()
        # would load — fusion sets self.dit, and the original load() then
        # short-circuits because self.dit is not None. If we fuse into a
        # different file than the pipeline expects to run inference against,
        # we get a base-fine-tune mismatch (e.g. dev weights running through
        # the distilled 8-step sigma schedule = undertrained denoise = blur).
        #
        # DistilledPipeline inherits `_dev_transformer = "transformer-dev..."`
        # from its TI2VidTwoStagesPipeline parent, but its load() actually
        # picks `transformer.safetensors` → `transformer-distilled.safetensors`
        # (see distilled.py). We must match THAT, not the inherited dev name.
        if distilled_cls is not None and isinstance(pipe, distilled_cls):
            p = pipe.model_dir / "transformer.safetensors"
            if not p.exists():
                p = pipe.model_dir / "transformer-distilled.safetensors"
            return p
        # Genuine dev-based pipelines: TI2VidTwoStagesPipeline (non-distilled),
        # TI2VidOneStagePipeline, TI2VidTwoStagesHQPipeline. They all load
        # via `_load_dev_transformer()` which reads `_dev_transformer`.
        dev_name = getattr(pipe, "_dev_transformer", None)
        if dev_name:
            return pipe.model_dir / dev_name
        # Fallback (shouldn't hit): distilled file resolution.
        p = pipe.model_dir / "transformer.safetensors"
        if not p.exists():
            p = pipe.model_dir / "transformer-distilled.safetensors"
        return p

    def _make_wrapper(orig_load):
        def patched_load(self):
            pending = getattr(self, "_pending_loras", None)
            if pending and self.dit is None and not getattr(self, "_loaded", False):
                tx_path = _resolve_tx_path(self)
                emit({"event": "log",
                      "line": f"Fusing {len(pending)} LoRA(s) into "
                              f"{os.path.basename(str(tx_path))}..."})
                weights = load_split_safetensors(tx_path, prefix="transformer.")
                weights = self._fuse_pending_loras(weights, pending)
                self.dit = LTXModel()
                apply_quantization(self.dit, weights)
                self.dit.load_weights(list(weights.items()))
                aggressive_cleanup()
            return orig_load(self)
        return patched_load

    for cls in classes:
        if getattr(cls, "_phosphene_lora_fix", False):
            continue
        cls.load = _make_wrapper(cls.load)
        cls._phosphene_lora_fix = True

    # Second entry point: TI2VidTwoStagesHQPipeline.generate_two_stage()
    # bypasses self.load() entirely — it calls `self._load_dev_transformer()`
    # directly when self.dit is None. So the load() wrapper above never fires
    # for HQ renders, and the LoRA silently isn't fused. Wrap the weight-loader
    # method itself so HQ also fuses. Other classes call _load_dev_transformer
    # from inside their load(); the load() wrapper above runs first and sets
    # self.dit, so the inner call short-circuits in `if self.dit is None`.
    def _make_dev_loader_wrapper(orig):
        def patched_load_dev(self):
            pending = getattr(self, "_pending_loras", None)
            # FIX 2026-05-14: removed the `_phosphene_dit_fused` flag guard.
            # That flag latched True after the first job in a batch and
            # never reset; the next job in the same panel pipeline reuse
            # would skip fusion → bare dev transformer → silent
            # "LoRA-not-applied" on every clip after the first. This method
            # is only called when self.dit is None (per HQ pipeline logic),
            # so when we reach this branch we always need to (re)fuse.
            if pending:
                dev_name = (getattr(self, "_dev_transformer", None)
                            or "transformer-dev.safetensors")
                tx_path = self.model_dir / dev_name
                emit({"event": "log",
                      "line": f"Fusing {len(pending)} LoRA(s) into "
                              f"{os.path.basename(str(tx_path))}..."})
                weights = load_split_safetensors(tx_path, prefix="transformer.")
                weights = self._fuse_pending_loras(weights, pending)
                dit = LTXModel()
                apply_quantization(dit, weights)
                dit.load_weights(list(weights.items()))
                aggressive_cleanup()
                return dit
            return orig(self)
        return patched_load_dev

    # Patch on BasePipeline once — every subclass that calls _load_dev_transformer
    # inherits it from BasePipeline, so a single wrap covers HQ + two-stage +
    # one-stage + keyframe + retake. Idempotent via class flag.
    try:
        from ltx_pipelines_mlx._base import BasePipeline as _BasePipeline
    except ImportError:
        _BasePipeline = None
    if _BasePipeline is not None and not getattr(_BasePipeline, "_phosphene_dev_loader_fix", False):
        _BasePipeline._load_dev_transformer = _make_dev_loader_wrapper(
            _BasePipeline._load_dev_transformer
        )
        _BasePipeline._phosphene_dev_loader_fix = True

    _LORA_PATCH_INSTALLED = True


def _attach_loras(pipe, loras: list[dict] | None) -> None:
    """Set _pending_loras on a freshly-constructed pipeline. The upstream
    base class checks this attribute inside load() and fuses the LoRA
    deltas into the transformer weights before quantization. Path on the
    wire may be a local file OR an HF repo id; we resolve HF ids to a
    local .safetensors here so the loader (mx.load) sees an absolute
    path it can actually open.

    NOTE: most subclass pipelines override `load()` and skip the upstream
    fusion path — :func:`_install_lora_fusion_patches` repairs that. It
    runs from `get_pipe` before any pipeline instantiation."""
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


_extend_model_dir: str | None = None


def get_pipe(kind: str, loras: list[dict] | None = None,
             model_dir: str | None = None):
    """kind in {'t2v','i2v','extend'}; loras is an optional list of
    {path, strength} dicts. When the requested LoRA set differs from
    the cached pipeline's, the pipeline is rebuilt — LoRA fusion is a
    one-shot weight transformation, not a runtime toggle.

    Y1.036 — `model_dir` overrides the helper-default LTX_MODEL env var on a
    per-call basis. Used for Extend, which needs the Q8 `transformer-dev`
    weights even on a Standard-tier render. Pre-Y1.024 the Q4 dir incidentally
    carried a copy of `transformer-dev.safetensors` (download bloat) so Extend
    silently loaded from there; the Y1.024 download filter pruned the dupe and
    exposed that Extend is structurally Q8-class. Cached alongside the LoRA
    fingerprint so a model_dir flip rebuilds the pipe."""
    global _t2v_pipe, _i2v_pipe, _extend_pipe
    global _t2v_lora_key, _i2v_lora_key, _extend_lora_key
    global _extend_model_dir
    # Upstream refactor 2026-05-09 (commits d6cc3d1, 493aec2) renamed +
    # removed several pipeline classes. Past intermediate commits had a
    # mix of old + new names. Three import strategies in priority order:
    #
    #   1. ALL old names present (pre-refactor) — best, motion-friendly
    #      single-stage I2V via `ImageToVideoPipeline.generate_from_image`.
    #   2. Only `ImageToVideoPipeline` present (intermediate state, e.g.
    #      32280b9 before 493aec2 removed it) — KEEP it for I2V motion;
    #      alias the others.
    #   3. None present (post-refactor) — alias all to new classes;
    #      I2V goes through DistilledPipeline.generate_two_stage which
    #      lacks CFG and locks frame 0 → no motion. Last resort.
    try:
        from ltx_pipelines_mlx import TextToVideoPipeline, ImageToVideoPipeline, ExtendPipeline
    except ImportError:
        # Try partial — preserve real ImageToVideoPipeline if it exists.
        try:
            from ltx_pipelines_mlx import ImageToVideoPipeline  # motion-friendly
            _has_real_i2v = True
        except ImportError:
            _has_real_i2v = False
        from ltx_pipelines_mlx import DistilledPipeline, RetakePipeline
        TextToVideoPipeline = DistilledPipeline
        if not _has_real_i2v:
            ImageToVideoPipeline = DistilledPipeline  # last-resort alias
        ExtendPipeline = RetakePipeline

    # Repair the subclass-override-skips-fusion bug before any pipe is built.
    _install_lora_fusion_patches()
    _install_video_decoder_patch()  # fps/frame_rate kwarg shim

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
            ext_dir = model_dir or MODEL_ID
            if (_extend_pipe is None
                    or _extend_lora_key != fp
                    or _extend_model_dir != ext_dir):
                if _extend_pipe is not None:
                    why = "LoRA set changed" if _extend_lora_key != fp else "model_dir changed"
                    emit({"event": "log",
                          "line": f"{why}; reloading Extend pipeline."})
                    _extend_pipe = None
                emit({"event": "log",
                      "line": f"Loading Extend pipeline (heavier — uses dev transformer at {ext_dir})..."})
                pipe = ExtendPipeline(
                    model_dir=ext_dir, gemma_model_id=GEMMA_PATH, low_memory=LOW_MEMORY,
                )
                _attach_loras(pipe, loras)
                _extend_pipe = pipe
                _extend_lora_key = fp
                _extend_model_dir = ext_dir
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


_hq_lora_key: str | None = None


def get_hq_pipe(model_dir: str, loras: list[dict] | None = None):
    """Returns the TwoStageHQPipeline lazily — Q8 model, res_2s sampler, CFG anchor.

    Same class handles both T2V (image=None) and I2V via the `image` kwarg of
    `generate_and_save`. We rebuild the pipe if the requested model_dir changes
    (e.g. user swapped Q8 for a different quant) OR if the requested LoRA set
    differs from the cached pipe's (fusion is a one-shot weight transform).

    LoRA support added 2026-05-12: character LoRAs are trained against the dev
    transformer (flow-matching, full sigma range), and HQ is the ONLY pipeline
    that runs the dev transformer with CFG and >8 steps. Distilled-path renders
    silently dropped LoRAs (fixed) but the result is still wrong because the
    deltas were learned against dev neuron states, not distilled. Routing
    LoRA renders to HQ is the only way to get a faithful character replay.
    """
    global _hq_pipe, _hq_model_dir, _hq_lora_key
    # Upstream `ltx-2-mlx` refactor 2026-05-09 (commits d6cc3d1, 493aec2,
    # 32280b9 — `refactor!: rename pipeline classes to match upstream
    # verbatim`) renamed TwoStageHQPipeline → TI2VidTwoStagesHQPipeline.
    # Defensive import so the helper works against both old and new
    # package versions. Constructor signature is unchanged.
    try:
        from ltx_pipelines_mlx.ti2vid_two_stages_hq import (
            TI2VidTwoStagesHQPipeline as TwoStageHQPipeline,
        )
    except ImportError:
        from ltx_pipelines_mlx.ti2vid_two_stages_hq import TwoStageHQPipeline

    # Reuse the same fusion-patch installer as get_pipe. Without this,
    # TI2VidTwoStagesHQPipeline.load() would still bypass _pending_loras.
    _install_lora_fusion_patches()
    _install_video_decoder_patch()  # fps/frame_rate kwarg shim

    fp = _lora_fingerprint(loras)

    with _pipe_lock:
        release_pipelines(keep_kind="hq")
        if (_hq_pipe is None
                or _hq_model_dir != model_dir
                or _hq_lora_key != fp):
            if _hq_pipe is not None:
                why = "LoRA set changed" if _hq_lora_key != fp else "model_dir changed"
                emit({"event": "log", "line": f"{why}; reloading HQ pipeline."})
                _hq_pipe = None
            emit({"event": "log", "line": f"Loading HQ pipeline (Q8 dev model — {model_dir})..."})
            _hq_pipe = TwoStageHQPipeline(
                model_dir=model_dir, gemma_model_id=GEMMA_PATH, low_memory=LOW_MEMORY,
            )
            _attach_loras(_hq_pipe, loras)
            _hq_model_dir = model_dir
            _hq_lora_key = fp
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
    # Pre-refactor packages: old TextToVideoPipeline.generate /
    #   ImageToVideoPipeline.generate_from_image — single-stage Q4
    #   path with explicit frame_rate plumbing.
    # Post-2026-05-09 refactor: those classes are gone. Q4 lives in the
    #   new DistilledPipeline (two-stage half-res → upscale → refine);
    #   the unified entry point is generate_two_stage(image=optional).
    #   It accepts **_unused_kwargs so num_steps/frame_rate are absorbed
    #   silently rather than ValueError-ing.
    # Detect by method presence — no class import gymnastics needed.
    if needs_image and hasattr(pipe, "generate_from_image"):
        # Probe whether this version of the method accepts frame_rate (the
        # codec patch adds this kwarg post-install; rolled-back source may
        # not have it). Skip kwargs the method doesn't accept.
        import inspect as _inspect
        sig = _inspect.signature(pipe.generate_from_image)
        call_kwargs = dict(
            prompt=kwargs["prompt"],
            image=kwargs.get("image"),
            height=kwargs["height"],
            width=kwargs["width"],
            num_frames=kwargs["num_frames"],
            seed=kwargs["seed"],
            num_steps=kwargs["num_steps"],
        )
        if "frame_rate" in sig.parameters:
            call_kwargs["frame_rate"] = kwargs.get("frame_rate", 24.0)
        return pipe.generate_from_image(**call_kwargs)
    if not needs_image and hasattr(pipe, "generate"):
        try:
            import inspect as _inspect
            sig = _inspect.signature(pipe.generate)
            call_kwargs = dict(
                prompt=kwargs["prompt"],
                height=kwargs["height"],
                width=kwargs["width"],
                num_frames=kwargs["num_frames"],
                seed=kwargs["seed"],
                num_steps=kwargs["num_steps"],
            )
            if "frame_rate" in sig.parameters:
                call_kwargs["frame_rate"] = kwargs.get("frame_rate", 24.0)
            return pipe.generate(**call_kwargs)
        except TypeError:
            # New DistilledPipeline.generate inherits from the two-stage
            # parent and doesn't accept frame_rate. Fall through to
            # generate_two_stage which absorbs everything via
            # **_unused_kwargs.
            pass
    # Unified new-API fallback (post-refactor packages).
    return pipe.generate_two_stage(
        prompt=kwargs["prompt"],
        image=kwargs.get("image") if needs_image else None,
        height=kwargs["height"],
        width=kwargs["width"],
        num_frames=kwargs["num_frames"],
        seed=kwargs["seed"],
        stage1_steps=kwargs.get("num_steps"),
        # frame_rate / num_steps absorbed by **_unused_kwargs in the new
        # signature — kept here so the call is identical to the old one
        # at the source level.
        frame_rate=kwargs.get("frame_rate", 24.0),
        num_steps=kwargs.get("num_steps"),
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
        # 2026-05-09 lab finding: with the previous tail = ceil(N/3), the
        # 8-step distilled schedule protected only steps 5,6,7 — leaving
        # step 4 cache-eligible. Step 4's relative MAE (~0.0245) sits
        # between Boost's threshold (0.02) and Turbo's (0.03), so Boost
        # protected it by chance and Turbo cached it — producing visible
        # eye/skin artifacts on the final frame. Bumping to ceil(N/2)
        # protects step 4 deterministically (~+13% wall, no more
        # late-step drift).
        protected_tail = min(len(steps), max(2, math.ceil(len(steps) / 2))) if steps else 0

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
            # standard 8-step schedule this protects steps 0, 1 and 4, 5, 6, 7;
            # Turbo can only cache stable middle steps (2, 3) where artifacts
            # are much less likely to show up as blurry hands/faces/eyes.
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
                frame_rate=float(p.get("frame_rate", 24.0)),
            )
            # Y1.037: short-clip VAE-streaming opt-out. Set the env var BEFORE
            # generate() so it propagates through the whole chain (the patched
            # decode_and_stream reads os.environ at decode call time).
            _apply_vae_streaming_decision(kwargs["num_frames"])
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
                emit({"event": "log", "line": f"step:generate mode={mode} {kwargs['width']}x{kwargs['height']} {kwargs['num_frames']}f @{kwargs['frame_rate']:.1f}fps steps={kwargs['num_steps']} accel={accel_mode} upscale=model"})
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
                # FIX 2026-05-14: upstream renamed fps= → frame_rate= (keyword-only).
                out_path = pipe._decode_and_save_video(video_latent, audio_latent, kwargs["output_path"], frame_rate=kwargs["frame_rate"])
                emit({"event": "log", "line": "step:decode_and_save done"})
            else:
                emit({"event": "log", "line": f"step:generate mode={mode} {kwargs['width']}x{kwargs['height']} {kwargs['num_frames']}f @{kwargs['frame_rate']:.1f}fps steps={kwargs['num_steps']} accel={accel_mode}"})
                video_latent, audio_latent = _generate_latents(pipe, needs_image=needs_image, kwargs=kwargs)
                emit({"event": "log", "line": "step:generate done"})
                emit({"event": "log", "line": "step:free_generation_modules start"})
                _free_pipe_for_decode(pipe)
                emit({"event": "log", "line": "step:free_generation_modules done"})
                emit({"event": "log", "line": "step:decode_and_save start"})
                # FIX 2026-05-14: upstream renamed fps= → frame_rate= (keyword-only).
                out_path = pipe._decode_and_save_video(video_latent, audio_latent, kwargs["output_path"], frame_rate=kwargs["frame_rate"])
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
            # Y1.036 — Extend needs the Q8 `transformer-dev` weights. Panel
            # passes the resolved Q8 path via params.model_dir; falling back
            # to the helper's MODEL_ID is the legacy behavior.
            ext_model_dir = p.get("model_dir")
            pipe = get_pipe("extend", loras=loras, model_dir=ext_model_dir)
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
            # FIX 2026-05-14: upstream made frame_rate= keyword-only required.
            pipe._decode_and_save_video(video_lat, audio_lat, p["output_path"], frame_rate=float(p.get("frame_rate", 24.0)))
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
            # LoRAs flow through the same wire shape as t2v/i2v. HQ is the
            # only path where dev-base character LoRAs actually transfer
            # cleanly (distilled inference + dev-trained LoRA = base-fine-tune
            # mismatch).
            hq_loras = p.get("loras") or []
            if hq_loras:
                emit({"event": "log",
                      "line": f"step:get_pipe kind=hq loras={len(hq_loras)}"})
            pipe = get_hq_pipe(model_dir, loras=hq_loras)
            # Y1.037: short-clip VAE-streaming opt-out (HQ T2V/I2V path).
            _apply_vae_streaming_decision(int(p["frames"]))
            kwargs = dict(
                prompt=p["prompt"],
                output_path=p["output_path"],
                height=int(p["height"]),
                width=int(p["width"]),
                num_frames=int(p["frames"]),
                # Upstream regression 2026-05-13: generate_and_save now requires
                # frame_rate as a keyword-only arg. LTX frame counts are 8k+1
                # paired with 24 fps everywhere in our panel; hardcode that here.
                frame_rate=float(p.get("frame_rate", 24.0)),
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
                # Bongmath inner-loop cap (HQ res_2s sampler). Default 100
                # matches upstream. Source: samplers.py:537 has a fixed
                # `for _ in range(bongmath_max_iter)` with no early exit, so
                # the cap IS the iteration count (not just a safety bound).
                # Each iter is pure latent algebra — no model forwards.
                bongmath_max_iter=int(p.get("bongmath_max_iter", 100)),
                # Upstream HQ exposes skip_step for each modality. The MLX
                # res_2s path now honors it as an opt-in experimental speed
                # knob; 0 preserves the locked recipe.
                video_skip_step=int(p.get("video_skip_step", 0)),
                audio_skip_step=int(p.get("audio_skip_step", 0)),
                # Stage-2 image-conditioning mode for I2V (HQ).
                # "full"  = re-encode reference at full res (upstream default)
                # "off"   = skip the full-res re-encode; saves the biggest
                #            single memory peak at the stage-1→2 boundary,
                #            necessary for I2V at 121f on 64 GB. Stage 1 has
                #            already anchored on the reference at half res.
                stage2_image_conditioning=str(
                    p.get("stage2_image_conditioning", "full")
                ),
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
                # Stock site-packages versions of TI2VidTwoStagesPipeline ship
                # a strict generate_and_save signature with no **kwargs catchall.
                # Phosphene passes bongmath_max_iter / stage2_image_conditioning /
                # etc., which would crash a stock install. Introspect once, drop
                # any kwarg the installed signature doesn't accept — better to
                # silently skip a feature flag than to fail the whole render.
                kwargs = _filter_unsupported_kwargs(pipe.generate_and_save, kwargs)
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
        # Keyframe interpolation — anchor images at chosen frame indices,
        # model fills the motion between them. Uses the Q8 dev transformer
        # + two-stage refine (same as HQ).
        #
        # Two input shapes are accepted:
        #
        # (A) Multi-keyframe (preferred — used by agents):
        #         "keyframe_images":  list[str]   absolute paths, length N >= 2
        #         "keyframe_indices": list[int]   pixel-frame indices, length N,
        #                                          strictly ascending, all in [0, frames-1]
        #
        # (B) FFLF backward-compat (used by the panel today):
        #         "start_image": str
        #         "end_image":   str
        #     Equivalent to multi-keyframe with indices [0, frames-1].
        #
        # If (A) fields are present they win; (B) is only checked when (A) is absent.
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
            num_frames = int(p["frames"])

            # ---- Resolve keyframes (multi-keyframe path or FFLF fallback) ----
            kf_images_in = p.get("keyframe_images")
            kf_indices_in = p.get("keyframe_indices")
            if kf_images_in is not None or kf_indices_in is not None:
                # Multi-keyframe — validate strictly so agent bugs surface early.
                if kf_images_in is None or kf_indices_in is None:
                    raise RuntimeError(
                        "keyframe_images and keyframe_indices must both be provided as lists"
                    )
                if not isinstance(kf_images_in, list) or not isinstance(kf_indices_in, list):
                    raise RuntimeError("keyframe_images and keyframe_indices must be lists")
                if len(kf_images_in) != len(kf_indices_in):
                    raise RuntimeError(
                        f"keyframe_images ({len(kf_images_in)}) and "
                        f"keyframe_indices ({len(kf_indices_in)}) must have the same length"
                    )
                if len(kf_images_in) < 2:
                    raise RuntimeError("at least 2 keyframes required")
                for path in kf_images_in:
                    if not isinstance(path, str) or not os.path.exists(path):
                        raise RuntimeError(f"keyframe image not found: {path}")
                idxs: list[int] = []
                for i in kf_indices_in:
                    try:
                        idx = int(i)
                    except (TypeError, ValueError):
                        raise RuntimeError(f"keyframe_indices must be integers, got: {i!r}")
                    if idx < 0 or idx >= num_frames:
                        raise RuntimeError(
                            f"keyframe_index {idx} out of range [0, {num_frames - 1}]"
                        )
                    idxs.append(idx)
                for a, b in zip(idxs, idxs[1:]):
                    if b <= a:
                        raise RuntimeError(
                            f"keyframe_indices must be strictly ascending, got {idxs}"
                        )
                kf_images = list(kf_images_in)
                kf_indices = idxs
                kf_mode_label = f"multi-{len(kf_images)}kf"
            else:
                # FFLF backward-compat — start + end at the boundaries.
                for k in ("start_image", "end_image"):
                    img = p.get(k)
                    if not img or not os.path.exists(img):
                        raise RuntimeError(f"{k} not found: {img}")
                kf_images = [p["start_image"], p["end_image"]]
                kf_indices = [0, num_frames - 1]
                kf_mode_label = "FFLF"

            pipe = get_kf_pipe(model_dir)
            # Y1.037: short-clip VAE-streaming opt-out (Keyframe path).
            _apply_vae_streaming_decision(num_frames)
            kwargs = dict(
                prompt=p["prompt"],
                output_path=p["output_path"],
                keyframe_images=kf_images,
                keyframe_indices=kf_indices,
                height=int(p["height"]),
                width=int(p["width"]),
                num_frames=num_frames,
                # Upstream regression 2026-05-13: generate_and_save needs
                # frame_rate (keyword-only required), not fps. Keep fps for
                # legacy compat; the filter step below drops whichever the
                # installed signature doesn't accept.
                fps=24,
                frame_rate=float(p.get("frame_rate", 24.0)),
                seed=seed,
                stage1_steps=int(p.get("stage1_steps", 15)),
                stage2_steps=int(p.get("stage2_steps", 3)),
                cfg_scale=float(p.get("cfg_scale", 3.0)),
            )
            emit({
                "event": "log",
                "line": (
                    f"Keyframe {kf_mode_label} — indices={kf_indices}, "
                    f"pipeline cover-crops all to {kwargs['width']}x{kwargs['height']}"
                ),
            })
            with _override_default_negative_prompt(p.get("negative_prompt")) as neg_active:
                if neg_active:
                    emit({"event": "log", "line": "Avoid terms active via native CFG negative prompt."})
                kwargs = _filter_unsupported_kwargs(pipe.generate_and_save, kwargs)
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
