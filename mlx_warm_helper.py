#!/usr/bin/env python3.11
"""LTX warm helper — long-lived process holding MLX pipelines in memory.

Reads JSON-line jobs from stdin, emits JSON-line events to stdout.

Actions:
  generate          — T2V or I2V (auto-resizes images to target dims via PIL cover-crop)
  generate_hq       — Q8 dev transformer + res_2s sampler + CFG (TwoStageHQPipeline)
  generate_keyframe — FFLF / multi-keyframe interpolation (KeyframeInterpolationPipeline)
  generate_a2v      — Audio-to-Video, optional image conditioning (A2VidPipelineTwoStage)
  extend            — chain a clip by N latent frames (uses ExtendPipeline)
  enhance_prompt    — Gemma rewriting for T2V / I2V prompts
  ping              — returns pong
  exit              — graceful shutdown

Auto-exits after LTX_IDLE_TIMEOUT seconds idle.
"""
from __future__ import annotations

import importlib.util
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

# ---- MLX lazy-eval defaults — MUST RUN BEFORE LTX IMPORTS ===================
# Upstream ltx_core_mlx reads `LTX2_DIT_EVAL_EVERY` (default 8) and
# `LTX2_GEMMA_EVAL_EVERY` (default 1) at MODULE IMPORT TIME. The defaults
# insert `mx.eval()` calls inside the DiT and Gemma loops every N blocks,
# which forces eager Metal command-buffer flushes. On M-series chips with
# enough unified memory, full lazy-graph mode (=0) is ~30-50× faster
# because the MLX runtime batches the entire step into one graph and the
# Metal driver doesn't pay command-buffer creation/destruction overhead
# per block.
#
# 2026-05-21 — Mr Bizarro reported a 10s T2V Balanced render taking
# ~3 min/step (vs ~10-15 s/step expected). Stack sampling showed the
# helper stuck in `IOGPUMetalCommandBufferStorageDealloc` /
# `MTLResourceList releaseAllObjectsAndReset` — the classic eager-eval
# command-buffer-churn signature. The aa235aa commit message claimed
# it set these defaults but the actual diff was incomplete (no
# `_default_lazy_eval_for_m4_max` function ever landed in the code).
# This block restores the intended behavior.
#
# Override path: explicit env var wins. A user / packager can pin
# `LTX2_DIT_EVAL_EVERY=8` to fall back to upstream defaults if a
# specific tier needs the safety of mid-step evaluation.
if "LTX2_DIT_EVAL_EVERY" not in os.environ:
    # 2026-05-21 perf tuning. Tested matrix on M4 Max 64GB doing
    # I2V Balanced 5s/121f:
    #   =0 (full lazy):   denoise 7s/step ✓ | post-decode HANG 14+ min
    #   =4 (mid lazy):    denoise 7s/step ✓ | post-decode HANG 6+ min
    #   =8 (upstream):    denoise 3 min/step | (untested past denoise)
    #   =1 (per-block):   denoise 7s/step ✓ | post-decode HANG 6+ min
    # The post-decode hang is in the function-return path AFTER the
    # upstream phase logs "Decoding done in X.Xs" — diagnostic markers
    # NEVER fire, suggesting MLX/Metal completion-handler chains hold the
    # GIL through the function exit. Filed as separate roadmap item; the
    # render output IS correct, just the helper sits idle for minutes
    # before signaling done.
    # Picking =1 because the denoise is fast and it matches the eager
    # per-block flush pattern that worked overnight on May 14-15.
    os.environ["LTX2_DIT_EVAL_EVERY"] = "1"
if "LTX2_GEMMA_EVAL_EVERY" not in os.environ:
    os.environ["LTX2_GEMMA_EVAL_EVERY"] = "1"
# ---- end early bootstrap ====================================================

# ---- config ------------------------------------------------------------------
# All paths come from env vars set by the panel. LTX_GEMMA is the active
# generation's render text encoder; LTX_ENHANCE_GEMMA is deliberately separate
# because prompt rewriting needs Gemma 3's generative language-model head while
# LTX-2.5 conditions renders with a non-generative Gemma 4 tower.
_ROOT = Path(__file__).resolve().parent
_Q4_LOCAL_PATH = _ROOT / "mlx_models" / "ltx-2.3-mlx-q4"
MODEL_ID = os.environ.get(
    "LTX_MODEL",
    str(_Q4_LOCAL_PATH) if _Q4_LOCAL_PATH.is_dir() else "dgrauet/ltx-2.3-mlx-q4",
)
GEMMA_PATH = os.environ.get("LTX_GEMMA", "mlx-community/gemma-3-12b-it-4bit")
ENHANCE_GEMMA_PATH = os.environ.get(
    "LTX_ENHANCE_GEMMA", "mlx-community/gemma-3-12b-it-4bit"
)
IDLE_TIMEOUT = int(os.environ.get("LTX_IDLE_TIMEOUT", "1800"))
LOW_MEMORY = os.environ.get("LTX_LOW_MEMORY", "true").lower() in ("true", "1", "yes")
# Low-RAM block streaming (v4.9.9): stream transformer blocks from disk instead
# of holding the whole DiT resident. The panel sets this for small-memory Macs
# (see HELPER_LOW_RAM_STREAM in mlx_ltx_panel.py). Slower per step, far
# smaller peak — the difference between a 16 GB Mac finishing a clip and
# being jetsam-killed. Renders that carry LoRAs keep the exact unfused
# runtime branch and skip streaming (the streamer cannot carry per-block
# adapters; the library refuses rather than silently fusing on a quantized
# pack), so the flag is applied per pipeline, not blindly.
LOW_RAM_STREAM = os.environ.get("LTX_LOW_RAM_STREAM", "0").lower() in ("true", "1", "yes")


def _stream_for(loras) -> bool:
    return bool(LOW_RAM_STREAM) and not loras
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


def _log_memory_pressure() -> None:
    """Emit a log line with current memory stats for diagnosing GPU timeouts."""
    try:
        import subprocess, re
        total = int(subprocess.run(["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=1).stdout.strip())
        vm = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=1).stdout
        m = re.search(r"page size of (\d+)", vm)
        page_size = int(m.group(1)) if m else 16384
        def pages(name: str) -> int:
            mm = re.search(rf"{re.escape(name)}:\s+(\d+)", vm)
            return int(mm.group(1)) if mm else 0
        used_bytes = (pages("Pages active") + pages("Pages wired down")
                      + pages("Pages occupied by compressor")) * page_size
        used_gb = used_bytes / 1024**3
        pct = round(used_bytes / total * 100) if total else 0
        emit({"event": "log", "line": f"[mem] used={used_gb:.1f}G/{total/1024**3:.0f}G ({pct}%)"})
    except Exception:
        pass


# =============================================================================
# PER-PHASE MLX MEMORY — the number that decides whether a render fits
# =============================================================================
#
# WHY THIS EXISTS
# ---------------
# `_log_memory_pressure()` above measures the whole MACHINE via vm_stat. That
# answers "is this Mac under pressure", which is not the same question as
# "which part of OUR render is the spike". H3's staged runner has printed a
# per-phase peak since the port, and that is exactly why its memory questions
# get answered in minutes; LTX had no equivalent, so the 2026-08 allocator-cache
# win (a third of peak, free) was found by a global A/B rather than by looking
# at a profile. 38.8% of the fleet renders below 48 GB, where the peak phase is
# the difference between a finished clip and a swap storm — those users deserve
# a log line that names the phase, and so do we.
#
# Peak is RESET at each phase boundary so every line is that phase's own high
# water mark rather than a running maximum that only ever reports the worst
# phase. The run maximum is tracked separately and emitted once at the end.

_MLX_RUN_PEAK_GIB = 0.0


def _mlx_mem_reset_run() -> None:
    """Start a fresh per-run memory profile."""
    global _MLX_RUN_PEAK_GIB
    _MLX_RUN_PEAK_GIB = 0.0
    try:
        import mlx.core as mx                                    # noqa: PLC0415
        mx.reset_peak_memory()
    except Exception:                                            # noqa: BLE001
        pass


def _mlx_phase_mem(label: str) -> float | None:
    """Log this phase's MLX peak, then reset so the next phase stands alone.

    Returns the phase peak in GiB, or None when MLX cannot be queried (an old
    library, or a build without the memory API) — callers treat it as optional
    telemetry and never branch on it.
    """
    global _MLX_RUN_PEAK_GIB
    try:
        import mlx.core as mx                                    # noqa: PLC0415
        gib = 1024.0 ** 3
        peak = mx.get_peak_memory() / gib
        active = mx.get_active_memory() / gib
        cache = mx.get_cache_memory() / gib
    except Exception:                                            # noqa: BLE001
        return None
    if peak > _MLX_RUN_PEAK_GIB:
        _MLX_RUN_PEAK_GIB = peak
    emit({"event": "log",
          "line": f"[mlx] {label}: peak {peak:.2f} GiB · active {active:.2f} · cache {cache:.2f}"})
    try:
        mx.reset_peak_memory()
    except Exception:                                            # noqa: BLE001
        pass
    return peak


def _mlx_mem_summary() -> float:
    """Emit the whole run's peak — the one number a Compact Mac cares about."""
    if _MLX_RUN_PEAK_GIB > 0:
        emit({"event": "log",
              "line": f"[mlx] run peak {_MLX_RUN_PEAK_GIB:.2f} GiB"})
    return _MLX_RUN_PEAK_GIB


def _aggressive_cleanup_before_generate() -> None:
    """Minimize Metal heap fragmentation before pipeline generation."""
    try:
        from ltx_core_mlx.utils.memory import aggressive_cleanup as _ac
        _ac()
    except Exception:
        pass


# =============================================================================
# A2V AUDIO ADHESION — the slider that was never connected (on the Q8 path)
# =============================================================================
#
# The panel has shipped an "Audio conditioning strength" control (0.5-5.0,
# "Higher = stronger audio adhesion") since A2V landed. On the Q8 two-stage
# path the engine has never had a parameter by that name: the vendored
# `A2VidPipelineTwoStage.generate_and_save()` does not accept
# `audio_conditioning_scale`, so `_filter_unsupported_kwargs` DROPPED it on
# every render. Verified by introspection, not by reading:
#
#     DROPPED every a2v render: ['audio_conditioning_scale']
#
# The real knob there is `modality_scale` on MultiModalGuiderParams — "how
# strongly the model reacts to the modality perturbation" — and
# `_denoise_stage1` hardcodes it: video 3.0, audio bare default. Nobody had
# ever swept it, because the control that looked like it did was inert.
#
# The Q4 distilled path is DIFFERENT and this patch must stay away from it:
# `A2VidDistilledPipeline` (local a2vid_distilled.py) natively accepts
# `audio_conditioning_scale` (it amplifies the audio tokens before the DiT)
# and its overridden generate_and_save uses plain `denoise_loop` — no
# guiders, so no `modality_scale` exists on that path at all. The distilled
# branch passes the kwarg through natively and does not touch this state.
#
# Patched here rather than in the vendored engine so it costs no fork tag and
# an older pipeline simply keeps its own defaults.
# Holder dict, not a module global: the a2v branch lives deep inside the
# main loop where a `global` statement collides with earlier assignments.
_A2V_STATE = {"modality_scale": None}


def _a2v_modality_scale_value(raw):
    """Parse the slider off the job params: '14.0' -> 14.0; ''/None/0/'0'/
    garbage/negative -> None (= keep the engine's own 3.0 default)."""
    try:
        value = float(raw or 0)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _a2v_stage1_guider_params(guider_params_cls, cfg_scale, stg_scale):
    """The (video, audio) guider params for a2v stage 1, exactly as the
    vendored `_denoise_stage1` builds them — except `modality_scale` reads
    `_A2V_STATE` instead of the hardcoded 3.0. Split out so the test suite
    can drive the real MultiModalGuiderParams class through it."""
    ms = _A2V_STATE.get("modality_scale")
    ms = 3.0 if ms is None else float(ms)
    video_gp = guider_params_cls(cfg_scale=cfg_scale, stg_scale=stg_scale,
                                 rescale_scale=0.7, modality_scale=ms,
                                 stg_blocks=[28])
    audio_gp = guider_params_cls()
    return video_gp, audio_gp


def _install_a2v_modality_patch() -> bool:
    """Make `_denoise_stage1` read `_A2V_STATE['modality_scale']`. Idempotent."""
    try:
        from ltx_pipelines_mlx.a2vid_two_stage import (             # noqa: PLC0415
            A2VidPipelineTwoStage as _P,
        )
        from ltx_core_mlx.components.guiders import (               # noqa: PLC0415
            MultiModalGuiderParams as _GP,
            create_multimodal_guider_factory as _mk,
        )
        from ltx_pipelines_mlx.utils.samplers import (               # noqa: PLC0415
            guided_denoise_loop as _loop,
        )
    except Exception:                                                # noqa: BLE001
        return False
    if getattr(_P, "_phos_modality_patched", False):
        return True

    def _denoise_stage1(self, x0_model, video_state, audio_state, video_embeds,
                        audio_embeds, neg_video_embeds, neg_audio_embeds,
                        sigmas, cfg_scale=3.0, stg_scale=1.0):
        video_gp, audio_gp = _a2v_stage1_guider_params(_GP, cfg_scale, stg_scale)
        # Build the factories BEFORE the flush, exactly as upstream does — the
        # flush frees memory and the original ordering is part of its contract.
        video_factory = _mk(video_gp, negative_context=neg_video_embeds)
        audio_factory = _mk(audio_gp, negative_context=neg_audio_embeds)
        self._pre_denoise_flush(video_state, audio_state)
        return _loop(
            model=x0_model, video_state=video_state, audio_state=audio_state,
            video_text_embeds=video_embeds, audio_text_embeds=audio_embeds,
            video_guider_factory=video_factory,
            audio_guider_factory=audio_factory,
            sigmas=sigmas)

    _P._denoise_stage1 = _denoise_stage1
    _P._phos_modality_patched = True
    return True


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
    except (BrokenPipeError, ConnectionResetError):
        # 2026-05-31 review fix: the panel pipe is gone — the parent died or
        # closed our stdout. Holding on just orphans a 20-30 GB helper that
        # the idle reaper won't reap for up to LTX_IDLE_TIMEOUT (default 30
        # min), pinning unified memory the whole time. Exit immediately. Use
        # os._exit so we skip atexit/MLX-Metal teardown that can itself hang
        # on a half-torn pipe.
        try:
            os._exit(0)
        except Exception:
            pass
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
_a2v_pipe = None         # A2VidPipelineTwoStage (Q8 dev + distilled LoRA stage 2)
_a2v_model_dir = None    # which model the A2V pipe was built against
_a2v_lora_key: tuple | None = None  # LoRA fingerprint for current A2V cache
_a2v_weight_key: tuple[str, str] | None = None  # (dev_transformer, distilled_lora) names
_a2v_distilled_pipe = None   # A2VidDistilledPipeline (Q4 distilled, no CFG)
_a2v_distilled_model_dir = None
_pipe_lock = threading.Lock()


# ---- TeaCache for Extend (2026-05-18) -----------------------------------------
#
# Lightricks' RetakePipeline.extend() shares its denoise loop
# (ltx_pipelines_mlx.utils.samplers.guided_denoise_loop) with the standard
# two-stage T2V pipeline's Stage 1 — same dev transformer, same Euler sampler,
# same ltx2_schedule sigma layout. That stage is exactly what TeaCache is
# calibrated against in ti2vid_two_stages.LTX2_TEACACHE_COEFFICIENTS. The
# coefficients are reusable here; what's missing is the wiring — extend()
# doesn't pass `teacache=` into guided_denoise_loop today.
#
# Rather than fork retake.py we monkey-patch the symbol the retake module
# imports, so only calls FROM retake (i.e. extend() and retake()) go through
# the patched version. Two-stages, HQ, and any other caller are untouched.
#
# Activation is per-call: the extend action below sets _EXTEND_TC_CONFIG before
# pipe.extend_from_video() and clears it after. The helper inference loop is
# already serialized (single _is_busy gate) so the module-level state is safe.
#
# Default threshold matches Stage 1's `LTX2_TEACACHE_THRESH=0.5` — moderate
# 1.2× speedup with ~22% skip rate. Aggressive users can pass higher values
# in the job spec (1.0 → ~2×, 1.5 → ~3× per the upstream comment).
_EXTEND_TC_CONFIG: dict | None = None
_A2V_TC_CONFIG: dict | None = None  # mirror gate for A2V (2026-05-18 PM)

try:
    import ltx_pipelines_mlx.retake as _ltx_retake_mod
    import ltx_pipelines_mlx.a2vid_two_stage as _ltx_a2v_mod
    from ltx_pipelines_mlx.ti2vid_two_stages import _build_teacache_controller as _build_stage1_teacache
    _orig_guided_denoise_loop_for_retake = _ltx_retake_mod.guided_denoise_loop
    _orig_guided_denoise_loop_for_a2v = _ltx_a2v_mod.guided_denoise_loop

    def _guided_denoise_loop_with_extend_teacache(*args, teacache=None, **kwargs):
        cfg = _EXTEND_TC_CONFIG
        if teacache is None and cfg and cfg.get("enable"):
            sigmas = kwargs.get("sigmas")
            n_steps = len(sigmas) - 1 if sigmas else cfg.get("num_steps", 12)
            try:
                teacache = _build_stage1_teacache(n_steps, cfg.get("thresh"))
            except Exception:
                # Don't fail the gen if calibration is somehow unusable.
                teacache = None
        return _orig_guided_denoise_loop_for_retake(*args, teacache=teacache, **kwargs)

    def _guided_denoise_loop_with_a2v_teacache(*args, teacache=None, **kwargs):
        # Mirror of the extend wrapper, scoped to the a2vid_two_stage import
        # site. A2V's Stage 1 is the same dev-transformer Euler loop the
        # standard two-stages pipeline runs (same calibration applies). The
        # distilled Stage 2 uses denoise_loop (no guidance) which we don't
        # touch — it's already fast and TeaCache wasn't designed for it.
        cfg = _A2V_TC_CONFIG
        if teacache is None and cfg and cfg.get("enable"):
            sigmas = kwargs.get("sigmas")
            n_steps = len(sigmas) - 1 if sigmas else cfg.get("num_steps", 10)
            try:
                teacache = _build_stage1_teacache(n_steps, cfg.get("thresh"))
            except Exception:
                teacache = None
        return _orig_guided_denoise_loop_for_a2v(*args, teacache=teacache, **kwargs)

    _ltx_retake_mod.guided_denoise_loop = _guided_denoise_loop_with_extend_teacache
    _ltx_a2v_mod.guided_denoise_loop = _guided_denoise_loop_with_a2v_teacache
    _EXTEND_TC_PATCH_OK = True
    _A2V_TC_PATCH_OK = True
except Exception as _tc_patch_exc:
    print(f"[warm-helper] TeaCache patch skipped: {_tc_patch_exc}", flush=True)
    _EXTEND_TC_PATCH_OK = False
    _A2V_TC_PATCH_OK = False


def release_pipelines(keep_kind=None):
    """Free every loaded pipeline except the one matching keep_kind.

    Each pipeline holds ~22 GB (Q4) or ~30 GB (Q8 dev) of weights. Holding
    two or more simultaneously on a 64 GB Mac OOMs the helper. Only one
    family stays resident at a time — switching mode reloads, but renders
    actually finish instead of getting SIGKILL'd by macOS.

    keep_kind ∈ {'t2v', 'i2v', 'extend', 'hq', 'keyframe', 'a2v'} or None (free all).
    Caller must hold _pipe_lock.
    """
    global _t2v_pipe, _i2v_pipe, _extend_pipe, _hq_pipe, _kf_pipe, _hq_model_dir, _kf_model_dir
    global _a2v_pipe, _a2v_model_dir
    global _a2v_distilled_pipe, _a2v_distilled_model_dir
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
    if keep_kind != "a2v" and _a2v_pipe is not None:
        _a2v_pipe = None; _a2v_model_dir = None; freed.append("A2V")
    if keep_kind != "a2v_distilled" and _a2v_distilled_pipe is not None:
        _a2v_distilled_pipe = None; _a2v_distilled_model_dir = None; freed.append("A2V-distilled")
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
# Dev-transformer filename the cached Extend pipe was built with. Named per
# generation by the panel; same string for 2.3 and 2.5 today, in the cache key
# so a future rename cannot be served by a stale pipe.
_extend_dev_name: str | None = None


def _lora_fingerprint_base(loras: list[dict] | None) -> tuple:
    """Stable hashable representation of a LoRA list. Order-insensitive
    so [{a,1},{b,2}] and [{b,2},{a,1}] hash to the same set — fusing
    is commutative."""
    if not loras:
        return ()
    return tuple(sorted(
        (str(l.get("path", "")), float(l.get("strength", 1.0)))
        for l in loras
    ))

def _lora_fingerprint(loras: list[dict] | None) -> tuple:
    """The pipeline cache key: the LoRA set AND whether this job streams
    blocks (v4.9.9). A LoRA-free job after a LoRA job must not reuse a
    pipeline built the other way round."""
    return _lora_fingerprint_base(loras) + (("stream",) if _stream_for(loras) else ())



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
    except (ValueError, OSError) as exc:
        # huggingface_hub's HFValidationError is a ValueError: "Repo id must be
        # in the form 'repo_name' or 'namespace/repo_name'". A user never typed
        # an id — the value is a LoRA path that no longer resolves (moved,
        # renamed, a relative path with a space). Name the file, not the rule.
        if type(exc).__name__ in ("HFValidationError",) or "Repo id must be" in str(exc):
            raise FileNotFoundError(
                f"LoRA file not found: {p} — it is not on disk and it is not a Hugging Face "
                f"repo id. It was probably moved or deleted after it was picked; pick it again.") from exc
        raise
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


def _construct_pipeline(cls, **kwargs):
    """Build a pipeline, passing only the constructor kwargs THIS vendored
    class accepts.

    4.10.2 added `low_ram_streaming=` to every pipeline construction. The
    text/image pipelines take it through `BasePipeline`; `RetakePipeline`
    (the Extend lane) declares its own `__init__` without it, so every Extend
    render on 4.10.2–4.10.5 died at construction with "unexpected keyword
    argument 'low_ram_streaming'" — 35 failures from one install in the fleet
    on the day 4.10.4 shipped. A dropped kwarg is logged, never fatal."""
    accepted = _filter_unsupported_kwargs(cls.__init__, dict(kwargs))
    dropped = sorted(set(kwargs) - set(accepted))
    if dropped:
        emit({"event": "log",
              "line": f"{cls.__name__} does not take {', '.join(dropped)} on this engine; "
                      f"constructing without."})
    return cls(**accepted)


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
_A2V_FRAME_RATE_PATCH_INSTALLED = False


# NOTE — an earlier `_PostDecodeWatchdog` daemon-thread class lived
# here. It cannot fire under the MLX/Metal post-decode hang: the
# completion-handler chain blocks every Python thread's GIL access,
# so the daemon thread is starved by the very thing it was meant to
# escape. The rescue lives in the panel (mlx_ltx_panel.py,
# `WarmHelper._build_post_decode_panic`) where SIGKILL across the
# subprocess boundary works regardless of what the helper is holding
# internally. See ROADMAP for the full diagnosis.



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


def _install_a2v_frame_rate_patch() -> None:
    """Make ``combined_image_conditionings`` tolerate the missing ``frame_rate``
    kwarg that ``a2vid_two_stage.py`` neglects to pass.

    Upstream LTX-2 MLX v0.14.0 made ``frame_rate`` a keyword-only required
    arg on ``utils._orchestration.combined_image_conditionings``, but
    ``a2vid_two_stage.A2VidPipelineTwoStage.generate`` still calls it
    without forwarding ``frame_rate``. Result: every A2V render with a
    reference image dies with::

        TypeError: combined_image_conditionings() missing 1 required
        keyword-only argument: 'frame_rate'

    Fix: wrap ``combined_image_conditionings`` so that when ``frame_rate``
    isn't supplied we default to ``24.0`` (the project's render FPS, and
    the only value any caller uses today). Other call sites that DO pass
    ``frame_rate`` (TI2VidTwoStages*, IC-LoRA, lipdub) are unaffected
    because their explicit kwarg overrides the default.

    Why a wrapper instead of editing the vendored file: the ltx-2-mlx
    checkout is pinned to a tag and any direct edit gets clobbered on
    re-clone. The patch is idempotent so repeated helper boots are safe.

    As of the v0.14.19 pin this shim is INERT on the paths it was written
    for — upstream 0.14.15 (#56) forwards ``frame_rate`` from the A2V and
    lipdub call sites. It stays as a belt-and-braces default because
    ``frame_rate`` is still keyword-only *required* on
    ``combined_image_conditionings``, so any future caller that forgets it
    would again die mid-render rather than degrade. Do not read its
    presence as evidence the upstream bug is still live.
    """
    global _A2V_FRAME_RATE_PATCH_INSTALLED
    if _A2V_FRAME_RATE_PATCH_INSTALLED:
        return
    try:
        import ltx_pipelines_mlx.utils._orchestration as _orch
    except Exception as exc:  # noqa: BLE001 — log + give up cleanly
        print(f"[warm-helper] A2V frame_rate patch skipped (no _orchestration): {exc}",
              flush=True)
        return
    _orig = _orch.combined_image_conditionings
    # 2026-05-28 fix (#5 claude3d M5): upstream ltx_pipelines_mlx is a moving
    # target. v0.14.0 required `frame_rate` as a keyword-only kwarg (our
    # original patch reason). Some v0.14.1+ point release DROPPED `frame_rate`
    # entirely — passing it now raises
    #   TypeError: combined_image_conditionings() got an unexpected keyword
    #   argument 'frame_rate'
    # which is the exact symptom on @claude3d's M5 install. Probe the live
    # signature ONCE at install time and route accordingly. If the user's
    # ltx-2-mlx is post-removal we silently strip `frame_rate`; if it's pre-
    # removal (or upstream re-adds it later) we keep forwarding. Robust to
    # future signature flips.
    import inspect as _inspect_a2v
    try:
        _orig_accepts_fr = "frame_rate" in _inspect_a2v.signature(_orig).parameters
    except (TypeError, ValueError):
        # Builtins or C-extension may refuse inspection — assume yes for
        # backward compatibility with the original patch behavior.
        _orig_accepts_fr = True

    def _wrapped(*args, frame_rate: float = 24.0, **kwargs):
        if _orig_accepts_fr:
            return _orig(*args, frame_rate=frame_rate, **kwargs)
        # Upstream removed frame_rate — strip it before forwarding so we
        # don't trip the post-removal path's TypeError.
        kwargs.pop("frame_rate", None)
        return _orig(*args, **kwargs)

    _orch.combined_image_conditionings = _wrapped
    # The A2V module already imported the original at module-load time
    # (line 194 in a2vid_two_stage.py: `from ltx_pipelines_mlx.utils.
    # _orchestration import combined_image_conditionings`). Patch THAT
    # binding too so the existing import picks up the wrapper.
    try:
        import ltx_pipelines_mlx.a2vid_two_stage as _a2v_mod
        if hasattr(_a2v_mod, "combined_image_conditionings"):
            _a2v_mod.combined_image_conditionings = _wrapped
    except Exception:  # noqa: BLE001 — module not imported yet → patch
        # will take effect on its later import (uses _orchestration name)
        pass
    _A2V_FRAME_RATE_PATCH_INSTALLED = True


def _install_lora_fusion_patches() -> None:
    """Install fail-closed native LoRA guards, or repair an older engine.

    Current engines already own the correct quantization-aware LoRA route; the
    first branch below preserves it and verifies the live attachment report.
    The rest of this function is the legacy compatibility path for engines
    whose subclasses bypassed ``_pending_loras`` entirely.

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

    # v0.14.19+ltx25.3 routes every modern pipeline through BasePipeline's
    # native LoRA-aware transformer loader.  The historical wrapper below was
    # written for older subclasses that bypassed that seam; leaving it active
    # now bypasses the native `auto -> unfused` path and re-quantizes character
    # deltas into Q4/Q8 weights.  That can erase identity while still producing
    # a perfectly plausible video.  Prefer the native loader when it exists,
    # and guard both its header routing and the modules it actually attached.
    try:
        from ltx_pipelines_mlx._base import BasePipeline as _NativeBasePipeline
    except ImportError:
        _NativeBasePipeline = None
    if (
        _NativeBasePipeline is not None
        and hasattr(_NativeBasePipeline, "_load_transformer_with_optional_streaming")
        and hasattr(_NativeBasePipeline, "_attach_pending_loras")
    ):
        if not getattr(_NativeBasePipeline, "_phosphene_lora_guard", False):
            _native_load_transformer = (
                _NativeBasePipeline._load_transformer_with_optional_streaming
            )

            def _guarded_native_load_transformer(self, transformer_path):
                pending = list(getattr(self, "_pending_loras", None) or [])
                active = [(str(path), float(strength))
                          for path, strength in pending if float(strength) != 0.0]
                self._phosphene_lora_preflight = []
                if active:
                    from lora_compat import (
                        validate_adapter_effects,
                        validate_lora_stack,
                    )

                    reports = validate_lora_stack(active, transformer_path)
                    # Names are not effect (#62). A character adapter can match
                    # every key, attach on every module, report a green tally —
                    # and still be numerically negligible, which two users spent
                    # days chasing because nothing in the log ever said how much
                    # weight the file actually carries. Now every render says it.
                    validate_adapter_effects(
                        active,
                        reporter=lambda line: emit({"event": "log", "line": line}),
                    )
                    self._phosphene_lora_preflight = list(zip(
                        reports, (strength for _, strength in active)
                    ))
                return _native_load_transformer(self, transformer_path)

            def _guarded_native_attach(self, dit, lora_paths):
                # Lazy import, matching the rest of this file: mlx.core is not
                # importable at module scope in every environment the helper
                # is parsed in, and this closure is the only scope that uses it.
                import mlx.core as mx
                from lora_compat import (
                    LoraCompatibilityError,
                    validate_runtime_application,
                )
                from ltx_core_mlx.loader.runtime_loras import load_and_attach_loras
                from ltx_core_mlx.utils.memory import aggressive_cleanup
                from ltx_pipelines_mlx.utils._orchestration import resolve_lora_path

                active = [(resolve_lora_path(path), float(strength))
                          for path, strength in lora_paths
                          if float(strength) != 0.0]
                if not active:
                    return
                expected = list(getattr(self, "_phosphene_lora_preflight", []) or [])
                if len(expected) != len(active):
                    names = ", ".join(Path(path).name for path, _ in active)
                    raise LoraCompatibilityError(
                        f"LoRA preflight state was lost before live attachment "
                        f"({names}). Rendering was refused before it could "
                        "produce a LoRA-free result."
                    )
                report = load_and_attach_loras(dit, active, verbose=self.verbose)
                validate_runtime_application(
                    expected,
                    (module.name for module in report.applied),
                    reporter=lambda line: emit({"event": "log", "line": line}),
                )
                # Mirror BasePipeline._attach_pending_loras: force the adapter
                # parameters out of MLX's lazy graph before cleanup.
                mx.eval(dit.parameters())
                aggressive_cleanup()

            _NativeBasePipeline._load_transformer_with_optional_streaming = (
                _guarded_native_load_transformer
            )
            _NativeBasePipeline._attach_pending_loras = _guarded_native_attach
            _NativeBasePipeline._phosphene_lora_guard = True
        _LORA_PATCH_INSTALLED = True
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

    from ltx_core_mlx.model.transformer.model import LTXModel, LTXModelConfig
    from ltx_core_mlx.utils.memory import aggressive_cleanup
    from ltx_core_mlx.utils.weights import apply_quantization, load_split_safetensors

    def _new_dit(tx_path):
        """An LTXModel shaped by the PACK'S OWN CONFIG, exactly as the library
        builds it (`_base.py`: LTXModel(LTXModelConfig.from_checkpoint_dir(...))).

        This used to be a bare `LTXModel()`, i.e. LTXModelConfig's DEFAULTS. On
        2.3 that was harmless because every default matched. On 2.5 it is not:
        the packs set `use_keyframes_abs_pos_embedding: true` and their
        checkpoints carry `transformer.keyframes_abs_pos_embedding`, while the
        default is False — so the model was built WITHOUT that parameter and
        load_weights refused the checkpoint with

            Received 1 parameters not in model: keyframes_abs_pos_embedding.

        Every LTX-2.5 render with a LoRA attached died there, which is every
        character render on the distilled path. It went unnoticed because the
        panel had been routing Balanced characters to the HQ pipeline (a 2.3
        speed optimisation, now version-gated), and that path builds its DiT
        somewhere else.

        Ask the pack, never assume: a future generation that adds another
        config-gated parameter is then right by construction rather than one
        more line here."""
        try:
            return LTXModel(LTXModelConfig.from_checkpoint_dir(Path(tx_path).parent))
        except Exception as exc:                      # noqa: BLE001
            # Degrade to the previous behaviour rather than failing the render:
            # an older library without from_checkpoint_dir, or a pack with no
            # readable config, is exactly the 2.3 case this used to serve.
            emit({"event": "log",
                  "line": f"WARN: could not read the transformer config next to "
                          f"{os.path.basename(str(tx_path))} ({exc}); building "
                          f"the DiT from defaults."})
            return LTXModel()

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
                self.dit = _new_dit(tx_path)
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
                dit = _new_dit(tx_path)
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


def _preflight_distilled_loras(
    loras: list[tuple[str, float]], model_dir: str | Path
) -> None:
    """Fail closed for IC-LoRA pipelines that own their fusion internally."""
    from lora_compat import (
        resolve_distilled_transformer,
        validate_adapter_effects,
        validate_lora_stack,
    )

    active = [(path, float(strength)) for path, strength in loras
              if float(strength) != 0.0]
    if not active:
        return
    validate_adapter_effects(
        active, reporter=lambda line: emit({"event": "log", "line": line})
    )
    transformer = resolve_distilled_transformer(model_dir)
    if transformer is None:
        raise RuntimeError(
            f"Cannot validate LoRAs: no distilled transformer found in {model_dir}."
        )
    reports = validate_lora_stack(active, transformer)
    for index, ((_, strength), report) in enumerate(zip(active, reports), start=1):
        emit({
            "event": "log",
            "line": (
                f"LoRA[{index}] strength={float(strength):.2f} "
                f"PREFLIGHT={report.matched_tensors}/{report.declared_tensors} "
                f"tensors file={report.lora_path.name}"
            ),
        })


_extend_model_dir: str | None = None
# The base pipelines cache their model_dir the same way Extend does, so a
# job that names a different model version rebuilds instead of silently
# rendering on the previous one's weights.
_t2v_model_dir: str | None = None
_i2v_model_dir: str | None = None


def get_pipe(kind: str, loras: list[dict] | None = None,
             model_dir: str | None = None,
             dev_transformer: str | None = None):
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
    fingerprint so a model_dir flip rebuilds the pipe.

    2026-08-11 — `model_dir` now applies to T2V and I2V too. It used to be
    Extend-only, which meant the BASE render was pinned to whatever
    LTX_MODEL this helper process was spawned with: a module-level
    singleton, chosen once, unreachable per job. That is fine while there
    is exactly one LTX generation on disk and fatal the moment there are
    two — the panel would have had to kill and respawn the helper to
    switch versions, mid-queue, with the weights still resident. Passing
    None keeps the old behaviour exactly (fall back to MODEL_ID), which is
    what every caller that has not been taught about versions still does.

    The helper stays version-AGNOSTIC on purpose: it takes a directory,
    never a version id. Which generation a directory holds is the
    checkpoint's business (2.5 makes the vendored config metadata-driven),
    and the panel's registry is the thing that knows which directory to
    name."""
    global _t2v_pipe, _i2v_pipe, _extend_pipe
    global _t2v_lora_key, _i2v_lora_key, _extend_lora_key
    global _extend_model_dir, _t2v_model_dir, _i2v_model_dir
    global _extend_dev_name
    try:
        from ltx_core_mlx.utils.memory import aggressive_cleanup as _ac
    except Exception:
        _ac = lambda: None
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
    _install_a2v_frame_rate_patch()  # A2V missing frame_rate= on combined_image_conditionings

    fp = _lora_fingerprint(loras)

    with _pipe_lock:
        # Free any other pipelines before loading a new one — strict
        # one-pipeline-at-a-time policy keeps memory bounded.
        release_pipelines(keep_kind=kind)
        if kind == "i2v":
            i2v_dir = model_dir or MODEL_ID
            if (_i2v_pipe is None or _i2v_lora_key != fp
                    or _i2v_model_dir != i2v_dir):
                if _i2v_pipe is not None:
                    why = ("LoRA set changed" if _i2v_lora_key != fp
                           else "model_dir changed")
                    emit({"event": "log",
                          "line": f"{why}; reloading I2V pipeline."})
                    _i2v_pipe = None
                    _ac()
                emit({"event": "log",
                      "line": "Loading I2V pipeline (first job is the slow one)..."})
                pipe = _construct_pipeline(
                    ImageToVideoPipeline,
                    model_dir=i2v_dir, gemma_model_id=GEMMA_PATH, low_memory=LOW_MEMORY,
                    low_ram_streaming=_stream_for(loras),
                )
                _attach_loras(pipe, loras)
                _i2v_pipe = pipe
                _i2v_lora_key = fp
                _i2v_model_dir = i2v_dir
            return _i2v_pipe
        if kind == "extend":
            ext_dir = model_dir or MODEL_ID
            # Named by the panel per generation; None keeps RetakePipeline's
            # own default, which is the same string today. Part of the cache
            # key for the same reason model_dir is.
            ext_dev = dev_transformer or "transformer-dev.safetensors"
            if (_extend_pipe is None
                    or _extend_lora_key != fp
                    or _extend_model_dir != ext_dir
                    or _extend_dev_name != ext_dev):
                if _extend_pipe is not None:
                    why = "LoRA set changed" if _extend_lora_key != fp else "model_dir changed"
                    emit({"event": "log",
                          "line": f"{why}; reloading Extend pipeline."})
                    _extend_pipe = None
                    _ac()
                emit({"event": "log",
                      "line": f"Loading Extend pipeline (heavier — uses dev transformer at {ext_dir})..."})
                pipe = _construct_pipeline(
                    ExtendPipeline,
                    model_dir=ext_dir, gemma_model_id=GEMMA_PATH, low_memory=LOW_MEMORY,
                    low_ram_streaming=_stream_for(loras),
                    dev_transformer=ext_dev,
                )
                _attach_loras(pipe, loras)
                _extend_pipe = pipe
                _extend_lora_key = fp
                _extend_model_dir = ext_dir
                _extend_dev_name = ext_dev
            return _extend_pipe
        # t2v
        t2v_dir = model_dir or MODEL_ID
        if (_t2v_pipe is None or _t2v_lora_key != fp
                or _t2v_model_dir != t2v_dir):
            if _t2v_pipe is not None:
                why = ("LoRA set changed" if _t2v_lora_key != fp
                       else "model_dir changed")
                emit({"event": "log",
                      "line": f"{why}; reloading T2V pipeline."})
                _t2v_pipe = None
                _ac()
            emit({"event": "log",
                  "line": "Loading T2V pipeline (first job is the slow one)..."})
            pipe = _construct_pipeline(
                TextToVideoPipeline,
                model_dir=t2v_dir, gemma_model_id=GEMMA_PATH, low_memory=LOW_MEMORY,
                low_ram_streaming=_stream_for(loras),
            )
            _attach_loras(pipe, loras)
            _t2v_pipe = pipe
            _t2v_lora_key = fp
            _t2v_model_dir = t2v_dir
        return _t2v_pipe


_hq_lora_key: str | None = None
# (dev_transformer, distilled_lora) the cached HQ pipe was built with. A
# generation switch changes the distilled LoRA filename, and a pipe reused
# across that switch would refine stage 2 with the previous generation's LoRA
# — which fuses, renders, and is wrong with nothing to attribute it to.
_hq_weight_key: tuple[str | None, str | None] | None = None


def _checkpoint_generation(model_dir: str) -> tuple[int, ...]:
    """The generation of the checkpoint in ``model_dir``, e.g. ``(2, 5)``.

    Returns the empty tuple when it cannot be read, which every
    generation-keyed default in the vendored library treats as "older than
    anything" and therefore answers with the legacy value. A pack whose
    config we cannot parse must render exactly as it did before.

    Reads `config.json` / `embedded_config.json` — the sidecars the quantiser
    writes next to the weights — and parses with the LIBRARY's own
    ``parse_model_version``, so the panel cannot disagree with the pipeline
    about what "2.5" means. Costs one small JSON read, no weights.
    """
    try:
        from ltx_core_mlx.model.transformer.model import parse_model_version
    except Exception:  # noqa: BLE001 — older library: no generations exist
        return ()
    import json as _json
    from pathlib import Path as _Path
    base = _Path(str(model_dir))
    for name in ("config.json", "embedded_config.json"):
        try:
            raw = _json.loads((base / name).read_text())
        except Exception:  # noqa: BLE001 — absent/unreadable/not JSON
            continue
        version = raw.get("model_version")
        if version is None:
            version = (raw.get("transformer") or {}).get("model_version")
        if version is None:
            continue
        try:
            return tuple(parse_model_version(str(version)))
        except Exception:  # noqa: BLE001
            continue
    return ()


def _hq_modality_scale(model_dir: str) -> float:
    """The HQ path's default ``modality_scale`` for this checkpoint.

    The detail-guidance slider (STG) has to hand the pipeline EXPLICIT guider
    params to make `stg_blocks=[28]` bite, and those params carry every other
    field too. Hardcoding `modality_scale=3.0` there meant the slider changed
    TWO things at once on LTX-2.5: the pipeline's own default became 1.0 in
    the v4.0 pin (isolated-modality guidance off — an owner-graded output
    change worth −60.7 s), but any render with detail guidance above 0 would
    silently re-enable the dead pass and pay for it again.

    So the value is asked of the library rather than restated here: the panel
    and `TI2VidTwoStagesHQPipeline` resolve it through the SAME function, and
    a future generation cannot make the two disagree. On a library that
    predates the keying — any older pin — the import fails and 3.0 comes back,
    which is byte-identical to what shipped before.
    """
    try:
        from ltx_pipelines_mlx.ti2vid_two_stages_hq import resolve_modality_scale
    except Exception:  # noqa: BLE001 — pre-25b9b8e library: 3.0 was the only value
        return 3.0
    try:
        return float(resolve_modality_scale(_checkpoint_generation(model_dir)))
    except Exception:  # noqa: BLE001 — never let this block a render
        return 3.0


def _thin_table_caps(model_dir: str) -> tuple[int, int] | None:
    """(stage1, stage2) step ceilings for the FIXED tables this checkpoint
    thins, or None when they cannot be read."""
    try:
        from ltx_pipelines_mlx.scheduler import distilled_presets_for
    except Exception:  # noqa: BLE001 — older library: leave requests alone
        return None
    try:
        vendor = (distilled_presets_for(_checkpoint_generation(model_dir)) or {}).get("vendor")
        if not vendor:
            return None
        return (max(1, len(vendor.stage1) - 1), max(1, len(vendor.stage2) - 1))
    except Exception:  # noqa: BLE001 — never block a render on a probe
        return None


def _thin_cap(model_dir: str, key: str, want: int) -> int:
    """Clamp a step count to what THIS checkpoint's fixed table can serve.

    A step count on a thinning lane means "give me N of the checkpoint's own
    steps" — it thins a fixed table. Asking for more than the table holds is
    asking it to PAD, which is a different operation with a different answer,
    so `thin_sigmas` refuses it. Correctly: a padded schedule is not a longer
    render, it is a made-up one.

    The panel must therefore never ask, and this is the belt behind that.
    It clamps the RESOLVED value — after a lane's own `p.get(key, default)` —
    because the value that broke Colorize came from a default, not from the
    form. An earlier version of this guard only inspected incoming params and
    sailed straight past HDR's own `stage1_steps=10`.

    Clamping rather than raising is deliberate: the user asked for a finer
    schedule than the checkpoint can serve, and its finest with a log line is a
    better answer than a dead render four minutes in.

    Only ever called from the lanes that thin a FIXED table (ICLoraPipeline:
    Colorize / Restore / Ingredients / Control / HDR). The CFG-dev lanes
    (`generate_hq` res_2s, `generate_keyframe`) COMPUTE a schedule of whatever
    length they are asked for, and their 10 and 20 are graded values — clamping
    those would silently change owner-approved output.
    """
    caps = _thin_table_caps(model_dir)
    if not caps:
        return want
    cap = caps[0] if key.startswith("stage1") else caps[1]
    if want > cap:
        emit({"event": "log",
              "line": f"{key}={want} exceeds what this checkpoint's schedule "
                      f"holds ({cap} steps); using {cap}. A step count thins "
                      f"the checkpoint's own table — it cannot pad it."})
        return cap
    return want


def get_hq_pipe(model_dir: str, loras: list[dict] | None = None,
                dev_transformer: str | None = None,
                distilled_lora: str | None = None):
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

    2026-08-12 — `dev_transformer` / `distilled_lora` are the panel's per-job
    names for the two files this pipeline loads out of `model_dir`. Passing
    None keeps the vendored class's own defaults, which are the 2.3 filenames;
    that is the byte-identical legacy behaviour for any caller that has not
    been taught about generations. The panel now always names them, because
    2.5's distilled LoRA is `...-450` where 2.3's is `...-384` and the vendored
    default would send the loader looking for a file the pack does not hold.

    Cached alongside model_dir and the LoRA fingerprint: a filename flip is a
    different pipeline, and a pipeline reused across a generation switch would
    quietly refine with the wrong LoRA.
    """
    global _hq_pipe, _hq_model_dir, _hq_lora_key, _hq_weight_key
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
    _install_a2v_frame_rate_patch()  # A2V missing frame_rate= on combined_image_conditionings

    fp = _lora_fingerprint(loras)

    try:
        from ltx_core_mlx.utils.memory import aggressive_cleanup as _ac
    except Exception:
        _ac = lambda: None
    weight_key = (dev_transformer, distilled_lora)
    with _pipe_lock:
        release_pipelines(keep_kind="hq")
        if (_hq_pipe is None
                or _hq_model_dir != model_dir
                or _hq_lora_key != fp
                or _hq_weight_key != weight_key):
            if _hq_pipe is not None:
                if _hq_lora_key != fp:
                    why = "LoRA set changed"
                elif _hq_weight_key != weight_key:
                    why = "HQ weight filenames changed"
                else:
                    why = "model_dir changed"
                emit({"event": "log", "line": f"{why}; reloading HQ pipeline."})
                _hq_pipe = None
                _ac()
            emit({"event": "log", "line": f"Loading HQ pipeline (Q8 dev model — {model_dir})..."})
            # Only pass what the caller named: an explicit None would override
            # the vendored default with nothing, and `distilled_lora=None`
            # makes _fuse_distilled_lora resolve the stem of None.
            hq_kwargs = {}
            if dev_transformer:
                hq_kwargs["dev_transformer"] = dev_transformer
            if distilled_lora:
                hq_kwargs["distilled_lora"] = distilled_lora
            _hq_pipe = TwoStageHQPipeline(
                model_dir=model_dir, gemma_model_id=GEMMA_PATH, low_memory=LOW_MEMORY,
                **hq_kwargs,
            )
            _attach_loras(_hq_pipe, loras)
            _hq_model_dir = model_dir
            _hq_lora_key = fp
            _hq_weight_key = weight_key
        return _hq_pipe


# Keyframe (FFLF) pipeline — two images locked at frame 0 + frame N-1, model
# interpolates between. Uses Q8 dev transformer + distilled LoRA stage 2.
_kf_pipe = None
_kf_model_dir = None
_kf_weight_key: tuple[str, str] | None = None


def get_kf_pipe(model_dir: str, dev_transformer: str | None = None,
                distilled_lora: str | None = None):
    """Returns the KeyframeInterpolationPipeline lazily.

    Keyframe REQUIRES explicit dev_transformer + distilled_lora at init time.
    The distilled-only path "hallucinates unrelated content during
    interpolation" — pipeline raises if you skip these.

    The names used to be the two literals from dgrauet/ltx-2.3-mlx-q8, written
    here. They now come from the panel per job, because the distilled LoRA's
    filename carries its alpha/rank and that number changes with the
    generation (2.3: 384, 2.5: 450). The literals remain as the fallback so a
    caller that does not name them behaves exactly as before.
    """
    global _kf_pipe, _kf_model_dir, _kf_weight_key
    from ltx_pipelines_mlx.keyframe_interpolation import KeyframeInterpolationPipeline

    # Install the same runtime patches that get_pipe / get_hq_pipe / get_a2v_pipe
    # install. Before 2026-05-25 this getter was the only one that skipped them,
    # so FFLF renders crashed with
    # `VideoDecoder.decode_and_stream() got an unexpected keyword argument
    # 'frame_rate'` on machines that downloaded a post-rename upstream build
    # (reported by @oo2music in #9). The install funcs are all idempotent.
    _install_lora_fusion_patches()
    _install_video_decoder_patch()
    _install_a2v_frame_rate_patch()

    dev_name = dev_transformer or "transformer-dev.safetensors"
    lora_name = distilled_lora or "ltx-2.3-22b-distilled-lora-384.safetensors"
    weight_key = (dev_name, lora_name)
    with _pipe_lock:
        release_pipelines(keep_kind="keyframe")
        if _kf_pipe is None or _kf_model_dir != model_dir or _kf_weight_key != weight_key:
            emit({"event": "log", "line": f"Loading Keyframe pipeline (Q8 dev model — {model_dir})..."})
            _kf_pipe = KeyframeInterpolationPipeline(
                model_dir=model_dir,
                gemma_model_id=GEMMA_PATH,
                low_memory=LOW_MEMORY,
                dev_transformer=dev_name,
                distilled_lora=lora_name,
                distilled_lora_strength=1.0,
            )
            _kf_model_dir = model_dir
            _kf_weight_key = weight_key
        return _kf_pipe


# Audio-to-Video (A2V) pipeline — drives video from an input audio waveform.
# Stage 1: Q8 dev transformer + CFG at half resolution, audio frozen.
# Stage 2: dev + distilled LoRA fused, refine video at full resolution.
# Image is OPTIONAL — passing it conditions frame 0 the same way I2V does, so
# the same pipeline serves both "pure A2V" (audio + prompt → video) and
# "Image + Audio" (audio + still + prompt → audio-driven video that opens on
# the reference frame). The upstream pipeline writes the original audio (not
# VAE-decoded) onto the final mp4, so no panel-side mux is needed.
def get_a2v_pipe(model_dir: str, loras: list[dict] | None = None,
                 dev_transformer: str | None = None,
                 distilled_lora: str | None = None):
    """Returns A2VidPipelineTwoStage lazily.

    A2V REQUIRES dev_transformer + distilled_lora at init time (same pattern
    as Keyframe), and like Keyframe those names now come from the panel per
    job — the distilled LoRA's filename carries its alpha/rank, which changes
    with the generation. The 2.3 literals remain as the fallback.

    The pipeline inherits from TI2VidTwoStagesPipeline so LoRA fusion goes
    through the same `_pending_loras` hook the patched load() consumes. Cached
    on (model_dir, lora_fingerprint, weight filenames) so a LoRA swap, a
    model-dir flip or a generation switch rebuilds; otherwise reuses.
    """
    global _a2v_pipe, _a2v_model_dir, _a2v_lora_key, _a2v_weight_key
    from ltx_pipelines_mlx.a2vid_two_stage import A2VidPipelineTwoStage

    _install_lora_fusion_patches()
    _install_video_decoder_patch()
    _install_a2v_frame_rate_patch()  # A2V missing frame_rate= on combined_image_conditionings
    fp = _lora_fingerprint(loras)

    try:
        from ltx_core_mlx.utils.memory import aggressive_cleanup as _ac
    except Exception:
        _ac = lambda: None
    dev_name = dev_transformer or "transformer-dev.safetensors"
    lora_name = distilled_lora or "ltx-2.3-22b-distilled-lora-384.safetensors"
    weight_key = (dev_name, lora_name)
    with _pipe_lock:
        release_pipelines(keep_kind="a2v")
        if (_a2v_pipe is None
                or _a2v_model_dir != model_dir
                or _a2v_lora_key != fp
                or _a2v_weight_key != weight_key):
            if _a2v_pipe is not None:
                if _a2v_lora_key != fp:
                    why = "LoRA set changed"
                elif _a2v_weight_key != weight_key:
                    why = "HQ weight filenames changed"
                else:
                    why = "model_dir changed"
                emit({"event": "log", "line": f"{why}; reloading A2V pipeline."})
                _a2v_pipe = None
                _ac()
            emit({"event": "log",
                  "line": f"Loading A2V pipeline (Q8 dev + distilled LoRA — {model_dir})..."})
            _a2v_pipe = A2VidPipelineTwoStage(
                model_dir=model_dir,
                gemma_model_id=GEMMA_PATH,
                low_memory=LOW_MEMORY,
                dev_transformer=dev_name,
                distilled_lora=lora_name,
                distilled_lora_strength=1.0,
            )
            _attach_loras(_a2v_pipe, loras)
            _a2v_model_dir = model_dir
            _a2v_lora_key = fp
            _a2v_weight_key = weight_key
        return _a2v_pipe


def get_a2v_distilled_pipe(model_dir: str):
    """Returns A2VidDistilledPipeline lazily.

    Q4-compatible distilled pipeline — no dev transformer, no CFG, 8+3 steps.
    Cached on model_dir so switching between Q4 and Q8 paths rebuilds.
    """
    global _a2v_distilled_pipe, _a2v_distilled_model_dir
    try:
        from a2vid_distilled import A2VidDistilledPipeline
    except ModuleNotFoundError:
        raise RuntimeError(
            "A2V distilled module not found — run Update to install it."
        ) from None

    _install_a2v_frame_rate_patch()

    try:
        from ltx_core_mlx.utils.memory import aggressive_cleanup as _ac
    except Exception:
        _ac = lambda: None
    with _pipe_lock:
        release_pipelines(keep_kind="a2v_distilled")
        if _a2v_distilled_pipe is None or _a2v_distilled_model_dir != model_dir:
            if _a2v_distilled_pipe is not None:
                emit({"event": "log", "line": "model_dir changed; reloading A2V distilled pipeline."})
                _a2v_distilled_pipe = None
                _ac()
            emit({"event": "log",
                  "line": f"Loading A2V distilled pipeline (Q4 — {model_dir})..."})
            _a2v_distilled_pipe = A2VidDistilledPipeline(
                model_dir=model_dir,
                gemma_model_id=GEMMA_PATH,
                low_memory=LOW_MEMORY,
            )
            _a2v_distilled_model_dir = model_dir
        return _a2v_distilled_pipe


# ---- prompt enhancement (Gemma language model) ------------------------------
# Separate from the pipeline's TextEncoder wrapper. The LanguageModel class
# supports `.enhance_t2v(prompt, seed)` / `.enhance_i2v(prompt, seed)` for
# prompt rewriting, which requires the generative Gemma 3 root even when the
# active LTX generation renders with Gemma 4. Loaded lazily on first enhance
# request. Held warm across calls; freed by `release_pipelines` when a render
# starts to keep memory below the 64 GB ceiling.
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
            # LOAD, THEN PUBLISH. Assigning the instance before load()
            # succeeded poisoned the singleton: the FIRST failure reported
            # its real cause (missing dir, OOM), and every later call in the
            # warm process hit the retained half-built instance and reported
            # the useless "Model not loaded. Call load() first." — which is
            # what the fleet counts (14 events / 6 installs on 4.8.0). A
            # failed load now leaves the slot None, so the next attempt
            # re-tries the load and reports the real error again.
            _candidate = GemmaLanguageModel()
            _candidate.load(ENHANCE_GEMMA_PATH)
            _gemma_lm = _candidate
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


def _build_live_preview(kwargs: dict):
    """A LivePreviewMonitor for this job, or None when the preview is off.

    THE LANE RULES ARE THE PANEL'S, NOT OURS. `live_preview_every` arrives in
    the job spec because the right value differs per PIPELINE — 1 on the
    distilled lane, 2 on res_2s, which evaluates the denoiser twice per stage-2
    step and whose odd anchor estimates come back patchy — and a client counting
    estimates would have to know which schedule is running. We take the number
    and use it.

    Never fatal. A missing checkpoint, an unreadable directory or a library
    without the module means the render proceeds with no preview: this is a
    monitor, and a monitor must not be able to stop the thing it watches."""
    d = kwargs.get("live_preview_dir")
    tae = kwargs.get("live_preview_tae")
    if not d or not tae:
        return None
    try:
        from ltx_pipelines_mlx.live_preview import LivePreviewMonitor
    except Exception as exc:                          # noqa: BLE001
        emit({"event": "log",
              "line": f"live preview unavailable in this engine build ({exc}) — "
                      f"rendering without it."})
        return None
    try:
        return LivePreviewMonitor(
            Path(d), Path(tae),
            output=Path(kwargs["output_path"]),
            every=max(1, int(kwargs.get("live_preview_every") or 1)),
        )
    except Exception as exc:                          # noqa: BLE001
        emit({"event": "log",
              "line": f"live preview could not start ({exc}) — rendering without it."})
        return None


def _generate_latents(pipe, *, needs_image: bool, kwargs: dict):
    # On second+ runs the video/audio decoders (~2.5 GB combined) remain
    # resident from the previous job's decode phase.  During the
    # block-by-block DiT+LoRA materialization that follows (48 blocks ×
    # ~300 MB each), the combined Metal heap can stall allocation past the
    # 10-second GPU watchdog threshold.  Free decoders here so they don't
    # compete with the DiT load; they reload lazily when decode starts.
    if LOW_MEMORY:
        if hasattr(pipe, "video_decoder_block"):
            pipe.video_decoder_block.free()
        if hasattr(pipe, "audio_decoder_block"):
            pipe.audio_decoder_block.free()
        try:
            from ltx_core_mlx.utils.memory import aggressive_cleanup as _ac
            _ac()
        except Exception:
            pass

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
    # Env gate (2026-05-26, perf-experiment E1): when PHOSPHENE_T2V_TWO_STAGE=1
    # is set, route T2V Standard through pipe.generate_two_stage() instead of
    # the legacy single-stage pipe.generate(). The two-stage path renders at
    # half-resolution (640×352 for 1280×704), spatially 2× upsamples the
    # latent via the Q4 upsampler, then runs a 3-step Stage-2 refine. It's the
    # same recipe HQ + I2V Standard already use successfully — I2V Standard
    # exercises it daily because DistilledPipeline has no `generate_from_image`
    # method, so I2V already falls through to generate_two_stage below.
    #
    # Predicted T2V wall reduction: ~30-35% on Standard (M4 Max 7:40 → 5:15,
    # M4 Pro 10 min → 6:45). Math:
    #   Stage-1 at half res ≈ 0.25× tokens × 8 steps = 2 step-equivalents
    #   Stage-2 at full res ≈ 1.0× tokens × 3 steps = 3 step-equivalents
    #   Upsampler ≈ ~30s constant
    #   Total ≈ 5.0 step-equivalents + 30s vs 8.0 step-equivalents native.
    #
    # DEFAULT = single-stage, by Mr Bizarro's explicit call (2026-05-27).
    # The A/B was run (tasks #49/#51, the e1_ab/compound_matrix HTML reports):
    # two-stage is ~30-35% faster (M4 Max 7:40 → ~5:15) BUT visibly SOFTER —
    # fine detail is reconstructed during the 2x upsample + 3-step refine
    # rather than denoised directly at full res. Mr Bizarro eyeballed both and
    # PREFERS single-stage's sharpness, so single-stage stays the default.
    # (The 2026-05-31 deep review flagged single-stage as upstream-"OOD"; that
    # concern is theoretical — the measured A/B showed single-stage sharper,
    # not artefact-broken, on real prompts. Do NOT flip the default to
    # two-stage without Mr Bizarro re-deciding.) Two-stage remains available
    # opt-in for users who want the speed and accept the softness:
    #   PHOSPHENE_T2V_TWO_STAGE=1
    _t2v_two_stage = os.environ.get("PHOSPHENE_T2V_TWO_STAGE", "").strip().lower() in ("1", "true", "yes", "on")
    if not needs_image and not _t2v_two_stage and hasattr(pipe, "generate"):
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
            # Only when the installed pipeline actually takes it — the panel
            # must never die in argparse-equivalent 30 s into a render because
            # an older engine build does not have the parameter.
            if "live_preview" in sig.parameters and kwargs.get("_live_preview") is not None:
                call_kwargs["live_preview"] = kwargs["_live_preview"]
            return pipe.generate(**call_kwargs)
        except TypeError:
            # New DistilledPipeline.generate inherits from the two-stage
            # parent and doesn't accept frame_rate. Fall through to
            # generate_two_stage which absorbs everything via
            # **_unused_kwargs.
            pass
    if not needs_image and _t2v_two_stage:
        # Log once per render so the A/B harness can correlate wall time
        # in the panel's log tab without grepping env at boot.
        emit({"event": "log",
              "line": "[t2v-two-stage] PHOSPHENE_T2V_TWO_STAGE=1 — routing T2V Standard through generate_two_stage (half-res → 2× upsample → Stage-2 refine)."})
    # Unified new-API fallback (post-refactor packages).
    _two_stage_kwargs = dict(
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
    if kwargs.get("_live_preview") is not None:
        try:
            import inspect as _inspect
            if "live_preview" in _inspect.signature(pipe.generate_two_stage).parameters:
                _two_stage_kwargs["live_preview"] = kwargs["_live_preview"]
        except (TypeError, ValueError):
            pass
    return pipe.generate_two_stage(**_two_stage_kwargs)


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

    import ltx_pipelines_mlx._base as base_pipeline
    import ltx_pipelines_mlx.utils.constants as constants

    targets = [
        mod for mod in (base_pipeline, constants)
        if hasattr(mod, "DEFAULT_NEGATIVE_PROMPT")
    ]
    if not targets:
        yield False
        return

    previous = {mod: mod.DEFAULT_NEGATIVE_PROMPT for mod in targets}
    for mod, value in previous.items():
        mod.DEFAULT_NEGATIVE_PROMPT = f"{value}, {neg}"
    try:
        yield True
    finally:
        for mod, value in previous.items():
            mod.DEFAULT_NEGATIVE_PROMPT = value


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
        # 2026-05-27 — per-mode tail protection (restores Turbo's distinctiveness).
        #
        # History: the 2026-05-09 fix (commit 3e4bfd8) bumped protected_tail
        # from ceil(N/3) to ceil(N/2) to eliminate step-4 eye/skin artifacts
        # under Turbo (step 4's relative MAE ~0.0245 sits between Boost's
        # 0.02 threshold and Turbo's 0.03 — Boost protected it by chance,
        # Turbo cached it and produced visible artifacts).
        #
        # Side effect of that fix: on the 8-step distilled schedule, cacheable
        # slots dropped from 3 (steps 2,3,4) to 2 (steps 2,3). Boost's
        # max_skips=2 saturated both, and Turbo's max_skips=3 hit the same
        # cap — making Turbo functionally identical to Boost. Compounded
        # with the bug where the accel patch only hit 3 of 9 denoise_loop
        # import sites (silent no-op on T2V Standard since the upstream
        # rename around May 9), Turbo's whole reason to exist has been
        # invisible for two months.
        #
        # Fix: per-mode tail. Boost keeps the strict tail (safe fast).
        # Turbo gets the original loose tail — step 4 is cacheable again,
        # so on warm-light prompts you may see the eye/skin mesh. That IS
        # the intentional tradeoff of Turbo ("aggressive fast, may
        # artifact"); users picked it knowing the cost. Boost remains the
        # artifact-free fast mode.
        if steps:
            if mode_name == "turbo":
                protected_tail = min(len(steps), max(2, math.ceil(len(steps) / 3)))
            else:
                protected_tail = min(len(steps), max(2, math.ceil(len(steps) / 2)))
        else:
            protected_tail = 0

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

    History: 2026-05-27 fix — `denoise_loop` is imported by name into
    EIGHT separate pipeline modules in upstream ltx_pipelines_mlx, each
    captures its own module-level binding at import time. Patching only
    `samplers.denoise_loop` doesn't reach the active T2V path because
    `_base.py:BasePipeline.generate` calls the binding it captured. Same
    for HQ (`ti2vid_two_stages_hq`), FFLF (`keyframe_interpolation`),
    A2V (`a2vid_two_stage`), Extend (`retake` — though that uses
    `guided_denoise_loop`), IC-LoRA, lipdub, and TI2V two-stage. We
    enumerate every site and patch the binding directly. Without this
    the accel modes were silent no-ops on every pipeline EXCEPT the
    `distilled.py` two-stage path — which itself only fires when the
    new PHOSPHENE_T2V_TWO_STAGE env gate is on. Hence the historical
    300-380s Boost/Turbo numbers stopped reproducing after the
    upstream class rename (TextToVideoPipeline → DistilledPipeline)
    around 2026-05-09. Resolved by patching `_base.denoise_loop` +
    every other import site.
    """
    global _ORIGINAL_DENOISE_LOOP, _CURRENT_ACCEL_MODE, _LAST_ACCEL_STATS

    requested = (mode or "off").strip().lower()
    if requested not in {"off", "boost", "turbo"}:
        requested = "off"

    # Modules in upstream ltx_pipelines_mlx that do
    #   `from ...samplers import denoise_loop`
    # at module-load time. Patching the source module is necessary but
    # not sufficient because every caller already has its own binding.
    # Enumerate explicitly; importing also forces module load if not
    # already imported (so the next pipeline build can pick up the
    # patched binding before its own first call).
    import ltx_pipelines_mlx.utils.samplers as samplers
    import ltx_pipelines_mlx._base as _base_mod
    import ltx_pipelines_mlx.distilled as distilled_mod
    _patch_targets = [samplers, _base_mod, distilled_mod]
    for _name in (
        "ti2vid_two_stages",
        "ti2vid_two_stages_hq",
        "keyframe_interpolation",
        "a2vid_two_stage",
        "lipdub",
        "ic_lora",
    ):
        try:
            _mod = __import__(f"ltx_pipelines_mlx.{_name}", fromlist=[_name])
            _patch_targets.append(_mod)
        except Exception:
            # Module may not exist in older upstream — best-effort.
            pass

    if _ORIGINAL_DENOISE_LOOP is None:
        _ORIGINAL_DENOISE_LOOP = samplers.denoise_loop

    if requested == _CURRENT_ACCEL_MODE:
        if requested != "off":
            _LAST_ACCEL_STATS = None
        return requested

    _LAST_ACCEL_STATS = None
    if requested == "off":
        target = _ORIGINAL_DENOISE_LOOP
    elif requested == "boost":
        target = _build_adaptive_x0_loop("boost", max_skips=2, video_thresh=0.02, audio_thresh=0.02)
    else:
        target = _build_adaptive_x0_loop("turbo", max_skips=3, video_thresh=0.03, audio_thresh=0.03)

    _patched = []
    for _mod in _patch_targets:
        if hasattr(_mod, "denoise_loop"):
            setattr(_mod, "denoise_loop", target)
            _patched.append(_mod.__name__.split(".")[-1])

    _CURRENT_ACCEL_MODE = requested
    emit({"event": "log",
          "line": f"accel:mode {requested} patched={','.join(_patched)}"})
    return requested


# ---- ltx-2-mlx version gate (2026-05-31 review fix) --------------------------
# ROOT CAUSE of every recent fire (#5, #9, accel regression): the helper
# monkey-patches a MOVING upstream (ltx-2-mlx) at runtime, the v0.14.0 "pin"
# is structurally leaky, and nothing asserted what's actually installed. The
# vendored clone has been observed dirty-but-reporting-0.14.0 on this very
# machine. This gate reads the installed package metadata at boot, compares to
# the pin Phosphene's patches are written against, and makes ANY skew loud +
# visible in the ready event (so every remote bug report carries it) instead
# of letting it surface as an un-triageable TypeError mid-render.
#
# 2026-08-12: this is a FORK BUILD, not an upstream tag. The vendored checkout
# is mrbizarro/ltx-2-mlx at the immutable tag `v0.14.19+ltx25.7` — v0.14.19 plus
# the LTX-2.5 port (keyframe pos-emb, Gemma 4 tower, ancestral sampler). The
# release segment stays 0.14.19 because that is genuinely what it branches from;
# the `+ltx25.N` local segment is what makes the two distinguishable.
#
# The N moves whenever the fork changes what a render COMPUTES, not merely when
# the SHA moves. `.2` was the ancestral sampler actually being reached on 2.5
# plus the vendor's stage-2 first sigma (0.85, not 2.3's 0.909375). `.3` is the
# v4.0 pin: isolated-modality guidance off by default on the 2.5 HQ path, the
# distilled lane refining in 2 steps, and a step count thinning the checkpoint's
# table instead of truncating it. `.4` is the v4.0.2 pin: the euler-ancestral
# step re-composites the SAMPLE against the clean latent, not only the x0
# estimate, so an image-conditioned 2.5 render stops discarding its own anchor
# after step 1. All of them change output on 2.5; none touches 2.3 — `.4` is
# proven sha256-identical on t2v, and every Euler lane short-circuits the guard.
#
# The TAG is the pin, not the SHA. A branch tip moves and a SHA on a rebased
# branch stops being fetchable; a tag is the only form of this reference that
# cannot rot under an existing install. install.js / update.js fetch the tag.
#
# That local segment exists FOR this gate. The fork originally carried the bare
# string "0.14.19", so a 2.5 runtime reported `match: true` against the 2.3 pin
# — a skew gate blind to the one skew that mattered. Bumping the pin here
# without bumping the packages (or the reverse) puts it back into permanent
# SKEW warnings, so the two move together or not at all.
_LTX_EXPECTED_VERSION = "0.14.19+ltx25.7"


def _detect_ltx_version() -> dict:
    """Best-effort: what ltx-pipelines-mlx is actually importable right now.

    Returns {version, expected, match, note}. Never raises — a detection
    failure is itself reported, not swallowed.
    """
    info = {"version": None, "expected": _LTX_EXPECTED_VERSION, "match": None, "note": ""}
    # 1. Package metadata (authoritative for `pip install`-ed dist).
    try:
        import importlib.metadata as _md
        for _dist in ("ltx-pipelines-mlx", "ltx_pipelines_mlx", "ltx-core-mlx"):
            try:
                info["version"] = _md.version(_dist)
                break
            except Exception:
                continue
    except Exception as exc:
        info["note"] = f"metadata probe failed: {exc}"
    # 2. Module __version__ as a cross-check — ONLY if ltx_pipelines_mlx is
    #    already imported. We deliberately do NOT import it here: forcing the
    #    MLX import chain at boot would add startup latency and side-effects.
    #    The metadata probe above is authoritative for the installed version;
    #    this is a free bonus cross-check when the module happens to be loaded.
    try:
        _lpm = sys.modules.get("ltx_pipelines_mlx")
        mod_v = getattr(_lpm, "__version__", None) if _lpm else None
        if mod_v and info["version"] and mod_v != info["version"]:
            info["note"] = (info["note"] + f" module __version__={mod_v} "
                            f"disagrees with metadata={info['version']}").strip()
        if mod_v and not info["version"]:
            info["version"] = mod_v
    except Exception:
        pass
    info["match"] = (info["version"] == _LTX_EXPECTED_VERSION)
    return info


_LTX_VERSION_INFO = _detect_ltx_version()
if not _LTX_VERSION_INFO["match"]:
    emit({"event": "log",
          "line": (f"WARNING ltx-2-mlx VERSION SKEW: installed={_LTX_VERSION_INFO['version']} "
                   f"expected={_LTX_EXPECTED_VERSION} — runtime patches are written "
                   f"against {_LTX_EXPECTED_VERSION}; behavior on this version is "
                   f"unvalidated. {_LTX_VERSION_INFO['note']}".strip())})

# ---- runtime fingerprint (mlx / chip / OS) ----------------------------------
# Every render log then self-documents the exact environment — the data we keep
# having to ask for on "mosaic" / garbled-output reports. Those are an MLX
# numerical-correctness issue on specific chip+mlx combos (the render completes
# and writes a normal-size file full of garbage pixels — not a crash, OOM, or
# corrupt weights), so the mlx/mlx-metal version and the Apple chip are exactly
# what's needed to triangulate. Never raises.
def _detect_runtime_env() -> dict:
    env = {"mlx": None, "mlx_metal": None, "mlx_lm": None,
           "chip": None, "macos": None, "arch": None}
    try:
        import importlib.metadata as _md
        for _k, _dist in (("mlx", "mlx"), ("mlx_metal", "mlx-metal"), ("mlx_lm", "mlx-lm")):
            try:
                env[_k] = _md.version(_dist)
            except Exception:
                pass
    except Exception:
        pass
    try:
        import platform as _pf
        env["arch"] = _pf.machine()
        env["macos"] = _pf.mac_ver()[0] or None
    except Exception:
        pass
    try:
        import subprocess as _sp
        _chip = _sp.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                        capture_output=True, text=True, timeout=3).stdout.strip()
        env["chip"] = _chip or None
    except Exception:
        pass
    return env


_RUNTIME_ENV = _detect_runtime_env()
# ASCII only — this is emitted to the helper's stdout, which the panel reads to
# parse events; a non-ASCII byte there can break the read on a non-UTF-8 locale
# (the same failure mode the version-skew emoji had). No "·" / emoji here.
emit({"event": "log",
      "line": (f"runtime | mlx={_RUNTIME_ENV.get('mlx')} "
               f"mlx-metal={_RUNTIME_ENV.get('mlx_metal')} | "
               f"chip={_RUNTIME_ENV.get('chip')} | "
               f"macOS={_RUNTIME_ENV.get('macos')} ({_RUNTIME_ENV.get('arch')})")})

# ---- MLX allocator cache policy ---------------------------------------------
# MLX does not return freed Metal buffers to the driver; it keeps them in an
# allocator cache so the next same-size allocation is free. The CEILING on that
# cache is derived from the MACHINE, not the workload — MLX defaults it to
# `0.95 * hw.memsize`, about 61 GB on a 64 GB Mac. Phosphene set no cache policy
# at all, so a five-second clip's cache grew until it was near the machine's
# ceiling and the footprint the render reported had nothing to do with what a
# five-second clip needs. (`_base.py` and `a2vid_distilled.py` call
# `set_cache_limit(0)`, but only on the low-memory/streaming paths the warm
# helper does not take.)
#
# The cap is a FRACTION OF PHYSICAL RAM, so it scales with the Mac the way the
# capability tiers do: ONE EIGHTH, floored at 2 GiB, ceilinged at 8 GiB.
#
#     16 GiB Compact      -> 2 GiB        64 GiB Comfortable -> 8 GiB
#     32 GiB Compact      -> 4 GiB        96 GiB Roomy       -> 8 GiB (ceiling)
#     48 GiB Comfortable  -> 6 GiB       128 GiB Studio      -> 8 GiB (ceiling)
#
# MEASURED, not chosen — mlx 0.31.1 (this repo's pin), Q4 768x432 121f, same
# prompt and seed, `/usr/bin/time -l` peak memory footprint:
#
#     no cap (what shipped) ....... 25.53 GB / 71.45 s
#     16 GB cap ................... 23.15 GB / 71.78 s
#     8 GiB cap (THIS policy) ..... 16.50 GB / 68.84 s
#
# and Q8 1024x576 121f at the 48 GiB Comfortable tier's own ceiling, where the
# policy asks for 6 GiB: 39.49 GB / 139.84 s -> 25.39 GB / 138.71 s. So about a
# THIRD of the peak footprint goes away on both, and the wall time does not pay
# for it. Output is sha256-identical to the uncapped render in both cases
# (144da74a... and f47d4dc8...): the cap changes the allocator's bookkeeping,
# never the arithmetic.
#
# The CEILING is measured too, not taste: on mlx 0.32.1, 8 GB -> 16 GB of cache
# bought 0.5 s and cost 7.7 GB of footprint. So is the FLOOR: a cache of zero
# gives back the most memory of all but pays ~1% wall, which makes it the
# override, not the default.
#
# Override: `LTX_MLX_CACHE_GIB` — a number in GiB, `0` for no cache at all,
# `off` to restore MLX's machine-sized default (the pre-policy behaviour).
# Unset / empty / `auto` runs the policy. An unparseable value runs the policy
# rather than failing a render over a typo in an env var.
MLX_CACHE_RAM_DIVISOR = 8
MLX_CACHE_FLOOR_BYTES = 2 * 1024**3
MLX_CACHE_CEIL_BYTES = 8 * 1024**3


def mlx_cache_limit_bytes(total_ram_bytes: int, override: str | None = None) -> int | None:
    """Bytes MLX may hold in its allocator cache. `None` = leave MLX's default.

    Pure on purpose: the policy for a 16 GB Mac has to be assertable from a
    64 GB one, and the tier table this was measured against is a simulation of
    exactly this number.
    """
    raw = (override or "").strip().lower()
    if raw in ("off", "default", "unset"):
        return None
    if raw and raw != "auto":
        try:
            gib = float(raw)
        except ValueError:
            gib = -1.0
        if gib >= 0:
            return int(gib * 1024**3)
        # Unparseable: fall through to the policy.
    if total_ram_bytes <= 0:
        # sysctl failed. Take the smallest cap rather than the machine's.
        return MLX_CACHE_FLOOR_BYTES
    return max(MLX_CACHE_FLOOR_BYTES,
               min(MLX_CACHE_CEIL_BYTES, int(total_ram_bytes) // MLX_CACHE_RAM_DIVISOR))


def _physical_ram_bytes() -> int:
    try:
        import subprocess as _sp
        out = _sp.run(["sysctl", "-n", "hw.memsize"],
                      capture_output=True, text=True, timeout=3).stdout.strip()
        return int(out)
    except Exception:
        return 0


_MLX_CACHE_RAM_BYTES = _physical_ram_bytes()
_MLX_CACHE_LIMIT = mlx_cache_limit_bytes(
    _MLX_CACHE_RAM_BYTES, os.environ.get("LTX_MLX_CACHE_GIB"))
_MLX_CACHE_ANNOUNCED = False


def apply_mlx_cache_policy() -> None:
    """Set — or RE-assert — the allocator cache cap. Cheap and idempotent.

    Re-asserted before every render, not only at startup, because the pipelines
    move it themselves: the low-memory and streaming paths call
    `mx.set_cache_limit(0)` and never put it back, so a policy applied once at
    boot would silently become "no cache" for the rest of a warm helper's life
    after a single A2V job.
    """
    global _MLX_CACHE_ANNOUNCED
    if _MLX_CACHE_LIMIT is None:
        return
    try:
        import mlx.core as mx
        if LOW_RAM_STREAM:
            # Streaming only pays if freed block buffers actually leave the
            # heap; a retained cache re-pins them. Keep the pipeline's own
            # set_cache_limit(0) instead of re-asserting the fractional cap.
            mx.set_cache_limit(0)
            if not _MLX_CACHE_ANNOUNCED:
                _MLX_CACHE_ANNOUNCED = True
                emit({"event": "log",
                      "line": "[mlx] low-RAM streaming: transformer blocks are read from disk as needed and the Metal cache is off — slower per step, far smaller peak."})
            return
        mx.set_cache_limit(int(_MLX_CACHE_LIMIT))
    except Exception as exc:                          # noqa: BLE001
        if not _MLX_CACHE_ANNOUNCED:
            _MLX_CACHE_ANNOUNCED = True
            emit({"event": "log",
                  "line": f"[mlx] cache policy not applied: {exc}"})
        return
    if not _MLX_CACHE_ANNOUNCED:
        _MLX_CACHE_ANNOUNCED = True
        _ram_gib = _MLX_CACHE_RAM_BYTES / 1024**3
        _default_gib = 0.95 * _ram_gib
        emit({"event": "log",
              "line": (f"[mlx] allocator cache capped at "
                       f"{_MLX_CACHE_LIMIT / 1024**3:.1f} GiB "
                       f"(1/{MLX_CACHE_RAM_DIVISOR} of {_ram_gib:.0f} GiB RAM; "
                       f"MLX would default to {_default_gib:.1f} GiB here)")})


apply_mlx_cache_policy()


def _live_preview_supported() -> bool:
    """Does the INSTALLED engine carry the live-preview module?

    `find_spec` on a submodule imports the PARENT package, which can raise
    anything at all on a half-installed or mismatched engine — and this probe
    runs on the ready path, where an exception would take the whole helper down
    over a monitor. Any failure means "cannot preview", which is the answer the
    user needs anyway."""
    try:
        return importlib.util.find_spec("ltx_pipelines_mlx.live_preview") is not None
    except Exception:                                 # noqa: BLE001
        return False


# ---- ready -------------------------------------------------------------------
emit({
    "event": "ready",
    "model": MODEL_ID,
    "gemma": GEMMA_PATH,
    "low_memory": LOW_MEMORY,
    "idle_timeout_sec": IDLE_TIMEOUT,
    "ltx_version": _LTX_VERSION_INFO["version"],
    "ltx_version_expected": _LTX_EXPECTED_VERSION,
    "ltx_version_match": _LTX_VERSION_INFO["match"],
    # CAN THIS ENGINE PREVIEW AT ALL? Probed as a CAPABILITY (does the module
    # import?), never inferred from the version string — a re-pin, a fork or a
    # hand-built venv must be able to answer this truthfully without anyone
    # updating a version table.
    #
    # This exists because the answer was previously discoverable only at render
    # time, in a log line nobody reads: an install whose vendored engine lagged
    # behind the panel simply never created `state/live`, while the panel went
    # on reporting live preview as ON. Zero frames, zero errors, zero
    # explanation. The panel can only tell the user the truth if the helper
    # tells the panel.
    "live_preview_supported": (
        importlib.util.find_spec("ltx_pipelines_mlx.live_preview") is not None),
    # CAN THIS ENGINE ENCODE FOR LTX-2.5 AT ALL? Same capability-probe shape as
    # live_preview_supported, for the same reason: an install whose vendored
    # engine predates the Gemma 4 tower (site-packages older than
    # v0.14.19+ltx25.3 — the 3.8→4.0 stale-updater path) routes 2.5 text
    # encoding into mlx_lm, which has no gemma4 module and dies with mlx_lm's
    # own words: "Model type gemma4_unified not supported." (19 fleet events /
    # 6 installs, all legacy). The fault self-heals on a SECOND Update click,
    # but nothing ever told the user that — the panel can only say it if the
    # helper reports the capability.
    "gemma4_tower_supported": (
        importlib.util.find_spec("ltx_core_mlx.text_encoders.gemma.gemma4")
        is not None),
    "low_ram_stream": bool(LOW_RAM_STREAM),
    "mlx_version": _RUNTIME_ENV.get("mlx"),
    "mlx_metal_version": _RUNTIME_ENV.get("mlx_metal"),
    # None when the policy is off (LTX_MLX_CACHE_GIB=off) — i.e. MLX's own
    # machine-sized default, which is what every install ran before v4.7.
    "mlx_cache_limit_gib": (None if _MLX_CACHE_LIMIT is None
                            else round(_MLX_CACHE_LIMIT / 1024**3, 3)),
    "chip": _RUNTIME_ENV.get("chip"),
    "macos": _RUNTIME_ENV.get("macos"),
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

    # ONE CHOKE POINT for schedule step counts, before any action reads them.
    # Every generate_* lane resolves its own stage1/stage2 defaults, and any of
    # them can be handed an explicit value by the form, by Load Params, or by a
    # sidecar replayed from another generation. Clamping here — once, against
    # the pack this job actually names — means no lane can ask a checkpoint to
    # pad a table it does not have, and a lane added tomorrow is covered
    # without anyone remembering to add it.
    if action == "exit":
        emit({"event": "exit", "reason": "shutdown"})
        os._exit(0)

    if action == "ping":
        emit({"event": "pong"})
        continue

    # THE OTHER choke point, same reason as the one above: the MLX allocator
    # cache cap is re-asserted here rather than in each generate_* lane,
    # because the pipelines drop it to zero on their low-memory paths and a
    # lane added tomorrow would otherwise inherit whatever the last job left.
    if isinstance(action, str) and action.startswith(("generate", "extend")):
        apply_mlx_cache_policy()

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
            _mlx_mem_reset_run()
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
            # The panel names the pack on the job spec now; None keeps the
            # pre-2026-08 behaviour (this helper's spawn-time LTX_MODEL).
            pipe = get_pipe("i2v" if needs_image else "t2v", loras=loras,
                            model_dir=p.get("model_dir"))
            emit({"event": "log", "line": "step:get_pipe done"})
            _log_memory_pressure()
            # NOT "weights": measured on a live 4.7.0 render, this reads
            # 0.00 GiB because `get_pipe` is lazy — the pack is faulted in
            # during generate, so its residency lands in the denoise figure.
            # Labelling it "weights" would have every reader conclude the
            # model is free.
            _mlx_phase_mem("pipeline init")
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
            # Named schedule preset for the 2.5 distilled lane ("fast" = the
            # graded 5+2 draft schedule — a DIFFERENT take, priced in the UI).
            # The panel gates it to the lane that defines it; the engine's
            # resolver validates it before stage 1 spends GPU. On an older
            # installed library _filter_unsupported_kwargs drops it with a
            # log line, so carrying the key is forward-safe by design.
            if p.get("schedule_preset"):
                kwargs["schedule_preset"] = str(p["schedule_preset"])
            # Inspire (loose reference): forwarded only when TRUE, so an
            # older installed library never sees a key it cannot take — and
            # _filter_unsupported_kwargs drops it with a log line even then.
            if p.get("loose_reference"):
                kwargs["loose_reference"] = True
            # Y1.037: short-clip VAE-streaming opt-out. Set the env var BEFORE
            # generate() so it propagates through the whole chain (the patched
            # decode_and_stream reads os.environ at decode call time).
            _apply_vae_streaming_decision(kwargs["num_frames"])
            # LIVE PREVIEW. Read-only: the thumbnail is a decode of the x0
            # estimate the denoise loop ALREADY HOLDS, so nothing is
            # recomputed, no scheduler is re-entered and no tensor the
            # denoiser owns is written. A render with it on is byte-identical
            # to the same render with it off — proven by hash, not asserted.
            for _k in ("live_preview_dir", "live_preview_tae", "live_preview_every"):
                if p.get(_k) is not None:
                    kwargs[_k] = p[_k]
            kwargs["_live_preview"] = _build_live_preview(kwargs)
            if kwargs["_live_preview"] is not None:
                emit({"event": "log",
                      "line": f"live preview on — every {kwargs.get('live_preview_every', 1)} "
                              f"estimate(s) → {p.get('live_preview_dir')}"})
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

            _aggressive_cleanup_before_generate()
            if use_model_upscale:
                emit({"event": "log", "line": f"step:generate mode={mode} {kwargs['width']}x{kwargs['height']} {kwargs['num_frames']}f @{kwargs['frame_rate']:.1f}fps steps={kwargs['num_steps']} accel={accel_mode} upscale=model"})
                # Step 1: generate latents (no save)
                video_latent, audio_latent = _generate_latents(pipe, needs_image=needs_image, kwargs=kwargs)
                emit({"event": "log", "line": "step:generate done"})
                _mlx_phase_mem("denoise")
                # Free DiT + text encoder before the upscale + VAE decode peak.
                emit({"event": "log", "line": "step:free_generation_modules start"})
                _free_pipe_for_decode(pipe)
                emit({"event": "log", "line": "step:free_generation_modules done"})
                # Step 2: latent x2 upscale
                emit({"event": "log", "line": "step:latent_upscale_x2 start"})
                video_latent = _model_upscale_video_latent(pipe, video_latent)
                emit({"event": "log", "line": f"step:latent_upscale_x2 done — latent {video_latent.shape[-2]}×{video_latent.shape[-1]}"})
                _mlx_phase_mem("latent upscale")
                # Free the upscaler before VAE decode (can be ~2-3 GB peak).
                _free_upscaler()
                # Step 3: VAE decode + save (decoder loads inside _decode_and_save_video).
                # FIX 2026-05-14: upstream renamed fps= → frame_rate= (keyword-only).
                out_path = pipe._decode_and_save_video(video_latent, audio_latent, kwargs["output_path"], frame_rate=kwargs["frame_rate"])
                # NOTE: lazy-graph cleanup happens AFTER emit(done_event)
                # below — see comment there. del-ing here would stall the
                # panel's wait_done while MLX tears down ~10 GB of Metal
                # buffers + lazy graph nodes synchronously.
                emit({"event": "log", "line": "step:decode_and_save done"})
                _mlx_phase_mem("decode+save")
                _mlx_mem_summary()
            _aggressive_cleanup_before_generate()
            if not use_model_upscale:
                emit({"event": "log", "line": f"step:generate mode={mode} {kwargs['width']}x{kwargs['height']} {kwargs['num_frames']}f @{kwargs['frame_rate']:.1f}fps steps={kwargs['num_steps']} accel={accel_mode}"})
                video_latent, audio_latent = _generate_latents(pipe, needs_image=needs_image, kwargs=kwargs)
                emit({"event": "log", "line": "step:generate done"})
                _mlx_phase_mem("denoise")
                emit({"event": "log", "line": "step:free_generation_modules start"})
                _free_pipe_for_decode(pipe)
                emit({"event": "log", "line": "step:free_generation_modules done"})
                emit({"event": "log", "line": "step:decode_and_save start"})
                # Post-decode hang: on DistilledPipeline.generate_two_stage
                # (T2V/I2V Balanced) the function-return path stalls 5-15 min
                # in MLX/Metal deallocator chains that hold every Python
                # thread's GIL access. An in-process daemon-thread watchdog
                # CAN'T fire (Metal holds something the watchdog needs to
                # advance). Rescue is done in the PANEL — `WarmHelper.run`
                # detects the decode-done log line, waits a grace period,
                # and SIGKILLs the helper if no done event arrives. The
                # output file is intact on disk by then so the panel
                # synthesizes a done event from the known output_path.
                # See `WarmHelper._build_post_decode_panic` in mlx_ltx_panel.py.
                # FIX 2026-05-14: upstream renamed fps= → frame_rate= (keyword-only).
                out_path = pipe._decode_and_save_video(video_latent, audio_latent, kwargs["output_path"], frame_rate=kwargs["frame_rate"])
                emit({"event": "log", "line": "step:decode_and_save done"})
                _mlx_phase_mem("decode+save")
                _mlx_mem_summary()
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
            # EMIT DONE FIRST — the user's render is complete and the file is
            # on disk. Cleanup below can take 10+ minutes on a 121-frame
            # Balanced render because MLX's deallocator has to walk the
            # entire lazy compute graph and tear down Metal buffers; doing
            # that BEFORE the done event made the panel sit on "running"
            # for the full cleanup window (Mr Bizarro caught this 2026-05-21).
            emit(done_event)
            # Now drop the latent refs. The panel has already moved on; the
            # helper just needs to be clean by the time the next job arrives.
            try:
                del video_latent, audio_latent
            except UnboundLocalError:
                pass
        except Exception as exc:
            _last_activity = time.time()
            # A USER STOP IS NOT A FAILURE. The abort sentinel raises
            # LivePreviewAborted between forwards, and the CLI turns that into
            # exit 75 precisely so a supervisor can tell "the user stopped this"
            # apart from "this crashed" without parsing anything. The panel is
            # that supervisor, and it gets a distinct event rather than a red
            # card: nothing was saved, but nothing went wrong either.
            if type(exc).__name__ == "LivePreviewAborted":
                # DROP THE WARM PIPELINE. The abort raises from BETWEEN two
                # forwards, so the pipeline is left mid-generate: latents,
                # scheduler position and the monitor's own state all belong to
                # a render that will never finish. The helper's whole point is
                # that it keeps that object alive for the next job — so the
                # next job inherited the wreckage and died instantly with an
                # empty error, and every job after it did too. Stop early
                # bricked the render lane until the panel was restarted.
                #
                # Found by rendering AFTER a stop rather than stopping and
                # walking away, which is the thing a user does and a curl test
                # does not. The next job pays one pipeline reload (~7 s) and
                # everything downstream is clean.
                try:
                    with _pipe_lock:
                        release_pipelines(None)
                except Exception:
                    pass
                emit({"event": "stopped", "id": job_id, "reason": str(exc),
                      "exit_code": 75})
            else:
                # DROP THE WARM PIPELINE ON REAL FAILURES TOO. The abort
                # branch above learned this lesson first: an exception can
                # leave the pipeline (or a half-loaded encoder) mid-state,
                # and the helper's job is to keep such objects alive for the
                # NEXT job — which then inherits the wreckage and fails with
                # a misleading second-order error ("Model not loaded. Call
                # load() first.", 14 fleet events). One pipeline reload
                # (~7 s) is the price of a clean slate after any failure.
                try:
                    with _pipe_lock:
                        release_pipelines(None)
                except Exception:
                    pass
                emit({"event": "error", "id": job_id, "error": str(exc),
                      "trace": traceback.format_exc()})
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
            _mlx_mem_reset_run()
            configure_acceleration("off")
            # Extend supports LoRAs via the same _pending_loras hook;
            # the dev transformer picks them up at load time just like T2V/I2V.
            loras = p.get("loras") or []
            # Y1.036 — Extend needs the Q8 `transformer-dev` weights. Panel
            # passes the resolved Q8 path via params.model_dir; falling back
            # to the helper's MODEL_ID is the legacy behavior.
            ext_model_dir = p.get("model_dir")
            pipe = get_pipe("extend", loras=loras, model_dir=ext_model_dir,
                            dev_transformer=p.get("dev_transformer"))
            video_path = p["video_path"]
            if not os.path.exists(video_path):
                raise RuntimeError(f"source video not found: {video_path}")
            # cfg_scale defaults to 1.0 (no classifier-free guidance) on 64 GB
            # Macs: CFG runs both conditional + unconditional through the dev
            # transformer, doubling activation memory and pushing 1280×704
            # extends into swap (240s/step instead of ~25s/step). The panel
            # exposes a "Fast" / "Quality" toggle that flips this to 3.0.
            cfg_scale = float(p.get("cfg_scale", 1.0))
            num_steps = int(p.get("steps", 8))

            # TeaCache for Extend (2026-05-18). Active only if the boot-time
            # patch landed AND the job spec doesn't explicitly opt out. The
            # monkey-patched guided_denoise_loop reads _EXTEND_TC_CONFIG to
            # decide whether to construct a controller. Threshold bumped
            # from 0.5 → 0.7 on 2026-05-21 — at the lower step count (8
            # default) more aggressive block-skip pays off more, and at
            # cfg=1.0 single-branch denoising the quality cost is minimal.
            # Job spec can still override via `teacache_thresh`.
            enable_tc = bool(p.get("enable_teacache", True)) and _EXTEND_TC_PATCH_OK
            tc_thresh = p.get("teacache_thresh")
            if tc_thresh is not None:
                try:
                    tc_thresh = float(tc_thresh)
                except (TypeError, ValueError):
                    tc_thresh = None
            if tc_thresh is None:
                tc_thresh = 0.7  # was upstream default 0.5
            _EXTEND_TC_CONFIG = {
                "enable": enable_tc,
                "thresh": tc_thresh,
                "num_steps": num_steps,
            }
            if enable_tc:
                emit({"event": "log",
                      "line": f"TeaCache active on extend (thresh={tc_thresh})."})
            with _override_default_negative_prompt(p.get("negative_prompt")) as neg_active:
                if neg_active:
                    emit({"event": "log", "line": "Avoid terms active via native CFG negative prompt."})
                video_lat, audio_lat = pipe.extend_from_video(
                    prompt=p["prompt"],
                    video_path=video_path,
                    extend_frames=int(p.get("extend_frames", 5)),
                    direction=p.get("direction", "after"),
                    seed=seed,
                    num_steps=num_steps,
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
            # Post-decode hang: Extend hits the same MLX/Metal deallocator
            # freeze as I2V Balanced. Rescue is panel-side in
            # `WarmHelper._build_post_decode_panic` — an in-process daemon
            # thread can't fire because Metal holds GIL access during the
            # deallocator chain. See the same comment in the `generate`
            # action above.
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
            # Always clear the per-call TeaCache config so a future
            # non-extend call (or extend with enable=False) doesn't
            # inherit a stale state.
            _EXTEND_TC_CONFIG = None
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
            _mlx_mem_reset_run()
            configure_acceleration("off")
            # LoRAs flow through the same wire shape as t2v/i2v. HQ is the
            # only path where dev-base character LoRAs actually transfer
            # cleanly (distilled inference + dev-trained LoRA = base-fine-tune
            # mismatch).
            hq_loras = p.get("loras") or []
            if hq_loras:
                emit({"event": "log",
                      "line": f"step:get_pipe kind=hq loras={len(hq_loras)}"})
            pipe = get_hq_pipe(model_dir, loras=hq_loras,
                               dev_transformer=p.get("dev_transformer"),
                               distilled_lora=p.get("distilled_lora"))
            # Y1.037: short-clip VAE-streaming opt-out (HQ T2V/I2V path).
            _apply_vae_streaming_decision(int(p["frames"]))
            lp_kwargs: dict = {}
            # LIVE PREVIEW ON THE HQ LANE TOO — it was only ever built in the
            # `generate` branch. High advertises preview_every=2 in the quality
            # table and published nothing, so the UI reported a MISSING DECODER
            # on a lane that had simply never been wired. Every-2 is the lane
            # rule, decided server-side: res_2s evaluates the denoiser twice per
            # stage-2 step and its odd ANCHOR estimates come back patchy, so the
            # even SUBSTEP ones are the clean series.
            # TI2VidTwoStagesHQPipeline.generate_and_save takes `live_preview`
            # directly; _filter_unsupported_kwargs below still guards a stock
            # upstream that does not.
            for _k in ("live_preview_dir", "live_preview_tae", "live_preview_every"):
                if p.get(_k) is not None:
                    lp_kwargs[_k] = p[_k]
            # output_path too: LivePreviewMonitor takes it, so leaving it out
            # made _build_live_preview raise KeyError into its own "never fatal"
            # except and return None — the preview silently not happening in the
            # one place whose entire job is to stop things silently not
            # happening. The `generate` branch passes the whole kwargs dict,
            # which is why it never hit this.
            if lp_kwargs:
                lp_kwargs["output_path"] = p["output_path"]
            _hq_preview = _build_live_preview(lp_kwargs) if lp_kwargs else None
            if _hq_preview is not None:
                emit({"event": "log",
                      "line": f"live preview on (HQ) — every "
                              f"{lp_kwargs.get('live_preview_every', 2)} estimate(s) "
                              f"→ {p.get('live_preview_dir')}"})
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
                # STG (Spatio-Temporal Guidance) — "detail guidance" slider.
                # Default 0.0 = OFF. The pipeline forces stg_blocks=[] when no
                # explicit guider params are passed, so stg_scale alone is a
                # no-op (an extra forward pass per step that's then discarded).
                # When stg_scale>0 we ALSO pass explicit video/audio guider
                # params with stg_blocks=[28] just below, which is what makes
                # the knob actually do something on the HQ path.
                stg_scale=float(p.get("stg_scale", 0.0)),
                enable_teacache=bool(p.get("enable_teacache", True)),
                teacache_thresh=float(p.get("teacache_thresh", 1.0)),
                # Bongmath inner-loop cap (HQ res_2s sampler). Default 100
                # matches upstream. Source: samplers.py:537 has a fixed
                # `for _ in range(bongmath_max_iter)` with no early exit, so
                # the cap IS the iteration count (not just a safety bound).
                # Each iter is pure latent algebra — no model forwards.
                bongmath_max_iter=int(p.get("bongmath_max_iter", 100)),
                # Inspire on the HQ lane: skip the masked-sample re-pin so
                # the reference guides subject/style and the composition
                # re-imagines itself. False (default) anchors the image for
                # real on 2.5 — the fix for the owner-reported "High ignores
                # my reference". Filtered out on an older library.
                loose_reference=bool(p.get("loose_reference", False)),
                # video_skip_step / audio_skip_step removed (v4.0.5): the
                # pinned generate_and_save accepts neither kwarg, so
                # _filter_unsupported_kwargs discarded them every render —
                # the comment above this line claimed a knob that never
                # reached the sampler. Wire MultiModalGuiderParams.skip_step
                # explicitly if the knob is ever wanted for real.
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
            # STG engage: stg_scale>0 only bites if stg_blocks is non-empty.
            # The HQ pipeline forces stg_blocks=[] in its own default guider
            # params, so we hand it explicit params with stg_blocks=[28] (the
            # block the one/two-stage pipelines perturb). Mirror the pipeline's
            # other defaults verbatim (cfg/rescale/modality) so STG is the only
            # thing that changes. At stg_scale<=0 we pass nothing → the pipeline
            # builds its own [] params → byte-identical to the pre-slider path.
            #
            # `modality_scale` is the one field that is NOT a constant: it is
            # generation-keyed, and `_hq_modality_scale` asks the library for
            # the same answer the pipeline's own default would compute. 2.3
            # still gets 3.0; 2.5 gets 1.0, so the slider changes STG and
            # nothing else. Hardcoding 3.0 here made the slider silently
            # re-enable the isolated-modality pass on 2.5 — the one the v4.0
            # pin removed by default.
            _stg = float(kwargs.get("stg_scale", 0.0))
            if _stg > 0.0:
                try:
                    from ltx_core_mlx.components.guiders import MultiModalGuiderParams
                    _modality = _hq_modality_scale(model_dir)
                    kwargs["video_guider_params"] = MultiModalGuiderParams(
                        cfg_scale=float(kwargs.get("cfg_scale", 3.0)),
                        stg_scale=_stg,
                        rescale_scale=0.45,
                        modality_scale=_modality,
                        stg_blocks=[28],
                    )
                    kwargs["audio_guider_params"] = MultiModalGuiderParams(
                        cfg_scale=7.0,
                        stg_scale=_stg,
                        rescale_scale=1.0,
                        modality_scale=_modality,
                        stg_blocks=[28],
                    )
                    emit({"event": "log", "line": f"STG detail guidance ON — stg_scale={_stg:g}, stg_blocks=[28], modality_scale={_modality:g}"})
                except Exception as _stg_exc:
                    # Never let STG wiring block a render — fall back to the
                    # plain (STG-off) HQ path if the import/shape ever drifts.
                    emit({"event": "log", "line": f"STG setup skipped ({_stg_exc}); rendering without STG."})
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
                if _hq_preview is not None:
                    kwargs["live_preview"] = _hq_preview
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
            _mlx_mem_reset_run()
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

            pipe = get_kf_pipe(model_dir,
                               dev_transformer=p.get("dev_transformer"),
                               distilled_lora=p.get("distilled_lora"))
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

    if action == "generate_a2v":
        # Audio-to-Video. Drives generation from an input audio waveform
        # plus prompt. Optional `image` conditions frame 0 (acts as I2V on
        # top of audio-driven generation). Pipeline writes the original
        # input audio onto the output mp4 — no panel-side mux needed.
        job_id = msg.get("id", "?")
        p = msg.get("params", {}) or {}
        model_dir = p.get("model_dir") or MODEL_ID
        seed = int(p.get("seed", -1))
        if seed == -1:
            seed = random.randint(0, 2**31 - 1)
        _is_busy = True
        try:
            t0 = time.time()
            _mlx_mem_reset_run()
            configure_acceleration("off")
            audio_path = p.get("audio_path") or ""
            if not audio_path:
                raise RuntimeError("audio_path is required for A2V")
            if not os.path.exists(audio_path):
                raise RuntimeError(f"audio file not found: {audio_path}")
            num_frames = int(p["frames"])
            loras = p.get("loras") or []
            pipe = get_a2v_pipe(model_dir, loras=loras,
                                dev_transformer=p.get("dev_transformer"),
                                distilled_lora=p.get("distilled_lora"))
            _apply_vae_streaming_decision(num_frames)
            kwargs = dict(
                prompt=p["prompt"],
                output_path=p["output_path"],
                audio_path=audio_path,
                height=int(p["height"]),
                width=int(p["width"]),
                num_frames=num_frames,
                frame_rate=float(p.get("frame_rate", 24.0)),
                seed=seed,
                stage1_steps=int(p.get("stage1_steps", 20)),
                stage2_steps=int(p.get("stage2_steps", 3)),
                cfg_scale=float(p.get("cfg_scale", 3.0)),
                stg_scale=float(p.get("stg_scale", 1.0)),
                audio_start_time=float(p.get("audio_start_time", 0.0)),
            )
            # Route the "Audio conditioning strength" slider to the knob that
            # exists. It was sent for years as an `audio_conditioning_scale`
            # kwarg that `generate_and_save` does not accept, so the signature
            # filter dropped it on EVERY render. The real control is
            # `modality_scale` on the stage-1 video guider, hardcoded to 3.0.
            _A2V_STATE["modality_scale"] = _a2v_modality_scale_value(
                p.get("audio_conditioning_scale"))
            if _A2V_STATE["modality_scale"] is not None:
                ok = _install_a2v_modality_patch()
                emit({"event": "log",
                      "line": f"[a2v] audio adhesion modality_scale="
                              f"{_A2V_STATE['modality_scale']:.2f} (patched={ok})"})
            # Optional reference image — when present, conditions frame 0
            # so the audio-driven generation opens on the user's still.
            ref_image = p.get("image") or None
            if ref_image:
                if not os.path.exists(ref_image):
                    raise RuntimeError(f"reference image not found: {ref_image}")
                kwargs["image"] = ref_image
            # audio_max_duration defaults inside the pipeline to
            # num_frames / frame_rate when omitted. The form may pin a
            # tighter clamp (e.g. user dragged a 30 s mp3 but only wants 5 s).
            amd = p.get("audio_max_duration")
            if amd is not None:
                try:
                    kwargs["audio_max_duration"] = float(amd)
                except (TypeError, ValueError):
                    pass
            # TeaCache for A2V (2026-05-18 PM). The Stage-1 dev-Euler loop
            # is exactly what LTX2_TEACACHE_COEFFICIENTS was calibrated for —
            # same model, same scheduler, same sigma layout. Default ON;
            # opt-out via job spec `enable_teacache=false`. The wrapper at
            # the top of this file reads _A2V_TC_CONFIG and constructs a
            # controller; we clear it in `finally` so a future non-a2v
            # call doesn't inherit a stale state.
            enable_tc = bool(p.get("enable_teacache", True)) and _A2V_TC_PATCH_OK
            tc_thresh = p.get("teacache_thresh")
            if tc_thresh is not None:
                try:
                    tc_thresh = float(tc_thresh)
                except (TypeError, ValueError):
                    tc_thresh = None
            _A2V_TC_CONFIG = {
                "enable": enable_tc,
                "thresh": tc_thresh,
                "num_steps": kwargs["stage1_steps"],
            }
            if enable_tc:
                emit({"event": "log",
                      "line": f"TeaCache active on A2V Stage 1 (thresh={tc_thresh if tc_thresh is not None else 'default 0.5'})."})
            emit({
                "event": "log",
                "line": (
                    f"step:generate_a2v {kwargs['width']}x{kwargs['height']} "
                    f"{kwargs['num_frames']}f @{kwargs['frame_rate']:.1f}fps "
                    f"stage1={kwargs['stage1_steps']} stage2={kwargs['stage2_steps']} "
                    f"cfg={kwargs['cfg_scale']} audio={os.path.basename(audio_path)}"
                    f"{' image=' + os.path.basename(ref_image) if ref_image else ''}"
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
            # Clear so the next non-A2V action doesn't inherit a stale gate.
            _A2V_TC_CONFIG = None
            _is_busy = False
        continue

    if action == "generate_a2v_distilled":
        # Audio-to-Video via the Q4 distilled pipeline (no dev transformer,
        # no CFG, 8+3 steps). Fits 24 GB systems where Q8 dev doesn't.
        job_id = msg.get("id", "?")
        p = msg.get("params", {}) or {}
        model_dir = p.get("model_dir") or MODEL_ID
        seed = int(p.get("seed", -1))
        if seed == -1:
            seed = random.randint(0, 2**31 - 1)
        _is_busy = True
        try:
            t0 = time.time()
            _mlx_mem_reset_run()
            configure_acceleration("off")
            audio_path = p.get("audio_path") or ""
            if not audio_path:
                raise RuntimeError("audio_path is required for A2V")
            if not os.path.exists(audio_path):
                raise RuntimeError(f"audio file not found: {audio_path}")
            num_frames = int(p["frames"])
            pipe = get_a2v_distilled_pipe(model_dir)
            _apply_vae_streaming_decision(num_frames)
            kwargs = dict(
                prompt=p["prompt"],
                output_path=p["output_path"],
                audio_path=audio_path,
                height=int(p["height"]),
                width=int(p["width"]),
                num_frames=num_frames,
                frame_rate=float(p.get("frame_rate", 24.0)),
                seed=seed,
                stage1_steps=int(p.get("stage1_steps", 8)),
                stage2_steps=int(p.get("stage2_steps", 3)),
                audio_start_time=float(p.get("audio_start_time", 0.0)),
                # NATIVE here, unlike the Q8 path: A2VidDistilledPipeline
                # accepts this kwarg and amplifies the audio tokens with it
                # before the DiT. Its denoise is plain `denoise_loop` — no
                # guiders — so the modality_scale patch has nothing to drive
                # on this path and must not claim otherwise in the log.
                audio_conditioning_scale=float(
                    p.get("audio_conditioning_scale") or 1.0),
            )
            ref_image = p.get("image") or None
            if ref_image:
                if not os.path.exists(ref_image):
                    raise RuntimeError(f"reference image not found: {ref_image}")
                kwargs["image"] = ref_image
            amd = p.get("audio_max_duration")
            if amd is not None:
                try:
                    kwargs["audio_max_duration"] = float(amd)
                except (TypeError, ValueError):
                    pass
            emit({
                "event": "log",
                "line": (
                    f"step:generate_a2v_distilled {kwargs['width']}x{kwargs['height']} "
                    f"{kwargs['num_frames']}f @{kwargs['frame_rate']:.1f}fps "
                    f"stage1={kwargs['stage1_steps']} stage2={kwargs['stage2_steps']} "
                    f"audio={os.path.basename(audio_path)}"
                    f" a2v_scale={kwargs.get('audio_conditioning_scale', 1.0)}"
                    f"{' image=' + os.path.basename(ref_image) if ref_image else ''}"
                ),
            })
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

    if action == "generate_hdr":
        # HDR via IC-LoRA. Phase 1 of IC-LoRA support in Phosphene.
        # Uses HDRICLoraPipeline from ltx_pipelines_mlx.hdr_ic_lora —
        # the upstream class that handles LoRA fusion at generate time,
        # the LogC3 inverse transform during decode, and writes the
        # standard SDR MP4 plus a companion .hdr.npz float32 tensor.
        # Phase 1 ships text-driven mode: video_conditioning defaults
        # to [] so no reference video is needed (the LoRA delta still
        # applies; the LogC3 inverse still runs). Phase 2 will add the
        # reference-video picker for SDR→HDR re-grading.
        #
        # Routing constraint: IC-LoRA requires the distilled checkpoint.
        # The panel forces quality=balanced when HDR is on, which means
        # model_dir lands on the Q4 distilled folder. Don't try to run
        # this against the Q8 dev model — the LoRA was trained against
        # the distilled checkpoint and the weights won't align.
        job_id = msg.get("id", "?")
        p = msg.get("params", {}) or {}
        model_dir = p.get("model_dir") or MODEL_ID
        seed = int(p.get("seed", -1))
        if seed == -1:
            seed = random.randint(0, 2**31 - 1)
        _is_busy = True
        try:
            t0 = time.time()
            _mlx_mem_reset_run()
            configure_acceleration("off")
            from ltx_pipelines_mlx.hdr_ic_lora import HDRICLoraPipeline
            loras = p.get("loras") or []
            if not loras:
                raise RuntimeError(
                    "HDR job requires the HDR LoRA in `loras`. The panel "
                    "should have injected Lightricks/LTX-2.3-22b-IC-LoRA-HDR."
                )
            # Resolve each LoRA path (HF repo id → local safetensors via
            # snapshot_download + largest-file pick; absolute path → pass-through).
            resolved = [
                (_resolve_lora_path(str(l["path"])), float(l.get("strength", 1.0)))
                for l in loras
            ]
            _preflight_distilled_loras(resolved, model_dir)
            num_frames = int(p["frames"])
            _apply_vae_streaming_decision(num_frames)
            # Tear down any existing cached pipeline before instantiating
            # HDRICLoraPipeline — it loads its own DiT + VAE encoder +
            # upsampler at init, so holding the t2v / i2v caches just
            # doubles the memory footprint.
            release_pipelines("hdr render incoming")
            pipe = HDRICLoraPipeline(
                model_dir=Path(model_dir),
                lora_paths=resolved,
                low_memory=LOW_MEMORY,
            )
            emit({"event": "log",
                  "line": f"step:generate_hdr {p['width']}x{p['height']} "
                          f"{num_frames}f @{float(p.get('frame_rate', 24.0)):.1f}fps "
                          f"stage1={int(p.get('stage1_steps', 10))} "
                          f"stage2={int(p.get('stage2_steps', 3))} "
                          f"loras={len(resolved)} "
                          f"ref_videos={len(p.get('video_conditioning') or [])}"})
            kwargs = dict(
                prompt=p["prompt"],
                output_path=p["output_path"],
                video_conditioning=p.get("video_conditioning") or [],
                height=int(p["height"]),
                width=int(p["width"]),
                num_frames=num_frames,
                frame_rate=float(p.get("frame_rate", 24.0)),
                seed=seed,
                stage1_steps=_thin_cap(model_dir, "stage1_steps",
                                       int(p.get("stage1_steps", 8))),
                stage2_steps=_thin_cap(model_dir, "stage2_steps",
                                       int(p.get("stage2_steps", 3))),
            )
            kwargs = _filter_unsupported_kwargs(pipe.generate_and_save, kwargs)
            out_path = pipe.generate_and_save(**kwargs)
            # Drop the HDRICLoraPipeline aggressively — we don't cache
            # it the way we do t2v/i2v because HDR jobs are rare and
            # the pipeline's DiT+upsampler+decoders cost is substantial.
            try:
                pipe = None
                from ltx_core_mlx.utils.memory import aggressive_cleanup as _ac
                _ac()
            except Exception:
                pass
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

    if action == "generate_restore":
        # IC-LoRA reference-conditioned render. Serves TWO panel modes through
        # one code path (both fuse an IC-LoRA + ride a source clip on the IC
        # reference channel via ICLoraPipeline):
        #   - Colorize (mode=restore): the source is a B&W clip; LoRA strength
        #     1.0; two-stage (skip_stage_2 unset). The first real IC-LoRA
        #     feature on un-gated weights (DoctorDiffusion Colorizer).
        #   - Ingredients (mode=ingredients): the "source" is the reference
        #     SHEET looped to N frames (the panel composes + writes it); LoRA
        #     strength 1.4; single-stage (skip_stage_2=True). Flagship
        #     multi-reference composition.
        # Mechanically this mirrors generate_hdr but with TWO deliberate
        # differences:
        #   (a) ICLoraPipeline (ltx_pipelines_mlx.ic_lora), NOT
        #       HDRICLoraPipeline. The community Colorize LoRA carries no
        #       hdr_transform metadata — only reference_downscale_factor=1 —
        #       so the LogC3 inverse + HDR .npz companion the HDR class adds
        #       would be wrong here. ICLoraPipeline fuses the LoRA, runs the
        #       two-stage walk, and writes a plain SDR mp4.
        #   (b) A non-empty SOURCE video is REQUIRED. It rides the IC
        #       reference channel as video_conditioning=[(src, strength)] —
        #       the B&W clip the user wants colorized. (HDR Phase 1 ran
        #       text-driven with video_conditioning=[].) Verified on MLX by
        #       the de-risk render (scratchpad/ic_colorize_smoke.py).
        #
        # Routing: like HDR, the LoRA was trained against the Q4 distilled
        # checkpoint, so the panel forces model_dir onto the Q4 distilled
        # folder. ffprobe (probe_video_info, called inside generate_and_save
        # for the reference video) needs FFMPEG_BIN on PATH — the helper
        # inherits it from the panel (mlx_ltx_panel.py:5008).
        job_id = msg.get("id", "?")
        p = msg.get("params", {}) or {}
        model_dir = p.get("model_dir") or MODEL_ID
        seed = int(p.get("seed", -1))
        if seed == -1:
            seed = random.randint(0, 2**31 - 1)
        _is_busy = True
        try:
            t0 = time.time()
            _mlx_mem_reset_run()
            configure_acceleration("off")
            from ltx_pipelines_mlx.ic_lora import ICLoraPipeline
            loras = p.get("loras") or []
            if not loras:
                raise RuntimeError(
                    "Restore job requires the Colorize IC-LoRA in `loras`. The "
                    "panel should have injected the curated colorize LoRA."
                )
            # A source video is mandatory — it IS the IC reference channel.
            video_conditioning = p.get("video_conditioning") or []
            if not video_conditioning:
                raise RuntimeError(
                    "Restore (Colorize) requires a source video. The panel "
                    "should have set video_conditioning=[(src, strength)] from "
                    "restore_video_path."
                )
            # Resolve each LoRA path (HF repo id → local safetensors via
            # snapshot_download + largest-file pick; absolute path → pass-through).
            resolved = [
                (_resolve_lora_path(str(l["path"])), float(l.get("strength", 1.0)))
                for l in loras
            ]
            _preflight_distilled_loras(resolved, model_dir)
            num_frames = int(p["frames"])
            _apply_vae_streaming_decision(num_frames)
            # Tear down any existing cached pipeline before instantiating
            # ICLoraPipeline — it loads its own DiT + VAE at init, so holding
            # the t2v / i2v caches just doubles the memory footprint.
            release_pipelines("restore render incoming")
            # gemma_model_id was never passed here, so every IC lane (Colorize,
            # Ingredients, Control, Upscale ×2) encoded its prompt with the
            # class default — Gemma 3 from the HF cache — even on an LTX-2.5
            # checkpoint, whose text encoder is the Gemma 4 fine-tune the
            # panel hands us in LTX_GEMMA. The references carried the shot so
            # nobody noticed; the prompt was the part being ignored.
            # lora_mode "unfused": keep the adapter as a runtime wrapper instead
            # of folding it into the quantized weights. Fusing re-quantizes
            # W + B@A back onto the int grid — the fork measures 22.6% of this
            # upscaler's delta lost at int8 and ~150% at int4 (i.e. nothing of
            # the adapter survives on Q4). The native loader attaches
            # `_pending_loras` exactly, at any bit width; the IC pipeline's own
            # fuse step is skipped by handing it an empty list, so the reference
            # downscale factor it would have read from the file is set by hand.
            lora_mode = str(p.get("lora_mode") or "fused")
            if lora_mode == "unfused":
                try:
                    _install_lora_fusion_patches()
                except Exception:                                  # noqa: BLE001
                    pass
                from ltx_pipelines_mlx.iclora_utils import read_lora_reference_downscale_factor
                pipe = ICLoraPipeline(
                    model_dir=Path(model_dir),
                    lora_paths=[],
                    gemma_model_id=GEMMA_PATH,
                    low_memory=LOW_MEMORY,
                )
                pipe._pending_loras = list(resolved)
                for _lp, _ in resolved:
                    _scale = read_lora_reference_downscale_factor(_lp)
                    if _scale != 1:
                        pipe.reference_downscale_factor = _scale
                emit({"event": "log",
                      "line": f"IC-LoRA applied UNFUSED (exact runtime wrapper), "
                              f"reference downscale ×{pipe.reference_downscale_factor}"})
            else:
                pipe = ICLoraPipeline(
                    model_dir=Path(model_dir),
                    lora_paths=resolved,
                    gemma_model_id=GEMMA_PATH,
                    low_memory=LOW_MEMORY,
                )
            emit({"event": "log",
                  "line": f"step:generate_restore {p['width']}x{p['height']} "
                          f"{num_frames}f @{float(p.get('frame_rate', 24.0)):.1f}fps "
                          f"stage1={int(p.get('stage1_steps', 8))} "
                          f"stage2={int(p.get('stage2_steps', 3))} "
                          f"loras={len(resolved)} "
                          f"ref_videos={len(video_conditioning)} "
                          f"ref_strengths={[round(float(s), 3) for _, s in video_conditioning]} "
                          f"ref_downscale={getattr(pipe, 'reference_downscale_factor', '?')}"})
            kwargs = dict(
                prompt=p["prompt"],
                output_path=p["output_path"],
                video_conditioning=video_conditioning,
                height=int(p["height"]),
                width=int(p["width"]),
                num_frames=num_frames,
                frame_rate=float(p.get("frame_rate", 24.0)),
                seed=seed,
                # Clamped to this checkpoint's fixed table — ICLoraPipeline THINS
                # it, so the table's own length is the ceiling and an over-ask is a
                # pad request the sampler refuses after the model has loaded.
                stage1_steps=_thin_cap(model_dir, "stage1_steps",
                                       int(p.get("stage1_steps", 8))),
                stage2_steps=_thin_cap(model_dir, "stage2_steps",
                                       int(p.get("stage2_steps", 3))),
            )
            # Ingredients (multi-reference) reuses this same action but runs the
            # public Space's single-stage recipe — skip_stage_2=True, generated
            # at 2x the sheet resolution. Colorize leaves this False (two-stage),
            # so restore stays byte-identical. Only set the kwarg when the panel
            # asked for it, so the pipeline default still wins otherwise.
            if p.get("skip_stage_2"):
                kwargs["skip_stage_2"] = True
            # Upscale ×2 (v4.11): one pass at the full target size — the
            # reference IS the low-res clip, so a half-res stage 1 would only
            # throw its detail away — and the "keep the shot" slider rides on
            # the reference's attention strength.
            if p.get("single_stage"):
                kwargs["single_stage"] = True
            if p.get("conditioning_attention_strength") is not None:
                kwargs["conditioning_attention_strength"] = float(
                    p["conditioning_attention_strength"])
            # Upscale ×2 from the clip itself (Stage 1 skipped, see
            # ICLoraPipeline.generate): the clip's latent is upsampled and only
            # `refine_steps` of the distilled tail run at full res.
            if p.get("source_video"):
                kwargs["source_video"] = str(p["source_video"])
                kwargs.pop("single_stage", None)
            if p.get("refine_steps") is not None:
                kwargs["refine_steps"] = int(p["refine_steps"])
            kwargs = _filter_unsupported_kwargs(pipe.generate_and_save, kwargs)
            # Ingredients is single-stage ONLY (the public Space's recipe). If the
            # imported pipeline package is an older/pinned build whose
            # generate_and_save lacks `skip_stage_2`, _filter_unsupported_kwargs
            # would have SILENTLY dropped it → a two-stage run whose clean,
            # no-LoRA Stage 2 refines against the raw reference and collapses the
            # output back to a faithful copy of the sheet (static, ~no motion).
            # Fail loudly here instead of shipping that broken render. Colorize
            # never sets skip_stage_2, so this guard is inert for it (byte-identical).
            if p.get("skip_stage_2") and "skip_stage_2" not in kwargs:
                raise RuntimeError(
                    "Ingredients requires single-stage generation (skip_stage_2), "
                    "but the imported ltx-pipelines-mlx build's generate_and_save "
                    "does not accept it — it would silently run two-stage and "
                    "produce a static copy of the reference sheet. Upgrade/repair "
                    "the ltx-2-mlx pipeline (need a build with skip_stage_2; "
                    "v0.14.8+ has it; the current pin is v0.14.19)."
                )
            out_path = pipe.generate_and_save(**kwargs)
            # Drop the pipeline aggressively — restore jobs are rare and the
            # DiT+VAE cost is substantial; don't cache it like t2v/i2v.
            try:
                pipe = None
                from ltx_core_mlx.utils.memory import aggressive_cleanup as _ac
                _ac()
            except Exception:
                pass
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
        #
        # 2026-05-20 — augmented system prompt + trigger-word preservation.
        # The upstream Lightricks system prompt is good but two problems
        # showed up in practice (Mr Bizarro report):
        #
        #   1. Trigger words like `bizarrotrn` survived but got CAPITALIZED
        #      ("Bizarrotrn"). The LoRA was trained against the exact
        #      lowercase token — re-casing breaks tokenization and the
        #      LoRA may not fire as cleanly.
        #
        #   2. The official prompt says "if input is vague, invent concrete
        #      details" → Gemma invented "futuristic motorcycle" / "neon-
        #      lit avenue" / "polished chrome" when the user just said
        #      "cool bike". That over-decorates renders away from the
        #      character LoRA's training distribution.
        #
        # Fix: pass a system_prompt= override that extends the official
        # one with PRESERVE-EXACTLY rules for the panel-supplied
        # preserve_tokens list, plus a "restrained invention" addendum.
        # Then do a post-hoc safety pass: if any preserve token isn't
        # in the output case-exact, splice it back in by replacing the
        # case-shifted variant.
        job_id = msg.get("id", "?")
        p = msg.get("params", {}) or {}
        user_prompt = (p.get("prompt") or "").strip()
        mode = (p.get("mode") or "t2v").lower()
        if mode not in ("t2v", "i2v"):
            mode = "t2v"
        seed = int(p.get("seed", 10))
        preserve_tokens = p.get("preserve_tokens") or []
        if not isinstance(preserve_tokens, list):
            preserve_tokens = []
        preserve_tokens = [str(t).strip() for t in preserve_tokens if str(t).strip()]
        if not user_prompt:
            emit({"event": "error", "id": job_id, "error": "empty prompt"})
            continue
        _is_busy = True
        try:
            t0 = time.time()
            _mlx_mem_reset_run()
            lm = get_gemma_lm()
            # Build augmented system prompt: official + Phosphene addendum.
            base_sys = (lm.default_gemma_t2v_system_prompt if mode == "t2v"
                        else lm.default_gemma_i2v_system_prompt)
            addendum_lines = [
                "",
                "#### Phosphene addendum (overrides any conflict above):",
                "- Preserve every camera-move and shot description the user",
                "  wrote verbatim. Do not add invented camera motion.",
                "- If the user said 'cool bike' / 'a guy' / vague nouns,",
                "  keep them generic. DO NOT invent specific brands, model",
                "  numbers, or location styles (no 'futuristic', 'neon-lit',",
                "  'polished chrome' unless the user said so).",
                "- Color and lighting: prefer one or two concrete words",
                "  ('warm afternoon sun', 'overcast sky') over flowery",
                "  cascades ('shimmering golden hour bathed in...').",
                "- Audio sentence stays as one trailing line that begins",
                "  with 'Audio:'.",
            ]
            if preserve_tokens:
                addendum_lines += [
                    "",
                    "#### LoRA trigger tokens — PRESERVE CASE-EXACT:",
                    ("The user's prompt contains LoRA trigger tokens that "
                     "MUST appear in the output exactly as written, "
                     "lowercase, no spelling changes, no capitalization "
                     "changes, no substitutions:"),
                    "  " + ", ".join(preserve_tokens),
                    ("These tokens identify trained character / style LoRAs. "
                     "Re-casing or rewording them breaks tokenization and "
                     "the LoRA will not fire. If you would normally rephrase "
                     "(e.g. 'Bizarrotrn the man' → 'a man named Bizarro'), "
                     "DO NOT — emit the token verbatim, in lowercase."),
                ]
            augmented_sys = base_sys + "\n" + "\n".join(addendum_lines)
            if mode == "t2v":
                enhanced = lm.enhance_t2v(user_prompt, seed=seed,
                                           system_prompt=augmented_sys)
            else:
                enhanced = lm.enhance_i2v(user_prompt, seed=seed,
                                           system_prompt=augmented_sys)
            # Post-hoc safety: case-exact restore for any preserve token
            # that Gemma still managed to mutate. Catches "Bizarrotrn" →
            # restore "bizarrotrn".
            restored: list[str] = []
            for tok in preserve_tokens:
                if not tok:
                    continue
                if tok in enhanced:
                    continue  # already case-exact, no work
                # Case-insensitive locate + replace with case-exact token.
                import re
                pat = re.compile(re.escape(tok), re.IGNORECASE)
                new_enhanced, n = pat.subn(tok, enhanced)
                if n > 0:
                    enhanced = new_enhanced
                    restored.append(tok)
            if restored:
                emit({"event": "log",
                      "line": f"  [enhance] restored case-exact trigger(s): {', '.join(restored)}"})
            elapsed = round(time.time() - t0, 2)
            _last_activity = time.time()
            emit({
                "event": "done", "id": job_id,
                "enhanced": enhanced,
                "original": user_prompt,
                "mode": mode,
                "elapsed_sec": elapsed,
                "preserve_tokens": preserve_tokens,
                "restored_tokens": restored,
            })
        except Exception as exc:
            _last_activity = time.time()
            emit({"event": "error", "id": job_id, "error": str(exc), "trace": traceback.format_exc()})
        finally:
            _is_busy = False
        continue

    emit({"event": "error", "error": f"unknown action: {action}"})

emit({"event": "exit", "reason": "stdin_closed"})
