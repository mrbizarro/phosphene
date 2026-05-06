"""DeepCache for LTX-2 — block-skip caching to compound with TeaCache (Turbo).

Idea: TeaCache (Turbo) saves whole denoise steps by replaying a cached residual.
This module saves time WITHIN a kept-step by replaying cached PER-BLOCK residuals
for the middle blocks of the DiT stack. Inspired by DeepCache (Ma et al. 2024)
adapted for the joint A/V DiT.

Mechanism:
  - On a "compute" step the patched ``LTX2VideoModel.__call__`` runs every block
    AND records ``out_hidden - in_hidden`` per block.
  - On a "skip" step it runs only the edge blocks (configurable [0, K) and
    [N-K, N)) and replaces the middle blocks with the captured residuals.

Public API:
    enable_block_skip(controller: BlockSkipController) -> None
    disable_block_skip() -> None
    BlockSkipController(num_blocks=48, edge_blocks=12, schedule=...)

Activation patterns:
    schedule="alternate"      [full, skip, full, skip, ...]   ~1.25-1.43x denoise
    schedule="3of5"           [full, skip, skip, full, skip]  ~1.43x denoise
    schedule="aggressive"     [full, skip, skip, skip, skip]  ~1.67x denoise

Edge blocks K is HOW MANY blocks at each end stay always-active. With N=48 blocks
and K=12, blocks [0..11] and [36..47] always run; the middle 24 are skip
candidates. K=12 is the DeepCache "preserve early & late" pattern.

Quality risk:
    DeepCache for U-Net diffusion has shown 1.5-2.5x speedup with negligible
    quality loss. For DiT specifically, results are mixed but generally positive.
    LTX-2's gated, multi-modal block structure has not been validated yet —
    A/B testing on real renders is required before production rollout.

Idempotency:
    apply() is safe to call multiple times. The original __call__ is preserved
    on the class as ``_orig___call__``. revert() restores it.
"""
from __future__ import annotations

import os
from typing import Any, Optional

import mlx.core as mx


# ---- Module-level controller state (one active per process) ----
_active_controller: Optional["BlockSkipController"] = None


def enable_block_skip(controller: "BlockSkipController") -> None:
    """Activate block-skip with the given controller.

    The controller carries the per-job state: how many compute steps have
    happened, which blocks to skip, captured residuals.
    """
    global _active_controller
    _active_controller = controller


def disable_block_skip() -> None:
    """Deactivate block-skip — patched __call__ falls back to the original."""
    global _active_controller
    _active_controller = None


def get_active_controller() -> Optional["BlockSkipController"]:
    return _active_controller


# ---- Controller --------------------------------------------------------


class BlockSkipController:
    """Decide per step whether to compute or replay block residuals.

    Args:
        num_blocks: Total blocks in the DiT stack (LTX-2 = 48).
        edge_blocks: Blocks on each end that always run. With edge=12, blocks
            [0..11] and [N-12..N-1] always run; only middle 24 are skip-eligible.
        schedule: One of:
            "alternate"   — full, skip, full, skip, ...
            "3of5"        — full, skip, skip, full, skip   (5-step Turbo pattern)
            "aggressive"  — full, skip, skip, skip, skip   (1 of 5 compute)
            list[bool]    — explicit pattern; len must >= num steps
        verbose: When True, log per-step decisions to stdout.

    State:
        Controller maintains internal step counter, per-block residual cache
        (video, audio), and pass-label cache (cond, uncond, ptb, mod) since
        the sampler may invoke the model multiple times per outer step.
    """

    def __init__(
        self,
        num_blocks: int = 48,
        edge_blocks: int = 12,
        schedule: str | list[bool] = "3of5",
        skip_blocks: list[int] | None = None,
        skip_stride: int = 1,
        verbose: bool = False,
    ):
        """
        Args (extended):
            skip_blocks: Explicit list of block indices to skip. If provided,
                overrides edge_blocks/skip_stride. None = use edge-based default.
            skip_stride: When using edge-based default, only skip every Nth
                block in the middle range. stride=1 = all middle blocks (default),
                stride=2 = every other, stride=3 = every third. Reduces total
                skip count without shrinking the edge.
        """
        self.num_blocks = num_blocks
        self.edge_blocks = edge_blocks
        self.skip_stride = max(1, skip_stride)
        self.verbose = verbose
        self._step = 0
        self._pass_step: dict[str, int] = {}  # pass_label -> outer-step-counter
        self._cache: dict[str, dict[int, tuple[mx.array, mx.array]]] = {}
        # Skip-set resolution: explicit list wins, else edge+stride
        if skip_blocks is not None:
            self._skip_set = set(int(i) for i in skip_blocks)
        else:
            mid = list(range(edge_blocks, num_blocks - edge_blocks))
            self._skip_set = set(mid[::self.skip_stride])
        # Schedule resolution
        if isinstance(schedule, list):
            self._pattern = list(schedule)
        else:
            self._pattern = self._resolve_schedule(schedule)

    @staticmethod
    def _resolve_schedule(name: str) -> list[bool]:
        # True = compute (full), False = skip
        if name == "alternate":
            return [True, False] * 32   # long enough for any reasonable step count
        elif name == "3of5":
            return [True, False, False, True, False] * 16
        elif name == "aggressive":
            return [True, False, False, False, False] * 16
        else:
            raise ValueError(f"Unknown schedule: {name!r}")

    @property
    def skip_set(self) -> set[int]:
        """Indices of blocks that are skip-eligible (resolved at construction)."""
        return self._skip_set

    def is_compute_step(self, pass_label: str) -> bool:
        """True if THIS pass should compute all blocks (and capture residuals).

        Pass labels (cond / uncond / ptb / mod) are tracked separately because
        each may have its own residual stream.
        """
        s = self._pass_step.get(pass_label, 0)
        return self._pattern[s % len(self._pattern)]

    def advance_pass(self, pass_label: str) -> None:
        self._pass_step[pass_label] = self._pass_step.get(pass_label, 0) + 1

    def reset(self) -> None:
        self._step = 0
        self._pass_step.clear()
        self._cache.clear()

    def get_cached_residuals(self, pass_label: str) -> Optional[dict[int, tuple[mx.array, mx.array]]]:
        return self._cache.get(pass_label)

    def store_cached_residuals(
        self, pass_label: str, residuals: dict[int, tuple[mx.array, mx.array]]
    ) -> None:
        # Keep ONE step of residuals per pass label
        self._cache[pass_label] = residuals


# ---- The patched forward ----------------------------------------------


def _patched_model_call(self, *args, **kwargs):
    """Drop-in replacement for ``LTXModel.__call__`` with optional block-skip.

    Falls through to the original implementation when no controller is active.
    Otherwise, intercepts the block-stack iteration to skip middle blocks on
    schedule and reuse captured residuals.

    NOTE: ``_block_skip_pass_label`` is popped UNCONDITIONALLY so that even when
    the patch is loaded but not active, an injected sampler kwarg never reaches
    the original ``__call__`` (which would reject it as unknown).
    """
    # ALWAYS pop our private kwarg first so it never leaks into the original
    pass_label = kwargs.pop("_block_skip_pass_label", "default")

    ctrl = _active_controller
    if ctrl is None:
        return _ORIGINAL_CALL(self, *args, **kwargs)

    # Sniff: if compute step, just run normally + tap residuals per block
    is_compute = ctrl.is_compute_step(pass_label)
    captured: dict[int, tuple[mx.array, mx.array]] = {} if is_compute else None
    cached = None if is_compute else ctrl.get_cached_residuals(pass_label)

    if not is_compute and cached is None:
        # No cache yet (first step) — must compute even if scheduled to skip.
        is_compute = True
        captured = {}

    # We need finer granularity than block_stack_override gives. The minimal-risk
    # approach: invoke the per-block iteration ourselves by stealing the
    # transformer_blocks list and the precomputed kwargs.
    # We do this by calling model with block_stack_override = our custom iterator.
    skip_set = ctrl.skip_set if not is_compute else set()

    # Build the override. It needs access to per-block kwargs computed inside
    # the original __call__ (rope_freqs, adaln_params, etc.). The cleanest way
    # is to compute those once at the top of the original __call__, then pass
    # them to our override via a closure. Since the original __call__ doesn't
    # expose those kwargs, we inline a copy of its logic here.
    return _call_with_block_skip(
        self,
        *args,
        **kwargs,
        ctrl=ctrl,
        pass_label=pass_label,
        is_compute=is_compute,
        captured=captured,
        cached=cached,
    )


def _call_with_block_skip(
    self,
    video_latent: mx.array,
    audio_latent: mx.array,
    timestep: mx.array,
    *,
    ctrl: BlockSkipController,
    pass_label: str,
    is_compute: bool,
    captured: Optional[dict],
    cached: Optional[dict],
    video_text_embeds=None,
    audio_text_embeds=None,
    video_positions=None,
    audio_positions=None,
    video_attention_mask=None,
    audio_attention_mask=None,
    video_timesteps=None,
    audio_timesteps=None,
    perturbations=None,
    tap=None,
    block_stack_override=None,
):
    """Replicates LTX2VideoModel.__call__ with per-block skip support.

    See model.py:212 for the original. Only the block-stack iteration differs;
    the prelude (timestep embeds, AdaLN, RoPE precompute) is unchanged. We must
    inline that prelude here because the original doesn't expose the precomputed
    intermediates.
    """
    # CRITICAL: when block_stack_override is also requested, fall back to the
    # original — it's a TeaCache full-step skip, which is more aggressive
    # than block-skip and we should let it win.
    #
    # IMPORTANT: do NOT advance our schedule on the fallback. TeaCache-cached
    # steps don't run any blocks — our schedule should track only the calls
    # where we actually decide compute-or-skip. Otherwise the schedule races
    # through positions while the residual cache remains stale, and "skip"
    # decisions reach across many outer steps' worth of latent evolution.
    if block_stack_override is not None:
        return _ORIGINAL_CALL(
            self,
            video_latent=video_latent,
            audio_latent=audio_latent,
            timestep=timestep,
            video_text_embeds=video_text_embeds,
            audio_text_embeds=audio_text_embeds,
            video_positions=video_positions,
            audio_positions=audio_positions,
            video_attention_mask=video_attention_mask,
            audio_attention_mask=audio_attention_mask,
            video_timesteps=video_timesteps,
            audio_timesteps=audio_timesteps,
            perturbations=perturbations,
            tap=tap,
            block_stack_override=block_stack_override,
        )

    # ----- Inline of LTX2VideoModel.__call__ prelude (model.py:260-355) -----
    # Cast inputs
    video_latent = video_latent.astype(mx.bfloat16)
    audio_latent = audio_latent.astype(mx.bfloat16)
    if video_text_embeds is not None:
        video_text_embeds = video_text_embeds.astype(mx.bfloat16)
    if audio_text_embeds is not None:
        audio_text_embeds = audio_text_embeds.astype(mx.bfloat16)

    video_hidden = self.patchify_proj(video_latent)
    audio_hidden = self.audio_patchify_proj(audio_latent)

    # Timestep + AdaLN
    timestep = timestep.astype(mx.bfloat16)
    t_emb = self._embed_timestep_scalar(timestep)
    av_ca_factor = (
        self.config.av_ca_timestep_scale_multiplier / self.config.timestep_scale_multiplier
    )
    from ltx_core_mlx.model.transformer.timestep_embedding import get_timestep_embedding

    t_emb_av_gate = get_timestep_embedding(
        timestep * self.config.timestep_scale_multiplier * av_ca_factor,
        self.config.timestep_embedding_dim,
    )

    if video_timesteps is not None:
        vt_emb = self._embed_timestep_per_token(video_timesteps)
        video_adaln_emb, video_embedded_ts = self._adaln_per_token(self.adaln_single, vt_emb)
        av_ca_video_emb, _ = self._adaln_per_token(self.av_ca_video_scale_shift_adaln_single, vt_emb)
    else:
        video_adaln_emb, video_embedded_ts = self.adaln_single(t_emb)
        av_ca_video_emb, _ = self.av_ca_video_scale_shift_adaln_single(t_emb)
    av_ca_a2v_gate_emb, _ = self.av_ca_a2v_gate_adaln_single(t_emb_av_gate)
    video_prompt_emb, _ = self.prompt_adaln_single(t_emb)

    if audio_timesteps is not None:
        at_emb = self._embed_timestep_per_token(audio_timesteps)
        audio_adaln_emb, audio_embedded_ts = self._adaln_per_token(self.audio_adaln_single, at_emb)
        av_ca_audio_emb, _ = self._adaln_per_token(self.av_ca_audio_scale_shift_adaln_single, at_emb)
    else:
        audio_adaln_emb, audio_embedded_ts = self.audio_adaln_single(t_emb)
        av_ca_audio_emb, _ = self.av_ca_audio_scale_shift_adaln_single(t_emb)
    av_ca_v2a_gate_emb, _ = self.av_ca_v2a_gate_adaln_single(t_emb_av_gate)
    audio_prompt_emb, _ = self.audio_prompt_adaln_single(t_emb)

    # RoPE
    video_rope_freqs = None
    audio_rope_freqs = None
    if video_positions is not None:
        video_rope_freqs = self._compute_rope_freqs(
            video_positions, self.config.video_num_heads, self.config.video_head_dim,
        )
    if audio_positions is not None:
        audio_rope_freqs = self._compute_rope_freqs(
            audio_positions, self.config.audio_num_heads, self.config.audio_head_dim,
            max_pos_override=list(self.config.audio_positional_embedding_max_pos),
        )

    cross_pe_max_pos = max(
        self.config.positional_embedding_max_pos[0],
        self.config.audio_positional_embedding_max_pos[0],
    )
    video_cross_rope_freqs = None
    audio_cross_rope_freqs = None
    if video_positions is not None:
        video_cross_rope_freqs = self._compute_rope_freqs(
            video_positions[:, :, 0:1], self.config.av_cross_num_heads,
            self.config.av_cross_head_dim, max_pos_override=[cross_pe_max_pos],
        )
    if audio_positions is not None:
        audio_cross_rope_freqs = self._compute_rope_freqs(
            audio_positions[:, :, 0:1], self.config.av_cross_num_heads,
            self.config.av_cross_head_dim, max_pos_override=[cross_pe_max_pos],
        )

    block_input_v = video_hidden
    block_input_a = audio_hidden

    # ----- Per-block iteration with skip support -----
    skip_set = ctrl.skip_set if not is_compute else set()
    for block_idx, block in enumerate(self.transformer_blocks):
        if block_idx in skip_set and cached is not None and block_idx in cached:
            v_res, a_res = cached[block_idx]
            video_hidden = video_hidden + v_res
            audio_hidden = audio_hidden + a_res
        else:
            v_in = video_hidden
            a_in = audio_hidden
            video_hidden, audio_hidden = block(
                video_hidden=video_hidden,
                audio_hidden=audio_hidden,
                video_adaln_params=video_adaln_emb,
                audio_adaln_params=audio_adaln_emb,
                video_prompt_adaln_params=video_prompt_emb,
                audio_prompt_adaln_params=audio_prompt_emb,
                av_ca_video_params=av_ca_video_emb,
                av_ca_audio_params=av_ca_audio_emb,
                av_ca_a2v_gate_params=av_ca_a2v_gate_emb,
                av_ca_v2a_gate_params=av_ca_v2a_gate_emb,
                video_text_embeds=video_text_embeds,
                audio_text_embeds=audio_text_embeds,
                video_rope_freqs=video_rope_freqs,
                audio_rope_freqs=audio_rope_freqs,
                video_cross_rope_freqs=video_cross_rope_freqs,
                audio_cross_rope_freqs=audio_cross_rope_freqs,
                video_attention_mask=video_attention_mask,
                audio_attention_mask=audio_attention_mask,
                perturbations=perturbations,
                block_idx=block_idx,
            )
            if captured is not None:
                # Capture only middle blocks (the ones we'd skip later)
                if block_idx in ctrl.skip_set:
                    captured[block_idx] = (video_hidden - v_in, audio_hidden - a_in)

    if captured is not None:
        ctrl.store_cached_residuals(pass_label, captured)
    ctrl.advance_pass(pass_label)

    if tap is not None:
        tap(video_hidden - block_input_v, audio_hidden - block_input_a)

    video_out = self._output_block(video_hidden, video_embedded_ts, self.scale_shift_table, self.proj_out)
    audio_out = self._output_block(
        audio_hidden, audio_embedded_ts, self.audio_scale_shift_table, self.audio_proj_out
    )
    return video_out, audio_out


# ---- Apply / revert ----------------------------------------------------

_ORIGINAL_CALL = None


def apply() -> None:
    """Patch LTX2VideoModel.__call__ with block-skip support.

    Idempotent — repeated calls are no-op after the first.
    """
    global _ORIGINAL_CALL
    from ltx_core_mlx.model.transformer.model import LTXModel

    if _ORIGINAL_CALL is not None:
        return  # already applied

    _ORIGINAL_CALL = LTXModel.__call__
    LTXModel.__call__ = _patched_model_call


def revert() -> None:
    """Restore the original __call__."""
    global _ORIGINAL_CALL
    from ltx_core_mlx.model.transformer.model import LTXModel

    if _ORIGINAL_CALL is None:
        return
    LTXModel.__call__ = _ORIGINAL_CALL
    _ORIGINAL_CALL = None


# ---- Optional: env-driven auto-enable ---------------------------------


def auto_enable_from_env() -> Optional[BlockSkipController]:
    """Enable block-skip from environment variables.

    LTX_BLOCK_SKIP=1                 — turn on
    LTX_BLOCK_SKIP_EDGE=12           — edge_blocks (default 12)
    LTX_BLOCK_SKIP_SCHEDULE=3of5     — alternate | 3of5 | aggressive (default 3of5)
    LTX_BLOCK_SKIP_STRIDE=1          — skip every Nth middle block (1 = all, 2 = every other)
    LTX_BLOCK_SKIP_VERBOSE=1         — log per-step decisions
    """
    if os.environ.get("LTX_BLOCK_SKIP", "0") not in ("1", "true", "yes"):
        return None
    edge = int(os.environ.get("LTX_BLOCK_SKIP_EDGE", "12"))
    schedule = os.environ.get("LTX_BLOCK_SKIP_SCHEDULE", "3of5")
    stride = int(os.environ.get("LTX_BLOCK_SKIP_STRIDE", "1"))
    verbose = os.environ.get("LTX_BLOCK_SKIP_VERBOSE", "0") in ("1", "true", "yes")
    apply()
    ctrl = BlockSkipController(num_blocks=48, edge_blocks=edge, schedule=schedule,
                                skip_stride=stride, verbose=verbose)
    enable_block_skip(ctrl)
    return ctrl


def set_config(
    schedule: Optional[str] = None,
    edge_blocks: int = 12,
    skip_stride: int = 1,
    skip_blocks: Optional[list[int]] = None,
    verbose: bool = False,
) -> Optional[BlockSkipController]:
    """Runtime config switch — install a new controller, replacing any active one.

    Pass schedule=None to DISABLE the patch.

    Args:
        schedule: "alternate" | "3of5" | "aggressive" | None (to disable)
        edge_blocks: edge size for default skip range
        skip_stride: stride in default middle range
        skip_blocks: explicit override (set of block indices)
        verbose: per-decision logging
    """
    if schedule is None:
        disable_block_skip()
        return None
    apply()  # idempotent
    ctrl = BlockSkipController(
        num_blocks=48, edge_blocks=edge_blocks, schedule=schedule,
        skip_stride=skip_stride, skip_blocks=skip_blocks, verbose=verbose,
    )
    enable_block_skip(ctrl)
    return ctrl
