#!/usr/bin/env python3
"""Idempotent FBCache patch for mflux's Qwen + Flux2 transformers.

Why this exists:
  mflux ships no diffusion-step caching (TeaCache / FBCache / etc.).
  Issue #113 (TeaCache request) has been open since early 2025 with
  no implementation. For Phosphene's Qwen-Image-Edit + Flux2 image
  paths we want the same acceleration HiDream got — skip 35 of 60
  transformer blocks on diffusion steps where the first-block residual
  is stable.

What this patches:
  1. mflux/models/qwen/model/qwen_transformer/qwen_transformer.py
     Replaces the inner `for idx, block in enumerate(self.transformer_blocks)`
     loop in QwenTransformer.__call__ with a call to a new method
     `_run_blocks_with_optional_fbcache` that implements the cache.

  Future patch targets (not yet implemented; needs more design work
  because the loop structure is more involved — double-stream then
  single-stream):
  2. mflux/models/flux2/model/flux2_transformer/transformer.py

Toggle:
  Set env var MFLUX_FB_CACHE=1 before invoking mflux to enable. With
  the var unset / 0, the patched code runs the original plain loop —
  no behavior change.

  Tuning:
    MFLUX_FB_THRESHOLD   (default 0.15) — relative-L1 cutoff on layer-0
                                          residual delta. Lower = stricter
                                          (more recompute, less risk).
                                          Higher = more skips, more risk
                                          of "creamy" output.
    MFLUX_FB_KEEP_LAST   (default 8)    — number of FINAL transformer
                                          blocks that always run, even
                                          on cached steps. Keeps the
                                          detail-recovery layers alive.

Pin contract:
  This patch is line-targeted against `mflux==0.17.5`. Bumping mflux
  WITHOUT re-validating this patch will silently break: either the patch
  marker is already present and we skip (potentially leaving the new
  mflux running unaccelerated) or our text-search fails and the install
  step errors out loud.

  To upgrade mflux:
    1. Bump pin in install_qwen.js + update.js.
    2. Re-run this patch script and verify each section reports
       "[patched]" not "[not found]" or "[already patched]".
    3. Bench Qwen Fast / Medium before+after on a known prompt.

Safe to re-run: each patch checks for its `FBCACHE_PATCH_v1` marker
before touching anything.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Walk the venv site-packages from the repo root. mflux always ends up
# under the ltx-2-mlx env regardless of who installed it.
VENV_ROOTS = [
    "ltx-2-mlx/env/lib/python3.11/site-packages",
    "ltx-2-mlx/.venv/lib/python3.11/site-packages",
]

PATCH_MARKER = "# FBCACHE_PATCH_v1"


def _find(rel: str) -> Path | None:
    for root in VENV_ROOTS:
        p = Path(root) / rel
        if p.exists():
            return p
    return None


# ---- Patch 1: Qwen transformer FBCache ---------------------------------------
# The exact slice of code we replace. Anchored on the loop body. If mflux
# refactors this we'll get a clean "[not found]" and the user knows to
# investigate.
QWEN_LOOP_OLD = """        for idx, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = QwenTransformer._apply_transformer_block(
                idx=idx,
                block=block,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                text_embeddings=text_embeddings,
                image_rotary_embeddings=image_rotary_embeddings,
            )"""

QWEN_LOOP_NEW = """        # """ + PATCH_MARKER + """ — optional First-Block Cache. Skips the
        # bulk of the transformer blocks on diffusion steps where the
        # first block's input->output residual is stable vs the prev step.
        # Off unless MFLUX_FB_CACHE=1 is set in the subprocess env (so
        # the original behaviour is preserved by default).
        encoder_hidden_states, hidden_states = QwenTransformer._run_blocks_with_optional_fbcache(
            t=t,
            blocks=self.transformer_blocks,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            text_embeddings=text_embeddings,
            image_rotary_embeddings=image_rotary_embeddings,
        )"""

# The new helper method that gets injected as a class method. Placed
# right after _apply_transformer_block so it stays adjacent in the file.
# Module-level cache state — one slot per running subprocess. The mflux
# subprocess is single-image-per-process for our usage so we don't need
# instance-keyed state. Reset when `t` decreases (= new generation).
QWEN_HELPER_INSERT_AFTER = """        return block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            text_embeddings=text_embeddings,
            image_rotary_emb=image_rotary_embeddings,
            block_idx=idx,
        )"""

QWEN_HELPER_NEW = QWEN_HELPER_INSERT_AFTER + """

    # """ + PATCH_MARKER + """ — FBCache helper. Skip middle transformer
    # blocks when the layer-0 residual barely changed step-to-step.
    # Always runs layer 0 (for the residual) and the last keep_last
    # layers (for detail recovery). Pure no-op when MFLUX_FB_CACHE!=1.
    _FB_CACHE_STATE: dict = {}

    @staticmethod
    def _run_blocks_with_optional_fbcache(
        t,
        blocks,
        hidden_states,
        encoder_hidden_states,
        encoder_hidden_states_mask,
        text_embeddings,
        image_rotary_embeddings,
    ):
        import os as _os
        if _os.environ.get(\"MFLUX_FB_CACHE\", \"\") != \"1\":
            for idx, block in enumerate(blocks):
                encoder_hidden_states, hidden_states = QwenTransformer._apply_transformer_block(
                    idx=idx,
                    block=block,
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    text_embeddings=text_embeddings,
                    image_rotary_embeddings=image_rotary_embeddings,
                )
            return encoder_hidden_states, hidden_states

        threshold = float(_os.environ.get(\"MFLUX_FB_THRESHOLD\", \"0.15\"))
        keep_last = int(_os.environ.get(\"MFLUX_FB_KEEP_LAST\", \"8\"))
        n_layers = len(blocks)
        keep_last = max(1, min(keep_last, n_layers - 2))
        pre_last_idx = n_layers - keep_last
        is_first_step = (int(t) == 0)
        # Shape-keyed cache. mflux's Qwen-Edit pipeline calls the
        # transformer TWICE per diffusion step under true-CFG (positive
        # cond + negative cond), and the two passes have DIFFERENT
        # encoder_hidden_states sequence lengths (positive prompt vs
        # negative prompt). One global cache cross-contaminates the two
        # and we see "Shapes (1,214,3072) and (1,187,3072) cannot be
        # broadcast." Key the cache by (hidden_shape, encoder_shape) so
        # cond and uncond get separate slots.
        key = (tuple(hidden_states.shape), tuple(encoder_hidden_states.shape))
        all_slots = QwenTransformer._FB_CACHE_STATE
        state = all_slots.get(key)
        if state is None or int(t) <= state.get(\"prev_t\", -1):
            state = {\"prev_t\": int(t)}
            all_slots[key] = state
            # If t reset, also clear all other shape slots (they're from
            # a previous generation that's now ended).
            if int(t) == 0:
                for k in list(all_slots.keys()):
                    if k != key:
                        del all_slots[k]
        else:
            state[\"prev_t\"] = int(t)

        # Always run block 0 — its output is our skip decision input.
        encoder_hidden_states, hidden_states = QwenTransformer._apply_transformer_block(
            idx=0,
            block=blocks[0],
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            text_embeddings=text_embeddings,
            image_rotary_embeddings=image_rotary_embeddings,
        )
        h_after_first = hidden_states
        e_after_first = encoder_hidden_states

        skip = False
        if (not is_first_step
                and state.get(\"prev_residual\") is not None
                and state.get(\"cached_h_pre_last\") is not None
                and state.get(\"cached_e_pre_last\") is not None):
            ref_prev = state[\"prev_residual\"]
            # We approximate the residual by comparing layer-0's image-
            # stream output across steps. (mflux's QwenTransformerBlock
            # mutates both streams in place — the image stream is the
            # dominant signal for the cache decision.)
            diff = mx.mean(mx.abs(h_after_first - ref_prev))
            ref = mx.mean(mx.abs(ref_prev)) + 1e-6
            rel = (diff / ref).item()
            skip = (rel < threshold)
            state[\"last_rel\"] = rel

        if skip:
            # Reuse the cached pre-last-layer state directly. We tried a
            # delta-corrected variant (cached + today's layer-0 delta) but
            # mflux's encoder_hidden_states can be different shape across
            # calls (Shape (1,214,3072) vs (1,187,3072) seen in practice),
            # which broke broadcasting. The direct reuse is simpler and
            # matches what worked for the HiDream FBCache port.
            hidden_states = state[\"cached_h_pre_last\"]
            encoder_hidden_states = state[\"cached_e_pre_last\"]
            for idx in range(pre_last_idx, n_layers):
                encoder_hidden_states, hidden_states = QwenTransformer._apply_transformer_block(
                    idx=idx,
                    block=blocks[idx],
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    text_embeddings=text_embeddings,
                    image_rotary_embeddings=image_rotary_embeddings,
                )
        else:
            for idx in range(1, n_layers):
                if idx == pre_last_idx:
                    state[\"cached_h_pre_last\"] = hidden_states
                    state[\"cached_e_pre_last\"] = encoder_hidden_states
                encoder_hidden_states, hidden_states = QwenTransformer._apply_transformer_block(
                    idx=idx,
                    block=blocks[idx],
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    text_embeddings=text_embeddings,
                    image_rotary_embeddings=image_rotary_embeddings,
                )
            if \"cached_h_pre_last\" not in state:
                state[\"cached_h_pre_last\"] = h_after_first
                state[\"cached_e_pre_last\"] = e_after_first

        state[\"prev_residual\"] = h_after_first
        return encoder_hidden_states, hidden_states"""


def _patch_qwen() -> int:
    target = _find("mflux/models/qwen/model/qwen_transformer/qwen_transformer.py")
    if target is None:
        print("[qwen] mflux qwen_transformer.py not found — mflux not installed in expected venv root")
        return 1
    src = target.read_text()
    if PATCH_MARKER in src:
        print(f"[qwen] {target.name}: already patched (marker present)")
        return 0
    if QWEN_LOOP_OLD not in src:
        print(f"[qwen] {target.name}: target loop NOT FOUND — mflux version drift?")
        print(f"       Pin contract: this script targets mflux==0.17.5.")
        print(f"       Either bump the pin + re-validate this script, or skip.")
        return 1
    if QWEN_HELPER_INSERT_AFTER not in src:
        print(f"[qwen] {target.name}: helper insertion anchor NOT FOUND")
        return 1
    new_src = src.replace(QWEN_LOOP_OLD, QWEN_LOOP_NEW)
    new_src = new_src.replace(QWEN_HELPER_INSERT_AFTER, QWEN_HELPER_NEW)
    target.write_text(new_src)
    print(f"[qwen] {target.name}: patched (FBCache injected, gated on MFLUX_FB_CACHE=1)")
    return 0


def main() -> int:
    rc = _patch_qwen()
    # Future: _patch_flux2() — flux2 transformer has a more complex
    # double-stream + single-stream loop; deferred until Qwen FBCache is
    # benched and proven.
    return rc


if __name__ == "__main__":
    sys.exit(main())
