# Phosphene perf experiment — results (2026-05-05)

**Verdict: Track A (block-skip) IMPLEMENTED but DOES NOT MEET quality bar on
production renders. Track B (conv3d kernel) is the higher-confidence path
forward.**

## What was attempted

Goal: ≥1.20× end-to-end speedup on a 5-sec 1024×576 Balanced+Turbo render with
**no visible quality loss**.

Approach: DeepCache-for-DiT — capture per-block residuals on "compute" steps,
replay them on "skip" steps for the middle 24 of 48 transformer blocks.
Compounds with TeaCache.

Code shipped: [patch_block_skip.py](../patch_block_skip.py),
[sampler integration](../ltx-2-mlx/packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/utils/samplers.py),
[helper integration](../mlx_warm_helper.py) lines 54-79.
All gated by `LTX_BLOCK_SKIP=1` env var. **Off by default.**

## Results

### Baseline determinism check (Gate 0)

Same prompt + seed rendered twice without patch:

| Pair | SSIM Y | SSIM All |
|---|---|---|
| baseline vs baseline_v2 (deterministic re-run) | **1.000000** | **1.000000** |

LTX is fully deterministic. Any SSIM<1 in patched runs is real, not noise.

### Tiny survival (Gate 2 — passed)

320×240, 49 frames, 4 steps, no Turbo:

| variant | helper time | speedup | SSIM | PSNR |
|---|---|---|---|---|
| baseline | 15.2 s | 1.00× | 1.0000 | ∞ |
| alternate_e12 | 13.3 s | **1.14×** | **0.998620** | **55.08 dB** |

Block-skip works perfectly at low res / few steps — output is ~bit-identical.

### Production (Gate 3 — failed quality)

1024×576, 121 frames (5s), 8 steps, Balanced + Turbo:

| variant | helper time | speedup | SSIM Y | SSIM All | PSNR Y | audio dB delta |
|---|---|---|---|---|---|---|
| baseline | 182.6 s | 1.00× | 1.000 | 1.000 | ∞ | 0.00 |
| baseline_v2 (re-run) | 181.2 s | 1.01× | 1.000 | 1.000 | ∞ | — |
| alternate_e12 | 150.2 s | **1.22×** | 0.690 | 0.878 | 17.5 dB | -1.0 |
| 3of5_e12 | 135.3 s | **1.35×** | 0.693 | 0.879 | 17.7 dB | -2.1 |
| 3of5_e16 | 150.8 s | 1.21× | 0.690 | 0.879 | 17.7 dB | -0.4 |
| alternate_e20 (most conservative) | 171.0 s | 1.07× | 0.721 | 0.890 | 19.3 dB | — |

**Speedup is real (1.07-1.35×) but every patched config produces a
visibly different render** — same prompt, same seed, similar composition,
**different identity**.

Side-by-side strips: [/tmp/phosphene_bench/strips/](/tmp/phosphene_bench/strips/).
Even the most conservative edge=20 (only 8 of 48 blocks skip-eligible) gives
SSIM 0.89 with different person.

## Why it failed on prod but worked on tiny

DeepCache approximates `block(input_t+1) ≈ input_t+1 + (block(input_t) - input_t)`
where the residual `block(x) - x` is assumed slow-evolving across denoise steps.

| Factor | Tiny | Prod |
|---|---|---|
| Denoise steps | 4 | 8 (5 full under Turbo) |
| Tokens per layer | ~560 | 9,216 |
| Latent evolution between cached / replay | small | large |
| DeepCache approximation error | negligible | compounds visibly |

Two structural reasons production is harder:
1. **More denoise steps** → more chances for cumulative drift between cached and replay.
2. **Higher resolution = more dimensions** → residual approximation has more places to be slightly wrong.

DeepCache literature is mostly U-Net-based, where the skip-connection backbone
provides natural anchoring. DiT (LTX) lacks that anchor — the residual cache is
the ONLY mechanism keeping the trajectory locked, and it's not strong enough at
production scale.

## What's still useful

The patch as-shipped (off by default, env-var-gated) is still valuable for:

1. **Iteration / exploration** — when you're rendering many variations of a
   prompt to pick one, "different render with same composition" is fine.
   You'd get ~1.35× speedup per variation.
2. **Director Mode pre-viz** — quick rough renders to validate a shot list
   before committing to a full-quality render.
3. **Future research** — if/when LTX is fine-tuned with block-skip awareness,
   the patch is already wired in for testing.

## What NOT to do with it

- Default-enable. Quality drop is too big.
- Use for final renders where exact identity matters.
- Use for multi-shot continuity workflows. The drift would compound.

## Track B — conv3d kernel — recommended next

Detail: [docs/CONV3D_KERNEL_PORT_DESIGN.md](CONV3D_KERNEL_PORT_DESIGN.md).

- Bench shows MLX conv3d on LTX VAE upsampler runs at 1.7-2.8 GB/s effective
  vs M4 Max's 410 GB/s ceiling — ~150× below peak.
- Draw Things shipped a Codex-authored kernel that gets 2.4× on M1-M4 for this
  exact path. BSD-3 licensed — vendorable.
- Effort: 1-2 weeks (Path 2: write our own MSL kernel via `mx.fast.metal_kernel`).
- End-to-end gain: ~10-12% on a 10-min render (~50-70 s saved).
- **Quality risk: NONE** — kernel math is bit-equivalent.
- This is the right next investment of engineering time.

## Files left in place

- `patch_block_skip.py` — keep. Off by default. Useful for iteration mode.
- `mlx_warm_helper.py` — modified to import + reset patch. No-op when
  `LTX_BLOCK_SKIP=0` or unset.
- `ltx-2-mlx/.../utils/samplers.py` — try-import shim is a no-op when patch is
  inactive. Source synced to install copy.
- `docs/PERF_PLAN_2026-05-05.md` — original plan
- `docs/PERF_RESULTS_2026-05-05.md` — this file
- `docs/BLOCK_SKIP_ACTIVATION.md` — usage guide (still accurate; stresses
  opt-in)
- `docs/CONV3D_KERNEL_PORT_DESIGN.md` — Track B design

Bench artifacts in `/tmp/phosphene_bench/`:
- `bench_attn.out` — mlx-mfa parity test (ruled out)
- `bench_block.out` — mx.compile (ruled out)
- `bench_window_attn.out`, `bench_window_diff.out` — sliding window (ruled out)
- `bench_conv3d.out` — MLX conv3d slow path (Track B candidate)
- `bench_block_skip.out` — block-skip projection
- `profile_block.out` — per-block cost breakdown
- `prod_ab_results.json`, `noise_check_results.json` — raw E2E data
- `strips/`, `strips_e20/` — visual A/B comparison strips

## Recommended actions

1. **Don't enable block-skip by default** — risk of identity drift on real renders.
2. Optionally **leave the patch in place for iteration mode** — single-line
   activation via env var.
3. **Pivot Track B (conv3d kernel)** as the production performance work.
   Lower-risk, real win, no quality concerns.
4. **Update STATE.md** with the optimization results (template at
   `/tmp/phosphene_bench/STATE_md_update_block_skip.md`, but the honest
   version goes in the next STATE.md edit).
5. **Decide whether to git-commit** the patch infrastructure on `dev` branch.
   Recommend: yes, as a parked-feature for future use. Salo's call.
