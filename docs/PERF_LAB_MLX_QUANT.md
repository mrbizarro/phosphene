# Perf lab — MLX quantization modes vs LTX-2.3 transformer

**Date:** 2026-05-07 · **Branch:** `perf-lab-mlx-quant` · **Hardware:** M4 Max 64 GB
**Status:** Hypothesis refuted. **Recommendation: ABANDON.**

## TL;DR

The hypothesis was: MLX's newer quant modes (`mxfp4`, `mxfp8`, `nvfp4`) might
beat the current dgrauet `affine int4 g64` weights on the LTX-2.3 transformer
denoise loop. They don't.

- **All four quant modes** (`affine`, `mxfp4`, `mxfp8`, `nvfp4`) are supported
  out of the box in MLX **0.31.1** (the version pinned for the audio fix).
  The May 5 perf-research doc's "MLX FP4 not supported yet" assumption is
  obsolete — they're all available now.
- At LTX's actual token counts (M ≈ 4 800 – 17 856 for our render menu),
  **bf16 matmul beats every quant mode by 7–15 %** on M4 Max. bf16 GEMM hits
  **14.6 TFLOPs** on a clean 4096³ matmul — within ~14 % of M4 Max's
  ~17 TFLOPs theoretical peak. There is no slack for a slower kernel to
  recover from a smaller weight footprint.
- At LTX's token counts, the new modes (`mxfp4`, `nvfp4`, `mxfp8`) are
  **1–3 % slower** than the current `affine int4 g64`. Switching modes
  is a regression, not an upgrade.
- Microbench infrastructure validated at small M: at M = 1 (bandwidth-bound,
  like LLM decode) `mxfp4` beats bf16 by **2.78×** on the FF projection.
  The quant kernels work; the workload is just in the wrong regime to
  benefit.

The brief's hard-stop fires. **Stopping. Do not integrate, do not keep
researching this lane.**

---

## Why this matters and what was tested

After the May 5 research blitz ruled out mlx-mfa SDPA, `mx.compile`, RoPE
caching, sliding-window attention, step reduction, and DeepCache-style
block-skip, the open question was whether MLX's newer FP4/FP8 quant modes
would beat the existing `affine int4` model. NVFP4 was specifically
called out in `docs/PERF_RESEARCH_2026-05-05.md` as the most promising
remaining candidate, blocked on "MLX FP4 support".

That assumption is now obsolete. MLX 0.31.1 supports all four. The
question becomes: do they actually run faster on our shapes?

### MLX 0.31.1 quantize API

Verified directly via `mx.quantize.__doc__`:

| mode    | group size | bits          | scale type | bias | status in 0.31.1 |
|---------|-----------|---------------|------------|------|------------------|
| affine  | 32, 64*, 128 | 2,3,4*,5,6,8 | input dtype | yes | OK (current Q4) |
| mxfp4   | 32*       | 4*            | e8m0       | no  | OK              |
| mxfp8   | 32*       | 8*            | e8m0       | no  | OK              |
| nvfp4   | 16*       | 4*            | e4m3       | no  | OK              |

`*` = default. Test commands in
`scripts/perf_lab/01_quant_modes_probe.py`. All four modes round-trip
quantize → quantized_matmul → eval cleanly on a 4096×4096 weight on M4
Max GPU.

### LTX transformer scope

From `mlx_models/ltx-2.3-mlx-q4/config.json`:

- 48 transformer blocks, each with video (4096-dim, 32 heads, head_dim=128)
  and audio (2048-dim, 32 heads, head_dim=64) streams.
- Per-block Linear shapes (the workhorses):
  - Video Q/K/V/O: 4096 × 4096 (×4 self-attn + ×4 text-cross + ×4 AV-cross)
  - Video FF: 4096 → 16384 (`proj_in`), 16384 → 4096 (`proj_out`)
  - Audio Q/K/V/O: 2048 × 2048
  - Audio FF: 2048 → 8192, 8192 → 2048
- `quantize_config.json` confirms the current Q4 model uses
  `bits=4 group_size=64 only_transformer_blocks=true` — VAE, audio VAE,
  vocoder, connector are bf16.

### Token-count sweep

LTX patchifier is identity (`patch_size_t = h = w = 1`); video token
count comes from the latent shape `(1 + (F-1)/8) × (H/32) × (W/32)`:

| recipe          | M (video tokens) |
|-----------------|------------------|
| 640 × 480 121f  | 4 800            |
| 1024 × 576 121f | 9 216            |
| 1280 × 704 121f | 14 080           |
| 1024 × 576 241f | 17 856           |

Audio token counts are O(256–1024) per clip — small enough that
quant might still help there, but audio is not on the critical
path (video dominates wall time).

---

## Microbench results

`scripts/perf_lab/02_microbench_linear.py`, 5 warmup + 30 timed runs per
cell, `mx.eval` after every call. Speedup = bf16 / mode (>1 = faster than
bf16). All numbers warm median.

### Video Q/K/V/O — 4096 × 4096

| M (tokens)      | bf16 ms | affine_int4_g64 | affine_int4_g32 | mxfp4_g32 | mxfp8_g32 | nvfp4_g16 |
|-----------------|--------:|----------------:|----------------:|----------:|----------:|----------:|
| 4 800           | 22.6    | 0.88×           | 0.88×           | 0.88×     | 0.86×     | 0.88×     |
| 9 216           | 41.5    | 0.88×           | 0.87×           | 0.87×     | 0.86×     | 0.87×     |
| 14 080          | 62.6    | 0.88×           | 0.87×           | 0.87×     | 0.85×     | 0.86×     |
| 17 856          | 78.7    | 0.88×           | 0.87×           | 0.86×     | 0.85×     | 0.87×     |

### Video FF in — 4096 → 16384

| M               | bf16 ms | affine_int4_g64 | mxfp4_g32 | mxfp8_g32 | nvfp4_g16 |
|-----------------|--------:|----------------:|----------:|----------:|----------:|
| 4 800           | 41.6    | 0.88×           | 0.87×     | 0.85×     | 0.87×     |
| 9 216           | 83.0    | 0.88×           | 0.87×     | 0.86×     | 0.87×     |
| 14 080          | 126.8   | 0.88×           | 0.87×     | 0.86×     | 0.87×     |
| 17 856          | 161.6   | 0.88×           | 0.86×     | 0.86×     | 0.88×     |

### Video FF out — 16384 → 4096

| M               | bf16 ms | affine_int4_g64 | mxfp4_g32 | mxfp8_g32 | nvfp4_g16 |
|-----------------|--------:|----------------:|----------:|----------:|----------:|
| 4 800           | 44.2    | 0.90×           | 0.89×     | 0.87×     | 0.88×     |
| 9 216           | 84.9    | 0.89×           | 0.89×     | 0.87×     | 0.88×     |
| 14 080          | 132.2   | 0.91×           | 0.89×     | 0.88×     | 0.88×     |
| 17 856          | 172.0   | 0.92×           | 0.93×     | 0.92×     | 0.93×     |

### Audio shapes (2048 × 2048, FF 2048↔8192)

Audio M = 256 / 1024. Roughly break-even — none of the modes give a
material win. Full table in `/tmp/phos_perf_lab/microbench.json`.

### Output error vs bf16 (max-abs on a single matmul)

| shape (worst case)    | affine_int4 | mxfp4 | mxfp8 | nvfp4 |
|-----------------------|------------:|------:|------:|------:|
| video_qkvo            | 0.36        | 0.45  | 0.23  | 0.39  |
| video_ff_in           | 0.37        | 0.46  | 0.25  | 0.44  |
| video_ff_out          | 0.68        | 0.90  | 0.48  | 0.78  |

`mxfp8` consistently has the lowest error (8-bit elements, e8m0 scale).
`mxfp4` and `nvfp4` have noticeably higher error than affine int4 on the
video FF — a tax to pay for kernels that aren't even faster.

### Sanity check at small M (bandwidth-bound regime)

`scripts/perf_lab/03_validate_bandwidth_regime.py`. Same harness, much
smaller M to confirm the bench infrastructure isn't broken:

| shape          | M  | bf16 ms | affine_int4 | mxfp4 | nvfp4 |
|----------------|----|--------:|------------:|------:|------:|
| qkvo 4096²     | 1  | 0.185   | **1.55×**   | 1.48× | 1.48× |
| qkvo 4096²     | 8  | 0.241   | 1.12×       | 1.42× | 1.44× |
| qkvo 4096²     | 64 | 0.293   | 0.97×       | 0.96× | 0.96× |
| qkvo 4096²     | 512| 1.369   | 0.91×       | 0.90× | 0.90× |
| ff_in 4096×16384 | 1 | 0.399 | 2.34×       | **2.78×** | **2.72×** |
| ff_in 4096×16384 | 8 | 0.852 | 2.03×       | 2.20× | 2.14× |
| ff_in 4096×16384 | 64| 0.895 | 1.04×       | 1.04× | 1.04× |
| ff_in 4096×16384 | 512 | 4.896 | 0.90×     | 0.89× | 0.89× |

The crossover is around M ≈ 64. LTX runs at M ≈ 4 800 – 17 856 — two
orders of magnitude past the crossover.

The bench is not lying about LTX. The kernels work. They just don't help
at this scale.

---

## Why bf16 wins at LTX scale

- bf16 matmul on M4 Max GPU: 14.6 TFLOPs measured on a clean 4096³
  GEMM. M4 Max bf16 peak is ~17 TFLOPs (Apple's published headline
  spec). MLX delivers ~86 % of peak. **There is no slack.** The kernel
  is already saturating the matrix-multiply pipeline.
- Quantized matmul has to dequantize per group on the fly. At small
  M this trades a little compute for a big bandwidth save (worth it).
  At large M the GEMM is compute-bound; the dequant overhead is pure
  cost, no recovery.
- This matches Apple's own messaging: M5's Neural Accelerators are the
  pathway for "free" matmul speedups (3–5×). M4 has no such hardware.
- The current `affine int4` Q4 model exists for **memory savings**
  (transformer 11 GB instead of ~22 GB bf16), not speed. That's still
  the right design choice for users with 24–48 GB Macs; it's the
  reason the model fits at all on Compact / Comfortable tiers.

---

## What we ALSO didn't gate on

The brief's escalation gate was "≥ 10 % microbench speedup → run one-block →
real render". The microbench shows the **opposite sign** at every
configuration we care about. So none of the gated next steps ran:

- ❌ One-block bench at 640 × 480 / 1024 × 576 / 1280 × 704 / 1024 × 576 241f
- ❌ Copied `mlx_models/ltx-2.3-mlx-q4-mxfp4-test` directory + transformer
  reconvert
- ❌ 512 × 288 / 25-frame smoke render
- ❌ Real render benchmarks (Balanced 1024 × 576 121f Exact/Turbo,
  Standard 1280 × 704 121f Exact/Turbo, 20 s 481f Turbo)

This is a feature, not a bug. The hard-stop in the brief is exactly to
prevent burning M4 Max time on a hypothesis the microbench has already
killed.

---

## Recommended next action

**3. Abandon** the MLX quant-mode lane.

MLX 0.31.1 supports `mxfp4` / `mxfp8` / `nvfp4` cleanly, but on M4 Max
they cannot beat `affine int4 g64` on LTX's compute-bound denoise loop.
The system is already at ~86 % of theoretical peak. There is nothing to
recover here without changing the hardware (M5 + Neural Accelerators)
or the algorithm (token merging, fewer-step distillation), as already
documented in `docs/PERF_RESEARCH_2026-05-05.md`.

### Adjacent finding worth flagging (NOT pursued in this session)

The data shows **bf16 transformer is 7–15 % faster than the current Q4
on M4 Max** at LTX denoise scale. The Q4 trade is "12 % slower per step,
half the weight footprint". On a 64 GB Mac Studio with audio + VAE +
encoder also resident, bf16 transformer (~22 GB instead of ~11 GB) might
fit and produce a real wall-time win for users on the Roomy / Studio
tier.

This is a separate hypothesis, requires:

- Confirm bf16 transformer fits in memory alongside Gemma encoder,
  VAE encoder, VAE decoder, audio VAE, vocoder during a render.
- Tier-gate it (Comfortable 64 GB and below probably can't afford
  the headroom; Roomy/Studio likely can).
- A/B render to confirm denoise wall-time ratio matches the
  microbench's 0.85–0.93×.

I am explicitly **not** doing this in this session because the brief
asked specifically about quantization modes. Worth a separate scoped
project if Salo wants to pursue it; pasting this section into a fresh
session prompt should be enough context.

---

## Files in this branch

- `scripts/perf_lab/01_quant_modes_probe.py` — verify MLX 0.31.1 quant API
- `scripts/perf_lab/02_microbench_linear.py` — full sweep, the main result
- `scripts/perf_lab/03_validate_bandwidth_regime.py` — small-M sanity check
- `docs/PERF_LAB_MLX_QUANT.md` — this doc
- `/tmp/phos_perf_lab/microbench.json` — full bench output (108 rows)

### Reproduction

```bash
cd /Users/salo/pinokio/api/phosphene-dev.git
git checkout perf-lab-mlx-quant
ltx-2-mlx/env/bin/python scripts/perf_lab/01_quant_modes_probe.py
ltx-2-mlx/env/bin/python scripts/perf_lab/02_microbench_linear.py \
    --warmup 3 --runs 30 --json /tmp/phos_perf_lab/microbench.json
ltx-2-mlx/env/bin/python scripts/perf_lab/03_validate_bandwidth_regime.py
```

Expected runtime: ~3 minutes total on a quiet M4 Max.

### Production safety

This branch:

- Does **not** touch `mlx_ltx_panel.py`, `mlx_warm_helper.py`,
  `patch_ltx_codec.py`, or anything in `ltx-2-mlx/`.
- Does **not** create or alter any model directory under `mlx_models/`.
- Does **not** modify production app behavior in any way.
- Has **not** been pushed. Awaiting Salo's call on what to do with it.
