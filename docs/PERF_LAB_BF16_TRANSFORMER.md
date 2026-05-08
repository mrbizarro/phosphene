# Perf lab — BF16 transformer vs current Q4

**Date:** 2026-05-07 · **Branch:** `perf-lab-bf16-transformer` · **Hardware:** M4 Max 64 GB
**Status:** Hypothesis refuted on 64 GB. **Recommendation: ABANDON for 64 GB. Lab-only / 96+ GB-only for further work.**

## TL;DR

The hypothesis was: since MLX's bf16 GEMM kernel is highly optimized and the
[PERF_LAB_MLX_QUANT.md](https://github.com/mrbizarro/phosphene/blob/perf-lab-mlx-quant/docs/PERF_LAB_MLX_QUANT.md) microbench showed bf16 is
7–15 % faster than `affine_int4` on real LTX shapes, swapping in a bf16
transformer should reduce wall time on Macs with enough memory (64 / 96 / 128 GB).

On a 64 GB M4 Max it does the opposite:

- **bf16 denoise is 2.33× SLOWER than Q4** on a 25-frame 512×288 tiny render
  (35.3 s vs 15.2 s). End-to-end is 1.94× slower (44.5 s vs 23.0 s).
- **Peak GPU memory hits 50.5 GiB** vs Q4's 25.7 GiB — within 14 GiB of
  64 GB ceiling, before activations / connector / Gemma / VAE / system.
- **Swap rose past the 8 GB safety threshold** (peak 8.13 GB) during the bf16
  run.
- The 38 GB transformer is mmap-loaded from disk; on a tight memory machine
  pages fault in *during* denoise, mixing weight reads with compute. That's
  what the microbench (which had everything pre-resident) couldn't predict.

The brief's hard-stop fires on **two** of four conditions: swap > 8 GB AND
bf16 slower in real denoise. **Stopping. Not running real benchmarks. Not
integrating.**

For 96 / 128 GB Macs the math suggests bf16 *might* fit without swap, in which
case the microbench prediction (~12 % faster denoise) would likely hold — but
we don't have that hardware here, so this remains untested.

---

## Source weights

| field | value |
|---|---|
| repo | `dgrauet/ltx-2.3-mlx` (the unquantized companion to `dgrauet/ltx-2.3-mlx-q4`) |
| file | `transformer-distilled.safetensors` |
| size on disk | 35.38 GiB / 38.0 GB |
| source | bf16 MLX-format port of `Lightricks/LTX-2.3/ltx-2.3-22b-distilled.safetensors` |
| license | LTX-2 Community License Agreement (same as the Q4 weights already in production) |
| commit sha | `baa5f235ea04fd9c95899d751295c4fd825ee4e2` |

We deliberately picked the **same distillation** as our Q4 baseline (no
`-1.1` suffix) to isolate the bf16-vs-Q4 question from the
1.1-vs-original-distillation question. If 1.1-distilled testing is wanted,
that's a separate download (also 38 GB) and a separate test against an
appropriate Q4 baseline.

The bf16 repo's non-transformer files (connector, vae_*, audio_vae,
vocoder, upscalers, configs) are byte-identical (same etag/sha256) to the
Q4 repo's, so we symlinked them from the existing local Q4 dir at
`mlx_models/ltx-2.3-mlx-q4/` rather than re-downloading. Only the 38 GB
transformer was fetched.

Test model dir layout:

```
mlx_models/ltx-2.3-mlx-bf16-test/
├── transformer-distilled.safetensors  -> ~/.cache/huggingface/hub/.../bd5c...  (38 GB bf16)
├── connector.safetensors              -> ../ltx-2.3-mlx-q4/connector.safetensors
├── vae_encoder.safetensors            -> ../ltx-2.3-mlx-q4/vae_encoder.safetensors
├── vae_decoder.safetensors            -> ../ltx-2.3-mlx-q4/vae_decoder.safetensors
├── audio_vae.safetensors              -> ../ltx-2.3-mlx-q4/audio_vae.safetensors
├── vocoder.safetensors                -> ../ltx-2.3-mlx-q4/vocoder.safetensors
├── spatial_upscaler_*.safetensors     -> ../ltx-2.3-mlx-q4/...
├── temporal_upscaler_*.safetensors    -> ../ltx-2.3-mlx-q4/...
├── ltx-2.3-22b-distilled-lora-384.safetensors -> ../ltx-2.3-mlx-q4/...
├── config.json                        -> ../ltx-2.3-mlx-q4/config.json
├── embedded_config.json               -> ../ltx-2.3-mlx-q4/embedded_config.json
└── split_model.json                   (rewritten — no quantize_config; lab-only marker)
```

## Loader smoke

`scripts/perf_lab/04_loader_smoke.py` directly probes the bf16 safetensors:

```
total keys:    4186
weight keys:   2236
.scales keys:  0   (>0 => quantized)
.biases keys:  0   (quant biases)
dtypes:        {bfloat16: 3896, float32: 290}
sample: transformer.transformer_blocks.0.attn1.to_q.weight  shape=(4096, 4096)  dtype=bfloat16
```

The 290 float32 keys are AdaLN `scale_shift_table` entries, RMSNorm gains,
and time-embedding params — kept fp32 for numerical stability per LTX
convention. The other 3 896 are the actual transformer weights, all bf16.

The pipeline-side loader (`ltx_core_mlx.utils.weights.apply_quantization`)
is gated on the presence of `.scales` keys, so on bf16 it's a no-op. No
code change needed in the pipeline to handle this dir.

## Tiny render: BF16 vs Q4

Same prompt, same seed, same shape, same render code (`scripts/perf_lab/05_render_bench.py`).
Prompt: *"A serene mountain valley at golden hour, soft cinematic light, gentle wind in the grass."*
Shape: 512×288, 25 frames, 8 steps, seed=12345. Both runs after Gemma was
HF-cached (no first-time download cost in the timings shown).

| phase | Q4 | BF16 | bf16/Q4 |
|---|---:|---:|---:|
| pipe_init | 0.03 ms | 0.03 ms | — |
| text_encode (Gemma + connector) | 6 317 ms | 5 137 ms | 0.81× |
| dit_load (mmap, deferred fault-in) | 0 ms | 0 ms | — |
| denoise_setup | 5.7 ms | 6.9 ms | — |
| **denoise** | **15 160 ms** | **35 324 ms** | **2.33×** |
| unpatchify | 0.1 ms | 0.1 ms | — |
| decode_and_save (VAE+audio+ffmpeg) | 1 142 ms | 3 678 ms | 3.22× |
| **total** | **22 967 ms** | **44 481 ms** | **1.94×** |

Per-step denoise: Q4 = 1.90 s/step, BF16 = 4.42 s/step. Both renders
produced valid 1-sec H.264 yuv444p video + 48 kHz AAC audio at 24 frames
(LTX drops the trailing frame).

### Memory and pressure during the bf16 run

| measurement | Q4 | BF16 |
|---|---:|---:|
| peak GPU memory (mx.metal.get_peak_memory) | 25.66 GiB | **50.49 GiB** |
| peak RSS during denoise | 22.41 GiB | 15.96 GiB |
| peak macOS memory pressure | 11 % | **85 %** |
| peak swap during run | varies* | **8.13 GB** ← tripped 8 GB hard-stop |

\* Q4 swap reading is contaminated by the prior bf16 run's leftover swap
(macOS doesn't aggressively shrink swap). The bf16 *delta* is what matters
— pressure went from steady-state to 85 % during its denoise.

The bf16 run's RSS during denoise is *lower* than its GPU peak, which seems
contradictory but isn't: `mx.load(...)` mmaps the safetensors and pages
are faulted in by access. The OS keeps actively-touched pages resident
(15.96 GiB worth), but the full 38 GB working set is only "reachable",
not committed. Under memory pressure the OS evicts and re-faults, mixing
disk I/O into the denoise critical path. **That's the perf killer.**

## Why bf16 is slower in real denoise (despite the microbench)

The [PERF_LAB_MLX_QUANT.md](https://github.com/mrbizarro/phosphene/blob/perf-lab-mlx-quant/docs/PERF_LAB_MLX_QUANT.md) microbench had the
weights pre-resident in MLX device memory and just timed the GEMM. In
that idealized regime, bf16 GEMM hits 14.6 TFLOPs (~86 % of M4 Max peak)
and beats every quant mode by 7–15 % on LTX shapes.

The real pipeline doesn't reproduce that regime on 64 GB:

1. **Memory-mapped weights, not pre-loaded.** `mx.load(...)` returns
   mmapped tensors. Pages are read from disk on first access. Q4's 11 GiB
   working set fits in residency cache after the first step; bf16's 38 GiB
   working set doesn't.
2. **System contention.** During denoise, Q4 has Gemma freed (~7 GiB),
   connector resident (~6 GiB), DiT resident (~11 GiB), VAE+audio not
   yet loaded → ~25 GiB live. With BF16, DiT alone is 38 GiB, plus the
   same connector and Gemma residue → 50+ GiB, leaving the OS roughly
   no room. macOS starts paging.
3. **Cascade into VAE decode.** Even after denoise, the bf16 transformer
   stays mmapped. VAE decode wants its own ~1 GiB. There's not enough
   resident slack, so VAE decode also hits I/O — that's what produces the
   3.2× slowdown on `decode_and_save`.

The microbench is correct about the GEMM kernel. The pipeline just can't
keep the weights resident enough to use it.

## What we did NOT run, and why

The brief's gate was "tiny render succeeds → run real benchmarks". Tiny
render *technically* succeeded (valid output), but the hard-stop conditions
fired:

- **Hard stop: swap > 8 GB** → tripped (8.13 GB) ✓
- **Hard stop: bf16 slower in real denoise** → tripped (2.33× slower) ✓

So we **did not run**:

- ❌ Quick 640×480 121f Exact
- ❌ Balanced 1024×576 121f Exact / Turbo
- ❌ Standard 1280×704 121f Exact / Turbo
- ❌ 20 s Balanced 1024×576 481f Turbo

A 121-frame render's working set is much larger than a 25-frame one, so
the swap pressure can only get worse. There is no realistic configuration
on a 64 GB Mac where bf16 wins on denoise wall time.

## 96 / 128 GB tier — untested

With 96 GB or 128 GB:

- 38 GB DiT + 13 GB Gemma+connector + 5 GB activations + 6 GB system =
  ~62 GB peak. **Fits with margin** on 96 GB; **trivially fits** on 128 GB.
- No swap → no I/O fault-in → microbench prediction (~12 % faster denoise)
  *should* hold.
- Multi-job throughput would be worse though: bf16 transformer evicts more
  of the OS's file cache, hurting other apps.

We don't have access to a 96 GB or 128 GB Mac in this lab session, so
this is **theoretical**. Anyone who wants to verify can run:

```bash
PATH=/Users/salo/pinokio/bin/ffmpeg-env/bin:$PATH \
ltx-2-mlx/env/bin/python scripts/perf_lab/05_render_bench.py \
  --model mlx_models/ltx-2.3-mlx-bf16-test \
  --width 1024 --height 576 --frames 121 --steps 8 --seed 12345 \
  --label bf16_balanced_1024x576_121f
```

on a 96+ GB machine. If denoise wall is < Q4's, integrate. If swap watcher
trips, abandon there too.

## Recommendation

**Abandon** for 64 GB Macs (Comfortable tier and below). The bf16
transformer is unworkable on this memory budget — it's 2× slower in real
denoise and pushes the OS into swap.

**Don't integrate as a Phosphene mode** without 96+ GB-tier verification
data. Even if it works on Roomy/Studio, only ~5 % of Phosphene users have
those machines (Pinokio analytics — to confirm with Salo). The integration
cost (settings UI, model dir handling, tier gating, switching logic) is
real, and the user-facing payoff is small. Keep the bench scripts and this
doc as evidence; don't ship the model dir.

If a Studio user explicitly asks: the test model dir
(`mlx_models/ltx-2.3-mlx-bf16-test`) and bench script let them try it
in five minutes. That's already enough; no production wiring needed.

### Adjacent finding worth remembering

The [PERF_LAB_MLX_QUANT.md](https://github.com/mrbizarro/phosphene/blob/perf-lab-mlx-quant/docs/PERF_LAB_MLX_QUANT.md) microbench predicted
bf16 would be 7–15 % faster than Q4 *per Linear*. This experiment shows
that holds **only** when the weights are fully resident. The lesson is
generalizable to any future "lower precision = faster?" experiment on
this hardware: **always include a pipeline-level test on the actual target
memory tier**, because microbench results don't survive memory pressure.

## Files in this branch

- `scripts/perf_lab/04_loader_smoke.py` — direct safetensors probe + pipeline load smoke
- `scripts/perf_lab/05_render_bench.py` — phase-timed render driver with safety watcher
- `mlx_models/ltx-2.3-mlx-bf16-test/` — test model dir (38 GB transformer + symlinks)
- `docs/PERF_LAB_BF16_TRANSFORMER.md` — this doc
- `/tmp/phos_perf_lab/renders/q4_tiny_512x288_25f.{mp4,json}` — Q4 baseline
- `/tmp/phos_perf_lab/renders/bf16_tiny_512x288_25f.{mp4,json}` — BF16 result

### Reproduction

```bash
cd /Users/salo/pinokio/api/phosphene-dev.git
git checkout perf-lab-bf16-transformer

# (one-time) download bf16 transformer (38 GB)
HF_HUB_ENABLE_HF_TRANSFER=1 ltx-2-mlx/env/bin/python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('dgrauet/ltx-2.3-mlx', 'transformer-distilled.safetensors')
"
ln -s ~/.cache/huggingface/hub/models--dgrauet--ltx-2.3-mlx/snapshots/baa5f235ea04fd9c95899d751295c4fd825ee4e2/transformer-distilled.safetensors \
  mlx_models/ltx-2.3-mlx-bf16-test/transformer-distilled.safetensors

# loader smoke
ltx-2-mlx/env/bin/python scripts/perf_lab/04_loader_smoke.py --model-dir mlx_models/ltx-2.3-mlx-bf16-test

# tiny renders
PATH=/Users/salo/pinokio/bin/ffmpeg-env/bin:$PATH \
  ltx-2-mlx/env/bin/python scripts/perf_lab/05_render_bench.py \
  --model mlx_models/ltx-2.3-mlx-q4 \
  --width 512 --height 288 --frames 25 --steps 8 --seed 12345 --label q4_tiny_512x288_25f

PATH=/Users/salo/pinokio/bin/ffmpeg-env/bin:$PATH \
  ltx-2-mlx/env/bin/python scripts/perf_lab/05_render_bench.py \
  --model mlx_models/ltx-2.3-mlx-bf16-test \
  --width 512 --height 288 --frames 25 --steps 8 --seed 12345 --label bf16_tiny_512x288_25f
```

### Production safety

This branch:

- Does **not** modify `mlx_ltx_panel.py`, `mlx_warm_helper.py`,
  `patch_ltx_codec.py`, or anything in `ltx-2-mlx/`.
- Does **not** alter any existing `mlx_models/` directory.
- Adds `mlx_models/ltx-2.3-mlx-bf16-test/` (additive, lab-only). The
  helper's default model is still `dgrauet/ltx-2.3-mlx-q4`. Nothing in
  the panel will pick up the test dir unless `LTX_MODEL` is overridden
  in the env.
- Has **not** been pushed. Awaiting Salo's call on what to do with it.

To clean up the bf16 test model dir at any time:
```bash
rm -rf mlx_models/ltx-2.3-mlx-bf16-test/
# To free the 38 GB cached download:
rm -rf ~/.cache/huggingface/hub/models--dgrauet--ltx-2.3-mlx/
```
