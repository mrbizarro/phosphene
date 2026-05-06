# Conv3d kernel port — design doc (Track B)

## Problem

MLX's built-in `nn.Conv3d` runs at **1.7-2.8 GB/s effective** on the LTX VAE
upsampler shapes (M4 Max bf16). M4 Max's bandwidth ceiling is **~410 GB/s**,
its theoretical bf16 compute is **~10-15 TFLOPs**. That's ~150× below peak.

The biggest 3D convs in the LTX VAE decoder (LatentUpsampler, 512→2048
channels, kernel=3, padding=1):

| Shape | C_in→C_out | MLX time | Effective bandwidth |
|---|---|---|---|
| 5s `D=16 H=18 W=32` | 512→2048 | 36.9 ms | 2.8 GB/s |
| 10s `D=31 H=18 W=32` | 512→2048 | 69.6 ms | 2.1 GB/s |
| 20s `D=61 H=18 W=32` | 512→2048 | 136.3 ms | 1.7 GB/s |

Source: [/tmp/phosphene_bench/bench_conv3d.out](file:///tmp/phosphene_bench/bench_conv3d.out)

Draw Things ([2-Days-to-Ship blog post](https://engineering.drawthings.ai/p/2-days-to-ship-codex-authored-metal))
shipped a Codex-authored 3D conv kernel that gives **2.4× on M1-M4** and
4.7× on M5 specifically for LTX-2 VAE decoding. Their kernel
([`ccv_nnc_mfa_conv3d.{cpp,hpp}`](https://github.com/liuliu/ccv/tree/unstable/lib/nnc/mfa/v2))
is BSD-3 licensed (compatible with our MIT).

End-to-end win envelope: **~10-12% on a 10-min render** (saves ~50-70 s).
Smaller than block-skip's ~25-30% but additive — and quality-risk-free
(kernel math is bit-equivalent).

## Two viable port paths

### Path 1 — Vendor MFA's full kernel via a Python C++ extension

- Pull `ccv_nnc_mfa_conv3d.{cpp,hpp}` + minimum supporting headers (mfa.h,
  defines, error.cpp) from `liuliu/ccv` at a pinned commit.
- Write `setup.py` / `pyproject.toml` to build the extension via setuptools
  (or scikit-build).
- Need to bridge to MLX's `mx.array` storage so we can read/write tensors.
  MLX exposes `mx.array.data_ptr()` for raw buffer access (Metal MTLBuffer).
- Replace `nn.Conv3d` calls in [`video_vae/convolution.py`](../ltx-2-mlx/packages/ltx-core-mlx/src/ltx_core_mlx/model/video_vae/convolution.py)
  with our wrapper.

**Effort:** 2-3 weeks. Risk: MLX↔MFA buffer interop nontrivial; CMake/setuptools-via-Metal
build is fragile.

### Path 2 (RECOMMENDED) — Custom MSL kernel via `mx.fast.metal_kernel`

MLX has [`mx.fast.metal_kernel(name, input_names, output_names, source)`](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)
which JIT-compiles a Metal shading language source string.

- Write our own `.metal` source for 3D convolution (~150-300 lines MSL).
- Reference MFA's algorithm (BSD-3, attribution required) for tile sizes and
  layout choices, but write fresh code that calls into MLX's kernel API.
- Wrap as a Python function `fast_conv3d_mlx(x, weight, bias, padding, ...)` →
  `mx.array`.
- Patch [`video_vae/convolution.py`](../ltx-2-mlx/packages/ltx-core-mlx/src/ltx_core_mlx/model/video_vae/convolution.py)
  Conv3dBlock to call our function when shape is in the optimized set.

**Effort:** 1-2 weeks. Risk: getting tile sizes right for the LTX shapes (D=16-61, H=18-72).
A naive im2col + GEMM fallback is reachable in ~2 days; matching MFA's 2.4× takes more
care.

## Implementation outline (Path 2)

### Phase 1: scaffold + naive kernel — ~3 days

```
patch_conv3d_kernel.py  ← runtime patch (same style as patch_block_skip)
    └─ fast_conv3d(x, weight, bias, k, stride, padding) -> mx.array
       └─ uses mx.fast.metal_kernel("conv3d_naive", source=NAIVE_MSL)
```

`NAIVE_MSL` does the basic gather: each output element = sum of K_d × K_h × K_w
input elements × weight. Single thread per output, no tiling. **Goal:
correctness, not speed.** Verify bit-equivalence vs `nn.Conv3d` on representative
shapes.

### Phase 2: tiled MSL kernel matching MFA performance — ~5-7 days

Replace `NAIVE_MSL` with a tiled version:
- 2D thread groups in (D, H) or (H, W) plane
- Cooperatively load input tiles into threadgroup memory
- Each thread computes a 4-8-element output stripe
- Use `simd_shuffle` for cross-lane reductions
- For C_in=512 paths, use 32-element vec loads (`half4` × 8 elements)

Reference: MFA's `ccv_nnc_mfa_conv3d.cpp` for tile-size heuristics. Specifically,
their kernel uses a `KernelDescriptor` that picks tile dims based on (input_dims,
output_channels, M-series gen). We replicate that table for M3/M4/M5.

### Phase 3: integration + benchmark — ~2 days

- Patch [`video_vae/convolution.py:Conv3dBlock`](../ltx-2-mlx/packages/ltx-core-mlx/src/ltx_core_mlx/model/video_vae/convolution.py)
  to call `fast_conv3d` when shape is supported, else fall back to `self.conv(x)`.
- Add env-var gate `LTX_FAST_CONV3D=1` for opt-in (same pattern as block-skip).
- Bench: VAE decode time on 5s/10s/20s clips, with and without.
- A/B render to confirm bit-equivalence (or near-bit-equivalent at fp16).

## Risk register

| Risk | Mitigation |
|---|---|
| MLX `mx.fast.metal_kernel` doesn't expose enough Metal features (e.g. simdgroup ops) | Test early in Phase 1 with a simple MSL kernel; if missing, fall back to Path 1 |
| Apple ships an improved conv3d in a near-future MLX release | Re-evaluate before Phase 2; if Apple's gain ≥ ours, abandon |
| Numerical drift breaks audio quality on long clips | Bench audio RMS + spectrum after every kernel iteration |
| Edge cases (padding, stride>1, dilation) explode kernel complexity | Initial scope: stride=1, dilation=1, padding=K//2 only — covers all LTX VAE convs |

## Where to look in our code

- VAE conv layers: [`packages/ltx-core-mlx/src/ltx_core_mlx/model/video_vae/convolution.py`](../ltx-2-mlx/packages/ltx-core-mlx/src/ltx_core_mlx/model/video_vae/convolution.py)
- Upsampler: [`packages/ltx-core-mlx/src/ltx_core_mlx/model/upsampler/model.py`](../ltx-2-mlx/packages/ltx-core-mlx/src/ltx_core_mlx/model/upsampler/model.py)
- ResNet blocks: [`packages/ltx-core-mlx/src/ltx_core_mlx/model/video_vae/resnet.py`](../ltx-2-mlx/packages/ltx-core-mlx/src/ltx_core_mlx/model/video_vae/resnet.py)
- Streaming decode patches: [`patch_ltx_codec.py`](../patch_ltx_codec.py) (Patch 5 — VAE temporal streaming; we'd add Patch 6 for fast conv3d)

## Decision: defer Track B until Track A is shipped

Track A (block-skip) gives ~3× more end-to-end speedup than Track B for ~½ the
implementation effort. Sequence Track A first, then revisit Track B once
A is live and quality-validated.
