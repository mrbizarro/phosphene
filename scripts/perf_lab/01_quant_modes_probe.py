"""Probe MLX quantization API surface.

Reports installed mlx version, supported quantize modes, and dispatches a
trivial round-trip quantize -> dequantize -> matmul on each mode to confirm
the kernels actually exist for our build.

Run from repo root:
    ltx-2-mlx/env/bin/python scripts/perf_lab/01_quant_modes_probe.py
"""

from __future__ import annotations
import sys
import inspect
from importlib.metadata import version as pkg_version

import mlx.core as mx
import mlx.nn as nn


def banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> int:
    banner("MLX build")
    print(f"mlx wheel version: {pkg_version('mlx')}")
    print(f"metal available  : {mx.metal.is_available()}")
    print(f"default device   : {mx.default_device()}")

    banner("API signatures")
    print("nn.quantize       :", inspect.signature(nn.quantize))
    print("nn.QuantizedLinear:", inspect.signature(nn.QuantizedLinear.__init__))

    # Pull supported modes out of the docstring (declarative source of truth).
    doc = mx.quantize.__doc__ or ""
    declared_modes = [m for m in ("affine", "mxfp4", "mxfp8", "nvfp4")
                      if f'"{m}"' in doc]
    banner("Modes declared by mx.quantize docstring")
    print(declared_modes)

    # Round-trip each mode on a small bf16 weight to confirm a kernel runs.
    banner("Mode round-trip (quantize -> quantized_matmul -> eval)")
    test_modes = [
        ("affine", 64, 4),
        ("affine", 64, 8),
        ("mxfp4", 32, 4),
        ("mxfp8", 32, 8),
        ("nvfp4", 16, 4),
    ]
    out_dims, in_dims = 4096, 4096  # representative LTX projection
    M = 256
    rng = mx.random.uniform(shape=(out_dims, in_dims), dtype=mx.bfloat16)
    x = mx.random.normal(shape=(M, in_dims), dtype=mx.bfloat16)
    mx.eval(rng, x)

    for mode, gs, bits in test_modes:
        try:
            res = mx.quantize(rng, group_size=gs, bits=bits, mode=mode)
            if mode == "affine":
                wq, scales, biases = res
            else:
                wq, scales = res[0], res[1]
                biases = None
            mx.eval(wq, scales)
            if biases is not None:
                mx.eval(biases)
            y = mx.quantized_matmul(
                x, wq, scales=scales, biases=biases,
                transpose=True, group_size=gs, bits=bits, mode=mode,
            )
            mx.eval(y)
            mem_bytes = wq.nbytes + scales.nbytes + (biases.nbytes if biases is not None else 0)
            print(f"  {mode:7s} gs={gs:3d} bits={bits} OK   "
                  f"out={tuple(y.shape)} dtype={y.dtype} "
                  f"weight_bytes={mem_bytes/1024:.1f} KiB")
        except Exception as e:
            print(f"  {mode:7s} gs={gs:3d} bits={bits} FAIL  {type(e).__name__}: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
