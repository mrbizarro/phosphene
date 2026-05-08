"""Sanity check: does the same bench infrastructure show quant winning at
small M (bandwidth-bound regime, like LLM decoding)?

If quant beats bf16 at M=1 / M=8 but not at M=4800+, that confirms the
microbench is correct and LTX's denoise is in the compute-bound regime
where quant cannot win on M4 Max.

Run:
    ltx-2-mlx/env/bin/python scripts/perf_lab/03_validate_bandwidth_regime.py
"""

from __future__ import annotations
import time
from statistics import median

import mlx.core as mx


SHAPES = [
    ("video_qkvo_4096x4096",   4096, 4096),
    ("video_ff_in_4096x16384", 16384, 4096),
]
SMALL_M = [1, 8, 64, 512]
MODES = [
    ("bf16",            None,    None, None),
    ("affine_int4_g64", "affine",  64,    4),
    ("mxfp4_g32",       "mxfp4",   32,    4),
    ("nvfp4_g16",       "nvfp4",   16,    4),
]


def time_ms(fn, runs=200):
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return median(times)


def main():
    mx.random.seed(42)
    print(f"{'shape':24s} {'M':>5s}  {'mode':18s} {'ms':>9s}  {'speedup':>8s}")
    for sname, out_dim, in_dim in SHAPES:
        w = mx.random.normal(shape=(out_dim, in_dim), dtype=mx.bfloat16) * 0.02
        mx.eval(w)
        for M in SMALL_M:
            x = mx.random.normal(shape=(M, in_dim), dtype=mx.bfloat16) * 0.5
            mx.eval(x)
            base = None
            for label, mode, gs, bits in MODES:
                if mode is None:
                    def fn(_w=w, _x=x):
                        y = mx.matmul(_x, _w.T); mx.eval(y)
                else:
                    res = mx.quantize(w, group_size=gs, bits=bits, mode=mode)
                    if mode == "affine":
                        wq, scales, biases = res
                        mx.eval(wq, scales, biases)
                    else:
                        wq, scales = res[0], res[1]
                        biases = None
                        mx.eval(wq, scales)
                    def fn(_x=x, _wq=wq, _s=scales, _b=biases,
                           _gs=gs, _bits=bits, _mode=mode):
                        y = mx.quantized_matmul(
                            _x, _wq, scales=_s, biases=_b,
                            transpose=True, group_size=_gs, bits=_bits, mode=_mode,
                        ); mx.eval(y)
                # warmup
                for _ in range(5):
                    fn()
                t = time_ms(fn, runs=200)
                if label == "bf16":
                    base = t
                    spd = "1.00x"
                else:
                    spd = f"{base/t:.2f}x"
                print(f"{sname:24s} {M:>5d}  {label:18s} {t:>9.4f}  {spd:>8s}")
        print()


if __name__ == "__main__":
    main()
