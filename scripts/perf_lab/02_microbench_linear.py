"""Microbenchmark a single Linear layer across MLX quant modes.

Sweeps real LTX-2.3 video / audio transformer GEMM shapes against a sweep of
token counts that match our common render resolutions. For each (shape, M,
mode) triple it reports cold time, warm median over 50 runs, and max-abs
output error vs the bf16 baseline.

Run:
    ltx-2-mlx/env/bin/python scripts/perf_lab/02_microbench_linear.py \\
        --warmup 5 --runs 50 --json /tmp/phos_perf_lab/microbench.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from statistics import median
from typing import Optional

import mlx.core as mx


# ---- LTX-2.3 transformer Linear shapes (out_dim, in_dim) -------------------
#
# Source: ltx-core-mlx model config + transformer/attention.py + feed_forward.py
# - num_layers=48, video_dim=4096, video_head_dim=128, num_heads=32
# - audio_dim=2048, audio_head_dim=64, audio_num_heads=32
# - cross_attention_dim=4096, audio_cross_attention_dim=2048
# - ff_mult=4.0
# Each video block has 4 self-attn + 4 text-cross + 4 AV-cross + 2 ff linears
# of these shapes (plus tiny gate-logits 4096->32 we omit — bandwidth-irrelevant).
SHAPES: list[tuple[str, int, int]] = [
    # name, out_dim, in_dim
    ("video_qkvo_4096x4096",   4096, 4096),
    ("video_ff_in_4096x16384", 16384, 4096),
    ("video_ff_out_16384x4096", 4096, 16384),
    ("audio_qkvo_2048x2048",   2048, 2048),
    ("audio_ff_in_2048x8192",  8192, 2048),
    ("audio_ff_out_8192x2048", 2048, 8192),
]

# Token-count sweep (M dimension of GEMM x @ w.T). Approximate values
# derived from LTX latent (1+(F-1)/8) frames * (H/32) * (W/32) tokens —
# patchifier is identity since patch_size_t=h=w=1.
TOKEN_COUNTS: list[tuple[str, int]] = [
    ("640x480_121f",    4800),    # 16 * 20 * 15
    ("1024x576_121f",   9216),    # 16 * 32 * 18
    ("1280x704_121f",  14080),    # 16 * 40 * 22
    ("1024x576_241f",  17856),    # 31 * 32 * 18
    ("audio_short",      256),    # representative audio token len, short clip
    ("audio_long",      1024),    # longer clip
]

# Quantization modes to test. (label, mode, group_size, bits)
MODES = [
    ("bf16",            None,    None, None),
    ("affine_int4_g64", "affine",  64,    4),
    ("affine_int4_g32", "affine",  32,    4),
    ("mxfp4_g32",       "mxfp4",   32,    4),
    ("mxfp8_g32",       "mxfp8",   32,    8),
    ("nvfp4_g16",       "nvfp4",   16,    4),
]


@dataclass
class Row:
    shape_name: str
    M: int
    M_label: str
    out_dim: int
    in_dim: int
    mode_label: str
    mode: Optional[str]
    group_size: Optional[int]
    bits: Optional[int]
    weight_bytes: int
    cold_ms: float
    warm_median_ms: float
    warm_min_ms: float
    warm_max_ms: float
    tokens_per_s: float
    max_abs_err_vs_bf16: Optional[float]


def time_call(fn, runs: int) -> list[float]:
    """Time a callable that internally evals its result. Returns ms list."""
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return times


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--json", type=str, default=None,
                    help="path to dump full results JSON")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shapes", type=str, default="all",
                    help="comma list of shape names, or 'all'")
    ap.add_argument("--tokens", type=str, default="all",
                    help="comma list of token labels, or 'all'")
    args = ap.parse_args()

    mx.random.seed(args.seed)

    shapes = (SHAPES if args.shapes == "all"
              else [s for s in SHAPES if s[0] in args.shapes.split(",")])
    tokens = (TOKEN_COUNTS if args.tokens == "all"
              else [t for t in TOKEN_COUNTS if t[0] in args.tokens.split(",")])

    rows: list[Row] = []

    for shape_name, out_dim, in_dim in shapes:
        # Audio shapes only meaningful at audio_* token counts; video shapes
        # only meaningful at video token counts. Filter to keep noise down.
        is_audio_shape = shape_name.startswith("audio_")
        for tlabel, M in tokens:
            is_audio_token = tlabel.startswith("audio_")
            if is_audio_shape != is_audio_token:
                continue

            print(f"\n--- shape={shape_name} M={M} ({tlabel}) ---")
            w_bf16 = mx.random.normal(shape=(out_dim, in_dim), dtype=mx.bfloat16) * 0.02
            x = mx.random.normal(shape=(M, in_dim), dtype=mx.bfloat16) * 0.5
            mx.eval(w_bf16, x)

            # Build a bf16 reference output for error measurements.
            y_ref = mx.matmul(x, w_bf16.T)
            mx.eval(y_ref)

            for label, mode, gs, bits in MODES:
                if mode is None:
                    # bf16 baseline matmul
                    def call_bf16(_w=w_bf16, _x=x):
                        y = mx.matmul(_x, _w.T)
                        mx.eval(y)
                    fn = call_bf16
                    weight_bytes = w_bf16.nbytes
                    err_max = 0.0
                else:
                    res = mx.quantize(w_bf16, group_size=gs, bits=bits, mode=mode)
                    if mode == "affine":
                        wq, scales, biases = res
                        mx.eval(wq, scales, biases)
                        weight_bytes = wq.nbytes + scales.nbytes + biases.nbytes
                    else:
                        wq, scales = res[0], res[1]
                        biases = None
                        mx.eval(wq, scales)
                        weight_bytes = wq.nbytes + scales.nbytes

                    # Correctness check: compute once, measure max abs error.
                    y_q = mx.quantized_matmul(
                        x, wq, scales=scales, biases=biases,
                        transpose=True, group_size=gs, bits=bits, mode=mode,
                    )
                    mx.eval(y_q)
                    err_max = float(mx.max(mx.abs(y_q - y_ref)).item())

                    def call_quant(_x=x, _wq=wq, _s=scales, _b=biases,
                                   _gs=gs, _bits=bits, _mode=mode):
                        y = mx.quantized_matmul(
                            _x, _wq, scales=_s, biases=_b,
                            transpose=True, group_size=_gs, bits=_bits,
                            mode=_mode,
                        )
                        mx.eval(y)
                    fn = call_quant

                # Cold timing: do one call after a small synchronize to
                # flush prior state.
                mx.synchronize()
                t0 = time.perf_counter()
                fn()
                cold_ms = (time.perf_counter() - t0) * 1000.0

                # Warm: warmup + runs
                for _ in range(args.warmup):
                    fn()
                times = time_call(fn, args.runs)

                warm_med = median(times)
                warm_min = min(times)
                warm_max = max(times)
                tps = M / (warm_med / 1000.0)

                rows.append(Row(
                    shape_name=shape_name, M=M, M_label=tlabel,
                    out_dim=out_dim, in_dim=in_dim,
                    mode_label=label, mode=mode, group_size=gs, bits=bits,
                    weight_bytes=weight_bytes,
                    cold_ms=cold_ms, warm_median_ms=warm_med,
                    warm_min_ms=warm_min, warm_max_ms=warm_max,
                    tokens_per_s=tps,
                    max_abs_err_vs_bf16=err_max,
                ))
                print(f"  {label:18s} cold={cold_ms:7.2f}ms  "
                      f"warm_med={warm_med:7.3f}ms  "
                      f"min={warm_min:7.3f}ms  max={warm_max:7.3f}ms  "
                      f"toks/s={tps/1e6:6.2f}M  "
                      f"weight={weight_bytes/1024**2:6.1f}MiB  "
                      f"err={err_max:.3g}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump([asdict(r) for r in rows], f, indent=2)
        print(f"\n[wrote {args.json} with {len(rows)} rows]")

    # Print compact ratio summary against bf16 baseline.
    print("\n=== Speedup vs bf16 (warm median) ===")
    print(f"{'shape':28s} {'tokens':14s} {'mode':18s} {'speedup':>8s}  err_vs_bf16")
    by_key: dict[tuple, dict[str, Row]] = {}
    for r in rows:
        by_key.setdefault((r.shape_name, r.M_label), {})[r.mode_label] = r
    for (sn, tl), d in by_key.items():
        base = d.get("bf16")
        if not base:
            continue
        for label in ("affine_int4_g64", "affine_int4_g32",
                      "mxfp4_g32", "mxfp8_g32", "nvfp4_g16"):
            if label not in d:
                continue
            r = d[label]
            speedup = base.warm_median_ms / r.warm_median_ms
            print(f"{sn:28s} {tl:14s} {label:18s} {speedup:7.2f}x  "
                  f"{r.max_abs_err_vs_bf16:.3g}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
