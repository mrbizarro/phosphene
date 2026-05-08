"""Loader smoke test for the bf16 transformer.

Loads the test pipeline against ``mlx_models/ltx-2.3-mlx-bf16-test`` and
reports:
  - whether the transformer state-dict has any ``.scales`` keys (= quantized)
  - dtypes of a sample of transformer weights
  - peak MLX active memory after load
  - process RSS after load
  - any missing / unexpected keys

Run after the 38 GB transformer download completes:
    ltx-2-mlx/env/bin/python scripts/perf_lab/04_loader_smoke.py
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from collections import Counter
from pathlib import Path

import mlx.core as mx

# Ensure unbuffered, helpful tracebacks
sys.stdout.reconfigure(line_buffering=True)


def fmt_gb(n: int | float) -> str:
    return f"{n / 1024**3:.2f} GiB"


def rss_gb() -> float:
    import resource
    # macOS: ru_maxrss is in BYTES (Linux: KB) — handle both.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / 1024**3
    return rss / 1024**2  # KB -> GiB


def smoke(model_dir: Path) -> None:
    print(f"\n=== smoke test: {model_dir} ===")
    print(f"exists: {model_dir.exists()}, is_dir: {model_dir.is_dir()}")
    if not model_dir.exists():
        print("  SKIP — directory missing")
        return

    files = sorted([p.name for p in model_dir.iterdir() if not p.name.startswith('.')])
    print(f"  {len(files)} files: {files[:6]}{'...' if len(files) > 6 else ''}")

    # 1. Probe the transformer file directly to count quantized vs unquantized keys.
    transformer = model_dir / "transformer-distilled.safetensors"
    if not transformer.exists():
        transformer = model_dir / "transformer.safetensors"
    print(f"\n[1] direct probe: {transformer.name} ({fmt_gb(transformer.stat().st_size)})")

    raw = mx.load(str(transformer))
    has_scales = sum(1 for k in raw if k.endswith(".scales"))
    has_biases = sum(1 for k in raw if k.endswith(".biases"))
    weight_keys = [k for k in raw if k.endswith(".weight")]
    dtypes_count: Counter = Counter(str(v.dtype) for v in raw.values())
    print(f"  total keys: {len(raw)}")
    print(f"  weight keys: {len(weight_keys)}")
    print(f"  .scales keys: {has_scales}  (>0 => quantized)")
    print(f"  .biases keys: {has_biases}  (quant biases)")
    print(f"  dtypes: {dict(dtypes_count)}")

    # Spot-check shapes of a transformer Linear weight to verify bf16-style
    # (out, in) layout vs Q4 packed (out, in*bits/32).
    sample_keys = [k for k in weight_keys if "transformer_blocks.0.attn1.to_q" in k]
    for k in sample_keys[:3]:
        print(f"  sample: {k}  shape={raw[k].shape}  dtype={raw[k].dtype}")

    # 2. Build the actual pipeline and load weights through it.
    from ltx_pipelines_mlx.ti2vid_one_stage import TextToVideoPipeline

    print(f"\n[2] pipeline build")
    mx.metal.reset_peak_memory()
    t0 = time.perf_counter()
    pipe = TextToVideoPipeline(
        model_dir=str(model_dir),
        gemma_model_id="mlx-community/gemma-3-12b-it-4bit",
        low_memory=False,  # match Comfortable tier
    )
    build_ms = (time.perf_counter() - t0) * 1000
    print(f"  TextToVideoPipeline.__init__: {build_ms:.1f} ms")

    # 3. Trigger weights load by encoding a tiny prompt (this loads Gemma +
    #    connector + transformer + VAE).
    print(f"\n[3] full weight load via _encode_text_and_load + lazy DiT load")
    t0 = time.perf_counter()
    video_embeds, audio_embeds = pipe._encode_text_and_load("hello world")
    text_ms = (time.perf_counter() - t0) * 1000
    mx.eval(video_embeds, audio_embeds)
    print(f"  text encode + connector: {text_ms:.1f} ms "
          f"(emb shapes {video_embeds.shape}, {audio_embeds.shape})")

    # The DiT load happens lazily before first denoise. Force it now by
    # poking the pipeline's lazy-load path: we use the same code as
    # generate(). The cleanest way is to call generate() with num_steps=1
    # but that'd be slow. Instead we mirror the load path manually.
    from ltx_core_mlx.utils.weights import apply_quantization, load_split_safetensors
    from ltx_core_mlx.model.transformer.model import LTXModel

    from ltx_core_mlx.utils.weights import apply_quantization, load_split_safetensors
    from ltx_core_mlx.model.transformer.model import LTXModel

    if pipe.dit is None:
        t0 = time.perf_counter()
        pipe.dit = LTXModel()
        transformer_path = model_dir / "transformer.safetensors"
        if not transformer_path.exists():
            transformer_path = model_dir / "transformer-distilled.safetensors"
        transformer_weights = load_split_safetensors(transformer_path, prefix="transformer.")
        load_ms = (time.perf_counter() - t0) * 1000
        print(f"  load_split_safetensors: {load_ms:.1f} ms ({len(transformer_weights)} keys)")

        # Sanity: was anything stripped by the prefix?
        any_scales = any(k.endswith(".scales") for k in transformer_weights)
        print(f"  weights still have .scales after prefix-strip? {any_scales}")

        t0 = time.perf_counter()
        apply_quantization(pipe.dit, transformer_weights)
        apply_q_ms = (time.perf_counter() - t0) * 1000
        print(f"  apply_quantization (no-op for bf16): {apply_q_ms:.1f} ms")

        t0 = time.perf_counter()
        pipe.dit.load_weights(list(transformer_weights.items()))
        load_w_ms = (time.perf_counter() - t0) * 1000
        print(f"  dit.load_weights: {load_w_ms:.1f} ms")

        # Sample a Linear in the loaded DiT and report its dtype / type.
        first_block = pipe.dit.transformer_blocks[0]
        to_q = first_block.attn1.to_q
        print(f"  first block.attn1.to_q type: {type(to_q).__name__}")
        if hasattr(to_q, "weight"):
            print(f"    weight shape={to_q.weight.shape} dtype={to_q.weight.dtype}")
        # Force the param graph to be evaluated so memory is realized.
        mx.eval(*[p for _, p in pipe.dit.parameters().items() if isinstance(p, mx.array)])

    # 4. Memory snapshot
    print(f"\n[4] memory after load")
    print(f"  mx.metal.get_active_memory: {fmt_gb(mx.metal.get_active_memory())}")
    print(f"  mx.metal.get_peak_memory : {fmt_gb(mx.metal.get_peak_memory())}")
    print(f"  process RSS              : {rss_gb():.2f} GiB")

    # 5. Cleanup
    pipe = None
    gc.collect()
    mx.metal.clear_cache()
    print(f"\n[5] after cleanup")
    print(f"  mx.metal.get_active_memory: {fmt_gb(mx.metal.get_active_memory())}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="mlx_models/ltx-2.3-mlx-bf16-test")
    ap.add_argument("--also-q4", action="store_true",
                    help="Also run smoke against Q4 dir for comparison")
    args = ap.parse_args()

    smoke(Path(args.model_dir).resolve())
    if args.also_q4:
        # Don't try to free between runs — separate processes are cleaner.
        # For now just print a hint.
        print("\nNote: rerun with --model-dir mlx_models/ltx-2.3-mlx-q4 to compare.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
