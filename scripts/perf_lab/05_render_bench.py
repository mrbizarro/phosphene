"""Render benchmark — Q4 vs BF16 transformer head-to-head.

Times pipe init / text+DiT load / denoise / decode-and-save with millisecond
precision and writes a JSON sidecar. Hard safety gates in a watcher thread:
trips if memory pressure > 92% or swap > 8 GB during the run.

Run:
    ltx-2-mlx/env/bin/python scripts/perf_lab/05_render_bench.py \
        --model mlx_models/ltx-2.3-mlx-q4 \
        --width 512 --height 288 --frames 25 --steps 8 \
        --seed 12345 --label q4_tiny

The render is ALWAYS the helper-equivalent path (TextToVideoPipeline +
generate_and_save). We split timing by:
  - pipe_init       : TextToVideoPipeline.__init__
  - text_encode     : _encode_text_and_load (Gemma + connector, then frees Gemma)
  - dit_load        : LTXModel + safetensors → device (the 11 GB Q4 / 38 GB bf16 step)
  - denoise         : just the denoise_loop, after manual setup
  - decode_and_save : VAE + audio + ffmpeg

Sidecar JSON includes per-phase ms, per-phase peak GPU memory, RSS, and
safety-watcher results.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import resource
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import mlx.core as mx

sys.stdout.reconfigure(line_buffering=True)


# ---- Safety gate -----------------------------------------------------------

def memory_pressure_pct() -> float:
    try:
        out = subprocess.check_output(["memory_pressure"], text=True, timeout=2)
        for line in out.splitlines():
            if "free percentage" in line:
                pct_free = int(line.rsplit(":", 1)[1].strip().rstrip("%"))
                return 100 - pct_free
    except Exception:
        pass
    return 0.0


def swap_used_gb() -> float:
    try:
        out = subprocess.check_output(["sysctl", "-n", "vm.swapusage"],
                                      text=True, timeout=2).strip()
        m = re.search(r"used\s*=\s*([\d.]+)([MGK])", out)
        if m:
            v, unit = float(m.group(1)), m.group(2)
            if unit == "G": return v
            if unit == "M": return v / 1024
            if unit == "K": return v / 1024 / 1024
    except Exception:
        pass
    return 0.0


class SafetyWatcher(threading.Thread):
    def __init__(self, max_pressure_pct: float, max_swap_gb: float, interval_s: float = 1.0):
        super().__init__(daemon=True)
        self.max_pressure_pct = max_pressure_pct
        self.max_swap_gb = max_swap_gb
        self.interval_s = interval_s
        self.tripped = False
        self.tripped_reason: Optional[str] = None
        self.peak_pressure = 0.0
        self.peak_swap = 0.0
        self._stop_evt = threading.Event()

    def stop(self) -> None:
        self._stop_evt.set()

    def run(self) -> None:  # type: ignore[override]
        while not self._stop_evt.is_set():
            try:
                p = memory_pressure_pct()
                s = swap_used_gb()
                self.peak_pressure = max(self.peak_pressure, p)
                self.peak_swap = max(self.peak_swap, s)
                if p > self.max_pressure_pct:
                    self.tripped = True
                    self.tripped_reason = f"memory_pressure {p}% > {self.max_pressure_pct}%"
                    return
                if s > self.max_swap_gb:
                    self.tripped = True
                    self.tripped_reason = f"swap {s:.2f} GB > {self.max_swap_gb} GB"
                    return
            except Exception:
                pass
            time.sleep(self.interval_s)


# ---- Memory helpers --------------------------------------------------------

def rss_gb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / 1024**3
    return rss / 1024**2


def gpu_active_gib() -> float:
    return mx.metal.get_active_memory() / 1024**3


def gpu_peak_gib() -> float:
    return mx.metal.get_peak_memory() / 1024**3


# ---- Phase timer -----------------------------------------------------------

class PhaseTimer:
    def __init__(self):
        self.phases: dict[str, float] = {}
        self.gpu_peak: dict[str, float] = {}
        self.rss: dict[str, float] = {}

    @contextmanager
    def phase(self, name: str):
        gc.collect()
        try:
            mx.metal.reset_peak_memory()
        except Exception:
            pass
        t0 = time.perf_counter()
        yield
        ms = (time.perf_counter() - t0) * 1000
        self.phases[name] = ms
        self.gpu_peak[name] = gpu_peak_gib()
        self.rss[name] = rss_gb()
        print(f"  [{name}] {ms/1000:7.2f} s  gpu_peak={self.gpu_peak[name]:5.2f} GiB  rss={self.rss[name]:5.2f} GiB")


# ---- Run one render --------------------------------------------------------

def run_render(
    *,
    model_dir: Path,
    prompt: str,
    width: int,
    height: int,
    frames: int,
    steps: int,
    seed: int,
    out_path: Path,
    low_memory: bool,
) -> dict:
    """Run a full T2V render with phase timings."""
    from ltx_pipelines_mlx.ti2vid_one_stage import TextToVideoPipeline
    from ltx_core_mlx.utils.weights import apply_quantization, load_split_safetensors
    from ltx_core_mlx.model.transformer.model import LTXModel, X0Model
    from ltx_core_mlx.components.patchifiers import compute_video_latent_shape
    from ltx_core_mlx.utils.positions import (
        compute_audio_positions, compute_audio_token_count, compute_video_positions,
    )
    from ltx_core_mlx.conditioning.types.latent_cond import create_initial_state
    from ltx_pipelines_mlx.scheduler import DISTILLED_SIGMAS
    from ltx_pipelines_mlx.utils.samplers import denoise_loop

    timer = PhaseTimer()

    print(f"\n--- render: {model_dir.name}  {width}x{height}  {frames}f  steps={steps}  seed={seed} ---")

    with timer.phase("pipe_init"):
        pipe = TextToVideoPipeline(
            model_dir=str(model_dir),
            gemma_model_id="mlx-community/gemma-3-12b-it-4bit",
            low_memory=low_memory,
        )

    with timer.phase("text_encode"):
        video_embeds, audio_embeds = pipe._encode_text_and_load(prompt)
        mx.eval(video_embeds, audio_embeds)

    with timer.phase("dit_load"):
        # Mirror the lazy DiT load path so we can time it separately from denoise.
        if pipe.dit is None:
            pipe.dit = LTXModel()
            transformer_path = model_dir / "transformer.safetensors"
            if not transformer_path.exists():
                transformer_path = model_dir / "transformer-distilled.safetensors"
            transformer_weights = load_split_safetensors(transformer_path, prefix="transformer.")
            apply_quantization(pipe.dit, transformer_weights)
            pipe.dit.load_weights(list(transformer_weights.items()))
            mx.eval(*[v for _, v in pipe.dit.parameters().items()
                      if isinstance(v, mx.array)])

    # Mirror generate() body so we can time just denoise_loop.
    F, H, W = compute_video_latent_shape(frames, height, width)
    audio_T = compute_audio_token_count(frames)

    with timer.phase("denoise_setup"):
        video_shape = (1, F * H * W, 128)
        audio_shape = (1, audio_T, 128)
        video_positions = compute_video_positions(F, H, W)
        audio_positions = compute_audio_positions(audio_T)
        video_state = create_initial_state(video_shape, seed, positions=video_positions)
        audio_state = create_initial_state(audio_shape, seed + 1, positions=audio_positions)
        sigmas = DISTILLED_SIGMAS[: steps + 1]
        x0_model = X0Model(pipe.dit)
        mx.eval(video_state.video_latent if hasattr(video_state, 'video_latent') else video_state)

    with timer.phase("denoise"):
        output = denoise_loop(
            model=x0_model,
            video_state=video_state,
            audio_state=audio_state,
            video_text_embeds=video_embeds,
            audio_text_embeds=audio_embeds,
            sigmas=sigmas,
        )
        mx.eval(output.video_latent, output.audio_latent)

    with timer.phase("unpatchify"):
        video_latent = pipe.video_patchifier.unpatchify(output.video_latent, (F, H, W))
        audio_latent = pipe.audio_patchifier.unpatchify(output.audio_latent)
        mx.eval(video_latent, audio_latent)

    with timer.phase("decode_and_save"):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Use the pipeline's existing decode-and-save path so we benefit from
        # the patches (yuv444p H.264 etc) the panel applies.
        pipe._decode_and_save_video(video_latent, audio_latent, str(out_path))

    return {
        "model_dir": str(model_dir),
        "prompt": prompt,
        "width": width, "height": height,
        "frames": frames, "steps": steps, "seed": seed,
        "phases_ms": timer.phases,
        "phase_peak_gpu_gib": timer.gpu_peak,
        "phase_rss_gib": timer.rss,
        "output": str(out_path),
        "latent_shape": [F, H, W],
        "audio_token_count": audio_T,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="A serene mountain valley at golden hour, soft cinematic light, gentle wind in the grass.")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=288)
    ap.add_argument("--frames", type=int, default=25)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out-dir", default="/tmp/phos_perf_lab/renders")
    ap.add_argument("--low-memory", action="store_true")
    ap.add_argument("--max-pressure", type=float, default=92.0)
    ap.add_argument("--max-swap-gb", type=float, default=8.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.label}.mp4"
    sidecar_path = out_dir / f"{args.label}.json"

    watcher = SafetyWatcher(args.max_pressure, args.max_swap_gb)
    watcher.start()
    t0 = time.perf_counter()
    err = None
    try:
        result = run_render(
            model_dir=Path(args.model).resolve(),
            prompt=args.prompt,
            width=args.width, height=args.height, frames=args.frames,
            steps=args.steps, seed=args.seed,
            out_path=out_path, low_memory=args.low_memory,
        )
    except Exception as e:
        err = repr(e)
        result = {"error": err}
    finally:
        watcher.stop()
        watcher.join(timeout=2)

    total_ms = (time.perf_counter() - t0) * 1000
    result.update({
        "label": args.label,
        "total_ms": total_ms,
        "watcher": {
            "tripped": watcher.tripped,
            "tripped_reason": watcher.tripped_reason,
            "peak_pressure_pct": watcher.peak_pressure,
            "peak_swap_gb": watcher.peak_swap,
        },
    })

    with open(sidecar_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n=== summary ({args.label}) ===")
    print(f"total: {total_ms/1000:.2f} s   error={err}")
    if "phases_ms" in result:
        for k, v in result["phases_ms"].items():
            print(f"  {k:18s} {v/1000:7.2f} s")
        peak = max(result.get("phase_peak_gpu_gib", {0: 0}).values())
        print(f"peak GPU: {peak:.2f} GiB")
    print(f"safety: tripped={watcher.tripped} "
          f"peak_pressure={watcher.peak_pressure:.0f}% peak_swap={watcher.peak_swap:.2f} GB")
    print(f"\nsidecar: {sidecar_path}")
    print(f"video:   {out_path}")

    if err is not None:
        return 1
    if watcher.tripped:
        print(f"\n*** SAFETY GATE TRIPPED: {watcher.tripped_reason} ***", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
