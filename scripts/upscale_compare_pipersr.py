#!/usr/bin/env python3
"""Compare Phosphene's ffmpeg Lanczos export with PiperSR + ffmpeg fit.

This is an experimental dev/lab tool, not a production export path. PiperSR's
code is AGPL-3.0 and its bundled model requires visible attribution for public
apps and separate licensing for commercial use, so keep this opt-in until the
licensing/product decision is made.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PINOKIO_FFMPEG = Path("/Users/salo/pinokio/bin/ffmpeg-env/bin/ffmpeg")
PINOKIO_FFPROBE = Path("/Users/salo/pinokio/bin/ffmpeg-env/bin/ffprobe")


def run(cmd: list[str], label: str) -> None:
    print(f"\n[{label}] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def ffmpeg_path() -> Path:
    if PINOKIO_FFMPEG.exists():
        return PINOKIO_FFMPEG
    found = shutil.which("ffmpeg")
    if not found:
        raise SystemExit("ffmpeg not found")
    return Path(found)


def ffprobe_path() -> Path:
    if PINOKIO_FFPROBE.exists():
        return PINOKIO_FFPROBE
    found = shutil.which("ffprobe")
    if not found:
        raise SystemExit("ffprobe not found")
    return Path(found)


def probe_video(path: Path) -> dict:
    cmd = [
        str(ffprobe_path()),
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_frames,duration",
        "-of", "json",
        str(path),
    ]
    data = json.loads(subprocess.check_output(cmd))
    return data["streams"][0]


def target_for_fit_720p(width: int, height: int) -> tuple[int, int, str]:
    if width >= height:
        return 1280, 720, "720p"
    return 720, 1280, "v720p"


def fit_filter(target_w: int, target_h: int) -> str:
    return (
        f"scale={target_w}:{target_h}:"
        "force_original_aspect_ratio=decrease:flags=lanczos,"
        f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:color=black"
    )


def make_lanczos(input_mp4: Path, output_mp4: Path, crf: str, pix_fmt: str, preset: str) -> float:
    info = probe_video(input_mp4)
    target_w, target_h, _ = target_for_fit_720p(int(info["width"]), int(info["height"]))
    t0 = time.perf_counter()
    run([
        str(ffmpeg_path()), "-hide_banner", "-y",
        "-i", str(input_mp4),
        "-vf", fit_filter(target_w, target_h),
        "-c:v", "libx264", "-pix_fmt", pix_fmt, "-crf", crf, "-preset", preset,
        "-movflags", "+faststart",
        "-c:a", "copy",
        str(output_mp4),
    ], "lanczos")
    return time.perf_counter() - t0


def load_pipersr_model():
    try:
        import coremltools as ct
        import pipersr
    except Exception as exc:
        raise SystemExit(
            "PiperSR is not installed in this Python. Install in the dev venv with:\n"
            "  ltx-2-mlx/env/bin/pip install pipersr\n"
            f"Original error: {exc}"
        )

    model_path = pipersr._find_model()  # PiperSR exposes this in 1.0.0.
    print(f"PiperSR model: {model_path}", flush=True)
    # coremltools 9 renamed CPU_AND_NEURAL_ENGINE to CPU_AND_NE. PiperSR 1.0.0
    # still uses the older name internally, so handle both here.
    compute_unit = getattr(ct.ComputeUnit, "CPU_AND_NE", None)
    if compute_unit is None:
        compute_unit = getattr(ct.ComputeUnit, "CPU_AND_NEURAL_ENGINE")
    return ct.models.MLModel(str(model_path), compute_units=compute_unit)


def _pad_to_tile(img: Image.Image, tile: int = 128) -> tuple[Image.Image, int, int]:
    w, h = img.size
    pad_w = ((w + tile - 1) // tile) * tile
    pad_h = ((h + tile - 1) // tile) * tile
    if pad_w == w and pad_h == h:
        return img, w, h

    # Edge extension avoids black tile borders without inventing content.
    arr = np.asarray(img)
    padded = np.pad(arr, ((0, pad_h - h), (0, pad_w - w), (0, 0)), mode="edge")
    return Image.fromarray(padded), w, h


def piper_upscale_image(model, img: Image.Image) -> Image.Image:
    """Run PiperSR's fixed 128x128 CoreML model as tiled 2x SR."""
    tile = 128
    padded, orig_w, orig_h = _pad_to_tile(img.convert("RGB"), tile=tile)
    pad_w, pad_h = padded.size
    out = Image.new("RGB", (pad_w * 2, pad_h * 2))

    for y in range(0, pad_h, tile):
        for x in range(0, pad_w, tile):
            crop = padded.crop((x, y, x + tile, y + tile))
            result = model.predict({"input_image": crop})
            out_tile = result.get("output_image")
            if out_tile is None:
                out_tile = next(iter(result.values()))
            out.paste(out_tile.convert("RGB"), (x * 2, y * 2))

    return out.crop((0, 0, orig_w * 2, orig_h * 2))


def piper_upscale_frames(frames_in: Path, frames_out: Path) -> float:
    frames_out.mkdir(parents=True, exist_ok=True)
    model = load_pipersr_model()
    files = sorted(frames_in.glob("frame_*.png"))
    if not files:
        raise SystemExit("No extracted frames found")

    t0 = time.perf_counter()
    for i, src in enumerate(files, 1):
        img = Image.open(src).convert("RGB")
        out_img = piper_upscale_image(model, img)
        out_img.save(frames_out / src.name)
        if i == 1 or i == len(files) or i % 20 == 0:
            print(f"PiperSR frames: {i}/{len(files)}", flush=True)
    return time.perf_counter() - t0


def make_pipersr(input_mp4: Path, output_mp4: Path, crf: str, pix_fmt: str, preset: str, keep_work: bool) -> float:
    info = probe_video(input_mp4)
    fps = info.get("avg_frame_rate") or "24/1"
    target_w, target_h, _ = target_for_fit_720p(int(info["width"]), int(info["height"]))
    t0 = time.perf_counter()

    work_parent = output_mp4.parent if keep_work else None
    with tempfile.TemporaryDirectory(prefix=f"{input_mp4.stem}_pipersr_", dir=work_parent) as tmp:
        work = Path(tmp)
        frames_in = work / "frames"
        frames_2x = work / "frames_2x"
        frames_in.mkdir()
        run([
            str(ffmpeg_path()), "-hide_banner", "-y",
            "-i", str(input_mp4),
            str(frames_in / "frame_%06d.png"),
        ], "extract")
        sr_sec = piper_upscale_frames(frames_in, frames_2x)
        print(f"PiperSR core pass: {sr_sec:.2f}s", flush=True)
        run([
            str(ffmpeg_path()), "-hide_banner", "-y",
            "-framerate", fps,
            "-i", str(frames_2x / "frame_%06d.png"),
            "-i", str(input_mp4),
            "-map", "0:v:0", "-map", "1:a?",
            "-vf", fit_filter(target_w, target_h),
            "-c:v", "libx264", "-pix_fmt", pix_fmt, "-crf", crf, "-preset", preset,
            "-movflags", "+faststart",
            "-c:a", "copy",
            str(output_mp4),
        ], "encode-pipersr")
    return time.perf_counter() - t0


def make_contact_sheet(paths: list[Path], output_jpg: Path) -> None:
    inputs: list[str] = []
    labels: list[str] = []
    for path in paths:
        inputs.extend(["-i", str(path)])
        labels.append(path.stem)
    # 3 columns: input/native, Lanczos, PiperSR. Frame sample around 40%.
    fit = "scale=426:320:force_original_aspect_ratio=decrease,pad=426:320:(ow-iw)/2:(oh-ih)/2:color=black"
    vf = (
        f"[0:v]{fit},drawtext=text='SOURCE':x=12:y=12:"
        "fontcolor=white:fontsize=24:box=1:boxcolor=black@0.55[v0];"
        f"[1:v]{fit},drawtext=text='LANCZOS':x=12:y=12:"
        "fontcolor=white:fontsize=24:box=1:boxcolor=black@0.55[v1];"
        f"[2:v]{fit},drawtext=text='PIPERSR':x=12:y=12:"
        "fontcolor=white:fontsize=24:box=1:boxcolor=black@0.55[v2];"
        "[v0][v1][v2]hstack=inputs=3"
    )
    run([
        str(ffmpeg_path()), "-hide_banner", "-y",
        *inputs,
        "-filter_complex", vf,
        "-frames:v", "1", "-update", "1",
        str(output_jpg),
    ], "contact-sheet")


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare Lanczos vs PiperSR exports for one Phosphene MP4.")
    ap.add_argument("input", type=Path)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--crf", default="18")
    ap.add_argument("--pix-fmt", default="yuv420p")
    ap.add_argument("--preset", default="medium")
    ap.add_argument("--keep-work", action="store_true")
    args = ap.parse_args()

    input_mp4 = args.input.expanduser().resolve()
    if not input_mp4.exists():
        raise SystemExit(f"Input not found: {input_mp4}")
    out_dir = (args.output_dir or input_mp4.parent).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    lanczos = out_dir / f"{input_mp4.stem}_compare_lanczos_720p.mp4"
    piper = out_dir / f"{input_mp4.stem}_compare_pipersr_720p.mp4"
    sheet = out_dir / f"{input_mp4.stem}_compare_sheet.jpg"

    print("PiperSR attribution: Powered by PiperSR from ModelPiper — https://modelpiper.com", flush=True)
    lanczos_sec = make_lanczos(input_mp4, lanczos, args.crf, args.pix_fmt, args.preset)
    piper_sec = make_pipersr(input_mp4, piper, args.crf, args.pix_fmt, args.preset, args.keep_work)
    make_contact_sheet([input_mp4, lanczos, piper], sheet)

    print("\nDone:")
    print(f"  Source:  {input_mp4}")
    print(f"  Lanczos: {lanczos} ({lanczos.stat().st_size / 1024 / 1024:.2f} MB, {lanczos_sec:.2f}s)")
    print(f"  PiperSR: {piper} ({piper.stat().st_size / 1024 / 1024:.2f} MB, {piper_sec:.2f}s)")
    print(f"  Sheet:   {sheet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
