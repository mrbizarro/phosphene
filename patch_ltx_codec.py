#!/usr/bin/env python3
"""Idempotent patch for the ltx-core-mlx video output codec.

Upstream encodes every output MP4 with `yuv420p crf 18` — the 4:2:0 chroma
subsampling produces visible JPEG-style block artifacts on faces / skin.
This patch swaps the default to `yuv444p crf 0` (lossless, no chroma
subsampling) and adds env-var overrides (`LTX_OUTPUT_PIX_FMT`,
`LTX_OUTPUT_CRF`) so users can flip back to the upstream behavior or to a
smaller-file near-lossless variant (e.g. yuv444p crf 12) without re-patching.

Safe to re-run — checks the file for the patch marker before touching anything.
"""
from __future__ import annotations

import sys
from pathlib import Path

CANDIDATES = [
    "ltx-2-mlx/env/lib/python3.11/site-packages/ltx_core_mlx/model/video_vae/video_vae.py",
    "ltx-2-mlx/.venv/lib/python3.11/site-packages/ltx_core_mlx/model/video_vae/video_vae.py",
]

OLD = 'cmd.extend(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", output_path])'
NEW = (
    '# PATCHED (LTX23MLX): default to lossless yuv444p crf 0 (no chroma\n'
    '        # subsampling, no JPEG-style block artifacts on faces). Override via env.\n'
    '        import os as _os\n'
    '        _pix = _os.environ.get("LTX_OUTPUT_PIX_FMT", "yuv444p")\n'
    '        _crf = _os.environ.get("LTX_OUTPUT_CRF", "0")\n'
    '        cmd.extend(["-c:v", "libx264", "-pix_fmt", _pix, "-crf", _crf, output_path])'
)


def main() -> int:
    target = next((Path(p) for p in CANDIDATES if Path(p).exists()), None)
    if target is None:
        print("video_vae.py not found in any expected venv location.", file=sys.stderr)
        print("Expected one of:", file=sys.stderr)
        for p in CANDIDATES:
            print(f"  {p}", file=sys.stderr)
        return 1

    text = target.read_text()
    if "LTX_OUTPUT_PIX_FMT" in text:
        print(f"already patched: {target}")
        return 0
    if OLD not in text:
        print(f"warning: expected line not found in {target} — upstream may have moved it", file=sys.stderr)
        return 1

    target.write_text(text.replace(OLD, NEW))
    print(f"patched {target} → yuv444p crf 0 lossless h264")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
