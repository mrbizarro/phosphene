#!/usr/bin/env python3
"""Idempotent patches against the ltx-core-mlx + ltx-pipelines-mlx packages.

Two upstream issues we patch around:

1. Output codec is `yuv420p crf 18` — 4:2:0 chroma subsampling produces
   visible JPEG-style block artifacts on faces / skin. Patched to
   `yuv444p crf 0` (lossless, no chroma subsampling) with env-var overrides
   (`LTX_OUTPUT_PIX_FMT`, `LTX_OUTPUT_CRF`).

2. ImageToVideoPipeline.generate_and_save doesn't free the DiT + text
   encoder before VAE decode (the parent T2V version does). On Q8 / 64 GB
   Macs this OOMs the helper subprocess on the I2V code path. Patched to
   add the same low_memory cleanup the parent has.

Safe to re-run — each patch checks for its marker before touching anything.
"""
from __future__ import annotations

import sys
from pathlib import Path

VENV_ROOTS = [
    "ltx-2-mlx/env/lib/python3.11/site-packages",      # Pinokio
    "ltx-2-mlx/.venv/lib/python3.11/site-packages",    # manual
    "ltx-2-mlx/packages/ltx-core-mlx/src",             # editable (ltx-core)
    "ltx-2-mlx/packages/ltx-pipelines-mlx/src",        # editable (ltx-pipelines)
]


def _find(rel: str) -> Path | None:
    """Resolve a package-relative path under the first venv root that contains it."""
    for root in VENV_ROOTS:
        p = Path(root) / rel
        if p.exists():
            return p
    return None


# ---- Patch 1: lossless h264 codec --------------------------------------------
PATCH_CODEC_OLD = 'cmd.extend(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", output_path])'
PATCH_CODEC_NEW = (
    '# PATCHED (LTX23MLX): default to lossless yuv444p crf 0 (no chroma\n'
    '        # subsampling, no JPEG-style block artifacts on faces). Override via env.\n'
    '        import os as _os\n'
    '        _pix = _os.environ.get("LTX_OUTPUT_PIX_FMT", "yuv444p")\n'
    '        _crf = _os.environ.get("LTX_OUTPUT_CRF", "0")\n'
    '        cmd.extend(["-c:v", "libx264", "-pix_fmt", _pix, "-crf", _crf, output_path])'
)

# ---- Patch 2: I2V free-DiT-before-VAE-decode (OOM fix) -----------------------
# The parent T2V's generate_and_save does this; the I2V override forgot to.
# On Q8 / 64 GB Macs the I2V path OOMs because dev transformer + Gemma + VAE
# coexist during stage-2 decode.
PATCH_I2V_OOM_OLD = '''        video_latent, audio_latent = self.generate_from_image(
            prompt=prompt,
            image=image,
            height=height,
            width=width,
            num_frames=num_frames,
            seed=seed,
            num_steps=num_steps,
        )

        return self._decode_and_save_video(video_latent, audio_latent, output_path)'''
PATCH_I2V_OOM_NEW = '''        video_latent, audio_latent = self.generate_from_image(
            prompt=prompt,
            image=image,
            height=height,
            width=width,
            num_frames=num_frames,
            seed=seed,
            num_steps=num_steps,
        )

        # PATCHED (LTX23MLX): mirror the parent T2V cleanup before VAE decode.
        # Without this, dev transformer + Gemma + VAE all live in memory during
        # decode and OOM the helper on Q8 / 64 GB Macs.
        if self.low_memory:
            self.dit = None
            self.text_encoder = None
            self.feature_extractor = None
            self._loaded = False
            try:
                from ltx_core_mlx.utils.memory import aggressive_cleanup
                aggressive_cleanup()
            except Exception:
                pass

        return self._decode_and_save_video(video_latent, audio_latent, output_path)'''


def apply_patch(target: Path, old: str, new: str, marker: str, label: str) -> bool:
    """Idempotently apply old→new replacement on `target`. Returns True if changed."""
    if target is None or not target.exists():
        print(f"  [{label}] target file not found", file=sys.stderr)
        return False
    text = target.read_text()
    if marker in text:
        print(f"  [{label}] already patched: {target}")
        return False
    if old not in text:
        print(f"  [{label}] expected text not found — upstream may have moved it", file=sys.stderr)
        return False
    target.write_text(text.replace(old, new))
    print(f"  [{label}] patched {target}")
    return True


def main() -> int:
    print("Applying LTX23MLX patches:")
    changed = 0

    # Patch 1: codec
    codec_target = _find("ltx_core_mlx/model/video_vae/video_vae.py")
    if apply_patch(codec_target, PATCH_CODEC_OLD, PATCH_CODEC_NEW,
                   marker="LTX_OUTPUT_PIX_FMT",
                   label="codec (yuv444p crf 0)"):
        changed += 1

    # Patch 2: I2V OOM cleanup before decode
    i2v_target = _find("ltx_pipelines_mlx/ti2vid_one_stage.py")
    if apply_patch(i2v_target, PATCH_I2V_OOM_OLD, PATCH_I2V_OOM_NEW,
                   marker="PATCHED (LTX23MLX): mirror the parent T2V cleanup",
                   label="I2V OOM (free DiT before decode)"):
        changed += 1

    if changed == 0:
        print("All patches already applied (or upstream layout changed).")
    else:
        print(f"Done — {changed} patch(es) applied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
