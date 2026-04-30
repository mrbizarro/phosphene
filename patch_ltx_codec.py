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

Known unfixed: KeyframeInterpolationPipeline OOMs at the stage-1 → stage-2
transition on 64 GB Macs at full resolution. We tried freeing/reloading
the DiT around the upscale and it didn't help. Worked around in the panel
by clamping keyframe-mode resolution to 768×432 by default — see the
`mode == "keyframe"` block in mlx_ltx_panel.py.

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
    '        # `+faststart` moves the moov atom to the front of the file so the\n'
    '        # gallery thumbnails (preload="metadata") can decode the first\n'
    '        # frame without downloading the full clip — without it the thumbs\n'
    '        # render black until clicked.\n'
    '        import os as _os\n'
    '        _pix = _os.environ.get("LTX_OUTPUT_PIX_FMT", "yuv444p")\n'
    '        _crf = _os.environ.get("LTX_OUTPUT_CRF", "0")\n'
    '        cmd.extend(["-c:v", "libx264", "-pix_fmt", _pix, "-crf", _crf,\n'
    '                    "-movflags", "+faststart", output_path])'
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


# NOTE: Keyframe interpolation OOMs at the stage-1 → stage-2 transition on
# 64 GB Macs. We tried free-DiT-before-upscale + reload-after-upscale; that
# *didn't* help (the reload hit the same memory peak from a different angle
# and added ~30s of wall time). Workaround is now in the panel side: keyframe
# mode runs at half resolution (640×352 stage-1, 1280×704 stage-2 still goes
# OOM; 384×216 stage-1 / 768×432 stage-2 fits). Looking for a real upstream
# fix later. Keep this comment so we don't re-introduce the failed patch.


# Outcome codes for apply_patch — three-valued (vs the old True/False) so
# main() can distinguish a genuinely missing target / drifted upstream from
# the no-op "already patched" case. Without this distinction the install
# used to exit 0 on a corrupt patch attempt and ship a broken pipeline.
OUTCOME_APPLIED = "applied"
OUTCOME_ALREADY = "already"
OUTCOME_MISSING = "missing"          # target file not on disk
OUTCOME_DRIFT   = "drift"            # target found but expected text isn't there


def _atomic_write(target: Path, text: str) -> None:
    """Write to a temp file in the same directory, fsync, then os.replace.
    Avoids the failure mode where Pinokio kills the install mid-write and
    leaves a half-written .py that imports as a SyntaxError forever."""
    import os, tempfile
    target_dir = target.parent
    fd, tmp_path = tempfile.mkstemp(prefix=target.name + ".", dir=str(target_dir))
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, target)
    except Exception:
        # Clean up the temp file if we never made it to the replace.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def apply_patch(target: Path, old: str, new: str, marker: str, label: str) -> str:
    """Idempotently apply old→new replacement on `target`. Returns one of
    OUTCOME_APPLIED / OUTCOME_ALREADY / OUTCOME_MISSING / OUTCOME_DRIFT —
    deep-review fix to surface upstream drift loudly instead of silently
    no-op'ing the patch and shipping a broken install."""
    if target is None or not target.exists():
        print(f"  [{label}] MISSING — target file not found", file=sys.stderr)
        return OUTCOME_MISSING
    text = target.read_text()
    if marker in text:
        print(f"  [{label}] already patched: {target}")
        return OUTCOME_ALREADY
    if old not in text:
        print(
            f"  [{label}] DRIFT — expected text not found in {target}. "
            f"Upstream likely restructured this file. The patch needs to be "
            f"updated (see patch_ltx_codec.py); the install will fail loud "
            f"rather than ship an unpatched copy.",
            file=sys.stderr,
        )
        return OUTCOME_DRIFT
    _atomic_write(target, text.replace(old, new))
    print(f"  [{label}] patched {target}")
    return OUTCOME_APPLIED


def main() -> int:
    print("Applying LTX23MLX patches:")
    outcomes: list[tuple[str, str]] = []

    # Patch 1: codec
    codec_target = _find("ltx_core_mlx/model/video_vae/video_vae.py")
    outcomes.append(("codec (yuv444p crf 0)", apply_patch(
        codec_target, PATCH_CODEC_OLD, PATCH_CODEC_NEW,
        marker="LTX_OUTPUT_PIX_FMT",
        label="codec (yuv444p crf 0)",
    )))

    # Patch 2: I2V OOM cleanup before decode
    # SHIP-BLOCKER history: install.js pins ltx-2-mlx to commit dcd639e
    # (the 0.1.0 audio baseline) because newer 0.2.0 commits regressed
    # audio amplitude by 22 dB. At dcd639e the I2V function inlines the
    # decode steps instead of calling _decode_and_save_video, so this
    # patch's old/new strings don't match. Mark it OPTIONAL — DRIFT is
    # treated as a warning (the patch only helps Q8 / 64 GB I2V runs;
    # T2V works fine without it). The codec patch above is REQUIRED.
    i2v_target = _find("ltx_pipelines_mlx/ti2vid_one_stage.py")
    i2v_outcome = apply_patch(
        i2v_target, PATCH_I2V_OOM_OLD, PATCH_I2V_OOM_NEW,
        marker="PATCHED (LTX23MLX): mirror the parent T2V cleanup",
        label="I2V OOM (free DiT before decode)",
    )
    if i2v_outcome == OUTCOME_DRIFT:
        print(
            "  [I2V OOM] note: this patch only matches the 0.2.0+ I2V "
            "structure. On the pinned 0.1.0 baseline (dcd639e) it's a no-op. "
            "T2V output is unaffected; HQ I2V on Q8 may OOM without it.",
            file=sys.stderr,
        )
        # Treat as ALREADY for the rollup so install doesn't fail.
        i2v_outcome = OUTCOME_ALREADY
    outcomes.append(("I2V OOM (free DiT before decode)", i2v_outcome))

    # (Keyframe OOM patch was removed — see NOTE in the patches block above.
    #  The fix is currently a panel-side resolution clamp, not a pipeline edit.)

    applied = sum(1 for _, o in outcomes if o == OUTCOME_APPLIED)
    already = sum(1 for _, o in outcomes if o == OUTCOME_ALREADY)
    failed  = [(label, o) for label, o in outcomes if o in (OUTCOME_MISSING, OUTCOME_DRIFT)]

    if failed:
        print(
            f"\nERROR: {len(failed)} patch(es) failed to apply:",
            file=sys.stderr,
        )
        for label, o in failed:
            print(f"  - [{label}] {o}", file=sys.stderr)
        print(
            "\nThis exits non-zero so install.js / update.js fail loud. "
            "The runtime depends on these patches; running with them missing "
            "produces broken output (chroma artifacts, mid-job OOMs).",
            file=sys.stderr,
        )
        return 2

    if applied == 0:
        print(f"All patches already applied ({already} already-patched, 0 changed).")
    else:
        print(f"Done — {applied} patch(es) newly applied, {already} already-patched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
