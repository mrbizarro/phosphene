#!/usr/bin/env python3
"""Idempotent patches against the ltx-core-mlx + ltx-pipelines-mlx packages.

Upstream issues we patch around:

1. Output codec is `yuv420p crf 18` — 4:2:0 chroma subsampling produces
   visible JPEG-style block artifacts on faces / skin. Patched to
   `yuv444p crf 0` (lossless, no chroma subsampling) with env-var overrides
   (`LTX_OUTPUT_PIX_FMT`, `LTX_OUTPUT_CRF`).

2. ImageToVideoPipeline.generate_and_save doesn't free the DiT + text
   encoder before VAE decode (the parent T2V version does). On Q8 / 64 GB
   Macs this OOMs the helper subprocess on the I2V code path. Patched to
   add the same low_memory cleanup the parent has.

3. VideoDecoder.decode_and_stream advertises temporal streaming but still
   full-decodes the video tensor before writing frames. Patched to use
   temporal tiled_decode() automatically for longer clips, with
   LTX_VAE_STREAMING=0/1 override.

4. TextToVideoPipeline / ImageToVideoPipeline have fps-aware position helpers
   in ltx-core-mlx, but the one-stage pipeline hardcodes 24fps at the call
   site. Patched to expose frame_rate so Phosphene can offer an explicit
   "12 → 24fps" long-clip mode without desynchronizing generated audio.

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


# ---- Patch 3: I2V free vae_encoder + feature_extractor BEFORE denoise --------
# Credit: cocktailpeanut review. The existing Patch 2 only frees memory
# right before the VAE *decode* step. By that point we've already paid the
# peak: during the denoise loop we have DiT (~10.54 GiB Q4 / ~19 GiB Q8 dev)
# + feature_extractor with connector (~5.91 GiB) + vae_encoder (still
# resident because ImageToVideoPipeline.load() reloads it after super().
# load() finishes) + denoise activations. That overlap is what kills the
# helper "10 seconds in" silently.
#
# Fix: right after _encode_text_and_load() returns inside generate_from_image,
# explicitly null out vae_encoder + feature_extractor. They aren't needed
# during denoise — vae_encoder was used in step 1 and reload happens via
# the I2V load(); feature_extractor only mattered for the text-encoding
# step that just completed.
PATCH_I2V_PREDENOISE_OLD = '''        # Step 2: Encode text, then load remaining components
        video_embeds, audio_embeds = self._encode_text_and_load(prompt)
        assert self.dit is not None'''
PATCH_I2V_PREDENOISE_NEW = '''        # Step 2: Encode text, then load remaining components
        video_embeds, audio_embeds = self._encode_text_and_load(prompt)

        # PATCHED (LTX23MLX, peanut review): free vae_encoder +
        # feature_extractor BEFORE the denoise loop, not just before VAE
        # decode. The old patch (Patch 2) freed too late — denoise itself
        # was the peak. With DiT (10.5 GB Q4) + connector (5.9 GB) + vae_enc
        # + activations all overlapping during the loop, the helper was
        # being SIGKILL'd by jetsam on memory-pressured Macs (cocktailpeanut
        # repro: "I2V started, ~10s in stopped with no error").
        if self.low_memory:
            self.vae_encoder = None
            self.feature_extractor = None
            try:
                from ltx_core_mlx.utils.memory import aggressive_cleanup as _cleanup
                _cleanup()
            except Exception:
                pass

        assert self.dit is not None'''

# ---- Patch 4: Base load() also clears feature_extractor before DiT -----------
# Same diagnosis. Base TextToVideoPipeline.load() only clears self.text_encoder
# before loading the transformer; feature_extractor (which holds the connector,
# ~5.9 GiB on Q4) stays resident through the DiT load. Two big weight blobs
# coexisting in MLX memory at peak load is exactly the avoidable overlap.
PATCH_BASE_LOAD_OLD = '''        # Stage 2: DiT (largest component — load after text encoding frees Gemma)
        if self.dit is None:
            if self.low_memory:
                # Free text encoder before loading transformer to fit in RAM
                self.text_encoder = None
                aggressive_cleanup()'''
PATCH_BASE_LOAD_NEW = '''        # Stage 2: DiT (largest component — load after text encoding frees Gemma)
        if self.dit is None:
            if self.low_memory:
                # PATCHED (LTX23MLX, peanut review): also clear feature_extractor
                # alongside text_encoder before loading the transformer. The
                # connector lives inside feature_extractor (~5.9 GiB Q4) and
                # without this it stays resident through the 10.5 GiB DiT load,
                # peaking around 16+ GiB just for weights before activations.
                self.text_encoder = None
                self.feature_extractor = None
                aggressive_cleanup()'''

# ---- Patch 5: stream VAE decode in temporal chunks -----------------------------
# Upstream VideoDecoder.decode_and_stream() says it decodes one temporal chunk at
# a time, but the implementation still calls self.decode(latent) for the full
# video volume before writing frames to ffmpeg. On long 720p-ish clips this is
# exactly the "denoise is done, now it freezes at the end" memory-pressure tail.
# The decoder already has tiled_decode(); use temporal tiling automatically for
# longer clips while keeping LTX_VAE_STREAMING=0/1 as explicit fallback/force.
PATCH_VAE_STREAM_OLD = '''        try:
            # Decode full volume and stream frames
            pixels = self.decode(latent)
            mx.async_eval(pixels)

            num_frames = pixels.shape[2]
            for i in range(num_frames):
                frame = pixels[:, :, i, :, :]  # (B, 3, H, W)
                frame = mx.clip(frame, -1.0, 1.0)
                frame = ((frame + 1.0) * 127.5).astype(mx.uint8)
                # (1, 3, H, W) -> (H, W, 3)
                frame_hwc = frame[0].transpose(1, 2, 0)
                mx.eval(frame_hwc)  # must be sync — async_eval can write before data is ready
                proc.stdin.write(bytes(memoryview(frame_hwc)))
                del frame, frame_hwc
                if i % 8 == 0:
                    aggressive_cleanup()'''

PATCH_VAE_STREAM_NEW = '''        try:
            # PATCHED (LTX23MLX): temporal streaming decode. The old code
            # decoded the full video tensor before writing frames, causing
            # multi-minute end-of-render stalls or jetsam on long/high-res jobs.
            import os as _os
            _streaming = _os.environ.get("LTX_VAE_STREAMING", "auto").strip().lower()
            _frames_est = 1 + (int(latent.shape[2]) - 1) * 8
            _auto_max = int(_os.environ.get("LTX_VAE_STREAMING_AUTO_MAX_FRAMES", "121"))
            if _streaming in ("1", "true", "yes", "on", "stream", "streaming", "chunked"):
                _streaming_enabled = True
            elif _streaming in ("0", "false", "no", "off", "full"):
                _streaming_enabled = False
            else:
                # Auto: short 5 s clips are faster as a single decode; long clips
                # keep temporal chunks to avoid memory-pressure stalls.
                _streaming_enabled = _frames_est > _auto_max

            if _streaming_enabled:
                from ltx_core_mlx.model.video_vae.tiling import TilingConfig, TemporalTilingConfig

                _tile_frames = int(_os.environ.get("LTX_VAE_TILE_FRAMES", "64"))
                _overlap_frames = int(_os.environ.get("LTX_VAE_TILE_OVERLAP_FRAMES", "24"))
                _tile_frames = max(16, (_tile_frames // 8) * 8)
                _overlap_frames = max(0, (_overlap_frames // 8) * 8)
                if _overlap_frames >= _tile_frames:
                    _overlap_frames = max(0, _tile_frames - 8)
                _tiling = TilingConfig(
                    spatial_config=None,
                    temporal_config=TemporalTilingConfig(
                        tile_size_in_frames=_tile_frames,
                        tile_overlap_in_frames=_overlap_frames,
                    ),
                )
                _frame_index = 0
                for pixels in self.tiled_decode(latent, _tiling):
                    mx.async_eval(pixels)
                    num_frames = pixels.shape[2]
                    for i in range(num_frames):
                        frame = pixels[:, :, i, :, :]  # (B, 3, H, W)
                        frame = mx.clip(frame, -1.0, 1.0)
                        frame = ((frame + 1.0) * 127.5).astype(mx.uint8)
                        # (1, 3, H, W) -> (H, W, 3)
                        frame_hwc = frame[0].transpose(1, 2, 0)
                        mx.eval(frame_hwc)  # must be sync — async_eval can write before data is ready
                        proc.stdin.write(bytes(memoryview(frame_hwc)))
                        del frame, frame_hwc
                        _frame_index += 1
                        if _frame_index % 8 == 0:
                            aggressive_cleanup()
                    del pixels
                    aggressive_cleanup()
            else:
                # Emergency fallback: original full-volume decode.
                pixels = self.decode(latent)
                mx.async_eval(pixels)

                num_frames = pixels.shape[2]
                for i in range(num_frames):
                    frame = pixels[:, :, i, :, :]  # (B, 3, H, W)
                    frame = mx.clip(frame, -1.0, 1.0)
                    frame = ((frame + 1.0) * 127.5).astype(mx.uint8)
                    # (1, 3, H, W) -> (H, W, 3)
                    frame_hwc = frame[0].transpose(1, 2, 0)
                    mx.eval(frame_hwc)  # must be sync — async_eval can write before data is ready
                    proc.stdin.write(bytes(memoryview(frame_hwc)))
                    del frame, frame_hwc
                    if i % 8 == 0:
                        aggressive_cleanup()'''


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


def apply_patch(target: Path, old: str, new: str, marker: str, label: str,
                upgrade_marker: str | None = None) -> str:
    """Idempotently apply old→new replacement on `target`. Returns one of
    OUTCOME_APPLIED / OUTCOME_ALREADY / OUTCOME_MISSING / OUTCOME_DRIFT —
    deep-review fix to surface upstream drift loudly instead of silently
    no-op'ing the patch and shipping a broken install.

    `upgrade_marker` (optional): a substring that exists in the NEW patch
    but not in the OLD already-applied version. If `marker` is found but
    `upgrade_marker` is NOT, an older version of our own patch is on disk
    — re-write the file to the latest content. Used for shipping fixes to
    users who already have an earlier patch applied (e.g. adding +faststart
    to the codec patch without forcing a venv rebuild)."""
    if target is None or not target.exists():
        print(f"  [{label}] MISSING — target file not found", file=sys.stderr)
        return OUTCOME_MISSING
    text = target.read_text()
    if marker in text:
        # Marker present → some version of our patch is on disk. If the
        # caller didn't supply an upgrade_marker we treat it as ALREADY.
        if upgrade_marker is None or upgrade_marker in text:
            print(f"  [{label}] already patched: {target}")
            return OUTCOME_ALREADY
        # Marker but no upgrade_marker → old patch version on disk. The
        # surrounding lines were rewritten by the previous patch, so the
        # OLD raw upstream string isn't there to find. Restore from import
        # of the function's source text by REVERTING our previous patch
        # back to OLD using an embedded undo, then re-apply NEW. We do
        # this by re-reading the upstream commit-pinned source.
        print(f"  [{label}] upgrading older patch: {target}")
        # Naive but works for our patches: re-write the entire file by
        # mapping NEW back to itself isn't useful. Instead, find the
        # specific patched line and replace with NEW — provided NEW
        # contains a unique-enough head/tail anchor. We rely on caller
        # setting OLD to a substring still present after patching.
        # SIMPLER ROUTE: write the upstream OLD string anywhere it would
        # have been. Since we can't easily reconstruct the un-patched
        # form, we instead just edit the patched site-packages directly
        # at runtime via direct Edit — this branch is reachable only
        # for the codec patch where +faststart was added. The narrow
        # marker-pair (old=cmd.extend without faststart, new=cmd.extend
        # with faststart) lets us find and replace.
        # Find the old line (without faststart) and replace with new line.
        old_one_liner = ('cmd.extend(["-c:v", "libx264", "-pix_fmt", _pix, '
                         '"-crf", _crf, output_path])')
        new_one_liner = ('cmd.extend(["-c:v", "libx264", "-pix_fmt", _pix, '
                         '"-crf", _crf,\n                    "-movflags", '
                         '"+faststart", output_path])')
        if old_one_liner in text:
            _atomic_write(target, text.replace(old_one_liner, new_one_liner))
            print(f"  [{label}] upgrade applied: {target}")
            return OUTCOME_APPLIED
        print(
            f"  [{label}] upgrade target text not found — patch shape may have "
            f"changed. Manual inspection needed.", file=sys.stderr,
        )
        return OUTCOME_DRIFT
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


def apply_one_stage_fps_patch(target: Path | None) -> str:
    """Expose frame_rate on the one-stage T2V/I2V pipeline.

    This is a multi-anchor patch because older installs may already have our
    I2V memory patches applied, which changes the exact generate_and_save()
    body. Keep it narrow and verify the final required markers instead of
    replacing one large fragile block.
    """
    label = "one-stage frame_rate (12→24fps long clips)"
    if target is None or not target.exists():
        print(f"  [{label}] MISSING — target file not found", file=sys.stderr)
        return OUTCOME_MISSING
    text = target.read_text()
    required = (
        "frame_rate: float = 24.0",
        "compute_audio_token_count(num_frames, fps=frame_rate)",
        "compute_video_positions(F, H, W, fps=frame_rate)",
        "fps=frame_rate",
    )
    if all(marker in text for marker in required):
        print(f"  [{label}] already patched: {target}")
        return OUTCOME_ALREADY

    original = text
    replacements = [
        (
            "        num_steps: int | None = None,\n"
            "    ) -> tuple[mx.array, mx.array]:",
            "        num_steps: int | None = None,\n"
            "        frame_rate: float = 24.0,\n"
            "    ) -> tuple[mx.array, mx.array]:",
        ),
        (
            "        num_steps: int | None = None,\n"
            "    ) -> str:",
            "        num_steps: int | None = None,\n"
            "        frame_rate: float = 24.0,\n"
            "    ) -> str:",
        ),
        (
            "        audio_T = compute_audio_token_count(num_frames)",
            "        audio_T = compute_audio_token_count(num_frames, fps=frame_rate)",
        ),
        (
            "        video_positions = compute_video_positions(F, H, W)",
            "        video_positions = compute_video_positions(F, H, W, fps=frame_rate)",
        ),
        (
            "            num_steps=num_steps,\n"
            "        )",
            "            num_steps=num_steps,\n"
            "            frame_rate=frame_rate,\n"
            "        )",
        ),
        (
            "return super().generate_and_save(prompt, output_path, height, width, num_frames, seed, num_steps)",
            "return super().generate_and_save(prompt, output_path, height, width, num_frames, seed, num_steps, frame_rate)",
        ),
        (
            "return self._decode_and_save_video(video_latent, audio_latent, output_path)",
            "return self._decode_and_save_video(video_latent, audio_latent, output_path, fps=frame_rate)",
        ),
    ]
    for old, new in replacements:
        text = text.replace(old, new)

    if text == original:
        print(
            f"  [{label}] DRIFT — no fps anchors matched in {target}.",
            file=sys.stderr,
        )
        return OUTCOME_DRIFT
    if not all(marker in text for marker in required):
        missing = [marker for marker in required if marker not in text]
        print(
            f"  [{label}] DRIFT — patch incomplete in {target}; missing {missing}",
            file=sys.stderr,
        )
        return OUTCOME_DRIFT
    _atomic_write(target, text)
    print(f"  [{label}] patched {target}")
    return OUTCOME_APPLIED


def main() -> int:
    print("Applying LTX23MLX patches:")
    outcomes: list[tuple[str, str]] = []

    # Patch 1: codec
    # `upgrade_marker="+faststart"` lets us upgrade installs where the
    # earlier version of this patch was applied (LTX_OUTPUT_PIX_FMT marker
    # present, but the +faststart movflag missing). Without the upgrade
    # path, those installs would never get the moov-at-front fix that lets
    # gallery thumbnails render the first frame without downloading the
    # full clip.
    codec_target = _find("ltx_core_mlx/model/video_vae/video_vae.py")
    outcomes.append(("codec (yuv444p crf 0 + faststart)", apply_patch(
        codec_target, PATCH_CODEC_OLD, PATCH_CODEC_NEW,
        marker="LTX_OUTPUT_PIX_FMT",
        upgrade_marker="+faststart",
        label="codec (yuv444p crf 0 + faststart)",
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

    # Patch 3: free vae_encoder + feature_extractor BEFORE the I2V denoise loop.
    # Same target file as Patch 2; different anchor. Optional like Patch 2 —
    # if dgrauet refactors generate_from_image upstream this becomes a no-op
    # warning instead of a hard install failure.
    pre_outcome = apply_patch(
        i2v_target, PATCH_I2V_PREDENOISE_OLD, PATCH_I2V_PREDENOISE_NEW,
        marker="PATCHED (LTX23MLX, peanut review): free vae_encoder",
        label="I2V free pre-denoise (peanut)",
    )
    if pre_outcome in (OUTCOME_DRIFT, OUTCOME_MISSING):
        print(
            "  [I2V free pre-denoise] note: anchor not found — pipeline may have "
            "been refactored upstream. Generation still works, just without the "
            "memory-overlap cleanup. T2V users unaffected.",
            file=sys.stderr,
        )
        pre_outcome = OUTCOME_ALREADY
    outcomes.append(("I2V free pre-denoise (peanut)", pre_outcome))

    # Patch 4: base TextToVideoPipeline.load() also clears feature_extractor
    # alongside text_encoder before loading the transformer. Affects T2V/I2V/
    # Extend — they all go through the same base load(). Strict subset of the
    # peanut-review fix: even when generate_from_image has its own cleanup,
    # the base load() reload-then-free-then-load-DiT path benefits from the
    # extra clear (no point holding feature_extractor through a 10 GiB DiT load).
    base_outcome = apply_patch(
        i2v_target, PATCH_BASE_LOAD_OLD, PATCH_BASE_LOAD_NEW,
        marker="PATCHED (LTX23MLX, peanut review): also clear feature_extractor",
        label="base load() free feature_extractor (peanut)",
    )
    if base_outcome in (OUTCOME_DRIFT, OUTCOME_MISSING):
        print(
            "  [base load() free feature_extractor] note: anchor not found — "
            "base load() may have been refactored upstream. Memory peak slightly "
            "higher than with this patch but generation is unaffected.",
            file=sys.stderr,
        )
        base_outcome = OUTCOME_ALREADY
    outcomes.append(("base load() free feature_extractor (peanut)", base_outcome))

    # Patch 5: temporal streaming VAE decode. Required for long/high-res renders:
    # without it, decode_and_stream still materializes the whole video tensor and
    # can look like a frozen final step under memory pressure.
    vae_stream_outcome = apply_patch(
        codec_target, PATCH_VAE_STREAM_OLD, PATCH_VAE_STREAM_NEW,
        marker="PATCHED (LTX23MLX): temporal streaming decode",
        label="VAE temporal streaming decode",
    )
    outcomes.append(("VAE temporal streaming decode", vae_stream_outcome))

    # Patch 6: expose frame_rate in the one-stage T2V/I2V path. This is the
    # shippable hook for the panel's explicit Long Clip Boost mode: generate
    # fewer semantic frames at 12fps, keep LTX audio duration aligned, then
    # interpolate the delivered export back to 24fps.
    #
    # Treat upstream drift as ADVISORY (same convention as patches 2/3/4):
    # when dgrauet refactors `ti2vid_one_stage.py`, the anchor disappears
    # and a hard install failure here would block every new user even
    # though the codec patch (the only one that actually affects T2V/I2V
    # correctness) applied. Without this patch, generation still works at
    # the engine's native frame rate; only the panel's explicit "Long Clip
    # Boost (12→24fps)" mode is unavailable until the patch is updated.
    fps_outcome = apply_one_stage_fps_patch(i2v_target)
    if fps_outcome in (OUTCOME_DRIFT, OUTCOME_MISSING):
        print(
            "  [one-stage frame_rate] note: anchor not found — pipeline may "
            "have been refactored upstream. T2V / I2V at native fps work fine; "
            "the panel's Long Clip Boost (12→24fps interpolation) mode is "
            "unavailable until the patch is updated.",
            file=sys.stderr,
        )
        fps_outcome = OUTCOME_ALREADY
    outcomes.append(("one-stage frame_rate (12→24fps long clips)", fps_outcome))

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
