module.exports = {
  run: [
    // Pull our panel repo (Phosphene fixes from mrbizarro/phosphene)
    {
      method: "shell.run",
      params: { message: "git pull" }
    },
    // Pull ltx-2-mlx HEAD. (We previously pinned to dcd639e thinking audio
    // regressed in dgrauet's commits; turned out the audio bug was in mlx
    // 0.31.2 itself. HEAD has the APIs the panel needs — cfg_scale on
    // extend_from_video, the I2V structure our OOM patch targets, etc.
    // git fetch + checkout main is idempotent.)
    {
      method: "shell.run",
      params: {
        path: "ltx-2-mlx",
        message: ["git fetch origin", "git checkout main", "git pull --ff-only origin main"]
      }
    },
    // Force-downgrade mlx to 0.31.1 — fixes 22 dB audio regression on mlx
    // 0.31.2. Existing users who installed before this commit have mlx 0.31.2
    // and quiet audio; clicking Update reinstalls to the pinned version.
    // --force-reinstall + --no-deps so we change ONLY mlx without disturbing
    // ltx-* / transformers / etc. (some of which depend on mlx>=0.31.0).
    // See install.js for the full diagnostic note.
    {
      method: "shell.run",
      params: {
        message: "./ltx-2-mlx/env/bin/pip install --force-reinstall --no-deps 'mlx==0.31.1' 'mlx-lm==0.31.1' 'mlx-metal==0.31.1'"
      }
    },
    // Re-install ltx-core-mlx + ltx-pipelines-mlx from local packages.
    // Critical for users who hit the dcd639e pin window (commits 157b259
    // through e02e288): their site-packages still has 0.1.0 installed
    // even after `git checkout main` updates the source tree to 0.2.0+.
    // Without this re-install they'd have working source but broken
    // installed code (e.g. ExtendPipeline.extend_from_video missing
    // cfg_scale kwarg). --force-reinstall guarantees overwrite;
    // --no-deps avoids re-resolving (and re-pulling) mlx etc.
    {
      method: "shell.run",
      params: {
        path: "ltx-2-mlx",
        message: "./env/bin/pip install --force-reinstall --no-deps ./packages/ltx-core-mlx ./packages/ltx-pipelines-mlx"
      }
    },
    // Re-apply patches. Codec patch is required; I2V OOM patch is a no-op
    // on dcd639e (older I2V structure) and reports drift gracefully now.
    // Pin to the venv's python3.11 to match install.js — `python3` on
    // Pinokio hosts isn't guaranteed to be 3.11 (or even present on PATH).
    {
      method: "shell.run",
      params: { message: "./ltx-2-mlx/env/bin/python3.11 patch_ltx_codec.py" }
    }
  ]
}
