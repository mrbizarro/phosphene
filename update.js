module.exports = {
  run: [
    // Note: fs.link is only declared in install.js, not here. New users
    // get the persistent drive on their first Install; users coming from
    // pre-Y1.004 will get it whenever they next reinstall (which is the
    // recommended recovery path anyway — see README). Running fs.link on
    // every Update would technically migrate their existing real folders
    // into the drive without a Reset, but it conflates two concerns
    // (durability of model assets vs. routine code updates) and adds a
    // 36 GB merge step to a flow that should be fast.

    // Resilient + branch-aware pull for the panel repo. Y1.015 made this
    // branch-aware so the same update.js works for both the production
    // panel (tracks `main`) and a local dev panel (tracks `dev`). The
    // currently-checked-out branch is whatever the install was set up
    // with; we pull origin/<that branch> rather than hardcoding `main`.
    //
    // Recovery from divergence (Y1.002) is preserved: if --ff-only
    // refuses, we fall back to reset --hard origin/<branch>. A Pinokio
    // panel install is never expected to carry local commits.
    {
      method: "shell.run",
      params: {
        message: [
          "git fetch origin",
          "BRANCH=$(git rev-parse --abbrev-ref HEAD)",
          "echo \"updating branch: $BRANCH\"",
          "git pull --ff-only origin $BRANCH || (echo 'history diverged from origin (force-push?); falling back to reset --hard' && git reset --hard origin/$BRANCH)",
          "git rev-parse --short HEAD"
        ]
      }
    },
    // Pull ltx-2-mlx HEAD with the same pattern. (We previously pinned
    // to dcd639e thinking audio regressed in dgrauet's commits; turned
    // out the audio bug was in mlx 0.31.2 itself. HEAD has the APIs the
    // panel needs — cfg_scale on extend_from_video, the I2V structure
    // our OOM patch targets, etc.)
    {
      method: "shell.run",
      params: {
        path: "ltx-2-mlx",
        message: [
          "git fetch origin",
          "git checkout main || true",
          "git pull --ff-only origin main || (echo 'ltx-2-mlx history diverged; resetting' && git reset --hard origin/main)",
          "git rev-parse --short HEAD"
        ]
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
    // Y1.022: hf_transfer is HuggingFace's Rust accelerator — 5-10× faster
    // downloads on big repos (notably the ~25 GB Q8 bundle). Existing
    // installs that pre-date Y1.022 don't have it, so a Q8 download was
    // bottlenecked at ~50 KB/s on throttled HF anonymous tier. Update
    // installs it idempotently. The panel sets HF_HUB_ENABLE_HF_TRANSFER=1
    // when invoking hf download; if the package is missing for any reason
    // the hf CLI just emits a warning and falls back to plain Python.
    {
      method: "shell.run",
      params: {
        message: "./ltx-2-mlx/env/bin/pip install --upgrade 'hf_transfer>=0.1.6'"
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
