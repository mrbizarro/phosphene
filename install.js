// Phosphene install — idempotent.
//
// Pinokio will re-run this whenever the user clicks "Install" or "Resume
// Install" (the latter fires when env_ready && !base_models_ready, see
// pinokio.js). Every step below is safe to repeat:
//
//   - clone:       skipped if ltx-2-mlx/.git exists
//   - venv:        skipped if env/bin/python3.11 exists (force 3.11)
//   - uv pip:      idempotent (already-installed packages are no-ops)
//   - patch:       patch_ltx_codec.py is idempotent + fails loud on drift
//   - hf download: resumes partial files, skips intact ones
//
// If the user's first install died after the venv was created but before
// the model downloads, hitting Resume Install picks up exactly where it
// left off without re-downloading the working pieces.
//
// IMPORTANT: we do NOT use Pinokio's `venv: "env"` directive to CREATE the
// venv — that uses conda-base's Python (currently 3.10 on the macOS bundle)
// which fails the ltx-core-mlx Python>=3.11 constraint. We force 3.11 with
// `uv venv --python 3.11` and then use `--python env/bin/python` on every
// pip step. The hf download steps still use `venv: "env"` for activation
// only (sourcing the existing 3.11 venv to put `hf` on PATH).

module.exports = {
  // Pulls in `huggingface-cli`/`hf`, `ffmpeg`, `git`, `uv`, `python3.11` etc.
  requires: { bundle: "ai" },
  run: [
    // ---- Apple Silicon gate ------------------------------------------------
    {
      when: "{{platform !== 'darwin' || arch !== 'arm64'}}",
      method: "notify",
      params: {
        html: "<b>Phosphene requires an Apple Silicon Mac (M1 or newer).</b><br>It will not run on Intel Macs, Linux, or Windows."
      },
      next: null
    },

    // ---- Clone ltx-2-mlx (skip if already cloned) -------------------------
    // Re-running install when the clone exists used to fail with
    // "destination path 'ltx-2-mlx' already exists and is not an empty
    // directory", aborting the whole install. Guard with `when:` so the
    // step is a no-op on Resume Install.
    {
      when: "{{!exists('ltx-2-mlx/.git')}}",
      method: "shell.run",
      params: {
        message: ["git clone https://github.com/dgrauet/ltx-2-mlx.git ltx-2-mlx"]
      }
    },

    // ---- ltx-2-mlx version: use HEAD (the original audio bug was MLX 0.31.2,
    //      not anything dgrauet shipped). Earlier we briefly pinned to
    //      dcd639e thinking the audio regression was in dgrauet's commits;
    //      empirical follow-up showed pinning mlx==0.31.1 alone fixes audio
    //      on HEAD. dcd639e was missing several APIs the panel calls
    //      (cfg_scale on extend_from_video, the 0.2.0 I2V structure our OOM
    //      patch targets, split_model.json filename resolution). HEAD with
    //      the mlx pin gives us working audio AND working Extend / I2V.
    //      Leaving this comment as a marker so we don't re-introduce the
    //      pin without re-verifying the audio path. ----

    // ---- Force Python 3.11 venv (SHIP-BLOCKER fix) ------------------------
    // Pinokio's `venv: "env"` shortcut creates a venv using whatever python
    // is on `conda activate base` — on machines where conda's base env is
    // Python 3.10 (the current macOS bundle reality), that venv has no
    // python3.11 and the MLX packages refuse to install:
    //   "ltx-core-mlx==0.2.0 depends on Python>=3.11"
    //
    // Worse, that error doesn't abort the install — Pinokio happily moves on
    // to download 35 GB of weights into a broken venv. So we explicitly
    // create the venv with `uv venv --python 3.11` before any pip step.
    //
    // Idempotency: if env/bin/python3.11 already exists we skip. If env
    // exists but is 3.10 (our exact failure mode), nuke and rebuild — those
    // dirs only hold a wrong-Python venv, no user data.
    {
      when: "{{!exists('ltx-2-mlx/env/bin/python3.11')}}",
      method: "shell.run",
      params: {
        path: "ltx-2-mlx",
        message: [
          "rm -rf env",
          "uv venv --python 3.11 --seed env"
        ]
      }
    },

    // ---- Install MLX pipeline packages into the 3.11 venv -----------------
    // `--python env/bin/python` pins the install to the venv we just made
    // (avoids any conda-base interference). uv pip install is idempotent —
    // already-installed packages are no-ops on Resume Install.
    // Non-editable install (no -e): packages get copied into
    // env/lib/python3.11/site-packages/ which is where patch_ltx_codec.py
    // looks for video_vae.py.
    // Pin huggingface-hub>=1.0 explicitly so older Pinokio bundles still
    // get the v1+ `hf` CLI used by the download steps below.
    //
    // SHIP-BLOCKER: pin mlx==0.31.1 (NOT 0.31.2). LTX 2.3 audio regresses
    // by 22 dB on mlx 0.31.2 — output peaks at -37 dB instead of the
    // expected -9 to -15 dB. Verified empirically by downgrading mlx in a
    // working install and re-running the same prompt:
    //   mlx 0.31.2 → max_volume -42.8 dB (broken)
    //   mlx 0.31.1 → max_volume -9.2  dB (working)
    // Same packages, same weights, same seed; only mlx differs. Numerical
    // change in 0.31.2 attenuates the vocoder output.
    {
      method: "shell.run",
      params: {
        path: "ltx-2-mlx",
        message: [
          // Force the mlx pin BEFORE installing ltx-* packages so their deps
          // resolve to the pinned version instead of pulling latest 0.31.x.
          "uv pip install --python env/bin/python 'mlx==0.31.1' 'mlx-lm==0.31.1' 'mlx-metal==0.31.1'",
          "uv pip install --python env/bin/python ./packages/ltx-core-mlx ./packages/ltx-pipelines-mlx",
          "uv pip install --python env/bin/python pillow numpy 'huggingface-hub>=1.0'"
        ]
      }
    },

    // ---- Apply patches (idempotent, fails loud on upstream drift) ---------
    // Codec → yuv444p crf 0 (lossless), I2V OOM cleanup before VAE decode.
    // Patch script exits non-zero if it can't find expected text — that
    // surfaces upstream-restructure problems instead of silently shipping
    // broken installs to users (deep-review recommendation).
    {
      method: "shell.run",
      params: {
        message: ["./ltx-2-mlx/env/bin/python3.11 patch_ltx_codec.py"]
      }
    },

    // ---- Download Q4 LTX 2.3 (~25 GB, resumable) --------------------------
    // `hf download` is the v1+ name (huggingface_hub deprecated `huggingface-cli`).
    // Resume + verify is built-in; --local-dir avoids the HF cache store so
    // the panel can point at the path directly with no symlink chase.
    // On Resume Install with base files already complete, this is a fast
    // verify pass (~seconds) — `hf` checks each file's hash and skips.
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "ltx-2-mlx",
        message: [
          "hf download dgrauet/ltx-2.3-mlx-q4 --local-dir ../mlx_models/ltx-2.3-mlx-q4"
        ]
      }
    },

    // ---- (no transformer.safetensors symlink needed on HEAD — 0.2.0 reads
    //      split_model.json to resolve transformer-distilled.safetensors.
    //      Symlink was a workaround for the dcd639e pin we no longer use.) ----

    // ---- Download Gemma 4-bit text encoder (~6 GB) ------------------------
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "ltx-2-mlx",
        message: [
          "hf download mlx-community/gemma-3-12b-it-4bit --local-dir ../mlx_models/gemma-3-12b-it-4bit"
        ]
      }
    },

    // ---- Done -------------------------------------------------------------
    {
      method: "notify",
      params: {
        html: "<b>Phosphene installed.</b><br>Click <b>Start</b> to launch the panel, then <b>Open Panel</b>."
      }
    }
  ]
}
