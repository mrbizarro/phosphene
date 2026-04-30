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

    // ---- Pin ltx-2-mlx to dcd639e (SHIP-BLOCKER fix) ----------------------
    // SHIP-BLOCKER history: dgrauet/ltx-2-mlx upstream made 100+ commits
    // between the 0.1.0 baseline and the 0.2.0 release HEAD. Several
    // touched the audio path (570cce8, ca784dd, 22adef1) and empirically
    // produced -22 dB output amplitude — peaks at -37 dB vs the working
    // baseline's -15 dB peaks. The "fixes" claim to align MLX numerics with
    // PyTorch reference, but the LTX 2.3 weights were trained against the
    // pre-fix MLX behavior, so post-fix output is quieter / hissy.
    //
    // dcd639e is the original "complete MLX port" commit at version 0.1.0,
    // which the user verified produces the expected audio levels. We pin
    // the working tree there explicitly. Bisecting the 100-commit range
    // for the actual regression is post-launch work.
    //
    // Idempotency: `git checkout` is safe to re-run on every install /
    // resume install — already-on-dcd639e is a no-op.
    {
      method: "shell.run",
      params: {
        path: "ltx-2-mlx",
        message: ["git fetch origin", "git checkout dcd639e"]
      }
    },

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
    {
      method: "shell.run",
      params: {
        path: "ltx-2-mlx",
        message: [
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
