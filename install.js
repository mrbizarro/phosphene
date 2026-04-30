// LTX23MLX install — idempotent.
//
// Pinokio will re-run this whenever the user clicks "Install" or "Resume
// Install" (the latter fires when env_ready && !base_models_ready, see
// pinokio.js). Every step below is safe to repeat:
//
//   - clone:       skipped if ltx-2-mlx/.git exists
//   - venv:        Pinokio's `venv:` directive is itself idempotent
//   - uv pip:      idempotent (already-installed packages are no-ops)
//   - patch:       patch_ltx_codec.py is idempotent + fails loud on drift
//   - hf download: resumes partial files, skips intact ones
//
// If the user's first install died after the venv was created but before
// the model downloads, hitting Resume Install picks up exactly where it
// left off without re-downloading the working pieces.

module.exports = {
  // Pulls in `huggingface-cli`/`hf`, `ffmpeg`, `git`, `uv`, `python3.11` etc.
  requires: { bundle: "ai" },
  run: [
    // ---- Apple Silicon gate ------------------------------------------------
    {
      when: "{{platform !== 'darwin' || arch !== 'arm64'}}",
      method: "notify",
      params: {
        html: "<b>LTX23MLX requires an Apple Silicon Mac (M1 or newer).</b><br>It will not run on Intel Macs, Linux, or Windows."
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

    // ---- Create venv + install MLX pipeline packages ----------------------
    // Pinokio's `venv: "env"` creates ./ltx-2-mlx/env/ on first run and
    // reuses it on subsequent runs. `uv pip install` is idempotent —
    // already-installed packages are no-ops. Non-editable install (no -e):
    // packages get copied into env/lib/.../site-packages/ which is where
    // patch_ltx_codec.py looks for video_vae.py.
    // Pin huggingface-hub>=1.0 explicitly so older Pinokio bundles still
    // get the v1+ `hf` CLI used by the download steps below.
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "ltx-2-mlx",
        message: [
          "uv pip install ./packages/ltx-core-mlx ./packages/ltx-pipelines-mlx",
          "uv pip install pillow numpy 'huggingface-hub>=1.0'"
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
        html: "<b>LTX23MLX installed.</b><br>Click <b>Start</b> to launch the panel, then <b>Open Panel</b>."
      }
    }
  ]
}
