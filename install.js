module.exports = {
  // Pulls in `huggingface-cli`, `ffmpeg`, `git`, `uv`, `python3.11` etc.
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

    // ---- Clone ltx-2-mlx (the MLX backend) --------------------------------
    {
      method: "shell.run",
      params: {
        message: ["git clone https://github.com/dgrauet/ltx-2-mlx.git ltx-2-mlx"]
      }
    },

    // ---- Create venv + install MLX pipeline packages ----------------------
    // Pinokio creates ./ltx-2-mlx/env/ and activates it for the message commands.
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "ltx-2-mlx",
        message: [
          "uv pip install -e packages/ltx-core-mlx packages/ltx-pipelines-mlx",
          "uv pip install pillow numpy"
        ]
      }
    },

    // ---- Apply the lossless-h264 codec patch (idempotent) -----------------
    // Switches output codec from yuv420p crf 18 (visible chroma blocks on faces)
    // to yuv444p crf 0 (lossless, no chroma subsampling). Adds env-var override
    // so users can flip back via LTX_OUTPUT_PIX_FMT / LTX_OUTPUT_CRF.
    {
      method: "shell.run",
      params: {
        message: ["python3 patch_ltx_codec.py"]
      }
    },

    // ---- Download the Q4 LTX 2.3 model (~25 GB, resumable) ----------------
    // `hf download` is the v1+ name (huggingface_hub deprecated `huggingface-cli`).
    // Both handle resume + verification natively; --local-dir avoids the HF cache
    // store so the panel can point at the path directly with no symlink chase.
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

    // ---- Download the Gemma 4-bit text encoder (~6 GB) --------------------
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
