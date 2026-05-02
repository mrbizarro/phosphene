module.exports = {
  // Optional official LTX 2.3 latent spatial upscaler.
  // ~1 GB, resumable. This is separate from Q8 so users can download the
  // lightweight model-backed upscaler asset without committing to the full
  // ~25 GB High-quality bundle.
  run: [
    {
      method: "notify",
      params: {
        html: "<b>Downloading LTX spatial upscaler (~1 GB)…</b><br>This is the official LTX 2.3 x2 latent upscaler. Resumable if interrupted."
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "ltx-2-mlx/env",
        env: { HF_HUB_ENABLE_HF_TRANSFER: "1" },
        message: [
          "hf download dgrauet/ltx-2.3-mlx-q8 spatial_upscaler_x2_v1_1.safetensors --local-dir mlx_models/ltx-2.3-mlx-q8"
        ]
      }
    },
    {
      method: "notify",
      params: {
        html: "<b>LTX upscaler ready.</b><br>The file is on disk for upcoming model-backed upscale experiments. The current 720p export path works immediately through ffmpeg Lanczos."
      }
    }
  ]
}
