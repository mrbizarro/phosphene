module.exports = {
  // On-demand Q8 download for the High quality tier.
  // ~37 GB, resumable. The panel auto-detects when it lands and unlocks
  // the High pill in the Quality picker.
  //
  // Y1.024: this used to be advertised as ~25 GB but actually pulled
  // ~82 GB — dgrauet's Q8 repo hosts THREE transformer variants
  // (transformer-dev, transformer-distilled, transformer-distilled-1.1),
  // TWO distilled-LoRA variants (-384 and -384-1.1), the x1.5 spatial
  // upscaler we don't wire, and the temporal upscaler we don't wire.
  // `hf download` with no filter grabs all of them. Reported by
  // @ContentForAll as "download stalls past 50 GB" — there was no stall;
  // the bundle really was ~82 GB and we were lying about the size.
  // The --include allowlist below pulls only the 8 files the panel
  // loads at runtime; total ~37 GB. Keep in sync with
  // required_files.json → repos[q8].download_include.
  run: [
    {
      method: "notify",
      params: {
        html: "<b>Downloading Q8 model (~37 GB)…</b><br>This is the High quality tier (Q8 two-stage HQ + TeaCache and FFLF keyframing). Resumable if interrupted."
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "ltx-2-mlx/env",
        // Y1.022: HF_HUB_ENABLE_HF_TRANSFER=1 turns on the Rust downloader
        // (~5-10× faster on the 37 GB Q8 bundle). hf_transfer is added by
        // install.js / update.js. Falls back to plain Python downloader
        // with a warning if the package isn't on disk for any reason.
        env: { HF_HUB_ENABLE_HF_TRANSFER: "1" },
        message: [
          "hf download dgrauet/ltx-2.3-mlx-q8 --local-dir mlx_models/ltx-2.3-mlx-q8 --include '*.json' --include 'transformer-dev.safetensors' --include 'connector.safetensors' --include 'ltx-2.3-22b-distilled-lora-384.safetensors' --include 'vae_decoder.safetensors' --include 'vae_encoder.safetensors' --include 'audio_vae.safetensors' --include 'vocoder.safetensors' --include 'spatial_upscaler_x2_v1_1.safetensors'"
        ]
      }
    },
    {
      method: "notify",
      params: {
        html: "<b>Q8 ready.</b><br>The High quality option will unlock automatically in the panel within a few seconds. FFLF keyframing also requires Q8 and is now usable."
      }
    }
  ]
}
