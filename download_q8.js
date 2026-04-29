module.exports = {
  // On-demand Q8 download for the High quality tier.
  // ~25 GB, resumable. The panel auto-detects when it lands and unlocks
  // the High pill in the Quality picker.
  run: [
    {
      method: "notify",
      params: {
        html: "<b>Downloading Q8 model (~25 GB)…</b><br>This is the High quality tier (Q8 two-stage HQ + TeaCache and FFLF keyframing). Resumable if interrupted."
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "ltx-2-mlx/env",
        message: [
          "hf download dgrauet/ltx-2.3-mlx-q8 --local-dir mlx_models/ltx-2.3-mlx-q8"
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
