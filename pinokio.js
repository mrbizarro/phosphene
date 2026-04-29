module.exports = {
  version: "7.0",
  title: "LTX23MLX",
  description: "[MAC ONLY] LTX Video 2.3 generation panel for Apple Silicon — local web UI, batch queue, lossless h264. Built on dgrauet/ltx-2-mlx.",
  icon: "icon.png",
  menu: async (kernel, info) => {
    const installed = info.exists("ltx-2-mlx/env") || info.exists("ltx-2-mlx/.venv")
    // Treat Q8 as installed only when the heavy + downstream-required files
    // are all present. transformer-dev alone is not enough: HQ stage-2 needs
    // the distilled LoRA, two-stage upscale needs the spatial upscaler.
    // A partial Q8 download otherwise hides the "Download Q8" menu item.
    const q8_installed = (
      info.exists("mlx_models/ltx-2.3-mlx-q8/transformer-dev.safetensors") &&
      info.exists("mlx_models/ltx-2.3-mlx-q8/ltx-2.3-22b-distilled-lora-384.safetensors") &&
      info.exists("mlx_models/ltx-2.3-mlx-q8/spatial_upscaler_x2_v1_1.safetensors") &&
      info.exists("mlx_models/ltx-2.3-mlx-q8/connector.safetensors")
    )
    // User-content folders persist across Reset (which only removes the
    // venv). Keep their shortcuts visible whenever they exist on disk
    // so users can still recover their renders / models / uploads.
    const has_outputs = info.exists("mlx_outputs")
    const has_models  = info.exists("mlx_models")
    const has_uploads = info.exists("panel_uploads")
    const running = {
      install:    info.running("install.js"),
      start:      info.running("start.js"),
      update:     info.running("update.js"),
      reset:      info.running("reset.js"),
      q8download: info.running("download_q8.js"),
    }

    if (running.install) {
      return [{ default: true, icon: "fa-solid fa-plug", text: "Installing", href: "install.js" }]
    }
    if (running.update) {
      return [{ default: true, icon: "fa-solid fa-rotate", text: "Updating", href: "update.js" }]
    }
    if (running.reset) {
      return [{ default: true, icon: "fa-solid fa-eraser", text: "Resetting", href: "reset.js" }]
    }
    if (running.q8download) {
      return [{ default: true, icon: "fa-solid fa-download", text: "Downloading Q8 (~25 GB)", href: "download_q8.js" }]
    }
    if (!installed) {
      // Even when uninstalled, expose user-content folders if they survived
      // a previous Reset so users can recover their work.
      const m = [{ default: true, icon: "fa-solid fa-plug", text: "Install", href: "install.js" }]
      if (has_outputs) m.push({ icon: "fa-solid fa-film",  text: "Outputs", href: "mlx_outputs?fs=true" })
      if (has_models)  m.push({ icon: "fa-solid fa-cube",  text: "Models",  href: "mlx_models?fs=true" })
      if (has_uploads) m.push({ icon: "fa-solid fa-image", text: "Uploads", href: "panel_uploads?fs=true" })
      return m
    }
    if (running.start) {
      const local = info.local("start.js")
      if (local && local.url) {
        return [
          { default: true, icon: "fa-solid fa-rocket", text: "Open Panel", href: local.url },
          { icon: "fa-solid fa-terminal", text: "Terminal",   href: "start.js" },
          { icon: "fa-solid fa-film",     text: "Outputs",    href: "mlx_outputs?fs=true" },
          { icon: "fa-solid fa-cube",     text: "Models",     href: "mlx_models?fs=true" },
          { icon: "fa-solid fa-image",    text: "Uploads",    href: "panel_uploads?fs=true" },
        ]
      }
      return [{ default: true, icon: "fa-solid fa-terminal", text: "Terminal", href: "start.js" }]
    }
    const baseMenu = [
      { default: true, icon: "fa-solid fa-power-off", text: "Start",   href: "start.js" },
      { icon: "fa-solid fa-film",     text: "Outputs", href: "mlx_outputs?fs=true" },
      { icon: "fa-solid fa-cube",     text: "Models",  href: "mlx_models?fs=true" },
      { icon: "fa-solid fa-image",    text: "Uploads", href: "panel_uploads?fs=true" },
    ]
    if (!q8_installed) {
      baseMenu.push({ icon: "fa-solid fa-download", text: "Download Q8 (~25 GB) — High quality + FFLF", href: "download_q8.js" })
    }
    baseMenu.push(
      { icon: "fa-solid fa-rotate", text: "Update", href: "update.js" },
      { icon: "fa-regular fa-circle-xmark", text: "Reset", href: "reset.js" },
    )
    return baseMenu
  }
}
