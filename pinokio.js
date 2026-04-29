module.exports = {
  version: "7.0",
  title: "LTX23MLX",
  description: "[MAC ONLY] LTX Video 2.3 generation panel for Apple Silicon — local web UI, batch queue, lossless h264. Built on dgrauet/ltx-2-mlx.",
  icon: "icon.png",
  menu: async (kernel, info) => {
    const installed = info.exists("ltx-2-mlx/env") || info.exists("ltx-2-mlx/.venv")
    const running = {
      install: info.running("install.js"),
      start:   info.running("start.js"),
      update:  info.running("update.js"),
      reset:   info.running("reset.js"),
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
    if (!installed) {
      return [{ default: true, icon: "fa-solid fa-plug", text: "Install", href: "install.js" }]
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
    return [
      { default: true, icon: "fa-solid fa-power-off", text: "Start",   href: "start.js" },
      { icon: "fa-solid fa-film",     text: "Outputs", href: "mlx_outputs?fs=true" },
      { icon: "fa-solid fa-cube",     text: "Models",  href: "mlx_models?fs=true" },
      { icon: "fa-solid fa-image",    text: "Uploads", href: "panel_uploads?fs=true" },
      { icon: "fa-solid fa-rotate",   text: "Update",  href: "update.js" },
      { icon: "fa-regular fa-circle-xmark", text: "Reset", href: "reset.js" },
    ]
  }
}
