// LTX23MLX Pinokio menu.
//
// Uses required_files.json as the single source of truth for what counts
// as "installed". The same file is consumed by mlx_ltx_panel.py (run-time
// completeness checks) and is what install.js / update.js are wired to
// produce — so the menu state never drifts from what the panel will accept.
//
// Three install levels we care about for menu rendering:
//   env_ready          — venv + ltx-2-mlx clone exist (install.js step 1-3)
//   base_models_ready  — Q4 + Gemma fully on disk (install.js step 4-5)
//   q8_ready           — optional Q8 bundle fully on disk (download_q8.js)
//
// Anything less than `env_ready && base_models_ready` means the user can't
// Start — we surface a Resume Install affordance instead of Start. This is
// the SHIP-BLOCKER from the deep review: a network hiccup after venv
// creation used to leave the menu showing Start, with the panel about to
// crash because Q4/Gemma aren't on disk.

const fs = require("fs")
const path = require("path")

function loadRequired(kernelDir) {
  // kernelDir is the install dir Pinokio gives us via info.path. Read the
  // JSON synchronously — Pinokio menus are sync today (info.exists is sync),
  // and this is small (< 1 KB) so blocking on it is fine.
  try {
    const p = path.join(kernelDir, "required_files.json")
    return JSON.parse(fs.readFileSync(p, "utf8"))
  } catch (e) {
    // Treat as completely uninstalled if the manifest is gone — the user is
    // going to need to reinstall anyway.
    return { repos: [], env: { marker_paths: [] }, min_size_bytes: 1024 }
  }
}

function repoComplete(installRoot, repo, minBytes) {
  // A repo is "complete" iff every listed file exists at >= minBytes under
  // its local_dir. Mirrors the Python-side _repo_missing in mlx_ltx_panel.py.
  for (const fname of (repo.files || [])) {
    try {
      const abs = path.join(installRoot, repo.local_dir, fname)
      const st = fs.statSync(abs)
      if (!st.isFile() || st.size < minBytes) return false
    } catch (e) {
      return false
    }
  }
  return true
}

module.exports = {
  version: "7.0",
  title: "LTX23MLX",
  description: "[MAC ONLY] LTX Video 2.3 generation panel for Apple Silicon — local web UI, batch queue, lossless h264. Built on dgrauet/ltx-2-mlx.",
  icon: "icon.png",
  menu: async (kernel, info) => {
    const required = loadRequired(info.path)
    const minBytes = required.min_size_bytes || 1024

    // --- env detection: either Pinokio's `env/` or manual `.venv/` ---
    const env_ready = (required.env.marker_paths || []).some(p => info.exists(p))

    // --- per-repo completeness from the unified manifest ---
    const repos = required.repos || []
    const baseRepos = repos.filter(r => r.kind === "base")
    const q8Repo    = repos.find(r => r.key === "q8")

    const base_ready = baseRepos.length > 0 && baseRepos.every(r => repoComplete(info.path, r, minBytes))
    const q8_ready   = q8Repo ? repoComplete(info.path, q8Repo, minBytes) : false

    // User-content folders persist across Reset (which only removes the venv).
    // Keep their shortcuts visible whenever they exist on disk so users can
    // still recover their renders / models / uploads.
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

    // Running states first — show what's in progress, hide everything else.
    if (running.install)    return [{ default: true, icon: "fa-solid fa-plug",     text: "Installing",                   href: "install.js" }]
    if (running.update)     return [{ default: true, icon: "fa-solid fa-rotate",   text: "Updating",                     href: "update.js" }]
    if (running.reset)      return [{ default: true, icon: "fa-solid fa-eraser",   text: "Resetting",                    href: "reset.js" }]
    if (running.q8download) return [{ default: true, icon: "fa-solid fa-download", text: "Downloading Q8 (~25 GB)",      href: "download_q8.js" }]

    // No env at all → fresh install path. Recovery shortcuts to user content
    // folders if a previous install left files behind.
    if (!env_ready) {
      const m = [{ default: true, icon: "fa-solid fa-plug", text: "Install", href: "install.js" }]
      if (has_outputs) m.push({ icon: "fa-solid fa-film",  text: "Outputs", href: "mlx_outputs?fs=true" })
      if (has_models)  m.push({ icon: "fa-solid fa-cube",  text: "Models",  href: "mlx_models?fs=true" })
      if (has_uploads) m.push({ icon: "fa-solid fa-image", text: "Uploads", href: "panel_uploads?fs=true" })
      return m
    }

    // Env exists but base models aren't fully there → SHIP-BLOCKER fix.
    // Don't show Start — the panel would crash on the first job. Run
    // install.js again (it's idempotent: skips clone + venv if present,
    // re-runs `hf download` which itself resumes any partial files).
    if (!base_ready) {
      const m = [
        { default: true, icon: "fa-solid fa-rotate-right", text: "Resume Install (base models incomplete)", href: "install.js" },
      ]
      if (has_outputs) m.push({ icon: "fa-solid fa-film",  text: "Outputs", href: "mlx_outputs?fs=true" })
      if (has_models)  m.push({ icon: "fa-solid fa-cube",  text: "Models",  href: "mlx_models?fs=true" })
      if (has_uploads) m.push({ icon: "fa-solid fa-image", text: "Uploads", href: "panel_uploads?fs=true" })
      m.push({ icon: "fa-regular fa-circle-xmark", text: "Reset", href: "reset.js" })
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

    // Healthy install — Start path.
    const baseMenu = [
      { default: true, icon: "fa-solid fa-power-off", text: "Start",   href: "start.js" },
      { icon: "fa-solid fa-film",  text: "Outputs", href: "mlx_outputs?fs=true" },
      { icon: "fa-solid fa-cube",  text: "Models",  href: "mlx_models?fs=true" },
      { icon: "fa-solid fa-image", text: "Uploads", href: "panel_uploads?fs=true" },
    ]
    if (!q8_ready) {
      baseMenu.push({ icon: "fa-solid fa-download", text: "Download Q8 (~25 GB) — High quality + FFLF", href: "download_q8.js" })
    }
    baseMenu.push(
      { icon: "fa-solid fa-rotate", text: "Update", href: "update.js" },
      { icon: "fa-regular fa-circle-xmark", text: "Reset", href: "reset.js" },
    )
    return baseMenu
  }
}
