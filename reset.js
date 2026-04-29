module.exports = {
  run: [
    // Wipe the ltx-2-mlx clone and the venv inside it. Models stay (huge).
    // Users who want to nuke models too should delete mlx_models/ via the
    // "Models" file-browser entry in the Pinokio sidebar.
    { method: "fs.rm", params: { path: "ltx-2-mlx" } }
  ]
}
