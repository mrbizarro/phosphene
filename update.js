module.exports = {
  run: [
    // Pull our panel repo
    {
      method: "shell.run",
      params: { message: "git pull" }
    },
    // Pull the ltx-2-mlx upstream
    {
      method: "shell.run",
      params: {
        path: "ltx-2-mlx",
        message: "git pull"
      }
    },
    // Re-apply the codec patch in case upstream changed the file
    {
      method: "shell.run",
      params: { message: "python3 patch_ltx_codec.py" }
    }
  ]
}
