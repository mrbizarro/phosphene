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
    // Re-apply the codec + I2V patches against the (possibly updated) upstream.
    // Pin to the venv's python3.11 to match install.js — `python3` on Pinokio
    // hosts isn't guaranteed to be 3.11 (or even present on PATH), and the
    // patch script imports nothing 3.11-specific but we want to fail
    // identically on both code paths.
    {
      method: "shell.run",
      params: { message: "./ltx-2-mlx/env/bin/python3.11 patch_ltx_codec.py" }
    }
  ]
}
