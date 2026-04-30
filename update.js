module.exports = {
  run: [
    // Pull our panel repo (Phosphene fixes from mrbizarro/phosphene)
    {
      method: "shell.run",
      params: { message: "git pull" }
    },
    // Re-pin ltx-2-mlx to dcd639e — DO NOT `git pull` blindly here.
    // upstream's 0.2.0 regressed audio amplitude by 22 dB; until we bisect
    // the actual regression and either patch it or move forward to a
    // post-fix commit, we hold dcd639e steady. install.js does the same.
    // git fetch is safe (just updates remote refs); checkout is idempotent.
    {
      method: "shell.run",
      params: {
        path: "ltx-2-mlx",
        message: ["git fetch origin", "git checkout dcd639e"]
      }
    },
    // Re-apply patches. Codec patch is required; I2V OOM patch is a no-op
    // on dcd639e (older I2V structure) and reports drift gracefully now.
    // Pin to the venv's python3.11 to match install.js — `python3` on
    // Pinokio hosts isn't guaranteed to be 3.11 (or even present on PATH).
    {
      method: "shell.run",
      params: { message: "./ltx-2-mlx/env/bin/python3.11 patch_ltx_codec.py" }
    }
  ]
}
