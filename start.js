module.exports = {
  daemon: true,
  run: [{
    method: "shell.run",
    params: {
      // Run the panel from the ltx-2-mlx venv so its stdlib + the helper share
      // one Python install. Panel itself only needs stdlib; the helper subprocess
      // also uses this venv (see LTX_HELPER_PYTHON below).
      venv: "ltx-2-mlx/env",
      env: {
        // Point the panel at the locally-installed weights — never the HF
        // repo id, otherwise the helper triggers a duplicate cache download
        // on first generation even though we just downloaded the model.
        LTX_MODEL: "{{cwd}}/mlx_models/ltx-2.3-mlx-q4",
        LTX_MODEL_HQ: "{{cwd}}/mlx_models/ltx-2.3-mlx-q8",
        LTX_GEMMA: "{{cwd}}/mlx_models/gemma-3-12b-it-4bit",
        LTX_MODELS_DIR: "{{cwd}}/mlx_models",
        LTX_Q8_LOCAL: "{{cwd}}/mlx_models/ltx-2.3-mlx-q8",
        LTX_HELPER_PYTHON: "{{cwd}}/ltx-2-mlx/env/bin/python3.11"
      },
      message: ["python mlx_ltx_panel.py"],
      // SHIP-BLOCKER history: we used to have extra `/errno/i` and `/error:/i`
      // patterns here (cargo-culted from comfy.git's start.js) with break:false
      // intended to just keep the shell running through Python tracebacks. But
      // Pinokio's event handler invokes downstream lookups on EVERY matched
      // event — when the panel ever logged a line containing "Errno" (Python's
      // `OSError [Errno N]` format), Pinokio tried to stat a file literally
      // named "Errno" inside the install dir and surfaced
      //   "ENOENT: no such file or directory, stat '.../phosphene.git/Errno'"
      // even though clicking Start was the trigger, not a real error from us.
      // Removed those patterns. We only need the URL match to advance to step 2.
      // Capture group `(http://...)` mirrors comfy.git's working pattern so
      // input.event[1] is the URL (full line is event[0]).
      on: [
        { event: "/(http:\\/\\/[a-zA-Z0-9.]+:[0-9]+)/", done: true }
      ]
    }
  }, {
    method: "local.set",
    params: { url: "{{input.event[1]}}" }
  }]
}
