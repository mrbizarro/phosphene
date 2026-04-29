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
        // Point the panel at the Pinokio-managed paths.
        LTX_GEMMA: "{{cwd}}/mlx_models/gemma-3-12b-it-4bit",
        LTX_MODELS_DIR: "{{cwd}}/mlx_models",
        LTX_HELPER_PYTHON: "{{cwd}}/ltx-2-mlx/env/bin/python3.11",
        // Q4 default; users can flip to q8 once that download lands.
        LTX_MODEL: "dgrauet/ltx-2.3-mlx-q4"
      },
      message: ["python mlx_ltx_panel.py"],
      on: [
        { event: "/http:\\/\\/\\S+/", done: true },
        { event: "/error:/i", break: false },
        { event: "/errno/i",  break: false }
      ]
    }
  }, {
    method: "local.set",
    params: { url: "{{input.event[0]}}" }
  }]
}
