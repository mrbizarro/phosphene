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
