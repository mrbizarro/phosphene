module.exports = {
  run: [
    {
      method: "notify",
      params: {
        html: "<b>Installing optional Sharp upscaler.</b><br>PiperSR is AGPL-3.0 software from ModelPiper. Public/commercial redistribution may require visible attribution or separate licensing."
      }
    },
    {
      method: "shell.run",
      params: {
        message: "./ltx-2-mlx/env/bin/pip install 'pipersr==1.0.0'"
      }
    },
    {
      method: "notify",
      params: {
        html: "<b>Sharp upscaler installed.</b><br>Restart Phosphene if the Sharp button is not visible yet."
      }
    }
  ]
}
