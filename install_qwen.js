// Optional installer for Qwen-Image-Edit-2509 (multi-reference image gen).
//
// What this gives the user:
//   - Compose a character + a place / character + product / character +
//     character into a single still, in seconds, on Apple Silicon.
//   - Powers the agent's `generate_shot_images(refs=[...])` flow for
//     cross-shot identity locking — no LoRA training required.
//   - The new "Image" tab in the panel uses this to drive manual
//     still generation feeding the library.
//
// Backed by:
//   - filipstrand/mflux (Apache 2.0) — `mflux-generate-qwen-edit` CLI
//   - Qwen/Qwen-Image-Edit-2509 weights (Apache 2.0, Alibaba Tongyi Lab)
//
// Disk + time:
//   pip step: ~30 s, ~150 MB.
//   First weight download is lazy — happens on first generation, lands
//   in ~/.cache/huggingface (~22-34 GB depending on quantization). The
//   user picks the quant in Settings → Image generation. We deliberately
//   do NOT pre-pull weights here so this install stays under a minute;
//   the first agent generation takes the longer hit and the panel
//   surfaces a "downloading model…" status during it.
//
// Idempotent:
//   pip install -U is a no-op when mflux is already at >=0.17.5.
//   Re-running this script is safe.
//
// Survives Pinokio Reset:
//   The mflux package is in `ltx-2-mlx/env/` which Reset wipes; user
//   re-runs install_qwen.js after Reset to re-install. Weights in
//   ~/.cache/huggingface are OUTSIDE the install dir and survive Reset
//   automatically — the second install is just the ~30 s pip step.
module.exports = {
  run: [
    {
      method: "notify",
      params: {
        html: [
          "<b>Installing Qwen-Image-Edit-2509 (multi-reference image gen).</b>",
          "<br>Apache 2.0 license. Powers character + place keyframe composition.",
          "<br>This step installs the <code>mflux</code> Python package (~150 MB).",
          "<br>Model weights (~22-34 GB) download lazily on first use to <code>~/.cache/huggingface</code>."
        ].join("")
      }
    },
    {
      method: "shell.run",
      params: {
        // -U upgrades if already installed at older version. The `>=0.17.5`
        // pin is the minimum where `mflux-generate-qwen-edit` ships with
        // the Edit-2509 multi-image fixes (vision encoder save bug
        // resolved in 0.17.5).
        message: "./ltx-2-mlx/env/bin/pip install -U 'mflux>=0.17.5'"
      }
    },
    {
      method: "shell.run",
      params: {
        // Sanity check: confirm the per-family CLI made it onto PATH.
        // Failure here means the pip step did not place the binary —
        // most often a stale cache. The exit-code-only check (no print)
        // matches the v2.0.5 install.js convention to avoid shell
        // rewriters mangling decorative output.
        message: "./ltx-2-mlx/env/bin/mflux-generate-qwen-edit --help > /dev/null"
      }
    },
    {
      method: "notify",
      params: {
        html: [
          "<b>Qwen-Image-Edit-2509 installed.</b>",
          "<br>Open the panel → Settings → Image generation → pick",
          "<br><code>Qwen-Image-Edit-2509 (multi-ref)</code>.",
          "<br>The new <b>Image</b> tab in the panel composes character + place stills",
          "<br>that drop straight into the library for the agent to use as keyframes."
        ].join("")
      }
    }
  ]
}
