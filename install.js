// Phosphene install — idempotent.
//
// Pinokio will re-run this whenever the user clicks "Install" or "Resume
// Install" (the latter fires when env_ready && !base_models_ready, see
// pinokio.js). Every step below is safe to repeat:
//
//   - clone:       skipped if ltx-2-mlx/.git exists
//   - venv:        skipped if env/bin/python3.11 exists (force 3.11)
//   - uv pip:      idempotent (already-installed packages are no-ops)
//   - patch:       patch_ltx_codec.py is idempotent + fails loud on drift
//   - hf download: resumes partial files, skips intact ones
//
// If the user's first install died after the venv was created but before
// the model downloads, hitting Resume Install picks up exactly where it
// left off without re-downloading the working pieces.
//
// IMPORTANT: we do NOT use Pinokio's `venv: "env"` directive to CREATE the
// venv — that uses conda-base's Python (currently 3.10 on the macOS bundle)
// which fails the ltx-core-mlx Python>=3.11 constraint. We force 3.11 with
// `uv venv --python 3.11` and then use `--python env/bin/python` on every
// pip step. The hf download steps still use `venv: "env"` for activation
// only (sourcing the existing 3.11 venv to put `hf` on PATH).

module.exports = {
  // Pulls in `huggingface-cli`/`hf`, `ffmpeg`, `git`, `uv`, `python3.11` etc.
  requires: { bundle: "ai" },
  run: [
    // ---- Apple Silicon gate ------------------------------------------------
    {
      when: "{{platform !== 'darwin' || arch !== 'arm64'}}",
      method: "notify",
      params: {
        html: "<b>Phosphene requires an Apple Silicon Mac (M1 or newer).</b><br>It will not run on Intel Macs, Linux, or Windows."
      },
      next: null
    },

    // ---- Persistent storage via fs.link (Y1.004+) -------------------------
    // Pinokio's fs.link maps these directories to a virtual drive that
    // lives OUTSIDE the panel install dir, so a Reset (which deletes
    // and re-clones the install) leaves the heavy assets intact. After
    // Reset → Install, fs.link re-creates the symlinks back into the
    // fresh clone and the drive is rediscovered automatically.
    //
    // What's in the drive:
    //   mlx_models/    LTX 2.3 weights (~36 GB), Gemma encoder, LoRAs
    //   mlx_outputs/   generated videos
    //   panel_uploads/ user-uploaded reference images
    //   state/         panel_settings.json, panel_queue.json, panel_hidden.json
    //
    // What's NOT linked:
    //   ltx-2-mlx/env/ — venv has historically been buggy under fs.link
    //                    (Python-version-restricted, pip mismatches);
    //                    rebuilds in ~5 min anyway. Models are the
    //                    expensive thing to lose, not the venv.
    //
    // First-run merge: if real folders already exist with content (e.g.
    // an upgrade from Y1.003-), fs.link merges them INTO the drive
    // before replacing them with symlinks. Idempotent on repeat runs.
    {
      method: "fs.link",
      params: {
        drive: {
          mlx_models:    "mlx_models",
          mlx_outputs:   "mlx_outputs",
          panel_uploads: "panel_uploads",
          state:         "state"
        }
      }
    },

    // ---- Clone ltx-2-mlx (skip if already cloned) -------------------------
    // Re-running install when the clone exists used to fail with
    // "destination path 'ltx-2-mlx' already exists and is not an empty
    // directory", aborting the whole install. Guard with `when:` so the
    // step is a no-op on Resume Install.
    {
      when: "{{!exists('ltx-2-mlx/.git')}}",
      method: "shell.run",
      params: {
        message: ["git clone https://github.com/dgrauet/ltx-2-mlx.git ltx-2-mlx"]
      }
    },

    // ---- ltx-2-mlx version: use HEAD (the original audio bug was MLX 0.31.2,
    //      not anything dgrauet shipped). Earlier we briefly pinned to
    //      dcd639e thinking the audio regression was in dgrauet's commits;
    //      empirical follow-up showed pinning mlx==0.31.1 alone fixes audio
    //      on HEAD. dcd639e was missing several APIs the panel calls
    //      (cfg_scale on extend_from_video, the 0.2.0 I2V structure our OOM
    //      patch targets, split_model.json filename resolution). HEAD with
    //      the mlx pin gives us working audio AND working Extend / I2V.
    //      Leaving this comment as a marker so we don't re-introduce the
    //      pin without re-verifying the audio path. ----

    // ---- Force Python 3.11 venv (SHIP-BLOCKER fix) ------------------------
    // Pinokio's `venv: "env"` shortcut creates a venv using whatever python
    // is on `conda activate base` — on machines where conda's base env is
    // Python 3.10 (the current macOS bundle reality), that venv has no
    // python3.11 and the MLX packages refuse to install:
    //   "ltx-core-mlx==0.2.0 depends on Python>=3.11"
    //
    // Worse, that error doesn't abort the install — Pinokio happily moves on
    // to download 35 GB of weights into a broken venv. So we explicitly
    // create the venv with `uv venv --python 3.11` before any pip step.
    //
    // Idempotency: if env/bin/python3.11 already exists we skip. If env
    // exists but is 3.10 (our exact failure mode), nuke and rebuild — those
    // dirs only hold a wrong-Python venv, no user data.
    {
      when: "{{!exists('ltx-2-mlx/env/bin/python3.11')}}",
      method: "shell.run",
      params: {
        path: "ltx-2-mlx",
        message: [
          // v2.0.3: log the toolchain BEFORE creating the venv so the
          // install log self-documents which Python uv landed on. A user
          // (KTDS) hit a silent "ModuleNotFoundError: ltx_pipelines_mlx"
          // after a green install — the most likely cause was uv falling
          // back to a Python that couldn't install mlx wheels, and we
          // had no log evidence to confirm. These echoes change nothing
          // operationally; they just leave a trail.
          "echo '=== install diagnostics: venv create ==='",
          "which uv && uv --version || echo 'uv NOT FOUND'",
          "which python3.11 && python3.11 --version || echo 'system python3.11 NOT FOUND (uv will try to fetch)'",
          "uname -a",
          "echo '=== /diagnostics ==='",
          "rm -rf env",
          "uv venv --python 3.11 --seed env",
          "echo '=== venv created ==='",
          "ls -la env/bin/python* 2>&1 || echo 'venv create FAILED'",
          "env/bin/python --version || echo 'venv python NOT executable'"
        ]
      }
    },

    // ---- Install MLX pipeline packages into the 3.11 venv -----------------
    // `--python env/bin/python` pins the install to the venv we just made
    // (avoids any conda-base interference). uv pip install is idempotent —
    // already-installed packages are no-ops on Resume Install.
    // Non-editable install (no -e): packages get copied into
    // env/lib/python3.11/site-packages/ which is where patch_ltx_codec.py
    // looks for video_vae.py.
    // Pin huggingface-hub>=1.0 explicitly so older Pinokio bundles still
    // get the v1+ `hf` CLI used by the download steps below.
    //
    // SHIP-BLOCKER: pin mlx==0.31.1 (NOT 0.31.2). LTX 2.3 audio regresses
    // by 22 dB on mlx 0.31.2 — output peaks at -37 dB instead of the
    // expected -9 to -15 dB. Verified empirically by downgrading mlx in a
    // working install and re-running the same prompt:
    //   mlx 0.31.2 → max_volume -42.8 dB (broken)
    //   mlx 0.31.1 → max_volume -9.2  dB (working)
    // Same packages, same weights, same seed; only mlx differs. Numerical
    // change in 0.31.2 attenuates the vocoder output.
    {
      method: "shell.run",
      params: {
        path: "ltx-2-mlx",
        message: [
          // v2.0.3: log Python identity before each pip step. KTDS hit a
          // silent missing-package install and we had nothing in the log
          // to diagnose it. These echoes leave a paper trail of which
          // interpreter is being targeted by --python env/bin/python.
          "echo '=== install diagnostics: pip install ==='",
          "env/bin/python --version || echo 'venv python NOT executable'",
          "env/bin/python -c 'import sys; print(\"sys.executable:\", sys.executable); print(\"sys.path[0]:\", sys.path[0] if sys.path else None)'",
          "echo '=== /diagnostics ==='",
          // Force the mlx pin BEFORE installing ltx-* packages so their deps
          // resolve to the pinned version instead of pulling latest 0.31.x.
          "uv pip install --python env/bin/python 'mlx==0.31.1' 'mlx-lm==0.31.1' 'mlx-metal==0.31.1'",
          "uv pip install --python env/bin/python ./packages/ltx-core-mlx ./packages/ltx-pipelines-mlx",
          // hf_transfer is HuggingFace's Rust-based downloader — 5-10× faster
          // than the default Python downloader for big repos like Q8 (~25 GB).
          // The panel sets HF_HUB_ENABLE_HF_TRANSFER=1 in download envs; if the
          // package is missing the hf CLI falls back gracefully with a warning.
          // litellm: agent's chat client (multi-provider router for OpenAI /
          // Anthropic / Ollama / mlx-lm.server). Pinned to >=1.83.14 — the
          // March 2026 PyPI supply-chain incident affected earlier 1.x
          // releases (stole SSH keys via a poisoned post-install script).
          // See agent/engine.py for routing details. Falls back to stdlib
          // urllib if missing — safe to omit but the loop is less robust.
          "uv pip install --python env/bin/python pillow numpy 'huggingface-hub>=1.0' 'hf_transfer>=0.1.6' 'litellm>=1.83.14'",
          // v2.0.3: post-install confirmation that the local packages
          // actually landed in site-packages. The Y1.034+ patch script's
          // i2v target tolerates a missing ltx_pipelines_mlx — without
          // this echo we'd discover the gap only at panel start time.
          "echo '=== post-pip site-packages check ==='",
          "ls env/lib/python3.11/site-packages/ | grep -E '^(ltx|mlx)' || echo 'WARN: no ltx_*/mlx packages in site-packages'",
          "echo '=== /site-packages check ==='"
        ]
      }
    },

    // ---- Apply patches (idempotent, fails loud on upstream drift) ---------
    // Codec → yuv444p crf 0 (lossless), I2V OOM cleanup before VAE decode.
    // Patch script exits non-zero if it can't find expected text — that
    // surfaces upstream-restructure problems instead of silently shipping
    // broken installs to users (deep-review recommendation).
    {
      method: "shell.run",
      params: {
        message: ["./ltx-2-mlx/env/bin/python3.11 patch_ltx_codec.py"]
      }
    },

    // ---- Sanity-import the pipeline packages (v2.0.2+) --------------------
    // SHIP-BLOCKER guard: at least one user (KTDS, May 4) reported a
    // "ModuleNotFoundError: No module named 'ltx_pipelines_mlx'" after a
    // green Pinokio install — the upstream pip step had silently failed
    // mid-install but the patch script's i2v target check tolerates a
    // missing ltx_pipelines_mlx file (demotes MISSING → ALREADY for that
    // specific patch), so the install reported success and the user only
    // learned about the breakage when they clicked Generate.
    //
    // This step imports both packages explicitly. If either is missing
    // the Python call exits non-zero, Pinokio marks the install step as
    // failed, and the user sees an actionable error instead of a 30 GB
    // download into a broken venv. Idempotent — costs ~300ms on a working
    // install.
    //
    // v2.0.5: stripped the print('venv OK: ...') decoration. KTDS (and one
    // other Twitter user) hit a SyntaxError on v2.0.4 where the literal
    // `OK:` was being mangled out of the Python string by something between
    // install.js and the executed shell line — `OK:` got cut from inside
    // and `OK` got appended after the closing shell quote, so Python saw
    // `...importable')OK` and bailed. The exit code from a successful
    // `import` is already the only success signal `shell.run` needs; the
    // print was decorative. Keeping the line minimal sidesteps whatever the
    // rewriter is doing.
    {
      method: "shell.run",
      params: {
        message: [
          "./ltx-2-mlx/env/bin/python3.11 -c \"import ltx_core_mlx, ltx_pipelines_mlx, mlx\""
        ]
      }
    },

    // ---- Download Q4 LTX 2.3 (~20 GB, resumable) --------------------------
    // `hf download` is the v1+ name (huggingface_hub deprecated `huggingface-cli`).
    // Resume + verify is built-in; --local-dir avoids the HF cache store so
    // the panel can point at the path directly with no symlink chase.
    // On Resume Install with base files already complete, this is a fast
    // verify pass (~seconds) — `hf` checks each file's hash and skips.
    //
    // Y1.024: explicit --include allowlist. dgrauet's Q4 repo hosts multiple
    // transformer variants (transformer-distilled, -distilled-1.1, -dev),
    // duplicate distilled LoRAs (-384, -384-1.1), and the x1.5/temporal
    // upscalers we don't use. Without filters `hf download` grabs the full
    // 56 GB tree; the panel only needs ~20 GB. Keep this list in sync with
    // required_files.json → repos[q4].download_include.
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "ltx-2-mlx",
        // Y1.022: HF_HUB_ENABLE_HF_TRANSFER=1 enables the Rust accelerator,
        // ~5-10× faster on 20 GB. Falls back gracefully if hf_transfer
        // isn't yet on disk (warning + plain Python downloader).
        env: { HF_HUB_ENABLE_HF_TRANSFER: "1" },
        message: [
          "hf download dgrauet/ltx-2.3-mlx-q4 --local-dir ../mlx_models/ltx-2.3-mlx-q4 --include '*.json' --include 'transformer-distilled.safetensors' --include 'connector.safetensors' --include 'vae_decoder.safetensors' --include 'vae_encoder.safetensors' --include 'audio_vae.safetensors' --include 'vocoder.safetensors'"
        ]
      }
    },

    // ---- (no transformer.safetensors symlink needed on HEAD — 0.2.0 reads
    //      split_model.json to resolve transformer-distilled.safetensors.
    //      Symlink was a workaround for the dcd639e pin we no longer use.) ----

    // ---- Download Gemma 4-bit text encoder (~6 GB) ------------------------
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "ltx-2-mlx",
        env: { HF_HUB_ENABLE_HF_TRANSFER: "1" },
        message: [
          "hf download mlx-community/gemma-3-12b-it-4bit --local-dir ../mlx_models/gemma-3-12b-it-4bit"
        ]
      }
    },

    // ---- Done -------------------------------------------------------------
    {
      method: "notify",
      params: {
        html: "<b>Phosphene installed.</b><br>Click <b>Start</b> to launch the panel, then <b>Open Panel</b>."
      }
    }
  ]
}
