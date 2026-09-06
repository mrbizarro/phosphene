// Phosphene install — idempotent.
//
// Pinokio will re-run this whenever the user clicks "Install" or "Resume
// Install" (the latter fires when env_ready && !base_models_ready, see
// pinokio.js). Every step below is safe to repeat:
//
//   - clone:       skipped if ltx-2-mlx/.git exists
//   - venv:        reused when its interpreter actually runs; rebuilt when
//                  it doesn't (self-healing — see that step's note)
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

    // ---- STOP LOUDLY if that clone did not land ---------------------------
    // Everything below this point assumes ltx-2-mlx/ exists: the version pin
    // runs with `path: "ltx-2-mlx"`, the venv is built inside it, every pip
    // step targets it. When the clone fails — no network, a VPN/proxy, GitHub
    // unreachable, a full disk — none of those steps abort the run. Each one
    // just spawns a shell into a directory that isn't there, prints nothing
    // useful, and the install marches on for another dozen steps before dying
    // for a reason that has long since scrolled off screen.
    //
    // That silence is what made the v3.5.0 restart loop unreadable: the
    // console filled with anonymous "Starting Shell / Terminated Shell" pairs
    // (Pinokio gives every command in a `message` array its own shell) and the
    // single line that explained anything was thousands of lines up. Fail
    // here, name the cause, and stop.
    {
      when: "{{!exists('ltx-2-mlx/.git')}}",
      method: "notify",
      params: {
        html: "<b>Install stopped: the video engine could not be downloaded.</b><br>Phosphene needs to download <code>github.com/dgrauet/ltx-2-mlx</code> and that step did not finish.<br><br>This is almost always a network problem (no connection, a VPN or proxy, or GitHub blocked) or a full disk. Check your connection and free space, then click <b>Install</b> again — anything already downloaded is kept and will not be downloaded twice."
      },
      next: null
    },

    // ---- ltx-2-mlx version: PIN to v0.14.19 (2026-08-11 catch-up). dgrauet asked on
    //      2026-05-12 to lock onto a tag because he pushes breaking changes
    //      upstream to sync with the official Lightricks repo. Without a
    //      tag pin, every fresh install (and every Update) would pull the
    //      next breaking push and Phosphene would fail to start.
    //
    //      v0.14.19 is the pinned tag against which the current panel +
    //      helper + patch_ltx_codec.py are validated (2026-08-11 catch-up
    //      from v0.14.8, 11 releases). What it brings, in the order it
    //      matters to us:
    //        - 0.14.11 — the AV cross-attention gate is finally read FROM the
    //          checkpoint (`av_ca_timestep_scale_multiplier` 1000, not the
    //          dataclass default 1). Audio/dialogue weighting changes for
    //          every render, toward upstream parity. This also lands
    //          `LTXModelConfig.from_checkpoint_dir()` — a metadata-driven
    //          config path we need for any second model generation.
    //        - 0.14.15 — `frame_rate` forwarded at the A2V/lipdub call sites
    //          (our `_install_a2v_frame_rate_patch` shim is now inert there),
    //          and muxed video is no longer truncated to the shortest stream.
    //        - 0.14.16 — quantized transformers load at ANY group_size, not
    //          just 64; ic-lora dev mode fuses the distilled LoRA correctly.
    //        - 0.14.19 — macOS GPU-watchdog kills are explained instead of
    //          dying cryptically; the DiT is freed before VAE decode in
    //          low-memory mode; the Gemma encoder stops widening a stricter
    //          cache limit.
    //      Weight layout is UNCHANGED by this bump. 0.14.13+ prefers a
    //      versioned `transformer-distilled-*.safetensors` when one exists
    //      and falls back to the unversioned name — our shipped model dirs
    //      have no `-1.1` files (update.js trims them), so resolution is
    //      byte-identical to v0.14.8. No re-download, no manifest change.
    //
    //      STAY PINNED here: do not auto-track upstream main. Tag-bumps are a
    //      deliberate decision — read his release notes, smoke-test the full
    //      modality matrix on dev, bump `_LTX_EXPECTED_VERSION` in
    //      mlx_warm_helper.py, then bump this pin AND update.js in one commit.
    //
    //      Idempotent — works on a fresh clone (already on the cloned
    //      branch's tip) AND on a re-install where the clone exists.
    //      2026-08-12 — THE PIN IS NOW A FORK BUILD, NOT AN UPSTREAM TAG.
    //      Vendored: mrbizarro/ltx-2-mlx `feat/ltx-2.5` @ e6be9d6, which is
    //      v0.14.19 (1192051) plus the LTX-2.5 port — keyframe pos-emb, the
    //      vendored Gemma 4 tower, the Euler-ancestral sampler, the duration
    //      head. dgrauet has no 2.5 branch; if he ports it we drop ours and
    //      go back to a tag.
    //
    //      This is what lets LTX-2.5 render THROUGH the panel. v0.14.19 could
    //      register the 2.5 packs but never load them: it does not build
    //      keyframes_abs_pos_embedding and cannot construct a Gemma 4 tower.
    //
    //      Every 2.5 flag defaults to its 2.3 value, so an unversioned
    //      checkpoint builds exactly what it built before — proven here by
    //      876/22 on the vendored suite, a byte-identical job-dict capture
    //      across 18 form shapes, and THREE real 2.3 draft renders whose mp4s
    //      are sha256-identical to the one recorded before any of this work
    //      (three arms, the last of which renders out of the INSTALLED
    //      packages with no PYTHONPATH override — i.e. the thing a user
    //      actually runs).
    //
    //      What this pin CHANGES, and only on 2.5: the Euler-ancestral sampler
    //      is finally reached (it had zero callers, so every prior 2.5 render
    //      used the 2.3 Euler step), and stage 2 starts at the vendor's 0.85
    //      instead of 2.3's 0.909375. Both are keyed off the checkpoint's own
    //      model_version, which is why 2.3 can be byte-stable across a change
    //      this large.
    //
    //      The packages report `0.14.19+ltx25.2`, and `_LTX_EXPECTED_VERSION`
    //      in mlx_warm_helper.py must equal that string. The local segment is
    //      the ONLY thing distinguishing this tree from upstream v0.14.19 at
    //      runtime — the release segment is deliberately unchanged. Move the
    //      two together.
    //      2026-08-13 (v4.0) — THE PIN IS A TAG, AND IT NO LONGER LIVES IN A
    //      LAUNCHER SCRIPT. Both lanes now call the one implementation in
    //      `scripts/pinokio/ltx_checkout.sh`, which holds the literal:
    //
    //          v0.14.19+ltx25.7   on mrbizarro/ltx-2-mlx
    //
    //      Two reasons, both structural. (1) A SHA on a branch is not a pin:
    //      v3.8.x fetched `feat/ltx-2.5` and checked out a SHA that the branch
    //      head had ALREADY moved past, so one rebase upstream would have
    //      stranded every existing install with an un-fetchable pin and a dead
    //      Update button. (2) A pin written in `update.js` can only ever move
    //      ONE CLICK LATE — Pinokio loads that file before our own step pulls
    //      the repo — so it has to live in the post-pull tree. install.js has
    //      no such constraint but shares the file anyway: before v4.0 the two
    //      lanes carried byte-identical copies of this block, and a duplicated
    //      fix is a fix that half-lands.
    //
    //      2026-08-14 (v4.0.2) — the pin moves to `v0.14.19+ltx25.6` for one
    //      change, and it is the reason image-to-video on 2.5 now animates the
    //      image you gave it: the euler-ancestral step rescaled and re-noised
    //      the pinned conditioning tokens on every intermediate step, so the
    //      clip was composed without them and only the terminal step stamped
    //      the image back in. The sample is now re-composited against the clean
    //      latent, which is the half of ComfyUI's 2.5 i2v graph this port had
    //      not implemented. t2v is sha256-identical across the change and every
    //      Euler lane never reaches the branch.
    //
    //      The packages report `0.14.19+ltx25.6`, and `_LTX_EXPECTED_VERSION`
    //      in mlx_warm_helper.py must equal that string. The local segment is
    //      the ONLY thing distinguishing this tree from upstream v0.14.19 at
    //      runtime — the release segment is deliberately unchanged. Move the
    //      two together. `node scripts/check_ltx_pin.js` is the gate that keeps
    //      this comment, the shell script and the helper constant in agreement.
    {
      method: "shell.run",
      params: {
        message: "bash scripts/pinokio/ltx_checkout.sh"
      }
    },

    // ---- Force Python 3.11 venv (SHIP-BLOCKER fix) ------------------------
    // Pinokio's `venv: "env"` shortcut creates a venv using whatever python is
    // on `conda activate base` — on machines where conda's base env is Python
    // 3.10 (the current macOS bundle reality), that venv has no python3.11 and
    // the MLX packages refuse to install ("ltx-core-mlx depends on
    // Python>=3.11"). Worse, that error doesn't abort the install — Pinokio
    // happily downloads 35 GB of weights into a broken venv. So we create the
    // venv with `uv venv --python 3.11` before any pip step.
    //
    // 3.8.3 — THIS IS THE STEP THAT HUNG PINOKIO (#56, and #50 reopened).
    // It was a "\n"-joined STRING, which kernel/shells.js launch() dispatches
    // as ONE write to the pty: 1,417 chars, the exact number @davidaircloud
    // measured on 3.8.0. Its source lines were all under 110 chars, so v3.6.2's
    // per-line rule reported it clean — the rule was right, the measurement was
    // wrong. The body (self-healing venv probe, macOS preflight, diagnostics,
    // rebuild) now lives in scripts/pinokio/ltx_venv.sh with its full
    // rationale. Gate: scripts/check_pinokio_scripts.js.
    {
      method: "shell.run",
      params: {
        path: "ltx-2-mlx",
        message: "bash ../scripts/pinokio/ltx_venv.sh"
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
    // SHIP-BLOCKER (2026-04): pin mlx==0.31.1 (NOT 0.31.2). LTX 2.3 audio
    // regresses by 22 dB on mlx 0.31.2 — output peaks at -37 dB instead of the
    // expected -9 to -15 dB. Verified empirically by downgrading mlx in a
    // working install and re-running the same prompt:
    //   mlx 0.31.2 → max_volume -42.8 dB (broken)
    //   mlx 0.31.1 → max_volume -9.2  dB (working)
    // Same packages, same weights, same seed; only mlx differs. Numerical
    // change in 0.31.2 attenuates the vocoder output.
    //
    // 2026-08-28 — mlx 0.32.1 and 0.32.2 EVALUATED AND HELD BACK, on MEMORY,
    // not on audio. Recorded here so the next reader does not re-derive it.
    //
    // The audio half of this blocker is 0.31.2-SPECIFIC and does not survive
    // into 0.32. Identical latents through the isolated vocoder differ by
    // 0.000001 dB, and five full renders per arm land at max_volume -1.7 vs
    // -1.8 dB, -9.2 vs -9.3 LUFS, -1.6 vs -1.7 true peak, no time shift, and
    // spectra within 0.1 dB from 250 Hz to 16 kHz. Claimed 22 dB, measured 0.1.
    // The speed is real as well: 1024x576 Q8 121f goes 139.8 s → 133.6 s
    // (-4.5%), all of it in DiT load and VAE decode, denoise a wash.
    //
    // What holds it back is PEAK MEMORY, and it is not a 64 GB-only problem.
    // 0.31.1's footprint is a function of the WORKLOAD; 0.32.x's is a function
    // of the MACHINE — it consumes ~95% of whatever ceiling MLX derives from
    // RAM (0.95 * hw.memsize; the same formula on both versions, so the
    // ceiling — and the appetite — scales down with the Mac, it does not go
    // away). Peak process footprint, /usr/bin/time -l, same prompt/seed/dims,
    // MLX's memory AND cache limits set to the value each Mac would derive:
    //   Q4 768x432 121f    16 GB Mac   17.41 → 20.35 GB  (0.32.2: 19.35)
    //                      32 GB Mac   25.66 → 33.18 GB  = 97% of physical
    //                      64 GB Mac   25.53 → 54.28 GB  = 2.13x
    //   Q8 1024x576 121f   48 GB Mac   39.49 → 49.95 GB  = 97% of physical
    // On this 64 GB box the Q4 render moved system memory pressure from 58% to
    // 84% and added 1.5 GB of swap; the Q8 one reached 87%. plan_memory_policy()
    // in mlx_ltx_panel.py calls a machine "pressured" at 82% and answers by
    // forcing the streamed VAE decode — ~30 s on a 5 s clip, which is more than
    // the 4.5% the move earns. So 0.32 would buy speed and spend more of it.
    //
    // Ruled out, so nobody chases it: Metal residency sets (new in 0.32) are
    // NOT the cause — MLX_RESIDENCY_SET_MAX_PCT=0 reproduces the same footprint
    // to within 300 KB across three runs.
    //
    // THE CACHE POLICY SHIPPED FIRST, ON THIS PIN, AND IT SETTLED THE QUESTION
    // THE OTHER WAY. mlx_warm_helper.py now caps the MLX allocator cache at one
    // eighth of physical RAM (floor 2 GiB, ceiling 8 GiB). Re-running the tier
    // table with that policy applied to BOTH versions — same prompt/seed,
    // /usr/bin/time -l peak footprint, share of the simulated Mac's RAM:
    //
    //   Q4 768x432 121f    0.31.1 today   0.31.1+policy   0.32.1+policy
    //     16 GiB (cap 2)   17.41 (101%)   16.50 ( 96%)    19.91 (116%)  WORSE
    //     32 GiB (cap 4)   25.66 ( 75%)   16.49 ( 48%)    23.40 ( 68%)
    //     64 GiB (cap 8)   25.53 ( 37%)   16.50 ( 24%)    27.87 ( 41%)  WORSE
    //   Q8 1024x576 121f
    //     48 GiB (cap 6)   39.49 ( 77%)   25.39 ( 49%)    42.11 ( 82%)  WORSE
    //
    // Three of the four tiers end up at a HIGHER share of physical RAM than
    // 0.31.1 uses today, and the 48 GiB Comfortable tier — the commonest
    // paying-attention Mac — lands at 81.7%, which is the 82% at which
    // plan_memory_policy() forces the streamed VAE decode, before Chrome and
    // Slack are counted.
    //
    // WHY THE CAP CANNOT RESCUE IT: the extra footprint is not all cache. With
    // the cache capped identically, 0.32.1's ACTIVE memory is +12.0 GB on Q4
    // (26.57 vs 14.53) and +29.2 GB on Q8 (52.23 vs 23.08). Capping an
    // allocator cache cannot reach that. Driving the cache to ZERO does get
    // 0.32.1 under today's numbers (Q8@48 GiB 32.25 GB / 138.37 s; Q4@16 GiB
    // 17.04 GB / 72.59 s) — and at cache 0 the entire speed advantage is gone
    // (-1.1% and -0.5% vs 0.31.1 today, and slower than 0.31.1 WITH the policy).
    // There is no cache setting where 0.32.1 beats 0.31.1+policy on both axes.
    //
    // So the move is REFUSED again, on the same ground, with the mitigation
    // now tried: step 1 of the route took the win the version move was supposed
    // to deliver, and took it on the safe pin. Every cache setting is
    // sha256-IDENTICAL output within a version, so none of this is a quality
    // question — but the two VERSIONS are not byte-identical with each other
    // (144da74a... vs 016d2dd2...), so a future attempt still owes a look at
    // faces, not just a green suite.
    //
    // And the pin does not move alone. scripts/check_post_update.js carries the
    // measurement: moving mlx to 0.32.x while mflux stays at 0.18.0 makes the
    // step-7 mflux resolve walk mlx back DOWN to 0.31.2 — the broken one — on
    // every fresh install and every Update, silently.
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
          //
          // SHIP-BLOCKER (2026-07-10, GitHub #40/#38/#37/#33): also pin
          // transformers <5.13.0. mlx-lm 0.31.1 declares `transformers>=5.0.0`
          // with NO upper bound, so any fresh install after transformers 5.13.0
          // dropped (~Jul 9) pulls 5.13.0 — which breaks mlx_lm.tokenizer_utils.
          // EVERY generation then crashes with "'str' object has no attribute
          // '__module__'": the Gemma text-encoder load silently no-ops ("done in
          // 0.0s") → downstream "Model not loaded. Call load() first." Known-good:
          // 5.7.0 (our validated build) and 5.12.x. Cap it on the SAME resolve as
          // mlx-lm so the constraint sticks. Diagnosed by @saved-j + @xandreau.
          //
          // The Update path enforces this too — scripts/post_update.sh step 2b,
          // a `require` (fatal) step. That sentence used to live here as "uv
          // downgrades an already-installed 5.13.0 on the next Update" and was
          // simply false for a month: nothing in the update path constrained
          // transformers at all, so an existing 5.13.0 survived every Update and
          // the install stayed unable to generate anything. A promise about
          // another file now names the step that keeps it.
          "uv pip install --python env/bin/python 'mlx==0.31.1' 'mlx-lm==0.31.1' 'mlx-metal==0.31.1' 'transformers>=5.0.0,<5.13.0'",
          // Y3 — Train Character ships in 3.0. Without ltx-trainer-mlx in
          // the venv, the trainer subprocess fails at `import yaml` because
          // pyyaml is a transitive dep of ltx-trainer (declared in its
          // pyproject). Codex pre-ship review 2026-05-18 caught this.
          //
          // `--build-constraints ../pip-build-constraints.txt` pins the wheel
          // BUILD backend (hatchling<1.32). Upstream's three pyprojects all
          // declare `readme = "../../README.md"` — a path outside the package
          // dir — which hatchling 1.32.0 turned into a hard error
          // ("Readme path must be within the project directory" →
          // metadata-generation-failed). uv resolves the build backend fresh
          // from PyPI into an isolated env, so from the day 1.32.0 shipped
          // this step failed for every NEW install on every pinned tag. See
          // pip-build-constraints.txt; update.js runs the same uv command
          // (it used to spell it `PIP_CONSTRAINT=`, which modern pip ignores
          // by design — one lane now, one failure mode).
          "uv pip install --python env/bin/python --build-constraints ../pip-build-constraints.txt ./packages/ltx-core-mlx ./packages/ltx-pipelines-mlx ./packages/ltx-trainer",
          // v4.0 — THE SECOND PASS IS THE POINT, and it closes a trap that has
          // shipped since the workspace landed. `ltx-2-mlx` is a uv WORKSPACE:
          // the line above (with deps, without --reinstall) links its members
          // EDITABLE. site-packages gets `_editable_impl_ltx_core_mlx.pth`
          // instead of a copy, so `import ltx_core_mlx` resolves to
          // `packages/ltx-core-mlx/src/...` — the GIT-TRACKED source. The codec
          // patch further down then finds no ltx_core_mlx directory in
          // site-packages and patches the tracked file, which is why a
          // PERFECTLY SUCCESSFUL install ended with
          //     M packages/ltx-core-mlx/src/.../video_vae.py
          // every single time — and why the v3.8.0 pin move hit "your local
          // changes would be overwritten by checkout" for the whole fleet.
          //
          // v3.8.1 made the pin move survive that (reset --hard first) and
          // v3.8.1's own notes filed this as the follow-up: cure the cause, so
          // a FRESH install no longer starts dirty. `--reinstall --no-deps`
          // replaces the .pth links with real copies and re-resolves nothing;
          // it is the exact command update.js has run for many releases, so
          // the end state is one every install already converges to on its
          // first Update. One lane, one runtime shape.
          "uv pip install --python env/bin/python --reinstall --no-deps --build-constraints ../pip-build-constraints.txt ./packages/ltx-core-mlx ./packages/ltx-pipelines-mlx ./packages/ltx-trainer",
          // Auto-caption (Gemma 3 12B via mlx-vlm) needs the mlx-vlm
          // package. Pinned to 0.4.4 — caption_with_gemma.py's import
          // surface (load, generate, prompt_utils.apply_chat_template)
          // is stable at that version. --no-deps so we don't drag in
          // mlx-vlm's heavy default deps (PIL>=10, av, etc. that fight
          // mflux/transformers pins). The runtime imports it lazily so
          // a partial install doesn't break the rest of the panel.
          "uv pip install --python env/bin/python --no-deps 'mlx-vlm==0.4.4'",
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
          //
          // smolagents: Phase 2 of the agent-layer refactor. Powers
          // the optional CodeAgent runtime in agent/runtime_smol.py,
          // selectable per-request via PHOSPHENE_RUNTIME=smol. smolagents
          // pulls transformers as a transitive dep — the huggingface-hub
          // floor is bumped to >=1.5.0 to satisfy transformers' pin.
          // smolagents itself ships with a pessimistic <1.0 hub pin that
          // is empirically benign in practice.
          //
          // The hub pin range we settle on (>=1.5.0,<2.0) satisfies:
          //   - mflux>=0.17.5            wants >=1.1.6,<2.0
          //   - transformers (5.7.0+)    wants >=1.5.0,<2.0
          //   - smolagents 1.24.0        warns about <1.0 but works
          //   - hf download CLI          needs v1+ for the new command name
          // 2026-05-31 review fix (E3): pin `certifi` explicitly. start.js
          // points SSL_CERT_FILE at certifi's cacert.pem (the v3.0.4 fix for
          // the CivitAI CERTIFICATE_VERIFY_FAILED on uv-Python). certifi was
          // only ever a transitive dep — if a future dep change drops it, the
          // SSL_CERT_FILE path vanishes and ALL panel stdlib HTTPS breaks.
          // Naming it here keeps the cert bundle guaranteed-present.
          "uv pip install --python env/bin/python certifi pillow numpy 'huggingface-hub>=1.5.0,<2.0' 'hf_transfer>=0.1.6' 'litellm>=1.83.14' 'smolagents>=1.24.0' 'pywebpush>=2.0'",
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

    // ---- Download LTX-2.5, the DEFAULT generation (~28 GB) ----------------
    // SHIP-BLOCKER, closed 2026-08-12. LTX-2.5 became the default generation
    // while nothing in this file downloaded it: `q4_25` and `gemma4_25` name
    // `mrbizarro/...` HuggingFace repos that DO NOT EXIST (our HF token is
    // read-only, and these packs are our own quantisation of a gated upstream).
    // On the machine that built them the default lane worked; on a FRESH
    // INSTALL there were no weights at all for the generation the panel boots
    // into. That is what these two steps fix.
    //
    // Not `hf download`: the packs are mirrored as assets on a release of the
    // public repo, the same lane the sample-character LoRA takes. Files over
    // GitHub's 2 GiB asset cap are published as 1.9 GB shards;
    // scripts/fetch_pack_release.py downloads them, checks each shard's
    // sha256, reassembles, and only renames a file into place once the
    // whole-file sha256 matches the published manifest. It is resumable and
    // idempotent, so Resume Install re-runs it for the price of a read pass
    // and re-downloads only what is actually missing.
    //
    // Both packs are REQUIRED, not optional. 2.5 conditions on its own Gemma 4
    // fine-tune and cannot use the Gemma 3 encoder above — the mismatch does
    // not raise, it silently encodes wrongly — so the two are fetched together
    // and the install stops if either fails, instead of marching on to a panel
    // that cannot render.
    //
    // 3.8.3: body moved to scripts/pinokio/ltx25_weights.sh. It was a
    // "\n"-joined string — ONE 748-char dispatch, not the short lines the old
    // note here claimed, because that is not what Pinokio writes to the pty.
    {
      method: "notify",
      params: {
        html: "<b>Downloading LTX-2.5 (~27 GB)…</b><br>The engine and its text encoder — this is what the panel renders with. Resumable: if this stops, run Install again and it picks up where it left off.<br><br>A full install is about <b>37 GB</b>: this, the Gemma 3 language model that Enhance and the Storyboard planner run on, and three small control LoRAs. <b>LTX-2.3 is no longer downloaded</b> — it is only needed to TRAIN a character, and the panel offers it in Settings → Models the moment you want it."
      }
    },
    {
      method: "shell.run",
      params: {
        message: "bash scripts/pinokio/ltx25_weights.sh"
      }
    },

    // ---- Image-engine pack (mflux: Ideogram 4 + Qwen-Edit) ----------------
    // 2026-06-13: Ideogram 4 + the visual text-placement editor are headline
    // features now, so the mflux image-engine runner ships by default instead
    // of only behind the optional "Install Qwen-Image-Edit" action. (Reported
    // by cocktailpeanut: token saved + regions drawn but Generate stayed
    // disabled because mflux — and thus mflux-generate-ideogram4 — was absent.)
    // Safe to bundle: install.js already pins huggingface-hub to a range mflux
    // requires, and mflux lives in the same venv as the LTX stack. BEST-EFFORT —
    // a pip hiccup must not fail the core (video) install; the panel surfaces a
    // one-click "Reinstall image engines" path if it didn't land. Two-step
    // (with-deps then --no-deps) mirrors install_qwen.js so transitive deps
    // resolve, then the version is locked + the FBCache patch applied.
    //
    // 3.8.3: body moved to scripts/pinokio/mflux_pack.sh (529-char dispatch).
    // update.js keeps its own near-identical copy inline at 498 — under the
    // ceiling, and deliberately left alone rather than merged here in a hotfix.
    {
      method: "shell.run",
      params: {
        message: "bash scripts/pinokio/mflux_pack.sh"
      }
    },

    // ---- Colorize IC-LoRA (~0.3 GB, un-gated community weights) -----------
    // Powers the Colorize restore mode (B&W clip → color). UN-GATED, so no
    // HF token is needed — but BEST-EFFORT regardless: a network hiccup (or a
    // future gating change) must NEVER fail the core video install. The panel
    // surfaces the same Repair path if the file didn't land, and the worker
    // falls back to resolving the repo id at first use. Lands the file at
    // mlx_models/loras/ic/ (matches required_files.json → repos[ic_colorize]
    // + CURATED_LORAS["colorize"].local_path).
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "ltx-2-mlx",
        env: { HF_HUB_ENABLE_HF_TRANSFER: "1" },
        message: [
          "echo 'Fetching the Colorize IC-LoRA (restore mode, ~0.3 GB, optional)…' && \\",
          "hf download DoctorDiffusion/LTX-2.3-IC-LoRA-Colorizer --local-dir ../mlx_models/loras/ic --include 'LTX-2.3-22b-IC-LoRA-Colorizer-0.9.safetensors' \\",
          "|| echo 'WARN: Colorize IC-LoRA fetch failed — video + image still work; the Colorize mode will fetch it on first use, or click Repair.'"
        ].join("\n")
      }
    },

    // ---- Ingredients IC-LoRA (~1.3 GB, un-gated mirror) -------------------
    // Powers the flagship Ingredients mode (2-8 refs → one composed clip).
    // The official Lightricks weight is GATED; DeepBeepMeep/LTX-2 carries the
    // BYTE-IDENTICAL file un-gated, so no HF token is needed. BEST-EFFORT: a
    // hiccup must never fail the core video install. CRITICAL: --include pulls
    // ONLY the one ingredients file — DeepBeepMeep/LTX-2 is a ~708 GB mega-repo,
    // so a bare `hf download` of it would be catastrophic. The worker falls
    // back to a targeted single-file fetch at first use; the panel surfaces
    // Repair. Lands the file at mlx_models/loras/ic/ (matches
    // required_files.json → repos[ic_ingredients] +
    // CURATED_LORAS["ingredients"].local_path).
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "ltx-2-mlx",
        env: { HF_HUB_ENABLE_HF_TRANSFER: "1" },
        message: [
          "echo 'Fetching the Ingredients IC-LoRA (multi-reference mode, ~1.3 GB, optional)…' && \\",
          "hf download DeepBeepMeep/LTX-2 --local-dir ../mlx_models/loras/ic --include 'ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors' \\",
          "|| echo 'WARN: Ingredients IC-LoRA fetch failed — video + image still work; the Ingredients mode will fetch it on first use, or click Repair.'"
        ].join("\n")
      }
    },

    // ---- Control (Union) IC-LoRA (~0.65 GB, OFFICIAL + un-gated) ----------
    // Powers the Control mode (drive motion/structure/composition from a
    // control video). This is the OFFICIAL Lightricks weight AND it is UN-GATED
    // + public, so — unlike Ingredients — no token, no mirror, no mega-repo
    // workaround: a plain single-file `hf download --include`, exactly like the
    // Colorize fetch above. BEST-EFFORT: a hiccup must never fail the core
    // video install. The worker falls back to resolving the repo id at first
    // use; the panel surfaces Repair. Lands the file at mlx_models/loras/ic/
    // (matches required_files.json → repos[ic_union_control] +
    // CURATED_LORAS["union-control"].local_path).
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "ltx-2-mlx",
        env: { HF_HUB_ENABLE_HF_TRANSFER: "1" },
        message: [
          "echo 'Fetching the Control IC-LoRA (Union, control mode, ~0.65 GB, optional)…' && \\",
          "hf download Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control --local-dir ../mlx_models/loras/ic --include 'ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors' \\",
          "|| echo 'WARN: Control IC-LoRA fetch failed — video + image still work; the Control mode will fetch it on first use, or click Repair.'"
        ].join("\n")
      }
    },

    // ---- Done -------------------------------------------------------------
    {
      method: "notify",
      params: {
        html: "<b>Phosphene installed</b> — video + image generation (incl. Ideogram 4).<br>Click <b>Start</b> to launch the panel, then <b>Open Panel</b>."
      }
    }
  ]
}
