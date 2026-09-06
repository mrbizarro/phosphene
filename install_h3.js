// Phosphene — optional Hailuo H3 engine pack. Idempotent.
//
// H3 (MiniMax-H3 FL2VA) is Phosphene's SECOND video engine: one prompt in,
// video + synced dialogue + sound out. It is deliberately NOT part of the base
// install:
//
//   * ~75 GB of weights on top of the ~36 GB base — most users never want it.
//   * Memory: the full bf16 engine peaks at ~42.8 GiB and wants a 60 GB+ Mac;
//     the compact Q8 engine built at the end of this script peaks at 25.63 GiB
//     and runs from 36 GB. Q8 has been the AUTO default on every machine since
//     v4.8.0, so 36 GB is the real floor — and it is the number
//     scripts/pinokio/h3_preflight.sh enforces and H3_MIN_RAM_GB_Q8 in
//     mlx_ltx_panel.py gates on. (This said "only runs on 64 GB+ Macs" for two
//     releases after that stopped being true.)
//   * MiniMax Community License — territory restrictions apply, so it must be
//     an explicit opt-in, not something that lands with the app.
//
// It also gets its OWN venv rather than sharing ltx-2-mlx/env, because the two
// engines disagree about MLX: Phosphene pins mlx==0.31.1 (0.31.2 regresses LTX
// audio by 22 dB), while the H3 port needs mlx>=0.32. Separate venvs mean an
// H3 install can never break LTX rendering, and removing H3 is `rm -rf`.
//
// Everything below is safe to re-run, and re-running it is ALSO the repair
// path — the Pinokio menu points "Repair Hailuo H3" at this same script:
//   clone   — skipped when minimax-h3-mlx/.git exists
//   venv    — reused when its interpreter actually runs; rebuilt when it
//             doesn't (see the self-healing note on that step)
//   uv pip  — already-installed packages are no-ops
//   weights — hf_hub_download resumes partial files, skips intact ones
// So a user whose engine broke re-runs this and pays ~2 minutes, not 75 GB.
//
// The panel discovers the result on its next /status tick (a couple of
// seconds) and unlocks the engine picker. No restart.

// The branch carrying the validated staged runner (scripts/generate_staged.py).
// NOTE: `--first-frame` (FL2VA image conditioning, i.e. Image mode on H3)
// landed AFTER this branch was published. The panel probes the installed
// runner for the flag and keeps Image mode on LTX when it's absent, so an
// older checkout degrades to Text-only instead of failing mid-render. Bump
// this pin when the first-frame work is published.
// v2 (2026-08-10): --lora (single adapter slot), TAE draft decode, native
// single-window 10s/15s drafts, Stage-A latent caching, the streaming Q8
// quantizer for 48 GB Macs, and the LoRA-on-quantized logical-dims fix.
// Published as a NEW branch: the histories diverged, and force-moving a
// branch users' installs already track is how updates break mid-clone.
const H3_BRANCH = "codex/h3-engine-v2"

module.exports = {
  requires: { bundle: "ai" },
  run: [
    {
      method: "notify",
      params: {
        // "Needs a 64 GB+ Apple Silicon Mac" until v4.8.0 made the compact Q8
        // engine the default everywhere and dropped the real floor to 36 GB.
        // The preflight two steps down enforces 36; this sentence was turning
        // away 36-48 GB Macs the installer would have happily served.
        html: "<b>Installing Hailuo H3 (second video engine, ~75 GB).</b><br>Generates picture, dialogue and sound together, and sits beside LTX in the engine switcher. Needs an Apple Silicon Mac with 36 GB+ of memory (the compact Q8 engine, built at the end of this install); 60 GB+ additionally unlocks the full bf16 quality engine. MiniMax Community License — territory restrictions apply. Resumable if interrupted."
      }
    },

    // ---- Hardware preflight ------------------------------------------------
    // Fail FAST and readably. Without this the install spends an hour pulling
    // 75 GB onto a machine that can never render with it — the panel would
    // then correctly refuse every H3 job and the user would rightly be angry.
    // Fail-open by design: if sysctl is missing or unparseable we proceed,
    // because a preflight that can't read the hardware must never be the thing
    // that blocks an otherwise-fine install.
    {
      method: "shell.run",
      params: {
        message: [
          // 3.8.3: the if/else was a "\n"-joined STRING, so Pinokio wrote all
          // 845 chars to the pty in one go (#56 / #50). Body — with the
          // fail-open rationale and the 36e9 floor — now in
          // scripts/pinokio/h3_preflight.sh. Gate: scripts/check_pinokio_scripts.js.
          "bash scripts/pinokio/h3_preflight.sh",
          "echo 'free disk space:'",
          "df -h ."
        ]
      }
    },

    // ---- Clone the engine --------------------------------------------------
    // Guarded: `git clone` into an existing directory fails and would abort
    // the whole install on a re-run / resume.
    {
      when: "{{!exists('minimax-h3-mlx/.git')}}",
      method: "shell.run",
      params: {
        message: [
          "git clone --branch " + H3_BRANCH + " https://github.com/mrbizarro/minimax-h3-mlx.git minimax-h3-mlx"
        ]
      }
    },

    // Pin to the branch on every run, so Resume Install after a partial clone
    // lands on the same code the panel was validated against.
    //
    // v4.0 — TWO BUGS OF THE SAME CLASS AS THE LTX LANE, both filed in the
    // v3.8.1 hotfix notes (§9.3) and the 2026-08-13 sweep (§8.6) and fixed here.
    //
    // 1. THE FETCH NEVER ADVANCED ANYTHING. `git fetch origin <branch>` writes
    //    FETCH_HEAD and updates `refs/remotes/origin/<branch>`; the following
    //    `git checkout <branch>` resolves the LOCAL branch, which the clone
    //    created once and which nothing has moved since. So every H3 user has
    //    been pinned to whatever the branch tip was on the day they first
    //    installed, and re-running the installer could never bring them
    //    forward. `--force -B <branch> FETCH_HEAD` makes the local branch point
    //    at what was actually fetched, which is what "pin to the branch on
    //    every run" always claimed to do.
    //
    // 2. A DIRTY TREE BLOCKED THE CHECKOUT, silently, the same way it killed
    //    the LTX pin move for the whole fleet on 3.8.0. Nothing of ours patches
    //    this tree today, which is exactly why it is worth guarding now rather
    //    than after the first user reports it: `minimax-h3-mlx/` is
    //    app-managed, its weights live outside it in ../mlx_models/hailuo-h3,
    //    and its venv is ignored, so reset --hard costs nothing and cannot
    //    touch user content. Same guard shape, same reasoning, one class.
    //    The guard tests `minimax_h3_mlx/` — the package directory, verified
    //    present at the branch tip; this repo has no root pyproject.toml, so
    //    the LTX lane's marker would have been wrong here.
    //
    // CONSEQUENCE, STATED RATHER THAN BURIED: an existing H3 user who re-runs
    // the installer now MOVES FORWARD to the branch tip instead of staying on
    // their install-day snapshot. That is what the step always said it did.
    // It is safe because the panel PROBES the installed runner for its
    // capabilities (`h3_supports_chain()`, `h3_supports_chain_prompts()`)
    // rather than assuming them, so a runner that gains or loses a flag is
    // handled by the UI instead of dying in argparse 30 s into a render.
    {
      method: "shell.run",
      params: {
        path: "minimax-h3-mlx",
        message: [
          // The pin itself lives in scripts/pinokio/h3_checkout.sh (#74: Update
          // must make the same move; one literal, two callers). H3_BRANCH above
          // is only the fresh-clone branch and must match it.
          "bash ../scripts/pinokio/h3_checkout.sh \"$(pwd)\"",
          "git rev-parse --short HEAD"
        ]
      }
    },

    // ---- Its own Python 3.11 venv -----------------------------------------
    // Same reasoning as install.js: Pinokio's `venv:` shortcut builds from
    // conda-base (currently 3.10 on the macOS bundle), which can't install the
    // MLX wheels. Force 3.11 with uv, then target it explicitly on every pip
    // step. If a wrong-Python venv is already there, nuke it — it holds no
    // user data.
    //
    // SELF-HEALING, and deliberately NOT gated on `when: !exists(...)`.
    // `uv venv` builds the interpreter as a symlink chain into Pinokio's
    // SHARED managed Python:
    //     .venv/bin/python3.11 -> python
    //     .venv/bin/python     -> <pinokio>/cache/XDG_DATA_HOME/uv/python/
    //                             cpython-3.11-macos-aarch64-none/bin/python3.11
    // That target is Pinokio's, not ours. Any other pack install (or any other
    // Pinokio app) that makes uv re-resolve, bump or prune the managed
    // interpreter leaves this chain DANGLING — the engine stops resolving even
    // though the venv, the clone and all ~75 GB of weights are untouched. That
    // is the v3.4.0 "installed other packs and Hailuo H3 vanished" report.
    //
    // A path-existence `when:` guard cannot fix that reliably: depending on
    // whether the check stats the link or its target, a dangling chain reads
    // as either present (repair silently skipped — the failure mode we must
    // never ship) or absent. So we ask the only question that actually
    // matters — does this interpreter RUN? — inside the shell, and rebuild
    // only when it doesn't. Healthy installs pay ~50 ms and skip the rebuild;
    // broken ones self-heal in ~2 minutes without re-downloading a byte.
    {
      method: "shell.run",
      params: {
        path: "minimax-h3-mlx",
        message: [
          "echo '=== H3 venv check ==='",
          [
            "if .venv/bin/python -c 'import sys' >/dev/null 2>&1; then",
            "  echo 'H3 venv healthy - reusing it'",
            "else",
            "  echo 'H3 venv missing or broken (dangling interpreter) - rebuilding, no weights are re-downloaded'",
            "  which uv && uv --version || echo 'uv NOT FOUND'",
            "  rm -rf .venv",
            "  uv venv --python 3.11 --seed .venv",
            "  .venv/bin/python --version || echo 'venv python NOT executable'",
            "fi",
          ].join("\n")
        ]
      }
    },

    {
      method: "shell.run",
      params: {
        path: "minimax-h3-mlx",
        message: [
          "echo '=== H3 dependencies ==='",
          "uv pip install --python .venv/bin/python -r requirements.txt",
          ".venv/bin/python -c \"import mlx.core as mx, numpy, PIL; print('H3 deps OK')\""
        ]
      }
    },

    // ---- Weights (~75 GB) --------------------------------------------------
    // The repo's own scripts/download_selected.py is the source of truth for
    // WHICH files the staged runner loads (pruned bf16 DiT + the Q8 compact
    // components + the upstream text-encoder config). Calling it instead of
    // re-listing filenames here means the manifest can never drift.
    //
    // It appends `models/` to whatever --root it is given, so the components
    // land at mlx_models/hailuo-h3/models/{deepbeep-pruned-bf16,ddalcu-q8,
    // upstream-meta}/. The panel resolves BOTH that shape and the flat one
    // (see _h3_model_roots in mlx_ltx_panel.py), so no post-download move is
    // needed — and a resumed install never has to reconcile a half-moved tree.
    //
    // mlx_models/ is fs.link-mapped to Pinokio's virtual drive, so these 75 GB
    // survive a Reset like every other weight.
    //
    // HF_HOME is declared explicitly (shell.run replaces the env wholesale, so
    // Pinokio's global ENVIRONMENT value doesn't reach the child). With
    // local_dir downloads, huggingface_hub writes straight to the destination
    // instead of duplicating into the hub cache.
    {
      method: "shell.run",
      params: {
        path: "minimax-h3-mlx",
        env: {
          HF_HOME: "{{cwd}}/cache/HF_HOME",
          HF_XET_HIGH_PERFORMANCE: "1"
        },
        message: [
          ".venv/bin/python scripts/download_selected.py --root '{{cwd}}/mlx_models/hailuo-h3'"
        ]
      }
    },

    // ---- Turbo v1.0 release asset (digest-checked fetch) -------------------
    // Target: lightx2v_v1.0_768p_ourlayout.safetensors — the runner-layout
    // repack, published as a release asset on the weights-ltx25-v1 lane,
    // digest-pinned. Provenance: repacked from
    // lightx2v/Minimax-h3-Turbo (Apache-2.0),
    // minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors (source SHA-256
    // 1bdabc2e9fce20b1db563b96bcf6e46adcad4c1964f423676436bf266cc7416c).
    // Do not substitute the raw v0.1 file
    // `minimax_h3_fl2v_turbo_4step_v0.1.safetensors`: its alpha/rank scale
    // lives OUTSIDE the checkpoint and at `--lora ...:1.0` it renders
    // coloured noise. The ONLY accepted v0.1 fallback is the alpha-folded
    // `lightx2v_v0.1_ourlayout_alpha8.safetensors`.
    //
    // BEST-EFFORT ON PURPOSE: the panel's /h3/turbo/install endpoint fetches
    // this same asset on demand with the same digest check, so a transient
    // network failure here must not fail a 75 GB install that is otherwise
    // complete. scripts/fetch_h3_turbo.py resumes partials, verifies
    // SHA-256 d51d626fe0845da7e5845a47c323cf3f29086d44d24cb1a4b980882488746197
    // and renames into place only after the check — a corrupt transfer is
    // deleted, never installed. (One short line per command: Pinokio 8.0.x
    // wedges its shell on ~350+ character lines.)
    {
      method: "shell.run",
      params: {
        path: "minimax-h3-mlx",
        message: [
          ".venv/bin/python '{{cwd}}/scripts/fetch_h3_turbo.py' --dir '{{cwd}}/mlx_models/hailuo-h3/models/turbo-lora' || echo 'TURBO FETCH FAILED - panel offers a one-click retry'"
        ]
      }
    },

    // ---- TAE: the fast draft decoder (~23 MB) ------------------------------
    // H3's video decoder is a 36-layer ViT and it is the biggest FIXED cost in
    // a render — 88 s of a 501 s clip, 77% of everything that isn't denoising.
    // madebyollin's TAE replaces it on DRAFT tiers only and takes a 640x384 5 s
    // draft from ~5 min to ~66 s. 23 MB against a 75 GB pack, so it is not
    // worth making optional; the panel offers the fast path only when this file
    // AND the runner flag are both present, and every delivery tier keeps the
    // untouched full VAE regardless.
    {
      method: "shell.run",
      params: {
        path: "minimax-h3-mlx",
        env: {
          HF_HOME: "{{cwd}}/cache/HF_HOME",
          HF_XET_HIGH_PERFORMANCE: "1"
        },
        message: [
          "mkdir -p '{{cwd}}/mlx_models/hailuo-h3/models/tae'",
          // The HF repo this used to come from (madebyollin/taeh3) was deleted
          // in 2026-09 and every H3 install died here with "Repository Not
          // Found" (Pinokio, @macstephen). The script fetches the same bytes
          // from a pinned commit of the author's GitHub repo, verifies the
          // sha256, and keeps HF as a fallback.
          ".venv/bin/python ../scripts/pinokio/h3_fetch_tae.py '{{cwd}}/mlx_models/hailuo-h3/models/tae/taeh3.safetensors'"
        ]
      }
    },

    // ---- Reduced-RAM engine (every Mac): build the Q8 DiT locally ----------
    // The Q8 pack halves the render peak (27.3 vs 42.8 GiB measured), which is
    // what makes H3 possible on this class of machine at all. Built locally by
    // quantize_stream.py — bounded memory (never holds more than one tensor,
    // CPU stream, deterministic), so the build itself runs fine on the same
    // 36 GB machine that could never load the model whole. ~22 GB on disk,
    // ~5 min.
    //
    // EVERY MACHINE BUILDS IT SINCE v4.8.0. This used to be skipped on 64 GB+
    // Macs ("bf16 is the quality default there") — then v4.8.0 made Q8 the AUTO
    // default everywhere without un-gating the build, so on a big Mac the new
    // default silently fell back to bf16 for want of a pack Install refused to
    // make: "i updated but still getting this much memory", python3.11 at
    // 47.76 GB. The gate now lives in h3_build_q8.sh with that history; here it
    // is unconditional. Idempotent: the pack's own quant_config.json gates the
    // re-run, so a healthy install pays one stat call.
    {
      method: "shell.run",
      params: {
        path: "minimax-h3-mlx",
        // 3.8.3: body moved to scripts/pinokio/h3_build_q8.sh (595-char
        // dispatch). {{cwd}} is substituted in the MESSAGE, so the app root
        // is passed as $1 rather than written into the script.
        message: [
          "bash ../scripts/pinokio/h3_build_q8.sh '{{cwd}}'"
        ]
      }
    },

    {
      method: "notify",
      params: {
        // 3.8.3: this used to say "the Video tab now has an Engine row". The
        // engine control has been the switcher in the TOP RIGHT of the header
        // since the engine table landed — there is no Engine row in the Video
        // tab any more, and #58 ("[bug] Engine Picker not showing") is a user
        // who installed H3 successfully and then went looking in the place this
        // sentence sent him. Say where the control actually is, and name the
        // one thing that can still hide it: the switcher only appears when
        // there is more than one engine this Mac can render with, and on a
        // sub-60 GB Mac that means the reduced-RAM Q8 pack built by the step
        // above must be on disk.
        html: "<b>Hailuo H3 ready.</b><br>Open the panel — the engine switcher is in the <b>top right of the header</b>, next to the memory and models pills: <b>LTX-2.3 | Hailuo H3</b>. (Not in the Video tab.) Restart the panel if it was already running. H3 serves Text and Image, and generates dialogue + sound from the same prompt, so write them into it."
      }
    }
  ]
}
