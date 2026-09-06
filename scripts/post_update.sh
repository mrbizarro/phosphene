#!/usr/bin/env bash
# =============================================================================
# post_update.sh — everything an Update does AFTER the panel repo has moved.
# =============================================================================
#
# WHY THIS FILE EXISTS
# --------------------
# Pinokio loads `update.js` from disk into memory and THEN runs it. Our step 1
# is what pulls the repo, so the repo moves to the new version MID-RUN and
# execution continues from the STALE in-memory script. cocktailpeanut hit this
# on his own machine and confirmed the sequencing by timestamp: the fix an
# app ships to its own
# installer always lands ONE CLICK LATE, and the user sees a failure they have
# already downloaded the fix for.
#
# So `update.js` is now THIN and STABLE — pull, then delegate here — and
# everything that can break lives in this file, which is read AFTER the pull
# and therefore always ships with the version it belongs to. A broken update
# step becomes fixable by shipping, the normal way. That kills the class for
# every future version.
#
# A SECOND PROPERTY, AND IT IS NOT INCIDENTAL
# -------------------------------------------
# v3.8.2's disaster was a Pinokio RUN that ended after a step which exited 0,
# because that step's stdout carried pip's "ERROR: pip's dependency resolver…"
# block. Steps 8-17 of 18 never ran — including the codec patch — and the
# update still presented as finished. Inside ONE shell script Pinokio sees ONE
# step, so no amount of scary text in the middle can silently drop the tail.
# The pip→uv conversion below stays anyway: two independent defences, because
# the failure mode is invisible.
#
# CONTRACT
# --------
#   $1  the app root, optional. Defaults to this file's parent's parent, so the
#       script is correct however Pinokio's cwd resolves — the v3.8.2 lesson
#       from patch_ltx_codec.py, which guessed from CWD and reported "MISSING"
#       on an install whose file was sitting right there.
#
# SEMANTICS — and the one thing collapsing 18 steps into 1 would have BROKEN
# ------------------------------------------------------------------------
# Under Pinokio each of these was its own step, so a non-zero exit aborted the
# run. Inside one script that is no longer automatic, and the first version of
# this file lost it: a `patch_ltx_codec.py` that failed its own verification
# printed its banner and the update carried on to report success — reinstating
# the exact silent-4:2:0 outcome v3.8.2 exists to prevent. Caught by the
# journey sim, on a synthetic pin whose upstream moved the ffmpeg line the
# patch anchors on.
#
# So failure semantics are now EXPLICIT rather than inherited:
#
#   require <what> -- <cmd...>   load-bearing. A non-zero exit prints a FATAL
#                                banner naming the step and exits, so Pinokio
#                                marks the Update failed. Steps 1-4.
#   plain invocation             best-effort, guarded with `|| echo WARN`. An
#                                Update must never be brickable by a network
#                                hiccup, and every one of these has a one-click
#                                retry in the panel. Steps 5-9.
#
# That is stricter than what shipped (where the boundary was implicit in the
# step split) and it is deliberate: the difference between "this update did
# nothing" and "this update quietly broke your renders" is exactly this line.
# =============================================================================

ROOT="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT" || { echo "FATAL: cannot enter app root $ROOT"; exit 1; }

require() {
  local what="$1"; shift
  [ "$1" = "--" ] && shift
  if ! "$@"; then
    echo ""
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "!! PHOSPHENE UPDATE FAILED: $what"
    echo "!! This step is not optional — stopping here rather than reporting a"
    echo "!! successful update that did not happen. Nothing has been deleted."
    echo "!! Re-run Update; if it fails the same way, the message above it is"
    echo "!! the one to report."
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    exit 1
  fi
}

PY="$ROOT/ltx-2-mlx/env/bin/python3.11"
VENV_PY="$ROOT/ltx-2-mlx/env/bin/python"
HF="$ROOT/ltx-2-mlx/env/bin/hf"
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "=== Phosphene post-update — app root: $ROOT ==="

# ---- 0. The venv itself, before anything tries to install into it -----------
# EVERY STEP BELOW TARGETS $VENV_PY. If that interpreter does not resolve, an
# Update is a long sequence of `uv pip install --python <nothing>` failures, and
# the first one to be load-bearing prints the FATAL banner naming a pin — which
# is not the problem and sends the report in the wrong direction.
#
# And it does not resolve more often than anyone expects. install.js builds the
# venv with `uv venv`, which makes env/bin/python a symlink chain into Pinokio's
# SHARED managed Python; any other pack, or any other Pinokio app, that makes uv
# re-resolve, bump or prune that interpreter leaves the chain DANGLING while
# env/ still sits there. That is the v3.4.0 "installed other packs and Hailuo H3
# vanished" report, one tree over — the LTX venv breaks the same way and had no
# self-heal on the Update path at all. Reinstall was the only cure, for a
# machine whose weights were all fine.
#
# Same implementation install.js runs (its own step, same cwd, same no-args
# call), so there is exactly one venv-repair behaviour in the app. It ASKS
# whether the interpreter runs and rebuilds only when it doesn't: a healthy
# install pays ~50 ms and skips, a broken one pays ~5 min and re-downloads
# nothing. Also carries the macOS-14 preflight, which is the right place for it.
#
# FATAL: if the venv cannot be made to work there is nothing this script can
# usefully do afterwards except fail slower and less clearly.
require "the LTX venv self-heal (rebuilds a dangling interpreter)" -- bash -c 'cd "$1/ltx-2-mlx" && bash ../scripts/pinokio/ltx_venv.sh' _ "$ROOT"

# ---- 1. The vendored engine pin --------------------------------------------
# One implementation, shared with install.js. Carries the 3.8.1 hotfix
# (reset --hard before the checkout) and the v4.0 move from a branch SHA to an
# immutable tag. The pin literal lives in that file, deliberately: see its
# header for why a pin held in update.js can only ever move one click late.
require "the vendored engine pin move" -- bash "$ROOT/scripts/pinokio/ltx_checkout.sh"

# ---- 2. mlx pin -------------------------------------------------------------
# mlx 0.31.2 attenuates the vocoder by ~22 dB (measured: max_volume -42.8 dB vs
# -9.2 dB on 0.31.1, same weights, same seed). --reinstall --no-deps so ONLY
# mlx moves and nothing that depends on it is re-resolved.
#
# --no-deps protects this step, but NOT the pin: step 7 below installs mflux
# WITH deps and can walk mlx back down from under it. That coupling, and the
# 2026-08-28 measurements behind holding at 0.31.1 rather than moving to
# 0.32.x, are in scripts/check_post_update.js and install.js. Read them before
# touching this line — and change both pins in one commit or not at all.
require "the mlx 0.31.1 pin" -- uv pip install --python "$VENV_PY" --reinstall --no-deps 'mlx==0.31.1' 'mlx-lm==0.31.1' 'mlx-metal==0.31.1'

# ---- 2b. THE TRANSFORMERS CAP — the one install.js has been promising -------
# install.js has said, since the 2026-07-10 ship-blocker, that "uv downgrades an
# already-installed 5.13.0 on the next Update". Nothing in the update path ever
# constrained transformers, so that sentence was false for a month: mlx-lm
# 0.31.1 declares `transformers>=5.0.0` with NO upper bound, and step 2 above
# uses --no-deps precisely so nothing else moves — which also means an existing
# 5.13.0 survives every Update untouched.
#
# The failure it leaves behind is total and silent: mlx_lm.tokenizer_utils
# breaks, the Gemma text-encoder load no-ops ("done in 0.0s"), and EVERY
# generation dies on "'str' object has no attribute '__module__'" →
# "Model not loaded. Call load() first." A user in that state can click Update,
# watch it succeed, and still not be able to render a single frame.
#
# FATAL, not best-effort: an install that finishes this script with 5.13.0
# cannot generate anything, so "continue and hope" is not a kindness. Resolved
# on the SAME uv invocation as the mlx trio (exactly as install.js:268 does it)
# so the solver cannot re-widen the cap while satisfying mlx-lm.
require "the transformers <5.13.0 cap (5.13.0 breaks every generation)" -- \
  uv pip install --python "$VENV_PY" 'mlx==0.31.1' 'mlx-lm==0.31.1' 'mlx-metal==0.31.1' 'transformers>=5.0.0,<5.13.0'

# ---- 3. Re-install the three vendored packages ------------------------------
# This is what actually moves an existing user's RUNTIME: a bare git checkout
# leaves the previous copy in site-packages. `--reinstall` also replaces the
# EDITABLE .pth links a pre-v4.0 install ended up with, which is what leaves
# the vendored tree clean for the next pin move.
#
# `--build-constraints` pins the build backend to hatchling<1.32. All three
# pyprojects declare `readme = "../../README.md"` — outside the package dir —
# which hatchling 1.32.0 made a hard error, so from the day it shipped every
# install and every Update died here on every pinned tag. uv, not pip: modern
# pip drops inherited constraints inside the isolated build env on purpose
# (`_PIP_IN_BUILD_IGNORE_CONSTRAINTS=1`), and its `--build-constraint` flag
# does not exist on the older pips this fleet was seeded with.
require "re-installing the three vendored packages" -- bash -c 'cd "$1/ltx-2-mlx" && uv pip install --python env/bin/python --reinstall --no-deps --build-constraints ../pip-build-constraints.txt ./packages/ltx-core-mlx ./packages/ltx-pipelines-mlx ./packages/ltx-trainer' _ "$ROOT"

# ---- 4. The codec patch — IMMEDIATELY after the reinstall -------------------
# It MUST be here: step 3 overwrites site-packages, so a patch applied before
# it is thrown away, and everything below is optional and could once have ended
# the run. On v3.8.1 this sat eleven steps down and never executed on the
# owner's machine — every render on that install encoded 4:2:0 with no error
# anywhere. patch_ltx_codec.py now resolves the target through the INTERPRETER
# (importlib.util.find_spec), not the filesystem, and verifies its own work,
# exiting non-zero behind a banner if the file it was supposed to change did
# not change.
require "the codec patch (without it every render encodes 4:2:0 - blocky faces)" -- "$PY" "$ROOT/patch_ltx_codec.py"

# ---- 5. Trainer transitive deps ---------------------------------------------
# ltx-trainer went in with --no-deps above, so pyyaml/pydantic/tqdm/rich were
# never resolved; the trainer subprocess dies at `import yaml` without them.
( cd "$ROOT/ltx-2-mlx" && uv pip install --python env/bin/python 'pyyaml>=6.0' 'pydantic>=2.0' 'tqdm>=4.65' 'rich>=13.0' )
# pywebpush signs the closed-tab completion alerts (Settings → Completion
# alerts). Optional: the panel offers the button only when it imports.
( cd "$ROOT/ltx-2-mlx" && uv pip install --python env/bin/python 'pywebpush>=2.0' )

# ---- 6. Optional runtime packages -------------------------------------------
# mlx-vlm --no-deps on purpose (its default deps fight our mflux/transformers/
# mlx pins; the panel imports it lazily). hf_transfer is HuggingFace's Rust
# accelerator — 5-10x on the big packs. certifi is load-bearing: start.js
# points SSL_CERT_FILE at its cacert.pem, and if it ever goes missing ALL panel
# stdlib HTTPS breaks. litellm is the agent's multi-provider chat client,
# floored at 1.83.14 because earlier 1.x was hit by the March 2026 PyPI
# supply-chain incident. smolagents needs uv rather than pip: it hard-pins
# huggingface-hub<1.0.0, which pip refuses to resolve against our >=1.5.0 floor
# (ResolutionImpossible — the "blue screen error flashing for a second every
# update"), while uv installs both and leaves it warned-but-functional.
( cd "$ROOT/ltx-2-mlx" && uv pip install --python env/bin/python --no-deps 'mlx-vlm==0.4.4' )
uv pip install --python "$VENV_PY" --upgrade 'hf_transfer>=0.1.6'
uv pip install --python "$VENV_PY" --upgrade certifi
uv pip install --python "$VENV_PY" --upgrade 'litellm>=1.83.14'
uv pip install --python "$VENV_PY" --upgrade 'huggingface-hub>=1.5.0,<2.0' 'smolagents>=1.24.0'

# ---- 7. The mflux image-engine pack (Ideogram 4 + Qwen-Edit) ----------------
# STANDARD since 2026-06-13, for every user — this is what unblocked Ideogram 4
# for everyone who never ran the optional installer. Two-step (with deps, then
# --reinstall --no-deps) mirrors install_qwen.js so the full transitive set
# resolves and then the version locks. BEST-EFFORT: video is unaffected if it
# fails, and the panel offers a one-click reinstall.
echo 'Installing/refreshing the mflux image-engine pack (Ideogram 4 + Qwen-Edit)…'
( uv pip install --python "$VENV_PY" 'mflux==0.18.0' \
  && uv pip install --python "$VENV_PY" --reinstall --no-deps 'mflux==0.18.0' \
  && uv pip install --python "$VENV_PY" 'mlx-teacache==0.4.1' ) \
  || echo 'WARN: mflux image-engine install hit an error — video is unaffected; re-run Update, or use the Reinstall image engines action, to retry.'

# The FBCache patch has to follow the step that installs what it patches.
# Idempotent — skips when its marker is present.
"$PY" "$ROOT/patch_mflux_fbcache.py"

# ---- 8. Weight self-heal ----------------------------------------------------
# Every fetch here is BEST-EFFORT. An Update must not be brickable, and each of
# these has a one-click retry in the panel (Models page / Repair).

# The Q4 spatial upscaler — the mosaic fix (#23). The Y1.024 download allowlist
# dropped it, so affected Q4 installs ran a RANDOMLY-INITIALISED upsampler and
# produced the rainbow-grid garbage. ~1 GB, resumable, skipped when present.
# ONLY for an install that already HAS the 2.3 pack. From v4.0 a fresh install
# fetches 2.5 only and LTX-2.3 is an in-panel offer, so running this
# unconditionally would create a 2.3 directory holding exactly one 1 GB file —
# which the Models modal would then report "partial" and Storage would offer to
# reclaim. A self-heal must not conjure the thing it heals.
if [ -d "$ROOT/mlx_models/ltx-2.3-mlx-q4" ]; then
  echo 'Ensuring the Q4 spatial upscaler is present (mosaic fix)…'
  "$HF" download dgrauet/ltx-2.3-mlx-q4 --local-dir "$ROOT/mlx_models/ltx-2.3-mlx-q4" --include 'spatial_upscaler_x2_v1_1.safetensors' \
    || echo 'WARN: spatial upscaler fetch failed — open the panel and click Repair to retry (fixes the mosaic).'
else
  echo 'LTX-2.3 is not installed — skipping its spatial-upscaler self-heal (install it from Settings → Models if you want to train a character).'
fi

# LTX-2.5, the DEFAULT generation (~28 GB). Anyone who installed before
# 2026-08-12 has 2.3 weights only, and 2.5 is what the panel boots into — so
# without this an Update leaves them on a default lane with nothing behind it.
# Mirrored as GitHub release assets (our own quantisation of a gated upstream,
# read-only HF token), hence the fetcher rather than `hf download`. It verifies
# what it already has, so the steady state is a read pass.
echo 'Ensuring the LTX-2.5 weights are present (default generation)…'
"$PY" "$ROOT/scripts/fetch_pack_release.py" --repo-key q4_25 --repo-key gemma4_25 --repo-key tae \
  || echo 'WARN: LTX-2.5 weight fetch failed — open the panel, go to Models, and click Download for the LTX 2.5 rows to retry. It resumes.'

# The three IC-LoRAs the optional modes need. Colorize + Control are un-gated
# originals; Ingredients comes from DeepBeepMeep/LTX-2, which mirrors the
# BYTE-IDENTICAL gated file un-gated — and --include is CRITICAL there because
# that repo is ~708 GB.
echo 'Ensuring the Colorize IC-LoRA is present (restore mode, optional)…'
"$HF" download DoctorDiffusion/LTX-2.3-IC-LoRA-Colorizer --local-dir "$ROOT/mlx_models/loras/ic" --include 'LTX-2.3-22b-IC-LoRA-Colorizer-0.9.safetensors' \
  || echo 'WARN: Colorize IC-LoRA fetch failed — the Colorize mode will fetch it on first use, or click Repair.'

echo 'Ensuring the Ingredients IC-LoRA is present (multi-reference mode, optional)…'
"$HF" download DeepBeepMeep/LTX-2 --local-dir "$ROOT/mlx_models/loras/ic" --include 'ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors' \
  || echo 'WARN: Ingredients IC-LoRA fetch failed — the Ingredients mode will fetch it on first use, or click Repair.'

echo 'Ensuring the Control IC-LoRA is present (Union, control mode, optional)…'
"$HF" download Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control --local-dir "$ROOT/mlx_models/loras/ic" --include 'ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors' \
  || echo 'WARN: Control IC-LoRA fetch failed — the Control mode will fetch it on first use, or click Repair.'

# ---- 8b. H3 compact (Q8) engine — the memory-halving default (optional) ----
# v4.8.0 made the Q8 DiT H3's AUTO default on every machine, but installs made
# before that never built the pack on 64 GB+ Macs — the install skipped it
# when bf16 was their default — so Update-only users kept rendering bf16
# silently at ~40-48 GB ("i updated but still getting this much memory").
# Update ships code; this step ships the default's weights. Gated on H3
# actually being installed here (pure-LTX installs skip in one stat call),
# idempotent via the pack's own quant_config.json, ~5 min and ~22 GB disk on
# the one run that builds. Cannot fail the update.
#
# THE GATE HAS TO ASK THE SAME QUESTION THE PANEL DOES. It hardcoded
# $ROOT/mlx_models/hailuo-h3/models/... and $ROOT/minimax-h3-mlx, so it was
# blind to both things that legitimately vary:
#   * LTX_H3_ROOT / LTX_H3_MODELS relocate the checkout and the weights — the
#     documented way (docs/H3_ENGINE.md) to stop a second install re-downloading
#     75 GB. mlx_ltx_panel.py honours them; this gate did not, so on exactly the
#     setup the docs recommend the Update reported nothing to do while the panel
#     ran bf16 at ~48 GB.
#   * the layout: upstream `download_selected.py --root X` appends `models/`,
#     the canonical campaign tree is flat, and the panel accepts BOTH
#     (`_h3_model_roots()`). A flat tree read as "H3 not installed" here.
# h3_build_q8.sh resolves the same two roots and the same two layouts itself —
# LTX_H3_MODELS reaches it through the environment — so it lands the pack beside
# whichever DiT this gate found.
H3_MODELS_ROOT="${LTX_H3_MODELS:-$ROOT/mlx_models/hailuo-h3}"
H3_CHECKOUT="${LTX_H3_ROOT:-$ROOT/minimax-h3-mlx}"
H3_DIT_REL='deepbeep-pruned-bf16/MiniMax-H3-FL2VA-pruned_bf16.safetensors'
if { [ -f "$H3_MODELS_ROOT/models/$H3_DIT_REL" ] || [ -f "$H3_MODELS_ROOT/$H3_DIT_REL" ]; } \
   && [ -d "$H3_CHECKOUT" ]; then
  # #74: the build script needs the v2 engine tree. Installs cloned before it
  # sat on the old branch and this step built against that forever. Move the
  # checkout to the pin first (the same move the H3 installer makes).
  echo 'Ensuring the H3 engine checkout is on the pinned branch…'
  bash "$ROOT/scripts/pinokio/h3_checkout.sh" "$H3_CHECKOUT" \
    || echo 'WARN: could not move the H3 checkout to its pinned branch — the Q8 build below may not run; re-run the H3 engine Install to retry.'
  echo 'Ensuring the H3 compact (Q8) engine is built (halves H3 render memory)…'
  ( cd "$H3_CHECKOUT" && bash "$ROOT/scripts/pinokio/h3_build_q8.sh" "$ROOT" ) \
    || echo 'WARN: H3 Q8 build failed — H3 keeps the full bf16 engine for now; re-run the H3 engine Install to retry.'
fi

# ---- 9. Trim variants we never load ----------------------------------------
# Pre-Y1.024 installs downloaded whole repos (Q4 56 GB instead of 20, Q8 82 GB
# instead of 37). `rm -f` is a no-op when the file is already gone.
# NOT trimmed, deliberately: ltx-2.3-mlx-q4/transformer-dev.safetensors — Train
# Character downloads that 11 GB file on demand, and removing it here would
# silently undo the training install on every update.
echo 'Trimming unused model variants from mlx_models/…'
rm -f "$ROOT/mlx_models/ltx-2.3-mlx-q4/transformer-distilled-1.1.safetensors"
rm -f "$ROOT/mlx_models/ltx-2.3-mlx-q4/ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
rm -f "$ROOT/mlx_models/ltx-2.3-mlx-q4/spatial_upscaler_x1_5_v1_0.safetensors"
rm -f "$ROOT/mlx_models/ltx-2.3-mlx-q4/temporal_upscaler_x2_v1_0.safetensors"
rm -f "$ROOT/mlx_models/ltx-2.3-mlx-q8/transformer-distilled.safetensors"
rm -f "$ROOT/mlx_models/ltx-2.3-mlx-q8/transformer-distilled-1.1.safetensors"
rm -f "$ROOT/mlx_models/ltx-2.3-mlx-q8/ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
rm -f "$ROOT/mlx_models/ltx-2.3-mlx-q8/spatial_upscaler_x1_5_v1_0.safetensors"
rm -f "$ROOT/mlx_models/ltx-2.3-mlx-q8/temporal_upscaler_x2_v1_0.safetensors"
echo 'Trim done.'

# ---- 10. THE IMPORT GATE — the last thing, because it is the only proof -----
#
# THE FLEET SAYS THIS IS NEEDED. `No module named 'ltx_pipelines_mlx'` is the
# SECOND most common error in the 14-day fleet read (464 events): an app that is
# installed, boots, and cannot render a single frame.
#
# install.js has guarded this since v2.0.2, for a reason written in its own
# comment: "the upstream pip step had silently failed mid-install but the patch
# script's i2v target check tolerates a missing ltx_pipelines_mlx file, so the
# install reported success and the user only learned about the breakage when
# they clicked Generate." Update never got the same guard. Step 3 above
# reinstalls the three packages and `require` catches a NON-ZERO pip — but pip
# exiting 0 is not the same claim as "the module imports", which is the only
# claim that matters, and the one the user finds out about at Generate time.
#
# So: import them. If that fails, do not fail the update yet — reinstall once
# and import again, because a torn site-packages is exactly what this repairs
# and the user should not have to run anything by hand. Only a second failure
# is fatal, and then `require` says so in the app's own words.
# The probe imports the same three packages as the gate but in semicolon form,
# so `check_post_update.js` can still tell them apart: the gate is the
# comma-form line it must find wrapped in `require`. (A one-package probe let
# a torn ltx_core_mlx skip the repair and fail the gate — review 2026-09-02.)
echo 'Verifying the render engine actually imports…'
if ! "$PY" -c "import ltx_core_mlx; import ltx_pipelines_mlx; import mlx" 2>/dev/null; then
  echo 'Render engine did not import — repairing the vendored packages once…'
  ( cd "$ROOT/ltx-2-mlx" && uv pip install --python env/bin/python --reinstall \
      --no-deps --build-constraints ../pip-build-constraints.txt \
      ./packages/ltx-core-mlx ./packages/ltx-pipelines-mlx ./packages/ltx-trainer ) \
    || true
fi
require "the render engine import gate" -- "$PY" -c "import ltx_core_mlx, ltx_pipelines_mlx, mlx"

echo "=== post-update complete ==="
