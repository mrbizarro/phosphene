#!/usr/bin/env bash
# The vendored ltx-2-mlx pin move — the ONE implementation, shared by both
# lanes. install.js dispatches `bash scripts/pinokio/ltx_checkout.sh`;
# scripts/post_update.sh calls the same file. Before v4.0 the two lanes carried
# byte-identical copies of this block and the hotfix note had to say so; a
# duplicated fix is a fix that half-lands.
#
# Semantics preserved from the inline payload: NO `set -e`. The original was a
# "\n"-joined string whose exit code was the last command's. Every step here is
# individually load-bearing and reports for itself; the final `git rev-parse`
# is what the caller sees.
#
# =============================================================================
# THE PIN — and why it lives HERE and not in update.js
# =============================================================================
# scripts/pinokio/README.md's house rule is "the pin SHA stays in the .js".
# That rule was written for the INSTALL lane, where the .js and the .sh are
# cloned together and the rule is about documentation locality. In the UPDATE
# lane it inverts, and the sequencing wins:
#
#   Pinokio loads update.js from disk BEFORE our step 1 pulls the repo, so
#   whatever update.js says is the version the user had, not the version they
#   just downloaded. A pin held in update.js therefore moves ONE CLICK LATE,
#   forever — the exact failure class the thin updater exists to kill
#   (cocktailpeanut's own diagnosis; see `scripts/post_update.sh`).
#
# So the pin lives in the POST-PULL tree, in this file, and both lanes read it
# from here. `scripts/check_ltx_pin.js` is the gate that keeps install.js's
# documentation block and this value from drifting apart.
#
# =============================================================================
# IT IS A TAG, NOT A SHA, AND NOT A BRANCH
# =============================================================================
# v3.8.0 through v3.8.3 pinned a bare SHA obtained via `git fetch fork
# feat/ltx-2.5`. That worked only for as long as the SHA stayed reachable from
# that branch: the branch head had ALREADY moved past it by the time this was
# checked, and a single force-push or rebase would have stranded
# every existing install with an un-fetchable pin and a dead Update button.
# A tag is the only form of this reference that an upstream history rewrite
# cannot take away.
#
# `v0.14.19+ltx25.7` on mrbizarro/ltx-2-mlx == v0.14.19 plus:
#   - the LTX-2.5 port (versioned DiT config, duration head, Euler-ancestral
#     sampler, the vendored Gemma 4 text tower, encoder keyed off the checkpoint)
#   - unfused runtime LoRAs (c78cc71..3259338)
#   - the opt-in diffusion video-VAE decoder (a9c3f32..cc23423)
#   - isolated-modality guidance OFF by default on the 2.5 HQ path (25b9b8e)
#   - the distilled schedule: 2-step refine on 2.5, and a step count that THINS
#     the checkpoint's own table rather than the caller's preset (dfee6cf,
#     a252325 — without a252325 the panel's explicit stage2_steps=3 RAISES on
#     every 2.5 distilled job)
#   - live preview + early abort, opt-in and inert when unused
#   - the i2v anchor: the sample is re-composited against the clean latent
#     after the euler-ancestral step (ee256f5). Without it the pinned
#     conditioning tokens were rescaled and re-noised on every intermediate
#     step, so a 2.5 i2v render composed the clip without ever seeing the
#     supplied image and only the terminal step stamped it back in.
#   - ltx25.7: IC-LoRA `source_video=` — Upscale ×2 from the clip's own latent
#     (Stage 1 skipped, latent-upsampled, N-step control-aware refine) and the
#     2.5 distilled schedules on the IC lane instead of the 2.3 constants.
#
# The three packages declare `0.14.19+ltx25.7`, and `_LTX_EXPECTED_VERSION` in
# mlx_warm_helper.py must equal that string. Move the two together or the
# version-skew gate reports a permanent false alarm.
#
# To bump: retag on the fork, change LTX_PIN here, change the documented value
# in install.js, bump the three pyprojects AND _LTX_EXPECTED_VERSION, run
# `node scripts/check_ltx_pin.js`, smoke-test on dev, push.
LTX_PIN="v0.14.19+ltx25.7"
LTX_FORK_URL="https://github.com/mrbizarro/ltx-2-mlx.git"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT/ltx-2-mlx" || { echo "FATAL: no vendored ltx-2-mlx tree at $ROOT/ltx-2-mlx"; exit 1; }

echo "vendored engine -> $LTX_PIN"

# Upstream's own tags, for `git describe`. Best-effort: the pin does not live
# here and dgrauet being unreachable must not stop an Update.
git fetch --tags origin || echo "WARN: could not reach origin for upstream tags (not fatal - the pin lives on the fork)"

# An existing install has only dgrauet as `origin`, so the fork remote is added
# when absent — and its URL is RESET every run. `get-url || add` alone can never
# correct a remote that is already there and wrong, which is a silent way to
# fetch from nowhere and then check out whatever local ref happens to share the
# pin's name. Found by the v4.0 journey sim doing exactly that.
git remote get-url fork > /dev/null 2>&1 || git remote add fork "$LTX_FORK_URL"
git remote set-url fork "$LTX_FORK_URL"

# EXPLICIT FORCED REFSPEC, not `git fetch --force fork tag <name>`. The `tag`
# shorthand does not honour --force, so a pin that was RE-tagged upstream (a
# corrected release, the ordinary reason to retag) leaves the stale local tag in
# place and the checkout below silently lands on the WRONG COMMIT while printing
# the right name. Also measured in the journey sim.
#
# And the fetch is FATAL. Without the pin fetched there is nothing correct to
# check out; carrying on would either fail obscurely or, worse, succeed against
# a stale local tag. An Update that stops here has changed nothing.
if ! git fetch fork "+refs/tags/$LTX_PIN:refs/tags/$LTX_PIN"; then
  echo "FATAL: could not fetch the vendored engine pin '$LTX_PIN' from $LTX_FORK_URL."
  echo "       Nothing has been changed. Check your connection and re-run."
  exit 1
fi

# --- 3.8.1 HOTFIX: discard local edits BEFORE the pin move -------------------
#
# On 3.8.0 this checkout failed for real users with
#   error: Your local changes to the following files would be overwritten by
#   checkout: packages/ltx-core-mlx/src/ltx_core_mlx/model/video_vae/video_vae.py
# and no retry could clear it — nothing in the flow ever un-dirtied the file,
# so the Update button was dead for good.
#
# WHY AN INSTALL IS DIRTY AT ALL. ltx-2-mlx is a uv WORKSPACE. Installing its
# members WITH deps and WITHOUT --reinstall links them EDITABLE: site-packages
# gets `_editable_impl_ltx_core_mlx.pth`, not a copy. patch_ltx_codec.py then
# finds no ltx_core_mlx directory in site-packages and patches the GIT-TRACKED
# source. (v4.0's install.js adds a --reinstall --no-deps pass so a FRESH
# install no longer ends this way — but every install made before it did, and
# this guard is what carries them across.)
#
# WHY reset --hard AND NOT `git checkout -- <file>`. This tree is APP-MANAGED
# end to end: models live in ../mlx_models, build constraints in
# ../pip-build-constraints.txt, and the only non-tracked thing inside it on a
# real install is the ignored env/ venv — which reset --hard does not touch, so
# the multi-GB venv survives. A surgical per-file revert fixes today's filename
# and breaks on the next one. The class is "app-managed checkout carries local
# edits across a pin move".
#
# The `-f` guard exists because Pinokio does not hard-fail a shell.run whose
# `path` is missing; without it a stray shell could run reset --hard in the
# panel repo.
if [ -f packages/ltx-core-mlx/pyproject.toml ]; then
  echo 'vendored tree is app-managed; discarding local edits before the pin move:'
  git status --porcelain | head -20
  git reset --hard HEAD
else
  echo 'WARN: not the vendored ltx-2-mlx tree - skipping reset'
fi

if ! git checkout "$LTX_PIN"; then
  echo "FATAL: could not check out the vendored engine pin '$LTX_PIN'."
  exit 1
fi

# Say what landed, and PROVE it is what was asked for. The whole point of a tag
# over a branch SHA is that this assertion can exist; a checkout that reports
# success while HEAD sits somewhere else is the failure mode the forced refspec
# above was added to close, and an assertion costs nothing.
WANT="$(git rev-parse "$LTX_PIN^{commit}")"
GOT="$(git rev-parse HEAD)"
if [ "$WANT" != "$GOT" ]; then
  echo "FATAL: HEAD is $GOT but the pin '$LTX_PIN' is $WANT."
  exit 1
fi
echo "vendored engine at $LTX_PIN ($(git rev-parse --short HEAD))"
