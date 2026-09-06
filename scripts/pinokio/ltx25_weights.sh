#!/usr/bin/env bash
# Download LTX-2.5 — the DEFAULT generation — and stop the install if it fails.
#
# Lifted out of install.js (748-char dispatch). See scripts/pinokio/README.md.
#
#   cwd : the app root
#
# SEMANTICS: was a "\n"-joined string, i.e. one shell, no `set -e`, exit code
# from the last command — with one deliberate `exit 1`. Unchanged.
#
# WHY NOT `hf download`: the packs are mirrored as assets on a release of the
# public repo (the same lane the sample-character LoRA takes) because our HF
# token is read-only and these are our own quantisation of a gated upstream.
# Files over GitHub's 2 GiB asset cap are published as 1.9 GB shards;
# fetch_pack_release.py downloads them, checks each shard's sha256,
# reassembles, and only renames a file into place once the whole-file sha256
# matches the published manifest. Resumable and idempotent, so Resume Install
# re-runs it for the price of a read pass.
#
# BOTH PACKS ARE REQUIRED, not optional. 2.5 conditions on its own Gemma 4
# fine-tune and cannot use the Gemma 3 encoder — the mismatch does not raise,
# it silently encodes wrongly — so they are fetched together and the install
# STOPS if either fails rather than marching on to a panel that cannot render.

PY=./ltx-2-mlx/env/bin/python3.11
# `tae` rides along: 22 MB, and it is what makes the live preview work. It is
# a `kind: base` row, so an install without it reads INCOMPLETE — fetching it
# here is what keeps that true rather than aspirational.
if $PY scripts/fetch_pack_release.py --repo-key q4_25 --repo-key gemma4_25 --repo-key tae; then
  echo 'LTX-2.5 weights ready (verified against the published manifest).'
else
  echo '=================================================================='
  echo 'PHOSPHENE COULD NOT DOWNLOAD THE LTX-2.5 WEIGHTS'
  echo 'These are the default model - the panel cannot generate without them.'
  echo 'This is almost always a network problem (no connection, a VPN or'
  echo 'proxy, or GitHub blocked) or a full disk: the two packs need 28 GB.'
  echo 'Fix that, then click Install again - nothing already downloaded is'
  echo 'downloaded twice.'
  echo '=================================================================='
  exit 1
fi

# Upscale x2 adapter (0.3 GB, optional): the LTX-2.5 Pixel Spatial Upscaler
# IC-LoRA behind Remix -> Upscale x2. Gated on the Hub, mirrored on the same
# release. BEST-EFFORT: a miss must never fail the install - the panel offers
# it in Settings -> Models and the mode refuses by name until it lands.
$PY scripts/fetch_pack_release.py --repo-key ic_upscale_x2 \
  || echo 'WARN: Upscale x2 adapter fetch failed - video still works; download it later from Settings -> Models.'
