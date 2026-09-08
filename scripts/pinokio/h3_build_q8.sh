#!/usr/bin/env bash
# Build the reduced-RAM Q8 DiT locally. Every Mac needs it since v4.8.0.
#
# Lifted out of install_h3.js (595-char dispatch). See
# scripts/pinokio/README.md.
#
#   cwd : minimax-h3-mlx (the H3 checkout — this script runs .venv/bin/python
#         and scripts/quantize_stream.py by RELATIVE path, so the cwd is not
#         cosmetic)
#   $1  : the app root. `{{cwd}}` is substituted by Pinokio in the MESSAGE, so
#         it cannot appear in this file — it arrives as an argument.
#   env : LTX_H3_MODELS, optional — the same override the panel reads
#         (mlx_ltx_panel.py H3_MODELS, documented in docs/H3_ENGINE.md). When a
#         dev box points its weights at a shared checkout, the panel looks for
#         the pack there and this script used to build it somewhere else
#         entirely, so the build "succeeded" and the engine stayed bf16.
#
# SEMANTICS: one shell, no `set -e`, exit code from the last command.
# Idempotent: `.built_ok` (written after a fresh-process validation) plus the
# shards the index names gate the re-run — see the block above the build.
#
# WHY IT MATTERS: the Q8 pack halves the render peak (25.63 vs 42.8 GiB
# measured), which is what puts H3 in reach of a 36 GB Mac at all — and the
# panel's `h3_capable()` returns False on a sub-60 GB Mac until this pack
# EXISTS ON DISK, so such a Mac with no pack gets no Engine switcher.
# quantize_stream.py never holds more than one tensor (CPU stream,
# deterministic), so the build runs fine on the same machine that could never
# load the model whole. ~22 GB on disk, ~5 min.
#
# Since v4.8.0 the Q8 engine is the panel's AUTO DEFAULT on every machine,
# so every machine builds the pack — see the gate comment below.

APP_ROOT="$1"
if [ -z "$APP_ROOT" ]; then
  # NAME THE INVOCATION. "no app root passed - skipping" told whoever hit this
  # (a hand-run repair, a new update step) that something was skipped and
  # nothing about how to not skip it. Both facts below are load-bearing: the
  # cwd, because the interpreter and the quantizer are resolved relatively,
  # and the argument, because `{{cwd}}` only exists inside a Pinokio message.
  echo 'h3_build_q8.sh: no app root passed - skipping the Q8 build.'
  echo ''
  echo 'Run it with cwd = the minimax-h3-mlx checkout, app root as $1:'
  echo '  cd <app-root>/minimax-h3-mlx \'
  echo '    && bash <app-root>/scripts/pinokio/h3_build_q8.sh <app-root>'
  echo ''
  echo 'e.g. for a default Pinokio install:'
  echo '  cd ~/pinokio/api/phosphene.git/minimax-h3-mlx \'
  echo '    && bash ../scripts/pinokio/h3_build_q8.sh ~/pinokio/api/phosphene.git'
  echo ''
  echo 'Set LTX_H3_MODELS first if your weights live outside the app root.'
  exit 2
fi

# THE ROOT THE PANEL WILL LOOK UNDER, not the one we assume. Two things vary:
#   * LTX_H3_MODELS relocates the whole weights tree (dev boxes sharing one
#     75 GB checkout between installs) — mlx_ltx_panel.py honours it, so a
#     build that ignores it lands where nothing reads.
#   * the LAYOUT under that root. upstream `download_selected.py --root X`
#     appends `models/`, while the canonical campaign tree is flat; the panel
#     accepts BOTH (`_h3_model_roots()`), and resolves the Q8 pack against
#     whichever root it finds — so the pack has to be built beside the DiT it
#     was quantized from, whichever of the two carried it.
MODELS="${LTX_H3_MODELS:-$APP_ROOT/mlx_models/hailuo-h3}"
DIT_REL="deepbeep-pruned-bf16/MiniMax-H3-FL2VA-pruned_bf16.safetensors"
# Probe in the panel's own order. Falls back to the nested layout when neither
# has the DiT, because that is what a fresh install_h3.js download writes — and
# the build then fails loudly on a missing source rather than silently on a
# guessed one.
LAYOUT="$MODELS/models"
for cand in "$MODELS/models" "$MODELS"; do
  if [ -f "$cand/$DIT_REL" ]; then
    LAYOUT="$cand"
    break
  fi
done

PACK="$LAYOUT/h3-dit-q8"
SRC="$LAYOUT/$DIT_REL"
# EVERY machine builds the pack now. Until v4.8.0 this step was skipped on
# 64 GB+ Macs ("bf16 is the quality default there") — and then v4.8.0 made Q8
# the AUTO DEFAULT everywhere without un-gating this build, so on a big Mac
# the new default silently fell back to bf16 for want of a pack that Install
# refused to build. Field report, day one: "i updated but still getting this
# much memory" — python3.11 at 47.76 GB, which is the bf16 engine plus cache.
# The default's weights must be installable wherever the default applies;
# 22 GB of disk and ~5 min is the price of the halving the release notes
# promised. Idempotent via the pack's own quant_config.json, as before.
# VALIDATE IN A FRESH PROCESS, AND MARK ONLY AFTER. quantize_stream.py writes
# quant_config.json and THEN validates by loading the whole Q8 tree in the same
# process that still holds the mmap-backed bf16 source and the quantization
# intermediates — on a 36 GB Mac macOS SIGKILLs it right there, after every
# shard is on disk and valid (#78: "Killed: 9 … quantize_stream.py", while a
# clean `load_dit()` of the same pack takes 3.6 s). And because the marker
# was already written, the retry said "already built" over a pack that never
# reported success. So: the marker is `.built_ok`, written by THIS script only
# after `load_dit()` succeeds in its own process; a pack with shards but no
# marker is validated (never rebuilt) before it is trusted; a pack whose index
# names a shard that is missing is rebuilt.
pack_shards_present() {
  # every shard the index names exists and is non-empty
  [ -f "$PACK/model.safetensors.index.json" ] || return 1
  .venv/bin/python - "$PACK" <<'PY'
import json, os, sys
pack = sys.argv[1]
idx = json.load(open(os.path.join(pack, "model.safetensors.index.json")))
names = sorted(set(idx.get("weight_map", {}).values()))
ok = bool(names) and all(os.path.getsize(os.path.join(pack, n)) > 0 for n in names if os.path.exists(os.path.join(pack, n))) \
     and all(os.path.exists(os.path.join(pack, n)) for n in names)
sys.exit(0 if ok else 1)
PY
}
validate_pack_fresh() {
  # load_dit() in a process that holds nothing else — the validation the
  # quantizer's own process could not survive on the 36 GB floor
  echo '=== Validating the Q8 engine in a fresh process ==='
  .venv/bin/python - "$PACK" <<'PY'
import sys, time
from minimax_h3_mlx.load import load_dit
t0 = time.time()
load_dit(sys.argv[1], verbose=False)
print(f"LOAD OK - pack valid ({time.time() - t0:.1f}s)", flush=True)
PY
}
if [ -f "$PACK/.built_ok" ] && pack_shards_present; then
  echo 'Q8 engine already built - skipping'
else
  if [ -f "$PACK/quant_config.json" ] && pack_shards_present; then
    echo 'Q8 engine shards are on disk but were never validated - validating instead of rebuilding'
  else
    echo '=== Building the reduced-RAM Q8 engine (~5 min, one time) ==='
    rm -f "$PACK/.built_ok"
    .venv/bin/python scripts/quantize_stream.py --src "$SRC" --out "$PACK"
    echo "quantizer exit $? (a kill during its own final validation is recovered below)"
  fi
  if pack_shards_present && validate_pack_fresh; then
    : > "$PACK/.built_ok"
    echo 'Q8 engine built and validated'
  else
    rm -f "$PACK/quant_config.json" "$PACK/.built_ok"
    echo 'WARN: the Q8 pack did not validate - it will be rebuilt on the next Install / Update'
    exit 1
  fi
fi
