#!/usr/bin/env bash
#
# Phosphene one-line recovery — gets a stuck install onto current main.
#
# Why this exists:
#   A one-time history scrub on 2026-05-01 (rewriting commit identities to
#   anonymize an author leak) force-pushed origin/main on
#   github.com/mrbizarro/phosphene. Any clone that existed before that push
#   has commits in its local main that no longer exist on origin, so plain
#   `git pull` refuses to fast-forward and Pinokio's Update step silently
#   stalls. From Y1.002 onward Pinokio's update.js + the in-panel
#   "magic version pill" both fall back to `git reset --hard origin/main`
#   on divergence — but to GET to Y1.002, an existing user needs this
#   one-time fix.
#
# What it does:
#   1. Locate the phosphene install dir (default: ~/pinokio/api/phosphene.git;
#      override via PHOSPHENE_DIR=/path/to/install).
#   2. Verify it's actually phosphene's repo (defensive: don't reset a
#      repo that just happens to live there).
#   3. Fetch origin and reset --hard to origin/main. Wipes the working
#      tree and any local commits — which a Pinokio install isn't expected
#      to have anyway. Models, queue, settings, and downloaded LoRAs all
#      live OUTSIDE git tracking, so they're untouched.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/mrbizarro/phosphene/main/recover.sh | bash
#   # — or after downloading —
#   bash recover.sh
#   # — or pointing at a non-default install dir —
#   PHOSPHENE_DIR=/Volumes/AI/pinokio/api/phosphene.git bash recover.sh

set -euo pipefail

DIR="${PHOSPHENE_DIR:-$HOME/pinokio/api/phosphene.git}"

if [[ ! -d "$DIR/.git" ]]; then
  echo "✗ no git repo at: $DIR" >&2
  echo "   set PHOSPHENE_DIR=/your/install/path and re-run." >&2
  exit 1
fi

cd "$DIR"

# Defensive sanity check — this script wipes the tree, so we don't run it
# on a non-phosphene repo by accident.
remote_url="$(git config --get remote.origin.url || echo '')"
if [[ "$remote_url" != *"mrbizarro/phosphene"* ]]; then
  echo "✗ this isn't phosphene (origin = $remote_url)" >&2
  echo "   refusing to reset --hard on an unrelated repo." >&2
  exit 1
fi

echo "→ phosphene install: $DIR"
echo "→ origin: $remote_url"
echo "→ before: $(git rev-parse --short HEAD 2>/dev/null || echo '?')"
echo

echo "→ git fetch origin..."
git fetch origin

echo "→ git reset --hard origin/main..."
git reset --hard origin/main

new_sha="$(git rev-parse --short HEAD)"
new_version="$(cat VERSION 2>/dev/null || echo 'pre-Y1.001')"

echo
echo "✓ recovered. now on:  $new_sha  (version $new_version)"
echo
echo "Next:"
echo "  1. Open Pinokio."
echo "  2. Click Update on the Phosphene panel — should run cleanly now."
echo "  3. Click Stop, then Start."
echo "  4. Hard-refresh the panel page in your browser (Cmd+Shift+R)."
echo
echo "From Y1.002 onward this fallback runs automatically inside Update,"
echo "so you should never need to run this script again."
