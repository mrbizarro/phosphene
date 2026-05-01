#!/usr/bin/env bash
#
# Phosphene git-repo recovery — escape hatch, NOT the recommended path.
#
# The recommended fix when Pinokio Update silently does nothing is:
#   In Pinokio: click Reset on the Phosphene panel, then Install.
# That's clean, doesn't depend on running unaudited shell scripts, and
# from Y1.004 onward Pinokio's fs.link drive preserves your models +
# outputs + settings across Reset/Install anyway.
#
# This script exists for users who'd rather avoid re-downloading the
# 36 GB of LTX weights — it resets only the git repo to current main,
# leaving everything else untouched. Read the source before running.
#
# Why it's needed at all:
#   A one-time history scrub on 2026-05-01 (rewriting commit identities
#   to anonymize an author leak) force-pushed origin/main. Clones from
#   before that push have orphaned commits in their local main, so
#   `git pull` refuses to fast-forward and Pinokio's Update step silently
#   stalls. Y1.002+ has a self-recovering update.js — but to GET to
#   Y1.002+, an existing pre-Y1.002 user needs either this script or
#   the Reset+Install path.
#
# What it does:
#   1. Locate the phosphene install dir (default: ~/pinokio/api/phosphene.git;
#      override via PHOSPHENE_DIR=/path/to/install).
#   2. Verify it's actually phosphene's repo (defensive: refuse to reset
#      a different repo that happens to live at that path).
#   3. git fetch origin && git reset --hard origin/main. The working tree
#      and any local commits get overwritten — which a Pinokio install
#      isn't expected to have anyway. Models, queue, settings, LoRAs all
#      live OUTSIDE git tracking, so they're untouched.
#
# Usage (read the script first, please):
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
