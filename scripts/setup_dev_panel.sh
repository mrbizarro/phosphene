#!/usr/bin/env bash
#
# Set up a local Phosphene Dev panel side-by-side with the production
# install. Y1.015+.
#
# What this gives you:
#   ~/pinokio/api/phosphene.git/      ← production panel (port 8198, main)
#   ~/pinokio/api/phosphene-dev.git/  ← dev panel        (port 8199, dev)
#
# The dev panel is auto-detected by the panel script: when its git
# branch is `dev`, the panel binds 8199 instead of 8198 and shows a
# "DEV" badge in the header. No code edits required — both clones run
# the same code, just on different branches.
#
# To save ~36 GB of duplicate model downloads, this script symlinks
# mlx_models/ from the production install into the dev panel. Outputs
# and state stay separate (each panel has its own state/, mlx_outputs/,
# panel_uploads/) so dev experiments don't pollute the production
# gallery.
#
# Run once:
#   bash scripts/setup_dev_panel.sh
#
# After that, open Pinokio. The "phosphene-dev.git" folder appears as a
# separate panel — Install (re-runs install.js for the venv + patches)
# then Start. Browse to http://127.0.0.1:8199 to use it.
#
# Workflow from then on:
#   1. Develop on the `dev` branch (push commits to origin/dev).
#   2. In Pinokio's Phosphene Dev panel, click Update → Stop → Start.
#   3. Verify the change works exactly like a user will see it.
#   4. Once happy: merge dev into main (`git checkout main && git merge dev && git push`).
#   5. Real users get the change on their next Pinokio Update.

set -euo pipefail

PROD_DIR="${PHOSPHENE_PROD_DIR:-$HOME/pinokio/api/phosphene.git}"
DEV_DIR="${PHOSPHENE_DEV_DIR:-$HOME/pinokio/api/phosphene-dev.git}"

if [[ ! -d "$PROD_DIR/.git" ]]; then
  echo "✗ no production install found at: $PROD_DIR" >&2
  echo "  install Phosphene first via Pinokio Discover, then re-run this script." >&2
  exit 1
fi

# ---- 1. clone (or skip if dev install already exists) ------------------------
if [[ -d "$DEV_DIR/.git" ]]; then
  echo "→ dev panel already exists at $DEV_DIR — skipping clone."
else
  echo "→ cloning phosphene into $DEV_DIR"
  git clone https://github.com/mrbizarro/phosphene.git "$DEV_DIR"
fi

# ---- 2. switch to dev branch -------------------------------------------------
cd "$DEV_DIR"
echo "→ fetching origin..."
git fetch origin
if git show-ref --verify --quiet refs/remotes/origin/dev; then
  # origin/dev exists upstream — track it
  if git show-ref --verify --quiet refs/heads/dev; then
    echo "→ dev branch already exists locally; switching to it"
    git checkout dev
    git pull --ff-only origin dev || true
  else
    echo "→ creating local dev tracking origin/dev"
    git checkout -b dev origin/dev
  fi
else
  echo "→ origin has no dev branch yet; creating one from main"
  git checkout -b dev
  echo "  (push it with: git push -u origin dev)"
fi

# ---- 3. share heavy assets via symlink ---------------------------------------
# mlx_models is the big one (~36 GB) — symlink so dev + prod share weights.
# In the production install, mlx_models is itself an fs.link symlink to a
# Pinokio drive folder; the symlink-of-a-symlink resolves fine through both.
# Outputs / state / uploads stay separate per panel by design.
for shared in mlx_models; do
  src="$PROD_DIR/$shared"
  dst="$DEV_DIR/$shared"
  if [[ -e "$src" && ! -L "$dst" ]]; then
    if [[ -d "$dst" && ! -L "$dst" ]]; then
      # If install.js already created it as a real dir on the dev side,
      # remove it before symlinking. Empty / freshly-created dirs only —
      # we never delete a real model collection.
      if [[ -z "$(ls -A "$dst" 2>/dev/null)" ]]; then
        rmdir "$dst"
      else
        echo "  (skip $shared: dev side already has content; not overwriting)"
        continue
      fi
    fi
    ln -s "$src" "$dst"
    echo "→ linked $shared → $src"
  fi
done

echo
echo "✓ Phosphene Dev panel ready"
echo "  Branch: $(git rev-parse --abbrev-ref HEAD)"
echo "  HEAD:   $(git rev-parse --short HEAD)"
echo "  Path:   $DEV_DIR"
echo
echo "Next:"
echo "  1. Open Pinokio. The 'phosphene-dev.git' folder appears as a separate panel."
echo "  2. Click Install — re-runs install.js for the venv + patches (idempotent)."
echo "  3. Click Start. Panel binds http://127.0.0.1:8199 with a DEV badge."
echo "  4. Push to origin/dev → click Update on that panel → test → merge to main."
