#!/usr/bin/env bash
# Local-dev launcher for the LTX23MLX panel. Pinokio uses pinokio.js +
# start.js; this is the equivalent for running outside Pinokio.
#
# Pins the panel to the venv's python3.11 so it shares the exact same
# stdlib + interpreter as the helper subprocess. Default `python3` on
# macOS is currently 3.9 (Xcode tools), which silently runs the panel
# but uses a `cgi.FieldStorage` API that's deprecated on 3.11+ and
# removed on 3.13 — so local dev was diverging from the Pinokio install.
#
# Usage:
#   ./run_panel.sh                # foreground
#   ./run_panel.sh --bg           # background (logs to panel.log)
#   PORT=8765 ./run_panel.sh      # override port

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MLX="$ROOT/ltx-2-mlx"
PY="$MLX/.venv/bin/python3.11"

if [[ ! -x "$PY" ]]; then
  echo "ERR: venv python3.11 not found at $PY" >&2
  echo "     Run the install (uv venv + uv pip install ...) first, or use Pinokio." >&2
  exit 1
fi

# Mirror the env Pinokio's start.js sets, but fall back to HF repo ids when
# the local dir doesn't exist. Without this fallback, a manual install where
# weights live in ~/.cache/huggingface gets mis-flagged as "base incomplete"
# in the panel UI even though the helper resolves them fine.
if [[ -z "${LTX_MODEL:-}" ]]; then
  if [[ -d "$ROOT/mlx_models/ltx-2.3-mlx-q4" ]]; then
    export LTX_MODEL="$ROOT/mlx_models/ltx-2.3-mlx-q4"
  else
    export LTX_MODEL="dgrauet/ltx-2.3-mlx-q4"
  fi
fi
if [[ -z "${LTX_GEMMA:-}" ]]; then
  if [[ -d "$ROOT/mlx_models/gemma-3-12b-it-4bit" ]]; then
    export LTX_GEMMA="$ROOT/mlx_models/gemma-3-12b-it-4bit"
  else
    export LTX_GEMMA="mlx-community/gemma-3-12b-it-4bit"
  fi
fi
export LTX_MODEL_HQ="${LTX_MODEL_HQ:-$ROOT/mlx_models/ltx-2.3-mlx-q8}"
export LTX_MODELS_DIR="${LTX_MODELS_DIR:-$ROOT/mlx_models}"
export LTX_Q8_LOCAL="${LTX_Q8_LOCAL:-$ROOT/mlx_models/ltx-2.3-mlx-q8}"
export LTX_HELPER_PYTHON="${LTX_HELPER_PYTHON:-$PY}"
export LTX_PORT="${PORT:-${LTX_PORT:-8198}}"

cd "$ROOT"

if [[ "${1:-}" == "--bg" ]]; then
  : > panel.log
  nohup "$PY" mlx_ltx_panel.py >> panel.log 2>&1 &
  echo "panel started in background, PID $!  log: $ROOT/panel.log"
else
  exec "$PY" mlx_ltx_panel.py
fi
