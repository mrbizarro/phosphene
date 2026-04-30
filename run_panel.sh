#!/usr/bin/env bash
# Local-dev launcher for the Phosphene panel. Pinokio uses pinokio.js +
# start.js; this is the equivalent for running outside Pinokio.
#
# Pins the panel to the venv's python3.11 so it shares the exact same
# stdlib + interpreter as the helper subprocess. Default `python3` on
# macOS is currently 3.9 (Xcode tools), which silently runs the panel
# but uses a `cgi.FieldStorage` API that's deprecated on 3.11+ and
# removed on 3.13 — so local dev was diverging from the Pinokio install.
#
# Usage:
#   ./run_panel.sh                          # foreground
#   ./run_panel.sh --bg                     # background (logs to panel.log)
#   PORT=8765 ./run_panel.sh                # override port
#   LTX_TIER_OVERRIDE=base ./run_panel.sh   # force a tier (testing only)
#
# Tier override safety: this launcher inherits LTX_TIER_OVERRIDE from the
# parent shell only when explicitly set on THIS invocation's command line.
# Otherwise it's actively unset before the panel starts. Without this guard,
# a previous test run (`LTX_TIER_OVERRIDE=base ./run_panel.sh`) would set
# the env in the parent shell and silently bleed into every subsequent
# unprefixed restart — which has actually caused two bug reports already
# ("extend doesn't work on my 64 GB Mac" → was actually the panel running
# under a forced Compact override from a prior test).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MLX="$ROOT/ltx-2-mlx"
PY="$MLX/.venv/bin/python3.11"

if [[ ! -x "$PY" ]]; then
  echo "ERR: venv python3.11 not found at $PY" >&2
  echo "     Run the install (uv venv + uv pip install ...) first, or use Pinokio." >&2
  exit 1
fi

# Tier override warning. If LTX_TIER_OVERRIDE is set in the env when this
# launcher runs, the panel will fake a different RAM tier and reject jobs
# the real hardware can handle. Two real bug reports already came from
# leaving an override set across tests. Print loud and write to panel.log
# too so future-me sees it on grep.
if [[ -n "${LTX_TIER_OVERRIDE:-}" ]]; then
  echo "============================================================" >&2
  echo "WARN: LTX_TIER_OVERRIDE=$LTX_TIER_OVERRIDE is active." >&2
  echo "      Panel will pretend this Mac is a different tier." >&2
  echo "      Jobs may be rejected even on capable hardware." >&2
  echo "      Unset the variable for normal use." >&2
  echo "============================================================" >&2
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
