"""Lifecycle for the bundled `mlx-lm.server` subprocess.

This is the engine the panel spawns when the user picks "Phosphene Local".
It speaks OpenAI-compatible chat completions on 127.0.0.1:8200 by default,
loaded against whatever model the user picks (or the bundled Gemma 3 12B
that ships with Phosphene as the LTX text encoder — same weights, dual
purpose).

We follow the exact same shape as `mlx_warm_helper.py`'s subprocess:
spawn from the venv, capture stdout/stderr to STATE['log'] for visibility,
SIGTERM on shutdown.

The model **stays loaded** between agent turns. Re-spawning per-turn
would re-tokenize the system prompt every time and burn 20–60 s on cold
weight reads. Kept warm, subsequent turns are first-token latency only.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from pathlib import Path

# Module-level state. One mlx-lm.server per panel process — the agent
# isn't worth running two of side-by-side on a 64 GB machine since the
# model already eats >10 GB.
_LOCK = threading.Lock()
_PROC: subprocess.Popen | None = None
_PORT: int = int(os.environ.get("LTX_AGENT_LOCAL_PORT", "8200"))
_MODEL_PATH: str = ""
_LAST_ERROR: str = ""


def status() -> dict:
    """Snapshot for the UI: is the local engine running, and on what?"""
    global _PROC
    with _LOCK:
        running = _PROC is not None and _PROC.poll() is None
        return {
            "running": running,
            "port": _PORT,
            "model_path": _MODEL_PATH,
            "pid": _PROC.pid if running else None,
            "last_error": _LAST_ERROR,
            "base_url": f"http://127.0.0.1:{_PORT}/v1",
        }


def is_running() -> bool:
    with _LOCK:
        return _PROC is not None and _PROC.poll() is None


def start(model_path: str, *, venv_python: str, log_sink=None) -> dict:
    """Spawn `python -m mlx_lm.server --model <path> --host 127.0.0.1 --port <port>`.

    `venv_python` is the same Python the helper subprocess uses. It must
    have `mlx_lm` importable (Phosphene's bundled venv does).

    `log_sink` is a callable `(line: str) -> None`. Each line of the
    server's stdout/stderr is forwarded to it (plumbing into STATE['log']).

    Returns the new status() dict.
    """
    global _PROC, _MODEL_PATH, _LAST_ERROR

    with _LOCK:
        if _PROC is not None and _PROC.poll() is None:
            # Already running. If the requested model differs, kill + respawn.
            if _MODEL_PATH and _MODEL_PATH == model_path:
                return _status_locked()
            _stop_locked(reason="model change")

        if not Path(venv_python).exists():
            _LAST_ERROR = f"venv python not found: {venv_python}"
            return _status_locked()

        # Defensive flags for mlx-lm 0.31.1.
        #
        # Bug reproduced 2026-05-06: the multi-message agent loop hits
        # `_merge_caches` → `BatchRotatingKVCache.merge` (mlx_lm/models/
        # cache.py:1364) and raises
        #   ValueError: [broadcast_shapes] Shapes (1,8,N,256) and
        #               (1,8,1024,256) cannot be broadcast.
        # mid-generation, hanging the request until the client times out
        # (300 s default). The shapes match Gemma's sliding-window
        # rotating cache (window=1024); merging caches that crossed the
        # window with caches that didn't gives mismatched temporal axes.
        #
        # Repro details: every second multi-turn request, with a long
        # system prompt (~3.5 k tokens) and a tool-result-bearing user
        # turn, triggers it reliably. `--prompt-concurrency 1` alone is
        # insufficient — the merge fires across cache *snapshots* even
        # for a single in-flight request.
        #
        # The combination below sidesteps both code paths:
        #   - --prompt-cache-size 0 keeps zero distinct caches between
        #     requests, so nothing to merge across the request boundary.
        #   - --prefill-step-size 8192 prefills the entire prompt as
        #     one chunk for any prompt under 8 k tokens (our system +
        #     tool-result history fits comfortably), so chunked-prefill
        #     caches never need merging within a request either.
        cmd = [
            venv_python, "-m", "mlx_lm.server",
            "--model", model_path,
            "--host", "127.0.0.1",
            "--port", str(_PORT),
            "--log-level", "WARNING",
            "--prompt-concurrency", "1",
            "--decode-concurrency", "1",
            "--prompt-cache-size", "0",
            "--prefill-step-size", "8192",
        ]
        env = os.environ.copy()
        # mlx-lm honors HF_HOME for model cache lookup. The bundled
        # Gemma is already absolute-pathed so this is mostly hygiene.
        try:
            _PROC = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
        except Exception as e:                  # noqa: BLE001
            _LAST_ERROR = f"spawn failed: {e}"
            _PROC = None
            return _status_locked()

        _MODEL_PATH = model_path
        _LAST_ERROR = ""

    # Outside the lock: pump output to the log sink in a daemon thread.
    if log_sink is not None and _PROC is not None and _PROC.stdout is not None:
        t = threading.Thread(
            target=_pump_lines,
            args=(_PROC, log_sink),
            daemon=True,
            name="mlx_lm_server_log_pump",
        )
        t.start()

    # Tiny grace so callers polling status() right after start see the
    # 'running' state instead of a race-window 'not running yet'.
    time.sleep(0.2)
    return status()


def stop(reason: str = "manual stop") -> dict:
    with _LOCK:
        _stop_locked(reason)
        return _status_locked()


def _stop_locked(reason: str) -> None:
    global _PROC, _LAST_ERROR
    if _PROC is None:
        return
    if _PROC.poll() is None:
        try:
            os.killpg(os.getpgid(_PROC.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        # Give it 3 s to exit cleanly, then SIGKILL the group.
        deadline = time.time() + 3.0
        while time.time() < deadline and _PROC.poll() is None:
            time.sleep(0.05)
        if _PROC.poll() is None:
            try:
                os.killpg(os.getpgid(_PROC.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
    _LAST_ERROR = f"stopped ({reason}, exit={_PROC.returncode})"
    _PROC = None


def _status_locked() -> dict:
    running = _PROC is not None and _PROC.poll() is None
    return {
        "running": running,
        "port": _PORT,
        "model_path": _MODEL_PATH,
        "pid": _PROC.pid if running else None,
        "last_error": _LAST_ERROR,
        "base_url": f"http://127.0.0.1:{_PORT}/v1",
    }


def _pump_lines(proc: subprocess.Popen, sink) -> None:
    """Background pump: forward each stdout line to the panel's log sink.

    mlx-lm's server emits structured log lines we want visible in the
    panel's Logs tab so users can see startup ("Loaded weights in 4.2s")
    and per-request progress.
    """
    if proc.stdout is None:
        return
    try:
        for raw in proc.stdout:
            try:
                line = raw.decode("utf-8", errors="replace").rstrip()
            except Exception:                   # noqa: BLE001
                continue
            if not line:
                continue
            try:
                sink(f"[mlx-lm] {line}")
            except Exception:                   # noqa: BLE001 — sink failure shouldn't break the pump
                pass
    except Exception:                           # noqa: BLE001 — process closed, exit pump
        return


# ---- Model discovery ----------------------------------------------------------
def discover_local_models(repo_root: Path) -> list[dict]:
    """List candidate chat models on disk under `mlx_models/`.

    The agent doesn't need a special download — the bundled Gemma 3 12B IT
    that ships as the LTX text encoder is also a perfectly capable chat
    model. Users who want a stronger agent (Qwen 3 Coder 30B, Devstral 24B)
    can drop them into `mlx_models/<name>/` and the picker shows them.

    A model dir must contain config.json + at least one .safetensors to
    be considered chat-usable.
    """
    models_root = repo_root / "mlx_models"
    if not models_root.is_dir():
        return []

    candidates = []
    for child in sorted(models_root.iterdir()):
        if not child.is_dir():
            continue
        # Skip the LTX repos — those are video model weights, not chat.
        name = child.name
        if name.startswith("ltx-") or name == "loras":
            continue
        if not (child / "config.json").exists():
            continue
        has_weights = any(child.glob("*.safetensors")) or any(child.glob("*.npz"))
        if not has_weights:
            continue
        # Heuristic for "chat-like": presence of tokenizer config and
        # an instruction-tuned name suffix. Better than scanning configs.
        is_chat_likely = (
            (child / "tokenizer.json").exists()
            or (child / "tokenizer_config.json").exists()
        )
        candidates.append({
            "name": name,
            "path": str(child),
            "chat_likely": is_chat_likely,
            "size_gb": _dir_size_gb(child),
        })
    return candidates


def _dir_size_gb(p: Path) -> float:
    try:
        total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        return round(total / (1024 ** 3), 2)
    except Exception:                           # noqa: BLE001
        return 0.0
