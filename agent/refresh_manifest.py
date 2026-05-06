"""Rewrite an Agentic Flows session manifest with current job statuses.

The agent writes the manifest right before it calls `finish` — at that
point, most of the shots are still queued or running. This script
re-reads the live queue/history from the panel, looks up every shot
in the session's `tool_state.submitted_shots`, and rewrites the
manifest so each row has the final status, output_path, and
elapsed_sec.

Usage:
    cd /path/to/phosphene-dev.git
    python -m agent.refresh_manifest <session_id> [panel_url]

    panel_url defaults to http://127.0.0.1:8198 (production panel) and
    falls back to 8199 (dev panel) if 8198 isn't responding.

Run it whenever you want — once at the end of an overnight batch is
typical.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from agent import runtime, tools


def _http_json(url: str, timeout: float = 5.0) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, json.JSONDecodeError, OSError):
        return None


def _resolve_panel(candidates: list[str]) -> str | None:
    for url in candidates:
        if _http_json(f"{url}/status") is not None:
            return url
    return None


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 1
    session_id = argv[1]
    panel_url_arg = argv[2] if len(argv) > 2 else None
    candidates = [panel_url_arg] if panel_url_arg else ["http://127.0.0.1:8198", "http://127.0.0.1:8199"]
    candidates = [c for c in candidates if c]
    panel = _resolve_panel(candidates)
    if not panel:
        print(f"could not reach a Phosphene panel on {candidates}")
        return 2
    print(f"using panel: {panel}")

    # Find the session on disk. STATE_DIR comes from env or the repo default.
    repo = Path(__file__).resolve().parent.parent
    state_dir = Path(os.environ.get("LTX_STATE_DIR", str(repo / "state"))) if False else (repo / "state")
    sess = runtime.load_session(session_id, state_dir)
    if sess is None:
        print(f"session not found: state/agent_sessions/{session_id}.json")
        return 3

    # Build the panel-backed lookups so write_session_manifest sees live state.
    def find_job(jid: str) -> dict | None:
        s = _http_json(f"{panel}/status") or {}
        cur = s.get("current") or {}
        if cur.get("id") == jid:
            return cur
        for j in (s.get("queue") or []) + (s.get("history") or []):
            if j.get("id") == jid:
                return j
        return None

    ops = tools.PanelOps(
        submit_job=lambda f: (_ for _ in ()).throw(RuntimeError("read-only refresh")),
        queue_snapshot=lambda: _http_json(f"{panel}/status") or {"current": None, "queue": [], "history": []},
        find_job=find_job,
        outputs_dir=repo / "mlx_outputs",
        uploads_dir=repo / "panel_uploads",
        capabilities={},
    )

    res = tools.dispatch(
        "write_session_manifest",
        {"title": sess.title, "output_name": "manifest.json"},
        ops, sess.tool_state,
    )
    if not res.get("ok"):
        print(f"manifest refresh failed: {res.get('error')}")
        return 4
    inner = res.get("result") or {}
    print(f"refreshed: {inner.get('manifest_path')}")
    print(f"  shots resolved: {inner.get('shot_count')}")
    if inner.get("missing_outputs"):
        print(f"  missing: {inner['missing_outputs']}")
    return 0


if __name__ == "__main__":
    import os
    sys.exit(main(sys.argv))
