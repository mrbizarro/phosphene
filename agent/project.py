"""Project-level persistent memory for the agent.

Sessions are short-lived chats. A "project" is the long-running thing —
the satirical SaaS spot the user is iterating on for a week, the wedding
recap they're rebuilding from a script, the character bible they keep
returning to. Across many sessions, the agent should remember:

  - the user's evolving direction ("we settled on documentary not
    cinematic", "the brother character has a moustache, NOT a beard")
  - prior decisions ("standard tier was too slow on 12-second shots,
    moved everything to balanced + Sharp 720p")
  - the running cast / anchor library ("the doctor's portrait lives at
    panel_uploads/doctor.png")

Mechanism: one markdown file at `<state_dir>/agent_project_notes.md`.
Entries are appended with a timestamp + author tag. The agent's system
prompt every turn includes the tail of this file (most recent ~6 KB),
so the agent can reference its past notes without re-reading them.

Two tools — `read_project_notes` and `append_project_notes` — let the
agent read the full file (when it needs to) and add new memory entries
when something durable was decided.

The notes file is plain markdown so the user can also read / edit it
manually. There is intentionally no schema.
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path


NOTES_FILENAME = "agent_project_notes.md"
DEFAULT_PROMPT_EXCERPT_BYTES = 6 * 1024     # how much we surface in the system prompt


def notes_path(state_dir: Path) -> Path:
    return state_dir / NOTES_FILENAME


def read_notes(state_dir: Path) -> str:
    """Return the full notes file, or "" if it doesn't exist."""
    p = notes_path(state_dir)
    if not p.is_file():
        return ""
    try:
        return p.read_text(encoding="utf-8")
    except OSError:
        return ""


def read_notes_excerpt(state_dir: Path,
                       max_bytes: int = DEFAULT_PROMPT_EXCERPT_BYTES) -> str:
    """Return the tail of the notes file, capped at `max_bytes`.

    The tail is what the agent has most recently learned, so it's the
    highest-signal slice for the system prompt. We cut on a newline
    boundary in the FIRST half of the tail — that way a tail that ends
    in a long unbroken paragraph still surfaces (otherwise we might find
    only a trailing "\\n" and accidentally return an empty excerpt).
    """
    full = read_notes(state_dir)
    if not full:
        return ""
    if len(full.encode("utf-8")) <= max_bytes:
        return full
    encoded = full.encode("utf-8")
    tail = encoded[-max_bytes:]
    # Search for the first newline in the FIRST HALF only. If we find one
    # there, trim a partial leading line. If the only newline lives in
    # the last half (or at the very end), keep the whole tail — losing a
    # mid-sentence start is better than throwing away content.
    half = max(1, len(tail) // 2)
    cut = tail[:half].find(b"\n")
    if cut == -1:
        return tail.decode("utf-8", errors="replace")
    return tail[cut + 1:].decode("utf-8", errors="replace")


def append_note(state_dir: Path, text: str, *,
                kind: str = "note", author: str = "agent") -> dict:
    """Append a timestamped entry to the notes file.

    Returns metadata: {path, bytes_written, total_bytes, kind, author, ts}.

    The format mirrors a chat log:

        ## 2026-05-06 14:33  [note · agent]
        the agreed master style is "documentary, kodak portra 400, …"

    The header on a fresh file describes what the file is for — written
    once on first append.
    """
    p = notes_path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    fresh = not p.is_file()
    text = (text or "").strip()
    if not text:
        raise ValueError("note text is empty")
    ts = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M")
    block = f"\n## {ts}  [{kind} · {author}]\n{text}\n"
    if fresh:
        block = (
            "# Phosphene Agentic Flows — project notes\n"
            "Persistent memory across sessions. The agent reads the tail\n"
            "of this file every turn; you can also edit it manually.\n"
            + block
        )
    with p.open("a", encoding="utf-8") as f:
        f.write(block)
    try:
        os.chmod(p, 0o600)
    except OSError:
        pass
    return {
        "path": str(p),
        "bytes_written": len(block),
        "total_bytes": p.stat().st_size,
        "kind": kind,
        "author": author,
        "ts": ts,
    }
