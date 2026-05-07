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


_RING_SLOTS = 7                                    # 7-version history → ~1 day on a chatty session


def _rotate_ring(p: Path) -> None:
    """Rotate one slot in the .bak ring buffer before write.

    Slot N becomes N+1 (oldest, slot 6, drops). Then current file copies
    to slot 0. Lets the user (or the agent's `restore_project_notes`
    affordance) undo a runaway append. Phase 0 #0.12 from the roadmap —
    project memory survives across sessions but a misclick or runaway
    agent can corrupt context. Cheap insurance.
    """
    try:
        if not p.is_file():
            return
        # Move existing slots up: 5 → 6, 4 → 5, …, 0 → 1.
        for i in range(_RING_SLOTS - 1, 0, -1):
            src = p.with_suffix(p.suffix + f".{i - 1}.bak")
            dst = p.with_suffix(p.suffix + f".{i}.bak")
            if src.exists():
                try:
                    os.replace(str(src), str(dst))
                except OSError:
                    pass
        # Copy current file to slot 0.
        slot0 = p.with_suffix(p.suffix + ".0.bak")
        try:
            data = p.read_bytes()
            slot0.write_bytes(data)
            try:
                os.chmod(slot0, 0o600)
            except OSError:
                pass
        except OSError:
            pass
    except Exception:                              # noqa: BLE001 — never block the append
        pass


def list_note_versions(state_dir: Path) -> list[dict]:
    """Return metadata for the available .bak slots — newest first."""
    p = notes_path(state_dir)
    out: list[dict] = []
    for i in range(_RING_SLOTS):
        slot = p.with_suffix(p.suffix + f".{i}.bak")
        if slot.is_file():
            try:
                stat = slot.stat()
                out.append({
                    "slot": i,
                    "path": str(slot),
                    "bytes": stat.st_size,
                    "mtime": stat.st_mtime,
                })
            except OSError:
                pass
    return out


def restore_note_version(state_dir: Path, slot: int) -> bool:
    """Replace the live notes file with the contents of `slot`.

    The restore is itself undoable on the next append — append_note
    rotates the current (post-restore) file into slot 0 then.

    Important: read the source bytes BEFORE we rotate, otherwise the
    rotation overwrites the requested slot with the current file's
    bytes and we'd "restore" the same thing we were trying to undo.
    """
    if slot < 0 or slot >= _RING_SLOTS:
        return False
    p = notes_path(state_dir)
    src = p.with_suffix(p.suffix + f".{slot}.bak")
    if not src.is_file():
        return False
    try:
        # Snapshot source contents first.
        data = src.read_bytes()
        tmp = p.with_suffix(p.suffix + ".restoretmp")
        tmp.write_bytes(data)
        os.replace(str(tmp), str(p))
        try:
            os.chmod(p, 0o600)
        except OSError:
            pass
        return True
    except OSError:
        return False


def append_note(state_dir: Path, text: str, *,
                kind: str = "note", author: str = "agent") -> dict:
    """Append a timestamped entry to the notes file.

    Returns metadata: {path, bytes_written, total_bytes, kind, author, ts}.

    The format mirrors a chat log:

        ## 2026-05-06 14:33  [note · agent]
        the agreed master style is "documentary, kodak portra 400, …"

    The header on a fresh file describes what the file is for — written
    once on first append. Before each append the previous state is
    rotated into a 7-slot ring buffer (`.0.bak` … `.6.bak`) so a runaway
    agent or misclick can be undone.
    """
    p = notes_path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    fresh = not p.is_file()
    text = (text or "").strip()
    if not text:
        raise ValueError("note text is empty")
    if not fresh:
        _rotate_ring(p)                            # snapshot before mutating
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
