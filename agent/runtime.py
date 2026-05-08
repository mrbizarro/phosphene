"""Agent runtime: sessions, the chat→tool loop, persistence.

A `Session` is one chat thread between the user and the agent. It owns
the message history, the active engine config, and a small `tool_state`
dict that tools mutate (e.g. `submitted_shots` from `submit_shot`).

`run_turn(session, user_message, ...)` is the entry point. It does:

    1. Append user_message to session.
    2. Call engine.chat() — get assistant reply.
    3. If the reply contains a fenced action block:
         - Dispatch the tool via tools.dispatch().
         - Append both the assistant reply (with the tool intent) and a
           synthetic user message containing <tool_result>...</tool_result>.
         - Loop back to step 2.
    4. Otherwise, the reply is the final assistant message. Stop.

The loop is bounded by `max_steps` so a misbehaving model can't burn the
whole context window.

Session state is persisted to `state/agent_sessions/<id>.json` after
every turn so a panel restart doesn't lose the conversation.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterator

from agent import engine, tools, prompts


MAX_STEPS_PER_TURN = 30           # safety cap on tool-loop iterations per user turn
SESSION_VERSION = 1


# Sessions are identified by uuid4().hex[:12] — a 12-char lowercase hex string.
# Reject anything else BEFORE touching the filesystem so a path like
# "../../etc/passwd" can't masquerade as a session id and cause a stat() / read
# of files outside state/agent_sessions/.
_VALID_SESSION_ID_RE = re.compile(r"^[0-9a-f]{12}$")


def is_valid_session_id(sid: str) -> bool:
    return bool(sid and _VALID_SESSION_ID_RE.fullmatch(sid))


@dataclass
class Session:
    session_id: str
    title: str
    messages: list[dict]                            # OpenAI-style {role, content}
    engine_config: engine.EngineConfig
    tool_state: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    finished: bool = False
    pinned: bool = False                            # surfaced at the top of the sidebar

    def to_dict(self) -> dict:
        # NEVER include api_key on the wire — to_public_dict() masks it. The
        # session is the response shape for /agent/sessions/<id>; without this,
        # opening DevTools or any browser cache shows the raw key.
        return {
            "version": SESSION_VERSION,
            "session_id": self.session_id,
            "title": self.title,
            "messages": self.messages,
            "engine_config": self.engine_config.to_public_dict(),
            "tool_state": self.tool_state,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "finished": self.finished,
            "pinned": self.pinned,
        }

    def to_persisted_dict(self) -> dict:
        """Disk shape — INCLUDES api_key so a panel restart doesn't lose it."""
        return {
            "version": SESSION_VERSION,
            "session_id": self.session_id,
            "title": self.title,
            "messages": self.messages,
            "engine_config": asdict(self.engine_config),
            "tool_state": self.tool_state,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "finished": self.finished,
            "pinned": self.pinned,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Session":
        return cls(
            session_id=d["session_id"],
            title=d.get("title", "Untitled"),
            messages=d.get("messages", []),
            engine_config=engine.EngineConfig(**(d.get("engine_config") or {})),
            tool_state=d.get("tool_state", {}),
            created_at=d.get("created_at", time.time()),
            updated_at=d.get("updated_at", time.time()),
            finished=d.get("finished", False),
            pinned=d.get("pinned", False),
        )


# ---- Storage --------------------------------------------------------------
def sessions_dir(state_dir: Path) -> Path:
    d = state_dir / "agent_sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_session(session: Session, state_dir: Path) -> None:
    session.updated_at = time.time()
    d = sessions_dir(state_dir)
    p = d / f"{session.session_id}.json"
    # Persisted shape keeps api_key so the conversation can resume after a
    # panel restart without the user re-entering their key. The file is
    # therefore secret-bearing and must NEVER appear in a permissive
    # mode, even momentarily — the previous tmp.write_text → os.replace
    # → chmod 0o600 sequence left the file world-readable for the
    # window between replace and chmod (≈microseconds, but still real).
    # Use os.open with O_CREAT|O_EXCL|0o600 so the file is born private,
    # then fsync + atomic rename. No other process can ever see a
    # readable handle.
    payload = json.dumps(session.to_persisted_dict(), indent=2).encode("utf-8")
    tmp_path = d / f".{p.name}.{os.getpid()}.tmp"
    fd = os.open(
        tmp_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_TRUNC,
        0o600,
    )
    try:
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(payload)
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except OSError:
                    pass
            # umask on some systems would have widened the mode the
            # kernel sets; chmod the temp explicitly before the rename.
            try:
                os.chmod(tmp_path, 0o600)
            except OSError:
                pass
            os.replace(tmp_path, p)                  # atomic — never see half-write on read
            # Belt-and-braces: chmod the final path too. os.replace
            # preserves the source's mode on most filesystems but a
            # paranoid extra chmod is free and protects against weird
            # mounts.
            try:
                os.chmod(p, 0o600)
            except OSError:
                pass
        except Exception:
            # Clean up the temp file if anything went wrong before
            # rename; otherwise the dir slowly accumulates orphaned
            # `.<id>.json.<pid>.tmp` files.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    finally:
        # If os.fdopen never ran (rare), close the descriptor manually.
        try:
            os.close(fd)
        except OSError:
            pass


def load_session(session_id: str, state_dir: Path) -> Session | None:
    if not is_valid_session_id(session_id):
        return None
    p = sessions_dir(state_dir) / f"{session_id}.json"
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return Session.from_dict(data)


def list_sessions(state_dir: Path) -> list[dict]:
    """Compact summaries — id, title, message count, last update.

    Includes:
      preview: first ~80 chars of the first user-text message (skipping
               attachment header and tool-result wrappers).
      pinned: surface above all time buckets in the sidebar.
      submitted_shot_ids: live cross-reference for the "running" dot —
                          the UI matches these against the panel queue.
    """
    out = []
    d = sessions_dir(state_dir)
    if not d.is_dir():
        return out
    for p in sorted(d.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        msgs = data.get("messages") or []
        preview = ""
        for m in msgs:
            if m.get("role") != "user":
                continue
            c = (m.get("content") or "").strip()
            if not c or c.startswith("<tool_result"):
                continue
            # Strip leading <attachments>...</attachments> block.
            if c.startswith("<attachments>"):
                end = c.find("</attachments>")
                if end != -1:
                    c = c[end + len("</attachments>"):].strip()
            preview = c[:120].replace("\n", " ").strip()
            if len(c) > 120:
                preview += "…"
            break
        sub_shots = (data.get("tool_state") or {}).get("submitted_shots") or []
        shot_ids = [s.get("job_id") for s in sub_shots if isinstance(s, dict) and s.get("job_id")]
        out.append({
            "session_id": data.get("session_id", p.stem),
            "title": data.get("title", "Untitled"),
            "messages": len(msgs),
            "shots_submitted": len(sub_shots),
            "submitted_shot_ids": shot_ids,
            "preview": preview,
            "pinned": bool(data.get("pinned", False)),
            "updated_at": data.get("updated_at", p.stat().st_mtime),
            "created_at": data.get("created_at", p.stat().st_ctime),
            "finished": data.get("finished", False),
        })
    return out


def delete_session(session_id: str, state_dir: Path) -> bool:
    if not is_valid_session_id(session_id):
        return False
    p = sessions_dir(state_dir) / f"{session_id}.json"
    if p.is_file():
        p.unlink()
        return True
    return False


def rename_session(session_id: str, new_title: str, state_dir: Path) -> bool:
    if not is_valid_session_id(session_id):
        return False
    sess = load_session(session_id, state_dir)
    if sess is None:
        return False
    new_title = (new_title or "").strip()[:120]
    if not new_title:
        return False
    sess.title = new_title
    save_session(sess, state_dir)
    return True


def set_pinned(session_id: str, pinned: bool, state_dir: Path) -> bool:
    if not is_valid_session_id(session_id):
        return False
    sess = load_session(session_id, state_dir)
    if sess is None:
        return False
    sess.pinned = bool(pinned)
    save_session(sess, state_dir)
    return True


def new_session(*, title: str, engine_config: engine.EngineConfig) -> Session:
    sid = uuid.uuid4().hex[:12]
    return Session(
        session_id=sid,
        title=title or f"Session {sid}",
        messages=[],
        engine_config=engine_config,
        tool_state={"session_id": sid, "title": title},
    )


# ---- The turn loop --------------------------------------------------------
@dataclass
class TurnEvent:
    """One observable event during a turn — for the UI's progressive feed."""
    kind: str                                       # "assistant" | "tool_call" | "tool_result" | "error" | "done"
    payload: dict


def run_turn(session: Session, user_message: str | None,
             panel_ops: tools.PanelOps,
             *,
             tools_doc: str,
             system_prompt_overrides: dict | None = None,
             max_steps: int = MAX_STEPS_PER_TURN,
             on_event: Callable[[TurnEvent], None] | None = None,
             ) -> Iterator[TurnEvent]:
    """Drive one user→assistant turn, including any tool-loop iterations.

    Yields TurnEvent objects so the panel HTTP layer can stream progress
    to the UI. The same events are also fed to `on_event` if provided
    (use this when the caller is collecting into a list rather than
    iterating).

    `user_message` may be None — in that case we resume an in-flight
    turn (e.g. after a panel restart left a half-finished session).
    """
    def emit(ev: TurnEvent):
        if on_event is not None:
            try:
                on_event(ev)
            except Exception:                       # noqa: BLE001
                pass
        return ev

    # 1. Inject (or refresh) the system prompt at index 0. The project
    # notes excerpt is read fresh every turn so a recently appended
    # memory becomes visible immediately on the next turn.
    from agent import project as _project
    notes_excerpt = ""
    try:
        notes_excerpt = _project.read_notes_excerpt(panel_ops.state_dir)
    except Exception:                                  # noqa: BLE001
        pass
    sys_prompt = prompts.build_system_prompt(
        capabilities=panel_ops.capabilities,
        tools_doc=tools_doc,
        repo_version=(system_prompt_overrides or {}).get("version", "v2.0.4"),
        project_notes=notes_excerpt,
    )
    if not session.messages or session.messages[0].get("role") != "system":
        session.messages.insert(0, {"role": "system", "content": sys_prompt})
    else:
        session.messages[0]["content"] = sys_prompt

    if user_message is not None:
        session.messages.append({"role": "user", "content": user_message})
        # Clear finished flag — a new user message resets the loop.
        session.finished = False

    # 2. Drive the loop.
    for step_i in range(max_steps):
        # Call engine
        try:
            result = engine.chat(session.messages, session.engine_config)
        except Exception as e:                      # noqa: BLE001
            err_msg = f"engine error: {e}"
            session.messages.append(
                {"role": "user",
                 "content": f"<tool_result>{{\"ok\":false,\"error\":\"{_escape(err_msg)}\"}}</tool_result>"})
            yield emit(TurnEvent("error", {"error": err_msg}))
            return

        assistant_content = result.content or ""
        # Persist reasoning alongside content so it survives panel restart
        # and the UI can re-render it on session load. Most chat models
        # don't return reasoning; the field is empty in that case.
        msg_dict: dict = {"role": "assistant", "content": assistant_content}
        if getattr(result, "reasoning", None):
            msg_dict["reasoning"] = result.reasoning
        session.messages.append(msg_dict)
        yield emit(TurnEvent("assistant", {
            "content": assistant_content,
            "reasoning": getattr(result, "reasoning", "") or "",
            "step": step_i,
            "model": result.model,
            "usage": result.usage,
        }))

        # 3. Look for an action block.
        action = tools.parse_action_block(assistant_content)
        if action is None:
            # Plain text reply, conversation pauses awaiting next user msg.
            yield emit(TurnEvent("done", {"reason": "no_action", "step": step_i}))
            return

        tool_name = action.get("tool", "")
        tool_args = action.get("args") or {}
        yield emit(TurnEvent("tool_call", {
            "tool": tool_name, "args": tool_args, "step": step_i,
        }))

        result_obj = tools.dispatch(tool_name, tool_args, panel_ops, session.tool_state)
        yield emit(TurnEvent("tool_result", {
            "tool": tool_name, "result": result_obj, "step": step_i,
        }))

        # 4. Append tool result as a user-role message.
        tool_result_msg = (
            f"<tool_result tool=\"{tool_name}\">\n"
            f"{json.dumps(result_obj, default=str)}\n"
            f"</tool_result>"
        )
        session.messages.append({"role": "user", "content": tool_result_msg})

        # 5. If the model called `finish`, OR a tool set the
        # `_finish_after_turn` flag in tool_state (e.g. submit_shots
        # with auto_finish=true — needed because the LTX worker pauses
        # the local chat engine the moment it picks up the FIRST job
        # from the queue, so any subsequent chat call would fail with
        # "Connection error"; the agent must commit the whole batch
        # AND signal completion in a single tool dispatch), stop the
        # loop.
        if tool_name == "finish" or session.tool_state.pop("_finish_after_turn", False):
            session.finished = True
            yield emit(TurnEvent("done", {"reason": "finished", "step": step_i}))
            return

    # Hit the cap — emit a synthetic "done" so the UI doesn't hang.
    yield emit(TurnEvent("done", {"reason": "max_steps_hit", "step": max_steps}))


def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("\"", "\\\"")


# ---- Tools doc renderer ----------------------------------------------------
def list_tool_catalog() -> list[dict]:
    """Structured catalog of every registered tool — for the UI's
    capabilities sheet (Phase 0 #0.5 from the roadmap). One row per
    tool with `name`, `summary` (first non-empty line of the docstring),
    and `body` (rest, trimmed). Stable order matches `render_tools_doc`.
    """
    order = [
        "estimate_shot", "submit_shot", "submit_shots", "get_queue_status",
        "wait_for_shot", "extract_frame", "inspect_clip",
        "generate_shot_images", "get_selected_anchors", "upload_image",
        "read_document", "list_loras", "get_master_style",
        "read_project_notes", "append_project_notes",
        "write_session_manifest", "finish",
    ]
    seen = set()
    out: list[dict] = []
    def _row(name: str) -> dict | None:
        fn = tools.TOOL_HANDLERS.get(name)
        if fn is None:
            return None
        doc = (fn.__doc__ or "").strip()
        if not doc:
            return {"name": name, "summary": "", "body": ""}
        lines = doc.split("\n")
        summary = lines[0].strip()
        body = "\n".join(lines[1:]).strip()
        return {"name": name, "summary": summary, "body": body}
    for name in order:
        row = _row(name)
        if row is not None:
            out.append(row)
            seen.add(name)
    for name in sorted(tools.TOOL_HANDLERS.keys()):
        if name in seen:
            continue
        row = _row(name)
        if row is not None:
            out.append(row)
    return out


def render_tools_doc() -> str:
    """Format the registered tools' docstrings for the system prompt.

    Each tool's docstring is included verbatim so the agent has the same
    contract as the dispatcher. Keep tool docstrings tight and accurate.
    """
    sections = []
    # Stable order so the prompt is reproducible across reboots. Order
    # groups by workflow phase rather than alphabetically:
    #   1. planning          — estimate_shot
    #   2. shot composition  — generate_shot_images, get_selected_anchors,
    #                          inspect_clip, extract_frame, upload_image,
    #                          list_library_images
    #   3. submission + wait — submit_shot, get_queue_status, wait_for_shot
    #   4. style + memory    — list_loras, get_master_style,
    #                          read_project_notes, append_project_notes,
    #                          read_document
    #   5. delivery          — write_session_manifest, finish
    # Adding a tool? Insert it in the right phase here AND keep its
    # @tool name aligned with the registry. Out-of-list names land at
    # the bottom alphabetically (the trailing fallback below).
    order = [
        # planning
        "estimate_shot",
        # shot composition (anchor stills + reference assets)
        "generate_shot_images", "get_selected_anchors", "list_library_images",
        "inspect_clip", "extract_frame", "upload_image",
        # submission + wait
        "submit_shot", "submit_shots", "get_queue_status", "wait_for_shot",
        # style + memory
        "list_loras", "get_master_style",
        "read_project_notes", "append_project_notes", "read_document",
        # delivery
        "write_session_manifest", "finish",
    ]
    for name in order:
        fn = tools.TOOL_HANDLERS.get(name)
        if fn is None:
            continue
        doc = (fn.__doc__ or "").strip()
        sections.append(f"## `{name}`\n\n{doc}")
    # Any new tools registered later land at the bottom alphabetically.
    for name in sorted(tools.TOOL_HANDLERS.keys()):
        if name in order:
            continue
        fn = tools.TOOL_HANDLERS[name]
        doc = (fn.__doc__ or "").strip()
        sections.append(f"## `{name}`\n\n{doc}")
    return "\n\n".join(sections)
