"""Phase 2 runtime: smolagents CodeAgent — drop-in for runtime.run_turn.

Same `Session` shape, same `TurnEvent` events, same `panel_ops` contract,
same on-disk JSON format. The difference: instead of our hand-rolled
fenced-block ReAct loop, the model emits Python code blocks that
smolagents' AST-based `LocalPythonExecutor` runs against our 17 tools.

Why bother:
  - Frees us from the mlx-lm KV-cache merge bug (smolagents doesn't
    rely on multi-request prompt caches the way our loop does).
  - Frees us from manual role-alternation collapsing (smolagents does
    its own message normalization internally).
  - Drops our 30-step arbitrary cap in favor of smolagents' configurable
    max_steps.
  - LLM emits code, which composes far better for our use case
    (chain `generate_shot_images(...)` → pick best → `submit_shot(...)`
    in one step instead of 3 round trips).

Why bound the eval to a parallel module rather than replacing
`runtime.py` directly: this is Phase 2 of the plan in
/Users/salo/.claude/plans/fancy-conjuring-lovelace.md — we're A/B'ing
both runtimes against the Bizarro pipeline before committing.

Toggle which runtime the panel uses via the env var
`PHOSPHENE_RUNTIME`:
  - unset / "legacy" → runtime.run_turn (default; what's shipping today)
  - "smol"           → runtime_smol.run_turn (this file)

The panel reads the var on each `run_turn` call so flipping it doesn't
require a panel restart.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Iterator

# smolagents may not be installed on older venvs (panels that haven't
# run update.js since this commit). Degrade gracefully — `run_turn`
# below falls through to runtime.run_turn instead of crashing on import.
try:
    from smolagents import CodeAgent, OpenAIModel, LiteLLMModel, Tool
    _HAS_SMOLAGENTS = True
    _SMOL_IMPORT_ERR = ""
except Exception as _e:                         # noqa: BLE001
    _HAS_SMOLAGENTS = False
    _SMOL_IMPORT_ERR = f"{type(_e).__name__}: {_e}"

from agent import engine, tools
# Reuse the legacy session + event dataclasses + system-prompt builder.
# DO NOT depend on agent.runtime.run_turn here — that creates a cycle
# only resolvable via lazy import at the fallback site below.
from agent.runtime import Session, TurnEvent, MAX_STEPS_PER_TURN
from agent import prompts


# ---- Public surface ---------------------------------------------------------
def is_smolagents_active() -> bool:
    """Diagnostic helper — surfaces in /agent/config GETs so the panel
    can show 'smolagents' vs 'legacy' in the status row."""
    return _HAS_SMOLAGENTS


def smolagents_import_error() -> str:
    """If is_smolagents_active() is False, this explains why."""
    return _SMOL_IMPORT_ERR


# ---- Tool input schemas (smolagents requires structured types) -------------
# smolagents validates tool args by these schemas. We declare them up-
# front instead of trying to introspect tools.py's @tool decorators —
# our handlers all take `args: dict` so signature inspection is useless.
#
# Schema shape per smolagents docs:
#   {
#     "<arg_name>": {
#       "type": "string" | "integer" | "number" | "boolean" | "array" | "object" | "null",
#       "description": "...",
#       "nullable": True,    # optional — defaults to False
#     },
#     ...
#   }
#
# Every arg here defaults to nullable=True so the agent can omit them.
# The underlying handler in tools.py is the source of truth for which
# args are required at runtime — bad calls return validation errors
# the agent can self-correct from.
_TOOL_SCHEMAS: dict[str, dict[str, dict]] = {
    "estimate_shot": {
        "duration_seconds": {"type": "number", "description": "Target clip length in seconds (e.g. 4.0).", "nullable": True},
        "quality": {"type": "string", "description": "'standard' or 'high' (Q4 vs Q8).", "nullable": True},
        "accel": {"type": "string", "description": "'distilled' (default) or 'base'.", "nullable": True},
        "mode": {"type": "string", "description": "'t2v', 'i2v', 'extend', or 'keyframe'.", "nullable": True},
        "sharp": {"type": "boolean", "description": "Sharper-but-slower temporal stride.", "nullable": True},
    },
    "submit_shot": {
        "prompt": {"type": "string", "description": "Full text prompt. Use the master-style suffix for visual continuity."},
        "mode": {"type": "string", "description": "'t2v' (default), 'i2v', 'extend', or 'keyframe'.", "nullable": True},
        "quality": {"type": "string", "description": "'standard' (Q4) or 'high' (Q8).", "nullable": True},
        "accel": {"type": "string", "description": "'distilled' (default) or 'base'.", "nullable": True},
        "duration_seconds": {"type": "number", "description": "Target clip duration.", "nullable": True},
        "width": {"type": "integer", "description": "Output width in px.", "nullable": True},
        "height": {"type": "integer", "description": "Output height in px.", "nullable": True},
        "no_music": {"type": "boolean", "description": "Skip the music/score track.", "nullable": True},
        "seed": {"type": "integer", "description": "Render seed (0 = random).", "nullable": True},
        "label": {"type": "string", "description": "Short label (e.g. 'S1', 'opening').", "nullable": True},
        "ref_image_path": {"type": "string", "description": "Path to keyframe / reference image.", "nullable": True},
        "source_clip": {"type": "string", "description": "Path to source video for extend.", "nullable": True},
        "extend_seconds": {"type": "number", "description": "Length of the extend.", "nullable": True},
        "extend_direction": {"type": "string", "description": "'forward' or 'backward'.", "nullable": True},
        "keyframes": {"type": "array", "description": "Multi-keyframe anchors: list of {image_path, time_seconds}.", "nullable": True},
        "loras": {"type": "array", "description": "List of LoRA names + weights to apply.", "nullable": True},
        "upscale_method": {"type": "string", "description": "Optional upscaler.", "nullable": True},
        "upscale": {"type": "boolean", "description": "Whether to upscale.", "nullable": True},
        "session_tag": {"type": "string", "description": "Optional session correlation tag.", "nullable": True},
    },
    "get_queue_status": {},
    "wait_for_shot": {
        "timeout_seconds": {"type": "number", "description": "Max seconds to wait.", "nullable": True},
        "poll_seconds": {"type": "number", "description": "Polling interval.", "nullable": True},
    },
    "extract_frame": {
        "clip_path": {"type": "string", "description": "Path to source clip.", "nullable": True},
        "job_id": {"type": "string", "description": "Alt: job id whose output to extract.", "nullable": True},
        "which": {"type": "string", "description": "'first', 'last', 'middle', or frame index.", "nullable": True},
        "output_name": {"type": "string", "description": "Output filename.", "nullable": True},
    },
    "write_session_manifest": {
        "title": {"type": "string", "description": "Optional title for the manifest.", "nullable": True},
        "shot_order": {"type": "array", "description": "Ordered list of job ids or labels.", "nullable": True},
        "output_name": {"type": "string", "description": "Output filename (will be sanitized).", "nullable": True},
    },
    "generate_shot_images": {
        "shot_label": {"type": "string", "description": "Label for the shot (e.g. 'S1').", "nullable": True},
        "prompt": {"type": "string", "description": "Full image prompt.", "nullable": True},
        "n": {"type": "integer", "description": "Number of candidates (default 4).", "nullable": True},
        "aspect": {"type": "string", "description": "'16:9', '9:16', '1:1', '21:9'.", "nullable": True},
        "seed_base": {"type": "integer", "description": "Base seed (0 = random).", "nullable": True},
        "append": {"type": "boolean", "description": "Append to existing batch instead of replacing.", "nullable": True},
        "refs": {"type": "array", "description": "1-3 reference image paths (Qwen-Edit character lock).", "nullable": True},
        "engine_override": {"type": "string", "description": "Optional engine preset.", "nullable": True},
    },
    "inspect_clip": {
        "job_id": {"type": "string", "description": "Job id to inspect.", "nullable": True},
        "clip_path": {"type": "string", "description": "Or path to a clip.", "nullable": True},
    },
    "get_selected_anchors": {},
    # TODO: this schema dict duplicates the real signatures in
    # agent/tools.py — they drift (P2-4 fix below). A future cleanup
    # should derive these from one source of truth.
    "upload_image": {
        # Real handler: agent/tools.py:_upload_image — _required(args, "attachment_id").
        "attachment_id": {"type": "string", "description": "Absolute path of an uploaded image (the path the chat injected via <attachments>)."},
    },
    "read_document": {
        # Real handler: agent/tools.py:_read_document — _required(args, "path").
        "path": {"type": "string", "description": "Absolute path of the attached PDF or text file (from <attachments>)."},
        "max_chars": {"type": "integer", "description": "Truncation cap (default 80000).", "nullable": True},
    },
    "get_master_style": {
        "shot_label": {"type": "string", "description": "Label whose style to fetch.", "nullable": True},
    },
    "list_loras": {},
    "list_library_images": {
        "limit": {"type": "integer", "description": "Max results (default 50).", "nullable": True},
        "since": {"type": "string", "description": "ISO datetime filter.", "nullable": True},
        "contains": {"type": "string", "description": "Substring filter on filename.", "nullable": True},
        "session_id": {"type": "string", "description": "Restrict to one session's outputs.", "nullable": True},
        "shot_label": {"type": "string", "description": "Restrict to one shot.", "nullable": True},
        "include_manual": {"type": "boolean", "description": "Include manual studio outputs.", "nullable": True},
        "include_agent": {"type": "boolean", "description": "Include agent-generated outputs.", "nullable": True},
    },
    "read_project_notes": {},
    "append_project_notes": {
        "text": {"type": "string", "description": "Text to append to notes."},
        "kind": {"type": "string", "description": "'note', 'style', 'cast', or 'decision'.", "nullable": True},
    },
    "finish": {
        "summary": {"type": "string", "description": "One-line summary of what was completed.", "nullable": True},
        "next_steps": {"type": "string", "description": "Optional follow-up suggestions.", "nullable": True},
    },
}


# ---- Tool wrapper -----------------------------------------------------------
class _LegacyToolWrapper(Tool if _HAS_SMOLAGENTS else object):
    """Bridges one of our tools.TOOL_HANDLERS handlers into a smolagents Tool.

    Smolagents calls `forward(**kwargs)`; we re-pack into the dict shape
    `tools.dispatch()` expects, then unwrap the dispatch result (which
    is `{"ok", "result", "error"}`) into a smolagents-friendly shape.
    Errors are returned in `out["error"]` so the agent can self-correct
    on the next step.

    `skip_forward_signature_validation` tells smolagents NOT to inspect
    our `forward(**kwargs)` signature against `inputs` — we accept all
    arg names dynamically so a single class serves all 17 tools without
    code-generating one subclass per tool.
    """

    output_type = "object"
    skip_forward_signature_validation = True

    def __init__(self, name: str, description: str, inputs: dict,
                 panel_ops: tools.PanelOps, tool_state: dict,
                 on_event: Callable[[TurnEvent], None] | None):
        # Tool's __init__ reads class attrs `name`, `description`,
        # `inputs`, `output_type`. Set them on the instance BEFORE
        # super().__init__ runs the validation.
        self.name = name
        self.description = description
        self.inputs = inputs
        super().__init__()
        self._panel_ops = panel_ops
        self._tool_state = tool_state
        self._on_event = on_event

    def forward(self, **kwargs) -> Any:                 # noqa: ANN401
        # Drop None values — smolagents passes nullable args as None
        # which our handlers may interpret as a string "None".
        clean = {k: v for k, v in kwargs.items() if v is not None}
        # on_event fires the panel's mid-loop save callback so a long-
        # running tool (like generate_shot_images) shows progress.
        if self._on_event is not None:
            try:
                self._on_event(TurnEvent("tool_call", {
                    "tool": self.name, "args": clean, "step": -1,
                }))
            except Exception:                            # noqa: BLE001
                pass
        out = tools.dispatch(self.name, clean, self._panel_ops, self._tool_state)
        if self._on_event is not None:
            try:
                self._on_event(TurnEvent("tool_result", {
                    "tool": self.name, "result": out, "step": -1,
                }))
            except Exception:                            # noqa: BLE001
                pass
        # Unwrap dispatch envelope. The legacy runtime serializes the
        # full {"ok", "result", "error"} dict into a `<tool_result>`
        # message and asks the LLM to read all three fields. That's a
        # footgun for CodeAgent: the model writes `len(submit_shot(...))`
        # and gets 3 (the dict's key count) instead of meaningful data.
        # Smolagents' more idiomatic pattern: return the unwrapped
        # result on success, raise on failure. The agent's exception
        # handler sees the error string in the next-step observation.
        if out.get("ok"):
            return out.get("result")
        raise RuntimeError(out.get("error") or f"{self.name} failed")


def _build_smol_tools(panel_ops: tools.PanelOps, tool_state: dict,
                      on_event: Callable[[TurnEvent], None] | None
                      ) -> list:
    """Instantiate smolagents Tool wrappers for every handler we have."""
    out = []
    for name, handler in tools.TOOL_HANDLERS.items():
        schema = _TOOL_SCHEMAS.get(name, {})
        # Description comes from the handler's docstring — same source the
        # legacy `tools_doc` uses, so the agent sees identical guidance.
        doc = (handler.__doc__ or "").strip()
        # Trim to first paragraph for the inline description; the full
        # docstring already lands in the system prompt via tools_doc.
        first_para = doc.split("\n\n", 1)[0].strip() if doc else f"{name} tool"
        # Cap at 600 chars; smolagents prepends every tool description
        # to its own prompt template and we don't want to bloat that.
        if len(first_para) > 600:
            first_para = first_para[:600] + "…"
        out.append(_LegacyToolWrapper(
            name=name,
            description=first_para or f"Phosphene {name} tool",
            inputs=schema,
            panel_ops=panel_ops,
            tool_state=tool_state,
            on_event=on_event,
        ))
    return out


# ---- Model construction -----------------------------------------------------
def _build_smol_model(cfg: engine.EngineConfig):
    """Map our EngineConfig to a smolagents Model.

    smolagents has separate classes for each provider. We always go
    through OpenAIModel for OpenAI-compat endpoints (mlx-lm.server,
    custom proxies) and LiteLLMModel for Anthropic / Ollama where
    LiteLLM's translation is the cleanest path.
    """
    if cfg.kind == "anthropic":
        return LiteLLMModel(
            model_id=f"anthropic/{cfg.model or 'claude-sonnet-4-5'}",
            api_key=cfg.api_key or None,
            temperature=cfg.temperature,
            max_completion_tokens=cfg.max_tokens,
        )
    if cfg.kind == "ollama":
        return LiteLLMModel(
            model_id=f"ollama/{cfg.model}",
            api_base=cfg.base_url.rstrip("/"),
            temperature=cfg.temperature,
            max_completion_tokens=cfg.max_tokens,
        )
    if cfg.kind == "phosphene_local":
        # mlx-lm.server identifies models by absolute load path. Pass
        # local_model_path through as model_id; OpenAIModel forwards it
        # verbatim as the OpenAI request body's `model` field.
        wire_model = cfg.local_model_path or cfg.model
        return OpenAIModel(
            model_id=wire_model,
            api_base=cfg.base_url.rstrip("/"),
            api_key="not-needed",
            temperature=cfg.temperature,
            max_completion_tokens=cfg.max_tokens,
        )
    # "custom" — arbitrary OpenAI-compat endpoint (LM Studio, vLLM, etc.)
    return OpenAIModel(
        model_id=cfg.model,
        api_base=cfg.base_url.rstrip("/"),
        api_key=cfg.api_key or "not-needed",
        temperature=cfg.temperature,
        max_completion_tokens=cfg.max_tokens,
    )


# ---- Prior-conversation rendering ------------------------------------------
def _render_prior_conversation(messages: list[dict], *, max_chars: int = 8000) -> str:
    """Project prior session messages into a Markdown digest the next
    smolagents turn can read.

    smolagents' `agent.run(task, reset=True)` resets memory, so a fresh
    agent has no built-in awareness of prior turns. We synthesize that
    awareness by including a compact rendering of prior messages in the
    `instructions=` field passed to CodeAgent. The format mirrors what
    the legacy runtime sees in session.messages but is condensed:

      User: <text>
      Assistant: <text> [tool used: name → ok=true]
      ...

    Tool result chips are summarized to one line per call so the
    instructions don't balloon. If the rendered length exceeds
    max_chars, we truncate at the FRONT (oldest first), keeping the
    most recent activity which the agent is most likely to need.
    """
    if not messages:
        return ""
    lines = []
    skip_system = True
    for m in messages:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if skip_system and role == "system":
            skip_system = False
            continue
        if not content:
            continue
        if content.startswith("<tool_result"):
            # Compact tool-result rendering. Pull `tool="..."` and the
            # ok flag from the embedded JSON.
            import re as _re
            tool_m = _re.search(r'tool="([^"]+)"', content)
            tool_name = tool_m.group(1) if tool_m else "?"
            ok = "ok" if '"ok": true' in content or '"ok":true' in content else "fail"
            lines.append(f"  [tool {tool_name} → {ok}]")
            continue
        if content.startswith("<attachments>"):
            # Skip the attachment manifest block; the file paths are
            # already accessible via the panel's uploads_dir.
            end = content.find("</attachments>")
            if end != -1:
                content = content[end + len("</attachments>"):].strip()
            if not content:
                continue
        prefix = "User" if role == "user" else "Assistant" if role == "assistant" else role.title()
        # Cap any single message at 1500 chars to bound total size.
        if len(content) > 1500:
            content = content[:1500] + "…"
        lines.append(f"{prefix}: {content}")

    text = "\n".join(lines)
    if len(text) > max_chars:
        # Truncate from the front (drop oldest) so the recent context
        # survives.
        text = text[-max_chars:]
        # Re-anchor at a line boundary.
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1:]
        text = "[…older history truncated…]\n" + text
    return text


# ---- Main entry -------------------------------------------------------------
def run_turn(session: Session, user_message: str | None,
             panel_ops: tools.PanelOps,
             *,
             tools_doc: str,
             system_prompt_overrides: dict | None = None,
             max_steps: int = MAX_STEPS_PER_TURN,
             on_event: Callable[[TurnEvent], None] | None = None,
             ) -> Iterator[TurnEvent]:
    """Drive one user→assistant turn through smolagents.

    Same signature + iterator semantics as `runtime.run_turn` so the
    panel HTTP layer can call either implementation interchangeably
    based on the PHOSPHENE_RUNTIME env var.
    """
    def emit(ev: TurnEvent):
        if on_event is not None:
            try:
                on_event(ev)
            except Exception:                           # noqa: BLE001
                pass
        return ev

    # Fall back to legacy runtime if smolagents isn't on this venv.
    if not _HAS_SMOLAGENTS:
        from agent import runtime as _legacy
        yield from _legacy.run_turn(
            session, user_message, panel_ops,
            tools_doc=tools_doc,
            system_prompt_overrides=system_prompt_overrides,
            max_steps=max_steps, on_event=on_event,
        )
        return

    # 1. Build system prompt — same builder the legacy runtime uses,
    # so the LTX domain knowledge in prompts.py is fully preserved.
    from agent import project as _project
    notes_excerpt = ""
    try:
        notes_excerpt = _project.read_notes_excerpt(panel_ops.state_dir)
    except Exception:                               # noqa: BLE001
        pass
    sys_prompt = prompts.build_system_prompt(
        capabilities=panel_ops.capabilities,
        tools_doc=tools_doc,
        repo_version=(system_prompt_overrides or {}).get("version", "v2.0.4-smol"),
        project_notes=notes_excerpt,
    )

    # 2. Append user_message to session.messages so the legacy session
    # JSON shape is preserved.
    if user_message is not None:
        session.messages.append({"role": "user", "content": user_message})
        session.finished = False

    # Reset the per-turn submit budget so the cap restarts fresh on every
    # user message. Enforcement itself lives in `tools.dispatch` — this
    # call just zeroes the counter the centralized check reads. Without
    # this, a smol CodeAgent could call `submit_shots` past the 8-shot
    # cap (security review H4 — was previously enforced only inside
    # legacy runtime's loop, leaving _LegacyToolWrapper.forward exposed).
    tools.reset_submit_budget(session.tool_state)

    # 3. Build instructions = system prompt + condensed prior conversation.
    # smolagents merges this into its own ReAct system prompt template;
    # the agent sees both our LTX guidance AND smolagents' code-block
    # protocol description.
    prior = _render_prior_conversation(session.messages[:-1])  # exclude this turn
    instructions = sys_prompt
    if prior:
        instructions += "\n\n# Prior conversation in this session\n\n" + prior

    # 4. Construct the CodeAgent.
    #
    # `executor_kwargs={"timeout_seconds": ...}` bumps LocalPythonExecutor's
    # per-step Python execution cap from its 30 s default. Our tools are
    # SLOW (generate_shot_images ~2-4 min for an mflux Qwen-Edit batch,
    # wait_for_shot can poll for tens of minutes). 30 s would kill nearly
    # every meaningful turn. The cap is on the executor, not on the LLM
    # request — the LLM still has cfg.max_tokens room.
    try:
        smol_model = _build_smol_model(session.engine_config)
        smol_tools = _build_smol_tools(panel_ops, session.tool_state, on_event=emit)
        agent = CodeAgent(
            tools=smol_tools,
            model=smol_model,
            instructions=instructions,
            additional_authorized_imports=[],   # NO imports — only our tools
            max_steps=max_steps,
            verbosity_level=0,
            executor_kwargs={"timeout_seconds": 1800},  # 30 min per step
        )
    except Exception as e:                              # noqa: BLE001
        err_msg = f"smolagents init failed: {type(e).__name__}: {e}"
        session.messages.append({
            "role": "user",
            "content": f"<tool_result>{{\"ok\":false,\"error\":\"{_escape(err_msg)}\"}}</tool_result>",
        })
        yield emit(TurnEvent("error", {"error": err_msg}))
        return

    # 5. Run the turn.
    task = user_message or "Continue from where you left off."
    try:
        final_answer = agent.run(task, reset=True)
    except Exception as e:                              # noqa: BLE001
        err_msg = f"smolagents run failed: {type(e).__name__}: {e}"
        session.messages.append({
            "role": "user",
            "content": f"<tool_result>{{\"ok\":false,\"error\":\"{_escape(err_msg)}\"}}</tool_result>",
        })
        yield emit(TurnEvent("error", {"error": err_msg}))
        return

    # 6. Walk agent.memory.steps and emit one set of events per
    # ActionStep. Per-inner-tool events (tool_call / tool_result for
    # the underlying handler) were already pushed via on_event from
    # inside _LegacyToolWrapper.forward() during the run — they fire
    # the panel's mid-loop save callback for incremental persistence.
    # Here we emit ONE tool_call ("python_interpreter") + ONE
    # tool_result per step for the response shape the panel renders.
    for i, step in enumerate(agent.memory.steps):
        cls = type(step).__name__
        if cls != "ActionStep":
            continue
        mo = (getattr(step, "model_output", None) or "")
        # Persist the assistant's full output (Thought + Code) so the
        # session log shows what the model produced.
        session.messages.append({"role": "assistant", "content": mo})
        yield emit(TurnEvent("assistant", {
            "content": mo,
            "reasoning": "",
            "step": i,
            "model": session.engine_config.model,
            "usage": {},
        }))
        # The python code executed by smolagents IS the tool call.
        # Surface it as a python_interpreter tool_call so the panel
        # UI's chip rendering pipeline catches it.
        for tc in (getattr(step, "tool_calls", None) or []):
            yield emit(TurnEvent("tool_call", {
                "tool": getattr(tc, "name", "python_interpreter"),
                "args": {"code": getattr(tc, "arguments", "")},
                "step": i,
            }))
        # Observation = stdout + final value from the python execution.
        obs = (getattr(step, "observations", None) or "")
        action_output = getattr(step, "action_output", None)
        if obs or action_output is not None:
            result_payload = {
                "ok": True,
                "observations": obs,
                "action_output": action_output,
            }
            yield emit(TurnEvent("tool_result", {
                "tool": "python_interpreter",
                "result": result_payload,
                "step": i,
            }))
            # Append the observation as a user-role message so legacy
            # session readers see the same `<tool_result>` pattern.
            session.messages.append({
                "role": "user",
                "content": (
                    "<tool_result tool=\"python_interpreter\">\n"
                    f"{json.dumps(result_payload, default=str)}\n"
                    "</tool_result>"
                ),
            })

    # 7. Mark finished + emit done. CodeAgent calls `final_answer` to
    # terminate — when the agent decides it's done, that tool returns
    # the final answer string and smolagents stops the loop. We treat
    # any successful return as "finished".
    session.finished = True
    yield emit(TurnEvent("done", {
        "reason": "finished",
        "result": str(final_answer)[:2000] if final_answer is not None else "",
        "steps_taken": len([s for s in agent.memory.steps if type(s).__name__ == "ActionStep"]),
    }))


def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("\"", "\\\"")
