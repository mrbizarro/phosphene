"""Tools the agent can call from inside its replies.

Convention: the model emits actions as fenced blocks like

    ```action
    {"tool": "submit_shot", "args": {...}}
    ```

The runtime extracts the JSON, calls `dispatch(name, args, ops)`, and
feeds the result back as a `<tool_result>...</tool_result>` user message.
The model loops on its next turn.

Why no OpenAI tool-calling spec: that spec varies across servers and is
fragile when the model is small or local. Fenced JSON blocks are
universal — every Chat Completions server passes them through verbatim,
the model can self-correct on malformed output, and a human reading the
transcript can see exactly what the agent intended.

`PanelOps` is the only interface to the rest of Phosphene. The panel
constructs an instance and passes it to `dispatch()`; tests can pass a
fake. tools.py imports nothing from mlx_ltx_panel.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


# ---- PanelOps: the surface the agent calls --------------------------------
@dataclass
class PanelOps:
    """Callbacks that let the agent reach into the running panel.

    All callables are concrete — the panel wires them up at startup. We use
    a dataclass of callables rather than an abstract base class so a test
    harness can construct one inline with lambdas.
    """

    # Submit a fully-built job dict to the queue. Returns the job dict
    # (with an `id` field). The panel implementation calls make_job() then
    # appends to STATE['queue'] under STATE['lock'].
    submit_job: Callable[[dict], dict]

    # Query the queue + history + current. Returns a snapshot dict shaped
    # like {"running": bool, "current": dict|None, "queue": [...], "history": [...]}.
    queue_snapshot: Callable[[], dict]

    # Look up a specific job by id across queue+history+current. Returns
    # the job dict or None.
    find_job: Callable[[str], dict | None]

    # Where the panel writes outputs. Used for resolving relative paths and
    # placing per-session manifest files.
    outputs_dir: Path

    # Where uploaded images go. The agent's keyframe_image arg can be an
    # absolute path; if it's relative, we resolve under uploads_dir.
    uploads_dir: Path

    # Hardware capabilities — `{"max_dim_t2v": 1280, "max_dim_kf": 768,
    # "allows_q8": True, "tier": "standard"}`. Lets the agent clamp its
    # plan to what this Mac can actually render.
    capabilities: dict

    # Optional callback returning the list of installed user LoRAs.
    # Wired by the panel; tests pass a lambda. None means "no LoRA
    # browser available in this host" — list_loras returns empty.
    list_loras_fn: Callable[[], list[dict]] | None = None

    # Persistent state root (for project notes, sessions, configs). The
    # agent's project-memory tools write under here so notes survive
    # across sessions and panel restarts.
    state_dir: Path = field(default_factory=lambda: Path("state"))


# ---- Tool registry ---------------------------------------------------------
TOOL_HANDLERS: dict[str, Callable] = {}


def tool(name: str):
    def deco(fn):
        TOOL_HANDLERS[name] = fn
        return fn
    return deco


def dispatch(tool_name: str, args: dict, ops: PanelOps,
             session_state: dict) -> dict:
    """Run a tool and return a JSON-serializable result.

    Result shape: `{"ok": bool, "result": Any, "error": str|None}`.
    The caller wraps this in a <tool_result> message back to the model.
    """
    handler = TOOL_HANDLERS.get(tool_name)
    if handler is None:
        return {
            "ok": False,
            "result": None,
            "error": (
                f"unknown tool {tool_name!r}. "
                f"Available: {sorted(TOOL_HANDLERS.keys())}"
            ),
        }
    try:
        result = handler(args, ops, session_state)
        return {"ok": True, "result": result, "error": None}
    except _ToolValidationError as e:
        return {"ok": False, "result": None, "error": f"validation: {e}"}
    except Exception as e:                      # noqa: BLE001
        return {"ok": False, "result": None, "error": f"{type(e).__name__}: {e}"}


class _ToolValidationError(Exception):
    pass


def _required(args: dict, key: str, kind: type = str):
    if key not in args:
        raise _ToolValidationError(f"missing required arg: {key}")
    val = args[key]
    if kind is int:
        try:
            return int(val)
        except Exception:
            raise _ToolValidationError(f"{key} must be int, got {val!r}")
    if kind is float:
        try:
            return float(val)
        except Exception:
            raise _ToolValidationError(f"{key} must be float, got {val!r}")
    return val


# ---- The actual tools ------------------------------------------------------
# Tool docstrings are embedded into the system prompt verbatim — keep
# them tight, factual, and aligned with what the implementation actually
# does. The agent reads them every turn.

@tool("estimate_shot")
def _estimate_shot(args: dict, ops: PanelOps, session: dict) -> dict:
    """Return the empirical wall-time for a planned shot.

    Numbers come from the published M4 Max 64 GB benchmark table. Lets the
    agent budget the overnight queue without trial-and-error.

    Args:
      duration_seconds: 5 | 10 | 20 (or close — we round to the nearest measured point)
      quality: "quick" | "balanced" | "standard" | "high"
      accel: "exact" | "boost" | "turbo" (ignored for high)
      mode: "t2v" | "i2v" | "keyframe" | "extend"
      sharp: bool — adds ~26 s for the PiperSR pass
    """
    duration = float(args.get("duration_seconds", 5))
    quality = (args.get("quality") or "balanced").lower()
    accel = (args.get("accel") or "turbo").lower()
    mode = (args.get("mode") or "t2v").lower()
    sharp = bool(args.get("sharp", quality == "balanced"))

    seconds = _estimate_wall_seconds(
        duration=duration, quality=quality, accel=accel, mode=mode, sharp=sharp,
    )
    return {
        "estimate_wall_seconds": seconds,
        "estimate_wall_human": _fmt_dur(seconds),
        "params_used": {"duration_seconds": duration, "quality": quality,
                        "accel": accel, "mode": mode, "sharp": sharp},
        "note": ("Empirical from M4 Max 64 GB; other Macs vary. Tier-clamped "
                 "modes (FFLF, Extend on Comfortable) max out at 768 px."),
    }


def _estimate_wall_seconds(*, duration: float, quality: str, accel: str,
                           mode: str, sharp: bool) -> float:
    """Single source of truth for shot wall-time estimates.

    Anchored on the table in CLAUDE.md §0 + STATE.md §3. Linear-interpolated
    along T^1.5 for non-canonical durations. Conservative side: round up.
    """
    # Per-mode per-quality 5-second baseline, in seconds
    base_5s: dict = {
        ("t2v", "quick"):    134,    # 2 m 14 s, 640×480
        ("t2v", "balanced"): 210,    # 3 m 30 s, 1024×576 + 720p Sharp + Turbo
        ("t2v", "standard"): 460,    # 7 m 40 s exact / 326 s turbo (~5 m 26 s)
        ("t2v", "high"):     711,    # 11 m 51 s, Q8 two-stage HQ
        ("i2v", "quick"):    150,
        ("i2v", "balanced"): 217,    # 3 m 37 s
        ("i2v", "standard"): 471,    # 7 m 51 s
        ("i2v", "high"):     750,
        ("keyframe", "balanced"): 329,    # ~ 5 m 29 s @ 768 px clamp on Comfortable
        ("keyframe", "standard"): 329,
        ("keyframe", "high"):     390,
        ("extend", "balanced"): 950,      # +3 s pass, ~15 m 50 s
        ("extend", "standard"): 950,
        ("extend", "high"):     1100,
    }
    key = (mode, quality)
    if key not in base_5s:
        # Fall back to T2V Standard if the agent picked a weird combo.
        base = base_5s[("t2v", "standard")]
    else:
        base = base_5s[key]
    # Accel discount only for t2v/i2v Standard. CLAUDE.md says boost ≈ -17%, turbo ≈ -29%.
    if mode in ("t2v", "i2v") and quality == "standard":
        if accel == "boost":
            base *= 0.83
        elif accel == "turbo":
            base *= 0.71
    # Length scaling: per-step cost grows ~T^1.5 with frame count.
    if duration > 0 and duration != 5:
        base *= (duration / 5.0) ** 1.5
    if sharp and mode in ("t2v", "i2v") and quality in ("balanced", "quick"):
        # Sharp wall-time is amortized into balanced's 210 s baseline already;
        # only add it for explicit sharp-on-quick or sharp-on-standard which
        # aren't in the table.
        if quality != "balanced":
            base += 26
    return round(base)


def _fmt_dur(seconds: float) -> str:
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


@tool("submit_shot")
def _submit_shot(args: dict, ops: PanelOps, session: dict) -> dict:
    """Append a render job to the panel's existing FIFO queue.

    Args (most map 1:1 to the panel's job spec — see CLAUDE.md §12):
      prompt: str (required)        — LTX prompt; follow the docs format
                                      (single paragraph, voice descriptor on
                                      every speech beat, single quotes around
                                      dialogue, ~1 action per 2-3 s).
      mode: str = "t2v"             — "t2v" | "i2v" | "keyframe" | "extend"
      quality: str = "balanced"     — "quick" | "balanced" | "standard" | "high"
      accel: str = "turbo"          — "exact" | "boost" | "turbo" (ignored for high/extend/keyframe)
      duration_seconds: float = 5   — clip length. Mapped to frames as 24*duration+1, snapped to 8k+1.
      width, height: int            — only for t2v. Falls back to the quality default.
      ref_image_path: str           — for i2v: absolute path to a previously-uploaded reference.
      keyframes: list[dict]         — for keyframe mode: [{image_path: str, frame_index: int}, ...]
                                      ≥2 items. Indices must be in [0, frames-1] and strictly increasing.
                                      THREE OR MORE keyframes is the path to multi-shot character
                                      continuity — see the system prompt's Character lock section.
      loras: list[dict]             — optional, works across ALL modes:
                                      [{name: "filename.safetensors", strength: 0.8}, ...]
                                      `name` must match a filename returned by list_loras().
                                      Strength clamped to [-2.0, 2.0]. Common values: 0.7-1.0
                                      for character-identity LoRAs; 0.5-0.8 for style LoRAs.
                                      For character lock across a multi-shot scene, pass the
                                      same LoRA on every submit_shot call (see system prompt).
      source_clip: str              — for extend mode: absolute path to source mp4.
      extend_seconds: float = 2     — for extend: how much to add. Maps to extend_frames.
      label: str                    — human-readable label visible in queue/history.
      no_music: bool = True         — when true (default), suppresses background
                                      music. Documentary work, dialogue scenes,
                                      and most agent batches don't want music
                                      under them. Pass `no_music: false` only
                                      when the user explicitly asks for music.
      upscale_method: str           — "lanczos" (default) | "pipersr" (Sharp ANE).
      session_tag: str              — copied to params for later filtering. The runtime
                                      passes the active agent session id automatically.

    Returns: {"job_id": "...", "queued_position": int, "estimated_wall_seconds": int, ...}
    """
    prompt = (args.get("prompt") or "").strip()
    if not prompt:
        raise _ToolValidationError("prompt is required and must not be empty")

    mode = (args.get("mode") or "t2v").lower()
    if mode not in ("t2v", "i2v", "keyframe", "extend"):
        raise _ToolValidationError(f"mode must be t2v|i2v|keyframe|extend, got {mode!r}")

    quality = (args.get("quality") or "balanced").lower()
    if quality not in ("quick", "balanced", "standard", "high"):
        raise _ToolValidationError(f"quality must be quick|balanced|standard|high, got {quality!r}")

    # accel: panel uses "off"/"boost"/"turbo" (panel UI labels "off" as
    # "Exact" — accept that alias for friendliness). Reject anything else
    # explicitly so a typo like accel="distilled" fails fast instead of
    # silently coercing to "turbo" and producing surprisingly motion-poor
    # output. (This is the bug we hit in session fd13625972ee — the agent
    # passed "distilled", got silently turned into "turbo", and the user
    # wondered why every clip was barely-moving.)
    raw_accel = (args.get("accel") or "turbo").lower()
    if raw_accel == "exact":
        accel = "off"
    elif raw_accel in ("off", "boost", "turbo"):
        accel = raw_accel
    else:
        raise _ToolValidationError(
            f"accel must be off|boost|turbo (or alias 'exact' for off), "
            f"got {raw_accel!r}. Default is 'turbo'. 'off' = full sampler "
            f"every step (best motion); 'boost' = ~10-15% faster; 'turbo' = "
            f"~20-25% faster (skips up to 3 steps via cache — costs motion "
            f"on subtle scenes)."
        )
    # Modes that don't allow acceleration are normalized later by the panel
    # but we mirror the rule here so estimates match what the queue runs.
    if mode in ("extend", "keyframe") or quality == "high":
        accel = "off"

    duration = float(args.get("duration_seconds", 5.0))
    if duration <= 0:
        raise _ToolValidationError("duration_seconds must be > 0")
    # LTX latents are groups of 8: frames = 24*duration + 1, snapped to 8k+1.
    target_frames = int(round(24 * duration)) + 1
    target_frames = ((target_frames - 1) // 8) * 8 + 1
    target_frames = max(25, target_frames)

    # Tier clamps (mirrors panel's HW tier system; agent is told these
    # via the system prompt, but enforce here as a guardrail).
    caps = ops.capabilities or {}
    max_dim_kf = int(caps.get("max_dim_kf") or 768)
    max_dim_t2v = int(caps.get("max_dim_t2v") or 1280)

    if mode == "t2v" or mode == "i2v":
        if quality == "quick":
            w_default, h_default = 640, 480
        elif quality in ("standard", "high"):
            w_default, h_default = 1280, 704
        else:
            w_default, h_default = 1024, 576
    else:
        # FFLF / Extend: clamp to tier
        w_default, h_default = min(768, max_dim_kf), min(416, max_dim_kf)

    width = int(args.get("width") or w_default)
    height = int(args.get("height") or h_default)
    if mode in ("t2v", "i2v"):
        width = min(width, max_dim_t2v)
        height = min(height, max_dim_t2v)
    else:
        width = min(width, max_dim_kf)
        height = min(height, max_dim_kf)

    # No-music defaults to TRUE for agent shots — the user's standing
    # preference. They expressed it explicitly: "by default no music".
    # Override per-shot when the user asks for music ("make this scene
    # with a piano track" → no_music: false).
    no_music = args.get("no_music")
    if no_music is None:
        no_music = True
    # Build the form dict that make_job() consumes.
    form: dict[str, str] = {
        "mode": mode,
        "prompt": prompt + (
            ". Audio: no music whatsoever, dialogue and ambient room tone only, no soundtrack."
            if no_music else ""
        ),
        "quality": quality,
        "accel": accel,
        "frames": str(target_frames),
        "width": str(width),
        "height": str(height),
        "seed": str(args.get("seed", -1)),
        "preset_label": (args.get("label") or "")[:80],
        "open_when_done": "off",
        "stop_comfy": "off",
    }

    # ---- Path-traversal hardening ----------------------------------------
    # I2V refs and keyframe images can live under uploads/ (user uploads,
    # agent-generated stills) OR outputs/ (frame extracts from prior
    # renders). Extend's source_clip must be an actual rendered clip, so
    # outputs-only is the right gate. Without these checks, a prompt-
    # injected agent could enqueue a job naming any readable file on disk
    # (and for I2V, the file's PIL-decodable bytes get encoded into the
    # video — a data-exfil channel). Same _ensure_under pattern the
    # extract_frame and refs paths already use.
    if mode == "i2v":
        ref = args.get("ref_image_path") or ""
        if not ref:
            raise _ToolValidationError("ref_image_path is required for mode=i2v")
        ref_str = _resolve_path(ref, ops.uploads_dir)
        try:
            ref_p = _ensure_under(Path(ref_str), [ops.uploads_dir, ops.outputs_dir])
        except _ToolValidationError as e:
            raise _ToolValidationError(
                f"ref_image_path must live under uploads/outputs: {ref!r} ({e})"
            ) from e
        if not ref_p.is_file():
            raise _ToolValidationError(f"ref_image_path not found: {ref_p}")
        form["image"] = str(ref_p)

    elif mode == "extend":
        src = args.get("source_clip") or ""
        if not src:
            raise _ToolValidationError("source_clip is required for mode=extend")
        src_str = _resolve_path(src, ops.outputs_dir)
        try:
            # Outputs-only: source_clip must be a previously-rendered video
            # the user owns. Restricting to outputs/ rules out an attempt
            # to extend an arbitrary mp4 from elsewhere on disk.
            src_p = _ensure_under(Path(src_str), [ops.outputs_dir])
        except _ToolValidationError as e:
            raise _ToolValidationError(
                f"source_clip must live under outputs: {src!r} ({e})"
            ) from e
        if not src_p.is_file():
            raise _ToolValidationError(f"source_clip not found: {src_p}")
        ext_secs = float(args.get("extend_seconds", 2.0))
        # Each extend latent = 8 video frames at 24 fps. So latents = secs * 24 / 8 = secs * 3.
        ext_latents = max(1, int(round(ext_secs * 3)))
        form["video_path"] = str(src_p)
        form["extend_frames"] = str(ext_latents)
        form["extend_direction"] = (args.get("extend_direction") or "after")
        form["extend_steps"] = "12"
        form["extend_cfg"] = "1.0"

    elif mode == "keyframe":
        kfs = args.get("keyframes") or []
        if not isinstance(kfs, list) or len(kfs) < 2:
            raise _ToolValidationError(
                "keyframes must be a list with ≥2 items "
                "(each {image_path, frame_index})"
            )
        # Validate ordering + bounds, resolve paths, enforce containment
        last_idx = -1
        normalized = []
        for kf in kfs:
            if not isinstance(kf, dict):
                raise _ToolValidationError("each keyframe must be a dict")
            img_str = _resolve_path(kf.get("image_path", ""), ops.uploads_dir)
            try:
                img_p = _ensure_under(Path(img_str), [ops.uploads_dir, ops.outputs_dir])
            except _ToolValidationError as e:
                raise _ToolValidationError(
                    f"keyframe image_path must live under uploads/outputs: "
                    f"{kf.get('image_path')!r} ({e})"
                ) from e
            if not img_p.is_file():
                raise _ToolValidationError(f"keyframe image not found: {img_p}")
            idx = int(kf.get("frame_index", 0))
            if idx < 0 or idx >= target_frames:
                raise _ToolValidationError(
                    f"frame_index {idx} out of range [0, {target_frames - 1}]"
                )
            if idx <= last_idx:
                raise _ToolValidationError("frame_index values must strictly increase")
            last_idx = idx
            normalized.append({"image_path": str(img_p), "frame_index": idx})
        # The first and last keyframes drop into the legacy two-keyframe panel
        # contract; intermediate keyframes go through the new `keyframes_json`
        # form field (see Layer 2 in mlx_ltx_panel.py).
        form["start_image"] = normalized[0]["image_path"]
        form["end_image"] = normalized[-1]["image_path"]
        if len(normalized) > 2:
            form["keyframes_json"] = json.dumps(normalized)
            form["keyframes_total_frames"] = str(target_frames)

    # ---- Optional LoRAs (works across all modes) ------------------------
    # Resolves filenames against installed LoRAs (so the agent can't apply
    # something that isn't on disk), then emits the JSON shape the panel's
    # parse_loras_from_form() consumes: [{path, strength}, ...].
    #
    # Character-lock workflow: agent reads the locked character LoRA from
    # project notes on every shot and passes it here. See the system prompt's
    # "Character lock via LoRA" section.
    loras_in = args.get("loras")
    if loras_in:
        if not isinstance(loras_in, list):
            raise _ToolValidationError(
                "loras must be a list of {name, strength} dicts"
            )
        installed: dict[str, str] = {}
        fn = getattr(ops, "list_loras_fn", None)
        if fn is not None:
            try:
                for entry in (fn() or []):
                    fname = entry.get("filename")
                    fpath = entry.get("path")
                    if fname and fpath:
                        installed[fname] = fpath
            except Exception as e:                          # noqa: BLE001
                raise _ToolValidationError(
                    f"could not enumerate installed LoRAs: {e}"
                ) from e
        if not installed:
            raise _ToolValidationError(
                "loras requested but no LoRAs are installed in this panel. "
                "Call list_loras() to confirm, or ask the user to install one "
                "via Settings → LoRAs → Browse."
            )
        resolved: list[dict] = []
        for item in loras_in:
            if not isinstance(item, dict):
                raise _ToolValidationError(
                    "each lora entry must be a dict {name, strength}"
                )
            name = (item.get("name") or "").strip()
            if not name:
                raise _ToolValidationError(
                    "each lora entry needs a 'name' (the filename returned by list_loras)"
                )
            if name not in installed:
                avail = sorted(installed)[:5]
                more = ", ..." if len(installed) > 5 else ""
                raise _ToolValidationError(
                    f"LoRA {name!r} not installed. Available: {avail}{more}"
                )
            try:
                strength = float(item.get("strength", 1.0))
            except (TypeError, ValueError):
                raise _ToolValidationError(
                    f"strength must be a number, got {item.get('strength')!r}"
                )
            strength = max(-2.0, min(2.0, strength))
            resolved.append({"path": installed[name], "strength": strength})
        if resolved:
            form["loras"] = json.dumps(resolved)

    # Sharp upscale path. Default: pipersr ON for balanced, off otherwise.
    method = (args.get("upscale_method") or "").lower()
    if not method:
        method = "pipersr" if quality == "balanced" else "lanczos"
    form["upscale_method"] = method
    form["upscale"] = (
        "fit_720p" if quality == "balanced"
        else (args.get("upscale") or "off")
    )

    # Tag for session filtering. Stored on the job's params so the runtime
    # can later pick out "all jobs from this agent session" without grepping
    # labels.
    session_tag = (args.get("session_tag") or session.get("session_id") or "")
    form["session_tag"] = session_tag

    # Submit through the panel callback. It builds the job + appends to queue.
    job = ops.submit_job(form)
    snap = ops.queue_snapshot()
    queue_pos = next(
        (i + 1 for i, j in enumerate(snap.get("queue") or []) if j["id"] == job["id"]),
        None,
    )

    est = _estimate_wall_seconds(
        duration=duration, quality=quality, accel=accel, mode=mode,
        sharp=(method == "pipersr"),
    )

    # Track the submitted shot in session state for later manifest writing.
    session.setdefault("submitted_shots", []).append({
        "job_id": job["id"],
        "label": form.get("preset_label") or "",
        "mode": mode,
        "quality": quality,
        "duration_seconds": duration,
        "frames": target_frames,
        "estimate_wall_seconds": est,
        "submitted_at": time.time(),
    })

    return {
        "job_id": job["id"],
        "queued_position": queue_pos,
        "queue_depth": len(snap.get("queue") or []),
        "estimated_wall_seconds": est,
        "estimated_wall_human": _fmt_dur(est),
        "frames": target_frames,
        "width": width,
        "height": height,
    }


@tool("submit_shots")
def _submit_shots(args: dict, ops: PanelOps, session: dict) -> dict:
    """Submit MULTIPLE shots in ONE tool call. This is the right tool
    when queuing a batch (a multi-shot short film, a 5-clip storyboard,
    overnight render of N scenes).

    **Why this exists.** When the panel's local chat engine
    (`phosphene_local`) is configured with `auto_pause_during_renders`
    (the default), the LTX worker stops mlx-lm.server the moment it
    picks up the FIRST job from the render queue — to free RAM for the
    diffusion run. Any subsequent chat call from the agent fails with
    `Connection error` until the queue drains. Calling individual
    `submit_shot` tools in sequence therefore gets you exactly ONE
    shot queued, then a crash. `submit_shots` batches the entire
    plan into a single tool dispatch + (by default) signals
    end-of-turn so the runtime exits before another chat call is
    needed. This is the path Salo's "queue them all so renders run
    overnight" workflow expects.

    Args:
      shots: list[dict] (required) — each item is the same arg shape
        as `submit_shot`. Required per shot:
          { "prompt": str, "mode": "i2v"|"t2v"|..., "label": str,
            "ref_image_path": str (i2v only),
            "duration_seconds": float, "quality": str, ... }
        Each item is dispatched through the same validation as a
        single `submit_shot`, so an invalid one fails its row but
        does NOT abort the rest of the batch.
      auto_finish: bool = true — when true (default), the runtime
        treats this dispatch as terminal and exits the agent loop
        after returning. Set to false ONLY when you have follow-up
        non-chat work that doesn't need the engine (rare).

    Returns:
      { "submitted": int,
        "failed": int,
        "results": [ {ok: bool, job_id?: str, error?: str, label?: str}, ... ],
        "queue_depth": int,
        "auto_finished": bool }

    The agent's chat reply that calls submit_shots is the LAST one
    until the user types again. Compose your written summary BEFORE
    the action block — that's what the user will read when they wake.
    """
    shots = args.get("shots")
    if not isinstance(shots, list) or not shots:
        raise _ToolValidationError(
            "shots must be a non-empty list of shot specs (each the "
            "same shape as submit_shot args). Got: "
            f"{type(shots).__name__}"
        )
    if len(shots) > 50:
        # Practical cap: 50 × 6 s standard turbo ≈ 4 hours wall.
        # Anything larger is almost certainly an agent loop bug.
        raise _ToolValidationError(
            f"shots list too long ({len(shots)}); cap is 50 per call. "
            "Split the batch into multiple turns if the user really "
            "wants more."
        )

    auto_finish = args.get("auto_finish")
    if auto_finish is None:
        auto_finish = True
    auto_finish = bool(auto_finish)

    results: list[dict] = []
    submitted = 0
    failed = 0
    for idx, shot in enumerate(shots):
        if not isinstance(shot, dict):
            results.append({
                "ok": False,
                "index": idx,
                "label": None,
                "error": "validation: shot must be a dict",
            })
            failed += 1
            continue
        label = (shot.get("label") or "")[:80]
        try:
            r = _submit_shot(shot, ops, session)
            results.append({
                "ok": True,
                "index": idx,
                "label": label or None,
                "job_id": r.get("job_id"),
                "estimated_wall_seconds": r.get("estimated_wall_seconds"),
                "frames": r.get("frames"),
                "width": r.get("width"),
                "height": r.get("height"),
            })
            submitted += 1
        except _ToolValidationError as e:
            results.append({
                "ok": False, "index": idx, "label": label or None,
                "error": f"validation: {e}",
            })
            failed += 1
        except Exception as e:                          # noqa: BLE001
            results.append({
                "ok": False, "index": idx, "label": label or None,
                "error": f"{type(e).__name__}: {e}",
            })
            failed += 1

    snap = ops.queue_snapshot() or {}
    queue_depth = len(snap.get("queue") or [])

    if auto_finish and submitted > 0:
        # Signal the runtime to exit the loop after this dispatch
        # returns. Required because the LTX worker pauses the local
        # chat engine on first render — any follow-up chat call would
        # fail with "Connection error". See agent/runtime.py for the
        # corresponding consumer of this flag.
        session["_finish_after_turn"] = True

    return {
        "submitted": submitted,
        "failed": failed,
        "results": results,
        "queue_depth": queue_depth,
        "auto_finished": (auto_finish and submitted > 0),
    }


@tool("get_queue_status")
def _get_queue_status(args: dict, ops: PanelOps, session: dict) -> dict:
    """Read the FIFO queue + currently running job + recent history.

    Use this between submissions to confirm the queue is moving, or at
    end-of-plan to summarize ETA.

    Returns: {
      running: bool, current: dict|None, queue: [...], queue_depth: int,
      total_estimated_wall_seconds: int, history_recent: [...]
    }
    """
    snap = ops.queue_snapshot()
    queue = snap.get("queue") or []
    history = (snap.get("history") or [])[:10]
    total = 0
    for j in queue:
        p = j.get("params", {})
        total += _estimate_wall_seconds(
            duration=max(1, int(p.get("frames", 121)) - 1) / 24.0,
            quality=p.get("quality", "balanced"),
            accel=p.get("accel", "turbo"),
            mode=p.get("mode", "t2v"),
            sharp=(p.get("upscale_method") == "pipersr"),
        )
    return {
        "running": snap.get("running", False),
        "current": _trim_job(snap.get("current")),
        "queue_depth": len(queue),
        "queue": [_trim_job(j) for j in queue],
        "total_estimated_wall_seconds": total,
        "total_estimated_wall_human": _fmt_dur(total),
        "history_recent": [_trim_job(j) for j in history],
    }


@tool("wait_for_shot")
def _wait_for_shot(args: dict, ops: PanelOps, session: dict) -> dict:
    """Block until the named job finishes (status in {"done","error","cancelled"}).

    Args:
      job_id: str (required)
      timeout_seconds: int = 1800   — hard cap. Returns timeout=true if hit.
      poll_seconds: float = 5

    Returns: {status, output_path, elapsed_sec, error, timed_out, job}
    """
    job_id = _required(args, "job_id")
    timeout = float(args.get("timeout_seconds", 1800))
    poll = max(1.0, float(args.get("poll_seconds", 5)))
    start = time.time()
    while True:
        j = ops.find_job(job_id)
        if j is None:
            return {"status": "missing", "error": "job not found",
                    "output_path": None, "elapsed_sec": None,
                    "timed_out": False, "job": None}
        status = j.get("status")
        if status in ("done", "error", "cancelled"):
            return {
                "status": status,
                "output_path": j.get("output_path"),
                "elapsed_sec": j.get("elapsed_sec"),
                "error": j.get("error"),
                "timed_out": False,
                "job": _trim_job(j),
            }
        if time.time() - start > timeout:
            return {
                "status": status,
                "output_path": j.get("output_path"),
                "elapsed_sec": time.time() - start,
                "error": None,
                "timed_out": True,
                "job": _trim_job(j),
            }
        time.sleep(poll)


@tool("extract_frame")
def _extract_frame(args: dict, ops: PanelOps, session: dict) -> dict:
    """Extract a single frame from a finished clip as a PNG.

    Use the `last` frame of shot N as the keyframe-0 of shot N+1 to anchor
    character continuity across cuts (see SDK_KEYFRAME_INTERPOLATION.md
    Problem 4).

    Args:
      job_id OR clip_path: identify the source. job_id resolves via the panel.
      which: "first" | "last" | "middle" | a specific frame index (int)
      output_name: optional override for the saved filename (under panel_uploads).

    Returns: {png_path, frame_index, source_clip}
    """
    clip = args.get("clip_path") or ""
    job_id = args.get("job_id")
    if job_id and not clip:
        j = ops.find_job(job_id)
        if not j:
            raise _ToolValidationError(f"job {job_id} not found")
        clip = j.get("output_path") or ""
    if not clip:
        raise _ToolValidationError("provide either job_id (finished) or clip_path")
    clip_p = _ensure_under(
        Path(clip), [ops.outputs_dir, ops.uploads_dir]
    )
    if not clip_p.is_file():
        raise _ToolValidationError(f"clip not found: {clip}")
    clip = str(clip_p)

    which = args.get("which", "last")
    nb = _probe_frame_count(clip)
    if isinstance(which, int) or (isinstance(which, str) and which.isdigit()):
        idx = max(0, min(int(which), nb - 1))
    elif which == "first":
        idx = 0
    elif which == "middle":
        idx = max(0, nb // 2)
    else:                                   # "last" or anything else
        idx = max(0, nb - 1)

    out_name = (args.get("output_name") or
                f"frame_{Path(clip).stem}_{idx:04d}.png")
    # Strip any directory traversal — output_name is a filename, not a path.
    # Without this, an LLM nudged by a poisoned doc could write outside uploads_dir.
    out_name = Path(out_name).name or f"frame_{idx:04d}.png"
    out_path = (ops.uploads_dir / out_name).resolve()
    if not out_path.is_relative_to(ops.uploads_dir.resolve()):
        raise _ToolValidationError(f"output path escapes uploads dir: {out_name}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ffmpeg seek-by-index. -frames:v 1 + select+filter is precise but slow
    # on long clips; we use trim by frame number which is exact for H.264.
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", clip,
        "-vf", f"select=eq(n\\,{idx})",
        "-vframes", "1",
        str(out_path),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0 or not out_path.is_file():
        raise RuntimeError(
            f"ffmpeg failed extracting frame {idx} from {clip}: "
            f"{completed.stderr[:400]}"
        )

    return {
        "png_path": str(out_path),
        "frame_index": idx,
        "source_clip": clip,
        "total_frames": nb,
    }


@tool("write_session_manifest")
def _write_session_manifest(args: dict, ops: PanelOps, session: dict) -> dict:
    """Write a JSON manifest listing the agent's submitted shots in cut order.

    The manifest is the deliverable for an overnight render: in the morning,
    the user opens `mlx_outputs/agentflow_<id>/manifest.json` and sees what
    the agent built. ffmpeg-concat is the user's choice in their editor —
    we deliberately do NOT auto-stitch.

    Args:
      output_name: str = "manifest.json"
      title: str — human-friendly project title
      shot_order: list[str] — explicit order of job_ids; defaults to
                              submission order from session state.

    Returns: {manifest_path, shot_count, missing_outputs}
    """
    title = args.get("title") or session.get("title") or "Untitled session"
    submitted = session.get("submitted_shots") or []
    order = args.get("shot_order") or [s["job_id"] for s in submitted]

    # Resolve current state of each shot
    rows = []
    missing = []
    for jid in order:
        j = ops.find_job(jid)
        if not j:
            missing.append(jid)
            continue
        p = j.get("params", {})
        rows.append({
            "job_id": jid,
            "label": p.get("label"),
            "mode": p.get("mode"),
            "quality": p.get("quality"),
            "frames": p.get("frames"),
            "status": j.get("status"),
            "output_path": j.get("output_path"),
            "elapsed_sec": j.get("elapsed_sec"),
            "prompt": p.get("prompt"),
        })

    session_id = session.get("session_id") or "session"
    out_dir = (ops.outputs_dir / f"agentflow_{session_id}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize output_name strictly:
    #   - reduce to its basename so `..`, absolute paths, and any nested
    #     dir components are stripped (a prompt-injected agent can no
    #     longer write outside out_dir even if it escapes the basename
    #     check via Unicode tricks — the resolve() check below is the
    #     real backstop)
    #   - require a `.json` suffix so the manifest can't masquerade as
    #     a script / config / dotfile
    #   - cap length to keep weird filesystems happy
    raw_name = (args.get("output_name") or "manifest.json").strip()
    safe_name = Path(raw_name).name
    if not safe_name or safe_name in (".", "..") or safe_name.startswith("."):
        safe_name = "manifest.json"
    if not safe_name.lower().endswith(".json"):
        safe_name = safe_name + ".json"
    safe_name = safe_name[:120]

    out_path = (out_dir / safe_name).resolve()
    # Belt-and-braces: reject if the resolved final path escapes out_dir
    # (catches the rare case where safe_name still contains a separator
    # the OS treated as path traversal — shouldn't happen post-basename
    # but the check is free).
    if not out_path.is_relative_to(out_dir):
        raise _ToolValidationError(
            f"output_name resolved outside session folder: {raw_name!r}"
        )

    manifest = {
        "schema": "phosphene/agentflow/manifest@1",
        "title": title,
        "session_id": session_id,
        "created_at_unix": time.time(),
        "shots": rows,
        "missing": missing,
        "stitching_hint": (
            "Cut these in your NLE in the order listed. "
            "Use first-frame-anchored keyframes (extracted via extract_frame) "
            "to make cross-shot cuts invisible."
        ),
    }
    # Atomic write: serialize to a temp file in the same directory, fsync,
    # then os.replace. A crash mid-write leaves either the previous
    # manifest or nothing — never a half-written JSON.
    tmp_path = out_path.with_name(f".{out_path.name}.{os.getpid()}.tmp")
    payload = json.dumps(manifest, indent=2).encode("utf-8")
    try:
        with open(tmp_path, "wb") as fh:
            fh.write(payload)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass  # fsync isn't critical for manifest correctness
        os.replace(tmp_path, out_path)
    except Exception:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise

    return {
        "manifest_path": str(out_path),
        "shot_count": len(rows),
        "missing_outputs": missing,
        "title": title,
    }


@tool("generate_shot_images")
def _generate_shot_images(args: dict, ops: PanelOps, session: dict) -> dict:
    """Generate candidate anchor stills for ONE shot.

    Used in the director-collaboration workflow:
      Phase A: agent calls this for each shot — 4 candidates appear in chat.
      Phase B: user picks the best one (UI thumbnail click).
      Phase C: agent submits the i2v video render with the chosen anchor.

    Args:
      shot_label: str (required) — the same label you'll pass to submit_shot
                                   in phase C, so anchors and renders match.
      prompt: str (required) — visual description for the still. Focus on
                              framing, lighting, character, expression,
                              setting. Do NOT include dialogue (audio comes
                              with the video render, not the still). Keep
                              it 60-120 words.
      n: int = 4 — number of candidates (1-8). 4 is the sweet spot.
      aspect: "16:9" | "9:16" | "1:1" | "4:3" | "21:9" | "3:4" = "16:9"
      seed_base: int = -1 — when >= 0, candidates use seed_base + i so the
                            run is reproducible. Use -1 for random seeds.
      append: bool = false — when true AND the same shot_label has
                            existing candidates, ADD a new "take" instead
                            of overwriting. New PNGs land under
                            `take_NN/cand_*.png`. Use this when the user
                            says "give me 4 more variations of S3" — the
                            previous candidates stay clickable and the new
                            take stacks below.
      refs: list[str] = [] — 1-3 absolute paths to reference images. Only
                            honored when the engine is Qwen-Image-Edit-2509
                            (`engine_override="qwen_edit"` or
                            `engine_override="qwen_edit_lightning"`).
                            With refs, the model COMPOSES the prompt onto
                            the references — character + place / character +
                            character / character + product. The primary
                            path for cross-shot character continuity: pass
                            the same character ref on every shot and the
                            identity is locked at the still stage.
      engine_override: str = "" — per-call engine choice, OVERRIDES the
                            user's saved Settings without changing it.
                            Valid values:
                              "flux2_edit"           4-step Q4 distilled FLUX.2
                                                     Klein-Edit, ~30 s/image,
                                                     NEEDS refs. Distilled =
                                                     fast but ILLUSTRATIVE
                                                     style (cartoonish).
                              "flux2_edit_high"      25-step Q8 NON-DISTILLED
                                                     FLUX.2 Klein-Base-Edit
                                                     with guidance 4.0,
                                                     ~3-5 min/image, NEEDS
                                                     refs. Use this when the
                                                     user asks for
                                                     PHOTOREALISTIC stills —
                                                     this is the default
                                                     photoreal path.
                              "qwen_edit_lightning"  4-step Q4 + Lightning LoRA,
                                                     ~10-15 s/image, NEEDS refs
                              "qwen_edit"            8-step Q4, ~1 min/image,
                                                     NEEDS refs
                              "qwen_edit_high"       30-step Q8 + guidance 4.0,
                                                     ~5 min/image, NEEDS refs.
                                                     Alternative photoreal
                                                     path; multi-ref-strong.
                              "kontext_high"         30-step Q8 FLUX.1 Kontext
                                                     guidance 4.5, ~3 min/image,
                                                     NEEDS refs. Mature
                                                     image-conditioning
                                                     family — different
                                                     aesthetic than flux2/qwen.
                              "flux2"                4-step T2I, ~15 s/image,
                                                     no refs supported
                              "z_image_turbo"        9-step T2I, ~30 s/image,
                                                     no refs supported
                              "mock"                 flat colored squares,
                                                     instant (testing only)
                            **Default for photoreal stills with refs:
                            "flux2_edit_high"** (or qwen_edit_high if
                            multi-character composition is needed).
                            Use the bare "flux2_edit" only when the user
                            explicitly asks for "fast iteration" or
                            "draft quality" — its distilled weights
                            produce illustrative output, not photographs.
                            Use this to react to errors: if a qwen_edit
                            call returns "needs at least 1 reference image"
                            and the user explicitly didn't provide one,
                            retry with engine_override="z_image_turbo"
                            instead of trying the same family again or
                            calling finish() with nothing rendered.

    Returns: {shot_label, prompt, candidates: [{png_path, seed, ...}, ...],
              takes: [{candidates, generated_at, prompt}, ...],
              elapsed_seconds, engine, take_index}.

    The UI renders the candidates as a clickable thumbnail grid in the
    tool-result card. When the user clicks one, that selection is recorded
    in `session.tool_state["selected_anchors"][shot_label]`. The agent
    can read the selections later via `get_selected_anchors`. With multiple
    takes, the user can pick from ANY take.
    """
    from agent import image_engine as _image_engine

    label = (args.get("shot_label") or "").strip() or "untitled"
    prompt = (args.get("prompt") or "").strip()
    if not prompt:
        raise _ToolValidationError("prompt is required")
    n = max(1, min(8, int(args.get("n", 4))))
    aspect = args.get("aspect", "16:9")
    seed_base = int(args.get("seed_base", -1))
    append = bool(args.get("append", False))

    # Reference images for multi-ref engines (Qwen-Image-Edit-2509). Each
    # ref is resolved relative to ops.uploads_dir if not absolute, then
    # path-traversal-validated to live under uploads_dir or outputs_dir.
    # _resolve_path alone joins-but-doesn't-validate — without
    # _ensure_under a prompt-injected `../../etc/...` could reach the
    # engine, which would then fail confusingly at PIL-decode time;
    # rejecting up-front gives a clean validation message and matches
    # the security stance extract_frame / inspect_clip already use.
    # Engines that don't support refs ignore them; the result dict
    # carries refs_ignored=True in that case.
    refs_in = args.get("refs") or []
    if not isinstance(refs_in, list):
        raise _ToolValidationError("refs must be a list of image paths")
    if len(refs_in) > 3:
        raise _ToolValidationError(
            f"refs supports at most 3 images (Qwen-Image-Edit-2509 limit), got {len(refs_in)}"
        )
    refs_resolved: list[str] = []
    for r in refs_in:
        if not isinstance(r, str) or not r.strip():
            raise _ToolValidationError("each ref must be a non-empty string path")
        rp_str = _resolve_path(r, ops.uploads_dir)
        try:
            rp = _ensure_under(Path(rp_str), [ops.uploads_dir, ops.outputs_dir])
        except _ToolValidationError as e:
            raise _ToolValidationError(
                f"ref path must live under uploads/outputs: {r!r} ({e})"
            ) from e
        if not rp.is_file():
            raise _ToolValidationError(f"ref image not found: {rp}")
        refs_resolved.append(str(rp))

    # Image engine config travels in PanelOps.capabilities so tools.py
    # stays free of panel imports.
    cfg_dict = (ops.capabilities or {}).get("image_engine_config") or {}
    cfg = _image_engine.ImageEngineConfig(**cfg_dict)

    # Agent-side engine override. Same preset names the panel's Image
    # Studio uses. Lets the agent switch engines per-call without
    # touching the user's persisted config — critical for cases like
    # "user said no refs, qwen_edit can't run, fall back to T2I" which
    # otherwise the agent retries-itself-into-finish() because it has
    # no path to satisfy the request. Unknown override raises a clear
    # validation error instead of silently using the saved config.
    engine_override = (args.get("engine_override") or "").strip()
    if engine_override:
        if engine_override == "flux2_edit":
            # Distilled klein-4b — fast but illustrative at Q4. mflux's
            # flux2_edit_generate.py FORCES guidance == 1.0 on distilled
            # bases; passing anything else would parser.error. Pass the
            # mflux SHORTHAND name (not the HF repo) so mflux resolves
            # the correct base config.
            #
            # Q6 instead of Q4: Apple-Silicon community consensus for
            # FLUX.2 — Q6 is ~4-6% quality loss vs full precision, Q4
            # is 8-12%. On a 64 GB M4 Max the per-image speed gap is
            # negligible (~1-2 s) but the visible quality jump is real
            # (skin texture, gear detail, photographic feel). Draw
            # Things ships Q5/Q6 as default for the same reason.
            cfg = _image_engine.ImageEngineConfig(
                kind="mflux",
                mflux_model="flux2-klein-4b",
                mflux_family="flux2_edit",
                mflux_quantize=6,
                mflux_steps=4,
                mflux_guidance=1.0,
            )
        elif engine_override == "flux2_edit_high":
            # Non-distilled klein-base-4b — accepts real CFG and renders
            # photorealistic. ~3-5 min/image at Q8. The default
            # photoreal path when the user has a character ref.
            #
            # CRITICAL: pass the SHORTHAND mflux model name
            # ("flux2-klein-base-4b") instead of an HF repo path.
            # mflux's flux2_edit_generate.py uses ModelConfig.from_name
            # to detect "is this a distilled base?" via substring match.
            # If you pass "black-forest-labs/FLUX.2-klein-4B" as model
            # the substring check sees "klein-4b" (no "base") and
            # rejects guidance != 1.0 — the photoreal path silently
            # falls back to the distilled config and you get the
            # illustrative output anyway.
            cfg = _image_engine.ImageEngineConfig(
                kind="mflux",
                mflux_model="flux2-klein-base-4b",
                mflux_family="flux2_edit",
                mflux_quantize=8,
                mflux_steps=25,
                mflux_guidance=4.0,
            )
        elif engine_override == "qwen_edit_lightning":
            # Qwen-Image-Edit-2511 + Lightning LoRA 4-step. 2511 is the
            # current best Qwen-Edit (Apache 2.0) — significantly stronger
            # character consistency than 2509, mitigated image drift on
            # multi-ref edits, multi-person fusion, built-in popular LoRAs.
            # The Lightning LoRA filename pin avoids mflux downloading all
            # 4 variants (bf16/fp32 × 4-step/8-step) and getting stuck on
            # ambiguous resolution.
            cfg = _image_engine.ImageEngineConfig(
                kind="mflux",
                mflux_model="Qwen/Qwen-Image-Edit-2511",
                mflux_family="qwen_edit",
                mflux_quantize=6,
                mflux_steps=4,
                mflux_lora_paths=[
                    "lightx2v/Qwen-Image-Edit-2511-Lightning:Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
                ],
                mflux_lora_scales=[1.0],
            )
        elif engine_override == "qwen_edit":
            # Standard Qwen-Edit-2511 at Q6 8-step. 2511 base, no
            # Lightning. ~1 min/image on M4 Max. Iteration tier with
            # 2511's improved consistency over 2509.
            cfg = _image_engine.ImageEngineConfig(
                kind="mflux",
                mflux_model="Qwen/Qwen-Image-Edit-2511",
                mflux_family="qwen_edit",
                mflux_quantize=6,
                mflux_steps=8,
            )
        elif engine_override == "qwen_edit_high":
            # Final-render preset: Qwen-Image-Edit-2511 Q8 + 40 steps.
            # 40 steps matches the Diffusers reference example for
            # Qwen-Edit (qwen.ai blog: 2511 official inference uses
            # num_inference_steps=40 + true_cfg_scale=4.0 +
            # guidance_scale=1.0 + negative_prompt=" "). mflux's --guidance
            # maps to true_cfg_scale on the qwen-edit CLI.
            cfg = _image_engine.ImageEngineConfig(
                kind="mflux",
                mflux_model="Qwen/Qwen-Image-Edit-2511",
                mflux_family="qwen_edit",
                mflux_quantize=8,
                mflux_steps=40,
                mflux_guidance=4.0,
            )
        elif engine_override == "kontext_high":
            # FLUX.1 Kontext-dev — image-conditioning, mature family.
            # Q8 + 30 steps + guidance 4.5 produces a different
            # aesthetic than flux2/qwen — useful when those two
            # produce inconsistent output for a specific composition.
            cfg = _image_engine.ImageEngineConfig(
                kind="mflux",
                mflux_model="black-forest-labs/FLUX.1-Kontext-dev",
                mflux_family="kontext",
                mflux_quantize=8,
                mflux_steps=30,
                mflux_guidance=4.5,
            )
        elif engine_override == "flux2":
            cfg = _image_engine.ImageEngineConfig(
                kind="mflux",
                mflux_model="Runpod/FLUX.2-klein-4B-mflux-4bit",
                mflux_family="flux2",
                mflux_quantize=4,
            )
        elif engine_override == "z_image_turbo":
            cfg = _image_engine.ImageEngineConfig(
                kind="mflux",
                mflux_model="filipstrand/Z-Image-Turbo-mflux-4bit",
                mflux_family="z_image_turbo",
                mflux_quantize=4,
            )
        elif engine_override == "mock":
            cfg = _image_engine.ImageEngineConfig(kind="mock")
        else:
            raise _ToolValidationError(
                f"engine_override={engine_override!r} unknown. Valid: "
                f"flux2_edit | flux2_edit_high | "
                f"qwen_edit | qwen_edit_lightning | qwen_edit_high | "
                f"kontext_high | flux2 | z_image_turbo | mock"
            )

    safe_label = re.sub(r"[^a-z0-9_-]", "_", label.lower())[:40] or "untitled"
    sid = session.get("session_id") or "session"
    base_dir = ops.uploads_dir / "agentflow" / sid / safe_label

    # Resolve existing takes so we can compute the next subdir.
    session.setdefault("anchor_candidates", {})
    existing = session["anchor_candidates"].get(label) or {}
    prior_takes = list(existing.get("takes") or [])
    # Migration path: an older session may have a flat "candidates" list
    # under the label without a "takes" array. Preserve as take_00.
    if not prior_takes and existing.get("candidates"):
        prior_takes = [{
            "candidates": existing["candidates"],
            "prompt": existing.get("prompt", prompt),
            "generated_at": existing.get("generated_at", time.time()),
        }]

    if append and prior_takes:
        take_index = len(prior_takes)
        out_dir = base_dir / f"take_{take_index:02d}"
    else:
        # Fresh run (or no append flag) — overwrite. Keep the old behavior
        # for the no-append branch so the agent's existing prompts work.
        prior_takes = []
        take_index = 0
        out_dir = base_dir

    t0 = time.time()
    candidates = _image_engine.generate(
        prompt=prompt, n=n, aspect=aspect,
        output_dir=out_dir,
        base_seed=(seed_base if seed_base >= 0 else None),
        refs=refs_resolved or None,
        config=cfg,
    )
    elapsed = round(time.time() - t0, 2)

    # Write a sidecar JSON next to each candidate so the library reader
    # (list_library_images) and the panel can recover full metadata
    # without re-deriving from path. Idempotent — overwrites if the
    # candidate is regenerated. Failure to write a sidecar isn't fatal:
    # the agent still gets the candidates list back, list_library_images
    # falls back to path inference for any sidecar-less PNG.
    generated_at = time.time()
    for c in candidates:
        png = c.get("png_path")
        if not png:
            continue
        sidecar_path = Path(png).with_suffix(Path(png).suffix + ".json")
        sidecar = {
            "schema": "phosphene/library/image@1",
            "png_path": png,
            "prompt": prompt,
            "refs": list(refs_resolved),
            "engine": c.get("engine"),
            "family": c.get("family"),
            "model": c.get("model"),
            "seed": c.get("seed"),
            "width": c.get("width"),
            "height": c.get("height"),
            "aspect": aspect,
            "session_id": sid,
            "shot_label": label,
            "take_index": take_index,
            "generated_at": generated_at,
            "refs_ignored": c.get("refs_ignored", False),
            "source": "agent.generate_shot_images",
        }
        try:
            sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
        except OSError:
            # Sidecar is best-effort; library reader has a fallback.
            pass

    new_take = {
        "candidates": candidates,
        "prompt": prompt,
        "generated_at": time.time(),
        "take_index": take_index,
    }
    all_takes = prior_takes + [new_take]
    session["anchor_candidates"][label] = {
        "prompt": prompt,
        "candidates": candidates,        # latest take, surfaced to legacy UI
        "takes": all_takes,
        "generated_at": time.time(),
    }

    return {
        "shot_label": label,
        "prompt": prompt,
        "candidates": candidates,
        "takes": all_takes,
        "take_index": take_index,
        "engine": cfg.kind,
        "elapsed_seconds": elapsed,
    }


@tool("inspect_clip")
def _inspect_clip(args: dict, ops: PanelOps, session: dict) -> dict:
    """Look up an existing rendered clip's full parameters from its sidecar.

    Used when the user references an earlier clip and asks for a
    variation — "remake S5 with more pause", "redo the doctor reveal at
    higher quality", "give me another take of this". Reads the .mp4.json
    sidecar that the panel writes next to every render and returns the
    original prompt, mode, quality, dimensions, frame count, seed, and
    any LoRAs / loras_json. Use the returned values as the BASELINE for
    your variation, then submit_shot with the user's modifications layered
    on top — same shot label suffixed (e.g. "S5 Wife Legacy v2") so the
    new render reads as a take, not a duplicate.

    Args (one of):
      job_id: str — agent's submitted shot id (e.g. "j-19dfe67c6e6-001"),
              OR a clip path like "/path/to/...mp4". The implementation
              accepts either.
      clip_path: str — alternative to job_id. Absolute mp4 path; the
              sidecar at <path>.json is read.

    Returns: {
      job_id, output_path, sidecar_path, status, elapsed_sec,
      prompt, negative_prompt, mode, quality, accel,
      width, height, frames, duration_seconds, seed_used,
      label, loras, hdr, upscale_method, upscale,
    }

    On miss: {"error": "no clip found for ..."}.
    """
    import json as _json

    job_id = (args.get("job_id") or "").strip()
    clip_path = (args.get("clip_path") or "").strip()
    if not job_id and not clip_path:
        raise _ToolValidationError("provide job_id or clip_path")

    # Resolve to a clip mp4 path.
    if job_id and not clip_path:
        j = ops.find_job(job_id)
        if not j:
            return {"error": f"no clip found for job_id {job_id}"}
        clip_path = (j.get("output_path") or "").strip()
        if not clip_path:
            return {"error": f"job {job_id} has no output_path yet (status={j.get('status')})"}

    p = Path(clip_path)
    if not p.is_absolute():
        p = (ops.outputs_dir / p).resolve()
    try:
        p = _ensure_under(p, [ops.outputs_dir, ops.uploads_dir])
    except _ToolValidationError as e:
        return {"error": str(e)}
    if not p.is_file():
        return {"error": f"clip not found on disk: {p}"}

    # Sidecar lives at <path>.json next to the mp4.
    sidecar = p.with_suffix(p.suffix + ".json")
    if not sidecar.is_file():
        return {"error": f"no sidecar at {sidecar} — clip was rendered before sidecars existed?"}
    try:
        data = _json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, _json.JSONDecodeError) as e:
        return {"error": f"sidecar unreadable: {e}"}

    params = data.get("params") or {}
    frames = int(params.get("frames", 0)) or None
    duration = round((frames - 1) / 24.0, 2) if frames and frames > 1 else None

    return {
        "job_id": data.get("queue_id") or job_id or "",
        "output_path": str(p),
        "sidecar_path": str(sidecar),
        "status": "done",
        "elapsed_sec": data.get("elapsed_sec"),
        "prompt": params.get("prompt"),
        "negative_prompt": params.get("negative_prompt") or "",
        "mode": params.get("mode") or "t2v",
        "quality": params.get("quality") or "balanced",
        "accel": params.get("accel") or "off",
        "width": params.get("width"),
        "height": params.get("height"),
        "frames": frames,
        "duration_seconds": duration,
        "seed_used": params.get("seed_used") or params.get("seed"),
        "label": params.get("label") or params.get("preset_label") or "",
        "loras": params.get("loras") or [],
        "hdr": bool(params.get("hdr")),
        "upscale_method": params.get("upscale_method") or "",
        "upscale": params.get("upscale") or "",
    }


@tool("get_selected_anchors")
def _get_selected_anchors(args: dict, ops: PanelOps, session: dict) -> dict:
    """Return the user's anchor selections so far.

    Between phase B (generate) and phase C (render), the user clicks
    thumbnails in the UI; those clicks are recorded in
    `session.tool_state["selected_anchors"]`. Call this at the START of
    phase C (when the user types "render") to discover which still goes
    with which shot, then submit each video with `mode: "i2v"` and the
    matching `ref_image_path`.

    Returns: {selected_anchors, missing, total_candidates, total_selected}.
    `missing` is a list of shot_labels with candidates but no user pick.
    """
    selected = session.get("selected_anchors", {})
    candidates = session.get("anchor_candidates", {})
    missing = [lbl for lbl in candidates.keys() if lbl not in selected]
    return {
        "selected_anchors": selected,
        "missing": missing,
        "total_candidates": len(candidates),
        "total_selected": len(selected),
    }


@tool("upload_image")
def _upload_image(args: dict, ops: PanelOps, session: dict) -> dict:
    """Resolve an image path the user attached to the chat.

    The chat UI persists user-attached files to `panel_uploads/` and
    surfaces the absolute path in the user message. This tool exists so
    the agent can refer to the file by its `attachment_id` (the path the
    chat injected) and confirm it's there before passing it as
    ref_image_path or in keyframes.

    Args:
      attachment_id: str — absolute path of an uploaded image
    """
    p = _required(args, "attachment_id")
    p = _resolve_path(p, ops.uploads_dir)
    pp = Path(p)
    if not pp.is_file():
        raise _ToolValidationError(f"attachment not found: {p}")
    return {
        "absolute_path": p,
        "size_bytes": pp.stat().st_size,
        "name": pp.name,
    }


@tool("read_document")
def _read_document(args: dict, ops: PanelOps, session: dict) -> dict:
    """Read a user-attached text or PDF document and return its contents.

    Use this when the user attaches a script, treatment, character bible,
    or any text/PDF they want you to act on. The chat UI puts the absolute
    path inside the user message's `<attachments>` block — pass that path
    here. Always read the doc BEFORE drafting a plan if one was attached.

    Args:
      path: str — absolute path of the attached file (from <attachments>).
      max_chars: int (optional, default 80000) — clamp the returned text
        so a long PDF doesn't blow the model's context. The tail is
        replaced with "[truncated, N more chars]" when this fires.

    Returns:
      { path, name, mime, kind, char_count, truncated, content,
        page_count? }

    Raises:
      _ToolValidationError if the path is outside uploads_dir, missing,
      or unreadable. PDFs require pypdf — if it isn't installed the tool
      returns {"error": "pypdf not installed; ..."} rather than raising.
    """
    raw = _required(args, "path")
    p = _ensure_under(Path(raw), [ops.uploads_dir])
    if not p.is_file():
        raise _ToolValidationError(f"document not found: {p}")

    max_chars = int(args.get("max_chars") or 80_000)
    name = p.name
    suffix = p.suffix.lower()

    text = ""
    page_count: int | None = None

    if suffix in (".txt", ".md", ".markdown", ".rst", ".json", ".csv"):
        try:
            raw_b = p.read_bytes()
        except OSError as e:
            raise _ToolValidationError(f"cannot read {name}: {e}") from e
        # Decode tolerantly — script writers sometimes paste text with
        # mixed encodings. Replacement is better than refusing the whole doc.
        text = raw_b.decode("utf-8", errors="replace")
        kind = "text"

    elif suffix == ".pdf":
        try:
            import pypdf                            # noqa: F401 — local import on purpose
        except ImportError:
            return {
                "path": str(p), "name": name, "mime": "application/pdf",
                "kind": "pdf", "error": (
                    "pypdf is not installed in Phosphene's venv. "
                    "Install with: ltx-2-mlx/env/bin/pip install pypdf — "
                    "then re-attach. Alternatively, paste the script text "
                    "directly into the chat."
                ),
            }
        try:
            reader = pypdf.PdfReader(str(p))
            pages: list[str] = []
            for i, page in enumerate(reader.pages):
                pages.append(page.extract_text() or "")
            text = "\n\n".join(pages).strip()
            page_count = len(reader.pages)
        except Exception as e:                      # noqa: BLE001
            raise _ToolValidationError(f"PDF parse failed for {name}: {e}") from e
        kind = "pdf"

    else:
        raise _ToolValidationError(
            f"unsupported document type {suffix!r}. Supported: "
            ".txt, .md, .pdf"
        )

    truncated = False
    full_len = len(text)
    if full_len > max_chars:
        head_len = max_chars - 64                   # leave room for the marker
        text = text[:head_len] + f"\n\n[truncated, {full_len - head_len} more chars]"
        truncated = True

    out = {
        "path": str(p),
        "name": name,
        "mime": "application/pdf" if kind == "pdf" else "text/plain",
        "kind": kind,
        "char_count": full_len,
        "truncated": truncated,
        "content": text,
    }
    if page_count is not None:
        out["page_count"] = page_count
    return out


@tool("get_master_style")
def _get_master_style(args: dict, ops: PanelOps, session: dict) -> dict:
    """Look up the master style suffix to lock in for the current project.

    Use this at the start of EVERY turn that will submit shots in a
    session that already has prior shots. It returns the locked-in
    style fragment so you can paste it verbatim onto every new prompt.
    Without this, the gallery looks like a horror movie + a vlog + a
    documentary stitched together — exactly what the user complained
    about ("the style of the shots is really uneven").

    Resolution order:
      1. Project notes — most recent `[style ·` entry from
         `read_project_notes()`. This is the canonical source; it
         survives panel restart.
      2. Most recent `submitted_shot` in the session — extract the
         tail of the prompt as a heuristic fallback if the agent
         forgot to call append_project_notes earlier.
      3. None — return an empty string. The caller should DECLARE
         the master style + immediately call append_project_notes
         before submitting the first shot.

    Returns: { found: bool, source: "notes"|"prior_shot"|"none",
               style: str, hint: str }.
    """
    from agent import project as _project
    notes = _project.read_notes(ops.state_dir) or ""
    # Walk the notes from the END looking for the most recent style entry.
    # Style entries look like "## 2026-05-07 03:14  [style · agent]\n<text>"
    if notes:
        marker = "[style ·"
        idx = notes.rfind(marker)
        if idx != -1:
            tail = notes[idx:]
            # The text follows the header line.
            nl = tail.find("\n")
            body_start = nl + 1 if nl != -1 else 0
            # Body ends at the next "## " (next entry) or end-of-file.
            next_entry = tail.find("\n## ", body_start)
            body_end = next_entry if next_entry != -1 else len(tail)
            style = tail[body_start:body_end].strip()
            if style:
                return {"found": True, "source": "notes", "style": style,
                        "hint": "Use this verbatim as the prompt suffix on every shot."}
    # Fallback: grab the most recent submitted_shot's prompt tail.
    submitted = session.get("submitted_shots") or []
    if submitted:
        last = submitted[-1] or {}
        last_prompt = last.get("prompt") or ""
        # Heuristic: the master style is usually the last sentence(s) of
        # the prompt — "Documentary realism, full-frame 16:9, ..."
        # Take everything after the last period that ends a content beat.
        if last_prompt:
            tail = last_prompt[-300:]
            return {"found": True, "source": "prior_shot",
                    "style": tail,
                    "hint": ("Heuristic — the tail of the most recent shot's "
                             "prompt. Re-use the look words; ignore the dialogue.")}
    return {"found": False, "source": "none", "style": "",
            "hint": ("No master style locked yet. DECLARE it on the next "
                     "shot AND call append_project_notes(kind='style', text=...) "
                     "so future turns can read it back.")}


@tool("list_loras")
def _list_loras(args: dict, ops: PanelOps, session: dict) -> dict:
    """List the LTX 2.3 LoRAs installed in this Phosphene panel.

    Returns the user's installed LoRAs from `mlx_models/loras/` —
    each with name, trigger words, recommended strength, base model,
    and the absolute file path. Use this BEFORE recommending a LoRA
    on a shot so you only suggest ones the user actually has.

    Two kinds of LoRA matter for the agent:

    - **Character LoRAs** — lock identity across every shot. The
      strongest character-lock path (see system prompt § Character
      lock — Path B). Common naming: contains a person/character
      name, has trigger_words for that character, base_model is
      LTX 2.3.
    - **Style LoRAs** — change look without touching identity (noir,
      sketch, painterly, etc.). Stack on top of a character LoRA.

    To USE a LoRA on a shot, pass its filename (e.g.
    "emma_v2.safetensors") in the `loras` arg of `submit_shot`:
        loras: [{"name": "emma_v2.safetensors", "strength": 0.85}]
    The panel matches by filename and fuses the weights at denoise
    time. If the LoRA has trigger_words, surface them in your prompt
    verbatim — the panel doesn't auto-inject.

    GETTING a LoRA the user doesn't have yet — three paths:
    1. **CivitAI browser** (in-panel): Settings → LoRAs → Browse.
       Behind a consent gate (NSFW LoRAs in particular). Many
       community-shared LTX 2.3 character + style LoRAs.
    2. **External training**: WaveSpeed cloud (no infra), RunPod /
       Lambda (CUDA boxes), or anywhere with the Lightricks
       `LTX-2 trainer`. We don't ship Apple-Silicon LTX training.
       After training, the user drops the `.safetensors` file into
       `mlx_models/loras/` directly.
    3. **From a friend / collaborator** with an existing file —
       same drop-into-loras-dir flow.

    The agent should NOT walk users through training — that's
    out-of-app work. Just point them at Path 1 (CivitAI browser) or
    summarize Path 2/3 if they ask.

    Returns: { count, loras: [ {filename, name, description, base_model,
              trigger_words, recommended_strength, path}, ... ] }.
    """
    fn = getattr(ops, "list_loras_fn", None)
    if fn is None:
        return {"count": 0, "loras": [], "note": "No LoRA picker available."}
    try:
        raw = fn() or []
    except Exception as e:                          # noqa: BLE001
        raise _ToolValidationError(f"list_loras failed: {e}") from e
    # Slim to what the agent needs — strip preview URLs etc. that
    # bloat the response.
    out = []
    for l in raw:
        out.append({
            "filename": l.get("filename"),
            "name": l.get("name"),
            "description": (l.get("description") or "")[:240],
            "base_model": l.get("base_model"),
            "trigger_words": l.get("trigger_words") or [],
            "recommended_strength": l.get("recommended_strength") or 1.0,
            "path": l.get("path"),
        })
    return {"count": len(out), "loras": out}


@tool("list_library_images")
def _list_library_images(args: dict, ops: PanelOps, session: dict) -> dict:
    """List images previously generated by this Phosphene panel.

    The library is the union of:
      - Agent-generated stills under `panel_uploads/agentflow/<session>/<label>/[take_NN]/cand_*.png`
      - Manually-generated stills under `panel_uploads/library/manual/*.png`
        (produced from the panel's Image tab)

    Each image's metadata comes from a sidecar `<png>.json` (written by
    `generate_shot_images`). For older PNGs without sidecars, metadata
    is reconstructed from the file path + mtime.

    USE THIS for cross-shot character continuity:
      1. User uploaded photos of a character to anchor on
      2. Earlier this session you ran `generate_shot_images` for shot S1
         and the user picked a candidate
      3. For shot S2, instead of regenerating from scratch, find the
         picked candidate via list_library_images and pass it as a `ref`
         to generate_shot_images for S2 (Qwen-Image-Edit-2509 will
         compose the same character into the new prompt).

    USE THIS to find references the user prepared earlier:
      The user can manually generate stills via the panel's Image tab
      and they all land in the library. Surfacing those stills via
      list_library_images means the agent can use them on shots without
      the user having to re-explain the character/place each time.

    Args:
      limit: int = 24 — max items to return (1-200), newest first
      since: float = 0 — unix timestamp filter; only return images with
                         generated_at > since. Use 0 for "all time".
      contains: str = None — case-insensitive substring filter on prompt
      session_id: str = None — restrict to one agent session id; pass
                              the active session_id ('current') to filter
                              to the in-flight session
      shot_label: str = None — restrict to one shot label
      include_manual: bool = True — include manually-generated stills
      include_agent: bool = True — include agent-generated stills

    Returns: {
      count, total_scanned, images: [{
        png_path, prompt, refs, engine, family, model, seed,
        width, height, aspect, session_id, shot_label, take_index,
        generated_at, source ("agent.generate_shot_images"|"panel.image_tab"|"unknown")
      }, ...]
    }
    Newest first by `generated_at` (or file mtime fallback).
    """
    limit = max(1, min(200, int(args.get("limit", 24))))
    since = float(args.get("since") or 0)
    contains = args.get("contains")
    contains_norm = (contains or "").strip().lower()
    sid_filter = args.get("session_id")
    if sid_filter == "current":
        sid_filter = session.get("session_id")
    label_filter = args.get("shot_label")
    include_manual = bool(args.get("include_manual", True))
    include_agent = bool(args.get("include_agent", True))

    roots: list[Path] = []
    if include_agent:
        roots.append(ops.uploads_dir / "agentflow")
    if include_manual:
        roots.append(ops.uploads_dir / "library" / "manual")

    # Scan caps. Without these, after months of agent use the dir holds
    # tens of thousands of files and rglob walks the entire tree before
    # the limit can help — every list_library_images call costs seconds.
    # MAX_SCAN bounds the worst case; MAX_DEPTH stops descent into
    # surprise-deep trees (a malicious upload nest, or a future bug
    # that lands files way down). The agentflow layout is 4 levels:
    # agentflow/<sid>/<label>/[take_NN]/cand_NN.png.
    MAX_DEPTH = 5
    MAX_SCAN = 5000
    items: list[dict] = []
    total_scanned = 0

    def _walk_capped(start: Path, base_depth: int = 0):
        """Yield .png files under `start` up to MAX_DEPTH below the
        given base. Pure os.scandir for performance — Path.rglob is
        2-3× slower at this size. Stops cleanly when MAX_SCAN is hit."""
        nonlocal total_scanned
        stack = [(start, base_depth)]
        while stack and total_scanned < MAX_SCAN:
            cur, depth = stack.pop()
            if depth > MAX_DEPTH:
                continue
            try:
                with os.scandir(cur) as it:
                    entries = list(it)
            except (PermissionError, FileNotFoundError, NotADirectoryError):
                continue
            for entry in entries:
                if total_scanned >= MAX_SCAN:
                    return
                # follow_symlinks=False: don't traverse into symlinks
                # the user / agent dropped that point outside the tree.
                try:
                    if entry.is_dir(follow_symlinks=False) and depth < MAX_DEPTH:
                        stack.append((Path(entry.path), depth + 1))
                    elif entry.is_file(follow_symlinks=False) and entry.name.lower().endswith(".png"):
                        total_scanned += 1
                        yield Path(entry.path)
                except OSError:
                    continue

    for root in roots:
        if not root.exists():
            continue
        for png in _walk_capped(root):
            sidecar = png.with_suffix(png.suffix + ".json")
            meta: dict = {}
            if sidecar.is_file():
                try:
                    meta = json.loads(sidecar.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    meta = {}
            # Backfill missing fields from path / stat for sidecar-less PNGs.
            if "png_path" not in meta:
                meta["png_path"] = str(png)
            if "generated_at" not in meta:
                try:
                    meta["generated_at"] = png.stat().st_mtime
                except OSError:
                    meta["generated_at"] = 0
            if "source" not in meta:
                meta["source"] = (
                    "agent.generate_shot_images" if "agentflow" in png.parts
                    else "panel.image_tab" if "library" in png.parts
                    else "unknown"
                )
            # Path-based shot_label / session_id fallback for old PNGs:
            # panel_uploads/agentflow/<sid>/<label>/[take_NN/]cand_*.png
            if "session_id" not in meta and "agentflow" in png.parts:
                try:
                    i = png.parts.index("agentflow")
                    if i + 1 < len(png.parts):
                        meta["session_id"] = png.parts[i + 1]
                    if i + 2 < len(png.parts):
                        meta["shot_label"] = png.parts[i + 2]
                except (ValueError, IndexError):
                    pass

            # Filters
            if since and meta.get("generated_at", 0) <= since:
                continue
            if contains_norm:
                p = (meta.get("prompt") or "").lower()
                if contains_norm not in p:
                    continue
            if sid_filter and meta.get("session_id") != sid_filter:
                continue
            if label_filter and meta.get("shot_label") != label_filter:
                continue

            items.append(meta)

    items.sort(key=lambda m: m.get("generated_at", 0), reverse=True)
    items = items[:limit]
    # Slim each entry to the fields the agent actually consumes; drops
    # noise like the full sidecar schema string.
    out = []
    for m in items:
        out.append({
            "png_path": m.get("png_path"),
            "prompt": m.get("prompt"),
            "refs": m.get("refs") or [],
            "engine": m.get("engine"),
            "family": m.get("family"),
            "model": m.get("model"),
            "seed": m.get("seed"),
            "width": m.get("width"),
            "height": m.get("height"),
            "aspect": m.get("aspect"),
            "session_id": m.get("session_id"),
            "shot_label": m.get("shot_label"),
            "take_index": m.get("take_index"),
            "generated_at": m.get("generated_at"),
            "source": m.get("source"),
        })
    return {"count": len(out), "total_scanned": total_scanned, "images": out}


@tool("read_project_notes")
def _read_project_notes(args: dict, ops: PanelOps, session: dict) -> dict:
    """Read the full project-notes file.

    Project notes are durable memory across sessions. The system prompt
    already includes a 6 KB tail every turn — call this tool only when
    you need OLDER context (e.g. the user references a decision from
    weeks ago, or you need to see the running cast list in full).

    Returns: { path, char_count, content }.
    """
    from agent import project as _project
    full = _project.read_notes(ops.state_dir)
    return {
        "path": str(_project.notes_path(ops.state_dir)),
        "char_count": len(full),
        "content": full,
    }


@tool("append_project_notes")
def _append_project_notes(args: dict, ops: PanelOps, session: dict) -> dict:
    """Append a durable memory entry to the project notes file.

    Use this whenever a decision deserves to outlive the current chat:
    a master style choice, a character bible entry, an anchor-PNG path
    you want reused, a tier choice the user explicitly approved. The
    entry is timestamped automatically.

    Args:
      text: str — what to remember (one short paragraph or bullet).
      kind: str (optional, default "note") — labels: "note", "style",
        "cast", "decision". Visible in the markdown header for
        skimmability.
    """
    from agent import project as _project
    text = _required(args, "text")
    kind = (args.get("kind") or "note").strip()[:24] or "note"
    try:
        return _project.append_note(ops.state_dir, text, kind=kind, author="agent")
    except ValueError as e:
        raise _ToolValidationError(str(e)) from e


@tool("finish")
def _finish(args: dict, ops: PanelOps, session: dict) -> dict:
    """Signal that the agent has finished its current task.

    The runtime uses this as an explicit loop terminator so the agent
    doesn't keep spinning after committing the plan. The summary is
    shown to the user as the final assistant message.

    Args:
      summary: str  — one-paragraph wrap-up
      next_steps: str  — optional. What the user should do when they wake.
    """
    session["finished"] = True
    return {
        "summary": args.get("summary", ""),
        "next_steps": args.get("next_steps", ""),
    }


# ---- helpers ---------------------------------------------------------------
def _trim_job(j: dict | None) -> dict | None:
    if not j:
        return None
    p = j.get("params", {}) or {}
    return {
        "id": j.get("id"),
        "status": j.get("status"),
        "label": p.get("label"),
        "mode": p.get("mode"),
        "quality": p.get("quality"),
        "frames": p.get("frames"),
        "width": p.get("width"),
        "height": p.get("height"),
        "session_tag": p.get("session_tag"),
        "output_path": j.get("output_path"),
        "elapsed_sec": j.get("elapsed_sec"),
        "queued_at": j.get("queued_at"),
        "started_at": j.get("started_at"),
        "finished_at": j.get("finished_at"),
        "error": j.get("error"),
    }


def _resolve_path(p: str, base: Path) -> str:
    """Make absolute. If `p` is relative, resolve under `base`."""
    if not p:
        return p
    pp = Path(p)
    if pp.is_absolute():
        return str(pp)
    return str(base / pp)


def _ensure_under(path: Path, allowed_roots: list[Path]) -> Path:
    """Resolve `path` and require it lives under one of `allowed_roots`.

    Defense against prompt-injected file paths reaching `inspect_clip` /
    `extract_frame`. Without this an LLM coaxed by a malicious doc could
    point ffmpeg at /etc/* or read sidecars outside the project tree.
    Symlink/.. tricks are neutralized because we resolve both ends.
    """
    resolved = path.resolve()
    rroots = []
    for r in allowed_roots:
        try:
            rroots.append(r.resolve())
        except OSError:
            pass
    for r in rroots:
        try:
            if resolved.is_relative_to(r):
                return resolved
        except ValueError:
            continue
    raise _ToolValidationError(
        f"path {resolved} is outside the allowed project directories"
    )


def _probe_frame_count(clip_path: str) -> int:
    """Get nb_frames via ffprobe. Falls back to a duration*fps estimate.

    Used for clamping `which` indices in extract_frame.
    """
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-count_frames", "-show_entries", "stream=nb_read_frames",
             "-of", "default=nokey=1:noprint_wrappers=1", clip_path],
            capture_output=True, text=True, timeout=30,
        )
        n = int(r.stdout.strip())
        if n > 0:
            return n
    except Exception:                               # noqa: BLE001
        pass
    # Fallback: parse duration * 24 fps. Conservative.
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "format=duration", "-of",
             "default=nokey=1:noprint_wrappers=1", clip_path],
            capture_output=True, text=True, timeout=10,
        )
        return max(1, int(float(r.stdout.strip()) * 24))
    except Exception:                               # noqa: BLE001
        return 121


# ---- Action-block extraction ----------------------------------------------
# The agent emits actions in fenced blocks. Be permissive with the
# fence syntax (some models prefer ```json, some ```action, some no
# language tag at all when they only emit one block).

_FENCE_PATTERNS = [
    re.compile(r"```(?:action|tool|json|json action|action_json)\s*\n(.+?)\n```", re.DOTALL | re.IGNORECASE),
    re.compile(r"```\s*\n(\{[^`]*?\"tool\"[^`]*?\})\s*\n```", re.DOTALL),
]


def parse_action_block(content: str) -> dict | None:
    """Find the first action block in a model reply and parse its JSON.

    Returns a dict like {"tool": "...", "args": {...}} or None if no block
    is present. Malformed JSON returns None — the runtime treats that
    case as "model emitted text but failed to invoke a tool" and replies
    asking it to fix the syntax.
    """
    for pat in _FENCE_PATTERNS:
        m = pat.search(content)
        if not m:
            continue
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "tool" in obj:
            obj.setdefault("args", {})
            return obj
    return None


def strip_action_block(content: str) -> str:
    """Return the assistant text with the action block removed.

    Used when we want to show the user the model's prose without the
    raw JSON cluttering the chat (the JSON is rendered as a tool-call
    chip in the UI instead).
    """
    out = content
    for pat in _FENCE_PATTERNS:
        out = pat.sub("", out)
    return out.strip()
