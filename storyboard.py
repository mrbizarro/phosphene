#!/usr/bin/env python3.11
"""Storyboard — plan a run of shots that share a character, then shoot them.


WHAT THIS IS, IN PHOSPHENE'S OWN TERMS
--------------------------------------
Phosphene's thesis is "your trained character, in any scene." A storyboard is simply that at
sequence scale: many scenes, one identity, planned before you spend the render time. It is
NOT a new studio bolted on the side — it is a layer ABOVE the existing modes that composes
them. A shot is just a normal panel job in one of the modes that already exist
(text / character / remix / keyframe / extend / a2v), so anything the panel can render, a
storyboard can schedule.

Naming follows the panel's plain voice — Text, Character, Extend, Remix, Train Character —
and names the artifact (the reviewable plan), because plan-before-you-render is the point.

WHAT IT REUSES RATHER THAN REBUILDS  (integration, not duplication)
-------------------------------------------------------------------
* Execution + crash-resume: the panel ALREADY has a persistent queue with crash-resume
  (`state/panel_queue.json`, `/queue/batch`). Shots are enqueued as ordinary jobs; we do NOT
  run a second scheduler and do NOT re-implement resume. storyboard.json holds the creative
  PLAN; panel_queue.json owns EXECUTION. Two files, two jobs, no overlap.
* Outputs: clips land in `mlx_outputs/` like every other render, so they appear in the normal
  gallery, carry the usual .json sidecar, and work with Params/Extend/Expand.
* Characters: `list_characters()` is the source of truth for casting.
* Tier limits: resolution is checked against the same cap-tier clamp the panel uses.

THE TWO HARDWARE FACTS THAT SHAPE THE ARCHITECTURE
---------------------------------------------------
1. `mlx_warm_helper` holds exactly ONE pipeline kind at a time — `_free_all_but(keep_kind)`
   nulls every other pipeline. A pipeline switch is a full reload, costing MINUTES.
2. Gemma occupies that same slot (`keep_kind != "gemma_lm"` frees it). On unified memory the
   planner and the renderer are mutually exclusive, not merely co-resident-but-tight.

Consequences, which are the whole architecture:
  * PLAN -> VALIDATE -> SHOOT run strictly sequentially. The LLM is evicted before any frame
    renders, and is never re-entered mid-shoot. The plan is fully materialized up front.
  * Shots are rendered GROUPED BY MODE, not in story order, so the helper reloads a pipeline
    once per kind instead of once per shot. Clips are re-sorted by `n` at assembly.

PLANNER MODEL: Qwen3.5, NOT Gemma  (decided 2026-07-24)
--------------------------------------------------------
WHAT ACTUALLY SHIPS IS GEMMA. Read this section as the decision record it is, not as a
description of the running system: `storyboard_planner.py` plans on
`mlx_models/gemma-3-12b-it-4bit` — the weights the panel already downloads for
`/prompt/enhance` — so the Storyboard tab costs a user zero new bytes. Qwen3.5 remains the
target and is a one-line switch (`LTX_STORYBOARD_PLANNER`); the reasoning below is why.
See the MODEL section at the top of `storyboard_planner.py` for the shipped arrangement.

Gemma 3 stays for `/prompt/enhance` — different task, already tuned, already loaded. But
planning is 100% structured output, and this project already documented that "Gemma 3 has no
native tool_calls". Gemma 4 has no clean official mlx-community 4-bit build (third-party /
uncensored / GGUF forks only), so it is not shippable.

  mlx-community/Qwen3.5-4B-4bit   3.06 GB  <- default planner
  mlx-community/Qwen3.5-9B-4bit   5.98 GB  <- optional upgrade
  gemma-3-12b-it-4bit             7.50 GB  <- stays, enhancement only

The planner is therefore better at the job AND smaller than the model it replaces. Fetched
lazily on first Storyboard use, so users who never touch the feature pay 0 bytes.

Qwen's "thinking" mode can leak reasoning into output, so validity is enforced in three
layers rather than hoped for: constrained decoding (mlx_lm.sample_utils.make_logits_processors,
present in the pinned mlx-lm 0.31.1), preamble stripping, then validate_storyboard() as the
final gate with a single repair retry.

A HARD-WON PANEL RULE THIS MODULE MUST HONOR
---------------------------------------------
`mlx_ltx_panel.py` submits character jobs with `enhance=false` and the comment
"CRITICAL: don't let Gemma strip the trigger". Gemma rewrites prompts and drops the trigger
token, which silently renders a stranger instead of the trained face. Therefore:
  * The planner emits FINAL prompt text. Shots are enqueued with enhance OFF.
  * Triggers are injected mechanically in Python, never left to the model.
  * `validate_storyboard()` fails a shot whose prompt lost its trigger — see the check below.

This module is deliberately pure-stdlib and side-effect-free with respect to models: nothing
here loads a pipeline. Phase 1 (plan) and Phase 3 (shoot) call OUT to the existing panel
machinery; everything in here is schema, validation, scheduling and durable state, so it can
be unit-tested on any machine with no GPU and no weights.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 1

# Modes a shot may use. These MUST exist as real panel modes — the planner is never trusted
# to invent one. Kept as a plain tuple so validation errors can list the legal set.
VALID_MODES = ("text", "character", "remix", "keyframe", "extend", "a2v")

# Modes whose pipeline the warm helper caches independently. Used only for grouping/estimation;
# the authoritative mapping lives in the helper. Shots that share a bucket render back-to-back
# without paying a pipeline reload.
_PIPELINE_BUCKET = {
    "text": "t2v",
    "character": "t2v",   # character is a UI intent over t2v + fused LoRAs
    "remix": "remix",
    "keyframe": "keyframe",
    "extend": "extend",
    "a2v": "a2v",
}

# Storyboard mode -> the mode string the PANEL'S job form actually speaks.
#
# This is load-bearing and it is not cosmetic. `mlx_ltx_panel.py` has exactly one backend
# video mode for both of v1's shot types: t2v. "Character" is a UI INTENT there, not a mode —
# the panel's own setMode('character') sets the hidden #mode field to 't2v' and lets
# `character_id` drive the LoRA stack ("Mode hidden field still 't2v' — backend doesn't know
# 'character' and doesn't need to"). Posting mode="character" or mode="text" to /queue/add
# reaches run_job_inner with a mode nothing dispatches on.
_PANEL_MODE = {
    "text": "t2v",
    "character": "t2v",
    "remix": "t2v",        # remix rides t2v + an IC-LoRA ref; not enqueued in v1 (no refs)
    "keyframe": "keyframe",
    "extend": "extend",
    "a2v": "a2v",
}

# Engines a shot may name. Anything else falls back to the built-in one.
VALID_ENGINES = ("ltx", "h3")
DEFAULT_ENGINE = "ltx"

# The FILM-level engine choice, which the user makes once in the brief.
#   "auto" — the plan decides per shot: a cast shot goes to LTX (that is where
#            character LoRAs load), everything else to H3.
#   "h3"   — every shot on Hailuo H3. The planner writes the whole film in H3's
#            three-field dialect. A trained character cannot come along: H3
#            stacks no LoRAs, so a cast shot would render a stranger.
#   "ltx"  — every shot on LTX-2.3. The planner drops the H3 dialect entirely.
# It is a PLANNING input, not a post-hoc filter: the two engines want prompts
# written differently (`_assemble_h3_prompt` vs `_assemble_ltx_prompt`), so
# changing it on an existing film means re-planning, not re-labelling.
ENGINE_MODES = ("auto", "h3", "ltx")
DEFAULT_ENGINE_MODE = "auto"

# A film is shot TWICE: a cheap draft pass you watch, then a delivery pass of
# only the shots you kept. Which pass a shot has already been through is a
# property of that PASS, and the board records it as the pass's own output key —
# never as the shot's single `status`, which only ever describes the most recent
# thing that happened to it.
#
# That distinction is load-bearing and it was once got wrong: the scheduler
# filtered on `status not in ("done", "skipped")` while the reconciler set
# `status = "done"` the moment the DRAFT landed, so by the time the delivery
# pass asked for work every shot looked finished. `/storyboard/render pass=final`
# answered 202 and enqueued nothing, for every film, forever.
PASS_NAMES = ("draft", "final")
_PASS_OUTPUT_KEY = {"draft": "draft_output", "final": "final_output"}
_PASS_JOB_KEY = {"draft": "draft_job_id", "final": "final_job_id"}


def pass_output_key(pass_name: str = "draft") -> str:
    """The board field holding this pass's rendered clip."""
    return _PASS_OUTPUT_KEY.get((pass_name or "draft").strip().lower(), "draft_output")


def pass_job_key(pass_name: str = "draft") -> str:
    """The board field holding this pass's job id."""
    return _PASS_JOB_KEY.get((pass_name or "draft").strip().lower(), "draft_job_id")


def shot_pass_done(shot: dict, pass_name: str = "draft") -> bool:
    """Has THIS shot already been through THIS pass? The only correct test."""
    return bool(shot.get(pass_output_key(pass_name)))


def shots_pending(shots: Iterable[dict], pass_name: str = "draft") -> list[dict]:
    """Shots this pass still has to render, in board order.

    A cut shot is out of every pass. Everything else is in until it has that
    pass's own output — a finished draft does NOT excuse a shot from delivery.
    """
    return [s for s in (shots or [])
            if isinstance(s, dict)
            and s.get("status") != "skipped"
            and not shot_pass_done(s, pass_name)]


def resolve_engine(shot: dict, *, engine_mode: str = DEFAULT_ENGINE_MODE,
                   h3_available: bool = True) -> str:
    """The engine this shot will ACTUALLY render on. One function, every caller.

    Precedence, strongest first:
      1. no H3 pack on this machine -> LTX, always;
      2. the shot is cast -> LTX, always (H3 stacks no LoRAs, so a cast shot on
         H3 renders a stranger — the exact failure `ensure_trigger` exists for);
      3. the film's engine mode, when it is not "auto";
      4. what the plan wrote on the shot.
    """
    if not h3_available:
        return "ltx"
    if shot.get("character_id"):
        return "ltx"
    mode = (engine_mode or DEFAULT_ENGINE_MODE).strip().lower()
    if mode in ("h3", "ltx"):
        return mode
    return shot_engine(shot)

# LTX renders at 24 fps and the sampler requires frames % 8 == 1.
LTX_FPS = 24

# H3's duration axis. The panel owns the canonical table (H3_TIERS, built from a quality axis
# and a length axis); these are the LENGTH keys and the frame count each one delivers, on the
# 17n+5 grid the runner snaps to. Duplicated here — as data, not as a second cost model — so
# `shot_to_job()` emits a self-consistent job dict without importing the panel (this module is
# deliberately pure-stdlib so it unit-tests with no GPU and no weights). make_job re-stamps
# width/height/frames from the real cell, so the panel is still the authority; a drift here
# can only make the pre-flight job dict look wrong, never make a render come out wrong.
H3_LENGTHS = ("3s", "5s", "10s", "15s")
_H3_LENGTH_SECONDS = {"3s": 3.0, "5s": 5.0, "10s": 10.0, "15s": 15.0}
_H3_LENGTH_FRAMES = {"3s": 73, "5s": 124, "10s": 243, "15s": 362}
# How many chained 5 s windows each length renders as. Anything past 5 s is
# N windows stitched by the runner, and every window is asked for a prompt.
_H3_LENGTH_WINDOWS = {"3s": 1, "5s": 1, "10s": 2, "15s": 3}

# Does this shot have spoken lines? The planner writes dialogue in explicit
# <d>…</d> tags, so this is a DERIVATION rather than a guess — which is the only
# reason the storyboard lane is allowed to decide `no_voice` automatically at
# all.
#
# The lookahead is load-bearing: a bare `<d>\s*\S` matches `<d></d>`, because
# the `<` of the CLOSING tag is itself non-whitespace. An empty tag is a planner
# artefact and means silence, so it must not load the voice.
_HAS_DIALOGUE = re.compile(r"<d>\s*(?!</\s*d\s*>)\S", re.I)

# A pass's quality (the LTX vocabulary the policy speaks) -> H3's canvas axis.
# quick 640x384 · standard 768x448 · high 1024x576 are the three offered canvases;
# "native" (1344x768) is deliberately unreachable from a storyboard pass — it is a ~20 min
# clip with Turbo and ~45 without, which is not a cost any pass-level choice should be able
# to opt someone into silently.
_H3_QUALITY_FOR_PASS = {
    "quick": "draft",
    "balanced": "standard",
    "standard": "high",
    "high": "high",
}

# Rough per-second-of-video render cost, by quality, in seconds of wall clock. Deliberately
# pessimistic — the estimate exists to stop someone starting a 6-hour job unaware, not to be
# a benchmark. Tuned against observed two-stage 1536x896 ~11 min for a 5 s clip.
_SECS_PER_VIDEO_SEC = {"quick": 24.0, "balanced": 60.0, "standard": 96.0, "high": 132.0}
_PIPELINE_LOAD_SECS = 90.0

# H3 FALLBACK cost, per second of video, by canvas. The panel passes `h3_cost=` into
# estimate() and that hook wins every time — it reads H3_TIERS[cell]["eta_min"], which is
# MEASURED wall clock per cell and is the only number allowed on screen. This table exists so
# storyboard.py keeps working (and keeps unit-testing) standalone, and it is derived from the
# same measurements rather than invented:
#   draft 3s     3.0 min / 3 s  =  60 s per video-second
#   standard 5s  9.1 min / 5 s  = 109
#   high 5s     18.8 min / 5 s  = 226   (no Turbo)
#   native 5s   44.9 min / 5 s  = 538   (no Turbo, never selected by a pass)
# An H3 clip is a fresh subprocess that loads its own weights, so this per-clip number is
# END TO END — which is why an H3 bucket adds no separate pipeline-load charge below.
_H3_SECS_PER_VIDEO_SEC = {"draft": 60.0, "standard": 109.0, "high": 226.0, "native": 538.0}


class StoryboardError(Exception):
    """Raised for malformed storyboards. Message is user-facing."""


# ---------------------------------------------------------------------------
# Durable state
# ---------------------------------------------------------------------------

def board_dir(state_dir: Path, board_id: str) -> Path:
    return Path(state_dir) / "storyboards" / board_id


def save_storyboard(state_dir: Path, board: dict) -> Path:
    """Write storyboard.json ATOMICALLY.

    Renders run for hours; the panel gets killed, Pinokio restarts, Macs sleep. A torn
    storyboard.json would lose the whole run, so we always write a temp file in the same directory
    and os.replace() it (atomic within a filesystem).
    """
    fid = board.get("id")
    if not fid:
        raise StoryboardError("storyboard has no id")
    d = board_dir(state_dir, fid)
    d.mkdir(parents=True, exist_ok=True)
    target = d / "storyboard.json"
    fd, tmp = tempfile.mkstemp(dir=str(d), prefix=".sb-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(board, fh, indent=2, ensure_ascii=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return target


def load_storyboard(state_dir: Path, board_id: str) -> dict:
    p = board_dir(state_dir, board_id) / "storyboard.json"
    if not p.is_file():
        raise StoryboardError(f"no such storyboard: {board_id}")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise StoryboardError(f"storyboard.json is corrupt: {e}") from e


def list_storyboards(state_dir: Path) -> list[dict]:
    """Newest first. Returns light summaries, not full specs."""
    root = Path(state_dir) / "storyboards"
    out: list[dict] = []
    if not root.is_dir():
        return out
    for d in root.iterdir():
        p = d / "storyboard.json"
        if not p.is_file():
            continue
        try:
            f = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        shots = f.get("shots") or []
        out.append({
            "id": f.get("id", d.name),
            "title": f.get("title", ""),
            "created_at": f.get("created_at", 0),
            "shots": len(shots),
            "done": sum(1 for s in shots if s.get("status") == "done"),
            "failed": sum(1 for s in shots if s.get("status") == "failed"),
        })
    out.sort(key=lambda r: r.get("created_at") or 0, reverse=True)
    return out


# ---------------------------------------------------------------------------
# Phase 2 — VALIDATE (no models, sub-second, runs BEFORE anything renders)
# ---------------------------------------------------------------------------

def validate_storyboard_detail(
    board: dict,
    *,
    known_character_ids: Iterable[str] = (),
    ref_root: Path | None = None,
    max_dim: int | None = None,
) -> list[dict]:
    """Every check `validate_storyboard()` makes, STRUCTURED. Empty list == good to shoot.

    Each entry is::

        {"code": "missing_trigger",         # stable machine name, never translated
         "n": 3 | None,                     # which shot, when it is about one
         "field": "prompt" | None,          # which control to focus / flag
         "message": "<the exact string validate_storyboard() has always returned>",
         "data": {...}}                     # whatever a fix button needs

    This exists so the UI can render its own human copy per `code` and still put a Fix button
    beside the right control, WITHOUT regexing English out of `message`. `validate_storyboard()`
    below is now a formatter over this, so the strings stay byte-identical by construction and
    the panel's copy can never drift from the check that produced it.

    The whole point of both: a two-hour render must never begin on a plan that cannot succeed.
    Everything here is cheap string/path/int checking, so it costs ~nothing and can run on
    every save. Errors read as a fix-list, not a stack trace.
    """
    errs: list[dict] = []
    chars = set(known_character_ids or ())

    def add(code: str, message: str, *, n: int | None = None,
            field: str | None = None, **data: Any) -> None:
        errs.append({"code": code, "n": n, "field": field,
                     "message": message, "data": data})

    if board.get("schema") != SCHEMA_VERSION:
        add("schema_version",
            f"schema version {board.get('schema')!r} — this build understands {SCHEMA_VERSION}",
            got=board.get("schema"), expected=SCHEMA_VERSION)
    if not str(board.get("id") or "").strip():
        add("board_id_empty", "storyboard.id is empty")

    shots = board.get("shots")
    if not isinstance(shots, list) or not shots:
        add("no_shots", "storyboard has no shots")
        return errs

    policy = board.get("policy") or {}
    seen_n: set[int] = set()

    # ---- locations ------------------------------------------------------
    locs = board.get("locations")
    known_locs: set[str] = set()
    known_views: dict[str, set[str]] = {}
    if locs is not None and not isinstance(locs, list):
        add("locations_shape", "storyboard.locations must be a list")
    else:
        seen_loc: set[str] = set()
        for i, loc in enumerate(locs or []):
            if not isinstance(loc, dict):
                add("location_not_object", f"location {i + 1}: not an object")
                continue
            lid = str(loc.get("id") or "").strip().lower()
            if not LOCATION_ID_RE.match(lid):
                add("location_id", f"location {i + 1}: id {loc.get('id')!r} must be "
                                   f"lowercase letters, digits, - or _", got=loc.get("id"))
                continue
            if lid in seen_loc:
                add("location_duplicate", f"location {i + 1}: duplicate id {lid!r}",
                    duplicate=lid)
                continue
            seen_loc.add(lid)
            known_locs.add(lid)
            desc = str(loc.get("description") or "").strip()
            if not desc:
                # An empty location is worse than none: it reads as continuity
                # being handled while injecting nothing at all.
                add("location_empty", f"location {lid!r} has no description — "
                                      f"it would pin nothing", location_id=lid)
            elif len(desc) > LOCATION_DESC_MAX:
                add("location_too_long",
                    f"location {lid!r}: description is {len(desc)} characters "
                    f"(max {LOCATION_DESC_MAX})", location_id=lid, length=len(desc))

            # ---- views: the same place, faced the other way -------------
            views = loc.get("views")
            known_views[lid] = set()
            if views is not None and not isinstance(views, list):
                add("views_shape", f"location {lid!r}: views must be a list",
                    location_id=lid)
                continue
            seen_view: set[str] = set()
            for j, view in enumerate(views or []):
                vid = str((view or {}).get("id") or "").strip().lower() \
                    if isinstance(view, dict) else ""
                if not VIEW_ID_RE.match(vid):
                    add("view_id",
                        f"location {lid!r}, view {j + 1}: id "
                        f"{(view or {}).get('id') if isinstance(view, dict) else view!r} "
                        f"must be lowercase letters, digits, - or _",
                        location_id=lid, got=(view.get("id") if isinstance(view, dict)
                                              else view))
                    continue
                if vid in seen_view:
                    add("view_duplicate",
                        f"location {lid!r}: duplicate view id {vid!r}",
                        location_id=lid, duplicate=vid)
                    continue
                seen_view.add(vid)
                # KNOWN even when its description is wrong: the view exists,
                # and reporting the real fault plus `unknown_view` on every
                # shot that names it buries the one line worth reading.
                known_views[lid].add(vid)
                vdesc = str(view.get("description") or "").strip()
                if not vdesc:
                    # Worse than no view: a shot naming it would fall back to
                    # the location description — the establishing angle — which
                    # is the exact prompt the reverse angle exists to escape.
                    add("view_empty",
                        f"location {lid!r}: view {vid!r} has no description — a shot "
                        f"naming it would fall back to the establishing angle",
                        location_id=lid, view=vid)
                elif len(vdesc) > LOCATION_DESC_MAX:
                    add("view_too_long",
                        f"location {lid!r}, view {vid!r}: description is {len(vdesc)} "
                        f"characters (max {LOCATION_DESC_MAX})",
                        location_id=lid, view=vid, length=len(vdesc))

    for idx, s in enumerate(shots):
        # `where` used to be computed from s.get() BEFORE the isinstance check, so a shot that
        # was a string (exactly the case the next line reports) raised AttributeError instead
        # of returning the error. Fall back to the positional number for a non-dict.
        if not isinstance(s, dict):
            add("shot_not_object", f"shot {idx + 1}: not an object", n=idx + 1)
            continue
        num = s.get("n", idx + 1)
        where = f"shot {num}"
        n_for_ui = num if isinstance(num, int) else idx + 1

        n = s.get("n")
        if not isinstance(n, int) or n < 1:
            add("shot_number", f"{where}: 'n' must be a positive integer",
                n=n_for_ui, field="n", got=n)
        elif n in seen_n:
            add("shot_duplicate", f"{where}: duplicate shot number {n}",
                n=n_for_ui, field="n", duplicate=n)
        else:
            seen_n.add(n)

        # A shot that names a location the board does not have would render
        # with NO location injected and look exactly like a shot that never
        # claimed one — the continuity failure arriving silently, which is the
        # thing locations exist to stop.
        lref = s.get("location_id")
        if lref is not None and str(lref).strip():
            lref = str(lref).strip().lower()
            if lref not in known_locs:
                add("unknown_location",
                    f"{where}: location {lref!r} is not on this storyboard",
                    n=n_for_ui, field="location_id", location_id=lref,
                    known=sorted(known_locs))
        else:
            lref = ""

        # A shot naming a view that is gone fails EXACTLY the way an unknown
        # location fails — the reverse angle silently composes from the
        # establishing description, so the car the whole view existed to get
        # out of frame is back in frame, and nothing said a word about it.
        vref = s.get("view")
        if vref is not None and str(vref).strip():
            vref = str(vref).strip().lower()
            if not lref:
                add("unknown_view",
                    f"{where}: view {vref!r} but the shot names no location — "
                    f"a view belongs to a location",
                    n=n_for_ui, field="view", view=vref)
            elif vref not in (known_views.get(lref) or set()):
                add("unknown_view",
                    f"{where}: view {vref!r} is not a view of location {lref!r}",
                    n=n_for_ui, field="view", view=vref, location_id=lref,
                    known=sorted(known_views.get(lref) or ()))

        eye = s.get("eyeline")
        if eye is not None and str(eye).strip() and str(eye).strip().lower() not in EYELINES:
            add("bad_eyeline",
                f"{where}: eyeline {eye!r} is not one of {', '.join(EYELINES)}",
                n=n_for_ui, field="eyeline", eyeline=eye, valid=list(EYELINES))

        mode = s.get("mode")
        if mode not in VALID_MODES:
            add("bad_mode",
                f"{where}: mode {mode!r} is not one of {', '.join(VALID_MODES)}",
                n=n_for_ui, field="mode", mode=mode, valid=list(VALID_MODES))

        prompt = (s.get("prompt") or "").strip()
        if not prompt:
            add("empty_prompt", f"{where}: empty prompt", n=n_for_ui, field="prompt")

        cid = s.get("character_id")
        if cid:
            if chars and cid not in chars:
                add("unknown_character",
                    f"{where}: character {cid!r} is not installed "
                    f"(have: {', '.join(sorted(chars)) or 'none'})",
                    n=n_for_ui, field="character_id",
                    character_id=cid, have=sorted(chars))
            # The trigger is injected mechanically, never trusted to the LLM — but if a plan
            # arrives with a character and the trigger is missing from the prompt, the LoRA
            # will not fire and the shot silently renders a stranger. Catch it here.
            trig = (s.get("trigger") or cid or "").strip()
            if trig and not re.search(rf"\b{re.escape(trig)}\b", prompt):
                add("missing_trigger",
                    f"{where}: prompt is missing the character trigger {trig!r}",
                    n=n_for_ui, field="prompt", trigger=trig)
        elif mode == "character":
            add("character_without_id",
                f"{where}: mode 'character' requires a character_id",
                n=n_for_ui, field="character_id")

        # No mouth moves without words. This blocks the render rather than
        # warning, on the same principle as everything else here: the failure
        # it prevents is a finished clip of somebody babbling, and you only
        # find out after the render.
        verb = shot_speech_problem(prompt)
        if verb:
            add("speech_without_words",
                f"{where}: {verb!r} implies someone is speaking, but no spoken "
                f"line is written — the model renders a moving mouth with "
                f"nothing to say. Write the line in quotes, or describe the "
                f"shot without speech.",
                n=n_for_ui, field="prompt", verb=verb)
        # And the inverse failure: words that exist but cannot survive the
        # clock. Blocks the render for the same reason everything here does —
        # the defect (a sentence cut off mid-word) is only visible AFTER the
        # render time is spent.
        pacing = shot_pacing_problem(prompt, s.get("duration_s") or 0)
        if pacing:
            add("dialogue_does_not_fit",
                f"{where}: {pacing}.",
                n=n_for_ui, field="prompt")

        dur = s.get("duration_s")
        # A take shot is as long as the take (up to 2 min); every other shot
        # is one clip and stops at 60.
        _dur_cap = max(TAKE_SECONDS) if s.get("take_seconds") in TAKE_SECONDS else 60
        if not isinstance(dur, (int, float)) or not (0 < float(dur) <= _dur_cap):
            add("bad_duration",
                f"{where}: duration_s must be between 0 and {_dur_cap} (got {dur!r})",
                n=n_for_ui, field="duration_s", duration_s=dur)

        refs = s.get("refs") or []
        if not isinstance(refs, list):
            add("refs_not_list", f"{where}: refs must be a list",
                n=n_for_ui, field="refs")
        else:
            for r in refs:
                rp = Path(r)
                if ref_root is not None and not rp.is_absolute():
                    rp = Path(ref_root) / rp
                if not rp.is_file():
                    add("ref_missing", f"{where}: reference image not found: {r}",
                        n=n_for_ui, field="refs", ref=str(r), name=Path(str(r)).name)
        if mode == "remix" and not refs:
            add("remix_needs_ref",
                f"{where}: mode 'remix' needs at least one reference image",
                n=n_for_ui, field="refs")

    # Resolution legality for the active tier. Q4 machines clamp; a plan that assumes 1536
    # on a 16 GB Mac would either clamp silently or swap, so surface it now.
    if max_dim:
        for key in ("draft", "final"):
            p = policy.get(key) or {}
            w, h = p.get("width"), p.get("height")
            if isinstance(w, int) and isinstance(h, int) and max(w, h) > max_dim:
                # `fit_*` is the offer the UI's one-click fix writes. It ships in
                # the error because the browser must never compute a canvas of
                # its own (docs/ARCHITECTURE.md) — and because the button used to
                # carry a hardcoded 1024x576, which on a 768px Mac was itself
                # illegal.
                fw, fh = fit_canvas(w, h, max_dim)
                add("over_cap",
                    f"policy.{key}: {w}x{h} exceeds this machine's {max_dim}px cap — "
                    f"lower it or the render will clamp",
                    field=f"policy.{key}", pass_name=key, width=w, height=h, max_dim=max_dim,
                    fit_width=fw, fit_height=fh)
    return errs


def validate_storyboard(
    board: dict,
    *,
    known_character_ids: Iterable[str] = (),
    ref_root: Path | None = None,
    max_dim: int | None = None,
) -> list[str]:
    """Return a list of human-readable problems. Empty list == good to shoot.

    A formatter over validate_storyboard_detail(): same checks, same order, byte-identical
    strings. Kept because it is the documented public surface and it reads better at a REPL.
    """
    return [e["message"] for e in validate_storyboard_detail(
        board,
        known_character_ids=known_character_ids,
        ref_root=ref_root,
        max_dim=max_dim,
    )]


# ---------------------------------------------------------------------------
# Phase 3 — SHOOT: scheduling
# ---------------------------------------------------------------------------

def shot_engine(shot: dict) -> str:
    """The engine a shot renders on, normalised. Unknown / missing -> the built-in one."""
    e = str(shot.get("engine") or "").strip().lower()
    return e if e in VALID_ENGINES else DEFAULT_ENGINE


def bucket_key(shot: dict) -> tuple[str, str]:
    """The scheduling bucket: (engine, pipeline kind).

    ENGINE IS PART OF THE KEY, and that is the whole fix. Keying on `mode` alone put an H3
    `text` shot in the same bucket as an LTX one, so a mixed film interleaved the two — and an
    H3 job *tears the LTX warm helper down* before it runs (run_h3_job_inner kills it; 40 GiB
    + the helper's weights does not fit on a 64 GB Mac) and the helper cold-starts again on the
    next LTX job. Interleaved, that teardown is paid per switch, which INVERTS the exact cost
    this grouping exists to avoid.
    """
    return (shot_engine(shot), _PIPELINE_BUCKET.get(shot.get("mode"), "t2v"))


def shooting_order(shots: list[dict], pass_name: str = "draft") -> list[dict]:
    """Group shots by pipeline bucket so the warm helper reloads once per KIND, not per shot.

    This is the single biggest wall-clock win on Apple Silicon and it exists purely because
    of `_free_all_but(keep_kind)`: rendering in story order across 3 interleaved modes costs
    one full pipeline load per switch (~90 s each), while grouped rendering costs one per
    bucket. 12 alternating shots: ~12 loads -> 3.

    Order within a bucket, and the order of buckets, is by first appearance in the story, so
    output stays deterministic and a resumed run reproduces the same sequence. Clips are
    re-sorted by `n` at assembly, so viewing order is unaffected.

    `pass_name` decides what "still to shoot" MEANS — see shots_pending(). Calling this
    without it schedules the draft pass, which is what every pre-existing caller wanted.
    """
    pending = shots_pending(shots, pass_name)
    bucket_first: dict[tuple[str, str], int] = {}
    for s in pending:
        b = bucket_key(s)
        n = s.get("n") or 0
        if b not in bucket_first or n < bucket_first[b]:
            bucket_first[b] = n
    return sorted(
        pending,
        key=lambda s: (bucket_first.get(bucket_key(s), 1 << 30), s.get("n") or 0),
    )


def h3_length_for(duration_s: float) -> str:
    """Snap a duration onto H3's length axis. Ties go to the shorter window."""
    d = float(duration_s or 0) or 5.0
    return min(H3_LENGTHS, key=lambda k: (abs(_H3_LENGTH_SECONDS[k] - d), _H3_LENGTH_SECONDS[k]))


def h3_quality_for(quality: str) -> str:
    """A pass's quality -> H3's canvas axis."""
    return _H3_QUALITY_FOR_PASS.get((quality or "").strip().lower(), "standard")


def ltx_frames_for(duration_s: float) -> int:
    """Seconds -> an LTX frame count on the sampler's `frames % 8 == 1` grid, at 24 fps.

    3 s -> 73 · 5 s -> 121 · 7 s -> 169 · 10 s -> 241, which is exactly the table docs/API.md
    publishes, so a storyboard shot and a hand-typed generation of the same length produce the
    same number of frames.
    """
    d = float(duration_s or 0)
    if d <= 0:
        return 0
    n = max(1, round(d * LTX_FPS))
    return max(9, 8 * round((n - 1) / 8) + 1)


def h3_chain_prompts_for(shot: dict) -> list[str]:
    """Per-window prompts for a chained H3 shot. `[]` when it isn't chained.

    A 10 s clip renders as two 5-second windows and a 15 s as three, and by
    default EVERY window is asked for the same prompt — so a one-off action
    ("he raises his arm", a line of dialogue) happens once per window and the
    clip reads as a repeat. That is the artifact `H3_CHAIN_PROMPT_HELP` in the
    panel describes, and it is the single most-reported thing about long H3
    clips.

    The planner writes ONE description per shot, so there is no second beat to
    hand window 2 — but there IS a `settle`: the state the shot is required to
    end in (law L3, and the field exists because the model skipped the law when
    it was only written as a rule). So the first window plays the shot, and
    every later window is told to hold that settled state and start nothing new.
    That is the honest reading of a 10 s shot whose action was written for one
    window, and it is strictly better than saying it twice.

    Window 1 is `""` — the panel's own contract for "use the main prompt" — so
    the shot's real prompt is never duplicated into the array.
    """
    if shot_engine(shot) != "h3":
        return []
    windows = _H3_LENGTH_WINDOWS.get(h3_length_for(shot.get("duration_s") or 0), 1)
    if windows <= 1:
        return []
    settle = (shot.get("settle") or "").strip().rstrip(".")
    if not settle:
        # Nothing honest to say about how it should continue — leave the array
        # empty, which is exactly today's behaviour, rather than inventing one.
        return []
    sound = (shot.get("soundscape") or "").strip()
    tail = ("integrated_multimodal_description: The shot continues without a cut, "
            "exactly where it left off. The camera holds completely still, the "
            "frame never moves - no pan, no push-in, no reframing. Nothing new "
            f"begins and no action restarts: {settle}, and that settled state is "
            "simply held for the whole window. Every face holds the exact angle "
            "to the lens it has at the start. No text appears at any point.")
    if sound:
        tail += "\n\noverall_soundscape: " + sound
    tail += "\n\nnon_diegetic_music: N/A"
    return [""] + [tail] * (windows - 1)


def shot_render_secs(shot: dict, policy_pass: dict, *, h3_cost=None) -> float:
    """Wall clock for ONE shot, on the engine it actually renders on.

    `h3_cost(quality_key, length_key) -> seconds | None` is the panel's measured per-cell hook
    (H3_TIERS[cell]["eta_min"] * 60). When it is absent or returns nothing we fall back to
    `_H3_SECS_PER_VIDEO_SEC`, which is derived from the same measurements. LTX keeps the
    per-second table it has always used.
    """
    dur = float(shot.get("duration_s") or 0)
    quality = policy_pass.get("quality", "balanced")
    if shot.get("take_seconds") in TAKE_SECONDS and shot_engine(shot) == "h3":
        # A take on H3 is parts of 15 s; price the parts, not the duration.
        parts = -(-(int(shot["take_seconds"]) // TAKE_BEAT_SECONDS) // 3)
        qk = h3_quality_for(quality)
        got = None
        if h3_cost is not None:
            try:
                got = h3_cost(qk, "15s")
            except Exception:
                got = None
        per = float(got) if got else _H3_SECS_PER_VIDEO_SEC.get(qk, 109.0) * _H3_LENGTH_SECONDS["15s"]
        return per * parts
    if shot_engine(shot) == "h3":
        lk = h3_length_for(dur)
        qk = h3_quality_for(quality)
        if h3_cost is not None:
            try:
                got = h3_cost(qk, lk)
            except Exception:
                got = None
            if got:
                return float(got)
        return _H3_SECS_PER_VIDEO_SEC.get(qk, 109.0) * _H3_LENGTH_SECONDS[lk]
    return dur * _SECS_PER_VIDEO_SEC.get(quality, 60.0)


def estimate(board: dict, *, pass_name: str = "final", h3_cost=None) -> dict:
    """Wall-clock estimate for a pass, accounting for pipeline reloads.

    Deliberately pessimistic. Its job is to let someone see '2 h 40 m' BEFORE committing,
    and to show what grouped scheduling saves versus naive story order.

    ENGINE-AWARE since 2026-08-11. It used to price every shot off the LTX per-second table,
    which under-reports an H3 shot by roughly 2x (an H3 5 s High clip is ~8.5 min measured
    against this table's ~5 min, and a Native 5 s is ~20). A number the summary bar prints
    before someone spends an afternoon has to be honest for the film they actually planned.

    Only LTX buckets are charged a pipeline load: an H3 job is a fresh subprocess that loads
    its own weights every time, so that cost is already inside its measured per-clip eta.
    """
    shots = shots_pending(board.get("shots") or [], pass_name)
    policy = (board.get("policy") or {}).get(pass_name) or {}

    render = sum(shot_render_secs(s, policy, h3_cost=h3_cost) for s in shots)

    grouped = {bucket_key(s) for s in shots}
    grouped_loads = len(grouped)
    grouped_ltx = sum(1 for b in grouped if b[0] != "h3")

    naive_loads = 0
    naive_ltx = 0
    prev = None
    for s in sorted(shots, key=lambda x: x.get("n") or 0):
        b = bucket_key(s)
        if b != prev:
            naive_loads += 1
            if b[0] != "h3":
                naive_ltx += 1
            prev = b

    engine_mix: dict[str, int] = {}
    for s in shots:
        e = shot_engine(s)
        engine_mix[e] = engine_mix.get(e, 0) + 1

    return {
        "pass": pass_name,
        "shots": len(shots),
        "render_secs": round(render),
        "pipeline_loads": grouped_loads,
        "total_secs": round(render + grouped_ltx * _PIPELINE_LOAD_SECS),
        "naive_total_secs": round(render + naive_ltx * _PIPELINE_LOAD_SECS),
        "saved_secs": round((naive_ltx - grouped_ltx) * _PIPELINE_LOAD_SECS),
        # Additive, for the UI: the run strip draws one segment per bucket, and the summary
        # bar says "Hailuo H3 isn't installed" only when it has to.
        "engine_mix": engine_mix,
        "runtime_secs": round(sum(float(s.get("duration_s") or 0) for s in shots)),
        "buckets": [
            {"engine": e, "kind": k,
             "shots": sorted(s.get("n") or 0 for s in shots if bucket_key(s) == (e, k))}
            for (e, k) in sorted(
                grouped,
                key=lambda b: min((s.get("n") or 0) for s in shots if bucket_key(s) == b),
            )
        ],
    }


def per_shot_estimate(board: dict, *, pass_name: str = "final", h3_cost=None) -> dict:
    """`{n: seconds}` for every shot in the board, on the same cost model estimate() uses.

    The shot card prints `~2 m` from this. Server-computed on purpose: the per-second constants
    and the measured H3 cells then live in exactly one place instead of being re-typed in JS.
    """
    policy = (board.get("policy") or {}).get(pass_name) or {}
    out: dict = {}
    for s in (board.get("shots") or []):
        if not isinstance(s, dict):
            continue
        out[str(s.get("n") or 0)] = round(shot_render_secs(s, policy, h3_cost=h3_cost))
    return out


def ensure_trigger(prompt: str, trigger: str) -> str:
    """Guarantee the character trigger is present, mechanically.

    The panel already learned this the hard way — character jobs are submitted with
    `enhance=false` under the comment "CRITICAL: don't let Gemma strip the trigger", because
    a rewritten prompt that loses the token renders a stranger's face and reads as a model
    bug. So we never rely on the planner to keep it: we put it there in Python.

    Idempotent, and word-boundary aware so "bizarrotrn" is not considered present merely
    because "bizarrotrnx" appears.
    """
    p = (prompt or "").strip()
    t = (trigger or "").strip()
    if not t:
        return p
    if re.search(rf"\b{re.escape(t)}\b", p):
        return p
    return f"{t} {p}" if p else t


# =============================================================================
# THE SPEECH LAW, AT BOARD LEVEL — no mouth moves without words to say
# =============================================================================
# `storyboard_planner` has enforced this on shots IT writes since the owner
# first reported "talking gibberish": a prompt that says a man is speaking and
# gives him nothing to say leaves the audio branch babbling. The law was
# correct and it was in the wrong place — it ran inside the planner, so a shot
# authored any other way (by hand, by the add-shot route, by an importer) sailed
# past it. The owner reported gibberish a SECOND time on a hand-written board,
# which is the same bug arriving through the door the first fix did not cover.
#
# This is the engine-agnostic form, and it deliberately does NOT reuse the
# planner's `_speech_violations`: that one runs on the planner's own draft,
# where a line lives in `<d>[English] ...</d>` and prose quotes are a FORM
# error to be rewrapped. By the time a shot is on a board its prompt is
# finished — H3 keeps the tag, LTX has already been converted to single quotes
# by `_strip_h3_markup` — so at this level the only question is whether WORDS
# ARE PRESENT AT ALL, in either wrapper.
_SPOKEN_WORDS_RE = re.compile(
    r"<d>\s*(?:\[[^\]]*\]\s*)?[^<]*\w+[^<]*</d>"      # H3: the tag, with content
    r"|['‘“\"]\s*[^'’”\"]*\w+\s+[^'’”\"]*['’”\"]",
    re.DOTALL)                                        # LTX: a quoted phrase, 2+ words

# Deliberately narrower than the planner's list: this one BLOCKS A RENDER, so
# it only carries verbs that unambiguously mean a mouth is producing speech.
# `brief` needs an object — a "briefing room" is a room, which is a false
# positive the planner's version already paid for once.
_IMPLIES_SPEECH_RE = re.compile(
    r"\b(?:explain(?:s|ing)?|describ(?:es|ing) (?:to|the situation)|discuss(?:es|ing)|"
    r"talk(?:s|ing)|speak(?:s|ing)|says?|saying|tell(?:s|ing)|address(?:es|ing)"
    r"|announc(?:es|ing)|declar(?:es|ing)|recit(?:es|ing)|narrat(?:es|ing)"
    r"|murmur(?:s|ing)|mutter(?:s|ing)|whisper(?:s|ing)|mumbl(?:es|ing)"
    r"|ask(?:s|ing)|answer(?:s|ing)|repl(?:y|ies|ying)|shout(?:s|ing)"
    r"|order(?:s|ing) (?:him|her|them|the)|instruct(?:s|ing))\b"
    # SUNG is a mouth producing words just as much as SPOKEN is — a "he sings"
    # shot with no line renders the same babbling jaw a "he says" shot does.
    # The lookbehinds keep birdsong out: "birds singing" is scenery, not a
    # mouth this law owns (same caution as `brief` needing an object above).
    r"|(?<!birds )(?<!bird )\b(?:sing(?:s|ing)|chant(?:s|ing))\b"
    r"|\b(?:his|her|their|its) voice\b"
    r"|\bin a (?:low|quiet|soft|hushed|loud|steady|calm|firm|gravelly|deep) voice\b"
    r"|\bmid-sentence\b|\bmid-speech\b", re.IGNORECASE)


def shot_speech_problem(prompt: str) -> str | None:
    """The speech verb in `prompt` that has no words to go with it, or None.

    A shot is SPOKEN (the words are in it) or SILENT (nothing implies a mouth
    moving). Implying speech and providing none is the one combination that is
    always wrong, because the model renders a talking head with nothing to say.
    """
    p = prompt or ""
    if _SPOKEN_WORDS_RE.search(p):
        return None
    m = _IMPLIES_SPEECH_RE.search(p)
    return m.group(0) if m else None


# --- dialogue PACING: a line must fit its shot, and it must CLOSE -----------
# The owner hit the same truncation twice in one day: a line sized to the idea
# instead of to the clock renders as a sentence that stops mid-word when the
# shot ends. His ruling, verbatim: "when people talk, there is a structure to
# the talk... a beginning and an end. It's not just open-ended sentences that
# get cut."
#
# The budget is measured, not guessed, and it BRACKETS the evidence: a 7-word
# line in a 4.04s shot delivered fine (needs the budget to allow >= 2.31 w/s
# after settle), and a 20-word line in a 7.04s shot was cut mid-phrase (needs
# it to refuse >= 3.31). 2.4 sits between the two with margin on both sides.
SPEECH_WORDS_PER_SEC = 2.4
# A WARM READ IS SLOWER, and the evidence forced this split: a 9-word line
# delivered "quietly, almost under her breath" truncated in a 5.04s shot that
# the 2.4 budget approves, and a 23-word slow read truncated at 13.04s. Both
# land near 1.7 w/s. One constant cannot bracket a game-show host and a lazy
# half-smile; the descriptor in front of the line is the tempo marking.
SPEECH_WORDS_PER_SEC_SLOW = 1.7
_SLOW_READ_RE = re.compile(
    r"\bsays? (?:[a-z]+[ ,]+)*?(?:slowly|softly|quietly|low\b|lazily)"
    r"|\bunder (?:his|her|their) breath\b|\blazy half-smile\b"
    r"|\bdrawls?\b|\bwhispers?\b",
    re.IGNORECASE)
# SUNG delivery is slower than speech but not half-tempo — a lyric stretches
# over the beat. Measured on the AVRELIVS "Amor fati" joint takes (2026-08-30):
# the owner-graded 121-frame take closed 6 sung words in 5.04 s, which needs
# the budget to allow >= 1.49 w/s after settle, and the 241-frame take carries
# 12 words in 10.04 s (>= 1.33). 1.5 admits both delivered cases; there is no
# truncated-sung case yet to bracket the refusal side, so the constant hugs
# the delivered evidence rather than guessing a ceiling. The 2.4 speaking
# budget would approve 21 words in the 10 s shot — nearly double what a sung
# mantra actually carries — which is why SUNG needs its own rate at all.
SPEECH_WORDS_PER_SEC_SUNG = 1.5
_SUNG_READ_RE = re.compile(
    r"(?<!birds )(?<!bird )\b(?:sing(?:s|ing)|chant(?:s|ing)|sung)\b",
    re.IGNORECASE)
SPEECH_SETTLE_S = 1.0


def is_slow_read(prompt: str) -> bool:
    """True when the voice descriptor asks for slow delivery."""
    return bool(_SLOW_READ_RE.search(prompt or ""))


def is_sung(prompt: str) -> bool:
    """True when the line is sung or chanted rather than spoken."""
    return bool(_SUNG_READ_RE.search(prompt or ""))

# Spoken spans, with their words, in both dialects. The single-quote form must
# survive apostrophes: a quote FLANKED BY LETTERS is punctuation inside a word
# ("There's"), not the end of the line — without that rule the counter stops
# counting at the first contraction.
_D_SPAN_RE = re.compile(r"<d>\s*(?:\[[^\]]*\]\s*)?(.*?)</d>", re.DOTALL)
_Q_SPAN_RE = re.compile(
    r"(?<![A-Za-z])'((?:[^']|(?<=[A-Za-z])'(?=[A-Za-z]))+?)'(?![A-Za-z])",
    re.DOTALL)


def spoken_spans(prompt: str) -> list[str]:
    """Every spoken line in `prompt`, tag form and quote form both."""
    p = prompt or ""
    out = [m.group(1).strip() for m in _D_SPAN_RE.finditer(p)]
    stripped = _D_SPAN_RE.sub(" ", p)
    out += [m.group(1).strip() for m in _Q_SPAN_RE.finditer(stripped)]
    return [s for s in out if s]


def speech_fit_frames(word_count: int, fps: int = 24, slow: bool = False,
                      sung: bool = False) -> int:
    """The smallest legal frame count (fps*n+1) that fits `word_count` words."""
    rate = (SPEECH_WORDS_PER_SEC_SUNG if sung
            else SPEECH_WORDS_PER_SEC_SLOW if slow
            else SPEECH_WORDS_PER_SEC)
    need_s = word_count / rate + SPEECH_SETTLE_S
    n = max(1, int((need_s * fps + fps - 1) // fps))
    return fps * n + 1


def shot_pacing_problem(prompt: str, duration_s: float) -> str | None:
    """Why this shot's dialogue will not survive its duration, or None.

    Two failure shapes, both of which render as a cut-off sentence:
      * OVERSTUFFED — more words than the clock can carry;
      * UNFINISHED  — the last line ends on a comma/dash/ellipsis/nothing, so
        even at a fitting length it sounds like the start of a thought.
    """
    spans = spoken_spans(prompt)
    if not spans:
        return None
    # Punctuation-only tokens are not words — an em-dash between clauses was
    # counted once and pushed a delivered-fine line over the budget.
    words = sum(1 for sp in spans for t in sp.split() if any(c.isalnum() for c in t))
    try:
        dur = float(duration_s or 0)
    except (TypeError, ValueError):
        dur = 0.0
    if dur > 0:
        sung = is_sung(prompt)
        slow = is_slow_read(prompt)
        rate = (SPEECH_WORDS_PER_SEC_SUNG if sung
                else SPEECH_WORDS_PER_SEC_SLOW if slow
                else SPEECH_WORDS_PER_SEC)
        allowed = max(0.0, (dur - SPEECH_SETTLE_S) * rate)
        if words > allowed:
            pace = ("at singing tempo" if sung
                    else "at this SLOW delivery" if slow
                    else "at speaking pace")
            return (f"{words} spoken words in a {dur:.1f}s shot — {pace} only "
                    f"~{int(allowed)} fit, so the line is cut off "
                    f"mid-sentence. Shorten the line, or lengthen the shot to "
                    f"at least {speech_fit_frames(words, slow=slow, sung=sung)} frames")
    last = spans[-1].rstrip()
    if last and last[-1] not in ".!?":
        return (f"the line ends {last[-12:]!r} — no full stop, so it plays as "
                f"a sentence that never finishes. Close the thought")
    return None


# =============================================================================
# LOCATIONS — the same room in every shot that claims to be in it
# =============================================================================
# A text-to-video model re-invents everything the prompt does not pin. Four
# shots that all said "dim room, cinematic close-up" came back as four rooms:
# a monitor-lit study, a brighter office with papers on the wall, a VINTAGE
# PARLOUR with no monitors at all, and a near-black void — with the collar
# changing between them. Nobody wrote a contradiction; the shots simply never
# agreed on anything, and unstated means re-rolled.
#
# So a location is a board-level ENTITY, exactly like a cast member, and a shot
# references it by id. The description is written once and injected into every
# shot that claims that location, which is the only way "the same room" can
# survive shots being re-rendered one at a time months apart.
LOCATION_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")
LOCATION_DESC_MAX = 600

# ---- VIEWS: a location is a place seen from ANGLES ---------------------------
# One description per location is one CAMERA POSITION pretending to be a place.
# Measured on the car-wash day: the establishing description ("a soapy blue
# sedan on the driveway") actively FIGHTS the reverse shot — the moment he
# turns to talk to her, the car is behind the camera and must not be in the
# prompt at all, the houses across the street must be, and the low sun that
# raked in from camera left now rakes in from camera RIGHT. That was hand-built
# as a second location (`carwash_reverse`); a view is that, first-class.
#
# So a location carries named VIEWS. Each view is a self-contained description
# of what the camera sees FACING that direction, light side included, and a
# shot picks `location_id` + `view`. A location with no views behaves exactly
# as it always has: its description is the only view there is.
VIEW_ID_RE = LOCATION_ID_RE

# Where the subject is looking, as a fact about the FRAME rather than about the
# room: off past the left edge, off past the right edge, or down the lens.
# Two shots that cut between two people must not claim the same one — that is
# the 180-degree line, and it is why this is a vocabulary and not free text.
EYELINES = ("left", "right", "lens")


def new_view(view_id: str, name: str, description: str = "") -> dict:
    return {"id": str(view_id).strip().lower(),
            "name": str(name).strip(),
            "description": str(description).strip()}


def new_location(loc_id: str, name: str, description: str = "",
                 views: list[dict] | None = None) -> dict:
    loc = {"id": str(loc_id).strip().lower(),
           "name": str(name).strip(),
           "description": str(description).strip()}
    # ABSENT, not empty, when there are none. Every board written before views
    # existed has no `views` key, and an empty list would be a new shape for
    # the panel and the validator to have opinions about for no gain.
    if views:
        loc["views"] = list(views)
    return loc


def board_locations(board: dict) -> dict[str, dict]:
    """`{id: location}` for a board, skipping anything malformed."""
    out: dict[str, dict] = {}
    for loc in (board.get("locations") or []):
        if isinstance(loc, dict) and str(loc.get("id") or "").strip():
            out[str(loc["id"]).strip().lower()] = loc
    return out


def merge_location_views(board_locs: list[dict] | None,
                         plan_locs: list[dict] | None) -> list[dict]:
    """The board's locations, wearing the views the planner derived.

    PATCH, NEVER OVERWRITE — the rule locations already cost this project once,
    when a re-plan assigned the board's `locations` wholesale and erased what
    the user had typed. The board's row stays authoritative for id, name and
    description: those are the user's own words out of the Locations box. Only
    `views` come from the plan, and only when the plan actually derived some —
    a re-plan with the geography pass off, or one whose floor plan came back as
    chatter, must not strip the views the last plan produced, because the shots
    on the board still name them and an unknown view id is a hard error.

    A location the plan knows and the board does not is appended rather than
    dropped: its shots point at it, and a board missing it fails validation on
    every one of them.
    """
    out: list[dict] = []
    plan_by_id = {str(l.get("id") or "").strip().lower(): l
                  for l in (plan_locs or []) if isinstance(l, dict) and l.get("id")}
    seen: set[str] = set()
    for loc in (board_locs or []):
        if not isinstance(loc, dict) or not str(loc.get("id") or "").strip():
            continue
        lid = str(loc["id"]).strip().lower()
        seen.add(lid)
        row = dict(loc)
        views = (plan_by_id.get(lid) or {}).get("views")
        if views:
            row["views"] = [dict(v) for v in views if isinstance(v, dict)]
        out.append(row)
    for lid, loc in plan_by_id.items():
        if lid not in seen:
            out.append(dict(loc))
    return out


def location_views(loc: dict | None) -> dict[str, dict]:
    """`{id: view}` for one location, skipping anything malformed."""
    out: dict[str, dict] = {}
    for v in ((loc or {}).get("views") or []):
        if isinstance(v, dict) and str(v.get("id") or "").strip():
            out[str(v["id"]).strip().lower()] = v
    return out


def shot_view(shot: dict, locations: dict[str, dict] | None = None) -> dict | None:
    """The view a shot names, or None when it names none / names one that is gone."""
    vid = str((shot or {}).get("view") or "").strip().lower()
    if not vid:
        return None
    loc = (locations or {}).get(str((shot or {}).get("location_id") or "").strip().lower())
    return location_views(loc).get(vid)


def shot_scene_text(shot: dict, locations: dict[str, dict] | None = None) -> str:
    """What the camera sees: the VIEW's description, else the location's.

    The fallback is the whole back-compat story in one line — a board with no
    views, or a shot that names no view, injects exactly what it always did.
    """
    view = shot_view(shot, locations)
    if view is not None:
        desc = str(view.get("description") or "").strip()
        if desc:
            return desc
    loc = (locations or {}).get(str((shot or {}).get("location_id") or "").strip().lower())
    return str((loc or {}).get("description") or "").strip()


def eyeline_clause(eyeline: str | None, pronoun: str = "") -> str:
    """"his eyes fixed past the right edge of frame" — or "" for lens/none.

    `lens` deliberately produces NOTHING. Looking down the barrel is what these
    models do unprompted for a piece to camera, and a sentence telling them to
    do it buys a stiffer performance than saying nothing. The value still
    exists because a shot has to be able to SAY it holds the lens — that is
    what makes the 180-degree check able to tell "faces the camera" apart from
    "nobody wrote an eyeline".
    """
    side = str(eyeline or "").strip().lower()
    if side not in ("left", "right"):
        return ""
    poss = {"he": "his", "she": "her", "they": "their",
            "him": "his", "her": "her"}.get(str(pronoun or "").strip().lower(), "")
    return "%seyes fixed past the %s edge of frame" % (poss + " " if poss else "", side)


def eyeline_complement(eyeline: str | None) -> str:
    """The other side of the line. `lens` and nothing have no complement."""
    return {"left": "right", "right": "left"}.get(str(eyeline or "").strip().lower(), "")


def board_wardrobe(board: dict) -> dict[str, str]:
    """`{character_id: what they are wearing in THIS film}`.

    The same failure locations fixed, one axis over. Four shots of one man came
    back in a navy suit, a navy suit, a period collar and a different period
    collar, because "a man in a dark suit" is a different suit every time it is
    re-rolled. Writing the outfit ONCE and attaching it to the character is the
    only version of this that holds across shots rendered separately — and it
    is what a hand-written board makes you do by copy-paste, which is exactly
    the copy-paste that drifts.
    """
    out: dict[str, str] = {}
    for row in (board.get("cast") or []):
        if isinstance(row, dict) and row.get("id"):
            w = str(row.get("wardrobe") or "").strip()
            if w:
                out[str(row["id"]).strip()] = w
    return out


def compose_shot_prompt(shot: dict, locations: dict[str, dict] | None = None,
                        wardrobe: dict[str, str] | None = None) -> str:
    """The prompt that is actually rendered: subject, action, FRAME, PLACE.

    Order is deliberate and matches how these models read a prompt — the
    subject and what it is doing first, then how it is framed, then where it
    is. Putting the location first buries the action in scenery. The EYELINE
    rides with the framing rather than with the room, because where a person
    looks is a fact about the frame, not about the scenery behind them.

    Every addition is appended rather than merged into `shot["prompt"]`, so
    the shot keeps the sentence a human wrote and editing a location — or
    flipping an eyeline to fix the 180-degree line — re-flows every shot that
    uses it without rewriting anybody's text.
    """
    trigger = (shot.get("trigger") or shot.get("character_id") or "").strip()
    parts = [ensure_trigger(shot.get("prompt") or "", trigger)]

    # WARDROBE goes right after the person, where a costume note belongs, and
    # only if the shot has not already said it — a board that spells the outfit
    # out per shot must not get it twice.
    cid = str(shot.get("character_id") or "").strip()
    outfit = (wardrobe or {}).get(cid, "").strip().rstrip(",")
    if outfit and outfit.lower() not in parts[0].lower():
        parts.append(outfit)

    framing = str(shot.get("framing") or "").strip().rstrip(",")
    if framing:
        parts.append(framing)

    eyes = eyeline_clause(shot.get("eyeline"), shot.get("pronoun") or "")
    if eyes:
        parts.append(eyes)

    # The VIEW, if the shot names one, otherwise the location as before.
    scene = shot_scene_text(shot, locations).rstrip(",")
    if scene:
        parts.append(scene)

    return ", ".join(p for p in parts if p)


# The pass's PIPELINE quality -> the CHARACTER quality token make_job accepts.
# Chosen by canvas and pipeline, not by name: quick is the small distilled
# draft (704x384), balanced/standard are the graded 1024x576 distilled recipe
# ("pro" — the one that holds identity best), and the two High tiers keep their
# own two-stage pipeline. Anything unrecognised lands on "pro", which is the
# recipe the character work is actually validated at.
CHARACTER_QUALITY_FOR_PASS: dict[str, str] = {
    "quick": "draft",
    "balanced": "pro",
    "standard": "pro",
    "high": "high",
    "high_720p": "high720",
}


# ---- ONE TAKE on the board --------------------------------------------------
# A film that is one shot: the planner writes one movement per five-second
# beat (as it does for a soundtrack slot), and the board keeps ONE shot that
# carries the beats. shot_to_job turns it into the panel's take fields, so the
# render is the same take a person gets from the Video tab.
TAKE_BEAT_SECONDS = 5
TAKE_SECONDS = (30, 45, 60, 90, 120)


def collapse_take(shots: list[dict], seconds: int) -> list[dict]:
    """The planner's N beat-shots become ONE take shot: the first shot's
    identity (mode, character, refs, title) with `beats` = every shot's own
    prompt, padded to the take's beat count, `duration_s` = the take."""
    beats_n = max(1, int(seconds) // TAKE_BEAT_SECONDS)
    items = [s for s in (shots or []) if isinstance(s, dict)]
    if not items:
        return []
    first = dict(items[0])
    beats = [str(s.get("prompt") or "").strip() for s in items][:beats_n]
    beats += [""] * (beats_n - len(beats))
    first.update({
        "n": 1,
        "prompt": beats[0] or first.get("prompt") or "",
        "beats": beats,
        "take_seconds": int(seconds),
        "duration_s": float(seconds),
        "frames": int(seconds) * 24 + 1,
        "title": first.get("title") or "One take",
    })
    for k in ("slot", "section"):
        first.pop(k, None)
    return [first]


def shot_wants_still(shot: dict) -> bool:
    """A shot an anchor still can start: text and character shots. Keyframe,
    extend and a2v already start from media of their own."""
    return isinstance(shot, dict) and str(shot.get("mode") or "text") in ("text", "character")


def still_prompt(prompt: str) -> str:
    """The shot prompt, re-aimed at ONE frame: camera moves and durations
    describe a clip, a still needs the composition at the first instant."""
    move = r"(push(es|ing)?( in)?|pull(s|ing)?( back| out)?|pan(s|ning)?|tilt(s|ing)?|track(s|ing)?|dolly(ing)?|orbit(s|ing)?|zoom(s|ing)?|crane(s)?|glide(s)?|drift(s)?|sweep(s)?|circle(s)?)"
    # a clause that is about the camera moving, from its start to the next punctuation
    p = re.sub(r"(,\s*)?(as\s+)?(the\s+|a\s+)?(camera|lens|shot|frame)\s+[^.,;]*?\b" + move + r"\b[^.,;]*", "", prompt, flags=re.I)
    p = re.sub(r"(,\s*)?(slow(ly)?|smooth(ly)?|gentl[ey]|fast|quick(ly)?)?\s*\b" + move + r"\b\s+(in|out|up|down|left|right|across|around|forward|back)?[^.,;]*", "", p, flags=re.I)
    p = re.sub(r"\s+([.,;])", r"\1", p)
    p = re.sub(r"([.,;])\1+", r"\1", p)
    p = re.sub(r",\s*\.", ".", p)
    p = re.sub(r"\s{2,}", " ", p).strip(" ,;.")
    return (p + ". The first frame of the shot, held still: the composition, framing and light exactly as the shot opens.").strip()


def shot_to_job(shot: dict, policy_pass: dict, *,
                board_id: str = "", board_title: str = "",
                h3_available: bool = True,
                engine_mode: str = DEFAULT_ENGINE_MODE,
                h3_chain_prompts: bool = False,
                long_windows: bool = False,
                style: str = "",
                locations: dict[str, dict] | None = None,
                wardrobe: dict[str, str] | None = None) -> dict:
    """Translate one storyboard shot into the panel's ORDINARY job form fields.

    Deliberately produces the same shape a human clicking Generate would produce, so shots
    flow through `/queue/add` -> `make_job` -> the normal worker, land in `mlx_outputs/`, and
    show up in the usual gallery with a usual sidecar. No private execution path.

    Every key below is in `make_job`'s allowlist. That is not a style note: a form field
    make_job does not name is silently dropped on /queue/add — the known trap in this codebase —
    so a control can look perfectly wired and do nothing at all.

    NOTE `enhance: "off"` is not optional — see ensure_trigger() above.
    """
    # compose_shot_prompt, not shot["prompt"] — the framing and the location
    # have to reach the model, and this is the one place every shot passes
    # through on its way to a render. Doing it at the call sites instead would
    # mean the estimate, the re-render and the gap-fill each getting their own
    # chance to forget.
    prompt = compose_shot_prompt(shot, locations, wardrobe)

    # The engine. Without this the job dict has no `engine` key, make_job falls back to
    # ENGINE_DEFAULT ("ltx"), and every H3 shot the planner wrote renders silently on a
    # different model — different aspect, no audio, none of it flagged anywhere.
    # resolve_engine() is the single place that decides, so the chip on the card, the estimate,
    # the scheduling bucket and this job dict can never disagree about what will run.
    engine = resolve_engine(shot, engine_mode=engine_mode, h3_available=h3_available)

    job = {
        # The panel has ONE backend mode for text and character alike; see _PANEL_MODE.
        "mode": _PANEL_MODE.get(shot.get("mode"), "t2v"),
        "engine": engine,
        "prompt": prompt,
        "quality": policy_pass.get("quality", "balanced"),
        "width": policy_pass.get("width"),
        "height": policy_pass.get("height"),
        "frames": policy_pass.get("frames"),
        "enhance": "off",            # never let Gemma touch a planned prompt
        # `auto_open` was never in make_job's allowlist, so it silently did nothing. The field
        # that actually exists is `open_when_done`, and it must be off: a 12-shot overnight run
        # must not pop a QuickTime window per shot.
        "open_when_done": "off",
    }

    # Duration. Without this every shot rendered at the pass's frame count, so the per-shot
    # length control was decoration. Each engine gets its OWN grid — LTX snaps to frames%8==1
    # at 24 fps, H3 renders whole cells off the 17n+5 grid and picks them by length key.
    duration_s = float(shot.get("duration_s") or 0)
    if engine == "h3":
        length = h3_length_for(duration_s) if duration_s > 0 else "5s"
        job["h3_length"] = length
        job["h3_quality"] = h3_quality_for(policy_pass.get("quality", "balanced"))
        # Per-window prompts, when the installed runner supports the flag. The
        # caller passes that capability in rather than this module probing for
        # it — storyboard.py stays model-free and panel-free.
        if h3_chain_prompts:
            chain = h3_chain_prompts_for(shot)
            if chain:
                job["h3_chain_prompts"] = json.dumps(chain)
        # make_job re-stamps geometry from the resolved cell; carrying the cell's own frame
        # count means the job dict is already self-consistent (and honest in a log or a test)
        # before it gets there.
        job["frames"] = _H3_LENGTH_FRAMES[length]
    elif duration_s > 0:
        job["frames"] = ltx_frames_for(duration_s)

    if shot.get("character_id"):
        job["character_id"] = shot["character_id"]
        # A CHARACTER JOB SPEAKS A DIFFERENT QUALITY VOCABULARY, and make_job
        # REFUSES the pipeline one. `resolve_character_quality()` accepts only
        # draft / pro / high / high720; a job carrying `quality="quick"` (the
        # default draft pass) or `"standard"` (the default delivery pass)
        # raises CharacterRequestError, which `_sb_render_thread` catches and
        # turns into a failed shot. Every cast shot in every film, both passes,
        # with the message "character quality must be draft, pro, high or
        # high720" as the only clue. Translating here is what makes the pass
        # policy and the character surfaces speak to the same make_job.
        #
        # The panel's `resolve_character_quality` is the source of truth for
        # what these keys MEAN; this table only says which of them a pass maps
        # onto. `test_storyboard_editor_api` pins the pair together so the two
        # cannot drift apart in silence.
        job["quality"] = CHARACTER_QUALITY_FOR_PASS.get(
            str(policy_pass.get("quality", "balanced")).strip().lower(), "pro")
        # THE VOICE LOADS ONLY WHEN THERE ARE LINES TO SAY.
        #
        # Stacking a character's audio LoRA on a shot with no speech spends the
        # audio branch on nothing and, on some prompts, invites gibberish. The
        # panel already has the manual escape hatch (the "No voice" pill); this
        # makes the DEFAULT right in the one place it can be known EXACTLY
        # rather than guessed.
        #
        # And it is exact here: the planner's H3 dialect carries dialogue in
        # explicit <d>…</d> tags, so this is a derivation, not a heuristic. A
        # shot with a <d> tag that has any content speaks; one without does not.
        # (The Manual tab cannot know this and does not pretend to — it sets a
        # default and says it did.)
        # ENGINE-AGNOSTIC, and it was not. This read `_HAS_DIALOGUE`, which
        # matches `<d>…</d>` and nothing else, under a comment claiming "it is
        # exact here… a derivation, not a heuristic". That is true of H3 and
        # FALSE of LTX: `_strip_h3_markup` has already turned the tag into
        # 'single quotes' by the time an LTX prompt exists, so the tag is never
        # present, `no_voice` was ALWAYS "on", and the trained voice LoRA was
        # stripped from EVERY LTX character shot that had a line to say.
        # The owner heard it as the characters never using their own voices,
        # reported it repeatedly, and it survived because the comment sounded
        # authoritative and was only checked against the engine it was true for.
        #
        # `_SPOKEN_WORDS_RE` is the same detector `shot_speech_problem` uses.
        # That is deliberate: one asks "are there words?" to decide whether the
        # shot is honest, the other to decide whether the voice loads. If those
        # two ever disagree you get one of the two bugs — a mouth with nothing
        # to say, or a line delivered by a stranger.
        job["no_voice"] = "off" if _SPOKEN_WORDS_RE.search(prompt) else "on"
    if shot.get("seed") is not None:
        job["seed"] = shot["seed"]

    # Queue linkage. `preset_label` is what the queue card, the Now card, the job pill and the
    # Recent row already print (`j.params.label || snippet(prompt)`), so a storyboard job
    # identifies itself in the bottom pane with no bottom-pane code at all. `session_tag`
    # carries the provenance the badge and the gallery group on. Both keys are already in
    # make_job's allowlist — no allowlist edit, which is the point.
    # ONE TAKE. The shot carries `take_seconds` + `beats`; the panel's make_job
    # does the rest (LTX: the windows chain; H3: 15 s parts that continue).
    if shot.get("take_seconds") in TAKE_SECONDS:
        job["take_seconds"] = str(int(shot["take_seconds"]))
        job["beats"] = json.dumps([str(b or "") for b in (shot.get("beats") or [])])
        if engine != "h3":
            job["frames"] = int(shot["take_seconds"]) * 24 + 1
    # ANCHOR STILL. When the shot has a still, the video STARTS from it: the
    # ordinary i2v path with the still as the image, anchored (the strict
    # pin, not Inspire). LTX only — H3's i2v is a different contract.
    if shot.get("still") and engine != "h3" and job["mode"] == "t2v":
        job["mode"] = "i2v"
        job["image"] = str(shot["still"])
        job["i2v_reference_mode"] = "anchor"
    # LONG WINDOWS. A shot longer than one LTX window (121 frames) renders as
    # a chain of windows on the Q8 dev transformer rather than being cut to
    # fit. One prompt per window: the shot's own prompt, with the board's
    # style and the shot's location as the invariants every window repeats.
    # Extend cannot re-inject an image per window, so an anchored long shot is
    # anchored by its first window only — the still opens it, the chain
    # carries it. See ltx_windows.py.
    if long_windows and engine != "h3" and int(job.get("frames") or 0) > 121 \
            and job["mode"] in ("t2v", "i2v"):
        job["temporal_mode"] = "windows"
        inv = [s for s in (style.strip(), str((shot.get("location") or "")).strip()) if s]
        if inv:
            job["window_invariants"] = "; ".join(inv)
    n = shot.get("n")
    if board_id and isinstance(n, int):
        job["session_tag"] = f"sb:{board_id}#{n}"
    if isinstance(n, int):
        title = (board_title or "").strip()
        job["preset_label"] = f"S{n:02d} · {title}" if title else f"S{n:02d}"

    # `shot["refs"]` is DELIBERATELY not mapped. The four ref-based modes (remix / keyframe /
    # extend / a2v) each want a different field — image, start_image + end_image,
    # video_path, ingredient_images_json — and mapping them half-way would enqueue a job that
    # renders silent t2v from a prompt while looking like it used the reference, which reads
    # as a model bug rather than a missing feature. v1 plans `text` + `character` only and the
    # planner is constrained to the same two, so a board that carries refs is hand-written or
    # from a future version; the validator still checks them and the schema still keeps them.
    return {k: v for k, v in job.items() if v is not None}


# THE ONE POLICY LITERAL. storyboard_planner.default_policy() had its own copy and
# they had drifted: this said Draft 640x448 (what a Quick render actually delivers,
# ffprobe-verified) while the planner said 640x480, a canvas the panel's own engine
# registry lists as never delivered. The main path masked it by keeping the board's
# existing policy, so only a direct planner consumer would ever have seen the
# fictional geometry. One literal, imported by the planner.
DEFAULT_POLICY: dict = {
    "draft": {"quality": "quick", "width": 640, "height": 448, "frames": 49},
    "final": {"quality": "balanced", "width": 1024, "height": 576, "frames": 121},
}


def default_policy() -> dict:
    """A fresh copy of DEFAULT_POLICY — callers mutate it (the planner clamps it)."""
    return {k: dict(v) for k, v in DEFAULT_POLICY.items()}


def fit_canvas(width: int, height: int, cap: int) -> tuple[int, int]:
    """The largest 8-aligned canvas with this aspect that a `cap`px Mac may render.

    THE ONE CLAMP FORMULA. The panel's policy builder, the per-quality canvas
    table the Quality chips are labelled from, and the over-cap error's own
    "use this instead" offer must all land on the SAME numbers — otherwise the
    button offers a size the validator then rejects, which is exactly the loop
    a 24 GB Mac was stuck in (offered 1024x576 against a 768px cap, and the
    guard around the write meant clicking it did nothing at all).
    """
    w, h = int(width), int(height)
    if cap <= 0 or max(w, h) <= cap:
        return w, h
    scale = cap / float(max(w, h))
    return max(64, int(w * scale) // 8 * 8), max(64, int(h * scale) // 8 * 8)


def new_storyboard(board_id: str, title: str, *, shots: list[dict] | None = None,
             cast: list[dict] | None = None, policy: dict | None = None,
             locations: list[dict] | None = None) -> dict:
    """Build an empty, schema-correct storyboard. Kept here so the planner, the tests and any
    future importer all produce the identical shape."""
    return {
        "schema": SCHEMA_VERSION,
        "id": board_id,
        "title": title,
        "created_at": int(time.time()),
        "cast": cast or [],
        "locations": locations or [],
        "policy": policy or default_policy(),
        "shots": shots or [],
    }
