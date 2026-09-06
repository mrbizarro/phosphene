#!/usr/bin/env python3
"""Storyboard planner — concept in, valid film spec out.

Companion to `storyboard.py`. That module owns the SCHEMA, the VALIDATOR, the durable
state and the scheduler. This module owns the one thing it deliberately left out: turning
a sentence a human typed into a `shots` list that `validate_storyboard()` accepts on the
first try.

    storyboard.py        schema + validate + shooting order + job translation   (no models)
    storyboard_planner.py  concept -> spec, using a small local LLM             (this file)
    mlx_ltx_panel.py       UI + queue + rendering                               (someone else)

Nothing here imports mlx in the panel's process. See MEMORY POLICY below.


MODEL: gemma-3-12b-it-4bit, the weights the panel ALREADY has
-------------------------------------------------------------
`storyboard.py`'s docstring says the planner is Qwen3.5-4B-4bit (decision 4492775, 2026-07-24).
That decision is sound on the merits and stays the target — but it is not shippable on this
machine today and the module must not pretend otherwise:

  * Qwen3.5-4B-4bit is NOT on disk. Nothing named qwen*-instruct/-4bit exists in
    `mlx_models/` or in the HF hub cache; `qwen-edit-2511-q6` is an mflux IMAGE model.
  * The volume has ~6.5 GiB free. A 3.06 GB download is technically survivable and
    strategically stupid, and the owner's standing rule for this task is explicit:
    do not download new models.
  * The 19 GB ollama `heretic-32b` is out of the question for a step that must run
    BEFORE a render on a unified-memory Mac.

So the planner runs on `mlx_models/gemma-3-12b-it-4bit` — the exact weights
`/prompt/enhance` already loads (`mlx_warm_helper.get_gemma_lm()`), through the exact
runtime (`mlx_lm.load` + `mlx_lm.generate` + `make_sampler`, mlx-lm 0.31.1 in
`ltx-2-mlx/env`). Zero new bytes on disk, and the failure mode "planner model missing" is
impossible for anyone who can already enhance a prompt.

The Qwen3.5 story is not lost, it is a one-line switch: point `LTX_STORYBOARD_PLANNER` (or
the `model_path=` argument) at any mlx-lm-loadable directory and everything else here is
unchanged. `storyboard.py`'s three-layer validity argument (strip preamble, coerce, then
validate with one repair retry) is implemented here verbatim and is model-agnostic; it is
what makes a 4-bit 12B good enough for the job.


MEMORY POLICY — why a SUBPROCESS and not the warm helper
---------------------------------------------------------
The hard rule from the owner: planning must not clog RAM, must be fully released before
a single frame renders, and must never be resident concurrently with a pipeline.

Three candidate paths were considered. The chosen one is the least-RAM path:

1. REJECTED — call the running warm helper's `enhance_prompt` action.
   It is the only text-LLM route the daemon exposes, and it is hard-wired for a different
   job: `max_new_tokens` is the library default 512 (a 6-shot plan is 1200-2500 tokens, so
   every plan would be truncated mid-JSON), the panel handler force-prepends the Lightricks
   T2V system prompt plus the Phosphene enhance addendum to whatever system prompt you
   supply, and it post-edits the OUTPUT by splicing character triggers back in. Making it
   carry a plan means editing `mlx_warm_helper.py` and `mlx_ltx_panel.py` — both owned by
   other agents on this task. Worse for RAM anyway: the helper holds Gemma warm
   indefinitely after the call (only `release_pipelines()` on the next render frees it), so
   ~7.9 GB sits resident through the user's review of the plan.

2. REJECTED — `mlx_lm.load()` inside the panel process.
   The panel is a long-lived server (it is running on Python 3.9 today). MLX allocates from
   a Metal buffer cache that is not fully returned to the OS when the Python objects are
   dropped, so "unloaded" would still show up as a permanently fatter panel. There is no
   honest `release()` you can write for that.

3. CHOSEN — a short-lived subprocess that loads, generates, and dies.
   `release()` reaps the child; process exit is the only 100%-deterministic reclaim MLX
   offers. Peak RSS is measured, not assumed — the child reports its own
   `ru_maxrss` and `mx.get_peak_memory()`, and the parent independently reads
   `RUSAGE_CHILDREN.ru_maxrss`, so the number in the report needs no cooperation to trust.
   `plan_film()` releases in a `finally:`; there is no code path that returns a plan with a
   model still loaded. Measured: ~7.9 GB peak, gone the instant the call returns.

The child is kept alive ACROSS the repair round-trip inside one `plan_film()` call, because
paying the load twice for one plan is waste, not discipline. It never survives the call.


OUTPUT DIALECT
--------------
Per-shot prompts are emitted in the dialect of the engine that will render them:

  * `engine: "h3"`  (default) — MiniMax H3's official three-field form:
        integrated_multimodal_description: [Shot 1] ...
        overall_soundscape: ...
        non_diegetic_music: ...
    with the laws that were paid for in render hours: the camera is always pinned,
    every action completes and then holds, faces never turn, dialogue is wrapped in
    `<d>[English] ...</d>` with the mouth explicitly told to stop, and
    `non_diegetic_music` carries instrumentation/tempo/dynamics rather than mood words.
    Sources: ~/AI/projects/hailuo-mlx/notes/H3_PROMPTING_GUIDE.md and the graded AURELIUS
    round-2 shot table.

  * `engine: "ltx"` — LTX 2.3's single-paragraph prose with a trailing `Audio:` line and a
    verbatim master-style suffix. Chosen automatically for any shot cast with a trained
    Phosphene character, because character LoRAs ARE LTX LoRAs: identity is the one mode
    H3 does not have. That is also what makes trigger injection safe — `ensure_trigger()`
    prepends, which is legal in LTX prose and would be illegal in H3 (a T2VA prompt must
    begin with `integrated_multimodal_description:`).


WHAT THE MODEL IS AND IS NOT ALLOWED TO DECIDE
-----------------------------------------------
Everything the validator checks mechanically is produced mechanically. The model returns a
small flat JSON object carrying only the creative payload: a film title, and per shot a
title, an optional character, a duration, a camera CHOICE from a closed set, a prose
description, a settled end state, a soundscape and a music line. Python owns `schema`,
`id`, `created_at`, shot numbering, `mode`, `engine`, `tier`, `seed`, `refs`, `policy`,
trigger injection and every clamp. A small model cannot fail a check it was never asked
to pass.

Two of the prompting laws are enforced the same way, for the same reason. Stated as rules,
they were ignored: over the first measured sweep (5 concepts, 26 shots) the model reused ONE
camera sentence for every shot of every film, and wrote an end-state clause on 2 shots out
of 26. They are now a one-word choice plus a short phrase, and PYTHON writes the sentence —
after which camera variety and end-state coverage were 100%. If a law can be reduced to a
choice, reduce it; only leave prose to the model where prose is the point.

Pure stdlib. Python 3.9-compatible (the panel runs on 3.9 today) — the 3.11 venv is used
only for the child process, which is the only thing that imports mlx.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent

# --------------------------------------------------------------------------------------
# Model + runtime location
# --------------------------------------------------------------------------------------
# Mirrors mlx_ltx_panel.py:74-75 so a user who moved LTX_MODELS_DIR does not have to move
# anything twice. LTX_STORYBOARD_PLANNER is the escape hatch for pointing at Qwen3.5 (or
# anything else mlx-lm can load) without touching this file.
MODELS_DIR = Path(os.environ.get("LTX_MODELS_DIR", str(ROOT / "mlx_models")))
DEFAULT_MODEL_PATH = Path(
    os.environ.get("LTX_STORYBOARD_PLANNER")
    or os.environ.get("LTX_GEMMA_PATH")
    or (MODELS_DIR / "gemma-3-12b-it-4bit")
)

# The child needs mlx + mlx_lm; the panel's own interpreter may not have them (it is 3.9).
# Same resolution order as the panel's _resolve_helper_python().
_VENV_PY = ROOT / "ltx-2-mlx" / "env" / "bin" / "python3.11"


def _resolve_worker_python() -> Path:
    cand = os.environ.get("LTX_HELPER_PYTHON")
    if cand and Path(cand).is_file():
        return Path(cand)
    if _VENV_PY.is_file():
        return _VENV_PY
    alt = ROOT / "ltx-2-mlx" / "env" / "bin" / "python"
    if alt.is_file():
        return alt
    return Path(sys.executable)


WORKER_PYTHON = _resolve_worker_python()

# Every worker reply is one line prefixed with this, so stray stdout from mlx-lm (progress
# bars, warnings) can never be mistaken for the protocol.
_SENTINEL = "@@PLANNER@@ "

# Modes this planner is allowed to emit. `remix`/`keyframe`/`extend`/`a2v` all require
# inputs the planner does not have (reference images, a prior clip, an audio file), and
# validate_storyboard() rejects them without those — so they are simply not on the menu.
_PLANNABLE_MODES = ("text", "character")

# H3 renders in ~5 s windows (124 frames @ 24 fps); longer clips are chained windows.
# The panel's tier grid offers exactly these lengths, so durations snap to them.
_H3_LENGTHS = (3.0, 5.0, 10.0, 15.0)

_MIN_DURATION = 1.0
_MAX_DURATION = 60.0            # validate_storyboard(): 0 < duration_s <= 60

DEFAULT_TEMPERATURE = 0.15      # low, deliberately: this is structured output, not vibes
DEFAULT_MAX_TOKENS = 3600
DEFAULT_TIMEOUT_S = 900


class PlannerError(Exception):
    """Raised only for programmer errors (bad arguments). Model failures never raise —
    they come back as a structured error dict the UI can render."""


# --------------------------------------------------------------------------------------
# The validator we must satisfy — imported, never reimplemented
# --------------------------------------------------------------------------------------

def _storyboard_module():
    """storyboard.py, or None if it cannot be imported.

    Same import path `_load_validator()` uses, but non-fatal: default_policy()
    is called from contexts that must not raise just because the schema module
    is unavailable, and it carries its own fallback literal.
    """
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        import storyboard  # type: ignore
        return storyboard
    except Exception:
        return None


def _load_validator():
    """Return (validate_fn, storyboard_module).

    The whole design principle of this module is that the schema is whatever
    `storyboard.py` says it is, so we call the real thing rather than modelling it.
    Tolerant about the name because the panel-side agent may expose it as `validate`.
    """
    sys_path_added = False
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
        sys_path_added = True
    try:
        import storyboard  # type: ignore
    except Exception as exc:  # pragma: no cover - only if storyboard.py is broken
        if sys_path_added:
            try:
                sys.path.remove(str(ROOT))
            except ValueError:
                pass
        raise PlannerError("cannot import storyboard.py: %s" % exc)
    fn = getattr(storyboard, "validate", None) or getattr(storyboard, "validate_storyboard", None)
    if fn is None:
        raise PlannerError("storyboard.py exposes neither validate() nor validate_storyboard()")
    return fn, storyboard


# --------------------------------------------------------------------------------------
# THE PROMPT
# --------------------------------------------------------------------------------------
# Everything below is few-shot first and rules second, on purpose. The exemplars are real
# graded work: two are trimmed from the AURELIUS round-2 table (owner-graded KEEP/MAYBE
# shots, ~/AI/projects/aurelius/video/clips/_work/shots_r2.py) and one is C1 from
# H3_PROMPTING_GUIDE.md §6.1. The abstract rules exist only to name what the exemplars
# already demonstrate, so a 12B model can pattern-match instead of reason.

_LAWS = """\
THE LAWS (each one was paid for with a wasted render; the examples above obey all of them)

L1  PIN THE CAMERA. Silence is not a tripod - an unnamed camera drifts. Every description
    names exactly ONE camera behaviour, as prose, with amplitude and speed. Locked shot:
    "The camera holds a static shot, the frame never moves - no pan, no push-in, no
    reframing." Moving shot: "The camera pushes in with small amplitude at slow speed."
L2  FACES NEVER TURN. Turning breaks identity. Heads stay square to the lens, shoulders do
    not pivot, nobody rotates toward or away from camera.
L3  THE ACTION COMPLETES, THEN HOLDS. If the arc is still running at the end of the clip
    the model invents - extra limbs, extra motion. Finish early and name the settled end
    state: "The movement is completely finished before the shot ends, and for the last two
    seconds <state>, with no new movement of any kind."
L4  ONE CONTINUOUS ACTION per 5 seconds for a human subject. Two beats maximum in 5 s, and
    only if the second is a reaction. Abstract/product/graphic subjects may have 3-4.
L5  NO UNANCHORED NEGATIONS. There is no negative prompt. Never write "not blurry", "high
    quality", "no distortion". The only refusals allowed are against things the model adds
    by itself: camera drift (L1) and unwanted lettering ("No text appears at any point.").
L6  DIALOGUE lives inside <d>[English] ...</d> and nowhere else. The speaker, their voice
    and their delivery are described OUTSIDE the tag. Immediately after the tag, stop the
    mouth: "his jaw ceases speaking motion and his mouth settles closed". One or two short
    sentences per 5 s. If a shot has no dialogue, it has no <d> at all.
L7  MUSIC IS INSTRUMENTATION, not mood. "Sparse piano at a slow tempo, joined by low
    strings that swell and cut out" - never "epic", "emotional", "uplifting". "N/A" is the
    correct value when there is no score, and is the right answer most of the time.
L8  SOUNDSCAPE IS AMBIENCE + PHYSICAL SOUND ONLY. Never repeat the dialogue there. 1-3
    sentences.
L9  ON-SCREEN TEXT: only if the brief asks for it, and then type the exact words in double
    quotes, 3-5 words, one row. Otherwise end the description with "No text appears at any
    point."
L10 CONVERT MOOD INTO BEHAVIOUR. Not "she is devastated" but "her eyes shine wet and she
    blinks once and keeps very still".
L11 THE FACE IS THE WHOLE POINT. A shot is judged on whether the face reads. If a person is
    on screen their face is IN the frame, whole, lit and turned to the lens - every time.
    NEVER write, and never imply, any of these unless the brief asked for it in so many
    words: "his face obscured", "seen from behind", "her back to the camera", "head out of
    frame", "cropped at the chin", "silhouetted against the light", "in silhouette", "his
    face hidden by his hands", "we never see her face". The temptation is strongest on the
    last shot of a film, where a silhouette feels like an ending - it is not, it is a shot
    with no face in it. End on the face instead.
    A face may be dark, wet, bruised, half in shadow or lit from one side. It may not be
    turned away, blocked, cut off by the frame edge, or reduced to an outline.
L12 A CAST CHARACTER'S APPEARANCE IS FIXED, AND YOU CANNOT SEE IT. A cast character is a
    trained model of a real, specific face. What they look like is already decided and is
    NOT in this brief - so any physical description you write is a guess, and a guess that
    disagrees with the trained face FIGHTS it and degrades the likeness.
    For a cast character write ONLY: the trigger and name, their ROLE, their WARDROBE,
    their ACTION and their EMOTION. Never their species or creature type, never their
    face, hair, eyes, skin or build, never their age. This includes BODY PARTS that only
    one kind of creature has - no paw, muzzle, snout, fur, tail, whiskers, claws, hooves,
    wings or scales. They have hands.
      WRONG  "bizarrotrn Bizarro, a grizzled badger in a military uniform, leans over ..."
      WRONG  "bizarrotrn Bizarro, a tall bearded man in his fifties, leans over ..."
      RIGHT  "bizarrotrn Bizarro, the unit's commander in a muddy field uniform, leans
              over the map table and traces the river with one finger"
    THIS LAW APPLIES TO CAST CHARACTERS AND TO NOBODY ELSE, and getting that wrong in the
    other direction ruins the film just as fast. EVERY OTHER CHARACTER **MUST** be
    described the way the concept asks - species, age, build, all of it. If the brief says
    the soldiers are humanoid animals, then the soldiers ARE humanoid animals, on screen,
    in the words, in as many shots as they appear: badgers, wolves, a boar sergeant. A
    plan that quietly turns them all into ordinary humans has thrown away the premise and
    is WRONG, even though every cast character in it is lawful.
    WHEN THE PREMISE COVERS EVERYONE - "all the characters are animals", "everyone is a
    robot" - the cast character is the ONE EXCEPTION, and the exception is SILENCE, not
    conversion. Do not say what they are. Do not turn the rest of the cast human to match
    them. Do not write a sentence explaining why they look different. Give the cast
    character a role, a uniform and an action, describe everyone else exactly as the
    premise demands, and let the trained face answer the only question you left open.
L13 NEVER WRITE SPEECH THE VIEWER CANNOT HEAR. A shot is SPOKEN, SUNG or SILENT, and you
    must choose one on purpose.
      SPOKEN  the exact words are in the shot, in the dialogue form: <d>[English] Move out,
              and keep to the treeline.</d> Short - one or two sentences. The words being
              present IN THAT TAG is what switches the voice on. Quotation marks are not a
              substitute: He says, "Move out" leaves the voice switched OFF and the model
              guessing. The tag, every time.
      SUNG    the exact lyric is in the shot in the same dialogue form, and the description
              says the character SINGS it: He sings <d>[English] Love your fate.</d> The
              mouth then performs the lyric — this is how a music video's lip-sync shot is
              written, and the real track is laid over the cut afterwards. Sung delivery is
              SLOW: budget about 1.5 words per second against speech's 2.5. A "sings" or
              "chants" with no lyric written is the same violation as a wordless "explains".
      SILENT  no speech happens at all. Then the description must not say that speech
              happens: no explains, briefs, tells, discusses, talks, orders, argues,
              murmurs, mutters, whispers, addresses, announces, sings, chants - and no
              describing a voice ("his voice low and authoritative"). The soundscape must
              not carry speech either: no murmur, chatter, voices, conversation, talking,
              singing.
    "He explains the mission, his voice low and authoritative" with no words written is the
    worst of both: the model is told a man is speaking and given nothing to say, so it
    invents mouth noise. If the beat needs him to brief the unit, WRITE THE LINE. If it
    does not, show him working the map in silence and put footsteps, wind, paper and radio
    static in the soundscape instead.
"""

_H3_EXAMPLES = """\
EXAMPLE SHOTS - H3 register. Copy this voice. Note that no "description" below contains a
camera sentence or an ending clause: those are the "camera" and "settle" keys.

{
  "n": 1,
  "title": "Clippers",
  "character_id": null,
  "duration_s": 5,
  "camera": "handheld",
  "face": "close",
  "description": "Live-action, cinematic, a close two-shot in a bright kitchen: a woman with a freshly buzzed head sits still while her teenage daughter stands behind her running hair clippers over her scalp, warm window light across both faces. Photoreal, heavy 35mm film grain. Played quietly and underplayed throughout: the daughter draws one more slow clipper pass and a small tuft of hair falls away, the mother's eyes shine wet and she blinks once and keeps very still, and she reaches up and closes her hand over her daughter's hand on her shoulder, and the smallest smile arrives at the corner of her mouth.",
  "settle": "that small smile is simply held, both of them still, hands joined on the shoulder",
  "soundscape": "The steady buzz of hair clippers, one soft unsteady breath, a dripping kitchen tap, and the hum of a fridge. Nobody speaks and no voice is heard at any point.",
  "music": "N/A"
}

{
  "n": 2,
  "title": "The box",
  "character_id": null,
  "duration_s": 5,
  "camera": "static",
  "face": "medium",
  "description": "Live-action, cinematic, a medium-wide shot at dusk of a man in an open-collared shirt beside a battered steel dumpster in an empty back street, a cardboard office box in his arms with a small potted plant balanced on top. Photoreal, heavy 35mm film grain. Blue dusk light. Everything happens at natural real-time speed, never in slow motion. He heaves the box up and away from his chest in one fast decisive movement and it drops hard into the dumpster with the plant tumbling in after it and a puff of dust rising, then his shoulders drop as the weight leaves him and he lets out one long exhale, his face staying square to the lens the whole time.",
  "settle": "he is standing empty-handed with his shoulders down and his face still to the lens",
  "soundscape": "Quiet back-street evening ambience with distant traffic, the hollow bang of cardboard hitting steel, a clatter of a plant pot, and one long relieved exhale.",
  "music": "N/A"
}

{
  "n": 3,
  "title": "Impossible",
  "character_id": null,
  "duration_s": 5,
  "camera": "push_in",
  "face": "close",
  "description": "Live-action, cinematic, a medium close-up of a man in a dark curly fur hat and heavy fur coat on an open dune ridge. He faces the camera squarely and holds eye contact for the entire duration as the wind lifts the fur at his collar. Hard low sun rakes from camera left at the end of the day, carving one bright warm edge down his cheekbone while the other side of his face falls into open shadow; the dune line behind him is a clean dark silhouette against a pale sky. The man, with a warm, measured, slightly gravelled voice (S1), says: <d>[English] They said this was impossible.</d> Exactly as his voice stops, his jaw ceases speaking motion and his mouth settles into a closed steady half-smile.",
  "settle": "his eyes stay on the lens and nothing but the fur at his collar moves",
  "soundscape": "A steady desert wind moves across open sand for the full duration, with the dry rustle of fur at his collar and one soft gust that rises and falls.",
  "music": "N/A"
}

{
  "n": 4,
  "title": "Crema",
  "character_id": null,
  "duration_s": 5,
  "camera": "push_in",
  "face": "none",
  "description": "Live-action, cinematic, an extreme macro of a warm glass cup under a polished steel spout, filling with espresso. A hard raking key light from camera left picks out the rim of the glass against a matte black background. Two dark streams meet and braid as they fall, the liquid climbing the glass while a dense hazelnut crema builds on the surface and settles into a smooth unbroken layer, one bead of condensation sliding down the outside of the glass.",
  "settle": "the crema lies flat and unbroken and the surface is completely still",
  "soundscape": "The low hiss of a pump, the fine trickle of liquid into glass, and a quiet kitchen room tone underneath.",
  "music": "One low sustained synth tone at a slow tempo that rises slightly as the glass fills and drops away at the end."
}
"""

_LTX_EXAMPLE = """\
EXAMPLE SHOT - LTX register. Use this voice ONLY for a shot whose "character_id" is not
null. It is one continuous prose paragraph, 70-120 words: no [Shot 1] marker, no <d> tags,
dialogue in single quotes with a voice descriptor in front of it, and no field labels. Do
NOT write the character's trigger word yourself - leave the person unnamed and described,
the trigger is attached afterwards.

{
  "n": 4,
  "title": "The confession",
  "character_id": "bizarrotrn",
  "duration_s": 10,
  "camera": "handheld",
  "face": "close",
  "description": "A weary man in a soft grey jacket sits in a sterile interview room, medium close-up, fluorescent overhead light, shallow depth of field. He breathes in, looks down at his hands, then up at the lens and holds there. He says quietly and clearly: 'I stopped leaving the chair.'",
  "settle": "he is still, jaw set, both eyes back on the lens",
  "soundscape": "Room tone, a fluorescent hum, one unsteady breath, clear dialogue.",
  "music": "N/A"
}

Words that wreck an LTX shot because they trigger letterbox bars: cinematic, filmic,
anamorphic, widescreen, epic, 2.39:1. Never use them in the LTX register. LTX also cannot
render crowds, rows of people, circles of seated people, or three or more faces the camera
must read - pick ONE principal and imply the rest in the soundscape. Do not describe
fingers gripping objects, and never ask for on-screen text.
"""


# ONE SHOT — a take that never cuts, offered to the planner as a cinematic tool it may
# reach for INSIDE an ordinary film. Before this the only way a take reached the board
# was the whole film being one (`_sb_take_concept` in the panel + collapse_take); the
# owner's ask was that the director could plan one shot of a film as a real one-shot
# "so it looks really like a real movie instead of just 5-second clips one after the
# other". The field names stay `take_seconds` + `beats` — the same wire the Video tab's
# take and the whole-film take already use — so the panel renders it with no new path.
_ONE_SHOT = """\
ONE SHOT - a take that never cuts, used as a cinematic tool

A One Shot is a single unbroken take of 30 to 120 seconds inside an otherwise ordinary
film: the camera never cuts, ONE movement happens per 5-second beat, and the WORLD changes
around the subject (a door opens, a street empties, someone steps into frame) while the
subject, the camera position and the light stay continuous from one beat into the next.
It is the right tool for a walk-and-talk, a chase or a POV ride, a monologue or a
confession, a reveal that needs unbroken time, or an arrival through a place. It is the
WRONG tool for a montage, for cross-cutting between places, and for anything that needs a
reverse angle or a second camera position - those are ordinary shots. Use it at most once
or twice per film, only where unbroken time earns something, and never for the whole film
unless the brief asks for one take. A One Shot counts as ONE shot in the shot count.

How to write one. The shot object keeps every key above and adds exactly two:
  "take_seconds"  one of 30, 45, 60, 90, 120
  "beats"         a list of EXACTLY take_seconds / 5 strings (30 -> 6, 60 -> 12), in
                  order, each one what happens NEXT in the same unbroken shot. Lead every
                  beat with the movement (of the subject, the camera or the world), name
                  the sound in every beat, and state the time of day and the weather ONCE,
                  in the first beat, and never change them - no 'dawn breaks', no lights
                  coming on. Every beat starts exactly where the one before it ends: no
                  new angle, no jump in time.
"description" is the whole take in one paragraph; "duration_s" equals "take_seconds".
Every other shot in the film has NEITHER key.

{
  "n": 3,
  "title": "Down the corridor",
  "character_id": null,
  "duration_s": 30,
  "camera": "tracking",
  "face": "medium",
  "description": "Live-action, cinematic, a medium shot moving backwards ahead of a nurse in pale blue scrubs as she walks the length of a hospital corridor at night under flat fluorescent light, past a wheeled trolley, a closed pharmacy hatch and a waiting man who rises as she passes, to the double doors of a ward, which she pushes open onto a dim room with one lit bed.",
  "settle": "she stands in the ward doorway, still, the lit bed ahead of her",
  "soundscape": "Soft soles on vinyl flooring, the fluorescent hum, a distant call bell, the double doors swinging.",
  "music": "N/A",
  "take_seconds": 30,
  "beats": [
    "Night, flat fluorescent light. She walks straight toward the lens down the empty corridor, soles squeaking on the vinyl, hands loose at her sides.",
    "A wheeled trolley rolls into frame from the left and she steps around it without slowing, its wheels rattling as it passes.",
    "The closed pharmacy hatch slides past on her right; her eyes flick to it and then hold the lens again, the fluorescent hum steady.",
    "A man on a bench rises as she reaches him and falls into step beside her, his coat brushing the wall, a call bell ringing somewhere behind them.",
    "The corridor bends left and the double doors of the ward come into view ahead; her pace slows and she lifts one hand toward them.",
    "She pushes the doors open and stops in the doorway, the doors swinging behind her, the one lit bed ahead of her in the dim room."
  ]
}
"""


def _build_system_prompt(engine_hint: str, has_characters: bool,
                         allow_hidden: bool = False) -> str:
    dialect = """\
THE TARGET DIALECT

Your "description", "soundscape" and "music" for an H3 shot are assembled by the program
into MiniMax H3's official three-field prompt, exactly like this - you never type the field
labels or the shot marker yourself:

    integrated_multimodal_description: [Shot 1] <your description>

    overall_soundscape: <your soundscape>

    non_diegetic_music: <your music>

So "description" must START with the style token ("Live-action, cinematic," / "3D CG," /
"2D-animated," / "claymation," / "vintage film,") followed immediately by the shot size,
and must read as one continuous paragraph of 70-140 words.
"""
    contract = """\
OUTPUT CONTRACT - read this twice

Reply with ONE JSON object and NOTHING else. No preamble, no explanation, no markdown
outside the JSON. The object has exactly two keys:

{
  "title": "<short title for the whole film, 2-6 words>",
  "shots": [ <one object per shot, in story order> ]
}

Each shot object has exactly these keys and no others (a One Shot adds two more - see
ONE SHOT below):

  "n"            integer, 1-based, in order
  "title"        2-5 words naming the beat
  "character_id" a cast id from CAST below, or null
  "duration_s"   3, 5 or 10
  "camera"       ONE of: static, push_in, pull_back, handheld, orbit, pan, tilt_up, tracking
  "face"         ONE of: close, medium, none. This is about WHO IS PRESENT, not about how
                 important the face is to you.
                   close  - a face is the subject of the shot and fills much of the frame
                   medium - a person is on screen anywhere at all - near, far, small, in the
                            background, in a wide shot, only their hands. Their face must
                            stay readable. When in doubt this is the answer.
                   none   - there is NO PERSON of any kind in this shot: an object, a
                            landscape, a machine, a graphic. Nobody. Not one.
                 There is no fourth option. See L11.
  "description"  the visual + action + dialogue paragraph, 70-140 words. It does NOT contain
                 a camera sentence and does NOT describe how the shot ends - those are the
                 "camera" and "settle" keys, and the program writes their sentences for you.
  "settle"       a short phrase naming the settled state the shot ENDS in, written to follow
                 "for the last two seconds ..." - e.g. "he is standing empty-handed with his
                 shoulders down". Every shot needs one.
  "soundscape"   1-3 sentences of ambience and physical sound
  "music"        1-2 sentences of instrumentation/tempo/dynamics, or "N/A"

The program - not you - assigns schema version, ids, seeds, modes, engines, resolutions and
character trigger words, and turns "camera" and "settle" into their sentences. Do not invent
other keys. Do not wrap the object in an array.
"""
    parts = [
        "You are the Phosphene storyboard planner.\n\n"
        "You turn one sentence of concept into a shot-by-shot film plan that a local video\n"
        "model will render unattended. You are the DIRECTOR, not a transcriber: the concept is\n"
        "INTENT, and your job is to express it as shots this model actually renders well.\n"
        "Every shot must be a single continuous take that can stand on its own - there is no\n"
        "editing, no cutting inside a shot, and no camera move that reframes.\n",
        contract,
    ]
    if engine_hint == "ltx":
        # Every shot will render on LTX, so the H3 dialect would only confuse the model.
        parts.append(_LTX_EXAMPLE)
        parts.append("EVERY shot in this film uses the LTX register shown above, whether or\n"
                     "not it has a character_id. Never write [Shot 1], <d> tags or field labels.\n")
    else:
        parts.append(dialect)
        parts.append(_H3_EXAMPLES)
        if has_characters and engine_hint != "h3":
            parts.append(_LTX_EXAMPLE)
    parts.append(_LAWS)
    parts.append("""\
COMPOSITION LIMITS - rewrite around these instead of asking for them

  crowd / audience / rally            -> ONE face from it, the rest implied in the soundscape
  circle of seated people, group ring -> a close shot on whoever is speaking
  rows of desks, classroom, newsroom  -> the principal in front, everything behind soft
  three or more faces the camera reads-> pick one principal; the others are off-frame
  fast hands, fingers gripping props  -> frame past the hands
  "the camera pulls back to reveal"   -> state the FINAL framing only, no reveal

ACID TEST before you write a shot: can a reader point at exactly one subject the camera
will hold? If not, rewrite it.

DURATION: 3 s for a beat or a cutaway, 5 s for one action or one short spoken line, 10 s
only when a line genuinely needs it. Speech runs about 2.5 words per second; a SUNG line
runs about 1.5 - a lyric stretches over the beat, so a sung shot needs noticeably more
seconds than its spoken twin. Add one second of breath before the line and about one and a
half seconds of silence after it, then round up. Never trim a line the brief gave you in
order to fit a duration - lengthen the shot.

VARIETY IS A HARD REQUIREMENT, not a preference. Before you answer, look down the list of
"camera" values you have written: no two consecutive shots may share one, and a six-shot
film must use at least three different ones. Do the same for shot size - vary across extreme
close-up, close-up, medium, wide. Let at most half the shots contain dialogue.

EVERY SHOT MUST CONTAIN A PHYSICAL ACTION that starts and finishes inside the clip. A shot
that only describes what a place looks like is a photograph, and this model fills the empty
time by inventing motion. If you cannot name something that moves, the shot does not exist.

WRITE WHAT A LENS COULD SEE. "A testament to human ingenuity", "a symbol of hope", "a space
that feels lived-in", "his face etched with solitude" are unrenderable - they instruct
nothing. Replace each with the visible fact underneath it.
""")
    # The One Shot exemplar is written in the H3 register; on an LTX-only film
    # the style token would teach the exact word the LTX rules forbid.
    parts.append(_ONE_SHOT if engine_hint != "ltx"
                 else _ONE_SHOT.replace("Live-action, cinematic, a medium shot", "A medium shot"))
    if allow_hidden:
        parts.append(
            "FACES: this brief explicitly asked for hidden or obscured faces, so L11 is "
            "relaxed for the shots where the brief wants it - you may use \"hidden\" as a\n"
            "\"face\" value on those shots. Every other shot still keeps its face readable.\n")
    return "\n".join(parts)


def _build_user_prompt(
    concept: str,
    n_shots: int,
    style: str,
    cast: Sequence[Dict[str, str]],
    must_include: Sequence[str],
    locations: Sequence[Dict[str, str]] = (),
    screenplay: str = "",
    floor_plan: str = "",
) -> str:
    lines = ["BRIEF", "", "CONCEPT: %s" % concept.strip(), "", "SHOT COUNT: exactly %d shots." % n_shots]
    if style and style.strip():
        lines += ["", "STYLE (applies to every shot; keep it identical across all of them): %s"
                  % style.strip()]
    if cast:
        lines += ["", "CAST - these are trained characters. A shot that features one sets"
                  " \"character_id\" to that id and is written in the LTX register:"]
        for c in cast:
            desc = (c.get("description") or "").strip()
            noun = (c.get("subject_noun") or "").strip()
            pron = (c.get("pronoun") or "").strip()
            facts = []
            if noun:
                facts.append("a %s" % noun)
            if pron:
                facts.append("%s/%s" % (pron, "him" if pron == "he" else
                                        "her" if pron == "she" else "them"))
            lines.append("  - %s%s%s%s" % (
                c["id"],
                (" (%s)" % c["name"]) if c.get("name") and c["name"] != c["id"] else "",
                (" - %s" % ", ".join(facts)) if facts else "",
                (": " + desc) if desc else "",
            ))
        # L12 restated where the cast is actually read, because the laws block is long and
        # this is the moment the model is deciding what these people look like.
        lines += [
            "  THE LINE ABOVE IS EVERYTHING YOU KNOW ABOUT HOW THEY LOOK, and it is all you",
            "  are allowed to imply. Their face is a trained model you cannot see. Give them",
            "  a role, wardrobe, action and emotion - never a species, creature type, face,",
            "  hair, eyes, build or age.",
            "  EVERYONE ELSE IN THE FILM IS DESCRIBED NORMALLY, and if the concept gives the",
            "  world a premise - humanoid animals, robots, a period, a species - that premise",
            "  applies to them in full and must be visible in the shots they appear in. The",
            "  cast character is the ONE you leave undescribed; they are not a reason to make",
            "  the rest of the film plain.",
            "  Any shot without a listed character sets \"character_id\": null.",
        ]
    else:
        lines += ["", "CAST: none. Every shot sets \"character_id\": null."]
    if locations:
        # The room is injected into the prompt LATER, from the board, for every
        # shot that names it. The model is told about them anyway so it writes
        # shots that belong somewhere, and so it does not spend its own prose
        # re-describing scenery that is about to be appended underneath it.
        lines += ["", "LOCATIONS - the film happens in these places, and ONLY these:"]
        has_views = False
        for loc in locations:
            lines.append("  - %s: %s" % (loc.get("name") or loc.get("id"),
                                         (loc.get("description") or "").strip()))
            views = [v for v in (loc.get("views") or [])
                     if isinstance(v, dict) and v.get("id") and v.get("description")]
            if not views:
                continue
            has_views = True
            lines.append("    VIEWS of %s - which way the camera points. Every shot here"
                         " picks ONE, by id:" % (loc.get("name") or loc.get("id")))
            for v in views:
                lines.append("      * %s (%s): %s" % (
                    str(v.get("id")).strip(), str(v.get("name") or "").strip(),
                    str(v.get("description") or "").strip()))
        lines += [
            "  Every shot sets \"location\" to one of the names above, exactly as written.",
            "  DO NOT describe the place in the shot description - it is added automatically",
            "  from the line above, identically on every shot that names it. Describe the",
            "  PEOPLE, the ACTION and the FRAMING only. A shot that re-describes the room in",
            "  its own words is how the same room comes back as four different rooms.",
        ]
        if has_views:
            lines += ["", _GEOGRAPHY_LAWS]
    if must_include:
        lines += ["", "MUST APPEAR somewhere in the film:"]
        for m in must_include:
            lines.append("  - %s" % str(m).strip())
    if floor_plan and floor_plan.strip():
        # WHERE EVERYTHING IS, written once. The views above are derived from
        # this paragraph, and a shot that knows what is behind whom stops
        # inventing a new room every time the camera turns around.
        lines += ["", "THE FLOOR PLAN. This is the space the scene happens in. It is",
                  "already decided - do not move anybody, and do not re-describe it in a",
                  "shot:", "", floor_plan.strip(), ""]
    if screenplay and screenplay.strip():
        # The scene, written first and handed down. Without this the model is
        # asked to invent structure and coverage in the same breath, and what
        # comes back is a run of equally-weighted moments — the owner's words:
        # "It is not working properly. It's just a succession of shots."
        lines += ["", "THE SCENE. This has already been written. Your job is to SHOOT it,",
                  "not to rewrite it. Cover these beats in order, keep every spoken line",
                  "word for word, and put each line in the shot where it is spoken:", "",
                  screenplay.strip(), "",
                  "A beat with no line is a shot with no line. Do not invent dialogue for",
                  "it and do not say that someone speaks in it."]
    lines += ["", "Return the JSON object now. %d shots. Nothing before it, nothing after it."
              % n_shots]
    return "\n".join(lines)


_SCREENPLAY_SYSTEM = """\
You are a screenwriter. You write the SCENE, not a shot list. Prose and dialogue only.

Rules:
1. It is ONE continuous scene in ONE place, unless the brief names more. Everyone in it can
   see and hear everyone else. Do not cut to somewhere nobody could hear the last line.
2. Give it a shape: an opening, a turn in the middle, and a button at the end. A run of
   equally-weighted moments is not a scene.
3. WRITE THE ACTUAL LINES. Every line of dialogue is words a person says out loud, in double
   quotes, attributed by name. Never "he explains the situation" — write what he says.
   A sung line works the same way: write the exact lyric in double quotes and mark the
   attribution "(sings)" after the name. Never "she sings the chorus" — write the words.
4. Spread the dialogue across the scene. Do not put every line in the first beat.
5. Some beats have no dialogue at all. Those are where the scene breathes; say what the
   person is DOING instead.
6. Keep it short: 8-14 beats, one or two sentences each. This is a 30-60 second scene.

Output plain text, one beat per line, in this form:

  BEAT - <what happens, present tense>
  NAME: "the line, exactly as spoken"

No preamble, no headings, no shot numbers, no camera directions. The camera is not your job.
"""


def _screenplay_text(resp: Dict[str, Any]) -> str:
    """The scene out of a generate() response, or "" if there is nothing usable.

    Small models like to answer a screenwriting brief with a paragraph of
    preamble ("Sure! Here is the scene:"), and a scene that begins with that
    would be handed to the shot pass as if it were a beat. Keep only the lines
    that look like the form that was asked for, and if none do, return nothing
    and let the plan proceed exactly as it did before this pass existed.
    """
    text = (resp or {}).get("text") or ""
    keep = [ln.rstrip() for ln in text.splitlines()
            if _SCREENPLAY_LINE_RE.match(ln.strip())]
    if len(keep) < 3:
        return ""
    return "\n".join(keep)[:4000]


# A beat line or an attributed line of dialogue. Anything else is chatter.
# The class carries parentheses so a sung attribution — NAME (sings): "..." —
# survives the filter; without them every lyric line was silently dropped.
_SCREENPLAY_LINE_RE = re.compile(r"^(?:BEAT\b|[A-Z][A-Za-z0-9 _\'()-]{0,30}:)")


def _build_screenplay_prompt(concept: str, n_shots: int, style: str,
                             cast: Sequence[Dict[str, str]],
                             must_include: Sequence[str],
                             locations: Sequence[Dict[str, str]] = ()) -> str:
    lines = ["CONCEPT: %s" % (concept or "").strip(), "",
             "LENGTH: about %d beats." % max(6, min(14, n_shots))]
    if cast:
        lines += ["", "WHO IS IN IT:"]
        for c in cast:
            who = c.get("name") or c.get("id")
            noun = (c.get("subject_noun") or "").strip()
            lines.append("  - %s%s" % (who, (" (%s)" % noun) if noun else ""))
        lines.append("  Use these names in the dialogue attributions.")
    if locations:
        lines += ["", "WHERE:"]
        for loc in locations:
            lines.append("  - %s: %s" % (loc.get("name") or loc.get("id"),
                                         (loc.get("description") or "").strip()))
        if len(locations) == 1:
            lines.append("  The whole scene happens HERE. Nobody leaves.")
    if must_include:
        lines += ["", "MUST HAPPEN:"]
        for m in must_include:
            lines.append("  - %s" % str(m).strip())
    lines += ["", "Write the scene now."]
    return "\n".join(lines)


# ======================================================================================
# THE GEOGRAPHY PASS — the screenplay is TIME, this is SPACE
# ======================================================================================
# The owner, after a full day of manual continuity work: "Do you notice all the work I
# had to do to make this scene happen in the same place and have proper angles? ... You
# need to first make a concept of the whole situation. For instance, a man or woman in a
# bar — behind him there is this, behind her there is that. When they sit together, this
# is what you see. So there is continuity between the shots."
#
# A scene is a SPACE before it is a shot list. This pass writes the floor plan ONCE —
# who stands where, what is behind each of them, where the light comes from — and then
# DERIVES the named views of the location from it, so the reverse angle automatically
# excludes what has moved behind the camera and the light lands on the correct side.
# The whole car-wash day was the prototype: carwash/carwash_reverse, the flipped sun,
# the no-car reverse background, the sign living only on the far side — all of it
# hand-built, all of it derivable from one paragraph.
# What the SHOT pass is told once the floor plan exists. Stated in the brief
# rather than in the system prompt because it is only true for a film whose
# locations actually carry views — a plan with none must read exactly as it did
# before this pass existed.
_GEOGRAPHY_LAWS = """\
  GEOGRAPHY - the scene happens in a space, and the space does not move.
  Every shot in a place that has views sets "view" to one of the view ids above, and
  "eyeline" to "left", "right" or "lens".
  "eyeline" is where the person LOOKS as the audience sees it: "right" means their eyes go
  off past the right edge of the frame, "left" past the left edge, "lens" straight down the
  barrel at the audience. Use "lens" only for a piece to camera.
  THE 180-DEGREE RULE. Two people talking keep the same sides of the screen for the whole
  conversation. If he looks frame-RIGHT at her, then in her shot she looks frame-LEFT back
  at him. Two shots that cut between them may NEVER claim the same eyeline - that is the
  cut where the audience watches both of them turn and stare at the same wall.
  A shot on a reverse view must not mention anything that view says is not in it. If the
  view says "no car in frame", the shot has no car in it, in any words.
  Never put a character's own body in the view behind them: what the camera sees over his
  shoulder is HER side of the room, not another copy of him."""


_GEOGRAPHY_SYSTEM = """\
You are a director blocking a scene. Before a single shot exists you decide where
everything IS, and then you describe the place from each direction the camera will point.

Work in two steps.

STEP 1 - THE FLOOR PLAN. One paragraph, present tense. Say where each person stands or
sits, where the key objects are, and where the light comes from. Say what is BEHIND each
person, because that is what the camera sees when it looks at them. Do not describe shots.

STEP 2 - THE VIEWS. Turn the floor plan into 2 to 4 camera directions and describe what
each one SEES. Rules that are not negotiable:

  * When two people face each other you write AT LEAST an establishing view and its
    REVERSE - the camera turned 180 degrees to look back the other way.
  * A view never contains what is behind the camera in that view. If the establishing
    view holds the car, the reverse view does NOT hold the car - and you SAY SO, in the
    words "no car in frame", for every prominent thing the other views hold. That
    sentence is what stops it being put back later.
  * THE LIGHT FLIPS. A sun raking in from camera LEFT rakes in from camera RIGHT once the
    camera turns 180 degrees. Every view names its side.
  * Never put a person's own body in the view behind them. The view behind HIM is what
    the camera sees over his shoulder, which is HER side of the room, not him.
  * Each view is self-contained: a reader who has only that one sentence must be able to
    picture the frame without the floor plan.

Return ONE JSON object and nothing else:

{"floor_plan": "<the paragraph from step 1>",
 "views": [{"location": "<the location name, exactly as it was given to you>",
            "id": "<short_lowercase_id>",
            "name": "<what it is, in a few words>",
            "light": "camera left" | "camera right",
            "description": "<what this camera sees, one or two sentences>"}]}

No preamble, no headings, nothing before the object and nothing after it.
"""


def _build_geography_prompt(concept: str, style: str,
                            cast: Sequence[Dict[str, str]],
                            locations: Sequence[Dict[str, Any]],
                            screenplay: str = "") -> str:
    lines = ["CONCEPT: %s" % (concept or "").strip()]
    if style and style.strip():
        lines += ["", "STYLE: %s" % style.strip()]
    if cast:
        lines += ["", "WHO IS IN IT:"]
        for c in cast:
            who = c.get("name") or c.get("id")
            noun = (c.get("subject_noun") or "").strip()
            lines.append("  - %s%s" % (who, (" (%s)" % noun) if noun else ""))
    lines += ["", "THE PLACES - block every one of them, by name:"]
    for loc in locations:
        lines.append("  - %s: %s" % (loc.get("name") or loc.get("id"),
                                     (loc.get("description") or "").strip()))
    if screenplay and screenplay.strip():
        # The scene, so the blocking serves what actually happens in it — who
        # turns to whom, and therefore which direction has to exist.
        lines += ["", "THE SCENE THAT HAPPENS HERE:", "", screenplay.strip()[:2000]]
    lines += ["", "Write the floor plan and the views now."]
    return "\n".join(lines)


# A view id is a location id: lowercase, digits, - and _. Kept local so the
# planner does not need storyboard.py imported to slugify one.
_VIEW_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")
_CAMERA_SIDE_RE = re.compile(r"camera[ -](left|right)", re.IGNORECASE)
MAX_VIEWS_PER_LOCATION = 4


def _slug_view_id(*candidates: Any) -> str:
    for cand in candidates:
        raw = re.sub(r"[^a-z0-9]+", "_", str(cand or "").strip().lower()).strip("_")[:40]
        if raw and _VIEW_ID_RE.match(raw):
            return raw
    return ""


def _fold_light(description: str, light: Any) -> str:
    """Make sure the view's own sentence says which side the light rakes from.

    The side is the single most load-bearing word in a reverse angle and the
    model likes to answer it in the `light` field and then forget it in the
    prose that actually reaches the renderer. Folded in only when the sentence
    does not already carry a side of its own.
    """
    desc = (description or "").strip()
    m = _CAMERA_SIDE_RE.search(str(light or ""))
    side = m.group(1).lower() if m else ""
    if not side:
        low = str(light or "").lower()
        side = "left" if "left" in low else "right" if "right" in low else ""
    if not side or _CAMERA_SIDE_RE.search(desc):
        return desc
    return "%s, the light rakes in from camera %s" % (desc.rstrip(" .,;"), side)


def _geography_plan(resp: Dict[str, Any],
                    locations: Sequence[Dict[str, Any]]) -> Tuple[str, Dict[str, List[Dict[str, str]]]]:
    """(floor plan, {location_id: [views]}) out of a generate() response.

    Empty on anything unusable, exactly like `_screenplay_text` — a model that
    answers a blocking brief with a paragraph of enthusiasm must leave the plan
    behaving precisely as it did before this pass existed, not hand chatter
    down as if it were geography.
    """
    obj = next((o for o in _json_dicts((resp or {}).get("text") or "")
                if isinstance(o.get("views"), list)), None)
    if obj is None:
        return "", {}
    floor = str(obj.get("floor_plan") or obj.get("floorplan") or
                obj.get("layout") or "").strip()[:1500]
    raw_views = obj["views"]

    by_name: Dict[str, Dict[str, Any]] = {}
    for loc in locations:
        for key in (loc.get("name"), loc.get("id")):
            if str(key or "").strip():
                by_name[str(key).strip().lower()] = loc

    out: Dict[str, List[Dict[str, str]]] = {}
    for v in raw_views:
        if not isinstance(v, dict):
            continue
        desc = _fold_light(str(v.get("description") or v.get("sees") or "").strip(),
                           v.get("light"))
        # A view is a DESCRIPTION. An id with nothing under it composes to the
        # location's own text and pretends coverage exists.
        if len(desc) < 20:
            continue
        want = str(v.get("location") or v.get("place") or "").strip().lower()
        loc = by_name.get(want)
        if loc is None and len(locations) == 1:
            loc = locations[0]
        if loc is None:
            continue
        lid = str(loc.get("id"))
        rows = out.setdefault(lid, [])
        if len(rows) >= MAX_VIEWS_PER_LOCATION:
            continue
        vid = _slug_view_id(v.get("id"), v.get("name"), "view_%d" % (len(rows) + 1))
        if not vid or any(r["id"] == vid for r in rows):
            vid = _slug_view_id("%s_%d" % (vid or "view", len(rows) + 1)) or \
                "view_%d" % (len(rows) + 1)
        if any(r["id"] == vid for r in rows):
            continue
        rows.append({"id": vid,
                     "name": str(v.get("name") or v.get("id") or vid).strip()[:80],
                     "description": desc[:600]})

    out = {k: v for k, v in out.items() if v}
    if not out:
        return "", {}
    return floor, out


def _merge_views(locations: Sequence[Dict[str, Any]],
                 views: Dict[str, List[Dict[str, str]]]) -> List[Dict[str, Any]]:
    """The board's locations, each carrying the views the floor plan derived.

    Copies rather than mutating: `locations` is the caller's list — on the
    panel it is the user's own Locations box, parsed — and a pass that edits it
    in place would leave the user's places rewritten by a model even on the
    paths where the plan is thrown away.
    """
    merged: List[Dict[str, Any]] = []
    for loc in locations:
        row = dict(loc)
        rows = views.get(str(row.get("id")))
        if rows:
            row["views"] = [dict(r) for r in rows]
        merged.append(row)
    return merged


# The eyeline vocabulary, and every way a model says it. Anything that does not
# land in here is dropped rather than written onto a shot: `eyeline` outside the
# vocabulary is a hard validator error, and a plan that cannot be rendered is a
# worse outcome than a shot with no eyeline on it.
_EYELINE_WORDS = {
    "left": "left", "frame-left": "left", "frame left": "left",
    "screen-left": "left", "screen left": "left", "camera-left": "left",
    "camera left": "left", "off-frame left": "left", "l": "left",
    "right": "right", "frame-right": "right", "frame right": "right",
    "screen-right": "right", "screen right": "right", "camera-right": "right",
    "camera right": "right", "off-frame right": "right", "r": "right",
    "lens": "lens", "camera": "lens", "to camera": "lens", "at camera": "lens",
    "into the lens": "lens", "down the lens": "lens", "to the lens": "lens",
}


def _eyeline_key(raw: Any) -> str:
    key = re.sub(r"\s+", " ", str(raw or "").strip().lower()).strip(".,")
    return _EYELINE_WORDS.get(key, "")


def _apply_geography(shot: Dict[str, Any], raw: Dict[str, Any], loc: Dict[str, Any],
                     n: int, warnings: List[str]) -> None:
    """Stamp `view` and `eyeline` onto one shot, matched against the floor plan."""
    views = [v for v in (loc.get("views") or []) if isinstance(v, dict) and v.get("id")]
    if views:
        want = str(raw.get("view") or raw.get("angle") or raw.get("facing") or "").strip().lower()
        hit = next((v for v in views
                    if want and want in (str(v.get("id") or "").strip().lower(),
                                         str(v.get("name") or "").strip().lower())), None)
        if hit is None and want:
            hit = next((v for v in views
                        if _slug_view_id(want) == str(v.get("id")).strip().lower()), None)
        if hit is None:
            # UNSPECIFIED COVERAGE IS THE MASTER. A shot that names no view
            # would compose from the location's own neutral sentence, which is
            # the pre-views behaviour and the thing this pass exists to end.
            # The first view is the establishing one by construction.
            hit = views[0]
            warnings.append(
                "shot %d named %sno view of %s — it was put on %r, the establishing view"
                % (n, ("view %r, which is not on the floor plan, so " % want) if want else "",
                   loc.get("name") or loc.get("id"), hit.get("id")))
        shot["view"] = str(hit["id"]).strip().lower()

    raw_eye = raw.get("eyeline") or raw.get("eye_line") or raw.get("looking")
    if raw_eye:
        eye = _eyeline_key(raw_eye)
        if eye:
            shot["eyeline"] = eye
        else:
            warnings.append("shot %d asked for eyeline %r, which is not left, right or "
                            "lens — the shot carries no eyeline" % (n, str(raw_eye)[:40]))


def _build_repair_prompt(bad_json: str, problems: Sequence[str], n_shots: int) -> str:
    return "\n".join([
        "Your previous reply did not pass validation.",
        "",
        "PROBLEMS FOUND:",
        "\n".join("  - %s" % p for p in problems),
        "",
        "YOUR PREVIOUS OUTPUT:",
        bad_json[:6000],
        "",
        "Fix ONLY those problems. Keep everything that was already good, word for word.",
        "Return the corrected JSON object - exactly %d shots, the same six keys per shot," % n_shots,
        "nothing before the object and nothing after it.",
    ])


def _build_film_feedback_prompt(previous: Dict[str, Any], note: str, n_shots: int) -> str:
    return "\n".join([
        "You already planned this film. The director has notes.",
        "",
        "DIRECTOR'S NOTES: %s" % note.strip(),
        "",
        "THE CURRENT PLAN:",
        json.dumps(_spec_to_model_view(previous), indent=1, ensure_ascii=False)[:9000],
        "",
        "Re-plan the film so the notes are satisfied. Change only what the notes require;",
        "leave everything else word for word as it was. Return the full corrected JSON",
        "object with exactly %d shots and nothing else." % n_shots,
    ])


def _build_shot_feedback_prompt(previous: Dict[str, Any], shot_n: int, note: str) -> str:
    shots = previous.get("shots") or []
    target = None
    for s in shots:
        if s.get("n") == shot_n:
            target = s
            break
    if target is None:
        raise PlannerError("shot %r is not in the plan (it has %d shots)" % (shot_n, len(shots)))
    neighbours = [s for s in shots if s.get("n") in (shot_n - 1, shot_n + 1)]
    return "\n".join([
        "You are re-rolling ONE shot of a film that is otherwise finished.",
        "",
        "THE FILM: %s" % (previous.get("title") or ""),
        "",
        "THE SHOTS AROUND IT, for continuity - do NOT return these:",
        json.dumps([_shot_to_model_view(s) for s in neighbours], indent=1, ensure_ascii=False)[:4000],
        "",
        "THE SHOT TO REPLACE (n=%d):" % shot_n,
        json.dumps(_shot_to_model_view(target), indent=1, ensure_ascii=False),
        "",
        "DIRECTOR'S NOTES ON THIS SHOT: %s" % note.strip(),
        "",
        "Return ONE JSON object: {\"title\": \"<the film title, unchanged>\", \"shots\": [ <the",
        "single replacement shot object, with n=%d> ]}. One shot only. Nothing else." % shot_n,
    ])


def _shot_to_model_view(shot: Dict[str, Any]) -> Dict[str, Any]:
    """The eight creative keys, as the model sees them (drops seeds/modes/engines/etc)."""
    out = {
        "n": shot.get("n"),
        "title": shot.get("title", ""),
        "character_id": shot.get("character_id"),
        "duration_s": shot.get("duration_s"),
        "camera": shot.get("camera", "static"),
        "face": shot.get("face", "medium"),
        "description": shot.get("description", ""),
        "settle": shot.get("settle", ""),
        "soundscape": shot.get("soundscape", ""),
        "music": shot.get("music", "N/A"),
    }
    # WHERE, and WHICH WAY. Only when the shot has them, so a board with no
    # locations shows the model exactly what it always showed. Without these a
    # film-level re-plan came back with the geography stripped — every shot
    # re-coerced onto the establishing view because nothing in the plan the
    # model was shown said which way it had been pointing.
    for key, src in (("location", "location_id"), ("view", "view"),
                     ("eyeline", "eyeline")):
        if shot.get(src):
            out[key] = shot[src]
    # A ONE SHOT keeps its take on a re-plan. Shown as the model wrote it
    # (`beat_lines`), not as assembled; a whole-film take collapsed from
    # beat-shots has only `beats`, which is what it gets.
    if shot.get("take_seconds"):
        out["take_seconds"] = shot["take_seconds"]
        out["beats"] = list(shot.get("beat_lines") or shot.get("beats") or [])
    return out


def _spec_to_model_view(spec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "title": spec.get("title", ""),
        "shots": [_shot_to_model_view(s) for s in (spec.get("shots") or [])],
    }


# --------------------------------------------------------------------------------------
# JSON extraction from small-model output
# --------------------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>|<thinking>.*?</thinking>", re.DOTALL | re.IGNORECASE)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")
_LINE_COMMENT_RE = re.compile(r"(^|\s)//[^\n]*")
_SMART = {"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'", "\u2013": "-", "\u2014": "-"}


def _balanced_objects(text: str) -> List[str]:
    """Every top-level {...} run in `text`, string- and escape-aware.

    A small model happily writes 'Here is your plan:' before the object and 'Let me know if
    you want changes!' after it, and sometimes emits two objects. Regex cannot match nested
    braces; a two-state scanner can, and it is 20 lines.
    """
    out: List[str] = []
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    out.append(text[start:i + 1])
                    start = -1
    if depth > 0 and start >= 0:
        # Truncated output (hit max_tokens mid-object). Hand back what we have; the
        # bracket-closing repair below often rescues it.
        out.append(text[start:])
    return out


def _soften(raw: str) -> str:
    s = raw
    for bad, good in _SMART.items():
        s = s.replace(bad, good)
    s = _LINE_COMMENT_RE.sub(r"\1", s)
    s = _TRAILING_COMMA_RE.sub(r"\1", s)
    return s


def _close_brackets(raw: str) -> str:
    """Best-effort close of an object truncated by the token limit.

    Closers are emitted from an actual stack, innermost first — appending all the `]`
    before all the `}` produces `..."]}}` for a truncated shot object, which is not JSON.
    """
    stack: List[str] = []
    in_str = False
    esc = False
    for ch in raw:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]":
            if stack and stack[-1] == ch:
                stack.pop()
    s = raw
    if in_str:
        s += '"'
    s = _TRAILING_COMMA_RE.sub(r"\1", s.rstrip().rstrip(","))
    return s + "".join(reversed(stack))


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Pull the plan object out of whatever the model actually said.

    Handles: clean JSON, ```json fences, bare ``` fences, prose before/after, `<think>`
    leakage, trailing commas, smart quotes, // comments, an object wrapped in a list, and
    output truncated mid-object by the token limit. Returns None only when there is no
    recoverable object at all.
    """
    for obj in _json_dicts(text):
        if "shots" in obj or "title" in obj:
            return obj
        # Model returned a bare list of shots under some other key.
        for v in obj.values():
            if isinstance(v, list) and v and isinstance(v[0], dict) and "description" in v[0]:
                return {"title": "", "shots": v}
    return None


def _json_dicts(text: str) -> Iterable[Dict[str, Any]]:
    """Every JSON object recoverable from a model reply, longest first.

    The recovery half of `extract_json_object`, split out because it is not
    plan-specific and a SECOND pass now needs it. The predicate is not shared
    on purpose: extract_json_object's last resort — "a list of dicts with a
    description under any key must be the shots" — read the geography pass's
    `views` array as a shot list and handed a floor plan to the coercer as a
    film. Two passes, two shapes, one recovery.
    """
    if not text:
        return
    body = _THINK_RE.sub(" ", text)

    candidates: List[str] = []
    for m in _FENCE_RE.finditer(body):
        candidates.append(m.group(1))
    # An unterminated fence is common when generation is truncated.
    if not candidates and "```" in body:
        candidates.append(body.split("```", 1)[1].lstrip("jsonJSON \n"))
    candidates.append(body)

    scanned: List[str] = []
    for c in candidates:
        scanned.extend(_balanced_objects(c))
        scanned.append(c.strip())

    # Longest first: a plan object is bigger than any stray fragment before it.
    for chunk in sorted({c for c in scanned if c.strip()}, key=len, reverse=True):
        for attempt in (chunk, _soften(chunk), _close_brackets(_soften(chunk))):
            try:
                obj = json.loads(attempt)
            except (ValueError, TypeError):
                continue
            if isinstance(obj, list):
                obj = next((o for o in obj if isinstance(o, dict)), None)
                if obj is None:
                    continue
            if isinstance(obj, dict):
                yield obj


# --------------------------------------------------------------------------------------
# Coercion — everything the validator checks, produced mechanically
# --------------------------------------------------------------------------------------

_DESC_KEYS = ("description", "integrated_multimodal_description", "visual", "prompt", "body", "text")
_SOUND_KEYS = ("soundscape", "overall_soundscape", "audio", "sound", "ambience")
_MUSIC_KEYS = ("music", "non_diegetic_music", "score", "soundtrack")

_SHOT_MARKER_RE = re.compile(r"^\s*\[Shot\s*\d+\][,\s]*", re.IGNORECASE)
_D_TAG_RE = re.compile(r"<d>\s*(?:\[[^\]]*\]\s*)?(.*?)</d>", re.DOTALL)

# --- the two laws the model demonstrably will not apply on its own ----------------------
# Measured over 5 concepts x 26 shots: with the laws stated as rules, the model reused ONE
# camera sentence for every shot in every film, and wrote the settle clause on 2 of 26
# shots. Both are now a small closed choice the model makes and a canonical sentence PYTHON
# writes — which is the same trick that makes the rest of the schema reliable.

_CAMERA_SENTENCES = {
    "static": "The camera holds a static shot, the frame never moves - no pan, no push-in, "
              "no reframing.",
    "push_in": "The camera pushes in with small amplitude at slow speed.",
    "pull_back": "The camera pulls back with small amplitude at slow speed.",
    "handheld": "The camera shakes slightly with small amplitude at slow speed, a handheld "
                "micro-sway that never reframes.",
    "orbit": "The camera arcs around the subject with small amplitude at slow speed.",
    "pan": "The camera pans with small amplitude at slow speed.",
    "tilt_up": "The camera tilts up with small amplitude at slow speed.",
    "tracking": "The camera tracks alongside the subject with medium amplitude at slow speed.",
}
CAMERA_KEYS = tuple(_CAMERA_SENTENCES)

# --- the face law ----------------------------------------------------------------------
# Faces are the quality metric for this project: a plan is judged on whether the face reads.
# Two failures were measured, so both are structural rather than advisory.
#
#   1. The model volunteers face-hiding framing, most often on the LAST shot of a film where
#      it reaches for something poetic — "silhouetted against the setting sun", "her
#      silhouette framed against the city lights", "his face obscured by the angle" (the last
#      one rendered a head half out of frame). Base rate before this law: 5 of 56 shots.
#   2. Turning breaks identity (H3 guide 7.2 / AURELIUS law 2).
#
# So `face` is a closed choice — close / medium / none / hidden — and Python writes the
# sentence, exactly as with `camera` and `settle`. `hidden` is refused unless the brief asked
# for it in so many words. Unambiguous blocking phrases are additionally scrubbed out of the
# prose, because the model writes them regardless of what it chose.

_FACE_LAW_CLOSE = (
    "The face fills much of the frame, and every face holds the exact angle to the lens it "
    "has at the start: heads stay square, shoulders do not pivot, and nobody rotates towards "
    "or away from the camera at any point. The face stays completely inside the frame for "
    "the entire duration with both eyes open and clearly readable, and is never cropped by "
    "the frame edge, never thrown into silhouette, and never covered by a hand, a prop or "
    "another person.")
_FACE_LAW_MEDIUM = (
    "Every face holds the exact angle to the lens it has at the start: heads stay square, "
    "shoulders do not pivot, and nobody rotates towards or away from the camera at any "
    "point. Each face stays completely inside the frame for the entire duration and is "
    "never cropped by the frame edge, never thrown into silhouette, and never covered by a "
    "hand, a prop or another person.")
_FACE_LAWS = {"close": _FACE_LAW_CLOSE, "medium": _FACE_LAW_MEDIUM, "none": "", "hidden": ""}
FACE_KEYS = ("close", "medium", "none", "hidden")

_PERSON_RE = re.compile(
    r"\b(man|woman|men|women|boy|girl|child|children|person|people|face|faces|keeper|"
    r"worker|he|she|his|her|him|hers|they|their|figure|hands?|crowd|speaker|"
    r"[a-z]+er's|[a-z]+man)\b", re.IGNORECASE)

# Unambiguous. These never describe anything but a face the viewer cannot read.
_FACE_BLOCK_RE = re.compile(
    r"\bobscur\w*"
    r"|\bfrom behind\b"
    r"|\bback to (?:the )?(?:camera|lens)\b"
    r"|\b(?:facing|turned|turning|looking) away from (?:the )?(?:camera|lens)\b"
    r"|\bout of (?:the )?frame\b"
    r"|\bcropped (?:out|off|at)\b"
    r"|\b(?:face|features|eyes) (?:is|are|were|stays?) (?:hidden|concealed|covered)\b"
    r"|\bhidden (?:by|behind|in) \w+"
    r"|\bconceal(?:s|ed|ing)\b"
    r"|\bwe (?:do not|don't|never) see\b", re.IGNORECASE)

# Ambiguous on its own — "the dune line behind him is a clean dark silhouette" and "the
# lighthouse stands silhouetted against the night sky" are good cinematography about things
# that have no face. Only the forms that bind the silhouette to a PERSON are blocking.
#
# The window is `[\w'\s]{0,20}?` — word characters, apostrophes and spaces only, so it
# reaches across "he's" and "she stands" but is stopped by any comma or full stop. An
# earlier `\W{0,8}` version was simply wrong: it required NON-word characters, so it matched
# neither "she stands silhouetted" nor "He's silhouetted", and both shipped.
_SIL_SUBJECT = (r"(?:he|she|they|man|woman|boy|girl|child|boxer|keeper|violinist|figure|"
                r"person|player|worker|dancer|singer|fighter)")
_PERSON_SILHOUETTE_RE = re.compile(
    r"^\s*silhouett"                                   # participial, inherits the subject
    r"|\b(?:his|her|their)\s+silhouette\b"
    r"|\b" + _SIL_SUBJECT + r"\b[\w'\s]{0,20}?\bsilhouett", re.IGNORECASE)

# The brief has to ask for a hidden face out loud before the planner will allow one.
_WANTS_HIDDEN_RE = re.compile(
    r"\bsilhouette|\bfrom behind\b|\bback to (?:the )?camera\b|\bfaceless\b|\bno faces?\b|"
    r"\bhidden face|\bface(?:s)? (?:hidden|obscured)|\banonymous\b|\bunidentified\b|"
    r"\bwithout showing (?:the |their |his |her )?face", re.IGNORECASE)


# --- the appearance law (L12) -----------------------------------------------------------
# A cast character is a trained LoRA. Its appearance is decided, it is not in the brief, and
# prose that guesses at it fights the weights at render time. The planner has no way to know
# this on its own, so it styles the cast like everything else in the film — measured live:
# "ww2 scene but main characters are humanoid animals, bizarrotrn is the boss of the team"
# produced "bizarrotrn Bizarro, a grizzled badger with a military uniform", assigning a
# SPECIES to the one face in the film that already has one.
#
# The detector is deliberately narrow, because the false positives are the expensive kind:
# wardrobe and rank are exactly what we WANT ("in a muddy officer's uniform", "the unit's
# commander"), and every non-cast character in an animal film is legitimately an animal. So
# it fires only on an appositive or copular phrase bound to the cast mention, and only when
# the part of that phrase describing what they ARE — the span before any wardrobe
# preposition — contains a species noun or a physical adjective.

_SPECIES = (
    r"badger|wolf|wolves|fox|vixen|bear|rabbit|hare|dog|hound|wolfhound|cat|feline|lion|"
    r"lioness|tiger|leopard|panther|jaguar|cheetah|stoat|weasel|otter|ferret|mink|marten|"
    r"mouse|mice|rat|squirrel|hedgehog|mole|shrew|boar|pig|hog|sow|stag|deer|doe|elk|moose|"
    r"goat|ram|sheep|ewe|bull|ox|cow|horse|stallion|mare|donkey|mule|zebra|"
    r"hawk|falcon|eagle|owl|raven|crow|rook|magpie|sparrow|finch|heron|stork|crane|gull|"
    r"duck|drake|goose|swan|rooster|cockerel|hen|chicken|turkey|pigeon|dove|"
    r"frog|toad|newt|lizard|gecko|snake|serpent|adder|viper|turtle|tortoise|crocodile|"
    r"fish|shark|whale|dolphin|seal|walrus|penguin|"
    r"ape|monkey|gorilla|chimp|chimpanzee|baboon|lemur|"
    r"dragon|griffin|gryphon|centaur|minotaur|satyr|faun|"
    r"elf|dwarf|orc|goblin|troll|ogre|gnome|hobbit|halfling|"
    r"vampire|werewolf|zombie|skeleton|ghost|demon|angel|deity|god|goddess|"
    r"robot|android|droid|cyborg|automaton|alien|extraterrestrial|mutant|"
    r"anthropomorphic|humanoid|creature|beast|critter|animal"
)
_SPECIES_RE = re.compile(r"\b(?:%s)s?\b" % _SPECIES, re.IGNORECASE)

_PHYS_ADJ = (
    r"old|older|elderly|aged|ancient|young|younger|youthful|teenage|teenaged|adolescent|"
    r"middle-aged|middleaged|boyish|girlish|weathered|grizzled|wizened|craggy|"
    r"tall|short|stocky|burly|brawny|hulking|lean|lanky|slim|slender|skinny|thin|scrawny|"
    r"wiry|muscular|muscled|husky|heavyset|heavy-set|stout|portly|obese|fat|plump|petite|"
    r"broad-shouldered|barrel-chested|squat|gaunt|"
    r"bald|balding|shaven|clean-shaven|bearded|moustached|mustachioed|whiskered|"
    r"long-haired|short-haired|red-haired|grey-haired|gray-haired|white-haired|"
    r"dark-haired|fair-haired|blonde|blond|brunette|redheaded|red-headed|ponytailed|"
    r"blue-eyed|green-eyed|brown-eyed|dark-eyed|wide-eyed|hollow-eyed|one-eyed|"
    r"scarred|freckled|wrinkled|pockmarked|jowly|chiselled|chiseled|square-jawed|"
    r"hook-nosed|snub-nosed|pale-skinned|dark-skinned|olive-skinned|ruddy|swarthy|"
    r"handsome|beautiful|pretty|ugly|homely|striking|"
    r"furred|furry|fur-covered|feathered|scaly|striped|spotted|shaggy|bristled|"
    r"black-furred|brown-furred|grey-furred|gray-furred|white-furred"
)
_PHYS_ADJ_RE = re.compile(r"\b(?:%s)\b" % _PHYS_ADJ, re.IGNORECASE)

# What the character WEARS, CARRIES or DOES. Never a violation, always desirable.
_WARDROBE_SPLIT_RE = re.compile(
    r"\b(?:in|wearing|dressed|clad|carrying|holding|armed|with)\b", re.IGNORECASE)
_WARDROBE_LEAD_RE = re.compile(
    r"^\s*(?:in|wearing|dressed|clad|carrying|holding|armed)\b", re.IGNORECASE)


def _cast_mention_re(trigger: str, name: str) -> Optional[Any]:
    names = [n for n in (trigger, name) if n]
    if not names:
        return None
    alt = "|".join(re.escape(n) for n in names)
    return r"\b(?:%s)\b(?:\s+(?:%s)\b)?" % (alt, alt)


def _appearance_violations(text: str, trigger: str, name: str = "") -> List[str]:
    """Appearance claims bound to a cast mention. Empty list = the prose is lawful."""
    mention = _cast_mention_re(trigger, name)
    if not mention or not text:
        return []
    pat = re.compile(r"%s\s*(?:,\s*|\s+(?:is|as)\s+)((?:a|an|the)\s+[^.;]{0,80})" % mention,
                     re.IGNORECASE)
    out: List[str] = []
    for m in pat.finditer(text):
        clause = m.group(1)
        head = re.sub(r"^(?:a|an|the)\s+", "", clause, flags=re.IGNORECASE)
        if _WARDROBE_LEAD_RE.match(head):
            continue
        being = _WARDROBE_SPLIT_RE.split(head, maxsplit=1)[0]
        if _SPECIES_RE.search(being) or _PHYS_ADJ_RE.search(being):
            out.append(clause.strip())
    return out


def _neutralise_appearance(text: str, trigger: str, name: str = "",
                           subject_noun: str = "") -> Tuple[str, List[str]]:
    """Last resort, after a re-plan has already been spent. Returns (text, cuts).

    TOKEN SCRUB, not excision. Two designs were tried and thrown away first: cutting the
    appositive ate the sentence's verb ("Bizarro, with a military uniform leans over the
    map table"), and swapping the whole appositive for the canon noun could not be bounded
    correctly when the clause held its own comma ("a tall, broad-shouldered man") — it
    either truncated the clause or swallowed the action after it.

    Replacing the offending WORDS in place cannot do either, because it restructures
    nothing: the species noun becomes the bundle's canon noun, physical adjectives are
    deleted, and the verb, the wardrobe and the rest of the sentence are untouchable.
    """
    mention = _cast_mention_re(trigger, name)
    if not mention or not text:
        return text, []
    noun = subject_noun or "figure"
    cuts: List[str] = []
    win = re.compile(r"(%s\s*(?:,\s*|\s+(?:is|as)\s+)(?:a|an|the)\s+)([^.;]{0,90})" % mention,
                     re.IGNORECASE)

    def fix(m):
        lead, body = m.group(1), m.group(2)
        being = _WARDROBE_SPLIT_RE.split(body, maxsplit=1)[0]
        tail = body[len(being):]
        if not (_SPECIES_RE.search(being) or _PHYS_ADJ_RE.search(being)):
            return m.group(0)
        cuts.append(being.strip().rstrip(","))
        fixed = _SPECIES_RE.sub(noun, being)
        fixed = _PHYS_ADJ_RE.sub("", fixed)
        # "humanoid fox" carries TWO species tokens and would become "man man".
        fixed = re.sub(r"\b%s(?:\s+%s\b)+" % (re.escape(noun), re.escape(noun)),
                       noun, fixed, flags=re.IGNORECASE)
        fixed = re.sub(r"\s*,\s*(?=,|$)", "", fixed)
        fixed = re.sub(r"^[\s,]+", "", fixed)
        fixed = re.sub(r"\s*,\s*", ", ", fixed)
        fixed = re.sub(r"\s{2,}", " ", fixed).strip() or noun
        return lead + fixed + ((" " + tail.lstrip()) if tail.strip() else "")

    out = win.sub(fix, text)
    out = re.sub(r"\ban (?=[bcdfghjklmnpqrstvwxyz])", "a ", out, flags=re.IGNORECASE)
    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r"\s+([,.])", r"\1", out)
    return out, cuts


# --- the speech law (L13) ---------------------------------------------------------------
# Measured live on the same film: shot 1 read "He explains the mission, his voice low and
# authoritative" with no line written anywhere, and the soundscape carried "the quiet murmur
# of other animals". The <d> gating did its job and kept the voice LoRA OFF — correctly,
# there were no words — but the prompt still TOLD the model a man was speaking. Given speech
# to render and no words to render, the audio branch babbles. The owner's verdict on the
# clip was "talking gibberish".
#
# So a shot is SPOKEN (the words are present, which is also what flips the voice on) or it
# is SILENT (nothing may imply speech). Describing speech without providing it is the one
# combination that is always wrong.

# `brief` is deliberately NOT `brief(?:s|ing)?`: the first live run flagged "a dimly lit
# briefing room" three times, because a briefing room is a room. The verb needs an object.
_SPEECH_VERB_RE = re.compile(
    r"\b(?:explain(?:s|ing)?|brief(?:s|ed)\b|briefing (?:the|his|her|their|them|us)\b|"
    r"describ(?:es|ing)|discuss(?:es|ing)|"
    r"talk(?:s|ing)?|speak(?:s|ing)?|says?|saying|tell(?:s|ing)?|address(?:es|ing)?|"
    r"order(?:s|ing)?|command(?:s|ing)?|instruct(?:s|ing)?|argu(?:es|ing)|"
    r"murmur(?:s|ing)?|mutter(?:s|ing)?|whisper(?:s|ing)?|mumbl(?:es|ing)|"
    r"announc(?:es|ing)|declar(?:es|ing)|recit(?:es|ing)|narrat(?:es|ing)|"
    r"chat(?:s|ting)?|converse(?:s)?|conversing|reply|replies|replying|"
    r"answer(?:s|ing)?|ask(?:s|ing)?|shout(?:s|ing)?|call(?:s|ing) out|"
    r"read(?:s|ing)? (?:it )?(?:aloud|out))\b"
    # SUNG counts: a singing mouth with no lyric babbles exactly like a
    # speaking one. Bird guards keep "birds singing" scenery out of the law.
    r"|(?<!birds )(?<!bird )\b(?:sing(?:s|ing)|chant(?:s|ing))\b", re.IGNORECASE)

# Describing a voice is describing speech, even with no verb.
_VOICE_DESC_RE = re.compile(
    r"\b(?:his|her|their|the|a|its)\s+voice\b"
    r"|\bin a (?:low|quiet|soft|hushed|loud|steady|calm|firm|gravelly|hoarse|deep)\s+voice\b"
    r"|\bvoice (?:low|quiet|soft|steady|calm|firm|even|hard)\b"
    r"|\bmid-sentence\b|\bmid-speech\b", re.IGNORECASE)

# Speech-shaped AUDIO. The soundscape babbles from these on its own.
_SPEECH_AUDIO_RE = re.compile(
    r"\b(?:murmur(?:s|ing)?|chatter(?:ing)?|voices?|conversation(?:s)?|talking|speech|"
    r"dialogue|whisper(?:s|ing)?|mutter(?:s|ing)?|shouting|shouts|yelling|calls?|"
    r"chant(?:s|ing)?|singing|song|barked orders?|radio chatter)\b", re.IGNORECASE)

# The soundscape's own way of saying "silent", which the H3 exemplar already uses. Its
# presence means the model has consciously chosen silence, so the audio cue check is moot.
_SILENCE_DECLARED_RE = re.compile(
    r"\bnobody speaks\b|\bno voice is heard\b|\bno one speaks\b|\bno speech\b|"
    r"\bno dialogue\b|\bnobody says\b", re.IGNORECASE)

# A NEGATED cue is the opposite of a violation — "no voices", "without chatter" and "not a
# word is spoken" are how a soundscape says silence, and the H3 exemplars use exactly that
# form. Matching them as speech was the first false positive this law produced: it fired on
# "Steady rain on stone, a distant gutter running, no voices."
_NEGATED_CUE_RE = re.compile(
    r"\b(?:no|not|without|never|nobody|none|silent|silence)\b[\w\s]{0,12}$", re.IGNORECASE)


def _unnegated_speech_audio(sound: str) -> Optional[str]:
    """The first speech cue in `sound` that is NOT negated, or None."""
    for m in _SPEECH_AUDIO_RE.finditer(sound or ""):
        if not _NEGATED_CUE_RE.search(sound[:m.start()]):
            return m.group(0)
    return None

_SILENCE_SENTENCE = "Nobody speaks and no voice is heard at any point."

# Silent stand-ins, for the mechanical fallback only. Each keeps the beat and removes the
# claim that a mouth is moving.
_SILENT_VERB_SWAP = (
    (re.compile(r"\bexplain(?:s|ing)?\b|\bbrief(?:s|ing)?\b|\bdescrib(?:es|ing)\b"
                r"|\bdiscuss(?:es|ing)\b|\bgoes over\b", re.IGNORECASE), "studies"),
    (re.compile(r"\border(?:s|ing)?\b|\bcommand(?:s|ing)?\b|\binstruct(?:s|ing)?\b",
                re.IGNORECASE), "signals"),
    (re.compile(r"\bmurmur(?:s|ing)?\b|\bmutter(?:s|ing)?\b|\bwhisper(?:s|ing)?\b"
                r"|\bmumbl(?:es|ing)\b", re.IGNORECASE), "leans in"),
    (re.compile(r"\btalk(?:s|ing)?\b|\bspeak(?:s|ing)?\b|\bchat(?:s|ting)?\b"
                r"|\bargu(?:es|ing)\b|\bconvers(?:es|ing)\b", re.IGNORECASE), "gestures"),
    (re.compile(r"\bsays?\b|\btell(?:s|ing)?\b|\bannounc(?:es|ing)\b|\bdeclar(?:es|ing)\b"
                r"|\baddress(?:es|ing)?\b|\brecit(?:es|ing)\b|\bnarrat(?:es|ing)\b"
                r"|\bask(?:s|ing)?\b|\banswer(?:s|ing)?\b|\brepl(?:y|ies|ying)\b"
                r"|\bshout(?:s|ing)?\b", re.IGNORECASE), "looks up"),
    # A wordless singing shot must not keep the claim that a mouth carries a
    # melody — swaying keeps the musical beat in the body instead.
    (re.compile(r"(?<!birds )(?<!bird )\bsing(?:s|ing)\b|\bchant(?:s|ing)\b",
                re.IGNORECASE), "sways"),
)


# DIALOGUE IN THE WRONG CLOTHES. The first live run under this law turned up a failure the
# owner's clip had not shown: four of twelve shots DID carry real spoken lines — "He says,
# 'Gentlemen, we have a situation'" — written as ordinary prose quotes instead of the <d>
# form. That is worse than it looks. The words are there, so silencing the shot would throw
# away the director's line; but no <d> tag means the voice gate never flips, so the trained
# voice stays off and the audio branch improvises around words it was never handed properly.
# The fix is a CONVERSION, not a deletion: same words, correct wrapper, gate on.
_PROSE_DIALOGUE_RE = re.compile(
    r"(\b(?:says?|said|replies|replied|answers?|answered|asks?|asked|adds?|added|"
    r"calls?|called|shouts?|shouted|whispers?|whispered|murmurs?|murmured|"
    r"orders?|ordered|announces?|announced|declares?|declared)\b\s*[,:]?\s*)"
    r"[\"'“‘]([^\"'”’]{3,220})[\"'”’]")

_JAW_STOP = " His jaw ceases speaking motion and his mouth settles closed."


def _has_dialogue(text: str) -> bool:
    """A <d> tag with actual words in it. An empty tag is not a line."""
    for m in _D_TAG_RE.finditer(text or ""):
        if (m.group(1) or "").strip():
            return True
    return False


def _prose_dialogue(text: str) -> List[str]:
    """Spoken lines written as prose quotes instead of <d> tags."""
    if _has_dialogue(text):
        return []
    return [m.group(2).strip() for m in _PROSE_DIALOGUE_RE.finditer(text or "")]


def _convert_prose_dialogue(text: str) -> Tuple[str, int]:
    """Rewrap prose-quoted lines as <d>[English] ...</d>. Returns (text, count)."""
    if _has_dialogue(text):
        return text, 0
    n = [0]

    def repl(m):
        n[0] += 1
        return "%s<d>[English] %s</d>" % (m.group(1), m.group(2).strip())

    out = _PROSE_DIALOGUE_RE.sub(repl, text or "")
    if n[0] and "ceases speaking motion" not in out:
        out = out.rstrip()
        # A closing </d> already ends the sentence — appending a full stop to it produces
        # "</d>." and the tag is the punctuation.
        if not out.endswith((".", "!", "?", ">")):
            out += "."
        out += _JAW_STOP
    return out, n[0]


def _speech_violations(desc: str, sound: str) -> List[str]:
    """Speech implied but never written. Empty list = the shot is honestly silent or spoken."""
    if _has_dialogue(desc):
        return []
    out: List[str] = []
    quoted = _prose_dialogue(desc or "")
    if quoted:
        # The line EXISTS — this is a form error, not a silence error, and it is reported
        # on its own so the fix is "rewrap it", never "delete it".
        return ["wrote the spoken line %r as prose quotes instead of <d>[English] ...</d>, "
                "so the voice never switches on" % quoted[0][:60]]
    m = _SPEECH_VERB_RE.search(desc or "")
    if m:
        out.append("speech verb %r with no spoken line" % m.group(0))
    m = _VOICE_DESC_RE.search(desc or "")
    if m:
        out.append("describes a voice (%r) with no spoken line" % m.group(0))
    if not _SILENCE_DECLARED_RE.search(sound or ""):
        cue = _unnegated_speech_audio(sound or "")
        if cue:
            out.append("soundscape carries speech (%r) with no spoken line" % cue)
    return out


def _neutralise_speech(desc: str, sound: str) -> Tuple[str, str, List[str]]:
    """Last resort: stage the shot silent rather than let it babble. (desc, sound, notes)."""
    notes: List[str] = []
    out_desc = desc or ""
    # CONVERT BEFORE SILENCING. A shot whose line is merely in the wrong wrapper keeps its
    # line — rewrapping flips the voice gate on and the beat survives intact. Silencing it
    # would delete words the director wrote, which is a worse outcome than the bug.
    converted, n_conv = _convert_prose_dialogue(out_desc)
    if n_conv:
        notes.append("rewrapped %d prose-quoted line(s) as <d>[English] ...</d>, so the "
                     "voice switches on" % n_conv)
        return converted, (sound or ""), notes
    for pat, repl in _SILENT_VERB_SWAP:
        new = pat.sub(repl, out_desc)
        if new != out_desc:
            notes.append("silenced a speech verb")
            out_desc = new
    # "his voice low and authoritative" and friends: drop the clause, keep the sentence.
    new = re.sub(r",\s*(?:his|her|their|its)\s+voice\b[^.;]{0,60}", "", out_desc,
                 flags=re.IGNORECASE)
    if new != out_desc:
        notes.append("dropped a voice description")
        out_desc = new
    out_sound = sound or ""
    if not _SILENCE_DECLARED_RE.search(out_sound):
        # PER FRAGMENT, not per sentence. A soundscape reads "Boots on wet planks, the quiet
        # murmur of other animals, rain on canvas." — dropping the whole sentence for one
        # bad clause throws away the boots and the rain, which were the good half of the
        # line and exactly what should carry the shot once the voices are gone.
        dropped = False
        sentences: List[str] = []
        for sent in re.split(r"(?<=[.!?])\s+", out_sound):
            if not sent.strip():
                continue
            if not _unnegated_speech_audio(sent):
                sentences.append(sent.strip())
                continue
            frags = [f.strip() for f in sent.split(",")]
            keep = [f for f in frags if f and not _unnegated_speech_audio(f)]
            dropped = dropped or len(keep) != len([f for f in frags if f])
            rebuilt = ", ".join(keep).strip(" .")
            if rebuilt:
                sentences.append(rebuilt + ".")
        if dropped:
            notes.append("removed speech from the soundscape")
        out_sound = " ".join(sentences).strip()
        out_sound = (out_sound + " " + _SILENCE_SENTENCE).strip() if out_sound \
            else _SILENCE_SENTENCE
    out_desc = re.sub(r"\s{2,}", " ", out_desc)
    out_desc = re.sub(r"\s+([,.])", r"\1", out_desc)
    return out_desc, out_sound, notes


def _face_level(raw: Any, desc: str) -> Tuple[str, bool]:
    """Normalise the model's `face` choice. Returns (level, was_overridden).

    `none` is the one value the model can use to switch the whole face law off, so it is the
    one value that is checked against the prose. Observed: a wide shot whose description read
    "showing the woman standing beside the neon sign, silhouetted against the vibrant lights"
    was labelled `face: "none"`, which disabled the scrub and shipped the silhouette. If
    there is a person in the description there is a face to protect, whatever the label says.
    """
    k = str(raw or "").strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {"closeup": "close", "close_up": "close", "tight": "close", "face": "close",
               "visible": "medium", "mid": "medium", "wide": "medium", "full": "medium",
               "no_face": "none", "nobody": "none", "n/a": "none", "": "",
               "obscured": "hidden", "silhouette": "hidden", "back": "hidden"}
    k = aliases.get(k, k)
    if k not in _FACE_LAWS:
        return ("medium" if _has_person(desc) else "none"), False
    if k == "none" and _has_person(desc):
        return "medium", True
    return k, False


def _scrub_face_blocking(text: str) -> Tuple[str, List[str]]:
    """Remove clauses that hide the face. Returns (text, removed clauses).

    Clause-level, like _clean_settle: sentences are split on `.` and clauses on `,`/`;`, and
    only the offending clause is dropped, so the rest of the direction survives. If every
    clause of a sentence is blocking, the sentence goes.
    """
    removed: List[str] = []
    out_sentences: List[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text or ""):
        if not sentence.strip():
            continue
        clauses = re.split(r"(?<=[,;])\s+", sentence)
        keep = []
        for c in clauses:
            if _FACE_BLOCK_RE.search(c) or _PERSON_SILHOUETTE_RE.search(c):
                removed.append(c.strip().rstrip(",;"))
                continue
            keep.append(c)
        if keep:
            joined = " ".join(keep).strip()
            joined = re.sub(r"[,;]\s*([.!?])", r"\1", joined).strip().rstrip(",;")
            if joined and joined[-1] not in ".!?":
                joined += "."
            out_sentences.append(joined)
    return " ".join(out_sentences).strip(), removed

_NO_TEXT = "No text appears at any point."

# H3 volunteers lettering unless refused — but refusing it on a shot that IS about lettering
# is worse than not refusing at all. Observed: a title-sequence plan that spelled a word in
# mercury and then told the model that no text may appear.
#
# The only reliable signal for "typography is intended" is a SHORT QUOTED RUN. Keyword
# matching was tried and rejected: "neon sign" tripped on a documentary about repairing neon
# (where the refusal is exactly right), and single-quoted LTX dialogue tripped on every
# talking shot. Keywords now only raise a warning; they never suppress the refusal.
_DQ_RE = re.compile(r'"([^"\n]{1,32})"')
_SQ_RE = re.compile(r"(?<![A-Za-z])'([^'\n]{1,32})'")   # lookbehind skips don't / it's
_TEXT_KEYWORD_RE = re.compile(
    r"\b(?:the word|the letter|the letters|spells?|spelling|typography|lettering|"
    r"title card|subtitle|caption)\b", re.IGNORECASE)


def _typography_strings(text: str) -> List[str]:
    """Quoted runs that read as ON-SCREEN TEXT rather than dialogue or an apostrophe.

    A title is a token or a shout ("PHOSPHENE", 'P'); dialogue is a sentence with spaces and
    mixed case. The spec for on-screen text is 3-5 words and <=32 characters, which is what
    the length bound encodes.
    """
    out = [m.group(1) for m in _DQ_RE.finditer(text or "")]
    for m in _SQ_RE.finditer(text or ""):
        s = m.group(1)
        if re.search(r"[A-Za-z0-9]", s) and (" " not in s or s == s.upper()):
            out.append(s)
    return out


def _camera_key(raw: Any) -> Tuple[str, bool]:
    """Canonical camera key + whether the input had to be forced.

    The stored `camera` field must be the key that was actually rendered, not whatever the
    model typed — a shot card reading `cam=medium` (observed: the model confused the `face`
    enum with this one) while the prompt says "holds a static shot" is a lie to the user.
    """
    k = str(raw or "").strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {"locked": "static", "locked_off": "static", "tripod": "static", "none": "static",
               "push": "push_in", "zoom_in": "push_in", "dolly_in": "push_in",
               "pull": "pull_back", "pull_out": "pull_back", "zoom_out": "pull_back",
               "dolly_out": "pull_back", "handheld_sway": "handheld", "sway": "handheld",
               "shake": "handheld", "arc": "orbit", "slow_orbit": "orbit", "orbit_slow": "orbit",
               "slow_pan": "pan", "tilt": "tilt_up", "track": "tracking", "truck": "tracking"}
    k = aliases.get(k, k)
    if k in _CAMERA_SENTENCES:
        return k, False
    return "static", True


def _camera_sentence(key: Any) -> str:
    return _CAMERA_SENTENCES[_camera_key(key)[0]]


def _clean_settle(state: str) -> str:
    """A settled state describes the SUBJECT, never the camera.

    The model reliably writes "the camera stops orbiting" / "the camera holds on the scene"
    here, which is a camera instruction pasted into a clause that ends "with no new movement
    of any kind" — it contradicts itself and duplicates the camera direction. Clauses that
    talk about the camera are dropped; if that empties the phrase, there is no settle.
    """
    s = (state or "").strip().rstrip(".")
    if not s:
        return ""
    keep = [p for p in re.split(r",\s*", s) if "camera" not in p.lower()]
    return ", ".join(p for p in keep if p.strip()).strip()


def _settle_sentence(state: str) -> str:
    s = _clean_settle(state)
    if not s:
        return ""
    return ("The movement is completely finished before the shot ends, and for the last two "
            "seconds %s, with no new movement of any kind." % s)


def _has_person(text: str) -> bool:
    return bool(_PERSON_RE.search(text or ""))


def _plain_punctuation(text: str) -> str:
    """Curly quotes, en/em dashes and ellipses out.

    H3 guide 7.6: punctuation and separators the model has not seen in training can be
    rendered as literal on-screen text. The model emits U+2019 constantly ("you've"), so it
    is normalised on the way into the prompt rather than left to chance."""
    out = text
    for bad, good in list(_SMART.items()) + [("…", "..."), (" ", " ")]:
        out = out.replace(bad, good)
    return out


def _first(d: Dict[str, Any], keys: Sequence[str], default: str = "") -> str:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, (int, float)):
            return str(v)
    return default


def _split_three_fields(desc: str) -> Tuple[str, str, str]:
    """If the model pasted the whole assembled prompt into one field, take it apart.

    Plain string slicing rather than a regex: the fields are literal labels in a fixed
    order, and a lazy DOTALL regex silently swallows the later labels into the first group.
    """
    low = desc.lower()
    if "integrated_multimodal_description" not in low:
        return desc, "", ""

    def cut(text: str, label: str) -> Tuple[str, str]:
        """-> (text before label, text after 'label:')"""
        i = text.lower().find(label)
        if i < 0:
            return text, ""
        rest = text[i + len(label):].lstrip()
        if rest.startswith(":"):
            rest = rest[1:]
        return text[:i], rest.strip()

    _, body = cut(desc, "integrated_multimodal_description")
    body, music = cut(body, "non_diegetic_music")
    body, sound = cut(body, "overall_soundscape")
    # non_diegetic_music can legally appear before overall_soundscape in sloppy output;
    # the second cut above already removed it from `body`, so nothing leaks either way.
    if music and "overall_soundscape" in music.lower():
        music, sound2 = cut(music, "overall_soundscape")
        sound = sound or sound2
    return body.strip(), sound.strip(), music.strip()


def _strip_h3_markup(text: str) -> str:
    """H3 markup -> LTX prose. `<d>[English] Hi.</d>` becomes `'Hi.'`, markers go away."""
    out = _D_TAG_RE.sub(lambda m: "'%s'" % m.group(1).strip(), text)
    out = re.sub(r"\[Shot\s*\d+\](?:\s*At\s*\d\d:\d\d\.\d+,)?", "", out, flags=re.IGNORECASE)
    out = re.sub(r"\(S\d(?:,S\d)*\)", "", out)
    out = re.sub(r"</?d>", "", out)
    return re.sub(r"\s{2,}", " ", out).strip()


def _fix_unbalanced_d(text: str) -> str:
    """A stray `<d>` with no `</d>` would be spoken as literal characters. Close it."""
    opens = len(re.findall(r"<d>", text))
    closes = len(re.findall(r"</d>", text))
    if opens == closes:
        return text
    if opens > closes:
        return text + ("</d>" * (opens - closes))
    return re.sub(r"</d>", "", text, count=(closes - opens))


def _snap_duration(value: Any, engine: str, default: float) -> float:
    """An ORDINARY shot's length on its engine's grid. Never called for a One Shot: a
    take is as long as its `take_seconds` (30-120 s) and this would clamp it to 60 and,
    on H3, snap it to 15 — see coerce_spec()."""
    try:
        d = float(value)
    except (TypeError, ValueError):
        d = default
    if not (d > 0):
        d = default
    d = max(_MIN_DURATION, min(_MAX_DURATION, d))
    if engine == "h3":
        d = min(_H3_LENGTHS, key=lambda cand: (abs(cand - d), cand))
    return float(d)


# ---- ONE SHOT: carrying `take_seconds` + `beats` through from the model ----------------
# The table of legal take lengths and the beat length live in storyboard.py (the same
# numbers the panel's take_plan uses); read from there, never copied — a second copy of
# a number that has drifted once is the next drift waiting.

def _take_table() -> Tuple[Tuple[int, ...], int]:
    sb = _storyboard_module()
    secs = tuple(getattr(sb, "TAKE_SECONDS", ()) or ()) if sb is not None else ()
    beat = int(getattr(sb, "TAKE_BEAT_SECONDS", 0) or 0) if sb is not None else 0
    return secs, beat


def _coerce_take_seconds(value: Any) -> Optional[int]:
    """A One Shot's length: the NEAREST legal take (ties to the shorter), or None when the
    shot is not a take — absent, null, false, 0, "off", or unparseable. "30s" counts."""
    if value is None or isinstance(value, bool):
        return None
    try:
        secs = float(str(value).strip().lower().rstrip("s"))
    except (TypeError, ValueError):
        return None
    if not secs > 0:
        return None
    table, _ = _take_table()
    if not table:
        return None
    return int(min(table, key=lambda t: (abs(t - secs), t)))


_BEAT_PREFIX_RE = re.compile(r"^\s*(?:beat\s*\d+\s*[:.)\-]\s*|\d+\s*[:.)]\s+)", re.IGNORECASE)


def _beat_items(raw: Any) -> List[str]:
    """Whatever the model wrote for `beats`, as a flat list of strings: a list of strings,
    a list of {text|beat|description: ...} objects, a {"1": ..} object, or one
    newline-separated string. A numbering prefix the model added ("3. ", "Beat 3:") is
    the model's, not the shot's, and comes off."""
    if isinstance(raw, dict):
        raw = list(raw.values())
    items: List[str] = []
    if isinstance(raw, str):
        items = [ln.strip() for ln in raw.splitlines()]
    elif isinstance(raw, (list, tuple)):
        for b in raw:
            if isinstance(b, dict):
                items.append(_first(b, ("text", "beat", "description", "prompt", "action")))
            elif b is None:
                items.append("")
            else:
                items.append(str(b).strip())
    return [_BEAT_PREFIX_RE.sub("", b).strip() for b in items]


def _coerce_beats(raw: Any, n: int) -> List[str]:
    """Exactly `n` beat strings. Extras are dropped, missing ones are blank — the panel
    holds the previous moment on a blank beat — so a miscounted list is padded or
    trimmed, never rejected."""
    items = _beat_items(raw)[:max(0, int(n))]
    return items + [""] * (max(0, int(n)) - len(items))


def _assemble_beats(shot: Dict[str, Any], lines: Sequence[str], style: str,
                    cast: Sequence[Dict[str, str]], sb: Any) -> List[str]:
    """Each written beat in the shot's own register, as a prompt the panel can hand to a
    part ON ITS OWN: the camera law, the face law and the soundscape ride on every beat
    (a later part is rendered from its beats alone, not from the description), the
    trigger is attached the way the main prompt gets it, and the settle rides on the LAST
    written beat only — "completely finished before the shot ends" on beat 2 of 12 would
    stop the take in its tracks. Blank beats stay blank."""
    out: List[str] = []
    last = max((i for i, ln in enumerate(lines) if ln), default=-1)
    for i, line in enumerate(lines):
        if not line:
            out.append("")
            continue
        view = dict(shot, description=line,
                    settle=(shot.get("settle") or "") if i == last else "")
        out.append(_reassemble_prompt(view, style, cast, sb))
    return out


def _normalise_cast(characters: Optional[Iterable[Any]]) -> List[Dict[str, str]]:
    """Accept list_characters() records, plain ids, or {'id':..,'trigger':..} dicts."""
    out: List[Dict[str, str]] = []
    for c in characters or ():
        if isinstance(c, str):
            cid = c.strip()
            if cid:
                out.append({"id": cid, "trigger": cid, "name": cid, "description": ""})
            continue
        if not isinstance(c, dict):
            continue
        cid = str(c.get("id") or c.get("trigger") or c.get("character_id") or "").strip()
        if not cid:
            continue
        out.append({
            "id": cid,
            "trigger": str(c.get("trigger") or cid).strip(),
            "name": str(c.get("name") or cid).strip(),
            "description": str(c.get("description") or c.get("bio") or "").strip(),
            # THE ONE APPEARANCE FACT THE BUNDLE ACTUALLY KNOWS. lora-lab writes
            # `subject_noun` ("man", "woman", "person") and `pronoun` into every
            # bundle.json, and the planner never saw either — so when a film's premise said
            # "everyone is an animal" the planner had nothing to contradict it with and
            # made the trained character a badger. It is deliberately the ONLY appearance
            # word that crosses: it is canon, it comes from the bundle rather than a guess,
            # and it is what the mechanical fallback substitutes back in.
            "subject_noun": str(c.get("subject_noun") or "").strip(),
            "pronoun": str(c.get("pronoun") or "").strip(),
        })
    return out


def _match_character(raw: Any, cast: Sequence[Dict[str, str]]) -> Optional[Dict[str, str]]:
    if raw is None or not cast:
        return None
    key = str(raw).strip()
    if not key or key.lower() in ("null", "none", "n/a", ""):
        return None
    low = key.lower()
    for c in cast:
        if c["id"].lower() == low or c["trigger"].lower() == low or c["name"].lower() == low:
            return c
    # The model wrote a description ("the fighter"); accept a containment match rather than
    # silently dropping the casting the director asked for.
    for c in cast:
        if c["id"].lower() in low or (c["name"] and c["name"].lower() in low):
            return c
    return None


def _seed_for(seed_base: int, n: int) -> int:
    return int((seed_base + n * 7919) % 2147483647)


def _stable_seed(concept: str) -> int:
    h = hashlib.sha256(concept.encode("utf-8")).hexdigest()[:8]
    return int(h, 16) % 2147483647


def _compose_body(desc: str, camera: Any, settle: str, face: str = "") -> str:
    """description + the camera law + the face law + the settle law, in exemplar order.

    If the model already wrote a camera sentence or an end-state clause of its own, that is
    honoured rather than duplicated — a prompt with two camera instructions is worse than a
    prompt with the wrong one.
    """
    body = _plain_punctuation(_SHOT_MARKER_RE.sub("", (desc or "").strip())).rstrip()
    if body and body[-1] not in ".!?":
        body += "."
    if "the camera" not in body.lower():
        body += " " + _camera_sentence(camera)
    law = _FACE_LAWS.get(face or "", "")
    if law and "holds the exact angle" not in body:
        body += " " + law
    if "completely finished before the shot ends" not in body.lower():
        s = _settle_sentence(_plain_punctuation(settle or ""))
        if s:
            body += " " + s
    if _NO_TEXT.lower() not in body.lower() and not _typography_strings(body):
        body += " " + _NO_TEXT
    return body


def _assemble_h3_prompt(desc: str, sound: str, music: str,
                        camera: Any = "static", settle: str = "", face: str = "") -> str:
    """The official three-field form. `[Shot 1]` carries no timestamp — every storyboard
    shot is one continuous take, so there is never a `[Shot 2]` inside a single prompt."""
    body = _fix_unbalanced_d(_compose_body(desc, camera, settle, face))
    sound = _plain_punctuation((sound or "").strip()) or "N/A"
    music = _plain_punctuation((music or "").strip()) or "N/A"
    return (
        "integrated_multimodal_description: [Shot 1] %s\n\n"
        "overall_soundscape: %s\n\n"
        "non_diegetic_music: %s" % (body, sound, music)
    )


def _assemble_ltx_prompt(desc: str, sound: str, style: str,
                         camera: Any = "static", settle: str = "", face: str = "") -> str:
    """LTX 2.3 prose: one paragraph, master style suffix verbatim, one trailing `Audio:`
    line (the shape mlx_warm_helper's enhance addendum says LTX was trained on)."""
    body = _strip_h3_markup(_compose_body(desc, camera, settle, face)).rstrip()
    if body and body[-1] not in ".!?":
        body += "."
    st = _plain_punctuation((style or "").strip().rstrip("."))
    if st and st.lower() not in body.lower():
        body += " %s." % st
    snd = _plain_punctuation((sound or "").strip().rstrip("."))
    if snd and snd.upper() != "N/A":
        body += " Audio: %s." % snd
    return body.strip()


def default_policy(max_dim: Optional[int] = None) -> Dict[str, Any]:
    """Same shape `storyboard.new_storyboard()` produces, clamped to the machine's cap.

    validate_storyboard() rejects a policy whose longest edge exceeds `max_dim`, so the
    clamp happens here rather than being discovered at validation time.
    """
    # IMPORTED, NOT RESTATED. This held its own copy and the two had drifted:
    # storyboard.py said Draft 640x448 — what a Quick render actually delivers,
    # ffprobe-verified — and this said 640x480, a canvas the panel's own engine
    # registry lists as never delivered. The docstring above claimed both were
    # "the same shape". The main panel path masked it by keeping the board's
    # existing policy, so only a direct planner consumer would ever have been
    # handed the fictional geometry.
    # NO FALLBACK LITERAL. A second copy of the numbers was kept here "in case
    # the import fails", and a spare copy of a value that has already drifted
    # once is not insurance, it is the next drift waiting. storyboard.py is a
    # hard dependency of this module — coerce_spec() already calls into it for
    # the schema — so if it cannot be imported the honest outcome is the error,
    # not a policy nobody will ever notice is stale.
    _sb = _storyboard_module()
    if _sb is None or not hasattr(_sb, "default_policy"):
        raise PlannerError(
            "cannot import storyboard.default_policy() — the planner takes the "
            "render policy from storyboard.py and has no copy of its own")
    policy = _sb.default_policy()
    if max_dim:
        for key in ("draft", "final"):
            p = policy[key]
            longest = max(p["width"], p["height"])
            if longest > max_dim:
                scale = float(max_dim) / float(longest)
                p["width"] = max(64, int(p["width"] * scale) // 8 * 8)
                p["height"] = max(64, int(p["height"] * scale) // 8 * 8)
    return policy


def coerce_spec(
    raw: Any,
    *,
    concept: str,
    n_shots: int,
    style: str = "",
    cast: Optional[Sequence[Dict[str, str]]] = None,
    board_id: Optional[str] = None,
    engine: str = "auto",
    tier: str = "draft",
    duration_s: float = 5.0,
    seed_base: Optional[int] = None,
    max_dim: Optional[int] = None,
    created_at: Optional[int] = None,
    allow_hidden_faces: bool = False,
    storyboard_mod: Any = None,
    locations: Optional[Sequence[Dict[str, Any]]] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """Turn whatever the model returned into a schema-correct storyboard.

    Returns (spec, warnings). Never raises on bad model output — anything unusable is
    replaced by something legal and named in `warnings`, so the caller can decide whether
    the repair round is worth 40 seconds.
    """
    warnings: List[str] = []
    cast = list(cast or ())
    locs = [l for l in (locations or ()) if isinstance(l, dict) and l.get("id")]
    if seed_base is None:
        seed_base = _stable_seed(concept)

    if not isinstance(raw, dict):
        raw = {}
        warnings.append("model returned no JSON object")

    shots_raw = raw.get("shots")
    if not isinstance(shots_raw, list):
        for v in raw.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                shots_raw = v
                warnings.append("shots were under a differently-named key")
                break
    if not isinstance(shots_raw, list):
        shots_raw = []
        warnings.append("model returned no shots array")

    shots: List[Dict[str, Any]] = []
    for idx, s in enumerate(shots_raw):
        if not isinstance(s, dict):
            warnings.append("shot %d was not an object" % (idx + 1))
            continue

        desc = _first(s, _DESC_KEYS)
        sound = _first(s, _SOUND_KEYS)
        music = _first(s, _MUSIC_KEYS, "N/A")
        if "integrated_multimodal_description" in desc.lower():
            desc, split_sound, split_music = _split_three_fields(desc)
            sound = split_sound or sound
            music = split_music or music
            warnings.append("shot %d pasted the assembled prompt into one field" % (idx + 1))
        desc = _SHOT_MARKER_RE.sub("", desc.strip())
        if not desc:
            warnings.append("shot %d had an empty description and was dropped" % (idx + 1))
            continue

        char = _match_character(s.get("character_id") or s.get("character"), cast)
        if s.get("character_id") and char is None and cast:
            warnings.append("shot %d named an unknown character %r — recast as uncast"
                            % (idx + 1, s.get("character_id")))

        if engine in ("h3", "ltx"):
            eng = engine
        else:
            # auto: a trained Phosphene character is an LTX LoRA, and identity is the one
            # thing H3 cannot do. Everything else goes to H3.
            eng = "ltx" if char else "h3"

        n = len(shots) + 1
        # ONE SHOT? `take_seconds` snaps to the nearest legal take; `beats` shapes up to
        # exactly take/5 entries (padded or trimmed, never rejected). A take with no
        # written beat at all is not a take — the model put the key on an ordinary
        # shot — and is planned as one.
        take_raw = s.get("take_seconds") if "take_seconds" in s else s.get("take")
        take = _coerce_take_seconds(take_raw)
        beat_lines: List[str] = []
        if take:
            _, beat_secs = _take_table()
            beat_lines = _coerce_beats(s.get("beats"), take // max(1, beat_secs))
            if not any(beat_lines):
                warnings.append("shot %d asked for a One Shot of %d s but wrote no beats — "
                                "planned as an ordinary shot" % (idx + 1, take))
                take, beat_lines = None, []
            else:
                if str(take_raw).strip().lower().rstrip("s") != str(take):
                    warnings.append("shot %d: take_seconds %r is not a legal take length — "
                                    "snapped to %d s" % (idx + 1, take_raw, take))
                given = len(_beat_items(s.get("beats")))
                if given != len(beat_lines):
                    warnings.append("shot %d wrote %d beats for a %d s One Shot (%d expected) "
                                    "— %s" % (idx + 1, given, take, len(beat_lines),
                                              "extras dropped" if given > len(beat_lines)
                                              else "the missing ones hold the previous moment"))
        if take:
            # A One Shot is as long as its take. _snap_duration would clamp it to 60 and,
            # on H3, snap it to 15 — the exact thing that must not happen to it.
            dur = float(take)
        else:
            dur = _snap_duration(s.get("duration_s") or s.get("duration") or s.get("seconds"),
                                 eng, duration_s)
        camera, cam_forced = _camera_key(s.get("camera") or s.get("camera_move"))
        if cam_forced and str(s.get("camera") or "").strip():
            warnings.append("shot %d asked for camera %r, which is not one of %s — locked off"
                            % (idx + 1, str(s.get("camera")).strip(), ", ".join(CAMERA_KEYS)))
        settle_raw = str(s.get("settle") or s.get("end_state") or s.get("ending") or "").strip()
        settle = _clean_settle(settle_raw)
        if settle_raw and not settle:
            warnings.append("shot %d described the camera instead of an end state" % (idx + 1))

        # --- the face law ---------------------------------------------------------------
        face, face_forced = _face_level(
            s.get("face") or s.get("face_visible") or s.get("framing"), desc)
        if face_forced:
            warnings.append("shot %d said no face, but a person is on screen — the face law "
                            "was applied anyway" % (idx + 1))
        if face == "hidden" and not allow_hidden_faces:
            face = "medium"
            warnings.append("shot %d asked to hide the face; the brief did not ask for that, "
                            "so the face is kept visible" % (idx + 1))
        if face in ("close", "medium"):
            desc, cut_d = _scrub_face_blocking(desc)
            settle, cut_s = _scrub_face_blocking(settle)
            settle = settle.rstrip(".")
            for c in (cut_d + cut_s):
                warnings.append("shot %d: removed face-hiding framing %r" % (idx + 1, c[:70]))
            if not desc.strip():
                warnings.append("shot %d was entirely face-hiding and was dropped" % (idx + 1))
                continue
        # --- the appearance law (L12) and the speech law (L13) --------------------------
        # Detected here so a direct caller of coerce_spec() is told, but NOT repaired here:
        # the plan loop gets to spend one targeted re-plan first, and a model rewrite beats
        # a regex rewrite every time it works. The mechanical fallback runs only after that
        # re-plan has been spent — see _enforce_laws().
        if char:
            for clause in _appearance_violations(desc, char["trigger"], char.get("name", "")):
                warnings.append("shot %d described %s's appearance (%r) — a cast character's "
                                "look belongs to the trained face, not the prompt"
                                % (idx + 1, char["trigger"], clause[:60]))
        for problem in _speech_violations(desc, sound):
            warnings.append("shot %d %s" % (idx + 1, problem))

        if _TEXT_KEYWORD_RE.search(desc) and not _DQ_RE.search(desc):
            warnings.append("shot %d names on-screen text but does not put it in double "
                            "quotes — H3 renders described strings as letter-shaped noise"
                            % (idx + 1))

        if eng == "h3":
            prompt = _assemble_h3_prompt(desc, sound, music, camera, settle, face)
        else:
            prompt = _assemble_ltx_prompt(desc, sound, style, camera, settle, face)

        shot: Dict[str, Any] = {
            "n": n,
            "title": str(s.get("title") or s.get("label") or "Shot %d" % n).strip()[:80],
            "mode": "character" if char else "text",
            "engine": eng,
            "tier": tier,
            "prompt": prompt,
            "duration_s": dur,
            "seed": _seed_for(seed_base, n),
            "refs": [],
            "status": "pending",
            # The creative payload is kept alongside the assembled prompt so a re-roll can
            # edit one field without parsing it back out of the finished string.
            "description": desc,
            "camera": camera,
            "face": face,
            "settle": settle,
            "soundscape": sound,
            "music": music,
        }
        # Which place this shot happens in. Matched on the NAME the model was
        # given; if it answers with something not on the list, or answers with
        # nothing, a film with exactly ONE location falls back to that one —
        # which is the common case and the one the whole feature is for.
        if locs:
            want = str(s.get("location") or s.get("place") or "").strip().lower()
            hit = next((l for l in locs
                        if want and want in (str(l.get("name") or "").lower(),
                                             str(l.get("id") or "").lower())), None)
            if hit is None and len(locs) == 1:
                hit = locs[0]
            if hit is not None:
                shot["location_id"] = hit["id"]
                # WHICH WAY THE CAMERA FACES, and where the eyes go. Both are
                # matched against the floor plan rather than trusted: an
                # unknown view id is a HARD validator error and an eyeline
                # outside the vocabulary is another, so nothing that did not
                # resolve is ever written onto a shot.
                _apply_geography(shot, s, hit, idx + 1, warnings)

        if char:
            shot["character_id"] = char["id"]
            shot["trigger"] = char["trigger"]
            if storyboard_mod is not None and hasattr(storyboard_mod, "ensure_trigger"):
                shot["prompt"] = storyboard_mod.ensure_trigger(shot["prompt"], char["trigger"])
            elif not re.search(r"\b%s\b" % re.escape(char["trigger"]), shot["prompt"]):
                shot["prompt"] = "%s %s" % (char["trigger"], shot["prompt"])
            if eng == "h3":
                # An H3 T2VA prompt must begin with the field label, so a prepended trigger
                # would be illegal (grammar rule 5). Put it inside the description instead.
                shot["prompt"] = _assemble_h3_prompt(
                    "%s The on-screen subject is %s." % (desc, char["trigger"]),
                    sound, music, camera, settle, face)
        if take:
            # The face law applies to every beat, not just the description: a later part
            # is rendered from its beats alone, so a silhouette written into beat 9 is a
            # silhouette on screen.
            if face in ("close", "medium"):
                cleaned: List[str] = []
                for bi, line in enumerate(beat_lines):
                    l2, cut = _scrub_face_blocking(line) if line else (line, [])
                    for c in cut:
                        warnings.append("shot %d, beat %d: removed face-hiding framing %r"
                                        % (idx + 1, bi + 1, c[:70]))
                    cleaned.append(l2)
                beat_lines = cleaned
            shot["take_seconds"] = take
            # `beat_lines` is what the model wrote (what a re-plan is shown); `beats` is
            # each of them assembled in the shot's register (what the panel renders) —
            # the same raw/assembled pair `description` / `prompt` already keep.
            shot["beat_lines"] = beat_lines
            shot["beats"] = _assemble_beats(shot, beat_lines, style, cast, storyboard_mod)
            shot["frames"] = take * 24 + 1
            # The prompt the panel reads first — the light lock, the file name, part 1 —
            # is the first beat, as collapse_take does for a whole-film take.
            if shot["beats"][0]:
                shot["prompt"] = shot["beats"][0]
        shots.append(shot)

    if len(shots) != n_shots:
        warnings.append("asked for %d shots, model returned %d" % (n_shots, len(shots)))
    cams = {_camera_sentence(s.get("camera")) for s in shots}
    if len(shots) > 2 and len(cams) == 1:
        # Not corrected — overriding a director's camera is worse than telling them. But it
        # is the single most common small-model tell, so the UI gets to say so.
        warnings.append("every shot uses the same camera behaviour (%s) — consider varying it"
                        % (shots[0].get("camera") or "static"))
    if len(shots) > 2 and not any((s.get("settle") or "").strip() for s in shots):
        warnings.append("no shot named an end state — H3 invents motion in the tail")

    title = str(raw.get("title") or raw.get("film_title") or "").strip()
    if not title:
        title = concept.strip()[:60] or "Untitled storyboard"
        warnings.append("model returned no title")

    created = int(created_at if created_at is not None else time.time())
    bid = board_id or "sb-%d-%s" % (created, hashlib.sha1(
        ("%s|%d" % (concept, seed_base)).encode("utf-8")).hexdigest()[:6])

    spec = {
        "schema": getattr(storyboard_mod, "SCHEMA_VERSION", 1) if storyboard_mod else 1,
        "id": bid,
        "title": title,
        "created_at": created,
        "cast": [{"id": c["id"], "trigger": c["trigger"], "name": c["name"]}
                 for c in cast if any(s.get("character_id") == c["id"] for s in shots)],
        "policy": default_policy(max_dim),
        "shots": shots,
    }
    # THE PLAN CARRIES THE PLACES IT NAMES. Shots have been stamped with
    # `location_id` since locations existed, and the spec they live in never
    # carried `locations` — so `validate_storyboard()` saw a shot pointing at a
    # place the board did not have and returned `unknown_location` for EVERY
    # shot. Every plan with a location in the brief spent its one repair
    # round-trip on a fault no model could fix and came back `invalid_plan`.
    # It survived because the panel keeps its own copy of the locations and
    # patches them onto the board after adoption, so the only visible symptom
    # was planning failing whenever the user filled the Locations box in.
    if locs:
        spec["locations"] = [dict(l) for l in locs]
    return spec, warnings


# --------------------------------------------------------------------------------------
# The model process
# --------------------------------------------------------------------------------------

class PlannerSession(object):
    """A short-lived child process holding the planner model.

    Use it as a context manager when you want several generations to share one load
    (`plan_film()` does this internally for the repair round). It is NEVER left running:
    `plan_film()` releases in a finally, and `__exit__` releases too.

        with PlannerSession() as s:
            out = s.generate(system="...", user="...")
        # model is gone here, guaranteed by process exit
    """

    def __init__(self,
                 model_path: Optional[Any] = None,
                 python_exe: Optional[Any] = None,
                 timeout_s: float = DEFAULT_TIMEOUT_S):
        self.model_path = Path(model_path or DEFAULT_MODEL_PATH)
        self.python_exe = Path(python_exe or WORKER_PYTHON)
        self.timeout_s = float(timeout_s)
        self.proc = None  # type: Optional[subprocess.Popen]
        self.stats = {
            "model_path": str(self.model_path),
            "python": str(self.python_exe),
            "load_s": None,
            "calls": 0,
            "gen_s_total": 0.0,
            "prompt_tokens": 0,
            "output_tokens": 0,
            "peak_rss_bytes": 0,
            "mx_peak_bytes": 0,
            "released": False,
        }

    # -- lifecycle ----------------------------------------------------------------
    def _spawn(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            return
        if not self.model_path.exists():
            raise PlannerError(
                "planner model not found at %s — set LTX_STORYBOARD_PLANNER to an "
                "mlx-lm-loadable directory" % self.model_path)
        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        self.proc = subprocess.Popen(
            [str(self.python_exe), str(Path(__file__).resolve()), "--serve"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env, cwd=str(ROOT), text=True, bufsize=1,
        )
        self.stats["released"] = False

    def release(self) -> Dict[str, Any]:
        """Kill the child and reclaim every byte. Idempotent; safe to call twice."""
        proc = self.proc
        self.proc = None
        if proc is None:
            self.stats["released"] = True
            return self.stats
        try:
            if proc.poll() is None and proc.stdin is not None:
                try:
                    proc.stdin.write(json.dumps({"action": "exit"}) + "\n")
                    proc.stdin.flush()
                except (OSError, ValueError):
                    pass
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        finally:
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                try:
                    if stream is not None:
                        stream.close()
                except Exception:
                    pass
        # Independent of anything the child said about itself.
        try:
            import resource
            rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            self.stats["children_maxrss_bytes"] = int(rss if sys.platform == "darwin" else rss * 1024)
        except Exception:
            pass
        self.stats["released"] = True
        return self.stats

    # Context manager so "load -> plan -> release" cannot be forgotten.
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
        return False

    unload = release   # alias: the panel's vocabulary is "unload"

    # -- generation ---------------------------------------------------------------
    def generate(self, system: str, user: str,
                 max_tokens: int = DEFAULT_MAX_TOKENS,
                 temperature: float = DEFAULT_TEMPERATURE,
                 seed: int = 10) -> Dict[str, Any]:
        self._spawn()
        req = {
            "action": "generate",
            "model_path": str(self.model_path),
            "system": system,
            "user": user,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "seed": int(seed),
        }
        assert self.proc is not None and self.proc.stdin is not None
        try:
            self.proc.stdin.write(json.dumps(req) + "\n")
            self.proc.stdin.flush()
        except (OSError, ValueError) as exc:
            raise PlannerError("planner subprocess died before it could be asked: %s" % exc)

        deadline = time.time() + self.timeout_s
        assert self.proc.stdout is not None
        while True:
            if self.proc.poll() is not None:
                err = ""
                try:
                    if self.proc.stderr is not None:
                        err = self.proc.stderr.read()[-2000:]
                except Exception:
                    pass
                raise PlannerError("planner subprocess exited (%s)%s"
                                   % (self.proc.returncode, (": " + err) if err else ""))
            line = self.proc.stdout.readline()
            if not line:
                raise PlannerError("planner subprocess closed its output")
            if line.startswith(_SENTINEL):
                resp = json.loads(line[len(_SENTINEL):])
                break
            if time.time() > deadline:
                raise PlannerError("planner timed out after %.0fs" % self.timeout_s)

        if resp.get("error"):
            raise PlannerError("planner model error: %s" % resp["error"])

        st = self.stats
        if st["load_s"] is None and resp.get("load_s") is not None:
            st["load_s"] = resp["load_s"]
        st["calls"] += 1
        st["gen_s_total"] = round(st["gen_s_total"] + float(resp.get("gen_s") or 0.0), 2)
        st["prompt_tokens"] += int(resp.get("prompt_tokens") or 0)
        st["output_tokens"] += int(resp.get("output_tokens") or 0)
        st["peak_rss_bytes"] = max(st["peak_rss_bytes"], int(resp.get("peak_rss_bytes") or 0))
        st["mx_peak_bytes"] = max(st["mx_peak_bytes"], int(resp.get("mx_peak_bytes") or 0))
        return resp


# --------------------------------------------------------------------------------------
# plan_film
# --------------------------------------------------------------------------------------

def is_plan_error(result: Any) -> bool:
    """True when plan_film() returned a structured error rather than a film spec."""
    return isinstance(result, dict) and result.get("ok") is False


def _error(kind: str, message: str, *, hint: str = "", problems: Optional[Sequence[str]] = None,
           raw: str = "", meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """A failure the UI can render as a sentence and a fix-list. Never a traceback."""
    return {
        "ok": False,
        "error": {
            "kind": kind,
            "message": message,
            "hint": hint,
            "problems": list(problems or ()),
            "raw_excerpt": (raw or "")[:1200],
        },
        "_planner": dict(meta or {}),
    }


def _parse_feedback(feedback: Any) -> Tuple[str, Optional[int], str]:
    """-> (mode, shot_n, note). mode is 'none' | 'film' | 'shot'."""
    if feedback is None:
        return "none", None, ""
    if isinstance(feedback, dict):
        note = str(feedback.get("note") or feedback.get("text") or feedback.get("feedback") or "").strip()
        shot = feedback.get("shot", feedback.get("shot_n", feedback.get("n")))
        if shot is not None:
            try:
                return "shot", int(shot), note
            except (TypeError, ValueError):
                pass
        return ("film", None, note) if note else ("none", None, "")
    text = str(feedback).strip()
    if not text:
        return "none", None, ""
    m = re.match(r"^\s*shots?\s*#?\s*(\d+)\s*[:\-\u2014]\s*(.+)$", text, re.IGNORECASE | re.DOTALL)
    if m:
        return "shot", int(m.group(1)), m.group(2).strip()
    return "film", None, text


def plan_film(
    concept: str,
    n_shots: int = 6,
    style: str = "",
    characters: Optional[Iterable[Any]] = None,
    must_include: Optional[Iterable[Any]] = None,
    feedback: Any = None,
    *,
    previous: Optional[Dict[str, Any]] = None,
    engine: str = "auto",
    tier: str = "draft",
    duration_s: float = 5.0,
    allow_hidden_faces: Optional[bool] = None,
    board_id: Optional[str] = None,
    locations: Optional[Iterable[Dict[str, Any]]] = None,
    screenplay: bool = True,
    geography: bool = True,
    known_character_ids: Optional[Iterable[str]] = None,
    ref_root: Optional[Any] = None,
    max_dim: Optional[int] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    seed: Optional[int] = None,
    max_tokens: Optional[int] = None,
    model_path: Optional[Any] = None,
    python_exe: Optional[Any] = None,
    session: Optional[PlannerSession] = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> Dict[str, Any]:
    """Concept -> film spec that `storyboard.validate_storyboard()` accepts.

    Returns EITHER a storyboard dict (schema/id/title/created_at/cast/policy/shots, plus a
    `_planner` block of metadata the validator ignores) OR a structured error dict — test
    with `is_plan_error(result)`. It never raises for model behaviour and never surfaces a
    traceback to a user.

    Args:
      concept        one or two sentences of intent. The only required argument.
      n_shots        how many shots to plan.
      style          master style, reused verbatim on every shot.
      characters     trained characters available for casting. Accepts `list_characters()`
                     records, bare id strings, or {'id','trigger','name','description'}.
                     A shot cast with one renders on LTX (that is where character LoRAs
                     live); everything else renders on H3.
      must_include   things that must appear somewhere in the film.
      feedback       None for a fresh plan. For a re-plan pass `previous=` plus either:
                       film-level : "make it colder, drop the voiceover"
                       per-shot   : "shot 4: he should not turn his head"
                                    or {"shot": 4, "note": "..."}
                     Per-shot re-rolls replace ONLY that shot; every other shot object is
                     carried across by reference, so the rest of the plan is byte-stable.
      locations      the places the film happens in, as board `locations` rows. They ride
                     into the returned spec so the shots that name them validate, and
                     they are what the geography pass hangs its views on.
      geography      write the FLOOR PLAN before the shots (default on): where everybody
                     stands, what is behind them, which side the light comes from — and
                     from that, 2-4 named VIEWS per location, so the reverse angle
                     excludes what is now behind the camera and the light flips sides.
                     Needs at least one location to be spatial about; with none it is
                     skipped and the plan behaves as it did before the pass existed.
      engine         "auto" (default) | "h3" | "ltx". Auto = H3 unless the shot is cast.
      tier           per-shot tier, default "draft" — plans are for reviewing, not shipping.
      allow_hidden_faces
                     None (default) auto-detects from the brief: a face may only be hidden
                     if the concept, style or must_include asked for it in so many words
                     ("silhouette", "from behind", "faceless", ...). Otherwise every shot
                     with a person carries the face law and face-hiding framing is scrubbed
                     out of the prose. Pass True/False to override the detection.
      max_dim        this machine's resolution cap; the policy is clamped to it so the
                     validator's tier check cannot fire.
      session        an already-open PlannerSession to borrow. If given it is NOT released
                     (the owner keeps it); otherwise a private one is opened and always
                     released before returning.

                     Pass one when the caller needs a handle on the running model — a
                     Cancel button is the case that matters: `plan_film()` blocks for
                     20-40 s, and the only way to stop it is `sess.release()` from another
                     thread, which kills the child and makes this call raise into its own
                     `finally`. A caller that supplies a session MUST release it.
    """
    t_start = time.time()
    if not str(concept or "").strip():
        raise PlannerError("concept is empty")
    n_shots = max(1, int(n_shots))
    cast = _normalise_cast(characters)
    must = [str(m).strip() for m in (must_include or ()) if str(m).strip()]
    locs = [l for l in (locations or ()) if isinstance(l, dict) and l.get("id")]
    fb_mode, fb_shot, fb_note = _parse_feedback(feedback)
    if fb_mode != "none" and previous is None:
        previous = feedback.get("previous") if isinstance(feedback, dict) else None
    if fb_mode != "none" and not isinstance(previous, dict):
        raise PlannerError("feedback needs the plan it refers to — pass previous=<spec>")

    # A hidden face is opt-in, and the brief is the only thing that may opt in. Detection
    # reads the concept, the style and the must-includes — not the model's output, which is
    # exactly the thing being guarded against.
    if allow_hidden_faces is None:
        brief_text = " ".join([concept or "", style or ""] + must)
        allow_hidden = bool(_WANTS_HIDDEN_RE.search(brief_text))
    else:
        allow_hidden = bool(allow_hidden_faces)

    validate, sb = _load_validator()
    seed_base = int(seed) if seed is not None else _stable_seed(concept)
    known_ids = list(known_character_ids) if known_character_ids is not None \
        else [c["id"] for c in cast]
    budget = int(max_tokens) if max_tokens else min(
        8192, max(1200, 380 * (1 if fb_mode == "shot" else n_shots) + 500))

    system = _build_system_prompt(engine, bool(cast), allow_hidden)
    if fb_mode == "shot":
        user = _build_shot_feedback_prompt(previous, fb_shot, fb_note)
    elif fb_mode == "film":
        user = _build_film_feedback_prompt(previous, fb_note, n_shots)
    else:
        user = _build_user_prompt(concept, n_shots, style, cast, must, locs)

    owned = session is None
    sess = session or PlannerSession(model_path=model_path, python_exe=python_exe,
                                     timeout_s=timeout_s)
    result: Dict[str, Any] = {}
    pass_warnings: List[str] = []
    scene_text = ""
    floor_plan = ""
    # Hoisted so the name exists on every path — the shot pass reads it below
    # to decide whether the prompt has to be rebuilt.
    _views: Dict[str, List[Dict[str, str]]] = {}
    if screenplay and fb_mode not in ("shot", "film"):
        # PASS ONE: write the scene. Only on a fresh plan — a re-roll already
        # has a scene and rewriting it would move the ground under the shot
        # being fixed.
        try:
            _sp = sess.generate(
                _SCREENPLAY_SYSTEM,
                _build_screenplay_prompt(concept, n_shots, style, cast, must, locs),
                max_tokens=1200, temperature=0.8, seed=seed_base % 100000)
            scene_text = _screenplay_text(_sp)
        except PlannerError:
            # A film with no scene is the behaviour this feature replaced, not a
            # failure worth aborting a plan over.
            scene_text = ""
    if geography and locs and fb_mode not in ("shot", "film"):
        # PASS TWO: block the space. Between the screenplay and the shots
        # because it reads the scene (who turns to whom decides which
        # directions have to exist) and the shots read it (a shot cannot name a
        # view that has not been derived yet). Skipped on a re-roll for the
        # same reason the screenplay is: the other shots are standing on this
        # geography and moving it under them is not a fix.
        #
        # It needs a location to be spatial ABOUT. With none, there is nowhere
        # to hang a view and no board field to carry it, so the pass is not run
        # and the plan behaves exactly as it did before it existed.
        try:
            _gp = sess.generate(
                _GEOGRAPHY_SYSTEM,
                _build_geography_prompt(concept, style, cast, locs, scene_text),
                max_tokens=1400, temperature=0.7, seed=(seed_base + 3) % 100000)
            floor_plan, _views = _geography_plan(_gp, locs)
        except PlannerError:
            floor_plan, _views = "", {}
        if _views:
            locs = _merge_views(locs, _views)
            pass_warnings.append(
                "geography: %d view(s) derived across %d location(s)"
                % (sum(len(v) for v in _views.values()), len(_views)))
        else:
            pass_warnings.append(
                "geography: the model returned no usable floor plan — shots will not "
                "declare views, exactly as before this pass existed")
    # VIEWS COUNT AS A REASON TO REBUILD THE PROMPT. `_geography_plan` derives
    # the floor plan from the model's own optional `floor_plan` key, so a reply
    # that carries views and no plan legitimately returns ("", {...views...}).
    # With the screenplay also empty — routine: `_screenplay_text` returns ""
    # whenever fewer than three lines match the form, and a PlannerError there
    # is deliberately tolerated — `locs` had been REPLACED by the merged list
    # while `user` was still the pre-geography prompt: no view ids in it, no
    # geography laws. The model then named no view, `_apply_geography` found
    # views on the location and no view on any shot, and stamped views[0] on
    # every shot with one warning each. That is the whole film pinned to the
    # establishing angle, which is the pre-views behaviour this pass exists to
    # end — arriving silently, through the pass itself.
    if scene_text or floor_plan or _views:
        user = _build_user_prompt(concept, n_shots, style, cast, must, locs,
                                  screenplay=scene_text, floor_plan=floor_plan)
    try:
        result = _plan_with_session(
            sess, system=system, user=user, fb_mode=fb_mode, fb_shot=fb_shot,
            previous=previous, validate=validate, sb=sb, concept=concept, n_shots=n_shots,
            style=style, cast=cast, board_id=board_id, engine=engine, tier=tier,
            duration_s=duration_s, seed_base=seed_base, max_dim=max_dim,
            known_ids=known_ids, ref_root=ref_root, temperature=temperature,
            budget=budget, model_path=model_path, t_start=t_start,
            allow_hidden_faces=allow_hidden, locs=locs,
            pass_warnings=pass_warnings)
    finally:
        # The model is gone before this function returns, on every path including an
        # exception. Peak RSS is only final once the child has been reaped, so the
        # measurement is patched in AFTER release.
        stats = sess.release() if owned else sess.stats
        blk = result.get("_planner") if isinstance(result, dict) else None
        if isinstance(blk, dict):
            blk.update(_session_meta_from(stats))
    return result


def _plan_with_session(sess, *, system, user, fb_mode, fb_shot, previous, validate, sb,
                       concept, n_shots, style, cast, board_id, engine, tier, duration_s,
                       seed_base, max_dim, known_ids, ref_root, temperature, budget,
                       model_path, t_start, allow_hidden_faces,
                       locs=(), pass_warnings=()) -> Dict[str, Any]:
    """The generate -> extract -> coerce -> validate -> ONE repair loop.

    Split out of plan_film() so the `finally:` that releases the model is three lines with
    nothing else in it — an unload that shares a code path with the happy path is an unload
    that eventually gets skipped.
    """
    meta = {"model": Path(model_path or sess.model_path).name, "attempts": 0}

    def coerce(obj):
        return _coerce_for_mode(
            obj, fb_mode, fb_shot, previous, concept=concept, n_shots=n_shots, style=style,
            cast=cast, board_id=board_id, engine=engine, tier=tier, duration_s=duration_s,
            seed_base=seed_base, max_dim=max_dim, sb=sb,
            allow_hidden_faces=allow_hidden_faces, locations=locs)

    def check(spec):
        errs = list(validate(spec, known_character_ids=known_ids,
                             ref_root=Path(ref_root) if ref_root else None, max_dim=max_dim))
        # `speech_without_words` is the board's version of L13 and it is a HARD
        # error there, correctly — a hand-authored board that would babble must
        # not reach a render. In here it is premature: this check runs before
        # `_enforce_laws`, which owns speech and can do better than reject —
        # it rewraps a prose-quoted line so the voice switches on, or stages the
        # shot honestly silent. Failing the plan here would spend the one repair
        # round-trip on a fault the next pass was about to fix for free, and
        # then return invalid_plan for it.
        # Dropping it is safe because it is not being waived, only deferred:
        # the panel validates the finished board again before it renders a
        # frame, and by then _enforce_laws has run.
        #
        # Filtered by CODE via the detail validator, not by matching English in
        # `errs` — `validate_storyboard()` returns formatted strings, so an
        # isinstance(dict) filter here silently matched nothing and every one of
        # these deferrals leaked through as an invalid_plan.
        detail = getattr(sb, "validate_storyboard_detail", None)
        if detail is None:
            return errs
        try:
            rows = detail(spec, known_character_ids=known_ids,
                          ref_root=Path(ref_root) if ref_root else None, max_dim=max_dim)
        except Exception:                                            # noqa: BLE001
            return errs
        # `dialogue_does_not_fit` is deferred for the same reason: the panel's
        # plan-adoption step FIXES it mechanically (it extends the shot to fit
        # the line via speech_fit_frames) — failing the plan here would spend
        # the repair round-trip on a fault that costs zero to fix.
        _DEFER = {"speech_without_words", "dialogue_does_not_fit"}
        return [r["message"] for r in rows if r.get("code") not in _DEFER]

    # ---- attempt 1 ------------------------------------------------------------------
    try:
        resp = sess.generate(system, user, max_tokens=budget,
                             temperature=temperature, seed=seed_base % 100000)
    except PlannerError as exc:
        return _error("model_unavailable", str(exc),
                      hint="Check that the planner model exists and that ltx-2-mlx/env "
                           "has mlx-lm installed.",
                      meta=dict(meta, elapsed_s=round(time.time() - t_start, 2)))
    meta["attempts"] = 1
    raw_text = resp.get("text") or ""
    obj = extract_json_object(raw_text)
    spec, warnings = coerce(obj)
    errs = check(spec)
    first_try = list(errs)
    count_off = (fb_mode != "shot") and (len(spec.get("shots") or []) != n_shots)
    first_try_clean = not first_try and not count_off and obj is not None

    # ---- ONE repair round-trip, carrying the REAL validator's words ------------------
    # Not a retry loop: a second failure means the concept is the problem, and burning
    # another 40 s of a user's evening to hear the same complaint helps nobody.
    if errs or count_off or obj is None:
        problems = list(errs)
        if count_off:
            problems.append("the plan has %d shots but exactly %d were requested"
                            % (len(spec.get("shots") or []), n_shots))
        if obj is None:
            problems.append("your reply did not contain a JSON object at all")
        try:
            resp2 = sess.generate(system, _build_repair_prompt(raw_text, problems, n_shots),
                                  max_tokens=budget,
                                  temperature=max(0.0, temperature * 0.5),
                                  seed=(seed_base + 1) % 100000)
            meta["attempts"] = 2
            raw2 = resp2.get("text") or ""
            obj2 = extract_json_object(raw2)
            if obj2 is not None:
                spec2, warn2 = coerce(obj2)
                errs2 = check(spec2)
                off2 = (fb_mode != "shot") and (len(spec2.get("shots") or []) != n_shots)
                # Keep the repair only if it is genuinely better, so a worse second draft
                # cannot destroy a first draft that merely had the wrong shot count.
                if (len(errs2), off2) < (len(errs), count_off):
                    spec, warnings, errs, count_off, raw_text = spec2, warn2, errs2, off2, raw2
                    warnings.append("repaired on the second pass")
        except PlannerError as exc:
            warnings.append("repair round failed: %s" % exc)

    # What the passes BEFORE this one had to say, in front of what the shot
    # pass had to say. Prepended here rather than earlier because the repair
    # round replaces `warnings` wholesale with the second draft's list.
    warnings[:0] = list(pass_warnings)

    if errs:
        return _error(
            "invalid_plan",
            "The planner could not turn this concept into a valid storyboard.",
            hint="Try a shorter, more concrete concept, or fewer shots.",
            problems=errs, raw=raw_text,
            meta=dict(meta, warnings=warnings, elapsed_s=round(time.time() - t_start, 2),
                      **_session_meta(sess)))

    # ---- L12 / L13: enforcement, not hope --------------------------------------------
    # After the plan is VALID, because both laws are about prose quality rather than schema
    # and a plan that fails the validator has a bigger problem than a species adjective.
    # Skipped on a per-shot re-roll: the caller is already editing one shot by hand, and
    # re-planning a re-plan from inside itself is how a 25 s call becomes a 4-minute one.
    _degraded: List[str] = []
    if fb_mode != "shot":
        def _replan_one(current, n, note):
            resp3 = sess.generate(
                system, _build_shot_feedback_prompt(current, n, note),
                max_tokens=min(budget, 1400),
                temperature=max(0.0, temperature * 0.6),
                seed=(seed_base + 7 + n) % 100000)
            meta["attempts"] = int(meta.get("attempts") or 0) + 1
            obj3 = extract_json_object(resp3.get("text") or "")
            if obj3 is None:
                return None
            fixed, warn3 = _coerce_for_mode(
                obj3, "shot", n, current, concept=concept, n_shots=n_shots, style=style,
                cast=cast, board_id=board_id, engine=engine, tier=tier,
                duration_s=duration_s, seed_base=seed_base, max_dim=max_dim, sb=sb,
                allow_hidden_faces=allow_hidden_faces, locations=locs)
            # A re-roll that breaks the SCHEMA is discarded outright — the law fix is not
            # worth an invalid plan, and the mechanical fallback can still clean the prose.
            if check(fixed):
                return None
            warnings.extend(w for w in warn3 if "law" not in w.lower())
            return fixed
        spec = _enforce_laws(spec, cast, warnings, replan=_replan_one, style=style, sb=sb)
        # The premise check runs LAST and only when the brief named creatures: the
        # appearance law is what puts it at risk, so it is checked after that law has
        # finished rewriting shots.
        _premise = _premise_species(" ".join([concept or "", style or ""]))
        spec = _enforce_premise(
            spec, cast, warnings, replan=_replan_one, style=style, sb=sb,
            premise=_premise)
        # SPACE, after the prose laws have finished rewriting shots. Safe in
        # any order in fact — it reads `eyeline`/`view`/`description` and
        # writes only `eyeline`, which no other law reads and which is composed
        # into the prompt at render time — but it is a scan, not a re-plan, and
        # scanning what the last mutation produced is the only honest place.
        spec = _enforce_geography(spec, locs, warnings)
        # ---- THE FINAL INVARIANT SCAN ------------------------------------
        # After the LAST mutation, whichever pass made it. Every pass before
        # this validated only the condition it owned, so the last repair could
        # silently undo an earlier guarantee and nothing looked.
        spec, _degraded = _assert_final_invariants(
            spec, cast, warnings, style=style, sb=sb, premise=_premise)
    else:
        # A RE-ROLL IS A PLAN TOO. The whole enforcement block above is skipped
        # for fb_mode == "shot" — correctly, because re-planning a re-plan from
        # inside itself turns a 25 s call into a four-minute one. But skipping
        # the RE-PLAN also skipped the final SCAN, so a re-rolled shot carrying
        # an L13 violation came back with ok: true, degraded: false and empty
        # reasons. The expensive part is the model round trip; scanning what it
        # returned costs nothing and is exactly the thing that must not be
        # optional. Mechanical repair still applies, and what cannot be repaired
        # is reported the same way a full plan reports it.
        _premise = _premise_species(" ".join([concept or "", style or ""]))
        # A RE-ROLLED SHOT CAN BREAK THE LINE with the neighbours it was cut
        # between — and unlike the prose laws this one costs no model call, so
        # there is no reason for the cheap path to skip it.
        spec = _enforce_geography(spec, locs or (spec.get("locations") or ()), warnings)
        spec, _degraded = _assert_final_invariants(
            spec, cast, warnings, style=style, sb=sb, premise=_premise)

    # A DEGRADED PLAN DOES NOT GET TO SAY ok=True. The final pass reports what it
    # could not repair — an unlawful shot, or a premise the film never showed —
    # and stamping success over that is how a known defect reaches a user with a
    # green tick on it. `ok` here is the PLANNER's metadata, not the top-level
    # error flag `is_plan_error()` reads, so a degraded plan is still a usable
    # storyboard the caller can render; it is simply no longer described as
    # clean, and `degraded_reasons` says why in the words the UI already shows.
    spec["_planner"] = dict(
        meta,
        ok=not _degraded,
        degraded=bool(_degraded),
        degraded_reasons=list(_degraded),
        warnings=warnings,
        first_try_errors=first_try,
        first_try_clean=first_try_clean,
        shot_count_ok=not count_off,
        engine_mix=_engine_mix(spec),
        concept=concept.strip(),
        feedback_mode=fb_mode,
        elapsed_s=round(time.time() - t_start, 2),
        **_session_meta(sess)
    )
    return spec


# ---- THE GEOGRAPHY LAWS ---------------------------------------------------------------
# Both are WARNING level and neither can fail a plan. A film whose eyelines are
# a little loose is still a film; a film that will not render is not. What they
# do is name the defect in the words the UI already shows, and — where the fix
# is a single discrete field with exactly one other legal value — apply it.

def _shots_in_cut_order(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The shots as the audience meets them. "Adjacent" means adjacent ON SCREEN."""
    rows = [s for s in (spec.get("shots") or ()) if isinstance(s, dict)]
    return sorted(rows, key=lambda s: s.get("n") if isinstance(s.get("n"), int) else 0)


def _enforce_eyelines(spec: Dict[str, Any], warnings: List[str]) -> Dict[str, Any]:
    """THE 180-DEGREE LINE: a cut between two people reverses the eyeline.

    He looks frame-right at her; she looks frame-LEFT back at him. Two adjacent
    shots of two DIFFERENT characters in the SAME place that both claim the
    same side are the cut where the audience watches both of them turn and
    stare at the same wall — the single most recognisable continuity error in
    the format, and the owner spent a day fixing it by hand.

    Repaired mechanically, which is only defensible because `eyeline` is a
    discrete field with exactly one complement: flipping it cannot damage
    anybody's prose, and the clause it produces is composed at render time
    (`compose_shot_prompt`), so nothing has to be re-assembled. Prose laws get
    a model round trip precisely because they do not have this property.
    """
    shots = _shots_in_cut_order(spec)
    for prev, cur in zip(shots, shots[1:]):
        a, b = prev.get("eyeline"), cur.get("eyeline")
        if a not in ("left", "right") or a != b:
            continue
        ca, cb = prev.get("character_id"), cur.get("character_id")
        if not ca or not cb or ca == cb:
            continue
        if (prev.get("location_id") or "") != (cur.get("location_id") or ""):
            continue
        flip = "right" if b == "left" else "left"
        cur["eyeline"] = flip
        warnings.append(
            "shots %s and %s cut between two people and both looked %s — the 180-degree "
            "line. Shot %s now looks %s, back at them."
            % (prev.get("n"), cur.get("n"), b, cur.get("n"), flip))
    return spec


# What a view says is NOT in it. "no car in frame", "without the sign".
# The floor plan is asked to write these sentences precisely so this check has
# something exact to stand on.
_ABSENCE_RE = re.compile(r"\b(?:no|without|not)\s+((?:[a-z][a-z'-]{2,}\s+){0,2}[a-z][a-z'-]{2,})",
                         re.IGNORECASE)
# Words that carry no object of their own — a view saying "no longer" or "not
# visible" is not naming a thing that must stay off screen.
_ABSENCE_STOP = {
    "longer", "visible", "seen", "shown", "there", "here", "part", "more", "less",
    "camera", "frame", "shot", "view", "side", "screen", "left", "right", "front",
    "behind", "back", "this", "that", "these", "those", "them", "their", "they",
    "anything", "anyone", "nothing", "else", "other", "another", "same", "and",
    "the", "one", "two", "any", "all", "with", "from", "into", "onto", "over",
    "under", "near", "past", "still", "yet", "just", "only", "even", "very",
}


def _absent_terms(view_desc: str) -> List[str]:
    """The things a view SAYS are not in it: "no car in frame" -> ["car"].

    Only the view's own negations. An earlier draft also inferred absence by
    diffing the content words of the other views, and that is wrong in the case
    the feature exists for: a reverse angle legitimately contains the OTHER
    person and everything they are holding, so every mention of her, her
    sponge, her wheel came back as a violation. The floor plan states absence
    out loud; nothing else is trustworthy enough to warn on.
    """
    out: List[str] = []
    for m in _ABSENCE_RE.finditer(view_desc or ""):
        for word in m.group(1).split():
            w = word.strip(".,;:'\"").lower()
            if len(w) >= 3 and w not in _ABSENCE_STOP and w not in out:
                out.append(w)
    return out


def _scan_reverse_objects(spec: Dict[str, Any],
                          locations: Sequence[Dict[str, Any]]) -> List[Tuple[int, str, str]]:
    """[(shot n, view id, the thing that is behind the camera in it)].

    The car-wash law: the reverse angle exists so the car is not in it, and a
    shot on that view that writes the car back in has undone the only reason
    it was derived.
    """
    by_id = {str(l.get("id")): l for l in (locations or ()) if isinstance(l, dict)}
    out: List[Tuple[int, str, str]] = []
    for s in (spec.get("shots") or ()):
        if not isinstance(s, dict) or not s.get("view"):
            continue
        loc = by_id.get(str(s.get("location_id") or ""))
        view = next((v for v in ((loc or {}).get("views") or ())
                     if isinstance(v, dict)
                     and str(v.get("id")).strip().lower() == str(s["view"]).strip().lower()),
                    None)
        if view is None:
            continue
        text = " ".join([str(s.get("description") or ""), str(s.get("settle") or "")])
        for term in _absent_terms(str(view.get("description") or "")):
            if re.search(r"\b%s(?:e?s)?\b" % re.escape(term), text, re.IGNORECASE):
                out.append((int(s.get("n") or 0), str(s["view"]), term))
                break
    return out


def _enforce_geography(spec: Dict[str, Any], locations: Sequence[Dict[str, Any]],
                       warnings: List[str]) -> Dict[str, Any]:
    """Both laws, in the order that makes the second one's warning readable."""
    spec = _enforce_eyelines(spec, warnings)
    for n, view, term in _scan_reverse_objects(spec, locations):
        # NOT repaired. Cutting a noun out of a sentence leaves prose nobody
        # wrote, and a targeted re-plan is a 20-second round trip spent on a
        # shot the user can re-roll deliberately with this sentence in hand.
        warnings.append(
            "shot %d is on view %r, which says the %s is not in it — and the shot puts "
            "the %s back in frame. Re-roll the shot, or move it to the view that holds "
            "the %s." % (n, view, term, term, term))
    return spec


def _scan_laws(spec: Dict[str, Any],
               cast: Sequence[Dict[str, str]]) -> List[Tuple[int, List[str]]]:
    """Every shot that breaks L12 or L13, as [(shot_n, [reasons])].

    One entry per SHOT, not per law, because a shot that trips both gets ONE re-plan with
    both restated — sending the model back twice for the same paragraph wastes 20 s and
    invites it to fix the second complaint by reintroducing the first.
    """
    by_id = {c["id"]: c for c in cast}
    out: List[Tuple[int, List[str]]] = []
    for s in spec.get("shots") or ():
        if not isinstance(s, dict):
            continue
        desc = s.get("description") or ""
        sound = s.get("soundscape") or ""
        reasons: List[str] = []
        char = by_id.get(s.get("character_id"))
        if char:
            for clause in _appearance_violations(desc, char["trigger"], char.get("name", "")):
                reasons.append(
                    "APPEARANCE (L12): you wrote %r. %s is a TRAINED face — you cannot see "
                    "it and must not describe it. Keep the role, the wardrobe, the action "
                    "and the emotion; delete the species, the build, the age and the "
                    "features." % (clause[:70], char["trigger"]))
        for problem in _speech_violations(desc, sound):
            reasons.append(
                "SPEECH (L13): %s. Either write the actual line as <d>[English] ...</d>, or "
                "stage the shot silent — no speech verbs, no described voice, and no "
                "murmur/chatter/voices in the soundscape." % problem)
        if reasons:
            out.append((int(s.get("n") or 0), reasons))
    return out


def _premise_species(brief: str) -> List[str]:
    """Species/creature words the BRIEF itself asked for. The film's premise, in the
    user's own words — never inferred, never the model's."""
    return sorted({m.group(0).lower() for m in _SPECIES_RE.finditer(brief or "")})


# Irregular plurals the species table lists in both forms. A brief that says
# "wolves" is satisfied by a shot that says "wolf", and vice versa.
_SPECIES_FORMS = {
    "wolf": "wolves", "wolves": "wolf", "mouse": "mice", "mice": "mouse",
    "goose": "geese", "geese": "goose", "ox": "oxen", "oxen": "ox",
    "sheep": "sheep", "deer": "deer", "fish": "fish",
}


def _premise_term_re(term: str) -> Any:
    """`fox` -> matches fox/foxes; `wolf` -> matches wolf/wolves."""
    forms = {term, _SPECIES_FORMS.get(term, "")} - {""}
    alts = sorted({re.escape(f) for f in forms})
    return re.compile(r"\b(?:%s)(?:e?s)?\b" % "|".join(alts), re.IGNORECASE)


def _premise_terms_present(text: str, premise: Sequence[str]) -> bool:
    """Does `text` show the premise's OWN creatures?

    It used to be enough for ANY species word to appear anywhere, so a brief
    asking for a fox was considered preserved by a shot containing a robot —
    the check passed while the film had quietly become a different film. The
    premise is the user's words; only the user's words can satisfy it.
    """
    return any(_premise_term_re(t).search(text or "") for t in premise)


def _premise_missing_terms(spec: Dict[str, Any], cast: Sequence[Dict[str, str]],
                           premise: Sequence[str]) -> List[str]:
    """Which of the brief's OWN creatures never made it onto the screen.

    ALL THE TERMS, NOT ANY OF THEM. "the crew are a fox and a badger" was
    counted as preserved by a film containing only the fox — half the premise
    silently dropped and the check said fine. Every term the brief names has to
    appear somewhere the law does not govern.

    THE PRESENCE BUDGET. Requiring all of them unconditionally would be its own
    lie: a brief naming five species cannot show five in a two-shot film, and
    the format's own COMPOSITION LIMITS forbid crowding a shot with subjects to
    make a checker happy. So the requirement is capped by the number of uncast
    shots — the places a creature can legitimately appear. With room for two,
    two distinct terms are required; with room for one, one is enough.
    """
    if not premise:
        return []
    cast_ids = {c["id"] for c in cast}
    uncast = [s for s in (spec.get("shots") or ())
              if isinstance(s, dict) and s.get("character_id") not in cast_ids]
    if not uncast:
        return []
    seen, unseen = [], []
    for term in premise:
        rx = _premise_term_re(term)
        (seen if any(rx.search(s.get("description") or "") for s in uncast)
         else unseen).append(term)
    budget = min(len(premise), len(uncast))
    if len(seen) >= budget:
        return []
    return unseen


def _premise_lost(spec: Dict[str, Any], cast: Sequence[Dict[str, str]],
                  premise: Sequence[str]) -> bool:
    """Did the appearance law eat the film's own premise?

    THE OVER-CORRECTION, measured. Told never to give the cast character a species, the
    planner generalised it to never mentioning species AT ALL: the owner's "main
    characters are humanoid animals" film came back with zero animal words across twelve
    shots — every soldier on both sides quietly turned into an ordinary human, which is a
    different film. Each shot was individually lawful, and the plan was wrong.

    So the check is on the shots the law does NOT govern. If the brief named creatures and
    not one uncast shot shows them, the premise is gone.
    """
    return bool(_premise_missing_terms(spec, cast, premise))


def _premise_note(premise: Sequence[str]) -> str:
    return ("This shot lost the film's premise. The brief asked for %s, and not one shot "
            "that is free to show them does. Re-write this shot so the characters in it "
            "are visibly what the concept says they are — the rule about not describing a "
            "trained cast character applies ONLY to that one character, never to anyone "
            "else, and it is not a reason to make the rest of the film plain."
            % ", ".join(premise[:4]))


def _law_note(reasons: Sequence[str]) -> str:
    return ("This shot breaks a hard law of the format. Fix ONLY this, keep everything else "
            "word for word:\n" + "\n".join("  - %s" % r for r in reasons))


def _reassemble_prompt(shot: Dict[str, Any], style: str,
                       cast: Sequence[Dict[str, str]], sb: Any) -> str:
    """Rebuild the assembled prompt after a mechanical edit to description/soundscape."""
    desc = shot.get("description") or ""
    sound = shot.get("soundscape") or ""
    music = shot.get("music") or "N/A"
    camera = shot.get("camera") or "static"
    settle = shot.get("settle") or ""
    face = shot.get("face") or "medium"
    char = None
    for c in cast:
        if c["id"] == shot.get("character_id"):
            char = c
            break
    if shot.get("engine") == "h3":
        if char:
            return _assemble_h3_prompt(
                "%s The on-screen subject is %s." % (desc, char["trigger"]),
                sound, music, camera, settle, face)
        return _assemble_h3_prompt(desc, sound, music, camera, settle, face)
    prompt = _assemble_ltx_prompt(desc, sound, style, camera, settle, face)
    if char:
        if sb is not None and hasattr(sb, "ensure_trigger"):
            prompt = sb.ensure_trigger(prompt, char["trigger"])
        elif not re.search(r"\b%s\b" % re.escape(char["trigger"]), prompt):
            prompt = "%s %s" % (char["trigger"], prompt)
    return prompt


def _assert_final_invariants(spec, cast, warnings, *, style, sb, premise=()):
    """The last word: every law, over the whole plan, after the last mutation.

    THE PASSES WERE NOT COMPOSABLE. Each one validated only the condition it
    owned — the law pass checked laws, the premise pass checked the premise —
    so the final repair could undo an earlier guarantee and nothing ever looked
    again. That is a structural hole, not a bug in any one pass: it reopens
    every time a new pass is added at the end.

    This runs after all of them, mechanically neutralises anything still
    standing, and — the part that matters — reports honestly when it cannot. A
    plan that reaches a user with a known violation must say so; the previous
    behaviour was to ship it under a warning claiming the repair had worked.
    """
    remaining = _scan_laws(spec, cast)
    if not remaining:
        missing = _premise_missing_terms(spec, cast, premise)
        if missing:
            msg = ("PLAN LOST PART OF ITS PREMISE: no shot shows %s. The concept "
                   "asked for it and the film does not have it."
                   % ", ".join(missing[:4]))
            warnings.append(msg)
            return spec, [msg]
        return spec, []
    by_id = {c["id"]: c for c in cast}
    for n, _reasons in remaining:
        for s in spec.get("shots") or ():
            if not isinstance(s, dict) or int(s.get("n") or 0) != n:
                continue
            touched = False
            char = by_id.get(s.get("character_id"))
            if char:
                fixed, cuts = _neutralise_appearance(
                    s.get("description") or "", char["trigger"], char.get("name", ""),
                    char.get("subject_noun", ""))
                if cuts:
                    s["description"] = fixed
                    touched = True
            d0, s0 = s.get("description") or "", s.get("soundscape") or ""
            d2, s2, _ = _neutralise_speech(d0, s0)
            if (d2, s2) != (d0, s0):
                s["description"], s["soundscape"] = d2, s2
                touched = True
            if touched:
                s["prompt"] = _reassemble_prompt(s, style, cast, sb)
                warnings.append("shot %d: final invariant pass had to repair it" % n)
    still = _scan_laws(spec, cast)
    degraded: List[str] = []
    if still:
        msg = ("PLAN SHIPPED WITH %d UNREPAIRED SHOT(S): %s. This is a defect, not a "
               "style note - re-roll those shots or re-word the concept."
               % (len(still), ", ".join(str(n) for n, _ in still)))
        warnings.append(msg)
        degraded.append(msg)
    # THE PREMISE IS AN INVARIANT TOO. This pass checked L12/L13 and stopped,
    # so a premise repair that failed was still the last word on the plan and
    # nothing downstream ever re-asked. Now it is asked here, after everything.
    missing = _premise_missing_terms(spec, cast, premise)
    if missing:
        msg = ("PLAN LOST PART OF ITS PREMISE: no shot shows %s. The concept asked "
               "for it and the film does not have it."
               % ", ".join(missing[:4]))
        warnings.append(msg)
        degraded.append(msg)
    return spec, degraded


def _enforce_premise(spec, cast, warnings, *, replan, premise, style, sb) -> Dict[str, Any]:
    """One re-plan to put the film's own premise back, if the appearance law ate it.

    Bounded to a SINGLE uncast shot: one example is enough to re-anchor the world, and
    this runs after every other pass, so it must not turn a 40 s plan into a 4 minute one.
    """
    if not _premise_lost(spec, cast, premise):
        return spec
    warnings.append("the plan lost the brief's premise (%s) — no uncast shot shows them"
                    % ", ".join(premise[:4]))
    cast_ids = {c["id"] for c in cast}
    target = next((s for s in (spec.get("shots") or ())
                   if isinstance(s, dict) and s.get("character_id") not in cast_ids), None)
    if target is None:
        warnings.append("every shot is cast, so there is nowhere to restate the premise")
        return spec
    n = int(target.get("n") or 0)
    try:
        candidate = replan(spec, n, _premise_note(premise))
    except PlannerError as exc:
        warnings.append("premise re-plan failed (%s)" % exc)
        return spec
    if candidate is None:
        return spec
    # A REPAIR THAT BREAKS A LAW IS NOT A REPAIR. The replan helper validates
    # schema only, so a shot that came back with the premise restored could also
    # come back describing speech it never writes — and it was accepted, under a
    # warning that said "the premise is back". Measured by the external review:
    # a returned shot containing "explains the mission" with no dialogue tags.
    #
    # The premise is the LESSER law: a film that keeps its animals but babbles is
    # worse than one that lost its animals, and the animals can be asked for
    # again in the concept. Neutralise first; if it still violates, REJECT.
    broke = dict(_scan_laws(candidate, cast))
    if n in broke:
        for s_ in candidate.get("shots") or ():
            if isinstance(s_, dict) and int(s_.get("n") or 0) == n:
                d0, s0 = s_.get("description") or "", s_.get("soundscape") or ""
                d2, s2, _ = _neutralise_speech(d0, s0)
                if (d2, s2) != (d0, s0):
                    s_["description"], s_["soundscape"] = d2, s2
                    s_["prompt"] = _reassemble_prompt(s_, style, cast, sb)
        if n in dict(_scan_laws(candidate, cast)):
            warnings.append("shot %d: the premise re-plan broke a hard law and was "
                            "REJECTED - the film keeps its previous shot %d" % (n, n))
            return spec
        warnings.append("shot %d: the premise re-plan needed silencing to stay lawful" % n)
    if _premise_lost(candidate, cast, premise):
        # Not neutralised mechanically: writing a badger into someone's film is authorship,
        # not repair, and a wrong guess is worse than the omission. The warning stands and
        # the user can re-roll the shot themselves.
        warnings.append("shot %d was re-planned and the premise is still missing — say it "
                        "again in the concept, or re-roll a shot" % n)
    else:
        warnings.append("shot %d: re-planned and the premise is back" % n)
    return candidate


def _enforce_laws(spec, cast, warnings, *, replan, style, sb) -> Dict[str, Any]:
    """L12/L13: one targeted re-plan per offending shot, then a mechanical fallback.

    `replan(current, n, note) -> spec | None` re-rolls ONE shot through the existing
    per-shot machinery, which carries every other shot across by reference — so a film
    whose shot 4 was fixed is byte-identical everywhere else.

    THE CURRENT SPEC IS AN ARGUMENT, and that is not decoration. It was a closure over the
    caller's `spec` for one draft, which meant every re-plan spliced its fix into the
    ORIGINAL plan: fix shot 1, then fix shot 2 against the unfixed plan, and shot 1's fix
    is gone. A live 4-shot film reported "shot 1: re-planned and now obeys", the same for
    2 and 3, and then failed the final scan on all three — three model calls spent to
    change nothing. Threading it through is what makes the fixes accumulate.

    The fallback is not optional. A law that is only enforced when the model cooperates is
    a suggestion, and L12 exists because the model demonstrably does not cooperate.
    """
    offenders = _scan_laws(spec, cast)
    if not offenders:
        return spec
    warnings.append("law check: %d shot(s) broke the appearance or speech law on the first "
                    "pass" % len(offenders))
    for n, reasons in offenders:
        try:
            candidate = replan(spec, n, _law_note(reasons))
        except PlannerError as exc:
            candidate = None
            warnings.append("shot %d: law re-plan failed (%s)" % (n, exc))
        if candidate is not None:
            still = dict(_scan_laws(candidate, cast))
            if n not in still:
                spec = candidate
                warnings.append("shot %d: re-planned and now obeys the law" % n)
                continue
            # The re-plan came back and broke it again — keep it (it may be better prose)
            # and let the mechanical pass finish the job.
            spec = candidate
        # --- mechanical fallback, second failure only --------------------------------
        by_id = {c["id"]: c for c in cast}
        for s in spec.get("shots") or ():
            if not isinstance(s, dict) or int(s.get("n") or 0) != n:
                continue
            char = by_id.get(s.get("character_id"))
            touched = False
            if char:
                fixed, cuts = _neutralise_appearance(
                    s.get("description") or "", char["trigger"], char.get("name", ""),
                    char.get("subject_noun", ""))
                if cuts:
                    s["description"] = fixed
                    touched = True
                    for c in cuts:
                        warnings.append("shot %d: appearance %r was written for a trained "
                                        "face and has been neutralised" % (n, c[:60]))
            d0, s0 = s.get("description") or "", s.get("soundscape") or ""
            d2, s2, notes = _neutralise_speech(d0, s0)
            # Keyed on the DIFF, not on `notes`: appending the silence sentence is a real
            # change that produces no note of its own, and a shot that reached here has
            # already been flagged — leaving it half-fixed is the worst outcome.
            if d2 != d0 or s2 != s0:
                s["description"], s["soundscape"] = d2, s2
                touched = True
                for note in (notes or ["staged the shot silent"]):
                    warnings.append("shot %d: %s (the shot is now honestly silent)"
                                    % (n, note))
            if touched:
                s["prompt"] = _reassemble_prompt(s, style, cast, sb)
    return spec


def _session_meta(sess: PlannerSession) -> Dict[str, Any]:
    return _session_meta_from(sess.stats)


def _session_meta_from(stats: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "model_path": stats.get("model_path"),
        "load_s": stats.get("load_s"),
        "gen_s": stats.get("gen_s_total"),
        "prompt_tokens": stats.get("prompt_tokens"),
        "output_tokens": stats.get("output_tokens"),
        "peak_rss_bytes": stats.get("peak_rss_bytes"),
        "peak_rss_gb": round((stats.get("peak_rss_bytes") or 0) / float(2 ** 30), 2),
        "mx_peak_gb": round((stats.get("mx_peak_bytes") or 0) / float(2 ** 30), 2),
        "model_released": bool(stats.get("released")),
    }


def _engine_mix(spec: Dict[str, Any]) -> Dict[str, int]:
    mix: Dict[str, int] = {}
    for s in spec.get("shots") or ():
        k = s.get("engine") or "h3"
        mix[k] = mix.get(k, 0) + 1
    return mix


def _coerce_for_mode(obj, fb_mode, fb_shot, previous, *, concept, n_shots, style, cast,
                     board_id, engine, tier, duration_s, seed_base, max_dim, sb,
                     allow_hidden_faces=False, locations=None):
    """Coerce a fresh plan, or splice one re-rolled shot into an existing plan."""
    if fb_mode != "shot":
        return coerce_spec(obj, concept=concept, n_shots=n_shots, style=style, cast=cast,
                           board_id=board_id, engine=engine, tier=tier,
                           duration_s=duration_s, seed_base=seed_base, max_dim=max_dim,
                           allow_hidden_faces=allow_hidden_faces, storyboard_mod=sb,
                           locations=locations)

    # Per-shot re-roll: coerce the single returned shot, then splice. Every other shot
    # object is carried across by reference, so the untouched part of the plan is
    # byte-identical when re-serialised.
    one, warnings = coerce_spec(obj, concept=concept, n_shots=1, style=style, cast=cast,
                                board_id=previous.get("id"), engine=engine, tier=tier,
                                duration_s=duration_s, seed_base=seed_base, max_dim=max_dim,
                                allow_hidden_faces=allow_hidden_faces, storyboard_mod=sb,
                                locations=locations)
    new_shots = one.get("shots") or []
    spec = copy.copy(previous)
    spec["shots"] = list(previous.get("shots") or [])
    if not new_shots:
        warnings.append("re-roll returned no usable shot; the original was kept")
        return spec, warnings
    replacement = new_shots[0]
    for i, s in enumerate(spec["shots"]):
        if s.get("n") == fb_shot:
            replacement["n"] = fb_shot
            # A re-roll should look different: nudge the seed rather than re-render the
            # same latent with new words.
            replacement["seed"] = _seed_for(seed_base + int(time.time()) % 9973, fb_shot)
            for carry in ("status",):
                if carry in s:
                    replacement[carry] = s[carry]
            spec["shots"][i] = replacement
            break
    else:
        warnings.append("shot %r was not in the plan; nothing replaced" % fb_shot)
    return spec, warnings


# --------------------------------------------------------------------------------------
# The worker process (runs under ltx-2-mlx/env python3.11 — the only place mlx is imported)
# --------------------------------------------------------------------------------------

def _worker_serve() -> int:
    import resource

    def emit(obj):
        sys.stdout.write(_SENTINEL + json.dumps(obj) + "\n")
        sys.stdout.flush()

    def peak_rss_bytes():
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(r if sys.platform == "darwin" else r * 1024)

    model = tokenizer = None
    loaded_path = None
    load_s = None

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError as exc:
            emit({"error": "bad request: %s" % exc})
            continue
        action = req.get("action")
        if action == "exit":
            break
        if action == "ping":
            emit({"pong": True, "peak_rss_bytes": peak_rss_bytes()})
            continue
        if action != "generate":
            emit({"error": "unknown action %r" % action})
            continue
        try:
            import mlx.core as mx
            from mlx_lm import load as mlx_lm_load, generate as mlx_generate
            from mlx_lm.sample_utils import make_sampler

            path = req["model_path"]
            if model is None or loaded_path != path:
                t0 = time.time()
                model, tokenizer = mlx_lm_load(path)
                loaded_path = path
                load_s = round(time.time() - t0, 2)

            messages = [
                {"role": "system", "content": req.get("system") or ""},
                {"role": "user", "content": req.get("user") or ""},
            ]
            try:
                chat = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                # Some chat templates refuse a system role — fold it into the user turn.
                chat = tokenizer.apply_chat_template(
                    [{"role": "user", "content": (req.get("system") or "") + "\n\n"
                      + (req.get("user") or "")}],
                    tokenize=False, add_generation_prompt=True)

            mx.random.seed(int(req.get("seed") or 10))
            t0 = time.time()
            text = mlx_generate(
                model=model, tokenizer=tokenizer, prompt=chat,
                max_tokens=int(req.get("max_tokens") or DEFAULT_MAX_TOKENS),
                sampler=make_sampler(temp=float(req.get("temperature") or 0.0)),
                verbose=False,
            )
            gen_s = round(time.time() - t0, 2)
            try:
                mx_peak = int(mx.get_peak_memory())
            except Exception:
                mx_peak = 0
            emit({
                "text": text,
                "load_s": load_s,
                "gen_s": gen_s,
                "prompt_tokens": len(tokenizer.encode(chat)),
                "output_tokens": len(tokenizer.encode(text)),
                "peak_rss_bytes": peak_rss_bytes(),
                "mx_peak_bytes": mx_peak,
            })
        except Exception as exc:  # never let a traceback reach the parent as protocol
            import traceback
            emit({"error": "%s: %s" % (type(exc).__name__, exc),
                  "trace": traceback.format_exc()[-1500:]})
    return 0


# --------------------------------------------------------------------------------------
# CLI — `--serve` is the worker; no args plans a film and prints it
# --------------------------------------------------------------------------------------

def _main(argv: Sequence[str]) -> int:
    if "--serve" in argv:
        return _worker_serve()
    import argparse
    ap = argparse.ArgumentParser(description="Plan a Phosphene storyboard from a concept.")
    ap.add_argument("concept", nargs="?", help="one or two sentences of intent")
    ap.add_argument("-n", "--shots", type=int, default=6)
    ap.add_argument("--style", default="")
    ap.add_argument("--character", action="append", default=[],
                    help="trigger of a trained character available for casting")
    ap.add_argument("--must", action="append", default=[])
    ap.add_argument("--engine", default="auto", choices=("auto", "h3", "ltx"))
    ap.add_argument("--tier", default="draft")
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--model", default=None)
    args = ap.parse_args([a for a in argv if a != "--serve"])
    if not args.concept:
        ap.error("a concept is required")
    out = plan_film(args.concept, n_shots=args.shots, style=args.style,
                    characters=args.character, must_include=args.must,
                    engine=args.engine, tier=args.tier, temperature=args.temperature,
                    model_path=args.model)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 1 if is_plan_error(out) else 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
