#!/usr/bin/env python3
"""The timeline editor's SERVER SIDE — proxies, peaks, and `edit.json`.

WHY THIS EXISTS
---------------
`storyboard_edit.py` answers "which seconds are good" and "where are the
beats". It produces a PLAN — a one-shot artefact, computed and consumed in the
same breath by the exporter. That is enough for an auto-edit and nowhere near
enough for a timeline a human can drag.

Three things have to exist before a browser can show a timeline at all, and
none of them belong in the panel's HTTP handler:

1. **PROXIES, and they are not an optimisation.** Measured on this corpus:
   our rendered clips contain exactly ONE keyframe — the whole clip is a
   single GOP — so a browser asked to seek to t=3.2 s must decode from frame
   zero. Chrome takes a **235 ms median, 1266 ms p90** to do it. Against an
   all-intra proxy the same seek is **3.5 ms**. A timeline is a machine for
   seeking; at a quarter of a second per scrub there is no timeline, there is
   a slideshow that apologises. Measured on ten AURELIUS clips (56.25 s of
   1024x576): 0.14 s of ffmpeg and 1.45 MB of proxy per 10 s of video, and
   the result really is all-intra — 123 keyframes in a 123-frame proxy,
   against 1 in its source. That trade is not close.

2. **WAVEFORM PEAKS, computed here.** A five-minute track decoded in the
   browser is ~85 MB of Float32Array before a single pixel is drawn. The same
   track reduced to min/max pairs at 100 buckets a second is 697 bytes per
   second of audio — 326 KB for the 479 s AMOR_FATI master, produced in
   0.42 s of numpy. The client draws; it does not decode.

3. **`edit.json`, which is NOT the storyboard.** The board is INTENT — the
   shots the film wants to contain, one entry per thing to render. The edit is
   ARRANGEMENT — what plays, from which second of which file, at which second
   of the film. They are the same list exactly until the first time somebody
   uses one shot twice or splits one in half, and then they are different
   shapes forever. Merging them would mean the render queue and the timeline
   fighting over one array. So: two files, side by side, `storyboard.json` and
   `edit.json`, and one function (`edit_from_plan`) that derives the second
   from the first when there isn't one yet.

WHAT THIS MODULE WILL NOT DO
----------------------------
It opens no sockets, imports nothing from `mlx_ltx_panel`, and runs no
subprocess of its own except ffprobe-free peak decoding through
`storyboard_edit`. Proxy building is expressed as an ARGV LIST that the panel
runs through its own `run_ffmpeg_tracked`, because the panel owns process
groups, cancellation and the log pane, and a module that shells out behind the
panel's back is a module that cannot be stopped.

DEPENDENCIES: numpy and the standard library. Same constraint, same reason, as
`storyboard_edit.py`.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

__all__ = [
    "EDIT_VERSION", "PEAKS_VERSION", "PROXY_RECIPE_VERSION",
    "CLIP_KINDS", "BRIGHTNESS_LIMIT",
    "proxy_dir", "proxy_fingerprint", "proxy_name", "proxy_cmd",
    "plan_proxies", "prune_proxies",
    "peaks_path", "compute_peaks", "save_peaks", "load_peaks",
    "clip_peaks_path", "clip_peaks", "prune_clip_peaks",
    "edit_path", "load_edit", "save_edit", "validate_edit", "normalise_edit",
    "migrate_edit", "clip_kind", "clip_brightness", "clip_carries_media",
    "clip_audio", "clip_audio_drift", "clip_audio_resync", "clip_muted",
    "clip_effects", "clip_length",
    "audio_effects", "audio_gain_points", "audio_gain_at",
    "MIX_BED_GAIN", "MIX_DUCK", "MIX_LEGACY_BED_GAIN", "MIX_DUCK_GAIN",
    "MIX_DUCK_ATTACK", "MIX_DUCK_RELEASE", "MIX_CEILING",
    "MIX_REPAIR_VERSION", "audio_mix", "bed_length", "audible_strips",
    "bed_duck_points", "bed_duck_suppressed", "bed_gain_points",
    "bed_gain_at", "bed_render_gain", "heal_mix",
    "overlay_items", "overlay_kind", "OVERLAY_SUFFIXES",
    "blocking_errors", "WARNING_CODES", "repair_audio_overlaps",
    "edit_digest", "session_token_path", "claim_session", "current_session",
    "session_is_current", "SESSION_STALE_AFTER",
    "edit_from_plan", "edit_to_cuts", "edit_duration", "edit_gaps",
    "edit_sync_flags",
    "music_window", "new_clip", "EditError", "EditConflict",
    "board_write_lock", "on_disk_revision",
    "history_dir", "archive_edit", "prune_history", "list_history",
    "restore_edit",
    "drafts_dir", "load_draft_index", "list_drafts", "create_draft",
    "duplicate_draft", "rename_draft", "delete_draft", "activate_draft",
    "write_backup", "pending_backup", "recover_backup", "discard_backup",
    "DRAFT_NAME_MAX",
    "export_nle", "NLE_FPS",
    "film_fps", "frame_seconds", "quantise_gap", "heal_subframe_gaps",
]


# 2 (2026-08-18) — clips grew `kind` and `adjust`. The bump is not decoration:
# `validate_edit` HARD-REFUSES any version it does not recognise, so a v2
# document opened by an older Phosphene stops loudly instead of silently
# dropping every slug and sliding the whole film off its beats. The read path
# upgrades v1 in place (`migrate_edit`), so nothing anyone has already cut has
# to be touched or re-saved.
EDIT_VERSION = 2
PEAKS_VERSION = 1

# WHAT A CLIP CAN BE. Three, and the list is closed on purpose:
#   video — a file with a picture and a source window. What the machine made.
#   still — an image held for the length of its slot. What the machine cannot
#           make: a title card, a photograph, a beat of nothing moving.
#   slug  — black, for a duration, with no file at all. Descript calls it a gap
#           clip; it is the primitive the render used to apologise for not
#           having ("N gap(s) … were closed by the concatenation").
# An ABSENT kind is a video, so every edit.json ever written is already valid.
CLIP_KINDS = ("video", "still", "slug")

# The one adjustment. ffmpeg's `eq=brightness` is an ADDITIVE offset in
# [-1, 1]; half of that range is already past "this shot is unusable" in both
# directions, and a clamp the validator enforces is what stops a typo in a
# saved document from rendering an hour of white.
BRIGHTNESS_LIMIT = 0.5

# Bumped whenever the proxy RECIPE changes. It is part of the content hash, so
# a bump rebuilds every proxy on the next prepare instead of leaving a mixed
# population of old and new files that all look equally valid.
PROXY_RECIPE_VERSION = 3

# 640 was chosen for seek speed and it made the preview look like a bad stream:
# the stage is ~1360 px wide on a normal window, so a 640-wide proxy was being
# upscaled >2x and the owner read it, correctly, as "the quality is pretty bad".
# All-intra is what makes seeking fast, not the pixel count — 1280 costs bytes
# on disk and nothing in seek latency.
PROXY_WIDTH = 1280
PROXY_CRF = "23"
PROXY_PRESET = "veryfast"
PROXY_PIX_FMT = "yuv420p"
PROXY_ABITRATE = "96k"     # proxy audio: intelligible, not archival

PEAKS_BUCKETS_PER_SECOND = 100
PEAKS_SCALE = 127          # peaks are ints in [-127, 127]
PEAKS_SR = 22050

# A timeline clip's source window and its slot on the film must be the same
# length — nothing here plays at anything but 1x. One millisecond of slack
# absorbs float rounding through JSON and no more.
# How many past saves each film keeps in history/. Fifty debounced saves at
# ~5 KB each is a quarter-megabyte for a full working session — nothing, next
# to what losing one good arrangement costs.
EDIT_HISTORY_KEEP = 50

LENGTH_TOLERANCE = 0.002
# Two clips that overlap by less than a frame at 24 fps are touching, not
# overlapping; the UI's drag maths lands there constantly.
TOUCH_TOLERANCE = 1.0 / 48.0

# Codes `validate_edit` reports but NO writer may refuse on. A warning is a
# note on a document that is going to disk either way — the rule is that
# persisting the user's work always wins, because the alternative was a red
# banner over an afternoon of cutting that could not be stored anywhere.
WARNING_CODES = frozenset({"clips_audio_overlap"})


class EditError(Exception):
    """Raised for an edit that must not be written. Message is user-facing."""


class EditConflict(EditError):
    """The document moved on between the caller's read and its write.

    A SUBCLASS SO NOBODY HAS TO CATCH IT TO STAY CORRECT: every existing
    `except EditError` around a save still refuses the write and still shows a
    sentence. A caller that WANTS to tell "somebody else got there first" from
    "this document is invalid" catches this first and reads `.revision`, which
    is what is actually on disk right now.
    """

    def __init__(self, message: str, *, revision: int = 0) -> None:
        super().__init__(message)
        self.revision = int(revision)


def _f(x, default: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    if v != v or v in (float("inf"), float("-inf")):
        return default
    return v


def _slug(text: str, limit: int = 40) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", str(text or "")).strip("-").lower()
    return (s[:limit] or "clip").strip("-") or "clip"


# The suffixes the media pool itself accepts as pictures, plus the two the
# overlay lane's own resolver already reads. One list, so a file that is a
# still on one lane cannot be a video on the other.
STILL_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff")


def _looks_like_a_still(path) -> bool:
    return str(path or "").strip().lower().endswith(STILL_SUFFIXES)


def clip_kind(clip) -> str:
    """`video` | `still` | `slug`. ABSENT IS `video`, and that is the migration.

    Every clip in every edit.json written before 2026-08-18 has no `kind`, and
    every one of them is a video. Reading the default here rather than stamping
    it on disk means a v1 document is v2-correct the moment it is loaded, with
    no rewrite pass and nothing to go wrong halfway through one.
    """
    if not isinstance(clip, dict):
        return "video"
    k = str(clip.get("kind") or "").strip().lower()
    if k in CLIP_KINDS:
        return k
    # ABSENT NO LONGER MEANS VIDEO ON ITS OWN. The paragraph above was true
    # while only renders reached the picture lane; a pool image can land here
    # now, and an unstamped .png answered "video" — which handed a still to a
    # <video> element (format error, black stage) and to the concat graph.
    # The overlay lane has always read the suffix when the field is absent
    # (`sbeOvKind`); this is the same rule, so the two lanes cannot disagree.
    # A pre-2026-08-18 document is unaffected: its clips are all .mp4.
    if _looks_like_a_still(clip.get("path")):
        return "still"
    return "video"


def clip_carries_media(clip) -> bool:
    """Does this clip point at a file on disk?

    A slug does not — it is `color=black` in the filtergraph and nothing else —
    which is why proxy planning, relink and the media-pool round trip all have
    to ask before they touch `clip["path"]`.
    """
    return clip_kind(clip) in ("video", "still")


# FRAMING — a zoom and a reframe, per clip. `frame: {zoom, x, y}`: the
# picture is magnified `zoom` times (1 is the whole frame, 3 is a third of
# it) and the window is centred at the fraction (`x`, `y`) of the source.
# A push-in on a face, a wider frame's crop to a tighter one, a still that
# needs to lose its margins — the reframe every NLE has and the render did
# not. Not an "effect" (`fx` is fades): it is a transform, and it keeps its
# own home the way brightness kept `adjust`. Neutral is absent.
FRAME_ZOOM_MIN = 1.0
FRAME_ZOOM_MAX = 3.0


def clip_frame(clip) -> dict:
    """`{zoom, x, y}`, clamped; `{1.0, 0.5, 0.5}` for a clip that has none."""
    c = clip if isinstance(clip, dict) else {}
    f = c.get("frame")
    f = f if isinstance(f, dict) else {}
    z = _f(f.get("zoom"), 1.0)
    if z != z or z <= 0:
        z = 1.0
    z = max(FRAME_ZOOM_MIN, min(FRAME_ZOOM_MAX, z))
    return {"zoom": round(z, 6),
            "x": round(max(0.0, min(1.0, _f(f.get("x"), 0.5))), 6),
            "y": round(max(0.0, min(1.0, _f(f.get("y"), 0.5))), 6)}


def clip_frame_is_neutral(clip) -> bool:
    f = clip_frame(clip)
    return abs(f["zoom"] - 1.0) < 1e-9


def clip_brightness(clip) -> float:
    """`adjust.brightness`, clamped, or 0.0 — the neutral that renders nothing.

    Zero is not "apply eq=brightness=0", it is "insert no filter at all", so an
    untouched clip's segment in the graph stays byte-identical to the graph
    that existed before adjustments did.
    """
    if not isinstance(clip, dict):
        return 0.0
    adj = clip.get("adjust")
    if not isinstance(adj, dict):
        return 0.0
    b = _f(adj.get("brightness"), 0.0)
    return max(-BRIGHTNESS_LIMIT, min(BRIGHTNESS_LIMIT, b))


def clip_audio(clip) -> dict:
    """The clip's SOUND window: `{start, end, film_start, linked}`.

    THE J-CUT AND THE L-CUT, and they are the same feature. The owner: "like a
    normal video editor... the sound attached to every clip... in case I want
    to move the clip, cut a little bit of the image, but leave the sound below
    and then connect it to make better transitions." And again: "you can edit
    dialogue in a way that you hear her voice while you're seeing him... I
    need to be able to leave some of the audio and drag only the image."

    ABSENT MEANS LINKED, and that is the whole migration: every clip ever
    written has no `audio` key and every one of them plays its own sound under
    its own picture. So EDIT_VERSION does not move, no document is rewritten,
    and a clip only carries the field once somebody has actually pulled the
    two apart.

    NOT A SECOND AUDIO TRACK. The refuse list still holds: one video track,
    one music lane, and per-clip sound that is linked by default. The audio
    windows may not overlap each other any more than the pictures may — a
    split edit is a butt join that lands somewhere else, not a mix.
    """
    if not isinstance(clip, dict):
        return {"start": 0.0, "end": 0.0, "film_start": 0.0,
                "linked": True, "coupled": False, "split": False,
                "speed": 1.0, "len": 0.0}
    vs, ve = _f(clip.get("start")), _f(clip.get("end"))
    fs = _f(clip.get("film_start"))
    # THE SOUND RUNS AT THE CLIP'S SPEED, linked or not: it came off the same
    # take, and retiming the take retimes both halves. `len` is the strip's
    # length ON THE FILM — the only clock the lane, the plan and the envelope
    # use — so no caller has to remember to divide.
    speed = clip_speed(clip)
    a = clip.get("audio")
    if not isinstance(a, dict) or clip_kind(clip) != "video":
        return {"start": round(vs, 6), "end": round(ve, 6),
                "film_start": round(fs, 6),
                "linked": True, "coupled": False, "split": False,
                "speed": speed,
                "len": round(max(0.0, ve - vs) / speed, 6)}
    # THE PRESENCE OF THE FIELD IS THE SWITCH, not the values in it. Deriving
    # "linked" from equality looked tidier and was wrong in the one case that
    # matters: unlinking writes the window the clip already has, so a clip the
    # user had just unlinked read as linked and refused to be dragged. The
    # toggle adds the object or deletes it; nothing else decides.
    #
    # ...AND `audio.linked` IS THE THIRD STATE. Re-linking used to DELETE the
    # field, which snapped the sound back under the picture and threw away the
    # J-cut the user had just built. The owner, describing what he wanted:
    # "You just drag it, and the sound below stays, and then you can lock it
    # and move it, and then the sound starts before the clip starts." So
    # re-linking now FREEZES the relationship instead: the window stays, the
    # flag says the two travel together, and the pair moves as one from then
    # on. Absent still means linked-and-in-sync, so no document written before
    # today changes and an in-sync re-link still deletes the field outright.
    s = _f(a.get("start"), vs)
    e = _f(a.get("end"), ve)
    f = _f(a.get("film_start"), fs)
    coupled = a.get("linked") is True
    return {"start": round(s, 6), "end": round(e, 6),
            "film_start": round(f, 6),
            # `linked` has always meant "this strip cannot be dragged on its
            # own", and that is true of a coupled pair too.
            "linked": coupled, "coupled": coupled, "split": True,
            "speed": speed, "len": round(max(0.0, e - s) / speed, 6)}


def clip_muted(clip) -> bool:
    """Is this clip's OWN sound silenced? `mute: true`, and nothing else.

    THE OWNER'S CASE: "We should have an option to mute the clip sound." An H3
    shot arrives with baked-in wind and ambience under the line, and on a music
    cut that is not a performance to be balanced — it is noise to be removed so
    the track can carry the moment.

    NOT THE SAME AS `has_audio: false`, which says the FILE has no audio track
    and is a fact about the source. This is a decision about the edit, and the
    two are painted differently for that reason.

    NOT THE SAME AS UNLINKING EITHER, and it composes with it in both
    directions: a J-cut can be muted (the strip stays where it was put, and is
    silent), and a muted clip can then be unlinked and slid. `mute` describes
    the sound wherever its strip happens to be.

    ABSENT IS AUDIBLE, the same migration `audio` gets: every clip ever written
    plays its own sound, so EDIT_VERSION does not move and nothing is rewritten
    until somebody actually presses the button.
    """
    if not isinstance(clip, dict):
        return False
    return clip.get("mute") is True and clip_kind(clip) == "video"


# SPEED, ON THE CLIP. `speed: 2.0` plays the source window twice as fast, so
# the slot on the film is `(end - start) / speed`. The bounds are ffmpeg's own
# honest range for `atempo` chained twice each way; below 0.25x a clip is a
# still with ambitions and above 4x it is a strobe. An ABSENT speed is 1.0,
# which is every clip ever written, so EDIT_VERSION does not move.
#
# NEVER AUTOMATIC. The owner's verdict on a slowed shot that read as an
# accident was "too slow-mo"; this is a control a person sets on a clip, not a
# thing the editor decides.
SPEED_MIN = 0.25
SPEED_MAX = 4.0


def clip_speed(clip) -> float:
    """The clip's play rate, clamped. 1.0 when absent, and always 1.0 for a
    still or a slug — they have no source clock to run fast or slow."""
    if not isinstance(clip, dict) or clip_kind(clip) != "video":
        return 1.0
    s = _f(clip.get("speed"), 1.0)
    if s <= 0:
        return 1.0
    return round(max(SPEED_MIN, min(SPEED_MAX, s)), 6)


def clip_length(clip) -> float:
    """How long this clip plays. The film slot and the source window agree —
    at the clip's speed."""
    if not isinstance(clip, dict):
        return 0.0
    n = _f(clip.get("film_end")) - _f(clip.get("film_start"))
    if n <= 0:
        n = (_f(clip.get("end")) - _f(clip.get("start"))) / clip_speed(clip)
    return max(0.0, round(n, 6))


def clip_effects(clip) -> dict:
    """THE ONE ACCESSOR. `{fade_in, fade_out, brightness}`, whatever the storage.

    See docs/EDITOR_EFFECTS_MODEL.md. `clip.fx` is the home for effects and
    absent means none, so no document ever written has to change and
    EDIT_VERSION does not move.

    BRIGHTNESS IS THE ONE LEGACY CITIZEN. It predates the model and stays at
    `clip.adjust.brightness` — a label is not worth a data migration, the same
    reasoning that kept `film_start` when the user-facing noun became
    "sequence". What changes is that "where is it stored" stops being a
    question any consumer has to answer: every output reads this and nothing
    else, so the next effect picks its storage on its own merits instead of
    adding a fourth code path.

    THE CLAMP LIVES HERE so all three outputs are handed numbers that are
    already legal. Two fades that crossed would ask ffmpeg for an opacity that
    is two things at once and would hand the NLEs keyframes out of order.
    """
    c = clip if isinstance(clip, dict) else {}
    fx = c.get("fx")
    fx = fx if isinstance(fx, dict) else {}
    n = clip_length(c)
    fin = max(0.0, _f(fx.get("fade_in")))
    fout = max(0.0, _f(fx.get("fade_out")))
    if n > 0:
        fin, fout = min(fin, n), min(fout, n)
        over = fin + fout - n
        if over > 0:
            # Give the overrun back proportionally rather than truncating one
            # of them: an edit that asked for two long fades meant "mostly
            # ramp", and silently zeroing the out-fade is not that.
            total = fin + fout
            fin, fout = fin - over * (fin / total), fout - over * (fout / total)
    return {"fade_in": round(fin, 6), "fade_out": round(fout, 6),
            "brightness": clip_brightness(c)}


# ---------------------------------------------------------------------------
# THE OVERLAY LANE — a SECOND video track, above the picture
# ---------------------------------------------------------------------------
# WHY ITS OWN LIST AND NOT A `kind` ON `clips`. The picture lane is ONE track
# and `validate_edit` refuses any two clips that overlap on it — that rule is
# what stops a concat from being handed two pictures for the same second. An
# overlay's whole purpose is to sit ON one of them, so putting overlays in the
# same list would mean weakening the rule that keeps the picture lane honest.
# A second list keeps both rules exact: pictures may not overlap each other,
# overlays may not overlap each other, and an overlay over a picture is not an
# overlap at all — it is the feature.
#
# The owner's case: an endcard PNG with real transparency, laid over the last
# seconds of the pull-out so the card sits in the sky.
OVERLAY_SUFFIXES = (".png", ".webp", ".tif", ".tiff")


# WHAT AN OVERLAY CAN BE. A card is a still; a matte or a loop is a video; a
# TITLE is `text` — pixels the render DRAWS rather than a file somebody
# uploaded. It is a citizen of this lane and not a system of its own because
# everything a title needs already exists here: alpha, fades, a slot on the
# film, z-order over the picture, and the one-lane rule. What a title adds is
# the string and how to set it.
OVERLAY_KINDS = ("still", "video", "text")


def overlay_kind(item) -> str:
    """`still`, `video` or `text`. An EXPLICIT kind wins; the suffix decides the rest.

    The explicit field has to win because a title has no path to probe, and
    the suffix fallback has to stay because every overlay written before titles
    existed carries only a path — so every existing document reads exactly as
    it did.
    """
    if not isinstance(item, dict):
        return "still"
    k = str(item.get("kind") or "").strip().lower()
    if k in OVERLAY_KINDS:
        return k
    return "still" if str(item.get("path") or "").lower().endswith(
        OVERLAY_SUFFIXES) else "video"


# ---------------------------------------------------------------------------
# TITLES — text on the overlay lane
# ---------------------------------------------------------------------------
# THE STYLE IS A SMALL, CLOSED SET, and every value is clamped by the one
# accessor so the preview, the render and any export are handed numbers that
# are already legal. `font_size` is in pixels AT A 1080-HIGH FRAME and scales
# with the frame the film is actually rendered at, so a title designed on the
# stage looks the same on a 576p draft and a 720p delivery. `x`/`y` are the
# anchor as a FRACTION of the frame — the same reason — and `align` says which
# edge of the text sits on `x`.
TEXT_STYLE_DEFAULTS = {
    "font_size": 64,
    "color": "#ffffff",
    "align": "center",
    "x": 0.5,
    "y": 0.5,
    "box": False,
    "box_color": "#000000",
    "box_opacity": 0.5,
}
TEXT_REFERENCE_HEIGHT = 1080
TEXT_FONT_SIZE_MIN = 8
TEXT_FONT_SIZE_MAX = 400
TEXT_MAX_CHARS = 400
TEXT_ALIGNS = ("left", "center", "right")
_HEX_RE = re.compile(r"^#[0-9a-f]{6}$")

# THE FONT IS RESOLVED EXPLICITLY, to a FILE, and verified before anything is
# drawn. Letting a text renderer "discover" a font goes wrong quietly on a
# machine with no fontconfig — the failure other editors have documented — so
# the panel names the file it will use, in this order, and refuses with a
# sentence when none of them exists. `LTX_TITLE_FONT` overrides the list.
# The stylesheet's stack for the on-stage preview is the same family order.
TITLE_FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
)


def _hex_colour(v, default: str) -> str:
    s = str(v or "").strip().lower()
    if re.fullmatch(r"#[0-9a-f]{3}", s):
        s = "#" + "".join(ch * 2 for ch in s[1:])
    return s if _HEX_RE.match(s) else default


def overlay_text(item) -> dict:
    """`{text, style}` for a title, clamped. THE ONE ACCESSOR.

    Absent style keys read the defaults, so a title written as `{"kind":
    "text", "text": "FIN"}` is complete. Every consumer reads this and nothing
    else, which is what keeps the stage and the render drawing the same card.
    """
    it = item if isinstance(item, dict) else {}
    text = str(it.get("text") or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")[:TEXT_MAX_CHARS]
    raw = it.get("style")
    raw = raw if isinstance(raw, dict) else {}
    d = TEXT_STYLE_DEFAULTS
    size = _f(raw.get("font_size"), d["font_size"])
    size = max(TEXT_FONT_SIZE_MIN, min(TEXT_FONT_SIZE_MAX, size))
    align = str(raw.get("align") or d["align"]).strip().lower()
    if align not in TEXT_ALIGNS:
        align = d["align"]
    style = {
        "font_size": round(size, 3),
        "color": _hex_colour(raw.get("color"), d["color"]),
        "align": align,
        "x": round(max(0.0, min(1.0, _f(raw.get("x"), d["x"]))), 6),
        "y": round(max(0.0, min(1.0, _f(raw.get("y"), d["y"]))), 6),
        "box": raw.get("box") is True,
        "box_color": _hex_colour(raw.get("box_color"), d["box_color"]),
        "box_opacity": round(max(0.0, min(1.0, _f(raw.get("box_opacity"),
                                                 d["box_opacity"]))), 6),
    }
    return {"text": text, "style": style}


def title_font_path() -> Path | None:
    """The font FILE a title is drawn with, or None when there is none."""
    env = os.environ.get("LTX_TITLE_FONT")
    if env:
        p = Path(env)
        return p if p.is_file() else None
    for c in TITLE_FONT_CANDIDATES:
        p = Path(c)
        if p.is_file():
            return p
    return None


def title_font_problem() -> str | None:
    """A sentence when no title can be drawn on this machine, else None.

    Asked at COMPILE time — before the ffmpeg command is built — so a missing
    font is a refusal with a reason, never a film with a hole where the title
    was.
    """
    env = os.environ.get("LTX_TITLE_FONT")
    if env and not Path(env).is_file():
        return (f"LTX_TITLE_FONT points at {env}, and there is no font file "
                f"there — fix the path or unset it to use the system font")
    if title_font_path() is None:
        return ("no font for titles: none of the system fonts the panel knows "
                "is installed — set LTX_TITLE_FONT to a .ttf file")
    return None


def _rgba(hex_colour: str, alpha: float) -> tuple:
    h = hex_colour.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16),
            int(round(255 * max(0.0, min(1.0, alpha)))))


def title_fingerprint(item, width: int, height: int) -> str:
    """What a title's pixels depend on, hashed — the raster's cache key."""
    t = overlay_text(item)
    font = title_font_path()
    raw = json.dumps([t, int(width), int(height), str(font) if font else "",
                      int(font.stat().st_mtime_ns) if font else 0],
                     sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8", "replace")).hexdigest()[:16]


def render_title(item, width: int, height: int, dest) -> Path:
    """Draw a title as an RGBA PNG the size of the frame. Returns the path.

    WHY A PICTURE AND NOT A FILTER. The render composites the overlay lane
    with one chain — `format=rgba`, scale, alpha fade, `overlay=` under an
    `enable=` window — and a title that arrives as a frame-sized RGBA image
    inherits every part of it: the fades, the z-order, the one-lane rule, the
    tpad past the last shot. A separate text filter would be a second path
    through that chain, and it would also depend on the ffmpeg BUILD: the
    text filter is optional at compile time and the Homebrew ffmpeg this very
    panel resolves on its author's machine does not carry it. A PNG needs
    nothing of ffmpeg but what every other card already needs.

    The font is `title_font_path()`, verified by `title_font_problem()` before
    this is ever called; a caller that skips the check gets an EditError here.
    """
    from PIL import Image, ImageDraw, ImageFont            # noqa: PLC0415
    problem = title_font_problem()
    if problem:
        raise EditError(problem)
    W, H = max(2, int(width)), max(2, int(height))
    t = overlay_text(item)
    st = t["style"]
    text = t["text"] or " "
    px = max(1, int(round(st["font_size"] * H / float(TEXT_REFERENCE_HEIGHT))))
    font = ImageFont.truetype(str(title_font_path()), px)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    spacing = max(0, int(round(px * 0.25)))
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing,
                                   align=st["align"])
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    ax, ay = st["x"] * W, st["y"] * H
    left = ax - tw * {"left": 0.0, "center": 0.5, "right": 1.0}[st["align"]]
    # KEPT INSIDE THE FRAME, the same clamp the stage applies: a wide title
    # anchored near an edge is pushed back in rather than cut off.
    left = max(0.0, min(float(W - tw), left))
    top = ay - th / 2.0
    if st["box"]:
        pad = px * 0.4
        draw.rectangle([left - pad, top - pad, left + tw + pad, top + th + pad],
                       fill=_rgba(st["box_color"], st["box_opacity"]))
    draw.multiline_text((left - bbox[0], top - bbox[1]), text, font=font,
                        fill=_rgba(st["color"], 1.0), spacing=spacing,
                        align=st["align"])
    dest = Path(str(dest))
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    img.save(tmp, "PNG")
    os.replace(tmp, dest)
    return dest


def overlay_items(edit) -> list[dict]:
    """The overlay lane, in film order. Absent is an empty lane."""
    rows = (edit or {}).get("overlays")
    if not isinstance(rows, list):
        return []
    out = [o for o in rows if isinstance(o, dict)]
    out.sort(key=lambda o: (_f(o.get("film_start")), str(o.get("id") or "")))
    return out


# ---------------------------------------------------------------------------
# THE SOUND'S OWN ENVELOPE — fades, and the keyframes underneath them
# ---------------------------------------------------------------------------
# "Fading and fade-out with keyframes should be very simple and intuitive for
# the sound as well."
#
# BOTH, FROM ONE MODEL. `fade_in`/`fade_out` are the simple case and the only
# thing the corner handles touch; `points` are the control case. They are not
# two features with two code paths — `audio_gain_points()` folds the fades into
# the point list and returns ONE breakpoint curve, and the preview, the render
# and the export all read that. So the simple case never has to discover
# keyframes, and a keyframed envelope never has to be re-expressed.
#
# `t` IS STRIP-RELATIVE SECONDS, never film seconds: a J-cut that slides its
# sound half a second earlier must not move every keyframe with it. Gain is
# LINEAR 0..1, because that is what ffmpeg's `volume` takes and what a level
# line on a strip means; dB is an export-time concern.
def audio_effects(item, length: float = 0.0) -> dict:
    """`{fade_in, fade_out, points}` for a strip or the bed. Clamped."""
    it = item if isinstance(item, dict) else {}
    afx = it.get("afx")
    afx = afx if isinstance(afx, dict) else {}
    n = max(0.0, _f(length))
    fin = max(0.0, _f(afx.get("fade_in")))
    fout = max(0.0, _f(afx.get("fade_out")))
    if n > 0:
        fin, fout = min(fin, n), min(fout, n)
        over = fin + fout - n
        if over > 0:
            total = fin + fout
            fin, fout = fin - over * (fin / total), fout - over * (fout / total)
    pts = []
    for row in (afx.get("points") or []):
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            continue
        t, g = _f(row[0]), _f(row[1])
        if t != t or g != g:
            continue
        if n > 0:
            t = max(0.0, min(n, t))
        pts.append([round(max(0.0, t), 6), round(max(0.0, min(1.0, g)), 6)])
    pts.sort(key=lambda r: r[0])
    return {"fade_in": round(fin, 6), "fade_out": round(fout, 6),
            "points": pts}


def _lerp_gain(pts, t: float) -> float:
    """The point curve at `t`. No points is a flat 1.0."""
    if not pts:
        return 1.0
    if t <= pts[0][0]:
        return pts[0][1]
    if t >= pts[-1][0]:
        return pts[-1][1]
    for (t0, g0), (t1, g1) in zip(pts, pts[1:]):
        if t0 <= t <= t1:
            if t1 - t0 <= 1e-9:
                return g1
            return g0 + (g1 - g0) * ((t - t0) / (t1 - t0))
    return pts[-1][1]


def audio_gain_points(item, length: float) -> list[list[float]]:
    """THE ONE CURVE: `[[t, gain], ...]` with the fades folded in.

    Every output reads this and nothing else, so a fade and a keyframe cannot
    disagree about what the sound does. Breakpoints are the union of the
    envelope's own points and the fade corners, each multiplied by the fade
    factor there — which is what makes a fade and a keyframed dip compose
    instead of one overriding the other.

    THE CLOCK IS THE STRIP AS IT PLAYS ON THE FILM, and speed does not move
    it. `length` is `clip_audio(c)["len"]` — the played length, already
    divided by the clip's speed — and `t` in every point and every fade is a
    second of THAT strip. Decided here, once, because the alternative (points
    on the source clock, scaled at read time) would put a fade of "1 s" in
    source seconds while the person typed it in film seconds, and the ffmpeg
    `volume` term runs AFTER `atempo`, on the played clock, so this is the only
    unit the render could apply without a conversion nothing else performs.
    A keyframe at 2 s stays at 2 s of the strip when the clip is retimed; the
    strip is what gets shorter or longer.
    """
    n = max(0.0, _f(length))
    if n <= 0:
        return []
    e = audio_effects(item, n)
    fin, fout, pts = e["fade_in"], e["fade_out"], e["points"]
    marks = {0.0, n}
    for t, _ in pts:
        marks.add(min(n, max(0.0, t)))
    if fin > 1e-9:
        marks.add(fin)
    if fout > 1e-9:
        marks.add(max(0.0, n - fout))
    out = []
    for t in sorted(marks):
        g = _lerp_gain(pts, t)
        if fin > 1e-9 and t < fin:
            g *= t / fin
        if fout > 1e-9 and t > n - fout:
            g *= max(0.0, (n - t) / fout)
        out.append([round(t, 6), round(max(0.0, min(1.0, g)), 6)])
    # A FLAT UNITY CURVE IS NO CURVE. Saying "the volume is 1 the whole way"
    # would put a filter in every graph to express nothing.
    if all(abs(g - 1.0) < 1e-9 for _, g in out):
        return []
    return out


def audio_gain_at(item, length: float, t: float) -> float:
    """The gain at one second. The preview's per-frame answer."""
    curve = audio_gain_points(item, length)
    if not curve:
        return 1.0
    return round(_lerp_gain(curve, max(0.0, _f(t))), 6)


# ---------------------------------------------------------------------------
# THE MIX — the bed's level, and what happens to it under a line
# ---------------------------------------------------------------------------
# THE DEFECT THIS CLOSES: the renderer was a SECOND AUTHOR of the soundtrack's
# level, and an invisible one. Under `mode: "under"` the ffmpeg graph held the
# bed at a hard-coded 0.20 (-14 dB) and then pushed it down a further ~11 dB
# through a `sidechaincompress` keyed on the clips' own audio — two decisions
# nobody made, in no document, on no screen. The preview applied neither, so
# the one surface the user checks his work on played a mix the file never had:
# "when you render it, there are some weird manipulations... the volume of the
# music goes low when the dialogue appears."
#
# The duck's own comment defended itself by saying it kept lines intelligible
# "without anybody automating a volume curve by hand". That was true when it
# was written and is false now — the bed and the strips both carry fades,
# keyframes and level lines, so the renderer is overriding numbers the user
# explicitly authored. Same shape as the save model's two-writers-no-rule.
#
# So the mix is a MODEL, and `docs/EDITOR_EFFECTS_MODEL.md` decides its shape:
# if the preview and the render can disagree about a gain, that gain is not in
# the model. A compressor's output is not a value in a document — it is a
# function of samples the browser never sees — so the duck is expressed the
# way every other effect here is: as a curve derived from the document, read
# identically by the preview, the ffmpeg render and (were it ever wanted) the
# NLE export. `bed_gain_points` is the one accessor, the way `clip_effects`
# and `audio_gain_points` are for their subjects.

# What a NEW timeline gets, and it is the honest one: the bed plays at the
# level the file is at, and nothing moves it.
MIX_BED_GAIN = 1.0
MIX_DUCK = False

# What the renderer USED to do to every `under` mix, with nobody's consent.
# `migrate_edit` writes this pair onto documents that predate the controls, so
# a film the owner has already approved keeps the levels it was approved at —
# the difference being that now they are on screen and he can change them.
MIX_LEGACY_BED_GAIN = 0.2

# HOW FAR THE BED STEPS BACK, and the numbers are the measured ones. The old
# `sidechaincompress` was tuned on a gated-tone rig (tone-under-tone, band
# split so neither leaked into the other's reading): threshold 0.04 / ratio 8
# measured 5.8 dB, which is not enough to hear under a line; 0.01 / 20
# measured 17.7 dB, which audibly pumps; 0.02 / 10 measured **11.4 dB** with
# the bed returning to full level between lines, which is the broadcast range.
# 11.4 dB down is a linear 0.269, so that is the depth the envelope uses and
# the reason it is not a round number.
MIX_DUCK_GAIN = 0.269
# Fast in so the bed is already down on the first syllable; slow out so it
# does not pump between the words of a sentence. Same pair as the compressor's
# attack/release, in seconds.
MIX_DUCK_ATTACK = 0.005
MIX_DUCK_RELEASE = 0.4

# THE SAFETY LIMITER, and it is NOT an artistic choice — it is the only thing
# standing between a hot line over a bed and a hard-clipped output. `amix`
# runs with `normalize=0` (halving both inputs to protect a headroom budget we
# have already set deliberately would be worse), so nothing else protects the
# sum: engine dialogue comes out at 0.35 RMS, and dialogue plus bed peaked the
# first under-mix film at 1.31 pre-encode with 1341 hard-clipped samples and
# not a word said about it. A THRESHOLD, not a ceiling — tanh saturates
# smoothly above it, so the true peak lands a little over (0.9 measured 0.98
# on the real film) while nothing hard-clips.
MIX_CEILING = 0.9

# THE MIX HEAL IS A ONE-TIME STAMP, so it needs a marker — unlike the
# sub-frame gap heal, which is arithmetic that finds nothing to do on a second
# run. Without the marker, a user who deliberately set the bed back to 1.0 and
# switched the duck off would have `normalise_edit` drop the now-neutral `mix`
# key and the next READ would helpfully put 0.20-and-ducked back: the rival
# author this whole file exists to remove, wearing a repair's clothes.
MIX_REPAIR_VERSION = 1


def audio_mix(audio) -> dict:
    """THE ONE ACCESSOR for the mix: `{bed_gain, duck}`, whatever the storage.

    Absent means the new default, which is "play the track and leave it
    alone". Every consumer reads this and nothing else, so "is there a mix
    block on this document" stops being a question anybody has to answer.
    """
    a = audio if isinstance(audio, dict) else {}
    m = a.get("mix")
    m = m if isinstance(m, dict) else {}
    g = m.get("bed_gain")
    gain = MIX_BED_GAIN if g is None else max(0.0, min(1.0, _f(g)))
    d = m.get("duck")
    return {"bed_gain": round(gain, 6),
            "duck": MIX_DUCK if d is None else bool(d)}


def bed_length(audio, film_len: float = 0.0) -> float:
    """How many seconds of the bed actually PLAY. The bed envelope's clock.

    THE BED'S ENVELOPE IS ON THE PLAYED WINDOW, not on the track, exactly as a
    clip strip's envelope is on the strip and not on the source file. Zero is
    the first second you hear, so trimming the head does not slide every
    keyframe with it and a fade dragged onto the block's corner means the
    corner it was dragged onto.

    THE FILM IS THE FALLBACK CLOCK, and this is the whole of the fix for the
    one shape the first mix pass left behind. A soundtrack whose document says
    nothing about how long the track is — no `duration`, no `trim_end` — used
    to make this return 0, which made `bed_gain_points` return an EMPTY curve,
    which means "no filter", which means the render played the bed at UNITY
    under the dialogue while the preview's own model drew it 19.6 dB lower.
    Silence-by-empty-curve is the one behaviour nobody asked for and it is the
    LOUD one, which is what made it dangerous rather than merely wrong.

    The honest reading of a bed with no stated length is that it plays under
    the FILM: the renderer trims the mix to the film anyway, so the last second
    anybody hears is the film's last second. That is a number both the browser
    and the render can compute from the same document, which is the only test
    a value in this model has to pass. `film_len` is `edit_duration(edit)` —
    the timeline's clock — and it is only ever consulted when the track's own
    length is absent.
    """
    a = audio if isinstance(audio, dict) else {}
    w = music_window(a)
    end = w["end"]
    if end is None:
        end = _f(a.get("duration"))
        if end <= 0:
            # WHAT REMAINS OF THE FILM after the bed starts, not the film's
            # whole length: a track that comes in at 0:04 of a six-second film
            # is heard for two seconds, and its envelope's clock is those two.
            return round(max(0.0, _f(film_len) - w["film_start"]), 6)
    return round(max(0.0, _f(end) - w["start"]), 6)


def audible_strips(edit) -> list[list[float]]:
    """The film seconds where a clip's OWN sound is playing: `[[start, end]]`.

    What the duck is keyed on, and the whole reason it can be a document value
    at all. A compressor asks "are these samples loud"; this asks "is there a
    sound strip here", which is a question the document answers and the
    browser can answer too. A clip with no audio track (`has_audio` false) and
    a MUTED clip are both silence — neither contributes a lane to the render,
    so neither may duck anything.

    Windows are merged when they touch, overlap, or sit closer together than
    one release: a bed that recovered fully in the eighth of a second between
    two lines would pump, which is the failure the release time exists to
    prevent, and merging is also what keeps the returned list non-overlapping
    so the curve built from it has its knots exactly at the marks.
    """
    wins: list[list[float]] = []
    for c in (edit or {}).get("clips") or []:
        if not isinstance(c, dict) or clip_kind(c) != "video":
            continue
        if c.get("has_audio") is False or clip_muted(c):
            continue
        w = clip_audio(c)
        s = _f(w["film_start"])
        e = s + w["len"]
        if e - s > 1e-9:
            wins.append([round(s, 6), round(e, 6)])
    wins.sort()
    out: list[list[float]] = []
    for s, e in wins:
        if out and s - out[-1][1] < MIX_DUCK_RELEASE - 1e-9:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def _duck_gain_at(wins, t: float) -> float:
    """The duck's value at one second, from the merged windows. Piecewise."""
    g = 1.0
    for s, e in wins:
        if t < s - 1e-9:
            continue
        if t <= e + 1e-9:
            k = 1.0 if MIX_DUCK_ATTACK <= 0 else min(1.0, (t - s) / MIX_DUCK_ATTACK)
            v = 1.0 - (1.0 - MIX_DUCK_GAIN) * max(0.0, k)
        else:
            k = 1.0 if MIX_DUCK_RELEASE <= 0 else (t - e) / MIX_DUCK_RELEASE
            if k >= 1.0:
                continue
            v = MIX_DUCK_GAIN + (1.0 - MIX_DUCK_GAIN) * k
        g = min(g, v)
    return max(0.0, min(1.0, g))


def bed_duck_points(edit, length: float, delay: float = 0.0) -> list[list[float]]:
    """The auto-duck as a breakpoint curve on the BED's clock.

    `delay` is the film second the bed starts at, so the windows — which are
    in film seconds — land on the bed's own zero. Knots go exactly where the
    function bends: each window's start, the end of its attack, its end, and
    the end of its release. Because `audible_strips` has already merged
    anything closer than one release, no release ramp ever meets the next
    attack, so linear interpolation between these knots is the function
    EXACTLY rather than an approximation of it.
    """
    n = max(0.0, _f(length))
    if n <= 0:
        return []
    d = _f(delay)
    wins = [[s - d, e - d] for s, e in audible_strips(edit)]
    wins = [[s, e] for s, e in wins if e > 0 and s < n]
    if not wins:
        return []
    marks = {0.0, round(n, 6)}
    for s, e in wins:
        for m in (s, s + MIX_DUCK_ATTACK, e, e + MIX_DUCK_RELEASE):
            if -1e-9 <= m <= n + 1e-9:
                marks.add(round(max(0.0, min(n, m)), 6))
    out = [[t, round(_duck_gain_at(wins, t), 6)] for t in sorted(marks)]
    if all(abs(g - 1.0) < 1e-9 for _, g in out):
        return []
    return out


def bed_duck_suppressed(edit) -> bool:
    """True when the duck is switched on but an authored envelope outranks it.

    THE PRECEDENCE RULE, and the reason it is a rule rather than a product:
    two curves multiplied together is exactly the invisible-second-author
    defect this model exists to remove, one layer up. A person who has drawn
    the bed's level has said what the bed does; an automatic curve that then
    moved it would be the renderer disagreeing with them again.
    """
    aud = (edit or {}).get("audio")
    if not isinstance(aud, dict):
        return False
    if not audio_mix(aud)["duck"]:
        return False
    return bool(audio_gain_points(aud, bed_length(aud, edit_duration(edit))))


def bed_gain_points(edit) -> list[list[float]]:
    """THE ONE BED CURVE: `[[t, gain], ...]` on the bed's own clock.

    Preview, render and export read this and nothing else, so there is no
    second place a gain can come from. Three terms, and the order they compose
    in is the whole of the precedence:

    1. `bed_gain` — a STATIC fader. Always applied. A scalar is not a curve;
       it is the number under the track head, and multiplying an envelope by
       a fader is what a fader IS in every mixer ever built.
    2. The AUTHORED envelope (`audio.afx`) — always applied when it exists.
    3. The AUTO-DUCK — applied ONLY when there is no authored envelope. See
       `bed_duck_suppressed`: never two curves at once, ever.

    An empty list means unity, and unity means NO FILTER — the same rule
    `audio_gain_points` follows, so a timeline nobody has mixed builds the
    identical ffmpeg graph it always did.
    """
    aud = (edit or {}).get("audio")
    if not isinstance(aud, dict) or not aud.get("path"):
        return []
    # THE FILM IS THE CLOCK WHEN THE TRACK WILL NOT SAY. See `bed_length`: a
    # bed of unknown length plays under the film, ducked as the document says,
    # rather than falling through an empty curve to full level.
    n = bed_length(aud, edit_duration(edit))
    if n <= 0:
        return []
    mix = audio_mix(aud)
    curve = audio_gain_points(aud, n)
    if not curve and mix["duck"]:
        curve = bed_duck_points(edit, n, music_window(aud)["delay"])
    g0 = mix["bed_gain"]
    if not curve:
        if abs(g0 - 1.0) < 1e-9:
            return []
        return [[0.0, round(g0, 6)], [round(n, 6), round(g0, 6)]]
    return [[t, round(max(0.0, min(1.0, g * g0)), 6)] for t, g in curve]


def bed_gain_at(edit, t: float) -> float:
    """The bed's gain at one second of its own clock. The preview's answer."""
    curve = bed_gain_points(edit)
    if not curve:
        return 1.0
    return round(_lerp_gain(curve, max(0.0, _f(t))), 6)


def bed_render_gain(edit) -> list[list[float]]:
    """The same curve on the FILM's clock — what the render's `volume` sees.

    The bed lane's chain is `atrim, asetpts, adelay, aresample, apad, atrim,
    asetpts`, and after that `t` is FILM seconds from zero. The bed's own zero
    is film second `delay`, so the one conversion between the two clocks lives
    here rather than in the renderer, where it would be a second opinion about
    what the curve means.
    """
    curve = bed_gain_points(edit)
    if not curve:
        return []
    d = music_window((edit or {}).get("audio") or {})["delay"]
    return [[round(t + d, 6), g] for t, g in curve]


def clip_audio_drift(clip) -> float:
    """How far an UNLINKED sound has drifted from the picture it came from.

    Both halves map film time to source time with a single constant — the
    clip's `film_start - start`, the strip's `audio.film_start - audio.start` —
    so the difference between the two constants is the number an NLE prints on
    its sync flag. POSITIVE means the sound plays LATE against the frame it was
    recorded with; zero means the strip is exactly where the picture would have
    played it. A linked clip cannot drift, by construction.

    DRIFT IS NOT AN ERROR. A J-cut IS a deliberate drift and the document is
    valid either way — this is the same kind of answer `edit_gaps` gives, so
    the panel can show the offset and offer to close it.
    """
    w = clip_audio(clip)
    if not w["split"]:
        return 0.0
    c = clip if isinstance(clip, dict) else {}
    # AT THE CLIP'S SPEED: a source second is `1/speed` film seconds on both
    # halves, so the two constants are compared on the film's clock.
    sp = w["speed"]
    return round((w["film_start"] - w["start"] / sp)
                 - (_f(c.get("film_start")) - _f(c.get("start")) / sp), 6)


def clip_audio_resync(clip) -> float:
    """The film second an unlinked strip has to start at to be in sync again.

    The strip keeps its own IN-POINT — re-matching is not un-trimming — so the
    answer is "where does that source second play now that the picture has
    moved", which is the picture's own mapping applied to `audio.start`.
    """
    w = clip_audio(clip)
    c = clip if isinstance(clip, dict) else {}
    return round(_f(c.get("film_start"))
                 + (w["start"] - _f(c.get("start"))) / w["speed"], 6)


# ===========================================================================
# PART A — proxies
# ===========================================================================
def proxy_dir(board_dir) -> Path:
    """`<board>/proxy/`. Not created here — `plan_proxies` does that."""
    return Path(str(board_dir)) / "proxy"


def proxy_fingerprint(src) -> str:
    """Content address for one source clip: path + mtime + size + recipe.

    NOT a hash of the file's bytes. Hashing 150 clips of 30 MB each to decide
    whether to rebuild a 7 MB proxy costs more than rebuilding all of them, and
    the panel's own outputs are write-once — a clip whose mtime and size are
    unchanged is the same clip. The recipe version is in the hash so changing
    the proxy settings invalidates every proxy exactly once, by construction.

    A missing file gets a fingerprint too (mtime and size read as 0), so the
    caller can plan the work and let ffmpeg produce the honest error.
    """
    p = Path(str(src))
    try:
        st = p.stat()
        mtime_ns, size = st.st_mtime_ns, st.st_size
    except OSError:
        mtime_ns, size = 0, 0
    raw = f"{p.resolve() if p.exists() else p}\0{mtime_ns}\0{size}\0{PROXY_RECIPE_VERSION}"
    return hashlib.sha1(raw.encode("utf-8", "replace")).hexdigest()[:12]


def proxy_name(src) -> str:
    """`<slug-of-the-source-name>_<fingerprint>.mp4`.

    The slug is decoration for a human reading the folder; the fingerprint is
    the identity. Two different sources that happen to share a basename get
    different files, and the same source rebuilt at the same mtime gets the
    same name — which is what makes reuse a `Path.is_file()` check rather than
    a manifest that can go stale.
    """
    p = Path(str(src))
    return f"{_slug(p.stem)}_{proxy_fingerprint(p)}.mp4"


def proxy_cmd(ffmpeg, src, dest, *, width: int = PROXY_WIDTH) -> list[str]:
    """The argv for ONE proxy. All-intra, small, picture only.

    `-g 1` (with `keyint_min=1` and `sc_threshold=0`, because libx264 will
    otherwise still make its own decisions) is the entire point: every frame is
    an I-frame, so a seek decodes exactly one frame instead of the whole clip.
    That is what turns a 235 ms scrub into a 3.5 ms one.

    `-bf 0` removes B-frames so display order and decode order are the same —
    a browser seeking backwards through B-pyramids re-reads more than it needs.

    AUDIO IS CARRIED (recipe v2). It was dropped on the reasoning that "the
    timeline's sound is the soundtrack" — true for a music video, wrong for
    everything else. On a dialogue film the clips ARE the performance, so a
    silent preview cannot tell you whether a cut lands on the line or halfway
    through the word. Reported by the owner cutting a dialogue short: "when you
    play the videos, they don't have sound."

    `0:a:0?` is optional on purpose — a clip with no audio track must still
    produce a proxy rather than failing the whole prepare. AAC at 96k is
    intelligible and costs a few hundred KB across a film.
    """
    return [
        str(ffmpeg), "-y", "-v", "error", "-nostdin",
        "-i", str(src),
        "-map", "0:v:0", "-map", "0:a:0?", "-sn", "-dn",
        "-c:a", "aac", "-b:a", PROXY_ABITRATE, "-ac", "2",
        # -2 keeps the height even (H.264 4:2:0 has no odd-dimension form) and
        # preserves the source aspect: a proxy that is a different SHAPE from
        # its source would put the playhead over the wrong part of the picture.
        #
        # min(width, iw) NEVER UPSCALES. Engine output is commonly 1024 wide, so
        # a flat `scale=1280` would spend bytes and encode time inventing detail
        # that is not in the source, and the preview would look no better for it.
        # Downscale-only: the proxy is either the source size or smaller.
        "-vf", f"scale='min({int(width)},iw)':-2:flags=bicubic",
        "-c:v", "libx264",
        "-g", "1", "-keyint_min", "1", "-sc_threshold", "0", "-bf", "0",
        "-crf", PROXY_CRF, "-preset", PROXY_PRESET,
        "-pix_fmt", PROXY_PIX_FMT,
        "-movflags", "+faststart",
        str(dest),
    ]


def plan_proxies(clips, board_dir, *, width: int = PROXY_WIDTH) -> dict:
    """Decide what to build, what to reuse, and what is now junk.

    Returns::

        {"dir": Path, "width": int,
         "build": [{"path", "proxy", "name"}],   # in the order given
         "reuse": [{"path", "proxy", "name"}],
         "stale": [Path]}                        # files to delete

    Deduplicated by fingerprint: a board that uses one shot three times builds
    ONE proxy. `stale` is every mp4 in the proxy directory that no live clip
    points at — which is exactly the set a re-render or a deleted shot leaves
    behind, and the only reason the folder does not grow forever.
    """
    d = proxy_dir(board_dir)
    build: list[dict] = []
    reuse: list[dict] = []
    wanted: set[str] = set()
    for c in clips:
        # A SLUG IS NOT A FILE and a still is not a seek. Neither has anything
        # an all-intra proxy could speed up — the slug has no source at all and
        # the still is one frame a browser paints from an <img> — so building
        # one would spend ffmpeg on a video nothing will ever play, and,
        # worse, put a name in `wanted` that a later prune would have to keep.
        if isinstance(c, dict) and clip_kind(c) != "video":
            continue
        raw = c.get("path") if isinstance(c, dict) else c
        if not raw:
            continue
        src = Path(str(raw))
        name = proxy_name(src)
        if name in wanted:
            continue
        wanted.add(name)
        row = {"path": str(src), "name": name, "proxy": str(d / name)}
        (reuse if (d / name).is_file() else build).append(row)
    stale = []
    if d.is_dir():
        for f in sorted(d.iterdir()):
            if f.is_file() and f.suffix == ".mp4" and f.name not in wanted:
                stale.append(f)
    return {"dir": d, "width": int(width), "build": build,
            "reuse": reuse, "stale": stale}


def prune_proxies(stale) -> int:
    """Delete the files `plan_proxies` called junk. Never raises."""
    n = 0
    for f in stale or []:
        try:
            Path(str(f)).unlink()
            n += 1
        except OSError:
            continue
    return n


# ===========================================================================
# PART B — waveform peaks
# ===========================================================================
def peaks_path(board_dir) -> Path:
    return Path(str(board_dir)) / "peaks.json"


def compute_peaks(audio_path, *,
                  buckets_per_second: int = PEAKS_BUCKETS_PER_SECOND,
                  sr: int = PEAKS_SR, decoder=None) -> dict:
    """Min/max pairs for a soundtrack, at `buckets_per_second` a second.

    Returns::

        {"version", "path", "duration", "sample_rate", "buckets_per_second",
         "count", "scale", "peaks": [min0, max0, min1, max1, ...]}

    Interleaved rather than two arrays because it halves the JSON punctuation,
    and integer rather than float because three characters per value instead of
    eighteen is the difference between 36 KB and 400 KB for the same picture. A
    waveform is drawn at one pixel per bucket; nothing downstream can see more
    precision than `scale` provides.

    `decoder` exists for tests and for a caller that already has the PCM;
    by default this uses `storyboard_edit._decode_pcm`, the same one ffmpeg
    call the beat tracker makes, so a board never decodes its track twice with
    two different resamplers.
    """
    if decoder is None:
        import storyboard_edit as sedit                              # noqa: PLC0415
        decoder = lambda p, rate: sedit._decode_pcm(p, rate)         # noqa: E731
    x = np.asarray(decoder(str(audio_path), sr), dtype=np.float32)
    if x.size == 0:
        raise EditError(f"{Path(str(audio_path)).name} carries no audio")
    bps = max(1, int(buckets_per_second))
    per = max(1, int(round(sr / float(bps))))
    n = int(x.size // per)
    if n < 1:
        n, per = 1, int(x.size)
    trimmed = x[:n * per].reshape(n, per)
    lo = np.clip(trimmed.min(axis=1) * PEAKS_SCALE, -PEAKS_SCALE, PEAKS_SCALE)
    hi = np.clip(trimmed.max(axis=1) * PEAKS_SCALE, -PEAKS_SCALE, PEAKS_SCALE)
    inter = np.empty(n * 2, dtype=np.int16)
    inter[0::2] = np.rint(lo).astype(np.int16)
    inter[1::2] = np.rint(hi).astype(np.int16)
    return {
        "version": PEAKS_VERSION,
        "path": str(audio_path),
        "duration": round(x.size / float(sr), 6),
        "sample_rate": int(sr),
        "buckets_per_second": round(sr / float(per), 6),
        "count": int(n),
        "scale": PEAKS_SCALE,
        "peaks": [int(v) for v in inter],
    }


def save_peaks(board_dir, peaks: dict) -> Path:
    # COMPACT, not pretty. `indent=2` puts every one of ~96,000 integers on its
    # own line and turns a 345 KB file into 795 KB — measured, on the real
    # 479 s track. Nobody reads a waveform by hand; the edit next door is the
    # document, this is data.
    return _atomic_json(peaks_path(board_dir), peaks, prefix=".peaks-",
                        indent=None)


def _same_file(a, b) -> bool:
    """Do these two strings name the same file? RESOLVED, not compared.

    `mlx_outputs/` is a symlink into Pinokio's shared drive, so the very same
    soundtrack is spelled two ways — `…/phosphene-dev.git/mlx_outputs/x.wav`
    and `…/drive/drives/peers/…/mlx_outputs/x.wav` — depending on which side
    wrote it down. A raw string compare called those different files and would
    have blanked the waveform on boards where nothing had changed at all: the
    invalidation fix, arriving as a worse bug than the one it fixed.
    """
    sa, sb = str(a or "").strip(), str(b or "").strip()
    if not sa or not sb:
        return sa == sb
    if sa == sb:
        return True
    try:
        return Path(sa).resolve() == Path(sb).resolve()
    except OSError:
        return False


def load_peaks(board_dir, *, path=None) -> dict | None:
    """The cached waveform, or None — INCLUDING when it is about another file.

    `peaks.json` records the track it was computed from. After the owner
    swapped the soundtrack the strip went on reading "44.99s" under a
    different file, because nothing compared the two: a cache that outlives
    its subject is not a cache, it is a wrong answer with a timestamp. A
    caller that knows which track is on the timeline passes it, and a
    disagreement reads as "no waveform yet" — which is true, and which the
    prepare step already knows how to fix.
    """
    p = peaks_path(board_dir)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    want = str(path or "").strip()
    if want and not _same_file(data.get("path"), want):
        return None
    return data


# ===========================================================================
# PART C — edit.json
# ===========================================================================
# ---------------------------------------------------------------------------
# PER-SOURCE WAVEFORMS — the same machinery the bed uses, one file per take
# ---------------------------------------------------------------------------
# A strip with no waveform is a rectangle you have to play to understand. The
# bed has had one since the lane existed; the clips' own sound — the thing you
# are actually cutting on a dialogue film — did not.
#
# ONE FILE PER SOURCE, NOT PER CLIP. The same take used twice draws the same
# waveform, and the strip decides which SLICE of it to show from its own source
# window — so trimming a strip re-slices rather than recomputing, and a J-cut
# that slides its sound shows the seconds it actually plays instead of the
# picture's.
#
# The name carries `proxy_fingerprint`, which is already path + mtime + size:
# a re-rendered take gets a new name and the stale one is simply never asked
# for again, so there is no invalidation to get wrong.
def clip_peaks_dir(board_dir) -> Path:
    return Path(str(board_dir)) / "peaks"


def clip_peaks_path(board_dir, src) -> Path:
    p = Path(str(src))
    return clip_peaks_dir(board_dir) / f"{_slug(p.stem, 32)}_{proxy_fingerprint(p)}.json"


def clip_peaks(board_dir, src, *, decoder=None,
               buckets_per_second: int = PEAKS_BUCKETS_PER_SECOND) -> dict:
    """The waveform for ONE source, cached beside the board.

    Raises `EditError` when the file has no audio, which is a fact about the
    take and not a failure: the lane already draws that state.
    """
    dst = clip_peaks_path(board_dir, src)
    got = _read_json(dst) if dst.is_file() else None
    if isinstance(got, dict) and got.get("peaks") is not None:
        return got
    data = compute_peaks(src, buckets_per_second=buckets_per_second,
                         decoder=decoder)
    dst.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(dst, data, prefix=".peaks-")
    return data


def prune_clip_peaks(board_dir, keep_paths) -> int:
    """Drop waveforms for takes the timeline no longer uses."""
    d = clip_peaks_dir(board_dir)
    if not d.is_dir():
        return 0
    wanted = {clip_peaks_path(board_dir, p).name for p in (keep_paths or [])}
    gone = 0
    for f in d.glob("*.json"):
        if f.name in wanted:
            continue
        try:
            f.unlink()
            gone += 1
        except OSError:
            continue
    return gone


def edit_path(board_dir) -> Path:
    return Path(str(board_dir)) / "edit.json"


def _atomic_json(target: Path, payload, *, prefix: str,
                 indent: int | None = 2) -> Path:
    """Same write `save_storyboard` uses, for the same reason.

    A torn `edit.json` is an hour of somebody's arrangement gone. Temp file in
    the SAME directory (os.replace is only atomic within a filesystem), fsync,
    replace.
    """
    target = Path(str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=prefix,
                               suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=indent, ensure_ascii=False,
                      separators=(",", ":") if indent is None else None)
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


def new_clip(path, start: float, end: float, film_start: float, *,
             source: str = "auto", **extra) -> dict:
    """One timeline clip, with the film slot derived rather than passed twice.

    `film_end` is never an independent number: a clip plays at 1x, so its slot
    is exactly as long as its window. Deriving it here is what stops the two
    from drifting apart in the one place they are both first written.
    """
    start, end = round(_f(start), 6), round(_f(end), 6)
    fs = round(_f(film_start), 6)
    clip = {
        "id": extra.pop("id", "") or _clip_id(path, start, fs),
        "path": str(path),
        "proxy": extra.pop("proxy", None),
        "start": start,
        "end": end,
        "film_start": fs,
        "film_end": round(fs + (end - start), 6),
        "source": source if source in ("auto", "human") else "auto",
        "locked": bool(extra.pop("locked", False)),
    }
    clip.update(extra)
    return clip


def _clip_id(path, start: float, film_start: float) -> str:
    raw = f"{path}\0{start:.6f}\0{film_start:.6f}\0{time.time_ns()}"
    return "k" + hashlib.sha1(raw.encode("utf-8", "replace")).hexdigest()[:10]


def edit_from_plan(plan, *, board_id: str = "", audio: dict | None = None,
                   beats: dict | None = None, proxies: dict | None = None,
                   settings: dict | None = None,
                   labels: dict | None = None) -> dict:
    """Derive a fresh `edit.json` from a `storyboard_edit.plan_cut()` plan.

    Every field the planner already computed is carried through unchanged —
    `start`/`end`/`film_start`/`film_end` are the plan's own numbers, not a
    re-derivation — and everything the planner explains (the snap, the window
    score, the notes) rides along as `analysis` so the UI can show WHY a cut
    landed where it did without asking the server a second question.

    `source` is `"auto"` on every clip, and that is a promise the save path
    keeps: a clip a human has touched comes back as `"human"` and a later
    re-plan can leave it alone.
    """
    proxies = proxies or {}
    labels = labels or {}
    clips = []
    for entry in plan or []:
        if not isinstance(entry, dict) or not entry.get("path"):
            continue
        path = str(entry["path"])
        # Only a KNOWN answer is carried. An absent `has_audio` means nobody
        # probed, which is not the same as silence — the lane draws such a
        # clip exactly as it always did.
        extra = ({"has_audio": bool(entry["has_audio"])}
                 if isinstance(entry.get("has_audio"), bool) else {})
        clips.append(new_clip(
            path,
            _f(entry.get("start")), _f(entry.get("end")),
            _f(entry.get("film_start")),
            source="auto",
            proxy=proxies.get(path),
            n=entry.get("n"),
            title=labels.get(path) or Path(path).stem,
            duration=round(_f((entry.get("window") or {}).get("source_duration")), 6) or None,
            analysis={
                "score": (entry.get("window") or {}).get("score"),
                "reason": (entry.get("window") or {}).get("reason"),
                "usable": (entry.get("window") or {}).get("usable", True),
                "snap": entry.get("snap"),
                "notes": entry.get("notes") or [],
            },
            **extra,
        ))
    return normalise_edit({
        "version": EDIT_VERSION,
        "board_id": board_id,
        "revision": 0,
        "source": "auto",
        "audio": audio,
        "beats": _slim_beats(beats),
        "clips": clips,
        "settings": settings or {},
    })


def _slim_beats(beats: dict | None) -> dict | None:
    """The beat grid a timeline needs, without the megabyte of diagnostics.

    `beat_map()` returns per-second metrics and runner-up hypotheses because a
    human recalibrating the tracker needs them. A timeline needs the grid and
    the confidence, and shipping the rest on every poll is 40x the payload for
    a number nobody draws.
    """
    if not isinstance(beats, dict):
        return None
    d = beats.get("diagnostics") or {}
    return {
        "bpm": beats.get("bpm"),
        "period": beats.get("period"),
        "phase": beats.get("phase"),
        "meter": beats.get("meter"),
        "confidence": beats.get("confidence"),
        "span": beats.get("span"),
        "beats": list(beats.get("beats") or []),
        "downbeats": list(beats.get("downbeats") or []),
        "grid_lock_ms": d.get("grid_lock_ms"),
        "tempo_drift_bpm": d.get("tempo_drift_bpm"),
    }


# ---------------------------------------------------------------------------
# TRANSITIONS — a typed object that OWNS A BOUNDARY, never an overlap
# ---------------------------------------------------------------------------
# WHY THIS IS NOT "LET THE PICTURES OVERLAP". A cross-dissolve IS two clips in
# the same second, and the validator's one-picture-at-a-time rule
# (`clips_overlap`) is load-bearing: the autosave, the crash backup, the
# assembler and every gap/overlap check downstream rest on it, and
# `WARNING_CODES` deliberately holds ONLY `clips_audio_overlap` because the
# last time an ordinary edit was made unsaveable it cost an afternoon. So the
# picture lane stays single-track and a transition is a SEPARATE list, the
# same shape discipline as `overlays`:
#
#     "transitions": [{"id": "t1", "after_clip": "<clip id>",
#                      "kind": "dissolve" | "fade_black", "duration": 0.5}]
#
# `after_clip` names the OUTGOING clip; the transition sits on the boundary
# between it and its successor in film order. One per boundary. The clips'
# own film_start/film_end DO NOT MOVE — the render gets its overlap from
# SOURCE HANDLES: half the duration of extra tail from beyond the outgoing
# clip's out-point and half of extra head from before the incoming clip's
# in-point, material that already exists past the trims. That is how every
# NLE builds a centred dissolve, it keeps the film exactly as long as the
# timeline says, and it means the sound needs nothing new at all — the audio
# plan never sees the extension.
#
# A side with no spare material is REFUSED with a sentence naming the side and
# how much it is short. A still and a slug have no source clock, so they have
# all the handles anybody could ask for.
TRANSITION_KINDS = ("dissolve", "fade_black")
TRANSITION_MIN = 1.0 / 24.0      # under one frame there is nothing to draw
TRANSITION_MAX = 2.0             # longer than this eats the shot


def transition_items(edit) -> list[dict]:
    """The raw rows, dicts only, in document order. Absent is an empty list."""
    rows = (edit or {}).get("transitions") if isinstance(edit, dict) else None
    if not isinstance(rows, list):
        return []
    return [t for t in rows if isinstance(t, dict)]


def transition_duration(row, out_len: float, in_len: float,
                        fps: float = 24.0) -> float:
    """The duration a boundary can actually carry: min(asked, half the
    shorter neighbour, TRANSITION_MAX), on an EVEN number of frames. The
    document keeps the number the user typed; every reader clamps through
    here.

    EVEN FRAMES, because the transition is split in half across the cut and
    each half is extra picture on one side: 0.8 s at 24 fps is 19.2 frames,
    which `xfade` rounds up on both runs and the film comes out one frame
    longer than its sound. 20 frames — ten a side — is exact.
    """
    d = _f((row or {}).get("duration"))
    if d <= 0:
        return 0.0
    d = max(0.0, min(d, TRANSITION_MAX, 0.5 * max(0.0, min(out_len, in_len))))
    f = max(1.0, _f(fps, 24.0))
    frames = 2 * int(round(d * f / 2.0))
    return round(frames / f, 6)


def _spare_source(clip, side: str) -> float | None:
    """Seconds of untrimmed source beyond the out-point (`tail`) or before
    the in-point (`head`). None when the source length is not known."""
    if clip_kind(clip) != "video":
        return float("inf")
    if side == "head":
        return max(0.0, _f(clip.get("start")))
    dur = clip.get("duration")
    if not isinstance(dur, (int, float)) or isinstance(dur, bool) or dur <= 0:
        return None
    return max(0.0, float(dur) - _f(clip.get("end")))


def resolve_transitions(edit) -> list[dict]:
    """Every transition row, resolved against the timeline it sits on.

    Each entry: `{id, after_clip, before_clip, kind, duration, half,
    out_index, in_index, at, problem}` where `problem` is None or a
    `{code, message}` a validator can report verbatim. `duration` is the
    CLAMPED one — the number the render will use — and `at` is the film
    second of the boundary. Indices are into the document's `clips` list.
    """
    clips = [c for c in ((edit or {}).get("clips") or []) if isinstance(c, dict)]
    order = sorted(range(len(clips)),
                   key=lambda i: (_f(clips[i].get("film_start")),
                                  str(clips[i].get("path"))))
    pos = {i: k for k, i in enumerate(order)}
    by_id = {}
    for i, c in enumerate(clips):
        cid = str(c.get("id") or "")
        if cid and cid not in by_id:
            by_id[cid] = i
    seen: set = set()
    out: list[dict] = []
    for n, row in enumerate(transition_items(edit)):
        label = f"transition {n + 1}"
        res = {"id": str(row.get("id") or ""), "after_clip": str(row.get("after_clip") or ""),
               "before_clip": "", "kind": str(row.get("kind") or "").strip().lower(),
               "duration": 0.0, "half": 0.0, "out_index": None, "in_index": None,
               "at": 0.0, "problem": None}

        def fail(code, message):
            res["problem"] = {"code": code, "message": message}
            out.append(res)

        aid = res["after_clip"]
        if not aid or aid not in by_id:
            fail("transition_unknown_clip",
                 f"{label}: after_clip {aid!r} names no clip on this timeline")
            continue
        oi = by_id[aid]
        res["out_index"] = oi
        k = pos[oi]
        if k + 1 >= len(order):
            fail("transition_last_clip",
                 f"{label}: clip {oi + 1} is the last clip — there is nothing "
                 f"after it to dissolve into")
            continue
        ii = order[k + 1]
        res["in_index"] = ii
        res["before_clip"] = str(clips[ii].get("id") or "")
        res["at"] = round(_f(clips[oi].get("film_end")), 6)
        if aid in seen:
            fail("transition_duplicate_boundary",
                 f"{label}: the cut after clip {oi + 1} already has a "
                 f"transition — one per boundary")
            continue
        seen.add(aid)
        if res["kind"] not in TRANSITION_KINDS:
            fail("transition_kind",
                 f"{label}: kind must be one of {', '.join(TRANSITION_KINDS)} "
                 f"(got {row.get('kind')!r})")
            continue
        raw_d = row.get("duration")
        if not isinstance(raw_d, (int, float)) or isinstance(raw_d, bool) \
                or raw_d != raw_d or raw_d <= 0:
            fail("transition_duration",
                 f"{label}: duration must be a number of seconds above 0")
            continue
        d = transition_duration(row, clip_length(clips[oi]), clip_length(clips[ii]),
                                fps=film_fps(edit))
        if d < TRANSITION_MIN - 1e-9:
            fail("transition_duration",
                 f"{label}: clips {oi + 1} and {ii + 1} are too short to "
                 f"carry a transition between them")
            continue
        half = round(d / 2.0, 6)
        res["duration"], res["half"] = d, half
        short = []
        # THE HANDLES ARE SOURCE SECONDS and the transition is film seconds:
        # a retimed clip needs `speed` times as much take for the same half.
        need_out = half * clip_speed(clips[oi])
        need_in = half * clip_speed(clips[ii])
        word = res["kind"].replace("_", " ")
        tail = _spare_source(clips[oi], "tail")
        if tail is None:
            short.append(f"the source length of clip {oi + 1} is not known — "
                         f"run Prepare so the panel can measure it")
        elif tail + 1e-6 < need_out:
            short.append(f"clip {oi + 1} has only {tail:.2f}s beyond its "
                         f"out-point and the {word} needs {need_out:.2f}s "
                         f"there — trim its tail in or shorten the transition")
        head = _spare_source(clips[ii], "head")
        if head is not None and head + 1e-6 < need_in:
            short.append(f"clip {ii + 1} has only {head:.2f}s before its "
                         f"in-point and the {word} needs {need_in:.2f}s "
                         f"there — trim its head in or shorten the transition")
        if short:
            fail("transition_no_handles", f"{label}: " + "; ".join(short))
            continue
        out.append(res)
    return out


def transition_problems(edit) -> list[dict]:
    """Only the transitions that cannot be rendered, with their sentences."""
    return [t["problem"] for t in resolve_transitions(edit) if t.get("problem")]


def validate_edit(edit) -> list[dict]:
    """Every reason this edit must not be written. Empty list == good.

    Each entry is `{"code", "where", "message"}` — a stable machine name, the
    clip index (or None for the document), and a sentence a person can act on.
    Structured for the same reason `validate_storyboard_detail` is: a UI that
    has to regex an error string is a UI that breaks when the string improves.

    Only genuine corruption is an ERROR. A gap in the timeline is not corrupt —
    it is the hole somebody is about to generate a shot into, and refusing to
    save it would make the feature impossible. Gaps come back from
    `edit_gaps()` as information; overlaps of the PICTURE are errors, because a
    single-track concat has no way to play two clips at once and the render
    would silently pick one.

    OVERLAPPING SOUND IS A WARNING, AND THAT COST A MAN HIS AFTERNOON.
    `clips_audio_overlap` was an error, and every writer of this document —
    Save, the autosave and the crash backup alike — refuses on any error. So
    the moment a J-cut pulled one line a quarter of a second under the outgoing
    shot, which is the most ordinary edit this feature exists to make, the
    board became unsaveable AND unbackupable: a red "SAVING IS FAILING" banner
    over work that could not be stored anywhere, while the user kept cutting.

    Two sounds meeting for 0.25 s is a crossfade in every NLE ever shipped, and
    the assembler has always resolved it deterministically — the incoming
    sound wins its start and the outgoing tail gives way (`_sb_split_audio_plan`
    in mlx_ltx_panel.py). So there is no corruption here to protect anyone
    from, only a note worth showing. `WARNING_CODES` names the codes that must
    never block a write; `blocking_errors()` is what a writer asks.

    PERSISTING THE USER'S WORK WINS. If a rule cannot decide between "refuse
    the save" and "let it through with a note", it says note.
    """
    errs: list[dict] = []

    def bad(code: str, message: str, where=None) -> None:
        e = {"code": code, "where": where, "message": message}
        if code in WARNING_CODES:
            e["severity"] = "warning"
        errs.append(e)

    if not isinstance(edit, dict):
        bad("not_an_object", "the edit is not a JSON object")
        return errs
    if edit.get("version") != EDIT_VERSION:
        bad("version", f"edit version {edit.get('version')!r} — this build "
                       f"understands {EDIT_VERSION}")
    clips = edit.get("clips")
    if not isinstance(clips, list):
        bad("clips_not_a_list", "clips must be a list")
        return errs

    audio = edit.get("audio")
    if audio is not None and not isinstance(audio, dict):
        bad("audio_shape", "audio must be an object or null")
    elif isinstance(audio, dict) and audio.get("path") is not None \
            and not isinstance(audio.get("path"), str):
        bad("audio_path", "audio.path must be a string")
    if isinstance(audio, dict) and audio.get("mode") is not None \
            and audio.get("mode") not in ("under", "replace"):
        # `replace` deletes every line of dialogue in the film, so a typo here
        # must not fall through to it by accident.
        bad("audio_mode", "audio.mode must be 'under' or 'replace'")
    if isinstance(audio, dict):
        # THE MIX IS OPTIONAL, the same migration the trims are: absent means
        # the new default, so every edit.json ever written is already valid.
        mix = audio.get("mix")
        if mix is not None and not isinstance(mix, dict):
            bad("audio_mix", "audio.mix must be an object or absent")
        elif isinstance(mix, dict):
            bg = mix.get("bed_gain")
            if bg is not None:
                if not isinstance(bg, (int, float)) or isinstance(bg, bool) \
                        or bg != bg or bg in (float("inf"), float("-inf")):
                    bad("audio_mix_bed_gain",
                        "audio.mix.bed_gain must be a number or absent")
                elif not 0.0 <= float(bg) <= 1.0:
                    bad("audio_mix_bed_gain_range",
                        "audio.mix.bed_gain must be between 0 and 1")
            dk = mix.get("duck")
            if dk is not None and not isinstance(dk, bool):
                bad("audio_mix_duck",
                    "audio.mix.duck must be true, false or absent")
        # THE TRIMS ARE OPTIONAL AND THAT IS THE MIGRATION. Absent means
        # untrimmed, so every edit.json ever written is already valid here and
        # EDIT_VERSION does not move for them.
        for key in ("trim_start", "trim_end"):
            v = audio.get(key)
            if v is None:
                continue
            if not isinstance(v, (int, float)) or isinstance(v, bool) \
                    or v != v or v in (float("inf"), float("-inf")):
                bad(f"audio_{key}", f"audio.{key} must be a number or absent")
            elif v < 0:
                bad(f"audio_{key}_range", f"audio.{key} must be >= 0")
        ts, te = audio.get("trim_start"), audio.get("trim_end")
        if isinstance(ts, (int, float)) and not isinstance(ts, bool) \
                and isinstance(te, (int, float)) and not isinstance(te, bool) \
                and float(te) <= float(ts):
            bad("audio_trim_window",
                f"audio.trim_end ({te}) must be after audio.trim_start ({ts})")
        off = audio.get("offset")
        if off is not None and (not isinstance(off, (int, float))
                                or isinstance(off, bool) or off != off
                                or off in (float("inf"), float("-inf"))):
            # NEGATIVE IS LEGAL — it is the music starting later than the film.
            # Only a non-number is refused.
            bad("audio_offset", "audio.offset must be a number")
        # THE BED'S OWN ENVELOPE, held to the same shape a strip's is. It was
        # readable by the model and normalised on the way out from the day the
        # envelope maths was shared, and validated by nothing — so a hand-
        # edited or agent-written `audio.afx` of the wrong shape reached
        # `bed_gain_points` instead of reaching a sentence.
        bafx = audio.get("afx")
        if bafx is not None and not isinstance(bafx, dict):
            bad("audio_afx", "audio.afx must be an object or absent")
        elif isinstance(bafx, dict):
            for key in ("fade_in", "fade_out"):
                v = bafx.get(key)
                if v is None:
                    continue
                if not isinstance(v, (int, float)) or isinstance(v, bool) \
                        or v != v or v in (float("inf"), float("-inf")):
                    bad(f"audio_afx_{key}",
                        f"audio.afx.{key} must be a number")
                elif v < 0:
                    bad(f"audio_afx_{key}_range",
                        f"audio.afx.{key} must be >= 0")
            bpts = bafx.get("points")
            if bpts is not None and not isinstance(bpts, list):
                bad("audio_afx_points",
                    "audio.afx.points must be a list or absent")
            elif isinstance(bpts, list):
                for row in bpts:
                    if not isinstance(row, (list, tuple)) or len(row) != 2 \
                            or any(not isinstance(x, (int, float))
                                   or isinstance(x, bool) or x != x
                                   for x in row):
                        bad("audio_afx_point",
                            "every audio.afx point is [seconds, gain]")
                        break

    spans: list[tuple[float, float, int]] = []
    for i, c in enumerate(clips):
        if not isinstance(c, dict):
            bad("clip_not_an_object", f"clip {i + 1} is not an object", i)
            continue
        if c.get("kind") is not None and str(c.get("kind")) not in CLIP_KINDS:
            bad("clip_kind", f"clip {i + 1}: kind must be one of "
                             f"{', '.join(CLIP_KINDS)} (got "
                             f"{c.get('kind')!r})", i)
            continue
        kind = clip_kind(c)
        path = c.get("path")
        # A SLUG HAS NO FILE. That is the whole of what a slug is: black for a
        # duration, `color=` in the graph, nothing on disk, nothing to probe,
        # nothing to proxy. Demanding a path from one is demanding a file for
        # the absence of a file.
        if kind != "slug":
            if not isinstance(path, str) or not path.strip():
                bad("clip_path", f"clip {i + 1} has no path", i)
                continue
            # A STILL'S IMAGE MUST EXIST, and it is checked HERE rather than at
            # the ffmpeg that would otherwise discover it. A video clip whose
            # file has gone still renders — the assembler drops it and says so
            # in `unreadable` — but a still is nothing BUT its image, so a
            # missing one is a clip that can never play.
            elif kind == "still" and not Path(path).is_file():
                bad("still_missing",
                    f"clip {i + 1}: there is no image at {path}", i)
        start, end = _f(c.get("start"), -1.0), _f(c.get("end"), -1.0)
        fs, fe = _f(c.get("film_start"), -1.0), _f(c.get("film_end"), -1.0)
        if fs < 0:
            bad("clip_film_start", f"clip {i + 1}: film_start must be >= 0", i)
        if fe <= fs:
            # For a still and a slug this is the ONLY length that exists — the
            # hold is the slot, so "a still needs a duration" and "film_end is
            # after film_start" are the same sentence.
            bad("clip_film_window", f"clip {i + 1}: film_end ({fe}) must be "
                                    f"after film_start ({fs})", i)
        # THE SOURCE WINDOW IS A VIDEO'S PROBLEM ALONE. A still has no source
        # clock to be a window into and a slug has no source at all; both have
        # their start/end SYNTHESISED from the slot by `normalise_edit`, so
        # policing a window they do not own would refuse the document for a
        # number the server itself wrote.
        if kind == "video":
            if start < 0:
                bad("clip_start", f"clip {i + 1}: start must be >= 0", i)
            if end <= start:
                bad("clip_window", f"clip {i + 1}: end ({end}) must be after "
                                   f"start ({start})", i)
            sp_raw = c.get("speed")
            if sp_raw is not None:
                if not isinstance(sp_raw, (int, float)) or isinstance(sp_raw, bool) \
                        or sp_raw != sp_raw or sp_raw in (float("inf"), float("-inf")):
                    bad("clip_speed", f"clip {i + 1}: speed must be a number", i)
                elif not SPEED_MIN - 1e-9 <= float(sp_raw) <= SPEED_MAX + 1e-9:
                    bad("clip_speed_range",
                        f"clip {i + 1}: speed is {float(sp_raw):g}x — it must be "
                        f"between {SPEED_MIN:g}x and {SPEED_MAX:g}x", i)
            sp = clip_speed(c)
            if end > start and fe > fs \
                    and abs((end - start) / sp - (fe - fs)) > LENGTH_TOLERANCE:
                bad("clip_length_mismatch",
                    f"clip {i + 1}: the source window is {end - start:.3f}s "
                    f"at {sp:g}x but its slot on the film is {fe - fs:.3f}s — "
                    f"a clip plays its window at its speed and nothing else", i)
            dur = c.get("duration")
            if isinstance(dur, (int, float)) and dur > 0 \
                    and end > float(dur) + 1e-3:
                bad("clip_past_the_end",
                    f"clip {i + 1}: the window ends at {end:.3f}s but the "
                    f"source is {float(dur):.3f}s long", i)
        adj = c.get("adjust")
        if adj is not None:
            if not isinstance(adj, dict):
                bad("clip_adjust", f"clip {i + 1}: adjust must be an object "
                                   f"or absent", i)
            elif adj.get("brightness") is not None:
                b = adj.get("brightness")
                if not isinstance(b, (int, float)) or isinstance(b, bool) \
                        or b != b or b in (float("inf"), float("-inf")):
                    bad("clip_brightness",
                        f"clip {i + 1}: adjust.brightness must be a number", i)
                elif abs(float(b)) > BRIGHTNESS_LIMIT + 1e-9:
                    bad("clip_brightness_range",
                        f"clip {i + 1}: adjust.brightness is {float(b):+.3f} — "
                        f"it must be between {-BRIGHTNESS_LIMIT:+.1f} and "
                        f"{BRIGHTNESS_LIMIT:+.1f}", i)
        fr = c.get("frame")
        if fr is not None:
            if not isinstance(fr, dict):
                bad("clip_frame", f"clip {i + 1}: frame must be an object or absent", i)
            elif kind == "slug" and any(fr.get(k) is not None for k in ("zoom", "x", "y")):
                bad("clip_frame_kind", f"clip {i + 1}: black has nothing to reframe", i)
            else:
                for key, lo, hi in (("zoom", FRAME_ZOOM_MIN, FRAME_ZOOM_MAX),
                                    ("x", 0.0, 1.0), ("y", 0.0, 1.0)):
                    v = fr.get(key)
                    if v is None:
                        continue
                    if not isinstance(v, (int, float)) or isinstance(v, bool) \
                            or v != v or v in (float("inf"), float("-inf")):
                        bad("clip_frame", f"clip {i + 1}: frame.{key} must be a number", i)
                    elif not lo - 1e-9 <= float(v) <= hi + 1e-9:
                        bad("clip_frame_range",
                            f"clip {i + 1}: frame.{key} is {float(v):g} — it must be "
                            f"between {lo:g} and {hi:g}", i)
        aud = c.get("audio")
        if aud is not None:
            # A STILL AND A SLUG HAVE NO SOUND TO SPLIT. Letting them carry the
            # field would be letting a document describe an edit the assembler
            # has no input to make.
            if kind != "video":
                bad("clip_audio_kind",
                    f"clip {i + 1}: only a video clip has sound to unlink", i)
            elif not isinstance(aud, dict):
                bad("clip_audio_shape",
                    f"clip {i + 1}: audio must be an object or absent", i)
            else:
                if aud.get("linked") is not None \
                        and not isinstance(aud.get("linked"), bool):
                    bad("clip_audio_linked",
                        f"clip {i + 1}: audio.linked must be true or absent", i)
                w = clip_audio(c)
                if w["start"] < 0:
                    bad("clip_audio_start",
                        f"clip {i + 1}: audio.start must be >= 0", i)
                if w["end"] <= w["start"]:
                    bad("clip_audio_window",
                        f"clip {i + 1}: audio.end ({w['end']}) must be after "
                        f"audio.start ({w['start']})", i)
                if w["film_start"] < 0:
                    bad("clip_audio_film_start",
                        f"clip {i + 1}: audio.film_start must be >= 0", i)
                dur = c.get("duration")
                if isinstance(dur, (int, float)) and dur > 0 \
                        and w["end"] > float(dur) + 1e-3:
                    bad("clip_audio_past_the_end",
                        f"clip {i + 1}: the sound ends at {w['end']:.3f}s but "
                        f"the source is {float(dur):.3f}s long", i)
        afx = c.get("afx")
        if afx is not None and not isinstance(afx, dict):
            bad("clip_afx", f"clip {i + 1}: afx must be an object or absent", i)
        elif isinstance(afx, dict):
            for key in ("fade_in", "fade_out"):
                v = afx.get(key)
                if v is None:
                    continue
                if not isinstance(v, (int, float)) or isinstance(v, bool) \
                        or v != v or v in (float("inf"), float("-inf")):
                    bad(f"clip_a{key}",
                        f"clip {i + 1}: afx.{key} must be a number", i)
                elif float(v) < 0:
                    bad(f"clip_a{key}_range",
                        f"clip {i + 1}: afx.{key} must be >= 0", i)
            pts = afx.get("points")
            if pts is not None and not isinstance(pts, list):
                bad("clip_afx_points",
                    f"clip {i + 1}: afx.points must be a list or absent", i)
            elif isinstance(pts, list):
                for row in pts:
                    if not isinstance(row, (list, tuple)) or len(row) != 2 \
                            or not all(isinstance(x, (int, float))
                                       and not isinstance(x, bool)
                                       for x in row):
                        bad("clip_afx_point",
                            f"clip {i + 1}: every afx point is [seconds, gain]",
                            i)
                        break
        fx = c.get("fx")
        if fx is not None:
            if not isinstance(fx, dict):
                bad("clip_fx", f"clip {i + 1}: fx must be an object or absent", i)
            else:
                for key in ("fade_in", "fade_out"):
                    v = fx.get(key)
                    if v is None:
                        continue
                    if not isinstance(v, (int, float)) or isinstance(v, bool) \
                            or v != v or v in (float("inf"), float("-inf")):
                        bad(f"clip_{key}",
                            f"clip {i + 1}: fx.{key} must be a number", i)
                    elif float(v) < 0:
                        bad(f"clip_{key}_range",
                            f"clip {i + 1}: fx.{key} must be >= 0", i)
        if c.get("mute") is not None:
            if not isinstance(c.get("mute"), bool):
                bad("clip_mute",
                    f"clip {i + 1}: mute must be true or absent", i)
            elif c.get("mute") and kind != "video":
                bad("clip_mute_kind",
                    f"clip {i + 1}: only a video clip has sound to mute", i)
        if kind != "video" and c.get("speed") is not None \
                and abs(_f(c.get("speed"), 1.0) - 1.0) > 1e-9:
            bad("clip_speed_kind",
                f"clip {i + 1}: only a video clip has a clock to run fast or "
                f"slow", i)
        if c.get("source") not in ("auto", "human"):
            bad("clip_source", f"clip {i + 1}: source must be 'auto' or "
                               f"'human' (got {c.get('source')!r})", i)
        if not isinstance(c.get("locked", False), bool):
            bad("clip_locked", f"clip {i + 1}: locked must be true or false", i)
        proxy = c.get("proxy")
        if proxy is not None:
            if not isinstance(proxy, str):
                bad("clip_proxy", f"clip {i + 1}: proxy must be a string or "
                                  f"null", i)
            elif Path(proxy).is_absolute() or ".." in Path(proxy).parts:
                # The proxy is served by name out of the board's own folder.
                # An absolute path or a `..` would make that route a file
                # browser for the whole disk.
                bad("clip_proxy_escapes",
                    f"clip {i + 1}: proxy must be a relative name inside the "
                    f"board folder (got {proxy!r})", i)
        if fe > fs:
            spans.append((fs, fe, i))

    # THE OVERLAY LANE, held to the same rule as the picture lane and for the
    # same reason: it is ONE track, so two cards cannot play at once. An
    # overlay sitting ON a clip is not an overlap — that is the feature.
    ov = edit.get("overlays") if isinstance(edit, dict) else None
    if ov is not None and not isinstance(ov, list):
        bad("overlays_shape", "overlays must be a list or absent")
    elif isinstance(ov, list):
        ospans = []
        for j, o in enumerate(ov):
            if not isinstance(o, dict):
                bad("overlay_shape", f"overlay {j + 1}: must be an object", None)
                continue
            ofs, ofe = _f(o.get("film_start")), _f(o.get("film_end"))
            if ofs < 0:
                bad("overlay_film_start",
                    f"overlay {j + 1}: film_start must be >= 0")
            if ofe <= ofs:
                bad("overlay_window",
                    f"overlay {j + 1}: film_end ({ofe}) must be after "
                    f"film_start ({ofs})")
            okind = overlay_kind(o)
            if okind == "video" and not str(o.get("path") or ""):
                bad("overlay_path", f"overlay {j + 1}: needs a file")
            if okind == "text":
                # A TITLE IS ITS STRING. An empty one would render nothing
                # and say nothing about why.
                txt = o.get("text")
                if not isinstance(txt, str) or not txt.strip():
                    bad("overlay_text_empty",
                        f"overlay {j + 1}: a title needs some text")
                elif len(txt) > TEXT_MAX_CHARS:
                    bad("overlay_text_long",
                        f"overlay {j + 1}: a title is at most "
                        f"{TEXT_MAX_CHARS} characters")
                st = o.get("style")
                if st is not None and not isinstance(st, dict):
                    bad("overlay_text_style",
                        f"overlay {j + 1}: style must be an object or absent")
                elif isinstance(st, dict):
                    for key in ("font_size", "x", "y", "box_opacity"):
                        v = st.get(key)
                        if v is None:
                            continue
                        if not isinstance(v, (int, float)) or isinstance(v, bool) \
                                or v != v or v in (float("inf"), float("-inf")):
                            bad("overlay_text_style",
                                f"overlay {j + 1}: style.{key} must be a number")
                    fs_v = st.get("font_size")
                    if isinstance(fs_v, (int, float)) and not isinstance(fs_v, bool) \
                            and not TEXT_FONT_SIZE_MIN <= fs_v <= TEXT_FONT_SIZE_MAX:
                        bad("overlay_text_style_range",
                            f"overlay {j + 1}: font_size is {fs_v:g} — it must be "
                            f"between {TEXT_FONT_SIZE_MIN} and {TEXT_FONT_SIZE_MAX}")
                    for key in ("x", "y", "box_opacity"):
                        v = st.get(key)
                        if isinstance(v, (int, float)) and not isinstance(v, bool) \
                                and not 0.0 <= v <= 1.0:
                            bad("overlay_text_style_range",
                                f"overlay {j + 1}: style.{key} must be between "
                                f"0 and 1")
                    if st.get("align") is not None \
                            and str(st.get("align")).lower() not in TEXT_ALIGNS:
                        bad("overlay_text_style",
                            f"overlay {j + 1}: style.align must be one of "
                            f"{', '.join(TEXT_ALIGNS)}")
                    for key in ("color", "box_color"):
                        v = st.get(key)
                        if v is not None and _hex_colour(v, "") == "":
                            bad("overlay_text_style",
                                f"overlay {j + 1}: style.{key} must be a hex "
                                f"colour like #ffcc00")
                    if st.get("box") is not None and not isinstance(st.get("box"), bool):
                        bad("overlay_text_style",
                            f"overlay {j + 1}: style.box must be true or false")
            ofx = o.get("fx")
            if ofx is not None and not isinstance(ofx, dict):
                bad("overlay_fx",
                    f"overlay {j + 1}: fx must be an object or absent")
            elif isinstance(ofx, dict):
                for key in ("fade_in", "fade_out"):
                    v = ofx.get(key)
                    if v is None:
                        continue
                    if not isinstance(v, (int, float)) or isinstance(v, bool) \
                            or v != v or v in (float("inf"), float("-inf")):
                        bad(f"overlay_{key}",
                            f"overlay {j + 1}: fx.{key} must be a number")
                    elif float(v) < 0:
                        bad(f"overlay_{key}_range",
                            f"overlay {j + 1}: fx.{key} must be >= 0")
            if ofe > ofs:
                ospans.append((ofs, ofe, j))
        ospans.sort()
        for (a_s, a_e, a_i), (b_s, b_e, b_i) in zip(ospans, ospans[1:]):
            if b_s < a_e - TOUCH_TOLERANCE:
                bad("overlays_overlap",
                    f"overlays {a_i + 1} and {b_i + 1} overlap "
                    f"({a_s:.3f}-{a_e:.3f}s and {b_s:.3f}-{b_e:.3f}s) — one "
                    f"overlay lane can only show one of them")
    # THE TRANSITIONS, each of which must own a real boundary and have the
    # source handles to build it from. Every code here is an ERROR and none is
    # in WARNING_CODES: a transition the render cannot honour is not a note,
    # it is a film that would come out with a hard cut where a dissolve was
    # promised — and the sentence says which side is short and by how much.
    txr = edit.get("transitions") if isinstance(edit, dict) else None
    if txr is not None and not isinstance(txr, list):
        bad("transitions_shape", "transitions must be a list or absent")
    elif isinstance(txr, list):
        for n, t in enumerate(txr):
            if not isinstance(t, dict):
                bad("transition_shape", f"transition {n + 1}: must be an object")
        for t in resolve_transitions(edit):
            if t.get("problem"):
                bad(t["problem"]["code"], t["problem"]["message"],
                    t.get("out_index"))
    spans.sort()
    for (a_s, a_e, a_i), (b_s, b_e, b_i) in zip(spans, spans[1:]):
        if b_s < a_e - TOUCH_TOLERANCE:
            bad("clips_overlap",
                f"clips {a_i + 1} and {b_i + 1} overlap on the film "
                f"({a_s:.3f}-{a_e:.3f}s and {b_s:.3f}-{b_e:.3f}s) — one video "
                f"track can only play one of them", b_i)
    # THE SAME RULE FOR THE SOUND, and it is what keeps a split edit from
    # becoming the multi-track mixer the refuse list bans. A J-cut is a butt
    # join that lands somewhere the picture does not; it is still one lane, so
    # two clips' sound may no more overlap than two clips' pictures.
    asp: list[tuple[float, float, int]] = []
    for i, c in enumerate(clips):
        if not isinstance(c, dict) or clip_kind(c) != "video":
            continue
        w = clip_audio(c)
        if w["end"] > w["start"]:
            asp.append((w["film_start"], w["film_start"] + w["len"], i))
    asp.sort()
    for (a_s, a_e, a_i), (b_s, b_e, b_i) in zip(asp, asp[1:]):
        if b_s < a_e - TOUCH_TOLERANCE:
            bad("clips_audio_overlap",
                f"the sound of clips {a_i + 1} and {b_i + 1} overlaps "
                f"({a_s:.3f}-{a_e:.3f}s and {b_s:.3f}-{b_e:.3f}s) — a split "
                f"edit moves the sound, it does not add a second track", b_i)
    return errs


# The fields a digest IGNORES, and why each one is on the list. A field the
# user cannot see may never be the reason they are asked a question.
_DIGEST_SKIP_DOC = ("revision", "updated_at", "origin", "duration",
                    "migrated_from", "repaired_audio_overlaps", "audio_repair",
                    # The mix heal's own bookkeeping. `audio.mix` IS content —
                    # it is two controls on screen — but the marker that says
                    # the stamp has run is not, and a document differing only
                    # by it must not raise a chip about a field nobody can see.
                    "repaired_mix", "mix_repair",
                    "archived_at", "label", "backup_of", "backed_up_at",
                    "backup_revision", "session", "session_at")
_DIGEST_SKIP_CLIP = ("proxy", "analysis", "has_audio", "n", "title",
                     "placed", "duration")


def _digest_clip(c) -> dict:
    """One clip, reduced to what a person can see and change."""
    if not isinstance(c, dict):
        return {}
    out = {k: v for k, v in c.items() if k not in _DIGEST_SKIP_CLIP}
    for k in ("start", "end", "film_start", "film_end"):
        if k in out:
            out[k] = round(_f(out.get(k)), 6)
    return out


def edit_digest(edit) -> str:
    """A canonical fingerprint of the ARRANGEMENT, and nothing else.

    THE BUG THIS EXISTS FOR. The recovery offer compared `json.dumps(clips)` on
    both sides — but `_sbe_payload` REWRITES every clip's `proxy` pointer on
    the way out, deliberately, so a proxy built after the last save becomes
    visible without a re-save. The client's copy therefore differs from the
    file in a field the user has never heard of, on any board whose proxies
    were built after its last save. Forever. What that produced was a
    full-width bar reading "A backup from 3 min ago holds 10 clip(s) — the
    saved draft has 10. Nothing has been changed": a question about a
    difference it could not name, over a document that had loaded correctly.

    So the comparison is of the things somebody edited — the windows, the
    slots, the sound, the adjustments, the soundtrack, the beats — with every
    derived, rewritten and bookkeeping field removed by name. Two documents
    with the same digest are the same film, and the panel has nothing to say.
    """
    doc = edit if isinstance(edit, dict) else {}
    body = {k: v for k, v in doc.items()
            if k not in _DIGEST_SKIP_DOC and k != "clips"}
    body["clips"] = [_digest_clip(c) for c in (doc.get("clips") or [])
                     if isinstance(c, dict)]
    raw = json.dumps(body, sort_keys=True, separators=(",", ":"),
                     default=str)
    return hashlib.sha1(raw.encode("utf-8", "replace")).hexdigest()[:16]


# ---- which tab is EDITING ------------------------------------------------
# THE CLAIM IS A LABEL, NOT A LOCK, AND IT IS TAKEN BY WRITING.
#
# It used to be taken by LOADING, and it used to gate the snapshot lane. That
# combination cost the owner an afternoon: a passive page load — a headless
# browser, a second window, an agent reading the board, a preview — claimed the
# board without editing anything, and the tab the person was actually cutting
# in was told `stale_session` and stopped writing snapshots for seven hours.
# Nothing on screen said so (see `sbeQueueSave`), and the safety net that was
# meant to catch exactly this had been starved of the timestamp it watches.
#
# Two rules come out of that, and both are load-bearing:
#
#   1. A READ CLAIMS NOTHING. Looking at a film is not editing it. Anything
#      that only reads — every GET, every agent, every preview — leaves the
#      claim exactly where it was.
#   2. THE LANE NEVER REFUSES. `write_backup`'s own docstring says the safety
#      net may not have an opinion; the session check was the last opinion it
#      had, and it was the one that switched the net off. The lane is versioned
#      now — ONE FILE PER SNAPSHOT, pruned, never overwritten — so the failure
#      the token was introduced for (a stale tab stomping a single shared slot)
#      cannot happen: a snapshot from any tab costs one file and can only ever
#      ADD a way back. Refusing one can only ever remove one.
#
# What the claim is still for: saying, in a tab, that somebody else is editing
# this film too. That is information a person can act on. It is never a reason
# to stop protecting them.
SESSION_STALE_AFTER = 60 * 60 * 12       # a token nobody has used all day


def session_token_path(board_dir) -> Path:
    return Path(str(board_dir)) / "session.json"


def current_session(board_dir) -> dict:
    """{"token", "at"} for the session that last claimed this board."""
    doc = _read_json(session_token_path(board_dir)) or {}
    if not isinstance(doc, dict):
        return {"token": "", "at": 0}
    return {"token": str(doc.get("token") or ""),
            "at": int(doc.get("at") or 0)}


def claim_session(board_dir, token: str) -> dict:
    """This tab is the one EDITING. Returns the claim.

    Called from the write path and from nowhere else. Claiming is not a lock
    and never refuses anybody: what it buys is that a tab can be TOLD another
    tab is editing the same film.
    """
    tok = str(token or "").strip()[:64]
    if not tok:
        raise EditError("a session needs a token")
    Path(str(board_dir)).mkdir(parents=True, exist_ok=True)
    doc = {"token": tok, "at": int(time.time())}
    _atomic_json(session_token_path(board_dir), doc, prefix=".session-")
    return doc


def session_is_current(board_dir, token: str) -> bool:
    """Does this token hold the claim? An UNCLAIMED board says yes.

    INFORMATION, NOT PERMISSION. Nothing refuses a write on this any more —
    see the block above `SESSION_STALE_AFTER`. It answers the question a tab
    asks in order to tell its user that somebody else is editing too.

    A board nobody has claimed is every board written before this existed, and
    an agent or a script posting to the routes has no tab to claim one — so the
    absence of a claim can never be a refusal. Only a claim by SOMEBODY ELSE,
    recently, is.
    """
    tok = str(token or "").strip()
    cur = current_session(board_dir)
    if not cur["token"] or not tok:
        return True
    if cur["token"] == tok:
        return True
    return (int(time.time()) - cur["at"]) > SESSION_STALE_AFTER


def blocking_errors(errs) -> list[dict]:
    """The subset of `validate_edit`'s answer that must stop a write."""
    return [e for e in (errs or [])
            if isinstance(e, dict) and e.get("code") not in WARNING_CODES]


def repair_audio_overlaps(edit: dict) -> int:
    """Trim the DUPLICATED strips an old split left behind. Returns how many.

    The split before the J-cut fix deep-copied `clip.audio` into the new half,
    so one strip became two claiming the same seconds of the same take. The
    document was legal when it was written and became unsaveable the day the
    overlap rule arrived — a board carrying the artifact could not be stored at
    all, which is a migration problem wearing a validation error's clothes.

    A MIGRATION, NOT A RESIDENT. This used to run on every read, which makes
    it a second author of a document the user owns — the exact thing the
    save-model ruling abolished. A repair that is wrong even once is then
    wrong forever, invisibly, on every load. `migrate_edit` runs it ONCE per
    board and stamps `audio_repair` so it never runs again.

    ONLY THE ARTIFACT, AND THE ARTIFACT IS NARROW. Using one take twice is
    ordinary editing, so "two clips of the same source" is evidence of
    nothing. What the old deep copy produced is far more specific:

      * BOTH halves carry an EXPLICIT `audio` object. A clip with no strip of
        its own gets a SYNTHESISED window from `clip_audio`, and comparing a
        synthesised window against a real one compares an edit to a default.
      * The two objects are IDENTICAL — the same source window AND the same
        `film_start` — because they are one dict copied twice.
      * They are ADJACENT and butt-joined, being the two halves of one clip.
      * NEITHER carries a deliberate offset. A strip whose `film_start`
        differs from its picture's is somebody's J-cut, and a J-cut is never
        repair material: "I wanted it a little before her showing up, and I
        cut it that way."
    """
    clips = [c for c in (edit or {}).get("clips") or [] if isinstance(c, dict)]
    order = sorted(clips, key=lambda x: _f(x.get("film_start")))
    fixed = 0
    for a, b in zip(order, order[1:]):
        if clip_kind(a) != "video" or clip_kind(b) != "video":
            continue
        # BOTH halves carry an EXPLICIT strip. A clip with no `audio` of its
        # own gets a SYNTHESISED window, and comparing that against a real one
        # compares an edit to a default — which is how a second, deliberate
        # use of the same take read as a copy of the first.
        aa, ba = a.get("audio"), b.get("audio")
        if not isinstance(aa, dict) or not isinstance(ba, dict):
            continue
        if str(a.get("path") or "") != str(b.get("path") or ""):
            continue
        wa, wb = clip_audio(a), clip_audio(b)
        # ONE DICT, COPIED TWICE: identical source window AND identical
        # film_start. The second half's `film_start` still points at the
        # FIRST half's position, which is exactly what makes it stale.
        if (wa["start"], wa["end"], wa["film_start"]) != \
                (wb["start"], wb["end"], wb["film_start"]):
            continue
        # The original was in sync, so the FIRST half still is. A first half
        # carrying an offset is somebody's J-cut, not a copy.
        if abs(wa["film_start"] - _f(a.get("film_start"))) > TOUCH_TOLERANCE:
            continue
        # The two halves of one split are adjacent AND butt-joined.
        if abs(_f(a.get("film_end")) - _f(b.get("film_start"))) > TOUCH_TOLERANCE:
            continue
        # Re-cut both at the picture's own boundary, in the SOURCE clock the
        # halves share — the rule the fixed `sbeSplitAt` follows, after the
        # fact.
        sp = _f(a.get("end"))
        if not (wa["start"] + TOUCH_TOLERANCE < sp < wa["end"] - TOUCH_TOLERANCE):
            continue
        a["audio"] = {"start": wa["start"], "end": round(sp, 6),
                      "film_start": wa["film_start"]}
        if wa.get("coupled"):
            a["audio"]["linked"] = True
        b["audio"] = {"start": round(sp, 6), "end": wb["end"],
                      "film_start": round(wb["film_start"] + (sp - wb["start"]), 6)}
        if wb.get("coupled"):
            b["audio"]["linked"] = True
        fixed += 1
    return fixed


def normalise_edit(edit: dict) -> dict:
    """Sort by film position, round, fill in what can be derived. No policy.

    Deliberately does NOT close gaps, renumber, or re-snap: an editor that
    quietly rearranges what the user saved is an editor the user stops
    trusting. The only thing invented here is a missing `id`.
    """
    out = dict(edit or {})
    out["version"] = EDIT_VERSION
    clips = [c for c in (out.get("clips") or []) if isinstance(c, dict)]
    clips.sort(key=lambda c: (_f(c.get("film_start")), str(c.get("path"))))
    for c in clips:
        for k in ("start", "end", "film_start", "film_end"):
            if k in c:
                c[k] = round(_f(c.get(k)), 6)
        kind = clip_kind(c)
        if kind == "video":
            # An absent `kind` stays absent. Stamping "video" on 400 clips
            # would rewrite every edit.json on the machine to say the thing
            # its absence already says.
            c.pop("kind", None)
        else:
            c["kind"] = kind
            # SYNTHESISED, not trusted. A still and a slug have exactly one
            # length — the slot — and deriving the window from it here is what
            # lets the client's trim machinery resize a still by moving its
            # slot and nothing else.
            c["start"] = 0.0
            c["end"] = round(max(0.0, _f(c.get("film_end"))
                                 - _f(c.get("film_start"))), 6)
            if kind == "slug":
                c["path"] = None
                c["proxy"] = None
            # Neither has a source clock, so neither has a source duration —
            # and a `duration` left over from a video would clamp the trim.
            c["duration"] = None
        # NEUTRAL IS ABSENT, the rule `adjust` and the music trims follow: an
        # unmuted clip is byte-identical to one written before mute existed.
        if kind == "video" and c.get("mute") is True:
            c["mute"] = True
        else:
            c.pop("mute", None)
        # SPEED, same rule: 1x is the absence of the field, clamped on the way
        # to disk so every output is handed a legal rate.
        sp = clip_speed(c)
        if kind == "video" and abs(sp - 1.0) > 1e-9:
            c["speed"] = sp
        else:
            c.pop("speed", None)
        # NEUTRAL IS ABSENT here too, so a timeline nobody has put an effect on
        # is byte-identical to one from before effects existed. The values are
        # written back CLAMPED, so what is on disk is what all three outputs
        # will be handed.
        e = clip_effects(c)
        keep = {k: e[k] for k in ("fade_in", "fade_out") if e[k] > 1e-9}
        if keep:
            c["fx"] = keep
        else:
            c.pop("fx", None)
        # THE SOUND'S ENVELOPE, same neutral-is-absent rule. Clamped on the
        # way to disk so all three outputs read the same legal curve.
        if kind == "video":
            w = clip_audio(c)
            ae = audio_effects(c, w["len"])
            akeep = {k: ae[k] for k in ("fade_in", "fade_out") if ae[k] > 1e-9}
            if ae["points"]:
                akeep["points"] = ae["points"]
            if akeep:
                c["afx"] = akeep
            else:
                c.pop("afx", None)
        else:
            c.pop("afx", None)
        if kind != "video":
            c.pop("audio", None)
        elif isinstance(c.get("audio"), dict):
            # Rounded, never removed: re-linking is the TOGGLE's job (it
            # deletes the field), and stripping it here because the numbers
            # happen to match the picture would silently re-link a clip the
            # user had deliberately unlinked and not yet moved.
            w = clip_audio(c)
            c["audio"] = {"start": w["start"], "end": w["end"],
                          "film_start": w["film_start"]}
            # ONLY WHEN IT SAYS SOMETHING, the rule `adjust` and the music
            # trims already follow: a free strip is the absence of the flag,
            # so a document that never coupled anything is byte-identical to
            # one written before coupling existed.
            if w["coupled"]:
                c["audio"]["linked"] = True
        elif c.get("audio") is not None:
            c.pop("audio", None)
        # FRAMING, neutral-is-absent like everything else here, and written
        # back clamped so all three outputs read the same window.
        if kind != "slug" and isinstance(c.get("frame"), dict) and not clip_frame_is_neutral(c):
            c["frame"] = clip_frame(c)
        else:
            c.pop("frame", None)
        adj = c.get("adjust")
        if isinstance(adj, dict):
            b = clip_brightness(c)
            rest = {k: v for k, v in adj.items() if k != "brightness"}
            # NEUTRAL IS ABSENT. Dragging the slider back to zero must leave a
            # document identical to one that never had a slider, or every clip
            # anyone ever touched carries a dead field forever.
            if abs(b) < 1e-9 and not rest:
                c.pop("adjust", None)
            else:
                c["adjust"] = dict(rest, brightness=round(b, 6)) if abs(b) >= 1e-9 else rest
        elif adj is not None:
            c.pop("adjust", None)
        if not c.get("id"):
            c["id"] = _clip_id(c.get("path"), _f(c.get("start")),
                               _f(c.get("film_start")))
        if c.get("source") not in ("auto", "human"):
            c["source"] = "auto"
        c["locked"] = bool(c.get("locked", False))
    out["clips"] = clips
    # THE OVERLAY LANE. Same discipline as the clips: rounded, sorted, derived
    # where it can be derived, and NOTHING invented. An empty lane is an absent
    # key, so a timeline that has never had an overlay is byte-identical to one
    # from before the lane existed.
    ovs = [o for o in (out.get("overlays") or []) if isinstance(o, dict)]
    ovs.sort(key=lambda o: (_f(o.get("film_start")), str(o.get("path") or "")))
    for o in ovs:
        for k in ("film_start", "film_end"):
            o[k] = round(_f(o.get(k)), 6)
        o["kind"] = overlay_kind(o)
        if o["kind"] == "text":
            # A TITLE IS ITS SLOT AND ITS STRING. No file, no source clock —
            # the same synthesis a still gets — and the style is written back
            # CLAMPED and only where it differs from the default, so a title
            # left at the defaults carries no style at all.
            o["start"] = 0.0
            o["end"] = round(max(0.0, o["film_end"] - o["film_start"]), 6)
            o["duration"] = None
            o["path"] = None
            o["proxy"] = None
            tt = overlay_text(o)
            o["text"] = tt["text"]
            skeep = {k: v for k, v in tt["style"].items()
                     if v != TEXT_STYLE_DEFAULTS[k]}
            if skeep:
                o["style"] = skeep
            else:
                o.pop("style", None)
        elif o["kind"] == "still":
            # A still is its slot and nothing else — the same synthesis a
            # still CLIP gets, and the reason the trim handles can resize it.
            o["start"] = 0.0
            o["end"] = round(max(0.0, o["film_end"] - o["film_start"]), 6)
            o["duration"] = None
        else:
            for k in ("start", "end"):
                o[k] = round(_f(o.get(k)), 6)
        if not o.get("id"):
            o["id"] = _clip_id(o.get("path") or o.get("text"), _f(o.get("start")),
                               _f(o.get("film_start")))
        if o.get("source") not in ("auto", "human"):
            o["source"] = "human"
        o["locked"] = bool(o.get("locked", False))
        e = clip_effects(o)
        keep = {k: e[k] for k in ("fade_in", "fade_out") if e[k] > 1e-9}
        if keep:
            o["fx"] = keep
        else:
            o.pop("fx", None)
    if ovs:
        out["overlays"] = ovs
    else:
        out.pop("overlays", None)
    # THE TRANSITIONS. Rounded, sorted by the boundary they sit on, ids
    # invented where missing, and NOTHING ELSE decided: the duration on disk
    # is the one the user asked for, and `transition_duration` clamps it on
    # every read against whatever the neighbours are that day — writing the
    # clamped number back would let one trim silently shorten a dissolve for
    # good. An empty list is an absent key.
    txs = []
    for t in transition_items(out):
        t = dict(t)
        t["after_clip"] = str(t.get("after_clip") or "")
        t["kind"] = str(t.get("kind") or "").strip().lower()
        t["duration"] = round(_f(t.get("duration")), 6)
        if not t.get("id"):
            t["id"] = _clip_id(t["after_clip"], t["duration"], 0.0)
        txs.append(t)
    at = {r["id"]: r["at"] for r in resolve_transitions({"clips": clips,
                                                         "transitions": txs})}
    txs.sort(key=lambda t: (at.get(t["id"], float("inf")), t["id"]))
    if txs:
        out["transitions"] = txs
    else:
        out.pop("transitions", None)
    out.setdefault("board_id", "")
    # NEUTRAL IS ABSENT, the same rule `adjust` follows one block up. Dragging
    # a trim handle back to the end of the track must leave a document
    # identical to one that was never trimmed, or every timeline anybody ever
    # touched the music on carries two dead fields forever.
    aud = out.get("audio")
    if isinstance(aud, dict):
        aud = dict(aud)
        if "offset" in aud:
            aud["offset"] = round(_f(aud.get("offset")), 6)
        ts = aud.get("trim_start")
        if isinstance(ts, (int, float)) and not isinstance(ts, bool):
            ts = round(max(0.0, _f(ts)), 6)
            if ts <= 0:
                aud.pop("trim_start", None)
            else:
                aud["trim_start"] = ts
        elif ts is not None:
            aud.pop("trim_start", None)
        te = aud.get("trim_end")
        if isinstance(te, (int, float)) and not isinstance(te, bool):
            te = round(_f(te), 6)
            dur = _f(aud.get("duration"))
            if te <= 0 or (dur > 0 and te >= dur - 1e-6):
                aud.pop("trim_end", None)
            else:
                aud["trim_end"] = te
        elif te is not None:
            aud.pop("trim_end", None)
        bed_afx = aud.get("afx")
        if isinstance(bed_afx, dict):
            # CLAMPED TO THE PLAYED WINDOW, not to the track. `bed_length` is
            # the bed envelope's clock (see its docstring) — clamping against
            # the whole file would let a fade-out sit past the last second
            # anybody hears, where no output could express it.
            be = audio_effects(aud, bed_length(aud, edit_duration(out)))
            bkeep = {k: be[k] for k in ("fade_in", "fade_out") if be[k] > 1e-9}
            if be["points"]:
                bkeep["points"] = be["points"]
            if bkeep:
                aud["afx"] = bkeep
            else:
                aud.pop("afx", None)
        elif bed_afx is not None:
            aud.pop("afx", None)
        # NEUTRAL IS ABSENT, the rule the trims and `afx` above already
        # follow. A mix left at the new default is byte-identical to a
        # document written before the mix was a thing anybody could set.
        mx = aud.get("mix")
        if isinstance(mx, dict):
            m = audio_mix(aud)
            mkeep = {}
            if abs(m["bed_gain"] - MIX_BED_GAIN) > 1e-9:
                mkeep["bed_gain"] = m["bed_gain"]
            if m["duck"] != MIX_DUCK:
                mkeep["duck"] = m["duck"]
            if mkeep:
                aud["mix"] = mkeep
            else:
                aud.pop("mix", None)
        elif mx is not None:
            aud.pop("mix", None)
        out["audio"] = aud
    out.setdefault("audio", None)
    out.setdefault("beats", None)
    out.setdefault("settings", {})
    out.setdefault("source", "auto")
    out["duration"] = edit_duration(out)
    return out


# ---------------------------------------------------------------------------
# ONE WRITER AT A TIME, PER BOARD
# ---------------------------------------------------------------------------
# THE DEFECT: `expect_revision` is a compare-and-swap whose compare and whose
# swap were in different places. The HTTP handler read the revision off disk,
# compared it, validated, and only then called `save_edit` — and the panel runs
# on a `ThreadingHTTPServer`, so two tabs whose debounces landed together both
# read revision 7, both compared 7 == 7, and both wrote. Both got HTTP 200.
# One arrangement was gone, recoverable only from `history/` and only by
# somebody who knew to look. The guard existed and the race walked through it.
#
# So the read, the check and the write happen inside ONE critical section, and
# it lives HERE rather than in the handler because this is the function every
# writer already goes through — the auto-editor, the relink, the drafts, the
# restore. A guard the caller has to remember to take is a guard.
#
# PER BOARD, KEYED BY THE RESOLVED DIRECTORY. Two people cutting two different
# films have nothing to say to each other and must not queue behind one
# another; two tabs on one film have everything to say to each other. An RLock
# so a writer that reaches a second write on the same board cannot deadlock
# itself. The registry is bounded by the number of boards that have ever been
# saved in this process, which is the number of boards.
_BOARD_LOCKS: dict[str, threading.RLock] = {}
_BOARD_LOCKS_GUARD = threading.Lock()


def board_write_lock(board_dir) -> threading.RLock:
    """The one lock that serialises writes to one board's edit.json."""
    key = str(Path(str(board_dir)).resolve())
    with _BOARD_LOCKS_GUARD:
        lk = _BOARD_LOCKS.get(key)
        if lk is None:
            lk = _BOARD_LOCKS[key] = threading.RLock()
        return lk


def on_disk_revision(board_dir) -> int:
    """The revision in the file RIGHT NOW, read raw and cheaply.

    Deliberately not `load_edit`: the question is "what number is on disk",
    not "what does this document mean once migrated", and a migration that
    ever touched `revision` would turn a comparison into a coin toss. A
    missing or unreadable file is revision 0 — the same answer the handler's
    `load_edit(...) or {}` already gives — because refusing a save over a file
    nobody can read would strand the work rather than persist it.
    """
    try:
        p = edit_path(board_dir)
        if not p.is_file():
            return 0
        doc = json.loads(p.read_text(encoding="utf-8"))
        return int((doc or {}).get("revision") or 0)
    except (OSError, ValueError, TypeError):
        return 0


def save_edit(board_dir, edit: dict, *, bump: bool = True,
              origin: str = "manual", expect: int | None = None) -> Path:
    """Validate, then write atomically. A bad edit NEVER lands on a good one.

    This is the whole reason `validate_edit` returns structured errors instead
    of raising at the first one: the caller gets the complete list of what is
    wrong with the document it tried to save, and the document already on disk
    is untouched. `os.replace` after `fsync` means the reader either sees the
    old file or the new one — never half of either.

    `expect` is the revision the caller believes is on disk, and passing it
    makes this a COMPARE-AND-SWAP rather than a write: the comparison happens
    under the board's write lock, in the same critical section as the write, so
    a second tab that slipped in between the caller's read and this call raises
    `EditConflict` instead of silently overwriting an arrangement. `None` means
    last-write-wins, which is what the deliberate "Keep mine" button asks for.

    THE LOCK IS TAKEN WHETHER OR NOT `expect` IS. Two unguarded writers still
    must not interleave a history copy with a replace.
    """
    errs = blocking_errors(validate_edit(edit))
    if errs:
        raise EditError("; ".join(e["message"] for e in errs[:6])
                        + (f" (+{len(errs) - 6} more)" if len(errs) > 6 else ""))
    doc = normalise_edit(edit)
    with board_write_lock(board_dir):
        return _save_edit_locked(board_dir, doc, bump=bump, origin=origin,
                                 expect=expect)


def _save_edit_locked(board_dir, doc: dict, *, bump: bool, origin: str,
                      expect: int | None) -> Path:
    """The critical section: compare, stamp, archive, replace. Lock held."""
    if expect is not None:
        current = on_disk_revision(board_dir)
        try:
            want = int(expect)
        except (TypeError, ValueError):
            want = -1
        if want != current:
            raise EditConflict(
                f"this timeline moved on without you — it is at revision "
                f"{current}, you saved from {expect}", revision=current)
        # THE COUNTER IS THE SERVER'S, and it counts from what is on disk. A
        # stale tab must not be able to wind it backwards, which is the same
        # rule the handler stated and could not enforce on its own.
        doc["revision"] = current
    if bump:
        try:
            doc["revision"] = int(doc.get("revision") or 0) + 1
        except (TypeError, ValueError):
            doc["revision"] = 1
    doc["updated_at"] = int(time.time())
    # WHO ASKED FOR THIS SAVE. Stamped in the document so a file read on its
    # own can still answer, and read by `archive_edit` to decide which lane
    # the OUTGOING document belongs in.
    doc["origin"] = origin if origin in ORIGINS else "manual"
    # EVERY SAVE KEEPS ITS PREDECESSOR. The editor used to hold exactly one
    # save per film: no history, and the undo stack dies with the tab — so an
    # arrangement edited past was gone the moment the debounce fired, with the
    # owner asking "there was a version I was working on that was better than
    # this one — is it lost?" and the only honest answer being yes. Before the
    # new document lands, the outgoing one is copied into history/, named by
    # its own revision. ~5 KB per save, capped, and a cut is suddenly a thing
    # you can walk back through instead of a thing you can only overwrite.
    try:
        cur = edit_path(board_dir)
        if cur.is_file():
            archive_edit(board_dir, json.loads(cur.read_text()))
            prune_history(board_dir)
    except Exception:                                                # noqa: BLE001
        # History must never block a save — losing a breadcrumb is nothing
        # next to refusing to persist the arrangement on screen.
        pass
    return _atomic_json(edit_path(board_dir), doc, prefix=".edit-")


# ---------------------------------------------------------------------------
# PART C2 — history: the autosaves, and the ones somebody named
# ---------------------------------------------------------------------------
# TWO KINDS OF FILE IN ONE FOLDER, told apart by their prefix rather than by
# reading them: `edit-r*.json` is an autosave and the prune walks it,
# `keep-r*.json` is a version a person named and the prune never sees it. That
# is the whole exemption, and it is a glob rather than a flag inside the file
# because a prune that has to open fifty documents to decide what to delete is
# a prune that fails halfway on the first corrupt one.
# THREE LANES IN ONE FOLDER, told apart by their PREFIX rather than by reading
# them. The owner: "the auto saves should be saved separately from the manual
# saves, at least, so the user can go back and see the manual saves." A person
# walking back through their work should meet their own decisions first, not a
# wall of machine noise — and the prune should never be able to eat a decision.
#   edit-r*  an automatic snapshot. Pruned, capped, secondary in the list.
#   save-r*  a save the USER pressed. Never auto-pruned, primary in the list.
#   keep-r*  a version the user NAMED. Never auto-pruned either.
# A glob rather than a flag inside the file, because a prune that has to open
# fifty documents to decide what to delete fails halfway on the first corrupt
# one. The origin is stamped in the JSON too, for anything reading a file on
# its own.
# Bumped only if a NEW legacy artifact is ever found. A board stamped with the
# current number has been through the migration and is never re-examined.
AUDIO_REPAIR_VERSION = 1
_AUTO_PREFIX = "edit-r"
# THE SIDE ARCHIVE. One file per snapshot, named by the clock, in the draft's
# own history folder — so writing a new one can never destroy the last one.
# The single `backup-<draft>.json` slot it replaces is why the client had to
# carry `if (SBE.backup) return false`: with one slot, snapshotting while an
# unanswered offer was on screen would have eaten the work the offer held, so
# the lane switched itself off for the rest of the session the moment a chip
# appeared. A lane that stops is not a safety net. See
# docs/EDITOR_SAVE_MODEL.md §2.
_SNAP_PREFIX = "snap-"
SNAPSHOT_KEEP = 20
_KEPT_PREFIX = "keep-r"
_MANUAL_PREFIX = "save-r"
ORIGINS = ("manual", "auto", "backup")


def _history_root(board_dir) -> Path:
    """Where the crash backups live, and the drafts' history folders under it."""
    return Path(str(board_dir)) / "history"


def _migrate_history_layout(board_dir) -> int:
    """Fold pre-drafts history entries into the FIRST draft's folder.

    Entries used to be named by revision alone — `save-r00003.json` — in one
    folder per BOARD, while every new draft restarts its revision counter at
    zero. So draft B's revision 3 collided with draft A's, `archive_edit`
    dropped the collision without a word (`if dst.exists(): return None`), and
    the rows that survived were listed with no draft attribution at all: the
    picker shown while B was open offered A's arrangements, and Restore wrote
    one of them into B — the exact overwrite the history folder exists to
    prevent, arriving through its own front door.

    What is on disk from before that fix belongs to the film's FIRST draft:
    those entries predate the second draft's existence, and the ones a second
    draft did write were the ones the collision swallowed. So they move there,
    rather than being left in a folder nothing lists any more.
    """
    root = _history_root(board_dir)
    if not root.is_dir():
        return 0
    legacy = [p for p in root.iterdir()
              if p.is_file() and p.name.endswith(".json")
              and p.name.startswith((_AUTO_PREFIX, _MANUAL_PREFIX,
                                     _KEPT_PREFIX))]
    if not legacy:
        return 0
    rows = load_draft_index(board_dir)["drafts"]
    home = root / _slug(str(rows[0]["slug"]), 40)
    home.mkdir(parents=True, exist_ok=True)
    moved = 0
    for src in legacy:
        dst = home / src.name
        if dst.exists():
            dst = home / (src.stem + "-legacy.json")
        try:
            src.replace(dst)
            moved += 1
        except OSError:
            continue
    return moved


def history_dir(board_dir, slug: str = "") -> Path:
    """The history folder for ONE draft — the named one, or the active one.

    A FOLDER PER DRAFT RATHER THAN A NAME PER DRAFT. The alternative was to
    fold the slug into the filename, which puts the draft's identity in a
    string three different globs have to parse back out — and a draft named
    `r00007` would then be indistinguishable from a revision. A directory
    scopes the prune, the listing and the restore by construction, and
    deleting a draft's past saves becomes deleting a directory.
    """
    _migrate_history_layout(board_dir)
    return _history_root(board_dir) / _slug(
        str(slug or load_draft_index(board_dir)["active"]), 40)


def archive_edit(board_dir, doc: dict, label: str = "") -> Path | None:
    """Copy a document into `history/`. Returns the file, or None if it exists.

    The label rides BOTH in the filename (so the prune can see it without
    opening anything) and inside the JSON (so the listing does not have to
    parse filenames back into prose). `archived_at` is stamped here because a
    file's mtime is the one piece of this that a backup restore can change.
    """
    hist = history_dir(board_dir)
    hist.mkdir(parents=True, exist_ok=True)
    try:
        rev = int(doc.get("revision") or 0)
    except (TypeError, ValueError):
        rev = 0
    text = str(label or "").strip()[:80]
    origin = str(doc.get("origin") or "auto").lower()
    if text:
        dst = hist / f"{_KEPT_PREFIX}{rev:05d}-{_slug(text, 40)}.json"
    elif origin == "manual":
        dst = hist / f"{_MANUAL_PREFIX}{rev:05d}.json"
    else:
        dst = hist / f"{_AUTO_PREFIX}{rev:05d}.json"
    if dst.exists():
        return None
    out = dict(doc)
    if text:
        out["label"] = text
    out["archived_at"] = int(time.time())
    dst.write_text(json.dumps(out), encoding="utf-8")
    return dst


def prune_history(board_dir, *, keep: int = EDIT_HISTORY_KEEP) -> int:
    """Drop the oldest AUTOSAVES past the cap. Named versions are never touched.

    A version somebody stopped to name is the one thing in this folder that is
    not a breadcrumb, and the fifty-save cap would otherwise delete it in an
    afternoon of debounced saves.
    """
    hist = history_dir(board_dir)
    if not hist.is_dir():
        return 0
    files = sorted(hist.glob(f"{_AUTO_PREFIX}*.json"))
    gone = 0
    for stale in files[:max(0, len(files) - int(keep))]:
        try:
            stale.unlink()
            gone += 1
        except OSError:
            continue
    return gone


def _history_meta(path: Path) -> dict:
    """One row for the picker. A file it cannot read is REPORTED, not skipped.

    A history entry that silently vanishes from the list is indistinguishable
    from one that was never written, and the whole point of this folder is that
    the user can see what they still have.
    """
    kept = path.name.startswith(_KEPT_PREFIX)
    manual = kept or path.name.startswith(_MANUAL_PREFIX)
    snap = path.name.startswith(_SNAP_PREFIX)
    row = {"file": path.name, "label": "", "revision": None, "snapshot": snap,
           "clips": None, "duration": None,
           "saved_at": None, "archived_at": None,
           "kept": kept, "manual": manual,
           "origin": "manual" if manual else "auto", "readable": True}
    try:
        st = path.stat()
        row["archived_at"] = int(st.st_mtime)
    except OSError:
        pass
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        row["readable"] = False
        return row
    if not isinstance(doc, dict):
        row["readable"] = False
        return row
    row["label"] = str(doc.get("label") or "")[:80]
    if doc.get("origin") in ORIGINS:
        row["origin"] = str(doc["origin"])
        row["manual"] = row["manual"] or row["origin"] == "manual"
    if snap:
        row["origin"] = "backup"
        row["manual"] = False
    try:
        row["revision"] = int(doc.get("revision") or 0)
    except (TypeError, ValueError):
        row["revision"] = 0
    clips = doc.get("clips")
    row["clips"] = len(clips) if isinstance(clips, list) else 0
    row["duration"] = edit_duration(doc)
    for key in ("archived_at", "updated_at"):
        v = doc.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            row["archived_at" if key == "archived_at" else "saved_at"] = int(v)
    return row


def list_history(board_dir) -> list[dict]:
    """The ACTIVE draft's archived documents, NEWEST FIRST.

    Scoped to the draft on screen, because a version picker that offers
    another draft's arrangements is a Restore button that overwrites the film
    you are looking at with a film you are not.
    """
    hist = history_dir(board_dir)
    if not hist.is_dir():
        return []
    # A WHITELIST, NOT A BLACKLIST. pathlib's glob returns dotfiles, and
    # `_atomic_json` writes its temp beside the target with a leading dot — so
    # a crash between mkstemp and os.replace left `.edit-XXXX.json` in here,
    # which listed as an unnamed row with revision None and was clickable.
    # Naming the three lanes we write means nothing else can appear.
    rows = [_history_meta(p) for p in hist.glob("*.json")
            if p.is_file() and p.name.startswith((_AUTO_PREFIX, _MANUAL_PREFIX,
                                                  _KEPT_PREFIX, _SNAP_PREFIX))]
    rows.sort(key=lambda r: (r.get("archived_at") or 0,
                             r.get("revision") or 0), reverse=True)
    return rows


def restore_edit(board_dir, name: str) -> dict:
    """Put a history entry back, keeping what it replaced. Returns the document.

    THE CURRENT ARRANGEMENT IS ARCHIVED FIRST. Restore is the one action in
    this editor that overwrites a document the user can see, and a restore that
    ate the thing it replaced would be the same bug the history folder exists
    to fix, arriving through the fix's own front door.

    The revision goes FORWARD, never back to the archived one: `revision` is
    how two tabs tell who moved last, and handing back an older number would
    make a stale tab look current.
    """
    hist = history_dir(board_dir)
    safe = Path(str(name or "")).name
    if not safe or safe != str(name or "") or not safe.endswith(".json") \
            or not safe.startswith((_AUTO_PREFIX, _MANUAL_PREFIX,
                                    _KEPT_PREFIX, _SNAP_PREFIX)):
        raise EditError("that is not a version of this film")
    # `hist` is the ACTIVE draft's folder, so a filename from another draft's
    # picker resolves outside it and is refused here rather than installed.
    src = hist / safe
    if not src.is_file() or src.resolve().parent != hist.resolve():
        raise EditError("there is no version by that name")
    try:
        doc = json.loads(src.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EditError(f"that version cannot be read: {exc}") from exc
    if not isinstance(doc, dict):
        raise EditError("that version is not an edit document")
    # A history file may predate the current schema — it is a document like any
    # other, and the read path is where documents are brought forward.
    doc = migrate_edit(doc)
    doc.pop("label", None)
    doc.pop("archived_at", None)
    current = None
    try:
        cur = edit_path(board_dir)
        if cur.is_file():
            current = json.loads(cur.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        current = None
    if current is not None:
        archive_edit(board_dir, current)
        prune_history(board_dir)
    try:
        doc["revision"] = int((current or {}).get("revision") or 0)
    except (TypeError, ValueError):
        doc["revision"] = 0
    # BLOCKING ONLY, for the same reason `recover_backup` does it: every
    # writer archives documents that carry warnings, so every reader must be
    # able to hand them back. A history entry holding a J-cut was
    # unrestorable — the version picker offered it and then refused it.
    errs = blocking_errors(validate_edit(doc))
    if errs:
        raise EditError("that version cannot be restored: "
                        + "; ".join(e["message"] for e in errs[:3]))
    save_edit(board_dir, doc)
    return load_edit(board_dir)


# ---------------------------------------------------------------------------
# THE FRAME GRID — a hole nobody can see, and nobody can close
# ---------------------------------------------------------------------------
# "There is like a black frame that flashes for a microsecond... it's not all
# the cuts, but some of the cuts have like a black frame between them. I tried
# to drag them close and whatever."
#
# The last sentence is the defect. Measured on the sequence that produced the
# report — 24 fps, one frame = 41.67 ms — three holes, all SHORTER THAN A
# FRAME: 20.94 ms (0.503 frame), 15.84 ms (0.380) and 4.00 ms (0.096). A hole
# that size is
#
#   * invisible — at the deepest zoom the timeline offers it is a fraction of
#     a pixel, so there is nothing on screen to drag;
#   * unreachable — dragging sets a gap from a pixel, and no pixel maps to
#     4 ms;
#   * unreportable — `edit_gaps` and `sbeHoles` both ignored anything under
#     half a frame, which is why the header read "1 hole" while three existed;
#   * and still LOUD, because the stage paints black wherever no clip is
#     playing, so a third of a frame of nothing is a black frame.
#
# WHERE THE GRID IS ENFORCED, AND WHERE IT IS NOT. On the GAPS, not on the
# absolute film positions. A clip's slot length IS its source trim — the
# validator refuses a document whose slot and window disagree by more than
# `LENGTH_TOLERANCE`, and rightly, because nothing here plays at anything but
# 1x — so snapping `film_end` to a frame would mean re-trimming what the user
# cut. With every gap an exact multiple of a frame and the film starting at 0,
# every CUT between two adjacent clips is exact, which is the property the
# black frame needed. Absolute positions inherit the trims' own precision and
# nothing but ordering reads them.
#
# WHY THIS RUNS ON EVERY READ AND `repair_audio_overlaps` DOES NOT. That one is
# a heuristic about intent and carries a version marker so a false positive
# costs one pass instead of every load. This is not a heuristic: a POSITIVE gap
# shorter than one frame cannot be authored, cannot be seen and cannot be
# rendered, so closing it removes nothing anybody put there. It is idempotent
# by construction — the second pass finds no gap to close — and a test pins
# that, because "fixes itself on open" and "drifts a little every open" look
# identical from one screenshot.
def film_fps(edit) -> float:
    """The sequence's frame rate. `NLE_FPS` unless the document names one."""
    fps = _f((edit or {}).get("fps"), 0.0)
    return fps if fps > 0 else float(NLE_FPS)


def frame_seconds(edit) -> float:
    """One frame, in seconds, at this sequence's rate."""
    return 1.0 / film_fps(edit)


def quantise_gap(gap: float, frame: float) -> float:
    """A lead gap, made legal: under one frame is NO gap. Larger is UNTOUCHED.

    Two things this deliberately is not.

    It is not `round(gap / frame)`. The reported 0.503-frame hole rounds UP to
    a whole frame of black, which is the same bug one frame louder.

    And it does not snap a LARGER gap to the grid. A gap of 43.2 frames is a
    black slug somebody placed on purpose, and moving it 8 ms to sit on a frame
    boundary would be rewriting a number the user chose to buy a property
    nothing reads — the cut either side of it is already exact once the gap is
    a fixed quantity. The rule is therefore the narrow one that closes the
    defect: a gap is either ZERO or at least one frame, and a gap somebody can
    see is never touched.
    """
    g = _f(gap)
    return 0.0 if g < frame - 1e-9 else round(g, 9)


def heal_subframe_gaps(edit: dict) -> list[dict]:
    """Close every sub-frame hole, rippling what follows. Mutates `edit`.

    Returns one row per hole closed, so a caller can say what it did. A LOCKED
    clip is an anchor and is never moved — the hole before it stays, because
    the alternative is honouring a pin by breaking it — and the ripple restarts
    from there so nothing downstream of an anchor inherits a shift the anchor
    did not take.

    An unlinked sound strip travels with its picture by the SAME delta, so a
    J-cut's offset is exactly what it was: `clip_audio_drift` is a difference
    of two constants and moving both by one number leaves it alone.
    """
    if not isinstance(edit, dict):
        return []
    clips = [c for c in (edit.get("clips") or []) if isinstance(c, dict)]
    clips.sort(key=lambda c: _f(c.get("film_start")))
    frame = frame_seconds(edit)
    closed: list[dict] = []
    cursor, delta = 0.0, 0.0
    for i, c in enumerate(clips):
        length = _f(c.get("film_end")) - _f(c.get("film_start"))
        if c.get("locked"):
            # An anchor does not move, so nothing after it is carrying a shift.
            cursor = max(cursor, _f(c.get("film_end")))
            delta = 0.0
            continue
        fs = _f(c.get("film_start")) - delta
        gap = fs - cursor
        if 1e-9 < gap < frame - 1e-9:
            closed.append({"after": i - 1, "film_start": round(cursor, 6),
                           "duration": round(gap, 6),
                           "frames": round(gap / frame, 4)})
            delta += gap
            fs = cursor
        if delta > 1e-9:
            c["film_start"] = round(fs, 6)
            c["film_end"] = round(fs + length, 6)
            aud = c.get("audio")
            if isinstance(aud, dict) and aud.get("film_start") is not None:
                aud["film_start"] = round(_f(aud.get("film_start")) - delta, 6)
        cursor = _f(c.get("film_end"))
    return closed


def heal_mix(edit: dict) -> bool:
    """Write the render's old hidden levels onto a document that predates the
    controls. Mutates `edit`; True when it wrote something.

    WHY A HEAL AND NOT A DEFAULT. Until today the ffmpeg graph held every
    `under` bed at 0.20 and ducked it against the dialogue, with no field
    anywhere saying so. Making the honest default (`1.0`, no duck) apply to
    those documents would silently re-mix films the owner has already listened
    to and approved. So the old behaviour is written DOWN, once, as the two
    values it always was: same render, and now it is on screen and reversible.

    ONLY `under` DOCUMENTS. `replace` never touched the bed — no gain term, no
    compressor — so stamping one would invent an attenuation that path never
    had. A `replace` timeline keeps the new default and renders byte-identical.
    """
    aud = edit.get("audio")
    if not isinstance(aud, dict) or not aud.get("path"):
        return False
    if isinstance(aud.get("mix"), dict):
        return False
    if str(aud.get("mode") or "replace").lower() != "under":
        return False
    aud["mix"] = {"bed_gain": MIX_LEGACY_BED_GAIN, "duck": True}
    return True


def _clips_copied(edit: dict) -> dict:
    """A shallow copy of `edit` whose CLIPS are copies too, strips included.

    `dict(edit)` copies the document and shares every clip in it, so a repair
    that writes `clip["audio"]` writes it into the caller's document — the
    exact defect `test_the_read_does_not_mutate_the_document_it_was_given`
    exists for, one level deeper than it was looking. `pending_backup` compares
    raw JSON on both sides, so a read that edits its argument makes the file
    and the snapshot differ over a repair neither of them contains.
    """
    out = dict(edit or {})
    out["clips"] = [dict(c) if isinstance(c, dict) else c
                    for c in (out.get("clips") or [])]
    for c in out["clips"]:
        if isinstance(c, dict) and isinstance(c.get("audio"), dict):
            c["audio"] = dict(c["audio"])
    return out


def migrate_edit(edit: dict) -> dict:
    """Bring an older document up to `EDIT_VERSION`. One way, on READ only.

    THE TRAP THIS EXISTS FOR: `validate_edit` refuses any version but the
    current one, and `save_edit` refuses to write anything `validate_edit`
    complained about. Bumping the version without a read-path upgrade would
    therefore not "refuse old builds" — it would refuse every timeline anybody
    already had, on their own machine, with their own build.

    Version 1 → 2 is a promotion and nothing else: v1 knew only video clips,
    and an absent `kind` already MEANS video (`clip_kind`). So the upgrade is
    the stamp on the document and no edit to a single clip — which is also why
    it cannot half-fail.

    A version from the FUTURE is left exactly as it arrived, so `validate_edit`
    gets to say the honest thing ("this build understands 2") instead of this
    function quietly pretending a document it has never seen is fine.
    """
    if not isinstance(edit, dict):
        return edit
    try:
        v = int(edit.get("version") or 0)
    except (TypeError, ValueError):
        return edit
    if 0 < v < EDIT_VERSION:
        edit = dict(edit)
        edit["version"] = EDIT_VERSION
        edit["migrated_from"] = v
    # THE DUPLICATED-STRIP ARTIFACT IS A ONE-TIME MIGRATION. It used to run on
    # every read, which made it a rival author: a document the user owns,
    # rewritten by a heuristic, every time it was opened. The marker is what
    # makes it a migration — a board that has been looked at once is never
    # looked at again, so a false positive can cost at most one pass instead
    # of every load for the life of the film.
    #
    # AND IT COPIES THE CLIPS FIRST. `repair_audio_overlaps` writes
    # `clip["audio"]` on both halves of a duplicated strip, and `dict(edit)`
    # shares every clip with the caller — so this repair used to reach back
    # into the document it was handed, which is the same "the read edited my
    # file" defect the mix stamp below already guards against.
    if edit.get("audio_repair") != AUDIO_REPAIR_VERSION:
        edit = _clips_copied(edit)
        if repair_audio_overlaps(edit):
            edit["repaired_audio_overlaps"] = True
        edit["audio_repair"] = AUDIO_REPAIR_VERSION
    # THE MIX STAMP, and it is a migration for the same reason: a document the
    # user owns must not be rewritten by a heuristic on every read. The marker
    # is what makes it one — see `MIX_REPAIR_VERSION` for what re-running it
    # would do to somebody who deliberately turned the duck off.
    #
    # COPY BEFORE WRITING. `pending_backup` compares raw JSON on both sides,
    # so a read that edits its argument in place makes the file and the
    # snapshot differ over a repair neither of them contains — which resurrects
    # the permanent interrogation `docs/EDITOR_SAVE_MODEL.md` exists to kill.
    # `test_the_read_does_not_mutate_the_document_it_was_given` is the gate.
    if edit.get("mix_repair") != MIX_REPAIR_VERSION:
        edit = dict(edit)
        if isinstance(edit.get("audio"), dict):
            edit["audio"] = dict(edit["audio"])
        if heal_mix(edit):
            edit["repaired_mix"] = True
        edit["mix_repair"] = MIX_REPAIR_VERSION
    # A HOLE SHORTER THAN A FRAME IS NOT A HOLE. See the block above
    # `film_fps`: unlike the audio repair, this carries no marker, because it
    # is not a heuristic about intent and running it twice finds nothing to do.
    edit = _clips_copied(edit)
    healed = heal_subframe_gaps(edit)
    # SET OR CLEARED, never left behind. A record of what THIS read closed that
    # survived into a read which closed nothing would report the same repair
    # forever — the panel would keep announcing a fix it had already made.
    if healed:
        edit["healed_subframe_gaps"] = healed
    else:
        edit.pop("healed_subframe_gaps", None)
    return edit


def load_edit(board_dir) -> dict | None:
    """The edit on disk, or None. A corrupt file raises rather than lying.

    Migration happens HERE, before anything else sees the document, because
    every other reader (the payload, the renderer, the validator on the way
    back in) would otherwise each need to know what version 1 looked like.
    """
    p = edit_path(board_dir)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EditError(f"edit.json is corrupt: {exc}") from exc
    if not isinstance(data, dict):
        raise EditError("edit.json is not an object")
    return migrate_edit(data)


def edit_to_cuts(edit: dict) -> list[dict]:
    """The film order, as the plan shape `_sb_assemble_film` already takes.

    One entry PER TIMELINE CLIP, in film order, including repeats of the same
    source — which is exactly the case the board could never express and the
    reason this file exists.
    """
    out = []
    by_id: dict[str, dict] = {}
    for c in (edit or {}).get("clips") or []:
        if not isinstance(c, dict):
            continue
        kind = clip_kind(c)
        if kind != "slug" and not c.get("path"):
            continue
        fs, fe = _f(c.get("film_start")), _f(c.get("film_end"))
        if kind == "video":
            start, end = _f(c.get("start")), _f(c.get("end"))
        else:
            # A still and a slug are their slot. `normalise_edit` already wrote
            # this pair; deriving it again here means a hand-built edit that
            # never went through normalise still renders the right length.
            start, end = 0.0, max(0.0, fe - fs)
        if end <= start:
            continue
        entry = {"path": (str(c["path"]) if c.get("path") else None),
                 "start": round(start, 6), "end": round(end, 6),
                 "film_start": round(fs, 6)}
        # ONLY WHEN THEY SAY SOMETHING. `_sb_cut_index` copies these through to
        # the filtergraph, and a plan that carried `kind: "video"` and an empty
        # `adjust` on every entry would change the assembler's output for every
        # film ever exported, to say nothing new.
        if kind != "video":
            entry["kind"] = kind
        b = clip_brightness(c)
        if abs(b) >= 1e-9:
            entry["adjust"] = {"brightness": round(b, 6)}
        if kind != "slug" and not clip_frame_is_neutral(c):
            entry["frame"] = clip_frame(c)
        if kind == "video":
            w = clip_audio(c)
            # Only a SPLIT clip says anything here, so a plan of ordinary cuts
            # produces the identical list it always did and the assembler below
            # builds the identical graph. A coupled pair is split too — its
            # offset is the J-cut, and an assembler that ignored it would
            # render the sound back under the picture.
            if w["split"]:
                entry["audio"] = {"start": w["start"], "end": w["end"],
                                  "film_start": w["film_start"]}
            # Only when it says something, so a film with nothing muted builds
            # the identical graph it always did.
            if clip_muted(c):
                entry["mute"] = True
        # EFFECTS RIDE ON EVERY KIND — a still and a slug fade like anything
        # else, and all three outputs read the same entry.
        e = clip_effects(c)
        fx = {k: e[k] for k in ("fade_in", "fade_out") if e[k] > 1e-9}
        if fx:
            entry["fx"] = fx
        if kind == "video":
            w = clip_audio(c)
            # ON THE PLAYED CLOCK — see `audio_gain_points`.
            gain = audio_gain_points(c, w["len"])
            if gain:
                entry["gain"] = gain
            # Only when it says something, like every other field here.
            sp = clip_speed(c)
            if abs(sp - 1.0) > 1e-9:
                entry["speed"] = sp
        out.append(entry)
        if c.get("id"):
            by_id[str(c["id"])] = entry
    out.sort(key=lambda e: e["film_start"])
    # THE TRANSITIONS RIDE ON THE ENTRIES THEY JOIN: `transition` on the
    # outgoing entry, `tx_in` (the extra head, in seconds) on the incoming.
    # Only a transition that resolved without a problem is stamped — one the
    # validator refused cannot reach the render through a stale document, and
    # `transition_problems` is what the render asks first.
    for t in resolve_transitions(edit):
        if t.get("problem"):
            continue
        a, b = by_id.get(t["after_clip"]), by_id.get(t["before_clip"])
        if a is None or b is None:
            continue
        ia, ib = out.index(a), out.index(b)
        if ib != ia + 1:
            continue
        a["transition"] = {"kind": t["kind"], "duration": t["duration"]}
        b["tx_in"] = t["half"]
    return out


def edit_duration(edit: dict) -> float:
    """Where the last clip ends on the film — INCLUDING any gaps before it."""
    ends = [_f(c.get("film_end")) for c in (edit or {}).get("clips") or []
            if isinstance(c, dict)]
    return round(max(ends), 6) if ends else 0.0


def music_window(audio, *, duration: float | None = None) -> dict:
    """What the soundtrack actually plays: `{start, end, delay, film_start}`.

    THE SOUNDTRACK IS AN OBJECT ON THE TIMELINE, not a global offset, and this
    is the one place that turns its three fields into the three numbers ffmpeg
    (and every exporter) needs. The owner's words: "features similar to what
    you did with the clips" — drag it, trim both ends, and it stays where it
    was put.

    The fields, and why they mean what they mean:

    * `offset` — the second of the TRACK that plays at film time 0. It has
      meant exactly that since before the editor existed, so every document on
      disk keeps working. What is new is that it may be NEGATIVE: the track
      begins `-offset` seconds INTO the film, with silence before it. That is
      the direction the old clamp made unreachable, and it is half of "back
      and forth however you want".
    * `trim_start` / `trim_end` — the in/out points INSIDE the track, in track
      seconds. ABSENT MEANS UNTRIMMED, which is why an edit.json written before
      today is still a valid one: no field, no trim, same graph.

    Returned:

    * `start` — where playback begins in the track: `max(trim_start, offset)`.
      A head trim and a positive offset are the same gesture from two
      directions, and whichever cuts more wins.
    * `end` — where it stops in the track, or None for "play to the end".
    * `delay`/`film_start` — the film second the music starts at, which is
      `start - offset`. Trimming the left edge must NOT slide the rest of the
      track earlier (that is a ripple, and music does not ripple), so the
      seconds a head trim removes come back as silence in front.
    """
    a = audio if isinstance(audio, dict) else {}
    off = _f(a.get("offset"))
    dur = _f(duration if duration is not None else a.get("duration"))
    ts = max(0.0, _f(a.get("trim_start")))
    te = a.get("trim_end")
    tail = _f(te) if isinstance(te, (int, float)) and not isinstance(te, bool) else None
    if tail is not None:
        if dur > 0:
            # A trim at (or past) the end of the track is not a trim — and
            # saying so here is what keeps the filtergraph byte-identical for
            # a handle dragged all the way back out. `normalise_edit` drops
            # the field for the same reason.
            tail = None if tail >= dur - 1e-6 else min(tail, dur)
        if tail is not None and tail <= ts:
            tail = None
    start = max(0.0, ts, off)
    if tail is not None and tail <= start:
        # A window the trims closed entirely. Nothing to play, and the honest
        # shape for that is "no music", not a zero-length atrim ffmpeg refuses.
        tail = None
        start = max(0.0, off)
    delay = max(0.0, start - off)
    return {"start": round(start, 6),
            "end": (round(tail, 6) if tail is not None else None),
            "delay": round(delay, 6),
            "film_start": round(delay, 6)}


def edit_gaps(edit: dict, *, tolerance: float | None = None) -> list[dict]:
    """Holes in the timeline: `[{"film_start", "film_end", "duration",
    "after"}]`.

    THE TOLERANCE IS HALF A FRAME AT THIS SEQUENCE'S RATE, and it used to be
    the constant `TOUCH_TOLERANCE` — which is 1/48, i.e. half a frame at 24 fps
    and the wrong number at any other rate. It also made this function the
    reason the header read "1 hole · 0.02s" over a film with THREE holes in it:
    two of them were under half a frame, so this list did not contain them and
    nothing else was looking. That class is closed at the source — `migrate_edit`
    heals a sub-frame hole out of existence on read — so what is left here is a
    threshold that means "shorter than half a frame is float noise", which is
    the only thing it can honestly mean once a hole is a whole number of frames.

    `after` is the index of the clip the hole follows, or -1 for a hole at the
    head of the film. This is the list the "generate into a gap" control is
    built from, and it is also the list the renderer has to disclose: the
    existing assembler CONCATENATES, so a gap closes and everything after it
    slides earlier. Reporting it is the difference between a known limitation
    and a film that mysteriously fell off the beat.
    """
    if tolerance is None:
        tolerance = frame_seconds(edit) / 2.0
    spans = sorted(
        ((_f(c.get("film_start")), _f(c.get("film_end")), i)
         for i, c in enumerate((edit or {}).get("clips") or [])
         if isinstance(c, dict) and _f(c.get("film_end")) > _f(c.get("film_start"))),
        key=lambda s: s[0])
    gaps: list[dict] = []
    cursor, after = 0.0, -1
    for s, e, i in spans:
        if s - cursor > tolerance:
            gaps.append({"film_start": round(cursor, 6),
                         "film_end": round(s, 6),
                         "duration": round(s - cursor, 6), "after": after})
        cursor = max(cursor, e)
        after = i
    return gaps


def edit_sync_flags(edit: dict, *,
                    tolerance: float = TOUCH_TOLERANCE) -> list[dict]:
    """Every unlinked pair whose sound no longer lines up with its picture.

    The sibling of `edit_gaps`, and information for exactly the same reason: a
    split edit IS a deliberate drift, so this can never be an error. What it
    can be is VISIBLE — the owner unlinked a clip, moved the picture, and had
    no way to see how far the two had come apart or to put them back:

        "instead of allowing me to remove or move what video is visible while
        leaving the sound intact and then rematching it, it is actually getting
        the audio out of sync."

    `resync_to` is the film second the strip goes back to; the panel prints
    `drift` on both halves and spends one click on the difference.
    """
    out: list[dict] = []
    for i, c in enumerate((edit or {}).get("clips") or []):
        if not isinstance(c, dict) or clip_kind(c) != "video":
            continue
        w = clip_audio(c)
        # A COUPLED PAIR IS NOT A DRIFT. Its offset is the relationship the
        # user froze, and the two travel together from then on — flagging it
        # would put a permanent warning on every J-cut in the film.
        if not w["split"] or w["coupled"]:
            continue
        drift = clip_audio_drift(c)
        if abs(drift) <= tolerance:
            continue
        out.append({"id": str(c.get("id") or ""), "where": i,
                    "drift": drift, "film_start": w["film_start"],
                    "resync_to": clip_audio_resync(c)})
    return out


# ===========================================================================
# PART D — the film, as a project somebody else's editor can open
# ===========================================================================
# WHY A FOLDER AND NOT A FILE. An XML that names `/Users/salo/mlx_outputs/…`
# is a project that works on exactly one machine until the first time anything
# moves, and then it is a timeline of red offline clips. Premiere and Resolve
# both relink by NAME within the project's own directory before they give up,
# so a folder whose XML points at a `media/` sitting beside it relinks on drop
# — on this machine, on the colourist's, on a drive.
#
# WHY HARDLINKS. The same bytes, a second name, no copy: a 30-shot film exports
# in milliseconds and costs nothing on disk. `os.link` fails across
# filesystems (EXDEV) and on exotic mounts, and the fallback is a real copy,
# because an export that refuses to happen is worse than an export that costs
# disk.
#
# WHY FCP7 XML. It is the ONE interchange both Premiere Pro and DaVinci Resolve
# import. AAF is Avid-first and lossy through Resolve; EDL cannot carry a still
# or a per-clip effect; Premiere's own .prproj is undocumented binary. Every
# other choice is two exporters.
#
# WHY A SEPARATE SCRIPT FOR AFTER EFFECTS. AE has no timeline-XML import at
# all — never has. The seamless path there is an ExtendScript the user runs
# from File > Scripts > Run Script File, which is why one is written next to
# the XML rather than a second XML AE would refuse.
NLE_FPS = 24

# The image extensions the still branch recognises. Deliberately short: this
# decides whether a clipitem is written with a source duration or as a held
# frame, and a wrong guess is a clip of the wrong length in somebody's NLE.
_STILL_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


def _xml_text(s) -> str:
    """Escape for XML character data AND attribute values."""
    return (str(s if s is not None else "")
            .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;").replace("'", "&apos;"))


def _jsx_string(s) -> str:
    """A path as an ExtendScript double-quoted literal.

    Windows paths and macOS volume names both put characters in here that end
    the string early: a backslash escapes the quote after it, a quote closes
    the literal, and a newline in a filename (legal on macOS) makes the script
    a syntax error. All three are escaped, and the result is wrapped in its own
    quotes so callers cannot forget to.
    """
    out = (str(s if s is not None else "")
           .replace("\\", "\\\\").replace('"', '\\"')
           .replace("\r", "\\r").replace("\n", "\\n"))
    return '"' + out + '"'


def _pathurl(p) -> str:
    """`file://localhost/…` with everything URL-unsafe percent-encoded.

    A space in a filename is the common case (a board titled "Gut Health") and
    an un-encoded one makes the pathurl unparseable to Premiere — the clip
    imports offline with no error anybody can act on. `quote` keeps `/` so the
    path survives as a path.
    """
    from urllib.parse import quote                                # noqa: PLC0415
    return "file://localhost" + quote(str(Path(str(p)).as_posix()), safe="/")


def _frames(seconds: float, fps: int = NLE_FPS) -> int:
    """Seconds → whole frames, rounded. An NLE has no other unit."""
    return int(round(_f(seconds) * float(fps)))


def _nle_media_name(src: Path, taken: dict) -> str:
    """A unique name inside `media/`, stable for the same source.

    Two shots called `S01_open.mp4` from two different boards would otherwise
    become one file and one of them would silently play the other. The
    fingerprint suffix only appears when there IS a collision, so the common
    export keeps readable names.
    """
    key = str(src)
    if key in taken:
        return taken[key]
    name = src.name
    if name in set(taken.values()):
        h = hashlib.sha1(key.encode("utf-8", "replace")).hexdigest()[:6]
        name = f"{src.stem}_{h}{src.suffix}"
    taken[key] = name
    return name


def _link_or_copy(src: Path, dest: Path, *, link=None) -> str:
    """Hardlink, or copy when the filesystem says no. Returns "link"|"copy".

    `link` is injectable so the fallback branch is testable without a second
    filesystem — the branch that only fires on a cross-device export is
    exactly the branch nobody would otherwise ever run.
    """
    import shutil                                                 # noqa: PLC0415
    linker = link or os.link
    if dest.exists():
        try:
            dest.unlink()
        except OSError:
            pass
    try:
        linker(str(src), str(dest))
        return "link"
    except (OSError, NotImplementedError, AttributeError):
        shutil.copy2(str(src), str(dest))
        return "copy"


def _nle_segments(clips, *, probe=None) -> list[dict]:
    """The export's own view of the timeline: one row per clip, in film order.

    `probe` answers {"w","h","duration"} for a file and is injected rather than
    imported, for the same reason `compute_peaks` takes a decoder: this module
    runs no subprocess of its own, and the panel already owns ffprobe.
    """
    rows: list[dict] = []
    for c in clips or []:
        if not isinstance(c, dict):
            continue
        kind = clip_kind(c)
        fs, fe = _f(c.get("film_start")), _f(c.get("film_end"))
        if fe <= fs:
            continue
        path = str(c.get("path") or "") if kind != "slug" else ""
        if kind != "slug" and not path:
            continue
        if kind == "video":
            start, end = _f(c.get("start")), _f(c.get("end"))
        else:
            start, end = 0.0, fe - fs
        info = {}
        if path and probe:
            try:
                info = probe(path) or {}
            except Exception:                                      # noqa: BLE001
                info = {}
        rows.append({
            "kind": kind,
            "path": path,
            "title": str(c.get("title") or (Path(path).stem if path else "black")),
            "start": start, "end": end,
            "film_start": fs, "film_end": fe,
            "brightness": clip_brightness(c),
            "frame": clip_frame(c),
            "fx": clip_effects(c),
            "w": int(info.get("w") or 0), "h": int(info.get("h") or 0),
            "has_audio": bool(info.get("has_audio")),
            "gain": audio_gain_points(c, clip_audio(c)["len"])
            if clip_kind(c) == "video" else [],
            # Carried for the far side to read. The XML's in/out (source) and
            # start/end (timeline) already disagree by exactly this ratio,
            # which is how an FCP7 importer infers a speed change.
            "speed": clip_speed(c),
            # Carried, never applied here: the export DISABLES the audio
            # clipitem rather than dropping it, so the editor on the far end
            # can see the decision and undo it.
            "muted": clip_muted(c),
            "source_duration": _f(info.get("duration"))
            or (fe - fs if kind != "video" else 0.0),
        })
    rows.sort(key=lambda r: r["film_start"])
    return rows


def _fcp7_levels(curve, fps: int = NLE_FPS) -> str:
    """The sound's envelope as LEVEL keyframes, which is what an NLE calls it.

    The same discipline the opacity keyframes follow: the decision travels, not
    a pre-mixed result. An editor on the far side gets a rubber-band they can
    drag rather than audio that has already been ducked into the file.

    FCP7's `audiolevels` parameter is linear 0..1 with 1 as unity, which is the
    same number `audio_gain_points` produces — so no conversion, and nothing to
    get wrong between the render and the export.
    """
    pts = list(curve or [])
    if len(pts) < 2:
        return ""
    kf = "".join(
        f"<keyframe><when>{_frames(t, fps)}</when><value>{g:.4f}</value></keyframe>"
        for t, g in pts)
    return ("<filter><effect><name>Audio Levels</name>"
            "<effectid>audiolevels</effectid>"
            "<effectcategory>audiolevels</effectcategory>"
            "<effecttype>audiolevels</effecttype><mediatype>audio</mediatype>"
            "<parameter><parameterid>level</parameterid><name>Level</name>"
            "<valuemin>0</valuemin><valuemax>3.98108</valuemax>"
            f"{kf}</parameter></effect></filter>")


def _fcp7_opacity(seg: dict, fps: int = NLE_FPS) -> str:
    """Fades as OPACITY KEYFRAMES, which is what an NLE calls this.

    NOBODY RECEIVES BAKED PIXELS. The alternative — pre-rendering the ramp
    into the media — would hand the next room a clip they cannot un-fade, and
    the export exists so the decision travels, not just its result. Opacity is
    a standard FCP7 filter that Premiere and Resolve both import, and the
    keyframes are draggable on arrival.

    Four keyframes, or none: 0 at the head, 100 once the in-fade lands, 100
    where the out-fade starts, 0 at the tail. `clip_effects` has already
    clamped them so the middle pair cannot cross.
    """
    fx = seg.get("fx") or {}
    fin = _f(fx.get("fade_in"))
    fout = _f(fx.get("fade_out"))
    if fin <= 1e-9 and fout <= 1e-9:
        return ""
    span = max(1, _frames(_f(seg.get("film_end")) - _f(seg.get("film_start")), fps))
    keys = []
    if fin > 1e-9:
        keys.append((0, 0.0))
        keys.append((min(span, _frames(fin, fps)), 100.0))
    else:
        keys.append((0, 100.0))
    if fout > 1e-9:
        keys.append((max(0, span - _frames(fout, fps)), 100.0))
        keys.append((span, 0.0))
    else:
        keys.append((span, 100.0))
    kf = "".join(f"<keyframe><when>{w}</when><value>{v:.2f}</value></keyframe>"
                 for w, v in keys)
    return ("<filter><effect><name>Opacity</name><effectid>opacity</effectid>"
            "<effectcategory>motion</effectcategory>"
            "<effecttype>motion</effecttype><mediatype>video</mediatype>"
            "<parameter><parameterid>opacity</parameterid>"
            "<name>opacity</name><valuemin>0</valuemin><valuemax>100</valuemax>"
            f"{kf}</parameter></effect></filter>")


def _fcp7_motion(seg: dict) -> str:
    """The reframe as Basic Motion — scale and centre — or "" when neutral.

    The same rule the opacity keyframes follow: the DECISION travels, not
    baked pixels. Scale is percent; the centre is the layer's offset from the
    frame's centre in frame widths/heights, which is `zoom * (0.5 - x)`: the
    point of the source that should sit at the middle, moved there.
    """
    f = seg.get("frame") or {}
    z = _f(f.get("zoom"), 1.0)
    if abs(z - 1.0) < 1e-9:
        return ""
    cx = z * (0.5 - _f(f.get("x"), 0.5))
    cy = z * (0.5 - _f(f.get("y"), 0.5))
    return ("<filter><effect><name>Basic Motion</name><effectid>basic</effectid>"
            "<effectcategory>motion</effectcategory>"
            "<effecttype>motion</effecttype><mediatype>video</mediatype>"
            "<parameter><parameterid>scale</parameterid><name>Scale</name>"
            "<valuemin>0</valuemin><valuemax>1000</valuemax>"
            f"<value>{z * 100.0:.2f}</value></parameter>"
            "<parameter><parameterid>center</parameterid><name>Center</name>"
            f"<value><horiz>{cx:.4f}</horiz><vert>{cy:.4f}</vert></value></parameter>"
            "</effect></filter>")


def _fcp7_rate(fps: int = NLE_FPS) -> str:
    # ntsc FALSE is load-bearing: at ntsc TRUE an NLE reads timebase 24 as
    # 23.976 and every cut past the first drifts one frame per 1000.
    return f"<rate><timebase>{int(fps)}</timebase><ntsc>FALSE</ntsc></rate>"


def _fcp7_file(fid: str, seg: dict, media_abs: Path, *, fps: int,
               declared: set) -> str:
    """A `<file>` element — full the FIRST time, an id reference after.

    FCP7 XML's rule, and both importers rely on it: a file is described once
    and every later clipitem points at the same id. Re-describing it makes
    Premiere import the same clip several times as several master items.
    """
    if fid in declared:
        return f'<file id="{_xml_text(fid)}"/>'
    declared.add(fid)
    name = Path(media_abs).name
    body = [f'<file id="{_xml_text(fid)}">',
            f"<name>{_xml_text(name)}</name>",
            f"<pathurl>{_xml_text(_pathurl(media_abs))}</pathurl>",
            _fcp7_rate(fps)]
    if seg["kind"] == "video" and seg["source_duration"] > 0:
        body.append(f"<duration>{_frames(seg['source_duration'], fps)}</duration>")
    body.append("<media><video><samplecharacteristics>")
    body.append(_fcp7_rate(fps))
    if seg["w"] and seg["h"]:
        body.append(f"<width>{int(seg['w'])}</width>"
                    f"<height>{int(seg['h'])}</height>")
    body.append("</samplecharacteristics></video>")
    if seg["has_audio"]:
        body.append("<audio><samplecharacteristics><depth>16</depth>"
                    "<samplerate>48000</samplerate></samplecharacteristics>"
                    "<channelcount>2</channelcount></audio>")
    body.append("</media></file>")
    return "".join(body)


def fcp7_xml(segments, *, name: str, media: dict, width: int, height: int,
             base, fps: int = NLE_FPS, audio: dict | None = None,
             overlays: list | None = None) -> str:
    """The sequence, as the one XML both Premiere and Resolve import.

    SLUGS ARE GAPS. A slug could be written as a `<generatoritem>` with the
    Slug/Color effect, and the effect ids for that differ between Premiere and
    Resolve — so the "one XML for both" promise would quietly become "one XML
    for one of them, and a red error in the other". A gap on a video track
    reads as black in every NLE ever made, costs nothing, and is honest. The AE
    script, which has no such ambiguity, gets real black solids.

    THE AUDIO IS STEMS, NOT THE MIX. The clips' own sound goes on A1 and the
    soundtrack goes on A2, unducked. The under-mix the renderer builds —
    sidechain compression against the dialogue, then a tanh ceiling — has no
    representation in an NLE's timeline at all, so baking it in would hand an
    editor a bed they cannot unmix and cannot re-balance. Stems are what the
    next room actually wants.
    """
    declared: set = set()
    # ABSOLUTE pathurls, pointing at the copies in this project's OWN media/.
    # A relative pathurl is not something either importer accepts, and an
    # absolute one into the original gallery is a project that dies the first
    # time anything moves. Absolute-into-media/ is the pair that works: it
    # opens instantly here, and when the folder is handed over, Premiere and
    # Resolve both fall back to matching by NAME inside the project's own
    # directory — which is precisely where the file is.
    mdir = Path(str(base)) / "media"
    file_ids: dict = {}
    total = max([_frames(s["film_end"], fps) for s in segments] or [0])
    v_items, a_items = [], []
    for i, seg in enumerate(segments):
        if seg["kind"] == "slug":
            continue                          # a gap on the track IS the slug
        abs_media = mdir / media[seg["path"]]
        # THE ID IS THE SOURCE'S, NOT THE SEGMENT'S. Keying it by position
        # gave the same clip used twice two different file ids, so it was
        # described twice and Premiere imported it as two master items — the
        # exact duplication the declare-once rule exists to prevent.
        fid = "file-" + str(file_ids.setdefault(seg["path"],
                                                len(file_ids) + 1))
        # in/out are the SOURCE window; start/end are the FILM slot. They are
        # different clocks and conflating them is the classic FCP7 XML bug —
        # the film plays, in the wrong order, at the wrong lengths.
        f_in, f_out = _frames(seg["start"], fps), _frames(seg["end"], fps)
        f_s, f_e = _frames(seg["film_start"], fps), _frames(seg["film_end"], fps)
        v_items.append(
            f'<clipitem id="clipitem-{i + 1}">'
            f"<name>{_xml_text(seg['title'])}</name>"
            f"<enabled>TRUE</enabled>"
            f"<duration>{max(1, f_out - f_in)}</duration>"
            f"{_fcp7_rate(fps)}"
            f"<start>{f_s}</start><end>{f_e}</end>"
            f"<in>{f_in}</in><out>{f_out}</out>"
            f"{_fcp7_file(fid, seg, abs_media, fps=fps, declared=declared)}"
            f"<compositemode>normal</compositemode>"
            f"{_fcp7_motion(seg)}"
            f"{_fcp7_opacity(seg, fps)}"
            f"</clipitem>")
        if seg["has_audio"]:
            # DISABLED, NOT DELETED. A muted clip whose audio clipitem was
            # simply omitted would arrive in Premiere as a shot that never had
            # sound, and the editor on the far end could not tell a decision
            # from a source without one. `<enabled>FALSE</enabled>` is the
            # timeline's own word for "this is here and it is off", and one
            # click puts it back.
            a_items.append(
                f'<clipitem id="clipitem-a{i + 1}">'
                f"<name>{_xml_text(seg['title'])}</name>"
                f"<enabled>{'FALSE' if seg.get('muted') else 'TRUE'}</enabled>"
                f"<duration>{max(1, f_out - f_in)}</duration>"
                f"{_fcp7_rate(fps)}"
                f"<start>{f_s}</start><end>{f_e}</end>"
                f"<in>{f_in}</in><out>{f_out}</out>"
                f'<file id="{_xml_text(fid)}"/>'
                f"<sourcetrack><mediatype>audio</mediatype>"
                f"<trackindex>1</trackindex></sourcetrack>"
                f"{_fcp7_levels(seg.get('gain'), fps)}"
                f"</clipitem>")
    # ---- V2: the overlay lane ------------------------------------------
    ov_items = []
    for j, o in enumerate(overlays or []):
        if not isinstance(o, dict):
            continue
        src = str(o.get("path") or "")
        if not src or src not in media:
            continue
        abs_media = mdir / media[src]
        fid = "file-" + str(file_ids.setdefault(src, len(file_ids) + 1))
        ofs = _frames(_f(o.get("film_start")), fps)
        ofe = _frames(_f(o.get("film_end")), fps)
        span = max(1, ofe - ofs)
        seg = {"kind": overlay_kind(o), "path": src,
               "title": str(o.get("title") or Path(src).stem),
               "start": _f(o.get("start")), "end": _f(o.get("end")) or (ofe - ofs) / float(fps),
               "film_start": _f(o.get("film_start")), "film_end": _f(o.get("film_end")),
               "w": 0, "h": 0, "has_audio": False,
               "source_duration": _f(o.get("duration")),
               "fx": clip_effects(o)}
        ov_items.append(
            f'<clipitem id="clipitem-ov{j + 1}">'
            f"<name>{_xml_text(seg['title'])}</name>"
            f"<enabled>TRUE</enabled>"
            f"<duration>{span}</duration>"
            f"{_fcp7_rate(fps)}"
            f"<start>{ofs}</start><end>{ofe}</end>"
            f"<in>{_frames(seg['start'], fps)}</in>"
            f"<out>{_frames(seg['start'], fps) + span}</out>"
            f"{_fcp7_file(fid, seg, abs_media, fps=fps, declared=declared)}"
            f"<compositemode>normal</compositemode>"
            f"{_fcp7_opacity(seg, fps)}"
            f"</clipitem>")
    ov_track = f"<track>{''.join(ov_items)}</track>" if ov_items else ""
    music_track = ""
    if audio and audio.get("path"):
        m_abs = mdir / media[str(audio["path"])]
        m_len = _frames(_f(audio.get("duration")) or (total / float(fps)), fps)
        # THE SAME WINDOW THE RENDER USES. An exported project whose music sits
        # where the trim handles did not put it is a project the editor has to
        # re-cut on arrival, which is the opposite of what the folder is for.
        win = music_window(audio)
        m_off = _frames(win["start"], fps)
        m_delay = _frames(win["delay"], fps)
        m_end = (_frames(win["end"], fps) if win["end"] is not None
                 else m_off + max(1, total - m_delay))
        music_track = (
            "<track>"
            f'<clipitem id="clipitem-music">'
            f"<name>{_xml_text(m_abs.name)}</name>"
            f"<enabled>TRUE</enabled><duration>{max(1, m_len)}</duration>"
            f"{_fcp7_rate(fps)}"
            f"<start>{m_delay}</start><end>{max(m_delay + 1, total)}</end>"
            f"<in>{m_off}</in><out>{max(m_off + 1, m_end)}</out>"
            f'<file id="file-music">'
            f"<name>{_xml_text(m_abs.name)}</name>"
            f"<pathurl>{_xml_text(_pathurl(m_abs))}</pathurl>"
            f"{_fcp7_rate(fps)}"
            f"<media><audio><channelcount>2</channelcount></audio></media>"
            f"</file>"
            f"<sourcetrack><mediatype>audio</mediatype>"
            f"<trackindex>1</trackindex></sourcetrack>"
            f"</clipitem></track>")
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<!DOCTYPE xmeml>\n"
        '<xmeml version="4">\n'
        '<sequence id="sequence-1">'
        f"<name>{_xml_text(name)}</name>"
        f"<duration>{total}</duration>"
        f"{_fcp7_rate(fps)}"
        "<media>"
        "<video>"
        "<format><samplecharacteristics>"
        f"{_fcp7_rate(fps)}"
        f"<width>{int(width)}</width><height>{int(height)}</height>"
        "<pixelaspectratio>square</pixelaspectratio>"
        "</samplecharacteristics></format>"
        # TWO VIDEO TRACKS NOW. V1 is the picture lane, which the validator
        # still refuses an overlap on. V2 is the OVERLAY lane, and it is the
        # reason a second track stopped describing a film this editor cannot
        # make: an overlay sitting on a clip is not an overlap, it is the
        # feature. Track order is the stacking order — later track, higher
        # layer — so V2 must come second.
        f"<track>{''.join(v_items)}</track>"
        f"{ov_track}"
        "</video>"
        "<audio>"
        f"<track>{''.join(a_items)}</track>"
        f"{music_track}"
        "</audio>"
        "</media>"
        "</sequence>\n"
        "</xmeml>\n")


def ae_jsx(segments, *, name: str, media: dict, width: int, height: int,
           fps: int = NLE_FPS, audio: dict | None = None,
           overlays: list | None = None) -> str:
    """An ExtendScript that BUILDS the comp, because AE cannot import a timeline.

    It locates its own folder (`File($.fileName).parent`) and imports from the
    `media/` beside it, so the same folder that relinks in Premiere works in AE
    after being moved or handed over.
    """
    lines = [
        "// Phosphene — " + str(name),
        "// After Effects has no timeline-XML import, so this script IS the",
        "// import: run it with File > Scripts > Run Script File…",
        "// It reads the media/ folder sitting next to itself, so move the",
        "// whole folder and it still works.",
        "(function () {",
        '  var here = File($.fileName).parent;',
        "  var proj = app.project || app.newProject();",
        "  app.beginUndoGroup(" + _jsx_string("Import " + str(name)) + ");",
        "  var comp = proj.items.addComp(%s, %d, %d, 1, %s, %d);"
        % (_jsx_string(name), int(width), int(height),
           f"{max([s['film_end'] for s in segments] or [0.0]):.6f}", int(fps)),
        "  var lay;",
        "  function bring(rel) {",
        "    var f = new File(here.fsName + rel);",
        "    if (!f.exists) { throw new Error('missing media: ' + f.fsName); }",
        "    return proj.importFile(new ImportOptions(f));",
        "  }",
        "  function bright(layer, v) {",
        "    // Brightness & Contrast. AE's Brightness is roughly [-150, 150]",
        "    // against a 0-255 pixel; ffmpeg's eq=brightness is an additive",
        "    // offset in [-1, 1] against a 0-1 pixel. 0.5 -> 75 is the same",
        "    // half-of-half. APPROXIMATE, not a match: AE and ffmpeg do not",
        "    // agree on gamma, and the render is the one that is exact.",
        "    var fx = layer.property('ADBE Effect Parade')",
        "               .addProperty('ADBE Brightness & Contrast 2');",
        "    fx.property(1).setValue(v);",
        "  }",
    ]
    for i, seg in enumerate(segments):
        ae_b = round(seg["brightness"] * 150.0, 3)
        if seg["kind"] == "slug":
            lines += [
                f"  // {i + 1}: black slug",
                "  lay = comp.layers.addSolid([0, 0, 0], %s, %d, %d, 1);"
                % (_jsx_string("black"), int(width), int(height)),
            ]
        else:
            rel = "/media/" + media[seg["path"]]
            lines += [
                f"  // {i + 1}: {seg['kind']}",
                "  lay = comp.layers.add(bring(%s));" % _jsx_string(rel),
            ]
            # startTime is where frame 0 of the SOURCE would sit, so a clip
            # trimmed from 2.0 s starts 2 s before its own in-point. Set it
            # BEFORE in/out or AE clamps the trim to the wrong window. A still
            # has no source window, so its frame 0 is its slot — and leaving
            # startTime at 0 would let AE's default two-second still duration
            # clamp any hold longer than that.
            lines.append("  lay.startTime = %.6f;"
                         % (seg["film_start"] - seg["start"]))
        lines += [
            "  lay.inPoint = %.6f;" % seg["film_start"],
            "  lay.outPoint = %.6f;" % seg["film_end"],
        ]
        if abs(seg["brightness"]) >= 1e-9:
            lines.append("  bright(lay, %s);" % ae_b)
        fr = seg.get("frame") or {}
        z = _f(fr.get("zoom"), 1.0)
        if abs(z - 1.0) >= 1e-9:
            # THE REFRAME AS SCALE + POSITION, the same arithmetic the FCP7
            # Basic Motion carries: the source point (x, y) is moved to the
            # comp's centre and the layer magnified around it.
            px = width / 2.0 + z * width * (0.5 - _f(fr.get("x"), 0.5))
            py = height / 2.0 + z * height * (0.5 - _f(fr.get("y"), 0.5))
            lines += [
                "  lay.property('ADBE Transform Group').property('ADBE Scale')"
                ".setValue([%.3f, %.3f]);" % (z * 100.0, z * 100.0),
                "  lay.property('ADBE Transform Group').property('ADBE Position')"
                ".setValue([%.3f, %.3f]);" % (px, py),
            ]
        fx = seg.get("fx") or {}
        f_in, f_out = _f(fx.get("fade_in")), _f(fx.get("fade_out"))
        if f_in > 1e-9 or f_out > 1e-9:
            # OPACITY KEYFRAMES, not a pre-rendered ramp: the editor on the
            # far side gets a fade they can drag, and AE's own timeline is
            # where a fade belongs.
            fs, fe = _f(seg["film_start"]), _f(seg["film_end"])
            pts = []
            if f_in > 1e-9:
                pts += [(fs, 0), (fs + f_in, 100)]
            else:
                pts += [(fs, 100)]
            if f_out > 1e-9:
                pts += [(fe - f_out, 100), (fe, 0)]
            else:
                pts += [(fe, 100)]
            lines.append("  op = lay.property('ADBE Transform Group')"
                         ".property('ADBE Opacity');")
            for when, val in pts:
                lines.append("  op.setValueAtTime(%.6f, %d);" % (when, val))
        gain = seg.get("gain") or []
        if len(gain) >= 2:
            # AE'S AUDIO LEVELS ARE dB, so the linear curve is converted here
            # and nowhere else — one conversion, at the one seam that needs it.
            # -96 dB is AE's own floor for silence; log(0) is not a number.
            lines.append("  au = lay.property('ADBE Audio Group')"
                         ".property('ADBE Audio Levels');")
            fs = _f(seg["film_start"])
            for t, g in gain:
                db = -96.0 if g <= 1e-6 else max(-96.0, 20.0 * math.log10(g))
                lines.append("  au.setValueAtTime(%.6f, [%.3f, %.3f]);"
                             % (fs + t, db, db))
        if seg.get("muted") and seg["kind"] != "slug":
            # The audio SWITCH, which is what AE's timeline calls the same
            # decision — the layer keeps its sound and does not play it.
            lines.append("  lay.audioEnabled = false;")
    # ---- THE OVERLAY LANE, on top ---------------------------------------
    # `comp.layers.add` inserts at index 1, so whatever is added LAST sits
    # highest. Adding the cards after every picture layer is the whole of the
    # stacking order — no index arithmetic, and it cannot drift.
    for j, o in enumerate(overlays or []):
        if not isinstance(o, dict):
            continue
        src = str(o.get("path") or "")
        if not src or src not in media:
            continue
        rel = "/media/" + media[src]
        ofs, ofe = _f(o.get("film_start")), _f(o.get("film_end"))
        lines += [
            f"  // overlay {j + 1}: {overlay_kind(o)}",
            "  lay = comp.layers.add(bring(%s));" % _jsx_string(rel),
            "  lay.startTime = %.6f;" % ofs,
            "  lay.inPoint = %.6f;" % ofs,
            "  lay.outPoint = %.6f;" % ofe,
        ]
        e = clip_effects(o)
        if e["fade_in"] > 1e-9 or e["fade_out"] > 1e-9:
            pts = []
            if e["fade_in"] > 1e-9:
                pts += [(ofs, 0), (ofs + e["fade_in"], 100)]
            else:
                pts += [(ofs, 100)]
            if e["fade_out"] > 1e-9:
                pts += [(ofe - e["fade_out"], 100), (ofe, 0)]
            else:
                pts += [(ofe, 100)]
            lines.append("  op = lay.property('ADBE Transform Group')"
                         ".property('ADBE Opacity');")
            for when, val in pts:
                lines.append("  op.setValueAtTime(%.6f, %d);" % (when, val))
    if audio and audio.get("path"):
        rel = "/media/" + media[str(audio["path"])]
        win = music_window(audio)
        lines += [
            "  // the soundtrack, as its own layer and NOT the ducked mix —",
            "  // the render's under-mix is not representable here, and stems",
            "  // are what an editor wants anyway.",
            "  lay = comp.layers.add(bring(%s));" % _jsx_string(rel),
            # startTime is where the track's OWN second zero would sit on the
            # comp: the film second the music starts at, less how far into the
            # track that second is. `+ 0.0` because -0.0 formats as
            # "-0.000000", which is valid ExtendScript and reads like a bug to
            # whoever opens the file.
            "  lay.startTime = %.6f;"
            % (win["delay"] - win["start"] + 0.0),
        ]
        if win["delay"] > 1e-6 or win["end"] is not None:
            # Only when the music was actually moved or trimmed — an untouched
            # soundtrack keeps the script it has always produced.
            end_film = (win["delay"] + (win["end"] - win["start"])
                        if win["end"] is not None
                        else max([s["film_end"] for s in segments] or [0.0]))
            lines += [
                "  lay.inPoint = %.6f;" % win["delay"],
                "  lay.outPoint = %.6f;" % max(win["delay"] + 1e-3, end_film),
            ]
    lines += [
        "  comp.openInViewer();",
        "  app.endUndoGroup();",
        "})();",
        "",
    ]
    return "\n".join(lines)


def export_nle(clips, dest_dir, *, name: str, fps: int = NLE_FPS,
               audio: dict | None = None, probe=None, link=None,
               width: int = 0, height: int = 0,
               overlays: list | None = None) -> dict:
    """Write `<name>_project/` — one XML, one AE script, and the media beside them.

    Returns {"ok", "dir", "xml", "jsx", "clips", "linked", "copied",
             "missing", "width", "height", "duration"}.
    """
    import shutil                                                 # noqa: PLC0415, F401
    segs = _nle_segments(clips, probe=probe)
    if not segs:
        raise EditError("there is nothing on the timeline to export")
    root = Path(str(dest_dir)) / f"{_slug(name, 60)}_project"
    media_dir = root / "media"
    media_dir.mkdir(parents=True, exist_ok=True)

    if not width or not height:
        width = max([s["w"] for s in segs] or [0]) or 1024
        height = max([s["h"] for s in segs] or [0]) or 576
    width += width % 2
    height += height % 2

    taken: dict = {}
    media: dict = {}
    linked = copied = 0
    missing: list[str] = []
    ovs = [o for o in (overlays or []) if isinstance(o, dict)
           and str(o.get("path") or "")]
    wanted = [s["path"] for s in segs if s["path"]]
    # The cards travel with the project like every other source, so the folder
    # opens on another machine with the overlay intact.
    wanted += [str(o["path"]) for o in ovs]
    if audio and audio.get("path"):
        wanted.append(str(audio["path"]))
    for src_str in wanted:
        if src_str in media:
            continue
        src = Path(src_str)
        if not src.is_file():
            missing.append(src_str)
            continue
        rel = _nle_media_name(src, taken)
        media[src_str] = rel
        how = _link_or_copy(src, media_dir / rel, link=link)
        linked += (how == "link")
        copied += (how == "copy")
    # A clip whose file has gone cannot be written into the XML — a pathurl to
    # nothing is an offline clip the editor has to hunt for, and silence about
    # it is worse than the gap.
    segs = [s for s in segs if not s["path"] or s["path"] in media]
    a_arg = audio if (audio and str(audio.get("path") or "") in media) else None

    xml_path = root / f"{_slug(name, 60)}.xml"
    jsx_path = root / f"{_slug(name, 60)}_ae.jsx"
    xml_path.write_text(
        fcp7_xml(segs, name=name, media=media, width=width, height=height,
                 overlays=ovs,
                 base=root, fps=fps, audio=a_arg), encoding="utf-8")
    jsx_path.write_text(
        ae_jsx(segs, name=name, media=media, width=width, height=height,
               overlays=ovs,
               fps=fps, audio=a_arg), encoding="utf-8")
    return {
        "ok": True, "dir": str(root), "xml": str(xml_path),
        "jsx": str(jsx_path), "clips": len(segs),
        "linked": linked, "copied": copied, "missing": missing,
        "media": sorted(media.values()),
        "width": width, "height": height,
        "duration": round(max([s["film_end"] for s in segs] or [0.0]), 3),
    }


# ===========================================================================
# PART C3 — DRAFTS: the user's own saves, and a backup that never overwrites one
# ===========================================================================
# THE OWNER, AFTER LOSING TWENTY MINUTES TO AN AUTOSAVE HE COULD NOT SEE:
# "the auto-saving function — I think it is not good. It's better that the
# user has the power to manage this feature. You can keep a backup in case
# it's needed... but he should have control over the saving, and only he
# should have that control. Additionally, renaming and access to all the
# drafts he makes are essential. He should be able to start a new draft and
# work on it, as well as copy and paste drafts."
#
# So: SAVE IS THE USER'S VERB. Nothing else writes the document he named.
#
# WHERE THINGS LIVE, AND WHY EDIT.JSON DID NOT MOVE. The ACTIVE draft is
# `edit.json` — the same path the renderer, the exporter, the payload and
# every board on disk already read. Only the INACTIVE drafts sit in
# `drafts/<slug>.json`. That is one copy of any draft, ever: nothing can
# diverge from itself, and a board written before drafts existed becomes a
# board with exactly one draft the first time anybody looks, with no
# migration pass and nothing to half-finish.
#
# THE BACKUP IS A DIFFERENT FILE ON PURPOSE. `history/backup-<slug>.json`
# never touches edit.json and never bumps a revision, so a crash costs the
# user nothing and a working session costs him no surprises. On open, a
# backup newer than the saved draft is an OFFER — see `pending_backup` — and
# applying it is a click, never a side effect.
DRAFT_INDEX_VERSION = 1
DRAFT_NAME_MAX = 60
_BACKUP_PREFIX = "backup-"


def drafts_dir(board_dir) -> Path:
    return Path(str(board_dir)) / "drafts"


def _draft_index_path(board_dir) -> Path:
    return drafts_dir(board_dir) / "index.json"


def _draft_slug(name: str, taken) -> str:
    base = _slug(name, 40) or "draft"
    slug, n = base, 2
    while slug in taken:
        slug = f"{base}-{n}"
        n += 1
    return slug


def load_draft_index(board_dir) -> dict:
    """The index, MIGRATING a pre-drafts board on the way out.

    A board that has only `edit.json` has exactly one draft and always did —
    it just had no name for it. Naming it here (rather than in a migration
    pass that rewrites files) means the upgrade cannot half-happen: the worst
    case is a board that gets its index written the first time it is read.
    """
    p = _draft_index_path(board_dir)
    idx = None
    if p.is_file():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("drafts"), list):
                idx = data
        except (OSError, ValueError):
            idx = None
    if idx is None:
        idx = {"version": DRAFT_INDEX_VERSION, "active": "draft-1",
               "drafts": [{"slug": "draft-1", "name": "Draft 1",
                           "created_at": int(time.time())}]}
    rows = [d for d in idx.get("drafts") or []
            if isinstance(d, dict) and d.get("slug")]
    if not rows:
        rows = [{"slug": "draft-1", "name": "Draft 1",
                 "created_at": int(time.time())}]
    idx["drafts"] = rows
    if idx.get("active") not in {d["slug"] for d in rows}:
        idx["active"] = rows[0]["slug"]
    idx["version"] = DRAFT_INDEX_VERSION
    return idx


def _save_draft_index(board_dir, idx: dict) -> Path:
    drafts_dir(board_dir).mkdir(parents=True, exist_ok=True)
    return _atomic_json(_draft_index_path(board_dir), idx, prefix=".drafts-")


def _draft_file(board_dir, slug: str) -> Path:
    return drafts_dir(board_dir) / f"{_slug(str(slug), 40)}.json"


def _draft_stats(doc) -> dict:
    if not isinstance(doc, dict):
        return {"clips": None, "duration": None, "revision": None,
                "saved_at": None}
    clips = doc.get("clips")
    return {"clips": len(clips) if isinstance(clips, list) else 0,
            "duration": edit_duration(doc),
            "revision": int(doc.get("revision") or 0),
            "saved_at": (int(doc["updated_at"])
                         if isinstance(doc.get("updated_at"), (int, float))
                         else None)}


def _read_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def list_drafts(board_dir) -> list[dict]:
    """Every draft this film has, the active one marked, with what it holds."""
    idx = load_draft_index(board_dir)
    out = []
    for d in idx["drafts"]:
        active = (d["slug"] == idx["active"])
        doc = _read_json(edit_path(board_dir) if active
                         else _draft_file(board_dir, d["slug"]))
        row = {"slug": d["slug"], "name": d.get("name") or d["slug"],
               "active": active,
               "created_at": d.get("created_at"),
               "exists": doc is not None}
        row.update(_draft_stats(doc))
        out.append(row)
    return out


def _stash_active(board_dir, idx: dict) -> None:
    """Put the active draft where an inactive one lives. Never loses it."""
    cur = edit_path(board_dir)
    if not cur.is_file():
        return
    doc = _read_json(cur)
    if doc is None:
        return
    drafts_dir(board_dir).mkdir(parents=True, exist_ok=True)
    _atomic_json(_draft_file(board_dir, idx["active"]), doc, prefix=".draft-")


def create_draft(board_dir, name: str, *, from_current: bool = False) -> dict:
    """A new draft, empty or copied from what is on screen. Becomes active.

    `from_current` is the "work on a variation without risking this one"
    gesture — the whole reason the owner asked for copy-and-paste drafts.
    """
    text = str(name or "").strip()[:DRAFT_NAME_MAX] or "Untitled draft"
    idx = load_draft_index(board_dir)
    slug = _draft_slug(text, {d["slug"] for d in idx["drafts"]})
    # MIGRATED, LIKE EVERY OTHER READ. `validate_edit` refuses any version but
    # the current one and `save_edit` refuses what it complains about, so
    # handing the raw file to save_edit meant copying a draft raised
    # "edit version 1 — this build understands 2" on every board written
    # before EDIT_VERSION went to 2 — which is every board anybody already
    # had. Migration is a read-path job and this is a read.
    current = migrate_edit(_read_json(edit_path(board_dir)) or {}) or {}
    _stash_active(board_dir, idx)
    if from_current and current:
        doc = dict(current)
    else:
        # An EMPTY draft still carries the soundtrack and the beat grid: they
        # are facts about the film, not about the arrangement, and re-pointing
        # at the track for every new idea is work nobody asked for.
        doc = {"version": EDIT_VERSION, "board_id": current.get("board_id", ""),
               "source": "human", "clips": [], "settings": {},
               "audio": current.get("audio"),
               "beats": current.get("beats")}
    doc["revision"] = 0
    return _land_draft(board_dir, idx, slug, text, doc)


def duplicate_draft(board_dir, slug: str, name: str = "") -> dict:
    """Copy a draft under a new name. The copy becomes active."""
    idx = load_draft_index(board_dir)
    known = {d["slug"]: d for d in idx["drafts"]}
    src = str(slug or idx["active"])
    if src not in known:
        raise EditError("there is no draft by that name")
    doc = _read_json(edit_path(board_dir) if src == idx["active"]
                     else _draft_file(board_dir, src))
    if doc is None:
        raise EditError("that draft has nothing in it to copy")
    text = (str(name or "").strip()[:DRAFT_NAME_MAX]
            or f"{known[src].get('name') or src} copy")
    return _land_copy(board_dir, idx, text, migrate_edit(doc))


def _land_copy(board_dir, idx: dict, text: str, doc: dict) -> dict:
    new_slug = _draft_slug(text, {d["slug"] for d in idx["drafts"]})
    _stash_active(board_dir, idx)
    doc = dict(doc)
    doc["revision"] = 0
    return _land_draft(board_dir, idx, new_slug, text, doc)


def _land_draft(board_dir, idx: dict, slug: str, text: str, doc: dict) -> dict:
    """Write the new draft's document, THEN point the index at it.

    ORDER IS THE WHOLE FUNCTION. The index used to be saved first, so a
    document `save_edit` refused left the film pointing at a draft that was
    never written: `list_drafts` showed the new name as the ACTIVE one while
    `edit.json` still held the old draft's arrangement, and the client was
    told 400 for an index it could not see had already moved. A write that
    lands and then a pointer that follows it cannot produce that state.

    The archive `save_edit` makes on the way past also lands correctly this
    way round — the outgoing document belongs to the draft that is still
    active, not to the one being created.
    """
    save_edit(board_dir, doc)
    idx["drafts"].append({"slug": slug, "name": text,
                          "created_at": int(time.time())})
    idx["active"] = slug
    _save_draft_index(board_dir, idx)
    return {"slug": slug, "name": text}


def rename_draft(board_dir, slug: str, name: str) -> dict:
    """A NAME CHANGE IS NOT A FILE MOVE. The slug is the identity and it stays
    put, so renaming can never orphan a draft's file or its backup."""
    text = str(name or "").strip()[:DRAFT_NAME_MAX]
    if not text:
        raise EditError("a draft needs a name")
    idx = load_draft_index(board_dir)
    for d in idx["drafts"]:
        if d["slug"] == str(slug):
            d["name"] = text
            _save_draft_index(board_dir, idx)
            return {"slug": d["slug"], "name": text}
    raise EditError("there is no draft by that name")


def delete_draft(board_dir, slug: str) -> dict:
    """Remove a draft. The LAST one cannot go, and neither can the active one
    without something to land on — an editor with no document is not a state
    this app has, and inventing it here would be inventing a bug."""
    idx = load_draft_index(board_dir)
    rows = idx["drafts"]
    if len(rows) < 2:
        raise EditError("this is the film's only draft")
    target = str(slug)
    if target not in {d["slug"] for d in rows}:
        raise EditError("there is no draft by that name")
    was_active = (idx["active"] == target)
    if was_active:
        # Land on the neighbour FIRST, then drop the file — so a failure
        # halfway leaves a film with a document rather than without one.
        others = [d["slug"] for d in rows if d["slug"] != target]
        activate_draft(board_dir, others[0])
        idx = load_draft_index(board_dir)
        rows = idx["drafts"]
    try:
        _draft_file(board_dir, target).unlink()
    except OSError:
        pass
    try:
        _backup_path(board_dir, target).unlink()
    except OSError:
        pass
    # "Its past saves go with it" — what the panel says when it asks, and now
    # what happens. A draft's history is its own folder, so this is one call
    # rather than a glob nobody can prove is scoped right.
    import shutil                                                # noqa: PLC0415
    shutil.rmtree(history_dir(board_dir, target), ignore_errors=True)
    idx["drafts"] = [d for d in rows if d["slug"] != target]
    _save_draft_index(board_dir, idx)
    return {"active": idx["active"], "deleted": target}


def activate_draft(board_dir, slug: str) -> dict:
    """Switch which draft `edit.json` is. The outgoing one is stashed first."""
    idx = load_draft_index(board_dir)
    target = str(slug)
    if target not in {d["slug"] for d in idx["drafts"]}:
        raise EditError("there is no draft by that name")
    if target == idx["active"]:
        return {"active": target, "changed": False}
    _stash_active(board_dir, idx)
    doc = _read_json(_draft_file(board_dir, target))
    if doc is None:
        # A draft whose file never landed (created, never saved) opens empty
        # rather than refusing — the alternative is a name in the list that
        # cannot be clicked.
        doc = {"version": EDIT_VERSION, "revision": 0, "source": "human",
               "clips": [], "settings": {}, "audio": None, "beats": None}
    doc = migrate_edit(doc)
    doc["revision"] = int(doc.get("revision") or 0)
    # The document lands BEFORE the pointer moves, for the reason spelled out
    # in `_land_draft`: an index that points at a document which was never
    # written is a film whose active draft holds another draft's arrangement.
    save_edit(board_dir, doc)
    idx["active"] = target
    _save_draft_index(board_dir, idx)
    try:
        _draft_file(board_dir, target).unlink()
    except OSError:
        pass
    return {"active": target, "changed": True}


# ---- the quiet lane -------------------------------------------------------
def _snapshot_paths(board_dir, slug: str = "") -> list[Path]:
    """Every snapshot of one draft, OLDEST FIRST. The legacy slot counts as one.

    Rule 6: nothing on disk is deleted. A board written before the lane was
    versioned has a single `backup-<draft>.json`, and it is read as the oldest
    snapshot of that draft rather than ignored or thrown away.
    """
    idx_slug = str(slug or load_draft_index(board_dir)["active"])
    hist = history_dir(board_dir, idx_slug)
    # MOVED, NOT READ IN PLACE. A legacy slot left at the root of history/ is
    # invisible to `list_history` and unreachable by `restore_edit`, both of
    # which are scoped to the draft's own folder — so it would have been a
    # snapshot the user could be told about and could not open. Renaming it
    # into the lane makes it a version like any other, and renaming is not
    # deleting: rule 6 holds.
    legacy = _backup_path(board_dir, idx_slug)
    if legacy.is_file():
        try:
            stamp = int(legacy.stat().st_mtime * 1000)
        except OSError:
            stamp = 0
        hist.mkdir(parents=True, exist_ok=True)
        dst = hist / f"{_SNAP_PREFIX}{stamp:013d}-000.json"
        bump = 0
        while dst.exists():
            bump += 1
            dst = hist / f"{_SNAP_PREFIX}{stamp:013d}-{bump:03d}.json"
        try:
            legacy.replace(dst)
        except OSError:
            pass
    out: list[Path] = []
    if hist.is_dir():
        out.extend(sorted(p for p in hist.glob(f"{_SNAP_PREFIX}*.json")
                          if p.is_file()))
    return out


def latest_snapshot(board_dir, slug: str = "") -> tuple[Path, dict] | None:
    """The newest readable snapshot of one draft, or None."""
    for p in reversed(_snapshot_paths(board_dir, slug)):
        doc = _read_json(p)
        if isinstance(doc, dict):
            return p, doc
    return None


def prune_snapshots(board_dir, *, keep: int = SNAPSHOT_KEEP) -> int:
    """Drop the oldest snapshots past the cap. Losing one is not an event."""
    paths = _snapshot_paths(board_dir)
    gone = 0
    for stale in paths[:max(0, len(paths) - int(keep))]:
        try:
            stale.unlink()
            gone += 1
        except OSError:
            continue
    return gone


def _backup_path(board_dir, slug: str) -> Path:
    # The crash lane sits at the ROOT of history/, beside the drafts' folders
    # rather than inside one: it is not a past save, and `list_history` walks
    # a draft's folder without needing to know it exists.
    return _history_root(board_dir) / f"{_BACKUP_PREFIX}{_slug(str(slug), 40)}.json"


def write_backup(board_dir, edit: dict, *, draft: str = "",
                 session: str = "") -> Path:
    """Crash insurance for the ACTIVE draft. Writes nothing the user owns.

    Validated like any other document — a backup that cannot be restored is
    not a backup — but it never lands on `edit.json` and never moves a
    revision, so the saved draft on disk is exactly what the user last saved
    and nothing else.

    `draft` IS THE PAYLOAD'S OWN ACCOUNT OF WHERE IT CAME FROM, and it is
    checked rather than trusted. The backup is debounced on the client and the
    server is threaded, so a write composed while draft A was on screen can
    arrive after the user has clicked draft B — and a backup filed under B's
    name is A's arrangement offered back as B's unsaved work, which
    `recover_backup` would then install over B's saved document. A caller that
    names its draft gets refused when it has left; one that names nothing
    keeps the old behaviour.
    """
    # THE SAFETY NET MAY NOT HAVE AN OPINION. A backup that refuses is the one
    # failure this file exists to prevent, so it declines only on the errors
    # that would make the document unrestorable.
    errs = blocking_errors(validate_edit(edit))
    if errs:
        raise EditError("; ".join(e["message"] for e in errs[:3]))
    idx = load_draft_index(board_dir)
    want = str(draft or "").strip()
    if want and want != idx["active"]:
        raise EditError("that backup belongs to a draft you have left")
    # WRITING IS WHAT CLAIMS THE BOARD, and it is the only thing that does.
    # This used to REFUSE a session that did not hold the claim, and the claim
    # was taken by loading the page — so opening the film in a second tab, or
    # in a headless browser, or as an agent, silently stopped the snapshot lane
    # in the tab the person was working in. The refusal is gone; the claim
    # moves to whoever is actually editing.
    if str(session or "").strip():
        try:
            claim_session(board_dir, session)
        except EditError:
            pass
    doc = normalise_edit(edit)
    # AN IDENTICAL SNAPSHOT IS NOT A SNAPSHOT. The lane is 20 files deep, and
    # on the day this was written it took ten of them in fifteen seconds — same
    # revision, same clips, same digest — which is half the net spent on one
    # arrangement and, worse, half the DISTINCT history evicted to hold it. The
    # debounce is doing its job; what it cannot know is that the timer it fired
    # was armed by something that did not change the film. `edit_digest` is the
    # same fingerprint `pending_backup` compares on, so "nothing to snapshot"
    # here means exactly "nothing that lane would have offered you".
    newest = latest_snapshot(board_dir, idx["active"])
    if newest is not None and edit_digest(newest[1]) == edit_digest(doc):
        return newest[0]
    doc["origin"] = "backup"
    doc["backup_of"] = idx["active"]
    doc["backed_up_at"] = int(time.time())
    # The revision `edit.json` stood at when this was composed. Nothing gates
    # on it today — the offer ends when the user ANSWERS it, not when a clock
    # says so — but a file on its own can now say which save it followed.
    doc["backup_revision"] = int((_read_json(edit_path(board_dir)) or {})
                                 .get("revision") or 0)
    # ONE FILE PER SNAPSHOT. The lane never overwrites and therefore never
    # has to stop: a new snapshot beside the last one cannot destroy work an
    # unanswered offer is holding, which is the entire reason the old single
    # slot had to be guarded and the guard switched the safety net off.
    hist = history_dir(board_dir, idx["active"])
    hist.mkdir(parents=True, exist_ok=True)
    # Named by the clock, and made unique when the clock does not move: two
    # snapshots inside one millisecond are ordinary on a fast machine, and a
    # collision here would be the overwrite this lane exists to make
    # impossible.
    # The counter is ALWAYS present, never appended only on collision: a name
    # that grows a suffix sorts BEFORE the one it collided with ('-' < '.'),
    # so the bumped file would have read as the older of the two and the lane
    # would have handed back the wrong snapshot. Uniform names sort right.
    stamp = int(time.time() * 1000)
    bump = 0
    dst = hist / f"{_SNAP_PREFIX}{stamp:013d}-{bump:03d}.json"
    while dst.exists():
        bump += 1
        dst = hist / f"{_SNAP_PREFIX}{stamp:013d}-{bump:03d}.json"
    out = _atomic_json(dst, doc, prefix=".snap-")
    prune_snapshots(board_dir)
    return out


def pending_backup(board_dir) -> dict | None:
    """Unsaved work sitting in the crash lane, or None.

    This is an OFFER, never an action: the panel shows what is in it and the
    user decides. Silently applying it would be the autosave he asked us to
    remove, wearing a different name.

    AN OFFER ENDS WHEN THE USER ANSWERS IT — Save deletes the file, Recover
    applies it, Discard drops it. It used to end when a clock said so, and the
    clock was wrong in both directions: `backed_up_at` and `updated_at` are
    whole seconds, so a backup written just BEFORE a save read as newer than
    it; and every write that is not the user's — a draft switch, an
    auto-edit, a restore — moved `updated_at` forward and buried an offer that
    still held the only copy of somebody's afternoon. What is left is the one
    honest reason not to ask: the backup holds nothing the saved draft does
    not already have.
    """
    idx = load_draft_index(board_dir)
    newest = latest_snapshot(board_dir, idx["active"])
    if newest is None:
        return None
    p, doc = newest
    saved = _read_json(edit_path(board_dir)) or {}
    stats = _draft_stats(doc)
    # CONTENT, BY DIGEST. The old test was `json.dumps(clips)` on both sides,
    # and `_sbe_payload` rewrites every clip's `proxy` on the way out — so the
    # client's copy differed from the file in a field the user has never heard
    # of, on any board whose proxies were built after its last save, forever.
    # `edit_digest` compares the arrangement and ignores everything derived.
    if edit_digest(doc) == edit_digest(saved):
        return None
    return {"file": p.name, "at": int(doc.get("backed_up_at") or 0),
            "draft": idx["active"], "clips": stats["clips"],
            "duration": stats["duration"]}


def recover_backup(board_dir) -> dict:
    """Apply the pending backup, keeping the saved draft it replaces.

    IT RECOVERS ONLY WHAT THE PANEL WOULD HAVE OFFERED. A client can hold a
    stale offer — it saved a second ago, the file is gone, the amber bar is
    still on screen in another tab — and applying that would archive the good
    save and install an older arrangement over it, which is the exact loss the
    backup exists to prevent, arriving through the recovery button.
    """
    idx = load_draft_index(board_dir)
    newest = latest_snapshot(board_dir, idx["active"])
    if newest is None:
        raise EditError("there is no snapshot to restore")
    _, doc = newest
    if pending_backup(board_dir) is None:
        raise EditError("that backup is older than your last save — "
                        "everything in it is already in this draft")
    doc = migrate_edit(dict(doc))
    doc.pop("backup_of", None)
    doc.pop("backed_up_at", None)
    doc["origin"] = "manual"      # recovering IS the user's decision
    saved = _read_json(edit_path(board_dir))
    if isinstance(saved, dict):
        archive_edit(board_dir, saved)
        prune_history(board_dir)
        doc["revision"] = int(saved.get("revision") or 0)
    # BLOCKING ONLY — the same rule `save_edit` and `write_backup` apply.
    # Using the raw list here meant a WARNING refused the recovery of work
    # that had been perfectly legal to write: `clips_audio_overlap` is in
    # WARNING_CODES, and it is what an ordinary J-cut reports. So a crash
    # over a split edit — the feature this editor exists for — wrote a
    # backup fine and then refused to give it back, permanently, because
    # the offer is never discarded on the failure path and an unanswered
    # offer suppresses further snapshots. Persisting the user's work wins.
    errs = blocking_errors(validate_edit(doc))
    if errs:
        raise EditError("that backup cannot be recovered: "
                        + "; ".join(e["message"] for e in errs[:3]))
    save_edit(board_dir, doc)
    discard_backup(board_dir)
    return load_edit(board_dir)


def discard_backup(board_dir) -> bool:
    """Answer the offer by dropping the newest snapshot. The rest stay.

    A save also calls this, and it means the same thing there: what the chip
    was offering is now in the document, so the chip has nothing to say.
    """
    idx = load_draft_index(board_dir)
    newest = latest_snapshot(board_dir, idx["active"])
    if newest is None:
        return False
    try:
        newest[0].unlink()
        return True
    except OSError:
        return False
