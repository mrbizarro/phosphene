#!/usr/bin/env python3
"""Sliding windows for LTX — the arithmetic and the per-window prompt contract.

WHY THIS EXISTS. One LTX pass tops out at short clips: past 241 frames the
panel falls back to `fps12_interp24` (fewer model frames, interpolated back to
24). H3 already chains windows; LTX did not. What turns 10-second clips into
minute-long ones is not a bigger pass, it is a SEQUENCE of passes where each
window continues the last one — and, just as importantly, where every window
gets its OWN prompt with the sequence-wide invariants re-injected, because a
single prompt repeated per window plays the shot's action once per window and
the clip reads as a stutter (the H3 "repeat" artefact, `H3_CHAIN_PROMPT_HELP`).

THE GEOMETRY, in frames:

    stride = window - discard - overlap
    count  = 1 + ceil((total - window + discard) / stride)

`window` is what one pass renders; `overlap` is the tail of the previous
window the next one re-sees as context (LTX's extend pipeline conditions on
the clip it is handed, so this is the tail that is kept for it); `discard` is
the tail dropped BEFORE conditioning, because the last frames of a window are
the least settled. `stride` is the new picture each later window adds.

ON LTX'S GRID. A window is `8k+1` frames (the sampler's rule) and the extend
pipeline adds latent frames in groups of 8, so `stride` must be a multiple of
8 — the planner rounds it DOWN to the grid and says so in the plan. The last
window may overshoot `total`; the delivered clip is trimmed to `total` at the
mux, never padded.

This module is PURE — numbers in, numbers out — so the plan, the prompt list
and their invariants are testable without a model in the room. The panel
(`mlx_ltx_panel.py`) turns the plan into one `generate` and N `extend` calls.
"""
from __future__ import annotations

import math

# 121 frames = 5 s at 24 fps, the canvas every LTX tier estimate is measured
# on. 9 frames of overlap is one latent group (8) plus the shared frame: the
# smallest context the extend pipeline can be handed that still spans a whole
# latent. Zero discard by default — dropping frames is a quality lever the
# user turns, not a tax every window pays.
DEFAULT_WINDOW = 121
DEFAULT_OVERLAP = 9
DEFAULT_DISCARD = 0
LATENT_GROUP = 8
MAX_WINDOWS = 24            # a minute and a half at the defaults; past that
                            # the chain drift nobody has measured owns the clip
FPS = 24


def snap_8k1(frames: int) -> int:
    """The sampler's grid: the nearest `8k+1` at or above 9."""
    n = max(1, int(frames))
    return max(9, LATENT_GROUP * round((n - 1) / LATENT_GROUP) + 1)


def stride_for(window: int, overlap: int, discard: int) -> int:
    """New frames per later window, on the latent grid (rounded DOWN)."""
    raw = int(window) - int(discard) - int(overlap)
    return max(LATENT_GROUP, (raw // LATENT_GROUP) * LATENT_GROUP)


def window_count(total: int, window: int, overlap: int, discard: int) -> int:
    """How many windows cover `total` frames. One when one pass is enough."""
    total, window = int(total), int(window)
    if total <= window:
        return 1
    stride = stride_for(window, overlap, discard)
    return 1 + int(math.ceil((total - window + int(discard)) / stride))


def plan_windows(total_frames: int, *, window: int = DEFAULT_WINDOW,
                 overlap: int = DEFAULT_OVERLAP, discard: int = DEFAULT_DISCARD,
                 fps: float = FPS) -> dict:
    """The window schedule for a clip of `total_frames`.

    Returns::

        {"total_frames", "window", "overlap", "discard", "stride", "count",
         "delivered_frames",          # what the chain produces before the trim
         "windows": [{"index", "start_frame", "frames", "new_frames",
                      "start_sec", "end_sec"}, ...],
         "notes": [...]}              # every rounding the planner made

    `windows[0]` is the first pass (`frames` = the window). Every later entry
    GENERATES `new_frames` (= stride + discard) on top of the kept picture,
    re-seeing `overlap` frames of the previous tail; `kept_frames` is how much
    of the clip is kept after that window's own discard.
    """
    notes: list[str] = []
    total = snap_8k1(total_frames)
    if total != int(total_frames):
        notes.append(f"total {int(total_frames)}f snapped to {total}f (8k+1)")
    win = snap_8k1(window)
    if win != int(window):
        notes.append(f"window {int(window)}f snapped to {win}f (8k+1)")
    ov = max(1, int(overlap))
    ov = LATENT_GROUP * ((ov - 1) // LATENT_GROUP) + 1
    if ov != int(overlap):
        notes.append(f"overlap {int(overlap)}f snapped to {ov}f (8k+1)")
    dc = max(0, int(discard))
    dc = (dc // LATENT_GROUP) * LATENT_GROUP
    if dc != int(discard):
        notes.append(f"discard {int(discard)}f snapped to {dc}f (8k)")
    if ov + dc >= win:
        raise ValueError(f"overlap {ov} + discard {dc} leaves no new frames "
                         f"in a {win}-frame window")
    stride = stride_for(win, ov, dc)
    count = window_count(total, win, ov, dc)
    if count > MAX_WINDOWS:
        raise ValueError(f"{total} frames would take {count} windows; the "
                         f"panel stops at {MAX_WINDOWS}")
    # EVERY WINDOW DROPS ITS LAST `dc` FRAMES before the next one conditions
    # on it — including the first and the last — so the kept picture is
    # `win - dc + (count - 1) * stride`, which is exactly what the count
    # formula solved for. A later window GENERATES `stride + dc` new frames
    # (the extend call's size) and KEEPS `stride` of them.
    rows = []
    kept = win - dc
    rows.append({"index": 0, "start_frame": 0, "frames": win, "new_frames": win,
                 "kept_frames": kept, "start_sec": 0.0,
                 "end_sec": round((kept - 1) / fps, 3)})
    for k in range(1, count):
        start = kept - ov
        gen = stride + dc
        kept = kept + stride
        rows.append({"index": k, "start_frame": start, "frames": kept + dc,
                     "new_frames": gen, "kept_frames": kept,
                     "start_sec": round(start / fps, 3),
                     "end_sec": round((kept - 1) / fps, 3)})
    return {"total_frames": total, "window": win, "overlap": ov, "discard": dc,
            "stride": stride, "count": count, "delivered_frames": kept,
            "windows": rows, "notes": notes}


def extend_latents(new_frames: int) -> int:
    """`extend_frames` for the helper: latent groups, never a fraction."""
    return max(1, int(new_frames) // LATENT_GROUP)


# ---------------------------------------------------------------------------
# THE PROMPT CONTRACT
# ---------------------------------------------------------------------------
# ONE PROMPT PER WINDOW. The first window gets the shot's real prompt. A later
# window gets ITS OWN line — the next beat of the action, written by whoever
# planned the clip — and, whether or not it has one, the sequence-wide
# INVARIANTS are re-injected: the things that must not change between passes
# (who is in frame, the light, the lens, the place). Without them every window
# is free to re-decide the world, and the seam is where it shows.
#
# LEAD WITH THE MOVE. LTX renders whatever the prompt names FIRST; leading
# with the subject instead of the movement froze shots (memory:
# ltx-prompt-lead-with-the-move). So a continuation line is placed FIRST and
# the invariants FOLLOW it — never the other way round.
CONTINUE = ("The shot continues without a cut, exactly where it left off, and "
            "the camera keeps the same framing.")
HOLD = ("Nothing new begins: the settled state of the previous moment is held, "
        "with only small natural motion.")


def window_prompts(base: str, per_window=None, invariants: str = "",
                   count: int = 1, settle: str = "") -> list[str]:
    """The prompt each window renders with. Length == `count`.

    `per_window[k]` is the line for window `k` (blank = none; index 0 blank
    means "use `base`" — the same contract the H3 chain uses). `invariants` is
    appended to every LATER window; `settle` is what a later window with no
    line of its own is told to hold, and with neither it holds the previous
    moment.
    """
    base = str(base or "").strip()
    inv = str(invariants or "").strip().rstrip(".")
    settle = str(settle or "").strip().rstrip(".")
    lines = [str(x or "").strip() for x in (per_window or [])]
    out: list[str] = []
    for k in range(max(1, int(count))):
        own = lines[k] if k < len(lines) else ""
        if k == 0:
            out.append(own or base)
            continue
        head = own or (f"{settle}, held." if settle else HOLD)
        parts = [head, CONTINUE]
        if inv:
            parts.append(f"Throughout: {inv}.")
        out.append(" ".join(p for p in parts if p))
    return out


def describe(plan: dict) -> str:
    """One line for a log or a card."""
    w = plan["windows"]
    secs = plan["delivered_frames"] / float(FPS)
    return (f"{plan['count']} window(s) of {plan['window']}f, +{plan['stride']}f "
            f"each (overlap {plan['overlap']}f, discard {plan['discard']}f) "
            f"→ {plan['delivered_frames']}f ≈ {secs:.1f}s for {plan['total_frames']}f asked"
            + (f"; last window ends at {w[-1]['end_sec']}s" if len(w) > 1 else ""))
