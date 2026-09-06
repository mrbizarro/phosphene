#!/usr/bin/env python3
"""Executable UI contract for the timeline editor's CLIENT.

`test_storyboard_editor_api.py` locks the server: proxies, peaks, `edit.json`,
the eight routes. This locks the other half — the half that actually decides
where a cut lands, because the maths that moves a clip lives in the browser.

The panel ships its client inside `mlx_ltx_panel.py` as a string, so a test that
"covers" it by grepping for function names would pass while every drag was off
by a frame. This extracts the REAL functions and RUNS them in node, exactly the
way `test_stage_live_preview.py` and the character round-trip gate do. If the
drag maths changes, this fails; if a function is renamed, extraction raises
rather than quietly reducing coverage back to grepping.

What is locked here:
  * peaks decode — interleaved int16 over `scale`, count clamped to the array
  * the beat grid is never extrapolated past the fitted `span`
  * snap catches a beat inside the tolerance and NOTHING outside it, and the
    override is a straight bypass
  * move / trim / ripple / split, including the 1x invariant the server refuses
    an edit for breaking, and the refusal to touch a locked clip
  * undo/redo
  * the save payload's shape, including `expect_revision` and the stripping of
    client-only bookkeeping
  * a 409 leaves the arrangement on screen
  * validation errors map back onto the clip that caused them
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

from extract_panel_js import (extract_element, extract_function,  # noqa: E402
                              panel_source)


def panel_html_render() -> str:
    """The page as the browser is handed it, tokens replaced."""
    import mlx_ltx_panel as _p                                    # noqa: PLC0415
    return (_p.HTML.replace("__SEQCAP__", _p.SEQ_NOUN_CAP)
            .replace("__SEQS__", _p.SEQ_NOUN_PL)
            .replace("__SEQ__", _p.SEQ_NOUN))

NODE = shutil.which("node")

# The model functions are pure — arrays in, arrays out. The four that are not
# (sbeSave, sbeMutate, sbeUndo, sbeRedo) reach for exactly four collaborators,
# and those are stubbed below rather than extracted, so a failure here is a
# failure of the thing under test.
FUNCTIONS = (
    "sbeNum", "sbeRound", "sbeFps", "sbeGridGap", "sbeLen",
    "sbePaintProtected",
    "sbeAdoptGaps", "sbeLayout",
    "sbeFilmDuration", "sbeClipAt", "sbeHoles", "sbeBeatGrid",
    "sbeGridIsAGuess", "sbeSnapTime", "sbeById", "sbeMoveTo", "sbeTrim",
    "sbeRippleDelete", "sbeLiftDelete", "sbeSplitAt", "sbeNewId", "sbePlaceUnplaced",
    "sbeCleanClip", "sbeSaveBody", "sbeErrorsByClip", "sbeDecodePeaks",
    "sbeFmtTime", "sbeMutate", "sbeUndo", "sbeRedo", "sbeSave",
    # Wave 2 — kinds, the one adjustment, and the drop maths.
    "sbeKind", "sbeBright", "sbeBrightnessCss", "sbeSetBrightness",
    "sbeDropIndex", "sbeInsertAt", "sbeReorderTo",
    # Wave 3 — the two sliders and the two monitors. Pure by design: "zoom
    # keeps the playhead still", "the slider's floor fits the whole film" and
    # "the monitor row fills the width" are all invariants nobody can check by
    # eye and everybody notices when they break.
    "sbeZoomFitPps", "sbeZoomFromSlider", "sbeZoomToSlider", "sbeZoomAnchor",
    "sbeZoomScroll", "sbeFollowScroll", "sbeMonitorFit",
    # The third slider nobody called a slider: the timeline's own top edge.
    # Where a dragged pixel lands is the whole feature and the one thing that
    # cannot be checked by looking at it.
    "sbeTlClamp", "sbeLaneHeights", "sbeTlPrefRead", "sbeTlPrefWrite",
    # The level line's geometry, which three gestures now share instead of
    # each carrying a copy of a 20px band.
    "sbeStripY", "sbeStripGain", "sbeStripEditable", "sbeKeysLegend",
    # ...and the span of it that the two corner handles have NOT taken. Pure
    # arithmetic on a polyline, which is the only part of "no pixel belongs to
    # two controls" that can be checked without a browser.
    "sbeLvlHitPath",
    # The two stage layers, and the painter that grades them. Opacity is the
    # ramp AND the layer switch, which is how a fade came to paint black over
    # a perfectly loaded video.
    "sbeFadePaint", "sbeOvPaint",
    # Wave 4 — the soundtrack as an object, and a timeline that does not end
    # where the clips do. Every one of these is arithmetic somebody would
    # otherwise have to check by dragging a strip and squinting.
    "sbeMusicWindow", "sbeMusicEdit", "sbeMusicSnaps", "sbeSnapToList",
    "sbeSpan", "sbeSnapshot", "sbeRestore",
    # Wave 4 — the versions picker's wording, which is the whole of what it
    # tells you about a file you are about to restore over your work.
    "sbeAgo", "sbeVersionLine",
    # J-cuts and L-cuts: the sound's own window.
    "sbeClipAudio", "sbeAudioField", "sbeSetAudioLink", "sbeAudioEdit",
    # ...and the pair, which is the half that was missing: how far the two have
    # come apart, what carries a strip through somebody else's ripple, and the
    # one click that puts it back.
    "sbeClipMuted", "sbeSetClipMute",
    # The preview's strip player: who owns a clip's sound, and which strips
    # are audible at a given film second.
    "sbeStripOwned", "sbePictureCarriesSound", "sbeStripsAt",
    # The effects model and its first citizen.
    "sbeClipLen", "sbeFx", "sbeFadeOpacityAt", "sbeSetFade",
    # "Did the film actually change?" — the question every track drag has to
    # answer on pointerup, and the one that has now twice silently discarded a
    # real edit because its list of fields was short.
    "sbeDragFingerprint",
    # The overlay lane — a second video track, above the picture.
    # The sound's envelope: fades and keyframes, one curve.
    "sbeAfx", "sbeLerpGain", "sbeGainPoints", "sbeGainAt", "sbeSetAudioFade",
    # Waveforms, keyframes, and the strip you can delete.
    "sbeWaveSlice", "sbeAfxWrite", "sbeAddKeyframe", "sbeMoveKeyframe",
    "sbeDeleteKeyframe", "sbeDeleteStrip",
    # THE MIX. The bed's level, and what happens to it under a line. Every one
    # of these is a mirror of a Python function of the same shape, and the
    # whole feature is that the two agree: the render used to hold the bed at
    # a hard-coded 0.20 and duck it through a compressor while the preview
    # applied neither, so the surface the user checks his work on played a mix
    # the file never had. These are the functions that make the two one.
    "sbeAudioMix", "sbeBedLen", "sbeAudibleStrips", "sbeDuckGainAt",
    "sbeBedDuckPoints", "sbeBedDuckSuppressed", "sbeBedGainPoints",
    "sbeBedGainAt", "sbeMixWrite", "sbeBedAfxWrite", "sbeSetBedFade",
    "sbeBedPointsWrite", "sbeBedAddKeyframe", "sbeBedMoveKeyframe",
    "sbeBedDeleteKeyframe",
    "sbeOvKind", "sbeOvAt", "sbeOvById", "sbeOvFits", "sbeOvMove", "sbeOvTrim",
    "sbeOvAdd", "sbeOvDelete", "sbeOvSetPath",
    # Editor v2 — speed on the clip, titles on the lane, transitions on the
    # cut. Every one is a mirror of a Python function of the same name-shape
    # and `test_editor_v2.py` runs the two side by side.
    "sbeSpeed", "sbeSetSpeed", "sbeOvText", "sbeHexColour", "sbeRgba",
    "sbeOvTextPlace", "sbeDuplicate", "sbeFraming", "sbeFramingIsNeutral", "sbeSetFraming",
    "sbeTxById", "sbeTxAfter", "sbeTxDuration", "sbeTxSpare", "sbeTxResolve",
    "sbeTxEdges", "sbeTxSet", "sbeTxDelete", "sbeTxPrune", "sbeTxRepoint",
    "sbeAudioDrift", "sbeAudioInSync", "sbeAudioIsThePicture",
    "sbeDriftLabel", "sbeSyncBadge",
    "sbeSyncMark", "sbeSyncCarry", "sbeResyncAudio",
    # The save that cannot be dropped, and the failure that cannot be missed.
    "sbeSaveInner", "sbeSaveAlarm", "sbeSaveAlarmClear", "sbeQueueSave",
    # The crash lane itself, so save → backup → recovery can be DRIVEN rather
    # than grepped for the presence of its guards.
    "sbeBackup", "sbeDraftOp", "sbeNameMode",
    # THE ONE NOTICE SURFACE. Four blocks that are not mutually exclusive used
    # to stack; this decides which is open and which is a chip.
    "sbePaintNotices", "sbeNoticeOpen", "sbeNoticeClick", "sbeNoticeLater",
    "sbeErrsToggle",
)

SHIM = r"""
'use strict';
// The four collaborators the mutating functions reach for. Stubs, not
// extractions: this gate is about the arrangement, not the paint.
const painted = [];
const states = [];
const toasts = [];
const saves = [];
function sbePaint() { painted.push(1); }
function sbePaintHead() {}
function sbePaintTrack() {}
function sbePaintChrome() {}
function sbeSetState(t, k) { states.push([t, k]); }
// The two the SAVE path repaints: a save answers the recovery offer, and the
// drafts rows carry the clip count it just changed.
const recoveryPaints = [];
const draftPaints = [];
function sbePaintRecovery() { recoveryPaints.push(SBE.backup); }
function sbePaintDraft() { draftPaints.push((SBE.drafts || []).length); }
// The draft switch reaches this now that the snapshot lane no longer refuses
// while an offer is open. Stubbed: this gate is about the arrangement, not
// about re-adopting a server payload.
const adopted = [];
function sbeAdopt(r, quiet) { adopted.push(r); }
async function sbeVersionsLoad() { return null; }
function sbeVersionsPaint() {}
function sbeRenderErrors(e) { lastErrors = e; }
function phosToast(m, o) { toasts.push(String(m)); }
function escapeHtml(s) { return String(s || ''); }
let lastErrors = null;
const SBE_MIN_CLIP = 0.2;
const SBE_UNDO_MAX = 80;
const SBE_SNAP_PX = 9;
const SBE_GUESS_CONFIDENCE = 0.4;
const SBE_BRIGHT_MAX = 0.5;
const SBE_STILL_SECONDS = 3.0;
// Editor v2. Keep equal to the panel's — test_editor_v2 reads both.
const SBE_SPEED_MIN = 0.25;
const SBE_SPEED_MAX = 4.0;
const SBE_TX_KINDS = ['dissolve', 'fade_black'];
const SBE_TX_MIN = 1 / 24;
const SBE_TX_MAX = 2.0;
const SBE_TX_LABELS = { dissolve: 'Dissolve', fade_black: 'Fade through black' };
const SBE_TEXT_DEFAULTS = { font_size: 64, color: '#ffffff', align: 'center',
                            x: 0.5, y: 0.5, box: false, box_color: '#000000',
                            box_opacity: 0.5 };
const SBE_TEXT_REF_H = 1080;
const SBE_TEXT_MAX = 400;
const SBE_FRAME_ZOOM_MAX = 3.0;
// Wave 3. Keep these equal to the panel's — a drifted constant here is a gate
// that passes while the browser does something else.
const SBE_TL_PAD = 24;
// The timeline's floor, its ceiling and the lanes between them. Both ends are
// the SUM of the lanes; `test_the_height_floor_is_the_sum_of_its_lanes` reads
// them back out of the panel and refuses a lane that was added without moving
// them, which is precisely how the box came to be 30px short of its contents.
const SBE_TL_CHROME = 32;
const SBE_LANES = [
  { key: 'ov',    base: 32, cap:  56, share: 0.08 },
  { key: 'track', base: 64, cap: 120, share: 0.14 },
  { key: 'alane', base: 44, cap: 190, share: 0.44 },
  { key: 'wave',  base: 108, cap: 240, share: 0.34 },
];
const SBE_TL_MIN_H = 280;
const SBE_TL_MAX_H = 638;
const SBE_TL_STEP = 12;
const SBE_TL_STEP_BIG = 40;
const SBE_LVL_PAD = 8;
const SBE_LVL_GRAB = 12;
// The corner handles' footprint, mirrored from the panel exactly as the lane
// table above is. The BROWSER is what catches a drift between these and the
// stylesheet — `scripts/measure_editor_layout.py` measures the three
// rectangles in a laid-out page — but the clipping arithmetic is pure and is
// checked here.
const SBE_AGRIP_W = 7;
const SBE_GRIP_SKIRT = 3;
const SBE_FADE_HIT = 22;
const SBE_LVL_CLEAR = SBE_AGRIP_W + SBE_GRIP_SKIRT + SBE_FADE_HIT + 1;
const SBE_LVL_MIN_SPAN = 12;
const SBE_PPS_MAX = 200;
const SBE_PPS_FLOOR = 0.5;
const SBE_ZOOM_TICKS = 1000;
const SBE_MON_GAP = 12;
const SBE_MON_MIN_H = 120;
const SBE_MON_RATIO = 2 / 3;
const SBE_MON_RATIO_MAX = 1;
const SBE_RAIL_MIN = 200;
const SBE_RAIL_MAX = 380;
// Wave 4 — the slack past the last frame, and the soundtrack's floor.
const SBE_SLACK_MIN = 3;
const SBE_SLACK_MAX = 15;
const SBE_SLACK_RATIO = 0.15;
const SBE_SPAN_MIN = 10;
const SBE_MIN_MUSIC = 0.5;
// The watchdog's grace, and the same threshold the protected chip goes cold
// on. Keep equal to the panel's — a test below reads both.
const SBE_SAVE_GRACE_MS = 12000;
// Half a frame at 24 fps — the same number TOUCH_TOLERANCE uses on the server.
const SBE_SYNC_TOL = 1 / 48;
// Urgency order for the notice surface. Keep equal to the panel's.
const SBE_NOTICE_ORDER = ['sbeConflict', 'sbeAlarm', 'sbeErrors', 'sbeRecover',
                          'sbeKeyed'];
function stubEl(id, hidden) {
  const cls = new Set();
  const attrs = {};
  return {
    id: id, hidden: !!hidden, textContent: '', innerHTML: '', dataset: {},
    // `style` is a plain bag, which is enough to answer the only question
    // this gate asks: WHAT DID THE PAINTER WRITE INLINE. An inline value beats
    // the stylesheet, so an inline value on a layer the stylesheet is trying
    // to hide is the whole of the bug.
    style: {},
    setAttribute: (k, v) => { attrs[k] = String(v); },
    getAttribute: (k) => (k in attrs ? attrs[k] : null),
    classList: {
      add: (c) => cls.add(c), remove: (c) => cls.delete(c),
      contains: (c) => cls.has(c),
      toggle: (c, on) => { if (on) cls.add(c); else cls.delete(c); return cls.has(c); },
    },
    _cls: cls,
  };
}
const els = {
  sbeConflict: stubEl('sbeConflict', true),
  sbeConflictText: stubEl('sbeConflictText'),
  sbeErrors: stubEl('sbeErrors', true),
  sbeAlarm: stubEl('sbeAlarm', true),
  sbeAlarmWhy: stubEl('sbeAlarmWhy'),
  sbeProtected: stubEl('sbeProtected', true),
  sbeRecover: stubEl('sbeRecover', true),
  sbeKeyed: stubEl('sbeKeyed', true),
  sbeKeyedWhat: stubEl('sbeKeyedWhat'),
  sbeRecoverWhat: stubEl('sbeRecoverWhat'),
  sbeNotices: stubEl('sbeNotices', true),
};
function sbeEl(id) { return els[id] || (els[id] = stubEl(id, false)); }
// Which notices are currently chips, in the surface's own order.
const foldedNotices = () => ['sbeConflict', 'sbeAlarm', 'sbeErrors', 'sbeRecover',
                             'sbeKeyed']
  .filter(id => (sbeEl(id)._cls || new Set()).has('is-folded'));
// The view preferences' actual home. A Map with the two methods the panel
// uses, so "it survives a reload" is a thing this gate can DRIVE rather than
// a thing it hopes about: read → drag → write → read again.
const STORE = new Map();
global.localStorage = {
  getItem: (k) => (STORE.has(k) ? STORE.get(k) : null),
  setItem: (k, v) => { STORE.set(k, String(v)); },
  removeItem: (k) => { STORE.delete(k); },
};
const saveTimers = [];
global.setTimeout = ((real) => function (fn, ms) {
  saveTimers.push(ms);
  return real(fn, ms);
})(global.setTimeout);
const SBE = {
  open: true, id: 'sb_t', edit: {}, clips: [], audio: null, beats: null,
  savePending: false, dirtyAt: 0, saveFailed: '',
  peaks: null, unplaced: [], prepare: {}, proxyUrl: '', revision: 0,
  dirty: false, saving: false, conflict: 0, sel: '', playhead: 0,
  playing: false, curId: '', pps: 40, undo: [], redo: [], errors: {},
  sentOrder: [], timer: null, saveTimer: null, raf: 0, drag: null,
  backup: null, drafts: [], activeDraft: 'draft-1', backingUp: false,
  backedUpAt: 0, noticeLead: '', backupHidden: false, errsOpen: false,
  // The timeline's height: the preference, what the window could give it, and
  // the ceiling the monitors impose.
  tlH: SBE_TL_MIN_H, tlNow: SBE_TL_MIN_H, tlMax: SBE_TL_MAX_H, tlDrag: null,
};
// One fetch stub, scripted per case.
let FETCHES = [];
let NEXT = null;
global.fetch = async (url, opts) => {
  FETCHES.push({ url, body: opts && opts.body });
  const r = NEXT;
  return { ok: r.status < 400, status: r.status, json: async () => r.body };
};
global.setTimeout = global.setTimeout;
function clip(o) {
  return Object.assign({
    id: o.id, path: o.path || ('/o/' + o.id + '.mp4'), proxy: null,
    start: 0, end: 2, film_start: 0, film_end: 2, source: 'auto',
    locked: false, duration: 10,
  }, o);
}
function lay(cs) { sbeAdoptGaps(cs); sbeLayout(cs); return cs; }
function shape(cs) {
  return cs.map(c => [c.id, +c.start.toFixed(3), +c.end.toFixed(3),
                      +c.film_start.toFixed(3), +c.film_end.toFixed(3)]);
}
// The invariant the server refuses an edit for breaking: the source window and
// the film slot are the same length, because nothing plays at anything but 1x.
function lengthsAgree(cs) {
  return cs.every(c => Math.abs((c.end - c.start) - (c.film_end - c.film_start)) <= 0.002);
}
function overlaps(cs) {
  const s = cs.slice().sort((a, b) => a.film_start - b.film_start);
  for (let i = 1; i < s.length; i++) {
    if (s[i].film_start < s[i - 1].film_end - 1 / 48) return true;
  }
  return false;
}
"""

BODY = r"""
const out = {};

// ---- peaks ---------------------------------------------------------------
out.peaks = sbeDecodePeaks({
  version: 1, count: 3, scale: 127, buckets_per_second: 100, duration: 0.03,
  peaks: [-127, 127, 0, 64, -32, 32],
});
out.peaksJunk = sbeDecodePeaks({ scale: 0, count: 99, peaks: [-10, 10] });
out.peaksNone = sbeDecodePeaks(null);

// ---- beat grid -----------------------------------------------------------
const beats = {
  bpm: 120, period: 0.5, phase: 0, meter: 4, confidence: 0.71,
  span: [0, 4], beats: [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4],
  downbeats: [0, 2, 4],
};
out.grid = sbeBeatGrid(beats, 0, 10, 0).map(b => [b.t, b.down]);
// The fit stops at 4 s. A film that runs to 30 s gets NO line past the span —
// beat_map refuses to extrapolate and so does the client.
out.gridPastSpan = sbeBeatGrid(beats, 4.01, 30, 0).length;
out.gridWindow = sbeBeatGrid(beats, 1, 2, 0).map(b => b.t);
out.gridOffset = sbeBeatGrid(beats, -2, 10, 1).map(b => b.t);
out.guess = sbeGridIsAGuess(beats);
out.guessLow = sbeGridIsAGuess(Object.assign({}, beats, { confidence: 0.31 }));

// ---- snap ----------------------------------------------------------------
out.snapCatches = sbeSnapTime(1.44, beats, 0.1, true, 0);
out.snapLeaves = sbeSnapTime(1.2, beats, 0.1, true, 0);
out.snapOff = sbeSnapTime(1.44, beats, 0.1, false, 0);
out.snapPrefersDownbeat = sbeSnapTime(2.0001, beats, 0.6, true, 0);
out.snapPastSpan = sbeSnapTime(9.9, beats, 0.6, true, 0);

// ---- layout --------------------------------------------------------------
let cs = lay([clip({ id: 'a', end: 2 }), clip({ id: 'b', end: 3 }),
              clip({ id: 'c', end: 1.5 })].map((c, i) => c));
// adoptGaps read film_start from the incoming doc; these three arrived stacked
// at 0, so the layout packs them in array order.
out.packed = shape(cs);
out.packedLengths = lengthsAgree(cs);

// A hole survives a round trip through adopt + layout — gaps are legal, and an
// editor that quietly closed them would make "generate into a gap" impossible.
let holed = [clip({ id: 'a', end: 2, film_start: 0, film_end: 2 }),
             clip({ id: 'b', end: 3, film_start: 5, film_end: 8 })];
lay(holed);
out.holeKept = shape(holed);
out.holes = sbeHoles(holed).map(g => [g.film_start, g.film_end, g.duration]);
out.duration = sbeFilmDuration(holed);
out.clipAt = (sbeClipAt(holed, 6) || {}).id;
out.clipInHole = sbeClipAt(holed, 3);

// ---- move ----------------------------------------------------------------
cs = lay([clip({ id: 'a', end: 2 }), clip({ id: 'b', end: 3 }), clip({ id: 'c', end: 1.5 })]);
let r = sbeMoveTo(cs, 'c', 0, { ripple: true });   // ⌘: the old free repack
out.moveToHead = shape(r.clips);
out.moveMarksHuman = sbeById(r.clips, 'c').source;
out.moveLengths = lengthsAgree(r.clips);
out.moveNoOverlap = !overlaps(r.clips);
// DEFAULT (no modifier): a clip moves only between its neighbours and they
// stay where they are — the Premiere / After Effects contract.
cs = lay([clip({ id: 'a', end: 2 }), clip({ id: 'b', end: 3 }), clip({ id: 'c', end: 1.5 })]);
r = sbeMoveTo(cs, 'c', 0);
out.moveClamped = shape(r.clips);                  // cannot pass b: stays at 5
cs = lay([clip({ id: 'a', end: 2 }), clip({ id: 'b', end: 3, film_start: 4, film_end: 7 }), clip({ id: 'c', end: 1 })]);
r = sbeMoveTo(cs, 'b', 3);                         // slides back into its own hole
out.moveKeepsNeighbours = shape(r.clips);

cs = lay([clip({ id: 'a', end: 2 }), clip({ id: 'b', end: 3 })]);
r = sbeMoveTo(cs, 'b', 6);       // dropped into open air: the hole is kept
out.moveOpensHole = shape(r.clips);

cs = lay([clip({ id: 'a', end: 2, locked: true }), clip({ id: 'b', end: 3 })]);
r = sbeMoveTo(cs, 'a', 9);
out.moveLocked = { ok: r.ok, why: r.why, shape: shape(r.clips) };

// A locked clip is an ANCHOR: the flow goes around it, never through it.
cs = lay([clip({ id: 'a', end: 2 }), clip({ id: 'L', end: 2, film_start: 3, film_end: 5, locked: true }),
          clip({ id: 'b', end: 4 })]);
out.anchored = shape(cs);
out.anchoredNoOverlap = !overlaps(cs);

// ---- trim ----------------------------------------------------------------
cs = lay([clip({ id: 'a', end: 2 }), clip({ id: 'b', end: 3 })]);
sbeTrim(cs, 'a', 'r', 1.25, { ripple: true });   // ⌘: length changes, tail ripples
out.trimRight = shape(cs);
out.trimRightLengths = lengthsAgree(cs);
// DEFAULT: the right handle opens a hole; the next clip does not move.
cs = lay([clip({ id: 'a', end: 2 }), clip({ id: 'b', end: 3 })]);
sbeTrim(cs, 'a', 'r', 1.25);
out.trimRightHole = shape(cs);
out.trimRightHoleLengths = lengthsAgree(cs);
// ...and it may grow back into that hole but never into the neighbour.
sbeTrim(cs, 'a', 'r', 5);
out.trimRightStopsAtNext = shape(cs);

cs = lay([clip({ id: 'a', end: 2 }), clip({ id: 'b', end: 3 })]);
sbeTrim(cs, 'b', 'l', 2.5);       // left handle: in-point moves, tail stays put
out.trimLeft = shape(cs);
out.trimLeftLengths = lengthsAgree(cs);

cs = lay([clip({ id: 'a', end: 2, duration: 2.4 })]);
sbeTrim(cs, 'a', 'r', 99);        // cannot pull past the end of the source
out.trimClampedToSource = shape(cs);
cs = lay([clip({ id: 'a', end: 2 })]);
sbeTrim(cs, 'a', 'l', 99);        // nor collapse the clip to nothing
out.trimClampedToMin = shape(cs);
cs = lay([clip({ id: 'a', end: 2, locked: true })]);
out.trimLocked = sbeTrim(cs, 'a', 'r', 1).why;

// ---- ripple + split ------------------------------------------------------
cs = lay([clip({ id: 'a', end: 2 }), clip({ id: 'b', end: 3 }), clip({ id: 'c', end: 1.5 })]);
r = sbeRippleDelete(cs, 'b');
out.ripple = shape(r.clips);
out.rippleLengths = lengthsAgree(r.clips);
cs = lay([clip({ id: 'a', end: 2, locked: true })]);
out.rippleLocked = sbeRippleDelete(cs, 'a').why;

// ---- lift: the shot leaves, ITS HOLE STAYS, nothing downstream moves ------
cs = lay([clip({ id: 'a', end: 2 }), clip({ id: 'b', end: 3 }), clip({ id: 'c', end: 1.5 })]);
out.liftWindowsBefore = cs.map(c => [c.id, c.film_start, c.film_end]);
r = sbeLiftDelete(cs, 'b');
out.liftWindows = r.clips.map(c => [c.id, c.film_start, c.film_end]);
out.liftKinds = r.clips.map(c => sbeKind(c));
out.liftLengths = lengthsAgree(r.clips);
out.liftHole = (() => { const h = sbeById(r.clips, 'b'); return [h.path, h.proxy, h.source]; })();
out.liftLocked = sbeLiftDelete(lay([clip({ id: 'a', end: 2, locked: true })]), 'a').why;
out.liftTwice = sbeLiftDelete(r.clips, 'b').why;

cs = lay([clip({ id: 'a', end: 2 }), clip({ id: 'b', end: 3 })]);
r = sbeSplitAt(cs, 1.2, 'a2');
out.split = shape(r.clips);
out.splitLengths = lengthsAgree(r.clips);
out.splitHuman = r.clips.map(c => c.source);
out.splitTooClose = sbeSplitAt(lay([clip({ id: 'a', end: 2 })]), 0.05).ok;
out.splitInHole = sbeSplitAt(holed, 3.5).ok;

// ---- place an unplaced clip ---------------------------------------------
cs = lay([clip({ id: 'a', end: 2, film_start: 0, film_end: 2 }),
          clip({ id: 'b', end: 2, film_start: 6, film_end: 8 })]);
r = sbePlaceUnplaced(cs, { path: '/o/new.mp4', duration_s: 3, n: 9, title: 'new',
                           slot: { film_start: 2.5, duration: 3 } }, 2.5);
out.placed = r.clips.map(c => [c.path, c.film_start, c.film_end, c.source]);
out.placedNoOverlap = !overlaps(r.clips);
// A shot generated for a hole must not push the film that follows it: the
// cuts after the hole are on beats they were chosen for.
cs = lay([clip({ id: 'a', end: 2, film_start: 0, film_end: 2 }),
          clip({ id: 'b', end: 2, film_start: 9, film_end: 11 })]);
r = sbePlaceUnplaced(cs, { path: '/o/fill.mp4', duration_s: 3,
                           slot: { film_start: 3, duration: 3 } }, 3);
out.filled = r.clips.map(c => [c.path.split('/').pop(), c.film_start, c.film_end]);
// ...but when the new clip is too long for the hole, the tail rides along
// rather than overlapping.
cs = lay([clip({ id: 'a', end: 2, film_start: 0, film_end: 2 }),
          clip({ id: 'b', end: 2, film_start: 5, film_end: 7 })]);
r = sbePlaceUnplaced(cs, { path: '/o/big.mp4', duration_s: 4,
                           slot: { film_start: 2, duration: 4 } }, 2);
out.overfilled = r.clips.map(c => [c.path.split('/').pop(), c.film_start, c.film_end]);
out.overfilledNoOverlap = !overlaps(r.clips);

// ---- undo / redo ---------------------------------------------------------
SBE.clips = lay([clip({ id: 'a', end: 2 }), clip({ id: 'b', end: 3 })]);
SBE.undo = []; SBE.redo = []; SBE.dirty = false;
sbeMutate(x => sbeRippleDelete(x, 'a'));
out.afterMutate = shape(SBE.clips);
out.dirtyAfterMutate = SBE.dirty;
sbeUndo();
out.afterUndo = shape(SBE.clips);
sbeRedo();
out.afterRedo = shape(SBE.clips);
sbeUndo(); sbeUndo();            // one step of history, two presses
out.undoFloor = shape(SBE.clips);
// A refused mutation writes NO undo step — otherwise ⌘Z would undo nothing.
SBE.clips = lay([clip({ id: 'a', end: 2, locked: true })]);
SBE.undo = []; SBE.redo = [];
sbeMutate(x => sbeRippleDelete(x, 'a'));
out.refusedNoUndo = SBE.undo.length;

// ---- save payload --------------------------------------------------------
const body = sbeSaveBody({
  id: 'sb_t', expect: 4,
  edit: { version: 1, revision: 4, audio: { path: '/m.wav' }, beats: beats,
          settings: { min_shot: 1.5 } },
  clips: lay([clip({ id: 'a', end: 2, source: 'human' })]),
});
out.saveBody = body;
out.saveStrippedPrivate = Object.keys(body.edit.clips[0]).filter(k => k.charAt(0) === '_');
out.saveNoExpect = sbeSaveBody({ id: 'x', edit: {}, clips: [], expect: null });

// ---- errors map onto clips ----------------------------------------------
out.errMap = sbeErrorsByClip(
  [{ code: 'clip_window', where: 1, message: 'clip 2: end must be after start' },
   { code: 'version', where: null, message: 'edit version 9' }],
  ['a', 'b']);

// ---- 409, and the arrangement that survives it --------------------------
SBE.clips = lay([clip({ id: 'a', end: 2 }), clip({ id: 'b', end: 3 })]);
SBE.dirty = true; SBE.revision = 3; SBE.saving = false; SBE.open = true;
NEXT = { status: 409, body: { ok: false, conflict: true, revision: 7,
                              error: 'this timeline moved on without you' } };
out.saveConflict = await sbeSave(true);
out.afterConflict = {
  clips: shape(SBE.clips), dirty: SBE.dirty, conflict: SBE.conflict,
  revision: SBE.revision, banner: !els.sbeConflict.hidden,
};
// "Keep mine" re-sends WITHOUT expect_revision — the only way to overwrite, and
// only because a human asked for it in as many words.
FETCHES = [];
NEXT = { status: 200, body: { ok: true, edit: { revision: 8, clips: [] }, unplaced: [] } };
out.forced = await sbeSave(true, true);
out.forcedBody = JSON.parse(FETCHES[0].body);
out.afterForced = { revision: SBE.revision, dirty: SBE.dirty };

// A 400 keeps the work AND says which clip is wrong.
SBE.clips = lay([clip({ id: 'a', end: 2 }), clip({ id: 'b', end: 3 })]);
SBE.dirty = true; SBE.conflict = 0;
NEXT = { status: 400, body: { ok: false, error: 'clip 2: no path',
                              errors: [{ code: 'clip_path', where: 1, message: 'clip 2: no path' }] } };
out.saveInvalid = await sbeSave(true);
out.afterInvalid = {
  clips: shape(SBE.clips), dirty: SBE.dirty,
  flagged: Object.keys(SBE.errors.byId || {}),
};

// A good save keeps the clips ON SCREEN — the server's normalised copy of the
// document is adopted, its clip array is not, because the user may have moved
// something while the request was in flight.
SBE.clips = lay([clip({ id: 'a', end: 2 })]);
SBE.dirty = true; SBE.revision = 8;
NEXT = { status: 200, body: { ok: true, edit: { revision: 9, clips: [{ id: 'z' }] },
                              unplaced: [{ path: '/o/z.mp4' }] } };
out.saveOk = await sbeSave(true);
out.afterOk = { clips: shape(SBE.clips), dirty: SBE.dirty, revision: SBE.revision,
                unplaced: SBE.unplaced.length };

out.fmt = [sbeFmtTime(0), sbeFmtTime(9.5), sbeFmtTime(75.25)];

// =========================================================================
// WAVE 2 — the three kinds, one adjustment, and the drop maths
// =========================================================================

// ---- kind: absent is video, and that IS the v1 migration -----------------
out.kinds = [sbeKind({}), sbeKind({ kind: 'still' }), sbeKind({ kind: 'slug' }),
             sbeKind({ kind: 'nonsense' }), sbeKind(null)];

// ---- brightness: clamped, and neutral means ABSENT -----------------------
out.bright = [sbeBright({}), sbeBright({ adjust: {} }),
              sbeBright({ adjust: { brightness: 0.2 } }),
              sbeBright({ adjust: { brightness: 9 } }),
              sbeBright({ adjust: { brightness: -9 } }),
              sbeBright({ adjust: { brightness: 'x' } })];
// CSS brightness() is multiplicative, ffmpeg's eq is additive; matched at
// mid-grey, which is where a person judges exposure.
out.brightCss = [sbeBrightnessCss(0), sbeBrightnessCss(0.25),
                 sbeBrightnessCss(0.5), sbeBrightnessCss(-0.5),
                 sbeBrightnessCss(-9)];
out.brightCssMidGrey = [0.5 + 0.25, 0.5 * sbeBrightnessCss(0.25)];

cs = lay([clip({ id: 'a', end: 2 })]);
out.setBrightOk = sbeSetBrightness(cs, 'a', 0.3).ok;
out.setBrightValue = sbeBright(sbeById(cs, 'a'));
out.setBrightHuman = sbeById(cs, 'a').source;
out.setBrightClamped = (sbeSetBrightness(cs, 'a', 4), sbeBright(sbeById(cs, 'a')));
// Back to zero leaves NO adjust key at all — an untouched clip and a clip
// dragged back to neutral must serialise identically.
sbeSetBrightness(cs, 'a', 0);
out.setBrightCleared = ('adjust' in sbeById(cs, 'a'));
out.setBrightNoop = sbeSetBrightness(cs, 'a', 0).ok;
out.setBrightGone = sbeSetBrightness(cs, 'zz', 0.2).why;

// ---- the save payload carries the kinds, synthesised the server's way ----
const kindBody = sbeSaveBody({
  id: 'sb_t', expect: null, edit: { version: 2 },
  clips: lay([
    clip({ id: 'v', end: 2, adjust: { brightness: 0.25 } }),
    clip({ id: 's', kind: 'still', path: '/img/card.png', end: 3, duration: 99 }),
    clip({ id: 'k', kind: 'slug', path: null, end: 1.5, adjust: { brightness: 0 } }),
  ]),
});
out.kindPayload = kindBody.edit.clips.map(c => ({
  id: c.id, kind: c.kind, path: c.path, start: c.start, end: c.end,
  dur: c.duration, adjust: c.adjust,
}));

// ---- where a drop lands: the midpoint rule ------------------------------
cs = lay([clip({ id: 'a', end: 2 }), clip({ id: 'b', end: 2 }),
          clip({ id: 'c', end: 2 })]);
out.dropIdx = [sbeDropIndex(cs, 0), sbeDropIndex(cs, 0.9), sbeDropIndex(cs, 1.1),
               sbeDropIndex(cs, 2.9), sbeDropIndex(cs, 3.1), sbeDropIndex(cs, 99)];
out.dropIdxEmpty = sbeDropIndex([], 5);

// ---- insert RIPPLES (unlike filling a hole, which must not) -------------
cs = lay([clip({ id: 'a', end: 2 }), clip({ id: 'b', end: 2 })]);
r = sbeInsertAt(cs, { path: '/o/n.mp4', duration_s: 1, title: 'n' }, 2.1);
out.insertRipple = r.clips.map(c => [c.id === r.added.id ? 'NEW' : c.id,
                                     c.film_start, c.film_end]);
out.insertLengths = lengthsAgree(r.clips);
out.insertNoOverlap = !overlaps(r.clips);
out.insertHuman = r.added.source;

// A slug has no path and no source duration, and gets both right.
cs = lay([clip({ id: 'a', end: 2 })]);
r = sbeInsertAt(cs, { kind: 'slug', title: 'black', duration_s: 2.5 }, 99);
out.insertSlug = { kind: r.added.kind, path: r.added.path, dur: r.added.duration,
                   len: sbeRound(sbeLen(r.added)),
                   at: [r.added.film_start, r.added.film_end] };
out.insertSlugNoOverlap = !overlaps(r.clips);
r = sbeInsertAt(lay([clip({ id: 'a', end: 2 })]),
                { kind: 'still', path: '/img/c.png', duration_s: 3 }, 0);
out.insertStill = { kind: r.added.kind, path: r.added.path, dur: r.added.duration,
                    index: r.index, at: [r.added.film_start, r.added.film_end] };

// A still is resized by the SAME trim machinery every other block uses, and
// nothing clamps it, because it has no source clock to run out of.
cs = lay([r.added, clip({ id: 'a', end: 2 })]);
sbeTrim(cs, r.added.id, 'r', 9, { ripple: true });   // ⌘: grow past the neighbour, which slides
out.stillStretched = sbeRound(sbeLen(sbeById(cs, r.added.id)));

// ---- reorder: closes the hole it leaves, opens none where it lands -------
cs = lay([clip({ id: 'a', end: 2 }), clip({ id: 'b', end: 3 }),
          clip({ id: 'c', end: 1 })]);
r = sbeReorderTo(cs, 'c', 0);
out.reorder = shape(r.clips);
out.reorderIds = r.clips.map(c => c.id);
out.reorderLengths = lengthsAgree(r.clips);
out.reorderNoOverlap = !overlaps(r.clips);
out.reorderKeepsLength = sbeFilmDuration(r.clips);
out.reorderHuman = sbeById(r.clips, 'c').source;
out.reorderLocked = sbeReorderTo(lay([clip({ id: 'a', end: 2, locked: true })]),
                                 'a', 9).why;
out.reorderGone = sbeReorderTo(cs, 'nope', 0).why;
// Contrast with a MOVE, which leaves the hole behind — both verbs exist and
// this is the difference between them.
cs = lay([clip({ id: 'a', end: 2 }), clip({ id: 'b', end: 2 }),
          clip({ id: 'c', end: 2 })]);
out.moveLeavesAHole = sbeMoveTo(cs, 'a', 9, { ripple: true }).clips.map(c => [c.id, c.film_start]);

// ---- WAVE 3: the two sliders ---------------------------------------------
// The owner's film, measured: 71.583338s in a 1108px-wide scroller.
const FILM = 71.583338;
const VIEW = 1108;
out.fit = sbeZoomFitPps(FILM, VIEW);
// At the fit scale the whole film is inside the box — that is what makes the
// slider's left end mean "all of it". sbePaint pads the inner by SBE_TL_PAD.
out.fitWidth = Math.max(320, Math.ceil(FILM * out.fit) + SBE_TL_PAD);
out.fitFitsTheBox = out.fitWidth <= VIEW + 1;
// sbeSpan() never returns less than 1, so this is the shortest film the fit
// is ever asked about — and it is still a number, not an infinity.
out.fitEmptyFilm = sbeZoomFitPps(1, VIEW);
// A box with no width yet (the tab has never been painted) must not produce
// a zero or a NaN that would then be stored as SBE.pps.
out.fitNarrow = sbeZoomFitPps(FILM, 0);

// The slider is logarithmic and round-trips.
out.sliderEnds = [sbeZoomFromSlider(0, 15.14, 200), sbeZoomFromSlider(1000, 15.14, 200)];
out.sliderRound = [0, 250, 500, 750, 1000].map(v =>
  sbeZoomToSlider(sbeZoomFromSlider(v, 15.14, 200), 15.14, 200));
out.sliderClamps = [sbeZoomToSlider(1, 15.14, 200), sbeZoomToSlider(9999, 15.14, 200)];
// Linear would put the whole useful range in the first fifth of the travel.
out.sliderMidIsNotArithmeticMean = sbeZoomFromSlider(500, 15.14, 200);

// ZOOM KEEPS THE PLAYHEAD STILL. The head is at 36s, the view is scrolled so
// it sits 400px in; after a zoom it must still sit 400px in.
let anchor = sbeZoomAnchor(36, 36 * 42 - 400, VIEW, 42);
out.anchorOnPlayhead = [Math.round(anchor.t * 1000) / 1000, Math.round(anchor.px)];
out.zoomHoldsThePlayhead = Math.round(36 * 96 - sbeZoomScroll(anchor, 96, 1e9));
// Off screen, it holds the middle of the view instead — there is no frame on
// screen to anchor on, and lurching to one that is not visible is worse.
anchor = sbeZoomAnchor(2, 30000, VIEW, 42);
out.anchorFallsBackToTheMiddle = [Math.round(anchor.t * 100) / 100, Math.round(anchor.px)];
// alt + wheel names its own anchor: the film time under the pointer.
anchor = sbeZoomAnchor(36, 1000, VIEW, 42, 30);
out.anchorExplicit = [anchor.t, Math.round(anchor.px)];
// The scroll is clamped to what there is: no negative scroll at the head of
// the film, no scrolling past the end.
out.zoomScrollClamps = [sbeZoomScroll({ t: 0, px: 400 }, 42, 5000),
                        sbeZoomScroll({ t: 1000, px: 0 }, 42, 5000)];

// ---- WAVE 3: follow, which PAGES rather than chases ----------------------
// On screen: the view does not move at all. This is the early return that
// keeps the follow from writing scrollLeft sixty times a second.
out.followStaysPut = sbeFollowScroll(500, 0, VIEW, 4000);
// Crossed the right edge: ONE jump, and the head lands near the left of the
// new screenful rather than in the middle of it.
out.followPages = sbeFollowScroll(1680, 0, VIEW, 4000);
out.followLead = 1680 - out.followPages;
// The frame after that, the head is on screen again and nothing moves.
out.followSettles = sbeFollowScroll(1700, out.followPages, VIEW, 4000);
out.followSettled = out.followSettles === out.followPages;
// Seeking backwards out of view pages the other way, and both ends clamp.
out.followBack = sbeFollowScroll(100, 3000, VIEW, 4000);
out.followClampsHead = sbeFollowScroll(-50, 3000, VIEW, 4000);
out.followClampsTail = sbeFollowScroll(99999, 0, VIEW, 4000);

// ---- WAVE 3: the monitor row ---------------------------------------------
// Both measured live, in the Editor's own tab, with the carwash film open.
const A = 16 / 9;
const near = (a, b) => Math.abs(a - b) <= 1.5;
// 1440x900: the column is 1110px wide and 307px of height are free.
let fit = sbeMonitorFit(1110, 307);
out.fit1440 = {
  fills: near(fit.total, 1110),
  prog: [Math.round(fit.progW), Math.round(fit.progH)],
  src: [Math.round(fit.srcW), Math.round(fit.srcH)],
  rail: Math.round(fit.rail),
  progAspect: Math.round(fit.progW / fit.progH * 1000) / 1000,
  srcAspect: Math.round(fit.srcW / fit.srcH * 1000) / 1000,
  split: Math.round(fit.srcW / (fit.srcW + fit.progW) * 100),
  programIsBigger: fit.progW > fit.srcW,
  withinBudget: fit.progH <= 307 + 0.5,
};
// 1900x1000: 1570px wide, 446px free. Same shape, bigger.
fit = sbeMonitorFit(1570, 446);
out.fit1900 = {
  fills: near(fit.total, 1570),
  prog: [Math.round(fit.progW), Math.round(fit.progH)],
  src: [Math.round(fit.srcW), Math.round(fit.srcH)],
  rail: Math.round(fit.rail),
  split: Math.round(fit.srcW / (fit.srcW + fit.progW) * 100),
  programIsBigger: fit.progW > fit.srcW,
  withinBudget: fit.progH <= 446 + 0.5,
};
// A TALL, NARROW WINDOW. The width runs out first, so the pair takes the
// width-derived height at exactly 40/60 and the leftover HEIGHT is left for
// the timeline to absorb — the row never eats more than it can fill.
fit = sbeMonitorFit(1110, 900);
out.fitTall = {
  fills: near(fit.total, 1110), ratio: Math.round(fit.ratio * 1000) / 1000,
  progH: Math.round(fit.progH), leftoverHeight: Math.round(900 - fit.progH),
  rail: Math.round(fit.rail),
};
// A VERY SHORT WINDOW. The source widens to keep the row full, and stops at
// equal monitors: the program is the one being cut and never becomes the
// smaller of the two.
fit = sbeMonitorFit(1900, 180);
out.fitShort = {
  ratio: Math.round(fit.ratio * 1000) / 1000,
  programIsBigger: fit.progW >= fit.srcW,
  rail: Math.round(fit.rail),
};
// The floor: a budget of nothing still leaves a picture worth the name.
fit = sbeMonitorFit(1110, 0);
out.fitFloor = Math.round(fit.progH);
// The rail is leftover, never a panel of its own: clamped at both ends.
out.railClamps = [Math.round(sbeMonitorFit(3000, 200).rail),
                  Math.round(sbeMonitorFit(700, 200).rail)];

// ---- Wave 4: the soundtrack is an object ---------------------------------
const MUS = (o) => Object.assign({ path: '/o/track.wav', duration: 60 }, o);
let mw = sbeMusicWindow(MUS({ offset: 0 }), 60);
out.musicPlain = [mw.head, mw.tail, mw.film_start, mw.film_end, mw.trimmed];
// A POSITIVE OFFSET IS A HEAD TRIM: track second 5 plays at film 0, so the
// block still starts at film 0 and ends five seconds earlier than the track.
mw = sbeMusicWindow(MUS({ offset: 5 }), 60);
out.musicOffsetIn = [mw.head, mw.film_start, mw.film_end];
// A NEGATIVE OFFSET IS THE DIRECTION THE OLD CLAMP MADE UNREACHABLE: the
// music starts four seconds INTO the film, from its own first second.
mw = sbeMusicWindow(MUS({ offset: -4 }), 60);
out.musicOffsetOut = [mw.head, mw.film_start, mw.film_end];
// Trims are track seconds, and a head trim does NOT slide the rest earlier.
mw = sbeMusicWindow(MUS({ offset: 0, trim_start: 10, trim_end: 20 }), 60);
out.musicTrimmed = [mw.head, mw.tail, mw.film_start, mw.film_end, mw.trimmed];
// Absent trims, an absent duration, an absent audio: no block, no crash.
out.musicNone = (() => { const w = sbeMusicWindow(null, 0); return [w.duration, w.tail]; })();

// DRAG. Moving writes the offset and NOTHING else — the window into the track
// is what a move must not touch.
let ed = sbeMusicEdit(MUS({ offset: 0, trim_start: 10, trim_end: 20 }), 'move', 7, 60);
out.musicMove = [ed.offset, ed.trim_start, ed.trim_end];
out.musicMoveLands = sbeMusicWindow(MUS(ed), 60).film_start;
// Dragging it back to the head of the film.
ed = sbeMusicEdit(MUS({ offset: -8 }), 'move', 0, 60);
out.musicMoveHome = ed.offset;
// TRIM LEFT: a new in-point, and the rest of the track stays where it was.
ed = sbeMusicEdit(MUS({ offset: 0 }), 'trimL', 12, 60);
out.musicTrimL = [ed.offset, ed.trim_start, ed.trim_end];
out.musicTrimLKeepsPlace = sbeMusicWindow(MUS(ed), 60).film_start;
// ...and it can never cross its own out-point, or go before the track starts.
ed = sbeMusicEdit(MUS({ offset: 0, trim_end: 6 }), 'trimL', 99, 60);
out.musicTrimLClamped = ed.trim_start;
ed = sbeMusicEdit(MUS({ offset: 0, trim_start: 20 }), 'trimL', 0, 60);
out.musicTrimLFloor = ed.trim_start;
// TRIM RIGHT: bounded by the track's own length and by the in-point.
ed = sbeMusicEdit(MUS({ offset: 0 }), 'trimR', 42, 60);
out.musicTrimR = ed.trim_end;
ed = sbeMusicEdit(MUS({ offset: 0 }), 'trimR', 999, 60);
out.musicTrimRClamped = ed.trim_end;
ed = sbeMusicEdit(MUS({ offset: 0, trim_start: 30 }), 'trimR', 0, 60);
out.musicTrimRFloor = ed.trim_end;

// THE EDGES SNAP TO CUTS, NOT TO BEATS — the grid is derived from this track
// at this offset, so snapping the music to it would be circular.
out.musicSnaps = sbeMusicSnaps(lay([clip({ id: 'a' }), clip({ id: 'b' })]));
out.musicSnapCatches = sbeSnapToList(2.04, [0, 2, 4], 0.2, true);
out.musicSnapIgnoresFar = sbeSnapToList(3.0, [0, 2, 4], 0.2, true);
out.musicSnapOverride = sbeSnapToList(2.04, [0, 2, 4], 0.2, false);

// ---- Wave 4: the timeline does not end where the clips end ---------------
SBE.clips = lay([clip({ id: 'a' }), clip({ id: 'b' })]);   // 4 s of film
SBE.audio = null; SBE.peaks = null;
out.spanShortFilm = sbeSpan();
// An EMPTY timeline still has a ruler worth reading.
SBE.clips = [];
out.spanEmpty = sbeSpan();
// The music past the last clip drags the scroller out with it.
SBE.clips = lay([clip({ id: 'a' }), clip({ id: 'b' })]);
SBE.audio = { path: '/o/track.wav', duration: 30, offset: -20 };
out.spanFollowsMusic = sbeSpan();
// ...and the slack is capped, so a long film does not open on a minute of
// nothing.
SBE.audio = null;
SBE.clips = lay([clip({ id: 'a', start: 0, end: 300, duration: 400 })]);
out.spanLongFilm = sbeSpan();
SBE.clips = []; SBE.audio = null;

// ---- Wave 4: one undo step, whatever moved -------------------------------
SBE.clips = lay([clip({ id: 'a' })]);
SBE.edit = { audio: { path: '/o/track.wav', offset: 0, duration: 60 } };
SBE.audio = SBE.edit.audio;
const snap = sbeSnapshot();
SBE.edit.audio = { path: '/o/track.wav', offset: -6, duration: 60 };
SBE.audio = SBE.edit.audio;
sbeRestore(snap);
out.undoRestoresMusic = SBE.audio.offset;
// The legacy shape — a bare array of clips — is still accepted, because a
// clip drag takes its own snapshot and has no audio to carry.
sbeRestore(JSON.stringify([clip({ id: 'z' })]));
out.undoLegacyShape = [SBE.clips.length, SBE.clips[0].id, SBE.audio.offset];
SBE.clips = []; SBE.edit = {}; SBE.audio = null;

// ---- Wave 4: the versions picker's wording -------------------------------
const NOW = 1000000;
out.ago = [sbeAgo(NOW - 5, NOW), sbeAgo(NOW - 300, NOW),
           sbeAgo(NOW - 7200, NOW), sbeAgo(NOW - 400000, NOW),
           sbeAgo(0, NOW)];
out.lineAuto = sbeVersionLine(
  { file: 'edit-r00007.json', label: '', revision: 7, clips: 9,
    duration: 39.71, archived_at: NOW - 300, kept: false, readable: true }, NOW);
out.lineKept = sbeVersionLine(
  { file: 'keep-r00007-good.json', label: 'the good one', revision: 7,
    clips: 1, duration: 2, archived_at: NOW - 60, kept: true,
    readable: true }, NOW);
// AN UNREADABLE FILE IS STILL A ROW. It vanishing would be indistinguishable
// from never having been written.
out.lineBad = sbeVersionLine(
  { file: 'edit-r00001.json', revision: null, clips: null, duration: null,
    archived_at: NOW - 10, kept: false, readable: false }, NOW);
// THE FALLBACK FOLLOWS THE LANE. An unnamed row in the lane headed YOUR SAVES
// OF THIS DRAFT is a save the user pressed, and calling it "autosave" is the
// one word that means it was not.
out.lineMine = sbeVersionLine(
  { file: 'save-r00091.json', label: '', revision: 91, clips: 9,
    duration: 46.7, archived_at: NOW - 20, kept: false, manual: true,
    readable: true }, NOW);

// ---- THE SAVE THAT CANNOT BE DROPPED, AND THE FAILURE THAT SCREAMS -------
// The owner cut for twenty minutes against a timeline nothing was writing.
// Everything below is that incident, turned into arithmetic.
SBE.clips = lay([clip({ id: 'a' })]);
SBE.edit = { clips: SBE.clips }; SBE.audio = null;
SBE.dirty = false; SBE.conflict = 0; SBE.saving = false;
SBE.savePending = false; SBE.dirtyAt = 0; SBE.saveFailed = '';
SBE.saveTimer = null; SBE.revision = 1; SBE.sel = '';
els.sbeAlarm = { hidden: true };
els.sbeAlarmWhy = { textContent: '' };

// A SAVE ARRIVING MID-FLIGHT IS REMEMBERED, NOT THROWN AWAY. This is the
// dropped-save bug: `if (SBE.saving) return` left dirty true with no timer
// pending, so that edit was never written again.
SBE.saving = true;
await sbeSave(true);
out.midFlightRemembered = SBE.savePending;
SBE.savePending = false;
await sbeSave(false);                      // the BUTTON, not the debounce
out.midFlightRemembersASave = SBE.savePending;
SBE.saving = false; SBE.savePending = false;

// THE FLAG CANNOT STICK. A throw between "saving = true" and the fetch used
// to wedge the editor for the rest of the session, silently.
const realBody = sbeSaveBody;
sbeSaveBody = () => { throw new Error('boom'); };
SBE.dirty = true;
await sbeSave(true);
sbeSaveBody = realBody;
out.flagClearedAfterThrow = SBE.saving;
out.throwRaisesAlarm = [!!SBE.saveFailed, els.sbeAlarm.hidden,
                        (states[states.length - 1] || [])[1]];

// A QUIET AUTOSAVE THAT FAILS IS STILL LOUD. `if (!quiet)` is what made the
// twenty minutes possible.
SBE.saveFailed = ''; els.sbeAlarm.hidden = true;
SBE.dirty = true; SBE.saving = false;
NEXT = { status: 500, body: { ok: false, error: 'disk is full' } };
await sbeSave(true);
out.quietFailureIsLoud = [!!SBE.saveFailed, els.sbeAlarm.hidden,
                          els.sbeAlarmWhy.textContent,
                          (states[states.length - 1] || [])[1]];

// A CONFLICT IS A FAILURE TO STORE, not a note.
SBE.saveFailed = ''; els.sbeAlarm.hidden = true; SBE.dirty = true;
NEXT = { status: 409, body: { ok: false, conflict: true, revision: 9 } };
await sbeSave(true);
out.conflictIsLoud = [!!SBE.saveFailed, els.sbeAlarm.hidden];

// ...and a save that LANDS takes the alarm down with it and stops the clock.
SBE.conflict = 0; SBE.dirty = true; SBE.dirtyAt = 1;
NEXT = { status: 200, body: { ok: true, edit: { revision: 12, clips: [] } } };
await sbeSave(true);
out.successClearsAlarm = [!!SBE.saveFailed, els.sbeAlarm.hidden, SBE.dirtyAt,
                          SBE.dirty, (states[states.length - 1] || [])[1]];
SBE.dirty = false; SBE.dirtyAt = 0; SBE.saveFailed = '';

// ---- THE CRASH LANE, DRIVEN AS A SEQUENCE --------------------------------
// save → backup → recovery, in that ORDER. Every defect this lane has had
// lived between two of those steps rather than inside any one of them, and a
// suite that only asserts each guard exists is exactly blind to that: the
// guard that refuses to overwrite an unanswered offer was correct, and it
// killed the whole lane for the session because nothing ever answered.
SBE.backup = { at: 1, clips: 3, duration: 6 };
SBE.backedUpAt = 0; SBE.dirty = true;
FETCHES = [];
NEXT = { status: 200, body: { ok: true } };
// THE LANE NEVER STOPS. It used to refuse while an offer was unanswered,
// because there was one backup file per draft and a new write would have
// destroyed the work the offer held. The lane is versioned now, so there is
// nothing left to guard — and a chip nobody dismissed no longer switches the
// safety net off for the rest of the session.
NEXT = { status: 200, body: { ok: true } };
out.backupWritesEvenWithAnOfferOpen = [!!(await sbeBackup(true)),
                                       FETCHES.length, SBE.backedUpAt > 0];
// Save IS the answer. The server drops the file and hands back a payload with
// no offer in it; the client follows the server rather than deciding.
SBE.dirty = true; SBE.dirtyAt = 5;
const paintsBefore = recoveryPaints.length;
NEXT = { status: 200, body: { ok: true, edit: { revision: 13, clips: [] },
                              backup: null,
                              drafts: [{ slug: 'draft-1', name: 'Draft 1',
                                         active: true, clips: 1 }] } };
await sbeSave(true);
out.saveAnswersTheOffer = [SBE.backup, recoveryPaints.length > paintsBefore,
                           (SBE.drafts || []).length];
// ...and the lane is alive again: the very next backup actually WRITES.
SBE.dirty = true;
FETCHES = [];
NEXT = { status: 200, body: { ok: true } };
out.backupLivesAfterASave = [!!(await sbeBackup(true)), FETCHES.length,
                             SBE.backedUpAt > 0];
out.backupNamesItsDraft = JSON.parse(FETCHES[0].body || '{}').draft;
// A backup that does not land is the safety net gone, and only the alarm says so.
SBE.dirty = true;
NEXT = { status: 500, body: { ok: false, error: 'disk is full' } };
out.backupFailureIsLoud = [await sbeBackup(true), !!SBE.saveFailed];
SBE.saveFailed = ''; els.sbeAlarm.hidden = true;
SBE.dirty = false; SBE.dirtyAt = 0; SBE.backup = null; SBE.drafts = [];

// ---- J-cuts and L-cuts ---------------------------------------------------
const AC = () => lay([clip({ id: 'a', start: 0, end: 4, duration: 10 }),
                      clip({ id: 'b', start: 0, end: 4, duration: 10 })]);
let ac = AC();
out.audioLinkedByDefault = (() => {
  const w = sbeClipAudio(ac[1]);
  return [w.start, w.end, w.film_start, w.linked];
})();
// UNLINKING IS NOT AN EDIT. It writes the window the clip already had.
let acr = sbeSetAudioLink(ac, 'b', false);
out.unlinkWritesTheSameWindow = [acr.ok, JSON.stringify(sbeById(acr.clips, 'b').audio)];
// ...and re-linking removes the field entirely.
out.relinkRemovesTheField = sbeById(sbeSetAudioLink(acr.clips, 'b', true).clips, 'b').audio || null;
// A LINKED CLIP REFUSES TO BE DRAGGED — the accident the default prevents.
out.linkedRefusesTheDrag = (() => {
  const q = sbeAudioEdit(AC(), 'b', 'move', 2);
  return [q.ok, q.why];
})();
// THE J-CUT: her voice starts a second before we see her, and the PICTURE
// does not move.
ac = sbeSetAudioLink(AC(), 'b', false).clips;
let j = sbeAudioEdit(ac, 'b', 'move', 3);
out.jCut = [j.ok, sbeById(j.clips, 'b').audio.film_start,
            sbeById(j.clips, 'b').film_start,
            sbeById(j.clips, 'b').audio.start];
// THE L-CUT: his line runs on under her picture — the tail is extended
// within the source it came from.
let l = sbeAudioEdit(sbeSetAudioLink(AC(), 'a', false).clips, 'a', 'trimR', 6);
out.lCut = [l.ok, sbeById(l.clips, 'a').audio.end,
            sbeById(l.clips, 'a').end, sbeById(l.clips, 'a').film_end];
// ...and it cannot run past the source clip.
let over = sbeAudioEdit(sbeSetAudioLink(AC(), 'a', false).clips, 'a', 'trimR', 99);
out.lCutClamped = sbeById(over.clips, 'a').audio.end;
// A HEAD TRIM MOVES THE IN-POINT AND LEAVES THE REST WHERE IT IS.
let h = sbeAudioEdit(sbeSetAudioLink(AC(), 'b', false).clips, 'b', 'trimL', 5);
out.headTrim = [sbeById(h.clips, 'b').audio.start,
                sbeById(h.clips, 'b').audio.film_start,
                sbeById(h.clips, 'b').start];
// A LOCKED CLIP'S SOUND IS LOCKED TOO.
let lk = sbeSetAudioLink(AC(), 'b', false).clips;
sbeById(lk, 'b').locked = true;
out.lockedRefuses = sbeAudioEdit(lk, 'b', 'move', 2).ok;
// A STILL HAS NO SOUND TO UNLINK.
out.stillRefuses = (() => {
  const cs = lay([clip({ id: 's', kind: 'still' })]);
  const q = sbeSetAudioLink(cs, 's', false);
  return [q.ok, q.why];
})();

// ---- WAVEFORMS, KEYFRAMES, AND DELETING HALF A CLIP ---------------------
// A synthetic take: one bucket per 10th of a second, a ramp so every slice is
// distinguishable from every other.
const PEAKS = (() => {
  const n = 100, arr = [];
  for (let i = 0; i < n; i++) { arr.push(-i, i); }
  return { version: 1, count: n, scale: 100, buckets_per_second: 10,
           duration: 10, peaks: arr };
})();
// THE STRIP'S OWN SOURCE WINDOW, never the picture's.
out.waveSliceIsTheStripsWindow = [
  sbeWaveSlice(PEAKS, 0, 1, 2).map(p => p.map(x => +x.toFixed(2))),
  sbeWaveSlice(PEAKS, 5, 6, 2).map(p => p.map(x => +x.toFixed(2))),
];
out.waveSliceHandlesZoomOut = sbeWaveSlice(PEAKS, 0, 10, 1)
  .map(p => p.map(x => +x.toFixed(2)));
out.waveSliceNoPeaksIsEmpty = sbeWaveSlice(null, 0, 1, 4).length;

const SND2 = () => sbeSetAudioLink(
  lay([clip({ id: 'a', start: 0, end: 4, duration: 10 })]), 'a', false).clips;
out.kfAdd = (() => {
  const cs = sbeAddKeyframe(SND2(), 'a', 2.0, 0.4).clips;
  return JSON.stringify(sbeById(cs, 'a').afx);
})();
out.kfNoTwoOnOneSecond = (() => {
  const cs = sbeAddKeyframe(SND2(), 'a', 2.0, 0.4).clips;
  const q = sbeAddKeyframe(cs, 'a', 2.0005, 0.9);
  return [q.ok, q.why];
})();
out.kfMoveAndSort = (() => {
  let cs = sbeAddKeyframe(SND2(), 'a', 1.0, 0.5).clips;
  cs = sbeAddKeyframe(cs, 'a', 3.0, 0.2).clips;
  cs = sbeMoveKeyframe(cs, 'a', 0, 3.5, 0.9).clips;   // past its neighbour
  return JSON.stringify(sbeById(cs, 'a').afx.points);
})();
out.kfMoveClamps = (() => {
  let cs = sbeAddKeyframe(SND2(), 'a', 1.0, 0.5).clips;
  cs = sbeMoveKeyframe(cs, 'a', 0, 99, 9).clips;
  return JSON.stringify(sbeById(cs, 'a').afx.points);
})();
out.kfDeleteLeavesNoTrace = (() => {
  let cs = sbeAddKeyframe(SND2(), 'a', 2.0, 0.4).clips;
  cs = sbeDeleteKeyframe(cs, 'a', 0).clips;
  return sbeById(cs, 'a').afx === undefined;
})();
out.kfDeleteKeepsAFade = (() => {
  let cs = sbeSetAudioFade(SND2(), 'a', 'in', 1.0).clips;
  cs = sbeAddKeyframe(cs, 'a', 2.0, 0.4).clips;
  cs = sbeDeleteKeyframe(cs, 'a', 0).clips;
  return JSON.stringify(sbeById(cs, 'a').afx);
})();
out.kfLocked = sbeAddKeyframe(
  lay([clip({ id: 'a', end: 4, locked: true })]), 'a', 1, 0.5).why;

// DELETE THE STRIP: the picture keeps playing, silent, and does not move.
out.deleteStrip = (() => {
  let cs = sbeAudioEdit(SND2(), 'a', 'move', 1.0).clips;   // a J-cut first
  const wasFilm = [sbeById(cs, 'a').film_start, sbeById(cs, 'a').film_end];
  const r = sbeDeleteStrip(cs, 'a');
  const c = sbeById(r.clips, 'a');
  return [r.ok, c.audio === undefined, !!c.mute, sbeClipMuted(c),
          [c.film_start, c.film_end], wasFilm];
})();
out.deleteStripIsNotARipple = (() => {
  let cs = sbeSetAudioLink(lay([clip({ id: 'a', end: 2 }),
                                clip({ id: 'b', end: 3 })]), 'a', false).clips;
  const before = cs.map(c => [c.id, c.film_start, c.film_end]);
  const r = sbeDeleteStrip(cs, 'a');
  return [JSON.stringify(before),
          JSON.stringify(r.clips.map(c => [c.id, c.film_start, c.film_end]))];
})();
out.deleteStripTwiceIsRefused = (() => {
  const cs = sbeDeleteStrip(SND2(), 'a').clips;
  const q = sbeDeleteStrip(cs, 'a');
  return [q.ok, q.why];
})();

// ---- THE SOUND'S ENVELOPE -----------------------------------------------
const SND = () => sbeSetAudioLink(
  lay([clip({ id: 'a', start: 0, end: 4, duration: 10 })]), 'a', false).clips;
out.afxFlatIsNothing = [sbeGainPoints({}, 4).length, sbeGainAt({}, 4, 2)];
out.afxSimpleCase = sbeGainPoints({ afx: { fade_in: 1, fade_out: 0.5 } }, 4);
out.afxKeyframes = sbeGainPoints({ afx: { points: [[1, 0.3], [2, 1]] } }, 4);
out.afxCompose = sbeGainPoints({ afx: { fade_in: 1, points: [[2, 0.25]] } }, 4);
out.afxGainAt = [0, 0.5, 1, 2, 3.75, 4]
  .map(t => sbeGainAt({ afx: { fade_in: 1, fade_out: 0.5 } }, 4, t));
out.afxClamp = (() => {
  const e = sbeAfx({ afx: { fade_in: 9, fade_out: 9 } }, 4);
  return [e.fade_in, e.fade_out];
})();
// THE SAME GESTURE THE PICTURE HAS, so the muscle memory transfers.
out.afxSet = (() => {
  let cs = sbeSetAudioFade(SND(), 'a', 'in', 1.0).clips;
  const on = JSON.stringify(sbeById(cs, 'a').afx);
  cs = sbeSetAudioFade(cs, 'a', 'in', 0).clips;
  return [on, sbeById(cs, 'a').afx === undefined, sbeById(cs, 'a').source];
})();
out.afxSetClampsToTheStrip = (() => {
  const cs = sbeSetAudioFade(SND(), 'a', 'in', 99).clips;
  return sbeAfx(sbeById(cs, 'a'), sbeClipAudio(sbeById(cs, 'a')).len).fade_in;
})();
out.afxLocked = sbeSetAudioFade(
  lay([clip({ id: 'a', end: 4, locked: true })]), 'a', 'in', 1).why;

// ---- THE OVERLAY LANE ---------------------------------------------------
// A second video track. Cards do NOT ripple: one is placed where somebody
// wants it in the finished sequence, so moving one moves one.
const CARD = () => [{ id: 'o1', kind: 'still', path: '/x/endcard.png',
                      start: 0, end: 3, film_start: 3, film_end: 6,
                      source: 'human', locked: false }];
out.ovKind = [sbeOvKind({ path: '/x/a.png' }), sbeOvKind({ path: '/x/a.mp4' }),
              sbeOvKind({ kind: 'video', path: '/x/a.png' })];
out.ovAt = [!!sbeOvAt(CARD(), 2.9), !!sbeOvAt(CARD(), 3.0),
            !!sbeOvAt(CARD(), 5.9), !!sbeOvAt(CARD(), 6.0)];
// MOVING ONE MOVES ONE, and a still is its slot.
out.ovMove = (() => {
  const r = sbeOvMove(CARD(), 'o1', 10);
  const o = sbeOvById(r.overlays, 'o1');
  return [r.ok, o.film_start, o.film_end, o.start, o.end, o.source];
})();
// ONE LANE: a move onto another card is refused, not stacked.
out.ovNoStacking = (() => {
  const two = CARD().concat([{ id: 'o2', kind: 'still', path: '/x/b.png',
                               start: 0, end: 2, film_start: 8, film_end: 10 }]);
  const q = sbeOvMove(two, 'o1', 8.5);
  return [q.ok, q.why];
})();
// TRIMMING changes only its own window — the picture beneath is not consulted.
out.ovTrim = (() => {
  let cs = sbeOvTrim(CARD(), 'o1', 'r', 8).overlays;
  const a = sbeOvById(cs, 'o1');
  cs = sbeOvTrim(cs, 'o1', 'l', 4).overlays;
  const b = sbeOvById(cs, 'o1');
  return [[a.film_end, a.end], [b.film_start, b.film_end, b.end]];
})();
// ADDING lands somewhere free rather than refusing.
out.ovAddSlidesPastAnother = (() => {
  const r = sbeOvAdd(CARD(), { path: '/x/b.png', duration_s: 2 }, 4);
  const o = r.added;
  return [r.ok, o.film_start, o.film_end, o.kind];
})();
// DELETE IS NOT A RIPPLE: the other card does not move.
out.ovDeleteMovesNothing = (() => {
  const two = CARD().concat([{ id: 'o2', kind: 'still', path: '/x/b.png',
                               start: 0, end: 2, film_start: 8, film_end: 10 }]);
  const r = sbeOvDelete(two, 'o1');
  return [r.ok, r.overlays.length, sbeOvById(r.overlays, 'o2').film_start];
})();
// THE SAME EFFECTS ACCESSOR a clip uses — the foundation paying off.
out.ovFades = (() => {
  const o = Object.assign({}, CARD()[0], { fx: { fade_in: 1.0 } });
  return [sbeFx(o).fade_in,
          [3, 3.5, 4, 5].map(t => sbeFadeOpacityAt(o, t))];
})();
out.ovFadeClamp = (() => {
  const o = Object.assign({}, CARD()[0], { fx: { fade_in: 9 } });
  return sbeFx(o).fade_in;
})();
// "KEEP ORIGINAL" IS A POINTER SWAP, and nothing else about the card moves —
// the original file was never modified, so undoing the auto-key is one field.
out.ovKeepOriginal = (() => {
  const keyed = CARD().map(o => Object.assign({}, o, { path: '/x/a.keyed.png' }));
  const r = sbeOvSetPath(keyed, 'o1', '/x/a.png');
  const o = sbeOvById(r.overlays, 'o1');
  return [r.ok, o.path, o.film_start, o.film_end,
          // the array it was given is not mutated — undo depends on that
          sbeOvById(keyed, 'o1').path];
})();
out.ovKeepOriginalOnAGoneCard = (() => {
  const r = sbeOvSetPath(CARD(), 'nope', '/x/a.png');
  return [r.ok, r.why];
})();

// ---- A HOLE SHORTER THAN A FRAME ---------------------------------------
// "A black frame that flashes for a microsecond... I tried to drag them close
// and whatever." His three holes were 0.503, 0.380 and 0.096 of a frame.
const HOLED = () => [
  clip({ id: 'a', start: 0, end: 4, film_start: 0, film_end: 4 }),
  clip({ id: 'b', start: 0, end: 4, film_start: 4.02094, film_end: 8.02094 }),
  clip({ id: 'c', start: 0, end: 4, film_start: 8.03678, film_end: 12.03678 }),
  clip({ id: 'd', start: 0, end: 4, film_start: 12.04078, film_end: 16.04078 }),
];
out.gridGapCollapsesSubFrameOnly = [
  sbeGridGap(0.02094), sbeGridGap(0.01584), sbeGridGap(0.004),
  sbeGridGap(1 / 24), sbeGridGap(0.5), sbeGridGap(1.8),
];
out.holedFilmLaysOutContiguous = (() => {
  const cs = lay(HOLED());
  return [shape(cs), sbeHoles(cs).length, lengthsAgree(cs)];
})();
// A DRAG CANNOT REOPEN ONE. `sbeLayout` quantises the gap a gesture produced,
// so there is no pixel that lands a third of a frame away from the cut.
out.dragCannotOpenASubFrameHole = (() => {
  const cs = lay(HOLED());
  const r = sbeMoveTo(cs, 'b', 4.017);
  const o = sbeById(r.clips, 'b');
  return [r.ok, o.film_start, sbeHoles(r.clips).length];
})();
// ...but a hole somebody can SEE survives a drag untouched.
out.dragKeepsAGapYouCanSee = (() => {
  const cs = lay(HOLED());
  const r = sbeMoveTo(cs, 'b', 6.5, { ripple: true });   // ⌘: c and d slide, the hole stays
  return [sbeById(r.clips, 'b').film_start, sbeHoles(r.clips).length];
})();
// THE HEAL CARRIES THE SOUND. Closing a hole is not the user sliding a
// picture, so an unlinked strip travels with it and the J-cut is unchanged.
out.adoptCarriesTheUnlinkedStrip = (() => {
  const cs = HOLED();
  cs[1].audio = { split: true, start: 0.5, end: 4.5, film_start: 3.52094 };
  const before = sbeAudioDrift(cs[1]);
  lay(cs);
  return [+before.toFixed(6), +sbeAudioDrift(cs[1]).toFixed(6),
          +cs[1].audio.film_start.toFixed(6)];
})();
// THE COUNTER STOPS LYING. The old threshold was the literal 1/48 — half a
// frame at 24 fps and the wrong number at any other — and two of his three
// holes were under it, which is why the header read "1 hole".
out.holeCounterOnHisFilm = (() => {
  return [sbeHoles(HOLED(), 1 / 48).length, sbeHoles(HOLED(), 1e-9).length,
          sbeHoles(lay(HOLED())).length];
})();

// ---- THE NET'S OWN STATE ------------------------------------------------
out.protectedChip = (() => {
  const read = () => ({ text: sbeEl('sbeProtected').textContent,
                        cold: sbeEl('sbeProtected').classList.contains('is-cold') });
  const was = { open: SBE.open, id: SBE.id, dirty: SBE.dirty,
                dirtyAt: SBE.dirtyAt, backedUpAt: SBE.backedUpAt };
  SBE.open = true; SBE.id = 'sb_t'; SBE.otherEditor = '';
  SBE.dirty = true; SBE.dirtyAt = Date.now();
  SBE.backedUpAt = Date.now() - 3000;   sbePaintProtected(); const fresh = read();
  SBE.backedUpAt = Date.now() - 40000;  sbePaintProtected(); const cold = read();
  SBE.backedUpAt = 0;                   sbePaintProtected(); const never = read();
  Object.assign(SBE, was);
  return [fresh, cold, never];
})();

// ---- THE EFFECTS MODEL --------------------------------------------------
const FADED = () => lay([clip({ id: 'a', start: 0, end: 4, duration: 10,
                                fx: { fade_in: 0.5, fade_out: 1.0 } })]);
out.fxAbsentIsNothing = (() => {
  const e = sbeFx(clip({ id: 'z', end: 4 }));
  return [e.fade_in, e.fade_out, e.brightness];
})();
out.fxOneAccessor = (() => {
  const e = sbeFx(clip({ id: 'a', end: 4, fx: { fade_in: 0.5 },
                         adjust: { brightness: 0.25 } }));
  return [e.fade_in, e.brightness];
})();
// THE SAME CLAMP THE SERVER APPLIES, and proportional rather than truncating.
out.fxClamp = (() => {
  const both = sbeFx(clip({ id: 'a', end: 4, film_end: 4,
                            fx: { fade_in: 3, fade_out: 3 } }));
  const lop = sbeFx(clip({ id: 'a', end: 4, film_end: 4,
                           fx: { fade_in: 3, fade_out: 1 } }));
  return [[both.fade_in, both.fade_out],
          [sbeRound(lop.fade_in + lop.fade_out), lop.fade_in > lop.fade_out]];
})();
// A VALUE PER FRAME, not a CSS transition: a scrub shows what is true at the
// second it landed on.
out.fxOpacityRamp = (() => {
  const c = sbeById(FADED(), 'a');
  return [0, 0.25, 0.5, 2, 3, 3.5, 4].map(t => sbeFadeOpacityAt(c, t));
})();
out.fxNoFadeIsAlwaysOpaque = (() => {
  const c = sbeById(lay([clip({ id: 'a', end: 4 })]), 'a');
  return [sbeFadeOpacityAt(c, 0), sbeFadeOpacityAt(c, 2)];
})();
// SETTING one, and NEUTRAL IS ABSENT on the way back out.
out.fxSet = (() => {
  let cs = FADED();
  cs = sbeSetFade(cs, 'a', 'out', 0).clips;
  const afterClear = JSON.stringify(sbeById(cs, 'a').fx);
  cs = sbeSetFade(cs, 'a', 'in', 0).clips;
  return [afterClear, sbeById(cs, 'a').fx === undefined,
          sbeById(cs, 'a').source];
})();
out.fxSetClamps = (() => {
  const cs = sbeSetFade(FADED(), 'a', 'in', 99).clips;
  return sbeFx(sbeById(cs, 'a')).fade_in;
})();
out.fxLockedRefuses = sbeSetFade(
  lay([clip({ id: 'a', end: 4, locked: true })]), 'a', 'in', 1).why;
out.fxReachesTheSavePayload = (() => {
  const c = sbeCleanClip(sbeById(FADED(), 'a'));
  return JSON.stringify(c.fx);
})();

// ---- THE PREVIEW PLAYS WHAT THE DOCUMENT SAYS ---------------------------
// HIS ARRANGEMENT, rev 97. One take twice: clip 5 whole at film 25.99, clip 6
// trimmed 0.42 s off the head at film 30.77 with its strip reaching back to
// 30.35. The J-cut was correct on disk, in the render and in the export, and
// inaudible in the one place he checks his work.
const JCUT = () => [
  {id:'c5', path:'/x/w.mp4', proxy:null, start:0.0, end:4.042, film_start:25.99,
   film_end:30.032, source:'human', locked:false, duration:4.042},
  {id:'c6', path:'/x/w.mp4', proxy:null, start:0.42, end:4.042, film_start:30.77,
   film_end:34.392, source:'human', locked:false, duration:4.042,
   audio:{start:0.0, end:4.042, film_start:30.35}}];

// WHO OWNS THE SOUND. Exactly one of the two, never both.
out.stripOwnership = (() => {
  const cs = JCUT();
  const plain = cs[0], split = cs[1];
  const muted = Object.assign({}, split, { mute: true });
  const still = { id: 's', kind: 'still', path: '/x/a.png', start: 0, end: 2,
                  film_start: 0, film_end: 2 };
  return {
    plain: [sbeStripOwned(plain), sbePictureCarriesSound(plain)],
    split: [sbeStripOwned(split), sbePictureCarriesSound(split)],
    muted: [sbeStripOwned(muted), sbePictureCarriesSound(muted)],
    still: [sbeStripOwned(still), sbePictureCarriesSound(still)],
  };
})();

// THE J-CUT IS AUDIBLE BEFORE THE PICTURE CUTS. 29.5 -> 31.5 across the cut.
out.stripAcrossHisCut = (() => {
  const cs = JCUT();
  const at = t => sbeStripsAt(cs, t).map(x => [x.id, x.at]);
  return {
    before: at(29.5),      // his line has not started yet
    lead:   at(30.4),      // ...it starts at 30.35, under the outgoing shot
    edge:   at(30.76),     // still ahead of the picture
    after:  at(31.0),      // and continuous across the cut at 30.77
    late:   at(34.3),
    ended:  at(34.4),      // 30.35 + 4.042 = 34.392
  };
})();

// THE MAPPING. Film second -> SOURCE second, and it is the strip's own clock.
out.stripMapping = (() => {
  const cs = JCUT();
  const one = t => (sbeStripsAt(cs, t)[0] || {}).at;
  return [one(30.35), one(30.77), one(31.35), one(34.39)];
})();

// TWO STRIPS AT ONCE. An L-cut runs the outgoing sound under the incoming
// picture, so both are audible and the player gives each a voice.
out.stripOverlap = (() => {
  const cs = [
    {id:'a', path:'/x/a.mp4', start:0, end:4, film_start:0, film_end:4,
     duration:10, audio:{start:0, end:6, film_start:0}},          // L-cut tail
    {id:'b', path:'/x/b.mp4', start:0, end:4, film_start:4, film_end:8,
     duration:10, audio:{start:0, end:4, film_start:3.5}}];       // J-cut head
  return {
    both: sbeStripsAt(cs, 4.5).map(x => [x.id, x.at]),
    onlyA: sbeStripsAt(cs, 2.0).map(x => x.id),
    onlyB: sbeStripsAt(cs, 6.5).map(x => x.id),
  };
})();

// A MUTED STRIP IS SILENT, and it is silent by not being there at all.
out.stripMuted = (() => {
  const cs = JCUT();
  cs[1] = Object.assign({}, cs[1], { mute: true });
  return [sbeStripsAt(cs, 30.4).length, sbeStripsAt(cs, 31.0).length];
})();

// A COUPLED pair is a strip too: it travels with its picture at a frozen
// offset, and the player is what makes that offset audible.
out.stripCoupled = (() => {
  const cs = [{id:'a', path:'/x/a.mp4', start:0, end:4, film_start:6,
               film_end:10, duration:10,
               audio:{start:0, end:4, film_start:4, linked:true}}];
  return [sbeStripOwned(cs[0]), sbeStripsAt(cs, 4.5).map(x => [x.id, x.at])];
})();

// AN ORDINARY CLIP IS NOT THE PLAYER'S BUSINESS — no strip, no voice, and the
// picture element goes on carrying its own sound as it always did.
out.stripIgnoresOrdinary = sbeStripsAt(lay([clip({id:'z', end:4})]), 1.0).length;

// ---- THE LOAD PATH REWRITES NOTHING -------------------------------------
// sb_carwash rev 97: one take used twice, the first whole with no strip of its
// own, the second trimmed 0.42 s off the head with a strip reaching back to
// cover it. The client must hand back exactly what the disk gave it — there is
// no load-time repair here, and there must never be one.
out.loadRewritesNothing = (() => {
  const disk = [
   {id:'c5', path:'/x/w.mp4', proxy:null, start:0.0, end:4.042, film_start:25.99,
    film_end:30.032, source:'human', locked:false, duration:4.042},
   {id:'c6', path:'/x/w.mp4', proxy:null, start:0.42, end:4.042, film_start:30.77,
    film_end:34.392, source:'human', locked:false, duration:4.042,
    audio:{start:0.0, end:4.042, film_start:30.35}}];
  const before = JSON.stringify(disk);
  const clips = sbeAdoptGaps(disk.map(c => Object.assign({}, c)));
  sbeLayout(clips);
  const saved = clips.map(sbeCleanClip);
  return {
    audioUntouched: JSON.stringify(sbeById(clips, 'c6').audio),
    firstHasNoStrip: sbeById(clips, 'c5').audio === undefined,
    drift: [sbeAudioDrift(sbeById(clips, 'c5')), sbeAudioDrift(sbeById(clips, 'c6'))],
    roundTrips: JSON.stringify(saved.map(c => [c.id, c.film_start, c.film_end,
                                               c.start, c.end, c.audio || null])),
    diskWasNotMutated: JSON.stringify(disk) === before,
  };
})();

// ---- MUTE: the clip's own sound, switched off ---------------------------
// "We should have an option to mute the clip sound." An H3 shot arrives with
// baked-in wind under the line, and on a music cut that is noise to remove.
out.muteAbsentIsAudible = sbeClipMuted(clip({ id: 'm' }));
out.muteWritesTheFlag = (() => {
  const cs = lay([clip({ id: 'a', start: 0, end: 4, duration: 10 })]);
  const q = sbeSetClipMute(cs, 'a', true);
  return [q.ok, sbeById(q.clips, 'a').mute, sbeClipMuted(sbeById(q.clips, 'a')),
          sbeById(q.clips, 'a').source];
})();
// UNMUTE RESTORES EXACTLY WHAT WAS THERE: the field goes, so the document is
// identical to one that was never muted.
out.muteUnmuteLeavesNoTrace = (() => {
  let cs = lay([clip({ id: 'a', start: 0, end: 4, duration: 10 })]);
  // `source` legitimately becomes 'human' — the user touched this clip — so
  // the comparison is of everything the mute could have left behind.
  const shape = x => { const o = sbeCleanClip(x); delete o.source; return JSON.stringify(o); };
  const was = shape(sbeById(cs, 'a'));
  cs = sbeSetClipMute(cs, 'a', true).clips;
  cs = sbeSetClipMute(cs, 'a', false).clips;
  return [was === shape(sbeById(cs, 'a')),
          sbeById(cs, 'a').mute === undefined];
})();
// MUTE AND UNLINK ARE INDEPENDENT, in both directions.
out.muteComposesWithUnlink = (() => {
  let cs = lay([clip({ id: 'a', start: 0, end: 4, duration: 10 }),
                clip({ id: 'b', start: 0, end: 4, duration: 10 })]);
  cs = sbeSetAudioLink(cs, 'b', false).clips;
  cs = sbeAudioEdit(cs, 'b', 'move', 2).clips;      // a J-cut
  cs = sbeSetClipMute(cs, 'b', true).clips;         // ...then mute it
  const c = sbeById(cs, 'b');
  const w = sbeClipAudio(c);
  const after = { muted: sbeClipMuted(c), split: w.split,
                  snd: [w.film_start, sbeRound(w.film_start + w.len)] };
  // ...and muting first, then unlinking, reaches the same place.
  let d = sbeSetClipMute(lay([clip({ id: 'z', start: 0, end: 4, duration: 10 })]),
                         'z', true).clips;
  d = sbeSetAudioLink(d, 'z', false).clips;
  return [after, sbeClipMuted(sbeById(d, 'z')),
          sbeClipAudio(sbeById(d, 'z')).split];
})();
// A CLIP WHOSE FILE HAS NO AUDIO HAS NOTHING TO SWITCH OFF.
out.muteRefusesASilentSource = (() => {
  const cs = lay([clip({ id: 'a', has_audio: false })]);
  const q = sbeSetClipMute(cs, 'a', true);
  return [q.ok, q.why];
})();
out.muteRefusesAStill = sbeSetClipMute(lay([clip({ id: 's', kind: 'still' })]),
                                       's', true).why;
// UNDO WALKS IT BACK, because it goes through the one lane every edit does.
out.muteIsUndoable = (() => {
  SBE.clips = lay([clip({ id: 'a', start: 0, end: 4, duration: 10 })]);
  SBE.undo = []; SBE.redo = [];
  sbeMutate(cs => sbeSetClipMute(cs, 'a', true));
  const on = sbeClipMuted(sbeById(SBE.clips, 'a'));
  sbeUndo();
  const off = sbeClipMuted(sbeById(SBE.clips, 'a'));
  sbeRedo();
  return [on, off, sbeClipMuted(sbeById(SBE.clips, 'a'))];
})();
// IT REACHES DISK. `sbeCleanClip` is the save payload's own shape.
out.muteSurvivesTheSavePayload = (() => {
  const cs = sbeSetClipMute(lay([clip({ id: 'a', start: 0, end: 4, duration: 10 })]),
                            'a', true).clips;
  return sbeCleanClip(sbeById(cs, 'a')).mute;
})();

// ---- the ONE notice surface ---------------------------------------------
// Four full-width blocks in the same column, and they are NOT mutually
// exclusive: the day the overlap rule refused a J-cut he had three of them on
// screen at once, the timeline pushed off the bottom, and the sentence that
// mattered was the third one down.
function noticeState(open, lead) {
  // An earlier case swaps a bare object over one of these; give them all a
  // classList back so the fold is observable rather than skipped.
  for (const id of SBE_NOTICE_ORDER) {
    if (!els[id] || !els[id].classList) els[id] = stubEl(id, true);
  }
  // The snapshot offer is the one QUIET notice: a chip even when alone.
  els.sbeRecover.dataset = { quiet: '1' };
  els.sbeKeyed.dataset = { quiet: '1' };
  for (const id of SBE_NOTICE_ORDER) sbeEl(id).hidden = open.indexOf(id) < 0;
  SBE.noticeLead = lead || '';
  const got = sbePaintNotices();
  return { lead: got, folded: foldedNotices(),
           wrapHidden: sbeEl('sbeNotices').hidden };
}
out.noticeNothingIsNoSurface = noticeState([]);
// A LONE NOTICE IS NEVER A CHIP — folding the only thing on screen would hide
// the sentence to save room nothing is asking for.
out.noticeAloneIsOpen = noticeState(['sbeAlarm']);
// ...but the SNAPSHOT offer is quiet: a chip even when it is the only notice,
// because it is an invitation to look and never a question to answer.
out.noticeQuietIsAlwaysAChip = noticeState(['sbeRecover']);
// The auto-key receipt is the second quiet citizen, and the least urgent thing
// on the screen: it reports something that ALREADY WORKED, so a save that is
// failing takes the width and the receipt sits beside it as a chip.
out.noticeKeyedAloneIsAChip = noticeState(['sbeKeyed']);
out.noticeKeyedNeverOutranksTheAlarm = noticeState(['sbeKeyed', 'sbeAlarm']);
out.noticeKeyedIsLastOfAll =
  noticeState(['sbeKeyed', 'sbeRecover', 'sbeErrors', 'sbeAlarm', 'sbeConflict']);
// Urgency order, and it is not the DOM order: the recovery offer is a
// question about last session and waits behind all three of the others.
out.noticeUrgencyOrder = noticeState(['sbeRecover', 'sbeErrors', 'sbeAlarm']);
out.noticeConflictWinsEverything =
  noticeState(['sbeRecover', 'sbeErrors', 'sbeAlarm', 'sbeConflict']);
// ...and the user overrides it by clicking a chip.
out.noticeClickOpensAChip = (() => {
  noticeState(['sbeAlarm', 'sbeRecover']);
  sbeNoticeOpen('sbeRecover');
  return { lead: SBE.noticeLead, folded: foldedNotices() };
})();
// A lead that goes away hands the surface back to urgency order rather than
// leaving every notice folded behind a chip nobody can see.
out.noticeLeadThatClosesReleases = (() => {
  noticeState(['sbeAlarm', 'sbeRecover'], 'sbeRecover');
  return noticeState(['sbeAlarm']);
})();
// "LATER" IS NOT "DISCARD". Discard deletes the backup, so it was never the
// button for "not now" — and with no third option the bar sat across the top
// of the film for the rest of the session.
out.noticeLaterHidesAndKeeps = (() => {
  SBE.backup = { at: 1, clips: 3, duration: 4 };
  SBE.backupHidden = false;
  sbeEl('sbeRecover').hidden = false;
  sbeNoticeLater();
  // The flag is what `sbePaintRecovery` reads on every repaint; the backup
  // FILE and the offer object are both untouched, which is the whole point.
  return [SBE.backupHidden, !!SBE.backup];
})();
SBE.backup = null; SBE.backupHidden = false;
sbeEl('sbeRecover').hidden = true; sbeEl('sbeAlarm').hidden = true;
sbeEl('sbeErrors').hidden = true; sbeEl('sbeConflict').hidden = true;
sbePaintNotices();

// ---- the PAIR: unlink, cut the picture, put it back ----------------------
// The owner, mid-film: "instead of allowing me to remove or move what video is
// visible while leaving the sound intact and then rematching it, it is actually
// getting the audio out of sync... what should happen is that I only cut the
// video and left the sound in place."
//
// Three butt-joined shots, each a full 4 s of its own 10 s take. Every case
// below reports the same four things — where the picture sits on the film, what
// it plays, where the SOUND sits, what IT plays — because "in sync" is a
// statement about the pair and not about either half.
const PAIR = () => lay([clip({ id: 'a', start: 0, end: 4, duration: 10 }),
                        clip({ id: 'b', start: 0, end: 4, duration: 10 }),
                        clip({ id: 'c', start: 0, end: 4, duration: 10 })]);
function aState(cs, id) {
  const c = sbeById(cs, id);
  const w = sbeClipAudio(c);
  return {
    vid: [sbeRound(c.film_start), sbeRound(c.film_end)],
    vsrc: [sbeRound(c.start), sbeRound(c.end)],
    snd: [w.film_start, sbeRound(w.film_start + w.len)],
    ssrc: [w.start, w.end],
    drift: sbeAudioDrift(c), linked: w.linked, coupled: w.coupled,
    split: w.split,
  };
}
const unlink = (cs, ids) => ids.reduce((x, id) => sbeSetAudioLink(x, id, false).clips, cs);

// 1. UNLINKING ANCHORS THE STRIP WHERE THE CLIP ALREADY IS, and nothing moves.
out.pairUnlink = aState(unlink(PAIR(), ['b']), 'b');

// 2. TRIMMING THE HEAD. The in-point and the slot move together, so the frames
//    that remain still play at the film second they always did — the sound is
//    not touched and does not have to be.
out.pairTrimHead = (() => {
  const cs = unlink(PAIR(), ['b']);
  sbeTrim(cs, 'b', 'l', 5);
  return [aState(cs, 'b'), aState(cs, 'c')];
})();

// 3. TRIMMING THE TAIL of the shot BEFORE it. The picture after a ripple is
//    rigidly translated, so its sound goes with it: this is the case that was
//    silently coming apart, three shots away from the handle being dragged.
out.pairTrimTailRipples = (() => {
  const cs = unlink(PAIR(), ['c']);
  sbeTrim(cs, 'a', 'r', 3, { ripple: true });   // ⌘: the tail slides
  return [aState(cs, 'a'), aState(cs, 'c')];
})();

// 4. MOVING THE PICTURE BY HAND. The one gesture whose whole point is to leave
//    the sound behind — and the clip NEXT to it still rides along.
out.pairMove = (() => {
  const cs = sbeMoveTo(unlink(PAIR(), ['b', 'c']), 'b', 6, { ripple: true }).clips;
  return [aState(cs, 'b'), aState(cs, 'c')];
})();

// 5. THE REMATCH. One call puts the strip back under the frame it came from,
//    and leaves it unlinked so it can be moved again.
out.pairResync = (() => {
  const cs = sbeMoveTo(unlink(PAIR(), ['b']), 'b', 6, { ripple: true }).clips;
  const r = sbeResyncAudio(cs, 'b');
  return [r.ok, aState(r.clips, 'b')];
})();
// ...and it is a RE-MATCH, not an un-trim: a sound trimmed to a later in-point
// keeps that in-point and lands where THAT second plays.
out.pairResyncKeepsTheTrim = (() => {
  let cs = unlink(PAIR(), ['a']);
  cs = sbeAudioEdit(cs, 'a', 'trimL', 1).clips;      // in-point 1 s into the take
  cs = sbeMoveTo(cs, 'a', 5, { ripple: true }).clips;
  const r = sbeResyncAudio(cs, 'a');
  return [r.ok, aState(r.clips, 'a')];
})();
out.pairResyncRefusesWhenAlreadyThere = (() => {
  const q = sbeResyncAudio(unlink(PAIR(), ['b']), 'b');
  return [q.ok, q.why];
})();
out.pairResyncRefusesALinkedClip = (() => {
  const q = sbeResyncAudio(PAIR(), 'b');
  return [q.ok, q.why];
})();

// 6. SPLIT. One strip becomes two that butt-join — never two claiming the same
//    seconds, which is what the deep copy used to produce and what the server
//    refuses as `clips_audio_overlap`.
out.pairSplit = (() => {
  const r = sbeSplitAt(unlink(PAIR(), ['b']), 6, 'b2');
  return [r.ok, aState(r.clips, 'b'), aState(r.clips, 'b2')];
})();
// A split of a DRIFTED pair keeps the drift on both halves — the cut is
// expressed in the source clock the two halves share.
out.pairSplitKeepsTheDrift = (() => {
  const cs = sbeMoveTo(unlink(PAIR(), ['b']), 'b', 6, { ripple: true }).clips;
  const r = sbeSplitAt(cs, 8, 'b2');
  return [r.ok, aState(r.clips, 'b').drift, aState(r.clips, 'b2').drift,
          aState(r.clips, 'b').snd, aState(r.clips, 'b2').snd];
})();
// ...and a cut the strip does not reach is an honest no, because "this half has
// no sound" is a state the document cannot express: an absent `audio` means
// LINKED, which would invent sound the film never had.
out.pairSplitOutsideTheStrip = (() => {
  let cs = unlink(PAIR(), ['a']);
  cs = sbeAudioEdit(cs, 'a', 'trimL', 2).clips;      // the strip now plays 2→4
  const q = sbeSplitAt(cs, 1, 'a2');
  return [q.ok, q.why, aState(cs, 'a').ssrc];
})();

// 7. RIPPLE DELETE pulls the film up, sound included.
out.pairRippleDelete = (() => {
  const cs = sbeRippleDelete(unlink(PAIR(), ['c']), 'b').clips;
  return aState(cs, 'c');
})();

// 8. RE-LINK FREEZES THE OFFSET INSTEAD OF THROWING IT AWAY. Deleting the
//    field snapped the sound back under the picture, so the one button that
//    said "link" destroyed the J-cut the moment it was made — which is why the
//    owner reached for LOCK, and why the clip then refused every drag.
out.pairRelinkFreezesTheOffset = (() => {
  let cs = sbeMoveTo(unlink(PAIR(), ['b']), 'b', 6, { ripple: true }).clips;
  cs = sbeSetAudioLink(cs, 'b', true).clips;
  return [aState(cs, 'b'), sbeById(cs, 'b').audio.linked === true];
})();
// ...and an IN-SYNC re-link still removes the field outright, so a split
// somebody tried and undid leaves a document identical to one that never had
// it.
out.pairRelinkInSyncRemovesTheField = (() => {
  const cs = sbeSetAudioLink(unlink(PAIR(), ['b']), 'b', true).clips;
  return [sbeById(cs, 'b').audio === undefined, aState(cs, 'b')];
})();
// A COUPLED pair travels together: the gesture that moves the picture moves
// the sound, offset intact. This is the "lock it and move it" he described.
out.pairCoupledTravels = (() => {
  let cs = sbeMoveTo(unlink(PAIR(), ['b']), 'b', 6, { ripple: true }).clips;   // -2 s J-cut
  cs = sbeSetAudioLink(cs, 'b', true).clips;                 // freeze it
  cs = sbeMoveTo(cs, 'b', 10, { ripple: true }).clips;       // now move the pair (⌘: past c)
  const s = aState(cs, 'b');
  const t = sbeTrim(cs, 'b', 'r', sbeById(cs, 'b').film_end - 1);
  return [s, aState(cs, 'b').drift, t.ok];
})();
// ...and it cannot be dragged on its own any more, which is what "linked"
// has always meant.
out.pairCoupledStripRefusesTheDrag = (() => {
  let cs = sbeMoveTo(unlink(PAIR(), ['b']), 'b', 6, { ripple: true }).clips;
  cs = sbeSetAudioLink(cs, 'b', true).clips;
  const q = sbeAudioEdit(cs, 'b', 'move', 2);
  return [q.ok, q.why];
})();
// THE TOGGLE ROUND TRIP, which is what he actually did: unlink, trim, link,
// move, unlink again. Both halves stay editable and the offset survives every
// step.
out.pairToggleRoundTrip = (() => {
  const seen = [];
  let cs = unlink(PAIR(), ['b']);
  const t1 = sbeTrim(cs, 'b', 'r', 7);                   // tail, unlinked
  seen.push([t1.ok, aState(cs, 'b').drift]);
  cs = sbeAudioEdit(cs, 'b', 'move', 2.5).clips;         // build the J-cut
  seen.push(aState(cs, 'b').drift);
  cs = sbeSetAudioLink(cs, 'b', true).clips;             // "lock" it
  seen.push([aState(cs, 'b').drift, aState(cs, 'b').coupled]);
  cs = sbeMoveTo(cs, 'b', 9, { ripple: true }).clips;    // move the pair (⌘: past c)
  seen.push([aState(cs, 'b').drift, aState(cs, 'b').snd]);
  const t2 = sbeTrim(cs, 'b', 'l', 9.5);                 // head, coupled
  seen.push([t2.ok, aState(cs, 'b').drift]);
  cs = sbeSetAudioLink(cs, 'b', false).clips;            // unlink again
  seen.push([aState(cs, 'b').drift, aState(cs, 'b').coupled,
             aState(cs, 'b').linked]);
  const t3 = sbeTrim(cs, 'b', 'r', sbeById(cs, 'b').film_end - 0.5);
  seen.push([t3.ok, aState(cs, 'b').drift]);
  return seen;
})();
// THE SCENARIO HE DESCRIBED, START TO FINISH. "I clip a little from the
// visuals and then move all together to fill the gap. So you start hearing
// the character before you see it." Trim the head off shot 2, which opens a
// gap and leaves the sound reaching back into shot 1; freeze that; then drag
// the pair left to close the gap and check that BOTH halves moved by the same
// amount and the sound still leads the picture.
out.pairHisScenario = (() => {
  let cs = unlink(PAIR(), ['b']);
  sbeTrim(cs, 'b', 'l', 5);                       // a little off the visuals
  const afterTrim = aState(cs, 'b');              // picture 5-8, sound 4-8
  cs = sbeSetAudioLink(cs, 'b', true).clips;      // "lock" the two together
  const frozen = aState(cs, 'b');
  cs = sbeMoveTo(cs, 'b', 4, { ripple: true }).clips; // move all together, gap closed (⌘)
  const moved = aState(cs, 'b');
  return {
    trimOpensTheGap: [afterTrim.vid, afterTrim.snd, afterTrim.drift],
    frozen: [frozen.coupled, frozen.drift],
    moved: [moved.vid, moved.snd, moved.drift],
    videoDelta: sbeRound(moved.vid[0] - frozen.vid[0]),
    soundDelta: sbeRound(moved.snd[0] - frozen.snd[0]),
    soundLeadsBy: sbeRound(moved.vid[0] - moved.snd[0]),
  };
})();

// A COUPLED pair is not a drift and is never flagged — its offset is the
// relationship the user froze.
out.pairCoupledIsNotFlagged = (() => {
  let cs = sbeMoveTo(unlink(PAIR(), ['b']), 'b', 6, { ripple: true }).clips;
  const free = sbeSyncBadge(sbeById(cs, 'b')).length > 0;
  cs = sbeSetAudioLink(cs, 'b', true).clips;
  return [free, sbeSyncBadge(sbeById(cs, 'b')).length > 0,
          aState(cs, 'b').drift];
})();
// ...but Resync still reaches it, and a rematched couple has nothing left to
// say, so the field goes.
out.pairResyncACouple = (() => {
  let cs = sbeMoveTo(unlink(PAIR(), ['b']), 'b', 6, { ripple: true }).clips;
  cs = sbeSetAudioLink(cs, 'b', true).clips;
  const r = sbeResyncAudio(cs, 'b');
  return [r.ok, sbeById(r.clips, 'b').audio === undefined, aState(r.clips, 'b')];
})();
// A COUPLED pair splits into two coupled pairs.
out.pairSplitACouple = (() => {
  let cs = sbeMoveTo(unlink(PAIR(), ['b']), 'b', 6, { ripple: true }).clips;
  cs = sbeSetAudioLink(cs, 'b', true).clips;
  const r = sbeSplitAt(cs, 8, 'b2');
  return [r.ok, aState(r.clips, 'b'), aState(r.clips, 'b2')];
})();

// 9. A HEAD TRIM MAY NOT SLIP THE PICTURE INSIDE ITS OWN SLOT. The lead gap
//    used to clamp at zero on its own, so pulling the head open with no room
//    moved the in-point while the slot stood still: the tail rippled the wrong
//    way and an unlinked sound was left describing a frame that had moved.
out.pairHeadTrimCannotSlip = (() => {
  const cs = unlink(lay([clip({ id: 'a', start: 0, end: 2, duration: 10 }),
                         clip({ id: 'b', start: 2, end: 5, film_start: 3,
                                film_end: 6, duration: 10 })]), ['b']);
  const q = sbeTrim(cs, 'b', 'l', 1);      // 2 s of head wanted, 1 s of room
  return [q.ok, aState(cs, 'b')];
})();
out.pairHeadTrimUsesTheRoomItHas = (() => {
  const cs = unlink(lay([clip({ id: 'a', start: 0, end: 2, duration: 10 }),
                         clip({ id: 'b', start: 2, end: 5, film_start: 3,
                                film_end: 6, duration: 10 })]), ['b']);
  sbeTrim(cs, 'b', 'l', 2.5);              // half the gap
  return aState(cs, 'b');
})();
// ...and against a neighbour there is no room at all, so the handle refuses
// rather than growing the clip out of its far end.
out.pairHeadTrimAgainstANeighbour = (() => {
  const cs = unlink(PAIR(), ['b']);
  const q = sbeTrim(cs, 'b', 'l', 3);
  return [q.ok, q.why, aState(cs, 'b')];
})();

// 10. THE FLAG'S ARITHMETIC. Positive is LATE, the tolerance is half a frame,
//     and the label is what both halves print.
out.pairDriftSigns = (() => {
  const early = sbeMoveTo(unlink(PAIR(), ['b']), 'b', 6, { ripple: true }).clips;   // sound early
  const late = sbeAudioEdit(unlink(PAIR(), ['b']), 'b', 'move', 6).clips;
  return [sbeAudioDrift(sbeById(early, 'b')), sbeAudioDrift(sbeById(late, 'b')),
          sbeDriftLabel(sbeAudioDrift(sbeById(early, 'b'))),
          sbeDriftLabel(sbeAudioDrift(sbeById(late, 'b')))];
})();
out.pairInSyncTolerance = (() => {
  const cs = unlink(PAIR(), ['b']);
  const near = sbeAudioEdit(cs, 'b', 'move', 4 + 1 / 96).clips;
  const past = sbeAudioEdit(cs, 'b', 'move', 4 + 1 / 12).clips;
  return [sbeAudioInSync(sbeById(cs, 'b')), sbeAudioInSync(sbeById(near, 'b')),
          sbeAudioInSync(sbeById(past, 'b')),
          sbeAudioInSync(sbeById(PAIR(), 'b'))];
})();
// A LINKED clip is never carried and never flagged: it has no field to write,
// and writing one would unlink it behind the user's back.
out.pairCarryLeavesLinkedAlone = (() => {
  const cs = PAIR();
  sbeTrim(cs, 'a', 'r', 3, { ripple: true });   // ⌘: the tail slides
  return [sbeById(cs, 'c').audio === undefined, aState(cs, 'c').drift];
})();

// ---- the boundaries, and a drag as the STREAM of events it really is -----
// Everything above trims INSIDE the source and calls each edit once. Both of
// tonight's arithmetic bugs lived on the other side of exactly those two
// lines.
//
// ASKING FOR MORE HEAD THAN THE TAKE HAS. `start` clamped at the top of the
// source while `film` kept sliding, so the strip's out-point moved left — an
// L-cut that silently shortened the line it was keeping.
out.headTrimPastTheSource = (() => {
  const cs = sbeSetAudioLink(lay([clip({ id: 'a', start: 0, end: 2, duration: 10 }),
                                  clip({ id: 'b', start: 0.5, end: 2, duration: 10 })]),
                             'b', false).clips;
  const was = sbeClipAudio(sbeById(cs, 'b'));
  const q = sbeAudioEdit(cs, 'b', 'trimL', 1.0);
  const now = sbeClipAudio(sbeById(q.clips, 'b'));
  const far = sbeClipAudio(sbeById(sbeAudioEdit(cs, 'b', 'trimL', 0).clips, 'b'));
  return { wasOut: sbeRound(was.film_start + was.len),
           outAt1: sbeRound(now.film_start + now.len),
           outAt0: sbeRound(far.film_start + far.len),
           start1: now.start, start0: far.start };
})();

// A DRAG IS N POINTERMOVES, NOT ONE. sbeOnMusicMove re-reads the already
// mutated object every event while `want` comes off a fixed anchor, so an
// edit that folds the previous offset back into its answer composes
// differently depending on how fast the mouse was going.
out.musicDragIsFrameRateIndependent = (() => {
  const drive = (audio, steps) => {
    let a = Object.assign({}, audio);
    for (let i = 1; i <= steps; i++) {
      const want = 6 * (i / steps);          // same gesture, more events
      a = Object.assign({}, a, sbeMusicEdit(a, 'move', want, 180));
    }
    return sbeMusicWindow(a, 180).film_start;
  };
  const base = { path: '/o/t.wav', duration: 180 };
  return {
    zeroOne: drive(Object.assign({ offset: 0 }, base), 1),
    zeroSix: drive(Object.assign({ offset: 0 }, base), 6),
    headOne: drive(Object.assign({ offset: 10 }, base), 1),
    headSix: drive(Object.assign({ offset: 10 }, base), 6),
    trimSix: drive(Object.assign({ offset: 0, trim_start: 10 }, base), 6),
  };
})();
// ...and a drag that lands where the block already was writes the SAME
// object, so sbeOnMusicUp's changed-test cannot mark a film dirty for a
// gesture nobody can see.
out.musicDragNowhereChangesNothing = (() => {
  const a = { path: '/o/t.wav', duration: 180, offset: 10, trim_start: 10 };
  const at = sbeMusicWindow(a, 180).film_start;
  const q = sbeMusicEdit(a, 'move', at, 180);
  return [q.offset, a.offset, q.trim_start];
})();

// ---- the draft switch, DRIVEN through its refusal ------------------------
// `sbeDraftOp`'s return value is the whole guard: the comment above its
// backup call states the invariant, and the call used to be awaited and
// discarded. With an unanswered offer on screen the backup cannot write, so
// the switch must refuse and nothing may leave the panel.
out.draftSwitchRefusesWhenTheBackupCannotWrite = await (async () => {
  // A REAL write failure, not an unanswered offer: the snapshot POST fails,
  // so the work on screen has nowhere to go and the switch must not proceed.
  SBE.backup = null;
  SBE.dirty = true; SBE.conflict = 0;
  FETCHES = [];
  NEXT = { status: 500, body: { ok: false, error: 'disk is full' } };
  const got = await sbeDraftOp('activate', { slug: 'take-two' });
  const out2 = [got, FETCHES.length, SBE.dirty,
                (toasts[toasts.length - 1] || '').slice(0, 34)];
  SBE.backup = null; SBE.dirty = false; SBE.saveFailed = '';
  els.sbeAlarm.hidden = true;
  return out2;
})();

// ---- undo reads the timeline, not the document --------------------------
// A soundtrack discovered from peaks.json is on the timeline before the
// arrangement was ever cut to it, so a snapshot of `SBE.edit.audio` carried
// null and the first ⌘Z of any clip edit deleted the track.
out.undoKeepsADiscoveredTrack = (() => {
  SBE.clips = lay([clip({ id: 'a' }), clip({ id: 'b' })]);
  SBE.edit = { clips: SBE.clips };                 // never auto-edited
  SBE.audio = { path: '/state/track.mp3', offset: 0, peaks: 'peaks.json',
                duration: 180 };
  SBE.undo = []; SBE.redo = []; SBE.dirty = false;
  sbeMutate(cs => sbeTrim(cs, 'a', 'r', 1.5));
  sbeUndo();
  const kept = SBE.audio && SBE.audio.path;
  sbeRedo();
  const after = SBE.audio && SBE.audio.path;
  return [kept || null, after || null, SBE.edit.audio || null];
})();
SBE.clips = []; SBE.edit = {}; SBE.audio = null; SBE.undo = []; SBE.redo = [];
SBE.dirty = false;

// ---- the timeline's own top edge ----------------------------------------
// The clamps, where a dragged pixel lands, and the fact that it survives a
// reload without ever touching the document.
const laneSum = (L) => SBE_TL_CHROME + L.ov + L.track + L.alane + L.wave;
out.tlClamp = {
  under: sbeTlClamp(10, 600),
  over: sbeTlClamp(9000, 600),
  exact: sbeTlClamp(400, 600),
  // A window that has nothing to give pins the handle at the floor rather
  // than handing the monitors a negative height.
  noRoom: sbeTlClamp(400, 0),
  // Nobody may ask for more than the lanes can use, whatever they measured.
  ceiling: sbeTlClamp(9000, 99999),
  junk: sbeTlClamp('kittens', 600),
};
out.lanesAtFloor = sbeLaneHeights(SBE_TL_MIN_H);
out.lanesAtCeiling = sbeLaneHeights(SBE_TL_MAX_H);
out.lanesMid = sbeLaneHeights(SBE_TL_MIN_H + 100);
// EVERY PIXEL IS SPENT: the four lanes plus the box's own chrome are the
// height that was asked for, at the floor, in the middle and at the top.
out.laneSums = [laneSum(out.lanesAtFloor) - SBE_TL_MIN_H,
                laneSum(out.lanesMid) - (SBE_TL_MIN_H + 100),
                laneSum(out.lanesAtCeiling) - SBE_TL_MAX_H];
// Where 100px of drag went, lane by lane.
out.laneGain = {
  ov: out.lanesMid.ov - out.lanesAtFloor.ov,
  track: out.lanesMid.track - out.lanesAtFloor.track,
  alane: out.lanesMid.alane - out.lanesAtFloor.alane,
  wave: out.lanesMid.wave - out.lanesAtFloor.wave,
};
// The picture track caps at 240 long before the sound lanes do; what it
// cannot take has to be OFFERED AGAIN, not dropped on the floor.
out.lanesPastTheTrackCap = sbeLaneHeights(SBE_TL_MIN_H + 340);
out.laneRedistributed = laneSum(out.lanesPastTheTrackCap) - (SBE_TL_MIN_H + 340);
// The ruler never grows: it is a scale, not a surface.
out.rulerFixed = [out.lanesAtFloor.ruler, out.lanesAtCeiling.ruler];

out.tlPref = (() => {
  const seen = {};
  STORE.clear();
  seen.fresh = sbeTlPrefRead();               // nothing stored yet
  sbeTlPrefWrite(392);
  seen.stored = STORE.get('phos_sbe_tl_h');
  seen.restored = sbeTlPrefRead();            // ...as a new tab would read it
  STORE.set('phos_sbe_tl_h', '99999');
  seen.absurd = sbeTlPrefRead();
  STORE.set('phos_sbe_tl_h', 'not-a-height');
  seen.junk = sbeTlPrefRead();
  seen.keys = [...STORE.keys()];
  STORE.clear();
  return seen;
})();
// AND IT IS NOT IN THE DOCUMENT. The save payload is the whole of what the
// server is told; a window height in there would bump the revision on a drag.
out.tlIsNotInTheSave = (() => {
  SBE.tlH = 480; SBE.tlNow = 480;
  const body = JSON.stringify(sbeSaveBody(Object.assign({}, SBE, {
    id: 'sb_t', edit: {}, clips: lay([clip({ id: 'a' })]), expect: 3,
  })));
  SBE.tlH = SBE_TL_MIN_H; SBE.tlNow = SBE_TL_MIN_H;
  return [body.indexOf('tlH'), body.indexOf('480'), body.indexOf('tl_h')];
})();

// ---- the level line, and the heights it has to survive -------------------
// THE 20PX BAND MUST NOT COME BACK. The gain a pointer means is derived from
// the strip's own height, so the round trip has to close at every height the
// share table can produce — not just the one somebody had on screen.
const stripHs = [SBE_TL_MIN_H, SBE_TL_MIN_H + 80, SBE_TL_MIN_H + 180,
                 SBE_TL_MAX_H].map(h => sbeLaneHeights(h).alane - 7);
out.stripHeights = stripHs;
out.lvlRoundTrip = (() => {
  const bad = [];
  for (const H of stripHs) {
    for (const g of [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]) {
      const back = sbeStripGain(200 + sbeStripY(g, H), 200, H);
      if (Math.abs(back - g) > 1e-9) bad.push([H, g, back]);
    }
  }
  return bad;
})();
// The two ends mean what they say at every height: the top of the strip is
// unity and the bottom is silence.
out.lvlEnds = stripHs.map(H => [sbeStripGain(200, 200, H),
                                sbeStripGain(200 + H, 200, H)]);
// ...and the middle is NOT hard-coded: at a tall strip, half way down is half.
out.lvlMid = stripHs.map(H => +sbeStripGain(200 + H / 2, 200, H).toFixed(3));
// THE HEADROOM, at the shortest strip the share table can produce. Unity has
// to sit far enough down that the whole 14px target is inside the box —
// "maybe you should put the orange line a little lower so it feels more
// draggable" was a line drawn ON the top edge with most of its target clipped
// away outside it.
out.lvlUnityY = stripHs.map(H => +sbeStripY(1, H).toFixed(2));
out.lvlSilenceGap = stripHs.map(H => +(H - sbeStripY(0, H)).toFixed(2));
// ---- the span of the line the corner handles have NOT taken ---------------
// A flat line across a 400-unit strip: the target starts and ends clear of
// both 22px handles, and its y is the line's own y.
out.lvlHitFlat = sbeLvlHitPath([[0, 1], [4, 1]],
                               t => (t / 4) * 400, g => sbeStripY(g, 40),
                               SBE_LVL_CLEAR, 400 - SBE_LVL_CLEAR);
// ...and a sloped one, so the cut lands ON the line rather than on a chord
// between whole points: gain 1 -> 0 across the strip, so at x=33 of 400 the
// gain is 1 - 33/400 and the y is that gain's y.
out.lvlHitSloped = sbeLvlHitPath([[0, 1], [4, 0]],
                                 t => (t / 4) * 400, g => sbeStripY(g, 40),
                                 SBE_LVL_CLEAR, 400 - SBE_LVL_CLEAR);
out.lvlHitSlopedWant = [+sbeStripY(1 - SBE_LVL_CLEAR / 400, 40).toFixed(2),
                        +sbeStripY(SBE_LVL_CLEAR / 400, 40).toFixed(2)];
// A strip with no room between the handles is offered NO target at all: an
// 8px stub of a control is worse than none, and the inspector is the route
// that never depends on width.
out.lvlHitNarrow = sbeLvlHitPath([[0, 1], [4, 1]],
                                 t => (t / 4) * 70, g => sbeStripY(g, 40),
                                 SBE_LVL_CLEAR, 70 - SBE_LVL_CLEAR);
// Segments entirely inside a handle's column are dropped, not clamped onto its
// edge — a target that ran to x=0 would be back under the fade handle.
out.lvlHitPoints = sbeLvlHitPath([[0, 1], [0.2, 0.5], [3.8, 0.5], [4, 1]],
                                 t => (t / 4) * 400, g => sbeStripY(g, 40),
                                 SBE_LVL_CLEAR, 400 - SBE_LVL_CLEAR);

// ---- "DID THE FILM ACTUALLY CHANGE?" -------------------------------------
// The question `sbeOnTrackUp` asks on every drag, and the one that has twice
// answered "no" over an edit somebody had just made. Every mode that writes a
// field is driven here, so a new mode is checked against the list rather than
// against whichever gesture the list was last written for.
out.dragFp = (() => {
  const base = () => [clip({ id: 'a', end: 4, film_end: 4 }),
                      clip({ id: 'b', start: 0, end: 4, film_start: 4, film_end: 8 })];
  const fp = cs => sbeDragFingerprint(cs);
  const moved = (mut) => { const cs = base(); const out2 = mut(cs); return fp(base()) !== fp(out2 || cs); };
  return {
    // Nothing moved: the snapshot is restored and no revision is burned.
    still: fp(base()) === fp(base()),
    // A FADE IS AN EDIT. `mode: 'fade'` writes `fx` and nothing else, so with
    // `fx` off the list the ramp was dragged, painted, and thrown away on
    // pointerup — which is the second half of "I'm not being able to do it".
    fade: moved(cs => sbeSetFade(cs, 'a', 'in', 0.75).clips),
    fadeOut: moved(cs => sbeSetFade(cs, 'a', 'out', 0.4).clips),
    // ...and the three that were already on it stay on it.
    trim: moved(cs => { sbeTrim(cs, 'a', 'r', 3); }),
    move: moved(cs => sbeMoveTo(cs, 'b', 5).clips),
    sound: moved(cs => { cs[0].audio = { start: 0, end: 4, film_start: 0.5 }; }),
  };
})();
// Which strips may be shaped at all — the rule the line, the ghost, the click
// and the inspector all now ask the same way.
out.editable = (() => {
  const mk = o => Object.assign(clip({ id: 'e', end: 3 }), o);
  return {
    linked: sbeStripEditable(mk({ audio: { start: 0, end: 3, film_start: 0, linked: true } })),
    unlinked: sbeStripEditable(mk({ audio: { start: 0, end: 3, film_start: 0, linked: false } })),
    locked: sbeStripEditable(mk({ locked: true, audio: { start: 0, end: 3, film_start: 0, linked: false } })),
    silent: sbeStripEditable(mk({ has_audio: false, audio: { start: 0, end: 3, film_start: 0, linked: false } })),
    nothing: sbeStripEditable(null),
  };
})();
out.legend = sbeKeysLegend();

// ---- the stage's two layers ---------------------------------------------
// OPACITY IS ALSO THE LAYER SWITCH: `.sbe-stage video` and `.sbe-stage
// img.sbe-still` are opacity:0 until one of them wins `.is-on`. A painter that
// writes the ramp onto BOTH turns the hidden one on, and the still — last in
// the stage, backed with #000 — paints a black rectangle over the video. The
// owner reported it as "videos are not loading"; they were loading fine.
out.stage = (() => {
  const V = sbeEl('sbeVideo'), I = sbeEl('sbeStill');
  const SRC = sbeEl('sbeSrcVideo'), SRCI = sbeEl('sbeSrcStill');
  const on = (el) => { V._cls.delete('is-on'); I._cls.delete('is-on');
                       if (el) el._cls.add('is-on'); };
  const snap = () => [V.style.opacity, I.style.opacity];
  // One clip, 0-4s on the film, with a one-second fade in.
  let cs = lay([clip({ id: 'a', end: 4 })]);
  const r = sbeSetFade(cs, 'a', 'in', 1);
  SBE.clips = r.ok ? r.clips : cs;
  SBE.curId = 'a';
  const seen = {};
  // 1. the VIDEO is the layer that is showing, half way through the fade
  on(V); SBE.playhead = 0.5; sbeFadePaint();
  seen.videoOnInFade = snap();
  // 2. ...and the STILL is, at the same instant
  on(I); sbeFadePaint();
  seen.stillOnInFade = snap();
  // 3. THE TRANSITION, which is the one that leaves a ghost: the layer that
  //    was carrying a ramp has to be handed back to the stylesheet when it
  //    stops being the layer that is showing.
  seen.videoAfterHandover = V.style.opacity;
  // 4. past the fade, the on layer is fully up
  on(V); SBE.playhead = 2; sbeFadePaint();
  seen.outsideFade = snap();
  // 5. nothing at the playhead at all
  SBE.clips = []; SBE.curId = ''; SBE.playhead = 9; sbeFadePaint();
  seen.noClip = snap();
  // 6. neither layer showing — nothing may be written to either
  on(null); sbeFadePaint();
  seen.neitherOn = snap();
  // 7. the SOURCE monitor's layers are not this painter's business
  seen.sourceUntouched = [SRC.style.opacity === undefined,
                          SRCI.style.opacity === undefined];
  SBE.clips = []; SBE.curId = ''; SBE.playhead = 0;
  return seen;
})();

// The sibling layer, for contrast: the overlay's switch is DISPLAY, so its
// opacity is free to be a ramp and nothing it writes can turn it on.
out.overlayLayer = (() => {
  const el = sbeEl('sbeOvLayer');
  SBE.overlays = [{ id: 'o1', path: '/o/card.png', film_start: 0, film_end: 4,
                    fx: { fade_in: 1 } }];
  SBE.playhead = 0.5;
  sbeOvPaint();
  const during = [el.style.opacity, el._cls.has('is-on')];
  SBE.overlays = [];
  sbeOvPaint();
  const after = [el.style.opacity, el._cls.has('is-on')];
  SBE.playhead = 0;
  return { during: during, after: after };
})();

process.stdout.write(JSON.stringify(out));
"""


def run_contract() -> dict:
    if NODE is None:
        raise unittest.SkipTest("node not on PATH")
    source = panel_source()
    script = (SHIM
              + "\n".join(extract_function(n, source) for n in FUNCTIONS)
              + "\n(async () => {\n" + BODY + "\n})();\n")
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
        fh.write(script)
        path = Path(fh.name)
    try:
        result = subprocess.run([NODE, str(path)], capture_output=True,
                                text=True, timeout=60)
        if result.returncode:
            raise AssertionError(result.stdout + "\n" + result.stderr)
        return json.loads(result.stdout)
    finally:
        path.unlink(missing_ok=True)


class TimelineClient(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.r = run_contract()

    # ---- the waveform ----------------------------------------------------
    def test_peaks_decode_interleaved_int16_over_the_scale(self):
        p = self.r["peaks"]
        self.assertEqual(p["count"], 3)
        self.assertEqual([round(v, 4) for v in p["lo"]], [-1.0, 0.0, -0.252])
        self.assertEqual([round(v, 4) for v in p["hi"]], [1.0, 0.5039, 0.252])
        self.assertEqual(p["rate"], 100)

    def test_a_peaks_file_that_lies_about_its_count_does_not_read_off_the_end(self):
        self.assertEqual(self.r["peaksJunk"]["count"], 1)
        self.assertIsNone(self.r["peaksNone"])

    # ---- the beat grid ---------------------------------------------------
    def test_the_grid_carries_beats_and_marks_downbeats(self):
        self.assertEqual(self.r["grid"][:3],
                         [[0, True], [0.5, False], [1, False]])
        self.assertIn([2, True], self.r["grid"])
        self.assertEqual(len(self.r["grid"]), 9)

    def test_the_grid_is_NEVER_extrapolated_past_the_fitted_span(self):
        # beat_map fits ONE tempo across a span because real tracks drift.
        # A line past that span is a confident wrong answer, so there is none.
        self.assertEqual(self.r["gridPastSpan"], 0)

    def test_the_grid_is_clipped_to_the_window_being_drawn(self):
        self.assertEqual(self.r["gridWindow"], [1, 1.5, 2])

    def test_an_audio_offset_shifts_the_grid_into_film_time(self):
        self.assertEqual(self.r["gridOffset"][0], -1)

    def test_a_weak_grid_is_flagged_as_a_guess(self):
        self.assertFalse(self.r["guess"])
        self.assertTrue(self.r["guessLow"])

    # ---- snap ------------------------------------------------------------
    def test_snap_catches_a_beat_inside_the_tolerance(self):
        self.assertEqual(self.r["snapCatches"], 1.5)

    def test_snap_leaves_a_time_that_is_not_near_a_beat_alone(self):
        self.assertEqual(self.r["snapLeaves"], 1.2)

    def test_the_override_is_a_straight_bypass(self):
        self.assertEqual(self.r["snapOff"], 1.44)

    def test_a_downbeat_wins_a_tie(self):
        self.assertEqual(self.r["snapPrefersDownbeat"], 2)

    def test_snap_past_the_span_is_a_no_op_not_an_invented_beat(self):
        self.assertEqual(self.r["snapPastSpan"], 9.9)

    # ---- layout ----------------------------------------------------------
    def test_the_film_slot_is_derived_from_the_source_window_every_time(self):
        self.assertTrue(self.r["packedLengths"])
        self.assertEqual(self.r["packed"],
                         [["a", 0, 2, 0, 2], ["b", 0, 3, 2, 5], ["c", 0, 1.5, 5, 6.5]])

    def test_a_hole_survives_the_round_trip(self):
        self.assertEqual(self.r["holeKept"],
                         [["a", 0, 2, 0, 2], ["b", 0, 3, 5, 8]])
        self.assertEqual(self.r["holes"], [[2, 5, 3]])
        self.assertEqual(self.r["duration"], 8)

    def test_clip_lookup_answers_honestly_over_a_hole(self):
        self.assertEqual(self.r["clipAt"], "b")
        self.assertIsNone(self.r["clipInHole"])

    # ---- move ------------------------------------------------------------
    def test_a_ripple_move_reorders_and_everything_after_it_slides(self):
        self.assertEqual(self.r["moveToHead"],
                         [["c", 0, 1.5, 0, 1.5], ["a", 0, 2, 1.5, 3.5],
                          ["b", 0, 3, 3.5, 6.5]])
        self.assertTrue(self.r["moveLengths"])
        self.assertTrue(self.r["moveNoOverlap"])

    def test_a_plain_move_stays_between_its_neighbours_and_moves_nothing_else(self):
        # c dragged to 0 cannot pass b: it stays where it was.
        self.assertEqual(self.r["moveClamped"],
                         [["a", 0, 2, 0, 2], ["b", 0, 3, 2, 5], ["c", 0, 1.5, 5, 6.5]])
        # b slides inside its own hole; a and c do not move.
        self.assertEqual(self.r["moveKeepsNeighbours"],
                         [["a", 0, 2, 0, 2], ["b", 0, 3, 3, 6], ["c", 0, 1, 7, 8]])

    def test_a_moved_clip_is_stamped_human(self):
        # The server keeps the other end of this promise: a re-plan can leave a
        # human's cut alone precisely because it is labelled.
        self.assertEqual(self.r["moveMarksHuman"], "human")

    def test_a_clip_dropped_into_open_air_keeps_the_hole_it_opened(self):
        self.assertEqual(self.r["moveOpensHole"],
                         [["a", 0, 2, 0, 2], ["b", 0, 3, 6, 9]])

    def test_a_locked_clip_refuses_to_move(self):
        self.assertFalse(self.r["moveLocked"]["ok"])
        self.assertEqual(self.r["moveLocked"]["why"], "locked")
        self.assertEqual(self.r["moveLocked"]["shape"][0], ["a", 0, 2, 0, 2])

    def test_the_flow_goes_AROUND_a_locked_clip_never_through_it(self):
        self.assertEqual(self.r["anchored"],
                         [["a", 0, 2, 0, 2], ["L", 0, 2, 3, 5], ["b", 0, 4, 5, 9]])
        self.assertTrue(self.r["anchoredNoOverlap"])

    # ---- trim ------------------------------------------------------------
    def test_the_right_handle_with_ripple_changes_the_length_and_slides_the_tail(self):
        self.assertEqual(self.r["trimRight"],
                         [["a", 0, 1.25, 0, 1.25], ["b", 0, 3, 1.25, 4.25]])
        self.assertTrue(self.r["trimRightLengths"])

    def test_the_right_handle_by_default_opens_a_hole_and_moves_nothing_else(self):
        self.assertEqual(self.r["trimRightHole"],
                         [["a", 0, 1.25, 0, 1.25], ["b", 0, 3, 2, 5]])
        self.assertTrue(self.r["trimRightHoleLengths"])
        # growing back stops at the neighbour, never pushes it
        self.assertEqual(self.r["trimRightStopsAtNext"],
                         [["a", 0, 2, 0, 2], ["b", 0, 3, 2, 5]])

    def test_the_left_handle_moves_the_in_point_and_leaves_the_tail_alone(self):
        self.assertEqual(self.r["trimLeft"],
                         [["a", 0, 2, 0, 2], ["b", 0.5, 3, 2.5, 5]])
        self.assertTrue(self.r["trimLeftLengths"])

    def test_a_trim_cannot_run_past_the_end_of_its_source(self):
        self.assertEqual(self.r["trimClampedToSource"], [["a", 0, 2.4, 0, 2.4]])

    def test_a_trim_cannot_collapse_a_clip_to_nothing(self):
        self.assertEqual(self.r["trimClampedToMin"], [["a", 1.8, 2, 1.8, 2]])

    def test_a_locked_clip_refuses_to_trim(self):
        self.assertEqual(self.r["trimLocked"], "locked")

    # ---- ripple + split --------------------------------------------------
    def test_ripple_delete_pulls_everything_after_it_up(self):
        self.assertEqual(self.r["ripple"],
                         [["a", 0, 2, 0, 2], ["c", 0, 1.5, 2, 3.5]])
        self.assertTrue(self.r["rippleLengths"])


    def test_a_lift_leaves_the_hole_and_moves_nothing_downstream(self):
        # The whole point: every film window is IDENTICAL before and after.
        self.assertEqual(self.r["liftWindows"], self.r["liftWindowsBefore"])
        self.assertEqual(self.r["liftKinds"], ["video", "slug", "video"])
        self.assertTrue(self.r["liftLengths"])
        self.assertEqual(self.r["liftHole"], [None, None, "human"])

    def test_a_lift_refuses_a_locked_clip_and_an_existing_hole(self):
        self.assertEqual(self.r["liftLocked"], "locked")
        self.assertEqual(self.r["liftTwice"], "already a hole")
    def test_a_locked_clip_refuses_to_be_deleted(self):
        self.assertEqual(self.r["rippleLocked"], "locked")

    def test_split_makes_two_clips_that_still_add_up(self):
        self.assertEqual(self.r["split"],
                         [["a", 0, 1.2, 0, 1.2], ["a2", 1.2, 2, 1.2, 2],
                          ["b", 0, 3, 2, 5]])
        self.assertTrue(self.r["splitLengths"])
        self.assertEqual(self.r["splitHuman"], ["human", "human", "auto"])

    def test_split_refuses_where_it_would_make_a_blink(self):
        self.assertFalse(self.r["splitTooClose"])

    def test_split_over_a_hole_is_an_honest_no(self):
        self.assertFalse(self.r["splitInHole"])

    # ---- placing ---------------------------------------------------------
    def test_a_generated_clip_lands_in_the_hole_it_was_ordered_for(self):
        self.assertEqual(self.r["placed"][1],
                         ["/o/new.mp4", 2.5, 5.5, "human"])
        self.assertTrue(self.r["placedNoOverlap"])

    def test_filling_a_hole_does_NOT_move_the_film_after_it(self):
        # Every other operation ripples. This one must not: the shot was
        # generated for this slot so the cuts around it stay on their beats.
        self.assertEqual(self.r["filled"],
                         [["a.mp4", 0, 2], ["fill.mp4", 3, 6], ["b.mp4", 9, 11]])

    def test_a_clip_too_long_for_its_hole_pushes_rather_than_overlaps(self):
        self.assertEqual(self.r["overfilled"],
                         [["a.mp4", 0, 2], ["big.mp4", 2, 6], ["b.mp4", 6, 8]])
        self.assertTrue(self.r["overfilledNoOverlap"])

    # ---- undo ------------------------------------------------------------
    def test_undo_and_redo_walk_the_arrangement_back_and_forward(self):
        self.assertEqual([c[0] for c in self.r["afterMutate"]], ["b"])
        self.assertEqual([c[0] for c in self.r["afterUndo"]], ["a", "b"])
        self.assertEqual([c[0] for c in self.r["afterRedo"]], ["b"])
        self.assertEqual([c[0] for c in self.r["undoFloor"]], ["a", "b"])
        self.assertTrue(self.r["dirtyAfterMutate"])

    def test_a_refused_edit_writes_no_undo_step(self):
        self.assertEqual(self.r["refusedNoUndo"], 0)

    # ---- save ------------------------------------------------------------
    def test_the_save_body_is_the_shape_the_route_documents(self):
        b = self.r["saveBody"]
        self.assertEqual(sorted(b.keys()), ["edit", "expect_revision", "id"])
        self.assertEqual(b["expect_revision"], 4)
        self.assertEqual(b["edit"]["board_id"], "sb_t")
        self.assertEqual(b["edit"]["clips"][0]["source"], "human")
        self.assertEqual(b["edit"]["beats"]["bpm"], 120)

    def test_client_only_bookkeeping_never_reaches_disk(self):
        self.assertEqual(self.r["saveStrippedPrivate"], [])

    def test_expect_revision_is_omitted_rather_than_sent_as_null(self):
        self.assertNotIn("expect_revision", self.r["saveNoExpect"])

    def test_every_error_is_pinned_to_the_clip_that_caused_it(self):
        m = self.r["errMap"]
        self.assertEqual(list(m["byId"].keys()), ["b"])
        self.assertEqual(m["byId"]["b"][0]["code"], "clip_window")
        self.assertEqual(m["doc"][0]["code"], "version")

    def test_a_409_keeps_the_arrangement_and_says_who_won(self):
        self.assertFalse(self.r["saveConflict"])
        a = self.r["afterConflict"]
        self.assertEqual([c[0] for c in a["clips"]], ["a", "b"])
        self.assertTrue(a["dirty"])          # still unsaved, still on screen
        self.assertEqual(a["conflict"], 7)
        self.assertEqual(a["revision"], 3)   # NOT quietly advanced
        self.assertTrue(a["banner"])

    def test_keep_mine_is_the_only_path_that_drops_expect_revision(self):
        self.assertTrue(self.r["forced"])
        self.assertNotIn("expect_revision", self.r["forcedBody"])
        self.assertEqual(self.r["afterForced"]["revision"], 8)
        self.assertFalse(self.r["afterForced"]["dirty"])

    def test_a_400_keeps_the_work_and_flags_the_clip(self):
        self.assertFalse(self.r["saveInvalid"])
        a = self.r["afterInvalid"]
        self.assertEqual([c[0] for c in a["clips"]], ["a", "b"])
        self.assertTrue(a["dirty"])
        self.assertEqual(a["flagged"], ["b"])

    def test_a_good_save_never_replaces_the_clips_under_the_users_hands(self):
        self.assertTrue(self.r["saveOk"])
        a = self.r["afterOk"]
        self.assertEqual([c[0] for c in a["clips"]], ["a"])
        self.assertFalse(a["dirty"])
        self.assertEqual(a["revision"], 9)
        self.assertEqual(a["unplaced"], 1)

    def test_times_are_printed_the_way_a_person_reads_them(self):
        self.assertEqual(self.r["fmt"], ["0:00.00", "0:09.50", "1:15.25"])

    # ---- WAVE 2: the three kinds -----------------------------------------
    def test_an_absent_kind_is_a_video_and_nonsense_is_too(self):
        # This is the whole v1 migration, on the client. Every clip in every
        # edit.json written before today has no `kind`; reading the default
        # rather than stamping it is what makes those documents correct the
        # moment they load, with nothing rewritten.
        self.assertEqual(self.r["kinds"],
                         ["video", "still", "slug", "video", "video"])

    def test_brightness_is_clamped_on_the_way_out_of_a_clip(self):
        # The same clamp the validator enforces on the way in. A slider that
        # could ask for a value the server refuses is a slider that produces a
        # red error box instead of a picture.
        self.assertEqual(self.r["bright"], [0, 0, 0.2, 0.5, -0.5, 0])

    def test_the_css_preview_matches_ffmpeg_at_mid_grey(self):
        # ffmpeg's eq=brightness is ADDITIVE; CSS brightness() is
        # MULTIPLICATIVE, and CSS has no additive form. They are matched where
        # a person judges exposure — mid-grey — and the drift at the ends is
        # why the strip says the preview is approximate.
        self.assertEqual(self.r["brightCss"], [1, 1.5, 2, 0, 0])
        lo, hi = self.r["brightCssMidGrey"]
        self.assertAlmostEqual(lo, hi, places=6)

    def test_setting_brightness_back_to_zero_removes_the_field(self):
        self.assertTrue(self.r["setBrightOk"])
        self.assertEqual(self.r["setBrightValue"], 0.3)
        self.assertEqual(self.r["setBrightHuman"], "human")
        self.assertEqual(self.r["setBrightClamped"], 0.5)
        # NEUTRAL IS ABSENT, exactly as normalise_edit writes it. Otherwise
        # every clip anybody ever selected carries a dead field forever.
        self.assertFalse(self.r["setBrightCleared"])
        self.assertFalse(self.r["setBrightNoop"])     # no undo step for a no-op
        self.assertEqual(self.r["setBrightGone"], "gone")

    def test_the_save_payload_synthesises_a_still_and_a_slug_the_servers_way(self):
        v, still, slug = self.r["kindPayload"]
        # A video keeps its window, its source duration and its grade, and
        # carries NO kind — absent is video, on the wire as well as on disk.
        self.assertNotIn("kind", v)
        self.assertEqual(v["adjust"], {"brightness": 0.25})
        self.assertEqual(v["dur"], 10)
        # A still is its slot: start 0, end = the hold, no source duration to
        # clamp the trim that is the only way to change it.
        self.assertEqual(still["kind"], "still")
        self.assertEqual((still["start"], still["end"]), (0, 3))
        self.assertIsNone(still["dur"])
        # A slug has no file at all, and a neutral grade is dropped.
        self.assertEqual(slug["kind"], "slug")
        self.assertIsNone(slug["path"])
        self.assertEqual((slug["start"], slug["end"]), (0, 1.5))
        self.assertNotIn("adjust", slug)

    # ---- WAVE 2: the drop maths ------------------------------------------
    def test_a_drop_lands_by_the_midpoint_rule(self):
        # Left half of a clip means before it, right half means after — the
        # rule every NLE uses, and the only reading that makes a drop exactly
        # on a boundary unambiguous.
        self.assertEqual(self.r["dropIdx"], [0, 0, 1, 1, 2, 3])
        self.assertEqual(self.r["dropIdxEmpty"], 0)

    def test_inserting_ripples_the_film_but_filling_a_hole_does_not(self):
        # Both behaviours are right and they are different verbs. A shot
        # generated for a hole must not move the cuts around it; a clip
        # dropped BETWEEN two others must, or the drop silently overwrites
        # whatever it landed on.
        self.assertEqual(self.r["insertRipple"],
                         [["a", 0, 2], ["NEW", 2, 3], ["b", 3, 5]])
        self.assertTrue(self.r["insertLengths"])
        self.assertTrue(self.r["insertNoOverlap"])
        self.assertEqual(self.r["insertHuman"], "human")
        # …and the un-rippled counterpart is still un-rippled.
        self.assertEqual(self.r["filled"],
                         [["a.mp4", 0, 2], ["fill.mp4", 3, 6], ["b.mp4", 9, 11]])

    def test_a_slug_arrives_with_no_path_and_no_source_duration(self):
        slug = self.r["insertSlug"]
        self.assertEqual(slug["kind"], "slug")
        self.assertIsNone(slug["path"])
        self.assertIsNone(slug["dur"])       # nothing for the trim to clamp to
        self.assertEqual(slug["len"], 2.5)
        self.assertTrue(self.r["insertSlugNoOverlap"])

    def test_a_still_is_resized_by_the_ordinary_trim_handles(self):
        still = self.r["insertStill"]
        self.assertEqual(still["kind"], "still")
        self.assertIsNone(still["dur"])
        self.assertEqual(still["at"], [0, 3])
        # No source clock means nothing to run out of: dragging the handle to
        # 9 s gives 9 s, where a video would have clamped at its own length.
        self.assertEqual(self.r["stillStretched"], 9)

    # ---- WAVE 2: reorder --------------------------------------------------
    def test_reorder_closes_the_hole_it_leaves_and_keeps_the_films_length(self):
        self.assertEqual(self.r["reorderIds"], ["c", "a", "b"])
        self.assertEqual(self.r["reorder"],
                         [["c", 0, 1, 0, 1], ["a", 0, 2, 1, 3],
                          ["b", 0, 3, 3, 6]])
        self.assertTrue(self.r["reorderLengths"])
        self.assertTrue(self.r["reorderNoOverlap"])
        self.assertEqual(self.r["reorderKeepsLength"], 6)
        self.assertEqual(self.r["reorderHuman"], "human")

    def test_reorder_refuses_a_locked_clip_and_a_missing_one(self):
        self.assertEqual(self.r["reorderLocked"], "locked")
        self.assertEqual(self.r["reorderGone"], "gone")

    def test_move_and_reorder_are_different_verbs_on_purpose(self):
        # A move puts a clip at a TIME and leaves a hole behind it — which is
        # the hole the generate control fills. A reorder puts it at a POSITION
        # and closes that hole. Shift picks; neither can be inferred.
        self.assertEqual(self.r["moveLeavesAHole"], [["b", 0], ["c", 2], ["a", 9]])

    # ---- the zoom slider's floor ----------------------------------------
    def test_the_zoom_floor_puts_the_WHOLE_film_in_the_window(self):
        # The defect this is the gate for: "when you get to the 36 seconds,
        # you cannot scroll." At 42 px/sec the owner's 71.6s film drew a
        # 3031px inner inside a 1108px box. The slider's left end is now
        # whatever px/sec makes that inner fit, so "all of it" is always one
        # drag away — and it is computed from the LIVE box, not a constant.
        self.assertAlmostEqual(self.r["fit"], 15.1432, places=3)
        self.assertTrue(self.r["fitFitsTheBox"])
        self.assertLessEqual(self.r["fitWidth"], 1108)

    def test_the_fit_survives_an_empty_film_and_an_unpainted_box(self):
        # Both of these are real states: a document with no clips, and the
        # first paint before the column has a width. Neither may produce a
        # zero, an infinity or a NaN, because the result is stored in SBE.pps
        # and every pixel on the track is multiplied by it.
        self.assertAlmostEqual(self.r["fitEmptyFilm"], 1084.0, places=1)
        self.assertGreater(self.r["fitNarrow"], 0)
        self.assertLess(self.r["fitNarrow"], 2)

    def test_the_slider_is_logarithmic_and_round_trips(self):
        lo, hi = self.r["sliderEnds"]
        self.assertAlmostEqual(lo, 15.14, places=2)      # the fit
        self.assertAlmostEqual(hi, 200.0, places=2)      # the ladder's top
        self.assertEqual(self.r["sliderRound"], [0, 250, 500, 750, 1000])
        self.assertEqual(self.r["sliderClamps"], [0, 1000])
        # Linear would put the whole useful range in the first fifth of the
        # travel: the arithmetic midpoint is 107 px/sec, the geometric one 55.
        self.assertLess(self.r["sliderMidIsNotArithmeticMean"], 80)

    # ---- zoom keeps the picture still ------------------------------------
    def test_zooming_holds_the_playhead_exactly_where_it_was(self):
        # The one thing nobody can check by eye and everybody notices: after
        # a zoom the frame you were looking at is under the same pixel.
        self.assertEqual(self.r["anchorOnPlayhead"], [36, 400])
        self.assertEqual(self.r["zoomHoldsThePlayhead"], 400)

    def test_a_playhead_off_screen_anchors_on_the_middle_of_the_view(self):
        # There is no frame on screen to hold, and lurching to one that is not
        # visible is worse than holding what the user is looking at.
        t, px = self.r["anchorFallsBackToTheMiddle"]
        self.assertAlmostEqual(t, 727.48, places=1)
        self.assertEqual(px, 554)                        # half of 1108

    def test_alt_wheel_anchors_on_the_time_under_the_pointer(self):
        self.assertEqual(self.r["anchorExplicit"][0], 30)

    def test_the_zoom_scroll_is_clamped_to_what_there_is_to_scroll(self):
        self.assertEqual(self.r["zoomScrollClamps"], [0, 5000])

    # ---- follow PAGES, it does not chase ---------------------------------
    def test_the_view_does_not_move_while_the_playhead_is_on_screen(self):
        # This early return is what stops the follow writing scrollLeft sixty
        # times a second and fighting the user's own panning.
        self.assertEqual(self.r["followStaysPut"], 0)

    def test_crossing_the_right_edge_pages_forward_ONCE(self):
        # Resolve's behaviour: one jump, the head lands near the left of a
        # fresh screenful, and the next frame moves nothing.
        self.assertAlmostEqual(self.r["followPages"], 1547.04, places=1)
        self.assertAlmostEqual(self.r["followLead"], 133.0, places=0)
        self.assertTrue(self.r["followSettled"])

    def test_the_follow_clamps_at_both_ends_of_the_film(self):
        self.assertEqual(self.r["followBack"], 0)
        self.assertEqual(self.r["followClampsHead"], 0)
        self.assertEqual(self.r["followClampsTail"], 4000)

    # ---- the monitor row -------------------------------------------------
    def test_the_two_monitors_fill_the_width_at_1440x900(self):
        # MEASURED, not assumed: 1110px of column and 307px of free height,
        # read off the Editor's own tab with the carwash film open. The whole
        # point of this pass is that the row leaves no black gutter, so the
        # first assertion is that the parts add up to the whole.
        f = self.r["fit1440"]
        self.assertTrue(f["fills"])
        self.assertEqual(f["prog"], [524, 295])
        self.assertEqual(f["src"], [350, 197])
        self.assertEqual(f["rail"], 212)
        self.assertEqual(f["progAspect"], 1.778)
        self.assertEqual(f["srcAspect"], 1.778)
        self.assertEqual(f["split"], 40)
        self.assertTrue(f["programIsBigger"])
        self.assertTrue(f["withinBudget"])

    def test_the_two_monitors_fill_the_width_at_1900x1000(self):
        f = self.r["fit1900"]
        self.assertTrue(f["fills"])
        self.assertEqual(f["split"], 40)
        self.assertTrue(f["programIsBigger"])
        self.assertTrue(f["withinBudget"])
        self.assertGreaterEqual(f["rail"], 200)
        self.assertLessEqual(f["rail"], 380)

    def test_a_tall_window_gives_the_leftover_HEIGHT_to_the_timeline(self):
        # When the width binds first the pair takes the width-derived height
        # at exactly 40/60 and stops. What is left over vertically is the
        # track's — `.sbe-tl` is the one row with flex-grow — which is why a
        # tall window gets a taller track instead of a dead band under the
        # sticky action bar.
        f = self.r["fitTall"]
        self.assertTrue(f["fills"])
        self.assertAlmostEqual(f["ratio"], 0.667, places=2)
        self.assertGreater(f["leftoverHeight"], 100)

    def test_a_short_window_widens_the_SOURCE_and_never_past_the_program(self):
        # Filling the row matters more than the exact split — a symmetric
        # black gutter is the bug — but the program is the thing being cut
        # and never becomes the smaller of the two.
        f = self.r["fitShort"]
        self.assertLessEqual(f["ratio"], 1.0)
        self.assertTrue(f["programIsBigger"])

    def test_the_picture_has_a_floor_and_the_rail_has_both(self):
        self.assertEqual(self.r["fitFloor"], 120)
        self.assertEqual(self.r["railClamps"][0], 380)   # never a second sidebar
        self.assertGreaterEqual(self.r["railClamps"][1], 200)   # never a slot


class TimelineMusic(unittest.TestCase):
    """Wave 4 — the soundtrack is a first-class object on the timeline."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.r = run_contract()

    def test_an_untrimmed_track_is_the_whole_track_at_film_zero(self):
        head, tail, fs, fe, trimmed = self.r["musicPlain"]
        self.assertEqual([head, tail, fs, fe], [0, 60, 0, 60])
        self.assertFalse(trimmed)

    def test_a_positive_offset_is_a_head_trim(self):
        # It has meant "the track second that plays at film 0" since before
        # the editor existed, and every edit.json on disk relies on that.
        self.assertEqual(self.r["musicOffsetIn"], [5, 0, 55])

    def test_a_negative_offset_starts_the_music_inside_the_film(self):
        # The direction the old `max(0, …)` clamp made unreachable, and half
        # of "back and forth however you want".
        self.assertEqual(self.r["musicOffsetOut"], [0, 4, 64])

    def test_trims_are_track_seconds_and_a_head_trim_does_not_ripple(self):
        head, tail, fs, fe, trimmed = self.r["musicTrimmed"]
        self.assertEqual([head, tail, fs, fe], [10, 20, 10, 20])
        self.assertTrue(trimmed)

    def test_no_soundtrack_is_not_a_crash(self):
        self.assertEqual(self.r["musicNone"], [0, None])

    def test_a_move_writes_the_offset_and_leaves_the_window_alone(self):
        offset, ts, te = self.r["musicMove"]
        self.assertEqual([ts, te], [10, 20])
        self.assertEqual(offset, 3)            # head 10 landing at film 7
        self.assertEqual(self.r["musicMoveLands"], 7)
        self.assertEqual(self.r["musicMoveHome"], 0)

    def test_trimming_the_left_edge_leaves_the_rest_where_it_was(self):
        offset, ts, te = self.r["musicTrimL"]
        self.assertEqual([offset, ts, te], [0, 12, None])
        # The seconds a head trim removes come back as silence in front — the
        # music that is left does NOT slide earlier.
        self.assertEqual(self.r["musicTrimLKeepsPlace"], 12)

    def test_neither_edge_can_cross_the_other_or_the_track(self):
        self.assertAlmostEqual(self.r["musicTrimLClamped"], 5.5, places=6)
        self.assertEqual(self.r["musicTrimLFloor"], 0)
        self.assertEqual(self.r["musicTrimR"], 42)
        self.assertEqual(self.r["musicTrimRClamped"], 60)
        self.assertAlmostEqual(self.r["musicTrimRFloor"], 30.5, places=6)

    def test_the_music_snaps_to_the_cuts_and_never_to_its_own_beat_grid(self):
        self.assertEqual(self.r["musicSnaps"], [0, 0, 2, 2, 4])
        self.assertEqual(self.r["musicSnapCatches"], 2)
        self.assertEqual(self.r["musicSnapIgnoresFar"], 3)
        self.assertEqual(self.r["musicSnapOverride"], 2.04)

    # ---- the scroller ----------------------------------------------------
    def test_the_timeline_runs_past_the_last_frame(self):
        # 4 s of film, so the slack is the 3 s floor — but the span floor for
        # a nearly-empty timeline wins, and either way there is track out
        # past the final cut to drag the music onto.
        self.assertGreater(self.r["spanShortFilm"], 4)
        self.assertGreaterEqual(self.r["spanShortFilm"], 10)

    def test_an_empty_timeline_still_has_a_ruler(self):
        self.assertEqual(self.r["spanEmpty"], 10)

    def test_music_placed_past_the_clips_extends_the_scroller(self):
        # 30 s of track starting 20 s into a 4 s film ends at 50 s, and the
        # timeline has to reach it or it could never have been dragged there.
        self.assertGreaterEqual(self.r["spanFollowsMusic"], 50)

    def test_the_slack_is_capped(self):
        # 300 s of film: 15 % would be 45 s of empty track on open.
        self.assertEqual(self.r["spanLongFilm"], 315)

    def test_undo_carries_the_music_and_still_accepts_the_old_shape(self):
        self.assertEqual(self.r["undoRestoresMusic"], 0)
        n, first, offset = self.r["undoLegacyShape"]
        self.assertEqual([n, first], [1, "z"])
        self.assertEqual(offset, 0)            # a clip snapshot leaves audio

    def test_one_undo_does_not_delete_a_soundtrack_nobody_edited(self):
        # `sbeFetchPeaks` fills SBE.audio and deliberately does NOT touch the
        # document, so on a film that was Prepared but never auto-edited the
        # snapshot carried null — and one ⌘Z of any clip edit hid the music
        # block, silenced the preview bed, and put the same null on the redo
        # stack so redo could not undo the undo.
        kept, after, doc = self.r["undoKeepsADiscoveredTrack"]
        self.assertEqual(kept, "/state/track.mp3")
        self.assertEqual(after, "/state/track.mp3")
        # ...and it is still not in the document: an undo must not be what
        # saves a soundtrack the arrangement was never cut to.
        self.assertIsNone(doc)

    def test_a_music_drag_lands_in_the_same_place_however_fast_the_mouse_is(self):
        # `out.offset = head - want` folded the PREVIOUS offset back into its
        # own answer, and a pointermove stream re-reads the mutated object
        # every event. On a positive-offset document the same six-second drag
        # landed at film 6 as six events and did not move the block at all as
        # one.
        d = self.r["musicDragIsFrameRateIndependent"]
        for key in ("zeroOne", "zeroSix", "headOne", "headSix", "trimSix"):
            self.assertAlmostEqual(d[key], 6, places=6, msg=key)

    def test_a_drag_that_moves_nothing_writes_nothing(self):
        # sbeOnMusicUp decides "did anything change" by comparing the three
        # fields, and `offset` used to change even when the block did not — so
        # a nudge that ended where it started marked the film dirty, burned an
        # undo step and queued a write.
        got, was, ts = self.r["musicDragNowhereChangesNothing"]
        self.assertEqual(got, was)
        self.assertEqual(ts, 10)


class SavingCannotFailQuietly(unittest.TestCase):
    """The twenty minutes the owner lost, as a gate.

    edit.json sat frozen on disk while he kept cutting, and the only thing on
    screen that knew was a small grey chip reading "not saved". Two defects
    made it possible — a save that arrived mid-flight was DROPPED, and the
    in-flight flag could stick after a throw — and one made it survivable for
    twenty minutes: the autosave's failure branch was silent by design.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.r = run_contract()
        cls.src = panel_source()

    def test_a_save_that_arrives_mid_flight_is_remembered(self):
        # ...AS THE KIND OF WRITE IT WAS. `sbeQueueSave` no longer saves — it
        # schedules a crash backup — so remembering the dropped save as a bare
        # `true` and re-queuing through that lane turned the second of two
        # rapid Save presses into a backup write, with edit.json left at the
        # older revision and the alarm reading clear.
        self.assertEqual(self.r["midFlightRemembered"], "backup")
        self.assertEqual(self.r["midFlightRemembersASave"], "save")
        fn = extract_function("sbeSave", self.src)
        self.assertIn("finally", fn)
        self.assertIn("if (again === 'save') sbeSave(quiet, force);", fn)

    def test_the_in_flight_flag_cannot_stick_after_a_throw(self):
        self.assertFalse(self.r["flagClearedAfterThrow"])
        raised, hidden, kind = self.r["throwRaisesAlarm"]
        self.assertTrue(raised)
        self.assertFalse(hidden)
        self.assertEqual(kind, "alarm")

    def test_a_failing_autosave_is_loud_even_though_nobody_asked(self):
        raised, hidden, why, kind = self.r["quietFailureIsLoud"]
        self.assertTrue(raised)
        self.assertFalse(hidden)
        self.assertIn("disk is full", why)
        self.assertEqual(kind, "alarm")

    def test_a_conflict_is_a_failure_to_store_and_says_so(self):
        raised, hidden = self.r["conflictIsLoud"]
        self.assertTrue(raised)
        self.assertFalse(hidden)

    def test_a_save_that_lands_takes_the_alarm_down(self):
        raised, hidden, dirty_at, dirty, kind = self.r["successClearsAlarm"]
        self.assertFalse(raised)
        self.assertTrue(hidden)
        self.assertEqual(dirty_at, 0)
        self.assertFalse(dirty)
        self.assertEqual(kind, "saved")

    # ---- the crash lane, as a sequence -----------------------------------
    def test_the_lane_keeps_writing_while_an_offer_is_open(self):
        # It used to refuse: one backup file per draft, so a new write would
        # have destroyed the work the offer held. The cure was worse — a chip
        # nobody dismissed switched the safety net off for the rest of the
        # session, silently. The lane is versioned now (one file per snapshot,
        # pruned), so a new snapshot cannot eat an old one.
        wrote, fetches, stamped = self.r["backupWritesEvenWithAnOfferOpen"]
        self.assertTrue(wrote)
        self.assertEqual(fetches, 1)
        self.assertTrue(stamped)

    def test_a_save_answers_the_offer_and_repaints_the_bar(self):
        # THE BUG THIS IS FOR: the ok branch cleared `dirty` and the alarm and
        # left `SBE.backup` set, so the amber bar stayed on screen over a saved
        # film with a "Recover it" button that would have reverted the save.
        backup, repainted, drafts = self.r["saveAnswersTheOffer"]
        self.assertIsNone(backup)
        self.assertTrue(repainted)
        self.assertEqual(drafts, 1)      # ...and the drafts rows are refreshed

    def test_the_backup_lane_is_alive_again_after_a_save(self):
        # The compound failure: with the offer never answered, every later
        # backup no-opped, `backedUpAt` stayed 0, and the twelve-second
        # watchdog raised the red NOT SAVED alarm on a healthy panel.
        wrote, fetches, stamped = self.r["backupLivesAfterASave"]
        self.assertTrue(wrote)
        self.assertEqual(fetches, 1)
        self.assertTrue(stamped)

    def test_the_backup_says_which_draft_it_was_composed_from(self):
        self.assertEqual(self.r["backupNamesItsDraft"], "draft-1")

    def test_a_backup_that_does_not_land_raises_the_alarm(self):
        wrote, raised = self.r["backupFailureIsLoud"]
        self.assertFalse(wrote)
        self.assertTrue(raised)

    def test_an_autoplay_refusal_never_becomes_a_stored_preference(self):
        # The second way the Editor could go permanently silent: Chrome
        # declines an unmuted play(), the editor mutes itself so the picture
        # still runs (right call) — and then WROTE that to localStorage, so one
        # refusal silenced clips AND soundtrack across every later reload.
        fn = extract_function("sbeSetMute", self.src)
        self.assertIn("remember !== false", fn)
        self.assertIn("localStorage.setItem('sbeMuted'", fn)
        for site in ("sbePlay", "sbeSrcPlay"):
            body = extract_function(site, self.src)
            if "browser blocked sound" not in body:
                continue
            self.assertIn("'browser blocked sound — click \u1f50a to unmute'"
                          .replace("\u1f50a", "\U0001f50a"), body + body)
        # Neither refusal site may persist it.
        self.assertEqual(self.src.count(
            "sbeSetMute(true, 'browser blocked sound — click \U0001f50a to unmute')"), 0)
        self.assertEqual(self.src.count(
            "sbeSetMute(true, 'browser blocked sound — click \U0001f50a to unmute', false)"), 2)
        # ...and it says so where somebody will see it, not in a 10.5px line.
        self.assertIn("phosToast", fn)
        self.assertIn("Your setting has not been changed", fn)

    def test_the_speaker_button_is_the_way_back_from_a_refusal(self):
        # Pressing it IS the user gesture the browser was waiting for, so the
        # recovery path restarts the picture and the bed rather than only
        # flipping a flag.
        fn = extract_function("sbeUnmuteFromRefusal", self.src)
        self.assertIn("sbeSetMute(false)", fn)
        self.assertIn("sbeMusicPlay()", fn)
        self.assertIn("sbeUnmuteFromRefusal()", self.src)

    def test_the_banner_says_the_one_sentence_that_matters(self):
        el = self.src[self.src.index('id="sbeAlarm"'):]
        el = el[:el.index("</div>")]
        self.assertIn("SAVING IS FAILING", el)
        self.assertIn("your changes are not being stored", el)
        self.assertIn("Try again", el)

    def test_the_tick_notices_work_that_is_not_reaching_the_disk(self):
        fn = extract_function("sbeTick", self.src)
        # Re-queue a dropped save, and raise the alarm past the grace — BEFORE
        # any of the early returns this function is full of.
        self.assertLess(fn.index("SBE.dirty && !SBE.conflict"),
                        fn.index("SBE.prepare || {}"))
        self.assertIn("sbeQueueSave()", fn)
        self.assertIn("SBE_SAVE_GRACE_MS", fn)


class SplitEditsInTheBrowser(unittest.TestCase):
    """"I need to be able to leave some of the audio and drag only the image."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.r = run_contract()
        cls.src = panel_source()

    def test_a_clip_carries_its_own_sound_until_somebody_says_otherwise(self):
        start, end, film, linked = self.r["audioLinkedByDefault"]
        self.assertEqual([start, end, film], [0, 4, 4])
        self.assertTrue(linked)

    def test_unlinking_writes_the_window_the_clip_already_had(self):
        ok, payload = self.r["unlinkWritesTheSameWindow"]
        self.assertTrue(ok)
        self.assertEqual(json.loads(payload),
                         {"start": 0, "end": 4, "film_start": 4})
        # ...and re-linking leaves no trace, so the document is identical to
        # one that was never split.
        self.assertIsNone(self.r["relinkRemovesTheField"])

    def test_a_linked_strip_cannot_be_dragged_by_accident(self):
        ok, why = self.r["linkedRefusesTheDrag"]
        self.assertFalse(ok)
        self.assertIn("unlink", why)

    def test_the_j_cut_moves_the_sound_and_not_the_picture(self):
        ok, audio_at, film_at, src_in = self.r["jCut"]
        self.assertTrue(ok)
        self.assertEqual(audio_at, 3)      # her voice arrives a second early
        self.assertEqual(film_at, 4)       # ...and her picture does not move
        self.assertEqual(src_in, 0)        # nor does the window into her file

    def test_the_l_cut_runs_the_sound_on_under_the_next_picture(self):
        ok, audio_end, video_end, film_end = self.r["lCut"]
        self.assertTrue(ok)
        self.assertEqual(audio_end, 6)     # two more seconds of his line
        self.assertEqual(video_end, 4)     # his picture is untouched
        self.assertEqual(film_end, 4)
        # ...and it cannot invent audio the source does not have.
        self.assertEqual(self.r["lCutClamped"], 10)

    def test_a_head_trim_leaves_the_rest_where_it_was(self):
        src_in, film_at, video_in = self.r["headTrim"]
        self.assertEqual(src_in, 1)        # one second later into the file
        self.assertEqual(film_at, 5)       # ...landing one second later
        self.assertEqual(video_in, 0)      # the picture's window is untouched

    def test_a_head_trim_past_the_source_never_moves_the_out_point(self):
        # The one boundary every audio case above stays inside. `start`
        # clamped at 0 while `film` applied the full, unclamped delta, so
        # pulling the sound's head back further than the take allows dragged
        # the strip's right edge left with it — the L-cut you were keeping,
        # quietly shortened.
        q = self.r["headTrimPastTheSource"]
        self.assertAlmostEqual(q["outAt1"], q["wasOut"], places=6)
        self.assertAlmostEqual(q["outAt0"], q["wasOut"], places=6)
        # ...and the in-point stops at the head of the source, as it must.
        self.assertEqual(q["start1"], 0)
        self.assertEqual(q["start0"], 0)

    def test_a_locked_clip_and_a_still_both_refuse(self):
        self.assertFalse(self.r["lockedRefuses"])
        ok, why = self.r["stillRefuses"]
        self.assertFalse(ok)
        self.assertIn("video clip", why)

    def test_the_three_layers_are_in_the_owners_order(self):
        inner = self.src[self.src.index('id="sbeInner"'):]
        inner = inner[:inner.index('id="sbeHead"')]
        for a, b in (("sbeRuler", "sbeTrack"), ("sbeTrack", "sbeAudioLane"),
                     ("sbeAudioLane", "sbeWave")):
            self.assertLess(inner.index('id="%s"' % a),
                            inner.index('id="%s"' % b))

    def test_the_toggle_is_offered_on_a_video_clip_and_only_there(self):
        fn = extract_function("sbePaintInspector", self.src)
        self.assertIn("sbeToggleAudioLink()", fn)
        self.assertIn("kind === 'video'", fn)
        self.assertIn("Unlink sound", fn)

    def test_the_height_floor_is_the_sum_of_its_lanes(self):
        # sbeFitMonitors budgets the monitors against this constant, so a lane
        # added without moving it steals the difference from the picture — and
        # that is not hypothetical: the floor was 190, itemised in a comment
        # that predated the overlay lane, so the box spent three releases 30px
        # shorter than its own contents with `overflow-y: hidden` over the
        # difference. The bottom of the soundtrack was drawn where nobody
        # could see it. Both ends are now COMPUTED here rather than matched as
        # strings, so a fifth lane turns this red instead of clipping.
        chrome = int(re.search(r"const SBE_TL_CHROME = (\d+);",
                               self.src).group(1))
        lanes = re.findall(
            r"\{ key: '(\w+)',\s+base:\s*(\d+), cap:\s*(\d+), share: ([\d.]+) \}",
            self.src)
        self.assertEqual([l[0] for l in lanes], ["ov", "track", "alane", "wave"])
        # These two are read by the layout harness's injected probe JS, so
        # the editor module publishes them as globalThis properties rather
        # than module-private consts (slice 3, docs/ARCHITECTURE.md).
        floor = int(re.search(r"globalThis\.SBE_TL_MIN_H = (\d+);", self.src).group(1))
        ceil_ = int(re.search(r"globalThis\.SBE_TL_MAX_H = (\d+);", self.src).group(1))
        self.assertEqual(floor, chrome + sum(int(l[1]) for l in lanes))
        self.assertEqual(ceil_, chrome + sum(int(l[2]) for l in lanes))
        # And the CSS fallback — what the page shows for the frame before the
        # JS runs — is the same number.
        self.assertIn("min-height: var(--sbe-tl-h, %dpx);" % floor, self.src)
        # Every lane's CSS fallback is its base, and the picture's max-height
        # is its cap. If those disagree with the table, flex fights the
        # distribution and the pixels go somewhere nobody chose.
        base = dict((l[0], l[1]) for l in lanes)
        cap = dict((l[0], l[2]) for l in lanes)
        self.assertIn("var(--sbe-ov-h, %spx)" % base["ov"], self.src)
        self.assertIn("var(--sbe-track-h, %spx); max-height: %spx;"
                      % (base["track"], cap["track"]), self.src)
        self.assertIn("var(--sbe-alane-h, %spx)" % base["alane"], self.src)
        self.assertIn("var(--sbe-wave-h, %spx)" % base["wave"], self.src)

    def test_the_dragged_height_goes_where_the_owner_asked_for_it(self):
        # "...enabling expansion in case you have some sound editing in
        # there." Shares, not proportions: scaling every lane by the same
        # factor would give the picture track 41% of the drag and the sound
        # strip 14%, which is the opposite of the sentence above.
        lanes = dict((m[0], float(m[3])) for m in re.findall(
            r"\{ key: '(\w+)',\s+base:\s*(\d+), cap:\s*(\d+), share: ([\d.]+) \}",
            self.src))
        self.assertAlmostEqual(sum(lanes.values()), 1.0, places=6)
        self.assertGreaterEqual(lanes["alane"] + lanes["wave"], 0.75)
        self.assertGreater(lanes["alane"], lanes["track"])
        # "They don't need to be this big": after the rebalance the picture
        # takes the SMALLEST share of a dragged pixel bar the overlay strip.
        self.assertLess(lanes["track"], lanes["wave"])


class ThePanelsClientPARSES(unittest.TestCase):
    """The whole script, as one file, through a real parser.

    Every other gate in this suite EXTRACTS the function it is about and runs
    that. So a stray paren three characters outside any function body — one too
    many closing an `escapeHtml(` argument — left 250 client tests green while
    the served page threw `SyntaxError: Unexpected token ')'` on load, `SBE`
    was never defined, and the Editor did not exist in the browser at all. The
    per-function harness is structurally incapable of seeing that: it never
    concatenates the file the browser is handed.

    This does. It is the cheapest test in the suite and it is the only one that
    fails when the panel is unopenable.
    """

    def test_the_served_script_parses_as_javascript(self):
        if NODE is None:
            raise unittest.SkipTest("node not on PATH")
        # The page lives on disk since slice 2 of the extraction
        # (docs/ARCHITECTURE.md) — webapp/index.html IS the file the
        # browser is handed, placeholders aside.
        page = (Path(__file__).resolve().parent
                / "webapp" / "index.html").read_text(encoding="utf-8")
        blocks = re.findall(r"<script>(.*?)</script>", page, re.S)
        self.assertTrue(blocks, "no <script> block found in the panel HTML")
        for i, js in enumerate(blocks):
            with tempfile.NamedTemporaryFile("w", suffix=".js",
                                             delete=False) as fh:
                fh.write(js)
                path = Path(fh.name)
            try:
                r = subprocess.run([NODE, "--check", str(path)],
                                   capture_output=True, text=True, timeout=60)
                self.assertEqual(r.returncode, 0,
                                 "script block %d does not parse:\n%s"
                                 % (i, r.stderr))
            finally:
                path.unlink(missing_ok=True)


class TheNounIsSingleSourced(unittest.TestCase):
    """"These are not films... something less bad than film."

    One noun, three forms, replaced at serve time — so changing his mind again
    costs one line and nothing else.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.src = panel_source()

    def test_there_is_exactly_one_place_to_change_it(self):
        self.assertIn('SEQ_NOUN = "sequence"', self.src)
        self.assertIn('SEQ_NOUN_PL = "sequences"', self.src)
        self.assertIn('SEQ_NOUN_CAP = "Sequence"', self.src)
        # ...and the template is what carries it, not 30 hardcoded strings.
        self.assertIn('.replace("__SEQ__", SEQ_NOUN)', self.src)
        self.assertIn('.replace("__SEQS__", SEQ_NOUN_PL)', self.src)
        self.assertIn('.replace("__SEQCAP__", SEQ_NOUN_CAP)', self.src)

    def test_the_longest_token_is_replaced_first(self):
        # `__SEQ__` is a prefix of neither, but `__SEQS__` and `__SEQCAP__`
        # both START with `__SEQ`, so replacing the short one first would
        # leave "sequenceS__" and "sequenceCAP__" on the screen.
        order = [self.src.index('.replace("__SEQCAP__"'),
                 self.src.index('.replace("__SEQS__"'),
                 self.src.index('.replace("__SEQ__", SEQ_NOUN)')]
        self.assertEqual(order, sorted(order))

    def test_the_editor_surface_carries_no_bare_film_noun(self):
        # The Editor's own markup: the pool tabs, the empty state, the
        # inspector summary. Internal field names are exempt by construction —
        # they are the schema, and a label is not worth a data migration.
        # The strings the owner named, by name. A slice-scan would be broader
        # and would also catch the STORYBOARD tab, where "Film" means the
        # finished render — a different noun for a different thing, and not
        # what he asked to rename.
        for gone in ('data-src="film">This film<',
                     'data-src="other">Other films<',
                     "' of film'",
                     'title="This film\'s shots, other films\' clips',
                     'Every draft this film has',
                     "'Open a film first"):
            self.assertNotIn(gone, self.src, gone)
        for now in ('>This __SEQCAP__<', '>Other __SEQS__<',
                    "' of the __SEQ__'", "'Open a __SEQ__ first"):
            self.assertIn(now, self.src, now)

    def test_no_token_survives_into_the_served_page(self):
        # The badge is concatenated in Python AFTER the template pass, so a
        # token there is a token the user reads. It uses the constant instead.
        import re as _re
        page = panel_html_render()
        self.assertEqual(_re.findall(r"__SEQ\w*", page), [])
        self.assertIn("This Sequence", page)
        self.assertIn("Other sequences", page)

    def test_the_schema_words_are_untouched(self):
        for keep in ("film_start", "film_end", "sbeFilmDuration"):
            self.assertIn(keep, self.src, keep)


class TheStripIsAnEditingSurface(unittest.TestCase):
    """Waveforms, a level line with keyframes, and half a clip you can delete."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.r = run_contract()
        cls.src = panel_source()

    def test_the_waveform_is_the_STRIPS_window_not_the_pictures(self):
        # Two different source windows must draw two different slices, or a
        # trimmed strip and a J-cut both show the wrong seconds.
        first, later = self.r["waveSliceIsTheStripsWindow"]
        self.assertNotEqual(first, later)
        self.assertEqual(first, [[-0.04, 0.04], [-0.09, 0.09]])
        self.assertEqual(later, [[-0.54, 0.54], [-0.59, 0.59]])

    def test_one_column_over_many_buckets_takes_the_extremes(self):
        # Otherwise a quiet frame in the middle of a loud second draws a hole
        # that is not there.
        self.assertEqual(self.r["waveSliceHandlesZoomOut"], [[-0.99, 0.99]])
        self.assertEqual(self.r["waveSliceNoPeaksIsEmpty"], 0)

    def test_one_waveform_per_SOURCE_asked_for_once(self):
        fn = extract_function("sbeWaveWant", self.src)
        self.assertIn("SBE.clipPeaks", fn)
        self.assertIn("hasOwnProperty", fn)     # asked once, cached forever
        # A take with no audio is a FACT, remembered as false so the lane
        # never asks again.
        self.assertIn("false", fn)
        self.assertIn("clip-peaks", fn)

    def test_a_point_is_added_moved_and_deleted(self):
        self.assertEqual(json.loads(self.r["kfAdd"]), {"points": [[2, 0.4]]})
        self.assertEqual(json.loads(self.r["kfMoveAndSort"]),
                         [[3, 0.2], [3.5, 0.9]])   # re-sorted past its neighbour
        self.assertEqual(json.loads(self.r["kfMoveClamps"]), [[4, 1]])
        self.assertTrue(self.r["kfDeleteLeavesNoTrace"])
        self.assertEqual(json.loads(self.r["kfDeleteKeepsAFade"]),
                         {"fade_in": 1})
        self.assertEqual(self.r["kfLocked"], "locked")

    def test_two_points_on_one_second_are_refused(self):
        # A discontinuity the envelope cannot express and the NLEs would
        # import out of order.
        ok, why = self.r["kfNoTwoOnOneSecond"]
        self.assertFalse(ok)
        self.assertIn("already a point", why)

    def test_the_gesture_precedence_on_the_strip_is_written_down(self):
        down = extract_function("sbeOnAudioDown", self.src)
        # sync flag -> corner handle -> keyframe dot -> grips -> body
        self.assertLess(down.index(".sbe-sync"), down.index(".sbe-kf"))
        self.assertLess(down.index(".sbe-kf"), down.index(".sbe-fade-h"))
        self.assertLess(down.index(".sbe-fade-h"), down.rindex(".sbe-aclip'"))
        # SHIFT-CLICK deletes: a modifier rather than a second affordance
        # drawn on a 6px target.
        self.assertIn("ev.shiftKey", down)
        self.assertIn("sbeDeleteKeyframe", down)
        # DOUBLE-CLICK adds, so plain drag stays "move the strip".
        dbl = extract_function("sbeOnAudioDbl", self.src)
        self.assertIn("sbeAddKeyframe", dbl)
        self.assertIn(".sbe-kf", dbl)                # not on top of a point
        self.assertIn(".sbe-fade-h", dbl)            # nor on a corner
        self.assertIn("dblclick", self.src)

    def test_deleting_the_strip_leaves_the_picture_playing_silent(self):
        # "You cannot unlock the clip and delete the upper part, nor delete
        # the lower part of the sound."
        ok, no_audio, flagged, reads_muted, after, before = \
            self.r["deleteStrip"]
        self.assertTrue(ok)
        self.assertTrue(no_audio)
        # Absent `audio` alone would mean LINKED — the clip's own sound would
        # play again, the opposite of what was asked for. The mute is what
        # makes the silence real in all three outputs.
        self.assertTrue(flagged)
        self.assertTrue(reads_muted)
        self.assertEqual(after, before)     # the picture does not move
        self.assertEqual(after, [0, 4])

    def test_deleting_the_strip_is_not_a_ripple(self):
        before, after = self.r["deleteStripIsNotARipple"]
        self.assertEqual(before, after)

    def test_a_clip_that_is_already_silent_says_so(self):
        ok, why = self.r["deleteStripTwiceIsRefused"]
        self.assertFalse(ok)
        self.assertIn("already silent", why)

    def test_the_verb_is_offered_only_once_the_halves_are_separate(self):
        # On a linked clip "delete the sound" and "mute" would be the same
        # button twice.
        fn = extract_function("sbePaintInspector", self.src)
        self.assertIn("sbeDeleteStripSel()", fn)
        i = fn.index("sbeDeleteStripSel()")
        self.assertIn("sbeClipAudio(c).split", fn[max(0, i - 400):i])

    def test_the_overlay_lane_teaches_itself(self):
        # An empty lane is a sentence, not a blank — the convention the track
        # and the four pool sources already follow.
        fn = extract_function("sbePaintOverlays", self.src)
        self.assertIn("sbe-track-empty", fn)
        self.assertIn("drop a still here", fn)
        self.assertIn("media pool", fn)


class TheSoundsEnvelopeOnTheClient(unittest.TestCase):
    """The preview has to move the same number the render and export do."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.r = run_contract()
        cls.src = panel_source()

    def test_a_flat_unity_curve_is_no_curve(self):
        self.assertEqual(self.r["afxFlatIsNothing"], [0, 1])

    def test_the_client_curve_is_the_servers_curve(self):
        # Byte-for-byte the same breakpoints the API suite pins for the same
        # inputs — two implementations of one model is how they drift.
        self.assertEqual(self.r["afxSimpleCase"],
                         [[0, 0], [1, 1], [3.5, 1], [4, 0]])
        self.assertEqual(self.r["afxKeyframes"],
                         [[0, 0.3], [1, 0.3], [2, 1], [4, 1]])
        self.assertEqual(self.r["afxCompose"],
                         [[0, 0], [1, 0.25], [2, 0.25], [4, 0.25]])
        self.assertEqual(self.r["afxClamp"], [2, 2])

    def test_the_gain_at_a_second_is_the_previews_answer(self):
        self.assertEqual(self.r["afxGainAt"], [0, 0.5, 1, 1, 0.5, 0])

    def test_setting_a_sound_fade_and_clearing_it(self):
        on, gone, source = self.r["afxSet"]
        self.assertEqual(json.loads(on), {"fade_in": 1})
        self.assertTrue(gone)                 # neutral is absent
        self.assertEqual(source, "human")
        self.assertEqual(self.r["afxSetClampsToTheStrip"], 4)
        self.assertEqual(self.r["afxLocked"], "locked")

    def test_the_strip_player_moves_the_gain(self):
        fn = extract_function("sbeStripSync", self.src)
        # STRIP-relative AND on the played clock: a J-cut that slides its
        # sound must not drag the ramp with it, and a retimed strip's second
        # is `(source second - in-point) / speed` (see audio_gain_points).
        self.assertIn("sbeGainAt(c2, win.len, (w.at - win.start) / win.speed)", fn)
        self.assertIn("w.at - win.start", fn)

    def test_the_bed_plays_the_WHOLE_mix_and_not_just_its_envelope(self):
        # The preview used to read the bed's `afx` and nothing else, while the
        # render held the bed at a hard-coded 0.20 and ducked it against the
        # dialogue. `sbeBedGainAt` is the one function both sides now read.
        fn = extract_function("sbeMusicSync", self.src)
        self.assertIn("a.volume = sbeBedGainAt(", fn)
        # ON THE BED'S OWN CLOCK — the played window, not the track — so a
        # soundtrack that was trimmed or dragged does not slide its own fade.
        self.assertIn("SBE.playhead - bw.film_start", fn)

    def test_the_strip_grows_the_SAME_corner_handle_the_picture_has(self):
        self.assertIn('data-afade="in"', self.src)
        self.assertIn('data-afade="out"', self.src)
        lane = extract_function("sbePaintAudioLane", self.src)
        self.assertIn("sbeAudioFadeMarks(c, w)", lane)

    def test_the_gesture_precedence_is_the_same_on_both_lanes(self):
        # The corner handle sits over the left grip's hit area, so it is
        # tested FIRST — one rule for both lanes, or the strip behaves
        # differently from the block above it for no reason a user could name.
        down = extract_function("sbeOnAudioDown", self.src)
        self.assertIn(".sbe-fade-h", down)
        # rindex: the strip BODY is the last thing tested, after the flag, the
        # keyframe dot and the corner handle — most specific gesture first.
        self.assertLess(down.index(".sbe-fade-h"), down.rindex(".sbe-aclip"))
        move = extract_function("sbeOnAudioMove", self.src)
        self.assertIn("d.mode === 'afade'", move)
        # ...and it resolves before the strip's own move/trim maths.
        self.assertLess(move.index("'afade'"), move.index("sbeAudioEdit"))

    def test_the_inspector_separates_picture_fades_from_sound_fades(self):
        fn = extract_function("sbePaintInspector", self.src)
        # THREE sections, which is what the effects model documents. The
        # sound's ramps live inside Sound rather than under a fourth heading:
        # two adjacent headings both saying "sound" pushed the inspector past
        # 460px in the rail.
        self.assertIn("sect('Effects'", fn)
        self.assertNotIn("sect('Sound fades'", fn)
        self.assertIn("sbeAudioFadeCommit", fn)
        sound = fn[fn.index("sect('Sound'"):fn.index("sect('Effects'")]
        self.assertIn("sbeAudioFadeCommit", sound)


class TheOverlayLaneOnTheClient(unittest.TestCase):
    """A second video track, above the picture. His endcard's lane."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.r = run_contract()
        cls.src = panel_source()

    def test_a_png_is_a_card_and_an_mp4_is_a_clip(self):
        self.assertEqual(self.r["ovKind"], ["still", "video", "video"])

    def test_a_card_covers_its_own_window_and_no_more(self):
        self.assertEqual(self.r["ovAt"], [False, True, True, False])

    def test_moving_one_moves_ONE_and_a_still_is_its_slot(self):
        ok, fs, fe, st, en, source = self.r["ovMove"]
        self.assertTrue(ok)
        self.assertEqual([fs, fe], [10, 13])
        self.assertEqual([st, en], [0, 3])     # the still follows its slot
        self.assertEqual(source, "human")

    def test_one_lane_means_no_stacking(self):
        ok, why = self.r["ovNoStacking"]
        self.assertFalse(ok)
        self.assertIn("already there", why)

    def test_trimming_changes_only_its_own_window(self):
        right, left = self.r["ovTrim"]
        self.assertEqual(right, [8, 5])        # end moved, length followed
        self.assertEqual(left, [4, 8, 4])

    def test_adding_lands_somewhere_free(self):
        ok, fs, fe, kind = self.r["ovAddSlidesPastAnother"]
        self.assertTrue(ok)
        self.assertEqual([fs, fe], [6, 8])     # pushed past the card at 3–6
        self.assertEqual(kind, "still")

    def test_removing_a_card_is_not_a_ripple(self):
        # The lane is a set of placements, not a queue: nothing else moves.
        ok, left, other_at = self.r["ovDeleteMovesNothing"]
        self.assertTrue(ok)
        self.assertEqual(left, 1)
        self.assertEqual(other_at, 8)

    def test_a_card_takes_its_fades_from_the_SAME_accessor(self):
        # The proof the effects foundation was worth building: an overlay is
        # not a second fade implementation.
        fade_in, ramp = self.r["ovFades"]
        self.assertEqual(fade_in, 1.0)
        self.assertEqual(ramp, [0, 0.5, 1, 1])
        self.assertEqual(self.r["ovFadeClamp"], 3)   # cannot outrun the card

    def test_the_lane_is_drawn_above_the_picture_track(self):
        # Lane order on screen is stacking order in the sequence.
        html = self.src[self.src.index('id="sbeInner"'):]
        self.assertLess(html.index('id="sbeOverlayLane"'),
                        html.index('id="sbeTrack"'))
        self.assertIn("sbePaintOverlays()", extract_function("sbePaint", self.src))

    def test_the_stage_carries_its_own_opacity(self):
        fn = extract_function("sbeOvPaint", self.src)
        self.assertIn("sbeFadeOpacityAt(o, now)", fn)
        self.assertIn("sbeOvAt(SBE.overlays, now)", fn)
        self.assertIn("sbeOvPaint();", extract_function("sbeFrame", self.src))
        # BEFORE the picture's early returns. A card may outlive the last shot,
        # and `sbeShowFrameAt` returns early on "nothing plays here" — which
        # left the card stuck on screen at full opacity past its own end.
        show = extract_function("sbeShowFrameAt", self.src)
        self.assertIn("sbeOvPaint(t);", show)
        self.assertLess(show.index("sbeOvPaint(t);"),
                        show.index("nothing plays here"))

    def test_the_lane_travels_with_the_document_and_the_undo_stack(self):
        self.assertIn("edit.overlays", extract_function("sbeSaveBody", self.src))
        self.assertIn("SBE.overlays", extract_function("sbeAdopt", self.src))
        snap = extract_function("sbeSnapshot", self.src)
        self.assertIn("overlays: SBE.overlays", snap)
        # An older snapshot carries no lane, and restoring `undefined` over a
        # live one would delete a card nobody asked to remove.
        rest = extract_function("sbeRestore", self.src)
        self.assertIn("s.overlays !== undefined", rest)

    def test_an_image_in_the_pool_can_become_a_card(self):
        self.assertIn("edPoolOverlay(", self.src)
        self.assertIn("ed-pool-ov", self.src)
        fn = extract_function("edPoolOverlay", self.src)
        self.assertIn("sbeOvAddAt", fn)
        self.assertIn("SBE.playhead", fn)


class AHoleShorterThanAFrameOnTheClient(unittest.TestCase):
    """The client half of the black-frame report. The server heals what is
    already on disk; this is what stops a gesture from making another one."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.r = run_contract()
        cls.src = panel_source()

    def test_under_a_frame_is_no_gap_and_anything_larger_is_untouched(self):
        got = self.r["gridGapCollapsesSubFrameOnly"]
        self.assertEqual(got[:3], [0, 0, 0])
        # NOT `round(gap / frame)`: his 0.503-frame hole would round UP to a
        # whole frame of black, which is the same bug one frame louder.
        self.assertAlmostEqual(got[3], 1 / 24, places=6)
        self.assertEqual(got[4:], [0.5, 1.8])

    def test_his_film_lays_out_with_the_cuts_touching(self):
        shape, holes, lengths = self.r["holedFilmLaysOutContiguous"]
        self.assertEqual([row[3] for row in shape], [0, 4, 8, 12])
        self.assertEqual(holes, 0)
        # ...without breaking the invariant the server refuses an edit for.
        self.assertTrue(lengths)

    def test_a_drag_cannot_open_one(self):
        ok, film_start, holes = self.r["dragCannotOpenASubFrameHole"]
        self.assertTrue(ok)
        self.assertEqual(film_start, 4)
        self.assertEqual(holes, 0)

    def test_a_gap_you_can_see_survives_a_drag(self):
        film_start, holes = self.r["dragKeepsAGapYouCanSee"]
        self.assertEqual(film_start, 6.5)
        self.assertEqual(holes, 1)

    def test_the_heal_carries_the_unlinked_strip(self):
        before, after, anchor = self.r["adoptCarriesTheUnlinkedStrip"]
        self.assertEqual(before, after)
        self.assertEqual(anchor, 3.5)

    def test_the_hole_counter_stops_lying(self):
        old, truth, now = self.r["holeCounterOnHisFilm"]
        # What the header said, what was actually there, and what is left.
        self.assertEqual([old, truth, now], [1, 3, 0])

    def test_the_threshold_is_derived_from_the_rate_not_written_as_1_48(self):
        fn = extract_function("sbeHoles", self.src)
        self.assertIn("sbeFps()", fn)
        self.assertNotIn("1 / 48", fn)


class ACardOnABlackPlateIsFixedOnTheWayIn(unittest.TestCase):
    """"If the picture comes in a format that doesn't work, automatically make
    it work" — and then say so, quietly, with the original one click away.

    The server half is in `test_storyboard_editor_api`. This is the half that
    decides WHICH LANE it happens on, whether the receipt can push the timeline
    down the screen, and whether "Keep original" actually restores anything.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.r = run_contract()
        cls.src = panel_source()

    def test_it_happens_on_the_way_to_the_OVERLAY_lane_and_nowhere_else(self):
        # A still on the PICTURE lane legitimately keeps its black — it is the
        # whole frame there. So the call sits in the overlay verb, and the
        # pool's ordinary add never learns about it.
        fn = extract_function("edPoolOverlay", self.src)
        self.assertIn("/storyboard/edit/overlay-key", fn)
        self.assertTrue(fn.lstrip().startswith("async "),
                        "the key has to be awaited before the card is placed")
        self.assertNotIn("overlay-key", extract_function("edPoolAdd", self.src))

    def test_it_posts_the_body_the_panel_can_actually_read(self):
        # MEASURED IN A BROWSER, not reasoned about. `_storyboard_post` parses
        # urlencoded bodies; a `FormData` sends multipart and arrives as an
        # empty form, so the route answered "that image is not in this panel's
        # outputs" about a file sitting in the outputs folder and the card went
        # on the lane still plated. Twenty green unit tests did not see it,
        # because the fake handler is handed the form dict directly.
        fn = extract_function("edPoolOverlay", self.src)
        self.assertIn("new URLSearchParams()", fn)
        self.assertNotIn("new FormData", fn)
        # ...which is the same shape the pool's other verb uses.
        self.assertIn("new URLSearchParams()",
                      extract_function("edPoolAdd", self.src))

    def test_a_card_that_could_not_be_measured_is_still_placed(self):
        # A panel that cannot answer must not cost the user their card.
        fn = extract_function("edPoolOverlay", self.src)
        self.assertIn("catch", fn)
        self.assertLess(fn.index("catch"), fn.index("sbeOvAddAt"))

    def test_the_receipt_is_a_quiet_chip_in_the_ONE_notice_surface(self):
        el = extract_element("sbeKeyed", self.src)
        self.assertIn('data-quiet="1"', el)
        self.assertIn('data-short="Black background removed"', el)
        # The sentence and the way back, in the markup a gate can read.
        body = self.src[self.src.index('id="sbeKeyed"'):]
        body = body[:body.index("</div>")]
        self.assertIn("Black background removed", body)
        self.assertIn("Keep original", body)
        self.assertIn("sbeKeyedKeepOriginal()", body)
        # Inside the container, so it can never become a fifth stacked banner.
        wrap = self.src[self.src.index('id="sbeNotices"'):]
        self.assertLess(wrap.index('id="sbeKeyed"'), wrap.index("sbe-vers"))

    def test_it_is_the_LEAST_urgent_thing_on_the_screen(self):
        # It reports something that already worked. A save that is failing
        # outranks it, always.
        order = self.src[self.src.index("const SBE_NOTICE_ORDER"):]
        order = order[:order.index("]")]
        self.assertIn("'sbeKeyed'", order)
        self.assertGreater(order.index("'sbeKeyed'"), order.index("'sbeAlarm'"))
        self.assertGreater(order.index("'sbeKeyed'"), order.index("'sbeErrors'"))

    def test_the_gates_copy_of_the_urgency_order_is_the_panels(self):
        # This constant is duplicated into the harness above. It was a comment
        # asking to be kept equal and nothing that checked — so a notice added
        # to the panel was silently uncovered here.
        panel_list = re.search(
            r"const SBE_NOTICE_ORDER = (\[[^\]]*\])", self.src).group(1)
        mine = re.search(r"const SBE_NOTICE_ORDER = (\[[^\]]*\])", SHIM).group(1)
        norm = lambda s: re.findall(r"'([a-zA-Z]+)'", s)      # noqa: E731
        self.assertEqual(norm(panel_list), norm(mine))

    def test_keep_original_restores_the_file_that_was_never_touched(self):
        ok, path, fs, fe, untouched = self.r["ovKeepOriginal"]
        self.assertTrue(ok)
        self.assertEqual(path, "/x/a.png")
        # Only the pointer moved: the card keeps its place and its length.
        self.assertEqual([fs, fe], [3, 6])
        # ...and the input array is left alone, which is what undo walks back.
        self.assertEqual(untouched, "/x/a.keyed.png")

    def test_keep_original_on_a_card_that_is_gone_changes_nothing(self):
        self.assertEqual(self.r["ovKeepOriginalOnAGoneCard"], [False, "gone"])

    def test_keep_original_is_an_ordinary_undoable_edit(self):
        fn = extract_function("sbeKeyedKeepOriginal", self.src)
        self.assertIn("sbeOvMutate", fn)
        # Not a private path around the door every other lane edit goes through.
        self.assertNotIn("SBE.overlays =", fn)

    def test_the_add_hands_back_the_row_it_added(self):
        # The receipt needs the id to be able to undo it, and guessing which
        # row is new is how the wrong card gets reverted.
        self.assertIn("return res.added", extract_function("sbeOvAddAt", self.src))

    def test_a_load_clears_a_receipt_that_no_longer_points_at_anything(self):
        self.assertIn("sbeKeyedDismiss", extract_function("sbeAdopt", self.src))

    def test_the_receipt_alone_is_still_a_chip(self):
        got = self.r["noticeKeyedAloneIsAChip"]
        self.assertEqual(got["folded"], ["sbeKeyed"])
        self.assertFalse(got["wrapHidden"])

    def test_a_failing_save_takes_the_width_and_the_receipt_sits_beside_it(self):
        got = self.r["noticeKeyedNeverOutranksTheAlarm"]
        self.assertEqual(got["lead"], "sbeAlarm")
        self.assertEqual(got["folded"], ["sbeKeyed"])

    def test_with_everything_open_the_receipt_is_last(self):
        got = self.r["noticeKeyedIsLastOfAll"]
        self.assertEqual(got["lead"], "sbeConflict")
        # Every other notice folds too — the surface is ONE ROW, and adding a
        # fifth citizen to it must not make it a stack. Measured in a browser
        # at 1920: the timeline sits at the same y with one chip open as with
        # two, and the surface costs 39px either way.
        self.assertEqual(got["folded"],
                         ["sbeAlarm", "sbeErrors", "sbeRecover", "sbeKeyed"])


class TheEffectsModelOnTheClient(unittest.TestCase):
    """docs/EDITOR_EFFECTS_MODEL.md, client side — and the inspector it asked
    for: "just set the base to have effects somewhere"."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.r = run_contract()
        cls.src = panel_source()

    def test_absent_means_no_effect(self):
        self.assertEqual(self.r["fxAbsentIsNothing"], [0, 0, 0])

    def test_one_accessor_whatever_the_storage(self):
        # Fades live in `fx`; brightness stays where history put it. A
        # consumer never has to know that.
        self.assertEqual(self.r["fxOneAccessor"], [0.5, 0.25])

    def test_the_client_clamps_exactly_as_the_server_does(self):
        both, lopsided = self.r["fxClamp"]
        self.assertEqual(both, [2, 2])
        total, in_is_bigger = lopsided
        self.assertEqual(total, 4)
        self.assertTrue(in_is_bigger)      # proportional, not truncated

    def test_the_preview_opacity_is_a_value_per_frame(self):
        # A 4 s clip, 0.5 s in and 1.0 s out. A scrub must show what is TRUE
        # at the second it landed on, not an animation that started on arrival.
        self.assertEqual(self.r["fxOpacityRamp"], [0, 0.5, 1, 1, 1, 0.5, 0])
        self.assertEqual(self.r["fxNoFadeIsAlwaysOpaque"], [1, 1])

    def test_clearing_a_fade_leaves_no_trace(self):
        after_clear, gone, source = self.r["fxSet"]
        self.assertEqual(json.loads(after_clear), {"fade_in": 0.5})
        self.assertTrue(gone)              # both cleared: no `fx` at all
        self.assertEqual(source, "human")

    def test_setting_one_clamps_against_the_other(self):
        self.assertEqual(self.r["fxSetClamps"], 3.2)   # 3.2 + 0.8 = the length

    def test_a_locked_clip_refuses(self):
        self.assertEqual(self.r["fxLockedRefuses"], "locked")

    def test_it_reaches_the_save_payload(self):
        self.assertEqual(json.loads(self.r["fxReachesTheSavePayload"]),
                         {"fade_in": 0.5, "fade_out": 1})

    def test_the_inspector_has_three_sections_and_effects_holds_brightness(self):
        # The reorganization IS the request: brightness was floating in the
        # middle of a flat run of buttons.
        fn = extract_function("sbePaintInspector", self.src)
        for name in ("'Clip'", "'Sound'", "'Effects'"):
            self.assertIn("sect(" + name, fn, name)
        eff = fn[fn.index("sect('Effects'"):]
        self.assertIn("adjust", eff)                  # brightness lives here
        self.assertIn("fadeRow('in'", eff)
        self.assertIn("fadeRow('out'", eff)
        # ...and the sections are real markup a gate can read.
        self.assertIn(".sbe-sect-h", self.src)

    def test_intuitive_first_precise_second(self):
        # The corner handle is how a fade is set in every NLE ever made; the
        # numeric field beside it is the exact second.
        self.assertIn('data-fade="in"', self.src)
        self.assertIn('data-fade="out"', self.src)
        self.assertIn(".sbe-fade-h", self.src)
        down = extract_function("sbeOnTrackDown", self.src)
        self.assertIn(".sbe-fade-h", down)
        # The handle is checked BEFORE the grips: it sits over the left grip's
        # hit area, and a fade drag is the more specific gesture.
        self.assertLess(down.index(".sbe-fade-h"), down.index(".sbe-grip"))
        move = extract_function("sbeOnTrackMove", self.src)
        self.assertIn("d.mode === 'fade'", move)
        self.assertIn("sbeSetFade", move)
        self.assertIn("sbeFadeCommit", extract_function("sbePaintInspector", self.src))

    def test_the_preview_paints_the_ramp_from_the_playhead(self):
        self.assertIn("sbeFadePaint();", extract_function("sbeFrame", self.src))
        self.assertIn("sbeFadePaint(t);",
                      extract_function("sbeShowFrameAt", self.src))
        fn = extract_function("sbeFadePaint", self.src)
        self.assertIn("sbeFadeOpacityAt", fn)
        self.assertIn("style.opacity", fn)
        # THE CLIP AT THE PLAYHEAD, not `curId` — that is the transport's own
        # bookkeeping and is only current while something plays, so a SCRUB
        # read the stale clip or none at all. Caught on the test panel: the
        # stage stayed fully opaque while the playhead sat inside a ramp.
        self.assertIn("sbeClipAt(SBE.clips, now)", fn)
        self.assertLess(fn.index("sbeClipAt"), fn.index("SBE.curId"))

    def test_the_block_draws_the_ramps_it_has(self):
        self.assertIn("sbeFadeMarks(c)", extract_function("sbePaintTrack", self.src))
        self.assertIn(".sbe-cl-fade", self.src)


class ThePreviewPlaysWhatTheDocumentSays(unittest.TestCase):
    """The J-cut was inaudible, and that was the whole complaint.

    The transport plays ONE <video> at a time and enters each clip at its
    picture boundary, so a clip's sound always landed with its picture no
    matter where its strip sat. Correct on disk, correct in the render's
    concat lanes, correct in the NLE export — and missing from the one place
    the user checks his work: "no matter what I do, this is always out."
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.r = run_contract()
        cls.src = panel_source()

    def test_exactly_one_thing_carries_a_clips_sound(self):
        # Both would play the same seconds twice, a beat apart.
        got = self.r["stripOwnership"]
        self.assertEqual(got["plain"], [False, True])   # the picture element
        self.assertEqual(got["split"], [True, False])   # the strip player
        self.assertEqual(got["muted"], [False, False])  # neither
        self.assertEqual(got["still"], [False, False])  # nothing to play

    def test_HIS_j_cut_is_audible_before_the_picture_arrives(self):
        # rev 97: strip at film 30.35, picture cut at 30.77.
        got = self.r["stripAcrossHisCut"]
        self.assertEqual(got["before"], [])                  # 29.5: not yet
        self.assertEqual(got["lead"], [["c6", 0.05]])        # 30.4: his line
        self.assertEqual(got["edge"], [["c6", 0.41]])        # 30.76: still early
        self.assertEqual(got["after"], [["c6", 0.65]])       # 31.0: across it
        self.assertEqual(got["late"], [["c6", 3.95]])
        self.assertEqual(got["ended"], [])                   # 34.392 and done

    def test_the_sound_and_the_picture_MEET_at_the_cut(self):
        # At film 30.77 the strip is at source 0.42 — which is exactly clip
        # 6's picture in-point. That is what makes his edit sync rather than
        # merely early, and it is the arithmetic the whole feature turns on.
        self.assertEqual(self.r["stripMapping"], [0, 0.42, 1, 4.04])

    def test_two_strips_sound_at_once_across_a_cut(self):
        # An L-cut tail and a J-cut head overlap by construction. The render's
        # concat lane resolves that by trimming the outgoing tail; the preview
        # sums them, which is what the person cutting needs to hear.
        got = self.r["stripOverlap"]
        self.assertEqual(got["both"], [["a", 4.5], ["b", 1]])
        self.assertEqual(got["onlyA"], ["a"])
        self.assertEqual(got["onlyB"], ["b"])
        # ...and the pool has a voice for each, with one spare.
        self.assertIn("const SBE_STRIP_VOICES = 3;", self.src)

    def test_a_muted_strip_is_silent_by_not_being_there(self):
        self.assertEqual(self.r["stripMuted"], [0, 0])

    def test_a_coupled_pair_is_played_at_its_frozen_offset(self):
        owned, at = self.r["stripCoupled"]
        self.assertTrue(owned)
        self.assertEqual(at, [["a", 0.5]])

    def test_an_ordinary_clip_is_not_the_players_business(self):
        self.assertEqual(self.r["stripIgnoresOrdinary"], 0)

    def test_the_picture_element_yields_whenever_a_strip_exists(self):
        fn = extract_function("sbeLoadInto", self.src)
        self.assertIn("!sbePictureCarriesSound(c)", fn)

    def test_the_player_is_driven_by_the_playhead_from_every_door(self):
        # Every frame while playing (a strip starts on its own clock, not at
        # the cuts), forced on a seek and on play, stopped on stop, and
        # re-synced after any edit — sbePaint is where every mutation, undo,
        # redo and adopt lands.
        self.assertIn("sbeStripSync();", extract_function("sbeFrame", self.src))
        self.assertIn("sbeStripSync(true);", extract_function("sbePlay", self.src))
        self.assertIn("sbeStripSync(true);", extract_function("sbeSeek", self.src))
        self.assertIn("sbeStripStop();", extract_function("sbeStop", self.src))
        self.assertIn("sbeStripSync();", extract_function("sbePaint", self.src))

    def test_a_seek_re_seeks_rather_than_tolerating_slip(self):
        fn = extract_function("sbeStripSync", self.src)
        self.assertIn("SBE_STRIP_SLIP", fn)
        self.assertIn("fresh || force", fn)
        # A voice is released BEFORE any is claimed, or an overlap of two
        # finds the pool full of yesterday's clips.
        # rindex: the FIRST `for (const w of want)` builds the live map; the
        # CLAIM loop is the last one, and that is the one release must precede.
        self.assertLess(fn.index("for (const a of pool)"),
                        fn.rindex("for (const w of want)"))
        # ...and the global mute reaches it.
        self.assertIn("a.muted = !!SBE.muted;", fn)


class TheLoadPathRewritesNothing(unittest.TestCase):
    """sb_carwash rev 97, on the client side.

    His symptom was the loaded board differing from disk, so the load path is
    the first suspect. It is byte-stable, there is no client-side repair, and
    this is the gate that keeps it that way.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.r = run_contract()
        cls.src = panel_source()

    def test_the_strip_survives_adopt_and_layout_untouched(self):
        got = self.r["loadRewritesNothing"]
        self.assertEqual(json.loads(got["audioUntouched"]),
                         {"start": 0.0, "end": 4.042, "film_start": 30.35})
        self.assertTrue(got["firstHasNoStrip"])
        self.assertTrue(got["diskWasNotMutated"])

    def test_his_j_cut_reads_as_in_sync_on_the_client_too(self):
        # He trimmed 0.42 s off the picture head and left the sound reaching
        # back, so the same source second still plays at the same film second.
        self.assertEqual(self.r["loadRewritesNothing"]["drift"], [0, 0])

    def test_the_save_payload_is_what_the_disk_gave_it(self):
        rows = json.loads(self.r["loadRewritesNothing"]["roundTrips"])
        self.assertEqual(rows[0][:5], ["c5", 25.99, 30.032, 0, 4.042])
        self.assertIsNone(rows[0][5])
        self.assertEqual(rows[1][:5], ["c6", 30.77, 34.392, 0.42, 4.042])
        self.assertEqual(rows[1][5],
                         {"start": 0.0, "end": 4.042, "film_start": 30.35})

    def test_there_is_no_client_side_repair_to_gate(self):
        # The server's legacy repair is a one-time, marker-gated migration.
        # The client has no mirror of it, and adding one would put a second
        # author back inside the load path.
        for name in ("sbeAdopt", "sbeAdoptGaps", "sbeLayout"):
            fn = extract_function(name, self.src)
            self.assertNotIn("repair", fn.lower(), name)


class MutingAClipsOwnSound(unittest.TestCase):
    """"We should have an option to mute the clip sound."

    An H3 shot arrives with baked-in wind and ambience under the line, and on a
    music cut that is not a performance to be balanced — it is noise to remove
    so the track can carry the moment.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.r = run_contract()
        cls.src = panel_source()

    def test_absent_is_audible_and_that_is_the_migration(self):
        self.assertFalse(self.r["muteAbsentIsAudible"])

    def test_the_toggle_writes_one_flag_and_stamps_the_clip_human(self):
        ok, flag, reads, source = self.r["muteWritesTheFlag"]
        self.assertTrue(ok)
        self.assertTrue(flag)
        self.assertTrue(reads)
        self.assertEqual(source, "human")

    def test_unmute_restores_exactly_what_was_there(self):
        same, gone = self.r["muteUnmuteLeavesNoTrace"]
        self.assertTrue(same)
        self.assertTrue(gone)

    def test_mute_and_unlink_are_independent_in_both_directions(self):
        after, muted_then_unlinked, split = self.r["muteComposesWithUnlink"]
        # A J-cut that is then muted keeps its strip exactly where it was put:
        # `mute` describes the sound WHEREVER its strip happens to be.
        self.assertTrue(after["muted"])
        self.assertTrue(after["split"])
        self.assertEqual(after["snd"], [2, 6])
        # ...and muting first, then unlinking, reaches the same place.
        self.assertTrue(muted_then_unlinked)
        self.assertTrue(split)

    def test_there_must_be_something_to_switch_off(self):
        ok, why = self.r["muteRefusesASilentSource"]
        self.assertFalse(ok)
        self.assertIn("no sound of its own", why)
        self.assertIn("video clip", self.r["muteRefusesAStill"])

    def test_undo_walks_it_back_and_redo_puts_it_on(self):
        on, off, again = self.r["muteIsUndoable"]
        self.assertEqual([on, off, again], [True, False, True])

    def test_it_reaches_the_save_payload(self):
        self.assertTrue(self.r["muteSurvivesTheSavePayload"])

    def test_the_strip_stays_and_says_muted(self):
        # The decision has to be visible and reversible in the place it was
        # made, so the strip is struck through rather than removed.
        fn = extract_function("sbePaintAudioLane", self.src)
        self.assertIn("sbeClipMuted(c)", fn)
        self.assertIn("is-silenced", fn)
        self.assertIn("MUTED", fn)
        self.assertIn(".sbe-aclip.is-silenced", self.src)
        # ...and it is NOT the same state as a file with no audio track.
        self.assertIn(".sbe-aclip.is-mute", self.src)

    def test_the_inspector_offers_it_on_any_video_clip_with_sound(self):
        fn = extract_function("sbePaintInspector", self.src)
        self.assertIn("sbeToggleClipMute()", fn)
        self.assertIn("Mute sound", fn)
        self.assertIn("Unmute sound", fn)
        self.assertIn("c.has_audio !== false", fn)

    def test_the_PREVIEW_is_the_third_output_and_agrees(self):
        # The render drops the clip's lane and the export disables its audio
        # track; the preview has to agree, or the one place the user checks his
        # work is the one place the decision does not exist.
        fn = extract_function("sbeLoadInto", self.src)
        # Muting now goes through the ownership helper, because a muted clip
        # and a clip whose sound the STRIP PLAYER owns both mean the same
        # thing to the picture element: not yours.
        self.assertIn("!!SBE.muted || !sbePictureCarriesSound(c)", fn)
        self.assertIn("sbeClipMuted(c)",
                      extract_function("sbePictureCarriesSound", self.src))


class ThePassiveViewerThatDisarmedTheNet(unittest.TestCase):
    """The owner lost unsaved work because somebody else OPENED his film.

    Seven hours of snapshots stopped and nothing on screen said so. The chain,
    each link measured on the code as it was:

      1. `GET /storyboard/edit?...&session=T` claimed the board unconditionally,
         so a page LOAD took the claim — a second window, a headless browser,
         an agent reading the board, a preview.
      2. The tab he was cutting in got 409 `stale_session` on its next
         snapshot and set `SBE.superseded = true`, permanently.
      3. `sbeQueueSave` returned on that flag BEFORE the line that sets
         `SBE.dirtyAt`, so the timestamp froze at its last pre-supersede value
         while `SBE.backedUpAt` stayed AHEAD of it.
      4. The 12-second watchdog gates on `backedUpAt < dirtyAt`. With the pair
         frozen that way it read "this work is backed up" forever, so the one
         thing built to catch a dead net could never fire.
      5. The single 9-second toast expired and the state line it set was
         overwritten by the very next edit. The tab looked completely normal.

    Rule 5 of the ruling is still honoured — a second tab is not a second
    opinion — but it is honoured by TELLING, never by stopping.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.src = panel_source()

    def test_a_read_claims_nothing(self):
        # The whole outage in one line: opening a film is not editing it.
        get = panel_source()
        head = get[get.index("def _storyboard_edit_get"):]
        head = head[:head.index("def _storyboard_post")]
        self.assertNotIn("claim_session", head)
        self.assertIn("A READ CLAIMS NOTHING", head)

    def test_writing_is_what_claims_the_board(self):
        src = (ROOT / "storyboard_editor.py").read_text()
        fn = src[src.index("def write_backup("):]
        fn = fn[:fn.index("\ndef ", 10)]
        self.assertIn("claim_session(board_dir, session)", fn)
        # ...and it does not refuse anybody.
        self.assertNotIn("raise EditError(\"this film is open in a newer", fn)

    def test_there_is_no_flag_that_can_switch_the_writer_off(self):
        self.assertNotIn("superseded: false", self.src)
        self.assertNotIn("SBE.superseded = true", self.src)
        queue = extract_function("sbeQueueSave", self.src)
        self.assertNotIn("return;", queue.split("SBE.dirtyAt")[0])

    def test_the_watchdogs_clock_is_armed_before_any_early_return(self):
        # Link 3. `dirtyAt` must be the FIRST thing the queue does, so no guard
        # added later can starve the watchdog the way `superseded` did.
        queue = extract_function("sbeQueueSave", self.src)
        body = queue[queue.index("{") + 1:]
        self.assertLess(body.index("SBE.dirtyAt = Date.now()"),
                        body.index("setTimeout"))
        for line in body.split("\n"):
            stripped = line.strip()
            if stripped.startswith("//") or not stripped:
                continue
            self.assertIn("dirtyAt", stripped)
            break

    def test_a_refused_snapshot_raises_the_alarm_like_any_other_failure(self):
        backup = extract_function("sbeBackup", self.src)
        # `stale_session` used to be intercepted as a non-failure ABOVE the
        # alarm. There is no interception left: every path that does not write
        # reaches the same sentence.
        self.assertNotIn("if (r && r.stale_session)", backup)
        self.assertIn("sbeSaveAlarm(", backup)
        self.assertNotIn("older session", backup)

    def test_the_alarm_is_the_persistent_kind_not_a_toast(self):
        # Link 5. A toast expires and a state string is overwritten by the next
        # edit; the alarm stays up until a write actually lands.
        alarm = extract_function("sbeSaveAlarm", self.src)
        self.assertNotIn("phosToast", alarm)
        clear = extract_function("sbeSaveAlarmClear", self.src)
        self.assertTrue(clear)

    def test_a_second_editor_is_information_and_nothing_more(self):
        backup = extract_function("sbeBackup", self.src)
        self.assertIn("SBE.otherEditor", backup)
        paint = extract_function("sbePaintProtected", self.src)
        self.assertIn("also open elsewhere", paint)


class TheNetSaysWhenItLastCaughtYou(unittest.TestCase):
    """There was no way — for the user OR for anybody helping him — to see
    that the snapshot lane had gone quiet. It was inferred from a directory
    listing, after the loss."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.r = run_contract()
        cls.src = panel_source()

    def test_it_says_how_long_ago_in_the_place_people_already_look(self):
        self.assertIn('id="sbeProtected"', self.src)
        self.assertLess(self.src.index('id="sbeState"'),
                        self.src.index('id="sbeProtected"'))

    def test_fresh_reads_as_protected_and_stale_reads_as_NOT(self):
        fresh, cold, never = self.r["protectedChip"]
        self.assertIn("protected", fresh["text"])
        self.assertFalse(fresh["cold"])
        # Past the watchdog's own grace, the chip goes cold on the SAME
        # threshold the banner alarms on — two indicators that can disagree
        # are one indicator and one bug.
        self.assertIn("NOT PROTECTED", cold["text"])
        self.assertTrue(cold["cold"])
        self.assertIn("not backed up yet", never["text"])

    def test_cold_is_stated_against_the_last_SUCCESSFUL_write(self):
        # Not against `dirtyAt`: the outage froze that pair in a way that read
        # "protected" for seven hours. How long ago the net actually caught
        # this tab cannot be faked by a stuck flag.
        fn = extract_function("sbePaintProtected", self.src)
        self.assertIn("SBE.backedUpAt", fn)
        cold = fn[fn.index("const cold"):]
        cold = cold[:cold.index(";")]
        self.assertNotIn("dirtyAt", cold)


class OneNoticeSurface(unittest.TestCase):
    """Four full-width banners in one column, and they are not mutually
    exclusive.

    Conflict, the save alarm, the validation list and the recovery offer each
    took the full width and stacked. The day the overlap rule refused a J-cut
    the owner had three of them up at once — the timeline pushed off the bottom
    of the screen, and the sentence that mattered was the third one down. They
    still exist as themselves; the surface decides which one is OPEN and folds
    the rest to a chip.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.r = run_contract()
        cls.src = panel_source()

    def test_no_notice_is_no_surface_at_all(self):
        self.assertTrue(self.r["noticeNothingIsNoSurface"]["wrapHidden"])

    def test_a_lone_notice_is_never_a_chip(self):
        # Folding the only thing on screen would hide the sentence to save
        # room nothing else is asking for.
        got = self.r["noticeAloneIsOpen"]
        self.assertEqual(got["lead"], "sbeAlarm")
        self.assertEqual(got["folded"], [])
        self.assertFalse(got["wrapHidden"])

    def test_the_snapshot_offer_is_a_chip_even_when_it_is_alone(self):
        # NO RECOVERY WALL. It was a full-width bar with a primary button,
        # asking a question that had to be answered before the film could be
        # worked on. It is an invitation to go and look.
        got = self.r["noticeQuietIsAlwaysAChip"]
        self.assertEqual(got["folded"], ["sbeRecover"])
        self.assertFalse(got["wrapHidden"])
        self.assertIn('data-quiet="1"', self.src)
        # ...and it points at the place versions live rather than acting.
        el = self.src[self.src.index('id="sbeRecover"'):]
        el = el[:el.index("</div>")]
        self.assertIn("sbeVersionsOpen()", el)
        self.assertNotIn('class="primary"', el)

    def test_urgency_decides_and_it_is_not_the_dom_order(self):
        # The recovery offer is a question about last session; it waits behind
        # every notice that is about the file in front of you.
        got = self.r["noticeUrgencyOrder"]
        self.assertEqual(got["lead"], "sbeAlarm")
        self.assertEqual(got["folded"], ["sbeErrors", "sbeRecover"])

    def test_a_conflict_outranks_everything(self):
        got = self.r["noticeConflictWinsEverything"]
        self.assertEqual(got["lead"], "sbeConflict")
        self.assertEqual(got["folded"], ["sbeAlarm", "sbeErrors", "sbeRecover"])

    def test_clicking_a_chip_opens_it(self):
        got = self.r["noticeClickOpensAChip"]
        self.assertEqual(got["lead"], "sbeRecover")
        self.assertEqual(got["folded"], ["sbeAlarm"])
        # ...and the handler is bound on the CONTAINER, so it survives every
        # repaint of the children and needs no rebinding anywhere.
        self.assertIn('onclick="sbeNoticeClick(event)"', self.src)
        self.assertIn(".is-folded", extract_function("sbeNoticeClick", self.src))

    def test_a_lead_that_closes_hands_the_surface_back(self):
        # Otherwise the override outlives its notice and the next one to open
        # arrives folded behind a chip nobody is looking for.
        got = self.r["noticeLeadThatClosesReleases"]
        self.assertEqual(got["lead"], "sbeAlarm")
        self.assertEqual(got["folded"], [])

    def test_later_is_not_discard(self):
        # Discard DELETES the backup, so it was never the button for "not
        # now" — and with no third option the bar sat across the top of the
        # film for the rest of the session over a question the user was not
        # ready to answer.
        hidden, kept = self.r["noticeLaterHidesAndKeeps"]
        self.assertTrue(hidden)
        self.assertTrue(kept)
        self.assertIn("!!SBE.backupHidden",
                      extract_function("sbePaintRecovery", self.src))
        # A NEW offer is not the one that was dismissed.
        self.assertIn("SBE.backupHidden = false", self.src)

    def test_the_validation_list_shows_one_line_and_folds_the_rest(self):
        # Nine clips failing the same rule printed nine copies of the same
        # sentence and pushed the timeline off the screen.
        fn = extract_function("sbeRenderErrors", self.src)
        self.assertIn("SBE.errsOpen", fn)
        self.assertIn("more", fn)
        self.assertIn("sbeErrsToggle()", fn)

    def test_every_banner_still_says_its_own_sentence(self):
        # The surface is CSS and a class, never a rewrite: folding may not be
        # the reason a message stops existing.
        for marker in ("SAVING IS FAILING", "Unsaved snapshot",
                       "The timeline was not saved.", "Load theirs", "Keep mine",
                       "Restore it", "Discard", "Later", "Open Versions"):
            self.assertTrue(marker in self.src, marker)
        for el_id in ("sbeConflict", "sbeAlarm", "sbeErrors", "sbeRecover"):
            self.assertIn('id="%s"' % el_id, self.src)
            self.assertIn('data-short=', self.src)


class TheSoundStaysWhereItWasPut(unittest.TestCase):
    """The other half of the J-cut: what happens to the strip AFTERWARDS.

    `audio.film_start` is an absolute film anchor and `sbeLayout` re-derives the
    film position of every clip from the running total of the lead gaps — the
    PICTURE's position and nothing else. So a right trim, a move, a ripple
    delete, an insert or a reorder slid everything downstream while every
    unlinked strip stood still, and a split edit made three shots earlier came
    apart without anybody touching it. Measured on the code as it was:

        trim the tail of shot 1  → shot 3 is +1.00s out
        move shot 2 by hand      → shot 3 is -2.00s out
        ripple-delete shot 2     → shot 3 is +4.00s out
        split an unlinked clip   → BOTH halves carry {0,4}@4 — one strip
                                   duplicated, which the server refuses as
                                   `clips_audio_overlap`
        head-trim with no room   → the picture slips inside its own slot and
                                   takes its own sound out by -1.00s

    The owner, cutting his film: "instead of allowing me to remove or move what
    video is visible while leaving the sound intact and then rematching it, it
    is actually getting the audio out of sync."
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.r = run_contract()
        cls.src = panel_source()

    def test_unlinking_leaves_the_pair_exactly_where_it_was(self):
        s = self.r["pairUnlink"]
        self.assertEqual(s["vid"], [4, 8])
        self.assertEqual(s["snd"], [4, 8])
        self.assertEqual(s["ssrc"], [0, 4])
        self.assertEqual(s["drift"], 0)
        self.assertFalse(s["linked"])

    def test_a_head_trim_does_not_move_the_sound_and_does_not_have_to(self):
        # The in-point and the slot move together, so the frames that remain
        # play at the film second they always did. The seconds of picture that
        # went away leave the sound reaching a second early — a J-cut for free
        # — and the pair is still IN SYNC, which is the whole claim.
        b, c = self.r["pairTrimHead"]
        self.assertEqual(b["vid"], [5, 8])
        self.assertEqual(b["vsrc"], [1, 4])
        self.assertEqual(b["snd"], [4, 8])          # not touched
        self.assertEqual(b["ssrc"], [0, 4])         # not re-windowed
        self.assertEqual(b["drift"], 0)
        # ...and a head trim ripples nothing, so the shot after it is untouched.
        self.assertEqual(c["vid"], [8, 12])

    def test_a_tail_trim_carries_the_sound_of_everything_it_ripples(self):
        # Shot 1 loses a second; shot 3's picture is rigidly translated back by
        # that second and its unlinked sound goes with it. This is the case the
        # owner was hitting from three shots away.
        a, c = self.r["pairTrimTailRipples"]
        self.assertEqual(a["vid"], [0, 3])
        self.assertEqual(c["vid"], [7, 11])
        self.assertEqual(c["snd"], [7, 11])         # was [8, 12] before the fix
        self.assertEqual(c["ssrc"], [0, 4])         # the window is never rewritten
        self.assertEqual(c["drift"], 0)

    def test_moving_a_picture_by_hand_leaves_ITS_OWN_sound_behind(self):
        # The one gesture whose point is to break the pair — and the flag says
        # by how much rather than pretending it did not happen.
        b, c = self.r["pairMove"]
        self.assertEqual(b["vid"], [6, 10])
        self.assertEqual(b["snd"], [4, 8])          # never silently dragged
        self.assertEqual(b["drift"], -2)            # the sound now runs early
        # ...while the shot the move pushed keeps its own pair together.
        self.assertEqual(c["vid"], [10, 14])
        self.assertEqual(c["snd"], [10, 14])        # was [8, 12] before the fix
        self.assertEqual(c["drift"], 0)

    def test_ripple_delete_pulls_the_sound_up_with_the_picture(self):
        s = self.r["pairRippleDelete"]
        self.assertEqual(s["vid"], [4, 8])
        self.assertEqual(s["snd"], [4, 8])          # was [8, 12] before the fix
        self.assertEqual(s["drift"], 0)

    def test_a_linked_clip_is_never_written_to_by_the_carry(self):
        # An absent `audio` means LINKED. Writing one to carry a ripple would
        # unlink a clip behind the user's back.
        absent, drift = self.r["pairCarryLeavesLinkedAlone"]
        self.assertTrue(absent)
        self.assertEqual(drift, 0)

    def test_split_cuts_the_strip_once_instead_of_copying_it_twice(self):
        ok, left, right = self.r["pairSplit"]
        self.assertTrue(ok)
        self.assertEqual(left["vid"], [4, 6])
        self.assertEqual(left["snd"], [4, 6])
        self.assertEqual(left["ssrc"], [0, 2])
        self.assertEqual(right["vid"], [6, 8])
        self.assertEqual(right["snd"], [6, 8])      # butt-joined, not overlapped
        self.assertEqual(right["ssrc"], [2, 4])
        self.assertEqual([left["drift"], right["drift"]], [0, 0])

    def test_split_keeps_the_drift_a_pair_already_had(self):
        # The cut is expressed in the source clock the two halves share, so a
        # J-cut survives being split rather than snapping back into sync.
        ok, dl, dr, snd_l, snd_r = self.r["pairSplitKeepsTheDrift"]
        self.assertTrue(ok)
        self.assertEqual([dl, dr], [-2, -2])
        self.assertEqual(snd_l, [4, 6])
        self.assertEqual(snd_r, [6, 8])

    def test_split_refuses_a_cut_the_strip_does_not_reach(self):
        # "This half has no sound" is a state the document cannot express — an
        # absent `audio` means LINKED, which would invent sound the film never
        # had — so the answer is an honest no rather than a guess.
        ok, why, ssrc = self.r["pairSplitOutsideTheStrip"]
        self.assertFalse(ok)
        self.assertIn("does not reach this cut", why)
        self.assertEqual(ssrc, [2, 4])

    def test_resync_puts_the_sound_back_under_the_frame_it_came_from(self):
        ok, s = self.r["pairResync"]
        self.assertTrue(ok)
        self.assertEqual(s["vid"], [6, 10])
        self.assertEqual(s["snd"], [6, 10])
        self.assertEqual(s["drift"], 0)
        # ...and it stays UNLINKED. Rematching is not re-linking.
        self.assertFalse(s["linked"])

    def test_resync_is_a_rematch_and_not_an_un_trim(self):
        # The strip's in-point is one second into the take, so it lands one
        # second after the picture does — that is what "in sync" means for a
        # sound somebody deliberately shortened.
        ok, s = self.r["pairResyncKeepsTheTrim"]
        self.assertTrue(ok)
        self.assertEqual(s["vid"], [5, 9])
        self.assertEqual(s["ssrc"], [1, 4])         # the trim survives
        self.assertEqual(s["snd"], [6, 9])
        self.assertEqual(s["drift"], 0)

    def test_resync_refuses_what_it_has_nothing_to_do_to(self):
        ok, why = self.r["pairResyncRefusesWhenAlreadyThere"]
        self.assertFalse(ok)
        self.assertIn("already in sync", why)
        ok, why = self.r["pairResyncRefusesALinkedClip"]
        self.assertFalse(ok)
        self.assertIn("already under its own picture", why)

    def test_relink_freezes_the_offset_instead_of_throwing_it_away(self):
        # It used to DELETE the field, which snapped the sound back under the
        # picture — so the one button that said "link" destroyed the J-cut the
        # moment it was made. That is why he reached for LOCK instead, and why
        # the clip then refused every drag with a forbidden cursor.
        s, flagged = self.r["pairRelinkFreezesTheOffset"]
        self.assertTrue(flagged)                    # audio.linked === true
        self.assertEqual(s["vid"], [6, 10])
        self.assertEqual(s["snd"], [4, 8])          # exactly where he put it
        self.assertEqual(s["drift"], -2)
        self.assertTrue(s["coupled"])
        self.assertTrue(s["linked"])                # ...and inert to a drag

    def test_an_in_sync_relink_still_leaves_no_trace(self):
        # The migration promise: a split somebody tried and undid produces a
        # document identical to one that never had the field.
        gone, s = self.r["pairRelinkInSyncRemovesTheField"]
        self.assertTrue(gone)
        self.assertFalse(s["split"])
        self.assertEqual(s["snd"], s["vid"])

    def test_a_coupled_pair_travels_together(self):
        # "You just drag it, and the sound below stays, and then you can lock
        # it and move it, and then the sound starts before the clip starts."
        s, drift, trim_ok = self.r["pairCoupledTravels"]
        self.assertEqual(s["vid"], [10, 14])
        self.assertEqual(s["snd"], [8, 12])         # carried, offset intact
        self.assertEqual(drift, -2)
        self.assertTrue(trim_ok)                    # and still trimmable

    def test_a_coupled_strip_cannot_be_dragged_on_its_own(self):
        ok, why = self.r["pairCoupledStripRefusesTheDrag"]
        self.assertFalse(ok)
        self.assertIn("travels with its picture", why)

    def test_the_toggle_round_trip_keeps_both_halves_editable(self):
        # What he actually did: unlink, trim, "lock", move, unlink again. Every
        # step is accepted and the offset he built survives all of them.
        tail, made, frozen, moved, head, freed, again = \
            self.r["pairToggleRoundTrip"]
        self.assertEqual(tail, [True, 0])           # tail trim while unlinked
        self.assertEqual(made, -1.5)                # the J-cut he dragged
        self.assertEqual(frozen, [-1.5, True])      # re-link keeps it
        self.assertEqual(moved, [-1.5, [7.5, 11.5]])   # the pair moves as one
        self.assertEqual(head, [True, -1.5])        # head trim, still coupled
        self.assertEqual(freed, [-1.5, False, False])  # unlink gives it back
        self.assertEqual(again, [True, -1.5])       # and it is draggable again

    def test_the_scenario_he_described_start_to_finish(self):
        # "I clip a little from the visuals and then move all together to fill
        # the gap. So you start hearing the character before you see it."
        q = self.r["pairHisScenario"]
        # A head trim leaves the strip reaching back under the shot before it —
        # and its DRIFT is zero, because the same source second still plays at
        # the same film second. That is the J-cut, and it is IN SYNC.
        self.assertEqual(q["trimOpensTheGap"], [[5, 8], [4, 8], 0])
        # ...so re-linking may not drop the field on the strength of the drift
        # alone: doing that deleted the second of sound the trim had just
        # exposed. Only a strip that IS the picture's own window can go.
        self.assertEqual(q["frozen"], [True, 0])
        # Then the pair moves as one unit to close the gap.
        self.assertEqual(q["videoDelta"], q["soundDelta"])
        self.assertEqual(q["videoDelta"], -1)
        self.assertEqual(q["moved"], [[4, 7], [3, 7], 0])
        self.assertEqual(q["soundLeadsBy"], 1)   # heard before he is seen

    def test_a_coupled_pair_is_never_flagged_as_out_of_sync(self):
        free, coupled, drift = self.r["pairCoupledIsNotFlagged"]
        self.assertTrue(free)        # a free strip that drifted carries a flag
        self.assertFalse(coupled)    # a frozen relationship does not
        self.assertEqual(drift, -2)  # ...even though the offset is real

    def test_resync_reaches_a_couple_and_leaves_nothing_behind(self):
        ok, gone, s = self.r["pairResyncACouple"]
        self.assertTrue(ok)
        self.assertTrue(gone)        # a rematched couple has nothing to say
        self.assertEqual(s["snd"], s["vid"])
        self.assertEqual(s["drift"], 0)

    def test_a_coupled_pair_splits_into_two_coupled_pairs(self):
        ok, left, right = self.r["pairSplitACouple"]
        self.assertTrue(ok)
        self.assertEqual([left["vid"], left["snd"]], [[6, 8], [4, 6]])
        self.assertEqual([right["vid"], right["snd"]], [[8, 10], [6, 8]])
        self.assertEqual([left["drift"], right["drift"]], [-2, -2])
        self.assertTrue(left["coupled"] and right["coupled"])

    def test_a_head_trim_can_never_slip_the_picture_inside_its_own_slot(self):
        # Two seconds of head asked for, one second of room: the handle takes
        # the second it has and the mapping of the frames it keeps is unchanged,
        # so the sound underneath is still describing the right frame. Before
        # the clamp the in-point moved the full two seconds while the slot stood
        # still — the tail grew the wrong way and the pair went -1.00s out.
        ok, s = self.r["pairHeadTrimCannotSlip"]
        self.assertTrue(ok)
        self.assertEqual(s["vid"], [2, 6])          # the tail does NOT move
        self.assertEqual(s["vsrc"], [1, 5])
        self.assertEqual(s["drift"], 0)
        got = self.r["pairHeadTrimUsesTheRoomItHas"]
        self.assertEqual(got["vid"], [2.5, 6])
        self.assertEqual(got["vsrc"], [1.5, 5])
        self.assertEqual(got["drift"], 0)
        # Hard against a neighbour there is no room at all, so it refuses.
        ok, why, s = self.r["pairHeadTrimAgainstANeighbour"]
        self.assertFalse(ok)
        self.assertEqual(why, "edge")
        self.assertEqual(s["vid"], [4, 8])

    def test_the_flag_prints_a_signed_number_and_late_is_positive(self):
        early, late, early_label, late_label = self.r["pairDriftSigns"]
        self.assertEqual([early, late], [-2, 2])
        self.assertEqual([early_label, late_label], ["-2.00s", "+2.00s"])

    def test_in_sync_has_a_tolerance_of_half_a_frame(self):
        exact, near, past, linked = self.r["pairInSyncTolerance"]
        self.assertTrue(exact)
        self.assertTrue(near)       # 1/96 s — inside half a frame at 24 fps
        self.assertFalse(past)      # 1/12 s — two frames, and it says so
        self.assertTrue(linked)     # a linked clip cannot drift

    def test_both_halves_of_a_drifted_pair_carry_the_flag(self):
        # The blocks sit in different lanes, so "are these still together" is
        # not a question the eye can answer at fit zoom.
        for fn in ("sbePaintTrack", "sbePaintAudioLane"):
            self.assertIn("sbeSyncBadge(c)", extract_function(fn, self.src))
        badge = extract_function("sbeSyncBadge", self.src)
        self.assertIn("sbeAudioInSync(c)", badge)
        self.assertIn("sbeDriftLabel", badge)
        self.assertIn('data-sync="', badge)

    def test_the_flag_is_a_button_and_neither_drag_handler_swallows_it(self):
        for fn in ("sbeOnTrackDown", "sbeOnAudioDown"):
            body = extract_function(fn, self.src)
            self.assertIn(".sbe-sync", body)
            self.assertIn("sbeResyncSel(badge.dataset.sync)", body)
            # ...and it happens BEFORE the drag is armed, or pointerdown wins.
            self.assertLess(body.index("sbe-sync"), body.index("setPointerCapture"))

    def test_the_inspector_offers_the_rematch_next_to_the_relink(self):
        fn = extract_function("sbePaintInspector", self.src)
        self.assertIn("sbeResyncSel()", fn)
        self.assertIn("Resync sound", fn)
        self.assertIn("sbeAudioInSync(c)", fn)
        # The two verbs are different and the inspector says which is which.
        self.assertLess(fn.index("Re-link sound"), fn.index("Resync sound"))

    def test_a_locked_shot_says_so_instead_of_a_forbidden_cursor(self):
        # `.sbe-clip.is-locked` sets `cursor: not-allowed` and hides both
        # grips, and the drag handler refused in silence — so a locked shot
        # read as an editor that had broken. He hit it precisely because Lock
        # was the only button that sounded like "keep these two together".
        self.assertIn(".sbe-clip.is-locked { cursor: not-allowed; }", self.src)
        body = extract_function("sbeOnTrackDown", self.src)
        self.assertIn("c.locked", body)
        self.assertIn("click Unlock in the", body)
        # ...and the refusal happens BEFORE the drag is armed.
        self.assertLess(body.index("c.locked"), body.index("setPointerCapture"))
        lane = extract_function("sbeOnAudioDown", self.src)
        self.assertIn("c.locked", lane)

    def test_the_right_grip_is_on_the_RIGHT(self):
        # `.sbe-grip { left: 0 }` was written after `.sbe-grip.r { right: 0 }`,
        # and a box with left, right and width all set is over-constrained —
        # LTR drops `right`. Both grips rendered on top of each other at the
        # LEFT edge of every clip: measured live, clip 315–455, `.l` 316–325,
        # `.r` 316–325. So the right edge had no handle (pulling the tail MOVED
        # the clip) and the left edge hit `.r`, later in DOM order, so reaching
        # for the head trimmed the TAIL. Both lanes, every clip.
        src = self.src
        widen = src.index(".sbe-grip { left: 0; }")
        fix = src.index(".sbe-grip.r { left: auto; right: 0; }")
        self.assertLess(widen, fix, "the reset must come AFTER the widening")
        # ...and the pseudo-element skirt still covers the border column.
        #
        # THE 3 IS A VARIABLE NOW, and this assertion had to follow it there.
        # The fade handle's inset is measured from the far side of this skirt —
        # `grip-w + skirt` is the first x on a block that is not a trim — so
        # the two numbers are one number, declared on `.sbe-tl` and read by
        # both. This used to assert the whole rule as one literal line, which
        # is a spelling and not a behaviour: what it is really protecting is
        # that the skirt is still NEGATIVE on the horizontal axis and zero on
        # the vertical, so the grip can be hit past what it draws without
        # growing over the block above or below it. The effect is measured for
        # real in `scripts/measure_editor_layout.py`, which reconstructs the
        # grip's true hit rectangle from this custom property and fails if
        # anything else has moved into it.
        skirt = src[src.index(".sbe-grip::before {"):]
        skirt = skirt[:skirt.index("}") + 1]
        self.assertIn("position: absolute", skirt)
        self.assertIn("inset: 0 calc(-1 * var(--sbe-grip-skirt", skirt)
        self.assertIn("--sbe-grip-skirt: 3px;", src)
        self.assertIn("--sbe-grip-w: 9px;", src)

    def test_the_corner_handle_stands_beside_the_grip_and_not_inside_it(self):
        """"Not sure that grabbing the top corner is working."

        Measured on his screen: `.sbe-fade-h` was 12x12 at `top:0; left:0` with
        `z-index: 7`, and `.sbe-grip` was 9x54 at the SAME corner with
        `z-index: 6` and a 3px hit skirt. On a 56px block the fade owned the
        top 22% of that corner and the trim owned the other 78%, so aiming at
        the corner hit trim almost every time. The handler was already correct
        — the fade is tested before the grips — and no handler can rescue a
        target that is not there.

        The two are separated in SPACE now: the grip keeps the outer strip for
        the block's full height because trimming is an edge gesture, and the
        handle sits inset from it, starting exactly where the grip's hit area
        stops. The inset is DERIVED from the grip's own two variables so the
        two can never drift; the rectangles are then measured for real, at both
        ends of the lane range, by `scripts/measure_editor_layout.py`.
        """
        src = self.src
        band = src[src.index(".sbe-fade-band {"):]
        band = band[:band.index("}")]
        # Inset from BOTH grips, by the grip's own width plus its own skirt.
        for side in ("left", "right"):
            self.assertIn(f"{side}: calc(var(--sbe-grip-w, 9px) + "
                          f"var(--sbe-grip-skirt, 3px))", band)
        # The two handles share a band too narrow for both rather than stack on
        # it, which is the other overlap the old absolute boxes could produce.
        self.assertIn("justify-content: space-between", band)
        self.assertIn("pointer-events: none", band)

        handle = src[src.index("    .sbe-fade-h {"):]
        handle = handle[:handle.index("}")]
        self.assertIn("flex: 0 1 var(--sbe-fade-hit, 22px)", handle)
        self.assertIn("pointer-events: auto", handle)
        # A REAL TARGET, sized against the SHORTEST lane the share table can
        # produce rather than against a lane the user can drag. 12px was a
        # fraction of a picture track that later lost a quarter of its height
        # in the calm-chrome rebalance, and nobody touched the handle.
        hit = int(re.search(r"--sbe-fade-hit:\s*(\d+)px", src).group(1))
        self.assertGreaterEqual(hit, 20)
        # THE SHORTEST LANE THE SHARE TABLE CAN PRODUCE, read back out of the
        # table rather than written down here — the whole point is that the
        # handle must not be sized against a number somebody can move. The
        # block inside a lane is inset 6px top and bottom; the strip 3px.
        lanes = dict(re.findall(
            r"\{ key: '(\w+)',\s+base:\s*(\d+), cap:\s*\d+, share: [\d.]+ \}",
            src))
        for key, inset in (("track", 12), ("alane", 7)):
            block = int(lanes[key]) - inset
            self.assertLess(
                hit, block,
                f"a {hit}px handle does not fit the {block}px block a "
                f"{lanes[key]}px {key} lane leaves at its floor")

        # ...and both lanes emit it, wrapped, so neither can drift from the
        # other: a strip that behaved differently from the block above it would
        # be a difference no user could name.
        track = extract_function("sbePaintTrack", src)
        lane = extract_function("sbePaintAudioLane", src)
        for fn, attr in ((track, "data-fade"), (lane, "data-afade")):
            self.assertIn('sbe-fade-band', fn)
            self.assertLess(fn.index("sbe-fade-band"), fn.index(attr + '="in"'))

    def test_the_level_line_declines_while_the_pointer_is_on_a_handle(self):
        # The clipped hit path keeps the LINE off the handles' pixels, but the
        # same pair also answers a proximity test that knows nothing about
        # paths — and the hover ghost asks it. Without this a fade handle sat
        # under a ghost promising a keyframe it could never place. Asked of the
        # real hit target rather than of a rectangle, so it stays exact at
        # every lane height and every zoom.
        fn = extract_function("sbeStripAt", self.src)
        self.assertIn("ev.target.closest('.sbe-fade-h')", fn)
        self.assertLess(fn.index(".sbe-fade-h"), fn.index("sbeStripEditable"))

    def test_the_sync_flag_stays_out_of_both_gestures_in_that_corner(self):
        # It was already clear of the grips. The fade handle then moved into
        # the space it was standing in, and the handle is higher in the stack —
        # so the flag would have had its clicks swallowed by a control that was
        # not there when its position was chosen. Derived from the same three
        # variables, so it cannot be left behind a second time.
        # The rule that PLACES it, which is the last `.sbe-sync {` that sets a
        # `right` — the ones after it only nudge `top` per block kind.
        css = [b for b in re.findall(r"\.sbe-sync \{(.*?)\}", self.src, re.S)
               if "right:" in b]
        self.assertTrue(css, "no .sbe-sync rule sets a `right`")
        css = css[-1]
        self.assertIn("var(--sbe-grip-w", css)
        self.assertIn("var(--sbe-grip-skirt", css)
        self.assertIn("var(--sbe-fade-hit", css)
        # ...and both pointerdown handlers still test it first, which is what
        # makes the z-index above the band safe rather than a second bug.
        for fn in ("sbeOnTrackDown", "sbeOnAudioDown"):
            body = extract_function(fn, self.src)
            self.assertLess(body.index(".sbe-sync"), body.index(".sbe-fade-h"))

    def test_a_drag_that_moved_only_the_SOUND_is_still_an_edit(self):
        # `sbeOnTrackUp` decides "did anything change" from a fingerprint of
        # four picture fields, and RESTORES the pointerdown snapshot when they
        # match. A gesture whose only effect was on a strip — a ripple carrying
        # an unlinked sound, a coupled pair travelling with its picture — was
        # therefore discarded: the document was never marked dirty and the
        # header went on saying "saved · revision N" over the work it had just
        # thrown away.
        # THE FINGERPRINT IS A NAMED FUNCTION NOW, so this is driven rather
        # than read: the closure it replaces was rewritten once per gesture and
        # was short each time. `sbeDragFingerprint` is exercised in the node
        # harness against every mode that writes a field — see
        # `TheSoundsEnvelopeOnTheClient.test_every_mode_a_track_drag_has`.
        body = extract_function("sbeOnTrackUp", self.src)
        self.assertIn("sbeDragFingerprint(before)", body)
        self.assertLess(body.index("sbeDragFingerprint(before)"),
                        body.index("SBE.dirty = true"))
        fp = extract_function("sbeDragFingerprint", self.src)
        self.assertIn("JSON.stringify(c.audio || null)", fp)

    def test_the_locked_toast_names_the_verb_he_actually_wanted(self):
        body = extract_function("sbeOnTrackDown", self.src)
        self.assertIn("locked to its place on the __SEQ__", body)
        self.assertIn("Link sound", body)

    def test_neither_lane_can_be_wedged_by_the_other_s_lost_pointerup(self):
        # A pointerup that lands off the edge leaves a drag object behind, and
        # a stale one makes the next gesture answer to the wrong lane.
        self.assertIn("SBE.audioDrag = null;", extract_function("sbeOnTrackDown", self.src))
        self.assertIn("SBE.drag = null;", extract_function("sbeOnAudioDown", self.src))

    def test_the_toggle_names_the_offset_it_is_about_to_freeze(self):
        fn = extract_function("sbePaintInspector", self.src)
        self.assertIn("'Link sound'", fn)
        self.assertIn("Re-link sound", fn)
        self.assertIn("sbeAudioInSync(c)", fn)

    def test_every_reflow_marks_the_sound_before_it_moves_the_picture(self):
        # The carry is the fix, and a new operation that calls sbeLayout without
        # it is the same bug again. Each of these reflows the track.
        for fn in ("sbeMoveTo", "sbeTrim", "sbeRippleDelete", "sbeSplitAt",
                   "sbeInsertAt", "sbeReorderTo", "sbePlaceUnplaced"):
            body = extract_function(fn, self.src)
            self.assertIn("sbeSyncMark(", body, fn)
            self.assertIn("sbeSyncCarry(", body, fn)
            self.assertLess(body.index("sbeSyncMark("), body.index("sbeSyncCarry("), fn)


class TimelineVersions(unittest.TestCase):
    """Wave 4 — what the picker says about a file you would restore over your
    own work. Wording is the entire interface here, so it is a gate."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.r = run_contract()
        cls.src = panel_source()

    def test_the_age_reads_as_a_duration_in_one_unit(self):
        self.assertEqual(self.r["ago"],
                         ["just now", "5 min ago", "2 h ago", "5 d ago", ""])

    def test_an_autosave_says_so_and_names_what_it_holds(self):
        line = self.r["lineAuto"]
        self.assertEqual(line["name"], "autosave")
        self.assertEqual(line["meta"], "revision 7 · 9 clips · 0:39.71 · 5 min ago")
        self.assertFalse(line["kept"])

    def test_an_unnamed_save_is_not_called_an_autosave(self):
        # The two-lane split exists to separate the user's decisions from the
        # machine's, and then labelled his decisions with the machine's word:
        # the lane headed YOUR SAVES OF THIS DRAFT listed rows called
        # "autosave".
        self.assertEqual(self.r["lineMine"]["name"], "Save")

    def test_the_verb_that_names_a_save_is_reachable_again(self):
        # The route is alive and documented; the drafts rewrite left the panel
        # with no way to press it, while the history went on rendering named
        # saves in their own accent — so the owner's film contains a kind of
        # entry he had no way to make again.
        self.assertIn('id="sbeKeepSaveBtn"', self.src)
        fn = extract_function("sbeKeepVersion", self.src)
        self.assertIn("'/storyboard/edit/version'", fn)
        self.assertIn("fd.set('label', label)", fn)
        self.assertIn("sbeVersionsLoad()", fn)

    def test_the_name_field_knows_which_verb_it_is(self):
        # Enter was hard-wired to "new draft", so pressing Rename on a row,
        # typing, and pressing Enter created a duplicate draft carrying the
        # half-edited name. The old order also focused the box BEFORE writing
        # the old name into it, so the caret sat behind the text and typing
        # appended to it.
        self.assertIn("sbeNameEnter()", self.src)
        enter = extract_function("sbeNameEnter", self.src)
        self.assertIn("SBE.renaming", enter)
        ren = extract_function("sbeDraftRename", self.src)
        self.assertIn("sbeNameMode(slug)", ren)
        self.assertIn("box.value = was || '';", ren)
        self.assertLess(ren.index("box.value = was"), ren.index("box.focus()"))
        self.assertIn("box.select()", ren)

    def test_the_drafts_list_cannot_be_squeezed_out_by_the_snapshots(self):
        # Both lists carried `flex: 1 1 auto; min-height: 0` inside one
        # 520px box, so expanding fifty automatic snapshots collapsed "This
        # film's drafts" — the headline of the whole feature — to an
        # eleven-pixel sliver, with the row for the draft you have OPEN the
        # one cut in half.
        css = self.src[self.src.index(".sbe-vers-list {"):]
        css = css[:css.index(".sbe-vers-row {")]
        self.assertIn("#sbeDraftList { flex: 0 0 auto;", css)
        self.assertIn("#sbeVersList { flex: 1 1 auto;", css)

    def test_a_named_version_shows_its_name(self):
        line = self.r["lineKept"]
        self.assertEqual(line["name"], "the good one")
        self.assertIn("1 clip ", line["meta"] + " ")
        self.assertTrue(line["kept"])

    def test_an_unreadable_entry_says_that_rather_than_disappearing(self):
        line = self.r["lineBad"]
        self.assertEqual(line["name"], "unreadable")
        self.assertTrue(line["bad"])

    # ---- the markup ------------------------------------------------------
    def test_the_save_button_says_it_saved(self):
        self.assertIn('id="sbeSaveBtn" onclick="sbeSaveNow()"', self.src)
        fn = extract_function("sbeSaveNow", self.src)
        self.assertIn("Saved — revision", fn)
        # ...and the autosave stays silent: it fires a second after every drag.
        self.assertNotIn("phosToast", extract_function("sbeQueueSave", self.src))

    def test_no_draft_verb_reaches_for_a_browser_dialog(self):
        # The rule is "names in this app are typed into panel controls, never
        # into a browser dialog", and it was enforced for `window.prompt`
        # only — so `sbeDraftDelete` gated the one DESTRUCTIVE verb in this
        # feature on a `window.confirm`, the single piece of Chrome chrome on
        # a claude.ai-grade surface. Delete is a two-step in the panel now.
        self.assertIn('id="sbeVersName"', self.src)
        for name in ("sbeDraftNew", "sbeDraftRename", "sbeDraftDelete",
                     "sbeKeepVersion"):
            fn = extract_function(name, self.src)
            for bad in ("window.prompt", "window.confirm", "window.alert"):
                self.assertNotIn(bad, fn, f"{name} / {bad}")
        for name in ("sbeDraftNew", "sbeDraftRename"):
            self.assertIn("sbeEl('sbeVersName')",
                          extract_function(name, self.src))
        arm = extract_function("sbeDraftDelete", self.src)
        self.assertIn("'Delete?'", arm)
        self.assertIn("btn.dataset.armed", arm)
        self.assertIn("'/storyboard/edit/draft'",
                      extract_function("sbeDraftOp", self.src))

    def test_only_the_user_writes_his_draft(self):
        # "He should have control over the saving, and only he should have
        # that control." The debounce every mutation calls now writes a crash
        # BACKUP; the Save button is the only thing that writes the document.
        q = extract_function("sbeQueueSave", self.src)
        self.assertIn("sbeBackup()", q)
        self.assertNotIn("sbeSave(", q)
        b = extract_function("sbeBackup", self.src)
        self.assertIn("'/storyboard/edit/backup'", b)
        self.assertNotIn("sbeAdopt", b)          # it changes nothing on screen
        self.assertIn("expect: null", b)         # it cannot conflict with anything
        # ...and leaving the tab backs up rather than saving over his draft.
        close = extract_function("sbeCloseDoc", self.src)
        self.assertIn("sbeBackup(true)", close)
        self.assertNotIn("sbeSave(true)", close)

    def test_opening_a_film_is_not_an_edit_to_it(self):
        # sbeAdopt sets the mode dropdown to agree with the document on every
        # load, and that call was marking the document dirty and queueing a
        # write — so every OPEN of every film produced a save nobody made.
        fn = extract_function("sbeSetMusicMode", self.src)
        self.assertIn("if (String(SBE.audio.mode || 'under') === mode) return;", fn)
        self.assertLess(fn.index("=== mode) return"), fn.index("SBE.dirty = true"))

    def test_the_lane_has_nothing_left_to_guard(self):
        # The guard is gone because the thing it guarded is gone: snapshots
        # are per-file now, so writing one cannot destroy another. What must
        # NOT come back is any early return keyed on an unanswered offer.
        fn = extract_function("sbeBackup", self.src)
        self.assertNotIn("if (SBE.backup) return false;", fn)
        self.assertIn("versioned now", fn)
        self.assertNotIn("return;", fn)   # it still answers its callers

    def test_a_backup_that_is_not_being_written_still_screams(self):
        # It is the safety net, and its absence is invisible by nature.
        self.assertIn("sbeSaveAlarm", extract_function("sbeBackup", self.src))

    def test_the_switch_is_REFUSED_when_the_backup_could_not_write(self):
        # Driven, not grepped: with an unanswered offer on screen the backup
        # cannot write, so the draft op must refuse before anything leaves the
        # panel and must not clear `dirty` on the way out.
        got, fetches, dirty, said = \
            self.r["draftSwitchRefusesWhenTheBackupCannotWrite"]
        self.assertIs(got, False)
        # The one fetch is the snapshot POST that failed; the draft op itself
        # never left the panel.
        self.assertEqual(fetches, 1)
        self.assertTrue(dirty)
        self.assertIn("unsaved changes", said)

    def test_nothing_swaps_the_document_on_a_backup_that_did_not_happen(self):
        # `await sbeBackup(true)` with the result thrown away: the switch went
        # ahead, cleared `dirty` and adopted the server's document — while
        # `activate_draft` stashes the last SAVED file, so the work on screen
        # went with no offer, no toast and nothing to click. The comment above
        # the call states the exact invariant the missing branch broke.
        fn = extract_function("sbeDraftOp", self.src)
        self.assertIn("!(await sbeBackup(true))", fn)
        self.assertIn("could not be snapshotted", fn)
        self.assertIn("return false;", fn)

    def test_nothing_acts_on_the_saved_file_after_a_save_that_failed(self):
        # Restore, relink, render and the NLE export all read what is on
        # DISK. A save that 409s or errors leaves the server holding an older
        # cut, and every one of these would then act on that one — the render
        # most visibly, by building a film that is quietly not the one on
        # screen.
        for name in ("sbeRestoreVersion", "sbeRelink", "sbeRenderFilm",
                     "sbeExportNle"):
            fn = extract_function(name, self.src)
            self.assertIn("!(await sbeSave(true))", fn, name)
            self.assertIn("kind: 'danger'", fn, name)

    def test_the_recovery_is_an_offer_and_never_an_action(self):
        # Silently applying a newer backup would be the autosave the owner
        # asked us to remove, wearing a different name.
        el = self.src[self.src.index('id="sbeRecover"'):]
        el = el[:el.index("</div>")]
        self.assertIn("Unsaved snapshot", el)
        self.assertIn("Restore it", el)
        self.assertIn("Discard", el)
        paint = extract_function("sbePaintRecovery", self.src)
        # It says WHAT IT IS, not what to do about it: the old sentence
        # compared clip counts out loud and then said "Nothing has been
        # changed", which read as a question about a difference it could not
        # name. See docs/EDITOR_SAVE_MODEL.md §4.
        self.assertIn("your saved draft is untouched", paint)
        self.assertNotIn("Nothing has been changed", paint)
        # The offer is painted on adopt; nothing calls sbeRecover for you.
        adopt = extract_function("sbeAdopt", self.src)
        self.assertIn("sbePaintRecovery()", adopt)
        self.assertNotIn("sbeRecover()", adopt)

    def test_the_users_saves_come_before_the_machines(self):
        # "The auto saves should be saved separately from the manual saves, at
        # least, so the user can go back and see the manual saves."
        fn = extract_function("sbeVersionsPaint", self.src)
        self.assertIn("Your saves of this draft", fn)
        self.assertIn("v.manual !== false", fn)
        self.assertIn("automatic snapshot", fn)
        self.assertIn("<details", fn)
        # The user's lane is drawn first, the machine's is folded under it.
        self.assertLess(fn.index("Your saves of this draft"),
                        fn.index("automatic snapshot"))

    def test_the_draft_you_are_cutting_names_itself(self):
        fn = extract_function("sbePaintDraft", self.src)
        self.assertIn("sbeDraftName", fn)
        self.assertIn("dot.hidden = !SBE.dirty", fn)
        # The chip and the dot cannot disagree — one drives the other.
        self.assertIn("sbePaintDraft()", extract_function("sbeSetState", self.src))

    def test_every_draft_verb_goes_through_one_door(self):
        fn = extract_function("sbeDraftOp", self.src)
        for op in ("new", "duplicate", "rename", "delete", "activate"):
            self.assertIn(op, self.src)
        # Switching drafts backs the current one up first, or the last few
        # minutes would be left pointing at a draft nobody is looking at.
        self.assertIn("await sbeBackup(true)", fn)
        self.assertIn("sbeAdopt(r, true)", fn)

    def test_the_panel_is_portalled_to_the_body(self):
        # The header and the stage column both carry overflow; a panel left
        # where it was declared is sliced at their edge.
        fn = extract_function("sbeVersionsEl", self.src)
        self.assertIn("document.body.appendChild", fn)

    def test_restore_adopts_the_server_payload_and_drops_the_undo_stack(self):
        fn = extract_function("sbeRestoreVersion", self.src)
        self.assertIn("'/storyboard/edit/restore'", fn)
        self.assertIn("SBE.undo.length = 0", fn)
        self.assertIn("sbeAdopt(r, true)", fn)
        # It says what happened to the arrangement it replaced.
        self.assertIn("was kept", fn)


class TheTimelineIsResizable(unittest.TestCase):
    """The top edge is a handle, and the height it drags lands on the SOUND.

    "The timeline is too constricted and cannot be expanded vertically. It
    needs to allow you to drag the upper side of the timeline, which will
    change the layout a little bit, enabling expansion in case you have some
    sound editing in there."

    Nothing here resizes an element: the drag moves one number and
    sbeFitMonitors sizes the monitors off what is left. So the tests are about
    that number — what it may be, where it goes, and where it is kept.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.r = run_contract()
        cls.src = panel_source()

    # ---- the clamps ------------------------------------------------------
    def test_the_timeline_can_never_collapse_or_eat_the_monitors(self):
        c = self.r["tlClamp"]
        self.assertEqual(c["under"], 280)      # its own contents are the floor
        self.assertEqual(c["over"], 600)       # what the window measured
        self.assertEqual(c["exact"], 400)
        # A window with nothing to spare pins the handle rather than handing
        # the monitors a negative height.
        self.assertEqual(c["noRoom"], 280)
        # ...and no measurement can authorise more than the lanes can use.
        self.assertEqual(c["ceiling"], 638)
        self.assertEqual(c["junk"], 280)

    def test_the_ceiling_leaves_the_monitors_a_picture(self):
        # The one clamp that keeps this honest lives in sbeFitMonitors: the
        # timeline's ceiling is the column minus the smallest thing that is
        # still a monitor, which is the floor sbeMonitorFit has always used.
        fn = extract_function("sbeFitMonitors", self.src)
        self.assertIn("SBE_MON_MIN_H", fn)
        self.assertIn("SBE.tlMax", fn)
        self.assertIn("sbeTlClamp(SBE.tlH", fn)
        # The preference is READ here and never written: shrink the window and
        # the height is given back, widen it and the drag is still there.
        self.assertNotIn("SBE.tlH =", fn)

    # ---- where the height goes ------------------------------------------
    def test_at_the_floor_the_sound_is_already_legible(self):
        # The bases the 2026-08-20 rebalance settled on. The clip strip starts
        # at 44 rather than 26 because a level line has to be a control AT
        # REST — "I still don't understand how to add keyframes" was said
        # about a 20px band — and the picture drops to 64 because its block is
        # a label with a thumbnail hint, not a poster.
        # ...and the soundtrack starts at 108 rather than 72 for the same
        # reason one lane up: the bed carries a level line, points and two
        # corner fade handles now, and its HEAD carries the mix — the level
        # and the duck the renderer used to keep to itself.
        self.assertEqual(self.r["lanesAtFloor"],
                         {"ruler": 18, "ov": 32, "track": 64,
                          "alane": 44, "wave": 108})

    def test_every_dragged_pixel_is_spent_on_a_lane(self):
        # Chrome + the four lanes == the height that was asked for, at both
        # ends and in the middle. A distribution that loses pixels leaves a
        # dead band under the soundtrack.
        self.assertEqual(self.r["laneSums"], [0, 0, 0])

    def test_the_sound_takes_the_biggest_share_because_that_was_the_ask(self):
        g = self.r["laneGain"]
        self.assertEqual(sum(g.values()), 100)
        # 70 of every 100px goes to the two sound lanes...
        self.assertGreaterEqual(g["alane"] + g["wave"], 70)
        # ...and the per-clip strip — the one with the level line and the
        # points on it — takes the single biggest share.
        self.assertEqual(max(g, key=g.get), "alane")
        # A 26px strip becomes one you can actually put a keyframe in.
        self.assertGreaterEqual(self.r["lanesMid"]["alane"], 60)

    def test_a_capped_lane_hands_its_share_back_rather_than_dropping_it(self):
        # 340px of drag fills the two lanes with the tightest caps — the
        # overlay strip and the per-clip sound — and what they cannot take is
        # offered AGAIN to the lanes still growing rather than left as dead
        # band at the bottom of the box.
        past = self.r["lanesPastTheTrackCap"]
        self.assertEqual(past["ov"], 56)
        self.assertEqual(past["alane"], 190)
        self.assertEqual(self.r["laneRedistributed"], 0)
        # Both survivors have MORE than their bare share of 340px (47.6 and
        # 115.6) — the difference is what the two capped lanes handed back.
        self.assertGreater(past["track"], 64 + 48)
        self.assertGreater(past["wave"], 108 + 116)

    def test_the_ruler_never_grows(self):
        # It is a scale, not a surface: a taller ruler is not a more legible
        # one, and every pixel it took would be a pixel off a waveform.
        self.assertEqual(self.r["rulerFixed"], [18, 18])

    def test_at_the_ceiling_every_lane_is_at_its_cap(self):
        self.assertEqual(self.r["lanesAtCeiling"],
                         {"ruler": 18, "ov": 56, "track": 120,
                          "alane": 190, "wave": 240})

    # ---- where the preference lives -------------------------------------
    def test_the_height_survives_a_reload_and_is_clamped_on_the_way_back(self):
        p = self.r["tlPref"]
        self.assertEqual(p["fresh"], 280)          # nothing stored → the floor
        self.assertEqual(p["stored"], "392")
        self.assertEqual(p["restored"], 392)       # what the next tab reads
        self.assertEqual(p["absurd"], 638)         # a stored screenful is capped
        self.assertEqual(p["junk"], 280)           # ...and junk is the floor
        self.assertEqual(p["keys"], ["phos_sbe_tl_h"])

    def test_the_height_is_a_view_preference_and_NOT_sequence_data(self):
        # It is not in the save payload, so a drag cannot bump the document's
        # revision, race the snapshot lane, or hand the next machine somebody
        # else's screen. localStorage is where every other view preference in
        # this panel already lives.
        self.assertEqual(self.r["tlIsNotInTheSave"], [-1, -1, -1])
        self.assertIn("localStorage.getItem('phos_sbe_tl_h')", self.src)
        self.assertIn("localStorage.setItem('phos_sbe_tl_h'", self.src)
        # The server's document model has never heard of it.
        model = (Path(__file__).resolve().parent
                 / "storyboard_editor.py").read_text(encoding="utf-8")
        self.assertNotIn("tl_h", model)

    # ---- the handle itself ----------------------------------------------
    def test_the_handle_is_on_the_timelines_top_edge_and_costs_no_height(self):
        el = extract_element("sbeTlGrab", self.src)
        self.assertIn("sbeTlGrabDown(event)", el)
        self.assertIn("sbeTlGrabMove(event)", el)
        self.assertIn("sbeTlGrabUp(event)", el)
        self.assertIn("sbeTlReset()", el)          # double-click resets
        self.assertIn('role="separator"', el)
        self.assertIn('tabindex="0"', el)
        # It lives INSIDE the transport, absolutely positioned over the gap
        # below it: sbeFitMonitors budgets against every child of the column
        # it can measure, so a grip with a height of its own would have been
        # paid for out of the picture.
        transport = self.src[self.src.index('<div class="sbe-transport">'):]
        transport = transport[:transport.index("<div class=\"sbe-tl\"")]
        self.assertIn('id="sbeTlGrab"', transport)
        css = self.src[self.src.index(".sbe-tl-grab {"):]
        css = css[:css.index(".sbe-tl-grab:focus-visible {")]
        self.assertIn("position: absolute", css)
        self.assertIn("cursor: row-resize", css)
        self.assertIn("bottom: -8px", css)

    def test_the_drag_captures_the_pointer_and_does_not_select_the_page(self):
        down = extract_function("sbeTlGrabDown", self.src)
        self.assertIn("setPointerCapture", down)
        self.assertIn("sbe-resizing", down)
        # NOT preventDefault: cancelling pointerdown suppresses the
        # compatibility mouse events, and the double-click reset is one.
        self.assertNotIn("ev.preventDefault()", down)
        up = extract_function("sbeTlGrabUp", self.src)
        self.assertIn("releasePointerCapture", up)
        self.assertIn("sbeTlPrefWrite", up)        # the drag is what persists
        self.assertIn("body.sbe-resizing { user-select: none;", self.src)

    def test_up_is_taller(self):
        # clientY falls as the pointer rises, so the delta is SUBTRACTED. Get
        # this backwards and the handle runs away from the pointer.
        move = extract_function("sbeTlGrabMove", self.src)
        self.assertIn("d.h0 - (ev.clientY - d.y0)", move)

    def test_the_handles_arrows_do_not_reach_the_editors_own_shortcuts(self):
        key = extract_function("sbeTlGrabKey", self.src)
        self.assertIn("stopPropagation", key)      # ←→ are a frame at a time
        self.assertIn("ArrowUp", key)
        self.assertIn("ArrowDown", key)

    def test_the_lanes_are_painted_at_the_height_they_were_given(self):
        # The strip's waveform and its level line are drawn into H, and H is
        # the lane's own height — not the 20px the strip used to be. A
        # keyframe's drag reads the strip's RECTANGLE for the same reason, so
        # the picture and the pointer cannot disagree.
        wave = extract_function("sbeStripWave", self.src)
        self.assertIn("const H = sbeStripH()", wave)
        self.assertNotIn("const H = 20", wave)
        # EVERY gesture that reads a level goes through the one pair, so none
        # of them can carry its own copy of a band again.
        move = extract_function("sbeOnAudioMove", self.src)
        self.assertIn("sbeStripGain(ev.clientY, r.top, r.height)", move)
        self.assertNotIn("/ 20)", move)
        dbl = extract_function("sbeOnAudioDbl", self.src)
        self.assertIn("sbeStripGain(ev.clientY, r.top, r.height)", dbl)
        self.assertNotIn("r.top - 3", dbl)
        wave = extract_function("sbeStripWave", self.src)
        self.assertIn("sbeStripY(g, H)", wave)
        # ...and the soundtrack's canvas reads the same distribution.
        cv = extract_function("sbePaintWave", self.src)
        self.assertIn("sbeLaneHeights(", cv)
        self.assertNotIn("const h = 54", cv)

    def test_the_handle_stands_down_where_the_page_is_the_scroller(self):
        # Below the stacking breakpoint sbeFitMonitors removes its variables
        # and returns; a handle that moved a number nobody reads would be a
        # control that does nothing.
        narrow = self.src[self.src.index("@media (max-width: 900px) {"):]
        narrow = narrow[:narrow.index("CAPABILITY TIER")]
        self.assertIn(".sbe-tl-grab { display: none; }", narrow)
        fn = extract_function("sbeFitMonitors", self.src)
        self.assertIn("'--sbe-tl-h'", fn)
        self.assertIn("removeProperty", fn)

    def test_the_resize_does_not_reflow_the_notice_surface(self):
        # THE rule: notices live in one compact surface and never push the
        # workspace around. The handle writes variables onto #sbTimeline and
        # touches nothing else, so there is no path from a drag to a notice.
        for fn in ("sbeTlSet", "sbeApplyTl", "sbeTlGrabDown", "sbeTlGrabUp",
                   "sbeTlReset"):
            body = extract_function(fn, self.src)
            self.assertNotIn("sbeNotice", body)
            self.assertNotIn("phosToast", body)
            self.assertNotIn("sbePaintNotices", body)


class TheLevelLineTeachesItself(unittest.TestCase):
    """"I still don't understand how to add keyframes, to be honest."

    Said after using it, which makes it a defect in the control and not in the
    user. The gesture was double-click to add, drag to set, shift-click to
    remove — three things knowable only by being told. These lock the routes
    that replaced it, and the arithmetic that has to hold at every lane height
    the resize handle can produce.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.r = run_contract()
        cls.src = panel_source()

    # ---- the arithmetic -------------------------------------------------
    def test_the_gain_round_trips_at_every_height_the_table_can_produce(self):
        # THE 20PX BAND MUST NOT COME BACK. It came back twice — once in the
        # drag and once in the double-click — because each gesture carried its
        # own copy. One pair now, and it closes exactly at four strip heights
        # spanning the whole range of the share table.
        self.assertEqual(self.r["lvlRoundTrip"], [])
        self.assertGreaterEqual(len(self.r["stripHeights"]), 4)
        self.assertLess(min(self.r["stripHeights"]), 40)
        self.assertGreater(max(self.r["stripHeights"]), 150)

    def test_the_top_of_a_strip_is_unity_and_the_bottom_is_silence(self):
        for pair in self.r["lvlEnds"]:
            self.assertEqual(pair, [1, 0])

    def test_half_way_down_is_half_at_every_height(self):
        # A hard-coded band gets this right at one height and wrong at every
        # other, which is exactly how the old bug read: correct on the lane it
        # was written for, silence everywhere below fourteen pixels.
        for mid in self.r["lvlMid"]:
            self.assertAlmostEqual(mid, 0.5, delta=0.02)

    def test_unity_sits_clear_of_the_strips_own_top_edge(self):
        """"Maybe you should put the orange line a little lower."

        Unity was drawn at y=1 — ON the strip's top edge. `.sbe-lvl-hit` is a
        14px stroke centred on the line, so six of its seven upper pixels fell
        outside the box and were clipped away by the SVG's own viewport,
        leaving a one-sided sliver to aim at that also competed with the lane's
        border. The headroom has to be at least half the stroke, at EVERY
        height the share table can produce — including the 37px floor, which is
        the one it is hardest to find room in.
        """
        half = 7                                  # half of the 14px target
        for h, y in zip(self.r["stripHeights"], self.r["lvlUnityY"]):
            self.assertGreaterEqual(
                y, half + 1,
                f"a {h}px strip draws unity at y={y}, so {half - y:g}px of its "
                f"own target is outside the box")
            # ...and silence keeps the same headroom at the other end, or the
            # bottom of the range becomes the ungrabbable one instead.
            self.assertGreaterEqual(h - y, half + 1)
        for gap in self.r["lvlSilenceGap"]:
            self.assertGreaterEqual(gap, half + 1)

    # ---- and the span of it that belongs to nothing else -----------------
    def test_the_level_target_stops_where_the_corner_handles_start(self):
        """No pixel belongs to two controls.

        Three affordances share a strip's top corner now — the trim grip, the
        fade handle and the level line — and the line is the one that runs
        under both of the others for its whole length. It is DRAWN end to end,
        because that is the truth about the gain; its TARGET stops clear of
        both handles. The rectangles are measured for real in
        `scripts/measure_editor_layout.py`; this is the arithmetic underneath.
        """
        d = self.r["lvlHitFlat"]
        self.assertTrue(d.startswith("M33.00,"), d)
        # 400 - 33 at the other end, and the y is the line's own y throughout.
        self.assertIn("367.00,", d)
        self.assertNotIn("M0.00", d)

    def test_the_cut_lands_on_the_line_and_not_on_a_chord(self):
        d = self.r["lvlHitSloped"]
        want_start, want_end = self.r["lvlHitSlopedWant"]
        # Gain ramps 1 -> 0 across the strip, so the y at each cut is the y of
        # the gain AT that x — interpolating between whole points instead would
        # put the target above or below the line it is supposed to be on.
        self.assertTrue(d.startswith(f"M33.00,{want_start:.2f}"),
                        f"{d} does not start at the line's own y {want_start}")
        self.assertTrue(d.endswith(f"367.00,{want_end:.2f}"),
                        f"{d} does not end at the line's own y {want_end}")

    def test_a_strip_with_no_room_between_the_handles_offers_no_line(self):
        # 70px of strip is 66 of corner handle. An 8px stub of a control is
        # worse than none: it teaches that the line is grabbable and then is
        # not, and the inspector's Sound section is the route that never
        # depends on how far the timeline is zoomed out.
        self.assertEqual(self.r["lvlHitNarrow"], "")

    def test_a_segment_inside_a_handles_column_is_dropped_not_clamped(self):
        # The first and last segments of this curve live entirely under the
        # corner handles. Clamping them to the cut would run the target back to
        # x=0, which is the overlap this whole change exists to remove.
        d = self.r["lvlHitPoints"]
        self.assertTrue(d.startswith("M33.00,"), d)
        self.assertNotIn("0.00,", d.split("L")[0][1:])
        # ...and what is left is the middle segment, cut at both ends.
        self.assertIn("367.00,", d)

    # ---- and the gesture is not thrown away on the way out ---------------
    def test_every_mode_a_track_drag_has_counts_as_an_edit(self):
        """A fade that snapped back the moment you let go.

        `sbeOnTrackUp` restores the pointerdown snapshot when the film did not
        change — necessary, because dragging a clip that is already hard
        against its neighbour clamps to nowhere and would otherwise burn a
        revision. But the fingerprint it decides on listed four picture fields,
        then five, and `mode: 'fade'` writes `fx` and NOTHING else. So the ramp
        followed the pointer for the whole drag, painted correctly, and was
        discarded on pointerup with the document still saying "saved".

        This drives every mode rather than reading the source, because the
        source has been right-looking and short twice.
        """
        d = self.r["dragFp"]
        self.assertTrue(d["still"],
                        "a drag that moved nothing must still compare equal, "
                        "or every clamped drag burns a revision")
        for mode in ("fade", "fadeOut", "trim", "move", "sound"):
            self.assertTrue(d[mode],
                            f"a drag whose only effect was `{mode}` compares "
                            f"equal to the state it started from, so "
                            f"sbeOnTrackUp restores the snapshot over it")

    def test_one_rule_says_which_strips_can_be_shaped(self):
        e = self.r["editable"]
        self.assertTrue(e["unlinked"])
        self.assertFalse(e["linked"])       # the rule the dblclick always had
        self.assertFalse(e["locked"])
        self.assertFalse(e["silent"])       # nothing to shape
        self.assertFalse(e["nothing"])

    # ---- hover teaches --------------------------------------------------
    def test_hovering_the_line_shows_where_the_point_would_go(self):
        fn = extract_function("sbeAudioGhost", self.src)
        self.assertIn("SBE_LVL_GRAB", fn)
        self.assertIn("sbeStripY(", fn)
        self.assertIn("SBE.kfGhost", fn)
        # ONE LANE, not the whole timeline: a ghost that repainted every clip
        # block and both canvases on each mouse move would make the pointer
        # feel heavy over the lane this is meant to make inviting.
        self.assertIn("sbePaintAudioLane()", fn)
        self.assertNotIn("sbePaint()", fn)
        # ...and it is only computed while nothing is being dragged.
        move = extract_function("sbeOnAudioMove", self.src)
        self.assertIn("if (!k && !SBE.audioDrag) { sbeAudioGhost(ev); return; }",
                      move)
        wave = extract_function("sbeStripWave", self.src)
        self.assertIn("sbe-kf-ghost", wave)

    def test_the_line_is_a_target_and_not_only_a_picture(self):
        wave = extract_function("sbeStripWave", self.src)
        # Three paths: a dark one under it, the line, and the fat invisible one
        # that makes it hittable.
        self.assertIn("sbe-lvl-u", wave)
        self.assertIn("sbe-lvl-hit", wave)
        css = self.src[self.src.index(".sbe-lvl-hit {"):]
        css = css[:css.index("}")]
        self.assertIn("pointer-events: stroke", css)
        self.assertIn("cursor: ns-resize", css)
        # And it is only offered on a strip that can take a point.
        self.assertIn("const shape = sbeStripEditable(c);", wave)

    def test_a_single_click_on_the_line_places_a_point_and_drags_it(self):
        fn = extract_function("sbeLevelClick", self.src)
        self.assertIn("sbeAddKeyframe(", fn)
        self.assertIn("SBE.kfDrag", fn)          # same gesture sets the level
        self.assertIn("setPointerCapture", fn)
        self.assertIn("SBE_LVL_GRAB", fn)
        down = extract_function("sbeOnAudioDown", self.src)
        # The grips are 7px of edge and the more specific target, so trimming
        # can never be mistaken for shaping.
        self.assertIn("if (!ev.target.closest('.sbe-grip') && sbeLevelClick(ev))",
                      down)
        # Double-click still adds one: muscle memory is not taken away to
        # teach somebody else.
        self.assertIn("sbeAddKeyframe(", extract_function("sbeOnAudioDbl", self.src))

    def test_removing_a_point_has_a_route_you_can_find(self):
        fn = extract_function("sbeOnAudioMenu", self.src)
        self.assertIn("sbe-kf", fn)
        self.assertIn("preventDefault", fn)
        self.assertIn("sbeDeleteKeyframe", fn)
        self.assertIn("alane.addEventListener('contextmenu', sbeOnAudioMenu);",
                      self.src)
        # Shift-click stays.
        self.assertIn("ev.shiftKey", extract_function("sbeOnAudioDown", self.src))

    def test_there_is_a_route_that_needs_no_gesture_at_all(self):
        fn = extract_function("sbeAddPointAtPlayhead", self.src)
        self.assertIn("SBE.playhead", fn)
        self.assertIn("sbeAddKeyframe(", fn)
        self.assertIn("sbeStripEditable(", fn)
        # It refuses OUTSIDE the strip rather than putting a point at an end
        # the user was not pointing at.
        self.assertIn("w.len + 1e-6", fn)
        insp = extract_function("sbePaintInspector", self.src)
        self.assertIn("sbeAddPointAtPlayhead()", insp)
        self.assertIn("sbeClearPoints()", insp)
        # A linked strip says what to do instead of offering a refusing button.
        self.assertIn("unlink the sound to shape its", insp)

    def test_the_legend_lists_the_gestures_that_actually_exist(self):
        legend = self.r["legend"]
        for phrase in ("click the yellow line", "drag it to set",
                       "right-click it to remove", "top edge"):
            self.assertIn(phrase, legend)
        # ONE copy of it: the markup renders the function, so the gestures a
        # user is told about cannot drift from the ones the lane implements.
        self.assertIn('<span class="sbe-note sbe-keys" id="sbeKeys"></span>',
                      self.src)
        self.assertIn("sbeKeysLegend()", extract_function("sbePaintKeys", self.src))


class TheStageLayersAreNotOverpainted(unittest.TestCase):
    """A fade may not turn on the layer the stylesheet is turning off.

    THE REGRESSION, reported as "videos are not loading": they were loading
    fine — ready state 4, frames decoded — and a black rectangle was sitting on
    top of them. `sbeFadePaint` wrote the ramp onto BOTH stage layers:

        if (v) v.style.opacity = String(o);
        if (img) img.style.opacity = String(o);

    But opacity is not only the ramp on this stage, it is the LAYER SWITCH:
    `.sbe-stage img.sbe-still` is opacity:0 until it wins `.is-on`. An inline
    value beats the stylesheet, so the hidden still was forced to 1 — and the
    still is last in `#sbeStage` and backed with `#000`. The unit tests were
    green throughout, because every one of them asked about the arrangement and
    none of them asked what the painter wrote onto an element.

    These run the real painter and read the inline styles back.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.r = run_contract()
        cls.src = panel_source()

    def test_only_the_layer_that_is_showing_is_graded(self):
        # The video is on: it carries the ramp, and the still is handed back to
        # the stylesheet — an EMPTY string, not "0". Writing "0" would be a
        # second source of truth for a state the class already owns.
        self.assertEqual(self.r["stage"]["videoOnInFade"], ["0.5", ""])

    def test_the_same_holds_when_the_still_is_the_one_showing(self):
        self.assertEqual(self.r["stage"]["stillOnInFade"], ["", "0.5"])

    def test_a_layer_that_stops_showing_gives_its_inline_value_back(self):
        # THE GHOST CASE, and the one a "write 0 to the other layer" fix would
        # have missed: a layer that carried 0.5 and then loses `.is-on` must be
        # cleared, or the stylesheet's opacity:0 is overridden by a stale 0.5
        # and half a still hangs over the picture.
        self.assertEqual(self.r["stage"]["videoAfterHandover"], "")

    def test_outside_a_fade_the_showing_layer_is_fully_up(self):
        self.assertEqual(self.r["stage"]["outsideFade"], ["1", ""])

    def test_with_no_clip_at_the_playhead_nothing_is_dimmed(self):
        self.assertEqual(self.r["stage"]["noClip"], ["1", ""])

    def test_with_neither_layer_showing_neither_is_written(self):
        self.assertEqual(self.r["stage"]["neitherOn"], ["", ""])

    def test_the_source_monitors_layers_are_not_this_painters_business(self):
        # The source stage carries the SAME opacity switch, so a painter that
        # reached for it would reproduce the bug one monitor to the left. It
        # does not touch them at all — not even to clear them.
        self.assertEqual(self.r["stage"]["sourceUntouched"], [True, True])

    def test_the_overlay_layer_switches_on_display_so_its_ramp_is_free(self):
        # The sibling case, locked so the contrast stays true: `.sbe-ov-layer`
        # is display:none/block, so opacity there is only ever a ramp and
        # nothing it writes can turn the layer on. If that switch is ever
        # changed to opacity, this test is the one that should be read first.
        during, after = self.r["overlayLayer"]["during"], self.r["overlayLayer"]["after"]
        self.assertEqual(during[0], "0.5")
        self.assertTrue(during[1])
        self.assertFalse(after[1])          # off, whatever the stale ramp says

    # ---- the gate, not the instance -------------------------------------
    def test_no_painter_may_write_opacity_onto_a_layer_opacity_switches(self):
        """Every function that writes `.style.opacity` on a stage layer must
        ask which layer is showing first.

        This is the cheap gate the bug got past. It reads the STYLESHEET for
        the layers whose visibility is switched with opacity, then reads the
        JS for anyone writing opacity onto them — and requires the write to be
        conditional on `is-on`. It fails on the code that shipped and passes on
        the fix.
        """
        # 1. which layers does the stylesheet switch with opacity?
        switched = [" ".join(m.group(1).split())
                    for m in re.finditer(r"([^\n{}]*\.is-on)\s*\{([^}]*)\}", self.src)
                    if "opacity" in m.group(2)]
        self.assertEqual(sorted(switched),
                         [".sbe-stage img.sbe-still.is-on", ".sbe-stage video.is-on"],
                         "a new opacity-switched layer appeared — every painter "
                         "that touches it has to ask `is-on` first")
        # ...and each has an OFF state, which is what an inline write defeats.
        self.assertIn("object-fit: contain; opacity: 0; background: #000;", self.src)

        # 2. the ids those rules govern, from the real markup.
        layers = ("sbeVideo", "sbeStill", "sbeSrcVideo", "sbeSrcStill")
        for i in layers:
            self.assertIn('id="%s"' % i, self.src)

        # 3. every opacity write in the file, with the function it lives in.
        offenders = []
        for m in re.finditer(r"\.style\.opacity\s*=", self.src):
            head = self.src.rfind("\nfunction ", 0, m.start())
            if head < 0:
                continue
            name = self.src[head + len("\nfunction "):self.src.index("(", head + 1)]
            body = extract_function(name.strip(), self.src)
            # COMMENTS DO NOT COUNT. The first version of this gate read the
            # raw body, and a comment explaining the rule was enough to satisfy
            # it — a gate that passes on prose is the same failure as a gate
            # that passes on grep.
            code = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
            code = re.sub(r"//[^\n]*", "", code)
            if not any(i in code for i in layers):
                continue          # not a stage layer: not this rule's business
            if "is-on" not in code:
                offenders.append(name.strip())
        self.assertEqual(offenders, [], "these paint a stage layer's opacity "
                         "without asking which layer is showing")


class TheHeaderIsOneRow(unittest.TestCase):
    """"The save button pops up sometimes and it's super big."

    Measured in the owner's browser at 1920x936, dpr 1: `sbeSaveBtn` 1554px
    wide, the header 125px tall in three rows, one spacer eating 1048 of them.

    THE CAUSE IS ONE MISSING DECLARATION. The global form rule is `input,
    textarea, select, button { width: 100% }`. `.ghost-btn` cancels it with
    `width: auto`; `button.primary` never did. Save is a ghost button until
    there is something to save, and `sbePaintChrome` swaps it to `.primary` the
    moment there is — so its flex basis became 100% of the header and the row
    wrapped around it. "Sometimes" was exactly "while unsaved". `flex-wrap:
    wrap` was not the cause, it was the permission.

    Render escaped the same trap by accident: it sits inside `.sbe-split`,
    whose inline-flex shrink-to-fit makes its 100% resolve against its own
    content. The old bottom bar carried `.sbe-actions .primary { width: auto }`
    for exactly this reason; deleting that bar moved a primary into a row with
    no such rule.

    Measured after the fix, in a real browser at four widths — 1920, 1520,
    1280 and 1100 — header 49px, one row, Save 56px, no horizontal overflow at
    any of them.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.src = panel_source()

    def test_the_head_cancels_the_global_stretch_for_both_kinds_of_button(self):
        rule = self.src[self.src.index(".sbe-head .ghost-btn, .sbe-head .primary {"):]
        rule = rule[:rule.index("}")]
        self.assertIn("width: auto", rule)
        self.assertIn("flex: 0 0 auto", rule)

    def test_the_head_may_not_wrap_and_says_so_at_a_specificity_that_wins(self):
        # `.carousel-head { flex-wrap: wrap }` is declared LATER in the sheet,
        # so a single-class `.sbe-head` rule loses to it on source order — the
        # first version of this fix was silently a no-op for that reason.
        self.assertIn(".carousel-head.sbe-head {", self.src)
        rule = self.src[self.src.index(".carousel-head.sbe-head {"):]
        rule = rule[:rule.index("}")]
        self.assertIn("flex-wrap: nowrap", rule)
        # Specificity, not source order, is what wins here — (0,2,0) beats
        # (0,1,0) wherever the two sit — and the one-class version of this
        # rule is what a future edit would reach for.
        self.assertIn("\n    .carousel-head {", self.src)
        self.assertNotIn("\n    .sbe-head { position: relative", self.src)

    def test_the_row_degrades_by_ellipsis_and_folding_never_by_stacking(self):
        # A header that stacks pushes the monitors and the timeline down the
        # column, which is the one thing this layout exists not to do.
        title = self.src[self.src.index(".sbe-head .sbe-title {"):]
        title = title[:title.index("}")]
        self.assertIn("min-width", title)          # or flex refuses to shrink
        self.assertIn("text-overflow: ellipsis", title)
        self.assertIn("white-space: nowrap", title)
        # ...and the two things that fold, in order, at their breakpoints.
        self.assertIn("#sbeStateText { display: none; }", self.src)
        self.assertIn("#sbeDraftName { display: none; }", self.src)

    def test_every_primary_the_client_creates_lives_under_a_width_rule(self):
        """THE CLASS: a `.primary` in a flex row inherits `width: 100%`.

        Whoever adds the next one has to give it a width, and this is the list
        of the places that already do.
        """
        # Where the client TURNS something into a primary at runtime...
        self.assertIn("classList.toggle('primary'", self.src)
        # ...and every editor row that hosts buttons cancels the stretch.
        for rule in (".sbe-head .ghost-btn, .sbe-head .primary {",
                     ".sbe-transport .primary {",
                     ".sbe-recover .ghost-btn, .sbe-recover .primary {",
                     ".sbe-vers-new .primary {"):
            self.assertIn(rule, self.src, "%s is gone — the row it styled "
                          "still hosts a button" % rule)
            block = self.src[self.src.index(rule):]
            block = block[:block.index("}")]
            self.assertIn("width: auto", block,
                          "%s hosts a button and does not cancel the global "
                          "button stretch" % rule)

    def test_the_two_rows_the_calm_layout_added_cannot_wrap_either(self):
        # The transport is the other flex row that gained controls, and the
        # gutter is a fixed column beside a scroller. Measured at 1100px: the
        # transport is 34px and one row, the gutter 124px.
        t = self.src[self.src.index("\n    .sbe-transport {"):]
        t = t[:t.index("}")]
        self.assertIn("flex-wrap: wrap", t)   # it MAY wrap: it is the row that
        # has somewhere to go — nothing below it is pinned to the viewport, and
        # sbeFitMonitors measures whatever height it ends up with. Stated here
        # so the difference from the header is a decision, not an oversight.
        g = self.src[self.src.index("\n    .sbe-gutter {"):]
        g = g[:g.index("}")]
        self.assertIn("flex: 0 0 124px", g)


class TheCalmChrome(unittest.TestCase):
    """The Editor rearranged — and the contract that nothing went missing.

    The bottom bar is deleted, Render came up rather than out, the four-button
    row collapsed into a chip and a kebab, and the soundtrack's row became the
    A2 lane header. Every one of those is a control MOVED, and a moved control
    is one rename away from being a control nobody can reach.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.src = panel_source()

    # Every id the old surface had, and the client still writes to. This list
    # IS the mapping table: if a rearrangement drops one, this turns red
    # rather than the button quietly ceasing to exist.
    REACHABLE = (
        "sbeTitle", "sbeKeepBtn", "sbeDraftName", "sbeDraftDot", "sbeState",
        "sbeUndoBtn", "sbeRedoBtn", "sbeSaveBtn", "sbeRenderBtn", "sbeNleBtn",
        "sbeRenderNote", "sbeVersBtn", "sbeImportBtn", "sbeAutoBtn",
        "sbeBoardBtn", "sbeCloseBtn", "sbeRelinkBtn", "sbeNotices",
        "sbeMusicPath", "sbeMusicMode", "sbeMusic", "sbeMusicChange",
        "sbePrepBtn", "sbePrepCancel", "sbePrepText", "sbePrepBar",
        "sbeMusicWarn", "sbeSnapOn", "sbeZoomRange", "sbeMuteBtn",
        "sbePlayBtn", "sbeTime", "sbeApprox", "sbeKeys", "sbeTlGrab",
        "sbeInspect", "sbeUnplacedWrap", "sbeSrcAddBtn", "sbeSrcPlayBtn",
    )

    def test_nothing_became_unreachable(self):
        for i in self.REACHABLE:
            self.assertIn('id="%s"' % i, self.src, "%s vanished" % i)

    def test_the_bottom_bar_is_gone_and_render_came_up(self):
        # 54px of chrome under the timeline held two buttons. Render is the
        # one thing this screen is FOR, so it is the one filled control on it.
        self.assertNotIn('<div class="sbe-actions">', self.src)
        head = self.src[self.src.index('<header class="carousel-head sbe-head">'):]
        head = head[:head.index("</header>")]
        self.assertIn('id="sbeRenderBtn"', head)
        self.assertIn('class="primary"', head)
        # Export is the same verb pointed elsewhere: under the caret.
        menu = self.src[self.src.index('id="sbeRenderMenu"'):]
        menu = menu[:menu.index("</div>")]
        self.assertIn("sbeNleBtn", menu)
        self.assertIn("sbeRenderNote", menu)

    def test_navigation_nests_under_the_kebab(self):
        menu = self.src[self.src.index('id="sbeMoreMenu"'):]
        menu = menu[:menu.index("\n      </div>")]
        for i in ("sbeVersBtn", "sbeImportBtn", "sbeAutoBtn", "sbeBoardBtn",
                  "sbeCloseBtn"):
            self.assertIn(i, menu)

    def test_a_menu_cannot_push_the_workspace_around(self):
        # THE rule this editor was built on: notices live in one compact
        # surface and never reflow the column. A menu is the same argument —
        # so every one of them is fixed, hidden, and measured off its anchor.
        css = self.src[self.src.index("\n    .sbe-pop {"):]
        css = css[:css.index("}")]
        self.assertIn("position: fixed", css)
        self.assertIn('.sbe-pop[hidden] { display: none !important; }', self.src)
        fn = extract_function("sbePopToggle", self.src)
        self.assertIn("getBoundingClientRect", fn)
        self.assertIn("window.innerWidth", fn)   # clamped on screen
        self.assertIn("window.innerHeight", fn)  # ...and flipped when low
        # One at a time, and click-away closes them.
        self.assertIn("sbePopCloseAll", fn)
        self.assertIn("document.addEventListener('click', sbePopGlobal, true);",
                      self.src)

    def test_escape_closes_the_menu_before_the_document(self):
        # Esc has always closed the document here; a menu is one layer above
        # it, and closing the whole cut out from under an open menu is the
        # surprise this ordering exists to prevent.
        i = self.src.index("if (sbePopAnyOpen()) { sbePopCloseAll(''); return; }")
        j = self.src.index("sbeClose();", i)
        self.assertLess(i, j)

    def test_the_soundtracks_row_became_its_lane_header(self):
        # It was a full-width strip at the top of the column, four inches from
        # the lane it acts on. Identity on the head, verbs in its menu.
        self.assertNotIn('<div class="sbe-prep" id="sbePrepare">', self.src)
        head = self.src[self.src.index('class="sbe-gh sbe-gh-mus"'):]
        head = head[:head.index("<div class=\"sbe-gh sbe-gh-foot\"")]
        self.assertIn("A2", head)
        self.assertIn('id="sbeMusicPath"', head)
        self.assertIn('id="sbeMusicMode"', head)
        self.assertIn("sbePopToggle('sbeMusicMenu'", head)
        menu = self.src[self.src.index('id="sbeMusicMenu"'):]
        menu = menu[:menu.index("\n      </div>")]
        for i in ("sbeMusic", "sbeMusicChange", "sbePrepBtn", "sbePrepCancel",
                  "sbePrepBar", "sbeMusicWarn"):
            self.assertIn(i, menu)

    def test_the_lane_heads_read_the_same_heights_as_the_lanes(self):
        # Two columns that describe the same four rows. They are driven by the
        # SAME variables, so no drag can put a label beside the wrong lane —
        # measured on screen at 1px, which is the scroller's own border.
        css = self.src[self.src.index("\n    .sbe-gutter {"):]
        css = css[:css.index(".sbe-gh-tag {")]
        for var in ("--sbe-ov-h", "--sbe-track-h", "--sbe-alane-h",
                    "--sbe-wave-h"):
            self.assertIn("var(%s" % var, css)
        # And the column budget now counts the WRAPPER, not the scroller.
        fn = extract_function("sbeFitMonitors", self.src)
        self.assertIn("kid.id === 'sbeTlWrap'", fn)

    def test_the_prose_became_affordances(self):
        # Two lines of preview caveat across the middle of the transport, and
        # a run-on keyboard sentence under the timeline. Both are true; both
        # cost a row every time they were shown.
        self.assertIn('class="sbe-info" id="sbeApprox"', self.src)
        self.assertIn("The preview is approximate", self.src)
        self.assertIn('id="sbeKeysBtn"', self.src)
        self.assertIn("sbePopToggle('sbeKeysPop', 'sbeKeysBtn')", self.src)

    def test_the_draft_chip_says_which_of_how_many(self):
        fn = extract_function("sbePaintDraft", self.src)
        self.assertIn("' of ' + n", fn)


class TimelineMarkup(unittest.TestCase):
    """The controls the JS drives have to EXIST, in the real markup."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.src = panel_source()

    def test_the_stage_carries_exactly_one_video(self):
        # Double-buffering measured no benefit against all-intra proxies, and
        # two elements is two decoders.
        stage = self.src[self.src.index('<div class="sbe-stage" id="sbeStage">'):]
        stage = stage[:stage.index("</div>", stage.index("sbeStageEmpty"))]
        self.assertEqual(stage.count("<video"), 1)

    def test_the_video_is_not_muted_in_the_markup(self):
        # Inverted deliberately. The attribute was "belt and braces" for a
        # loader that hard-muted anyway; with sbeSetMute owning mute state, a
        # `muted` in the markup is a second source of truth that disagrees with
        # the first. There is no `autoplay` on this element, so nothing plays
        # without sbePlay, which sets muted explicitly and has its own fallback.
        el = extract_element("sbeVideo", self.src)
        self.assertNotIn("muted", el)
        self.assertNotIn("autoplay", el)

    def test_nothing_in_the_editor_hides_media_with_display_none(self):
        # WebKit will not load a display:none <video>.
        css = self.src[self.src.index(".sbe-stage video {"):]
        css = css[:css.index(".sbe-stage-badge")]
        self.assertIn("opacity: 0", css)
        self.assertNotIn("display: none", css)

    def test_the_preview_says_it_is_an_approximation(self):
        el = self.src[self.src.index('id="sbeApprox"'):]
        el = el[:el.index("</span>")]
        self.assertIn("approximate", el)
        self.assertIn("render is exact", el)

    def test_the_auto_edit_warns_that_it_discards_the_arrangement(self):
        fn = extract_function("sbeAuto", self.src)
        self.assertIn("THROWS AWAY", fn)

    def test_the_render_discloses_that_the_concat_closes_gaps(self):
        fn = extract_function("sbeRenderFilm", self.src)
        self.assertIn("gaps_note", fn)
        self.assertIn("CONCATENATES", fn)

    def test_generate_shows_the_params_that_will_ACTUALLY_render(self):
        # make_job silently drops any form field it does not name, so the
        # server reads the queued job back. The client has to show that, not
        # the request it sent.
        fn = extract_function("sbeGenSubmit", self.src)
        self.assertIn("r.params", fn)
        self.assertIn("sbeGenParams", fn)

    def test_the_preview_prefers_the_proxy_and_labels_the_fallback(self):
        fn = extract_function("sbeClipUrl", self.src)
        self.assertIn("c.proxy", fn)
        self.assertIn("proxy_url", panel_source()[:0] + "proxy_url")
        badge = extract_function("sbeShowFrameAt", self.src)
        self.assertIn("SOURCE (slow", badge)

    # The three below are SOURCE assertions, not behaviour, because the thing
    # under test is CSS and there is no way to run CSS in node. They are here
    # because each one is a rule a browser caught and a reader would not:
    # deleting any of them puts the storyboard's planning column back on top of
    # the editor, which is exactly the bug they were written for.
    def test_the_editor_gets_its_own_two_column_window(self):
        # This test used to lock a MODE: `body.sbe-open` narrowed the
        # storyboard's planning column to 300px while the timeline was open
        # inside it. The 2026-08-17 review's verdict was that the mode was the
        # bug — "the editor deserves its own window … called Editor" — so the
        # rule it locked is gone and this locks its replacement: the Editor is
        # a workflow with two columns of its own, the pool and the cut.
        self.assertNotIn("sbe-open main.layout", self.src)
        css = self.src[self.src.index("THE EDITOR OWNS ITS OWN WINDOW"):]
        css = css[:css.index(".sbe { gap")]
        wide, _, after = css.partition("@media (max-width: 1100px)")
        self.assertIn('body[data-workflow="editor"] main.layout', wide)
        self.assertIn("minmax(0, 300px)", wide)
        # The breakpoint block, brace-matched so the assertions below cannot
        # wander into the rules that follow it.
        depth, narrow = 0, ""
        for ch in after:
            narrow += ch
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
        # The pool is the tab's other half. Folding it away at a breakpoint —
        # which is what the storyboard rule did to its column — would leave
        # the Editor with no way to reach a clip. It stacks instead.
        self.assertNotIn("display: none", narrow)
        self.assertIn("grid-row: 2 / 3", narrow)
        # The editor's own stage element, and the fold that gives it the tab.
        self.assertIn('body[data-workflow="editor"] #edStage { display: flex; }',
                      self.src)

    def test_the_stage_is_TWO_monitors_and_the_source_is_its_own(self):
        # "a split screen with the left screen showing you when you click on
        # the clips … you have two screens." The program monitor is unchanged
        # (the test above still counts exactly one video in it); the source is
        # a second element with its own video, still, badge and empty state.
        row = self.src[self.src.index('<div class="sbe-monitors" id="sbeMonitors">'):]
        row = row[:row.index('<div class="sbe-transport">')]
        self.assertIn('id="sbeSrcVideo"', row)
        self.assertIn('id="sbeSrcStill"', row)
        self.assertIn('id="sbeSrcAddBtn"', row)
        self.assertIn('id="sbeStage"', row)                 # the program
        self.assertIn("Click a clip to preview it.", row)
        # Two monitors, two videos — and no third one that nobody owns.
        self.assertEqual(row.count("<video id="), 2)
        # The source is a player, not a second editor: no track, no trim.
        self.assertNotIn("sbe-track", row)

    def test_only_one_monitor_ever_plays(self):
        # Two elements is two decoders only if both decode. sbePlay stops the
        # source and sbeSrcPlay stops the program, so it is one at a time —
        # which is also the only way "one opinion about sound" can be true.
        self.assertIn("sbeSrcStop()", extract_function("sbePlay", self.src))
        self.assertIn("sbeStop()", extract_function("sbeSrcPlay", self.src))
        self.assertIn("sbeSrcStop()", extract_function("sbeSuspend", self.src))

    def test_the_mute_switch_governs_BOTH_monitors(self):
        fn = extract_function("sbeSetMute", self.src)
        self.assertIn("sbeSrcVideo", fn)
        self.assertIn("sv.muted = SBE.muted", fn)

    def test_a_deliberate_pause_is_not_mistaken_for_a_blocked_autoplay(self):
        # Measured: pressing Play on the program rejected the source's pending
        # play() with AbortError, the catch read that as "the browser blocked
        # sound", MUTED the editor and started the source playing again under
        # the program. Both players now tell the two apart.
        for name in ("sbeSrcPlay", "sbePlay"):
            fn = extract_function(name, self.src)
            self.assertIn("AbortError", fn)

    def test_the_source_prefers_a_proxy_and_says_when_it_has_none(self):
        # A pool row has no proxy of its own — it is a path, not a clip — so
        # the URL is the proxy of a clip already cut from the same file when
        # there is one. A Generations row has never been near this document
        # and plays the file, which is fine forward-only and is labelled.
        fn = extract_function("sbeSrcUrl", self.src)
        self.assertIn("c.proxy", fn)
        self.assertIn("SBE.proxyUrl", fn)
        self.assertIn("/file?path=", fn)
        self.assertIn("source file", extract_function("sbePaintSource", self.src))

    def test_clicking_a_pool_row_previews_it_and_the_plus_still_adds(self):
        # The verb on the row CHANGED, which is the whole two-screen point:
        # "you can watch the clips before you add them". The old verb keeps a
        # control of its own so nobody who already knows a clip has to watch
        # it — and a <button> inside a <button> is not markup, hence the div.
        fn = extract_function("edPoolPaint", self.src)
        self.assertIn('onclick="edPoolPreview(', fn)
        self.assertIn('class="ed-pool-add"', fn)
        self.assertIn("edPoolAdd(", fn)
        self.assertIn('role="button"', fn)
        self.assertNotIn('<button type="button" class="ed-pool-row"', fn)
        # Add-from-the-source-monitor is the SAME call, by construction.
        self.assertIn("edPoolAdd(SBE.srcIndex)", extract_function("sbeSrcAdd", self.src))

    def test_a_drag_that_ends_in_a_click_does_not_also_preview(self):
        # edPoolAdd has carried this guard since drop-where-you-dropped-it
        # shipped; the preview needs the same one or every drop would change
        # the left screen as well as the track.
        self.assertIn("ED.suppressClick", extract_function("edPoolPreview", self.src))

    def test_the_preview_frame_is_the_shape_of_the_picture(self):
        # `width: 100%` + `aspect-ratio` + `max-height` do not compose: a block
        # box's auto width is stretch-fit and outranks the ratio, so the frame
        # came out 1070x300 with a 16:9 picture floating in the middle of it and
        # ~270px of black either side. Measured in a browser; reported as
        # "it looks weird, it looks warped".
        css = self.src[self.src.index(".sbe-stage {"):]
        css = css[:css.index(".sbe-stage video")]
        self.assertIn("aspect-ratio: 16 / 9", css)
        # Parse real declarations: the prose above them quotes the broken rule,
        # and a substring test also trips over the legitimate `max-width: 100%`.
        body = css[css.index("{") + 1:]
        body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
        decls = {}
        for part in body.split(";"):
            if ":" in part:
                prop, _, val = part.partition(":")
                decls[prop.strip()] = val.strip()
        self.assertEqual(decls.get("width"), "fit-content")
        self.assertEqual(decls.get("max-width"), "100%")   # cap, not stretch
        self.assertEqual(decls.get("margin-inline"), "auto")
        # A definite height is what lets the ratio drive the width. Without it
        # `fit-content` collapses instead. It is a variable now rather than a
        # min() — sbeFitMonitors measures and writes it, and .sbe-monitors
        # carries the fallback for the frame before that runs.
        self.assertEqual(decls.get("height"), "var(--sbe-prog-h)")
        self.assertNotIn("max-height: min(", css)

    def test_the_stage_leaves_room_for_everything_under_it(self):
        # The budget was `100vh - 748px` and it was measured against a column
        # that was 170px shorter than this one (the editor's stage-pane did
        # not span the bottomPane's grid row) and against a stage that was
        # alone on its row. Both changed, so the number did.
        # RE-MEASURED 2026-08-18 at 1440x900, live, in the Editor's own tab:
        # the column is 100vh - 116px (88px of header + 28px of stage-pane
        # padding) and everything else in it comes to 452px — header 50,
        # prepare 55, transport 32, ruler + waveform + 78px track + 12px
        # scrollbar 164, inspector 38, action bar 65, six 8px gaps 48.
        # Hence 568. Both directions are locked: too small is a cramped
        # picture, too large is the sticky Render bar back on the timeline.
        css = self.src[self.src.index(".sbe-monitors {"):]
        css = css[:css.index(".sbe-mon {")]
        self.assertIn("calc(100vh - 568px)", css)
        self.assertNotIn("calc(100vh - 748px)", css)
        self.assertNotIn("calc(100vh - 700px)", css)
        self.assertNotIn("calc(100vh - 819px)", css)
        # The whole reason the column got 170px taller. This rule read
        # `grid-row: 1 / 2` for two releases, which reserved the bottomPane's
        # row across the entire grid while only the 300px left column had
        # anything in it — 1140x170 of page background under the timeline,
        # which is the band the owner circled.
        self.assertIn("grid-column: 2 / 3; grid-row: 1 / span 2;", self.src)

    def test_the_timeline_scroller_shows_a_scrollbar_you_can_actually_see(self):
        # THE defect: "when you get to the 36 seconds, you cannot scroll."
        # `overflow-x: auto` was always here and the box always could scroll —
        # but macOS overlay scrollbars paint nothing until something is
        # already scrolling, and nothing on a mouse scrolls a horizontal box.
        # Declaring ::-webkit-scrollbar opts out of overlay scrollbars; naming
        # the STANDARD `scrollbar-width` puts them straight back, because
        # Chrome 121+ ignores every ::-webkit- rule once it is present.
        # Measured: with scrollbar-width, offsetHeight - clientHeight was 2px
        # (the borders alone); without it, 14px.
        css = self.src[self.src.index("\n    .sbe-tl {"):]
        css = css[:css.index(".sbe-tl-inner {")]
        self.assertIn("overflow-x: scroll", css)
        self.assertIn(".sbe-tl::-webkit-scrollbar { height: 12px; }", css)
        # Declarations only — the comment above them quotes the property it is
        # warning about, exactly like the aspect-ratio test one screen up.
        decls = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
        self.assertNotIn("scrollbar-width", decls)
        # And it is the row that absorbs the column's leftover height, which
        # is what keeps a tall window from ending in a dead band. Its floor is
        # a variable because the user owns it — the handle on the transport's
        # bottom edge writes it — and the fallback is the constant.
        self.assertIn("flex: 1 1 auto", css)
        # The floor lives on the WRAPPER now — the gutter shares the height —
        # and the scroller's job is to fill it.
        wrap = self.src[self.src.index("\n    .sbe-tlwrap {"):]
        wrap = wrap[:wrap.index("}")]
        self.assertIn("min-height: var(--sbe-tl-h, 280px)", wrap)
        self.assertIn(".sbe > .sbe-tlwrap { flex: 1 1 auto; }", self.src)

    def test_the_zoom_slider_exists_beside_the_buttons(self):
        # "Normal video editors usually have two sliders: one to move and one
        # to make the clips smaller." The scrollbar above is the first.
        zoom = self.src[self.src.index('<span class="sbe-zoom">'):]
        zoom = zoom[:zoom.index("</span>", zoom.index("sbeZoomRange"))]
        self.assertIn('id="sbeZoomRange"', zoom)
        self.assertIn('type="range"', zoom)
        self.assertIn('oninput="sbeZoomSlide(', zoom)
        self.assertIn("sbeZoom(-1)", zoom)          # the buttons stay
        self.assertIn("sbeZoom(1)", zoom)
        # The − / + ladder is clamped to the same live floor as the slider, or
        # the two controls would disagree about what "all the way out" means.
        self.assertIn("sbeZoomMin()", extract_function("sbeZoom", self.src))
        self.assertIn("sbeZoomMin()", extract_function("sbeZoomSlide", self.src))

    def test_shift_wheel_pans_and_alt_wheel_zooms(self):
        fn = extract_function("sbeOnTlWheel", self.src)
        self.assertIn("ev.altKey", fn)
        self.assertIn("ev.shiftKey", fn)
        self.assertIn("sbeZoomTo(", fn)
        self.assertIn("scrollLeft", fn)
        # A plain vertical wheel belongs to the column, which scrolls.
        self.assertIn("ev.deltaX", fn)

    def test_the_monitor_row_is_measured_rather_than_assumed(self):
        # "Measure, don't guess" — and the one thing it may NOT measure is the
        # row it is sizing, or the budget becomes its own input. The timeline
        # is counted at its CSS floor for exactly that reason.
        fn = extract_function("sbeFitMonitors", self.src)
        self.assertIn("SBE_TL_MIN_H", fn)
        self.assertIn("clientHeight", fn)
        self.assertIn("sbeMonitorFit(", fn)
        self.assertIn("--sbe-prog-h", fn)
        self.assertIn("--sbe-rail-w", fn)
        # The corrective pass is bounded: it can only shrink, and only once.
        self.assertIn("scrollHeight - col.clientHeight", fn)
        self.assertEqual(fn.count("apply(budget"), 2)

    def test_the_unbounded_list_is_the_one_that_scrolls_not_the_inspector(self):
        # Measured live at 1440x900 on a nine-clip film with thirteen unplaced
        # shots: #sbeInspect was 36px around 223px of content, so "Unlink
        # sound" — the only entry point the J/L feature has — was off-screen,
        # and the one control fully visible at the bottom of that unmarked
        # scroll was Ripple delete. The inspector's content is bounded and
        # known; the unplaced strip is not.
        css = self.src[self.src.index(".sbe-rail {"):]
        css = css[:css.index(".sbe-stage {")]
        self.assertIn(".sbe-rail > .sbe-inspect {\n      flex: 0 0 auto", css)
        self.assertIn(".sbe-rail > .sbe-unplaced {\n      flex: 1 1 auto", css)
        # ...and it still fills the rail when there is nothing unplaced, or it
        # is a small card floating in page background.
        self.assertIn(".sbe-rail:has(> .sbe-unplaced[hidden]) > .sbe-inspect", css)

    def test_the_column_is_re_fitted_when_its_BOX_changes_not_just_the_window(self):
        # Measured: a viewport change from 1900x1000 to 1440x900 left both
        # monitors 437px tall inside a 501px column — 16:9 boxes squashed to
        # 1.15:1 by `max-width` — because the window's resize event never
        # reached sbePaint. The column is the box the budget is about.
        wire = self.src[self.src.index("(function sbeWire() {"):]
        wire = wire[:wire.index("\n})();")]
        self.assertIn("ResizeObserver", wire)
        self.assertIn("edStage", wire)
        # Coalesced, and NOT on rAF: measured in a preview pane the compositor
        # had backgrounded, where the animation-frame callback never ran at
        # all and the monitors kept the previous window's height. A resize is
        # exactly when a background tab is about to be looked at.
        self.assertIn("pending = setTimeout(", wire)
        self.assertNotIn("pending = requestAnimationFrame", wire)

    def test_the_action_bar_is_one_row_because_the_button_is_not_full_bleed(self):
        # `button { width: 100% }` is the panel's base rule and this bar never
        # overrode it: the Render button measured 1590px, wrapped to its own
        # line, and doubled the sticky bar's height onto the inspector.
        css = self.src[self.src.index(".sbe-actions {"):]
        css = css[:css.index(".sbe-note {")]
        self.assertIn(".sbe-actions .primary", css)
        self.assertIn("width: auto", css)

    def test_the_preview_is_not_hard_muted(self):
        # Proxies carry audio (recipe v2+) and on a dialogue film the clips ARE
        # the performance. `v.muted = true` in the loader silently threw that
        # away, so the fix that gave proxies an audio track shipped inaudible
        # and the owner reported the same bug twice: "when you run in the
        # timeline, you cannot listen to the sound."
        fn = extract_function("sbeLoadInto", self.src)
        self.assertNotIn("v.muted = true", fn)
        self.assertIn("v.muted = !!SBE.muted", fn)

    def test_sound_is_on_by_default(self):
        # A muted editor is indistinguishable from a broken one, so the stored
        # preference has to default to audible: only an explicit '1' mutes.
        state = self.src[self.src.index("window.SBE = {"):]
        state = state[:state.index("const SBE_MIN_CLIP")]
        self.assertIn("localStorage.getItem('sbeMuted') === '1'", state)

    def test_one_switch_drives_picture_soundtrack_and_button(self):
        fn = extract_function("sbeSetMute", self.src)
        for target in ("v.muted = SBE.muted", "a.muted = SBE.muted", "sbeMuteBtn"):
            self.assertIn(target, fn)
        self.assertIn("localStorage.setItem('sbeMuted'", fn)

    def test_blocked_autoplay_falls_back_to_muted_rather_than_to_nothing(self):
        # Unmuted play needs user activation. If Chrome refuses, losing the
        # sound beats losing the picture — and a Play button that visibly does
        # nothing is the worst of the three.
        fn = extract_function("sbePlay", self.src)
        self.assertIn("v.muted = true", fn)
        self.assertIn("sbeSetMute(true", fn)

    def test_leaving_the_tab_suspends_the_editor_and_never_closes_it(self):
        # The regression this replaces: sbSyncStage owned `body.sbe-open` and
        # workflowSwitch ran sbeTeardown() on the way out, so glancing at the
        # gallery threw away the open document and its undo stack. A tab
        # switch is not a close — it stops the clock and the picture, and
        # flushes the debounced save, and that is all.
        sync = extract_function("sbSyncStage", self.src)
        self.assertNotIn("classList.toggle('sbe-open'", sync)
        self.assertNotIn("SBE.open", sync)
        self.assertNotIn("sbeTeardown", extract_function("sbTeardown", self.src))
        wf = extract_function("workflowSwitch", self.src)
        self.assertIn("sbeSuspend()", wf)
        self.assertNotIn("sbeTeardown()", wf)
        susp = extract_function("sbeSuspend", self.src)
        self.assertIn("sbeStop()", susp)                    # the picture stops
        self.assertIn("sbeSave(true)", susp)                # the work is kept
        self.assertNotIn("SBE.open = false", susp)          # the document is not

    def test_the_document_only_closes_when_the_document_is_closed(self):
        close = extract_function("sbeCloseDoc", self.src)
        self.assertIn("SBE.open = false", close)
        self.assertIn("SBE.clips = []", close)
        # Esc and the header's Close both come here.
        self.assertIn("sbeCloseDoc()", extract_function("sbeClose", self.src))

    def test_the_surface_is_called_Editor_before_it_is_called_anything(self):
        # One name, everywhere — and the default is the name of the place,
        # not the name of the widget. It said "Timeline" while the tab said
        # Editor and the rail said Arrange: three words for one screen.
        self.assertIn('id="sbeTitle">Editor<', self.src)
        self.assertIn(">Open in Editor<", self.src)
        self.assertIn('<span class="sb-rail-t">Edit</span>', self.src)
        self.assertNotIn(">Arrange<", self.src)

    def test_the_editor_tab_exists_and_is_always_reachable(self):
        # A tab with no board open is the whole point: "an editor that only
        # exists inside a storyboard is an editor most clips can never reach".
        self.assertIn('<button data-workflow="editor">', self.src)
        self.assertIn(">Editor<", self.src)
        # It MUST be in the localStorage restore list or the tab never comes
        # back across a reload — the trap the comment beside it documents.
        boot = self.src[self.src.index("const saved = localStorage.getItem('phos_workflow')"):]
        boot = boot[:boot.index("workflowSwitch(saved);")]
        self.assertIn("saved === 'editor'", boot)
        # And it opens with nothing on disk: an empty state, not an error.
        self.assertIn('id="edEmpty"', self.src)
        self.assertIn("edShowPicker", extract_function("edInit", self.src))

    def test_a_remembered_document_that_is_gone_falls_back_to_the_picker(self):
        # The silent-failure restore the review flagged: a deleted film left
        # the Editor showing an error box for a film nobody asked to open.
        init = extract_function("edInit", self.src)
        self.assertIn("onMissing", init)
        load = extract_function("sbeLoad", self.src)
        self.assertIn("SBE.onMissing", load)
        self.assertIn("r.corrupt", load)     # corrupt is still NOT swallowed

    def test_the_document_id_is_not_the_open_boards_id(self):
        fn = extract_function("sbeOpen", self.src)
        self.assertNotIn("SBE.id = SB.id", fn)
        self.assertNotIn("if (!SB.id) return", fn)
        self.assertIn("edDoc()", fn)         # last document, remembered
        self.assertIn("edRemember(want)", fn)

    def test_the_narrow_layout_stops_stretching_so_the_page_can_scroll(self):
        # .layout is `flex: 1 1 auto` in a flex body, so the breakpoint's
        # `height: auto` never did anything: the grid kept the full viewport
        # height, its three auto rows were squeezed, and ~480px of the brief
        # spilled through the stage.
        mq = self.src[self.src.index('@media (max-width: 900px)'):]
        mq = mq[:mq.index(".sb-shot-head { flex-wrap: wrap; }")]
        self.assertIn("flex: 0 0 auto;", mq)

    # ---- the media pool -------------------------------------------------
    def test_the_window_prompt_import_is_gone(self):
        # It asked the user to type the NUMBER of a film from a numbered list,
        # in an OS dialog, while the gallery holding every clip they had ever
        # made was display:none for the whole surface.
        self.assertNotIn("sbeImportOpen", self.src)
        self.assertNotIn("Bring clips from which film?", self.src)

    def test_the_pool_offers_all_three_sources(self):
        pane = self.src[self.src.index('id="edSectionTab"'):]
        pane = pane[:pane.index('id="edPoolNote"')]
        for src in ('data-src="film"', 'data-src="other"', 'data-src="gallery"'):
            self.assertIn(src, pane)
        for word in ("This __SEQCAP__", "Other __SEQS__", "Generations"):
            self.assertIn(word, pane)

    def test_each_source_is_fed_by_the_endpoint_that_owns_it(self):
        fn = extract_function("edPoolRefresh", self.src)
        self.assertIn("SBE.pool", fn)            # this film — off the payload
        self.assertIn("'/outputs?limit=", fn)    # the generations gallery
        other = extract_function("edPoolLoadFilms", self.src)
        self.assertIn("'/storyboard/list'", other)
        shots = extract_function("edPoolLoadFilmShots", self.src)
        self.assertIn("'/storyboard/get?id='", shots)

    def test_adding_a_clip_lands_it_on_the_TRACK_and_saves(self):
        # Not on the board, and not in a chip row waiting to be placed: the
        # click said "put this in my film", so it goes on the timeline.
        fn = extract_function("edPoolAdd", self.src)
        self.assertIn("'/storyboard/edit/add-clip'", fn)
        self.assertIn("sbePlaceUnplaced", fn)
        self.assertIn("sbeFilmDuration(cs)", fn)          # a CLICK: at the END
        self.assertIn("sbeInsertAt(cs, item, at)", fn)    # a DROP: where it fell
        self.assertIn("sbeSave(true)", fn)

    def test_adding_a_clip_from_another_film_uses_the_import_subset_param(self):
        # `only=` has been on the server since the import shipped and no
        # screen could send it.
        self.assertIn("fd.set('only'", extract_function("edPoolAdd", self.src))

    def test_adding_a_clip_never_ejects_the_user_from_the_editor(self):
        # The old import ended with sbOpen(SBE.id), which threw the user back
        # onto the shot list — after an action whose whole purpose was to
        # carry on cutting.
        fn = extract_function("edPoolAdd", self.src)
        self.assertNotIn("workflowSwitch", fn)
        self.assertNotIn("sbOpen(", fn)
        self.assertNotIn("sbShow(", fn)

    def test_pool_thumbnails_are_lazy_because_media_elements_are_finite(self):
        # Chrome caps media elements per document and the carousel already
        # holds hundreds on a working install. src is attached on the way in
        # and taken away on the way out.
        fn = extract_function("edPoolPaint", self.src)
        self.assertIn("data-src=", fn)
        self.assertNotIn('video preload="metadata" muted playsinline src=', fn)
        obs = extract_function("edPoolObserve", self.src)
        self.assertIn("IntersectionObserver", obs)
        self.assertIn("removeAttribute('src')", obs)
        # …and the first screenful does not wait for a callback, because
        # intersection callbacks do not run in an occluded tab.
        self.assertIn(".slice(0, 12)", obs)

    # ---- relink ---------------------------------------------------------
    def test_the_relink_banner_says_how_many_and_offers_one_button(self):
        fn = extract_function("sbePaintRelink", self.src)
        self.assertIn("finished since this cut", fn)
        self.assertIn("SBE.relink", fn)
        bar = self.src[self.src.index('id="sbeRelink"'):]
        bar = bar[:bar.index("</div>")]
        self.assertIn("sbeRelink()", bar)
        self.assertIn("Use the finished versions", bar)

    def test_relink_saves_first_then_asks_the_server_to_rewrite(self):
        # The server rewrites edit.json on disk; an unsaved arrangement would
        # be rewritten out from under itself.
        fn = extract_function("sbeRelink", self.src)
        self.assertLess(fn.index("sbeSave(true)"),
                        fn.index("/storyboard/edit/relink"))
        self.assertIn("sbeAdopt(r, true)", fn)

    # ---- WAVE 2: the surfaces the new verbs need -------------------------
    def test_the_pool_offers_images_as_a_fourth_source(self):
        pane = self.src[self.src.index('id="edSectionTab"'):]
        pane = pane[:pane.index('id="edPoolNote"')]
        self.assertIn('data-src="images"', pane)
        self.assertIn("Images", pane)
        # The gallery has carried kind:'image' since it was unified; what was
        # missing was a clip that could BE one.
        fn = extract_function("edPoolRefresh", self.src)
        self.assertIn("o.kind === 'image'", fn)
        self.assertIn("kind: 'still'", fn)

    def test_a_still_thumbnail_is_capped_like_every_other_thumbnail(self):
        # The still branch emits a bare <img> and the sizing rule named only
        # `video` and `.ed-pool-blank`, so an image row had no width cap:
        # measured live at 180x101 in a 261px row with the text column
        # squeezed to 22px, every filename ellipsised to one character and a
        # dot, rows three times the height of the Generations tab next door.
        css = self.src[self.src.index(".ed-pool-row {"):]
        css = css[:css.index(".ed-pool-row .ed-pool-name {")]
        self.assertIn(".ed-pool-row img, .ed-pool-row video", css)
        # ...and the text column has to be able to claim what is left.
        self.assertIn(".ed-pool-row .ed-pool-meta {\n      min-width: 0; "
                      "flex: 1 1 auto;", css)

    def test_black_is_made_not_picked(self):
        # It has no file, so it has no row in any list. It is a length.
        pane = self.src[self.src.index('id="edSectionTab"'):]
        pane = pane[:pane.index('id="edPoolNote"')]
        self.assertIn('id="edSlugSecs"', pane)
        self.assertIn("edAddSlug()", pane)
        fn = extract_function("edAddSlug", self.src)
        self.assertIn("kind: 'slug'", fn)
        self.assertIn("sbeInsertAt", fn)
        # No server round trip: nothing on disk to check, no geometry to
        # probe, no proxy to build.
        self.assertNotIn("fetch(", fn)

    def test_a_still_previews_from_image_not_file(self):
        # /file is OUTPUT-bound and the image library lives under
        # panel_uploads, so the obvious fallback would 404 every still.
        fn = extract_function("sbeClipUrl", self.src)
        self.assertIn("'/image?path='", fn)
        self.assertLess(fn.index("/image?path="), fn.index("/file?path="))
        self.assertIn('id="sbeStill"', self.src)

    def test_a_slug_previews_as_black_with_no_file_on_disk(self):
        fn = extract_function("sbeShowFrameAt", self.src)
        self.assertIn("kind === 'slug'", fn)
        # Both layers off leaves the stage's own black — the one case where
        # the preview is not an approximation at all.
        head = fn[fn.index("kind === 'slug'"):]
        head = head[:head.index("kind === 'still'")]
        self.assertIn("v.classList.remove('is-on')", head)
        self.assertNotIn("sbeLoadInto", head)

    def test_a_still_never_reaches_the_missing_proxy_warning(self):
        # "SOURCE (slow — run Prepare)" is advice to run a Prepare that would
        # do nothing: a still needs no proxy and a slug has no file.
        fn = extract_function("sbePaintTrack", self.src)
        self.assertIn("kind === 'video' && !c.proxy ? 'slow' : ''", fn)

    def test_the_track_paints_the_three_kinds_distinctly(self):
        fn = extract_function("sbePaintTrack", self.src)
        self.assertIn("'sbe-clip is-' + kind", fn)
        self.assertIn("sbe-cl-thumb", fn)          # a still shows its picture
        self.assertIn("'black'", fn)               # a slug says what it is
        css = self.src[self.src.index("THE TIMELINE EDITOR — the sixth"):]
        css = css[:css.index("The panel has no layout breakpoint today")]
        self.assertIn(".sbe-clip.is-slug", css)
        self.assertIn(".sbe-clip.is-still", css)

    def test_the_brightness_slider_previews_live_and_commits_once(self):
        # oninput at pointer speed would push eighty undo steps and eighty
        # saves for one gesture; onchange fires once, when the drag ends.
        fn = extract_function("sbePaintInspector", self.src)
        self.assertIn('oninput="sbeBrightPreview(this.value)"', fn)
        self.assertIn('onchange="sbeBrightCommit(this.value)"', fn)
        preview = extract_function("sbeBrightPreview", self.src)
        self.assertNotIn("sbeMutate", preview)
        self.assertNotIn("sbeQueueSave", preview)
        commit = extract_function("sbeBrightCommit", self.src)
        self.assertIn("sbeSetBrightness", commit)
        self.assertIn("sbeQueueSave", commit)
        # A slug's colour is what a slug IS — there is nothing to grade.
        self.assertIn("(kind === 'slug') ? '' :", fn)

    def test_the_preview_filter_drives_both_stage_layers(self):
        # One place decides what the stage looks like, so the video and the
        # still can never disagree about the grade.
        fn = extract_function("sbeApplyPreviewFilter", self.src)
        self.assertIn("sbeBrightnessCss", fn)
        self.assertIn("sbeEl('sbeVideo')", fn)
        self.assertIn("sbeEl('sbeStill')", fn)

    def test_the_pool_drag_is_pointer_based_and_portals_its_ghost(self):
        # HTML5 dragstart would swallow the track's pointer capture; the same
        # substrate the track already uses coexists by construction.
        self.assertNotIn("ondragstart", self.src)
        start = extract_function("edPoolDragStart", self.src)
        self.assertIn("pointermove", start)
        self.assertIn("pointerup", start)
        move = extract_function("edPoolDragMove", self.src)
        self.assertIn("document.body.appendChild", move)   # NOT inside the list
        self.assertIn("ed-drag-ghost", move)
        end = extract_function("edPoolDragEnd", self.src)
        # A press that never leaves the row ends in a click on it, and without
        # the flag that click would act a second time — adding the clip at the
        # end of the track, and (since the two-screen pass) also changing the
        # left monitor. But a drag ONTO THE TRACK fires no click on the row at
        # all, so arming the flag unconditionally left it set to eat the next
        # genuine click: drop a clip, click another to preview it, nothing
        # happens until the second click. The guard is armed only when the
        # pointer came back up inside the row it started in.
        self.assertIn("d.el.contains(ev.target)", end)
        self.assertNotIn("ED.suppressClick = true", end)
        # ...and the pool's drag stops the browser painting the whole editor
        # in selection blue on its way past.
        self.assertIn("ev.preventDefault()", start)
        add = extract_function("edPoolAdd", self.src)
        self.assertIn("if (ED.suppressClick && dropAt === undefined)", add)
        # Add-at-the-end still works and still lands at the end — it is the +
        # on the row now rather than the whole row, because clicking the row
        # previews. `stopPropagation` is what keeps one press from doing both.
        pool = extract_function("edPoolPaint", self.src)
        self.assertIn("edPoolAdd(' + i + ')", pool)
        self.assertIn("event.stopPropagation()", pool)
        self.assertIn('onpointerdown="edPoolDragStart(event,', pool)

    def test_shift_reorders_and_a_plain_drag_still_moves(self):
        down = extract_function("sbeOnTrackDown", self.src)
        self.assertIn("ev.shiftKey ? 'reorder' : 'move'", down)
        move = extract_function("sbeOnTrackMove", self.src)
        self.assertIn("sbeReorderTo(SBE.clips", move)
        self.assertIn("sbeMoveTo(SBE.clips", move)
        # The legend teaches it, in the same voice as the Alt override.
        self.assertIn("reorder instead of move", extract_function(
            "sbeKeysLegend", self.src))

    def test_the_nle_export_is_offered_and_says_what_the_audio_is(self):
        self.assertIn('id="sbeNleBtn"', self.src)
        self.assertIn("Export for Premiere / Resolve / AE", self.src)
        fn = extract_function("sbeExportNle", self.src)
        self.assertIn("'/storyboard/edit/export-nle'", fn)
        # Stems, not the ducked mix — said on screen, not in a support thread.
        self.assertIn("STEMS", fn)
        self.assertIn("/storyboard/edit/reveal", fn)

    def test_the_music_lane_is_below_the_clip_track(self):
        # "It is unusual that the music is above the clips; it's not normal."
        # The inner is a flex column with no `order` anywhere, so DOM order is
        # screen order and this assertion is the layout.
        inner = self.src[self.src.index('id="sbeInner"'):]
        inner = inner[:inner.index('id="sbeHead"')]
        self.assertLess(inner.index('id="sbeRuler"'), inner.index('id="sbeTrack"'))
        self.assertLess(inner.index('id="sbeTrack"'), inner.index('id="sbeWave"'))
        self.assertLess(inner.index('id="sbeWave"'), inner.index('id="sbeWaveNone"'))
        self.assertNotIn("order:", inner)

    def test_every_empty_surface_says_what_is_true_and_what_to_do(self):
        # A pool that goes blank reads as broken, and the four sources fail
        # for four different reasons.
        fn = extract_function("edPoolPaint", self.src)
        for phrase in ("No generations yet",
                       "No images yet — press Upload",
                       "No other __SEQS__ yet",
                       "This __SEQ__ has no rendered clips yet",
                       "matches"):
            self.assertIn(phrase, fn)
        # ...and an empty timeline names the next move rather than sitting
        # there with a clip count of zero — ON THE TIMELINE. It used to be
        # written into the inspector, a short auto-scrolled box in the corner,
        # so the user of a brand-new draft read the sentence starting mid-word
        # while the track itself rendered nothing at all.
        track = extract_function("sbePaintTrack", self.src)
        self.assertIn("sbe-track-empty", track)
        self.assertIn("Nothing on the timeline yet", track)
        self.assertNotIn("Nothing on the timeline yet",
                         extract_function("sbePaintInspector", self.src))
        # The program monitor's empty state was markup nobody ever filled, so
        # the two monitors spoke two languages about the same condition.
        chrome = extract_function("sbePaintChrome", self.src)
        self.assertIn("sbeStageEmpty", chrome)
        self.assertIn("Nothing on the timeline yet", chrome)

    def test_the_pool_can_take_a_file_from_the_user(self):
        # "You cannot upload your own images and insert them into the
        # timeline." The input is hidden because a bare file input cannot be
        # made to look like anything else in this panel; the button is the
        # control, and it belongs to the source that shows what it uploads.
        self.assertIn('id="edPoolFile"', self.src)
        self.assertIn('id="edPoolUploadBtn"', self.src)
        el = extract_element("edPoolFile", self.src)
        self.assertIn("multiple", el)
        for ext in (".png", ".jpg", ".webp", ".mp4", ".mov", ".webm"):
            self.assertIn(ext, el)
        src = extract_function("edPoolSrc", self.src)
        self.assertIn("edPoolUploadBtn", src)
        fn = extract_function("edPoolUpload", self.src)
        self.assertIn("'/storyboard/edit/upload'", fn)
        # ONE REQUEST PER FILE, so a rejected one names itself instead of
        # failing the batch behind it.
        self.assertIn("for (const f of list)", fn)

    def test_a_picture_is_added_as_a_still_whatever_the_row_forgot(self):
        # A pool row that reached the add path without its `kind` — an upload
        # dragged onto the track — made the client ask /file for a PNG and
        # hand it to a <video>. Black frame, no explanation.
        fn = extract_function("edPoolAdd", self.src)
        self.assertIn("row.kind === 'still'", fn)
        self.assertIn("png|jpe?g|webp", fn)

    def test_an_uploaded_clip_is_a_pool_row_like_any_other(self):
        fn = extract_function("edPoolUploadRow", self.src)
        self.assertIn("'still'", fn)
        self.assertIn("SBE_STILL_SECONDS", fn)
        refresh = extract_function("edPoolRefresh", self.src)
        self.assertIn("'/storyboard/edit/uploads'", refresh)
        self.assertIn("ED.uploaded", refresh)

    def test_the_editor_uses_the_panels_own_colour_tokens(self):
        # --ink / --ink-500 / --line are not defined anywhere in this panel.
        css = self.src[self.src.index("THE TIMELINE EDITOR — the sixth"):]
        css = css[:css.index("The panel has no layout breakpoint today")]
        for stray in ("var(--ink", "var(--line", "var(--ink-500"):
            self.assertNotIn(stray, css)
        self.assertIn("var(--panel", css)
        self.assertIn("var(--border", css)


# ---------------------------------------------------------------------------
# ONE SHOT IS A MODE. The owner's ruling: a continuous shot of 30 s – 2 min is
# a different way of making a video, not a length on the clip form — its own
# chip in the mode bar, its own panel (length, beats, anchor image, the two
# continuity toggles), and a normal clip must never carry take_seconds. These
# run the REAL setMode / oneshotEnter / oneshotLeave / setTakeSeconds against
# a DOM shim built from the REAL chips in the markup, so "leaving the mode
# zeroes the take" is executed, not grepped for.
# ---------------------------------------------------------------------------

ONESHOT_FUNCTIONS = (
    "setMode", "defaultRemixMode", "updatePromptPlaceholder",
    "takePartSeconds", "setTakeSeconds", "takePrefill", "takePrefillClick",
    "beatsInput", "oneshotActive", "oneshotBackendMode", "oneshotEnter",
    "oneshotLeave", "oneshotRefreshLabels", "oneshotSyncAnchor",
    "_setTakeToggle", "setTakeLightLock", "setTakeRetake",
    "takeLengthLabel", "oneshotSummary", "restoreFoldedLtxLength", "framesToDuration",
)

ONESHOT_SHIM = r"""
'use strict';
// A classList that REMEMBERS, because the assertions read it back.
function _cls() {
  const set = new Set();
  return {
    add(...a) { a.forEach(x => set.add(x)); },
    remove(...a) { a.forEach(x => set.delete(x)); },
    toggle(x, f) { const on = (f === undefined) ? !set.has(x) : !!f; on ? set.add(x) : set.delete(x); return on; },
    contains(x) { return set.has(x); },
    get list() { return [...set]; },
  };
}
const _els = {};
function _mk(id, props) {
  let _v = '';
  const e = Object.assign({
    id, textContent: '', placeholder: '', className: '', innerHTML: '',
    dataset: {}, hidden: false, style: {}, classList: _cls(), files: [],
    querySelector() { return null; }, querySelectorAll() { return []; },
    setAttribute(k, v) { this['_attr_' + k] = v; }, getAttribute(k) { return this['_attr_' + k] ?? null; },
    removeAttribute(k) { delete this['_attr_' + k]; if (k === 'src') this.src = ''; },
    appendChild() {}, remove() {}, addEventListener() {}, focus() {}, blur() {},
    dispatchEvent() { return true; },
  }, props || {});
  Object.defineProperty(e, 'value', {
    get() { return _v; },
    set(x) { _v = (x === null || x === undefined) ? '' : String(x); },
    enumerable: true, configurable: true,
  });
  if (props && Object.prototype.hasOwnProperty.call(props, 'value')) e.value = props.value;
  _els[id] = e;
  return e;
}
// The REAL chips: every data-mode in #modeGroup, every data-take in #takeGroup,
// the two toggle pairs — read out of index.html by the test and handed in.
const MODE_CHIPS = __MODE_CHIPS__.map((m, i) => _mk('_mode' + i, { dataset: { mode: m } }));
const TAKE_CHIPS = __TAKE_CHIPS__.map((t, i) => {
  const parts = _mk('_parts' + i);
  return _mk('_take' + i, { dataset: { take: String(t) }, querySelector: (sel) => sel === '.take-parts' ? parts : null });
});
const LIGHT_CHIPS = ['on', 'off'].map((v, i) => _mk('_light' + i, { dataset: { takeLight: v } }));
const RETAKE_CHIPS = ['on', 'off'].map((v, i) => _mk('_retake' + i, { dataset: { takeRetake: v } }));
global.document = {
  getElementById: (id) => _els[id] || _mk(id),
  querySelector: () => null,
  querySelectorAll: (sel) => ({
    '#modeGroup .pill-btn': MODE_CHIPS,
    '#takeGroup .pill-btn': TAKE_CHIPS,
    '#takeLightLockGroup .pill-btn': LIGHT_CHIPS,
    '#takeRetakeGroup .pill-btn': RETAKE_CHIPS,
  })[sel] || [],
  createElement: () => _mk('_tmp'),
  addEventListener: () => {},
  readyState: 'complete',
  body: { dataset: { engine: 'ltx' }, classList: _cls() },
};
global.window = global;
global.console = console;
const FPS = 24;
// The LTX length axis as BOOT ships it — key + frames is what the restore reads.
const BOOT = { ltx: { default_length: '5s', lengths: [
  { key: '3s', frames: 73 }, { key: '5s', frames: 121 }, { key: '10s', frames: 241 } ] } };
// The page's initial state, as the markup ships it.
_mk('mode', { value: 't2v' });
_mk('ltx_length', { value: '10s' });
_mk('frames', { value: '241' });
_mk('duration', { value: '10.00' });
_mk('take_seconds', { value: '0' });
_mk('take_light_lock', { value: 'on' });
_mk('take_retake', { value: 'on' });
_mk('takeAxes', { hidden: __PANEL_HIDDEN__ });
_mk('beatsRow', { hidden: true });
_mk('prompt', { value: 'She pushes off down the avenue. A van sweeps past. She drops off the kerb.' });
_mk('i2vMode', { value: 'i2v' });   // Image mode's audio-source select, as shipped
let currentMode = 't2v';
// setMode's collaborators. Stubs, not extractions: the property under test is
// the One Shot contract, not the paint.
const REMIX_MODES = ['ingredients', 'control', 'restore'];
let LAST_STATUS = null;
global.ingredientsServed = () => true;
global._portalLoraPicker = () => {};
global.renderLorasList = () => {};
global.updateAccelAvailability = () => {};
global.updateTemporalAvailability = () => {};
global.updateDerived = () => {};
global.updateCustomizeSummary = () => {};
global.updateModelsCard = () => {};
global._updateCharsPickerVisibility = () => {};
global._autoMainOutputsFilterForMode = () => {};
global._syncEngineForMode = () => {};
global.isKeyframeModeChipActive = () => false;
global.setQuality = () => {};
const refreshed = [];
global.takeRefresh = async () => { refreshed.push(document.getElementById('take_seconds').value); };
global.fetch = async () => ({ json: async () => ({ ok: true }) });
"""

ONESHOT_BODY = r"""
const $ = (id) => document.getElementById(id);
const active = (list, key) => list.filter(b => b.classList.contains('active')).map(b => b.dataset[key]);
const out = {};
out.start = { mode: $('mode').value, take: $('take_seconds').value, panelHidden: $('takeAxes').hidden };

setMode('oneshot');
out.enter = {
  currentMode, mode: $('mode').value, take: $('take_seconds').value,
  panelHidden: $('takeAxes').hidden, beatsRowHidden: $('beatsRow').hidden,
  bodyClass: document.body.classList.contains('oneshot-mode'),
  chips: active(MODE_CHIPS, 'mode'), takeChips: active(TAKE_CHIPS, 'take'),
  light: $('take_light_lock').value, retake: $('take_retake').value,
  lightChips: active(LIGHT_CHIPS, 'takeLight'), retakeChips: active(RETAKE_CHIPS, 'takeRetake'),
  parts: TAKE_CHIPS.map(b => b.querySelector('.take-parts').textContent),
  note: $('takeEngineNote').textContent,
  beats: $('beats').value, beatsHint: $('beatsHint').textContent,
  placeholder: $('prompt').placeholder,
  refreshed: refreshed.slice(),
};

// The switcher moves to H3: the parts are 15 s now, and the note says so.
document.body.dataset.engine = 'h3';
oneshotRefreshLabels();
out.h3 = { parts: TAKE_CHIPS.map(b => b.querySelector('.take-parts').textContent), note: $('takeEngineNote').textContent };
document.body.dataset.engine = 'ltx';
oneshotRefreshLabels();

// An anchor image flips the backend mode to i2v; clearing it flips back.
$('image').value = '/uploads/frame_one.png';
oneshotSyncAnchor();
out.anchored = { mode: $('mode').value, thumbHidden: $('oneshotAnchorThumb').hidden,
                 thumbSrc: $('oneshotAnchorThumb').src, clearHidden: $('oneshotAnchorClear').hidden,
                 name: $('oneshotAnchorName').textContent };
$('image').value = '';
oneshotSyncAnchor();
out.unanchored = { mode: $('mode').value, thumbHidden: $('oneshotAnchorThumb').hidden, clearHidden: $('oneshotAnchorClear').hidden };

// The toggles write the hidden fields make_job reads, as on/off.
setTakeLightLock('off'); setTakeRetake('off');
out.toggledOff = { light: $('take_light_lock').value, retake: $('take_retake').value,
                   lightChips: active(LIGHT_CHIPS, 'takeLight'), retakeChips: active(RETAKE_CHIPS, 'takeRetake') };
setTakeLightLock('garbage'); setTakeRetake(undefined);
out.toggledBack = { light: $('take_light_lock').value, retake: $('take_retake').value };

// A longer shot, and the beats button over a box that already has lines.
setTakeSeconds(120);
$('beats_text').value = 'only one line';
beatsInput();
out.len120 = { take: $('take_seconds').value, takeChips: active(TAKE_CHIPS, 'take'), beats: $('beats').value };
takePrefillClick();
out.prefilled = { lines: $('beats_text').value.split('\n').length, beats: $('beats').value };

// The footer strip's summary, per engine, from the same length.
out.summary = {
  ltx60: oneshotSummary(60, 'ltx'), h3_90: oneshotSummary(90, 'h3'),
  ltx30: oneshotSummary(30, 'ltx'), ltx120: oneshotSummary(120, 'ltx'), off: oneshotSummary(0, 'ltx'),
  labels: [30, 45, 60, 90, 120].map(takeLengthLabel),
};
// An engine round trip while the Length strip is folded: H3's tier wrote 73
// into #frames; the restore puts the folded LTX length (10s → 241) back.
$('frames').value = '73'; $('duration').value = '3.00';
out.restored = { ok: restoreFoldedLtxLength(), frames: $('frames').value, duration: $('duration').value };
// LEAVING the mode: any other setMode zeroes the take and folds the panel —
// and puts the two continuity fields back to on, so a normal clip's sidecar
// carries no One Shot noise. Both are left OFF here on purpose.
setTakeLightLock('off'); setTakeRetake('off');
setMode('t2v');
out.leave = {
  currentMode, mode: $('mode').value, take: $('take_seconds').value, beats: $('beats').value,
  panelHidden: $('takeAxes').hidden, bodyClass: document.body.classList.contains('oneshot-mode'),
  chips: active(MODE_CHIPS, 'mode'),
  light: $('take_light_lock').value, retake: $('take_retake').value,
  lightChips: active(LIGHT_CHIPS, 'takeLight'), retakeChips: active(RETAKE_CHIPS, 'takeRetake'),
};
// ...including the early-return modes (character returns before the generic
// path), which is where a hook placed too low would miss.
setMode('oneshot');
out.reenter = { take: $('take_seconds').value, panelHidden: $('takeAxes').hidden, takeChips: active(TAKE_CHIPS, 'take') };
setMode('character');
out.leaveViaCharacter = { currentMode, mode: $('mode').value, take: $('take_seconds').value, panelHidden: $('takeAxes').hidden };
// A One Shot with an anchor enters straight into i2v.
$('image').value = '/uploads/frame_one.png';
setMode('oneshot');
out.enterAnchored = { mode: $('mode').value, take: $('take_seconds').value };
setMode('i2v');
out.leaveToImage = { mode: $('mode').value, take: $('take_seconds').value, panelHidden: $('takeAxes').hidden };
process.stdout.write(JSON.stringify(out));
"""


def _oneshot_markup(src: str) -> dict:
    """The real chips out of the real markup, for the shim."""
    html = (ROOT / "webapp" / "index.html").read_text(encoding="utf-8")
    bar = html[html.index('id="modeGroup"'):]
    bar = bar[:bar.index("</div>")]
    modes = re.findall(r'data-mode="([^"]+)"', bar)
    group = html[html.index('id="takeGroup"'):]
    group = group[:group.index("</div>")]
    takes = [int(t) for t in re.findall(r'data-take="(\d+)"', group)]
    panel = extract_element("takeAxes", src)
    return {"modes": modes, "takes": takes, "panel_hidden": " hidden" in panel or panel.endswith("hidden>")}


def run_oneshot_contract() -> dict:
    if NODE is None:
        raise unittest.SkipTest("node not on PATH")
    source = panel_source()
    m = _oneshot_markup(source)
    shim = (ONESHOT_SHIM
            .replace("__MODE_CHIPS__", json.dumps(m["modes"]))
            .replace("__TAKE_CHIPS__", json.dumps(m["takes"]))
            .replace("__PANEL_HIDDEN__", "true" if m["panel_hidden"] else "false"))
    # The length table is a module const, not a function — read it as it is.
    choices = re.search(r"^const TAKE_CHOICES = \[[^\]]*\];", source, re.M)
    if not choices:
        raise AssertionError("TAKE_CHOICES not found in the panel source")
    # The remembered length is a module `let`, read as it is like the table.
    last = re.search(r"^let _oneshotLastSeconds = \d+;", source, re.M)
    if not last:
        raise AssertionError("_oneshotLastSeconds not found in the panel source")
    script = (shim + choices.group(0) + "\n" + last.group(0) + "\n"
              + "\n".join(extract_function(n, source) for n in ONESHOT_FUNCTIONS)
              + "\n" + ONESHOT_BODY)
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
        fh.write(script)
        path = Path(fh.name)
    try:
        result = subprocess.run([NODE, str(path)], capture_output=True,
                                text=True, timeout=60)
        if result.returncode:
            raise AssertionError(result.stdout + "\n" + result.stderr)
        return json.loads(result.stdout)
    finally:
        path.unlink(missing_ok=True)


class OneShotIsAMode(unittest.TestCase):
    """The mode chip, the panel, and the one property that matters most: a
    normal clip never carries take_seconds."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.src = panel_source()
        cls.html = (ROOT / "webapp" / "index.html").read_text(encoding="utf-8")
        cls.r = run_oneshot_contract()

    # ---- the markup ---------------------------------------------------------
    def test_the_chip_is_in_the_mode_bar_between_image_and_fflf(self):
        bar = self.html[self.html.index('id="modeGroup"'):]
        bar = bar[:bar.index("</div>")]
        modes = re.findall(r'data-mode="([^"]+)"', bar)
        self.assertIn("oneshot", modes)
        self.assertEqual(modes.index("oneshot"), modes.index("i2v") + 1)
        self.assertEqual(modes[modes.index("oneshot") + 1], "keyframe")
        chip = re.search(r'<button[^>]*data-mode="oneshot"[^>]*>(.*?)</button>', bar, re.S).group(1)
        self.assertTrue(chip.startswith("One Shot"))
        self.assertIn("never cuts", chip)

    def test_the_panel_ships_folded_and_keeps_its_ids(self):
        el = extract_element("takeAxes", self.src)
        self.assertIn("hidden", el)
        self.assertIn("oneshot-panel", el)
        panel = self.html[self.html.index('id="takeAxes"'):]
        panel = panel[:panel.index('id="takeEngineNote"')]
        for needed in ('id="takeGroup"', 'id="beatsRow"', 'id="beats_text"', 'id="beatsHint"',
                       'id="takeEstimate"', 'id="beatsPrefillBtn"', 'id="oneshotAnchorFile"',
                       'id="oneshotAnchorThumb"', 'id="oneshotAnchorClear"',
                       'id="takeLightLockGroup"', 'id="takeRetakeGroup"',
                       'href="/docs/prompting"'):
            self.assertIn(needed, panel, needed)
        # In its own mode there is no Off chip — leaving the mode is Off.
        takes = re.findall(r'data-take="(\d+)"', panel)
        self.assertEqual(takes, ["30", "45", "60", "90", "120"])

    def test_the_two_continuity_fields_are_in_the_video_form_and_default_on(self):
        form = self.html[self.html.index('id="genForm"'):]
        form = form[:form.index("</form>")]
        for name in ("take_light_lock", "take_retake"):
            el = extract_element(name, form)
            self.assertIn('type="hidden"', el)
            self.assertIn(f'name="{name}"', el)
            self.assertIn('value="on"', el)
        # ...and the fields they sit beside, so FormData posts the whole shot.
        for name in ("take_seconds", "beats", "image"):
            self.assertIn(f'name="{name}"', form)

    def test_the_old_name_is_gone_from_everything_a_user_reads(self):
        stray = []
        files = [ROOT / "webapp" / "index.html", ROOT / "docs" / "PROMPTING.md",
                 *sorted((ROOT / "webapp" / "js").glob("*.js"))]
        for f in files:
            for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                if re.search(r"\bone[ -]take\b", line, re.I):
                    stray.append(f"{f.name}:{i}: {line.strip()[:80]}")
        self.assertEqual(stray, [], "the feature is called One Shot")

    # ---- the executed contract ----------------------------------------------
    def test_a_fresh_page_carries_no_take(self):
        self.assertEqual(self.r["start"], {"mode": "t2v", "take": "0", "panelHidden": True})

    def test_entering_the_mode_opens_the_panel_on_t2v_with_a_length(self):
        e = self.r["enter"]
        self.assertEqual(e["currentMode"], "oneshot")
        self.assertEqual(e["mode"], "t2v")
        self.assertEqual(e["take"], "60")
        self.assertFalse(e["panelHidden"])
        self.assertFalse(e["beatsRowHidden"])
        self.assertTrue(e["bodyClass"])
        self.assertEqual(e["chips"], ["oneshot"])
        self.assertEqual(e["takeChips"], ["60"])
        self.assertIn("never cuts", e["placeholder"])
        self.assertEqual(e["refreshed"], ["60"], "the estimate is asked for on entry")

    def test_the_beats_prefill_from_the_prompt_on_entry(self):
        e = self.r["enter"]
        self.assertEqual(json.loads(e["beats"])[:3],
                         ["She pushes off down the avenue.", "A van sweeps past.", "She drops off the kerb."])
        self.assertIn("12 lines of 5 s", e["beatsHint"])
        self.assertIn("3 written", e["beatsHint"])
        self.assertIn("leave a line blank to hold on the scene", e["beatsHint"])
        self.assertNotIn("beats of", e["beatsHint"])

    def test_the_toggles_default_on_and_write_on_off(self):
        e = self.r["enter"]
        self.assertEqual((e["light"], e["retake"]), ("on", "on"))
        self.assertEqual((e["lightChips"], e["retakeChips"]), (["on"], ["on"]))
        t = self.r["toggledOff"]
        self.assertEqual((t["light"], t["retake"]), ("off", "off"))
        self.assertEqual((t["lightChips"], t["retakeChips"]), (["off"], ["off"]))
        self.assertEqual(self.r["toggledBack"], {"light": "on", "retake": "on"})

    def test_parts_are_10s_on_ltx_and_15s_on_h3(self):
        # One line per chip — "3 × 10 s", not "3 PARTS OF 10 S" wrapping to a
        # 95px chip — so the strip keeps the Quality strip's height.
        self.assertEqual(self.r["enter"]["parts"],
                         ["3 × 10 s", "5 × 10 s", "6 × 10 s", "9 × 10 s", "12 × 10 s"])
        self.assertEqual(self.r["enter"]["note"], "LTX — 10-second parts that continue from the last frame.")
        self.assertEqual(self.r["h3"]["parts"],
                         ["2 × 15 s", "3 × 15 s", "4 × 15 s", "6 × 15 s", "8 × 15 s"])
        self.assertEqual(self.r["h3"]["note"], "Hailuo H3 — 15-second parts that continue from each other.")
        self.assertNotIn("proven", self.r["h3"]["note"])

    def test_the_footer_strip_says_the_shot_not_a_five_second_clip(self):
        s = self.r["summary"]
        self.assertEqual(s["ltx60"], "1 min · 6 parts of 10 s")
        self.assertEqual(s["h3_90"], "1½ min · 6 parts of 15 s")
        self.assertEqual(s["ltx30"], "30 s · 3 parts of 10 s")
        self.assertEqual(s["ltx120"], "2 min · 12 parts of 10 s")
        self.assertEqual(s["off"], "", "no shot, no summary — the clip line stays")
        self.assertEqual(s["labels"], ["30 s", "45 s", "1 min", "1½ min", "2 min"])
        fn = extract_function("updateDerived", self.src)
        self.assertIn("oneshotSummary", fn)
        self.assertIn("take_seconds", fn)
        self.assertLess(fn.index("derivedFooter"), fn.index("oneshotSummary"),
                        "the summary is the FOOTER strip's line")

    def test_an_engine_round_trip_leaves_the_folded_ltx_length_alone(self):
        r = self.r["restored"]
        self.assertTrue(r["ok"])
        self.assertEqual((r["frames"], r["duration"]), ("241", "10.00"))
        fn = extract_function("setEngine", self.src)
        self.assertIn("restoreFoldedLtxLength", fn)
        # The snap from H3's 17n+5 grid is what wrote "3s" onto the folded
        # strip; in One Shot the restore runs INSTEAD of it, not after it.
        i = fn.index("currentMode === 'oneshot'")
        self.assertLess(i, fn.index("snapFramesTo8kPlus1();"))
        self.assertIn("setTakeSeconds", fn, "the mode's own state is re-asserted after the swap")

    def test_an_anchor_image_makes_it_i2v_and_clearing_it_makes_it_t2v(self):
        a = self.r["anchored"]
        self.assertEqual(a["mode"], "i2v")
        self.assertFalse(a["thumbHidden"])
        self.assertIn("frame_one.png", a["thumbSrc"])
        self.assertFalse(a["clearHidden"])
        self.assertEqual(a["name"], "frame_one.png")
        self.assertEqual(self.r["unanchored"], {"mode": "t2v", "thumbHidden": True, "clearHidden": True})
        self.assertEqual(self.r["enterAnchored"], {"mode": "i2v", "take": "120"},
                         "the remembered length, not the default")

    def test_a_length_chip_and_the_write_the_beats_button(self):
        self.assertEqual(self.r["len120"]["take"], "120")
        self.assertEqual(self.r["len120"]["takeChips"], ["120"])
        self.assertEqual(json.loads(self.r["len120"]["beats"]), ["only one line"])
        self.assertEqual(self.r["prefilled"]["lines"], 3)

    def test_leaving_the_mode_zeroes_the_take_and_folds_the_panel(self):
        for key in ("leave", "leaveViaCharacter", "leaveToImage"):
            l = self.r[key]
            self.assertEqual(l["take"], "0", key)
            self.assertTrue(l["panelHidden"], key)
        l = self.r["leave"]
        self.assertEqual(l["currentMode"], "t2v")
        self.assertEqual(l["mode"], "t2v")
        self.assertEqual(l["beats"], "")
        self.assertFalse(l["bodyClass"])
        self.assertEqual(l["chips"], ["t2v"])
        self.assertEqual(self.r["leaveViaCharacter"]["currentMode"], "character")
        self.assertEqual(self.r["leaveToImage"]["mode"], "i2v")

    def test_leaving_the_mode_puts_the_continuity_fields_back_to_on(self):
        # Both were switched OFF before leaving. They are hidden inputs in the
        # video form, so FormData posts them with every clip — a normal clip
        # must not carry an "off" from a mode it is not in.
        l = self.r["leave"]
        self.assertEqual((l["light"], l["retake"]), ("on", "on"))
        self.assertEqual((l["lightChips"], l["retakeChips"]), (["on"], ["on"]))

    def test_the_length_is_remembered_across_leaving_and_re_entering(self):
        # 120 was chosen, the mode was left (take → 0) and re-entered: the
        # panel reopens on 2 min, not on the 1 min default.
        self.assertEqual(self.r["reenter"], {"take": "120", "panelHidden": False, "takeChips": ["120"]})

    # ---- the copy -----------------------------------------------------------
    def _panel_text(self):
        panel = self.html[self.html.index('id="takeAxes"'):]
        panel = panel[:panel.index('id="takeEngineNote"')]
        panel = re.sub(r"<!--.*?-->", " ", panel, flags=re.S)
        return re.sub(r"<[^>]+>", " ", panel)

    def test_the_word_take_is_not_in_anything_a_user_reads(self):
        text = self._panel_text()
        self.assertFalse(re.search(r"\b(re)?takes?\b", text, re.I), text)
        for fn in ("beatsInput", "oneshotRefreshLabels", "takeRefresh", "oneshotSummary", "takeLengthLabel"):
            body = extract_function(fn, self.src)
            strings = re.findall(r"""(['"`])((?:(?!\1).)*)\1""", body)
            for _, lit in strings:
                if lit[:1] in "./#":      # a selector, an id or a URL — not copy
                    continue
                self.assertFalse(re.search(r"\b(re)?takes?\b", lit, re.I), f"{fn}: {lit}")

    def test_the_panel_copy(self):
        text = " ".join(self._panel_text().split())
        for needed in ("Start frame", "optional · the shot starts from this picture",
                       "Split my prompt into beats",
                       "one line per 5 seconds — what happens in that moment",
                       "Lock the light", "keeps the time of day and weather the same in every line",
                       "Redo a part that drifts", "if a part changes the light, it is rendered once more"):
            self.assertIn(needed, text, needed)
        for gone in ("Anchor image", "Write the beats for me", "Retake", "holds the moment"):
            self.assertNotIn(gone, text, gone)
        # The split button lives in the Beats label row, not on a row of its own.
        row = self.html[self.html.index('id="beatsRow"'):self.html.index('id="beats_text"')]
        self.assertIn('id="beatsPrefillBtn"', row)
        self.assertNotIn("beats-tools", self.html)
        # The hint counter and the overflow line, executed.
        self.assertIn("extra line", extract_function("beatsInput", self.src))
        self.assertIn("will be dropped", extract_function("beatsInput", self.src))

    def test_the_storyboard_names_it_one_shot(self):
        chip = re.search(r'<button[^>]*data-sb-shots="take"[^>]*>(.*?)</button>', self.html, re.S).group(1)
        self.assertIn(">One Shot<", chip)
        row = self.html[self.html.index('id="sbTakeRow"'):self.html.index('id="sbTakeGroup"')]
        self.assertIn("One Shot length", row)
        self.assertNotIn("How long", row)
        # ...and the row really folds: .cz-control sets a display of its own.
        css = (ROOT / "webapp" / "style" / "panel.css").read_text(encoding="utf-8")
        self.assertRegex(css, r"\.cz-control\[hidden\]\s*\{\s*display:\s*none\s*!important;?\s*\}")

    def test_the_leave_hook_runs_before_every_early_return(self):
        fn = extract_function("setMode", self.src)
        self.assertLess(fn.index("oneshotLeave"), fn.index("if (mode === 'train')"))
        self.assertLess(fn.index("oneshotLeave"), fn.index("if (mode === 'character')"))
        self.assertLess(fn.index("oneshotLeave"), fn.index("if (mode === 'image')"))

    def test_load_params_reopens_a_one_shot_from_its_sidecar(self):
        fn = extract_function("loadParams", self.src)
        self.assertIn("setMode('oneshot')", fn)
        self.assertLess(fn.index("setMode('oneshot')"), fn.index("setMode('extend')"))
        self.assertIn("setTakeLightLock", fn)
        self.assertIn("setTakeRetake", fn)

    def test_the_engine_switch_refreshes_the_part_labels(self):
        fn = extract_function("setEngine", self.src)
        self.assertIn("oneshotRefreshLabels", fn)
        self.assertIn("takeRefresh", fn)

    def test_both_engines_serve_the_mode(self):
        fn = extract_function("engineServesMode", self.src)
        self.assertIn("mode === 'oneshot'", fn)


if __name__ == "__main__":
    unittest.main(verbosity=2)
