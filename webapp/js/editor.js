// webapp/js/editor.js — extracted verbatim from the panel page's inline
// script block (slice 3 of docs/ARCHITECTURE.md). ES module: top-level
// declarations are module-private; the publish block at the bottom is
// the module's public surface.
// ===========================================================================
// THE TIMELINE — the client half of /storyboard/edit
// ===========================================================================
//
// The shot list is INTENT: what the film should contain, one card per thing to
// render. This is ARRANGEMENT: what plays, from which second of which file, at
// which second of the film. Two documents on disk for the reason
// storyboard_editor.py gives — the moment one shot is used twice, or one is
// split in half, they are different shapes forever.
//
// THREE MEASURED FACTS SHAPE EVERY DECISION BELOW.
//
//  1. A source clip is a single GOP: 235 ms median to seek in Chrome, 1266 ms
//     p90. The all-intra proxy seeks in 3.5 ms. So the preview NEVER points at
//     clip.path when clip.proxy exists, and the one place it may (no proxy yet)
//     says so on the badge instead of pretending.
//
//  2. Two <video> elements measured NO benefit over one once proxies are in
//     play — the swap costs what the seek costs, and two decoders is two
//     decoders. There is one video element here, deliberately.
//
//  3. The preview freezes 50-67 ms at each cut (1.2-1.6 frames at 24 fps) and
//     Safari's seeking is accurate to 3-6 frames. That is stated once, quietly,
//     under the transport, and never papered over: the render is exact, the
//     preview is an approximation of it.
//
// Two traps this feature already paid for, both load-bearing: `muted` is set
// BEFORE the first play() or Chrome silently refuses autoplay, and elements are
// hidden with opacity, never display:none, or WebKit will not load them.

// ---- HOW TALL THE TIMELINE IS, AND WHERE THE HEIGHT GOES ----------------
// Declared HERE rather than with the other SBE_* constants below, because
// SBE's own initialiser reads the stored preference through them and a `const`
// three hundred lines further down is a temporal dead zone, not a value.
//
// The timeline box is a ruler, four lanes, a scrollbar gutter and two borders.
// Only the lanes grow.
const SBE_TL_CHROME = 32;        // 18 ruler + 12 scrollbar gutter + 2 border
// THE LANES, in the order they are on screen. `base` is the floor each one has
// always had, `cap` is the point past which more height stops buying anything,
// and `share` is what fraction of every dragged pixel it takes.
//
// DISTRIBUTED BY SHARE, NOT PROPORTIONAL TO WHAT IS THERE. Proportional
// scaling would hand the picture track 41% of the drag (78 of the 188px of
// lane) and the sound strip 14% — the exact opposite of the reason the handle
// was asked for. 70% of every pixel goes to the two SOUND lanes instead, so
// 100px of drag turns a 26px strip into ~68px: the level line's usable band
// more than triples and a keyframe becomes a target rather than a rumour. The
// ruler takes none of it — a ruler is not more legible for being taller.
//
// REBALANCED 2026-08-20, after the owner used it: "the squares for the video
// design, like the video in the timeline, are expanded too much; they are too
// big. They don't need to be this big." A picture block is a LABEL WITH A
// THUMBNAIL HINT — a name, a timecode, and enough of the frame behind them to
// tell two shots apart. It is not a poster: the frame it stands for is already
// on the program monitor, full size, two inches above. So the picture lane's
// base drops 78 -> 64 (a 52px block: name row, meta row, and a band of picture)
// and its ceiling 240 -> 120, which is the tallest a block can be and still
// read as a block. Both ends were checked on screen.
//
// The height it gave up went to the sound, and so does the drag: the picture's
// share falls 0.22 -> 0.14 and the two sound lanes rise to 0.78 of every
// dragged pixel. The bases rise too — 26 -> 44 on the clip strip and 54 -> 72
// on the soundtrack — because the level line has to be legible AT REST, not
// only after somebody has discovered the handle and pulled it.
const SBE_LANES = [
  { key: 'ov',    base: 32, cap:  56, share: 0.08 },   // overlay: a card is placed, not performed
  { key: 'track', base: 64, cap: 120, share: 0.14 },   // the picture: a labelled block, not a poster
  { key: 'alane', base: 44, cap: 190, share: 0.44 },   // per-clip sound: waveform, level line, points
  // The soundtrack. Its base rose 72 -> 108 when the bed stopped being a
  // rectangle you can only slide: it now carries a waveform, a level line,
  // draggable points and two corner fade handles, exactly like the strip lane
  // above it — and that lane's base rose 26 -> 44 for the same reason, which
  // is that a control has to be legible AT REST and not only after somebody
  // has discovered it. The HEAD grew with it (the level and the duck now live
  // there), and head and lane read the same variable on purpose.
  { key: 'wave',  base: 108, cap: 240, share: 0.34 },  // the soundtrack
];
// Both ends are the SUM of the parts, never a hand-picked round number — the
// floor is every lane at its base (which is what the box needs to draw itself
// without clipping) and the ceiling is every lane at its cap (past which the
// drag would be buying dead band). `test_the_height_floor_is_the_sum_of_its
// _lanes` asserts exactly that, because the 190 this replaced was itemised in
// a comment and the overlay lane was added without moving it: the box has
// been 30px short of its own contents, with `overflow-y: hidden` over the
// difference, since that lane shipped.
globalThis.SBE_TL_MIN_H = 280;
globalThis.SBE_TL_MAX_H = 638;
// How far one arrow key moves the edge, and one arrow key with Shift.
const SBE_TL_STEP = 12;
const SBE_TL_STEP_BIG = 40;

// ---- THE CORNER HANDLES' FOOTPRINT, as the SVG has to know it -----------
// These MIRROR `--sbe-grip-w`, `--sbe-grip-skirt` and `--sbe-fade-hit` in the
// stylesheet, and they are here because the level line is drawn, not styled:
// its target is a path this file emits, so the only way to keep that path off
// the fade handles' pixels is to know in JS where those pixels are.
//
// A MIRROR IS A THING THAT DRIFTS, so the browser is what catches it, not a
// grep: `scripts/measure_editor_layout.py` measures the three rectangles in a
// laid-out page and fails if any two of them intersect. If somebody changes
// the CSS and forgets these, the gate goes red with both numbers in it.
const SBE_AGRIP_W = 7;      // a strip's grip is thinner than a block's
const SBE_GRIP_SKIRT = 3;   // ...and can be hit this far past what it draws
const SBE_FADE_HIT = 22;    // the fade handle's square target
// The first x on a strip that no corner handle stands on, and the last.
// THE +1 IS THE SVG'S OWN ROUNDING. The strip's <svg> carries an integer
// `width` attribute over a viewBox of the same integer, but the element is
// laid out at the strip's REAL fractional width — so one user unit is
// px/round(px) CSS pixels and the clip lands up to half a pixel off. Measured
// without it: the target's left edge came back at 471.84 against a handle
// ending at 472.00, which is a 0.16px overlap and a red gate. A pixel of
// margin costs nothing and is not a number anybody has to maintain.
const SBE_LVL_CLEAR = SBE_AGRIP_W + SBE_GRIP_SKIRT + SBE_FADE_HIT + 1;
// Below this there is no line left to aim at, so none is offered.
const SBE_LVL_MIN_SPAN = 12;

// ---- THE LEVEL LINE'S GEOMETRY, in ONE place ----------------------------
// "I still don't understand how to add keyframes, to be honest." — said after
// using it, which makes it a discoverability defect and not a user one. The
// gesture was double-click-to-add, drag-to-set, shift-click-to-remove: three
// things knowable only by being told.
//
// Everything that reads or writes a level now goes through this pair, because
// three gestures were each carrying their own copy of the arithmetic and two
// of them had already drifted onto a hard-coded 20px band. `sbeStripY` and
// `sbeStripGain` are exact inverses at every strip height — there is a test
// that round-trips them at five of them.
// HEADROOM, AND IT IS THE WHOLE OF WHY THE LINE IS DRAGGABLE. "You are able to
// drag the sound thingy, but it's not super user-friendly. Maybe you should put
// the orange line a little lower so it feels more draggable." At 1px, unity was
// drawn ON the strip's own top edge: the target is a 14px stroke centred on the
// line, so six of its seven upper pixels fell outside the strip and were
// clipped away by the SVG viewport, leaving a one-sided sliver to aim at that
// also competed with the lane's border.
//
// EIGHT IS DERIVED, NOT PICKED: half of `.sbe-lvl-hit`'s 14px stroke, plus one,
// so the entire target lies inside the strip at every height the share table
// can produce — 37px at the floor included. Unity still means "the top of the
// scale", which was built deliberately and is not what he asked to change; it
// simply no longer means "the top pixel of the box".
globalThis.SBE_LVL_PAD = 8;      // the line never touches the strip's own edge
// How near the pointer has to be for the LINE to answer rather than the strip
// under it. Twelve pixels is the same order as the dot's own target, and it is
// what makes "click the line" a gesture instead of a bullseye.
const SBE_LVL_GRAB = 12;
function sbeStripY(gain, H) {
  const h = Math.max(2 * SBE_LVL_PAD + 1, sbeNum(H));
  return (1 - Math.max(0, Math.min(1, sbeNum(gain)))) * (h - 2 * SBE_LVL_PAD)
         + SBE_LVL_PAD;
}
function sbeStripGain(y, top, H) {
  const h = Math.max(2 * SBE_LVL_PAD + 1, sbeNum(H));
  return Math.max(0, Math.min(1,
    1 - (sbeNum(y) - sbeNum(top) - SBE_LVL_PAD) / (h - 2 * SBE_LVL_PAD)));
}

window.SBE = {
  open: false, id: '', title: '',
  edit: null, clips: [], audio: null, beats: null,
  peaks: null, peaksFor: '', unplaced: [], pool: [], relink: [], prepare: {},
  onMissing: null,
  proxyUrl: '', revision: 0, dirty: false, saving: false, conflict: 0,
  // THE SAVE'S OWN BOOKKEEPING. `savePending` is the save that arrived while
  // another was in flight and used to be thrown away; `dirtyAt` is when the
  // oldest unwritten change was made, which is what lets the tick notice that
  // nothing has reached the disk for too long; `saveFailed` is the reason the
  // alarm is up, and it is a string precisely so it can be shown.
  savePending: false, dirtyAt: 0, saveFailed: '',
  // The quiet lane's own two facts, and the drafts this film has.
  backingUp: false, backedUpAt: 0,
  drafts: [], activeDraft: '', backup: null,
  // Which row the name field is currently editing, '' when it is making a new
  // draft. One field with two verbs and no mode is a field that picks the
  // wrong one every time Enter is pressed.
  renaming: '',
  sel: '', playhead: 0, playing: false, curId: '', pps: 42,
  undo: [], redo: [], errors: {}, sentOrder: [],
  // THE ONE NOTICE SURFACE. `noticeLead` is the chip the user clicked open,
  // `backupHidden` is "Later" on the recovery offer (this session, this
  // offer), `errsOpen` is the validation list unfolded past its first line.
  noticeLead: '', backupHidden: false, errsOpen: false,
  // THE OVERLAY LANE — a second video track, above the picture. Its own list
  // for the same reason the server keeps one: the picture lane may not
  // overlap itself, and an overlay's whole job is to sit ON a picture.
  overlays: [], ovSel: '', ovDrag: null,
  // The last card whose black backdrop was keyed on the way in: which overlay
  // it became, and the untouched file it came from. This is what "Keep
  // original" undoes, and it is null whenever that receipt is not on screen.
  keyed: null,
  // One waveform per SOURCE, not per clip: the same take used twice draws the
  // same picture and each strip slices its own window out of it.
  clipPeaks: null, kfDrag: null,
  // THIS TAB'S IDENTITY, for the life of the tab. Whoever loaded last owns
  // the snapshot lane; a tab that finds another editor is TOLD, and goes on
  // read-only rather than writing its stale state over the live one every
  // debounce. See docs/EDITOR_SAVE_MODEL.md §5.
  session: 'ss' + Math.random().toString(16).slice(2, 10) +
           (Date.now() % 1048576).toString(16),
  // WHO ELSE IS EDITING, and when this tab was last actually protected.
  // There is deliberately no flag here that can stop the writer: the one that
  // used to exist (`superseded`) was set by a passive page load in another
  // browser and switched this tab's safety net off for seven hours.
  otherEditor: '', protectedAt: 0,
  timer: null, saveTimer: null, raf: 0, drag: null, awaitingClip: 0,
  // The soundtrack's own drag. Separate from `drag` because the two lanes are
  // separate objects and a pointer is only ever on one of them.
  musicDrag: null, audioDrag: null,
  music: '', musicEl: null, musicOk: true, rendering: false, muteToasted: false,
  // `dropAt` is where the insertion line is painted while something is being
  // dragged onto the track — null when nothing is. `lastTs` is the wall clock
  // the transport falls back on for the clips that carry no <video> to read a
  // currentTime off (a still, a slug, a hole).
  dropAt: null, lastTs: 0,
  // THE SOURCE MONITOR. `source` is the pool row on the left screen (null
  // when nothing has been clicked yet) and `srcIndex` is where it sat in the
  // painted pool, which is what "Add to timeline" hands back to edPoolAdd so
  // the two doors take exactly the same path. Never both monitors at once:
  // sbeSrcPlay stops the program and sbePlay stops the source.
  source: null, srcIndex: -1, srcPlaying: false,
  // Sound is ON. It survives a reload because a muted editor is indistinguishable
  // from a broken one, and that is the bug this default is paying off.
  muted: (localStorage.getItem('sbeMuted') === '1'),
  // HOW TALL THE TIMELINE IS, and who decides. `tlH` is the user's preference
  // — restored from this browser, never from the document — `tlNow` is what
  // the layout could actually give it once the window had its say, and
  // `tlMax` is what the window has left before the monitors stop being
  // monitors. The handle drags `tlH`; sbeFitMonitors writes the other two.
  // `laneAt` is the height the two SOUND lanes were last DRAWN at, which is
  // not always the height they are currently given — see the end of
  // sbeFitMonitors for the one pass where those two disagree.
  tlH: sbeTlPrefRead(), tlNow: 0, tlMax: 0, tlDrag: null, laneAt: -1,
  // THE GHOST POINT. Where a click on the level line would put one — null
  // whenever the pointer is not near a line it could edit. It is the whole of
  // "hover teaches": the control answers before it is used.
  kfGhost: null,
};

const SBE_MIN_CLIP = 0.2;        // shorter than this is not a shot, it is a blink
const SBE_UNDO_MAX = 80;
const SBE_SNAP_PX = 9;           // how near the pointer must be to catch a beat
const SBE_GUESS_CONFIDENCE = 0.4; // below this the grid is drawn as a guess
// The same clamp `storyboard_editor.BRIGHTNESS_LIMIT` enforces on the way in.
// Half of ffmpeg's additive range is already past "unusable" in both
// directions, and a slider that can ask for a value the server will refuse is
// a slider that produces a red error box instead of a picture.
const SBE_BRIGHT_MAX = 0.5;
// How long a still holds when it lands. Long enough to read a title card,
// short enough that nobody has to trim it DOWN before the film is watchable —
// and it is the trim handles' job from there.
const SBE_STILL_SECONDS = 3.0;
// ---- the two sliders, and the two monitors ------------------------------
// sbePaint pads the inner by this much past the last frame, so "the whole
// film fits" means (viewport - SBE_TL_PAD) / span, not viewport / span.
const SBE_TL_PAD = 24;
// THE TIMELINE DOES NOT END WHERE THE CLIPS END. "The film's length and the
// audio's placement are independent facts and the UI must treat them that
// way" — so the scroller runs past the last frame by this much, and the music
// can be dragged out there instead of stopping dead against the final cut.
// Proportional with both ends pinned: 3 s is enough room to grab an edge on a
// short cut, 15 s stops a ten-minute film from opening on two minutes of
// nothing.
const SBE_SLACK_MIN = 3;
const SBE_SLACK_MAX = 15;
const SBE_SLACK_RATIO = 0.15;
// An EMPTY timeline is still a timeline: a ruler you can read and a lane the
// music can be dropped into. Without a floor the span is one second, the
// ruler is a single tick, and the first thing the Editor shows a new film is
// a box that looks broken.
const SBE_SPAN_MIN = 10;
// The shortest the soundtrack can be trimmed to. Same reason SBE_MIN_CLIP
// exists: a zero-length window is one ffmpeg refuses and one no pointer can
// grab back.
const SBE_MIN_MUSIC = 0.5;
// HALF A FRAME AT 24 fps, and the same number the server's TOUCH_TOLERANCE
// uses to decide whether two clips touch. Below it a sound is in sync: every
// window on the timeline is rounded to a microsecond, so an exact-zero test
// would flag a strip that is one float away from where it started.
const SBE_SYNC_TOL = 1 / 48;
// The timeline's floor, its ceiling and the lanes between them are declared
// with SBE itself, up where its initialiser can read them.
const SBE_PPS_MAX = 200;         // the old zoom ladder's top step
const SBE_PPS_FLOOR = 0.5;       // below this a cut is thinner than a pixel
const SBE_ZOOM_TICKS = 1000;     // the range input's resolution
// The gap between the monitors, and the smallest picture worth calling one.
const SBE_MON_GAP = 12;
const SBE_MON_MIN_H = 120;
// Source height / program height. 2/3 is the 40/60 split by width that was
// asked for; sbeMonitorFit widens it toward 1 (equal monitors) when the
// window is too short to fill the row at 40/60, because a symmetric black
// gutter either side of the pair is the thing this whole pass is removing.
const SBE_MON_RATIO = 2 / 3;
const SBE_MON_RATIO_MAX = 1;
// The inspector rail beside the monitors. Narrower than this it is a slot,
// wider than this it is a second sidebar — and either way it is the width the
// monitors could not use, not a width of its own choosing.
const SBE_RAIL_MIN = 200;
const SBE_RAIL_MAX = 380;

function sbeEl(id) { return document.getElementById(id); }
function sbeNum(v, d) { const n = Number(v); return (n === n && isFinite(n)) ? n : (d || 0); }
function sbeRound(v) { return Math.round(sbeNum(v) * 1e6) / 1e6; }
function sbeFps() { return (typeof FPS === 'number' && FPS > 0) ? FPS : 24; }

// ---------------------------------------------------------------------------
// THE MODEL. Everything in this section is pure: arrays in, arrays out, no DOM,
// no fetch. That is what makes the drag maths, the snap, the ripple and the
// save payload testable in node rather than by eye — see
// test_storyboard_editor_ui.py, which runs these exact functions.
// ---------------------------------------------------------------------------
// SPEED, ON THE CLIP. The mirror of `clip_speed()`: video only, clamped to
// the same range, 1x when absent. Every length on the film goes through
// `sbeLen`, so the one division below is the whole of what retiming does to
// the layout — a 2x clip owns half the slot its window used to.
const SBE_SPEED_MIN = 0.25;
const SBE_SPEED_MAX = 4.0;
function sbeSpeed(c) {
  if (!c || sbeKind(c) !== 'video') return 1;
  const s = sbeNum(c.speed, 1);
  if (!(s > 0)) return 1;
  return sbeRound(Math.max(SBE_SPEED_MIN, Math.min(SBE_SPEED_MAX, s)));
}
function sbeLen(c) {
  return Math.max(SBE_MIN_CLIP, (sbeNum(c.end) - sbeNum(c.start)) / sbeSpeed(c));
}

// Each clip owns the gap that PRECEDES it. That single choice is what makes
// every operation below a one-liner: a move is "change which gap you own", a
// ripple delete is "take your gap with you", and the film position of every
// clip is a running total that cannot drift from its own source window.
// A LEAD GAP LIVES ON THE FRAME GRID. Under one frame is NO gap.
//
// The owner's report was "a black frame that flashes for a microsecond... I
// tried to drag them close and whatever" — and the second half is the defect.
// The holes were 0.503, 0.380 and 0.096 of a frame: too small to see at any
// zoom, too small for any pixel to address, and the stage paints black
// wherever no clip is playing. Rounding to the NEAREST frame would have turned
// the 0.503 into a whole frame of black, which is the same bug one frame
// louder; anything under a frame is float noise from a drag and a JSON round
// trip, and zero is its only honest reading. See the block above `film_fps` in
// storyboard_editor.py for why the grid is enforced on gaps and not on
// absolute film positions.
function sbeGridGap(gap) {
  const f = 1 / sbeFps();
  const g = Math.max(0, sbeNum(gap));
  // A LARGER GAP IS LEFT EXACTLY AS AUTHORED. Snapping a 43.2-frame slug to
  // 43 frames would rewrite a number the user chose to buy a property nothing
  // reads: the cuts either side of a gap are already exact once the gap is a
  // fixed quantity. See `quantise_gap` in storyboard_editor.py.
  return (g < f - 1e-9) ? 0 : sbeRound(g);
}

function sbeAdoptGaps(clips) {
  let cursor = 0;
  // A HEAL CARRIES THE SOUND; A GESTURE DOES NOT. Closing a sub-frame hole is
  // not the user sliding a picture, so an unlinked strip travels with it by
  // the same delta and the J-cut's offset is exactly what it was. Dragging is
  // the opposite by design — the picture moves and the strip stays, which is
  // how a J-cut is made in the first place — so `sbeLayout` never does this.
  for (const c of clips) {
    const raw = Math.max(0, sbeNum(c.film_start) - cursor);
    const gap = sbeGridGap(raw);
    const shift = raw - gap;
    if (shift > 1e-9 && c.audio && c.audio.film_start !== undefined
        && c.audio.film_start !== null) {
      c.audio = Object.assign({}, c.audio,
                              { film_start: sbeRound(sbeNum(c.audio.film_start) - shift) });
    }
    c._gap = gap;
    if (c.locked) c._pin = sbeNum(c.film_start);
    cursor = sbeNum(c.film_end) || (sbeNum(c.film_start) + sbeLen(c));
  }
  return clips;
}

// Re-derive film_start / film_end from the lead gaps. film_end is NEVER an
// independent number — nothing plays at anything but 1x, and the server refuses
// an edit whose slot and window disagree by more than a millisecond.
function sbeLayout(clips) {
  const locks = [];
  for (const c of clips) {
    if (c.locked) {
      const s = (c._pin === undefined || c._pin === null) ? sbeNum(c.film_start) : sbeNum(c._pin);
      locks.push([s, s + sbeLen(c)]);
    }
  }
  locks.sort((a, b) => a[0] - b[0]);
  let cursor = 0;
  for (const c of clips) {
    const len = sbeLen(c);
    if (c.locked) {
      const s = (c._pin === undefined || c._pin === null) ? sbeNum(c.film_start) : sbeNum(c._pin);
      c._pin = s;
      c.film_start = sbeRound(s);
      c.film_end = sbeRound(s + len);
      cursor = Math.max(cursor, c.film_end);
      continue;
    }
    // ON THE GRID AT THE LAST POSSIBLE MOMENT. Adopting quantises what came
    // off disk; this quantises what a DRAG just produced, so a gesture cannot
    // reopen the hole a load closed. Both call the same function, so there is
    // one definition of "a gap" and not two that agree today.
    let s = cursor + sbeGridGap(c._gap);
    // A locked clip is an anchor, so a free one flows AROUND it rather than
    // through it. Bounded restart: every push moves s strictly forward.
    for (let pass = 0; pass < locks.length + 1; pass++) {
      let moved = false;
      for (const L of locks) {
        if (s < L[1] - 1e-9 && s + len > L[0] + 1e-9) { s = L[1]; moved = true; }
      }
      if (!moved) break;
    }
    c.film_start = sbeRound(s);
    c.film_end = sbeRound(s + len);
    cursor = c.film_end;
  }
  clips.sort((a, b) => sbeNum(a.film_start) - sbeNum(b.film_start));
  return clips;
}

// ONE LANGUAGE ON THE TRACK. Adjacent blocks were labelled in two: one said
// what the shot IS, the next was a model filename complete with resolution,
// an extension tag and a timestamp — "bizarrotrn_the_man_in_the_loud_4_
// dn768x416_ext6_20260818_215…" — and at fit zoom both truncate, so the track
// read as a row of half-words. The exact filename stays on the block's title.
function sbeNiceName(s) {
  let t = String(s || '').trim();
  if (!t) return '';
  if (t.indexOf('/') >= 0) t = t.split('/').pop();
  t = t.replace(/\.[a-z0-9]{2,5}$/i, '');
  t = t.replace(/_\d{8}_\d{6}$/, '')
       .replace(/_dn\d+x\d+/gi, '')
       .replace(/_ext\d+/gi, '')
       .replace(/_\d+p\b/gi, '')
       .replace(/_seed_\d+$/i, '')
       .replace(/_\d+$/, '');
  t = t.replace(/[_-]+/g, ' ').replace(/\s+/g, ' ').trim();
  if (!t) return String(s || '');
  return t.charAt(0).toUpperCase() + t.slice(1);
}

function sbeFilmDuration(clips) {
  let m = 0;
  for (const c of clips || []) m = Math.max(m, sbeNum(c.film_end));
  return sbeRound(m);
}

// THE SOUNDTRACK IS AN OBJECT, and this is its geometry. The mirror of
// `music_window()` in storyboard_editor.py — the same three fields turned into
// the same numbers, so what the strip shows and what the render builds cannot
// come apart.
//   offset      the TRACK second that plays at film 0. Negative means the
//               music starts that many seconds INTO the film.
//   trim_start  the in-point inside the track. Absent means 0.
//   trim_end    the out-point inside the track. Absent means "to the end".
// Returns the block's film span and the window it plays from the track.
function sbeMusicWindow(audio, dur) {
  const a = audio || {};
  const off = sbeNum(a.offset);
  const total = Math.max(0, sbeNum(dur, 0) || sbeNum(a.duration, 0));
  const ts = Math.max(0, sbeNum(a.trim_start));
  let te = (a.trim_end === null || a.trim_end === undefined)
    ? null : sbeNum(a.trim_end);
  // A handle dragged back out to the end of the track is not a trim — the
  // same rule the server's `music_window` and `normalise_edit` follow, so the
  // strip stops calling itself trimmed the moment the field stops being saved.
  if (te !== null && total > 0) te = (te >= total - 1e-6) ? null : Math.min(te, total);
  if (te !== null && te <= ts) te = null;
  const head = Math.max(0, ts, off);
  let tail = (te !== null) ? te : (total > 0 ? total : null);
  if (tail !== null && tail <= head) tail = null;
  // Where the block sits on the FILM. A head trim does not slide the rest of
  // the track earlier — music does not ripple — so the seconds it removes come
  // back as silence in front, which is exactly `head - offset`.
  const filmStart = sbeRound(Math.max(0, head - off));
  const filmEnd = (tail !== null) ? sbeRound(tail - off) : null;
  return { offset: off, duration: total, head: sbeRound(head),
           tail: (tail === null ? null : sbeRound(tail)),
           trimmed: (ts > 0) || (te !== null),
           film_start: filmStart,
           film_end: (filmEnd === null ? null : Math.max(filmStart, filmEnd)) };
}

// One gesture on the music, as pure arithmetic. `mode` is 'move' | 'trimL' |
// 'trimR' and `want` the film second the pointer landed on; the return is the
// three fields to write back, never a mutation — the same contract every clip
// edit above follows, and the reason a drag can be tested in node.
function sbeMusicEdit(audio, mode, want, dur) {
  const w = sbeMusicWindow(audio, dur);
  const total = w.duration;
  const ts = Math.max(0, sbeNum((audio || {}).trim_start));
  const te = (w.tail === null) ? (total > 0 ? total : null) : w.tail;
  const out = { offset: w.offset, trim_start: ts,
                trim_end: ((audio || {}).trim_end === null
                           || (audio || {}).trim_end === undefined)
                  ? null : sbeNum((audio || {}).trim_end) };
  if (mode === 'move') {
    // Dragging the block moves the whole object: the window into the track is
    // untouched, only where it lands on the film changes.
    //
    // SOLVED FOR THE OFFSET, NOT ROUTED THROUGH THE HEAD. `w.head` is
    // max(0, trim_start, offset), so on a document whose offset is positive —
    // a documented, supported state; `music_window`'s own docstring calls it a
    // head trim — writing `head - want` fed the PREVIOUS offset back into the
    // answer. A pointermove stream re-reads the already-mutated object every
    // event, so the same six-second drag landed at film 6 when it arrived as
    // six events and did not move the block at all when it arrived as one: a
    // gesture whose result depended on how fast the mouse was going. Worse,
    // `offset` changed even when the block visibly did not, so the drag
    // marked the film dirty, burned an undo step and queued a write.
    // Both branches below give film_start === t under sbeMusicWindow and
    // neither reads the offset, so N events converge on what one event does.
    const t = Math.max(0, sbeNum(want));
    out.offset = sbeRound(ts > 0 ? (ts - t) : -t);
    return out;
  }
  if (mode === 'trimL') {
    // The left edge chooses a new in-point. Bounded by the head of the track
    // one way and by the out-point (less the floor) the other.
    const filmEnd = (w.film_end === null) ? null : w.film_end;
    let t = Math.max(0, sbeNum(want));
    if (filmEnd !== null) t = Math.min(t, filmEnd - SBE_MIN_MUSIC);
    let inPoint = sbeRound(t + w.offset);
    inPoint = Math.max(0, inPoint);
    if (te !== null) inPoint = Math.min(inPoint, sbeRound(te - SBE_MIN_MUSIC));
    out.trim_start = Math.max(0, sbeRound(inPoint));
    return out;
  }
  // trimR — the out-point, bounded by the in-point and by the track's own end.
  let t = Math.max(0, sbeNum(want));
  let outPoint = sbeRound(t + w.offset);
  outPoint = Math.max(sbeRound(Math.max(ts, w.head) + SBE_MIN_MUSIC), outPoint);
  if (total > 0) outPoint = Math.min(outPoint, sbeRound(total));
  out.trim_end = sbeRound(outPoint);
  return out;
}

// The times a music edge is allowed to click onto: the start of the film and
// every cut in it. NOT the beat grid — the grid is derived from this very
// track at this very offset, so snapping the music to it would be snapping a
// ruler to marks the ruler drew.
function sbeMusicSnaps(clips) {
  const out = [0];
  for (const c of clips || []) {
    out.push(sbeRound(sbeNum(c.film_start)));
    out.push(sbeRound(sbeNum(c.film_end)));
  }
  return out;
}

function sbeSnapToList(t, marks, tol, enabled) {
  if (!enabled) return sbeRound(t);
  let best = null, dist = Infinity;
  for (const m of marks || []) {
    const d = Math.abs(m - t);
    if (d > tol || d >= dist - 1e-9) continue;
    dist = d; best = m;
  }
  return best === null ? sbeRound(t) : sbeRound(best);
}

// THE CLIP'S OWN SOUND, and the mirror of `clip_audio()` on the server.
// ABSENT MEANS LINKED — the field appears only once somebody has pulled the
// picture and the sound apart, so every clip on every disk is already valid.
function sbeClipAudio(c) {
  const vs = sbeNum((c || {}).start), ve = sbeNum((c || {}).end);
  const fs = sbeNum((c || {}).film_start);
  const a = (c || {}).audio;
  // THE SOUND RUNS AT THE CLIP'S SPEED, linked or not — the mirror of
  // `clip_audio()`. `len` is the strip's length ON THE FILM.
  const sp = sbeSpeed(c);
  if (!a || typeof a !== 'object' || sbeKind(c) !== 'video') {
    return { start: vs, end: ve, film_start: fs, linked: true,
             coupled: false, split: false, speed: sp,
             len: sbeRound(Math.max(0, ve - vs) / sp) };
  }
  // THE PRESENCE OF THE FIELD IS THE SWITCH, not the values in it. Unlinking
  // writes the window the clip already has, so an equality test read a
  // just-unlinked clip as linked and refused to drag it.
  //
  // ...AND `audio.linked` IS THE THIRD STATE: split, but travelling with its
  // picture at the offset the user chose. `linked` still means "this strip
  // cannot be dragged on its own", which is true of both; `split` means "the
  // sound is described separately", which is what the assembler needs.
  const s2 = sbeNum(a.start, vs), e2 = sbeNum(a.end, ve);
  const f2 = sbeNum(a.film_start, fs);
  const coupled = a.linked === true;
  return { start: sbeRound(s2), end: sbeRound(e2), film_start: sbeRound(f2),
           linked: coupled, coupled: coupled, split: true, speed: sp,
           len: sbeRound(Math.max(0, e2 - s2) / sp) };
}

// Rebuild the stored object from a window, keeping the coupling flag out of
// the document unless it is true — a free strip must be byte-identical to the
// one every edit.json on disk already carries.
function sbeAudioField(w, coupled) {
  const o = { start: sbeRound(w.start), end: sbeRound(w.end),
              film_start: Math.max(0, sbeRound(w.film_start)) };
  if (coupled) o.linked = true;
  return o;
}

// UNLINKING IS NOT AN EDIT TO THE SOUND. It writes the window the clip
// already had, which is what makes the toggle safe to press: nothing moves
// until you move it.
//
// AND NEITHER IS RE-LINKING, ANY MORE. It used to DELETE the field, which
// snapped the sound back under the picture — so the moment the owner had
// built the J-cut he wanted, the one button that said "link" threw it away,
// and he reached for LOCK instead (which makes a clip refuse every drag with
// a forbidden cursor, and looked like the editor had broken). Re-linking now
// FREEZES the relationship: the window stays exactly where he put it and the
// pair travels as one from then on. "You just drag it, and the sound below
// stays, and then you can lock it and move it, and then the sound starts
// before the clip starts."
//
// An IN-SYNC re-link still deletes the field outright, so a split somebody
// tried and undid leaves a document identical to one that never had it.
function sbeSetAudioLink(clips, id, linked) {
  const c = sbeById(clips, id);
  if (!c) return { clips: clips, ok: false, why: 'gone' };
  if (sbeKind(c) !== 'video') {
    return { clips: clips, ok: false, why: 'only a video clip has sound of its own' };
  }
  const out = clips.map(x => Object.assign({}, x));
  const t = sbeById(out, id);
  const w = sbeClipAudio(c);
  if (linked) {
    if (sbeAudioIsThePicture(c)) delete t.audio;
    else t.audio = sbeAudioField(w, true);
  } else {
    t.audio = sbeAudioField(w, false);
  }
  t.source = 'human';
  return { clips: out, ok: true, coupled: linked && !!t.audio };
}

// One gesture on one clip's sound. `mode` is 'move' | 'trimL' | 'trimR', and
// none of them touches the picture — that is the entire point of the feature.
// A LINKED clip refuses: a strip that moved without being unlinked would be
// the accident the default exists to prevent.
function sbeAudioEdit(clips, id, mode, want) {
  const c = sbeById(clips, id);
  if (!c) return { clips: clips, ok: false, why: 'gone' };
  const w = sbeClipAudio(c);
  if (w.coupled) {
    return { clips: clips, ok: false,
             why: 'this sound travels with its picture at the offset you set — unlink it to move it on its own' };
  }
  if (w.linked) return { clips: clips, ok: false, why: 'unlink the sound first' };
  if (c.locked) return { clips: clips, ok: false, why: 'locked' };
  const src = sbeNum(c.duration, 0);
  const out = clips.map(x => Object.assign({}, x));
  const t = sbeById(out, id);
  let start = w.start, end = w.end, film = w.film_start;
  const t0 = Math.max(0, sbeNum(want));
  if (mode === 'move') {
    film = sbeRound(t0);
  } else if (mode === 'trimL') {
    // The left edge picks a new IN-POINT and leaves the rest where it is —
    // the same rule the music's head trim follows, and the reason an L-cut
    // does not drag the next line early with it.
    //
    // THE CLAMP IS FOLDED BACK INTO BOTH FIELDS OR IT IS NOT A CLAMP. `start`
    // used to stop at the head of the source while `film` kept sliding, so
    // asking for more head than the take has moved the strip's OUT-point
    // left — the one thing a left-edge trim must never do. That is the L-cut
    // case exactly: you pull his line back under the previous shot and the
    // tail silently loses the same seconds off the end of the line you were
    // keeping.
    // FILM seconds on the lane, SOURCE seconds in the window: a retimed
    // strip moves `speed` seconds of take for every film second dragged.
    const room = Math.min(t0, sbeRound(film + (end - start) / w.speed - SBE_MIN_CLIP));
    let delta = sbeRound(room - film);
    if (start + delta * w.speed < 0) delta = sbeRound(-start / w.speed);
    start = sbeRound(start + delta * w.speed);
    film = sbeRound(film + delta);
  } else {
    let e = sbeRound(start + Math.max(SBE_MIN_CLIP, t0 - film) * w.speed);
    if (src > 0) e = Math.min(e, sbeRound(src));
    end = Math.max(sbeRound(start + SBE_MIN_CLIP), e);
  }
  t.audio = { start: sbeRound(start), end: sbeRound(end),
              film_start: Math.max(0, sbeRound(film)) };
  t.source = 'human';
  return { clips: out, ok: true };
}

// ---- the pair, and how far it has come apart -------------------------------
// THE MIRROR OF `clip_audio_drift()`. Both halves map film time to source time
// with one constant each — the picture's `film_start - start`, the strip's
// `audio.film_start - audio.start` — so the difference between the two is the
// number an NLE prints on its sync flag. POSITIVE means the sound plays LATE
// against the frame it was recorded with. A linked clip cannot drift.
function sbeAudioDrift(c) {
  const w = sbeClipAudio(c);
  if (!w.split) return 0;
  // On the film's clock: a source second is `1/speed` film seconds on both
  // halves, the same arithmetic `clip_audio_drift()` does.
  return sbeRound((w.film_start - w.start / w.speed)
                  - (sbeNum((c || {}).film_start) - sbeNum((c || {}).start) / w.speed));
}

function sbeAudioInSync(c) { return Math.abs(sbeAudioDrift(c)) <= SBE_SYNC_TOL; }

// IN SYNC IS NOT THE SAME AS "NOTHING TO SAY". A head trim moves the picture's
// in-point and its slot together, so the strip is left reaching a second back
// under the shot before it — the J-cut he described, "you start hearing the
// character before you see it" — and its DRIFT is zero, because the same
// source second still plays at the same film second. Only a strip that is
// literally the picture's own window can be dropped; deciding on drift alone
// deleted the field and took that extra second of sound with it.
// THE CLIP'S OWN SOUND, SWITCHED OFF. The mirror of `clip_muted()`.
// "We should have an option to mute the clip sound." An H3 shot arrives with
// baked-in wind and ambience under the line, and on a music cut that is not a
// performance to be balanced — it is noise to be removed so the track can
// carry the moment.
//
// NOT `has_audio === false`, which says the FILE has no audio track and is a
// fact about the source. This is a decision about the edit, and the two are
// painted differently for that reason. It composes with unlinking in both
// directions: `mute` describes the sound wherever its strip happens to be.
function sbeClipMuted(c) {
  return !!(c && c.mute === true) && sbeKind(c) === 'video';
}

// ABSENT IS AUDIBLE, the same rule `adjust` and `audio` follow: an unmuted
// clip is byte-identical to one written before mute existed.
function sbeSetClipMute(clips, id, on) {
  const c = sbeById(clips, id);
  if (!c) return { clips: clips, ok: false, why: 'gone' };
  if (sbeKind(c) !== 'video') {
    return { clips: clips, ok: false, why: 'only a video clip has sound to mute' };
  }
  if (c.has_audio === false) {
    return { clips: clips, ok: false,
             why: 'this clip has no sound of its own to mute' };
  }
  if (!!c.mute === !!on) return { clips: clips, ok: false, why: 'no change' };
  const out = clips.map(x => Object.assign({}, x));
  const t = sbeById(out, id);
  if (on) t.mute = true; else delete t.mute;
  t.source = 'human';
  return { clips: out, ok: true };
}

function sbeAudioIsThePicture(c) {
  const w = sbeClipAudio(c);
  if (!w.split) return true;
  return Math.abs(w.start - sbeNum((c || {}).start)) < 1e-9
      && Math.abs(w.end - sbeNum((c || {}).end)) < 1e-9
      && Math.abs(w.film_start - sbeNum((c || {}).film_start)) < 1e-9;
}

function sbeDriftLabel(d) {
  const v = sbeNum(d);
  return (v > 0 ? '+' : '-') + Math.abs(v).toFixed(2) + 's';
}

// ---- carrying the sound through a RIPPLE ----------------------------------
// THE BUG THIS EXISTS FOR. `audio.film_start` is an absolute film anchor, and
// `sbeLayout` re-derives the film position of every clip from the running
// total of the lead gaps — the PICTURE's position and nothing else. So a right
// trim, a move, a ripple delete, an insert or a reorder slid every clip after
// it while every unlinked strip stood still, and a J-cut the user had placed
// three shots earlier silently came apart:
//
//   "instead of allowing me to remove or move what video is visible while
//    leaving the sound intact and then rematching it, it is actually getting
//    the audio out of sync."
//
// A ripple is a RIGID TRANSLATION of everything downstream, so the sound rides
// with it — that is what keeps a pair in sync through an edit made somewhere
// else. The clip the gesture is actually ON is exempt: moving a picture off
// its own sound is the whole point of the feature, and `sbeAudioEdit` is the
// only other thing allowed to write the strip's position.
// WHAT IS RECORDED IS THE ANCHOR, not the film position: `film_start - start`
// is the film second at which this take's source zero would play, and it is
// the only number a sound has to follow. A ripple changes it by the whole
// translation; a head trim moves the slot and the in-point together and does
// not change it at all, which is exactly why a trim must leave the strip
// alone while a ripple must not.
function sbeSyncMark(clips) {
  const m = {};
  for (const c of clips || []) {
    if (!c || c.id === undefined || c.id === null) continue;
    if (sbeClipAudio(c).split) {
      m[c.id] = sbeRound(sbeNum(c.film_start) - sbeNum(c.start));
    }
  }
  return m;
}

function sbeSyncCarry(clips, mark, exempt) {
  const skip = {};
  for (const id of (exempt || [])) skip[String(id)] = true;
  for (const c of clips || []) {
    if (!c) continue;
    const a = c.audio;
    if (!a || typeof a !== 'object') continue;
    // THE EXEMPTION IS FOR A FREE STRIP ONLY. A COUPLED pair travels with its
    // picture by definition — that is what the user froze when he re-linked
    // it — so the gesture that moves the picture moves it too, including the
    // deliberate drag the free strip is exempt from.
    if (a.linked !== true && skip[String(c.id)]) continue;
    const was = mark ? mark[c.id] : undefined;
    if (was === undefined) continue;
    const d = sbeRound((sbeNum(c.film_start) - sbeNum(c.start)) - was);
    if (Math.abs(d) < 1e-9) continue;
    c.audio = sbeAudioField(
      { start: sbeNum(a.start), end: sbeNum(a.end),
        film_start: sbeNum(a.film_start) + d }, a.linked === true);
  }
  return clips;
}

// THE REMATCH the owner asked for by name. One click puts the sound back under
// the frame it came from: the strip keeps its own in-point — re-matching is not
// un-trimming, an L-cut's tail stays as long as it was made — and moves to the
// film second where that source second now plays.
function sbeResyncAudio(clips, id) {
  const c = sbeById(clips, id);
  if (!c) return { clips: clips, ok: false, why: 'gone' };
  const w = sbeClipAudio(c);
  if (!w.split) {
    return { clips: clips, ok: false,
             why: 'this clip\'s sound is already under its own picture' };
  }
  if (c.locked) return { clips: clips, ok: false, why: 'locked' };
  const want = sbeRound(sbeNum(c.film_start) + (w.start - sbeNum(c.start)) / w.speed);
  if (want < 0) {
    return { clips: clips, ok: false,
             why: 'the sound would have to start before the __SEQ__ does — move the picture later first' };
  }
  if (Math.abs(want - w.film_start) < 1e-9) {
    return { clips: clips, ok: false, why: 'that sound is already in sync' };
  }
  const out = clips.map(x => Object.assign({}, x));
  const t = sbeById(out, id);
  // A COUPLED pair rematched is a pair with nothing left to say, so the field
  // goes rather than persisting as a coupling at zero offset.
  t.audio = sbeAudioField({ start: w.start, end: w.end, film_start: want },
                          w.coupled);
  // A COUPLED pair rematched onto its own window is exactly a plain linked
  // clip, so the field goes. A FREE strip keeps its field even when the
  // numbers now match the picture: the user unlinked it on purpose and
  // stripping it here would silently re-link a strip he still means to drag —
  // the same rule `normalise_edit` follows on the way to disk.
  if (w.coupled && sbeAudioIsThePicture(t)) delete t.audio;
  t.source = 'human';
  return { clips: out, ok: true };
}

// ---- THE STRIP PLAYER'S MODEL -------------------------------------------
// THE PREVIEW MUST PLAY WHAT THE DOCUMENT SAYS. Until now it could not: the
// transport plays ONE <video> at a time and enters each clip at its picture
// boundary, so a clip's sound always landed with its picture no matter where
// its strip sat. The whole J/L-cut feature was inaudible — correct on disk,
// correct in the render's concat lanes, correct in the NLE export, and
// missing from the one place the user checks his work:
//
//   "I wanted it a little before her showing up, and I cut it that way. But
//    no matter what I do, this is always out."
//
// WHO OWNS A CLIP'S SOUND. Exactly one of the two, never both, or the same
// seconds play twice. The picture element keeps it when the strip IS the
// picture — that is every ordinary clip, and it costs nothing. The moment a
// strip exists in its own right, the player takes it.
// ---- THE EFFECTS MODEL, client side -------------------------------------
// The mirror of `clip_effects()`. ONE ACCESSOR, whatever the storage: `fx` is
// the home for effects, and brightness stays where history put it because a
// label is not worth a data migration. See docs/EDITOR_EFFECTS_MODEL.md.
function sbeClipLen(c) {
  const n = sbeNum((c || {}).film_end) - sbeNum((c || {}).film_start);
  return Math.max(0, n > 0 ? n
    : (sbeNum((c || {}).end) - sbeNum((c || {}).start)) / sbeSpeed(c));
}

// Sets the clip's play rate and RIPPLES: the slot shrinks or grows in place
// and everything after it slides, sound included. The window is untouched —
// speed changes how fast the same seconds of the take play, never which.
function sbeSetSpeed(clips, id, v) {
  const c = sbeById(clips, id);
  if (!c) return { clips: clips, ok: false, why: 'gone' };
  if (c.locked) return { clips: clips, ok: false, why: 'locked' };
  if (sbeKind(c) !== 'video') {
    return { clips: clips, ok: false, why: 'only a video clip has a clock to run fast or slow' };
  }
  const want = sbeRound(Math.max(SBE_SPEED_MIN, Math.min(SBE_SPEED_MAX, sbeNum(v, 1) || 1)));
  if (Math.abs(want - sbeSpeed(c)) < 1e-9) return { clips: clips, ok: false, why: 'no change' };
  const mark = sbeSyncMark(clips);
  const out = clips.map(x => Object.assign({}, x));
  const t = sbeById(out, id);
  // NEUTRAL IS ABSENT: 1x is no field, so an untouched clip is byte-identical
  // to one from before speed existed.
  if (Math.abs(want - 1) < 1e-9) delete t.speed; else t.speed = want;
  t.source = 'human';
  sbeLayout(out);
  sbeSyncCarry(out, mark, [id]);
  return { clips: out, ok: true };
}

function sbeFx(c) {
  const fx = ((c || {}).fx && typeof c.fx === 'object') ? c.fx : {};
  const n = sbeClipLen(c);
  let fin = Math.max(0, sbeNum(fx.fade_in));
  let fout = Math.max(0, sbeNum(fx.fade_out));
  if (n > 0) {
    fin = Math.min(fin, n); fout = Math.min(fout, n);
    // THE SAME CLAMP THE SERVER APPLIES, and it is here for the same reason:
    // two fades that crossed would ask for an opacity that is two things at
    // once. Proportional, because "two long fades" means mostly-ramp and
    // zeroing one of them is not that.
    const over = fin + fout - n;
    if (over > 0) {
      const total = fin + fout;
      fin -= over * (fin / total); fout -= over * (fout / total);
    }
  }
  return { fade_in: sbeRound(fin), fade_out: sbeRound(fout),
           brightness: sbeBright(c) };
}

// THE OPACITY AT A FILM SECOND, and the preview's whole honesty about fades:
// a value per frame rather than a CSS transition, so scrubbing shows what is
// true at that second instead of an animation that started when you arrived.
// `edges` is a transition's share of this clip's two ends ({head, tail} in
// seconds, from `sbeTxEdges`). ONE <video> cannot show two pictures, so on
// the stage a dissolve is previewed as a ramp through the stage's black on
// both sides of the cut — the exact picture for a fade through black, an
// honest approximation for a dissolve, and the badge says which.
function sbeFadeOpacityAt(c, t, edges) {
  const e = sbeFx(c);
  const ed = edges || { head: 0, tail: 0 };
  const fin = Math.max(e.fade_in, sbeNum(ed.head)), fout = Math.max(e.fade_out, sbeNum(ed.tail));
  if (fin <= 1e-9 && fout <= 1e-9) return 1;
  const fs = sbeNum((c || {}).film_start), fe = sbeNum((c || {}).film_end);
  const now = sbeNum(t);
  let o = 1;
  if (fin > 1e-9 && now < fs + fin) {
    o = Math.min(o, Math.max(0, (now - fs) / fin));
  }
  if (fout > 1e-9 && now > fe - fout) {
    o = Math.min(o, Math.max(0, (fe - now) / fout));
  }
  return Math.max(0, Math.min(1, sbeRound(o)));
}

// One fade, set in seconds. Clamped by `sbeFx` on the way back out, so the UI
// can hand this a drag distance without doing the arithmetic itself.
function sbeSetFade(clips, id, edge, seconds) {
  const c = sbeById(clips, id);
  if (!c) return { clips: clips, ok: false, why: 'gone' };
  if (c.locked) return { clips: clips, ok: false, why: 'locked' };
  const key = (edge === 'out') ? 'fade_out' : 'fade_in';
  const want = Math.max(0, Math.min(sbeClipLen(c), sbeNum(seconds)));
  const now = sbeFx(c);
  if (Math.abs(now[key] - want) < 1e-6 && !(want === 0 && c.fx && c.fx[key])) {
    return { clips: clips, ok: false, why: 'no change' };
  }
  const out = clips.map(x => Object.assign({}, x));
  const t = sbeById(out, id);
  const fx = Object.assign({}, t.fx || {});
  if (want > 1e-9) fx[key] = sbeRound(want); else delete fx[key];
  // NEUTRAL IS ABSENT: a clip whose fades are both zero carries no `fx` at
  // all, so it is byte-identical to one from before effects existed.
  if (fx.fade_in > 1e-9 || fx.fade_out > 1e-9) t.fx = fx; else delete t.fx;
  t.source = 'human';
  return { clips: out, ok: true };
}

// ---- PER-SOURCE WAVEFORMS -----------------------------------------------
// THE STRIP'S OWN SOURCE WINDOW, never the picture's. A strip trimmed to
// 1.2–3.4 draws those seconds of the take; a J-cut that slides its sound half
// a second earlier still draws the seconds it PLAYS, because the window is a
// fact about the sound and the film position is not.
function sbeWaveSlice(peaks, from, to, cols) {
  if (!peaks || !peaks.peaks || !peaks.count) return [];
  const bps = sbeNum(peaks.buckets_per_second, 100) || 100;
  const scale = sbeNum(peaks.scale, 127) || 127;
  const n = Math.max(1, Math.floor(sbeNum(cols)));
  const b0 = Math.max(0, Math.floor(sbeNum(from) * bps));
  const b1 = Math.min(sbeNum(peaks.count), Math.ceil(sbeNum(to) * bps));
  const span = Math.max(1, b1 - b0);
  const out = [];
  for (let i = 0; i < n; i++) {
    // One column may cover many buckets when the strip is zoomed out; take
    // the extremes across them, or a quiet frame in the middle of a loud
    // second draws a hole that is not there.
    const s0 = b0 + Math.floor((i / n) * span);
    const s1 = Math.max(s0 + 1, b0 + Math.floor(((i + 1) / n) * span));
    let lo = 0, hi = 0;
    for (let b = s0; b < s1 && b < b1; b++) {
      const mn = peaks.peaks[b * 2], mx = peaks.peaks[b * 2 + 1];
      if (mn < lo) lo = mn;
      if (mx > hi) hi = mx;
    }
    out.push([lo / scale, hi / scale]);
  }
  return out;
}

// Lazily, once per source, and never twice for the same take.
function sbeWaveWant(path) {
  const key = String(path || '');
  if (!key || !SBE.open) return null;
  if (!SBE.clipPeaks) SBE.clipPeaks = {};
  if (Object.prototype.hasOwnProperty.call(SBE.clipPeaks, key)) {
    return SBE.clipPeaks[key];
  }
  SBE.clipPeaks[key] = null;                      // in flight: ask once
  fetch('/storyboard/edit/clip-peaks?id=' + encodeURIComponent(SBE.id)
        + '&path=' + encodeURIComponent(key))
    .then(r => r.json())
    .then(d => {
      // A take with no audio is a FACT, and `false` is how the lane
      // remembers it so it never asks again.
      SBE.clipPeaks[key] = (d && d.peaks) ? d : false;
      sbePaintAudioLane();
    })
    .catch(() => { SBE.clipPeaks[key] = false; });
  return null;
}

// ---- THE SOUND'S OWN ENVELOPE -------------------------------------------
// The mirror of `audio_gain_points()`. ONE CURVE, fades and keyframes folded
// together, read by the preview exactly as the render and the export read the
// server's copy — so the simple case never has to discover keyframes and a
// keyframed envelope never has to be re-expressed. `t` is STRIP-RELATIVE, so
// sliding a J-cut does not drag every point with it.
function sbeAfx(item, len) {
  const a = ((item || {}).afx && typeof item.afx === 'object') ? item.afx : {};
  const n = Math.max(0, sbeNum(len));
  let fin = Math.max(0, sbeNum(a.fade_in));
  let fout = Math.max(0, sbeNum(a.fade_out));
  if (n > 0) {
    fin = Math.min(fin, n); fout = Math.min(fout, n);
    const over = fin + fout - n;
    if (over > 0) {
      const total = fin + fout;
      fin -= over * (fin / total); fout -= over * (fout / total);
    }
  }
  const pts = [];
  for (const row of (a.points || [])) {
    if (!row || row.length !== 2) continue;
    let t = sbeNum(row[0]);
    if (n > 0) t = Math.max(0, Math.min(n, t));
    pts.push([sbeRound(t), sbeRound(Math.max(0, Math.min(1, sbeNum(row[1]))))]);
  }
  pts.sort((x, y) => x[0] - y[0]);
  return { fade_in: sbeRound(fin), fade_out: sbeRound(fout), points: pts };
}

function sbeLerpGain(pts, t) {
  if (!pts || !pts.length) return 1;
  if (t <= pts[0][0]) return pts[0][1];
  if (t >= pts[pts.length - 1][0]) return pts[pts.length - 1][1];
  for (let i = 0; i < pts.length - 1; i++) {
    const a = pts[i], b = pts[i + 1];
    if (t >= a[0] && t <= b[0]) {
      if (b[0] - a[0] <= 1e-9) return b[1];
      return a[1] + (b[1] - a[1]) * ((t - a[0]) / (b[0] - a[0]));
    }
  }
  return pts[pts.length - 1][1];
}

function sbeGainPoints(item, len) {
  const n = Math.max(0, sbeNum(len));
  if (n <= 0) return [];
  const e = sbeAfx(item, n);
  const marks = { 0: true };
  marks[n] = true;
  for (const p of e.points) marks[Math.min(n, Math.max(0, p[0]))] = true;
  if (e.fade_in > 1e-9) marks[e.fade_in] = true;
  if (e.fade_out > 1e-9) marks[Math.max(0, n - e.fade_out)] = true;
  const out = [];
  for (const k of Object.keys(marks).map(Number).sort((a, b) => a - b)) {
    let g = sbeLerpGain(e.points, k);
    if (e.fade_in > 1e-9 && k < e.fade_in) g *= k / e.fade_in;
    if (e.fade_out > 1e-9 && k > n - e.fade_out) {
      g *= Math.max(0, (n - k) / e.fade_out);
    }
    out.push([sbeRound(k), sbeRound(Math.max(0, Math.min(1, g)))]);
  }
  // A FLAT UNITY CURVE IS NO CURVE, the same rule the server follows.
  return out.every(p => Math.abs(p[1] - 1) < 1e-9) ? [] : out;
}

function sbeGainAt(item, len, t) {
  const curve = sbeGainPoints(item, len);
  if (!curve.length) return 1;
  return sbeRound(sbeLerpGain(curve, Math.max(0, sbeNum(t))));
}

// ---- THE MIX: the bed's level, and what happens to it under a line -------
// The browser half of `storyboard_editor.audio_mix` and everything under it.
// Every function here is a mirror of a Python one of the same shape, and that
// is the entire point of the feature: the render used to hold the bed at a
// hard-coded 0.20 and duck it through a compressor keyed on the dialogue,
// while the preview applied neither — so the one surface the user checks his
// work on played a mix the file never had. A gain that only one of the three
// outputs can express is not in the model.
//
// KEEP THESE EQUAL TO THE SERVER'S. A drifted constant here is a preview that
// plays a different film from the render, quietly, which is the exact defect
// this block exists to close. `test_the_client_and_the_server_agree_about_the
// _mix_constants` reads both out and refuses a difference.
const SBE_MIX_BED_GAIN = 1.0;
const SBE_MIX_DUCK = false;
const SBE_MIX_DUCK_GAIN = 0.269;      // the measured 11.4 dB, in linear
const SBE_MIX_DUCK_ATTACK = 0.005;
const SBE_MIX_DUCK_RELEASE = 0.4;

function sbeAudioMix(audio) {
  const a = audio || {};
  const m = (a.mix && typeof a.mix === 'object') ? a.mix : {};
  const g = (m.bed_gain === null || m.bed_gain === undefined)
    ? SBE_MIX_BED_GAIN : Math.max(0, Math.min(1, sbeNum(m.bed_gain)));
  const d = (m.duck === null || m.duck === undefined) ? SBE_MIX_DUCK : !!m.duck;
  return { bed_gain: sbeRound(g), duck: d };
}

// THE BED'S ENVELOPE IS ON THE PLAYED WINDOW, not on the track — zero is the
// first second you hear, exactly as a clip strip's envelope is on the strip
// and not on the source file. That is what makes the corner handles mean the
// corner they were dragged onto after the music has been trimmed or moved.
//
// THE DOCUMENT ONLY, AND THEN THE FILM. `filmLen` is `sbeFilmDuration(clips)`
// and the peaks probe is deliberately NOT in this chain, which is the second
// half of the fix the film-clock fallback belongs to. The renderer cannot see
// peaks.json — it reads edit.json — so a bed length taken from the probe is a
// number only one of the two sides can compute, and a gain only one side can
// compute is exactly the invisible-second-author defect this whole block
// exists to remove. `storyboard_editor.bed_length(audio, film_len)` is this
// function, term for term, and `test_editor_mix` runs the two side by side
// over a TABLE of documents so a divergence turns a suite red instead of
// turning the preview into a liar.
function sbeBedLen(audio, filmLen) {
  const a = audio || {};
  const w = sbeMusicWindow(a, 0);          // 0 => the document's own duration
  if (w.film_end === null) {
    // No stated length. The bed plays under the FILM — what remains of it
    // after the block starts — because that is where the render stops the mix
    // anyway. Returning 0 here is what used to empty the curve, and an empty
    // curve means NO FILTER, which is the bed at full level over the dialogue.
    return Math.max(0, sbeRound(Math.max(0, sbeNum(filmLen)) - w.film_start));
  }
  return Math.max(0, sbeRound(w.film_end - w.film_start));
}

// The film seconds where a clip's OWN sound plays. What the duck is keyed on,
// and the reason it can be a document value at all: a compressor asks "are
// these samples loud", this asks "is there a sound strip here" — a question
// the document answers and the browser can answer identically.
function sbeAudibleStrips(clips) {
  const wins = [];
  for (const c of (clips || [])) {
    if (sbeKind(c) !== 'video') continue;
    if (c.has_audio === false || sbeClipMuted(c)) continue;
    const w = sbeClipAudio(c);
    if (w.len > 1e-9) wins.push([sbeRound(w.film_start), sbeRound(w.film_start + w.len)]);
  }
  wins.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const out = [];
  for (const [s, e] of wins) {
    // Merged when they sit closer than one release: a bed that recovered
    // fully in the eighth of a second between two lines would pump, and
    // merging is also what keeps every knot of the curve below on a mark.
    if (out.length && s - out[out.length - 1][1] < SBE_MIX_DUCK_RELEASE - 1e-9) {
      out[out.length - 1][1] = Math.max(out[out.length - 1][1], e);
    } else out.push([s, e]);
  }
  return out;
}

function sbeDuckGainAt(wins, t) {
  let g = 1;
  for (const [s, e] of wins) {
    if (t < s - 1e-9) continue;
    let v;
    if (t <= e + 1e-9) {
      const k = SBE_MIX_DUCK_ATTACK <= 0 ? 1
        : Math.min(1, (t - s) / SBE_MIX_DUCK_ATTACK);
      v = 1 - (1 - SBE_MIX_DUCK_GAIN) * Math.max(0, k);
    } else {
      const k = SBE_MIX_DUCK_RELEASE <= 0 ? 1 : (t - e) / SBE_MIX_DUCK_RELEASE;
      if (k >= 1) continue;
      v = SBE_MIX_DUCK_GAIN + (1 - SBE_MIX_DUCK_GAIN) * k;
    }
    g = Math.min(g, v);
  }
  return Math.max(0, Math.min(1, g));
}

function sbeBedDuckPoints(clips, len, delay) {
  const n = Math.max(0, sbeNum(len));
  if (n <= 0) return [];
  const d = sbeNum(delay);
  const wins = sbeAudibleStrips(clips)
    .map(([s, e]) => [s - d, e - d]).filter(([s, e]) => e > 0 && s < n);
  if (!wins.length) return [];
  const marks = new Set([0, sbeRound(n)]);
  for (const [s, e] of wins) {
    for (const m of [s, s + SBE_MIX_DUCK_ATTACK, e, e + SBE_MIX_DUCK_RELEASE]) {
      if (m >= -1e-9 && m <= n + 1e-9) marks.add(sbeRound(Math.max(0, Math.min(n, m))));
    }
  }
  const out = Array.from(marks).sort((a, b) => a - b)
    .map(t => [t, sbeRound(sbeDuckGainAt(wins, t))]);
  return out.every(p => Math.abs(p[1] - 1) < 1e-9) ? [] : out;
}

// THE PRECEDENCE, ASKED ONCE. A person who has drawn the bed's level has said
// what the bed does; an automatic curve that then moved it would be the
// renderer disagreeing with them again — the invisible second author, one
// layer up. So the authored envelope WINS and the duck stands down, and the
// track head says so rather than leaving two controls quietly fighting.
function sbeBedDuckSuppressed(audio, filmLen) {
  if (!sbeAudioMix(audio).duck) return false;
  return sbeGainPoints(audio || {}, sbeBedLen(audio, filmLen)).length > 0;
}

// THE ONE BED CURVE. Preview, render and export read this and nothing else.
// Three terms: the static fader (always), the authored envelope (always, when
// it exists), the auto-duck (only when nothing was authored). Never two
// curves at once.
function sbeBedGainPoints(audio, clips, filmLen) {
  const a = audio || {};
  if (!a.path) return [];
  const n = sbeBedLen(a, filmLen);
  if (n <= 0) return [];
  const mix = sbeAudioMix(a);
  let curve = sbeGainPoints(a, n);
  if (!curve.length && mix.duck) {
    curve = sbeBedDuckPoints(clips, n, sbeMusicWindow(a, 0).film_start);
  }
  const g0 = mix.bed_gain;
  if (!curve.length) {
    if (Math.abs(g0 - 1) < 1e-9) return [];
    return [[0, sbeRound(g0)], [sbeRound(n), sbeRound(g0)]];
  }
  return curve.map(p => [p[0], sbeRound(Math.max(0, Math.min(1, p[1] * g0)))]);
}

function sbeBedGainAt(audio, clips, filmLen, t) {
  const curve = sbeBedGainPoints(audio, clips, filmLen);
  if (!curve.length) return 1;
  return sbeRound(sbeLerpGain(curve, Math.max(0, sbeNum(t))));
}

// ---- writing the mix back ----------------------------------------------
// Same shape as `sbeAfxWrite`: a new object out, neutral fields deleted, so a
// bed nobody has mixed is byte-identical to one from before the mix existed.
function sbeMixWrite(audio, patch) {
  const a = Object.assign({}, audio || {});
  const mix = Object.assign({}, sbeAudioMix(a), patch || {});
  const out = {};
  if (Math.abs(mix.bed_gain - SBE_MIX_BED_GAIN) > 1e-9) out.bed_gain = sbeRound(mix.bed_gain);
  if (!!mix.duck !== SBE_MIX_DUCK) out.duck = !!mix.duck;
  if (Object.keys(out).length) a.mix = out; else delete a.mix;
  return a;
}

function sbeBedAfxWrite(audio, afx) {
  const a = Object.assign({}, audio || {});
  const has = afx && (afx.fade_in > 1e-9 || afx.fade_out > 1e-9
                      || (afx.points && afx.points.length));
  if (has) a.afx = afx; else delete a.afx;
  return a;
}

function sbeSetBedFade(audio, edge, seconds, filmLen) {
  const n = Math.max(0, sbeBedLen(audio, filmLen));
  const key = (edge === 'out') ? 'fade_out' : 'fade_in';
  const afx = Object.assign({}, (audio || {}).afx || {});
  const want = Math.max(0, Math.min(n, sbeNum(seconds)));
  if (want > 1e-9) afx[key] = sbeRound(want); else delete afx[key];
  return sbeBedAfxWrite(audio, afx);
}

function sbeBedPointsWrite(audio, pts, filmLen) {
  const n = Math.max(0, sbeBedLen(audio, filmLen));
  const afx = Object.assign({}, (audio || {}).afx || {});
  const kept = (pts || []).slice().sort((x, y) => x[0] - y[0])
    .map(pr => [sbeRound(Math.max(0, Math.min(n, pr[0]))),
                sbeRound(Math.max(0, Math.min(1, pr[1])))]);
  if (kept.length) afx.points = kept; else delete afx.points;
  return sbeBedAfxWrite(audio, afx);
}

function sbeBedAddKeyframe(audio, t, gain, filmLen) {
  const n = Math.max(0, sbeBedLen(audio, filmLen));
  const at = Math.max(0, Math.min(n, sbeNum(t)));
  const pts = sbeAfx(audio || {}, n).points.slice();
  // A SECOND POINT ON THE SAME SECOND IS NOT A POINT — the same rule the
  // strips follow, and for the same reason: the envelope cannot express a
  // discontinuity and the NLEs would import the keyframes out of order.
  for (const pr of pts) if (Math.abs(pr[0] - at) < 1e-3) return null;
  pts.push([at, Math.max(0, Math.min(1, sbeNum(gain, 1)))]);
  return sbeBedPointsWrite(audio, pts, filmLen);
}

function sbeBedMoveKeyframe(audio, index, t, gain, filmLen) {
  const n = Math.max(0, sbeBedLen(audio, filmLen));
  const pts = sbeAfx(audio || {}, n).points.slice();
  if (!pts.length) return null;
  const i = Math.max(0, Math.min(pts.length - 1, sbeNum(index)));
  pts[i] = [Math.max(0, Math.min(n, sbeNum(t))),
            Math.max(0, Math.min(1, sbeNum(gain)))];
  return sbeBedPointsWrite(audio, pts, filmLen);
}

function sbeBedDeleteKeyframe(audio, index, filmLen) {
  const n = Math.max(0, sbeBedLen(audio, filmLen));
  const pts = sbeAfx(audio || {}, n).points.slice();
  const i = sbeNum(index);
  if (i < 0 || i >= pts.length) return null;
  pts.splice(i, 1);
  return sbeBedPointsWrite(audio, pts, filmLen);
}

// ---- KEYFRAMES: the control case ---------------------------------------
// Points are STRIP-RELATIVE seconds and a linear 0..1 gain, exactly as the
// server stores them. Adding, dragging and deleting are three ops on one
// list, and all three go through `sbeAfx` on the way out so what the UI holds
// is always what the three outputs will read.
function sbeAfxWrite(clips, id, pts) {
  const out = clips.map(x => Object.assign({}, x));
  const t = sbeById(out, id);
  const afx = Object.assign({}, t.afx || {});
  const kept = (pts || []).slice().sort((a, b) => a[0] - b[0])
    .map(pr => [sbeRound(pr[0]), sbeRound(Math.max(0, Math.min(1, pr[1])))]);
  if (kept.length) afx.points = kept; else delete afx.points;
  const has = afx.fade_in > 1e-9 || afx.fade_out > 1e-9
              || (afx.points && afx.points.length);
  if (has) t.afx = afx; else delete t.afx;
  t.source = 'human';
  return { clips: out, ok: true };
}

function sbeAddKeyframe(clips, id, t, gain) {
  const c = sbeById(clips, id);
  if (!c) return { clips: clips, ok: false, why: 'gone' };
  if (c.locked) return { clips: clips, ok: false, why: 'locked' };
  const w = sbeClipAudio(c);
  const at = Math.max(0, Math.min(w.len, sbeNum(t)));
  const pts = sbeAfx(c, w.len).points.slice();
  // A SECOND POINT ON THE SAME SECOND IS NOT A POINT, it is a discontinuity
  // the envelope cannot express and the NLEs would import out of order.
  for (const pr of pts) {
    if (Math.abs(pr[0] - at) < 1e-3) {
      return { clips: clips, ok: false, why: 'there is already a point here' };
    }
  }
  pts.push([at, Math.max(0, Math.min(1, sbeNum(gain, 1)))]);
  return sbeAfxWrite(clips, id, pts);
}

function sbeMoveKeyframe(clips, id, index, t, gain) {
  const c = sbeById(clips, id);
  if (!c) return { clips: clips, ok: false, why: 'gone' };
  if (c.locked) return { clips: clips, ok: false, why: 'locked' };
  const w = sbeClipAudio(c);
  const pts = sbeAfx(c, w.len).points.slice();
  const i = Math.max(0, Math.min(pts.length - 1, sbeNum(index)));
  if (!pts.length) return { clips: clips, ok: false, why: 'gone' };
  pts[i] = [Math.max(0, Math.min(w.len, sbeNum(t))),
            Math.max(0, Math.min(1, sbeNum(gain)))];
  return sbeAfxWrite(clips, id, pts);
}

function sbeDeleteKeyframe(clips, id, index) {
  const c = sbeById(clips, id);
  if (!c) return { clips: clips, ok: false, why: 'gone' };
  if (c.locked) return { clips: clips, ok: false, why: 'locked' };
  const w = sbeClipAudio(c);
  const pts = sbeAfx(c, w.len).points.slice();
  const i = sbeNum(index);
  if (i < 0 || i >= pts.length) return { clips: clips, ok: false, why: 'gone' };
  pts.splice(i, 1);
  return sbeAfxWrite(clips, id, pts);
}

// AFTER UNLINK, BOTH HALVES ARE FIRST-CLASS. "You can unlock the clips, but
// you cannot unlock the clip and delete the upper part, nor delete the lower
// part of the sound."
//
// DELETING THE STRIP leaves the picture playing SILENT, and that state is
// exactly expressible today: drop the window and mute the clip. Absent
// `audio` alone would mean LINKED, which plays the clip's own sound again —
// the opposite of what was asked for — so the mute is what makes the silence
// real, in the preview, the render and the export alike.
//
// NOT A RIPPLE. Removing a sound must not move the picture it was under, nor
// anything after it: this is one clip's own field, and nothing reflows.
function sbeDeleteStrip(clips, id) {
  const c = sbeById(clips, id);
  if (!c) return { clips: clips, ok: false, why: 'gone' };
  if (c.locked) return { clips: clips, ok: false, why: 'locked' };
  if (sbeKind(c) !== 'video') {
    return { clips: clips, ok: false, why: 'only a video clip has sound' };
  }
  if (sbeClipMuted(c) && !sbeClipAudio(c).split) {
    return { clips: clips, ok: false, why: 'this clip is already silent' };
  }
  const out = clips.map(x => Object.assign({}, x));
  const t = sbeById(out, id);
  delete t.audio;
  delete t.afx;                 // an envelope with nothing to shape
  t.mute = true;
  t.source = 'human';
  return { clips: out, ok: true };
}

// One audio fade, in seconds, on a clip's strip. Mirrors `sbeSetFade` so the
// muscle memory from the picture's corner handle transfers exactly.
function sbeSetAudioFade(clips, id, edge, seconds) {
  const c = sbeById(clips, id);
  if (!c) return { clips: clips, ok: false, why: 'gone' };
  if (c.locked) return { clips: clips, ok: false, why: 'locked' };
  const w = sbeClipAudio(c);
  const len = Math.max(0, w.len);
  const key = (edge === 'out') ? 'fade_out' : 'fade_in';
  const want = Math.max(0, Math.min(len, sbeNum(seconds)));
  const out = clips.map(x => Object.assign({}, x));
  const t = sbeById(out, id);
  const afx = Object.assign({}, t.afx || {});
  if (want > 1e-9) afx[key] = sbeRound(want); else delete afx[key];
  const has = afx.fade_in > 1e-9 || afx.fade_out > 1e-9
              || (afx.points && afx.points.length);
  if (has) t.afx = afx; else delete t.afx;
  t.source = 'human';
  return { clips: out, ok: true };
}

// ---- THE OVERLAY LANE'S MODEL -------------------------------------------
// Overlays do NOT ripple. A card is placed where somebody wants it in the
// finished film, so moving one moves one, and trimming one changes only its
// own window — the picture underneath is not consulted and does not shift.
function sbeOvKind(o) {
  const k = String((o || {}).kind || '').toLowerCase();
  if (k === 'still' || k === 'video' || k === 'text') return k;
  return /\.(png|webp|tiff?)$/i.test(String((o || {}).path || ''))
    ? 'still' : 'video';
}

// A TITLE IS AN OVERLAY WHOSE PIXELS ARE DRAWN. The mirror of
// `overlay_text()`: the same defaults, the same clamps, so the card the stage
// paints and the card the render rasterises are the same card. `font_size`
// is px at a 1080-high frame; `x`/`y` are fractions of the frame.
const SBE_TEXT_DEFAULTS = { font_size: 64, color: '#ffffff', align: 'center',
                            x: 0.5, y: 0.5, box: false, box_color: '#000000',
                            box_opacity: 0.5 };
const SBE_TEXT_REF_H = 1080;
const SBE_TEXT_MAX = 400;
function sbeHexColour(v, d) {
  let s = String(v || '').trim().toLowerCase();
  if (/^#[0-9a-f]{3}$/.test(s)) s = '#' + s.slice(1).split('').map(ch => ch + ch).join('');
  return /^#[0-9a-f]{6}$/.test(s) ? s : d;
}
function sbeOvText(o) {
  const it = o || {};
  const raw = (it.style && typeof it.style === 'object') ? it.style : {};
  const d = SBE_TEXT_DEFAULTS;
  const size = Math.max(8, Math.min(400, sbeNum(raw.font_size, d.font_size) || d.font_size));
  let align = String(raw.align || d.align).toLowerCase();
  if (align !== 'left' && align !== 'center' && align !== 'right') align = d.align;
  const clamp01 = (v, dv) => Math.max(0, Math.min(1, sbeNum(v, dv)));
  return {
    text: String(it.text || '').replace(/\r\n?/g, '\n').slice(0, SBE_TEXT_MAX),
    style: {
      font_size: sbeRound(size), color: sbeHexColour(raw.color, d.color),
      align: align, x: sbeRound(clamp01(raw.x, d.x)), y: sbeRound(clamp01(raw.y, d.y)),
      box: raw.box === true, box_color: sbeHexColour(raw.box_color, d.box_color),
      box_opacity: sbeRound(clamp01(raw.box_opacity, d.box_opacity)),
    },
  };
}

function sbeOvAt(overlays, t) {
  const now = sbeNum(t);
  for (const o of overlays || []) {
    if (now >= sbeNum(o.film_start) - 1e-6 && now < sbeNum(o.film_end)) return o;
  }
  return null;
}

function sbeOvById(overlays, id) {
  for (const o of overlays || []) if (o.id === id) return o;
  return null;
}

// ONE LANE, so a move that would land on top of another card is refused
// rather than silently stacked — the same rule the picture lane lives by.
function sbeOvFits(overlays, id, fs, fe) {
  for (const o of overlays || []) {
    if (o.id === id) continue;
    if (fs < sbeNum(o.film_end) - 1e-6 && fe > sbeNum(o.film_start) + 1e-6) {
      return false;
    }
  }
  return true;
}

function sbeOvMove(overlays, id, filmStart) {
  const o = sbeOvById(overlays, id);
  if (!o) return { overlays: overlays, ok: false, why: 'gone' };
  if (o.locked) return { overlays: overlays, ok: false, why: 'locked' };
  const len = Math.max(SBE_MIN_CLIP, sbeNum(o.film_end) - sbeNum(o.film_start));
  const fs = Math.max(0, sbeNum(filmStart));
  if (!sbeOvFits(overlays, id, fs, fs + len)) {
    return { overlays: overlays, ok: false,
             why: 'another overlay is already there' };
  }
  const out = overlays.map(x => Object.assign({}, x));
  const t = sbeOvById(out, id);
  t.film_start = sbeRound(fs);
  t.film_end = sbeRound(fs + len);
  if (sbeOvKind(t) === 'still') { t.start = 0; t.end = sbeRound(len); }
  t.source = 'human';
  return { overlays: out, ok: true };
}

function sbeOvTrim(overlays, id, edge, filmTime) {
  const o = sbeOvById(overlays, id);
  if (!o) return { overlays: overlays, ok: false, why: 'gone' };
  if (o.locked) return { overlays: overlays, ok: false, why: 'locked' };
  let fs = sbeNum(o.film_start), fe = sbeNum(o.film_end);
  if (edge === 'l') fs = Math.min(fe - SBE_MIN_CLIP, Math.max(0, sbeNum(filmTime)));
  else fe = Math.max(fs + SBE_MIN_CLIP, sbeNum(filmTime));
  if (!sbeOvFits(overlays, id, fs, fe)) {
    return { overlays: overlays, ok: false, why: 'another overlay is there' };
  }
  const out = overlays.map(x => Object.assign({}, x));
  const t = sbeOvById(out, id);
  t.film_start = sbeRound(fs);
  t.film_end = sbeRound(fe);
  // A STILL IS ITS SLOT, the same synthesis a still clip gets — which is why
  // a card is resized by dragging its edges and nothing else.
  if (sbeOvKind(t) === 'still') { t.start = 0; t.end = sbeRound(fe - fs); }
  t.source = 'human';
  return { overlays: out, ok: true };
}

function sbeOvAdd(overlays, item, filmStart) {
  const dur = Math.max(SBE_MIN_CLIP, sbeNum(item.duration_s, 0) || 3);
  let fs = Math.max(0, sbeNum(filmStart));
  // Land somewhere free rather than refusing: the lane is one track, and a
  // card dropped onto another card means "after it".
  for (let i = 0; i < (overlays || []).length + 1; i++) {
    if (sbeOvFits(overlays, null, fs, fs + dur)) break;
    let push = fs;
    for (const o of overlays) {
      if (fs < sbeNum(o.film_end) - 1e-6 && fs + dur > sbeNum(o.film_start) + 1e-6) {
        push = Math.max(push, sbeNum(o.film_end));
      }
    }
    fs = push;
  }
  const kind = (item.kind === 'text') ? 'text'
    : (/\.(png|webp|tiff?)$/i.test(String(item.path || '')) ? 'still' : 'video');
  const o = {
    id: sbeNewId(), kind: kind,
    path: (kind === 'text') ? null : item.path, title: item.title || '',
    start: 0, end: sbeRound(dur),
    film_start: sbeRound(fs), film_end: sbeRound(fs + dur),
    source: 'human', locked: false,
  };
  if (kind === 'text') {
    o.text = String(item.text || 'Title');
    if (item.style && typeof item.style === 'object') o.style = Object.assign({}, item.style);
  }
  return { overlays: (overlays || []).concat([o]), ok: true, added: o };
}

function sbeOvDelete(overlays, id) {
  const o = sbeOvById(overlays, id);
  if (!o) return { overlays: overlays, ok: false, why: 'gone' };
  if (o.locked) return { overlays: overlays, ok: false, why: 'locked' };
  // NOT A RIPPLE. Removing a card must not move the picture under it, nor the
  // next card: the lane is a set of placements, not a queue.
  return { overlays: (overlays || []).filter(x => x.id !== id), ok: true };
}

// ---------------------------------------------------------------------------
// TRANSITIONS — a typed object that OWNS A BOUNDARY, never an overlap
// ---------------------------------------------------------------------------
// The mirror of `resolve_transitions()`. A transition names the OUTGOING clip
// and sits on the cut between it and its successor; the clips' own slots do
// not move, and the render builds the overlap from SOURCE HANDLES — half the
// duration of extra tail past the out-point, half of extra head before the
// in-point. So the one thing the client has to know, and say, is whether
// those handles exist. `sbeTxResolve` answers per boundary, with the same
// sentence the server's validator would refuse the save with.
const SBE_TX_KINDS = ['dissolve', 'fade_black'];
const SBE_TX_MIN = 1 / 24;
const SBE_TX_MAX = 2.0;
const SBE_TX_LABELS = { dissolve: 'Dissolve', fade_black: 'Fade through black' };

function sbeTxById(txs, id) {
  for (const t of txs || []) if (t.id === id) return t;
  return null;
}
function sbeTxAfter(txs, clipId) {
  for (const t of txs || []) if (t.after_clip === clipId) return t;
  return null;
}
function sbeTxDuration(row, outLen, inLen, fps) {
  const d0 = sbeNum((row || {}).duration);
  if (d0 <= 0) return 0;
  const d = Math.max(0, Math.min(d0, SBE_TX_MAX, 0.5 * Math.max(0, Math.min(outLen, inLen))));
  const f = Math.max(1, sbeNum(fps, 24) || 24);
  // EVEN FRAMES — half a side each — the same rule `transition_duration`
  // applies, so the number the inspector shows is the number the film gets.
  const frames = 2 * Math.round(d * f / 2);
  return sbeRound(frames / f);
}
function sbeTxSpare(c, side) {
  if (sbeKind(c) !== 'video') return Infinity;
  if (side === 'head') return Math.max(0, sbeNum(c.start));
  const dur = sbeNum(c.duration, 0);
  if (!(dur > 0)) return null;
  return Math.max(0, dur - sbeNum(c.end));
}
function sbeTxResolve(clips, txs, fps) {
  const order = (clips || []).slice().sort((a, b) =>
    (sbeNum(a.film_start) - sbeNum(b.film_start)) || String(a.path).localeCompare(String(b.path)));
  const pos = {};
  order.forEach((c, k) => { pos[c.id] = k; });
  const seen = {};
  const out = [];
  (txs || []).forEach((row, n) => {
    const label = 'transition ' + (n + 1);
    const res = { id: row.id, after_clip: String(row.after_clip || ''), before_clip: '',
                  kind: String(row.kind || '').toLowerCase(), duration: 0, half: 0,
                  at: 0, problem: null };
    const fail = (code, message) => { res.problem = { code: code, message: message }; out.push(res); };
    const k = pos[res.after_clip];
    if (k === undefined) { fail('transition_unknown_clip', label + ': names no clip on this timeline'); return; }
    const a = order[k];
    if (k + 1 >= order.length) { fail('transition_last_clip', 'this is the last clip — there is nothing after it to dissolve into'); return; }
    const b = order[k + 1];
    res.before_clip = b.id;
    res.at = sbeRound(sbeNum(a.film_end));
    if (seen[res.after_clip]) { fail('transition_duplicate_boundary', 'this cut already has a transition — one per boundary'); return; }
    seen[res.after_clip] = true;
    if (SBE_TX_KINDS.indexOf(res.kind) < 0) { fail('transition_kind', label + ': unknown kind ' + res.kind); return; }
    if (!(sbeNum(row.duration) > 0)) { fail('transition_duration', 'the duration must be above 0 s'); return; }
    const d = sbeTxDuration(row, sbeLen(a), sbeLen(b), fps);
    if (d < SBE_TX_MIN - 1e-9) { fail('transition_duration', 'these two clips are too short to carry a transition between them'); return; }
    res.duration = d;
    res.half = sbeRound(d / 2);
    const word = res.kind.replace('_', ' ');
    const needOut = res.half * sbeSpeed(a), needIn = res.half * sbeSpeed(b);
    const short = [];
    const tail = sbeTxSpare(a, 'tail');
    if (tail === null) short.push('the source length of the outgoing clip is not known — run Prepare so the panel can measure it');
    else if (tail + 1e-6 < needOut) short.push('the outgoing clip has only ' + tail.toFixed(2) + 's beyond its out-point and the ' + word + ' needs ' + needOut.toFixed(2) + 's there — trim its tail in or shorten the transition');
    const head = sbeTxSpare(b, 'head');
    if (head !== null && head + 1e-6 < needIn) short.push('the incoming clip has only ' + head.toFixed(2) + 's before its in-point and the ' + word + ' needs ' + needIn.toFixed(2) + 's there — trim its head in or shorten the transition');
    if (short.length) { fail('transition_no_handles', short.join('; ')); return; }
    out.push(res);
  });
  return out;
}
// The half-durations a clip's two ends give to transitions, for the preview.
function sbeTxEdges(clips, txs, clipId, fps) {
  const e = { head: 0, tail: 0 };
  for (const r of sbeTxResolve(clips, txs, fps)) {
    if (r.problem) continue;
    if (r.after_clip === clipId) e.tail = r.half;
    if (r.before_clip === clipId) e.head = r.half;
  }
  return e;
}
function sbeTxSet(txs, afterId, kind, duration) {
  const out = (txs || []).map(t => Object.assign({}, t));
  let t = sbeTxAfter(out, afterId);
  if (!t) { t = { id: sbeNewId(), after_clip: afterId }; out.push(t); }
  t.kind = (SBE_TX_KINDS.indexOf(kind) >= 0) ? kind : 'dissolve';
  t.duration = sbeRound(Math.max(SBE_TX_MIN, Math.min(SBE_TX_MAX, sbeNum(duration, 0.5) || 0.5)));
  return { transitions: out, ok: true, transition: t };
}
function sbeTxDelete(txs, afterId) {
  const t = sbeTxAfter(txs, afterId);
  if (!t) return { transitions: txs, ok: false, why: 'gone' };
  return { transitions: (txs || []).filter(x => x.after_clip !== afterId), ok: true };
}
// Drops what no longer owns a boundary: a clip that left, or one that became
// the last clip.
function sbeTxPrune(txs, clips) {
  const order = (clips || []).slice().sort((a, b) => sbeNum(a.film_start) - sbeNum(b.film_start));
  const ok = {};
  order.forEach((c, k) => { if (k + 1 < order.length) ok[c.id] = true; });
  const seen = {};
  return (txs || []).filter(t => {
    if (!ok[t.after_clip] || seen[t.after_clip]) return false;
    seen[t.after_clip] = true;
    return true;
  });
}
function sbeTxRepoint(txs, fromId, toId) {
  return (txs || []).map(t => (t.after_clip === fromId)
    ? Object.assign({}, t, { after_clip: toId }) : Object.assign({}, t));
}
// The lane's own door, the same shape as `sbeOvMutate`.
function sbeTxMutate(fn) {
  const before = sbeSnapshot();
  const res = fn(SBE.transitions || []);
  if (!res || res.ok === false) {
    if (res && res.why) phosToast(res.why, {});
    return false;
  }
  SBE.undo.push(before);
  if (SBE.undo.length > SBE_UNDO_MAX) SBE.undo.shift();
  SBE.redo.length = 0;
  SBE.transitions = res.transitions;
  SBE.dirty = true;
  sbeSetState('unsaved changes', 'dirty');
  sbePaint();
  sbeQueueSave();
  return res;
}
function sbeTxCommit(kind, duration) {
  if (!SBE.txSel) return;
  sbeBlurControl();
  const after = SBE.txSel;
  if (!kind || kind === 'none') { sbeTxRemoveSel(); return; }
  sbeTxMutate(ts => sbeTxSet(ts, after, kind, duration));
}
function sbeTxRemoveSel() {
  if (!SBE.txSel) return;
  const after = SBE.txSel;
  if (!sbeTxAfter(SBE.transitions, after)) { sbePaint(); return; }
  sbeTxMutate(ts => sbeTxDelete(ts, after));
}

function sbeStripOwned(c) {
  if (sbeKind(c) !== 'video') return false;
  if (sbeClipMuted(c)) return false;            // muted plays from nowhere
  return !!sbeClipAudio(c).split;
}

function sbePictureCarriesSound(c) {
  return sbeKind(c) === 'video' && !sbeClipMuted(c) && !sbeClipAudio(c).split;
}

// EVERY STRIP AUDIBLE AT FILM SECOND `t`, and the SOURCE second each one is
// at. More than one is the normal case across a split edit — a J-cut is two
// sounds overlapping by construction — so this returns a list and the player
// gives each a voice of its own. The render's concat lane resolves the same
// overlap by trimming the outgoing tail; here they simply sum, which is what
// the person cutting needs to hear.
function sbeStripsAt(clips, t) {
  const now = sbeNum(t);
  const out = [];
  for (const c of clips || []) {
    if (!sbeStripOwned(c)) continue;
    const w = sbeClipAudio(c);
    const from = w.film_start, to = sbeRound(w.film_start + w.len);
    if (now < from - 1e-6 || now >= to) continue;
    out.push({ id: c.id, path: c.path || '',
               at: sbeRound(w.start + (now - from) * w.speed),
               from: from, to: to });
  }
  out.sort((a, b) => (a.from - b.from) || String(a.id).localeCompare(String(b.id)));
  return out;
}

function sbeClipAt(clips, t) {
  for (const c of clips || []) {
    if (t >= sbeNum(c.film_start) && t < sbeNum(c.film_end)) return c;
  }
  return null;
}

// The holes, recomputed locally while dragging. The server reports the same
// list on every read (edit_gaps); this is the live copy so the track does not
// have to round-trip to draw a hole the user just opened.
// HALF A FRAME AT THIS SEQUENCE'S RATE, and it used to be the literal 1/48 —
// which is half a frame at 24 fps and the wrong number at any other. It is
// also why the header said "1 hole · 0.02s" over a film with three holes in
// it: two were under half a frame, so this list did not contain them and
// nothing else was counting. Sub-frame holes cannot exist any more (they are
// closed on load and unreachable by a drag), so what this threshold means now
// is "shorter than half a frame is float noise" — which is the only thing it
// can honestly mean once every hole is a whole number of frames.
function sbeHoles(clips, tolerance) {
  const tol = (tolerance === undefined) ? 0.5 / sbeFps() : tolerance;
  const spans = (clips || []).filter(c => sbeNum(c.film_end) > sbeNum(c.film_start))
    .map(c => [sbeNum(c.film_start), sbeNum(c.film_end)])
    .sort((a, b) => a[0] - b[0]);
  const out = [];
  let cursor = 0;
  for (const s of spans) {
    if (s[0] - cursor > tol) {
      out.push({ film_start: sbeRound(cursor), film_end: sbeRound(s[0]),
                 duration: sbeRound(s[0] - cursor) });
    }
    cursor = Math.max(cursor, s[1]);
  }
  return out;
}

// THE BEAT GRID, AND THE PROMISE NOT TO EXTRAPOLATE IT.
//
// beat_map() fits ONE constant tempo across a SPAN, because real tracks drift
// and a grid fitted over eight minutes is wrong at both ends. Beats outside
// that span do not exist — asking for one gets "no beat here" from the server,
// and inventing one here would put a confident wrong line under someone's cut.
function sbeBeatGrid(beats, from, to, offset) {
  if (!beats) return [];
  const off = sbeNum(offset);
  const span = Array.isArray(beats.span) ? beats.span : null;
  const lo = span ? sbeNum(span[0]) : -Infinity;
  const hi = span ? sbeNum(span[1]) : Infinity;
  const downs = {};
  for (const d of (beats.downbeats || [])) downs[sbeRound(d)] = true;
  const out = [];
  for (const b of (beats.beats || [])) {
    const t = sbeNum(b);
    if (t < lo - 1e-6 || t > hi + 1e-6) continue;       // never past the fit
    const film = sbeRound(t - off);
    if (film < from - 1e-6 || film > to + 1e-6) continue;
    out.push({ t: film, down: !!downs[sbeRound(t)] });
  }
  // A downbeat the beat list does not carry is still a downbeat.
  for (const d of (beats.downbeats || [])) {
    const t = sbeNum(d);
    if (t < lo - 1e-6 || t > hi + 1e-6) continue;
    const film = sbeRound(t - off);
    if (film < from - 1e-6 || film > to + 1e-6) continue;
    if (!out.some(x => Math.abs(x.t - film) < 1e-6)) out.push({ t: film, down: true });
  }
  out.sort((a, b) => a.t - b.t);
  return out;
}

function sbeGridIsAGuess(beats) {
  return !!beats && sbeNum(beats.confidence, 1) < SBE_GUESS_CONFIDENCE;
}

// Nearest beat within `tol` seconds, downbeats winning ties. `enabled` false
// (the checkbox off, or Alt held) returns t untouched — the override has to be
// a straight bypass or it is not an override.
function sbeSnapTime(t, beats, tol, enabled, offset) {
  if (!enabled || !beats) return sbeRound(t);
  const grid = sbeBeatGrid(beats, t - tol, t + tol, offset);
  if (!grid.length) return sbeRound(t);
  let best = null, dist = Infinity;
  for (const g of grid) {
    const d = Math.abs(g.t - t);
    if (d > tol) continue;
    if (d < dist - 1e-9 || (Math.abs(d - dist) < 1e-9 && g.down && (!best || !best.down))) {
      dist = d; best = g;
    }
  }
  return best ? sbeRound(best.t) : sbeRound(t);
}

// ---- the four edits -------------------------------------------------------
// Each takes the clip array and returns {clips, ok, why}. None of them touches
// a locked clip, and every one of them stamps source:'human' — a promise the
// server keeps in the other direction: a later re-plan can leave a human's cut
// alone precisely because it is labelled.
function sbeById(clips, id) {
  for (const c of clips || []) if (c.id === id) return c;
  return null;
}

// HOW A DRAG BEHAVES — the NLE contract, 2026-09-05.
// Premiere, Resolve and After Effects all agree on two defaults, and the owner
// asked for exactly them: dragging a clip's BODY moves that clip and nothing
// else (it slides in the room between its neighbours and stops at them), and
// pulling an EDGE changes that clip's length and leaves everything after it
// where it was — a hole opens or closes, the rest of the film does not move.
// The old behaviour, where every gesture repacked the sequence and slid the
// whole tail, is what those programs call a RIPPLE, and it is still here as a
// modifier: hold ⌘ (or Ctrl) while dragging. Shift still reorders.
function sbeMoveTo(clips, id, filmStart, opts) {
  const ripple = !!(opts && opts.ripple);
  const c = sbeById(clips, id);
  if (!c) return { clips: clips, ok: false, why: 'gone' };
  if (c.locked) return { clips: clips, ok: false, why: 'locked' };
  const len = sbeLen(c);
  const want0 = Math.max(0, sbeNum(filmStart));
  if (!ripple) {
    // Between its neighbours, and the neighbours stay put: this clip's own
    // gap sets where it lands, the next clip's gap is corrected by the same
    // amount so its film position does not change.
    const mark = sbeSyncMark(clips);
    const i = clips.indexOf(c);
    const prev = i > 0 ? clips[i - 1] : null;
    const next = (i + 1 < clips.length) ? clips[i + 1] : null;
    const floor = prev ? sbeNum(prev.film_end) : 0;
    const ceil = next ? sbeNum(next.film_start) - len : Infinity;
    if (ceil < floor - 1e-9) return { clips: clips, ok: false, why: 'tight' };
    const want = Math.max(floor, Math.min(ceil, want0));
    const nextStart = next ? sbeNum(next.film_start) : null;
    c._gap = Math.max(0, sbeRound(want - floor));
    if (next && !next.locked) next._gap = Math.max(0, sbeRound(nextStart - (want + len)));
    sbeLayout(clips);
    sbeSyncCarry(clips, mark, [id]);
    c.source = 'human';
    return { clips: clips, ok: true };
  }
  const want = want0;
  const centre = want + len / 2;
  const mark = sbeSyncMark(clips);
  const rest = clips.filter(x => x !== c);
  let idx = 0;
  for (const x of rest) {
    if (sbeNum(x.film_start) + sbeLen(x) / 2 < centre) idx++; else break;
  }
  rest.splice(idx, 0, c);
  c._gap = 0;
  sbeLayout(rest);                       // where would it land packed tight?
  const prev = rest[rest.indexOf(c) - 1];
  const floor = prev ? sbeNum(prev.film_end) : 0;
  c._gap = Math.max(0, sbeRound(want - floor));
  sbeLayout(rest);
  // The clip being dragged is exempt: its sound stays exactly where it is,
  // which is the J-cut. Everything the reflow pushed takes its sound along.
  sbeSyncCarry(rest, mark, [id]);
  c.source = 'human';
  return { clips: rest, ok: true };
}

// Both handles follow the pointer, which is the property that makes a timeline
// feel like a timeline. The left one moves the clip's in-point AND its slot, so
// the tail does not move and a hole opens behind it (holes are legal here —
// they are what the generate control fills). The right one changes the length,
// so everything after it ripples.
function sbeTrim(clips, id, edge, filmTime, opts) {
  // `opts.ripple` (⌘ / Ctrl held): the tail slides, as it did before
  // 2026-09-05. Default: the other clips do not move — see sbeMoveTo.
  const ripple = !!(opts && opts.ripple);
  const c = sbeById(clips, id);
  if (!c) return { clips: clips, ok: false, why: 'gone' };
  if (c.locked) return { clips: clips, ok: false, why: 'locked' };
  const mark = sbeSyncMark(clips);
  // A DRAG IS FILM SECONDS; THE WINDOW IS SOURCE SECONDS. At 2x a handle
  // pulled one second along the film moves the in- or out-point two seconds
  // of the take, which is what keeps the slot following the pointer.
  const sp = sbeSpeed(c);
  if (edge === 'l') {
    const d = (sbeNum(filmTime) - sbeNum(c.film_start)) * sp;
    let s = sbeNum(c.start) + d;
    s = Math.max(0, Math.min(sbeNum(c.end) - SBE_MIN_CLIP, s));
    // A HEAD TRIM MOVES THE SLOT WITH THE IN-POINT OR IT IS NOT A TRIM. The
    // lead gap used to clamp at zero on its own, so pulling the head of a
    // butt-joined clip further open moved `start` while `film_start` stood
    // still: the picture SLIPPED inside its slot, the clip grew to the right
    // instead of the left (against this function's own contract, "the tail
    // does not move"), and an unlinked sound was left describing a frame that
    // no longer plays under it. Trimming may not change the film→source
    // mapping of the frames it keeps; that invariant is what makes "the sound
    // stays put" correct rather than lucky.
    const room = Math.max(0, sbeNum(c._gap)) * sp;
    if (!ripple && s < sbeNum(c.start) - room) s = sbeRound(sbeNum(c.start) - room);
    const applied = s - sbeNum(c.start);
    if (Math.abs(applied) < 1e-9) return { clips: clips, ok: false, why: 'edge' };
    c.start = sbeRound(s);
    // Ripple on the head: the head stays where the previous clip ends and
    // everything after slides by the length change (the gap is left alone).
    if (!ripple) c._gap = Math.max(0, sbeRound(sbeNum(c._gap) + applied / sp));
  } else {
    const d = (sbeNum(filmTime) - sbeNum(c.film_end)) * sp;
    let e = sbeNum(c.end) + d;
    const srcDur = sbeNum(c.duration, 0);
    if (srcDur > 0) e = Math.min(srcDur, e);
    e = Math.max(sbeNum(c.start) + SBE_MIN_CLIP, e);
    const i = clips.indexOf(c);
    const next = (i + 1 < clips.length) ? clips[i + 1] : null;
    if (!ripple && next) {
      // The clip may grow into the hole after it, never into the next clip.
      const room = Math.max(0, sbeNum(next._gap)) * sp;
      e = Math.min(e, sbeNum(c.end) + room);
    }
    if (Math.abs(e - sbeNum(c.end)) < 1e-9) return { clips: clips, ok: false, why: 'edge' };
    const grew = (e - sbeNum(c.end)) / sp;          // film seconds
    c.end = sbeRound(e);
    // Default: the next clip does not move — its gap takes the difference.
    if (!ripple && next && !next.locked) next._gap = Math.max(0, sbeRound(sbeNum(next._gap) - grew));
  }
  c.source = 'human';
  sbeLayout(clips);
  // The trimmed clip keeps its own mapping by construction (the head moves the
  // slot with the in-point, the tail moves neither), so its sound must NOT be
  // touched. A tail trim ripples everything after it, and those take theirs.
  sbeSyncCarry(clips, mark, [id]);
  return { clips: clips, ok: true };
}

// DUPLICATE: the same shot again, right after itself, with everything it
// carries — window, speed, fades, grade, mute — and its sound LINKED: a
// copied J-cut strip would sit under the original's seconds, which is the
// one overlap the sound lane refuses. The copy is its own clip from then on.
function sbeDuplicate(clips, id) {
  const c = sbeById(clips, id);
  if (!c) return { clips: clips, ok: false, why: 'gone' };
  const mark = sbeSyncMark(clips);
  const b = JSON.parse(JSON.stringify(c));
  for (const k of Object.keys(b)) if (k.charAt(0) === '_') delete b[k];
  b.id = sbeNewId();
  delete b.audio;
  b.locked = false;
  b.source = 'human';
  b._gap = 0;
  // NAMED AS A COPY, or two identical blocks side by side cannot be told
  // apart — the review pressed D twice and could not say which was which.
  const base = String(c.title || String(c.path || '').split('/').pop() || 'clip').replace(/ \(copy( \d+)?\)$/, '');
  b.title = base + ' (copy)';
  const out = clips.slice();
  out.splice(out.indexOf(c) + 1, 0, b);
  sbeLayout(out);
  sbeSyncCarry(out, mark, [b.id]);
  return { clips: out, ok: true, added: b };
}

function sbeRippleDelete(clips, id) {
  const c = sbeById(clips, id);
  if (!c) return { clips: clips, ok: false, why: 'gone' };
  if (c.locked) return { clips: clips, ok: false, why: 'locked' };
  const mark = sbeSyncMark(clips);
  const out = clips.filter(x => x !== c);   // its lead gap leaves with it
  sbeLayout(out);
  sbeSyncCarry(out, mark, []);              // nothing here is a deliberate slide
  return { clips: out, ok: true, removed: c };
}

// LIFT: take the shot out and LEAVE ITS HOLE. Every clip after it stays
// exactly where it was. This is what an editor means by "delete" most of
// the time — the shot is wrong, the cut around it is right — and it is the
// NLE default (Delete lifts, Shift+Delete ripples). The hole is a slug, a
// real object on the timeline: black, no file, same window. It can be filled
// from the pool, trimmed, or ripple-deleted later when the cut is ready to
// close up. A ripple here would move every downstream cut off its beat.
function sbeLiftDelete(clips, id) {
  const c = sbeById(clips, id);
  if (!c) return { clips: clips, ok: false, why: 'gone' };
  if (c.locked) return { clips: clips, ok: false, why: 'locked' };
  if (sbeKind(c) === 'slug') return { clips: clips, ok: false, why: 'already a hole' };
  const len = sbeLen(c);
  const out = clips.map(x => (x === c) ? {
    id: c.id, kind: 'slug', path: null, proxy: null, duration: null,
    start: 0, end: sbeRound(len),
    film_start: c.film_start, film_end: c.film_end,
    source: 'human', locked: false, _gap: c._gap,
  } : Object.assign({}, x));
  sbeLayout(out);
  return { clips: out, ok: true, removed: c };
}

function sbeSplitAt(clips, t, newId, transitions) {
  const c = sbeClipAt(clips, t);
  if (!c) return { clips: clips, ok: false, why: 'nothing there' };
  if (c.locked) return { clips: clips, ok: false, why: 'locked' };
  // `off` is FILM seconds into the slot; the cut in the take is `off * speed`.
  const off = sbeNum(t) - sbeNum(c.film_start);
  const sp = sbeSpeed(c);
  if (off < SBE_MIN_CLIP || sbeLen(c) - off < SBE_MIN_CLIP) {
    return { clips: clips, ok: false, why: 'too close to an edge' };
  }
  // THE SOUND IS CUT WHERE THE PICTURE IS, NOT COPIED TWICE. The deep copy
  // below used to carry `audio` into the new half verbatim, so one strip became
  // two claiming the same seconds of the same take — which the server refuses
  // as `clips_audio_overlap`, and which would have played the line twice if it
  // had not. The cut is expressed in the SOURCE clock both halves share, so
  // each half comes out of the split at the drift it went in with.
  const w = sbeClipAudio(c);
  let cutA = null, cutB = null;
  if (w.split) {
    const sp2 = sbeRound(sbeNum(c.start) + off * sp);
    if (sp2 < w.start + SBE_MIN_CLIP || sp2 > w.end - SBE_MIN_CLIP) {
      // The strip has been trimmed to a window this cut falls outside, so one
      // half would end up with no sound at all — a state the document cannot
      // express (an absent `audio` means LINKED, which would invent sound the
      // film does not have). Say so instead of guessing.
      return { clips: clips, ok: false,
               why: 'the unlinked sound does not reach this cut — move the '
                    + 'strip back or re-link it first' };
    }
    cutA = sbeAudioField({ start: w.start, end: sp2,
                           film_start: w.film_start }, w.coupled);
    cutB = sbeAudioField({ start: sp2, end: w.end,
                           film_start: w.film_start + (sp2 - w.start) / sp },
                         w.coupled);
  }
  const mark = sbeSyncMark(clips);
  const b = JSON.parse(JSON.stringify(c));
  b.id = newId || sbeNewId();
  b.start = sbeRound(sbeNum(c.start) + off * sp);
  b._gap = 0;
  b.source = 'human';
  c.end = b.start;
  c.source = 'human';
  if (cutA) { c.audio = cutA; b.audio = cutB; } else { delete b.audio; }
  const out = clips.slice();
  out.splice(out.indexOf(c) + 1, 0, b);
  sbeLayout(out);
  // Both halves were just given the position they are meant to have; the
  // reflow only ever moves what is AFTER them.
  sbeSyncCarry(out, mark, [c.id, b.id]);
  const res = { clips: out, ok: true, added: b };
  // A TRANSITION AFTER THE SPLIT CLIP NOW FOLLOWS ITS SECOND HALF — that is
  // the half that owns the boundary the transition was on. Left on the first
  // half it would move to the fresh cut in the middle of the shot.
  if (transitions) res.transitions = sbeTxRepoint(transitions, c.id, b.id);
  return res;
}

function sbeNewId() {
  return 'k' + Math.random().toString(16).slice(2, 10) +
         (Date.now() % 65536).toString(16);
}

// A clip the board has rendered but the timeline has never seen. `slot` is
// where the person who ordered it wanted it — carried on the board by
// edit/generate — so a shot generated for a hole lands in that hole rather than
// at the end of the film.
function sbePlaceUnplaced(clips, item, filmStart) {
  const dur = Math.max(SBE_MIN_CLIP, sbeNum(item.duration_s, 0)
                       || sbeNum((item.slot || {}).duration, 0) || 5);
  const c = {
    id: sbeNewId(), path: item.path, proxy: item.proxy || null,
    start: 0, end: sbeRound(dur), film_start: 0, film_end: 0,
    source: 'human', locked: false, n: item.n,
    title: item.title || '', duration: sbeNum(item.duration_s, 0) || null,
  };
  const want = Math.max(0, sbeNum(filmStart));
  const mark = sbeSyncMark(clips);
  const out = clips.slice();
  let idx = out.length;
  for (let i = 0; i < out.length; i++) {
    if (sbeNum(out[i].film_start) >= want - 1e-6) { idx = i; break; }
  }
  // Where the clip AFTER the hole is standing, before any of this. Filling a
  // hole must not move the rest of the film: a shot was generated for this
  // slot precisely so the cuts around it would stay on their beats. Every
  // other operation here ripples; this one does not, and that is the whole
  // difference between "fill the hole" and "insert a clip".
  const follower = out[idx] || null;
  const pinned = follower ? sbeNum(follower.film_start) : null;
  out.splice(idx, 0, c);
  c._gap = 0;
  sbeLayout(out);
  const prev = out[out.indexOf(c) - 1];
  c._gap = Math.max(0, sbeRound(want - (prev ? sbeNum(prev.film_end) : 0)));
  sbeLayout(out);
  if (follower && pinned !== null) {
    // If the new clip overruns where the follower stood there is no room to
    // keep the promise, and the tail rides along rather than overlapping.
    follower._gap = Math.max(0, sbeRound(pinned - sbeNum(c.film_end)));
    sbeLayout(out);
  }
  sbeSyncCarry(out, mark, []);
  return { clips: out, ok: true, added: c };
}

// ---------------------------------------------------------------------------
// THE THREE KINDS
// ---------------------------------------------------------------------------
// ABSENT IS VIDEO, on the client for the same reason it is on the server: every
// clip in every edit.json written before today has no `kind`, and every one of
// them is a video. Reading the default rather than stamping it means nothing
// has to be rewritten and a v1 document is correct the moment it loads.
function sbeKind(c) {
  const k = String((c && c.kind) || '').toLowerCase();
  if (k === 'still' || k === 'slug' || k === 'video') return k;
  // ABSENT NO LONGER MEANS VIDEO ON ITS OWN — see clip_kind() in
  // storyboard_editor.py. A pool image can reach the picture lane unstamped,
  // and answering "video" for a .png handed a still to a <video> element:
  // format error, black stage, and the same mistake in the render. The
  // overlay lane has always read the suffix here; both lanes now agree.
  return /\.(png|jpe?g|webp|tiff?)$/i.test(String((c && c.path) || ''))
    ? 'still' : 'video';
}

function sbeBright(c) {
  const a = (c && c.adjust) || {};
  return Math.max(-SBE_BRIGHT_MAX, Math.min(SBE_BRIGHT_MAX, sbeNum(a.brightness, 0)));
}

// THE PREVIEW IS AN APPROXIMATION AND SAYS SO. ffmpeg's `eq=brightness` is an
// ADDITIVE offset; CSS `filter: brightness()` is MULTIPLICATIVE, and CSS has no
// additive form. So the two are matched where a viewer judges exposure — at
// mid-grey, where `0.5 + b` and `0.5 * (1 + 2b)` are the same number. The ends
// drift, the badge says approximate, and the render is the exact one.
function sbeBrightnessCss(b) {
  return Math.max(0, sbeRound(1 + 2 * sbeNum(b)));
}

// FRAMING — the mirror of `clip_frame()`: zoom 1–3 and the anchor as
// fractions, clamped, neutral when absent. A slug has nothing to reframe.
const SBE_FRAME_ZOOM_MAX = 3.0;
function sbeFraming(c) {
  const f = (c && c.frame && typeof c.frame === 'object') ? c.frame : {};
  let z = sbeNum(f.zoom, 1);
  if (!(z > 0)) z = 1;
  z = Math.max(1, Math.min(SBE_FRAME_ZOOM_MAX, z));
  const cl = (v) => Math.max(0, Math.min(1, sbeNum(v, 0.5)));
  return { zoom: sbeRound(z), x: sbeRound(cl(f.x)), y: sbeRound(cl(f.y)) };
}
function sbeFramingIsNeutral(c) { return Math.abs(sbeFraming(c).zoom - 1) < 1e-9; }
function sbeSetFraming(clips, id, field, v) {
  const c = sbeById(clips, id);
  if (!c) return { clips: clips, ok: false, why: 'gone' };
  if (c.locked) return { clips: clips, ok: false, why: 'locked' };
  if (sbeKind(c) === 'slug') return { clips: clips, ok: false, why: 'black has nothing to reframe' };
  const cur = sbeFraming(c);
  const next = Object.assign({}, cur);
  next[field] = sbeNum(v, cur[field]);
  const fr = sbeFraming({ frame: next });
  if (Math.abs(fr.zoom - cur.zoom) < 1e-9 && Math.abs(fr.x - cur.x) < 1e-9 && Math.abs(fr.y - cur.y) < 1e-9) {
    return { clips: clips, ok: false, why: '' };
  }
  const out = clips.map(x => Object.assign({}, x));
  const t = sbeById(out, id);
  // NEUTRAL IS ABSENT: zoom 1 is no field, whatever the anchor says.
  if (Math.abs(fr.zoom - 1) < 1e-9) delete t.frame; else t.frame = fr;
  t.source = 'human';
  return { clips: out, ok: true };
}

function sbeSetBrightness(clips, id, v) {
  const c = sbeById(clips, id);
  if (!c) return { clips: clips, ok: false, why: 'gone' };
  const b = Math.max(-SBE_BRIGHT_MAX, Math.min(SBE_BRIGHT_MAX, sbeNum(v)));
  if (Math.abs(b - sbeBright(c)) < 1e-6) return { clips: clips, ok: false, why: 'unchanged' };
  // NEUTRAL IS ABSENT, exactly as the server normalises it: dragging back to
  // zero must leave a clip identical to one nobody ever touched, or every clip
  // anybody ever selected carries a dead field for the rest of its life.
  if (Math.abs(b) < 1e-6) delete c.adjust;
  else c.adjust = Object.assign({}, c.adjust || {}, { brightness: sbeRound(b) });
  c.source = 'human';
  return { clips: clips, ok: true };
}

// WHERE A DROP LANDS. The midpoint rule every NLE uses: on the left half of a
// clip means before it, on the right half means after it — which is the only
// reading that makes a drop exactly on a boundary unambiguous.
function sbeDropIndex(clips, filmTime) {
  const want = Math.max(0, sbeNum(filmTime));
  for (let i = 0; i < (clips || []).length; i++) {
    if (want < sbeNum(clips[i].film_start) + sbeLen(clips[i]) / 2) return i;
  }
  return (clips || []).length;
}

// INSERT WITH RIPPLE — the opposite of sbePlaceUnplaced, and both are right.
// Filling a hole must NOT move the film: a shot was generated for that slot
// precisely so the cuts around it stayed on their beats. Dropping a NEW clip
// between two others must move it, or the drop silently overwrites whatever it
// landed on.
function sbeInsertAt(clips, item, filmTime) {
  const kind = String(item.kind || 'video');
  const dur = Math.max(SBE_MIN_CLIP,
                       sbeNum(item.duration_s, 0) || sbeNum(item.duration, 0) || 5);
  const c = {
    id: sbeNewId(),
    path: (kind === 'slug') ? null : (item.path || ''),
    proxy: item.proxy || null,
    start: 0, end: sbeRound(dur), film_start: 0, film_end: 0,
    source: 'human', locked: false,
    title: item.title || '',
    // A still and a slug have no source clock, so they have no source
    // duration — and a `duration` left over from a video would clamp the trim
    // that is the only way to change how long they hold.
    duration: (kind === 'video') ? (sbeNum(item.duration_s, 0) || null) : null,
  };
  if (kind !== 'video') c.kind = kind;
  if (item.n !== undefined && item.n !== null) c.n = item.n;
  const mark = sbeSyncMark(clips);
  const out = clips.slice();
  const idx = sbeDropIndex(out, filmTime);
  out.splice(idx, 0, c);
  c._gap = 0;                       // hard against its new neighbour
  sbeLayout(out);                   // everything after it rides along
  sbeSyncCarry(out, mark, []);      // ...sound included
  return { clips: out, ok: true, added: c, index: idx };
}

// REORDER: a ripple delete and an insert, in one gesture. sbeMoveTo puts a clip
// at a TIME and leaves a hole where it was; this puts it at a POSITION and
// closes the hole behind it, which is what "drag to reorder" means everywhere
// else. Both are on the same drag, told apart by Shift, because both are things
// people want and neither can be inferred from the pointer.
function sbeReorderTo(clips, id, filmTime) {
  const c = sbeById(clips, id);
  if (!c) return { clips: clips, ok: false, why: 'gone' };
  if (c.locked) return { clips: clips, ok: false, why: 'locked' };
  const mark = sbeSyncMark(clips);
  const rest = clips.filter(x => x !== c);
  const idx = sbeDropIndex(rest, filmTime);
  rest.splice(idx, 0, c);
  c._gap = 0;
  sbeLayout(rest);
  sbeSyncCarry(rest, mark, [id]);   // the clip being dragged keeps its J-cut
  c.source = 'human';
  return { clips: rest, ok: true, index: idx };
}

// ---- the save payload -----------------------------------------------------
// Transient client bookkeeping (`_gap`, `_pin`) is stripped: it is derivable
// from what is saved, and a field the server does not know is a field that
// outlives the reason it existed.
function sbeCleanClip(c) {
  const out = {};
  for (const k of Object.keys(c)) {
    if (k.charAt(0) === '_') continue;
    out[k] = c[k];
  }
  out.start = sbeRound(out.start);
  out.end = sbeRound(out.end);
  out.film_start = sbeRound(out.film_start);
  out.film_end = sbeRound(out.film_end);
  out.locked = !!out.locked;
  out.source = (out.source === 'human') ? 'human' : 'auto';
  const kind = sbeKind(c);
  if (kind === 'video') {
    delete out.kind;
  } else {
    out.kind = kind;
    // SYNTHESISED, the same way normalise_edit synthesises them, so the
    // document the client sends and the document the server writes back are
    // the same document. Trimming a still moves its slot; the window follows.
    out.start = 0;
    out.end = sbeRound(Math.max(0, out.film_end - out.film_start));
    out.duration = null;
    if (kind === 'slug') { out.path = null; out.proxy = null; }
  }
  const b = sbeBright(out);
  if (Math.abs(b) < 1e-6) delete out.adjust;
  else out.adjust = { brightness: sbeRound(b) };
  // 1x is the absence of the field, the same way the server writes it.
  const sp = sbeSpeed(c);
  if (kind !== 'video' || Math.abs(sp - 1) < 1e-9) delete out.speed;
  else out.speed = sp;
  if (kind === 'slug' || sbeFramingIsNeutral(c)) delete out.frame;
  else out.frame = sbeFraming(c);
  return out;
}

function sbeSaveBody(state) {
  const edit = Object.assign({}, state.edit || {});
  edit.clips = (state.clips || []).map(sbeCleanClip);
  // The lane travels with the document. Client-only bookkeeping is stripped
  // the same way a clip's is.
  edit.overlays = (state.overlays || []).map(o => {
    const out = {};
    for (const k of Object.keys(o)) if (k.charAt(0) !== '_') out[k] = o[k];
    return out;
  });
  // THE BOUNDARIES TRAVEL TOO. An empty list is the server's absent key.
  edit.transitions = (state.transitions || []).map(t => Object.assign({}, t));
  edit.board_id = state.id;
  const body = { id: state.id, edit: edit };
  if (state.expect !== null && state.expect !== undefined) body.expect_revision = state.expect;
  return body;
}

// The server answers a bad save with EVERY error at once and writes nothing, so
// the client can light up every offending clip in one pass instead of playing
// whack-a-mole. `where` is the index in the array we sent.
function sbeErrorsByClip(errors, order) {
  const out = { doc: [], byId: {} };
  for (const e of errors || []) {
    const w = e.where;
    if (w === null || w === undefined || !order || !order.length || !order[w]) {
      out.doc.push(e);
      continue;
    }
    const id = order[w];
    (out.byId[id] = out.byId[id] || []).push(e);
  }
  return out;
}

// Interleaved int16 over `scale`, which is how a five-minute track arrives as
// 326 KB instead of 85 MB of Float32Array. The client draws; it never decodes.
function sbeDecodePeaks(doc) {
  if (!doc || !Array.isArray(doc.peaks)) return null;
  const scale = sbeNum(doc.scale, 127) || 127;
  const n = Math.min(sbeNum(doc.count, 0) || (doc.peaks.length >> 1),
                     doc.peaks.length >> 1);
  const lo = new Array(n), hi = new Array(n);
  for (let i = 0; i < n; i++) {
    lo[i] = sbeNum(doc.peaks[i * 2]) / scale;
    hi[i] = sbeNum(doc.peaks[i * 2 + 1]) / scale;
  }
  return { count: n, lo: lo, hi: hi,
           rate: sbeNum(doc.buckets_per_second, 100) || 100,
           duration: sbeNum(doc.duration, 0) };
}

function sbeFmtTime(t) {
  t = Math.max(0, sbeNum(t));
  const m = Math.floor(t / 60);
  const s = t - m * 60;
  return m + ':' + (s < 10 ? '0' : '') + s.toFixed(2);
}

// ---------------------------------------------------------------------------
// PAN AND ZOOM — the two sliders, as arithmetic
// ---------------------------------------------------------------------------
// Everything here is pure: seconds and pixels in, pixels out. The DOM callers
// are sbeZoomTo / sbeZoomSlide / sbePaintHead, and the reason these are
// separate is that "zooming keeps the playhead where it was" is the one part
// of a timeline nobody can check by eye and everybody notices when it is
// wrong — so it is checked in node instead. See test_storyboard_editor_ui.py.

// The scale at which the WHOLE film is inside the window. This is the
// minimum of the zoom slider, which is why a 71.6s film can never again be
// longer than the box it is drawn in.
function sbeZoomFitPps(span, viewport) {
  const w = Math.max(80, sbeNum(viewport) - SBE_TL_PAD);
  const s = Math.max(0.1, sbeNum(span));
  return Math.max(SBE_PPS_FLOOR, w / s);
}

// The slider is logarithmic. Linear px/sec spends four fifths of its travel
// between 100 and 200 px/sec, where nothing changes, and crosses the useful
// range — a whole film to a single cut — in the first inch.
function sbeZoomFromSlider(v, minPps, maxPps) {
  const lo = Math.max(SBE_PPS_FLOOR, sbeNum(minPps, 1));
  const hi = Math.max(lo * 1.0001, sbeNum(maxPps, SBE_PPS_MAX));
  const f = Math.max(0, Math.min(1, sbeNum(v) / SBE_ZOOM_TICKS));
  return lo * Math.pow(hi / lo, f);
}

function sbeZoomToSlider(pps, minPps, maxPps) {
  const lo = Math.max(SBE_PPS_FLOOR, sbeNum(minPps, 1));
  const hi = Math.max(lo * 1.0001, sbeNum(maxPps, SBE_PPS_MAX));
  const p = Math.max(lo, Math.min(hi, sbeNum(pps, lo)));
  return Math.round(Math.log(p / lo) / Math.log(hi / lo) * SBE_ZOOM_TICKS);
}

// WHAT THE VIEW HOLDS STILL WHILE THE SCALE CHANGES. The playhead if it is on
// screen — that is the frame you are looking at and the one an NLE anchors on
// — and the middle of the view if it is not.
function sbeZoomAnchor(playhead, scrollLeft, viewport, pps, at) {
  if (at !== undefined && at !== null) {
    return { t: Math.max(0, sbeNum(at)), px: sbeNum(at) * sbeNum(pps) - sbeNum(scrollLeft) };
  }
  const px = sbeNum(playhead) * sbeNum(pps) - sbeNum(scrollLeft);
  if (px >= 0 && px <= sbeNum(viewport)) return { t: Math.max(0, sbeNum(playhead)), px: px };
  const mid = sbeNum(viewport) / 2;
  return { t: Math.max(0, (sbeNum(scrollLeft) + mid) / Math.max(1e-6, sbeNum(pps))), px: mid };
}

// Where the scroller has to sit for `anchor.t` to land back under
// `anchor.px` at the new scale, clamped to what there is to scroll.
function sbeZoomScroll(anchor, pps, maxScroll) {
  const want = sbeNum(anchor.t) * sbeNum(pps) - sbeNum(anchor.px);
  return Math.max(0, Math.min(Math.max(0, sbeNum(maxScroll)), want));
}

// PAGE, DO NOT CHASE. A view that re-centres on every frame fights the user's
// own panning and makes the track feel like it is sliding out from under the
// pointer; Resolve pages when the head crosses the edge, so this does too —
// one jump, then the head walks across a fresh screenful. Returns the CURRENT
// scroll unchanged while the head is still on screen, which is how the caller
// knows to leave the DOM alone.
function sbeFollowScroll(headPx, scrollLeft, viewport, maxScroll) {
  const sl = Math.max(0, sbeNum(scrollLeft));
  const w = Math.max(1, sbeNum(viewport));
  const x = sbeNum(headPx);
  if (x >= sl && x <= sl + w - 2) return sl;
  const lead = w * 0.12;
  return Math.max(0, Math.min(Math.max(0, sbeNum(maxScroll)), x - lead));
}

// ---------------------------------------------------------------------------
// THE MONITOR ROW — how two 16:9 pictures fill one strip
// ---------------------------------------------------------------------------
// A 16:9 pair at the asked-for 40/60 needs 2.96 px of width for every px of
// height. No window this panel runs in has that much of both: measured at
// 1440x900, the cut column is 1110px wide and has ~307px of height to spare,
// and 307px of height buys a pair 886px wide. The other 224px is the rail —
// the inspector and the unplaced strip, moved out of the vertical stack and
// into the space the monitors cannot reach. That is what makes "fill the
// width" true instead of aspirational.
//
//   * Height first: the monitors get the whole vertical budget, because
//     height is the scarce thing and a bigger picture is what was asked for.
//   * Then the rail gets what the pair leaves, clamped. If the leftover is
//     wider than the clamp, the SOURCE widens toward the program — up to
//     equal monitors and no further, since the program is the one being cut.
//   * If the width runs out first (a tall, narrow window), the pair takes the
//     width-derived height at exactly 40/60 and the leftover HEIGHT goes to
//     the timeline, which is the row that grows.
function sbeMonitorFit(width, budget, opts) {
  const o = opts || {};
  const gap = sbeNum(o.gap, SBE_MON_GAP);
  const minH = sbeNum(o.minH, SBE_MON_MIN_H);
  const pref = sbeNum(o.ratio, SBE_MON_RATIO);
  const maxR = sbeNum(o.maxRatio, SBE_MON_RATIO_MAX);
  const railMin = sbeNum(o.railMin, SBE_RAIL_MIN);
  const railMax = sbeNum(o.railMax, SBE_RAIL_MAX);
  const A = 16 / 9;
  const total = Math.max(2 * minH * A + railMin + 2 * gap, sbeNum(width));
  const cap = Math.max(minH, sbeNum(budget));
  // The most the pair may ever take, and the height at which 40/60 fills it.
  const pairMax = total - railMin - 2 * gap;
  const wide = (pairMax - gap) / (A * (1 + pref));
  let h, r;
  if (wide <= cap) { h = wide; r = pref; }
  else {
    h = cap;
    // Widen the source only as far as the rail's own maximum allows: past
    // that the rail stops being leftover and starts being a panel.
    const room = total - railMax - 2 * gap - gap;
    r = Math.max(pref, Math.min(maxR, room / (A * h) - 1));
  }
  const progW = h * A;
  const srcW = h * r * A;
  const rail = Math.max(railMin, Math.min(railMax, total - progW - srcW - 2 * gap));
  return { progH: h, progW: progW, srcH: h * r, srcW: srcW, ratio: r,
           rail: rail, gap: gap, total: progW + srcW + rail + 2 * gap };
}

// ---------------------------------------------------------------------------
// THE TIMELINE'S HEIGHT — one number, dragged from its top edge
// ---------------------------------------------------------------------------
// "The timeline is too constricted and cannot be expanded vertically. It needs
// to allow you to drag the upper side of the timeline, which will change the
// layout a little bit, enabling expansion in case you have some sound editing
// in there."
//
// Nothing here resizes an element. The drag moves ONE number — how much of the
// cut column the timeline is entitled to — and sbeFitMonitors sizes the
// monitors off whatever is left, which is the arithmetic that was already
// there. So up is "more track, smaller picture", down is its exact inverse,
// and no two boxes can ever overlap or leave a gutter between them.

// What the layout will allow, given what the window has left. Both ends are
// hard: the timeline may never be shorter than its own contents, and it may
// never take so much that the monitors stop being monitors — `max` is
// sbeFitMonitors' measurement of that, and it is clamped again here so a
// caller that has not measured yet cannot ask for a screenful.
function sbeTlClamp(want, max) {
  const hi = Math.max(SBE_TL_MIN_H,
                      Math.min(SBE_TL_MAX_H, sbeNum(max, SBE_TL_MAX_H)));
  return Math.round(Math.max(SBE_TL_MIN_H,
                             Math.min(hi, sbeNum(want, SBE_TL_MIN_H))));
}

// WHERE A DRAGGED PIXEL LANDS. Every lane starts at the height it has always
// had and takes its share of what the drag added, up to its own cap; whatever
// a capped lane cannot take is offered again to the ones still growing, so the
// height is spent rather than lost. Pure: px in, px out, no DOM — the
// distribution is the whole feature and it is not checkable by eye.
function sbeLaneHeights(tlH) {
  const out = { ruler: 18 };
  for (const L of SBE_LANES) out[L.key] = L.base;
  let left = Math.max(0, Math.round(sbeNum(tlH, SBE_TL_MIN_H)) - SBE_TL_MIN_H);
  // Four passes is one more than the number of lanes that can cap while
  // another still has room, so this terminates with the height spent or every
  // lane full. A `while` here would be a loop whose bound is an argument.
  for (let pass = 0; pass < 4 && left > 0.5; pass++) {
    let share = 0;
    for (const L of SBE_LANES) if (out[L.key] < L.cap) share += L.share;
    if (share <= 0) break;
    const pool = left;
    for (const L of SBE_LANES) {
      if (out[L.key] >= L.cap) continue;
      const take = Math.min(pool * (L.share / share), L.cap - out[L.key]);
      out[L.key] += take;
      left -= take;
    }
  }
  for (const L of SBE_LANES) out[L.key] = Math.round(out[L.key]);
  return out;
}

// THE PREFERENCE IS THIS BROWSER'S, NOT THE FILM'S. It never goes near
// edit.json: a window height is not sequence data, writing it there would bump
// the document's revision every time somebody dragged an edge, put one
// machine's screen into a file two other surfaces render, and race the
// snapshot lane for a number no renderer will ever read. localStorage is where
// every other view preference in this panel already lives — `sbeMuted`, the
// open document, the workflow tab.
function sbeTlPrefRead() {
  let v = NaN;
  try { v = parseInt(localStorage.getItem('phos_sbe_tl_h') || '', 10); } catch (e) {}
  return (v === v) ? sbeTlClamp(v, SBE_TL_MAX_H) : SBE_TL_MIN_H;
}
function sbeTlPrefWrite(px) {
  try {
    localStorage.setItem('phos_sbe_tl_h',
                         String(sbeTlClamp(px, SBE_TL_MAX_H)));
  } catch (e) {}
}

// ---------------------------------------------------------------------------
// LOADING AND SAVING
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// THE EDITOR TAB — the surface, and which document is in it
// ---------------------------------------------------------------------------
// The document id is NOT the open board's id any more. It is remembered, it
// survives a tab switch and a reload, and it can be nothing at all: an Editor
// that only exists inside an open storyboard is an editor most clips can never
// reach. `edDoc()` and `edRemember()` are the whole of that memory.
function edDoc() {
  try { return localStorage.getItem('phos_ed_doc') || ''; } catch (e) { return ''; }
}
function edRemember(id) {
  try {
    if (id) localStorage.setItem('phos_ed_doc', id);
    else localStorage.removeItem('phos_ed_doc');
  } catch (e) {}
}

// Exactly one of the two states is on screen: a document, or the invitation
// to pick one. `hidden` alone, no classes — #edStage is a plain flex column.
function edShow(which) {
  const doc = document.getElementById('sbTimeline');
  const empty = document.getElementById('edEmpty');
  if (doc) doc.hidden = (which !== 'doc');
  if (empty) empty.hidden = (which !== 'empty');
  const stage = document.getElementById('edStage');
  if (stage) stage.scrollTop = 0;
}

// Entering the tab. Re-entry is not re-opening: if the document is already
// loaded it is resumed, because a tab switch is not a close.
function edInit() {
  if (SBE.open && SBE.id) { sbeResume(); return; }
  const last = edDoc();
  if (last) {
    // A remembered id can point at a film that was deleted while the Editor
    // was closed. Opening it must fail into the picker, not into a red box —
    // the silent-failure restore the review flagged.
    sbeOpen(last, { onMissing: () => { edRemember(''); edShowPicker(); } });
    return;
  }
  edShowPicker();
}

// The empty state, painted from the board list. This is "pick a film / start
// from clips": every film with something on disk, plus the pool on the left.
async function edShowPicker() {
  edShow('empty');
  // With no document open "this film" has nothing in it, so the pool opens
  // on the generations — which is the source that always has something.
  if (ED.src === 'film') edPoolSrc('gallery'); else edPoolRefresh();
  const box = document.getElementById('edPick');
  if (box) box.innerHTML = '<span class="sbe-note">reading your __SEQS__…</span>';
  let boards = [];
  try {
    const r = await (await fetch('/storyboard/list')).json();
    boards = ((r && r.boards) || []).filter(b => (b.clips || 0) > 0);
  } catch (e) { boards = []; }
  if (!box) return;
  if (!boards.length) {
    box.innerHTML = '<span class="sbe-note">Nothing has been rendered yet. ' +
      'Anything you generate — on any engine — shows up in the media pool ' +
      'on the left, and one click puts it on the track.</span>';
    return;
  }
  box.innerHTML = boards.map(b =>
    '<button type="button" onclick="edOpenBoard(\'' + escapeHtml(b.id) + '\')">' +
    '<svg class="ph" aria-hidden="true"><use href="#ph-film-strip"/></svg>' +
    escapeHtml(b.title || b.id) +
    '<span class="sub">' + (b.clips || 0) + ' clip' + ((b.clips === 1) ? '' : 's') +
    '</span></button>').join('');
}

// The one door, from anywhere: the rail's step 3, a board row, the picker.
function edOpenBoard(id) {
  try { _uiEvent('feature_used', {feature: 'editor_open'}); } catch (_) {}
  if (!id) return;
  if (typeof workflowSwitch === 'function') workflowSwitch('editor');
  sbeOpen(id);
}

function sbeOpen(id, opts) {
  const want = String(id || SBE.id || edDoc() || '');
  if (!want) { edShowPicker(); return; }
  if (SBE.open && SBE.id && SBE.id !== want) sbeCloseDoc({ quiet: true });
  SBE.open = true;
  SBE.id = want;
  SBE.onMissing = (opts && opts.onMissing) || null;
  edRemember(want);
  // The title comes off the edit payload (it always carried one). Reading it
  // off the open board tied the Editor's header to the Storyboard's state.
  SBE.title = (SBE.id === SB.id
    ? ((((SB.payload || {}).board) || {}).title || '') : '') || SBE.title || 'Editor';
  sbeEl('sbeTitle').textContent = SBE.title;
  edShow('doc');
  sbeSetState('loading…', '');
  sbeLoad();
  if (SBE.timer) clearInterval(SBE.timer);
  SBE.timer = setInterval(sbeTick, 1500);
  window.addEventListener('resize', sbePaint);
  if (typeof edPoolRefresh === 'function') edPoolRefresh();
}

// Leaving the TAB. The document stays open — clips, undo stack, playhead and
// all — and only the things that cost something while nobody is looking stop.
function sbeSuspend() {
  if (!SBE.open) return;
  sbeStop();
  sbeSrcStop();          // both screens, or the left one plays on in a tab nobody is looking at
  if (SBE.dirty && !SBE.conflict) sbeSave(true);
  if (SBE.timer) { clearInterval(SBE.timer); SBE.timer = null; }
  window.removeEventListener('resize', sbePaint);
}

function sbeResume() {
  if (!SBE.open) return;
  edShow('doc');
  if (SBE.timer) clearInterval(SBE.timer);
  SBE.timer = setInterval(sbeTick, 1500);
  window.removeEventListener('resize', sbePaint);
  window.addEventListener('resize', sbePaint);
  sbePaint();
  sbeLoad(true);
  if (typeof edPoolRefresh === 'function') edPoolRefresh();
}

// Closing the DOCUMENT. This is the only thing that drops SBE.open, and it is
// reached from Esc, from the header's Close, and from opening another film.
function sbeCloseDoc(opts) {
  sbeStop();
  sbeVersionsClose();     // a picker for a document that is gone is a lie
  // NOT A SAVE. Leaving the tab is not the user asking for his draft to be
  // rewritten; the backup is what catches the work, and the offer on the way
  // back in is what returns it.
  if (SBE.open && SBE.dirty && !SBE.conflict) {
    // Fire-and-forget, because leaving is not a thing to be refused — but
    // SAID OUT LOUD if it did not land, since after this the only copy of
    // those minutes was the one that was not written.
    const was = SBE.title || 'this draft';
    sbeBackup(true).then(kept => {
      if (!kept) {
        phosToast('The unsaved changes in "' + was + '" could not be backed ' +
                  'up before it closed. Open it again and press Save if it ' +
                  'still has them.', { kind: 'danger', duration: 9000 });
      }
    });
  }
  if (SBE.timer) { clearInterval(SBE.timer); SBE.timer = null; }
  if (SBE.saveTimer) { clearTimeout(SBE.saveTimer); SBE.saveTimer = null; }
  SBE.open = false;
  SBE.id = '';
  SBE.clips = [];
  SBE.undo.length = 0; SBE.redo.length = 0;
  window.removeEventListener('resize', sbePaint);
  if (!(opts && opts.quiet)) { edRemember(''); edShowPicker(); }
}

// Kept under its old name because it is the door Esc and the header use.
function sbeClose() { sbeCloseDoc(); }

// The film this cut belongs to, in the tab that owns films. One direction of
// the door the rail's step 3 opens the other way.
function sbeGoToBoard() {
  if (!SBE.id) return;
  if (typeof workflowSwitch === 'function') workflowSwitch('storyboard');
  if (typeof sbOpenAt === 'function') sbOpenAt(SBE.id, 'shots');
}

// The document is gone and nothing about it is worth keeping.
function sbeTeardown() { sbeCloseDoc({ quiet: true }); }

async function sbeLoad(quiet) {
  let r;
  try {
    r = await (await fetch('/storyboard/edit?id=' + encodeURIComponent(SBE.id)
              + '&session=' + encodeURIComponent(SBE.session))).json();
  } catch (e) {
    if (!quiet) sbeSetState('the panel did not answer', 'dirty');
    return;
  }
  if (!r || !r.ok) {
    // THE DOCUMENT IS GONE, not broken. A remembered id whose film was
    // deleted has to land on the picker — the Editor's own empty state — and
    // not on an error box for a film the user never asked to open.
    if (SBE.onMissing && r && !r.corrupt) {
      const fn = SBE.onMissing;
      SBE.onMissing = null;
      SBE.open = false; SBE.id = '';
      if (SBE.timer) { clearInterval(SBE.timer); SBE.timer = null; }
      fn();
      return;
    }
    // A corrupt edit.json is NEVER silently replaced — the server says so and
    // the only honest thing to do is repeat it.
    sbeShowDocError(r && r.corrupt
      ? (r.error || 'edit.json is corrupt') + ' — nothing was overwritten. Fix or delete the file beside the board.'
      : ((r && r.error) || 'could not read this timeline'));
    return;
  }
  sbeAdopt(r, quiet);
}

function sbeAdopt(r, quiet) {
  SBE.onMissing = null;
  // The payload has always carried the title; the Editor now uses it, so the
  // header names the film it is cutting rather than whatever board is open.
  if (r.title) {
    SBE.title = r.title;
    const t = sbeEl('sbeTitle');
    if (t) t.textContent = SBE.title;
  }
  SBE.edit = r.edit || {};
  SBE.audio = SBE.edit.audio || null;
  SBE.beats = SBE.edit.beats || null;
  SBE.revision = sbeNum(SBE.edit.revision, 0);
  SBE.proxyUrl = r.proxy_url || '';
  SBE.unplaced = r.unplaced || [];
  SBE.pool = r.clips || [];
  SBE.relink = r.relink || [];
  SBE.sections = r.sections || [];
  SBE.prepare = r.prepare || {};
  if (r.drafts) SBE.drafts = r.drafts;
  if (r.active_draft !== undefined) SBE.activeDraft = r.active_draft || '';
  SBE.backup = r.backup || null;
  SBE.backupHidden = false;          // a NEW offer is not the one dismissed
  SBE.overlays = (SBE.edit.overlays || []).map(o => Object.assign({}, o));
  SBE.transitions = (SBE.edit.transitions || []).map(t => Object.assign({}, t));
  SBE.ovSel = '';
  SBE.txSel = '';
  SBE.clips = sbeAdoptGaps((SBE.edit.clips || []).map(c => Object.assign({}, c)));
  sbeLayout(SBE.clips);
  SBE.dirty = false;
  SBE.conflict = 0;
  SBE.errors = {};
  sbeEl('sbeConflict').hidden = true;
  // The auto-key receipt is about ONE placement on the timeline that was just
  // replaced. Carrying it across a load would leave an "undo" pointing at an
  // overlay id that is no longer in the document.
  if (typeof sbeKeyedDismiss === 'function') sbeKeyedDismiss();
  sbePaintNotices();
  if (SBE.audio && SBE.audio.path && !sbeEl('sbeMusic').value) {
    sbeEl('sbeMusic').value = SBE.audio.path;
  }
  sbePaintMusicName();
  // The saved mode wins over the control's default, or reopening a film would
  // silently re-arm `replace` on a timeline that was mixed `under`.
  sbeSetMusicMode((SBE.audio && SBE.audio.mode) || 'under');
  sbeSetState(r.generated ? 'cut by the auto-editor' : 'saved · revision ' + SBE.revision,
              'saved');
  sbeSyncMusic();
  sbePaintDraft();
  sbePaintRecovery();
  sbePaintRelink();
  sbeDeliverPaint();
  if (ED.src === 'film') edPoolRefresh();
  sbeFetchPeaks();
  sbePaint();
  // Land on a picture, not on black. A timeline that opens dark reads as
  // broken for the second and a half before the first click.
  if (!SBE.playing && !SBE.drag) sbeShowFrameAt(SBE.playhead);
  if (!quiet && r.generated) {
    phosToast('No timeline existed, so the auto-editor cut one — every shot at its best window, on the beat.',
              { duration: 7000 });
  }
}

async function sbeFetchPeaks() {
  try {
    const res = await fetch('/storyboard/edit/peaks?id=' + encodeURIComponent(SBE.id));
    if (!res.ok) { SBE.peaks = null; SBE.peaksFor = ''; sbePaint(); return; }
    const doc = await res.json();
    SBE.peaks = sbeDecodePeaks(doc);
    SBE.peaksFor = String(doc.path || '');
    // THE WAVEFORM EXISTS BEFORE THE EDIT KNOWS ABOUT IT. `prepare` writes
    // peaks.json beside the board; only an auto-edit writes the soundtrack
    // INTO edit.json. Showing an empty strip over a track that is right there
    // would be the UI lying about the server's own state — so the axis comes
    // off the peaks document, which carries its own path and duration. This
    // does NOT touch SBE.edit, so nothing here can be saved as if the
    // arrangement had a soundtrack it was never cut to.
    if (!SBE.audio && doc && doc.path) {
      SBE.audio = { path: String(doc.path), offset: 0, peaks: 'peaks.json',
                    duration: sbeNum(doc.duration) };
      if (!sbeEl('sbeMusic').value) {
        sbeEl('sbeMusic').value = String(doc.path);
        sbePaintMusicName();
      }
    }
  } catch (e) { SBE.peaks = null; SBE.peaksFor = ''; }
  sbePaint();
}

// The soundtrack row, as a name. `sbeMusic` is still the input every other
// caller reads and writes — this only decides which of the two is on screen.
function sbeMusicEditPath(on) {
  const row = sbeEl('sbePrepare');
  if (row && row.classList) row.classList.toggle('is-editing', !!on);
  if (on) { const b = sbeEl('sbeMusic'); if (b && b.focus) { try { b.focus(); b.select(); } catch (e) {} } }
  else sbePaintMusicName();
}

function sbePaintMusicName() {
  // `sbeMusicPath`, NOT `sbeMusicName` — the latter is the label painted onto
  // the music BLOCK down on the timeline, and two elements answering to one
  // id is one element and one bug.
  const el = sbeEl('sbeMusicPath');
  if (!el) return;
  const full = String((sbeEl('sbeMusic') || {}).value || '');
  const base = full.split('/').pop() || '';
  el.textContent = base || 'no soundtrack';
  el.title = full || 'No soundtrack yet — press Change… to point at one';
}

// "PROTECTED 4s AGO", or the truth when it is not. Painted from the tick, so
// the number is the real age and not the age it had when something last
// happened to call a setter.
function sbePaintProtected() {
  const el = sbeEl('sbeProtected');
  if (!el) return;
  if (!SBE.open || !SBE.id) { el.hidden = true; return; }
  const age = SBE.backedUpAt ? Math.round((Date.now() - SBE.backedUpAt) / 1000) : null;
  // COLD IS "THERE IS UNSAVED WORK AND THE NET HAS NOT CAUGHT THIS TAB", and
  // it is deliberately the SAME threshold the watchdog alarms on, so the chip
  // and the banner can never disagree. Stated against `backedUpAt` rather than
  // against `dirtyAt`: the outage this exists for froze `dirtyAt` in the past
  // while `backedUpAt` stayed ahead of it, and every test written in terms of
  // that pair read "protected" for seven hours. How long ago the net actually
  // caught this tab cannot be faked by a stuck flag.
  const cold = SBE.dirty && (!SBE.backedUpAt
            || (Date.now() - SBE.backedUpAt) > SBE_SAVE_GRACE_MS);
  let text;
  if (age === null) text = SBE.dirty ? 'not backed up yet' : '';
  else if (age < 60) text = 'protected ' + age + 's ago';
  else text = 'protected ' + Math.round(age / 60) + 'm ago';
  if (cold) text = 'NOT PROTECTED — ' + text;
  // A SECOND EDITOR IS INFORMATION, NEVER A REASON TO STOP. It used to be the
  // reason this tab stopped writing entirely.
  if (SBE.otherEditor) text = (text ? text + ' · ' : '') + 'also open elsewhere';
  el.textContent = text;
  el.hidden = !text;
  el.classList.toggle('is-cold', !!cold);
}

function sbeSetState(text, kind) {
  const el = sbeEl('sbeState');
  if (!el) return;
  sbeEl('sbeStateText').textContent = text;
  el.classList.toggle('is-dirty', kind === 'dirty');
  el.classList.toggle('is-saved', kind === 'saved');
  el.classList.toggle('is-alarm', kind === 'alarm');
  // The dot beside the draft's name agrees with the chip, always — two
  // indicators that can disagree are one indicator and one bug.
  if (typeof sbePaintDraft === 'function') sbePaintDraft();
  if (typeof sbePaintProtected === 'function') sbePaintProtected();
}

function sbeShowDocError(msg) {
  const box = sbeEl('sbeErrors');
  box.hidden = false;
  box.innerHTML = '<b>' + escapeHtml(String(msg)) + '</b>';
  sbeSetState('not saved', 'dirty');
}

// Every mutation goes through here: one undo snapshot, one layout, one repaint,
// one debounced save. Nothing else is allowed to touch SBE.clips, which is why
// undo can be a straight array of snapshots rather than a command log.
// ONE UNDO STEP, WHATEVER MOVED. The stack used to hold a bare array of
// clips, which was true for exactly as long as the arrangement was only
// clips: the soundtrack is an object on the timeline now, and an object you
// cannot ⌘Z is half an object. A snapshot carries both. The array form is
// still accepted on the way back in, because a clip drag takes its own
// snapshot at pointerdown and there is no reason to make it carry audio it
// cannot change.
// THE SNAPSHOT IS OF THE TIMELINE, NOT OF THE DOCUMENT. It used to read
// `SBE.edit.audio`, and `sbeFetchPeaks` deliberately fills `SBE.audio`
// WITHOUT touching the document — a soundtrack discovered from peaks.json is
// on the timeline before the arrangement was ever cut to it. So on a film
// that was Prepared but never auto-edited, the snapshot carried `audio: null`
// and one ⌘Z of any clip edit deleted the track: the block vanished, the
// preview bed went silent, and redo could not bring it back because the redo
// stack had been handed the same null. Only a reload restored it.
function sbeSnapshot(audio) {
  return JSON.stringify({
    clips: SBE.clips,
    // THE LANE IS PART OF THE ARRANGEMENT, so undo and redo cover it. Without
    // this a card placed and then undone would stay on screen while the
    // pictures walked back without it.
    overlays: SBE.overlays || [],
    transitions: SBE.transitions || [],
    audio: (audio === undefined)
      ? (SBE.audio || (SBE.edit && SBE.edit.audio) || null) : audio,
  });
}

function sbeRestore(json) {
  const s = JSON.parse(json);
  if (Array.isArray(s)) { SBE.clips = s; return; }
  SBE.clips = s.clips || [];
  // An older snapshot (one taken before the lane existed) carries no
  // `overlays`, and restoring `undefined` over a live lane would delete a
  // card nobody asked to remove — absent means "unchanged", not "empty".
  if (s.overlays !== undefined) SBE.overlays = s.overlays || [];
  if (s.transitions !== undefined) SBE.transitions = s.transitions || [];
  SBE.edit = SBE.edit || {};
  // ...and a restore never COMMITS a discovered track into the document. The
  // arrangement owns a soundtrack only once somebody has placed it (see
  // sbeApplyMusic); undoing a trim must not be what saves one.
  if (SBE.edit.audio) SBE.edit.audio = s.audio || null;
  SBE.audio = s.audio || null;
}

// The overlay lane's own door, and it is the same door: one undo step, one
// dirty flag, one queued snapshot. `sbeMutate` works on clips; this works on
// the lane, and both push the SAME snapshot shape so undo walks either.
function sbeOvMutate(fn) {
  const before = sbeSnapshot();
  const res = fn(SBE.overlays || []);
  if (!res || res.ok === false) {
    if (res && res.why) phosToast(res.why, {});
    return false;
  }
  SBE.undo.push(before);
  if (SBE.undo.length > SBE_UNDO_MAX) SBE.undo.shift();
  SBE.redo.length = 0;
  SBE.overlays = res.overlays;
  SBE.dirty = true;
  sbeSetState('unsaved changes', 'dirty');
  sbePaint();
  sbeQueueSave();
  // The mutation's own result, so a caller that needs to talk ABOUT what it
  // added — the auto-key notice needs the id to be able to undo it — does not
  // have to guess which row is new.
  return res;
}

function sbeOvAddAt(item, at) {
  const res = sbeOvMutate(os => sbeOvAdd(os, item, at));
  if (!res) return null;
  phosToast('Overlay added — it composites over the picture. Drag it along '
            + 'the lane, pull either end for its length, and fade it in from '
            + 'Effects.', { duration: 7000 });
  return res.added || null;
}

// THE ONE PLACE A KEYED OVERLAY GOES BACK TO THE FILE IT CAME FROM.
// "Keep original" is a normal edit — one undo step, one dirty flag — because
// it is a decision the user made, and every other decision in this editor is
// undoable. The original was never modified, so this is a pointer swap.
function sbeOvSetPath(overlays, id, path) {
  const o = sbeOvById(overlays, id);
  if (!o) return { overlays: overlays, ok: false, why: 'gone' };
  const out = overlays.map(x => Object.assign({}, x));
  sbeOvById(out, id).path = path;
  return { overlays: out, ok: true };
}

function sbeOvDeleteSel() {
  if (!SBE.ovSel) return;
  const id = SBE.ovSel;
  const ok = sbeOvMutate(os => sbeOvDelete(os, id));
  if (ok) { SBE.ovSel = ''; sbePaint(); }
}

function sbeMutate(fn) {
  const before = sbeSnapshot();
  const res = fn(SBE.clips);
  if (!res || res.ok === false) {
    if (res && res.why === 'locked') phosToast('That shot is locked.', {});
    else if (res && res.why) phosToast(res.why, {});
    return false;
  }
  SBE.undo.push(before);
  if (SBE.undo.length > SBE_UNDO_MAX) SBE.undo.shift();
  SBE.redo.length = 0;
  SBE.clips = res.clips;
  // A CLIP THAT LEFT TAKES ITS BOUNDARY WITH IT. Any mutation may remove or
  // reorder clips; a transition naming a clip that is gone, or one that has
  // become the last clip, is pruned here rather than refused at the save.
  if (res.transitions) SBE.transitions = res.transitions;
  SBE.transitions = sbeTxPrune(SBE.transitions || [], SBE.clips);
  SBE.dirty = true;
  sbeSetState('unsaved changes', 'dirty');
  sbePaint();
  sbeQueueSave();
  return true;
}

function sbeUndo() {
  if (!SBE.undo.length) return;
  SBE.redo.push(sbeSnapshot());
  sbeRestore(SBE.undo.pop());
  SBE.dirty = true;
  sbeSetState('unsaved changes', 'dirty');
  sbePaint();
  sbeQueueSave();
}

function sbeRedo() {
  if (!SBE.redo.length) return;
  SBE.undo.push(sbeSnapshot());
  sbeRestore(SBE.redo.pop());
  SBE.dirty = true;
  sbeSetState('unsaved changes', 'dirty');
  sbePaint();
  sbeQueueSave();
}

// NOTHING WRITES THE USER'S DRAFT BUT THE USER. The owner, after losing an
// afternoon to a save he could not see: "it's better that the user has the
// power to manage this feature... he should have control over the saving, and
// only he should have that control. You can keep a backup in case it's
// needed."
//
// So this — the lane every mutation calls — no longer saves. It writes a
// CRASH BACKUP to a file beside the draft, which never lands on edit.json and
// never moves a revision. The Save button is the only thing that writes the
// document he named, and the only thing that can change what he gets back.
function sbeQueueSave() {
  // THE WATCHDOG'S CLOCK IS ARMED FIRST, BEFORE ANY EARLY RETURN CAN SKIP IT.
  // This line used to sit UNDER `if (SBE.superseded) return`, and that one
  // line's position is the whole of a seven-hour outage: once a tab was
  // superseded, `dirtyAt` froze at its last pre-supersede value while
  // `backedUpAt` stayed AHEAD of it, so the 12-second watchdog's
  // `backedUpAt < dirtyAt` test read "this work is backed up" forever and the
  // alarm it exists to raise could never fire. Nothing may be allowed to
  // return from this function without first recording that there is unwritten
  // work and when it appeared.
  if (!SBE.dirtyAt) SBE.dirtyAt = Date.now();
  if (SBE.saveTimer) clearTimeout(SBE.saveTimer);
  SBE.saveTimer = setTimeout(() => { SBE.saveTimer = null; sbeBackup(); }, 1400);
}

// The backup. Same payload the save sends, a different door, and it answers
// to nobody: no expect_revision (it cannot conflict with anything), no adopt
// (it changes nothing on screen), no toast when it works.
// TRUE MEANS "WHAT IS ON SCREEN IS ALSO SOMEWHERE ELSE" — and it is the
// answer callers act on, so every path that does NOT write says false. Three
// of them used to return `undefined`, which `sbeDraftOp` read as fine and
// then cleared `dirty` on.
async function sbeBackup(quiet) {
  if (!SBE.open || !SBE.id) return false;
  // A POST already in flight carries the state it was built from, not the
  // edit that arrived since — so it is not this call landing.
  if (SBE.backingUp) return false;
  // THE LANE NEVER STOPS. It used to refuse while an offer was unanswered —
  // there was exactly ONE backup file per draft, so writing a new one would
  // have destroyed the work the offer was holding. The cure was worse than
  // the disease: a chip nobody dismissed switched the safety net off for the
  // rest of the session, silently. The lane is versioned now (one file per
  // snapshot, pruned), so a new snapshot cannot eat an old one and there is
  // nothing left to guard. See docs/EDITOR_SAVE_MODEL.md §2.
  if (SBE.saveTimer) { clearTimeout(SBE.saveTimer); SBE.saveTimer = null; }
  SBE.backingUp = true;
  let r;
  try {
    const body = sbeSaveBody({ id: SBE.id, edit: SBE.edit, clips: SBE.clips,
                               overlays: SBE.overlays, expect: null });
    // WHICH DRAFT THIS WAS COMPOSED FROM. The server files the backup under
    // the draft that is active when the write LANDS, and this one is
    // debounced — so without the name in the body, a backup of the draft you
    // just left is offered back as the unsaved work of the one you opened.
    body.draft = SBE.activeDraft || '';
    body.session = SBE.session;
    const res = await fetch('/storyboard/edit/backup', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body) });
    r = await res.json();
  } catch (e) {
    r = { ok: false, error: String(e) };
  } finally {
    SBE.backingUp = false;
  }
  if (r && r.ok) {
    SBE.backedUpAt = Date.now();
    // Somebody else is editing this film too. Worth saying; never worth
    // stopping for.
    const cur = (r.session || {}).token || '';
    SBE.otherEditor = (cur && cur !== SBE.session) ? cur : '';
    sbeSaveAlarmClear();
    sbePaintProtected();
    return true;
  }
  // A REFUSED SNAPSHOT IS THE SAFETY NET OFF, whatever the reason given.
  // `stale_session` used to be handled HERE as a special non-failure: the tab
  // set a flag, cleared the alarm, showed one 9-second toast and never wrote
  // again. The state line it set was overwritten by the very next edit, so a
  // tab that had stopped protecting its user looked completely normal — the
  // same class as the `if (SBE.backup) return false` this file already had to
  // delete. The server no longer refuses on session; if some future build
  // does, it falls through to the alarm below like every other failure, and
  // the alarm does not go away until a write lands.
  // A BACKUP THAT IS NOT BEING WRITTEN IS THE SAFETY NET GONE, and the user
  // cannot tell by looking. It is the same alarm because it is the same
  // sentence: what is on screen is not anywhere else.
  sbeSaveAlarm('the crash backup is not being written (' +
               ((r && r.error) || 'the panel did not answer') +
               ') — press Save');
  return false;
}

// THE OUTER HALF OF SAVING, AND IT EXISTS BECAUSE OF A REAL LOSS. The owner
// cut for twenty minutes and nothing reached the disk: edit.json sat frozen
// while he kept working, and the only thing on screen that knew was a small
// grey chip. Two defects made that possible and both are closed here.
//
// ONE — `if (SBE.saving) return` DROPPED THE SAVE. A debounced save that
// arrived while another was in flight simply vanished: `SBE.dirty` stayed
// true, no timer was left pending, and that edit was never written again
// unless the user happened to touch the film once more. Now it is remembered
// and re-queued the moment the one in flight lands.
//
// TWO — THE FLAG COULD STICK. `SBE.saving = false` sat on the happy path,
// after code that can throw (the payload is built before the fetch). One
// throw and the editor stops saving FOREVER, silently, for the rest of the
// session. `finally` is the whole fix, and it is the difference between an
// error and a lost afternoon.
async function sbeSave(quiet, force) {
  if (!SBE.open) return;
  // WHAT KIND OF WRITE IS PENDING, not just that one is. `sbeQueueSave` no
  // longer saves — it schedules a crash BACKUP — so remembering a dropped
  // save as a bare `true` and re-queuing through that lane turned the second
  // of two rapid Save presses into a backup write, leaving edit.json at the
  // older revision with the alarm reading clear.
  if (SBE.saving) { SBE.savePending = quiet ? 'backup' : 'save'; return 'busy'; }
  if (SBE.saveTimer) { clearTimeout(SBE.saveTimer); SBE.saveTimer = null; }
  SBE.saving = true;
  let ok = false;
  try {
    ok = await sbeSaveInner(quiet, force);
  } catch (e) {
    // A throw here is the case that used to wedge the flag. It is also the
    // one nothing on screen could have told you about.
    sbeSaveAlarm('the editor could not build the save (' +
                 ((e && e.message) || String(e)) + ')');
  } finally {
    SBE.saving = false;
    const again = SBE.savePending;
    SBE.savePending = false;
    if (again && SBE.dirty && !SBE.conflict) {
      if (again === 'save') sbeSave(quiet, force);
      else sbeQueueSave();
    }
  }
  return ok;
}

// THE ALARM. A save that is not happening is not a status, it is an
// emergency: everything on screen is about to be lost and only this says so.
// It survives until a save actually lands — no timeout, no fade — because the
// failure it reports survives too.
function sbeSaveAlarm(why) {
  SBE.saveFailed = why || 'the panel did not store your changes';
  const box = sbeEl('sbeAlarm');
  if (box) {
    box.hidden = false;
    sbeEl('sbeAlarmWhy').textContent = SBE.saveFailed;
  }
  sbePaintNotices();
  sbeSetState('NOT SAVED', 'alarm');
}

function sbeSaveAlarmClear() {
  if (!SBE.saveFailed) return;
  SBE.saveFailed = '';
  const box = sbeEl('sbeAlarm');
  if (box) box.hidden = true;
  if (SBE.noticeLead === 'sbeAlarm') SBE.noticeLead = '';
  sbePaintNotices();
}

async function sbeSaveInner(quiet, force) {
  const order = SBE.clips.map(c => c.id);
  SBE.sentOrder = order;
  const body = sbeSaveBody({ id: SBE.id, edit: SBE.edit, clips: SBE.clips,
                             overlays: SBE.overlays,
                             expect: force ? null : SBE.revision });
  let r;
  try {
    const res = await fetch('/storyboard/edit/save', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body) });
    r = await res.json();
    r._status = res.status;
  } catch (e) {
    r = { ok: false, error: String(e), _status: 0 };
  }
  if (r.ok) {
    SBE.errors = {};
    sbeEl('sbeErrors').hidden = true;
    // Adopt the server's copy of the document, but NOT its clip array — the
    // user may have moved something in the milliseconds the save was in
    // flight, and throwing that away is the one thing an editor must never do.
    SBE.edit = Object.assign({}, r.edit || {}, { clips: SBE.edit.clips || [] });
    SBE.revision = sbeNum((r.edit || {}).revision, SBE.revision + 1);
    SBE.unplaced = r.unplaced || [];
    SBE.prepare = r.prepare || SBE.prepare;
    SBE.dirty = false;
    SBE.conflict = 0;
    sbeEl('sbeConflict').hidden = true;
  sbePaintNotices();
    SBE.dirtyAt = 0;
    // A SAVE IS THE USER ANSWERING THE OFFER. The server deletes the backup
    // on a successful save, so this follows it rather than deciding on its
    // own — and clearing it is what brings the crash lane back to life:
    // sbeBackup refuses to write while an offer is unanswered, so a film that
    // opened with one used to lose its safety net for the whole session and
    // then be told, every twelve seconds, that saving was failing.
    SBE.backup = r.backup || null;
  SBE.backupHidden = false;          // a NEW offer is not the one dismissed
    sbePaintRecovery();
    // The drafts rows carry clip counts and "saved N min ago" — the first
    // thing a user checks after saving is the panel that reports on saving.
    if (r.drafts) { SBE.drafts = r.drafts; sbePaintDraft(); }
    sbeSaveAlarmClear();
    sbeSetState('saved · revision ' + SBE.revision, 'saved');
    sbePaint();
    return true;
  }
  if (r._status === 409 && r.conflict) {
    // Honest, and it does not choose for you: another tab is ahead, your
    // arrangement is still on this screen, and both ways out are one click.
    SBE.conflict = sbeNum(r.revision, 0);
    sbeEl('sbeConflict').hidden = false;
    sbePaintNotices();
    sbeEl('sbeConflictText').textContent =
      'Another tab saved this timeline (it is at revision ' + SBE.conflict +
      ', you started from ' + SBE.revision + '). Nothing here has been lost.';
    sbeSaveAlarm('another tab is at revision ' + SBE.conflict +
                 ' and this one is not being stored — choose which arrangement wins');
    return false;
  }
  if (Array.isArray(r.errors) && r.errors.length) {
    SBE.errors = sbeErrorsByClip(r.errors, order);
    SBE.errors.all = r.errors || [];
    sbeRenderErrors(r.errors);
    sbeSaveAlarm(r.errors.length + ' problem' + (r.errors.length === 1 ? '' : 's') +
                 ' in the timeline — nothing was written');
    sbePaint();
    return false;
  }
  // LOUD WHETHER OR NOT ANYBODY ASKED. This branch used to be silent for the
  // autosave — `if (!quiet)` — which is precisely the path that runs while
  // somebody is working and precisely the path that lost twenty minutes.
  sbeSaveAlarm(r.error || 'the panel did not answer the save');
  if (!quiet) phosToast(r.error || 'Could not save the timeline.', { kind: 'danger' });
  return false;
}

function sbeRenderErrors(errors) {
  const box = sbeEl('sbeErrors');
  if (!errors || !errors.length) {
    box.hidden = true; box.innerHTML = '';
    if (SBE.noticeLead === 'sbeErrors') SBE.noticeLead = '';
    sbePaintNotices();
    return;
  }
  box.hidden = false;
  // THE FIRST SENTENCE, THEN THE REST ON REQUEST. Nine clips failing the same
  // rule printed nine lines of the same sentence and pushed the timeline off
  // the screen; the one thing the reader needs is what is wrong and how many
  // are wrong with it.
  const rest = errors.length - 1;
  box.innerHTML = '<b>The timeline was not saved.</b><ul>' +
    errors.map((e, i) => '<li' + (i && !SBE.errsOpen ? ' hidden' : '') + '>' +
      escapeHtml(e.message || e.code || '') + '</li>').join('') +
    '</ul>' +
    (rest > 0
      ? '<button type="button" class="ghost-btn" onclick="sbeErrsToggle()">'
        + (SBE.errsOpen ? 'Show less' : escapeHtml('and ' + rest + ' more'))
        + '</button>'
      : '');
  sbePaintNotices();
}

function sbeErrsToggle() {
  SBE.errsOpen = !SBE.errsOpen;
  sbeRenderErrors(SBE.errors && SBE.errors.all);
}

async function sbeTakeTheirs() {
  SBE.conflict = 0;
  SBE.dirty = false;
  sbeEl('sbeConflict').hidden = true;
  sbePaintNotices();
  await sbeLoad();
  phosToast('Loaded the other tab\'s arrangement.', {});
}

async function sbeForceSave() {
  const ok = await sbeSave(false, true);
  if (ok) phosToast('Saved over the other tab\'s version.', { kind: 'success' });
}

// ---------------------------------------------------------------------------
// SAVE, SAID OUT LOUD — and the versions behind it
// ---------------------------------------------------------------------------
// The autosave stays SILENT: it fires a second after every drag, and a toast
// on each one would be a notification storm that teaches you to ignore
// notifications. The button is the opposite — it is pressed by somebody who
// wants to be told it worked, so it says which revision they now have.
async function sbeSaveNow() {
  const ok = await sbeSave(false);
  if (ok) {
    phosToast('Saved — revision ' + SBE.revision, { kind: 'success' });
  } else if (ok === 'busy') {
    // Two very different reasons used to arrive as the same `undefined`, and
    // this is the one where telling somebody their work needed no saving is
    // exactly wrong: it is unwritten and a save is on its way.
    phosToast('A save is already on its way — this one will follow it.', {});
  } else if (ok === undefined) {
    phosToast('Nothing to save yet.', {});
  }
}

// "4 minutes ago", in the smallest unit that still reads as a duration. Pure,
// and `now` is a parameter so the gate is not a clock race.
function sbeAgo(when, now) {
  const t = sbeNum(when, 0);
  if (!t) return '';
  const d = Math.max(0, sbeNum(now, Date.now() / 1000) - t);
  if (d < 45) return 'just now';
  if (d < 5400) return Math.round(d / 60) + ' min ago';
  if (d < 172800) return Math.round(d / 3600) + ' h ago';
  return Math.round(d / 86400) + ' d ago';
}

// One row of the picker, as the two strings it shows. Pure so the wording is
// a gate rather than something to squint at in a screenshot.
function sbeVersionLine(v, now) {
  const row = v || {};
  // THE FALLBACK FOLLOWS THE LANE. The split exists to separate the user's
  // decisions from the machine's, and then labelled his decisions with the
  // machine's word: the lane headed YOUR SAVES OF THIS DRAFT listed rows
  // called "autosave", which is the one word that means "not yours".
  const name = row.label ? String(row.label)
    : (row.readable === false ? 'unreadable'
       : (row.manual ? 'Save' : 'autosave'));
  const bits = [];
  if (row.revision !== null && row.revision !== undefined) {
    bits.push('revision ' + sbeNum(row.revision, 0));
  }
  if (row.clips !== null && row.clips !== undefined) {
    bits.push(sbeNum(row.clips, 0) + (sbeNum(row.clips, 0) === 1 ? ' clip' : ' clips'));
  }
  if (sbeNum(row.duration, 0) > 0) bits.push(sbeFmtTime(row.duration));
  const ago = sbeAgo(row.archived_at || row.saved_at, now);
  if (ago) bits.push(ago);
  return { name: name, meta: bits.join(' · '),
           kept: !!row.kept, bad: row.readable === false };
}

function sbeVersionsEl() {
  const el = sbeEl('sbeVersions');
  // PORTAL, on first open. The header and the stage column both carry
  // overflow, so a panel left where it was declared is sliced at their edge.
  if (el && el.parentElement !== document.body) document.body.appendChild(el);
  return el;
}

function sbeVersionsClose() {
  const el = sbeEl('sbeVersions');
  if (el) el.hidden = true;
}

async function sbeVersionsOpen(focusName) {
  if (!SBE.open || !SBE.id) {
    phosToast('Open a __SEQ__ first — versions belong to a timeline.', {});
    return;
  }
  const el = sbeVersionsEl();
  if (!el) return;
  // A SECOND PRESS CLOSES IT. Every other popover in this panel toggles;
  // this one re-positioned itself and fired another /versions fetch.
  const from = focusName ? 'keep' : 'vers';
  if (!el.hidden && el.dataset.from === from) { sbeVersionsClose(); return; }
  el.dataset.from = from;
  const trig = sbeEl(focusName ? 'sbeKeepBtn' : 'sbeVersBtn');
  const r = trig ? trig.getBoundingClientRect() : { bottom: 60, right: 400 };
  el.hidden = false;
  el.style.top = (r.bottom + 6) + 'px';
  el.style.left = Math.max(8, Math.min(window.innerWidth - 408,
                                       r.right - 400)) + 'px';
  if (focusName) {
    const q = sbeEl('sbeVersName');
    if (q && q.focus) { try { q.focus(); q.select(); } catch (e) {} }
  }
  await sbeVersionsLoad();
}

async function sbeVersionsLoad() {
  const list = sbeEl('sbeVersList');
  if (list) list.innerHTML = '<div class="sbe-vers-empty">Reading the history…</div>';
  let r;
  try {
    r = await (await fetch('/storyboard/edit/versions?id=' +
                           encodeURIComponent(SBE.id))).json();
  } catch (e) { r = { ok: false, error: String(e) }; }
  if (!r || !r.ok) {
    if (list) {
      list.innerHTML = '<div class="sbe-vers-empty">' +
        escapeHtml((r && r.error) || 'could not read this __SEQ__\'s history') +
        '</div>';
    }
    return;
  }
  sbeVersionsPaint(r.versions || [], sbeNum(r.keep, 50));
}

function sbeDraftsPaint() {
  const list = sbeEl('sbeDraftList');
  if (!list) return;
  const rows = SBE.drafts || [];
  list.innerHTML = '<div class="sbe-vers-head">This __SEQCAP__\'s drafts</div>' +
    rows.map(d => {
      const meta = [sbeNum(d.clips, 0) + ' clip' + (sbeNum(d.clips, 0) === 1 ? '' : 's')];
      if (sbeNum(d.duration, 0) > 0) meta.push(sbeFmtTime(d.duration));
      if (d.saved_at) meta.push('saved ' + sbeAgo(d.saved_at));
      else meta.push('never saved');
      return '<div class="sbe-vers-row' + (d.active ? ' is-active' : '') + '">' +
        '<span class="sbe-vr-main">' +
        '<span class="sbe-vr-name">' + escapeHtml(d.name) +
        (d.active ? ' · open' : '') + '</span>' +
        '<div class="sbe-vr-meta">' + escapeHtml(meta.join(' · ')) + '</div>' +
        '</span>' +
        (d.active ? '' : '<button type="button" class="ghost-btn" data-op="open" data-slug="' +
          escapeHtml(d.slug) + '" data-name="' + escapeHtml(d.name) + '">Open</button>') +
        '<button type="button" class="ghost-btn" data-op="dup" data-slug="' +
          escapeHtml(d.slug) + '" data-name="' + escapeHtml(d.name) + '">Copy</button>' +
        '<button type="button" class="ghost-btn" data-op="rename" data-slug="' +
          escapeHtml(d.slug) + '" data-name="' + escapeHtml(d.name) + '">Rename</button>' +
        (rows.length > 1 ? '<button type="button" class="ghost-btn" data-op="del" data-slug="' +
          escapeHtml(d.slug) + '" data-name="' + escapeHtml(d.name) + '">Delete</button>' : '') +
        '</div>';
    }).join('');
  list.querySelectorAll('button[data-op]').forEach(b => {
    b.addEventListener('click', () => {
      const slug = b.dataset.slug, name = b.dataset.name;
      if (b.dataset.op === 'open') sbeDraftOpen(slug, name);
      else if (b.dataset.op === 'dup') sbeDraftDuplicate(slug, name);
      else if (b.dataset.op === 'rename') sbeDraftRename(slug, name);
      else sbeDraftDelete(slug, name, b);
    });
  });
}

function sbeVersionsPaint(rows, keep) {
  const list = sbeEl('sbeVersList');
  const note = sbeEl('sbeVersNote');
  sbeDraftsPaint();
  if (note) {
    note.textContent = 'Save writes the draft — nothing else does. Below are ' +
      'this draft\'s past saves: every Save keeps the one before it, the last ' +
      keep + ' of them, and Restore keeps what it replaces.';
  }
  if (!list) return;
  if (!rows.length) {
    list.innerHTML = '<div class="sbe-vers-head">Past saves of this draft</div>' +
      '<div class="sbe-vers-empty">None yet — the second Save writes the ' +
      'first one.</div>';
    return;
  }
  const now = Date.now() / 1000;
  // HIS DECISIONS FIRST. "The auto saves should be saved separately from the
  // manual saves, at least, so the user can go back and see the manual
  // saves." The automatic lane is real history and stays reachable — it is
  // just not what somebody walking back through their own work is looking
  // for, and a wall of debounce noise buries the four saves that matter.
  const mine = rows.filter(v => v.manual !== false);
  const auto = rows.filter(v => v.manual === false);
  const draw = (v) => {
    const line = sbeVersionLine(v, now);
    return '<div class="sbe-vers-row' + (line.kept ? ' is-kept' : '') +
      (line.bad ? ' is-bad' : '') + '">' +
      '<span class="sbe-vr-main">' +
      '<span class="sbe-vr-name">' + escapeHtml(line.name) + '</span>' +
      '<div class="sbe-vr-meta">' + escapeHtml(line.meta) + '</div>' +
      '</span>' +
      (line.bad ? '' :
        '<button type="button" class="ghost-btn" data-file="' +
        escapeHtml(v.file) + '">Restore</button>') +
      '</div>';
  };
  list.innerHTML =
    '<div class="sbe-vers-head">Your saves of this draft ' +
    '<span>type a name above and press Keep to mark one</span></div>' +
    (mine.length ? mine.map(draw).join('')
                 : '<div class="sbe-vers-empty">None yet — press Save.</div>') +
    (auto.length
      ? '<details class="sbe-vers-auto"><summary>' + auto.length +
        ' automatic snapshot' + (auto.length === 1 ? '' : 's') + '</summary>' +
        auto.map(draw).join('') + '</details>'
      : '');
  list.querySelectorAll('button[data-file]').forEach(b => {
    b.addEventListener('click', () => sbeRestoreVersion(b.dataset.file));
  });
}

// ---- the drafts, as verbs -------------------------------------------------
// Every one of these lands through the SAME door — POST edit/draft — because
// they are five edits to one index, and five routes would be five places for
// the active pointer to end up wrong. Each answers with the whole read
// payload, so the client adopts a document rather than reasoning about one.
async function sbeDraftOp(op, extra, note) {
  const fd = new URLSearchParams();
  fd.set('id', SBE.id);
  fd.set('op', op);
  for (const k in (extra || {})) fd.set(k, String(extra[k]));
  // The draft on screen is about to stop being the active one. Its unsaved
  // work goes to its backup first — the server moves the SAVED file, and
  // without this the last few minutes would be left pointing at a draft
  // nobody is looking at any more.
  //
  // AND THE ANSWER IS CHECKED. The invariant this call exists for is the one
  // line above; ignoring its result meant the switch went ahead anyway, then
  // cleared `dirty` and adopted the server's document — while `activate_draft`
  // stashes the last SAVED file, so the work on screen went with no offer, no
  // toast and nothing to click.
  if (SBE.dirty && !SBE.conflict && !(await sbeBackup(true))) {
    phosToast('This draft has unsaved changes that could not be snapshotted. ' +
              'Press Save first — switching now would leave them behind.',
              { kind: 'danger', duration: 8000 });
    return false;
  }
  let r;
  try { r = await (await fetch('/storyboard/edit/draft', { method: 'POST', body: fd })).json(); }
  catch (e) { r = { ok: false, error: String(e) }; }
  if (!r || !r.ok) {
    phosToast((r && r.error) || 'That draft action did not work.', { kind: 'danger' });
    return false;
  }
  sbeNameMode('');
  if (op !== 'rename') {
    SBE.undo.length = 0; SBE.redo.length = 0;
    SBE.dirty = false; SBE.dirtyAt = 0;
    sbeAdopt(r, true);
  } else {
    SBE.drafts = r.drafts || SBE.drafts;
    sbePaintDraft();
  }
  await sbeVersionsLoad();
  if (note) phosToast(note(r), { kind: 'success', duration: 5000 });
  return true;
}

// ONE FIELD, AND IT KNOWS WHICH VERB IT IS. Enter used to be hard-wired to
// "new draft", so pressing Rename on a row, typing, and pressing Enter — the
// obvious gesture — created a duplicate draft carrying the half-edited name
// instead of renaming anything.
function sbeNameEnter() {
  if (SBE.renaming) return sbeDraftRename(SBE.renaming, '');
  return sbeDraftNew(true);
}

function sbeNameMode(slug) {
  SBE.renaming = slug || '';
  const btn = sbeEl('sbeVersKeep');
  if (btn) {
    btn.textContent = SBE.renaming ? 'Rename' : 'Copy';
    btn.title = SBE.renaming
      ? 'Rename the draft — the timeline is not touched'
      : 'A new draft holding a copy of the timeline on screen';
  }
}

function sbeDraftNew(fromCurrent) {
  const box = sbeEl('sbeVersName');
  const name = String((box && box.value) || '').trim();
  if (!name) {
    phosToast('Give the draft a name — that is how you get back to it.', {});
    if (box && box.focus) { try { box.focus(); } catch (e) {} }
    return;
  }
  sbeNameMode('');
  if (box) box.value = '';
  return sbeDraftOp('new', { name: name, from: fromCurrent ? 'current' : 'empty' },
                    () => 'Working on "' + name + '" now. ' +
                          (fromCurrent ? 'It holds a copy of the cut you were on.'
                                       : 'Empty timeline, same soundtrack.'));
}

function sbeDraftDuplicate(slug, name) {
  return sbeDraftOp('duplicate', { slug: slug },
                    r => 'Copied "' + name + '" — you are on the copy now.');
}

function sbeDraftOpen(slug, name) {
  return sbeDraftOp('activate', { slug: slug },
                    () => 'Opened "' + name + '".');
}

function sbeDraftRename(slug, was) {
  // The panel's own input, reused: a rename is a name, and names in this app
  // are typed into panel controls, never into a browser dialog.
  const box = sbeEl('sbeVersName');
  const name = String((box && box.value) || '').trim();
  if (!name) {
    // ARMED, NOT JUST ASKED. The field is prefilled and SELECTED, and the
    // panel remembers which row it belongs to — the old order focused first
    // and wrote the value after, so the caret sat behind the old name and
    // typing appended to it.
    phosToast('Edit the name above, then press Rename.', {});
    sbeNameMode(slug);
    if (box) {
      box.value = was || '';
      try { box.focus(); box.select(); } catch (e) {}
    }
    return;
  }
  sbeNameMode('');
  if (box) box.value = '';
  return sbeDraftOp('rename', { slug: slug, name: name },
                    () => 'Renamed to "' + name + '".');
}

// The verb the drafts rewrite dropped. It names the save ALREADY ON DISK —
// no revision bump, no write to edit.json, nothing about the timeline
// changes, which is exactly why it is safe to press at any moment.
async function sbeKeepVersion() {
  const box = sbeEl('sbeVersName');
  const label = String((box && box.value) || '').trim();
  if (!label) {
    phosToast('Type a name for this save first — that is what makes it one ' +
              'you can find again.', {});
    if (box && box.focus) { try { box.focus(); } catch (e) {} }
    return;
  }
  const fd = new URLSearchParams();
  fd.set('id', SBE.id);
  fd.set('label', label);
  let r;
  try { r = await (await fetch('/storyboard/edit/version', { method: 'POST', body: fd })).json(); }
  catch (e) { r = { ok: false, error: String(e) }; }
  if (!r || !r.ok) {
    phosToast((r && r.error) || 'That save could not be named.', { kind: 'danger' });
    return;
  }
  sbeNameMode('');
  if (box) box.value = '';
  if (r.already) {
    phosToast('That save already has a name — nothing was changed.', {});
  } else {
    phosToast('Kept "' + label + '" — revision ' + sbeNum(r.revision, 0) +
              '. The fifty-save prune will never take it.',
              { kind: 'success', duration: 5000 });
  }
  await sbeVersionsLoad();
}

// A DESTRUCTIVE ACTION ASKS IN THE PANEL'S OWN VOICE. This used to be a
// `window.confirm` — a Chrome-chrome modal in the middle of a claude.ai-grade
// surface, and the only piece of UI in this feature that was not the app's
// own. Its sibling `sbeDraftRename` carries the comment saying so, and the
// test enforcing it named only `window.prompt`, which is how the confirm
// walked past. Two clicks, four seconds to change your mind.
function sbeDraftDelete(slug, name, btn) {
  if (btn && btn.dataset && btn.dataset.armed !== '1') {
    btn.dataset.armed = '1';
    btn.dataset.was = btn.textContent;
    btn.textContent = 'Delete?';
    btn.title = 'Press again to delete "' + name + '". Its past saves go with '
              + 'it; the other drafts are untouched.';
    btn.classList.add('is-danger');
    setTimeout(() => {
      if (!btn.dataset || btn.dataset.armed !== '1') return;
      btn.dataset.armed = '';
      btn.textContent = btn.dataset.was || 'Delete';
      btn.classList.remove('is-danger');
    }, 4000);
    return;
  }
  return sbeDraftOp('delete', { slug: slug }, () => 'Deleted "' + name + '".');
}

// The draft's name IS the button, so which document is being cut is on screen
// without asking; the dot is the whole of "not saved yet".
function sbePaintDraft() {
  const row = (SBE.drafts || []).filter(d => d.active)[0];
  const el = sbeEl('sbeDraftName');
  // "Draft 2 of 3", not "Draft 2": the chip is the ONE place this __SEQ__ says
  // how many drafts it has, now that the Drafts button is inside the ⋯ menu.
  // Without the count the chip reads as a label rather than as a door.
  const n = (SBE.drafts || []).length;
  if (el) {
    el.textContent = ((row && row.name) || 'Draft')
      + (n > 1 ? ' of ' + n : '');
  }
  const dot = sbeEl('sbeDraftDot');
  if (dot) dot.hidden = !SBE.dirty;
  const btn = sbeEl('sbeKeepBtn');
  if (btn) {
    btn.title = SBE.dirty
      ? 'Unsaved changes in "' + ((row && row.name) || 'this draft') +
        '" — press Save to store them'
      : 'Draft "' + ((row && row.name) || '') + '" — saved';
  }
}

// The offer, never the action.
// WHICH ONE IS OPEN. Urgency order, and it is not the DOM order: a conflict
// means two tabs are fighting over the file and every other message is about
// the file you are about to lose; the alarm means nothing is being stored at
// all; the validation list means this particular save did not land; the
// recovery offer is a question about last session and can wait behind all
// three. `SBE.noticeLead` is the user overriding that — clicking a chip opens
// it — and it is cleared the moment that notice goes away.
// `sbeKeyed` sits last on purpose: it reports something that ALREADY WORKED —
// a card was made usable — so it is the one notice on this screen that is never
// a question, and it must not out-rank a save that is failing.
const SBE_NOTICE_ORDER = ['sbeConflict', 'sbeAlarm', 'sbeErrors', 'sbeRecover',
                          'sbeKeyed'];

function sbePaintNotices() {
  const wrap = sbeEl('sbeNotices');
  if (!wrap || !wrap.classList) return;
  const open = SBE_NOTICE_ORDER.filter(id => {
    const el = sbeEl(id);
    return el && el.hidden === false;
  });
  wrap.hidden = !open.length;
  // A QUIET NOTICE IS A CHIP EVEN WHEN IT IS ALONE. The snapshot offer is an
  // invitation to go and look, not a question that must be answered before the
  // film can be worked on — so it never takes the width, and it never becomes
  // the lead unless the user clicks it open.
  const loud = open.filter(id => !((sbeEl(id).dataset || {}).quiet));
  let lead = open.indexOf(SBE.noticeLead) >= 0 ? SBE.noticeLead : loud[0];
  for (const id of SBE_NOTICE_ORDER) {
    const el = sbeEl(id);
    if (!el || !el.classList) continue;
    // A LONE NOTICE IS NEVER A CHIP. Folding the only thing on screen would
    // hide the sentence to save room nothing else is asking for — and a
    // notice that is not up at all is not a chip either, or the class outlives
    // the message and the next one to open arrives already folded.
    const fold = open.indexOf(id) >= 0 && id !== lead
              && (open.length > 1 || !!((el.dataset || {}).quiet));
    el.classList.toggle('is-folded', fold);
  }
  return lead;
}

// Clicking a folded chip promotes it. Bound on the CONTAINER, so it survives
// every repaint of the children and needs no rebinding anywhere.
function sbeNoticeOpen(id) {
  SBE.noticeLead = id;
  sbePaintNotices();
}

function sbeNoticeClick(ev) {
  const chip = ev.target.closest ? ev.target.closest('.is-folded') : null;
  if (!chip || !chip.id) return;
  ev.preventDefault();
  sbeNoticeOpen(chip.id);
}

// "Later" — hide the recovery offer for this session without answering it.
// Discard DELETES the backup, so it was never the button for "not now".
function sbeNoticeLater() {
  SBE.backupHidden = true;
  sbePaintRecovery();
}

function sbePaintRecovery() {
  const bar = sbeEl('sbeRecover');
  if (!bar) return;
  const b = SBE.backup;
  bar.hidden = !b || !!SBE.backupHidden;
  if (bar.hidden) { sbePaintNotices(); return; }
  // WHAT IT IS, NOT WHAT TO DO ABOUT IT. The old sentence read out both clip
  // counts and then declared that nothing had changed — a question about a
  // difference it could not name, over a document that had loaded correctly.
  // This says when, and how big, and leaves the decision where versions live.
  sbeEl('sbeRecoverWhat').textContent =
    'from ' + sbeAgo(b.at) + ' · ' + sbeNum(b.clips, 0) + ' clips · ' +
    sbeFmtTime(b.duration) + ' — your saved draft is untouched.';
  sbePaintNotices();
}

async function sbeRecover() {
  const fd = new URLSearchParams();
  fd.set('id', SBE.id);
  let r;
  try { r = await (await fetch('/storyboard/edit/recover', { method: 'POST', body: fd })).json(); }
  catch (e) { r = { ok: false, error: String(e) }; }
  if (!r || !r.ok) {
    phosToast((r && r.error) || 'That backup could not be recovered.', { kind: 'danger' });
    return;
  }
  SBE.undo.length = 0; SBE.redo.length = 0;
  SBE.dirty = false; SBE.dirtyAt = 0;
  sbeAdopt(r, true);
  phosToast('Recovered — ' + SBE.clips.length + ' clip(s). The draft it ' +
            'replaced was kept in this draft\'s history.',
            { kind: 'success', duration: 6000 });
}

async function sbeDiscardBackup() {
  const fd = new URLSearchParams();
  fd.set('id', SBE.id);
  try { await fetch('/storyboard/edit/discard-backup', { method: 'POST', body: fd }); }
  catch (e) {}
  SBE.backup = null;
  sbePaintRecovery();
}

async function sbeRestoreVersion(file) {
  if (!file) return;
  const fd = new URLSearchParams();
  fd.set('id', SBE.id);
  fd.set('file', file);
  // The arrangement on screen is about to be replaced, so it goes to disk
  // first — the server archives what it overwrites, and an unsaved drag would
  // otherwise be the one thing in this feature that history could not keep.
  // Checked, for the same reason `sbeDraftOp` checks its backup: if the save
  // 409s or errors the server archives the OLD document, restore overwrites,
  // and the work goes with the undo stack that could have brought it back.
  if (SBE.dirty && !SBE.conflict && !(await sbeSave(true))) {
    phosToast('Your current arrangement could not be saved, so restoring ' +
              'would lose it. Fix the save first.',
              { kind: 'danger', duration: 8000 });
    return;
  }
  let r;
  try { r = await (await fetch('/storyboard/edit/restore', { method: 'POST', body: fd })).json(); }
  catch (e) { r = { ok: false, error: String(e) }; }
  if (!r || !r.ok) {
    phosToast((r && r.error) || 'That version could not be restored.', { kind: 'danger' });
    return;
  }
  SBE.undo.length = 0; SBE.redo.length = 0;
  sbeAdopt(r, true);
  sbeVersionsPaint(r.versions || [], 50);
  phosToast('Restored — ' + (SBE.clips.length) + ' clip(s), ' +
            sbeFmtTime(sbeFilmDuration(SBE.clips)) +
            '. The arrangement it replaced was kept.',
            { kind: 'success', duration: 6000 });
}

// ---------------------------------------------------------------------------
// PAINT
// ---------------------------------------------------------------------------
function sbeSpan() {
  // THE SCROLLER RUNS PAST THE LAST FRAME. The ruler is drawn against the
  // longer of the film and the soundtrack — and then past both, by the slack
  // above, because "the film's length and the audio's placement are
  // independent facts". Without the tail the music could not be dragged one
  // second beyond the final cut: there was no timeline out there to drag it
  // onto. The old `film + 120` ceiling is gone with it; it capped the RULER
  // while the waveform kept going, so the far end of a long track was drawn
  // and unreachable at the same time.
  const film = sbeFilmDuration(SBE.clips);
  const w = sbeMusicWindow(SBE.audio, SBE.peaks ? SBE.peaks.duration : 0);
  const music = (w.film_end === null) ? w.film_start : w.film_end;
  const base = Math.max(film, music, 0);
  const slack = Math.min(SBE_SLACK_MAX,
                         Math.max(SBE_SLACK_MIN, base * SBE_SLACK_RATIO));
  return Math.max(SBE_SPAN_MIN, base + slack);
}

function sbePx(t) { return sbeNum(t) * SBE.pps; }

function sbePaint() {
  if (!SBE.open) return;
  const span = sbeSpan();
  // NEVER FURTHER OUT THAN "THE WHOLE FILM". The floor moves with the window
  // and with the film's length, so widening the window re-fits instead of
  // leaving a strip of empty track on the right, and the slider's left end
  // keeps meaning what it says.
  const floor = sbeZoomMin();
  if (SBE.pps < floor) SBE.pps = floor;
  const width = Math.max(320, Math.ceil(span * SBE.pps) + SBE_TL_PAD);
  const inner = sbeEl('sbeInner');
  if (!inner) return;
  inner.style.width = width + 'px';
  sbePaintRuler(span, width);
  sbePaintWave(span, width);
  sbePaintMusic(width);
  sbePaintOverlays();
  // THE STAGE'S CARD FOLLOWS THE LANE. A title removed from the lane (or
  // undone back onto it) must leave or return on the monitor in the same
  // paint, not on the next scrub — the review caught a deleted title still
  // painting over the picture.
  sbeOvPaint();
  sbePaintTrack();
  sbePaintAudioLane();
  sbePaintHead();
  sbePaintInspector();
  sbePaintHeads();
  sbePaintMix();
  sbePaintKeys();
  sbePaintChrome();
  // AN EDIT MOVES THE GROUND. Every mutation, undo, redo and adopt lands
  // here, so this is the one place that cannot be forgotten. Unforced: the
  // slip tolerance keeps a drag from re-seeking sixty times a second.
  sbeStripSync();
  // LAST, and after the inspector: the row's budget is what the column has
  // left once everything else has been laid out, and the inspector is the
  // one that changes height when a clip is selected.
  sbeFitMonitors();
}

// Measure, then size. Everything in the cut column except the monitor row and
// the timeline is read off the DOM here; the timeline is counted at the height
// it is ENTITLED to rather than at its live height, because it is the row that
// absorbs whatever is left over — reading it would make this function its own
// input and the two would oscillate. That entitlement used to be the CSS floor
// and is now the user's own number, which is the whole of the resize handle:
// nothing is resized, one term of this subtraction moved.
function sbeFitMonitors() {
  const row = sbeEl('sbeMonitors');
  const col = sbeEl('edStage');
  const plan = sbeEl('sbTimeline');
  if (!row || !col || !plan || plan.hidden) return;
  // Inside the stacking breakpoint the PAGE is the scroller and the column
  // has no fixed height to budget against, so the CSS there is the answer.
  if (window.matchMedia && window.matchMedia('(max-width: 900px)').matches) {
    for (const p of ['--sbe-prog-h', '--sbe-src-h', '--sbe-rail-w', '--sbe-row-h']) {
      row.style.removeProperty(p);
    }
    // The lanes go back to their CSS heights too — a stacked page has no
    // column to take the height from, and the handle is display:none there.
    for (const p of ['--sbe-tl-h', '--sbe-ov-h', '--sbe-track-h',
                     '--sbe-alane-h', '--sbe-wave-h']) {
      plan.style.removeProperty(p);
    }
    return;
  }
  const gap = parseFloat(getComputedStyle(plan).rowGap) || 8;
  let used = 0, shown = 0;
  for (const kid of plan.children) {
    const h = kid.getBoundingClientRect().height;
    if (!h && kid !== row) continue;
    shown++;
    if (kid === row) continue;
    used += (kid.id === 'sbeTlWrap') ? 0 : h;
  }
  // A monitor column is taller than its picture by its own label bar and the
  // gap above it. Read off the PROGRAM column, not the row: the row's height
  // is the thing being solved for, and reading that would be a circle. The
  // bar has a fixed min-height and does not wrap, so this is a constant.
  const prog = sbeEl('sbeStage');
  const col2 = prog ? prog.parentElement : null;
  const chrome = (prog && col2)
    ? Math.max(0, Math.round(col2.getBoundingClientRect().height -
                             prog.getBoundingClientRect().height))
    : 30;
  // WHAT THE TWO OF THEM HAVE TO SHARE, and then the split. The ceiling on
  // the timeline's share is the one clamp that keeps this honest: whatever the
  // handle was dragged to, the monitors keep SBE_MON_MIN_H — a 120px picture
  // is small but it is still a picture, and it is the same floor sbeMonitorFit
  // has always refused to go under.
  const avail = col.clientHeight - used - gap * Math.max(0, shown - 1) - chrome;
  SBE.tlMax = Math.max(SBE_TL_MIN_H,
                       Math.min(SBE_TL_MAX_H, Math.round(avail - SBE_MON_MIN_H)));
  // The PREFERENCE is never overwritten by a window that cannot honour it —
  // shrink the browser and the timeline gives the height back; widen it again
  // and the drag the user made is still there.
  const want = sbeTlClamp(SBE.tlH, SBE.tlMax);
  const budget = avail - want;
  const apply = (b) => {
    const fit = sbeMonitorFit(row.clientWidth, b);
    row.style.setProperty('--sbe-prog-h', Math.round(fit.progH) + 'px');
    row.style.setProperty('--sbe-src-h', Math.round(fit.srcH) + 'px');
    row.style.setProperty('--sbe-rail-w', Math.round(fit.rail) + 'px');
    row.style.setProperty('--sbe-row-h', Math.round(fit.progH + chrome) + 'px');
    // A 16:9 PAIR IS USUALLY WIDTH-LIMITED, so the monitors routinely decline
    // part of the budget they were offered. That leftover has always gone to
    // the timeline — it is the flex row — but it went entirely onto the
    // picture track, which is capped at 240 and then wastes it. Handing it to
    // the same distribution the drag uses means a tall window arrives with
    // taller sound lanes instead of a dead band under them.
    // THE TERM IS SIGNED, and that is what makes the correction below work at
    // the top of the handle's range: there the monitors are already at their
    // own floor and cannot give anything back, so whatever the column is
    // overflowing by has to come off the TIMELINE or it does not come off
    // anything. Floored, never negative — the box may not shrink into itself.
    sbeApplyTl(Math.max(SBE_TL_MIN_H, want + Math.round(b - fit.progH)));
  };
  apply(budget);
  // ONE CORRECTION, NEVER A LOOP. Rounding, a sticky bar's own padding and a
  // row that came out a pixel taller than the sum of its parts all land here
  // as a few pixels of column scroll — measured at 10px at 1900x1000. Reading
  // the overflow back and taking it off the budget is exact where a guessed
  // safety margin would be a permanent tax on the picture. Guarded so it can
  // only ever shrink, and only once per paint.
  const over = col.scrollHeight - col.clientHeight;
  if (over > 1 && over < 200) apply(budget - over);
  // THE TWO SOUND LANES ARE DRAWN, NOT STYLED, so a variable cannot resize
  // them on its own: the soundtrack's canvas carries a backing store and a
  // clip strip's waveform is emitted at its own height. Both were painted
  // BEFORE this function ran — it is deliberately last, because the
  // inspector's height is one of its inputs — so when this pass moves the
  // height, they are one pass behind and a 54px bitmap is left stretched over
  // a 122px lane. Redrawn here, and only when the number actually moved.
  if (SBE.laneAt !== SBE.tlNow) {
    SBE.laneAt = SBE.tlNow;
    sbePaintAudioLane();
    const inner = sbeEl('sbeInner');
    if (inner) sbePaintWave(sbeSpan(), parseFloat(inner.style.width) || 0);
  }
}

// The one writer of the timeline's height. Five variables on #sbTimeline: the
// box, and the four lanes inside it. Everything downstream — the CSS, the
// strip waveform's viewBox, the soundtrack canvas — reads the same numbers, so
// the picture and the pointer maths cannot disagree about where a strip is.
function sbeApplyTl(px) {
  const tl = Math.max(SBE_TL_MIN_H, Math.round(sbeNum(px, SBE_TL_MIN_H)));
  SBE.tlNow = tl;
  const plan = sbeEl('sbTimeline');
  if (!plan || !plan.style) return;
  const L = sbeLaneHeights(tl);
  plan.style.setProperty('--sbe-tl-h', tl + 'px');
  plan.style.setProperty('--sbe-ov-h', L.ov + 'px');
  plan.style.setProperty('--sbe-track-h', L.track + 'px');
  plan.style.setProperty('--sbe-alane-h', L.alane + 'px');
  plan.style.setProperty('--sbe-wave-h', L.wave + 'px');
}

// How tall one clip's sound strip is drawn, in the same pixels the lane was
// given. Read from the number rather than from the DOM because the lane is
// painted BEFORE sbeFitMonitors runs in any given paint, and a strip that
// measured the live lane would be drawing last frame's height.
function sbeStripH() {
  // SEVEN, not six: the lane is border-box with a 1px top rule, and the strip
  // inside it is inset 3px top and bottom. Off by that one pixel the viewBox
  // and the box disagree and every waveform is drawn 4% tall.
  return Math.max(14, sbeLaneHeights(SBE.tlNow || SBE.tlH).alane - 7);
}

// ---- the drag itself ----------------------------------------------------
// Pointer capture, because a resize that stops working the moment the pointer
// leaves the 10px strip it started on is a resize that fights you.
function sbeTlGrabDown(ev) {
  SBE.tlDrag = { y0: ev.clientY, h0: SBE.tlNow || SBE.tlH };
  try { ev.currentTarget.setPointerCapture(ev.pointerId); } catch (e) {}
  document.body.classList.add('sbe-resizing');
  // NOT preventDefault: cancelling pointerdown suppresses the compatibility
  // mouse events, and the double-click reset is one of them. The body class
  // is what stops the drag selecting the transport's labels.
}

function sbeTlGrabMove(ev) {
  const d = SBE.tlDrag;
  if (!d) return;
  // UP IS TALLER. clientY falls as the pointer rises, so the delta is
  // subtracted and the edge follows the pointer exactly.
  sbeTlSet(d.h0 - (ev.clientY - d.y0));
  ev.preventDefault();
}

function sbeTlGrabUp(ev) {
  if (!SBE.tlDrag) return;
  SBE.tlDrag = null;
  document.body.classList.remove('sbe-resizing');
  try { ev.currentTarget.releasePointerCapture(ev.pointerId); } catch (e) {}
  sbeTlPrefWrite(SBE.tlH);
}

// Set, lay out, repaint — in that order. sbeApplyTl runs FIRST so the lanes
// already have their new heights when sbePaint draws into them, and
// sbeFitMonitors (last inside sbePaint) gives the monitors what is left and
// corrects tlNow for what the window could actually spare.
function sbeTlSet(px) {
  SBE.tlH = sbeTlClamp(px, SBE.tlMax);
  sbeApplyTl(SBE.tlH);
  sbePaint();
}

// DOUBLE-CLICK RESETS, the same gesture the fade corners and the level line
// already answer to on this timeline.
function sbeTlReset() {
  sbeTlSet(SBE_TL_MIN_H);
  sbeTlPrefWrite(SBE.tlH);
}

function sbeTlGrabKey(ev) {
  const step = ev.shiftKey ? SBE_TL_STEP_BIG : SBE_TL_STEP;
  const now = SBE.tlNow || SBE.tlH;
  if (ev.key === 'ArrowUp') sbeTlSet(now + step);
  else if (ev.key === 'ArrowDown') sbeTlSet(now - step);
  else if (ev.key === 'Home') sbeTlSet(SBE_TL_MIN_H);
  else return;
  // The Editor's own shortcuts own the arrows (a frame at a time) and Home.
  // A focused handle is not the timeline, so the event stops here.
  ev.preventDefault();
  ev.stopPropagation();
  sbeTlPrefWrite(SBE.tlH);
}

function sbePaintRuler(span, width) {
  const r = sbeEl('sbeRuler');
  const steps = [0.5, 1, 2, 5, 10, 15, 30, 60, 120];
  let step = steps[steps.length - 1];
  for (const s of steps) { if (s * SBE.pps >= 54) { step = s; break; } }
  // ONE FORMAT PER ZOOM, DECIDED FROM THE STEP. Stripping the fractional
  // zeros per LABEL made neighbouring ticks different shapes — 0:00.50 beside
  // 0:01 beside 0:01.50 — so the eye read a rhythm change where the time is
  // perfectly regular, and tabular numerals cannot save a ruler whose labels
  // have different lengths.
  const decimals = step < 1;
  let html = '';
  for (let t = 0; t <= span + 1e-6; t += step) {
    const x = sbePx(t);
    const lab = decimals ? sbeFmtTime(t) : sbeFmtTime(t).replace(/\.00$/, '');
    html += '<i style="left:' + x.toFixed(1) + 'px"></i>' +
            '<b style="left:' + x.toFixed(1) + 'px">' + lab + '</b>';
  }
  // THE SONG'S SECTIONS, as faint bands behind the ticks: the arc the
  // Director cut to, on the film's clock. Track seconds become film seconds
  // through the same music window the bed and the beat grid use, so a
  // trimmed or slid soundtrack moves its sections with it.
  html += sbeSectionBands(SBE.sections, SBE.audio, SBE.peaks ? SBE.peaks.duration : 0, span);
  r.innerHTML = html;
  r.style.width = width + 'px';
  if (r.classList) r.classList.toggle('has-sections', !!(SBE.sections && SBE.sections.length));
}

function sbeSectionBands(sections, audio, trackDur, span) {
  if (!sections || !sections.length) return '';
  const w = sbeMusicWindow(audio || {}, trackDur || 0);
  let out = '';
  for (const s of sections) {
    const a = w.film_start + (sbeNum(s.start) - w.head);
    const b = w.film_start + (sbeNum(s.end) - w.head);
    const x0 = Math.max(0, a), x1 = Math.min(span, b);
    if (x1 - x0 <= 1e-6) continue;
    const px = sbePx(x1 - x0);
    const e = sbeNum(s.energy);
    const heat = e >= 0.75 ? 'loud' : (e <= 0.35 ? 'quiet' : 'steady');
    out += '<u class="sbe-sec is-' + escapeHtml(String(s.label || 'verse')) + '" ' +
           'title="' + escapeHtml(String(s.label || '') + ' · ' + sbeFmtTime(x0) + '–' + sbeFmtTime(x1)
             + ' · ' + heat + ' — the Director cuts ' + (String(s.label) === 'chorus' ? 'fastest' : (String(s.label) === 'intro' || String(s.label) === 'outro' ? 'slowest' : 'at the base pace') + ' here')) + '" ' +
           'style="left:' + sbePx(x0).toFixed(1) + 'px;width:' + px.toFixed(1) + 'px">' +
           (px > 40 ? escapeHtml(String(s.label || '')) : '') + '</u>';
  }
  return out;
}

function sbePaintWave(span, width) {
  const cv = sbeEl('sbeWave');
  const none = sbeEl('sbeWaveNone');
  // The soundtrack's lane takes its share of a dragged edge too — it is the
  // other surface you edit sound on. Read from the same distribution the CSS
  // variable came from, so the canvas's backing store and its box agree.
  const h = sbeLaneHeights(SBE.tlNow || SBE.tlH).wave;
  if (!SBE.peaks) {
    cv.hidden = true;
    none.hidden = false;
    none.style.width = width + 'px';
    none.innerHTML = SBE.audio && SBE.audio.path
      ? 'No waveform yet — press Prepare to build the proxies, the waveform and the beat grid.'
      : 'No soundtrack on this timeline. Point one at it below and press Prepare.';
    return;
  }
  none.hidden = true;
  cv.hidden = false;
  const dpr = window.devicePixelRatio || 1;
  cv.width = Math.ceil(width * dpr);
  cv.height = Math.ceil(h * dpr);
  cv.style.width = width + 'px';
  cv.style.height = h + 'px';
  const g = cv.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, width, h);
  const css = getComputedStyle(document.body);
  const mid = h / 2;

  // Beats go UNDER the waveform — they are the reference, not the subject.
  // Downbeats go OVER it, further down this function: a bar line you cannot
  // find behind the loudest part of the track is not a grid, it is a texture.
  const guess = sbeGridIsAGuess(SBE.beats);
  const grid = sbeBeatGrid(SBE.beats, 0, sbeSpan(), (SBE.audio || {}).offset);
  for (const b of grid) {
    if (b.down) continue;
    g.strokeStyle = guess ? 'rgba(182,191,209,0.16)' : 'rgba(88,166,255,0.30)';
    g.lineWidth = 1;
    const x = Math.round(sbePx(b.t)) + 0.5;
    g.beginPath(); g.moveTo(x, 0); g.lineTo(x, h); g.stroke();
  }

  const off = sbeNum((SBE.audio || {}).offset);
  const p = SBE.peaks;
  const win = sbeMusicWindow(SBE.audio, p.duration);
  const w = Math.max(1, SBE.pps / p.rate);
  const strong = (css.getPropertyValue('--border-strong') || '#3d477a').trim();
  // THE TRIMMED-AWAY PART IS STILL DRAWN, faintly. A trim you cannot see the
  // other side of is a trim you cannot pull back — the seconds are not gone,
  // they are outside the window, and the picture should say which.
  for (let i = 0; i < p.count; i++) {
    const tt = i / p.rate;                 // where this bucket is IN THE TRACK
    const t = tt - off;                    // ...and where that lands on the film
    if (t < 0) continue;
    const x = sbePx(t);
    if (x > width) break;
    const live = tt >= win.head - 1e-9
              && (win.tail === null || tt <= win.tail + 1e-9);
    g.fillStyle = strong;
    g.globalAlpha = live ? 1 : 0.28;
    const y0 = mid - p.hi[i] * mid * 0.94;
    const y1 = mid - p.lo[i] * mid * 0.94;
    g.fillRect(x, y0, w, Math.max(1, y1 - y0));
  }
  g.globalAlpha = 1;
  for (const b of grid) {
    if (!b.down) continue;
    g.strokeStyle = guess ? 'rgba(182,191,209,0.42)' : 'rgba(88,166,255,0.85)';
    g.lineWidth = 1;
    const x = Math.round(sbePx(b.t)) + 0.5;
    g.beginPath(); g.moveTo(x, 0); g.lineTo(x, h); g.stroke();
  }

  // Where the music runs out. Past this line there is no grid and no waveform,
  // and the film is on its own.
  if (p.duration > 0) {
    const x = Math.round(sbePx(p.duration - off)) + 0.5;
    if (x < width) {
      g.strokeStyle = 'rgba(210,153,34,0.6)';
      g.lineWidth = 1;
      g.beginPath(); g.moveTo(x, 0); g.lineTo(x, h); g.stroke();
    }
  }
}

// THE BLOCK, over the picture of the track. It exists whenever there is a
// soundtrack and a length to give it — the waveform is not a precondition,
// because `audio.duration` is known before prepare has run and a strip you
// cannot grab until you have pressed Prepare is a strip that looks broken.
function sbePaintMusic(width) {
  const el = sbeEl('sbeMusicClip');
  if (!el) return;
  const a = SBE.audio;
  const w = sbeMusicWindow(a, SBE.peaks ? SBE.peaks.duration : 0);
  if (!a || !a.path || w.duration <= 0 || w.tail === null) {
    el.hidden = true;
    return;
  }
  el.hidden = false;
  const left = sbePx(w.film_start);
  const right = sbePx(w.film_end === null ? w.film_start : w.film_end);
  el.style.left = left.toFixed(1) + 'px';
  el.style.width = Math.max(3, right - left).toFixed(1) + 'px';
  el.classList.toggle('is-sel', SBE.sel === '@music');
  const name = String(a.path || '').split('/').pop();
  const played = Math.max(0, (w.film_end === null ? 0 : w.film_end) - w.film_start);
  const label = name + ' · ' + sbeFmtTime(w.film_start) + ' · ' +
    played.toFixed(2) + 's' + (w.trimmed ? ' · trimmed' : '');
  sbeEl('sbeMusicName').textContent = (right - left > 120) ? label : '';
  el.title = name + '\nStarts at ' + sbeFmtTime(w.film_start) +
    ' of the __SEQ__, from ' + sbeFmtTime(w.head) + ' of the track.' +
    '\nDrag to move it (the others stay) · pull either end to trim · hold Alt to ignore the cuts · hold ⌘ to slide everything after it.' +
    '\nCorners fade it · double-click the level line to add a point.';
  sbePaintBedLevel(el, Math.max(3, right - left));
}

// THE BED'S LEVEL LINE, POINTS AND FADE RAMPS. Everything here is the strip
// painter's arithmetic with one substitution — the bed's curve instead of a
// clip's — because a second copy of it is a second thing to drift. The
// waveform is NOT redrawn: `#sbeWave` already paints the whole track under
// this block, and drawing it twice would be two pictures of one sound.
function sbePaintBedLevel(el, px) {
  const layer = el.querySelector('.sbe-mc-lvl');
  const band = el.querySelector('.sbe-fade-band');
  if (!layer) return;
  const a = SBE.audio || {};
  // THE FILM, NOT THE PROBE. The bed's clock is the document's own length and
  // the film's when the document is silent — the render reads edit.json and
  // has never seen peaks.json, so anything the probe contributed here would be
  // a level only the browser could compute. See `sbeBedLen`.
  const dur = sbeFilmDuration(SBE.clips);
  const n = sbeBedLen(a, dur);
  const H = Math.max(14, Math.round(el.getBoundingClientRect().height)
                     || (sbeLaneHeights(SBE.tlNow || SBE.tlH).wave - 8));
  // WHAT THE RENDER WILL DO, not what was authored. The fader and the duck
  // are in this line as well as the envelope, so the picture of the level is
  // the level — which is the entire complaint this feature answers.
  const curve = sbeBedGainPoints(a, SBE.clips, dur);
  const pts = curve.length ? curve : [[0, 1], [n || 1, 1]];
  const yOf = g => sbeStripY(g, H);
  const xOf = t => (n > 0 ? (t / n) * px : 0);
  let line = '';
  for (let i = 0; i < pts.length; i++) {
    line += (i ? 'L' : 'M') + xOf(pts[i][0]).toFixed(2) + ','
          + yOf(pts[i][1]).toFixed(2);
  }
  let body = '<path class="sbe-lvl-u" d="' + line + '"/>'
           + '<path class="sbe-lvl" d="' + line + '"/>';
  // THE TARGET STOPS WHERE THE CORNER HANDLES START — the same clip the
  // strips apply, for the same reason: two controls claiming one rectangle is
  // how the fade handle got lost inside the trim grip in the first place.
  const hit = sbeLvlHitPath(pts, xOf, yOf, SBE_LVL_CLEAR, px - SBE_LVL_CLEAR);
  if (hit) body += '<path class="sbe-lvl-hit" d="' + hit + '"/>';
  const gh = SBE.kfGhost;
  if (gh && gh.id === '@music') {
    body += '<circle class="sbe-kf-ghost" cx="' + xOf(gh.t).toFixed(2) + '" '
          + 'cy="' + yOf(gh.g).toFixed(2) + '" r="'
          + Math.max(3.6, Math.min(6.5, H / 5.5)).toFixed(2) + '"/>';
  }
  // Only the USER's points get a handle, exactly as on a strip: the fade
  // corners are already a gesture of their own, and a dot on them would offer
  // two ways to drag one number in opposite directions.
  const own = sbeAfx(a, n).points;
  const R = Math.max(3.2, Math.min(6, H / 6));
  for (let i = 0; i < own.length; i++) {
    body += '<circle class="sbe-kf" data-bkf="' + i + '" '
          + 'cx="' + xOf(own[i][0]).toFixed(2) + '" '
          + 'cy="' + yOf(sbeLerpGain(curve.length ? curve : own, own[i][0])).toFixed(2) + '" '
          + 'style="--kf-r:' + R.toFixed(2) + 'px" '
          + 'r="' + R.toFixed(2) + '"/>';
  }
  layer.innerHTML = '<svg width="' + Math.max(1, px).toFixed(0) + '" height="' + H
    + '" viewBox="0 0 ' + Math.max(1, px).toFixed(0) + ' ' + H
    + '" preserveAspectRatio="none">' + body + '</svg>';
  // A BED TOO SHORT TO CARRY THEM OFFERS NEITHER. An 8px stub of a control is
  // worse than none — the same rule the strips' handles follow.
  if (band) band.hidden = !(px >= 2 * SBE_LVL_CLEAR + SBE_LVL_MIN_SPAN);
}

// ---- the music, dragged --------------------------------------------------
// Same three modes as a clip, the same pointer-capture shape, and the same
// rule about a gesture that moved the pointer without moving the film.
// WHERE THE POINTER IS ON THE BED, in the coordinates a level cares about.
// The strip lane's `sbeStripAt` for the soundtrack: how far along the played
// window, and what gain that height means.
function sbeBedAt(ev) {
  const el = sbeEl('sbeMusicClip');
  if (!el || el.hidden) return null;
  const a = SBE.audio;
  if (!a || !a.path) return null;
  // NOT WHILE THE POINTER IS ON A CORNER HANDLE. The clipped hit path keeps
  // the LINE off those pixels, but the proximity test below knows nothing
  // about paths — and it is what the hover ghost asks, so without this a fade
  // handle would sit under a ghost promising a point it could never place.
  if (ev.target.closest('.sbe-fade-h')) return null;
  const r = el.getBoundingClientRect();
  const n = sbeBedLen(a, sbeFilmDuration(SBE.clips));
  if (n <= 0) return null;
  return {
    rect: r, len: n,
    t: n * Math.max(0, Math.min(1, (ev.clientX - r.left) / Math.max(1, r.width))),
    g: sbeStripGain(ev.clientY, r.top, r.height),
  };
}

// A click within SBE_LVL_GRAB of the bed's level places a point and starts
// dragging it — one gesture, the same one the strips answer to.
function sbeBedLevelClick(ev) {
  const at = sbeBedAt(ev);
  if (!at) return false;
  const dur = sbeFilmDuration(SBE.clips);
  const onLine = sbeStripY(sbeBedGainAt(SBE.audio, SBE.clips, dur, at.t),
                           at.rect.height);
  const near = Math.abs((ev.clientY - at.rect.top) - onLine) <= SBE_LVL_GRAB;
  if (!near && !ev.target.closest('.sbe-lvl-hit')) return false;
  const before = JSON.stringify(SBE.audio || {});
  const next = sbeBedAddKeyframe(SBE.audio, at.t, at.g, dur);
  if (!next) return false;
  sbeSetAudio(next);
  SBE.sel = '@music';
  SBE.kfGhost = null;
  const pts = sbeAfx(SBE.audio, at.len).points;
  let idx = 0;
  for (let i = 0; i < pts.length; i++) {
    if (Math.abs(pts[i][0] - sbeRound(at.t)) < 1e-3) { idx = i; break; }
  }
  SBE.musicDrag = { mode: 'bkf', index: idx, rect: at.rect,
                    before: before, moved: true };
  try { sbeEl('sbeMusicLane').setPointerCapture(ev.pointerId); } catch (e) {}
  SBE.dirty = true;
  sbeSetState('unsaved changes', 'dirty');
  sbePaint();
  return true;
}

function sbeOnMusicDown(ev) {
  const el = sbeEl('sbeMusicClip');
  if (!el || el.hidden) return;
  const a = SBE.audio;
  if (!a || !a.path) return;
  const lane = sbeEl('sbeMusicLane');
  const w = sbeMusicWindow(a, SBE.peaks ? SBE.peaks.duration : 0);
  // The BED's clock, which is not the block's picture: see `sbeBedLen`.
  const dur = sbeFilmDuration(SBE.clips);
  SBE.sel = '@music';
  // GESTURE PRECEDENCE, MOST SPECIFIC FIRST, and it is deliberately the same
  // list the clip-sound lane uses — one rule for both, or the soundtrack would
  // behave differently from the strip above it for no reason a user could name.
  //   1. a point on the level line
  //   2. a corner fade handle  (sits over the grip's hit area)
  //   3. the level line itself
  //   4. the grips             (trim)
  //   5. the block             (move)
  const kf = ev.target.closest('.sbe-kf');
  if (kf && kf.dataset.bkf !== undefined) {
    const before = JSON.stringify(a);
    // SHIFT-CLICK DELETES. A modifier rather than a second affordance drawn on
    // a 6px target, which would be a delete nobody meant.
    if (ev.shiftKey) {
      const gone = sbeBedDeleteKeyframe(a, sbeNum(kf.dataset.bkf), dur);
      if (gone) { sbeSetAudio(gone); sbeMusicCommit(before); }
      ev.preventDefault();
      return;
    }
    SBE.musicDrag = { mode: 'bkf', index: sbeNum(kf.dataset.bkf),
                      rect: el.getBoundingClientRect(),
                      before: before, moved: false };
    try { lane.setPointerCapture(ev.pointerId); } catch (e) {}
    ev.preventDefault();
    sbePaint();
    return;
  }
  const fh = ev.target.closest('.sbe-fade-h');
  if (fh && fh.dataset.bfade !== undefined) {
    const e0 = sbeAfx(a, sbeBedLen(a, dur));
    SBE.musicDrag = {
      mode: 'bfade', edge: fh.dataset.bfade, x0: ev.clientX,
      f0: (fh.dataset.bfade === 'out') ? e0.fade_out : e0.fade_in,
      before: JSON.stringify(a), moved: false,
    };
    try { lane.setPointerCapture(ev.pointerId); } catch (e) {}
    ev.preventDefault();
    sbePaint();
    return;
  }
  if (!ev.target.closest('.sbe-grip') && sbeBedLevelClick(ev)) {
    ev.preventDefault();
    return;
  }
  const grip = ev.target.closest('.sbe-grip');
  SBE.musicDrag = {
    mode: grip ? (grip.classList.contains('r') ? 'trimR' : 'trimL') : 'move',
    x0: ev.clientX,
    t0: sbeTimeFromEvent(ev, lane),
    fs0: w.film_start,
    fe0: (w.film_end === null ? w.film_start : w.film_end),
    before: JSON.stringify(a), moved: false,
  };
  el.classList.add('is-drag');
  try { lane.setPointerCapture(ev.pointerId); } catch (e) {}
  ev.preventDefault();
  sbePaint();
}

// ONE WRITER for the soundtrack object, so every gesture that changes it —
// drag, trim, fade, point, level, duck — marks the document dirty the same way
// and none of them can forget the queued save.
function sbeSetAudio(next) {
  SBE.audio = next;
  SBE.edit = SBE.edit || {};
  SBE.edit.audio = next;
}

function sbeMusicCommit(beforeJson) {
  // THE FINGERPRINT IS THE WHOLE OBJECT, not a list of three field names. The
  // list was `offset`, `trim_start`, `trim_end` — written when those were all
  // a soundtrack had — and a fade, a keyframe or a level change would have
  // compared EQUAL and been silently discarded on pointerup. That defect has
  // already shipped twice on this timeline under the name `sbeDragFingerprint`.
  if (JSON.stringify(SBE.audio || {}) === beforeJson) { sbePaint(); return; }
  SBE.undo.push(sbeSnapshot(JSON.parse(beforeJson)));
  if (SBE.undo.length > SBE_UNDO_MAX) SBE.undo.shift();
  SBE.redo.length = 0;
  SBE.dirty = true;
  sbeSetState('unsaved changes', 'dirty');
  sbePaint();
  sbeQueueSave();
}

function sbeOnMusicMove(ev) {
  const d = SBE.musicDrag;
  const dur = sbeFilmDuration(SBE.clips);   // the BED's clock — see sbeBedLen
  // NOTHING BEING DRAGGED IS A HOVER, and a hover is where the level line
  // teaches — the same answer the strip lane gives.
  if (!d) { sbeBedGhost(ev); return; }
  if (d.mode === 'bkf') {
    const r = d.rect;
    const n = sbeBedLen(SBE.audio, dur);
    const t = n * Math.max(0, Math.min(1,
      (ev.clientX - r.left) / Math.max(1, r.width)));
    d.moved = true;
    const next = sbeBedMoveKeyframe(SBE.audio, d.index, t,
                                    sbeStripGain(ev.clientY, r.top, r.height), dur);
    if (next) sbeSetAudio(next);
    sbePaint();
    return;
  }
  if (Math.abs(ev.clientX - d.x0) > 3) d.moved = true;
  if (!d.moved) return;
  if (d.mode === 'bfade') {
    // The out-fade grows LEFTWARD from the block's right edge, so its pointer
    // delta is negated — same as the strips'.
    const adt = (d.edge === 'out') ? -(ev.clientX - d.x0) / SBE.pps
                                   : (ev.clientX - d.x0) / SBE.pps;
    sbeSetAudio(sbeSetBedFade(SBE.audio, d.edge, Math.max(0, d.f0 + adt), dur));
    sbePaint();
    return;
  }
  const dt = (ev.clientX - d.x0) / SBE.pps;
  const tol = SBE_SNAP_PX / SBE.pps;
  const snapOn = sbeSnapEnabled(ev);
  const marks = sbeMusicSnaps(SBE.clips);
  const anchor = (d.mode === 'trimR') ? d.fe0 : d.fs0;
  const want = sbeSnapToList(Math.max(0, anchor + dt), marks, tol, snapOn);
  const next = sbeMusicEdit(SBE.audio, d.mode, want,
                            SBE.peaks ? SBE.peaks.duration : 0);
  sbeApplyMusic(next);
  sbePaint();
}

function sbeOnMusicUp(ev) {
  const d = SBE.musicDrag;
  SBE.musicDrag = null;
  if (!d) return;
  const el = sbeEl('sbeMusicClip');
  if (el) el.classList.remove('is-drag');
  if (!d.moved) {
    // A press that never travelled is a scrub, exactly as it is anywhere else
    // on this lane — the block covers most of the waveform, so swallowing the
    // click would take the lane's scrubber away with it. A press that landed
    // ON a control is not a scrub, though: a point tapped and released has
    // selected itself, and jumping the playhead would be a surprise.
    if (d.mode !== 'bkf' && d.mode !== 'bfade') {
      sbeSeek(sbeTimeFromEvent(ev, sbeEl('sbeMusicLane')));
      return;
    }
  }
  sbeMusicCommit(d.before);
}

// HOVER TEACHES, on the bed as on the strips: near the line, a ghost point
// follows the pointer so the gesture is visible before it is performed.
function sbeBedGhost(ev) {
  const at = sbeBedAt(ev);
  if (at) {
    const dur = sbeFilmDuration(SBE.clips);
    const onLine = sbeStripY(sbeBedGainAt(SBE.audio, SBE.clips, dur, at.t),
                             at.rect.height);
    if (Math.abs((ev.clientY - at.rect.top) - onLine) <= SBE_LVL_GRAB) {
      const g = SBE.kfGhost;
      if (!g || g.id !== '@music' || Math.abs(g.t - at.t) > 0.01) {
        SBE.kfGhost = { id: '@music', t: at.t, g: at.g };
        sbePaint();
      }
      return;
    }
  }
  if (SBE.kfGhost && SBE.kfGhost.id === '@music') {
    SBE.kfGhost = null;
    sbePaint();
  }
}

// DOUBLE-CLICK ADDS A POINT on the bed's body, exactly as on a strip. Plain
// drag stays "move the music" — the gesture this lane already had — so the
// control case is opt-in and the simple case is untouched.
function sbeOnMusicDbl(ev) {
  const el = sbeEl('sbeMusicClip');
  if (!el || el.hidden || !ev.target.closest('#sbeMusicClip')) return;
  if (ev.target.closest('.sbe-kf') || ev.target.closest('.sbe-fade-h')) return;
  const at = sbeBedAt(ev);
  if (!at) return;
  const before = JSON.stringify(SBE.audio || {});
  const next = sbeBedAddKeyframe(SBE.audio, at.t, at.g,
                                 sbeFilmDuration(SBE.clips));
  if (!next) return;
  SBE.sel = '@music';
  sbeSetAudio(next);
  sbeMusicCommit(before);
  ev.preventDefault();
}

// Writing the three fields back, and the ONE place that commits a soundtrack
// discovered from peaks.json into the document. `sbeFetchPeaks` deliberately
// does not — a track the arrangement was never cut to must not be saved as if
// it had been — but dragging its strip IS the user saying "this one, here",
// so from that gesture on it belongs to the edit.
function sbeApplyMusic(fields) {
  const a = Object.assign({}, SBE.audio || {});
  a.offset = sbeRound(sbeNum(fields.offset));
  if (fields.trim_start > 0) a.trim_start = sbeRound(fields.trim_start);
  else delete a.trim_start;
  if (fields.trim_end !== null && fields.trim_end !== undefined
      && sbeNum(fields.trim_end) > 0) a.trim_end = sbeRound(fields.trim_end);
  else delete a.trim_end;
  sbeSetAudio(a);
}

// ---- THE MIX CONTROLS, on the A2 head -----------------------------------
// The two numbers the renderer used to keep to itself. `sbeBedGainSlide` is
// the live one — the slider moves, the level line under it moves, the bed you
// are hearing moves — and the commit is what enters the undo stack, so a drag
// across the whole range is one undo and not sixty.
function sbeBedGainSlide(v) {
  if (!SBE.audio || !SBE.audio.path) return;
  if (!SBE.bedGainBefore) SBE.bedGainBefore = JSON.stringify(SBE.audio);
  sbeSetAudio(sbeMixWrite(SBE.audio, { bed_gain: Math.max(0, Math.min(1, sbeNum(v) / 100)) }));
  sbePaintMix();
  sbePaint();
  sbeMusicSync();
}

function sbeBedGainCommit(v) {
  if (!SBE.audio || !SBE.audio.path) return;
  const before = SBE.bedGainBefore || JSON.stringify(SBE.audio);
  SBE.bedGainBefore = null;
  sbeSetAudio(sbeMixWrite(SBE.audio, { bed_gain: Math.max(0, Math.min(1, sbeNum(v) / 100)) }));
  sbePaintMix();
  sbeMusicCommit(before);
  sbeMusicSync();
}

function sbeSetBedDuck(on) {
  if (!SBE.audio || !SBE.audio.path) {
    // Nothing to duck. Put the box back rather than recording a decision
    // about a soundtrack that does not exist.
    sbePaintMix();
    return;
  }
  const before = JSON.stringify(SBE.audio);
  sbeSetAudio(sbeMixWrite(SBE.audio, { duck: !!on }));
  sbePaintMix();
  sbeMusicCommit(before);
}

// The head, painted from the document — never the other way round, which is
// the bug `sbeSetMusicMode`'s comment records: a control that writes on every
// load turns every OPEN of a film into a save the user never made.
function sbePaintMix() {
  const a = SBE.audio || {};
  const has = !!a.path;
  const mix = sbeAudioMix(a);
  const sl = sbeEl('sbeBedGain');
  const num = sbeEl('sbeBedGainNum');
  const dk = sbeEl('sbeBedDuck');
  const note = sbeEl('sbeBedDuckNote');
  const head = document.querySelector('.sbe-gh-mus');
  const pct = Math.round(mix.bed_gain * 100);
  if (sl) { sl.value = String(pct); sl.disabled = !has; }
  if (num) num.textContent = pct + '%';
  if (dk) { dk.checked = !!mix.duck; dk.disabled = !has; }
  // A CONTROL THAT HAS STOPPED ACTING SAYS SO, LOUDLY. The duck stands down
  // under an authored envelope — never two curves on one level — and a ticked
  // box that was quietly doing nothing is the silent guard this editor has
  // paid for more than once.
  const off = has && sbeBedDuckSuppressed(a, sbeFilmDuration(SBE.clips));
  if (note) {
    note.hidden = !off;
    note.textContent = off ? 'off — your own level line is driving the bed' : '';
  }
  if (head) head.classList.toggle('is-duck-off', off);
}

function sbePaintTrack() {
  const track = sbeEl('sbeTrack');
  const holes = sbeHoles(SBE.clips);
  // THE MESSAGE BELONGS WHERE THE EMPTINESS IS. This sentence used to live in
  // the inspector, which is a short auto-scrolled box in the corner — so on a
  // brand-new draft the user read "…media pool to put it here, or Add black
  // for a gap to fill", starting mid-word, while the track itself, the thing
  // that was empty, rendered nothing at all.
  let html = SBE.clips.length ? '' :
    '<div class="sbe-track-empty">Nothing on the timeline yet — click a clip '
    + 'in the media pool to put it here, or Add black for a gap to fill.</div>';
  for (const g of holes) {
    const w = sbePx(g.duration);
    html += '<div class="sbe-gap" data-gap-start="' + g.film_start + '" ' +
            'data-gap-dur="' + g.duration + '" ' +
            'title="Nothing plays here. Click to generate a shot for it." ' +
            'style="left:' + sbePx(g.film_start).toFixed(1) + 'px;width:' + w.toFixed(1) + 'px">' +
            (w > 84 ? escapeHtml(g.duration.toFixed(2) + 's hole · fill it') : '+') +
            '</div>';
  }
  for (const c of SBE.clips) {
    const w = sbePx(sbeLen(c));
    const kind = sbeKind(c);
    const bright = sbeBright(c);
    const bad = SBE.errors.byId && SBE.errors.byId[c.id];
    const sp = sbeSpeed(c);
    const flag = c.locked ? 'lock'
      : (Math.abs(sp - 1) >= 1e-6 ? sp.toFixed(2).replace(/\.?0+$/, '') + 'x'
      : (Math.abs(bright) >= 1e-6
          ? (bright > 0 ? '+' : '') + bright.toFixed(2)
          // "slow" is a VIDEO's problem. A still has no proxy because it needs
          // none and a slug has no file at all, so flagging either as
          // un-scrubbable would be advice to run a Prepare that would do
          // nothing.
          : (kind === 'video' && !c.proxy ? 'slow' : '')));
    const cls = 'sbe-clip is-' + kind + (c.id === SBE.sel ? ' is-sel' : '')
              + (c.id === SBE.curId ? ' is-playing' : '')
              + (bad ? ' is-bad' : '') + (c.locked ? ' is-locked' : '')
              + (Math.abs(bright) >= 1e-6 ? ' is-graded' : '')
              + (Math.abs(sp - 1) >= 1e-6 ? ' is-retimed' : '')
              + (!sbeFramingIsNeutral(c) ? ' is-framed' : '')
              + (flag ? ' has-flag' : '');
    // A still and a slug have no source window to report — their only number
    // is the hold, and printing "0.00→3.00" of a clock they do not have reads
    // as a trim somebody made.
    const meta = (kind === 'video')
      ? (sbeNum(c.start).toFixed(2) + '→' + sbeNum(c.end).toFixed(2) +
         ' · ' + sbeLen(c).toFixed(2) + 's' + (Math.abs(sp - 1) >= 1e-6 ? ' @' + sp + 'x' : '')
         + (!sbeFramingIsNeutral(c) ? ' · \u2316' + sbeFraming(c).zoom.toFixed(1) + 'x' : ''))
      : (sbeLen(c).toFixed(2) + 's hold' + (!sbeFramingIsNeutral(c) ? ' · \u2316' + sbeFraming(c).zoom.toFixed(1) + 'x' : ''));
    const label = (kind === 'slug')
      ? 'black'
      : (c.title || String(c.path || '').split('/').pop() || 'clip');
    // The still paints its own picture behind its name. One <img> per still on
    // the track, which is the same budget the pool already spends per row and
    // nothing like the media-element cap a <video> would eat.
    const thumb = (kind === 'still' && c.path)
      ? '<img class="sbe-cl-thumb" alt="" src="/image?w=240&path=' +
        encodeURIComponent(c.path) + '">'
      : '';
    html += '<div class="' + cls + '" data-id="' + escapeHtml(c.id) + '" '
          + 'data-kind="' + escapeHtml(kind) + '" '
          + 'data-source="' + escapeHtml(c.source || 'auto') + '" '
          + 'title="' + escapeHtml(label + (bad ? '\n' + bad[0].message : '')) + '" '
          + 'style="left:' + sbePx(c.film_start).toFixed(1) + 'px;width:' + w.toFixed(1) + 'px">'
          + thumb
          + '<div class="sbe-cl-name">' + escapeHtml(sbeNiceName(label)) + '</div>'
          + (w > 96 ? '<div class="sbe-cl-meta">' + escapeHtml(meta) + '</div>' : '')
          + (flag ? '<div class="sbe-cl-flag">' + escapeHtml(flag) + '</div>' : '')
          + sbeSyncBadge(c)
          + sbeFadeMarks(c)
          + '<div class="sbe-grip l"></div><div class="sbe-grip r"></div>'
          // THE TWO CORNER HANDLES, IN A BAND INSET FROM BOTH GRIPS. The
          // wrapper is what keeps the fade gesture and the trim gesture off
          // each other's pixels (see `.sbe-fade-band`) — they were stacked in
          // the same corner and the trim won almost every aim.
          + (c.locked ? ''
             : '<div class="sbe-fade-band">'
               + '<div class="sbe-fade-h in" data-fade="in"></div>'
               + '<div class="sbe-fade-h out" data-fade="out"></div>'
               + '</div>')
          + '</div>';
  }
  track.innerHTML = html;
  // THE CUTS, each one a handle. A transition is not a lane and not a block:
  // it is a property of the boundary between two clips, so that is where it
  // is found — a small mark on every cut, a band the length of the
  // transition once one exists, and the inspector to set it. Drawn AFTER the
  // blocks so it sits on top of both grips.
  const order = SBE.clips.slice().sort((a, b) => sbeNum(a.film_start) - sbeNum(b.film_start));
  const resolved = sbeTxResolve(order, SBE.transitions || [], sbeFps());
  for (let k = 0; k + 1 < order.length; k++) {
    const a = order[k], b = order[k + 1];
    const at = sbeNum(a.film_end);
    if (sbeNum(b.film_start) - at > 0.5 / sbeFps()) continue;   // a hole, not a cut
    const r = resolved.find(x => x.after_clip === a.id) || null;
    const sel = SBE.txSel === a.id;
    const bad = r && r.problem;
    const x = sbePx(at);
    if (r && !bad) {
      const w = Math.max(6, sbePx(r.duration));
      html = '<div class="sbe-cut is-tx is-' + r.kind + (sel ? ' is-sel' : '') + '" data-after="'
           + escapeHtml(a.id) + '" title="' + escapeHtml(SBE_TX_LABELS[r.kind] + ' · '
           + r.duration.toFixed(2) + 's, centred on the cut. Click to change it.')
           + '" style="left:' + (x - w / 2).toFixed(1) + 'px;width:' + w.toFixed(1) + 'px">'
           + (w > 44 ? '<span>' + escapeHtml((r.kind === 'dissolve' ? '\u2715 ' : '\u25d1 ')
              + r.duration.toFixed(2) + 's') + '</span>' : '') + '</div>';
    } else {
      html = '<div class="sbe-cut' + (sel ? ' is-sel' : '') + (bad ? ' is-bad' : '')
           + '" data-after="' + escapeHtml(a.id) + '" title="'
           + escapeHtml(bad ? r.problem.message : 'Cut. Click to put a transition here.')
           + '" style="left:' + (x - 6).toFixed(1) + 'px"></div>';
    }
    track.insertAdjacentHTML('beforeend', html);
  }
  track.style.width = sbeEl('sbeInner').style.width;
  if (SBE.dropAt !== null && SBE.dropAt !== undefined) {
    const line = document.createElement('div');
    line.className = 'sbe-drop-line';
    line.style.left = sbePx(SBE.dropAt).toFixed(1) + 'px';
    track.appendChild(line);
  }
}

// The ramps, drawn to scale on the block: a fade you can see the length of
// without selecting the clip.
function sbeFadeMarks(c) {
  const e = sbeFx(c);
  let out = '';
  if (e.fade_in > 1e-9) {
    out += '<div class="sbe-cl-fade in" style="left:0;width:'
        + sbePx(e.fade_in).toFixed(1) + 'px"></div>';
  }
  if (e.fade_out > 1e-9) {
    out += '<div class="sbe-cl-fade out" style="right:0;width:'
        + sbePx(e.fade_out).toFixed(1) + 'px"></div>';
  }
  return out;
}

// THE SYNC FLAG, and it goes on BOTH halves — the picture block and the sound
// strip — because the pair is what has drifted and either one is where the eye
// happens to be. It is a button: clicking it is the rematch.
function sbeSyncBadge(c) {
  if (sbeKind(c) !== 'video') return '';
  const w = sbeClipAudio(c);
  // A COUPLED pair is never flagged: its offset is the relationship the user
  // froze and the two travel together, so it cannot come apart. Flagging it
  // would put a permanent warning on every J-cut in the film.
  if (!w.split || w.coupled || sbeAudioInSync(c)) return '';
  const d = sbeAudioDrift(c);
  return '<div class="sbe-sync" data-sync="' + escapeHtml(c.id) + '" '
       + 'title="' + escapeHtml('The sound is ' + sbeDriftLabel(d) + ' out of sync '
           + 'with its own picture (' + (d > 0 ? 'late' : 'early') + '). '
           + 'Click to put it back under the frame it came from.') + '">'
       + escapeHtml(sbeDriftLabel(d)) + '</div>';
}

// THE SOUND, UNDER THE PICTURE THAT MADE IT. One strip per video clip,
// directly beneath its block: linked ones are dim and inert, unlinked ones
// carry the same grips a clip does.
function sbePaintAudioLane() {
  const lane = sbeEl('sbeAudioLane');
  if (!lane) return;
  let html = '';
  for (const c of SBE.clips) {
    if (sbeKind(c) !== 'video') continue;
    const w = sbeClipAudio(c);
    const x = sbePx(w.film_start);
    const px = sbePx(w.len);
    // A CLIP WITH NO SOUND AT ALL SAYS SO. `has_audio` rides on the pool row
    // and the clip (the probe already reads streams for `duration`), and it
    // is absent on documents written before it existed — so only an explicit
    // false is silence, and everything else draws as it always did. The rule
    // that paints this state has shipped since the lane did; the flag that
    // reaches it was hard-coded to false, so the lane claimed sound it did
    // not have.
    const mute = c.has_audio === false;
    const off = sbeClipMuted(c);
    const cls = 'sbe-aclip ' + (w.linked ? 'is-linked' : 'is-split')
              + (w.coupled ? ' is-coupled' : '')
              + (mute ? ' is-mute' : '')
              + (off ? ' is-silenced' : '')
              + (c.id === SBE.sel ? ' is-sel' : '');
    // ONE WORD, EIGHT TIMES, IN A 26PX STRIP. The label is only news when the
    // sound has been pulled off its picture — the lane's own fill says the
    // rest, and a column of identical labels says nothing at all. A COUPLED
    // strip is news of a different kind: it is offset ON PURPOSE and moving
    // with its picture, so the label is the offset itself.
    // MUTED SAYS MUTED, over everything else the strip might have said. The
    // strip STAYS — it is still where the sound would be, and the decision has
    // to be visible and reversible in the place it was made.
    const label = mute ? 'no sound'
                : (off ? ('MUTED · ' + w.len.toFixed(2) + 's')
                : (w.coupled ? ('J-cut · ' + sbeDriftLabel(sbeAudioDrift(c)))
                : (w.linked ? '' : ('sound · ' + w.len.toFixed(2) + 's'))));
    html += '<div class="' + cls + '" data-id="' + escapeHtml(c.id) + '" '
          + 'title="' + escapeHtml(off
              ? 'Muted — this clip\'s own sound is switched off in the preview, the render and the export. Unmute it in the inspector.'
              : w.coupled
              ? 'Linked at ' + sbeDriftLabel(sbeAudioDrift(c)) + ' — the sound keeps this offset and travels with the picture. Unlink it to slide it on its own.'
              : (w.linked
              ? 'This clip\'s sound moves with its picture. Unlink it in the inspector to slide it under the neighbour (a J-cut or an L-cut).'
              : 'Unlinked — drag to slide the sound, pull either end to trim it. The picture does not move.')) + '" '
          + 'style="left:' + x.toFixed(1) + 'px;width:' + Math.max(2, px).toFixed(1) + 'px">'
          + (mute ? '' : sbeStripWave(c, w, px))
          + (px > 70 ? '<span class="sbe-aclip-t">' + escapeHtml(label) + '</span>' : '')
          + (px > 46 ? sbeSyncBadge(c) : '')
          + sbeAudioFadeMarks(c, w)
          + (w.linked ? '' : '<div class="sbe-grip l"></div><div class="sbe-grip r"></div>')
          // The same band the picture block uses, on a strip that can be
          // shorter still — 37px at the lane's floor, which is why the handle
          // is sized against that floor and not against the lane.
          + (w.linked || c.locked ? ''
             : '<div class="sbe-fade-band">'
               + '<div class="sbe-fade-h in" data-afade="in"></div>'
               + '<div class="sbe-fade-h out" data-afade="out"></div>'
               + '</div>')
          + '</div>';
  }
  lane.innerHTML = html;
  lane.style.width = sbeEl('sbeInner').style.width;
}

async function edPoolOverlay(i) {
  const rows = (document.getElementById('edPoolList') || {})._rows || [];
  const r = rows[i];
  if (!r || !SBE.open) {
    phosToast('Open a __SEQ__ first — an overlay belongs to a timeline.', {});
    return;
  }
  // A CARD THAT ARRIVES ON A BLACK PLATE IS FIXED ON THE WAY IN, and only on
  // the way to THIS lane. The server measures the file and either hands back a
  // keyed derivative or the path exactly as it came; either way the lane gets
  // one path, and preview, render and export all read that same file.
  const item = { path: r.path, title: r.title || '', duration_s: 3 };
  let keyed = null;
  try {
    // URLSearchParams, like every other editor POST. A `FormData` here sends
    // multipart, which `_storyboard_post` does not parse — the route answered
    // "that image is not in this panel's outputs" for a file sitting in the
    // outputs folder, and the card was placed plated with nothing on screen
    // to say the fix had not run.
    const fd = new URLSearchParams();
    fd.set('path', r.path);
    const res = await (await fetch('/storyboard/edit/overlay-key',
                                   { method: 'POST', body: fd })).json();
    if (res && res.ok && res.keyed && res.path) { item.path = res.path; keyed = res; }
  } catch (e) {
    // The lane still works with the file exactly as it arrived. A card that
    // could not be measured is not a card that cannot be placed.
  }
  const added = sbeOvAddAt(item, SBE.playhead);
  if (added && keyed) sbeKeyedNotice(added.id, keyed);
}

// SAY SO QUIETLY, AND ALLOW UNDO. Automatic is not the same as silent, and it
// is certainly not the same as irreversible: the picture the user chose was
// changed, so the change is named, and the file it came from is one click away.
// It goes in the ONE notice surface, as a quiet chip, so it can never push the
// timeline down the screen — the rule the four stacked banners broke.
function sbeKeyedNotice(ovId, res) {
  const bar = sbeEl('sbeKeyed');
  if (!bar) return;
  SBE.keyed = { id: ovId, original: res.original || '', path: res.path || '' };
  const what = sbeEl('sbeKeyedWhat');
  if (what) {
    what.textContent = 'from ' + sbeNiceName(
      res.name || String(res.original || '').split('/').pop() || 'that image');
  }
  bar.hidden = false;
  sbePaintNotices();
}

function sbeKeyedKeepOriginal() {
  const k = SBE.keyed;
  sbeKeyedDismiss();
  if (!k || !k.id || !k.original) return;
  if (sbeOvMutate(os => sbeOvSetPath(os, k.id, k.original))) {
    phosToast('Kept the original — the card is back on its black background.',
              { duration: 5000 });
  }
}

function sbeKeyedDismiss() {
  SBE.keyed = null;
  const bar = sbeEl('sbeKeyed');
  if (bar) bar.hidden = true;
  if (SBE.noticeLead === 'sbeKeyed') SBE.noticeLead = '';
  sbePaintNotices();
}

function sbePaintOverlays() {
  const lane = sbeEl('sbeOverlayLane');
  if (!lane || !lane.classList) return;
  let html = '';
  for (const o of SBE.overlays || []) {
    const x = sbePx(sbeNum(o.film_start));
    const w = Math.max(2, sbePx(sbeNum(o.film_end) - sbeNum(o.film_start)));
    const e = sbeFx(o);
    const okind = sbeOvKind(o);
    const name = (okind === 'text')
      ? (sbeOvText(o).text.split('\n')[0] || 'Title')
      : sbeNiceName(o.title || String(o.path || '').split('/').pop() || 'card');
    html += '<div class="sbe-ov is-' + okind + (o.id === SBE.ovSel ? ' is-sel' : '') + '" '
          + 'data-id="' + escapeHtml(o.id) + '" '
          + 'title="' + escapeHtml((okind === 'text'
              ? 'Title — drawn over the picture. '
              : 'Overlay — composited over the picture. ')
              + 'Drag to move, pull either end to change how long.') + '" '
          + 'style="left:' + x.toFixed(1) + 'px;width:' + w.toFixed(1) + 'px">'
          + (okind === 'still' && o.path
              ? '<img class="sbe-ov-thumb" alt="" src="/image?w=240&path='
                + encodeURIComponent(o.path) + '">'
              : '')
          + (okind === 'text' ? '<span class="sbe-ov-t">T</span>' : '')
          + (w > 60 ? escapeHtml(name) : '')
          + (e.fade_in > 1e-9 ? '<div class="sbe-cl-fade in" style="left:0;width:'
              + sbePx(e.fade_in).toFixed(1) + 'px"></div>' : '')
          + (e.fade_out > 1e-9 ? '<div class="sbe-cl-fade out" style="right:0;width:'
              + sbePx(e.fade_out).toFixed(1) + 'px"></div>' : '')
          + '<div class="sbe-grip l"></div><div class="sbe-grip r"></div>'
          + '</div>';
  }
  // AN EMPTY LANE IS A SENTENCE, NOT A BLANK — the convention the track and
  // the four pool sources already follow. Without it the overlay lane is a
  // strip of nothing that teaches nobody it exists.
  lane.innerHTML = html || '<div class="sbe-track-empty sbe-ovlane-empty">'
    + 'Overlay lane — drop a still here, press \u25a3 on an image in the '
    + 'media pool, or Add title for text over the picture at the playhead.</div>';
  lane.style.width = sbeEl('sbeInner').style.width;
}

// THE STAGE'S CARD. Driven by the playhead like the strip player, because an
// overlay is its own lane and does not care which picture is underneath.
function sbeOvPaint(t) {
  const el = sbeEl('sbeOvLayer');
  if (!el || !el.classList) return;
  const tx = sbeEl('sbeOvText');
  const now = (t === undefined) ? SBE.playhead : sbeNum(t);
  const o = sbeOvAt(SBE.overlays, now);
  const txOff = () => { if (tx && tx.classList) tx.classList.remove('is-on'); };
  if (!o) { el.classList.remove('is-on'); txOff(); return; }
  if (sbeOvKind(o) === 'text') {
    // A TITLE IS DRAWN IN THE DOM, not re-rendered: the same string, the same
    // anchor and the same size rule (`font_size` at 1080 high, scaled to
    // the stage) the rasteriser uses, so what is on the stage is what lands
    // in the file. The stage's own height is the scale; the style values
    // are written as custom properties and the stylesheet does the rest.
    el.classList.remove('is-on');
    if (!tx || !tx.classList) return;
    const tt = sbeOvText(o);
    const st = tt.style;
    const stage = sbeEl('sbeStage');
    const H = (stage && stage.clientHeight) ? stage.clientHeight : SBE_TEXT_REF_H;
    if (tx.textContent !== tt.text) tx.textContent = tt.text;
    tx.style.setProperty('--tx-size', (st.font_size * H / SBE_TEXT_REF_H).toFixed(2) + 'px');
    tx.style.setProperty('--tx-color', st.color);
    tx.style.setProperty('--tx-y', (st.y * 100).toFixed(3) + '%');
    tx.style.setProperty('--tx-align', st.align);
    tx.style.setProperty('--tx-box', st.box ? sbeRgba(st.box_color, st.box_opacity) : 'transparent');
    tx.style.opacity = String(sbeFadeOpacityAt(o, now));
    tx.classList.add('is-on');
    sbeOvTextPlace(tx, st, stage);
    return;
  }
  txOff();
  if (!o.path) { el.classList.remove('is-on'); return; }
  const url = '/image?w=1280&path=' + encodeURIComponent(o.path);
  if (el.getAttribute('src') !== url) el.setAttribute('src', url);
  el.style.opacity = String(sbeFadeOpacityAt(o, now));
  el.classList.add('is-on');
}

// WHERE THE BOX SITS, IN PIXELS, KEPT INSIDE THE FRAME. The anchor is a
// fraction and the align says which edge of the text sits on it; measured
// against the stage so a wide title at x=0.2 is pushed back in rather than
// cut at the frame's edge — the same clamp `render_title` applies.
function sbeOvTextPlace(tx, st, stage) {
  const W = (stage && stage.clientWidth) ? stage.clientWidth : 0;
  const tw = (tx && tx.offsetWidth) ? tx.offsetWidth : 0;
  const shift = st.align === 'left' ? 0 : (st.align === 'right' ? 1 : 0.5);
  if (!W || !tw || !tx.style || !tx.style.setProperty) {
    if (tx && tx.style && tx.style.setProperty) {
      tx.style.setProperty('--tx-left', (st.x * 100).toFixed(3) + '%');
      tx.style.setProperty('--tx-shift', (-shift * 100) + '%');
    }
    return;
  }
  let left = st.x * W - tw * shift;
  left = Math.max(0, Math.min(W - tw, left));
  tx.style.setProperty('--tx-left', left.toFixed(1) + 'px');
  tx.style.setProperty('--tx-shift', '0%');
}

function sbeRgba(hex, alpha) {
  const h = sbeHexColour(hex, '#000000').slice(1);
  return 'rgba(' + parseInt(h.slice(0, 2), 16) + ',' + parseInt(h.slice(2, 4), 16)
       + ',' + parseInt(h.slice(4, 6), 16) + ',' + Math.max(0, Math.min(1, sbeNum(alpha))).toFixed(3) + ')';
}

// THE TITLE'S ONE DOOR. Every field on a title — its text, its size, where
// it sits — goes through here, as one undo step and one queued snapshot,
// the way every other overlay edit does.
function sbeOvTextCommit(field, value) {
  if (!SBE.ovSel) return;
  sbeBlurControl();
  const id = SBE.ovSel;
  sbeOvMutate(os => {
    const o = sbeOvById(os, id);
    if (!o || sbeOvKind(o) !== 'text') return { overlays: os, ok: false, why: 'gone' };
    const out = os.map(x => Object.assign({}, x));
    const t = sbeOvById(out, id);
    if (field === 'text') {
      const s = String(value || '').replace(/\r\n?/g, '\n').slice(0, SBE_TEXT_MAX);
      if (!s.trim()) return { overlays: os, ok: false, why: 'a title needs some text' };
      if (s === String(t.text || '')) return { overlays: os, ok: false, why: '' };
      t.text = s;
    } else {
      const st = Object.assign({}, t.style || {});
      if (field === 'box') st.box = !!value;
      else if (field === 'align') st.align = String(value);
      else if (field === 'color' || field === 'box_color') st[field] = sbeHexColour(value, SBE_TEXT_DEFAULTS[field]);
      else st[field] = sbeNum(value, SBE_TEXT_DEFAULTS[field]);
      // NEUTRAL IS ABSENT, the server's rule: a style back at the defaults
      // is no style.
      const clean = sbeOvText({ style: st }).style;
      const keep = {};
      for (const k of Object.keys(clean)) if (clean[k] !== SBE_TEXT_DEFAULTS[k]) keep[k] = clean[k];
      if (Object.keys(keep).length) t.style = keep; else delete t.style;
      if (JSON.stringify(t.style || null) === JSON.stringify(o.style || null)) {
        return { overlays: os, ok: false, why: '' };
      }
    }
    t.source = 'human';
    return { overlays: out, ok: true };
  });
  sbeOvPaint();
}

// ADD TITLE: a text overlay at the playhead, selected, with the inspector's
// text box focused so the next thing typed is the title. No modal, no
// prompt — the inspector is where every other property of an overlay lives,
// so it is where the text lives too.
function edAddTitle() {
  if (!SBE.open || !SBE.id) {
    phosToast('Open a __SEQ__ first — a title belongs to a timeline.', {});
    return;
  }
  const res = sbeOvMutate(os => sbeOvAdd(os, { kind: 'text', text: 'Title',
                                               title: 'Title', duration_s: 3 },
                                         SBE.playhead));
  if (!res || !res.added) return;
  SBE.ovSel = res.added.id;
  SBE.sel = '';
  SBE.txSel = '';
  sbePaint();
  sbeOvPaint();
  const box = sbeEl('sbeOvTextBox');
  if (box && box.focus) { try { box.focus(); box.select && box.select(); } catch (e) {} }
}

function sbeOnOvDown(ev) {
  const blk = ev.target.closest ? ev.target.closest('.sbe-ov') : null;
  if (!blk) { SBE.ovSel = ''; sbePaint(); return; }
  const id = blk.dataset.id;
  SBE.ovSel = id;
  SBE.sel = '';
  SBE.txSel = '';
  const o = sbeOvById(SBE.overlays, id);
  if (!o) return;
  const grip = ev.target.closest('.sbe-grip');
  SBE.ovDrag = { id: id, x0: ev.clientX,
                 mode: grip ? (grip.classList.contains('r') ? 'trimR' : 'trimL') : 'move',
                 fs0: sbeNum(o.film_start), fe0: sbeNum(o.film_end),
                 moved: false, lane0: JSON.stringify(SBE.overlays),
                 before: sbeSnapshot() };
  try { sbeEl('sbeOverlayLane').setPointerCapture(ev.pointerId); } catch (e) {}
  ev.preventDefault();
  sbePaint();
}

function sbeOnOvMove(ev) {
  const d = SBE.ovDrag;
  if (!d) return;
  if (Math.abs(ev.clientX - d.x0) > 3) d.moved = true;
  if (!d.moved) return;
  const dt = (ev.clientX - d.x0) / SBE.pps;
  const tol = SBE_SNAP_PX / SBE.pps;
  const marks = sbeMusicSnaps(SBE.clips);      // the CUTS: what a card aims at
  let r;
  if (d.mode === 'move') {
    r = sbeOvMove(SBE.overlays, d.id,
                  sbeSnapToList(Math.max(0, d.fs0 + dt), marks, tol,
                                sbeSnapEnabled(ev)));
  } else if (d.mode === 'trimL') {
    r = sbeOvTrim(SBE.overlays, d.id, 'l',
                  sbeSnapToList(Math.max(0, d.fs0 + dt), marks, tol,
                                sbeSnapEnabled(ev)));
  } else {
    r = sbeOvTrim(SBE.overlays, d.id, 'r',
                  sbeSnapToList(Math.max(0, d.fe0 + dt), marks, tol,
                                sbeSnapEnabled(ev)));
  }
  if (r && r.ok) SBE.overlays = r.overlays;
  sbePaint();
}

function sbeOnOvUp(ev) {
  const d = SBE.ovDrag;
  SBE.ovDrag = null;
  if (!d) return;
  if (!d.moved) { sbePaint(); return; }
  if (d.lane0 === JSON.stringify(SBE.overlays)) { sbePaint(); return; }
  SBE.undo.push(d.before);
  if (SBE.undo.length > SBE_UNDO_MAX) SBE.undo.shift();
  SBE.redo.length = 0;
  SBE.dirty = true;
  sbeSetState('unsaved changes', 'dirty');
  sbePaint();
  sbeQueueSave();
}

// The strip's own ramps, drawn to scale. The SAME triangle the picture uses,
// because it is the same gesture and the muscle memory should transfer.
// The waveform and the level line, as ONE inline SVG per strip. An <svg>
// rather than a <canvas> because the lane is repainted on every drag and a
// canvas would need its own size/DPR bookkeeping to stay crisp; the point
// count here is a few hundred, which is nothing for the DOM.
// THE LEVEL LINE'S TARGET, CUT TO THE SPAN NOBODY ELSE OWNS. Same polyline as
// the one that is drawn, clipped to [x0, x1] with the y at each cut
// interpolated along the segment it falls in — so the target sits exactly on
// the line it belongs to rather than on a straight line between whole points.
// Returns '' when there is not enough left to aim at, and the caller then
// offers no target at all: an 8px stub of a control is worse than none, and
// the inspector's Sound section is the route that never depends on width.
function sbeLvlHitPath(pts, xOf, yOf, x0, x1) {
  if (!pts || pts.length < 2 || !(x1 - x0 >= SBE_LVL_MIN_SPAN)) return '';
  let d = '';
  for (let i = 0; i < pts.length - 1; i++) {
    const ax = xOf(pts[i][0]), bx = xOf(pts[i + 1][0]);
    const ay = yOf(pts[i][1]), by = yOf(pts[i + 1][1]);
    if (bx < x0 || ax > x1) continue;
    const sx = Math.max(ax, x0), ex = Math.min(bx, x1);
    const yAt = (x) => (bx - ax <= 1e-9) ? by : ay + (by - ay) * ((x - ax) / (bx - ax));
    d += (d ? 'L' : 'M') + sx.toFixed(2) + ',' + yAt(sx).toFixed(2)
       + 'L' + ex.toFixed(2) + ',' + yAt(ex).toFixed(2);
  }
  return d;
}

function sbeStripWave(c, w, px) {
  const peaks = sbeWaveWant(c.path);
  // THE STRIP IS AS TALL AS THE LANE WAS DRAGGED TO. Everything below is
  // expressed in H, so a taller lane is not a stretched picture — it is more
  // waveform, a longer level line and a point that is genuinely easier to hit.
  const H = sbeStripH(), MID = H / 2;
  const cols = Math.max(4, Math.floor(px));
  let body = '';
  if (peaks) {
    const slice = sbeWaveSlice(peaks, w.start, w.end, cols);
    let d = '';
    for (let i = 0; i < slice.length; i++) {
      const lo = Math.max(-1, slice[i][0]), hi = Math.min(1, slice[i][1]);
      const x = (i / cols) * px;
      d += 'M' + x.toFixed(2) + ',' + (MID - hi * MID).toFixed(2)
         + 'L' + x.toFixed(2) + ',' + (MID - lo * MID).toFixed(2);
    }
    body += '<path class="sbe-wave-p" d="' + d + '"/>';
  }
  // THE LEVEL LINE, drawn ON the waveform because that is what it acts on.
  // Unity sits at the top: a gain of 1 is "all of it", and a line that fell
  // to the middle at unity would read as half.
  const curve = sbeGainPoints(c, w.len);
  const pts = curve.length ? curve : [[0, 1], [w.len, 1]];
  const yOf = g => sbeStripY(g, H);
  const xOf = t => (w.len > 0 ? (t / w.len) * px : 0);
  let line = '';
  for (let i = 0; i < pts.length; i++) {
    line += (i ? 'L' : 'M') + xOf(pts[i][0]).toFixed(2) + ','
          + yOf(pts[i][1]).toFixed(2);
  }
  // THREE PATHS, ONE LINE. A dark stroke underneath so the level reads over a
  // loud passage instead of dissolving into it; the line itself; and a fat
  // transparent one on top that is the TARGET — `pointer-events: stroke`, so
  // the line is a thing you can hit rather than a thing you can see. Without
  // it the only way to reach a level was to already know a gesture.
  body += '<path class="sbe-lvl-u" d="' + line + '"/>';
  body += '<path class="sbe-lvl" d="' + line + '"/>';
  const shape = sbeStripEditable(c);
  // THE TWO ENDS OF THE LINE BELONG TO THE FADE HANDLES. The line is DRAWN end
  // to end, because that is the truth about the gain — but its TARGET stops
  // where the corner handles start, so the third affordance in this corner
  // cannot be ambiguous with the other two either. Without this the fat
  // transparent stroke ran the full width, under both 22px handles: the
  // handles win on z-index so nothing was actually stolen, but two controls
  // claiming the same rectangle is how the fade handle got lost inside the
  // trim grip in the first place, and "it happens to stack right" is not a
  // property anybody can see or keep.
  if (shape) {
    // IN USER UNITS, WHICH ARE NOT CSS PIXELS HERE. The strip is border-box
    // and 1px of border on each side, so its <svg> is laid out over `px - 2`
    // CSS pixels while its viewBox is `px` units wide — every user unit is a
    // hair narrower than a pixel, and the clip landed INSIDE the handle by
    // that fraction. Small on a wide strip and not small on a narrow one
    // (the error is 2/px of the span), so it is converted rather than padded.
    const scale = px / Math.max(1, px - 2);
    const clear = SBE_LVL_CLEAR * scale;
    const hit = sbeLvlHitPath(pts, xOf, yOf, clear, px - clear);
    if (hit) body += '<path class="sbe-lvl-hit" d="' + hit + '"/>';
  }
  // THE GHOST: where the click would land, drawn before it is clicked.
  const gh = SBE.kfGhost;
  if (shape && gh && gh.id === c.id) {
    body += '<circle class="sbe-kf-ghost" cx="' + xOf(gh.t).toFixed(2) + '" '
          + 'cy="' + yOf(gh.g).toFixed(2) + '" r="'
          + Math.max(3.6, Math.min(6.5, H / 5.5)).toFixed(2) + '"/>';
  }
  // Only the USER's points get a handle. The fade corners are already a
  // gesture of their own, and putting a dot on them would offer two ways to
  // drag the same number in opposite directions.
  const own = sbeAfx(c, w.len).points;
  // The dot grows with the strip, within reason: 3.2 was the whole of the
  // target in a 20px lane and is the floor here, and past ~6 a handle starts
  // hiding the line it is on.
  const R = Math.max(3.2, Math.min(6, H / 6));
  for (let i = 0; i < own.length; i++) {
    body += '<circle class="sbe-kf" data-kf="' + i + '" '
          + 'cx="' + xOf(own[i][0]).toFixed(2) + '" '
          + 'cy="' + yOf(sbeLerpGain(curve.length ? curve : own, own[i][0])).toFixed(2) + '" '
          + 'style="--kf-r:' + R.toFixed(2) + 'px" '
          + 'r="' + R.toFixed(2) + '"/>';
  }
  return '<svg class="sbe-wave-svg" width="' + Math.max(1, px).toFixed(0)
       + '" height="' + H + '" viewBox="0 0 ' + Math.max(1, px).toFixed(0)
       + ' ' + H + '" preserveAspectRatio="none">' + body + '</svg>';
}

// ONE RULE, ASKED ONCE. A level belongs to a strip you can shape: not locked,
// and unlinked — the same rule the double-click has always enforced, now named
// so the line, the ghost, the click and the inspector cannot disagree about it.
function sbeStripEditable(c) {
  if (!c || c.locked) return false;
  if (c.has_audio === false) return false;
  return !sbeClipAudio(c).linked;
}

// Where the pointer is, in the coordinates a level cares about: which strip,
// how far along it, and what gain that height means. Null when the pointer is
// not over a strip that could take a point.
function sbeStripAt(ev) {
  const blk = ev.target && ev.target.closest ? ev.target.closest('.sbe-aclip') : null;
  if (!blk) return null;
  // NOT WHILE THE POINTER IS ON A CORNER HANDLE. The clipped hit path already
  // keeps the LINE off those pixels, but this pair also answers a proximity
  // test that knows nothing about paths — and it is what the hover ghost asks,
  // so without this a fade handle would sit under a ghost promising a
  // keyframe it could never place. Asked of the real hit target rather than of
  // a rectangle, so it stays exact at every lane height and every zoom.
  if (ev.target.closest('.sbe-fade-h')) return null;
  const c = sbeById(SBE.clips, blk.dataset.id);
  if (!c || !sbeStripEditable(c)) return null;
  const r = blk.getBoundingClientRect();
  const w = sbeClipAudio(c);
  return {
    c: c, id: c.id, rect: r, len: w.len,
    t: w.len * Math.max(0, Math.min(1, (ev.clientX - r.left) / Math.max(1, r.width))),
    g: sbeStripGain(ev.clientY, r.top, r.height),
  };
}

// HOVER TEACHES. Near the line, a ghost point follows the pointer and the
// cursor turns into the one you use to change a level — so the answer to "how
// do I add a keyframe" is visible before anything is clicked. Away from it the
// strip is what it always was: a thing you drag.
function sbeAudioGhost(ev) {
  const at = sbeStripAt(ev);
  let want = null;
  if (at) {
    const H = at.rect.height;
    const onLine = sbeStripY(sbeGainAt(at.c, at.len, at.t), H);
    if (Math.abs((ev.clientY - at.rect.top) - onLine) <= SBE_LVL_GRAB) {
      want = { id: at.id, t: sbeRound(at.t), g: sbeRound(at.g) };
    }
  }
  const now = SBE.kfGhost;
  const same = (!want && !now) || (want && now && want.id === now.id
    && Math.abs(want.t - now.t) < 1e-4 && Math.abs(want.g - now.g) < 2e-3);
  if (same) return;
  SBE.kfGhost = want;
  // ONE LANE, NOT THE WHOLE TIMELINE. A ghost that repainted every clip block
  // and both canvases on every mouse move would make the pointer feel heavy
  // over the one lane this is meant to make inviting.
  sbePaintAudioLane();
}

function sbeAudioGhostClear() {
  if (!SBE.kfGhost) return;
  SBE.kfGhost = null;
  sbePaintAudioLane();
}

// A SINGLE CLICK ON THE LINE PLACES A POINT — and the same gesture goes on to
// drag it, so "click, then set the level" is one movement. Double-click still
// works, because muscle memory should not be taken away to teach somebody
// else. Returns true when it claimed the gesture.
function sbeLevelClick(ev) {
  const at = sbeStripAt(ev);
  if (!at) return false;
  const H = at.rect.height;
  const onLine = sbeStripY(sbeGainAt(at.c, at.len, at.t), H);
  const near = Math.abs((ev.clientY - at.rect.top) - onLine) <= SBE_LVL_GRAB;
  if (!near && !ev.target.closest('.sbe-lvl-hit')) return false;
  const before = sbeSnapshot();
  const r = sbeAddKeyframe(SBE.clips, at.id, at.t, at.g);
  if (!r.ok) return false;
  SBE.clips = r.clips;
  SBE.sel = at.id;
  SBE.kfGhost = null;
  // The point it just made is the point the drag continues on, so a level is
  // set in one gesture rather than two.
  const pts = sbeAfx(sbeById(SBE.clips, at.id), at.len).points;
  let idx = 0;
  for (let i = 0; i < pts.length; i++) {
    if (Math.abs(pts[i][0] - sbeRound(at.t)) < 1e-3) { idx = i; break; }
  }
  SBE.kfDrag = { id: at.id, index: idx, before: before, moved: true,
                 rect: at.rect };
  try { sbeEl('sbeAudioLane').setPointerCapture(ev.pointerId); } catch (e) {}
  SBE.dirty = true;
  sbeSetState('unsaved changes', 'dirty');
  sbePaint();
  return true;
}

// RIGHT-CLICK REMOVES. Shift-click still does too, but a modifier on a 6px
// target is not a route anybody finds — and a context menu is where every
// other application on this machine keeps "remove".
function sbeOnAudioMenu(ev) {
  const kf = ev.target && ev.target.closest ? ev.target.closest('.sbe-kf') : null;
  if (!kf) return;
  const blk = ev.target.closest('.sbe-aclip');
  const c = blk ? sbeById(SBE.clips, blk.dataset.id) : null;
  if (!c || c.locked) return;
  ev.preventDefault();
  SBE.sel = c.id;
  sbeMutate(cs => sbeDeleteKeyframe(cs, c.id, sbeNum(kf.dataset.kf)));
}

// THE ROUTE THAT NEEDS NO GESTURE AT ALL. The inspector's Sound section can
// put a point exactly where the playhead is, which is also the only way to
// place one at a frame you have actually listened to.
function sbeAddPointAtPlayhead() {
  const c = sbeById(SBE.clips, SBE.sel);
  if (!c) return;
  if (!sbeStripEditable(c)) {
    phosToast('Unlink this clip\'s sound first — a level line belongs to a '
              + 'strip you can shape.', { duration: 6000 });
    return;
  }
  const w = sbeClipAudio(c);
  const t = sbeNum(SBE.playhead) - sbeNum(w.film_start);
  if (t < -1e-6 || t > w.len + 1e-6) {
    phosToast('Move the playhead over this clip\'s sound first — a point is '
              + 'placed where you are listening.', { duration: 6000 });
    return;
  }
  const at = Math.max(0, Math.min(w.len, t));
  const r = sbeAddKeyframe(SBE.clips, c.id, at, sbeGainAt(c, w.len, at));
  if (!r.ok) {
    phosToast(r.why === 'there is already a point here'
      ? 'There is already a point at the playhead — drag it, or right-click it '
        + 'to remove it.' : 'That clip is locked.', { duration: 6000 });
    return;
  }
  sbeMutate(() => ({ clips: r.clips, ok: true }));
}

function sbeClearPoints() {
  const c = sbeById(SBE.clips, SBE.sel);
  if (!c) return;
  sbeMutate(cs => sbeAfxWrite(cs, c.id, []));
}

// THE LEGEND, in one function, because it is the only thing that tells a new
// user what the gestures are and it has to be true. The Keys chip renders it.
function sbeKeysLegend() {
  const k = t => '<span class="sbe-kbd">' + t + '</span>';
  return [
    ['Drag', 'move a clip · handles to trim'],
    [k('Space'), 'play / pause'],
    [k('←') + k('→'), 'one frame'],
    [k('S'), 'split at the playhead'],
    [k('⌫'), 'ripple delete'],
    [k('Alt'), 'held: ignore the beat grid'],
    [k('Shift'), 'held: reorder instead of move'],
    ['Levels', 'click the yellow line to add a point · drag it to set the '
             + 'level · right-click it to remove · ' + k('Shift')
             + '-click removes too'],
    ['Sound', 'the music strip drags and trims like a clip'],
    ['Timeline', 'drag its top edge for more height · double-click to reset'],
  ].map(r => '<span class="sbe-key-row"><b>' + r[0] + '</b>' + r[1] + '</span>')
   .join('');
}

function sbeAudioFadeMarks(c, w) {
  const e = sbeAfx(c, w.len);
  let out = '';
  if (e.fade_in > 1e-9) {
    out += '<div class="sbe-cl-fade in" style="left:0;width:'
        + sbePx(e.fade_in).toFixed(1) + 'px"></div>';
  }
  if (e.fade_out > 1e-9) {
    out += '<div class="sbe-cl-fade out" style="right:0;width:'
        + sbePx(e.fade_out).toFixed(1) + 'px"></div>';
  }
  return out;
}

// DOUBLE-CLICK ADDS A POINT, on the strip body and nowhere else. Plain drag
// stays "move the strip" — the gesture every lane on this timeline already
// uses — so the control case is opt-in and the simple case is untouched.
function sbeOnAudioDbl(ev) {
  const blk = ev.target.closest ? ev.target.closest('.sbe-aclip') : null;
  if (!blk) return;
  if (ev.target.closest('.sbe-kf') || ev.target.closest('.sbe-fade-h')) return;
  const c = sbeById(SBE.clips, blk.dataset.id);
  if (!c) return;
  const w = sbeClipAudio(c);
  if (w.linked) {
    phosToast('Unlink this clip\'s sound first — a level line belongs to a '
              + 'strip you can shape.', { duration: 6000 });
    return;
  }
  const r = blk.getBoundingClientRect();
  const t = w.len * Math.max(0, Math.min(1, (ev.clientX - r.left) / Math.max(1, r.width)));
  // THE STRIP'S OWN RECTANGLE, through the one pair every gesture shares. The
  // `- 3` and `/ 20` this replaces were a lane that no longer has that inset
  // and a height the top edge can now drag to four times over: on a tall strip
  // every double-click past the first fourteen pixels landed a point at
  // silence, whatever the pointer was actually pointing at.
  const g = sbeStripGain(ev.clientY, r.top, r.height);
  SBE.sel = c.id;
  sbeMutate(cs => sbeAddKeyframe(cs, c.id, t, g));
  ev.preventDefault();
}

function sbeAudioFadeCommit(edge, v) {
  if (!SBE.sel) return;
  sbeMutate(cs => sbeSetAudioFade(cs, SBE.sel, edge, v));
}

function sbeOnAudioDown(ev) {
  // The sync flag is a BUTTON inside the strip, and pointerdown would start a
  // drag before its click ever fired.
  const badge = ev.target.closest('.sbe-sync');
  if (badge) { ev.preventDefault(); sbeResyncSel(badge.dataset.sync); return; }
  // THE SAME PRECEDENCE THE PICTURE LANE SET: the corner handle sits over the
  // left grip's hit area, and a fade drag is the more specific gesture of the
  // two, so it is tested first. One rule for both lanes or the strip would
  // behave differently from the block above it for no reason a user could
  // name.
  // GESTURE PRECEDENCE ON THIS LANE, most specific first, and it is the same
  // shape the picture lane already uses:
  //   1. the sync flag      — a button
  //   2. a keyframe dot     — 6px of target inside the strip body
  //   3. the corner handle  — sits over the left grip's hit area
  //   4. the grips          — trim
  //   5. the strip body     — move  (and DOUBLE-CLICK there adds a point)
  // The dot is smaller than the body and is tested before it, so grabbing a
  // point can never be mistaken for moving the strip.
  const kf = ev.target.closest('.sbe-kf');
  if (kf) {
    const blk1 = ev.target.closest('.sbe-aclip');
    const c1 = blk1 ? sbeById(SBE.clips, blk1.dataset.id) : null;
    if (c1 && !c1.locked) {
      SBE.sel = c1.id;
      // SHIFT-CLICK DELETES. A modifier rather than a second affordance
      // drawn on a 6px target, which would be a delete nobody meant.
      if (ev.shiftKey) {
        sbeMutate(cs => sbeDeleteKeyframe(cs, c1.id, sbeNum(kf.dataset.kf)));
        ev.preventDefault();
        return;
      }
      SBE.kfDrag = { id: c1.id, index: sbeNum(kf.dataset.kf),
                     before: sbeSnapshot(), moved: false,
                     rect: blk1.getBoundingClientRect() };
      try { sbeEl('sbeAudioLane').setPointerCapture(ev.pointerId); } catch (e) {}
      ev.preventDefault();
      sbePaint();
    }
    return;
  }
  const afh = ev.target.closest('.sbe-fade-h');
  if (afh) {
    const blk0 = ev.target.closest('.sbe-aclip');
    const c0 = blk0 ? sbeById(SBE.clips, blk0.dataset.id) : null;
    if (c0) {
      SBE.sel = c0.id;
      const w0 = sbeClipAudio(c0);
      const e0 = sbeAfx(c0, w0.len);
      SBE.audioDrag = {
        id: c0.id, mode: 'afade', edge: afh.dataset.afade, x0: ev.clientX,
        f0: (afh.dataset.afade === 'out') ? e0.fade_out : e0.fade_in,
        before: sbeSnapshot(), moved: false,
      };
      try { sbeEl('sbeAudioLane').setPointerCapture(ev.pointerId); } catch (e) {}
      ev.preventDefault();
      sbePaint();
    }
    return;
  }
  // THE LINE ITSELF, after the two handles at the strip's corners and before
  // the strip body: a click within SBE_LVL_GRAB of the level places a point
  // and starts dragging it, which is the gesture the owner could not find. The
  // grips are tested first — they are 7px of edge and the more specific target
  // — so trimming a strip can never be mistaken for shaping it.
  if (!ev.target.closest('.sbe-grip') && sbeLevelClick(ev)) {
    ev.preventDefault();
    return;
  }
  const blk = ev.target.closest('.sbe-aclip');
  if (!blk) return;
  const id = blk.dataset.id;
  SBE.sel = id;
  SBE.drag = null;          // a pointerup lost off the edge must not wedge a lane
  const c = sbeById(SBE.clips, id);
  if (!c) return;
  const w = sbeClipAudio(c);
  if (w.linked) {
    // Not an error — an instruction. The toggle is one click away and the
    // inspector is already showing this clip.
    sbePaint();
    phosToast(w.coupled
      ? 'This sound is linked to its picture at ' + sbeDriftLabel(sbeAudioDrift(c))
        + ' and moves with it. Unlink it in the inspector to slide it on its own.'
      : 'That clip\'s sound is linked to its picture. Unlink it in the '
        + 'inspector to slide it under the neighbour.', { duration: 6000 });
    return;
  }
  if (c.locked) {
    sbePaint();
    phosToast('That shot is locked, so its sound is locked with it. Click '
              + 'Unlock in the inspector.', { duration: 6000 });
    return;
  }
  const lane = sbeEl('sbeAudioLane');
  const grip = ev.target.closest('.sbe-grip');
  SBE.audioDrag = {
    id: id, mode: grip ? (grip.classList.contains('r') ? 'trimR' : 'trimL') : 'move',
    x0: ev.clientX, fs0: w.film_start, fe0: w.film_start + w.len,
    before: sbeSnapshot(), moved: false,
  };
  blk.classList.add('is-drag');
  try { lane.setPointerCapture(ev.pointerId); } catch (e) {}
  ev.preventDefault();
  sbePaint();
}

function sbeOnAudioMove(ev) {
  const k = SBE.kfDrag;
  // NOTHING IS BEING DRAGGED: this is a hover, and a hover is where the level
  // line teaches. Cheap and early — one rect read, and a repaint only when the
  // ghost actually moves.
  if (!k && !SBE.audioDrag) { sbeAudioGhost(ev); return; }
  if (k) {
    const c = sbeById(SBE.clips, k.id);
    if (!c) return;
    const w = sbeClipAudio(c);
    const r = k.rect;
    const t = w.len * Math.max(0, Math.min(1,
      (ev.clientX - r.left) / Math.max(1, r.width)));
    // THE BAND IS THE STRIP, measured. It used to be `- 3` and `/ 20`: a 3px
    // inset the SVG no longer has and the height of a lane that can now be
    // dragged to eight times that. `r` is the strip's own rectangle, so the
    // maths follows the lane wherever the handle put it. UNITY IS THE TOP, so
    // the gain falls as the pointer descends — which is what the line the user
    // is looking at does.
    const g = sbeStripGain(ev.clientY, r.top, r.height);
    k.moved = true;
    const res = sbeMoveKeyframe(SBE.clips, k.id, k.index, t, g);
    if (res.ok) SBE.clips = res.clips;
    sbePaint();
    return;
  }
  const d = SBE.audioDrag;
  if (!d) return;
  if (Math.abs(ev.clientX - d.x0) > 3) d.moved = true;
  if (!d.moved) return;
  if (d.mode === 'afade') {
    const adt = (d.edge === 'out') ? -(ev.clientX - d.x0) / SBE.pps
                                   : (ev.clientX - d.x0) / SBE.pps;
    const r2 = sbeSetAudioFade(SBE.clips, d.id, d.edge, Math.max(0, d.f0 + adt));
    if (r2.ok) SBE.clips = r2.clips;
    sbePaint();
    return;
  }
  const dt = (ev.clientX - d.x0) / SBE.pps;
  const tol = SBE_SNAP_PX / SBE.pps;
  const marks = sbeMusicSnaps(SBE.clips);      // the CUTS: what a J-cut aims at
  const anchor = (d.mode === 'trimR') ? d.fe0 : d.fs0;
  const want = sbeSnapToList(Math.max(0, anchor + dt), marks, tol,
                             sbeSnapEnabled(ev));
  const r = sbeAudioEdit(SBE.clips, d.id, d.mode, want);
  if (r.ok) SBE.clips = r.clips;
  sbePaint();
}

function sbeOnAudioUp(ev) {
  const k = SBE.kfDrag;
  SBE.kfDrag = null;
  if (k) {
    if (k.moved) {
      SBE.undo.push(k.before);
      if (SBE.undo.length > SBE_UNDO_MAX) SBE.undo.shift();
      SBE.redo.length = 0;
      SBE.dirty = true;
      sbeSetState('unsaved changes', 'dirty');
      sbeQueueSave();
    }
    sbePaint();
    return;
  }
  const d = SBE.audioDrag;
  SBE.audioDrag = null;
  if (!d) return;
  document.querySelectorAll('.sbe-aclip.is-drag').forEach(el => el.classList.remove('is-drag'));
  if (!d.moved) { sbePaint(); return; }
  SBE.undo.push(d.before);
  if (SBE.undo.length > SBE_UNDO_MAX) SBE.undo.shift();
  SBE.redo.length = 0;
  SBE.dirty = true;
  sbeSetState('unsaved changes', 'dirty');
  sbePaint();
  sbeQueueSave();
}

function sbeToggleAudioLink() {
  const c = sbeById(SBE.clips, SBE.sel);
  if (!c) return;
  const w = sbeClipAudio(c);
  const drift = sbeAudioDrift(c);
  const ok = sbeMutate(cs => sbeSetAudioLink(cs, c.id, !w.linked));
  if (!ok) return;
  phosToast(w.linked
    ? 'Sound unlinked — drag the strip under this clip to slide it, or pull '
      + 'either end. The picture stays where it is, and it stays draggable '
      + 'from both edges.'
    : (Math.abs(drift) > SBE_SYNC_TOL
        ? 'Linked at ' + sbeDriftLabel(drift) + ' — the sound keeps that '
          + 'offset and now travels with the picture. Resync puts it back '
          + 'under its own frame.'
        : 'Sound re-linked to the picture.'), { duration: 7000 });
}

function sbeDeleteStripSel() {
  const c = sbeById(SBE.clips, SBE.sel);
  if (!c) return;
  const ok = sbeMutate(cs => sbeDeleteStrip(cs, c.id));
  if (!ok) return;
  phosToast('Sound removed — the picture plays silent and stays exactly where '
            + 'it is. Unmute in Sound to bring it back.', { duration: 7000 });
}

function sbeToggleClipMute() {
  const c = sbeById(SBE.clips, SBE.sel);
  if (!c) return;
  const on = !sbeClipMuted(c);
  const ok = sbeMutate(cs => sbeSetClipMute(cs, c.id, on));
  if (!ok) return;
  phosToast(on
    ? 'Clip sound muted — it is off in the preview, the render and the export. '
      + 'The strip stays where it is, and unmuting puts it back.'
    : 'Clip sound unmuted.', { duration: 6000 });
}

// THE REMATCH, from the flag on either half or from the inspector.
function sbeResyncSel(id) {
  const who = id || SBE.sel;
  const c = sbeById(SBE.clips, who);
  if (!c) return;
  SBE.sel = who;
  const was = sbeAudioDrift(c);
  const ok = sbeMutate(cs => sbeResyncAudio(cs, who));
  if (!ok) { sbePaint(); return; }
  phosToast('Sound rematched to its picture — it was ' + sbeDriftLabel(was)
            + ' out. It is still unlinked, so it can be moved again.',
            { duration: 6000 });
}

function sbePaintHead() {
  const el = sbeEl('sbeHead');
  if (el) el.style.left = sbePx(SBE.playhead).toFixed(1) + 'px';
  sbeFollow();
  const t = sbeEl('sbeTime');
  if (t) {
    t.innerHTML = escapeHtml(sbeFmtTime(SBE.playhead)) +
      ' <span>/ ' + escapeHtml(sbeFmtTime(sbeFilmDuration(SBE.clips))) + '</span>';
  }
}

// ---------------------------------------------------------------------------
// THE POPOVER, once, for four menus
// ---------------------------------------------------------------------------
// Every menu on this screen holds the REAL controls — moved, not rebuilt — so
// an id the client already writes to keeps working wherever it now lives. The
// panel is `position: fixed` and measured against the anchor, which is the
// whole reason it can exist at all: an open menu must not push the workspace
// around, the same rule the notice surface has always followed.
function sbePopToggle(id, anchorId) {
  const el = sbeEl(id);
  if (!el) return;
  const open = el.hidden;
  sbePopCloseAll(open ? id : '');
  if (!open) return;
  el.hidden = false;
  const a = sbeEl(anchorId);
  const r = a ? a.getBoundingClientRect() : null;
  const box = el.getBoundingClientRect();
  if (r) {
    // Right-aligned to the anchor, flipped up when there is no room below,
    // and clamped so a menu can never be opened off the edge of the window.
    let left = Math.min(window.innerWidth - box.width - 8,
                        Math.max(8, r.right - box.width));
    let top = r.bottom + 6;
    if (top + box.height > window.innerHeight - 8) {
      top = Math.max(8, r.top - box.height - 6);
    }
    el.style.left = Math.round(left) + 'px';
    el.style.top = Math.round(top) + 'px';
  }
  el.dataset.anchor = anchorId || '';
}

function sbePopCloseAll(except) {
  for (const id of ['sbeRenderMenu', 'sbeMoreMenu', 'sbeKeysPop', 'sbeMusicMenu']) {
    if (id === except) continue;
    const el = sbeEl(id);
    if (el && !el.hidden) el.hidden = true;
  }
}

// A menu closes when you click away from it, and when you press Escape —
// except that Escape already closes the DOCUMENT here, so an open menu eats
// the first press and the second one leaves, which is the order every other
// application uses.
function sbePopGlobal(ev) {
  if (ev.target && ev.target.closest
      && (ev.target.closest('.sbe-pop') || ev.target.closest('[aria-haspopup]'))) return;
  sbePopCloseAll('');
}

function sbePopAnyOpen() {
  for (const id of ['sbeRenderMenu', 'sbeMoreMenu', 'sbeKeysPop', 'sbeMusicMenu']) {
    const el = sbeEl(id);
    if (el && !el.hidden) return true;
  }
  return false;
}

// The two lane heads that carry a fact rather than a label: how much picture
// there is, and how much of the sound has been pulled off it.
function sbePaintHeads() {
  const pic = sbeEl('sbeHeadPic');
  if (pic) {
    const n = SBE.clips.length;
    pic.textContent = n
      ? n + ' clip' + (n === 1 ? '' : 's') + ' · ' + sbeFmtTime(sbeFilmDuration(SBE.clips))
      : 'empty';
  }
  const aud = sbeEl('sbeHeadAud');
  if (aud) {
    let split = 0, muted = 0;
    for (const c of SBE.clips) {
      if (sbeKind(c) !== 'video') continue;
      if (sbeClipAudio(c).split) split++;
      if (sbeClipMuted(c)) muted++;
    }
    const bits = [];
    if (split) bits.push(split + ' unlinked');
    if (muted) bits.push(muted + ' muted');
    aud.textContent = bits.length ? bits.join(' · ') : 'linked to the picture';
    aud.classList.toggle('is-live', split > 0);
  }
}

// The legend is painted rather than written into the markup so there is one
// copy of it — see sbeKeysLegend.
function sbePaintKeys() {
  const el = sbeEl('sbeKeys');
  if (el && !el._done) { el.innerHTML = sbeKeysLegend(); el._done = 1; }
}

function sbePaintInspector() {
  const box = sbeEl('sbeInspect');
  // A SELECTED CUT OWNS THE INSPECTOR. The boundary between two clips is a
  // subject in its own right — it is what a transition is a property of —
  // so it gets the same box every other subject gets, and the same shape:
  // what it is, then its controls.
  if (SBE.txSel && !SBE.sel && !SBE.ovSel) {
    const order = SBE.clips.slice().sort((p, q) => sbeNum(p.film_start) - sbeNum(q.film_start));
    const k = order.findIndex(c => c.id === SBE.txSel);
    const a = order[k], b = order[k + 1];
    if (!a || !b) { SBE.txSel = ''; }
    else {
      const row = sbeTxAfter(SBE.transitions, a.id);
      const r = sbeTxResolve(order, SBE.transitions || [], sbeFps()).find(x => x.after_clip === a.id) || null;
      const name = (c) => (sbeKind(c) === 'slug') ? 'black'
        : sbeNiceName(c.title || String(c.path || '').split('/').pop() || 'clip');
      const kind = row ? String(row.kind) : 'none';
      const dur = row ? sbeNum(row.duration, 0.5) : 0.5;
      const opt = (v, label) => '<option value="' + v + '"' + (kind === v ? ' selected' : '') + '>' + label + '</option>';
      const cap = Math.max(SBE_TX_MIN, Math.min(SBE_TX_MAX, 0.5 * Math.min(sbeLen(a), sbeLen(b))));
      const why = !row
        ? 'A transition here pulls ' + 'extra picture from beyond each clip\'s trim — '
          + 'the cut itself does not move, and neither does the sound.'
        : (r && r.problem ? r.problem.message
           : (r ? SBE_TX_LABELS[r.kind] + ' over ' + r.duration.toFixed(2) + 's, centred on the cut: '
                  + r.half.toFixed(2) + 's of handle each side. '
                  + (r.kind === 'dissolve'
                     ? 'The preview only approximates it (one picture at a time); the render is a true dissolve.'
                     : 'The preview shows it; the render is exact.') : ''));
      box.innerHTML =
        '<b>Cut · ' + escapeHtml(name(a)) + ' \u2192 ' + escapeHtml(name(b)) + '</b>' +
        '<span>at ' + sbeNum(a.film_end).toFixed(2) + 's on the __SEQ__</span>' +
        '<div class="sbe-sect"><div class="sbe-sect-h">Transition</div>' +
        '<div class="sbe-sect-b">' +
        '<span class="sbe-fade-row"><label for="sbeTxKind">Kind</label>' +
        '<select class="sb-input sbe-tx-kind" id="sbeTxKind" ' +
          'onchange="sbeTxCommit(this.value, sbeEl(\'sbeTxDur\').value)">' +
        opt('none', 'None — a hard cut') + opt('dissolve', 'Dissolve') +
        opt('fade_black', 'Fade through black') + '</select></span>' +
        '<span class="sbe-fade-row"><label for="sbeTxDur">Length</label>' +
        '<input type="number" class="sb-input sbe-fade-num" id="sbeTxDur" min="' +
          SBE_TX_MIN.toFixed(2) + '" max="' + cap.toFixed(2) + '" step="0.05" value="' + dur.toFixed(2) + '" ' +
          'onchange="sbeTxCommit(sbeEl(\'sbeTxKind\').value, this.value)" ' +
          'title="Seconds, centred on the cut. Snapped to an even number of frames.">' +
        '<span class="sbe-adj-val">s</span></span>' +
        (row ? '<button type="button" class="ghost-btn" onclick="sbeTxRemoveSel()">Remove</button>' : '') +
        '</div></div>' +
        '<span class="sbe-why' + (r && r.problem ? ' is-bad' : '') + '">' + escapeHtml(why) + '</span>';
      return;
    }
  }
  // A SELECTED CARD OWNS THE INSPECTOR. It is a lane of its own, so it gets
  // the same two sections that mean anything for it — what it is, and its
  // effects — and none of the Sound section, which it has no half of.
  const ov = sbeOvById(SBE.overlays, SBE.ovSel);
  if (ov && !SBE.sel) {
    const oe = sbeFx(ov);
    const okind = sbeOvKind(ov);
    const orow = (edge, val) =>
      '<span class="sbe-fade-row">' +
      '<label for="sbeOvFade' + edge + '">Fade ' + edge + '</label>' +
      '<input type="number" class="sb-input sbe-fade-num" id="sbeOvFade' + edge + '" ' +
        'min="0" step="0.05" value="' + val.toFixed(2) + '" ' +
        'onchange="sbeOvFadeCommit(\'' + edge + '\', this.value)">' +
      '<span class="sbe-adj-val">s</span></span>';
    // A TITLE'S OWN SECTION. The text and how it sits, beside the same
    // Effects every overlay has — a title is a card the render draws, and
    // the inspector says so by giving it the card's box plus one section.
    let textSect = '';
    if (okind === 'text') {
      const tt = sbeOvText(ov);
      const st = tt.style;
      const alignOpt = (v, l) => '<option value="' + v + '"' + (st.align === v ? ' selected' : '') + '>' + l + '</option>';
      textSect =
        '<div class="sbe-sect"><div class="sbe-sect-h">Text</div>' +
        '<div class="sbe-sect-b sbe-title-b">' +
        '<textarea class="sb-input sbe-title-text" id="sbeOvTextBox" rows="2" maxlength="' + SBE_TEXT_MAX + '" ' +
          'placeholder="What the title says" onchange="sbeOvTextCommit(\'text\', this.value)">' +
          escapeHtml(tt.text) + '</textarea>' +
        '<span class="sbe-fade-row"><label for="sbeOvTxSize">Size</label>' +
        '<input type="number" class="sb-input sbe-fade-num" id="sbeOvTxSize" min="8" max="400" step="4" ' +
          'value="' + st.font_size.toFixed(0) + '" onchange="sbeOvTextCommit(\'font_size\', this.value)" ' +
          'title="Pixels on a 1080-high frame; scales with the film."></span>' +
        '<span class="sbe-fade-row"><label for="sbeOvTxColor">Colour</label>' +
        '<input type="color" class="sbe-swatch" id="sbeOvTxColor" value="' + st.color + '" ' +
          'onchange="sbeOvTextCommit(\'color\', this.value)"></span>' +
        '<span class="sbe-fade-row"><label for="sbeOvTxAlign">Align</label>' +
        '<select class="sb-input sbe-tx-kind" id="sbeOvTxAlign" onchange="sbeOvTextCommit(\'align\', this.value)">' +
        alignOpt('left', 'Left') + alignOpt('center', 'Centre') + alignOpt('right', 'Right') + '</select></span>' +
        '<span class="sbe-fade-row sbe-title-pos"><label for="sbeOvTxX">Across</label>' +
        '<input type="range" id="sbeOvTxX" min="0" max="1" step="0.01" value="' + st.x + '" ' +
          'oninput="sbeOvTextPreview(\'x\', this.value)" onchange="sbeOvTextCommit(\'x\', this.value)"></span>' +
        '<span class="sbe-fade-row sbe-title-pos"><label for="sbeOvTxY">Down</label>' +
        '<input type="range" id="sbeOvTxY" min="0" max="1" step="0.01" value="' + st.y + '" ' +
          'oninput="sbeOvTextPreview(\'y\', this.value)" onchange="sbeOvTextCommit(\'y\', this.value)"></span>' +
        '<label class="check sbe-title-box"><input type="checkbox" id="sbeOvTxBox"' + (st.box ? ' checked' : '') +
          ' onchange="sbeOvTextCommit(\'box\', this.checked)"> Box behind</label>' +
        (st.box
          ? '<span class="sbe-fade-row"><label for="sbeOvTxBoxColor">Box colour</label>' +
            '<input type="color" class="sbe-swatch" id="sbeOvTxBoxColor" value="' + st.box_color + '" ' +
              'onchange="sbeOvTextCommit(\'box_color\', this.value)">' +
            '<input type="range" id="sbeOvTxBoxOp" min="0" max="1" step="0.05" value="' + st.box_opacity + '" ' +
              'title="Box opacity" onchange="sbeOvTextCommit(\'box_opacity\', this.value)"></span>'
          : '') +
        '</div></div>';
    }
    box.innerHTML =
      '<b>' + escapeHtml(okind === 'text'
          ? (sbeOvText(ov).text.split('\n')[0] || 'Title')
          : sbeNiceName(ov.title || String(ov.path || '').split('/').pop() || 'overlay')) + '</b>' +
      '<span>overlay · ' + escapeHtml(okind === 'text' ? 'title' : okind) + '</span>' +
      '<span>' + sbeNum(ov.film_start).toFixed(2) + '–' +
      sbeNum(ov.film_end).toFixed(2) + 's on the __SEQ__</span>' +
      '<span class="sbe-why">' + (okind === 'text'
        ? 'Drawn over the picture by the render, in the same font the stage shows. '
          + 'Fades and position are kept in the preview and the film.'
        : 'Composited over the picture. Its transparency is '
          + 'kept in the preview, the render and the export.') + '</span>' +
      textSect +
      '<div class="sbe-sect"><div class="sbe-sect-h">Effects</div>' +
      '<div class="sbe-sect-b">' + orow('in', oe.fade_in)
      + orow('out', oe.fade_out) + '</div></div>' +
      '<div class="sbe-sect"><div class="sbe-sect-h">Overlay</div>' +
      '<div class="sbe-sect-b">' +
      '<button type="button" class="ghost-btn" onclick="sbeOvDeleteSel()">' +
      (okind === 'text' ? 'Remove title' : 'Remove overlay') + '</button>' +
      '</div></div>';
    return;
  }
  const c = sbeById(SBE.clips, SBE.sel);
  if (!c) {
    const n = SBE.clips.length;
    const holes = sbeHoles(SBE.clips);
    const guess = sbeGridIsAGuess(SBE.beats);
    let bits = [n + ' clip' + (n === 1 ? '' : 's'),
                sbeFmtTime(sbeFilmDuration(SBE.clips)) + ' of the __SEQ__'];
    if (holes.length) {
      bits.push(holes.length + ' hole' + (holes.length === 1 ? '' : 's') + ' · ' +
                holes.reduce((a, g) => a + g.duration, 0).toFixed(2) + 's');
    }
    if (SBE.beats && SBE.beats.bpm) {
      bits.push(Math.round(SBE.beats.bpm) + ' bpm' +
                (guess ? ' — a guess, confidence ' + sbeNum(SBE.beats.confidence).toFixed(2) : ''));
    }
    box.innerHTML = '<span class="sbe-why">' + escapeHtml(bits.join(' · ')) +
      (n ? '. Click a clip to see why it was cut where it was.'
         : '.') + '</span>';
    return;
  }
  const a = c.analysis || {};
  const kind = sbeKind(c);
  const bright = sbeBright(c);
  const errs = (SBE.errors.byId || {})[c.id] || [];
  const why = [];
  if (a.reason) why.push(a.reason);
  if (a.snap && a.snap.kind && a.snap.kind !== 'none') {
    why.push('snapped to the ' + a.snap.kind + ' (' + Math.round(sbeNum(a.snap.shift_ms)) + ' ms)');
  }
  if (a.usable === false) why.push('the auto-editor did not think this window was usable');
  for (const note of (a.notes || [])) why.push(note);
  const label = (kind === 'slug') ? 'Black'
    : (c.title || String(c.path || '').split('/').pop() || 'clip');
  // Only a video HAS a source window; the other two are their slot and nothing
  // else, so the line that would report their in/out is the line that would
  // invent one.
  const src = (kind === 'video')
    ? ('<span>source ' + sbeNum(c.start).toFixed(2) + '–' + sbeNum(c.end).toFixed(2) + 's' +
       (c.duration ? ' of ' + sbeNum(c.duration).toFixed(2) + 's' : '') + '</span>')
    : ('<span>' + escapeHtml(kind === 'slug'
        ? 'black — no file, drawn by the render'
        : 'a still, held for its slot') +
       ' · drag a handle to change how long</span>');
  // BRIGHTNESS, AND NOT A COLOUR PAGE. One constant-per-clip value, the flat
  // Adjust panel CapCut ships, refused for a slug because a slug's colour is
  // what a slug IS.
  const adjust = (kind === 'slug') ? '' :
    '<span class="sbe-adjust">Brightness' +
    '<input type="range" id="sbeBright" min="-0.5" max="0.5" step="0.01" ' +
      'value="' + bright.toFixed(2) + '" ' +
      'oninput="sbeBrightPreview(this.value)" ' +
      'onchange="sbeBrightCommit(this.value)" ' +
      'title="Approximate on screen, exact in the render.">' +
    '<span class="sbe-adj-val" id="sbeBrightVal">' +
      (bright > 0 ? '+' : '') + bright.toFixed(2) + '</span>' +
    (Math.abs(bright) >= 1e-6
      ? '<button type="button" class="ghost-btn" onclick="sbeBrightCommit(0)">Reset</button>'
      : '') +
    '</span>';
  // THREE SECTIONS, because the inspector grew one control at a time and had
  // become a flat run of buttons with a brightness slider floating in the
  // middle. Clip / Sound / Effects — and Effects is the home the ruling asked
  // for, so the next one lands without a decision.
  const sect = (name, body) => body
    ? '<div class="sbe-sect"><div class="sbe-sect-h">' + name + '</div>' +
      '<div class="sbe-sect-b">' + body + '</div></div>'
    : '';
  const e = sbeFx(c);
  const fadeRow = (edge, val) =>
    '<span class="sbe-fade-row">' +
    '<label for="sbeFade' + edge + '">Fade ' + (edge === 'in' ? 'in' : 'out') + '</label>' +
    '<input type="number" class="sb-input sbe-fade-num" id="sbeFade' + edge + '" ' +
      'min="0" max="' + sbeClipLen(c).toFixed(2) + '" step="0.05" ' +
      'value="' + val.toFixed(2) + '" ' +
      'onchange="sbeFadeCommit(\'' + edge + '\', this.value)" ' +
      'title="Seconds. Drag the corner of the clip for the same thing by eye.">' +
    '<span class="sbe-adj-val">s</span>' +
    (val > 1e-6
      ? '<button type="button" class="ghost-btn" onclick="sbeFadeCommit(\'' + edge + '\', 0)">Clear</button>'
      : '') +
    '</span>';
  box.innerHTML =
    '<b>' + escapeHtml(label) + '</b>' + src +
    '<span>film ' + sbeNum(c.film_start).toFixed(2) + '–' + sbeNum(c.film_end).toFixed(2) + 's</span>' +
    '<span>' + escapeHtml(c.source === 'human' ? 'moved by hand' : 'placed by the auto-editor') + '</span>' +
    // THE PAIR'S OWN LINE. An unlinked strip is either where the picture would
    // have played it or it is not, and the number is the only way to tell at
    // any zoom — this is the sentence the owner had no way to read.
    ((kind === 'video' && sbeClipAudio(c).split)
      ? '<span>' + escapeHtml(sbeClipAudio(c).coupled
          ? 'sound linked at ' + sbeDriftLabel(sbeAudioDrift(c))
            + ' — it travels with this picture'
          : (sbeAudioInSync(c)
             ? 'sound unlinked, in sync with its picture'
             : 'sound unlinked, ' + sbeDriftLabel(sbeAudioDrift(c)) + ' out of sync ('
               + (sbeAudioDrift(c) > 0 ? 'late' : 'early') + ')')) + '</span>'
      : '') +
    ((kind === 'video' && !c.proxy)
      ? '<span>no proxy — scrubbing this one decodes from the top of the clip</span>' : '') +
    ((kind === 'video' && !sbeShotForClip(c))
      ? '<span>not a storyboard shot — it cannot be retaken from here, only re-cut</span>' : '') +
    '<span class="sbe-why">' + escapeHtml(why.join(' · ')) + '</span>' +
    (errs.length ? '<span style="color:var(--danger)">' + escapeHtml(errs[0].message) + '</span>' : '') +
    sect('Clip',
      // SPEED, AS A CONTROL AND NEVER A GUESS. A number with the four rates
      // people actually reach for beside it; the slot follows and the rest
      // of the film ripples. Only a video clip has a clock.
      ((kind === 'video')
        ? '<span class="sbe-fade-row sbe-speed-row"><label for="sbeSpeed">Speed</label>' +
          '<input type="number" class="sb-input sbe-fade-num" id="sbeSpeed" min="' + SBE_SPEED_MIN +
            '" max="' + SBE_SPEED_MAX + '" step="0.05" value="' + sbeSpeed(c).toFixed(2) + '" ' +
            'onchange="sbeSpeedCommit(this.value)" ' +
            'title="Plays the same seconds of the take faster or slower. The slot on the film changes; everything after it moves.">' +
          '<span class="sbe-adj-val">x</span>' +
          [0.5, 1, 2].map(v => '<button type="button" class="ghost-btn sbe-speed-pill' +
            (Math.abs(sbeSpeed(c) - v) < 1e-6 ? ' is-on' : '') + '" onclick="sbeSpeedCommit(' + v + ')" ' +
            'title="' + (v === 1 ? 'Normal speed' : (v < 1 ? 'Half speed — twice as long on the film' : 'Double speed — half as long on the film')) + '">' + v + 'x</button>').join('') +
          '</span>'
        : '') +
      '<button type="button" class="ghost-btn" onclick="sbeToggleLock()" ' +
      'title="' + (c.locked ? 'Let this shot move and trim again.' : 'Pin this shot to its place on the film; everything else flows around it.') + '">' +
      (c.locked ? 'Unlock' : 'Lock') + '</button>' +
      '<button type="button" class="ghost-btn" onclick="sbeDuplicateSel()" ' +
      'title="The same shot again, right after this one — window, speed, fades and grade included. Everything after it slides. (D)">Duplicate</button>' +
      // RETAKE: send this clip back through the renderer and get a new take
      // offered against it, in place. Only a clip that came from a shot has
      // a prompt to start from — and the inspector says so when it cannot.
      ((kind === 'video' && sbeShotForClip(c))
        ? '<button type="button" class="ghost-btn" onclick="sbeRetakeSel()" ' +
          'title="Render a new take of this shot — same character, a new seed, the prompt to edit. When it lands you choose whether it replaces this clip.">Retake</button>'
        : '') +
      '<button type="button" class="ghost-btn" onclick="sbeLiftSelected()" ' +
      'title="Take this shot out and leave its hole. Nothing else moves. (Delete)">Lift</button>' +
      '<button type="button" class="ghost-btn" onclick="sbeRippleSelected()" ' +
      'title="Take this shot out and close the gap. Everything after it slides earlier. (Shift+Delete)">Ripple delete</button>') +
    sect('Sound',
    // THE J-CUT'S ONE SWITCH. Only a video clip has sound of its own, so
    // only a video clip is offered it.
    (kind === 'video'
      ? '<button type="button" class="ghost-btn" onclick="sbeToggleAudioLink()" ' +
        'title="' + escapeHtml(sbeClipAudio(c).linked
          ? 'Let this clip\'s sound move and trim on its own — the J-cut and the L-cut. The picture stays draggable from both edges.'
          : (sbeAudioInSync(c)
             ? 'Put the sound back under its own picture'
             : 'Keep the ' + sbeDriftLabel(sbeAudioDrift(c)) + ' offset you made '
               + 'and travel together from here. Resync is the button that '
               + 'puts it back under its own frame.')) + '">' +
        (sbeClipAudio(c).linked ? 'Unlink sound'
         : (sbeAudioIsThePicture(c) ? 'Re-link sound'
            : 'Link sound' + (sbeAudioInSync(c) ? ''
               : ' at ' + sbeDriftLabel(sbeAudioDrift(c))))) + '</button>'
      : '') +
    // MUTE, and it is offered whether the sound is linked, unlinked or
    // travelling — muting silences the CLIP's sound wherever its strip is.
    // Refused only for a clip whose file has no audio track at all, because
    // there is nothing there to switch off.
    ((kind === 'video' && c.has_audio !== false)
      ? '<button type="button" class="ghost-btn" onclick="sbeToggleClipMute()" ' +
        'title="' + escapeHtml(sbeClipMuted(c)
          ? 'Let this clip\'s own sound play again'
          : 'Switch this clip\'s own sound off — in the preview, in the render '
            + 'and in the export. The soundtrack is not affected.') + '">' +
        (sbeClipMuted(c) ? 'Unmute sound' : 'Mute sound') + '</button>'
      : '') +
    // DELETE THE STRIP, not the clip. Offered only once the two halves are
    // actually separate — on a linked clip "delete the sound" and "mute" would
    // be the same button twice.
    ((kind === 'video' && sbeClipAudio(c).split)
      ? '<button type="button" class="ghost-btn" onclick="sbeDeleteStripSel()" ' +
        'title="' + escapeHtml('Remove this clip\'s sound. The picture keeps '
          + 'playing, silent, and does not move.') + '">Delete sound</button>'
      : '') +
    // RESYNC IS NOT RE-LINK, and the two sit next to each other so the
    // difference is legible: re-link puts the sound back UNDER the picture
    // permanently, resync only slides it back into place and leaves it free.
    ((kind === 'video' && sbeClipAudio(c).split && !sbeAudioInSync(c))
      ? '<button type="button" class="ghost-btn" onclick="sbeResyncSel()" ' +
        'title="' + escapeHtml('Slide the sound back to where its own picture '
          + 'plays it. The trim you gave it is kept, and it stays unlinked.') +
        '">Resync sound (' + escapeHtml(sbeDriftLabel(sbeAudioDrift(c))) + ')</button>'
      : '') +
    // THE SOUND'S OWN RAMPS LIVE HERE, not under a fourth heading. Three
    // sections is what docs/EDITOR_EFFECTS_MODEL.md describes and what the
    // rail has room for; a "Sound fades" heading of its own pushed the
    // inspector past 460px and made two adjacent headings both say "sound".
    ((kind === 'video' && c.has_audio !== false)
      ? (() => {
          const w2 = sbeClipAudio(c);
          const ae = sbeAfx(c, w2.len);
          const arow = (edge, val) =>
            '<span class="sbe-fade-row">' +
            '<label for="sbeAFade' + edge + '">Fade ' + edge + '</label>' +
            '<input type="number" class="sb-input sbe-fade-num" ' +
              'id="sbeAFade' + edge + '" min="0" max="' + w2.len.toFixed(2) + '" ' +
              'step="0.05" value="' + val.toFixed(2) + '" ' +
              'onchange="sbeAudioFadeCommit(\'' + edge + '\', this.value)" ' +
              'title="Seconds. Drag the corner of the sound strip for the same thing by eye, or double-click the strip to add a level point.">' +
            '<span class="sbe-adj-val">s</span></span>';
          // LEVELS, SPELLED OUT. The line on the strip is the fast way and it
          // now teaches itself; this is the way that needs no gesture at all,
          // and it is also the only way to place a point at a frame you have
          // actually listened to. On a linked strip it says what to do rather
          // than offering a button that would refuse.
          const shape = sbeStripEditable(c);
          const n = ae.points.length;
          const levels = '<span class="sbe-fade-row sbe-levels">'
            + '<label>Levels</label>'
            + (shape
               ? '<button type="button" class="ghost-btn" '
                 + 'onclick="sbeAddPointAtPlayhead()" '
                 + 'title="Puts a level point where the playhead is. On the '
                 + 'strip: click the yellow line to add one, drag it to set '
                 + 'the level, right-click it to remove it.">'
                 + 'Add point at playhead</button>'
                 + (n ? '<button type="button" class="ghost-btn" '
                        + 'onclick="sbeClearPoints()" '
                        + 'title="Removes every level point on this strip. The '
                        + 'fades stay.">Clear ' + n + '</button>'
                      : '<span class="sbe-adj-val">click the yellow line</span>')
               : '<span class="sbe-adj-val">unlink the sound to shape its '
                 + 'level</span>')
            + '</span>';
          return arow('in', ae.fade_in) + arow('out', ae.fade_out) + levels;
        })()
      : '') +
    '') +
    // EFFECTS. Brightness is the one legacy citizen — it stays at
    // `adjust.brightness` on disk, because a label is not worth a data
    // migration — and it is PRESENTED here, beside the fades, because this is
    // where a person looks for it.
    sect('Effects', adjust + fadeRow('in', e.fade_in) + fadeRow('out', e.fade_out)
         + ((kind !== 'slug') ? (() => {
             const fr = sbeFraming(c);
             const on = !sbeFramingIsNeutral(c);
             return '<span class="sbe-fade-row sbe-frame-row"><label for="sbeFrameZoom">Zoom</label>'
               + '<input type="range" id="sbeFrameZoom" min="1" max="' + SBE_FRAME_ZOOM_MAX + '" step="0.05" '
               + 'value="' + fr.zoom + '" oninput="sbeFramingPreview(\'zoom\', this.value)" '
               + 'onchange="sbeFramingCommit(\'zoom\', this.value)" '
               + 'title="Magnify the picture and reframe it. Approximate on the stage; the render crops the source exactly.">'
               + '<span class="sbe-adj-val" id="sbeFrameZoomVal">' + fr.zoom.toFixed(2) + 'x</span>'
               + (on ? '<button type="button" class="ghost-btn" onclick="sbeFramingReset()">Reset</button>' : '')
               + '</span>'
               + (on
                  ? '<span class="sbe-fade-row sbe-frame-row"><label for="sbeFrameX">Across</label>'
                    + '<input type="range" id="sbeFrameX" min="0" max="1" step="0.01" value="' + fr.x + '" '
                    + 'oninput="sbeFramingPreview(\'x\', this.value)" onchange="sbeFramingCommit(\'x\', this.value)"></span>'
                    + '<span class="sbe-fade-row sbe-frame-row"><label for="sbeFrameY">Down</label>'
                    + '<input type="range" id="sbeFrameY" min="0" max="1" step="0.01" value="' + fr.y + '" '
                    + 'oninput="sbeFramingPreview(\'y\', this.value)" onchange="sbeFramingCommit(\'y\', this.value)"></span>'
                  : '');
           })() : ''));
}

// The slider moves at pointer speed and the undo stack does not. `oninput`
// paints the CSS filter and the number and touches NOTHING else; `onchange`,
// which fires once when the drag ends, is the edit. Committing on every input
// event would push eighty undo steps and eighty saves for one gesture.
function sbeBrightPreview(v) {
  const b = Math.max(-SBE_BRIGHT_MAX, Math.min(SBE_BRIGHT_MAX, sbeNum(v)));
  const out = sbeEl('sbeBrightVal');
  if (out) out.textContent = (b > 0 ? '+' : '') + b.toFixed(2);
  sbeApplyPreviewFilter(b);
}

// The two position sliders move at pointer speed and the undo stack does
// not — the same split the brightness slider makes. `oninput` paints, and
// `onchange` (once, when the drag ends) is the edit.
function sbeOvTextPreview(field, v) {
  const o = sbeOvById(SBE.overlays, SBE.ovSel);
  if (!o || sbeOvKind(o) !== 'text') return;
  const tx = sbeEl('sbeOvText');
  if (!tx || !tx.style || !tx.style.setProperty) return;
  const val = Math.max(0, Math.min(1, sbeNum(v)));
  if (field === 'y') { tx.style.setProperty('--tx-y', (val * 100).toFixed(3) + '%'); return; }
  const st = Object.assign({}, sbeOvText(o).style, { x: val });
  sbeOvTextPlace(tx, st, sbeEl('sbeStage'));
}

function sbeOvFadeCommit(edge, v) {
  if (!SBE.ovSel) return;
  const id = SBE.ovSel;
  sbeOvMutate(os => {
    const o = sbeOvById(os, id);
    if (!o) return { overlays: os, ok: false, why: 'gone' };
    const len = Math.max(0, sbeNum(o.film_end) - sbeNum(o.film_start));
    const key = (edge === 'out') ? 'fade_out' : 'fade_in';
    const want = Math.max(0, Math.min(len, sbeNum(v)));
    const out = os.map(x => Object.assign({}, x));
    const t = sbeOvById(out, id);
    const fx = Object.assign({}, t.fx || {});
    if (want > 1e-9) fx[key] = sbeRound(want); else delete fx[key];
    if (fx.fade_in > 1e-9 || fx.fade_out > 1e-9) t.fx = fx; else delete t.fx;
    t.source = 'human';
    return { overlays: out, ok: true };
  });
}

function sbeFadeCommit(edge, v) {
  if (!SBE.sel) return;
  const ok = sbeMutate(cs => sbeSetFade(cs, SBE.sel, edge, v));
  if (!ok) { sbePaintInspector(); return; }
}

function sbeBrightCommit(v) {
  if (!SBE.sel) return;
  const ok = sbeMutate(cs => sbeSetBrightness(cs, SBE.sel, v));
  if (!ok) { sbePaintInspector(); return; }
  sbeQueueSave();
}

// A CONTROL THAT KEEPS FOCUS EATS THE SPACE BAR — the next press re-clicks
// it instead of toggling play. Every inspector commit hands focus back.
function sbeBlurControl() {
  try {
    const a = document.activeElement;
    if (a && a.blur && a.tagName !== 'TEXTAREA') a.blur();
  } catch (e) {}
}

function sbeSpeedCommit(v) {
  if (!SBE.sel) return;
  sbeBlurControl();
  const ok = sbeMutate(cs => sbeSetSpeed(cs, SBE.sel, v));
  if (!ok) { sbePaintInspector(); return; }
  // The stage is showing a frame of a clip whose clock just changed.
  if (!SBE.playing) sbeShowFrameAt(SBE.playhead);
}

// One place decides what the stage looks like, so the video layer and the
// still layer can never disagree about the grade.
// THE PREVIEW'S OPACITY, a value per frame rather than a CSS transition: a
// scrub has to show what is TRUE at that second, not an animation that
// started when you arrived. The stage's black is what a fade to black fades
// to, so this is the whole of it on the picture lane.
function sbeFadePaint(t) {
  const v = sbeEl('sbeVideo');
  const img = sbeEl('sbeStill');
  const now = (t === undefined) ? SBE.playhead : sbeNum(t);
  // THE CLIP AT THE PLAYHEAD, not `curId`. `curId` is the transport's own
  // bookkeeping and is only current while something is playing — reading it
  // during a SCRUB gave the stale clip, or none at all, and the ramp a person
  // is dragging is exactly the one they are scrubbing to look at.
  const c = sbeClipAt(SBE.clips, now) || sbeById(SBE.clips, SBE.curId);
  const o = c ? sbeFadeOpacityAt(c, now, sbeTxEdges(SBE.clips, SBE.transitions, c.id, sbeFps())) : 1;
  // OPACITY IS ALSO THE LAYER SWITCH. `.sbe-still` is opacity:0 until it wins
  // `.is-on`, so writing the ramp onto BOTH layers turned the hidden one on --
  // and the still, being last in the stage and backed with #000, painted a
  // black rectangle over a perfectly loaded video. Grade the layer that is
  // showing; hand the other one back to the stylesheet.
  const paint = (el) => {
    if (!el) return;
    el.style.opacity = el.classList.contains('is-on') ? String(o) : '';
  };
  paint(v);
  paint(img);
}

function sbeApplyPreviewFilter(b, frame) {
  const css = (Math.abs(sbeNum(b)) < 1e-6) ? '' : 'brightness(' + sbeBrightnessCss(b) + ')';
  const v = sbeEl('sbeVideo');
  const i = sbeEl('sbeStill');
  if (v) v.style.filter = css;
  if (i) i.style.filter = css;
  sbeApplyPreviewFraming(frame);
}

// THE REFRAME ON THE STAGE: the layer scaled about the anchor. Approximate
// — the stage letterboxes with object-fit, so a picture that is not 16:9
// scales about a point slightly off the source's own fraction — and the
// inspector says so; the render's crop is exact.
function sbeApplyPreviewFraming(frame) {
  const f = sbeFraming({ frame: frame || {} });
  const t = (Math.abs(f.zoom - 1) < 1e-9) ? '' : 'scale(' + f.zoom.toFixed(3) + ')';
  const o = (f.x * 100).toFixed(2) + '% ' + (f.y * 100).toFixed(2) + '%';
  for (const el of [sbeEl('sbeVideo'), sbeEl('sbeStill')]) {
    if (!el || !el.style) continue;
    el.style.transform = t;
    el.style.transformOrigin = t ? o : '';
  }
}

// The sliders move at pointer speed and the undo stack does not — `oninput`
// paints, `onchange` is the edit, the brightness slider's own split.
function sbeFramingPreview(field, v) {
  const c = sbeById(SBE.clips, SBE.sel);
  if (!c) return;
  const f = sbeFraming(c);
  f[field] = sbeNum(v, f[field]);
  sbeApplyPreviewFraming(f);
  const out = sbeEl('sbeFrameZoomVal');
  if (out && field === 'zoom') out.textContent = sbeFraming({ frame: f }).zoom.toFixed(2) + 'x';
}
function sbeFramingCommit(field, v) {
  if (!SBE.sel) return;
  sbeBlurControl();
  const ok = sbeMutate(cs => sbeSetFraming(cs, SBE.sel, field, v));
  if (!ok) { sbePaintInspector(); return; }
  const c = sbeById(SBE.clips, SBE.sel);
  if (c) sbeApplyPreviewFraming(sbeFraming(c));
}
function sbeFramingReset() {
  if (!SBE.sel) return;
  sbeMutate(cs => {
    const c = sbeById(cs, SBE.sel);
    if (!c || !c.frame) return { clips: cs, ok: false, why: '' };
    const out = cs.map(x => Object.assign({}, x));
    delete sbeById(out, SBE.sel).frame;
    sbeById(out, SBE.sel).source = 'human';
    return { clips: out, ok: true };
  });
  sbeApplyPreviewFraming(null);
}

function sbePaintChrome() {
  const board = sbeEl('sbeBoardBtn');
  if (board) board.hidden = !SBE.id;
  sbeEl('sbeUndoBtn').disabled = !SBE.undo.length;
  sbeEl('sbeRedoBtn').disabled = !SBE.redo.length;
  const save = sbeEl('sbeSaveBtn');
  save.disabled = !SBE.dirty || SBE.saving;
  // THE ROW HAS A HIERARCHY WHILE THERE IS SOMETHING TO DO. Seven buttons at
  // one fill made the safest action and the one that LEAVES look alike, and
  // nothing in the row changed when the state chip went amber.
  if (save.classList) {
    save.classList.toggle('primary', !!SBE.dirty && !SBE.saving);
    save.classList.toggle('ghost-btn', !(SBE.dirty && !SBE.saving));
  }
  sbeEl('sbePlayBtn').textContent = SBE.playing ? 'Pause' : 'Play';
  // The zoom slider is a VIEW of SBE.pps, not a second copy of it: the − / +
  // buttons, alt + wheel and a resize all move the handle by coming through
  // here, so the control can never disagree with the track it is scaling.
  const zr = sbeEl('sbeZoomRange');
  if (zr) {
    const lo = sbeZoomMin();
    zr.value = String(sbeZoomToSlider(SBE.pps, lo, Math.max(lo, SBE_PPS_MAX)));
  }
  const pn = sbeEl('sbeProgName');
  if (pn) {
    const n = SBE.clips.length;
    pn.textContent = n + ' clip' + (n === 1 ? '' : 's') + ' · ' +
                     sbeFmtTime(sbeFilmDuration(SBE.clips));
  }
  // THE PROGRAM MONITOR'S EMPTY STATE WAS MARKUP NOBODY FILLED. The element
  // and its style shipped with the two-monitor pass and nothing ever set its
  // text, so the source monitor greeted an empty state with a sentence while
  // the program monitor showed a black rectangle and a 10px chip — two
  // monitors speaking two languages about the same condition.
  const pe = sbeEl('sbeStageEmpty');
  if (pe) {
    if (SBE.clips.length) { pe.hidden = true; pe.textContent = ''; }
    else {
      pe.textContent = 'Nothing on the timeline yet — drag a clip from the '
                     + 'media pool, or press + on one.';
      pe.hidden = false;
    }
  }
  // The source monitor's own chrome — same rule, one painter.
  sbePaintSource();
  // unplaced
  const wrap = sbeEl('sbeUnplacedWrap');
  const list = SBE.unplaced || [];
  wrap.hidden = !list.length;
  if (list.length) {
    sbeEl('sbeUnplaced').innerHTML = list.map((u, i) => {
      const at = (u.slot && u.slot.film_start !== undefined)
        ? sbeNum(u.slot.film_start) : sbeFilmDuration(SBE.clips);
      // The timecode is the same on every row, so it belongs on the button's
      // title rather than repeated thirteen times down a 230px column.
      return '<span class="sbe-chip" title="' +
        escapeHtml((u.title || ('shot ' + u.n)) + ' — would land at ' +
                   sbeFmtTime(at)) + '">' +
        '<b>' + escapeHtml(sbeNiceName(u.title || ('shot ' + u.n))) + '</b>' +
        '<span>' + escapeHtml(u.duration_s ? sbeNum(u.duration_s).toFixed(1) + 's' : (u.pass || '')) + '</span>' +
        '<button type="button" class="ghost-btn" onclick="sbePlace(' + i + ')">Place</button></span>';
    }).join('');
  }
  // prepare
  const job = SBE.prepare || {};
  const running = job.state === 'running';
  sbeEl('sbePrepBtn').disabled = running;
  sbeEl('sbePrepCancel').hidden = !running;
  sbeEl('sbeAutoBtn').disabled = running;
  const bar = sbeEl('sbePrepBar');
  bar.hidden = !running;
  if (running) {
    const total = sbeNum(job.total, 0);
    const pct = total ? Math.min(100, Math.round(sbeNum(job.done) / total * 100)) : 12;
    bar.querySelector('i').style.width = pct + '%';
    const stage = { proxies: 'building proxies', peaks: 'reading the waveform',
                    beats: 'finding the beat', start: 'starting' }[job.stage] || job.stage || 'working';
    sbeEl('sbePrepText').textContent = stage +
      (total ? ' · ' + sbeNum(job.done) + ' of ' + total : '') +
      (job.current ? ' · ' + job.current : '');
  } else if (job.state === 'failed') {
    sbeEl('sbePrepText').textContent = 'Prepare failed: ' + (job.error || 'unknown reason');
  } else if (job.state === 'cancelled') {
    sbeEl('sbePrepText').textContent = 'Prepare cancelled. Whatever it built is kept.';
  } else if (job.state === 'done') {
    const failed = (job.failed || []).length;
    sbeEl('sbePrepText').textContent =
      'Ready · ' + sbeNum(job.built) + ' proxy(s) built' +
      (failed ? ', ' + failed + ' clip(s) could not be read' : '') +
      (job.peaks_error ? ' · ' + job.peaks_error : '') +
      (job.beats_error ? ' · ' + job.beats_error : '') +
      // A grid the ARRANGEMENT has never seen. Prepare found the beat, but only
      // the auto-editor writes it into edit.json — so say that plainly instead
      // of drawing a timeline with no lines on it and letting the user wonder.
      ((job.beats && !SBE.beats)
        ? ' · found the beat at ' + Math.round(sbeNum(job.beats.bpm)) +
          ' bpm — this arrangement was cut without it, so Auto-edit is what ' +
          'puts the cuts on it'
        : '');
  } else {
    sbeEl('sbePrepText').textContent =
      'The soundtrack drives the waveform and the beat grid.';
  }
}

// ---------------------------------------------------------------------------
// POINTER: scrub, drag, trim
// ---------------------------------------------------------------------------
function sbeTimeFromEvent(ev, el) {
  const r = el.getBoundingClientRect();
  return Math.max(0, (ev.clientX - r.left + el.scrollLeft * 0) / SBE.pps);
}

function sbeSnapEnabled(ev) {
  const box = sbeEl('sbeSnapOn');
  const on = !box || box.checked;
  return on && !(ev && (ev.altKey || ev.metaKey));   // Alt is the override
}

function sbeOnTrackDown(ev) {
  const track = sbeEl('sbeTrack');
  // Same as the lane: the flag is a button, not part of the block.
  const badge = ev.target.closest('.sbe-sync');
  if (badge) { ev.preventDefault(); sbeResyncSel(badge.dataset.sync); return; }
  const gap = ev.target.closest('.sbe-gap');
  if (gap) {
    sbeGenOpen(sbeNum(gap.dataset.gapStart), sbeNum(gap.dataset.gapDur));
    return;
  }
  // A CUT IS A SUBJECT OF ITS OWN. Clicking the mark on a boundary selects
  // the boundary, and the inspector becomes the place to put a transition on
  // it — no lane, no panel, nothing that reads as bolted on.
  const cut = ev.target.closest('.sbe-cut');
  if (cut) {
    ev.preventDefault();
    SBE.txSel = cut.dataset.after;
    SBE.sel = '';
    SBE.ovSel = '';
    sbePaint();
    return;
  }
  const blk = ev.target.closest('.sbe-clip');
  if (!blk) { SBE.txSel = ''; sbeSeek(sbeTimeFromEvent(ev, track)); sbePaint(); return; }
  const id = blk.dataset.id;
  SBE.sel = id;
  SBE.ovSel = '';           // one inspector, one subject
  SBE.txSel = '';
  SBE.audioDrag = null;     // same insurance the lane takes against the other
  const c = sbeById(SBE.clips, id);
  if (!c) return;
  // A LOCKED SHOT SAYS SO. Until now it refused every drag in silence: the CSS
  // turned the cursor to `not-allowed`, hid both grips, and nothing on screen
  // said why or what to press. The owner hit it while cutting — "trying to
  // drag the video clip from EITHER edge shows a forbidden cursor" — because
  // Lock was the only button that sounded like "keep these two together",
  // which is what re-link now actually does.
  if (c.locked) {
    sbePaint();
    // AND IT NAMES THE OTHER VERB. Lock pins a shot to its place on the film
    // so everything else flows around it; it is NOT the button that makes a
    // clip and its sound travel together, which is what he was reaching for
    // when he pressed it. That one is Link sound.
    phosToast('That shot is locked to its place on the __SEQ__, so it cannot be '
              + 'moved or trimmed from either edge — click Unlock in the '
              + 'inspector. To move a shot and its sound TOGETHER, unlink the '
              + 'sound, put it where you want it, then press Link sound.',
              { duration: 9000 });
    return;
  }
  // THE CORNER HANDLE, before the grips: it sits on top of the left grip's
  // hit area at the head of the block, and a fade drag is the more specific
  // gesture of the two.
  const fh = ev.target.closest('.sbe-fade-h');
  if (fh) {
    const e0 = sbeFx(c);
    SBE.drag = { id: id, mode: 'fade', edge: fh.dataset.fade, x0: ev.clientX,
                 f0: (fh.dataset.fade === 'out') ? e0.fade_out : e0.fade_in,
                 moved: false, before: JSON.stringify(SBE.clips) };
    try { track.setPointerCapture(ev.pointerId); } catch (e) {}
    ev.preventDefault();
    sbePaint();
    return;
  }
  // AN EDGE IS A TRIM WHEREVER YOU GRAB IT. The grips are drawn 9 px wide; a
  // pointer within the edge zone of the block trims even when it lands a
  // pixel outside the grip element — the reach every NLE gives an edge.
  let grip = ev.target.closest('.sbe-grip');
  if (!grip) {
    const br = blk.getBoundingClientRect();
    const EDGE = 10;
    if (ev.clientX - br.left <= EDGE) grip = { classList: { contains: () => false } };
    else if (br.right - ev.clientX <= EDGE) grip = { classList: { contains: (k) => k === 'r' } };
  }
  // SHIFT IS REORDER. The hint strip already teaches "hold Alt to ignore the
  // beat", so a second modifier on the same gesture is the idiom this timeline
  // already has — and it is the only way to offer both verbs without a mode
  // switch nobody would find. Free drag stays the default: it is what every
  // arrangement on disk was made with, and it is the one that can open a hole
  // for the generate control to fill.
  const mode = grip ? (grip.classList.contains('r') ? 'trimR' : 'trimL')
                    : (ev.shiftKey ? 'reorder' : 'move');
  // ⌘ / CTRL IS RIPPLE: the gesture also slides everything after the clip,
  // the way it did before 2026-09-05. Read again on every move so it can be
  // pressed or released mid-drag, as in Premiere.
  SBE.drag = { id: id, mode: mode, x0: ev.clientX, t0: sbeTimeFromEvent(ev, track),
               fs0: sbeNum(c.film_start), fe0: sbeNum(c.film_end), moved: false,
               ripple: !!(ev.metaKey || ev.ctrlKey),
               before: JSON.stringify(SBE.clips) };
  if (mode === 'move') blk.classList.add('is-drag');
  track.classList.toggle('is-ripple', SBE.drag.ripple);
  try { track.setPointerCapture(ev.pointerId); } catch (e) {}
  ev.preventDefault();
  sbePaint();
}

function sbeOnTrackMove(ev) {
  const d = SBE.drag;
  if (!d) return;
  const track = sbeEl('sbeTrack');
  const dt = (ev.clientX - d.x0) / SBE.pps;
  if (Math.abs(ev.clientX - d.x0) > 3) d.moved = true;
  if (!d.moved) return;
  const tol = SBE_SNAP_PX / SBE.pps;
  const snapOn = sbeSnapEnabled(ev);
  const off = (SBE.audio || {}).offset;
  if (d.mode === 'fade') {
    // Dragging INWARD lengthens the ramp, whichever end you grabbed — the
    // gesture every NLE uses, and the only one that reads as "pull the fade
    // out of the corner".
    const dt = (d.edge === 'out') ? -(ev.clientX - d.x0) / SBE.pps
                                  : (ev.clientX - d.x0) / SBE.pps;
    const r = sbeSetFade(SBE.clips, d.id, d.edge, Math.max(0, d.f0 + dt));
    if (r.ok) SBE.clips = r.clips;
    sbePaint();
    return;
  }
  if (d.mode === 'reorder') {
    // No snap: a reorder does not choose a TIME, it chooses a neighbour, and
    // pulling the drop index onto a beat would mean nothing.
    const r = sbeReorderTo(SBE.clips, d.id, Math.max(0, d.t0 + dt));
    if (r.ok) SBE.clips = r.clips;
  } else {
    // A move or a trim always works from the SNAPSHOT: the model clamps at
    // neighbours, and re-applying deltas to an already-clamped state would
    // let a pointer that kept going drag the clip through the wall.
    const ripple = !!(ev.metaKey || ev.ctrlKey);
    d.ripple = ripple;
    track.classList.toggle('is-ripple', ripple);
    SBE.clips = JSON.parse(d.before);
    const o = { ripple: ripple };
    if (d.mode === 'move') {
      const want = sbeSnapTime(Math.max(0, d.fs0 + dt), SBE.beats, tol, snapOn, off);
      const r = sbeMoveTo(SBE.clips, d.id, want, o);
      if (r.ok) SBE.clips = r.clips;
    } else if (d.mode === 'trimL') {
      const want = sbeSnapTime(Math.max(0, d.fs0 + dt), SBE.beats, tol, snapOn, off);
      sbeTrim(SBE.clips, d.id, 'l', want, o);
    } else {
      const want = sbeSnapTime(Math.max(0, d.fe0 + dt), SBE.beats, tol, snapOn, off);
      sbeTrim(SBE.clips, d.id, 'r', want, o);
    }
  }
  sbePaint();
}

// EVERYTHING A DRAG ON THE TRACK CAN MOVE, in one string. `sbeOnTrackUp` asks
// "did the film actually change" and restores the pointerdown snapshot when it
// did not — dragging a clip left when it is already hard against its neighbour
// is the common case, and without this every such gesture would burn a
// revision and push an undo step that undoes nothing.
//
// SO THE LIST HAS TO BE COMPLETE, and twice now it has not been: a gesture
// whose only effect was on a strip compared equal and was discarded, and then
// a fade did the same. The rule is that every field any `SBE.drag.mode` writes
// belongs here — trimL and trimR move `start`/`end`, move and reorder move
// `film_start`, a ripple moves `audio`, and fade moves `fx`. Named and pure so
// a new mode can be checked against it instead of against a closure.
function sbeDragFingerprint(cs) {
  return (cs || []).map(c => [c.id, c.start, c.end, c.film_start,
                              JSON.stringify(c.audio || null),
                              JSON.stringify(c.fx || null)].join(':')).join('|');
}

function sbeOnTrackUp(ev) {
  const d = SBE.drag;
  SBE.drag = null;
  if (!d) return;
  document.querySelectorAll('.sbe-clip.is-drag').forEach(el => el.classList.remove('is-drag'));
  const trk = sbeEl('sbeTrack'); if (trk) trk.classList.remove('is-ripple');
  if (!d.moved) { sbePaint(); return; }
  // A DRAG THAT MOVED THE POINTER BUT NOT THE FILM IS NOT AN EDIT. Dragging a
  // clip left when it is already hard against its neighbour is the common case
  // — the maths clamps, nothing lands anywhere new, and without this the
  // document would still come back dirty, stamped `human`, with a revision
  // burned and an undo step that undoes nothing.
  // ...AND THE SOUND IS PART OF THE GEOMETRY. The fingerprint listed four
  // picture fields, so a gesture whose only effect was on a strip — a ripple
  // carrying an unlinked sound, a coupled pair travelling with its picture —
  // compared equal to the state it started from, and this line then RESTORED
  // the snapshot: the edit was discarded, the document was never marked
  // dirty, and the header went on saying "saved · revision N" over work that
  // had just been thrown away.
  // ...AND SO IS THE FADE, which is the SECOND HALF of "not sure that grabbing
  // the top corner is working. I'm not being able to do it." The corner handle
  // was too small to hit — that is the CSS — but on the times he DID hit it,
  // this comparison then threw the result away: `mode: 'fade'` moves `fx` and
  // nothing else, `fx` was not in the fingerprint, so the drag compared equal
  // to the state it started from and the snapshot was restored on pointerup.
  // The ramp followed the pointer the whole way and snapped back the moment he
  // let go. Exactly the bug the paragraph above describes, one field along —
  // which is why the fingerprint is a NAMED function now, listing every field
  // a track drag can move, rather than a closure written to the needs of
  // whichever gesture was being added at the time.
  const before = JSON.parse(d.before);
  if (sbeDragFingerprint(before) === sbeDragFingerprint(SBE.clips)) {
    SBE.clips = before; sbePaint(); return;
  }
  // The drag mutated SBE.clips live so the block could follow the pointer;
  // register the ONE undo step now, from the snapshot taken on pointerdown.
  SBE.undo.push(d.before);
  if (SBE.undo.length > SBE_UNDO_MAX) SBE.undo.shift();
  SBE.redo.length = 0;
  SBE.dirty = true;
  sbeSetState('unsaved changes', 'dirty');
  sbePaint();
  sbeQueueSave();
}

function sbeSeek(t) {
  const dur = Math.max(sbeFilmDuration(SBE.clips), 0);
  SBE.playhead = Math.max(0, Math.min(dur, sbeNum(t)));
  sbePaintHead();
  // A SCRUB LANDS ON THE RIGHT SOURCE SECOND. Forced, because the ground has
  // moved: the slip tolerance is for drift during playback, not for a seek.
  sbeStripSync(true);
  sbeShowFrameAt(SBE.playhead);
}

// ---------------------------------------------------------------------------
// PREVIEW — one <video>, proxies, and no pretence of frame accuracy
// ---------------------------------------------------------------------------
function sbeClipUrl(c) {
  // A STILL IS SERVED BY /image, NOT /file. `/file` is OUTPUT-bound and the
  // image library lives under panel_uploads — so the obvious fallback here
  // would 404 every still in the panel. `/image` already accepts OUTPUT,
  // UPLOADS and the state dir, which is exactly the set the pool draws from.
  if (sbeKind(c) === 'still') {
    return '/image?path=' + encodeURIComponent(c.path);
  }
  if (c.proxy && SBE.proxyUrl) {
    return SBE.proxyUrl + encodeURIComponent(String(c.proxy).split('/').pop());
  }
  return '/file?path=' + encodeURIComponent(c.path);
}

function sbeLoadInto(v, c, at) {
  return new Promise(resolve => {
    // The picture used to be hard-muted here, on the reasoning that sound comes
    // from the soundtrack <audio> and Chrome refuses unmuted autoplay. Both
    // halves were wrong once proxies carried an audio track: most timelines have
    // no soundtrack at all, and their sound — dialogue, the thing you are
    // cutting on — lives in the clips. Play is user-initiated, so there is
    // activation to spend; sbePlay falls back to muted if Chrome disagrees.
    // THE THIRD OUTPUT. The render drops the clip's lane and the export
    // disables its audio track; the preview has to agree or the one place the
    // user checks his work is the one place the decision does not exist.
    //
    // ...AND IT YIELDS TO THE STRIP PLAYER. A clip whose sound is described
    // separately is played by `sbeStripSync` at the strip's own film position;
    // leaving it on the picture element too would play those seconds twice,
    // a beat apart.
    v.muted = !!SBE.muted || !sbePictureCarriesSound(c);
    v.volume = 1;
    v.playsInline = true;
    // THE CLIP'S SPEED IS THE ELEMENT'S RATE. The browser plays the same
    // seconds of the take at the rate the render's `setpts` will, so the
    // preview and the file agree about how fast the shot goes by.
    try { v.playbackRate = sbeSpeed(c); } catch (e) {}
    const url = sbeClipUrl(c);
    const seek = () => {
      const t = Math.max(0, Math.min(sbeNum(c.end) - 1e-3, sbeNum(at)));
      const done = () => { v.removeEventListener('seeked', done); resolve(); };
      v.addEventListener('seeked', done);
      try { v.currentTime = t; } catch (e) { v.removeEventListener('seeked', done); resolve(); }
      setTimeout(() => { v.removeEventListener('seeked', done); resolve(); }, 2500);
    };
    if (v.dataset.src === url && v.readyState >= 1) { seek(); return; }
    v.dataset.src = url;
    const meta = () => { v.removeEventListener('loadedmetadata', meta); seek(); };
    v.addEventListener('loadedmetadata', meta);
    const fail = () => { v.removeEventListener('error', fail); resolve(); };
    v.addEventListener('error', fail);
    v.src = url;
    try { v.load(); } catch (e) {}
  });
}

async function sbeShowFrameAt(t) {
  const v = sbeEl('sbeVideo');
  const img = sbeEl('sbeStill');
  if (!v) return;
  // THE OVERLAY LANE IS NOT THE PICTURE'S BUSINESS, so it is painted BEFORE
  // any of the picture's early returns. A card may outlive the last shot —
  // that is the whole reason the render pads the base with black — and this
  // function bails out as soon as no clip covers the playhead, which left the
  // card stuck on screen at full opacity past its own end. Caught on the test
  // panel, not by a unit test.
  sbeOvPaint(t);
  const c = sbeClipAt(SBE.clips, t);
  SBE.curId = c ? c.id : '';
  if (!c) {
    v.classList.remove('is-on');
    if (img) img.classList.remove('is-on');
    sbeEl('sbeBadge').textContent = SBE.clips.length ? 'nothing plays here' : 'no clips';
    return;
  }
  const kind = sbeKind(c);
  sbeApplyPreviewFilter(sbeBright(c), sbeFraming(c));
  if (kind === 'slug') {
    // NO FILE, SO NOTHING TO LOAD. Both layers off leaves the stage's own
    // black, which is the frame the render will write — the one case where the
    // preview is not an approximation at all.
    v.classList.remove('is-on');
    if (img) img.classList.remove('is-on');
    sbeEl('sbeBadge').textContent = 'black · ' + sbeLen(c).toFixed(2) + 's';
    sbePaintTrack();
    return;
  }
  if (kind === 'still') {
    // An <img> cannot land on the wrong frame and cannot be slow to seek, so
    // a still never reaches the missing-proxy path the badge exists to warn
    // about. Swap the src only when it changed — reassigning it re-decodes.
    v.classList.remove('is-on');
    if (img) {
      const url = sbeClipUrl(c);
      if (img.getAttribute('src') !== url) img.src = url;
      img.classList.add('is-on');
    }
    sbeEl('sbeBadge').textContent =
      (c.title || String(c.path || '').split('/').pop()) + ' · still · ' +
      sbeLen(c).toFixed(2) + 's';
    sbePaintTrack();
    return;
  }
  if (img) img.classList.remove('is-on');
  await sbeLoadInto(v, c, sbeNum(c.start) + (t - sbeNum(c.film_start)) * sbeSpeed(c));
  v.classList.add('is-on');
  sbeEl('sbeBadge').textContent =
    (c.title || c.path.split('/').pop()) + ' · ' + sbeNum(c.start).toFixed(2) + '–' +
    sbeNum(c.end).toFixed(2) + ' · ' + (c.proxy ? 'proxy' : 'SOURCE (slow — run Prepare)');
  // A SCRUB SHOWS THE TRUE OPACITY at the second it landed on, which is the
  // only way a fade can be judged without playing the whole clip.
  sbeFadePaint(t);
  sbePaintTrack();
}

// ---------------------------------------------------------------------------
// THE SOURCE MONITOR — the left screen
// ---------------------------------------------------------------------------
// A pool row is a path and a title, not a clip: it has no proxy of its own
// until it lands on the track, and a row from the Generations tab has never
// been near this document. So the URL is the proxy of a clip already cut from
// the same file when there is one, and the file itself when there is not.
// That is fine here in a way it would not be on the timeline: a source
// monitor plays forward from zero, it does not scrub, so the 235 ms seek that
// makes source files useless under a playhead never happens.
function sbeSrcUrl(row) {
  if (!row || !row.path) return '';
  if (row.kind === 'still') return '/image?path=' + encodeURIComponent(row.path);
  const cut = (SBE.clips || []).find(c => c.path === row.path && c.proxy);
  if (cut && SBE.proxyUrl) {
    return SBE.proxyUrl + encodeURIComponent(String(cut.proxy).split('/').pop());
  }
  return '/file?path=' + encodeURIComponent(row.path);
}

// CLICKING A ROW WATCHES IT. It does not add it — that was the old verb for
// the whole row and it is now the + at the row's right edge, which is the
// swap the two-screen layout is for: "you can watch the clips before you add
// them".
function edPoolPreview(i) {
  // A drag ends in a click event on the row it started from. edPoolAdd has
  // carried this guard since drag-to-drop shipped; the preview needs it for
  // the same reason, or every drop would also change the left screen.
  if (ED.suppressClick) { ED.suppressClick = false; return; }
  const list = document.getElementById('edPoolList');
  const row = ((list || {})._rows || [])[i];
  if (!row) return;
  SBE.source = row;
  SBE.srcIndex = i;
  // Mark the row in place rather than repainting the list: a repaint drops
  // and re-attaches every thumbnail's src, which is sixty decoders for a
  // one-class change.
  list.querySelectorAll('.ed-pool-row').forEach((el, k) =>
    el.classList.toggle('is-source', k === i));
  sbeSrcLoad(row);
  sbePaintSource();
}

function sbeSrcLoad(row) {
  const v = sbeEl('sbeSrcVideo');
  const img = sbeEl('sbeSrcStill');
  const empty = sbeEl('sbeSrcEmpty');
  if (empty) empty.hidden = true;
  const url = sbeSrcUrl(row);
  if (row.kind === 'still') {
    sbeSrcStop();
    if (v) { try { v.pause(); } catch (e) {} v.classList.remove('is-on'); }
    if (img) {
      if (img.getAttribute('src') !== url) img.src = url;
      img.classList.add('is-on');
    }
    return;
  }
  if (img) img.classList.remove('is-on');
  if (!v) return;
  if (v.getAttribute('src') !== url) { v.src = url; try { v.load(); } catch (e) {} }
  v.classList.add('is-on');
  // Clicking a clip means "let me see it", so it plays. The program stops
  // first — one picture, one soundtrack, one thing to listen to.
  sbeSrcPlay();
}

async function sbeSrcPlay() {
  const v = sbeEl('sbeSrcVideo');
  if (!v || !SBE.source || SBE.source.kind === 'still') return;
  sbeStop();                      // the program yields; they never overlap
  v.muted = !!SBE.muted;
  v.volume = 1;
  SBE.srcPlaying = true;
  try { await v.play(); }
  catch (e) {
    // NOT EVERY REJECTED play() IS AN AUTOPLAY REFUSAL. Calling pause() while
    // a play promise is pending rejects it with AbortError — and pressing
    // Play on the program does exactly that, on purpose. The first version of
    // this caught that abort, decided the browser had blocked sound, muted
    // the editor and started the source playing again UNDER the program.
    // Measured: source paused:false, readyState 0, and a muted timeline.
    if (!SBE.srcPlaying) return;                 // stopped on purpose; stay stopped
    if (e && e.name === 'AbortError') return;
    // Same autoplay bargain the program makes: losing the sound beats losing
    // the picture, and a Play button that visibly does nothing is worse still.
    if (!v.muted) {
      v.muted = true;
      sbeSetMute(true, 'browser blocked sound — click 🔊 to unmute', false);
      try { await v.play(); } catch (e2) {}
    }
  }
  sbePaintSource();
}

function sbeSrcStop() {
  const v = sbeEl('sbeSrcVideo');
  SBE.srcPlaying = false;
  if (v) { try { v.pause(); } catch (e) {} }
  sbePaintSource();
}

function sbeSrcToggle() { SBE.srcPlaying ? sbeSrcStop() : sbeSrcPlay(); }

// The button under the left screen. Exactly the + on the row, by construction:
// same function, same index, same server call.
function sbeSrcAdd() {
  if (!SBE.source || SBE.srcIndex < 0) return;
  edPoolAdd(SBE.srcIndex);
}

function sbePaintSource() {
  const row = SBE.source;
  const name = sbeEl('sbeSrcName');
  const add = sbeEl('sbeSrcAddBtn');
  const play = sbeEl('sbeSrcPlayBtn');
  const badge = sbeEl('sbeSrcBadge');
  const empty = sbeEl('sbeSrcEmpty');
  if (!name) return;
  if (!row) {
    name.textContent = 'Nothing loaded';
    if (add) add.disabled = true;
    if (play) { play.disabled = true; play.textContent = 'Play'; }
    if (badge) badge.hidden = true;
    if (empty) empty.hidden = false;
    return;
  }
  const secs = sbeNum(row.duration_s, 0);
  name.textContent = (row.title || String(row.path || '').split('/').pop() || 'clip') +
    (secs ? ' · ' + secs.toFixed(1) + 's' : '');
  name.title = row.path || '';
  // Disabled when the row is no longer in the painted pool — a search or a
  // tab change can take it away, and an Add that pointed at whatever moved
  // into that slot would put the wrong clip on the film.
  if (add) add.disabled = !(SBE.open && SBE.id) || SBE.srcIndex < 0;
  if (play) {
    play.disabled = (row.kind === 'still');
    play.textContent = SBE.srcPlaying ? 'Pause' : 'Play';
  }
  if (empty) empty.hidden = true;
  if (badge) {
    // The same honesty the program badge carries: say when the picture is the
    // source file rather than a proxy, because that is the slow one.
    const proxied = (SBE.clips || []).some(c => c.path === row.path && c.proxy);
    badge.hidden = false;
    badge.textContent = (row.kind === 'still') ? 'still'
      : (proxied ? 'proxy' : 'source file');
  }
}

// What the soundtrack does to the clips' own sound. Lives on the edit so it
// survives a reload and so the render uses what the screen is showing.
function sbeMusicMode() {
  const el = sbeEl('sbeMusicMode');
  return (el && el.value === 'replace') ? 'replace' : 'under';
}

// SETTING A CONTROL TO WHAT IT ALREADY SAYS IS NOT AN EDIT. `sbeAdopt` calls
// this on every load to make the dropdown agree with the document — and it
// was marking that document dirty and queueing a write, so EVERY OPEN of
// every film produced a save the user never made. Under the old autosave that
// was silent revision churn; under drafts it would have been worse, because
// the backup lane would have overwritten an unanswered recovery offer with
// the very arrangement the offer existed to replace.
function sbeSetMusicMode(v) {
  const mode = (v === 'replace') ? 'replace' : 'under';
  const el = sbeEl('sbeMusicMode');
  if (el) el.value = mode;
  if (SBE.audio) {
    if (String(SBE.audio.mode || 'under') === mode) return;
    SBE.audio.mode = mode;
    SBE.dirty = true;
    sbeSetState('unsaved changes', 'dirty');
    sbeQueueSave();
  }
  // Only `replace` needs saying: it is the one that destroys something.
  const warn = sbeEl('sbeMusicWarn');
  if (warn) {
    warn.hidden = (mode !== 'replace');
    warn.textContent = 'any dialogue in the clips is lost';
  }
}

// ---------------------------------------------------------------------------
// THE MEDIA POOL — three sources, one verb
// ---------------------------------------------------------------------------
// `ED.src` is which source is showing, `ED.rows` is what it holds. Nothing is
// cached across sources on purpose: the gallery changes while you cut, and a
// pool that shows yesterday's list is worse than a pool that takes 40 ms.
window.ED = { src: 'film', rows: [], films: [], film: '', loading: false,
              limit: 60, io: null, drag: null, suppressClick: false,
              // What THIS session uploaded, kept in front of the list. The
              // server puts an image somewhere the gallery already walks and a
              // clip somewhere the uploads route lists, so both come back on
              // their own — but `list_outputs` holds a file back for two
              // seconds while it decides nobody is still writing it, and "it
              // appears in a moment" is not what "immediately" means.
              uploaded: [] };

function edPoolSrc(name) {
  ED.src = name;
  const tabs = document.getElementById('edPoolTabs');
  if (tabs) tabs.querySelectorAll('.pill-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.src === name));
  const pick = document.getElementById('edPoolFilm');
  if (pick) pick.hidden = (name !== 'other');
  // Upload belongs to the source that shows what was uploaded.
  const up = document.getElementById('edPoolUploadBtn');
  if (up) up.hidden = (name !== 'images');
  ED.limit = 60;
  edPoolRefresh();
}

// BRING YOUR OWN FILE. "You cannot upload your own images and insert them
// into the timeline" — and a clip from a phone is the same feature, so both
// are taken. One request per file so a rejected one names itself instead of
// failing a batch, and the answers go straight to the front of the pool.
async function edPoolUpload(files) {
  const list = Array.from(files || []);
  if (!list.length) return;
  const btn = document.getElementById('edPoolUploadBtn');
  if (btn) { btn.disabled = true; btn.textContent = 'Uploading…'; }
  const bad = [];
  let done = 0;
  for (const f of list) {
    const fd = new FormData();
    fd.append('file', f);
    let r;
    try {
      r = await (await fetch('/storyboard/edit/upload',
                             { method: 'POST', body: fd })).json();
    } catch (e) { r = { ok: false, error: String(e) }; }
    if (r && r.ok) {
      done++;
      ED.uploaded = [edPoolUploadRow(r)].concat(
        (ED.uploaded || []).filter(x => x.path !== r.path));
    } else {
      bad.push((f.name || 'file') + ' — ' + ((r && r.error) || 'upload failed'));
    }
  }
  if (btn) { btn.disabled = false; btn.textContent = 'Upload'; }
  if (ED.src !== 'images') edPoolSrc('images'); else await edPoolRefresh(true);
  if (done) {
    phosToast(done + (done === 1 ? ' file is' : ' files are') +
              ' in the pool — press + to put it at the end, or drag it onto '
              + 'the track.',
              { kind: 'success', duration: 5000 });
  }
  for (const msg of bad) phosToast(msg, { kind: 'danger', duration: 7000 });
}

// One upload's answer, as the row shape every pool source produces.
function edPoolUploadRow(r) {
  const still = String(r.kind || 'still') === 'still';
  return {
    path: r.path, title: r.title || String(r.path || '').split('/').pop(),
    kind: still ? 'still' : undefined,
    sub: 'uploaded · ' + (still ? 'still' : 'clip') +
         (r.duration_s ? ' · ' + sbeNum(r.duration_s).toFixed(1) + 's' : ''),
    duration_s: still ? SBE_STILL_SECONDS : r.duration_s,
    uploaded: true,
  };
}

// One row shape for all three sources, so the paint and the click do not have
// to know where a clip came from: {path, title, sub, duration_s, from, only}.
async function edPoolRefresh(force) {
  const list = document.getElementById('edPoolList');
  const note = document.getElementById('edPoolNote');
  if (!list) return;
  if (ED.loading) return;
  ED.loading = true;
  if (force) ED.limit = 60;
  list.innerHTML = '<span class="ed-pool-note">reading…</span>';
  if (note) note.textContent = '';
  try {
    if (ED.src === 'film') {
      ED.rows = (SBE.pool || []).map(c => ({
        path: c.path, title: c.title || ('shot ' + c.n),
        sub: (c.pass || '') + (c.duration_s ? ' · ' + sbeNum(c.duration_s).toFixed(1) + 's' : '') +
             (c.placed ? ' · on the track' : ''),
        duration_s: c.duration_s, placed: !!c.placed, n: c.n,
      }));
    } else if (ED.src === 'other') {
      await edPoolLoadFilms();
      ED.rows = await edPoolLoadFilmShots(ED.film);
    } else if (ED.src === 'images') {
      // THE GALLERY ALWAYS HAD THESE. `/outputs` has carried `kind: 'image'`
      // since the unified gallery shipped; what was missing was a clip that
      // could BE one. A still lands with a default hold and is resized by the
      // same trim handles every other block has.
      const r = await (await fetch('/outputs?limit=400&offset=0')).json();
      const gallery = ((r && r.outputs) || [])
        .filter(o => o.kind === 'image' && !o.hidden)
        .map(o => ({
          path: o.path, title: o.name, kind: 'still',
          sub: 'still · ' + (o.engine ? String(o.engine).toUpperCase() + ' · ' : '') +
               (o.mtime || '').slice(0, 10),
          duration_s: SBE_STILL_SECONDS,
        }));
      // The clips somebody brought with them. Deliberately NOT in OUTPUT —
      // that folder means "the panel made this" — so they come from their own
      // listing and sit at the front, with this session's uploads ahead of
      // them so a file is in the list the instant it lands.
      let mine = [];
      try {
        const u = await (await fetch('/storyboard/edit/uploads')).json();
        mine = ((u && u.uploads) || []).map(o => ({
          path: o.path, title: o.name, sub: 'uploaded · clip',
          uploaded: true,
        }));
      } catch (e) { mine = []; }
      const seen = {};
      ED.rows = (ED.uploaded || []).concat(mine, gallery)
        .filter(row => (row.path && !seen[row.path]) && (seen[row.path] = 1));
    } else {
      const r = await (await fetch('/outputs?limit=400&offset=0')).json();
      ED.rows = ((r && r.outputs) || [])
        .filter(o => o.kind === 'video' && !o.hidden)
        .map(o => ({
          path: o.path, title: o.name,
          sub: (o.clip_sec ? sbeNum(o.clip_sec).toFixed(1) + 's · ' : '') +
               (o.engine ? String(o.engine).toUpperCase() + ' · ' : '') +
               (o.mtime || '').slice(0, 10),
          duration_s: o.clip_sec,
        }));
    }
  } catch (e) {
    ED.rows = [];
    if (note) note.textContent = 'The panel did not answer.';
  }
  ED.loading = false;
  edPoolPaint();
}

async function edPoolLoadFilms() {
  let boards = [];
  try {
    const r = await (await fetch('/storyboard/list')).json();
    boards = ((r && r.boards) || []).filter(b => (b.clips || 0) > 0 && b.id !== SBE.id);
  } catch (e) { boards = []; }
  ED.films = boards;
  if (!ED.film || !boards.some(b => b.id === ED.film)) {
    ED.film = boards.length ? boards[0].id : '';
  }
  const pick = document.getElementById('edPoolFilm');
  if (pick) {
    pick.innerHTML = boards.map(b =>
      '<option value="' + escapeHtml(b.id) + '"' + (b.id === ED.film ? ' selected' : '') +
      '>' + escapeHtml(b.title || b.id) + ' · ' + (b.clips || 0) + ' clips</option>').join('');
  }
}

function edPoolPickFilm(id) { ED.film = id; ED.limit = 60; edPoolRefresh(); }

async function edPoolLoadFilmShots(id) {
  if (!id) return [];
  let r;
  try { r = await (await fetch('/storyboard/get?id=' + encodeURIComponent(id))).json(); }
  catch (e) { return []; }
  const shots = (((r || {}).board) || {}).shots || [];
  return shots.map(sh => {
    const path = sh.final_output || sh.draft_output || '';
    if (!path || sh.status === 'skipped') return null;
    return { path: path, title: sh.title || sh.prompt || ('shot ' + sh.n),
             sub: 'S' + String(sh.n).padStart(2, '0') + ' · ' +
                  (sh.final_output ? 'delivery' : 'draft') +
                  (sh.duration_s ? ' · ' + sbeNum(sh.duration_s).toFixed(1) + 's' : ''),
             duration_s: sh.duration_s, from: id, only: sh.n };
  }).filter(Boolean);
}

function edPoolPaint() {
  const list = document.getElementById('edPoolList');
  const note = document.getElementById('edPoolNote');
  if (!list) return;
  const q = ((document.getElementById('edPoolSearch') || {}).value || '')
    .trim().toLowerCase();
  const rows = (ED.rows || []).filter(r =>
    !q || (r.title || '').toLowerCase().indexOf(q) !== -1 ||
    (r.path || '').toLowerCase().indexOf(q) !== -1);
  if (!rows.length) {
    list.innerHTML = '<span class="ed-pool-note">' +
      // AN EMPTY LIST IS A SENTENCE, NOT A BLANK. Each of these says what is
      // true of THIS source and what the next move is — a pool that goes
      // quiet reads as broken, and the four sources fail for four different
      // reasons.
      (q
        ? 'Nothing here matches “' + escapeHtml(q) + '”. Clear the filter to see everything.'
        : ED.src === 'gallery'
          ? 'No generations yet. Render something in Video or Storyboard and it lands here.'
        : ED.src === 'images'
          ? 'No images yet — press Upload to bring one in from this Mac, or make one in the Image studio.'
        : ED.src === 'other'
          ? (ED.films && ED.films.length
              ? 'That __SEQ__ has no rendered clips yet.'
              : 'No other __SEQS__ yet. Every __SEQ__ you make shows up here to borrow clips from.')
        : 'This __SEQ__ has no rendered clips yet. Render its shots in Storyboard, or take one from another source above.') + '</span>';
    if (note) note.textContent = '';
    return;
  }
  const show = rows.slice(0, ED.limit);
  // `#t=0.1` is what makes a <video preload="metadata"> paint a frame instead
  // of black — no thumbnail pipeline, no extra request, and it is the same
  // /file route with ranges the gallery already uses.
  //
  // The src is NOT in the markup. Chrome caps how many media elements one
  // document may have (the carousel alone already holds ~240 on a working
  // install), and a pool that attached sixty more on paint would spend that
  // budget on rows nobody has scrolled to. `data-src` + an observer means at
  // most the visible dozen are ever loaded, and leaving a row unloads it.
  // A ROW IS A DIV WITH A ROLE, NOT A BUTTON. It has to hold a button of its
  // own now — the + that still adds without previewing — and a <button>
  // inside a <button> is not markup any browser will keep.
  const srcPath = (SBE.source || {}).path || '';
  list.innerHTML = show.map((r, i) =>
    '<div class="ed-pool-row' + ((srcPath && r.path === srcPath) ? ' is-source' : '') +
    '" role="button" tabindex="0" onclick="edPoolPreview(' + i + ')" ' +
    'onkeydown="if(event.key===\'Enter\'||event.key===\' \'){event.preventDefault();edPoolPreview(' + i + ');}" ' +
    'onpointerdown="edPoolDragStart(event,' + i + ')" ' +
    'title="' + escapeHtml(r.path) + '">' +
    // An image has no frame to seek to, so it is an <img> — and the /image
    // route's own `w=` resize is what keeps a 14 MB PNG from being decoded at
    // full size for a 44 px thumbnail.
    (r.kind === 'still'
      ? '<img alt="" data-src="/image?w=180&path=' + encodeURIComponent(r.path) + '">'
      : '<video preload="metadata" muted playsinline data-src="/file?path=' +
        encodeURIComponent(r.path) + '#t=0.1"></video>') +
    '<span class="ed-pool-meta">' +
      '<span class="ed-pool-name">' + escapeHtml(sbeNiceName(r.title || '')) + '</span>' +
      '<span class="ed-pool-sub">' + escapeHtml(r.sub || '') + '</span>' +
    '</span>' +
    '<button type="button" class="ed-pool-add" ' +
    'title="Put this clip at the end of the __SEQ__" ' +
    'onclick="event.stopPropagation();edPoolAdd(' + i + ')">+</button>' +
    // AN IMAGE CAN BE A CARD. A transparent PNG belongs on the overlay lane,
    // not at the end of the picture track, and this is the one click that
    // says so — it lands at the playhead, where you are looking.
    (r.kind === 'still'
      ? '<button type="button" class="ed-pool-add ed-pool-ov" ' +
        'title="Lay this over the picture at the playhead (overlay lane)" ' +
        'onclick="event.stopPropagation();edPoolOverlay(' + i + ')">▣</button>'
      : '') +
    '</div>').join('');
  list._rows = show;
  edPoolObserve(list);
  // The row the source monitor is showing may have moved, or be gone: a
  // search, a tab change and a refresh all repaint this list, and the index
  // "Add to timeline" hands back has to be the one in the list on screen.
  if (SBE.source) {
    SBE.srcIndex = show.findIndex(r => r.path === SBE.source.path);
    sbePaintSource();
  }
  if (note) {
    note.textContent = rows.length > show.length
      ? (show.length + ' of ' + rows.length + ' shown.')
      : (rows.length + ' clip' + (rows.length === 1 ? '' : 's') +
         ' · click one to watch it, + to put it at the end, or drag it onto the track.');
  }
}

// One observer for the pool's whole life, re-pointed at each repaint. It
// attaches a thumbnail's src on the way in and takes it away on the way out,
// so the number of live decoders is the number of rows you can see.
function edPoolObserve(list) {
  if (!('IntersectionObserver' in window)) {
    list.querySelectorAll('[data-src]').forEach(v => {
      v.src = v.dataset.src; });
    return;
  }
  if (ED.io) ED.io.disconnect();
  // The first screenful loads WITHOUT waiting for the observer. Intersection
  // callbacks do not run while a tab is occluded, and a pool of black
  // rectangles is indistinguishable from a pool of broken clips — so the rows
  // that are on screen at paint time never depend on a callback at all.
  [...list.querySelectorAll('[data-src]')].slice(0, 12).forEach(v => {
    if (!v.getAttribute('src')) v.src = v.dataset.src;
  });
  ED.io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      const v = e.target;
      if (e.isIntersecting) {
        if (!v.getAttribute('src') && v.dataset.src) v.src = v.dataset.src;
      } else if (v.getAttribute('src')) {
        v.removeAttribute('src');
        // Only a <video> holds a decoder to release; load() on an <img> is not
        // a function, and calling it would throw once per row leaving the view.
        if (v.tagName === 'VIDEO') { try { v.load(); } catch (err) {} }
      }
    }
  }, { root: list, rootMargin: '150px' });
  list.querySelectorAll('[data-src]').forEach(v => ED.io.observe(v));
}

// THE VERB. A clip joins the timeline, gets its proxy built before it lands,
// and the user does not move: adding to a cut is not a reason to leave it.
async function edPoolAdd(i, dropAt) {
  const list = document.getElementById('edPoolList');
  const row = ((list || {})._rows || [])[i];
  if (!row) return;
  // A DRAG ENDS IN A CLICK EVENT. The browser fires one after every pointerup
  // on a <button>, so without this flag every drop would also add the clip a
  // second time, at the end of the track.
  if (ED.suppressClick && dropAt === undefined) { ED.suppressClick = false; return; }
  if (!SBE.open || !SBE.id) {
    phosToast('Open a __SEQ__ first — the Editor holds one timeline at a time.', {});
    return;
  }
  // The row's own word first, then the file's. The server checks the suffix
  // too — this is the half that keeps the CLIENT from asking /file for a
  // picture, which is what painted a black frame.
  const isStill = (row.kind === 'still')
    || /\.(png|jpe?g|webp|bmp|tiff?)$/i.test(String(row.path || ''));
  const note = document.getElementById('edPoolNote');
  if (note) {
    note.textContent = isStill
      ? ('reading ' + (row.title || 'that image') + '…')
      : ('building the proxy for ' + (row.title || 'that clip') + '…');
  }
  const fd = new URLSearchParams();
  fd.set('id', SBE.id);
  if (isStill) { fd.set('kind', 'still'); fd.set('path', row.path); }
  else if (row.from) { fd.set('from', row.from); fd.set('only', String(row.only)); }
  else fd.set('path', row.path);
  if (row.title) fd.set('title', row.title);
  let r;
  try { r = await (await fetch('/storyboard/edit/add-clip', { method: 'POST', body: fd })).json(); }
  catch (e) { r = { ok: false, error: String(e) }; }
  if (!r.ok) {
    if (note) note.textContent = '';
    phosToast(r.error || 'That clip could not be added.', { kind: 'danger' });
    return;
  }
  const c = r.clip || {};
  const item = {
    path: c.path || row.path, proxy: c.proxy || null,
    kind: isStill ? 'still' : 'video',
    duration_s: (isStill ? SBE_STILL_SECONDS
                         : (c.duration_s || row.duration_s || null)),
    title: c.title || row.title || '', n: c.n,
  };
  // A CLICK LANDS AT THE END; A DROP LANDS WHERE IT WAS DROPPED. The first
  // cannot move anybody's cuts, which is why it is still the default verb; the
  // second was ASKED for, and a drop that ignored where the pointer was would
  // be a drag with no meaning.
  let at;
  const ok = sbeMutate(cs => {
    if (dropAt === undefined || dropAt === null) {
      at = sbeFilmDuration(cs);
      return sbePlaceUnplaced(cs, item, at);
    }
    at = Math.max(0, sbeNum(dropAt));
    return sbeInsertAt(cs, item, at);
  });
  if (!ok) { if (note) note.textContent = ''; return; }
  await sbeSave(true);
  if (note) note.textContent = '';
  phosToast('Added at ' + sbeFmtTime(at) +
            (isStill ? ' · still' : ' · proxy ready.'),
            { kind: 'success', duration: 4000 });
  // The board may have gained a shot (an import), and the payload's `unplaced`
  // and `clips` are now stale. Re-read quietly; nothing on screen moves.
  sbeLoad(true);
}

// ---------------------------------------------------------------------------
// DRAG: a pool row onto the track
// ---------------------------------------------------------------------------
// POINTER EVENTS, NOT HTML5 DRAG-AND-DROP. The track's own move/trim gestures
// are already pointerdown/move/up with pointer capture, and mixing the two
// models means a dragstart that swallows the capture and a drop the track
// never hears. This is the same substrate the track uses, so the two coexist
// by construction rather than by luck. Click-to-add is untouched: a press that
// never travels more than a few pixels is still a click.
function edPoolDragStart(ev, i) {
  const list = document.getElementById('edPoolList');
  const row = ((list || {})._rows || [])[i];
  if (!row || !SBE.open || !SBE.id) return;
  if (ev.button !== undefined && ev.button !== 0) return;
  // The ELEMENT the press started on, not just the data row — the guard at
  // the end of the drag needs to know whether the click the browser is about
  // to fire will land back here.
  ED.drag = { row: row, index: i, x0: ev.clientX, y0: ev.clientY,
              moved: false, ghost: null,
              el: (ev.target && ev.target.closest)
                ? ev.target.closest('.ed-pool-row') : null };
  // The pool's drag is the one gesture in this panel that let the browser
  // start a text selection underneath it, so every drop painted the entire
  // editor — labels, times, hint bar, rail — in selection blue until the
  // next click.
  if (ev.preventDefault) ev.preventDefault();
  window.addEventListener('pointermove', edPoolDragMove);
  window.addEventListener('pointerup', edPoolDragEnd, { once: true });
}

function edPoolDragMove(ev) {
  const d = ED.drag;
  if (!d) return;
  if (!d.moved) {
    if (Math.abs(ev.clientX - d.x0) + Math.abs(ev.clientY - d.y0) < 5) return;
    d.moved = true;
    // PORTALLED TO <body>, because the pool list has its own overflow and a
    // ghost parented inside it is clipped the instant it leaves the column —
    // which is every drag, since the track is in the other column.
    d.ghost = document.createElement('div');
    d.ghost.className = 'ed-drag-ghost';
    d.ghost.textContent = d.row.title || d.row.path || 'clip';
    document.body.appendChild(d.ghost);
  }
  d.ghost.style.left = (ev.clientX + 12) + 'px';
  d.ghost.style.top = (ev.clientY + 12) + 'px';
  const track = sbeEl('sbeTrack');
  const over = edPoolOverTrack(ev, track);
  track.classList.toggle('is-dropping', over);
  SBE.dropAt = over ? edPoolDropTime(ev, track) : null;
  sbePaintTrack();
}

// The film time under the pointer, snapped to the boundary a drop would use —
// so the line the user sees is the position the clip will take, not the pixel
// their finger happens to be over.
function edPoolDropTime(ev, track) {
  const t = sbeTimeFromEvent(ev, track);
  const idx = sbeDropIndex(SBE.clips, t);
  const prev = SBE.clips[idx - 1];
  return prev ? sbeNum(prev.film_end) : 0;
}

function edPoolOverTrack(ev, track) {
  if (!track) return false;
  const r = track.getBoundingClientRect();
  // Generous vertically: a drop aimed at a 78px strip from another column
  // should not fail because the pointer was four pixels high.
  return ev.clientX >= r.left && ev.clientX <= r.right &&
         ev.clientY >= r.top - 24 && ev.clientY <= r.bottom + 24;
}

async function edPoolDragEnd(ev) {
  const d = ED.drag;
  ED.drag = null;
  window.removeEventListener('pointermove', edPoolDragMove);
  if (!d) return;
  if (d.ghost) d.ghost.remove();
  const track = sbeEl('sbeTrack');
  if (track) track.classList.remove('is-dropping');
  const at = SBE.dropAt;
  SBE.dropAt = null;
  sbePaintTrack();
  if (!d.moved) return;                       // a press that never travelled
  // ARMED ONLY IF THE CLICK IS ACTUALLY COMING. "A drag ends in a click event
  // on the row it started from" is true for a press that never left the row
  // and false for every drag onto the track — which is all of them. So the
  // flag outlived the gesture and ate the user's next genuine click: you drop
  // a clip on the timeline, click another clip to preview it, and the source
  // monitor does nothing until you click a second time.
  ED.suppressClick = !!(d.el && ev.target && d.el.contains(ev.target));
  if (at === null || at === undefined) return;   // dropped in open space
  await edPoolAdd(d.index, at);
}

// ---------------------------------------------------------------------------
// BLACK — the one clip with no file
// ---------------------------------------------------------------------------
// No server round trip, because there is nothing on disk to check, no
// geometry to probe and no proxy to build. It is a length and a kind.
function edAddSlug() {
  if (!SBE.open || !SBE.id) {
    phosToast('Open a __SEQ__ first — the Editor holds one timeline at a time.', {});
    return;
  }
  const box = document.getElementById('edSlugSecs');
  const secs = Math.max(SBE_MIN_CLIP, Math.min(60, sbeNum(box && box.value, 2) || 2));
  const at = sbeFilmDuration(SBE.clips);
  const ok = sbeMutate(cs => sbeInsertAt(cs, {
    kind: 'slug', title: 'black', duration_s: secs,
  }, at));
  if (!ok) return;
  sbeSave(true);
  phosToast(secs.toFixed(1) + 's of black at ' + sbeFmtTime(at) + '.',
            { kind: 'success', duration: 3500 });
}

// ---------------------------------------------------------------------------
// RELINK — draft → delivery
// ---------------------------------------------------------------------------
function sbePaintRelink() {
  const bar = sbeEl('sbeRelink');
  if (!bar) return;
  const all = SBE.relink || [];
  const rows = all.filter(r => !r.retake);
  const takes = all.filter(r => r.retake && !sbeRetakeDismissed(r));
  bar.hidden = !rows.length;
  if (rows.length) {
    sbeEl('sbeRelinkText').textContent = rows.length +
      (rows.length === 1 ? ' shot was' : ' shots were') +
      ' finished since this cut — use the finished versions.';
  }
  // A RETAKE IS ONE DECISION PER TAKE, so each gets its own line and its
  // own two answers. The old take stays until the person says otherwise.
  const tb = sbeEl('sbeRetakes');
  if (!tb) return;
  tb.hidden = !takes.length;
  if (!takes.length) return;
  tb.innerHTML = takes.map(r =>
    '<span class="sbe-relink-row">' +
    '<span>New take of <b>' + escapeHtml(sbeNiceName(r.title || ('shot ' + r.n))) + '</b> is ready.</span>' +
    '<button type="button" class="ghost-btn" onclick="sbeRetakeUse(\'' + escapeHtml(r.id) + '\')" ' +
      'title="Replace the clip with the new take. Same cut, same timings.">Use it</button>' +
    '<button type="button" class="ghost-btn" onclick="sbeRetakeKeep(\'' + escapeHtml(r.id) + '\', \'' + escapeHtml(r.to) + '\')" ' +
      'title="Keep the take that is on the timeline. The new one stays in the media pool.">Keep the old one</button>' +
    '</span>').join('');
}

// "Keep the old one" is a decision about ONE file against ONE clip, kept in
// this browser: the take stays in the pool, and this line does not come back
// for it.
function sbeRetakeDismissed(r) {
  try {
    const seen = JSON.parse(localStorage.getItem('phos_retake_keep') || '[]');
    return seen.indexOf(String(r.id) + '|' + String(r.to)) >= 0;
  } catch (e) { return false; }
}
function sbeRetakeKeep(id, to) {
  try {
    const seen = JSON.parse(localStorage.getItem('phos_retake_keep') || '[]');
    seen.push(String(id) + '|' + String(to));
    localStorage.setItem('phos_retake_keep', JSON.stringify(seen.slice(-200)));
  } catch (e) {}
  sbePaintRelink();
}
async function sbeRetakeUse(id) {
  if (SBE.dirty && !SBE.conflict && !(await sbeSave(true))) {
    phosToast('Your arrangement could not be saved, and the swap works on the saved file — fix the save first.',
              { kind: 'danger', duration: 8000 });
    return;
  }
  const fd = new URLSearchParams();
  fd.set('id', SBE.id);
  fd.set('only', id);
  let r;
  try { r = await (await fetch('/storyboard/edit/relink', { method: 'POST', body: fd })).json(); }
  catch (e) { r = { ok: false, error: String(e) }; }
  if (!r || !r.ok) { phosToast((r && r.error) || 'The take could not be swapped in.', { kind: 'danger' }); return; }
  SBE.undo.length = 0; SBE.redo.length = 0;
  sbeAdopt(r, true);
  phosToast('The new take is on the timeline. Same cut, same timings.', { kind: 'success', duration: 5000 });
}

async function sbeRelink() {
  const btn = sbeEl('sbeRelinkBtn');
  if (btn) btn.disabled = true;
  // Save first: the server rewrites the file on disk, and an unsaved
  // arrangement would be rewritten out from under itself. CHECKED — a save
  // that did not land means the server is holding an older cut, and relinking
  // that one writes it back over what is on screen.
  if (SBE.dirty && !SBE.conflict && !(await sbeSave(true))) {
    if (btn) btn.disabled = false;
    phosToast('Your arrangement could not be saved, and relinking works on ' +
              'the saved file — fix the save first.',
              { kind: 'danger', duration: 8000 });
    return;
  }
  const fd = new URLSearchParams();
  fd.set('id', SBE.id);
  let r;
  try { r = await (await fetch('/storyboard/edit/relink', { method: 'POST', body: fd })).json(); }
  catch (e) { r = { ok: false, error: String(e) }; }
  if (btn) btn.disabled = false;
  if (!r || !r.ok) {
    phosToast((r && r.error) || 'Those clips could not be relinked.', { kind: 'danger' });
    return;
  }
  SBE.undo.length = 0; SBE.redo.length = 0;
  sbeAdopt(r, true);
  phosToast((r.relinked || 0) + ' clip(s) now play the finished files. ' +
            'Same cuts, same timings.', { kind: 'success', duration: 6000 });
}

// ---------------------------------------------------------------------------
// IMPORT — clips from another film, into this one
// ---------------------------------------------------------------------------
// The pool is a column away, and on a stacked window it is above the fold.
// This is the way back to it from the prepare bar.
function edPoolFocus() {
  const pane = document.getElementById('edSectionTab');
  if (pane && pane.scrollIntoView) pane.scrollIntoView({ behavior: 'smooth', block: 'start' });
  const q = document.getElementById('edPoolSearch');
  if (q) { try { q.focus(); } catch (e) {} }
}

function sbeTogglePlay() { SBE.playing ? sbeStop() : sbePlay(); }

// Sound. One switch drives the picture, the soundtrack bed and the button, so
// they can never disagree about whether this timeline is audible.
// A BROWSER REFUSING TO AUTOPLAY IS NOT A PREFERENCE, and this is the second
// half of why the Editor could go permanently silent. When Chrome declines an
// unmuted play() the editor mutes itself so the picture still runs — right
// call — but it wrote that decision to localStorage, so ONE refusal silenced
// every later session: clips AND soundtrack, across reloads, with a 34px
// speaker button as the only way back. Reported as total preview audio loss,
// which is exactly what it is.
//
// `remember` is FALSE for that path: the mute lasts as long as the tab, the
// stored preference is left alone, and it says so out loud instead of
// appending a sentence to a 10.5px line under the transport.
function sbeSetMute(on, note, remember) {
  SBE.muted = !!on;
  if (remember !== false) {
    try { localStorage.setItem('sbeMuted', SBE.muted ? '1' : '0'); } catch (e) {}
  }
  const v = sbeEl('sbeVideo');
  if (v) { v.muted = SBE.muted; v.volume = 1; }
  // ONE OPINION ABOUT SOUND, across both monitors. A source screen with its
  // own mute would be a second source of truth for the exact state this
  // switch exists to keep single.
  const sv = sbeEl('sbeSrcVideo');
  if (sv) { sv.muted = SBE.muted; sv.volume = 1; }
  const a = SBE.musicEl;
  if (a) { a.muted = SBE.muted; }
  const b = sbeEl('sbeMuteBtn');
  if (b) {
    b.textContent = SBE.muted ? '🔇' : '🔊';
    b.classList.toggle('is-off', SBE.muted);
    b.title = note || (SBE.muted ? 'Sound off — click to unmute (M)' : 'Sound on (M)');
  }
  if (note) {
    const el = sbeEl('sbeApprox');
    if (el && el.dataset.muteNoted !== '1') {
      el.dataset.muteNoted = '1';
      el.textContent = note + ' ' + el.textContent;
    }
    if (typeof phosToast === 'function' && !SBE.muteToasted) {
      SBE.muteToasted = true;
      phosToast('The browser would not start sound without a click, so the '
                + 'preview is muted for now — press the speaker (or M) to '
                + 'turn it back on. Your setting has not been changed.',
                { kind: 'warn', duration: 9000 });
    }
  }
}

// THE ONE-CLICK WAY BACK, and the reason the refusal above is survivable: a
// press of the speaker is a user gesture, so the sound it turns on is sound
// the browser will now allow.
function sbeUnmuteFromRefusal() {
  SBE.muteToasted = false;
  sbeSetMute(false);
  if (SBE.playing) { const v = sbeEl('sbeVideo'); if (v) { try { v.play(); } catch (e) {} } }
  sbeMusicPlay();
}

// TAKE THE STAGE TO A CLIP, WHATEVER KIND IT IS. The video branch loads,
// seeks and plays; the other two only decide which layer is lit, because
// there is nothing to decode, nothing to seek, and no clock to start.
async function sbeEnter(c, at) {
  const v = sbeEl('sbeVideo');
  const img = sbeEl('sbeStill');
  sbeApplyPreviewFilter(sbeBright(c), sbeFraming(c));
  const kind = sbeKind(c);
  if (kind === 'video') {
    if (img) img.classList.remove('is-on');
    await sbeLoadInto(v, c, (at === undefined || at === null) ? sbeNum(c.start) : sbeNum(at));
    v.classList.add('is-on');
    try { v.playbackRate = sbeSpeed(c); } catch (e) {}
    try { await v.play(); } catch (e) {}
    return;
  }
  try { v.pause(); } catch (e) {}
  v.classList.remove('is-on');
  if (img) {
    if (kind === 'still' && c.path) {
      const url = sbeClipUrl(c);
      if (img.getAttribute('src') !== url) img.src = url;
      img.classList.add('is-on');
    } else {
      img.classList.remove('is-on');   // a slug: the stage's own black
    }
  }
}

async function sbePlay() {
  if (SBE.playing) return;
  const c = sbeClipAt(SBE.clips, SBE.playhead) || SBE.clips[0];
  if (!c) return;
  sbeSrcStop();          // the source yields; the two monitors never overlap
  SBE.playing = true;
  SBE.lastTs = 0;
  sbePaintChrome();
  if (!sbeClipAt(SBE.clips, SBE.playhead)) SBE.playhead = sbeNum(c.film_start);
  const v = sbeEl('sbeVideo');
  SBE.curId = c.id;
  if (sbeKind(c) === 'video') {
    // ENTERING MID-CLIP: film seconds into the slot are `speed` source
    // seconds into the take — the review caught a 2x clip re-seeking on
    // every frame because this seek landed at the 1x position.
    await sbeLoadInto(v, c, sbeNum(c.start) + (SBE.playhead - sbeNum(c.film_start)) * sbeSpeed(c));
    v.classList.add('is-on');
    try { v.playbackRate = sbeSpeed(c); } catch (e) {}
    try {
      await v.play();
    } catch (e) {
      // NotAllowedError = no user activation to spend (autoplay policy). Losing
      // the picture is worse than losing the sound, so fall back to muted and say
      // so, rather than leaving a Play button that visibly does nothing.
      // An AbortError is a pause we asked for (the source monitor taking the
      // screen), and treating that as a block would mute the editor for a
      // reason that has nothing to do with the browser.
      if (!SBE.playing || (e && e.name === 'AbortError')) { /* stopped on purpose */ }
      else if (!v.muted) {
        v.muted = true;
        sbeSetMute(true, 'browser blocked sound — click 🔊 to unmute', false);
        try { await v.play(); } catch (e2) {}
      }
    }
    sbeApplyPreviewFilter(sbeBright(c), sbeFraming(c));
  } else {
    await sbeEnter(c);
  }
  sbeMusicPlay();
  sbeStripSync(true);
  SBE.raf = requestAnimationFrame(sbeFrame);
}

function sbeStop() {
  SBE.playing = false;
  SBE.lastTs = 0;
  if (SBE.raf) { cancelAnimationFrame(SBE.raf); SBE.raf = 0; }
  const v = sbeEl('sbeVideo');
  if (v) { try { v.pause(); } catch (e) {} }
  sbeMusicStop();
  sbeStripStop();
  if (SBE.open) sbePaintChrome();
}

async function sbeFrame() {
  if (!SBE.playing) return;
  const v = sbeEl('sbeVideo');
  // THE WALL CLOCK, for the clips that do not carry one. A still and a slug
  // have no <video> whose currentTime the playhead can be read off, so without
  // this the transport froze the instant a film reached its first black.
  // Capped at a quarter second so a backgrounded tab resumes rather than
  // jumping half a film forward on its first frame back.
  const now = (typeof performance !== 'undefined' && performance.now)
    ? performance.now() : Date.now();
  const dt = SBE.lastTs ? Math.max(0, Math.min(0.25, (now - SBE.lastTs) / 1000)) : 0;
  SBE.lastTs = now;
  const c = sbeById(SBE.clips, SBE.curId);
  if (c && sbeKind(c) !== 'video') {
    SBE.playhead += dt;
    if (SBE.playhead >= sbeNum(c.film_end) - 1e-3) {
      const next = SBE.clips[SBE.clips.indexOf(c) + 1];
      if (!next) {
        SBE.playhead = sbeNum(c.film_end);
        sbeStop(); sbePaintHead(); return;
      }
      SBE.playing = false;                       // hold the loop across the cut
      SBE.playhead = sbeNum(next.film_start);
      SBE.curId = next.id;
      await sbeEnter(next);
      SBE.playing = true;
      sbeMusicSync();
      sbePaintTrack();
    }
  } else if (c) {
    SBE.playhead = sbeNum(c.film_start) + Math.max(0, v.currentTime - sbeNum(c.start)) / sbeSpeed(c);
    if (v.currentTime >= sbeNum(c.end) - 1e-3 || v.ended) {
      const next = SBE.clips[SBE.clips.indexOf(c) + 1];
      if (!next) { sbeStop(); sbePaintHead(); return; }
      SBE.playing = false;                       // hold the loop across the cut
      SBE.playhead = sbeNum(next.film_start);
      SBE.curId = next.id;
      await sbeEnter(next);
      SBE.playing = true;
      sbeMusicSync();
      sbePaintTrack();
    }
  } else {
    // A hole: the picture goes black and the clock keeps running, which is
    // exactly what the render will do if the gap is ever filled.
    v.classList.remove('is-on');
    const img = sbeEl('sbeStill');
    if (img) img.classList.remove('is-on');
    const nxt = SBE.clips.find(x => sbeNum(x.film_start) > SBE.playhead);
    SBE.playhead += dt || (1 / 60);
    if (!nxt) { sbeStop(); sbePaintHead(); return; }
    if (SBE.playhead >= sbeNum(nxt.film_start)) {
      SBE.curId = nxt.id;
      await sbeEnter(nxt);
      sbeMusicSync();
    }
  }
  // EVERY FRAME, because a strip starts and stops on its own clock rather
  // than at the cuts — `sbeMusicSync` can ride the transitions, this cannot.
  sbeStripSync();
  sbeFadePaint();
  sbeOvPaint();
  sbePaintHead();
  SBE.raf = requestAnimationFrame(sbeFrame);
}

// The soundtrack is best-effort and says so when it cannot be had: /file serves
// what is under mlx_outputs, and a track living anywhere else is still perfectly
// good for the waveform, the grid and the render — it just cannot be auditioned
// in this browser.
function sbeSyncMusic() {
  const path = (SBE.audio || {}).path || '';
  if (!path) { SBE.musicEl = null; SBE.music = ''; return; }
  if (SBE.music === path && SBE.musicEl) return;
  SBE.music = path;
  SBE.musicOk = true;
  const a = new Audio('/file?path=' + encodeURIComponent(path));
  a.preload = 'auto';
  a.muted = !!SBE.muted;
  a.addEventListener('error', () => {
    SBE.musicOk = false;
    const el = sbeEl('sbeApprox');
    if (el && el.dataset.musicNoted !== '1') {
      el.dataset.musicNoted = '1';
      el.textContent = el.textContent +
        ' The soundtrack cannot be played from here (it lives outside mlx_outputs) — the waveform, the beat grid and the render still use it.';
    }
  });
  SBE.musicEl = a;
}

// THE BED PLAYS THE WINDOW THE STRIP SHOWS. Both of these read `offset` and
// nothing else, so a track whose intro had been trimmed off played untrimmed
// in the Editor and trimmed in the file — the exact "what the strip shows and
// what the render builds cannot come apart" contract sbeMusicWindow claims to
// establish, broken on the one surface the user listens to.
function sbeMusicAt(t) {
  const w = sbeMusicWindow(SBE.audio, SBE.peaks ? SBE.peaks.duration : 0);
  if (t < w.film_start - 1e-3) return null;               // not in yet
  if (w.film_end !== null && t >= w.film_end) return null;  // out already
  return Math.max(0, w.head + (t - w.film_start));
}
function sbeMusicPlay() {
  const a = SBE.musicEl;
  if (!a || !SBE.musicOk) return;
  const at = sbeMusicAt(SBE.playhead);
  try {
    if (at === null) { a.pause(); return; }
    a.currentTime = at;
    a.play().catch(() => {});
  } catch (e) {}
}
function sbeMusicSync() {
  const a = SBE.musicEl;
  if (!a || !SBE.musicOk || !SBE.playing) return;
  const want = sbeMusicAt(SBE.playhead);
  if (want === null) { try { if (!a.paused) a.pause(); } catch (e) {} return; }
  // THE MIX, HEARD. Not the bed's envelope alone — the whole of what the
  // render will apply to this track: the fader, the envelope, and the duck if
  // it is on and nothing outranks it. `sbeBedGainPoints` is the same function
  // the render's `volume` expression is built from and the same one the level
  // line draws, so this <audio> element and the mp4 cannot come apart. It was
  // the envelope and nothing else, while the render silently held the bed at
  // 0.20 and ducked it — "when you render it, there are some weird
  // manipulations... the volume of the music goes low when the dialogue
  // appears."
  //
  // ON THE BED'S OWN CLOCK, which is the played window and not the track: the
  // playhead less the film second the music starts at.
  const bw = sbeMusicWindow(SBE.audio, SBE.peaks ? SBE.peaks.duration : 0);
  a.volume = sbeBedGainAt(SBE.audio || {}, SBE.clips,
                          sbeFilmDuration(SBE.clips),
                          SBE.playhead - bw.film_start);
  try {
    if (a.paused) a.play().catch(() => {});
    if (Math.abs(a.currentTime - want) > 0.25) a.currentTime = want;
  } catch (e) {}
}
function sbeMusicStop() { const a = SBE.musicEl; if (a) { try { a.pause(); } catch (e) {} } }

// ---- THE STRIP PLAYER ---------------------------------------------------
// A POOL, NOT AN ELEMENT. The music bed is one track and one <audio>; clip
// strips overlap by design — that is what a split edit IS — so a J-cut needs
// two voices sounding at once across the cut. Three: two for the overlap and
// one spare, so a strip that starts while two are still ringing does not have
// to steal one that is audible.
const SBE_STRIP_VOICES = 3;
// The same tolerance the music bed uses. Re-seeking on every frame would
// stutter; letting it drift further than a quarter second would be audible.
const SBE_STRIP_SLIP = 0.25;

function sbeStripPool() {
  if (!SBE.stripEls) SBE.stripEls = [];
  while (SBE.stripEls.length < SBE_STRIP_VOICES) {
    const a = new Audio();
    a.preload = 'auto';
    a.dataset.clip = '';
    SBE.stripEls.push(a);
  }
  return SBE.stripEls;
}

// Drive every audible strip from the PLAYHEAD, exactly as the bed is driven —
// independent of which picture is on stage, which is the entire point.
// `force` re-seeks rather than tolerating slip: a scrub or an edit has moved
// the ground, so "close enough" is the wrong answer.
function sbeStripSync(force) {
  if (!SBE.open) return;
  const pool = sbeStripPool();
  const want = sbeStripsAt(SBE.clips, SBE.playhead);
  const live = {};
  for (const w of want) live[w.id] = w;
  // A voice whose strip has gone quiet is released BEFORE any is claimed, or
  // an overlap of two would find the pool full of yesterday's clips.
  for (const a of pool) {
    if (a.dataset.clip && !live[a.dataset.clip]) {
      try { a.pause(); } catch (e) {}
      a.dataset.clip = '';
    }
  }
  for (const w of want) {
    let a = null;
    for (const el of pool) if (el.dataset.clip === w.id) { a = el; break; }
    let fresh = false;
    if (!a) {
      for (const el of pool) if (!el.dataset.clip) { a = el; break; }
      if (!a) continue;                  // more overlap than voices: skip one
      a.dataset.clip = w.id;
      fresh = true;
    }
    const url = w.path ? ('/file?path=' + encodeURIComponent(w.path)) : '';
    if (url && a.dataset.src !== url) {
      a.dataset.src = url;
      a.src = url;
      fresh = true;
    }
    a.muted = !!SBE.muted;
    // THE ENVELOPE, HEARD. The render builds a `volume` expression and the
    // export writes level keyframes; the preview has to move the same number
    // or the one place the user checks his work is the one place the fade
    // does not exist. `w.at - w.start` is the STRIP-relative second, which is
    // the envelope's own clock.
    const c2 = sbeById(SBE.clips, w.id);
    const win = c2 ? sbeClipAudio(c2) : null;
    // The envelope's clock is the strip AS PLAYED — film seconds into it —
    // which is `(source second - in-point) / speed`.
    a.volume = win ? sbeGainAt(c2, win.len, (w.at - win.start) / win.speed) : 1;
    try { a.playbackRate = win ? win.speed : 1; } catch (e) {}
    try {
      if (fresh || force || Math.abs(a.currentTime - w.at) > SBE_STRIP_SLIP) {
        a.currentTime = w.at;
      }
      if (SBE.playing) { if (a.paused) a.play().catch(() => {}); }
      else if (!a.paused) a.pause();
    } catch (e) {}
  }
}

function sbeStripStop() {
  for (const a of (SBE.stripEls || [])) {
    try { a.pause(); } catch (e) {}
  }
}

// ---------------------------------------------------------------------------
// COMMANDS
// ---------------------------------------------------------------------------
// THE VIEW FOLLOWS THE PICTURE, ONE SCREENFUL AT A TIME. Called from
// sbePaintHead, which runs on every animation frame while the film plays, so
// the cheap early return in sbeFollowScroll is what keeps it from writing
// scrollLeft sixty times a second and cancelling a drag of the scrollbar.
function sbeFollow() {
  if (!SBE.playing) return;
  const box = sbeEl('sbeScroll');
  if (!box) return;
  const max = Math.max(0, box.scrollWidth - box.clientWidth);
  if (max <= 0) return;
  const want = sbeFollowScroll(sbePx(SBE.playhead), box.scrollLeft,
                               box.clientWidth, max);
  if (Math.abs(want - box.scrollLeft) > 1) box.scrollLeft = want;
}

// ---- zoom, all three doors: the buttons, the slider, and alt + wheel ----
// The floor is never a constant: it is whatever puts THIS film in THIS
// window, so the slider's left end always means "all of it".
function sbeZoomMin() {
  const box = sbeEl('sbeScroll');
  return sbeZoomFitPps(sbeSpan(), box ? box.clientWidth : 900);
}

// `at` is a film time to hold still (alt + wheel passes the one under the
// pointer); without it the playhead is held if it is on screen, the middle of
// the view if it is not.
function sbeZoomTo(pps, at) {
  const box = sbeEl('sbeScroll');
  const lo = sbeZoomMin();
  const hi = Math.max(lo, SBE_PPS_MAX);
  const want = Math.max(lo, Math.min(hi, sbeNum(pps, SBE.pps)));
  if (!box) { SBE.pps = want; sbePaint(); return; }
  const anchor = sbeZoomAnchor(SBE.playhead, box.scrollLeft, box.clientWidth,
                               SBE.pps, at);
  SBE.pps = want;
  sbePaint();
  box.scrollLeft = sbeZoomScroll(anchor, SBE.pps,
                                 Math.max(0, box.scrollWidth - box.clientWidth));
  sbePaintChrome();
}

// The − / + buttons keep the old ladder, clamped to the live floor so they can
// never step past "the whole film" and leave the slider disagreeing with them.
function sbeZoom(dir) {
  const lo = sbeZoomMin();
  const steps = [8, 12, 18, 26, 42, 64, 96, 140, 200].filter(s => s > lo);
  steps.unshift(lo);
  let i = 0;
  for (let k = 0; k < steps.length; k++) if (steps[k] <= SBE.pps + 1e-6) i = k;
  sbeZoomTo(steps[Math.max(0, Math.min(steps.length - 1, i + dir))]);
}

function sbeZoomSlide(v) {
  const lo = sbeZoomMin();
  sbeZoomTo(sbeZoomFromSlider(v, lo, Math.max(lo, SBE_PPS_MAX)));
}

// shift + wheel pans, alt + wheel zooms around the pointer — the two gestures
// every NLE and every DAW already has. A plain wheel is left alone: it belongs
// to the column, which scrolls vertically.
function sbeOnTlWheel(ev) {
  const box = sbeEl('sbeScroll');
  if (!box) return;
  if (ev.altKey) {
    ev.preventDefault();
    const r = box.getBoundingClientRect();
    const at = (box.scrollLeft + (ev.clientX - r.left)) / Math.max(1e-6, SBE.pps);
    const k = Math.pow(1.0015, -(ev.deltaY || 0));
    sbeZoomTo(SBE.pps * k, at);
    return;
  }
  // A trackpad sends horizontal intent as deltaX with no modifier at all, and
  // refusing that would be refusing the one gesture Mac users already know.
  const dx = (ev.shiftKey ? (ev.deltaY || ev.deltaX) : ev.deltaX) || 0;
  if (!dx) return;
  ev.preventDefault();
  box.scrollLeft += dx;
}

function sbeRippleSelected() {
  if (!SBE.sel) return;
  sbeMutate(cs => sbeRippleDelete(cs, SBE.sel));
  SBE.sel = '';
  sbePaint();
}

function sbeLiftSelected() {
  if (!SBE.sel) return;
  sbeMutate(cs => sbeLiftDelete(cs, SBE.sel));
  SBE.sel = '';
  sbePaint();
}

function sbeDuplicateSel() {
  if (!SBE.sel) return;
  let added = null;
  const ok = sbeMutate(cs => { const r = sbeDuplicate(cs, SBE.sel); if (r.ok) added = r.added; return r; });
  if (!ok) return;
  // THE COPY IS THE SELECTION NOW — it is the thing just made, and the one
  // the next key press should act on.
  if (added) { SBE.sel = added.id; sbePaint(); }
  sbeBlurControl();
}

function sbeSplitHere() {
  sbeMutate(cs => sbeSplitAt(cs, SBE.playhead, undefined, SBE.transitions || []));
}

function sbeToggleLock() {
  const c = sbeById(SBE.clips, SBE.sel);
  if (!c) return;
  const before = JSON.stringify(SBE.clips);
  c.locked = !c.locked;
  if (c.locked) c._pin = sbeNum(c.film_start); else delete c._pin;
  c.source = 'human';
  sbeLayout(SBE.clips);
  SBE.undo.push(before);
  SBE.redo.length = 0;
  SBE.dirty = true;
  sbeSetState('unsaved changes', 'dirty');
  sbePaint();
  sbeQueueSave();
}

function sbePlace(i) {
  const u = (SBE.unplaced || [])[i];
  if (!u) return;
  const at = (u.slot && u.slot.film_start !== undefined)
    ? sbeNum(u.slot.film_start) : sbeFilmDuration(SBE.clips);
  const ok = sbeMutate(cs => sbePlaceUnplaced(cs, u, at));
  if (ok) {
    SBE.unplaced = SBE.unplaced.filter((_, k) => k !== i);
    sbePaint();
  }
}

// ---------------------------------------------------------------------------
// PREPARE / AUTO / GENERATE / RENDER
// ---------------------------------------------------------------------------
async function sbePrepare() {
  const fd = new URLSearchParams();
  fd.set('id', SBE.id);
  const music = (sbeEl('sbeMusic').value || '').trim();
  if (music) fd.set('music', music);
  let r;
  try { r = await (await fetch('/storyboard/edit/prepare', { method: 'POST', body: fd })).json(); }
  catch (e) { r = { ok: false, error: String(e) }; }
  if (!r.ok) {
    phosToast(r.error || 'Could not start.', { kind: 'danger' });
    if (r.prepare) { SBE.prepare = r.prepare; sbePaintChrome(); }
    return;
  }
  SBE.prepare = r.prepare || { state: 'running', stage: 'start' };
  sbePaintChrome();
}

async function sbePrepareCancel() {
  const fd = new URLSearchParams(); fd.set('id', SBE.id);
  try { await fetch('/storyboard/edit/cancel', { method: 'POST', body: fd }); } catch (e) {}
}

async function sbeAuto() {
  if (!confirm('Re-cut this film from scratch?\n\n' +
      'The auto-editor picks the best window of every clip and cuts on the beat. ' +
      'It THROWS AWAY the arrangement on screen — every trim, every move, every split.\n\n' +
      'The clips themselves are untouched.')) return;
  const fd = new URLSearchParams();
  fd.set('id', SBE.id);
  const music = (sbeEl('sbeMusic').value || '').trim();
  if (music) fd.set('music', music);
  let r;
  try { r = await (await fetch('/storyboard/edit/auto', { method: 'POST', body: fd })).json(); }
  catch (e) { r = { ok: false, error: String(e) }; }
  if (!r.ok) { phosToast(r.error || 'The auto-edit failed.', { kind: 'danger' }); return; }
  SBE.undo.length = 0; SBE.redo.length = 0;
  sbeAdopt(r, true);
  phosToast('Re-cut · ' + (r.edit.clips || []).length + ' clips.', { kind: 'success' });
}

let _sbeGenAt = 0;
// THE SHOT A CLIP CAME FROM, from the pool the payload already carries.
// By path — a clip on the timeline is a file, and the pool row for that
// file knows its shot number, its prompt and its character.
function sbeShotForClip(c) {
  if (!c || !c.path) return null;
  for (const r of (SBE.pool || [])) if (r.path === c.path) return r;
  return null;
}

// RETAKE: the same modal a hole uses, seeded from the clip — its prompt to
// edit, its length as the length, its slot as the slot — and `retake_of` so
// the new take comes back offered against THIS clip rather than as a loose
// shot under the timeline.
let _sbeRetakeOf = '';
function sbeRetakeSel() {
  const c = sbeById(SBE.clips, SBE.sel);
  const shot = c ? sbeShotForClip(c) : null;
  if (!c || !shot) { phosToast('Only a clip that came from a shot can be retaken.', {}); return; }
  const len = Math.max(0.5, sbeNum(shot.duration_s, 0) || (sbeLen(c) * sbeSpeed(c) + 1));
  sbeGenOpen(sbeNum(c.film_start), len, { retakeOf: c.id, prompt: shot.prompt || '',
                                          name: shot.title || String(c.path).split('/').pop() });
}

function sbeGenOpen(filmStart, duration, opts) {
  _sbeGenAt = sbeNum(filmStart);
  _sbeRetakeOf = (opts && opts.retakeOf) || '';
  sbeEl('sbeGenWhere').textContent = _sbeRetakeOf
    ? ('A new take of ' + (opts.name || 'this clip') + ': the same shot and character, a new ' +
       'seed, the prompt below to edit. When it lands, a line above the timeline offers it ' +
       'against this clip — Use it, or keep the old one.')
    : ('Nothing plays between ' + sbeFmtTime(filmStart) + ' and ' +
       sbeFmtTime(sbeNum(filmStart) + sbeNum(duration)) + '. This shot is written to the ' +
       'storyboard and rendered like any other.');
  const ttl = sbeEl('sbeGenTitle');
  if (ttl) ttl.textContent = _sbeRetakeOf ? 'Retake this shot' : 'Fill this hole';
  const after = sbeEl('sbeGenAfter');
  if (after) after.hidden = !!_sbeRetakeOf;
  const box = sbeEl('sbeGenPrompt');
  if (box) box.value = (opts && opts.prompt) ? opts.prompt : '';
  const go = sbeEl('sbeGenGo');
  if (go) go.textContent = _sbeRetakeOf ? 'Queue the retake' : 'Queue the shot';
  // FLOOR, not round: a shot longer than the hole it was ordered for pushes
  // everything after it off the beat the moment it is placed.
  sbeEl('sbeGenDuration').value =
    Math.max(0.5, Math.floor(sbeNum(duration) * 10) / 10).toFixed(1);
  sbeEl('sbeGenParams').hidden = true;
  sbeEl('sbeGenParams').textContent = '';
  sbeEl('sbeGenGo').disabled = false;
  sbeEl('sbeGenModal').classList.add('show');
  setTimeout(() => { try { sbeEl('sbeGenPrompt').focus(); } catch (e) {} }, 30);
}
function sbeGenClose() { sbeEl('sbeGenModal').classList.remove('show'); }

async function sbeGenSubmit() {
  const prompt = (sbeEl('sbeGenPrompt').value || '').trim();
  if (!prompt) { phosToast('Write what the shot shows.', {}); return; }
  const btn = sbeEl('sbeGenGo');
  btn.disabled = true;
  const fd = new URLSearchParams();
  fd.set('id', SBE.id);
  fd.set('prompt', prompt);
  fd.set('duration', String(sbeNum(sbeEl('sbeGenDuration').value, 5)));
  fd.set('film_start', String(_sbeGenAt));
  if (_sbeRetakeOf) fd.set('retake_of', _sbeRetakeOf);
  const on = document.querySelector('#sbeGenPass .pill-btn.active');
  fd.set('pass', (on && on.dataset.pass) || 'draft');
  let r;
  try { r = await (await fetch('/storyboard/edit/generate', { method: 'POST', body: fd })).json(); }
  catch (e) { r = { ok: false, error: String(e) }; }
  if (!r.ok) {
    btn.disabled = false;
    phosToast(r.error || 'Could not queue the shot.', { kind: 'danger' });
    return;
  }
  // WHAT WILL ACTUALLY RENDER. make_job silently drops any form field it does
  // not name, so the server reads the params back off the queued job and hands
  // them here rather than echoing the request.
  const box = sbeEl('sbeGenParams');
  box.hidden = false;
  box.textContent = 'queued as shot ' + r.n + ' (job ' + r.job_id + ')\n' +
    Object.keys(r.params || {}).map(k => k + ': ' + r.params[k]).join('\n');
  SBE.awaitingClip = Date.now();
  phosToast('Shot ' + r.n + ' is in the queue. It appears under the timeline when it lands.',
            { kind: 'success', duration: 6000 });
}

// DELIVER AS — remembered in this browser, posted with every render, and
// painted onto the pills when the menu is built.
const SBE_DELIVER_FORMATS = ['h264', 'hevc', 'prores'];
const SBE_DELIVER_SIZES = ['native', '1080p', '2160p'];
const SBE_DELIVER_FINISH = ['none', 'grain', 'heavy_grain'];
function sbeDeliverGet() {
  let d = {};
  try { d = JSON.parse(localStorage.getItem('phos_deliver') || '{}') || {}; } catch (e) { d = {}; }
  return { format: SBE_DELIVER_FORMATS.indexOf(d.format) >= 0 ? d.format : 'h264',
           size: SBE_DELIVER_SIZES.indexOf(d.size) >= 0 ? d.size : 'native',
           finish: SBE_DELIVER_FINISH.indexOf(d.finish) >= 0 ? d.finish : 'none' };
}
function sbeDeliverPick(key, value) {
  const d = sbeDeliverGet();
  d[key] = value;
  try { localStorage.setItem('phos_deliver', JSON.stringify(d)); } catch (e) {}
  sbeDeliverPaint();
}
function sbeDeliverPaint() {
  const d = sbeDeliverGet();
  document.querySelectorAll('#sbeDeliverFormat .pill-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.format === d.format));
  document.querySelectorAll('#sbeDeliverSize .pill-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.size === d.size));
  document.querySelectorAll('#sbeDeliverFinish .pill-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.finish === d.finish));
  const btn = sbeEl('sbeRenderBtn');
  if (btn && !SBE.rendering) {
    const short = ({ h264: '', hevc: 'HEVC', prores: 'ProRes' })[d.format];
    const size = d.size === 'native' ? '' : (d.size === '2160p' ? '4K' : '1080p');
    const fin = d.finish === 'none' ? '' : (d.finish === 'heavy_grain' ? 'heavy grain' : 'grain');
    const tag = [short, size, fin].filter(Boolean).join(' ');
    btn.textContent = tag ? ('Render · ' + tag) : 'Render';
    btn.title = 'Assemble the film as ' + ({ h264: 'H.264', hevc: 'HEVC', prores: 'ProRes 422 HQ' })[d.format]
      + (d.size === 'native' ? ', as cut' : (d.size === '2160p' ? ' at 4K' : ' at 1080p'))
      + '. Change it under the arrow.';
  }
}

async function sbeRenderFilm() {
  if (SBE.rendering) return;
  // THE RENDER READS THE SAVED FILE, so a save that did not land renders the
  // previous cut — a film that is silently not the one on screen, which is
  // the worst possible way to find out about a failing save.
  if (SBE.dirty && !(await sbeSave(true))) {
    phosToast('The timeline could not be saved, and the render builds from ' +
              'the saved file — it would make the previous cut. Fix the save ' +
              'first.', { kind: 'danger', duration: 8000 });
    return;
  }
  const holes = sbeHoles(SBE.clips);
  if (holes.length && !confirm(
      holes.length + ' hole(s) totalling ' +
      holes.reduce((a, g) => a + g.duration, 0).toFixed(2) + 's are still empty.\n\n' +
      'The assembler CONCATENATES — a hole closes and everything after it slides ' +
      'earlier, off the beat it was cut to.\n\nRender anyway?')) return;
  const btn = sbeEl('sbeRenderBtn');
  const prev = btn.textContent;
  SBE.rendering = true;
  btn.disabled = true;
  btn.textContent = 'Assembling…';
  sbeEl('sbeRenderNote').textContent =
    'One ffmpeg pass over ' + SBE.clips.length + ' clips. This takes as long as an encode takes.';
  const fd = new URLSearchParams();
  fd.set('id', SBE.id);
  const music = (sbeEl('sbeMusic').value || '').trim();
  if (music) { fd.set('music', music); fd.set('music_mode', sbeMusicMode()); }
  const dl = sbeDeliverGet();
  fd.set('format', dl.format);
  fd.set('size', dl.size);
  fd.set('finish', dl.finish);
  let r;
  try { r = await (await fetch('/storyboard/edit/render', { method: 'POST', body: fd })).json(); }
  catch (e) { r = { ok: false, error: String(e) }; }
  SBE.rendering = false;
  btn.disabled = false;
  btn.textContent = prev;
  if (!r.ok) {
    sbeEl('sbeRenderNote').textContent = '';
    phosToast(r.error || 'The film could not be assembled.', { kind: 'danger', duration: 8000 });
    return;
  }
  sbeEl('sbeRenderNote').textContent = (r.gaps_note || '') +
    ' Wrote ' + Math.round(sbeNum(r.duration)) + 's to ' + (r.path || '').split('/').pop();
  phosToast('Rendered ' + r.clips + ' clips · ' + Math.round(sbeNum(r.duration)) + 's' +
            (r.deliver && r.deliver.label ? ' · ' + r.deliver.label : '') +
            (r.deliver && r.deliver.format === 'prores'
              ? ' — a ProRes .mov: it opens in an NLE or QuickTime, not in this preview' : '') +
            (r.gaps_note ? ' — ' + r.gaps_note : ''),
            { kind: 'success', duration: (r.gaps_note || (r.deliver && r.deliver.format === 'prores')) ? 11000 : 6000 });
  // The render ENDS ON THE FILM. Before this the timeline wrote an mp4 into
  // mlx_outputs/storyboards/, printed one line of grey text under the button,
  // and left the user looking at the timeline wondering where the film went.
  // The Editor's document is not necessarily the storyboard's open board any
  // more, so the board is opened before the film screen is asked for it.
  const focus = (r.path || '').split('/').pop();
  if (typeof workflowSwitch === 'function') workflowSwitch('storyboard');
  if (SBE.id !== SB.id && typeof sbOpen === 'function') await sbOpen(SBE.id);
  sbFilmOpen({ focus: focus });
}

// ---------------------------------------------------------------------------
// EXPORT FOR AN NLE — the film as a project the next room can open
// ---------------------------------------------------------------------------
async function sbeExportNle() {
  try { _uiEvent('feature_used', {feature: 'editor_export'}); } catch (_) {}
  if (!SBE.open || !SBE.id) return;
  // Same rule as the render: the project is written from the saved file.
  if (SBE.dirty && !(await sbeSave(true))) {
    phosToast('The timeline could not be saved, and the project is written ' +
              'from the saved file — fix the save first.',
              { kind: 'danger', duration: 8000 });
    return;
  }
  const btn = sbeEl('sbeNleBtn');
  const prev = btn.textContent;
  btn.disabled = true;
  btn.textContent = 'Writing the project…';
  const fd = new URLSearchParams();
  fd.set('id', SBE.id);
  let r;
  try { r = await (await fetch('/storyboard/edit/export-nle', { method: 'POST', body: fd })).json(); }
  catch (e) { r = { ok: false, error: String(e) }; }
  btn.disabled = false;
  btn.textContent = prev;
  if (!r || !r.ok) {
    phosToast((r && r.error) || 'The project folder could not be written.',
              { kind: 'danger', duration: 8000 });
    return;
  }
  const folder = (r.dir || '').split('/').pop();
  // WHAT CAME OUT AND WHAT DID NOT, in the note rather than in a support
  // thread later: the audio is stems, and a slug is a gap the NLE draws black.
  sbeEl('sbeRenderNote').textContent =
    folder + ' · ' + r.clips + ' clips · open the .xml in Premiere or Resolve, ' +
    'run the _ae.jsx from After Effects. Sound comes out as STEMS — clip audio ' +
    'and the soundtrack on separate tracks, not the ducked mix.' +
    ((r.missing || []).length ? ' ' + r.missing.length + ' file(s) were gone and left out.' : '');
  phosToast('Project folder ready · ' + r.clips + ' clips, ' +
            (r.linked || 0) + ' hardlinked' +
            ((r.copied || 0) ? ', ' + r.copied + ' copied' : ''),
            { kind: 'success', duration: 8000 });
  // The folder, revealed. A path in a toast is a path somebody has to
  // retype; `open` on the server is the one honest "here it is".
  const rf = new URLSearchParams();
  rf.set('id', SBE.id);
  rf.set('what', 'project');
  try { await fetch('/storyboard/edit/reveal', { method: 'POST', body: rf }); }
  catch (e) {}
}

// ---------------------------------------------------------------------------
// TICK, KEYS, WIRING
// ---------------------------------------------------------------------------
// How long an unwritten change may sit before the editor stops calling it
// "unsaved changes" and starts calling it what it is. Three ticks: long enough
// that a slow save is not slandered, short enough that nobody gets twenty
// minutes into a cut that is not being stored.
const SBE_SAVE_GRACE_MS = 12000;

async function sbeTick() {
  if (!SBE.open || document.hidden) return;
  // THE WATCHDOG, AND IT RUNS BEFORE ANYTHING ELSE IN THE TICK. Every earlier
  // path out of this function was also a path past the one check that notices
  // work is not reaching the disk.
  //
  // Two jobs. First, nothing unsaved is ever left with no save on its way:
  // dirty, not saving, no timer pending is a dropped save, and re-queueing is
  // free. Second, if the oldest unwritten change is older than the grace, the
  // alarm goes up — whatever the reason, whoever swallowed it.
  // UNSAVED IS A LEGITIMATE STATE NOW — it means the user has not pressed
  // Save, which is his call. What is NOT legitimate is unsaved AND unbacked:
  // that is work living in one browser tab and nowhere else. So the watchdog
  // guards the BACKUP, not the save.
  sbePaintProtected();
  if (SBE.dirty && !SBE.conflict && SBE.id) {
    if (!SBE.backingUp && !SBE.saveTimer) sbeQueueSave();
    if (SBE.dirtyAt && Date.now() - SBE.dirtyAt > SBE_SAVE_GRACE_MS
        && SBE.backedUpAt < SBE.dirtyAt) {
      sbeSaveAlarm('these changes have not been backed up for ' +
                   Math.round((Date.now() - SBE.dirtyAt) / 1000) +
                   ' seconds — press Save to store them');
    }
  }
  const job = SBE.prepare || {};
  if (job.state === 'running') {
    try {
      const r = await (await fetch('/storyboard/edit/status?id=' +
                                   encodeURIComponent(SBE.id))).json();
      const was = job.state;
      SBE.prepare = r.prepare || {};
      sbePaintChrome();
      if (was === 'running' && SBE.prepare.state !== 'running') {
        SBE.peaksFor = '';
        await sbeLoad(true);
      }
    } catch (e) {}
    return;
  }
  // A shot generated into a hole is rendering somewhere in the panel's one
  // queue. Poll for it — but never while the user has unsaved work in flight,
  // because a reload would be a reload over their arrangement.
  if (SBE.awaitingClip && !SBE.dirty && !SBE.drag && !SBE.saving) {
    const before = (SBE.unplaced || []).length;
    await sbeLoad(true);
    if ((SBE.unplaced || []).length > before) {
      SBE.awaitingClip = 0;
      phosToast('The new shot has landed — place it on the timeline.', { kind: 'success' });
    } else if (Date.now() - SBE.awaitingClip > 45 * 60 * 1000) {
      SBE.awaitingClip = 0;
    }
  }
}

document.addEventListener('keydown', (ev) => {
  if (!SBE.open || document.body.dataset.workflow !== 'editor') return;
  const t = ev.target;
  if (t && /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName || '')) return;
  if (document.querySelector('.modal-bg.show')) return;
  const step = 1 / sbeFps();
  if (ev.key === ' ') { ev.preventDefault(); sbeTogglePlay(); return; }
  if (ev.key === 'ArrowLeft') { ev.preventDefault(); sbeStop(); sbeSeek(SBE.playhead - (ev.shiftKey ? step * 10 : step)); return; }
  if (ev.key === 'ArrowRight') { ev.preventDefault(); sbeStop(); sbeSeek(SBE.playhead + (ev.shiftKey ? step * 10 : step)); return; }
  if ((ev.key === 'Delete' || ev.key === 'Backspace') && SBE.txSel) { ev.preventDefault(); sbeTxRemoveSel(); return; }
  if ((ev.key === 'Delete' || ev.key === 'Backspace') && SBE.ovSel && !SBE.sel) { ev.preventDefault(); sbeOvDeleteSel(); return; }
  if ((ev.key === 'Delete' || ev.key === 'Backspace') && ev.shiftKey) { ev.preventDefault(); sbeRippleSelected(); return; }
  if (ev.key === 'Delete' || ev.key === 'Backspace') { ev.preventDefault(); sbeLiftSelected(); return; }
  if (ev.key === 's' || ev.key === 'S') { ev.preventDefault(); sbeSplitHere(); return; }
  if ((ev.key === 'd' || ev.key === 'D') && !ev.metaKey && !ev.ctrlKey && SBE.sel) { ev.preventDefault(); sbeDuplicateSel(); return; }
  if ((ev.metaKey || ev.ctrlKey) && (ev.key === 'z' || ev.key === 'Z')) {
    ev.preventDefault();
    ev.shiftKey ? sbeRedo() : sbeUndo();
    return;
  }
  if (ev.key === 'm' || ev.key === 'M') {
    ev.preventDefault();
    SBE.muted ? sbeUnmuteFromRefusal() : sbeSetMute(true);
    return;
  }
  if (ev.key === 'Escape') {
    ev.preventDefault();
    // The panel first: Escape means "close the thing on top", and closing the
    // whole document out from under an open picker is a surprise. A menu is
    // the same argument one layer higher.
    if (sbePopAnyOpen()) { sbePopCloseAll(''); return; }
    const vers = document.getElementById('sbeVersions');
    if (vers && !vers.hidden) { sbeVersionsClose(); return; }
    sbeClose();
  }
  // RENDER HAS A KEY NOW that it is the only filled control on the screen.
  if ((ev.key === 'r' || ev.key === 'R') && (ev.metaKey || ev.ctrlKey)) {
    ev.preventDefault();
    sbeRenderFilm();
  }
});

// A menu closes when you click away from it. Capture phase, like the picker
// below, so the control underneath still gets its click.
document.addEventListener('click', sbePopGlobal, true);

// Click-away, the same shape the engine menu uses. Capture phase, so a click
// on a control underneath still lands after the panel is gone.
document.addEventListener('click', (ev) => {
  const vers = document.getElementById('sbeVersions');
  if (!vers || vers.hidden) return;
  if (ev.target.closest('#sbeVersions') || ev.target.closest('#sbeVersBtn')
      || ev.target.closest('#sbeKeepBtn')) return;
  sbeVersionsClose();
}, true);

(function sbeWire() {
  const track = document.getElementById('sbeTrack');
  if (!track) return;
  sbeSetMute(SBE.muted);   // the button must agree with the stored state on load
  track.addEventListener('pointerdown', sbeOnTrackDown);
  track.addEventListener('pointermove', sbeOnTrackMove);
  track.addEventListener('pointerup', sbeOnTrackUp);
  track.addEventListener('pointercancel', sbeOnTrackUp);
  // Pan and zoom over the whole scroller — ruler, waveform and track at once,
  // because they are one view of one film and panning one of them alone would
  // be a bug. `passive: false` or preventDefault is a no-op and the column
  // scrolls out from under the gesture.
  const box = document.getElementById('sbeScroll');
  if (box) box.addEventListener('wheel', sbeOnTlWheel, { passive: false });
  // A RESIZE OBSERVER ON THE COLUMN, not just window.onresize. Measured: a
  // viewport change from 1900x1000 to 1440x900 left both monitors 437px tall
  // in a 501px column — 16:9 boxes squashed to 1.15:1 by `max-width` because
  // the window event never reached sbePaint. The column is the box the budget
  // is actually about, and it changes for reasons the window never hears:
  // the bottom pane opening, a scrollbar arriving, the pool being folded.
  // Coalesced on a TIMER, not on requestAnimationFrame. rAF does not run in
  // a tab the compositor has decided is not visible — measured in a headless
  // preview pane, where the callback simply never fired and the monitors kept
  // the previous window's height — and a resize is exactly the moment a
  // background tab is about to become the foreground one. sbePaint cannot
  // change this box's size, so this cannot feed itself.
  const colBox = document.getElementById('edStage');
  if (colBox && 'ResizeObserver' in window) {
    let pending = 0;
    new ResizeObserver(() => {
      if (pending) return;
      pending = setTimeout(() => { pending = 0; if (SBE.open) sbePaint(); }, 0);
    }).observe(colBox);
  }
  // THE MUSIC LANE. Listeners on the LANE and not on the block: the block is
  // repainted (and its class list rewritten) on every frame of the drag, and a
  // pointer captured by an element the paint replaces is a drag that dies
  // halfway. The lane outlives the gesture.
  const lane = document.getElementById('sbeMusicLane');
  if (lane) {
    lane.addEventListener('pointerdown', (ev) => {
      if (!ev.target.closest('#sbeMusicClip')) return;
      sbeStop();
      sbeOnMusicDown(ev);
    });
    lane.addEventListener('pointermove', sbeOnMusicMove);
    lane.addEventListener('pointerup', sbeOnMusicUp);
    lane.addEventListener('pointercancel', sbeOnMusicUp);
    lane.addEventListener('dblclick', sbeOnMusicDbl);
    // The ghost belongs to the pointer, so it goes when the pointer does.
    lane.addEventListener('pointerleave', () => {
      if (SBE.kfGhost && SBE.kfGhost.id === '@music') {
        SBE.kfGhost = null;
        sbePaint();
      }
    });
  }
  const alane = document.getElementById('sbeAudioLane');
  if (alane) {
    alane.addEventListener('pointerdown', (ev) => { sbeStop(); sbeOnAudioDown(ev); });
    alane.addEventListener('dblclick', sbeOnAudioDbl);
    alane.addEventListener('pointermove', sbeOnAudioMove);
    alane.addEventListener('pointerup', sbeOnAudioUp);
    alane.addEventListener('pointercancel', sbeOnAudioUp);
    // The two halves of "hover teaches": the ghost follows the pointer while
    // it is near a level, and it goes when the pointer does.
    alane.addEventListener('pointerleave', sbeAudioGhostClear);
    alane.addEventListener('contextmenu', sbeOnAudioMenu);
  }
  const scrubbers = ['sbeRuler', 'sbeWave', 'sbeWaveNone'];
  for (const id of scrubbers) {
    const el = document.getElementById(id);
    if (!el) continue;
    el.addEventListener('pointerdown', (ev) => {
      sbeStop();
      el.setPointerCapture && el.setPointerCapture(ev.pointerId);
      el.dataset.scrub = '1';
      sbeSeek(sbeTimeFromEvent(ev, el));
    });
    el.addEventListener('pointermove', (ev) => {
      if (el.dataset.scrub !== '1') return;
      SBE.playhead = Math.max(0, Math.min(sbeFilmDuration(SBE.clips),
                                          sbeTimeFromEvent(ev, el)));
      sbePaintHead();
    });
    const end = (ev) => {
      if (el.dataset.scrub !== '1') return;
      el.dataset.scrub = '';
      sbeSeek(sbeTimeFromEvent(ev, el));
    };
    el.addEventListener('pointerup', end);
    el.addEventListener('pointercancel', end);
  }
  const tabs = document.getElementById('edPoolTabs');
  if (tabs) {
    tabs.addEventListener('click', (ev) => {
      const b = ev.target.closest('.pill-btn');
      if (b) edPoolSrc(b.dataset.src);
    });
  }
  const pass = document.getElementById('sbeGenPass');
  if (pass) {
    pass.addEventListener('click', (ev) => {
      const b = ev.target.closest('.pill-btn');
      if (!b) return;
      pass.querySelectorAll('.pill-btn').forEach(x => x.classList.toggle('active', x === b));
    });
  }
})();

document.getElementById('sbStyle') && document.getElementById('sbStyle')
  .addEventListener('input', () => {});

function workflowSwitch(name) {
  // 2026-05-17 — Characters is no longer its own top-level tab. The
  // chip strip is integrated into Manual (T2V). If we get a stale
  // 'characters' value from localStorage, snap to 'manual'.
  if (name === 'characters') name = 'manual';
  // Q4 tier uses the distilled A2V pipeline (no Q8 dev required).
  document.querySelectorAll('#workflowTabs button[data-workflow]')
    .forEach(b => b.classList.toggle('active', b.dataset.workflow === name));
  const manual = document.getElementById('genForm');
  const studio = document.getElementById('studioSection');
  const train = document.getElementById('trainSection');
  const audioTab = document.getElementById('audioSectionTab');
  const sbTab = document.getElementById('sbSectionTab');
  const edTab = document.getElementById('edSectionTab');
  const characters = document.getElementById('charactersSection');  // dead HTML; hide defensively
  // Set body data attribute so CSS can switch the layout per workflow.
  document.body.setAttribute('data-workflow', name);
  // All non-Manual panes start hidden; the active branch re-shows its own.
  if (studio) studio.classList.remove('show');
  if (train) train.classList.remove('show');
  if (audioTab) audioTab.style.display = 'none';
  if (sbTab) sbTab.style.display = 'none';
  if (edTab) edTab.style.display = 'none';
  if (characters) characters.classList.remove('show');
  // The board poller is the Storyboard tab's only timer and it stops on exit —
  // no new polling loop runs while the tab is closed.
  if (name !== 'storyboard') {
    if (typeof sbTeardown === 'function') sbTeardown();
    document.body.classList.remove('sb-full');
  }
  // LEAVING THE EDITOR IS NOT CLOSING THE DOCUMENT. The storyboard's own
  // teardown used to take the editor down with it, so glancing at the gallery
  // threw away the undo stack and the open timeline; you came back to the shot
  // list. Suspend stops the clock and the picture and keeps everything else —
  // the document closes when the user closes it, and only then.
  if (name !== 'editor' && typeof sbeSuspend === 'function') sbeSuspend();
  if (name === 'storyboard') {
    // Storyboard is a layer ABOVE the video modes, not one of them: it submits
    // a brief to a planner and its output is a plan, not a clip. Hence a
    // workflow tab rather than a 6th chip in #modeGroup.
    if (manual) manual.style.display = 'none';
    if (sbTab) sbTab.style.display = 'block';
    if (typeof sbInit === 'function') sbInit();
  } else if (name === 'studio') {
    // Studio is its own top-level tab now (was a mode chip inside
    // Manual). The setMode('image') logic still wires up the studio
    // pane + portals the LoRA picker into the studio composer; just
    // hide #genForm + show #studioSection on top of that.
    if (manual) manual.style.display = 'none';
    if (studio) studio.classList.add('show');
    if (typeof setMode === 'function') setMode('image');
  } else if (name === 'train') {
    // Train Character is its own workflow tier. Hide the manual render
    // form, show the train section, run its init.
    if (manual) manual.style.display = 'none';
    if (train) train.classList.add('show');
    if (typeof trainInit === 'function') trainInit();
  } else if (name === 'audio') {
    // Audio → Video — own workflow tab. Pure A2V (audio + prompt) and
    // Image + Audio (audio + still + prompt) both live here, behind a
    // single drop-zone + optional image picker. Routes to make_job
    // with mode='a2v'.
    if (manual) manual.style.display = 'none';
    if (audioTab) audioTab.style.display = 'block';
    if (typeof audioStudioInit === 'function') audioStudioInit();
  } else if (name === 'editor') {
    // The Editor. Engine-agnostic and board-agnostic: it opens the document
    // it had open last, and an empty timeline is a legitimate place to stand.
    if (manual) manual.style.display = 'none';
    if (edTab) edTab.style.display = 'flex';
    if (typeof edInit === 'function') edInit();
  } else {
    // Manual — show the video form, restore the previous video mode.
    if (manual) manual.style.display = '';
  }
  // The Ideogram edit canvas belongs to the Images workflow only — re-sync
  // so leaving suspends it (player/outputs restored) and returning to
  // Images with Ideogram active brings it back with its boxes intact.
  if (typeof ideoSyncStage === 'function') ideoSyncStage();
  // The engine switcher is a Video-form control (each ENGINES row names the
  // `surfaces` it belongs to). Offering a render engine while the user is in
  // Images or Train Character would be a choice that changes nothing, so the
  // switcher and its divider fold away with the form they belong to.
  if (typeof renderEngineSwitch === 'function') {
    try { renderEngineSwitch(); } catch (e) {}
  }
  try { localStorage.setItem('phos_workflow', name); } catch(e) {}
}

document.querySelectorAll('#workflowTabs button[data-workflow]').forEach(b => {
  b.addEventListener('click', () => workflowSwitch(b.dataset.workflow));
});


// Initial workflow tab restore from localStorage.
try {
  const saved = localStorage.getItem('phos_workflow');
  // 2026-05-17 — Characters tab removed; 'characters' now snaps to
  // 'manual' (workflowSwitch handles the alias). 'studio' is new.
  // 2026-05-18 — 'audio' is the Audio → Video workflow tab.
  // 2026-08-11 — 'storyboard' is the plan-a-film workflow tab. It MUST be in
  // this list or the tab simply never restores across a reload.
  // 2026-08-17 — 'editor' is the timeline's own tab. Same rule, same list.
  if (saved === 'studio' || saved === 'train' || saved === 'characters' ||
      saved === 'audio' || saved === 'storyboard' || saved === 'editor') {
    workflowSwitch(saved);
  }
  // Clear any stale agent-fullscreen flag from the removed chat surface.
  try { localStorage.removeItem('phos_agent_fullscreen'); } catch(e) {}
} catch(e) {}


// ---- published to the page --------------------------------------------------
// Inline handlers in the markup, the other files, and the repo's browser
// harnesses (scripts/measure_editor_layout.py) resolve these through the
// global scope; everything NOT listed here is private to this module.
Object.assign(globalThis, {
  sbeStripY, sbeStripGain, sbeEl, sbeNum,
  sbeRound, sbeFps, sbeLen, sbeGridGap,
  sbeAdoptGaps, sbeLayout, sbeFilmDuration, sbeMusicWindow,
  sbeMusicEdit, sbeMusicSnaps, sbeSnapToList, sbeClipAudio,
  sbeAudioField, sbeSetAudioLink, sbeAudioEdit, sbeAudioDrift,
  sbeAudioInSync, sbeClipMuted, sbeSetClipMute, sbeAudioIsThePicture,
  sbeDriftLabel, sbeSyncMark, sbeSyncCarry, sbeResyncAudio,
  sbeClipLen, sbeFx, sbeFadeOpacityAt, sbeSetFade,
  sbeWaveSlice, sbeWaveWant, sbeAfx, sbeLerpGain,
  sbeGainPoints, sbeGainAt, sbeAudioMix, sbeBedLen,
  sbeAudibleStrips, sbeDuckGainAt, sbeBedDuckPoints, sbeBedDuckSuppressed,
  sbeBedGainPoints, sbeBedGainAt, sbeMixWrite, sbeBedAfxWrite,
  sbeSetBedFade, sbeBedPointsWrite, sbeBedAddKeyframe, sbeBedMoveKeyframe,
  sbeBedDeleteKeyframe, sbeAfxWrite, sbeAddKeyframe, sbeMoveKeyframe,
  sbeDeleteKeyframe, sbeDeleteStrip, sbeSetAudioFade, sbeOvKind,
  sbeOvAt, sbeOvById, sbeOvFits, sbeOvMove,
  sbeOvTrim, sbeOvAdd, sbeOvDelete, sbeStripOwned,
  // Editor v2: speed on the clip, titles on the overlay lane, transitions on
  // the cut.
  sbeSpeed, sbeSetSpeed, sbeSpeedCommit,
  sbeOvText, sbeHexColour, sbeRgba, sbeOvTextCommit, sbeOvTextPreview, edAddTitle,
  sbeSectionBands, sbeShotForClip, sbeRetakeSel, sbeRetakeDismissed,
  sbeDeliverGet, sbeDeliverPick, sbeDeliverPaint, sbeDuplicate, sbeDuplicateSel,
  sbeFraming, sbeFramingIsNeutral, sbeSetFraming, sbeApplyPreviewFraming, sbeFramingPreview,
  sbeFramingCommit, sbeFramingReset,
  sbeRetakeKeep, sbeRetakeUse,
  sbeOvTextPlace, sbeBlurControl,
  sbeTxById, sbeTxAfter, sbeTxDuration, sbeTxSpare, sbeTxResolve, sbeTxEdges,
  sbeTxSet, sbeTxDelete, sbeTxPrune, sbeTxRepoint, sbeTxMutate, sbeTxCommit,
  sbeTxRemoveSel,
  sbePictureCarriesSound, sbeStripsAt, sbeClipAt, sbeHoles,
  sbeBeatGrid, sbeGridIsAGuess, sbeSnapTime, sbeById,
  sbeMoveTo, sbeTrim, sbeRippleDelete, sbeLiftDelete, sbeSplitAt,
  sbeNewId, sbePlaceUnplaced, sbeKind, sbeBright,
  sbeBrightnessCss, sbeSetBrightness, sbeDropIndex, sbeInsertAt,
  sbeReorderTo, sbeCleanClip, sbeSaveBody, sbeErrorsByClip,
  sbeDecodePeaks, sbeFmtTime, sbeZoomFitPps, sbeZoomFromSlider,
  sbeZoomToSlider, sbeZoomAnchor, sbeZoomScroll, sbeFollowScroll,
  sbeMonitorFit, sbeTlClamp, sbeLaneHeights, sbeTlPrefRead,
  sbeTlPrefWrite, edDoc, edRemember, edInit,
  edShowPicker, edOpenBoard, sbeOpen, sbeSuspend,
  sbeCloseDoc, sbeClose, sbeGoToBoard, sbeTeardown,
  sbeLoad, sbeAdopt, sbeFetchPeaks, sbeMusicEditPath,
  sbePaintProtected, sbeSetState, sbeSnapshot, sbeRestore,
  sbeOvMutate, sbeOvAddAt, sbeOvSetPath, sbeMutate,
  sbeUndo, sbeRedo, sbeQueueSave, sbeBackup,
  sbeSave, sbeSaveAlarm, sbeSaveAlarmClear, sbeSaveInner,
  sbeRenderErrors, sbeErrsToggle, sbeTakeTheirs, sbeForceSave,
  sbeSaveNow, sbeAgo, sbeVersionLine, sbeVersionsEl,
  sbeVersionsClose, sbeVersionsOpen, sbeVersionsLoad, sbeVersionsPaint,
  sbeDraftOp, sbeNameEnter, sbeNameMode, sbeDraftNew,
  sbeDraftRename, sbeKeepVersion, sbeDraftDelete, sbePaintDraft,
  sbePaintNotices, sbeNoticeOpen, sbeNoticeClick, sbeNoticeLater,
  sbePaintRecovery, sbeRecover, sbeDiscardBackup, sbeRestoreVersion,
  sbeSpan, sbePaint, sbeFitMonitors, sbeApplyTl,
  sbeStripH, sbeTlGrabDown, sbeTlGrabMove, sbeTlGrabUp,
  sbeTlSet, sbeTlReset, sbeTlGrabKey, sbePaintWave,
  sbeOnMusicMove, sbeOnMusicUp, sbeBedGainSlide, sbeBedGainCommit,
  sbeSetBedDuck, sbePaintTrack, sbeFadeMarks, sbeSyncBadge,
  sbePaintAudioLane, edPoolOverlay, sbeKeyedKeepOriginal, sbeKeyedDismiss,
  sbePaintOverlays, sbeOvPaint, sbeOnOvDown, sbeOnOvMove,
  sbeOnOvUp, sbeLvlHitPath, sbeStripWave, sbeStripEditable,
  sbeStripAt, sbeAudioGhost, sbeLevelClick, sbeOnAudioMenu,
  sbeAddPointAtPlayhead, sbeClearPoints, sbeKeysLegend, sbeAudioFadeMarks,
  sbeOnAudioDbl, sbeAudioFadeCommit, sbeOnAudioDown, sbeOnAudioMove,
  sbeToggleAudioLink, sbeDeleteStripSel, sbeToggleClipMute, sbeResyncSel,
  sbePaintHead, sbePopToggle, sbePopCloseAll, sbePopGlobal,
  sbePopAnyOpen, sbePaintKeys, sbePaintInspector, sbeBrightPreview,
  sbeFadeCommit, sbeBrightCommit, sbeFadePaint, sbeApplyPreviewFilter,
  sbePaintChrome, sbeOnTrackDown, sbeOnTrackMove, sbeDragFingerprint,
  sbeOnTrackUp, sbeSeek, sbeClipUrl, sbeLoadInto,
  sbeShowFrameAt, sbeSrcUrl, edPoolPreview, sbeSrcPlay,
  sbeSrcStop, sbeSrcToggle, sbeSrcAdd, sbePaintSource,
  sbeMusicMode, sbeSetMusicMode, edPoolSrc, edPoolUpload,
  edPoolUploadRow, edPoolRefresh, edPoolLoadFilms, edPoolPickFilm,
  edPoolLoadFilmShots, edPoolPaint, edPoolObserve, edPoolAdd,
  edPoolDragStart, edPoolDragMove, edPoolDragEnd, edAddSlug,
  sbePaintRelink, sbeRelink, edPoolFocus, sbeTogglePlay,
  sbeSetMute, sbeUnmuteFromRefusal, sbePlay, sbeStop,
  sbeFrame, sbeMusicPlay, sbeMusicSync, sbeStripSync,
  sbeStripStop, sbeZoomMin, sbeZoomTo, sbeZoom,
  sbeZoomSlide, sbeOnTlWheel, sbePrepare, sbePrepareCancel,
  sbeAuto, sbeGenClose, sbeGenSubmit, sbeRenderFilm,
  sbeExportNle, sbeTick, workflowSwitch,
  // inline-handler targets: generated markup resolves these through the
  // global scope (the v4.9.0 regression, PR #69)
  sbeLiftSelected, sbeOvDeleteSel, sbeOvFadeCommit, sbePlace,
  sbeRippleSelected, sbeToggleLock,
});
