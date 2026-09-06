# The Editor's effects model

Status: **foundation**, 2026-08-20. Sibling to `docs/EDITOR_SAVE_MODEL.md`.

## The ruling

> "If we are doing that, I think we probably need to better organize the
> functionalities. Just set the base to have effects somewhere, and then we
> will find a way to better integrate them." — and, on how:
> "Tightly integrate it. Don't do it in a wrong way; do it in the right way."

So: a base first, features hanging off it. Not six patches.

## The rule that decides what ships

**An effect that cannot be honestly expressed in all three outputs does not
ship.** The three are:

| output | what it is |
|---|---|
| preview | the `<video>` on the stage, the strip player, the music bed |
| render | the one ffmpeg filtergraph `_sb_film_filtergraph` builds |
| export | FCP7 XML (Premiere/Resolve) and the After Effects script |

This is not a style note. `mute` set the standard: it is expressed as **the
absence of an audio lane** in the render — no volume filter, no mixer, nothing
that could sum — as `<enabled>FALSE</enabled>` on the FCP7 audio clipitem
(disabled, not deleted, so the editor downstream can see the decision and undo
it), and as `lay.audioEnabled = false` in AE. Each output says the same thing
in its own idiom rather than one output faking another.

An effect that would need to be baked into pixels to leave the panel, or that
only one of the three can carry, is a worse feature than no feature: it makes
the preview lie.

## Where an effect lives

`clip.fx` is the home. Additive, absent means no effect, and it is written
only when it says something — the same neutral-is-absent rule `adjust`, the
music trims and `mute` already follow, so a timeline nobody has touched is
byte-identical to one from before effects existed.

```jsonc
"fx": { "fade_in": 0.5, "fade_out": 0.75 }
```

`EDIT_VERSION` does not move. The migration rule this repo has used since
`clip.audio` holds: a field whose absence means the old behaviour needs no
version bump, because every document ever written is already valid under it.

### The one legacy citizen

Brightness predates this model and lives at `clip.adjust.brightness`. It is
**not** moved into `fx`. A label is not worth a data migration — the same
reasoning that kept `film_start` when the user-facing noun became "sequence".

What changes is that there is now **one accessor**, `clip_effects(clip)`,
returning `{fade_in, fade_out, brightness}` whatever the storage. Every output
reads that and nothing else, so "where is it stored" stops being a question
any consumer has to answer, and the next effect chooses its storage on its own
merits without a fourth code path appearing.

The inspector presents brightness in the **Effects** section beside the fades,
because that is where a person looks for it. Presentation follows the model;
storage follows history.

## The inspector has sections now

The inspector grew one control at a time and was a flat run of buttons with a
brightness slider floating in the middle. It now has three:

* **Clip** — what this thing is: source window, film slot, provenance, lock,
  ripple delete.
* **Sound** — link/unlink, resync, mute. Everything about the strip.
* **Effects** — brightness, fade in, fade out. The home the ruling asked for,
  and the place the next effect lands without a decision.

## Fades — the first citizen

**Preview**: an honest opacity ramp on the stage element, computed from the
playhead against the clip's own film window. Not a CSS transition — a value
per frame, so scrubbing shows the true opacity at that second.

**Render**: ffmpeg `fade=t=in:st=…:d=…` and `fade=t=out:st=…:d=…` on the
segment's own timeline, after the brightness term and before the format
conversion. Fades are per-segment, so they compose with trims without knowing
about them.

**Export**: opacity keyframes. FCP7 gets a `<filter>` carrying the standard
Opacity effect with four keyframes (0 → 100 → 100 → 0 at the right frames);
AE gets `lay.opacity.setValueAtTime(...)`. Both are native, editable, and
undoable on the far side — nobody receives baked pixels.

### The clamp

Each fade is ≥ 0, and `fade_in + fade_out ≤ the clip's length`. Two fades that
crossed would ask ffmpeg for an opacity that is two things at once, and would
give the NLEs keyframes out of order. The clamp is applied in the model, so
all three outputs get the same already-legal numbers.

## The mix — and the second author it removes

Status: **built**, 2026-08-21.

The renderer was a second author of the soundtrack's level, and an invisible
one. Under `music.mode == "under"` the ffmpeg graph held every bed at a
hard-coded `volume=0.20` (−14 dB) and then pushed it down a further ~11 dB
through a `sidechaincompress` keyed on the clips' own audio. Neither number
was in any document, on any screen, or in the preview:

> "when you render it, there are some weird manipulations… the volume of the
> music goes low when the dialogue appears. The mix is weird."

The duck defended itself in its own comment: it kept lines intelligible
"without anybody automating a volume curve by hand". True when it was written,
false now — the bed and the strips both carry fades, keyframes and level
lines, so the renderer was overriding numbers the user had explicitly
authored. That is the save model's two-writers-no-rule, one floor down.

### Where it lives

`audio.mix`, two fields, both neutral-is-absent:

```jsonc
"mix": { "bed_gain": 0.2, "duck": true }
```

`audio_mix()` is the one accessor. `bed_gain_points()` is the ONE CURVE —
`[[t, gain], …]` on the bed's own clock — and the preview, the ffmpeg render
and the exports read that and nothing else.

### The bed's clock, and what happens when the track will not say

`bed_length()` is that clock: the seconds of the bed that actually PLAY, so a
fade dragged onto the block's corner means the corner it was dragged onto.
It resolves in one order, **and both implementations resolve it the same way**:

1. `trim_end` — an out-point the user set.
2. `audio.duration` — the track's own length, as the document records it.
3. **the FILM** — what is left of `edit_duration()` after the bed starts.

Rule 3 is the fix for the one shape the first mix pass left behind. A bed with
no `duration` and no `trim_end` — reachable from `/storyboard/edit/save`, and
written by `_sbe_auto_edit` itself when both the peaks and the probe fail —
made `bed_length()` return 0, which made the curve EMPTY, which means **no
filter**, which is the bed at FULL LEVEL over the dialogue. Measured
end-to-end on a real ffmpeg render with an 8 kHz bed and a 300 Hz "line":
**+20.5 dB above what the browser drew**, in the direction nobody wants.
Silence-by-empty-curve is the one behaviour nobody asked for and it is the loud
one. The honest reading of a bed of unknown length is that it plays under the
film — the renderer trims the mix to the film anyway — so that is what both
sides now say. Same document, same rig, after: **−0.08 dB.**

**The peaks probe is NOT in that chain, deliberately.** `peaks.json` is a
picture of the file and the renderer has never read it; a bed length taken from
it is a number only the browser can compute, and a gain only one side can
compute is the invisible-second-author defect this whole section exists to
remove. `sbeBedLen(audio, filmLen)` therefore takes the FILM's clock, not the
probe's duration, and is `storyboard_editor.bed_length(audio, film_len)` term
for term.

`test_editor_mix.TheTwoImplementationsAgreeOverEveryShapeOfDocument` is the
gate: real client JS in node against real Python, over a TABLE of documents —
no duration, no trim_end, bed longer than the film, bed shorter than the film,
authored envelopes, duck on and off, muted clips, clips with no audio, J-cuts,
closed windows — asserting the curves are identical POINT FOR POINT and that
the ffmpeg expression, evaluated, is the curve the browser drew. The suite that
missed the divergence compared the five constants and four documents; a mix
that exists as two implementations needs a table, not a spot check.

### The duck is a curve, not a compressor

This is the load-bearing change. A compressor's output is a function of
samples the browser never sees, so it could never be a value in a document and
the preview could never play it. `bed_duck_points()` expresses the same
decision as breakpoints derived from the TIMELINE: the bed steps back wherever
a clip's own sound is playing — `audible_strips()`, which skips muted clips and
clips with no audio track — by the measured **11.4 dB** (linear 0.269), in
5 ms, back up over 400 ms. Windows closer together than one release are merged,
so the bed does not scramble back up between two lines.

Measured on a real render (a 120 Hz bed under a 900 Hz "line", band-split by
projection): the film's bed sits at **0.4995** where the document says 0.5000,
and the duck measures **−11.45 dB** against a model that says −11.40.

### The precedence rule

Three terms, and **never two curves on one level**:

| term | when it applies |
|---|---|
| `bed_gain` | ALWAYS. A scalar fader, not a curve — multiplying an envelope by a fader is what a fader is. |
| `audio.afx` (authored) | ALWAYS, when it exists. |
| the auto-duck | ONLY when there is no authored envelope. |

A person who has drawn the bed's level has said what the bed does. An
automatic curve that then moved it would be the renderer disagreeing with them
again. `bed_duck_suppressed()` answers this, and the A2 head **says so** —
*"off — your own level line is driving the bed"* — because a ticked box quietly
doing nothing is the silent guard this editor keeps paying for.

### Defaults, and the one-time heal

New documents get the honest default: **bed 1.0, no duck.** Documents written
before the controls existed are stamped, on read, with what the renderer was
already doing to them — `{bed_gain: 0.2, duck: true}` — so a film the owner has
already approved keeps the levels it was approved at, the difference being that
they are now on screen. `replace` documents are left completely alone: that
path never touched the bed, so stamping one would invent an attenuation it
never had.

The stamp carries `mix_repair`, unlike the sub-frame gap heal which deliberately
has no marker. It needs one because it is a *decision*, not arithmetic:
`normalise_edit` drops a mix that is back at the default, so without the marker
a user who switched the duck off would have the next read helpfully put it
back — the rival author, wearing a repair's clothes.

### The one term that is NOT in the document

`MIX_CEILING` — the `asoftclip` knee on the summed mix. It is a safety
limiter, not an artistic choice: `amix` runs `normalize=0`, so nothing else
protects the sum, and hot dialogue over a bed peaked the first under-mix film
at 1.31 with 1341 hard-clipped samples and said nothing.

It cannot honestly be in the model, because it acts on the SUM of two signals
and a browser playing two `<audio>` elements never computes that sum — a
preview could only fake it. So instead of faking it, **the render measures what
it did**: `mix_peak`, `mix_ceiling` and `mix_limited` ride on the film's facts
and its sidecar. A net that has acted says so, which is the rule the save model
reduces to.

### The known limit

The assembler CONCATENATES, so a hole in the timeline closes and everything
after it slides earlier. The duck's windows are on the EDIT's clock — the same
clock the bed's own `adelay` placement already uses — so on a film with gaps
the bed and its duck stay consistent with each other and with the preview,
and both are displaced together by exactly the amount `gaps_note` already
discloses. Fixing that means fixing the concatenation, not the mix.

## Editor v2 — speed, titles, transitions (2026-09-05)

Three more citizens of the same model, each expressed in all three outputs
or honestly declared where one cannot carry it.

### Speed — on the clip

`clip.speed`, 0.25–4.0, video only, **absent is 1x, never automatic** (the
owner's verdict on a slowed shot that read as an accident was "too slow-mo").
`clip_speed()` is the accessor; `sbeLen` / `clip_length` divide by it, so the
slot is `(end - start) / speed` and everything downstream that measures the
film reads the played length. **Preview**: `<video>.playbackRate`, and the strip
player's too. **Render**: `setpts=(PTS-STARTPTS)/speed` on the picture, `atempo`
chained past ffmpeg's 0.5–2.0 window on the sound, the same `apad,atrim` tail
to the played length. **Export**: carried as `speed` on the row; the FCP7 XML's
in/out and start/end already disagree by that ratio, which is how an importer
infers a speed change — no explicit Time Remap effect is written.

**The envelope's clock does not move.** `afx` fades and points are seconds of
the strip AS PLAYED; `audio_gain_points()` documents why (the `volume` term
runs after `atempo`, and a fade of "1 s" was typed in film seconds). A keyframe
at 2 s is at 2 s of the strip after a retime; the strip is what changed length.

### Titles — on the overlay lane

An overlay with `kind: "text"` — a card the render DRAWS. `overlay_text()`
returns `{text, style}` clamped, `style` is written only where it differs from
`TEXT_STYLE_DEFAULTS`. **Preview**: a DOM element on the stage at the same
anchor (`x`/`y` fractions of the frame) and the same size rule (`font_size` at
1080 high, scaled to the stage). **Render**: a frame-sized RGBA PNG rasterised
with Pillow from an explicitly resolved font FILE, verified BEFORE ffmpeg is
built, fed through the very overlay chain an uploaded card takes — so fades,
z-order and the one-lane rule come free. Not `drawtext`: that filter is a
build option and the Homebrew ffmpeg this panel resolves on the author's own
machine does not carry it. **Export**: not carried (a title has no path); the
lane skips it and says nothing false.

### Transitions — on the cut

The hard one, and the one the picture-lane rule forbade as an overlap. A
transition is a **typed object that owns a boundary** (`transitions[]`,
`after_clip`), never a second picture in the same second: the clips' slots do
not move, the film is exactly as long as the timeline, and the render gets its
overlap from SOURCE HANDLES — half the duration of extra tail past the outgoing
out-point, half of extra head before the incoming in-point, centred on the cut,
the way an NLE builds one. No handles, no transition: `transition_no_handles`
names the side and the shortfall, in the validator and again at `render`. Every
transition code is an error; `WARNING_CODES` is untouched.

**Preview**: the stage has ONE `<video>`, so a dissolve is previewed as a ramp
through black on both sides of the cut (exact for `fade_black`, an honest
approximation for `dissolve`, and the inspector says so). **Render**: the
picture concat is split at the boundary into runs and re-joined with `xfade`
(`fade` / `fadeblack`); the sound takes the lane path it already had for J-cuts
and is byte-for-byte unchanged. **Export**: not carried yet — the cut is
exported as a butt join.

**Where it lives on screen**: the boundary. A small mark on every cut, a band
the length of the transition once one exists, the inspector to set kind and
length. No lane, no panel.

### Framing — a zoom and a reframe, per clip

`clip.frame = {zoom, x, y}`, zoom 1–3, the window centred at the fraction
(x, y) of the source; `clip_frame()` is the accessor, neutral is absent, a
slug has nothing to reframe. **Preview**: the stage layer scaled about the
same anchor (CSS, approximate — the render is exact). **Render**: `crop` of
the source's own pixels, before `fps`/`scale`, expressed with `iw`/`ih` so
the same string serves a 640-wide draft and a 1280-wide delivery, even-sized
for 4:2:0. **Export**: FCP7 Basic Motion (scale %, centre = `zoom·(0.5−x)`
frame widths) and AE scale + position — the decision, not baked pixels.

### Markers — deliberately not built

They are for hour-long timelines; ours are ~90 s and cut to a beat grid the
panel computes. Adding them would be copying without a reason.

## What is deliberately not built yet

* **Overlay lane** (his headline want: an endcard PNG over the sky). It is the
  next merge, and it uses this model — an overlay item carries `fx` and gets
  its fades from the same accessor and the same three expressions.
* ~~**Audio fades with keyframes**, waveforms on clip strips~~ — shipped, and
  the music bed has them now too: the same `.sbe-fade-band`, the same
  grip-derived clearances, the same 22×22 handles and the same level line with
  points. Its rows are in `scripts/measure_editor_layout.py`'s table, so
  "big enough to hit" and "nothing else claims these pixels" are asserted at
  the lane's floor and its ceiling.

The foundation is here so those land as features rather than as three more
scattered controls.
