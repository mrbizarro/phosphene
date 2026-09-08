# Two characters talking in LTX — what the model wants (research, 2026-09-07)

Eight renders of "two aliens at dinner" in one day taught the panel a lot the hard
way. Before rendering a ninth, this is what Lightricks' own guides, the LTX-2 model
card, the community workflows and the open issues say about dialogue, multiple
characters and continuity — and the recipe that follows from it.

## 1. Lip-sync with several faces in frame is an open problem, not a prompt bug

* Lightricks' ComfyUI issue #395 ("How to control who is speaking… when there are
  multiple characters in the image?") describes exactly what the owner heard:
  "Character A or B starts lip syncing randomly to parts of the dialogue… very
  often they both appear to talk the same words simultaneously." It is **open,
  with no maintainer answer**.
* The community's working answers are all the same shape: **one speaking face per
  generation** (render one character at a time and composite), masks with
  external audio, or pre-recorded dialogue. "Insisting" in the prompt who talks
  ("the woman in the red dress is talking, the man is silently listening") helps
  sometimes and fails often — which is what our v2/v3 measured.
* The audio guide's own advice: **one speaker per line, short lines**, the
  speaker and delivery written before the quote ("she says in a calm, seasoned
  voice: '…'"), ~9 s of speech at a natural pace, the word "lip sync" in the
  prompt, a well-lit front-facing face, and "let the face arrive first" — a line
  that starts on frame one gives the sync nothing to lock onto.

**So:** a conversation is written so that *only the speaker's face is on screen
while a line plays*. The panel's One Shot staging rule (a beat names only who
should be in the picture) is the same idea; it is just the wrong container.

## 2. The right container is a multi-shot generation, not a one shot

LTX-2.5's headline feature is **native multi-shot**: several connected shots with
explicit cuts inside ONE prompt and ONE generation, with the model keeping
"character identity, environment, lighting and voice" across the cuts because it
reasons about the whole sequence at once. Lightricks' 2.5 prompt guide:

* "Use a single continuous take when you want unbroken camera motion, intimate
  performance, or dialogue that must stay lip-synced in one framing. … When a
  scene involves dialogue, multiple beats, or precise timing, write it in a
  screenplay style."
* "Prefer 2–4 shots in one generation." At every cut: **name the edit in prose**
  ("a hard cut to…"), **re-establish the new shot** (scale, angle, who is in
  frame, light), **re-identify returning characters with the same descriptors,
  never a pronoun**, and **say what the sound does** ("the room tone continues
  across the cut", "the dialogue drops; only wind remains").
* Lock what must persist in words: "Preserve her face, red coat, blue-magenta
  lighting… across every shot."

A two-person conversation is therefore **shot / reverse shot inside one
generation**: a medium shot of A saying its line, *a hard cut to* a medium shot of
B at the same table in the same light answering, the room tone continuing. One
face per shot, so nothing lip-syncs in unison; one generation, so it is one
table, one pair, one voice each. The "two different tables, the character
changes in the middle" of the long-table take was the cost of stitching three
separate generations of a scene the model was never allowed to see whole.

## 3. Continuity across generations, when a scene must outlast one

* Community extension workflows (RuneXX, Kijai, the IAMCCS extension nodes) feed
  the **last ~3 seconds (73 frames)** of the previous clip as context — not one
  frame — colour-match the new segment to the old, blend the audio, keep segments
  10–20 s, and expect drift after ~60 s. "Incorrect slices can result in weird
  seam timing, sudden resets, or continuity breaks."
* The panel's One Shot hands off ONE frame. That is why every join stalls and
  restarts (measured this morning) and why a scene drifts across parts. The
  vendored `RetakePipeline.extend_from_video` conditions on a source clip; fed a
  73-frame tail, it is the community method. The earlier "windows chain" that
  went to mush used a 121-frame tail on Q4 — a different experiment.
* The vendored fork also carries **Prompt Relay** (`--segment TEXT [LEN]`): each
  segment's words are gated to a slice of the timeline inside one generation.
  It cannot stop a visible mouth from moving, but it can keep speaker B's line
  from bleeding into speaker A's seconds. The panel does not expose it yet.

## 4. The recipe for the aliens (and any two-hander)

One generation per exchange, 10 s (241 frames) at High 720p on this Mac, written
as a screenplay-style multi-shot prompt:

```
Night, a candle-lit dining room. A medium shot of a tall thin alien in a deep
violet robe, pale grey-green skin, seated at a small round table, warm
candlelight on its face, alone in frame. It looks up and says slowly, in a deep
resonant voice: "The humans split the atom before they learned to share bread."
A hard cut to a medium shot, same table, same candlelight, of a short round
alien in a plain grey robe, darker olive skin, wide head, alone in frame; the
room tone continues across the cut. It answers quickly, in a thin high voice:
"Every species we ever watched said that about its neighbours." Preserve both
aliens' faces, robes and the candlelit room across every shot. Lip sync.
```

Rules baked into it: one speaker per shot and per line; the speaker described
before the quote, every time, with the same words; the cut named; the sound
across the cut stated; the "preserve" sentence; "lip sync"; the face arrives
before the line. A 30 s conversation is three such generations, joined on cuts
(a cut hides a seam; a one shot cannot), each re-describing both aliens with the
same descriptors so the model re-creates the same pair. If a scene must be one
unbroken take longer than 10 s, extend with a 73-frame tail as context, not a
frame (§3) — that path is unbuilt in the panel today.

## 4b. Two faults measured on a 90 s monologue take, and their fixes

* **Voice over a closed mouth** at the end of parts and on the first frame: 10–14-word
  lines in 5 s beats. Speech runs ~2.5 words/s, so the audio overran the beat and was
  cut at the part boundary. Fix: ≤7 words per beat, the silence after the line written,
  a silent look before the first line (the panel's own duration rule; §4 already
  assumed it).
* **"Cuts blended with an overexposure effect"**: an image-to-video part drifts in
  exposure inside itself (mean luma 137 → 178 across ten seconds) and the next part
  re-levels one frame after its anchored first frame (178 → 157), so a boundary colour
  match sees nothing to fix. Fix: `scripts/join_smooth.py --hold-exposure` pins every
  frame's mean luma to the take's first second (smoothed over 12 frames) — flat at 159
  across the same 30 s that swung 126–175 — plus the seam colour fade and a 120 ms audio
  crossfade. Valid only when the light is constant by contract, which a one shot's light
  lock makes true; a scene that is meant to darken must not use it.

## 4c. A talking character across parts: hand off on a talking frame, gate the sync

* A continuation part anchored on a **closed, silent mouth** will not lip-sync the
  next line (measured: mouth-opening-vs-voice r −0.15 / +0.20; the community's
  extension note says the same). Anchored on a frame from **mid-word** it syncs
  (+0.47). One Shot's `take_handoff=speech` trims each part where its line ends
  (`take_speech_end`, 0.4 s pad) and anchors the next part there; each part's
  first beat opens with a silent look so the sync has a moment to lock.
* Even a first part is a roll: identical settings gave +0.41 and −0.27 on two
  lines. So every spoken part is **measured** (`take_lipsync_score`, the same
  measure as `scripts/lipsync_check.py`) and **retaken once** under 0.20 with a
  fresh seed; the better clip is kept.
* One real line (≤18 words) per 10 s part, never a line per 5 s beat; no
  cut/edit/shot/scene words in the dialogue; a trained face renders on Q8 Pro.

## 5. Proof (rendered 2026-09-07, 10 s, High 720p, seed 4242)

The §4 prompt, submitted as one ordinary text-to-video job (no take, no
handoff): 0–5 s the tall alien alone at the candle-lit table saying its line;
a clean hard cut at 5 s; 5–10 s the short alien alone at the same table
answering, brightly lit, expression readable. One face per shot, one room, one
pair — the first render of the day where nothing could speak in unison and
nothing changed tables. `mlx_outputs/ltx_multishot_dialogue_aliens_10s_high720p.mp4`.

## Sources

* Lightricks, *LTX-2.5 prompt guide* — https://ltx.io/blog/ltx-2-5-prompt-guide
* Lightricks, *Prompting guide for LTX-2* — https://ltx.io/blog/prompting-guide-for-ltx-2
* Lightricks/LTX-2 README and the LTX-2 model card (structure, "frame count 8k+1")
* Lightricks/ComfyUI-LTXVideo issue #395 (multi-character lip-sync, open) —
  https://github.com/Lightricks/ComfyUI-LTXVideo/issues/395
* RuneXX/LTX-2.3-Workflows discussions #116 (dual character) and #131 (multi-extend:
  3 s of context frames, colour match, 20 s groups, ~60 s before drift)
* Kijai/LTXV2_comfy discussion #36 (extend any video: 73-frame minimum overlap)
* fal.ai, *How to use LTX-2.5* (multi-shot: name the edit, re-establish, re-identify,
  state the sound; ≤3 shots; let the face arrive before the line)
* ltx23.org, *LTX 2.5 release guide* (native multi-shot, "Preserve … across every shot")
* ltx.io/blog *How to add AI voiceovers & dialogue* (one speaker per line, short lines)
