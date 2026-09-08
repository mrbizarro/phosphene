# Writing for Phosphene — the short version

Copy this whole page into your own assistant (ChatGPT, Claude, Codex) and ask it to write your beats. The Storyboard planner already follows these rules on its own; this page is for when you write by hand.

## One Shot (a clip longer than one pass)

- **One beat = five seconds = one thing that happens.** Lead with the movement: what the subject does, what the camera does, what enters the frame.
- **A scene change gets a beat of its own.** Going through a door, into a room, onto a bridge: give it its own five seconds, and put the *approach* in the beat before it. A scene change hidden at the end of a busy beat is the one the model skips.
- **A reveal gets a beat of its own.** "The camera comes round to his face" works when it is the whole beat, not the tail of a beat that also carries the table, the wife and the hat.
- **Settle before the turn.** The beat before a scene change or a reveal should slow down; the model needs a moment to land before it can pivot.
- **Repeat the invariants every beat.** The subject's look, the camera position, the palette, the sound. The engine only remembers the last frame; the words carry the rest. Phosphene prepends your style block to every beat for you.
- **Something new every second.** A near miss, an object entering, a light change. A beat that describes a mood and no event reads as a still.
- **One short line per beat, and write the silence.** Speech runs about 2.5 words a second, so a 5-second beat holds one line of at most seven words, with a breath before it and the silence after it written out ("then he is silent, his mouth settles closed, and only the wind is heard"). A 12-word line spills past the beat and past the part boundary as voice over a closed mouth (measured 2026-09-07). Open the take with a silent look before the first line — a line on frame one gives the lip sync nothing to lock onto.
- **Lock the light.** Say the time of day and the weather once, in the first beat, and never imply another hour later ("neon comes on", "dawn breaks"). Phosphene appends a continuity sentence to every beat and measures each part; a part whose light drifts is retaken once on its own.
- **One camera move for the whole shot, written once, in the Camera line.** Direction and speed: "a slow, steady clockwise arc around the table at eye level, wide to close". The engine carries the picture across the join between parts, never the motion — a part that starts from a still picks its own move, and the seam shows as a camera that changes direction or stops and restarts. Phosphene puts your Camera line first in every part ("the camera is already moving — … — and continues in the same direction at the same steady speed"). Keep the beats' own camera words consistent with it: no "the camera settles", "swings round" or "pulls back" halfway unless that IS the move for the whole shot. One slow, constant move reads as one shot; three moves read as three.
- **Sound is part of the shot.** Say what we hear: footsteps, a band from a doorway, rain on an awning. Without sound cues the audio is near-silent ambience.

## Start from a still

- Design frame one as an image first (Phosphene makes one for you, or bring your own). Composition, palette and character are decided on an image that takes seconds, not a clip that takes an hour.
- Match the still's aspect to the shot canvas; the panel picks the nearest.

## What not to write

- No "fast forward", "time-lapse" or "montage" in a One Shot: the model cuts.
- No two subjects at once unless one of them is the point of the beat.
- **Two characters talking: read [`docs/LTX_DIALOGUE.md`](LTX_DIALOGUE.md) first.** The model's own answer is a multi-shot generation (shot / reverse shot inside ONE prompt, one speaking face per shot, the cut named, the sound stated, the characters re-identified with the same words), not a one shot stitched from parts.
- **The two-speaker staging that works (rendered 2026-09-07):** a long table with one speaker at each end and ONE slow lateral track along it. Speaker A alone in frame for its lines (a beat that names only A), a silent beat of empty table as A slides out (names nobody), speaker B entering alone at the far end for its answer (the first beat that names B). Never both in frame while either speaks. A profile two-shot, an orbit, or "over B's shoulder" all put two mouths in frame or add a body.
- **A beat names only who should be in the picture.** Whoever a beat's text names is composed into that beat's frame — "only A is in the frame" does not remove a B the same text describes, and "over B's shoulder, B's back to the camera" adds a third body rather than turning B around (both measured, 2026-09-07). To have one face on screen, write a beat that mentions one character and nothing about the other; introduce the second one in the beat where it enters.
- **Two speakers must be impossible to confuse, and only one speaks per beat.** Two look-alikes (the two aliens of the first dinner take) get their lines swapped, answer themselves, or say a line in unison — nothing in the picture tells the engine whose mouth a line belongs to. Give each speaker a different build, a different colour to wear and one vocal trait ("the tall one, deep and slow"; "the short one, thin and quick") in the first beat, repeat the pair every beat, write ONE line per beat, and say that the other one listens with its mouth closed.
- **A character scene holds ONE human face.** A trained character's LoRA paints every face in the frame: the second person at the table comes out as the character too, whoever the prompt says they are, and two characters' LoRAs together give both people the stronger one's face (measured 2026-09-07, Q8 Pro). Put a robot, an animal, a puppet or an off-screen voice across the table — not a second person.
- No text on screen; no lettering on signs unless you accept invented words.

## Time and quality

- H3 native (1344×768) with Turbo v4: about 31 minutes per 5 seconds on an M4 Max 64 GB. A minute is an evening.
- A 5-second test at High (1024×576) costs 14 minutes. Test the first beat before you commit the night.
