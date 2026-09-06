# Storyboard — the planner

The Storyboard tab turns one concept sentence into a film of shots (3 / 5 / 10 s
each), then renders them through the ordinary job queue. The planner lives in
`storyboard_planner.py` (`plan_film`: screenplay → geography → shots, then the
laws); the schema, validation, shooting order, pricing and the shot → job
mapping live in `storyboard.py`. The HTTP surface is documented in
`docs/API.md` (the `/storyboard/*` routes); the prompt-writing rules the
planner follows are in `docs/PROMPTING.md`. This page covers the one planning
tool that is not a 3 / 5 / 10 s shot.

## One Shot inside a film

**What it is.** A One Shot is one scene of an ordinary multi-shot film rendered
as a single unbroken take of 30, 45, 60, 90 or 120 seconds: the camera never
cuts, one movement happens per five-second beat, and the world changes around
the subject while the subject, the camera position and the light stay
continuous. The planner chooses it on its own, as a cinematic tool, when a
scene earns unbroken time — a walk-and-talk, a chase or a POV ride, a
monologue or a confession, a reveal, an arrival through a place — and leaves
it alone for a montage, cross-cutting, or anything that needs a reverse angle.
At most one or two per film, never the whole film unless the brief asks for one
take (that is the existing whole-film take: the board's `take_seconds` +
`collapse_take`). A One Shot counts as one shot in the shot count.

**What the planner writes.** The shot keeps every ordinary key and adds two:

| field | value |
|---|---|
| `take_seconds` | one of `30, 45, 60, 90, 120` — anything else the model writes snaps to the nearest |
| `beats` | exactly `take_seconds / 5` strings, in order, each what happens NEXT in the same unbroken shot: led by the movement, naming the sound, with the time of day and the weather stated once in the first beat and never changed |

On the board the shot then carries `take_seconds`, `duration_s == take_seconds`,
`frames == take_seconds × 24 + 1`, `beat_lines` (the beats as the model wrote
them — what a re-plan is shown) and `beats` (each beat assembled in the shot's
own register — the camera law, the face law, the soundscape and the character
trigger on every beat, the settle on the last written beat only). The shot's
`prompt` is its first beat. A shot without `take_seconds` is untouched by any
of this.

**Shaping, never rejecting.** A beat count that is off is padded with blanks
(the panel holds the previous moment on a blank beat) or trimmed; the planner
says so in its warnings. A `take_seconds` with no written beat at all is not a
take — the shot is planned as an ordinary one. A hand-edited `take_seconds`
that is not one of the five lengths is the one thing the validator refuses
(`bad_take_seconds`), because the job would otherwise silently render a normal
clip of the wrong length. The face law scrubs every beat, not just the
description.

**How it renders.** `shot_to_job` posts the One Shot as ONE job —
`take_seconds` plus the `beats` JSON, padded to the take's count and each
written beat composed with the shot's location, wardrobe, framing and eyeline
(a later part is rendered from its beats alone). The panel then does exactly
what it does for a take from the Video tab: parts that continue from each
other's last frame — 15 s parts (three beats) on H3, 10 s parts (two beats) on
LTX — with the light lock and the drift retake. The estimate prices the take
by those parts (`take_parts`): on H3 the measured 15 s cell per part, on LTX
the take's seconds. The take is exempt from the long-shot windows chain.

**Limits.** 30–120 s in the five steps above; one or two per film; the
planner's laws (camera, faces, dialogue, silence) apply to every beat; the
whole-film take and a One Shot do not nest — if a film that contains a One
Shot is collapsed into one take, its beats are kept in order rather than
dropped.

The tests are `test_storyboard_one_shot.py`; the whole-film take is covered in
`test_take.py`.
