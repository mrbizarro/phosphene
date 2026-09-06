# Phosphene roadmap

Living document. Items listed here are committed for upcoming releases but
not yet shipped. Priorities are loose — they move based on what the v3.0
post-ship feedback surfaces. Filed publicly so contributors know where the
project is going.

Status notation: `[ ]` planned · `[~]` in progress · `[x]` shipped (will
be removed from this file once it's in a stable release).

## Near term (post v3.0)

### `[ ]` Three-aspect character LoRAs (vertical / horizontal / general)

The v3.0 character training trains at 576×576 square. This works well
for square + vertical inference, but degrades on widescreen medium
shots — the LoRA never saw the 1024×576 token grid during training,
so face geometry can break in 16:9. The fix: ship three modalities.

- **General** — current behavior, trained at 576×576 square. Default,
  works adequately at every aspect.
- **Horizontal** — trained at 1024×576 (16:9). Pinned to widescreen
  inference for medium shots that need clean face detail.
- **Vertical** — trained at 576×1024 (9:16). Pinned to vertical
  social-format clips.

UI: aspect selector in the Train tab, naming convention
`<trigger>_v2[_wide|_vert].safetensors`. Character picker groups all
three variants under one character; either auto-routes by output
aspect, or shows a small chip to override.

Foundation already in: lora_lab pipeline accepts width × height
independently as of commit `3f49ca3`. Just needs the UI surface.

Estimated effort: 2–3 hours panel work + 3.5h training per variant.

### `[ ]` Scene-continuity recipe (no-training, ship first)

Lock two characters into the same environment across cuts using
existing Phosphene primitives — no scene-LoRA training required.
Higher quality and lower operator risk than training a scene LoRA
per location, per research conducted 2026-05-20.

Workflow:
1. Image Studio → Qwen-Image-Edit-2511 multi-ref. Drop 1–3 photos
   of the target location (the room, the lighting, the props).
   Generate a composed still with both characters present in the
   scene.
2. Pick the best still. Send to I2V (or FFLF) with no character
   LoRAs — identity is carried by the still's VAE-encoded latent,
   not by a LoRA delta.
3. Repeat for each cut with the SAME location stills as references.
   Lighting + decor stay consistent.

Effort: ~0.5–1 day, docs + a small UX pass to surface the recipe
from the Train tab (a "Lock to a location" link or hint). No code
changes to the training pipeline.

### `[ ]` Scene / location / room LoRAs (power-user follow-up)

After the no-training recipe ships, add scene LoRAs as the path for
users who want a reusable "location" they can stack with any
character. Builds on the existing style-training pipeline
(`train_type=style` already wired in `/train/start`).

Validated training recipe (per 2026-05-20 research):
- Dataset: 15–20 stills of one place, 4–8 angles, mix of with-people
  / empty-room (60-70% / 30-40%)
- Caption format: describe what VARIES across stills (lighting,
  framing, who's in frame) but NOT the room itself — the trigger
  carries the room. ("Caption everything your LoRA is *not*
  supposed to control" — Civitai/Hunyuan rule of thumb.)
- Resolution: 1024×576 widescreen (matches inference aspect; needs
  the widescreen-training support from commit `3f49ca3`).
- Preset: Quick (rank 16, 600 steps from 30 epochs × 20 imgs) at
  ~30 min wall on M4 Max 64GB.

Stacking with character LoRAs:
- Documented LTX guidance: keep total combined strength under 2.0,
  practical sweet spot below 1.5.
- Starting weights: character 0.9–1.0, scene 0.5–0.7. The character
  LoRA's larger weight deltas need the scene to be turned down so
  the scene effect isn't drowned (matches the 2026-05-20 chartest
  diag observation about strength dominance).
- Reduce `strength_clip` on the scene LoRA before `strength_model`
  if the room effect is still too faint.

Engineering: ~1.5 days. Extend `TRAIN_TYPES` to include `scene`,
add a fallback caption body string, add a "Locations" subnav under
the Train tab. Touch points: `mlx_ltx_panel.py:785-801, 818, 5241,
5260, 5295-5358` and `lora_lab/train_character.py:277-290`.

### `[ ]` Multi-character workflow

A real workflow for putting two trained characters in the same
clip. The current two-LoRA stacking math produces a hybrid face
(both identities averaged at every spatial token — see the
2026-05-19 multi-character research). Path forward: Qwen-Image-Edit
multi-ref → composed still → I2V with no LoRAs (identity carried
by the still's VAE latent). The infra exists; this is mostly UX
work to make the four-step flow a single tab.

### `[ ]` Stacking-aware strength balance

When a user stacks a style or scene LoRA on top of a character LoRA,
the character's much-larger weight deltas at strength 1.0 dominate
the style/scene LoRA at strength 1.0 and the style/scene effect
disappears. UI fix: when a user adds a non-character LoRA on top of
a character, auto-reduce `character_strength` to ~0.7 with a
tooltip explaining why. Or expose a single "stylize this character"
preset that handles the math.

### `[ ]` Per-render TeaCache override on the form

The new default TC=1.8 (reverted from a wrongly-calibrated 1.0 on
2026-05-20) works well for most character renders, but power users
may want to dial it for specific scenes. Expose `teacache_thresh`
as a slider on the advanced section of the Generate form.

### `[~]` IC-LoRA pipeline support — HDR ships first, generic infra follows

Research summary (`/tmp/phosphene-walk/ic_vs_id_lora_research.md`):
IC-LoRA is just a regular LoRA delta + a training-data trick + an
inference-time input contract. The DiT itself isn't modified — what
changes is that reference frames are concatenated to noise tokens along
the sequence axis at negative RoPE positions, and the LoRA learns to
copy structure / identity / look from one half to the other via
attention. The HDR LoRA additionally bakes a LogC3 inverse transform
into the decode path to recover 16-bit linear HDR from the VAE's
`[-1, 1]` output range. The upstream MLX port already has the pieces:

- `ltx_pipelines_mlx.ic_lora.ICLoraPipeline` — generic IC-LoRA pipeline
- `ltx_pipelines_mlx.hdr_ic_lora.HDRICLoraPipeline` — HDR subclass that
  auto-detects the LogC3 config from LoRA safetensors metadata and
  saves both an SDR MP4 preview + a companion `.hdr.npz` float32 HDR
  tensor (F, H, W, 3 linear scene-light)
- `ltx_pipelines_mlx.iclora_utils.append_ic_lora_reference_video_conditionings`
  — handles reference-video encoding + RoPE positioning. Tolerates an
  empty `video_conditioning=[]` (text-driven mode: LoRA delta still
  applies, no IC reference needed).

Plan in phases, each shippable on its own:

**Phase 1 — HDR text-driven mode (PLUMBING SHIPPED, BUTTON PULLED).**
The full plumbing landed in v3.0 dev: `generate_hdr` helper action,
`HDRICLoraPipeline` wiring, panel routing, all working end-to-end.
But the UI button was pulled 2026-05-21 (commit `<hdr-pull>`) because
the killer use case is HDR + character — and that combo is
architecturally unsound on the single-pass distilled path. Pure-text
HDR isn't useful enough on its own to justify the button. Re-exposing
the pill is a one-line un-comment at `mlx_ltx_panel.py:17441` once
Phase 2 ships and the two-pass character flow works. The helper
action stays callable via the HTTP API (`/run` with `hdr=on`) for any
script that wants pure-text HDR today.

**Correction discovered 2026-05-21:** The premise I worked from in
the original phasing (character LoRAs trained against Q8 dev, HDR
trained against distilled, deltas misaligned) was wrong. Verified by
inspection:

- Phosphene's lora_lab points the trainer at `mlx_models/ltx-2.3-mlx-q4/`
  via `resolve_default_model_dir()`.
- That directory contains ONLY `transformer-distilled.safetensors`
  (no dev variant; the Q4 download filter excludes it).
- The upstream `ltx_trainer_mlx.model_loader` picks transformer files
  in order: `transformer.safetensors`, `transformer-distilled.safetensors`,
  `transformer-dev.safetensors`.
- → Character LoRAs are trained against the **distilled** checkpoint,
  same base as HDR-IC. The combo on the distilled path stacks
  CLEANLY — no fine-tune misalignment.

The Phosphene-side comment calling the base "dev transformer" (still
present at `lora_lab/train_character.py:742`) is historical confusion
and should be updated.

What this means for HDR + character:
- The single-pass distilled-path combo (which Phase 1 plumbing
  supports today) should produce a faithful character + HDR output.
  The only quality cap is "distilled = 8 steps, HQ dev = ~30 steps"
  — that's a generic speed/quality tradeoff on the distilled lane,
  not an HDR-specific fidelity killer.
- The "experimental" framing I baked into the UI pill tooltip was
  overly cautious. When the pill comes back, drop that warning.
- Phase 2 (two-pass re-grade) is still architecturally clean and
  remains the path for HQ-quality character + HDR. But Phase 1
  alone should now be acceptable for character work.

The HDR button was still pulled for v3.0 (Mr Bizarro request — wants
to ship the release and validate HDR separately), but it's a
one-line un-comment to bring back once you want to test the
single-pass distilled-path character + HDR combo.

**Phase 2 — Two-pass character + HDR via SDR→HDR re-grade (HQ path).**
This is what Mr Bizarro initially asked for. Workflow:
  1. User renders normally at Q8 dev with character LoRA → full
     character fidelity SDR MP4.
  2. User clicks "Re-grade to HDR" on the output card (next to
     Animate / Quality buttons).
  3. Panel queues a `generate_hdr` job with
     `video_conditioning=[(source.mp4, 1.0)]` and `loras=[HDR-IC]`
     (NO character LoRA — character is baked into the SDR reference
     latents).
  4. HDRICLoraPipeline reads the SDR video, encodes it as the IC
     reference, generates a HDR-graded version conditioned on the
     reference latents. LogC3 inverse on decode → float32 scene-linear
     `.hdr.npz` companion + a re-graded SDR preview MP4.
Both stages use their natural base (Q8 dev for character, distilled
for HDR-IC). Character fidelity survives because the SDR reference
carries the identity through to the HDR stage as conditioning, not
as a fused LoRA delta. This is also the Lightricks-recommended
workflow per the HDR LoRA's own README ("video conversion from 8 bit
SDR to 16 bit HDR").

Once Phase 2 ships, re-expose the HDR pill (which becomes "Re-grade
to HDR" on output cards rather than a pre-submit form toggle).

**Phase 3 — Motion-Track and Union-Control IC-LoRAs.** Two more
Lightricks IC-LoRAs at `Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control`
and `Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control`. Both need a
reference video with specific conditioning content (motion-track
needs sparse colored splines from a tracker like SpatialTrackerV2;
union-control needs canny+depth+pose). For Phosphene users this
means we need to either (a) ship spline/depth/pose extractors as a
pre-processing pass, or (b) accept user-provided reference videos.
Likely (b) for v3.x, (a) as a stretch.

**Phase 4 — IC-LoRA training.** Extend `lora_lab` with a paired-dataset
trainer. Each sample is `(reference_clip, target_clip)`. Training loop
builds `concat(ref_tokens_at_negative_RoPE, target_tokens)` and
computes loss only on the target half. LoRA weights file stays a
normal `.safetensors`. Add `train_type='iclora'` to the trainer UI.

**Phase 5 — Character + IC-LoRA stacking.** The prize: stack a
character LoRA (face identity) with an IC-LoRA (e.g. Union-Control)
on the same DiT. Both are weight deltas so it's mechanically possible,
but untested upstream — deltas may fight, distilled-only constraint
limits to balanced quality. Experimental; gate behind a flag until
results are validated.

**Naming note.** Don't conflate "ID-LoRA" with "character LoRA".
The published ID-LoRA (arXiv 2603.10256) is actually an IC-LoRA that
conditions on a reference image + ~5s audio for talking-head video.
Phosphene's character LoRAs are plain weight-delta LoRAs trained on
15-50 photos with a trigger word — Lightricks calls these "character
LoRAs" formally, never "ID-LoRA". Keep "character LoRA" in code and
UI to avoid the collision.

Tracking: the HDR-specific Phase 1 work landed via re-exposing the
hdr toggle (commit `<hdr-ship>`); the generic Phases 2–5 stay open
in this entry.

### `[x]` I2V Balanced + Extend perf — post-decode hang (FIXED 2026-05-21)

**Fix shipped: panel-side watchdog SIGKILLs hung helper from outside
the GIL-blocked process.** See commit `94bd696`. Detail below kept
for posterity + so the next person to hit something MLX-deallocator-
shaped knows where to look.

After the May 9 upstream `ltx-2-mlx` refactor, I2V Balanced (and T2V
Balanced) route through `DistilledPipeline.generate_two_stage(image=...)`.
The actual render completed in ~3 min for a 5s I2V at 1024×576, but
the helper then hung 5-15 min before signaling done to the panel.
Extend exhibited the EXACT SAME hang on `ExtendPipeline`.

The first fix attempt (commit `adc1cd2`, reverted) shipped an
in-helper daemon-thread watchdog. **It does not work**: Metal's
command-buffer completion handlers block every Python thread's GIL
access during the deallocator chain, so the watchdog thread is
starved by the very thing it was meant to escape. Observed live:
ps showed all 16 helper threads at 0% CPU during the hang.

Working fix (commit `94bd696`): rescue from THE PANEL (separate
process, no GIL contention with the helper). `WarmHelper.
_build_post_decode_panic` returns a (log_hook, panic_check) pair
honored by `_read_until`. The log_hook spots the decode-done log
line and arms a 45s grace clock; if the helper is still silent
after grace AND the output file is on disk, panic_check SIGKILLs
the helper and returns a synthetic done event from the known
output_path. Helper respawns on next job (~30s spawn cost).

Validated end-to-end same evening on the previously-hung Extend
case: 533s total wall, watchdog rescue logged exactly as designed.

Diagnostic findings (2026-05-21 session):
- Output mp4 IS written to disk before the hang starts.
- Stack samples show the helper deep in Metal command-buffer
  deallocation (`IOGPUMetalCommandBufferStorageDealloc`,
  `MTLResourceList releaseAllObjectsAndReset`, etc).
- Diagnostic prints placed AFTER `_decode_and_save_video()` never
  fire — the function-return path itself is the hang. Most likely
  MLX/Metal completion-handler chains holding the GIL through
  Python frame teardown.
- LTX2_DIT_EVAL_EVERY tuning has no effect (tested 0/1/4/8).
- A2V (A2VidPipelineTwoStage), FFLF, T2V Character HQ
  (TI2VidTwoStagesHQPipeline) all return cleanly — confirming the
  hang is specific to the distilled-fused-then-decode pattern.

**Symptom-level fix shipped 2026-05-21** (commit `94bd696` — panel-
side watchdog SIGKILLs hung helper). Quality-level work still open:

1. **For Q8 (≥48 GB) tiers:** route Balanced I2V/T2V through
   `TI2VidTwoStagesPipeline` instead of `DistilledPipeline`. That class:
   - Runs full-resolution, not half-res with Stage-2 upscale.
   - Supports CFG (`cfg_scale=3.0` validated).
   - Supports TeaCache (`enable_teacache=True, teacache_thresh=N`).
   - Already used by the working HQ path.
   The helper's `get_pipe('i2v', ...)` currently aliases to
   DistilledPipeline; needs an explicit branch for Balanced→
   TI2VidTwoStagesPipeline on the dev model. Eliminates the hang
   entirely (TI2VidTwoStages doesn't trigger it) AND gives better
   I2V quality on the distilled-grid artifact #5 users see.
2. **For Q4 (<48 GB) tiers:** the panel watchdog now hides the
   hang from users (file lands, panel reports done, helper
   respawns). Quality fix still needs the same TI2VidTwoStages
   route on Q4 weights — but until then, watchdog rescue makes
   the symptom invisible.
3. **Reddit / issue #5** users on small Macs are hitting both this
   hang AND the geometric-grid artifact (which is the DistilledPipeline
   structure producing weak I2V on the distilled checkpoint). The
   hang is now invisible; the artifact still requires the Q8/Q4
   TI2VidTwoStages route above.

## Mid term

### `[ ]` In-panel "Report bug" button

Auto-zip recent panel log + system info + the most recent failed
sidecar, pre-fill a GitHub issue. Cuts the support loop from
"please paste the panel log" to one click. (Tracked partly via
issue #2 — Akossimon's pain point.)

### `[ ]` Aspect-bucketed single LoRA (alternative to three modalities)

Instead of training three separate LoRAs per character, train ONE
LoRA on a mix of aspect crops. Robust to any inference aspect at
the cost of slightly weaker per-aspect identity capture. Worth
benchmarking against the three-modality approach once both are
implemented.

### `[ ]` Audio-only LoRA training (voice without face)

The voice LoRA phase already runs as a sub-step of character
training. Expose it as a standalone train_type so a user can train
just a voice (e.g. clone a podcast host's delivery) without
gathering face photos.

### `[ ]` HiDream as a panel-internal install

HiDream-O1-Image-Dev currently requires a separate one-time clone
of the HIDREAM lab repo into `$HIDREAM_LAB_DIR`. Move this to a
standard Pinokio install button so it's one-click parity with the
mflux Qwen-Edit install.

### `[ ]` /queue/retry preserves character_strength

Currently `/queue/retry` rebuilds the form with `character_id` +
`quality` + `loras` but not `character_strength`. Once strength
balancing is exposed in the UI, retries need to round-trip it.

## Longer term

### `[ ]` Multi-keyframe interpolation surface

The SDK supports multi-keyframe interpolation (see
`docs/SDK_KEYFRAME_INTERPOLATION.md`). The panel exposes only the
first/last frame mode. Building a proper multi-keyframe timeline
UI would unlock storyboarding workflows.

### `[ ]` LoRA marketplace integration

CivitAI browsing already works for downloading LTX 2.3 LoRAs.
Adding upload (publish your trained Phosphene LoRA to CivitAI in
one click, with sidecar metadata auto-populated) closes the loop.

### `[ ]` Hosted-cloud rendering fallback

Some users run on Apple Silicon Macs that are too small for Q8 HQ.
A "render this on a beefier rented box" option (BYO HF Inference
Endpoint or Replicate) would let them author character LoRAs
locally and render at HQ remotely. Strictly opt-in — Phosphene's
core promise is local + no cloud + no API key, so this would live
behind a clearly-labeled flag.

## Won't do (until proven otherwise)

### `[x] no cuda fallback`

Phosphene is Apple Silicon + MLX by design. The performance and
unified-memory advantages on M-series chips are the entire reason
the project exists. A CUDA port would be a different project.

### `[x] no Windows / Linux build`

Same reason as above. MLX is Apple-only by Apple's design.

---

If you'd like to contribute to one of the planned items, the GitHub
issues at https://github.com/mrbizarro/phosphene/issues are the right
place — comment on a tracking issue or open a new one referencing the
roadmap entry you're picking up.
