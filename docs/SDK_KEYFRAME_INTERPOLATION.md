# Multi-keyframe interpolation — the SDK shot-composition primitive

Status: **research + design doc.** Not yet implemented. Documents a known gap in Phosphene where the underlying engine supports more than the UI exposes.

## TL;DR

LTX 2.3's keyframe-interpolation pipeline takes a **list of images** and a **list of frame indices** and generates the motion that connects them. Drop a still at frame 0 and frame 240 and it fills the 240 frames in between. Drop FOUR stills across the timeline and it fills three connecting motions — character keeps continuity, lighting keeps continuity, the model interpolates a coherent path.

Phosphene's engine already supports this. The UI artificially restricts it to *exactly two* keyframes (first frame + last frame at indices 0 and N-1). The "FFLF" mode in the panel is an artificial special case of a much more general capability sitting in `ltx_pipelines_mlx`.

Exposing the full multi-keyframe interface unlocks the **shot-composition primitive** the agentic SDK roadmap is built around: an agent picks N anchor frames, the model fills the connecting motion, and the result is a clip whose composition the human can predict from the keyframes alone.

This doc is the literature review + design sketch.

## What ComfyGuy9000 was demoing

The technique referenced in the May 2026 Twitter thread is **first-frame-last-frame method** — generate or pick two still images, then run an LTX 2.3 keyframe-interpolation pass that fills the motion between them.

The repo cited (`Deno2026/comfyui-deno-custom-nodes`) is a ComfyUI workflow toolkit that ergonomically threads multiple keyframe images into LTX. The headline node is `(Deno) LTX Sequencer` — accepts up to 50 keyframe images, lets you place each at an arbitrary frame index or timecode, with per-keyframe strength control 0.0–1.0.

The keyframe injection itself is upstream Lightricks code (`comfy_extras.nodes_lt.LTXVAddGuide`):

```python
encoded_image, encoded_latent = LTXVAddGuide.encode(
    vae, latent_width, latent_height, image, scale_factors)

conditioning_frame_idx, latent_idx = LTXVAddGuide.get_latent_index(
    positive, latent_length, len(encoded_image), frame_index, scale_factors)

positive, negative, latent_image, noise_mask = LTXVAddGuide.append_keyframe(
    positive, negative, conditioning_frame_idx, latent_image, noise_mask,
    encoded_latent, strength, scale_factors)
```

What's happening:
1. **VAE-encode each keyframe** at the latent resolution the sampler is operating at.
2. **Compute the latent temporal index** corresponding to the requested pixel-frame index (LTX has an 8× temporal compression — pixel-frame `K` lives at latent slot `K // 8`).
3. **Splice the encoded keyframe into the latent tensor** at that slot, and **mask the slot in the noise_mask** so the sampler doesn't overwrite it during diffusion.
4. **Strength** controls how strict the binding is: 1.0 = pure keyframe (latent slot held exactly), <1.0 = some noise still applied so the keyframe acts as a soft target rather than a hard constraint.

Result: the diffusion process is steered through the keyframes. Frames between them are generated to smoothly connect the anchors.

## What Phosphene's engine already supports

The MLX port (`ltx-2-mlx/packages/ltx-pipelines-mlx`) ports Lightricks's reference `KeyframeInterpolationPipeline`. Verified-from-source signature:

```python
class KeyframeInterpolationPipeline(TwoStagePipeline):
    def interpolate(
        self,
        prompt: str,
        keyframe_images: list[Image.Image | str],   # arbitrary length
        keyframe_indices: list[int],                # pixel-frame indices, 0-based
        height: int = 480,
        width: int = 704,
        num_frames: int = 97,
        fps: float = 24.0,
        seed: int = 42,
        stage1_steps: int | None = None,
        stage2_steps: int | None = None,
        cfg_scale: float = 1.0,
        ...
    ) -> tuple[mx.array, mx.array]:
```

`keyframe_images` and `keyframe_indices` are both **lists**. The function loops over them, encodes each at the appropriate stage resolution, and threads them through stage 1 (half-res guided denoise) and stage 2 (full-res refinement). Multi-keyframe is the native shape of the API.

Today's restriction is one layer up — in the panel's worker:

```
mlx_warm_helper.py:
    keyframe_images=[p["start_image"], p["end_image"]],
    keyframe_indices=[0, num_frames - 1],
```

Two keyframes, hardcoded to first and last. The panel's `mode == "keyframe"` only accepts `start_image` and `end_image` form fields. UI offers two image drop-zones.

Net: every other keyframe slot the engine could use is silently discarded.

## Why this matters for the SDK

The Phosphene 2.x roadmap target is an **agentic flow** that turns one user idea into a coherent multi-shot piece. Reaching that requires a building block where:

1. The agent decides what should happen visually.
2. The model fills in the motion.
3. The composition is bounded — predictable from the inputs alone, no random drift mid-shot.

Multi-keyframe interpolation is exactly that block. The agent can:

- Generate (or curate) N still images for a shot. Each still is the AGENT'S decision about composition, character pose, lighting, framing.
- Pass them to LTX with frame-index placements that match the desired pacing.
- Receive a clip whose start, middle, and end states are KNOWN. The model only fills the motion between them.

This addresses three problems we've documented elsewhere this week:

### Problem 1 — multi-shot character drift

The mom-kid scene experiment (M1, M2, M3 in `mlx_outputs/`, May 4) confirmed that running the same prompt at three different angles with different seeds produces three different women. Single-prompt continuity across shots fails naively.

**With multi-keyframe**: the agent generates one keyframe per shot of the same character (extracted from a master shot, or one Gemma-text-to-image render reused across shots), feeds that frame as keyframe-0 of each shot. The character is anchored at every shot start by construction.

### Problem 2 — shot composition is a roll of dice

Today's T2V is "describe the scene in words and pray." There's no way to point the model at a SPECIFIC framing or pose. The Director Mode roadmap (`HAI-155` in Linear) bottlenecks here — without compositional control, "render N shots and stitch" is what we tested and it doesn't compose into a coherent scene.

**With multi-keyframe**: composition is *inputs*, not *prompts*. The Director picks the shot's start image and end image (and optionally middle anchors), and the motion is the only degree of freedom the model has.

### Problem 3 — face quality at distance

LTX 2.3 deforms faces below ~80 px in-frame size. Wide shots that begin or end with a small-in-frame character produce unusable opening / closing seconds.

**With multi-keyframe**: pin the keyframe at a moment when the face is at viable size and let the motion away-from-camera happen between keyframes. The face is "correct" at the anchor; the model interpolates the small-face frames as motion blur and we don't depend on those frames to be face-readable.

### Problem 4 — cross-shot continuity

Today, shot N's last frame and shot N+1's first frame have no connection. An NLE editor cuts between them and the cut is jarring because the model didn't know to align them.

**With multi-keyframe**: shot N's last frame is fed as shot N+1's keyframe-0. The two clips share an exact boundary frame. The cut becomes invisible — same character, same pose, same lighting.

This is the SDK's compositional primitive in one sentence: **the editor controls the keyframes, the model controls the motion.**

## Implementation plan

Three layers, in order:

### Layer 1 — helper (smallest change, biggest unlock)

`mlx_warm_helper.py` action `generate_keyframe`. Currently:

```python
keyframe_images=[p["start_image"], p["end_image"]],
keyframe_indices=[0, num_frames - 1],
```

Change to read `keyframe_images` and `keyframe_indices` directly from the job spec (lists). Default to the existing 2-keyframe behavior if the spec only provides `start_image` + `end_image` (backward compat). New job spec accepts:

```json
{
  "action": "generate_keyframe",
  "params": {
    "keyframe_images": ["<path1>", "<path2>", "<path3>"],
    "keyframe_indices": [0, 60, 120],
    ...
  }
}
```

Three lines of code, no model change, no upstream change.

### Layer 2 — panel (form parsing + worker dispatch)

`mlx_ltx_panel.py` `mode == "keyframe"` branch. Currently parses `start_image` + `end_image` form fields and constructs the helper job spec. Change to parse a **list of keyframes** from the form (e.g. `keyframe_image_0`, `keyframe_image_1`, ... or a JSON-encoded list field). Build the lists for the helper.

Backward compat: if the form has `start_image` + `end_image` only, behave as today.

### Layer 3 — UI (the user-visible surface)

The current Keyframe section in the panel HTML has two drop-zones. Replace with a list:

- "+ Add keyframe" button that adds a row
- Each row: image drop-zone, frame index slider (0 to num_frames-1) OR timecode input (0 to duration), per-keyframe strength slider (0.0–1.0; for now, default 1.0 — strength isn't yet exposed by the underlying pipeline so this is forward-compat)
- Reorder by drag, delete row by × button
- Min 2 keyframes enforced (first + last); max 8 for sanity (reference Deno node allows up to 50, but we should respect memory budget on Comfortable tier)

Per-keyframe **strength** isn't currently a kwarg on `KeyframeInterpolationPipeline.interpolate()`. Adding it requires a small upstream patch in `ltx-2-mlx`. Out of scope for first ship — strength=1.0 is the right default for a "keyframe = anchor" interpretation.

### Layer 4 — Director Mode integration (later)

Once multi-keyframe ships, the Director Mode (HAI-155) gains the compositional primitive it needs. Director's planner:

1. Generates a script and a shot list.
2. For each shot, generates anchor stills (Gemma text-to-image, or extraction from the previous shot's end frame).
3. Submits each shot as a multi-keyframe job to Phosphene.
4. Stitches the resulting clips with ffmpeg-concat (no transitions needed — keyframes guarantee continuity).

The agent never has to "describe a scene in words and pray." Composition is concrete, predictable, controllable.

## Memory budget

A keyframe job today, on Comfortable tier (M-Max 64 GB), runs at clamped 768×416 (Y9 in Linear notes the OOM history at full 1280×704 — kept the clamp). N keyframes don't change peak memory significantly — each VAE-encode is one image at a time, and the sampler latent is the same shape regardless of how many slots are pinned.

Risk-of-OOM ranking: same as current FFLF. No new memory pressure from going from 2 keyframes to 8.

## Risks

- **Quality at high keyframe density**: not all keyframe placements compose. Two anchors 240 frames apart probably interpolate cleanly. Eight anchors over 240 frames forces fast intermediate motion that might artifact. Need empirical batch — "what's the maximum useful keyframe density for a 10-second clip at our resolution" is a real question. First batch of post-implementation experiments should sweep this.
- **Strength control gap**: per-keyframe strength is in the ComfyUI/Deno design, not yet in our upstream. Adding it later requires a small kernel change in `LTXVAddGuide.append_keyframe`-equivalent code in ltx-pipelines-mlx. For first ship default to strength=1.0 (pure anchor).
- **VAE encoding of the keyframes adds front-end time**. Currently a 2-keyframe FFLF VAE-encodes 2 images at start. Going to 8 keyframes means 8 encodes — a few seconds added, not significant compared to the ~5-minute denoise.

## What this doesn't solve

- **Audio synchronization across keyframe-bound clips**. LTX generates audio jointly with video. If multiple shots are stitched via cross-shot keyframe sharing, audio at the boundary is generated independently per shot. Audio cuts will be hard cuts. Acceptable for cinema (cut-on-cut is normal), problematic for continuous-music underscore. Future work.
- **Identity stability of the keyframe images themselves**. If the agent generates the keyframes via Gemma text-to-image (or any other still-image model), those images need to have consistent identity across shots. That's an upstream still-image problem (LoRA-tied character generation, IP-Adapter-style identity extraction). Not solved here.

## References

- Lightricks `KeyframeInterpolationPipeline` source — `ltx-2-mlx/packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/keyframe_interpolation.py` in this repo (vendored).
- `LTXVAddGuide` upstream — ComfyUI's `comfy_extras/nodes_lt.py` (the `add_guide` family of methods).
- `Deno2026/comfyui-deno-custom-nodes` — the multi-keyframe sequencer node, MIT-licensed Python wrappers around `LTXVAddGuide`. Useful as ergonomic reference for the panel UI layout.
- Phosphene experimental data:
    - `mlx_outputs/mlx_keyframe_768x416_121f_20260503_170311.mp4` — Alice castle interpolation, 2-keyframe FFLF working today.
    - `mlx_outputs/mlx_t2v_1024x576_481f_20260504_15{1806,3902,5946}_720p.mp4` — M1/M2/M3 mom-kid trio. Demonstrates the multi-shot character drift problem multi-keyframe is meant to solve.

## Open questions for v0.1

1. Should the panel UI represent keyframe placement on a TIME slider (seconds) or a FRAME slider (24fps integer indices)? Frame indices are precise but unfriendly; seconds are friendly but require rounding to the latent grid (multiples of `1/3` second since 8 frames = `1/3` second at 24fps). Suggest seconds with `0.33`-second snap.
2. How does the panel handle a keyframe placed *between* latent grid points? `LTXVAddGuide.get_latent_index` rounds; we should mirror that rounding and surface the actual placed frame in the UI.
3. Does the agent (Director Mode) want to specify keyframes by image-path or by image-content? If by content, it implies a still-image model running upstream of every keyframe — adds 30+ sec per keyframe to total wall time. Worth designing the Director's interface around this.
