# Multi-keyframe interpolation — the SDK shot-composition primitive

Status: **Layer 1 shipped 2026-05-06** — helper now accepts arbitrary keyframe lists. Layer 2 (panel HTTP form-parsing) and Layer 3 (UI) not yet. Engine has supported this all along; we just exposed it. See [Agent API contract](#agent-api-contract-layer-1-2026-05-06) for the call shape.

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

### Layer 1 — DONE (2026-05-06)

Helper `generate_keyframe` action now accepts arbitrary `keyframe_images` + `keyframe_indices` lists with strict input validation. The hardcoded 2-keyframe path remains as a backward-compat fallback so the panel keeps working unchanged. See `mlx_warm_helper.py` lines ~1251-1360.

Full agent contract below.

### Layer 2 — panel HTTP form parsing — NOT YET

`mlx_ltx_panel.py` `mode == "keyframe"` branch. Currently parses `start_image` + `end_image` form fields and constructs the helper job spec. Change to parse a **list of keyframes** from the form (e.g. `keyframe_image_0`, `keyframe_image_1`, ... or a JSON-encoded list field). Build the lists for the helper.

Backward compat: if the form has `start_image` + `end_image` only, behave as today.

This is the layer the agent will use if it goes through the panel's HTTP queue endpoint instead of talking to the helper directly.

### Layer 3 — UI (the user-visible surface) — NOT YET

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

## Agent API contract (Layer 1 — 2026-05-06)

The helper subprocess (`mlx_warm_helper.py`) is a long-lived JSON-line server. Each line on stdin is one job; each line on stdout is one event. To use multi-keyframe interpolation, write a single JSON line to the helper's stdin with `action: "generate_keyframe"`.

### Request shape

```json
{
  "id": "<your-unique-job-id-string>",
  "action": "generate_keyframe",
  "params": {
    "prompt": "A young woman walks across a sunlit kitchen, opens the fridge, takes out an apple.",
    "keyframe_images": [
      "/abs/path/to/keyframe_0.png",
      "/abs/path/to/keyframe_1.png",
      "/abs/path/to/keyframe_2.png"
    ],
    "keyframe_indices": [0, 60, 120],
    "frames": 121,
    "width": 768,
    "height": 416,
    "output_path": "/abs/path/to/output.mp4",
    "seed": -1,
    "stage1_steps": 15,
    "stage2_steps": 3,
    "cfg_scale": 3.0,
    "negative_prompt": null,
    "model_dir": null
  }
}
```

### Field rules

| Field | Type | Required | Notes |
|---|---|---|---|
| `id` | str | yes | Echoed back in `done` / `error` events so the agent can match. Use any unique string (UUID, monotonic counter, etc). |
| `action` | str | yes | Must be exactly `"generate_keyframe"`. |
| `params.prompt` | str | yes | Scene description. The model uses this for the motion between keyframes; the keyframes themselves dominate composition. |
| `params.keyframe_images` | list[str] | yes (multi-keyframe) | Absolute file paths. PNG/JPEG. Length must equal `keyframe_indices`. **N ≥ 2.** Files must exist at job-submission time. |
| `params.keyframe_indices` | list[int] | yes (multi-keyframe) | Pixel-frame indices, 0-based, **strictly ascending**, all in `[0, frames-1]`. Must include at least 2 entries. Same length as `keyframe_images`. |
| `params.start_image` + `params.end_image` | str | yes (FFLF backward-compat) | Used only if `keyframe_images`/`keyframe_indices` are absent. Equivalent to `keyframe_indices=[0, frames-1]`. |
| `params.frames` | int | yes | Total pixel frames. Latent grid is 8 frames per slot, so use 8k+1 (49, 121, 241). 24 fps is hardcoded ⇒ 121 frames = ~5.0 sec. |
| `params.width`, `params.height` | int | yes | Output dimensions. Comfortable tier (M-Max 64 GB) is clamped to **768×416** for keyframe jobs (OOM history at full 1280×704). |
| `params.output_path` | str | yes | Absolute path where the resulting `.mp4` will be written (with audio). |
| `params.seed` | int | optional | `-1` (default) = random, otherwise fixed. Echoed back in the `done` event as `seed_used` so the agent can reproduce. |
| `params.stage1_steps` | int | optional | Default `15`. Stage-1 (half-res guided denoise) sigma count. Quality knob. |
| `params.stage2_steps` | int | optional | Default `3`. Stage-2 (full-res refine) sigma count. |
| `params.cfg_scale` | float | optional | Default `3.0`. Classifier-free-guidance strength. |
| `params.negative_prompt` | str \| null | optional | If provided, used as native CFG negative. Null = use default. |
| `params.model_dir` | str \| null | optional | Override the transformer directory. Null = default Q8 dev path. Don't change this unless you know why. |

### Response events

The helper emits JSON lines on stdout:

```json
{"event": "log",      "line": "Keyframe multi-3kf — indices=[0, 60, 120], pipeline cover-crops all to 768x416"}
{"event": "progress", "id": "...", "phase": "stage1", "step": 5, "total": 15, "pct": 33}
{"event": "progress", "id": "...", "phase": "stage2", "step": 1, "total": 3,  "pct": 90}
{"event": "done",     "id": "...", "output": "/abs/path/out.mp4", "elapsed_sec": 312.4, "seed_used": 1234567}
```

On any failure (validation error, OOM, model error):

```json
{"event": "error", "id": "...", "error": "<message>", "trace": "<python-traceback>"}
```

The helper continues running after errors; submit the next job whenever ready.

### Validation errors (fail-fast at submission, before any GPU work)

The helper rejects bad jobs immediately so agent bugs surface early:

- `keyframe_images and keyframe_indices must both be provided as lists`
- `keyframe_images and keyframe_indices must be lists`
- `keyframe_images (N) and keyframe_indices (M) must have the same length`
- `at least 2 keyframes required`
- `keyframe image not found: <path>`
- `keyframe_indices must be integers, got: <repr>`
- `keyframe_index <i> out of range [0, <frames-1>]`
- `keyframe_indices must be strictly ascending, got <list>`

### Concurrency model

The helper handles **one job at a time**. The agent's queue should serialize submissions or wait for `done` / `error` before submitting the next job. Multiple jobs interleaved on stdin will queue but the helper won't process the second until the first finishes.

If you need parallelism, spawn multiple helpers (each takes ~30 GB VRAM + ~30s warm-up; M4 Max 64 GB fits 1 reliably, 2 is tight).

### Wall-time budget

For a 121-frame (5-sec) clip at 768×416 with `stage1_steps=15`, `stage2_steps=3`:

- Cold helper (first job after start): ~30 s warm-up + ~5 min generate = ~5:30
- Warm helper (subsequent jobs): ~5 min generate

Going from 2 keyframes to 8 keyframes adds a few seconds of front-end VAE encoding, not significant against the ~5-min denoise.

### Concrete example — Director Mode three-shot scene

```python
import json, subprocess, uuid, sys

helper = subprocess.Popen(
    ["./ltx-2-mlx/env/bin/python3.11", "-u", "mlx_warm_helper.py"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    text=True, bufsize=1,
)

def submit(job):
    helper.stdin.write(json.dumps(job) + "\n")
    helper.stdin.flush()
    while True:
        line = helper.stdout.readline()
        if not line: raise RuntimeError("helper died")
        evt = json.loads(line)
        print(evt, file=sys.stderr)
        if evt.get("event") == "done"  and evt.get("id") == job["id"]: return evt
        if evt.get("event") == "error" and evt.get("id") == job["id"]: raise RuntimeError(evt["error"])

# Shot 1 — establishing wide. 4 keyframes for explicit choreography.
result = submit({
    "id": str(uuid.uuid4()),
    "action": "generate_keyframe",
    "params": {
        "prompt": "A woman walks across a kitchen, dappled morning light through window blinds.",
        "keyframe_images": [
            "/tmp/agent/kitchen_01_start.png",
            "/tmp/agent/kitchen_02_mid_left.png",
            "/tmp/agent/kitchen_03_mid_right.png",
            "/tmp/agent/kitchen_04_at_fridge.png",
        ],
        "keyframe_indices": [0, 40, 80, 120],
        "frames": 121,
        "width": 768, "height": 416,
        "output_path": "/tmp/agent/shot_01.mp4",
        "seed": 42,
    }
})
# result["output"], result["elapsed_sec"], result["seed_used"]
```

### What's NOT in this layer

- **Per-keyframe strength.** All keyframes are pinned at strength 1.0 (pure anchor). The model cannot "lightly suggest" a keyframe yet — it's all-or-nothing. Strength control requires an upstream patch in `ltx-pipelines-mlx`.
- **Panel HTTP route.** Layer 2 is needed to reach this from the panel's queue (HTTP form-data). Today the only path is the helper's stdin.
- **Panel UI.** Today's panel still has 2 image drop-zones. Adding more rows is Layer 3.
- **Audio across keyframe boundaries between shots.** Each clip is its own audio render — boundaries are hard cuts. See "What this doesn't solve" below.

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
