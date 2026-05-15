# Phosphene HTTP API

External agents (Claude Code, Codex, OSS, or anything that speaks HTTP) drive Phosphene through this API. The in-panel chat was removed 2026-05-15 — there's no agent embedded inside the panel anymore. Everything that the chat used to do can be done by an external agent through these endpoints.

## Server

- **Base URL:** `http://127.0.0.1:8199`
- **Process:** `mlx_ltx_panel.py` (in the repo root)
- **Wire format:** HTTP/1.1, form-encoded `POST` bodies. JSON exceptions are noted per-endpoint.
- **Auth:** none — bound to loopback only.

## Conventions

- Every JSON response is `Content-Type: application/json`.
- Errors return `{"error": "<message>"}` plus a non-2xx HTTP status.
- The `/agent/*` namespace is reserved-and-removed: any path under `/agent/` except `/agent/image/config` returns **410 Gone** with `{"error": "agentic flows removed; see docs/API.md"}`. New endpoints should not be added under `/agent/`.

---

## Generation — submit and manage jobs

### `POST /queue/add` — enqueue a video render

Add a job to the panel's queue. Returns immediately; the helper renders it asynchronously.

**Form fields** (all required unless marked optional):

| Field | Type | Note |
|---|---|---|
| `mode` | `t2v` \| `i2v` \| `extend` \| `keyframe` \| `image` | Job type. |
| `prompt` | string | Full prompt text. Trigger words for LoRAs go here. |
| `negative_prompt` | string | Optional. Usually empty. |
| `width`, `height` | int | Both divisible by 32. |
| `frames` | int | Must satisfy `frames % 8 == 1`. 121 = 5s, 169 = 7s, 241 = 10s. |
| `frame_rate` | float | Default `24`. LTX-2.3 was trained at 24 fps; deviation degrades quality. |
| `seed` | int or `-1` | `-1` = random. |
| `quality` | `quick` \| `balanced` \| `standard` \| `high` | **For character LoRA work, use `high`.** `balanced` silently routes >121f clips to the Q4 distilled transformer where current LoRAs lose identity. |
| `stage1_steps`, `stage2_steps` | int | HQ two-stage pipeline. Validated defaults: `10` / `3`. |
| `teacache_thresh` | float | Validated speedup plateau at `1.8–2.0`. Below `1.6` no speedup. Higher = more aggressive caching. |
| `cfg_scale` | float | Classifier-free guidance. `3.0` is the validated default. |
| `bongmath_max_iter` | int | Sampler inner-loop cap. `100` is upstream max. |
| `upscale` | `off` \| `fit_720p` \| `fit_1080p` | Optional post-process. |
| `upscale_method` | `lanczos` \| `pipersr` | If `upscale != off`. |
| `accel` | `off` \| (other modes — see panel source) | Acceleration knob. |
| `enhance` | `true` \| `false` | If `true`, Gemma rewrites the prompt before encoding. **Set `false` when the prompt contains LoRA trigger words** — the rewriter can strip them. |
| `hdr` | `true` \| `false` | HDR ic-lora pass. |
| `image` | path | I2V only. Absolute path to a reference PNG/JPG. |
| `audio` | path | Optional audio reference. |
| `label` | string | Optional UI label for the queue card. |
| `temporal_mode` | `native` \| (tiled modes) | `native` for normal jobs. |
| `loras` | JSON-encoded array | See LoRA payload below. |

**LoRA payload:** the `loras` field is a JSON-encoded array of `{path, strength}` objects. Stack as many as needed:

```json
[
  {"path": "/Users/salo/pinokio/api/phosphene-dev.git/mlx_models/loras/ariatrn_v2.safetensors", "strength": 1.0},
  {"path": "/Users/salo/pinokio/api/phosphene-dev.git/mlx_models/loras/ariatrn.audio.safetensors", "strength": 1.0}
]
```

**Return:** `{"ok": true, "id": "j-<...>"}` — the job ID for polling.

**Example:**

```bash
curl -s -X POST http://127.0.0.1:8199/queue/add \
  --data-urlencode "mode=t2v" \
  --data-urlencode "prompt=Cinematic close-up of bizarrotrn man in a wood-paneled study, golden hour, photorealistic." \
  --data-urlencode "width=1024" --data-urlencode "height=576" \
  --data-urlencode "frames=169" --data-urlencode "steps=8" \
  --data-urlencode "quality=high" --data-urlencode "temporal_mode=native" \
  --data-urlencode "stage1_steps=10" --data-urlencode "stage2_steps=3" \
  --data-urlencode "teacache_thresh=2.0" --data-urlencode "cfg_scale=3.0" \
  --data-urlencode "bongmath_max_iter=100" --data-urlencode "accel=off" \
  --data-urlencode "enhance=false" --data-urlencode "upscale=fit_720p" \
  --data-urlencode "upscale_method=lanczos" --data-urlencode "seed=-1" \
  --data-urlencode 'loras=[{"path":"/Users/salo/pinokio/api/phosphene-dev.git/mlx_models/loras/bizarrotrn_v2.safetensors","strength":1.0},{"path":"/Users/salo/pinokio/api/phosphene-dev.git/mlx_models/loras/bizarrotrn.audio.safetensors","strength":1.0}]'
```

### `POST /run` — alias for `/queue/add` (identical behavior).

### `POST /queue/retry`

Re-queue a job from history with the same params. `params.open_when_done` is forced to `false` (retries are background).

| Field | Type | Note |
|---|---|---|
| `id` | string | Source job ID. |

Returns `{"ok": true, "id": "<new-id>", "source_id": "<old-id>"}`.

### `POST /queue/remove`

Remove a queued job. (Running job can't be removed — use `/stop`.)

| Field | Type | Note |
|---|---|---|
| `id` | string | Job ID. |

Returns `{"ok": true}`.

### `POST /queue/clear`

Empty the queue. Returns `{"ok": true, "cleared": <count>}`.

### `POST /queue/pause`, `POST /queue/resume`

Pause/resume the queue dispatcher (does not affect a job already running).

### `POST /stop`

Request the running job to stop. Returns `{"ok": true}`.

### `POST /queue/batch`

Submit multiple prompts as one batch.

| Field | Type | Note |
|---|---|---|
| `prompts` | string | Newline-separated prompts. |
| (plus all the standard `/queue/add` fields, applied to each row) | | |

---

## Status — poll job state

### `GET /status`

Returns the full panel state.

```json
{
  "running": false,
  "paused": false,
  "current": null,
  "queue": [],
  "history": [ { "id": "j-...", "status": "done", "params": {...}, "elapsed_sec": 426.1, "output_path": "...", "raw_path": "...", "error": null }, ... ],
  "log": [ "[HH:MM:SS] helper line...", ... ]
}
```

Field semantics:
- `current` — the job actively running (`null` when idle).
- `queue` — jobs waiting, FIFO order.
- `history` — completed/failed jobs. Each entry has `status`, `elapsed_sec`, `output_path`, `raw_path` (native pre-upscale), `error`.
- `log` — rolling buffer of helper stdout lines (last ~50 jobs).

**Polling loop:** GET `/status` every 15–30s. A job is terminal when its `status` is one of `done`, `failed`, `cancelled`, `error`. Until then it's `queued` or `running`.

**Encoding caveat:** the log buffer can contain literal control characters from prompts with embedded newlines. Strip them before JSON-parsing:

```python
import re, json
raw = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', ' ', raw)
data = json.loads(raw, strict=False)
```

---

## LoRAs

### `GET /loras`

Returns `{"user": [...], "curated": [...], "loras_dir": "...", ...}`. Each entry has `id`, `name`, `path`, `filename`, `size_bytes`, `recommended_strength`, `kind` (`train_character` for LoRAs trained via the panel; `null` otherwise), and CivitAI metadata if known.

### `POST /loras/refresh`

Re-scan `mlx_models/loras/`. Returns the same payload as `/loras`.

### `POST /loras/delete`

| Field | Type | Note |
|---|---|---|
| `id` | string | LoRA id from `/loras`. |

Deletes the file. Returns `{"ok": true}`.

---

## Training

### `POST /train/upload`

`multipart/form-data` upload. Adds images to a training dataset.

| Field | Type | Note |
|---|---|---|
| `train_job_id` | string | Optional — autogenerated if omitted. |
| `files[]` | files | One or more PNG/JPG/JPEG/WEBP/BMP. |

Returns the dataset state (`train_job_id`, image count, etc.).

### `POST /train/start`

Kick off a character LoRA training. The panel shells out to `lora_lab.train_character` and streams progress events back into `STATE['log']`.

| Field | Type | Note |
|---|---|---|
| `train_job_id` | string | From `/train/upload`. |
| `trigger` | string | Trigger token, e.g. `bizarrotrn`. Must be compound and rare so it doesn't collide with normal language. |
| `preset` | `quick` \| `medium` \| `high` | Hyperparameter preset. `high` = rank 32, 5000 steps. |
| `image_count` | int | Confirmed by the panel from the uploaded images. |

Advanced overrides (all optional, fall back to preset defaults):

| Field | Type |
|---|---|
| `rank`, `alpha`, `steps`, `lr`, `resolution` | int / float |
| `caption_strategy` | `class_word` \| `trigger_only` \| `auto_caption` |
| `crop_strategy` | `center` |

Returns `{"ok": true, "id": "j-<...>"}`. Track progress via `/status` (the training job appears in `current` then `history`).

### `POST /train/install`

After training succeeds, the LoRA is automatically copied into `mlx_models/loras/`. This endpoint is for explicit re-install / metadata refresh.

### `POST /train/delete`, `POST /train/remove-image`

Delete training datasets / individual training images.

---

## Image generation (separate from video)

### `POST /image/generate`

Generates still images via the bundled image engine (mflux, BFL, mock backend). Pluggable per call.

`Content-Type: application/json`. Payload:

```json
{
  "prompt": "...",
  "n": 4,
  "engine_override": "mflux",
  "aspect": "1:1",
  "refs": ["/path/to/ref.png"],
  "loras": [{"path": "...", "strength": 1.0}]
}
```

See `image_engine.py` for backend-specific options.

### `GET /agent/image/config`, `POST /agent/image/config`

Kept under the legacy `/agent/` path for backward compatibility with the Image Studio frontend. Read/write the image engine config (default backend, mflux family, BFL key, etc.). Despite the path, **this is the image-engine config, NOT the removed chat agent**.

---

## Output management

### `POST /output/delete`

| Field | Type | Note |
|---|---|---|
| `path` | string | Must be under `mlx_outputs/`. |

### `POST /output/hide`, `POST /output/show`, `POST /output/show_all`

Hide/unhide outputs from the gallery without deleting the file.

### `POST /output/open_folder`

Opens the output's containing folder in Finder (Mac).

---

## Helper / system

### `POST /helper/restart`

Restart the warm helper subprocess (used after settings changes that need a fresh process).

### `POST /stop_comfy`

Shut down a ComfyUI process if Phosphene started one.

### `GET /version/check`, `POST /version/pull`

Self-update against the GitHub repo.

### `POST /settings`

| Field | Type | Note |
|---|---|---|
| (any setting key) | string | See `state/panel_settings.json` for the canonical set. |

### `POST /prompt/enhance`

`Content-Type: application/json`. Pre-rewrites a prompt with Gemma. Used by the panel's "✨ Enhance" button. **Avoid when LoRA trigger words are present** — the rewriter can drop them.

```json
{ "prompt": "...", "mode": "t2v" }
```

Returns `{"original": "...", "enhanced": "..."}`.

### `POST /upload`

`multipart/form-data`. Uploads a reference image for I2V or keyframe modes. Returns `{"path": "<absolute path under panel_uploads/>"}`.

---

## Sidecar JSON format

Every completed clip writes a sidecar at `<output>.mp4.json` alongside the MP4. Schema (simplified):

```json
{
  "output": "/abs/path/to/clip_720p.mp4",
  "raw_output": "/abs/path/to/clip.mp4",
  "params": { /* the full params from /queue/add, plus seed_used, model, etc. */ },
  "command": "helper",
  "started": "YYYY-MM-DD HH:MM:SS",
  "elapsed_sec": 426.13,
  "video_duration_sec": 7.0,
  "fps": 24,
  "model": "dgrauet/ltx-2.3-mlx-q4",
  "queue_id": "j-<...>",
  "helper_elapsed_sec": 425.5,
  "output_codec": { "preset": "standard", "pix_fmt": "yuv420p", "crf": "18" },
  "memory_policy": { "requested": "auto", "effective": "auto", "frames": 169, "tier": "standard", "pressure_pct": 30, "swap_gb": 3.9 }
}
```

Read sidecars to compare runs without re-querying the panel.

---

## Production recipe (validated 2026-05-15)

For character-LoRA video work, the validated locked recipe:

```
mode:               t2v
width × height:     1024 × 576   (also validated: 736 × 416 for ~2× faster wall)
frames:             169 (7s) / 241 (10s) / 361 (15s)
quality:            high          ← CRITICAL — forces Q8 dev transformer
temporal_mode:      native
stage1_steps:       10
stage2_steps:       3
teacache_thresh:    1.8–2.0       (plateau speedup; below 1.6 = no speedup)
cfg_scale:          3.0
bongmath_max_iter:  100
upscale:            off | fit_720p (lanczos)
accel:              off
enhance:            false         ← critical when trigger words are in the prompt
seed:               42 for A/B, -1 for production variety
LoRAs:              <trigger>_v2.safetensors + <trigger>.audio.safetensors, both @ 1.0
```

Wall on M4 Max 64 GB: ~7:06 per 7s clip at 1024×576, ~3:24 at 736×416.

---

## Worked example — one agent driver

Submit a clip with two LoRAs, poll until done, print the output path. Pure stdlib.

```python
import json, time, re, urllib.parse, urllib.request

PANEL = "http://127.0.0.1:8199"

def submit(prompt, loras, **overrides):
    form = {
        "mode": "t2v",
        "prompt": prompt,
        "width": "1024", "height": "576", "frames": "169", "steps": "8",
        "seed": "-1",
        "quality": "high", "temporal_mode": "native",
        "stage1_steps": "10", "stage2_steps": "3",
        "teacache_thresh": "1.8", "cfg_scale": "3.0",
        "bongmath_max_iter": "100", "accel": "off",
        "enhance": "false",
        "upscale": "fit_720p", "upscale_method": "lanczos",
        "loras": json.dumps(loras),
    }
    form.update({k: str(v) for k, v in overrides.items()})
    data = urllib.parse.urlencode(form).encode()
    req = urllib.request.Request(f"{PANEL}/queue/add", data=data, method="POST")
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())["id"]

def wait_terminal(job_id, timeout=1800):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with urllib.request.urlopen(f"{PANEL}/status", timeout=15) as resp:
            raw = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", resp.read().decode())
        d = json.loads(raw, strict=False)
        for src in ([d.get("current")] if d.get("current") else [], d.get("queue") or [], d.get("history") or []):
            for j in src:
                if not j or j.get("id") != job_id:
                    continue
                st = j.get("status")
                if st in ("done", "failed", "cancelled", "error"):
                    return j
        time.sleep(20)
    raise TimeoutError(f"job {job_id} did not finish")

if __name__ == "__main__":
    LORAS_DIR = "/Users/salo/pinokio/api/phosphene-dev.git/mlx_models/loras"
    job_id = submit(
        prompt="Cinematic close-up of bizarrotrn man in a wood-paneled study, photorealistic.",
        loras=[
            {"path": f"{LORAS_DIR}/bizarrotrn_v2.safetensors", "strength": 1.0},
            {"path": f"{LORAS_DIR}/bizarrotrn.audio.safetensors", "strength": 1.0},
        ],
    )
    print(f"queued: {job_id}")
    result = wait_terminal(job_id)
    print(f"status={result['status']} wall={result['elapsed_sec']}s output={result.get('output_path')}")
```

That's the full pattern: submit → poll → read output path.

---

## What was removed (and what replaced it)

The in-panel agentic flows feature (chat-driven shot planner) was removed 2026-05-15. Pre-removal snapshot is durable on GitHub as tag `pre-agent-removal-2026-05-15`.

Everything that the chat used to do — plan multiple shots, generate candidate anchor images, render I2V conditioned on those anchors, manage sessions — is achievable by an external agent through the API documented above. Specifically:
- Shot planning: agent-side text reasoning, no panel involvement.
- Anchor generation: `POST /image/generate`.
- Anchor-to-video: `POST /queue/add` with `mode=i2v` and `image=<anchor_path>`.
- Session state: agent maintains it externally.

External agents (Claude Code, Codex, OSS) have full panel functionality through this API surface.
