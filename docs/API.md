# Phosphene HTTP API

External agents (Claude Code, Codex, OSS, or anything that speaks HTTP) drive Phosphene through this API. The in-panel chat was removed 2026-05-15 — there's no agent embedded inside the panel anymore. Everything that the chat used to do can be done by an external agent through these endpoints.

## Server

- **Base URL:** `http://127.0.0.1:8198` on a normal install. **`8199` is the DEV
  profile's port**, and every example on this page uses it because that is the
  panel these were written against. The rule is one line in the source
  (`mlx_ltx_panel.py:275`): `DEFAULT_PORT = 8199 if PROFILE == "dev" else 8198`,
  overridable with `LTX_PORT`. If a request here returns nothing, you are
  almost certainly talking to a port that has no panel on it — try the other one.
- **Process:** `mlx_ltx_panel.py` (in the repo root)
- **Wire format:** form-encoded `POST` bodies; JSON exceptions are noted
  per-endpoint. Responses are **HTTP/1.0** — `BaseHTTPRequestHandler`'s
  default, never overridden — so the connection closes after every
  request and there is no keep-alive to pipeline against.
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
| `frame_rate` | float | Default `24`. LTX is trained at 24 fps; deviation degrades quality. |
| `character_strength` | float | Default `1.0`. The FACE LoRA's strength, 0–2. |
| `character_voice_strength` | float | Default `1.0`. The VOICE LoRA's strength, 0–2, applied to `<trigger>.audio.safetensors` only. It is a separate number from the face because the face file's audio-branch deltas are noise and are louder than the voice file's signal at equal strength. The measurement argues for running it hotter (1.2 is parity, 1.4 has real headroom); a graded A/B said otherwise, and `1.0` won. Raise it in small steps only when a specific voice is being drowned. |
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
| `image` | path | I2V only. Absolute path to a reference PNG/JPG. **Required** — since v4.6.1 an omitted or empty `image` on `i2v` / `i2v_clean_audio` is refused at job start with a readable error instead of defaulting to `examples/reference.png`, a demo file no install has ever shipped (that default was the widest-spread render failure in the fleet: `image not found: <path>`, 35 events across 22 people in 14 days). `LTX_DEFAULT_IMAGE` still supplies a default when it points at a file that actually exists. |
| `audio` | path | Optional audio reference. |
| `label` | string | Optional UI label for the queue card. |
| `temporal_mode` | `native` \| `fps12_interp24` \| `windows`. **`windows`** renders past one 121-frame window as a SEQUENCE — one `generate`, then one `extend` per later window on the kept tail of the last (`ltx_windows.py`: `stride = window - discard - overlap`, `count = 1 + ceil((total - window + discard) / stride)`), trimmed to the asked length. Needs the Q8 add-on and a tier that allows Extend; refused with a sentence otherwise. Stored on the job as `long_mode`. |
| `window_prompts` | JSON array (or newline-separated) of one line per window; line 1 blank = the prompt; a later blank line holds the previous moment. Each later window's prompt is its line FIRST, then the continuation clause, then `Throughout: <window_invariants>.` — the H3 chain contract, and the lead-with-the-move rule. |
| `window_invariants` | What must not change between windows (who, where, light, lens). Re-injected into every later window. |
| `loras` | JSON-encoded array | See LoRA payload below. |

**LoRA payload:** the `loras` field is a JSON-encoded array of `{path, strength}` objects. Stack as many as needed: On Hailuo H3, stacking needs the engine's repeatable `--lora` (`/status.h3.loras.max_stack` > 1); an older pack takes one adapter and the `h3_lora_slot` field decides whether it goes to Turbo or to the LoRA.

```json
[
  {"path": "<panel-root>/mlx_models/loras/ariatrn_v2.safetensors", "strength": 1.0},
  {"path": "<panel-root>/mlx_models/loras/ariatrn.audio.safetensors", "strength": 1.0}
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
  --data-urlencode 'loras=[{"path":"<panel-root>/mlx_models/loras/bizarrotrn_v2.safetensors","strength":1.0},{"path":"<panel-root>/mlx_models/loras/bizarrotrn.audio.safetensors","strength":1.0}]'
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

### One take — `take_seconds` + `beats` on `POST /queue/add`

A clip longer than a single pass, on either engine. `take_seconds` is one of
30 / 45 / 60 / 90 / 120; `beats` is a JSON list (or newline text) with one
prompt per five seconds — a blank beat holds the previous moment, extras are
dropped. The take overrules the engine's own length field:

| engine | what make_job does | worker |
|---|---|---|
| `ltx` | `long_mode=windows`, `frames = seconds·24+1`, `window_prompts = beats` | the windows chain (needs the Q8 pack) |
| `h3` | `h3_length=15s`, parts of three beats | `run_take_job_inner`: one ordinary H3 render per part, each starting from the last frame of the one before (`--first-frame`), parts hidden from the gallery, joined into `<name>_take<seconds>s.mp4` with a sidecar carrying `take.{seconds,beats,parts}` |

`params.take` on the job: `{seconds, beats, parts, frames, engine, beat_prompts}`.
Load Params restores a take as a take.

### `GET /take/estimate`

`?engine=h3|ltx&quality=<engine quality key>&seconds=<take_seconds>` →
`{ ok, seconds, beats, parts, engine, frames, minutes, eta, needs_q8 }`. Minutes
come from the same cost model as every other estimate (H3: the measured 15 s
cell × parts); `null` where this Mac has no number (LTX windows are unpriced).

### `GET /status`

Every `outputs[]` row carries `q`: the output's searchable words, lower-cased, built from its sidecar in the listing pass that already reads it — prompt, label, mode, quality, engine, character, seed, `WxH`, `Nf`, LoRA file stems, the model directory name, the temporal mode and `windows`. The gallery's search box is a substring test over `name + q`; every typed word must match, and typing pulls the older outputs in so the search is over everything.

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

**Polling loop:** GET `/status` every 15–30s. A job is terminal when its `status` is one of `done`, `failed`, `cancelled`, `stopped`, `error`. Until then it's `queued` or `running`.

`stopped` means the user stopped the render on purpose (see `POST /stop?mode=early`). It is **not** a failure: nothing was saved because the clip was never finished, but nothing went wrong. `stopped_reason` carries the runner's own sentence.

#### `current.progress`

Present while a job runs.

```jsonc
{
  "pct": 41, "phase": "denoise", "phase_label": "Denoising · step 5 / 10",
  "elapsed_sec": 62, "remaining_sec": 84, "eta_sec": 146,
  "preview": {
    "url": "/image?path=…/live/preview_latest.png&t=1786…",  // cache-busted server-side
                          // /image, NOT /file: the preview lives under STATE_DIR and
                          // /file serves OUTPUT only. Writing /file here is what shipped
                          // a correct preview to a broken-image glyph in every client.
    "estimate": 4, "total": 12,
    "meaningful": true,   // THE gate — the server decides, the client renders
    "abortable": true,    // meaningful && the runner published an abort sentinel
    "saves_sec": 182      // remaining_sec at the moment it became abortable
  },
  "preview_url": "/image?path=…"  // top-level alias of preview.url, same /image
                                  // form, present only once meaningful
}
```

`preview` is absent unless the live preview is on (Settings → Memory / speed) and the tiny decoder is installed. **`meaningful` is computed server-side and must not be re-derived**: the rule differs per pipeline — estimate 6 of 8 on the distilled schedule, whose first estimates are still essentially noise; estimate 2 on the two-stage HQ path, which evaluates the denoiser twice per stage-2 step. A client counting estimates would have to know which schedule is running.

#### `POST /stop?mode=early`

Asks the running render to stop at the next forward boundary by touching an `ABORT` sentinel. The runner exits **75** — a distinct code meaning "the user stopped this", not a crash — and the job resolves `status: "stopped"` with no output file. Returns `404` when nothing is rendering and `409` when the render has no live preview to stop through (use the hard `POST /stop`, which kills the helper).

**Encoding caveat:** the log buffer can contain literal control characters from prompts with embedded newlines. Strip them before JSON-parsing:

```python
import re, json
raw = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', ' ', raw)
data = json.loads(raw, strict=False)
```

---

## LoRAs

### `GET /loras`

Returns `{"user": [...], "curated": [...], "loras_dir": "...", ...}`. Each entry has `id`, `name`, `path`, `filename`, `size_bytes`, `recommended_strength`, `kind` (`train_character` for LoRAs trained via the panel; `null` otherwise), `adapter_verdict` (see below), and CivitAI metadata if known.

#### `adapter_verdict` — whether this file can do anything

A LoRA can train to completion, write a valid safetensors, load without a warning, and change **nothing**: the deltas it learned are numerically too small to move the model. Every structural gate passes, because every structural gate asks whether the KEYS land and none asks whether there is anything in them (#61, #62).

`adapter_verdict` (`_adapter_verdict`) is read out of the sidecar's `adapter_strength` block and is one of:

| Value | Meaning |
|---|---|
| `ok` | Measured, and the delta RMS is in the working band. |
| `weak` | Measured, and the deltas are at or under `WEAK_DELTA_RMS` (`2.0e-4`). The file is kept; the render will very likely look LoRA-free. |
| `unknown` | Trained by a build that did not measure. **Not a synonym for `weak`** — silence is not weakness, and every LoRA older than the measurement reads this way. |

Anything else is whatever the trainer called it, passed through verbatim.

Callers should surface `weak` **before** a render is spent finding out, and should not decorate `unknown`.

`GET /train/list` carries the same verdict plus **`adapter_advice`** — one sentence naming what to do about it, empty when the verdict is `ok` or `unknown`. It is hardware-aware: on a sub-64 GB Mac it does **not** say "train again on High", because that machine cannot reach the graded recipe (see the profile section below). The panel's Train tab renders it as a banner above the trained-character chips and in the finished job's history row; a verdict a user cannot act on is a measurement, not an answer (#62).

### `GET /hf/loras`

Hugging Face as a browsable source beside CivitAI. `?lane=h3|ltx&q=<a name, author:someone, or owner/repo>&refresh=1`.
Phosphene names no org: the query is a search (empty = the lane's default search term), an author listing, or one repo;
results are kept by lane from the repo name and tags, and each repo's largest `.safetensors` plus its example clip become one item in the same shape the CivitAI grid renders
(`name`, `creator`, `likes`, `size_kb`, `preview_url` / `preview_type`, `download_url`, `filename`, `base_model`, `hf_url`, `source: "huggingface"`).
Cached ten minutes per lane and query.

### `POST /hf/loras/download`

Form `repo`, `filename`, `meta` (the item JSON). Downloads through `huggingface_hub` (the saved HF token when present)
into the lane's LoRA directory, runs the H3 layout probe, and writes the same sidecar a CivitAI install writes plus
`source`, `hf_repo`, `hf_url`. Returns `{ ok, skipped, name, path, sidecar_path, lane, layout, converted }`.

### `GET /loras/updates`

Asks CivitAI whether any installed LoRA with a `civitai_id` in its sidecar has a
newer version (`modelVersions[0].id` ≠ the sidecar's `civitai_version_id`). One
request per such LoRA; nothing is downloaded.

```jsonc
{ "ok": true, "checked": 3,
  "items": [ { "path": "...", "name": "...", "current_version_id": 1, "latest_version_id": 2,
               "latest_version_name": "v2", "published_at": "...", "base_model": "LTXV 2.3",
               "download_url": "...",            // what /civitai/download takes
               "meta": { ... } } ],             // the sidecar the download will write
  "errors": [ { "path": "...", "error": "..." } ] }
```

Installing the update is the ordinary `POST /civitai/download` with that
`download_url` + `meta`; the new file lands beside the old one, which stays
until deleted.

### `POST /loras/guide`

Form: `path` (an installed LoRA from `GET /loras`). The planner model writes a
one-paragraph guide — what the LoRA does, how to prompt it, a strength to start
from, what it fights with — from the sidecar's own name / description / trigger
words, and saves it in the sidecar under `guide`; `GET /loras` rows then carry
`guide`. Refused with 409 while a render runs (the planner is a 12B model).
Returns `{ ok, guide }`.

### `POST /loras/refresh`

Re-scan `mlx_models/loras/`. Returns the same payload as `/loras`.

### `POST /loras/delete`

| Field | Type | Note |
|---|---|---|
| `id` | string | LoRA id from `/loras`. |

Deletes the file. Returns `{"ok": true}`.

### `POST /h3/loras/import`

`multipart/form-data`. Installs one **Hailuo H3** adapter you already have into H3's own
LoRA library (`mlx_models/hailuo-h3/loras/`, or wherever the installed pack keeps it).
Deliberately NOT `/upload`: reference media can be any image or audio, whereas an H3
adapter has to pass the runner's layout contract before it may appear in the picker.

| Field | Type | Note |
|---|---|---|
| `file` | file | Exactly one `.safetensors`. The filename is reduced to its basename and sanitised; it may not collide with a file already in the library. |

The body is **streamed** to a hidden staging directory — not buffered — so a multi-gigabyte
adapter costs about 1 MiB of resident memory. `Content-Length` is required and capped at
`H3_LORA_UPLOAD_MAX_BYTES` (4 GiB).

What happens to the file, in order: staged under `.import.<pid>.<tid>/<name>` →
`_h3_lora_prepare` (a ComfyUI repack has its `diffusion_model.` namespace stripped in
place, a header-only rewrite) → the import-only payload and target-module checks →
`os.replace` into the library → sidecar JSON. A refusal at any step leaves nothing behind.

Returns:

```json
{
  "ok": true,
  "path": "/abs/path/in/the/h3/library.safetensors",
  "filename": "my_lora.safetensors",
  "name": "my lora",
  "layout": "bare" | "comfyui",
  "converted": false,
  "pairs": 208,
  "recommended_strength": 1.0,
  "scale_source": "folded" | "per_module" | "uniform" | "advisory" | "none",
  "scale_evidence": "…"
}
```

`recommended_strength` is written into the sidecar and picked up by the picker's per-LoRA
strength control. The H3 loader applies `B @ A` and no alpha, so an adapter whose PEFT
`alpha/rank` is not already folded into `lora_B` needs that number here — this is reported,
never used as grounds for refusal.

Errors: `411` when `Content-Length` is absent or zero, `413` when the body exceeds the cap,
`400` for a bad file (wrong extension, empty, duplicate name, unusable layout, malformed
payload, wrong model), `500` otherwise. Every message names the file the user chose.

---

## Training

### `POST /train/upload`

`multipart/form-data` upload. Adds **one image** to a training dataset — this route is
one-file-per-request, not a batch. Send the images in a loop, echoing the returned
`train_job_id` back on every call after the first so the whole set lands in one directory.

| Field | Type | Note |
|---|---|---|
| `job_id` | string | Optional on the FIRST call (a fresh id is minted), required after, to keep the batch together. Note the name: `job_id`, **not** `train_job_id` — the response returns it as `train_job_id`. |
| `file` | file | Exactly ONE PNG/JPG/JPEG/WEBP/BMP, or a `.txt` caption. |

The request body is capped at `TRAIN_MAX_BYTES_PER_IMAGE` (32 MB). Because that cap is on the
whole request and the route takes one file, a multi-file body is refused even when every
individual image is small.

Returns the dataset state (`train_job_id`, image count, etc.).

### `POST /train/upload-voice`

`multipart/form-data` upload. Attaches an optional voice clip to a training dataset. One clip per dataset — re-uploading overwrites.

| Field | Type | Note |
|---|---|---|
| `job_id` | string | Required. From `/train/upload` (a dataset must exist first). |
| `file` | file | `.wav`, `.mp3`, `.m4a`, or `.flac`, max 50 MB. Aim for 5–30 s, clean, single speaker. Duration is not validated server-side — the panel UI's `<audio controls>` preview is the user's confirmation. |

Returns `{"ok": true, "job_id": "...", "filename": "voice.wav", "path": "...", "size": <bytes>}`.

### `POST /train/remove-voice`

Form-encoded. Removes the voice clip attached to a dataset (no-op if none exists).

| Field | Type | Note |
|---|---|---|
| `train_job_id` | string | Required. |

Returns `{"ok": true, "job_id": "...", "removed": ["voice.wav"]}`.

### `POST /train/start`

Kick off a character LoRA training. The panel shells out to `lora_lab.train_character` and streams progress events back into `STATE['log']`. If `train_audio=true` is set, a SECOND subprocess (`lora_lab.train_audio`) chains after the face phase to train a paired voice LoRA.

| Field | Type | Note |
|---|---|---|
| `train_job_id` | string | From `/train/upload`. |
| `trigger` | string | Trigger token, e.g. `bizarrotrn`. Must be compound, rare and **letters-only** so it doesn't collide with normal language — digits tokenize as single, very common tokens (`mrz07` → `m / rz / 0 / 7`), and every trigger that has carried an identity here was letters-only; the panel suggests `<3 consonants>trn` (`mrztrn`) and logs a note when a trigger carries digits (#62). The face LoRA lands at `mlx_models/loras/<trigger>_v2.safetensors`; the optional audio LoRA at `<trigger>.audio.safetensors`. |
| `preset` | `quick` \| `medium` \| `high` | Hyperparameter preset. **Default changed in v4.6.1: omitting it now gives `high` for `train_type=character` and `quick` for `train_type=style`** (`TRAIN_DEFAULT_PRESET`) — it used to give `quick` for both, which is the tier that has never carried an identity (#62). See the table below — **the step count is not a constant, and on a sub-64 GB Mac neither is anything else.** |
| `image_count` | int | Confirmed by the panel from the uploaded images. |
| `train_audio` | `true` \| `false` | Optional. Default `false`. When `true`, a voice clip must already exist for this `train_job_id` (uploaded via `/train/upload-voice`) — otherwise `/train/start` returns 400. |
| `audio_steps` | int | Optional. Default `250`. Number of audio-LoRA training steps. Heuristic: ~30 s/step on M4 Max. Smoke = `100`, Standard = `250`, Long = `500`. |
| `audio_rank` | int | Optional. Default `16`. LoRA rank for the audio adapter. |

Advanced overrides (all optional, fall back to preset defaults):

| Field | Type |
|---|---|
| `rank`, `alpha`, `steps`, `lr`, `resolution` | int / float |
| `caption_strategy` | `class_word` \| `trigger_only` \| `auto_caption` |
| `crop_strategy` | `center` |

Returns `{"ok": true, "id": "j-<...>"}`. Track progress via `/status` (the training job appears in `current` then `history`). During the run the `current.progress.phase_label` reads `Training face · step N / M` then `Training voice · step N / M` if the audio phase runs. If audio training fails, the face LoRA is still kept; `error` is set with `audio training failed: …` so callers can surface the partial success.

#### What the presets actually are

Steps are **derived**, not fixed: `steps = epochs × image_count`, floored at 1 and capped by the preset's `max_steps` (`_preset_steps_for`, `mlx_ltx_panel.py:1629`). The cap is part of the server contract — a hand-written API call must not be able to enqueue a multi-thousand-step run on a memory-constrained Mac.

On a Mac with **64 GB or more** (`TRAIN_PRESETS`, `mlx_ltx_panel.py:1558`):

| `preset` | epochs | rank | resolution | `max_steps` | Graded for identity? |
|---|---|---|---|---|---|
| `quick` | 30 | 8 | 512 | 8000 | **No.** The pill says "identity ungraded". |
| `medium` | 60 | 16 | 576 | 12000 | **No.** Same. |
| `high` | 100 | 32 | 512 | 20000 | **Yes** — the validated v2 recipe (Aria_v2, Bizarro_v2). The default, and the one the pill badges "Recommended". |

Measured with `lora_compat.py`: rank-32 adapters that carry an identity sit at 5.4e-04 to 1.6e-03 `delta_rms`; two independent users' rank-8 `quick` runs measured **1.54e-04** and **1.98e-04**, both at or under the `WEAK_DELTA_RMS` floor of 2.0e-04 (#62).

So `high` at 50 images is 5000 steps, at 20 images 2000, at 300 images 20000 (capped). Only the rank-32 recipe has ever been put in front of a human eye and called good.

#### ⚠️ Under 64 GB, every preset is quietly a different preset

`_select_train_profile` (`mlx_ltx_panel.py:1650`) runs once at import and, when total RAM is **`0 < ram < 64` GB**, it **mutates `TRAIN_PRESETS` and `TRAIN_STYLE_PRESETS` in place**. The names do not change. The numbers all do:

| `preset` | epochs | rank | resolution | `max_steps` | target modules |
|---|---|---|---|---|---|
| `quick` | 2 | 4 | 384 | 120 | compact set |
| `medium` | 5 | 8 | 384 | 300 | compact set |
| `high` | 10 | 8 | 448 | 500 | compact set |

`make_job` mirrors the same clamps server-side, so `rank` / `steps` / `resolution` passed as advanced overrides are **capped too** (`max_rank` 8, `max_resolution` 448, `max_steps` 500) — you cannot ask for the graded recipe on this hardware and get it.

This is a deliberate trade and the reason for it is real: full 512 px LTX LoRA training materialises backward activations for the dev transformer and a 48 GB Mac falls into swap thrash and never finishes. But be clear about what it means: **on a sub-64 GB Mac there is no preset that trains the recipe anybody has graded.** rank 8 at ≤500 steps is exactly the regime `adapter_verdict` was built to catch, so expect `weak` more often here, and read it as "this hardware, this recipe" rather than "this dataset".

`GET /status` exposes the active profile as `train_profile` (`key`: `compact_training` | `full_training`, plus `label`, `ram_gb`, `max_rank`, `max_resolution`, `max_steps`, `note`) **and the effective tables themselves** as `train_presets` / `train_style_presets` (`mlx_ltx_panel.py:21842-21844`). **Read those before trusting a preset name** — they are the post-mutation values, so they are the only place the two tables above are told apart at runtime.

#### The `adapter_strength` trainer event

Not an analytics event — nothing about it leaves the machine (see [docs/ANALYTICS.md](ANALYTICS.md) for the complete list of what does). It is a progress event on the lora_lab subprocess stream, emitted once after the trainer measures the file it just wrote (`mlx_ltx_panel.py:18005`). The panel logs the median delta RMS, the max, and how many modules are carrying, then **finishes the job in a WARNING state rather than a bare `done`** when the verdict is not `ok`. The verdict is written into the LoRA's sidecar as `adapter_strength: {"verdict": …}` and read back out as `adapter_verdict` on `/loras` and `/characters`.

### `POST /train/install`

After training succeeds, the LoRA is automatically copied into `mlx_models/loras/`. This endpoint is for explicit re-install / metadata refresh.

### `POST /train/delete`, `POST /train/remove-image`

Delete training datasets / individual training images.

---

## Characters

Discover and drive trained character LoRAs as bundles.

### `GET /characters`

Returns `{"characters": [...]}`. Each entry describes one character discovered in `mlx_models/loras/`:

| Field | Type | Note |
|---|---|---|
| `id`, `trigger` | string | Same value — the rare trigger token. |
| `name`, `pronoun`, `subject_noun` | string | From the optional `mlx_models/characters/<trigger>/bundle.json`. |
| `face_lora_path` | string | Absolute path to `<trigger>_v2.safetensors`. Always present. |
| `audio_lora_path` | string \| null | Absolute path to `<trigger>.audio.safetensors`. `null` when the character is silent. |
| `audio_lora` | string \| null | Alias for `audio_lora_path` — preferred for new callers. |
| `voice_sample` | string \| null | Absolute path to `<trigger>.voice.<ext>` (the original training clip). For playback / inspection only; the model uses the audio LoRA, not the raw clip. |
| `has_voice` | bool | `true` iff the audio LoRA is on disk. Silent characters (face LoRA only) are returned with `has_voice: false`. Callers should skip audio cues in prompts and not stack the (absent) audio LoRA. |
| `sample_image_url` | string \| null | URL to a preview image from the training dataset. |
| `sheet_image_path`, `sheet_image_url` | string \| null | The generated multi-view character sheet (`mlx_models/characters/<trigger>/sheet.png`), when one exists. Made by `POST /characters/<id>/sheet/generate`, served by `GET /characters/<id>/sheet`. |

Discovery rule: `<trigger>_v2.safetensors` is required; everything else is optional.

### `GET /characters/<id>/preview`

Serves the sample training image for the character (PNG/JPEG).

### `GET /characters/<id>/sheet`

Serves the generated character sheet PNG. `?w=<px>` returns a cached thumbnail (JPEG) through the same resize lane as `/image?w=`. 404 until a sheet has been generated.

### `POST /characters/<id>/sheet/generate`

Renders a multi-view turnaround sheet from the character's reference image (bundle avatar first, training-sample fallback) and composites it into `mlx_models/characters/<id>/sheet.png` + a `sheet.json` sidecar (`phosphene/character_sheet@1`). Synchronous, like `/image/generate` — the caller blocks for the whole render; 429 when the GPU is already held by a render/training/image job (never queued).

`Content-Type: application/json`. Every field optional (an empty body renders the default 3-view sheet):

```json
{
  "engine_override": "hidream_inline",
  "views": ["front", "profile_left", "three_quarter"],
  "wardrobe": "a red flight jacket",
  "seed": -1
}
```

Only ref-honoring engines are accepted (`hidream_*_inline`, `qwen_edit_*_inline`, `mock_inline`) — a text-only engine would render a stranger. `seed >= 0` gives view *i* `seed + i` so a whole sheet reproduces from one number. Every view prompt pins wardrobe to the reference ("wearing exactly the same clothes as in the reference image"); `wardrobe` re-states the outfit in words on top of that. Views after the first also chain the first rendered view as a second reference: measured on a dim reference image, solo side-angle renders re-imagined hair color across seeds and phrasings, and a clean frontal from the same run re-anchors the attributes the raw reference under-specifies. Per-view framing can still drift (a subject off-center in one view) — regenerate or vary the seed. On success, `bundle.json`'s `preview` is pointed at `sheet.png` **only** if it was null — a curated preview is never clobbered.

### `POST /characters/<id>/generate`

Assembles the locked production recipe + queues a T2V render. The endpoint accepts `prompt_body`, `framing`, `duration`, `quality`, and an optional integer `seed` (`-1` means random, matching `/queue/add`). For exact replay/debug calls, callers may pass `full_prompt` plus HQ override fields such as `stage1_steps`, `stage2_steps`, `teacache_thresh`, `cfg_scale`, `bongmath_max_iter`, `video_skip_step`, and `audio_skip_step`. The audio LoRA is only stacked when `has_voice` is true.

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

## Storyboard — plan a film, then shoot it

A **board** is a film: a concept, a cast, locations, and a numbered shot list. Every render it starts goes through the same `make_job` → queue contract as `/queue/add`; none of these routes has a private execution path. Boards live under `state/storyboards/<board_id>/`, with `storyboard.json` written atomically — renders run for hours and a torn board would lose the whole run.

### Read

| Route | Returns |
|---|---|
| `GET /storyboard/list` | Every board, newest first. |
| `GET /storyboard/get?id=<bid>` | One board document. |
| `GET /storyboard/films` | The assembled films on disk — player path, runtime, picture size, size on disk, when it was made. `list_outputs` globs `OUTPUT/*.mp4` and never descends, so without this the one thing the feature makes is the one thing the app cannot show you. |

### Brief switches (on `plan`)

| Field | Board key | Effect |
|---|---|---|
| `auto` | `auto` | after the plan: render every shot, cut, assemble — no more buttons |
| `anchor_stills` | `anchor_stills` | before a text/character shot renders, an ordinary image job makes its first frame (`still_job_id` → `still`; the character's sheet is the reference through `qwen_edit_inline` when it exists) and the clip renders as **i2v · anchor** from it. A still that fails sets `still_error` and the shot renders unanchored, once |
| `long_windows` | `long_windows` | an LTX shot longer than 121 frames renders as `temporal_mode=windows` (a chain on the Q8 dev transformer) with the board's style and the shot's location as `window_invariants`, instead of being cut to fit. Extend cannot re-inject an image per window, so an anchored long shot is anchored by its first window only |
| `take_seconds` | `take_seconds` | **one take**: the film is one shot of 30 / 45 / 60 / 90 / 120 s. The planner writes one beat per 5 s (`_sb_take_concept`), the board keeps ONE shot with `beats` (`collapse_take`), the card shows the beats for editing, and `shot_to_job` posts `take_seconds` + `beats` — the Video tab's take, so the render is the same on both doors. `0` turns it off |

Each field is patched only when sent; an absent field keeps the board's value.

`POST /storyboard/restill` — form `id`, `n`, optional `pass`: forgets shot `n`'s
still and the clip rendered from it, turns `anchor_stills` on if it was off,
and renders that shot alone (the still first, then the clip from it). Same
refusals as `render` (planning, already rendering).

### Write — `POST /storyboard/<action>`

**The Director.** `plan` accepts `soundtrack` (an audio file on this Mac) and `bars_per_shot` (1/2/4, default 2). With a track the plan is a MUSIC VIDEO: `beat_map()` — the same fit the Editor's Prepare runs — turns the downbeats into slots of `bars_per_shot` bars, the shot count is the slot count (the `shots` chips stand down; capped at `STORYBOARD_MAX_SHOTS` with a note), the planner's brief carries the bpm, the slot length, an arc instruction and two laws (begin every description with the movement; no dialogue, the track replaces every clip's sound), and each planned shot's `duration_s` is its slot plus a one-second handle, with `slot: {start, end}` on the shot. **The song map decides how often.** `song_map()` (storyboard_edit) puts SECTIONS on the same bar grid — `{start, end, label, energy, brightness}` with labels intro / verse / chorus / bridge / outro from position, relative loudness and brightness — and `director_pacing()` turns `bars_per_shot` into a per-section stride: the chorus cuts twice as often, the intro and outro half as often. Each slot carries its `section`; the brief tells the planner which shots fall in which section and how hot it is; a track the map cannot read is cut at the base stride and the log says so. The board records `soundtrack: {path, bpm, bars_per_shot, total_sec, count, slots, sections}`, the Editor's first auto-cut uses that track when Prepare has not cached another — so the film opens already cut on the beat — and `GET /storyboard/edit` returns `sections` so the ruler can paint the arc. A bad path is a 400 before the planner runs; the planner stage `grid` is reported while the beat is read.

**Auto.** `plan` accepts `auto=1`: when the plan lands the draft render of every shot starts on its own (`_sb_auto_after_plan`, once the planner has given the memory back), and when the render thread finishes with every shot rendered the film is cut (`_sbe_auto_edit` — on the beat when there is a track) and assembled by the same assembler Render uses (`_sb_auto_film`); the board records `auto_film`. A shot that did not render leaves the film waiting and the log says how many. Stop still stops the render; the cut and the film are the ordinary ones and can be re-cut afterwards.

Form-encoded unless noted. `id` is the board id throughout.

| Action | Fields | Note |
|---|---|---|
| `plan` | `concept`, `id`, `notes`, `shots`, `must`, `engine`, `character_id`, `style`, `locations`, `wardrobe` | Plan or re-plan. **409 while the renderer holds memory or the one planner slot is taken.** `shots` is honoured or refused — never silently swapped for a different count. A re-plan PATCHES locations and wardrobe rather than overwriting them. |
| `cancel` | `id` | Release a planner slot. |
| `save` | JSON `{id, board}` | Write a hand-edited board. |
| `grade` | `id`, `n`, `grade`, `note` | Grade one shot. |
| `estimate` | `id`, `pass`, `only` | Cost/wall estimate before spending. |
| `render` | `id`, `pass`, `only`, `out`, `music`, `music_mode` | Shoot the board. |
| `stop` | `id` | Stop the running board render. |
| `replan-shots` | `id`, `ns` | Re-roll named shots. A repair that breaks a law is rejected; re-rolls splice into the CURRENT plan, not the original. |
| `add-shot` | `id`, `path` | |
| `import-shots` | `id`, `from`, `only` | REFERENCE clips from another board. Provenance (`imported_from`), the source LOCATIONS and the source CAST all travel with them — leaving any one behind breaks something specific. **Refuses while the target film is rendering.** |
| `export` | `id`, `auto_edit`, `music`, `target_seconds`, `music_mode` | |
| `reveal` | `id`, `name`, `what` | Show in Finder (macOS). |
| `delete` | `id` | |

### Board schema — the geography fields

A shot is composed from its location. Since 2026-08-18 a location can also be **faced from more than one direction**, which is what makes a reverse angle a reverse angle instead of a second location.

```jsonc
"locations": [
  {"id": "carwash", "name": "The car wash", "description": "…",
   "views": [                                  // ABSENT, never [], when there are none
     {"id": "establishing", "name": "…",
      "light": "camera left",                  // flips with the camera
      "description": "…"},                     // self-contained: readable without the floor plan
     {"id": "reverse", "name": "…",
      "light": "camera right",
      "description": "… no car in frame …"}    // says what it does NOT hold, in those words
   ]}
],
"shots": [
  {"n": 1, "location_id": "carwash",
   "view": "reverse",        // must be a view id OF that location
   "eyeline": "right",       // "left" | "right" | "lens"
   …}
]
```

| Field | Where | Rules |
|---|---|---|
| `locations[].views` | location | List of `{id, name, light, description}`. **Absent, not `[]`,** when a location has none — every board written before views existed has no key, and an empty list would be a new shape for the validator to have opinions about for no gain. Ids follow the same charset as location ids. |
| `shots[].view` | shot | A view id belonging to that shot's `location_id`. A view on a shot with no location, or a view that is not one of that location's, is a **validation error** (`unknown_view`) and fails exactly the way an unknown location does — a silently-dropped view composes the reverse angle from the establishing description, and the car the view existed to get out of frame is back in it with nothing said. |
| `shots[].eyeline` | shot | One of `left`, `right`, `lens` — a fact about the FRAME, not the room. `right` = the eyes go off past the right edge. Anything else is a validation error (`bad_eyeline`). |

**What the server does with them.** `shot_scene_text` injects the VIEW's description when a shot names one and falls back to the location's otherwise — so a board with no views, or a shot that names no view, renders exactly as it always did. `eyeline_clause` composes `"his eyes fixed past the right edge of frame"` at render time; **`lens` deliberately produces nothing** (telling these models to look down the barrel buys a stiffer performance than silence), and it exists so a shot can *say* it holds the lens, which is what lets the 180-degree check tell "faces the camera" apart from "nobody wrote one".

**The 180-degree rule is enforced, not suggested.** After planning, `_enforce_eyelines` walks the shots in *screen* order and flips the second of any two adjacent shots that cut between two **different** characters in the **same** location and both claim the same side, and says so in `warnings`. It repairs mechanically because `eyeline` is a discrete field with exactly one complement — flipping it cannot damage anybody's prose. Prose-level laws get a model round trip instead, precisely because they lack that property.

---

## Editor — the timeline

The Editor is a **top-level workflow**, not a stage of the storyboard: engine-agnostic, opens without a board being mid-render, and its document is `edit.json` under the board dir. One video track, one music lane. Per-clip sound is linked by default and can be pulled apart (J/L cuts) — that is **not** a second audio track.

### Read

| Route | Returns |
|---|---|
| `GET /storyboard/edit?id=<bid>` | The arrangement. On the first call for a board it runs the auto-edit ONCE, persists it into the automatic lane, and returns it with `generated: true`. Every later GET is a file read. A corrupt `edit.json` returns **500 with `corrupt: true`** and is never silently replaced — that would throw away the arrangement it is a broken copy of. |
| `GET /storyboard/edit/status?id=<bid>` | The prepare job's state. |
| `GET /storyboard/edit/peaks?id=<bid>` | The waveform. Hundreds of KB — fetch once, **never on a poll**. `404` until `prepare` has run. |
| `GET /storyboard/edit/proxy?id=<bid>&name=<file>.mp4` | A proxy video, served with **range support**. Safari will not seek a `<video>` whose server answers plain `200`, which would undo the entire reason proxies exist. `name` must match `[A-Za-z0-9._-]{1,120}\.mp4` — it is a basename the server minted, so anything else is a probe. |
| `GET /storyboard/edit/uploads` | Videos the user brought with them (`panel_uploads/timeline/`, newest first, capped at 200). Kept out of `OUTPUT` on purpose, which is why they need a route at all; images do not, they land in the library the gallery already walks. |
| `GET /storyboard/edit/drafts?id=<bid>` | `{active, drafts: [...], backup}` — see **Drafts** below. |
| `GET /storyboard/edit/versions?id=<bid>` | **Metadata only**: name, revision, when kept, clip count, duration, plus `keep` (= `EDIT_HISTORY_KEEP`, 50). The documents themselves are read once, by `restore`, on the user's word. Opening the panel must not be a download. |

### Write — `POST /storyboard/edit/<sub>`

| Sub | Fields | Note |
|---|---|---|
| `prepare` | `id`, `music`, `target_seconds` | Build proxies + peaks. `400` if `music` is not a file. |
| `add-clip` | `id`, `from`, `only`, `path`, `title`, `kind` | `kind` is `video` \| `still` \| `slug` — passing it is what stops an image landing as a video with no frames. |
| `relink` | `id`, `only` | Re-point clips at their finished (delivery) files. **A finished RETAKE is never part of that batch** — it is offered against its clip with `retake: true` in the payload's `relink` rows and adopted one clip at a time with `only=<clip id>`. |
| `cancel` | `id` | Cancel the prepare job. |
| `auto` | `id`, `music`, `target_seconds`, `min_shot`, `max_shot` | Re-run the machine's cut. Lands in the **automatic** lane. |
| `save` | JSON `{id, edit, expect_revision?}` | See below. |
| `draft` | `id`, `op`, `slug`, `name`, `from` | See **Drafts**. |
| `backup` | JSON `{id, edit}` | The quiet lane. See **Drafts**. |
| `recover` | `id` | Adopt the pending backup. |
| `discard-backup` | `id` | Throw it away. |
| `version` | `id`, `label` | Name a version. **Naming is not saving** — no revision bump, no write to `edit.json`, nothing about the timeline changes, which is exactly why it is safe to press at any moment. |
| `restore` | `id`, `file` | Restore a history entry. Refuses while the film renders. |
| `render` | `id`, `out`, `music`, `music_mode`, `format`, `size` | Assemble the film. `music_mode` is `replace` \| `under`. **Deliver as:** `format` is `h264` (default, the panel's preset) \| `hevc` (VideoToolbox, `hvc1`, about half the bytes) \| `prores` (422 HQ, 10-bit 4:2:2, PCM sound, a `.mov` no browser previews); `size` is `native` (as cut) \| `1080p` \| `2160p` — one Lanczos scale after the overlays, up only, never a crop, skipped when the cut is already that height. `finish` is `none` \| `grain` \| `heavy_grain` — moving film grain (`noise` t+u, strength 9 / 18) added AFTER the size so a 4K delivery gets 4K grain; a delivery-time treatment, never in the document, never in the preview or the export. A different delivery is a different file: `<slug>_film_hevc_1080p.mp4`, `<slug>_film_prores.mov`, `<slug>_film_1080p_grain.mp4`. The facts carry `deliver: {format, size, finish, label}`. |
| `export-nle` | `id` | FCP7 XML + After Effects JSX + linked media. |
| `reveal` | `id`, `what` | Show in Finder. |
| `generate` | `id`, `prompt`, `duration`, `duration_s`, `film_start`, `pass`, `character_id`, `seed`, `title`, `engine`, `trigger`, `retake_of` | Shoot a new clip straight into a slot. With `retake_of=<clip id>` it is a **retake**: the clip's own shot is CLONED (character, refs, location, engine) and only the prompt, the length and the seed change; the new shot carries `edit_slot.retake_of`, and when it renders the Editor offers it against that clip ("Use it" / "Keep the old one"). 400 if the clip is no longer on the timeline. |

**`save` semantics.** `revision` is the **server's** counter, taken from disk, never from what the client remembered — a stale tab must not be able to wind it backwards. Pass `expect_revision` to find out you were overtaken instead of silently overwriting the other tab: mismatch returns **`409` with `conflict: true`** and the current revision. Validation errors come back as **all of them at once** (`errors: [...]`, `400`) with the file on disk untouched, so a client can highlight every bad clip in one pass instead of playing whack-a-mole.

### `edit.json` schema

`version` is **2** (`EDIT_VERSION`). Version 1 is upgraded on READ only (`migrate_edit`) — bumping the version without a read-path upgrade would not refuse old builds, it would refuse every timeline anybody already had. A version from the future is left alone so the validator can say the honest thing.

```jsonc
{
  "version": 2,
  "board_id": "sb_…",
  "revision": 17,
  "origin": "manual",          // "manual" | "auto" | "backup" — who asked for this save
  "updated_at": 1786…,
  "duration": 43.5,
  "clips": [ /* below */ ],
  "audio": { /* below */ },
  "beats": null,
  "settings": {}
}
```

#### `clips[]`

| Field | Note |
|---|---|
| `id`, `path`, `proxy` | `path`/`proxy` are `null` on a slug. |
| `start`, `end` | The window INSIDE the source. |
| `film_start`, `film_end` | The slot on the film. **Never independent** — a clip plays at 1x, so `film_end - film_start == end - start` within `LENGTH_TOLERANCE` (2 ms). |
| `kind` | `video` (**absent means video** — stamping it on 400 clips would rewrite every `edit.json` on the machine to say what its absence already says), `still` (held for the length of its slot; `start`/`end` are synthesised from the slot, `duration: null`), `slug` (black, no file). |
| `source` | `auto` \| `human`. |
| `locked` | bool. |
| `adjust` | `{brightness}`, an ffmpeg `eq=brightness` additive offset clamped to ±`BRIGHTNESS_LIMIT` (0.5). **Neutral is absent** — dragging the slider back to zero leaves a document identical to one that never had a slider. |
| `frame` | `{zoom, x, y}` — a reframe. The picture is magnified `zoom` times (1–3) and the window is centred at the fraction (`x`, `y`) of the source. Render: a `crop` of the source's own pixels before the fit, so one string is right at every size; preview: a CSS scale about the same anchor (approximate); export: FCP7 Basic Motion (scale + centre) and AE scale + position. **Neutral is absent**; a slug has nothing to reframe. |
| `speed` | Play rate, `SPEED_MIN`–`SPEED_MAX` (0.25–4.0), video only. The slot is `(end - start) / speed`, so the 1x rule above reads "at its speed". Picture is `setpts`, sound is `atempo` chained past ffmpeg's 0.5–2.0 window, and the sound's envelope (`afx`) stays on the strip's PLAYED clock — a keyframe at 2 s is still at 2 s of the strip after a retime. **Absent is 1x.** Never set automatically. |
| `audio` | The split edit. See below. |

#### `clip.audio` — the J-cut and the L-cut

```jsonc
"audio": {"start": 3.5, "end": 7.0, "film_start": 12.25}
```

Video clips only. Same three numbers as the picture, for the SOUND — so the sound can start before its picture (J) or run past it (L), which is how a cut stops being a butt join.

**Absent means linked**, and that is the whole migration: every clip ever written has no `audio` key and every one of them plays its own sound under its own picture. `EDIT_VERSION` did not move for it and no document was rewritten.

**The PRESENCE of the field is the switch, not the values in it.** Do not derive "linked" from `audio` equalling the picture window: unlinking writes the window the clip already has, so a clip the user had just unlinked would read as linked and refuse to be dragged. The toggle adds the object or deletes it; nothing else decides. `normalise_edit` rounds the field but **never removes it** for the same reason, and strips it entirely from a still or a slug.

Audio windows may not overlap each other any more than the pictures may. This is still one video track and one music lane — a split edit is a butt join that lands somewhere else, not a mix.

#### `transitions[]` — a typed object that owns a BOUNDARY

```jsonc
"transitions": [{"id": "t1", "after_clip": "<clip id>",
                 "kind": "dissolve" | "fade_black", "duration": 0.5}]
```

A cross-dissolve is two clips in the same second, and the picture lane's one-clip-at-a-time rule (`clips_overlap`) is load-bearing — so a transition is **not** an overlap. It names the OUTGOING clip and sits on the cut between it and its successor in film order; one per boundary. **The clips' own `film_start`/`film_end` do not move** and the film stays exactly as long as the timeline says. The render pulls half the duration of extra tail from beyond the outgoing clip's out-point and half of extra head from before the incoming clip's in-point (source the trims left behind), splits the picture concat at the boundary and joins the two runs with `xfade` (`fade` / `fadeblack`) centred on the cut. The sound needs nothing new: it takes the lane path (`_sb_split_audio_plan`) exactly as a J-cut does.

`duration` is clamped on every read (`transition_duration`) to `min(asked, half the shorter neighbour, 2.0 s)` and snapped to an even number of frames; the document keeps the number typed. A side with no spare material is refused — `transition_no_handles` names the side and the shortfall ("clip 2 has only 0.10s before its in-point and the dissolve needs 0.25s there"). Every transition code (`transitions_shape`, `transition_shape`, `transition_unknown_clip`, `transition_last_clip`, `transition_duplicate_boundary`, `transition_kind`, `transition_duration`, `transition_no_handles`) is an **error**; none is in `WARNING_CODES`. `render` refuses with the same sentence. A still and a slug have all the handles anybody could ask for.

#### `overlays[]` — cards, mattes, and titles

`kind` is `still` | `video` | `text`. An explicit `kind` wins; with none, the path's suffix decides (`.png`/`.webp`/`.tif` are stills), so every overlay written before titles existed reads as it did. A **title** is an overlay whose pixels the render draws:

```jsonc
{"id": "o1", "kind": "text", "text": "FIN", "film_start": 40.0, "film_end": 44.0,
 "style": {"font_size": 96, "color": "#ffffff", "align": "center", "x": 0.5, "y": 0.8,
           "box": true, "box_color": "#000000", "box_opacity": 0.5},
 "fx": {"fade_in": 0.5, "fade_out": 0.5}}
```

`font_size` is px at a 1080-high frame and scales with the film; `x`/`y` are the anchor as fractions of the frame; `align` says which edge of the text sits on `x`. `overlay_text()` is the one accessor and the style is written only where it differs from the defaults. The font is a FILE — `LTX_TITLE_FONT`, else the first of `TITLE_FONT_CANDIDATES` present on the machine — verified before ffmpeg is built; with none, `render` answers 400 with a sentence rather than a film with a hole in it. The raster is a frame-sized RGBA PNG beside the film (`.titles/`) fed through the same overlay chain as an uploaded card, so a title inherits the lane's alpha, fades and z-order. Titles are drawn on the stage in the DOM at the same anchor and size; they do not travel in the NLE export (no path).

#### `audio` — the soundtrack, as an object on the timeline

```jsonc
"audio": {"path": "…", "duration": 180.0, "peaks": "peaks.json",
          "offset": -4.0, "trim_start": 12.0, "trim_end": 96.0}
```

| Field | Meaning |
|---|---|
| `offset` | The second of the TRACK that plays at film time 0. It has meant that since before the editor existed, so every document on disk keeps working. **It may now be negative**: the track begins `-offset` seconds into the film, with silence in front. |
| `trim_start` / `trim_end` | In/out points INSIDE the track, in track seconds. **Absent means untrimmed**, which is why an `edit.json` written before this is still valid: no field, no trim, same filtergraph. A trim at or past the end of the track is not a trim, and `normalise_edit` drops it — a handle dragged all the way back out leaves a byte-identical graph. |

`music_window(audio, duration=…)` is the single place these become the numbers ffmpeg and every exporter need: `start = max(trim_start, offset)` (a head trim and a positive offset are the same gesture from two directions; whichever cuts more wins), `end` or `None` for "play to the end", and `delay`/`film_start` = `start - offset`. **Trimming the left edge does not slide the rest of the track earlier** — that is a ripple, and music does not ripple — so the seconds a head trim removes come back as silence in front. A window the trims close entirely returns "no music" rather than a zero-length `atrim` ffmpeg refuses.

### Drafts, backups, and history — three different things

**Drafts** are named variations of the same film, in `drafts/` with an `index.json` naming the active one. A board that has only `edit.json` has exactly one draft and always did; it just had no name for it. `load_draft_index` names it on the way out rather than in a rewrite pass, so the upgrade cannot half-happen.

`POST /storyboard/edit/draft` is **one route with five verbs** — they are five edits to the same active pointer, and splitting them would be five places for that pointer to be wrong:

| `op` | Extra fields | Effect |
|---|---|---|
| `new` | `name`, `from=current` | New draft, empty or copied from what is on screen. Becomes active. |
| `duplicate` | `slug`, `name` | |
| `rename` | `slug`, `name` | |
| `delete` | `slug` | |
| `activate` | `slug` | Stashes the current active draft first — never loses it. |

`new`, `duplicate`, `delete` and `activate` return **`409` with `busy: true` while that film is rendering** — the render is reading this document clip by clip, and swapping the timeline under it is the same hazard `import-shots` and `restore` refuse.

**The backup is ours, the saving is the user's.** `POST /storyboard/edit/backup` writes `history/backup-<draft>.json` and **nothing else** — no `edit.json`, no revision, no conflict check. The user's saved draft stays exactly what he last saved. It is surfaced by `GET …/drafts` as `backup` and **offered**, never applied: `recover` adopts it, `discard-backup` throws it away.

**History is two lanes plus a keep lane**, told apart by filename prefix rather than by opening the files — a prune that has to read fifty documents to decide what to delete fails halfway on the first corrupt one:

| Prefix | What | Pruned? |
|---|---|---|
| `edit-r*` | An automatic snapshot. | Yes, capped at `EDIT_HISTORY_KEEP` (50). |
| `save-r*` | A save the user pressed. | **Never.** |
| `keep-r*` | A version the user named (`version`). | **Never.** |

Every save archives its predecessor before the new document lands (~5 KB each). History failing must never block a save — losing a breadcrumb is nothing next to refusing to persist the arrangement on screen. `origin` is also stamped inside the JSON, for anything reading one file on its own.

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

## Push alerts — the completion alert for a closed tab

The in-tab chime and browser alert (Settings → Completion alerts) need the page
open. Web Push does not: the browser keeps a subscription for this origin, the
panel signs each message with a VAPID key pair it generates once
(`state/vapid.json`), and the browser's push service wakes `/sw.js` to show the
notification. No relay: the panel posts straight to the browser vendor. Offered
only when `pywebpush` imports (`GET /settings` → `push_available`).

| Route | What |
|---|---|
| `GET /sw.js` | the service worker, served from the root scope with `Service-Worker-Allowed: /` |
| `GET /push/key` | `{ ok, available, public_key, subscriptions }` — the VAPID public key the browser subscribes with (503 when unavailable) |
| `POST /push/subscribe` | form `subscription` = the `PushSubscription` JSON; one entry per endpoint |
| `POST /push/unsubscribe` | form `endpoint` |
| `POST /push/test` | sends "This is what a finished render will say." to every subscriber; `{ ok, sent }` |

Every job that ends `done` or `failed` pushes `Phosphene — render done` /
`— a render failed` with the job's label (or the first 60 characters of its
prompt), off the GPU lock, only while `notify_done` is on. A subscription the
vendor reports gone (404 / 410) is dropped.

### `POST /settings`

| Field | Type | Note |
|---|---|---|
| (any setting key) | string | See `state/panel_settings.json` for the canonical set. |
| `notify_done` | `1`/`0` | Completion alerts: a chime in the tab when a render finishes or fails, and a browser notification when the tab is in the background and the person has allowed them. Default on; `GET /settings` returns it. |

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
  "model": "mlx_models/ltx-2.5-mlx-q4",
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
    # Adjust to wherever your Phosphene install lives:
    LORAS_DIR = "<panel-root>/mlx_models/loras"
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
