# Phosphene — agent operating manual

This file is the single source of truth for how an AI assistant should
work on this codebase. Read it first, every session, before touching
any file. The generic Pinokio scaffolding starts at "## Generic Pinokio
Development Guide" further down — that's the upstream template. This top
section overrides it where they conflict.

---

## 1. What Phosphene is

A free desktop panel that wraps Lightricks' **LTX 2.3** model running
natively on Apple's **MLX** framework, exposed as a one-click install
through **Pinokio**. Generates video and audio jointly in a single
forward pass on Apple Silicon Macs. Apple-Silicon-only.

Public artifacts:
- GitHub: `github.com/mrbizarro/phosphene` (public, MIT for the panel)
- Pinokio Discover: searchable as "Phosphene"
- Twitter handle: `@AIBizarrothe`

The panel itself is MIT. Model weights have Lightricks' own license.

## 2. Where to work

**The Pinokio install dir IS the canonical workspace.** There is no
separate dev folder. Old "local dev" copy was deleted to consolidate.

```
/Users/salo/pinokio/api/phosphene.git/        ← edit here
├── .git/                                       ← real working copy of mrbizarro/phosphene
├── ltx-2-mlx/                                  ← upstream package, cloned at install
│   ├── env/                                    ← Python 3.11 venv
│   └── packages/{ltx-core-mlx,ltx-pipelines-mlx}
├── mlx_models/                                 ← downloaded weights
│   ├── ltx-2.3-mlx-q4/                         ← Q4 base (~25 GB)
│   ├── ltx-2.3-mlx-q8/                         ← optional Q8 (~25 GB)
│   └── gemma-3-12b-it-4bit/                    ← text encoder (~7.5 GB)
├── mlx_outputs/                                ← generated mp4s
├── panel_uploads/                              ← user-uploaded reference images
├── assets/                                     ← logos, banner, hero clip
├── launch/                                     ← launch post drafts (X, Reddit, Pinokio, CivitAI)
├── mlx_ltx_panel.py                            ← MAIN: HTTP server, HTML UI, queue
├── mlx_warm_helper.py                          ← subprocess holding MLX pipelines warm
├── pinokio.js / install.js / start.js / update.js / reset.js / download_q8.js
├── patch_ltx_codec.py                          ← applies patches to upstream package
└── required_files.json                         ← single source of truth for "installed"
```

To work: `cd /Users/salo/pinokio/api/phosphene.git/`, edit, `git commit`,
`git push`. Pinokio's Update button does `git pull` here. No separate
copy to keep in sync.

## 3. Architecture (request → frame)

```
Pinokio app (port 42000)
   └── proxies to → mlx_ltx_panel.py (port 8198)
                       │
                       ├── serves a single-page HTML/JS UI from one Python process
                       ├── exposes HTTP API: /run, /status, /upload, /helper/restart, ...
                       ├── owns a job queue (FIFO) + worker thread
                       └── speaks JSON-over-stdin/stdout to:
                              │
                              └── mlx_warm_helper.py (subprocess in same venv)
                                     ├── holds MLX pipeline objects warm across jobs
                                     ├── pipeline classes from ltx-2-mlx package
                                     ├── reaps itself after LTX_IDLE_TIMEOUT (1800s default)
                                     ├── emits events: log / done / error / exit
                                     └── on death → panel sees stdout EOF
```

Both processes run inside `ltx-2-mlx/env` (the Python 3.11 venv). Helper
restarts cheaply via `POST /helper/restart` to pick up code changes
without restarting the panel.

## 4. Locked dependency versions

These are not arbitrary — every pin is a paid lesson:

| Pin | Why |
|---|---|
| `python>=3.11` (force `uv venv --python 3.11`) | Pinokio's `venv:` shortcut defaults to conda's base 3.10. ltx-core-mlx requires `>=3.11`. |
| `mlx==0.31.1` | mlx 0.31.2 introduced a numerical regression that attenuates LTX 2.3 vocoder output by ~22 dB. Verified: same model + same prompt + same seed → -42 dB peak on 0.31.2 vs -9 dB peak on 0.31.1. |
| `mlx-lm==0.31.1` `mlx-metal==0.31.1` | Same release line, kept consistent with mlx. |
| `huggingface-hub>=1.0` | `hf` CLI replaced `huggingface-cli`; older Pinokio bundles ship < 1.0. |
| `ltx-2-mlx` at `main` (HEAD) | dcd639e (0.1.0) was tried as a pin during audio investigation but breaks the Extend `cfg_scale` API and forces a `transformer.safetensors` symlink hack. HEAD + the mlx pin is the right combination. |

When changing a version, document the test that proved the new pin is OK.

## 5. The patch system (`patch_ltx_codec.py`)

We patch the upstream `ltx-2-mlx` package after `pip install`. Patches
are idempotent and fail loud on upstream drift (don't silently ship a
broken pipeline). Each patch has an `OLD` text it expects to find and a
`NEW` text it replaces with, plus a `marker` to detect "already patched."

Currently shipped patches:

1. **Codec (`yuv444p crf 0` + `+faststart`)** — required.
   Patches `ltx_core_mlx/model/video_vae/video_vae.py`. Default ffmpeg
   args become lossless H.264 (yuv444p crf 0) and the moov atom moves
   to the front of the file so gallery thumbnails render the first
   frame without downloading the full clip.

2. **I2V free DiT before VAE decode** — optional (warns if upstream changed).
   Patches `ImageToVideoPipeline.generate_and_save` in
   `ltx_pipelines_mlx/ti2vid_one_stage.py`. Frees DiT + text encoder +
   feature extractor before the VAE decode step.

3. **I2V free vae_encoder + feature_extractor BEFORE denoise** — optional.
   Patches `generate_from_image` to null `self.vae_encoder` and
   `self.feature_extractor` right after `_encode_text_and_load`
   returns. The connector inside feature_extractor is ~5.91 GiB on Q4
   and was sitting resident through denoise for no reason.

4. **Base `load()` also clears feature_extractor** — optional.
   Patches `TextToVideoPipeline.load`. Drops one extra ~5.91 GiB blob
   from the peak when the DiT loads.

The "upgrade" path in `apply_patch()` lets old patched installs receive
new patch versions without a venv rebuild — used when we added
`+faststart` to the existing codec patch.

## 6. Critical history (so we don't relitigate)

- **mlx 0.31.2 audio regression** (resolved): symptom was "audio sounds
  like static / very quiet." Root cause was an MLX numerical change
  attenuating the vocoder output by ~22 dB. Fix: `mlx==0.31.1` pin.
  Bisecting which mlx commit caused it is a v1.1 task.
- **dcd639e pin detour** (resolved): briefly pinned `ltx-2-mlx` to
  commit `dcd639e` (0.1.0) thinking the audio regression was in
  dgrauet's commits. It wasn't. Pin caused Extend API mismatch
  (`cfg_scale` kwarg unknown) and required a transformer.safetensors
  symlink because the 0.1.0 loader hardcoded that filename. Reverted.
- **Pinokio menu API mismatch** (resolved): `info.path` is a function
  on newer Pinokio, a string on older. `pinokio.js` now detects both
  shapes via `getInstallRoot(info)`.
- **start.js URL hijack** (resolved): the `errno`/`error:` event
  patterns were cargo-culted from comfy.git's start.js. They captured
  any "Errno" line from the helper into `input.event[0]` and the panel
  URL got set to the literal string "Errno." Removed those patterns.
- **silent helper death** (mitigated, not fixed): on memory-pressured
  Macs the helper subprocess is SIGKILL'd by macOS jetsam during I2V
  pipeline load. Patches 3 and 4 reduced peak; the panel now names
  the exit signal (SIGKILL/SIGSEGV/SIGABRT/SIGBUS) and points users at
  `~/Library/Logs/DiagnosticReports/python3.11_*.crash` for diagnosis.
- **HTML5 form validation silent block** (resolved): a hidden Extend
  input had `value=2 min=0.4 step=0.5`. Chrome's validator only accepts
  values on the (min + N*step) sequence, so 2 was off-grid → invalid
  → entire form silently failed to submit. Fixed `min=0.5`.

## 7. Working rules — non-negotiable

### Identity
- **Author/committer:** `mrbizarro <mrbizarro@users.noreply.github.com>`.
  Already set in this repo's local git config. Never override.
- **Never use the user's real first name** in commits, code, file
  contents, or any artifact that lands on disk or in git. Same for
  any third party (reviewers, collaborators) — refer to them by
  technical role only ("upstream review", "external reviewer", "user
  feedback") if at all.
- **Never add `Co-Authored-By:` trailers.** Phosphene reads as a
  serious project shipped by `mrbizarro`. Co-author trailers showing
  "Claude" make it look like a vibe-coded toy.

### Commit message style
- Subject line in imperative mood, 50–72 chars max, no trailing period.
  Use `feat:` / `fix:` / `refactor:` / `docs:` / `chore:` / `perf:` /
  `test:` prefixes when they help.
- Body explains the **why**: the symptom, the root cause, the chosen
  fix. The diff already shows the what.
- No first-person voice. Third person, neutral, technical.
- No conversation transcript. No "user said X", "after we discussed Y",
  "per the review on Tuesday." Just describe the change.
- Cite concrete things: file paths, line numbers, package versions,
  upstream commits, signal names. Avoid vague nouns ("the issue").
- Reference upstream context where it helps future maintainers (HF
  metadata size, upstream commit SHA, GitHub issue number).
- One commit = one logical change. Don't bundle UI polish with bug
  fixes with version pins.

### Examples — bad vs good

❌ Bad:
```
fix the audio thing peanut found

cocktailpeanut said the audio sounded like static so we tried a few
things and pinning mlx fixed it. great catch!
```

✅ Good:
```
fix(audio): pin mlx==0.31.1 to recover LTX 2.3 vocoder amplitude

mlx 0.31.2 introduces a numerical change that attenuates the LTX 2.3
vocoder output by ~22 dB. Verified empirically with the same prompt,
seed, and weights:
  mlx 0.31.2 → max_volume -42.8 dB (audible only with significant boost)
  mlx 0.31.1 → max_volume   -9.2 dB

install.js installs the pinned trio (mlx, mlx-lm, mlx-metal) before
the ltx-* packages so transitive resolution does not pull latest.
update.js force-reinstalls them with --no-deps so existing installs
that already pulled 0.31.2 can recover by clicking Update.
```

### Code style
- Comments explain non-obvious decisions and trade-offs. They earn
  their keep by explaining context the diff cannot.
- When fixing a bug, leave a comment with the symptom + cause + fix
  near the change so the next reader doesn't reintroduce it.
- Patches in `patch_ltx_codec.py` always include a comment block
  above each patch explaining what the upstream code does, why we
  patch it, and what the patch achieves.

### Working with the running install
- The Pinokio panel may already be running on port 8198 when you
  start a session. **Never spawn your own panel from the command
  line** — Pinokio expects to own that port and will fail Start
  with `Address already in use` if you do.
- To pick up code changes without restarting Pinokio, hit
  `POST /helper/restart` — that reloads the helper subprocess
  against the patched site-packages.
- For HTML/JS/CSS changes you do need a panel restart. Either
  click Stop + Start in Pinokio, or `pkill -f mlx_ltx_panel.py`
  and click Start.

### Disk consolidation
- Old `/Users/salo/Documents/Codex/...` dev folder was deleted.
  Don't recreate. The Pinokio install is canonical.
- `~/.cache/huggingface/hub/models--*` was also deleted. Pinokio
  install pulls weights into `mlx_models/` directly via `hf
  download --local-dir`, not the cache. Don't reintroduce HF cache
  symlinks.

## 8. Test workflow

For HTML/UI/JS changes:
1. Edit `mlx_ltx_panel.py`
2. Restart panel (Pinokio Stop + Start, OR `pkill -f mlx_ltx_panel.py`
   then click Start in Pinokio)
3. Hard-refresh browser (Cmd+Shift+R)
4. Verify the served HTML contains the change:
   `python3 -c "import urllib.request; print('your-marker' in urllib.request.urlopen('http://127.0.0.1:8198/').read().decode())"`

For helper changes (Python pipeline, patches):
1. Edit `mlx_warm_helper.py` or `patch_ltx_codec.py`
2. If touching patches: re-run `./ltx-2-mlx/env/bin/python3.11 patch_ltx_codec.py`
3. `curl -X POST http://127.0.0.1:8198/helper/restart`
4. Submit a test job via `curl -X POST http://127.0.0.1:8198/run -d "mode=t2v" -d "prompt=..." -d "width=512" -d "height=288" -d "frames=49" -d "steps=8" -d "quality=draft"`
5. Poll `/status` until `current` is null and verify the new
   `history[0]` entry is `done`.

For audio testing — do not use generic "wizard in forest" prompts.
Use prompts with explicit dialogue or sound events so artifacts are
audible:

```
A woman in her 30s with auburn hair, sitting at a wooden table in a
softly lit kitchen, speaks directly to the camera with a small smile
and clear, confident voice. She says "I really did not think I would
see you again, but here we are." Audio: clear human voice, slight
room reverb, gentle ambient kitchen sounds.
```

Measure with ffmpeg:
```
ffmpeg -i out.mp4 -af "volumedetect" -f null - 2>&1 | grep -E "max_volume|mean_volume"
```
Healthy levels: `max_volume` between -2 dB and -15 dB on dialogue
prompts. Below -25 dB peak suggests a regression.

## 9. Pinokio lifecycle scripts

| Script | Trigger | What it does |
|---|---|---|
| `install.js` | First Install or Resume Install | Clone `ltx-2-mlx`, force Python 3.11 venv, install MLX trio at pinned versions, install ltx-* packages, run `patch_ltx_codec.py`, `hf download` Q4 + Gemma. Idempotent. |
| `start.js` | Start button | Spawn panel inside the venv with `LTX_*` env vars set. Capture URL via regex with capture group, route to `local.set` via `input.event[1]`. |
| `update.js` | Update button | `git pull` panel + ltx-2-mlx, force-reinstall mlx pin, force-reinstall ltx-* packages, re-run patches. |
| `reset.js` | Reset button | Wipe venv + clones. **Preserves** `mlx_outputs/`, `panel_uploads/`, `mlx_models/` (those are user content / huge weight files). |
| `download_q8.js` | Download Q8 button | `hf download dgrauet/ltx-2.3-mlx-q8` to `mlx_models/ltx-2.3-mlx-q8`. |
| `pinokio.js` | Always | Renders the menu. State driven by `required_files.json` (env marker + per-repo file completeness). |

## 10. Files to never touch unprompted

- Anything inside `mlx_models/` — these are downloaded weights, GB-scale
- Anything inside `mlx_outputs/` — user-generated content
- Anything inside `panel_uploads/` — user uploads
- `panel_queue.json`, `panel_hidden.json` — runtime state
- `logs/`, `cache/`, `__pycache__/` — runtime caches
- The `.clinerules`, `.cursorrules`, `.windsurfrules`, `.geminiignore`,
  `GEMINI.md`, `QWEN.md`, `AGENTS.md` files — these are generic Pinokio
  scaffolding mirroring this `CLAUDE.md`. Edit `CLAUDE.md` first if a
  project-wide change is needed; the rest are autogenerated mirrors.

## 11. HTTP API reference (panel)

The panel listens on `127.0.0.1:8198` and serves a single-page HTML/JS
UI plus a JSON API. Pinokio proxies its own port 42000 to 8198. All
endpoints are unauthenticated (loopback only).

### GET endpoints

| Path | Returns | Notes |
|---|---|---|
| `/` | HTML | The single-page UI. Built from one big template string in `mlx_ltx_panel.py`. |
| `/status` | JSON | Snapshot of `STATE` — running flag, current job, queue, history (top 20), log tail, tier, memory pressure, helper alive flag, comfy PIDs, repo download state. Polled by the UI every ~1s. |
| `/uploads` | JSON | Recent files in `panel_uploads/` for the picker's "click to reuse" strip. |
| `/models` | JSON | Per-repo install completeness — driven by `required_files.json`. |
| `/file?path=…` | bytes (mp4/png/jpg) | Range-aware video serve so `<video>` tags can seek without re-downloading. |
| `/image?path=…` | bytes (image) | Same idea for image previews. |
| `/sidecar?path=…` | JSON | Per-output sidecar (job params, output stats) if it exists. |
| `/assets/<file>` | bytes | Static files from `assets/`. |

### POST endpoints

| Path | Body | Returns | Effect |
|---|---|---|---|
| `/run` `/queue/add` | urlencoded job spec | `{ok, id}` | Creates a job, appends to queue. Field reference below. |
| `/queue/batch` | urlencoded with `prompts` (newline-separated, `---` chunks) | `{ok, added, ids}` | Splits the prompts on `^\s*---\s*$` boundaries, creates one job per chunk. |
| `/queue/remove` | `id=<jid>` | `{removed: bool}` | Drop a queued job by id. |
| `/queue/clear` | (none) | `{cleared: int}` | Clear all queued jobs (running job continues). |
| `/queue/pause` | (none) | `{paused: true}` | Worker stops popping new jobs after the current one. |
| `/queue/resume` | (none) | `{paused: false}` | Worker resumes. |
| `/output/hide` | `path=<p>` | `{hidden: path}` | Adds path to the gallery's hidden set (persisted). |
| `/output/show` | `path=<p>` | `{shown: path}` | Inverse. |
| `/output/show_all` | (none) | `{unhidden_count: int}` | Clear hidden set. |
| `/upload` | multipart `image=<file>` | `{ok, path}` | Saves to `panel_uploads/<ts>_<safename>`. |
| `/helper/restart` | (none) | `{ok}` | SIGTERM the helper subprocess; the next job auto-respawns it. Useful for picking up site-packages changes. |
| `/stop` | (none) | `{ok}` | Cancel the current job (kills helper + mux). Worker advances. |
| `/stop_comfy` | (none) | `{ok, killed}` | SIGTERM any ComfyUI process matching `LTX_COMFY_PATTERN`. |
| `/open_pinokio` | (none) | `{ok}` | macOS-only: focus the Pinokio app. |
| `/models/download` | `key=q4|q8|gemma` | `{ok}` | Spawn `hf download` for the named repo. Streams to log. |
| `/models/cancel` | (none) | `{ok, killed}` | SIGTERM the active download. |
| `/prompt/enhance` | `prompt=<p>` | `{ok, original, enhanced}` | Calls helper's Gemma-rewrite path. |

## 12. Job spec — the contract for `/run` and `/queue/add`

Every form field below maps to `params.<field>` on the resulting job.
`make_job()` in `mlx_ltx_panel.py:1140` is the canonical reference.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `mode` | str | `"t2v"` | One of: `t2v`, `i2v`, `i2v_clean_audio`, `extend`, `keyframe`. |
| `prompt` | str | `"A cinematic atmospheric scene"` | Empty falls back to the default. |
| `width` | int | `1280` | Clamped to `>= 32`. T2V only — image flows derive from aspect+quality. |
| `height` | int | `704` | Same as width. |
| `frames` | int | `121` | Number of video frames. The 8k+1 rule (latents are groups of 8). |
| `steps` | int | `8` | Denoising steps. T2V/I2V Q4 distilled requires `>= 8` (schedule is locked). |
| `seed` | int-as-str | `"-1"` | `"-1"` → random. Stored as string so the form roundtrip is lossless. |
| `image` | path | `REFERENCE` | Only used when `mode != "t2v"`. |
| `audio` | path | `AUDIO_DEFAULT` | Only used when `mode == "i2v_clean_audio"` (mux external audio). |
| `video_path` | path | `""` | **Required for `mode=extend`**. Source mp4 to extend. |
| `extend_frames` | int | `5` | Latent-frame count to add. Each latent = 8 video frames. So 5 latents = 40 frames ≈ 1.67s at 24 fps. |
| `extend_direction` | str | `"after"` | `"after"` or `"before"`. |
| `extend_steps` | int | `12` | Denoising steps for the extend pipeline (uses dev transformer). |
| `extend_cfg` | float | `1.0` | CFG scale. 1.0 = no guidance (faster, less RAM). 3.0 = guided (Quality mode). |
| `start_image` | path | `""` | **Required for `mode=keyframe`**. First frame. |
| `end_image` | path | `""` | **Required for `mode=keyframe`**. Last frame. |
| `enhance` | "on"/"off" | `"off"` | Reserved (panel uses the explicit `/prompt/enhance` button instead). |
| `stop_comfy` | "on"/"off" | `"off"` | Kill ComfyUI before this render to free MPS memory. |
| `open_when_done` | "on"/"off" | `"off"` | macOS `open` the output when finished. |
| `quality` | str | `"standard"` | `"draft"`, `"standard"`, or `"high"`. Drives dimensions, step counts, and pipeline. |
| `no_music` | "on"/"off" | `"off"` | When on, the panel appends a no-music audio constraint to the prompt at submit time. |
| `preset_label` | str | `""` | Optional human-readable label for the queue/history rows. |

## 13. Mode → pipeline → transformer routing

| Mode | Pipeline class | Transformer | Quality flag | Notes |
|---|---|---|---|---|
| `t2v` | `TextToVideoPipeline` | distilled Q4 | draft → smaller dims; standard → full; high → `TwoStageHQPipeline` w/ Q8 dev | Most common path. |
| `i2v` | `ImageToVideoPipeline` | distilled Q4 | same as T2V | Adds VAE-encoder pass on the input image. |
| `i2v_clean_audio` | `ImageToVideoPipeline` | distilled Q4 | same | Generates LTX video, then panel-side ffmpeg mux replaces the LTX audio with `audio` param. |
| `extend` | `ExtendPipeline` | **dev** Q4 | always uses Q4 dev; cfg_scale dial via `extend_cfg` | Heavier (~75 s/step on 1280×704 cfg=1.0). |
| `keyframe` (FFLF) | `KeyframePipeline` | **Q8 dev + distilled LoRA** | requires Q8 download | Disabled on `base` tier. |

The helper `get_pipe(kind)` lazily loads the pipeline class corresponding
to the mode. Switching modes between jobs frees the previous pipeline
(`release_pipelines(keep_kind=...)`) — strict one-pipeline-at-a-time
policy keeps memory bounded.

## 14. Quality tier mapping

`quality` is independent of the hardware tier. Hardware tier *clamps*
the maximum dimensions per mode; `quality` picks the pipeline + step
count *within* that clamp.

| `quality` | Pipeline used | Dimensions | Steps | Approx wall on M4 64 GB |
|---|---|---|---|---|
| `draft` | distilled Q4 | half of standard (rounded to multiples of 32) | 8 | ~2 min @ 512×288 |
| `standard` | distilled Q4 | full target (clamped by tier) | 8 | ~7 min @ 1280×704 |
| `high` | TwoStageHQPipeline (Q8 dev + res_2s sampler + TeaCache) | full | 30 + extra | ~12 min @ 1280×704 |

`high` requires the Q8 model on disk (~25 GB extra). Disabled when the
panel detects Q8 incomplete.

## 15. Hardware tier system

`SYSTEM_TIER` is detected once at boot from `sysctl hw.memsize`. The
override `LTX_TIER_OVERRIDE=base|standard|high|pro` forces a tier
(useful for testing what other users see). `CAPABILITIES` in
`mlx_ltx_panel.py:584` is the canonical table; abbreviated:

| Tier | RAM range | Friendly | t2v/i2v max dim | Q8 | FFLF | Extend |
|---|---|---|---|---|---|---|
| `base` | < 48 GB | Compact | 768 | ❌ | ❌ | ❌ |
| `standard` | 48–79 GB | Comfortable | 1280 | ✅ | ✅ | ✅ |
| `high` | 80–119 GB | Roomy | 1280 | ✅ | ✅ (more frames) | ✅ |
| `pro` | ≥ 120 GB | Studio | 0 (no clamp) | ✅ | ✅ | ✅ |

The thresholds are intentional. M-Studio 64 GB is the canonical
"standard" hardware. Anything below 48 GB cannot fit Q8 weights
(transformer-dev.safetensors is 19.18 GiB plus working tensors).

## 16. `required_files.json` schema

Single source of truth for "what counts as installed." Read by:
- `pinokio.js` → menu rendering (Install vs Resume Install vs Start)
- `mlx_ltx_panel.py` → `/status` and per-job validation
- (indirectly) `install.js` and `download_q8.js` — their `hf download`
  paths must match the `local_dir` declared here.

```jsonc
{
  "_comment": "...",
  "repos": [
    {
      "key": "q4" | "gemma" | "q8",
      "kind": "base" | "optional",          // "base" = blocks Start, "optional" = unlock features
      "name": "human-readable label",
      "blurb": "what this is for",
      "repo_id": "<hf-org/repo>",            // for `hf download`
      "local_dir": "mlx_models/<repo-name>", // relative to install root
      "size_gb": 25,                          // approximate, displayed in UI
      "files": [                              // every required file inside local_dir
        "transformer-distilled.safetensors",
        "connector.safetensors",
        ...
      ]
    }
  ],
  "env": {
    "marker_paths": [                         // OR-checked; first existing path wins
      "ltx-2-mlx/env/pyvenv.cfg",             // canonical Pinokio install
      "ltx-2-mlx/.venv/pyvenv.cfg",           // alternate manual layouts
      ...
    ]
  },
  "min_size_bytes": 1024                      // file must be >= this to count as "complete"
}
```

When the upstream model layout changes, **edit this file first**. The
menu, panel, and install scripts auto-pick up the change.

## 17. Helper protocol (panel ↔ helper)

The helper subprocess (`mlx_warm_helper.py`) speaks newline-delimited
JSON over stdin/stdout. The panel's `WarmHelper` class wraps it.

### Actions panel sends

```jsonc
// Generation request (T2V, I2V, I2V-clean-audio)
{ "action": "generate",
  "id": "<job-id>",
  "params": { "mode": "t2v"|"i2v"|"i2v_clean_audio",
              "prompt": "...", "output_path": "...",
              "height": 704, "width": 1280, "frames": 121,
              "seed": 42, "steps": 8, "image": "<path>" } }

// Extend request
{ "action": "extend",
  "id": "<job-id>",
  "params": { "video_path": "...", "prompt": "...",
              "extend_frames": 5, "extend_direction": "after",
              "extend_steps": 12, "cfg_scale": 1.0,
              "output_path": "...", "seed": 42 } }

// Keyframe (FFLF)
{ "action": "keyframe", "id": "...", "params": {...} }

// Prompt rewrite (Gemma)
{ "action": "enhance", "prompt": "..." }
```

### Events helper emits

```jsonc
{ "event": "ready",  "model": "<path>", "low_memory": true }    // on startup
{ "event": "log",    "line": "free-form panel log line" }        // any informational message
{ "event": "done",   "id": "...", "output": "<path>",
                     "elapsed_sec": 33.6, "seed_used": 42 }
{ "event": "error",  "id": "...", "error": "<message>",
                     "trace": "<python traceback>" }
{ "event": "exit",   "reason": "python_normal_exit"
                            | "sigterm(15)"
                            | "sigterm(2)"
                            | "idle" }                            // graceful exit
{ "event": "enhanced", "original": "...", "enhanced": "..." }    // /prompt/enhance result
```

**No exit event + stdout pipe closes = non-graceful death** (SIGKILL,
SIGSEGV, SIGABRT, SIGBUS). The panel inspects `proc.poll()` returncode
and translates the negative signal number to a name in the user-facing
error.

### Step breadcrumb log lines

The helper emits these inside `generate`/`extend` actions so the panel
can localize a silent death to a phase:

```
step:get_pipe kind=t2v
step:get_pipe done
step:generate_and_save mode=i2v 1280x704 121f steps=8
step:generate_and_save done
```

When a job dies silently, the panel log's last `step:*` line tells you
which phase (pipeline init vs actual generation).

## 18. Memory budget model

Concrete weight sizes (HuggingFace repo metadata, Q4 unless noted):

| Component | Size on disk | Resident peak (rough) | Notes |
|---|---|---|---|
| `transformer-distilled.safetensors` (Q4) | 10.54 GiB | ~12 GiB | T2V/I2V "DiT" |
| `transformer-dev.safetensors` (Q4) | 11.3 GiB (rough) | ~13 GiB | Used by Extend |
| `transformer-dev.safetensors` (Q8) | 19.18 GiB | ~22 GiB | High-quality + FFLF |
| `connector.safetensors` (Q4) | 5.91 GiB | ~6 GiB | Lives inside `feature_extractor` |
| `Gemma-3-12B-it-4bit` | 7.48 GiB | ~9 GiB | Text encoder, freed before DiT load |
| `vae_decoder.safetensors` | 814 MB | ~1 GiB | Used at the end of generation |
| `vae_encoder.safetensors` | 638 MB | ~800 MB | I2V only (encodes input image) |
| `audio_vae.safetensors` | 107 MB | ~150 MB | |
| `vocoder.safetensors` | 258 MB | ~350 MB | |

Peak during T2V Standard 1280×704 (with patches applied): roughly
DiT + active activations + decoder ≈ 20–22 GiB resident. Without
the patches it's 26+ GiB and OOMs on memory-pressured 64 GB Macs.

Peak during I2V is +800 MB for the encoder pass + ~6 GiB for the
connector that lingered in `feature_extractor` (Patches 3 + 4 fix
the lingering — see §5).

macOS `jetsam` is the OOM killer. It enforces memory pressure based
on a percentage that varies with paging activity, swap usage, and other
heuristics — there is no fixed "kill at X% pressure" rule. Closing
Chrome, Slack, iOS Simulator before a Standard render is the safest
single thing a user can do.

## 19. Debug recipes

### "Helper died silently — where did it die?"

```bash
# Scan the panel log for the last step:* breadcrumb
curl -s http://127.0.0.1:8198/status | python3 -c \
  "import json,sys; [print(l) for l in json.load(sys.stdin)['log'] if 'step:' in l][-3:]"
```

If last step is `step:get_pipe` → death during pipeline init (model
load). Likely OOM or weight-corrupt. If last step is
`step:generate_and_save` → death during gen (denoise / VAE decode).
Likely OOM peak or Metal kernel issue.

For non-OOM crashes, check `~/Library/Logs/DiagnosticReports/python3.11_*.crash`
or `*.ips` from around the crash time.

### "Panel says port 8198 already in use"

```bash
lsof -nP -iTCP:8198 -sTCP:LISTEN          # who has it
pkill -f mlx_ltx_panel.py                  # nuke all panels
```

Common cause: a previous manual `python mlx_ltx_panel.py` is still
running. Pinokio's start.js can't take over a port someone else owns.
**Don't spawn the panel manually unless you're going to kill it before
handing back.**

### "I edited mlx_ltx_panel.py but the UI hasn't changed"

The HTML/JS is baked into the Python file as a template string. The
running panel served the old version. Restart it:
- via Pinokio: Stop → Start
- or via shell: `pkill -f mlx_ltx_panel.py` then click Start in Pinokio

Then **hard-refresh the browser** (Cmd+Shift+R) to bypass JS cache.

### "I edited the helper or a patch but nothing changed"

```bash
# 1. Re-apply patches (idempotent, applies to site-packages)
cd ~/pinokio/api/phosphene.git
./ltx-2-mlx/env/bin/python3.11 patch_ltx_codec.py

# 2. Restart the helper subprocess (panel keeps running)
curl -X POST http://127.0.0.1:8198/helper/restart

# 3. Submit a test job
curl -X POST http://127.0.0.1:8198/run \
  -d "mode=t2v" -d "prompt=quick test" \
  -d "width=512" -d "height=288" -d "frames=49" \
  -d "steps=8" -d "quality=draft"

# 4. Wait for done
until [ "$(curl -s http://127.0.0.1:8198/status | python3 -c 'import json,sys;d=json.load(sys.stdin);print(0 if d.get(\"current\") else 1)')" = "1" ]; do sleep 5; done
```

### "How loud / clean is the audio in this clip?"

```bash
FFPROBE="/Volumes/Topaz Video AI 4.0.6/Topaz Video AI.app/Contents/MacOS/ffprobe"
FFMPEG="/Volumes/Topaz Video AI 4.0.6/Topaz Video AI.app/Contents/MacOS/ffmpeg"
"$FFMPEG" -i out.mp4 -af "volumedetect" -f null - 2>&1 | grep -E "max_volume|mean_volume"
```

Healthy on dialogue: `max_volume` between -2 and -15 dB. Below -25 dB
peak is the regression fingerprint — usually means a wrong mlx version.

### "Re-encode for X / web (yuv420p, faststart, 8 Mbps)"

```bash
"$FFMPEG" -y -i in.mp4 \
  -c:v h264_videotoolbox -profile:v high -pix_fmt yuv420p \
  -b:v 8M -maxrate 12M -bufsize 16M \
  -movflags +faststart \
  -c:a aac -b:a 192k \
  out_x.mp4
```

X (Twitter) rejects yuv444p with a misleading "aspect ratio" error.
Use this recipe for any social upload.

### "Stuck queue / can't cancel"

```bash
curl -X POST http://127.0.0.1:8198/stop          # cancel current job
curl -X POST http://127.0.0.1:8198/queue/clear   # empty the queue
curl -X POST http://127.0.0.1:8198/helper/restart # nuke the helper
```

### "Force re-download a corrupt model file"

```bash
# Delete the file in question, then click Install in Pinokio (resumes
# only the missing files; intact ones are skipped via `hf` hash check).
rm ~/pinokio/api/phosphene.git/mlx_models/ltx-2.3-mlx-q4/<bad-file>.safetensors
# Then in Pinokio: Phosphene → Install (or Resume Install)
```

### "Force a different hardware tier for testing"

```bash
LTX_TIER_OVERRIDE=base ./run_panel.sh    # pretend this Mac is < 48 GB
```

`run_panel.sh` actively unsets this on every run unless explicitly set
on the command line — there's a hard-learned bug history about leaked
overrides bleeding across panel restarts.

### "What does `/status` actually return?"

```bash
curl -s http://127.0.0.1:8198/status | python3 -m json.tool | head -60
```

Top-level keys: `running`, `paused`, `current`, `queue`, `history`,
`log`, `tier`, `memory`, `helper`, `download`, `repos_ready`,
`repos_total`, `base_available`, `q8_available`, `q8_missing`,
`comfy_pids`, `outputs`, `server_now`, `avg_elapsed_sec`.

## 20. Patch authoring template (`patch_ltx_codec.py`)

When adding a new patch:

```python
# ---- Patch N: <one-line summary> --------------------------------------------
# Why: <symptom + root cause + why this fix>. Reference upstream
# file:line and any external context (commit SHA, GitHub issue, paper).
PATCH_<NAME>_OLD = '''<exact text expected to be in the upstream file>'''
PATCH_<NAME>_NEW = '''<replacement text — include a `# PATCHED (LTX23MLX): ...` comment>'''
```

Then in `main()`:

```python
target = _find("<relative/path/to/upstream.py>")
outcome = apply_patch(
    target, PATCH_<NAME>_OLD, PATCH_<NAME>_NEW,
    marker="<unique substring that proves NEW is on disk>",
    label="<NAME>",
    upgrade_marker="<optional: substring that distinguishes a newer NEW from an older NEW>",
)
# If this patch is optional (don't fail install on drift):
if outcome in (OUTCOME_DRIFT, OUTCOME_MISSING):
    print("  [<NAME>] note: ...", file=sys.stderr)
    outcome = OUTCOME_ALREADY
outcomes.append(("<NAME>", outcome))
```

Conventions:
- `marker` is a substring **inside** `PATCH_<NAME>_NEW` that's unlikely
  to appear in the unpatched file. Convention: include
  `PATCHED (LTX23MLX...)` somewhere in the new text.
- `upgrade_marker` lets you ship a v2 of a patch without forcing users
  to nuke their venv. Example: codec patch's `+faststart` was added
  later; `upgrade_marker="+faststart"` distinguishes v1 (no faststart)
  from v2 and rewrites the line in place.
- Patches that target an upstream code structure that may legitimately
  vanish (e.g. an internal helper that gets refactored away) should be
  marked **optional** in main() so install warns instead of fails.

## 21. Glossary

- **LTX 2.3** — Lightricks' latent text-to-video diffusion model with
  joint audio synthesis. Trained on a large internal dataset.
- **MLX** — Apple's open-source ML array framework, native to Metal.
  Like JAX/PyTorch but built for Apple Silicon. Not a torch shim.
- **Q4 / Q8** — Quantization levels of the LTX 2.3 weights. Q4 is
  4-bit quantized (smaller, faster, slightly lower quality), Q8 is
  8-bit (larger, slower, sharper detail). Phosphene defaults to Q4;
  Q8 is an optional upgrade.
- **DiT** — Diffusion Transformer. The main denoising network. The
  largest single weight tensor.
- **VAE** — Variational Autoencoder. Encodes pixels ↔ latents. The
  encoder is used in I2V (encodes the input image); the decoder is
  used at the end of every generation.
- **FFLF** — First-Frame / Last-Frame interpolation. Provide two images,
  the model generates the motion between them. Requires Q8.
- **BWE** — Bandwidth Extension. Inside the audio vocoder, upsamples
  a low-rate signal to full audio rate (108 sequential BigVGAN convs).
- **Vocoder** — Converts mel-spectrogram features to PCM audio. The
  final stage of LTX's audio pipeline.
- **TeaCache** — A diffusion-step caching technique that skips
  redundant computation when consecutive sigmas are similar. Used in
  the High quality two-stage path.
- **jetsam** — macOS' kernel-level memory pressure killer. Sends SIGKILL
  to processes when system memory gets tight. The dominant cause of
  silent helper death on Macs at full RAM.
- **moov atom** — The MP4 metadata index. With `+faststart` it lives
  at the front of the file (allows progressive playback / instant
  thumbnails); without, it lives at the end (browser must download
  the whole file before showing the first frame).
- **Pinokio** — One-click installer + runtime for AI apps on Mac/Linux/Windows.
  Phosphene ships as a Pinokio app. cocktailpeanut authored Pinokio.
- **mrbizarro** — The GitHub handle that authors this repo. Use this,
  always, in commits and metadata.

## 22. Open questions & v1.1 candidates

Tracked here so they don't get lost between sessions.

- **mlx 0.31.1 → 0.31.2 audio bisect** — find the exact MLX commit
  that attenuates LTX 2.3's vocoder by 22 dB. File upstream issue.
- **A2V (audio-to-video) mode** — `ltx-2-mlx` upstream has
  `a2vid_two_stage.py` and `a2vid_two_stage_hq.py` but the panel
  doesn't expose them. UI work + helper plumbing required.
- **In-app HF token field** — currently we recommend `hf auth login`
  in Terminal. A panel-side settings field that writes
  `~/.cache/huggingface/token` would be friendlier for non-CLI users.
- **Pre-flight RAM advisory** — before submitting a heavy job, warn
  if `s.memory.swap_gb > 8` or `pressure_pct > 75`. Suggest Draft.
- **Audio mode dropdown** — supersede the binary "No music" toggle
  with: With music / Voice + ambient / Sound effects / Silent. Each
  option appends/prepends a different prompt constraint.
- **Deeper helper breadcrumbs** — current `step:*` markers cover the
  outer call. Adding `image_encode_start`, `text_encode_done`,
  `denoise_start`, `decode_start` inside the pipeline (via wrapper
  patches) would localize silent deaths down to the phase.
- **Q8 path validation** — Q8 generation has not been end-to-end
  tested on a 64 GB Mac since the codec/I2V patches landed. Schedule
  a 25 GB Q8 download + High quality T2V test.
- **FFLF validation** — same as above. Requires Q8.

## 23. External references

- LTX-2 paper / model card: https://huggingface.co/Lightricks
- MLX framework: https://github.com/ml-explore/mlx
- `ltx-2-mlx` (the MLX port we wrap): https://github.com/dgrauet/ltx-2-mlx
- Pinokio runtime: https://pinokio.computer
- Q4 weights: `dgrauet/ltx-2.3-mlx-q4` on HF
- Q8 weights: `dgrauet/ltx-2.3-mlx-q8` on HF
- Gemma text encoder: `mlx-community/gemma-3-12b-it-4bit` on HF

---

# Generic Pinokio Development Guide

(Everything below this line is the upstream Pinokio scaffolding shipped
with every new project. Useful as a reference when working with the
launcher scripts but the Phosphene-specific rules above take
precedence on conflict.)

# Development Guide for Pinokio Projects

## Non-Negotiable Execution Workflow

To guarantee every contribution follows this guide precisely, obey this checklist **before any edits** and **again before finalizing**. Do not skip or reorder.
1. **AGENTS Snapshot:** Re-open this file and write down (in your working notes or response draft) the exact sections relevant to the requested task. No work begins until this snapshot exists.
2. **Destination Resolution:** Before creating or editing any Pinokio launcher files, resolve `PINOKIO_HOME` to an absolute path and record the intended destination root. If running outside Pinokio's own managed runtime, resolve in this order: `~/.pinokio/config.json` `home`, then `GET http://127.0.0.1:42000/pinokio/home` and use its `path` value, and if loopback is unreachable but `access` exists in `~/.pinokio/config.json`, retry the same request against `<protocol>://<host>:<port>/pinokio/home`, then the `PINOKIO_HOME` environment variable. If `PINOKIO_HOME` is still unresolved, stop and ask the user. Never silently use the current workspace as the launcher destination.
3. **Example Lock-in:** Identify the closest matching script in `/Users/salo/pinokio/prototype/system/examples`. Record its path and keep it open while editing. Every launcher change must mirror that reference unless the user explicitly instructs otherwise.
4. **Pre-flight Checklist:** Convert the applicable rules from this document and `PINOKIO.md` at /Users/salo/pinokio/prototype/PINOKIO.md into a task-specific checklist (install/start/reset/update structure, regex patterns, menu defaults, log checks, destination path, etc.). Confirm each item is ticked **before** making changes.
5. **Mid-task Verification:** Any time you touch a Pinokio script, cross-check the corresponding example line to ensure syntax and structure match. Document the reference (example path + line) in your reasoning.
6. **Exit Checklist:** Before responding to the user, revisit the pre-flight checklist and explicitly confirm every item is satisfied. If anything diverges from the example or these rules, fix it first.

If any step cannot be completed, stop immediately and ask the user how to proceed. These six steps are mandatory for every session.

### Critical Pattern Lock: Capturing Web UI URLs

When writing `start.js` (or any script that needs to surface a web URL for a server):

1. **Always copy the capture block from an example such as `system/examples/mochi/start.js`.**
```javascript
on: [{
  event: "/(http:\\/\\/[0-9.:]+)/",
  done: true
}]
```

2. **Set the local variable using the captured match exactly as below (The regex capture object is passed in as `input.event`, so need to use the index 1 inside the parenthesis):**
```javascript
{
  method: "local.set",
  params: {
    url: "{{input.event[1]}}"
  }
}
```

3. Always try to come up with the most generic regex.
4. During the exit checklist, explicitly confirm that the `url` local variable is set via `local.set` API by using the captured regex object as passed in as `input.event` from the previous `shell.run` step.

Deviation from this pattern requires written approval from the user.

- Make sure to keep this entire document and `PINOKIO.md` at /Users/salo/pinokio/prototype/PINOKIO.md in memory with high priority before making any decision. Pinokio is a system that makes it easy to write launchers through scripting by providing various cross-platform APIs, so whenever possible you should prioritize using Pinokio API over lower level APIs.
- When writing pinokio scripts, ALWAYS check the examples folder (in /Users/salo/pinokio/prototype/system/examples folder) to see if there are existing example scripts you can imitate, instead of assuming syntax.
- When implementing pinokio script APIs and you cannot infer the syntax just based on the examples, always search the API documentation `PINOKIO.md` at /Users/salo/pinokio/prototype/PINOKIO.md to use the correct syntax instead of assuming the syntax.
- When trying to fix something or figure out what's going on, ALWAYS start by checking the `logs` folder before doing anything else, as mentioned in the "Troubleshooting with Logs" section.
- Finally, make sure to ALWAYS follow all the items in the "best practices" section below.

## Determine User Intent
If the initial prompt is simply a URL and nothing else, check the website content and determine the intent, and ask the user to confirm. For example a URL may point to

1. A Tutorial: the intent may be to implement a demo for the tutorial and build a launcher.
2. A Demo: the intent may be a 1-click launcher for the demo
3. Open source project: the intent may be a 1-click launcher for the project 
4. Regular website: the intent may be to clone the website and a launcher.
5. There can be other cases, but try to guess.

## Working With Launchers

Apply this section only when the task is to create, modify, debug, review, or document a Pinokio launcher project.

If the request is not about launcher work, do not force an app-launcher vs plugin-launcher decision.

When the task does involve launcher work, first determine whether the request is for an app launcher or a plugin launcher. These are separate project types and must not be mixed.

### Mandatory Destination Resolution
- Before creating, editing, or moving any launcher files, resolve `PINOKIO_HOME` to an absolute path.
- If running outside Pinokio's own managed runtime, resolve `PINOKIO_HOME` in this order:
  1. `~/.pinokio/config.json` -> `home`
  2. `GET http://127.0.0.1:42000/pinokio/home` -> `path`
  3. If loopback is unreachable but `access` exists in `~/.pinokio/config.json`, retry the same request against `<protocol>://<host>:<port>/pinokio/home`
  4. `PINOKIO_HOME` environment variable
- Normalize the resolved value to an absolute path before using it.
- If neither source yields a valid `PINOKIO_HOME`, stop immediately and ask the user how to proceed. Do not guess. Do not silently fall back to the current workspace.
- If the current workspace is outside the resolved `PINOKIO_HOME/api` and `PINOKIO_HOME/plugin` trees, treat the current workspace only as source material, reference material, or evidence. Do not create the launcher in that workspace.
- Before any file creation, record and verify the exact target path:
  - app launcher: `PINOKIO_HOME/api/<unique_name>`
  - plugin launcher: `PINOKIO_HOME/plugin/<unique_name>`
- If the unique folder name is not obvious, ask the user before creating the target folder.

### 1. App launchers
- App launchers must live under `PINOKIO_HOME/api/<unique_name>`.
- App launchers are usually project-local launchers that manage one app in its own launcher/app folder.
- If you are already inside the target app launcher folder, build in that folder.
- If you are not already inside an app launcher folder, create a new folder under `PINOKIO_HOME/api/<unique_name>`.
- If the folder name is not obvious from the project or the user has not provided one, ask the user to confirm the folder name before creating it.
- Do not place app launchers under `PINOKIO_HOME/plugin`.

### 2. Plugin launchers
- Plugin launchers must live under `PINOKIO_HOME/plugin/<unique_name>`.
- Plugin launchers are reusable shared tools that are installed once and then used across many different folders.
- Standalone plugin launchers should keep `path: "plugin"` in the root `pinokio.js` so Pinokio installs them into `PINOKIO_HOME/plugin`.
- If you are already inside the target plugin launcher folder, build in that folder.
- If you are not already inside a plugin launcher folder, create a new folder under `PINOKIO_HOME/plugin/<unique_name>`.
- If the folder name is not obvious from the project or the user has not provided one, ask the user to confirm the folder name before creating it.
- Do not place plugin launchers under `PINOKIO_HOME/api`.
- When a plugin is meant to operate on the user's current project, its `run` step should target the caller's folder with `{{args.cwd}}` instead of the plugin folder itself.

### 3. Apply structure rules only after choosing the launcher type
- App launchers and plugin launchers are peers. Do not treat a plugin launcher as a special case of an app launcher, or vice versa.
- Decide the launcher type and destination folder first, then apply the project structure and script rules below.

## Project Structure

Pinokio projects normally follow a standardized structure with app logic separated from launcher scripts:

Pinokio projects follow a standardized structure with app logic separated from launcher scripts:

```
project-root/
├── app/                 # Self-contained app logic (can be standalone repo)
│   ├── package.json     # Node.js projects
│   ├── requirements.txt # Python projects
│   └── ...              # Other language-specific files
├── README.md            # Documentation
├── install.js           # Installation script
├── start.js             # Launch script
├── update.js            # Update script (for updating the scripts and app logic to the latest)
├── reset.js             # Reset dependencies script
├── pinokio.js           # UI generator script
└── pinokio.json         # Metadata (title, description, icon)
```

- Keep app code in `/app` folder only (never in root)
- Store all launcher files in project root (never in `/app`)
- `/app` folder should be self-contained and publishable


The only exceptions are serverless web apps---purely frontend only web applications that do NOT have a server component and connect to 3rd party API endpoints--in which case the folder structure looks like the following (No need for launcher scripts since the index.html will automatically launch. The only thing needed is the metadata file named pinokio.json):

```
project-root/
├── index.html           # The serverless web app entry point
├── ...
├── README.md            # Documentation
└── pinokio.json         # Metadata (title, description, icon)
```

IMPORTANT: ALWAYS try to follow the best practices in the examples folder (/Users/salo/pinokio/prototype/system/examples) instead of trying to come up with your own structure. The examples have been optimized for the best user experience.

## Launcher Project Working Directory

- The project working directory for a script is always the same directory as the script location.
- For example, when you run `shell.run` API inside `pinokio/start.js`, the default path for shell execution is `pinokio`.
- If the launcher files are in the project root path, then the default path for shell execution is the project root.
- Therefore, it is important to specify the correct `path` attribute when running `shell.run` API commands.

Example: in the following project structure:

```
project-root/
├── pinokio/                 # Pinokio launcher folder
│    ├── start.js             # Launch script
│    ├── pinokio.js           # UI generator script
│    └── pinokio.json         # Metadata (title, description, icon)
└─── backend/
     ├── requirements.txt          # App dependencies
     └── app.py                    # App code
```

The `pinokio/start.js` should use the correct path `../backend` as the `path` attribute, as follows:

```
{
  run: [{
    ...
  }, {
    method: "shell.run",
    params: {
      message: "python app.py",
      venv: "env",
      path: "../backend"
    }
  }, {
    ...
  }]
}
```

## Development Workflow

### 1. Understanding the Project
- Check `SPEC.md` in project root. If the file exists, use that to learn about the project details (what and how to build)
- If no `SPEC.md` exists, build based on user requirements
### 2. Modifying Existing Launcher Projects
If we are starting with existing launcher script files, work with the existing files instead of coming up with your own.
- **Preserve existing functionality:** Only modify necessary parts
- **Don't touch working scripts:** Unless adding/updating specific commands
- **Follow existing conventions:** Match the style and structure already present
### 3. Try to adopt from examples as much as possible
- If starting from scratch, first determine what type of project you will be building, and then check the examples folder (/Users/salo/pinokio/prototype/system/examples) to see if you can adopt them instead of coming up everything from scratch.
- Even if there are no relevant examples, check the examples to get inspiration for how you would structure the script files even if you have to write from scratch.
### 4. Writing from scratch as a last resort
If there are relevant examples to adopt from, write the scripts from scratch, but just make sure to follow the requirements in the next section.
### 5. Debugging
When the user reports something is not working, ALWAYS inspect the logs folder to get all the execution logs. For more info on how this works, check the "Troubleshooting with Logs" section below.

## Script Requirements

### 1. 1-click launchable
- The main purpose of Pinokio is to provide an easy interface to invoke commands, which may include launching servers, installing programs, etc. Make sure the final product provides ways to install, launch, reset, and update whatever is needed.

### 2. Write Documentation
- ALWAYS write a documentation. A documentation must be stored as `README.md` in the project root folder, along with the rest of the pinokio launcher script files. A documentation file must contain:
  - What the app does
  - How to use the app
  - API documentation for programmatically accessing the app's main features (Javascript, Python, and Curl)

## Types of launchers
## 1. Launching servers
- When an app requires launching a server, here are the commonly used scripts:
  - `install.js`: a script to install the app
  - `start.js`: a script to start the app
  - `reset.js`: a script to reset all the dependencies installed in the `install.js` step. used if the user wants to restart from scratch
  - `update.js`: a script to update the launcher AND the app in case there are new updates. Involves pulling in the relevant git repositories installed through `install.js` (often it's the script repo and some git repositories cloned through the install steps if any)
  - `pinokio.js`: the launcher script that ties all of the above scripts together by providing a UI that links to these scripts.
  - `pinokio.json`: For metadata

Here's a basic server launcher script example (`start.js`). Unless there's a special reason you need to use another pattern, this is the most recommended pattern. Use this or adopt it as needed, but NEVER try something else unless there's a good reason you should not take this approach:

```javascript
module.exports = {
  // By setting daemon: true, the script keeps running even after all items in the `run` array finishes running. Mandatory for launching servers, since otherwise the shells running the server process will get killed after the scripts finish running.
  daemon: true,
  run: [
    {
      // The "shell.run" API for running a shell session
      method: "shell.run",
      params: {
        // Edit 'venv' to customize the venv folder path
        venv: "env",
        // Edit 'env' to customize environment variables (see documentation)
        env: { },
        // Edit 'path' to customize the path to start the shell from
        path: "app",
        // Edit 'message' to customize the commands, or to run multiple commands
        message: [
          "python app.py",
        ],
        on: [{
          // The regular expression pattern to monitor.
          // Whenever each "event" pattern occurs in the shell terminal, the shell will return,
          // and the script will go onto the next step.
          // The regular expression match object will be passed on to the next step as `input.event`
          // Useful for capturing the URL at which the server is running (in case the server prints some message about where the server is running)
          "event": "/(http:\/\/\\S+)/", 

          // Use "done": true to move to the next step while keeping the shell alive.
          // Use "kill": true to move to the next step after killing the shell.
          "done": true
        }]
      }
    },
    {
      // This step sets the local variable 'url'.
      // This local variable will be used in pinokio.js to display the "Open WebUI" tab when the value is set.
      method: "local.set",
      params: {
        // the input.event is the regular expression match object from the previous step
        // In this example, since the pattern was "/(http:\/\/\\S+)/", input.event[1] will include the exact http url match caputred by the parenthesis.
        // Therefore setting the local variable 'url'
        url: "{{input.event[1]}}"
      }
    }
  ]
}
```

## 2. Launching serverless web apps

- In case of purely static web apps WITHOUT servers or backends (for example an HTML based app that connects to 3rd party servers--either remote or localhost), we do NOT need the launcher scripts.
- In these cases, simply include `index.html` in the project root folder and everything should automatically work. No need for any of the pinokio launcher scripts. (Do 
- You still need to include the metadata file so they show up properly on pinokio:
  - `pinokio.json`: For metadata

## 3. Launching quick scripts without web UI

- In many cases, we may not even need a web UI, but instead just a simple way to run scripts.
- This may include TUI (Terminal User Interface) apps, a simple launcher 
- In these cases, all we need is the launcher file `pinokio.js`, which may link to multiple scripts. In this case, there are no web apps (no serverless apsp, no servers), but instead just the default pinokio launcher UI that calls a bunch of scripts.
- Here are some examples:
  - A pinokio script to toggle the desktop theme between dark and light
    - Write some code (python or javascript or whatever)
    - Write a `toggle.js` pinokio script that executes the code
    - Write a `pinokio.js` launcher script to create a sidebar UI that displays the `toggle.js` so the user can simply click the "toggle" button to toggle back and forth between desktop themes
  - A pinokio script to fetch some file
    - Write some code (python or javascript or whatever)
    - Write a `fetch.js` pinokio script that executes the code
    - Write a `pinokio.js` launcher script to create a sidebar UI that displays the `fetch.js` so the user can simply click the "fetch" button to fetch some data.
- You still need to include the metadata file so they show up properly on pinokio:
  - `pinokio.json`: For metadata

## API

This section lists all the script APIs available on Pinokio. To learn the details of how they are used, you can:
1. Check the examples in the /Users/salo/pinokio/prototype/system/examples folder
2. Read the `PINOKIO.md` at /Users/salo/pinokio/prototype/PINOKIO.md further documentation on the full syntax

### Script API

These APIs can be used to describe each step in a pinokio script:
- shell.run: run shell commands
- input: accept user input
- filepicker: accept file upload
- fs.write: write to file
- fs.read: read from file
- fs.copy: copy files
- fs.download: download files
- fs.link: create a symbolic link (or junction on windows) for folders
- fs.open: open the system file explorer at a given path
- fs.cat: print file contents
- jump: jump to a specific step
- local.set: set local variables for the currently running script
- json.set: update a json file
- json.rm: remove keys from a json file
- json.get: get values from a json file
- log: print to the web terminal
- net: make network requests
- notify: display a notification
- script.download: download a script from a git uri
- script.start: start a script
- script.stop: stop a script
- script.return: return values if the current script was called by a caller script, so the caller script can utilize the return value as `input`
- web.open: open a url in web browser
- hf.download: huggingfac-cli download API
### Template variables
The following variables are accessible inside template expressions (example `{{args.command}` in scripts, resulting in dynamic behaviors of scripts:
- input: An input is a variable that gets passed from one RPC call to the next
- args: args is the parameter object that gets passed into the script (via pinokio.js `params`). Unlike `input` which takes the value passed in from the immediately previous step, `args` is a global value that is the same through out the entire script execution.
- local: local variable object that can be set with `local.set` API
- self: refers to the script file itself (which is JSON or JavaScript). For example if `start.js` that's currently running has `daemon: true` set, `{{self.daemon}}` will evaluate to true.
- uri: The current script uri
- port: The next available port. Very useful when you need to launch an app at a specific port without port conflicts.
- cwd: The current script execution folder path
- platform: The current operating system. May be one of the following: `darwin`, `win32`, `linux`
- arch: The current system architecture. May be one of the following: x32, x64, arm, arm64, s390, s390x, mipsel, ia32, mips, ppc, ppc64
- gpus: array of available GPUs on the machine (example: `['apple']`, `['nvidia']`)
- gpu: the first available GPU (example: `nvidia`)
- current: The current variable points to the index of the currently executing instruction within the run array.
- next: The next variable points to the index of the next instruction to be executed. (null if the current instruction is the final instruction in the run array)
- envs: You can access the environment variables of the currently running process with envs object.
- which: Check whether a command exists and return its absolute path (example: `{{which('winget')}}`). This is the correct way to resolve command paths inside reproducible Pinokio scripts, including custom shell selection such as `shell: "{{which('bash')}}"`. If you are outside a Pinokio-managed shell and only need to inspect Pinokio's environment manually, use `pterm which <command>`, but do NOT copy that user-specific absolute path into launcher scripts.
- exists: Check whether a file or folder exists at the specified relative path (example: `"when": "{{!exists('app')}}"`). Can be used with the `when` attribute to determine a path's existence and trigger custom logic. Use relative paths and it will resolve automatically to the current execution folder. 
- running: Check whether a script file is running (example: `"when": "{{!running('start.js')}}"`). Can be used with the `when` attribute to determine a path's existence and trigger custom logic. Use relative paths and it will resolve automatically to the current execution folder. 
- os: Pinokio exposes the node.js os module through the os variable.
- path: Pinokio exposes the node.js path module through the os variable (example: `{{path.resolve(...)}}`

## System Capabilities
### Package Management (Use in Order of Preference)
The following package managers come pre-installed with Pinokio, so whenever you need to install a 3rd party binary, remember that these are available. Also, you can assume these are available and include the following package manager commands in Pinokio scripts:
1. **UV** - For Python packages (preferred over pip)
2. **NPM** - For Node.js packages  
3. **Conda** - For cross-platform 3rd party binaries
4. **Brew** - Mac-only fallback when other options unavailable
5. **Git** - Full access to git is available.
6. **Bun** - For managing bun packages
**Important:** Include all install commands in the install script for reproducibility.
### HTTPS Proxy Support
- All HTTP servers automatically get HTTPS endpoints
- Convention: `http://localhost:<PORT>` → `https://<PORT>.localhost`
- Full proxy list available at: `http://localhost:2019/config/`
### Pterm Features:
- **Clipboard Access:** Read from or Write to system clipboard via pinokio Pterm CLI (`pterm clipboard` command.)
- **Notifications:** Send desktop alerts via pinokio pterm CLI (`pterm push` command.)
- **Script Testing:** Run launcher scripts via pinokio pterm CLI (`pterm start` command.)
- **File Selection:** Use built-in filepicker for user file/folder input (`pterm filepicker` command.)
- **Command Path Resolution:** Inspect the absolute path of any command as seen by Pinokio via `pterm which <command>`. Use this for debugging or external local tooling, especially when a helper process did not inherit Pinokio's `PATH`, for example `pterm which bash` on Windows. Do NOT hardcode the returned absolute path into launcher scripts; use `which()` or `kernel.which()` in the script itself instead.
- **Git Operations:** Clone repositories, push to GitHub
- **GitHub Integration:** Full GitHub CLI support (`gh` commands)

## Troubleshooting with Logs
Pinokio stores the logs for everything that happened in terminal at the following locations, so you can make use of them to determine what's going on:

### Log Structure
In case there is a `pinokio` folder in the project root folder, you should be able to find the logs folder here:

```
pinokio/
└── logs/   # Direct user interaction logs
    ├── api/     # Launcher script logs (install.js, start.js, etc.)
    ├── dev/     # AI coding tool logs (organized by tool)
    └── shell/   # Direct user interaction logs
```

Otherwise, the `logs` folder should be found at project root:

```
logs/
├── api/     # Launcher script logs (install.js, start.js, etc.)
├── dev/     # AI coding tool logs (organized by tool)
└── shell/   # Direct user interaction logs
```

### Log File Naming
- Unix timestamps for each session
- Special "latest" file contains most recent session logs
- **Default:** Use "latest" files for current issues
- **Historical:** Use timestamped files for pattern analysis and the full history.

## Best practices
### 0. Always reference the logs when debugging
- When the user asks to fix something, ALWAYS check the logs folder first to check what went wrong. Check the "Troubleshooting with Logs" section.
### 1. Shell commands for launching programs
- Launch flags related
  - Try as hard as possible to minimize launch flags and parameters when launching an app. For example, instead of `python app.py --port 8610`, try to do `python app.py` unless really necessary. The only exception is when the only way to launch the app is to specify the flags.
- Launch IP related
  - Always try to find a way to launch servers at 127.0.0.1 or localhost, often by specifying launch flags or using environment variables. Some apps launch apps at 0.0.0.0 by default but we do not want this.
- Launch Port related
  - In case the app itself automatically launches at the next available port by default (for example Gradio does this), do NOT specify port, since it's taken care of by the app itself. Always try to minimize the amount of code.
  - If the install instruction says to launch at a specific port, don't use the hardcoded port they suggest since there's a risk of port conflicts. Instead, use Pinokio's `{{port}}` template expression to automatically get the next available port.
  - For example, if the instruction says `python app.py --port 7860`, don't use that hardcoded port since there might be another app running at that port. Instead, automatically assign the next available port like this: `python app.py --port {{port}}`
  - Note that the `{{port}}` expression always returns the next immediately available port for each step, so if you have multiple steps in a script and use `{{port}}` in multiple steps, the value will be different. So if you want to launch at the next available port and then later reuse that port, you will need to first use `{{port}}` to get the next available port, and save the value in local variable using `local.set`, and then use the `{{local.<variable_name>}}` expression later.
### 2. shell.run API
- When writing `shell.run` API requests, always use relative paths (no absolute paths) for the `path` field. For example, if you need to run a command from `app` folder, the `path` attribute should simply be `app`, instead of its full absolute path.
- If a launcher needs to use a command that Pinokio already provides, prefer resolving it with `{{which('command')}}` inside the script instead of assuming the command name will always be on `PATH`.
- Do NOT automatically avoid `bash`-based install commands on Windows. Pinokio's Windows environment includes `bash` through its bundled toolchain, so commands such as `curl -fsSL ... | bash` are acceptable when they run inside a Pinokio-managed shell and there is no simpler cross-platform alternative.
- If a Windows launcher needs to run the shell itself in bash instead of the default `cmd.exe`, set `shell: "{{which('bash')}}"` on the `shell.run` step.
- If a separate debugging process or external local tool did not inherit Pinokio's environment, you may use `pterm which <command>` to inspect what Pinokio would resolve. Do NOT turn that result into a hardcoded script path; for launcher scripts, always use `which()` or `kernel.which()` so the script stays reproducible across machines.
### 2. Package managers
- When installing python packages, try best to use `uv` instead of `pip` even if the install instruction says to use pip. Instead of `pip install -r requirements.txt`, you can simply use `uv pip install -r requirements.txt` for example. Even if the project's own README says use pip or poetry, first check if there's a way to use uv instead.
- When you need to install some global package, try to use `conda` as much as possible. Even on macs, `brew` should be only used if there are no `conda` options.
### 3. Minimal Always
- If you are starting with existing script files, before modifying, creating, or removing any script files, first look at `pinokio.js` to understand which script files are actually used in the launcher. The only script files used are the ones mentioned in the `pinokio.js` file. The `pinokio.js` file is the file that constructs the UI dynamically.
- Do not create a redundant script file that does something that already exists. Instead modify the existing script file for the feature. For example, do not create an `install.json` file for installation if `install.js` already exists. Instead, modify the `install.js` file.
- Pinokio accepts both JSON and JS script files, so when determining whether a script for a specific purpose already exists, check both JSON and JS files mentioned in the `pinokio.js` file. Do not create script files for rendundant purpose.
- When building launchers for existing projects cloned from a repository, try to stay away from modifying the project folder (the `/Users/salo/pinokio/api/phosphene.git` folder), even if installations are failing. Instead, try to work around it by creating additional files in the launcher folder, and using those files IN ADDITION to the default project.
  - The only exception when you may need to make changes to the project folder is when the user explicitly wants to modify the existing project. Otherwise if the purpose is to simply write a launcher, the app logic folder should never be touched.
- When running shell commands, take full advantage of the Pinokio `shell.run` API, which provides features like `env`, `venv`, `input`, `path`, `sudo`, `on`, etc. which can greatly reduce the amount of script code.
  - Python apps: Always use virtual environments via `venv` attribute. This attribute automatically creates a venv or uses if it already exists.
### 4. Try to support Cross-platform as much as possible
- Use cross-platform shell commands only.
- This means, prefer to use commands that work on all platforms instead of the current platform.
- If there are no cross platform commands, use Pinokio's template expressions to conditionally use commands depending on `platform`, `arch`, etc.
- Also try to utilize Pinokio Pterm APIs for various cross-platform system features.
- If it is impossible to implement a cross platform solution (due to the nature of the project itself), set the `platform`, `arch`, and/or `gpu` attributes of the `pinokio.json` file to declare the limitation.
- Pinokio provides various APIs for cross-platform way of calling commonly used system functions, or lets you selectively run commands depending on `platform`, `arch`, etc.
### 5. Do not make assumptions about Pinokio API
- Do NOT make assumptions about which Pinokio APIs exist. Check the documentation.
- Do NOT make assumptions about the Pinokio API syntax. Follow the documentation.
### 6. Scripts must be able to replicate install and launch steps 100%
- The whole point of the scripts is for others to easily download and invoke them via Pinokio interface with one click. Therefore, do not assume the end user's system state, and make everything self-contained.
- When a 3rd party package needs to be installed, or a 3rd party repository needs to be downloaded, include them in the scripts.
### 7 Dynamic UI rendering
- The `pinokio.js` launcher script can change dynamically depending on the current state of the script execution. Which means, depending on what the file returns, it can determine what the sidebar looks like at any given moment of the script cycle.
  - `info.exists(relative_path)`: The `info.exists` can be used to check whether a relative path (relative to the script root path) exists. The `pinokio.js` file can determine which menu items to return based on this value at any given moment.
  - `info.running(relative_path)`: The `info.running` can be used to check whether a script at a relative path is currently running (relative to the script root path) exists. The `pinokio.js` file can determine which menu items to return based on this value at any given moment.
  - `info.local(relative_path)`: The `info.local` can be used to return all the local variables tied to a script that's currently running. The `pinokio.js` file can determine which menu items to return based on this value at any given moment.
  - `default`: set the `default` attribute on any menu item for whichever menu needs to be selected by default at a given step. Some example scenarios:
    - during the install process, the `install.js` menu item needs to be set as the `default`, so it automatically executes the script
    - when launching the `start.js` menu item needs to be set as the `default`, so it automatically executes the script
    - after the app has launched, the `default` needs to be set on the web UI URL, so the user is sent to the actual app automatically.
  - Check the examples in the /Users/salo/pinokio/prototype/system/examples folder to see how these are being used.
### 8. No need for stop scripts
- `pinokio.js` does NOT need a separate `stop` script. Every script that can be started can also be natively stopped through the Pinokio UI, therefore you do not need a separate stop script for start script
### 9. Writing launchers for existing projects
- When writing or modifying pinokio launcher scripts, figure out the install/launch steps by reading the project folder `app`.
- In most cases, the `README.md` file in the `/Users/salo/pinokio/api/phosphene.git` folder contains the instructions needed to install and run the app, but if not, figure out by scanning the rest of the project files.
- Install scripts should work for each specific operating system, so ignore Docker related instructions. Instead use install/launch instructions for each platform.
### 10. Retrofitting an already-working setup
- Sometimes the user starts outside Pinokio, gets an app working through ad-hoc commands, and only later asks to turn that work into a Pinokio launcher.
- In this case, treat the current working setup and the successful session context as the highest priority source of truth. Do NOT restart from scratch if the app is already working.
- First capture the exact install and launch steps that already succeeded: cloned repositories, package manager commands, environment variables, model downloads, ports, working directories, helper scripts, and any fixes that were required.
- Then convert that knowledge into reproducible Pinokio scripts (`install.js`, `start.js`, `reset.js`, `update.js`, `pinokio.js`, `pinokio.json`) instead of telling the user to manually repeat the ad-hoc process.
- When the successful setup lives in a non-Pinokio folder, use that folder as evidence only. Resolve `PINOKIO_HOME` first, then produce the final launcher in the proper Pinokio location (`PINOKIO_HOME/api/<unique_name>` or `PINOKIO_HOME/plugin/<unique_name>`) unless the user explicitly asks for another layout.
- Replace machine-specific state with reproducible steps. Never hardcode absolute paths, user-specific cache locations, session-only ports, or one-off manual edits if they can be expressed in the launcher.
- Do not simply encode whatever happened to work on the current machine. Generalize the result into the broadest practical cross-platform, cross-machine launcher, and if limitations are unavoidable, declare them explicitly in `pinokio.json` instead of silently baking in local assumptions.
- If the app is already installed but the exact setup steps are partially missing, inspect the current working tree, generated files, dependency manifests, shell history when available, and logs to reconstruct the smallest reliable install and start flow.
- Verify from as clean a state as practical. A launcher is only done when another user could reproduce the working result without relying on undocumented steps from the original ad-hoc session.
### 11. Don't use Docker unless really necessary
- Some projects suggest docker as installation options. But even in these cases, try to find "development" options to launch the app without relying on Docker, as much as possible. We do not need Docker since we can automatically install and launch apps specifically for the user's platform, since we can write scripts that run cross platform.
### 12. pinokio.json
- Do not touch the `version` field since the version is the script schema version and the one pre-set in `pinokio.js` must be used.
- `icon`: It's best if we have a user friendly icon to represent the app, so try to get an image and link it from `pinokio.json`.
  - If the git repository for the `/Users/salo/pinokio/api/phosphene.git` folder points to GitHub (for example https://github.com/<USERNAME>/<REPO_NAME>`, ask the user if they want to download the icon from GitHub, and if approved, get the `avatar_url` by fetching `https://api.github.com/users/<USERNAME>`, and then download the image to the root folder as `icon.png`, and set `icon.png` as the `icon` field of the `pinokio.json`. 
### 13. Gitignore
- When a launcher involves cloning 3rd party repositories, downloading files dynamically, or some files to be generated, these need to be included in the .gitignore file. This may include things like:
  - Cloning git repositories
  - Downloading files
  - Dynamically creating files during installation or running, such as Sqlite Databases, or environment variables, or anything specific to the user.
- Make sure these file paths are included in the .gitignore file, and if not, include them in .gitignore.

## AI Libraries (Pytorch, Xformers, Triton, Sageattention, etc.)
If the launcher is for running AI models locally, the install script must declare the AI bundle so Pinokio can install the machine-level prerequisites before the script runs:

```
// install.js
module.exports = {
  requires: {
    bundle: "ai"
  },
  run: [
    ...
  ]
}
```

This is required even when the script also uses `torch.js`. The AI bundle is what triggers installation of common local AI prerequisites such as CUDA on NVIDIA systems and Hugging Face CLI.

If the launcher has a dedicated built-in script named `torch.js`, it can be used as follows:

```
// install.js
module.exports = {
  requires: {
    bundle: "ai"
  },
  run: [
    // Edit this step with your custom install commands
    {
      method: "shell.run",
      params: {
        venv: "venv",                // Edit this to customize the venv folder path
        path: "app",
        message: [
          "uv pip install -r requirements.txt"
        ],
      }
    },
    // Delete this step if your project does not use torch
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          path: "app",
          venv: "venv",                // Edit this to customize the venv folder path
          // xformers: true   // uncomment this line if your project requires xformers
          // triton: true   // uncomment this line if your project requires triton
          // sageattention: true   // uncomment this line if your project requires sageattention
          // flashattention: true   // uncomment this line if your project requires flashattention
        }
      }
    },
  ]
}
```

The `torch.js` script also includes ways to install pytorch dependent libraries such as xformers, triton, sagetattention. If any of these libraries need to be installed, use the torch.js to install in order to install them cross platform.


## Quick Reference
### Essential Documentation
- **Pinokio Programming:** See `PINOKIO.md` at /Users/salo/pinokio/prototype/PINOKIO.md → "Programming Pinokio" section
- **Dynamic Menus:** See `PINOKIO.md` at /Users/salo/pinokio/prototype/PINOKIO.md → "Dynamic menu rendering" section  
- **CLI Commands:** See `PTERM.md` at /Users/salo/pinokio/prototype/PTERM.md
### Common Patterns
- **Python Virtual Env:** `shell.run` with `venv` attribute
- **Cross-platform Commands:** Always test on multiple platforms
- **Error Handling:** Check logs/api for launcher issues
- **GitHub Operations:** Use `gh` CLI for advanced GitHub features
## Development Principles
1. **Minimize Shell Usage:** Leverage API parameters instead of raw commands
2. **Maintain Separation:** Keep app logic and launchers separate
3. **Follow Conventions:** Match existing project patterns
4. **Test Thoroughly:** Use CLI to verify launcher functionality
5. **Document Changes:** Update relevant metadata and documentation
