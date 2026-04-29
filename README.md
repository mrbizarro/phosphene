# LTX23MLX — local control panel for LTX Video 2.3 on Apple Silicon

A polished web panel that wraps the [`dgrauet/ltx-2-mlx`](https://github.com/dgrauet/ltx-2-mlx) MLX port of LTX Video 2.3. Runs entirely locally on Apple Silicon Macs — no cloud, no API keys, no subscription. Persistent batch queue, warm pipeline subprocess, image-to-video with auto cover-crop, lossless h264 output, output gallery with hide/unhide, extend mode for longer clips, and crash-safe queue resume.

The CLI works but isn't ergonomic for batch generation. ComfyUI on Apple Silicon is currently broken/CPU-bound for LTX 2.3 (8 min per step on M4 Max in some reports). This panel fills the gap.

---

## Credits — upstream work this is built on

This is a wrapper, not a fork. All the hard model work belongs to the people below. The panel adds workflow, UX, and a few small fixes.

- **[Lightricks](https://github.com/Lightricks/LTX-Video)** — original LTX Video model. The architecture, training, weights, and the joint audio+video design are all theirs. LTX 2.3 22B distilled is what generates every clip.
- **[@dgrauet](https://github.com/dgrauet/ltx-2-mlx)** — the MLX port. Without their work there is no LTX on Apple Silicon worth talking about. Pure Apple-Metal port of the LTX-2 reference, three-package monorepo (inference, pipelines, training), and the Q4/Q8 quantized model bundles on Hugging Face.
- **MLX team @ Apple** — the framework that makes Metal-native ML on Mac fast enough to be worth running locally.
- **[mlx-community](https://huggingface.co/mlx-community)** — the Gemma-3-12B-it-4bit text encoder distribution.

---

## What this adds on top of the upstream stack

The pieces below are what justify the panel existing as its own project. Most are not in upstream `ltx-2-mlx`, and a couple are bug fixes I plan to PR back.

| What | Why it matters |
|---|---|
| **Persistent batch queue** with crash-resume | Submit 60 prompts, sleep, wake up to 60 clips. State written to `panel_queue.json` so a panel/Mac restart doesn't lose work-in-progress. |
| **Warm helper subprocess** holding MLX pipelines | Spawning the CLI fresh per job is ~100 lines of redundant Python startup. Helper holds T2V / I2V / Extend pipelines lazily, auto-exits after 30 min idle. |
| **Image cover-crop pre-flight** | Drop a 4K image into I2V; pipeline cover-crops to model-safe dims (multiples of 32) without distortion. Upload via drag-and-drop or path. |
| **Lossless h264 output** *(yuv444p, crf 0)* | Upstream pipeline encodes with `yuv420p crf 18` — 4:2:0 chroma subsampling produces visible JPEG-style block artifacts on faces. Patched to `yuv444p crf 0` (no chroma subsampling, mathematically lossless), with env-var overrides. **Will PR upstream.** |
| **Aspect + duration-driven sizing** | Pick aspect from a dropdown, type seconds, frames auto-snap to LTX's required `8k+1` rule. No more manually computing frame counts. |
| **Extend mode** | Chain clips beyond the single-shot frame budget. Click "⏭ Extend" on any output, get one-click setup with that clip as the source. |
| **Output gallery** with auto-thumbnails, hide/unhide, sidecar JSONs | Every render writes `.mp4.json` with full params. "↩ Load params" reproduces any past clip. Hide videos from the panel without deleting from disk. |
| **Live status pills** — memory, swap, Comfy, helper, queue, job | Health dashboard at the top: never wonder if the Mac is swapping or if something silently died. |
| **Auto-detect & stop ComfyUI** before 720p renders | ComfyUI idle costs ~27 GB; both alive at 720p+ thrashes on 64 GB Macs. Panel kills Comfy via `pgrep` (no hardcoded PID), opt-in per render. |
| **`caffeinate -i`** while queue is non-empty | Display can sleep, system can't. Released automatically when queue drains. |
| **Diegetic-dialogue prompt patterns documented** | LTX 2.3 produces voiceover by default; with the right phrasing (voice-quality descriptor + mouth beat before quote + tight framing) it produces in-mouth lipsync. Documented and enforced via the prompt examples in this repo. |
| **Stop-current vs pause-queue separation** | Stop kills the running render and advances to the next; Pause holds the queue while the current finishes. |

---

## Requirements

- Apple Silicon Mac (M1+). Memory bandwidth bottlenecks dominate — M4 Max / Ultra are 4–6× faster than base M-series.
- macOS 14+
- 32 GB RAM minimum (Q4); 64 GB+ recommended for 720p+ or longer clips
- ~80 GB free disk for the Q4 model bundle
- Python 3.11 (for the MLX pipeline). The panel itself runs on Python 3.9+ stdlib only.
- ffmpeg with libx264 + AAC. Pinokio's bundled ffmpeg is auto-detected; Homebrew also works.

## Install

### Option A — Pinokio one-click (recommended)

1. Open Pinokio
2. **Discover → Download from URL** → paste `https://github.com/mrbizarro/LTX23MLX`
3. Click **Install**. Pinokio handles the rest:
   - Apple-Silicon hardware gate (refuses to install on Intel / Linux / Windows)
   - Clones `dgrauet/ltx-2-mlx`, creates a Python 3.11 venv via `uv`, installs the MLX pipelines
   - Applies the lossless h264 codec patch (idempotent, runs again on update)
   - Downloads Q4 model (`dgrauet/ltx-2.3-mlx-q4`, ~25 GB) and Gemma 4-bit (~6 GB) via `huggingface-cli` (resumes if interrupted)
4. Click **Start** → **Open Panel** → http://127.0.0.1:8198

For the High quality tier (Q8 two-stage + TeaCache), download the Q8 model separately afterward (one-time, ~25 GB extra). See [Quality tiers](#quality-tiers) below.

### Option B — manual

```bash
# 1. Clone this panel
git clone https://github.com/mrbizarro/LTX23MLX.git
cd LTX23MLX

# 2. Clone ltx-2-mlx alongside (default panel layout assumes ./ltx-2-mlx/)
git clone https://github.com/dgrauet/ltx-2-mlx.git ltx-2-mlx
cd ltx-2-mlx
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e packages/ltx-core-mlx packages/ltx-pipelines-mlx
deactivate
cd ..

# 3. Apply the lossless-output patch (until upstream PR lands)
#    See PATCH below — one-line edit to ltx-core-mlx/.../video_vae.py

# 4. Run the panel — first generation auto-downloads model weights from HF
python3 mlx_ltx_panel.py
# → http://127.0.0.1:8198
```

First render will pull `dgrauet/ltx-2.3-mlx-q4` (~25 GB) and `mlx-community/gemma-3-12b-it-4bit` (~6 GB) from Hugging Face into `~/.cache/huggingface/`. Subsequent renders are instant to start.

## Quality tiers

The panel exposes three render quality tiers via a dropdown in the form:

| Tier | Model | Mode | Steps | ~Time (5s clip) | Use case |
|---|---|---|---|---|---|
| **Draft** | Q4 distilled | one-stage | 4 | ~3 min | Prompt scouting / seed picking |
| **Standard** *(default)* | Q4 distilled | one-stage | 8 | ~7 min | Most renders |
| **High** | **Q8** | two-stage HQ + TeaCache | stage1=15, stage2=3 | ~12 min | Keeper renders. Better face fidelity (Q8 + dev model + CFG anchor). |

Same-seed re-roll is the workflow: pick a seed at Draft (cheap), confirm at Standard, finalize at High when it matters. Each output's sidecar JSON has the seed; click "↩ Load params" to recreate the same shot at a different tier.

### Enabling High tier (optional, ~25 GB extra)

Q8 is a separate download. Run this one-time inside the Pinokio install dir (or the manual install dir):

```bash
huggingface-cli download dgrauet/ltx-2.3-mlx-q8 --local-dir mlx_models/ltx-2.3-mlx-q8
```

The panel auto-detects Q8 on disk and enables the High option in the dropdown.

## Configuration via env vars

The panel auto-detects everything by default but every path is overridable:

| Env var | Default | What |
|---|---|---|
| `LTX_STUDIO_ROOT` | dir of the panel script | repo root |
| `LTX_MLX_PATH` | `$ROOT/ltx-2-mlx` | ltx-2-mlx clone location |
| `LTX_HELPER_PYTHON` | `$LTX_MLX_PATH/.venv/bin/python3.11` | the Python that runs the MLX pipeline |
| `LTX_GEMMA` | `mlx-community/gemma-3-12b-it-4bit` (HF) | Gemma encoder — set to local path to skip HF download |
| `LTX_MODEL` | `dgrauet/ltx-2.3-mlx-q4` | LTX model id; flip to q8 if you have it |
| `LTX_FFMPEG` | auto (PATH → Pinokio → Homebrew) | ffmpeg binary |
| `LTX_PORT` | `8198` | panel HTTP port |
| `LTX_HELPER_IDLE_TIMEOUT` | `1800` | helper auto-exits after this many seconds idle |
| `LTX_HELPER_LOW_MEMORY` | `true` | drop pipeline weights between jobs to free RAM |
| `LTX_OUTPUT_PIX_FMT` | `yuv444p` | output codec (lossless default) |
| `LTX_OUTPUT_CRF` | `0` | h264 crf — 0=lossless, 18=visually lossless w/ smaller files |

## Patch — until upstream PR lands

In `ltx-2-mlx/packages/ltx-core-mlx/src/ltx_core_mlx/model/video_vae/video_vae.py`, find the line that builds the ffmpeg encode command (around line 379) and swap the codec:

```python
# OLD — produces JPEG-style chroma artifacts on faces:
cmd.extend(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", output_path])

# NEW — lossless, no chroma subsampling, env-overridable:
import os as _os
_pix = _os.environ.get("LTX_OUTPUT_PIX_FMT", "yuv444p")
_crf = _os.environ.get("LTX_OUTPUT_CRF", "0")
cmd.extend(["-c:v", "libx264", "-pix_fmt", _pix, "-crf", _crf, output_path])
```

## Performance reference (base M4 Mac Studio, 64 GB)

| Resolution | Frames | Steps | ~Time per render |
|---|---|---|---|
| 512×288 | 25 (1s) | 8 | ~30–60s |
| 1280×704 | 121 (5s) | 8 | ~7 min |
| 1280×704 | 241 (10s) | 8 | ~20 min |
| 1408×800 | 121 (5s) | 8 | ~9 min |

M-Max tier divides by ~3–4×. M-Ultra by ~6×.

## Roadmap

- [ ] Submit the lossless-output patch as a PR upstream to `dgrauet/ltx-2-mlx`
- [ ] **Pinokio one-click installer** — Pinokio's native `hf.download` step handles the 60 GB download with resume + progress out of the box; only Apple-Silicon gating needs custom logic. Reference templates: [pinokiofactory/MFLUX-WEBUI](https://github.com/pinokiofactory/MFLUX-WEBUI) (closest stylistic match — also Mac-only MLX), [pinokiofactory/comfy](https://github.com/pinokiofactory/comfy) (canonical reusable HF-download pattern via `hf.json` sub-script), [pinokiofactory/hunyuanvideo](https://github.com/pinokiofactory/hunyuanvideo) (precedent for tens-of-GB installer)
- [ ] Q8 + two-stage mode for higher face fidelity (per Lightricks recommendations)
- [ ] Auto-chain Extend (one-click 15s+ via base + extend×2 stitched)
- [ ] Lazy first-run model downloader inside the panel itself (parallel to Pinokio path — for users who don't use Pinokio)
- [ ] Env-var override for output destination dir per run

### Pinokio install — design notes

The installer is a small folder with `pinokio.json` (metadata), `pinokio.js` (sidebar menu), `install.js` (clone → venv → `uv pip install` → `hf.download` × 2 → patch step), `start.js` (daemon-launch the panel and capture the localhost URL), and a reusable `hf.json` for the model downloads. Apple-Silicon gate is `{{platform !== 'darwin' || arch !== 'arm64'}}` + `notify` + `next: null` at the top of `install.js`. To get into Pinokio's "Discover" feed, tag the repo `pinokio` on GitHub; for an officially featured listing, publish under [pinokiofactory](https://github.com/pinokiofactory).

## License

MIT — see [LICENSE](LICENSE). LTX Video 2.3 weights are subject to Lightricks' license — read theirs separately.
