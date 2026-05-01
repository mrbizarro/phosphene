<p align="center">
  <img src="assets/phosphene_banner.png" alt="Phosphene" width="100%">
</p>

<p align="center">
  <strong>Local video and audio generation for Apple Silicon.</strong><br>
  A free desktop panel for <a href="https://github.com/Lightricks/LTX-Video">LTX 2.3</a> running natively in <a href="https://github.com/ml-explore/mlx">MLX</a>.<br>
  One-click install via <a href="https://pinokio.computer">Pinokio</a>. No cloud, no API key, no subscription.
</p>

<p align="center">
  <a href="https://x.com/AIBizarrothe">
    <img src="assets/bizarro-avatar.jpg" width="40" height="40" style="border-radius: 50%;" alt="by Mr. Bizarro on X">
  </a>
  <br>
  <em>by <a href="https://x.com/AIBizarrothe">Mr. Bizarro</a> on X</em>
</p>

---

## What it looks like

<p align="center">
  <img src="assets/screenshot_panel_full.jpg" alt="Phosphene panel — full UI with mode pills, prompt area, output gallery" width="100%">
</p>

Single page, no node graph. Pick a mode, type a prompt, hit Generate.
Outputs land in the gallery on the right. Every clip carries audio.

<p align="center">
  <img src="assets/screenshot_modes.jpg" alt="Mode and Quality pill selectors" width="65%">
</p>

---

## What's different

The differentiator is **audio**. LTX 2.3 generates video and audio in
**one forward pass** — they share the diffusion process, so timing is
tied at the frame level. Footsteps land on the right frame. Lip
movement matches dialogue. Ambient hum is conditioned on what you see.

| | Phosphene | Wan / Hunyuan / Mochi (Mac) | Pika / Runway | ComfyUI + LTX 2.3 |
|---|---|---|---|---|
| Joint audio + video | ✅ one pass | ❌ silent | ✅ (cloud) | ✅ |
| Native MLX (no torch shim) | ✅ | ❌ MPS shim | n/a | ❌ |
| Local, no API | ✅ | ✅ | ❌ | ✅ |
| One-click install | ✅ Pinokio | varies | n/a | ❌ node graph |
| Persistent batch queue | ✅ crash-resume | ❌ | ✅ | ❌ |
| Lossless H.264 output | ✅ yuv444p crf 0 | yuv420p | varies | yuv420p |

---

## Modes

Four generation modes. All produce video + synced audio.

| Mode | Inputs | Use case |
|---|---|---|
| **T2V** — text → video | prompt | The default. Type a scene, get 5 seconds with sound. |
| **I2V** — image → video | prompt + reference image | Animate a still. Auto cover-crop to model dimensions. |
| **FFLF** — first / last frame | prompt + start image + end image | Two images bookend the clip; the model fills the motion between. Requires Q8. |
| **Extend** — continue a clip | existing mp4 + prompt | Append seconds onto a previous render. Audio continuous across the join. |

Plus a **Prompt Enhance** button that uses Gemma 3 12B (4-bit, locally)
to rewrite your prompt in the structure LTX 2.3 was trained on.

---

## Quality tiers

Three render levels picked per-job. All use the same prompt; the model
and step count change.

| Tier | Model | Time @ 1280×704 | Use case |
|---|---|---|---|
| **Draft** | Q4 distilled | ~2 min (half resolution) | Iterate on prompts and seeds before committing. |
| **Standard** | Q4 distilled | ~7 min | The daily driver. Q4 weights (~25 GB on disk). |
| **High** | Q8 two-stage + TeaCache | ~12 min | Sharper detail, fewer artifacts on faces and text. Optional Q8 download (~25 GB extra). Required for FFLF. |

---

## Hardware tiers

The panel detects your Mac's RAM at boot and gates features to fit.
Apple Silicon only — no Intel, no Linux, no Windows path. MLX is
Apple-only by design.

| RAM | Tier | What runs |
|---|---|---|
| < 48 GB | Compact | T2V / I2V at smaller dimensions (≤ 768 long-side) |
| 48–79 GB | Comfortable | Full 1280×704 at all modes — the canonical tier (M-Studio 64 GB) |
| 80–119 GB | Roomy | Longer clips, full Q8, FFLF unrestricted |
| ≥ 120 GB | Studio | No clamps |

LTX 2.3's working memory is real — there is no shortcut around it.
Standard 1280×704 generation peaks around 22 GiB resident; High mode
with the Q8 dev transformer (~19 GiB on disk) is closer to 38 GiB.
The tier system enforces this honestly instead of letting you queue
jobs that fall out of the OOM killer.

---

## Install

### Option A — Pinokio one-click (recommended)

1. Open Pinokio.
2. **Discover → Download from URL** → paste
   `https://github.com/mrbizarro/phosphene`
3. Click **Install**. Pinokio handles the rest:
   - Apple Silicon hardware gate
   - Clones [`dgrauet/ltx-2-mlx`](https://github.com/dgrauet/ltx-2-mlx),
     creates a Python 3.11 venv via `uv`, installs the MLX pipelines
     at the locked versions
   - Applies the codec + memory-overlap patches (idempotent, fails
     loud on upstream drift)
   - Downloads Q4 model (~25 GB) + Gemma encoder (~7.5 GB) via `hf
     download` — resumable
4. Click **Start** → **Open Panel** → http://127.0.0.1:8198

For the High quality tier (Q8 two-stage + TeaCache), download the Q8
model afterward via the **Download Q8** button in the panel sidebar
(one-time, ~25 GB extra).

### Faster downloads (recommended for Q8)

Hugging Face throttles unauthenticated downloads. Log in once and
downloads run **~10× faster**:

```bash
# Get a token (read-only is fine) at https://huggingface.co/settings/tokens
hf auth login
# Paste the token when prompted.
```

The `hf` binary inside Pinokio's install reads from the same standard
token file (`~/.cache/huggingface/token`), so future downloads
auto-authenticate. No env var fiddling.

### Option B — manual

```bash
# 1. Clone this panel
git clone https://github.com/mrbizarro/phosphene.git
cd phosphene

# 2. Clone ltx-2-mlx alongside (default panel layout assumes ./ltx-2-mlx/)
git clone https://github.com/dgrauet/ltx-2-mlx.git ltx-2-mlx
cd ltx-2-mlx
uv venv --python 3.11 --seed env
./env/bin/pip install ./packages/ltx-core-mlx ./packages/ltx-pipelines-mlx
./env/bin/pip install pillow numpy 'huggingface-hub>=1.0' \
  'mlx==0.31.1' 'mlx-lm==0.31.1' 'mlx-metal==0.31.1'
cd ..

# 3. Apply patches (codec + memory-overlap)
./ltx-2-mlx/env/bin/python3.11 patch_ltx_codec.py

# 4. Run the panel
./ltx-2-mlx/env/bin/python3.11 mlx_ltx_panel.py
# → http://127.0.0.1:8198
```

> **Why the version pins?** `mlx 0.31.2` introduced a numerical change
> that attenuates the LTX 2.3 vocoder output by ~22 dB (verified
> empirically: same prompt + seed + weights → -42 dB peak on 0.31.2 vs
> -9 dB peak on 0.31.1). Pinning to 0.31.1 is the recovery. See
> `CLAUDE.md` for the full version-pin rationale.

---

## Prompting for sound

LTX 2.3 conditions audio on prompt content. A visual-only prompt
produces near-silent ambient. A prompt with explicit audio cues
produces layered foreground sound.

| Prompt | Audio result |
|---|---|
| `"wizard in a forest"` | quiet room tone |
| `"wizard in a forest — low whispered chant, ember crackle, distant owl"` | audible chant + crackle + owl, all timed to the visuals |

**Pro tip:** describe the soundscape the same way you describe the
scene. Voice quality first ("clear, confident voice"), specific sound
events, then ambient. The Prompt Enhance button enforces this
structure automatically.

If you don't want music in the output (because music is hard to
remove cleanly in post), toggle the **🚫 No music** pill next to
Enhance — it appends an audio constraint to the prompt at submit time.

---

## Output format

Pick a preset from the **⚙ Settings** modal in the panel header. Three
built-ins, plus a Custom mode for advanced overrides:

| Preset | pix_fmt | CRF | ~Size (5s @ 1280×704) | Use case |
|---|---|---|---|---|
| **Standard** ⭐ default | yuv420p | 18 | ~7 MB | Visually lossless to most viewers. Plays everywhere — X, Instagram, Discord. The default for new installs. |
| **Archival** | yuv444p | 0 | ~50 MB | Mathematically lossless. Use when you'll re-encode in post or need a master. |
| **Web** | yuv420p | 23 | ~3 MB | Smallest files. For mobile, embedding, or quick previews. |
| **Custom** | choose | 0–30 | varies | 10-bit HDR, format-specific delivery, etc. |

Settings persist to `panel_settings.json` and apply to every new
render. The helper subprocess restarts automatically when you change
codec settings, so the change takes effect on the next render.

**`+faststart`** is always on, regardless of preset. The `moov` atom
sits at the front of the file so gallery thumbnails render the first
frame instantly without downloading the full clip.

For social uploads (X especially rejects `yuv444p`), re-encode with:

```bash
ffmpeg -i in.mp4 \
  -c:v h264_videotoolbox -profile:v high -pix_fmt yuv420p \
  -b:v 8M -maxrate 12M -bufsize 16M \
  -movflags +faststart \
  -c:a aac -b:a 192k \
  out.mp4
```

---

## Performance reference

Wall-clock times on an **M4 Mac Studio, 64 GB**:

| Mode | Resolution | Frames | Steps | ~Time |
|---|---|---|---|---|
| T2V Draft | 512×288 | 49 (2s) | 8 | ~35 s |
| T2V Standard | 1280×704 | 121 (5s) | 8 | ~7 min |
| I2V Standard | 1280×704 | 121 (5s) | 8 | ~7 min |
| Extend (Q4 dev, cfg=1.0) | 768×416 | +6 latents (~2s) | 12 | ~12 min |
| Extend (Q4 dev, cfg=3.0 "Quality") | 1280×704 | +6 latents | 12 | ~30 min |
| High (Q8 two-stage) | 1280×704 | 121 | s1=15 + s2=3 | ~12 min |
| FFLF (Q8) | 768×416 | 121 | s1=15 + s2=3 | ~5 min |

M-Max divides by ~3×. M-Ultra by ~6×. Compact tier (< 48 GB) takes
roughly 2× longer at clamped resolutions because of swap pressure.

---

## Configuration via env vars

Every path is overridable. Defaults are auto-detected.

| Env var | Default | What |
|---|---|---|
| `LTX_PORT` | `8198` | Panel HTTP port |
| `LTX_MODEL` | `dgrauet/ltx-2.3-mlx-q4` | Q4 model path or HF id |
| `LTX_MODEL_HQ` | `mlx_models/ltx-2.3-mlx-q8` | Q8 model path |
| `LTX_GEMMA` | `mlx-community/gemma-3-12b-it-4bit` | Gemma encoder path or HF id |
| `LTX_HELPER_PYTHON` | `ltx-2-mlx/env/bin/python3.11` | Python that runs the helper |
| `LTX_HELPER_IDLE_TIMEOUT` | `1800` | Helper auto-exits after this many seconds idle |
| `LTX_LOW_MEMORY` | `true` | Drop pipeline weights between jobs to free RAM |
| `LTX_OUTPUT_PIX_FMT` | `yuv444p` | Output codec pix_fmt |
| `LTX_OUTPUT_CRF` | `0` | H.264 CRF — 0 = lossless |
| `LTX_TIER_OVERRIDE` | _(unset)_ | Force a hardware tier (`base \| standard \| high \| pro`) — testing only |

---

## Credits

This is a wrapper, not a fork. All the hard model work belongs to:

- **[Lightricks](https://github.com/Lightricks/LTX-Video)** — original
  LTX 2.3 model, weights, and joint audio + video architecture
- **[@dgrauet](https://github.com/dgrauet/ltx-2-mlx)** — the MLX
  port (`ltx-2-mlx`). Without their work there is no LTX on Apple
  Silicon worth talking about. Q4 / Q8 quantized weights on Hugging
  Face. Phosphene wraps their package.
- **[Apple ML team](https://github.com/ml-explore/mlx)** — the MLX
  framework that makes Metal-native ML on Mac fast enough to run
  generative video locally
- **[mlx-community](https://huggingface.co/mlx-community)** — the
  Gemma 3 12B 4-bit text encoder distribution
- **[@cocktailpeanut](https://twitter.com/cocktailpeanut)** —
  Pinokio itself, which makes one-click installers like this
  possible

The panel adds: persistent batch queue, warm helper subprocess,
hardware-tier feature gating, lossless H.264 + faststart output,
output gallery with sidecar params, and the Pinokio install scripts.

---

## Known limits

- **Apple Silicon only.** No Intel, Linux, or Windows. MLX is
  Apple-only by design.
- **Memory pressure can SIGKILL the helper** on Macs at full RAM.
  Patches in `patch_ltx_codec.py` reduce peak by ~6 GiB during I2V
  denoise; closing Chrome, Slack, and iOS Simulator before a Standard
  render is the safest single thing a user can do. The panel surfaces
  the exit signal name (SIGKILL / SIGSEGV / SIGABRT) when the helper
  dies non-gracefully so issues are diagnosable.
- **Localhost only.** The panel binds to `127.0.0.1` with no auth.
  Not designed for LAN exposure or tunneling.
- **A2V (audio → video) not yet wired.** Upstream supports it; the
  panel UI does not expose it. v1.1.

---

## Roadmap

- [ ] A2V mode in the panel (upstream `a2vid_two_stage.py` exists)
- [ ] Pre-flight RAM advisory before submitting heavy jobs
- [ ] In-app HF token field (currently requires `hf auth login` in
      Terminal)
- [ ] Audio mode dropdown: With music / Voice + ambient / SFX-only / Silent
- [ ] Bisect `mlx 0.31.1 → 0.31.2` to identify and file the audio
      regression upstream

---

## License

**Panel:** MIT — see [LICENSE](LICENSE).

**LTX Video 2.3 weights:** Lightricks' own license. Read it before
commercial use.

**MLX framework:** Apache 2.0.

**Gemma 3 12B weights:** Google's terms.
