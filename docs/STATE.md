# Phosphene — project state, history, open work

Current version: **v2.0.4** on `main` (commit `74c7bd1`, May 5 2026).
Latest on `dev`: **Agentic Flows + multi-keyframe Layer 2** (May 6 2026, awaiting promotion).
Live URL: `https://github.com/mrbizarro/phosphene` · Linear project: `https://linear.app/hairstylemojo/project/phosphene-9c11240704bb`

This doc is the **session-start handoff**. A new Claude window entering this project should read this first, then `CLAUDE.md` (architecture), then the relevant Linear issues.

---

## 1. Where the code lives

Two clones on Salo's Mac, both managed by Pinokio:

| Path | Branch tracked | Port | Role |
|---|---|---|---|
| `/Users/salo/pinokio/api/phosphene-dev.git/` | `dev` | 8199 | Active development. Most edits land here first. |
| `/Users/salo/pinokio/api/phosphene.git/` | `main` | 8198 | Production / daily driver. Salo's actual usage. |

GitHub is the source of truth (memory: `feedback_github_source_of_truth.md`). Branch policy is strict: **NEVER push to `main` without Salo's explicit OK** (memory: `phosphene_dev_workflow.md`). When ready to promote, fast-forward only — `git merge --ff-only dev` from main, never a merge commit.

State directories that live OUTSIDE the repo via Pinokio's `fs.link`:

- `mlx_models/` → ~63 GB of LTX 2.3 weights (Q4, Q8, Gemma encoder, PiperSR upscaler). Shared between dev and prod via symlink chain.
- `mlx_outputs/` → all rendered mp4s + sidecar JSON files.
- `panel_uploads/` → user-uploaded reference images for I2V / FFLF.
- `state/` → `panel_settings.json`, `panel_queue.json`, `panel_hidden.json`. Survives a Pinokio Reset.

A Pinokio Reset wipes the install dir but preserves all four — Salo can Reset → Install without losing renders or settings.

## 2. Current capabilities (v2.0.4)

**Modes**
- T2V — text → video
- I2V — image → video
- FFLF (keyframe) — first/last frame interpolation
- Extend — append seconds onto an existing clip

**Quality dial**
- Quick · Balanced · Standard · High (Q8 two-stage HQ + TeaCache)

**Speed dial**
- Exact · Boost · Turbo (adaptive cached denoise)

**Sharp upscale**
- PiperSR on the Apple Neural Engine, optional install via `install_sharp.js`

**Joint audio + video**
- Synced lip movement, footsteps, ambient bed (mlx 0.31.1 pin holds the audio fix)

**Hardware tier system**
- Compact / Comfortable / Roomy / Studio with per-tier feature gating
- Reference benchmarks throughout this doc are on **M4 Max 64 GB** (Comfortable tier)

**Other**
- CivitAI LoRA browser built-in
- Spicy mode gate (NSFW LoRAs hidden by default, opt-in toggle in Settings)
- Per-job progress bar (phase-aware, denoise-step-aware)
- Gallery with cache-bust URLs, no more black-clip race
- 80+ GB less disk than pre-Y1.024 installs (filtered hf downloads)

**Agentic Flows (v2.0.5+, May 6–7 2026)**
- Engine kinds: `phosphene_local` (mlx-lm), `ollama`, `custom` (any
  OpenAI-compat), `anthropic` (Messages API, native preset)
- Two operating modes: `plan_sleep` (default — engine auto-stops after
  agent's `finish` call so RAM goes back to LTX renderer) /
  `interactive` (engine stays resident)
- Sessions sidebar (Cmd+K) with pinned/preview/rename/delete + auto
  search across titles
- "Queue them" batch bar above composer for explicit user-driven batch
- Multi-take per shot (`generate_shot_images append:true` adds Take
  N+1 below previous)
- Anchor pick / un-pick (re-click toggles), per-grid pick-state badge
- Project notes file (`state/agent_project_notes.md`) +
  `read_project_notes` / `append_project_notes` tools
- read_document tool (txt/md inline; PDF if pypdf installed)
- Image-engine plumbing: mock / mflux / bfl backends
- RAM headroom chip in agent header — green/amber/red based on free
  GB vs configured chat model size
- Memory-pressure guard: refuses to auto-spawn local engine when system
  is in swap or > 92% pressure
- Reasoning-model handling — `engine.chat()` reads `message.reasoning`
  separately from `message.content`; falls back when content is empty,
  raises informative error on length truncation
- Default `max_tokens` 8192 (was 3072 — too small for Qwen 3.6 / R1
  thinking budgets)
- Scroll-pinning + "↓ New messages" pill (no more auto-scroll yank)
- Stop button on long turns; abort via AbortController
- Offline banner (Phosphene-branded, pulsing) when /status fails
  twice in a row
- Phosphene-branded assistant avatar (favicon glyph, not "C")
- Live phasing on typing indicator: "Calling submit_shot · 12s",
  "Queued ce5c, planning next", etc.
- One-click "Stop engine" button (frees ~22 GB without going to
  Settings)
- Plan/Interactive mode pill toggle in agent header

**Frontend extraction (parked on `frontend-extraction` branch)**
- `webapp/` directory: index.html, style/all.css, js/main.js,
  vendor/marked.min.js + dompurify.min.js (MIT/Apache licenses)
- Panel slimmed from 16,223 → 5,866 lines (-10,357)
- New `/webapp/*` static route, `/api/page-config` endpoint
- Markdown rendering swapped to `marked.parse + DOMPurify.sanitize`
- Validated end-to-end on port 8210; merge to dev when reviewed

## 3. Marquee benchmarks (M4 Max 64 GB, sidecar-measured)

| Recipe | 5 sec | 10 sec | 20 sec |
|---|---|---|---|
| T2V Balanced + Turbo + 720p Sharp | 3:30 | 8:07 | 21:38 |
| T2V Quick + Turbo + 720p Sharp | — | — | 10:32 |
| T2V Standard 1280×704 Exact | 7:40 | — | — |
| T2V Standard Turbo | 5:26 | — | — |
| T2V High Q8 (max quality, no Sharp) | 11:51 | — | — |
| I2V Balanced + Turbo + Sharp | 3:37 | 8:26 | — |
| Extend +3 s on Q8 dev (768 px clamp) | — | 15:50 | — |
| FFLF (clamped 768×416, Comfortable tier) | — | 5:29 | — |

Per-step cost scales **~T^1.5** with frame count (218 s/step at 481f vs ~30 s/step at 121f, same width). Sub-quadratic — confirms LTX uses windowed/factorized attention. **20-second single clips are production-viable**; 30 sec at 1024×576 is plausible, 60 sec needs lower res or research breakthrough.

## 4. Version history (compressed)

Pre-2.0 was the `Y1.NNN` sequential counter. v2.0.0 cut over to semver on May 3 2026.

**Y1.001 → Y1.013** (Apr 28–30) — First usable T2V/I2V renders. Audio SHIP-BLOCKER fixed by pinning `mlx==0.31.1` (0.31.2 attenuated vocoder by 22 dB).

**Y1.014 → Y1.024** — Hardware-tier system, Boost/Turbo speed modes (adaptive denoise caching), CivitAI LoRA browser, Q8 two-stage HQ tier, FFLF + Extend modes, `hf_transfer` downloads, Q4/Q8 download filter (saved ~80 GB on existing installs via `update.js` trim).

**Y1.025 → Y1.035** (Codex-led arc) — Sharp upscale via PiperSR (Apple Neural Engine), I2V tail-stall fix (`Y1.034` free DiT before VAE decode), VAE temporal-streaming for long clips (`Y1.035`), license / install hardening, Spicy mode gate prep.

**Y1.036 → Y1.039** — Fixed `Y1.024` Extend regression (route to Q8 dev transformer), VAE auto-streaming threshold (recovered ~7 % on short clips), Now-card progress bar rewrite (phase + denoise-step aware), gallery black-frame race fix.

**v2.0.0** (May 3) — Marquee release. 2.0 badge in panel header, semver versioning starts.
**v2.0.1** — Spicy mode toggle gates NSFW LoRA visibility.
**v2.0.2** — Install fails loud when pipeline packages are missing (sanity-import step in `install.js`).
**v2.0.3** — Install log self-documents Python toolchain (uv version, system python presence, post-pip site-packages list).
**v2.0.4** (May 5) — Strip em-dash from install.js sanity check. Was breaking install on some Pinokio shells (KTDS + second user hit identical SyntaxError). Pure ASCII now.
**v2.0.5** (May 6) — Drop the `print('venv OK: ...')` decoration from the sanity-import step. KTDS reproduced the SyntaxError on v2.0.4 — turns out something in their environment (Pinokio's command preprocessor or a user-side rewriter) was cutting the literal `OK:` out of the Python string AND appending `OK` after the closing shell quote, so Python received `...importable')OK` and bailed. Removing the print sidesteps the rewriter entirely. The exit code from a successful `import` is the only success signal `shell.run` needs anyway.

Full git log on `main`. Tags `v2.0.0` through `v2.0.4` published. v2.0.5 awaiting promotion on dev.

## 5. The folder layout

```
phosphene-dev.git/
├── pinokio.js / pinokio.json          ← Pinokio menu logic + manifest
├── install.js / update.js             ← idempotent install / update flows
├── install_sharp.js                   ← optional PiperSR Sharp installer
├── download_q8.js                     ← optional Q8 weights download
├── download_upscaler.js               ← optional spatial upscaler download
├── start.js                           ← Pinokio start script (launches the panel)
├── reset.js                           ← Pinokio reset script
├── recover.sh                         ← rare-case manual recovery
│
├── mlx_ltx_panel.py                   ← the panel HTTP server (~9000 lines, single file)
│   ├── /status, /queue/*, /run, /upload, /file, /civitai/*, /loras, /settings ...
│   ├── HTML+CSS+JS for the UI all inlined as page() string
│   └── Worker thread + helper subprocess management
│
├── mlx_warm_helper.py                 ← persistent helper subprocess (~1300 lines)
│   ├── Loads + holds T2V/I2V/Extend/HQ/Keyframe pipelines from ltx_pipelines_mlx
│   ├── Reads job specs from stdin, emits events to stdout
│   └── action types: generate / generate_keyframe / extend
│
├── patch_ltx_codec.py                 ← idempotent runtime patches against installed
│   ├── Patch 1: codec → yuv444p crf 0 + faststart (lossless H.264)
│   ├── Patch 2: I2V free DiT before decode (matches T2V cleanup)
│   ├── Patch 3: free vae_encoder pre-denoise (peanut review)
│   ├── Patch 4: free feature_extractor in base load() (peanut review)
│   └── Patch 5: VAE temporal streaming decode (long clips no longer freeze)
│
├── required_files.json                ← single source of truth for "installed"
├── VERSION                            ← read by panel + version-check loop
├── .env.local                         ← LINEAR_API_KEY (gitignored, chmod 600)
│
├── README.md                          ← user-facing docs (homepage on GitHub)
├── CLAUDE.md / AGENTS.md / GEMINI.md / QWEN.md
│   ← agent manuals (architecture, conventions, history)
├── docs/                              ← long-form internal docs
│   ├── STATE.md                       ← this file
│   └── SDK_KEYFRAME_INTERPOLATION.md  ← multi-keyframe interpolation design + plan
├── launch/                            ← marketing copy (Pinokio article, X thread, Reddit, etc.)
│
├── ltx-2-mlx/                         ← upstream MLX port (separate clone of dgrauet/ltx-2-mlx)
│   └── env/                           ← Python 3.11 venv (uv-managed)
├── mlx_models/                        ← weights (~63 GB, fs.link symlink)
├── mlx_outputs/                       ← rendered mp4s + sidecars (fs.link symlink)
├── panel_uploads/                     ← user reference images (fs.link symlink)
├── state/                             ← panel_settings/queue/hidden.json (fs.link symlink)
├── cache/                             ← HF_HOME for downloads
└── logs/                              ← Pinokio's own command-execution logs
```

`mlx_ltx_panel.py` is the heart of it — almost all panel behavior lives there. `mlx_warm_helper.py` is the long-running inference subprocess. `patch_ltx_codec.py` is a 500-line runtime modifier that fixes upstream code without forking it.

## 6. What worked / didn't this session (May 3–5 2026)

### Cinematic capability findings (from rendering ~30 clips)

**The model's wheelhouse**
- Human cinematic moments. Faces at medium and tighter, body language, atmospheric scenes.
- Static or near-static camera works better than moving camera.
- 2–3 dialogue turns per clip work cleanly when prompt follows LTX's docs literally:
    - Single continuous paragraph (NOT uppercase character cards)
    - Voice descriptor on every speech beat (not just first)
    - Single quotes around dialogue
    - Action density ~1 explicit beat per 2–3 sec of clip
- Joint audio + video really IS jointly diffused — lip-sync is uncannily tight.

**The model's weaknesses (avoid in prompts)**
- **Hands and held objects** — fingers morph, written text squiggles, pen/needle/cup interactions look off.
- **High-motion physics** — skater kickflips, water splash, motorcycle blur are out of distribution.
- **Faces below ~80 px in-frame size** — model fills a face-shape but identity-broken. Wide shots of single characters are unusable in their first/last seconds. ([Salo's discovery May 4](#))
- **Multi-shot continuity is naive-failure** — same prompt + different seed = different person. The mom-kid scene experiment (M1 / M2 / M3 in `mlx_outputs/`) confirmed three different women across three angles despite identical character description.

### What earns 20 seconds
- 6–9 explicit beats described in the prompt. Anything less and the model fills with stasis.
- Static or near-static camera. Camera motion costs visual coherence.
- Specific named actions ("she turns slowly", "she breathes out", "the streetlight flickers off") give anchor points.

### Empirical experiment outcomes

- **M1/M2/M3 mom-kid trio** (1024×576, Balanced + Turbo + Sharp, ~21 min each): demonstrated multi-shot character drift problem. Three different women across three angles.
- **N1–N10 cinematographic moments** (May 4): ten 20-sec clips at varying shot scales. Tested medium / wide / two-shot composition with body-language-only prompts (no hands, no held objects). Output quality varied; faces are stable when in the safe pixel range.
- **E-DRAFT** (May 4): tested low-res draft → high-res commit hypothesis. Same prompt + seed at 640×480 vs 1024×576. Salo: low-res output not usable due to face-distance issue. Premise was flawed because lower res = worse faces.
- **E-ANCHOR** (May 4): I2V from M1 frame to test character anchoring. Result was inconclusive in the session; final clip is at `mlx_outputs/` if needed for review.
- **20-sec single-clip viability** (May 4): confirmed at Balanced 1024×576 + Turbo + Sharp. ~21 min wall, audio synced, characters stable.

## 7. Known bugs / fixed bugs

### Fixed this week
- **Y1.034 silent regression**: VAE temporal-streaming patch tiled even on short clips, adding ~30 s decode tax. Fixed in Y1.037 with auto-mode threshold.
- **Y1.024 Extend regression**: my own download-filter pruned `transformer-dev.safetensors` from Q4 dir, breaking Extend mode for Q4-only installs. Fixed in Y1.036 by routing Extend to Q8.
- **I2V tail stall**: 60+ s frozen final-step on long I2V renders. Fixed in Y1.034 (free DiT before VAE decode).
- **Now-card progress bar mis-paced**: Quick crawled, High blasted past 99%. Fixed in Y1.039 with phase-aware + denoise-step-aware computation.
- **New clips appeared black for 2–3 minutes**: gallery race + browser cache holding incomplete bytes. Fixed in Y1.039 with in-flight skip + mtime cache-bust + `Cache-Control: no-cache`.
- **S2 noir dialogue attribution swap**: "Same thing, honey" delivered by wrong character. Root cause: prompt format diverged from LTX docs. Documented in Linear HAI-152.
- **v2.0.2 install sanity check broken by em-dash**: Pinokio shells mangled the unicode em-dash, triggering Python SyntaxError, falsely failing every install. Fixed in v2.0.4 (ASCII colon).

### Fixed in v2.0.5 (May 6–7 2026, agentic-flows polish session)
- **Stage NOW RENDERING stuck at 0%**: `current.progress` schema went from flat float to structured object (`{phase, phase_label, pct, elapsed_sec, eta_sec, denoise_step}`). Stage pane was calling `Number(progressObj)` → NaN → 0%. Fixed by reading `.pct` directly with backward-compat fallback to flat float.
- **Offline banner was a 1990s solid-red bar**: floating Phosphene-pill at top-center, pulsing logo icon, three-part text, slide-in animation.
- **Typing indicator stuck on "Drafting plan"**: only re-evaluated when message count grew. Now refreshes every 1.5s with elapsed seconds appended after 6s ("Thinking · 12s") and contextual phrases like "Calling submit_shot", "Queued ce5c, planning next".
- **Auto-scroll fought the user**: every async render did `scrollTop = scrollHeight`, yanking users back when scrolled up. Now uses scroll-pinning — auto-scroll only when within 80px of bottom; "↓ New messages" pill appears otherwise.
- **No abort on long turns**: × button on typing row aborts in-flight fetch via AbortController. Server-side run_turn keeps stepping until it finishes (no kill switch in runtime today); UI is unstuck immediately.
- **Tool cards visually outweighed prose**: 3px accent borders + bold accent-bright + monospace 12px overpowered the assistant's planning text. Now muted 2px, smaller, near-transparent — reads as side metadata.
- **Anchor selection only changeable, never removable**: re-clicking the same anchor un-picks it. Per-grid pick-state badge ("✓ picked" / "click to pick") visible at all times. Backend accepts empty `png_path` as remove signal.
- **No batch trigger**: "Queue them" pill above composer when picks > submitted_shots. One-click injects a structured message authorizing the agent to submit each pick as i2v.
- **Multiple takes lost previous**: same-label calls overwrote candidates. New `append:true` arg writes under `take_NN/` subdir + appends to `takes` array. Chat renders all takes stacked with "Take N / M" labels.
- **Memory crashes from OOM**: macOS suspended Claude.app at 83 GB while LTX rendered + Qwen 35B occupied 22 GB. Memory guard now refuses to auto-spawn engine at >92% pressure or >8 GB swap. Plan-and-sleep mode auto-stops engine after `finish` so RAM goes back to renderer.
- **Reasoning models returned empty content** (CRITICAL — found May 7 1am during overnight session): Qwen 3.6 Abliterated splits output into `message.reasoning` + `message.content`. With `max_tokens=3072`, reasoning consumed budget and content came back empty with `finish_reason="length"` → runtime saw empty assistant message → ended turn. Agent appeared "stuck". Fixed: bumped default to 8192, engine.chat() now reads reasoning, raises actionable error on length truncation, falls back to reasoning when content is empty.
- **Agent identity drift**: avatar was orange "C" reading as generic chat bot. Now circular Phosphene logo (favicon glyph), label "Phosphene". Identity is the brand, not the underlying model.

### Open / partially understood
- **KTDS install case** (Linear HAI-156): user reported `ModuleNotFoundError: No module named 'ltx_pipelines_mlx'` after a "green" install. Symptom is now possibly explained by the v2.0.2/v2.0.3 em-dash bug (the sanity check itself was broken, install completed green for the wrong reason). Asked for log tail + VERSION; pending response. Suggested fix: Reset → Install on v2.0.4.

## 8. Open work / future direction

Everything below is also tracked in Linear (HAI-150 → HAI-158 under the Phosphene project). This section duplicates the most current state for fast scan.

### Multi-keyframe interpolation as SDK shot-composition primitive

**See:** `docs/SDK_KEYFRAME_INTERPOLATION.md` (full design + research review).

**TL;DR**: ComfyGuy9000 demoed first-frame-last-frame method via `Deno2026/comfyui-deno-custom-nodes`. Phosphene's `ltx_pipelines_mlx.KeyframeInterpolationPipeline` already accepts arbitrary `list[Image]` keyframes + `list[int]` indices — but our panel/helper artificially restrict it to 2 keyframes (start + end). Exposing the full multi-keyframe API gives us the agentic-flow compositional primitive: agent picks N stills, model fills the motion, character is anchored at every shot start.

**Status (2026-05-06)**:
- **Layer 1 — DONE.** Helper `generate_keyframe` action accepts arbitrary `keyframe_images` + `keyframe_indices` lists, with strict validation. Backward-compatible with the old `start_image`/`end_image` shape so the panel keeps working.
- **Layer 2 — DONE (commit 1afa1be).** `mlx_ltx_panel.py:make_job` reads a `keyframes_json` form field (JSON-encoded list of `{image_path, frame_index}` plus a `keyframes_total_frames` companion). The keyframe branch in `run_job_inner` decodes, validates strictly-increasing indices within `[0, frames-1]`, and forwards `keyframe_images` + `keyframe_indices` arrays to the helper. Backward compat preserved: empty `keyframes_json` falls back to `start_image`/`end_image`.
- **Layer 3 (panel UI multi-row keyframe list) — NOT YET.** The manual UI still has 2 drop-zones. Agents already use the full primitive via `submit_shot(keyframes=[{image_path, frame_index}, ...])`.

**Today's agent path**: through the panel — `agent.tools.submit_shot` composes the form including `keyframes_json` and POSTs to `/queue/add`. The legacy stdin-direct path still works for non-panel callers.

### Long-video research (Strategy A / B / C)

Goal: 1-minute final video on M4 Max 64 GB, ~40-60 min wall time acceptable.

- Strategy A — push single LTX clip beyond 10 sec. 20-sec proven at 1024×576 + Turbo + Sharp. 30-sec untested; 60-sec needs research.
- Strategy B — Extend chaining. ~16 min per +3 s pass, ~4.5 h total for 1-min. Audio continuous.
- Strategy C — multi-scene assembly via LLM-driven shot-list planner. ~42-49 min total, hides cuts cinematically. **This is what the multi-keyframe SDK enables.**

Codex deep-research brief drafted; awaiting return for literature review on FreeNoise / FIFO-Diffusion / StreamingT2V applicability.

Salo also has Claude.ai / ChatGPT deep-research running (May 5) on inference speed without quality loss.

### Director Mode (agent workflow) — SHIPPED as Agentic Flows

What ships: a chat-driven shot planner tab in the panel. User pastes a script or idea, agent breaks it into shots, queues every shot through the existing FIFO queue, writes a `manifest.json`, and finishes. Designed for overnight batch rendering. Auto-stitch is intentionally NOT included — manifest is the deliverable; cuts belong to the user.

See preceding "Agentic Flows" section + `docs/AGENTIC_FLOWS.md` for the full reference.

Long-video research (per-shot length sweet spot, FreeNoise / FIFO-Diffusion / StreamingT2V applicability) still pending Codex deep-research return.

### Speed optimization candidates (from May 4 research session)

Ordered by what to try first:

1. **Two-stage workflow: draft + commit** — render 5-sec at full res first, then 20-sec same seed if approved. ~6× faster iteration. Replaces the failed "low-res draft" idea (faces don't survive res drop).
2. **Skip Sharp on batch testing** — ~26-100 s saved per clip during iteration.
3. **Pre-warm helper on panel boot** — saves ~30 s on first job of a session.
4. **Resume cancelled jobs from latent checkpoint** — recovers ~10 min per cancellation in iterative work. Higher engineering cost.
5. **Character anchoring via I2V keyframe** — quality unlock, not speed (but enables SDK).
6. **Two parallel helpers on 64 GB** — 2× throughput on batch renders. Refactor risk.

### Optimization paths ruled out (May 5 lab — see PERF_RESEARCH_2026-05-05.md)

Full research log: `docs/PERF_RESEARCH_2026-05-05.md`. Tested + ruled out:
mlx-mfa SDPA, `mx.compile`, RoPE caching, sliding-window attention, 8→6/4 step
reduction (catastrophic on the distilled model), block-skip caching (DeepCache
for DiT — works at tiny scale, fails at production: SSIM 0.69-0.72, "different
identity"). Most useful finding: **conv3d kernel port is NOT a real M4 path
forward** — MLX already uses steel implicit-GEMM at 50-70% of M4 peak; the
Draw Things "2.4×" was vs MPSGraph (which MLX doesn't use). Saves 1-2 weeks.

Block-skip patch infrastructure (with full A/B strips and per-config numbers)
parked on the `experiment/block-skip` branch — reusable if Lightricks ships a
block-skip-aware fine-tune.

Honest verdict: M4 Max + MLX 0.31 + LTX-2.3 Q4 distilled is already running at
50-70% of theoretical peak. Real breakthroughs need M5 hardware (Neural
Accelerators, ~3× free), NVFP4 quantization (when MLX supports it), or
research-grade work on token merging.

### Marketing / launch (HAI-157, HAI-158)

- Tweet thread + slides drafted in scrollback (5-6 tweets, copy-paste ready).
- Personal-account post drafted for `@AIBizarrothe`.
- Launch copy bundle in `launch/` folder (Pinokio article, X, Reddit, CivitAI).
- Sample mp4s + frames cached in `/tmp/phos_frames/`, `/tmp/phos_frames2/`, `/tmp/phos_dialogue/`, `/tmp/phos_lab_frames/`, `/tmp/phos_sdk_frames/`.
- Awaiting Salo's launch timing call.

## 9. Hard constraints (don't violate)

- **Apple Silicon (M1+) only**. No PyTorch, no CUDA, no MPS shim. Native MLX or it doesn't ship.
- **Joint audio + video must remain**. That's the differentiator vs Wan / Hunyuan / Mochi. We don't drop audio for length.
- **Existing queue + helper + patch architecture stays intact**. No new microservices.
- **Branch policy**: dev → main only via fast-forward, only with Salo's explicit OK.
- **Salo's voice in writing**: copy-edit, don't rewrite. See memory file `feedback_copy_edit_dont_rewrite.md`. Tweets, posts, README copy — fix typos and grammar, never restructure or stack value-prop language.

## 10. Memory pointers (for next-Claude)

Files in `/Users/salo/.claude/projects/-Users-salo-AI/memory/`:

- `phosphene_dev_workflow.md` — branch + dev-panel discipline
- `phosphene_linear_project.md` — Linear project location + credentials
- `feedback_copy_edit_dont_rewrite.md` — when Salo asks me to look at his writing
- `feedback_writing_style.md` — schematic, short, plain
- `feedback_github_source_of_truth.md` — git fetch first, surface drift
- `feedback_dont_ask_to_save_memory.md` — when Salo states a fact, save it; don't ask
- `claudio_repo.md` — shared infra, hosts `.env` with Linear key
- `ltx_video_setup.md` — older notes from MLX vs Comfy decisions

## 11. Linear board

`https://linear.app/hairstylemojo/project/phosphene-9c11240704bb` — Phosphene project under HAI team (free plan caps at 2 teams).

Issue prefixes are `HAI-NN` because of the team constraint. Active:

- HAI-150 History (Done — reference doc)
- HAI-151 Current state (Done — reference doc)
- HAI-152 Lab batch 1 (In Progress — folded into this STATE.md going forward)
- HAI-153 Lab batch 2 (Backlog — depends on what comes next)
- HAI-154 Long-video research Strategy A/B/C (Backlog)
- HAI-155 Director Mode agent workflow → SHIPPED as Agentic Flows (2026-05-06, dev branch)
- HAI-156 KTDS install case (In Progress — pending log tail)
- HAI-157 Tweet thread + writeup launch (Backlog — drafts ready)
- HAI-158 Marketing scenes (In Progress)

## 12. How to start a fresh session

1. `cd /Users/salo/pinokio/api/phosphene-dev.git/`
2. `git fetch origin && git status -sb` — surface any drift first
3. Read this file (`docs/STATE.md`) and `CLAUDE.md`
4. Skim Linear `HAI-150` through `HAI-158` for state of each workstream
5. Check the dev panel is alive: `curl -s http://127.0.0.1:8199/status | python3 -m json.tool | head -10`
6. Last 5 commits on dev: `git log --oneline -5 dev`

If you find on-disk state contradicts this doc (paths moved, commits diverged), surface that to Salo before working around it. Updating this doc at session-end is part of the loop.
