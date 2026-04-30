# Phosphene — Launch Channel Strategy

## 1. CivitAI — the actual play

**Verdict:** CivitAI is one of the highest-leverage placements. The panel itself is not the primary asset — the videos it produces are. CivitAI hosts: (a) Models/Checkpoints, (b) LoRAs, (c) Workflows, (d) Posts (image/video galleries), (e) Articles (long-form), and (f) a curated **Tools** section. Phosphene is conceptually a "Tool" but that section is curated rather than open-submission.

The audience signal is overwhelming: 12+ active LTX 2.3 workflows, multiple LTX-2.3 LoRAs (Pose Helper, Camera Zoom Out, Video Reasoning VBVR, IC Detailer), a dedicated `ltx-video` tag.

**Recommended play, in order:**

1. **Posts (videos) — primary.** Fastest, lowest-friction surface. Every post with a video and a copy-pasteable prompt becomes a SEO/discovery node. Tag aggressively: `ltx-video`, `LTX-2`, `LTX-2.3`, `apple-silicon`, `mlx`, `mac`, `audio-to-video`. In every description: one sentence "made with Phosphene (free Mac panel) — [repo link]" plus the full prompt.
2. **Article — secondary, high ROI.** Articles support headings, links, embedded images, durable SEO. One article: *"Phosphene: Running LTX-2.3 Natively on Apple Silicon (no Comfy, no CUDA)."*
3. **Comment on existing LTX-2.3 LoRA pages** — *helpful first*, mention Phosphene only when asked.
4. **Don't pitch the Tools section yet.** Wait for ~50+ posts and visible traction.

### CivitAI demo-video copy (5 drafts)

**Demo 1 — "Neon rain alley" (T2V + audio)**

> A neon-lit Tokyo back-alley at 3am, light rain pooling on the asphalt, a cat slinking along the wall, distant traffic hum and pachinko bells. Slow handheld dolly forward.
>
> Made with **Phosphene** — a free local panel for LTX-2.3 on Apple Silicon (M-series). Audio + video are generated jointly in one pass — that ambient rain and the pachinko bells are not added in post. Render time on M3 Ultra: ~3:20 for a 5s clip @ 1280×704. One-click install via Pinokio Discover.
>
> Repo: github.com/mrbizarro/LTX23MLX
> Settings: LTX-2.3 dev, 20 steps, CFG 3, 24fps, 5s, seed 42
> Tags: ltx-video, LTX-2.3, apple-silicon, mlx, mac, audio-to-video

**Demo 2 — "Ocean drone shot" (I2V + audio)**

> Image input: a still photo of a coastal cliff at golden hour.
> Prompt: Aerial drone slowly orbits left around the headland, waves crash against rocks below, gulls circle, gentle wind buffets the mic.
>
> I2V with synced audio — the wave timing and gull cries align with the visual motion. Generated locally on a MacBook Pro M4 Max in ~4:10. No cloud, no GPU rental, no upload of the source photo.
>
> Phosphene wraps dgrauet/ltx-2-mlx and adds: prompt history, batch queue, audio mute toggle, presets for portrait/landscape/square. Free, open source.
>
> Repo: github.com/mrbizarro/LTX23MLX
> Tags: ltx-video, LTX-2.3, i2v, apple-silicon, mlx

**Demo 3 — "Portrait — TikTok native"**

> A street musician plays harmonica under a subway arch, train rumbles pass overhead, busker's hat at his feet jingles when a coin drops. 1080×1920 portrait.
>
> LTX-2.3 was trained on portrait data — output goes straight to Reels/TikTok with no crop. Phosphene exposes the portrait preset on first launch.
>
> M2 Max, 64GB unified, 5s @ 1080×1920, ~5:00. Audio mix is mono-but-spatial; the coin-drop is in the right channel.
>
> Repo: github.com/mrbizarro/LTX23MLX
> Tags: ltx-video, LTX-2.3, portrait, vertical, tiktok, mlx

**Demo 4 — "LoRA test: Camera Zoom Out (I2V) + Phosphene"**

> Tested the [Camera Zoom Out LoRA by EKKIVOK](https://civitai.com/models/2490345) on Phosphene — works as expected. LoRA loading is in the LoRA picker (drop the .safetensors into ~/Phosphene/loras/, refresh).
>
> Image input: macro shot of a single dandelion seed.
> Prompt: A dandelion seed in tight macro, camera dollies smoothly outward to reveal an entire field at sunset, gentle wind in the grass.
>
> Cross-promo for the LoRA author. Settings: LoRA strength 0.85, base CFG 3, 20 steps.
>
> Repo: github.com/mrbizarro/LTX23MLX
> Tags: ltx-video, LTX-2.3, lora, i2v, mlx

**Demo 5 — "Stress test: 20s clip, audio sync"**

> Pushing LTX-2.3 to its 20-second limit, all-local on M3 Ultra. Prompt: A blacksmith hammering glowing steel in a candlelit workshop, sparks fly, hammer strikes ring out and decay into the wood-beamed ceiling, bellows wheeze rhythmically.
>
> 20s at 1280×704 took ~14:00 on M3 Ultra, 192GB unified. Memory ceiling: ~62GB peak. The hammer strikes sync to the visual impacts on every blow — no post.
>
> Phosphene exposes duration as a slider; previously you'd be hand-editing config in ltx-2-mlx.
>
> Repo: github.com/mrbizarro/LTX23MLX
> Tags: ltx-video, LTX-2.3, long-form, audio-sync, m3-ultra, mlx

### CivitAI article draft (~520 words)

> **Phosphene: Running LTX-2.3 Natively on Apple Silicon (no Comfy, no CUDA)**
>
> If you have a recent Mac and you've watched the LTX-2.3 wave from the sidelines because every workflow on this site assumes an RTX card — Phosphene is for you.
>
> **What it is.** Phosphene is a free, open-source desktop panel for LTX-Video 2.3 that runs on Apple Silicon via MLX (Apple's native ML framework). It generates **video and audio jointly** in a single pass — no separate audio pipeline, no after-the-fact mux. It's a thin, opinionated wrapper around `dgrauet/ltx-2-mlx`, the pure-MLX port of Lightricks LTX-2.
>
> **Why it exists.** ComfyUI on Mac works, but it's heavy: GGUF detours, MPS backend quirks, audio-as-second-pass. The MLX port from dgrauet is the right substrate — it talks to Metal directly and uses Apple's unified memory architecture, which is exactly the trick that makes a 64GB MacBook Pro outperform what its on-paper VRAM would suggest. Phosphene's job is just to make it one click instead of a Python invocation.
>
> **Install.** Phosphene is in the Pinokio Discover page. If you have Pinokio (free), you click install — that's it. First run downloads the LTX-2.3 weights from `dgrauet/ltx-2.3-mlx` on Hugging Face. No CLI, no venv juggling.
>
> **What it does well.**
> - **T2V and I2V** with audio in one pass, 24fps, up to 20s, up to 1280×704 (or 1080×1920 portrait — TikTok-native).
> - **Presets** for landscape / portrait / square so you don't fish for resolutions.
> - **LoRA loading** — drop `.safetensors` into `~/Phosphene/loras/` and pick from the menu. Yes, the LTX-2.3 LoRAs already on this site work (I tested Pose Helper, Camera Zoom Out, and the VBVR Reasoning LoRA).
> - **Prompt history + batch queue** so you can leave it cooking while you sleep.
>
> **What it doesn't do (yet).**
> - No upscaler tab — for now use a separate pass (RTX VSR not applicable; MLX upscalers are on the roadmap).
> - No keyframe interpolation UI yet.
> - No CivitAI direct download integration (planned — that's how I want to integrate with the LoRAs on this site).
>
> **Hardware floor.** M2 / M3 with 16GB unified can do 5s @ 1280×704 in ~6–10 min. M3 Ultra with 96GB+ handles 20s clips comfortably. The 8GB-VRAM rules from the NVIDIA workflows on this site don't translate — Apple's unified memory means everything in RAM is everything available to the model.
>
> **Try it, break it, tell me.**
> - Pinokio: search "Phosphene" in Discover.
> - GitHub: github.com/mrbizarro/LTX23MLX
> - Issues / feature requests welcome.
>
> Massive credit to **dgrauet** for the MLX port (the hard part), to **Lightricks** for open-weighting LTX-2.3, and to **cocktailpeanut** for accepting it into Pinokio Factory.

---

## 2. ProductHunt

**Verdict: Yes, launch — but it's a single-shot, plan it carefully.** Recent 2026 data shows local/open-source tools placing top-10. PH still rewards the "local alternative to a cloud-only thing" frame.

**Tagline (60 chars):** `Local AI video + audio on your Mac. Free. One click.` (54)

**Description (260 chars):** `Phosphene is a free Mac app that generates video AND audio together using LTX-2.3 — natively on Apple Silicon via MLX. No GPU, no cloud, no subscription. One-click install via Pinokio. Open source. Works offline.` (253)

**First comment (300 words):**

> Hey ProductHunt, I'm Salo (@AIBizarrothe). I built Phosphene because every "run AI video locally" guide assumes you own an RTX card — and the few that do work on Mac are buried in CLI flags, GGUF detours, and audio-as-an-afterthought.
>
> LTX-2.3 (Lightricks' open model) is special: it generates **video and audio jointly**, in one pass. The hammer hits on screen are the hammer hits in the audio. The footsteps sync. Most "AI video" you've seen is silent or post-dubbed — this isn't.
>
> The breakthrough that made a Mac app possible was **dgrauet's pure-MLX port** of LTX-2.3 (huge credit). MLX is Apple's ML framework that talks directly to Metal and uses unified memory, so your 64GB MacBook Pro outperforms what raw VRAM math would suggest. dgrauet did the hard part. Phosphene is the thin, opinionated UI on top: prompt box, image drop, presets, LoRA loader, batch queue. One-click install via Pinokio Discover.
>
> What you get:
> - T2V and I2V with audio, up to 20s, up to 1280×704 or 1080×1920 portrait
> - Native LoRA support (CivitAI LoRAs work)
> - Prompt history, batch queue
> - Free, open source, MIT
>
> What it isn't: an LTX Studio replacement. If you need 4K cinematic with director tools, Lightricks' own LTX Studio is the answer. Phosphene is for makers who want to iterate fast, locally, on a Mac, without paying per render.
>
> Roadmap: keyframe UI, MLX-native upscaler, CivitAI LoRA browser inside the app.
>
> Try it, break it, tell me what's wrong. Issues on GitHub: github.com/mrbizarro/LTX23MLX
>
> Thanks to dgrauet, Lightricks, the MLX team at Apple, and cocktailpeanut for getting it into Pinokio.

**Launch-day mechanics:** Tuesday or Wednesday 12:01am PT. Don't ask for upvotes (against rules) — line up 10–15 friends to *check it out* in the morning. Reply to every comment within 30 min for the first 6 hours.

---

## 3. Newsletters worth pitching

| Newsletter | Audience | URL | Why it fits | Tone |
|---|---|---|---|---|
| **The Neuron Daily** | 1M+ general/maker | theneurondaily.com | Already covered LTX 2.3 launch (verified). Mac angle = fresh follow-up they'd love. | Casual, hooky |
| **TLDR AI** | 500K+ devs/builders | tldr.tech/ai | Devs care about local-first, open weights, MLX | Brief, technical, no fluff |
| **Ben's Bites** | 120K+ AI builders/founders | bensbites.com | Builder narrative, "exited founder" tone | Casual, founder-voice |
| **Latent Space (swyx)** | 200K+ AI engineers | latent.space | swyx is Mac-pilled and MLX-curious | Technical, depth |
| **The Rundown AI** | 2M+ general | therundown.ai | Mass reach but generic-AI tone; pitch only the consumer angle | Hook-first, accessible |
| **AI Tidbits (Sahar Mor)** | 50K daily, technical | aitidbits.ai | Local AI / open-weights narrative is core | Technical, concise |

**Pitch premise (3 lines, riff per newsletter):**

> Lightricks open-sourced LTX-2.3 in January — the first model to generate **video and audio jointly**. Until now running it locally meant an RTX card. Phosphene is a free Mac app (Apple Silicon, MLX-native, one-click via Pinokio) that closes the gap — dgrauet did the MLX port; I built the panel. Repo + 30s demo: [link]. Happy to provide a Mac-specific angle, screen recording, or quote.

**Skip:** The Gradient (academic), Import AI (Jack Clark — policy).

---

## 4. YouTube + TikTok + IG Reels

### YouTube — yes, one walkthrough video

**Title candidates:**
- "I made a free Mac app for AI video — and it generates audio too"
- "Local AI video on a MacBook (no GPU, no cloud) — Phosphene + LTX-2.3"
- "This Mac app generates video AND audio together — runs offline, free"

**Thumbnail concept:** Split screen — left half is the Phosphene panel UI. Right half is a frame from the generated video. Big text overlay: `LOCAL · MAC · FREE`.

**Script outline (5–7 min):**
1. **0:00–0:20 Cold open** — "This 5-second clip with the hammer sounds and sparks? Generated in 3 minutes on a MacBook. Locally. Free. Both the video AND the audio in one shot."
2. **0:20–0:50 What LTX-2.3 is**
3. **0:50–1:30 Why Mac was hard / why dgrauet's MLX port changed it**
4. **1:30–2:30 Install demo** (real screen recording)
5. **2:30–4:30 Three demos** — T2V, I2V, portrait
6. **4:30–5:30 LoRA support** — drop a CivitAI LoRA in
7. **5:30–6:30 Where it falls short, what's next**
8. **6:30–end** — Repo link, credits

### TikTok / IG Reels — yes, 30s hook

**Format:** Screen recording on top half, your face/voice on bottom. Vertical 1080×1920.

**30s script:**
> [0–3s] "Free AI video on a MacBook? Watch."
> [3–10s] "Type a prompt." [show typing]
> [10–13s] "One click."
> [13–25s] "Both video AND audio, in one shot, generated locally."
> [25–30s] "Phosphene. Free. Apple Silicon. Link in bio."

---

## 5. Direct outreach to upstream + ecosystem

### dgrauet (MLX porter) — the #1 ask

**Handle:** GitHub `@dgrauet`. **X handle: not verified** — check their GitHub README and profile.

**Draft (GitHub issue or DM, ~110 words):**

> Hey — Salo here (@AIBizarrothe). I built Phosphene, a Mac app that wraps your `ltx-2-mlx` with a one-click Pinokio install, prompt history, LoRA loading, and a few presets. Repo: github.com/mrbizarro/LTX23MLX. The README credits you front-and-center — your port is the load-bearing piece, the panel is just the seatbelt.
>
> Two asks: (1) any factual corrections you want me to make in the README before launch — I want to represent what `ltx-2-mlx` does accurately. (2) Want to be tagged in the launch posts (X / ProductHunt / CivitAI article)? Happy to write you in however you prefer, or stay quiet if you'd rather not.
>
> Either way — thank you for the port. Truly.

### cocktailpeanut (Pinokio) — verified handle: **@cocktailpeanut** (X)

**Draft (~85 words):**

> Hey — Phosphene is now in Pinokio Discover (LTX-2.3 panel for Mac, MLX-native, generates video+audio jointly). Thank you for accepting it into Factory. Quick public launch is going up [date] — I credit Pinokio prominently. If you want to amplify, the demo videos are at [link]; if you'd rather not, totally fine, just wanted to give a heads-up before it goes loud.
>
> Repo: github.com/mrbizarro/LTX23MLX. M-series only for now, 16GB+ unified memory.

### Lightricks / LTX team

**Verified handles:** `@Lightricks` (corporate X), `@LTXStudio` (verify before use).

**Draft (~140 words, send via GitHub Discussions on Lightricks/LTX-2 OR partnerships email):**

> Hi — I'm Salo (@AIBizarrothe). I built **Phosphene**, a free, open-source Mac panel that wraps an MLX port of LTX-2.3 (dgrauet's port) so M-series Mac users can run T2V/I2V with joint audio in one click via Pinokio.
>
> Repo: github.com/mrbizarro/LTX23MLX. Demos: [link]. ~30s install, no CLI, no GGUF.
>
> The reason I'm reaching out: open-weighting LTX-2.3 made all of this possible. I'd love to (a) get a sanity check from your team that we're representing LTX-2.3 accurately in Phosphene's UI/copy, (b) explore whether a "made by community" cross-mention on LTX Studio's channels makes sense at launch, (c) hear how we could properly support the LTX-2 LoRA ecosystem from inside the app.
>
> Happy to jump on a call. Either way — thank you for open-weighting LTX-2.3.

### Awni Hannun (MLX) — verified handle: **@awnihannun** (X)

**Draft (~90 words):**

> @awnihannun — wanted to flag: Phosphene, a free Mac panel for LTX-2.3 (Lightricks' open-weighted audio+video model), is shipping on top of dgrauet's pure-MLX port. End-to-end MLX, generates 5s of synchronized video+audio on a MacBook in ~3min. One-click install via Pinokio.
>
> Repo: github.com/mrbizarro/LTX23MLX. 30s demo: [link].
>
> Wanted you to see what MLX is making possible at the creative-tool layer, not just LLM inference. Genuinely thank you for the framework — none of this exists otherwise.

---

## 6. Apple developer / WWDC angle

**Verdict: Mostly a dead end, with one narrow exception.**

- **Swift Student Challenge** — student-only.
- **Apple Design Awards** — App Store apps only.
- **Narrow exception: The MLX team itself.** Post a project showcase issue in `ml-explore/mlx` Discussions. **Effort:** Low. **Impact:** Low-medium.

**Recommendation:** Skip Apple's official channels except the MLX repo Discussions.

---

## 7. Lightricks formal partnership

**Verdict: Worth one polite outreach; don't over-invest.** Lightricks has ElevenLabs / Fal / Replicate / ComfyUI / OpenArt integrations. A community-built Mac wrapper is on-brand.

**Use the draft from Section 5.**

---

## Dead ends — channels that won't work

- **r/macgaming** — off-topic.
- **HackerNews Show HN** — only if Lightricks or Awni signal-boosts first (otherwise brutal on AI wrappers).
- **The Gradient / Import AI** — academic/research wrong audience.
- **Apple Design Awards / App Store featuring** — gated by App Store distribution.

---

## Prioritized punch list (solo maker, limited time)

In strict descending order of impact-per-hour:

1. **DM dgrauet on GitHub** — cannot launch without their blessing/credit alignment. ~15 min. Blocking dependency.
2. **CivitAI: 5 demo posts + 1 article** — one focused afternoon. Highest organic discoverability.
3. **30s vertical demo video (TikTok / Reels / X)** — ~3 hours. Force multiplier.
4. **DM @awnihannun and @cocktailpeanut on X** — 30 min total. Asymmetric upside.
5. **ProductHunt launch (Tuesday/Wednesday 12:01am PT)** — one-shot, plan a week out.
6. **Email Lightricks via partnerships + post in `Lightricks/LTX-2` GitHub Discussions** — ~45 min.
7. **Newsletter pitches: The Neuron Daily, TLDR AI, Ben's Bites, Latent Space** (in that order) — 4 emails, 15 min each.

Skip for v1: full-length YouTube walkthrough, Apple Design Awards / SSC, r/macapps.
