# Phosphene — Reddit + Community Launch Posts

## 1. r/StableDiffusion

**Suggested flair:** Tools (alternative: Resource)

**Title** (under 100 chars):
`Phosphene: free local LTX-2 panel for Apple Silicon — generates joint audio+video in one pass`

**Body:**

Quick share for the Mac folks here. Phosphene is a free panel that wraps the MLX port of Lightricks' LTX Video 2.3 so it runs natively on Apple Silicon, no PyTorch shim, no cloud.

The thing that surprised me most building this: LTX-2 generates **audio and video jointly in a single pass**. Not video-then-Suno, not silent-then-foley — the model emits a synced audio track alongside the frames. As far as I know, Wan 2.x, Hunyuan Video, Mochi, and CogVideoX are all silent. So if you're tired of pairing two models for ambient/diegetic sound, this might be interesting.

**What it does**
- T2V (text → video+audio)
- I2V (image → video+audio)
- FFLF (first frame + last frame interpolation)
- Extend (continue an existing clip)
- 5s clips, 24 fps, up to 1280×704 on the higher memory tiers

**Hardware reality**
M-series only. Tiered by unified memory:
- 64 GB: 1280×704 T2V/I2V works; FFLF capped at 768 px
- 32 GB: 768 px max
- 16 GB / 24 GB: 512 px, expect swapping
- ~7 min per 5s 1280×704 clip on a base M4 64 GB. Slower on M1/M2.

**Install**
One-click via Pinokio (submitted to Discover, listing pending). Or clone and run the panel directly — instructions in the README.

**Credits**
The actual MLX port is `dgrauet/ltx-2-mlx` — Phosphene is a panel/UX layer on top, all the hard inference work is dgrauet's. Model is Lightricks' LTX Video 2.3 (their license applies to the weights). MLX itself is Apple's framework. Phosphene the panel is MIT.

**Known limits / honest caveats**
- Apple Silicon only, no Intel Mac, no Windows, no Linux
- LoRA support not in yet (next milestone)
- No built-in upscaler yet (also next)
- First run downloads ~weights, takes a while
- 5s hard cap per generation; use Extend to chain
- Quality on faces is good after a codec patch (see comment below) but not Veo-tier — this is a 2B-class model

**Coming next**: LoRA loading, an integrated upscaler pass, longer chained Extends with audio continuity.

Repo: github.com/mrbizarro/phosphene (rename to Phosphene pending)

Feedback welcome, especially on the prompt formats that work best for the audio side — still mapping that out.

**First comment (you post yourself):**

For the curious on the quality side — early builds had visible JPEG-style chroma blocks on faces and skin. Tracked it to the muxer running yuv420p at default CRF. Patched the panel to write **yuv444p at CRF 0** (lossless) before any optional re-encode. Difference is noticeable on close-ups and dark scenes, basically free. If you've got an existing LTX MLX setup and faces look chunky, that's almost certainly where it's coming from.

---

## 2. r/LocalLLaMA

**Title:**
`Phosphene: native MLX panel for LTX-2 video on Apple Silicon (no torch shim, lossless mux)`

**Body:**

Local-models crowd — this is video rather than text, but the "runs on your Mac without phoning home" angle is the same. Sharing a panel I built around `dgrauet/ltx-2-mlx`, the MLX port of Lightricks LTX Video 2.3.

**Why MLX matters here**
The mainstream way to run LTX is PyTorch + MPS, which works but leaves performance on the table and chews unified memory. dgrauet did the real work — a proper MLX port, not a torch-MPS bridge. Phosphene is the panel/UX on top: model fetch, params, history, mux. The inference is native MLX, native Metal, no Python-side fp16/fp32 dance.

**Tiered memory gating**
Probably the most LocalLLaMA-relevant detail. Unified memory tiers map to capability gates the panel enforces up front:
- 16 / 24 GB: 512 px, single short clip, expect heavy compression/swap
- 32 GB: 768 px works comfortably
- 64 GB: 1280×704 T2V/I2V; FFLF still capped at 768 px because two-frame conditioning balloons activation memory
- 96 / 128 GB: same cap as 64 GB for now (model is the bottleneck, not RAM)

Better to refuse a job than OOM the whole box mid-generation.

**The codec patch**
Worth flagging because it's the kind of thing that matters once you actually look at output. Upstream defaulted to yuv420p in the muxer — for AI video that means JPEG-style chroma blocks on faces and gradients. Switched to **yuv444p, CRF 0 (lossless)** as the canonical output, with optional re-encode after. Visible improvement on close-ups, no perf hit.

**Numbers**
~7 min for a 5s 1280×704 clip on base M4 64 GB. ~12-15 min on M2 Pro 32 GB at 768 px. M1 series works but is slow.

**Install**
Pinokio one-click (Discover listing pending), or run the panel directly from the repo.

**Credits**: dgrauet (MLX port), Lightricks (model + weights license), Apple MLX team (framework). Phosphene panel is MIT; weights stay under Lightricks' license.

Repo: github.com/mrbizarro/phosphene

Roadmap: LoRA, upscaler pass, longer chained extends. Happy to answer questions about the MLX side.

---

## 3. r/aivideo

**Title:**
`Phosphene — free local AI video for Mac, and it actually generates the audio too`

**Body:**

Hey aivideo. Wanted to share something I've been working on: **Phosphene**, a free panel that runs Lightricks' LTX-2 model locally on Apple Silicon Macs.

The reason I'm excited about this one specifically: **it makes the audio at the same time as the video**, in one pass. Most local video models (Wan, Hunyuan, Mochi) are silent — you generate a clip, then go pair it with Suno or ElevenLabs or library sfx. LTX-2 just… emits a synced audio track with the frames. Footsteps land on footsteps. Wind matches wind. It's not perfect, but it's a different workflow.

**What you can make**
- Text → 5s video+audio at up to 1280×704
- Image → animated 5s clip (great for stills you've already generated in SD/Flux)
- First frame + last frame → interpolated motion between them
- Extend an existing clip to keep going

**Demo prompts to try**
- "rainstorm at night through a Tokyo alley, neon reflections in puddles, distant traffic"
- "close-up of a candle flame flickering in a quiet room, faint hum of a refrigerator"
- "ocean waves crashing on black volcanic rock, golden hour, gulls in the distance"

The audio on the rain/ocean/fire prompts is where it really shines — diegetic ambient sound is its strongest mode.

**Hardware**
Apple Silicon only. 64 GB unified gets you the full resolution. 32 GB does 768 px. Lower tiers work but smaller.

**Cost**
Free. Local. No subs, no credits, no queue.

Built on `dgrauet/ltx-2-mlx` (the MLX port) and Lightricks' LTX Video 2.3. Big credit to both — Phosphene is the panel layer.

Repo: github.com/mrbizarro/phosphene
Install via Pinokio one-click (Discover listing pending).

Drop your favorite prompts and clips, would love to see what people make with the audio side.

---

## 4. r/macapps

**Title:**
`Phosphene — generate AI video locally on your Mac, no cloud, no subscription`

**Body:**

Sharing a free Mac app I made for anyone who wants to generate AI video without paying a monthly subscription or sending anything to the cloud.

**One-sentence "what is this"**: Phosphene runs an open-source AI video model (LTX-2, by Lightricks) entirely on your Mac and gives you a clean panel to make 5-second clips from a text prompt or an image.

**Why Mac people might care**
- 100% local — nothing uploaded, nothing tracked, runs on your machine
- Free, MIT license — no trial, no credits, no "Pro" tier
- Apple Silicon native — uses Apple's MLX framework, takes proper advantage of unified memory and the Neural Engine path
- Generates audio with the video in the same pass (most AI video tools don't)

**Honest hardware requirements**
- Apple Silicon Mac only (M1 or newer — no Intel)
- 16 GB minimum, 32 GB recommended, 64 GB for full resolution
- Generation takes a few minutes per clip — this isn't realtime, it's "set it going and grab coffee"

**Install**
Easiest path: install Pinokio (a free one-click installer for AI tools) and grab Phosphene from there. Listing pending; meanwhile the GitHub README has direct steps.

**What you can do with it**
- Animate a still photo (image → video)
- Generate short clips from a prompt (text → video)
- Interpolate between two frames you provide
- Extend an existing clip

**What it isn't**
Not a full video editor. Not Veo / Sora quality — those are billion-dollar cloud systems. This is "good open-source video, on your Mac, free." For a lot of uses that's the right tradeoff.

Repo: github.com/mrbizarro/phosphene

Built on `dgrauet/ltx-2-mlx` (MLX port) and Lightricks LTX Video 2.3.

---

## 5. Hacker News (Show HN)

**Title:**
`Show HN: Phosphene – local AI video panel for Apple Silicon`

**Body:**

Phosphene is a panel that runs Lightricks' LTX Video 2.3 locally on Apple Silicon, wrapping the MLX port (`dgrauet/ltx-2-mlx`).

A few things that I think are actually novel rather than just another wrapper:

**Native MLX, not torch-MPS.** Most local video on Mac runs PyTorch with the MPS backend — workable, but it leaves Metal performance on the table and the memory model is awkward. The underlying port is real MLX. Phosphene is the panel on top: param UI, queue, history, mux.

**Joint audio+video in one pass.** LTX-2 emits a synced audio track with the frames, in the same generation. As far as I can tell every other open local video model — Wan, Hunyuan, Mochi, CogVideoX — is silent. Whether the audio is *good* is prompt-dependent; ambient/diegetic works best, dialogue doesn't.

**Tiered memory gating.** The panel checks unified memory at startup and refuses jobs it would OOM on. 16/24 GB → 512 px. 32 GB → 768 px. 64 GB → 1280×704 T2V/I2V (FFLF still capped at 768 px because two-frame conditioning balloons activations). I'd rather refuse than crash.

**Lossless mux.** Upstream defaulted to yuv420p, which produced JPEG-style chroma artifacts on faces. Patched to yuv444p CRF 0 as canonical, optional re-encode after. Free quality win.

**Numbers**: ~7 min for a 5s 1280×704 clip on a base M4 64 GB. M1/M2 work, slower.

**Deferred honestly**: LoRA support, integrated upscaler, longer chained extends — not in yet, on the roadmap. 5s hard cap per single generation. No Intel Mac, no Linux, no Windows.

**Credits**: dgrauet for the MLX port, Lightricks for the model (their license on the weights), Apple for MLX. Phosphene the panel is MIT.

Repo: https://github.com/mrbizarro/phosphene
Install via Pinokio (Discover listing pending) or run directly.

Happy to answer questions about the MLX path, memory tiering, or the audio side.

---

## 6. Other Community Channels

### Lightricks Discord / official LTX channels

> Hi Lightricks team — thanks for releasing LTX Video 2.3 openly. I built a free Mac panel called **Phosphene** that wraps `dgrauet/ltx-2-mlx` (the MLX port of 2.3) for Apple Silicon users. One-click Pinokio install, supports T2V/I2V/FFLF/Extend, generates the joint audio+video in one pass at up to 1280×704 on 64 GB Macs.
>
> Wanted to share it here in case it's useful for the Mac portion of the community, and to make sure credit reads correctly — the README points back to your model and license, and to dgrauet for the port. If anything in the wording or attribution looks off to you, please let me know and I'll fix it. Would also love any pointers on prompt structures the team has found work best for the audio side.
>
> Repo: github.com/mrbizarro/phosphene

### Pinokio Discord

> Hey Pinokio folks — submitted **Phosphene** to Discover (local LTX-2 video panel for Apple Silicon, joint audio+video). Listing pending. While I wait, would love any feedback from people running it via the install.json — particularly on first-run weight download UX and whether the memory-tier check fires correctly on your hardware. Thanks for building Pinokio, made the packaging side an order of magnitude easier.
>
> Repo: github.com/mrbizarro/phosphene

### MLX community on Hugging Face

> Sharing a Mac-native MLX project for the community: **Phosphene**, a panel around `dgrauet/ltx-2-mlx` (LTX Video 2.3 port). Notable bits from an MLX angle: joint audio+video in one pass, tiered unified-memory gating (16/32/64 GB capability tiers), yuv444p CRF 0 lossless mux to avoid chroma artifacts on faces. ~7 min per 5s 1280×704 clip on base M4 64 GB. All actual MLX inference work is dgrauet's; Phosphene is the UX layer. Would value technical feedback on the memory tiering and any obvious wins I'm leaving on the table.
>
> Repo: github.com/mrbizarro/phosphene

### Direct DM to dgrauet

> Hey — I'm Mr. Bizarro (AIBizarrothe on X). Built a free Mac panel called Phosphene that wraps your `ltx-2-mlx` port. Credit to you is prominent in the README and in every launch post — your port is the actual inference work, mine is just the panel/UX layer. About to do a small launch wave (HN, r/StableDiffusion, r/LocalLLaMA, Twitter, Pinokio Discover). Wanted to give you a heads-up since some of that traffic may land on your repo too. If anything in the attribution looks wrong or you'd like me to change wording / link target, please tell me and I'll fix it before the posts go up. Repo: github.com/mrbizarro/phosphene. Thanks for the port — it's what made this possible.

### r/macsetups (long shot)

> If you bought a high-RAM Apple Silicon Mac partly for AI work and haven't found much to actually run on it: **Phosphene** generates AI video locally with synced audio, free. 64 GB hits 1280×704, 32 GB does 768 px. Apple Silicon only. Built on Lightricks' LTX-2 model + dgrauet's MLX port. github.com/mrbizarro/phosphene

### Skip r/macgaming

Wrong audience. Gaming sub, not creator/AI sub. Don't post there.

### AI Newsletters — targets, not pitches yet

- **Ben's Bites** — covers tools/launches, Mac-local angle is on-brand. Strong target.
- **TLDR AI** — picks up Show HNs that get traction, so HN performance gates this one. Submit after HN if it lands.
- **Smol AI / Latent Space (swyx)** — local-models lean, MLX angle plays well, audience is technical. Good target.
- **The Rundown AI** — large but consumer-leaning; "free local AI video on Mac" frames cleanly. Possible.
- **AlphaSignal** — research-flavored, less likely unless the MLX/codec patches read as novel enough. Lower priority.
- **Last Week in AI / Import AI (Jack Clark)** — research weight; probably skip unless coverage is from a community-tools angle.

Pitch order if HN goes well: Ben's Bites + Smol AI first, then Rundown, then TLDR.

---

## 7. Cross-Channel Content Calendar (2 weeks)

Working in EST. Tuesday/Wednesday morning is the standard launch window for HN and big subs.

**Week 1 — Launch wave**

- **Day 1 (Tue) — 8:30am EST**: HN Show HN post goes up. Same morning, ~10am: r/StableDiffusion. These are the two big ones; ship them on the same day so anyone cross-checking sees activity in both.
- **Day 1 (Tue) — afternoon**: Twitter announcement thread (handled by the Twitter agent) referencing both posts.
- **Day 2 (Wed) — morning**: r/LocalLLaMA. Different audience, separate post, different angle. Don't double-post on Day 1 or it reads spammy.
- **Day 2 (Wed) — same day**: DM dgrauet (ideally before Day 1, but no later than Day 2 if launch caught fire). Lightricks Discord post.
- **Day 3 (Thu)**: Pinokio Discord post (#showcase). MLX community on HF.
- **Day 4 (Fri)**: r/aivideo (lighter tone, weekend-friendly creative sub).

**Week 2 — Slower-burn + iteration**

- **Day 7 (Mon)**: r/macapps. Different audience entirely, fresh start to the week.
- **Day 8 (Tue)**: Follow-up Twitter thread with user clips harvested from week 1.
- **Day 9 (Wed)**: r/macsetups (low-priority, only if bandwidth).
- **Day 10 (Thu)**: Newsletter pitches go out — Ben's Bites + Smol AI first, gated by Day 1 HN performance. If HN flopped, soften the pitch; if it landed, lead with the numbers.
- **Day 12 (Sat)**: "What people made with Phosphene this week" recap post on Twitter and as a comment thread bump on r/StableDiffusion / r/aivideo (only if there's real material to show — don't fake engagement).
- **Day 14 (Mon)**: Decision point — if LoRA support shipped during the wave, plan a v1.1 announcement post (r/StableDiffusion loves LoRA news). If not, hold for the actual milestone rather than cycle-empty.

**Hard rules**
- Never two big-sub posts the same day.
- DM dgrauet **before** anything else on Day 1 if possible.
- Don't post Friday afternoon EST or weekends for technical subs (HN, LocalLLaMA, SD) — they get buried.
- r/aivideo and r/macapps tolerate weekend posting fine.
