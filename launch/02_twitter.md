# Phosphene — X Launch Package

## 1. Hero Tweet

**Variant C — announcement (CHOSEN — use this)**

> Happy to share something I've been building — Phosphene.
>
> A free local video+audio generator for Apple Silicon, running LTX 2.3 in MLX. Built with a lot of help from Claude. Mostly works. There are bugs. PRs welcome.
>
> One-click install via Pinokio. 🧵
>
> [video]

Tone: builder-honest announcement. Not hype, not humble-brag — straight "I made a thing, here it is, it's flawed, help me make it better." The Claude credit upfront is the differentiator: nobody else is being honest about how their AI tool was built. Lead with that.

**Variant A — technical wow (alt)**

> Local generative video on a Mac just got weird.
>
> Phosphene: a free panel that runs LTX 2.3 natively in MLX. 5-second 1280×704 clip in ~7 min on an M4 Mac Studio. With sound. Generated jointly, in one pass.
>
> No cloud. No queue. One-click install.
>
> [video]

**Variant B — creator vibe (alt)**

> made this on my Mac. no cloud, no API key, no rendering farm.
>
> the audio? generated in the same pass as the video. that's LTX 2.3.
>
> built a free panel for it called Phosphene. link below.
>
> [video]

## 2. Thread (reply chain to hero — paired with Variant C)

**1/** What it does:
- T2V — text → video+audio
- I2V — image → video+audio
- FFLF — first + last frame, model fills the middle
- Extend — add seconds to an existing clip

LTX 2.3 generates audio jointly with video. Most local video models are silent. This isn't.

**2/** Real numbers on an M4 Mac Studio 64 GB:
- 5s @ 1280×704 — ~7 min
- High quality (Q8) — ~12 min
- FFLF @ 768 — ~5 min

Not a render farm. But: no cloud, no API key, no queue, no monthly bill.

**3/** Hardware tiers auto-gate by RAM:
- 32 GB → Compact tier (lower res, shorter clips)
- 64 GB → comfortable floor for full 1280×704
- 96 / 128 GB → longer clips and Q8 without pain

Generative video is memory-hungry. Not negotiable.

**4/** Honest disclaimer:
Claude (Anthropic) wrote a lot of this with me. I designed, directed, reviewed, tested, broke, and fixed. There are bugs. Some edges are rough. The repo is open — file issues, send PRs.

github.com/mrbizarro/phosphene

**5/** Install is one click via Pinokio. No conda, no venv, no CUDA cosplay on a Mac. Search "Phosphene" in the Pinokio Discover tab.

**6/** Credit where it's due:
- LTX 2.3 model: @Lightricks
- MLX port: @dgrauet
- MLX framework: Apple ML team
- Pinokio one-click runtime: @cocktailpeanut

Phosphene is the panel tying them together. None of it exists without their work.

**7/** Free, MIT licensed (panel — model has Lightricks' license, read it before you build a business on it).

If you make something weird with it, tag me.
If something's broken, file an issue.
If you want to fix it, PRs welcome.

Follow @AIBizarrothe for updates.

**8/** Roadmap: LoRAs (style, camera motion, HDR), CivitAI model browser, prompt enhance via Gemma is already shipped. Local generative video is going to get strange in a good way over the next year. Glad to be early.

**9/** If you read this far — thank you.

Local AI tools live or die by word of mouth. A like, a repost, a quote with your take, a comment with what broke — all of it helps this reach the people who'd actually use it.

RT the hero if you think a Mac dev should see it. 🙏

## 3. Reply Variants

**On "M1/M2 base 8 GB?"**
> No. 32 GB is the floor (Compact tier — lower res, shorter clips). 64 GB is where it's actually pleasant. Generative video models hold the whole denoising state in RAM. There's no shortcut.

**On "vs. Wan / Hunyuan / Mochi?"**
> Two real differences: LTX 2.3 generates audio jointly with video — those models are silent. And it runs natively in MLX, not a torch shim. On Apple Silicon that's a meaningful speed/memory delta, not a footnote.

**On "is the model open?"**
> Lightricks released the LTX 2.3 weights publicly under their own license. Phosphene (the panel) is MIT. Read Lightricks' license for the model itself before you build a business on it — I'm not the right person to summarize their terms.

**On "commercial use?"**
> Read the Lightricks license. Not legal advice from me. The panel is MIT so the wrapper is yours; the model output terms come from Lightricks.

**On "longer clips?"**
> Use Extend mode. Feed it the last frame of your previous clip and it continues. ~11 min per additional 5 seconds on 64 GB. Stitch a few of those and you have a real shot.

**On "quality tier differences?"**
> Standard ≈ 7 min / 5s clip @ 1280×704, the daily driver. High (Q8) ≈ 12 min, cleaner detail, fewer artifacts on faces and text. FFLF @ 768 ≈ 5 min, useful for interpolation work where you already have endpoints.

## 4. Quote-tweet / Boost Variants

**For a Lightricks post about LTX 2.3:**
> If anyone wants to run this locally on Apple Silicon, there's a free panel — Phosphene — that wraps the MLX port. One-click via Pinokio. Joint audio+video on an M-series Mac is genuinely surreal the first time.

**For a dgrauet post about the MLX port:**
> Building on top of this — Phosphene is a panel UI for the port with batch queue, FFLF, Extend, and hardware-tier gating. Everything good about it traces back to your work. Link in profile.

**For a cocktailpeanut / Pinokio post:**
> Phosphene ships as a Pinokio one-click. Local LTX 2.3 video+audio on a Mac, no terminal. The Pinokio install story is what makes this remotely shareable to non-CLI people — thank you for that.

## 5. Pin the Post — Instructions

Pin the **hero tweet** (Variant C — announcement), not the thread root individually — the pinned tweet should auto-expand the thread for new visitors. Keep it pinned for at least two weeks; local-AI Twitter discovers things on a delay. Replace it only when you ship a meaningful update (LoRAs, CivitAI integration) worth re-pinning around.

## 6. Asset Descriptions

**Asset 1 — "Wizard, fireflies, low chant"** (recommended hero)
- Prompt: *"A weathered wizard in a dim forest clearing at dusk, slowly raising both hands as fireflies spiral up from his palms in a widening helix. Soft volumetric light through pine branches. Camera: slow push-in. Audio: low whispered chant in an unknown language, crackling ember pops, distant owl."*
- Why: Audio is unmistakably tied to the visual (chant + lip movement, embers + spark VFX). One watch and the differentiator lands without you having to caption it.

**Asset 2 — "Rain on a tin roof, woman lighting a cigarette"**
- Prompt: *"Close-up of a woman in her 40s on a porch at night, lighting a cigarette. The flame catches on the second strike. Heavy rain in the background, neon sign reflecting in a puddle. Audio: tin-roof rain, lighter strike-strike-flick, slow exhale."*
- Why: Three distinct audio events (strikes, flick, exhale) sync to specific frames. Skeptics replay it to check the sync — that's the conversion moment.

**Asset 3 — "Tokyo alley, footsteps, vending machine"**
- Prompt: *"First-person walk down a narrow Tokyo alley at 2am, neon kanji signs, light rain on asphalt. Pass a humming vending machine. Camera: handheld POV, slight bob. Audio: wet footsteps on concrete, distant traffic, the specific 50Hz hum of a vending machine, light rain."*
- Why: The vending-machine hum is the stress test — ambient texture audio that locally-generated video models simply do not produce. If the hum is there, the claim is true. Strong technical credibility shot.
