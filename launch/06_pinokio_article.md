# Phosphene — Pinokio article (paste into beta.pinokio.co/posts/new)

Tone: informal, friendly, technical claim up front but no jargon dump.
Length: short enough to read in 90 seconds. People scroll.

Suggested title (pick one):
- **Phosphene — local video + audio on a Mac, with sound, in one pass.**
- **Phosphene: a free Mac panel for LTX 2.3 (joint audio + video, runs in MLX)**
- **I made a thing: Phosphene — generative video on Apple Silicon, with sound built in**

Recommended: the first one. The "with sound, in one pass" phrasing is the hook.

---

## ARTICLE BODY (paste this part)

Hey Pinokio 👋

Quick story: I've been using LTX 2.3 locally for a few weeks (the MLX port from [@dgrauet](https://github.com/dgrauet/ltx-2-mlx) is genuinely incredible — pure Apple Metal, no CUDA cosplay), and the thing that kept blowing my mind is **the audio**. LTX 2.3 doesn't generate video and then bolt on a soundtrack — it generates **video and audio jointly, in one forward pass.** Footsteps land on the right frame. Lip movement matches the chant. Ambient hum is tied to the visual.

Built a panel for it called **Phosphene**. It's a Pinokio one-click install. Mac-only (Apple Silicon — sorry, the architecture won't run on Intel, Linux, or Windows).

[ATTACH: assets/phosphene_banner.png — the hero phosphene field render]

### What it does

Four generation modes, all with synced audio:

- **T2V** — text → video + audio. Type a scene, get a 5-second clip with sound.
- **I2V** — start from a still image. Cover-crops to model dimensions, animates from there.
- **FFLF** — first frame + last frame. Drop in two images, the model fills the middle (great for stylized morphs).
- **Extend** — feed it an existing clip, get N more seconds tacked on. Stitch a few of these together for a real shot.

Plus a **prompt enhance** button that uses Gemma 3 12B (locally!) to rewrite your prompt in the structure LTX 2.3 was trained on. Saves the trial-and-error.

### Numbers (M4 Mac Studio 64 GB)

- 5-sec clip, 1280×704, Standard quality: **~7 minutes**
- High quality (Q8): ~12 minutes
- FFLF interpolation: ~5 minutes

Slower than an H100. Faster than a credit card. No queue, no API key, no monthly bill.

### Hardware tiers

The panel auto-detects your Mac's RAM and gates accordingly:

| RAM | Tier | What runs |
|---|---|---|
| 32 GB | Compact | Lower res, shorter clips |
| 64 GB | Comfortable | Full 1280×704 (the sweet spot) |
| 96 GB | High | Longer clips + Q8 |
| 128+ GB | Pro | Everything, no clamps |

Generative video is genuinely memory-hungry — there's no shortcut. But on a 64 GB Mac it's a real workflow.

[ATTACH: a generated hero clip — assets/phosphene_hero_x.mp4 — with sound on, *please*]

### What the audio actually does

Most local video models (Wan, Hunyuan, Mochi) ship video only. You add music in post. LTX 2.3 generates *both* and the timing is tied — three audio events in five seconds (lighter strike, strike, exhale) sync to specific frames. Skeptics will replay the clip three times to check the sync. That's the conversion moment.

**Pro tip:** describe the soundscape in your prompt. "Wizard in forest" gives you near-silent ambient. "Wizard in forest, low whispered chant, ember crackle, distant owl" gives you a layered audio bed. The audio model is conditional on prompt cues — feed it cues.

### Credits

Phosphene is a panel on top of three other people's work:

- **[@Lightricks](https://www.lightricks.com/)** — the LTX 2.3 model and the open weights
- **[@dgrauet](https://github.com/dgrauet/ltx-2-mlx)** — the MLX port (this is the work)
- **[Apple's MLX team](https://github.com/ml-explore/mlx)** — the framework
- **[@cocktailpeanut](https://twitter.com/cocktailpeanut)** — Pinokio itself, which is why one-click installers like this exist

I just glued things together with a UI on top.

### Try it

It's in Pinokio Discover — search **Phosphene**.

Or: [github.com/mrbizarro/phosphene](https://github.com/mrbizarro/phosphene) for the source. Free, MIT licensed (panel — model has Lightricks' license, read it before commercial use).

If something breaks, file an issue. If you want to fix something, PRs welcome. If you make something weird with it, tag me on X — [@AIBizarrothe](https://x.com/AIBizarrothe).

Cheers 🎬🔊

---

## ASSETS TO UPLOAD WITH THE POST

In order of importance:

1. **Hero video clip** — `mlx_outputs/<the latest standard wizard clip>.mp4` once today's hero generation finishes. Re-encode for web (yuv420p + faststart) before upload.
2. **Banner image** — `assets/phosphene_banner.png` (the "Phosphene field" hero illustration with the wordmark)
3. **Logo** — `assets/phosphene_logo.png` (square mark + wordmark for the post sidebar / thumbnail)
4. **Optional second video** — any of your earlier I2V or Extend outputs that have good visuals + audio sync

## NOTES FOR YOU (Salo)

- The article is in the user's voice ("I made a thing"), not yours — so you're the speaker, but it's casual ("hey Pinokio 👋").
- Don't paste the title section, that's just for picking. Start the post body at "Hey Pinokio".
- The "ATTACH:" markers are inline placeholders — when posting, drop the actual image or video at that spot.
- If beta.pinokio.co supports markdown, paste as-is. If it strips formatting, the structure still reads okay because I used short paragraphs and bullets.
- Cross-link this post in your X thread (variant C from launch/02_twitter.md) once it's up. They amplify each other.
