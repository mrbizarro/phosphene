# Phosphene — Pinokio Discover Copy Pack

## 1. Pinokio Manifest (`pinokio.json`)

**title** (40 chars)
`Phosphene — LTX 2.3 Video on Apple Silicon`

**description** (193 chars)
`Local text/image-to-video on Mac with sound. Wraps the LTX 2.3 MLX port: lossless h264, first/last-frame interpolation, batch queue, RAM-aware feature gating. 32 GB minimum, 64 GB comfortable.`

---

## 2. Pinokio Discover Tile

**One-liner** (58 chars)
`Local video + audio generation for Apple Silicon, no cloud.`

**Bullets**
- Text or image to 5-second 1280×704 clips with native sound, on-device.
- First/last-frame interpolation and clip-extend, lossless yuv444p h264 out.
- Auto-detects your RAM tier (32/64/96/128 GB) and gates what would OOM.

---

## 3. README Hero

## Phosphene

Phosphene is a local generative-video panel for Apple Silicon Macs. It wraps `dgrauet/ltx-2-mlx` — the MLX port of Lightricks' LTX Video 2.3 — in a one-click Pinokio app, so you can run text-to-video and image-to-video on your own machine without sending a frame to anyone's server.

The thing most local video tools don't do: **Phosphene generates audio together with the video.** LTX 2.3's joint model produces ambient sound, footsteps, and dialogue from the same prompt that drives the picture. You get a clip that already has its own world.

A 5-second 1280×704 render takes about 7 minutes on an M4 Mac Studio in Q4 distilled mode, or about 12 minutes in Q8 two-stage when you want sharper faces. Output is lossless h264 at yuv444p crf 0 — we patched the upstream default of yuv420p because it dropped visible chroma blocks on skin tones.

The panel adds first/last-frame interpolation, a clip-extender, a Gemma prompt-enhance button, a persistent batch queue (submit 60 prompts, sleep, wake to 60 clips), and an in-app model downloader so you never touch the command line.

32 GB minimum, 64 GB comfortable, 96+ GB if you want everything unlocked. Free, MIT, made by Mr. Bizarro ([@AIBizarrothe](https://x.com/AIBizarrothe)).

---

## 4. What This Adds

**Generation modes**
- Text-to-video at 1280×704, 5-second clips, MLX-native on Apple Silicon.
- Image-to-video — drag in a still, get motion that respects the source.
- First/last-frame interpolation: give two images, model fills the in-between.
- Extend: add seconds onto either end of any existing clip.
- Joint audio + video — ambient, dialogue, foley from the same prompt.

**Panel UX**
- One-click Pinokio install, in-panel model downloader, no terminal needed.
- Drag-and-drop image picker with a recent-uploads strip for quick re-use.
- Gemma prompt-enhance button rewrites prose into LTX's structured format.
- Persistent batch queue survives panel restarts and overnight sleep cycles.
- Hardware tier auto-detect: Compact 32 / Comfortable 64 / Roomy 96 / Studio 128+.

**Performance**
- Q4 distilled: ~7 min per 5-second 1280×704 clip on M4 Mac Studio.
- Q8 two-stage: ~12 min per clip, noticeably sharper on faces and text.
- Lossless h264 output (yuv444p crf 0) — fixes upstream chroma blocking on skin.
- RAM-aware feature gating prevents OOM crashes on smaller Macs.
- Local-only: nothing leaves the machine, no API keys, no rate limits.

---

## 5. Screenshot / Asset Shot List

- **Hero clip (autoplay loop, with sound on hover):** Cheshire cat in a foggy forest saying "we're all mad here" — proves joint audio+video in 4 seconds. Use Q8 for the face.
- **Panel screenshot, full window:** showing the mode tabs (T2V / I2V / FFLF / Extend), the drag-drop picker mid-hover with a recent-uploads strip visible, and a queue with 3 jobs running.
- **FFLF demo, side-by-side:** left frame = still pond, right frame = same pond with a stone splash mid-air, center = 5-second video filling the motion. Single composite PNG.
- **Hardware tier modal screenshot:** the auto-detect dialog showing "Detected: 64 GB — Comfortable tier. Q8 two-stage enabled, 96 GB-only features hidden."
- **Render comparison still:** close-up face crop, yuv420p vs yuv444p, labeled. Sells the lossless patch in one glance.
- **Batch queue timelapse (15-sec sped-up screen recording):** 60 prompts dropping in, panel grinding overnight, morning view of the output folder filling with thumbnails.
