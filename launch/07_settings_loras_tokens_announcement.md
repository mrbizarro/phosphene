# Phosphene update announcement (settings + LoRAs + tokens)

Three drafts. Voice: casual, direct, light comic timing, no marketing
buzzwords, owns mistakes openly. **Pick one** and paste — or mix.

Recommended: **Option B** (short thread). Hero is short enough to land
on its own, and the replies give people who care the technical detail
without bloating the timeline.

---

## Option A — single punchy tweet (~280 chars)

```
Phosphene update.

Renders are 7× smaller by default — X uploads work now. LoRAs are in (HDR is one click, CivitAI browser built in). No more .env files for tokens, paste them in Settings.

Update in Pinokio. Make weird shit. 🎬
```

256 chars. Good for casual X drop. Use when you don't want to thread.

---

## Option B — short thread (RECOMMENDED)

### Hero (245 chars)

```
Phosphene update — shipped a lot this weekend.

Renders are 7× smaller by default (X uploads work now), LoRAs are in (HDR is one click, CivitAI browser built in), and no more .env files anywhere.

Update in Pinokio. 🎬

🧵 details
```

### 1/ — the codec confession (276 chars)

```
On output: I was shipping yuv444p crf 0 lossless by default. Great for color grading, terrible for X — which straight up rejects yuv444p with a misleading error.

That was a design mistake. Default is now Standard (yuv420p crf 18, ~7 MB per 5s clip). Lossless is still there.
```

### 2/ — what "still there" means (251 chars)

```
The lossless preset is now called "Video production" because that's who actually picks it — colorists, editors, VFX folks who'll grade or composite downstream.

If you just want a clip that plays on X, Standard is your default and you don't have to think about it.
```

### 3/ — LoRAs (276 chars)

```
LoRAs work end-to-end now.

— HDR: one-click toggle (it's the official Lightricks LoRA fused at load time, you don't need to know that)
— Drop any .safetensors in mlx_models/loras/ → shows up in the picker with strength slider
— Built-in CivitAI browser: search, preview, install
```

### 4/ — no env files (276 chars)

```
Asking normal users to set CIVITAI_API_KEY and HF_TOKEN in their shell felt wrong. Most people don't know what an env var is.

Settings modal now has a tokens section. Paste your CivitAI key + HF token once, stored locally, never sent anywhere except the actual auth headers.
```

### 5/ — credits + close (236 chars)

```
Update via Pinokio Discover or hit Update if you've already got it. Click ⚙ for tokens + output presets, expand LoRAs in the form, browse CivitAI in-app.

Big thanks to @cocktailpeanut for catching the gnarliest bugs along the way.

Make weird shit and tag me.
```

---

## Option C — Discord / Reddit / Pinokio post (long-form)

```
Phosphene update — shipped a lot this weekend, three things you'll feel:

**1. Renders are 7× smaller by default.**
The output was lossless yuv444p crf 0. Great for color grading, terrible
for X — which straight-up rejects yuv444p with a misleading error. That
was a design mistake on my end. Default's now Standard (yuv420p crf 18,
~7 MB per 5s clip, plays everywhere).

The lossless preset is still there, renamed "Video production" because
that's who actually picks it. Plus a Web preset for smallest files and
a Custom mode if you want a specific pix_fmt + CRF.

**2. LoRAs work end-to-end.**
- HDR is a one-click toggle next to the Enhance button. (It's actually
  the official Lightricks HDR LoRA fused at pipeline-load time. You
  don't need to know that.)
- Drop any .safetensors into mlx_models/loras/ and it shows up in the
  LoRA picker, with a strength slider per row.
- Built-in CivitAI browser: search filtered to LTX 2.3 LoRAs, preview
  the actual animations (most LTX LoRAs ship video previews), Install
  button drops the file in mlx_models/loras/ with the right metadata
  sidecar. Auto-enables the LoRA after install.

**3. No more .env files.**
Asking normal users to set CIVITAI_API_KEY and HF_TOKEN in their shell
felt wrong. Most people don't know what an env var is.

Settings modal now has a tokens section. Paste your CivitAI key + HF
token (read access is enough), click Apply. Stored locally in
panel_settings.json, never sent anywhere except the actual auth
headers to civitai.com / huggingface.co.

Power users can still override with shell env vars — settings win when
both are set.

---

Update via Pinokio Discover or hit Update if you already have Phosphene
installed. Click the ⚙ in the header for token + output settings,
expand the LoRAs section in the form, browse CivitAI in-app.

Big thanks to @cocktailpeanut who caught the gnarliest bugs along the
way (silent helper crashes, audio regressions, the whole start.js URL
thing).

Make weird shit and tag me at @AIBizarrothe.
```

---

## Voice notes for the next announcement

What I'm matching here:

- **Open with the user-felt outcome, not the implementation.** "Renders
  are 7× smaller" lands harder than "we now ship yuv420p by default."
- **Own mistakes openly.** "That was a design mistake on my end" reads
  honest; "we've optimized the encoding pipeline" reads like marketing.
- **One throwaway joke per beat, max.** "You don't need to know that"
  about the HDR LoRA. "Make weird shit" as the close. Don't pile.
- **No buzzwords.** No "powerful", "robust", "seamless", "reimagined".
- **Concrete numbers.** "7×", "5s clip", "yuv420p crf 18" — anchors the
  claim.
- **Credit collaborators by handle, not by role.** @cocktailpeanut, not
  "the Pinokio creator."
- **Close with action, not signature.** "Update in Pinokio." or "Make
  weird shit and tag me." Not "Cheers!" or "Looking forward to feedback."
