# Anonymous usage analytics

Phosphene sends a small number of anonymous counts so that bugs in the field
get noticed. This page is the complete specification: every event, every
field, every value. If the panel ever sends something that isn't listed here,
that's a bug — please open an issue.

**In one line:** counts and hardware classes, never content. No prompts, no
filenames, no paths, no images, no video, no audio, no seeds, no LoRA or
character names.

---

## The short version

| | |
|---|---|
| **What identifies you** | One random UUID, generated on your Mac, tied to nothing |
| **How often** | Once ever when the install is new, once per panel start, plus once per finished render. No heartbeats |
| **Where it goes** | PostHog (`us.i.posthog.com`), or a self-hosted endpoint you choose |
| **Location** | None sent, and every event tells the receiver not to derive one from your IP — [details](#location) |
| **How to turn it off** | Settings → *Anonymous usage analytics* → **Turn off**, or `PHOSPHENE_ANALYTICS_DISABLED=1` |
| **Default** | ON, and the shipped build carries a working key — see below |
| **What you can inspect** | `state/usage-log.jsonl` — a plain-text copy of everything the panel sends |
| **Effect on renders** | None. Background thread, 2 s timeout, all failures ignored |

**The build you install ships a project key.** Phosphene is distributed as
source — you `git clone` it through Pinokio — so a key left empty in the
repository would never reach anyone and these counts would not exist. What is
committed, in `mlx_ltx_panel.py` as `ANALYTICS_KEY_DEFAULT`, is a PostHog
**project key** (`phc_…`): write-only and client-side by design. It can send
events. It cannot read one back, list anything, or change the project. Reading
the data needs a separate *personal* key, which is **not** in this repository.
Shipping a client key in the source is what PostHog documents these keys for —
so the thing worth checking is not that a key exists, it's what the panel does
with it, which is the rest of this page.

**A fresh clone therefore does send.** Clone it, run it, and it sends exactly
the events listed below and nothing else. The first time it ever runs, it says
so — one line, in the panel log, before you have read anything:

> `Phosphene sends anonymous usage counts (version, hardware class, render
> stats, error signatures - never your prompts or media). Disable in Settings.`

**Turning it off is the toggle, not the key.** Settings → *Anonymous usage
analytics* → **Turn off** stops the network calls *and* the local log;
`PHOSPHENE_ANALYTICS_DISABLED=1` does the same and beats the setting. Emptying
the project-key field does **not** silence the panel: that field is an override
for forks and self-hosters, and an empty override falls back to the shipped
key. Use the toggle.

---

## How to turn it off

**Settings → Anonymous usage analytics → Turn off.** One click; no confirm
step. Turning it off stops the network calls *and* stops writing the local
log — `_analytics_capture()` returns before it builds a payload, so there is
nothing left to send or mirror.

Before the panel has ever written a settings file — or if you'd rather set it
in your environment — use:

```sh
PHOSPHENE_ANALYTICS_DISABLED=1
```

That env var wins over the setting, always.

**Clearing the project key is not an off switch.** It is an override field: an
empty override resolves back to the key this build ships with. The two things
above are the off switches, and they are the only two.

---

## What identifies you

A single random **UUID4**, generated the first time an event is captured and
stored as `analytics_install_id` in `state/panel_settings.json`.

It is **not derived from anything**: not your hardware serial, not the MAC
address, not the username, not the hostname, not the install path. Delete the
key (or the file) and this install becomes a brand-new anonymous install with
no way to correlate it to the old one. The panel shows you the exact value in
Settings.

---

## Location

The panel sends no location field of any kind — no country, no city, no region,
no timezone, no coordinates, no locale. It never has.

That on its own is not the whole promise, because a receiver can *derive* a
location from where a request came from, and PostHog does that by default. So
every event now also carries two instructions telling it not to:

| Property | Value | What it does |
|---|---|---|
| `$geoip_disable` | `true` | PostHog's GeoIP step returns immediately instead of deriving country, city, subdivision, timezone and the city's coordinates |
| `$ip` | `"0.0.0.0"` | PostHog copies the connecting address into the event's `$ip` property only when the event didn't bring its own. Bringing one keeps the real address off the stored event |

**What that does not do, said plainly:** the request still arrives over TCP from
a real address, and nothing inside a request body can change that — that is how
HTTP works, for this panel and for every other program on your Mac. What the
two flags control is what the receiver is instructed to *derive* from that
address and *store* on the event. Discarding it at the edge as well is a
setting on the receiving project, not something this source tree can promise
you, so this page does not.

If you want the connection itself gone: turn analytics off, or point
`PHOSPHENE_ANALYTICS_HOST` at a receiver you run (below).

**A note on `0.0.0.0`, since it looks arbitrary.** The obvious spelling —
sending `$ip: null` — does nothing at all: PostHog's ingest fills the property
in when the event's value is *falsy*, so a null is silently replaced by your
real address. It has to be a non-empty string. `127.0.0.1` would be the natural
choice and is the wrong one: PostHog rewrites loopback and `192.168.*` to a real
address in Sweden as a local-development convenience, which would invent a
location the day the disable flag ever went missing.

---

## Events

Seven events. That's the whole list. (An eighth is described at the end
because people expect it — it is documented as the thing we deliberately don't
send.)

### `app_installed`

Once per install, **ever** — fired on the first boot that an install id
reports, immediately before its first `app_boot`.

| Field | Type | Example | Notes |
|---|---|---|---|
| `version` | string | `"3.7.0"` | Which release a new install actually landed on |
| `chip_family` | string | `"M4 Max"` | Same hardware *class* as on `app_boot` |
| `ram_gb` | int | `64` | Unified memory, rounded to whole GB |

**Why it exists:** it is the denominator. Without it, "new installs over time"
has to be reverse-engineered from first-sightings of unique install ids, which
is both fiddly and worse for you — it is the shape of question that tempts
someone to start profiling ids. One counter answers it instead.

**Once-ever, and how:** a boolean `analytics_install_reported` is written to
`state/panel_settings.json` the moment the event is captured, and it is checked
before every subsequent boot. Rebooting the panel never re-counts. Deleting
your install id (or the settings file) makes a genuinely new install, which
reports its own `app_installed` once — that is the same "no way to correlate it
to the old one" property described above, not a loophole in it.

### `app_boot`

Once per panel start.

| Field | Type | Example | Notes |
|---|---|---|---|
| `version` | string | `"3.4.1"` | Contents of the repo `VERSION` file |
| `os_version` | string | `"26.4"` | macOS major.minor. Patch level deliberately dropped |
| `chip_family` | string | `"M4 Max"` | Parsed from the CPU brand string. A hardware *class*, identical across every machine of that model. `"unknown"` or `"non-apple-silicon"` when unparseable |
| `ram_gb` | int | `64` | Unified memory, rounded to whole GB |
| `cap_tier` | string | `"q8"` | `q4` or `q8` — which capability surface the UI is showing. A statement about this Mac's RAM folded down to what the generation can serve, NOT about which weights are on disk |
| `model_version` | string | `"ltx25"` | Which LTX generation this install serves (`ltx23` / `ltx25`). Added in v4.0 as a NEW field rather than by redefining `cap_tier`, so the existing capability series stays comparable across the 2.5 cutover |
| `packs` | object | `{"h3":true,"sharp":false,"q8":true,"qwen":false}` | Booleans only: is each optional pack installed |
| `h3_chain_supported` | bool | `false` | Whether the installed H3 runner supports window chaining (10 s / 15 s tiers) |

### `render_completed`

Once per job that finishes successfully — every engine and every mode,
including image and training jobs.

| Field | Type | Example | Notes |
|---|---|---|---|
| `engine` | string | `"ltx"` | `ltx` or `h3` |
| `mode` | string | `"i2v"` | `t2v`, `i2v`, `extend`, `keyframe`, `a2v`, `restore`, `ingredients`, `control`, `upscale`, `image`, `train` |
| `tier` | string | `"standard"` | LTX quality (`quick`/`balanced`/`standard`/`high`) or the H3 composite cell (`standard_5s`, `high_10s`, …) |
| `duration_bucket` | string | `"5-15m"` | Legacy 5-value bucket, kept one more release for dashboard continuity, then removed in favour of `wall_sec_bucket` |
| `wall_sec_bucket` | int | `480` | The lower edge of a log-spaced ladder (`15, 30, 45, 60, 90, 120, 180, 240, 300, 420, 600, 900, 1200, 1800, 2400, 3600, 5400`). Sharp enough to see a performance regression, far too coarse to be a timing fingerprint. Absent when the wall clock is unknown |
| `resolution` | string | `"1216x704"` | Output dimensions, or `"unknown"` |
| `canvas_class` | string | `"720p"` | Coarse megapixel class: `<=480p`, `576p`, `720p`, `1080p`, `native+` |
| `frames` | int | `121` | Frame count |
| `version` | string | `"4.0.6"` | Same value `app_boot` sends. Repeated here so "failure rate by version" is a one-click breakdown instead of a join |
| `chip_family` | string | `"M4 Max"` | Hardware **class**, identical across every machine of that model — same value `app_boot` sends |
| `ram_gb` | int | `64` | Hardware class — same value `app_boot` sends |
| `os_version` | string | `"macOS 15"` | Coarse — same value `app_boot` sends |
| `steps` | int | `8` | Sampler depth the job ran — separates a real regression from a user on 30 steps |
| `accel` | string | `"off"` | `off` / `boost` / `turbo` |
| `temporal_mode` | string | `"native"` | `native` or `fps12_interp24` (Long Clip Boost) |
| `upscale` | string | `"fit_720p"` | `off` / `fit_720p` / `fit_1080p` / `x2` |
| `upscale_method` | string | `"lanczos"` | `lanczos` or `pipersr` |
| `schedule_preset` | string | `"default"` | `default` or `fast` — the LTX-2.5 draft schedule's adoption |
| `chain_windows` | int | `2` | H3 chained-window count (1 = single pass) |
| `chain_prompts_used` | bool | `false` | Did the per-window shot list carry any text |
| `lora_count` | int | `1` | **Count only. Which adapters is deliberately never sent** |
| `lora_kinds` | string | `"style"` | Closed vocabulary: `none` / `style` / `character` / `mixed` |
| `character_used` | bool | `true` | A cast character drove this render |
| `audio_mode` | string | `"joint"` | `joint` / `none` (external audio replaces the generated track) / `a2v_dub` / `h3_native` |
| `first_render` | bool | `true` | Present once per install, on its first successful render ever — the activation funnel without a join |
| `source` | string | `"storyboard"` | v4.9.7 — which surface queued the job. **Closed vocabulary**: `form`, `batch`, `storyboard`, `characters`, `image_studio`, `retry`, `api`, `chain` (an Upscale ×2 queued automatically behind an H3 draft), `unknown`. Makes "how much of the rendering is Storyboard" a one-click breakdown |

### `render_failed`

Same as `render_completed` (minus `first_render`), plus:

| Field | Type | Example | Notes |
|---|---|---|---|
| `error_class` | string | `"metal_watchdog"` | **A closed 17-value taxonomy** (`refused`, `oom_jetsam`, `metal_watchdog`, `native_crash`, `helper_start_timeout`, `helper_exit`, `model_missing`, `model_corrupt`, `download_failed`, `venv_broken`, `bad_params`, `input_missing`, `disk_full`, `export_failed`, `timeout`, `cancelled_race`, `other`). Classification runs on the original error text locally; **only the class leaves the machine.** `refused` is in the taxonomy but never rides on a `render_failed` event — it is the value that routes the event to `render_refused` instead, below |
| `error_fingerprint` | string | `"a3f09c21e7b4"` | Only when `error_class` is `other`: 12 hex chars of the SHA-256 of the already-scrubbed first line. Lets "the same unknown error, 17 times, all on M1 Max" be counted without transmitting the text — the readable line stays in your own `state/usage-log.jsonl` |
| `error_signature` | string | `"RuntimeError: helper exited before first frame"` | **The only free-text field the panel sends**, see the scrubbing rules below. Kept for ONE transition release alongside `error_class`, then removed |

Cancelled jobs are **not** reported — a user cancelling is not a signal about
the software.

Neither are **refusals** — those get their own event, next.

> **⚠️ A SERIES CHANGES SHAPE FROM v4.6.1 — `model_missing` gains, `other`
> loses.** Four render faults that mean *"your Hailuo H3 clone is older than
> this panel"* (its runner has no `--lora`, no `--first-frame`, or no
> `--chain-windows`) used to classify as `other`, because each raise site wrote
> its own sentence and no needle matched any of them. They are not unknowns:
> they are one well-understood fault with one remedy — re-run **Install Hailuo
> H3**, which keeps every weight already on disk. On the owner's ruling
> (2026-08-23) they now classify as **`model_missing`**, matched on the shared
> phrase `H3_RUNNER_BEHIND` (`the installed Hailuo H3 runner is behind this
> panel`) so a fifth flag added later is classified without another edit.
>
> Consequences for anyone reading the fleet: a step up in `model_missing` and a
> step down in `other` at that release is **this change, not a regression**, and
> the affected `error_fingerprint` values stop being emitted entirely (a
> fingerprint rides only on `other`). Do not compare `model_missing` across the
> boundary without saying which side you are on.

> **⚠️ A SECOND SERIES CHANGES SHAPE FROM v4.6.1 — `input_missing` gains, and
> it was previously EMPTY.** Its needles were `does not exist` / `no longer
> exists`, and **no raise site in this codebase has ever said either** — every
> one of them says *not found*. So the class matched nothing at all, and the
> widest-spread real failure in the fleet (`image not found: <path>` — 35 events
> across 22 people in 14 days) sat in `other`. It now matches, on the
> colon-anchored `not found:` plus `no longer on disk`.
>
> The same change fixed a **misclassification driven by the user's own
> filesystem**: classification runs on the raw error text *including the path*,
> and `download_failed` matches the bare word `download`, so a missing reference
> image living in `~/Downloads` was counted as a failed fetch. `input_missing`
> is now asked before the loose half of `download_failed` (a single needle,
> `repo not found`, is hoisted above it so a hub lookup stays a fetch fault).
>
> So at that release expect: `input_missing` to go from ~zero to a real number,
> `other` to drop again, and a small, permanent drop in `download_failed`. All
> three are this change.

### `feature_used` (v4.9.7)

The surfaces render events cannot see. One event per action, no free text.

| prop | type | example | why |
|---|---|---|---|
| `feature` | string | `"storyboard_plan"` | **Closed vocabulary**: `storyboard_plan`, `storyboard_export`, `editor_open`, `editor_export`, `civitai_download`, `sample_character`, `train_start`, `enhance_prompt`. Unknown names are dropped locally, never sent |
| `detail` | string | `"nle"` | Optional, lowercase `[a-z0-9_.-]` only, ≤32 chars — a sub-choice within the feature |
| `version` | string | `"4.9.7"` | As on `app_boot` |

### `app_updated` (v4.9.7)

Sent once at the first boot after the running version changed. Answers "did people move to the new release" without a per-install join across `app_boot` rows.

| prop | type | example | why |
|---|---|---|---|
| `from_version` | string | `"4.9.5"` | The version the previous boot ran (kept locally in settings) |
| `to_version` | string | `"4.9.7"` | The version now running |

### `update_prompt` (v4.9.7)

How the update pop-up and banner are answered — the only way to know whether the "please update" surfaces work.

| prop | type | example | why |
|---|---|---|---|
| `action` | string | `"later"` | **Closed vocabulary**: `shown`, `update_now`, `later` (the pop-up), `banner_update`, `banner_later` (the banner), `restart_needed` (the "Restart to finish update" pill, once per page load) |
| `version` | string | `"4.9.5"` | The version that was prompted |

### `broadcast_seen` (v4.9.7)

One event when a developer broadcast (`BROADCAST.json`) is acknowledged. No text — which message is derivable from the date.

### `queue_paused_breaker` (v4.9.7)

The queue paused itself after three identical failures in a row.

| prop | type | example | why |
|---|---|---|---|
| `n_failed` | int | `3` | The streak length when it fired |
| `queued` | int | `12` | Jobs left waiting |
| `error_class` | string | `"model_missing"` | The same closed taxonomy as `render_failed` |

The browser reports `update_prompt`, `broadcast_seen` and the Editor's two `feature_used` values through **one** route, `POST /analytics/ui`, which allowlists every (event, value) pair and answers 400 to anything else. Everything else is emitted server-side.

### `render_refused`

Once per job the panel **declined on purpose**. Same fields as
`render_failed`, except that the three error fields are replaced by one:

| Field | Type | Example | Notes |
|---|---|---|---|
| `refusal` | string | `"ingredients_generation"` | **A closed vocabulary** — `ingredients_generation`, `hardware_tier`, `h3_ram`, `h3_mode`, `h3_lora_slots`, `stale_engine` (v4.9+: a 2.5 render refused because the vendored engine predates the Gemma 4 tower — remedy is a second Update click). `image_ram` (v4.9.3+: the Image Studio memory guard — the engine holds more weights than this Mac can hold, or too little is free right now). `pack_missing` (v4.9.3+: High / Keyframes / Extend chosen on a Mac that has not downloaded the Q8 add-on — remedy is Settings → Models). Which guard said no. No free text, no fingerprint: the message is our own copy, so there is nothing to scrub and nothing unknown to count |

#### The rule — a refusal is not a failure

This is the distinction the two event names exist to keep apart, and it is
the one thing on this page most likely to get quietly re-merged by someone
adding a chart:

> **`render_failed`** — the engine tried and something went wrong. A bug, a
> crash, a missing file, an out-of-memory. Every one of these is a defect
> report.
>
> **`render_refused`** — the panel understood the request perfectly and
> declined it, because *this install* cannot serve that capability: the wrong
> hardware tier, the wrong engine, or the wrong model generation. Nothing
> broke. No GPU time was spent. The message names the way out.

**Therefore: `render_refused` is not in the failure rate, is not in the render
count, and is not in the engine mix — neither numerator nor denominator.** A
refusal did not render, so counting it as a render (successful *or* failed)
makes both numbers lie.

**Why a separate event rather than an `error_class` value.** Both were built
and the event won. A class value keeps one event and one funnel, but it makes
every query that touches failures carry an `error_class != 'refused'`
exclusion — in this file, in `_USAGE_FLEET_QUERIES`, and in every PostHog
insight anyone writes afterwards. One forgotten exclusion silently restores
the exact bug this replaced. A separate event name is right by default: a
query that asks about `render_failed` gets failures, always, without anyone
having to remember anything.

**What it is actually for.** A refusal is a *product* signal, not an
engineering one. The right reading of a tall `refusal` bar is "the UI offered
a control that can never work on that install" — the fix is upstream, in the
form, not in the engine. That is precisely how this event came to exist: on
2026-08-23 the largest single `render_failed` signature in the whole fleet
turned out to be the Ingredients-on-2.5 refusal — 65 events from 16 different
people, all of them clicking a button that could not possibly succeed. The
guard was right; offering the button was not. Both were fixed in the same
change.

**2026-08-28 — the `h3_ram` slug's WORDING moved; the series did not.** That
refusal used to say "Hailuo H3 needs about 64 GB of unified memory", a number
neither of the product's two floors has ever been (60 on the bf16 lane, 46 on
the Q8 DiT lane). It now says one of three things, one per RAM band — see
`h3_ram_verdict()`. The slug, the event and the series are unchanged, and
`_ANALYTICS_REFUSAL_REASONS` gained needles for the two new sentences **while
keeping the retired one**, so a replayed pre-2026-08-28 usage log still
classifies as `refused` instead of falling back into `other`. Nothing raises
the old sentence any more. Counts across the cutover stay comparable; a rise
in `h3_ram` after it is a real change in who is being refused, not a rewording
artefact.

**A refusal still shows as an error in your own panel.** The job lands in
history with a red card and the refusal text, because nothing was produced
and that is honest. The queue does not gain a fourth status. The split is in
what gets *counted*, which is what was wrong.

#### How `error_signature` is scrubbed

In this order:

1. **First line only.** Tracebacks and multi-line detail are discarded.
2. **Exact content redaction.** This job's prompt, negative prompt, image
   path, audio path, output path, character name and training-job id are
   removed by exact substring match → `<redacted>`. This is the defense that
   matters: the realistic leak is an exception that quotes your prompt back.
3. **Path stripping.** Anything shaped like an absolute path — `/Users/…`,
   `~/…`, `/private/var/…`, `/Volumes/…`, or any run of two or more path
   segments — becomes `<path>`.
4. **Truncation to 120 characters.**

Strings shorter than 6 characters are not redacted, so a terse prompt can't
blank out ordinary words in an error message.

### `pack_state_change`

Fired at boot when an optional pack's installed state differs from the
previous boot. At most four per boot; usually zero.

| Field | Type | Example |
|---|---|---|
| `pack` | string | `"h3"` — one of `h3`, `sharp`, `q8`, `qwen` |
| `from` | bool | `true` |
| `to` | bool | `false` |

A `true → false` transition means a pack that *was* installed is no longer
detectable — a broken install, not a user choice. This is the whole reason
this event exists: nothing else in the panel notices that today.

### `star_prompt`

Fired when someone answers the one-time GitHub star ask — at most once per
install, and only on a deliberate click. Nothing fires when the ask is
ignored or dismissed.

| Field | Type | Example |
|---|---|---|
| `via` | string | `"link"` — one of `link` (opened the repo) or `already` (said they had already starred) |

A closed two-value vocabulary, coerced in the handler: anything else becomes
`link`. No account, no username, no repo state — the panel cannot see whether
a star was actually given, and does not ask GitHub. `via` exists only to keep
"a click we caused" separate from "a click that told us we did not need to
ask", which is the whole question the ask was added to answer.

### `engine_selected` — not implemented, on purpose

The obvious way to count engine-picker usage is a ping per click. That adds a
network request to an interaction that currently has none, and it would be by
far the chattiest event in the system. The same question is answered for free
by the `engine` field on every render event plus the `packs.h3` field on
`app_boot`, so the picker is measured by what people actually render rather
than what they click. If clicks-without-renders ever become the question,
that's the point to add it.

---

## What is never sent

Not "we try not to send" — these are dropped by name before any payload is
built, and there's a test asserting it:

> `prompt`, `negative_prompt`, `override_prompt`, `caption`, `image`,
> `image_path`, `images`, `audio`, `audio_path`, `video`, `output`,
> `output_path`, `raw_output`, `native_output`, `path`, `paths`, `file`,
> `filename`, `files`, `dir`, `directory`, `root`, `first_frame`,
> `last_frame`, `refs`, `reference`, `seed_image`, `lora`, `loras`,
> `lora_path`, `lora_paths`, `character`, `trigger`, `trigger_words`,
> `hostname`, `username`, `user`, `email`, `home`, `command`, `cmd`, `argv`,
> `env`, `token`, `key`, `api_key`

No media file ever leaves the machine under any circumstance — there is no
code path that reads a rendered file for analytics purposes.

---

## The local log

Every captured event is appended to **`state/usage-log.jsonl`** *before* the
network is touched, and independently of whether the send then succeeds. One
JSON object per line:

```json
{"event":"render_completed","props":{"engine":"ltx","mode":"t2v","tier":"standard","duration_bucket":"2-5m","wall_sec_bucket":180,"resolution":"1216x704","canvas_class":"720p","frames":121,"version":"4.0.6","chip_family":"M4 Max","ram_gb":64,"os_version":"macOS 15","steps":8,"accel":"off","temporal_mode":"native","upscale":"fit_720p","upscale_method":"lanczos","schedule_preset":"default","chain_windows":1,"chain_prompts_used":false,"lora_count":0,"lora_kinds":"none","character_used":false,"audio_mode":"joint"},"install_id":"…","ts":1754400000.0,"at":"2026-08-05 17:40:00","utc":"2026-08-05T14:40:00Z"}
```

Two reasons it exists: it's the "this machine" data source for the Usage
section of the stats dashboard, and it means you never have to take this
document's word for anything — you can read exactly what the panel sent, line
for line, after the fact.

The file is capped at **5 MB**; at the cap the oldest half is dropped. It's
gitignored (both via `state/` and by basename) and never leaves your Mac.

---

## Owner setup — activating this

Both key fields live in **Settings → Anonymous usage analytics → Maintainer /
self-hosting keys**, and are stored in `state/panel_settings.json` (mode
`0600`). One of the two keys is committed and one must never be — that
asymmetry is the whole design.

### 1. The capture key — PostHog **Project API key** (shipped)

PostHog → *Project settings* → *Project API key* (starts `phc_…`). Write-only:
it can send events and do nothing else, which is why it is safe to hold on disk
and safe to commit. **Phosphene's own project key is already in the source** as
`ANALYTICS_KEY_DEFAULT`, and it is what a stock install reports with.

Paste a different one into **PostHog project key (capture)** to point a fork at
its own project. That field *overrides* the shipped key; clearing it falls back
to the shipped key rather than switching capture off. To send nothing, use the
toggle or `PHOSPHENE_ANALYTICS_DISABLED=1`.

Env override: `PHOSPHENE_ANALYTICS_KEY`.

### 2. The read key — PostHog **Personal API key** (never committed)

PostHog → *Personal settings* → *Personal API keys* → create one with **read**
scope on the project (starts `phx_…`).

Paste it into **PostHog personal API key (fleet view)**. This unlocks the
fleet numbers in the Usage section at <http://127.0.0.1:8199/stats>. It is
never used for sending, only for querying.

Env override: `PHOSPHENE_ANALYTICS_QUERY_KEY`.

### Self-hosting

| Variable | Default | Purpose |
|---|---|---|
| `PHOSPHENE_ANALYTICS_HOST` | `https://us.i.posthog.com` | Ingestion endpoint. Must accept a PostHog-shaped single-event `POST /i/v0/e/` |
| `PHOSPHENE_ANALYTICS_API_HOST` | `https://us.posthog.com` | Query API host for the fleet view |
| `PHOSPHENE_ANALYTICS_PROJECT` | `@current` | PostHog project id used in the query URL |
| `PHOSPHENE_ANALYTICS_DISABLED` | *(unset)* | `1` disables everything, overriding the setting |

---

## The dashboard — `/stats` → Usage

The maintainer dashboard at <http://127.0.0.1:8199/stats> (127.0.0.1-only,
like every other panel endpoint) has a **Usage** section fed by
`GET /stats/usage`. Two tiers, one renderer:

- **`this mac`** — aggregated from `state/usage-log.jsonl`. Always available,
  needs no keys, works offline. Labelled *"this machine only — add a PostHog
  query key in Settings for fleet data"*.
- **`fleet`** — aggregated by PostHog across every install that has pinged.
  Requires the personal API key. Cached to `state/usage-fleet.json` for **6
  hours**; the section's *refresh* button bypasses the cache.

It shows: weekly active installs, renders this week, H3 share, error rate,
refusals this week, the top 5 error signatures of the last 7 days, the top
refusals of the last 7 days (with how many *distinct people* hit each — 65
refusals from 16 people is a product bug, 65 from one person is a bookmark),
version / chip / memory distributions, and a **pack-regression alert** that
turns red when any pack went `true → false` in the last week. Refusals sit in
their own tile and their own column, never inside the error rate.

The fleet view runs eleven read-only aggregate HogQL `SELECT`s against `events`.
Each is independent — one failing leaves that panel empty rather than
collapsing the view. If every query fails, the section falls back to local
data with a visible warning.

---

## Design notes

**Why default ON.** An opt-in version of this shipped on 2026-05-21 and was
reverted the next day (`da1d6f5`). It was off by default, which meant it would
have told us nothing even if it had stayed. The trade this version makes
instead: default ON *with a key that really ships*, paid for by a far smaller
payload, a one-line disclosure in the boot log the first time it runs, a
one-click off switch that stops the local log too, and a plain-text copy on
your own disk of every event that leaves.

**A correction, recorded rather than quietly fixed.** Until 2026-08-12 this
page said the public repo shipped no key and that a fresh clone sent nothing.
That stopped being true on 2026-08-09 (`acfbdc7`), when the project key was
committed on purpose — and the page kept saying it for three days. Nothing
extra was ever collected; the page was simply wrong about the default, which
for a page whose entire job is being checkable is the worse kind of wrong.
Adding an event or a field is a documentation change first: if `ANALYTICS.md`
and the dry-run suite aren't in the same commit, the commit is incomplete.

**Why it can't break a render.** `_analytics_capture()` builds the payload,
starts a daemon thread and returns. Delivery has a 2-second timeout and a bare
`except: pass` around the entire path. Nothing is retried and nothing is
queued — a dropped event is strictly preferable to state that outlives the
render it describes. There are no background timers and no heartbeats.

**Where it's wired.** Two call sites, both in `mlx_ltx_panel.py`:
`_analytics_boot()` in `__main__` (never at import time, so `import
mlx_ltx_panel` from a script sends nothing), and `_analytics_render_event()`
in `worker_loop`'s `finally` — the single point every job from every engine
passes through, so a future engine is counted for free.

---

## Verifying it yourself

```sh
# The dry-run suite: no network, no panel, isolated temp state dir.
python3 scripts/test_analytics_dryrun.py
```

56 tests covering: the shipped key is a write-only `phc_` project key and is
really the one that goes on the wire, the toggle and the env kill-switch each
produce zero sockets *and* zero log lines, the exact field set of every event,
every event name the panel can fire is documented on this page, the
refusal/fault fork (a refusal is `render_refused` and never a `render_failed`
carrying `error_class: refused`; a real crash is untouched; refusals stay out
of the local error rate), prompt/path
non-leakage (including a prompt quoted inside an exception), forbidden-key
dropping, the geo-disable and `$ip` flags riding on *every* event type and
being un-overridable by a call site, bucketing, log rotation and the local
aggregates.

To watch it live, tail the local log while you render:

```sh
tail -f state/usage-log.jsonl
```

## Fleet dashboard tiles (HogQL)

The queries behind the **Phosphene Fleet** dashboard's tiles, kept here so the board can be rebuilt (PostHog → New insight → SQL → paste → Add to dashboard). Each is a one-click breakdown; none joins across installs.

**Activation — installs vs first renders per day (14d)**

```sql
select toDate(timestamp) as day, countIf(event='app_installed') as installs, countIf(event='render_completed' and toString(properties.first_render)='true') as first_renders
from events where timestamp > now() - interval 14 day group by day order by day
```

**Versions in use — active installs by version (7d)**

```sql
select properties.version as version, count(distinct distinct_id) as installs
from events where timestamp > now() - interval 7 day and event='app_boot' group by version order by installs desc
```

**Failure rate by version (7d)**

```sql
select properties.version as version, countIf(event='render_completed') as ok, countIf(event='render_failed') as failed, round(100*failed/greatest(ok+failed,1),1) as fail_pct
from events where timestamp > now() - interval 7 day group by version having ok+failed >= 10 order by version
```

**Why renders fail — error class (7d)**

```sql
select properties.error_class as class, count() as events, count(distinct distinct_id) as installs
from events where event='render_failed' and timestamp > now() - interval 7 day group by class order by events desc
```

**Where we said no — refusals (7d)**

```sql
select properties.refusal as reason, count() as events, count(distinct distinct_id) as installs
from events where event='render_refused' and timestamp > now() - interval 7 day group by reason order by events desc
```

**Where renders come from — source (7d)**

```sql
select properties.source as source, count() as renders, count(distinct distinct_id) as installs
from events where event in ('render_completed','render_failed') and timestamp > now() - interval 7 day group by source order by renders desc
```

**Feature use (7d)**

```sql
select properties.feature as feature, count() as uses, count(distinct distinct_id) as installs
from events where event='feature_used' and timestamp > now() - interval 7 day group by feature order by uses desc
```

**Update prompt — how people answer it (7d)**

```sql
select properties.action as action, count() as events, count(distinct distinct_id) as installs
from events where event='update_prompt' and timestamp > now() - interval 7 day group by action order by events desc
```

**Updates landed per day (14d)**

```sql
select toDate(timestamp) as day, count(distinct distinct_id) as installs_updated
from events where event='app_updated' and timestamp > now() - interval 14 day group by day order by day
```

**Hardware mix of active installs (7d)**

```sql
select multiIf(toFloat(properties.ram_gb) <= 16, '≤16 GB', toFloat(properties.ram_gb) <= 24, '24 GB', toFloat(properties.ram_gb) <= 36, '32–36 GB', toFloat(properties.ram_gb) <= 64, '48–64 GB', '96 GB+') as ram, count(distinct distinct_id) as installs
from events where event='app_boot' and timestamp > now() - interval 7 day group by ram order by installs desc
```

**Renders per active install (7d)**

```sql
select bucket, count() as installs from (select distinct_id, multiIf(c=0,'0', c<=2,'1–2', c<=10,'3–10', c<=50,'11–50','50+') as bucket from (select distinct_id, countIf(event='render_completed') as c from events where timestamp > now() - interval 7 day group by distinct_id)) group by bucket order by bucket
```

**Queue breaker fires (14d)**

```sql
select toDate(timestamp) as day, count() as fires, count(distinct distinct_id) as installs
from events where event='queue_paused_breaker' and timestamp > now() - interval 14 day group by day order by day
```
