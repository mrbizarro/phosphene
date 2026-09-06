# Phosphene — project state, history, open work

> **🚀 2026-09-06 — v4.10.0 ships.** Everything on this branch since v4.9.5
> goes public as one release: Editor v2 (transitions, speed, titles, the
> Director + song map, retake, completion alerts, deliver-as, duplicate,
> search, framing), the timeline's NLE gestures, anchor stills, long shots,
> LoRA guides + update checks, closed-tab alerts, light theme, **One take**
> on both engines, Turbo v4-600 EMA, the constant-time windows chain, and the
> three 4.9.x hotfixes. The release media is *The Commuter*, a 75 s H3 one
> take at native. Also in: `docs/PROMPTING.md` served at `/docs/prompting`
> with a copy button; the planner's take brief carries the night's rules
> (a scene change or a reveal is a beat of its own, a settle before it); H3
> native exports are no longer shrunk to 720p by default.
> **Same morning, continuity.** The Commuter turned from night to grey day
> inside part 2 and back. `take_light_lock` reads the time of day and
> weather out of the prompt and appends one continuity sentence to every
> beat (both engines); `take_drift` measures mean luma first-vs-last frame
> per part and the runner retakes a drifting part once (lock doubled, seed
> +101), keeping the steadier clip and hiding the other; the planner brief
> states the light once and forbids a beat that implies another hour.

> **🎬 2026-09-05 — Editor v2 on dev/beta, UNRELEASED: speed, titles, transitions, the Director, sliding windows.**
> From the long-form editing brief (the measured gaps only, built our way).
> **Transitions** are a typed object on a BOUNDARY (`transitions[]`,
> `after_clip`), never a picture overlap: the clips' slots do not move, the
> film stays the timeline's length, the render pulls half the duration of
> source handle either side of the cut and joins the two concat runs with
> `xfade` centred on it; the sound takes the J-cut lane path untouched. No
> handles → refused with the side and the shortfall named; every code is an
> error, `WARNING_CODES` untouched. Even-frame quantised (0.8 s came out one
> frame long before). **Speed** on the clip (0.25–4x, never automatic):
> `setpts` + chained `atempo`, `(end-start)/speed` everywhere, envelopes on
> the strip's PLAYED clock (decided once in `audio_gain_points`). **Titles**
> are overlays with `kind: "text"`, rasterised with Pillow from an
> explicitly resolved font FILE into the same overlay chain an uploaded card
> takes — not `drawtext`, which the Homebrew ffmpeg this panel resolves on
> the owner's Mac does not carry; DOM preview at the same anchor/size. UI:
> a mark on every cut → inspector; Add title beside Add black; Speed in the
> Clip section. **The Director**: a soundtrack on the storyboard brief →
> `beat_map` slots → the planner writes one movement per slot (lead with
> the move; no dialogue) → shots get slot+1 s → the Editor opens the film
> cut on the downbeats under the track (`_sb_director_grid`,
> `_sbe_auto_edit` fallback). **Sliding windows** for LTX (`ltx_windows.py`
> — stride/count arithmetic over LTX windows, our code): third answer on the Long
> clips row, one `generate` + N `extend` on the kept tail with one prompt
> per window and re-injected invariants, trimmed to length; needs Q8, refused
> with Extend's own sentences otherwise. Markers deliberately not built.
> Proven: `smoke_dissolve.mp4` (2x clip + dissolve + boxed title, 5.00 s,
> real ffmpeg) and 1456 tests green incl. `test_editor_v2`,
> `test_director`, `test_ltx_windows`; fast gates green. Not carried to the
> NLE export: transitions (butt join) and titles (no path); speed rides as
> in/out vs start/end. A real windows render has NOT run (GPU, Q8) — the
> chain is exercised with a mocked helper only.
> **Same day, the next gap: the SONG MAP.** `storyboard_edit.song_map`
> — per-bar RMS + spectral centroid on the fitted downbeat grid, section
> boundaries at the peaks of a 4-bar level jump (non-maximum suppressed),
> labels by position and relative energy — and `director_pacing` (chorus
> cuts 2x as often, intro/outro half). The Director's slots follow it and
> the brief names the arc ("shots 3–10 are the chorus, peak energy");
> the Editor ruler paints the sections. Measured on AMOR FATI: 126.7 bpm,
> intro 0–37 s (0.39), chorus 37–67 s (0.94), then verses, in 2.1 s. numpy
> only — no librosa, no whisper. Labels are a heuristic and are said to be.
> **Same day, two more gaps:** **Retake** — the Editor's inspector
> sends a clip back through the renderer (`edit/generate` with
> `retake_of`, which clones the clip's own shot and changes only prompt /
> length / seed); the finished take comes back flagged in the relink rows
> and is adopted per clip (`relink` with `only`) with "Use it / Keep the
> old one" — never the batch drafts→finals rewrite. **Completion alerts**
> — `notify_done` (default on): a Web-Audio chime in the tab on done or
> failed, a browser Notification when the tab is hidden and allowed;
> Settings row with the permission ask. The poller keys on history ids and
> the first poll only records what is already done.
> **Deliver as** on the Editor's Render menu: H.264 / HEVC (VideoToolbox,
> hvc1) / ProRes 422 HQ (.mov, 10-bit 4:2:2, PCM) × as cut / 1080p / 4K
> (one Lanczos scale after the overlays, up only). Different delivery,
> different file name; the films list shows .mov. Proven with real
> encodes of the v2 smoke doc (test_deliver + ffprobe).
> **Duplicate** (Clip section, key D) and **gallery search**: each output
> row carries `q` (sidecar words — prompt, mode, quality, engine, WxH,
> frames, seed, LoRA stems, model, character); the Outputs head has a
> search box, every word must match, typing pulls the older outputs in.
> **Framing** (`clip.frame = {zoom, x, y}`): crop of the source's own
> pixels before the fit in the render, CSS scale on the stage, Basic
> Motion / AE scale+position in the exports. Inspector: Effects → Zoom,
> Across, Down. Two UX-agent passes over the day's work (17 + 8 findings)
> all applied: stale stage after a title delete, hidden per-window fields,
> a lagging windows hint, ruler bands too small to read, a search title
> that went stale on clear, an unnamed Duplicate, "Fill this hole" on a
> retake — each fixed and re-verified on the :8240 test panel.
> **Finish** on Deliver as (clean / grain / heavy grain — `noise` t+u after
> the size, delivery only, named in the file). **Auto** on the storyboard
> brief: plan → every shot renders (`_sb_auto_after_plan` once the planner
> gives the memory back) → cut on the beat → film assembled
> (`_sb_auto_film`), each step the existing one; a shot that failed leaves
> the film waiting and says so. A **real sliding-windows render** ran on
> the :8240 panel with both GPU locks taken (the stale 4-day-old ones
> overridden and announced): 233 f at 512×256 quick, window 1 in 42 s,
> window 2 an extend on the Q8 dev transformer with its own prompt.
> Done in 5 min 04 s: 233 delivered frames (9.7 s), continuity held across
> the seam at 5.0 s and window 2 played its own line (`windows_proof.mp4`,
> sidecar `windows` block). Locks released after.
> **Same day, the last eight gaps from the brief, our way:**
> **Anchor stills** — a brief switch; before a text/character shot renders,
> an ordinary image job makes its first frame (`still_prompt` drops the
> camera-move clauses; the character's sheet is the reference through
> `qwen_edit_inline` when it exists), the render thread waits for the
> batch, `_sb_reconcile` folds it back as `shot.still`, and the clip
> renders as **i2v · anchor** from it; a failed still is written on the
> card and the shot renders unanchored once. **Long windows** — a second
> switch: an LTX shot over 121 frames becomes `temporal_mode=windows` with
> style + location as invariants (an anchored long shot is anchored by its
> first window only — extend cannot re-inject an image). **LoRA guides**
> (`POST /loras/guide`, planner-written, kept in the sidecar, released
> after) and **LoRA update checks** (`GET /loras/updates` vs
> `modelVersions[0]`; the install is the ordinary `/civitai/download`).
> **Closed-tab push** — Web Push with a panel-generated VAPID pair, `/sw.js`
> at the root scope, `/push/*`; every done/failed job pushes its label off
> the GPU lock while Completion alerts are on; `pywebpush` on both install
> lanes, the button appears only when it imports. **Light theme**
> (Settings → Appearance, a token block on `html[data-theme="light"]`) and a
> live "N left" on the Storyboard run bar. Reference Edit already covers
> image editing (Qwen-Image-Edit engines); inpaint/outpaint would need a
> mask model and were not built. `test_anchor_stills` pins it; fast gates
> green. A UX review pass then fixed fifteen findings: the still phase is
> visible while it runs (card placeholder + "Still for shot N" on the run
> bar), a **New still** button per card (`POST /storyboard/restill`),
> stills skipped for H3 shots, the three brief switches read back from the
> board and carry a "?" note, the long-shot switch is disabled without Q8,
> the light palette is stamped on `<html>` from a cookie before first paint
> and its hard-coded dark surfaces are overridden, the invented CSS tokens
> are real ones, the LoRA Update badge is a button with a download icon and
> the check is remembered per browser, guides have a busy state and are
> disabled while a render runs, and Completion alerts say "closed-tab
> alerts" instead of "push". Proven on :8240: still (FLUX, 13 s) → i2v
> anchor (41 s), frame 0 = the still; `/loras/updates` found one real
> update (EditAnything → LTX 2.5 v2.0); `/loras/guide` wrote a real guide
> (6.5 s).
> **Same day, ONE TAKE.** A 60 s H3 ride (a hen on a skateboard through
> twelve city environments, camera behind her) was first made by a script
> driving four 15 s jobs, each from the last frame of the one before. That
> script is now the panel: the Video tab's "One take" row (30 s … 2 min) with
> one beat per 5 s, prefilled from the prompt; `take_plan` / `take_beats` /
> `take_estimate_minutes` in the panel; on LTX it is the windows chain, on H3
> `run_take_job_inner` runs ordinary H3 renders per part and joins them; the
> engine's own length pills lock while a take is on; `/take/estimate` prices
> it (a minute at H3 high ≈ 3 h 50 on this Mac). No user copy says "windows",
> "chain" or "passes". `test_take` pins the arithmetic, the make_job mapping
> and the runner with a stubbed engine. The Storyboard has the same door:
> a fifth chip on the Shots row, **1 · one take**, with a length; the planner
> writes beats instead of shots, the board keeps one shot with the beats
> (editable on the card), and it renders as the same take.
> **Same day, the timeline learns the NLE contract.** Dragging a clip's body
> moves that clip alone, between its neighbours, and stops at them; pulling an
> edge changes that clip's length and leaves everything after it where it was
> (a hole opens or closes). The old behaviour — every gesture repacked the
> sequence and slid the whole tail — is a RIPPLE now: hold ⌘ (or Ctrl) while
> dragging (a badge on the track says so); Shift still reorders. An edge is a
> trim anywhere within 10 px of it. Drags re-apply from the pointerdown
> snapshot so a clamped clip cannot creep through a neighbour. Seven timeline
> scenarios that meant "slide the rest" now pass the ripple flag; two new ones
> pin the defaults.
> **Same evening, the windows chain runs in constant time.** Measured on a
> 30 s LTX take at 640×448: window 2 took 10 min, window 3 17, window 4 26 —
> each extend was handed the WHOLE clip so far and Extend encodes and
> conditions on every frame it is given. `_run_windows_chain` now cuts the
> last window (121 f, after the plan's discard) as the context for each
> extend, keeps only the new frames of each pass as a piece, and joins the
> pieces at the end (frame-exact `select` filters; the exact strings are
> proven on a real clip in the test and by hand). A one-minute ride on H3
> (four 15 s parts from each other's last frame) took 3 h 53 at 1024×576.
> **Same evening, Turbo moves to the community's adapter.** larryvrh's
> **v4 step-600 EMA** (780 MB, bare runner layout, sha `5f3a626c…a416d3`)
> resolves FIRST in `H3_TURBO_LORA_CANDIDATES`; the LightX2V v1.0 repack,
> the folded v0.1 and ckpt500 stay as fallbacks so no install loses Turbo.
> Steps follow the adapter (`h3_turbo_steps`: 7 sigma points = 6 forwards for
> v4, the card's sweet spot — 4 smears fast motion; 4 points for the 4-step
> adapters), and `_h3_retune_turbo_estimates` re-prices every tier cell for
> the installed adapter after the resolver exists (high_15s Turbo ~25 → ~44
> min, native_15s ~57 min → ~1 h 44; honest, not optimistic). The managed
> download fetches v4 from its author's repo, digest-pinned
> (`H3_TURBO_ASSETS` / `_h3_turbo_asset`); the release asset stays reachable
> by key. `test_h3_turbo_adapter` 13. A/B against exact and the old adapter
> pending on the test panel.

> **🚨 2026-09-06 — v4.9.8: H3 install/update broken for everyone by a deleted HF repo — fixed.**
> `madebyollin/taeh3` on Hugging Face is gone (401/404); install_h3.js fetched
> the 22 MB TAE draft decoder from it, so every H3 install and "Update Hailuo
> H3" died at that step (Pinokio, @macstephen). New
> `scripts/pinokio/h3_fetch_tae.py`: pinned GitHub commit of
> madebyollin/taehv, size + sha256 verified, atomic, idempotent, HF fallback.
> The LTX live-preview TAE was never at risk (mirrored on weights-ltx25-v1).
> Lesson: any third-party download in the install path must be mirrored or
> hash-pinned — HF repos vanish. #62: Piotr's clean-folder run still shows no
> identity; sidecar shows base = LTX-2.3 Q8 (by design — trainer is 2.3-based
> and adapters transfer, the sample character proves it); asked him for the
> adapter file to test here.

> **📈 2026-09-05 — v4.9.7: the fleet picture gets its blind spots filled (analytics only).**
> `source` on every render event; `feature_used` (storyboard_plan/export,
> editor_open/export, civitai_download, sample_character, train_start);
> `app_updated`; `update_prompt` (shown/later/update_now/banner_*/
> restart_needed); `broadcast_seen`; `queue_paused_breaker`. One browser
> route `POST /analytics/ui` with a strict allowlist. Validated offline on a
> scratch panel; 59 analytics tests. The 12 HogQL tile queries for the
> PostHog "Phosphene Fleet" board are in docs/ANALYTICS.md — the panel's
> query key lacks `insight:write`/`dashboard:write`, so adding them to the
> board needs a scoped key (owner) or a paste per tile. Stats page: weights
> release shows complete pack downloads (~1.2K), not the 84K per-file sum.
> NOTE: promoted by cherry-pick onto origin/main — dev/beta also carry the
> other window's unreleased UX batch + Director (835b2c5, 14592a8…).

> **🩹 2026-09-05 — v4.9.6: H3 sidecars follow the clip to Trash (#77).**
> `post_output_delete._expand_for_media` now collects `<stem>.wav`,
> `<stem>_source.wav`, `<stem>.stage_a.npz` (+ bare `.stage_a`). Validated on
> a scratch panel: H3 clip → 5 files trashed, LTX clip → the same 2 as
> before, an unrelated .wav untouched. #76 closed (reporter confirmed the
> Update path). Pinokio posts for 4.9.4→4.9.6 still pending (owner login).

> **🩹 2026-09-05 — v4.9.5: character training actually re-trains (#62 cache), H3 shutdown abort (#76).**
> Both validated end-to-end before promote: real Gemma preprocess twice on
> a 3-image set — one changed caption re-encoded exactly that one file,
> unchanged inputs reused; stub H3 engine that writes the clip then aborts
> → job done + log line, stub that aborts before writing → still failed.
> Full gates green. Owner rule from today: bug fixes ship same day, validated
> (memory feedback_ship_bugfixes_validated).

> **🩹 2026-09-05 — after v4.9.4: two more on dev/beta (fe31e41), UNRELEASED.**
> **#76 (PhantombrainM):** H3 helper aborted at interpreter shutdown AFTER
> the MP4 was written (mlx#4248 stream teardown → PyThreadState_Get /
> SIGABRT); panel called finished renders failures. Engine fix pushed to
> minimax-h3-mlx `codex/h3-engine-v2` 79c252b (+ cherry-picked to the
> owner's `codex/live-preview` 7cdcf99): guarded `atexit.register(
> mx.clear_streams)`. Panel guard `_h3_clip_is_complete` keeps a whole clip
> (mtime/size/ffprobe) when the helper exits non-zero. **#62 root cause
> candidate FOUND on our side:** the vendored preprocess skips cached
> conditions/latents by INDEX FILENAME only — a re-train from the same
> folder with a new trigger trained on the OLD captions, a dropped photo
> misaligned every later latent. `lora_lab/preprocess_images.py` now
> writes `.precomputed/manifest.json` and invalidates exactly what changed
> (`_reconcile_precomputed`, 5 tests). Asked PiotrAstroCamp for one
> fresh-folder run to confirm (his sample-character test proved his
> render path is fine). Fleet 12 h after 4.9.4: 35 installs on it, no new
> failure class; the "[Errno 2]" venv_broken signature is pre-existing.
> Pinokio post for 4.9.4 still NOT made (Chrome logged out of pinokio.co).

> **🩹 2026-09-04 — v4.9.4: the four follow-ups from 4.9.3's first day, shipped.**
> Queue circuit breaker (3 identical failures → pause + why), H3 failures
> carry their last engine line, stage step counts capped at the checkpoint's
> tables, dead HiDream image setting falls back. Fleet-driven, all verified
> on scratch panels / tests; full gates green. Promoted by the snapshot ritual.

> **📈 2026-09-04 — first 12 h of v4.9.3 (UNRELEASED follow-ups on dev/beta: b1ffb4b).**
> 40 installs on 4.9.3 within 12 h, 15 new installs. Its raw failure count
> (394) is ONE 16 GB install with an incomplete Q4 pack failing 371 times
> (227 in ten minutes; it had failed 479× on 4.6.0) — not a regression. Fix
> on dev: the queue pauses itself after three identical failures in a row
> (`_CONSEC_FAIL` in worker_loop, verified on a scratch panel). H3: 20
> failures / 3 installs all read "exited with code 1 — see the log above";
> per-install history shows those installs failed on older versions too
> and an old-branch install (cf14779a) renders fine after the #74 branch
> move — no sign the move broke anything. Fix on dev: the H3 failure now
> carries the last engine line. Also fixed on dev (same day): explicit stage step counts are capped at
> what the checkpoint's own tables hold (`_clamp_stage_steps_to_tables`,
> the "cannot thin … up to 12 steps" class); a saved HiDream image engine
> with no venv promotes to an installed mflux family instead of failing
> every Studio render. Pinokio: no replies to the 4.9.3 post or from @vxlab
> yet; no new threads. GitHub: no new activity on any open issue.

> **🩹 2026-09-03 — v4.9.3: the full-review fixes, shipped as a plain bugfix.**
> Everything in the 09-02 review entry below, plus #74 (Update now moves the
> H3 engine checkout to `codex/h3-engine-v2` via the new shared
> `scripts/pinokio/h3_checkout.sh` before building the Q8 engine — diagnosed
> by @blackest). Image-mode identity on the 24 GB lane was REPRODUCED AS
> WORKING (Q4 · Balanced · Anchor, seed 4242: frame one = the reference, the
> character holds to the last frame) — the Pinokio report from @vxlab is not
> a code defect on the default path; answered with the Anchor/Inspire and
> train-a-character guidance. Gates: full 58/0/0 including the codec check on
> that clip. Promoted by the snapshot ritual.

> **🔎 2026-09-02 — FULL REVIEW after the fast releases (4.9.0→4.9.2): fix wave on dev/beta, UNRELEASED (dcb2589).**
> Four audit lanes + fleet analytics. **Verdicts:** the module split is sound
> (634 publishes checked, zero order hazards) and the route move is
> behaviour-identical (2,568 probes old vs new); update from 4.8.2/4.6.0 and
> fresh install both rehearsed clean. **What shipped broken and is now fixed
> on beta:** (1) Storyboard on a 768-cap Mac — every NEW film was born with an
> illegal 1024×576 delivery (clamp never ran for a fresh board), Render dead
> from the first click, fix button offered the same illegal size, Quality
> section crushed to 2px at Pinokio width, dropdowns replaced by the poll
> (#71 follow-up; `storyboard.fit_canvas`, `test_storyboard_capfit.py`);
> (2) character rename always failed (route split lost one body-read
> preamble); (3) trained character "Use in video" never sent the LoRA
> (called a serialiser that never existed); (4) Image Studio offered 8/24 GB
> Macs engines that can't fit — 15/16 "unclassified" failures on 4.9.0 —
> now disabled with "needs a 32 GB Mac", Studio moves to Auto, guard is a
> refusal (`image_ram`); (5) High/Keyframes/Extend without the Q8 pack →
> refusal (`pack_missing`) pointing at Settings → Models, not a CLI line;
> (6) Image mode with no image stopped at the button; (7) header memory
> readout clipped at 1300px; (8) in-app Update now skipped post_update.sh
> when only it changed + self-repair probe covered 1 of 3 packages +
> production serves boot-time webapp snapshot; (9) classifier: helper_exit
> moved below model_missing/disk_full (was swallowing ~500 events/wk),
> stuck partial downloads bounded. **Gate added:** lint_webapp fails on a
> handler calling a function nothing declares. Gates: full 58/0/0, --fast
> 56/0/2 on the final tree.
> **Fleet (3 days):** 501 active installs/wk, 93 new on 09-01; 4.9.0 fail
> rate 9.6% (4.8.1 was 23%); **43% of active installs rendered nothing all
> week** — activation is the product problem, not stability. 22% of the
> fleet has ≤24 GB. i2v > t2v; characters used in ~4% of renders; no events
> for Storyboard/Editor use at all (instrumentation gap).
> **Open:** Pinokio has NO Official Update post for 4.9.0/4.9.1/4.9.2 (rule:
> post on every release); Pinokio question from @vxlab (24 GB, i2v loses the
> reference face) unanswered; Lane B P1 — keyframe/audio/LoRA "not found"
> gates lack the plain-English remedy the i2v gate has; Lane A — 3 function
> names reached only through strings (`install_card` table, Python-emitted
> `openModelsModal`, `${cta.run}`) that the lint cannot see.
> **Process lesson:** two review agents ran `pkill -f mlx_ltx_panel.py` and
> killed the owner's 8199 panel — briefs must mandate PID kills (memory
> `feedback_agents_never_pkill_by_pattern`).

> **🩹 2026-09-02 — v4.9.2: three issue fixes (#73, #61, #62), shipped the same day as v4.9.1.**
> **#73 — an interrupted image-engine download looked ready and crashed every
> render.** Qwen-Image-Edit-2511 stopped at 38 of 54 GB with 14 blobs still
> `.incomplete`; `/image/engine_status` said cached; mflux resolves a Hub id
> cached-first and its completeness test only asks whether every file pattern
> has SOME match, so it loaded the shards it had and died in its weight loader
> without ever calling the one function that resumes. Now: `image_engine.
> hf_repo_partial_download()` reads the one unambiguous sign huggingface_hub
> leaves (`blobs/*.incomplete`); the status route reports `partial` and stops
> saying cached (pill "resume · N of ~M GB", button "Generate · resumes the
> download first"); and a pre-flight before every mflux spawn removes the
> `snapshots/` symlink tree so mflux's own download rule runs, re-links every
> finished blob and resumes the rest — nothing downloads twice. The error the
> panel showed was the HEAD of the traceback (`stderr_tail[:1200]`), cut off
> right before the exception; both slices now keep the END. `test_image_
> partial_download.py`. **#61 — training dies when the GPU watchdog kills the
> caption encode.** The render helper's mitigation (retry once at a shorter
> Gemma pad) is ported: the panel places the kill from the trainer's own
> phase banners, relaunches the trainer ONCE for a SIGABRT-with-signature in
> `text_encode` with `LTX2_GEMMA_MAX_LENGTH=256` (honoured by a wrapper in
> `lora_lab/preprocess_images.py` around the vendored `encode_all_layers`,
> default 1024), wipes the half-written caption conditions so every caption
> is encoded at one length, and leaves the image latents (the expensive
> half) for the preprocessor to skip. Kills in other phases keep the canvas
> advice. `test_train_encode_retry.py`. **#62 — the trigger the panel
> suggested was half digits.** `mrz07` tokenizes through Gemma as `m / rz /
> 0 / 7`; every trigger that has ever carried a face here is letters-only
> (`bizarrotrn` → `b / izarro / trn`, `elontrn`, `ariatrn`), and the rank-32
> High trainings reported as "active but no identity" all used the digit
> shape (`mmx26`, `sfw25`, `3Mar26`). Unproven as THE cause — it is being
> A/B'd with the reporter — but suggesting the shape known to work costs
> nothing: `_suggest_trigger_token()` and its JS mirror now emit
> `<3 consonants>trn`, the Train tab warns (never blocks) on a digit, the
> trainer logs the same note, API.md says letters-only.

> **🩹 2026-09-02 — v4.9.1: the buttons v4.9.0 broke, fixed the same day they were reported.**
> The module split had one blind spot: a function referenced ONLY from its own
> module's generated markup (`onclick="deleteOutput('…')"` inside a template
> literal) looked module-internal to the extraction analysis and went
> unpublished — but inline attributes resolve against the GLOBAL scope at click
> time. 38 handlers died silently: gallery card actions, LoRA management,
> training controls, timeline editing, CivitAI downloads, storyboard board
> actions. @blackest reported it, diagnosed it and sent PR #69 within a day
> (thank you); the shipped fix is the union of the PR's list and an
> independent scan (which also caught `sbeLiftSelected`), folded into each
> module's one publish block. **The class is now a build failure:**
> `scripts/lint_webapp.mjs` scans every `on*=`/`javascript:`/`setAttribute('on…')`
> string in the page and the modules and refuses any call to an unpublished
> top-level function — proven red on the v4.9.0 tree, green on this one. Also
> in: a per-thumbnail "×" on the Recent-uploads strip (`POST /upload/delete`,
> path-bound, clears the thumbnail cache — a Pinokio ask from @le_wib). Riding
> along from a parallel session: Editor Delete now LIFTS (leaves a slug) and
> Shift+Delete ripples (00c248d); the theater's mute is remembered across
> clips and restarts (a157b9b, Pinokio report by fuschichou); Update proves
> the render packages import and self-repairs once before claiming success
> (3698c53 — the fleet's second-commonest error).

> **🧱 2026-09-01 — v4.9.0: the frontend and the routes leave the monolith.**
> Shipped to public `main` the same day, at the owner's explicit green light,
> with the residual risks named below accepted ("we will discover if there is
> something serious on the way"). Nothing user-visible changes in this release
> — same screens, same renders, byte-for-byte — but every future fix lands in
> a small named file instead of a 72k-line monolith.
> `mlx_ltx_panel.py` carried the whole product as one 72,542-line file — the
> shape that shipped the built-twice bug class. On `dev` (beta), in ~20 sliced
> commits (a0ee1ff..a9c5f66), each gated green before the next:
>
> * **Slice 1 — CSS** → `webapp/style/panel.css`, served by a path-bound
>   `/webapp/` route with `__ENGINE_RULES__` substituted at serve time;
>   byte-verified identical page. `test_panel_assets.py` pins it.
> * **Slice 2 — the page** → `webapp/index.html`, read once at import,
>   `page()` substitutions unchanged; byte-verified modulo the build SHA.
> * **Slice 3 — the JS** → 12 ES modules under `webapp/js/` (boot, stage,
>   characters, engines, queue, settings, loras, preview, health, storyboard,
>   editor, main), extracted bottom-up so execution order never changed; each
>   module publishes its outside-facing surface explicitly
>   (`Object.assign(globalThis, …)`), `main.js` deliberately loads LAST so the
>   kickoffs keep the ordering guarantee hoisting used to give. The page's
>   inline script is ONE line: `const BOOT = __BOOTSTRAP__;`.
>   `scripts/lint_webapp.mjs` (eslint, dev-only, in release_gates) enforces
>   no-undef/no-redeclare over the real scope model + a cross-file
>   duplicate-publish check.
> * **Slice 4 — the routes**: all 101 routes moved from the do_GET/do_POST
>   if/elif chains into `panel/routes_*.py` (stats, meta, models, loras,
>   train, files, queue, characters, storyboard, image) — exact paths in
>   `panel.routes`' tables, startswith/endswith families in ordered pattern
>   lists; registration refuses duplicates; the dispatchers are ~35 lines.
>   `test_routes.py` refuses a chain arm outright and pins the pattern order.
>   The chains' hidden body-parse coupling became `Handler._read_form_body()`.
> * `mlx_ltx_panel.py` is now ~24.5k lines of pure server; the panel was
>   re-verified in a live browser after every module and every route family
>   (all seven surfaces, poll streaming, /run round-trip under a paused queue).
>   `docs/ARCHITECTURE.md` is the authority on what lives where and where new
>   code goes; CLAUDE.md points at it.
>
> **Residual risk accepted at ship time:** the from-zero install with real
> weight downloads and a real prior-install Update click were rehearsed
> structurally (fresh clone boots and serves everything; a v4.8.2 checkout
> fast-forwards and boots) but not performed on a clean physical machine;
> this release adds NO new weight-pack requirements, so the update's weights
> half is vacuous by construction.
>
> **NEXT (in order):** (1) migrate the remaining extraction-based suites to
> import the real modules via `scripts/webapp_import_shim.mjs` — the pattern
> is proven on `test_h3_lora_import_ui`; the shared node harness behind
> `test_storyboard_editor_ui` / `test_editor_mix` / `test_character_roundtrip`
> / `test_stage_live_preview` / `test_spicy_contract` is one coherent package,
> and `scripts/extract_panel_js.py` is deleted only when the last one moves.
> (2) The per-module `Object.assign` publish lists can shrink as those
> migrations land. (3) An end-to-end render smoke on the dev panel before any
> promote (none of this is released; v4.8.2 remains the public tip).

> **🩹 2026-09-01 — v4.8.2: a bug-fix release, nothing else.** Thirteen fixes from a full-codebase review driven by fleet analytics and open issues. What users stop suffering: "I updated and nothing changed" cases from a status chip that had been built twice and half-worked by luck; renders that died with a meaningless code when the Mac ran out of memory now say so and say what to do; a render that silently hung forever now stops itself after 20 minutes with an honest message; the "H3 needs repairing" card stops blaming the wrong thing and shows exactly what is missing (#68); Macs with 36-45 GB stop being told to install H3 by one screen and refused by another; weights moved to an external drive stop triggering a phantom 75 GB re-download; Update now repairs a broken Python environment instead of failing with a red banner; an old hidden speed control restored from ancient renders no longer crashes new ones; the log pane stops un-selecting text while users try to copy an error; character training sidecars stop lying about what they trained on, widescreen training actually trains widescreen, and the adapter-strength check stops ranking working character files below broken ones (#62 collateral). Detail per fix in the commit log (989aaf1..8db82d9).
>
> **Gates at the promote tree:** scripts/release_gates.sh full run — 50 PASS / 0 FAIL (the runner itself is new in this release and caught one of this release's own bugs before commit).

> **🚚 2026-08-31 — v4.8.1 PUBLIC (`d790203`): the update that actually
> delivers the memory fix.** v4.8.0 promised H3 at half the memory and could not
> keep it on any install that already had H3: the compact Q8 engine was never
> built on 64 GB+ Macs, and **an update ships code, not weights** — so H3 kept
> rendering with the full bf16 engine at ~40–48 GB and nothing said why. Update
> now builds the compact engine itself when H3 is installed (one time, ~5 min,
> ~22 GB, safe to repeat); measured on the field scenario, DiT load **38.56 →
> 20.22 GiB**, denoise peak 21.26 GiB, no wall-clock cost. Silent fallbacks are
> gone — the render log names the reason in plain words and `/status` reports it
> as data. Also: the a2v **Audio conditioning strength** slider is connected to a
> control that exists (silently dropped on the Q8 path since it shipped), and
> Storyboard treats **sung shots as first-class** (lyric in the dialogue form,
> budgeted at a measured singing tempo, trained voice carried; a wordless "she
> sings" is caught like a wordless "he explains"). Settings → Hailuo H3 model →
> Full still forces the master.

> **🪶 2026-08-29 — v4.8.0 PUBLIC (`f97ea3c`): Hailuo H3 uses half the memory,
> and runs on 36 GB Macs.** H3 handed the bf16 master to any Mac ≥60 GB because
> its peak fits — it fits and leaves nothing behind it. Measured at 640×432 / 73
> frames, one prompt one seed: bf16 38.56 GiB resident / 38.89 peak / 196.9 s vs
> **Q8 20.22 / 21.38 / 200.1 s — 45% less memory for 1.6% more time**. The older
> "~12% slower" note that protected the heavy default was seven times worse than
> reality. `h3_dit` had been honoured by the engine and validated by the settings
> handler for months with **no control ever built** (and it was write-only, so
> any control would have opened on "Automatic"); it is now a Settings row
> carrying the measured numbers. The runner's `--memory-gb 58` clamp was leaving
> 32/48 GB Macs **1.0 GB of their own machine** — every size now keeps ≥6 GB, for
> free, since H3's real peak is 25.63 GiB. **Q8 floor 46 GB → 36 GB**, verified
> byte-identical at a simulated 36 GB budget; 32 GB stays deliberately excluded.
> Prompt embeddings are cached instead of re-encoded through a 26.28 GB text
> encoder every render.

> **🧠 2026-08-29 — v4.7.0 PUBLIC (`a9e7848`): the render that did not fit,
> fits.** A third of every render's peak was the MLX allocator's cache and
> nothing capped it. Capped now: a Q4 768×432 121-frame render went **25.53 GB →
> 16.50 GB** peak on a 64 GB machine and got *faster* (71.45 → 68.84 s), with
> **sha256-identical frames** — not a quality trade, memory that was never doing
> any work. On a 16 GB Mac the same render moved from 101% of RAM to 96%: the
> difference between swapping and running. The cap sizes itself from the machine
> (1/8 of physical RAM, floor 2 GiB, ceiling 8 GiB), override `LTX_MLX_CACHE_GIB`.
> The Remix engine is renamed **Motion Control** after two separate people asked
> us to build the feature it already was. **H3 LoRA import from a file** landed
> via PR #65 (@sahilkashyap64), reading a converter's own alpha metadata instead
> of assuming 1.0, and costing +2.7 MB of RAM instead of +9.5 GB. Fixes: the
> default training preset could not carry a face (#62); i2v could substitute a
> reference image you never chose; and **updates land on time** — Pinokio reads
> `update.js` into memory *then* runs it, so an update that fixed the updater
> always arrived one click late; `update.js` is now thin and delegates to a file
> read after the pull. 2,830 tests, 23 skipped.

> **🤝 2026-08-29 — PR #65 lands: "Import H3 LoRA" is a real control, after
> three fixes. Contributed by @sahilkashyap64, who also filed the issue (#64)
> it closes. On `dev`.**
>
> His endpoint and staging design are what shipped: a dedicated
> `POST /h3/loras/import` (not `/upload` — an adapter has to pass the runner's
> layout contract before it may appear in the picker), `open("xb")` + `fsync` +
> `os.replace`, and an `H3_LORA_IMPORT_LOCK` held across the whole
> stage → validate → publish sequence so two requests cannot race the same
> name. Path handling held under every probe thrown at it — traversal,
> absolute paths, a symlink swapped mid-import, a header declaring 2^63 bytes,
> eight concurrent imports of one filename. He also fixed a genuine latent bug
> in passing: **`import struct` was missing on `dev`**, and
> `_h3_lora_effective_alpha` has been unreachable dead code since it was
> written — his PR is the first caller it ever had.
>
> **Three things were fixed on top of his commits before merge.**
>
> **1. The new gates ran on the RENDER path.** `_h3_lora_prepare` is called at
> CivitAI install AND at render dispatch, not only by the importer. The PR
> inserted the payload/target-module validation BEFORE `_h3_lora_strip_prefix`,
> so keys still carried `diffusion_model.` while the DiT's stems are bare —
> every module "missing". Measured against the 14 real H3 adapters on this
> machine: **five that pass on `dev` were refused with the patch** (lightx2v
> v1.1 comfyui, both Kijai repacks, drbaph, lightx2v v0.1 comfy), and the
> CivitAI path DELETES the file after a refusal. The repo's own note above
> `H3_LORAS_DIRNAME` says most H3 LoRAs on CivitAI are ComfyUI repacks. Fixed
> by restoring `_h3_lora_prepare` byte-identical to `dev` and moving the new
> checks into `_h3_lora_validate_for_import`, which the importer calls AFTER
> prepare returns, when the keys are bare. **No fixture in the suite used a
> prefixed key** — which is exactly why CI was green while the real files were
> broken; `_prefixed_lora()` is now the regression gate.
>
> **2. The non-unit alpha/rank refusal rejected correct adapters on an
> unreliable number.** Kijai's 8-step (metadata `alpha: "8"`,
> `baked_scale: "0.0625"`) computed "alpha/rank 4"; his 4-step (`alpha: "128"`,
> `baked_scale: "1.0"`) computed "21.33". Both files state the scale is already
> folded into `lora_B`. The numbers came from `_h3_lora_effective_alpha`
> pairing one metadata alpha with whatever module `for k in header` reached
> first — and those two files carry 88 and 72 DISTINCT per-module ranks (2–173),
> so the answer was dict-order dependent. It also contradicted this project's
> own shipped position, that alpha != rank is normal and lightx2v's 8-step
> loads at **0.0625**. Now: never refused. `_h3_lora_scale_report` believes
> per-module `.alpha` scalars and the three folded-scale markers the real files
> carry (`baked_scale`, `peft_scale_folded_into_B`, `training_scale`), divides
> a bare metadata alpha only when the rank is unambiguous, and otherwise says
> so; the result is written to the sidecar as `recommended_strength` and told
> to the user. The picker already has a per-LoRA strength control, so the false
> refusal became the feature.
>
> **3. The 512 MiB cap excluded the adapters people want, and was
> simultaneously too high for RAM.** larryvrh's turbo v4 — the community
> flagship — is **780 MB** and got a flat 413; drbaph 620 MB; the lightx2v
> adapters 1.38–1.96 GB. Meanwhile `_parse_multipart_form` buffers the body,
> the email parser copies it, `get_payload(decode=True)` copies it again, then
> a BytesIO, then `.read()`. Measured with `ru_maxrss` on this machine: a
> **296 MB import cost +3,982 MB of peak RSS (13.5x)** and a 744 MB one
> **+9,496 MB (12.8x)** — next to H3's ~40 GiB render footprint. The endpoint
> now streams the multipart part straight into the staging file
> (`_stream_multipart_file_part`, 1 MiB window): the same two files cost
> **+3.1 MB and +3.3 MB**. With the amplification gone the cap could be honest:
> **4 GiB**. End to end through the live endpoint, the 780 MB larryvrh file
> imports in **1.7 s for +2.7 MB of panel RSS**.
>
> Also: refusals named the staging temp
> (`.name.22850.6152073216.uploading`) in an `alert()` — staging moved into a
> hidden temp DIRECTORY so the staged file keeps the user's own filename and
> every message names what they picked. `_h3_lora_target_modules` now goes
> through `lora_compat.read_tensor_header` (LRU-cached; it was re-reading the
> ~26 GB DiT's header uncached on every call). All-or-nothing matching became
> the same **0.90** ratio `lora_compat` has always used on the LTX lane. The
> import writes a sidecar like the CivitAI path, so the row has a name and a
> strength. The button is a ghost, not a second `--accent` primary next to
> Browse CivitAI. "(1 module pairs)" agrees with its number.
>
> **Validation.** All 14 real adapters re-run through `_h3_lora_prepare`:
> identical to `dev`, 12 pass / 2 refused (both genuinely diffusers-layout).
> All 12 loadable ones imported through the live HTTP endpoint. `importH3Lora`
> driven in a browser on a real panel — plural copy, the 0.0625 strength
> surfaced, the ComfyUI repack converted, a duplicate refused by the user's own
> filename, a `.zip` refused client-side. Suite **1643 → 1687**, green. His
> eight behavioural tests are kept; `test_h3_lora_import_http.py` (14) covers
> the wire — 411, 413, framing, traversal, the streaming window, and that a
> refused body still leaves the socket — and `test_h3_lora_import_ui.py` (10)
> RUNS `importH3Lora` in node via `scripts/extract_panel_js.py` instead of
> grepping the source for it.


> **⛔ 2026-08-28 — the mlx 0.32 move, tried properly and refused a second
> time. The mitigation worked; it just did not work on 0.32. On `dev`.**
>
> The route the previous pass laid out was (1) ship a cache policy on the
> current pin, (2) move `mlx 0.32.1` + `mflux 0.19.1` in one commit, (3) re-run
> the tier table. **Step 1 shipped and is a real win** (entry below). **Step 3
> then failed step 2, so step 2 was reverted before it was committed.**
>
> Same policy on both versions, same prompt and seed, `/usr/bin/time -l` peak
> memory footprint, and the share of the simulated Mac's physical RAM:
>
> | render | tier (cap) | 0.31.1 today | 0.31.1 + policy | **0.32.1 + policy** |
> |---|---|---|---|---|
> | Q4 768×432 121f | 16 GiB (2 GiB) | 17.41 (101%) | 16.50 (96%) | **19.91 (116%)** ✗ |
> | Q4 768×432 121f | 32 GiB (4 GiB) | 25.66 (75%) | 16.49 (48%) | 23.40 (68%) ✓ |
> | Q4 768×432 121f | 64 GiB (8 GiB) | 25.53 (37%) | 16.50 (24%) | **27.87 (41%)** ✗ |
> | Q8 1024×576 121f | 48 GiB (6 GiB) | 39.49 (77%) | 25.39 (49%) | **42.11 (82%)** ✗ |
>
> **Three of four tiers end at a higher share of RAM than 0.31.1 uses today**,
> and the 48 GiB Comfortable tier — the commonest paying-attention Mac — lands
> at **81.7%**, which is the 82% at which `plan_memory_policy()` forces the
> streamed VAE decode, before Chrome and Slack are counted. That was the stated
> gate. It failed.
>
> **WHY THE CAP CANNOT RESCUE IT, and this is the new fact.** The previous pass
> concluded "the growth is allocator cache". Half right. With the cache capped
> *identically on both versions*, 0.32.1's **active** memory is still
> **+12.0 GB** on Q4 (26.57 vs 14.53) and **+29.2 GB** on Q8 (52.23 vs 23.08).
> No cache policy can reach that. Driving the cache to **zero** does get 0.32.1
> under today's numbers — Q8@48 GiB **32.25 GB / 138.37 s**, Q4@16 GiB
> **17.04 GB / 72.59 s** — and erases the whole speed advantage in the same
> move (−1.1% and −0.5% against 0.31.1 today; *slower* than 0.31.1 with the
> policy). **There is no cache setting at which 0.32.1 beats 0.31.1 + policy on
> both axes.** Step 1 took the win the version move was supposed to deliver,
> and took it on the safe pin.
>
> **FACES: not the blocker.** Q8 1024×576 121f with the trained character LoRA
> (`bizarrotrn_v2` @ 1.0), two seeds, one arm per MLX version. Identity,
> framing, expression, skin and hair detail are indistinguishable; frame-level
> PSNR between the arms is **41.8–43.5 dB** at four matched timestamps, i.e.
> sub-perceptual numerical noise, not a different shot. Audio is clean on both
> (max_volume −3.1 dB, mean −17.6 / −19.6 dB, identical between versions —
> more evidence the 22 dB story is 0.31.2-only). Clips + A/B frame strips +
> a clipgrade manifest: `state/mlx032_faces/`.
>
> **WHAT WAS PROVED AND THEN PUT BACK**, so a future attempt is a pin move plus
> this table rather than a re-derivation:
> - The whole `post_update.sh` sequence, run in order in a copy of the real
>   venv with the new pins, lands on **mlx 0.32.1 / mlx-metal 0.32.1 /
>   mlx-lm 0.31.1 / mflux 0.19.1 / transformers 5.7.0** — the step-7 mflux
>   resolve touches mflux, pillow and torch and leaves mlx alone. The 0.18.0
>   trap (which walks mlx down to **0.31.2**) is real and is what the pin-pair
>   gate exists for; it was negative-tested in both directions again.
> - `patch_mflux_fbcache.py`'s two anchors are present exactly once each in the
>   mflux 0.19.1 wheel, and the patch applied cleanly and idempotently to a
>   real 0.19.1 install with the injected helper importable.
> - mflux 0.19.1's console-script set is a strict superset of 0.18.0's, so no
>   `image_engine.py` dispatch target disappears.
> - `mlx-lm` is not part of the pair: no 0.32.x exists and it declares
>   `mlx>=0.30.4` with no upper bound.
>
> **What would change the answer** is upstream shrinking 0.32.x's active-memory
> appetite — not another knob on our side. Doc-truth fixed in passing: three
> files still said the mflux pin was `0.17.5` when it has been `0.18.0`.

> **🧊 2026-08-28 — Phosphene had no MLX cache policy. It has one now, and it
> takes a third off the peak memory of every render for free. On `dev`.**
>
> MLX does not hand freed Metal buffers back to the driver — it keeps them in an
> allocator cache. The ceiling on that cache is a number about the MACHINE, not
> the job: `0.95 * hw.memsize`, about 61 GB on a 64 GB Mac. Phosphene never set
> one. The single `set_cache_limit(0)` anywhere near this code is in the
> engine's `_base.py`, on the `low_ram_streaming` path the warm helper does not
> enable — a policy nothing calls.
>
> `mlx_warm_helper.py` now caps it at **one eighth of physical RAM**, floored at
> 2 GiB and ceilinged at 8 GiB — so it scales with the Mac the way the
> capability tiers do: 16 GiB → 2, 32 → 4, 48 → 6, 64 and up → 8.
>
> | render, mlx 0.31.1 (the current pin) | today | with the policy | wall |
> |---|---|---|---|
> | Q4 768×432 121f, 64 GiB, cap 8 GiB | 25.53 GB | **16.50 GB** (−35%) | 71.45 → **68.84 s** |
> | Q8 1024×576 121f, 48 GiB, cap 6 GiB | 39.49 GB | **25.39 GB** (−36%) | 139.84 → **138.71 s** |
>
> Both arms are **sha256-identical** to the uncapped render (`144da74a…`,
> `f47d4dc8…`). The cap changes the allocator's bookkeeping, never the
> arithmetic — which is exactly why it could ship on the CURRENT pin, alone,
> and be believed.
>
> **Two details that are the whole difference between a policy and a comment.**
> It is applied at helper start AND re-asserted at the main loop's action choke
> point before every render, because the pipelines' own low-memory paths call
> `set_cache_limit(0)` and never put it back — startup-only would decay to "no
> cache" for the rest of a warm helper's life after one A2V job. And the value
> is a **pure function** (`mlx_cache_limit_bytes`), so `test_mlx_cache_policy.py`
> pins what a 16 GB Mac gets from a 64 GB one — 13 tests, read out of the helper
> with `ast` rather than copied, so a rename fails the suite instead of silently
> testing a duplicate.
>
> Override: `LTX_MLX_CACHE_GIB` — a number in GiB, `0` for no cache at all,
> `off` to restore MLX's machine-sized default.
>
> **Why now:** this is step 1 of the mlx 0.32 route recorded below. It is a
> standalone win on 0.31.1, and it is the thing that makes the version move
> safe — uncapped, 0.32.x fills whatever ceiling it is handed.

> **🧮 2026-08-28 — the mlx pin: the reason it exists is gone, and it stays
> anyway. Measured, not argued. On `dev`.**
>
> The `mlx==0.31.1` pin has carried one sentence since April: *0.31.2
> attenuates the LTX vocoder by 22 dB.* That sentence is **true, and it is
> about 0.31.2 only.** Isolated vocoder path, identical latents in: peak delta
> **0.000001 dB** on 0.32.1. Five full renders per arm: max_volume **-1.7 vs
> -1.8 dB**, integrated loudness **-9.2 vs -9.3 LUFS**, true peak -1.6 vs -1.7,
> zero time shift, spectrum within 0.1 dB from 250 Hz to 16 kHz. **Claimed
> 22 dB, measured 0.1.** The speed is real too — Q8 1024×576 121f goes
> **139.8 s → 133.6 s (-4.5%)**, all of it in DiT load (-28%) and VAE decode
> (-43%), denoise a wash — and the pin is now actively blocking mflux 0.19.1,
> which declares `mlx>=0.32.0`.
>
> **And it does not ship, on two independent counts.**
>
> **1. Peak memory, on every tier — not just this 64 GB box.** MLX derives its
> cache AND memory limits from `0.95 * hw.memsize` on both versions, so the
> tiers were simulated by setting those limits to what each Mac would compute
> and rendering at that tier's real canvas. **0.31.1's footprint is a function
> of the workload; 0.32.x's is a function of the machine** — it takes ~95% of
> whatever ceiling it is handed. `/usr/bin/time -l` peak, same prompt/seed:
>
> | render | simulated Mac | 0.31.1 | 0.32.1 | 0.32.2 | vs physical RAM |
> |---|---|---|---|---|---|
> | Q4 768×432 121f | 16 GB | 17.41 GB | 20.35 GB | 19.35 GB | 101% → 118% |
> | Q4 768×432 121f | 32 GB | 25.66 GB | 33.18 GB | — | 75% → **97%** |
> | Q4 768×432 121f | 64 GB (default) | 25.53 GB | 54.28 GB | — | 37% → 79% |
> | Q8 1024×576 121f | 48 GB | 39.49 GB | 49.95 GB | 50.01 GB | 77% → **97%** |
>
> The 48 GB row is the **Comfortable tier's own compact profile** (max_dim
> 1024) — the most common paying-attention machine — and it goes from roomy to
> at-the-wall. On this box the Q4 render moved system pressure **58% → 84%**
> and added **1.5 GB of swap**; the Q8 one hit **87%**. `plan_memory_policy()`
> calls a machine pressured at **82%** and answers by forcing the streamed VAE
> decode: **~30 s on a 5 s clip**, which is more than the 4.5% the move earns.
> Phosphene would pay for its own speedup twice.
>
> **`mlx==0.32.2` changes nothing here.** 48 GB tier: 50.01 GB vs 0.32.1's
> 49.95 GB. It is ~1 GB better on the 16 GB Q4 case and identical on Q8 — not a
> reason to prefer it, not a reason to avoid it.
>
> **Ruled out, so nobody chases it again:** Metal residency sets, new in 0.32
> (`MLX_RESIDENCY_SET_MAX_PCT`, default 5% of the working set). Setting it to
> **0** reproduces the footprint to within **300 KB** across three runs
> (20.3544 / 20.3546 / 20.3547 GB). The growth is allocator **cache** — and
> capping the cache does not just mitigate it, it **inverts the comparison**.
> Same Q4 render, default machine limits, one `mx.set_cache_limit()` call:
>
> | arm | cache cap | peak footprint | wall |
> |---|---|---|---|
> | 0.31.1 (today) | none | 25.53 GB | 71.45 s |
> | 0.31.1 | 16 GB | 23.15 GB | 71.78 s |
> | 0.32.1 | none | 54.28 GB | 72.45 s |
> | 0.32.1 | 16 GB | 35.40 GB | 67.77 s |
> | **0.32.1** | **8 GB** | **27.74 GB** | **68.24 s** |
> | 0.32.1 | 0 | 17.04 GB | 72.18 s |
>
> **0.32.1 with an 8 GB cache cap is roughly today's memory at -4.5% wall; at
> cache 0 it is a third LESS memory than today** for +1%. And the cap is
> provably free of side effects — every cache setting inside a version is
> **sha256-identical** output (`144da74a…` for all 0.31.1 arms including the
> 16 and 32 GB tier sims, `016d2dd2…` for all 0.32.1 arms). The two versions
> do NOT agree byte-for-byte with each other, which is its own reason the move
> needs an eye on faces and not just a green suite.
>
> **2. The pin cannot move alone — and moving it alone lands users on 0.31.2.**
> `post_update.sh` step 7 (and `mflux_pack.sh`, and `install_qwen.js`) installs
> `mflux==0.18.0` **with deps**, after step 2 pins mlx. mflux 0.18.0 declares
> `mlx>=0.27.0,<0.32.0`, so that later resolve walks mlx back **down**.
> Measured in a copy of the real venv: `uv pip install 'mflux==0.18.0'` into a
> 0.32.2 env proposes `- mlx==0.32.2 / + mlx==0.31.2` — the exact release the
> whole ship-blocker exists to keep off users' machines, arriving silently on
> every fresh install and every Update. Invisible today only because 0.31.1
> sits inside mflux's range.
>
> **Shipped instead:** `scripts/check_post_update.js` gained a **pin-pair
> gate** — mlx and mflux are one decision, asserted across `install.js`,
> `post_update.sh`, `mflux_pack.sh` and `install_qwen.js`, failing the moment
> either moves without the other, with the resolver transcript in the comment.
> Negative-tested both directions. The measurements are recorded in
> `install.js` above the pin and in the mlx row of `CLAUDE.md` §4, so the next
> agent inherits the numbers instead of re-rendering them.
>
> **The route to 0.32, in order:** (1) an explicit MLX **cache-limit policy**
> in `mlx_warm_helper.py` — worth doing on the CURRENT pin, where 0.31.1
> already gives back **39.54 → 23.61 GB** at Q8 for 1–2% wall, and Phosphene
> sets no MLX cache policy at all today (`_base.py`'s `set_cache_limit(0)`
> sits on the `low_ram_streaming` path the warm helper never enables);
> (2) **mflux 0.18.0 → 0.19.1 in the same commit** as the pin — its FBCache
> patch anchors were checked against the 0.19.1 wheel and both survive, so it
> is tractable, but Qwen-Edit / Ideogram output wants an eye; (3) re-run the
> tier table against the capped policy. **`MLX_SDPA_BLOCKS` is not on that
> list:** its win is at head dim 96, and LTX-2.5 uses 128 (video) and 64
> (audio), so there is nothing for it to speed up here.
>
> Not done, deliberately: the cache-limit policy itself. It changes the memory
> behaviour of every render on the daily driver, and the brief for this session
> was "if a tier fails, stop" — a tier failed.

> **🎬 2026-08-28 — a shipped feature two users asked us to build, and a
> refusal quoting a number the product stopped believing. Both on `dev`.**
>
> **1. Union Control was findable by nobody.** It has worked for months —
> official un-gated Lightricks weights, in `required_files.json`, wired to a
> real pipeline — and @sohaibpp on Pinokio ("Motion control") plus a long-time
> user on X ("would love to see IC-LoRA support … exposed in the UI") both
> asked for it as a MISSING feature. Nothing on screen carried the words they
> searched for: the mode was **"Control"**, one click inside **"Remix"**, whose
> only visible sub-line said "your media → new video". The fix is the copy.
> The Remix pill now reads **"motion control · refs · color"**, the sub-chip is
> **"Motion Control"** with the adapter named in its tooltip, the section names
> Union Control, and `#remixSubGroup` gets a 3-column grid so a three-tool row
> stops living in five columns (chips 91 → 155 px, no wrap).
> New doc: **`docs/MOTION_CONTROL.md`**, linked from a new README "Remix"
> subsection. Gate: `test_control_discoverability.py`, 29 tests.
>
> **The honest half, stated in the UI.** Union Control follows whatever control
> signal it is given, so a pose/depth/canny SEQUENCE works today — and
> **Phosphene ships no preprocessor** to derive one from an ordinary clip. The
> section says so, and the chip's sub-line deliberately does not advertise
> "pose · depth", because a four-word label cannot carry the caveat.
>
> **MEASURED, and it changes what we say about 2.5.** Motion Control on the
> default generation is **half-working, in the half that is hard to notice**.
> Two A/B pairs, same clip, same prompt, same seed, each generation with its
> own text encoder: on **2.3** the camera move transferred AND the prompt
> repainted the world (a monk on a salt flat; a knight in plate armour). On
> **2.5** the camera move still transferred — the control clip rides a *pinned*
> reference latent at follow 1.0, adapter or no adapter — and the prompt did
> not: both renders came back as the control clip's own subject, one of them
> smeared through the middle. This is NOT the Ingredients failure (inert, and
> refused). It is offered, with `LTX_CONTROL_GENERATION_NOTE` shown beside the
> picker only where it is true (`ltx_control_full_repaint()` →
> `control_full_repaint` → `_paintControlGenNote()`).
> **OWNER CALL OUTSTANDING:** all three Remix tools ride 2.3-trained adapters,
> so whether Motion Control (and Colorize) should be *refused* on 2.5 the way
> Ingredients is, is one policy question about the whole group — deliberately
> not settled from a discoverability pass.
>
> **2. `h3_ram` refused with "about 64 GB" — a number no floor here has ever
> been.** The bf16 floor is `H3_MIN_RAM_GB` = 60; the Q8 DiT lane's is
> `H3_MIN_RAM_GB_Q8` = 46, and building that pack was, in its own comment,
> what "puts H3 in reach of 48 GB Macs". Worse, `h3_capable()` gates the
> engine switcher and returns False below 60 GB until the Q8 pack exists on
> disk — so a 48 GB Mac saw **no H3 at all** and never learned that a
> 5-minute, zero-download local build (`scripts/pinokio/h3_build_q8.sh`) would
> enable it. `h3_ram_verdict()` now owns one sentence per band and the refusal,
> the `make_job` fallback, the switcher tooltip, the engine-row note and a new
> inline card branch all read it. That band is `capable: true, available:
> false, needs_q8_dit: true, repairable: true` — the dashed offer the chrome
> already knows how to draw. The retired sentence survives only as an analytics
> needle so replayed logs still classify as `refused`.
> Gate: 13 new tests in `test_refusal_gates.py`.
>
> **3. Control and Colorize floored their canvas to /32.** Both derive dims
> from the SOURCE CLIP inside `run_job_inner`, after `make_job`'s /64
> normalisation has run, so they had their own copy of the rule and it was the
> wrong one. A 768×416 control clip rendered **768×384** with the log line and
> the sidecar both saying 416 — "Width × Height LIES", and here it also crushed
> the reference 8% vertically onto the canvas we named. One `ltx_floor_canvas()`
> now serves make_job, Control and Colorize.
>
> **Three example renders live in `mlx_outputs/`, each beside its source clip**
> (`example_control_{craneout,skater,aerial}.mp4` + `*_source.mp4`), with
> sidecars. Every prompt contains **no camera direction at all**, so every
> camera move in them arrived from the control clip.

> **🎯 2026-08-23 — #62: the interface recommended a preset that cannot carry a
> face, and the number that proved it had no UI. Fixed on `dev`.**
>
> **The defect was in the markup.** The Quick pill was BOTH `active` and badged
> "Recommended" while training at rank 8 — the one tier nobody has ever graded
> on an identity. A first-time user (`@PhantombrainM`, 4.6.0, M4 Max Studio)
> read the label, trained 18 images, and measured **1.98e-04** with the
> self-service CLI: under `WEAK_DELTA_RMS` (2.0e-04). `@blackest` had already
> measured **1.54e-04** on a controlled 15-image Quick run against **5.36e-04**
> for the same dataset on High. Two independent users, same trap.
>
> * **The badge and the default moved together**, because to most people the
>   default IS the recommendation. Character now pre-selects and badges `high`;
>   style keeps `quick`, which is what rank 8/16 is genuinely good at. Both come
>   from one server-side `TRAIN_DEFAULT_PRESET`, served in the bootstrap, so
>   `make_job`'s fallback and the pill cannot drift apart again.
> * **The subtitles carry the measured numbers**, not adjectives: Quick reads
>   "a look, not a face · identity ungraded", and a note under the strip states
>   the band (5.4e-04 – 1.6e-03 for adapters that carry a face) against Quick's
>   two field measurements.
> * **The loop after training is closed.** The verdict has existed since 4.6.0
>   and lived in a log line and a sidecar field — a user with a weak LoRA had to
>   read a 21-comment GitHub thread to learn what his own number meant. That was
>   the real product failure. `/train/list` now carries `adapter_advice` beside
>   `adapter_verdict`; the Train tab renders a banner naming the fix, the chips
>   carry a WEAK/DEAD badge, and a training job that finished weak no longer
>   reads as a plain `done` in history.
>
> **THE SUB-64 GB FINDING, measured from the table and not guessed.**
> `_select_train_profile` rewrites both preset tables under 64 GB, and there
> **High is rank 8 / 500 steps / 448px on `to_q` + `to_v` only** — half the
> projections. That is strictly LESS adapter than the ≥64 GB *Quick* (rank 8,
> 512px, all four, 450–540 steps on a real dataset), which has measured 1.54e-04
> and 1.98e-04 in the field. So **no menu choice on a sub-64 GB Mac reaches the
> graded recipe**, and "just use High" is advice that hardware cannot honour.
> The compact High subtitle now says "NOT the rank-32 recipe", the preset note
> says it in full, and `_train_weak_advice` says something different there.
>
> **Still open, and this change does not touch it:** `@blackest`'s High run
> measured 5.36e-04 — inside the working band — and the trigger still did not
> bind (a LoRA trained on a woman rendered a generic man). Magnitude is
> necessary and not sufficient. E2 (the full rank-32 / 3,700-step run) is the
> experiment that speaks to that.

> **🎚 2026-08-23 — the two rulings, and the widest-spread failure in the fleet.**
>
> * **Compact-tier HQ chips: shown-disabled, not hidden** (owner ruling). The
>   `display: none` on `body[data-cap-tier="q4"] #qualityGroup [data-quality^=
>   "high"]` is gone. `ltx_tiers_payload()` already stamps every `hq` cell
>   `available: false` with a written reason, so the chips now arrive greyed,
>   struck through, `aria-disabled`, reason in the title — and a click writes
>   the same reason into `#engineRowNote` (a tooltip is not discoverable on a
>   chip somebody has just tapped). Hiding a capability is only kind when there
>   is nothing to say; there was something to say.
> * **"Re-run Install Hailuo H3" is `model_missing`, not `other`** (owner
>   ruling). Four raise sites — `--lora` twice, `--first-frame`,
>   `--chain-windows` — each wrote its own sentence, so all four sat in `other`
>   beside genuine unknown crashes. They now share `H3_RUNNER_BEHIND` and one
>   needle, so a fifth flag is classified for free. **`docs/ANALYTICS.md`
>   records that this moves an existing series.**
> * **`image not found: <path>` — 35 events, 22 people, 14 days — was a
>   SERVER-SIDE DEFAULT.** `make_job` filled an empty `image` field with
>   `examples/reference.png`, a demo file that has never existed on any install
>   (not in git, not in `required_files.json`, created by no installer). When
>   the control became a picker the CLIENT-side pre-fill was deliberately
>   removed; the server's was left behind. ~1.6 events per person is the
>   signature of a first-encounter mistake everybody makes once. Three fixes:
>   the default is empty unless `LTX_DEFAULT_IMAGE` names a real file; `i2v` /
>   `i2v_clean_audio` validate their input at job start like every other mode
>   already did; and the picker clears a dead pick at PICK time (only on a 404,
>   so a hiccup cannot throw away a good one) instead of letting a broken
>   preview read as "an image is selected". The A2V lane had already noticed the
>   same default and filtered it out at its own call site — every other lane
>   inherited it.
> * **`input_missing` matched NOTHING before today.** Its needles were "does not
>   exist" / "no longer exists"; every raise site in this codebase says "not
>   found". So the fleet's widest failure sat in `other` wearing a fingerprint.
>   It also now outranks the loose half of `download_failed`, because
>   classification runs on the raw text INCLUDING THE PATH and a great many
>   missing reference images live in `~/Downloads`.

> **🎚 2026-08-21 — two release-gate findings closed on `dev` (`947a183`,
> `9c2f63d`, `490d942`). Both were measured, neither was a guess.**
>
> **1. A bed with no stated length rendered at FULL level.** The mix fix in
> `85f3d43` closed the common case; one shape survived. `bed_length` fell back
> to `audio.duration`, so a soundtrack with no `duration` and no `trim_end`
> measured 0 seconds → `bed_gain_points` returned an EMPTY curve → an empty
> curve means *no filter* → the render played the bed at unity over the
> dialogue while the browser drew it 20 dB lower and the preview played what it
> drew. Reachable two ways, both verified: `POST /storyboard/edit/save`
> persists the shape happily, and `_sbe_auto_edit` writes `"duration": None`
> itself when both the peaks and `probe_media` fail.
> **The ruling: the render matches the preview.** A bed of unknown length plays
> UNDER THE FILM — what is left of `edit_duration` after the block starts, which
> is where the renderer trims the mix anyway. Silence-by-empty-curve is the one
> behaviour nobody asked for and it is the loud one.
> **Measured end-to-end** on a real ffmpeg render through
> `_sb_film_filtergraph` (8 kHz bed, 300 Hz "line", band split, browser JS in
> node as the reference): **+20.52 dB before, −0.08 dB after.**
> The client's half is in it too — `sbeBedLen` was handed `SBE.peaks.duration`,
> a probe of the file the renderer has never read, so a bed length only the
> browser could compute. The peaks are out of that chain; `sbeBedLen(audio,
> filmLen)` is `bed_length(audio, film_len)` term for term. The block geometry
> and the waveform still read the probe, which is what a picture of a file is
> for.
> **The gate that missed it is now a TABLE.** `test_editor_mix` compared five
> constants and four documents; it now runs the real client JS in node against
> the real Python over **32 documents** — no duration, no trim_end, bed longer
> than the film, bed shorter, authored envelopes, duck on/off, muted clips,
> clips with no audio, a J-cut, trims that close the window — asserting the
> curves are identical point for point, that they agree *between* the knots,
> and that the ffmpeg expression EVALUATED is the curve the browser drew.
> Removing the fallback turns 21 tests red; the old suite passed.
>
> **2. Two tabs saving at once were both told they had won.**
> `expect_revision` is a compare-and-swap whose compare and whose swap were in
> different critical sections: the handler read the revision, compared it,
> validated, and only then called `save_edit` — on a `ThreadingHTTPServer`. Two
> debounces landing together both read revision 7, both compared 7 == 7, both
> wrote, **both got HTTP 200**, and one arrangement was gone (recoverable only
> from `history/`, and only by somebody who knew to look). Read-check-write is
> one critical section now, and it lives in `save_edit(..., expect=N)` — the
> function every writer already goes through — behind a **per-board** RLock, so
> unrelated films never queue behind one another. The loser gets the identical
> 409 the sequential case produces. `EditConflict` subclasses `EditError`, so
> every existing `except` still refuses. Driven by two real threads held past
> the read-and-compare; a serial test cannot see this defect at all.
> **A save with no `expect_revision` is accepted, and logged** — the client
> deliberately omits it for "Keep mine", so refusing would strand the
> arrangement on screen with no button that could answer. It stops being
> *silent*: the log names the revision it landed on and says "last write wins".
>
> **Also taken, same visit:** `migrate_edit` shallow-copied and then let
> `repair_audio_overlaps` write `clip["audio"]` on clips still shared with the
> caller — a READ editing its argument, which makes the file and the snapshot
> differ over a repair neither contains. The deep-copy dance is one named helper
> (`_clips_copied`) used by both healers.
>
> **Suite: 1558 (was 1537), green** via
> `./ltx-2-mlx/env/bin/python3.11 -m unittest discover -s . -q`.
> (`test_prompt_enhance_endpoint` fails identically on unmodified `dev` in a
> tree without the helper venv — environment, not this work.)
>
> **KNOWN AND DEFERRED — filed here, not fixed:**
> - **A deleted clip source is invisible until render.** `_sb_timeline_segments`
>   drops an unreadable input and the assembler concatenates, so the film comes
>   out silently shorter. `gaps` get a `gaps_note` sentence in the render
>   result; `unreadable` gets a list (`film["unreadable"]`, named in the
>   markdown) and **no note** — and nothing on the timeline says a source is
>   gone before you press render. The fix is a preflight readability check
>   surfaced on `_sbe_payload`, plus the missing note. Not a small change.
> - **The Editor's `do_GET` has no exception handler** where `_storyboard_post`
>   has `except Exception → 500 JSON`. An exception in `_storyboard_edit_get`
>   propagates and the browser sees a dropped connection instead of a sentence.
>   **Deliberately not wrapped here:** the `proxy` branch streams a video
>   through `_serve_video_with_range`, so a naive wrapper that answers with
>   `_json` after bytes are already on the wire writes a second complete
>   response onto the same socket — the exact defect that route's docstring
>   records. Doing it safely needs response-started tracking, which is not a
>   two-line change.

> **🪪 2026-08-20 — the build stamp names the code that is running.** One
> commit on `dev` (`fe22ed9`), no engine move, no weights move. The header
> stamp is how the owner tells which build he is on; it was answering with the
> working tree, and the page was answering with a commit from May.
>
> **THE REPORT (2026-08-19).** HEAD was `4f65fb5`, the tree was clean, `POST
> /restart` returned ok, the page then served `4f65fb5`'s JS — and the header
> read `dev · c5dc04c`, a SHA from 2026-05-21. The reflog settles it: HEAD went
> `c7d4154 → 4f65fb5` at 13:33 and `→ ccae93b` at 14:28 that day and was never
> `c5dc04c`, so no `git rev-parse` produced that string. **It came out of the
> served page.** A comment inside `renderVersionPill` carried two real SHAs
> formatted exactly like the stamp it described — `"you're on 3.0.0 · dev ·
> c5dc04c (2026-05-21)"` — and the RUNNING build appeared nowhere in the HTML
> at all. Thirteen real commit SHAs ship in that page as historical references
> (`"the loadParams fix (b024bb5)"`); none of them was ever the answer. The
> page answered "which build is this?" with a decoy, confidently, in the right
> format. `<meta name="phosphene-build">` is the answer now, server-rendered,
> and the pill's own comments hold placeholders.
>
> **AND THE STAMP WAS WRONG ANYWAY, in the other direction.** Three failures,
> one defect:
> - **The stamp described the TREE, not the process.** `local_short`,
>   `local_version`, `local_branch` and `local_commit_date` are refreshed from
>   disk by `_detect_local_install_state()` at the top of every remote poll —
>   deliberately, so a tree that was dirty at boot and is clean now stops being
>   suppressed. So within thirty minutes of a pull under a live panel the
>   header advertises the new build while the old code keeps serving. That is
>   v4.4.0's "updates silently kept old code" with the alarm rewired to lie.
>   **Reproduced on the real install before the fix**: :8199 was running
>   `d8e54cf`, `fe22ed9` landed on disk, one `/version/check` and the header
>   claimed `fe22ed9` — code that process had never loaded.
> - **`stale_process` could not cover it.** A boolean bolted onto a
>   disk-derived stamp, so the restart tooltip printed the disk label on BOTH
>   sides: *"Phosphene 4.6.0 is on disk, but this panel process loaded 4.6.0."*
>   Most fixes land without a VERSION bump, so on `dev` that is the normal
>   case. Both builds are named with their SHAs now.
> - **The boot SHA was not captured at boot.** `_capture_boot_head()`'s only
>   caller was `version_check_loop`, a thread `__main__` starts `if
>   VERSION_CHECK_ENABLED`. With `PHOSPHENE_DISABLE_VERSION_CHECK=1` it stayed
>   `None`, and `disk != boot` is False when boot is None — the detector dead
>   again, for exactly the users who opted out of nagging, under a comment
>   claiming it was "deliberately captured at IMPORT".
>
> **THE FIX** is a boot SNAPSHOT rather than a boot SHA — sha, version, branch
> and commit date frozen together at import, beside the definition.
> `get_version_state()` returns both builds unconditionally: `local_*` is what
> the process loaded, `disk_*` is the tree right now. Nothing has to check a
> flag before it may trust a field.
>
> **THE SAME DEFECT, everywhere disk spoke for the process:** the header
> version badge; `/panel/bug-context`, which now also discloses a moved tree
> (*"the fix didn't work"* from an un-restarted panel usually IS that); the
> three analytics events — an `app_boot` from a build that never booted
> corrupts the version funnel; and the film credit, which was crediting a
> renderer that never touched the clip. `running_version()` is the one label
> for all of them.
>
> **GATES.** Nine new tests in `test_stale_process.py` (18 total), and all
> nine fail on the code they replace, each for its own reason: `'n3wn3wn' !=
> '0ld0ld0'` for the stamp following the tree, an unexpectedly-`None` boot SHA
> for the import capture, `['c5dc04c','1ea5f1d'] != []` for the decoy.
> `test_the_version_pill_ships_no_sha_literal` is the one that keeps the decoy
> out: nothing in the code that RENDERS the stamp may carry a SHA of its own.
> 1408 python tests and the three node gates pass.

> **🩹 2026-08-19 overnight — the polish wave: the drafts feature stops
> losing work.** Nine commits on `dev` (`487dd46..HEAD`), no promote, no
> engine move, no weights move. A five-agent review of last night's
> twenty-two commits found four blockers and eighteen confirmed majors in a
> feature that shipped with 1,097 green tests; every one of them lived in the
> gap between "each guard exists" (which the suites pin) and "the state
> machine composes" (which they did not). One claim was REFUTED by the
> red-team and is NOT fixed — see the bottom of this entry.
>
> **THE FOUR BLOCKERS.**
> - **The crash-backup lane destroyed itself.** `sbeBackup` refuses to write
>   while a recovery offer is unanswered — correct — and NOTHING answered it.
>   A save cleared `dirty` and the alarm and left `SBE.backup` set, so on any
>   film that opened with a backup every later backup no-opped, `backedUpAt`
>   stayed 0, and the twelve-second watchdog raised the full-width red SAVING
>   IS FAILING banner on a panel whose saves were all landing. The amber bar
>   stayed over the saved film with a Recover button that would have reverted
>   the save. The save route deletes the file now, the client follows the
>   payload, and `recover_backup` refuses anything `pending_backup` would not
>   have offered. `pending_backup` also stopped deciding on a wall clock: both
>   stamps are whole seconds, and every write that is not the user's (a draft
>   switch, an auto-edit, a restore) used to bury an offer holding the only
>   copy of somebody's afternoon. **An offer ends when the user answers it.**
> - **Copying a draft raised on every board anybody already had.**
>   `create_draft(from_current)` and `duplicate_draft` read `edit.json` raw and
>   handed it to `save_edit`, whose validator refuses any version but the
>   current one — and migration is read-path only, so every board written
>   before `EDIT_VERSION` went to 2 is still v1 on disk. Worse, the index was
>   written BEFORE the document, so the refusal left the film pointing at a
>   draft that was never created. `_land_draft` writes first and moves the
>   pointer after, for create, duplicate and activate alike.
> - **History was keyed on revision alone while revisions restart per draft.**
>   Draft B's revision 3 collided with draft A's, `archive_edit` dropped the
>   collision without a word, and the picker shown while B was open listed A's
>   arrangements — with Restore ready to write one of them into B. **History is
>   a folder per draft now** (`history/<slug>/`), so the prune, the listing and
>   the restore are scoped by construction; what was already on disk moves into
>   the FIRST draft's folder. Deleting a draft takes its past saves with it,
>   which is what the panel has been claiming it does.
> - **The clip inspector was a 36px scroll window.** At 1440x900 with thirteen
>   unplaced shots it held 223px of content in 36px, so "Unlink sound" — the
>   only entry point the J/L feature has — was off-screen and the one control
>   fully visible at the bottom of that unmarked scroll was Ripple delete. The
>   rail's unbounded list is the scroller now, not the inspector.
>
> **THE ARITHMETIC, all three proved in node against the extracted functions.**
> One ⌘Z deleted a peaks-discovered soundtrack (the snapshot read the DOCUMENT
> while `sbeFetchPeaks` deliberately writes only the timeline); a head trim
> past the source dragged the strip's out-point left with it (the clamp landed
> on `start` and not on `film`); and a music drag depended on mouse speed
> (`offset = head - want` folds the previous offset back in, and a pointermove
> stream re-reads the mutated object every event — the same six-second drag
> landed at film 6 as six events and did not move the block at all as one).
>
> **THE REST, briefly.** Nothing swaps the document on a save that did not
> happen (draft switch, restore, relink, render, NLE export — the render most
> visibly, since it would otherwise build the previous cut); the backup names
> the draft it was composed from, so a debounced write cannot land on the one
> you just opened; the geography pass's derived views reach the model even
> when the reply carried no floor plan (they did not, and every shot was then
> coerced onto the establishing view); `GET edit/uploads` answered and then
> told the dispatcher it had not, writing a second 404 behind a 200 — the
> stand-in that hid it now asserts the return value; naming a save is
> reachable again; the two lists in the drafts panel stopped competing for one
> box; and the preview bed plays the window the strip shows.
>
> **THE SURFACE.** Two rung scales (`--ctl-h*`, `--fs-*`) replaced eight
> control heights and eleven type sizes; the header has a hierarchy instead of
> seven identical buttons; the soundtrack is a name rather than an absolute
> path; the rail's unplaced strip is a list rather than a ragged pill shelf;
> the track prettifies model filenames; the alarm reads at 7:1 instead of 3.3;
> the app header stopped clipping its own right edge at 1440.
>
> **NOT FIXED, deliberately.** (1) The claim that a re-plan wipes derived views
> off the board before the planner runs was **REFUTED** by the red-team: the
> guard at `mlx_ltx_panel.py:23504` is never satisfied together with an
> existing board — `sbPlan` sends `locations` and no `id`, `sbReplan`/
> `sbTryAgain` send `id` and no `locations` — so nothing in the panel can
> reach it. Do not re-chase it. (2) README's release banner still says v4.1.1
> and the Editor is still absent from its tab list; both are release
> decisions, and announcing the Editor early is the exact mistake already made
> once. (3) `lora_lab`'s CLI presets still differ from the panel's — the
> comment claiming they are a mirror is gone, the divergence is documented,
> and which way to resolve it is a product call.
>
> Gates at every commit: 21 suites (1,134 tests), `check_ltx_pin`,
> `check_pinokio_scripts`. Verified live on the 8799 test panel, including a
> full 42.9s assemble.

> **✂️ 2026-08-18 night — the Editor stops being a viewer, and a scene stops
> being a list of shots.** Twenty-two commits on `dev`
> (`4a49103..378054a`), no promote, no engine move, no weights move. Two
> campaigns: the timeline became a document somebody can OWN, and the planner
> learned that a scene is a space before it is a shot list.
>
> **THE TIMELINE IS A DOCUMENT NOW, not a render's opinion of itself.**
> - **The music is under the picture, and it is an object.** The soundtrack
>   sat above the clips as a global `offset` — one number, clamped to
>   non-negative, unreachable in the direction that matters. `music_window()`
>   (`storyboard_editor.py:1288`) is the one place that turns three fields into
>   the three numbers ffmpeg needs: `offset` may now be NEGATIVE (the track
>   starts *into* the film, silence in front), and `trim_start`/`trim_end` are
>   in/out points inside the track. A head trim does NOT ripple — music does
>   not ripple — so the seconds a trim removes come back as silence. Absent
>   means untrimmed and `normalise_edit` deletes a neutral trim
>   (`storyboard_editor.py:906-928`), so a handle dragged all the way back out
>   leaves a byte-identical filtergraph and every edit.json written before
>   tonight is still a valid one.
> - **The timeline outlives the clips.** A board whose renders were deleted used
>   to open on nothing, which made "the film" and "the files" the same object.
>   They are not.
> - **Split edits — the sound stops being the picture's shadow.** J-cuts and
>   L-cuts are one feature and they are the owner's words: *"leave some of the
>   audio and drag only the image."* A clip may carry `audio:
>   {start, end, film_start}` (`clip_audio`, `storyboard_editor.py:205`). **The
>   PRESENCE of the field is the switch, not the values in it** — deriving
>   "linked" from equality read as linked the instant somebody unlinked a clip
>   they had not yet moved, and the clip refused to drag. It is still ONE video
>   track and ONE music lane: a split edit is a butt join that lands somewhere
>   else, not a second audio track, and the audio windows may not overlap any
>   more than the pictures may. `EDIT_VERSION` did not move for it — an absent
>   `audio` key means linked, which every document already says.
> - **DRAFTS: the saving is the user's, the backup is ours.** `drafts/` holds
>   named variations with an `index.json` pointing at the active one, and the
>   migration happens on READ (`load_draft_index`,
>   `storyboard_editor.py:1913`) — a board that has only `edit.json` has
>   exactly one draft and always did; it just had no name for it. The upgrade
>   cannot half-happen, because the worst case is an index written the first
>   time the board is read. `POST /storyboard/edit/draft` is ONE route with
>   five verbs (`new` / `duplicate` / `rename` / `delete` / `activate`) because
>   they are five edits to the same active pointer and splitting them would be
>   five places for it to be wrong; `activate`/`delete`/`new`/`duplicate`
>   refuse while that film is rendering, the same guard `import-shots` and
>   `restore` take. The unsaved work rides a separate quiet lane —
>   `history/backup-<draft>.json`, no `edit.json` write, no revision, no
>   conflict check — and is OFFERED back on next open rather than applied.
> - **Two-lane save history.** The owner: *"the auto saves should be saved
>   separately from the manual saves."* Three prefixes in one folder, told
>   apart by their name rather than by opening them (`storyboard_editor.py:988`):
>   `edit-r*` autosave (pruned, capped at `EDIT_HISTORY_KEEP = 50`), `save-r*`
>   a save the user pressed, `keep-r*` a version he named. **The prune never
>   sees the last two.** A glob and not a flag inside the file, because a prune
>   that has to open fifty documents to decide what to delete fails halfway on
>   the first corrupt one. `GET /storyboard/edit/versions` is METADATA ONLY —
>   opening the panel must not be a download.
> - **A save that cannot be dropped, and a failure that screams.** The editor
>   saves on a debounce; a debounce that loses its tab loses the work. The
>   failure path is now a banner, not a console line — the one state where
>   quiet is the wrong default.
> - **A browser refusing to autoplay is not a preference.** The mute fallback
>   is session-only: Chrome's gesture gate made the panel remember a decision
>   the user never took.
> - **The pool takes a file you already have.** Upload-to-pool, with the `kind`
>   fix, so an image lands as a **still** and not as a video with no frames —
>   a still is held for the length of its slot, `start`/`end` synthesised from
>   the slot (`normalise_edit`), and a slug is black with no file at all.
>   `EDIT_VERSION 1 → 2` is for `kind` + `adjust`; `migrate_edit`
>   (`storyboard_editor.py:1182`) upgrades v1 on READ, one way, so bumping the
>   version refuses old *builds* and not every timeline anybody already had.
> - **A test instance cannot pass for the real one.** A TEST badge on the
>   panel chrome. Two panels on 8198/8199 and no way to tell them apart is how
>   an evening gets spent debugging the wrong process.
> - **An empty list is a sentence, not a blank.**
>
> **A SCENE IS A SPACE BEFORE IT IS A SHOT LIST.** The owner: *"a man or woman
> in a bar — behind him there is this, behind her there is that."* The planner
> now blocks the space first (`_GEOGRAPHY_SYSTEM`,
> `storyboard_planner.py:757`): one floor-plan paragraph, then 2–4 named camera
> **views** DERIVED from it. A location carries `views: [{id, name, light,
> description}]` and a shot picks `location_id` + `view` + `eyeline`
> (`storyboard.py:1059-1090`). The car-wash day was the prototype and every
> piece of it was hand-built: `carwash`/`carwash_reverse` as two locations, the
> flipped sun, the no-car reverse background. All of it is derivable from one
> paragraph.
> - **The laws are enforced, not just written.** A view never contains what is
>   behind the camera in that view, and it must SAY so in the words "no car in
>   frame" — that sentence is what `_ABSENCE_RE`
>   (`storyboard_planner.py:2927`) has to stand on. The light flips with the
>   camera. Nobody's own body goes in the view behind them.
> - **The 180-degree check reads the CUT, not the file.** `_enforce_eyelines`
>   (`storyboard_planner.py:2891`) walks shots in *screen* order and flips the
>   second of two adjacent shots that cut between two DIFFERENT characters in
>   the SAME place and both claim the same side. Repaired mechanically, which
>   is only defensible because `eyeline` is a discrete field with exactly one
>   complement: flipping it cannot damage anybody's prose, and the clause is
>   composed at render time (`eyeline_clause`). Prose laws get a model round
>   trip precisely because they do NOT have that property. `lens` deliberately
>   emits nothing — telling these models to look down the barrel buys a stiffer
>   performance than saying nothing — but the value exists so a shot can SAY it
>   holds the lens, which is what lets the check tell "faces the camera" apart
>   from "nobody wrote an eyeline".
> - **The derived views land on the user's OWN locations, and reach the board
>   at plan adoption.** `merge_location_views` (`storyboard.py:1098`) dresses
>   the board's locations in the views the plan derived and never strips the
>   last plan's views, because the shots point at them. Back-compat is one
>   line: `shot_scene_text` falls back to the location's description, so a
>   board with no views, or a shot that names no view, injects exactly what it
>   always did. `views` is ABSENT, never `[]`.
>
> **A GREEN TALLY WAS NEVER PROOF THE FILE COULD DO ANYTHING** (`037f7fc`,
> closing the panel half of the #61/#62 entry below). The trainer measures its
> own delta RMS against `WEAK_DELTA_RMS` and emits `adapter_strength`; the
> panel reads it, logs median/max/carrying-modules, and a verdict that is not
> `ok` finishes the job **in a WARNING state rather than a bare done** — the
> file is kept, but the word for it is not "success". It rides into the sidecar
> and out through `list_characters`/`/loras` as `adapter_verdict`
> (`mlx_ltx_panel.py:2200`), so the library can warn before somebody spends a
> render finding out. A LoRA from a build that never measured reads
> **`unknown`, never `weak`** — silence is not weakness, and a chip on every
> older file would be noise nobody could act on. Same commit: **train preset
> honesty** — the rank-32 recipe is the only one ever graded on faces
> (Aria_v2, Bizarro_v2), so Quick and Medium now say *"identity ungraded"* on
> the pill (`mlx_ltx_panel.py:1567,1573`); saying "fast" and letting the user
> infer "as good, sooner" is the pill doing the lying. **Issue #46** — the A2V
> warning shipped a rule its own reporter refuted: no frames×area constant
> separates the four datapoints (832×480×721 = 287.9 Mpx holds while
> 1024×576×481 = 283.7 Mpx falls apart — a *smaller* product failing), but
> per-frame area separates them cleanly (clean 0.307/0.399, failing
> 0.590/0.922, both dying near frame 450). `A2V_PIXEL_BUDGET` is gone, replaced
> by a knee at **0.45 Mpx/frame**; the canvas is the lever and length is nearly
> free below it. **Pinokio (fuschichou)** — `#remixSubGroup` is a SIBLING of the
> mode bar, not a child, and all six `body[data-workflow] #modeGroup` rules
> omitted it, so the bar hid and the row stayed on Audio, Train, Storyboard,
> Editor, Characters and Studio; added to all six, with a gate that counts the
> two ids and fails if they are ever listed apart. And the **caption green
> summary counted the images, not the files it wrote** (`e35f29b`) — a tally
> that cannot be wrong in the direction that matters. The strength verdict is
> also reachable **from a shell** now, on a file you already have (`9dd90fa`).
>
> **Gates:** `check_pinokio_scripts` PASS (worst dispatch 377/500) ·
> `check_ltx_pin` PASS (unmoved) · analytics dry-run 48 — and its
> "every event the panel fires is documented" test still passes, so
> `docs/ANALYTICS.md` is unchanged **and true**: `adapter_strength` is a
> trainer→panel progress event on the lora_lab stream, not an analytics event,
> and nothing new leaves the machine. Suites touched tonight, all green:
> `test_storyboard` 111 · `test_storyboard_planner` 151 (1 skipped) ·
> `test_storyboard_assembly` 111 · `test_storyboard_editor_api` 250 ·
> `test_storyboard_editor_ui` 181 · `test_lora_compat` 28 ·
> `test_caption_counts` 4 · `test_geometry_grid` 12 · `test_stale_process` 9.
>
> **Docs squared up in the same pass:** `docs/API.md` gained the Storyboard +
> Editor surface it never had (board schema incl. `locations[].views`,
> `shots[].view`, `shots[].eyeline`; `edit.json` incl. `clip.audio` and the
> soundtrack's `offset`/`trim_start`/`trim_end`; the drafts, versions and
> history routes; `adapter_verdict`), and its `/train/start` preset line was
> **wrong** — `high` is not "rank 32, 5000 steps", it is `epochs × image_count`
> capped by the preset, and on a **sub-64 GB Mac `_select_train_profile`
> silently rewrites every preset to rank ≤8 / ≤448 px / ≤500 steps**, which is
> the ungraded regime by definition. That trap is now stated in API.md and in
> the README's Train Character section. **NOT fixed, flagged:**
> `lora_lab/train_character.py:109` still says its presets "must stay in
> lockstep with the panel JS mirror" and they have not been for a long time
> (lab `quick` = rank 16/1500 steps/576 px vs panel `quick` = rank 8/30
> epochs/512 px; two unrelated ETA estimators), and `CLAUDE.md` still opens on
> "Current state — v3.0.6", documents port 8198 only, has no Storyboard or
> Editor in its API section, and its §24 describes an Agentic Flows module that
> was removed in 2026-05.

> **🔍 2026-08-18 — trained LoRAs come out inert (#61, #62): the measurement
> exists now, the cause does not yet.** Two users on v4.5.0 trained characters
> that finished clean and changed nothing. Their render log is the important
> artifact: `LoRA mode unfused: 1152 modules attached … 0 skipped`,
> `FUSED=1152/1152 tensors (576/576 modules)`, `strength=1.00`, on the 2.5 q8
> distilled lane — **every gate we own passed**, because every gate we own asks
> whether the KEYS land, and none asks whether there is anything in them.
> `3c53c21` closes that half: `lora_compat.measure_adapter_effect` returns the
> exact `‖B @ A‖_F` per module (via `trace((BᵀB)(AAᵀ))`, two r×r matmuls, ~1 s
> on a 500 MB adapter), the helper prints `delta_rms` beside the FUSED= tally on
> both LoRA routes, training measures the file it just wrote and stamps the
> verdict into the sidecar, and an adapter whose every product is exactly zero
> is now REFUSED at render instead of returning a LoRA-free video.
>
> **The calibration, measured on this disk** (per-entry delta RMS, median):
> `elontrn_v2` 1.63e-03 · `ariatrn_v2` 1.45e-03 · `eltrumpo_v2` 1.41e-03 ·
> `bizarrotrn_v2` 8.84e-04 · third-party `LTX2.3-Rogue` 1.84e-03 ·
> `Fantasy_Painterly` 5.36e-04 · `bizarrotrn.audio` 4.85e-04 · a
> four-step adapter 1.72e-05. `WEAK_DELTA_RMS = 2.0e-4` sits under everything
> that works and two orders over an untrained file.
>
> **Two negative results, so nobody re-runs them.** (1) *Magnitude is not the
> regression.* Fresh runs on the current pin, same data, rank 8, lr 1e-4:
> 60 steps → 1.43e-04, 240 steps → 2.43e-04. That is √-shaped accumulation
> (4× the steps, 1.7× the delta), and extrapolating it to 5000 steps lands at
> ~1.1e-03 — i.e. exactly where the May-trained `eltrumpo_v2` (1.41e-03)
> actually sits. The current trainer accumulates like the one that produced the
> shipped characters. (2) *Layout is not the regression.* Emitted keys are
> byte-for-byte the layout the shipped adapters carry, and both match the 2.3
> AND 2.5 transformers 576/576 — verified with `inspect_lora_compatibility`.
> What remains is DIRECTION: a 240-step run on the same person's images is no
> more aligned with `eltrumpo_v2` (mean per-module cosine +0.052) than an
> unrelated character is (+0.043), while two runs of the current engine agree
> with each other (+0.171). Suggestive, not conclusive — different rank, 8 of
> 37 images, 240 steps against 5000.
>
> **🧭 2026-08-22 — the #61/#62 question CHANGES: magnitude is necessary, not sufficient.**
> @blackest ran the controlled experiment nobody had: two LoRAs, **same 15-image dataset**,
> differing only in preset. Quick 1.54e-04 · High 5.36e-04 · High on a second dataset
> 5.75e-04 · `bizarrotrn_v2` 8.84e-04. Their script is independent of ours and returns
> **8.84e-04 on `bizarrotrn_v2`** — three significant figures against our own tool, so their
> numbers are directly comparable to our calibration table.
>
> **The finding that redirects the investigation:** their High run measured **5.36e-04 — INSIDE
> the working band — and the trigger did not bind at all.** A LoRA trained on a woman rendered a
> generic man. `WEAK_DELTA_RMS` would have passed that file. The gate can only catch an adapter
> that moved *nothing*; it cannot catch one that moved and learned the wrong thing. Every
> conclusion of the form "the delta is healthy, so the file is fine" is therefore unsound, and
> that includes the reasoning that sent @Morac2 to measure his own adapter.
>
> **Next hypothesis — trigger binding, not the optimiser.** Captions reach training as
> `[VISUAL]: <trigger>, The character stands…` (visible in #62's log), i.e. the trigger sits
> inside a structured prefix rather than where a plain class-word caption would put it. Untested.
> The experiment is a caption-format A/B at fixed rank/steps/seed, judged by whether the trigger
> summons the identity — NOT by delta RMS, which is now known not to answer the question.
>
> **Also confirmed, and it is a trap worth stating plainly:** `_select_train_profile` rewrites the
> preset table below **64 GB**, where **High is rank 8 / 500 steps / 448 px** under a pill still
> named "High". A sub-64 GB Mac cannot reach the graded rank-32 recipe by any menu choice. If
> blackest's box is under 64 GB, their "High" was never the graded recipe.
>
> **E2 IS RUNNING** (this 64 GB M4 Max, on the right side of that line): 37 images, rank 32 /
> alpha 32 / 3700 steps / lr 1e-4 / 512 px — the validated v2 recipe — retraining `eltrumpo`.
> Measured 2.32 s/step, ETA ~2.4 h. The prior `eltrumpo_v2` (1.41e-03, demonstrably working —
> it carries the face through 6 shots of "The Long Night") is backed up at
> `mlx_models/loras/_backup_20260822/` because the new file lands on the same name.
>
> **Shipped in passing** (`8949adb`): `POST /train/upload` is documented as `files[]` +
> `train_job_id`; it takes **`file`** (ONE image per request) + **`job_id`**, so following the
> docs 400s. And its size refusal said "max N bytes per file" for a 73 MB *batch* whose largest
> image was 2.4 MB — the cap is on the request. Both corrected.

> **📌 2026-08-21 — E1 is now the reporter's to run, and the probe was fixed
> before he was pointed at it.** `python3 lora_compat.py <adapter>` is live on
> **public `main`** (`69ee0b4`, cherry-picked onto the v4.6.0 release tree — so
> main is ONE COMMIT past the `v4.6.0` tag, same version string, deliberate).
> It printed three `divide by zero` and three `overflow encountered in matmul`
> lines before every verdict: Accelerate's BLAS raising spurious FPE flags on
> Apple Silicon, on inputs that are entirely finite (`bizarrotrn_v2` is 2304
> F32 tensors, zero non-finite values), results identical with the flags
> masked. Masked at the two matmuls only. **A real NaN now SKIPS the module
> instead of entering the sample** — it could not before: `max(nan, 0.0)`
> returns nan in CPython, `sqrt(nan)` is nan, `nan == 0.0` is False, so the
> value entered `rms` and `rms.sort()` left the median undefined. The number
> this whole investigation rests on could have been quietly wrong on any file
> with one bad tensor. Calibration re-measured and unchanged. `test_lora_compat`
> 28 green on the release tree; the raw-URL path verified end to end from
> GitHub against a real adapter. @Morac2 asked on #62 and was given the
> command; he has kept his training data, so E2's input exists if E1 points at
> training.
> ⚠️ **Promotional posts for v4.6.0 are LIVE** (owner, same evening) — character
> training is the feature under that traffic and its root cause is still OPEN.

> **The two experiments that settle it, in order.** (E1, one minute) @Morac2
> offered his 132 MB adapter and his training data — measure the file with
> `measure_adapter_effect`: in the working band means the file is fine and the
> fault is downstream of it; ~1e-04 means it is not. (E2, one night, needs a
> box with headroom — this 64 GB Mac swapped 27 GB and filled the disk trying)
> the same 37 images at rank 32 / lr 1e-4 / 5000 steps under the current pin
> versus under `v0.14.8`, both measured and both rendered. Panel-side follow-ups
> (the `adapter_strength` event has no UI yet; Quick trains at rank 8 and no
> tier below rank 32 has ever been graded for identity) are written up but NOT
> implemented — another agent held `mlx_ltx_panel.py`.
> ⚠️ **That last sentence is SUPERSEDED as of the night of 2026-08-18**
> (`037f7fc`, top entry): the panel reads `adapter_strength`, warns on a
> not-`ok` verdict, carries it out as `adapter_verdict`, and Quick/Medium say
> "identity ungraded" on the pill. E1 and E2 are still open.

> **⏪ 2026-08-18 — v4.6.0 promoted and WITHDRAWN the same day, on the owner's
> order.** The Editor release went to public `main` (`5b26f91`), tag and GitHub
> release included, and the owner pulled it back within two hours: *"you
> shipped too early... I want to work on it properly."* Public `main` is back
> at v4.5.0 (`ee278e0`); the tag and the release are deleted; the Pinokio
> announcement was never posted. Nothing else moved — `dev` and `beta` keep all
> of it, the from-zero validation stands, and the "SHIPPED in v4.6.0" notes
> below now describe the WITHDRAWN promote. v4.6.0 ships again when the owner
> has cut with the Editor himself and says so — that is the gate that was
> missing: green checks are not the same thing as the owner having used the
> feature.

> **🎬 2026-08-18 — v4.6.0: the Editor becomes a place, and a clip stops having to be a video.** The timeline shipped as the storyboard's sixth stage state, and everything wrong with it followed from that: it could not open without a board, its document id WAS the board id, it died on a tab switch, and its picture was whatever the shot list left over — 150px on any window under 1279px tall. It is now a top-level workflow beside Storyboard, engine-agnostic, with the two columns an editor wants.
>
> - **A media pool, replacing a `window.prompt()` that asked the user to type the NUMBER of a film** — while the gallery holding every clip this panel has ever made was `display:none` for the whole surface. Four sources (this film · other films via `import-shots` · the generations · Images, which the gallery has always held and which nothing could use until a clip could be a STILL) plus an Add black control, because black has no file and therefore no row in any list. One verb: click a row and the clip lands on the track without leaving the Editor. `phos_ed_doc` REMEMBERS the document rather than inheriting it, falling back to the picker when the film it names is gone; leaving the tab SUSPENDS (clock, picture, flushed save) where it used to tear the document down, so glancing at the gallery cost an undo stack.
> - **Three kinds of clip.** `kind: video | still | slug`, absent meaning video, so every `edit.json` on every machine is already a valid v2 document. `EDIT_VERSION 2` shipped WITH the read-path migration, because `validate_edit` hard-refuses a version it does not know — bumping alone would not have refused old builds, it would have refused every timeline anybody already had. A still is `-loop 1 -framerate F -t D` and never goes through ffprobe; a slug consumes no input at all (`color=` is a source filter). That is what forced the input-index refactor: segment index stopped being input index and every input after the first slug shifted. Indices are assigned in one place now, next to the argv fragments that honour them, and the soundtrack is `n_inputs`.
> - **One slider.** Brightness, constant per clip, clamped to ±0.5 and ABSENT when neutral, so a clip nobody graded serialises exactly as it did yesterday. The preview uses CSS `filter: brightness()`, which is MULTIPLICATIVE where ffmpeg's `eq=brightness` is ADDITIVE — CSS has no additive form — so the two are matched at mid-grey, where a person judges exposure, and the strip already says the preview is approximate. `oninput` paints; `onchange` commits, because a slider at pointer speed would otherwise push eighty undo steps and eighty saves for one gesture.
> - **Drag and drop, on the substrate the track already used.** The panel had no `dragstart` anywhere and the track's own gestures are pointerdown/move/up with capture; HTML5 drag-and-drop would have swallowed that capture and the track would never have heard the drop. Same substrate, so the two coexist by construction. A drop lands WHERE IT WAS DROPPED, with ripple, by the midpoint rule every NLE uses; click-to-add still lands at the END, because it cannot move anybody's cuts. On the track, Shift reorders and a plain drag moves — a move puts a clip at a TIME and leaves the hole the generate control fills, a reorder puts it at a POSITION and closes that hole, and neither can be inferred from the pointer. The ghost portals to `<body>`: parented in the pool list it is clipped the instant it leaves the column, which is every drag. A drag ends in a click on a `<button>`, so a flag stops every drop adding its clip a second time.
> - **NLE export — the film as a project somebody else's editor can open.** `<film>_project/` with an FCP7 XML (the one interchange Premiere and Resolve both import), an ExtendScript for After Effects (which has no timeline import at all and never has), and a `media/` of HARDLINKS, `os.link` falling back to a real copy across filesystems. The pathurls are absolute into that `media/`, which is what makes the folder relink on drop. Slugs are written as timeline GAPS rather than generator items: the effect ids for a colour generator differ between the two NLEs, so "one XML for both" would quietly have become one XML for one of them, and a gap reads as black in every editor ever made. The audio is STEMS — clip sound on A1, soundtrack on A2, unducked — because the render's sidechained under-mix has no representation in an NLE timeline, and baking it in would hand an editor a bed they cannot unmix.
> - **ONE assembler.** Export wrote `<slug>_film.mp4` from a second auto-editor over its own copies; the Editor's Render wrote `<slug>_timeline.mp4` from the cut, with the soundtrack. Same folder, same board, two films, and a chip naming the button as the only way to tell them apart. Export now delegates to the timeline whenever an `edit.json` has clips, passes `music`/`music_mode`, and both doors write the same name.
> - **RELINK, the live bug.** `_sbe_board_clips` picks delivery over draft, but `edit.json` freezes the path at the moment of the cut and nothing rewrote it — so "Finish keepers" rendered full-size files the film never used, and the next Prepare pruned their proxies. The server flags them on the GET; one button rewrites the paths, keeps every cut and timing, and builds the delivery proxies before answering. Never automatic: the arrangement is the human's.
> - **A soundtrack can sit UNDER the dialogue instead of deleting it.** `music.mode` is `replace` (unchanged, still the default, byte-identical graph) or `under` (clips keep their audio; the bed is attenuated and then ducked by the dialogue via `sidechaincompress`). The duck constants were MEASURED on a band-split gated-tone rig, not chosen: 0.04/8 gives 5.8 dB, too shallow to hear under a line; 0.02/10 gives 11.4 dB and shipped; 0.01/20 gives 17.7 dB and audibly pumps. The bed returns to full level between lines in all three. End to end through the real filtergraph it measures 7.0 dB with the dialogue intact.
> - **The under-mix was clipping, and the obvious fix is a no-op here.** The first film mixed `under` peaked at 1.31 pre-encode with 1341 hard-clipped samples: `amix` carries `normalize=0` so nothing at all was protecting the sum, and engine dialogue is hot (0.35 RMS on an opening line). `alimiter=limit=0.97` is the filter this obviously wants and on this build it does NOTHING to float samples — measured on the real mix, input peak 1.3075, output peak 1.3075, unchanged with `level=disabled`, unchanged with an `aformat` to s16 before it, unchanged with one after it, reporting nothing either way. A "fixed" render was shipped that measured WORSE (1.279 against 1.216) before anybody looked at the output peak rather than the filtergraph. `asoftclip=type=tanh` at threshold 0.9 holds: the finished film measures peak 0.9678 with zero clipped samples, against 1.279 and 654 before. One test pins the ceiling as the last stage before `[aout]`; a second BANS the string `alimiter` from the graph, so the inviting cleanup cannot be done by someone who has not measured the output.
> - **The trained voice was stripped from every LTX character shot.** Owner-reported more than once. `shot_to_job` decides `no_voice`, and `no_voice=on` drops `<trigger>.audio.safetensors` from the LoRA stack — it decided it with `_HAS_DIALOGUE`, which matches `<d>…</d>` and nothing else. That is exact for H3 and wrong for LTX: `_strip_h3_markup` has ALREADY rewritten the tag away by the time an LTX prompt exists, so `no_voice` was always "on" and every LTX character shot with a line rendered with the face LoRA alone. Confirmed on a real sidecar — `no_voice: true`, LoRA stack of one. It survived because the comment above it said the derivation was exact; it was, for the one engine anybody checked. The gate now shares `_SPOKEN_WORDS_RE` with `shot_speech_problem`, and a test asserts the two agree: when they diverge you get one of exactly two bugs — a mouth with nothing to say, or a real line in a stranger's voice — and this repo has now shipped both.
> - **A line must fit its shot, and it must close.** `shot_pacing_problem()` rejects OVERSTUFFED (more words than the duration can carry) and UNFINISHED (the last line ends on a comma, dash, ellipsis or nothing). The budget BRACKETS the measured evidence rather than guessing: 7 words in 4.04 s delivered fine, so it must allow ≥2.31 w/s; 20 words in 7.04 s was cut mid-phrase, so it must refuse ≥3.31; 2.4 w/s sits between with margin both ways. A warm read is slower — the day's renders split cleanly by voice descriptor at ~2.4 w/s bright against ~1.7 warm, and a 23-word slow read passed the flat budget at 13.04 s and was cut anyway — so `is_slow_read()` reads the descriptor as a tempo marking and both the budget and `speech_fit_frames()` honour it. Hard error on hand-authored boards, where the author is present to choose between cutting words and adding seconds; mechanically REPAIRED at plan adoption, because bouncing a free fix through a 40-second repair round-trip helps nobody. Found in review: the first cut of the descriptor regex was assembled inside a non-raw triple-quoted string, so every `\b` arrived as a literal backspace and the pattern could never have matched — caught because the new tests run against the real module rather than trusting the diff.
> - **LOCATIONS and WARDROBE — written once, injected everywhere.** Four shots of the same character, each saying "dim room, cinematic close-up", came back as a monitor-lit study, a brighter office with papers on the wall, a vintage parlour with no monitors in it, and a near-black void, with the collar changing between them. Nobody wrote a contradiction: the shots simply never agreed on anything, and what a prompt leaves unstated is re-rolled per shot. A location is a board-level entity now — id, name, description — and a shot references one by id; wardrobe rides with the cast member rather than the shot. Both are injected at the single choke point every render path already passes through (`shot_to_job`, `compose_shot_prompt`) rather than at the call sites, which would give the estimate, the re-render and the gap-fill each their own chance to forget. `unknown_location` is the validator that matters most: without it a shot claiming a room this board never heard of renders with no room injected and looks identical to a shot that never claimed one — the continuity failure arriving silently. Additive: a board with no locations composes byte-identically, and schema stays 1.
> - **A screenplay pass, before any shot exists.** `plan_film` now writes concept → screenplay → shots, handing the beats and the actual spoken lines down to the shot pass with the instruction to keep every line word for word. There was no screenplay step at all, so structure and coverage were invented in the same breath. Off by `screenplay=False`; a per-shot re-roll never regenerates it, because the other shots are standing on it; a model that answers with "Sure! Here is the scene:" and no beats is discarded rather than handed down as if it were a scene.
> - **Clips can come from another film.** A board is a timeline, one to one, and coverage is not — the moment B-roll or alternates get rendered as a second board, their clips are unreachable from the first board's timeline, so the answer to "I have clips in two projects" was to render them again. `_sb_import_shots` + `POST /storyboard/import-shots` REFERENCE clips rather than copying them, and three things travel with them because leaving each behind breaks something specific: `imported_from` provenance (without it a re-plan rewrites somebody else's work), the source LOCATIONS (an imported shot pointing at an unknown location correctly refuses to render, so a successful import would produce an unrenderable film), and the source CAST (or wardrobe anchoring silently stops applying). It refuses while the target film is rendering — a running render owns the board file and writes each shot's status back as it lands, so an import holding a read from a second earlier would drop finished clips at the exact moment somebody is collecting them. Found by doing it: an import during a live four-shot render happened to survive, and that is luck, not a design.
> - **The film has a home.** Both assemblies now END on a film state — player, runtime, picture size, size on disk, when it was made, which button made it, the folder it lives in, Show in Finder revealing the FILE, and earlier films listed underneath — served by a new `GET /storyboard/films`. It was needed because `list_outputs` globs `OUTPUT/*.mp4` and never descends, so the one thing this feature makes was the one thing the app could not show you; the user was left on a list of individual shots wondering which one was the movie.
> - **Ten review findings, fixed.** Re-plan erased the film's locations and wardrobe — `/storyboard/plan` assigned both unconditionally from the form and only the FIRST plan's client sends them, so a re-plan wiped the continuity anchors with nothing on any screen showing they were gone; the server patches now instead of overwriting. Prepare DELETED the proxies of exactly the clips the user had just imported, because `plan_proxies` prunes anything not in the clip list it is given and it was given board shots only; it is fed board clips UNION every path in `edit.json`. The step rail — the only door to the editor and the finished film — sat 400+px below the player, nested inside the element that takes the gallery's slot; it is the stage pane's first child now, with a CSS belt so it cannot appear outside the storyboard workflow. The tab strip did not wrap in a 300px pane, so the Editor tab was simply not there to click, the same way the Train tab went missing once. The Render button was 1590px wide — the panel's base `button { width: 100% }` was never overridden in that bar — which made the sticky bar 89px instead of 65. The picture's column budget was stale at 819px, measured with a rail and an 89px bar that are not in the Editor's own tab; re-measured to 748, which took the preview from 173px to 244px on the same window with the column still not scrolling. Intersection callbacks do not run in an occluded tab, so the pool's first screenful loads without waiting for one. The grade flag was wide enough to collide with the clip name, which always ran under it. Four pool sources on a 40% basis wrapped two-and-two instead of three-and-one, which read as Images being an afterthought rather than a peer. And a negative-zero `startTime` in the AE script, which is valid ExtendScript and reads like a bug.
>
> **Validated in a browser and with ffprobe, not asserted.** On a real v1 `edit.json` — nine clips, cut in the Editor, never touched by this branch — opening it returned version 2 with `migrated_from 1`, validated clean, and re-saved with nothing rewritten in any clip. A four-kind timeline was then built through the SHIPPING verbs: ripple delete and the trim handles, a real `PointerEvent` drag from the Images row onto the track, the Add black button, and the inspector's own slider. The drop landed at 1.5 s BETWEEN the two videos by the midpoint rule and rippled the second to 4.5 s; the click that follows every pointerup did not add the clip a second time. **The render ffprobes at 6.500000 s exactly, 156 frames at 24 fps, 1280x720, two streams, 44100 Hz stereo**, and per-slot mean luma proves what the graph did rather than that it ran: video 102.71 · STILL 74.19 with 0 scene changes across the hold · the same window graded −0.30 at 29.30 against 98.28 ungraded · SLUG 16.00, dead flat limited-range black. The under-mix survived a still and a slug in the graph, which is the thing most likely to have broken quietly: max_volume −4.0 dB against the 0.9 ceiling, bed alone under the still (−34.7) and the black (−46.6), ducked under the dialogue (−14.5 / −24.6). The project folder, parsed with `xml.etree`: xmeml v4, timebase 24, ntsc FALSE, 156 frames, ONE video track with three clipitems and the slug left as a 48-frame gap, two audio tracks, four files declared once each with absolute pathurls into the folder's own `media/`, every media file at link count 2 and zero bytes copied. The `.jsx` parses as JavaScript and maps −0.30 to −45.0 of After Effects' Brightness.
>
> **Gates:** `check_ltx_pin` PASS (pin unmoved at `v0.14.19+ltx25.6`) · `check_pinokio_scripts` PASS (worst 377/500) · `check_output_codec` PASS · `node --check` install/update · `py_compile` panel + image engine · Ideogram from-zero (model deleted, token-less re-download and render) PASS · **20 root suites, 826 tests green** · `scripts/test_analytics_dryrun` 48/48, green for the first time since v4.2.0. No engine re-pin, no weights change, no render path touched: `ltx-2-mlx/`, `required_files.json`, `install.js` and `update.js` are byte-identical to v4.5.0.
>
> **Recorded so nobody hunts it, and NOT a regression:** on the validation browser no `<video>` element reaches `readyState 1` at all right now — Chrome's own built-in player stalls on the same proxy file that `curl` serves in 13 ms with correct 206 ranges. Every `<img>` path (stills on the stage, on the track, in the pool) loads fine in the same tab. The timeline's picture was therefore verified through ffprobe and per-slot luma on the rendered file rather than through the live preview.
>
> **A human must still judge by eye:** how the cut FEELS at the seams, whether the beat lines read at default zoom, whether the ducked bed sits right under a real performance, and whether the stage/track proportions are right on a 14" screen.

> **🎞️ 2026-08-16 — SHIPPED in v4.6.0: the timeline gets a face.** ⚠️ **PARTLY SUPERSEDED by the v4.6.0 entry at the top.** The timeline is no longer the storyboard tab's sixth stage state — it is its own top-level Editor tab with its own document id, and the `body.sbe-open` layout takeover described below was deleted rather than kept. Everything else here stands as the record. The editor's server half (proxies, peaks, `edit.json`, eight routes) shipped in `3b129ca`/`3096da5` with nothing to drive it. This is the interface, built as the storyboard tab's **sixth stage state** — not a widget bolted beside it. Vanilla JS, inline CSS, zero dependencies, same idioms as the shot list it sits next to (`sbe*` / `.sbe-*` beside `sb*` / `.sb-*`).
>
> - **One `<video>`, and the measurements say why.** Double-buffering measured no benefit once proxies are all-intra, so there is one element and one decoder. The preview never points at `clip.path` when a proxy exists (235 ms median seek vs 3.5 ms); the one case where it must says `SOURCE (slow — run Prepare)` on the badge instead of pretending. `muted` is set before the first `play()` (Chrome refuses autoplay silently) and nothing is hidden with `display:none` (WebKit will not load it). The approximation is stated once, quietly, under the transport: the browser lands within a few frames of each cut and freezes for about one there; the render is exact.
> - **Every clip owns the gap that precedes it.** That one choice makes a move, a ripple delete and a split one-liners each, and makes `film_end` derivable rather than storable — so the 1x invariant the server refuses an edit for breaking cannot be broken by the client. Both trim handles follow the pointer (left moves the in-point and leaves the tail alone; right ripples), a **locked** clip is an anchor the flow goes around rather than through, and a drag that moves the pointer but not the film is discarded rather than saved as a no-op edit that burns a revision.
> - **The beat grid is never extrapolated.** `beat_map()` fits ONE tempo across a span because real tracks drift; beats outside it do not exist, and neither do lines. Below 0.4 confidence the grid draws muted and the inspector says it is a guess, with the number.
> - **The waveform appears before `edit.json` knows about it.** `prepare` writes `peaks.json`; only an auto-edit writes the soundtrack INTO the edit. Rather than show an empty strip over a track that is right there, the axis comes off the peaks document — and the prepare line says plainly that the arrangement on screen was cut without the grid, and that Auto-edit is what puts the cuts on it.
> - **Filling a hole does NOT move the film after it** — the one operation here that does not ripple, because the shot was generated for that slot so the cuts around it would stay on their beats. `edit/generate` shows the params read back off the QUEUED JOB, not the request, because `make_job` silently drops any field it does not name.
> - **409 is answered honestly:** the other tab's revision, your revision, your arrangement still on screen, and two buttons. `Keep mine` is the only path that drops `expect_revision`, and only because a human clicked it. A 400 lights up every offending clip at once from the server's `errors[]`.
> - **The timeline owns the tab while it is open.** `body.sbe-open` folds the planning column away and gives the stage the window. Not a preference: the brief plans a film that has already been shot by the time anyone opens a timeline, and a timeline is the one surface here whose usefulness is measured in pixels per second.
> - **A narrow-window bug that predates all of this, fixed on the way past.** `.layout` is `flex: 1 1 auto` inside a `display:flex; min-height:100vh` body, so the storyboard breakpoint's `height: auto` never did anything — the grid kept the full 812px, its three `auto` rows were squeezed, the form column got ~316px for ~730px of brief, and `overflow: visible` spilled the remaining **479px straight down through the stage**. The ENGINE strip and the character picker printed over the shot list and over the editor. `flex: 0 0 auto` in that same block is the whole fix, and it is what the block's own comment ("Stacked, the PAGE scrolls") has been promising since it was written.
> - **Validated in a browser, not asserted.** A sandbox panel on :8799 against a real 5-shot board and a real track: prepare built 5 proxies in 0.85 s, tracked 104.195 BPM (confidence 0.54, 82 beats / 20 downbeats), drag-reorder / ripple-trim / split-at-playhead / ⌘Z / place-into-slot all driven with a real pointer, a real 409 forced from a second client, and `edit/render` produced a 27.875 s film with the gaps note disclosed. Checked at **800px and 1400px**: nothing outside `#sbStage` intersects the editor at either width, and the clip track clears the sticky action bar at both. Six bugs were found that way and fixed: `[hidden]` losing to `display:flex`, the stage stealing the track's height inside `#sbStage`'s flex column, the no-op drag, the follower sliding when a hole was filled, the spilled planning column above, and the video's 34vh cap pushing the track under the action bar on a 900px window (it now yields to the timeline: `min(34vh, 340px, calc(100vh - 700px))`).
>
> **Gates:** `check_ltx_pin` PASS · `check_pinokio_scripts` PASS · `check_output_codec` PASS · **588 root tests green** (`test_storyboard_editor_ui.py`, 56 new, extracts the real client functions and runs them in node).
>
> **A human must still judge by eye:** how the cut FEELS at the seams, whether the beat lines read at default zoom, and whether the stage/track proportions are right on a 14" screen.

> **✂️ 2026-08-16 — SHIPPED in v4.6.0: the storyboard export learns to edit.** Assembled films read as clips glued end to end because that is what they were: `_sb_export` concatenated WHOLE clips in `n` order, so every soft head and degrading tail played, and every cut landed wherever a 5.125 s render happened to stop.
>
> - **`storyboard_edit.py`** (new, root) — pure analysis, no HTTP, no panel import, numpy + stdlib only. `best_window()` scores every candidate window inside a clip on sharpness (variance of Laplacian), stability (frame-to-frame difference scored as a BAND, and on the window's quietest quarter as well as its mean, so a mostly-frozen window cannot average its way into the healthy range), luma sanity (near-black and blown-out windows are VETOED, not marked down — the white-tail bug class), and a configurable head/tail positional prior. Every window returns `per_second` diagnostics and a sentence saying what it beat and by how much. `beat_map()` is onset-flux + autocorrelation + a log-normal tempo prior, refined by a phase-folded grid search; it returns `confidence` and the evidence behind it. `plan_cut()` snaps each cut to a downbeat, then a beat, then leaves it alone — and says which, and by how many milliseconds.
> - **Trimming happens in the filtergraph that was already running.** `_sb_film_filtergraph` gained `cuts` and `music`; `trim`/`atrim` + `setpts` per segment, one decode, one encode. A second ffmpeg pass would have cost a whole generation of quality for nothing. The plan is matched to segments **by path, not by position**, because the assembler drops clips ffprobe cannot read.
> - **Opt-in, and provably so.** `_sb_export(auto_edit=False)` is the default, the assembler is called with the same two positional arguments it always was, and `test_no_plan_and_no_music_is_byte_identical_to_the_old_graph` pins the pre-existing graph as a literal.
> - **Validated on the real AURELIUS project**, not on fixtures: ten 1080p renders and AMOR_FATI (7:59). Detected **126.66 BPM** over the first 100 s against the shotbook's ~126, onset energy locked to **±5.55 ms**, **10/10 cuts on the grid** (6 downbeats, 4 beats), 0.00 ms residual, 57.50 s of film in **13.1 s** of wall time. The 60 s target is unreachable and the module says why rather than faking it: eleven beats is 5.211 s and the clips hold 5.167 s, so ten beats per shot is the ceiling.
> - **Honest limits recorded:** the tempo model is ONE constant BPM in 4/4 — the track drifts +0.61 BPM across 100 s and fits at 128.43 across the whole 8 minutes, which is why the grid is fitted to the span being cut rather than the whole file. Octave margin over the 84.4 BPM rival is only 0.15. Grid times carry a measured ~10 ms lead from the analysis window.
>
> **Gates:** `check_ltx_pin` PASS · `check_pinokio_scripts` PASS · `check_output_codec` PASS · 429 root tests green (`test_storyboard_edit` 55 new, `test_storyboard_assembly` 80). `scripts/test_analytics_dryrun` has 3 failures that pre-date this work (verified against a stashed tree) — all three were one undocumented event, `star_prompt`, and were fixed in v4.6.0 (`598d367`).

> **🎛️ 2026-08-16 — v4.5.0: the health cluster becomes one chip, and two fixes that came from outside.**
>
> - **One health chip.** Six pills truncated on a 14" window and could not take a seventh. The chip states the summary — worst state wins the colour, memory stays on the face — and a portaled popover holds Tier / Memory / Helper / Models / ComfyUI / Queue / Render. The pills are unchanged: same ids, same updaters, relocated once at boot and read back, so the summary can never disagree with them.
> - **safetensors data now lands 16-byte aligned** — the lead came from tgo-app-dev/vpipe, who measured 39.1 GB of silent copy-instead-of-mmap on a checkpoint whose data began at 8 (mod 16). Metal buffer offsets must be 16-byte aligned and a misaligned section does not error. The obvious fix is backwards: the file is `[8-byte length][blob][data]`, so padding the blob to a multiple of 16 lands data at 8 mod 16 — misaligned EVERY time. Measured across 200 header lengths: old rule 103/200 bad, naive %16 200/200 bad, correct rule 0/200. `test_safetensors_alignment.py` reads the padding expression out of the panel source so it cannot drift.
> - **The 48 GB generation clamp now respects `LTX_TIER_OVERRIDE`.** Reported on Reddit with the diagnosis already done: `_select_generation_profile()` accepts `tier_key` and never reads it, so a requested 1088x1472 was reshaped to 768x1024 while memory sat at 9.4/48 GB, and forcing the tier moved the modal and nothing else. Their measurement — 12.79 s/it unclamped vs 13.4 s/it clamped, 284 s, no swap, no OOM — is real data from hardware we do not own. An explicit high/pro override now lifts the cap, and `LTX_GENERATION_PROFILE=full` lifts it outright. The 64 GB default STAYS: it was put there by a real swap thrash and one clean result does not retire it.
>
> **Gates:** `check_ltx_pin` PASS · `check_pinokio_scripts` PASS · `check_output_codec` PASS · 15 panel suites green · `node -c` install/update.

> **🔁 2026-08-15 — v4.4.0: every update since v4.0.5 has silently kept running the old code.** The most important release of the day, and it exists because the owner said "I updated and I still don't see any of the changes."
>
> - **The "Restart to finish update" pill has never fired. Not once.** v4.0.5 compared disk HEAD against `_VERSION_STATE["local_sha"]`, described in its own docstring as *the boot snapshot*. It is not one: `_check_remote_once()` calls `_detect_local_install_state()` at the top of **every poll** — deliberately, so a tree that was dirty at boot and is clean now stops being suppressed without a restart — and that same call refreshes `local_sha`. Within one 30-minute interval `local_sha` equals disk again, so `disk_sha != boot_sha` is false forever. The detector could never report the one condition it exists to report.
> - **What that cost every user.** Pull an update, get no prompt, keep running the old process, conclude the update did nothing. Diagnosed on the owner's own install: panel process started **13:20:48**, code updated **20:09:22**, `/version` cheerfully reporting `stale_process: false` while serving a 4.0.9 UI from a 4.3.0 checkout for seven hours. The public tree was verified correct at `325504d` before anything was changed — the code shipped fine, the process never loaded it.
> - **Fix:** `_BOOT_HEAD_SHA`, captured once at process start and never refreshed, is what disk is compared against. `test_stale_process.py` (5) pins it, and `test_polled_state_does_not_defeat_it` writes a fresh `local_sha` exactly the way a poll would and asserts staleness survives — the original bug stated as a test. A missing git binary still reads as "not stale" rather than as a permanent restart nag.
> - **One click to update, not two.** The banner's **Update now** opened a `confirm()` asking whether you meant to press the button that says Update now, and then an `alert()` after the pull whose only content was what to do next. The confirm now guards only the version PILL (a small target that does four other things by state), and the post-pull instruction renders inline in the banner, which is already on screen: *"Updated to 4.4.0 — restart to finish."*
> - **The star ask has a permanent home.** It rendered only inside the update banner, which is gated on being BEHIND — so anyone who keeps Phosphene current, the population most likely to star it, could never see it, and there was no way to reach it deliberately. Now a static header link beside the bug and X icons: no badge, no pulse, no counter, and it retires the banner ask when used.
>
> **Gates:** `check_ltx_pin` PASS (pin unmoved) · `check_pinokio_scripts` PASS · `check_output_codec` PASS · **13 panel suites** green · `node -c` install/update · `py_compile` panel + helper + image engine.
>
> **Queued, not in this release:** the Kijai rank-resized Turbo adapter (0.44 GB vs our 1.96 GB, equal-or-better in a locked A/B) needs its asset published and the resolver allowlist moved; and the health-cluster redesign is waiting on the owner's pick between one consolidated chip and icon-first pills.

> **🎛️ 2026-08-15 — v4.3.0: the engine picker becomes a dropdown, the header stops fighting for room, and contributing has an address.**
>
> - **The picker is a dropdown.** Two segments plus the health cluster already overflowed a 14" window, and a third engine could not have fitted at any width — which mattered, because `ENGINES` already carries a **Flux Video** row behind `LTX_ENGINE_PREVIEW=1`. A trigger showing only the ACTIVE engine costs one header slot no matter how many engines exist, and it *states* the selection instead of leaving it to a highlight. Each menu row now carries its tagline, because the menu is where someone chooses BETWEEN engines: "every mode, LoRAs, characters" / "joint video + dialogue + sound" / "weights announced, not released yet". Every piece of state logic — install offers, repair, capability, mode gating, `engineSegClick` — is untouched; only the presentation changed.
> - **The menu is PORTALED to `<body>`** and positioned fixed. `<header>` is `overflow:hidden` — the same clipping that produced the "avatar cut off" report — so a menu rendered inside it would have been sliced at the header's edge.
> - **The RAM pill lost three quarters of its words.** `21.0 / 64 GB · 33% pressure` → **`21/64 GB · 33%`**. That badge answers one question at a glance — am I near the ceiling — and the decimal on used never changed a decision. The full sentence (`21.4 of 64 GB in use · 33% memory pressure · swap 4.6 GB`) moved to the tooltip, where it is available without costing header room on every machine.
> - **The standalone version badge is gone.** `#versionPill` already carries the version in every state it can be in — `Up to date · 4.2.0`, `Update to 4.3.0`, `Checking · 4.2.0`, `4.2.0 · dev · <sha>`, `4.2.0 · offline`. Printing it twice cost a slot the health cluster needed on exactly the machines where the cluster was already truncating.
> - **CONTRIBUTING.md, and the thing it exists to say: a CONFLICTING PR is not a rejected PR.** Public `main` is a curated snapshot commit, so nothing can fast-forward onto it and *every* incoming PR shows conflicting regardless of quality. Two well-diagnosed contributor PRs sat in draft looking broken because of it. The doc states the actual path — reviewed as a patch, applied to the tree, shipped, credited by name — and says plainly that a change which will not land gets told so, in the PR.
> - **One row, both asks, once ever.** The contributing link rides the same one-time row as the star ask rather than adding a second dismissible prompt: the moment someone is already looking at our GitHub link is the moment the contributing link is worth showing.
>
> **Gates:** `check_ltx_pin` PASS · `check_pinokio_scripts` PASS (worst 377/500) · `check_output_codec` PASS · all 12 panel suites green · verified in-browser with `LTX_ENGINE_PREVIEW=1` so all three engines render: menu portaled to body, LTX ticked as active, H3 offering its 75 GB install, Flux inert as "soon".

> **⭐ 2026-08-15 — v4.2.0: updates announce themselves, and the star ask is a local flag rather than a tracking problem.** Finding out you were behind required clicking the header pill, which is a check nobody performs.
>
> - **A banner, deliberately not a modal.** An update is not an emergency, and a dialog that blocks a running render is worse than being out of date. It states the available version and the one you are on, offers Update now (reusing the pill's existing pull path — one implementation, not two) and Later. `update_banner_dismissed` stores a **VERSION string, not a boolean**, so dismissing 4.1.1 cannot silence 4.2.0.
> - **The star ask rides the update moment and appears at most once per install.** It is the one point where the user is already waiting on us. Clicking through, or saying "already did", writes `star_prompt_done` and the row never returns — *including* on every future update, which the tests pin explicitly.
> - **Not an analytics lookup, on purpose.** GitHub can only answer "did THIS AUTHENTICATED USER star it", so suppressing the prompt for people who already starred would mean asking them to sign into GitHub inside a local video panel. A local flag answers the same question with no accounts and no privacy story to defend. `/star-click` records one **anonymous count** with a `via` of `link` or `already` (junk normalises to `link`) and carries no identity, matching every other event we send.
> - **Two live bugs found by testing it rather than reading it.** `/settings` is form-encoded: a JSON body parses to nothing and returns a cheerful `ok:true` having saved nothing, so the banner dismissed itself on screen and returned on the next boot. And form values arrive as strings, where `bool("false")` is True — which would have made "don't ask again" impossible to clear. Both pinned by `test_update_banner.py` (6). A third, the panel's global `button { width: 100% }`, turned every banner button into a full-width bar until the classes opted out explicitly.
> - **Verified in a browser, not asserted:** banner shows while behind → both clicks persist (`star_prompt_done: true`, `update_banner_dismissed: "4.1.1"`) → the same version stays hidden → a NEW version brings the banner back with the star row still gone.
>
> **Also:** the README's "Current release" line said **v4.0.4** while 4.1.1 was out — the first thing anyone landing on the repo read was a five-release-stale project. Updated, along with the repo description (which never mentioned Hailuo or H3) and ten new topics.
>
> **Gates:** `check_ltx_pin` PASS (pin unmoved) · all 12 panel suites green · `check_output_codec` PASS · `py_compile`.

> **🩹 2026-08-15 — v4.1.1: the Metal-watchdog retry finally arms on the machines that needed it.** Every render dying with `helper exited from SIGABRT` on an M2 Max, and the error text inviting the user to file an issue about a retry that could not fire.
>
> - **The mitigation was right and never ran.** `_METAL_TIMEOUT_RX` matched `kIOGPUCommandBufferCallbackErrorTimeout` and `Caused GPU Timeout Error`. macOS names the watchdog kill by how it decided to kill you, and an M2 Max on macOS 26.5.2 kills a long prompt-encode command buffer as **`ImpactingInteractivity`** instead. Same kill, different word, so the Gemma-encode fallback (truncate `LTX2_GEMMA_MAX_LENGTH`, retry once) sat armed-but-unreachable while the render aborted at `[Encoding prompt]`.
> - **Root-caused by the reporter, not by us.** @ybekocak captured the crash line by running the helper's job by hand, proved the engine CLI succeeded on the same weights/venv/machine, and identified the regex as the gap — with the fix confirmed working on their hardware before it was written down here. #44 (M1 Max, v3.2.6) is the same failure wearing the other code.
> - **Deliberately not the whole family.** `OutOfMemory` and `InnocentVictim` are different failures; a shorter prompt is not their fix, and arming a retry on them spends a second render to reach the same end. `test_metal_watchdog_signature.py` (5) pins both positives and all three negatives, and extracts the pattern FROM the panel source so the test cannot drift away from what ships.
>
> **Gates:** `check_ltx_pin` PASS (pin unmoved at `v0.14.19+ltx25.6`) · all 11 panel suites green · `check_output_codec` PASS · `py_compile` panel + helper + image engine.

> **🔁 2026-08-15 — v4.1.0: the forming take MOVES.** The live preview has always decoded a short window of latent frames per publish — 17 on LTX, 14 on H3 — and then thrown all but one away, so the stage showed a still that twitched every few seconds. It now keeps them. Same decode, same cost, one extra encode: an animated WebP the browser loops natively.
>
> - **THE PIN MOVES: `v0.14.19+ltx25.5` → `v0.14.19+ltx25.6`**, and it is the LTX half of the loop. `live_preview.py` encodes the frames it already had into `preview_latest.webp` (42 ms/frame) alongside the PNG it always wrote, and announces it as `preview_loop` / `preview_loop_frames` in `status.json`. The still is still written and still served as `still_url`, so nothing that consumed the old contract breaks. The H3 half rides its own runner (`codex/live-preview`, `c45a8d4`, 40 ms/frame) and reaches installs by re-running Install Hailuo H3.
> - **Both adapters prefer the loop and fall back to the still**, by the same three lines: read the announced `preview_loop`, use it only if the file is actually on disk, otherwise the latest PNG. A runner that never announces a loop — an older H3 clone, a lane that publishes stills only — is untouched and unaware.
> - **Measured, not asserted:** a real 704×416 t2v render on a clean 4.1.0 panel published the loop at the FIRST forward — `preview_loop = preview_latest.webp`, `preview_loop_frames = 17`, a **16-frame 512×288 animation in 16 KB**. `/status` carried `preview.animated = true` with both `url` (the WebP) and `still_url`, and the browser DOM confirmed the stage took it: `playerWrap.className = "player-wrap live-stage"`, `_liveStageOwnsPlayer = true`, the stage image src ending `preview_latest.webp`.
>
> **Gates at the promote tree:** `check_ltx_pin` PASS at `v0.14.19+ltx25.6` (tag verified to CONTAIN the three pyproject bumps, checkout HEAD == tag, installed packages report `0.14.19+ltx25.6` == `_LTX_EXPECTED_VERSION` — the skew class that bit the 25.5 bump, checked explicitly this time) · vendored engine `test_live_preview` + `test_repin_masked_sample` 19 · all 10 panel suites green (`test_stage_live_preview` 18, `test_schedule_preset` 19, `test_geometry_grid` 6, `test_h3_turbo_adapter` 11, `test_character_roundtrip` 35, `test_lora_compat`, `test_spicy_contract`, `test_storyboard`, `test_prompt_enhance_endpoint`, analytics) · `check_output_codec` PASS (yuv420p per sidecar, faststart present) · `py_compile` panel + helper + image engine · `node -c` install.js + update.js · no plain-pip mflux.
>
> **Known and NOT fixed here — the forming take can land in the Now card instead of the stage.** The stage is refused whenever `_liveStageMediaHeld()` is true — a video playing, or playback controls touched within `LIVE_STAGE_PLAYBACK_HOLD_MS` — which is v4.0.4's deliberate "a clip you are watching is never stolen" rule. The finished clip auto-selects and **autoplays** when a render completes, so starting the next render while it plays hands the stage to the chip and the preview to the Now card. `normalizeLivePreview` returns `eligible: true` for an H3 job the moment frames exist, so this is not an eligibility gap. The fix belongs in the hold rule (an autoplaying auto-selected clip is not "a clip you are watching"), and it needs a reproduction on the shipped build before it is changed.

> **🎯 2026-08-15 — v4.0.7: the reference image is honoured on every tier, "Inspire" becomes a mode, Ingredients stops pretending, and the analytics finally answer per-machine questions.** Four owner reports in one release, plus the schema-v2 render events.
>
> - **THE PIN MOVES: `v0.14.19+ltx25.4` → `v0.14.19+ltx25.5`.** +ltx25.4 fixed the i2v anchor on the euler-ancestral loop and left `res2s_denoise_loop` open as an owner-visible decision — the lane 2.5's HQ tiers actually run. The owner's report is that decision arriving: a reference image on High / High·720p produced a high-quality clip of **something else**. Same mechanism, second loop: res_2s injects SDE noise **unmasked** at both substep and step level, so every forward after step 1 saw noise where the anchor should be and the terminal denoise stamped the image back only into the delivered frame. `res2s_denoise_loop` now takes `repin_masked_sample` (default False = the old bytes) and `ti2vid_two_stages_hq` **resolves it per generation** — so 2.5 anchors i2v for real and **2.3 is byte-identical by construction**, which is precisely the objection that deferred the fix, answered structurally rather than by measurement.
> - **Inspire ships as a MODE, because the accident was worth keeping.** `denoise_loop` takes the same flag defaulting True, so passing False deliberately restores the pre-fix drift: the reference guides subject, style and palette while the shot composes itself. Image mode on 2.5 gets **Anchor / Inspire** pills that say out loud that Inspire will not show your picture as frame one. Lane-gated in `make_job` exactly like `schedule_preset` (2.5 + LTX + i2v only; a stale pill resets quietly), threaded to the engine as `loose_reference` on both the HQ and distilled lanes, and it round-trips through Load Params, the Customize summary and the ⓘ modal from day one. `tests/test_repin_masked_sample.py` (4) pins all four directions, on schedules that deliberately stop before the terminal 0.0 — that final denoise masks x0 on every path and is exactly what made the original defect invisible.
> - **Ingredients refuses the generation it cannot serve.** Its IC-LoRA is 2.3-trained and Lightricks has published **no 2.5 one** (their entire 2.5 IC-LoRA catalogue is a pixel upscaler). On 2.5 the in-context transfer does not graft onto the layers, and because the reference sheet rides at **strength 0.0 by design** (at 1.0 the model copies the sheet verbatim — measured), a dead adapter means the references are simply **absent**: the owner got an unrelated subject at full two-stage cost, ~11 GPU-minutes. One predicate (`ltx_generation_serves_ingredients`) now feeds the worker's refusal, the bootstrap flag and the tests; the chip paints unavailable with the reason and points at Inspire. 2.3 installs are untouched, and the day a 2.5 adapter ships this is one registry row.
> - **H3 can render vertical.** Every H3 canvas shipped landscape, which left social formats unreachable on that engine. Portrait is the **same canvas rotated** — 576×1024 is the same 0.59 MP, the same packed-row count, therefore the same wall clock and peak as the shipped 1024×576 cell — so every tier estimate stays true by construction. Deliberately a per-render flip applied where `make_job` stamps the cell's geometry, **not** a third tier axis: `h3_tier` is the wire format every sidecar carries and a key change would strand replay for every clip ever rendered.
> - **H3's missing live preview now says why — and the CLAIM IN THIS BULLET WAS WRONG, corrected in v4.0.9.** What shipped in 4.0.7 said "no published H3 runner branch implements `--live-preview`… that half is not built yet". The runner half **was** built: `live_preview.py`, its tests, a bench and eleven CLI options on branch `codex/live-preview`, validated at the time. It had simply never been **pushed**, and the owner's install pointed at a perf worktree (`codex/opt-turbo` on `codex/perf-experiments`) rather than the published runner — two facts that together made an unpublished feature look like an unbuilt one. The lesson is the size of the claim: "I could not find it on the remote" is not "it does not exist". It is published now (`codex/live-preview` pushed; `codex/h3-engine-v2`, the branch `install_h3.js` pins, fast-forwarded onto it), proven with **7 preview frames** published by a real H3 render, and the note's copy now points at the one action that fixes it — re-run Install Hailuo H3.
> - **ANALYTICS SCHEMA V2 — the render events answer the questions the fleet exists for.** `version` / `chip_family` / `ram_gb` / `os_version` lived on `app_boot` only, so "failure rate by chip" or "render time by machine class" needed a per-install join nobody had written and PostHog's UI cannot do in a click. Every render event now carries them, plus **`wall_sec_bucket`** (numeric lower edge of a 17-rung log ladder, so percentiles work natively and an 8:19 → 12:00 regression moves buckets where the old 5-value `duration_bucket` could not see it), `canvas_class`, and a **closed 16-value `error_class`** classified LOCALLY — only the class leaves the machine, with a 12-hex fingerprint of the already-scrubbed line when the class is `other`, so "the same unknown thing, 17 times, all on M1 Max" is countable without transmitting the text. Adoption fields (steps, accel, temporal_mode, upscale(+method), schedule_preset, chain_windows, chain_prompts_used, lora_count — **count only, never which** — lora_kinds, character_used, audio_mode) and a once-ever `first_render` activation flag complete it. `docs/ANALYTICS.md` documents every field in the same commit; suite 43 → 48.
>
> **Gates at the promote tree:** `check_ltx_pin` PASS at `v0.14.19+ltx25.5` · vendored engine suite green · `check_pinokio_scripts` PASS · `check_post_update` PASS · `assert_registry` 169 · `assert_schedules` 45 · `check_output_codec` PASS · `test_repin_masked_sample` 4 · `test_schedule_preset` 8 → 19 (Anchor/Inspire lane gate, Ingredients predicate + bootstrap, H3 orientation) · `test_geometry_grid` 6 · `test_character_roundtrip` 35 · `test_stage_live_preview` 18 · `test_h3_turbo_adapter` 11 · `test_lora_compat` · `test_spicy_contract` · `test_prompt_enhance_endpoint` · analytics 48 · storyboard suites · py_compile · page parse.

> **🔧 2026-08-14 night — v4.0.5 ASSEMBLED on dev/beta (`65880b3` + version `66f0dbd`), HELD for the owner's word.** The two recovered audits' punch-list, closed in one campaign. (1) **H3 Turbo install is LIVE again**: the v1.0 runner-layout repack is a published release asset on `weights-ltx25-v1` (sha256 `d51d626f…`, verified byte-identical against the local repack the selection was graded on); `/h3/turbo/install` streams it partial→verify→atomic-rename, `install_h3.js` fetches the same asset best-effort via `scripts/fetch_h3_turbo.py` (resumable, digest-checked, short Pinokio-safe lines), and the retired **ckpt500-EMA is accepted again as the LAST fallback** — dropping it in 4.0.4 un-Turboed installs that rendered fine on it. Raw v0.1 stays refused by name. Turbo gate 5→11 tests incl. script↔panel pin parity. (2) **The dead HQ Speed control is gone** (skip-step fields were dropped at the engine boundary; both pills ran identical settings under a "~12% faster" claim), and the collapsed summary stops printing "exact speed" from the retired accel input. (3) **The real speed control**: the engine's 2.5-only `fast` schedule preset (F6S2, 5+2 forwards, owner-graded, **a different take**) wired end to end — Speed pills → `schedule_preset` → make_job (lane-gated: dropped with a sentence on 2.3/HQ/H3) → helper → `generate_two_stage`; tier cells carry `fast_eta` (cost-model 5+2/8+2 ratio, so fixed costs aren't discounted), chips re-price while Fast is armed, and Load Params round-trips it from day one (`test_schedule_preset.py`, 8). (4) **Geometry truth**: make_job floors LTX canvases to the two-stage engine's /64 grid and frames to 8k+1 (floored, never raised) and pushes a sentence when it does; dims inputs step by 64; warnings state what WILL render; H3 tier-cell geometry untouched (`test_geometry_grid.py`, 6). (5) **Load Params restores the cast** (owner-reported): the fire-and-forget IIFE is replaced by an awaited restore through `applyCharacterSelection()` — the new non-toggling hydrator — with visible failure when the character left the library; quality chip by the sidecar's own token, explicit `no_voice` with touched provenance, Extend's four fields, STG (absence = Off), and the Length chip repaints after programmatic frame assignment. A hidden checked No-voice can no longer rewrite a plain render (submit gated on a cast; every hide path clears the box). (6) **One health cluster**: the six machine-state pills live in a single bordered container that is the header's only shrinkable region — narrow windows degrade it to colored state dots, then collapse it; the creator avatar is never clipped again. (7) **Stale-process pill**: `/version` compares HEAD-on-disk to HEAD-at-boot each poll; a promote landing under a running panel renders "Restart to finish update" — **validated live**: the dev panel booted at `90b8a6e` reported `stale_process: true` the moment `65880b3` landed. (8) Keyframe sidecars stamp the resolved q8 pack, not the 2.3-defaulting `MODEL_ID_HQ`. **Gates all green at the tree** (pinokio scripts 377/500 worst · ltx pin unmoved at `v0.14.19+ltx25.4` · post_update · registry 169 · schedules 45 · character round-trip 35 · lora_compat · stage_live_preview · spicy · turbo 11 · schedule_preset 8 · geometry 6 · storyboard suites · py_compile · page parse). **Known-remaining / descoped**: the owner's "stray −1 bar" did not reproduce in either live panel's idle state (needs a screenshot the moment it appears); HDR Load-Params restore is moot while the HDR pill ships commented out (roadmap-parked); render validation of the fast preset + Turbo v1.0 timing run deferred until the GPU frees (the eltrumpo retrain holds both locks overnight). **Audit sources**: `~/AI/projects/phosphene/notes/customize_audit_2026-08-14.md` + `journey_audit_2026-08-14.md` (recovered from the sandboxed sessions' stdout; the FIELD REGISTRY campaign spec in the journey audit is the post-4.0.5 systemic fix).

**Public release: `v4.5.0`** (2026-08-14 — the live preview leaves the postage stamp and plays full-size on the main stage; the Spicy option follows the Settings switch on both engines; H3's Turbo adapter selection moves to LightX2V v1.0 768p and its automatic install is paused until the repack is published. Carries v4.0.3's picker-LoRA crash fix and the v4.0.2 content: the 720p High tier, i2v that animates the image you gave it, characters on the engine's native runtime-LoRA route, and the vendored pin at `v0.14.19+ltx25.4`; see the entries below. The narrative in this paragraph stops at v3.1.1 and is kept as history). **The 2.3/2.5 divergence that ran through the v3.8.x line was closed by v4.0.0** — LTX-2.5 is the registered default and the vendored port is pinned at the tag `v0.14.19+ltx25.4`. v3.8.x registered 2.3 only and pinned `871694d`, because the 2.5 packs were not publicly downloadable at cut time; they are now. See `CLAUDE.md`'s pin table. The entire v3.0 line shipped — v3.0.0 (Characters / Voice / Image Studio / A2V, May 23) → v3.0.1 (FFLF crash fix) → v3.0.2 (Boost/Turbo accel restored after a 2-month silent regression) → v3.0.3 (HiDream hidden, #15) → v3.0.4 (CivitAI SSL) → v3.0.5 (A2V kwarg signature shim, #5) → v3.0.6 (deep-review hardening) → v3.0.7 (GPU-race fix + /status version surfacing) → **v3.0.8 (ltx-2-mlx v0.14.8 catch-up — codec-only patches, I2V "mosaic"/Metal-watchdog fix #17, ASCII version-skew handshake fix)** → **v3.0.9 (custom I2V Width×height inputs restored for power users; Train-tab dev-transformer Download "unknown install key" fix — form-encoding, reported by @cocktailpeanut)** → **v3.0.10 (model-integrity self-heal — boot scan + `/status.model_integrity` + a one-click Repair banner for corrupt/partial weights, the leading *non-Metal* "mosaic" garbage-decode cause; safetensors header+size check)** → **v3.0.11 (latent helper-handshake hang fixed: `_read_until` mixed `select()` on the raw fd with a buffered `readline()`, so any event the helper emitted *right before* `ready` — a version-skew line, the new runtime fingerprint — got stranded in the TextIOWrapper buffer and the panel hung 120 s → "helper failed to start: None". This is the true root cause behind the mislabeled "⚠️-emoji" handshake break. Now reads the raw fd via `os.read()` into a carry buffer and drains every complete line before re-`select()`-ing. PLUS: every render log self-documents its runtime — mlx/mlx-metal version + Apple chip + macOS — surfaced in `/status`, the `helper ready ·` line, and a standalone `runtime |` line, the exact data needed to triangulate the remaining MLX-numerical "mosaic")** → **v3.0.12 (the REAL mosaic fix + 3 more. The mosaic is NOT a code bug or the missing-upscaler theory (both disproved by local repro — the bare 6-file Q4 set renders byte-identical-clean on M4 Max); it's stale/corrupt weight *content* — right size, wrong bytes — that v3.0.10's header+size check can't see (confirmed: ronyeoh/#18 + claude3d/#5 fixed it only by a fresh full re-download). Added **deep checksum verify vs upstream**: Settings → Model files → "Verify model files (checksum)" / `/models/verify-deep` hashes every weight + compares to HuggingFace's published SHA-256 (proven to match local), mismatches feed the existing one-click Repair re-download. Validated: flipped 16 bytes mid-file → header+size passed, deep-verify caught it. ALSO in v3.0.12: character-LoRA-shows-as-"style-only" fix (#5 — `list_user_loras` recovers trigger+kind from the `_v2` filename when the sidecar is missing); `cgi`→`email` multipart migration (Python-3.13-ready, removes the boot DeprecationWarning, adopted from @ssfeather's PR #22); dynamic 3–8 multi-keyframe UI (adopted from @youngbee12's PR #20, ported clean with the integrity feature preserved + a JS ReferenceError fixed). Folds in v3.0.11's handshake fix + runtime fingerprint.)** → **v3.0.13 (character LoRA fixes, #5 @claude3d: a multi-word/spaced character filename no longer mislabels as style-only — the id regex gated out spaces before the sidecar was read; and the Character tab no longer silently falls back to plain T2V — a UI desync left the avatar's selection ring lit after `character_id` was cleared. Plus an `enhancePrompt` element-id fix. Cherry-picked to prod `ce03139`, release `814dc2d`.)** → **v3.1.0 (Ideogram 4 — open-weight 9.3B text-rendering image model + a visual text-placement canvas that places exact text via normalized-bbox captions. mflux 0.18 `mflux-generate-ideogram4`; gated `ideogram-ai/ideogram-4-fp8` ~26 GB, needs a HF Read token + license. Fixed the gated-download 401 that would have hit EVERY user — the panel now exports its configured token to subprocess env via `_sync_hf_token_to_env` so a stale `hf auth login` cache can't shadow it; proven clean-room. Validated end-to-end: 8-text-element field-guide poster rendered clean, every label placed per its bbox.)** → **v3.1.1 (Ideogram output-card buttons hotfix, reported by Mr Bizarro: an Ideogram image's prompt is a caption JSON, which broke every output surface that fed it into a plain box / inline onclick — the Recent-list Animate button was worked-but-dead because apostrophes/&/< in the caption terminated the single-quoted onclick. Display-layer fix: `_displayPromptFor` derives `high_level_description`, `escapeHtml` the onclick args, friendly `_imgEngineLabel`, and `loadParams` restores an Ideogram image into the visual canvas instead of dumping JSON. `params.prompt` stays the caption JSON; renders unaffected. Validated live.)** All validated end-to-end on the dev panel (deep-verify corrupt-detect + repair, /loras, byte-identical upload, multi-keyframe render, FFLF regression). See §4 for the full version history. The May-17 Codex C+ UI restructure (capability tiers, Q4 surface, Character as 5th mode pill, accel kill, HQ-speed move) and the Train-tab / LoRA-chrome work that followed are all baseline now, folded into v3.0.x. There is no `v2.0.6` tag.

> **🎬 2026-08-14 — v4.0.4: the forming take plays on the MAIN STAGE.** The live preview existed since v3.7 and lived in a postage stamp next to the queue, which is not where anyone looks while they wait. It now takes the big player.
>
> - **A preview-capable render owns the idle stage from the moment it starts.** Before the lane's own meaningful gate it shows a calm `Finding the shot…` — the model is warming and there is nothing honest to draw yet. After it, the cache-busted preview image at full size with a `LIVE` badge, `forming take · step n/m`, the existing ETA, and the existing cooperative **Stop early**. The thresholds stay server-owned and per-lane (`6` on distilled lanes, `2` published estimates on `res_2s`; H3's `h3-live-preview/1` adapter marks its proven first forward meaningful), so the UI never invents a moment the engine has not reached.
> - **A clip you are watching is never stolen.** A playing video — or one whose controls you touched in the last 12 seconds — keeps the stage, and a small `LIVE · return to render` chip is the explicit way back. Picking an output from the list while a take is forming is a choice, not an accident, and it is honoured.
> - **Completion cross-fades instead of flashing.** The last preview frame stays mounted until `list_outputs()` actually admits the new mp4, then the decoded finished video fades over it. The deliberate two-second in-flight/mtime listing cutoff used to produce an empty player for exactly that gap; it no longer can.
> - **One poll feeds both surfaces.** `/status` normalises a single live-preview object consumed by the full stage and the compact Now thumbnail. No second request, no second timer, no new queue-render path. H3's preview wiring is capability-probed the same way v4.0.2's `stale_engine` reporting is: a runner with `--live-preview` / `--live-preview-dir` gets the H3 TAE checkpoint and a per-job live directory, an older runner gets an unchanged argv, and H3 exit `75` maps to `stopped`, never `failed`. Special LTX lanes that publish no preview are untouched.
> - **16 tests** (`test_stage_live_preview.py`) execute the panel's REAL client functions extracted from `mlx_ltx_panel.py` in Node, plus the Python H3 file-schema adapter — the warming lane, full-stage ownership, both do-not-steal cases and both halves of the completion handoff.
>
> **🌶️ Spicy is one switch, and it is the Settings switch.** The per-render NSFW option was gated inconsistently across the two engines, so the answer to "is Spicy on?" depended on which engine was selected. `spicyModeEnabled()` is now the single client predicate and `spicy_mode_enabled()` the single server one: with Settings off the CivitAI `Show NSFW` control is hidden (fail-closed in the static markup, not by a post-load hide), neither the first search nor `Load more` may submit `nsfw=true`, and the server forces the upstream query to `nsfw=false` and filters NSFW-flagged cards as defence in depth. Engine selection now changes only the LoRA family, which is the one thing it should ever have changed. The modal also waits for `/settings` before its first search, closing the stale-checkbox race. `test_spicy_contract.py` runs the real extracted `civitaiSearch()` against a DOM shim with the hidden box force-checked: the captured URL carries no `nsfw=true`, and the Python boundary receipt independently proves the server refuses a hand-sent `nsfw=True`.
>
> **⚡ H3 Turbo's adapter selection moves to LightX2V v1.0 768p — and its automatic install is PAUSED until the repack ships.** The resolver is an exact ordered allowlist, never a glob: `lightx2v_v1.0_768p_ourlayout.safetensors` first, the alpha-folded `lightx2v_v0.1_ourlayout_alpha8.safetensors` only as a fallback. The raw upstream `minimax_h3_fl2v_turbo_4step_v0.1.safetensors` is refused by name — its alpha/rank factor lives outside the checkpoint, so `--lora …:1.0` applies a 16× oversized delta and renders coloured noise. Runner invocation is the native `--lora PATH:1.0`; the Larry-specific `--lora-adaln` companion path is gone. Selection credit: **core_tan's public LoRA testing** plus the owner's visual review. **The honest consequence:** the runner-layout repack is not yet published as a digest-pinned release asset, so `/h3/turbo/install` fails closed with the exact publication requirement, `install_available` is `false`, and **the retired ckpt500-EMA adapter is no longer accepted** — anyone whose Turbo worked on v4.0.3 will see Turbo report its adapter missing until the asset lands. Full-quality H3 renders and every LTX lane are unaffected. `docs/H3_ENGINE.md` now marks the v1.0 estimates as derived rather than claiming the retired adapter's measured wall clock, because v1.0 has had no end-to-end timing run. `test_h3_turbo_adapter.py` (5 tests) pins the preference order, the never-select list, and the fail-closed installer.
>
> **Gates at the promote tree:** `check_pinokio_scripts` PASS · `check_ltx_pin` PASS at `v0.14.19+ltx25.4` (unmoved) · `check_post_update` PASS · `test_stage_live_preview` 16 · `test_spicy_contract` 4 · `test_h3_turbo_adapter` 5 · `test_character_roundtrip` 27 · `test_lora_compat` 9 · storyboard suites 164 · `py_compile` across panel, helper, image engine, storyboard, planner, lora_compat, codec patch · page parse clean. No engine re-pin, no weights change, no registry change.
>
> **🩹 2026-08-14 — v4.0.3: hotfix — picker-LoRA renders crashed with a missing import; cast characters were unaffected.** v4.0.2's native-attach guard (`_guarded_native_attach` in `mlx_warm_helper.py`) calls `mx.eval()`, but this helper imports `mlx.core` lazily inside each function that needs it, and the new closure never got its own import. Every render that stacked a LoRA chosen from the library picker — any style or character file selected by hand — died with `NameError: name 'mx' is not defined`. **A cast character took a different branch and never reached that line, which is why the release's own validation passed and the defect reached users.** The fix is the four lines that add the lazy `import mlx.core as mx` to the closure, in scope, matching the rest of the file (`78be971`). Nothing else changed: same engine pin `v0.14.19+ltx25.4`, same weights, same registry, and no render path other than the crashing one behaves differently. `test_lora_compat.py` 9/9 and the character round-trip gate 27/27 both green at the shipped tree.
>
> **🩹 2026-08-14 — v4.0.2: the release the two owner-reported defects earned, plus a vendored-engine re-pin.** Two things that were reported as "it renders, it is just wrong" turned out to be routing bugs deep enough to need the engine tag moved. Everything else here is the round-4 punch-list from v4.0.1, closed.
>
> - **THE PIN MOVES: `v0.14.19+ltx25.3` → `v0.14.19+ltx25.4`.** One change, and it is the i2v anchor described in the entry below. `_LTX_EXPECTED_VERSION`, `ltx_checkout.sh`, `install.js`, `README.md` and the `CLAUDE.md` pin row move in the same commit, which is what `scripts/check_ltx_pin.js` exists to enforce. The vendored suite is **967 passed, 22 skipped** at the new tag — the same result the old one recorded.
> - **`High · 720p`, and every LTX ETA repriced.** The 1280×704 canvas ships as its own tier rather than as a redefinition of `High`; measured at ~8 min from a real render. See the ETA entry below for why redefining a shipped key was refused.
> - **Characters ride the engine's native runtime-LoRA route.** The panel still installed its historical fusion shim, which intercepted the pipeline load and fused adapters into the QUANTIZED weights before the native `auto → unfused` path could be chosen — the class of bug that erases part of a trained delta while still returning an entirely plausible clip. Every attachment is now verified against the file's own expected module list, and a zero or anomalously partial fuse **refuses with the filename** instead of rendering a stranger. Incompatible files are filtered out of every picker and re-checked at enqueue, so stale browser state and direct API calls cannot bypass the filter. **Trained characters may look slightly different from this release on — closer to their training**, because the delta is no longer being partly quantised away.
> - **`/prompt/enhance` returned an empty body.** `f1d2139`'s version-aware render-encoder seam set `LTX_GEMMA` to the active generation's render text tower (Gemma 4 on 2.5), and `get_gemma_lm()` read the same variable — so the enhancer tried to build a language model out of a tower with no `lm_head` and no KV cache. The doomed load burned the client's 120 s deadline, and the JSON 500 was then written to a socket that had already gone: `BrokenPipeError` while flushing headers, which is exactly what "empty response" looks like from the browser. Now two independent variables (`LTX_GEMMA` for render, `LTX_ENHANCE_GEMMA` for enhancement), an enhance-only 90 s helper deadline that beats the UI's, and malformed helper events converted to a real JSON error instead of raising past the handler's exception boundary.
> - **The trigger guard sees the cast character, and stops repeating itself.** Covered-ness is a property of the TRIGGER, not of a library path: the cast character's own triggers are added to the covered set (the backend expands them server-side, so they are attached by definition), and a trigger warns once regardless of how many library rows carry it.
> - **A saved clip is called what it is called.** `Content-Disposition: inline; filename="<basename>"` on `/file` and `/image`, across 200, 206 and the suffix-range form browsers use to find the moov atom — `curl -OJ` now saves `01_bizarro_closeup.mp4` instead of a file literally named `file`, and byte-range playback is proven unbroken by a two-part range reassembly hashing identical to disk. `/image?w=480` names the original's stem with the SERVED extension, so a re-encoded thumbnail is not saved as `.png`.
> - **Live preview says when it cannot run.** An install whose vendored engine predates the live-preview module produced no `state/live` directory at all, and the panel showed an empty frame forever. The helper now probes the CAPABILITY — does the module import? — rather than comparing version strings, and the panel reports `stale_engine` with no CTA link, because the Update button lives in the Pinokio sidebar and a dead-end link reads as "we handled it". The reachable state that caused it: v3.8.1's in-memory updater has no `post_update.sh` and hardcodes an engine checkout of `871694d`, so one click could move the panel forward and the engine BACKWARD. A second Update self-heals it; nothing used to say so.
> - **The round-4 punch-list from v4.0.1, closed.** The updater's obstruction guard tested `-e`, so an untracked *directory* refused the update outright — fleet-wide update refusal for any release that adds a file under `logs/`, `cache/` or `mlx_models/`; it now refuses only genuine obstructions, preserves untracked files, and says which path stopped it. The character round-trip gate executes the submit half instead of duplicating its expressions. Every Q8 notice resolves the generation actually installed, so a 2.3-pinned install stops being quoted the 2.5 pack's size and name.
> - **The codec gate stopped failing on a healthy tree.** A sidecar with no recorded `output_codec` request is a SKIP, not an assertion — lab and bench harnesses write those, and asserting them against the patched default made the gate exit 1 on a good tree and red-banner every user through the install self-report.
> - **The sample character downloads again**, and `run_panel.sh` looks for `env/` — the venv the installer has always created — instead of the `.venv/` no install on earth has.
>
> **Gates at the promote tree:** `check_pinokio_scripts` PASS (worst dispatch 377/500) · `check_ltx_pin` PASS at `v0.14.19+ltx25.4` · `check_post_update` PASS · `assert_registry` 154 passed · `assert_schedules` 42 passed · `check_output_codec` PASS + 19 self-tests · `test_character_roundtrip` 27 · `test_lora_compat` 9 · storyboard suites 164 · the vendored port suite 967 passed / 22 skipped. Validated with a real q4 draft i2v render through the panel path under both GPU locks: 38.89 s, `ltx_version_match: true`, and the composition holds across the whole clip instead of collapsing after the first step.
>
> **🎬 2026-08-14 — i2v on LTX-2.5 stopped animating your image, and now it does again.** Owner-reported: Image mode with a supplied picture kept the character and the style but produced a NEW COMPOSITION instead of animating the image. Root cause is in the vendored engine's 2.5 sampler, not the panel: the i2v anchor is pinned by `denoise_mask=0`, and the port re-pins the model's **x0 estimate** every step but never re-pins the **sample**. Under Euler that was invisible — velocity is exactly 0 at a pinned token, so the step returns it unchanged. The euler-**ancestral** step 2.5 selects rescales every token by `alpha_next/alpha_down` and adds fresh Gaussian noise, unmasked: measured, the anchor is halved and buried under noise 0.86× its own scale after the FIRST step, so every forward from step 2 on saw garbage where the image should be. The terminal step stamps the image back in, which is why the delivered frame 0 matched while the clip was a different shot. Fix: re-composite the sample against `clean_latent`/`denoise_mask` after the ancestral step, at all three sites in `utils/samplers.py`, guarded by the existing uniform-mask flags. **Receipts** — same image, prompt and seed 424242, frame-vs-input normalised cross-correlation: frame 0 `+0.978 → +0.975` (pinned either way), frame 36 `+0.093 → +0.788`, frame 72 `+0.090 → +0.719`. Blast radius proven zero: t2v at a fixed seed is **sha256-identical** before and after (uniform mask short-circuits the guard), and every Euler lane — 2.3 i2v, keyframe/flf2v with its deliberate eta 0, a2v, extend, restore — never reaches the branch. **SHIPPED in v4.0.2** — the fix lives in the vendored engine, so it required a fork tag: `mrbizarro/ltx-2-mlx` `v0.14.19+ltx25.4` (`ee256f5`), with `_LTX_EXPECTED_VERSION` and every pin reference moved in the same commit. HQ stage 1's `res2s` loop has the same defect class and is left as a separate owner-visible decision because fixing it is not byte-identical on 2.3.
>
> **⏱️ 2026-08-14 — LTX tier ETAs told the truth about a recipe nobody ran; now they tell it about the one everybody runs — and a new tier owns the big canvas.** Owner-reported: "High says ~4 min, takes ~8; Standard says 3, isn't." Root causes, all fixed: (1) **the chip priced a schedule the lane never ran** — High's cell described 8+3 while the `generate_hq` dispatch has always defaulted 10+3 (two independent literals). The dispatch defaults now READ `LTX_HQ_STAGE1/2`, so cell and lane cannot drift again. (2) **`LTX_LOAD_SEC` 10.9 → 31.0**: the old value summed only the load phases the render log prints; a real render also pays helper spawn, job plumbing, encode/mux tail and sidecar write, so every model-priced chip ran ~20 s hot. (3) **the power law under-prices small canvases** (Quick ≈ Balanced at 5 s on M4 Max — a real fact, not a bug). Fix: a three-arm bench through the REAL panel path (isolated dev panel, helper restarted per arm, GPU locks held) — quick 161.8 s, balanced 162.5 s, standard 230.1 s → measured rows for every offered 5s cell, and non-5s lengths now scale the measured 5s anchor by the model's own length ratio instead of trusting its absolute price. **The 1280×704 canvas is a NEW TIER, not a redefinition**: `High · 720p` (`high_720p`), two-stage HQ at 10+3 / CFG 3.0 / TeaCache 1.8, priced from the owner's own 491.03 s panel render and carrying the measured 49.7 GiB / 64 GB requirement. `High` keeps the 1024×576 canvas it has always shipped — redefining a shipped key would have doubled the cost of a tier under every existing user and re-pointed every sidecar ever written at a canvas it was not rendered on. `scripts/assert_registry.py` now pins all five canvases so it cannot recur.
>
> **🎞️ 2026-08-14 — RENDER-LEVEL codec gate: a produced file finally has an assertion on it.** The v3.8.1 lesson closed: every gate was static and fleet-wide 4:2:0 shipped with all of them green. New `scripts/check_output_codec.py` (stdlib-only, importable) ffprobes a real mp4 and fails non-zero unless (a) `pix_fmt` matches what was requested — `--expect` > the sidecar's render-time `output_codec.pix_fmt` > `LTX_OUTPUT_PIX_FMT` > the patched default `yuv444p` — and (b) `+faststart` is present (pure-python moov/mdat atom walk): the unpatched upstream line writes neither, so faststart is the fingerprint that catches the class even when the requested pix_fmt equals upstream's hardcoded yuv420p, where pix_fmt alone is blind. Default target = the newest clip under `mlx_outputs/` whose sidecar actually RECORDS a codec request (H3's mux never went through the patch; lab and bench harnesses write sidecars with no `output_codec` block at all, and asserting those against the patched default made the gate fail on a healthy tree and red-banner users — a sidecar with no recorded request is now a SKIP, not an assertion). When the sidecar names a `native_output` the NATIVE file is checked — the panel-side upscale re-encode always stamps settings-codec + faststart and would mask a broken patch. Wired as a MANDATORY pre-promote step (`docs/RELEASE_CHECKLIST.md` §3a) + a gate-table row in `CLAUDE.md`, and every install self-reports the SAME check — the panel imports the script (no drift possible) into `/status.model_integrity.output_codec`, cached by path+mtime inside the 120 s integrity cache; mismatch = red banner with its own headline and deliberately NO Repair button, since re-downloading weights cannot fix an unapplied patch.
>
> **🧪 2026-08-14 — the graded-pipeline rule is now GATED, per model version.** c366e71 fixed the Characters endpoint's hardcoded `"quality": ["high"]` — the literal that had routed every Characters-tab render on 2.5 onto the two-stage HQ path (~246 s, 29.5 GB add-on) in defiance of `character_render_quality()`'s ruling, because f65ea9b fixed the endpoint's `quality` VARIABLE and missed the job_form three screens below — but nothing pinned it against coming back. `TestPipelineQualityPerVersion` (fa1fd99, in the round-trip gate) runs the REAL `do_POST` over a stub transport and captures the form at `make_job`, the seam the bug lived on: ltx25 submits `balanced`, ltx23 submits `high` — asserted per version by flipping `ACTIVE_MODEL_VERSION`, not just on whichever generation is active today; an explicit caller quality still wins; draft/pro pick a canvas, never a pipeline; stage1/stage2 steps ride only when the caller sent them. Mutation-verified (reintroducing the literal fails 3 of 6); gate 19/19. Also audited on the way in: the job_form's `teacache_thresh 1.8` / `cfg_scale 3.0` defaults are inert on the distilled lane (its dispatch never forwards them to the helper) and byte-equal to `make_job`'s own defaults, so they stay.
>
> **🩹 2026-08-14 — v4.0.1: the point release an external review earned.** Four review rounds over the v4.0.0 tree, each one probing the code rather than reading it. Nothing here is a new feature; every line is a thing v4.0.0 got wrong.
>
> - **THE UPDATER IS TRANSACTIONAL, AND IT WAS NOT.** The fetch result was ignored and every pull failure went to `git reset --hard $UPSTREAM` — a LOCAL tracking ref, so when the fetch was the thing that failed, the reset landed on the stale commit already checked out, exited 0, and `post_update.sh` then ran against the old tree. Update reported success and nothing had changed. Now: a failed fetch is fatal, divergence is PROVEN by `git rev-list --count $U..HEAD` rather than inferred from whatever made the merge fail, a dirty worktree stops the run instead of being destroyed, and an untracked file that upstream now tracks is named and refused rather than deleted as an "obstruction". Measured on real repositories: offline → exit 1, HEAD unmoved · fast-forward → converges · dirty tracked file → edit preserved · untracked collision → path named, file intact.
> - **The Update also never repaired the known-broken transformers.** `install.js` had promised since 2026-07-10 that "uv downgrades an already-installed 5.13.0 on the next Update"; nothing in the update path ever constrained transformers, so that sentence was false for a month and an install stuck on 5.13.0 could click Update, watch it succeed, and still not render a frame. `post_update.sh` step 2b is a fatal `require` on the same uv resolve as the mlx trio. Proven in a throwaway venv seeded with the broken version.
> - **The HQ tier never had a live preview at all.** `high.preview_every=2` promised a lane `generate_hq` never wired, on both sides — and `_build_live_preview` raised into its own never-fatal `except` and returned `None`, so the one component whose job is to stop things silently not happening was silently not happening. Real HQ render: 14 PNGs published every-2. The byte-identity invariant holds on the new lane — preview ON and OFF give the same sha256.
> - **The storyboard planner's laws are composable now.** Law enforcement ran first and premise repair ran last, so a repair could hand back a shot with the premise restored AND an unwritten speech act attached, under a warning saying the premise was back. Re-rolls spliced their fix into the ORIGINAL plan, so fixing shot 2 threw away shot 1's fix. And the appearance law generalised itself into never naming a species at all — the owner's "humanoid animals" film came back with twelve shots of ordinary humans. Fixed as a class: a repair that breaks a law is rejected, the final invariant scan runs after the LAST mutation whichever pass made it, every premise term is required (capped by a presence budget — a five-species brief cannot fill a two-shot film), and a degraded plan says `degraded` out loud instead of returning `ok: true`.
> - **Characters: displayed = submitted = restored.** The two surfaces generated different jobs for the same character (face 0.8 here, 1.0 there, voice never submitted at all), Load Params knew only the retired 736×416 so every Draft reopened as Pro, and the voice was never restored. One contract now, and the gate EXECUTES the panel's own JavaScript against a DOM shim — `scripts/extract_panel_js.py` pulls the real functions out of the panel and raises on a rename, so coverage cannot decay back to grepping. Its first version passed by string-matching while the live loader was broken.
> - **Readiness is declared per capability, not per file label.** `kind: "base"` had come to mean both "fetched on a fresh install" and "the panel cannot render without it". Gemma 3 is the first and not the second, so a half-downloaded PLANNER model hid Start on an install that renders perfectly. `required_files.json` now declares what `render` / `enhance` / `characters` / `high_tier` / `live_preview` / `h3` each need, and both the panel and `pinokio.js` read it.
> - **Every offer follows the generation actually installed.** A 2.3-pinned install was offered the 30 GB 2.5 Q8 pack it does not load; the H3 entry disagreed with the panel about split model roots and offered a 75 GB install for weights already on disk; the missing-decoder notice fired over H3 renders, image jobs and training runs with an Install link for a decoder they never use. All three resolve the active version — `pinokio.js` reads `LTX_MODEL_VERSION` out of the `ENVIRONMENT` file the launcher sources.
> - **Copy and store listing.** The Pinokio store listing still said LTX 2.3 three weeks after 2.5 shipped. H3 is described as a peer of LTX everywhere rather than a lesser tier, a missing engine is an offer rather than a downgrade, and `docs/H3_ENGINE.md` says to put persistent overrides in `ENVIRONMENT` — the shell `export` that "loses" a working H3 install on the next restart.
>
> **Known-remaining, carried to v4.0.2** (round-4 review, all narrow, all with fixes in progress): the updater's obstruction set matches exact untracked leaf paths, so a parent-directory collision or a gitignored file (`ENVIRONMENT`) can still be cleared by a legitimate reset after genuine divergence; the Character gate duplicates the submit expressions rather than executing that half; and an LTX-2.3 Q8 click still receives the 2.5 pack's size and name in its notification text (the download itself is correct).

> **🎬 2026-08-13 — v4.0 INTEGRATION: LTX-2.5 becomes the generation the panel SERVES, not just the one it loads.** The registry has carried 2.5 since 2026-08-12 and the weights have been mirrored since the same day; what was missing was the product. The panel still called the engine "LTX-2.3", still priced tiers with two hand-rolled subtitle writers printing 2.3-era minutes, still offered a duration one lane silently rounded away, and had no way to say "you can have the good tier, it costs 29.5 GB, here is the button." The FROM-ZERO gate — a fresh Pinokio install rendering one clip — was the last blocker before public promote, and it passed.
>
> - **The switcher says `LTX 2.5`, and the generation line is DERIVED.** The label is the family name; the build is its own span, resolved in `engines_payload()` from the active version. A user pinned back with `LTX_MODEL_VERSION=ltx23` reads `LTX 2.3`; the day 2.6 lands neither string is edited. A static `"LTX-2.5"` would have been the same bug one generation later.
> - **A tier table, so every LTX estimate is measured or says it isn't.** LTX gets H3's two-axis quality × length grammar — the same functions generalised, not forked, with the engine argument defaulted to `'h3'` so all **560** H3 chips are byte-identical across the change (sha256 `679937ef…`). The cost model is fitted to two renders made that day and **cross-validated**: pricing `balanced/5s/q8` — a cell neither anchor is — gives 142.8 s against the 139.4 s measured, 2.4 % out. Three measured rows, not twenty invented ones. Three subtitle writers deleted or neutered.
> - **The anti-mosaic preflight had been DEAD since 2.5 became the default** (`_canonical_layout()` compared MODEL_ID against the *active* generation's q4 path). The guard written after the June-2026 rainbow-mosaic investigation was itself switched off in the field. Fixed and proven with the bug's own repro — withhold `spatial_upscaler_x2_v1_1.safetensors` and the refusal names the file — plus a **pre-render** layer that disables Generate before Gemma loads instead of failing 30 s in.
> - **The 7-second lie, and the three other duration tables behind it.** A shot set to 7 s on H3 snapped to 5 s (124 frames) while the select still read "7 s" and the estimate agreed with the select. Every menu now reads the engine's own table; an off-axis board round-trips with `7 s · nearest 5s`, and H3's 15 s is reachable for the first time.
> - **2.5 characters were routed onto an ungraded pipeline, and then could not render at all.** `balanced_q8_fast` (a 2.3 speed optimisation, no version gate) rerouted Balanced → `high`, which demanded the 29.5 GB HQ add-on; and the panel's LoRA-fusion patch built its DiT from `LTXModel()` **defaults**, so every 2.5 render with a LoRA attached died on `Received 1 parameters not in model: keyframes_abs_pos_embedding`. Both fixed; characters now render q8 + distilled — the recipe every graded 2.5 clip ran — proven end to end **with the add-on withheld from disk**, clean identity at frame 60.
> - **The voice gets its own strength, and the graded default is `1.0`.** The face file's audio-branch deltas are noise and at equal strength they are LOUDER than the voice file's signal (median ‖D‖ 1.45 vs 1.10) — which is the argument for giving the voice its own number. The measurement then argued for turning it up (parity 1.2, headroom at 1.4) and **the owner's ear rejected that: 1.0 vs 1.4 graded side by side, verdict "candidate seems less good".** So `1.0/1.0` is the default and it is settled, not provisional. The SPLIT CONTROL ships and the help-dot explains the mechanism; what it no longer does is recommend climbing the ladder. `character_voice_strength` is asserted into `params` in the same commit that adds it — the allowlist trap, closed with a test rather than a hope.
> - **Live preview + Stop early.** `--live-preview tae` with the 22 MB decoder now shipping as an ordinary `github-release` registry row. Preview on vs off is **byte-identical** (`79ec11a7…` both) with the preview demonstrably running; an abort exits **75**, writes no mp4, and resolves as `status: "stopped"` — muted card, "Nothing was saved" — not a failure. The `meaningful` gate is server-owned because the rule differs per pipeline (estimate 6 on distilled, 2 on res_2s); a Stop button over a noise field aborts takes that were going to be fine.
> - **Settings → Storage**, measured by walking the real directories rather than reading `size_gb`: LTX-2.3 57.8 GB, the High add-on 29.5 GB (sized from its two filenames — a directory walk there deletes the q8 pack), Gemma 3 8.1 GB shown greyed with *why it cannot go*. `/models/remove` refuses the active model, an in-flight folder and an unregistered key, and never takes a path from the client.
> - **Both foundation defects closed**: `assert_registry` is **113 passed, 0 failed, 0 known defects pinned** — the first honestly green run since the markers went in.
>
> **FROM-ZERO: PASSED.** A fresh Pinokio install, from an empty directory, through the install lane and out the other side rendering a clip. That gate blocks public promote and nothing substitutes for it.

> **📦 2026-08-12 — THE SHIP BLOCKER IS CLOSED: the LTX-2.5 packs are published, and a fresh install can now fetch the default generation.** LTX-2.5 became the default the same morning, while `install.js` still downloaded only 2.3 — `q4_25` / `gemma4_25` named `mrbizarro/...` HuggingFace repos **that do not exist and never will** (our token is read-only, and these packs are our own quantisation of a gated upstream). On the box that built them the default lane was correct; on a **fresh install there were no weights at all for the generation the panel boots into**, and nothing in the product would have said so until the first render failed. `ltx25_pin_move.md` §8.1 called it the one blocker; this is it, closed.
>
> - **Mirrored as GitHub release assets** — `weights-ltx25-v1`, tagged on **public `main` d4c6ac5** (v3.7.0's release commit, `--latest=false` so it does not displace v3.7.0), the same lane the `bizarrotrn_v2` sample character has taken since July. `q4_25` (20.74 GB) + `gemma4_25` (6.73 GB).
> - **A release asset is capped at 2 GiB and the transformer is 11.32 GB**, so anything over the cap is published as ordered 1.9 GB shards. `scripts/publish_pack_release.py` writes, uploads and **deletes one shard at a time** — publishing a 21 GB pack costs 1.9 GB of scratch, not a second copy of the pack, on machines already tight enough that we quantised the model for them. Uploads resume by skipping assets already on the release; the **manifest goes up last**, so a half-published release cannot be consumed.
> - **`scripts/fetch_pack_release.py` is the consumer, and it is stdlib-only** because it runs from `install.js` before anything optional exists and from inside the panel. Every shard is checked against its own sha256 as it lands, the reassembled file against the file sha256, and a file is renamed into place **only after that second check** — so a killed download never leaves something the loader would pick up. Resume works at two levels: an HTTP Range request inside a shard, and a sidecar at shard granularity that is trusted **only when the partial's length agrees with it**.
> - **The manifest is NOT a second format.** It is the `phosphene_quant_manifest.json` `scripts/quantize_ltx.py` already emits, plus a `release` block and a `shards` list per file; `files: {name: {bytes, sha256}}` is preserved exactly, so the panel's existing `_manifest_meta()` deep-verify reader consumes either.
> - **A real finding: the in-pack quant manifest is stale for two sidecars.** `embedded_config.json` (says 2,557 B and config.json's hash; is 7,589 B) and `split_model.json` (says 302 B; is 305 B) were rewritten by the conversion step *after* the quantiser wrote its manifest. **Every `.safetensors` hash matches**, which is why deep-verify never noticed — it only hashes safetensors. The publisher publishes the **files'** hashes, records the disagreement in `quant_manifest_drift`, and does not "fix" the quantiser's deterministic output to hide it.
> - **Licence, as files rather than intentions.** These weights are a Derivative under §1.5 of the **LTX-2.x Community License Agreement**, redistributable under §3 on conditions 3.1–3.5. The complete Agreement (Attachment A and all, fetched verbatim from `Lightricks/LTX-2`) and a `NOTICE.md` enumerating exactly what we changed — layout rewrite, 4/8-bit quantisation of the DiT block linears at group size 64, a provenance metadata key, generated sidecar configs, a pure byte split for transport — and what we did **not** (no training, no distillation, nothing removed or circumvented from the Licensor's safety or use restrictions) ship as release assets **and are fetched into the pack directory**, so the terms land on the user's disk next to the weights. The §3.5 Commercial-Entity notice ($10M revenue → paid licence from Lightricks) is stated in both. No territory restriction exists in this licence; that was H3's problem, not this one.
> - **The panel got a second download lane, not a second downloader.** `_download_thread` picks its command from the registry — a repo with a `mirror` block spawns the fetcher, everything else spawns `hf` exactly as before. The retry loop, the character-at-a-time log streaming, the single global download slot and cancel-by-process-group are all shared. Repair works unchanged on the new lane, and the missing-`hf` guard no longer blocks a pack that never needed hf.
> - **`install.js` fails loud** if the 2.5 fetch fails (it is the default lane, and a panel that cannot render is not a successful install); **`update.js` is best-effort** (an Update must not be brickable, and the Models page now retries through the same fetcher). `pinokio.js` needed no change: `baseRepos` already filters `kind === "base"`, so an install without 2.5 weights already renders **Resume Install** — the self-heal path was always there, it just had nothing to fetch.
> - **FROM ZERO, against the real release** — the gate this ships on. Both packs pulled into an **empty directory** with the exact command `install.js` runs, against the real release. `q4_25`: 19 files, 20.74 GB. `gemma4_25`: 11 files, 6.73 GB. Every shard and every reassembled file sha256-verified, `--check-only` green on both afterwards. The reassembled 11,320,074,467-byte transformer and 6,344,495,432-byte connector hash **identical to the packs that were published** (`7c8c4f71a4cf…`, `120b43898339…`). The transformer then went through the **vendored loader** — `load_split_safetensors` → `apply_quantization` → `load_weights` → `mx.eval`, the production path — and came back `loaded: true, n_weights: 7355, loader_detected_bits: 4` in 7.6 s. sha256 proves the bytes arrived; only that proves they are a model. Separately, the published manifests were checked against the **real on-disk packs** and matched every file, which is the assertion that the mirror is the pack and not a copy of it. **Daemon :8199 never touched** — PID 58950, 10 h 51 m uptime, answering 200 before, during and after.
> - **16 tests**, `scripts/test_pack_release.py`, publisher and fetcher exercised together against a loopback HTTP server through the same code paths a real publish and a real fresh install take. A corrupted shard never reaches the pack **and no longer abandons the rest of it** (that one was found by the test and fixed in the fetcher); an interrupted download resumes instead of restarting; a complete pack is left alone; a pack missing a mandatory file cannot be published at all.
> - **Still open: `q8_25` is deliberately NOT mirrored.** `e870061` added `transformer-dev.safetensors` and `ltx-2.5-22b-distilled-lora-450.safetensors` to it (30 → 58 GB) and neither exists on disk yet. A release cut from the old file list would produce a pack the panel reports incomplete on arrival — the June mechanism verbatim — so `q8_25` gets no `mirror` block until the pack is complete. The publisher would refuse anyway; that refusal is the point.
>
> **SAME DAY, PHASES 2 + 3 — THE MIRROR IS COMPLETE: 89 assets, 86.99 GB, q4 + gemma4 + q8 + hq, all un-gated.** Owner's call was ship q4+q8 now and let High follow, which required a split: `e870061` had listed `transformer-dev.safetensors` and `ltx-2.5-22b-distilled-lora-450.safetensors` inside `q8_25`, so a pack that was **complete on disk, gates passed, 30 GB** was both unpublishable and un-installable while a 42 GB download it did not contain finished. Those two files became their own download unit, `hq_25`, sharing `local_dir` with the q8 pack because they are loaded out of it **by name** — two entries, one directory, on purpose: an entry is a *download unit*, not a directory.
>
> - **The gate e870061 bought was moved, not given back.** Splitting the file list without moving the gate would have restored the exact bug it fixed, so the four two-stage gates now ask `hq_surface_missing()` (the union of pack and add-on) and still refuse by the same two filenames; `q8_available` — which is what the UI reads to enable High / Extend / Keyframe / FFLF — now means "that surface can run", or the wave-through simply relocates from the job gate to the pill. `balanced_q8_fast` deliberately still asks `q8_missing_files()`: Q8 Fast runs the distilled transformer and never touches the dev one. **Inert for 2.3 by construction** (it declares no `hq_addon_repo_key`), verified by calling the real functions against the real registry.
> - **A DOWNLOAD UNIT IS NOT A DIRECTORY — caught mid-upload.** `pack_files()` published every file in the pack directory, so the moment the add-on's weights landed in the q8 directory the *running* q8 publish began uploading 29 GB of add-on under a `q8_25__` prefix — silently restoring the coupling the split existed to remove. Killed it, deleted three stray assets; nothing was ever advertised because the manifest uploads last. Now a host publishes its directory minus anything another entry with the same `local_dir` declares, and a guest declares `publish_scope: "files"` and cannot sweep up its host's sidecars. Both directions tested.
> - **Resume could skip a file whose content changed.** GitHub publishes no asset checksum, so resume compares sizes — blind to same-size content changes. Not hypothetical: the in-pack `phosphene_quant_manifest.json` was rewritten by the dev-pack agent between two publishes (gaining the sha256 rows that let deep-verify check both new files). Anything ≤ 16 MB is now always re-uploaded; big shards keep the size skip, where such a change still fails loudly on the fetcher's sha256 instead of installing.
> - **Also survived, in production:** a real `dial tcp: lookup uploads.github.com: i/o timeout` that killed the first q8 run on its last file (re-running skipped ~29 GB and finished the tail), and free space hitting **322 MiB** mid-upload while the dev build churned — the upload lived because its footprint is one shard.
> - **FROM ZERO, all four lanes, byte-identical to source:** q4 `7c8c4f71…`, gemma `a3e8b162…`, q8 `36a346c1…`, dev `90b7e01d…` + LoRA `86370bbf…`. `loader_detected_bits` came back **4** for q4 and **8** for q8 and dev, so each lane is the pack it claims to be. **The split is proven, not asserted:** the q8 lane fetched into an empty directory contains no add-on file — a 30 GB q8 install, not a 59 GB one. Daemon :8199 untouched throughout (PID 58950, 14 h 09 m). Final free space 66 GiB.

> **🌍 2026-08-12 — THE GEOIP KILL (analytics schema-v2 Phase 1): every event now tells PostHog not to locate it.** The panel has never sent a location field. It didn't have to: PostHog derives country, city, subdivision, timezone and the city's coordinates from the connecting IP by **default**, and the previous posture — in `_analytics_post`'s own docstring — was *"we neither add to nor suppress that."* Now suppressed, on every event, via `_ANALYTICS_RECEIVER_DIRECTIVES` spread **last** into the properties block so no call site can override it.
>
> - **`$geoip_disable: True` — real, and the only payload-level mechanism there is.** Verified against PostHog's own ingest source rather than assumed: the GeoIP transformation's first statement is `if (event.properties?.$geoip_disable or empty(event.properties?.$ip))`, and `$geoip_disable` is the same property `posthog-python` sets for `disable_geoip=True` (it is on that SDK's `$`-property allowlist).
> - **`$ip: None` — THE SPEC WAS WRONG, and it is the finding worth remembering.** The spec (§2.8) and the Phase 0 handoff both prescribed `"$ip": None`. **It does nothing.** PostHog's ingest fills the property from the socket with `if (!properties['$ip'] && event.ip) properties['$ip'] = event.ip` — `null` is *falsy*, so it is the default with extra steps and the real address lands on the event anyway. There is also **no top-level `ip` field** to set: `RawEvent` in `rust/common/types/src/event.rs` has `token / distinct_id / uuid / event / properties / timestamp / offset / $set / $set_once` and nothing else, so an `"ip"` key beside `"event"` is deserialised into the void. The working version is a **truthy** placeholder: `ANALYTICS_IP_PLACEHOLDER = "0.0.0.0"`, which occupies the property so the connecting address is never written onto the stored event.
> - **Why `0.0.0.0` and not `127.0.0.1`.** The GeoIP transformation rewrites loopback and `192.168.*` to a real address in Sweden as a local-dev convenience — a loopback placeholder would *manufacture* a location the day the disable flag went missing. The constant carries that reasoning inline so nobody "tidies" it.
> - **What is NOT claimed anywhere.** The request still arrives over TCP from a real address and no field inside a body changes that. `docs/ANALYTICS.md` now has a **Location** section that says exactly that, in those words, and stops: the flags control what the receiver is instructed to *derive* and *store*, and discarding it at the edge is a project-side setting this source tree cannot promise. Phase 0's lesson — never document more than the code does — applied to its own follow-up.
> - **Suite 38 → 43 GREEN, and it fires EVERY event type.** New `TestReceiverDirectives`: all three directives intact on `app_installed` / `app_boot` / `pack_state_change` / `render_completed` / `render_failed`, with a coverage guard that re-reads `_analytics_capture("…")` call sites out of the source so a sixth event type cannot be added and go unchecked; no `$geoip_*` key but the disable flag and no `country`/`city`/`latitude`/`timezone` substring anywhere in a body; the placeholder is truthy and is neither trap value; a call site passing `$geoip_disable: False` loses; and the local mirror stays free of transport plumbing. `props_of()` now fails on any **undeclared** `$` key rather than ignoring it.
> - **Mutation-tested — 7/7 red.** Drop `$geoip_disable`; `$ip = None`; `$ip = "127.0.0.1"`; drop `$ip`; drop `$process_person_profile`; spread the directives FIRST so props win; smuggle a `$geoip_country_code` onto a payload. Each turns the matching test red and the suite restores clean.
> - **Owner action, deliberately not claimed as done:** PostHog project settings → *IP data capture* → **discard** (org-level default also exists), so the promise survives a payload flag ever being missed; and eyeball one live event for absent `$geoip_*`. Nothing in the repo asserts either.
> - **Still open** (schema v2): `error_class` replacing free-text `error_signature` (migrate `top_errors` in the same commit), version/chip context on render events, `update_state`, `wall_sec_bucket`, Phases 5–8.

> **🔦 2026-08-12 — ANALYTICS TRUTH PASS: the transparency doc had been lying for three days, and the guard suite that was supposed to stop that was RED.** No behaviour changed — this is docs, comments and tests only. `acfbdc7` (2026-08-09) committed the live `phc_` project key on purpose and correctly; what it did not do was update the page whose entire value is that a suspicious user can check it. So `docs/ANALYTICS.md` still promised *"The public repo ships with no key, so a fresh clone sends nothing until its owner configures one"* while every fresh clone reported from first boot. **README.md was worse** — "No telemetry", twice, on the front page, where a reader who never opens `ANALYTICS.md` would stop.
>
> - **The claim is extinct repo-wide** (11 sites): `docs/ANALYTICS.md` (the short-version paragraph, the owner-setup section, the design note), `README.md` ×2, seven comment/copy sites in `mlx_ltx_panel.py` — the module header's `INERT BY DEFAULT IN THE PUBLIC TREE` bullet, `_analytics_key()`'s docstring, `_analytics_enabled()`'s docstring, both settings-default comments, `update_settings()`'s "returns the panel to fully-inert", the `_analytics_post` early return, the `__main__` call-site — plus the Settings modal `<details>` copy. A grep sweep for the claim's whole vocabulary (`inert`, `no key`, `opens no`, `no socket`, `sends nothing`, `no telemetry`, `public tree`) returns only true statements and the one paragraph that describes the correction.
> - **A SECOND false claim was nearly shipped, and is the finding worth remembering.** The analytics spec's own replacement draft said *"Clear the key in Settings and the panel opens no socket at all."* **It doesn't.** `_analytics_key()` resolves `saved or ANALYTICS_KEY_DEFAULT`, so an empty Settings field is *no override*, not *off*, and a user who "cleared the key" is still reporting. Verified empirically before anything was written (sockets counted against a spied `urlopen`). Every surface now says the same thing: **the toggle and `PHOSPHENE_ANALYTICS_DISABLED=1` are the off switches, and they are the only two.** If that fallback is ever changed to make an empty field mean off, the doc changes in the same commit — there is now a test pinning it.
> - **`app_installed` documented.** It has shipped and fired since the module landed, and `ANALYTICS.md` never named it — a bug by that page's own opening rule. Fields (`version`, `chip_family`, `ram_gb`), why it exists (it is the denominator; without it "new installs over time" tempts someone to profile ids), and the once-ever mechanism (`analytics_install_reported`).
> - **Suite RED → 38/38 GREEN, rewritten not deleted.** `test_shipped_default_key_is_empty` → `test_shipped_key_is_a_write_only_project_key` (the real invariant was never "no key", it was "no READ-capable key" — `phx_`/`phs_` must never ship). `test_no_key_means_no_socket` → `test_opt_out_is_the_toggle_not_the_key`. The third failure was a **stale fixture with correct new behaviour**: `h3_tier: "5s"` is a LENGTH, never a tier key, so `"unknown"` is right — fixture moved to the real legacy key `hq_5s` → `standard_5s`, with `"5s"` kept as the worked example in a new closed-vocabulary test. Added `test_app_installed_fields` and a **documentation-parity test**: every event name `_analytics_capture()` can be called with must appear in `ANALYTICS.md`, so the next undocumented event fails the build instead of shipping.
> - **Validated:** `py_compile`; the analytics settings-section re-parsed out of the served HTML with a tag checker (balanced, `details`/`summary` intact); 38/38 suite; and the new guards **mutation-tested** — blank the key, ship a `phx_` key, ignore the toggle, undocument an event, break legacy tier resolution, re-fire `app_installed`: **9/9 turn the matching test red.** A guard that cannot fail is how `acfbdc7` got through.
> - **NOT done here, still open** (schema v2): `$geoip_disable` + `$ip: null` on the payload (GeoIP is still on by default) — ⚠️ **SUPERSEDED 2026-08-12 by the GeoIP-kill entry above; and `$ip: null` as written here is a no-op, see there for what actually works**, `error_class` replacing free-text `error_signature`, version/chip context on render events, `update_state`, `wall_sec_bucket`. Phase 0 was shipped alone deliberately — the doc had to stop being false before anything else was worth arguing about.

> **🎬 2026-08-07 — PER-WINDOW PROMPTS: the "10s just repeats after 5 seconds" report, fixed (dev only, not shipped).** Reported in public on v3.6.0 by **@Wizard_1981**: 10 s and 15 s clips *"just repeat immediately after 5 seconds."* The owner's instruction was direct — *"fix it and ship, make it user friendly, add ? if you need tooltip."*
>
> **The frames were never duplicated.** A 10 s render was checked frame by frame: the background progresses continuously across the seam, every frame is unique. What repeated was the **action** — a 10 s clip is two chained 5 s windows, a 15 s clip is three, and **every window was handed the same prompt**, so a DISCRETE action ("he raises his arm", a scripted line) is performed again in window 2 because that is exactly what window 2 was asked for. A CONTINUOUS action ("he walks") carries on fine. Our own chained-tier note has said this in words since v3.4.1 — and `scripts/generate_staged.py` has taken **`--chain-prompts`** the whole time. Nothing was missing but the UI.
>
> **The control.** A fourth H3 control, **Per-window prompts**, between the Length strip and Speed — the same `.cz-control` grammar as Speed / Steps / Export, with the page's existing `.toggle-pill` ("One line per window") and a **`?`** that reveals the explanation in place. It appears **only when the selected Length is chained** and follows the Length strip live: 10 s → two labelled textareas (*Window 1 · 0–5s*, *Window 2 · 5–10s*), 15 s → three. **The default is untouched** — off posts nothing and every window gets the main prompt, byte-for-byte what shipped. On seeds every box with the main prompt so the user EDITS a shot list instead of facing blank boxes; a box left **empty falls back to the main prompt** at render time, never to an empty string (the runner refuses one).
>
> **`?` is the panel's first help affordance.** Every explanation in this page lived in a `title=` tooltip — invisible on a trackpad, unreachable on a touch screen. `.help-dot` is one 15 px round button that toggles a sentence in place with `aria-expanded`; the copy is **Python-owned** (`H3_CHAIN_PROMPT_HELP`, shipped on `/status.h3.chain_prompt_help`) so it can never drift from the mechanism it describes.
>
> **Wiring, with the two traps this file already documents.** (1) **`h3_chain_prompts` is in the `make_job` allowlist** — one field, a JSON array of strings, because the window count follows the Length axis and a fixed set of field names would either cap the feature or drift from the table. Normalised server-side **after** the tier fallback settles: never more prompts than windows, never fewer without padding, `[]` on every non-chained length and on the LTX lane (so a 15 s request that degraded to 5 s cannot send three prompts into a one-window job). (2) **`--chain-prompts` is PROBED on the installed runner**, not assumed — it landed on `codex/h3-engine` *after* chaining did, so a pack exists that renders 10 s happily and would die on an argparse error 30 s in. `h3_supports_chain_prompts()` is a second probe alongside `h3_supports_chain()`; when it is false the control is hidden, `make_job` drops the list with a sentence, and `h3_visible_tiers()` swaps `H3_TIER_CHAIN_NOTE_LEGACY` back onto the chained cells so that user keeps the honest warning instead of being sent looking for a control they don't have. Also: the runner treats the positional prompt and `--chain-prompts` as **mutually exclusive**, so the shot-list path drops the positional argument entirely; the list goes over as a **`.json` file** rather than a `' ||| '` argv string, which is what keeps a prompt containing `|||` from breaking the render. Sidecar carries the RAW list in `params.h3_chain_prompts` (blanks and all, so Load Params restores the FORM) and the RESOLVED list in `h3.chain_prompts` (so the ⓘ modal shows what each window was actually asked for, in a new **Window prompts** row). Draft→Finish carries the shot list over — same length means the same window count, so entry *i* still means window *i*.
>
> **Validated statically — no GPU, no panel restart** (the owner held :8199): `py_compile`; the inline `<script>` regex-extracted → `node --check`; **99 Python assertions** unit-calling the REAL `make_job` / `h3_normalize_chain_prompts` / `h3_visible_tiers` / `h3_status`; **33 argv assertions** reaching the REAL `run_h3_job_inner`'s `subprocess.Popen` with a captured argv — asserting the positional prompt is dropped exactly when `--chain-prompts` is present, that a blank window resolves to the main prompt, that an all-identical list takes the DEFAULT path, that an older pack degrades instead of failing — and then feeding the file the panel wrote to the REAL runner's own `parse_chain_prompts`, which accepts it and raises on a count mismatch; **62 JS assertions** running the REAL panel functions, extracted verbatim from the inline script, against the REAL `/status.h3` payload in a stubbed DOM (10 s → 2 boxes, 15 s → 3, 5 s hides the control, 10 s→15 s keeps both beats and seeds the third, 15 s→10 s never posts a third, hostile markup cannot break out of a textarea).
>
> **Remaining gate: one live click-through** on a box with the H3 pack installed — expand the control on a 10 s render, write two beats, confirm the `[h3] window 1/2` / `2/2` lines in the log and a real two-beat clip.
> **🧭 2026-08-06 — H3's controls become QUALITY × LENGTH (dev only, not shipped).** The owner's report, verbatim: *"The things like draft HQ, H3 tier, all that should be more like presets: draft quality, high quality, medium quality… but the user should be able to decide if he wants to make 5s or 10s and which tier. You took away all the options from the user by creating this cooked version — somebody may want to run the HQ 5s version for 10s, and it's now not possible from the UI."* He was right, and the matrix was worse than "mostly empty": our best canvas, 1024×576, the ONLY exact 16:9 H3 can serve, could produce a 5-second clip and nothing else. There was never a technical reason — chained windows keep memory flat per window (13,580 rows/window at 768×448, 40.2 GiB peak at 10 s and at 15 s alike), so every canvas reaches every length; the six fixed tiers were just six points someone had happened to measure.
>
> **Two independent axes replace the tier strip.** QUALITY picks the canvas — **Draft** 640×384 (5:3) · **Standard** 768×448 (12:7) · **High** 1024×576 (16:9) · **Native** 1344×768 (7:4). LENGTH picks the duration — **3s · 5s · 10s · 15s** (73 / 124 / 243 / 362 frames, all on the 17n+5 grid; anything past 5 s is N chained 5 s windows). 16 offered cells where there were 6, and every cell is reachable in two clicks.
>
> **Naming is a correction, not a coat of paint.** The old table called 768×448 **"HQ"** while delivering 0.34 MP — the owner hit that lie the same day. The presets are now named for where they sit on the ladder (Draft → Standard → High → Native), and no preset makes a quality claim its pixels cannot cash.
>
> **NATIVE 1344×768 SHIPS, because Turbo changed the verdict.** It sat out of the table for one reason: 44:51 for a 5 s window, which the owner looked at and passed on. Measured 2026-08-06 (fur-coat protocol, seed 161616, `codex/opt_out/nativeturbo/nt.log`): 3 Turbo forwards at 329.1 / 333.5 s, 38.1 GiB active, **19.9 min end to end, R2-class**. The picture he rejected at 44:51 now costs roughly what 1024×576 costs *without* Turbo. It is the model's own canvas — what `resolve_canvas_size` picks for itself at H3's 1.03 MP clamp, and the ceiling of these 768p-class open weights. It is **7:4, not 16:9**, so it letterboxes on export and the export note says so.
>
> **The ETA is COMPUTED per combination, not looked up** — it has to be: 16 cells, only 7 of them ever rendered end to end. Three measured pieces: (1) packed rows are exactly linear in pixels × latent frames, calibrated on the two counts the campaign measured (22,923 at 1024×576/124f, 13,580 at 768×448/124f) and reproducing the other two it published to the digit; (2) seconds-per-forward is a power law in rows, exponent **1.665** fitted to the two ENDS of the measured range (126.0 s at 1024×576, 315.0 s at 1344×768) — which then predicts every other measured point without being told about it (768×448 → 8.6 min for 5 s vs 9.1 measured; 640×384 → 3.0 min for 3 s vs ~3; 10 s chain → 17.2 vs 17:05; 15 s chain → 25.8 vs 26:34; native 5 s → 45.3 vs 44:51); (3) fixed cost per window = a constant staged load (~40 s) plus a VAE decode that scales with decoded pixels — replacing the old flat 2 min/window, which was right at the delivery canvas and much too heavy at Draft. Chained lengths pay the fixed cost PER WINDOW. Where an end-to-end wall clock exists it WINS over the model and the cell is flagged `eta_measured` / `turbo_measured`; **Turbo is now measured at BOTH top canvases** (High 8.5 min, Native 19.9 min), so the derivation only fills the cheap half of the table.
>
> **Compatibility — `h3_tier` remains the wire format.** It is what every sidecar carries, what Load Params / Draft→Finish / list_outputs / the analytics event read; changing it would strand every clip ever rendered. Keys are now composite (`high_10s`), and `H3_TIER_ALIASES` maps every legacy key: `draft_3s`→`draft_3s`, `hq_3s`→`standard_3s`, `hq_5s`→`standard_5s`, `wide_5s`→`high_5s`, `long_10s`→`standard_10s`, `long_15s`→`standard_15s`, `long_10s_dense`→`standard_10s_dense`, `wide_draft_3s`→`preview_3s`. `h3_quality` + `h3_length` ride alongside as first-class fields and WIN server-side when both are valid; **all three are in the make_job allowlist** (the known trap).
>
> **Draft→Finish now means "same length, HIGHER QUALITY".** Under fixed tiers it picked another tier, and a tier bundled canvas with duration — so finishing a 3 s draft at "HQ 5s" silently made the clip two seconds longer, and finishing a 10 s draft was not expressible at all. The picker now lists the canvases above the clip's own, at the clip's own length; the default is ONE rung up (the same cheap-next-step instinct the old `hq_5s` default encoded), and it is offered on any clip that has a rung above it — a Standard 10 s clip can now be committed to High 10 s or Native 10 s.
>
> **Gating preserved and improved.** 10 s / 15 s still need `--chain-windows` on the INSTALLED runner. Previously those cells were DROPPED from the strip when the flag was absent, so the user had no way to learn a pack update would bring them back; they are now shown greyed with the reason, and `make_job` falls back along the length axis while KEEPING the canvas the user chose (the old fallback jumped to `hq_5s` by name, changing the canvas out from under them). The chained-window artefact note fires at ANY quality now, and composes with the Draft note. `LTX_H3_DENSE_10S` survives as an env-gated fifth LENGTH, capped to the canvases where a single 243f pass actually fits in 64 GB.
>
> **Validated statically — no GPU, no panel restart** (the owner was rendering on :8199 throughout): `py_compile`; the inline `<script>` regex-extracted → `node --check`; **379 Python assertions** unit-calling the REAL `make_job` and the REAL cost/tier helpers via importlib (387 with both lab flags on); **120 JS assertions** running the REAL panel functions, extracted verbatim from the inline script, against the REAL `/status` payload in a stubbed DOM. Every legacy key asserted to resolve to the right geometry; every cell asserted to round-trip through `make_job`; the no-chain fallback asserted to keep the canvas.
>
> **📯 2026-08-03 — v3.4.0 SHIPPED PUBLIC: the Hailuo H3 engine.** Public `origin/main` `0ef1a98`, tag `v3.4.0`, tree==beta. Engine picker (LTX-2.3 | Hailuo H3) in the Video tab; H3 in Text+Image (Image = first-frame conditioning); 4 measured tiers (Draft 3s ~3min / HQ 3s ~5min / HQ 5s ~8min / Long 10s ~36min); optional install_h3.js (~75GB, 64GB-RAM gated, clones mrbizarro/minimax-h3-mlx@codex/h3-engine); warm-helper teardown before H3 jobs; subprocess isolation — LTX untouched. Validated end-to-end by an independent agent (real t2v + first-frame i2v renders with audio, stop test, LTX regression, 5/5 SHIP). Release post LIVE: <https://beta.pinokio.co/posts/01kz48kve8g17ymk8dtms5j2g7> (robot-chef validation clip as hero). Residual known risk (accepted, owner-driven release): install_h3.js has never run on a truly clean machine — weights sources verified fetchable+ungated and the branch verified byte-identical to the validated runner; if a fresh-install report comes in, treat as hotfix priority. H3 experiment records: ~/AI/projects/hailuo-mlx/.

> **🧪 2026-08-05 — v3.4.1: chained H3 tiers, the H3 export pass, crash-reap, and BT.709 on every delivered clip.** (**SHIPPED PUBLIC same day** — `f66e59c`, tag `v3.4.1`; release post <https://beta.pinokio.co/posts/01kz986wqj2nemwm7f3vr1bm8c>.) Four changes, all validated on the dev box before the push.
>
> - **Chained 10 s / 15 s H3 tiers** (`e831fb3`). Anything past 5 s now renders as N chained 5 s windows — window N's last decoded frame re-enters as window N+1's first-frame keyframe. 10 s went 36:12 (one dense 243f pass) → **17:05**; 15 s is newly reachable at **26:34**; memory is FLAT (13,580 rows/window, 40.2 GiB peak) so duration is linear in wall clock. The dense 10 s path stays behind `LTX_H3_DENSE_10S=1` for A/B. Honest artefact reported as data: one prompt is asked of every window, so a scripted line lands once per window — the tier carries a `note` the panel renders under the strip. Chaining needs `--chain-windows` on the INSTALLED runner, so it is probed (`/status.h3.chain`), the chained tiers are withheld when absent, and `make_job` falls back to the longest single-pass tier instead of failing a queued job.
> - **H3 gets the export pass LTX already had** (`e831fb3`). H3's native 768×448 (12:7) no longer lands in a gallery of 1280×720 clips: same ffmpeg recipe (lanczos fit inside the canvas, pad the remainder, user's codec settings, `+faststart`, audio copied through). Default `fit_720p`, `fit_1080p` option, `off` for native. `h3_upscale` is in the make_job allowlist; native stays on disk but hidden, exactly like the LTX path; `UPSCALE_TAGS` is module-level so `/output/delete` and Load Params can't drift from `compute_upscale_plan`.
> - **A SIGKILLed panel's children get reaped at the next boot** (`1dde141`). Both heavy children spawn with `start_new_session=True` (so `/stop` can killpg the tree), which means `kill -9 <panel>` leaves them running — and the restarted panel re-queues the in-flight job and starts a SECOND 40 GiB render next to the orphan. Each spawn now drops `state/h3_running.json` / `state/helper_running.json`; `reap_orphan_subprocesses()` runs before `worker_loop`.
> - **BT.709 on every delivered clip** (`6b8f34f`). Every mp4 the panel handed the user read back `color_primaries/transfer/space = unknown`, so players guessed and the reported symptom was washed-out colour. **On ffmpeg 8 the output flags alone are not enough** — the filtergraph's frame properties override `-color_primaries`/`-color_trc`/`-colorspace`, and every export pass has a `-vf`. Measured on both builds a user can hit (8.0.1 bundled, 8.1.2 Homebrew): flags alone → primaries only; flags + a trailing `setparams` → all three. Applied to the H3 export, the LTX fit_720p/1080p upscale, the i2v audio mux, Long Clip Boost, and both PiperSR encodes. `setparams` is probed once and falls back rather than losing a render on an ancient build.
>
> **Corner tests run this session (dev box, documented H3 env):**
> - **Scheduling — PASS.** Two batches. Batch 1 (5 jobs queued in the same second: LTX → H3 → H3 → H3 → LTX) executed strictly serially, each job starting the second the previous finished, 5/5 done. Batch 2 (H3 → LTX → H3) confirmed helper teardown before each H3 job and lazy respawn for the LTX job in between. Exports verified: H3 Draft → `1280×720`, 72f in / 72f out, AAC copied through, sidecar carries the full `upscale` plan; LTX Quick + `fit_720p` → `1280×720`, 48f in / 48f out. Gallery lists only the `_720p` files; natives hidden.
> - **Crash-resume — PASS on the reap, PARTIAL on the live-orphan leg.** An *unplanned* crash mid-batch (see the collision note below) gave the real thing: panel gone, job persisted as `running`. Next boot cleared the stale guard without signalling anything and re-queued the job as `queued` — resumed, not zombie, no second render. The kill path itself was then exercised deliberately with decoy children on all three branches: **orphan with a dead owning panel → reaped** (`[boot] reaping orphaned h3 subprocess pid=… / h3 orphan gone`); **child whose owning panel is still alive → untouched, guard preserved**; **recycled pid whose cmdline no longer matches the needle → untouched, guard dropped**. What was NOT re-run end-to-end: `kill -9` against a live 40 GiB H3 render — see below.
>
> **⚠️ Machine collision, not a code fault.** Mid-test another agent started a separate ~40 GiB `generate_staged.py` run (the `hdloop` HD-loop campaign under `~/AI/projects/hailuo-mlx/codex/opt_out/`). Two H3 renders do not fit in 64 GB: the OS killed this panel *and* its H3 child, which is why the crash above was real rather than staged. Free memory stayed at ~11% afterwards, so no competing render was started to finish the live-orphan leg — re-run `kill -9` mid-H3-render when the box is idle. **Check `pgrep -f generate_staged` before any H3 work on this machine, and re-check it mid-run.**
>
> **Deferred (not touched here, deliberately):** the *base* render — LTX with export `off` — is still untagged, because that mp4 is written by the vendored `ltx_core_mlx` encode that `patch_ltx_codec.py` patches at install time. That call has no `-vf`, so flags alone would be enough there; it was left alone because widening the CODEC-ONLY patch on a release commit risks the installer. Same file, same visit: that encode passes `-shortest` alongside `-c:a aac`, which is why a 49-frame LTX render delivers 48 frames (the panel's duration model already assumes this). Both belong in one careful pass over `patch_ltx_codec.py`.

> **🚨 2026-08-05 — v3.4.0 FIELD REGRESSION FIXED on `dev`/`beta`: "I installed other packs and Hailuo H3 vanished."** A public v3.4.0 user installed H3, rendered with it happily (10 s clips on an M5 Pro), then installed other optional packs — and H3 disappeared from the panel. He ran **Reset**; H3 was still gone.
>
> **Root cause — `mlx_ltx_panel.py:_h3_python()` + `pinokio.js` h3_ready, both trusting `.venv/bin/python3.11`.** Nothing is ever deleted. `install_h3.js` builds the H3 venv with `uv venv`, and uv creates the interpreter as a symlink chain into Pinokio's **shared, app-external** managed Python:
>
> ```
> minimax-h3-mlx/.venv/bin/python3.11 -> python
> minimax-h3-mlx/.venv/bin/python     -> <pinokio>/cache/XDG_DATA_HOME/uv/python/
>                                        cpython-3.11-macos-aarch64-none/bin/python3.11
> ```
>
> That target belongs to Pinokio, not to Phosphene (verified on this machine against the LTX venv, which has the identical chain). Any other pack install — or any other Pinokio app — that makes uv re-resolve, bump or prune the managed interpreter leaves the chain **dangling**. `Path.exists()` follows symlinks, so `_h3_python()` returns `None`, `h3_available()` goes false, and the panel tells a user with **~75 GB of H3 weights still on disk** that H3 is "not installed". The engine pill demotes, the tier strip disappears — "H3 vanished".
>
> **Why Reset made it worse, not better.** `reset.js` removes only `ltx-2-mlx/`, so it never touches `minimax-h3-mlx/` — it *cannot* repair this. And because Reset makes `env_ready` false, `pinokio.js` early-returns at the `!env_ready` branch, which never reaches the H3 row — so after Reset there was **no H3 affordance anywhere in the product**. That is the whole reported experience, exactly.
>
> **This bug class was already documented in this repo and we repeated it.** `required_files.json → env._comment` records the same trap for the LTX venv, cocktailpeanut-confirmed: *"we used to point at `bin/python3.11` but uv creates that as a symlink chain … Pinokio's `info.exists()` check returned false on that chain … use `pyvenv.cfg` instead."* H3 shipped pointing at `bin/python3.11` anyway. **Any new venv-backed pack must not probe `bin/python*`.**
>
> **Fixed (`3c9bdf6`):**
> - `h3_paths()` / `h3_status()` now report **why**, not just that it's unavailable: `reason` (`ok` | `not_installed` | `missing_weights` | `missing_venv` | `missing_runner`) plus `repairable`, `venv_broken`, `weights_ok`, all on `/status.h3` and the page bootstrap. `repairable` = weights on disk, only code/venv broken.
> - The panel offers **repair, not reinstall**: engine pill (`needs repair · weights kept`), the H3 card (rewritten for the repair case), the job-time error, and a new branch in `updateModelsCard()` — gated on `repairable`, so a user who never installed H3 is never nagged.
> - `pinokio.js` probes the interpreter with **`fs.existsSync`** (follows symlinks → a dangling chain is definitively false) instead of `info.exists`, so menu and panel agree by construction. Adds **"Repair Hailuo H3 (weights kept — no re-download)"** and keeps it reachable from **every** menu state, including post-Reset.
> - `install_h3.js`'s venv step no longer trusts a path guard — a dangling chain can read as *present* depending on whether the check stats the link or the target, which would silently skip the repair. It now asks whether the interpreter **runs** and rebuilds only when it doesn't (~50 ms on a healthy install, ~2 min self-heal when broken, zero bytes re-downloaded).
> - `reset.js` documents that `minimax-h3-mlx/` and `mlx_models/hailuo-h3/` are user content and must never be added to it.
> - **Follow-up, same session (`87b28f4`): the LTX venv carried the IDENTICAL guard.** `install.js` gated its venv rebuild on `when: !exists('ltx-2-mlx/env/bin/python3.11')`, and the LTX chain into Pinokio's shared uv Python is the same one (verified above — it's what the H3 comparison was made against). Replaced with the install_h3.js run-probe shape: `env/bin/python3.11 -c 'import sys'` in-shell → reuse (~50 ms) or rebuild (~5 min, zero re-downloads); the macOS-14 preflight + v2.0.3 diagnostics kept inside the rebuild branch; python3.11 (not python) so a legacy conda-3.10 venv still rebuilds; the if/else ships as one joined multi-line message (the mflux-step idiom) so it runs as a single compound. Lower urgency than H3 was — `reset.js` DOES delete `ltx-2-mlx/`, so Reset+Install already recovered a broken LTX venv — but Reset is no longer the only cure. Gates run: `node --check` + the exact message string extracted from the module and executed in bash+zsh across healthy / dangling-terminal-target / absent / fake-macOS-13 sims (healthy reused untouched; dangling healed via real `uv venv` back to the exact production chain and converged to "healthy" on re-run; Ventura sim exited 1 before uv). The standing rule holds product-wide now: **no venv-backed pack probes `bin/python*` by path anywhere** — remaining `exists()` uses are `.git` clone guards and the pyvenv.cfg/`fs.existsSync` probes. Remaining ideal gate before any public promote: one real fresh Pinokio install.
>
> **Validation.** Sandbox at the DEFAULT paths (no env overrides) driving the **real** `h3_paths()` (spliced from the panel) and the **real** `pinokio.js` menu block, across healthy → stomp → Reset → repair. Before: panel "not installed", menu after Reset `[Install, Models]`. After: panel `reason=missing_venv repairable=true venv_broken=true weights_ok=true`, menu `Repair Hailuo H3` present in the healthy *and* post-Reset states, back to clean once the venv is rebuilt — **5/5 weight components untouched throughout**. `py_compile` + `node --check` on all nine root scripts. The live panel was **not** restarted (a Codex GPU experiment held the box), so this is code-level + simulation validation; a real Pinokio click-through of Repair is the remaining gate.
>
> **User recovery on v3.4.0 today, without the fix:** sidebar → **Install Hailuo H3** again. It is idempotent, the weights are still on disk, so it rebuilds the venv and skips every intact file — minutes, not 75 GB. If they already hit Reset, they must click **Install** (base) first, because the H3 entry only renders once the LTX env is back.
>
> **Public hotfix?** Judgement: **ride v3.4.1**, don't emergency-patch v3.4.0. It is a discoverability/labelling failure, not data loss (nothing is deleted, no render is corrupted), the workaround is one sidebar click, and H3 is a 64 GB-only opt-in pack so the blast radius is small. But v3.4.1 should ship **soon** rather than waiting to accumulate — every affected user currently reads this as "I lost my 75 GB download".

> **📈 2026-08-05 — anonymous usage analytics on `dev`/`beta` (NOT public, and INERT until a key is pasted).** ⚠️ **THIS WHOLE ENTRY IS SUPERSEDED** — the "inert / no key / a fresh clone sends nothing" posture below was true when written and was deliberately reversed by `acfbdc7` on 2026-08-09. Read the 2026-08-12 truth-pass entry at the top of this file for what is true now, including the 35/35 validation line further down (the suite is 38/38 and its no-key tests were rewritten). The counterpart to the H3-vanish fix above: that bug existed in the field for days and we only learned about it because one user wrote in. `pack_state_change` is the detector for exactly that class — a pack going `true → false` between boots is a broken install, and the Usage section of `/stats` now surfaces the count in red.
>
> **Read `docs/ANALYTICS.md`** — it is the complete schema (every event, every field) and the owner-activation instructions.
>
> - **Five events, counts only.** `app_boot` (version, os_version major.minor, chip_family parsed to `M4 Max`, ram_gb, cap_tier, `packs{h3,sharp,q8,qwen}`, h3_chain_supported) · `render_completed` / `render_failed` (engine, mode, tier, bucketed duration, resolution, frames, + a scrubbed `error_signature` on failure) · `pack_state_change` (pack, from, to). No prompts, filenames, paths, media, seeds, LoRA or character names. `engine_selected` was deliberately **dropped** — it would add a request path to a click that currently has none, and `engine` on every render event answers the same question.
> - **Inert in the public tree.** `ANALYTICS_KEY_DEFAULT = ""`, and an empty key means **no socket is ever opened** — not "a request that fails". A fresh clone sends nothing, forever, until the owner pastes a PostHog project key. This is the property that makes shipping it defensible after the 2026-05 opt-in module was reverted (`da1d6f5`, "not going to be well accepted in the open source world"). — **⚠️ SUPERSEDED 2026-08-09 by `acfbdc7`**, which committed the live `phc_` key on purpose: a stock install now reports, and the off switch is the Settings toggle (clearing the key field falls back to the shipped default). True as written on 2026-08-05; left standing as the record, corrected in the 2026-08-12 truth-pass entry at the top of this file.
> - **Cannot touch a render.** Capture builds the payload, starts a daemon thread, returns. 2 s timeout, bare `except: pass` around the whole delivery path, no retries, no queue, no heartbeats, no background timers. Wired at exactly two call sites: `_analytics_boot()` in `__main__` (never at import time) and `_analytics_render_event()` in `worker_loop`'s `finally` — the one point every job from every engine passes through.
> - **Visible + reversible.** One-line ASCII disclosure in the boot log on first run; Settings → *Anonymous usage analytics*, default **ON**, one click off (no confirm dance — turning it off must be easier than leaving it on). `PHOSPHENE_ANALYTICS_DISABLED=1` beats the setting. A legacy `analytics_enabled: false` from the reverted May module is preserved by `setdefault`, so that user's opt-out is honoured rather than silently flipped back on.
> - **Local mirror + the dashboard.** Every captured event also appends to gitignored `state/usage-log.jsonl` (5 MB cap, oldest half dropped) — the always-works "this machine" data source *and* a plain-text record of exactly what the panel would transmit. New `GET /stats/usage` (127.0.0.1-only like the rest) serves local aggregates, and upgrades to fleet-wide numbers when a PostHog **personal** API key is configured (ten read-only HogQL SELECTs, cached to `state/usage-fleet.json` for 6 h, per-query failure isolated). New **Usage** section in `panel_assets/stats.html` in the same card/KPI language as the GitHub sections: 4 headline tiles, top-5 error signatures, version/chip/memory distributions, and the pack-regression alert.
> - **Two keys, two jobs.** Capture = PostHog **Project** key (`phc_…`, write-only, turns the pings on). Fleet = PostHog **Personal** API key (`phx_…`, read-only, turns the dashboard numbers on). Both in Settings → *Maintainer / self-hosting keys*; env overrides `PHOSPHENE_ANALYTICS_KEY` / `_QUERY_KEY`; self-host via `PHOSPHENE_ANALYTICS_HOST` / `_API_HOST`.
>
> **Validation (code-level — a Codex GPU experiment held the box, so the live panel was NOT restarted).** `python3 scripts/test_analytics_dryrun.py` — **35/35 pass**, no network, isolated temp state dir: zero sockets with no key; nothing written at all when disabled; exact field set asserted per event against the doc; non-leakage proven including a prompt quoted inside an exception message (exact-secret redaction runs *before* path stripping) and paths the job never mentioned (HF cache, venv); forbidden-key dropping; bucketing; log rotation; local aggregates. The **real** `Handler` was then served on a spare port (8478, isolated state dir, no worker/GPU/helper): `GET /stats/usage` → 200 with correct local aggregates and zero leakage in the response body; `GET /settings` exposes `analytics_enabled` + `has_analytics_*` booleans + the install id, and never the raw keys. The dashboard section was driven through fleet / local-fallback / empty / endpoint-404 payloads in a browser against a throwaway static server — all four render correctly, and a 404 (older panel) leaves the section hidden. **Remaining gate: a live-panel pass — boot disclosure line, the Settings toggle round-trip, and a real render writing a `render_completed` row.**

> **🎛 2026-08-05 — H3 selectable sampler depth on `dev`/`beta` (`ae298da`): Steps pills Auto / 12 / 16 / 20.** Asked for by @ivanfioravanti ("Have you updated UI too (selectable number of steps?)"), who runs the official 20-step recipe. Tiers keep baking the validated 9-step point as **Auto**; a pinned depth rides the SAME params path the tier default does — `make_job` parses `h3_steps` (auto/garbage → 0-sentinel, ints clamped 4–30), stamps the resolved count into `steps`, and keeps `h3_steps` alongside so the ⓘ modal can print "tier default overridden". `run_h3_job_inner` needed **zero** changes (it already read `p["steps"] or tier["steps"]`). UI: a Steps row under Export in the H3 strip (same pill language), time-cost multipliers in the tooltips (~×1.4/×1.9/×2.4 = (N−1)/8 forwards), survives tier switches + engine swaps + reloads (`phos_h3_steps` localStorage, restored in `_restoreH3TierEarly`), mirrored into the shared hidden `steps` so the queue card/estimate stay honest, shown in the Customize summary only when pinned. Validated: `py_compile`, `node --check` on the extracted inline script, and 8/8 `make_job` unit cases (override / auto / absent / clamp-high / clamp-low / garbage / chained-tier override / LTX isolation). Live click-through deferred with the other panel gates (Codex GPU campaign holds the box). This is the first slice of the promised Ivan "advanced mode" — larger canvases stay with the quality campaign.

> **📦 2026-08-05 — v3.4.2 SHIPPED PUBLIC** (same-day follow-up to v3.4.1): H3 **Steps pills** (Auto/12/16/20 — asked by @ivanfioravanti, who runs the official 20-step recipe), the whole triage sweep below (#45 #44 #48 #36 #35 fixes), the LTX/H3 venv **self-heal** hardening from the field-regression class, and the analytics groundwork — **fully inert in the public build**: `ANALYTICS_KEY_DEFAULT=""` means no socket is ever opened; nothing is sent by any public install unless a maintainer key is deliberately baked in. (**⚠️ SUPERSEDED** — true of v3.4.2 as shipped; `acfbdc7` deliberately baked the key in on 2026-08-09. See the 2026-08-12 truth-pass entry at the top.) Deferred (GPU still owned by the quality campaign): live click-throughs of the steps pills, #44 fallback on a real M1, #48 end-to-end extend chain, and the fresh-Pinokio-install pass — same owner-accepted residual-risk posture as v3.4.1; treat any fresh-install report as hotfix priority.

> **🧹 2026-08-05 — FULL GitHub triage sweep (5 parallel agents, 6 commits on `dev`/`beta`, all 12 open issues handled).** Every open issue answered as the maintainer; 4 closed (#43 #34 stale-blank, #45 fixed, #47 answered, #36 implemented), 8 kept open with substantive replies. Commits, in order:
> - `2f5ab92` **#45 LoRA strength bar** — `#genForm` ID-specificity beat the compact row styles (readout became an unreadable sliver behind the spin button) + the range never got the chrome reset. CLOSED.
> - `a4e033a`+`b3f46a5` **#44 M1 Metal watchdog kill in Gemma prompt-encode** — NOT the Enhance feature (mandatory text encoder; every LTX render). Panel now sniffs the watchdog signature and retries the job ONCE with `LTX2_GEMMA_MAX_LENGTH=256`, sticky per boot, never pre-emptive — healthy Macs byte-identical. Also documented why user env `export`s never reach the panel (Pinokio `shell.run` replaces env wholesale; only `start.js` `env` entries arrive). OPEN pending reporter's 256-token datapoint (if THAT still times out → driver issue, upstream).
> - `a420eec` **#48 Extend progressive distortion** — Extend consumed the visible `_720p` EXPORT (2 resamples + 2 H.264 generations before the VAE) → now resolves the hidden native via sidecar (`_native_render_for()`); `_ensure_downscaled` squashed aspect 3.85% (independent /32 floors) → aspect-preserving scale+crop, dims unchanged. **NOT fixed (upstream, vendored):** per-round re-encode of the accumulated clip + audio resampled to 16 kHz per round before the audio VAE (the audible bulk; >8 kHz destroyed per extend) — candidate for a dgrauet upstream PR, same lane as AdaLN. Stated plainly on the issue. OPEN.
> - `a21e188` **#36 placement-aware integrity** — deep-verify + `/status` now catch *shadow* (undeclared same-name file differing from the declared copy) and *displaced* (declared file missing but same-name candidate elsewhere; size-matched candidates hashed only then). 9/9 unit cases incl. false-positive traps; 0 findings on this machine's real install. CLOSED.
> - `59ed751` **#35 train preflight poisoned its own well** — the one-click download pulled the Q4 repo's ~11 GB QUANTIZED transformer-dev (trainer refuses it via `FULL_DEV_MIN_BYTES` → dead LoRA) and showed green at a 1 KB presence gate. Now downloads `dgrauet/ltx-2.3-mlx-q8` (~21 GB full-precision, matching the trainer's q8-sibling resolution) and `ready` requires ≥15 GB. 6/6 sparse-file unit cases. #35 OPEN pending reporter (voice-LoRA regression is separate).
> - Response-only: **#30** (48 GB truth: cap_tier hides nothing at 48 GB — the separate `<64 GB` boundary shapes jobs; real numbers given, no promises), **#24** (enhancer is positive-only; a ~90-term default negative already rides CFG paths — auto-negative isn't a uniform win), **#21** (alt-base training blocked by inference-path match, honest requirements listed), **#46** (audio_start_time supported by pipelines+helper but never wired into the panel — plumbing task filed; A2V collapses past ~257 frames ≈ 10.7 s @24fps, workaround ≤10 s given), **#47** (prompting guide answer, CLOSED; publicly owned the Avoid-field gap: silently ignored in Control/Colorize/Ingredients and under H3).
> - Also from the sweep: JS training-preset fallback drifted (3000/5000/7000 vs real 8000/12000/20000); `docs/API.md` documents an `enhance` param with no consumers; `_train_install_dev_transformer` was item 3 of #36 and is now `59ed751`. Deferred live gates (panel restart still blocked by the GPU campaign): #44 fallback on a real M1, #48 end-to-end extend chain, #45 visual once the panel restarts.

> **✨ 2026-08-06 — "Draft → Finish" for H3 on `dev`/`beta`.** Mr Bizarro, on seeing H3's 3-minute drafts: *"Draft mode! We have draft mode in LTX! We can ship this!"* The Draft tier already existed; what didn't was the workflow that makes a cheap tier worth having — **one click that re-queues the identical job at a delivery tier**.
>
> - **Where.** A `Finish at HQ 5s · ~8 min` split button leading the player's action cluster (left of Params/Extend), visible **only** when the selected clip is an H3 render whose tier carries `draft: true`. The caret beside it is a native `<select>` listing every non-draft tier with its own eta; the choice persists (`phos_h3_finish_tier`). A clip already at a delivery tier gets nothing — there is nothing to offer.
> - **Seed carry-over is the feature.** The finish job takes the draft's sidecar `seed_used` — the integer the H3 path actually rolled — never the `-1` the user submitted. Get that wrong and the button returns a *different clip* and the whole affordance is a lie. Same contract the Manual `loadParams` fix (`b024bb5`) established. Also carried: prompt, first-frame image (i2v), `engine=h3`, `h3_upscale`, and a pinned `h3_steps`. Every one of those keys was **verified present in the `make_job` params allowlist** — the standing trap in this file is that an unlisted field silently no-ops on `/queue/add`.
> - **Honest labelling, one source of truth.** The eta on the button and in the picker is the tier's own `eta` string from `H3_TIERS`; nothing computes a time in JS. The tier's full label is used (`HQ 5s`, not `HQ`) because "HQ" alone reads identically for the 3 s and 5 s tiers. The label survives the ≤1100 px compact breakpoint that drops every other action to icon-only — the cost *is* the decision.
> - **Draft honesty note.** `draft_3s` now carries a `note` (`H3_TIER_DRAFT_NOTE`) rendered under the tier strip in the existing `engine-hint` style, exactly like the chained tiers' artefact warning: *"Draft is for composition, motion and dialogue timing — faces and fine detail only resolve at the higher tiers."* The 0.25 MP pass is below the community's ~0.5 MP practical floor and video structure resolves in the final denoise step, so this is said out loud rather than letting a user conclude H3 is bad.
> - **How it's wired.** Deliberately the Load Params pattern, not a new endpoint: read `/sidecar`, restore into `#genForm` (mode → engine → prompt → first frame → export → steps → **tier last**, since `setH3Tier` stamps width/height/frames → seed), then `requestSubmit()` so the one submit path keeps owning the double-click guard and the prompt modifiers. The form is left showing exactly what was queued. Two guards worth knowing: `setEngine('h3')` is called **after** `setMode` (which re-applies the *persisted* engine) and its return value is checked — an H3-unavailable box gets a toast instead of a silent LTX render at H3 geometry; and on i2v the hidden `#mode` is pinned to `i2v` because `setMode` otherwise copies `#i2vMode`, which may still be sitting on `i2v_clean_audio` (a mode H3 does not serve).
> - **Free metadata.** `list_outputs()` now returns `engine` + `h3_tier` per entry, lifted out of the sidecar read it *already* performs for `elapsed_sec` — so the affordance decides synchronously and no `/sidecar` fetch rides the gallery-click path.
>
> **Validation (static — the GPU is held by a showcase render batch, panel NOT restarted, nothing rendered).** `py_compile`; `node --check` on the extracted inline `<script>`; **74 assertions, 0 failures** across three suites: 29 on the pure `h3FinishFieldsFromSidecar()` extracted verbatim from the shipped script (seed from `seed_used` not `-1`; `seed_used: 0` honoured as a real seed; `seed_used: -1` distrusted; i2v first frame carried; t2v first frame deliberately empty; non-H3 / missing-image sidecars refused; pre-v3.4.1 and pre-v3.4.2 sidecars degrade to defaults; exact emitted key set), 17 on the affordance against a DOM stub + the **real** tier table dumped from `h3_status()` (shown only for drafts; hidden for delivery tiers, LTX clips, and tiers this install no longer offers — no first-tier fallback; label equals the table's eta; picker never lists the draft; stale localStorage tier recovers to the server default), and 28 in Python via `importlib` driving the **real** `make_job()` (every carried field reaches `job["params"]`, seed verbatim, finish-tier geometry replaces the draft's 640×384, pinned steps resolve, plus a regression guard that a plain draft submission is unchanged).
>
> **Remaining gate — one live click-through when the GPU frees up:** render a Draft, confirm the button appears with the right eta, click it, and verify the queued job's sidecar reports the **same `seed_used`** as the draft at the finish tier's geometry. Also worth an eyeball: the split button in the vertical-media action-cluster variant, and the draft note under the tier strip. Note this box currently reports `h3_available=false` (the known dangling-uv-venv case — run **Repair Hailuo H3** from the Pinokio sidebar first).

> **⚡ 2026-08-14 — H3 Turbo default moves to LightX2V v1.0 768p.** The resolver now prefers `lightx2v_v1.0_768p_ourlayout.safetensors` and falls back only to the alpha-folded `lightx2v_v0.1_ourlayout_alpha8.safetensors`. It never accepts raw v0.1, whose external alpha/rank factor makes scale 1.0 render coloured noise. The runner invocation is the native `--lora PATH:1.0`; the old Larry-specific AdaLN companion path is gone. The default was chosen after **core_tan's public LoRA testing plus the owner's visual review**. LightX2V's source repo is Apache-2.0; the exact v1.0 source digest and the runner-layout release-asset TODO live beside the fetch lane in `install_h3.js`.
>
> **⚡ 2026-08-06 — H3 "Turbo" on `dev`/`beta`: a 4-step distillation LoRA, shipped as a speed mode.** Mr Bizarro, on the graded clips: *"ok lets ship it with that. looks pretty decent."* [`larryvrh/MiniMax-H3-Turbo-Lora`](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora) (**Apache-2.0**) distils H3's sampler from 9 sigma points / 8 forwards down to **4 points / 3 forwards**. Measured at 1152×640 · 124f on this M4 Max: **19:26 → 11:29**, per-forward cost unchanged, peak +0.7 GiB. Graded on the standing mouth-first doctrine it came out **more resolved than the owner's own 19:26 PASS** on mouth, beard and eyes — what remains is a slightly harder light. Full evidence: `~/AI/projects/hailuo-mlx/notes/TURBO_LORA_PROBE.md` (§10a–10g).
>
> - **Runner first — the public fork carries LoRA support now** (`mrbizarro/minimax-h3-mlx@codex/h3-engine` `b617bc4..b00cc14`, the branch `install_h3.js` clones). New `minimax_h3_mlx/lora.py` + `--lora PATH[:SCALE]` / `--lora-mode` / `--lora-adaln` / `--lora-audit` on `scripts/generate_staged.py`, plus the probes that justify the design. **Addition-only: 1028 lines added, 0 removed**, and every runtime block is gated behind an `if <lora flag>`, so a job without `--lora` executes the byte-identical code path it did before. The probe had already proven the stronger property — `--lora …:0` renders **sha256-identical** to no `--lora` at all, in both modes.
> - **Two traps this lineage sets, both measured, both in the commit message so the next person doesn't re-find them.** (1) The fused `qkv_proj` rows are **per-head** `(heads, 3, head_dim)` here and **contiguous** `(3, heads, head_dim)` in the ComfyUI definition the LoRA was trained through — an ANOVA on row norms separates them at F=1362 vs 83, and applying `lora_B` unpermuted scrambles the update across heads and presents as *"the LoRA does nothing"* rather than as an error. (2) **Fusing into the bf16 base destroys ~87 % of the update** (delta is 3.4e-4 of |W|; bf16's relative ULP is 3.9e-3), so the shipping path is a run-time low-rank matmul — +2.3 % per forward on a draft canvas, below run-to-run noise on a hero one. `--lora-mode fuse` survives only as the control arm.
> - **The second file, and why it is not a 66 GB download.** The adapter's 51 adaLN modules expect the 2688-d timestep embedding that DeepBeepMeep's pruned export replaced with a 64-d curve, so they cannot be wrapped — they are instead folded **exactly** into the precomputed modulation cache (free per forward), given the upstream `time_embedder`. `scripts/fetch_time_embedder.py` recovers those four tensors by **HTTP range request** — 63 MB read out of a 66 GB release, because safetensors puts every tensor's byte offsets in a header at the front of the file — and `verify_time_embedder.py` proves they are the right ones against DeepBeepMeep's own published curve to a residual of **4e-12** (against 1.8e-4 for the bare sinusoid). Hardened for other people's machines on the way in: it no longer invents an `HF_HOME`, and a bf16 re-upload is widened exactly instead of being written as raw `uint16` — which would have saved cleanly and then multiplied as integers.
> - **Panel: a Speed row under Steps.** `Standard | Turbo · ~5 min`, in the existing pill language, `data-h3-only`. The estimate is **per tier** (`turbo_eta`, computed in `_build_h3_tiers()` from `H3_TURBO_SPEEDUP = 0.6` applied to each tier's own eta → Draft ~2 min, HQ 3s ~3, HQ 5s ~5, Long 10s ~10, Long 15s ~16) and the tooltip says out loud that it is derived from this tier's own time, **not measured at this canvas** — the 11:29 was rendered at 1152×640, which is not a tier on this table. Turbo and Steps are the same axis, so Turbo wins: the Steps pills go visibly dead, and `make_job` drops the override with a log line rather than rendering an unvalidated combination. One line of honest copy under the row: *"Turbo is a 4-step distillation LoRA — fewer denoise passes, same model. Slightly harder contrast than the 9-step default."*
> - **`h3_turbo` is in the `make_job` params allowlist.** The standing trap in this file, and it would have bitten exactly here: the control would look wired, pass every validation, and silently render at 9 steps.
> - **Availability gating, three separable states.** `h3_status()` gains a `turbo` block: `supported` (the **installed** runner has `--lora-adaln` — probed via the existing `_h3_runner_has_flag`, same as `--first-frame` and `--chain-windows`, so a pack cloned before today hides the row entirely instead of dying on an argparse error 30 s in), `downloaded` (both files really present), `available` (both). Following the H3-vanish lesson the file-probe is a **real** one: `is_file()` so a dangling symlink is definitively false, **plus a size floor** — an interrupted download otherwise leaves a short file that passes every existence check and fails deep inside the runner.
> - **Download flow: one click, 0.8 GB, nothing bundled.** `POST /h3/turbo/install` → `_h3_install_turbo()`, built on the same single download slot and log-streaming shape as `_train_install_dev_transformer`, but **two steps**: `hf download --include` for the 744 MB adapter, then the pack's own venv running `fetch_time_embedder.py` for the ~60 MB range-fetch. Preflighted before a single byte moves (H3 complete? runner new enough?), `HF_HOME` pinned inside the H3 models tree so the index JSON can't land in `~/.cache`, and the dashed pill flips live off `/status` the moment both files check out — no reload. **The weights are never committed anywhere:** the adapter is Apache-2.0 and the base model stays under the MiniMax Community License with its territory exclusions, so we ship code and download instructions only.
> - **Which file, and the two we deliberately did not ship.** `minimax_h3_turbo_4step_ema_ckpt500.safetensors` — larryvrh's own full-model file. The **non-EMA** ckpt500 is the sharpest of the family and is rejected anyway: it pushes the over-etched-highlight look the owner has already flagged and runs the audio to −0.3 dB with no headroom. The **third-party ComfyUI conversions** are bit-exact subsets that drop the 51 adaLN pairs this runner can apply — strictly less, for no upside.
>
> **Validation (static — a showcase render batch owns the GPU; the panel was NOT restarted and nothing was rendered).** `py_compile` on all 9 changed runner files and on `mlx_ltx_panel.py`; `node --check` on the extracted inline `<script>` (12,116 lines, clean); a runner argument-parse smoke test that slices the parser out of `main()` with `ast` so **mlx is never imported** (no-`--lora` defaults all falsy → every guard skipped, the shipping invocation, `:0` and colon-in-path specs, a rejected `--lora-mode`, and an assertion that all three new runtime blocks are flag-gated); and **49 assertions, 0 failures** in two Python suites driving the **real** panel via `importlib` — 24 on the helpers (per-tier etas; the probe rejecting nothing/truncated/dangling-symlink and accepting full-size; `runner_too_old` vs `not_downloaded` vs `ok`; the status block present even when H3 itself is missing; the installer refusing cleanly) and 25 on the **real `make_job()`** (`h3_turbo` reaching `job["params"]`, steps pinned to 4, geometry still the tier's, a Steps override dropped with a log line, both unavailable states demoting to Standard, no leakage onto the LTX lane, chained tiers taking Turbo, and a regression guard that a plain non-Turbo submission is unchanged).
>
> **Remaining gates — a live click-through when the GPU frees:** (1) click the dashed pill and let the real 0.8 GB download run, confirming both files land in `mlx_models/hailuo-h3/turbo-lora/` and the pill flips live without a reload; (2) render one clip with Turbo on and confirm the log's `LoRA … 208 applied` + `adaLN LoRA absorbed … 50 blocks + final layer` lines, the ⓘ modal's Turbo row, and the wall clock against the tier's `turbo_eta`; (3) the same tier with Turbo **off**, to see the estimate's error bar in practice; (4) Draft → Finish from a Turbo draft, confirming Turbo carries over. Note this box still reports `h3_available=false` (the known dangling-uv-venv case) — run **Repair Hailuo H3** from the Pinokio sidebar first, or the Speed row will correctly stay hidden.

> **📐 2026-08-06 — H3 gets a TRUE 16:9 tier, and every tier now says its aspect out loud.** Mr Bizarro: *"H3 users are getting bars on every clip for no reason."* He was right, and nothing in the UI admitted it. Every H3 tier rendered at an odd ratio and then got **pillarboxed** by the export pass — 768×448 is 12:7, 640×384 is 5:3, our best hero canvas 1152×640 is 9:5, and none of them is 16:9 (1.7778). Meanwhile LTX's Balanced preset renders 1024×576, which IS 16:9, so its 720p export is a pure 1.25× scale with no padding at all.
>
> - **The arithmetic, and why there is exactly one answer.** The runner errors unless both axes are multiples of 32 (`if args.height % 32 or args.width % 32: parser.error(...)` — verified in `generate_staged.py`, not taken on trust). Exact 16:9 on that grid means width `512k`, height `288k`, and only three of those exist near H3's envelope (`packing.py`: SHORT_EDGE 768, MAX_PIXELS 768·1344, CANVAS_MULTIPLE 32): **512×288** (0.15 MP), **1024×576** (0.59 MP), **1536×864** (1.33 MP — *over* the model's own MAX_PIXELS of 1.03 MP, and dearer than the native 1344×768 canvas that measured 44.8 min per 5 s window; not shipped, and now written down so nobody re-derives it).
> - **`wide_5s` — 1024×576 · 124f · 9 sigma points, `~17-19 min`.** Not extrapolated: the quality loop's R1 run measured **exactly this geometry at exactly these 8 forwards** on this M4 Max — 22,923 packed rows, 126.0 s/step, 90.5 s VAE decode, 10.71 GiB decode peak, **18.8 min wall**, denoise flat at 37.6 GiB (identical to 768×448 — the DiT weights dominate, there is no memory wall). The same probe put 768×448/124f at 9.1 min against this table's advertised ~8 min, which is where the 17-19 band comes from. **On Turbo the same canvas is MEASURED at 8.5 min** — see the cost-model correction below. Same seed, same still, same forwards, judged at 1:1 on a 1080p delivery, 1024×576 against 768×448: eyebrows become individual hairs, forehead gets pores instead of wax, eyelashes exist at all, fur reads as strands — **and** the pillarbox disappears while the 1080p enlargement drops 2.41× → 1.875×.
> - **A 16:9 draft (512×288, ~2 min) is wired but OFF, behind `LTX_H3_WIDE_DRAFT=1`** — the same pattern `LTX_H3_DENSE_10S` uses. Judgement call, stated plainly: it would be the cleanest draft→finish pair on the table (identical aspect, an exact 2× to Wide 5s, so the framing a draft shows is the framing that ships), but 0.15 MP is a third of the pixels the Draft note *already* hedges about, and **nothing in this campaign has ever rendered below 640×384** — the ladder only ever went up. Shipping an unmeasured bottom-of-ladder tier is how a user concludes "H3 is bad", which is the exact failure the draft note exists to prevent. One env var and one render decides it.
> - **Every tier now labels its own aspect,** derived from its width/height by `_h3_aspect()` and appended to the `spec` the chip prints — `1024×576 · 124f · 16:9` against `768×448 · 124f · 12:7`. Derived, never typed, so an advertised ratio cannot drift from the geometry that renders. The `wide` boolean rides along for anything that needs the machine-readable form.
> - **And a live sentence under the Export row** (`#h3ExportNote`, the existing `engine-hint` style): *"720p: pure 1.25× scale to 1280×720 — no bars, no padding."* vs *"720p: 12:7 fits to 1234×720 inside 1280×720 — 23 px bars left and right."* It is per **(tier × export target)**, because that is what the answer actually depends on, and the strings are generated server-side by `_h3_export_notes()` **from `compute_upscale_plan` itself** — the copy cannot disagree with the ffmpeg command because it is built from it. Hidden on Native. Incidentally this corrects a long-standing wording error in this file and the code: 12:7 is *taller* than 16:9, so it fits by height and the bars land at the **sides**. It was never letterboxing; it was pillarboxing, 23 px at 720p and 34 px at 1080p — the 34 matches the campaign's own measurement of "1851×1080 of content".
> - **`compute_upscale_plan` takes a pure-scale path on a matched aspect.** It used to emit `scale=…:force_original_aspect_ratio=decrease,pad=…` unconditionally — correct pixels for 1024×576, but with a pad filter padding a zero-width bar. Now `w·target_h == h·target_w` picks `scale=W:H:flags=lanczos` alone, and the plan carries **`pad`** plus **`fit_w`/`fit_h`** (the content size inside the canvas) so a sidecar reader can tell bars from picture without re-deriving it. Every non-matching source keeps the fit-and-pad filtergraph **byte-for-byte** — that is a pinned regression test, not a hope.
> - **Defaults deliberately NOT changed, and the code says why.** `H3_TIER_DEFAULT` stays `draft_3s`: Wide is the recommended *delivery* tier, which is a different job from what a first-time H3 user should be pointed at, and pre-selecting a ~19 min render is not a recommendation, it is spending someone's afternoon. `H3_TIER_FINISH_DEFAULT` stays `hq_5s` for the same reason — the Finish click is a cost commitment made from a 3-minute draft, and moving it from ~8 to ~19 min without the user choosing it is the exact surprise the per-tier etas exist to prevent. Wide sits one click away in the same picker with its own eta, and the choice persists. Both are one-line flips the day a live click-through backs the canvas.
> - **One bug fixed on the way past:** the pre-chaining-runner fallback picked "the last single-pass row in the table" by reversed scan. That meant `hq_5s` yesterday and would have silently started meaning the ~19 min `wide_5s` today. It now names `H3_TIER_FINISH_DEFAULT` explicitly — a fallback nobody asked for has to be the cheap one.
>
> **Validation (static — a 1024×576 Turbo validation render owns the GPU; the panel was NOT restarted and nothing was rendered here).** `py_compile`; `node --check` on the extracted inline `<script>` (12,137 lines, clean); **120 assertions, 0 failures** across two suites: 113 in Python via `importlib` driving the **real** module — the four asked-for `compute_upscale_plan` cases (1024×576→720p is `scale=1280:720:flags=lanczos`, `pad=False`, exactly 1.25× on both axes; 1024×576→1080p likewise at 1.875×; 768×448→720p and →1080p **string-identical** to the shipped fit+pad filtergraph, with `fit_w/fit_h` = 1234×720 and 1851×1080) plus portrait, `x2`, model-upscaled, already-at-target and unknown-mode paths, 640×384 / 1280×704 / 1152×640 regression guards, the whole tier table (32-grid, 17n+5 grid, derived aspect on every spec, exactly one 16:9 tier, both defaults unmoved), the export-note generator, `_h3_aspect` on eight canvases, the **real `make_job()`** (geometry stamped, LTX lane untouched, unknown tier still falls back), the chained-tier fallback under a monkeypatched pre-chaining runner (→ `hq_5s`, **not** `wide_5s`), `h3_visible_tiers()` attaching `export_note` without mutating the module table, and the env-gated draft appearing and disappearing with `LTX_H3_WIDE_DRAFT`; and 7 in node on `_h3SyncExportNote()` **extracted verbatim** from the shipped script, driven against a DOM stub + the real tier table (Wide/HQ × 720p/1080p/Native, a chained tier, an unknown tier key, and a pre-change server whose tiers carry no `export_note` — which degrades to silence rather than throwing).
>
> **Remaining gates — one live click-through when the GPU frees:** (1) render `Wide · 5s` and confirm the wall clock lands in the 17-19 band and the delivered `_720p.mp4` is 1280×720 with **no bars** and `ffprobe`-clean; (2) eyeball the tier strip at six chips — the specs are longer now (`768×448 · 362f · 3×5s · 12:7` is the widest) and may wrap to a second line inside their chips, which is survivable but worth seeing; (3) confirm the export sentence flips live when the Export pills are clicked and when the tier changes; (4) Draft → Finish into Wide 5s, checking the same `seed_used` reaches a 1024×576 render. Note this box still reports `h3_available=false` (the known dangling-uv-venv case) — run **Repair Hailuo H3** from the Pinokio sidebar first. **Open question for the owner, answered here as a recommendation:** `1024×576` should replace `1152×640` as the internal hero canvas for future experiments — it is exact 16:9, 20 % fewer pixels, 26 % cheaper per step, and exports pure-scale to both 720p and 1080p; keep `1344×768` for hero-grade stills-quality work when hours are acceptable. Chained 10 s / 15 s at 1024×576 (~37 / ~56 min) are the obvious next tiers and were deliberately not added here.

> **⏱ 2026-08-06 — the flat `H3_TURBO_SPEEDUP = 0.6` was WRONG, and it is gone.** Mr Bizarro rendered the tier that shipped an hour earlier, on Turbo, while the tier work was in flight — **1024×576 / 124f / `--steps 4` (3 forwards), ckpt500-EMA adapter → 8.5 min**, against a chip advertising ~11. Log `codex/opt_out/wide169/w169.log`, clip `w169_1024x576.mp4`.
>
> **Why a single multiplier could never work.** Turbo always runs **3 forwards**, whatever the tier bakes, and the fixed cost — staged loads, text encode, adaLN cache, video + audio VAE decode, mux — does not shrink at all. So the saving is a function of *how many forwards the tier would otherwise have run*, and 0.6 was only ever the answer for the canvas it was fitted on. Both arms, same box, same canvas, same seed:
>
> | 1024×576 · 124f | forwards | s/step | denoise | fixed | wall | vs full |
> |---|---:|---:|---:|---:|---:|---:|
> | full (`QUALITY_LOOP.md` R1) | 8 | 126.0 | 1008 s | ~131 s | **18.8 min** | — |
> | **Turbo** (`wide169/w169.log`) | **3** | 128.0 / 127.4 / 123.9 | 379.3 s | 130.7 s | **8.5 min** | **0.45×** |
>
> The fixed 130.7 s breaks down as dit_load 8.5 s + adaLN cache/noise 11.8 s + video VAE decode 88.4 s + audio VAE 0.9 s + encode/mux 5.2 s + text encode. Peak 42.71 GiB on the Turbo arm. The old 0.59 came from the 1152×640 hero (19:26 → 11:29) — which runs **six** forwards. Both numbers are right; the multiplier was the wrong shape.
>
> **What ships now.** `H3_TURBO_SPEEDUP` deleted, replaced by `H3_TURBO_FORWARDS = 3` and `H3_TIER_FIXED_MIN = 2.0` (per *window*, so chained tiers pay it per window — that is exactly what a chain does). Each tier's Turbo cost is derived from its **own geometry**: `per_forward = (eta − fixed) / (windows × (steps − 1))`, then `turbo = windows × 3 × per_forward + fixed`, clamped so Turbo can never read slower than the tier and never dips below its own load/decode floor. Re-prices the whole table:
>
> | tier | forwards | eta | turbo (was → now) |
> |---|---:|---|---|
> | Draft · 3s | 8 → 3 | ~3 min | ~2 → **~2 min** (0.79× — overhead-dominated, correctly barely moves) |
> | HQ · 3s | 8 → 3 | ~4-5 min | ~3 → **~3 min** |
> | HQ · 5s | 8 → 3 | ~8 min | ~5 → **~4 min** |
> | **Wide · 5s** | 8 → 3 | ~17-19 min | ~11 → **~8-9 min, MEASURED** |
> | Long · 10s | 16 → 6 | ~17 min | ~10 → **~9 min** |
> | Long · 15s | 24 → 9 | ~27 min | ~16 → **~14 min** |
>
> The derivation reproduces R1's 126.0 s/step from the 18.8 min figure alone and predicts 8.4 min for the Turbo arm against 8.5 measured — but `wide_5s` carries the **measurement**, not the derivation (`turbo_eta` stamped in the table with `turbo_measured: True`; the derivation never overwrites a measurement). The Turbo tooltip now distinguishes the two kinds of number instead of blurring them: *"Measured end to end at this exact canvas."* for Wide, and *"Turbo runs 3 forwards instead of 8, over the same fixed load/decode time. Not measured at this canvas."* for the rest — with the forward counts read from the tier, so a chained tier correctly says 9 instead of 24.
>
> **Validation.** `py_compile`; `node --check` (12,154 lines); **164 assertions, 0 failures** — 152 in Python (the whole table re-priced to the exact strings above, `H3_TURBO_SPEEDUP` asserted *absent*, measured-flag exactly on `wide_5s`, per-window forward counts, Turbo-never-slower and never-below-the-floor invariants on every tier, and the model re-deriving both measured arms) and 12 in node on `h3TurboPillLabel()` / `renderH3Turbo()` extracted verbatim (the measured tooltip, the derived tooltip, and a chained tier's forward count). **Lesson worth keeping: a speed feature priced by one ratio is a bug waiting for its second data point.**

> **🎚 2026-08-06 — the UX/UI pass: the engine moves to the top right, H3 gets LTX's information architecture, and there is now an ENGINE REGISTRY.** Mr Bizarro: *"lets do an ux ui pass. cos hailuo page is not as well done as ltx that need to be fixed. also hailuo and ltx modes should be selected on the upper right from respective logos. make phosphene look fully integrated again. soon flux video will release weights also so you can imagine where things are going."* Mock reviewed and approved first (*"Yeah, it looks good for me. I think it's good."*), then implemented to match.
>
> **1. `ENGINES` — one server-side table, and everything engine-shaped is rendered from it.** Two hardcoded chips were already one too many: the H3 pill, its dashed not-installed state, the mode gate, the `body[data-h3-engine]` CSS hook and the surface swap were **five** places that all had to agree, and a third engine meant touching all five. The table now drives: the header switcher (label, mark, accent, order, tooltip), the per-engine CSS emitted at `__ENGINE_RULES__` (`--eng-accent/-dim/-soft` *and* the `[data-<id>-only]` fold rules), the mode gate (`modes` + `excluded_modes`), the workflow it belongs to (`surfaces`), and which bootstrap key carries its capability probe (`probe: "h3"` → `BOOT.h3`, from the unchanged `h3_status()`). `make_job` validates `engine` against `ENGINE_IDS` instead of a hardcoded pair and calls the SAME `engine_serves_mode()` the switcher uses for the affordance. **Adding an engine is one entry plus one `<symbol>`.** Proven, not asserted: a real `flux` entry ships behind `LTX_ENGINE_PREVIEW=1` (the `LTX_H3_DENSE_10S` pattern) so the N-engine path is exercised end to end — with it on, the switcher renders three segments, `[data-flux-only]` starts working with no CSS edit, and `make_job` refuses `engine=flux` as `state: "announced"`. Off by default: the weights do not exist yet and the panel does not ship vapourware.
>
> **2. The switcher is in the header, top right, and it is a control — not another pill.** After the spacer, before `#tierPill`, with a hairline divider between them: **choice on the left of the hairline, state on the right.** 28 px segments in a 36 px track, one notch taller than the 34 px status chips, so it does not read as a badge. Under 1500 px the labels drop and the marks carry it (the header is `nowrap` + `overflow:hidden`, so something had to give before the pills got clipped). Every gate that existed is preserved and now generic: **not capable → not rendered at all** (a 32 GB Mac still never learns H3 exists, and with one engine left the switcher *and* its divider disappear); **capable but not installed → dashed segment + a `75 GB` badge in the engine's own colour, click opens the install card**; **installed but broken → the same shape with a `repair` badge and a card that never mentions a download** (the v3.4.0 report, verbatim); **installed but wrong mode → inert, with the reason in `#engineRowNote` where the surface changed rather than in the header where the click happened**. The old inline `#engineRow` is gone — there is exactly one picker.
>
> **3. The marks are ours.** Three original stroked monograms in the existing sprite sheet — an **L** with a play wedge, an **H** with three ascending level bars (the 3, and the sound), an **F** with a flow chevron. No third-party brand logo is scraped or bundled; this ships in an open-source repo. Their accents are the three stops of the **Phosphene wordmark gradient itself** — cyan `#5EEAFF` LTX, pink `#FF2E9F` H3, violet `#B14AFF` Flux — so the engines are coloured out of the product's own identity rather than a new palette. If one ever reads wrong in situ it is one field in the `ENGINES` row; nothing in the markup or the stylesheet knows a hex code.
>
> **4. H3 now has LTX's information architecture, because that was the actual complaint.** Before: the tier strip, then **four unrelated flat pill rows** (Export / Steps / Speed) with hand-rolled inline styles and three notes floating loose between them — *and* the Customize disclosure open below all of it still showing LTX's Width×height, LTX Export, LTX audio source and LTX long-clips, none of which an H3 render uses. After: the tier strip is the primary choice (H3's Quality), and **Speed → Steps → Export move into the SAME `<details id="customizeDetails">` LTX uses**, as `.cz-control` blocks with real `.cz-label`s, the same `.pill-group` grammar, and each note sitting under the control it belongs to. Speed comes first because Turbo overrules Steps. `updateCustomizeSummary()` already handled H3, so the summary line finally says something worth reading: `768×448 · 124f · 12:7 · Turbo · 4 steps · 720p export`. Every LTX-only control gained `data-ltx-only` and folds away. `.mini-fields` went from a fixed `repeat(3, 1fr)` to auto-flow columns so H3's surviving Seed field fills the row instead of sitting marooned in the first third. The six-tier strip's wrap is now deliberate rather than incidental — 3 columns past four, a deeper row gap than column gap, top-aligned chips with a floor height, and specs that break at their own ` · ` separators so the longest (`768×448 · 362f · 3×5s · 12:7`) wraps cleanly inside its chip.
>
> **5. "Fully integrated" is a scoped tint, not a repaint.** The active engine's accent applies to exactly two things: the primary strip's active chip and the Customize chips inside `#genForm`. The mode bar, workflow tabs, Image Studio, Train and every other surface keep the house blue. Switching engines should feel like changing lens.
>
> **Nothing was dropped.** `h3_tier`, `h3_steps`, `h3_turbo`, `h3_upscale`, the export note, the measured-vs-derived Turbo tooltip (`wide_5s` still the only measured one), the aspect in every spec, and Draft → Finish all keep their ids, their hidden inputs and their place in the `make_job` allowlist. `_engineRowVisible`, `_h3ServesMode`, `setEngine` and `#engineRowNote` keep their names so nothing that calls them broke; `body[data-h3-engine]` folded into `body[data-engine]` with the four JS readers rewritten.
>
> **Validation (static — the panel was NOT restarted; a review instance holds :8199 and a sibling agent was in the same repo).** `py_compile`; `node --check` on the extracted inline script; the real module driven via `importlib` for `ENGINE_IDS` / `engines_payload()` (JSON-serialisable, no callables) / `_engine_css()` / `engine_serves_mode()` across 7 modes × 2 engines; **`make_job` called for real** — every one of the five engine fields survives the allowlist (`engine`, `h3_tier`, `h3_upscale`, `h3_steps`, `h3_turbo`) with tier geometry stamped, plus 11 gate cases (h3+keyframe → ltx, h3+i2v_clean_audio → ltx, bogus id → ltx, announced → ltx, tier/upscale/steps coercion, Turbo-wins, Turbo cleared on the LTX lane). The switcher itself was driven in **node against a DOM stub with real fixtures from `h3_status()`**: all **nine** states render exactly as approved — LTX active, H3 active, needs-install (`75 GB`), needs-repair (`repair`), wrong mode (`inert` + `text · image`), character mode (excluded), under-64 GB (switcher *and* divider hidden), Images workflow (hidden, off-surface), and the three-engine row with Flux `soon` — and `engineSegClick` routes correctly in all four cases (install card / select / inert no-op / announced no-op). Rendered-page checks: no placeholder left unsubstituted, CSS braces balanced, **no new duplicate element ids** (the five in the page are all pre-existing at HEAD). Mock: `scratchpad/engine_ux_mock.html`.
>
> **Remaining gate — one live click-through when :8199 frees:** flip LTX ⇄ H3 in the header and confirm the surface swap has no flash and no half-lit strip; open Customize on H3 and confirm Speed/Steps/Export are the only controls in it; confirm the six-tier strip wraps 3+3 with the 15 s spec intact; confirm Turbo still disables the Steps pills and the export note still updates per tier × target; confirm the header does not clip at the owner's window width; and confirm Draft → Finish still queues (it forces the engine through `setEngine('h3')` and reads `#engineRowNote` for its failure toast).


> **🧭 2026-08-06 — the engine scoping pass: every control on the Video surface now belongs to an engine, and H3's two most expensive knobs came back out of hiding.** Mr Bizarro, on the switcher shipped an hour earlier: *"images, audio, trained character and all that are related to other things, so when Hailuo is selected they should not be around… Enhance: does it work? Avoid: does it work? No music, No voice — those are options for LTX… I see you removed the step options we had before, no?… make sure there are not other things that are escaping you."* Three separate findings in one message, and all three were right. The previous pass scoped the **mode strip** per engine and stopped there; the composer and the player were never audited.
>
> **1. The audit — every control classified by what it DOES, not what it says.** Nine controls were escaping, five of them harmful rather than merely irrelevant:
>
> | Control | Verdict | Why — grounded in mechanism |
> |---|---|---|
> | **Enhance** (`#enhanceBtn`) | **LTX only** | Its own tooltip already said it: "rewrite your prompt in the style **LTX 2.3** was trained on". On H3 it is *destructive* — H3's trained control path is `<d>[English] …</d>`, `(S1)`, `[Shot N]` and the three labelled fields, and a Gemma rewrite strips all of them. H3 prompts reach the encoder verbatim. |
> | **Avoid +** / `#avoidRow` | **LTX only** | **H3 has no negative prompt at all.** Guidance-distilled: no CFG, no unconditional branch, one forward per step; all three official ComfyUI templates ship zero negative-prompt nodes. `run_h3_job` never reads `negative_prompt`. A one-line hint takes its place saying what *does* work: refusals as plain prose, and only against what H3 adds unasked — camera drift and on-screen text. |
> | **No music** (`#noMusicPill`) | **both, re-implemented per engine** | The only composer tool that is right on both engines — both volunteer a score, neither lets you remove one afterwards. The *mechanism* differs: LTX keeps its prose audio directive; H3 now gets `non_diegetic_music: N/A`, the **trained field value** for "no score", skipped if the prompt already sets that field. Tooltip swaps with it (`data-title-ltx` / `data-title-h3` → `_syncEnginePromptTools`). |
> | **No voice** (`#noVoicePill`) | **LTX only** | What it does is drop a character's *audio LoRA* from the stack. Characters are an LTX construct. |
> | **Characters avatar strip** (`#manualCharactersPickerSlot`) | **LTX only** | The loudest leak: it lives in `t2v`, which H3 *does* serve, so it stayed on screen through the switch. A character is a pair of LTX LoRAs fused into an LTX checkpoint; `character` is in H3's `excluded_modes` for that reason. |
> | **Character quality strip** (`#qualityGroupCharacter`) | **LTX only** | Its visibility is owned by `_applyCharacterQualityStripVisibility`, which only asks "is a character selected?" — so with a character active it stayed lit **beside** the H3 tier strip. Two primary strips, both claiming to set the render. |
> | Seed · reference image · Batch · Stop-ComfyUI · Open-when-done | **both, correct as-is** | Seed: H3 resolves `-1` itself. Image: first-frame conditioning is real on H3. Batch posts `FormData(genForm)` through the same `make_job`. ComfyUI kill matters more on H3 (~40 GiB peak), not less. |
> | Duration/Frames · Orientation · W×H · LoRAs · Accel · HQ-speed · STG · Long-clips · LTX Export · Method · I2V audio source | already `data-ltx-only` | verified, unchanged |
> | Extend / FFLF / Colorize / Ingredients / Control sections | **left untagged, deliberately** | They belong to their **mode**, not to LTX. The mode gate is registry-driven (`engine_serves_mode`, enforced again in `make_job`), so a future engine that *does* serve Extend inherits the section for free. Hardcoding `data-ltx-only` there would be the bug this table exists to prevent. |
>
> All of it declarative, through the `[data-<engine>-only]` fold rules `_engine_css()` already emits from the `ENGINES` table — no new mechanism, and a third engine scopes itself.
>
> **2. Speed and Steps come back out of Customize — the regression he spotted.** He couldn't find Steps and concluded it had been deleted. It hadn't; it had been folded into the collapsed `#customizeDetails` alongside Speed and Export on an LTX-parity argument that was **misapplied**. LTX can demote its sampler knobs because LTX's primary decision is the Quality strip, which already carries them. H3's tier strip does not: **Turbo and Steps each swing wall clock by roughly 2×** (`~17-19 min → ~8-9 min` measured on Wide 5s; Auto→20 is ~2.4× the tier's time) and neither is implied by any tier chip. A control that expensive does not belong behind a disclosure triangle. Both now sit in `#h3PrimaryControls` directly under the tier strip, same ids, same handlers, same chip grammar, Speed first because Turbo overrules Steps. **Export stays in Customize** — it is an ffmpeg scale-and-pad *after* the render, costs no wall clock and changes no pixel the model produced. The `.eng-primary` class exists only because the active-pill accent rules were scoped to `.cz-body`; leaving Customize would otherwise have cost the chips their engine colour.
>
> **A display bug fell out of making Steps visible.** `setH3Turbo` pinned the shared hidden `steps` to 4 and *then* called `setH3Steps('auto')`, which re-derives it from the tier — so the derived line and the queue card read "Steps 9" on a Turbo render. Server-side was always correct (`make_job` stamps `H3_TURBO_STEPS`); this was the form lying about what it was about to do, which is exactly the thing the move was meant to fix. Order swapped.
>
> **3. The player action cluster is scoped to the SELECTED OUTPUT's engine, not the form's.** `list_outputs()` already returns `engine` per entry (lifted into the Draft→Finish work), so this costs no request on the gallery-click path. **Extend is hidden on an H3 clip**: it runs the LTX Q8 extend sampler, which would continue an H3 render with different weights, a different frame grid (8k+1 vs 17n+5) and **no audio branch at all** — the joint soundtrack that is the entire point of H3 would simply stop. H3's answer to "make it longer" is window chaining, and chaining is a **tier** — a choice made before the render, not an action on the result — so there is nothing coherent to offer. Rather than a dead button with an explanation, the pointer went where the user can act on it: `#h3Hint` under the tier strip now reads *"Length is the tier's: the 10 s / 15 s tiers chain windows at render time, so there is no Extend afterwards."* **Expand** is a lightbox — universal, untouched. **Params** already restores engine/tier/Turbo/steps. **Animate** is photo-only and its `i2v` target is a mode H3 serves, so it is correct on both. **Finish-at-HQ** was already gated on `o.engine === 'h3'` **and** a `draft: true` tier — re-verified.
>
> **4. A payload scrub, so the record matches the surface.** A fold rule hides a control; it does not empty the input behind it, and `FormData` reads the input. A user who picked a character, typed an Avoid line, then switched to H3 was still POSTing `character_id` + a LoRA stack + `negative_prompt` on a job that reads none of them. Nothing broke — `run_h3_job` ignores all three — but the queue card, the ⓘ modal and the sidecar then describe a render that didn't happen, and Load Params replays the fiction. Cleared at submit for `engine=h3`; `seed` and `image` deliberately preserved (both real on H3). The LoRA-orphan confirm is skipped on H3 for the same reason.
>
> **Validation (static — the panel was NOT restarted; a review instance holds :8199).** `py_compile`; `node --check` on the extracted inline `<script>`; **135 assertions, 0 failures** across three suites — 107 in Python via `importlib` driving the **real** module (`_engine_css` fold rule per registry id; `engine_serves_mode` on eight modes; the rendered page's tag balance asserted against **HEAD's** baseline rather than an absolute zero, see the pre-existing `<main>` note below; every JS-referenced id present exactly once with HTML comments stripped so tombstones can't inflate the count; each control's scope attribute asserted individually; Speed/Steps proven *outside* `#customizeDetails` and *inside* `.quick-settings` after `#h3TierGroup`; Export proven still inside; the CSS accent + layout rules present; and the **real `make_job()`** across clean-H3 / pinned-Steps / LTX / stale-tab-in-a-foreign-mode, confirming tier-stamped geometry, tier-derived steps rather than the LTX hidden `steps=8`, LTX post-processing neutralised, and a scrubbed H3 job carrying no character or LoRA stack), and 28 in node against a DOM stub on three helpers **extracted verbatim from the shipped script** — `_syncEnginePromptTools` (LTX copy / H3 copy / unknown-engine fallback), the No-music branch (LTX phrasing on LTX, `non_diegetic_music: N/A` on H3, **no LTX phrasing leaking onto the H3 lane**, skipped when the prompt already sets the field, no-op when off), and the payload scrub (four fields cleared on H3, `seed`+`image` preserved, LTX untouched).
>
> **Pre-existing, found in passing, NOT fixed here:** `<main class="layout">` in the page template is **never closed** — verified identical at HEAD `07d2656`, so it predates this work. Browsers auto-close it at `</body>` and it renders fine, which is why nothing has ever caught it. Worth a one-line fix on a quiet commit rather than inside a UX pass.
>
> **Remaining gate — one live click-through when :8199 frees:** flip to H3 and confirm the composer strip shows only No-music + the negative-prompt hint (no Enhance, no Avoid, no avatars); select a character on LTX, then switch to H3, and confirm **both** the avatar strip and the character quality strip disappear; confirm Speed + Steps render under the tier strip in engine colour and that Turbo still greys the Steps pills *there*; select an existing H3 clip and confirm Extend is gone while Expand/Params remain; select an LTX clip and confirm Extend returns.
>
> **📋 H3-NATIVE CONTROLS WORTH BUILDING — proposal only, nothing built.** He asked for research on what H3-specific controls we *could* add. It already exists: `~/AI/projects/hailuo-mlx/notes/H3_PROMPTING_GUIDE.md` (§1–§4, §7) and `~/AI/projects/minimax-prompting/SKILL.md`. The finding that frames all of it: ***`grep` finds no use of `<d>`, `<scenetrans>`, `overall_soundscape` or `[Shot N]` anywhere in the panel or the H3 runner*** — every clip we have ever shipped used plain prose, on an encoder that was trained on a structured format. Ranked by value ÷ effort:
>
> | # | Control | Value | Effort | Verdict |
> |---|---|---|---|---|
> | **1** | **Dialogue helper** — a small "add a line" affordance that wraps text as `The <voice description> <subject> (S1) says: <d>[English] …</d>` and appends what the face does as the line lands. | **Highest.** `<d>` / `</d>` are **genuine declared special tokens** in the shipped `tokenizer_config.json`, and `text_encoder.py` loads via `AutoTokenizer.from_pretrained` with `add_special_tokens=False` — meaning embedded specials are **already matched atomically with no code change**. Dialogue is H3's headline capability and we have never once used its trained syntax. | **Lowest** — pure client-side string assembly into the existing textarea. No server, no runner, no schema. | **Build first.** Best ratio on the table by a wide margin. |
> | **2** | **Three-field composer** — an optional structured mode giving `integrated_multimodal_description` / `overall_soundscape` / `non_diegetic_music` their own fields, concatenated on submit; free-prose stays the default. | **High.** The format the encoder was trained on (guide §1.1). Also makes the No-music toggle shipped today a first-class field instead of an append, and gives `overall_soundscape` somewhere to live. Word budget 350–500 in field 1, 1–4 sentences in field 2, 1–3 in field 3. | **Medium** — a UI mode plus one honest question: whether `prompt` stays the assembled string (Load Params + sidecars keep working unchanged) or the three parts are stored separately. Recommend **assembled**, with the parts kept as extra sidecar keys. | **Build second.** |
> | **3** | **Per-window prompts for the chained 10 s / 15 s tiers** — N textareas, one per window, joined with ` \|\|\| `. | **High and it closes a shipped defect.** The tier `note` we render today warns that one prompt is asked of every window, so a scripted line is spoken **once per window**. `--chain-prompts` on the runner **already fixes this** and the panel does not pass it (`make_job`/`run_h3_job` build `--chain-windows` + `--chain-total-frames` only). Guide §7.7 lists the artefact as **FIXED by `--chain-prompts`**. | **Medium** — one new `make_job` allowlist key (`h3_chain_prompts`), one `cmd +=`, a runner-flag probe in the `_h3_runner_has_flag` pattern, and N textareas driven off the tier's `chain_windows`. Ships with a caveat from the same table: writing the **same** soundscape text into every window, because differing audio clauses cause the ~6 dB ambience step at the seams. | **Build third** — highest *user-visible defect closure*, but it needs the flag probe and a live chained render to verify. |
> | **4** | **`<\|lyrics_start\|>` … `<\|lyrics_end\|>` — the undocumented sung-lyrics path.** | **Unknown, potentially the most interesting thing in the guide.** Declared in the shipped tokenizer, **absent from both official prompt guides** (grep: 0 hits), so nobody in the ecosystem is using it. Both guides route singing through `<d>` instead. | **Trivial to try, unbounded to productise.** | **Do the experiment, do NOT ship a control yet.** Guide §4.2 E6 is the A/B (2 × HQ-5s, ~16 min) and **step 0 is free**: load the tokenizer, print the ids, confirm `<d>` is **1 token**. If it is 3+, the whole special-token story collapses and items 1 and 4 become style experiments instead of control-path work. Run step 0 before committing to item 1. |
>
> **Two things explicitly NOT worth building,** so they don't get re-proposed: a **negative-prompt box** for H3 (there is no such field — see the audit above), and **bracketed camera commands** (`[Push in]` etc., guide §3.3 — hosted-API only, they render as prose locally). The camera *does* need naming in every prompt — silence means drift — but as prose, which the three-field composer's field 1 already guides.


Current `dev` head: see `git log -1` for the live SHA. `dev` tracks `beta/main` (private repo — see §1).

> ## 🆕 IN PROGRESS on `dev` 2026-08-03 — **Hailuo H3 as a SECOND video engine** (not pushed, VERSION untouched)
> MiniMax-H3 (FL2VA) — one prompt → picture + dialogue + sound together — wired in behind an **engine picker** in the Video tab: `LTX-2.3 | Hailuo H3`. Optional pack (~75 GB, 64 GB+ Macs, MiniMax Community License with territory restrictions); LTX stays the default and its pipeline is untouched.
> **Read `docs/H3_ENGINE.md` first** — architecture, env vars, tier rationale, dev-box wiring, troubleshooting.
> - **Subprocess engine**, like the mflux image engines: spawn the validated `scripts/generate_staged.py`, stream stdout into `push()`, read its metrics JSON. The ONE cross-engine interaction: `run_h3_job_inner` kills the warm helper first (H3 peaks ~40 GiB, both don't fit on 64 GB); it respawns lazily on the next LTX job.
> - **Queue-native**: `engine` + `h3_tier` are in the `make_job` allowlist (the known silent-no-op trap); `run_job_inner` dispatches on `params.engine` before every LTX-only clamp; output lands in `mlx_outputs/` with a normal `.mp4.json` sidecar so the gallery just works; `/stop` kills the process group (SIGTERM → SIGKILL after 8 s) and an atexit hook covers a Pinokio quit.
> - **Tiers as data** (`H3_TIERS`, served via `/status.h3.tiers`): Draft 3s 640×384/73f/9pts ~3 min · HQ 3s 768×448/73f/9pts ~4-5 min · HQ 5s 768×448/124f/9pts ~8 min · Long 10s 768×448/243f/**16pts** ~36 min. 8 forwards is free ≤ ~13k packed rows; the 10 s tier is ~25k rows where 8 forwards ghosts.
> - **Install**: `install_h3.js` (clone → own Python 3.11 venv → `download_selected.py`) + a `pinokio.js` entry gated to 64 GB+ machines; the panel's Models modal carries a matching row and an install card. The H3 venv is separate on purpose — Phosphene pins `mlx==0.31.1` (0.31.2 = −22 dB LTX audio), the H3 port needs `mlx>=0.32`.
> - **⚠️ OPEN — `--first-frame` isn't on the published branch.** Image mode on H3 needs FL2VA first-frame conditioning, which lives on a local-only branch; GitHub `mrbizarro/minimax-h3-mlx` has only `main` + `codex/practical-apple-silicon` (the pin). The panel probes the installed runner (`h3_supports_first_frame` → `/status.h3.first_frame`) and keeps Image on LTX when absent, so users degrade to Text-only rather than crashing. **Push the first-frame work and bump `H3_BRANCH` in `install_h3.js` before this ships.**
> - **Validated on dev**: py_compile + `node --check` on both JS files, panel restarted with the H3 env overrides (HTTP 200, `/status.h3` fully resolved), picker + tier markup present in the served HTML, and a real Draft job spawned → logs streamed → Stop killed the whole group clean (job `cancelled`, zero orphans). **NOT yet done: a full end-to-end H3 render + gallery/sidecar check, and Image mode.**

> ## ✅ SHIPPED PUBLIC 2026-07-24 — Remix wave = **v3.3.0** (`733e37f`, tagged). Read this first.
> Public `origin/main` = `733e37f`, VERSION **3.3.0**, tag `v3.3.0`; tree identical to beta. Promoted via a single clean `release(v3.3.0)` commit (worktree, tree brought to beta exactly, verified identical before push) crediting @anton-vsh (#29) + @anubissbe (#31). **Launch post PUBLISHED** on Pinokio: <https://beta.pinokio.co/posts/01ky9kqqzx8nte88y8s39h5bj2> (recipe graphic + demo video + UX shot; sample-character CTA woven in).
> **Ship gates cleared:** weight-manifest audit PASSED + every new IC-LoRA weight verified to resolve/fetch from its HF repo (dgrauet/DoctorDiffusion/DeepBeepMeep/Lightricks); Q4-character honesty label shipped (CSS-gated to `data-cap-tier=q4`, verified). **Sample character** (Bizarro) hosted + one-click download live.
> **NOT done (residual):** (1) a true bare-metal fresh-Pinokio-install render pass — not feasible in-session (tens of GB); mitigated by the manifest+fetchability audit + each mode's build-time validation, but worth a real clean-install smoke test when convenient. (2) saved-j's #35/#36 train-run validation (needs a real training run with their dataset). (3) HF mirror of the sample character (blocked — read-only token).
>
> --- historical (pre-3.3.0) ---
> **v3.2.7 shipped public 2026-07-23** (`0220e19`, tagged) = fixes only. The Remix wave was the next release (now shipped as v3.3.0 above).
>
> **✅ Ship gate ALREADY PASSED — fresh-install weight-manifest audit (the mosaic bug class).**
> Every weight the new features load was checked against the download manifests:
> - `ltx-2.3-22b-ic-lora-ingredients-0.9` · `ltx-2.3-22b-ic-lora-union-control-ref0.5` · `LTX-2.3-22b-IC-LoRA-Colorizer-0.9` → present in **both** `install.js` and `update.js` ✅
> - `spatial_upscaler_x2_v1_1` → present in install.js + update.js + download_q8.js ✅
> - `transformer-dev` + `ltx-2.3-22b-distilled-lora-384` → in **`download_q8.js`**, which is correct: `_kf_pipe` (keyframe interp) and `_a2v_pipe` (A2V Q8, PR #29) are Q8-tier pipelines. Not a gap. ✅
> - `audio_vae` (A2V) → in install.js Q4 manifest ✅
> **No missing-weight gap. This is the check that would have caught the 7-week Q4 mosaic bug.**
>
> **✅ DONE 2026-07-23 — shippable sample character (kills the #1 barrier: "must train a LoRA before you can try Remix").**
> - `bizarrotrn_v2.safetensors` (817 MB, the exact character from the demo clip) hosted as a **public GitHub release asset**: `https://github.com/mrbizarro/phosphene/releases/tag/sample-character-bizarro` (HTTP 200 verified; sha256 `a52e648a…`). Visual LoRA only (no voice) — the audio sibling is optional per `list_characters()`.
> - **In-app one-click** "Get a sample character (Bizarro)" button in BOTH empty states (Character-mode strip `charsEmpty` + standalone Characters grid `charactersEmpty`). Backend: `POST /characters/download-sample` (202 + bg thread, atomic .partial→rename, sha256-verified, **refuses to overwrite** an existing `bizarrotrn_v2`) + `GET /characters/download-sample/status` poll. Commit `033acb5` (beta). Verified live end-to-end (both buttons render + click cleanly; already-installed path; character discovered as "Bizarro").
> - Remix draft post updated to explain the Character requirement + point at the one-click sample.
> - ⚠️ **HF hosting was NOT possible** — this machine's HF token is READ-only (salocharly). If Mr Bizarro wants it mirrored on Hugging Face too, that needs a WRITE token he provides. GitHub release is the shipped path.
>
> **REMAINING before pushing Remix public:**
> 1. **From-zero fresh-Pinokio-install run** — the manifest audit is static analysis; still do one real clean install + one render per new mode (Ingredients / Control / Colorize / Ingredients×Character / A2V). This is the last hard gate ([[feedback_validate_from_zero]]).
> 2. **Q4-character label — DECIDED, NOT YET IMPLEMENTED.** Decision: character stays UNLOCKED on Q4 (check-in data: 8 GB = 20%, 16 GB = 25% of users → locking out 45% costs more than a softer face). Add a short "Q4 fallback — identity is approximate; Q8 gives faithful faces" note near the character quality strip. Relevant code: `setMode()` character branch ~`mlx_ltx_panel.py:23745`, quality chips ~20903/21131, `_validate_character_quality` ~5607 (server already allows Q4). Reversible one-liner if Mr Bizarro disagrees.
> 3. **Publish the staged Remix post WITH the release** — full text + live `assets.pinokio.co` media URLs backed up outside this repo. ⚠️ The Pinokio composer AUTO-SAVES and the browser draft was overwritten by the v3.2.7 post — use the backup file, not the composer draft.
> 4. Bump VERSION (3.2.13 → 3.3.0 suggested; it's a feature wave, not a patch) + tag.
> 5. saved-j's training validation — a real train run to confirm the precision fix (#35/#36) end-to-end.


> **Session 2026-07-20 — Pinokio notifications + GitHub triage + #35/#36 training fix + identity scrub (beta `3164512`, still v3.2.13, NOT public):**
> - **Pinokio inbox → zero.** Answered all 3 substantive @bizarro posts: **@hottboytank** (auto-titled "python3.11 NOT FOUND", but the real blocker is **macOS 13 Ventura** — `mlx 0.31.1` has no `macosx_13_0` wheel, so `ltx_core_mlx` never installs; also 8 GB M1, under floor → told him to upgrade to macOS 14+); **@fardad_resin** (#showcase resin-pendant Reel prompt → 3 tips: Q8 for macro detail, video can't render legible calligraphy, image→video for the hero shot); **@shaurya11** (transformers-5.13.0 crash, self-resolved → warm close). Marked all read.
> - **GitHub: PR queue emptied, every open issue answered.** **PR #29** (@anton-vsh A2V distilled) adopted onto beta (was already in `d335c7c`) + **closed** as adopted. Issues: **#43** (samuellzengkang) = the harmless stats-403 → replied + FIXED (below); **#36** (@saved-j) → replied owning the 6-day silence + real diagnosis + immediate workaround; **#34** circle-back (auto-close in a few days if silent); #35 kept open (tracks the training fix); #30/#24/#21 unchanged.
> - **🔧 #35/#36 training fix — SHIPPED PUBLIC in v3.4.1** (verified present on `main` 2026-08-05; the "needs clean-install validation" gate below was overtaken by the release train). Companion fix `59ed751` (2026-08-05): the Train preflight itself was still DOWNLOADING the quantized Q4 dev and green-lighting it at a 1 KB presence gate — now pulls the ~21 GB full-precision Q8 copy and requires ≥15 GB to show ready. `_patch_loader_prefer_dev_transformer` only checked the dev file *existed*, not its precision → a **4-bit quantized dev (~11 GB)** in the Q4 dir got loaded → near-zero LoRA (exactly @saved-j's #36). Now **size-aware**: full dev ~19 GB vs quantized ~11 GB; prefers the largest dev ≥15 GB across the Q4 dir + Q8 sibling, warns loudly on a shadowing quantized one. Error msg corrected 11→19 GB. **@saved-j's voice-LoRA regression is SEPARATE** (voice never touches the dev transformer) — asked him for the first ~40 train-log lines. **TODO: reproduce from a clean install with saved-j's attached Archive.zip, then promote.**
> - **🔒 Identity scrub (`b04fc8b`)**: `elontrn`/`Eltrumpo` (politically-loaded) appeared in **8 internal comments across ROADMAP.md + train_character.py + mlx_ltx_panel.py — and were ALREADY in PUBLIC v3.2.6.** Renamed to neutral `chartest`; comment-only, no logic change. Reaches public with the next promote.
> - **🔧 #43 stats-noise fix (`3164512`)**: the stats-dashboard fetch 403 (token scope / rate limit) was pushed as "stats: fetch failed" every cycle → users filed it as a bug. Now warns **once**, phrased as a harmless maintainer-only refresh skip, re-arms on the next success.
> - **PUBLIC-PROMOTE PUNCH-LIST (all gated on Mr Bizarro's OK + from-zero validation):** (1) **character-on-Q4 face decision** — PR #29 relaxed the Q8-forced-character gate so 16 GB users get character on Q4 at a softer-identity cost; the Q8 reference face is rendered (`mlx_outputs/cinematic_waist_up_medium_close_up.mp4`), the Q4 side is still un-rendered → DECISION NEEDED before promote. (2) validate the #35/#36 train fix on a clean install (saved-j's data). (3) **macOS-14 installer preflight** still TODO (promised @hottboytank) — fail early with a clear "needs macOS 14+" instead of the cryptic mlx-wheel error.

> **Session 2026-07-10 — P0 install-breaker fix + full GitHub triage + triage-automation repair + /stats issues board:**
> - **🔴 P0 (public): `transformers 5.13.0` breaks EVERY fresh install.** `mlx-lm 0.31.1` requires `transformers>=5.0.0` (no upper bound); after 5.13.0 dropped (~Jul 9) it broke `mlx_lm.tokenizer_utils` → Gemma encoder silently no-ops → every generation crashes (`'str' object has no attribute '__module__'` / "Model not loaded"). 4 reports (#40/#38/#37/#33). **Fixed:** pinned `transformers>=5.0.0,<5.13.0` on the mlx-lm `uv pip install` step in `install.js` (beta `e5ef1b3`). Known-good: 5.7.0 (our build) + 5.12.x. Existing users' manual unblock: `cd ltx-2-mlx && uv pip install --python env/bin/python 'transformers>=5.0.0,<5.13.0'`. **NOT yet on public main** — needs the OK to cherry-pick + push (users have the workaround). #40 kept open as the tracker; #33/#37/#38 closed as dups.
> - **GitHub backlog fully triaged:** replied to all; closed #33/#37/#38 (transformers dups) + PR #39 (mode-bar CSS, superseded by the beta Remix rework). #34 replied (empty body, awaiting details). **#35 + #36 kept open, tracking a code fix:** the trainer's `_patch_loader_prefer_dev_transformer` picks the **Q4-quantized** `transformer-dev.safetensors` (~11 GB in the q4 dir) over the **full-precision Q8 dev** (~20 GB in the q8 dir) → near-zero LoRA weights (root cause of "trained LoRA doesn't apply"). FIX TODO (validate with a real train run): prefer the full/Q8 dev + placement-aware integrity (warn on a quantized dev in the q4 dir; verify `spatial_upscaler_x2_v1_1.safetensors` present; size-check vs expected). PRs #31/#29 already reviewed (ADOPT-WITH-CHANGES).
> - **Triage automation repaired.** It wasn't just "launchd disabled" — the launchd job was retired Jul 8 and replaced by a Claude Routine (`phosphene-morning-triage`, enabled, 8am) that (a) silently no-op'd (fetch failed in the unattended env — PATH/`gh`) and (b) was **drafts-only by design** → contributors never got answered. Fixed both: (1) hardened the triage fetch script — 3-try network retry + absolute `GH=$(command -v gh || …)`; (2) rewrote `~/.claude/scheduled-tasks/phosphene-morning-triage/SKILL.md` into a **gated posting spec** modeled on the working Bloom triage: reply-on-playbook-match / label / close-dup, **escalate anything new, NEVER push code**, ≤6 posts/run, skeleton-brief-first, always leaves a brief. Reads `issue_triage_playbook.md`. Armed for next 8am.
> - **/stats now lists issues + PRs.** New "Needs attention" board (open issues | open PRs, newest-first, author/age/labels/badge) — `scripts/fetch_repo_stats.py` gains `fetch_open_items()`, `panel_assets/stats.html` gains the board section (beta `709850d`). Data in the gitignored `state/stats-data.jsonl` (non-sensitive fields only); /stats stays 127.0.0.1-only. Reads at request time (no restart needed).
> - Stats snapshot: 92★ (nearing 100), 14 forks, ~4.3k users, 95 clones/day.

> **Session 2026-07-06 — Remix launch assets + full GitHub triage (no code ship; beta unchanged at v3.2.13):**
> - **Remix launch post STAGED on Pinokio** — composer draft under @bizarro, **NOT published** (gated on the public promote). Media-first: recipe graphic + 26s demo video + panel UX shot, all re-hosted on `assets.pinokio.co`. Leads with "character LoRA × ingredients." The recipe (both graphic AND demo) shows the FULL ingredient set per gag: **[trained face] + [location] + [wardrobe] = clip** (bizarrotrn + glacier + red bikini · + desert + fur coat · + beach + ski jacket).
> - **"Funny wardrobe" hero clips → `state/hero_demo/funny/clips_v2/`** (bizarro dressed wrong for 3 climates). Framing fix that mattered: chest-up WITH headroom — the old "face filling the *upper* frame" prompt literally cropped the head off — plus exaggerated comedic action (shiver/fan/sweat) for motion. char strength **1.4**, **121f (5.0s) = single-pass clean**, two-stage 1536×896, ~11 min/clip.
> - **Wardrobe product shots → `state/hero_demo/funny/wardrobe/`** via `mflux-generate-qwen-edit` **gray-canvas text-to-image trick** (Lightning-4step LoRA, `-q 8`, ~39 GB peak → STOP the panel first). Prompt shape: "Completely replace the plain gray image. Generate a clean product photograph of <garment>, laid flat and centered on a soft light-gray studio background…". Reusable for any product/ingredient image.
> - **Pinokio media injection (the method that works — see memory `phosphene_pinokio_posting`):** `file_upload` MCP rejects self-generated files ("only files the user shared"); instead b64→clipboard→paste into an in-page sink `<textarea>`→`atob`→`File`→set on the composer's `input[type=file]`→dispatch `change`. Composer uploads to its CDN and inserts markdown **at the body textarea's cursor** (must `setSelectionRange(end)` on the body first, or nothing inserts). Then rewrite the whole body via the native value setter to place media exactly. Video embeds as `[video](url.mp4 "poster.webp")` with an auto-poster; videos → 720p, ≤25 MB.
> - **GitHub fully triaged (mrbizarro/phosphene):** all 6 issues answered; **#32/#28/#27 closed**; #30 open (tracks PR #31); #24/#21 open (FR backlog). **#32** (character-LoRA-zero-effect, near-zero weights) = the **distilled-vs-dev transformer trap** — already fixed in v3.2.5's `_patch_loader_prefer_dev_transformer` (forces `transformer-dev.safetensors`, hard-fails if missing); reporter was on an older build. **PRs #31 (48 GB memory) + #29 (A2V distilled)** both deep-reviewed → **ADOPT-WITH-CHANGES**, review comments posted, public merge **held for Mr Bizarro**. Key #31 catch: its `full_training` caps (7000 steps / 768 px) are applied **globally** → would silently cap documented 20k-step / 1024px-widescreen runs on 64/128 GB Macs; asked to scope to <64 GB.
> - **OPEN:** fresh-install-from-zero validation of the Remix wave → then Mr Bizarro's OK to promote public `main` + hit Publish on the staged post.

> **Latest dev (2026-06-28, on `beta/main`, NOT yet public) — the IC-LoRA feature wave:**
> The ltx-community IC-LoRAs that ride the in-context pipeline Phosphene already vendors (the one powering HDR) are now wired as first-class modes. **BETA only — none public yet**; each gated behind fresh-Pinokio-install validation + Mr Bizarro's explicit OK before promote + a Pinokio post. `dev`/beta is at **v3.2.11** (public is still v3.1.1).
> - **v3.2.6 — STG slider** (spatio-temporal guidance; off by default).
> - **v3.2.7 — Q4 "mosaic" root-cause fix + 2 more.** The Q4 mosaic was `spatial_upscaler_x2` missing from the Q4 manifest block (listed only under Q8) AND a Verify blind spot (it iterates each repo's `files[]`, so never noticed). Added the upscaler to q4 `files[]`+`download_include`. Plus: Ideogram silently dropping stacked LoRAs (guard only matched "qwen" in the filename → others forwarded to mflux-generate-ideogram4 → 0 keys matched + false success; now `fam=="ideogram"` drops ALL non-native LoRAs); and a latent `generate_hdr` dispatch bug (called non-existent `helper.send()/wait_done()` → `HELPER.run`).
> - **v3.2.8 — Colorize restore mode** (community DoctorDiffusion weight; B&W clip → colorized; off by default, kept minimal).
> - **v3.2.9 — Ingredients (flagship multi-reference IC-LoRA).** 2–8 reference images (face + prop + location) → one recomposed clip. The recompose lever is reference **STRENGTH** (not conditioning-attention — a CAS sweep was flat): 0.0 lets the fused LoRA carry the subjects so the model composes a NEW scene instead of copying the 2×2 sheet (MAD 4→86 across strength 1.0→0.0; only 0.0 = clean single shot). Single-stage Q4, generate at 2×. Un-gated mirror of the official weight (DeepBeepMeep/LTX-2, targeted single-file fetch — never a snapshot of the 708 GB mega-repo).
> - **v3.2.10 — Control (Union) mode.** Official UN-gated Lightricks Union-Control IC-LoRA; a control clip drives motion/structure/composition on the reference channel at FOLLOW strength (~1.0 — inverse of Ingredients' 0.0). Two-stage, Q4.
> - **v3.2.11 — Ingredients × Character (the differentiator).** Optional **Character** dropdown in Ingredients mode stacks a trained character LoRA ON TOP of the Ingredients IC-LoRA, so the SAME trained face lands in every composed scene — identity from the LoRA (cranked above the ingredients weight; relative weight is the lever), composition from the references, ref-strength stays 0.0 so the scene still recomposes. The character's trigger rides a hidden field; the server prepends it to the action if missing. Nobody upstream has shipped this. **Validated end-to-end through the real panel queue** (POST `/queue/add` urlencoded → `make_job` allowlist → dispatch stack): bizarrotrn rendered in a red puffer + neon alley, face tight, via the dropdown. Hero demo (3 scenes — guitar/neon, jacket/summit, husky/dunes; same face across all) in `state/hero_demo/` (off-repo) with an `index.html` report. References generated by Phosphene's OWN qwen-image-edit (gray-canvas text-to-image trick) — Ideogram-4/HiDream aren't downloaded on the dev box. Code: `make_job` allowlist `ingredient_char_lora`/`_strength`/`_trigger`; dispatch stacks into `ingredients_loras`; UI `#ingredientCharWrap` + `populateIngredientCharLoras()` (filters `_knownUserLoras` kind=train_character). Memory: `phosphene_ic_lora_opportunity`, `feedback_phosphene_faces_priority`.
> - **v3.2.12 — Remix IA consolidation + the "hallucination" fix + cross-Mac.** (a) **Remix mode** — the 3 IC-LoRA pills (Colorize/Ingredients/Control) were cluttering the mode bar (9 pills); collapsed into ONE **Remix** parent pill + a second-level sub-row (`#remixSubGroup`: Ingredients/Control/Colorize). Pure UI grouping — backend modes stay ingredients/control/restore (`REMIX_MODES` + a `remix` pseudo-mode in setMode that resumes the last sub-tool; sub-pills set the real mode; parent pill lit for any sub-mode; sub-row hidden off-Remix). Browser-validated end-to-end (Remix→sub-row+Ingredients, Control switches section, Text hides the row). Mode bar 9→7 pills. (b) **The dreamy "hallucination" look was SINGLE-STAGE, not Q4** — A/B proved it: running the 2nd (refine+2x-upscale) stage cleans it AND outputs 1536x896. Wired into the quality pills: **Quick/Draft = single-stage** (fast, the dreamy look on purpose); **Balanced (default)/Standard/High = two-stage** (clean, full-res). Q8 deliberately NOT used for Ingredients — it sharpened faces but tripped the bizarrotrn gold-sparkle artifact and is off-spec for the Q4-trained IC-LoRA (the right "Q8" is dev-transformer + distilled-lora-384, validated in `state/hero_demo/q8_experiment/` but parked). (c) **Cross-Mac**: all 3 Remix dispatches already clamp gen size via `tier_max_dim("i2v")` → small Macs auto-render ≤768px (no OOM); the quality pills are the memory/quality dial (Quick = lightest); High hidden on Q4 tier. Heroes re-rendered two-stage (clean, 1536x896) in `state/hero_demo/clean/`. Open follow-up: a Q8/dev "max-face" path for Ingredients as the mosaic-Mac fallback (parked — sparkle caveat + plumbing).
> - **NEXT:** fresh-install-from-zero validation of the wave; then explicit OK to promote public + the Pinokio post (leads with character × ingredients). Inpaint/Outpaint IC-LoRAs remain (need the `conditioning_attention_mask` param; the outpaint weight is already on disk).
>
> **Latest dev (2026-06-11, on `beta/main`, NOT yet public):**
> - **LoRA/character fixes — SHIPPED PUBLIC as `v3.0.13`** (beta `f4f992e` → cherry-picked to prod `ce03139`, release commit `814dc2d`, tag `v3.0.13`, 2026-06-11). Two bugs @claude3d reported on v3.0.12, fixed + validated live in the dev panel: (1) a character LoRA whose filename contains a **space** (e.g. `Annie Phosphene_v2.safetensors`) was misclassified as style-only / "No trained characters yet" even with a correct sidecar — `_CHARACTERS_ID_RE` rejected spaces and gated `list_characters()` *before* the sidecar was read. Widened the regex to `[A-Za-z0-9 _-]+` and made `_character_safe_id` URL-decode so spaced ids resolve in the `/characters/<id>/{preview,generate,delete,rename}` routes (unquote-then-validate stays traversal-safe). (2) Character tab could silently render plain T2V — a **UI desync** left the avatar's `.active` ring ON after the hidden `character_id` was cleared on a mode switch, so Generate shipped `character_id=""` → no LoRA. ("Params shows Text" is by design — `character_id` drives the LoRA stack, not the mode field.) The ring now re-renders in lockstep with the cleared selection. Plus a latent `enhancePrompt` wrong-element-id fix. **Deliberately NOT bundled with the Ideogram work below** so the fixes can ship without waiting on the gate. Validation: `/characters` lists a spaced-name char; in-browser pick→switch-away→switch-back leaves the ring OFF + field empty, re-pick ships `character_id`.
> - **Ideogram 4 engine + visual text-placement canvas — SHIPPED PUBLIC as `v3.1.0`** (2026-06-11). Open-weight 9.3B text-rendering model + a client-side bbox text-placement canvas. mflux 0.18 `mflux-generate-ideogram4`; gated `ideogram-ai/ideogram-4-fp8` (~26 GB; user needs a HF **Read** token + accept the license). **Root-caused + fixed the gated-download 401 that would have hit every user:** the image-engine mflux subprocess inherits `os.environ`, but the panel never exported its configured token there, so huggingface_hub fell back to a stale `hf auth login` cache (a different/unauthorized account) → 401. `_sync_hf_token_to_env()` now pushes the settings token → `HF_TOKEN` env (wins over the cache) at boot + per image job. Proven clean-room (fresh empty env: old path 401s, fix authenticates). Validated end-to-end: an 8-text-element field-guide poster rendered clean with every label placed per its bbox. Gate-error message reworded to name a wrong-account token instead of telling users to re-accept the license. Memory: `phosphene_hf_ideogram_account` + `feedback_fixes_must_reach_users`.
> - **Mosaic (Q4 on certain Apple GPUs) — confirmed an upstream engine bug.** dgrauet ([ltx-2-mlx#40](https://github.com/dgrauet/ltx-2-mlx/issues/40)) traced it to the **MLX 4-bit Metal kernel on specific GPU sub-families** — NOT monotonic by chip (his M2 Pro is clean; an M4 Max reporter, shdwmacca, mosaics). He gave a 30-sec `mlx_q4_check.py` repro (saved `/tmp/mlx_q4_check.py`); our M4 Max is clean (`applegpu_g16s`, all rows <1%). **Still need the script output from an affected box** (poppy0396 M3 Ultra / elbarto M3 Max / rathore M2) — relayed to GitHub #23 + #40; **still to relay on the Pinokio @rathore thread (Mr Bizarro pastes — Mac browser isn't logged into Pinokio).** Workaround: render High/Q8. Detail: memory `phosphene_ltx_pin_v0148.md`.
> - **8 AM triage cron fixed** — `fetch_morning_brief.sh` had been dead since **May 31** (exit 127: `gh` is in `~/bin`, off the launchd PATH). Fixed + verified outside this repo. Closed GitHub #18 (ronyeoh) + #19 (shakeworks).

Live URL: `https://github.com/mrbizarro/phosphene` · Linear project: `https://linear.app/hairstylemojo/project/phosphene-9c11240704bb`

This doc is the **session-start handoff**. A new Claude window entering this project should read this first, then `CLAUDE.md` (architecture), then the relevant Linear issues.

> **Authoritative engineering snapshot:** the 2026-05-31 deep stabilization review — risk register + phased stabilization plan. It is the source of truth for current bug state and supersedes the historical log in §7 below.

---

## 1. Where the code lives

**Repo split (2026-05-22):** there are now two GitHub repos:

- **Public `mrbizarro/phosphene`** — `main`-only, stable releases. The public `dev` branch was **DELETED**. Anyone with a hand-configured public-`dev` install must reinstall from `main`.
- **Private `mrbizarro/phosphene-beta`** — daily development. Holds `main` (daily dev) plus `archive/*` experimental branches.

Two clones on Mr Bizarro's Mac, both managed by Pinokio:

| Path | Branch tracked | Port | Role |
|---|---|---|---|
| `~/pinokio/api/phosphene-dev.git/` | local `dev` → `beta/main` (private) | 8199 | Active development. Most edits land here first. |
| `~/pinokio/api/phosphene.git/` | `main` → public `origin` | 8198 | Production / daily driver. Mr Bizarro's actual usage. |

The local branch is still named `dev`, but it tracks **`beta/main`** (private), NOT the deleted public `dev`. `update.js` auto-detects upstream (`@{upstream}`), so the dev clone pulls from `beta` and the prod clone pulls from public `main`.

GitHub is the source of truth (memory: `feedback_github_source_of_truth.md`). Branch policy is strict:

- Push daily work to **`beta`**, never to a public `dev` (it no longer exists).
- **Promotion to PUBLIC `main` is the gated step — NEVER push public `main` without Mr Bizarro's explicit OK** (memory: `phosphene_dev_workflow.md`).

State directories that live OUTSIDE the repo via Pinokio's `fs.link`:

- `mlx_models/` → ~63 GB of LTX 2.3 weights (Q4, Q8, Gemma encoder, PiperSR upscaler). Shared between dev and prod via symlink chain.
- `mlx_outputs/` → all rendered mp4s + sidecar JSON files.
- `panel_uploads/` → user-uploaded reference images for I2V / FFLF.
- `state/` → `panel_settings.json`, `panel_queue.json`, `panel_hidden.json`. Survives a Pinokio Reset.

A Pinokio Reset wipes the install dir but preserves all four — Mr Bizarro can Reset → Install without losing renders or settings.

## 2. Current capabilities (shipped in v3.0.6)

The May-17 Codex C+ items (Studio/Train tabs, Character 5th mode, capability tiers) are baseline now — they shipped as part of the v3.0 line.

**Workflow tabs (top nav)**
- Manual — video composer (T2V / Character / I2V / FFLF / Extend)
- Studio — image generation (was a mode chip inside Manual until 2026-05-17, commit `37c9d21`)
- Train Character — dataset → LoRA training, with Gemma 3 auto-caption + letterbox crop

**Modes (inside Manual)**
- Text — pure text→video
- Character (2026-05-17 commit `e420e3a`) — first-class mode for trained character LoRAs. Submits `mode=t2v` server-side; backend dispatches on `character_id`. Auto-stacks face + audio LoRAs, swaps the quality strip to Q8-only.
- Image — image→video (I2V)
- FFLF — first/last frame keyframe interpolation
- Extend — append seconds onto an existing clip

**Capability tier system (2026-05-17 commit `64dad87`)**
- `body[data-cap-tier="q4|q8"]` set at request time from `SYSTEM_CAPS.allows_q8`.
- `q4` (sub-48GB Macs): FFLF / Extend / Character mode pills hidden; chip strip hidden; Q8-Draft/Q8-Pro chips hidden; "High" chip in default strip hidden; skip-step toggle hidden. Manual collapses cleanly to Text/Image.
- `q8` (48GB+): full surface, Q4 still reachable via the default Quality strip for plain T2V/I2V.
- `LTX_FORCE_CAP_TIER=q4` env override lets a Q8 dev machine view the Q4 surface for testing.

**Quality dial** (mode-aware)
- Non-character T2V / I2V: `Quick · Balanced · Standard · High`. Quick / Balanced / Standard route to Q4 distilled; High routes to Q8 two-stage HQ + TeaCache.
- Character mode: 2-chip strip `Q8 Draft (736×416) · Q8 Pro (1024×576)`. Both submit `quality=high`. Default strip is hidden — character LoRAs can't fuse into Q4 distilled (mismatched sigma schedule produces identity-mushed output). Backend REJECTS `character_id + quality != high` with a 400 (commit `8b5a3cf`).
- Extend mode: 2 pills `Q8 Draft (12 steps · 64 GB safe) · Q8 Pro (30 steps · 96+ GB)`. Same labels as Character for vocabulary consistency; mechanism is the Extend-specific sampler (extend_steps + extend_cfg).

**HQ speed dial** (Customize accordion, visible only when quality=high)
- Fast — TeaCache + skip-step, ~12% faster on Q8 HQ (validated 2026-05-15 Codex contact sheet, ~426s → ~372s on a 7s 1024×576 clip).
- Exact — TeaCache only, reference quality (use if a specific LoRA / prompt looks degraded under Fast).
- The legacy Q4-distilled-only `Boost / Turbo` accel pill row was killed from the public surface (commit `e8a7f75`); the hidden `#accel` input survives for sidecar restore compat.

**Sharp upscale**
- PiperSR on the Apple Neural Engine, optional install via `install_sharp.js`

**Joint audio + video**
- Synced lip movement, footsteps, ambient bed (mlx 0.31.1 pin holds the audio fix)

**Hardware tier system**
- Compact / Comfortable / Roomy / Studio with per-tier feature gating
- Reference benchmarks throughout this doc are on **M4 Max 64 GB** (Comfortable tier)

**Other**
- CivitAI LoRA browser built-in
- LoRA picker per-row chrome: rename (sidecar-only, on-disk filename preserved), download (Content-Disposition attachment, streamed in 1 MiB chunks), companion-aware delete (also trashes the upscaled `_720p.mp4` + sidecar). Commit `0dba2dc`.
- Per-job progress bar (phase-aware, denoise-step-aware)
- Gallery with cache-bust URLs, no more black-clip race
- 80+ GB less disk than pre-Y1.024 installs (filtered hf downloads)
- Spicy mode gate (NSFW LoRAs hidden by default, opt-in toggle in Settings)

**Player + Expand lightbox (2026-05-17 commit `4987022`)**
- Player surface reads media natural dimensions into a `--media-aspect` CSS custom property on `loadedmetadata`; vertical clips letterbox correctly instead of being head-to-toe cropped by the prior hardcoded 16:9 + object-fit:cover.
- Expand button is now a real fullscreen modal (was inline-positioned and dumped the `<video>` at native dims inline).
- Aspect picker promoted out of Customize into a compact "Orientation" pill row under Quality.

**Train Character workflow (significant 2026-05-17 additions)**
- Gemma 3 auto-caption — one-click `[VISUAL]: <trigger>, <description>` per-image captioning via local `mlx-community/gemma-3-12b-it-4bit` weights (the same Gemma the prompt enhancer already downloads). New `caption_with_gemma.py` subprocess via `mlx-vlm==0.4.4` (pinned with `--no-deps` to avoid upgrading mlx-lm beyond 0.31.1). `POST /train/auto-caption`, `GET /train/auto-caption/status`. End-to-end verified at 87s for a 37-image dataset. Commit `e839bc2`.
- Letterbox crop strategy — pill row under the Quality preset. Center crop = scale-and-center-crop to square (legacy default, best for tight portraits). Letterbox = scale longer dim to target + pad shorter dim with black bars (preserves wide-shot proportions — addresses "blurry medium-long shots" issue when training on portrait-only crops). Trainer canvas stays a fixed square so the dataloader is unchanged. Commit `7a46b96`.
- Voice (audio LoRA) toggle defaults ON if a voice clip is uploaded. Commit `ea2cf02`.
- `/stop` button actually kills the training subprocess now (was a known no-op; trainer survived for hours after Stop, blocking the queue). `start_new_session=True` on both face + audio trainer Popens + SIGTERM via killpg with 8s SIGKILL fallback. Commit `b6d1222`.
- Vendored `lora_lab/` into the panel — installer-only users get training out of the box. `LTX_LORA_LAB_ROOT` env var still lets a dev iterate against `~/AI/projects/lora-lab/`. Commit `e9ce853`.

**Server-side validation (2026-05-17 commit `8b5a3cf`)**
- `_validate_character_quality(form)` runs on `/run`, `/queue/add`, `/queue/batch`. Refuses any submission with `character_id` set and `quality != "high"` with a descriptive 400 — defense-in-depth so a stale form or scripted call can't ship the broken Q4+character combination.

**Speed dial (legacy, killed 2026-05-17)**
- The pre-2026-05-17 `Exact · Boost · Turbo` accel pill row only ever fired on the Q4 distilled path; the HQ pipeline ignored it. Killed from public surface in pass 5 (commit `e8a7f75`) — the hidden `#accel` input survives so saved-state restore paths keep working. If a future Q4 lab tool needs the row back, restore it behind `body[data-cap-tier="q4"]` and wire it explicitly to the Q4 path.

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

**Image Studio (Klein/FLUX retired 2026-05-13; HiDream hidden since v3.0.3)**
- **Shipped default: Qwen-Image-Edit-2509** with a Lightning 4-step fast tier. Three tiers:
  - **Qwen-Image-Edit-2509** — Fast (Lightning, 4-step Q6 + FBCache,
    ~1:20 / image, multi-ref), Medium (8-step Q6 + FBCache, ~2:05 /
    image, multi-ref), Quality (40-step Q8 + CFG 4.0 + FBCache, ~3:50
    / image, multi-ref). (The `2511` string in the codebase is the
    Lightning **LoRA** path `lightx2v/Qwen-Image-Edit-2511-Lightning`,
    not the base model.)
- **HiDream-O1-Image-Dev is HIDDEN** since v3.0.3 (issue #15 — held
  pending its lab repo going public). The code is present and still
  reachable via a saved config or `engine_override`, just not in the
  visible dropdown. When exposed it offers Fast (3-step, ~3:45 / 4-img
  batch), Medium (6-step + FBCache, ~6 min, character-preserving),
  Quality (12-step + light FBCache, ~9 min, best detail), and lives in
  a separate one-time clone (`HIDREAM_LAB_DIR` env var or
  `~/HIDREAM-O1-MLX-LAB-active`); resolved in `image_engine.py`.
- Q6 default quantization for mflux Qwen tiers — Apple-Silicon
  community sweet spot, ~4-6% quality loss vs full precision (Q4 was
  8-12%), per-image speed gap negligible on M4 Max. Q8 reserved for
  the Quality tier.
- Image jobs go through the same queue worker as video — they appear
  in Now / Queue / Recent / Logs alongside video jobs (was: synchronous
  HTTP, invisible in panel).
- Right-pane viewer is mode-aware: `<img>` tag for image mode,
  `<video>` for video. Carousel thumbnails ditto.
- OUTPUTS gallery filter chips: All / Videos / Photos. Auto-flips
  on mode change (Photos on `setMode('image')`, Videos on
  t2v/i2v/keyframe/extend); manual clicks persist.
- Animate button on photo cells → pre-fills the i2v form (mode, image
  picker, prompt) so the user can tweak before clicking Generate.
  Pre-fill, not auto-submit.
- Pre-flight RAM/disk check rejects oversized jobs before mflux
  launches (24 GB Qwen-Edit on a Mac with 8 GB free was silently
  SIGABRTing mid-Metal). Lookup table per (family, quantize); compares
  to vm_stat free GB. `PHOSPHENE_SKIP_PREFLIGHT=1` escape hatch.
- Report-bug button (neon-pulsing icon in header pill row) opens a
  modal pre-filled with sysinfo + git branch/sha + sw_vers + last 50
  log lines, generates a github.com/.../issues/new link with
  labels=bug, optionally bundles the latest 5 .ips crash files into
  /tmp/phosphene-bug-TS.zip.
- FLUX / Klein-Edit / Klein-Base-Edit were retired 2026-05-13 — the
  flux2_edit family stopped being competitive with Qwen-Edit at the
  same step counts, and dropping it freed UI vocabulary for the Qwen
  Fast/Medium/Quality tiers above. (HiDream was later hidden in v3.0.3,
  so the visible dropdown today is Qwen-only.)

**Frontend extraction (parked on `archive/frontend-extraction`, private beta)**
- `webapp/` directory: index.html, style/all.css, js/main.js,
  vendor/marked.min.js + dompurify.min.js (MIT/Apache licenses)
- Panel slimmed from 16,223 → 5,866 lines (-10,357)
- New `/webapp/*` static route, `/api/page-config` endpoint
- Markdown rendering swapped to `marked.parse + DOMPurify.sanitize`
- Validated end-to-end on port 8210. Post-split this is an `archive/*`
  experimental branch on private `phosphene-beta`, not a merge-to-`dev`
  candidate. Note: a separate `cuda-port` branch (Phase 0 spike only)
  also lives on private beta — **not shipped, not on the Mac product.**

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

**"v2.0.6" dev codename** (May 8 2026; shipped as part of v3.0.0) — Image Studio overhaul + agent quality pass + security review. Headline ships:
- Image jobs flow through the unified queue worker (Now / Queue / Recent / Logs); the in-Studio gallery is gone, the unified Recent tab covers it.
- Mode-aware right-pane viewer + OUTPUTS gallery (All / Videos / Photos chips, auto-flip on mode change, "Animate" button on photo cells pre-fills i2v).
- Q6 default quantization for mflux + non-distilled photoreal presets (`flux2_edit_high`, `qwen_edit_high`, `kontext_high`); klein-4B prompt structure taught to the agent (subject → environment → style → technical hierarchy). *(This session bumped the Qwen-Edit default toward 2511, but the **shipped base model is 2509** — see §2; `2511` is the Lightning LoRA path only.)*
- `submit_shots` plural tool — agent batches a multi-shot plan in one dispatch + finishes the turn before auto-pause kills the engine (used to crash mid-batch).
- Phase C i2v prompt-writing rules in `prompts.py` (forbid still-prompt reuse, require explicit motion beats, ~1 beat / 2-3 sec); production recipe taught as Balanced + Sharp 720p.
- Pre-flight RAM/disk check + Metal/MLX SIGABRT detection with actionable OOM hint (no more silent exit -6).
- Report-bug button — neon-pulsing icon, opens pre-filled GitHub issue with sysinfo + git sha + last 50 log lines + optional .ips crash bundle.
- Manual mode genuinely hides the AF pane (was a `display:flex` vs `[hidden]` no-op).
- Agent header strip rebuilt; Outputs photo/video filter wired from `setMode`.
- Composer in Image Studio restyled to match the video form's polish.
- Agent now switches engines via `engine_override` arg on `generate_shot_images`.
- Security review pass: 0 CRITICAL, 4 HIGH, 6 MEDIUM identified — all 10 shipped this session. (See Known bugs section for details.)

> Note: "v2.0.6" was the dev codename for the work that ultimately shipped as **v3.0.0**. There is **no `v2.0.6` tag** — the published 2.x tags stop at v2.0.5.

**v3.0.0** (May 23 2026) — Marquee release. Folded in the full Characters workflow (Train Character tab + first-class Character mode), Voice/audio LoRAs, the standalone Image Studio (Qwen-Image-Edit-2509 default + Lightning 4-step), A2V, the Codex C+ capability-tier UI restructure, and the Image Studio overhaul above. The in-panel agentic-flows chat was retired in this release.
**v3.0.1** — FFLF crash fix.
**v3.0.2** — Boost/Turbo accel restored after a 2-month silent regression (git-archaeology'd the per-mode tail + re-anchored the accel patch across all `denoise_loop` import sites).
**v3.0.3** — HiDream hidden from the visible dropdown until its lab repo is public (issue #15). Code stays, reachable via saved config / `engine_override`.
**v3.0.4** — CivitAI SSL fix.
**v3.0.5** — A2V `frame_rate` kwarg signature shim (issue #5).
**v3.0.6** — Deep-review hardening: CivitAI token-leak fix, dead-HDR revival (`HELPER_LOW_MEMORY` → `LOW_MEMORY` NameError), GPU-contention guard (inline image vs in-flight video render), boot-time version-gate against the ltx-2-mlx pin.

**Published tags (verified `git tag`): `v2.0.0`–`v2.0.5` and `v3.0.0`–`v3.0.6`.** Current public release is **v3.0.6**. (The `VERSION` file in-repo may lag the tag — read the tag, not the file, for "what release is this.")

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
├── ltx-2-mlx/                         ← upstream MLX port, PINNED v0.14.0 (clone of dgrauet/ltx-2-mlx; SHA b35254a)
│   └── env/                           ← Python 3.11 venv (uv-managed)
├── mlx_models/                        ← weights (~63 GB, fs.link symlink)
├── mlx_outputs/                       ← rendered mp4s + sidecars (fs.link symlink)
├── panel_uploads/                     ← user reference images (fs.link symlink)
├── state/                             ← panel_settings/queue/hidden.json (fs.link symlink)
├── cache/                             ← HF_HOME for downloads
└── logs/                              ← Pinokio's own command-execution logs
```

`mlx_ltx_panel.py` is the heart of it — almost all panel behavior lives there. `mlx_warm_helper.py` is the long-running inference subprocess. `patch_ltx_codec.py` is a runtime modifier that fixes upstream code without forking it.

> The vendored `ltx-2-mlx` is pinned at **v0.14.0** (SHA `b35254a`). The deep review flags the **runtime-monkey-patch + version-skew axis as the single top fragility** — the panel/helper patch a moving upstream at runtime, and (pre-Phase-0) nothing asserted the imported `ltx_pipelines_mlx` was actually v0.14.0. The stabilization plan there (Phase 0 loud version gate, Phase 3 retire the runtime-patch class) addresses it.

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
- **Faces below ~80 px in-frame size** — model fills a face-shape but identity-broken. Wide shots of single characters are unusable in their first/last seconds. ([Mr Bizarro's discovery May 4](#))
- **Multi-shot continuity is naive-failure** — same prompt + different seed = different person. The mom-kid scene experiment (M1 / M2 / M3 in `mlx_outputs/`) confirmed three different women across three angles despite identical character description.

### What earns 20 seconds
- 6–9 explicit beats described in the prompt. Anything less and the model fills with stasis.
- Static or near-static camera. Camera motion costs visual coherence.
- Specific named actions ("she turns slowly", "she breathes out", "the streetlight flickers off") give anchor points.

### Empirical experiment outcomes

- **M1/M2/M3 mom-kid trio** (1024×576, Balanced + Turbo + Sharp, ~21 min each): demonstrated multi-shot character drift problem. Three different women across three angles.
- **N1–N10 cinematographic moments** (May 4): ten 20-sec clips at varying shot scales. Tested medium / wide / two-shot composition with body-language-only prompts (no hands, no held objects). Output quality varied; faces are stable when in the safe pixel range.
- **E-DRAFT** (May 4): tested low-res draft → high-res commit hypothesis. Same prompt + seed at 640×480 vs 1024×576. Mr Bizarro: low-res output not usable due to face-distance issue. Premise was flawed because lower res = worse faces.
- **E-ANCHOR** (May 4): I2V from M1 frame to test character anchoring. Result was inconclusive in the session; final clip is at `mlx_outputs/` if needed for review.
- **20-sec single-clip viability** (May 4): confirmed at Balanced 1024×576 + Turbo + Sharp. ~21 min wall, audio synced, characters stable.

## 7. Known bugs

> **Source of truth for bug state: the deep stabilization review** (full verified risk register + phased plan). The list below is the short reconciliation; the review has the complete severity-ranked register and the fix directions. The CHANGELOG further down is the historical record of what was fixed when — not a list of live bugs.

### Currently open

Reconciled against the deep review (2026-05-31). Everything that was a recent fire — the I2V/Extend post-decode hang (`94bd696`), FFLF crash (v3.0.1), Boost/Turbo regression (v3.0.2), A2V `frame_rate` kwarg (v3.0.5), the HDR NameError / CivitAI token leak / GPU contention (v3.0.6) — is **fixed**. The HiDream no-deadline reader now has a panel-side watchdog too (v3.0.6). What genuinely remains:

**Root-cause fragility (deep review §2, top of the register)**
- **Runtime-monkey-patch / version-skew axis** — the panel + helper patch a moving upstream (`ltx-2-mlx`) at runtime. v3.0.6 added a boot-time version gate (Phase 0); the structural fix (retire the runtime-patch class, re-pin to a SHA-pinned submodule) is Phase 3 of the plan, not yet done.
- **No test seam** — `mlx_ltx_panel.py` is ~28k lines, all 69 routes dispatched via a flat `if path==` chain with zero unit-testable surface. This is *why* regressions reach users. Phase 2 carves the first seams.

**Confirmed correctness / robustness items still open**
- **Recipe-override guardrails** — advanced LoRA-training overrides (rank/lr/steps) bypass the validated-recipe clamps (`train_character.py:135-144`, panel `:4807-4821`). Needs whitelist + a "non-standard recipe" warning.
- **`/status` polling cost** — every poll, per open tab, does a pgrep subprocess + filesystem scans (`mlx_ltx_panel.py:7845-7919`). Split fast fields from slow install-state probes / cache the slow group.
- **154 silent `except: pass` sites**, including persistence + chmod paths (`mlx_ltx_panel.py`). Tier them (best-effort vs state/security vs control-flow) behind a greppable `_swallow(label)` helper.
- Plus a tail of Med/Low items in the review (HiDream preflight exemption, `/prompt/enhance` shares the render lock, stats JSONL unbounded/growth, `character_runtime.py` dead/divergent, BFL URL host-validation, voice-silence trim). See the review's §2 table for the full set + fix directions.

**Model-capability limits (not bugs — won't be "fixed", design around them)**
- **Multi-shot character continuity is naive-failure** — same prompt + different seed = different person. IC-LoRA (deep review §6) is the proposed lever.
- **Faces below ~80 px in-frame** identity-break; **hands / held objects** and **high-motion physics** are out of distribution. See §6.

**Agent caveat (carried over)**
- **Qwen 3.6 reasoning loops** when planning large multi-shot batches — recursive chain-of-thought can exhaust any token budget. Workarounds: prefer Gemma 12B (`mlx_models/gemma-3-12b-it-4bit`, no reasoning blocks, 7.5 GB) or the Anthropic API for 20+ shot batches; trigger 5 shots at a time in plain text. (Note: the in-panel agentic-flows chat was retired in v3.0.0; this applies to the remaining `/prompt/enhance` + any external-engine use.)
- **KTDS install case** (Linear HAI-156): `ModuleNotFoundError: ltx_pipelines_mlx` after a "green" install. Likely the old v2.0.2/v2.0.3 em-dash sanity-check bug (install went green for the wrong reason); fixed in v2.0.4. The v3.0.6 boot version-gate + `--force-reinstall` install determinism should close this class. Pending the user's log tail to confirm.

---

### CHANGELOG (historical — what was fixed, newest first)

> **Read the gap honestly:** this section stopped being maintained after v3.0.6. Everything
> between v3.0.7 and v3.7.0 was released with its notes in the `release(vX.Y.Z)` commit body
> and on <https://github.com/mrbizarro/phosphene/releases>, which is where the per-release
> record actually lives (there is no `CHANGELOG.md`). The dated blockquote log at the top of
> this file is the engineering handoff, not a user changelog. v3.8.0 gets an entry here
> because a release that changes the install path deserves a line somebody can find with grep.

#### v3.8.0 — Storyboard, and the install that had been failing for everyone new
- **Storyboard tab** — a concept sentence is planned into a shot list, edited by the user, then
  rendered through the ordinary queue, grouped by pipeline rather than story order. The planner
  runs on the `gemma-3-12b-it-4bit` weights the panel already downloads, so the feature costs
  no new bytes. Plus the face law (`face` as a closed enum; person-silhouettes 5 → 0 across a
  56-shot corpus).
- **Install + Update fixed (no issue number — it hit everyone)** — hatchling 1.32.0 made
  `readme = "../../README.md"` a hard error and all three vendored `ltx-*` packages declare
  exactly that, so every fresh install and every Update click had been dying at
  `metadata-generation-failed` on every pinned tag. Both paths now pin the build backend
  through `pip-build-constraints.txt` (`hatchling<1.32`) **via uv** — `PIP_CONSTRAINT=` is dead
  on modern pip. The update.js half of that was found during release validation and did not
  exist on this branch until the port-back; dev's Update path was broken for every beta user.
- **#52 dead update pill (@Morac2)**, **#46 audio start offset (@blackest)**, **#53 stretched
  first keyframe (@Morac2)**, H3 menu opening at 46 GB, the model-version registry, and the
  analytics truth pass + geoip kill. Full text in the release commit body.
- **Shipped WITHOUT LTX-2.5** — the `ltx25` entry was curated out of the release tree and 2.3
  is the only generation there. That curation is public-only; this branch keeps 2.5 as the
  default until v4.0.

#### v3.0.6 — deep-review hardening
- **HDR action un-deaded** — `generate_hdr` referenced undefined `HELPER_LOW_MEMORY` → NameError killed every HDR job. One-token rename to `LOW_MEMORY` (`mlx_warm_helper.py`).
- **CivitAI token-leak fix** — downloader leaked the API token on redirect + used a weak `endswith("civitai.com")` host check. Exact-host allowlist + redirect handler that strips `Authorization`; prefer header over `?token=`.
- **GPU-contention guard** — inline `/image/generate` and the video worker shared no GPU lock; a concurrent mflux + LTX render could OOM the Mac. Now mutually exclusive.
- **Boot version-gate** — helper reads `ltx_pipelines_mlx` version at boot, compares to the expected v0.14.0 pin, and surfaces a loud panel-log banner on mismatch (Phase 0 of the stabilization plan).
- **HiDream `select()`+deadline reader** — the HiDream subprocess reader had no deadline (a hung render blocked the queue forever); now reuses the mflux deadline+`killpg` loop. (HiDream is dropdown-hidden but reachable via saved config / `engine_override`.)

#### v3.0.5 — A2V kwarg signature shim (issue #5)
- **A2V died ~10 s into every render with a reference image** (`combined_image_conditionings() missing 1 required keyword-only argument: 'frame_rate'`). Upstream v0.14.0 made `frame_rate=` mandatory, but `a2vid_two_stage.py` / `lipdub.py` don't forward it. Fixed via runtime monkey-patch `_install_a2v_frame_rate_patch()` (`frame_rate=24.0` default, idempotent). Commit `681f429`. (The shim is still required at v0.14.8 — the upstream bug is live; deep review flags hardcoding 24.0 vs the real fps as a Med item.)

#### v3.0.4 — CivitAI SSL fix

#### v3.0.3 — HiDream hidden (issue #15)
- HiDream removed from the visible Image Studio dropdown until its lab repo is public. Code stays; reachable via saved config / `engine_override`.

#### v3.0.2 — Boost/Turbo accel restored (2-month silent regression)
- Git-archaeology'd a regression where the Boost/Turbo accel path silently stopped firing. Re-anchored the accel `denoise_loop` replacement across all import sites + restored the per-mode tail. (Commit `2694f9f` shipped the issue-#12 install gate earlier; accel fix is the v3.0.2 headline.)

#### v3.0.1 — FFLF crash fix
- **Extend downscale crash** — `_ensure_downscaled` wrote to `<name>.mp4.partial`; ffmpeg can't infer mp4 from `.partial`. Added `-f mp4`. Commit `736ca0d`.
- **Image Studio submitted Qwen-Image-Edit jobs when the add-on wasn't installed** (issue #12). `/image/engine_status` now returns `family_installed` per engine; the engine pill turns red with an install tooltip; Generate refuses upfront. Commit `2694f9f`.
- **Silent panel boot when helper venv missing** (issue #5 footnote) — now logs a single stderr warning naming both probed paths + the `LTX_HELPER_PYTHON` override. Commit `fa17c61`.

#### v3.0.0 (May 23) — marquee release
The Characters / Voice / Image Studio / A2V release. Folded in the May-17 Codex C+ UI restructure, the Train-tab + LoRA-chrome work, the Image Studio overhaul, and the post-decode-hang fix. The in-panel agentic-flows chat was retired here.

**Stats dashboard — panel-internal (private).** The dashboard is served by the panel at **`http://127.0.0.1:8199/stats`** (127.0.0.1-only; panel must be running). Data lives at **`state/stats-data.jsonl`** — gitignored, on the user's Mac only, never on the public repo. `panel_assets/stats.html` holds the template (public code; only the data is private). Panel background thread `stats_fetch_loop` runs `scripts/fetch_repo_stats.py` once at startup (if data is missing/stale ≥ 6h) and daily thereafter. Token resolution (first hit wins): `PHOSPHENE_REPO_STATS_TOKEN` → `GH_STATS_TOKEN` → `GH_TOKEN` / `GITHUB_TOKEN` → `gh auth token`; skipped silently if none. Zero setup for the user — just open `/stats`. See `scripts/STATS_DASHBOARD.md`. *(An earlier GitHub-Pages + committed-JSONL dashboard and an opt-in analytics module were both rolled back before launch; analytics removed entirely in `da1d6f5`. The brief window where stats data touched the public repo (`151d0d2`..`827c5d8`) held one snapshot of public aggregate counts only — nothing private leaked.)*

**Post-decode hang FIXED** (commit `94bd696`). A first attempt at an in-helper daemon-thread watchdog (`adc1cd2`) did not fire in practice — Metal's command-buffer completion handlers block every Python thread's GIL during the deallocator chain, so the watchdog was starved by the very thing it was meant to escape. Working fix rescues from **the panel** (separate process, GIL irrelevant): `WarmHelper._build_post_decode_panic` returns a `(log_hook, panic_check)` pair; `log_hook` spots upstream's `[Decoding ... done in X.Xs]` and arms a 45s grace clock; `panic_check` runs every 500ms and, if grace expired + output file on disk > 8KB, SIGKILLs the helper and returns a synthetic done event. Helper respawns on the next job (~30s). Armed for `generate` (T2V/I2V Balanced) + `extend` only. Validated on the 768×416 +6f Extend that previously hung 5-13 min. Bundled: Extend default steps 12→8 + TeaCache threshold 0.5→0.7 (~6 min wall); gallery `_dnWxH` dn-cache leak fix (the spurious "21:19" duration label).

**Codex C+ UI restructure (2026-05-17, 30+ commits).** Driven by Codex's C+ recommendation (Q4 vs Q8 as separate surfaces). Per-bug:
- Player aspect-ratio cropped vertical clips (`.player-surface` hardcoded 16:9 + `object-fit:cover`) → read natural dims on `loadedmetadata` into `--media-aspect`, height-driven sizing for verticals, `object-fit:contain`. Commit `4987022`.
- Expand button was inline-positioned, not a modal (`.expand-lightbox` had no CSS) → real fullscreen modal. Commit `4987022`.
- `/output/delete` orphaned the raw mp4 after upscale → collects every companion via sidecar fields + `UPSCALE_TAGS` heuristic. Commit `0dba2dc`.
- `/sidecar` 404'd on a raw card after upscale → now walks the `UPSCALE_TAGS` family. Commit `331795a`.
- `/stop` didn't kill the training subprocess (trainer Popens inherited the panel's process group) → `start_new_session=True`, `STATE["train_pgid"]` tracked, SIGTERM via killpg + 8s SIGKILL fallback. Commit `b6d1222`.
- `/queue/batch` rendered 5s clips when curl sent `duration=10` (client-side duration→frames math skipped) → derive frames via `_duration_to_8k_frames`. Commit `038a0a1`.
- Train Character High preset subtitle skew (`~4 h · 768px` vs canonical `5000 steps · 512px · ~2h50m`). Commit `4255f12`.
- Dual quality strip rendered on top of each other (`.quality-strip{display:grid}` outranked UA `[hidden]`) → `.quality-strip[hidden]{display:none!important}`. Commit `7bd5057`.
- Train voice toggle defaulted OFF even with a clip uploaded → defaults ON. Commit `ea2cf02`.
- `caption_strategy="user_provided"` rejected by the trainer → alias map in lora-lab (`b04eaab`); panel-side defense-in-depth `8b5a3cf`.
- Q4-distilled inference of dev-trained character LoRAs gave generic output (wrong base) → UI forces Q8 chips for characters; backend rejects `character_id + quality != high`. Commits `1d7983a`, `8b5a3cf`.
- HQ-speed Fast pill inactive at boot (boot cascade cleared `.active`) → commits `1056c99`, `04d2ffd`; the HQ-speed pill in Customize is now the single source of truth.

**Image Studio + agent quality pass (the "v2.0.6" codename work, ~18 commits).** Per-bug:
- klein-4B prompt-structure mismatch + Q4 default → taught the subject→environment→style→technical hierarchy in `agent/prompts.py`; Q6 default in `ImageEngineConfig.mflux_quantize`.
- Image jobs invisible in Now/Queue/Recent/Logs (`/image/generate` bypassed the queue) → routed `mode='image'` through `make_job` + `run_image_job_inner`; `_IMG_STUDIO_LOCK` arbitrates the sync agent path.
- Redundant in-pane Image Studio gallery deleted (unified Recent tab covers it).
- i2v "barely moves, just a zoom out" (agent reused the still prompt) → "Phase C — writing prompts FOR i2v" rules in `agent/prompts.py` (forbid still-prompt reuse, require explicit motion beats).
- 400×400 still-output mystery (`flux2_edit` referenced in saved configs but never wired) → added to `MFLUX_FAMILY_BIN`/`MFLUX_FAMILY_DEFAULTS`, refs routed to `--image-paths`; added non-distilled photoreal presets. *(FLUX/Klein later removed 2026-05-13.)*
- Issue #2 (Metal abort crash) — mflux SIGABRTs (exit -6, uncatchable) when 24 GB Qwen-Edit runs with 8 GB free → pre-flight RAM/disk check + SIGABRT detector with an OOM hint + Report-bug button.
- `auto_pause_during_renders` killed mid-batch agent calls → `submit_shots` plural tool batches the whole plan + a `_finish_after_turn` flag.
- `submit_shot` coerced invalid accel values silently → strict validation; "exact" accepted as a friendly alias for "off".
- Agent obeyed broken user instructions (e.g. `aspect="1:1"` on a 16:9 i2v) → "push back when instructions produce broken output" rule.
- AF pane stayed visible in Manual mode (`[hidden]` overridden by `display:flex`) → explicit `style.display` toggling.

**Security review pass — 0 CRITICAL · 4 HIGH · 6 MEDIUM, all 10 shipped.**
- HIGH — reject `Origin: null` in `_is_local_request`; validate `mflux_python_path` at save; validate `model_path` at `/agent/local/start` (HF `<owner>/<name>` against an owner allow-list; local paths must resolve under `mlx_models/` or HF cache); cap `submit_shot`/`submit_shots` calls per turn.
- MEDIUM — `/sidecar?path=` requires a media file before serving the `.json`; `/agent/models/install` reuses the owner allow-list; `_save_settings` writes with O_EXCL + fsync + os.replace + chmod 0600; `inspect_clip` prompt fields wrapped + truncated (prompt-injection defang); `read_document` PDF branch rejects > 50 MB + 30 s watchdog; `/output/hide` containment check.

**Agentic-flows polish (the "v2.0.5" codename work).** Stage stuck at 0% (progress schema changed flat-float → object); offline banner restyled; typing indicator now refreshes every 1.5s with elapsed seconds; auto-scroll switched to scroll-pinning + "↓ New messages" pill; abort on long turns via AbortController; tool cards de-emphasized; anchor un-pick on re-click; "Queue them" batch pill; multi-take `append:true`; OOM memory guard (refuse engine auto-spawn >92% pressure / >8 GB swap); **reasoning-model empty-content fix** (Qwen 3.6 splits `reasoning`/`content`; bumped `max_tokens` 3072→8192, engine.chat() reads reasoning, raises on length truncation); Phosphene-branded assistant avatar. Also three Phase 0 agentic items: engine-readiness banner (`7334836`), turn-summary chip (`134b5b1`), inline wall-time predictor (`43a7c3b`).

#### Semver 2.x line (May 3–6)
- **v2.0.0** (May 3) — marquee release; semver versioning starts.
- **v2.0.1** — Spicy mode toggle gates NSFW LoRA visibility.
- **v2.0.2** — install fails loud when pipeline packages are missing.
- **v2.0.3** — install log self-documents the Python toolchain.
- **v2.0.4** (May 5) — strip em-dash from the install.js sanity check (Pinokio shells mangled the unicode em-dash → false SyntaxError → every install failed). Pure ASCII now.
- **v2.0.5** (May 6) — drop the `print('venv OK: ...')` decoration from the sanity-import step (a user-side rewriter was cutting `OK:` out of the string and appending `OK` after the shell quote). The import exit code is the only success signal needed.

#### Pre-semver Y1.NNN line (Apr 28 – May 3)
- **Y1.001 → Y1.013** — first usable T2V/I2V; audio SHIP-BLOCKER fixed by pinning `mlx==0.31.1` (0.31.2 attenuated the vocoder by 22 dB).
- **Y1.014 → Y1.024** — hardware-tier system, Boost/Turbo speed modes, CivitAI LoRA browser, Q8 two-stage HQ tier, FFLF + Extend, `hf_transfer`, Q4/Q8 download filter (~80 GB saved on existing installs).
- **Y1.025 → Y1.035** (Codex-led) — Sharp upscale (PiperSR), I2V tail-stall fix (`Y1.034`, free DiT before VAE decode), VAE temporal-streaming (`Y1.035`), license/install hardening.
- **Y1.036 → Y1.039** — fixed the `Y1.024` Extend regression (route to Q8), VAE auto-streaming threshold (recovered ~7% on short clips; the `Y1.034` patch had tiled even short clips for a ~30 s tax), Now-card progress-bar rewrite (phase + denoise-step aware), gallery black-frame race fix.
- **S2 noir dialogue attribution swap** — wrong character delivered "Same thing, honey"; root cause was prompt format diverging from LTX docs. Linear HAI-152.
## 8. Open work / future direction

Everything below is also tracked in Linear (HAI-150 → HAI-158 under the Phosphene project). This section duplicates the most current state for fast scan.

### Loose ends from May 8 session

- **Qwen-Image-Edit-2511 weights download paused** at ~54 GB partial in `cache/HF_HOME/hub`. User OK'd to keep when it completes; the old 2509 cache (~54 GB) should be deleted once 2511 is intact. Resume the download at the next session start.
- **Issue #2 (Akossimon Metal abort)** — SIGABRT detection + pre-flight RAM check shipped, but awaiting user repro details to confirm the fix lands their case.
- **L-tier security items still open** — especially L2 (anchors / select containment) from the May 8 audit. Re-run the audit on a fresh `/tmp/phos_audit/security-review.md` (the old one is gone with the next reboot).
- **Image Studio "auto" engine pill** still shows the literal string "auto"; should resolve to the actual saved-engine status server-side and display the resolved name.
- **Dead code cleanup** — `_imgStudioRefreshLibraryLegacy` + `imgStudioCopyPath` (~40 lines) can be deleted in a follow-up pass; they're vestigial after the unified Recent tab landed.

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

Mr Bizarro also has Claude.ai / ChatGPT deep-research running (May 5) on inference speed without quality loss.

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
- Awaiting Mr Bizarro's launch timing call.

## 9. Hard constraints (don't violate)

- **Apple Silicon (M1+) only**. No PyTorch, no CUDA, no MPS shim. Native MLX or it doesn't ship.
- **Joint audio + video must remain**. That's the differentiator vs Wan / Hunyuan / Mochi. We don't drop audio for length.
- **Existing queue + helper + patch architecture stays intact**. No new microservices.
- **Branch policy** (post-2026-05-22 split, see §1): there is **no public `dev`** branch — it was deleted. Daily work goes to private `beta/main` (the local `dev` clone tracks it). **Promotion to PUBLIC `main` is the gated step — only with Mr Bizarro's explicit OK.**
- **Mr Bizarro's voice in writing**: copy-edit, don't rewrite. See memory file `feedback_copy_edit_dont_rewrite.md`. Tweets, posts, README copy — fix typos and grammar, never restructure or stack value-prop language.

## 10. Memory pointers (for next-Claude)

See local memory files for the cross-cutting workflow context — branch discipline (`phosphene_dev_workflow`), Linear credentials (`phosphene_linear_project`), writing style feedback (`feedback_copy_edit_dont_rewrite`, `feedback_writing_style`), source-of-truth discipline (`feedback_github_source_of_truth`), memory-save reflex (`feedback_dont_ask_to_save_memory`), shared infra (`claudio_repo`), historical MLX/Comfy decisions (`ltx_video_setup`).

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

1. `cd ~/pinokio/api/phosphene-dev.git/`
2. `git fetch origin && git status -sb` — surface any drift first
3. Read this file (`docs/STATE.md`) AND check `git log --oneline dev -25` — recent commits move faster than this doc; the v2.0.6 May 8 batch is a good example (~18 commits in one session). Read `CLAUDE.md` for architecture.
4. Skim Linear `HAI-150` through `HAI-158` for state of each workstream
5. Check the dev panel is alive: `curl -s http://127.0.0.1:8199/status | python3 -m json.tool | head -10`
6. Last 5 commits on dev: `git log --oneline -5 dev`

If you find on-disk state contradicts this doc (paths moved, commits diverged), surface that to Mr Bizarro before working around it. Updating this doc at session-end is part of the loop.
