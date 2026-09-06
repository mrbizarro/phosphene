# Phosphene panel architecture

This document describes how the panel's code is laid out, how the server
and the browser halves talk to each other, and — most importantly —
**where new code goes**. It exists because `mlx_ltx_panel.py` grew past
72,000 lines with the entire frontend embedded as one Python string, and
that shape produced real shipped bugs: two parallel sessions each built
the same feature, in the same file, and nothing noticed until the field
did. The panel is being restructured incrementally; this document is
updated with every slice that lands and always describes the CURRENT
state, with the plan noted where it differs.

## The two halves

```
mlx_ltx_panel.py        ← the server: HTTP routes, job queue, engine
                          registry, tiers, settings, analytics
webapp/                 ← the frontend, as plain files served from disk
├── index.html          ← the page markup. The inline <script> is ONE line —
                          `const BOOT = __BOOTSTRAP__;` — the substitution
                          seam that must pass through page(). All other JS
                          lives in the modules below, whose tags follow in
                          the order the regions sat in the original block.
├── js/                 ← the page's ES modules (slice 3, landed), in tag order:
│   ├── boot.js         ← BOOT-derived consts, keyframe mode, output filters
│   ├── stage.js        ← Image Studio + the Ideogram text-placement canvas
│   ├── characters.js   ← Audio→Video, Characters and Train panes + quality pills
│   ├── engines.js      ← engine registry, H3 shape, header switcher, tier axes
│   ├── queue.js        ← form submit, pickers, poll()/status render, outputs
│   ├── settings.js     ← models card, tier gates, bug/settings modals, pills
│   ├── loras.js        ← LoRA picker, video lanes, manual characters picker
│   ├── preview.js      ← live preview, character controls, CivitAI/models modals
│   ├── health.js       ← version pill, update pop-up/banner, health chip
│   ├── storyboard.js   ← the Storyboard tab
│   ├── editor.js       ← the timeline editor
│   └── main.js         ← the kickoff sequence — ALWAYS the last tag (see below)
└── style/
    └── panel.css       ← ALL panel CSS (slice 1, landed)
mlx_warm_helper.py      ← the render subprocess (JSON over stdin/stdout);
                          not part of the web frontend at all
ltx_windows.py          ← PURE: the sliding-window schedule and per-window
                          prompt contract for long LTX clips; the panel turns
                          it into one generate + N extend calls
storyboard_editor.py    ← the Editor's document model: clips, overlays
                          (cards and titles), transitions, the mix — every
                          accessor the preview, render and export share
panel_assets/
└── stats.html          ← the /stats dashboard (predates webapp/; same
                          served-from-disk idea)
```

No build step, no bundler, no framework, no TypeScript. Files under
`webapp/` are served verbatim by the panel's `/webapp/` route
(path-bound to the directory, `Cache-Control: no-cache` because they
change on every git pull under a running panel and carry no cache-bust
token).

### The restructuring plan (what's landed, what's next)

1. **CSS → `webapp/style/panel.css`** — LANDED. The `<style>` block is
   gone from the embedded page; `test_panel_assets.py` pins that it
   stays gone.
2. **Markup → `webapp/index.html`** — LANDED. The panel reads the file
   ONCE at import into `HTML` (deliberately not per-request: the page a
   running panel serves is the build its process booted with — a git
   pull under a running panel must not hand new markup to old code) and
   `page()` applies the placeholder substitutions unchanged. The
   substitution seams (`__BOOTSTRAP__` etc.) stay in the file on disk.
   Byte-verified: `page()` output identical before/after, modulo the
   build-stamp SHA. Transitional bridge: `scripts/extract_panel_js.py`'s
   `panel_source()` returns the `.py` and `index.html` concatenated so
   every extraction-based test keeps addressing real code wherever it
   lives; tests that grep "the panel source" for page content read the
   concat too. Slice 3 replaces both with imports of the real files.
3. **JS → ES modules under `webapp/js/`** — LANDED, one module per
   commit, loaded with `<script type="module">`, extracted BOTTOM-UP so
   document order — and therefore execution order — never changed: the
   one-line inline script runs during parse, then the module tags run
   in document order, exactly the order those regions ran when they
   were one block. `main.js` is deliberately the LAST tag: in the
   single-block days every function was hoisted, so the first poll()
   could never race a definition; with modules, only "kickoffs run
   after every module has evaluated" preserves that guarantee. **Any
   run-once-at-startup call goes in main.js**, never at a feature
   module's top level.

   **The module pattern** (see the bottom of any `webapp/js/*.js`):
   - Top-level declarations are module-private. What the page's inline
     handlers, the other files, or the repo's browser harnesses
     (`scripts/measure_editor_layout.py`) reference is published
     explicitly — functions via one `Object.assign(globalThis, {...})`
     block at the bottom, variables by declaring them as
     `globalThis.X = ...` instead of `const/let`. Publish only what has
     an outside caller; the lists shrink as tests migrate to imports.
   - ES modules are strict-mode: the block was verified strict-clean
     before the split (`scripts/lint_webapp.mjs`'s pre-flight).
   - `.js` files are served with the `__SEQ__`/`__SEQS__`/`__SEQCAP__`
     noun seams substituted by the `/webapp/` route (same values and
     order as `page()`), because module files never pass through
     `page()`.
   - `scripts/lint_webapp.mjs` (run by release_gates when eslint is
     installed, `npm install` first) enforces no-undef/no-redeclare
     with the real scope model plus a cross-file duplicate-publish
     check; `test_no_duplicate_defs.py` forbids the same top-level
     name in two frontend files outright.
4. **Routes → the panel/ package** — LANDED. All 101 routes moved out
   of the `do_GET`/`do_POST` if/elif chains into `panel/routes_*.py`
   modules (stats, meta, models, loras, train, files, queue,
   characters, storyboard, image), registered into `panel.routes`'
   tables: exact paths in `GET_ROUTES`/`POST_ROUTES`, the
   startswith/endswith families in the ordered `GET_PATTERNS`/
   `POST_PATTERNS` lists (order is load-bearing —
   `/x/sheet/generate` ends with both `/sheet/generate` and
   `/generate`). The dispatchers are ~35 lines each and hold no
   handler logic. Registration refuses duplicates at import;
   `test_routes.py` refuses a chain arm outright and pins the
   pattern order.

   **The P convention**: route modules never `import mlx_ltx_panel`
   (the panel usually runs as `__main__`; importing it by name would
   execute the whole module a second time — port bind, threads,
   everything). Each module declares `P = None` and the panel assigns
   the RUNNING module object into it at wiring time; handlers reach
   panel globals as `P.<name>` at request time. Form-reading POST
   handlers open with `h._read_form_body()` (the shared guarded read);
   multipart handlers read their own stream and must NOT call it.

Every slice ships as its own small commit with
`bash scripts/release_gates.sh --fast` green before the next starts.

## How the server half and the client half talk

### Page render (`page()` + template substitution)

`GET /` serves `page()`: the embedded `HTML` template with a set of
`__PLACEHOLDER__` substitutions applied at request time. The
placeholders are the ONLY dynamic seams in the page — everything else
is static text. Current placeholders: `__BOOTSTRAP__` (see below),
`__PROFILE_BADGE__`, `__SEQ__`/`__SEQS__`/`__SEQCAP__`,
`__Q8_CHARACTER_INSTALL_COPY__`, `__BUILD_STAMP__`,
`__PANEL_VERSION__`, `__ENGINE_RULES__`, `__CAP_TIER__`.

### The BOOT contract (`__BOOTSTRAP__`)

The server builds one JSON object in `page()` and substitutes it into
`const BOOT = __BOOTSTRAP__;` in the page's script. This is the single
source of truth for everything the UI must know at first paint:
presets, aspects, tier tables (`tier`, `quality_times`, `cap_tier`),
the engine registry (`engines`, `default_engine`), the H3 and LTX
tier/estimate tables (`h3`, `ltx`), storyboard state, train presets,
and profile/model identity.

**Rules that must survive the restructuring:**

- Tiers, estimates and the engine registry are built SERVER-SIDE and
  arrive via BOOT. The browser never computes a canvas size, a frame
  count, or a wall-clock estimate of its own — it looks them up.
- The BOOT shape is append-only in practice: fields are added, not
  renamed, because sidecars and Load Params round-trip through it.
- The engine registry additionally emits per-engine CSS
  (`_engine_css()`): accent variables and `data-<id>-only` fold rules.
  That CSS is substituted into `webapp/style/panel.css` at serve time
  by the `/webapp/` route — the stylesheet ON DISK carries the
  `__ENGINE_RULES__` placeholder, so the registry stays single-source.

### After first paint

The page polls `GET /status` (queue, history, helper health, download
state, memory) and drives everything else through the HTTP API
documented in `docs/API.md`. Forms POST to `/run`, `/queue/add`,
`/settings`, etc. There is no websocket; state flows through polling.

### Static file routes

| route | directory | notes |
|---|---|---|
| `/webapp/*` | `webapp/` | the frontend's own files; `.css` gets `__ENGINE_RULES__` substituted at serve time; `no-cache` |
| `/assets/*` | `assets/` | images only (logos, avatars); cached 1 day |
| `/stats` | `panel_assets/stats.html` | the repo-stats dashboard |

All routes are loopback-only (`_is_local_request`) and path-bound to
their directory — a `..` in the URL cannot escape.

## Adding a feature: where things go

- **New styles** → `webapp/style/panel.css`. Never a `<style>` block or
  inline `style=` fragments in the Python string
  (`test_panel_assets.py` fails the build if a `<style>` block returns).
- **New markup** → `webapp/index.html`. **New JS** → the matching
  module in `webapp/js/` (see the tree above for what each owns),
  NEVER the Python file and NEVER the inline block — the inline block
  is one line and stays that way. This rule is the entire point of the
  restructuring: the panel file is where two sessions built the same
  feature twice without noticing.
  - A function another file (or an inline handler in the markup) must
    call goes in the module's `Object.assign(globalThis, {...})`
    publish block; everything else stays module-private.
  - Shared mutable state a second file writes is declared once, as a
    column-0 `globalThis.X = ...`, in the module that owns it.
  - Anything that must run once at startup goes in `main.js` (the last
    module tag), never at a feature module's top level.
  - Run `node scripts/lint_webapp.mjs` (or `npm run lint`) — it
    catches a missed publish, a name declared twice, and two files
    claiming the same global.
- **New server state the UI needs at first paint** → a field in the
  BOOT object in `page()`, built server-side.
- **New server state the UI needs live** → a field in `/status`.
- **New endpoints** → a handler in the matching `panel/routes_*.py`,
  registered with `@get`/`@post` (exact path) or `@get_when`/
  `@post_when` (pattern). NEVER an if-arm in `do_GET`/`do_POST` —
  `test_routes.py` fails the build on one. Every route string appears
  exactly once; registration itself refuses duplicates.
- **New render behaviour** → the helper (`mlx_warm_helper.py`) and the
  job spec (`make_job()`), not the frontend.

## The pre-change checklist

CLAUDE.md carries THE STRUCTURE LAW (the numbered one-home-per-change
rules); this is the mechanical routine that goes with it. For any change
to panel code, frontend or server:

```bash
# while working — cheap, instant, catches the module-scope mistakes
node scripts/lint_webapp.mjs

# the suites nearest your change (they auto-run in the gates too)
./ltx-2-mlx/env/bin/python3.11 -m unittest test_panel_assets test_routes test_no_duplicate_defs

# before EVERY commit — must exit 0; key the commit on ITS exit code
bash scripts/release_gates.sh --fast

# before promoting anything — the full battery, no skips accepted
bash scripts/release_gates.sh
```

And the reload rules while iterating: CSS or JS → hard-refresh the
browser, no restart; `index.html` markup or any Python → restart the
panel, then hard-refresh.

## Gates that guard this structure

- `test_panel_assets.py` — webapp/ files exist, are served correctly,
  the dynamic CSS seam substitutes, no `<style>` returns to the page,
  the inline `<script>` stays exactly one line (the BOOT seam), and no
  markup creeps into the server Python.
- `test_no_duplicate_defs.py` — no duplicate JS function declarations
  or HTML ids in the page (the built-twice incident's tombstone).
- `bash scripts/release_gates.sh --fast` — the whole battery; must be
  green after every slice.
