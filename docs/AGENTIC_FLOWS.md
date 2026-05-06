# Agentic Flows — chat-driven shot planner

Status: **First ship 2026-05-06**, alongside multi-keyframe Layer 2.

The Agentic Flows tab in Phosphene's panel is a chat where you paste a
script (or an idea) and the agent breaks it into shots, estimates the
overnight wall time, and queues every shot through the same FIFO queue
the manual UI uses. You go to sleep; the queue runs; in the morning
you wake to a folder of mp4s and a `manifest.json` listing the cut
order. You stitch the film in your editor.

## Where to find it

Panel header → form-pane → top tab strip: **Manual | Agentic Flows**.
Manual is the existing form. Agentic Flows is the new tab. Selection
persists in localStorage.

## How a session goes (3-phase director loop)

The default workflow is collaboration: agent generates candidate
stills, you pick the best one per shot, agent renders videos with
your picks as i2v anchors. Locking the look BEFORE the 4-minute
video render dramatically reduces "guess and pray" t2v failures.

1. **Phase A — You paste** a brief, an idea, or a full script. The
   composer is one big textarea — Cmd/Ctrl+Enter sends.
2. **Agent plans in plain text** with a master style suffix, director's
   adjustments, numbered shot list. Nothing queued yet.
3. **Phase B — Agent generates anchor candidates.** For each shot it
   calls `generate_shot_images` (4 candidates default). Each call
   appears as a tool-result card with a thumbnail grid below. Click
   a thumbnail to pick — green border + checkmark. Selection persists
   in `session.tool_state.selected_anchors`.
4. **Phase C — You type "render".** Agent reads `get_selected_anchors`,
   submits each shot via `submit_shot` with `mode: "i2v"` and
   `ref_image_path: <picked anchor>`, writes `manifest.json`, calls
   `finish`. Renders run overnight on your locked anchors.

You can opt out of the anchor flow ("just t2v, skip the picker") for
abstract cutaways or fast tests, in which case the agent skips
phase B and submits straight to t2v.

The agent intentionally does **not** auto-stitch the clips. It writes
the manifest; the cut belongs to you.

## Fullscreen / focus mode

Click the expand icon in the agent header (between the "+" new-chat
button and the gear) to spread the chat across the full window —
the panel header, bottom pane (Now/Queue/Recent/Logs), and workflow
tab strip all fold away. **Esc** exits. State persists in
localStorage so reopening the tab keeps you in fullscreen.

Useful when planning a long multi-shot piece: more vertical real
estate for the conversation, less chrome competing for attention.

## Engine options

The agent talks to any OpenAI-compatible chat endpoint. Three setups:

### 1. Phosphene Local (default)

Spawns `mlx_lm.server` against the bundled `gemma-3-12b-it-4bit`
weights. These ship with Phosphene as the LTX text encoder — the same
model file is also a capable instruction-tuned chat model. **Zero
extra download** to start using the agent on a fresh install.

Click *Settings → Start* to spawn the server (~10 s cold start, ~30 s
to first token after weights load).

### 2. Local with a stronger model

Drop a chat-capable MLX 4-bit model directory into `mlx_models/` —
`config.json` + a `tokenizer.json` + at least one `.safetensors`.
Examples that work well:

- `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit` — recommended
  upgrade for serious agent work. ~17 GB, ~3.3 B active params via
  MoE, fast inference, strong tool-use.
- `mlx-community/Devstral-Small-2505-4bit` — Mistral's purpose-built
  agent model. ~14 GB.
- `mlx-community/Qwen2.5-7B-Instruct-4bit` — small fast fallback.

The Settings drawer's *Local model* dropdown auto-discovers any dir
under `mlx_models/` that looks chat-capable.

### 3. Custom (cloud or LAN)

Pick *Custom* in the engine kind picker, then fill:
- *Base URL*: e.g. `https://api.openai.com/v1`,
  `https://api.anthropic.com/v1` (their OpenAI-compat shim), or your
  LM Studio box at `http://192.168.x.x:1234/v1`.
- *Model*: the model id the endpoint expects (`claude-sonnet-4-6`,
  `gpt-5`, etc.).
- *API key*: stored in `state/agent_config.json` and only sent as the
  `Authorization` header. `GET /agent/config` masks it (`has_api_key`
  bool only).

## What the agent knows

The system prompt the agent sees on every turn encodes Phosphene's
operator manual: modes, empirical wall times from `CLAUDE.md` §0,
LTX 2.3 failure modes from `STATE.md` §6, prompting rules, and tool
docstrings. Highlights it acts on:

- LTX is good at: talking heads, medium shots, interviews, slow
  push-ins, mockumentary / news / clinical settings, 2–3 dialogue
  turns per clip, joint audio + video.
- LTX is bad at: hands and held objects, high-motion physics, faces
  below ~80 px in-frame, multi-shot character drift across cuts.
- Hardware tier clamps (Comfortable: 1280 px t2v, 768 px FFLF/Extend).
- Quality / accel / mode → wall-time table for honest ETAs.
- Cross-shot continuity workflow: render shot 1 → `extract_frame`
  the last frame → use as `ref_image_path` for shot 2's i2v anchor.

Update `docs/STATE.md` or `CLAUDE.md` and the system prompt picks the
new knowledge up on the next turn — single source of truth.

## Tool reference

Tools the agent can call. Docstrings are pulled into the system prompt
verbatim (see `agent/runtime.render_tools_doc()`); kept here for human
reference too.

| Tool | What it does |
|---|---|
| `estimate_shot` | Wall-time estimate for a planned shot from the empirical M4 Max table. T^1.5 length scaling. No side effects. |
| `submit_shot` | Append a render job to the panel's FIFO queue. Wraps `make_job` + `STATE['queue'].append()`. Returns job id, queue position, ETA. |
| `get_queue_status` | Snapshot: running, queue, recent history, total wall ETA. |
| `wait_for_shot` | Block until a job hits done/error/cancelled. Used when the agent needs a finished output to chain into the next shot. |
| `extract_frame` | Pull first/last/middle frame of a finished clip as PNG. Used for cross-shot anchoring (last frame of N → keyframe-0 of N+1). |
| `upload_image` | Resolve a chat-attached image's absolute path before passing it as `ref_image_path` or in `keyframes`. |
| `write_session_manifest` | JSON file listing all submitted shots in cut order. Lands in `mlx_outputs/agentflow_<session_id>/`. Doesn't auto-stitch. |
| `finish` | Explicit loop terminator. The runtime stops calling the model after this. |

The fenced-block protocol: the model emits a single
` ```action {...} ``` ` block per turn. The runtime parses, dispatches,
and feeds the result back as a `<tool_result>...</tool_result>` user
message. Universal across every Chat Completions server we've tested.

## Wire flow (debugging)

```
Browser (Agentic Flows tab)
    │
    │  POST /agent/sessions/<id>/message
    │  { "message": "<user prose>" }
    ▼
mlx_ltx_panel.py (HTTP handler)
    │
    │  agent_runtime.run_turn(session, msg, panel_ops, …)
    ▼
agent/runtime.py
    ├── builds system prompt (prompts.py + capabilities snapshot)
    ├── engine.chat(messages, EngineConfig)  ← OpenAI-compat HTTP
    │       │
    │       ▼
    │   mlx_lm.server (local, port 8200)  OR  cloud endpoint
    │
    ├── tools.parse_action_block(reply)
    ├── tools.dispatch(name, args, panel_ops, session_state)
    │       └── panel_ops.submit_job(form)
    │              └── make_job(form) + STATE['queue'].append(...)
    │
    └── loop until no action block OR finish OR max_steps
```

## Image generation backends (Phase B)

The agent uses a pluggable image engine to generate anchor stills.
Configure under *Settings → Image generation*. Three backends ship:

### `mock` (default — zero setup)

Flat-colored PNGs drawn with PIL. Each candidate carries a
distinguishable hue + the prompt excerpt overlaid in white. Free,
instant, useful for testing the pick-flow without spending a dime.
Not for actual production renders — just verifies the UX.

### `mflux` — Phosphene Local Flux (recommended for local)

`mflux` is a MLX-native port of Flux that runs on Apple Silicon. It
ships with several Flux variants; **Flux Krea Dev** (`krea-dev`) is
the recommended default — Krea AI's fine-tune of Flux.1 Dev with
better photorealism, stronger documentary / interview / clinical
aesthetics. Particularly well-suited to the kinds of looks the agent
gravitates toward.

Install once into the panel's bundled venv:

```bash
ltx-2-mlx/env/bin/pip install mflux
```

That brings the `mflux-generate` CLI into `ltx-2-mlx/env/bin/` where
the panel auto-discovers it. First generate downloads the Krea Dev
4-bit weights to `~/.cache/huggingface` (~6 GB), one-time.

Performance on M4 Max 64 GB: ~25–60 s per image (model loads each
call — first candidate per shot is the slow one, subsequent ones
amortize once HF caches are warm). 4 candidates per shot ≈ 2–4 min.
12 shots ≈ 25–50 min for a complete plan's anchors.

Settings exposed:
- **Model**: `krea-dev` (recommended) / `dev` / `schnell` / Custom HF id
- **Custom path**: e.g. `filipstrand/FLUX.1-Krea-dev-mflux-4bit`
- **Base model**: required when Custom path is used (`krea-dev` /
  `dev` / `schnell`)
- **Steps**: 25 for dev/krea, 4 for schnell
- **Quantize**: 4 (~6 GB) or 8 (~12 GB, sharper)

### `bfl` — Black Forest Labs API (recommended for cloud)

The canonical Flux source. `flux-dev` ≈ $0.025/img, `flux-pro` ≈
$0.05, `flux-pro-1.1` ≈ $0.04, `flux-schnell` ≈ $0.003. Roughly
10–15 s per image. Get a key at `https://api.bfl.ml`; paste into
*Settings → Image generation → BFL API key*. Stored in
`state/agent_image_config.json`, only sent as the `X-Key` header.

### Future backends

The interface is open — `_generate_<kind>()` in
`agent/image_engine.py` is the only place to add. Likely candidates:

- **DiffusionKit / CoreML** for ANE acceleration (~10 s / image, fully
  local). Heavier install (CoreML model conversion).
- **Replicate** API (mirrors BFL flow + supports Flux Krea hosted)
- **fal.ai** API (very fast, generous trial credits)

## Known limitations + workarounds

- **Bundled Gemma 3 12B IT is good at planning a shot list** but a
  larger model (Qwen3 Coder 30B-A3B) handles long scripts and tool
  chaining noticeably better. If your script has more than ~10 shots
  and needs to chain anchored frames, install a stronger model in
  `mlx_models/` and pick it via Settings.

- **mlx-lm 0.31.1 has a KV-cache merge bug** (`BatchRotatingKVCache.
  merge` raises on Gemma-style sliding-window caches). Phosphene
  spawns `mlx_lm.server` with `--prompt-cache-size 0
  --prefill-step-size 8192 --prompt-concurrency 1
  --decode-concurrency 1` to sidestep both code paths. Documented in
  `agent/local_server.py:start()`.

- **Cross-shot character drift** is LTX's structural limit; the agent
  can mitigate via i2v / multi-keyframe anchoring but cannot eliminate.
  If your script demands exact continuity (the same actor across many
  shots), expect some drift and plan for tighter coverage.

- **No streaming for now.** The chat sends one turn, blocks until the
  whole tool loop completes, then re-renders. A multi-shot plan can
  take 2–5 minutes of model time on the bundled Gemma; the *Thinking…*
  indicator is the only feedback during that window. Streaming via
  Server-Sent Events is a v2 enhancement.

## File layout

```
agent/
├── __init__.py
├── engine.py          ← OpenAI-compat client + role-alternation normalizer
├── local_server.py    ← mlx_lm.server lifecycle + model discovery
├── prompts.py         ← system prompt builder (sources from STATE.md / CLAUDE.md)
├── tools.py           ← tool definitions + dispatcher + PanelOps protocol
└── runtime.py         ← Session, run_turn loop, persistence

state/
├── agent_config.json  ← persisted EngineConfig (api_key never returned via HTTP)
└── agent_sessions/
    └── <session_id>.json  ← one chat thread, atomic-replaced after every turn

mlx_outputs/
└── agentflow_<session_id>/
    └── manifest.json  ← cut-order index for an overnight render
```

## Hard constraints honored

- **Apple Silicon only** — no PyTorch, no CUDA. mlx-lm runs the agent's
  LLM the same way the helper runs LTX.
- **No new microservices** — the agent runs inside the panel process.
  mlx_lm.server is a subprocess, exactly like `mlx_warm_helper.py`.
- **Joint audio + video stays** — the agent submits standard t2v / i2v
  / keyframe / extend jobs through the same path as the manual UI; no
  bypass of the audio-bearing pipeline.
- **Branch policy** — every Agentic Flows commit lands on `dev` first.
  Promotion to `main` waits for explicit OK.

## ✦ The agent-tool connection rule (binding)

**Every new feature added to the manual UI must also be exposed to
the agent as a tool.** No exceptions.

The agent's value is "you can drive the panel by chatting." When a
new manual control ships without a matching agent tool, the user has
to drop out of chat to use it — defeats the point.

How to honor this rule when adding a feature:

1. **Implement the manual UI** in `mlx_ltx_panel.py` (HTML/JS) or
   `agent/*.py` first. Get it working through the existing endpoints.
2. **Decide the tool surface.** The tool name should match how the
   user would describe the action ("install_lora", "extract_frame",
   "set_quality"). Args mirror the most useful manual fields, defaults
   are the panel's defaults.
3. **Add the tool to `agent/tools.py`** with the `@tool("name")`
   decorator. Wire it through `PanelOps` callbacks (no panel imports
   inside the tools module — see existing pattern at the top of
   `agent/tools.py`).
4. **Document it in the system prompt** (`agent/prompts.py`,
   `# Tools you can call` section + `# Director's craft` if it
   changes the recommended workflow). The agent doesn't read this
   doc — only its prompt.
5. **Render the tool's results sanely.** If the result has visual
   content (PNGs, clips), extend `renderToolResultCard()` in
   `mlx_ltx_panel.py` so it surfaces the content inline.
6. **Add a row to the Tool reference table above** in this file.
7. **Update `docs/STATE.md`** with the new capability so future
   Phosphene sessions discover it.

Reference example: when LoRA management was added to the manual UI,
the agent gained `install_lora`, `list_loras`, `select_lora` tools.
When the keyframe-interpolation mode was added, `submit_shot` gained
the `keyframes_json` arg. When project-notes shipped, the agent got
`read_project_notes` + `append_project_notes`. The pattern is
consistent — preserve it.

Anti-pattern (don't do this): "I'll add this to the UI now and wire
the agent later." Later never happens; the manual feature drifts
ahead of the agent and the user develops a habit of using it manually
even when chat would be faster.
