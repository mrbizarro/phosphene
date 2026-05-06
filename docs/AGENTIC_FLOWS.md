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

## How a session goes

1. **You paste** a brief, an idea, or a full script. The composer is
   one big textarea — Cmd/Ctrl+Enter sends.
2. **The agent plans in plain text first.** A numbered shot list with
   one-line visual descriptions, durations, and a total wall-time
   estimate. Nothing is queued yet.
3. **You approve.** "go", "queue it", "yes" — anything that reads as
   confirmation. Tweak the plan in the chat first if you want.
4. **The agent submits each shot** by calling `submit_shot` on the
   panel's existing `/queue/add` endpoint. Each submission produces a
   tool-result chip in the chat with the job id and ETA.
5. **The agent writes a `manifest.json`** listing the queued shots in
   cut order. Lands in `mlx_outputs/agentflow_<session_id>/`.
6. **The agent calls `finish`** — the loop ends and you're returned
   control. The manual queue keeps rendering while you sleep.

The agent intentionally does **not** auto-stitch the clips. It writes
the manifest; the cut belongs to you.

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
