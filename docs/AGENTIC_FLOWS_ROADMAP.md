# Phosphene Agentic Flows — Roadmap to Pro-App Caliber

_Generated 2026-05-07 by deep UX audit. 50+ items across 4 phases._

## Executive summary

The Agentic Flows tab today is **technically dense and functionally rich** — a careful operator manual, four engine kinds, a memory-pressure guard, a Stage pane that shows live render progress, anchor-pick collaboration, project memory, multi-take support, reasoning surfacing. The single biggest gap is **legibility of capability**: a first-time user opens the tab, sees three vague conversation starters and an engine pill, and has no idea that the agent can read a PDF script, generate Flux anchors, write a manifest, lock a master style across sessions, or refine an existing clip. Sixteen tools, five engines, three image backends, two operating modes, four quality tiers, four shot modes — none of it is discoverable. The product reads as a chat box, not as a director's workstation.

The single biggest opportunity is to **convert hidden depth into a structured workspace**. Replace the pure-chat metaphor with a three-panel surface where chat is the conversation channel, but the project's *state* — script, characters, master style, shot list, picks, renders, manifest — lives in dedicated, navigable, editable structures around it. Sessions become projects. Every artifact the agent produces is first-class and editable by the user. The agent stops being a chatbot that emits files and becomes a director's seat-of-control. This is the move from "Claude can render videos" to "Phosphene's directing agent."

The recommended order of operations: **Phase 0** lands table-stakes professionalism (real first-message streaming, recovery from common error modes, in-app docs, accessibility floor). **Phase 1** rebuilds discoverability and information architecture (capabilities palette, command palette, structured project memory, smart defaults). **Phase 2** adds the wow features that turn casual users into evangelists (project templates, voice memos, comparison view, auto-recovery, NLE export, smart routing). **Phase 3** is the long tail — collaboration, scheduled queues, macros. Phase 0 is roughly 2 weeks of work; Phase 1 is 4–6 weeks; Phase 2 is 4–6 weeks; Phase 3 ships opportunistically.

## Phase 0 — table stakes (~2 weeks)

| # | Item | Effort | Why |
|---|---|---|---|
| 0.1 | SSE chat streaming | L | Demo-killer wait; content visible while it generates is the largest perceived-quality lift |
| 0.2 | Stop button that actually stops (server-side cancel) | M | Today's × only aborts the client; server keeps spending tokens |
| 0.3 | Recoverable error cards with one-click remediations | M | Errors as paths forward (Retry / Switch model / Stop engine) |
| 0.4 | In-app capability tour on first run | M | Three-phase loop, anchor-pick UX, overnight-batch model |
| 0.5 | `/help` slash command + capabilities sheet | S | Surfaces every tool's docstring as a structured catalog |
| 0.6 | Keyboard accessibility floor (focus-visible, aria-live, reduced-motion) | M | WCAG AA fail today |
| 0.7 | Session restore on panel restart | M | Half-rendered batches feel orphaned; resume / discard banner |
| 0.8 | Default-engine recommendation banner | S | New users run bundled Gemma forever; surface Qwen / Anthropic when appropriate |
| 0.9 | "What just happened?" turn summary | S | 30-card-deep turns lose context; one-line summary at the bottom |
| 0.10 | Cost / wall-time predictor before submit | S | Total wall + $ before saying "Confirm batch" |
| 0.11 | Engine readiness check at session start | S | Surface 401 / unreachable BEFORE user uploads attachments |
| 0.12 | Atomic project-notes history + undo | S | 7-version ring buffer; restore previous version |

## Phase 1 — high-impact polish (4–6 weeks)

| # | Item | Effort |
|---|---|---|
| 1.1 | Command palette (Cmd+P) — sessions / shots / tools / settings | M |
| 1.2 | Capabilities panel + tool catalog | M |
| 1.3 | Structured project memory (replace single markdown file) | L |
| 1.4 | Live anchor preview + thumbnail-while-rendering | M |
| 1.5 | Smart defaults from history | M |
| 1.6 | Information architecture — collapse the agent header | S |
| 1.7 | Inline error cards on tool failures | S |
| 1.8 | Comparison view — two takes side by side | M |
| 1.9 | Adaptive engine routing (small for tools, big for prose) | L |
| 1.10 | Voice input → script paste (Whisper) | M |
| 1.11 | Project templates (documentary / music video / talking head…) | M |
| 1.12 | Slash commands in composer (`/queue` `/style` `/picks` `/cost` `/finish`) | S |
| 1.13 | Sketch mode / dry-run on submit_shot | S |
| 1.14 | Refine UX — show context inline, not just a chip | S |
| 1.15 | RAM chip → live system pulse with breakdown popover | S |
| 1.16 | Discoverability for engine kinds (radio cards w/ pricing) | S |
| 1.17 | Session export to markdown | S |
| 1.18 | Project export bundle (zip with manifest + mp4s + picks + transcript) | M |
| 1.19 | Reasoning indicator with throughput | S |
| 1.20 | Modal focus traps + escape keys | S |
| 1.21 | Empty states everywhere have actions | S |

## Phase 2 — delight (4–6 weeks)

| # | Item | Effort |
|---|---|---|
| 2.1 | Live thumbnail preview during LTX render | L |
| 2.2 | Shot timeline view (drag to reorder) | L |
| 2.3 | NLE export — Final Cut XML / Premiere XML | M |
| 2.4 | Auto-recovery from engine errors mid-batch | M |
| 2.5 | Smart-route to Anthropic on Qwen reasoning loop detection | M |
| 2.6 | Inline annotation on a clip | L |
| 2.7 | Pre-defined macros (saved prompt sequences) | M |
| 2.8 | Background mode + scheduled queues | M |
| 2.9 | LoRA picker the agent can recommend from | S |
| 2.10 | Acid-test "this prompt won't work" guardrail w/ Director's note | M |
| 2.11 | Cross-session search | M |
| 2.12 | Showcase / public gallery (static HTML) | L |
| 2.13 | Director's chair keyboard shortcuts (J/K/L/P/R) | S |
| 2.14 | Anchor prompt expansion ("more like this but darker") | S |
| 2.15 | Dry-run cost estimate per-shot | S |

## Phase 3 — long tail (opportunistic)

| # | Item |
|---|---|
| 3.1 | Multi-machine collaboration (iCloud-synced sessions) |
| 3.2 | Session fork ("try a different direction") |
| 3.3 | Live URL share for trusted machines |
| 3.4 | Image-engine: DiffusionKit / CoreML on ANE |
| 3.5 | Sketched anchor input (canvas painter → ControlNet Flux) |
| 3.6 | Multi-agent — director + cinematographer split |
| 3.7 | Real-time render preview in chat |
| 3.8 | Auto-stitch optional draft cut |
| 3.9 | Engine A/B comparison mode |
| 3.10 | Semantic diff between takes |
| 3.11 | Whisper inline as speech-to-shot (silence-segmented chunking) |
| 3.12 | OBS / streaming integration |
| 3.13 | Pull-request-style review of the agent's plan |

## Critical implementation seams

Every roadmap item names a specific file + function or HTML element. Most land in:

- `mlx_ltx_panel.py` (~16,900 lines) — single file with HTTP server + inlined HTML/CSS/JS. Key seams: `#agentChat`, `agentSend()`, `renderEmpty()`, `renderToolResultCard()`, `renderAnchorGrid()`, agent header, Stage pane (`agentStageRender()`), settings drawer, sessions sidebar (`aspInit()`), composer.
- `agent/runtime.py` — `Session` + `run_turn`. SSE streaming, cancel support, auto-recovery, multi-engine routing.
- `agent/tools.py` — every tool with docstrings consumed by the system prompt. New tools, `dry_run` flag, `prompt_diff` arg.
- `agent/prompts.py` — system prompt builder. Project templates inject addenda; structured project memory replaces `{project_notes_block}`.
- `agent/engine.py` — OpenAI/Anthropic client. SSE deltas, cancel-via-connection-close, reasoning-loop detection emits structured `recommend_engine` error.
- `agent/project.py` — markdown notes. Add `read_structured()` parsing `[<kind> · ...]` sections; add ring buffer for undo.
- `agent/image_engine.py` — backends. Refactor to a generator so candidates stream as they finish.
- `docs/AGENTIC_FLOWS.md` + `docs/STATE.md` — source of truth. Every shipped item gets a row.

## The agent-tool connection rule (from `docs/AGENTIC_FLOWS.md`)

Every new manual UI feature must also be exposed to the agent as a tool. New roadmap features that introduce manual surfaces (timeline view, comparison view, macros, scheduled queues) need matching agent tools.
