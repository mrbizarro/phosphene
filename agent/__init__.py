"""Agentic Flows runtime for Phosphene.

A thin chat-driven planner that turns one user idea (or a full script) into
a queue of LTX 2.3 renders. Engine-pluggable: ships with a local mlx-lm
server pointed at the bundled Gemma encoder, but any OpenAI-compatible
endpoint (mlx-lm, LM Studio, Ollama, OpenAI, Anthropic compat, ...) works.

The agent is a guest of the existing panel — same process, same queue,
same helper. No new microservice. The runtime composes existing primitives
(`/queue/add`, multi-keyframe interpolation) via tool calls the model
emits as fenced ```action JSON blocks in its replies.

Public surface:
    engine.EngineConfig, engine.chat
    runtime.AgentSession, runtime.step
    tools.TOOLS, tools.dispatch
    prompts.build_system_prompt
"""
