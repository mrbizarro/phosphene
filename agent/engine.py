"""OpenAI-compatible chat client.

One function: send a list of messages, get back the assistant's text.
No streaming, no tool-calling spec — the agent emits actions as fenced
```action JSON blocks in plain content (see prompts.py). That convention
works on every Chat Completions server we'd want to support, regardless
of whether the server / model implements OpenAI's tool-calling shape.

LiteLLM-backed multi-provider router when available (free retries +
normalized error messages + provider abstraction for OpenAI / Anthropic
/ Ollama / custom OpenAI-compat). Transparently falls back to a stdlib
urllib path on venvs that haven't run `update.js` yet.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field

# LiteLLM is added by install.js / update.js. If a user is on an older
# venv we keep working via the stdlib path below.
try:
    import litellm
    from litellm import completion as _litellm_completion
    # Quiet by default — Phosphene's panel log already shows turn events.
    # Telemetry off: LiteLLM otherwise pings their backend on first import.
    try:
        litellm.telemetry = False
        litellm.set_verbose = False
        litellm.suppress_debug_info = True
        # Don't log raw payloads (contains user prompts + tool results).
        litellm.drop_params = True
    except Exception:                           # noqa: BLE001 — best effort
        pass
    _HAS_LITELLM = True
except Exception:                               # noqa: BLE001
    _HAS_LITELLM = False


# 15 minutes. Local reasoning-class models (Qwen 3.6, DeepSeek R1) on
# long contexts can easily take 5-10 minutes per turn — they emit
# 1500-3000 tokens of chain-of-thought BEFORE the answer. Combined with
# a multi-shot session that has 50+ messages of history, inference is
# slow. The earlier 5-min cap was hitting in the middle of legitimate
# turns; user saw "engine error: timed out" mid-batch.
DEFAULT_TIMEOUT_S = 900


@dataclass
class EngineConfig:
    """How the agent talks to its LLM.

    `kind` is informational — both branches end up calling the same
    OpenAI-compatible /chat/completions endpoint. The local branch just
    promises a panel-spawned mlx-lm.server on `base_url`.
    """

    kind: str = "phosphene_local"          # "phosphene_local" | "ollama" | "custom" | "anthropic"
    base_url: str = "http://127.0.0.1:8200/v1"
    model: str = "mlx-community/gemma-3-12b-it-4bit"
    api_key: str = ""
    temperature: float = 0.4
    # 8192 was 3072 — reasoning-class models (Qwen 3.6, DeepSeek R1) put
    # their full chain-of-thought into the response, separately from the
    # final answer, and 3k tokens isn't enough room for both. Symptom on
    # the older budget: content="", finish_reason="length", and the agent
    # appears "stuck" because the runtime sees an empty assistant message.
    max_tokens: int = 8192
    # mlx-lm.server boot args — only used when kind == "phosphene_local".
    # `local_model_path` may be a Hugging Face id ("Qwen/Qwen3-30B-A3B-Instruct-4bit")
    # OR an absolute path to a local model dir (the bundled Gemma path).
    local_model_path: str = ""
    # Anthropic-only: API version pinned in the `anthropic-version` header.
    # Anthropic adds breaking changes per dated version; pin to a known-good.
    anthropic_version: str = "2023-06-01"
    # Operating mode for the agent.
    #   "plan_sleep"  — agent plans, queues all shots, calls finish; the panel
    #                   auto-stops the local engine the moment finish lands so
    #                   the chat model's RAM (often 20+ GB for Qwen 35B) is
    #                   handed back to the LTX renderer for the overnight run.
    #                   The right default on a 64 GB Mac.
    #   "interactive" — engine stays resident across finishes so follow-up
    #                   chat is instant. Use only when the Mac has headroom
    #                   (no concurrent renders, or a small chat model).
    mode: str = "plan_sleep"
    # When True (default), the panel auto-stops the local engine the moment
    # the LTX worker picks up a render job, and auto-restarts it once the
    # queue drains. Solves the 64 GB Mac swap-thrash problem where the
    # chat model (~7-22 GB resident) and the LTX renderer (~22-30 GB)
    # compete for unified memory and the system pages out denoise tensors.
    # Only applies when kind == "phosphene_local"; cloud engines (anthropic,
    # ollama-on-localhost, custom) don't compete for RAM and are never
    # paused. Caveat: an agent in `interactive` mode that's mid-turn when
    # a render starts will see its in-flight engine.chat() fail; the
    # runtime surfaces this and the user retries after the queue drains.
    # In `plan_sleep` mode the agent always finishes before submitting,
    # so no conflict.
    auto_pause_during_renders: bool = True

    def to_public_dict(self) -> dict:
        """Strip the api_key for safe display in /agent/config GETs."""
        d = asdict(self)
        if d.get("api_key"):
            d["api_key"] = ""
            d["has_api_key"] = True
        else:
            d["has_api_key"] = False
        return d


@dataclass
class ChatResult:
    content: str
    finish_reason: str = "stop"
    usage: dict = field(default_factory=dict)
    model: str = ""
    # Some chat servers (mlx-lm with reasoning models like Qwen 3.6,
    # DeepSeek R1) return the model's chain-of-thought in a separate
    # `reasoning` field on the response message. We surface it so the
    # UI can show "what the model was thinking" — invisible reasoning
    # is the #1 cause of "agent appears stuck" reports. Empty for
    # non-reasoning models / non-supporting servers.
    reasoning: str = ""


def _normalize_for_wire(messages: list[dict]) -> list[dict]:
    """Collapse consecutive same-role messages into one for transport.

    Gemma (and several other open chat models) ship with chat templates
    that require strict user/assistant alternation. Our runtime stores
    the raw conversation — tool results land as a user-role wrapper
    after an assistant tool-call, and a new user turn after a tool-
    result-tool-result-finish chain produces user→user adjacency. We
    keep the raw shape on disk (the UI renders each tool result as its
    own chip), and merge here so the wire form is always valid.

    The system message is preserved as-is; only user/assistant runs are
    coalesced. Joins are by double newline so the model can see the
    boundary.
    """
    if not messages:
        return messages
    out: list[dict] = []
    for m in messages:
        if not out:
            out.append(dict(m))
            continue
        prev = out[-1]
        if prev.get("role") == m.get("role") and m.get("role") in ("user", "assistant"):
            prev["content"] = (prev.get("content") or "") + "\n\n" + (m.get("content") or "")
        else:
            out.append(dict(m))
    return out


def chat(messages: list[dict], config: EngineConfig,
         *, timeout: int = DEFAULT_TIMEOUT_S) -> ChatResult:
    """Send a chat completion request. Returns the assistant text.

    Raises RuntimeError on transport failure or non-200 response. Server-
    side errors bubble up with the upstream body so the user sees what
    the engine actually said (helpful for debugging API key / model name
    typos in the Settings drawer).

    Routes through LiteLLM when available; else stdlib urllib.
    """
    if _HAS_LITELLM:
        return _chat_litellm(messages, config, timeout=timeout)
    if config.kind == "anthropic":
        return _chat_anthropic_urllib(messages, config, timeout=timeout)
    return _chat_urllib_openai(messages, config, timeout=timeout)


# -- LiteLLM path -------------------------------------------------------------
def _chat_litellm(messages: list[dict], config: EngineConfig,
                  *, timeout: int) -> ChatResult:
    """Route through LiteLLM. Maps EngineConfig.kind to a provider prefix.

    LiteLLM provider prefixes (https://docs.litellm.ai/docs/providers):
      - openai/<id>     — OpenAI proper OR any OpenAI-compatible api_base
      - anthropic/<id>  — Anthropic Messages API
      - ollama/<id>     — Ollama (api_base = base_url)
    """
    if config.kind == "anthropic":
        litellm_model = f"anthropic/{config.model or 'claude-sonnet-4-5'}"
        # Use Anthropic's default endpoint unless the user pointed elsewhere
        # (e.g. an enterprise Bedrock proxy speaking the Anthropic shape).
        # base_url defaults to https://api.anthropic.com/v1; LiteLLM has its
        # own default if api_base is omitted, so only pass it through when
        # the user clearly customized it.
        api_base = config.base_url.rstrip("/") if config.base_url and "anthropic.com" not in config.base_url else None
    elif config.kind == "ollama":
        litellm_model = f"ollama/{config.model}"
        api_base = config.base_url.rstrip("/")
    elif config.kind == "phosphene_local":
        # mlx-lm.server identifies models by their LOAD PATH, not by short
        # name. Its /v1/chat/completions 404s with a HuggingFace lookup error
        # if it receives a short name it interprets as a HF repo id and
        # tries to download. So for the local engine we always pass the
        # absolute path as the request "model" field.
        wire_model = config.local_model_path or config.model
        litellm_model = f"openai/{wire_model}"
        api_base = config.base_url.rstrip("/")
    else:  # "custom" — arbitrary OpenAI-compat endpoint
        litellm_model = f"openai/{config.model}"
        api_base = config.base_url.rstrip("/")

    norm = _normalize_for_wire(messages)

    kwargs: dict = {
        "model": litellm_model,
        "messages": norm,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "timeout": timeout,
        "stream": False,
    }
    if api_base:
        kwargs["api_base"] = api_base
    if config.api_key:
        kwargs["api_key"] = config.api_key
    elif config.kind == "phosphene_local":
        # mlx-lm.server doesn't enforce auth but LiteLLM's OpenAI client
        # complains if api_key is unset. Any non-empty string works.
        kwargs["api_key"] = "not-needed"

    if config.kind == "anthropic" and config.anthropic_version:
        kwargs["extra_headers"] = {"anthropic-version": config.anthropic_version}

    try:
        resp = _litellm_completion(**kwargs)
    except Exception as e:                      # noqa: BLE001 — LiteLLM raises rich types
        # Surface as RuntimeError so runtime.py's existing handler shows
        # it inline in the chat. Include the type name so failure modes
        # are distinguishable in logs.
        msg = f"engine error ({type(e).__name__}): {e}"
        if config.kind == "phosphene_local" and "Connection" in str(e):
            msg += "  — If using Phosphene Local, the model server may not be up yet (click Start Local Engine in the Agentic Flows settings drawer)."
        raise RuntimeError(msg) from e

    # Normalize LiteLLM's response object to our ChatResult shape.
    try:
        choice = resp.choices[0]
        msg = choice.message
    except (AttributeError, IndexError) as e:
        raise RuntimeError(f"engine response malformed: {e!r}") from e

    content = (getattr(msg, "content", None) or "")
    finish = getattr(choice, "finish_reason", None) or "stop"

    # Reasoning content (Anthropic extended thinking, mlx-lm reasoning
    # models). LiteLLM normalizes both to `reasoning_content`. Some
    # OpenAI-compat servers expose `reasoning` directly — try both.
    reasoning = (
        getattr(msg, "reasoning_content", None)
        or getattr(msg, "reasoning", None)
        or ""
    )

    # Reasoning truncation — preserve the same UX as the urllib path.
    if not content and reasoning:
        if finish == "length":
            raise RuntimeError(
                "Reasoning model truncated mid-thought "
                f"({len(reasoning)} chars of reasoning, no answer). "
                f"Bump 'Max tokens' in agent settings — current is "
                f"{config.max_tokens}; try 12000+ for Qwen 3.6 / DeepSeek R1."
            )
        # No length issue but content empty — promote reasoning so
        # SOMETHING surfaces in the chat.
        content = reasoning
        reasoning = ""

    usage: dict = {}
    try:
        u = getattr(resp, "usage", None)
        if u is not None:
            if hasattr(u, "model_dump"):
                usage = u.model_dump()
            elif hasattr(u, "dict"):
                usage = u.dict()
            else:
                usage = dict(u)
    except Exception:                           # noqa: BLE001
        pass

    return ChatResult(
        content=content,
        finish_reason=finish,
        usage=usage,
        reasoning=reasoning,
        model=getattr(resp, "model", "") or config.model,
    )


# -- Stdlib urllib fallback ---------------------------------------------------
def _chat_urllib_openai(messages: list[dict], config: EngineConfig,
                        *, timeout: int) -> ChatResult:
    """Pre-LiteLLM path. Same as the original engine.chat() body."""
    url = config.base_url.rstrip("/") + "/chat/completions"
    if config.kind == "phosphene_local" and config.local_model_path:
        wire_model = config.local_model_path
    else:
        wire_model = config.model
    body = {
        "model": wire_model,
        "messages": _normalize_for_wire(messages),
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", errors="replace")
        except Exception:                       # noqa: BLE001
            pass
        raise RuntimeError(
            f"engine returned HTTP {e.code} ({e.reason}). "
            f"URL: {url}  Detail: {detail[:600]}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"engine unreachable at {url}: {e.reason}. "
            "If using Phosphene Local, the model server may not be up yet "
            "— click Start Local Engine in the Agentic Flows settings drawer."
        ) from e

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"engine returned non-JSON body ({len(raw)} bytes). "
            f"First 200 chars: {raw[:200]!r}"
        ) from e

    if "choices" not in data or not data["choices"]:
        raise RuntimeError(f"engine response missing 'choices': {data!r}")

    msg = data["choices"][0].get("message") or {}
    content = msg.get("content") or ""
    finish = data["choices"][0].get("finish_reason") or "stop"
    usage = data.get("usage") or {}
    reasoning = msg.get("reasoning") or ""
    if not content and reasoning:
        if finish == "length":
            raise RuntimeError(
                "Reasoning model truncated mid-thought "
                f"({len(reasoning)} chars of reasoning, no answer). "
                f"Bump 'Max tokens' in agent settings — current is "
                f"{config.max_tokens}; try 12000+ for Qwen 3.6 / DeepSeek R1."
            )
        content = reasoning
        reasoning = ""

    return ChatResult(content=content, finish_reason=finish, usage=usage,
                      reasoning=reasoning,
                      model=data.get("model", config.model))


def _chat_anthropic_urllib(messages: list[dict], config: EngineConfig,
                           *, timeout: int) -> ChatResult:
    """Pre-LiteLLM Anthropic path — `/v1/messages` shape.

    Differences from the OpenAI shape we already handle:
      - System prompt is a top-level `system` STRING, not a {role:"system"}
        item in `messages`. We pop it out (assumes only one — see
        _normalize_for_wire which never adds a second).
      - `messages` must strictly alternate user/assistant (same as Gemma).
        _normalize_for_wire already collapses adjacent roles so this works.
      - Auth is via `x-api-key` header, NOT `Authorization: Bearer`.
      - Response is `{content: [{type:"text", text:"..."}], stop_reason}`
        — no `choices` array. Concatenate text blocks for the result.
      - `max_tokens` is REQUIRED.
    """
    base = config.base_url.rstrip("/") or "https://api.anthropic.com/v1"
    url = base + "/messages"
    norm = _normalize_for_wire(messages)
    system_text = ""
    if norm and norm[0].get("role") == "system":
        system_text = (norm[0].get("content") or "").strip()
        norm = norm[1:]
    body: dict = {
        "model": config.model or "claude-sonnet-4-5",
        "messages": norm,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
    }
    if system_text:
        body["system"] = system_text

    headers = {
        "Content-Type": "application/json",
        "anthropic-version": config.anthropic_version or "2023-06-01",
    }
    if config.api_key:
        headers["x-api-key"] = config.api_key

    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", errors="replace")
        except Exception:                       # noqa: BLE001
            pass
        nice = detail
        try:
            j = json.loads(detail)
            if isinstance(j, dict) and isinstance(j.get("error"), dict):
                err = j["error"]
                nice = f"{err.get('type','?')}: {err.get('message','?')}"
        except Exception:                       # noqa: BLE001
            pass
        raise RuntimeError(
            f"Anthropic API returned HTTP {e.code} ({e.reason}). "
            f"Detail: {nice[:600]}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Anthropic API unreachable at {url}: {e.reason}."
        ) from e

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Anthropic returned non-JSON ({len(raw)} bytes). "
            f"First 200 chars: {raw[:200]!r}"
        ) from e

    blocks = data.get("content") or []
    text_parts = [b.get("text") or "" for b in blocks if b.get("type") == "text"]
    content = "".join(text_parts)
    finish = data.get("stop_reason") or "end_turn"
    usage = data.get("usage") or {}
    return ChatResult(content=content, finish_reason=finish, usage=usage,
                      model=data.get("model", config.model))


def health_check(config: EngineConfig, *, timeout: int = 5) -> tuple[bool, str]:
    """Light probe of the engine. Returns (ok, message).

    `mlx-lm.server` exposes /v1/models. OpenAI / Anthropic / LM Studio
    all expose the same. We don't insist on a specific shape — any 200
    response means the endpoint is alive and accepts auth (if provided).

    Stays on stdlib urllib regardless of whether LiteLLM is installed —
    a 5s probe of /v1/models doesn't benefit from LiteLLM's retry/timeout
    machinery, and a transport-only probe leaves the LLM cold (faster).
    """
    base = config.base_url.rstrip("/") or (
        "https://api.anthropic.com/v1" if config.kind == "anthropic" else ""
    )
    url = base + "/models"
    headers = {}
    if config.kind == "anthropic":
        headers["anthropic-version"] = config.anthropic_version or "2023-06-01"
        if config.api_key:
            headers["x-api-key"] = config.api_key
    elif config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return True, f"engine reachable ({resp.status} on /v1/models)"
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return False, "engine returned 401 — check API key"
        return False, f"engine returned HTTP {e.code}"
    except urllib.error.URLError as e:
        return False, f"unreachable: {e.reason}"
    except Exception as e:                      # noqa: BLE001
        return False, f"probe failed: {e}"


def is_litellm_active() -> bool:
    """Diagnostic helper — surfaces in /agent/config GETs so the panel
    can show 'LiteLLM' vs 'urllib' in the status row."""
    return _HAS_LITELLM
