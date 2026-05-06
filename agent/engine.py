"""OpenAI-compatible chat client.

One function: send a list of messages, get back the assistant's text.
No streaming, no tool-calling spec — the agent emits actions as fenced
```action JSON blocks in plain content (see prompts.py). That convention
works on every Chat Completions server we'd want to support, regardless
of whether the server / model implements OpenAI's tool-calling shape.

Stdlib only. Phosphene's venv has urllib + json; we don't pull a new dep.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field


# 5 minutes. Local models can take 30+ s to first token on a cold cache;
# the first request after spawning mlx-lm.server is the slowest. Keep
# this generous so an under-warmed model doesn't fail the first turn.
DEFAULT_TIMEOUT_S = 300


@dataclass
class EngineConfig:
    """How the agent talks to its LLM.

    `kind` is informational — both branches end up calling the same
    OpenAI-compatible /chat/completions endpoint. The local branch just
    promises a panel-spawned mlx-lm.server on `base_url`.
    """

    kind: str = "phosphene_local"          # "phosphene_local" | "ollama" | "custom"
    base_url: str = "http://127.0.0.1:8200/v1"
    model: str = "mlx-community/gemma-3-12b-it-4bit"
    api_key: str = ""
    temperature: float = 0.4
    max_tokens: int = 3072
    # mlx-lm.server boot args — only used when kind == "phosphene_local".
    # `local_model_path` may be a Hugging Face id ("Qwen/Qwen3-30B-A3B-Instruct-4bit")
    # OR an absolute path to a local model dir (the bundled Gemma path).
    local_model_path: str = ""
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

    Raises RuntimeError on transport failure or non-200 response.
    Server-side errors bubble up with the upstream body so the user
    sees what the engine actually said (helpful for debugging API key
    / model name typos in the Settings drawer).
    """
    url = config.base_url.rstrip("/") + "/chat/completions"
    # mlx-lm.server identifies models by their LOAD PATH, not by short
    # name. Its /v1/models endpoint returns the absolute path (e.g.
    # "/Users/.../mlx_models/gemma-3-12b-it-4bit") as the model id, and
    # /v1/chat/completions 404s with a HuggingFace lookup error
    # ("Repository Not Found for url: https://huggingface.co/api/models/
    # gemma-3-12b-it-4bit/revision/main") if it receives anything else
    # — it interprets unknown short names as HF repo ids and tries to
    # download them. So for the local engine we always pass the absolute
    # path as the request "model" field.
    #
    # Ollama and Custom OpenAI-compat endpoints get `config.model` as-is
    # — Ollama wants tags like "qwen2.5-coder:32b", remote APIs want
    # short ids like "claude-sonnet-4-6" or "gpt-5".
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
        # Read the body so the caller can see why the server rejected.
        detail = ""
        try:
            detail = e.read().decode("utf-8", errors="replace")
        except Exception:
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
    return ChatResult(content=content, finish_reason=finish, usage=usage,
                      model=data.get("model", config.model))


def health_check(config: EngineConfig, *, timeout: int = 5) -> tuple[bool, str]:
    """Light probe of the engine. Returns (ok, message).

    `mlx-lm.server` exposes /v1/models. OpenAI / Anthropic / LM Studio
    all expose the same. We don't insist on a specific shape — any 200
    response means the endpoint is alive and accepts auth (if provided).
    """
    url = config.base_url.rstrip("/") + "/models"
    headers = {}
    if config.api_key:
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
    except Exception as e:                  # noqa: BLE001 — surface any other transport hiccup
        return False, f"probe failed: {e}"
