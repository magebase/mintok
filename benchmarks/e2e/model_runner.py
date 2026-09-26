"""Model API runners for benchmark harnesses: Anthropic Messages and
OpenRouter (OpenAI-compatible, free models available).

Stdlib ``urllib`` only — no SDK dependency, no secrets in code. The API
key comes from the provider's environment variable
(``MINTOK_MODEL_API_KEY`` for Anthropic, ``OPENROUTER_API_KEY`` for
OpenRouter) and is never printed, logged, or embedded in a request body.

Both providers normalize to one internal shape so the loop driver stays
provider-agnostic: content blocks (text + ``tool_use``), a stop reason,
and a ``UsageRecord``. Raw provider usage is preserved verbatim in the
usage sidecars — raw usage is immutable evidence; the price table is a
replaceable interpretation applied afterward.

Usage convention: ``input_tokens`` EXCLUDES cached tokens (matches
Anthropic reporting; OpenRouter's ``prompt_tokens`` maps here with
cached tokens subtracted when reported).
"""

from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Callable
from typing import Any

from mintok.billing import UsageRecord

ANTHROPIC_DEFAULT_BASE = "https://api.anthropic.com"
OPENROUTER_DEFAULT_BASE = "https://openrouter.ai/api/v1"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 4096
#: Generation settings recorded in run manifests. Values marked
#: provider-fixed are recorded as such, never invented.
GENERATION_DEFAULTS = {"temperature": 0.0, "top_p": "provider-default", "max_output_tokens": DEFAULT_MAX_TOKENS}

#: A transport performs one HTTP POST and returns the parsed JSON reply.
Transport = Callable[[str, dict[str, str], bytes], dict[str, Any]]


def provider_base(provider: str, base_url: str | None = None) -> str:
    if provider == "openrouter":
        return (base_url or os.environ.get("OPENROUTER_BASE_URL") or OPENROUTER_DEFAULT_BASE).rstrip("/")
    return (base_url or ANTHROPIC_DEFAULT_BASE).rstrip("/")


def provider_headers(provider: str, api_key: str) -> dict[str, str]:
    if provider == "openrouter":
        return {
            "content-type": "application/json",
            "authorization": f"Bearer {api_key}",
        }
    return {
        "content-type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
    }


def provider_key_env(provider: str) -> str:
    return "OPENROUTER_API_KEY" if provider == "openrouter" else "MINTOK_MODEL_API_KEY"


def build_request(
    provider: str,
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    api_key: str,
    base_url: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    generation: dict[str, Any] | None = None,
) -> tuple[str, dict[str, str], bytes]:
    """Build the URL, headers, and JSON body for one chat API call.

    The Anthropic path enables prompt caching on the system prompt via an
    ephemeral ``cache_control`` block. Messages arrive in the internal
    block shape and are translated to the provider's wire format.
    """
    gen = dict(GENERATION_DEFAULTS)
    gen.update(generation or {})
    if provider == "openrouter":
        url = provider_base(provider, base_url) + "/chat/completions"
        wire_messages = [_openrouter_message(m) for m in messages]
        body: dict[str, Any] = {
            "model": model,
            "max_tokens": gen["max_output_tokens"],
            "temperature": gen["temperature"],
            "messages": [{"role": "system", "content": system}, *wire_messages],
        }
        if not isinstance(gen["top_p"], str):  # provider-fixed stays unset
            body["top_p"] = gen["top_p"]
        if tools:
            body["tools"] = [
                {"type": "function", "function": tool} for tool in tools
            ]
    else:
        url = provider_base(provider, base_url) + "/v1/messages"
        headers = provider_headers(provider, api_key)
        body = {
            "model": model,
            "max_tokens": gen["max_output_tokens"],
            "temperature": gen["temperature"],
            "system": [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ],
            "messages": messages,
        }
        if not isinstance(gen["top_p"], str):
            body["top_p"] = gen["top_p"]
        if tools:
            body["tools"] = tools
        return url, headers, json.dumps(body).encode("utf-8")
    return url, provider_headers(provider, api_key), json.dumps(body).encode("utf-8")


def _openrouter_message(message: dict[str, Any]) -> dict[str, Any]:
    """Internal Anthropic-shaped block message -> OpenAI wire message."""
    content = message.get("content")
    if isinstance(content, str):
        return {"role": message["role"], "content": content}
    if message["role"] == "assistant":
        text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
        calls = [
            {
                "id": b["id"],
                "type": "function",
                "function": {"name": b["name"], "arguments": json.dumps(b.get("input", {}))},
            }
            for b in content
            if b.get("type") == "tool_use"
        ]
        out: dict[str, Any] = {"role": "assistant", "content": text or None, "tool_calls": calls}
        return out
    # tool_result blocks -> one OpenAI tool message per result
    raise ValueError("tool results must be flattened before sending; see _openrouter_messages")


def _openrouter_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wire: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if message["role"] == "user" and isinstance(content, list):
            for block in content:
                if block.get("type") == "tool_result":
                    wire.append(
                        {
                            "role": "tool",
                            "tool_call_id": block["tool_use_id"],
                            "content": block.get("content", ""),
                        }
                    )
                else:
                    wire.append({"role": "user", "content": block.get("text", "")})
        else:
            wire.append(_openrouter_message(message))
    return wire


def parse_response(provider: str, payload: dict[str, Any], model_hint: str = "") -> tuple[str, list[dict[str, Any]], UsageRecord]:
    """Normalize a provider reply to (stop_reason, blocks, UsageRecord)."""
    if provider == "openrouter":
        choice = payload["choices"][0]
        message = choice["message"]
        stop = "tool_use" if choice.get("finish_reason") == "tool_calls" else "end_turn"
        blocks: list[dict[str, Any]] = []
        if message.get("content"):
            blocks.append({"type": "text", "text": message["content"]})
        for call in message.get("tool_calls", []) or []:
            blocks.append(
                {
                    "type": "tool_use",
                    "id": call["id"],
                    "name": call["function"]["name"],
                    "input": json.loads(call["function"]["arguments"] or "{}"),
                }
            )
        usage_raw = payload.get("usage", {})
        cached = 0
        details = usage_raw.get("prompt_tokens_details") or {}
        cached = details.get("cached_tokens", 0)
        prompt = usage_raw.get("prompt_tokens", 0)
        usage = UsageRecord(
            model=payload.get("model", model_hint),
            input_tokens=max(prompt - cached, 0),
            cached_input_tokens=cached,
            output_tokens=usage_raw.get("completion_tokens", 0),
            reasoning_tokens=usage_raw.get("completion_tokens_details", {}).get("reasoning_tokens", 0),
        )
        return stop, blocks, usage
    usage = payload["usage"]
    usage_out = UsageRecord(
        model=payload["model"],
        input_tokens=usage["input_tokens"],
        cached_input_tokens=usage.get("cache_read_input_tokens", 0),
        cache_write_tokens=usage.get("cache_creation_input_tokens", 0),
        output_tokens=usage["output_tokens"],
        reasoning_tokens=usage.get("reasoning_tokens", 0),
    )
    return payload.get("stop_reason", ""), list(payload.get("content", [])), usage_out


def run_turn(
    provider: str,
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    *,
    transport: Transport | None = None,
    tools: list[dict[str, Any]] | None = None,
    base_url: str | None = None,
    generation: dict[str, Any] | None = None,
) -> tuple[str, list[dict[str, Any]], UsageRecord]:
    """One agent turn: returns (stop_reason, content blocks, UsageRecord)."""
    api_key = os.environ.get(provider_key_env(provider))
    if not api_key:
        raise RuntimeError(f"{provider_key_env(provider)} not set")
    if transport is None:
        transport = _urllib_transport
    url, headers, body = build_request(
        provider, model, system, messages, api_key, base_url, tools, generation
    )
    payload = transport(url, headers, body)
    return parse_response(provider, payload, model_hint=model)


def _urllib_transport(url: str, headers: dict[str, str], body: bytes) -> dict[str, Any]:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request) as reply:
        return json.loads(reply.read().decode("utf-8"))


def run_completion(
    provider: str,
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    *,
    transport: Transport | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> tuple[str, UsageRecord]:
    """Text-only completion helper (no tool plumbing)."""
    stop, blocks, usage = run_turn(provider, model, system, messages, transport=transport, tools=tools)
    return "".join(b.get("text", "") for b in blocks if b.get("type") == "text"), usage
