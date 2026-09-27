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
import socket
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


def resolve_api_key(provider: str) -> str:
    key_env = provider_key_env(provider)
    val = os.environ.get(key_env)
    if val and val.strip():
        return val.strip()
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[2]
    for fn in (".env.directories", ".env"):
        env_file = repo_root / fn
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line.startswith(f"{key_env}="):
                    v = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if v:
                        os.environ[key_env] = v
                        return v
    return os.environ.get(key_env, "")


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
        # Single flattening point: plural form splits user tool_result
        # blocks into wire tool messages; never pre-flatten at call sites.
        wire_messages = _openrouter_messages(messages)
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
                # internal schema key is input_schema; OpenAI wire wants parameters
                {"type": "function", "function": {"name": tool["name"], "description": tool.get("description", ""), "parameters": tool.get("input_schema") or tool.get("parameters", {})}}
                for tool in tools
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
    content = content or []  # None/empty content (e.g. tool-only turns)
    if message["role"] == "assistant":
        text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
        calls = []
        use_idx = 0
        for b in content:
            if b.get("type") == "tool_use":
                calls.append(
                    {
                        "id": b.get("id") or f"call_{use_idx}",
                        "type": "function",
                        "function": {"name": b["name"], "arguments": json.dumps(b.get("input", {}))},
                    }
                )
                use_idx += 1
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
                            "tool_call_id": block.get("tool_use_id") or "unknown",
                            "content": block.get("content", ""),
                        }
                    )
                else:
                    wire.append({"role": "user", "content": block.get("text", "")})
        else:
            wire.append(_openrouter_message(message))
    return wire


def parse_response(provider: str, payload: dict[str, Any], model_hint: str = "", requested_model: str = "") -> tuple[str, list[dict[str, Any]], UsageRecord, dict[str, Any]]:
    """Normalize a provider reply to (stop, blocks, UsageRecord, request_meta).

    ``request_meta`` ties each request to what actually served it — on
    OpenRouter, the returned model slug, upstream provider, and generation
    id — so a silent upstream routing change cannot hide inside a run.
    """
    if provider == "openrouter":
        choice = payload["choices"][0]
        message = choice["message"]
        stop = "tool_use" if choice.get("finish_reason") == "tool_calls" else "end_turn"
        blocks: list[dict[str, Any]] = []
        if message.get("content"):
            blocks.append({"type": "text", "text": message["content"]})
        for idx, call in enumerate(message.get("tool_calls", []) or []):
            # Some upstreams omit tool_call ids; synthesize stable ones —
            # the same enumeration order is used when flattening back.
            blocks.append(
                {
                    "type": "tool_use",
                    "id": call.get("id") or f"call_{idx}",
                    "name": call["function"]["name"],
                    "input": json.loads(call["function"]["arguments"] or "{}"),
                }
            )
        usage_raw = payload.get("usage", {})
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
        meta = {
            "requested_model": requested_model or model_hint,
            "returned_model": payload.get("model", ""),
            "upstream_provider": payload.get("provider", ""),
            "request_id": payload.get("id", ""),
        }
        return stop, blocks, usage, meta
    usage = payload["usage"]
    usage_out = UsageRecord(
        model=payload["model"],
        input_tokens=usage["input_tokens"],
        cached_input_tokens=usage.get("cache_read_input_tokens", 0),
        cache_write_tokens=usage.get("cache_creation_input_tokens", 0),
        output_tokens=usage["output_tokens"],
        reasoning_tokens=usage.get("reasoning_tokens", 0),
    )
    meta = {"requested_model": requested_model or model_hint, "returned_model": payload.get("model", ""),
            "upstream_provider": "anthropic", "request_id": payload.get("id", "")}
    return payload.get("stop_reason", ""), list(payload.get("content", [])), usage_out, meta


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
) -> tuple[str, list[dict[str, Any]], UsageRecord, dict[str, Any]]:
    """One agent turn: (stop_reason, content blocks, UsageRecord, request_meta)."""
    api_key = resolve_api_key(provider)
    if not api_key:
        raise RuntimeError(f"{provider_key_env(provider)} not set")
    if transport is None:
        transport = _urllib_transport
    url, headers, body = build_request(
        provider, model, system, messages, api_key, base_url, tools, generation
    )
    payload = transport(url, headers, body)
    return parse_response(provider, payload, model_hint=model, requested_model=model)


def _urllib_transport(url: str, headers: dict[str, str], body: bytes) -> dict[str, Any]:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        # Long timeout: free-tier models can reason for minutes, but a
        # stalled connection must not hang the whole benchmark forever.
        with urllib.request.urlopen(request, timeout=600) as reply:
            payload = json.loads(reply.read().decode("utf-8"))
    except urllib.error.HTTPError as err:  # surface the provider's reason
        detail = err.read().decode("utf-8", "replace")[:2000]
        raise RuntimeError(f"HTTP {err.code} from provider: {detail}") from err
    except (TimeoutError, socket.timeout, urllib.error.URLError) as err:
        raise RuntimeError(f"request timed out or connection failed: {err}") from err
    # OpenRouter signals upstream failures inside HTTP 200: {"error": {...}}
    if payload.get("error"):
        err = payload["error"]
        raise RuntimeError(
            f"provider error {err.get('code')}: {err.get('message')}"
            f" ({(err.get('metadata') or {}).get('error_type', 'unknown')})"
        )
    return payload


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
    stop, blocks, usage, _meta = run_turn(provider, model, system, messages, transport=transport, tools=tools)
    return "".join(b.get("text", "") for b in blocks if b.get("type") == "text"), usage
