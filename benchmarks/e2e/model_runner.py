"""Anthropic Messages API runner for benchmark harnesses.

Stdlib ``urllib`` only — no SDK dependency, no secrets in code. The API
key comes from the ``MINTOK_MODEL_API_KEY`` environment variable and is
never printed, logged, or embedded in a request body.

Usage mapping convention: MinTok's ``UsageRecord.input_tokens`` EXCLUDES
cached tokens. Anthropic's ``usage.input_tokens`` already excludes cache
reads and cache creation, so it passes through unchanged;
``cache_read_input_tokens`` maps to ``cached_input_tokens`` and
``cache_creation_input_tokens`` maps to ``cache_write_tokens``.
"""

from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Callable
from typing import Any

from mintok.billing import UsageRecord

DEFAULT_BASE_URL = "https://api.anthropic.com"
API_KEY_ENV = "MINTOK_MODEL_API_KEY"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 4096

#: A transport performs one HTTP POST and returns the parsed JSON reply.
Transport = Callable[[str, dict[str, str], bytes], dict[str, Any]]


def build_request(
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    api_key: str,
    base_url: str | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> tuple[str, dict[str, str], bytes]:
    """Build the URL, headers, and JSON body for one Messages API call.

    Prompt caching is enabled on the system prompt via an ephemeral
    ``cache_control`` block. The API key appears only in the headers.
    """
    url = (base_url or DEFAULT_BASE_URL).rstrip("/") + "/v1/messages"
    headers = {
        "content-type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
    }
    body: dict[str, Any] = {
        "model": model,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "system": [
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": messages,
    }
    if tools:
        body["tools"] = tools
    return url, headers, json.dumps(body).encode("utf-8")


def parse_blocks(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Content blocks of a reply: text plus tool_use blocks."""
    return list(payload.get("content", []))


def tool_uses(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The tool_invocation blocks a reply requests."""
    return [b for b in blocks if b.get("type") == "tool_use"]


def parse_response(payload: dict[str, Any]) -> UsageRecord:
    """Map an Anthropic reply onto a UsageRecord (see module docstring)."""
    usage = payload["usage"]
    return UsageRecord(
        model=payload["model"],
        input_tokens=usage["input_tokens"],
        cached_input_tokens=usage.get("cache_read_input_tokens", 0),
        cache_write_tokens=usage.get("cache_creation_input_tokens", 0),
        output_tokens=usage["output_tokens"],
    )


def run_completion(
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    *,
    transport: Transport | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> tuple[str, UsageRecord]:
    """Run one completion and return ``(text, UsageRecord)``.

    ``transport`` defaults to a ``urllib.request`` POST. The API key is
    read from the environment before any network attempt is made.
    """
    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"{API_KEY_ENV} not set")
    if transport is None:
        transport = _urllib_transport
    url, headers, body = build_request(model, system, messages, api_key, tools=tools)
    payload = transport(url, headers, body)
    text = "".join(
        block.get("text", "") for block in payload.get("content", []) if block.get("type") == "text"
    )
    return text, parse_response(payload)


def run_turn(
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    *,
    transport: Transport | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> tuple[str, list[dict[str, Any]], UsageRecord]:
    """One agent turn: returns (stop_reason, content blocks, usage)."""
    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"{API_KEY_ENV} not set")
    if transport is None:
        transport = _urllib_transport
    url, headers, body = build_request(model, system, messages, api_key, tools=tools)
    payload = transport(url, headers, body)
    return payload.get("stop_reason", ""), parse_blocks(payload), parse_response(payload)


def _urllib_transport(url: str, headers: dict[str, str], body: bytes) -> dict[str, Any]:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request) as reply:
        return json.loads(reply.read().decode("utf-8"))
