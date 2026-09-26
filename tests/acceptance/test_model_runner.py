from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

scenarios("model_runner.feature")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "mintok_bench_model_runner", _REPO_ROOT / "benchmarks" / "e2e" / "model_runner.py"
)
assert _SPEC is not None and _SPEC.loader is not None
model_runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(model_runner)

SYSTEM = "You are a bench agent"
MESSAGES = [{"role": "user", "content": "Reply with pong"}]
REPLY: dict[str, Any] = {
    "model": "claude-sonnet-4-5",
    "content": [{"type": "text", "text": "pong"}],
    "usage": {
        "input_tokens": 1200,
        "cache_read_input_tokens": 8000,
        "cache_creation_input_tokens": 2000,
        "output_tokens": 900,
    },
}


class FakeTransport:
    """Stands in for the network; records every call it receives."""

    def __init__(self, reply: dict[str, Any]) -> None:
        self.reply = reply
        self.calls: list[tuple[str, dict[str, str], bytes]] = []

    def __call__(self, url: str, headers: dict[str, str], body: bytes) -> dict[str, Any]:
        self.calls.append((url, headers, body))
        return self.reply


class MustNotBeCalled:
    def __call__(self, url: str, headers: dict[str, str], body: bytes) -> dict[str, Any]:
        raise AssertionError("network transport must not be invoked")


@given("a fake Anthropic transport")
def fake_transport(ctx: SimpleNamespace) -> None:
    ctx.transport = FakeTransport(REPLY)


@given(parsers.parse('the model api key is set to "{key}"'))
def set_api_key(ctx: SimpleNamespace, key: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(model_runner.API_KEY_ENV, key)


@given("the model api key is unset")
def unset_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(model_runner.API_KEY_ENV, raising=False)


@given("a transport that must never be called")
def forbidden_transport(ctx: SimpleNamespace) -> None:
    ctx.transport = MustNotBeCalled()


@when(parsers.parse('the request is built for model "{model}" with api key "{key}"'))
def build_request(ctx: SimpleNamespace, model: str, key: str) -> None:
    ctx.api_key = key
    ctx.url, ctx.headers, ctx.body_bytes = model_runner.build_request(
        model, SYSTEM, MESSAGES, key
    )
    ctx.body = json.loads(ctx.body_bytes.decode("utf-8"))


@when("the reply is parsed")
def parse_reply(ctx: SimpleNamespace) -> None:
    ctx.usage = model_runner.parse_response(ctx.reply)


@when(
    parsers.parse(
        'a completion is run for system "{system}" and messages asking to reply "{word}"'
    )
)
def run_completion(ctx: SimpleNamespace, system: str, word: str) -> None:
    ctx.text, ctx.usage = model_runner.run_completion(
        "claude-sonnet-4-5",
        system,
        [{"role": "user", "content": f"Reply with {word}"}],
        transport=ctx.transport,
    )


@when("a completion is attempted")
def attempt_completion(ctx: SimpleNamespace) -> None:
    try:
        model_runner.run_completion(
            "claude-sonnet-4-5", SYSTEM, MESSAGES, transport=ctx.transport
        )
    except RuntimeError as error:
        ctx.error = error
    else:
        ctx.error = None


@then(parsers.parse('the request URL ends with "{suffix}"'))
def request_url(ctx: SimpleNamespace, suffix: str) -> None:
    assert ctx.url.endswith(suffix), ctx.url


@then(parsers.parse('the request headers carry "{header}" and "{other}"'))
def request_headers(ctx: SimpleNamespace, header: str, other: str) -> None:
    assert ctx.headers[header] == ctx.api_key
    assert other in ctx.headers


@then("the body carries the model, max_tokens, system and messages")
def body_shape(ctx: SimpleNamespace) -> None:
    assert ctx.body["model"] == "claude-sonnet-4-5"
    assert ctx.body["max_tokens"] > 0
    assert ctx.body["system"][0]["text"] == SYSTEM
    assert ctx.body["messages"] == MESSAGES


@then(parsers.parse('the system prompt carries an ephemeral "{attribute}" block'))
def cache_control(ctx: SimpleNamespace, attribute: str) -> None:
    assert ctx.body["system"][0][attribute] == {"type": "ephemeral"}


@then("the serialized body does not contain the api key")
def no_key_in_body(ctx: SimpleNamespace) -> None:
    assert ctx.api_key not in ctx.body_bytes.decode("utf-8")


@given(
    parsers.parse(
        "an Anthropic reply with {input} input, {cache_read} cache read, "
        "{cache_creation} cache creation and {output} output tokens"
    )
)
def anthropic_reply(
    ctx: SimpleNamespace, input: str, cache_read: str, cache_creation: str, output: str
) -> None:
    ctx.reply = {
        "model": "claude-sonnet-4-5",
        "content": [{"type": "text", "text": "pong"}],
        "usage": {
            "input_tokens": int(input),
            "cache_read_input_tokens": int(cache_read),
            "cache_creation_input_tokens": int(cache_creation),
            "output_tokens": int(output),
        },
    }


@then(
    parsers.parse(
        "the usage record has {input:d} input, {cached:d} cached input, "
        "{write:d} cache write and {output:d} output tokens"
    )
)
def usage_mapping(ctx: SimpleNamespace, input: int, cached: int, write: int, output: int) -> None:
    assert ctx.usage.model == "claude-sonnet-4-5"
    assert ctx.usage.input_tokens == input
    assert ctx.usage.cached_input_tokens == cached
    assert ctx.usage.cache_write_tokens == write
    assert ctx.usage.output_tokens == output
    assert ctx.usage.reasoning_tokens == 0


@then(parsers.parse('the completion text is "{text}"'))
def completion_text(ctx: SimpleNamespace, text: str) -> None:
    assert ctx.text == text


@then(parsers.parse('a RuntimeError names the missing environment variable "{name}"'))
def missing_key_error(ctx: SimpleNamespace, name: str) -> None:
    assert isinstance(ctx.error, RuntimeError)
    assert name in ctx.error.args[0]
