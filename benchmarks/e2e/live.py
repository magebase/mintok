"""Live agent loop for isolated arm runs.

Drives one frontier-model agent over one task copy: each turn sends the
conversation to the model with the arm's tool schemas, executes the
requested tool calls through the logging shim (agent_cli.py, the only
sanctioned interface), and feeds results back until the model stops
calling tools or the turn cap hits. The shim log stays the exact record
of tool-side context; the loop only orchestrates.

Testable offline: pass ``completion`` (a run_turn-compatible callable) to
drive the loop from a scripted fake model.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HARNESS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HARNESS_ROOT / "benchmarks" / "e2e"))
sys.path.insert(0, str(HARNESS_ROOT / "src"))

from model_runner import run_turn  # noqa: E402

AGENT_CLI = Path(__file__).resolve().parent / "agent_cli.py"


def _empty_usage() -> dict:
    return {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "cache_write_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "latency_s": 0.0,
    }


def _add_usage(total: dict, usage) -> None:  # usage: UsageRecord | None
    if usage is None:
        return
    for field_ in ("input_tokens", "cached_input_tokens", "cache_write_tokens", "output_tokens", "reasoning_tokens"):
        total[field_] += getattr(usage, field_, 0)
MAX_TURNS = 25
VERIFY_RESERVE = 3  # turns before the cap at which the verify nudge fires

DISCIPLINE = {
    "control": (
        "You are a careful engineer working in a copy of a Python repository. "
        "Solve the task using ordinary shell tooling (grep/sed/python one-liners). "
        "Run the test suite with the suite tool before finishing. "
        "When you are done, reply with a short summary and no tool call."
    ),
    "S": (
        "You are a careful engineer working in a copy of a Python repository. "
        "Solve the task. Start with the slice tool: give it a short description "
        "of what you need to find or change, and it returns ranked exact source "
        "regions (file:line) with excerpts. Use read(path, start, end) only for "
        "regions the slice did not cover — raw reads are tracked as fallback. "
        "Edit with patch(file, start, end) — the slice gives you the line "
        "ranges. Run the suite tool to verify, then reply with a short summary "
        "and no tool call."
    ),
}

TOOL_SCHEMAS = {
    "S": [
        {
            "name": "slice",
            "description": (
                "Ranked exact source regions for a description of what to find "
                "or change. Returns file:line regions plus excerpts under a "
                "token budget. Prefer this over reading the file."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "description": {"type": "string", "description": "what to find or change, in plain words"},
                },
                "required": ["description"],
            },
        },
        {
            "name": "read",
            "description": "Bounded raw-file region read (fallback; tracked).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start": {"type": "integer"},
                    "end": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
        {
            "name": "patch",
            "description": "Replace a 1-based inclusive line range with new source.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "file": {"type": "string"},
                    "start": {"type": "integer"},
                    "end": {"type": "integer"},
                    "source": {"type": "string", "description": "replacement lines"},
                },
                "required": ["file", "start", "end", "source"],
            },
        },
        {
            "name": "suite",
            "description": "Run the task copy's full test suite.",
            "input_schema": {"type": "object", "properties": {}},
        },
    ],
    "control": [
        {
            "name": "shell",
            "description": "Run a shell command in the task root.",
            "input_schema": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
        {
            "name": "suite",
            "description": "Run the task copy's full test suite.",
            "input_schema": {"type": "object", "properties": {}},
        },
    ],
}


def execute_tool(root: Path, log: Path, policy: str, name: str, args: dict) -> tuple[str, int]:
    """One tool call through the logging shim; returns (output, exit_code).

    A malformed tool call is a recoverable model error, never a harness
    crash: bad/missing arguments come back as an error tool result that
    names the expected schema, and the model retries.
    """
    try:
        if name == "shell":
            cmd = [sys.executable, str(AGENT_CLI), "--root", str(root), "--log", str(log), "--policy", policy, "shell", args["command"]]
            stdin = None
        elif name == "slice":
            cmd = [sys.executable, str(AGENT_CLI), "--root", str(root), "--log", str(log), "--policy", policy, "slice", args["description"]]
            stdin = None
        elif name == "read":
            cmd = [sys.executable, str(AGENT_CLI), "--root", str(root), "--log", str(log), "--policy", policy, "read", args["path"]]
            if args.get("start") is not None:
                cmd += [str(args["start"])]
                if args.get("end") is not None:
                    cmd += [str(args["end"])]
            stdin = None
        elif name == "patch":
            cmd = [sys.executable, str(AGENT_CLI), "--root", str(root), "--log", str(log), "--policy", policy, "patch", "--file", args["file"], "--start", str(args["start"]), "--end", str(args["end"]), "--stdin"]
            stdin = args["source"]
        elif name == "suite":
            cmd = [sys.executable, str(AGENT_CLI), "--root", str(root), "--log", str(log), "--policy", policy, "suite", "--quiet"]
            stdin = None
        else:
            return f"error: unknown tool {name}", 1
    except (KeyError, TypeError) as err:
        expected = next((t["input_schema"] for t in TOOL_SCHEMAS[policy] if t["name"] == name), {})
        return (
            f"error: malformed {name} arguments ({err}); expected schema: {json.dumps(expected)}",
            1,
        )
    proc = subprocess.run(cmd, input=stdin, capture_output=True, text=True, timeout=700)
    return (proc.stdout + proc.stderr).rstrip(), proc.returncode


def run_loop(
    root: Path,
    log: Path,
    policy: str,
    instruction: str,
    model: str,
    completion=None,
    max_turns: int = MAX_TURNS,
    provider: str = "anthropic",
    generation: dict | None = None,
) -> dict:
    """Drive one agent to completion; returns a turn and usage summary.

    With a real model, per-call usage is accumulated (input, cached
    input, cache writes, output, reasoning, latency) and written next to
    the shim log as ``<stem>.usage.json`` — raw provider usage, immutable
    evidence; ``usd`` is one interpretation via the recorded price table,
    with per-turn billing rows in ``<stem>.billing.jsonl`` for
    recomputation when prices change.
    """
    import os
    import time

    from mintok.billing import PriceTable, billing_row
    from model_runner import parse_response, provider_key_env

    prices = PriceTable.load(os.environ.get("MINTOK_PRICE_OVERRIDES"))
    price_note = "overrides" if os.environ.get("MINTOK_PRICE_OVERRIDES") else "defaults-2026-06"

    system = DISCIPLINE[policy]
    tools = TOOL_SCHEMAS[policy]
    messages: list[dict] = [{"role": "user", "content": instruction}]
    turns = 0
    completed = False
    usage_total = _empty_usage()
    per_turn: list[dict] = []
    request_metas: list[dict] = []
    # Verify-then-stop (arm-neutral): a workspace with unverified edits
    # never reaches the turn cap silently. Near the cap the model is
    # nudged toward the suite; at the cap the harness runs the suite
    # itself once and reserves exactly one reaction turn.
    dirty = False
    nudged = False
    forced_verify = False
    budget = max_turns
    while not completed:
        remaining = budget - turns
        if remaining <= 0:
            if dirty and not forced_verify:
                forced_verify = True
                budget += 1  # reserve one reaction turn after the check
                out, _code = execute_tool(root, log, policy, "suite", {})
                messages.append({
                    "role": "user",
                    "content": f"[harness] turn budget exhausted with unverified edits; forced verification:\n{out}",
                })
                continue
            break
        if dirty and not nudged and remaining <= VERIFY_RESERVE:
            nudged = True
            messages.append({
                "role": "user",
                "content": "[harness] turn budget nearly exhausted with unverified edits. "
                "Run the suite now; stop when it passes.",
            })
        started = time.monotonic()
        if completion is not None:
            stop, blocks, usage = completion(model, system, messages, tools=tools)[:3]
            meta = {}
        else:
            from model_runner import build_request, _urllib_transport

            api_key = os.environ[provider_key_env(provider)]
            # build_request flattens internal blocks to the provider wire
            # exactly once; pre-flattening here would strip tool_call_id.
            url, headers, body = build_request(
                provider, model, system, messages, api_key, tools=tools, generation=generation
            )
            payload = None
            for attempt in range(8):  # free-tier upstreams overload for minutes
                try:
                    payload = _urllib_transport(url, headers, body)
                    break
                except RuntimeError as err:
                    if attempt == 7:
                        raise
                    wait = min(15.0 * (3 ** attempt), 300.0)
                    print(f"  [retry {attempt + 1}/7 after {wait:.0f}s: {err}]")
                    time.sleep(wait)
            stop, blocks, usage, meta = parse_response(provider, payload, model_hint=model, requested_model=model)
        usage_total["latency_s"] += time.monotonic() - started
        _add_usage(usage_total, usage)
        if meta:
            request_metas.append(meta)
        if usage is not None:
            try:
                breakdown = prices.cost(model, usage)
            except KeyError:
                breakdown = None  # unknown model: raw usage preserved, $ deferred
            if breakdown is not None:
                per_turn.append(billing_row(log.stem, log.stem, turns, usage, breakdown))
        turns += 1
        calls = [b for b in blocks if b.get("type") == "tool_use"]
        if not calls or stop != "tool_use":
            completed = True
            break
        results = []
        for call in calls:
            out, code = execute_tool(root, log, policy, call["name"], call.get("input", {}))
            if call["name"] in ("patch", "shell"):
                dirty = True
            elif call["name"] == "suite":
                dirty = False
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": call["id"],
                    "content": out,
                    "is_error": code != 0,
                }
            )
        messages.append({"role": "assistant", "content": blocks})
        messages.append({"role": "user", "content": results})
    usage_total["turns"] = turns
    usage_total["model"] = model
    usage_total["provider"] = provider
    usage_total["price_table"] = price_note
    usage_total["usd"] = round(sum(row["cost_total_usd"] for row in per_turn), 6)
    if meta:
        # What actually served the requests: requested vs returned model,
        # upstream provider, request ids. OpenRouter can re-route silently.
        for row, m in zip(per_turn, request_metas):
            row.update(m)
        usage_total["requests"] = request_metas
    if completion is None:  # real runs only; fake completions bill nothing
        log.parent.mkdir(parents=True, exist_ok=True)
        (log.parent / f"{log.stem}.usage.json").write_text(json.dumps(usage_total, indent=2))
        with (log.parent / f"{log.stem}.billing.jsonl").open("w") as fh:
            for row in per_turn:
                fh.write(json.dumps(row) + "\n")
    return {"turns": turns, "completed": completed, "usage": usage_total}


def main() -> int:
    parser = argparse.ArgumentParser(prog="live")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--policy", required=True, choices=sorted(DISCIPLINE))
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--provider", default="anthropic", choices=["anthropic", "openrouter"])
    parser.add_argument("--model", default="claude-sonnet-4-5")
    args = parser.parse_args()
    summary = run_loop(args.root, args.log, args.policy, args.instruction, args.model, provider=args.provider)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
