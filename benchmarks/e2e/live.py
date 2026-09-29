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

from mintok.escalation import EscalationController, EscalationLevel, TrajectoryEvent  # noqa: E402
from model_runner import run_turn  # noqa: E402

AGENT_CLI = Path(__file__).resolve().parent / "agent_cli.py"


def _load_controller(log_path: Path) -> EscalationController:
    controller = EscalationController()
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                e = json.loads(line)
                controller.record_event(
                    TrajectoryEvent(
                        tool=e.get("tool", ""),
                        args=e.get("args"),
                        output=str(e.get("output", "")),
                        exit_code=e.get("exit_code", 0),
                        tokens=e.get("package_tokens", e.get("tokens", 0)),
                    )
                )
            except Exception:
                continue
    return controller


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
    "adaptive": (
        "You are an expert engineer working in a copy of a Python repository. "
        "Solve the task using MinTok adaptive tooling. "
        "Start with the slice tool to find candidate source locations. "
        "Read exact regions with read(path, start, end) and edit with patch(file, start, end, source). "
        "If slice was too narrow, use broaden. If a test fails with a traceback, use trace_slice. "
        "If you cannot find a file or want to locate definitions, use find_files or grep. "
        "Verify your changes with the suite tool before finishing. "
        "When the suite passes and you are done, reply with a short summary and no tool call."
    ),
    "v3": (
        "You are an expert software engineer working in a copy of a repository. "
        "Solve the task using all available tools (shell, patch, read, grep, find_files, suite). "
        "All tool outputs are virtualized: voluminous outputs are summarized with an 'obs:<id>' handle. "
        "Use expand(handle, filter) if you need raw output lines from an observation handle. "
        "Use investigate_failure or localize_symbol for fast deterministic macro-inspections. "
        "Run the suite tool before finishing. When tests pass, reply with a short summary and no tool call."
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
    "adaptive": [
        {
            "name": "slice",
            "description": (
                "Ranked exact source regions for a description of what to find "
                "or change. Returns file:line regions plus excerpts under a "
                "token budget."
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
            "name": "broaden",
            "description": "Expanded source regions with wider token budget when slice was too narrow.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "description": {"type": "string", "description": "description for expanded search"},
                },
                "required": ["description"],
            },
        },
        {
            "name": "trace_slice",
            "description": "Slice source around the failing frame of a test traceback.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "target": {"type": "string", "description": "optional file:line target frame"},
                },
            },
        },
        {
            "name": "find_files",
            "description": "Find files in repository matching a glob pattern (e.g. '*parser*' or '*.lark').",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "glob pattern"},
                },
            },
        },
        {
            "name": "grep",
            "description": "Grep for text or regex across repository source files.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "regex or string pattern to search"},
                    "path": {"type": "string", "description": "optional subpath to limit search"},
                },
                "required": ["pattern"],
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
        {
            "name": "shell",
            "description": "Run a shell command in the task root (unlocked when escalation level 4 is reached).",
            "input_schema": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
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
    "v3": [
        {
            "name": "shell",
            "description": "Run an arbitrary shell command in the task root. Output is virtualized if large.",
            "input_schema": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
        {
            "name": "expand",
            "description": "Expand an observation handle (obs:xxxx) to inspect raw output lines.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "handle": {"type": "string", "description": "observation handle like 'obs:71af'"},
                    "filter": {"type": "string", "description": "optional substring filter"},
                    "start": {"type": "integer", "description": "start line (0-indexed)"},
                    "count": {"type": "integer", "description": "maximum lines to retrieve"},
                },
                "required": ["handle"],
            },
        },
        {
            "name": "read",
            "description": "Bounded raw-file region read.",
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
            "name": "grep",
            "description": "Grep for text or regex across repository source files.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "regex or string pattern to search"},
                    "path": {"type": "string", "description": "optional subpath to limit search"},
                },
                "required": ["pattern"],
            },
        },
        {
            "name": "find_files",
            "description": "Find files in repository matching a glob pattern (e.g. '*parser*' or '*.lark').",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "glob pattern"},
                },
            },
        },
        {
            "name": "slice",
            "description": "Ranked exact source regions for a description of what to find or change.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "description": {"type": "string", "description": "what to find or change, in plain words"},
                },
                "required": ["description"],
            },
        },
        {
            "name": "investigate_failure",
            "description": "Macro-action: locate failing stack frame, map to AST symbol, and extract slice + callers.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "traceback": {"type": "string", "description": "optional traceback text"},
                },
            },
        },
        {
            "name": "localize_symbol",
            "description": "Macro-action: locate definition, signature, and callers for a symbol in 1 step.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "symbol name"},
                },
                "required": ["symbol"],
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
        elif name == "broaden":
            cmd = [sys.executable, str(AGENT_CLI), "--root", str(root), "--log", str(log), "--policy", policy, "broaden", args["description"]]
            stdin = None
        elif name == "trace_slice":
            cmd = [sys.executable, str(AGENT_CLI), "--root", str(root), "--log", str(log), "--policy", policy, "trace_slice"]
            if args.get("target"):
                cmd.append(str(args["target"]))
            stdin = None
        elif name == "find_files":
            cmd = [sys.executable, str(AGENT_CLI), "--root", str(root), "--log", str(log), "--policy", policy, "find_files"]
            if args.get("pattern"):
                cmd.append(str(args["pattern"]))
            stdin = None
        elif name == "grep":
            cmd = [sys.executable, str(AGENT_CLI), "--root", str(root), "--log", str(log), "--policy", policy, "grep", str(args["pattern"])]
            if args.get("path"):
                cmd.extend(["--path", str(args["path"])])
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
        elif name == "expand":
            cmd = [sys.executable, str(AGENT_CLI), "--root", str(root), "--log", str(log), "--policy", policy, "expand", args["handle"]]
            if args.get("filter"):
                cmd.extend(["--filter", str(args["filter"])])
            if args.get("start") is not None:
                cmd.extend(["--start", str(args["start"])])
            if args.get("count") is not None:
                cmd.extend(["--count", str(args["count"])])
            stdin = None
        elif name == "investigate_failure":
            cmd = [sys.executable, str(AGENT_CLI), "--root", str(root), "--log", str(log), "--policy", policy, "investigate_failure"]
            if args.get("traceback"):
                cmd.extend(["--traceback", str(args["traceback"])])
            stdin = None
        elif name == "localize_symbol":
            cmd = [sys.executable, str(AGENT_CLI), "--root", str(root), "--log", str(log), "--policy", policy, "localize_symbol", str(args["symbol"])]
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
    if policy == "v3":
        from mintok.conversation import StateCompiler
        from mintok.repo_profile import scan_repo_profile

        profile = scan_repo_profile(root)
        profile_header = profile.render_context()
        messages: list[dict] = [{"role": "user", "content": f"{profile_header}\n\nTask:\n{instruction}"}]
        compiler = StateCompiler(initial_goal=instruction)
    else:
        messages: list[dict] = [{"role": "user", "content": instruction}]
        compiler = None
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
    current_level = EscalationLevel.SLICE_BOUNDED
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
        if policy == "adaptive":
            ctrl = _load_controller(log)
            active_names = set(ctrl.active_tools())
            turn_tools = [t for t in TOOL_SCHEMAS["adaptive"] if t["name"] in active_names]
            if ctrl.level > current_level:
                current_level = ctrl.level
                if ctrl.level == EscalationLevel.BROADEN:
                    messages.append({
                        "role": "user",
                        "content": "[harness] Escalation Level 1 reached: 'broaden' tool is unlocked to expand slice search.",
                    })
                elif ctrl.level == EscalationLevel.TRACE_SLICED:
                    messages.append({
                        "role": "user",
                        "content": "[harness] Escalation Level 2 reached: 'trace_slice' tool is unlocked to slice around test traceback frames.",
                    })
                elif ctrl.level == EscalationLevel.TARGETED_DISCOVERY:
                    messages.append({
                        "role": "user",
                        "content": "[harness] Escalation Level 3 reached: targeted discovery tools (find_files, grep) are now unlocked to locate files.",
                    })
                elif ctrl.level == EscalationLevel.FULL_FALLBACK:
                    messages.append({
                        "role": "user",
                        "content": "[harness] Escalation Level 4 reached: stagnation detected. Unrestricted shell is now unlocked to diagnose, run tests, and fix the issue directly.",
                    })
        elif policy == "v3":
            turn_tools = tools
            if compiler is not None and turns >= 4 and turns % 3 == 0 and len(messages) > 4:
                compiled_view = compiler.state.render()
                messages = [
                    messages[0],
                    {"role": "user", "content": f"[harness compiled working state]\n{compiled_view}"},
                ] + messages[-2:]
        else:
            turn_tools = tools
        started = time.monotonic()
        if completion is not None:
            stop, blocks, usage = completion(model, system, messages, tools=turn_tools)[:3]
            meta = {}
        else:
            from model_runner import build_request, _urllib_transport, resolve_api_key

            api_key = resolve_api_key(provider)
            # build_request flattens internal blocks to the provider wire
            # exactly once; pre-flattening here would strip tool_call_id.
            url, headers, body = build_request(
                provider, model, system, messages, api_key, tools=turn_tools, generation=generation
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
            if dirty and turns < budget:
                out, _code = execute_tool(root, log, policy, "suite", {})
                dirty = False
                messages.append({"role": "assistant", "content": blocks})
                messages.append({
                    "role": "user",
                    "content": f"[harness] Verification check (you stopped with unverified edits):\n{out}\n"
                    "If tests failed, fix them. If tests passed, summarize your fix and finish with no tool call.",
                })
                continue
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
        if policy == "v3" and compiler is not None:
            for call, res in zip(calls, results):
                compiler.process_turn(
                    role="tool",
                    content="",
                    tool=call["name"],
                    tool_args=call.get("input", {}),
                    tool_output=res.get("content", ""),
                )
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
