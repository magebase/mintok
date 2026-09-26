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
MAX_TURNS = 25

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
    """One tool call through the logging shim; returns (output, exit_code)."""
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
) -> dict:
    """Drive one agent to completion; returns a turn summary."""
    system = DISCIPLINE[policy]
    tools = TOOL_SCHEMAS[policy]
    messages: list[dict] = [{"role": "user", "content": instruction}]
    turns = 0
    completed = False
    for _ in range(max_turns):
        turn_out = completion or run_turn
        stop, blocks, usage = turn_out(model, system, messages, tools=tools)
        turns += 1
        calls = [b for b in blocks if b.get("type") == "tool_use"]
        if not calls or stop != "tool_use":
            completed = True
            break
        results = []
        for call in calls:
            out, code = execute_tool(root, log, policy, call["name"], call.get("input", {}))
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
    return {"turns": turns, "completed": completed}


def main() -> int:
    parser = argparse.ArgumentParser(prog="live")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--policy", required=True, choices=sorted(DISCIPLINE))
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--model", default="claude-sonnet-4-5")
    args = parser.parse_args()
    summary = run_loop(args.root, args.log, args.policy, args.instruction, args.model)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
