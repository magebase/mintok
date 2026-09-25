"""Logging agent shim for the end-to-end benchmark.

Every agent tool call goes through this shim and is appended, one JSON object
per line, to a trajectory log: timestamp, tool name, arguments, and the full
output the agent saw. The shim is the only sanctioned interface for the
benchmark arms, so the log is an exact record of tool-side context exposure.

Subcommands:
  shell "CMD"            run CMD (shell) in the task root; any command (control arm)
  query OP TARGET        mintok ABI query (find|symbol|effects|callers|writers|summary)
  change --target SID    replace one symbol; source via --source-file or --stdin
  verify -- ARGV...      run a check command, get pass/fail + tail
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HARNESS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HARNESS_ROOT / "src"))

from mintok.abi import AgentABI  # noqa: E402

VENV_PY = HARNESS_ROOT / ".venv" / "bin" / "python"


def append(log_path: Path, entry: dict) -> None:
    entry["ts"] = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent_cli")
    parser.add_argument("--root", type=Path, required=True, help="task working copy")
    parser.add_argument("--log", type=Path, required=True, help="trajectory JSONL")
    sub = parser.add_subparsers(dest="tool", required=True)

    shell = sub.add_parser("shell", help="run a shell command in the task root")
    shell.add_argument("command")

    query = sub.add_parser("query", help="mintok ABI query")
    query.add_argument("op")
    query.add_argument("target")

    change = sub.add_parser("change", help="replace one symbol via the mintok ABI")
    change.add_argument("--target", required=True)
    src = change.add_mutually_exclusive_group(required=True)
    src.add_argument("--source-file")
    src.add_argument("--stdin", action="store_true")

    remove = sub.add_parser("remove", help="delete one indexed symbol via the mintok ABI")
    remove.add_argument("--target", required=True)

    verify = sub.add_parser("verify", help="run a check command via the mintok ABI")
    verify.add_argument("argv", nargs="+")

    suite = sub.add_parser("suite", help="run the task copy's full test suite")
    suite.add_argument("--quiet", action="store_true")

    args = parser.parse_args(argv)
    entry: dict = {"tool": args.tool}

    if args.tool == "shell":
        entry["args"] = args.command
        proc = subprocess.run(
            args.command, shell=True, cwd=args.root, capture_output=True, text=True, timeout=600
        )
        entry["output"] = (proc.stdout + proc.stderr).rstrip()
        entry["exit_code"] = proc.returncode
        append(args.log, entry)
        print(entry["output"])
        return proc.returncode

    if args.tool == "query":
        entry["args"] = {"op": args.op, "target": args.target}
        try:
            output = AgentABI(args.root).query(args.op, args.target)
        except Exception as exc:  # noqa: BLE001 - the agent sees the failure
            entry["output"] = f"error: {exc}"
            entry["exit_code"] = 1
            append(args.log, entry)
            print(entry["output"])
            return 1
        entry["output"] = output
        entry["exit_code"] = 0
        append(args.log, entry)
        print(output)
        return 0

    if args.tool == "change":
        entry["args"] = {"target": args.target}
        if args.stdin:
            source = sys.stdin.read()
        else:
            source = Path(args.source_file).read_text()
        try:
            abi = AgentABI(args.root)
            result = abi.change(args.target, source)
        except Exception as exc:  # noqa: BLE001 - the agent sees the failure
            entry["output"] = f"rejected: {exc}"
            entry["exit_code"] = 1
            append(args.log, entry)
            print(entry["output"])
            return 1
        entry["output"] = (
            f"accepted: {result.symbol.id}\ninterface_changed: {result.interface_changed}\n"
            f"new_signature: {result.symbol.signature}"
        )
        entry["exit_code"] = 0
        append(args.log, entry)
        print(entry["output"])
        return 0

    if args.tool == "remove":
        entry["args"] = {"target": args.target}
        try:
            result = AgentABI(args.root).remove(args.target)
        except Exception as exc:  # noqa: BLE001 - the agent sees the failure
            entry["output"] = f"rejected: {exc}"
            entry["exit_code"] = 1
            append(args.log, entry)
            print(entry["output"])
            return 1
        deps = ", ".join(result.dependents) if result.dependents else "none"
        entry["output"] = f"removed: {result.removed.id}\nremaining references from: {deps}"
        entry["exit_code"] = 0
        append(args.log, entry)
        print(entry["output"])
        return 0

    if args.tool == "suite":
        env = {"PATH": "/usr/bin:/bin:/usr/local/bin", "PYTHONPATH": str(args.root / "src")}
        proc = subprocess.run(
            [str(VENV_PY), "-m", "pytest", "-q"] + (["--no-header", "-x"] if args.quiet else []),
            cwd=args.root,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        entry["output"] = (proc.stdout + proc.stderr).rstrip()
        entry["exit_code"] = proc.returncode
        append(args.log, entry)
        print(entry["output"][-3000:])
        return proc.returncode

    if args.tool == "verify":
        entry["args"] = " ".join(args.argv)
        result = AgentABI(args.root).verify(args.argv, timeout=600)
        entry["output"] = result.tail
        entry["exit_code"] = result.exit_code
        append(args.log, entry)
        print(("PASS" if result.passed else "FAIL") + "\n" + result.tail)
        return result.exit_code

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
