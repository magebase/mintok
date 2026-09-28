"""Logging agent shim for the end-to-end benchmark.

Every agent tool call goes through this shim and is appended, one JSON object
per line, to a trajectory log: timestamp, tool name, arguments, and the full
output the agent saw. The shim is the only sanctioned interface for the
benchmark arms, so the log is an exact record of tool-side context exposure.

Arms are selected with --policy, which gates the tool surface:

  control   shell only (ordinary tooling)
  mintok    full ABI only (query/change/add/remove/verify/suite), no shell
  B         shell + read fast paths (query find|symbol)
  C         B + attribution (callers|writers|effects|summary)
  D         C + inspect (batched symbol bundle)
  E         D + semantic edits (change/add/remove/patch)
  F         E, but shell is locked until the circuit breaker trips
  G         F + packet (local planner bundle); prompt starts with packet

Circuit breaker (arms F/G): the shim refuses `shell` while the semantic
channel is under its budget; once cumulative semantic query tokens exceed
SEM_TOKEN_LIMIT, or query count exceeds QUERY_LIMIT, or two edits were
rejected, shell is unlocked and the downside vs control is bounded.

Subcommands:
  shell "CMD"            run CMD (shell) in the task root (policy-gated)
  slice DESCRIPTION      ranked exact source regions under a budget (arm S)
  query OP TARGET        ABI query (find|symbol|effects|callers|writers|summary|inspect)
  inspect TARGET         batched bundle for a known symbol
  packet DESCRIPTION     local planner: resolve + full change bundle
  read PATH [START END]  bounded raw-file region read
  change --target SID    replace one symbol; source via --source-file or --stdin
  add --target SID       create a new top-level symbol (see --module)
  remove --target SID    delete one indexed symbol
  patch --file PATH      replace a 1-based inclusive line range
  verify -- ARGV...      run a check command, get pass/fail + tail
  suite                  run the task copy's full test suite
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HARNESS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HARNESS_ROOT / "src"))

from mintok.abi import AgentABI  # noqa: E402
from mintok.continuity import LearnedState, file_digests  # noqa: E402
from mintok.escalation import (  # noqa: E402
    EscalationController,
    EscalationLevel,
    TrajectoryEvent,
    extract_traceback_target,
)
from mintok.slicer import EXPANDED_BUDGET  # noqa: E402
from mintok.slicer import INITIAL_BUDGET as INITIAL_SLICE_BUDGET  # noqa: E402
from mintok.tokens import estimate_tokens  # noqa: E402

VENV_PY = HARNESS_ROOT / ".venv" / "bin" / "python"

READ_OPS = {"find", "symbol"}
ATTRIBUTION_OPS = {"callers", "writers", "effects", "summary"}
SEMANTIC_TOOLS = {"query", "inspect", "packet"}
SEM_TOKEN_LIMIT = 1500
QUERY_LIMIT = 12
REJECT_LIMIT = 2

POLICIES: dict[str, dict] = {
    "control": {"tools": {"shell", "suite"}, "ops": set(), "breaker": False},
    "mintok": {
        "tools": SEMANTIC_TOOLS | {"change", "add", "remove", "verify", "suite"},
        "ops": READ_OPS | ATTRIBUTION_OPS | {"inspect"},
        "breaker": False,
    },
    "B": {"tools": {"shell", "read", "suite"}, "ops": READ_OPS, "breaker": False},
    "C": {
        "tools": {"shell", "read", "suite"},
        "ops": READ_OPS | ATTRIBUTION_OPS,
        "breaker": False,
    },
    "D": {
        "tools": {"shell", "read", "suite"},
        "ops": READ_OPS | ATTRIBUTION_OPS | {"inspect"},
        "breaker": False,
    },
    "E": {
        "tools": {"shell", "read", "suite"} | SEMANTIC_TOOLS | {"change", "add", "remove", "patch"},
        "ops": READ_OPS | ATTRIBUTION_OPS | {"inspect"},
        "breaker": False,
    },
    "F": {
        "tools": {"shell", "read", "suite"} | SEMANTIC_TOOLS | {"change", "add", "remove", "patch"},
        "ops": READ_OPS | ATTRIBUTION_OPS | {"inspect"},
        "breaker": True,  # shell unlocks when the breaker trips
    },
    "G": {
        "tools": {"shell", "read", "suite"} | SEMANTIC_TOOLS | {"change", "add", "remove", "patch"},
        "ops": READ_OPS | ATTRIBUTION_OPS | {"inspect"},
        "breaker": True,
    },
    # H: C + adaptive multi-query batching (turn-reduction experiment)
    "H": {
        "tools": {"shell", "read", "suite"},
        "ops": READ_OPS | ATTRIBUTION_OPS | {"batch"},
        "breaker": False,
    },
    # I: C + shell-output virtualization (raw stdout never reaches context)
    "I": {
        "tools": {"shell", "read", "suite", "result"},
        "ops": READ_OPS | ATTRIBUTION_OPS,
        "breaker": False,
        "virtual_shell": True,
    },
    # J: C + deterministic codemods (no freeform semantic edits)
    "J": {
        "tools": {"shell", "read", "suite", "codemod"},
        "ops": READ_OPS | ATTRIBUTION_OPS,
        "breaker": False,
    },
    # K: C + transparent semantic continuity (no new ops; reads of already-
    # learned facts compress to one-line verdicts, writes report the delta)
    "K": {
        "tools": {"shell", "read", "suite"},
        "ops": READ_OPS | ATTRIBUTION_OPS,
        "breaker": False,
        "continuity": True,
    },
    # S: large-module slice backend. Intelligence is local: ranked exact
    # source regions under a hard budget. Bounded raw reads are the tracked
    # fallback; edits are line-range patches over the slice's coordinates.
    "S": {
        "tools": {"slice", "read", "patch", "suite"},
        "ops": set(),
        "breaker": False,
    },
    # adaptive: MinTok 2.0 progressive escalation controller. Starts with
    # precision slice, broadens context if blind, and unlocks targeted
    # discovery and full shell fallback when stagnation is detected.
    "adaptive": {
        "tools": {
            "slice",
            "broaden",
            "trace_slice",
            "find_files",
            "grep",
            "read",
            "patch",
            "suite",
            "shell",
        },
        "ops": set(),
        "breaker": False,
        "escalation": True,
    },
}


def append(log_path: Path, entry: dict) -> None:
    entry["ts"] = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")


def breaker_state(log_path: Path) -> dict:
    """Cumulative semantic-channel usage from the trajectory so far."""
    sem_tokens = queries = rejects = 0
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            if e.get("tool") in SEMANTIC_TOOLS or (
                e.get("tool") == "query" and e.get("op") in (READ_OPS | ATTRIBUTION_OPS | {"inspect"})
            ):
                sem_tokens += estimate_tokens(e.get("output", ""))
                queries += 1
            if e.get("tool") in {"change", "add", "remove", "patch"} and str(e.get("output", "")).startswith("rejected"):
                rejects += 1
    return {"sem_tokens": sem_tokens, "queries": queries, "rejects": rejects}


def tripped(state: dict) -> bool:
    return (
        state["sem_tokens"] > SEM_TOKEN_LIMIT
        or state["queries"] > QUERY_LIMIT
        or state["rejects"] >= REJECT_LIMIT
    )


def breaker_report(state: dict) -> str:
    return (
        f"circuit breaker armed: semantic {state['sem_tokens']}/{SEM_TOKEN_LIMIT} tok, "
        f"{state['queries']}/{QUERY_LIMIT} queries, {state['rejects']}/{REJECT_LIMIT} rejected edits; "
        "shell unlocks when any limit is exceeded"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent_cli")
    parser.add_argument("--root", type=Path, required=True, help="task working copy")
    parser.add_argument("--log", type=Path, required=True, help="trajectory JSONL")
    parser.add_argument(
        "--policy",
        default="control",
        help="arm policy gating the tool surface (control|mintok|B|C|D|E|F|G|H|I|J|K|S)",
    )
    sub = parser.add_subparsers(dest="tool", required=True)

    shell = sub.add_parser("shell", help="run a shell command in the task root")
    shell.add_argument("command")

    query = sub.add_parser("query", help="mintok ABI query")
    query.add_argument("op")
    query.add_argument("target")

    inspect = sub.add_parser("inspect", help="batched bundle for one symbol")
    inspect.add_argument("target")

    packet = sub.add_parser("packet", help="local planner: resolve a description to a bundle")
    packet.add_argument("description")

    readc = sub.add_parser("read", help="bounded raw-file region read")
    readc.add_argument("path")
    readc.add_argument("start", nargs="?", type=int, default=None)
    readc.add_argument("end", nargs="?", type=int, default=None)

    def source_parser(p: argparse.ArgumentParser) -> None:
        src = p.add_mutually_exclusive_group(required=True)
        src.add_argument("--source-file")
        src.add_argument("--stdin", action="store_true")

    change = sub.add_parser("change", help="replace one symbol via the mintok ABI")
    change.add_argument("--target", required=True)
    source_parser(change)

    add = sub.add_parser("add", help="create a new top-level symbol via the mintok ABI")
    add.add_argument("--target", required=True, help="new symbol id, e.g. mintok.errors:MintokError")
    add.add_argument("--module", required=True, help="module file path relative to the root")
    add.add_argument("--imports", nargs="*", default=[], help="import lines to splice at the top")
    source_parser(add)

    remove = sub.add_parser("remove", help="delete one indexed symbol via the mintok ABI")
    remove.add_argument("--target", required=True)

    patch = sub.add_parser("patch", help="replace a line range; parse-checked escape hatch")
    patch.add_argument("--file", required=True)
    patch.add_argument("--start", type=int, required=True)
    patch.add_argument("--end", type=int, required=True)
    source_parser(patch)

    verify = sub.add_parser("verify", help="run a check command via the mintok ABI")
    verify.add_argument("argv", nargs="+")

    codemod = sub.add_parser("codemod", help="deterministic structural transform (arm J)")
    codemod.add_argument("op", choices=["rename", "add_parameter", "add_import"])
    codemod.add_argument("--target", help="symbol id (rename, add_parameter)")
    codemod.add_argument("--to", help="new identifier (rename)")
    codemod.add_argument("--name", help="parameter name (add_parameter)")
    codemod.add_argument("--default", help="parameter default (add_parameter)")
    codemod.add_argument("--file", help="file path (add_import)")
    codemod.add_argument("--statement", help="import line (add_import)")

    result = sub.add_parser("result", help="page or search virtualized shell output (arm I)")
    result.add_argument("handle", help="output handle, e.g. R3")
    result.add_argument("action", choices=["page", "find"])
    result.add_argument("spec", nargs="?", default="", help="start-end (page) or pattern (find)")

    suite = sub.add_parser("suite", help="run the task copy's full test suite")
    suite.add_argument("--quiet", action="store_true")

    slc = sub.add_parser("slice", help="ranked exact source regions for a task description (arm S)")
    slc.add_argument("description", help="what to find/change, in task words")

    broaden = sub.add_parser("broaden", help="expanded source regions with wider budget (arm adaptive)")
    broaden.add_argument("description", help="what to find/change, in task words")

    trace = sub.add_parser("trace_slice", help="slice around test failure traceback or target (arm adaptive)")
    trace.add_argument("target", nargs="?", default=None, help="optional file:line target frame")

    find_f = sub.add_parser("find_files", help="find files in repo matching pattern (arm adaptive)")
    find_f.add_argument("pattern", nargs="?", default="*", help="glob pattern, e.g. *parser*")

    grep_p = sub.add_parser("grep", help="grep pattern in repo source files (arm adaptive)")
    grep_p.add_argument("pattern", help="text or regex pattern")
    grep_p.add_argument("--path", default=None, help="subpath to search within")

    args = parser.parse_args(argv)
    policy = POLICIES.get(args.policy)
    if policy is None:
        parser.error(f"unknown policy {args.policy}")
    state_path = args.log.parent / f"{args.log.stem}.state.json"

    def load_state() -> LearnedState:
        if state_path.exists():
            return LearnedState.from_json(state_path.read_text())
        return LearnedState()

    entry: dict = {"tool": args.tool}

    def finish(output: str, exit_code: int, extra: dict | None = None) -> int:
        entry["output"] = output
        entry["exit_code"] = exit_code
        if extra:
            entry.update(extra)
        append(args.log, entry)
        print(output)
        return exit_code

    def load_controller() -> EscalationController:
        controller = EscalationController()
        if args.log.exists():
            for line in args.log.read_text().splitlines():
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

    def gated(tool: str) -> bool:
        """True when the policy allows this tool right now."""
        if policy.get("escalation"):
            ctrl = load_controller()
            return tool in ctrl.active_tools()
        if tool == "shell" and policy["breaker"]:
            return tripped(breaker_state(args.log))
        return tool in policy["tools"] or (tool == "query" and getattr(args, "op", None) in policy["ops"])

    if args.tool == "shell":
        entry["args"] = args.command
        if not gated("shell"):
            if policy.get("escalation"):
                ctrl = load_controller()
                return finish(
                    f"locked: shell is locked at Level {ctrl.level.value} ({ctrl.level.name}); "
                    "unlocks upon stagnation or repeated test failures",
                    3,
                )
            state = breaker_state(args.log)
            return finish(f"locked: {breaker_report(state)}", 3)
        if re.search(r"(holdout_solutions|tasks_holdout|\.\./)", args.command):
            return finish("locked: shell command attempts to access paths outside the task root", 3)
        if re.search(r"\bfind\s+/(?:\s|$)|(?:^|[;&|\s])cd\s+/(?:\s|;|&|$)", args.command):
            return finish("locked: commands escaping the task root to '/' are not permitted; execute relative to task root", 3)
        try:
            proc = subprocess.Popen(
                args.command,
                shell=True,
                cwd=args.root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            stdout, stderr = proc.communicate(timeout=45)
            output = (stdout + stderr).rstrip()
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            try:
                import signal
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.communicate()
            except Exception:
                pass
            return finish("error: shell command timed out after 45s", 124)
        if policy.get("virtual_shell"):
            # Virtualization: the raw stream is stored locally; the agent sees
            # a capped head plus a handle it can page or search on demand.
            out_dir = args.log.parent / "shellout"
            out_dir.mkdir(parents=True, exist_ok=True)
            handles = sorted(out_dir.glob(f"{args.log.stem}-R*.txt"))
            n = (int(handles[-1].stem.rsplit("-R", 1)[1]) + 1) if handles else 1
            full_path = out_dir / f"{args.log.stem}-R{n}.txt"
            full_path.write_text(output)
            lines = output.splitlines()
            head = "\n".join(lines[:40])
            summary = (
                f"{head}\n"
                f"[R{n}] {len(lines)} lines, exit {proc.returncode}; full: {full_path}\n"
                f"page: result R{n} page 40-80 | find: result R{n} find PATTERN"
            )
            return finish(summary, proc.returncode)
        return finish(output, proc.returncode)

    if args.tool == "result":
        entry["args"] = {"handle": args.handle, "action": args.action, "spec": args.spec}
        full_path = args.log.parent / "shellout" / f"{args.log.stem}-{args.handle}.txt"
        if not full_path.is_file():
            return finish(f"error: no stored output {args.handle}", 1)
        lines = full_path.read_text().splitlines()
        if args.action == "page":
            try:
                start_s, end_s = args.spec.split("-", 1)
                start, end = int(start_s), int(end_s)
            except ValueError:
                return finish("error: page spec is START-END", 1)
            window = lines[max(start, 1) - 1 : min(end, start + 199)]
            out = "\n".join(f"{i}  {lines[i - 1]}" for i in range(max(start, 1), max(start, 1) + len(window)))
            return finish(out if out else "(empty range)", 0)
        if not args.spec:
            return finish("error: find needs a pattern", 1)
        hits = [
            f"{i}  {line}"
            for i, line in enumerate(lines, 1)
            if args.spec.lower() in line.lower()
        ]
        body = "\n".join(hits[:60])
        note = f"\n({len(hits)} hits, showing {min(len(hits), 60)})" if len(hits) > 60 else ""
        return finish((body + note) if body else "(no hits)", 0)

    if args.tool == "codemod":
        entry["args"] = {
            k: getattr(args, k)
            for k in ("op", "target", "to", "name", "default", "file", "statement")
            if getattr(args, k) is not None
        }
        if "codemod" not in policy["tools"]:
            return finish(f"locked: tool 'codemod' not in policy {args.policy}", 3)
        try:
            abi = AgentABI(args.root)
            kw = {k: v for k, v in entry["args"].items() if k != "op"}
            result = abi.codemod(args.op, **kw)
            output = f"codemod {args.op} ok: {result}"
            if policy.get("continuity"):
                state = load_state()
                delta = state.write_delta(abi.ir, file_digests(args.root))
                if delta:
                    output += "\n" + delta
                    entry["learned_delta"] = delta
                state_path.write_text(state.to_json())
        except Exception as exc:  # noqa: BLE001 - the agent sees the failure
            return finish(f"rejected: {exc}", 1)
        return finish(output, 0)

    if args.tool == "query":
        entry["args"] = {"op": args.op, "target": args.target}
        if args.op not in policy["ops"] and "query" not in policy["tools"]:
            return finish(f"locked: query op '{args.op}' not in policy {args.policy}", 3)
        try:
            abi = AgentABI(args.root)
            output = abi.query(args.op, args.target)
            cont = None
            if policy.get("continuity"):
                state = load_state()
                digests = file_digests(args.root)
                if args.op in ("symbol", "summary"):
                    verdict = state.symbol_verdict(digests, abi.ir, args.target)
                    if verdict:
                        output, cont = verdict, "verdict"
                    elif args.target in abi.ir.symbols:
                        state.observe_symbol(abi.ir.symbols[args.target], digests)
                        cont = "full"
                elif args.op in ("callers", "writers", "effects"):
                    verdict = state.relation_verdict(args.op, args.target, digests)
                    if verdict:
                        output, cont = verdict, "verdict"
                    else:
                        count = len(output.splitlines()) if output else 0
                        state.observe_relation(args.op, args.target, digests, count)
                        cont = "full"
                state_path.write_text(state.to_json())
        except Exception as exc:  # noqa: BLE001 - the agent sees the failure
            return finish(f"error: {exc}", 1)
        return finish(output, 0, {"continuity": cont} if cont else None)

    if args.tool == "inspect":
        entry["args"] = {"target": args.target}
        try:
            output = AgentABI(args.root).inspect(args.target)
        except Exception as exc:  # noqa: BLE001 - the agent sees the failure
            return finish(f"error: {exc}", 1)
        return finish(output, 0)

    if args.tool == "packet":
        entry["args"] = {"description": args.description}
        try:
            output = AgentABI(args.root).task_packet(args.description)
        except Exception as exc:  # noqa: BLE001 - the agent sees the failure
            return finish(f"error: {exc}", 1)
        return finish(output, 0)

    if args.tool == "read":
        entry["args"] = {"path": args.path, "start": args.start, "end": args.end}
        path = (args.root / args.path).resolve()
        try:
            path.relative_to(args.root.resolve())
        except ValueError:
            return finish("locked: read escapes the task root", 3)
        if not path.is_file():
            return finish(f"error: no such file {args.path}", 1)
        lines = path.read_text().splitlines()
        start = max(args.start or 1, 1)
        end = min(args.end or start + 199, len(lines), start + 399)
        output = "\n".join(f"{i}  {lines[i - 1]}" for i in range(start, end + 1))
        return finish(output if output else "(empty range)", 0)

    abi_required = {"change", "add", "remove", "patch", "verify", "suite"}
    if args.tool in abi_required and args.tool not in policy["tools"]:
        return finish(f"locked: tool '{args.tool}' not in policy {args.policy}", 3)

    if args.tool in {"change", "add", "remove", "patch"}:
        if args.stdin:
            source = sys.stdin.read()
        else:
            source = Path(args.source_file).read_text()
        try:
            abi = AgentABI(args.root)
            if args.tool == "change":
                entry["args"] = {"target": args.target, "source_tokens": estimate_tokens(source)}
                result = abi.change(args.target, source)
                output = (
                    f"accepted: {result.symbol.id}\ninterface_changed: {result.interface_changed}\n"
                    f"new_signature: {result.symbol.signature}"
                )
            elif args.tool == "add":
                entry["args"] = {
                    "target": args.target,
                    "module": args.module,
                    "imports": len(args.imports),
                    "source_tokens": estimate_tokens(source),
                }
                result = abi.add(args.target, source, args.module, imports=args.imports)
                output = (
                    f"added: {result.symbol.id}\ncreated_file: {result.created_file}\n"
                    f"new_signature: {result.symbol.signature}"
                )
            elif args.tool == "remove":
                entry["args"] = {"target": args.target}
                result = abi.remove(args.target)
                deps = ", ".join(result.dependents) if result.dependents else "none"
                output = f"removed: {result.removed.id}\nremaining references from: {deps}"
            else:
                path = (args.root / args.file).resolve()
                try:
                    path.relative_to(args.root.resolve())
                except ValueError:
                    return finish("locked: patch escapes the task root", 3)
                entry["args"] = {
                    "file": args.file,
                    "start": args.start,
                    "end": args.end,
                    "source_tokens": estimate_tokens(source),
                }
                result = abi.patch(args.file, args.start, args.end, source)
                output = f"patched: {result.path} lines replaced: {result.replaced_lines}"
        except Exception as exc:  # noqa: BLE001 - the agent sees the failure
            return finish(f"rejected: {exc}", 1)
        if policy.get("continuity"):
            state = load_state()
            delta = state.write_delta(abi.ir, file_digests(args.root))
            if delta:
                output += "\n" + delta
                entry["learned_delta"] = delta
            state_path.write_text(state.to_json())
        return finish(output, 0)

    if args.tool == "suite":
        env = dict(os.environ)
        src_dirs = [args.root, args.root / "src"]
        env["PYTHONPATH"] = ":".join(str(p) for p in src_dirs if p.exists())
        pytest_bin = "/home/aqua/.local/bin/pytest" if Path("/home/aqua/.local/bin/pytest").exists() else "pytest"
        proc = subprocess.run(
            [pytest_bin, "-q"] + (["--no-header", "-x"] if args.quiet else []),
            cwd=args.root,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        return finish((proc.stdout + proc.stderr).rstrip(), proc.returncode)

    if args.tool == "slice":
        entry["args"] = args.description
        if "slice" not in policy["tools"]:
            return finish(f"locked: tool 'slice' not in policy {args.policy}", 3)
        try:
            from mintok.slicer import slice_task

            pkg = slice_task(args.root, args.description)
        except Exception as exc:  # noqa: BLE001 - the agent sees the failure
            return finish(f"error: {exc}", 1)
        entry["package_tokens"] = pkg.tokens
        entry["budget"] = pkg.budget
        entry["expanded"] = pkg.budget > INITIAL_SLICE_BUDGET
        return finish(pkg.text, 0)

    if args.tool == "broaden":
        entry["args"] = args.description
        if not gated("broaden"):
            return finish(f"locked: tool 'broaden' not in policy {args.policy}", 3)
        try:
            from mintok.slicer import slice_task

            pkg = slice_task(args.root, args.description, initial_budget=EXPANDED_BUDGET, expanded_budget=3000)
        except Exception as exc:  # noqa: BLE001 - the agent sees the failure
            return finish(f"error: {exc}", 1)
        entry["package_tokens"] = pkg.tokens
        entry["budget"] = pkg.budget
        entry["expanded"] = True
        return finish(pkg.text, 0)

    if args.tool == "trace_slice":
        entry["args"] = args.target
        if not gated("trace_slice"):
            return finish(f"locked: tool 'trace_slice' not in policy {args.policy}", 3)
        target = args.target
        if not target and args.log.exists():
            for line in reversed(args.log.read_text().splitlines()):
                if not line.strip():
                    continue
                try:
                    e = json.loads(line)
                    if e.get("tool") in ("suite", "verify") and e.get("exit_code", 0) != 0:
                        frame = extract_traceback_target(str(e.get("output", "")))
                        if frame:
                            target = f"{frame[0]}:{frame[1]}"
                            break
                except Exception:
                    continue
        if not target:
            return finish("error: no traceback target specified and no failing test found in log", 1)
        parts = target.split(":", 1)
        rel_path = parts[0]
        lineno = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
        target_file = (args.root / rel_path).resolve()
        try:
            target_file.relative_to(args.root.resolve())
        except ValueError:
            return finish("locked: trace_slice target escapes task root", 3)
        if not target_file.is_file():
            return finish(f"error: target file not found {rel_path}", 1)
        lines = target_file.read_text().splitlines()
        start = max(1, lineno - 30)
        end = min(len(lines), lineno + 30)
        excerpts = "\n".join(f"{i}  {lines[i - 1]}" for i in range(start, end + 1))
        output = f"trace target: {rel_path}:{start}-{end} (failing line {lineno})\n--- excerpts ---\n{excerpts}"
        return finish(output, 0)

    if args.tool == "find_files":
        entry["args"] = args.pattern
        if not gated("find_files"):
            return finish(f"locked: tool 'find_files' not in policy {args.policy}", 3)
        import fnmatch

        pat = args.pattern or "*"
        matches = []
        ignored = {".git", "__pycache__", ".venv", ".tox", "build", "dist", ".pytest_cache", ".eggs"}
        for p in sorted(args.root.rglob("*")):
            if any(part in ignored for part in p.parts):
                continue
            if p.is_file():
                rel = str(p.relative_to(args.root))
                if fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(p.name, pat):
                    matches.append(rel)
                    if len(matches) >= 50:
                        break
        if not matches:
            return finish(f"no files found matching '{pat}'", 0)
        return finish("\n".join(matches[:50]), 0)

    if args.tool == "grep":
        entry["args"] = {"pattern": args.pattern, "path": args.path}
        if not gated("grep"):
            return finish(f"locked: tool 'grep' not in policy {args.policy}", 3)
        search_root = (args.root / args.path).resolve() if args.path else args.root.resolve()
        try:
            search_root.relative_to(args.root.resolve())
        except ValueError:
            return finish("locked: grep path escapes the task root", 3)
        hits = []
        ignored = {".git", "__pycache__", ".venv", ".tox", "build", "dist", ".pytest_cache", ".eggs"}
        pat_re = re.compile(args.pattern, re.IGNORECASE)
        candidates = sorted(search_root.rglob("*.py")) if search_root.is_dir() else [search_root]
        for p in candidates:
            if any(part in ignored for part in p.parts):
                continue
            if p.is_file():
                try:
                    rel = str(p.relative_to(args.root))
                    for idx, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
                        if pat_re.search(line):
                            hits.append(f"{rel}:{idx}: {line.strip()[:120]}")
                            if len(hits) >= 40:
                                break
                except Exception:
                    continue
            if len(hits) >= 40:
                break
        if not hits:
            return finish(f"no hits for '{args.pattern}'", 0)
        return finish("\n".join(hits[:40]), 0)

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
