"""End-to-end benchmark runner: prepare copies, score runs, emit records.

The frontier agent itself is the outer coding agent (this harness cannot spawn
or bill frontier models); arms are executed as isolated subagent sessions over
task copies, with every tool call flowing through agent_cli.py so tool-side
context exposure is measured exactly from the trajectory logs.

Subcommands:
  prepare --dest DIR                 export the current commit into DIR (clean tree)
  check --root COPY --task ID        run suite + task checker; write score JSON next to log
  record --arm ARM --task ID         fold one trajectory log + score into runs/<arm>.jsonl
  report                             aggregate runs/ into the proxy efficiency report
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HARNESS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HARNESS_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mintok.metrics import RunRecord  # noqa: E402
from mintok.tokens import estimate_tokens  # noqa: E402
from tasks import TASKS  # noqa: E402

VENV_PY = HARNESS_ROOT / ".venv" / "bin" / "python"
RUNS_DIR = Path(__file__).resolve().parent / "runs"
PROMPT_OVERHEAD_TOKENS = 250  # arm rules + JSON report instructions, identical for both arms


def task_by_id(task_id: str) -> dict:
    for task in TASKS:
        if task["id"] == task_id:
            return task
    raise SystemExit(f"unknown task {task_id}")


def prepare(dest: Path) -> None:
    if dest.exists():
        subprocess.run(["rm", "-rf", str(dest)], check=True)
    dest.mkdir(parents=True)
    archive = subprocess.run(["git", "archive", "HEAD"], cwd=HARNESS_ROOT, capture_output=True, check=True)
    subprocess.run(["tar", "-x", "-C", str(dest)], input=archive.stdout, check=True)
    print(dest)


def check(root: Path, task_id: str, log: Path) -> None:
    task = task_by_id(task_id)
    env_out = subprocess.run(
        [str(VENV_PY), "-m", "pytest", "-q"],
        cwd=root,
        env={"PATH": "/usr/bin:/bin:/usr/local/bin", "PYTHONPATH": str(root / "src")},
        capture_output=True,
        text=True,
        timeout=600,
    )
    suite_ok = env_out.returncode == 0

    # Checkers are plain Python written against the task copy: they import
    # mintok.*, may reference ``root``, and may spawn subprocesses. Run them
    # with the copy's src first on sys.path (in-process and inherited), and
    # with any main-repo mintok modules purged from the import cache.
    scope: dict = {"root": str(root)}
    saved_modules = {
        k: v for k, v in sys.modules.items() if k == "mintok" or k.startswith("mintok.")
    }
    for k in saved_modules:
        del sys.modules[k]
    old_cwd = os.getcwd()
    old_pp = os.environ.get("PYTHONPATH")
    os.chdir(root)
    os.environ["PYTHONPATH"] = str(root / "src")
    sys.path.insert(0, str(root / "src"))
    try:
        code = compile(task["check"], f"<check:{task_id}>", "exec")
        exec(code, scope)  # noqa: S102 - benchmark checkers are first-party
        check_ok, detail = scope.get("ok", False), scope.get("detail", "")
    except (Exception, SystemExit) as exc:  # noqa: BLE001 - reported, not raised
        check_ok, detail = False, f"{type(exc).__name__}: {exc}"
    finally:
        os.chdir(old_cwd)
        sys.path.remove(str(root / "src"))
        if old_pp is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = old_pp
        for k in [k for k in sys.modules if k == "mintok" or k.startswith("mintok.")]:
            del sys.modules[k]
        sys.modules.update(saved_modules)

    entries = [json.loads(line) for line in log.read_text().splitlines() if line.strip()] if log.exists() else []
    tool_output_tokens = sum(estimate_tokens(e.get("output", "")) for e in entries)
    tool_input_tokens = sum(estimate_tokens(json.dumps(e.get("args", ""))) for e in entries)
    latency = (entries[-1]["ts"] - entries[0]["ts"]) if len(entries) > 1 else 0.0
    score = {
        "task_id": task_id,
        "suite_ok": suite_ok,
        "check_ok": bool(check_ok),
        "detail": detail,
        "solved": suite_ok and bool(check_ok),
        "turns": len(entries),
        "tool_output_tokens": tool_output_tokens,
        "tool_input_tokens": tool_input_tokens,
        "prompt_tokens": estimate_tokens(task["instruction"]) + PROMPT_OVERHEAD_TOKENS,
        "latency_s": round(latency, 1),
        "calls": [f"{e['tool']}:{json.dumps(e.get('args', ''))[:60]}" for e in entries],
    }
    out = log.parent / f"{log.stem}.score.json"
    out.write_text(json.dumps(score, indent=2))
    print(json.dumps({k: score[k] for k in ("task_id", "suite_ok", "check_ok", "detail", "turns")}))


def record(arm: str, task_id: str, log: Path) -> None:
    score = json.loads((log.parent / f"{log.stem}.score.json").read_text())
    run = RunRecord(
        task_id=task_id,
        arm=arm,
        solved=score["solved"],
        frontier_usd=0.0,  # billed cost unobservable outside an API harness (see RESULTS.md)
        input_tokens=score["tool_output_tokens"] + score["prompt_tokens"],
        output_tokens=score["tool_input_tokens"],
        turns=score["turns"],
        latency_s=score["latency_s"],
    )
    RUNS_DIR.mkdir(exist_ok=True)
    with (RUNS_DIR / f"{arm}.jsonl").open("a") as fh:
        fh.write(json.dumps(run_record_dict(run)) + "\n")
    print(json.dumps(run_record_dict(run)))


def run_record_dict(run: RunRecord) -> dict:
    return {
        "task_id": run.task_id,
        "arm": run.arm,
        "solved": run.solved,
        "frontier_usd": run.frontier_usd,
        "input_tokens": run.input_tokens,
        "output_tokens": run.output_tokens,
        "turns": run.turns,
        "latency_s": run.latency_s,
    }


def report() -> None:
    arms: dict[str, list[dict]] = {}
    for path in sorted(RUNS_DIR.glob("*.jsonl")):
        arm = path.stem
        arms[arm] = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    header = f"{'arm':<10}{'tasks':>7}{'solved':>8}{'solve%':>8}{'turns':>7}{'tool_out_tok':>14}{'in_tok/solved':>15}{'s/Mtok':>8}"
    lines = [header, "-" * len(header)]
    for arm, rows in arms.items():
        solved = sum(r["solved"] for r in rows)
        out_tok = sum(r["input_tokens"] for r in rows)  # RunRecord.input_tokens = tool outputs + prompt
        per_solved = out_tok / solved if solved else float("inf")
        solve_m = (solved / out_tok * 1_000_000) if out_tok else 0.0
        lines.append(
            f"{arm:<10}{len(rows):>7}{solved:>8}{100 * solved / len(rows):>7.0f}%"
            f"{sum(r['turns'] for r in rows):>7}{out_tok:>14}{per_solved:>15.0f}{solve_m:>8.2f}"
        )
    print("\n".join(lines))

    print("\nper-task paired view (control vs mintok tool-context tokens):")
    by_task: dict[str, dict[str, dict]] = {}
    for arm, rows in arms.items():
        for r in rows:
            by_task.setdefault(r["task_id"], {})[arm] = r
    for task_id in sorted(by_task):
        pair = by_task[task_id]
        c = pair.get("control", {})
        m = pair.get("mintok", {})
        ct, mt = c.get("input_tokens", 0), m.get("input_tokens", 0)
        ratio = f"{ct / mt:>5.1f}x" if mt else "   -- "
        mark = lambda a: "OK " if a.get("solved") else ("NO " if a else "-- ")  # noqa: E731
        print(f"{task_id:<32}{mark(c)}{mark(m)}  {ct:>6} {mt:>6} {ratio}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="e2e")
    sub = parser.add_subparsers(dest="cmd", required=True)

    prep = sub.add_parser("prepare")
    prep.add_argument("--dest", type=Path, required=True)

    chk = sub.add_parser("check")
    chk.add_argument("--root", type=Path, required=True)
    chk.add_argument("--task", required=True)
    chk.add_argument("--log", type=Path, required=True)

    rec = sub.add_parser("record")
    rec.add_argument("--arm", required=True)
    rec.add_argument("--task", required=True)
    rec.add_argument("--log", type=Path, required=True)

    sub.add_parser("report")

    args = parser.parse_args()
    if args.cmd == "prepare":
        prepare(args.dest)
    elif args.cmd == "check":
        check(args.root, args.task, args.log)
    elif args.cmd == "record":
        record(args.arm, args.task, args.log)
    else:
        report()


if __name__ == "__main__":
    main()
