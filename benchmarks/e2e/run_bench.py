"""End-to-end benchmark runner: prepare copies, score runs, emit records.

The frontier agent itself is the outer coding agent (this harness cannot spawn
or bill frontier models); arms are executed as isolated subagent sessions over
task copies, with every tool call flowing through agent_cli.py so tool-side
context exposure is measured exactly from the trajectory logs.

Subcommands:
  prepare --dest DIR [--task ID]     export the task's repo into DIR (clean tree;
                                     mintok tasks use the git archive, fixture
                                     tasks copy benchmarks/e2e/fixtures/<repo>)
  check --root COPY --task ID        run suite + task checker; write score JSON next to log
  record --arm ARM --task ID         fold one trajectory log + score into runs/<arm>.jsonl
  report                             aggregate runs/ into the proxy efficiency report
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
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
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
GENERATED_TASKS = Path(__file__).resolve().parent / "tasks_generated.json"
PROMPT_OVERHEAD_TOKENS = 250  # arm rules + JSON report instructions, identical for both arms


def all_tasks() -> list[dict]:
    """The 30 dev-set tasks plus the generated stratified eval tasks."""
    tasks = list(TASKS)
    if GENERATED_TASKS.exists():
        tasks += json.loads(GENERATED_TASKS.read_text())
    return tasks


def task_by_id(task_id: str) -> dict:
    for task in all_tasks():
        if task["id"] == task_id:
            return task
    raise SystemExit(f"unknown task {task_id}")


def prepare(dest: Path, task_id: str | None = None) -> None:
    if dest.exists():
        subprocess.run(["rm", "-rf", str(dest)], check=True)
    dest.mkdir(parents=True)
    repo = task_by_id(task_id).get("repo", "mintok") if task_id else "mintok"
    if repo == "mintok":
        archive = subprocess.run(["git", "archive", "HEAD"], cwd=HARNESS_ROOT, capture_output=True, check=True)
        subprocess.run(["tar", "-x", "-C", str(dest)], input=archive.stdout, check=True)
    else:
        shutil.copytree(FIXTURES_DIR / repo, dest, dirs_exist_ok=True)
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

    header = (
        f"{'arm':<10}{'tasks':>7}{'solved':>8}{'solve%':>8}"
        f"{'tn/att':>8}{'tn/solv':>9}{'tok/att':>9}{'tok/solv':>10}{'s/Mtok':>8}"
    )
    print(header)
    print("-" * len(header))
    for arm, rows in arms.items():
        n = len(rows)
        solved = sum(r["solved"] for r in rows)
        out_tok = sum(r["input_tokens"] for r in rows)  # tool outputs + prompt
        turns = sum(r["turns"] for r in rows)
        print(
            f"{arm:<10}{n:>7}{solved:>8}{100 * solved / n:>7.0f}%"
            f"{turns / n:>8.1f}{(turns / solved) if solved else float('inf'):>9.1f}"
            f"{out_tok / n:>9.0f}{(out_tok / solved) if solved else float('inf'):>10.0f}"
            f"{(solved / out_tok * 1_000_000) if out_tok else 0.0:>8.2f}"
        )

    print("\npaired ratios on jointly solved tasks (control tokens / other-arm tokens):")
    by_task: dict[str, dict[str, dict]] = {}
    for arm, rows in arms.items():
        for r in rows:
            by_task.setdefault(r["task_id"], {})[arm] = r
    base = "control" if "control" in arms else next(iter(arms))
    for arm in sorted(arms):
        if arm == base:
            continue
        ratios = []
        for task_id, pair in sorted(by_task.items()):
            c, m = pair.get(base), pair.get(arm)
            if c and m and c["solved"] and m["solved"] and m["input_tokens"]:
                ratios.append(c["input_tokens"] / m["input_tokens"])
        if not ratios:
            print(f"  {base} vs {arm}: no jointly solved tasks")
            continue
        geo = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
        wins = sum(r > 1 for r in ratios)
        print(
            f"  {base} vs {arm:<10} n={len(ratios):>2}  geomean {geo:>5.2f}x  "
            f"median {sorted(ratios)[len(ratios) // 2]:>5.2f}x  "
            f"min {min(ratios):>5.2f}x  max {max(ratios):>6.2f}x  wins {wins}/{len(ratios)}"
        )

    print("\nper-task paired view (control vs each arm, tool-context tokens):")
    for task_id in sorted(by_task):
        pair = by_task[task_id]
        cells = []
        for arm in [base] + sorted(a for a in arms if a != base):
            r = pair.get(arm)
            if not r:
                cells.append(f"{arm[:4]}:  --  ")
            else:
                mark = "OK" if r["solved"] else "NO"
                cells.append(f"{arm[:4]}:{mark}{r['input_tokens']:>6}")
        print(f"{task_id:<32}" + "  ".join(cells))


def main() -> None:
    parser = argparse.ArgumentParser(prog="e2e")
    sub = parser.add_subparsers(dest="cmd", required=True)

    prep = sub.add_parser("prepare")
    prep.add_argument("--dest", type=Path, required=True)
    prep.add_argument("--task", help="task id; its repo field selects the source tree")

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
        prepare(args.dest, args.task)
    elif args.cmd == "check":
        check(args.root, args.task, args.log)
    elif args.cmd == "record":
        record(args.arm, args.task, args.log)
    else:
        report()


if __name__ == "__main__":
    main()
