"""End-to-end benchmark runner: prepare copies, score runs, emit records.

The frontier agent itself is the outer coding agent (this harness cannot spawn
or bill frontier models); arms are executed as isolated subagent sessions over
task copies, with every tool call flowing through agent_cli.py so tool-side
context exposure is measured exactly from the trajectory logs.

Subcommands:
  prepare --dest DIR [--task ID]     export the task's repo into DIR (clean tree;
                                     mintok tasks use the git archive, fixture
                                     tasks CoW/copy benchmarks/e2e/fixtures/<repo>)
  check --root COPY --task ID        run suite + task checker; write score JSON next to log
  record --arm ARM --task ID         fold one trajectory log + score into runs/<arm>.jsonl
  report                             aggregate runs/ into the proxy efficiency report
  fingerprint --task ID              emit the immutable task fingerprint (prompt/checker/fixtures)
  verify --task ID --fingerprint F   refuse to run a drifted task (non-zero exit on mismatch)
  fast8                              print the FAST-8 smoke suite (id and behavior class)
  replay --logs DIR [--out FILE]     recompute per-task metrics from raw trajectory logs
                                     with zero agent runs
  oracle --arm A --arm B             offline oracle router over two arms' run records:
                                     per-task cheaper-solver selection, upper bound on
                                     routing headroom (zero agent runs)
  route --task ID                    deterministic pre-flight routing decision for one task
  route --eval                       route every generated task from pre-flight features
                                     only (wording + target LOC), log predictions with
                                     actual outcomes to runs/router-predictions.jsonl,
                                     and score the routed system against uniform arms
                                     (zero agent runs; slicer routes fall back to control)
  slice --task ID                    render the deterministic slice package for one task
  slice --eval                       sizing check: slice every large-module task from
                                     pre-flight info and report package tokens vs the
                                     arms' actual tool-context (sizing only — not a
                                     solve prediction; the oracle waits for real runs)
  promote-large                      ISOLATED live test on the 15 large-module tasks:
                                     slicer arm (policy S) vs the frozen control-eval
                                     trajectories, scored against the pre-registered
                                     promotion bar. Requires MINTOK_MODEL_API_KEY.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HARNESS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HARNESS_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mintok.funnel import FAST8, task_fingerprint, verify_fingerprint, TaskIntegrityError  # noqa: E402
from mintok.metrics import RunRecord  # noqa: E402
from mintok.tokens import estimate_tokens  # noqa: E402
from tasks import TASKS  # noqa: E402

VENV_PY = HARNESS_ROOT / ".venv" / "bin" / "python"
RUNS_DIR = Path(__file__).resolve().parent / "runs"
E2E_DIR = Path(__file__).resolve().parent
FIXTURES = E2E_DIR / "fixtures"
TASKS_JSON = E2E_DIR / "tasks_generated.json"
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
GENERATED_TASKS = Path(__file__).resolve().parent / "tasks_generated.json"
HOLDOUT_TASKS = Path(__file__).resolve().parent / "tasks_holdout.json"
PROMPT_OVERHEAD_TOKENS = 250  # arm rules + JSON report instructions, identical for both arms


def all_tasks() -> list[dict]:
    """The 30 dev-set tasks plus the generated stratified eval tasks and holdout tasks."""
    tasks = list(TASKS)
    if GENERATED_TASKS.exists():
        tasks += json.loads(GENERATED_TASKS.read_text())
    if HOLDOUT_TASKS.exists():
        tasks += json.loads(HOLDOUT_TASKS.read_text())
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
        source = FIXTURES_DIR / repo
        # Copy-on-write first (reflink on btrfs/XFS/NFS 4.2, near-instant);
        # plain deep copy where the filesystem cannot CoW. Never hardlink:
        # agents edit files in place and would mutate the pristine fixture.
        reflink = subprocess.run(
            ["cp", "-a", "--reflink=auto", str(source) + "/.", str(dest)],
            capture_output=True,
        )
        if reflink.returncode != 0:
            shutil.copytree(source, dest, dirs_exist_ok=True)
    # Automated Isolation Assertion: agent-visible filesystem must not contain
    # solution files, reference fixes, checker internals, or holdout metadata.
    forbidden = ["holdout_solutions.json", "tasks_holdout.json", "holdout_fingerprints.json"]
    for name in forbidden:
        assert not (dest / name).exists(), f"Isolation breach: {name} found in workspace {dest}"
        assert not any(dest.rglob(name)), f"Isolation breach: {name} found nested in workspace {dest}"
    print(dest)



def check_task(root: Path, task_id: str, checker_src: str, log: Path) -> dict:
    """Suite + checker against one task copy; checker in a fresh subprocess.

    The checker MUST NOT run in-process: a checker that imports fixture
    modules (``biglib`` etc.) poisons ``sys.modules`` for every later
    check, silently validating task 1's repo state forever after. A
    subprocess makes cross-task cache pollution impossible.
    """
    env_out = subprocess.run(
        [str(VENV_PY), "-m", "pytest", "-q"],
        cwd=root,
        env={"PATH": "/usr/bin:/bin:/usr/local/bin", "PYTHONPATH": str(root / "src")},
        capture_output=True,
        text=True,
        timeout=600,
    )
    suite_ok = env_out.returncode == 0

    check_env = dict(os.environ)
    check_env["PYTHONPATH"] = str(root / "src")
    check_out = subprocess.run(
        [sys.executable, "-c", f"root = {str(root)!r}\n" + checker_src],
        cwd=root,
        env=check_env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    check_ok = check_out.returncode == 0
    err_text = (check_out.stderr or check_out.stdout).strip()
    detail = "" if check_ok else (err_text.splitlines()[-1] if err_text else "checker failed")

    entries = [json.loads(line) for line in log.read_text().splitlines() if line.strip()] if log.exists() else []
    task = task_by_id(task_id) if task_id in {t["id"] for t in json.loads(TASKS_JSON.read_text())} else {"instruction": ""}
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
    return score


def check(root: Path, task_id: str, log: Path) -> None:
    """Score one task copy with its frozen checker (subprocess-isolated)."""
    check_task(root, task_id, task_by_id(task_id)["check"], log)


def record(arm: str, task_id: str, log: Path) -> None:
    score = json.loads((log.parent / f"{log.stem}.score.json").read_text())
    usage_path = log.parent / f"{log.stem}.usage.json"
    usage = json.loads(usage_path.read_text()) if usage_path.exists() else None
    run = RunRecord(
        task_id=task_id,
        arm=arm,
        solved=score["solved"],
        frontier_usd=usage["usd"] if usage else 0.0,  # billed via provider telemetry when present
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

    print("\ntail cost per arm (tool-context tokens per attempted task):")
    print(f"{'arm':<10}{'median':>8}{'p90':>8}{'p95':>8}{'max':>8}")
    for arm, rows in sorted(arms.items()):
        toks = sorted(r["input_tokens"] for r in rows)
        if not toks:
            continue

        def pct(p: float) -> int:
            return toks[min(len(toks) - 1, math.ceil(p * len(toks)) - 1)]

        print(
            f"{arm:<10}{toks[len(toks) // 2]:>8}"
            f"{pct(0.90):>8}{pct(0.95):>8}{max(toks):>8}"
        )

    print("\nper-stratum solve rate and tokens (generated eval tasks only):")
    klass_of = {t["id"]: t.get("klass", "mintok_dev") for t in all_tasks()}
    strata: dict[str, dict[str, list[dict]]] = {}
    for arm, rows in arms.items():
        for r in rows:
            k = klass_of.get(r["task_id"])
            if k is None:
                continue  # dev-set tasks have no stratum
            strata.setdefault(k, {}).setdefault(arm, []).append(r)
    if strata:
        skel = f"{'stratum':<30}{'arm':<10}{'n':>4}{'solved':>8}{'tok/solv':>10}"
        print(skel)
        print("-" * len(skel))
        for k in sorted(strata):
            for arm, rows in sorted(strata[k].items()):
                solved = sum(r["solved"] for r in rows)
                tok = sum(r["input_tokens"] for r in rows)
                print(
                    f"{k:<30}{arm:<10}{len(rows):>4}{solved:>8}"
                    f"{(tok / solved) if solved else float('inf'):>10.0f}"
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


def fixture_root_for(task_id: str) -> Path | None:
    repo = task_by_id(task_id).get("repo", "mintok")
    return FIXTURES_DIR / repo if repo != "mintok" else None


def fingerprint(task_id: str) -> None:
    print(json.dumps(task_fingerprint(task_by_id(task_id), fixture_root_for(task_id)), indent=2))


def verify(task_id: str, fingerprint_file: Path) -> None:
    stored = json.loads(fingerprint_file.read_text())
    try:
        verify_fingerprint(task_by_id(task_id), stored, fixture_root_for(task_id))
    except TaskIntegrityError as exc:
        print(f"REFUSED: {exc}")
        raise SystemExit(2)
    print(f"ok: {task_id} matches its fingerprint")


def fast8() -> None:
    for task_id, klass in FAST8:
        print(f"{task_id:<32}{klass}")


def replay(logs_dir: Path, out: Path | None) -> None:
    """Recompute per-task trajectory metrics from raw shim logs. Zero runs."""
    rows = []
    for log in sorted(Path(logs_dir).glob("*.jsonl")):
        entries = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
        if not entries:
            continue
        score_path = log.parent / f"{log.stem}.score.json"
        solved = json.loads(score_path.read_text())["solved"] if score_path.exists() else None
        rows.append(
            {
                "task": log.stem,
                "turns": len(entries),
                "tool_output_tokens": sum(estimate_tokens(e.get("output", "")) for e in entries),
                "tool_input_tokens": sum(estimate_tokens(json.dumps(e.get("args", ""))) for e in entries),
                "latency_s": round((entries[-1]["ts"] - entries[0]["ts"]) if len(entries) > 1 else 0.0, 1),
                "solved": solved,
            }
        )
    text = json.dumps(rows, indent=2) if out is None else None
    if out is not None:
        out.write_text(json.dumps(rows, indent=2))
        print(f"{len(rows)} trajectories -> {out}")
    else:
        print(text)


def oracle(arms: list[str]) -> None:
    """Offline oracle router: per-task best-arm selection upper bound."""
    from mintok.metrics import oracle_router

    assert len(arms) == 2, "exactly two arms"
    records = []
    for arm_file in arms:
        path = RUNS_DIR / f"{arm_file}.jsonl"
        records += [
            RunRecord(
                task_id=r["task_id"],
                arm=r["arm"],
                solved=r["solved"],
                frontier_usd=r.get("frontier_usd", 0.0),
                input_tokens=r.get("input_tokens", 0),
                output_tokens=r.get("output_tokens", 0),
                turns=r.get("turns", 0),
            )
            for r in (json.loads(line) for line in path.read_text().splitlines() if line.strip())
        ]
    a, b = arms
    rep = oracle_router(records, a, b)
    uniform_a = rep.uniform(records, a)
    uniform_b = rep.uniform(records, b)
    base, other = (uniform_a, uniform_b) if uniform_a >= uniform_b else (uniform_b, uniform_a)

    def pct_of(x: float, y: float) -> float:
        return (1 - y / x) * 100 if x else 0.0

    print(f"oracle routing over {rep.tasks} tasks ({a} vs {b}):")
    print(f"  {'uniform':<10}{'solved':>8}{'tok/attempt':>13}{'vs best-arm':>13}")
    print(f"  {a:<10}{sum(r.solved for r in records if r.arm == a):>8}{uniform_a:>13.0f}")
    print(f"  {b:<10}{sum(r.solved for r in records if r.arm == b):>8}{uniform_b:>13.0f}")
    print(f"  {'oracle':<10}{rep.solved:>8}{rep.tokens_per_attempt:>13.0f}")
    print(f"\nheadroom vs cheaper uniform arm: {base / other:.2f}x uniform, "
          f"{base / rep.tokens_per_attempt:.2f}x routed")
    print(f"routed: {rep.tokens_per_solved:.0f} tok/solved, p95 {rep.p95}, max {rep.max_tokens}, "
          f"{rep.solves_per_mtok:.1f} solves/Mtok")
    chosen = {}
    for arm in (a, b):
        chosen[arm] = sum(1 for v in rep.routed_to.values() if v == arm)
    print(f"routes: " + ", ".join(f"{arm}: {n}" for arm, n in chosen.items()))


def _task_sizes(repo: str, instruction: str) -> dict[str, int]:
    """LOC of .py paths named in the instruction, from the pristine fixture."""
    import re as _re

    root = FIXTURES / repo
    sizes = {}
    for path in set(_re.findall(r"[\w./-]+\.py", instruction)):
        f = root / path
        if f.is_file():
            sizes[path] = sum(1 for _ in f.open(errors="replace"))
    return sizes


def route_cmd(task_id: str | None, eval_all: bool) -> None:
    """Deterministic pre-flight router: decide, log, and score offline."""
    from mintok.router import BACKEND_CONTROL, BACKEND_SLICER, PreFlightFeatures, prediction_record, route

    tasks = {t["id"]: t for t in json.loads(TASKS_JSON.read_text())}
    if task_id is not None:
        task = tasks[task_id]
        feats = PreFlightFeatures(task["instruction"], _task_sizes(task["repo"], task["instruction"]))
        dec = route(feats)
        print(f"{task_id}: backend={dec.backend} class={dec.predicted_class} "
              f"expected_cost={dec.expected_relative_cost:.2f} confidence={dec.confidence:.2f}")
        for reason in dec.reasons:
            print(f"  - {reason}")
        return

    arms = {}
    for arm in ("control-eval", "C-eval"):
        arms[arm] = {json.loads(l)["task_id"]: json.loads(l)
                     for l in (RUNS_DIR / f"{arm}.jsonl").read_text().splitlines() if l.strip()}

    # Ideal backend per true stratum, from the frozen-eval per-stratum table
    # (api/lookup/feature/schema -> C; cross/refactor -> control; large -> slicer).
    IDEAL_SEMANTIC = {"api_signature_propagation", "simple_lookup", "feature_addition", "schema_or_framework_change"}

    records, routed_solved, routed_tokens, routed_max = [], 0, 0, 0
    klass_stats, backend_correct = {}, [0, 0]
    for task_id, task in sorted(tasks.items()):
        feats = PreFlightFeatures(task["instruction"], _task_sizes(task["repo"], task["instruction"]))
        dec = route(feats)
        backend = BACKEND_CONTROL if dec.backend == BACKEND_SLICER else dec.backend
        actual_arm = arms["C-eval" if backend == "semantic-C" else "control-eval"][task_id]
        actual = {
            arm_: {
                "tokens": arms[f"{a}-eval"][task_id]["input_tokens"],
                "solved": arms[f"{a}-eval"][task_id]["solved"],
            }
            for arm_, a in (("control", "control"), ("semantic-C", "C"))
        }
        rec = prediction_record(task_id, feats, dec, actual=actual)
        rec["actual"] = actual
        records.append(rec)

        true_klass = task["klass"]
        stats = klass_stats.setdefault(true_klass, [0, 0])
        stats[1] += 1
        if dec.predicted_class == true_klass:
            stats[0] += 1
        ideal = "semantic-C" if true_klass in IDEAL_SEMANTIC else BACKEND_CONTROL
        backend_correct[1] += 1
        if backend == ideal:
            backend_correct[0] += 1
        routed_solved += actual_arm["solved"]
        routed_tokens += actual_arm["input_tokens"]
        routed_max = max(routed_max, actual_arm["input_tokens"])

    out = RUNS_DIR / "router-predictions.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in records))

    n = len(records)
    print(f"pre-flight router over {n} tasks (wording + target LOC only):")
    print(f"  exact-class accuracy: {sum(s[0] for s in klass_stats.values())}/{n}; "
          f"backend accuracy: {backend_correct[0]}/{n}")
    for klass, (ok, tot) in sorted(klass_stats.items()):
        print(f"    {klass:32s} {ok:3d}/{tot}")
    print(f"  routes: " + ", ".join(
        f"{b}: {sum(1 for r in records if r['backend'] == b)}"
        for b in (BACKEND_CONTROL, "semantic-C", BACKEND_SLICER)))
    print(f"  routed system (slicer->control): solved {routed_solved}/{n}, "
          f"tok/attempt {routed_tokens / n:.0f}, max {routed_max}")
    ctrl = sum(r["actual"]["control"]["tokens"] for r in records)
    sem = sum(r["actual"]["semantic-C"]["tokens"] for r in records)
    print(f"  vs uniform: control {ctrl / n:.0f} ({ctrl / routed_tokens:.2f}x routed), "
          f"C {sem / n:.0f} ({sem / routed_tokens:.2f}x routed)")
    print(f"  predictions logged: {out}")


def slice_cmd(task_id: str | None, eval_all: bool) -> None:
    """Deterministic slice backend: render packages and size them offline."""
    from mintok.slicer import slice_task

    tasks = {t["id"]: t for t in json.loads(TASKS_JSON.read_text())}
    if task_id is not None:
        task = tasks[task_id]
        pkg = slice_task(FIXTURES / task["repo"], task["instruction"])
        print(pkg.text)
        print(f"[package {pkg.tokens} tokens, budget {pkg.budget}"
              f"{', TRUNCATED' if pkg.truncated else ''}]")
        return

    tasks = {t["id"]: t for t in json.loads(TASKS_JSON.read_text()) if t["klass"] == "large_file_navigation"}
    arms = {}
    for arm in ("control-eval", "C-eval"):
        arms[arm] = {json.loads(l)["task_id"]: json.loads(l)
                     for l in (RUNS_DIR / f"{arm}.jsonl").read_text().splitlines() if l.strip()}
    rows = []
    for task_id, task in sorted(tasks.items()):
        pkg = slice_task(FIXTURES / task["repo"], task["instruction"])
        rows.append((task_id, pkg.tokens, pkg.truncated,
                     arms["control-eval"][task_id]["input_tokens"],
                     arms["C-eval"][task_id]["input_tokens"],
                     arms["control-eval"][task_id]["solved"],
                     arms["C-eval"][task_id]["solved"]))
    print(f"slice sizing over {len(rows)} large-module tasks (deterministic, zero runs):")
    print(f"  {'task':22s}{'pkg_tok':>8}{'trunc':>7}{'ctrl_tok':>9}{'C_tok':>7}{'ctrl_ok':>8}{'C_ok':>6}")
    for task_id, ptok, trunc, ctok, mtok, cok, mok in rows:
        print(f"  {task_id:22s}{ptok:>8}{('Y' if trunc else 'n'):>7}{ctok:>9}{mtok:>7}{str(cok):>8}{str(mok):>6}")
    avg = sum(r[1] for r in rows) / len(rows)
    ctrl = sum(r[3] for r in rows) / len(rows)
    print(f"  avg package {avg:.0f} tok vs control avg {ctrl:.0f} tok of tool-context "
          f"({ctrl / avg:.1f}x headroom); pre-registered bar: <=800 tok/solved good, "
          "<=600 strong, <=400 excellent at control's solve rate")


def promote_large(model: str, mock: bool = False, provider: str = "anthropic", holdout: bool = False) -> None:
    """Isolated live test: slicer arm vs control on large tasks.

    Manifest guard: the frozen control-eval trajectories are reused ONLY
    when their recorded manifest matches the live slicer configuration
    (provider, model, reasoning effort, system prompt, toolset, harness
    version, task set). Otherwise — and whenever no manifest exists — a
    fresh control arm runs on the same tasks in the same session. With
    --mock the model is a scripted fake; plumbing only, nothing recorded.
    """
    import hashlib
    import os

    from mintok.funnel import (
        RunManifest,
        SlicerRun,
        attribute_failure,
        manifests_compatible,
        slicer_promotion,
        task_fingerprint,
    )

    from live import DISCIPLINE, TOOL_SCHEMAS, run_loop
    from model_runner import GENERATION_DEFAULTS

    def fake_completion(model_name: str, system: str, messages: list, tools: list | None = None):
        """Scripted fake agent: primary tool, then suite, then stop."""
        calls = sum(1 for m in messages if m.get("role") == "assistant")
        primary = tools[0]["name"] if tools else "suite"
        arg = {"description": "the task targets"} if primary == "slice" else {"command": "true"}
        if calls >= 2:
            return "end_turn", [{"type": "text", "text": "mock complete"}], None
        return (
            "tool_use",
            [{"type": "tool_use", "id": f"m{calls}", "name": primary, "input": arg}],
            None,
        )

    def arm_manifest(policy: str) -> RunManifest:
        prompt = DISCIPLINE[policy]
        binding, repo_hashes = [], {}
        for t in sorted(tasks, key=lambda t: t["id"]):
            fp = task_fingerprint(t, FIXTURES / t["repo"])
            binding.append(f"{t['id']}:{fp['prompt_hash']}:{fp['checker_hash']}:{fp.get('fixture_hash', '')}")
            repo_hashes[t["repo"]] = fp.get("fixture_hash", "")
        return RunManifest(
            provider=provider,
            provider_api_version=(
                "openrouter/v1 (provider-fixed)" if provider == "openrouter" else "anthropic 2023-06-01"
            ),
            model=model,
            reasoning_effort=os.environ.get("MINTOK_REASONING_EFFORT", "none"),
            temperature=str(GENERATION_DEFAULTS["temperature"]),
            top_p=str(GENERATION_DEFAULTS["top_p"]),
            max_output_tokens=str(GENERATION_DEFAULTS["max_output_tokens"]),
            system_prompt_hash=hashlib.sha256(prompt.encode()).hexdigest()[:16],
            agent_instruction_hash=hashlib.sha256(
                b"task instruction delivered verbatim as the first user message"
            ).hexdigest()[:16],
            toolset_hash=hashlib.sha256(
                json.dumps(TOOL_SCHEMAS[policy], sort_keys=True).encode()
            ).hexdigest()[:16],
            harness_version=subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=HARNESS_ROOT, capture_output=True, text=True,
            ).stdout.strip(),
            task_set_hash=hashlib.sha256("\n".join(binding).encode()).hexdigest()[:16],
            repo_snapshot_hashes=",".join(f"{k}:{v}" for k, v in sorted(repo_hashes.items())),
        )

    if holdout:
        tasks = json.loads(HOLDOUT_TASKS.read_text())
        slicer_arm = "slicer-holdout"
        control_arm_fresh = "control-holdout"
    else:
        tasks = [t for t in json.loads(TASKS_JSON.read_text()) if t["klass"] == "large_file_navigation"]
        slicer_arm = "slicer-large"
        control_arm_fresh = "control-large"

    scratch = Path("/home/aqua/bench-run")
    scratch.mkdir(parents=True, exist_ok=True)

    # Resume from durable records: a free-tier quota (e.g. 50 requests/day)
    # can interrupt a 30-trajectory run; never change arms mid-experiment —
    # skip tasks already recorded and continue from here, days later.
    def recorded_tasks(arm: str) -> set[str]:
        path = RUNS_DIR / f"{arm}.jsonl"
        if not path.exists():
            return set()
        return {json.loads(l)["task_id"] for l in path.read_text().splitlines() if l.strip()}

    done_slicer = recorded_tasks(slicer_arm)
    done_control = recorded_tasks(control_arm_fresh)

    frozen_path = (RUNS_DIR / f"{control_arm_fresh}.manifest.json") if holdout else (RUNS_DIR / "control-eval.manifest.json")
    frozen = RunManifest.from_json(frozen_path.read_text()) if frozen_path.exists() else None
    live_manifest = arm_manifest("S")
    compatible, reasons = manifests_compatible(frozen, live_manifest)
    if compatible:
        control_arm = control_arm_fresh if holdout else "control-eval"
        print(f"manifest guard: frozen {control_arm} matches live config; reusing control trajectories")
    else:
        control_arm = control_arm_fresh
        print(f"manifest guard: frozen control NOT reusable ->")
        for reason in reasons:
            print(f"  - {reason}")
        print(f"running {len(tasks)} fresh control trajectories in this session (worth the spend)")

    runs: list[SlicerRun] = []
    control_rows: dict[str, dict] = {}
    for task in tasks:
        task_id = task["id"]
        # Paired execution interleaving:
        # Determine randomized arm order per task using a deterministic hash seed
        # to ensure reproducibility while balancing execution order across the 80 runs.
        seed_byte = hashlib.sha256(f"{task_id}:{model}:arm_order".encode()).digest()[0]
        arm_order = ["control", "slicer"] if (seed_byte % 2 == 1) else ["slicer", "control"]

        for arm_to_run in arm_order:
            if arm_to_run == "slicer":
                copy = scratch / "copies" / f"{task_id}-{slicer_arm}"
                log = scratch / "logs" / f"{task_id}-{slicer_arm}.jsonl"
                if task_id in done_slicer:
                    print(f"resume: {task_id} slicer already recorded; using durable record")
                else:
                    prepare(copy, task_id)
                    log.unlink(missing_ok=True)
                    run_loop(
                        copy, log, "S", task["instruction"], model,
                        completion=fake_completion if mock else None,
                        provider=provider,
                    )
                check(copy, task_id, log)

                entries = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
                slice_entries = [e for e in entries if e.get("tool") == "slice"]
                slice_tok = sum(estimate_tokens(e.get("output", "")) for e in slice_entries)
                fallback_tok = sum(estimate_tokens(e.get("output", "")) for e in entries if e.get("tool") == "read")
                edit_rejections = sum(
                    1 for e in entries if e.get("tool") == "patch" and str(e.get("output", "")).startswith("rejected")
                )
                slice_spans = []
                for e in slice_entries:
                    slice_spans += re.findall(r"([\w./-]+\.py):(\d+)-(\d+)", e.get("output", ""))
                accepted_patches = [
                    (e["args"]["file"], e["args"]["start"], e["args"]["end"])
                    for e in entries
                    if e.get("tool") == "patch" and str(e.get("output", "")).startswith("patched")
                ]
                target_in_slice = any(
                    any(p == f and int(s) <= a and int(t) >= b for p, s, t in slice_spans)
                    for f, a, b in accepted_patches
                )
                score = json.loads((log.parent / f"{log.stem}.score.json").read_text())
                prompt_tokens = estimate_tokens(task["instruction"]) + PROMPT_OVERHEAD_TOKENS
                flags = dict(
                    solved=score["solved"],
                    target_in_slice=target_in_slice,
                    slice_dominated=slice_tok >= fallback_tok,
                    slice_truncated=any(e.get("expanded") for e in slice_entries),
                    edit_rejections=edit_rejections,
                    suite_ok=score["suite_ok"],
                )
                klass = attribute_failure(**flags)
                runs.append(
                    SlicerRun(
                        task_id=task_id,
                        solved=score["solved"],
                        tokens=score["tool_output_tokens"] + prompt_tokens,
                        turns=score["turns"],
                        slice_tokens=slice_tok,
                        fallback_tokens=fallback_tok,
                        expanded=flags["slice_truncated"],
                    )
                )
                if not mock:
                    if task_id not in done_slicer:
                        record(slicer_arm, task_id, log)
                usage_path = log.parent / f"{log.stem}.usage.json"
                usd = json.loads(usage_path.read_text())["usd"] if usage_path.exists() else 0.0
                note = f" failure={klass}" if not score["solved"] else ""
                print(f"{task_id}: solved={score['solved']} tokens={runs[-1].tokens} turns={runs[-1].turns} "
                      f"slice={slice_tok} fallback={fallback_tok} usd={usd:.4f}{note}")

            elif arm_to_run == "control" and control_arm == control_arm_fresh:
                if task_id in done_control:
                    print(f"resume: {task_id} control already recorded; using durable record")
                else:
                    ccopy = scratch / "copies" / f"{task_id}-{control_arm_fresh}"
                    prepare(ccopy, task_id)
                    clog = scratch / "logs" / f"{task_id}-{control_arm_fresh}.jsonl"
                    clog.unlink(missing_ok=True)
                    run_loop(
                        ccopy, clog, "control", task["instruction"], model,
                        completion=fake_completion if mock else None,
                        provider=provider,
                    )
                    check(ccopy, task_id, clog)
                    if not mock:
                        record(control_arm, task_id, clog)
                cscore = json.loads(
                    (scratch / "logs" / f"{task_id}-{control_arm_fresh}.score.json").read_text()
                )
                cusage_path = scratch / "logs" / f"{task_id}-{control_arm_fresh}.usage.json"
                prompt_tokens = estimate_tokens(task["instruction"]) + PROMPT_OVERHEAD_TOKENS
                control_rows[task_id] = {
                    "solved": cscore["solved"],
                    "input_tokens": cscore["tool_output_tokens"] + prompt_tokens,
                    "turns": cscore["turns"],
                    "frontier_usd": json.loads(cusage_path.read_text())["usd"] if cusage_path.exists() else 0.0,
                }


    if control_arm == "control-eval":
        control_rows = {
            json.loads(l)["task_id"]: json.loads(l)
            for l in (RUNS_DIR / f"{control_arm}.jsonl").read_text().splitlines()
            if l.strip()
        }

    if not mock:
        RUNS_DIR.mkdir(exist_ok=True)
        (RUNS_DIR / f"{slicer_arm}.manifest.json").write_text(live_manifest.to_json())
        if control_arm == control_arm_fresh:
            (RUNS_DIR / f"{control_arm}.manifest.json").write_text(arm_manifest("control").to_json())

    rep = slicer_promotion(runs, solve_target=len(tasks))
    ctrl_solved = sum(control_rows[r.task_id]["solved"] for r in runs if r.task_id in control_rows)

    def usage_rows(prefix: str) -> list[dict]:
        rows = []
        for r in runs:
            upath = scratch / "logs" / f"{r.task_id}-{prefix}.usage.json"
            rows.append(json.loads(upath.read_text()) if upath.exists() else {})
        return rows

    slicer_usage = usage_rows(slicer_arm)
    control_usage = usage_rows(control_arm_fresh) if control_arm == control_arm_fresh else []

    def tok_sum(rows: list[dict], key: str) -> int:
        return sum(u.get(key, 0) for u in rows)

    slicer_usd_total = sum(u.get("usd", 0.0) for u in slicer_usage)
    control_usd_total = sum(u.get("usd", 0.0) for u in control_usage)
    slicer_usd_per_task = sorted(u.get("usd", 0.0) for u in slicer_usage)
    control_usd_per_task = sorted(u.get("usd", 0.0) for u in control_usage)
    slicer_solved_tok = sum(r.tokens for r in runs if r.solved)
    control_solved_tok = sum(control_rows[r.task_id]["input_tokens"] for r in runs if control_rows[r.task_id]["solved"])
    label = "DRY RUN (mock model — plumbing only, verdict not meaningful)" if mock else "live"
    print(f"\nslicer promotion ({rep.attempted} large-module tasks, {label}) vs {control_arm}:")
    hdr = f"  {'metric':<22}{'control':>16}{'slicer':>16}{'delta':>10}"
    print(hdr)

    def row(name: str, c, s, fmt: str = "{:.4f}") -> None:
        def cell(v) -> str:
            if v is None:
                return "—"
            return fmt.format(v) if isinstance(v, (int, float)) else str(v)

        delta = f"{(s / c if c else 0):.2f}x" if isinstance(c, (int, float)) and c and isinstance(s, (int, float)) else "-"
        print(f"  {name:<22}{cell(c):>16}{cell(s):>16}{delta:>10}")

    control_turns = sum(control_rows[r.task_id].get("turns", 0) for r in runs)
    row("solved", ctrl_solved, rep.solved, "{:d}")
    row(
        "$/solved",
        (control_usd_total / ctrl_solved) if ctrl_solved and control_usage else None,
        (slicer_usd_total / rep.solved) if rep.solved else None,
    )
    row("total api $", control_usd_total or None, slicer_usd_total or None)
    row("input tokens", tok_sum(control_usage, "input_tokens") or None, tok_sum(slicer_usage, "input_tokens") or None, "{:d}")
    row("cache reads", tok_sum(control_usage, "cached_input_tokens") or None, tok_sum(slicer_usage, "cached_input_tokens") or None, "{:d}")
    row("cache writes", tok_sum(control_usage, "cache_write_tokens") or None, tok_sum(slicer_usage, "cache_write_tokens") or None, "{:d}")
    row("output tokens", tok_sum(control_usage, "output_tokens") or None, tok_sum(slicer_usage, "output_tokens") or None, "{:d}")
    row("reasoning tokens", tok_sum(control_usage, "reasoning_tokens") or None, tok_sum(slicer_usage, "reasoning_tokens") or None, "{:d}")
    row("turns/task", (control_turns / rep.attempted) if control_usage else None,
        rep.turns_per_attempt, "{:.1f}")
    row(
        "tool-context/solved",
        (control_solved_tok / ctrl_solved) if ctrl_solved else None,
        (slicer_solved_tok / rep.solved) if rep.solved else None,
        "{:.0f}",
    )

    def provider_tokens(u: dict) -> int:
        return sum(u.get(k, 0) for k in ("input_tokens", "cached_input_tokens", "cache_write_tokens", "output_tokens", "reasoning_tokens"))

    slicer_ptok_solved = sum(
        provider_tokens(u) for u, r in zip(slicer_usage, runs) if r.solved
    )
    control_ptok_solved = sum(
        provider_tokens(u) for u, r in zip(control_usage, runs) if control_rows[r.task_id]["solved"]
    )
    row("provider tok/solved",
        (control_ptok_solved / ctrl_solved) if ctrl_solved and control_usage else None,
        (slicer_ptok_solved / rep.solved) if rep.solved else None,
        "{:.0f}")
    # Tail gate on tokens: the free-model runs price at $0, so p95/max
    # regressions must be visible in token space, not just dollar space.
    control_ptok_task = [provider_tokens(u) for u in control_usage] or None
    slicer_ptok_task = [provider_tokens(u) for u in slicer_usage] or None
    if slicer_ptok_task and any(slicer_ptok_task):
        s_sorted = sorted(slicer_ptok_task)
        n9 = max(1, math.ceil(0.95 * len(s_sorted))) - 1
        c_sorted = sorted(control_ptok_task) if control_ptok_task else []
        c_n9 = max(1, math.ceil(0.95 * len(c_sorted))) - 1 if c_sorted else 0
        row("p95 provider tok/task",
            c_sorted[c_n9] if c_sorted else None, s_sorted[n9], "{:d}")
        row("max provider tok/task",
            max(c_sorted) if c_sorted else None, max(s_sorted), "{:d}")
    if not mock and slicer_usd_total == 0.0:
        print("  (free-model run: $/solved is uninformative at $0; headline is "
              "provider tokens/solved and tool-context/solved)")
    print(f"  {'slice acceptance':<22}{'-':>16}{rep.acceptance_rate:>16.2f}")
    print(f"  {'raw fallback rate':<22}{'-':>16}{rep.fallback_rate:>16.2f}")
    print(f"  {'expanded slices':<22}{'-':>16}{rep.expanded_rate:>16.2f}")
    if slicer_usd_per_task and any(slicer_usd_per_task):
        n9 = max(1, math.ceil(0.95 * len(slicer_usd_per_task))) - 1
        c_n9 = max(1, math.ceil(0.95 * len(control_usd_per_task))) - 1 if control_usd_per_task else 0
        row("p95 $/task", control_usd_per_task[c_n9] if control_usage else None, slicer_usd_per_task[n9])
        row("max $/task", max(control_usd_per_task) if control_usage else None, max(slicer_usd_per_task))
    print(f"  VERDICT: {rep.verdict.upper()} (bar: solve >= {len(tasks)}/{len(tasks)}, "
          f"<=800 promote / <=600 strong / <=400 excellent)")
    if holdout:
        solve_drop_pp = (ctrl_solved - rep.solved) / len(tasks)
        correctness_pass = solve_drop_pp <= 0.05
        if slicer_ptok_solved and rep.solved and control_ptok_solved and ctrl_solved:
            s_pts = slicer_ptok_solved / rep.solved
            c_pts = control_ptok_solved / ctrl_solved
            ratio_val = c_pts / s_pts if s_pts else 0.0
            if not correctness_pass:
                h_verdict = f"REJECT (solve rate drop {solve_drop_pp * 100:.1f}pp > 5pp guard)"
            elif ratio_val >= 4.0:
                h_verdict = f"EXCELLENT ({ratio_val:.2f}x provider tok/solved >= 4.0x)"
            elif ratio_val >= 3.0:
                h_verdict = f"STRONG ({ratio_val:.2f}x provider tok/solved >= 3.0x)"
            elif ratio_val >= 2.0:
                h_verdict = f"PASS ({ratio_val:.2f}x provider tok/solved >= 2.0x)"
            else:
                h_verdict = f"REJECT ({ratio_val:.2f}x provider tok/solved < 2.0x threshold)"
            print(f"  HOLDOUT GATES: {h_verdict}")

        # Exact paired statistics
        c_only = sum(1 for r in runs if not r.solved and control_rows.get(r.task_id, {}).get("solved"))
        s_only = sum(1 for r in runs if r.solved and not control_rows.get(r.task_id, {}).get("solved"))
        both_solve = sum(1 for r in runs if r.solved and control_rows.get(r.task_id, {}).get("solved"))
        both_fail = sum(1 for r in runs if not r.solved and not control_rows.get(r.task_id, {}).get("solved"))
        print("\n  paired solve breakdown:")
        print(f"    both solve:         {both_solve:>3d}")
        print(f"    control-only solve: {c_only:>3d}")
        print(f"    slicer-only solve:  {s_only:>3d}")
        print(f"    both fail:          {both_fail:>3d}")

        # On both-solved tasks:
        both_solved_ratios: list[float] = []
        for r, u_s, u_c in zip(runs, slicer_usage, control_usage):
            if r.solved and control_rows.get(r.task_id, {}).get("solved"):
                s_tok = provider_tokens(u_s)
                c_tok = provider_tokens(u_c)
                if s_tok > 0:
                    both_solved_ratios.append(c_tok / s_tok)

        if both_solved_ratios:
            bs = sorted(both_solved_ratios)
            def pctl(vals: list[float], q: float) -> float:
                return vals[min(len(vals) - 1, max(0, math.ceil(q * len(vals)) - 1))]
            med = bs[len(bs) // 2]
            geom = math.exp(sum(math.log(max(1e-6, x)) for x in bs) / len(bs))
            print("\n  both-solved provider-token ratios (control / slicer savings):")
            print(f"    median:            {med:.2f}x")
            print(f"    geometric mean:    {geom:.2f}x")
            print(f"    p25:               {pctl(bs, 0.25):.2f}x")
            print(f"    p75:               {pctl(bs, 0.75):.2f}x")
            print(f"    p95:               {pctl(bs, 0.95):.2f}x")
            print(f"    max:               {max(bs):.2f}x")

        # Template family breakdown
        by_family: dict[str, dict] = {}
        for t, r, u_s, u_c in zip(tasks, runs, slicer_usage, control_usage):
            fam = t.get("template", "unknown")
            if fam not in by_family:
                by_family[fam] = {
                    "count": 0, "s_solved": 0, "c_solved": 0,
                    "s_tok": 0, "c_tok": 0,
                }
            f_entry = by_family[fam]
            f_entry["count"] += 1
            if r.solved:
                f_entry["s_solved"] += 1
            if control_rows.get(r.task_id, {}).get("solved"):
                f_entry["c_solved"] += 1
            f_entry["s_tok"] += provider_tokens(u_s)
            f_entry["c_tok"] += provider_tokens(u_c)

        print("\n  template family breakdown:")
        print(f"    {'family':<12}{'tasks':>6}{'ctrl_ok':>9}{'slc_ok':>9}{'ctrl_tok':>11}{'slc_tok':>11}{'ratio':>8}")
        for fam, d in sorted(by_family.items()):
            c_tok = d["c_tok"]
            s_tok = d["s_tok"]
            ratio_str = f"{c_tok / s_tok:.2f}x" if s_tok > 0 else "—"
            print(f"    {fam:<12}{d['count']:>6}{d['c_solved']:>9}{d['s_solved']:>9}{c_tok:>11}{s_tok:>11}{ratio_str:>8}")



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

    fpr = sub.add_parser("fingerprint")
    fpr.add_argument("--task", required=True)

    vfy = sub.add_parser("verify")
    vfy.add_argument("--task", required=True)
    vfy.add_argument("--fingerprint", type=Path, required=True)

    sub.add_parser("fast8")

    rpl = sub.add_parser("replay")
    rpl.add_argument("--logs", type=Path, required=True)
    rpl.add_argument("--out", type=Path, default=None)

    orc = sub.add_parser("oracle")
    orc.add_argument("--arm", action="append", required=True)

    rte = sub.add_parser("route")
    rte.add_argument("--task", default=None)
    rte.add_argument("--eval", action="store_true")

    slc = sub.add_parser("slice")
    slc.add_argument("--task", default=None)
    slc.add_argument("--eval", action="store_true")

    prm = sub.add_parser("promote-large")
    prm.add_argument("--model", required=True, help="exact model snapshot id; never a random router slug")
    prm.add_argument("--provider", default="anthropic", choices=["anthropic", "openrouter"])
    prm.add_argument("--mock", action="store_true", help="scripted fake model; plumbing only")
    prm.add_argument("--holdout", action="store_true", help="run on fresh 40-task holdout set instead of 15-task dev set")

    args = parser.parse_args()
    if args.cmd == "prepare":
        prepare(args.dest, args.task)
    elif args.cmd == "check":
        check(args.root, args.task, args.log)
    elif args.cmd == "record":
        record(args.arm, args.task, args.log)
    elif args.cmd == "fingerprint":
        fingerprint(args.task)
    elif args.cmd == "verify":
        verify(args.task, args.fingerprint)
    elif args.cmd == "fast8":
        fast8()
    elif args.cmd == "replay":
        replay(args.logs, args.out)
    elif args.cmd == "oracle":
        oracle(args.arm)
    elif args.cmd == "route":
        route_cmd(args.task, args.eval)
    elif args.cmd == "slice":
        slice_cmd(args.task, args.eval)
    elif args.cmd == "promote-large":
        if args.model.lower() in ("openrouter/free", "openrouter/auto", "free", "auto"):
            raise SystemExit(
                "refusing to run: --model must be one specific model slug, never a "
                "random router slug — same-model control requires a fixed upstream"
            )
        promote_large(args.model, mock=args.mock, provider=args.provider, holdout=args.holdout)
    else:
        report()


if __name__ == "__main__":
    main()
