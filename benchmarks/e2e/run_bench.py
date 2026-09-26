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


def promote_large(model: str, mock: bool = False) -> None:
    """Isolated live test: slicer arm vs control on the same 15 large tasks.

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
    )

    from live import DISCIPLINE, TOOL_SCHEMAS, run_loop

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
        return RunManifest(
            provider="anthropic",
            model=model,
            reasoning_effort=os.environ.get("MINTOK_REASONING_EFFORT", "none"),
            system_prompt_hash=hashlib.sha256(prompt.encode()).hexdigest()[:16],
            toolset_hash=hashlib.sha256(
                json.dumps(TOOL_SCHEMAS[policy], sort_keys=True).encode()
            ).hexdigest()[:16],
            harness_version=subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=HARNESS_ROOT, capture_output=True, text=True,
            ).stdout.strip(),
            task_set_hash=hashlib.sha256(TASKS_JSON.read_bytes()).hexdigest()[:16],
        )

    tasks = [t for t in json.loads(TASKS_JSON.read_text()) if t["klass"] == "large_file_navigation"]
    scratch = Path("/home/aqua/bench-run")
    scratch.mkdir(parents=True, exist_ok=True)

    frozen_path = RUNS_DIR / "control-eval.manifest.json"
    frozen = RunManifest.from_json(frozen_path.read_text()) if frozen_path.exists() else None
    live_manifest = arm_manifest("S")
    compatible, reasons = manifests_compatible(frozen, live_manifest)
    if compatible:
        control_arm = "control-eval"
        print("manifest guard: frozen control-eval matches live config; reusing control trajectories")
    else:
        control_arm = "control-large"
        print("manifest guard: frozen control NOT reusable ->")
        for reason in reasons:
            print(f"  - {reason}")
        print("running 15 fresh control trajectories in this session (worth the spend)")

    runs: list[SlicerRun] = []
    control_rows: dict[str, dict] = {}
    for task in tasks:
        task_id = task["id"]
        copy = scratch / "copies" / f"{task_id}-slicer"
        prepare(copy, task_id)
        log = scratch / "logs" / f"{task_id}-slicer.jsonl"
        log.unlink(missing_ok=True)
        run_loop(
            copy, log, "S", task["instruction"], model,
            completion=fake_completion if mock else None,
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
            record("slicer-large", task_id, log)
        usage_path = log.parent / f"{log.stem}.usage.json"
        usd = json.loads(usage_path.read_text())["usd"] if usage_path.exists() else 0.0
        note = f" failure={klass}" if not score["solved"] else ""
        print(f"{task_id}: solved={score['solved']} tokens={runs[-1].tokens} turns={runs[-1].turns} "
              f"slice={slice_tok} fallback={fallback_tok} usd={usd:.4f}{note}")

        if control_arm == "control-large":
            ccopy = scratch / "copies" / f"{task_id}-control"
            prepare(ccopy, task_id)
            clog = scratch / "logs" / f"{task_id}-control-large.jsonl"
            clog.unlink(missing_ok=True)
            run_loop(
                ccopy, clog, "control", task["instruction"], model,
                completion=fake_completion if mock else None,
            )
            check(ccopy, task_id, clog)
            if not mock:
                record(control_arm, task_id, clog)
            cscore = json.loads((clog.parent / f"{clog.stem}.score.json").read_text())
            cusage_path = clog.parent / f"{clog.stem}.usage.json"
            control_rows[task_id] = {
                "solved": cscore["solved"],
                "input_tokens": cscore["tool_output_tokens"] + prompt_tokens,
                "frontier_usd": json.loads(cusage_path.read_text())["usd"] if cusage_path.exists() else 0.0,
            }

    if control_arm == "control-eval":
        control_rows = {
            json.loads(l)["task_id"]: json.loads(l)
            for l in (RUNS_DIR / f"{control_arm}.jsonl").read_text().splitlines()
            if l.strip()
        }

    RUNS_DIR.mkdir(exist_ok=True)
    (RUNS_DIR / "slicer-large.manifest.json").write_text(live_manifest.to_json())
    if control_arm == "control-large" and not mock:
        (RUNS_DIR / f"{control_arm}.manifest.json").write_text(arm_manifest("control").to_json())

    rep = slicer_promotion(runs, solve_target=len(tasks))
    ctrl_solved = sum(control_rows[r.task_id]["solved"] for r in runs)
    ctrl_tok = sum(control_rows[r.task_id]["input_tokens"] for r in runs)
    slicer_usd = 0.0
    for r in runs:
        upath = scratch / "logs" / f"{r.task_id}-slicer.usage.json"
        if upath.exists():
            slicer_usd += json.loads(upath.read_text())["usd"]
    label = "DRY RUN (mock model — plumbing only, verdict not meaningful)" if mock else "live"
    print(f"\nslicer promotion ({rep.attempted} large-module tasks, {label}) vs {control_arm}:")
    print(f"  slicer:  solved {rep.solved}/{rep.attempted}, tok/solved {rep.tokens_per_solved:.0f}, "
          f"turns/attempt {rep.turns_per_attempt:.1f}, p95 {rep.p95_tokens}, max {rep.max_tokens}, "
          f"${slicer_usd:.4f} total" + (f" (${slicer_usd / rep.solved:.4f}/solved)" if rep.solved else ""))
    print(f"  control: solved {ctrl_solved}/{rep.attempted}, tok/attempt {ctrl_tok / rep.attempted:.0f}")
    print(f"  slice acceptance {rep.acceptance_rate:.2f}, raw-fallback rate {rep.fallback_rate:.2f}, "
          f"expanded-slice rate {rep.expanded_rate:.2f}")
    if control_arm == "control-large" and not mock:
        c_usd = sum(r.get("frontier_usd", 0.0) for r in control_rows.values())
        if rep.solved and ctrl_solved:
            print(f"  $/solved: slicer ${slicer_usd / rep.solved:.4f} vs control ${c_usd / ctrl_solved:.4f}")
    print(f"  VERDICT: {rep.verdict.upper()} (bar: solve >= {len(tasks)}/15, "
          f"<=800 promote / <=600 strong / <=400 excellent)")


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
    prm.add_argument("--model", default="claude-sonnet-4-5")
    prm.add_argument("--mock", action="store_true", help="scripted fake model; plumbing only")

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
        promote_large(args.model, mock=args.mock)
    else:
        report()


if __name__ == "__main__":
    main()
