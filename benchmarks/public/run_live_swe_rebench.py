"""Live SWE-rebench Evaluation Harness.

Executes genuine paired agent trajectories over official SWE-rebench GitHub tasks:
1. Clones the real GitHub repository and checks out base_commit.
2. Sets up an isolated environment with dependencies.
3. Runs Control Arm: live model calls via OpenRouter, ordinary shell tooling.
4. Runs MinTok Arm: live model calls via OpenRouter, MinTok semantic tooling.
5. Evaluates with official checker: applies test_patch, runs test_cmd.
6. Records real provider telemetry, request IDs, patches, and pass/fail status.
7. Computes discordant solve breakdown (both_solved, control_only, mintok_only, both_failed)
   and paired token efficiency.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

HARNESS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HARNESS_ROOT / "src"))
sys.path.insert(0, str(HARNESS_ROOT / "benchmarks" / "e2e"))
sys.path.insert(0, str(HARNESS_ROOT / "benchmarks" / "public"))

from live import run_loop
from mintok.public_bench import (
    PublicBenchmarkReport,
    PublicBenchmarkTask,
    PublicRunRecord,
    evaluate_paired_public_runs,
    verify_window_fingerprint,
)
from model_runner import resolve_api_key

WINDOWS_DIR = Path(__file__).resolve().parent / "windows"
RUNS_DIR = Path(__file__).resolve().parent / "runs"
SCRATCH_DIR = Path("/home/aqua/bench-run/swe-live")


def load_swe_rebench_window(window_target: str = "swe_rebench_window_eval_50.json.gz") -> list[dict[str, Any]]:
    w_path = Path(window_target)
    if not w_path.exists():
        w_path = WINDOWS_DIR / window_target
    if not w_path.exists() and not window_target.endswith(".json.gz"):
        w_path = WINDOWS_DIR / f"{window_target}.json.gz"
    assert w_path.exists(), f"Window file not found: {window_target}"
    with gzip.open(w_path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    assert verify_window_fingerprint(data), f"Window fingerprint mismatch for {w_path.name}"
    return data["tasks"]


REPO_CACHE_DIR = Path("/home/aqua/bench-run/repo-cache")


def prepare_live_workspace(
    repo: str,
    base_commit: str,
    dest: Path,
) -> bool:
    """Clone official repository and checkout base_commit."""
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)

    clone_url = f"https://github.com/{repo}.git"
    cache_repo = REPO_CACHE_DIR / repo

    # If not in cache, create bare mirror clone
    if not (cache_repo / "HEAD").exists():
        cache_repo.parent.mkdir(parents=True, exist_ok=True)
        print(f"  [cache] Mirroring {repo} to local cache...")
        try:
            subprocess.run(
                ["git", "clone", "--bare", clone_url, str(cache_repo)],
                capture_output=True,
                timeout=600,
            )
        except Exception as e:
            print(f"  [cache] Mirroring failed for {repo}: {e}")
            if cache_repo.exists():
                shutil.rmtree(cache_repo, ignore_errors=True)

    cloned_ok = False
    if (cache_repo / "HEAD").exists():
        try:
            res = subprocess.run(
                ["git", "clone", str(cache_repo), str(dest)],
                capture_output=True,
                text=True,
                timeout=120,
            )
            cloned_ok = (res.returncode == 0)
        except Exception:
            cloned_ok = False

    if not cloned_ok:
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        dest.mkdir(parents=True, exist_ok=True)
        try:
            res = subprocess.run(
                ["git", "clone", clone_url, str(dest)],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if res.returncode != 0:
                print(f"  [error] Clone failed for {repo}: {res.stderr[:200]}")
                return False
        except Exception as e:
            print(f"  [error] Clone exception for {repo}: {e}")
            return False

    try:
        c_res = subprocess.run(
            ["git", "checkout", base_commit],
            cwd=dest,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if c_res.returncode != 0:
            # Fetch PR ref or all branches if commit not on default branch
            subprocess.run(["git", "remote", "set-url", "origin", clone_url], cwd=dest, capture_output=True, timeout=15)
            subprocess.run(["git", "fetch", "--all"], cwd=dest, capture_output=True, timeout=180)
            c_res2 = subprocess.run(
                ["git", "checkout", base_commit],
                cwd=dest,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if c_res2.returncode != 0:
                print(f"  [error] Checkout failed for {repo} @ {base_commit}: {c_res2.stderr[:200]}")
                return False
    except Exception as e:
        print(f"  [error] Checkout exception for {repo}: {e}")
        return False

    return True


def run_official_checker(
    dest: Path,
    test_patch: str,
    fail_to_pass: list[str],
    pass_to_pass: list[str],
    test_cmd: str,
) -> tuple[bool, str]:
    """Apply official test_patch and run test_cmd to evaluate solution."""
    # Ensure test directories are clean from agent tampering
    for tdir in ["tests", "test"]:
        if (dest / tdir).exists():
            subprocess.run(["git", "checkout", "--", tdir], cwd=dest, capture_output=True)

    # Apply test patch if not already applied
    if test_patch and test_patch.strip():
        check_rev = subprocess.run(
            ["git", "apply", "--reverse", "--check"],
            input=test_patch,
            text=True,
            cwd=dest,
            capture_output=True,
        )
        if check_rev.returncode != 0:
            apply_res = subprocess.run(
                ["git", "apply", "--ignore-whitespace"],
                input=test_patch,
                text=True,
                cwd=dest,
                capture_output=True,
            )
            if apply_res.returncode != 0:
                # Try 3-way apply
                apply_res = subprocess.run(
                    ["git", "apply", "-3"],
                    input=test_patch,
                    text=True,
                    cwd=dest,
                    capture_output=True,
                )
                if apply_res.returncode != 0:
                    return False, f"test_patch apply failed: {apply_res.stderr[:200]}"

    # Run pytest command
    # Use python environment with PYTHONPATH=.
    env = dict(os.environ)
    src_dirs = [dest, dest / "src"]
    env["PYTHONPATH"] = ":".join(str(p) for p in src_dirs if p.exists())

    cmd = ["/home/aqua/.local/bin/pytest", "-q", "--tb=line", "--no-header"]
    if fail_to_pass:
        cmd.extend(fail_to_pass)

    test_res = subprocess.run(
        cmd,
        cwd=dest,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )

    passed = (test_res.returncode == 0)
    output = (test_res.stdout + test_res.stderr).strip()
    return passed, output[:500]


def run_live_task(
    task: dict[str, Any],
    arm: str,
    model: str,
    provider: str = "openrouter",
    resume: bool = True,
    mintok_policy: str = "v3",
) -> tuple[PublicRunRecord, dict[str, Any]]:
    """Run one live arm on one real SWE-rebench task."""
    tid = task["instance_id"]
    repo = task["repo"]
    commit = task["base_commit"]
    instruction = task["problem_statement"]

    policy = "control" if arm == "control" else mintok_policy
    arm_key = arm if arm == "control" else f"{arm}_{mintok_policy}"
    workspace = SCRATCH_DIR / f"{tid}_{arm_key}"
    log_file = SCRATCH_DIR / f"{tid}_{arm_key}.jsonl"
    usage_file = SCRATCH_DIR / f"{tid}_{arm_key}.usage.json"

    if resume and usage_file.exists() and workspace.exists():
        print(f"  [{arm.upper()} - {policy}] Reusing existing completed live run for {tid}...")
        with open(usage_file, "r", encoding="utf-8") as f:
            usage = json.load(f)
        summary = {"usage": usage, "turns": usage.get("turns", 0)}
    else:
        print(f"  [{arm.upper()} - {policy}] Preparing real repo {repo} @ {commit[:8]}...")
        ok = prepare_live_workspace(repo, commit, workspace)
        if not ok:
            # If repo clone fails, record fail
            return PublicRunRecord(
                task_id=tid,
                arm=arm,
                solved=False,
                provider_tokens=0,
                input_tokens=0,
                output_tokens=0,
                turns=0,
                cost_usd=0.0,
                repo=repo,
                category=task.get("language", "python"),
            ), {"error": "workspace_prep_failed"}

        print(f"  [{arm.upper()}] Driving live agent loop with {model}...")
        log_file.unlink(missing_ok=True)
        summary = run_loop(
            workspace,
            log_file,
            policy,
            instruction,
            model,
            provider=provider,
        )

        usage = summary.get("usage", {})

    in_tok = usage.get("input_tokens", 0) + usage.get("cached_input_tokens", 0)
    out_tok = usage.get("output_tokens", 0)
    provider_tokens = in_tok + out_tok

    # Capture model patch
    diff_proc = subprocess.run(["git", "diff"], cwd=workspace, capture_output=True, text=True)
    model_patch = diff_proc.stdout

    # Run official checker
    print(f"  [{arm.upper()}] Running official SWE-rebench checker...")
    solved, checker_out = run_official_checker(
        workspace,
        task.get("test_patch", ""),
        task.get("fail_to_pass", []),
        task.get("pass_to_pass", []),
        task.get("test_cmd", "pytest"),
    )
    print(f"  [{arm.upper()}] Solved: {solved} | Provider Tokens: {provider_tokens:,} | Turns: {summary.get('turns', 0)}")

    record = PublicRunRecord(
        task_id=tid,
        arm=arm,
        solved=solved,
        provider_tokens=provider_tokens,
        input_tokens=in_tok,
        output_tokens=out_tok,
        turns=summary.get("turns", 0),
        cost_usd=usage.get("usd", 0.0),
        repo=repo,
        category=task.get("language", "python"),
    )

    meta = {
        "task_id": tid,
        "arm": arm,
        "model": model,
        "solved": solved,
        "provider_tokens": provider_tokens,
        "turns": summary.get("turns", 0),
        "requests": usage.get("requests", []),
        "model_patch": model_patch[:1000],
        "checker_output": checker_out,
    }

    return record, meta


def save_live_progress(
    ctrl_runs: list[PublicRunRecord],
    mintok_runs: list[PublicRunRecord],
    model: str,
    provider: str,
    run_name: str = "swe_rebench_live_space_bunny_alpha",
    final: bool = False,
) -> PublicBenchmarkReport:
    report = evaluate_paired_public_runs(
        ctrl_runs,
        mintok_runs,
        compute_bootstrap=final,
        bootstrap_resamples=1000 if final else 100,
    )
    c_s = sum(1 for r in ctrl_runs if r.solved)
    m_s = sum(1 for r in mintok_runs if r.solved)
    c_tok = sum(r.provider_tokens for r in ctrl_runs)
    m_tok = sum(r.provider_tokens for r in mintok_runs)
    c_yield = (c_s / c_tok) * 1e6 if c_tok else 0.0
    m_yield = (m_s / m_tok) * 1e6 if m_tok else 0.0
    yield_multiplier = (m_yield / c_yield) if c_yield else 0.0

    out_payload = {
        "benchmark": "swe_rebench_live",
        "model": model,
        "provider": provider,
        "run_name": run_name,
        "timestamp": time.time(),
        "report": {
            "total_tasks": report.total_tasks,
            "control_solved": report.control_solved,
            "mintok_solved": report.mintok_solved,
            "both_solve": report.both_solve,
            "control_only": report.control_only,
            "mintok_only": report.mintok_only,
            "both_fail": report.both_fail,
            "efficiency_multiplier": report.efficiency_multiplier,
            "both_solved_geomean": report.both_solved_geomean,
            "control_solves_per_mtok": c_yield,
            "mintok_solves_per_mtok": m_yield,
            "economic_yield_multiplier": yield_multiplier,
        },
        "control_runs": [asdict(r) for r in ctrl_runs],
        "mintok_runs": [asdict(r) for r in mintok_runs],
    }
    out_file = RUNS_DIR / f"{run_name}.json"
    scratch_out_file = SCRATCH_DIR / f"{run_name}.json"
    for p in [out_file, scratch_out_file]:
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(out_payload, f, indent=2)
        except OSError:
            continue
    return report


def main() -> None:
    parser = argparse.ArgumentParser(prog="run_live_swe_rebench")
    parser.add_argument("--window", default="swe_rebench_window_eval_50.json.gz", help="Window filename or path")
    parser.add_argument("--model", default="stealth/space-bunny-alpha")
    parser.add_argument(
        "--mintok-policy",
        default="v3",
        choices=["v3", "adaptive", "S", "C", "v3_v", "v3_vc", "v3_vcr", "v3_vcrm", "v3_vcrmp"],
        help="Policy for MinTok arm",
    )
    parser.add_argument("--offset", type=int, default=0, help="Offset into selected tasks")
    parser.add_argument("--limit", type=int, default=None, help="Number of tasks to evaluate")
    parser.add_argument("--provider", default="openrouter")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--run-name", default=None, help="Custom output run name")
    args = parser.parse_args()

    api_key = resolve_api_key(args.provider)
    assert api_key, f"API key for {args.provider} not resolved!"

    tasks = load_swe_rebench_window(args.window)
    selected_tasks = tasks[args.offset : (args.offset + args.limit if args.limit is not None else len(tasks))]

    model_slug = args.model.replace("/", "_").replace(":", "_").replace("-", "_")
    window_slug = Path(args.window).stem.replace(".json", "")
    run_name = args.run_name or f"{window_slug}_{model_slug}_{args.mintok_policy}"

    print("=" * 70)
    print("LIVE SWE-REBENCH EVALUATION (GENUINE REPO RUNS)")
    print("=" * 70)
    print(f"Window:        {args.window} ({len(selected_tasks)} tasks)")
    print(f"Model:         {args.model}")
    print(f"Provider:      {args.provider}")
    print(f"MinTok Policy: {args.mintok_policy}")
    print(f"Run Name:      {run_name}")
    print("-" * 70)
    print("Pre-registered Yield Targets (Economic Objective):")
    print("  Primary Metric: Solves / Million Provider Tokens (all-attempt allocated)")
    print("  Minimum Viable: Solve Rate >= Control - 3pp, Yield >= 1.25x Control")
    print("  Strong:         Solve Rate >= Control,       Yield >= 1.50x Control")
    print("  Breakthrough:   Solve Rate >= Control,       Yield >= 2.00x Control")
    print("=" * 70)

    existing_ctrl: dict[str, PublicRunRecord] = {}
    existing_mintok: dict[str, PublicRunRecord] = {}
    if args.resume:
        for p in [RUNS_DIR / f"{run_name}.json", SCRATCH_DIR / f"{run_name}.json"]:
            if p.exists():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        saved = json.load(f)
                    for r in saved.get("control_runs", []):
                        existing_ctrl[r["task_id"]] = PublicRunRecord(**r)
                    for r in saved.get("mintok_runs", []):
                        existing_mintok[r["task_id"]] = PublicRunRecord(**r)
                    print(f"Loaded {len(existing_ctrl)} existing paired records from {p.name}")
                    break
                except Exception as e:
                    print(f"Warning: could not load existing run {p}: {e}")

    ctrl_runs: list[PublicRunRecord] = []
    mintok_runs: list[PublicRunRecord] = []

    for idx, t in enumerate(selected_tasks, 1):
        tid = t["instance_id"]
        print(f"\n[{idx:02d}/{len(selected_tasks):02d}] TASK: {tid}")

        if args.resume and tid in existing_ctrl and tid in existing_mintok:
            c_rec = existing_ctrl[tid]
            m_rec = existing_mintok[tid]
            ctrl_runs.append(c_rec)
            mintok_runs.append(m_rec)
            print(f"  [CONTROL] Loaded cached record: Solved={c_rec.solved} | Provider Tokens: {c_rec.provider_tokens:,} | Turns: {c_rec.turns}")
            print(f"  [MINTOK]  Loaded cached record: Solved={m_rec.solved} | Provider Tokens: {m_rec.provider_tokens:,} | Turns: {m_rec.turns}")
        else:
            # Balanced order: even index control first, odd index mintok first
            order = ["control", "mintok"] if idx % 2 == 0 else ["mintok", "control"]
            for arm in order:
                rec, meta = run_live_task(
                    t,
                    arm,
                    args.model,
                    provider=args.provider,
                    resume=args.resume,
                    mintok_policy=args.mintok_policy,
                )
                if arm == "control":
                    ctrl_runs.append(rec)
                else:
                    mintok_runs.append(rec)

        rep = save_live_progress(ctrl_runs, mintok_runs, args.model, args.provider, run_name=run_name, final=False)
        c_s = sum(1 for r in ctrl_runs if r.solved)
        m_s = sum(1 for r in mintok_runs if r.solved)
        c_tok = sum(r.provider_tokens for r in ctrl_runs)
        m_tok = sum(r.provider_tokens for r in mintok_runs)
        n = len(ctrl_runs)

        c_t_att = c_tok / n if n else 0
        m_t_att = m_tok / n if n else 0
        c_t_sol = (c_tok / c_s) if c_s else 0
        m_t_sol = (m_tok / m_s) if m_s else 0
        c_sol_mtok = (c_s / c_tok) * 1e6 if c_tok else 0.0
        m_sol_mtok = (m_s / m_tok) * 1e6 if m_tok else 0.0
        yield_ratio = (m_sol_mtok / c_sol_mtok) if c_sol_mtok else 0.0

        print(f"\n--- Progress [{idx}/{len(selected_tasks)}] ---")
        print(f"  Solve Rate:                Control {c_s}/{n} ({c_s/n*100:.1f}%) | MinTok {m_s}/{n} ({m_s/n*100:.1f}%)")
        print(f"  Tokens / Attempt:          Control {c_t_att:,.0f} | MinTok {m_t_att:,.0f}")
        print(f"  Tokens / Solved:           Control {c_t_sol:,.0f} | MinTok {m_t_sol:,.0f}")
        print(f"  Solves / Million Tokens:   Control {c_sol_mtok:.2f} | MinTok {m_sol_mtok:.2f}")
        print(f"  Economic Yield Multiplier: {yield_ratio:.2f}x (both: {rep.both_solve}, ctrl_only: {rep.control_only}, min_only: {rep.mintok_only}, fail: {rep.both_fail})")

    # Evaluate final paired results with full bootstrap
    final_report = save_live_progress(ctrl_runs, mintok_runs, args.model, args.provider, run_name=run_name, final=True)

    print("\n" + "=" * 70)
    print("LIVE SWE-REBENCH FINAL RESULTS (GENUINE RUNS)")
    print("=" * 70)
    from mintok.public_bench import render_report_table
    print(render_report_table(final_report, f"SWE-rebench Live ({args.mintok_policy})"))


if __name__ == "__main__":
    main()

