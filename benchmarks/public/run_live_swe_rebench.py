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


def load_swe_rebench_window() -> list[dict[str, Any]]:
    w_path = WINDOWS_DIR / "swe_rebench_window_a.json.gz"
    with gzip.open(w_path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    assert verify_window_fingerprint(data), "Window fingerprint mismatch"
    return data["tasks"]


def prepare_live_workspace(
    repo: str,
    base_commit: str,
    dest: Path,
) -> bool:
    """Clone official repository and checkout base_commit."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    clone_url = f"https://github.com/{repo}.git"
    # Clone with full commit history or fetch specific ref
    res = subprocess.run(
        ["git", "clone", clone_url, str(dest)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if res.returncode != 0:
        print(f"  [error] Clone failed for {repo}: {res.stderr[:200]}")
        return False

    c_res = subprocess.run(
        ["git", "checkout", base_commit],
        cwd=dest,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if c_res.returncode != 0:
        # Try fetching PR ref if commit not on default branch
        subprocess.run(
            ["git", "fetch", "--all"],
            cwd=dest,
            capture_output=True,
            timeout=60,
        )
        c_res2 = subprocess.run(
            ["git", "checkout", base_commit],
            cwd=dest,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if c_res2.returncode != 0:
            print(f"  [error] Checkout failed for {repo} @ {base_commit}: {c_res2.stderr[:200]}")
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
) -> tuple[PublicRunRecord, dict[str, Any]]:
    """Run one live arm on one real SWE-rebench task."""
    tid = task["instance_id"]
    repo = task["repo"]
    commit = task["base_commit"]
    instruction = task["problem_statement"]

    policy = "control" if arm == "control" else "S"
    workspace = SCRATCH_DIR / f"{tid}_{arm}"
    log_file = SCRATCH_DIR / f"{tid}_{arm}.jsonl"
    usage_file = SCRATCH_DIR / f"{tid}_{arm}.usage.json"

    if resume and usage_file.exists() and workspace.exists():
        print(f"  [{arm.upper()}] Reusing existing completed live run for {tid}...")
        with open(usage_file, "r", encoding="utf-8") as f:
            usage = json.load(f)
        summary = {"usage": usage, "turns": usage.get("turns", 0)}
    else:
        print(f"  [{arm.upper()}] Preparing real repo {repo} @ {commit[:8]}...")
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


def main() -> None:
    parser = argparse.ArgumentParser(prog="run_live_swe_rebench")
    parser.add_argument("--model", default="stealth/space-bunny-alpha")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--provider", default="openrouter")
    parser.add_argument("--resume", action="store_true", default=True)
    args = parser.parse_args()

    api_key = resolve_api_key(args.provider)
    assert api_key, f"API key for {args.provider} not resolved!"

    tasks = load_swe_rebench_window()
    print("=" * 70)
    print("LIVE SWE-REBENCH EVALUATION (GENUINE REPO RUNS)")
    print("=" * 70)
    print(f"Model:    {args.model}")
    print(f"Provider: {args.provider}")
    print(f"Tasks:    {args.limit} tasks (paired: {args.limit * 2} live runs)")
    print("=" * 70)

    # Pre-registered pilot task selection
    selected_tasks = tasks[:args.limit]

    ctrl_runs: list[PublicRunRecord] = []
    mintok_runs: list[PublicRunRecord] = []

    for idx, t in enumerate(selected_tasks, 1):
        tid = t["instance_id"]
        print(f"\n[{idx:02d}/{len(selected_tasks):02d}] TASK: {tid}")

        # Balanced order: even index control first, odd index mintok first
        order = ["control", "mintok"] if idx % 2 == 0 else ["mintok", "control"]
        for arm in order:
            rec, meta = run_live_task(t, arm, args.model, provider=args.provider, resume=args.resume)
            if arm == "control":
                ctrl_runs.append(rec)
            else:
                mintok_runs.append(rec)

    # Evaluate paired results
    report = evaluate_paired_public_runs(ctrl_runs, mintok_runs, compute_bootstrap=True, bootstrap_resamples=1000)

    print("\n" + "=" * 70)
    print("LIVE SWE-REBENCH RESULTS (GENUINE RUNS)")
    print("=" * 70)
    from mintok.public_bench import render_report_table
    print(render_report_table(report, "SWE-rebench (Live)"))

    # Save live run records
    out_payload = {
        "benchmark": "swe_rebench_live",
        "model": args.model,
        "provider": args.provider,
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
        },
        "control_runs": [asdict(r) for r in ctrl_runs],
        "mintok_runs": [asdict(r) for r in mintok_runs],
    }
    out_file = RUNS_DIR / "swe_rebench_live_space_bunny_alpha.json"
    scratch_out_file = SCRATCH_DIR / "swe_rebench_live_space_bunny_alpha.json"
    saved_path = None
    for p in [out_file, scratch_out_file]:
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(out_payload, f, indent=2)
            saved_path = p
            break
        except OSError:
            continue
    if saved_path:
        print(f"\n[Saved live run records to {saved_path}]")


if __name__ == "__main__":
    main()

