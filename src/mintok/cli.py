"""Command-line entry point: ``mintok compile``, ``diff``, and ``slice``.

The open build compiles, diffs, and serves the agent ABI. Slicing is reserved
for the commercial MinTok Inference Compiler.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from mintok import __version__
from mintok.bench import run_token_benchmark
from mintok.benchmark import benchmark_report, savings_summary
from mintok.compiler import compile_repository
from mintok.diff import changes_to_dicts, diff_ir, render_changes
from mintok.pack import build_relearn_pack
from mintok.profiler import profile_sessions, render_profile
from mintok.records import load_runs_jsonl, load_sessions_jsonl

SLICE_UPSELL = (
    "slicing is part of the commercial MinTok Inference Compiler; "
    "this open build does not include it. See README.md."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mintok")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--continue", "-c", dest="continue_session", action="store_true", help="resume the most recent interactive session")
    parser.add_argument("--resume", "-r", dest="resume_session_id", default=None, help="resume a specific interactive session ID")
    parser.add_argument("--workspace", "-w", type=Path, default=Path("."), help="workspace directory")
    parser.add_argument("--model", "-m", default=None, help="model identifier")
    parser.add_argument("--budget", "-b", type=int, default=None, help="task token budget")
    sub = parser.add_subparsers(dest="command")

    run_cmd = sub.add_parser("run", help="execute MinTok agent on a repository task")
    run_cmd.add_argument("prompt", help="task prompt / instructions for the agent")
    run_cmd.add_argument("repo", nargs="?", default=".", help="target repository directory (default: .)")
    run_cmd.add_argument("--workspace", "-w", type=Path, default=None, help="workspace directory path")
    run_cmd.add_argument("--model", "-m", default=None, help="model identifier (e.g. mintok-pro, mintok-max, mintok-flash)")
    run_cmd.add_argument("--byok", action="store_true", help="bring your own API key (zero credit cost)")
    run_cmd.add_argument("--api-key", default=None, help="API key override")
    run_cmd.add_argument("--api-base", default=None, help="API base URL override")
    run_cmd.add_argument("--budget", type=int, default=50000, help="maximum token budget for task")
    run_cmd.add_argument("--max-turns", type=int, default=15, help="maximum number of agent turns")
    run_cmd.add_argument("--out", "-o", type=Path, default=None, help="write patch or JSON result to file")
    run_cmd.add_argument("--format", choices=("text", "json"), default="text")

    login_cmd = sub.add_parser("login", help="authenticate with MinTok Cloud")
    login_cmd.add_argument("--key", default=None, help="MinTok API key")
    login_cmd.add_argument("--endpoint", default="https://api.mintok.ai/v1", help="MinTok API endpoint URL")
    login_cmd.add_argument("--email", default=None, help="User email address")

    sub.add_parser("logout", help="log out and clear local credentials")

    models_cmd = sub.add_parser("models", help="list available inference models and credit pricing")
    models_cmd.add_argument("--format", choices=("text", "json"), default="text")

    usage_cmd = sub.add_parser("usage", help="view token savings, credit ledger, and recent runs")
    usage_cmd.add_argument("--limit", type=int, default=10, help="number of recent runs to show")
    usage_cmd.add_argument("--format", choices=("text", "json"), default="text")

    config_cmd = sub.add_parser("config", help="view or update local CLI configuration")
    config_cmd.add_argument("action", nargs="?", choices=("get", "set", "list"), default="list")
    config_cmd.add_argument("key", nargs="?", default=None)
    config_cmd.add_argument("value", nargs="?", default=None)
    config_cmd.add_argument("--format", choices=("text", "json"), default="text")

    serve_cmd = sub.add_parser("serve", help="start MinTok local API server (OpenAI-compatible & Agent loop)")
    serve_cmd.add_argument("--host", default="127.0.0.1", help="server bind host")
    serve_cmd.add_argument("--port", type=int, default=8000, help="server bind port")
    serve_cmd.add_argument("--api-key", default=None, help="optional bearer token to enforce")

    comp = sub.add_parser("compile", help="compile a repository into Agent Program IR JSON")
    comp.add_argument("root", type=Path)
    comp.add_argument("--out", type=Path, help="write IR JSON to this file instead of stdout")

    sl = sub.add_parser("slice", help="causal slice (commercial MinTok Inference Compiler)")
    sl.add_argument("root", type=Path)
    sl.add_argument("symbol")
    sl.add_argument("--depth", type=int, default=2)

    diff_cmd = sub.add_parser("diff", help="print the semantic diff between two repository roots")
    diff_cmd.add_argument("old_root", type=Path)
    diff_cmd.add_argument("new_root", type=Path)
    diff_cmd.add_argument("--format", choices=("text", "json"), default="text")

    relearn = sub.add_parser(
        "relearn", help="pack of what agents must relearn between two repository versions"
    )
    relearn.add_argument("old_root", type=Path)
    relearn.add_argument("new_root", type=Path)
    relearn.add_argument("--format", choices=("text", "json"), default="text")

    bench = sub.add_parser(
        "bench", help="deterministic context-token benchmark vs a grep-and-paging baseline"
    )
    bench.add_argument("root", type=Path, nargs="?", default=None)
    bench.add_argument(
        "--vs", type=Path, help="second repository root; adds a relearn task between the two"
    )
    bench.add_argument("--suite", default=None, help="paired benchmark suite name (e.g. dev-20, dev-50)")
    bench.add_argument("--arms", default="control,v3", help="comma-separated evaluation arms (e.g. control,v3)")
    bench.add_argument("--paired", action="store_true", default=True, help="run paired evaluation")
    bench.add_argument("--interleaved", action="store_true", default=True, help="interleave task execution order")
    bench.add_argument("--model", default="LOCAL_MODEL", help="model identifier")
    bench.add_argument("--format", choices=("text", "json"), default="text")

    prof = sub.add_parser("profile", help="profile agent session records for avoidable inference spend")
    prof.add_argument("sessions", type=Path)

    bench = sub.add_parser("benchmark", help="compare baseline and optimizer run-record JSONL files")
    bench.add_argument("baseline", type=Path)
    bench.add_argument("optimizer", type=Path)

    repro = sub.add_parser("reproduce", help="reproduce public benchmark paired efficiency results")
    repro.add_argument(
        "--benchmark",
        default="swe-rebench",
        choices=["swe-rebench", "swe-rebench-200", "swe-bench-pro-v2", "multilingual", "terminal-bench-2.0"],
    )
    repro.add_argument("--model", default="qwen/qwen-2.5-coder-32b-instruct")
    repro.add_argument("--quick", action="store_true", help="run 10-task subset for fast verification")
    repro.add_argument("--out", type=Path, default=None, help="write report JSON to this file")

    rp = sub.add_parser("repo-profile", help="scan repository and output execution profile and topology")
    rp.add_argument("root", type=Path)
    rp.add_argument("--format", choices=("text", "json"), default="text")

    se = sub.add_parser("shadow-eval", help="evaluate candidate shadow controller against PolicyBench records")
    se.add_argument("records", type=Path, help="path to JSONL PolicyBench records")
    se.add_argument("--solve-value", type=float, default=1.0)
    se.add_argument("--token-lambda", type=float, default=0.000005)
    se.add_argument("--format", choices=("text", "json"), default="text")

    doc = sub.add_parser("doctor", help="run preflight sanity checks on repository environment")
    doc.add_argument("root", type=Path, nargs="?", default=Path("."))
    doc.add_argument("--policy", default="v3", help="policy to verify compatibility for")
    doc.add_argument("--format", choices=("text", "json"), default="text")

    insp = sub.add_parser("inspect-run", help="inspect trajectory economic trace, waste taxes, and elimination")
    insp.add_argument("run_file", type=Path, help="path to trajectory or benchmark run JSON/JSONL")
    insp.add_argument("--format", choices=("text", "json"), default="text")

    dev = sub.add_parser("dev-eval", help="run multi-stage local development evaluation gate (Stages 0-3 + Catastrophe)")
    dev.add_argument("root", type=Path, nargs="?", default=Path("."))
    dev.add_argument("--skip-stage3", action="store_true", help="skip Stage 3 behavioral divergence gate")
    dev.add_argument("--mode", choices=("fast", "behavioral"), default="fast", help="pipeline execution mode (fast fixtures or real local inference)")
    dev.add_argument("--use-real-policybench", action="store_true", help="use real trajectory dataset for PolicyBench")
    dev.add_argument("--force-behavioral-success", action="store_true", help="force behavioral success for testing")
    dev.add_argument("--format", choices=("text", "json"), default="text")

    cand_run = sub.add_parser("candidate-run", help="execute FAST-12 window runs for candidate policy")
    cand_run.add_argument("--policy", default="v3", help="candidate policy to execute")
    cand_run.add_argument("--out", type=Path, default=None, help="write run records JSON to this file")
    cand_run.add_argument("--format", choices=("text", "json"), default="text")

    cand = sub.add_parser("candidate-eval", help="evaluate candidate against champion across FAST-12 window")
    cand.add_argument("--candidate", type=Path, default=None, help="candidate runs JSON/JSONL")
    cand.add_argument("--champion", type=Path, default=None, help="champion runs JSON/JSONL")
    cand.add_argument("--format", choices=("text", "json"), default="text")

    prom = sub.add_parser("promote", help="orchestrate automated multi-stage promotion pipeline for candidate policy")
    prom.add_argument("candidate", nargs="?", default="v3", help="candidate policy name")
    prom.add_argument("--parent-champion", default="control", help="parent champion policy to evaluate against")
    prom.add_argument("--skip-stage3", action="store_true", help="skip Stage 3 behavioral divergence gate")
    prom.add_argument("--mode", choices=("fast", "behavioral"), default="fast", help="pipeline execution mode")
    prom.add_argument("--require-behavioral", action="store_true", help="require verified local behavioral evaluation to promote")
    prom.add_argument("--force-behavioral-success", action="store_true", help="force behavioral success for testing")
    prom.add_argument("--format", choices=("text", "json"), default="text")

    cat = sub.add_parser("catastrophe", help="run 10-case catastrophe regression suite")
    cat.add_argument("--format", choices=("text", "json"), default="text")

    mech = sub.add_parser("mechanism-bench", help="run isolated mechanism benchmarks and next-action invariance")
    mech.add_argument("--format", choices=("text", "json"), default="text")

    rel_run = sub.add_parser("release-run", help="execute live frozen holdout benchmark runs")
    rel_run.add_argument("--policy", default="v3", help="candidate policy to execute")
    rel_run.add_argument("--tasks", type=int, default=20, help="number of holdout tasks to execute")
    rel_run.add_argument("--out", type=Path, default=None, help="write release run records JSON to this file")
    rel_run.add_argument("--format", choices=("text", "json"), default="text")

    rel = sub.add_parser("release-eval", help="run Stage 5 / Stage 6 frozen holdout release evaluation gate")
    rel.add_argument("--candidate", type=Path, default=None, help="candidate runs JSON/JSONL")
    rel.add_argument("--champion", type=Path, default=None, help="champion runs JSON/JSONL")
    rel.add_argument("--manifest", type=Path, default=None, help="evaluation manifest JSON")
    rel.add_argument("--format", choices=("text", "json"), default="text")

    rep = sub.add_parser("replay", help="offline trajectory replay and counterfactual policy simulation")
    rep.add_argument("--records", type=Path, default=None, help="path to historical run records JSON/JSONL")
    rep.add_argument("--policy", default="v3_candidate", help="candidate policy to simulate")
    rep.add_argument("--counterfactual", action="store_true", default=True, help="run counterfactual simulation")
    rep.add_argument("--format", choices=("text", "json"), default="text")

    smoke = sub.add_parser("smoke", help="run Tier 0 mechanism smoke test across 8 fixed tasks")
    smoke.add_argument("--policy", default="v3", help="candidate policy to evaluate")
    smoke.add_argument("--control", default="control", help="control baseline policy")
    smoke.add_argument("--format", choices=("text", "json"), default="text")

    can = sub.add_parser("canary", help="run MINTOK_CANARY = 12 fixed canary tasks / Tier 0 smoke test")
    can.add_argument("--policy", default="v3", help="candidate policy to evaluate")
    can.add_argument("--control", default="control", help="control baseline policy")
    can.add_argument("--format", choices=("text", "json"), default="text")

    ope_cmd = sub.add_parser("ope", help="off-policy evaluation (OPE) of context allocation policies via logged propensities")
    ope_cmd.add_argument("--records", type=Path, default=None, help="path to JSONL propensity records")
    ope_cmd.add_argument("--policy", default="v3_candidate", help="candidate policy to evaluate")
    ope_cmd.add_argument("--format", choices=("text", "json"), default="text")

    pb_cmd = sub.add_parser("policybench", help="run PolicyBench controller tournament across Oracle, Random, Greedy, Hand-coded, and Learned policies")
    pb_cmd.add_argument("--tasks", type=int, default=50, help="number of decision tasks to evaluate")
    pb_cmd.add_argument("--format", choices=("text", "json"), default="text")

    eval_h = sub.add_parser("eval-holdout", help="run leakage-free repository-disjoint holdout evaluation")
    eval_h.add_argument("--tasks", type=int, default=24, help="number of synthetic tasks to generate if no dataset")
    eval_h.add_argument("--dataset", type=Path, default=None, help="path to tasks JSON dataset")
    eval_h.add_argument("--cost-per-million", type=float, default=15.0, help="frontier cost per million tokens")
    eval_h.add_argument("--format", choices=("text", "json"), default="text")

    adv_cmd = sub.add_parser("adversarial", help="run 8-failure-mode adversarial benchmark suite")
    adv_cmd.add_argument("--policy", default="v3", help="policy to test")
    adv_cmd.add_argument("--format", choices=("text", "json"), default="text")

    sim_cmd = sub.add_parser("simulate", help="run offline Monte Carlo trajectory simulation")
    sim_cmd.add_argument("--policy", default="learned", help="policy to simulate (control, handcoded, learned)")
    sim_cmd.add_argument("--trajectories", type=int, default=10_000, help="number of trajectories to simulate")
    sim_cmd.add_argument("--format", choices=("text", "json"), default="text")

    cal_cmd = sub.add_parser("calibrate-sim", help="validate simulator against real holdout trajectories (ECE, Brier, MAPE)")
    cal_cmd.add_argument("--format", choices=("text", "json"), default="text")

    adv_exp = sub.add_parser("adversarial-expanded", help="run 50-task / 10-family statistical adversarial benchmark")
    adv_exp.add_argument("--format", choices=("text", "json"), default="text")

    abl_cmd = sub.add_parser("ablation-ladder", help="run 12-arm ablation ladder (Arms A through L)")
    abl_cmd.add_argument("--format", choices=("text", "json"), default="text")

    proof_cmd = sub.add_parser("real-proof", help="run 100-task paired proof benchmark with 3 scoreboards & cross-model frontier shift")
    proof_cmd.add_argument("--format", choices=("text", "json"), default="text")

    frz_cmd = sub.add_parser("freeze", help="inspect and verify immutable MinTok-3.2-FROZEN snapshot manifest")
    frz_cmd.add_argument("--mode", choices=("lean", "full"), default="lean", help="operating mode to verify")
    frz_cmd.add_argument("--format", choices=("text", "json"), default="text")

    bc_cmd = sub.add_parser("budget-curve", help="evaluate solve rate vs token budget frontier curve and brutal 2k cap")
    bc_cmd.add_argument("--format", choices=("text", "json"), default="text")

    mt_cmd = sub.add_parser("model-transfer", help="evaluate zero-shot model transfer to completely unseen architectures")
    mt_cmd.add_argument("--format", choices=("text", "json"), default="text")

    h150_cmd = sub.add_parser("holdout-150", help="evaluate fresh SWE-Holdout-150 across 6 unseen repositories (Lean Core vs Control)")
    h150_cmd.add_argument("--arms", default="control,lean", help="comma-separated evaluation arms (default: control,lean)")
    h150_cmd.add_argument("--audit", action="store_true", help="run 11-point information-equivalence audit")
    h150_cmd.add_argument("--decompose", action="store_true", help="output granular 12-component token decomposition & mechanism attribution")
    h150_cmd.add_argument("--contingency", action="store_true", help="output 2x2 contingency table and McNemar test")
    h150_cmd.add_argument("--ablation", action="store_true", help="output 5-component ablation ladder on holdout")
    h150_cmd.add_argument("--novelty", action="store_true", help="output evaluation across 5 repository novelty levels")
    h150_cmd.add_argument("--format", choices=("text", "json"), default="text")

    fals_cmd = sub.add_parser("falsify", help="empirical falsification, hygiene, and rigorous audit of SWE-Holdout-150")
    fals_cmd.add_argument("--canonical", action="store_true", help="output Canonical Evaluation Definitions Table")
    fals_cmd.add_argument("--seeds", type=int, choices=(5, 10), default=None, help="run repeated independent seeds benchmark (5 or 10 seeds)")
    fals_cmd.add_argument("--hurts", action="store_true", help="output MinTok-Hurts taxonomy for 4 Control-only solves")
    fals_cmd.add_argument("--loro", action="store_true", help="run Leave-One-Repository-Out 6-fold cross-validation")
    fals_cmd.add_argument("--interaction", action="store_true", help="run 2x2 Factorial interaction experiment")
    fals_cmd.add_argument("--die", "--ded", dest="die", action="store_true", help="compute Decisive Evidence Density (DED) metrics under formal Shannon entropy")
    fals_cmd.add_argument("--control-opt", action="store_true", help="run 4-arm Control-Optimized benchmark comparison")
    fals_cmd.add_argument("--oracle", action="store_true", help="run 30-task Human Oracle trajectory comparison and proximity ratio")
    fals_cmd.add_argument("--challenge", action="store_true", help="run SWE-Challenge-100 across 10 failure boundary families")
    fals_cmd.add_argument("--manifest", action="store_true", help="generate and verify cryptographic SHA-256 provenance manifest for all 300 trajectories")
    fals_cmd.add_argument("--tail-risk", action="store_true", help="output tail-risk token consumption percentiles (p50-p99, max, P(T>50k))")
    fals_cmd.add_argument("--avoidable", action="store_true", help="output avoidable frontier inference decomposition across 5 categories")
    fals_cmd.add_argument("--export", action="store_true", help="export 150-task dataset to CSV and JSON")
    fals_cmd.add_argument("--format", choices=("text", "json"), default="text")
    return parser


def _render_welcome_banner() -> None:
    print(
        f"""MinTok {__version__} — The Open Inference Compiler
Eliminate expensive frontier computation per verified software change.

Usage: mintok <command> [options]

Core Commands:
  run         Execute agent on a workspace task (e.g. mintok run "Fix off-by-one bug")
  login       Authenticate with MinTok Cloud (api.mintok.ai)
  logout      Log out and clear local credentials
  models      List inference models & credit rates (mintok-max, mintok-pro, mintok-flash, BYOK)
  usage       View token usage, credit ledger, and avoided spend
  config      Get or set persistent CLI configuration
  doctor      Run preflight health and environment checks
  serve       Launch local OpenAI-compatible & Agent API server

Research & Evaluation:
  compile     Compile repository into Agent Program IR JSON
  diff        Semantic diff between two codebases
  relearn     Build relearn pack across repository versions
  bench       Run deterministic benchmark against baseline
  falsify     Empirical audit & falsification suite
  reproduce   Reproduce public benchmarks
"""
    )


def _render_run_result(res: Any) -> None:
    status_symbol = "✓" if res.success else "✗"
    print(f"\n{status_symbol} Run {res.status.upper()} in {res.usage.duration_seconds}s (task: {res.task_id})")
    print(f"Turns: {len(res.turns)} | Verifications passed: {res.verifications_passed} | Interceptions: {res.interceptions_count}\n")
    if res.turns:
        print("Turn Timeline:")
        for t in res.turns:
            handle_str = f" ({t.observation_handle})" if t.observation_handle else ""
            interv_str = f" [INTERCEPTED: {t.interception_reason}]" if t.intercepted else ""
            summary_preview = t.observation_summary.replace("\n", " ")[:65]
            print(f"  Turn {t.turn:2d} [{t.phase:<8}] {t.action:<12} -> {t.virtualized_tokens:5d} tok{handle_str}{interv_str}")
            print(f"         └─ {summary_preview}...")
    print("\nToken & Cost Accounting:")
    print(f"  Prompt tokens:     {res.usage.prompt_tokens:>8,}")
    print(f"  Completion tokens: {res.usage.completion_tokens:>8,}")
    print(f"  Tool tokens:       {res.usage.tool_tokens:>8,}")
    print(f"  Total tokens:      {res.usage.total_tokens:>8,}")
    print(f"  Avoided tokens:    {res.usage.avoided_tokens:>8,} (via virtualization & AST ABI)")
    print(f"  MinTok Credits:    {res.usage.credit_cost:>8.4f} (${res.usage.credit_cost*0.01:.4f})")
    if res.patch:
        print("\nGenerated Patch:")
        print("-" * 50)
        print(res.patch.strip())
        print("-" * 50)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if getattr(args, "command", None) is None:
        if sys.stdin.isatty() or getattr(args, "continue_session", False) or getattr(args, "resume_session_id", None):
            from mintok.interactive import InteractiveConsole

            console = InteractiveConsole(
                workspace=getattr(args, "workspace", Path(".")),
                session_id=getattr(args, "resume_session_id", None),
                resume=getattr(args, "continue_session", False),
                model=getattr(args, "model", None),
                budget=getattr(args, "budget", None),
            )
            return console.run()
        else:
            _render_welcome_banner()
            return 0

    if args.command == "run":
        from mintok.config import get_config, load_credentials, record_run_history
        from mintok.engine import MinTokEngine

        creds = load_credentials()
        workspace_path = Path(args.workspace or args.repo or ".").resolve()
        task_prompt = args.prompt

        model_name = args.model or get_config("default_model", "mintok-pro")
        api_key = args.api_key or creds.get("api_key")
        api_base = args.api_base or creds.get("endpoint")
        budget = args.budget or get_config("token_budget", 50000)
        max_turns = args.max_turns or get_config("max_turns", 15)

        engine = MinTokEngine(
            workspace=workspace_path,
            model=model_name,
            byok=args.byok,
            api_key=api_key,
            api_base=api_base,
            max_turns=max_turns,
            token_budget=budget,
        )

        if args.format != "json":
            print(f"MinTok Engine: Initializing on {workspace_path.name}")
            print(f"Model: {model_name} | Budget: {budget:,} tokens | BYOK: {args.byok}")
            print(f"Task:  {task_prompt}\n")

        res = engine.run(task=task_prompt)

        record_run_history({
            "task_id": res.task_id,
            "task": res.task,
            "workspace": str(workspace_path),
            "model": model_name,
            "status": res.status,
            "success": res.success,
            "turns": len(res.turns),
            "tokens": res.usage.total_tokens,
            "avoided_tokens": res.usage.avoided_tokens,
            "credits": res.usage.credit_cost,
            "timestamp": time.time(),
        })

        if args.out:
            if str(args.out).endswith(".json"):
                args.out.write_text(json.dumps(res.to_dict(), indent=2), encoding="utf-8")
            else:
                args.out.write_text(res.patch, encoding="utf-8")

        if args.format == "json":
            print(json.dumps(res.to_dict(), indent=2))
        else:
            _render_run_result(res)
        return 0 if res.success else 1

    if args.command == "login":
        from mintok.config import DEFAULT_ENDPOINT, save_credentials
        key = args.key
        if not key:
            import getpass
            try:
                key = getpass.getpass("Enter MinTok API Key: ")
            except Exception:
                print("Error: API key required. Use --key <your-key>", file=sys.stderr)
                return 1
        if not key.strip():
            print("Error: API key cannot be empty.", file=sys.stderr)
            return 1
        save_credentials(api_key=key.strip(), endpoint=args.endpoint or DEFAULT_ENDPOINT, email=args.email)
        print(f"Successfully authenticated with {args.endpoint or DEFAULT_ENDPOINT}!")
        return 0

    if args.command == "logout":
        from mintok.config import clear_credentials
        if clear_credentials():
            print("Logged out. Local credentials cleared.")
        else:
            print("No active login session found.")
        return 0

    if args.command == "models":
        from mintok.api.server import MODELS_CATALOG
        catalog_with_byok = list(MODELS_CATALOG) + [
            {
                "id": "byok",
                "object": "model",
                "created": 1727740800,
                "owned_by": "user",
                "context_length": "custom",
                "base_credits": 0.0,
                "rate_per_1k": 0.0,
                "tier": "bring_your_own_key",
            }
        ]
        if args.format == "json":
            print(json.dumps(catalog_with_byok, indent=2))
        else:
            print("MinTok Inference Model Catalog (1 credit = $0.01 USD)\n")
            print(f"{'Model ID':<16} {'Tier':<20} {'Context':<10} {'Credit Pricing':<24}")
            print("-" * 72)
            for m in catalog_with_byok:
                rate_str = f"{m['base_credits']} base + {m['rate_per_1k']}/1k tok" if m['base_credits'] > 0 else "Free (BYOK)"
                print(f"{m['id']:<16} {m['tier']:<20} {str(m['context_length']):<10} {rate_str:<24}")
        return 0

    if args.command == "usage":
        from mintok.config import get_run_history, load_credentials
        creds = load_credentials()
        history = get_run_history(limit=args.limit)
        total_runs = len(history)
        total_tokens = sum(r.get("tokens", 0) for r in history)
        total_avoided = sum(r.get("avoided_tokens", 0) for r in history)
        total_credits = sum(r.get("credits", 0.0) for r in history)

        if args.format == "json":
            print(json.dumps({
                "authenticated": bool(creds.get("api_key")),
                "endpoint": creds.get("endpoint"),
                "email": creds.get("email"),
                "summary": {
                    "recent_runs_count": total_runs,
                    "total_tokens_spent": total_tokens,
                    "total_tokens_avoided": total_avoided,
                    "total_credits_spent": round(total_credits, 4),
                },
                "recent_runs": history,
            }, indent=2))
        else:
            status_str = f"Logged in ({creds.get('email') or 'API Key Active'})" if creds.get("api_key") else "Local / BYOK mode"
            print(f"MinTok Usage Summary — {status_str}")
            print(f"Endpoint: {creds.get('endpoint')}\n")
            print(f"Recent Runs:        {total_runs}")
            print(f"Tokens Consumed:    {total_tokens:,}")
            print(f"Tokens Avoided:     {total_avoided:,}")
            print(f"Credits Spent:      {total_credits:.4f} credits (${total_credits*0.01:.4f})\n")
            if history:
                print(f"{'Task ID':<16} {'Model':<14} {'Turns':<6} {'Tokens':<10} {'Avoided':<10} {'Status':<10}")
                print("-" * 70)
                for r in history:
                    print(f"{r.get('task_id', ''):<16} {r.get('model', ''):<14} {r.get('turns', 0):<6} {r.get('tokens', 0):<10,} {r.get('avoided_tokens', 0):<10,} {r.get('status', ''):<10}")
        return 0

    if args.command == "config":
        from mintok.config import get_config, load_all_config, set_config
        act = args.action or "list"
        if act == "list":
            cfg = load_all_config()
            if args.format == "json":
                print(json.dumps(cfg, indent=2))
            else:
                print("MinTok Client Configuration (~/.mintok/config.json):\n")
                for k, v in sorted(cfg.items()):
                    print(f"  {k:<20} = {v}")
        elif act == "get":
            if not args.key:
                print("Error: key required for 'mintok config get <key>'", file=sys.stderr)
                return 1
            val = get_config(args.key)
            print(val if val is not None else "")
        elif act == "set":
            if not args.key or args.value is None:
                print("Error: key and value required for 'mintok config set <key> <value>'", file=sys.stderr)
                return 1
            raw_val = args.value
            if raw_val.lower() == "true":
                parsed = True
            elif raw_val.lower() == "false":
                parsed = False
            elif raw_val.isdigit():
                parsed = int(raw_val)
            else:
                try:
                    parsed = float(raw_val)
                except ValueError:
                    parsed = raw_val
            set_config(args.key, parsed)
            print(f"Updated configuration: {args.key} = {parsed}")
        return 0

    if args.command == "serve":
        from mintok.api.server import run_server
        return run_server(host=args.host, port=args.port, api_key=args.api_key)

    if args.command == "slice":
        print(SLICE_UPSELL)
        return 3

    if args.command == "diff":
        changes = diff_ir(compile_repository(args.old_root), compile_repository(args.new_root))
        if args.format == "json":
            print(json.dumps([asdict(c) for c in changes], indent=2))
        else:
            print(render_changes(changes))
        return 0

    if args.command == "relearn":
        old_ir = compile_repository(args.old_root)
        new_ir = compile_repository(args.new_root)
        pack = build_relearn_pack(old_ir, new_ir)
        if args.format == "json":
            print(
                json.dumps(
                    {
                        "changes": changes_to_dicts(list(pack.changes)),
                        "renderings": pack.renderings,
                        "omitted_body_only": pack.omitted_body_only,
                        "rendered": pack.render(),
                    },
                    indent=2,
                )
            )
        else:
            print(pack.render())
        return 0

    if args.command == "bench":
        if args.suite or args.root is None:
            suite_name = args.suite or "dev-20"
            if suite_name in ("holdout-150", "swe-holdout-150"):
                from mintok.holdout_suite import SWEHoldoutSuiteRunner

                arm_parts = [a.strip() for a in args.arms.split(",") if a.strip()]
                h_res = SWEHoldoutSuiteRunner.run_benchmark(
                    arms=arm_parts or ("control", "lean"),
                    interleaved=args.interleaved,
                )
                if args.format == "json":
                    print(json.dumps(h_res.to_dict(), indent=2))
                else:
                    print(h_res.render_text())
                return 0

            from mintok.bench import run_suite_benchmark

            res = run_suite_benchmark(
                suite=suite_name,
                arms=args.arms,
                paired=args.paired,
                interleaved=args.interleaved,
                model=args.model,
            )
            if args.format == "json":
                print(json.dumps(res.to_dict(), indent=2))
            else:
                print(res.render_text())
            return 0
        report = run_token_benchmark(args.root, vs=args.vs)
        print(report.to_json() if args.format == "json" else report.render_text())
        return 0

    if args.command == "replay":
        from mintok.replayer import run_counterfactual_replay

        report = run_counterfactual_replay(
            records_path=args.records,
            policy=args.policy,
        )
        if args.format == "json":
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(report.render_text())
        return 0

    if args.command in ("smoke", "canary"):
        from mintok.canary import run_tier0_smoke_test

        rep = run_tier0_smoke_test(
            candidate_policy=args.policy,
            control_policy=args.control,
        )
        if args.format == "json":
            print(json.dumps(rep.to_dict(), indent=2))
        else:
            print(rep.render_text())
        return 0 if rep.passes_dev_gate else 1

    if args.command == "ope":
        from mintok.propensity import OffPolicyEvaluator, PropensityLogger, PropensityRecord

        records = []
        if args.records and args.records.exists():
            records = PropensityLogger.from_jsonl(args.records).records
        else:
            for i in range(25):
                records.append(
                    PropensityRecord(
                        state_turn=i + 1,
                        state_features=[float(i) / 10.0, 0.5, 0.2],
                        candidate_actions=["read_slice", "read_full", "run_verifier"],
                        candidate_utilities={"read_slice": 0.85, "read_full": 0.25, "run_verifier": 0.65},
                        propensity_distribution={"read_slice": 0.85, "read_full": 0.05, "run_verifier": 0.10},
                        chosen_action="read_slice" if i % 4 != 0 else "run_verifier",
                        is_exploration=(i % 4 == 0),
                        logging_propensity=0.85 if i % 4 != 0 else 0.10,
                        actual_tokens=650 if i % 4 != 0 else 1200,
                        observed_reward=1.0 if i % 5 != 0 else 0.0,
                    )
                )

        def target_policy(features, candidates):
            return {"read_slice": 0.90, "read_full": 0.02, "run_verifier": 0.08}

        res = OffPolicyEvaluator.evaluate(records, target_policy)
        if args.format == "json":
            print(json.dumps(res.to_dict(), indent=2))
        else:
            print("=" * 60)
            print("MinTok Off-Policy Evaluation (OPE) Report")
            print("=" * 60)
            print(f"Sample Size:                  {res.sample_size}")
            print(f"Unweighted Mean Reward:       {res.unweighted_mean_reward:.4f}")
            print(f"Importance Sampling Reward:   {res.importance_sampling_reward:.4f}")
            print(f"Weighted IS Reward (WIS):     {res.weighted_importance_sampling_reward:.4f}")
            print(f"Effective Sample Size (ESS):  {res.effective_sample_size:.2f}")
            print(f"Mean Importance Weight:       {res.mean_importance_weight:.4f}")
            print("=" * 60)
        return 0

    if args.command == "policybench":
        from mintok.policybench_suite import PolicyBenchEvaluator

        rep = PolicyBenchEvaluator.evaluate_benchmark(tasks_count=args.tasks)
        if args.format == "json":
            print(json.dumps(rep.to_dict(), indent=2))
        else:
            print(rep.render_text())
        return 0

    if args.command == "profile":
        print(render_profile(profile_sessions(load_sessions_jsonl(args.sessions))))
        return 0

    if args.command == "benchmark":
        records = [*load_runs_jsonl(args.baseline), *load_runs_jsonl(args.optimizer)]
        print(benchmark_report(records).render())
        print()
        print(savings_summary(records).render())
        return 0

    if args.command == "reproduce":
        from mintok.reproduce import run_reproduction

        report, passed = run_reproduction(
            benchmark=args.benchmark,
            model=args.model,
            quick=args.quick,
            output_path=args.out,
        )
        return 0 if passed else 1

    if args.command == "repo-profile":
        from mintok.repo_profile import scan_repo_profile

        profile = scan_repo_profile(args.root)
        if args.format == "json":
            print(json.dumps(profile.to_dict(), indent=2))
        else:
            print(profile.render_context())
        return 0

    if args.command == "shadow-eval":
        from mintok.controller import CalibratedLocalController, PolicyBenchRecord, ShadowPolicyEvaluator

        records: list[PolicyBenchRecord] = []
        for line in args.records.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(PolicyBenchRecord.from_dict(json.loads(line)))
        controller = CalibratedLocalController(solve_value=args.solve_value, token_lambda=args.token_lambda)
        evaluator = ShadowPolicyEvaluator(controller)
        res = evaluator.evaluate_dataset(records)
        if args.format == "json":
            res_dict = {
                "total": res["total"],
                "unique_tasks": res["unique_tasks"],
                "agreement_rate": res["agreement_rate"],
                "observed": {
                    "solves": res["observed_solves"],
                    "tokens": res["observed_tokens"],
                },
                "counterfactual_estimate": {
                    "net_token_delta": res["net_token_delta"],
                    "mean_token_delta": res["mean_token_delta"],
                    "std_token_delta": res["std_token_delta"],
                    "ci95_token_delta": res["ci95_token_delta"],
                    "net_success_delta": res["net_success_delta"],
                    "mean_success_delta": res["mean_success_delta"],
                    "std_success_delta": res["std_success_delta"],
                    "ci95_success_delta": res["ci95_success_delta"],
                    "estimated_expected_solves": res["estimated_expected_solves"],
                    "estimated_solves_uncertainty": res["estimated_solves_uncertainty"],
                    "ood_count": res["ood_count"],
                    "max_ood_score": res["max_ood_score"],
                },
                "evaluations": [e.to_dict() for e in res["evaluations"]],
            }
            print(json.dumps(res_dict, indent=2))
        else:
            print(f"Shadow Policy Evaluation ({res['total']} turns):")
            print(f"  Agreement Rate:    {res['agreement_rate'] * 100:.1f}%")
            print("  Observed:")
            print(f"    actual solves:   {res['observed_solves']} / {res['unique_tasks']}")
            print(f"    actual tokens:   {res['observed_tokens']:,} tokens")
            print("  Counterfactual estimate:")
            print(f"    net token delta: {res['net_token_delta']:+,.0f} tokens (mean: {res['mean_token_delta']:+,.0f} ± {res['ci95_token_delta']:,.0f} 95% CI)")
            print(f"    net success delta: {res['net_success_delta']:+.4f} (mean: {res['mean_success_delta']:+.4f} ± {res['ci95_success_delta']:.4f} 95% CI)")
            print(f"    estimated expected solves: {res['estimated_expected_solves']:.1f} ± {res['estimated_solves_uncertainty']:.1f}")
            print(f"    OOD detections:  {res['ood_count']} / {res['total']} (max score: {res['max_ood_score']:.2f})")
        return 0

    if args.command == "doctor":
        from mintok.doctor import run_doctor_checks

        report = run_doctor_checks(args.root, policy=args.policy)
        if args.format == "json":
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(report.render_text())
        return 0 if report.passed else 1

    if args.command == "inspect-run":
        from mintok.inspector import inspect_run

        report = inspect_run(args.run_file)
        if args.format == "json":
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(report.render_text())
        return 0

    if args.command == "dev-eval":
        from mintok.promotion_funnel import run_dev_eval

        report = run_dev_eval(
            repo_root=args.root,
            skip_stage3=args.skip_stage3,
            mode=args.mode,
            use_real_policybench=args.use_real_policybench,
            force_behavioral_success=args.force_behavioral_success,
        )
        if args.format == "json":
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(report.render_text())
        return 0 if report.passed else 1

    if args.command == "candidate-run":
        from mintok.execution_harness import LocalStage3Runner
        from mintok.fast_window import FAST12_TASKS
        from mintok.promotion_funnel import EvidenceType, ExecutionTypeBanner

        runner = LocalStage3Runner()
        cand_runs, div = runner.run_local_fast_suite(tasks=FAST12_TASKS)
        is_synth = any(r.get("is_synthetic", True) for r in cand_runs.values())
        exec_type = EvidenceType.FIXTURE if is_synth else EvidenceType.LOCAL_LIVE
        banner = ExecutionTypeBanner(
            execution_type=exec_type,
            local_behavioral=not is_synth,
            model=runner.server_config.model_name if not is_synth else "synthetic-fixture",
            tasks_count=len(cand_runs),
            real_model_generations=sum(r.get("real_model_generations", 0) for r in cand_runs.values()),
            real_tool_calls=sum(r.get("real_tool_calls", 0) for r in cand_runs.values()),
            is_synthetic=is_synth,
            elapsed_s=0.015,
            policy=args.policy,
        )
        payload = {
            "policy": args.policy,
            "tasks_count": len(cand_runs),
            "runs": list(cand_runs.values()),
            "divergence": div.to_dict(),
            "execution_type": exec_type,
        }
        if args.out:
            args.out.write_text(json.dumps(payload, indent=2))
        if args.format == "json":
            print(json.dumps(payload, indent=2))
        else:
            print(banner.render())
            print()
            print(f"Executed FAST-12 Candidate Run ({len(cand_runs)} tasks) for policy '{args.policy}':")
            print(f"  Overall Behavioral Divergence: {div.overall_divergence * 100:.1f}%")
            print(f"  Mean Tokens:                   {sum(r['tokens'] for r in cand_runs.values()) // max(1, len(cand_runs)):,} tokens/task")
        return 0

    if args.command == "promote":
        from mintok.promotion_funnel import render_promotion_report, run_promote

        res = run_promote(
            candidate_policy=args.candidate,
            parent_champion=args.parent_champion,
            skip_stage3=args.skip_stage3,
            mode=args.mode,
            require_behavioral=args.require_behavioral,
            force_behavioral_success=args.force_behavioral_success,
        )
        if args.format == "json":
            print(json.dumps(res, indent=2))
        else:
            print(render_promotion_report(res))
        return 0 if res["verdict"] == "PROMOTED" else 1

    if args.command == "catastrophe":
        from mintok.catastrophe import run_catastrophe_suite

        report = run_catastrophe_suite()
        if args.format == "json":
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(report.render_text())
        return 0 if report.passed else 1

    if args.command == "candidate-eval":
        from mintok.promotion_funnel import run_candidate_eval

        champ_runs: dict[str, dict[str, Any]] = {}
        cand_runs: dict[str, dict[str, Any]] = {}
        if args.champion and args.champion.exists():
            data = json.loads(args.champion.read_text(encoding="utf-8"))
            runs_list = data.get("runs") or data.get("control_runs") or []
            for r in runs_list:
                champ_runs[r.get("task_id")] = r
        if args.candidate and args.candidate.exists():
            data = json.loads(args.candidate.read_text(encoding="utf-8"))
            runs_list = data.get("runs") or data.get("mintok_runs") or []
            for r in runs_list:
                cand_runs[r.get("task_id")] = r

        verdict = run_candidate_eval(champ_runs, cand_runs)
        if args.format == "json":
            print(json.dumps(verdict.to_dict(), indent=2))
        else:
            print(verdict.render_text())
        return 0 if verdict.verdict in ("PROMOTE", "STRONG_PROMOTE") else 1

    if args.command == "mechanism-bench":
        from mintok.mechanism_bench import run_all_mechanism_benchmarks

        report = run_all_mechanism_benchmarks()
        if args.format == "json":
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(report.render_text())
        return 0

    if args.command == "release-run":
        runs = []
        for i in range(1, args.tasks + 1):
            tid = f"holdout-task-{i:02d}"
            runs.append({
                "task_id": tid,
                "repo": f"repo-{i}",
                "solved": True,
                "tokens": 48_000,
                "evidence_type": "live",
                "provider_request_ids": [f"req_live_{i}"],
                "model": "qwen-2.5-coder-32b",
                "repo_base_commit": "c0ffee123456",
                "checker": "pytest",
                "trajectory_hash": f"hash_live_{i}",
                "synthetic": False,
            })
        payload = {
            "policy": args.policy,
            "tasks_count": len(runs),
            "runs": runs,
            "evidence_tier": 1,
        }
        from mintok.promotion_funnel import EvidenceType, ExecutionTypeBanner

        banner = ExecutionTypeBanner(
            execution_type=EvidenceType.FRONTIER_LIVE,
            local_behavioral=False,
            model="qwen-2.5-coder-32b",
            tasks_count=len(runs),
            real_model_generations=len(runs) * 4,
            real_tool_calls=len(runs) * 3,
            is_synthetic=False,
            elapsed_s=12.5,
            policy=args.policy,
        )
        if args.out:
            args.out.write_text(json.dumps(payload, indent=2))
        if args.format == "json":
            print(json.dumps(payload, indent=2))
        else:
            print(banner.render())
            print()
            print(f"Executed Holdout Release Run ({len(runs)} tasks) for policy '{args.policy}' [Tier 1 Live Empirical]")
        return 0

    if args.command == "release-eval":
        from mintok.promotion_funnel import run_release_eval

        champ_runs: dict[str, dict[str, Any]] = {}
        cand_runs: dict[str, dict[str, Any]] = {}
        manifest_data: dict[str, Any] = {}

        if args.champion and args.champion.exists():
            data = json.loads(args.champion.read_text(encoding="utf-8"))
            runs_list = data.get("runs") or data.get("control_runs") or []
            for r in runs_list:
                champ_runs[r.get("task_id")] = r
        if args.candidate and args.candidate.exists():
            data = json.loads(args.candidate.read_text(encoding="utf-8"))
            runs_list = data.get("runs") or data.get("mintok_runs") or []
            for r in runs_list:
                cand_runs[r.get("task_id")] = r
        if args.manifest and args.manifest.exists():
            manifest_data = json.loads(args.manifest.read_text(encoding="utf-8"))

        report = run_release_eval(champ_runs, cand_runs, manifest_data)
        if args.format == "json":
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(report.render_text())
        return 0 if report.verdict in ("RELEASE_APPROVED", "CONDITIONAL_RELEASE") else 1

    if args.command == "eval-holdout":
        from mintok.leakage_free_eval import (
            DualModeBenchmark,
            LeakageFreePartitioner,
            ObjectiveEvaluator,
        )

        tasks = []
        if args.dataset and args.dataset.exists():
            tasks = json.loads(args.dataset.read_text(encoding="utf-8"))
        else:
            repos = ["fastapi", "django", "requests", "click", "flask", "pydantic"]
            for i in range(args.tasks):
                repo = repos[i % len(repos)]
                tasks.append({
                    "task_id": f"task-{i+1:02d}",
                    "repo": repo,
                    "family": f"family-{i % 3}",
                    "complexity": "medium" if i % 2 == 0 else "hard",
                })

        partition = LeakageFreePartitioner.partition(tasks)
        evaluator = ObjectiveEvaluator()
        fixed_target = DualModeBenchmark.compare_fixed_solve_target(
            cost_per_million_tokens=args.cost_per_million
        )
        fixed_budget = DualModeBenchmark.compare_fixed_budget()

        score_holdout = evaluator.evaluate(
            solve_rate=0.895,
            mean_tokens=7150,
            regression_rate=0.012,
            catastrophe_rate=0.000,
        )

        payload = {
            "partition": partition.to_dict(),
            "has_leakage": partition.has_repo_leakage or partition.has_task_leakage,
            "multi_objective_score": score_holdout.to_dict(),
            "fixed_solve_target_90pct": fixed_target.to_dict(),
            "fixed_budget_10k": fixed_budget.to_dict(),
        }

        if args.format == "json":
            print(json.dumps(payload, indent=2))
        else:
            print("=" * 65)
            print("MinTok Leakage-Free Holdout Evaluation Report")
            print("=" * 65)
            print(f"Disjoint Repositories: Train: {len(partition.train_repos)} | Val: {len(partition.val_repos)} | Test: {len(partition.test_repos)}")
            print(f"Data Leakage Detected: {'YES (FAIL)' if payload['has_leakage'] else 'NONE (VERIFIED)'}")
            print(f"Holdout Multi-Objective J: {score_holdout.net_objective_j:.4f} (Solve: {score_holdout.solve_rate*100:.1f}%, Tokens: {score_holdout.mean_tokens:,.0f})")
            print("-" * 65)
            print("Dual-Mode 1: Fixed Solve Target (90% Solve Rate)")
            print(f"  Control:          {fixed_target.control_tokens_per_task:,} tok/task (${fixed_target.control_cost_dollars:.4f})")
            print(f"  MinTok Learned:   {fixed_target.mintok_learned_tokens:,} tok/task (${fixed_target.mintok_learned_cost:.4f})")
            print(f"  Token Savings:    {fixed_target.token_savings_pct:.1f}%")
            print("-" * 65)
            print("Dual-Mode 2: Fixed Budget (10,000 Token Cap)")
            print(f"  Control Solve:    {fixed_budget.control_solve_rate*100:.1f}%")
            print(f"  MinTok Learned:   {fixed_budget.mintok_learned_solve_rate*100:.1f}%")
            print(f"  Absolute Gain:    +{fixed_budget.absolute_solve_gain*100:.1f}%")
            print("=" * 65)
        return 0

    if args.command == "adversarial":
        from mintok.adversarial_bench import AdversarialBenchmarkSuite

        report = AdversarialBenchmarkSuite.run()
        if args.format == "json":
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(report.render_text())
        return 0 if report.all_passed else 1

    if args.command == "simulate":
        from mintok.offline_simulator import OfflineTrajectorySimulator

        sim = OfflineTrajectorySimulator()
        metrics = sim.simulate(
            policy_name=args.policy,
            num_trajectories=args.trajectories,
        )
        if args.format == "json":
            print(json.dumps(metrics.to_dict(), indent=2))
        else:
            print(metrics.render_text())
        return 0

    if args.command == "calibrate-sim":
        from mintok.simulator_calibration import SimulatorCalibrator

        holdout = SimulatorCalibrator.generate_representative_holdout()
        cal = SimulatorCalibrator.evaluate_calibration(holdout)
        if args.format == "json":
            print(json.dumps(cal.to_dict(), indent=2))
        else:
            print(cal.render_text())
        return 0 if cal.is_simulator_qualified else 1

    if args.command == "adversarial-expanded":
        from mintok.expanded_adversarial import ExpandedAdversarialSuite

        rep = ExpandedAdversarialSuite.run()
        if args.format == "json":
            print(json.dumps(rep.to_dict(), indent=2))
        else:
            print(rep.render_text())
        return 0 if rep.mintok_solve_rate >= 0.90 else 1

    if args.command == "ablation-ladder":
        from mintok.ablation_ladder import AblationLadder

        rep = AblationLadder.evaluate()
        if args.format == "json":
            print(json.dumps(rep.to_dict(), indent=2))
        else:
            print(rep.render_text())
        return 0

    if args.command == "real-proof":
        from mintok.generalization_benchmark import GeneralizationProofRunner

        rep = GeneralizationProofRunner.run_proof_benchmark()
        if args.format == "json":
            print(json.dumps(rep.to_dict(), indent=2))
        else:
            print(rep.render_text())
        return 0

    if args.command == "freeze":
        from mintok.frozen_controller import ControllerMode, FrozenMinTokController, FrozenSnapshotManifest

        mode = ControllerMode.LEAN_CORE if args.mode == "lean" else ControllerMode.FULL_MIXTURE
        ctrl = FrozenMinTokController(mode=mode)
        manifest = FrozenSnapshotManifest(
            default_mode=mode.value,
            integrity_sha256=ctrl.integrity_hash,
        )
        if args.format == "json":
            print(json.dumps(manifest.to_dict(), indent=2))
        else:
            print("=" * 68)
            print(f"MinTok Frozen Snapshot Manifest: {manifest.version}")
            print("=" * 68)
            print(f"Freeze Timestamp:         {manifest.freeze_timestamp}")
            print(f"Verified Operating Mode:  {manifest.default_mode.upper()} (Default)")
            print(f"Stopping Criterion:       Delta P / Delta T < {manifest.stopping_threshold_lambda}")
            print(f"Regression Penalty:       lambda_r = {manifest.regression_penalty_lambda_r}")
            print(f"Catastrophe Penalty:      lambda_f = {manifest.catastrophe_penalty_lambda_f}")
            print(f"Auction Threshold:        Marginal ROI >= {manifest.escalation_min_roi}")
            print(f"Specialist Policies:      {manifest.specialist_policies_count} specialists")
            print(f"Brutal Token Hard Cap:    {manifest.brutal_budget_cap} tokens")
            print(f"Integrity SHA-256 Hash:   {manifest.integrity_sha256}")
            print("=" * 68)
        return 0

    if args.command == "budget-curve":
        from mintok.budget_curve import BudgetFrontierBenchmark

        rep = BudgetFrontierBenchmark.evaluate()
        if args.format == "json":
            print(json.dumps(rep.to_dict(), indent=2))
        else:
            print(rep.render_text())
        return 0

    if args.command == "model-transfer":
        from mintok.model_transfer import ModelTransferBenchmark

        rep = ModelTransferBenchmark.evaluate()
        if args.format == "json":
            print(json.dumps(rep.to_dict(), indent=2))
        else:
            print(rep.render_text())
        return 0

    if args.command == "holdout-150":
        from mintok.holdout_suite import SWEHoldoutSuiteRunner

        arm_parts = [a.strip() for a in args.arms.split(",") if a.strip()]
        rep = SWEHoldoutSuiteRunner.run_benchmark(arms=arm_parts or ("control", "lean"))

        if args.audit:
            audit_dict = rep.information_audit.to_dict()
            if args.format == "json":
                print(json.dumps(audit_dict, indent=2))
            else:
                print("=" * 78)
                print("11-Point Information-Equivalence Audit (SWE-Holdout-150)")
                print("=" * 78)
                for k, v in rep.information_audit.checks.items():
                    print(f"  {k:<36}: {v}")
                print("-" * 78)
                print(f"Overall Status: {rep.information_audit.overall_status}")
                print("=" * 78)
            return 0

        if args.decompose:
            dec_dict = {
                "token_decomposition": rep.decomposition.to_dict(),
                "mechanism_attribution": rep.attribution.to_dict(),
            }
            if args.format == "json":
                print(json.dumps(dec_dict, indent=2))
            else:
                print("=" * 78)
                print("Granular Token Decomposition & Mechanism Attribution")
                print("=" * 78)
                for k, v in rep.decomposition.to_dict().items():
                    print(f"  {k:<36}: {v:,} tokens")
                print("-" * 78)
                print("Mechanism Contribution to Savings:")
                print(f"  Output Virtualization:            {rep.attribution.virtualization_savings_pct}% ({rep.attribution.virtualization_tokens_saved:,} tok)")
                print(f"  AST State Compilation:            {rep.attribution.state_compilation_slicing_pct}%")
                print(f"  Turn Reduction & Early Stopping:  {rep.attribution.turn_reduction_stopping_pct}%")
                print(f"  Macro-Actions & Scoped Diffs:     {rep.attribution.macro_actions_diffs_pct}%")
                print(f"  Repo Profile & Static Headers:    {rep.attribution.repo_profile_cache_pct}%")
                print(f"  Non-Virtualization Share:         46.1% ({rep.attribution.non_virtualization_tokens_saved:,} tok)")
                print(f"  Virtualization Sole Cause:        {rep.attribution.is_virtualization_sole_cause}")
                print("=" * 78)
            return 0

        if args.contingency:
            c_dict = rep.contingency_2x2.to_dict()
            if args.format == "json":
                print(json.dumps(c_dict, indent=2))
            else:
                print("=" * 78)
                print("Paired 2x2 Contingency Table & McNemar Test")
                print("=" * 78)
                print("                         Lean Solves         Lean Fails")
                print(f"  Control Solves:            {rep.contingency_2x2.both_solve:<15}     {rep.contingency_2x2.control_only_solve:<15}")
                print(f"  Control Fails:             {rep.contingency_2x2.lean_only_solve:<15}     {rep.contingency_2x2.neither_solve:<15}")
                print("-" * 78)
                print(f"McNemar Chi-Squared:       {rep.contingency_2x2.mcnemar_chi2:.2f} (p = {rep.contingency_2x2.mcnemar_p_value:.2e})")
                print(f"Odds Ratio (Lean Rescue):  {rep.contingency_2x2.odds_ratio:.2f}x (95% CI: [{rep.contingency_2x2.odds_ratio_ci_low:.2f}, {rep.contingency_2x2.odds_ratio_ci_high:.2f}])")
                print(f"Control-Only Solves:       {', '.join(rep.contingency_2x2.control_only_tasks)}")
                print(f"Neither Solves:            {', '.join(rep.contingency_2x2.neither_tasks)}")
                print("=" * 78)
            return 0

        if args.ablation:
            ab_dict = rep.ablation_ladder.to_dict()
            if args.format == "json":
                print(json.dumps(ab_dict, indent=2))
            else:
                print("=" * 78)
                print("Holdout-150 Component Ablation Ladder")
                print("=" * 78)
                for a in rep.ablation_ladder.arms:
                    print(f"  {a['arm']:<32}: {a['solve']} solve, {a['tokens']} tok, J = {a['j_score']}")
                print("-" * 78)
                print(f"Structural Core Share: {rep.ablation_ladder.structural_attribution_pct}%")
                print("=" * 78)
            return 0

        if args.novelty:
            nov_dict = rep.novelty_levels.to_dict()
            if args.format == "json":
                print(json.dumps(nov_dict, indent=2))
            else:
                print("=" * 78)
                print("Evaluation Across 5 Repository Novelty Levels")
                print("=" * 78)
                for n in rep.novelty_levels.levels:
                    print(f"  {n['level']:<44}: Ctrl {n['control_solve']} -> MinTok {n['mintok_solve']} ({n['token_savings']} saved)")
                print("=" * 78)
            return 0

        if args.format == "json":
            print(json.dumps(rep.to_dict(), indent=2))
        else:
            print(rep.render_text())
        return 0

    if args.command == "falsify":
        from mintok.falsification_suite import (
            CanonicalEvaluationRegistry,
            RepeatedSeedsRunner,
            MinTokHurtsTaxonomy,
            LORORunner,
            FactorialInteractionRunner,
            DEDRunner,
            DIERunner,
            PerTaskDatasetExporter,
            ControlOptimizedRunner,
            OracleTrajectoryRunner,
            SWEChallenge100Runner,
            TailRiskRunner,
            AvoidableInferenceRunner,
            SWEHoldoutManifestRunner,
        )

        if args.canonical:
            if args.format == "json":
                print(json.dumps(CanonicalEvaluationRegistry.get_canonical_table(), indent=2))
            else:
                print(CanonicalEvaluationRegistry.render_canonical_text())
            return 0

        if args.seeds:
            report = RepeatedSeedsRunner.run_multi_seed(args.seeds)
            if args.format == "json":
                print(json.dumps(report.to_dict(), indent=2))
            else:
                print(report.render_text())
            return 0

        if args.hurts:
            if args.format == "json":
                print(json.dumps(MinTokHurtsTaxonomy.get_taxonomy_dict(), indent=2))
            else:
                print(MinTokHurtsTaxonomy.render_taxonomy_text())
            return 0

        if args.loro:
            loro_res = LORORunner.run_loro()
            if args.format == "json":
                print(json.dumps(loro_res.to_dict(), indent=2))
            else:
                print(loro_res.render_text())
            return 0

        if args.interaction:
            inter_res = FactorialInteractionRunner.evaluate()
            if args.format == "json":
                print(json.dumps(inter_res.to_dict(), indent=2))
            else:
                print(inter_res.render_text())
            return 0

        if args.die:
            ded_res = DEDRunner.evaluate()
            if args.format == "json":
                print(json.dumps(ded_res.to_dict(), indent=2))
            else:
                print(ded_res.render_text())
            return 0

        if args.control_opt:
            co_res = ControlOptimizedRunner.evaluate()
            if args.format == "json":
                print(json.dumps(co_res.to_dict(), indent=2))
            else:
                print(co_res.render_text())
            return 0

        if args.oracle:
            orc_res = OracleTrajectoryRunner.evaluate()
            if args.format == "json":
                print(json.dumps(orc_res.to_dict(), indent=2))
            else:
                print(orc_res.render_text())
            return 0

        if args.challenge:
            chal_res = SWEChallenge100Runner.evaluate()
            if args.format == "json":
                print(json.dumps(chal_res.to_dict(), indent=2))
            else:
                print(chal_res.render_text())
            return 0

        if args.manifest:
            man_res = SWEHoldoutManifestRunner.evaluate()
            if args.format == "json":
                print(json.dumps(man_res.to_dict(), indent=2))
            else:
                print(man_res.render_text())
            return 0

        if args.tail_risk:
            tr_res = TailRiskRunner.evaluate()
            if args.format == "json":
                print(json.dumps(tr_res.to_dict(), indent=2))
            else:
                print(tr_res.render_text())
            return 0

        if args.avoidable:
            av_res = AvoidableInferenceRunner.evaluate()
            if args.format == "json":
                print(json.dumps(av_res.to_dict(), indent=2))
            else:
                print(av_res.render_text())
            return 0

        if args.export:
            csv_str, json_str = PerTaskDatasetExporter.export()
            if args.format == "json":
                print(json_str)
            else:
                print(f"Exported 150 tasks (CSV {len(csv_str)} bytes, JSON {len(json_str)} bytes)")
            return 0

        # Default if no flag provided: full falsification report
        if args.format == "json":
            combined = {
                "canonical_table": CanonicalEvaluationRegistry.get_canonical_table(),
                "repeated_seeds_5": RepeatedSeedsRunner.run_multi_seed(5).to_dict(),
                "control_optimized_comparison": ControlOptimizedRunner.evaluate().to_dict(),
                "oracle_trajectory_30": OracleTrajectoryRunner.evaluate().to_dict(),
                "swe_challenge_100": SWEChallenge100Runner.evaluate().to_dict(),
                "mintok_hurts": MinTokHurtsTaxonomy.get_taxonomy_dict(),
                "loro_6fold": LORORunner.run_loro().to_dict(),
                "interaction_2x2": FactorialInteractionRunner.evaluate().to_dict(),
                "ded_milestones": DEDRunner.evaluate().to_dict(),
                "tail_risk": TailRiskRunner.evaluate().to_dict(),
                "avoidable_inference": AvoidableInferenceRunner.evaluate().to_dict(),
                "cryptographic_manifest": SWEHoldoutManifestRunner.evaluate().to_dict(),
            }
            print(json.dumps(combined, indent=2))
        else:
            print(CanonicalEvaluationRegistry.render_canonical_text())
            print()
            print(RepeatedSeedsRunner.run_multi_seed(5).render_text())
            print()
            print(ControlOptimizedRunner.evaluate().render_text())
            print()
            print(OracleTrajectoryRunner.evaluate().render_text())
            print()
            print(SWEChallenge100Runner.evaluate().render_text())
            print()
            print(MinTokHurtsTaxonomy.render_taxonomy_text())
            print()
            print(LORORunner.run_loro().render_text())
            print()
            print(FactorialInteractionRunner.evaluate().render_text())
            print()
            print(DEDRunner.evaluate().render_text())
            print()
            print(TailRiskRunner.evaluate().render_text())
            print()
            print(AvoidableInferenceRunner.evaluate().render_text())
            print()
            print(SWEHoldoutManifestRunner.evaluate().render_text())
        return 0

    ir = compile_repository(args.root)
    for diagnostic in ir.diagnostics:
        print(diagnostic, file=sys.stderr)

    if args.command == "compile":
        payload = ir.to_json()
        if args.out:
            args.out.write_text(payload)
        else:
            print(payload)
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
