"""Command-line entry point: ``mintok compile``, ``diff``, and ``slice``.

The open build compiles, diffs, and serves the agent ABI. Slicing is reserved
for the commercial MinTok Inference Compiler.
"""

from __future__ import annotations

import argparse
import json
import sys
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
    sub = parser.add_subparsers(dest="command", required=True)

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
    bench.add_argument("root", type=Path)
    bench.add_argument(
        "--vs", type=Path, help="second repository root; adds a relearn task between the two"
    )
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

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
        report = run_token_benchmark(args.root, vs=args.vs)
        print(report.to_json() if args.format == "json" else report.render_text())
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
                "agreement_rate": res["agreement_rate"],
                "net_token_delta": res["net_token_delta"],
                "net_success_delta": res["net_success_delta"],
                "evaluations": [e.to_dict() for e in res["evaluations"]],
            }
            print(json.dumps(res_dict, indent=2))
        else:
            print(f"Shadow Policy Evaluation ({res['total']} turns):")
            print(f"  Agreement Rate:    {res['agreement_rate'] * 100:.1f}%")
            print(f"  Net Token Delta:   {res['net_token_delta']:+,.0f} tokens")
            print(f"  Net Success Delta: {res['net_success_delta']:+.4f}")
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
