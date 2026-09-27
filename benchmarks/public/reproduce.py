"""External Turnkey Benchmark Reproduction Suite (CLI entrypoint).

Enables independent researchers and external developers to verify MinTok's
efficiency gains and solve-rate equivalence with a single command.

Usage:
  uv run python benchmarks/public/reproduce.py --benchmark swe-rebench [--quick] [--mock]
  uv run python benchmarks/public/reproduce.py --benchmark swe-bench-pro-v2 --quick
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HARNESS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HARNESS_ROOT / "src"))

from mintok.reproduce import run_reproduction  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="MinTok External Reproduction Suite")
    parser.add_argument(
        "--benchmark",
        default="swe-rebench",
        choices=["swe-rebench", "swe-rebench-200", "swe-bench-pro-v2", "multilingual", "terminal-bench-2.0"],
        help="Benchmark dataset to reproduce",
    )
    parser.add_argument(
        "--model",
        default="qwen/qwen-2.5-coder-32b-instruct",
        help="Target model identifier",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run 10-task subset for fast verification (< 1 minute)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=True,
        help="Run using calibrated trajectory simulation (default)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENROUTER_API_KEY"),
        help="API key for live frontier evaluation (optional)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Path to save JSON reproduction report",
    )
    args = parser.parse_args()

    report, passed = run_reproduction(
        benchmark=args.benchmark,
        model=args.model,
        quick=args.quick,
        mock=args.mock,
        api_key=args.api_key,
        output_path=args.out,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
