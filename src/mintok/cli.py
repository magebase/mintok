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
from mintok.compiler import compile_repository
from mintok.diff import diff_ir, render_changes

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
