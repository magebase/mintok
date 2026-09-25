"""Deterministic context-token benchmark: grep-and-paging vs the mintok ABI.

Unlike mintok.benchmark (which compares run records from real agent arms), this
harness needs no model calls at all. For a fixed set of representative agent
questions, both arms construct the context needed to answer them and the tokens
are estimated with the project's pluggable estimator (chars/4). The baseline arm
models a competent grep-and-paging agent: it ingests grep hit lines plus the
source of every definition containing a hit, because hit lines alone carry no
semantics. The treatment arm ingests only the corresponding ABI answer. Subject
selection, hit matching, and ordering are all deterministic; the harness is
paired per task.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from mintok.abi import AgentABI, tool_surface_tokens
from mintok.compiler import compile_repository
from mintok.compiler.python import iter_python_files
from mintok.diff import diff_ir
from mintok.ir import ProgramIR, Symbol
from mintok.pack import build_relearn_pack
from mintok.tokens import estimate_tokens

NOTES = (
    "estimated tokens (chars/4), no model calls; baseline models a competent "
    "grep-and-paging agent (hit lines + sources of hit-containing definitions); "
    "treatment pays the ABI tool surface once per session, baseline tool "
    "schemas are not modeled (conservative toward mintok)"
)


@dataclass(frozen=True, slots=True)
class TaskResult:
    task_id: str
    task_class: str
    baseline_tokens: int
    treatment_tokens: int


@dataclass(slots=True)
class BenchReport:
    tasks: list[TaskResult] = field(default_factory=list)
    fixed_tool_surface_tokens: int = 0

    @property
    def baseline_total(self) -> int:
        return sum(t.baseline_tokens for t in self.tasks)

    @property
    def treatment_total(self) -> int:
        return sum(t.treatment_tokens for t in self.tasks) + self.fixed_tool_surface_tokens

    @property
    def reduction_ratio(self) -> float:
        if self.treatment_total == 0:
            return float("inf")
        return self.baseline_total / self.treatment_total

    @property
    def saved_pct(self) -> float:
        if self.baseline_total == 0:
            return 0.0
        return 100.0 * (1 - self.treatment_total / self.baseline_total)

    def to_dict(self) -> dict:
        return {
            "baseline_tokens": self.baseline_total,
            "treatment_tokens": self.treatment_total,
            "fixed_tool_surface_tokens": self.fixed_tool_surface_tokens,
            "reduction_ratio": round(self.reduction_ratio, 2),
            "tokens_saved_pct": round(self.saved_pct, 1),
            "notes": NOTES,
            "tasks": [asdict(t) for t in self.tasks],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def render_text(self) -> str:
        width = max((len(t.task_id) for t in self.tasks), default=10)
        lines = [
            f"{'task':<{width}}  {'class':<9}  {'baseline':>8}  {'mintok':>7}",
            *(
                f"{t.task_id:<{width}}  {t.task_class:<9}  {t.baseline_tokens:>8}  "
                f"{t.treatment_tokens:>7}"
                for t in self.tasks
            ),
            f"{'total':<{width}}  {'':<9}  {self.baseline_total:>8}  {self.treatment_total:>7}",
            f"token reduction: {self.reduction_ratio:.1f}x ({self.saved_pct:.1f}% fewer context tokens)",
        ]
        return "\n".join(lines)


class _SourceIndex:
    """File texts and symbol sources for baseline (grep-and-paging) accounting."""

    def __init__(self, root: Path, ir: ProgramIR) -> None:
        self.ir = ir
        self.lines = {
            p.relative_to(root).as_posix(): p.read_text().splitlines()
            for p in iter_python_files(root)
        }

    def symbol_source(self, sym: Symbol) -> str:
        lines = self.lines[sym.source.path]
        return "\n".join(lines[sym.source.start_line - 1 : sym.source.end_line])

    def grep_tokens(self, needle: str) -> int:
        hits = "\n".join(
            line for lines in self.lines.values() for line in lines if needle in line
        )
        return estimate_tokens(hits)

    def symbols_whose_source_contains(self, needle: str) -> list[Symbol]:
        """Innermost definitions whose source contains the needle.

        A class symbol's span covers its methods; a paging agent reads the
        smallest enclosing definition, so outer spans that contain an
        innermost match are not counted again.
        """
        matched = [s for s in self.ir.symbols.values() if needle in self.symbol_source(s)]

        def strictly_contains(outer: Symbol, inner: Symbol) -> bool:
            same_path = outer.source.path == inner.source.path
            same_span = (outer.source.start_line, outer.source.end_line) == (
                inner.source.start_line,
                inner.source.end_line,
            )
            return (
                same_path
                and not same_span
                and outer.source.start_line <= inner.source.start_line
                and inner.source.end_line <= outer.source.end_line
            )

        return sorted(
            (
                s
                for s in matched
                if not any(other is not s and strictly_contains(other, s) for other in matched)
            ),
            key=lambda s: s.id,
        )

    def source_tokens(self, symbols: list[Symbol]) -> int:
        return sum(estimate_tokens(self.symbol_source(s)) for s in symbols)


def _short_name(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def run_token_benchmark(
    root: str | Path,
    vs: str | Path | None = None,
    samples: int = 3,
) -> BenchReport:
    root = Path(root)
    ir = compile_repository(root)
    abi = AgentABI(root, ir)
    view = _SourceIndex(root, ir)

    defs = sorted(
        (s for s in ir.symbols.values() if s.kind in ("function", "method")),
        key=lambda s: s.id,
    )
    with_callees = [s for s in defs if ir.facts_for(s.id, "calls")]
    with_callers = [s for s in defs if ir.callers_of(s.id)]
    attributes = sorted({f.object.rsplit(".", 1)[1] for f in ir.facts if f.predicate == "writes"})

    report = BenchReport(fixed_tool_surface_tokens=tool_surface_tokens())

    # find: locate definitions by name.
    names: list[str] = []
    for sym in (with_callees + with_callers)[: 2 * samples]:
        short = _short_name(sym.name)
        if short not in names:
            names.append(short)
    for name in names[:samples]:
        matched = view.symbols_whose_source_contains(name)
        report.tasks.append(
            TaskResult(
                task_id=f"find:{name}",
                task_class="find",
                baseline_tokens=view.grep_tokens(name) + view.source_tokens(matched),
                treatment_tokens=estimate_tokens(abi.query("find", name)),
            )
        )

    # semantics: understand a symbol and its direct callees.
    for sym in with_callees[:samples]:
        callees = [ir.symbols[f.object] for f in ir.facts_for(sym.id, "calls") if f.object in ir.symbols]
        report.tasks.append(
            TaskResult(
                task_id=f"semantics:{sym.id}",
                task_class="semantics",
                baseline_tokens=estimate_tokens(view.symbol_source(sym)) + view.source_tokens(callees),
                treatment_tokens=estimate_tokens(abi.query("summary", sym.id)),
            )
        )

    # callers: attribute calls to their callers without reading source.
    for sym in with_callers[:samples]:
        short = _short_name(sym.name)
        containing = [s for s in view.symbols_whose_source_contains(short)]
        report.tasks.append(
            TaskResult(
                task_id=f"callers:{sym.id}",
                task_class="callers",
                baseline_tokens=view.grep_tokens(short) + view.source_tokens(containing),
                treatment_tokens=estimate_tokens(abi.query("callers", sym.id)),
            )
        )

    # writers: locate attribute writers without reading source.
    for attribute in attributes[:2]:
        needle = f"self.{attribute}"
        report.tasks.append(
            TaskResult(
                task_id=f"writers:{attribute}",
                task_class="writers",
                baseline_tokens=view.grep_tokens(needle) + view.source_tokens(view.symbols_whose_source_contains(needle)),
                treatment_tokens=estimate_tokens(abi.query("writers", attribute)),
            )
        )

    # relearn: what a text-diff pager re-reads vs what interface hashes force.
    if vs is not None:
        vs = Path(vs)
        old_ir = compile_repository(vs)
        changes = diff_ir(old_ir, ir)
        changed_files = {
            ir.symbols[c.symbol_id].source.path
            for c in changes
            if c.symbol_id in ir.symbols
        } | {
            old_ir.symbols[c.symbol_id].source.path
            for c in changes
            if c.symbol_id in old_ir.symbols
        }
        old_view = _SourceIndex(vs, old_ir)
        baseline = sum(
            estimate_tokens("\n".join(old_view.lines[path]))
            for path in sorted(changed_files)
            if path in old_view.lines
        )
        pack = build_relearn_pack(old_ir, ir)
        report.tasks.append(
            TaskResult(
                task_id="relearn",
                task_class="relearn",
                baseline_tokens=baseline,
                treatment_tokens=estimate_tokens(pack.render()),
            )
        )

    return report
