"""Deterministic context-token benchmark: two baselines vs the mintok ABI.

Unlike mintok.benchmark (which compares run records from real agent arms), this
harness needs no model calls at all. For a fixed set of representative agent
questions, each arm constructs the context needed to answer them and tokens are
estimated with the project's pluggable estimator (chars/4). Two baselines are
reported:

- **raw** — a competent grep-and-paging agent: hit lines plus the source of
  every innermost definition containing a hit (hit lines alone carry no
  semantics).
- **strong** — modern tooling without semantic facts: an outline/symbols view
  for lookups (matching def lines), ripgrep hits plus enclosing signature lines
  for attribution, the body itself for semantics (no textual tool can enumerate
  raises/calls without it), and a unified diff for relearning. The strong
  baseline is cheaper but unsound: it cannot tell calls from mentions or prove
  an interface unchanged.

The treatment arm ingests only the corresponding ABI answer. Subject selection,
hit matching, and ordering are all deterministic; the harness is paired per
task. Request/retry overheads, reasoning tokens, and index construction are
trajectory-level costs and are intentionally not modeled here.
"""

from __future__ import annotations

import difflib
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
    "estimated tokens (chars/4), no model calls; raw baseline models a "
    "competent grep-and-paging agent (hit lines + sources of innermost "
    "hit-containing definitions); strong baseline models modern tooling "
    "without semantic facts (outline views, rg hits + enclosing signatures, "
    "unified diff for relearn); treatment pays the ABI tool surface once per "
    "session, baseline tool schemas are not modeled (conservative toward "
    "mintok); trajectory costs (requests, retries, reasoning, indexing) are "
    "not modeled"
)


@dataclass(frozen=True, slots=True)
class TaskResult:
    task_id: str
    task_class: str
    baseline_tokens: int
    treatment_tokens: int
    strong_tokens: int = 0


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
    def strong_total(self) -> int:
        return sum(t.strong_tokens for t in self.tasks)

    @property
    def reduction_ratio(self) -> float:
        if self.treatment_total == 0:
            return float("inf")
        return self.baseline_total / self.treatment_total

    @property
    def strong_reduction_ratio(self) -> float:
        if self.treatment_total == 0:
            return float("inf")
        return self.strong_total / self.treatment_total

    @property
    def saved_pct(self) -> float:
        if self.baseline_total == 0:
            return 0.0
        return 100.0 * (1 - self.treatment_total / self.baseline_total)

    def to_dict(self) -> dict:
        return {
            "baseline_tokens": self.baseline_total,
            "strong_tokens": self.strong_total,
            "treatment_tokens": self.treatment_total,
            "fixed_tool_surface_tokens": self.fixed_tool_surface_tokens,
            "reduction_ratio": round(self.reduction_ratio, 2),
            "strong_reduction_ratio": round(self.strong_reduction_ratio, 2),
            "tokens_saved_pct": round(self.saved_pct, 1),
            "notes": NOTES,
            "tasks": [asdict(t) for t in self.tasks],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def render_text(self) -> str:
        width = max((len(t.task_id) for t in self.tasks), default=10)
        lines = [
            f"{'task':<{width}}  {'class':<9}  {'baseline':>8}  {'strong':>7}  {'mintok':>7}",
            *(
                f"{t.task_id:<{width}}  {t.task_class:<9}  {t.baseline_tokens:>8}  "
                f"{t.strong_tokens:>7}  {t.treatment_tokens:>7}"
                for t in self.tasks
            ),
            (
                f"{'total':<{width}}  {'':<9}  {self.baseline_total:>8}  "
                f"{self.strong_total:>7}  {self.treatment_total:>7}"
            ),
            f"token reduction vs grep-and-paging: {self.reduction_ratio:.1f}x ({self.saved_pct:.1f}% fewer context tokens)",
            f"token reduction vs strong tooling: {self.strong_reduction_ratio:.1f}x",
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

    def _innermost(self, path: str, lineno: int) -> Symbol | None:
        best: Symbol | None = None
        for sym in self.ir.symbols.values():
            if (
                sym.source.path == path
                and sym.source.start_line <= lineno <= sym.source.end_line
                and (best is None or sym.source.start_line > best.source.start_line)
            ):
                best = sym
        return best

    def strong_find_tokens(self, name: str) -> int:
        """Outline/symbols view: one ``path:line:def`` line per matching def."""
        out: list[str] = []
        for path, lines in sorted(self.lines.items()):
            for i, line in enumerate(lines):
                stripped = line.lstrip()
                if stripped.startswith(("def ", "async def ")) and name in line:
                    out.append(f"{path}:{i + 1}:{line.strip()}")
        return estimate_tokens("\n".join(out))

    def strong_hits_tokens(self, needle: str) -> int:
        """Ripgrep hits plus enclosing signature lines (breadcrumb attribution)."""
        out: list[str] = []
        seen_defs: set[tuple[str, int]] = set()
        for path, lines in sorted(self.lines.items()):
            for i, line in enumerate(lines):
                if needle not in line:
                    continue
                out.append(f"{path}:{i + 1}:{line.strip()}")
                sym = self._innermost(path, i + 1)
                if sym and sym.source.start_line != i + 1 and (path, sym.source.start_line) not in seen_defs:
                    seen_defs.add((path, sym.source.start_line))
                    out.append(
                        f"{path}:{sym.source.start_line}:"
                        f"{lines[sym.source.start_line - 1].strip()}"
                    )
        return estimate_tokens("\n".join(out))

    def strong_diff_tokens(self, old_view: "_SourceIndex", paths: list[str]) -> int:
        """Unified-diff view of the changed files (modern textual relearn)."""
        out: list[str] = []
        for path in paths:
            old_lines = old_view.lines.get(path, [])
            new_lines = self.lines.get(path, [])
            out += list(
                difflib.unified_diff(
                    old_lines, new_lines, fromfile=f"old/{path}", tofile=f"new/{path}", lineterm=""
                )
            )
        return estimate_tokens("\n".join(out))


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
                strong_tokens=view.strong_find_tokens(name),
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
                # No textual tool can enumerate raises/calls without the body.
                strong_tokens=estimate_tokens(view.symbol_source(sym)),
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
                strong_tokens=view.strong_hits_tokens(short),
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
                strong_tokens=view.strong_hits_tokens(needle),
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
        strong = old_view.strong_diff_tokens(view, sorted(changed_files))
        pack = build_relearn_pack(old_ir, ir)
        report.tasks.append(
            TaskResult(
                task_id="relearn",
                task_class="relearn",
                baseline_tokens=baseline,
                strong_tokens=strong,
                treatment_tokens=estimate_tokens(pack.render()),
            )
        )

    return report
