"""Large-module slice backend: precision source slicing, zero prose.

Control proves that targeted raw source beats semantic reads on huge
modules. The slicer's job is to pick those targets more deterministically
than grep: rank exact source regions from the Agent Program IR (lexical
matches, call edges, field writes, test references) and return file:line
excerpts under a hard token budget. It never summarizes, never narrates,
and never dumps whole files — full source is available only as an explicit
agent-initiated fallback outside this module.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

from mintok.compiler.python import compile_repository
from mintok.tokens import estimate_tokens

INITIAL_BUDGET = 800
EXPANDED_BUDGET = 2000

SECTIONS = ("target candidates", "relevant callers", "relevant writes", "relevant tests")
_MAX_CALLERS, _MAX_WRITES, _MAX_TESTS = 6, 3, 3

_STOPWORDS = frozenset(
    "the a an and or of to in on for with must should when that this from into only also make "
    "fix it its is are be been so such by as at not no but if then than them they their there "
    "before after while during without within over under out up down do does done can could "
    "would will shall may might value values".split()
)

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
_EXCERPT_MARK = "--- excerpts ---"


@dataclass(frozen=True, slots=True)
class Region:
    path: str
    start: int  # 1-based, inclusive
    end: int
    kind: str  # definition | caller | write | test
    label: str
    score: float


@dataclass(slots=True)
class Slice:
    text: str
    regions: list[Region] = field(default_factory=list)
    tokens: int = 0
    budget: int = INITIAL_BUDGET
    truncated: bool = False
    targets: list[str] = field(default_factory=list)


def _terms(instruction: str) -> tuple[list[str], list[str]]:
    """Strong terms (name-like identifiers) vs weak terms (plain words)."""
    strong: list[str] = []
    weak: list[str] = []
    for raw in _IDENT.findall(instruction):
        for part in raw.split("."):
            low = part.lower()
            if not part or low in _STOPWORDS:
                continue
            if "_" in part or part[0].isupper():
                if part not in strong:
                    strong.append(part)
            elif len(part) > 3 and part not in weak:
                weak.append(part)
    return strong, weak


def _last_token(qualname: str) -> str:
    return qualname.split(".")[-1]


def _score_symbols(ir, strong: list[str], weak: list[str]) -> list[tuple[float, object]]:
    """Rank symbols; ties prefer deeper qualnames (a method over its class)."""
    scored = []
    for sym in ir.symbols.values():
        score = 0.0
        last = _last_token(sym.name).lower()
        for term in strong:
            if term.lower() == last or sym.name.lower() == term.lower():
                score = max(score, 12.0)
            elif term.lower() in sym.name.lower():
                score = max(score, 6.0)
        for term in weak:
            if term == last:
                score = max(score, 12.0)
            elif re.search(rf"\b{re.escape(term)}\b", sym.signature):
                score += 1.0
        if score:
            scored.append((score, sym))
    scored.sort(key=lambda pair: (-pair[0], -pair[1].name.count("."), pair[1].id))
    return scored


def _test_regions(root: Path, names: list[str]) -> list[Region]:
    """Line-level mentions of target names in test files, via AST walk."""
    regions: list[Region] = []
    wanted = set(names)

    def visit(node: ast.AST, fn: str | None, rel: str) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            fn = node.name
        hit = None
        if isinstance(node, ast.Name) and node.id in wanted:
            hit = node.id
        elif isinstance(node, ast.Attribute) and node.attr in wanted:
            hit = node.attr
        if hit is not None and fn is not None:
            regions.append(Region(rel, node.lineno, node.lineno, "test", fn, 1.0))
        for child in ast.iter_child_nodes(node):
            visit(child, fn, rel)

    for path in sorted(root.rglob("test_*.py")):
        try:
            tree = ast.parse(path.read_text(errors="replace"))
        except SyntaxError:
            continue
        visit(tree, None, path.relative_to(root).as_posix())
    regions.sort(key=lambda r: (r.path, r.start, -(r.end - r.start)))
    return regions


def _class_member_regions(root: Path, sym, strong: list[str]) -> list[Region]:
    """A class target emits its header plus ranked member methods.

    Dumping a 900-line class is the packet failure mode at class
    granularity: rank members instead — the named method first, then
    methods sharing its name terms, then siblings in source order — and
    let the greedy budget inclusion drop what does not fit.
    """
    path = sym.source.path
    try:
        tree = ast.parse((root / path).read_text(errors="replace"))
    except SyntaxError:
        return [Region(path, sym.source.start_line, sym.source.end_line, "definition", sym.name, 12.0)]
    class_name = _last_token(sym.name)
    node = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == class_name),
        None,
    )
    if node is None:
        return [Region(path, sym.source.start_line, sym.source.end_line, "definition", sym.name, 12.0)]
    regions = [Region(path, node.lineno, node.lineno, "class_header", class_name, 12.0)]
    methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    for method in methods:
        score = 0.0
        for term in strong:
            if term.lower() == method.name.lower():
                score = max(score, 12.0)
            elif term.lower() in method.name.lower():
                score = max(score, 4.0)
        regions.append(
            Region(
                path,
                method.lineno,
                method.end_lineno or method.lineno,
                "definition",
                f"{class_name}.{method.name}",
                score or 0.5,
            )
        )
    regions.sort(key=lambda r: -r.score)
    return regions


def _is_test_path(path: str) -> bool:
    parts = Path(path).parts
    return (len(parts) > 1 and parts[0] in ("tests", "test")) or Path(path).name.startswith("test_")


def _excerpt(root: Path, region: Region) -> str:
    lines = (root / region.path).read_text(errors="replace").splitlines()
    body = "\n".join(lines[region.start - 1 : region.end])
    return f"{region.path}:{region.start}-{region.end}\n{body}"


def _render_line(region: Region) -> str:
    if region.kind == "write":
        return f"  {region.label} @ {region.path}:{region.start}"
    if region.kind == "test":
        return f"  {region.path}:{region.start} {region.label}"
    return f"  {region.path}:{region.start}-{region.end} {region.label}()"


def slice_task(
    repo_root: str | Path,
    instruction: str,
    initial_budget: int = INITIAL_BUDGET,
    expanded_budget: int = EXPANDED_BUDGET,
) -> Slice:
    """Rank and render the deterministic source slice for one task."""
    root = Path(repo_root)
    ir = compile_repository(root)
    strong, weak = _terms(instruction)
    scored = _score_symbols(ir, strong, weak)

    exact = [(score, sym) for score, sym in scored if score >= 12.0] or scored[:2]
    # A class target is not dumped whole: it expands to a header plus
    # ranked member methods (_class_member_regions). A method target
    # stays a plain definition; when both the class and one of its
    # methods match, the class expansion dedupes the shared member.
    targets = exact or scored[:2]

    by_id = ir.symbols
    regions: list[Region] = []
    seen_spans: set[tuple[str, int]] = set()
    for score, sym in targets:
        if sym.kind == "class":
            for region in _class_member_regions(root, sym, strong):
                key = (region.path, region.start)
                if key not in seen_spans:
                    seen_spans.add(key)
                    regions.append(region)
        else:
            key = (sym.source.path, sym.source.start_line)
            if key not in seen_spans:
                seen_spans.add(key)
                regions.append(
                    Region(sym.source.path, sym.source.start_line, sym.source.end_line, "definition", sym.name, score)
                )

    # Primary target first: the highest-scored definition (a class's
    # named method beats its container), then the class header, then
    # remaining members in score order (stable: source order on ties).
    definitions = sorted(
        (r for r in regions if r.kind == "definition"), key=lambda r: -r.score
    )
    headers = [r for r in regions if r.kind == "class_header"]
    regions = definitions[:1] + headers + definitions[1:]

    callers: list[Region] = []
    writes: list[Region] = []
    for _, sym in targets:
        for cid in {f.subject for f in ir.callers_of(sym.id)}:
            caller = by_id.get(cid)
            if caller is None or any(
                r.path == caller.source.path and r.start == caller.source.start_line for r in regions
            ):
                continue
            callers.append(
                Region(caller.source.path, caller.source.start_line, caller.source.end_line, "caller", caller.name, 2.0)
            )
        if sym.kind == "class":
            writers = [
                f for f in ir.facts if f.predicate == "writes" and f.object.startswith(sym.id + ".")
            ]
        else:
            # Method target: attribute the fields it writes and note the
            # sibling methods writing the same fields.
            class_id = sym.id.rsplit(".", 1)[0] if "." in sym.name else None
            own_fields = {
                f.object.split(".")[-1]
                for f in ir.facts
                if f.subject == sym.id and f.predicate == "writes"
            }
            writers = [
                f
                for f in ir.facts
                if f.predicate == "writes"
                and class_id is not None
                and f.object.startswith(class_id + ".")
                and f.object.split(".")[-1] in own_fields
            ]
        for fact in writers[:_MAX_WRITES]:
            writer = by_id.get(fact.subject)
            if writer is None or any(
                r.path == writer.source.path and r.start == writer.source.start_line for r in regions
            ):
                continue
            writes.append(
                Region(
                    writer.source.path,
                    writer.source.start_line,
                    writer.source.end_line,
                    "write",
                    fact.object.split(":")[-1],
                    2.0,
                )
            )

    callers.sort(key=lambda r: (r.path, r.start))
    # A caller under the test tree is a test reference; keep one section.
    live_callers = [r for r in callers if not _is_test_path(r.path)]
    test_callers = [
        Region(r.path, r.start, r.end, "test", r.label, r.score)
        for r in callers
        if _is_test_path(r.path)
    ]
    regions += live_callers[:_MAX_CALLERS]
    regions += writes[:_MAX_WRITES]
    tests = _test_regions(root, [_last_token(sym.name) for _, sym in targets]) + test_callers
    tests.sort(key=lambda r: (r.path, r.start, -(r.end - r.start)))
    kept: list[Region] = []
    for region in tests:
        if any(
            other is not region
            and other.path == region.path
            and other.start <= region.start
            and other.end >= region.end
            for other in tests
        ):
            continue
        kept.append(region)
    regions += kept[:_MAX_TESTS]

    budget = initial_budget
    definition_regions = [r for r in regions if r.kind == "definition"]
    header_tokens = sum(estimate_tokens(_excerpt(root, r)) for r in regions if r.kind == "class_header")
    if definition_regions and header_tokens == 0:
        # Whole-symbol targets (functions) keep the expand-on-oversize rule.
        definition_tokens = sum(estimate_tokens(_excerpt(root, r)) for r in definition_regions)
        if definition_tokens > initial_budget:
            budget = expanded_budget

    # Greedy inclusion against the real rendered size, so headers and
    # excerpts count toward the budget the way the agent pays for them.
    # The primary definition is never dropped; member methods beyond the
    # budget are (that is the point of member-method slicing), as are
    # non-definition regions.
    included: list[Region] = []
    dropped = False
    primary_kept = False
    for region in regions:
        is_primary = region.kind == "definition" and not primary_kept
        if is_primary:
            primary_kept = True
            included.append(region)
            continue
        candidate = included + [region]
        if estimate_tokens(_render(root, candidate, budget)) > budget:
            dropped = True
            continue
        included.append(region)

    text = _render(root, included, budget)
    tokens = estimate_tokens(text)
    return Slice(
        text=text,
        regions=included,
        tokens=tokens,
        budget=budget,
        truncated=dropped or tokens > budget,
        targets=[sym.name for _, sym in targets],
    )


def _render(root: Path, regions: list[Region], budget: int) -> str:
    out: list[str] = []
    groups = (
        ("class_header", "target candidates"),
        ("definition", "target candidates"),
        ("caller", "relevant callers"),
        ("write", "relevant writes"),
        ("test", "relevant tests"),
    )
    rendered_titles: set[str] = set()
    for kind, title in groups:
        group = [r for r in regions if r.kind == kind]
        if not group:
            continue
        if title not in rendered_titles:
            out.append(f"{title}:")
            rendered_titles.add(title)
        out.extend(_render_line(r) for r in group)
    out.append(_EXCERPT_MARK)
    for region in sorted(regions, key=lambda r: (r.path, r.start)):
        out.append(_excerpt(root, region))
    out.append(f"[budget {budget}]")
    return "\n".join(out)
