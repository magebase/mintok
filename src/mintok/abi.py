"""Agent ABI: one compact interface over the semantic IR (query / change / add / verify)."""

from __future__ import annotations

import ast
import json
import subprocess
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from mintok.compiler import compile_repository
from mintok.ir import EFFECT_PREDICATES, Fact, ProgramIR, Symbol, render_fact
from mintok.tokens import estimate_tokens

# The open protocol reserves the ``slice`` query op for the commercial MinTok
# Inference Compiler; the open reference build implements the deterministic ops below.
OPEN_QUERY_OPS = ("symbol", "effects", "callers", "find", "writers", "summary")

TOOL_SURFACE: list[dict] = [
    {
        "name": "query",
        "description": (
            "Read semantic facts. op: symbol|effects|callers|find|writers|summary"
            " (slice: commercial build)"
        ),
        "params": {"op": "str", "target": "symbol id | name pattern | attribute"},
    },
    {
        "name": "change",
        "description": "Replace one symbol's definition with new source, or remove it (remove op)",
        "params": {"target": "symbol id", "source": "str"},
    },
    {
        "name": "add",
        "description": "Create a new top-level symbol in an existing or new module file",
        "params": {"target": "new symbol id", "module": "file path", "source": "str"},
    },
    {
        "name": "verify",
        "description": "Run a check command; returns pass/fail and output tail",
        "params": {"argv": "list[str]"},
    },
]

# ``EFFECT_PREDICATES`` is imported from mintok.ir above and re-exported here
# so existing consumers of ``mintok.abi.EFFECT_PREDICATES`` keep working.


def tool_surface_json() -> str:
    return json.dumps(TOOL_SURFACE, separators=(",", ":"))


class ChangeRejected(ValueError):
    pass


def _defined_name(node: ast.stmt) -> str | None:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.name
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    return None


def _splice_imports(text: str, imports: Sequence[str]) -> str:
    """Insert import lines after the docstring and any __future__ imports."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return "\n".join(imports) + "\n\n" + text
    line = 0
    for node in tree.body:
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ) or (isinstance(node, ast.ImportFrom) and node.module == "__future__"):
            line = node.end_lineno or line
        else:
            break
    lines = text.splitlines()
    return "\n".join(lines[:line] + list(imports) + lines[line:]) + ("\n" if text.endswith("\n") else "")


@dataclass(frozen=True, slots=True)
class ChangeResult:
    symbol: Symbol
    interface_changed: bool


@dataclass(frozen=True, slots=True)
class RemoveResult:
    removed: Symbol
    dependents: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AddResult:
    symbol: Symbol
    created_file: bool


@dataclass(frozen=True, slots=True)
class PatchResult:
    path: str
    replaced_lines: int


@dataclass(frozen=True, slots=True)
class VerifyResult:
    passed: bool
    exit_code: int
    tail: str


class AgentABI:
    def __init__(self, root: str | Path, ir: ProgramIR | None = None) -> None:
        self.root = Path(root)
        self.ir = ir if ir is not None else compile_repository(self.root)

    def _symbol(self, symbol_id: str) -> Symbol:
        try:
            return self.ir.symbols[symbol_id]
        except KeyError:
            raise KeyError(f"unknown symbol {symbol_id}") from None

    def get_symbol(self, symbol_id: str) -> str:
        """L1 rendering: signature plus effect facts, never the body."""
        sym = self._symbol(symbol_id)
        lines = [f"{sym.id} {sym.kind} {sym.signature}"]
        lines += [render_fact(f) for f in self.get_effects(symbol_id)]
        return "\n".join(lines)

    def find_symbols(self, pattern: str) -> list[Symbol]:
        """L0: definitions matching a name pattern, one signature line each."""
        needle = pattern.lower()
        return sorted(
            (s for s in self.ir.symbols.values() if needle in s.name.lower()),
            key=lambda s: s.id,
        )

    def get_writers(self, attribute: str) -> list[Fact]:
        """Reverse lookup over ``writes`` facts; no source paging needed."""
        suffix = f".{attribute}"
        return sorted(
            (f for f in self.ir.facts if f.predicate == "writes" and (f.object.endswith(suffix))),
            key=lambda f: f.subject,
        )

    def get_summary(self, symbol_id: str) -> str:
        """L2 rendering: L1 plus the docstring summary line, still never the body."""
        head = self.get_symbol(symbol_id)
        doc = self._doc_summary(symbol_id)
        return f"{head}\nsummary {doc}" if doc else head

    def _doc_summary(self, symbol_id: str) -> str:
        snippet = textwrap.dedent(self.source_of(symbol_id))
        try:
            tree = ast.parse(snippet)
        except SyntaxError:
            return ""
        doc = ast.get_docstring(tree.body[0]) if tree.body else None
        if not doc:
            return ""
        return doc.strip().splitlines()[0].strip()

    def get_effects(self, symbol_id: str) -> list[Fact]:
        self._symbol(symbol_id)
        return [f for p in EFFECT_PREDICATES for f in self.ir.facts_for(symbol_id, p)]

    def get_callers(self, symbol_id: str) -> list[Fact]:
        self._symbol(symbol_id)
        return self.ir.callers_of(symbol_id)

    def source_of(self, symbol_id: str) -> str:
        """L4: raw source. Callers should reach this only when semantics were insufficient."""
        sym = self._symbol(symbol_id)
        lines = (self.root / sym.source.path).read_text().splitlines()
        return "\n".join(lines[sym.source.start_line - 1 : sym.source.end_line])

    def query(self, op: str, target: str) -> str:
        if op == "symbol":
            return self.get_symbol(target)
        if op == "effects":
            return "\n".join(render_fact(f) for f in self.get_effects(target))
        if op == "callers":
            return "\n".join(f"{f.subject} {f.confidence:.2f}" for f in self.get_callers(target))
        if op == "find":
            return "\n".join(f"{s.id} {s.signature}" for s in self.find_symbols(target))
        if op == "writers":
            return "\n".join(f"{f.subject} writes {f.object}" for f in self.get_writers(target))
        if op == "summary":
            return self.get_summary(target)
        if op == "slice":
            raise NotImplementedError(
                "slicing is part of the commercial MinTok Inference Compiler; "
                "this open build does not include it (see README.md)"
            )
        raise ValueError(f"unknown query op {op!r}")

    def change(self, symbol_id: str, source: str) -> ChangeResult:
        sym = self._symbol(symbol_id)
        new_src = textwrap.dedent(source).strip("\n")
        try:
            new_tree = ast.parse(new_src)
        except SyntaxError as exc:
            raise ChangeRejected(f"new source does not parse: {exc.msg}") from None
        short_name = sym.name.rsplit(".", 1)[-1]
        if len(new_tree.body) != 1:
            raise ChangeRejected(f"new source must define exactly one symbol named {short_name}")
        new_node = new_tree.body[0]
        if isinstance(new_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if sym.kind not in ("function", "method") or new_node.name != short_name:
                raise ChangeRejected(f"new source must define exactly one function named {short_name}")
        elif isinstance(new_node, ast.ClassDef):
            if sym.kind != "class" or new_node.name != short_name:
                raise ChangeRejected(f"new source must define exactly one class named {short_name}")
        elif isinstance(new_node, (ast.Assign, ast.AnnAssign)):
            target = new_node.target if isinstance(new_node, ast.AnnAssign) else new_node.targets[0]
            if sym.kind != "constant" or not isinstance(target, ast.Name) or target.id != short_name:
                raise ChangeRejected(f"new source must define exactly one assignment to {short_name}")
            if isinstance(new_node, ast.AnnAssign) and new_node.value is None:
                raise ChangeRejected("constants need a value; use the remove op to delete them")
        else:
            raise ChangeRejected(f"new source must define one function, class, or assignment named {short_name}")

        path = self.root / sym.source.path
        lines = path.read_text().splitlines()
        original = lines[sym.source.start_line - 1]
        indent = original[: len(original) - len(original.lstrip())]
        replacement = [indent + line if line else line for line in new_src.splitlines()]
        updated = lines[: sym.source.start_line - 1] + replacement + lines[sym.source.end_line :]
        text = "\n".join(updated) + "\n"
        try:
            ast.parse(text)
        except SyntaxError as exc:
            raise ChangeRejected(f"file would not parse after change: {exc.msg}") from None

        path.write_text(text)
        self.ir = compile_repository(self.root)
        new_sym = self.ir.symbols[symbol_id]
        return ChangeResult(new_sym, new_sym.interface_hash != sym.interface_hash)

    def add(
        self, symbol_id: str, source: str, module_path: str, imports: Sequence[str] = ()
    ) -> AddResult:
        """Create a new top-level symbol in an existing or new module file."""
        if symbol_id in self.ir.symbols:
            raise ChangeRejected(f"{symbol_id} is already indexed; use change")
        new_src = textwrap.dedent(source).strip("\n")
        try:
            new_tree = ast.parse(new_src)
        except SyntaxError as exc:
            raise ChangeRejected(f"new source does not parse: {exc.msg}") from None
        short_name = symbol_id.rsplit(":", 1)[-1]
        if len(new_tree.body) != 1 or _defined_name(new_tree.body[0]) != short_name:
            raise ChangeRejected(f"new source must define exactly one symbol named {short_name}")

        path = self.root / module_path
        created_file = not path.exists()
        if created_file:
            text = "".join(f"{imp}\n" for imp in imports) + ("\n" if imports else "") + new_src + "\n"
        else:
            text = path.read_text()
            if imports:
                text = _splice_imports(text, imports)
            text = text.rstrip("\n") + "\n\n\n" + new_src + "\n"
        try:
            ast.parse(text)
        except SyntaxError as exc:
            raise ChangeRejected(f"file would not parse after add: {exc.msg}") from None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        self.ir = compile_repository(self.root)
        if symbol_id not in self.ir.symbols:
            raise ChangeRejected(f"{symbol_id} was not indexed after writing {module_path}")
        return AddResult(self.ir.symbols[symbol_id], created_file)

    def inspect(self, symbol_id: str) -> str:
        """One-call bundle for a known symbol: L2 rendering plus callers."""
        self._symbol(symbol_id)
        text = self.get_summary(symbol_id)
        callers = self.get_callers(symbol_id)
        if callers:
            text += "\ncallers:"
            text += "\n" + "\n".join(f"  {f.subject} {f.confidence:.2f}" for f in callers[:10])
        return text

    def task_packet(self, description: str) -> str:
        """Local planner: resolve a description and assemble the change bundle.

        One frontier round-trip instead of several: target resolution, L2
        semantics, callers, source span, and referencing tests.
        """
        matches = [s for s in self.find_symbols(description) if s.kind != "constant"]
        if not matches:
            matches = self.find_symbols(description)
        if not matches:
            raise ChangeRejected(f"no symbol matches {description!r}")
        sym = matches[0]
        text = f"target {sym.id}\nsignature: {sym.signature}\n{self.inspect(sym.id)}\n"
        text += f"source: {sym.source.path}:{sym.source.start_line}-{sym.source.end_line}\n"
        short = sym.name.rsplit(".", 1)[-1]
        tests_dir = self.root / "tests"
        if tests_dir.is_dir():
            referencing = sorted(
                {
                    p.relative_to(self.root).as_posix()
                    for p in tests_dir.rglob("*.py")
                    if short in p.read_text()
                }
            )
            text += "tests: " + (", ".join(referencing[:5]) if referencing else "(none)") + "\n"
        else:
            text += "tests: (none)\n"
        return text

    def patch(self, file_path: str, start: int, end: int, source: str) -> PatchResult:
        """Escape hatch: replace a 1-based inclusive line range, keeping the file parseable.

        Covers everything symbol-level ops cannot express (module docstrings,
        import lines, test files); the parse check is the only guardrail.
        """
        path = self.root / file_path
        if not path.is_file():
            raise ChangeRejected(f"unknown file {file_path}")
        if start < 1 or end < start or end > len(path.read_text().splitlines()):
            raise ChangeRejected("invalid line range")
        new_src = textwrap.dedent(source).strip("\n")
        if not new_src:
            raise ChangeRejected("empty patch; use the remove op to delete symbols")
        try:
            ast.parse(new_src)
        except SyntaxError as exc:
            raise ChangeRejected(f"patch does not parse: {exc.msg}") from None
        lines = path.read_text().splitlines()
        updated = lines[: start - 1] + new_src.splitlines() + lines[end:]
        text = "\n".join(updated) + "\n"
        try:
            ast.parse(text)
        except SyntaxError as exc:
            raise ChangeRejected(f"file would not parse after patch: {exc.msg}") from None
        path.write_text(text)
        self.ir = compile_repository(self.root)
        return PatchResult(file_path, end - start + 1)

    def remove(self, symbol_id: str) -> RemoveResult:
        """Delete one indexed symbol's span; report symbols that still reference it."""
        sym = self._symbol(symbol_id)
        dependents = tuple(
            sorted(
                {f.subject for f in self.ir.facts if f.object == symbol_id and f.subject != symbol_id}
            )
        )
        path = self.root / sym.source.path
        lines = path.read_text().splitlines()
        original = lines[sym.source.start_line - 1]
        indent = original[: len(original) - len(original.lstrip())]

        def spliced(fill: list[str]) -> str:
            body = lines[: sym.source.start_line - 1] + fill + lines[sym.source.end_line :]
            return "\n".join(body) + ("\n" if body else "")

        # Removing a class's last member would leave an empty body; keep the
        # file parseable by splicing in ``pass`` at the same indentation.
        text = spliced([])
        if sym.kind == "method":
            try:
                ast.parse(text)
            except SyntaxError:
                text = spliced([f"{indent}pass"])
        try:
            ast.parse(text)
        except SyntaxError as exc:
            raise ChangeRejected(f"file would not parse after removal: {exc.msg}") from None
        path.write_text(text)
        self.ir = compile_repository(self.root)
        return RemoveResult(sym, dependents)

    def verify(self, argv: list[str], timeout: float = 300.0, tail_lines: int = 20) -> VerifyResult:
        proc = subprocess.run(argv, cwd=self.root, capture_output=True, text=True, timeout=timeout)
        output = (proc.stdout + proc.stderr).rstrip().splitlines()
        return VerifyResult(proc.returncode == 0, proc.returncode, "\n".join(output[-tail_lines:]))


def tool_surface_tokens() -> int:
    return estimate_tokens(tool_surface_json())
