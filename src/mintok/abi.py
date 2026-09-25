"""Agent ABI: one compact interface over the semantic IR (query / change / verify)."""

from __future__ import annotations

import ast
import json
import subprocess
import textwrap
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
        "description": "Replace one symbol's definition with new source",
        "params": {"target": "symbol id", "source": "str"},
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


@dataclass(frozen=True, slots=True)
class ChangeResult:
    symbol: Symbol
    interface_changed: bool


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
        if sym.kind == "class":
            raise ChangeRejected("class-level replacement is not supported; change methods individually")
        new_src = textwrap.dedent(source).strip("\n")
        try:
            new_tree = ast.parse(new_src)
        except SyntaxError as exc:
            raise ChangeRejected(f"new source does not parse: {exc.msg}") from None
        short_name = sym.name.rsplit(".", 1)[-1]
        if (
            len(new_tree.body) != 1
            or not isinstance(new_tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef))
            or new_tree.body[0].name != short_name
        ):
            raise ChangeRejected(f"new source must define exactly one function named {short_name}")
        if new_tree.body[0].decorator_list:
            raise ChangeRejected("decorators are outside the replaced range; keep them unchanged")

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

    def verify(self, argv: list[str], timeout: float = 300.0, tail_lines: int = 20) -> VerifyResult:
        proc = subprocess.run(argv, cwd=self.root, capture_output=True, text=True, timeout=timeout)
        output = (proc.stdout + proc.stderr).rstrip().splitlines()
        return VerifyResult(proc.returncode == 0, proc.returncode, "\n".join(output[-tail_lines:]))


def tool_surface_tokens() -> int:
    return estimate_tokens(tool_surface_json())
