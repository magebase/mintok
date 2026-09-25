"""Python source → Agent Program IR using only the stdlib ``ast`` module."""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from mintok.ir import Fact, ProgramIR, SourceRef, Symbol, stable_hash

SKIP_DIRS = {".venv", "venv", "__pycache__", "node_modules", "build", "dist"}

FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef


@dataclass(slots=True)
class _Def:
    id: str
    kind: str
    node: ast.AST
    class_id: str | None = None


@dataclass(slots=True)
class _Module:
    name: str
    path: str
    tree: ast.Module
    defs: dict[str, _Def] = field(default_factory=dict)
    top_level: dict[str, str] = field(default_factory=dict)
    imports: dict[str, tuple[str, str]] = field(default_factory=dict)


def module_name(relative: Path) -> str:
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def iter_python_files(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root)
        if any(p in SKIP_DIRS or p.startswith(".") for p in rel.parts[:-1]):
            continue
        yield path


def compile_repository(root: str | Path) -> ProgramIR:
    root = Path(root)
    ir = ProgramIR()
    modules: list[_Module] = []

    for path in iter_python_files(root):
        rel = path.relative_to(root)
        try:
            tree = ast.parse(path.read_text(), filename=str(rel))
        except SyntaxError as exc:
            ir.diagnostics.append(f"{rel}:{exc.lineno}: syntax error: {exc.msg}")
            continue
        modules.append(_collect(module_name(rel), rel.as_posix(), tree))

    index: dict[str, _Def] = {d.id: d for m in modules for d in m.defs.values()}
    pkg_imports: dict[str, dict[str, tuple[str, str]]] = {m.name: m.imports for m in modules}
    methods_by_name: dict[str, list[str]] = {}
    for d in index.values():
        if d.kind == "method":
            methods_by_name.setdefault(d.id.rsplit(".", 1)[1], []).append(d.id)

    for module in modules:
        for d in module.defs.values():
            facts = _extract_facts(d, module, index, methods_by_name, pkg_imports)
            ir.facts.extend(facts)
            ir.symbols[d.id] = _make_symbol(d, module, facts)
    return ir


def _collect(name: str, path: str, tree: ast.Module) -> _Module:
    module = _Module(name=name, path=path, tree=tree)
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            for alias in node.names:
                module.imports[alias.asname or alias.name] = (node.module, alias.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sid = f"{name}:{node.name}"
            module.defs[sid] = _Def(sid, "function", node)
            module.top_level[node.name] = sid
        elif isinstance(node, ast.ClassDef):
            cid = f"{name}:{node.name}"
            module.defs[cid] = _Def(cid, "class", node)
            module.top_level[node.name] = cid
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    mid = f"{cid}.{item.name}"
                    module.defs[mid] = _Def(mid, "method", item, class_id=cid)
    return module


def _walk_local(node: ast.AST) -> Iterator[ast.AST]:
    """Walk a definition's body without descending into nested definitions."""
    stack = list(ast.iter_child_nodes(node))
    while stack:
        child = stack.pop()
        yield child
        if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            stack.extend(ast.iter_child_nodes(child))


def _resolve_name(
    name: str,
    module: _Module,
    index: dict[str, _Def],
    pkg_imports: dict[str, dict[str, tuple[str, str]]] | None = None,
) -> str | None:
    if name in module.top_level:
        return module.top_level[name]
    if name in module.imports:
        src_module, src_name = module.imports[name]
        # Follow package-level re-exports (``from pkg.impl import Thing`` in a
        # package ``__init__``), bounded so import cycles terminate.
        for _ in range(3):
            candidate = f"{src_module}:{src_name}"
            if candidate in index:
                return candidate
            next_map = (pkg_imports or {}).get(src_module)
            if next_map and src_name in next_map:
                src_module, src_name = next_map[src_name]
            else:
                return None
    return None


def _annotation_types(
    fn: FunctionNode,
    module: _Module,
    index: dict[str, _Def],
    pkg_imports: dict[str, dict[str, tuple[str, str]]],
) -> dict[str, str]:
    types: dict[str, str] = {}
    all_args = fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs
    for arg in all_args:
        if isinstance(arg.annotation, ast.Name):
            target = _resolve_name(arg.annotation.id, module, index, pkg_imports)
            if target and index[target].kind == "class":
                types[arg.arg] = target
    return types


def _resolve_call(
    call: ast.Call,
    d: _Def,
    module: _Module,
    index: dict[str, _Def],
    methods_by_name: dict[str, list[str]],
    local_types: dict[str, str],
    pkg_imports: dict[str, dict[str, tuple[str, str]]],
) -> list[tuple[str, float]]:
    func = call.func
    if isinstance(func, ast.Name):
        target = _resolve_name(func.id, module, index, pkg_imports)
        return [(target, 1.0)] if target else []
    if not isinstance(func, ast.Attribute):
        return []

    receiver_class: str | None = None
    value = func.value
    if isinstance(value, ast.Name):
        if value.id == "self" and d.class_id:
            receiver_class = d.class_id
        elif value.id in local_types:
            receiver_class = local_types[value.id]
        else:
            resolved = _resolve_name(value.id, module, index, pkg_imports)
            if resolved and index[resolved].kind == "class":
                receiver_class = resolved
    elif isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
        resolved = _resolve_name(value.func.id, module, index, pkg_imports)
        if resolved and index[resolved].kind == "class":
            receiver_class = resolved

    if receiver_class:
        target = f"{receiver_class}.{func.attr}"
        return [(target, 1.0)] if target in index else []

    candidates = methods_by_name.get(func.attr, [])
    return [(c, 1.0 / len(candidates)) for c in sorted(candidates)]


def _exception_name(exc: ast.expr | None) -> str | None:
    if exc is None:
        return None
    if isinstance(exc, ast.Call):
        exc = exc.func
    return ast.unparse(exc)


def _extract_facts(
    d: _Def,
    module: _Module,
    index: dict[str, _Def],
    methods_by_name: dict[str, list[str]],
    pkg_imports: dict[str, dict[str, tuple[str, str]]],
) -> list[Fact]:
    if d.kind == "class":
        return []
    fn = d.node
    assert isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
    local_types = _annotation_types(fn, module, index, pkg_imports)
    calls: dict[str, float] = {}
    raises: set[str] = set()
    writes: set[str] = set()

    for node in _walk_local(fn):
        if isinstance(node, ast.Call):
            for target, confidence in _resolve_call(
                node, d, module, index, methods_by_name, local_types, pkg_imports
            ):
                if target != d.id:
                    calls[target] = max(calls.get(target, 0.0), confidence)
        elif isinstance(node, ast.Raise):
            name = _exception_name(node.exc)
            if name:
                raises.add(name)
        elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)) and d.class_id:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self":
                    writes.add(f"{d.class_id}.{t.attr}")

    facts = [Fact(d.id, "calls", t, round(c, 4)) for t, c in sorted(calls.items())]
    facts += [Fact(d.id, "raises", r) for r in sorted(raises)]
    facts += [Fact(d.id, "writes", w) for w in sorted(writes)]
    return facts


def _strip_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        body = body[1:]
    return body or [ast.Pass()]


def _normalized_dump(node: ast.AST) -> str:
    node = copy.deepcopy(node)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        node.body = _strip_docstring(node.body)
    elif isinstance(node, ast.ClassDef):
        # Method bodies are hashed on their own symbols; the class hash covers only its shape.
        node.body = _strip_docstring(node.body)
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                item.body = [ast.Pass()]
    return ast.dump(node, include_attributes=False)


def _signature(node: ast.AST) -> str:
    if isinstance(node, ast.ClassDef):
        bases = ", ".join(ast.unparse(b) for b in node.bases)
        return f"class {node.name}({bases})" if bases else f"class {node.name}"
    assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
    sig = f"{prefix}{node.name}({ast.unparse(node.args)})"
    if node.returns is not None:
        sig += f" -> {ast.unparse(node.returns)}"
    return sig


def _make_symbol(d: _Def, module: _Module, facts: list[Fact]) -> Symbol:
    node = d.node
    signature = _signature(node)
    if isinstance(node, ast.ClassDef):
        members = sorted(
            item.name for item in node.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
        interface = stable_hash(signature, *members)
    else:
        interface = stable_hash(signature, *(f"{f.predicate}|{f.object}|{f.confidence}" for f in facts))
    return Symbol(
        id=d.id,
        kind=d.kind,
        name=d.id.split(":", 1)[1],
        module=module.name,
        signature=signature,
        source=SourceRef(module.path, node.lineno, node.end_lineno or node.lineno),
        body_hash=stable_hash(_normalized_dump(node)),
        interface_hash=interface,
    )
