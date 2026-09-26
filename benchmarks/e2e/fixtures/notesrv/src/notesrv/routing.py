"""HTTP routing table with '<param>' path patterns."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ResolvedRoute:
    handler: object
    params: dict[str, str]


class RouteTable:
    """Exact and parameterized routes, e.g. "/notes/<id>"."""

    def __init__(self) -> None:
        self._routes: list[tuple[str, str, object]] = []

    def add_route(self, method: str, pattern: str, handler) -> None:
        method = method.upper()
        for existing_method, existing_pattern, _ in self._routes:
            if existing_method == method and existing_pattern == pattern:
                raise ValueError(f"duplicate route: {method} {pattern}")
        self._routes.append((method, pattern, handler))

    def resolve(self, method: str, path: str) -> ResolvedRoute | None:
        """Longest-pattern match wins; params extracted from '<name>' slots."""
        method = method.upper()
        best: ResolvedRoute | None = None
        best_len = -1
        for route_method, pattern, handler in self._routes:
            if route_method != method:
                continue
            params = _match(pattern, path)
            if params is not None and len(pattern) > best_len:
                best = ResolvedRoute(handler=handler, params=params)
                best_len = len(pattern)
        return best

    @property
    def routes(self) -> list[tuple[str, str]]:
        return [(m, p) for m, p, _ in self._routes]


def _match(pattern: str, path: str) -> dict[str, str] | None:
    """Match a path against a pattern; return params or None."""
    pattern_parts = pattern.strip("/").split("/") if pattern.strip("/") else []
    path_parts = path.strip("/").split("/") if path.strip("/") else []
    if len(pattern_parts) != len(path_parts):
        return None
    params: dict[str, str] = {}
    for pat, actual in zip(pattern_parts, path_parts):
        if pat.startswith("<") and pat.endswith(">"):
            params[pat[1:-1]] = actual
        elif pat != actual:
            return None
    return params
