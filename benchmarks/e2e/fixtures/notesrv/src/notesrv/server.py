"""Server glue: a routing table over the handlers plus an http.server adapter.

The pure ``dispatch`` function is what tests exercise; the BaseHTTPRequestHandler
subclass only parses the socket side and delegates to it.
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler

from notesrv.auth import AuthError
from notesrv.handlers import Handlers
from notesrv.jsonutil import dumps, exception_to_response, loads
from notesrv.routing import RouteTable

READ_ACTIONS = {"GET"}
WRITE_ACTIONS = {"POST", "PUT", "PATCH"}
DELETE_ACTIONS = {"DELETE"}


def build_routes(handlers: Handlers) -> RouteTable:
    """Wire the standard notes API surface."""
    table = RouteTable()
    table.add_route("POST", "/notes", handlers.handle_create)
    table.add_route("GET", "/notes", handlers.handle_list)
    table.add_route("GET", "/notes/<note_id>", handlers.handle_get)
    table.add_route("PUT", "/notes/<note_id>", handlers.handle_update)
    table.add_route("DELETE", "/notes/<note_id>", handlers.handle_delete)
    return table


def dispatch(
    routes: RouteTable,
    handlers: Handlers,
    method: str,
    path: str,
    body: str,
    token: str | None = None,
) -> tuple[int, str]:
    """Resolve a route and run it; returns (status, json_text)."""
    status, payload = _dispatch_payload(routes, handlers, method, path, body, token)
    return status, dumps(payload)


def _dispatch_payload(
    routes: RouteTable,
    handlers: Handlers,
    method: str,
    path: str,
    body: str,
    token: str | None,
):
    resolved = routes.resolve(method, path)
    if resolved is None:
        return 404, {"error": "no route"}
    try:
        note_id = int(resolved.params["note_id"]) if "note_id" in resolved.params else None
    except ValueError:
        return 400, {"error": "note id must be an integer"}

    if method == "GET":
        if note_id is None:
            return handlers.handle_list(token=token, tag=_query_tag(path))
        return handlers.handle_get(token=token, note_id=note_id)
    if method in WRITE_ACTIONS:
        try:
            parsed = loads(body) if body else {}
        except ValueError as exc:
            return exception_to_response(exc)
        if note_id is None:
            return handlers.handle_create(token=token, body=parsed)
        return handlers.handle_update(token=token, note_id=note_id, body=parsed)
    if method in DELETE_ACTIONS:
        return handlers.handle_delete(token=token, note_id=note_id)
    return exception_to_response(AuthError(f"unsupported method: {method}"))


def _query_tag(path: str) -> str | None:
    """Extract ?tag= from a request path, or None."""
    if "?" not in path:
        return None
    query = path.split("?", 1)[1]
    for pair in query.split("&"):
        if pair.startswith("tag="):
            return pair[4:] or None
    return None


def make_handler_class(routes: RouteTable, handlers: Handlers) -> type[BaseHTTPRequestHandler]:
    """Build a BaseHTTPRequestHandler subclass bound to a route table."""

    class NotesRequestHandler(BaseHTTPRequestHandler):
        def _handle(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length).decode("utf-8") if length else ""
            status, payload_text = dispatch(routes, handlers, self.command, self.path, body)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            if payload_text:
                self.wfile.write(payload_text.encode("utf-8"))

        def do_GET(self) -> None:  # noqa: N802 - stdlib naming
            self._handle()

        def do_POST(self) -> None:  # noqa: N802
            self._handle()

        def do_PUT(self) -> None:  # noqa: N802
            self._handle()

        def do_DELETE(self) -> None:  # noqa: N802
            self._handle()

        def log_message(self, format: str, *args) -> None:  # noqa: A002
            pass  # keep benchmark runs quiet

    return NotesRequestHandler
