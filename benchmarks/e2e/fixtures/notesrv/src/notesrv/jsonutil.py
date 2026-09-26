"""JSON helpers shared by handlers and the server glue."""

from __future__ import annotations

import json

from notesrv.auth import AuthError
from notesrv.models import NoteError
from notesrv.storage import NotFoundError


def dumps(payload: dict) -> str:
    """Compact JSON with sorted keys for deterministic responses."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def loads(text: str) -> dict:
    """Parse a JSON object body; ValueError on malformed input."""
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("body must be a JSON object")
    return obj


def error_response(status: int, message: str) -> tuple[int, dict]:
    """Standard error envelope."""
    return status, {"error": message}


def exception_to_response(exc: Exception) -> tuple[int, dict]:
    """Map domain exceptions to HTTP-style (status, payload) pairs."""
    if isinstance(exc, NotFoundError):
        return error_response(404, "note not found")
    if isinstance(exc, AuthError):
        return error_response(403, str(exc))
    if isinstance(exc, NoteError):
        return error_response(422, str(exc))
    return error_response(400, str(exc))
