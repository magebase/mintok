"""Request handlers: pure functions from (params, body) to (status, payload)."""

from __future__ import annotations

from notesrv.auth import AuthError, require_token
from notesrv.jsonutil import exception_to_response
from notesrv.models import Note, NoteError
from notesrv.storage import MemoryStore, NotFoundError


class Handlers:
    """Binds a store to the CRUD handlers used by the routing table."""

    def __init__(self, store: MemoryStore | None = None) -> None:
        self.store = store if store is not None else MemoryStore()

    def handle_create(self, token: str | None, body: dict) -> tuple[int, dict]:
        try:
            require_token(token, "write")
            note = self.store.create(Note.from_dict(body))
            return 201, note.to_dict()
        except (AuthError, NotFoundError, NoteError) as exc:
            return exception_to_response(exc)

    def handle_get(self, token: str | None, note_id: int) -> tuple[int, dict]:
        try:
            require_token(token, "read")
            return 200, self.store.get(note_id).to_dict()
        except (AuthError, NotFoundError, NoteError) as exc:
            return exception_to_response(exc)

    def handle_list(self, token: str | None, tag: str | None = None) -> tuple[int, dict]:
        try:
            require_token(token, "read")
            notes = self.store.list(tag=tag)
            return 200, {"notes": [n.to_dict() for n in notes]}
        except AuthError as exc:
            return exception_to_response(exc)

    def handle_update(self, token: str | None, note_id: int, body: dict) -> tuple[int, dict]:
        try:
            require_token(token, "write")
            return 200, self.store.update(note_id, body).to_dict()
        except (AuthError, NotFoundError, NoteError) as exc:
            return exception_to_response(exc)

    def handle_delete(self, token: str | None, note_id: int) -> tuple[int, dict]:
        try:
            require_token(token, "delete")
            deleted = self.store.delete(note_id)
            if not deleted:
                raise NotFoundError(str(note_id))
            return 204, {}
        except (AuthError, NotFoundError) as exc:
            return exception_to_response(exc)
