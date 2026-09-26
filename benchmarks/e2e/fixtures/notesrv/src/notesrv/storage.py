"""In-memory note storage."""

from __future__ import annotations

from notesrv.models import Note


class NotFoundError(KeyError):
    """Raised when a note id does not exist."""


class MemoryStore:
    """Durable enough for a demo; ids are sequential starting at 1."""

    def __init__(self) -> None:
        self._notes: dict[int, Note] = {}
        self._next_id = 1

    def create(self, note: Note) -> Note:
        note.note_id = self._next_id
        self._next_id += 1
        self._notes[note.note_id] = note
        return note

    def get(self, note_id: int) -> Note:
        try:
            return self._notes[note_id]
        except KeyError:
            raise NotFoundError(str(note_id)) from None

    def list(self, tag: str | None = None) -> list[Note]:
        """All notes ordered by id; optionally filtered by tag."""
        notes = [self._notes[i] for i in sorted(self._notes)]
        if tag is None:
            return notes
        wanted = tag.strip().lower()
        return [n for n in notes if wanted in n.tags]

    def update(self, note_id: int, payload: dict) -> Note:
        note = self.get(note_id)
        merged = note.to_dict()
        merged.update(payload)
        merged["id"] = note_id
        updated = Note(
            title=merged["title"],
            body=merged["body"],
            tags=merged["tags"],
            note_id=note_id,
        )
        self._notes[note_id] = updated
        return updated

    def delete(self, note_id: int) -> bool:
        if note_id not in self._notes:
            return False
        del self._notes[note_id]
        return True

    def search(self, query: str) -> list[Note]:
        return [n for n in self.list() if n.matches(query)]
