"""Note domain model."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class NoteError(ValueError):
    """Raised when note payloads are malformed."""


@dataclass
class Note:
    """A note with a title, body and tags. ``note_id`` is assigned by storage."""

    title: str
    body: str
    tags: list[str] = field(default_factory=list)
    note_id: int = 0

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise NoteError("title must not be blank")
        self.tags = [t.strip().lower() for t in self.tags if t.strip()]

    def to_dict(self) -> dict:
        return {
            "id": self.note_id,
            "title": self.title,
            "body": self.body,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Note":
        try:
            return cls(
                title=payload["title"],
                body=payload.get("body", ""),
                tags=list(payload.get("tags", [])),
            )
        except KeyError as exc:
            raise NoteError(f"missing field: {exc}") from None

    def matches(self, query: str) -> bool:
        """Case-insensitive substring match over title and body."""
        needle = query.lower()
        return needle in self.title.lower() or needle in self.body.lower()
