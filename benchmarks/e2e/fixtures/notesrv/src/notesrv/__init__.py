"""Notesrv: a tiny stdlib HTTP note service (handlers, storage, auth stub, routing)."""

from notesrv.storage import MemoryStore

__all__ = ["MemoryStore"]
