"""Record and batch primitives flowing through a pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Record:
    """One unit of data: a key, a value, and free-form metadata."""

    key: str
    value: str
    meta: dict = field(default_factory=dict)

    def with_meta(self, **extra) -> "Record":
        merged = dict(self.meta)
        merged.update(extra)
        return Record(key=self.key, value=self.value, meta=merged)

    def copy(self) -> "Record":
        return Record(key=self.key, value=self.value, meta=dict(self.meta))


@dataclass
class RecordBatch:
    """An ordered group of records processed together."""

    name: str
    records: list[Record] = field(default_factory=list)

    def add(self, record: Record) -> None:
        self.records.append(record)

    def __len__(self) -> int:
        return len(self.records)

    def keys(self) -> list[str]:
        return [r.key for r in self.records]
