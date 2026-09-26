"""Readers turn raw sources into Record objects."""

from __future__ import annotations

import json

from pipeline.errors import ConfigurationError
from pipeline.records import Record, RecordBatch


class Reader:
    """Base class for all readers."""

    name = "base"

    def read(self) -> RecordBatch:
        raise NotImplementedError


class MemoryReader(Reader):
    """Reads from an in-memory list of (key, value) pairs."""

    name = "memory"

    def __init__(self, pairs: list[tuple[str, str]], batch_name: str = "memory") -> None:
        self.pairs = list(pairs)
        self.batch_name = batch_name

    def read(self) -> RecordBatch:
        batch = RecordBatch(name=self.batch_name)
        for key, value in self.pairs:
            batch.add(Record(key=key, value=value))
        return batch


class CsvReader(Reader):
    """Parses CSV text with a header row into records."""

    name = "csv"

    def __init__(self, text: str, batch_name: str = "csv") -> None:
        self.text = text
        self.batch_name = batch_name

    def read(self) -> RecordBatch:
        lines = [ln for ln in self.text.splitlines() if ln.strip()]
        if not lines:
            return RecordBatch(name=self.batch_name)
        header = lines[0].split(",")
        if len(set(header)) != len(header):
            raise ConfigurationError("duplicate column names")
        batch = RecordBatch(name=self.batch_name)
        for line in lines[1:]:
            cells = line.split(",")
            row = dict(zip(header, cells))
            key = row.pop("key", "")
            value = row.pop("value", "")
            batch.add(Record(key=key, value=value, meta=row))
        return batch


class JsonlReader(Reader):
    """Parses one JSON object per line into records."""

    name = "jsonl"

    def __init__(self, text: str, batch_name: str = "jsonl") -> None:
        self.text = text
        self.batch_name = batch_name

    def read(self) -> RecordBatch:
        batch = RecordBatch(name=self.batch_name)
        for line in self.text.splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            batch.add(Record(key=obj["key"], value=obj["value"], meta=obj.get("meta", {})))
        return batch
