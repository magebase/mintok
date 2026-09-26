"""Transforms map or filter single records; compose chains them."""

from __future__ import annotations

from typing import Callable

from pipeline.errors import SkipRecord
from pipeline.records import Record

Transform = Callable[[Record], Record]


def uppercase_value(record: Record) -> Record:
    """Uppercase the record value."""
    out = record.copy()
    out.value = out.value.upper()
    return out


def trim_value(record: Record) -> Record:
    """Strip surrounding whitespace from the value."""
    out = record.copy()
    out.value = out.value.strip()
    return out


def prefix_key(prefix: str) -> Transform:
    """Return a transform that prefixes every record key."""

    def transform(record: Record) -> Record:
        out = record.copy()
        out.key = f"{prefix}{out.key}"
        return out

    return transform


def drop_empty() -> Transform:
    """Drop records whose value is empty after trimming."""

    def transform(record: Record) -> Record:
        if not record.value.strip():
            raise SkipRecord("empty value")
        return record

    return transform


def annotate_meta(**fields) -> Transform:
    """Attach constant metadata to every record."""

    def transform(record: Record) -> Record:
        return record.with_meta(**fields)

    return transform


def compose(*transforms: Transform) -> Transform:
    """Left-to-right composition of transforms."""

    def transform(record: Record) -> Record:
        for step in transforms:
            record = step(record)
        return record

    return transform
