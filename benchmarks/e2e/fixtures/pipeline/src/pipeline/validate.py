"""Record validation rules."""

from __future__ import annotations

from pipeline.errors import PipelineError, SkipRecord
from pipeline.records import Record, RecordBatch


class ValidationError(PipelineError):
    """Raised when a record violates a validation rule."""


def require_key(record: Record) -> Record:
    if not record.key:
        raise SkipRecord("missing key")
    return record


def require_type(expected: type) -> "callable":
    """Require the meta field ``typed_value`` to have the given type."""

    def rule(record: Record) -> Record:
        value = record.meta.get("typed_value")
        if not isinstance(value, expected):
            raise ValidationError(f"typed_value must be {expected.__name__}")
        return record

    return rule


def validate_batch(batch: RecordBatch, rules: list) -> tuple[RecordBatch, list[tuple[str, str]]]:
    """Apply rules to every record.

    Returns (clean_batch, skipped) where skipped lists (key, reason) for
    records dropped by SkipRecord. ValidationError is not caught here.
    """
    clean = RecordBatch(name=batch.name)
    skipped: list[tuple[str, str]] = []
    for record in batch.records:
        try:
            for rule in rules:
                record = rule(record)
            clean.add(record)
        except SkipRecord as exc:
            skipped.append((record.key, str(exc)))
    return clean, skipped
