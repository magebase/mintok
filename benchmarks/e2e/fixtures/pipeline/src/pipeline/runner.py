"""Pipeline runner: reader -> transforms -> sink."""

from __future__ import annotations

from dataclasses import dataclass

from pipeline.readers import Reader
from pipeline.records import RecordBatch
from pipeline.sinks import Sink
from pipeline.transforms import Transform
from pipeline.validate import validate_batch


@dataclass
class RunStats:
    processed: int = 0
    skipped: int = 0
    invalid: int = 0


class Pipeline:
    """Runs a reader through optional validation rules and transforms into a sink."""

    def __init__(
        self,
        reader: Reader,
        sink: Sink,
        transforms: list[Transform] | None = None,
        rules: list | None = None,
    ) -> None:
        self.reader = reader
        self.sink = sink
        self.transforms = transforms or []
        self.rules = rules or []

    def run(self) -> RunStats:
        stats = RunStats()
        batch = self.reader.read()

        if self.rules:
            batch, skipped = validate_batch(batch, self.rules)
            stats.skipped = len(skipped)

        for step in self.transforms:
            batch = self._apply(batch, step)

        self.sink.write(batch)
        self.sink.close()
        stats.processed = len(batch)
        return stats

    def _apply(self, batch: RecordBatch, step: Transform) -> RecordBatch:
        from pipeline.errors import SkipRecord

        out = RecordBatch(name=batch.name)
        for record in batch.records:
            try:
                out.add(step(record))
            except SkipRecord:
                continue
        return out
