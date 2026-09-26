"""Sinks collect the output of a pipeline run."""

from __future__ import annotations

from pathlib import Path

from pipeline.records import RecordBatch


class Sink:
    """Base class for all sinks."""

    name = "base"
    accepts_batches = False

    def write(self, item) -> None:
        raise NotImplementedError

    def close(self) -> None:
        """Flush and finalize; default is a no-op."""


class ListSink(Sink):
    """Accumulates records in a list."""

    name = "list"

    def __init__(self) -> None:
        self.records = []

    def write(self, item) -> None:
        if isinstance(item, RecordBatch):
            self.records.extend(item.records)
        else:
            self.records.append(item)


class CountingSink(Sink):
    """Counts records without retaining them."""

    name = "counting"

    def __init__(self) -> None:
        self.count = 0

    def write(self, item) -> None:
        if isinstance(item, RecordBatch):
            self.count += len(item.records)
        else:
            self.count += 1


class FileSink(Sink):
    """Writes one '<key>=<value>' line per record to a text file."""

    name = "file"

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lines: list[str] = []

    def write(self, item) -> None:
        if isinstance(item, RecordBatch):
            for record in item.records:
                self._lines.append(f"{record.key}={record.value}")
        else:
            self._lines.append(f"{item.key}={item.value}")

    def close(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("\n".join(self._lines) + ("\n" if self._lines else ""))
