"""Open record formats for agent sessions and benchmark runs (JSONL).

Records carry verifiable cost and outcome data — the input to the profiler and
the benchmark. The formats are open; learning policies over the accumulated
records belongs to the commercial layer.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from mintok.metrics import RunRecord


@dataclass(frozen=True, slots=True)
class ContextRead:
    """One context item read ``count`` times within a session."""

    key: str
    tokens: int
    count: int = 1


@dataclass(frozen=True, slots=True)
class SessionRecord:
    """One agent session: task, model, tokens, cost, and outcome."""

    session_id: str
    task_id: str
    task_class: str
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    tool_calls: int = 0
    usd: float = 0.0
    solved: bool = False
    context_reads: tuple[ContextRead, ...] = ()

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cache_read_ratio(self) -> float:
        return self.cache_read_tokens / self.input_tokens if self.input_tokens else 1.0

    @property
    def cost_per_token(self) -> float:
        return self.usd / self.total_tokens if self.total_tokens else 0.0


def _write_jsonl(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def dump_sessions_jsonl(sessions: Iterable[SessionRecord], path: str | Path) -> None:
    _write_jsonl([asdict(s) for s in sessions], path)


def load_sessions_jsonl(path: str | Path) -> list[SessionRecord]:
    sessions: list[SessionRecord] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        reads = tuple(ContextRead(**read) for read in row.pop("context_reads", []))
        sessions.append(SessionRecord(**row, context_reads=reads))
    return sessions


def dump_runs_jsonl(records: Iterable[RunRecord], path: str | Path) -> None:
    _write_jsonl([asdict(r) for r in records], path)


def load_runs_jsonl(path: str | Path) -> list[RunRecord]:
    runs: list[RunRecord] = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            runs.append(RunRecord(**json.loads(line)))
    return runs
