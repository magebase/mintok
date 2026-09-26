"""Journal entry primitives: lines and balanced entries."""

from __future__ import annotations

from dataclasses import dataclass, field

from webledger.money import is_zero


@dataclass
class EntryLine:
    """One leg of a journal entry: exactly one of debit/credit is non-zero."""

    account_number: str
    debit_cents: int = 0
    credit_cents: int = 0

    def __post_init__(self) -> None:
        if self.debit_cents < 0 or self.credit_cents < 0:
            raise ValueError("amounts must be non-negative")
        if self.debit_cents and self.credit_cents:
            raise ValueError("a line cannot be both debit and credit")
        if is_zero(self.debit_cents) and is_zero(self.credit_cents):
            raise ValueError("a line must carry an amount")

    @property
    def signed_cents(self) -> int:
        """Positive for debits, negative for credits."""
        return self.debit_cents - self.credit_cents


@dataclass
class JournalEntry:
    """A dated, memo-tagged group of balanced lines."""

    date: str  # ISO date string, e.g. "2024-03-01"
    memo: str
    lines: list[EntryLine] = field(default_factory=list)

    @property
    def total_debits(self) -> int:
        return sum(line.debit_cents for line in self.lines)

    @property
    def total_credits(self) -> int:
        return sum(line.credit_cents for line in self.lines)

    @property
    def balanced(self) -> bool:
        return self.total_debits == self.total_credits and self.total_debits > 0
