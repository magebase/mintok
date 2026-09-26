"""Entry validation rules."""

from __future__ import annotations

from webledger.accounts import ChartOfAccounts
from webledger.entries import JournalEntry


class ValidationError(Exception):
    """Raised when a journal entry violates a ledger rule."""


def validate_entry(entry: JournalEntry, chart: ChartOfAccounts) -> None:
    """Raise ValidationError with a precise message for the first problem."""
    if not entry.lines:
        raise ValidationError("entry has no lines")
    if not entry.balanced:
        raise ValidationError(
            f"unbalanced entry: debits {entry.total_debits} != credits {entry.total_credits}"
        )
    for line in entry.lines:
        if not chart.has(line.account_number):
            raise ValidationError(f"unknown account: {line.account_number}")
    seen = {line.account_number for line in entry.lines}
    if len(seen) < 2:
        raise ValidationError("entry must touch at least two accounts")


def validate_date(date: str) -> None:
    """Dates must be ISO yyyy-mm-dd; kept separate so importers can reuse it."""
    parts = date.split("-")
    if len(parts) != 3 or len(parts[0]) != 4 or not all(p.isdigit() for p in parts):
        raise ValidationError(f"date must be ISO yyyy-mm-dd: {date!r}")
