"""Fiscal periods: open/close gates around posting."""

from __future__ import annotations

from dataclasses import dataclass

from webledger.journal import Journal
from webledger.validation import ValidationError, validate_date


@dataclass
class FiscalPeriod:
    """A named period with ISO date bounds."""

    name: str
    start: str
    end: str
    closed: bool = False


class PeriodBook:
    """Routes entries to periods and refuses to touch closed ones."""

    def __init__(self, journal: Journal) -> None:
        self.journal = journal
        self.periods: dict[str, FiscalPeriod] = {}

    def open_period(self, name: str, start: str, end: str) -> FiscalPeriod:
        if name in self.periods:
            raise ValidationError(f"period already exists: {name}")
        period = FiscalPeriod(name=name, start=start, end=end)
        self.periods[name] = period
        return period

    def close_period(self, name: str) -> FiscalPeriod:
        period = self._require(name)
        period.closed = True
        return period

    def reopen_period(self, name: str) -> FiscalPeriod:
        period = self._require(name)
        period.closed = False
        return period

    def assert_postable(self, date: str) -> FiscalPeriod:
        """Find the period covering the date and assert it is open."""
        validate_date(date)
        for period in self.periods.values():
            if period.start <= date <= period.end:
                if period.closed:
                    raise ValidationError(f"period {period.name} is closed")
                return period
        raise ValidationError(f"no open period covers {date}")

    def post_to_period(self, date: str, memo: str, debit_account: str, credit_account: str, cents: int):
        """Post through the period gate; used by callers that respect close-out."""
        self.assert_postable(date)
        return self.journal.post_simple(date, memo, debit_account, credit_account, cents)

    def _require(self, name: str) -> FiscalPeriod:
        period = self.periods.get(name)
        if period is None:
            raise ValidationError(f"no such period: {name}")
        return period
