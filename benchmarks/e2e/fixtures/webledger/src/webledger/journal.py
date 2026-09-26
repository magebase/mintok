"""The journal: validated posting and per-account queries."""

from __future__ import annotations

from webledger.accounts import ChartOfAccounts
from webledger.entries import EntryLine, JournalEntry
from webledger.validation import ValidationError, validate_date, validate_entry


class Journal:
    """An append-only list of validated entries."""

    def __init__(self, chart: ChartOfAccounts) -> None:
        self.chart = chart
        self._entries: list[JournalEntry] = []

    def post(self, entry: JournalEntry) -> JournalEntry:
        validate_entry(entry, self.chart)
        self._entries.append(entry)
        return entry

    def post_simple(self, date: str, memo: str, debit_account: str, credit_account: str, cents: int) -> JournalEntry:
        """Convenience for two-line entries."""
        return self.post(
            JournalEntry(
                date=date,
                memo=memo,
                lines=[
                    EntryLine(debit_account, debit_cents=cents),
                    EntryLine(credit_account, credit_cents=cents),
                ],
            )
        )

    def entries(self) -> list[JournalEntry]:
        return list(self._entries)

    def entries_for_account(self, account_number: str) -> list[JournalEntry]:
        """Entries touching the account, in posting order."""
        return [
            e
            for e in self._entries
            if any(line.account_number == account_number for line in e.lines)
        ]

    def balance(self, account_number: str) -> int:
        """Net debit-positive balance of one account."""
        total = 0
        for entry in self._entries:
            for line in entry.lines:
                if line.account_number == account_number:
                    total += line.signed_cents
        return total

    def net_income_cents(self) -> int:
        """Income minus expenses across all posted entries."""
        income = 0
        for entry in self._entries:
            for line in entry.lines:
                account = self.chart.get(line.account_number)
                if account.type.value == "income":
                    income += line.credit_cents - line.debit_cents
                elif account.type.value == "expense":
                    income -= line.debit_cents - line.credit_cents
        return income
