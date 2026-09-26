"""Text importers that turn pasted journal data into entries.

Two parsers with deliberately similar shapes: a pipe-delimited free-form
parser and a strict CSV parser. (Duplicated patterns are part of the fixture.)
"""

from __future__ import annotations

from webledger.accounts import ChartOfAccounts
from webledger.entries import EntryLine, JournalEntry
from webledger.validation import ValidationError


def parse_entries(text: str, chart: ChartOfAccounts) -> list[JournalEntry]:
    """Parse free-form blocks:

        2024-03-01 | Office supplies
        500 debit 100
        500 credit 200

    The first line is date + memo; following lines are amounts.
    """
    entries: list[JournalEntry] = []
    date = ""
    memo = ""
    lines: list[EntryLine] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if "|" in stripped:
            if date and lines:
                entries.append(_finish(date, memo, lines))
                lines = []
            date_part, _, memo_part = stripped.partition("|")
            date = date_part.strip()
            memo = memo_part.strip()
        else:
            amount_part, direction, account = stripped.split()
            cents = int(round(float(amount_part) * 100))
            if direction == "debit":
                lines.append(EntryLine(account, debit_cents=cents))
            elif direction == "credit":
                lines.append(EntryLine(account, credit_cents=cents))
            else:
                raise ValidationError(f"unknown direction: {direction}")
    if date and lines:
        entries.append(_finish(date, memo, lines))
    for entry in entries:
        entry_chart_check(entry, chart)
    return entries


def parse_csv_lines(text: str) -> list[JournalEntry]:
    """Parse strict 'date,memo,debit_account,credit_account,cents' rows."""
    entries: list[JournalEntry] = []
    for raw in text.splitlines():
        if not raw.strip() or raw.startswith("#"):
            continue
        date, memo, debit, credit, cents = raw.split(",")
        entries.append(
            JournalEntry(
                date=date,
                memo=memo,
                lines=[
                    EntryLine(debit, debit_cents=int(cents)),
                    EntryLine(credit, credit_cents=int(cents)),
                ],
            )
        )
    return entries


def _finish(date: str, memo: str, lines: list[EntryLine]) -> JournalEntry:
    return JournalEntry(date=date, memo=memo, lines=list(lines))


def entry_chart_check(entry: JournalEntry, chart: ChartOfAccounts) -> None:
    """Importer-level account check (validation.py also checks on posting)."""
    for line in entry.lines:
        if not chart.has(line.account_number):
            raise ValidationError(f"unknown account: {line.account_number}")
