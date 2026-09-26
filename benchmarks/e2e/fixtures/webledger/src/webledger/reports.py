"""Financial reports rendered from a journal and chart of accounts."""

from __future__ import annotations

from webledger.accounts import AccountType, ChartOfAccounts
from webledger.journal import Journal
from webledger.money import format_amount


def trial_balance(journal: Journal) -> dict[str, int]:
    """Net debit-positive balance per account number, sorted, non-zero only."""
    numbers = set()
    for entry in journal.entries():
        for line in entry.lines:
            numbers.add(line.account_number)
    balances = {n: journal.balance(n) for n in numbers}
    return {n: b for n, b in sorted(balances.items()) if b != 0}


def income_statement(journal: Journal) -> dict[str, int]:
    """Totals per income/expense account plus the net result."""
    statement: dict[str, int] = {}
    for entry in journal.entries():
        for line in entry.lines:
            account = journal.chart.get(line.account_number)
            if account.type is AccountType.INCOME:
                statement[account.number] = statement.get(account.number, 0) - line.signed_cents
            elif account.type is AccountType.EXPENSE:
                statement[account.number] = statement.get(account.number, 0) + line.signed_cents
    statement["net"] = journal.net_income_cents()
    return statement


def render_trial_balance(journal: Journal, chart: ChartOfAccounts) -> str:
    """Fixed-width text report: number, name, balance."""
    lines = ["TRIAL BALANCE", "-" * 40]
    for number, balance in trial_balance(journal).items():
        account = chart.get(number)
        lines.append(f"{number}  {account.name:<24} {format_amount(balance):>12}")
    lines.append("-" * 40)
    return "\n".join(lines)
