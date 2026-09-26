"""Chart of accounts and account types."""

from __future__ import annotations

import enum
from dataclasses import dataclass

from webledger.money import format_amount


class AccountType(enum.Enum):
    ASSET = "asset"
    LIABILITY = "liability"
    EQUITY = "equity"
    INCOME = "income"
    EXPENSE = "expense"


DEBIT_NORMAL = {AccountType.ASSET, AccountType.EXPENSE}


@dataclass
class Account:
    """A ledger account identified by a numeric string."""

    number: str
    name: str
    type: AccountType

    def __post_init__(self) -> None:
        if not self.number.isdigit():
            raise ValueError(f"account number must be numeric: {self.number!r}")
        if not self.name.strip():
            raise ValueError("account name must not be blank")


class ChartOfAccounts:
    """The set of valid accounts for a ledger."""

    def __init__(self) -> None:
        self._accounts: dict[str, Account] = {}

    def add(self, account: Account) -> Account:
        if account.number in self._accounts:
            raise ValueError(f"duplicate account number: {account.number}")
        self._accounts[account.number] = account
        return account

    def get(self, number: str) -> Account:
        account = self._accounts.get(number)
        if account is None:
            raise KeyError(f"unknown account: {number}")
        return account

    def has(self, number: str) -> bool:
        return number in self._accounts

    def by_type(self, account_type: AccountType) -> list[Account]:
        return sorted(
            (a for a in self._accounts.values() if a.type is account_type),
            key=lambda a: a.number,
        )

    def default_chart(self) -> "ChartOfAccounts":
        """Populate the standard synthetic demo chart."""
        self.add(Account("100", "Cash", AccountType.ASSET))
        self.add(Account("110", "Accounts Receivable", AccountType.ASSET))
        self.add(Account("200", "Accounts Payable", AccountType.LIABILITY))
        self.add(Account("300", "Owner Equity", AccountType.EQUITY))
        self.add(Account("400", "Sales Revenue", AccountType.INCOME))
        self.add(Account("500", "Office Expense", AccountType.EXPENSE))
        return self

    def total_for(self, numbers: list[str], balances: dict[str, int]) -> str:
        """Formatted sum of balances for a list of accounts (report helper)."""
        total = sum(balances.get(n, 0) for n in numbers)
        return format_amount(total)
