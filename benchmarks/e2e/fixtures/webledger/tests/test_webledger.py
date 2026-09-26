import pytest

from webledger.accounts import Account, AccountType, ChartOfAccounts
from webledger.entries import EntryLine, JournalEntry
from webledger.fiscal import PeriodBook
from webledger.importers import parse_csv_lines, parse_entries
from webledger.journal import Journal
from webledger.money import format_amount, to_cents
from webledger.reports import income_statement, render_trial_balance, trial_balance
from webledger.validation import ValidationError


@pytest.fixture
def chart():
    return ChartOfAccounts().default_chart()


@pytest.fixture
def journal(chart):
    return Journal(chart)


def post_sale(journal, cents=15000, date="2024-03-02"):
    journal.post_simple(date, "sale", "110", "400", cents)


def test_money_helpers():
    assert to_cents(12.34) == 1234
    assert format_amount(1234567) == "12,345.67"
    assert format_amount(-500) == "-5.00"


def test_chart_of_accounts(chart):
    assert chart.has("100")
    assert [a.number for a in chart.by_type(AccountType.ASSET)] == ["100", "110"]
    with pytest.raises(ValueError):
        chart.add(Account("100", "Duplicate", AccountType.ASSET))
    with pytest.raises(ValueError):
        chart.add(Account("abc", "Bad", AccountType.ASSET))
    with pytest.raises(KeyError):
        chart.get("999")


def test_entry_line_rules():
    with pytest.raises(ValueError):
        EntryLine("100")  # no amount
    with pytest.raises(ValueError):
        EntryLine("100", debit_cents=10, credit_cents=10)
    assert EntryLine("100", debit_cents=10).signed_cents == 10
    assert EntryLine("100", credit_cents=10).signed_cents == -10


def test_post_rejects_unbalanced(journal):
    entry = JournalEntry(
        date="2024-03-01",
        memo="oops",
        lines=[EntryLine("100", debit_cents=500), EntryLine("400", credit_cents=400)],
    )
    with pytest.raises(ValidationError, match="unbalanced"):
        journal.post(entry)


def test_post_rejects_unknown_account(journal):
    entry = JournalEntry(
        date="2024-03-01",
        memo="oops",
        lines=[EntryLine("100", debit_cents=500), EntryLine("777", credit_cents=500)],
    )
    with pytest.raises(ValidationError, match="unknown account"):
        journal.post(entry)


def test_balances_and_income(journal):
    post_sale(journal, 15000)
    journal.post_simple("2024-03-05", "supplies", "500", "100", 3200)
    assert journal.balance("110") == 15000
    assert journal.balance("400") == -15000
    assert journal.balance("500") == 3200
    assert journal.net_income_cents() == 15000 - 3200
    assert len(journal.entries_for_account("100")) == 1


def test_trial_balance_and_income_statement(journal):
    post_sale(journal)
    journal.post_simple("2024-03-05", "supplies", "500", "100", 3200)
    tb = trial_balance(journal)
    assert set(tb) == {"110", "400", "500", "100"}
    statement = income_statement(journal)
    assert statement["400"] == 15000
    assert statement["500"] == 3200
    assert statement["net"] == 11800
    report = render_trial_balance(journal, journal.chart)
    assert "TRIAL BALANCE" in report and "Sales Revenue" in report


def test_period_book_gates_posting(journal):
    book = PeriodBook(journal)
    book.open_period("2024-03", "2024-03-01", "2024-03-31")
    book.post_to_period("2024-03-10", "sale", "110", "400", 5000)
    book.close_period("2024-03")
    with pytest.raises(ValidationError, match="closed"):
        book.post_to_period("2024-03-15", "sale", "110", "400", 1000)
    book.reopen_period("2024-03")
    book.post_to_period("2024-03-15", "sale", "110", "400", 1000)
    assert journal.balance("110") == 6000


def test_period_book_unknown_dates(journal):
    book = PeriodBook(journal)
    book.open_period("2024-03", "2024-03-01", "2024-03-31")
    with pytest.raises(ValidationError):
        book.assert_postable("2024-04-01")
    with pytest.raises(ValidationError):
        book.open_period("2024-03", "2024-04-01", "2024-04-30")


def test_importers(chart, journal):
    text = (
        "2024-03-01 | Office supplies\n"
        "32.00 debit 500\n"
        "32.00 credit 100\n"
        "\n"
        "2024-03-02 | Sale\n"
        "150.00 debit 110\n"
        "150.00 credit 400\n"
    )
    entries = parse_entries(text, chart)
    assert len(entries) == 2
    assert entries[0].memo == "Office supplies"
    assert entries[0].balanced
    for entry in entries:
        journal.post(entry)
    assert journal.balance("100") == -3200

    rows = "# comment\n2024-03-03,fee,500,100,750\n"
    parsed = parse_csv_lines(rows)
    assert len(parsed) == 1
    assert parsed[0].total_debits == 750
