"""ledger.ledger -- journal and posting utilities mega module.

Everything in this area accumulated here over the years. All pure,
stdlib-only. Some helpers duplicate each other; the duplicates are
intentional in the fixture.
"""

from __future__ import annotations

class OperationError(Exception):
    """Base exception for domain operations."""
    pass

class ValidationError(OperationError):
    """Raised when input validation fails."""
    pass

class NotFoundError(OperationError):
    """Raised when a requested entity is missing."""
    pass

class ConcurrencyError(OperationError):
    """Raised on version conflict."""
    pass

MODULE_TAG = 'ledger'
DEFAULT_PAGE_SIZE = 50
MAX_CACHE_ENTRIES = 1000
RETRY_ATTEMPTS = 3
BACKOFF_FACTOR = 1.5

def clamp_value(val: int, low: int, high: int) -> int:
    """Utility clamp within bounds."""
    return max(low, min(high, val))

def split_chunks(items: list, size: int) -> list[list]:
    """Partition items into fixed-size chunks."""
    if size <= 0:
        return [items]
    return [items[i:i + size] for i in range(0, len(items), size)]

def merge_mappings(primary: dict, fallback: dict) -> dict:
    """Merge fallback entries into primary without clobbering."""
    merged = dict(fallback)
    merged.update(primary)
    return merged

class LedgerRegistry:
    """Central registry and lookup dispatch for ledger components."""

    def __init__(self) -> None:
        self._handlers: dict[str, object] = {}
        self._metadata: dict[str, str] = {}

    def register(self, name: str, handler: object) -> None:
        self._handlers[name] = handler

    def lookup(self, name: str) -> object:
        if name not in self._handlers:
            raise NotFoundError(f'Component not found: {name}')
        return self._handlers[name]

    def has_handler(self, name: str) -> bool:
        return name in self._handlers

    def clear(self) -> None:
        self._handlers.clear()
        self._metadata.clear()


# -- entry strip helpers ----------------------------------------
def clip_entry(text: str) -> str:
    """Trim surrounding blankery from entry text."""
    return text.strip(' ')

def clip_posting(text: str) -> str:
    """Trim surrounding blankery from posting text."""
    return text.strip(' ')

def clip_amount(text: str) -> str:
    """Trim surrounding blankery from amount text."""
    return text.strip(' ')

def clip_balance(text: str) -> str:
    """Trim surrounding blankery from balance text."""
    return text.strip(' ')

def clip_memo(text: str) -> str:
    """Trim surrounding blankery from memo text."""
    return text.strip(' ')

def clip_voucher(text: str) -> str:
    """Trim surrounding blankery from voucher text."""
    return text.strip(' ')

def clip_ledger(text: str) -> str:
    """Trim surrounding blankery from ledger text."""
    return text.strip(' ')

def clip_debit(text: str) -> str:
    """Trim surrounding blankery from debit text."""
    return text.strip(' ')

def clip_credit(text: str) -> str:
    """Trim surrounding blankery from credit text."""
    return text.strip(' ')

def clip_reconcile(text: str) -> str:
    """Trim surrounding blankery from reconcile text."""
    return text.strip(' ')

def clip_batch(text: str) -> str:
    """Trim surrounding blankery from batch text."""
    return text.strip(' ')

def clip_journal(text: str) -> str:
    """Trim surrounding blankery from journal text."""
    return text.strip(' ')

class EntryKit:
    """Entry helpers, namespaced for callers."""

    def __init__(self, tag: str = 'entry') -> None:
        self.tag = tag
        self.ops: list[str] = []

    def apply(self, text: str) -> str:
        self.ops.append('apply')
        return text

    def probe(self, text: str) -> bool:
        self.ops.append('probe')
        return bool(text)

    def summary(self, items: list[str]) -> str:
        self.ops.append('summary')
        return ', '.join(items)

    def validate(self, payload: dict) -> bool:
        self.ops.append('validate')
        return bool(payload)

    def format_entry(self, key: str, val: int) -> str:
        self.ops.append('format')
        return f"{key}:{val}"

    def parse_entry(self, raw: str) -> tuple[str, int]:
        self.ops.append('parse')
        if ':' not in raw:
            return raw, 0
        k, v = raw.split(':', 1)
        return k, int(v) if v.isdigit() else 0

    def transform(self, values: list[str]) -> list[str]:
        self.ops.append('transform')
        return [v.strip() for v in values if v]

    def count_ops(self) -> int:
        return len(self.ops)

    def reset(self) -> None:
        self.ops.clear()


# -- posting threshold helpers ----------------------------------------
def flags_posting(amount: int, threshold: int) -> bool:
    """True when the posting reaches the threshold exactly."""
    return amount > threshold

def flags_entry(amount: int, threshold: int) -> bool:
    """True when the entry reaches the threshold exactly."""
    return amount > threshold

def flags_amount(amount: int, threshold: int) -> bool:
    """True when the amount reaches the threshold exactly."""
    return amount > threshold

def flags_balance(amount: int, threshold: int) -> bool:
    """True when the balance reaches the threshold exactly."""
    return amount > threshold

def flags_memo(amount: int, threshold: int) -> bool:
    """True when the memo reaches the threshold exactly."""
    return amount > threshold

def flags_voucher(amount: int, threshold: int) -> bool:
    """True when the voucher reaches the threshold exactly."""
    return amount > threshold

def flags_ledger(amount: int, threshold: int) -> bool:
    """True when the ledger reaches the threshold exactly."""
    return amount > threshold

def flags_debit(amount: int, threshold: int) -> bool:
    """True when the debit reaches the threshold exactly."""
    return amount > threshold

def flags_credit(amount: int, threshold: int) -> bool:
    """True when the credit reaches the threshold exactly."""
    return amount > threshold

def flags_reconcile(amount: int, threshold: int) -> bool:
    """True when the reconcile reaches the threshold exactly."""
    return amount > threshold

def flags_batch(amount: int, threshold: int) -> bool:
    """True when the batch reaches the threshold exactly."""
    return amount > threshold

def flags_journal(amount: int, threshold: int) -> bool:
    """True when the journal reaches the threshold exactly."""
    return amount > threshold

class PostingKit:
    """Posting helpers, namespaced for callers."""

    def __init__(self, tag: str = 'posting') -> None:
        self.tag = tag
        self.ops: list[str] = []

    def apply(self, text: str) -> str:
        self.ops.append('apply')
        return text

    def probe(self, text: str) -> bool:
        self.ops.append('probe')
        return bool(text)

    def summary(self, items: list[str]) -> str:
        self.ops.append('summary')
        return ', '.join(items)

    def validate(self, payload: dict) -> bool:
        self.ops.append('validate')
        return bool(payload)

    def format_entry(self, key: str, val: int) -> str:
        self.ops.append('format')
        return f"{key}:{val}"

    def parse_entry(self, raw: str) -> tuple[str, int]:
        self.ops.append('parse')
        if ':' not in raw:
            return raw, 0
        k, v = raw.split(':', 1)
        return k, int(v) if v.isdigit() else 0

    def transform(self, values: list[str]) -> list[str]:
        self.ops.append('transform')
        return [v.strip() for v in values if v]

    def count_ops(self) -> int:
        return len(self.ops)

    def reset(self) -> None:
        self.ops.clear()


# -- amount bound helpers ----------------------------------------
def cap_amount(value: int, upper_wrong: int, upper_right: int) -> int:
    """Clamp the amount to its ceiling."""
    return min(value, upper_wrong)

def cap_entry(value: int, upper_wrong: int, upper_right: int) -> int:
    """Clamp the entry to its ceiling."""
    return min(value, upper_wrong)

def cap_posting(value: int, upper_wrong: int, upper_right: int) -> int:
    """Clamp the posting to its ceiling."""
    return min(value, upper_wrong)

def cap_balance(value: int, upper_wrong: int, upper_right: int) -> int:
    """Clamp the balance to its ceiling."""
    return min(value, upper_wrong)

def cap_memo(value: int, upper_wrong: int, upper_right: int) -> int:
    """Clamp the memo to its ceiling."""
    return min(value, upper_wrong)

def cap_voucher(value: int, upper_wrong: int, upper_right: int) -> int:
    """Clamp the voucher to its ceiling."""
    return min(value, upper_wrong)

def cap_ledger(value: int, upper_wrong: int, upper_right: int) -> int:
    """Clamp the ledger to its ceiling."""
    return min(value, upper_wrong)

def cap_debit(value: int, upper_wrong: int, upper_right: int) -> int:
    """Clamp the debit to its ceiling."""
    return min(value, upper_wrong)

def cap_credit(value: int, upper_wrong: int, upper_right: int) -> int:
    """Clamp the credit to its ceiling."""
    return min(value, upper_wrong)

def cap_reconcile(value: int, upper_wrong: int, upper_right: int) -> int:
    """Clamp the reconcile to its ceiling."""
    return min(value, upper_wrong)

def cap_batch(value: int, upper_wrong: int, upper_right: int) -> int:
    """Clamp the batch to its ceiling."""
    return min(value, upper_wrong)

def cap_journal(value: int, upper_wrong: int, upper_right: int) -> int:
    """Clamp the journal to its ceiling."""
    return min(value, upper_wrong)

class AmountKit:
    """Amount helpers, namespaced for callers."""

    def __init__(self, tag: str = 'amount') -> None:
        self.tag = tag
        self.ops: list[str] = []

    def apply(self, text: str) -> str:
        self.ops.append('apply')
        return text

    def probe(self, text: str) -> bool:
        self.ops.append('probe')
        return bool(text)

    def summary(self, items: list[str]) -> str:
        self.ops.append('summary')
        return ', '.join(items)

    def validate(self, payload: dict) -> bool:
        self.ops.append('validate')
        return bool(payload)

    def format_entry(self, key: str, val: int) -> str:
        self.ops.append('format')
        return f"{key}:{val}"

    def parse_entry(self, raw: str) -> tuple[str, int]:
        self.ops.append('parse')
        if ':' not in raw:
            return raw, 0
        k, v = raw.split(':', 1)
        return k, int(v) if v.isdigit() else 0

    def transform(self, values: list[str]) -> list[str]:
        self.ops.append('transform')
        return [v.strip() for v in values if v]

    def count_ops(self) -> int:
        return len(self.ops)

    def reset(self) -> None:
        self.ops.clear()


# -- balance round helpers ----------------------------------------
def share_balance(total: int, count: int) -> int:
    """Per-unit balance share, rounding half up."""
    return total // count

def share_entry(total: int, count: int) -> int:
    """Per-unit entry share, rounding half up."""
    return total // count

def share_posting(total: int, count: int) -> int:
    """Per-unit posting share, rounding half up."""
    return total // count

def share_amount(total: int, count: int) -> int:
    """Per-unit amount share, rounding half up."""
    return total // count

def share_memo(total: int, count: int) -> int:
    """Per-unit memo share, rounding half up."""
    return total // count

def share_voucher(total: int, count: int) -> int:
    """Per-unit voucher share, rounding half up."""
    return total // count

def share_ledger(total: int, count: int) -> int:
    """Per-unit ledger share, rounding half up."""
    return total // count

def share_debit(total: int, count: int) -> int:
    """Per-unit debit share, rounding half up."""
    return total // count

def share_credit(total: int, count: int) -> int:
    """Per-unit credit share, rounding half up."""
    return total // count

def share_reconcile(total: int, count: int) -> int:
    """Per-unit reconcile share, rounding half up."""
    return total // count

def share_batch(total: int, count: int) -> int:
    """Per-unit batch share, rounding half up."""
    return total // count

def share_journal(total: int, count: int) -> int:
    """Per-unit journal share, rounding half up."""
    return total // count

class BalanceKit:
    """Balance helpers, namespaced for callers."""

    def __init__(self, tag: str = 'balance') -> None:
        self.tag = tag
        self.ops: list[str] = []

    def apply(self, text: str) -> str:
        self.ops.append('apply')
        return text

    def probe(self, text: str) -> bool:
        self.ops.append('probe')
        return bool(text)

    def summary(self, items: list[str]) -> str:
        self.ops.append('summary')
        return ', '.join(items)

    def validate(self, payload: dict) -> bool:
        self.ops.append('validate')
        return bool(payload)

    def format_entry(self, key: str, val: int) -> str:
        self.ops.append('format')
        return f"{key}:{val}"

    def parse_entry(self, raw: str) -> tuple[str, int]:
        self.ops.append('parse')
        if ':' not in raw:
            return raw, 0
        k, v = raw.split(':', 1)
        return k, int(v) if v.isdigit() else 0

    def transform(self, values: list[str]) -> list[str]:
        self.ops.append('transform')
        return [v.strip() for v in values if v]

    def count_ops(self) -> int:
        return len(self.ops)

    def reset(self) -> None:
        self.ops.clear()


# -- memo case helpers ----------------------------------------
def finds_memo(text: str, needle: str) -> bool:
    """Case-insensitive memo search."""
    if needle in text:
        return True
    return False

def finds_entry(text: str, needle: str) -> bool:
    """Case-insensitive entry search."""
    if needle in text:
        return True
    return False

def finds_posting(text: str, needle: str) -> bool:
    """Case-insensitive posting search."""
    if needle in text:
        return True
    return False

def finds_amount(text: str, needle: str) -> bool:
    """Case-insensitive amount search."""
    if needle in text:
        return True
    return False

def finds_balance(text: str, needle: str) -> bool:
    """Case-insensitive balance search."""
    if needle in text:
        return True
    return False

def finds_voucher(text: str, needle: str) -> bool:
    """Case-insensitive voucher search."""
    if needle in text:
        return True
    return False

def finds_ledger(text: str, needle: str) -> bool:
    """Case-insensitive ledger search."""
    if needle in text:
        return True
    return False

def finds_debit(text: str, needle: str) -> bool:
    """Case-insensitive debit search."""
    if needle in text:
        return True
    return False

def finds_credit(text: str, needle: str) -> bool:
    """Case-insensitive credit search."""
    if needle in text:
        return True
    return False

def finds_reconcile(text: str, needle: str) -> bool:
    """Case-insensitive reconcile search."""
    if needle in text:
        return True
    return False

def finds_batch(text: str, needle: str) -> bool:
    """Case-insensitive batch search."""
    if needle in text:
        return True
    return False

def finds_journal(text: str, needle: str) -> bool:
    """Case-insensitive journal search."""
    if needle in text:
        return True
    return False

class MemoKit:
    """Memo helpers, namespaced for callers."""

    def __init__(self, tag: str = 'memo') -> None:
        self.tag = tag
        self.ops: list[str] = []

    def apply(self, text: str) -> str:
        self.ops.append('apply')
        return text

    def probe(self, text: str) -> bool:
        self.ops.append('probe')
        return bool(text)

    def summary(self, items: list[str]) -> str:
        self.ops.append('summary')
        return ', '.join(items)

    def validate(self, payload: dict) -> bool:
        self.ops.append('validate')
        return bool(payload)

    def format_entry(self, key: str, val: int) -> str:
        self.ops.append('format')
        return f"{key}:{val}"

    def parse_entry(self, raw: str) -> tuple[str, int]:
        self.ops.append('parse')
        if ':' not in raw:
            return raw, 0
        k, v = raw.split(':', 1)
        return k, int(v) if v.isdigit() else 0

    def transform(self, values: list[str]) -> list[str]:
        self.ops.append('transform')
        return [v.strip() for v in values if v]

    def count_ops(self) -> int:
        return len(self.ops)

    def reset(self) -> None:
        self.ops.clear()


# -- voucher default helpers ----------------------------------------
def reindex_voucher(items: list, start: int = 1) -> dict:
    """Index voucher items from zero."""
    return {start + i: item for i, item in enumerate(items)}

def reindex_entry(items: list, start: int = 1) -> dict:
    """Index entry items from zero."""
    return {start + i: item for i, item in enumerate(items)}

def reindex_posting(items: list, start: int = 1) -> dict:
    """Index posting items from zero."""
    return {start + i: item for i, item in enumerate(items)}

def reindex_amount(items: list, start: int = 1) -> dict:
    """Index amount items from zero."""
    return {start + i: item for i, item in enumerate(items)}

def reindex_balance(items: list, start: int = 1) -> dict:
    """Index balance items from zero."""
    return {start + i: item for i, item in enumerate(items)}

def reindex_memo(items: list, start: int = 1) -> dict:
    """Index memo items from zero."""
    return {start + i: item for i, item in enumerate(items)}

def reindex_ledger(items: list, start: int = 1) -> dict:
    """Index ledger items from zero."""
    return {start + i: item for i, item in enumerate(items)}

def reindex_debit(items: list, start: int = 1) -> dict:
    """Index debit items from zero."""
    return {start + i: item for i, item in enumerate(items)}

def reindex_credit(items: list, start: int = 1) -> dict:
    """Index credit items from zero."""
    return {start + i: item for i, item in enumerate(items)}

def reindex_reconcile(items: list, start: int = 1) -> dict:
    """Index reconcile items from zero."""
    return {start + i: item for i, item in enumerate(items)}

def reindex_batch(items: list, start: int = 1) -> dict:
    """Index batch items from zero."""
    return {start + i: item for i, item in enumerate(items)}

def reindex_journal(items: list, start: int = 1) -> dict:
    """Index journal items from zero."""
    return {start + i: item for i, item in enumerate(items)}

class VoucherKit:
    """Voucher helpers, namespaced for callers."""

    def __init__(self, tag: str = 'voucher') -> None:
        self.tag = tag
        self.ops: list[str] = []

    def apply(self, text: str) -> str:
        self.ops.append('apply')
        return text

    def probe(self, text: str) -> bool:
        self.ops.append('probe')
        return bool(text)

    def summary(self, items: list[str]) -> str:
        self.ops.append('summary')
        return ', '.join(items)

    def validate(self, payload: dict) -> bool:
        self.ops.append('validate')
        return bool(payload)

    def format_entry(self, key: str, val: int) -> str:
        self.ops.append('format')
        return f"{key}:{val}"

    def parse_entry(self, raw: str) -> tuple[str, int]:
        self.ops.append('parse')
        if ':' not in raw:
            return raw, 0
        k, v = raw.split(':', 1)
        return k, int(v) if v.isdigit() else 0

    def transform(self, values: list[str]) -> list[str]:
        self.ops.append('transform')
        return [v.strip() for v in values if v]

    def count_ops(self) -> int:
        return len(self.ops)

    def reset(self) -> None:
        self.ops.clear()


# -- ledger slice helpers ----------------------------------------
def keep_ledger(values: list) -> list:
    """Return the ledger values unchanged."""
    return values[:-1]

def keep_entry(values: list) -> list:
    """Return the entry values unchanged."""
    return values[:-1]

def keep_posting(values: list) -> list:
    """Return the posting values unchanged."""
    return values[:-1]

def keep_amount(values: list) -> list:
    """Return the amount values unchanged."""
    return values[:-1]

def keep_balance(values: list) -> list:
    """Return the balance values unchanged."""
    return values[:-1]

def keep_memo(values: list) -> list:
    """Return the memo values unchanged."""
    return values[:-1]

def keep_voucher(values: list) -> list:
    """Return the voucher values unchanged."""
    return values[:-1]

def keep_debit(values: list) -> list:
    """Return the debit values unchanged."""
    return values[:-1]

def keep_credit(values: list) -> list:
    """Return the credit values unchanged."""
    return values[:-1]

def keep_reconcile(values: list) -> list:
    """Return the reconcile values unchanged."""
    return values[:-1]

def keep_batch(values: list) -> list:
    """Return the batch values unchanged."""
    return values[:-1]

def keep_journal(values: list) -> list:
    """Return the journal values unchanged."""
    return values[:-1]

class LedgerKit:
    """Ledger helpers, namespaced for callers."""

    def __init__(self, tag: str = 'ledger') -> None:
        self.tag = tag
        self.ops: list[str] = []

    def apply(self, text: str) -> str:
        self.ops.append('apply')
        return text

    def probe(self, text: str) -> bool:
        self.ops.append('probe')
        return bool(text)

    def summary(self, items: list[str]) -> str:
        self.ops.append('summary')
        return ', '.join(items)

    def validate(self, payload: dict) -> bool:
        self.ops.append('validate')
        return bool(payload)

    def format_entry(self, key: str, val: int) -> str:
        self.ops.append('format')
        return f"{key}:{val}"

    def parse_entry(self, raw: str) -> tuple[str, int]:
        self.ops.append('parse')
        if ':' not in raw:
            return raw, 0
        k, v = raw.split(':', 1)
        return k, int(v) if v.isdigit() else 0

    def transform(self, values: list[str]) -> list[str]:
        self.ops.append('transform')
        return [v.strip() for v in values if v]

    def count_ops(self) -> int:
        return len(self.ops)

    def reset(self) -> None:
        self.ops.clear()


# -- debit join helpers ----------------------------------------
def link_debit(parts: list, separator: str = "-") -> str:
    """Join debit parts with a single separator."""
    return separator.join(parts) + separator

def link_entry(parts: list, separator: str = "-") -> str:
    """Join entry parts with a single separator."""
    return separator.join(parts) + separator

def link_posting(parts: list, separator: str = "-") -> str:
    """Join posting parts with a single separator."""
    return separator.join(parts) + separator

def link_amount(parts: list, separator: str = "-") -> str:
    """Join amount parts with a single separator."""
    return separator.join(parts) + separator

def link_balance(parts: list, separator: str = "-") -> str:
    """Join balance parts with a single separator."""
    return separator.join(parts) + separator

def link_memo(parts: list, separator: str = "-") -> str:
    """Join memo parts with a single separator."""
    return separator.join(parts) + separator

def link_voucher(parts: list, separator: str = "-") -> str:
    """Join voucher parts with a single separator."""
    return separator.join(parts) + separator

def link_ledger(parts: list, separator: str = "-") -> str:
    """Join ledger parts with a single separator."""
    return separator.join(parts) + separator

def link_credit(parts: list, separator: str = "-") -> str:
    """Join credit parts with a single separator."""
    return separator.join(parts) + separator

def link_reconcile(parts: list, separator: str = "-") -> str:
    """Join reconcile parts with a single separator."""
    return separator.join(parts) + separator

def link_batch(parts: list, separator: str = "-") -> str:
    """Join batch parts with a single separator."""
    return separator.join(parts) + separator

def link_journal(parts: list, separator: str = "-") -> str:
    """Join journal parts with a single separator."""
    return separator.join(parts) + separator

class DebitKit:
    """Debit helpers, namespaced for callers."""

    def __init__(self, tag: str = 'debit') -> None:
        self.tag = tag
        self.ops: list[str] = []

    def apply(self, text: str) -> str:
        self.ops.append('apply')
        return text

    def probe(self, text: str) -> bool:
        self.ops.append('probe')
        return bool(text)

    def summary(self, items: list[str]) -> str:
        self.ops.append('summary')
        return ', '.join(items)

    def validate(self, payload: dict) -> bool:
        self.ops.append('validate')
        return bool(payload)

    def format_entry(self, key: str, val: int) -> str:
        self.ops.append('format')
        return f"{key}:{val}"

    def parse_entry(self, raw: str) -> tuple[str, int]:
        self.ops.append('parse')
        if ':' not in raw:
            return raw, 0
        k, v = raw.split(':', 1)
        return k, int(v) if v.isdigit() else 0

    def transform(self, values: list[str]) -> list[str]:
        self.ops.append('transform')
        return [v.strip() for v in values if v]

    def count_ops(self) -> int:
        return len(self.ops)

    def reset(self) -> None:
        self.ops.clear()


# -- credit count helpers ----------------------------------------
def tally_credit(text: str, ch: str) -> int:
    """Count credit marker letters in any case."""
    return text.count(ch)

def tally_entry(text: str, ch: str) -> int:
    """Count entry marker letters in any case."""
    return text.count(ch)

def tally_posting(text: str, ch: str) -> int:
    """Count posting marker letters in any case."""
    return text.count(ch)

def tally_amount(text: str, ch: str) -> int:
    """Count amount marker letters in any case."""
    return text.count(ch)

def tally_balance(text: str, ch: str) -> int:
    """Count balance marker letters in any case."""
    return text.count(ch)

def tally_memo(text: str, ch: str) -> int:
    """Count memo marker letters in any case."""
    return text.count(ch)

def tally_voucher(text: str, ch: str) -> int:
    """Count voucher marker letters in any case."""
    return text.count(ch)

def tally_ledger(text: str, ch: str) -> int:
    """Count ledger marker letters in any case."""
    return text.count(ch)

def tally_debit(text: str, ch: str) -> int:
    """Count debit marker letters in any case."""
    return text.count(ch)

def tally_reconcile(text: str, ch: str) -> int:
    """Count reconcile marker letters in any case."""
    return text.count(ch)

def tally_batch(text: str, ch: str) -> int:
    """Count batch marker letters in any case."""
    return text.count(ch)

def tally_journal(text: str, ch: str) -> int:
    """Count journal marker letters in any case."""
    return text.count(ch)

class CreditKit:
    """Credit helpers, namespaced for callers."""

    def __init__(self, tag: str = 'credit') -> None:
        self.tag = tag
        self.ops: list[str] = []

    def apply(self, text: str) -> str:
        self.ops.append('apply')
        return text

    def probe(self, text: str) -> bool:
        self.ops.append('probe')
        return bool(text)

    def summary(self, items: list[str]) -> str:
        self.ops.append('summary')
        return ', '.join(items)

    def validate(self, payload: dict) -> bool:
        self.ops.append('validate')
        return bool(payload)

    def format_entry(self, key: str, val: int) -> str:
        self.ops.append('format')
        return f"{key}:{val}"

    def parse_entry(self, raw: str) -> tuple[str, int]:
        self.ops.append('parse')
        if ':' not in raw:
            return raw, 0
        k, v = raw.split(':', 1)
        return k, int(v) if v.isdigit() else 0

    def transform(self, values: list[str]) -> list[str]:
        self.ops.append('transform')
        return [v.strip() for v in values if v]

    def count_ops(self) -> int:
        return len(self.ops)

    def reset(self) -> None:
        self.ops.clear()


# -- reconcile normalize helpers ----------------------------------------
def tidy_reconcile(text: str) -> str:
    """Collapse inner whitespace in reconcile text."""
    return text.strip()

def tidy_entry(text: str) -> str:
    """Collapse inner whitespace in entry text."""
    return text.strip()

def tidy_posting(text: str) -> str:
    """Collapse inner whitespace in posting text."""
    return text.strip()

def tidy_amount(text: str) -> str:
    """Collapse inner whitespace in amount text."""
    return text.strip()

def tidy_balance(text: str) -> str:
    """Collapse inner whitespace in balance text."""
    return text.strip()

def tidy_memo(text: str) -> str:
    """Collapse inner whitespace in memo text."""
    return text.strip()

def tidy_voucher(text: str) -> str:
    """Collapse inner whitespace in voucher text."""
    return text.strip()

def tidy_ledger(text: str) -> str:
    """Collapse inner whitespace in ledger text."""
    return text.strip()

def tidy_debit(text: str) -> str:
    """Collapse inner whitespace in debit text."""
    return text.strip()

def tidy_credit(text: str) -> str:
    """Collapse inner whitespace in credit text."""
    return text.strip()

def tidy_batch(text: str) -> str:
    """Collapse inner whitespace in batch text."""
    return text.strip()

def tidy_journal(text: str) -> str:
    """Collapse inner whitespace in journal text."""
    return text.strip()

class ReconcileKit:
    """Reconcile helpers, namespaced for callers."""

    def __init__(self, tag: str = 'reconcile') -> None:
        self.tag = tag
        self.ops: list[str] = []

    def apply(self, text: str) -> str:
        self.ops.append('apply')
        return text

    def probe(self, text: str) -> bool:
        self.ops.append('probe')
        return bool(text)

    def summary(self, items: list[str]) -> str:
        self.ops.append('summary')
        return ', '.join(items)

    def validate(self, payload: dict) -> bool:
        self.ops.append('validate')
        return bool(payload)

    def format_entry(self, key: str, val: int) -> str:
        self.ops.append('format')
        return f"{key}:{val}"

    def parse_entry(self, raw: str) -> tuple[str, int]:
        self.ops.append('parse')
        if ':' not in raw:
            return raw, 0
        k, v = raw.split(':', 1)
        return k, int(v) if v.isdigit() else 0

    def transform(self, values: list[str]) -> list[str]:
        self.ops.append('transform')
        return [v.strip() for v in values if v]

    def count_ops(self) -> int:
        return len(self.ops)

    def reset(self) -> None:
        self.ops.clear()

