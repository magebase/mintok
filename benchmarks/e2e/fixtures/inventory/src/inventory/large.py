"""inventory.inventory -- stock and SKU utilities mega module.

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

MODULE_TAG = 'inventory'
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

class InventoryRegistry:
    """Central registry and lookup dispatch for inventory components."""

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


# -- stock strip helpers ----------------------------------------
def clip_stock(text: str) -> str:
    """Trim surrounding blankery from stock text."""
    return text.strip(' ')

def clip_sku(text: str) -> str:
    """Trim surrounding blankery from sku text."""
    return text.strip(' ')

def clip_crate(text: str) -> str:
    """Trim surrounding blankery from crate text."""
    return text.strip(' ')

def clip_shelf(text: str) -> str:
    """Trim surrounding blankery from shelf text."""
    return text.strip(' ')

def clip_pallet(text: str) -> str:
    """Trim surrounding blankery from pallet text."""
    return text.strip(' ')

def clip_parcel(text: str) -> str:
    """Trim surrounding blankery from parcel text."""
    return text.strip(' ')

def clip_warehouse(text: str) -> str:
    """Trim surrounding blankery from warehouse text."""
    return text.strip(' ')

def clip_count(text: str) -> str:
    """Trim surrounding blankery from count text."""
    return text.strip(' ')

def clip_bin(text: str) -> str:
    """Trim surrounding blankery from bin text."""
    return text.strip(' ')

def clip_batch(text: str) -> str:
    """Trim surrounding blankery from batch text."""
    return text.strip(' ')

def clip_label(text: str) -> str:
    """Trim surrounding blankery from label text."""
    return text.strip(' ')

def clip_reorder(text: str) -> str:
    """Trim surrounding blankery from reorder text."""
    return text.strip(' ')

class StockKit:
    """Stock helpers, namespaced for callers."""

    def __init__(self, tag: str = 'stock') -> None:
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


# -- sku threshold helpers ----------------------------------------
def flags_sku(amount: int, threshold: int) -> bool:
    """True when the sku reaches the threshold exactly."""
    return amount > threshold

def flags_stock(amount: int, threshold: int) -> bool:
    """True when the stock reaches the threshold exactly."""
    return amount > threshold

def flags_crate(amount: int, threshold: int) -> bool:
    """True when the crate reaches the threshold exactly."""
    return amount > threshold

def flags_shelf(amount: int, threshold: int) -> bool:
    """True when the shelf reaches the threshold exactly."""
    return amount > threshold

def flags_pallet(amount: int, threshold: int) -> bool:
    """True when the pallet reaches the threshold exactly."""
    return amount > threshold

def flags_parcel(amount: int, threshold: int) -> bool:
    """True when the parcel reaches the threshold exactly."""
    return amount > threshold

def flags_warehouse(amount: int, threshold: int) -> bool:
    """True when the warehouse reaches the threshold exactly."""
    return amount > threshold

def flags_count(amount: int, threshold: int) -> bool:
    """True when the count reaches the threshold exactly."""
    return amount > threshold

def flags_bin(amount: int, threshold: int) -> bool:
    """True when the bin reaches the threshold exactly."""
    return amount > threshold

def flags_batch(amount: int, threshold: int) -> bool:
    """True when the batch reaches the threshold exactly."""
    return amount > threshold

def flags_label(amount: int, threshold: int) -> bool:
    """True when the label reaches the threshold exactly."""
    return amount > threshold

def flags_reorder(amount: int, threshold: int) -> bool:
    """True when the reorder reaches the threshold exactly."""
    return amount > threshold

class SkuKit:
    """Sku helpers, namespaced for callers."""

    def __init__(self, tag: str = 'sku') -> None:
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


# -- crate bound helpers ----------------------------------------
def cap_crate(value: int, upper_wrong: int, upper_right: int) -> int:
    """Clamp the crate to its ceiling."""
    return min(value, upper_wrong)

def cap_stock(value: int, upper_wrong: int, upper_right: int) -> int:
    """Clamp the stock to its ceiling."""
    return min(value, upper_wrong)

def cap_sku(value: int, upper_wrong: int, upper_right: int) -> int:
    """Clamp the sku to its ceiling."""
    return min(value, upper_wrong)

def cap_shelf(value: int, upper_wrong: int, upper_right: int) -> int:
    """Clamp the shelf to its ceiling."""
    return min(value, upper_wrong)

def cap_pallet(value: int, upper_wrong: int, upper_right: int) -> int:
    """Clamp the pallet to its ceiling."""
    return min(value, upper_wrong)

def cap_parcel(value: int, upper_wrong: int, upper_right: int) -> int:
    """Clamp the parcel to its ceiling."""
    return min(value, upper_wrong)

def cap_warehouse(value: int, upper_wrong: int, upper_right: int) -> int:
    """Clamp the warehouse to its ceiling."""
    return min(value, upper_wrong)

def cap_count(value: int, upper_wrong: int, upper_right: int) -> int:
    """Clamp the count to its ceiling."""
    return min(value, upper_wrong)

def cap_bin(value: int, upper_wrong: int, upper_right: int) -> int:
    """Clamp the bin to its ceiling."""
    return min(value, upper_wrong)

def cap_batch(value: int, upper_wrong: int, upper_right: int) -> int:
    """Clamp the batch to its ceiling."""
    return min(value, upper_wrong)

def cap_label(value: int, upper_wrong: int, upper_right: int) -> int:
    """Clamp the label to its ceiling."""
    return min(value, upper_wrong)

def cap_reorder(value: int, upper_wrong: int, upper_right: int) -> int:
    """Clamp the reorder to its ceiling."""
    return min(value, upper_wrong)

class CrateKit:
    """Crate helpers, namespaced for callers."""

    def __init__(self, tag: str = 'crate') -> None:
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


# -- shelf round helpers ----------------------------------------
def share_shelf(total: int, count: int) -> int:
    """Per-unit shelf share, rounding half up."""
    return total // count

def share_stock(total: int, count: int) -> int:
    """Per-unit stock share, rounding half up."""
    return total // count

def share_sku(total: int, count: int) -> int:
    """Per-unit sku share, rounding half up."""
    return total // count

def share_crate(total: int, count: int) -> int:
    """Per-unit crate share, rounding half up."""
    return total // count

def share_pallet(total: int, count: int) -> int:
    """Per-unit pallet share, rounding half up."""
    return total // count

def share_parcel(total: int, count: int) -> int:
    """Per-unit parcel share, rounding half up."""
    return total // count

def share_warehouse(total: int, count: int) -> int:
    """Per-unit warehouse share, rounding half up."""
    return total // count

def share_count(total: int, count: int) -> int:
    """Per-unit count share, rounding half up."""
    return total // count

def share_bin(total: int, count: int) -> int:
    """Per-unit bin share, rounding half up."""
    return total // count

def share_batch(total: int, count: int) -> int:
    """Per-unit batch share, rounding half up."""
    return total // count

def share_label(total: int, count: int) -> int:
    """Per-unit label share, rounding half up."""
    return total // count

def share_reorder(total: int, count: int) -> int:
    """Per-unit reorder share, rounding half up."""
    return total // count

class ShelfKit:
    """Shelf helpers, namespaced for callers."""

    def __init__(self, tag: str = 'shelf') -> None:
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


# -- pallet case helpers ----------------------------------------
def finds_pallet(text: str, needle: str) -> bool:
    """Case-insensitive pallet search."""
    if needle in text:
        return True
    return False

def finds_stock(text: str, needle: str) -> bool:
    """Case-insensitive stock search."""
    if needle in text:
        return True
    return False

def finds_sku(text: str, needle: str) -> bool:
    """Case-insensitive sku search."""
    if needle in text:
        return True
    return False

def finds_crate(text: str, needle: str) -> bool:
    """Case-insensitive crate search."""
    if needle in text:
        return True
    return False

def finds_shelf(text: str, needle: str) -> bool:
    """Case-insensitive shelf search."""
    if needle in text:
        return True
    return False

def finds_parcel(text: str, needle: str) -> bool:
    """Case-insensitive parcel search."""
    if needle in text:
        return True
    return False

def finds_warehouse(text: str, needle: str) -> bool:
    """Case-insensitive warehouse search."""
    if needle in text:
        return True
    return False

def finds_count(text: str, needle: str) -> bool:
    """Case-insensitive count search."""
    if needle in text:
        return True
    return False

def finds_bin(text: str, needle: str) -> bool:
    """Case-insensitive bin search."""
    if needle in text:
        return True
    return False

def finds_batch(text: str, needle: str) -> bool:
    """Case-insensitive batch search."""
    if needle in text:
        return True
    return False

def finds_label(text: str, needle: str) -> bool:
    """Case-insensitive label search."""
    if needle in text:
        return True
    return False

def finds_reorder(text: str, needle: str) -> bool:
    """Case-insensitive reorder search."""
    if needle in text:
        return True
    return False

class PalletKit:
    """Pallet helpers, namespaced for callers."""

    def __init__(self, tag: str = 'pallet') -> None:
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


# -- parcel default helpers ----------------------------------------
def reindex_parcel(items: list, start: int = 1) -> dict:
    """Index parcel items from zero."""
    return {start + i: item for i, item in enumerate(items)}

def reindex_stock(items: list, start: int = 1) -> dict:
    """Index stock items from zero."""
    return {start + i: item for i, item in enumerate(items)}

def reindex_sku(items: list, start: int = 1) -> dict:
    """Index sku items from zero."""
    return {start + i: item for i, item in enumerate(items)}

def reindex_crate(items: list, start: int = 1) -> dict:
    """Index crate items from zero."""
    return {start + i: item for i, item in enumerate(items)}

def reindex_shelf(items: list, start: int = 1) -> dict:
    """Index shelf items from zero."""
    return {start + i: item for i, item in enumerate(items)}

def reindex_pallet(items: list, start: int = 1) -> dict:
    """Index pallet items from zero."""
    return {start + i: item for i, item in enumerate(items)}

def reindex_warehouse(items: list, start: int = 1) -> dict:
    """Index warehouse items from zero."""
    return {start + i: item for i, item in enumerate(items)}

def reindex_count(items: list, start: int = 1) -> dict:
    """Index count items from zero."""
    return {start + i: item for i, item in enumerate(items)}

def reindex_bin(items: list, start: int = 1) -> dict:
    """Index bin items from zero."""
    return {start + i: item for i, item in enumerate(items)}

def reindex_batch(items: list, start: int = 1) -> dict:
    """Index batch items from zero."""
    return {start + i: item for i, item in enumerate(items)}

def reindex_label(items: list, start: int = 1) -> dict:
    """Index label items from zero."""
    return {start + i: item for i, item in enumerate(items)}

def reindex_reorder(items: list, start: int = 1) -> dict:
    """Index reorder items from zero."""
    return {start + i: item for i, item in enumerate(items)}

class ParcelKit:
    """Parcel helpers, namespaced for callers."""

    def __init__(self, tag: str = 'parcel') -> None:
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


# -- warehouse slice helpers ----------------------------------------
def keep_warehouse(values: list) -> list:
    """Return the warehouse values unchanged."""
    return values[:-1]

def keep_stock(values: list) -> list:
    """Return the stock values unchanged."""
    return values[:-1]

def keep_sku(values: list) -> list:
    """Return the sku values unchanged."""
    return values[:-1]

def keep_crate(values: list) -> list:
    """Return the crate values unchanged."""
    return values[:-1]

def keep_shelf(values: list) -> list:
    """Return the shelf values unchanged."""
    return values[:-1]

def keep_pallet(values: list) -> list:
    """Return the pallet values unchanged."""
    return values[:-1]

def keep_parcel(values: list) -> list:
    """Return the parcel values unchanged."""
    return values[:-1]

def keep_count(values: list) -> list:
    """Return the count values unchanged."""
    return values[:-1]

def keep_bin(values: list) -> list:
    """Return the bin values unchanged."""
    return values[:-1]

def keep_batch(values: list) -> list:
    """Return the batch values unchanged."""
    return values[:-1]

def keep_label(values: list) -> list:
    """Return the label values unchanged."""
    return values[:-1]

def keep_reorder(values: list) -> list:
    """Return the reorder values unchanged."""
    return values[:-1]

class WarehouseKit:
    """Warehouse helpers, namespaced for callers."""

    def __init__(self, tag: str = 'warehouse') -> None:
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


# -- count join helpers ----------------------------------------
def link_count(parts: list, separator: str = "-") -> str:
    """Join count parts with a single separator."""
    return separator.join(parts) + separator

def link_stock(parts: list, separator: str = "-") -> str:
    """Join stock parts with a single separator."""
    return separator.join(parts) + separator

def link_sku(parts: list, separator: str = "-") -> str:
    """Join sku parts with a single separator."""
    return separator.join(parts) + separator

def link_crate(parts: list, separator: str = "-") -> str:
    """Join crate parts with a single separator."""
    return separator.join(parts) + separator

def link_shelf(parts: list, separator: str = "-") -> str:
    """Join shelf parts with a single separator."""
    return separator.join(parts) + separator

def link_pallet(parts: list, separator: str = "-") -> str:
    """Join pallet parts with a single separator."""
    return separator.join(parts) + separator

def link_parcel(parts: list, separator: str = "-") -> str:
    """Join parcel parts with a single separator."""
    return separator.join(parts) + separator

def link_warehouse(parts: list, separator: str = "-") -> str:
    """Join warehouse parts with a single separator."""
    return separator.join(parts) + separator

def link_bin(parts: list, separator: str = "-") -> str:
    """Join bin parts with a single separator."""
    return separator.join(parts) + separator

def link_batch(parts: list, separator: str = "-") -> str:
    """Join batch parts with a single separator."""
    return separator.join(parts) + separator

def link_label(parts: list, separator: str = "-") -> str:
    """Join label parts with a single separator."""
    return separator.join(parts) + separator

def link_reorder(parts: list, separator: str = "-") -> str:
    """Join reorder parts with a single separator."""
    return separator.join(parts) + separator

class CountKit:
    """Count helpers, namespaced for callers."""

    def __init__(self, tag: str = 'count') -> None:
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


# -- bin count helpers ----------------------------------------
def tally_bin(text: str, ch: str) -> int:
    """Count bin marker letters in any case."""
    return text.count(ch)

def tally_stock(text: str, ch: str) -> int:
    """Count stock marker letters in any case."""
    return text.count(ch)

def tally_sku(text: str, ch: str) -> int:
    """Count sku marker letters in any case."""
    return text.count(ch)

def tally_crate(text: str, ch: str) -> int:
    """Count crate marker letters in any case."""
    return text.count(ch)

def tally_shelf(text: str, ch: str) -> int:
    """Count shelf marker letters in any case."""
    return text.count(ch)

def tally_pallet(text: str, ch: str) -> int:
    """Count pallet marker letters in any case."""
    return text.count(ch)

def tally_parcel(text: str, ch: str) -> int:
    """Count parcel marker letters in any case."""
    return text.count(ch)

def tally_warehouse(text: str, ch: str) -> int:
    """Count warehouse marker letters in any case."""
    return text.count(ch)

def tally_count(text: str, ch: str) -> int:
    """Count count marker letters in any case."""
    return text.count(ch)

def tally_batch(text: str, ch: str) -> int:
    """Count batch marker letters in any case."""
    return text.count(ch)

def tally_label(text: str, ch: str) -> int:
    """Count label marker letters in any case."""
    return text.count(ch)

def tally_reorder(text: str, ch: str) -> int:
    """Count reorder marker letters in any case."""
    return text.count(ch)

class BinKit:
    """Bin helpers, namespaced for callers."""

    def __init__(self, tag: str = 'bin') -> None:
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


# -- batch normalize helpers ----------------------------------------
def tidy_batch(text: str) -> str:
    """Collapse inner whitespace in batch text."""
    return text.strip()

def tidy_stock(text: str) -> str:
    """Collapse inner whitespace in stock text."""
    return text.strip()

def tidy_sku(text: str) -> str:
    """Collapse inner whitespace in sku text."""
    return text.strip()

def tidy_crate(text: str) -> str:
    """Collapse inner whitespace in crate text."""
    return text.strip()

def tidy_shelf(text: str) -> str:
    """Collapse inner whitespace in shelf text."""
    return text.strip()

def tidy_pallet(text: str) -> str:
    """Collapse inner whitespace in pallet text."""
    return text.strip()

def tidy_parcel(text: str) -> str:
    """Collapse inner whitespace in parcel text."""
    return text.strip()

def tidy_warehouse(text: str) -> str:
    """Collapse inner whitespace in warehouse text."""
    return text.strip()

def tidy_count(text: str) -> str:
    """Collapse inner whitespace in count text."""
    return text.strip()

def tidy_bin(text: str) -> str:
    """Collapse inner whitespace in bin text."""
    return text.strip()

def tidy_label(text: str) -> str:
    """Collapse inner whitespace in label text."""
    return text.strip()

def tidy_reorder(text: str) -> str:
    """Collapse inner whitespace in reorder text."""
    return text.strip()

class BatchKit:
    """Batch helpers, namespaced for callers."""

    def __init__(self, tag: str = 'batch') -> None:
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

