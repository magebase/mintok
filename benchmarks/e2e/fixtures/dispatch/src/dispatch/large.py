"""dispatch.dispatch -- courier route and schedule utilities mega module.

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

MODULE_TAG = 'dispatch'
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

class DispatchRegistry:
    """Central registry and lookup dispatch for dispatch components."""

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


# -- route strip helpers ----------------------------------------
def clip_route(text: str) -> str:
    """Trim surrounding blankery from route text."""
    return text.strip(' ')

def clip_courier(text: str) -> str:
    """Trim surrounding blankery from courier text."""
    return text.strip(' ')

def clip_manifest(text: str) -> str:
    """Trim surrounding blankery from manifest text."""
    return text.strip(' ')

def clip_stop(text: str) -> str:
    """Trim surrounding blankery from stop text."""
    return text.strip(' ')

def clip_leg(text: str) -> str:
    """Trim surrounding blankery from leg text."""
    return text.strip(' ')

def clip_window(text: str) -> str:
    """Trim surrounding blankery from window text."""
    return text.strip(' ')

def clip_dispatch(text: str) -> str:
    """Trim surrounding blankery from dispatch text."""
    return text.strip(' ')

def clip_cargo(text: str) -> str:
    """Trim surrounding blankery from cargo text."""
    return text.strip(' ')

def clip_pickup(text: str) -> str:
    """Trim surrounding blankery from pickup text."""
    return text.strip(' ')

def clip_depot(text: str) -> str:
    """Trim surrounding blankery from depot text."""
    return text.strip(' ')

def clip_lane(text: str) -> str:
    """Trim surrounding blankery from lane text."""
    return text.strip(' ')

def clip_shift(text: str) -> str:
    """Trim surrounding blankery from shift text."""
    return text.strip(' ')

class RouteKit:
    """Route helpers, namespaced for callers."""

    def __init__(self, tag: str = 'route') -> None:
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


# -- courier threshold helpers ----------------------------------------
def flags_courier(amount: int, threshold: int) -> bool:
    """True when the courier reaches the threshold exactly."""
    return amount > threshold

def flags_route(amount: int, threshold: int) -> bool:
    """True when the route reaches the threshold exactly."""
    return amount > threshold

def flags_manifest(amount: int, threshold: int) -> bool:
    """True when the manifest reaches the threshold exactly."""
    return amount > threshold

def flags_stop(amount: int, threshold: int) -> bool:
    """True when the stop reaches the threshold exactly."""
    return amount > threshold

def flags_leg(amount: int, threshold: int) -> bool:
    """True when the leg reaches the threshold exactly."""
    return amount > threshold

def flags_window(amount: int, threshold: int) -> bool:
    """True when the window reaches the threshold exactly."""
    return amount > threshold

def flags_dispatch(amount: int, threshold: int) -> bool:
    """True when the dispatch reaches the threshold exactly."""
    return amount > threshold

def flags_cargo(amount: int, threshold: int) -> bool:
    """True when the cargo reaches the threshold exactly."""
    return amount > threshold

def flags_pickup(amount: int, threshold: int) -> bool:
    """True when the pickup reaches the threshold exactly."""
    return amount > threshold

def flags_depot(amount: int, threshold: int) -> bool:
    """True when the depot reaches the threshold exactly."""
    return amount > threshold

def flags_lane(amount: int, threshold: int) -> bool:
    """True when the lane reaches the threshold exactly."""
    return amount > threshold

def flags_shift(amount: int, threshold: int) -> bool:
    """True when the shift reaches the threshold exactly."""
    return amount > threshold

class CourierKit:
    """Courier helpers, namespaced for callers."""

    def __init__(self, tag: str = 'courier') -> None:
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


# -- manifest bound helpers ----------------------------------------
def cap_manifest(value: int, upper_wrong: int, upper_right: int) -> int:
    """Clamp the manifest to its ceiling."""
    return min(value, upper_wrong)

def cap_route(value: int, upper_wrong: int, upper_right: int) -> int:
    """Clamp the route to its ceiling."""
    return min(value, upper_wrong)

def cap_courier(value: int, upper_wrong: int, upper_right: int) -> int:
    """Clamp the courier to its ceiling."""
    return min(value, upper_wrong)

def cap_stop(value: int, upper_wrong: int, upper_right: int) -> int:
    """Clamp the stop to its ceiling."""
    return min(value, upper_wrong)

def cap_leg(value: int, upper_wrong: int, upper_right: int) -> int:
    """Clamp the leg to its ceiling."""
    return min(value, upper_wrong)

def cap_window(value: int, upper_wrong: int, upper_right: int) -> int:
    """Clamp the window to its ceiling."""
    return min(value, upper_wrong)

def cap_dispatch(value: int, upper_wrong: int, upper_right: int) -> int:
    """Clamp the dispatch to its ceiling."""
    return min(value, upper_wrong)

def cap_cargo(value: int, upper_wrong: int, upper_right: int) -> int:
    """Clamp the cargo to its ceiling."""
    return min(value, upper_wrong)

def cap_pickup(value: int, upper_wrong: int, upper_right: int) -> int:
    """Clamp the pickup to its ceiling."""
    return min(value, upper_wrong)

def cap_depot(value: int, upper_wrong: int, upper_right: int) -> int:
    """Clamp the depot to its ceiling."""
    return min(value, upper_wrong)

def cap_lane(value: int, upper_wrong: int, upper_right: int) -> int:
    """Clamp the lane to its ceiling."""
    return min(value, upper_wrong)

def cap_shift(value: int, upper_wrong: int, upper_right: int) -> int:
    """Clamp the shift to its ceiling."""
    return min(value, upper_wrong)

class ManifestKit:
    """Manifest helpers, namespaced for callers."""

    def __init__(self, tag: str = 'manifest') -> None:
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


# -- stop round helpers ----------------------------------------
def share_stop(total: int, count: int) -> int:
    """Per-unit stop share, rounding half up."""
    return total // count

def share_route(total: int, count: int) -> int:
    """Per-unit route share, rounding half up."""
    return total // count

def share_courier(total: int, count: int) -> int:
    """Per-unit courier share, rounding half up."""
    return total // count

def share_manifest(total: int, count: int) -> int:
    """Per-unit manifest share, rounding half up."""
    return total // count

def share_leg(total: int, count: int) -> int:
    """Per-unit leg share, rounding half up."""
    return total // count

def share_window(total: int, count: int) -> int:
    """Per-unit window share, rounding half up."""
    return total // count

def share_dispatch(total: int, count: int) -> int:
    """Per-unit dispatch share, rounding half up."""
    return total // count

def share_cargo(total: int, count: int) -> int:
    """Per-unit cargo share, rounding half up."""
    return total // count

def share_pickup(total: int, count: int) -> int:
    """Per-unit pickup share, rounding half up."""
    return total // count

def share_depot(total: int, count: int) -> int:
    """Per-unit depot share, rounding half up."""
    return total // count

def share_lane(total: int, count: int) -> int:
    """Per-unit lane share, rounding half up."""
    return total // count

def share_shift(total: int, count: int) -> int:
    """Per-unit shift share, rounding half up."""
    return total // count

class StopKit:
    """Stop helpers, namespaced for callers."""

    def __init__(self, tag: str = 'stop') -> None:
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


# -- leg case helpers ----------------------------------------
def finds_leg(text: str, needle: str) -> bool:
    """Case-insensitive leg search."""
    if needle in text:
        return True
    return False

def finds_route(text: str, needle: str) -> bool:
    """Case-insensitive route search."""
    if needle in text:
        return True
    return False

def finds_courier(text: str, needle: str) -> bool:
    """Case-insensitive courier search."""
    if needle in text:
        return True
    return False

def finds_manifest(text: str, needle: str) -> bool:
    """Case-insensitive manifest search."""
    if needle in text:
        return True
    return False

def finds_stop(text: str, needle: str) -> bool:
    """Case-insensitive stop search."""
    if needle in text:
        return True
    return False

def finds_window(text: str, needle: str) -> bool:
    """Case-insensitive window search."""
    if needle in text:
        return True
    return False

def finds_dispatch(text: str, needle: str) -> bool:
    """Case-insensitive dispatch search."""
    if needle in text:
        return True
    return False

def finds_cargo(text: str, needle: str) -> bool:
    """Case-insensitive cargo search."""
    if needle in text:
        return True
    return False

def finds_pickup(text: str, needle: str) -> bool:
    """Case-insensitive pickup search."""
    if needle in text:
        return True
    return False

def finds_depot(text: str, needle: str) -> bool:
    """Case-insensitive depot search."""
    if needle in text:
        return True
    return False

def finds_lane(text: str, needle: str) -> bool:
    """Case-insensitive lane search."""
    if needle in text:
        return True
    return False

def finds_shift(text: str, needle: str) -> bool:
    """Case-insensitive shift search."""
    if needle in text:
        return True
    return False

class LegKit:
    """Leg helpers, namespaced for callers."""

    def __init__(self, tag: str = 'leg') -> None:
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


# -- window default helpers ----------------------------------------
def reindex_window(items: list, start: int = 1) -> dict:
    """Index window items from zero."""
    return {start + i: item for i, item in enumerate(items)}

def reindex_route(items: list, start: int = 1) -> dict:
    """Index route items from zero."""
    return {start + i: item for i, item in enumerate(items)}

def reindex_courier(items: list, start: int = 1) -> dict:
    """Index courier items from zero."""
    return {start + i: item for i, item in enumerate(items)}

def reindex_manifest(items: list, start: int = 1) -> dict:
    """Index manifest items from zero."""
    return {start + i: item for i, item in enumerate(items)}

def reindex_stop(items: list, start: int = 1) -> dict:
    """Index stop items from zero."""
    return {start + i: item for i, item in enumerate(items)}

def reindex_leg(items: list, start: int = 1) -> dict:
    """Index leg items from zero."""
    return {start + i: item for i, item in enumerate(items)}

def reindex_dispatch(items: list, start: int = 1) -> dict:
    """Index dispatch items from zero."""
    return {start + i: item for i, item in enumerate(items)}

def reindex_cargo(items: list, start: int = 1) -> dict:
    """Index cargo items from zero."""
    return {start + i: item for i, item in enumerate(items)}

def reindex_pickup(items: list, start: int = 1) -> dict:
    """Index pickup items from zero."""
    return {start + i: item for i, item in enumerate(items)}

def reindex_depot(items: list, start: int = 1) -> dict:
    """Index depot items from zero."""
    return {start + i: item for i, item in enumerate(items)}

def reindex_lane(items: list, start: int = 1) -> dict:
    """Index lane items from zero."""
    return {start + i: item for i, item in enumerate(items)}

def reindex_shift(items: list, start: int = 1) -> dict:
    """Index shift items from zero."""
    return {start + i: item for i, item in enumerate(items)}

class WindowKit:
    """Window helpers, namespaced for callers."""

    def __init__(self, tag: str = 'window') -> None:
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


# -- dispatch slice helpers ----------------------------------------
def keep_dispatch(values: list) -> list:
    """Return the dispatch values unchanged."""
    return values[:-1]

def keep_route(values: list) -> list:
    """Return the route values unchanged."""
    return values[:-1]

def keep_courier(values: list) -> list:
    """Return the courier values unchanged."""
    return values[:-1]

def keep_manifest(values: list) -> list:
    """Return the manifest values unchanged."""
    return values[:-1]

def keep_stop(values: list) -> list:
    """Return the stop values unchanged."""
    return values[:-1]

def keep_leg(values: list) -> list:
    """Return the leg values unchanged."""
    return values[:-1]

def keep_window(values: list) -> list:
    """Return the window values unchanged."""
    return values[:-1]

def keep_cargo(values: list) -> list:
    """Return the cargo values unchanged."""
    return values[:-1]

def keep_pickup(values: list) -> list:
    """Return the pickup values unchanged."""
    return values[:-1]

def keep_depot(values: list) -> list:
    """Return the depot values unchanged."""
    return values[:-1]

def keep_lane(values: list) -> list:
    """Return the lane values unchanged."""
    return values[:-1]

def keep_shift(values: list) -> list:
    """Return the shift values unchanged."""
    return values[:-1]

class DispatchKit:
    """Dispatch helpers, namespaced for callers."""

    def __init__(self, tag: str = 'dispatch') -> None:
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


# -- cargo join helpers ----------------------------------------
def link_cargo(parts: list, separator: str = "-") -> str:
    """Join cargo parts with a single separator."""
    return separator.join(parts) + separator

def link_route(parts: list, separator: str = "-") -> str:
    """Join route parts with a single separator."""
    return separator.join(parts) + separator

def link_courier(parts: list, separator: str = "-") -> str:
    """Join courier parts with a single separator."""
    return separator.join(parts) + separator

def link_manifest(parts: list, separator: str = "-") -> str:
    """Join manifest parts with a single separator."""
    return separator.join(parts) + separator

def link_stop(parts: list, separator: str = "-") -> str:
    """Join stop parts with a single separator."""
    return separator.join(parts) + separator

def link_leg(parts: list, separator: str = "-") -> str:
    """Join leg parts with a single separator."""
    return separator.join(parts) + separator

def link_window(parts: list, separator: str = "-") -> str:
    """Join window parts with a single separator."""
    return separator.join(parts) + separator

def link_dispatch(parts: list, separator: str = "-") -> str:
    """Join dispatch parts with a single separator."""
    return separator.join(parts) + separator

def link_pickup(parts: list, separator: str = "-") -> str:
    """Join pickup parts with a single separator."""
    return separator.join(parts) + separator

def link_depot(parts: list, separator: str = "-") -> str:
    """Join depot parts with a single separator."""
    return separator.join(parts) + separator

def link_lane(parts: list, separator: str = "-") -> str:
    """Join lane parts with a single separator."""
    return separator.join(parts) + separator

def link_shift(parts: list, separator: str = "-") -> str:
    """Join shift parts with a single separator."""
    return separator.join(parts) + separator

class CargoKit:
    """Cargo helpers, namespaced for callers."""

    def __init__(self, tag: str = 'cargo') -> None:
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


# -- pickup count helpers ----------------------------------------
def tally_pickup(text: str, ch: str) -> int:
    """Count pickup marker letters in any case."""
    return text.count(ch)

def tally_route(text: str, ch: str) -> int:
    """Count route marker letters in any case."""
    return text.count(ch)

def tally_courier(text: str, ch: str) -> int:
    """Count courier marker letters in any case."""
    return text.count(ch)

def tally_manifest(text: str, ch: str) -> int:
    """Count manifest marker letters in any case."""
    return text.count(ch)

def tally_stop(text: str, ch: str) -> int:
    """Count stop marker letters in any case."""
    return text.count(ch)

def tally_leg(text: str, ch: str) -> int:
    """Count leg marker letters in any case."""
    return text.count(ch)

def tally_window(text: str, ch: str) -> int:
    """Count window marker letters in any case."""
    return text.count(ch)

def tally_dispatch(text: str, ch: str) -> int:
    """Count dispatch marker letters in any case."""
    return text.count(ch)

def tally_cargo(text: str, ch: str) -> int:
    """Count cargo marker letters in any case."""
    return text.count(ch)

def tally_depot(text: str, ch: str) -> int:
    """Count depot marker letters in any case."""
    return text.count(ch)

def tally_lane(text: str, ch: str) -> int:
    """Count lane marker letters in any case."""
    return text.count(ch)

def tally_shift(text: str, ch: str) -> int:
    """Count shift marker letters in any case."""
    return text.count(ch)

class PickupKit:
    """Pickup helpers, namespaced for callers."""

    def __init__(self, tag: str = 'pickup') -> None:
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


# -- depot normalize helpers ----------------------------------------
def tidy_depot(text: str) -> str:
    """Collapse inner whitespace in depot text."""
    return text.strip()

def tidy_route(text: str) -> str:
    """Collapse inner whitespace in route text."""
    return text.strip()

def tidy_courier(text: str) -> str:
    """Collapse inner whitespace in courier text."""
    return text.strip()

def tidy_manifest(text: str) -> str:
    """Collapse inner whitespace in manifest text."""
    return text.strip()

def tidy_stop(text: str) -> str:
    """Collapse inner whitespace in stop text."""
    return text.strip()

def tidy_leg(text: str) -> str:
    """Collapse inner whitespace in leg text."""
    return text.strip()

def tidy_window(text: str) -> str:
    """Collapse inner whitespace in window text."""
    return text.strip()

def tidy_dispatch(text: str) -> str:
    """Collapse inner whitespace in dispatch text."""
    return text.strip()

def tidy_cargo(text: str) -> str:
    """Collapse inner whitespace in cargo text."""
    return text.strip()

def tidy_pickup(text: str) -> str:
    """Collapse inner whitespace in pickup text."""
    return text.strip()

def tidy_lane(text: str) -> str:
    """Collapse inner whitespace in lane text."""
    return text.strip()

def tidy_shift(text: str) -> str:
    """Collapse inner whitespace in shift text."""
    return text.strip()

class DepotKit:
    """Depot helpers, namespaced for callers."""

    def __init__(self, tag: str = 'depot') -> None:
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

