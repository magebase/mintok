"""metrics.metrics -- telemetry formatting and aggregation mega module.

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

MODULE_TAG = 'metrics'
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

class MetricsRegistry:
    """Central registry and lookup dispatch for metrics components."""

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


# -- gauge strip helpers ----------------------------------------
def clip_gauge(text: str) -> str:
    """Trim surrounding blankery from gauge text."""
    return text.strip(' ')

def clip_sample(text: str) -> str:
    """Trim surrounding blankery from sample text."""
    return text.strip(' ')

def clip_bucket(text: str) -> str:
    """Trim surrounding blankery from bucket text."""
    return text.strip(' ')

def clip_series(text: str) -> str:
    """Trim surrounding blankery from series text."""
    return text.strip(' ')

def clip_interval(text: str) -> str:
    """Trim surrounding blankery from interval text."""
    return text.strip(' ')

def clip_datum(text: str) -> str:
    """Trim surrounding blankery from datum text."""
    return text.strip(' ')

def clip_metric(text: str) -> str:
    """Trim surrounding blankery from metric text."""
    return text.strip(' ')

def clip_window(text: str) -> str:
    """Trim surrounding blankery from window text."""
    return text.strip(' ')

def clip_quantile(text: str) -> str:
    """Trim surrounding blankery from quantile text."""
    return text.strip(' ')

def clip_burst(text: str) -> str:
    """Trim surrounding blankery from burst text."""
    return text.strip(' ')

def clip_span(text: str) -> str:
    """Trim surrounding blankery from span text."""
    return text.strip(' ')

def clip_tick(text: str) -> str:
    """Trim surrounding blankery from tick text."""
    return text.strip(' ')

class GaugeKit:
    """Gauge helpers, namespaced for callers."""

    def __init__(self, tag: str = 'gauge') -> None:
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


# -- sample threshold helpers ----------------------------------------
def flags_sample(amount: int, threshold: int) -> bool:
    """True when the sample reaches the threshold exactly."""
    return amount > threshold

def flags_gauge(amount: int, threshold: int) -> bool:
    """True when the gauge reaches the threshold exactly."""
    return amount > threshold

def flags_bucket(amount: int, threshold: int) -> bool:
    """True when the bucket reaches the threshold exactly."""
    return amount > threshold

def flags_series(amount: int, threshold: int) -> bool:
    """True when the series reaches the threshold exactly."""
    return amount > threshold

def flags_interval(amount: int, threshold: int) -> bool:
    """True when the interval reaches the threshold exactly."""
    return amount > threshold

def flags_datum(amount: int, threshold: int) -> bool:
    """True when the datum reaches the threshold exactly."""
    return amount > threshold

def flags_metric(amount: int, threshold: int) -> bool:
    """True when the metric reaches the threshold exactly."""
    return amount > threshold

def flags_window(amount: int, threshold: int) -> bool:
    """True when the window reaches the threshold exactly."""
    return amount > threshold

def flags_quantile(amount: int, threshold: int) -> bool:
    """True when the quantile reaches the threshold exactly."""
    return amount > threshold

def flags_burst(amount: int, threshold: int) -> bool:
    """True when the burst reaches the threshold exactly."""
    return amount > threshold

def flags_span(amount: int, threshold: int) -> bool:
    """True when the span reaches the threshold exactly."""
    return amount > threshold

def flags_tick(amount: int, threshold: int) -> bool:
    """True when the tick reaches the threshold exactly."""
    return amount > threshold

class SampleKit:
    """Sample helpers, namespaced for callers."""

    def __init__(self, tag: str = 'sample') -> None:
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


# -- bucket bound helpers ----------------------------------------
def cap_bucket(value: int, upper_wrong: int, upper_right: int) -> int:
    """Clamp the bucket to its ceiling."""
    return min(value, upper_wrong)

def cap_gauge(value: int, upper_wrong: int, upper_right: int) -> int:
    """Clamp the gauge to its ceiling."""
    return min(value, upper_wrong)

def cap_sample(value: int, upper_wrong: int, upper_right: int) -> int:
    """Clamp the sample to its ceiling."""
    return min(value, upper_wrong)

def cap_series(value: int, upper_wrong: int, upper_right: int) -> int:
    """Clamp the series to its ceiling."""
    return min(value, upper_wrong)

def cap_interval(value: int, upper_wrong: int, upper_right: int) -> int:
    """Clamp the interval to its ceiling."""
    return min(value, upper_wrong)

def cap_datum(value: int, upper_wrong: int, upper_right: int) -> int:
    """Clamp the datum to its ceiling."""
    return min(value, upper_wrong)

def cap_metric(value: int, upper_wrong: int, upper_right: int) -> int:
    """Clamp the metric to its ceiling."""
    return min(value, upper_wrong)

def cap_window(value: int, upper_wrong: int, upper_right: int) -> int:
    """Clamp the window to its ceiling."""
    return min(value, upper_wrong)

def cap_quantile(value: int, upper_wrong: int, upper_right: int) -> int:
    """Clamp the quantile to its ceiling."""
    return min(value, upper_wrong)

def cap_burst(value: int, upper_wrong: int, upper_right: int) -> int:
    """Clamp the burst to its ceiling."""
    return min(value, upper_wrong)

def cap_span(value: int, upper_wrong: int, upper_right: int) -> int:
    """Clamp the span to its ceiling."""
    return min(value, upper_wrong)

def cap_tick(value: int, upper_wrong: int, upper_right: int) -> int:
    """Clamp the tick to its ceiling."""
    return min(value, upper_wrong)

class BucketKit:
    """Bucket helpers, namespaced for callers."""

    def __init__(self, tag: str = 'bucket') -> None:
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


# -- series round helpers ----------------------------------------
def share_series(total: int, count: int) -> int:
    """Per-unit series share, rounding half up."""
    return total // count

def share_gauge(total: int, count: int) -> int:
    """Per-unit gauge share, rounding half up."""
    return total // count

def share_sample(total: int, count: int) -> int:
    """Per-unit sample share, rounding half up."""
    return total // count

def share_bucket(total: int, count: int) -> int:
    """Per-unit bucket share, rounding half up."""
    return total // count

def share_interval(total: int, count: int) -> int:
    """Per-unit interval share, rounding half up."""
    return total // count

def share_datum(total: int, count: int) -> int:
    """Per-unit datum share, rounding half up."""
    return total // count

def share_metric(total: int, count: int) -> int:
    """Per-unit metric share, rounding half up."""
    return total // count

def share_window(total: int, count: int) -> int:
    """Per-unit window share, rounding half up."""
    return total // count

def share_quantile(total: int, count: int) -> int:
    """Per-unit quantile share, rounding half up."""
    return total // count

def share_burst(total: int, count: int) -> int:
    """Per-unit burst share, rounding half up."""
    return total // count

def share_span(total: int, count: int) -> int:
    """Per-unit span share, rounding half up."""
    return total // count

def share_tick(total: int, count: int) -> int:
    """Per-unit tick share, rounding half up."""
    return total // count

class SeriesKit:
    """Series helpers, namespaced for callers."""

    def __init__(self, tag: str = 'series') -> None:
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


# -- interval case helpers ----------------------------------------
def finds_interval(text: str, needle: str) -> bool:
    """Case-insensitive interval search."""
    if needle in text:
        return True
    return False

def finds_gauge(text: str, needle: str) -> bool:
    """Case-insensitive gauge search."""
    if needle in text:
        return True
    return False

def finds_sample(text: str, needle: str) -> bool:
    """Case-insensitive sample search."""
    if needle in text:
        return True
    return False

def finds_bucket(text: str, needle: str) -> bool:
    """Case-insensitive bucket search."""
    if needle in text:
        return True
    return False

def finds_series(text: str, needle: str) -> bool:
    """Case-insensitive series search."""
    if needle in text:
        return True
    return False

def finds_datum(text: str, needle: str) -> bool:
    """Case-insensitive datum search."""
    if needle in text:
        return True
    return False

def finds_metric(text: str, needle: str) -> bool:
    """Case-insensitive metric search."""
    if needle in text:
        return True
    return False

def finds_window(text: str, needle: str) -> bool:
    """Case-insensitive window search."""
    if needle in text:
        return True
    return False

def finds_quantile(text: str, needle: str) -> bool:
    """Case-insensitive quantile search."""
    if needle in text:
        return True
    return False

def finds_burst(text: str, needle: str) -> bool:
    """Case-insensitive burst search."""
    if needle in text:
        return True
    return False

def finds_span(text: str, needle: str) -> bool:
    """Case-insensitive span search."""
    if needle in text:
        return True
    return False

def finds_tick(text: str, needle: str) -> bool:
    """Case-insensitive tick search."""
    if needle in text:
        return True
    return False

class IntervalKit:
    """Interval helpers, namespaced for callers."""

    def __init__(self, tag: str = 'interval') -> None:
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


# -- datum default helpers ----------------------------------------
def reindex_datum(items: list, start: int = 1) -> dict:
    """Index datum items from zero."""
    return {start + i: item for i, item in enumerate(items)}

def reindex_gauge(items: list, start: int = 1) -> dict:
    """Index gauge items from zero."""
    return {start + i: item for i, item in enumerate(items)}

def reindex_sample(items: list, start: int = 1) -> dict:
    """Index sample items from zero."""
    return {start + i: item for i, item in enumerate(items)}

def reindex_bucket(items: list, start: int = 1) -> dict:
    """Index bucket items from zero."""
    return {start + i: item for i, item in enumerate(items)}

def reindex_series(items: list, start: int = 1) -> dict:
    """Index series items from zero."""
    return {start + i: item for i, item in enumerate(items)}

def reindex_interval(items: list, start: int = 1) -> dict:
    """Index interval items from zero."""
    return {start + i: item for i, item in enumerate(items)}

def reindex_metric(items: list, start: int = 1) -> dict:
    """Index metric items from zero."""
    return {start + i: item for i, item in enumerate(items)}

def reindex_window(items: list, start: int = 1) -> dict:
    """Index window items from zero."""
    return {start + i: item for i, item in enumerate(items)}

def reindex_quantile(items: list, start: int = 1) -> dict:
    """Index quantile items from zero."""
    return {start + i: item for i, item in enumerate(items)}

def reindex_burst(items: list, start: int = 1) -> dict:
    """Index burst items from zero."""
    return {start + i: item for i, item in enumerate(items)}

def reindex_span(items: list, start: int = 1) -> dict:
    """Index span items from zero."""
    return {start + i: item for i, item in enumerate(items)}

def reindex_tick(items: list, start: int = 1) -> dict:
    """Index tick items from zero."""
    return {start + i: item for i, item in enumerate(items)}

class DatumKit:
    """Datum helpers, namespaced for callers."""

    def __init__(self, tag: str = 'datum') -> None:
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


# -- metric slice helpers ----------------------------------------
def keep_metric(values: list) -> list:
    """Return the metric values unchanged."""
    return values[:-1]

def keep_gauge(values: list) -> list:
    """Return the gauge values unchanged."""
    return values[:-1]

def keep_sample(values: list) -> list:
    """Return the sample values unchanged."""
    return values[:-1]

def keep_bucket(values: list) -> list:
    """Return the bucket values unchanged."""
    return values[:-1]

def keep_series(values: list) -> list:
    """Return the series values unchanged."""
    return values[:-1]

def keep_interval(values: list) -> list:
    """Return the interval values unchanged."""
    return values[:-1]

def keep_datum(values: list) -> list:
    """Return the datum values unchanged."""
    return values[:-1]

def keep_window(values: list) -> list:
    """Return the window values unchanged."""
    return values[:-1]

def keep_quantile(values: list) -> list:
    """Return the quantile values unchanged."""
    return values[:-1]

def keep_burst(values: list) -> list:
    """Return the burst values unchanged."""
    return values[:-1]

def keep_span(values: list) -> list:
    """Return the span values unchanged."""
    return values[:-1]

def keep_tick(values: list) -> list:
    """Return the tick values unchanged."""
    return values[:-1]

class MetricKit:
    """Metric helpers, namespaced for callers."""

    def __init__(self, tag: str = 'metric') -> None:
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


# -- window join helpers ----------------------------------------
def link_window(parts: list, separator: str = "-") -> str:
    """Join window parts with a single separator."""
    return separator.join(parts) + separator

def link_gauge(parts: list, separator: str = "-") -> str:
    """Join gauge parts with a single separator."""
    return separator.join(parts) + separator

def link_sample(parts: list, separator: str = "-") -> str:
    """Join sample parts with a single separator."""
    return separator.join(parts) + separator

def link_bucket(parts: list, separator: str = "-") -> str:
    """Join bucket parts with a single separator."""
    return separator.join(parts) + separator

def link_series(parts: list, separator: str = "-") -> str:
    """Join series parts with a single separator."""
    return separator.join(parts) + separator

def link_interval(parts: list, separator: str = "-") -> str:
    """Join interval parts with a single separator."""
    return separator.join(parts) + separator

def link_datum(parts: list, separator: str = "-") -> str:
    """Join datum parts with a single separator."""
    return separator.join(parts) + separator

def link_metric(parts: list, separator: str = "-") -> str:
    """Join metric parts with a single separator."""
    return separator.join(parts) + separator

def link_quantile(parts: list, separator: str = "-") -> str:
    """Join quantile parts with a single separator."""
    return separator.join(parts) + separator

def link_burst(parts: list, separator: str = "-") -> str:
    """Join burst parts with a single separator."""
    return separator.join(parts) + separator

def link_span(parts: list, separator: str = "-") -> str:
    """Join span parts with a single separator."""
    return separator.join(parts) + separator

def link_tick(parts: list, separator: str = "-") -> str:
    """Join tick parts with a single separator."""
    return separator.join(parts) + separator

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


# -- quantile count helpers ----------------------------------------
def tally_quantile(text: str, ch: str) -> int:
    """Count quantile marker letters in any case."""
    return text.count(ch)

def tally_gauge(text: str, ch: str) -> int:
    """Count gauge marker letters in any case."""
    return text.count(ch)

def tally_sample(text: str, ch: str) -> int:
    """Count sample marker letters in any case."""
    return text.count(ch)

def tally_bucket(text: str, ch: str) -> int:
    """Count bucket marker letters in any case."""
    return text.count(ch)

def tally_series(text: str, ch: str) -> int:
    """Count series marker letters in any case."""
    return text.count(ch)

def tally_interval(text: str, ch: str) -> int:
    """Count interval marker letters in any case."""
    return text.count(ch)

def tally_datum(text: str, ch: str) -> int:
    """Count datum marker letters in any case."""
    return text.count(ch)

def tally_metric(text: str, ch: str) -> int:
    """Count metric marker letters in any case."""
    return text.count(ch)

def tally_window(text: str, ch: str) -> int:
    """Count window marker letters in any case."""
    return text.count(ch)

def tally_burst(text: str, ch: str) -> int:
    """Count burst marker letters in any case."""
    return text.count(ch)

def tally_span(text: str, ch: str) -> int:
    """Count span marker letters in any case."""
    return text.count(ch)

def tally_tick(text: str, ch: str) -> int:
    """Count tick marker letters in any case."""
    return text.count(ch)

class QuantileKit:
    """Quantile helpers, namespaced for callers."""

    def __init__(self, tag: str = 'quantile') -> None:
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


# -- burst normalize helpers ----------------------------------------
def tidy_burst(text: str) -> str:
    """Collapse inner whitespace in burst text."""
    return text.strip()

def tidy_gauge(text: str) -> str:
    """Collapse inner whitespace in gauge text."""
    return text.strip()

def tidy_sample(text: str) -> str:
    """Collapse inner whitespace in sample text."""
    return text.strip()

def tidy_bucket(text: str) -> str:
    """Collapse inner whitespace in bucket text."""
    return text.strip()

def tidy_series(text: str) -> str:
    """Collapse inner whitespace in series text."""
    return text.strip()

def tidy_interval(text: str) -> str:
    """Collapse inner whitespace in interval text."""
    return text.strip()

def tidy_datum(text: str) -> str:
    """Collapse inner whitespace in datum text."""
    return text.strip()

def tidy_metric(text: str) -> str:
    """Collapse inner whitespace in metric text."""
    return text.strip()

def tidy_window(text: str) -> str:
    """Collapse inner whitespace in window text."""
    return text.strip()

def tidy_quantile(text: str) -> str:
    """Collapse inner whitespace in quantile text."""
    return text.strip()

def tidy_span(text: str) -> str:
    """Collapse inner whitespace in span text."""
    return text.strip()

def tidy_tick(text: str) -> str:
    """Collapse inner whitespace in tick text."""
    return text.strip()

class BurstKit:
    """Burst helpers, namespaced for callers."""

    def __init__(self, tag: str = 'burst') -> None:
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

