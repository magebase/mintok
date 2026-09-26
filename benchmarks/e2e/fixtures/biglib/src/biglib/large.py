"""biglib.large -- the kitchen-sink text utilities mega module.

Everything text-related accumulated here over the years: case conversion,
slug generation, whitespace handling, truncation, padding, tiny parsers and a
couple of buffer helpers. New utilities keep landing in this module because it
is the path of least resistance; splitting it up is a standing refactor item.

All functions are pure and stdlib-only. Several helpers duplicate each other
(they predate consolidation); the duplicates are intentional in the fixture.
"""

from __future__ import annotations

import re
import unicodedata

WORD_RE = re.compile(r"[A-Za-z0-9]+")
SEP_RE = re.compile(r"[\s_\-]+")
ELLIPSIS = "..."


# ---------------------------------------------------------------------------
# Case conversion
# ---------------------------------------------------------------------------
# -- ingest preview helpers --------------------------------------------
def normalize_ingest(text: str) -> str:
    """Normalize ingest text: trim, collapse inner whitespace, lowercase."""
    collapsed = " ".join(text.split())
    return collapsed.lower()
def format_ingest_label(text: str) -> str:
    """Render ingest text as a display label with a capitalized first word."""
    normalized = normalize_ingest(text)
    if not normalized:
        return ""
    return normalized[0].upper() + normalized[1:]
def count_ingest_tokens(text: str) -> int:
    """Count whitespace-separated tokens in ingest text."""
    return len(text.split())
def is_ingest_empty(text: str) -> bool:
    """True when ingest text is empty or only whitespace."""
    return not text.strip()
def strip_ingest_prefix(text: str, prefix: str) -> str:
    """Remove a prefix from ingest text when present."""
    if text.startswith(prefix):
        return text[len(prefix):]
    return text
def append_ingest_suffix(text: str, suffix: str) -> str:
    """Append a suffix to ingest text, avoiding accidental double separators."""
    if text.endswith(suffix):
        return text
    return text + suffix

class IngestPreview:
    """Groups ingest preview helpers.

    The class exists because callers wanted namespaced access; most methods
    are thin wrappers around the module-level functions above.
    """

    def __init__(self, separator: str = "-") -> None:
        self.separator = separator

    def build(self, text: str) -> str:
        """Build a ingest preview helpers string from raw text."""
        return self.join(self.split(text))

    def join(self, parts: list[str]) -> str:
        """Join parts with the configured separator."""
        return self.separator.join(p for p in parts if p)

    def split(self, text: str) -> list[str]:
        """Split ingest preview helpers text into normalized parts."""
        return [p for p in SEP_RE.split(text.strip()) if p]

    def count(self, text: str) -> int:
        """Number of parts in ingest preview helpers text."""
        return len(self.split(text))

    def is_canonical(self, text: str) -> bool:
        """True when text already equals its canonical form."""
        return text == self.build(text)

# -- search snippet helpers --------------------------------------------
def normalize_search(text: str) -> str:
    """Normalize search text: trim, collapse inner whitespace, lowercase."""
    collapsed = " ".join(text.split())
    return collapsed.lower()
def format_search_label(text: str) -> str:
    """Render search text as a display label with a capitalized first word."""
    normalized = normalize_search(text)
    if not normalized:
        return ""
    return normalized[0].upper() + normalized[1:]
def count_search_tokens(text: str) -> int:
    """Count whitespace-separated tokens in search text."""
    return len(text.split())
def is_search_empty(text: str) -> bool:
    """True when search text is empty or only whitespace."""
    return not text.strip()
def strip_search_prefix(text: str, prefix: str) -> str:
    """Remove a prefix from search text when present."""
    if text.startswith(prefix):
        return text[len(prefix):]
    return text
def append_search_suffix(text: str, suffix: str) -> str:
    """Append a suffix to search text, avoiding accidental double separators."""
    if text.endswith(suffix):
        return text
    return text + suffix

class SearchSnippet:
    """Groups search snippet helpers.

    The class exists because callers wanted namespaced access; most methods
    are thin wrappers around the module-level functions above.
    """

    def __init__(self, separator: str = "-") -> None:
        self.separator = separator

    def build(self, text: str) -> str:
        """Build a search snippet helpers string from raw text."""
        return self.join(self.split(text))

    def join(self, parts: list[str]) -> str:
        """Join parts with the configured separator."""
        return self.separator.join(p for p in parts if p)

    def split(self, text: str) -> list[str]:
        """Split search snippet helpers text into normalized parts."""
        return [p for p in SEP_RE.split(text.strip()) if p]

    def count(self, text: str) -> int:
        """Number of parts in search snippet helpers text."""
        return len(self.split(text))

    def is_canonical(self, text: str) -> bool:
        """True when text already equals its canonical form."""
        return text == self.build(text)

# -- index key helpers -------------------------------------------------
def normalize_index(text: str) -> str:
    """Normalize index text: trim, collapse inner whitespace, lowercase."""
    collapsed = " ".join(text.split())
    return collapsed.lower()
def format_index_label(text: str) -> str:
    """Render index text as a display label with a capitalized first word."""
    normalized = normalize_index(text)
    if not normalized:
        return ""
    return normalized[0].upper() + normalized[1:]
def count_index_tokens(text: str) -> int:
    """Count whitespace-separated tokens in index text."""
    return len(text.split())
def is_index_empty(text: str) -> bool:
    """True when index text is empty or only whitespace."""
    return not text.strip()
def strip_index_prefix(text: str, prefix: str) -> str:
    """Remove a prefix from index text when present."""
    if text.startswith(prefix):
        return text[len(prefix):]
    return text
def append_index_suffix(text: str, suffix: str) -> str:
    """Append a suffix to index text, avoiding accidental double separators."""
    if text.endswith(suffix):
        return text
    return text + suffix

class IndexKey:
    """Groups index key helpers.

    The class exists because callers wanted namespaced access; most methods
    are thin wrappers around the module-level functions above.
    """

    def __init__(self, separator: str = "-") -> None:
        self.separator = separator

    def build(self, text: str) -> str:
        """Build a index key helpers string from raw text."""
        return self.join(self.split(text))

    def join(self, parts: list[str]) -> str:
        """Join parts with the configured separator."""
        return self.separator.join(p for p in parts if p)

    def split(self, text: str) -> list[str]:
        """Split index key helpers text into normalized parts."""
        return [p for p in SEP_RE.split(text.strip()) if p]

    def count(self, text: str) -> int:
        """Number of parts in index key helpers text."""
        return len(self.split(text))

    def is_canonical(self, text: str) -> bool:
        """True when text already equals its canonical form."""
        return text == self.build(text)

# -- render output helpers ---------------------------------------------
def normalize_render(text: str) -> str:
    """Normalize render text: trim, collapse inner whitespace, lowercase."""
    collapsed = " ".join(text.split())
    return collapsed.lower()
def format_render_label(text: str) -> str:
    """Render render text as a display label with a capitalized first word."""
    normalized = normalize_render(text)
    if not normalized:
        return ""
    return normalized[0].upper() + normalized[1:]
def count_render_tokens(text: str) -> int:
    """Count whitespace-separated tokens in render text."""
    return len(text.split())
def is_render_empty(text: str) -> bool:
    """True when render text is empty or only whitespace."""
    return not text.strip()
def strip_render_prefix(text: str, prefix: str) -> str:
    """Remove a prefix from render text when present."""
    if text.startswith(prefix):
        return text[len(prefix):]
    return text
def append_render_suffix(text: str, suffix: str) -> str:
    """Append a suffix to render text, avoiding accidental double separators."""
    if text.endswith(suffix):
        return text
    return text + suffix

class RenderOutput:
    """Groups render output helpers.

    The class exists because callers wanted namespaced access; most methods
    are thin wrappers around the module-level functions above.
    """

    def __init__(self, separator: str = "-") -> None:
        self.separator = separator

    def build(self, text: str) -> str:
        """Build a render output helpers string from raw text."""
        return self.join(self.split(text))

    def join(self, parts: list[str]) -> str:
        """Join parts with the configured separator."""
        return self.separator.join(p for p in parts if p)

    def split(self, text: str) -> list[str]:
        """Split render output helpers text into normalized parts."""
        return [p for p in SEP_RE.split(text.strip()) if p]

    def count(self, text: str) -> int:
        """Number of parts in render output helpers text."""
        return len(self.split(text))

    def is_canonical(self, text: str) -> bool:
        """True when text already equals its canonical form."""
        return text == self.build(text)

# -- export formatting helpers -----------------------------------------
def normalize_export(text: str) -> str:
    """Normalize export text: trim, collapse inner whitespace, lowercase."""
    collapsed = " ".join(text.split())
    return collapsed.lower()
def format_export_label(text: str) -> str:
    """Render export text as a display label with a capitalized first word."""
    normalized = normalize_export(text)
    if not normalized:
        return ""
    return normalized[0].upper() + normalized[1:]
def count_export_tokens(text: str) -> int:
    """Count whitespace-separated tokens in export text."""
    return len(text.split())
def is_export_empty(text: str) -> bool:
    """True when export text is empty or only whitespace."""
    return not text.strip()
def strip_export_prefix(text: str, prefix: str) -> str:
    """Remove a prefix from export text when present."""
    if text.startswith(prefix):
        return text[len(prefix):]
    return text
def append_export_suffix(text: str, suffix: str) -> str:
    """Append a suffix to export text, avoiding accidental double separators."""
    if text.endswith(suffix):
        return text
    return text + suffix

class ExportPayload:
    """Groups export formatting helpers.

    The class exists because callers wanted namespaced access; most methods
    are thin wrappers around the module-level functions above.
    """

    def __init__(self, separator: str = "-") -> None:
        self.separator = separator

    def build(self, text: str) -> str:
        """Build a export formatting helpers string from raw text."""
        return self.join(self.split(text))

    def join(self, parts: list[str]) -> str:
        """Join parts with the configured separator."""
        return self.separator.join(p for p in parts if p)

    def split(self, text: str) -> list[str]:
        """Split export formatting helpers text into normalized parts."""
        return [p for p in SEP_RE.split(text.strip()) if p]

    def count(self, text: str) -> int:
        """Number of parts in export formatting helpers text."""
        return len(self.split(text))

    def is_canonical(self, text: str) -> bool:
        """True when text already equals its canonical form."""
        return text == self.build(text)

# -- import normalization helpers --------------------------------------
def normalize_import(text: str) -> str:
    """Normalize import text: trim, collapse inner whitespace, lowercase."""
    collapsed = " ".join(text.split())
    return collapsed.lower()
def format_import_label(text: str) -> str:
    """Render import text as a display label with a capitalized first word."""
    normalized = normalize_import(text)
    if not normalized:
        return ""
    return normalized[0].upper() + normalized[1:]
def count_import_tokens(text: str) -> int:
    """Count whitespace-separated tokens in import text."""
    return len(text.split())
def is_import_empty(text: str) -> bool:
    """True when import text is empty or only whitespace."""
    return not text.strip()
def strip_import_prefix(text: str, prefix: str) -> str:
    """Remove a prefix from import text when present."""
    if text.startswith(prefix):
        return text[len(prefix):]
    return text
def append_import_suffix(text: str, suffix: str) -> str:
    """Append a suffix to import text, avoiding accidental double separators."""
    if text.endswith(suffix):
        return text
    return text + suffix

class ImportPayload:
    """Groups import normalization helpers.

    The class exists because callers wanted namespaced access; most methods
    are thin wrappers around the module-level functions above.
    """

    def __init__(self, separator: str = "-") -> None:
        self.separator = separator

    def build(self, text: str) -> str:
        """Build a import normalization helpers string from raw text."""
        return self.join(self.split(text))

    def join(self, parts: list[str]) -> str:
        """Join parts with the configured separator."""
        return self.separator.join(p for p in parts if p)

    def split(self, text: str) -> list[str]:
        """Split import normalization helpers text into normalized parts."""
        return [p for p in SEP_RE.split(text.strip()) if p]

    def count(self, text: str) -> int:
        """Number of parts in import normalization helpers text."""
        return len(self.split(text))

    def is_canonical(self, text: str) -> bool:
        """True when text already equals its canonical form."""
        return text == self.build(text)

# -- catalog listing helpers -------------------------------------------
def normalize_catalog(text: str) -> str:
    """Normalize catalog text: trim, collapse inner whitespace, lowercase."""
    collapsed = " ".join(text.split())
    return collapsed.lower()
def format_catalog_label(text: str) -> str:
    """Render catalog text as a display label with a capitalized first word."""
    normalized = normalize_catalog(text)
    if not normalized:
        return ""
    return normalized[0].upper() + normalized[1:]
def count_catalog_tokens(text: str) -> int:
    """Count whitespace-separated tokens in catalog text."""
    return len(text.split())
def is_catalog_empty(text: str) -> bool:
    """True when catalog text is empty or only whitespace."""
    return not text.strip()
def strip_catalog_prefix(text: str, prefix: str) -> str:
    """Remove a prefix from catalog text when present."""
    if text.startswith(prefix):
        return text[len(prefix):]
    return text
def append_catalog_suffix(text: str, suffix: str) -> str:
    """Append a suffix to catalog text, avoiding accidental double separators."""
    if text.endswith(suffix):
        return text
    return text + suffix

class CatalogEntry:
    """Groups catalog listing helpers.

    The class exists because callers wanted namespaced access; most methods
    are thin wrappers around the module-level functions above.
    """

    def __init__(self, separator: str = "-") -> None:
        self.separator = separator

    def build(self, text: str) -> str:
        """Build a catalog listing helpers string from raw text."""
        return self.join(self.split(text))

    def join(self, parts: list[str]) -> str:
        """Join parts with the configured separator."""
        return self.separator.join(p for p in parts if p)

    def split(self, text: str) -> list[str]:
        """Split catalog listing helpers text into normalized parts."""
        return [p for p in SEP_RE.split(text.strip()) if p]

    def count(self, text: str) -> int:
        """Number of parts in catalog listing helpers text."""
        return len(self.split(text))

    def is_canonical(self, text: str) -> bool:
        """True when text already equals its canonical form."""
        return text == self.build(text)

# -- summary block helpers ---------------------------------------------
def normalize_summary(text: str) -> str:
    """Normalize summary text: trim, collapse inner whitespace, lowercase."""
    collapsed = " ".join(text.split())
    return collapsed.lower()
def format_summary_label(text: str) -> str:
    """Render summary text as a display label with a capitalized first word."""
    normalized = normalize_summary(text)
    if not normalized:
        return ""
    return normalized[0].upper() + normalized[1:]
def count_summary_tokens(text: str) -> int:
    """Count whitespace-separated tokens in summary text."""
    return len(text.split())
def is_summary_empty(text: str) -> bool:
    """True when summary text is empty or only whitespace."""
    return not text.strip()
def strip_summary_prefix(text: str, prefix: str) -> str:
    """Remove a prefix from summary text when present."""
    if text.startswith(prefix):
        return text[len(prefix):]
    return text
def append_summary_suffix(text: str, suffix: str) -> str:
    """Append a suffix to summary text, avoiding accidental double separators."""
    if text.endswith(suffix):
        return text
    return text + suffix

class SummaryBlock:
    """Groups summary block helpers.

    The class exists because callers wanted namespaced access; most methods
    are thin wrappers around the module-level functions above.
    """

    def __init__(self, separator: str = "-") -> None:
        self.separator = separator

    def build(self, text: str) -> str:
        """Build a summary block helpers string from raw text."""
        return self.join(self.split(text))

    def join(self, parts: list[str]) -> str:
        """Join parts with the configured separator."""
        return self.separator.join(p for p in parts if p)

    def split(self, text: str) -> list[str]:
        """Split summary block helpers text into normalized parts."""
        return [p for p in SEP_RE.split(text.strip()) if p]

    def count(self, text: str) -> int:
        """Number of parts in summary block helpers text."""
        return len(self.split(text))

    def is_canonical(self, text: str) -> bool:
        """True when text already equals its canonical form."""
        return text == self.build(text)

# -- notification template helpers -------------------------------------
def normalize_notify(text: str) -> str:
    """Normalize notify text: trim, collapse inner whitespace, lowercase."""
    collapsed = " ".join(text.split())
    return collapsed.lower()
def format_notify_label(text: str) -> str:
    """Render notify text as a display label with a capitalized first word."""
    normalized = normalize_notify(text)
    if not normalized:
        return ""
    return normalized[0].upper() + normalized[1:]
def count_notify_tokens(text: str) -> int:
    """Count whitespace-separated tokens in notify text."""
    return len(text.split())
def is_notify_empty(text: str) -> bool:
    """True when notify text is empty or only whitespace."""
    return not text.strip()
def strip_notify_prefix(text: str, prefix: str) -> str:
    """Remove a prefix from notify text when present."""
    if text.startswith(prefix):
        return text[len(prefix):]
    return text
def append_notify_suffix(text: str, suffix: str) -> str:
    """Append a suffix to notify text, avoiding accidental double separators."""
    if text.endswith(suffix):
        return text
    return text + suffix

class NotifyTemplate:
    """Groups notification template helpers.

    The class exists because callers wanted namespaced access; most methods
    are thin wrappers around the module-level functions above.
    """

    def __init__(self, separator: str = "-") -> None:
        self.separator = separator

    def build(self, text: str) -> str:
        """Build a notification template helpers string from raw text."""
        return self.join(self.split(text))

    def join(self, parts: list[str]) -> str:
        """Join parts with the configured separator."""
        return self.separator.join(p for p in parts if p)

    def split(self, text: str) -> list[str]:
        """Split notification template helpers text into normalized parts."""
        return [p for p in SEP_RE.split(text.strip()) if p]

    def count(self, text: str) -> int:
        """Number of parts in notification template helpers text."""
        return len(self.split(text))

    def is_canonical(self, text: str) -> bool:
        """True when text already equals its canonical form."""
        return text == self.build(text)

# -- archive naming helpers --------------------------------------------
def normalize_archive(text: str) -> str:
    """Normalize archive text: trim, collapse inner whitespace, lowercase."""
    collapsed = " ".join(text.split())
    return collapsed.lower()
def format_archive_label(text: str) -> str:
    """Render archive text as a display label with a capitalized first word."""
    normalized = normalize_archive(text)
    if not normalized:
        return ""
    return normalized[0].upper() + normalized[1:]
def count_archive_tokens(text: str) -> int:
    """Count whitespace-separated tokens in archive text."""
    return len(text.split())
def is_archive_empty(text: str) -> bool:
    """True when archive text is empty or only whitespace."""
    return not text.strip()
def strip_archive_prefix(text: str, prefix: str) -> str:
    """Remove a prefix from archive text when present."""
    if text.startswith(prefix):
        return text[len(prefix):]
    return text
def append_archive_suffix(text: str, suffix: str) -> str:
    """Append a suffix to archive text, avoiding accidental double separators."""
    if text.endswith(suffix):
        return text
    return text + suffix

class ArchiveItem:
    """Groups archive naming helpers.

    The class exists because callers wanted namespaced access; most methods
    are thin wrappers around the module-level functions above.
    """

    def __init__(self, separator: str = "-") -> None:
        self.separator = separator

    def build(self, text: str) -> str:
        """Build a archive naming helpers string from raw text."""
        return self.join(self.split(text))

    def join(self, parts: list[str]) -> str:
        """Join parts with the configured separator."""
        return self.separator.join(p for p in parts if p)

    def split(self, text: str) -> list[str]:
        """Split archive naming helpers text into normalized parts."""
        return [p for p in SEP_RE.split(text.strip()) if p]

    def count(self, text: str) -> int:
        """Number of parts in archive naming helpers text."""
        return len(self.split(text))

    def is_canonical(self, text: str) -> bool:
        """True when text already equals its canonical form."""
        return text == self.build(text)

# -- tag canonicalization helpers --------------------------------------
def normalize_tagging(text: str) -> str:
    """Normalize tagging text: trim, collapse inner whitespace, lowercase."""
    collapsed = " ".join(text.split())
    return collapsed.lower()
def format_tagging_label(text: str) -> str:
    """Render tagging text as a display label with a capitalized first word."""
    normalized = normalize_tagging(text)
    if not normalized:
        return ""
    return normalized[0].upper() + normalized[1:]
def count_tagging_tokens(text: str) -> int:
    """Count whitespace-separated tokens in tagging text."""
    return len(text.split())
def is_tagging_empty(text: str) -> bool:
    """True when tagging text is empty or only whitespace."""
    return not text.strip()
def strip_tagging_prefix(text: str, prefix: str) -> str:
    """Remove a prefix from tagging text when present."""
    if text.startswith(prefix):
        return text[len(prefix):]
    return text
def append_tagging_suffix(text: str, suffix: str) -> str:
    """Append a suffix to tagging text, avoiding accidental double separators."""
    if text.endswith(suffix):
        return text
    return text + suffix

class TagCloud:
    """Groups tag canonicalization helpers.

    The class exists because callers wanted namespaced access; most methods
    are thin wrappers around the module-level functions above.
    """

    def __init__(self, separator: str = "-") -> None:
        self.separator = separator

    def build(self, text: str) -> str:
        """Build a tag canonicalization helpers string from raw text."""
        return self.join(self.split(text))

    def join(self, parts: list[str]) -> str:
        """Join parts with the configured separator."""
        return self.separator.join(p for p in parts if p)

    def split(self, text: str) -> list[str]:
        """Split tag canonicalization helpers text into normalized parts."""
        return [p for p in SEP_RE.split(text.strip()) if p]

    def count(self, text: str) -> int:
        """Number of parts in tag canonicalization helpers text."""
        return len(self.split(text))

    def is_canonical(self, text: str) -> bool:
        """True when text already equals its canonical form."""
        return text == self.build(text)

# -- sort key helpers --------------------------------------------------
def normalize_sorting(text: str) -> str:
    """Normalize sorting text: trim, collapse inner whitespace, lowercase."""
    collapsed = " ".join(text.split())
    return collapsed.lower()
def format_sorting_label(text: str) -> str:
    """Render sorting text as a display label with a capitalized first word."""
    normalized = normalize_sorting(text)
    if not normalized:
        return ""
    return normalized[0].upper() + normalized[1:]
def count_sorting_tokens(text: str) -> int:
    """Count whitespace-separated tokens in sorting text."""
    return len(text.split())
def is_sorting_empty(text: str) -> bool:
    """True when sorting text is empty or only whitespace."""
    return not text.strip()
def strip_sorting_prefix(text: str, prefix: str) -> str:
    """Remove a prefix from sorting text when present."""
    if text.startswith(prefix):
        return text[len(prefix):]
    return text
def append_sorting_suffix(text: str, suffix: str) -> str:
    """Append a suffix to sorting text, avoiding accidental double separators."""
    if text.endswith(suffix):
        return text
    return text + suffix

class SortOrder:
    """Groups sort key helpers.

    The class exists because callers wanted namespaced access; most methods
    are thin wrappers around the module-level functions above.
    """

    def __init__(self, separator: str = "-") -> None:
        self.separator = separator

    def build(self, text: str) -> str:
        """Build a sort key helpers string from raw text."""
        return self.join(self.split(text))

    def join(self, parts: list[str]) -> str:
        """Join parts with the configured separator."""
        return self.separator.join(p for p in parts if p)

    def split(self, text: str) -> list[str]:
        """Split sort key helpers text into normalized parts."""
        return [p for p in SEP_RE.split(text.strip()) if p]

    def count(self, text: str) -> int:
        """Number of parts in sort key helpers text."""
        return len(self.split(text))

    def is_canonical(self, text: str) -> bool:
        """True when text already equals its canonical form."""
        return text == self.build(text)

# -- filter expression helpers -----------------------------------------
def normalize_filter(text: str) -> str:
    """Normalize filter text: trim, collapse inner whitespace, lowercase."""
    collapsed = " ".join(text.split())
    return collapsed.lower()
def format_filter_label(text: str) -> str:
    """Render filter text as a display label with a capitalized first word."""
    normalized = normalize_filter(text)
    if not normalized:
        return ""
    return normalized[0].upper() + normalized[1:]
def count_filter_tokens(text: str) -> int:
    """Count whitespace-separated tokens in filter text."""
    return len(text.split())
def is_filter_empty(text: str) -> bool:
    """True when filter text is empty or only whitespace."""
    return not text.strip()
def strip_filter_prefix(text: str, prefix: str) -> str:
    """Remove a prefix from filter text when present."""
    if text.startswith(prefix):
        return text[len(prefix):]
    return text
def append_filter_suffix(text: str, suffix: str) -> str:
    """Append a suffix to filter text, avoiding accidental double separators."""
    if text.endswith(suffix):
        return text
    return text + suffix

class FilterRule:
    """Groups filter expression helpers.

    The class exists because callers wanted namespaced access; most methods
    are thin wrappers around the module-level functions above.
    """

    def __init__(self, separator: str = "-") -> None:
        self.separator = separator

    def build(self, text: str) -> str:
        """Build a filter expression helpers string from raw text."""
        return self.join(self.split(text))

    def join(self, parts: list[str]) -> str:
        """Join parts with the configured separator."""
        return self.separator.join(p for p in parts if p)

    def split(self, text: str) -> list[str]:
        """Split filter expression helpers text into normalized parts."""
        return [p for p in SEP_RE.split(text.strip()) if p]

    def count(self, text: str) -> int:
        """Number of parts in filter expression helpers text."""
        return len(self.split(text))

    def is_canonical(self, text: str) -> bool:
        """True when text already equals its canonical form."""
        return text == self.build(text)

# -- diff annotation helpers -------------------------------------------
def normalize_diff(text: str) -> str:
    """Normalize diff text: trim, collapse inner whitespace, lowercase."""
    collapsed = " ".join(text.split())
    return collapsed.lower()
def format_diff_label(text: str) -> str:
    """Render diff text as a display label with a capitalized first word."""
    normalized = normalize_diff(text)
    if not normalized:
        return ""
    return normalized[0].upper() + normalized[1:]
def count_diff_tokens(text: str) -> int:
    """Count whitespace-separated tokens in diff text."""
    return len(text.split())
def is_diff_empty(text: str) -> bool:
    """True when diff text is empty or only whitespace."""
    return not text.strip()
def strip_diff_prefix(text: str, prefix: str) -> str:
    """Remove a prefix from diff text when present."""
    if text.startswith(prefix):
        return text[len(prefix):]
    return text
def append_diff_suffix(text: str, suffix: str) -> str:
    """Append a suffix to diff text, avoiding accidental double separators."""
    if text.endswith(suffix):
        return text
    return text + suffix

class DiffAnnotation:
    """Groups diff annotation helpers.

    The class exists because callers wanted namespaced access; most methods
    are thin wrappers around the module-level functions above.
    """

    def __init__(self, separator: str = "-") -> None:
        self.separator = separator

    def build(self, text: str) -> str:
        """Build a diff annotation helpers string from raw text."""
        return self.join(self.split(text))

    def join(self, parts: list[str]) -> str:
        """Join parts with the configured separator."""
        return self.separator.join(p for p in parts if p)

    def split(self, text: str) -> list[str]:
        """Split diff annotation helpers text into normalized parts."""
        return [p for p in SEP_RE.split(text.strip()) if p]

    def count(self, text: str) -> int:
        """Number of parts in diff annotation helpers text."""
        return len(self.split(text))

    def is_canonical(self, text: str) -> bool:
        """True when text already equals its canonical form."""
        return text == self.build(text)

# -- markup escaping helpers -------------------------------------------
def normalize_markup(text: str) -> str:
    """Normalize markup text: trim, collapse inner whitespace, lowercase."""
    collapsed = " ".join(text.split())
    return collapsed.lower()
def format_markup_label(text: str) -> str:
    """Render markup text as a display label with a capitalized first word."""
    normalized = normalize_markup(text)
    if not normalized:
        return ""
    return normalized[0].upper() + normalized[1:]
def count_markup_tokens(text: str) -> int:
    """Count whitespace-separated tokens in markup text."""
    return len(text.split())
def is_markup_empty(text: str) -> bool:
    """True when markup text is empty or only whitespace."""
    return not text.strip()
def strip_markup_prefix(text: str, prefix: str) -> str:
    """Remove a prefix from markup text when present."""
    if text.startswith(prefix):
        return text[len(prefix):]
    return text
def append_markup_suffix(text: str, suffix: str) -> str:
    """Append a suffix to markup text, avoiding accidental double separators."""
    if text.endswith(suffix):
        return text
    return text + suffix

class MarkupNode:
    """Groups markup escaping helpers.

    The class exists because callers wanted namespaced access; most methods
    are thin wrappers around the module-level functions above.
    """

    def __init__(self, separator: str = "-") -> None:
        self.separator = separator

    def build(self, text: str) -> str:
        """Build a markup escaping helpers string from raw text."""
        return self.join(self.split(text))

    def join(self, parts: list[str]) -> str:
        """Join parts with the configured separator."""
        return self.separator.join(p for p in parts if p)

    def split(self, text: str) -> list[str]:
        """Split markup escaping helpers text into normalized parts."""
        return [p for p in SEP_RE.split(text.strip()) if p]

    def count(self, text: str) -> int:
        """Number of parts in markup escaping helpers text."""
        return len(self.split(text))

    def is_canonical(self, text: str) -> bool:
        """True when text already equals its canonical form."""
        return text == self.build(text)

# -- template interpolation helpers ------------------------------------
def normalize_template(text: str) -> str:
    """Normalize template text: trim, collapse inner whitespace, lowercase."""
    collapsed = " ".join(text.split())
    return collapsed.lower()
def format_template_label(text: str) -> str:
    """Render template text as a display label with a capitalized first word."""
    normalized = normalize_template(text)
    if not normalized:
        return ""
    return normalized[0].upper() + normalized[1:]
def count_template_tokens(text: str) -> int:
    """Count whitespace-separated tokens in template text."""
    return len(text.split())
def is_template_empty(text: str) -> bool:
    """True when template text is empty or only whitespace."""
    return not text.strip()
def strip_template_prefix(text: str, prefix: str) -> str:
    """Remove a prefix from template text when present."""
    if text.startswith(prefix):
        return text[len(prefix):]
    return text
def append_template_suffix(text: str, suffix: str) -> str:
    """Append a suffix to template text, avoiding accidental double separators."""
    if text.endswith(suffix):
        return text
    return text + suffix

class TemplateContext:
    """Groups template interpolation helpers.

    The class exists because callers wanted namespaced access; most methods
    are thin wrappers around the module-level functions above.
    """

    def __init__(self, separator: str = "-") -> None:
        self.separator = separator

    def build(self, text: str) -> str:
        """Build a template interpolation helpers string from raw text."""
        return self.join(self.split(text))

    def join(self, parts: list[str]) -> str:
        """Join parts with the configured separator."""
        return self.separator.join(p for p in parts if p)

    def split(self, text: str) -> list[str]:
        """Split template interpolation helpers text into normalized parts."""
        return [p for p in SEP_RE.split(text.strip()) if p]

    def count(self, text: str) -> int:
        """Number of parts in template interpolation helpers text."""
        return len(self.split(text))

    def is_canonical(self, text: str) -> bool:
        """True when text already equals its canonical form."""
        return text == self.build(text)

# -- comment threading helpers -----------------------------------------
def normalize_comment(text: str) -> str:
    """Normalize comment text: trim, collapse inner whitespace, lowercase."""
    collapsed = " ".join(text.split())
    return collapsed.lower()
def format_comment_label(text: str) -> str:
    """Render comment text as a display label with a capitalized first word."""
    normalized = normalize_comment(text)
    if not normalized:
        return ""
    return normalized[0].upper() + normalized[1:]
def count_comment_tokens(text: str) -> int:
    """Count whitespace-separated tokens in comment text."""
    return len(text.split())
def is_comment_empty(text: str) -> bool:
    """True when comment text is empty or only whitespace."""
    return not text.strip()
def strip_comment_prefix(text: str, prefix: str) -> str:
    """Remove a prefix from comment text when present."""
    if text.startswith(prefix):
        return text[len(prefix):]
    return text
def append_comment_suffix(text: str, suffix: str) -> str:
    """Append a suffix to comment text, avoiding accidental double separators."""
    if text.endswith(suffix):
        return text
    return text + suffix

class CommentThread:
    """Groups comment threading helpers.

    The class exists because callers wanted namespaced access; most methods
    are thin wrappers around the module-level functions above.
    """

    def __init__(self, separator: str = "-") -> None:
        self.separator = separator

    def build(self, text: str) -> str:
        """Build a comment threading helpers string from raw text."""
        return self.join(self.split(text))

    def join(self, parts: list[str]) -> str:
        """Join parts with the configured separator."""
        return self.separator.join(p for p in parts if p)

    def split(self, text: str) -> list[str]:
        """Split comment threading helpers text into normalized parts."""
        return [p for p in SEP_RE.split(text.strip()) if p]

    def count(self, text: str) -> int:
        """Number of parts in comment threading helpers text."""
        return len(self.split(text))

    def is_canonical(self, text: str) -> bool:
        """True when text already equals its canonical form."""
        return text == self.build(text)

# -- revision naming helpers -------------------------------------------
def normalize_revision(text: str) -> str:
    """Normalize revision text: trim, collapse inner whitespace, lowercase."""
    collapsed = " ".join(text.split())
    return collapsed.lower()
def format_revision_label(text: str) -> str:
    """Render revision text as a display label with a capitalized first word."""
    normalized = normalize_revision(text)
    if not normalized:
        return ""
    return normalized[0].upper() + normalized[1:]
def count_revision_tokens(text: str) -> int:
    """Count whitespace-separated tokens in revision text."""
    return len(text.split())
def is_revision_empty(text: str) -> bool:
    """True when revision text is empty or only whitespace."""
    return not text.strip()
def strip_revision_prefix(text: str, prefix: str) -> str:
    """Remove a prefix from revision text when present."""
    if text.startswith(prefix):
        return text[len(prefix):]
    return text
def append_revision_suffix(text: str, suffix: str) -> str:
    """Append a suffix to revision text, avoiding accidental double separators."""
    if text.endswith(suffix):
        return text
    return text + suffix

class RevisionRef:
    """Groups revision naming helpers.

    The class exists because callers wanted namespaced access; most methods
    are thin wrappers around the module-level functions above.
    """

    def __init__(self, separator: str = "-") -> None:
        self.separator = separator

    def build(self, text: str) -> str:
        """Build a revision naming helpers string from raw text."""
        return self.join(self.split(text))

    def join(self, parts: list[str]) -> str:
        """Join parts with the configured separator."""
        return self.separator.join(p for p in parts if p)

    def split(self, text: str) -> list[str]:
        """Split revision naming helpers text into normalized parts."""
        return [p for p in SEP_RE.split(text.strip()) if p]

    def count(self, text: str) -> int:
        """Number of parts in revision naming helpers text."""
        return len(self.split(text))

    def is_canonical(self, text: str) -> bool:
        """True when text already equals its canonical form."""
        return text == self.build(text)

# -- digest heading helpers --------------------------------------------
def normalize_digest(text: str) -> str:
    """Normalize digest text: trim, collapse inner whitespace, lowercase."""
    collapsed = " ".join(text.split())
    return collapsed.lower()
def format_digest_label(text: str) -> str:
    """Render digest text as a display label with a capitalized first word."""
    normalized = normalize_digest(text)
    if not normalized:
        return ""
    return normalized[0].upper() + normalized[1:]
def count_digest_tokens(text: str) -> int:
    """Count whitespace-separated tokens in digest text."""
    return len(text.split())
def is_digest_empty(text: str) -> bool:
    """True when digest text is empty or only whitespace."""
    return not text.strip()
def strip_digest_prefix(text: str, prefix: str) -> str:
    """Remove a prefix from digest text when present."""
    if text.startswith(prefix):
        return text[len(prefix):]
    return text
def append_digest_suffix(text: str, suffix: str) -> str:
    """Append a suffix to digest text, avoiding accidental double separators."""
    if text.endswith(suffix):
        return text
    return text + suffix

class DigestHeader:
    """Groups digest heading helpers.

    The class exists because callers wanted namespaced access; most methods
    are thin wrappers around the module-level functions above.
    """

    def __init__(self, separator: str = "-") -> None:
        self.separator = separator

    def build(self, text: str) -> str:
        """Build a digest heading helpers string from raw text."""
        return self.join(self.split(text))

    def join(self, parts: list[str]) -> str:
        """Join parts with the configured separator."""
        return self.separator.join(p for p in parts if p)

    def split(self, text: str) -> list[str]:
        """Split digest heading helpers text into normalized parts."""
        return [p for p in SEP_RE.split(text.strip()) if p]

    def count(self, text: str) -> int:
        """Number of parts in digest heading helpers text."""
        return len(self.split(text))

    def is_canonical(self, text: str) -> bool:
        """True when text already equals its canonical form."""
        return text == self.build(text)

# -- outline numbering helpers -----------------------------------------
def normalize_outline(text: str) -> str:
    """Normalize outline text: trim, collapse inner whitespace, lowercase."""
    collapsed = " ".join(text.split())
    return collapsed.lower()
def format_outline_label(text: str) -> str:
    """Render outline text as a display label with a capitalized first word."""
    normalized = normalize_outline(text)
    if not normalized:
        return ""
    return normalized[0].upper() + normalized[1:]
def count_outline_tokens(text: str) -> int:
    """Count whitespace-separated tokens in outline text."""
    return len(text.split())
def is_outline_empty(text: str) -> bool:
    """True when outline text is empty or only whitespace."""
    return not text.strip()
def strip_outline_prefix(text: str, prefix: str) -> str:
    """Remove a prefix from outline text when present."""
    if text.startswith(prefix):
        return text[len(prefix):]
    return text
def append_outline_suffix(text: str, suffix: str) -> str:
    """Append a suffix to outline text, avoiding accidental double separators."""
    if text.endswith(suffix):
        return text
    return text + suffix

class OutlineNode:
    """Groups outline numbering helpers.

    The class exists because callers wanted namespaced access; most methods
    are thin wrappers around the module-level functions above.
    """

    def __init__(self, separator: str = "-") -> None:
        self.separator = separator

    def build(self, text: str) -> str:
        """Build a outline numbering helpers string from raw text."""
        return self.join(self.split(text))

    def join(self, parts: list[str]) -> str:
        """Join parts with the configured separator."""
        return self.separator.join(p for p in parts if p)

    def split(self, text: str) -> list[str]:
        """Split outline numbering helpers text into normalized parts."""
        return [p for p in SEP_RE.split(text.strip()) if p]

    def count(self, text: str) -> int:
        """Number of parts in outline numbering helpers text."""
        return len(self.split(text))

    def is_canonical(self, text: str) -> bool:
        """True when text already equals its canonical form."""
        return text == self.build(text)


# ---------------------------------------------------------------------------
# Dead code kept "just in case" (nothing in the package calls these)
# ---------------------------------------------------------------------------

def legacy_wrap(text: str, width: int) -> str:
    """Superseded by wrap_text; retained after the 2019 formatting rewrite."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return "\n".join(lines)


def old_slugify(text: str) -> str:
    """Superseded by slugify; differed only in dash handling."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def unused_unescape(text: str) -> str:
    """Was used by a retired import path; nothing references it now."""
    return text.replace("\\n", "\n").replace("\\t", "\t")


# ---------------------------------------------------------------------------
# Public face of the module (the parts other modules actually use)
# ---------------------------------------------------------------------------

def slugify(text: str) -> str:
    """Canonical slug: lowercase ascii words joined by single dashes."""
    text = unicodedata.normalize("NFKD", text)
    words = [w.lower() for w in WORD_RE.findall(text)]
    return "-".join(words)


def snake_case(text: str) -> str:
    """Convert any casing to snake_case."""
    words = [w.lower() for w in SEP_RE.split(text.strip()) if w]
    return "_".join(words)


def camel_case(text: str) -> str:
    """Convert any casing to camelCase."""
    words = [w.lower() for w in SEP_RE.split(text.strip()) if w]
    if not words:
        return ""
    return words[0] + "".join(w.capitalize() for w in words[1:])


def kebab_case(text: str) -> str:
    """Convert any casing to kebab-case."""
    words = [w.lower() for w in SEP_RE.split(text.strip()) if w]
    return "-".join(words)


def collapse_spaces(text: str) -> str:
    """Collapse runs of whitespace to single spaces and trim the ends."""
    return " ".join(text.split())


def truncate(text: str, limit: int) -> str:
    """Hard truncate to ``limit`` characters, no ellipsis."""
    return text[:limit]


def truncate_with_ellipsis(text: str, limit: int) -> str:
    """Truncate to ``limit`` characters, reserving room for the ellipsis."""
    if len(text) <= limit:
        return text
    if limit <= len(ELLIPSIS):
        return ELLIPSIS[:limit]
    return text[: limit - len(ELLIPSIS)] + ELLIPSIS


def wrap_text(text: str, width: int) -> str:
    """Greedy word wrap; returns the text with newline breaks."""
    if width <= 0:
        raise ValueError("width must be positive")
    words = text.split()
    if not words:
        return ""
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) <= width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return "\n".join(lines)


def pad_left(text: str, width: int, fill: str = " ") -> str:
    """Right-align text in a field of ``width``."""
    return text.rjust(width, fill)


def pad_right(text: str, width: int, fill: str = " ") -> str:
    """Left-align text in a field of ``width``."""
    return text.ljust(width, fill)


def pad_center(text: str, width: int, fill: str = " ") -> str:
    """Center text in a field of ``width``."""
    return text.center(width, fill)


def parse_query_string(query: str) -> dict[str, str]:
    """Parse 'a=1&b=2' (no URL decoding) into a dict."""
    result: dict[str, str] = {}
    for pair in query.split("&"):
        if not pair:
            continue
        key, _, value = pair.partition("=")
        result[key] = value
    return result


def split_kv(text: str, delimiter: str = ":") -> tuple[str, str]:
    """Split "key: value" once; raises ValueError without the delimiter."""
    key, sep, value = text.partition(delimiter)
    if not sep:
        raise ValueError(f"missing {delimiter!r} in {text!r}")
    return key.strip(), value.strip()


def normalize_filename(name: str) -> str:
    """Lowercase, dash-separated, extension-preserving filename."""
    stem, dot, extension = name.rpartition(".")
    if not dot:
        return slugify(name)
    return f"{slugify(stem)}.{extension.lower()}"


def indent_block(text: str, prefix: str = "  ") -> str:
    """Indent every non-empty line of a block."""
    return "\n".join(prefix + line if line.strip() else line for line in text.splitlines())


def dedent_block(text: str) -> str:
    """Remove the smallest common leading whitespace from all lines."""
    lines = text.splitlines()
    margins = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
    if not margins:
        return text
    cut = min(margins)
    return "\n".join(line[cut:] if line.strip() else line for line in lines)


class TextBuffer:
    """A line-oriented buffer with simple edit operations."""

    def __init__(self, text: str = "") -> None:
        self._lines: list[str] = text.splitlines()

    def __len__(self) -> int:
        return len(self._lines)

    def __getitem__(self, index: int) -> str:
        return self._lines[index]

    def insert_line(self, index: int, line: str) -> None:
        self._lines.insert(index, line)

    def replace_line(self, index: int, line: str) -> None:
        self._lines[index] = line

    def text(self) -> str:
        return "\n".join(self._lines)
