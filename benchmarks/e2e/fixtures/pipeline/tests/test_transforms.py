import pytest

from pipeline.errors import SkipRecord
from pipeline.records import Record
from pipeline.transforms import (
    annotate_meta,
    compose,
    drop_empty,
    prefix_key,
    trim_value,
    uppercase_value,
)


def test_uppercase_and_trim():
    record = Record("a", "  hello  ")
    assert uppercase_value(record).value == "  HELLO  "
    assert trim_value(record).value == "hello"
    assert record.value == "  hello  "  # transforms are pure


def test_prefix_key():
    out = prefix_key("v2:")(Record("a", "x"))
    assert out.key == "v2:a"


def test_drop_empty_raises_skip():
    with pytest.raises(SkipRecord):
        drop_empty()(Record("a", "   "))


def test_annotate_meta_merges():
    out = annotate_meta(source="batch7")(Record("a", "x", meta={"n": 1}))
    assert out.meta == {"n": 1, "source": "batch7"}


def test_compose_runs_left_to_right():
    chain = compose(trim_value, uppercase_value, prefix_key("p-"))
    out = chain(Record("a", "  hi "))
    assert (out.key, out.value) == ("p-a", "HI")
