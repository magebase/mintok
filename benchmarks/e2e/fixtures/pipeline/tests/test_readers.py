import pytest

from pipeline.errors import ConfigurationError
from pipeline.readers import CsvReader, JsonlReader, MemoryReader
from pipeline.records import RecordBatch

CSV_TEXT = "key,value,unit\ncolour,red,spectral\nsize,large,\n"
JSONL_TEXT = '{"key": "a", "value": "1"}\n\n{"key": "b", "value": "2", "meta": {"n": 2}}\n'


def test_memory_reader_builds_batch():
    batch = MemoryReader([("a", "1"), ("b", "2")]).read()
    assert isinstance(batch, RecordBatch)
    assert batch.keys() == ["a", "b"]


def test_csv_reader_maps_columns_to_meta():
    batch = CsvReader(CSV_TEXT).read()
    assert batch.keys() == ["colour", "size"]
    first = batch.records[0]
    assert first.value == "red"
    assert first.meta == {"unit": "spectral"}


def test_csv_reader_rejects_duplicate_columns():
    with pytest.raises(ConfigurationError):
        CsvReader("key,key\n1,2").read()


def test_csv_reader_empty_text():
    assert len(CsvReader("").read()) == 0


def test_jsonl_reader_skips_blank_lines():
    batch = JsonlReader(JSONL_TEXT).read()
    assert batch.keys() == ["a", "b"]
    assert batch.records[1].meta == {"n": 2}
