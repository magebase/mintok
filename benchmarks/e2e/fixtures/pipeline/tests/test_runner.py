import pytest

from pipeline.errors import ConfigurationError, ExhaustedRetries, SkipRecord
from pipeline.readers import MemoryReader
from pipeline.registry import create, list_names, register, reset_registry
from pipeline.retry import RetryPolicy, attempt
from pipeline.runner import Pipeline
from pipeline.sinks import CountingSink, FileSink, ListSink
from pipeline.transforms import drop_empty, prefix_key, uppercase_value
from pipeline.validate import require_key


@pytest.fixture(autouse=True)
def clean_registry():
    reset_registry()
    yield
    reset_registry()


def test_registry_round_trip():
    from pipeline.readers import MemoryReader

    register("reader", "memory", MemoryReader)
    assert "memory" in list_names("reader")
    reader = create("reader", "memory", pairs=[("a", "1")])
    assert reader.read().keys() == ["a"]
    with pytest.raises(ConfigurationError):
        register("reader", "memory", MemoryReader)
    with pytest.raises(ConfigurationError):
        create("sink", "missing")


def test_registry_rejects_unknown_kind():
    with pytest.raises(ConfigurationError):
        register("widget", "w", lambda: None)


def test_retry_policy_counts_attempts():
    policy = RetryPolicy(max_attempts=3)
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ValueError("boom")
        return "ok"

    assert attempt(flaky, policy) == "ok"
    assert len(calls) == 3


def test_retry_gives_up():
    policy = RetryPolicy(max_attempts=2)
    with pytest.raises(ExhaustedRetries):
        attempt(lambda: (_ for _ in ()).throw(ValueError("no")), policy)


def test_retry_non_matching_error_not_retried():
    policy = RetryPolicy(max_attempts=3, retry_on=(KeyError,))
    calls = []

    def boom():
        calls.append(1)
        raise ValueError("wrong type")

    with pytest.raises(ExhaustedRetries):
        attempt(boom, policy)
    assert len(calls) == 1


def test_pipeline_end_to_end():
    reader = MemoryReader([("a", " alpha "), ("b", ""), ("c", " gamma ")])
    sink = ListSink()
    pipe = Pipeline(reader, sink, transforms=[uppercase_value], rules=[require_key])
    stats = pipe.run()
    assert stats.processed == 3
    assert [r.value for r in sink.records] == [" ALPHA ", "", " GAMMA "]
    assert stats.skipped == 0


def test_pipeline_drops_empty_with_rule():
    reader = MemoryReader([("a", "alpha"), ("b", "  ")])
    sink = ListSink()
    pipe = Pipeline(reader, sink, transforms=[drop_empty()])
    stats = pipe.run()
    assert stats.processed == 1
    assert sink.records[0].key == "a"


def test_counting_sink_and_file_sink(tmp_path):
    from pipeline.readers import CsvReader

    reader = CsvReader("key,value\na,1\nb,2\n")
    sink = FileSink(tmp_path / "out" / "lines.txt")
    stats = Pipeline(reader, sink).run()
    assert stats.processed == 2
    assert sink.path.read_text() == "a=1\nb=2\n"

    counter = CountingSink()
    Pipeline(MemoryReader([("x", "1"), ("y", "2")]), counter).run()
    assert counter.count == 2
