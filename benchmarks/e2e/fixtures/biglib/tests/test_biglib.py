import biglib.large as large
from biglib.numbers import clamp, digit_sum, distribute, luhn_like_checksum
from biglib.strings import asset_name, display_name, setting_from_text, slug_pair


def test_large_case_helpers():
    assert large.snake_case("Hello World") == "hello_world"
    assert large.camel_case("hello world") == "helloWorld"
    assert large.kebab_case("Hello World") == "hello-world"
    assert large.slugify("Hello, World! 2024") == "hello-world-2024"


def test_large_whitespace_and_truncation():
    assert large.collapse_spaces("  a   b  ") == "a b"
    assert large.truncate("abcdef", 3) == "abc"
    assert large.truncate_with_ellipsis("abcdef", 5) == "ab..."
    assert large.truncate_with_ellipsis("abc", 5) == "abc"


def test_large_wrap_and_padding():
    assert large.wrap_text("one two three four", 7) == "one two\nthree\nfour"
    assert large.pad_left("7", 3) == "  7"
    assert large.pad_right("7", 3, ".") == "7.."
    assert large.pad_center("hi", 6, "*") == "**hi**"


def test_large_parsers():
    assert large.parse_query_string("a=1&b=2") == {"a": "1", "b": "2"}
    assert large.split_kv("host: example") == ("host", "example")
    import pytest

    with pytest.raises(ValueError):
        large.split_kv("no-delimiter")


def test_large_filename_normalization():
    assert large.normalize_filename("My Report.PDF") == "my-report.pdf"
    assert large.normalize_filename("README") == "readme"


def test_large_text_buffer():
    buffer = large.TextBuffer("one\ntwo")
    buffer.insert_line(1, "1.5")
    buffer.replace_line(0, "ONE")
    assert buffer.text() == "ONE\n1.5\ntwo"
    assert len(buffer) == 3


def test_large_dead_helpers_still_behave():
    assert large.legacy_wrap("a bb ccc", 3) == "a\nbb\nccc"
    assert large.old_slugify("Hello World") == "hello-world"
    assert large.unused_unescape("a\\nb") == "a\nb"


def test_large_section_functions_exist():
    # Spot-check the generated per-section helpers.
    for name in (
        "normalize_ingest",
        "format_search_label",
        "count_index_tokens",
        "strip_render_prefix",
        "append_export_suffix",
        "normalize_import",
    ):
        assert hasattr(large, name), name


def test_large_section_classes_exist():
    for cls in ("IngestPreview", "SearchSnippet", "IndexKey", "RenderOutput", "TemplateContext"):
        assert hasattr(large, cls), cls
    preview = large.IngestPreview()
    assert preview.build("hello world") == "hello-world"
    assert preview.is_canonical("hello-world")


def test_numbers():
    assert clamp(15, 0, 10) == 10
    assert digit_sum(1234) == 10
    assert luhn_like_checksum(18) == 1  # digits 1,8 -> double the 1 -> sum 10 -> 1
    assert distribute(7, 3) == [3, 2, 2]


def test_strings_module():
    assert display_name("  hello   world ") == "Hello world"
    assert slug_pair("Big Report") == ("big-report", "big-report")
    assert setting_from_text("Max Retries: 4") == ("max_retries", "4")
    assert asset_name("quarterly summary") == "quarterlysummary.txt"
