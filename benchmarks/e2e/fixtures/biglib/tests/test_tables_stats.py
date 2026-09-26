from biglib.stats import mean, median, summarize, variance
from biglib.tables import banner, column_totals, money_row, render_table


def test_stats_basics():
    assert mean([1, 2, 3, 4]) == 2.5
    assert median([3, 1, 2]) == 2
    assert median([4, 1, 2, 3]) == 2.5
    assert variance([0, 0, 10, 10]) == 25.0


def test_stats_summarize_clamps_precision():
    summary = summarize([1, 2, 3, 4], precision=9)
    assert summary == {"mean": 2.5, "median": 2.5, "variance": 1.25}


def test_render_table():
    table = render_table(["id", "name"], [["1", "alpha"], ["22", "b"]])
    lines = [line.rstrip() for line in table.splitlines()]
    assert lines[0] == "id | name"
    assert lines[1] == "---+------"
    assert lines[2] == "1  | alpha"
    assert lines[3] == "22 | b"


def test_banner_and_totals():
    assert banner("hi", 6).splitlines()[1] == "  hi  "
    assert column_totals([[1, 2], [3, 4], [5]]) == [9, 6]
    assert money_row(1234, 8) == "   12.34"
