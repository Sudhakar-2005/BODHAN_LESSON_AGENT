"""Deterministic HTML-table structure tests (Phase 3B-1 Task 2)."""

from __future__ import annotations

from bodhan.document import parse_html_tables, render_table_markdown
from bodhan.document.tables import merge_tables
from bodhan.domain.models import BlockKind, OCRBlock


def table_block(text: str, block_id: str = "p0-b0") -> OCRBlock:
    return OCRBlock(
        block_id=block_id,
        page_index=0,
        reading_order=0,
        text=text,
        kind=BlockKind.TABLE,
        provider_label="Table",
    )


def test_simple_two_column_table() -> None:
    grid = parse_html_tables(
        table_block(
            "<table><tr><th>Name</th><th>Value</th></tr><tr><td>Angle</td><td>60</td></tr></table>"
        )
    )
    assert grid is not None
    assert (grid.rows, grid.columns) == (2, 2)
    assert [(c.row, c.column, c.text, c.is_header) for c in grid.cells] == [
        (0, 0, "Name", True),
        (0, 1, "Value", True),
        (1, 0, "Angle", False),
        (1, 1, "60", False),
    ]
    assert grid.incomplete is False


def test_multi_column_table_with_numeric_cells() -> None:
    grid = parse_html_tables(
        table_block(
            "<table><tr><th>A</th><th>B</th><th>C</th></tr>"
            "<tr><td>1</td><td>2.5</td><td>3</td></tr></table>"
        )
    )
    assert grid is not None
    assert grid.columns == 3
    assert [cell.text for cell in grid.row_cells(1)] == ["1", "2.5", "3"]


def test_multiline_cells_empty_cells_and_math() -> None:
    grid = parse_html_tables(
        table_block(
            "<table><tr><th>Term</th><th>Formula</th></tr>"
            "<tr><td>line one<br>line two</td><td>$x^2$</td></tr>"
            "<tr><td></td><td></td></tr></table>"
        )
    )
    assert grid is not None
    assert grid.row_cells(1)[0].text == "line one\nline two"
    assert grid.row_cells(1)[1].text == "$x^2$"
    assert grid.row_cells(2)[0].text == ""


def test_malformed_html_is_flagged_for_review() -> None:
    grid = parse_html_tables(table_block("<table><tr><td>A</td><td>B</td></tr><tr><td>C</td>"))
    assert grid is not None
    assert grid.incomplete is True
    assert grid.review_reason is not None
    assert grid.rows >= 1  # no content lost: what was parsed is preserved


def test_no_table_markup_returns_none() -> None:
    assert parse_html_tables(table_block("just a paragraph")) is None


def test_tables_spanning_pages_merge_with_provenance() -> None:
    first = parse_html_tables(
        table_block(
            "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>", "p0-b0"
        )
    )
    second = parse_html_tables(
        OCRBlock(
            block_id="p1-b0",
            page_index=1,
            reading_order=0,
            text="<table><tr><td>3</td><td>4</td></tr></table>",
            kind=BlockKind.TABLE,
        )
    )
    assert first is not None and second is not None
    merged = merge_tables(first, second)
    assert merged.rows == 3
    assert [cell.row for cell in merged.cells] == [0, 0, 1, 1, 2, 2]
    assert merged.source_block_ids == ["p0-b0", "p1-b0"]


def test_markdown_rendering_is_deterministic_and_preserves_translations() -> None:
    grid = parse_html_tables(
        table_block("<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>")
    )
    assert grid is not None
    first = render_table_markdown(grid, {(1, 0): "एक", (1, 1): "दो"})
    second = render_table_markdown(grid, {(1, 0): "एक", (1, 1): "दो"})
    assert first == second
    assert "| A | B |" in first
    assert "| एक | दो |" in first
    assert "<!-- table needs review" not in first


def test_incomplete_table_is_marked_in_markdown() -> None:
    grid = parse_html_tables(table_block("<table><tr><td>A</td><td>B</td></tr><tr><td>C</td>"))
    assert grid is not None and grid.incomplete
    rendered = render_table_markdown(grid, {})
    assert "needs review" in rendered
