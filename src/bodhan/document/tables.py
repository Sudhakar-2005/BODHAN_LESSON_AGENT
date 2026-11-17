"""HTML table → structured grid → deterministic Markdown (Phase 3B-1 Task 2).

The OCR provider is queried with ``table_format=html``. This module never
rewrites content: it preserves cell text, row/column indices and provenance,
marks incomplete markup for review, and renders Markdown deterministically
after Indic-Translate has translated individual cells.
"""

from __future__ import annotations

from html.parser import HTMLParser
from typing import Any

from bodhan.document.models import TableCell, TableGrid
from bodhan.domain.models import OCRBlock


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[dict[str, Any]]] = []
        self.current_row: list[dict[str, Any]] | None = None
        self.current_cell: dict[str, Any] | None = None
        self.in_table = False
        self.saw_table = False
        self.malformed = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self.in_table = True
            self.saw_table = True
        elif tag == "tr" and self.in_table:
            self.current_row = []
        elif tag in {"td", "th"} and self.in_table:
            if self.current_row is None:
                self.current_row = []
                self.malformed = True
            self.current_cell = {"text": [], "header": tag == "th"}
        elif tag == "br" and self.current_cell is not None:
            self.current_cell["text"].append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self.current_cell is not None:
            assert self.current_row is not None
            self.current_row.append(
                {
                    "text": "".join(self.current_cell["text"]).strip(),
                    "header": self.current_cell["header"],
                }
            )
            self.current_cell = None
        elif tag == "tr" and self.current_row is not None:
            self.rows.append(self.current_row)
            self.current_row = None
        elif tag == "table":
            if self.current_cell is not None and self.current_row is not None:
                self.current_row.append(
                    {
                        "text": "".join(self.current_cell["text"]).strip(),
                        "header": self.current_cell["header"],
                    }
                )
            if self.current_row is not None:
                self.rows.append(self.current_row)
            self.in_table = False

    def handle_data(self, data: str) -> None:
        if self.current_cell is not None:
            self.current_cell["text"].append(data)


def parse_html_tables(block: OCRBlock) -> TableGrid | None:
    """Parse one OCR table block; ``None`` means there is no table markup."""
    if "<table" not in block.text.lower():
        return None
    parser = _TableParser()
    try:
        parser.feed(block.text)
        parser.close()
    except Exception:  # HTMLParser is lenient; never crash the pipeline
        return TableGrid(
            block_id=block.block_id,
            page_index=block.page_index,
            rows=0,
            columns=0,
            incomplete=True,
            review_reason="table markup could not be parsed",
            source_block_ids=[block.block_id],
        )
    if not parser.saw_table or not parser.rows:
        return TableGrid(
            block_id=block.block_id,
            page_index=block.page_index,
            rows=0,
            columns=0,
            incomplete=True,
            review_reason="no table rows found in OCR markup",
            source_block_ids=[block.block_id],
        )

    cells: list[TableCell] = []
    widths = {len(row) for row in parser.rows}
    for row_index, row in enumerate(parser.rows):
        for column, raw in enumerate(row):
            cells.append(
                TableCell(
                    row=row_index,
                    column=column,
                    text=str(raw["text"]),
                    is_header=bool(raw["header"]),
                    source_block_id=block.block_id,
                )
            )
    incomplete = parser.malformed or parser.in_table or len(widths) > 1
    reason = None
    if parser.malformed:
        reason = "malformed HTML table markup"
    elif parser.in_table:
        reason = "unclosed table element"
    elif len(widths) > 1:
        reason = f"inconsistent column counts across rows: {sorted(widths)}"
    return TableGrid(
        block_id=block.block_id,
        page_index=block.page_index,
        rows=len(parser.rows),
        columns=max(widths, default=0),
        cells=cells,
        incomplete=incomplete,
        review_reason=reason,
        source_block_ids=[block.block_id],
    )


def merge_tables(first: TableGrid, second: TableGrid) -> TableGrid:
    """Merge a table continued onto the next page (deterministically)."""
    offset = first.rows
    cells = list(first.cells) + [
        cell.model_copy(update={"row": cell.row + offset}) for cell in second.cells
    ]
    return TableGrid(
        block_id=first.block_id,
        page_index=first.page_index,
        rows=first.rows + second.rows,
        columns=max(first.columns, second.columns),
        cells=cells,
        incomplete=first.incomplete or second.incomplete,
        review_reason=first.review_reason or second.review_reason,
        source_block_ids=first.source_block_ids + second.source_block_ids,
    )


def _escape(text: str) -> str:
    return " ".join(text.replace("|", "\\|").split()).replace("\n", "<br>")


def render_table_markdown(grid: TableGrid, translations: dict[tuple[int, int], str]) -> str:
    """Render a readable Markdown table from translated cells."""
    if not grid.rows or not grid.cells:
        note = grid.review_reason or "no table structure extracted"
        return f"_<!-- table needs review: {note} -->_"
    header_row = 0
    for cell in grid.cells:
        if cell.is_header:
            header_row = cell.row
            break

    def value(cell: TableCell) -> str:
        return _escape(translations.get((cell.row, cell.column), cell.text))

    lines = [
        "| " + " | ".join(value(cell) for cell in grid.row_cells(header_row)) + " |",
        "| " + " | ".join("---" for _ in grid.row_cells(header_row)) + " |",
    ]
    for row in range(grid.rows):
        if row == header_row:
            continue
        cells = grid.row_cells(row)
        if cells:
            lines.append("| " + " | ".join(value(cell) for cell in cells) + " |")
    if grid.incomplete:
        lines.append(f"<!-- table needs review: {grid.review_reason or 'incomplete table'} -->")
    return "\n".join(lines)
