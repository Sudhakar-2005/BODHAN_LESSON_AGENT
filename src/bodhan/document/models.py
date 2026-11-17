"""Small, explicit data models for document-intelligence decisions."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SuppressionDecision(BaseModel):
    """Why a page-furniture block was excluded from downstream content."""

    block_id: str
    page_index: int
    text: str
    reason: str
    evidence: dict[str, object] = Field(default_factory=dict)


class ContinuityDecision(BaseModel):
    """A deterministic merge of two consecutive blocks across a page boundary."""

    target_block_id: str
    source_block_ids: list[str]
    reason: str
    evidence: dict[str, object] = Field(default_factory=dict)


class FigureAssociation(BaseModel):
    """Caption ↔ figure association based on page and positional evidence."""

    caption_block_id: str
    figure_block_id: str | None
    reason: str
    confidence: float | None = None


class TableCell(BaseModel):
    """One table cell with stable row/column provenance."""

    row: int
    column: int
    text: str
    is_header: bool = False
    source_block_id: str


class TableGrid(BaseModel):
    """A structured table reconstructed from provider HTML."""

    block_id: str
    page_index: int
    rows: int
    columns: int
    cells: list[TableCell] = Field(default_factory=list)
    incomplete: bool = False
    review_reason: str | None = None
    source_block_ids: list[str] = Field(default_factory=list)

    def row_cells(self, row: int) -> list[TableCell]:
        return sorted((cell for cell in self.cells if cell.row == row), key=lambda c: c.column)
