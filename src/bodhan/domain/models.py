"""Domain models for the BODHAN LESSON AGENT pipeline.

Design rules:
- Pure data only: pydantic models and enums, no filesystem/network access.
- Every OCR-derived structure carries its provenance (page index, bbox,
  confidence) so low-quality output can be flagged instead of silently dropped.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class StageName(StrEnum):
    """Ordered pipeline stages."""

    INGEST = "ingest"
    PREPROCESS = "preprocess"
    EXTRACT = "extract"
    STRUCTURE = "structure"
    QUALITY = "quality"
    TRANSLATE = "translate"
    COMPOSE = "compose"
    NARRATE = "narrate"
    EXPORT = "export"


class BlockKind(StrEnum):
    """Structural classification of an OCR block."""

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    EQUATION = "equation"
    FIGURE = "figure"
    FIGURE_CAPTION = "figure_caption"
    PAGE_HEADER = "page_header"
    PAGE_FOOTER = "page_footer"
    UNKNOWN = "unknown"


class QualityFlag(StrEnum):
    """Reason a block was flagged for review."""

    LOW_CONFIDENCE = "low_confidence"
    SUSPECTED_GARBLED = "suspected_garbled"
    READING_ORDER_AMBIGUOUS = "reading_order_ambiguous"
    UNTRANSLATABLE = "untranslatable"


class OCRBlock(BaseModel):
    """A single block of extracted text with layout provenance."""

    block_id: str
    page_index: int = Field(ge=0)
    text: str
    kind: BlockKind = BlockKind.UNKNOWN
    reading_order: int = Field(ge=0, default=0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    bbox: tuple[float, float, float, float] | None = None
    flags: list[QualityFlag] = Field(default_factory=list)
    # Raw provider metadata preserved verbatim (documented OCR fields).
    provider_label: str | None = None
    provider_type: str | None = None

    @property
    def is_low_confidence(self) -> bool:
        return self.confidence is not None and self.confidence < 0.85

    def flag(self, flag: QualityFlag) -> None:
        if flag not in self.flags:
            self.flags.append(flag)


class OCRResult(BaseModel):
    """Extraction output for one page, in reading order."""

    page_index: int = Field(ge=0)
    blocks: list[OCRBlock] = Field(default_factory=list)
    provider: str
    provider_request_id: str | None = None
    # --- preserved raw metadata (documented response fields) ---------------
    page_markdown: str | None = None  # choices[0].message.content
    raw_blocks: list[dict[str, Any]] = Field(default_factory=list)  # blocks[]
    usage: dict[str, int] = Field(default_factory=dict)  # token accounting
    rate_limit: dict[str, int] = Field(default_factory=dict)  # published limits

    @property
    def flagged_blocks(self) -> list[OCRBlock]:
        return [b for b in self.blocks if b.flags]


class Page(BaseModel):
    """One ingested page of a chapter."""

    page_index: int = Field(ge=0)
    source_path: str
    image_path: str | None = None
    width: int | None = None
    height: int | None = None


class Chapter(BaseModel):
    """A single textbook chapter, the unit of work for the pipeline."""

    chapter_id: str
    title: str
    source_path: str
    pages: list[Page] = Field(default_factory=list)
    language_code: str = "unknown"
    target_language_code: str = "unknown"


class Lesson(BaseModel):
    """Composed, ready-to-narrate lesson."""

    lesson_id: str
    chapter_id: str
    title: str
    markdown: str
    source_language_code: str
    target_language_code: str
    flagged_block_ids: list[str] = Field(default_factory=list)


class AudioSegment(BaseModel):
    """One narrated audio file (a lesson section or block run)."""

    segment_id: str
    lesson_id: str
    text: str
    audio_path: str
    duration_ms: int | None = None
