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
    TRANSCRIBE = "transcribe"
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
    FIGURE_ASSOCIATION_AMBIGUOUS = "figure_association_ambiguous"


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
    # Provenance of blocks merged by document-level continuity analysis.
    # Empty for ordinary OCR blocks (fully backward compatible).
    source_block_ids: list[str] = Field(default_factory=list)
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
    # Optional user-provided glossary seeds (English terms the lesson must
    # cover) - supplied by the assignment CLI's --glossary option and handed
    # to the lesson planner as required glossary entries.
    glossary_seeds: list[dict[str, str]] = Field(default_factory=list)


class ScaffoldItem(BaseModel):
    """One non-block lesson text unit with full translation provenance.

    Every item is authored in English (by the LLM planner or as a fixed UI
    label) and translated by Bodhan Indic-Translate. ``source_en`` -> ``text``
    plus ``provider``/``usage`` make the target-language string traceable to
    the translation call that produced it.
    """

    item_id: str
    kind: str  # title|label|section_heading|objective|summary|glossary_*|exercise_*|narration
    source_en: str
    text: str
    provider: str = "bodhan-translate"
    usage: dict[str, int] = Field(default_factory=dict)
    related_block_ids: list[str] = Field(default_factory=list)


class Lesson(BaseModel):
    """Composed, ready-to-narrate lesson."""

    lesson_id: str
    chapter_id: str
    title: str
    markdown: str
    source_language_code: str
    target_language_code: str
    flagged_block_ids: list[str] = Field(default_factory=list)
    scaffold: list[ScaffoldItem] = Field(default_factory=list)

    def label(self, key: str, default: str = "") -> str:
        """Target-language UI label (translated via Indic-Translate).

        Returns ``default`` when the label was never translated (legacy
        lessons composed before the scaffold existed).
        """
        item_id = f"label:{key}"
        for item in self.scaffold:
            if item.kind == "label" and item.item_id == item_id:
                return item.text
        return default


class AudioSegment(BaseModel):
    """One narrated audio file (a lesson section or block run)."""

    segment_id: str
    lesson_id: str
    text: str
    audio_path: str
    duration_ms: int | None = None
