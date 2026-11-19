"""Deterministic document-understanding helpers (Phase 3B-1).

Pure functions only: no provider calls, no LLM, no file I/O. All decisions are
traceable to OCR block ids and recorded by the structure stage.
"""

from bodhan.document.continuity import analyze_continuity
from bodhan.document.equations import (
    TextSegment,
    preserve_math_only,
    prose_to_translate,
    protect_math,
    reassemble,
    restore_math,
    segment_mixed_text,
    validate_math_spans,
    validate_placeholders,
)
from bodhan.document.figures import associate_figures
from bodhan.document.models import (
    ContinuityDecision,
    FigureAssociation,
    SuppressionDecision,
    TableCell,
    TableGrid,
)
from bodhan.document.suppression import analyze_suppression
from bodhan.document.tables import parse_html_tables, render_table_markdown

__all__ = [
    "ContinuityDecision",
    "FigureAssociation",
    "SuppressionDecision",
    "TableCell",
    "TableGrid",
    "TextSegment",
    "analyze_continuity",
    "analyze_suppression",
    "associate_figures",
    "parse_html_tables",
    "preserve_math_only",
    "prose_to_translate",
    "protect_math",
    "reassemble",
    "render_table_markdown",
    "restore_math",
    "segment_mixed_text",
    "validate_math_spans",
    "validate_placeholders",
]
