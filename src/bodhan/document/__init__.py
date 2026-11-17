"""Deterministic document-understanding helpers (Phase 3B-1).

Pure functions only: no provider calls, no LLM, no file I/O. All decisions are
traceable to OCR block ids and recorded by the structure stage.
"""

from bodhan.document.continuity import analyze_continuity
from bodhan.document.equations import protect_math, restore_math, validate_placeholders
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
    "analyze_continuity",
    "analyze_suppression",
    "associate_figures",
    "parse_html_tables",
    "protect_math",
    "render_table_markdown",
    "restore_math",
    "validate_placeholders",
]
