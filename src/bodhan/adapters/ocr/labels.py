"""Deterministic mapping of the official 37-class layout-label taxonomy.

Source: Bodhan API reference -> Document OCR API -> "Layout labels".
Every documented label has exactly one target :class:`BlockKind`; unknown or
unmapped labels resolve to ``UNKNOWN``, which is what the structure stage
hands to the LLM for ambiguous-case classification.
"""

from __future__ import annotations

from bodhan.domain.models import BlockKind

#: All 37 labels of the documented taxonomy.
DOCUMENTED_LABELS: frozenset[str] = frozenset(
    {
        "Advertisement",
        "Answer",
        "Author",
        "Chapter-end-section",
        "Chapter-title",
        "Chart",
        "Code",
        "Contact-info",
        "Dateline",
        "Diagram",
        "Equation",
        "Expression",
        "Flag",
        "Folio",
        "Footer",
        "Footnote",
        "Header",
        "Image",
        "Image-caption",
        "Index",
        "Infobox",
        "List",
        "MCQ",
        "Page-number",
        "Paragraph",
        "Placeholder-text",
        "Question",
        "Reference",
        "Section-title",
        "Solved-example",
        "Sub-section-title",
        "Sub-sub-section-title",
        "Table",
        "Table-caption",
        "Table-of-contents",
        "Title",
        "Website-link",
    }
)

#: The fixed label -> BlockKind mapping (deterministic, no LLM involved).
LABEL_TO_KIND: dict[str, BlockKind] = {
    # Headings
    "Chapter-title": BlockKind.HEADING,
    "Chapter-end-section": BlockKind.HEADING,
    "Title": BlockKind.HEADING,
    "Section-title": BlockKind.HEADING,
    "Sub-section-title": BlockKind.HEADING,
    "Sub-sub-section-title": BlockKind.HEADING,
    # Prose
    "Paragraph": BlockKind.PARAGRAPH,
    "Answer": BlockKind.PARAGRAPH,
    "Question": BlockKind.PARAGRAPH,
    "Solved-example": BlockKind.PARAGRAPH,
    "Reference": BlockKind.PARAGRAPH,
    "Dateline": BlockKind.PARAGRAPH,
    "Contact-info": BlockKind.PARAGRAPH,
    "Website-link": BlockKind.PARAGRAPH,
    "Code": BlockKind.PARAGRAPH,
    "Flag": BlockKind.PARAGRAPH,
    "Infobox": BlockKind.PARAGRAPH,
    "MCQ": BlockKind.PARAGRAPH,
    "Author": BlockKind.PARAGRAPH,
    "Footnote": BlockKind.PARAGRAPH,
    # Lists
    "List": BlockKind.LIST_ITEM,
    "Table-of-contents": BlockKind.LIST_ITEM,
    "Index": BlockKind.LIST_ITEM,
    # Math / tables / captions
    "Equation": BlockKind.EQUATION,
    "Expression": BlockKind.EQUATION,
    "Table": BlockKind.TABLE,
    "Table-caption": BlockKind.FIGURE_CAPTION,
    "Image-caption": BlockKind.FIGURE_CAPTION,
    # Pictorial regions (detected, usually transcribed as "")
    "Image": BlockKind.FIGURE,
    "Chart": BlockKind.FIGURE,
    "Diagram": BlockKind.FIGURE,
    # Page furniture
    "Header": BlockKind.PAGE_HEADER,
    "Footer": BlockKind.PAGE_FOOTER,
    "Folio": BlockKind.PAGE_FOOTER,
    "Page-number": BlockKind.PAGE_FOOTER,
    # Deliberately unmapped (judged by quality/structure stages instead)
    "Advertisement": BlockKind.UNKNOWN,
    "Placeholder-text": BlockKind.UNKNOWN,
}

assert set(LABEL_TO_KIND) == DOCUMENTED_LABELS, "label map must cover all 37 labels"


def map_label(label: str | None) -> BlockKind:
    """Map a provider layout label to a :class:`BlockKind` (never raises)."""
    if not label:
        return BlockKind.UNKNOWN
    return LABEL_TO_KIND.get(label.strip(), BlockKind.UNKNOWN)
