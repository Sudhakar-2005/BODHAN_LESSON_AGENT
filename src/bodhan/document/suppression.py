"""Deterministic detection of repeated page furniture (Phase 3B-1, Task 1).

Only blocks with *evidence* are suppressed: an explicit provider furniture
label, a standalone page number, or a near-identical string repeated at the
same page position on several pages. A single heading or paragraph near a
page edge is never removed merely because of its position.
"""

from __future__ import annotations

import math
import re
from difflib import SequenceMatcher

from bodhan.document.models import SuppressionDecision
from bodhan.domain.models import BlockKind, OCRBlock

_FURNITURE_LABELS = {"Header", "Footer", "Folio", "Page-number"}
_PAGE_NUMBER = re.compile(r"^[\s\-\u2013\u2014·.|_]*\d{1,4}[\s\-\u2013\u2014·.|_]*$")
_PAGE_WORD_NUMBER = re.compile(
    r"^(?:page|folio|p)\.?\s*[-:]?\s*\d{1,4}[\s\-\u2013\u2014·.]*$", re.IGNORECASE
)
_NON_SEMANTIC = re.compile(r"[\W\d_]+", re.UNICODE)


def _normalised(text: str) -> str:
    """Strip digits/punctuation so 'Chapter 1' and 'Chapter 2' compare equal."""
    return " ".join(_NON_SEMANTIC.sub(" ", text).lower().split())


def _similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right, autojunk=False).ratio()


def _position_bands(blocks: list[OCRBlock]) -> dict[str, str]:
    """Map block id -> 'top'/'bottom'/'middle' using per-page position."""
    per_page: dict[int, list[OCRBlock]] = {}
    for block in blocks:
        per_page.setdefault(block.page_index, []).append(block)
    bands: dict[str, str] = {}
    for page_blocks in per_page.values():
        count = len(page_blocks)
        edge = max(1, math.ceil(count * 0.2))
        for index, block in enumerate(page_blocks):
            if index < edge:
                bands[block.block_id] = "top"
            elif index >= count - edge:
                bands[block.block_id] = "bottom"
            else:
                bands[block.block_id] = "middle"
    return bands


def analyze_suppression(
    blocks: list[OCRBlock], *, min_pages: int = 2, similarity_threshold: float = 0.85
) -> tuple[list[OCRBlock], list[SuppressionDecision]]:
    """Return ``(active_blocks, decisions)`` without mutating the inputs."""
    if min_pages < 2:
        raise ValueError("min_pages must be at least 2 to avoid removing single-page content")

    ordered = sorted(blocks, key=lambda b: (b.page_index, b.reading_order))
    bands = _position_bands(ordered)
    groups: dict[tuple[str, str], list[OCRBlock]] = {}
    for block in ordered:
        key = _normalised(block.text)
        if key:
            groups.setdefault((bands[block.block_id], key), []).append(block)

    decisions: list[SuppressionDecision] = []
    suppressed_ids: set[str] = set()

    for block in ordered:
        label = block.provider_label or ""
        text = block.text.strip()
        band = bands[block.block_id]
        key = _normalised(block.text)
        group = groups.get((band, key), []) if key else []
        pages = sorted({candidate.page_index for candidate in group})
        similarity = max(
            (_similarity(_normalised(block.text), _normalised(c.text)) for c in group),
            default=0.0,
        )
        evidence: dict[str, object] = {
            "provider_label": label,
            "position": band,
            "pages": pages,
            "similarity": round(similarity, 4),
        }
        reason: str | None = None

        if label in _FURNITURE_LABELS:
            reason = f"provider label {label!r} marks page furniture"
        elif _PAGE_NUMBER.fullmatch(text) or _PAGE_WORD_NUMBER.fullmatch(text):
            if block.kind not in {BlockKind.TABLE, BlockKind.EQUATION}:
                reason = "standalone page number with no surrounding content"
        elif len(pages) >= min_pages and band in {"top", "bottom"} and not block.flags:
            # Flagged blocks (e.g. low-confidence OCR) are evidence that the
            # content needs review, not that it is page furniture.
            # A genuine heading is kept unless it repeats on three or more
            # pages (a running chapter title), never on a single page.
            needs = 3 if block.kind is BlockKind.HEADING else min_pages
            if len(pages) >= needs and (similarity >= similarity_threshold or not key):
                kind = "running title" if band == "top" else "repeated footer"
                reason = f"{kind} repeated on {len(pages)} pages (similarity {similarity:.2f})"

        if reason is not None:
            suppressed_ids.add(block.block_id)
            decisions.append(
                SuppressionDecision(
                    block_id=block.block_id,
                    page_index=block.page_index,
                    text=block.text,
                    reason=reason,
                    evidence=evidence,
                )
            )

    active = [block for block in ordered if block.block_id not in suppressed_ids]
    return active, decisions
