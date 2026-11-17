"""Document-level continuation detection (Phase 3B-1 Task 3).

Pure and deterministic. Only adjacent blocks on consecutive pages are
considered; merges require positive textual/list evidence. Every merge is
recorded with its source block ids, and the original OCR artifacts are never
changed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

from bodhan.document.models import ContinuityDecision
from bodhan.domain.models import BlockKind, OCRBlock, QualityFlag

_TERMINAL = ("।", ".", "?", "!", ":")
_NUMBERED = re.compile(r"^\s*(\d+)\s*[.)]\s+")
_BULLET = re.compile(r"^\s*[-•*]\s+")


@dataclass
class ContinuityResult:
    blocks: list[OCRBlock]
    decisions: list[ContinuityDecision] = field(default_factory=list)
    table_pairs: list[tuple[str, str]] = field(default_factory=list)


def _merge_text(previous: str, following: str) -> str:
    left = previous.rstrip()
    if left.endswith("--"):
        return f"{left} {following.strip()}"
    if left.endswith("-"):
        return f"{left[:-1]}{following.strip()}"
    return f"{left} {following.strip()}"


def _list_marker(text: str) -> tuple[str, int | None] | None:
    numbered = _NUMBERED.match(text)
    if numbered:
        return "numbered", int(numbered.group(1))
    if _BULLET.match(text):
        return "bullet", None
    return None


def _continuation_kind(previous: OCRBlock, following: OCRBlock) -> str | None:
    if following.page_index != previous.page_index + 1:
        return None
    if following.kind in {BlockKind.HEADING, BlockKind.PAGE_HEADER, BlockKind.PAGE_FOOTER}:
        return None
    if previous.kind is BlockKind.TABLE and following.kind is BlockKind.TABLE:
        return "table"
    if previous.kind is BlockKind.LIST_ITEM and following.kind is BlockKind.LIST_ITEM:
        first, second = _list_marker(previous.text), _list_marker(following.text)
        if first and second and first[0] == second[0]:
            if first[0] == "numbered" and second[1] != (first[1] or 0) + 1:
                return None
            return "list"
        return None
    if previous.kind in {BlockKind.PARAGRAPH, BlockKind.UNKNOWN} and following.kind in {
        BlockKind.PARAGRAPH,
        BlockKind.UNKNOWN,
    }:
        if previous.text.rstrip().endswith(_TERMINAL):
            return None
        if not previous.text.strip() or not following.text.strip():
            return None
        return "paragraph"
    return None


def _merged_block(previous: OCRBlock, following: OCRBlock, kind: str) -> OCRBlock:
    flags = list(dict.fromkeys([*previous.flags, *following.flags]))
    confidences = [c for c in (previous.confidence, following.confidence) if c is not None]
    return previous.model_copy(
        update={
            "block_id": f"merged-{previous.block_id}--{following.block_id}",
            "text": _merge_text(previous.text, following.text)
            if kind != "table"
            else previous.text,
            "kind": previous.kind,
            "confidence": min(confidences) if confidences else None,
            "bbox": previous.bbox,
            "flags": [QualityFlag(f) if isinstance(f, str) else f for f in flags],
            "source_block_ids": [
                *previous.source_block_ids,
                previous.block_id,
                *following.source_block_ids,
                following.block_id,
            ],
        }
    )


def analyze_continuity(blocks: list[OCRBlock]) -> ContinuityResult:
    """Merge justified cross-page continuations; table pairs are returned separately."""
    ordered = sorted(blocks, key=lambda b: (b.page_index, b.reading_order))
    by_page: dict[int, list[OCRBlock]] = {}
    for block in ordered:
        by_page.setdefault(block.page_index, []).append(block)

    decisions: list[ContinuityDecision] = []
    table_pairs: list[tuple[str, str]] = []
    consumed: set[str] = set()
    replacements: dict[str, OCRBlock] = {}

    pages = sorted(by_page)
    for previous_page, next_page in pairwise(pages):
        if next_page != previous_page + 1:
            continue
        previous = by_page[previous_page][-1]
        following = by_page[next_page][0]
        if previous.block_id in consumed or following.block_id in consumed:
            continue
        kind = _continuation_kind(previous, following)
        if kind is None:
            continue
        evidence: dict[str, Any] = {
            "previous_page": previous.page_index,
            "following_page": following.page_index,
            "previous_block_id": previous.block_id,
            "following_block_id": following.block_id,
        }
        if kind == "table":
            table_pairs.append((previous.block_id, following.block_id))
            decisions.append(
                ContinuityDecision(
                    target_block_id=previous.block_id,
                    source_block_ids=[previous.block_id, following.block_id],
                    reason="table continues onto the next page (grids merged in structure/tables.json)",
                    evidence=evidence,
                )
            )
            continue
        merged = _merged_block(previous, following, kind)
        replacements[previous.block_id] = merged
        consumed.update({previous.block_id, following.block_id})
        decisions.append(
            ContinuityDecision(
                target_block_id=merged.block_id,
                source_block_ids=merged.source_block_ids,
                reason=(
                    f"{kind} continuation across pages {previous.page_index} -> "
                    f"{following.page_index}"
                    + (" (hyphenated word joined)" if previous.text.rstrip().endswith("-") else "")
                ),
                evidence=evidence,
            )
        )

    result: list[OCRBlock] = []
    for block in ordered:
        if block.block_id in consumed:
            if block.block_id in replacements and replacements[block.block_id] not in result:
                result.append(replacements[block.block_id])
            continue
        result.append(block)
    return ContinuityResult(blocks=result, decisions=decisions, table_pairs=table_pairs)
