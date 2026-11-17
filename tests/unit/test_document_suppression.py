"""Deterministic tests for page furniture suppression (Phase 3B-1 Task 1)."""

from __future__ import annotations

from bodhan.document import analyze_suppression
from bodhan.domain.models import BlockKind, OCRBlock


def block(
    block_id: str,
    page: int,
    order: int,
    text: str,
    *,
    kind: BlockKind = BlockKind.PARAGRAPH,
    label: str | None = None,
) -> OCRBlock:
    return OCRBlock(
        block_id=block_id,
        page_index=page,
        reading_order=order,
        text=text,
        kind=kind,
        provider_label=label,
    )


def test_identical_repeated_headers_are_suppressed_with_reasons() -> None:
    blocks = [
        block("p0-b0", 0, 0, "Mathematics Grade 9", label="Paragraph"),
        block("p0-b1", 0, 1, "Angles and lines."),
        block("p1-b0", 1, 0, "Mathematics Grade 9", label="Paragraph"),
        block("p1-b1", 1, 1, "More content."),
    ]
    active, decisions = analyze_suppression(blocks)

    assert [b.block_id for b in active] == ["p0-b1", "p1-b1"]
    assert [d.block_id for d in decisions] == ["p0-b0", "p1-b0"]
    assert all("repeated" in d.reason for d in decisions)
    assert all(d.evidence["pages"] == [0, 1] for d in decisions)


def test_slightly_varying_headers_with_page_numbers_are_suppressed() -> None:
    blocks = [
        block("p0-b0", 0, 0, "Chapter 1  Page 1"),
        block("p0-b1", 0, 1, "First paragraph."),
        block("p1-b0", 1, 0, "Chapter 1  Page 2"),
        block("p1-b1", 1, 1, "Second paragraph."),
    ]
    active, decisions = analyze_suppression(blocks)

    assert [b.block_id for b in active] == ["p0-b1", "p1-b1"]
    assert len(decisions) == 2
    assert all(d.evidence["similarity"] >= 0.85 for d in decisions)


def test_repeated_footers_are_suppressed() -> None:
    blocks = [
        block("p0-b0", 0, 0, "Body text."),
        block("p0-b1", 0, 1, "NCERT textbook"),
        block("p1-b0", 1, 0, "Another body."),
        block("p1-b1", 1, 1, "NCERT textbook"),
    ]
    active, decisions = analyze_suppression(blocks)

    assert [b.block_id for b in active] == ["p0-b0", "p1-b0"]
    assert all("footer" in d.reason for d in decisions)


def test_standalone_page_numbers_are_suppressed() -> None:
    blocks = [
        block("p0-b0", 0, 0, "12", kind=BlockKind.PAGE_FOOTER),
        block("p0-b1", 0, 1, "— 12 —", kind=BlockKind.PAGE_FOOTER),
        block("p1-b0", 1, 0, "Page 13", kind=BlockKind.PAGE_FOOTER),
        block("p1-b1", 1, 1, "Content."),
    ]
    active, decisions = analyze_suppression(blocks)

    assert [b.block_id for b in active] == ["p1-b1"]
    assert len(decisions) == 3
    assert all("page number" in d.reason for d in decisions)


def test_genuine_heading_near_top_is_kept() -> None:
    blocks = [
        block("p0-b0", 0, 0, "Lines and Angles", kind=BlockKind.HEADING),
        block("p0-b1", 0, 1, "A line has many angles."),
    ]
    active, decisions = analyze_suppression(blocks)

    assert [b.block_id for b in active] == ["p0-b0", "p0-b1"]
    assert decisions == []


def test_genuine_bottom_content_is_kept() -> None:
    blocks = [
        block("p0-b0", 0, 0, "Exercise 6.1"),
        block("p0-b1", 0, 1, "Find the missing angle."),
        block("p1-b0", 1, 0, "Summary"),
        block("p1-b1", 1, 1, "The angles of a triangle sum to 180."),
    ]
    active, decisions = analyze_suppression(blocks)

    assert len(active) == 4
    assert decisions == []


def test_provider_furniture_label_is_suppressed_even_on_one_page() -> None:
    blocks = [
        block("p0-b0", 0, 0, "Copyright 2026", label="Header"),
        block("p0-b1", 0, 1, "Real content."),
    ]
    active, decisions = analyze_suppression(blocks)

    assert [b.block_id for b in active] == ["p0-b1"]
    assert "provider label" in decisions[0].reason


def test_document_without_headers_or_footers_has_no_decisions() -> None:
    blocks = [
        block("p0-b0", 0, 0, "Introduction"),
        block("p0-b1", 0, 1, "First thought."),
        block("p1-b0", 1, 0, "Next section"),
        block("p1-b1", 1, 1, "A different thought."),
    ]
    active, decisions = analyze_suppression(blocks)

    assert active == blocks
    assert decisions == []


def test_inputs_are_not_mutated_and_min_pages_is_validated() -> None:
    blocks = [block("p0-b0", 0, 0, "Header", label="Header")]
    original = blocks[0].model_dump()
    analyze_suppression(blocks)
    assert blocks[0].model_dump() == original

    try:
        analyze_suppression(blocks, min_pages=1)
    except ValueError as exc:
        assert "at least 2" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("min_pages=1 was accepted")
