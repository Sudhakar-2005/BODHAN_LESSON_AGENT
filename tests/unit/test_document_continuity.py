"""Cross-page continuity tests (Phase 3B-1 Task 3)."""

from __future__ import annotations

from bodhan.document import analyze_continuity
from bodhan.domain.models import BlockKind, OCRBlock


def block(
    block_id: str,
    page: int,
    order: int,
    text: str,
    kind: BlockKind = BlockKind.PARAGRAPH,
) -> OCRBlock:
    return OCRBlock(block_id=block_id, page_index=page, reading_order=order, text=text, kind=kind)


def test_paragraph_continuation_is_merged_with_provenance() -> None:
    blocks = [
        block("p0-b1", 0, 0, "This sentence continues"),
        block("p1-b0", 1, 0, "on the next page."),
    ]
    result = analyze_continuity(blocks)

    assert len(result.blocks) == 1
    merged = result.blocks[0]
    assert merged.text == "This sentence continues on the next page."
    assert merged.source_block_ids == ["p0-b1", "p1-b0"]
    assert result.decisions[0].source_block_ids == ["p0-b1", "p1-b0"]


def test_unrelated_paragraphs_are_not_joined() -> None:
    blocks = [
        block("p0-b1", 0, 0, "This paragraph is complete."),
        block("p1-b0", 1, 0, "A different idea begins."),
    ]
    result = analyze_continuity(blocks)

    assert len(result.blocks) == 2
    assert result.decisions == []


def test_hyphenated_word_split_across_pages_is_joined_carefully() -> None:
    blocks = [
        block("p0-b1", 0, 0, "This is an exam-"),
        block("p1-b0", 1, 0, "ple of a split word."),
    ]
    result = analyze_continuity(blocks)

    assert result.blocks[0].text == "This is an example of a split word."


def test_numbered_list_continuation_preserves_order() -> None:
    blocks = [
        block("p0-b1", 0, 0, "1. First step"),
        block("p1-b0", 1, 0, "2. Second step", BlockKind.LIST_ITEM),
    ]
    blocks[0] = blocks[0].model_copy(update={"kind": BlockKind.LIST_ITEM})
    result = analyze_continuity(blocks)

    assert len(result.blocks) == 1
    assert "2. Second step" in result.blocks[0].text
    assert result.decisions[0].reason.startswith("list continuation")


def test_non_sequential_list_number_is_not_merged() -> None:
    blocks = [
        block("p0-b1", 0, 0, "1. First", BlockKind.LIST_ITEM),
        block("p1-b0", 1, 0, "7. Unrelated", BlockKind.LIST_ITEM),
    ]
    result = analyze_continuity(blocks)
    assert len(result.blocks) == 2


def test_bulleted_list_continuation_is_merged() -> None:
    blocks = [
        block("p0-b1", 0, 0, "- first point", BlockKind.LIST_ITEM),
        block("p1-b0", 1, 0, "- second point", BlockKind.LIST_ITEM),
    ]
    result = analyze_continuity(blocks)
    assert len(result.blocks) == 1


def test_table_pair_is_recorded_for_grid_merge() -> None:
    first = block("p0-b2", 0, 2, "<table><tr><td>A</td></tr></table>", BlockKind.TABLE)
    second = block("p1-b0", 1, 0, "<table><tr><td>B</td></tr></table>", BlockKind.TABLE)
    result = analyze_continuity([first, second])

    assert result.table_pairs == [("p0-b2", "p1-b0")]
    assert result.decisions[0].source_block_ids == ["p0-b2", "p1-b0"]
    assert len(result.blocks) == 2  # grids, not raw text, are merged downstream


def test_page_gaps_and_headings_prevent_continuation() -> None:
    blocks = [
        block("p0-b1", 0, 0, "unfinished thought"),
        block("p2-b0", 2, 0, "on page three"),
        block("p3-b1", 3, 0, "Chapter 7", BlockKind.HEADING),
    ]
    result = analyze_continuity(blocks)
    assert len(result.blocks) == 3
    assert result.decisions == []
