"""Figure/caption association tests (Phase 3B-1 Task 4)."""

from __future__ import annotations

from bodhan.document import associate_figures
from bodhan.domain.models import BlockKind, OCRBlock


def named(block_id: str, page: int, order: int, kind: BlockKind, bbox=None) -> OCRBlock:
    return OCRBlock(
        block_id=block_id,
        page_index=page,
        reading_order=order,
        text="" if kind is BlockKind.FIGURE else "caption text",
        kind=kind,
        bbox=bbox,
    )


def test_figure_followed_by_caption_is_associated() -> None:
    blocks = [
        named("fig", 0, 0, BlockKind.FIGURE, (0, 0, 100, 100)),
        named("cap", 0, 1, BlockKind.FIGURE_CAPTION, (0, 110, 100, 130)),
    ]
    associations = associate_figures(blocks)

    assert associations[0].figure_block_id == "fig"
    assert "nearest" in associations[0].reason


def test_caption_above_figure_is_associated() -> None:
    blocks = [
        named("cap", 0, 0, BlockKind.FIGURE_CAPTION, (0, 0, 100, 20)),
        named("fig", 0, 1, BlockKind.FIGURE, (0, 30, 100, 130)),
    ]
    associations = associate_figures(blocks)
    assert associations[0].figure_block_id == "fig"


def test_multiple_figures_on_one_page_uses_nearest() -> None:
    blocks = [
        named("far", 0, 0, BlockKind.FIGURE, (300, 0, 400, 100)),
        named("near", 0, 1, BlockKind.FIGURE, (0, 0, 100, 100)),
        named("cap", 0, 2, BlockKind.FIGURE_CAPTION, (0, 110, 100, 130)),
    ]
    associations = associate_figures(blocks)
    assert associations[0].figure_block_id == "near"


def test_ambiguous_caption_is_not_guessed() -> None:
    blocks = [
        named("left", 0, 0, BlockKind.FIGURE, (0, 0, 100, 100)),
        named("right", 0, 1, BlockKind.FIGURE, (200, 0, 300, 100)),
        named("cap", 0, 2, BlockKind.FIGURE_CAPTION, (100, 50, 200, 70)),
    ]
    associations = associate_figures(blocks)
    assert associations[0].figure_block_id is None
    assert "ambiguous" in associations[0].reason


def test_missing_caption_figure_is_reported() -> None:
    blocks = [named("cap", 0, 0, BlockKind.FIGURE_CAPTION)]
    associations = associate_figures(blocks)
    assert associations[0].figure_block_id is None
    assert "no figure" in associations[0].reason


def test_empty_figure_text_does_not_invent_a_description() -> None:
    figure = named("fig", 0, 0, BlockKind.FIGURE, (0, 0, 50, 50))
    caption = named("cap", 0, 1, BlockKind.FIGURE_CAPTION, (0, 60, 50, 80))
    associations = associate_figures([figure, caption])

    assert figure.text == ""
    assert caption.text == "caption text"  # caption preserved, never replaced
    assert associations[0].figure_block_id == "fig"
