"""Figure ↔ caption association from OCR provenance (Phase 3B-1 Task 4).

Pure and conservative: a caption is associated only with a figure on the same
page that is clearly nearest. Ambiguous or missing pairs are reported rather
than guessed; no image content is invented.
"""

from __future__ import annotations

from bodhan.document.models import FigureAssociation
from bodhan.domain.models import BlockKind, OCRBlock


def _center(bbox: tuple[float, float, float, float] | None) -> tuple[float, float] | None:
    if bbox is None:
        return None
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def _distance(caption: OCRBlock, figure: OCRBlock) -> float:
    caption_center, figure_center = _center(caption.bbox), _center(figure.bbox)
    if caption_center and figure_center:
        return (
            (caption_center[0] - figure_center[0]) ** 2
            + (caption_center[1] - figure_center[1]) ** 2
        ) ** 0.5
    return float(abs(caption.reading_order - figure.reading_order))


def associate_figures(blocks: list[OCRBlock]) -> list[FigureAssociation]:
    """Associate each caption with the nearest same-page figure, if any."""
    figures_by_page: dict[int, list[OCRBlock]] = {}
    for block in blocks:
        if block.kind is BlockKind.FIGURE:
            figures_by_page.setdefault(block.page_index, []).append(block)

    associations: list[FigureAssociation] = []
    for caption in sorted(blocks, key=lambda b: (b.page_index, b.reading_order)):
        if caption.kind is not BlockKind.FIGURE_CAPTION:
            continue
        candidates = figures_by_page.get(caption.page_index, [])
        if not candidates:
            associations.append(
                FigureAssociation(
                    caption_block_id=caption.block_id,
                    figure_block_id=None,
                    reason="no figure block on the same page",
                )
            )
            continue
        ranked = sorted(
            ((_distance(caption, figure), figure) for figure in candidates),
            key=lambda pair: pair[0],
        )
        best_distance, best = ranked[0]
        if len(ranked) > 1 and ranked[1][0] <= best_distance * 1.15 + 1:
            associations.append(
                FigureAssociation(
                    caption_block_id=caption.block_id,
                    figure_block_id=None,
                    reason="ambiguous: multiple figures are nearly equidistant",
                    confidence=round(best_distance, 4),
                )
            )
            continue
        if best_distance > 300:
            associations.append(
                FigureAssociation(
                    caption_block_id=caption.block_id,
                    figure_block_id=None,
                    reason="no figure close enough to this caption",
                    confidence=round(best_distance, 4),
                )
            )
            continue
        associations.append(
            FigureAssociation(
                caption_block_id=caption.block_id,
                figure_block_id=best.block_id,
                reason="nearest same-page figure (page and positional evidence)",
                confidence=round(1 / (1 + best_distance), 4),
            )
        )
    return associations
