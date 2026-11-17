"""Offline integration: document intelligence through structure → translate.

Uses fakes only; no API calls, no LLM content generation, no audio.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bodhan.domain.models import BlockKind, OCRBlock, OCRResult, StageName
from bodhan.pipeline.stages import structure, translate
from bodhan.pipeline.stages.narrate import for_narration, split_sections
from bodhan.pipeline.state import load_json, save_json
from tests.conftest import make_context


def ocr_block(
    block_id: str, page: int, order: int, text: str, kind: BlockKind, label: str | None = None
) -> OCRBlock:
    return OCRBlock(
        block_id=block_id,
        page_index=page,
        reading_order=order,
        text=text,
        kind=kind,
        provider_label=label,
    )


def seed(ctx, blocks: list[OCRBlock]) -> None:
    pages = [
        {"page_index": 0, "source_path": "chapter.pdf"},
        {"page_index": 1, "source_path": "chapter.pdf"},
    ]
    save_json(ctx.store, StageName.EXTRACT, "pages.json", pages)
    save_json(ctx.store, StageName.QUALITY, "pages.json", pages)
    save_json(
        ctx.store,
        StageName.EXTRACT,
        "results.json",
        [
            OCRResult(
                page_index=page,
                blocks=[b for b in blocks if b.page_index == page],
                provider="indic-ocr",
                provider_request_id=f"req-{page}",
            ).model_dump()
            for page in (0, 1)
        ],
    )


@pytest.mark.asyncio
async def test_structure_translate_preserves_structure_and_excludes_furniture(
    tmp_path: Path,
) -> None:
    ctx = make_context(tmp_path)
    seed(
        ctx,
        [
            ocr_block("p0-b0", 0, 0, "Chapter 1", BlockKind.PAGE_HEADER, "Header"),
            ocr_block("p0-b1", 0, 1, "$a^2 + b^2 = c^2$", BlockKind.EQUATION),
            ocr_block(
                "p0-b2",
                0,
                2,
                "<table><tr><th>Angle</th><th>Value</th></tr><tr><td>Right</td><td>$90^\\circ$</td></tr></table>",
                BlockKind.TABLE,
            ),
            ocr_block("p0-b3", 0, 3, "", BlockKind.FIGURE, "Image"),
            ocr_block("p0-b4", 0, 4, "Figure 1: Angles", BlockKind.FIGURE_CAPTION),
            ocr_block("p0-b5", 0, 5, "This text continues", BlockKind.PARAGRAPH),
            ocr_block("p1-b0", 1, 0, "Chapter 1", BlockKind.PAGE_HEADER, "Header"),
            ocr_block("p1-b1", 1, 1, "on the following page.", BlockKind.PARAGRAPH),
        ],
    )

    summary = await structure.run(ctx)
    assert summary["suppressed_blocks"] == 2
    assert summary["continuity_merges"] == 1
    assert summary["tables"] == 1
    assert summary["figure_associations"] == 1

    suppressed = load_json(ctx.store, StageName.STRUCTURE, "suppressed.json")
    assert {entry["block_id"] for entry in suppressed} == {"p0-b0", "p1-b0"}
    continuity = load_json(ctx.store, StageName.STRUCTURE, "continuity.json")
    merged_id = continuity[0]["target_block_id"]
    assert "p0-b5" in continuity[0]["source_block_ids"]

    # quality stage consumes structure/blocks.json in the real pipeline.
    active = load_json(ctx.store, StageName.STRUCTURE, "blocks.json")
    save_json(ctx.store, StageName.QUALITY, "blocks.json", active)
    save_json(ctx.store, StageName.QUALITY, "quality_report.json", {"flagged": 0, "entries": []})

    result = await translate.run(ctx)
    items = load_json(ctx.store, StageName.TRANSLATE, "translations.json")["items"]

    assert "p0-b0" not in items  # suppressed furniture is never translated
    assert items["p0-b1"] == "$a^2 + b^2 = c^2$"  # equation verbatim
    assert "| [hi] Angle | [hi] Value |" in items["p0-b2"]
    assert "| [hi] Right | $90^\\circ$ |" in items["p0-b2"]  # math cell restored, not translated
    assert items["p0-b4"] == "[hi] Figure 1: Angles"  # caption survived
    assert merged_id in items
    assert "This text continues" in items[merged_id]
    assert result["table_cells_translated"] >= 3

    tables = load_json(ctx.store, StageName.TRANSLATE, "tables.json")
    assert tables["tables"][0]["source_block_ids"] == ["p0-b2"]
    assert {cell["row"] for cell in tables["tables"][0]["cells"]} == {0, 1}

    # Narration consumes the composed markdown; a rendered table remains
    # available to the narrator as readable text (no new translation path).
    markdown = "# Lesson\n\n## Angles\n\n" + items["p0-b2"] + "\n"
    sections = split_sections(markdown)
    assert len(sections) == 1
    spoken = for_narration(sections[0][1])
    assert "[hi] Angle, [hi] Value" in spoken
    assert "[hi] Right, $90^\\circ$" in spoken
    assert "---" not in spoken  # Markdown separator row is not narrated


@pytest.mark.asyncio
async def test_table_continuation_merges_grids_and_removes_duplicate_block(
    tmp_path: Path,
) -> None:
    ctx = make_context(tmp_path, chapter_id="table-continue")
    seed(
        ctx,
        [
            ocr_block(
                "p0-b0",
                0,
                0,
                "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>",
                BlockKind.TABLE,
            ),
            ocr_block(
                "p1-b0",
                1,
                0,
                "<table><tr><td>3</td><td>4</td></tr></table>",
                BlockKind.TABLE,
            ),
        ],
    )

    summary = await structure.run(ctx)
    assert summary["continuity_merges"] == 1
    assert [b["block_id"] for b in load_json(ctx.store, StageName.STRUCTURE, "blocks.json")] == [
        "p0-b0"
    ]
    grids = load_json(ctx.store, StageName.STRUCTURE, "tables.json")
    assert grids["p0-b0"]["rows"] == 3
    assert grids["p0-b0"]["source_block_ids"] == ["p0-b0", "p1-b0"]

    active = load_json(ctx.store, StageName.STRUCTURE, "blocks.json")
    save_json(ctx.store, StageName.QUALITY, "blocks.json", active)
    save_json(ctx.store, StageName.QUALITY, "quality_report.json", {"flagged": 0, "entries": []})

    await translate.run(ctx)
    items = load_json(ctx.store, StageName.TRANSLATE, "translations.json")["items"]
    assert list(items) == ["p0-b0"]
    assert "| [hi] A | [hi] B |" in items["p0-b0"]
    assert "| [hi] 3 | [hi] 4 |" in items["p0-b0"]
