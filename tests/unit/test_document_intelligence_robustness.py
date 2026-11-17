"""Phase 3B-2 Task 7: document intelligence + failure scenarios (offline)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from bodhan.document import validate_placeholders
from bodhan.domain.models import BlockKind, OCRBlock, OCRResult, StageName
from bodhan.pipeline.stages import structure, translate
from bodhan.pipeline.state import load_json, save_json
from tests.conftest import make_context


def blk(
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


def seed(blocks: list[OCRBlock]) -> Any:
    from tempfile import mkdtemp

    ctx = make_context(Path(mkdtemp()), chapter_id="doc-robustness")
    pages = [{"page_index": 0, "source_path": "c.pdf"}, {"page_index": 1, "source_path": "c.pdf"}]
    save_json(ctx.store, StageName.EXTRACT, "pages.json", pages)
    save_json(ctx.store, StageName.QUALITY, "pages.json", pages)
    save_json(ctx.store, StageName.QUALITY, "quality_report.json", {"flagged": 0, "entries": []})
    save_json(
        ctx.store,
        StageName.EXTRACT,
        "results.json",
        [
            OCRResult(
                page_index=page,
                blocks=[b for b in blocks if b.page_index == page],
                provider="indic-ocr",
            ).model_dump()
            for page in (0, 1)
        ],
    )
    return ctx


class SelectiveTranslate:
    provider = "selective"

    def __init__(self, fail_on: str) -> None:
        self.fail_on = fail_on

    async def translate(self, text: str, *, target_language: str) -> Any:
        from types import SimpleNamespace

        if self.fail_on in text:
            raise RuntimeError("simulated provider failure")
        return SimpleNamespace(
            text=f"[hi] {text}",
            provider="bodhan-translate",
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_multipage_table_with_translation_failure_keeps_first_page_cells() -> None:
    ctx = seed(
        [
            blk(
                "p0-b0",
                0,
                0,
                "<table><tr><th>A</th><th>B</th></tr><tr><td>ok</td><td>also ok</td></tr></table>",
                BlockKind.TABLE,
            ),
            blk(
                "p1-b0",
                1,
                0,
                "<table><tr><td>fail-cell</td><td>last</td></tr></table>",
                BlockKind.TABLE,
            ),
        ]
    )
    await structure.run(ctx)
    save_json(
        ctx.store,
        StageName.QUALITY,
        "blocks.json",
        load_json(ctx.store, StageName.STRUCTURE, "blocks.json"),
    )
    ctx.translate = SelectiveTranslate("fail-cell")  # type: ignore[assignment]
    from bodhan.pipeline.errors import PipelineError

    with pytest.raises(PipelineError):
        await translate.run(ctx)

    checkpoint_path = ctx.store.stage_dir(StageName.TRANSLATE.value) / "blocks" / "p0-b0.json"
    import json

    cells = json.loads(checkpoint_path.read_text(encoding="utf-8"))["table"]["cells"]
    assert any(cell["status"] == "ok" for cell in cells)
    assert any(cell["status"] == "failed" for cell in cells)


@pytest.mark.asyncio
async def test_cross_page_paragraph_with_interrupted_rerun_resumes() -> None:
    ctx = seed(
        [
            blk("p0-b1", 0, 0, "A continued paragraph", BlockKind.PARAGRAPH),
            blk("p1-b0", 1, 0, "finishes here.", BlockKind.PARAGRAPH),
        ]
    )
    await structure.run(ctx)
    active = load_json(ctx.store, StageName.STRUCTURE, "blocks.json")
    assert len(active) == 1
    assert active[0]["source_block_ids"] == ["p0-b1", "p1-b0"]

    save_json(ctx.store, StageName.QUALITY, "blocks.json", active)
    ctx.translate = SelectiveTranslate("NEVER")  # type: ignore[assignment]
    await translate.run(ctx)
    first = load_json(ctx.store, StageName.TRANSLATE, "translations.json")["items"]
    assert "continued paragraph" in str(first)

    class CountingTranslate(SelectiveTranslate):
        def __init__(self, fail_on: str) -> None:
            super().__init__(fail_on)
            self.calls: list[str] = []

        async def translate(self, text: str, *, target_language: str) -> Any:
            self.calls.append(text)
            return await super().translate(text, target_language=target_language)

    counter = CountingTranslate("NEVER")
    ctx.translate = counter  # type: ignore[assignment]
    await translate.run(ctx)
    assert counter.calls == []  # idempotent rerun: checkpoint reused


@pytest.mark.asyncio
async def test_numbered_list_continues_across_pages() -> None:
    ctx = seed(
        [
            blk("p0-b1", 0, 1, "1. First", BlockKind.LIST_ITEM),
            blk("p0-b2", 0, 2, "2. Second", BlockKind.LIST_ITEM),
            blk("p1-b0", 1, 0, "3. Third", BlockKind.LIST_ITEM),
        ]
    )
    await structure.run(ctx)
    decisions = load_json(ctx.store, StageName.STRUCTURE, "continuity.json")
    assert any("list continuation" in d["reason"] for d in decisions)


@pytest.mark.asyncio
async def test_figure_caption_survives_translation_and_suppression() -> None:
    ctx = seed(
        [
            blk("p0-b0", 0, 0, "Book", BlockKind.PAGE_HEADER, "Header"),
            blk("p0-b1", 0, 1, "Book", BlockKind.PAGE_HEADER, "Header"),
            blk("p0-b2", 0, 2, "", BlockKind.FIGURE, "Image"),
            blk("p0-b3", 0, 3, "Figure 1: a line", BlockKind.FIGURE_CAPTION),
        ]
    )
    await structure.run(ctx)
    active = load_json(ctx.store, StageName.STRUCTURE, "blocks.json")
    assert {b["block_id"] for b in active} == {"p0-b2", "p0-b3"}
    assert (
        load_json(ctx.store, StageName.STRUCTURE, "figures.json")[0]["figure_block_id"] == "p0-b2"
    )
    save_json(ctx.store, StageName.QUALITY, "blocks.json", active)
    ctx.translate = SelectiveTranslate("NEVER")  # type: ignore[assignment]
    await translate.run(ctx)
    items = load_json(ctx.store, StageName.TRANSLATE, "translations.json")["items"]
    assert items["p0-b3"] == "[hi] Figure 1: a line"


def test_equation_in_table_placeholder_validation_rejects_corruption() -> None:
    from bodhan.document import protect_math

    protected, mapping, _ = protect_math("Use $x^2$")
    assert mapping
    restored, error = validate_placeholders("Use corrupted", mapping)
    assert restored is None and "missing" in str(error)
    restored, error = validate_placeholders(protected, mapping)
    assert restored == "Use $x^2$" and error is None


@pytest.mark.asyncio
async def test_malformed_table_is_flagged_not_dropped() -> None:
    ctx = seed(
        [
            blk(
                "p0-b0",
                0,
                0,
                "<table><tr><td>A</td><td>B</td></tr><tr><td>C</td>",
                BlockKind.TABLE,
            )
        ]
    )
    await structure.run(ctx)
    grids = load_json(ctx.store, StageName.STRUCTURE, "tables.json")
    assert grids["p0-b0"]["incomplete"] is True
    save_json(
        ctx.store,
        StageName.QUALITY,
        "blocks.json",
        load_json(ctx.store, StageName.STRUCTURE, "blocks.json"),
    )
    ctx.translate = SelectiveTranslate("NEVER")  # type: ignore[assignment]
    await translate.run(ctx)
    assert (
        "needs review"
        in load_json(ctx.store, StageName.TRANSLATE, "translations.json")["items"]["p0-b0"]
    )


@pytest.mark.asyncio
async def test_suppressed_header_never_reappears_in_composed_markdown(tmp_path: Path) -> None:
    ctx = seed(
        [
            blk("p0-b0", 0, 0, "Repeated", BlockKind.PAGE_HEADER, "Header"),
            blk("p1-b0", 1, 0, "Repeated", BlockKind.PAGE_HEADER, "Header"),
            blk("p0-b1", 0, 1, "Real body", BlockKind.PARAGRAPH),
        ]
    )
    await structure.run(ctx)
    active = load_json(ctx.store, StageName.STRUCTURE, "blocks.json")
    assert {b["block_id"] for b in active} == {"p0-b1"}
