"""Checkpoint metadata must record protected math/unit span counts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from bodhan.domain.models import BlockKind, OCRBlock, StageName
from bodhan.pipeline.errors import PipelineError
from bodhan.pipeline.stages import translate as translate_stage
from bodhan.pipeline.stages.translate import _source_hash
from bodhan.pipeline.state import save_json
from tests.conftest import make_context


def seed(tmp_path: Path, texts: dict[str, str]) -> Any:
    ctx = make_context(tmp_path, chapter_id="translation-metadata")
    pages = [{"page_index": 0, "source_path": "chapter.pdf"}]
    save_json(ctx.store, StageName.QUALITY, "pages.json", pages)
    save_json(
        ctx.store,
        StageName.QUALITY,
        "blocks.json",
        [
            OCRBlock(
                block_id=block_id,
                page_index=0,
                reading_order=index,
                text=text,
                kind=BlockKind.PARAGRAPH,
            ).model_dump()
            for index, (block_id, text) in enumerate(texts.items())
        ],
    )
    save_json(ctx.store, StageName.STRUCTURE, "tables.json", {})
    return ctx


def record(ctx: Any, block_id: str) -> dict[str, Any]:
    path = ctx.store.stage_dir(StageName.TRANSLATE.value) / "blocks" / f"{block_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.asyncio
async def test_math_span_count_is_persisted(tmp_path: Path) -> None:
    ctx = seed(tmp_path, {"p0-b0": "The value $x^2$ is positive."})
    await translate_stage.run(ctx)
    result = record(ctx, "p0-b0")
    assert result["status"] == "ok"
    assert result["math_span_count"] == 1
    assert result["unit_span_count"] == 0


@pytest.mark.asyncio
async def test_unit_span_count_is_persisted(tmp_path: Path) -> None:
    ctx = seed(tmp_path, {"p0-b0": "Value in mm and µm here."})
    await translate_stage.run(ctx)
    result = record(ctx, "p0-b0")
    assert result["status"] == "ok"
    assert result["math_span_count"] == 0
    assert result["unit_span_count"] == 2


@pytest.mark.asyncio
async def test_repeated_math_and_units_are_counted(tmp_path: Path) -> None:
    ctx = seed(tmp_path, {"p0-b0": "$l$ and $m$ and $l$; units mm and µm."})
    await translate_stage.run(ctx)
    result = record(ctx, "p0-b0")
    assert result["math_span_count"] == 3
    assert result["unit_span_count"] == 2


@pytest.mark.asyncio
async def test_math_only_block_persists_counts_without_api_call(tmp_path: Path) -> None:
    ctx = seed(tmp_path, {"p0-b0": "$x^2 + y^2$"})
    result = await translate_stage.run(ctx)
    stored = record(ctx, "p0-b0")
    assert result["blocks"] == 1
    assert stored["math_span_count"] == 1
    assert stored["unit_span_count"] == 0


@pytest.mark.asyncio
async def test_legacy_checkpoint_without_count_fields_remains_reusable(tmp_path: Path) -> None:
    text = "plain prose"
    ctx = seed(tmp_path, {"p0-b0": text})
    directory = ctx.store.stage_dir(StageName.TRANSLATE.value) / "blocks"
    directory.mkdir(parents=True, exist_ok=True)
    legacy = {
        "block_id": "p0-b0",
        "target_language": "hi",
        "source_hash": _source_hash(text, "hi"),
        "status": "ok",
        "translation": "legacy translation",
        "usage": {},
    }
    (directory / "p0-b0.json").write_text(json.dumps(legacy), encoding="utf-8")

    result = await translate_stage.run(ctx)

    assert result["reused_blocks"] == 1
    assert record(ctx, "p0-b0") == legacy


@pytest.mark.asyncio
async def test_failed_translation_does_not_publish_success_metadata(tmp_path: Path) -> None:
    ctx = seed(tmp_path, {"p0-b0": "Value $x$ in mm."})

    class FailingTranslate:
        provider = "failing"

        async def translate(self, text: str, *, target_language: str) -> Any:
            raise RuntimeError("simulated provider failure")

        async def aclose(self) -> None:
            return None

    ctx.translate = FailingTranslate()  # type: ignore[assignment]
    with pytest.raises(PipelineError):
        await translate_stage.run(ctx)
    stored = record(ctx, "p0-b0")
    assert stored["status"] == "failed"
    assert stored["translation"] is None
    assert stored["math_span_count"] == 0
    assert stored["unit_span_count"] == 0
