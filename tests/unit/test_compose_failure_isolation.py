"""Phase 3B-2 Task 2: composition validation and idempotence (offline)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bodhan.domain.models import BlockKind, OCRBlock, StageName
from bodhan.pipeline.errors import PipelineError
from bodhan.pipeline.stages import compose
from bodhan.pipeline.state import load_json, save_json
from tests.conftest import FakeLLM, make_context


class TranslateEcho:
    provider = "echo"

    async def translate(self, text: str, *, target_language: str) -> object:
        return SimpleNamespace(
            text=f"[{target_language}] {text}",
            provider="bodhan-translate",
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )

    async def aclose(self) -> None:
        return None


class FakePlanner(FakeLLM):
    """Minimal LLM double: returns a valid English plan via complete_json."""

    async def complete(self, messages, **kwargs):
        from bodhan.ports.llm import LLMResponse

        payload = {
            "title_en": "Lesson",
            "sections": [
                {"heading_en": "First", "block_ids": ["p0-b0"]},
                {"heading_en": "Second", "block_ids": ["p0-b1"]},
            ],
            "objectives_en": [],
            "summary_en": "",
            "glossary_en": [],
            "exercises_en": [],
            "narration_en": [],
        }
        return LLMResponse(text=json.dumps(payload), model="fake", provider="fake", usage={})


def seed(tmp_path: Path, *, suppressed: bool = False) -> object:
    ctx = make_context(tmp_path)
    pages = [{"page_index": 0, "source_path": "c.pdf"}]
    save_json(ctx.store, StageName.QUALITY, "pages.json", pages)
    save_json(ctx.store, StageName.TRANSLATE, "pages.json", pages)
    save_json(
        ctx.store,
        StageName.QUALITY,
        "blocks.json",
        [
            OCRBlock(
                block_id="p0-b0", page_index=0, text="one", kind=BlockKind.PARAGRAPH
            ).model_dump(),
            OCRBlock(
                block_id="p0-b1", page_index=0, text="two", kind=BlockKind.PARAGRAPH
            ).model_dump(),
        ],
    )
    save_json(
        ctx.store,
        StageName.TRANSLATE,
        "translations.json",
        {"target_language": "hi", "items": {"p0-b0": "body one", "p0-b1": "body two"}},
    )
    if suppressed:
        save_json(
            ctx.store,
            StageName.STRUCTURE,
            "suppressed.json",
            [{"block_id": "p0-b1", "page_index": 0, "text": "Header", "reason": "test"}],
        )
    ctx.translate = TranslateEcho()  # type: ignore[assignment]
    ctx.llm = FakePlanner()  # type: ignore[assignment]
    return ctx


@pytest.mark.asyncio
async def test_sections_are_persisted_independently_and_idempotently(tmp_path: Path) -> None:
    ctx = seed(tmp_path)

    first = await compose.run(ctx)
    sections_dir = ctx.store.stage_dir(StageName.COMPOSE.value) / "sections"
    records = [json.loads((sections_dir / f"section-{i:02d}.json").read_text()) for i in (1, 2)]

    assert [r["block_ids"] for r in records] == [["p0-b0"], ["p0-b1"]]
    assert [r["section_number"] for r in records] == [1, 2]
    assert first["section_artifacts"] == 2
    lesson = load_json(ctx.store, StageName.COMPOSE, "lesson.json")
    assert lesson["markdown"].index("[hi] First") < lesson["markdown"].index("[hi] Second")
    hashes = [r["body_hash"] for r in records]

    await compose.run(ctx)
    again = [json.loads((sections_dir / f"section-{i:02d}.json").read_text()) for i in (1, 2)]
    assert [r["body_hash"] for r in again] == hashes


@pytest.mark.asyncio
async def test_suppressed_block_reference_is_rejected(tmp_path: Path) -> None:
    ctx = seed(tmp_path, suppressed=True)

    with pytest.raises(PipelineError, match="suppressed"):
        await compose.run(ctx)


@pytest.mark.asyncio
async def test_malformed_markdown_is_rejected(tmp_path: Path) -> None:
    ctx = seed(tmp_path)
    original = compose.Lesson.model_validate  # keep a handle for clarity

    async def empty_plan(*args: object, **kwargs: object):
        from bodhan.ports.llm import LLMResponse

        return LLMResponse(
            text=json.dumps(
                {
                    "title_en": "Title",
                    "sections": [],
                    "objectives_en": [],
                    "summary_en": "",
                    "glossary_en": [],
                    "exercises_en": [],
                    "narration_en": [],
                }
            ),
            model="fake",
            provider="fake",
            usage={},
        )

    ctx.llm.complete = empty_plan  # type: ignore[assignment]
    with pytest.raises(PipelineError, match=r"sections|invalid Markdown"):
        await compose.run(ctx)
    assert original  # model untouched
