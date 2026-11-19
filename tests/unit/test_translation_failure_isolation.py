"""Phase 3B-2 Task 1: translation checkpointing and failure isolation."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from bodhan.domain.models import BlockKind, OCRBlock, StageName
from bodhan.pipeline.errors import PipelineError
from bodhan.pipeline.stages import translate as translate_stage
from bodhan.pipeline.state import load_json, save_json
from tests.conftest import make_context


class ScriptedTranslate:
    provider = "scripted-translate"

    def __init__(self, handler) -> None:
        self.handler = handler
        self.calls: list[str] = []

    async def translate(self, text: str, *, target_language: str) -> Any:
        self.calls.append(text)
        return SimpleNamespace(**self.handler(text, len(self.calls)))

    async def aclose(self) -> None:
        return None


def block(block_id: str, text: str, kind: BlockKind = BlockKind.PARAGRAPH) -> OCRBlock:
    return OCRBlock(block_id=block_id, page_index=0, reading_order=0, text=text, kind=kind)


def seed(tmp_path: Path, blocks: list[OCRBlock], tables: dict[str, Any] | None = None) -> Any:
    ctx = make_context(tmp_path, chapter_id="translate-robustness")
    pages = [{"page_index": 0, "source_path": "chapter.pdf"}]
    save_json(ctx.store, StageName.QUALITY, "pages.json", pages)
    save_json(ctx.store, StageName.QUALITY, "blocks.json", [b.model_dump() for b in blocks])
    save_json(
        ctx.store,
        StageName.STRUCTURE,
        "tables.json",
        tables
        or {"p0-b0": {"block_id": "p0-b0", "page_index": 0, "rows": 1, "columns": 1, "cells": []}},
    )
    return ctx


def record(ctx, block_id: str) -> dict[str, Any]:
    safe = block_id.replace("-", "-")
    path = ctx.store.stage_dir(StageName.TRANSLATE.value) / "blocks" / f"{safe}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def ok(text: str, call_number: int) -> dict[str, Any]:
    return {
        "text": f"T{call_number}:{text}",
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


@pytest.mark.asyncio
async def test_one_failed_block_keeps_siblings_and_resumes_only_failure(tmp_path: Path) -> None:
    blocks = [block("p0-b0", "alpha"), block("p0-b1", "beta"), block("p0-b2", "gamma")]
    ctx = seed(tmp_path, blocks)

    def handler(text: str, call: int) -> dict[str, Any]:
        if "beta" in text:
            raise RuntimeError("simulated transient failure")
        return ok(text, call)

    ctx.translate = ScriptedTranslate(handler)  # type: ignore[assignment]
    first = await translate_stage.run(ctx)

    translations = load_json(ctx.store, StageName.TRANSLATE, "translations.json")
    assert set(translations["items"]) == {"p0-b0", "p0-b2"}
    assert translations["failed_blocks"][0]["block_id"] == "p0-b1"
    assert record(ctx, "p0-b1")["status"] == "failed"
    assert record(ctx, "p0-b0")["status"] == "ok"
    assert first["usage"]["total_tokens"] == 4  # only successful calls counted

    fixed = ScriptedTranslate(ok)
    ctx.translate = fixed  # type: ignore[assignment]
    second = await translate_stage.run(ctx)

    assert second["reused_blocks"] == 2
    assert fixed.calls == ["beta"]  # only the failed block was retried
    assert len(load_json(ctx.store, StageName.TRANSLATE, "translations.json")["items"]) == 3


@pytest.mark.asyncio
async def test_interrupted_rerun_reuses_valid_checkpoints(tmp_path: Path) -> None:
    blocks = [block("p0-b0", "alpha"), block("p0-b1", "beta")]
    ctx = seed(tmp_path, blocks)
    calls: list[str] = []

    class Interrupt(ScriptedTranslate):
        async def translate(self, text: str, *, target_language: str) -> Any:
            calls.append(text)
            if "beta" in text:
                raise KeyboardInterrupt("simulated interruption")
            return SimpleNamespace(**ok(text, len(calls)))

    ctx.translate = Interrupt(ok)  # type: ignore[assignment]
    with pytest.raises(KeyboardInterrupt):
        await translate_stage.run(ctx)

    assert record(ctx, "p0-b0")["status"] == "ok"
    ctx.translate = ScriptedTranslate(ok)  # type: ignore[assignment]
    result = await translate_stage.run(ctx)
    assert result["reused_blocks"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        {"text": "the value is 5"},
        {"text": "\ue0000\ue001 \ue0000\ue001 duplicate"},
    ],
)
async def test_math_is_preserved_when_provider_returns_altered_text(
    tmp_path: Path, response: dict[str, Any]
) -> None:
    """Span-split translation never sends math to the provider, so even a
    provider that returns placeholder-like or altered text cannot corrupt the
    mathematical span; it is verified byte-for-byte on reconstruction."""
    ctx = seed(tmp_path, [block("p0-b0", "Value $x^2$ here")])
    ctx.translate = ScriptedTranslate(lambda text, call: response)  # type: ignore[assignment]

    result = await translate_stage.run(ctx)

    assert result["status"] == "completed"
    checkpoint = record(ctx, "p0-b0")
    assert checkpoint["status"] == "ok"
    assert "$x^2$" in checkpoint["translation"]


@pytest.mark.asyncio
async def test_malformed_provider_response_is_isolated(tmp_path: Path) -> None:
    ctx = seed(tmp_path, [block("p0-b0", "alpha")])
    ctx.translate = ScriptedTranslate(lambda text, call: {"text": None})  # type: ignore[assignment]

    with pytest.raises(PipelineError, match="translation failed for every block"):
        await translate_stage.run(ctx)
    assert "missing text" in record(ctx, "p0-b0")["error"]


@pytest.mark.asyncio
async def test_table_cell_failure_preserves_successful_siblings(tmp_path: Path) -> None:
    tables = {
        "p0-b0": {
            "block_id": "p0-b0",
            "page_index": 0,
            "rows": 1,
            "columns": 2,
            "incomplete": False,
            "review_reason": None,
            "source_block_ids": ["p0-b0"],
            "cells": [
                {
                    "row": 0,
                    "column": 0,
                    "text": "good",
                    "is_header": True,
                    "source_block_id": "p0-b0",
                },
                {
                    "row": 0,
                    "column": 1,
                    "text": "bad",
                    "is_header": True,
                    "source_block_id": "p0-b0",
                },
            ],
        }
    }
    ctx = seed(
        tmp_path,
        [block("p0-b0", "<table></table>", BlockKind.TABLE), block("p0-b1", "sibling")],
        tables,
    )

    def handler(text: str, call: int) -> dict[str, Any]:
        if "bad" in text:
            raise RuntimeError("simulated cell failure")
        return ok(text, call)

    ctx.translate = ScriptedTranslate(handler)  # type: ignore[assignment]
    await translate_stage.run(ctx)

    checkpoint = record(ctx, "p0-b0")
    assert checkpoint["status"] == "failed"
    cells = checkpoint["table"]["cells"]
    assert [cell["status"] for cell in cells] == ["ok", "failed"]
    assert cells[0]["translated_text"] != "good"  # provider-prefixed translation kept
    assert "p0-b1" in load_json(ctx.store, StageName.TRANSLATE, "translations.json")["items"]


@pytest.mark.asyncio
@pytest.mark.parametrize("error", ["rate_limit", "auth", "quota"])
async def test_provider_failures_are_recorded_without_unbounded_retry(
    tmp_path: Path, error: str
) -> None:
    ctx = seed(tmp_path, [block("p0-b0", "alpha")])
    calls: list[int] = []

    def handler(text: str, call: int) -> dict[str, Any]:
        calls.append(call)
        if error == "rate_limit":
            raise RuntimeError("429 retry-after=1")
        if error == "auth":
            raise RuntimeError("401 invalid key")
        raise RuntimeError("429 out of credit")

    ctx.translate = ScriptedTranslate(handler)  # type: ignore[assignment]
    with pytest.raises(PipelineError):
        await translate_stage.run(ctx)

    assert calls == [1]  # adapters own bounded retries; the stage never loops
    assert record(ctx, "p0-b0")["status"] == "failed"
