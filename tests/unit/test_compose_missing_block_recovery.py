"""Bounded missing-block recovery for the lesson planner (offline)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from bodhan.domain.models import BlockKind, OCRBlock, StageName
from bodhan.pipeline.errors import PipelineError
from bodhan.pipeline.stages import compose
from bodhan.pipeline.state import load_json, save_json
from bodhan.ports.llm import LLMResponse
from tests.conftest import make_context


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


class QueuedPlanner:
    """Return queued LLM responses and retain every call for assertions."""

    provider = "fake"
    model = "fake"

    def __init__(self, responses: list[dict[str, Any]], *, fail_on: int | None = None) -> None:
        self.responses = responses
        self.calls: list[list[Any]] = []
        self.fail_on = fail_on

    async def complete(self, messages: list[Any], **_: object) -> LLMResponse:
        self.calls.append(messages)
        if self.fail_on is not None and len(self.calls) == self.fail_on:
            raise PipelineError("planner unavailable")
        if not self.responses:
            raise AssertionError("unexpected extra LLM call")
        return LLMResponse(
            text=json.dumps(self.responses.pop(0)), model="fake", provider="fake", usage={}
        )


def plan(sections: list[tuple[str, list[str]]]) -> dict[str, Any]:
    return {
        "title_en": "Lesson",
        "sections": [{"heading_en": heading, "block_ids": ids} for heading, ids in sections],
        "objectives_en": [],
        "summary_en": "",
        "glossary_en": [],
        "exercises_en": [],
        "narration_en": [],
    }


def recovery(sections: list[tuple[str, list[str]]]) -> dict[str, Any]:
    return {"sections": [{"heading_en": heading, "block_ids": ids} for heading, ids in sections]}


def seed(
    tmp_path: Path,
    ids: list[str],
    responses: list[dict[str, Any]],
    *,
    orders: dict[str, int] | None = None,
    fail_on: int | None = None,
) -> Any:
    ctx = make_context(tmp_path)
    pages = [{"page_index": 0, "source_path": "c.pdf"}]
    save_json(ctx.store, StageName.QUALITY, "pages.json", pages)
    save_json(ctx.store, StageName.TRANSLATE, "pages.json", pages)
    orders = orders or {}
    save_json(
        ctx.store,
        StageName.QUALITY,
        "blocks.json",
        [
            OCRBlock(
                block_id=block_id,
                page_index=0,
                reading_order=orders.get(block_id, index),
                text=f"source {block_id}",
                kind=BlockKind.PARAGRAPH,
            ).model_dump()
            for index, block_id in enumerate(ids)
        ],
    )
    save_json(
        ctx.store,
        StageName.TRANSLATE,
        "translations.json",
        {"target_language": "hi", "items": {block_id: f"body {block_id}" for block_id in ids}},
    )
    ctx.translate = TranslateEcho()  # type: ignore[assignment]
    ctx.llm = QueuedPlanner(responses, fail_on=fail_on)  # type: ignore[assignment]
    return ctx


def user_payload(calls: list[list[Any]], index: int) -> dict[str, Any]:
    content = calls[index][1].content
    marker = "Assign these missing blocks to lesson sections:\n"
    return json.loads(content.split(marker, 1)[1])


@pytest.mark.asyncio
async def test_complete_initial_plan_makes_exactly_one_call(tmp_path: Path) -> None:
    ctx = seed(tmp_path, ["p0-b0", "p0-b1"], [plan([("First", ["p0-b0"]), ("Second", ["p0-b1"])])])
    result = await compose.run(ctx)
    assert len(ctx.llm.calls) == 1
    assert result["sections"] == 2


@pytest.mark.asyncio
async def test_one_missing_block_recovery_request_contains_only_missing_id(tmp_path: Path) -> None:
    ctx = seed(
        tmp_path,
        ["p0-b0", "p0-b1"],
        [
            plan([("First", ["p0-b0"])]),
            recovery([("Second", ["p0-b1"])]),
        ],
    )
    await compose.run(ctx)
    assert len(ctx.llm.calls) == 2
    assert [item["id"] for item in user_payload(ctx.llm.calls, 1)["missing_blocks"]] == ["p0-b1"]


@pytest.mark.asyncio
async def test_multiple_missing_blocks_recovered_in_one_request(tmp_path: Path) -> None:
    ctx = seed(
        tmp_path,
        ["p0-b0", "p0-b1", "p0-b2", "p0-b3"],
        [
            plan([("First", ["p0-b0"])]),
            recovery([("Rest", ["p0-b1", "p0-b2", "p0-b3"])]),
        ],
    )
    await compose.run(ctx)
    assert len(ctx.llm.calls) == 2
    payload = user_payload(ctx.llm.calls, 1)
    assert [item["id"] for item in payload["missing_blocks"]] == ["p0-b1", "p0-b2", "p0-b3"]


@pytest.mark.asyncio
async def test_successful_recovery_merges_sections_deterministically(tmp_path: Path) -> None:
    ctx = seed(
        tmp_path,
        ["p0-b0", "p0-b1", "p0-b2"],
        [
            plan([("First", ["p0-b0"])]),
            recovery([("Second", ["p0-b1"]), ("Third", ["p0-b2"])]),
        ],
    )
    await compose.run(ctx)
    stored = load_json(ctx.store, StageName.COMPOSE, "plan.json")
    assert [section["block_ids"] for section in stored["sections"]] == [["p0-b0", "p0-b1", "p0-b2"]]


@pytest.mark.asyncio
async def test_recovery_omitting_an_id_fails(tmp_path: Path) -> None:
    ctx = seed(
        tmp_path,
        ["p0-b0", "p0-b1", "p0-b2"],
        [
            plan([("First", ["p0-b0"])]),
            recovery([("Second", ["p0-b1"])]),
        ],
    )
    with pytest.raises(PipelineError, match="did not assign ids"):
        await compose.run(ctx)


@pytest.mark.asyncio
async def test_recovery_unknown_id_fails(tmp_path: Path) -> None:
    ctx = seed(
        tmp_path,
        ["p0-b0", "p0-b1"],
        [
            plan([("First", ["p0-b0"])]),
            recovery([("Second", ["p0-b9"])]),
        ],
    )
    with pytest.raises(PipelineError, match="non-missing id"):
        await compose.run(ctx)


@pytest.mark.asyncio
async def test_initial_duplicate_id_fails_without_recovery(tmp_path: Path) -> None:
    ctx = seed(
        tmp_path,
        ["p0-b0", "p0-b1"],
        [plan([("First", ["p0-b0", "p0-b0"])])],
    )
    with pytest.raises(PipelineError, match="duplicated"):
        await compose.run(ctx)
    assert len(ctx.llm.calls) == 1


@pytest.mark.asyncio
async def test_malformed_recovery_response_fails_safely(tmp_path: Path) -> None:
    ctx = seed(tmp_path, ["p0-b0", "p0-b1"], [plan([("First", ["p0-b0"])])])
    ctx.llm.responses.append({})  # type: ignore[attr-defined]
    with pytest.raises(PipelineError, match="recovery"):
        await compose.run(ctx)
    assert len(ctx.llm.calls) == 2


@pytest.mark.asyncio
async def test_recovery_request_failure_has_actionable_diagnostic(tmp_path: Path) -> None:
    ctx = seed(
        tmp_path,
        ["p0-b0", "p0-b1"],
        [plan([("First", ["p0-b0"])])],
        fail_on=2,
    )
    with pytest.raises(PipelineError, match="planner unavailable"):
        await compose.run(ctx)
    assert len(ctx.llm.calls) == 2


@pytest.mark.asyncio
async def test_original_assignments_are_unchanged_after_merge(tmp_path: Path) -> None:
    ctx = seed(
        tmp_path,
        ["p0-b0", "p0-b1"],
        [plan([("Original", ["p0-b0"])]), recovery([("Added", ["p0-b1"])])],
    )
    await compose.run(ctx)
    stored = load_json(ctx.store, StageName.COMPOSE, "plan.json")
    assert stored["sections"][0]["heading_en"] == "Original"
    assert stored["sections"][0]["block_ids"][0] == "p0-b0"
    assert stored["sections"][0]["block_ids"] == ["p0-b0", "p0-b1"]


@pytest.mark.asyncio
async def test_recovery_preserves_source_reading_order(tmp_path: Path) -> None:
    ctx = seed(
        tmp_path,
        ["p0-b0", "p0-b1", "p0-b2"],
        [
            plan([("Middle", ["p0-b1"])]),
            recovery([("Wrong", ["p0-b2", "p0-b0"])]),
        ],
        orders={"p0-b0": 0, "p0-b1": 1, "p0-b2": 2},
    )
    with pytest.raises(PipelineError, match="reading order"):
        await compose.run(ctx)


@pytest.mark.asyncio
async def test_recovery_cannot_generate_indian_language_text(tmp_path: Path) -> None:
    ctx = seed(
        tmp_path,
        ["p0-b0", "p0-b1"],
        [plan([("First", ["p0-b0"])]), recovery([("பகுதி", ["p0-b1"])])],
    )
    with pytest.raises(PipelineError, match="Indian-language text"):
        await compose.run(ctx)


@pytest.mark.asyncio
async def test_complete_plan_never_triggers_unnecessary_second_request(tmp_path: Path) -> None:
    ctx = seed(
        tmp_path,
        ["p0-b0", "p0-b1", "p0-b2"],
        [plan([("All", ["p0-b0", "p0-b1", "p0-b2"])])],
    )
    await compose.run(ctx)
    assert len(ctx.llm.calls) == 1


@pytest.mark.asyncio
async def test_recovery_cannot_cause_infinite_retries(tmp_path: Path) -> None:
    ctx = seed(
        tmp_path,
        ["p0-b0", "p0-b1"],
        [plan([("First", ["p0-b0"])]), recovery([("Second", ["p0-b1"])])],
        fail_on=2,
    )
    ctx.llm.responses.append(recovery([("Second", ["p0-b1"])]))  # type: ignore[attr-defined]
    with pytest.raises(PipelineError):
        await compose.run(ctx)
    assert len(ctx.llm.calls) == 2


@pytest.mark.asyncio
async def test_phase6ae_p6_b11_omission_is_recovered(tmp_path: Path) -> None:
    """Fixture reproducing the Phase 6AE plan that omitted p6-b11."""
    ctx = seed(
        tmp_path,
        ["p6-b10", "p6-b11"],
        [
            plan([("Cells", ["p6-b10"])]),
            recovery([("Cell structures", ["p6-b11"])]),
        ],
    )
    result = await compose.run(ctx)
    stored = load_json(ctx.store, StageName.COMPOSE, "plan.json")
    assert {block_id for section in stored["sections"] for block_id in section["block_ids"]} == {
        "p6-b10",
        "p6-b11",
    }
    assert result["sections"] == 1
    assert stored["sections"][0]["heading_en"] == "Cells"


@pytest.mark.asyncio
async def test_no_recovery_provenance_is_consistent(tmp_path: Path) -> None:
    ctx = seed(tmp_path, ["p0-b0", "p0-b1"], [plan([("All", ["p0-b0", "p0-b1"])])])
    await compose.run(ctx)
    stored = load_json(ctx.store, StageName.COMPOSE, "plan.json")
    provenance = stored["recovery"]
    assert provenance == {
        "used": False,
        "version": compose._RECOVERY_VERSION,
        "method": "source-order-insert-v1",
        "requests": 0,
        "original_block_ids": ["p0-b0", "p0-b1"],
        "recovered_block_ids": [],
    }


@pytest.mark.asyncio
async def test_missing_block_in_middle_is_inserted_by_source_order(tmp_path: Path) -> None:
    ctx = seed(
        tmp_path,
        ["p0-b0", "p0-b1", "p0-b2"],
        [
            plan([("Section", ["p0-b0", "p0-b2"])]),
            recovery([("Ignored heading", ["p0-b1"])]),
        ],
        orders={"p0-b0": 0, "p0-b1": 1, "p0-b2": 2},
    )
    await compose.run(ctx)
    stored = load_json(ctx.store, StageName.COMPOSE, "plan.json")
    assert stored["sections"] == [
        {"heading_en": "Section", "block_ids": ["p0-b0", "p0-b1", "p0-b2"]}
    ]


@pytest.mark.asyncio
async def test_ambiguous_placement_fails_with_diagnostic(tmp_path: Path) -> None:
    ctx = seed(
        tmp_path,
        ["p0-b0", "p0-b1", "p0-b2", "p0-b3"],
        [
            plan([("First", ["p0-b0"]), ("Last", ["p0-b3"])]),
            recovery([("Gap", ["p0-b1", "p0-b2"])]),
        ],
        orders={"p0-b0": 0, "p0-b1": 1, "p0-b2": 2, "p0-b3": 3},
    )
    with pytest.raises(PipelineError, match="ambiguous"):
        await compose.run(ctx)


@pytest.mark.asyncio
async def test_multiple_recovered_blocks_are_placed_in_distinct_sections(tmp_path: Path) -> None:
    ids = ["p0-b0", "p0-b1", "p0-b2", "p0-b3", "p0-b4", "p0-b5"]
    ctx = seed(
        tmp_path,
        ids,
        [
            plan([("First", ["p0-b0", "p0-b3"]), ("Second", ["p0-b4"])]),
            recovery([("Extra", ["p0-b1", "p0-b2", "p0-b5"])]),
        ],
        orders={block_id: index for index, block_id in enumerate(ids)},
    )
    await compose.run(ctx)
    stored = load_json(ctx.store, StageName.COMPOSE, "plan.json")
    assert [section["block_ids"] for section in stored["sections"]] == [
        ["p0-b0", "p0-b1", "p0-b2", "p0-b3"],
        ["p0-b4", "p0-b5"],
    ]


@pytest.mark.asyncio
async def test_recovery_provenance_records_ids_and_request_count(tmp_path: Path) -> None:
    ctx = seed(
        tmp_path,
        ["p0-b0", "p0-b1"],
        [plan([("First", ["p0-b0"])]), recovery([("Second", ["p0-b1"])])],
    )
    await compose.run(ctx)
    provenance = load_json(ctx.store, StageName.COMPOSE, "plan.json")["recovery"]
    assert provenance["used"] is True
    assert provenance["requests"] == 1
    assert provenance["original_block_ids"] == ["p0-b0"]
    assert provenance["recovered_block_ids"] == ["p0-b1"]
    assert provenance["version"] == compose._RECOVERY_VERSION


def test_malformed_provenance_is_rejected() -> None:
    with pytest.raises(PipelineError, match="provenance"):
        compose._validate_provenance({"used": "yes"})
    with pytest.raises(PipelineError, match="provenance"):
        compose._validate_provenance(
            {
                "used": True,
                "version": "v1",
                "method": "m",
                "requests": -1,
                "original_block_ids": [],
                "recovered_block_ids": [],
            }
        )


def test_legacy_plan_consumers_accept_optional_provenance() -> None:
    raw = plan([("First", ["p0-b0"])])
    known = {"p0-b0"}
    assert compose._validate_plan(raw, known).sections[0].block_ids == ("p0-b0",)
    with_provenance = dict(raw)
    with_provenance["recovery"] = compose._recovery_provenance(["p0-b0"], [], 0)
    assert compose._validate_plan(with_provenance, known).sections[0].block_ids == ("p0-b0",)


@pytest.mark.asyncio
async def test_phase6ae_later_page_block_is_not_appended(tmp_path: Path) -> None:
    """p6-b11 must be placed before later-page content, not after it."""
    ctx = seed(
        tmp_path,
        ["p6-b10", "p6-b11", "p6-b12"],
        [
            plan([("Cells", ["p6-b10", "p6-b12"])]),
            recovery([("Extra", ["p6-b11"])]),
        ],
        orders={"p6-b10": 0, "p6-b11": 1, "p6-b12": 2},
    )
    await compose.run(ctx)
    stored = load_json(ctx.store, StageName.COMPOSE, "plan.json")
    assert stored["sections"][0]["block_ids"] == ["p6-b10", "p6-b11", "p6-b12"]
