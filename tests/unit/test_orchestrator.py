"""Tests for pipeline orchestration: ordering, checkpointing, error handling."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from bodhan.config import Settings
from bodhan.domain.models import Chapter, StageName
from bodhan.pipeline import PipelineError, run_pipeline
from bodhan.pipeline.orchestrator import STAGES
from bodhan.pipeline.stages.base import StageContext
from bodhan.services.artifact_store import FileArtifactStore


class FakeProvider:
    """Placeholder provider: orchestration tests never invoke it."""

    pass


def make_context(tmp_path: Path) -> StageContext:
    settings = Settings(project_root=tmp_path, data_dir="data")
    store = FileArtifactStore(tmp_path / "data" / "work" / "chapter-01")
    chapter = Chapter(
        chapter_id="chapter-01",
        title="Chapter 1",
        source_path=str(tmp_path / "chapter-01.pdf"),
    )
    return StageContext(
        settings=settings,
        store=store,
        chapter=chapter,
        ocr=FakeProvider(),  # type: ignore[arg-type]
        llm=FakeProvider(),  # type: ignore[arg-type]
        translate=FakeProvider(),  # type: ignore[arg-type]
        tts=FakeProvider(),  # type: ignore[arg-type]
        transcribe=FakeProvider(),  # type: ignore[arg-type]
        logger=logging.getLogger("test"),
    )


def test_stage_order_is_fixed() -> None:
    names = [name for name, _ in STAGES]
    assert names == [
        StageName.INGEST,
        StageName.PREPROCESS,
        StageName.EXTRACT,
        StageName.STRUCTURE,
        StageName.QUALITY,
        StageName.TRANSLATE,
        StageName.COMPOSE,
        StageName.NARRATE,
        StageName.TRANSCRIBE,
        StageName.EXPORT,
    ]


async def test_stage_failure_is_recorded_in_manifest(tmp_path: Path) -> None:
    ctx = make_context(tmp_path)  # source PDF deliberately does not exist

    with pytest.raises(PipelineError, match="source file not found"):
        await run_pipeline(ctx)

    entry = ctx.store.read_manifest()["stages"]["ingest"]
    assert entry["status"] == "failed"
    assert "not found" in entry["error"]


async def test_completed_stages_are_skipped(tmp_path: Path) -> None:
    ctx = make_context(tmp_path)
    ctx.store.write_manifest(
        {
            "chapter_id": "chapter-01",
            "stages": {name.value: {"status": "done"} for name, _ in STAGES},
        }
    )

    manifest = await run_pipeline(ctx)
    assert manifest["status"] == "done"
    assert len(manifest["stages"]) == len(STAGES)


async def test_force_reruns_completed_stage(tmp_path: Path) -> None:
    ctx = make_context(tmp_path)
    ctx.store.write_manifest({"stages": {StageName.INGEST.value: {"status": "done"}}})

    # force=True re-executes ingest, which fails because the source is missing.
    with pytest.raises(PipelineError):
        await run_pipeline(ctx, force=True)

    assert ctx.store.read_manifest()["stages"]["ingest"]["status"] == "failed"


async def test_only_runs_single_stage(tmp_path: Path) -> None:
    ctx = make_context(tmp_path)

    # extract cannot run: preprocess never produced pages.json.
    with pytest.raises(PipelineError, match="run stage 'preprocess' first"):
        await run_pipeline(ctx, only=StageName.EXTRACT)

    manifest = ctx.store.read_manifest()
    assert set(manifest["stages"]) == {"extract"}
    assert manifest["stages"]["extract"]["status"] == "failed"
