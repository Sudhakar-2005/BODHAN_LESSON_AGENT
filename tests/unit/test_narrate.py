"""Narrate stage tests: preserve completed segments, structured failure, resume.

Task 1 guarantees proven here:
- Indic-Speak (the fake here) is the only engine - the stage records
  ``tts_provider`` in structured output and no fallback can ever run;
- on failure the completed audio segments are preserved on disk and in
  ``segments.json``;
- the failure is recorded in ``narrate/status.json`` (status, failed
  segment, error) and the stage fails loudly;
- a re-run resumes at the first missing segment instead of re-speaking
  the whole lesson.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bodhan.adapters.tts import TTSProviderError
from bodhan.domain.models import AudioSegment, Lesson, Page, StageName
from bodhan.pipeline.errors import PipelineError
from bodhan.pipeline.stages import narrate
from bodhan.pipeline.stages.base import StageContext
from bodhan.pipeline.state import save_json
from bodhan.ports.tts import TTSPort
from tests.conftest import make_context

SECTION_COUNT = 3
LESSON_MARKDOWN = "\n\n".join(
    f"## Section {index}\n\nBody text for section {index}." for index in range(1, 4)
)


class ScriptedTTS(TTSPort):
    """Fake single engine: succeeds until ``fail_at`` (segment id), then raises."""

    def __init__(self, fail_at: str | None = None) -> None:
        self.fail_at = fail_at
        self.calls: list[str] = []

    async def synthesize(
        self, text: str, *, segment_id: str, lesson_id: str, output_path: Path
    ) -> AudioSegment:
        self.calls.append(segment_id)
        if segment_id == self.fail_at:
            raise TTSProviderError("indic-speak unavailable", status_code=503)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"RIFF....WAVE fake audio")
        return AudioSegment(
            segment_id=segment_id,
            lesson_id=lesson_id,
            text=text,
            audio_path=str(output_path),
        )


def _seed_compose_artifacts(ctx: StageContext) -> None:
    """Minimal upstream artifacts so narrate can run in isolation."""
    save_json(
        ctx.store,
        StageName.COMPOSE,
        "pages.json",
        [Page(page_index=0, source_path="chapter.pdf").model_dump()],
    )
    lesson = Lesson(
        lesson_id="lesson-01",
        chapter_id="chapter-01",
        title="Chapter 1",
        markdown=LESSON_MARKDOWN,
        source_language_code="en",
        target_language_code="hi",
    )
    save_json(ctx.store, StageName.COMPOSE, "lesson.json", lesson)


def _status(ctx: StageContext) -> dict:
    path = ctx.store.stage_dir(narrate.NAME.value) / "status.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _segments(ctx: StageContext) -> list[dict]:
    path = ctx.store.stage_dir(narrate.NAME.value) / "segments.json"
    return json.loads(path.read_text(encoding="utf-8"))


async def test_narrate_success_writes_structured_status(tmp_path: Path) -> None:
    ctx = make_context(tmp_path, pages=1)
    _seed_compose_artifacts(ctx)
    tts = ScriptedTTS()
    ctx.tts = tts  # type: ignore[assignment]

    result = await narrate.run(ctx)

    assert result["segments"] == SECTION_COUNT
    assert result["tts_provider"] == "ScriptedTTS"
    assert tts.calls == [f"seg-{i:02d}" for i in range(1, SECTION_COUNT + 1)]

    status = _status(ctx)
    assert status["status"] == "done"
    assert status["tts_provider"] == "ScriptedTTS"
    assert status["completed_segments"] == [f"seg-{i:02d}" for i in range(1, 4)]
    assert status["total_sections"] == SECTION_COUNT
    assert "failed_segment" not in status
    assert len(_segments(ctx)) == SECTION_COUNT


async def test_narrate_failure_preserves_segments_and_records_error(tmp_path: Path) -> None:
    ctx = make_context(tmp_path, pages=1)
    _seed_compose_artifacts(ctx)
    ctx.tts = ScriptedTTS(fail_at="seg-02")  # type: ignore[assignment]

    with pytest.raises(PipelineError, match="seg-02"):
        await narrate.run(ctx)

    # Completed audio is preserved on disk and in segments.json.
    segments = _segments(ctx)
    assert [s["segment_id"] for s in segments] == ["seg-01"]
    narrate_dir = ctx.store.stage_dir(narrate.NAME.value)
    assert (narrate_dir / "seg-01.mp3").is_file()
    assert not (narrate_dir / "seg-02.mp3").exists()

    # The failure is recorded in structured output.
    status = _status(ctx)
    assert status["status"] == "failed"
    assert status["failed_segment"] == "seg-02"
    assert "TTSProviderError" in status["error"]
    assert status["completed_segments"] == ["seg-01"]
    assert status["total_sections"] == SECTION_COUNT
    assert status["skipped_empty"] == 0


async def test_narrate_resumes_at_first_missing_segment(tmp_path: Path) -> None:
    ctx = make_context(tmp_path, pages=1)
    _seed_compose_artifacts(ctx)

    # Attempt 1: fails at seg-02 (seg-01 completed and preserved).
    ctx.tts = ScriptedTTS(fail_at="seg-02")  # type: ignore[assignment]
    with pytest.raises(PipelineError, match="seg-02"):
        await narrate.run(ctx)

    # Attempt 2: the engine works again - only the missing segments run.
    resumed = ScriptedTTS()
    ctx.tts = resumed  # type: ignore[assignment]
    result = await narrate.run(ctx)

    assert resumed.calls == ["seg-02", "seg-03"]  # seg-01 was NOT re-spoken
    assert result["segments"] == SECTION_COUNT
    status = _status(ctx)
    assert status["status"] == "done"
    assert status["completed_segments"] == [f"seg-{i:02d}" for i in range(1, 4)]
    assert len(_segments(ctx)) == SECTION_COUNT


async def test_status_records_single_engine_only(tmp_path: Path) -> None:
    """No fallback provider can ever be recorded or invoked."""
    ctx = make_context(tmp_path, pages=1)
    _seed_compose_artifacts(ctx)
    ctx.tts = ScriptedTTS(fail_at="seg-03")  # type: ignore[assignment]

    with pytest.raises(PipelineError):
        await narrate.run(ctx)

    status = _status(ctx)
    provider = status["tts_provider"].lower()
    assert "edge" not in provider
    assert "fallback" not in provider
    assert provider == "scriptedtts"  # exactly the configured engine


async def test_missing_audio_forces_resynthesis(tmp_path: Path) -> None:
    """A preserved entry whose file was deleted must be spoken again."""
    ctx = make_context(tmp_path, pages=1)
    _seed_compose_artifacts(ctx)
    ctx.tts = ScriptedTTS()  # type: ignore[assignment]
    await narrate.run(ctx)

    narrate_dir = ctx.store.stage_dir(narrate.NAME.value)
    (narrate_dir / "seg-01.mp3").unlink()

    tts = ScriptedTTS()
    ctx.tts = tts  # type: ignore[assignment]
    result = await narrate.run(ctx)

    assert tts.calls == ["seg-01"]  # only the missing file was regenerated
    assert result["segments"] == SECTION_COUNT
