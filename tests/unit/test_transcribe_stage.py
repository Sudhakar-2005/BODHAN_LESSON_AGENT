"""Stage tests for transcribe: checkpointing, resume, integrity (Phase 2A).

Covers the assignment rules at stage level: every documented record field is
written, Hindi/Tamil/Bengali target languages propagate untouched, provider
text stays verbatim (never translated, never the English narration script,
never through the LLM), a failed segment preserves earlier successes and is
the ONLY thing retranscribed on resume, and the stage can never be marked
successful while a required segment is missing.
"""

from __future__ import annotations

import io
import json
import wave
from pathlib import Path

import pytest

from bodhan.adapters.transcribe import TranscriptionProviderError
from bodhan.domain.models import AudioSegment, Page, StageName
from bodhan.pipeline.errors import PipelineError
from bodhan.pipeline.stages import transcribe as transcribe_stage
from bodhan.pipeline.stages.base import StageContext
from bodhan.pipeline.state import load_json, save_json, save_pages
from tests.conftest import FakeTranscribe, make_context

RECORD_DIR = StageName.TRANSCRIBE.value
REQUIRED_FIELDS = {
    "section_id",
    "audio_file",
    "target_language",
    "provider",
    "model",
    "transcription_text",
    "latency_ms",
    "usage",
    "request_id",
    "status",
}


def make_wav(seconds: float = 0.5, *, rate: int = 24000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * int(seconds * rate))
    return buffer.getvalue()


def seed_narrate(ctx: StageContext, count: int = 4) -> list[AudioSegment]:
    """Create the narrate artifacts (pages + segments + WAV files) this stage reads."""
    narrate_dir = ctx.store.stage_dir(StageName.NARRATE.value)
    narrate_dir.mkdir(parents=True, exist_ok=True)
    save_pages(ctx.store, StageName.NARRATE, [Page(page_index=0, source_path="chapter.pdf")])
    segments: list[AudioSegment] = []
    for index in range(1, count + 1):
        path = narrate_dir / f"seg-{index:02d}.wav"
        path.write_bytes(make_wav())
        segments.append(
            AudioSegment(
                segment_id=f"seg-{index:02d}",
                lesson_id="lesson-01",
                text=f"narration script {index}",  # English source: never the expected text
                audio_path=str(path),
            )
        )
    save_json(ctx.store, StageName.NARRATE, "segments.json", [s.model_dump() for s in segments])
    return segments


def record_of(ctx: StageContext, index: int) -> dict:
    path = ctx.store.stage_dir(RECORD_DIR) / f"segment-{index:02d}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def manifest_of(ctx: StageContext) -> dict:
    return load_json(ctx.store, StageName.TRANSCRIBE, "manifest.json")


class FlakyTranscribe(FakeTranscribe):
    """Fails one designated segment (by audio stem) and succeeds elsewhere."""

    def __init__(self, fail_for: str) -> None:
        super().__init__()
        self.fail_for = fail_for

    async def transcribe(self, audio_path: Path, *, language: str | None = None):
        if audio_path.stem == self.fail_for:
            raise TranscriptionProviderError("simulated provider failure", status_code=503)
        return await super().transcribe(audio_path, language=language)


# --- success path ----------------------------------------------------------


async def test_success_writes_every_documented_field(tmp_path: Path) -> None:
    ctx = make_context(tmp_path)
    segments = seed_narrate(ctx, 3)
    fake = FakeTranscribe()
    ctx.transcribe = fake

    result = await transcribe_stage.run(ctx)

    assert result["segments"] == 3
    assert result["transcribed"] == 3
    assert result["provider"] == "fake-transcribe"
    # The exact files produced by Indic-Speak are what got transcribed.
    assert fake.calls == [str(Path(segment.audio_path)) for segment in segments]
    assert fake.languages == ["hi", "hi", "hi"]

    record = record_of(ctx, 1)
    assert set(record) >= REQUIRED_FIELDS
    assert record["section_id"] == "seg-01"
    assert record["audio_file"] == "seg-01.wav"
    assert record["target_language"] == "hi"
    assert record["provider"] == "fake-transcribe"
    assert record["model"] == "indic-transcribe"
    assert record["status"] == "ok"
    assert "error" not in record  # recorded only where available
    assert record["transcription_text"] == "[hi] seg-01 का पाठ"
    assert record["latency_ms"] == 7
    assert record["usage"] == {"audio_seconds": 0.5}
    assert record["request_id"] == "req-seg-01"
    assert record["raw"] == [{"text": "[hi] seg-01 का पाठ"}]
    assert record["audio_size_bytes"] == Path(segments[0].audio_path).stat().st_size
    # pages propagate to the new stage directory (existing architecture).
    assert (ctx.store.stage_dir(RECORD_DIR) / "pages.json").is_file()


async def test_manifest_summarizes_metadata_usage_and_latency(tmp_path: Path) -> None:
    ctx = make_context(tmp_path)
    seed_narrate(ctx, 3)
    ctx.transcribe = FakeTranscribe()

    await transcribe_stage.run(ctx)

    manifest = manifest_of(ctx)
    assert manifest["status"] == "done"
    assert manifest["target_language"] == "hi"
    assert manifest["provider"] == "fake-transcribe"
    assert manifest["model"] == "indic-transcribe"
    assert manifest["total_segments"] == 3
    assert manifest["completed_segments"] == ["seg-01", "seg-02", "seg-03"]
    assert manifest["failed_segments"] == []
    assert manifest["usage"] == {"audio_seconds": 1.5}
    assert manifest["total_latency_ms"] == 21
    assert [entry["section_id"] for entry in manifest["segments"]] == [
        "seg-01",
        "seg-02",
        "seg-03",
    ]


# --- Task 4: language + data integrity -------------------------------------


@pytest.mark.parametrize("language", ["hi", "ta", "bn"])
async def test_target_language_propagates_verbatim(tmp_path: Path, language: str) -> None:
    ctx = make_context(tmp_path, target=language)
    seed_narrate(ctx, 2)
    fake = FakeTranscribe()
    ctx.transcribe = fake

    result = await transcribe_stage.run(ctx)

    assert fake.languages == [language, language]
    assert result["target_language"] == language
    for index in (1, 2):
        record = record_of(ctx, index)
        assert record["target_language"] == language
        # Provider output kept as-is: prefix marks the language it was asked for.
        assert record["transcription_text"] == f"[{language}] seg-{index:02d} का पाठ"


async def test_provider_text_is_verbatim_and_separate_from_narration(tmp_path: Path) -> None:
    ctx = make_context(tmp_path)
    segments = seed_narrate(ctx, 2)
    narration_before = load_json(ctx.store, StageName.NARRATE, "segments.json")
    ctx.transcribe = FakeTranscribe()

    await transcribe_stage.run(ctx)

    record = record_of(ctx, 1)
    # Verbatim provider output, unrelated to the (English) narration script.
    assert record["transcription_text"] == "[hi] seg-01 का पाठ"
    assert record["transcription_text"] != segments[0].text
    assert segments[0].text == "narration script 1"
    # No translation, no LLM rewrite: neither provider was touched.
    assert ctx.translate.calls == []  # type: ignore[attr-defined]
    assert ctx.llm.calls == []  # type: ignore[attr-defined]
    # The narration script itself was not modified or merged into transcribe/.
    assert load_json(ctx.store, StageName.NARRATE, "segments.json") == narration_before
    assert "narration_script" not in record


# --- checkpoint / resume ---------------------------------------------------


async def test_second_run_never_retranscribes_completed_segments(tmp_path: Path) -> None:
    ctx = make_context(tmp_path)
    seed_narrate(ctx, 3)
    fake = FakeTranscribe()
    ctx.transcribe = fake

    await transcribe_stage.run(ctx)
    first_calls = list(fake.calls)
    first_record = record_of(ctx, 2)

    await transcribe_stage.run(ctx)

    assert len(first_calls) == 3
    assert fake.calls == first_calls  # zero repeat transcription
    assert manifest_of(ctx)["resumed_cached"] == 3
    assert record_of(ctx, 2) == first_record  # records untouched on resume


async def test_segment_four_failure_preserves_earlier_work_and_resumes(
    tmp_path: Path,
) -> None:
    """Evidence: segments 1-3 are NOT retranscribed when the run resumes."""
    ctx = make_context(tmp_path, chapter_id="resume-chapter")
    segments = seed_narrate(ctx, 4)
    ctx.transcribe = FlakyTranscribe("seg-04")

    with pytest.raises(PipelineError, match="re-run to resume"):
        await transcribe_stage.run(ctx)

    manifest = manifest_of(ctx)
    assert manifest["status"] == "failed"
    assert manifest["completed_segments"] == ["seg-01", "seg-02", "seg-03"]
    assert manifest["failed_segments"][0]["section_id"] == "seg-04"
    for index in (1, 2, 3):
        assert record_of(ctx, index)["status"] == "ok"
    failed = record_of(ctx, 4)
    assert failed["status"] == "failed"
    assert "simulated provider failure" in failed["error"]

    # Resume with a healthy provider: ONLY the failed segment is called.
    resumed = FakeTranscribe()
    ctx.transcribe = resumed
    await transcribe_stage.run(ctx)

    assert resumed.calls == [str(Path(segments[3].audio_path))]
    manifest = manifest_of(ctx)
    assert manifest["status"] == "done"
    assert manifest["completed_segments"] == ["seg-01", "seg-02", "seg-03", "seg-04"]
    assert manifest["resumed_cached"] == 3
    assert record_of(ctx, 1)["transcription_text"] == "[hi] seg-01 का पाठ"  # kept verbatim


async def test_replaced_audio_is_retranscribed(tmp_path: Path) -> None:
    ctx = make_context(tmp_path)
    segments = seed_narrate(ctx, 2)
    fake = FakeTranscribe()
    ctx.transcribe = fake

    await transcribe_stage.run(ctx)
    # Narration regenerated: same file name, new content.
    Path(segments[1].audio_path).write_bytes(make_wav(0.75))
    await transcribe_stage.run(ctx)

    assert fake.calls == [
        str(Path(segments[0].audio_path)),
        str(Path(segments[1].audio_path)),
        str(Path(segments[1].audio_path)),
    ]


# --- never success with missing segments -----------------------------------


async def test_missing_audio_never_marks_stage_success(tmp_path: Path) -> None:
    ctx = make_context(tmp_path)
    segments = seed_narrate(ctx, 2)
    Path(segments[1].audio_path).unlink()
    ctx.transcribe = FakeTranscribe()

    with pytest.raises(PipelineError, match="1/2"):
        await transcribe_stage.run(ctx)

    manifest = manifest_of(ctx)
    assert manifest["status"] == "failed"
    assert manifest["segments"][1]["status"] == "failed"
    assert "FileNotFoundError" in manifest["segments"][1]["error"]
    assert record_of(ctx, 1)["status"] == "ok"  # the good work is preserved
    assert record_of(ctx, 2)["status"] == "failed"


async def test_without_narrate_output_the_stage_fails(tmp_path: Path) -> None:
    ctx = make_context(tmp_path)
    ctx.transcribe = FakeTranscribe()

    with pytest.raises(PipelineError, match="run stage 'narrate' first"):
        await transcribe_stage.run(ctx)


async def test_undocumented_target_language_rejected_before_any_call(tmp_path: Path) -> None:
    ctx = make_context(tmp_path, target="ks-Deva")  # translation enum only
    seed_narrate(ctx, 2)
    fake = FakeTranscribe()
    ctx.transcribe = fake

    with pytest.raises(PipelineError, match="documented transcription enum"):
        await transcribe_stage.run(ctx)

    assert fake.calls == []
