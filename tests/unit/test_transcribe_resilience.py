"""Offline regression tests for resilient Indic-Transcribe (Phase 6AP).

All network is mocked with ``httpx.MockTransport`` and provider fakes - no
Bodhan/Gemini calls, no keys, no credits. Covers header-less 429 backoff,
retry-after, pacing, and per-chunk checkpoint persistence/resume at both the
adapter and stage layers.
"""

from __future__ import annotations

import asyncio
import io
import json
import wave
from pathlib import Path
from typing import Any

import httpx
import pytest

from bodhan.adapters.transcribe import BodhanTranscribeAdapter, TranscriptionProviderError
from bodhan.domain.models import AudioSegment, Page, StageName
from bodhan.pipeline.errors import PipelineError
from bodhan.pipeline.stages import transcribe as stage
from bodhan.pipeline.state import load_json, save_json, save_pages
from tests.conftest import FakeTranscribe, make_context

BASE_URL = "https://api.test/v1/audio/transcriptions"
RATE_LIMIT_MESSAGE = "Maximum rate limit reached for the model"


def make_wav(seconds: float, *, rate: int = 8000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * int(seconds * rate))
    return buffer.getvalue()


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Capture asyncio sleeps (tenacity backoff + adapter pacing) without waiting."""
    recorded: list[float] = []
    real_sleep = asyncio.sleep

    async def fake_sleep(seconds: float, *args: Any, **kwargs: Any) -> None:
        recorded.append(float(seconds))
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    return recorded


def make_adapter(handler: Any, **kwargs: Any) -> BodhanTranscribeAdapter:
    defaults: dict[str, Any] = {
        "base_url": BASE_URL,
        "api_key": "test-key",
        "timeout_s": 5.0,
        "max_retries": 3,
    }
    defaults.update(kwargs)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return BodhanTranscribeAdapter(client=client, **defaults)  # type: ignore[arg-type]


def rate_limit() -> httpx.Response:
    return httpx.Response(429, json={"message": RATE_LIMIT_MESSAGE})


def text_ok(text: str = "पाठ") -> httpx.Response:
    return httpx.Response(200, json={"text": text})


def seed_narrate(ctx: Any, seconds: float = 65.0) -> list[AudioSegment]:
    directory = ctx.store.stage_dir(StageName.NARRATE.value)
    directory.mkdir(parents=True, exist_ok=True)
    save_pages(ctx.store, StageName.NARRATE, [Page(page_index=0, source_path="chapter.pdf")])
    path = directory / "seg-01.wav"
    path.write_bytes(make_wav(seconds))
    segment = AudioSegment(
        segment_id="seg-01",
        lesson_id="lesson-01",
        text="narration script (never the transcript)",
        audio_path=str(path),
    )
    save_json(ctx.store, StageName.NARRATE, "segments.json", [segment.model_dump()])
    return [segment]


def chunk_record(ctx: Any, index: int) -> dict[str, Any]:
    path = (
        ctx.store.stage_dir(StageName.TRANSCRIBE.value)
        / "chunks"
        / "seg-01"
        / f"chunk-{index:02d}.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def chunk_files(ctx: Any) -> list[Path]:
    directory = ctx.store.stage_dir(StageName.TRANSCRIBE.value) / "chunks" / "seg-01"
    return sorted(directory.glob("chunk-*.json")) if directory.is_dir() else []


class ChunkFake(FakeTranscribe):
    """Fails chunk requests whose filename contains any of ``fail_tokens``."""

    def __init__(self, *fail_tokens: str) -> None:
        super().__init__()
        self.fail_tokens = fail_tokens

    async def transcribe(self, audio_path: Path, *, language: str | None = None) -> Any:
        if any(token in audio_path.name for token in self.fail_tokens):
            raise TranscriptionProviderError("simulated chunk failure", status_code=503)
        return await super().transcribe(audio_path, language=language)


# --- adapter-level: retry, backoff, pacing ---------------------------------


async def test_429_without_retry_after_is_retried(tmp_path: Path, sleeps: list[float]) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return rate_limit() if len(calls) < 3 else text_ok("ठीक है")

    adapter = make_adapter(handler)
    wav = tmp_path / "seg-01.wav"
    wav.write_bytes(make_wav(1.0))

    result = await adapter.transcribe(wav, language="ta")
    await adapter.aclose()

    assert result.text == "ठीक है"
    assert len(calls) == 3  # header-less rate limit no longer terminal
    assert len(sleeps) >= 2  # bounded backoff actually waited between retries


async def test_429_with_retry_after_is_honoured(tmp_path: Path, sleeps: list[float]) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(
                429,
                json={"message": RATE_LIMIT_MESSAGE},
                headers={"retry-after": "5"},
            )
        return text_ok()

    adapter = make_adapter(handler)
    wav = tmp_path / "seg-01.wav"
    wav.write_bytes(make_wav(1.0))

    await adapter.transcribe(wav, language="ta")
    await adapter.aclose()

    assert len(calls) == 2
    assert sleeps and max(sleeps) >= 5.0  # provider delay honoured, capped at 60 s


async def test_exponential_backoff_is_increasing_and_bounded(
    tmp_path: Path, sleeps: list[float]
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return rate_limit()

    adapter = make_adapter(handler, max_retries=4)
    wav = tmp_path / "seg-01.wav"
    wav.write_bytes(make_wav(1.0))

    with pytest.raises(TranscriptionProviderError):
        await adapter.transcribe(wav, language="ta")
    await adapter.aclose()

    assert len(sleeps) == 3
    assert sleeps[0] < sleeps[1] < sleeps[2]
    assert max(sleeps) <= 30.0


async def test_maximum_retry_enforcement(tmp_path: Path, sleeps: list[float]) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return rate_limit()

    adapter = make_adapter(handler, max_retries=2)
    wav = tmp_path / "seg-01.wav"
    wav.write_bytes(make_wav(1.0))

    with pytest.raises(TranscriptionProviderError) as exc:
        await adapter.transcribe(wav, language="ta")
    await adapter.aclose()

    assert exc.value.status_code == 429
    assert len(calls) == 2  # initial attempt + exactly one retry, then stop
    assert len(sleeps) == 1


async def test_sequential_request_pacing(tmp_path: Path, sleeps: list[float]) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return text_ok(f"भाग{len(calls)}")

    adapter = make_adapter(handler, pace_s=0.05)
    wav = tmp_path / "seg-01.wav"
    wav.write_bytes(make_wav(65.0))  # 3 x 30 s chunks

    result = await adapter.transcribe(wav, language="ta")
    await adapter.aclose()

    assert len(calls) == 3
    assert sleeps == [0.05, 0.05, 0.05]  # one paced sleep per sequential request
    assert result.text == "भाग1 भाग2 भाग3"


async def test_budget_429_remains_terminal(tmp_path: Path, sleeps: list[float]) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(429, json={"message": "out of credit"})

    adapter = make_adapter(handler)
    wav = tmp_path / "seg-01.wav"
    wav.write_bytes(make_wav(1.0))

    with pytest.raises(TranscriptionProviderError):
        await adapter.transcribe(wav, language="ta")
    await adapter.aclose()

    assert len(calls) == 1  # explicit budget failure is never retried
    assert sleeps == []


# --- stage-level: chunk checkpoint persistence and resume -----------------


async def test_successful_chunk_checkpoints_are_persisted(tmp_path: Path) -> None:
    ctx = make_context(tmp_path, target="ta")
    seed_narrate(ctx)
    ctx.transcribe = FakeTranscribe()

    await stage.run(ctx)

    assert len(chunk_files(ctx)) == 3
    for index in (1, 2, 3):
        record = chunk_record(ctx, index)
        assert record["status"] == "ok"
        assert record["text"]
        identity = record["identity"]
        assert identity["chunk_index"] == index
        assert identity["chunk_count"] == 3
        assert identity["target_language"] == "ta"
        assert identity["chunk_sha256"]
        assert identity["start_frame"] < identity["end_frame"]


async def test_resume_after_interruption_retries_only_missing_chunk(tmp_path: Path) -> None:
    ctx = make_context(tmp_path, target="ta")
    seed_narrate(ctx)
    ctx.transcribe = ChunkFake("part03")

    with pytest.raises(PipelineError):
        await stage.run(ctx)

    assert [chunk_record(ctx, i)["status"] for i in (1, 2, 3)] == ["ok", "ok", "failed"]
    assert load_json(ctx.store, StageName.TRANSCRIBE, "manifest.json")["status"] == "failed"

    resumed = FakeTranscribe()
    ctx.transcribe = resumed
    await stage.run(ctx)

    assert len(resumed.calls) == 1
    assert "part03" in resumed.calls[0]
    assert [chunk_record(ctx, i)["status"] for i in (1, 2, 3)] == ["ok", "ok", "ok"]
    assert load_json(ctx.store, StageName.TRANSCRIBE, "manifest.json")["status"] == "done"


async def test_completed_chunks_are_not_call_twice(tmp_path: Path) -> None:
    ctx = make_context(tmp_path, target="ta")
    seed_narrate(ctx)
    ctx.transcribe = ChunkFake("part02")
    with pytest.raises(PipelineError):
        await stage.run(ctx)

    # Keep the failed segment record so the stage takes the chunk-resume path.
    resumed = FakeTranscribe()
    ctx.transcribe = resumed
    await stage.run(ctx)
    first_calls = list(resumed.calls)

    # Force the chunk path again by marking the (now ok) segment failed; every
    # chunk checkpoint remains valid and must be reused with zero provider calls.
    seg_path = ctx.store.stage_dir(StageName.TRANSCRIBE.value) / "segment-01.json"
    record = json.loads(seg_path.read_text(encoding="utf-8"))
    record["status"] = "failed"
    seg_path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")

    again = FakeTranscribe()
    ctx.transcribe = again
    await stage.run(ctx)

    assert len(first_calls) == 1  # only part02 was missing after the interruption
    assert again.calls == []  # no duplicate calls for completed chunks
    assert load_json(ctx.store, StageName.TRANSCRIBE, "manifest.json")["chunks_reused"] == 3


async def test_audio_change_invalidates_chunk_checkpoints(tmp_path: Path) -> None:
    ctx = make_context(tmp_path, target="ta")
    segments = seed_narrate(ctx)
    ctx.transcribe = FakeTranscribe()
    await stage.run(ctx)
    old_identity = chunk_record(ctx, 1)["identity"]

    Path(segments[0].audio_path).write_bytes(make_wav(70.0))  # same name, new audio
    changed = FakeTranscribe()
    ctx.transcribe = changed
    await stage.run(ctx)

    assert len(changed.calls) == 3  # all chunks retranscribed, none reused
    new_identity = chunk_record(ctx, 1)["identity"]
    assert (
        new_identity["audio_size_bytes"] != old_identity["audio_size_bytes"]
        or new_identity["audio_mtime_ns"] != old_identity["audio_mtime_ns"]
    )
    assert load_json(ctx.store, StageName.TRANSCRIBE, "manifest.json")["chunks_reused"] == 0


async def test_language_mismatch_invalidates_chunk_checkpoint(tmp_path: Path) -> None:
    ctx = make_context(tmp_path, target="ta")
    seed_narrate(ctx)
    ctx.transcribe = FakeTranscribe()
    await stage.run(ctx)

    path = ctx.store.stage_dir(StageName.TRANSCRIBE.value) / "segment-01.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["status"] = "failed"  # force the chunk-resume path
    path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    chunk_path = (
        ctx.store.stage_dir(StageName.TRANSCRIBE.value) / "chunks" / "seg-01" / "chunk-02.json"
    )
    chunk = json.loads(chunk_path.read_text(encoding="utf-8"))
    chunk["identity"]["target_language"] = "hi"
    chunk_path.write_text(json.dumps(chunk, ensure_ascii=False), encoding="utf-8")

    retried = FakeTranscribe()
    ctx.transcribe = retried
    await stage.run(ctx)

    assert len(retried.calls) == 1
    assert "part02" in retried.calls[0]
    assert chunk_record(ctx, 2)["identity"]["target_language"] == "ta"


async def test_chunks_are_reconstructed_in_exact_order(tmp_path: Path) -> None:
    ctx = make_context(tmp_path, target="ta")
    seed_narrate(ctx)
    ctx.transcribe = FakeTranscribe()

    await stage.run(ctx)

    text = ctx.store.stage_dir(StageName.TRANSCRIBE.value) / "segment-01.json"
    record = json.loads(text.read_text(encoding="utf-8"))
    assert record["transcription_text"] == (
        "[ta] seg-01.part01 का पाठ [ta] seg-01.part02 का पाठ [ta] seg-01.part03 का पाठ"
    )
    assert record["chunk_count"] == 3


async def test_middle_chunk_failure_keeps_other_completed_chunks(tmp_path: Path) -> None:
    ctx = make_context(tmp_path, target="ta")
    seed_narrate(ctx)
    ctx.transcribe = ChunkFake("part02")

    with pytest.raises(PipelineError):
        await stage.run(ctx)

    assert chunk_record(ctx, 1)["status"] == "ok"
    assert chunk_record(ctx, 2)["status"] == "failed"
    assert chunk_record(ctx, 3)["status"] == "ok"
    assert "simulated chunk failure" in chunk_record(ctx, 2)["error"]
    segment = json.loads(
        (ctx.store.stage_dir(StageName.TRANSCRIBE.value) / "segment-01.json").read_text(
            encoding="utf-8"
        )
    )
    assert segment["status"] == "failed"
    assert "chunk(s) failed" in segment["error"]


async def test_full_segment_reconstruction_after_resume(tmp_path: Path) -> None:
    ctx = make_context(tmp_path, target="ta")
    seed_narrate(ctx)
    ctx.transcribe = ChunkFake("part01", "part03")
    with pytest.raises(PipelineError):
        await stage.run(ctx)

    ctx.transcribe = FakeTranscribe()
    await stage.run(ctx)

    manifest = load_json(ctx.store, StageName.TRANSCRIBE, "manifest.json")
    assert manifest["status"] == "done"
    segment = json.loads(
        (ctx.store.stage_dir(StageName.TRANSCRIBE.value) / "segment-01.json").read_text(
            encoding="utf-8"
        )
    )
    assert segment["status"] == "ok"
    assert segment["chunk_count"] == 3
    assert segment["transcription_text"] == (
        "[ta] seg-01.part01 का पाठ [ta] seg-01.part02 का पाठ [ta] seg-01.part03 का पाठ"
    )


# --- compatibility with the existing 6AO checkpoints ----------------------


async def test_legacy_ok_segment_checkpoint_is_reused_verbatim(tmp_path: Path) -> None:
    ctx = make_context(tmp_path, target="ta")
    segments = seed_narrate(ctx, seconds=1.0)
    audio = Path(segments[0].audio_path)
    stat = audio.stat()
    directory = ctx.store.stage_dir(StageName.TRANSCRIBE.value)
    directory.mkdir(parents=True, exist_ok=True)
    legacy = {
        "section_id": "seg-01",
        "audio_file": audio.name,
        "audio_size_bytes": stat.st_size,
        "audio_mtime_ns": stat.st_mtime_ns,
        "target_language": "ta",
        "provider": "bodhan-transcribe",
        "model": "indic-transcribe",
        "transcription_text": "पुराना पाठ",
        "latency_ms": 100,
        "usage": {},
        "request_id": "legacy-req",
        "status": "ok",
        "raw": [{"text": "पुराना पाठ"}],
    }
    (directory / "segment-01.json").write_text(
        json.dumps(legacy, ensure_ascii=False), encoding="utf-8"
    )
    fake = FakeTranscribe()
    ctx.transcribe = fake

    await stage.run(ctx)

    assert fake.calls == []  # the real seg-01 checkpoint needs no provider call
    record = json.loads((directory / "segment-01.json").read_text(encoding="utf-8"))
    assert record["transcription_text"] == "पुराना पाठ"
    assert load_json(ctx.store, StageName.TRANSCRIBE, "manifest.json")["resumed_cached"] == 1


async def test_legacy_failed_segment_checkpoint_is_retranscribed(tmp_path: Path) -> None:
    ctx = make_context(tmp_path, target="ta")
    segments = seed_narrate(ctx, seconds=1.0)
    audio = Path(segments[0].audio_path)
    stat = audio.stat()
    directory = ctx.store.stage_dir(StageName.TRANSCRIBE.value)
    directory.mkdir(parents=True, exist_ok=True)
    legacy_failed = {
        "section_id": "seg-01",
        "audio_file": audio.name,
        "audio_size_bytes": stat.st_size,
        "audio_mtime_ns": stat.st_mtime_ns,
        "target_language": "ta",
        "provider": "bodhan-transcribe",
        "model": "indic-transcribe",
        "status": "failed",
        "error": "transcribe HTTP 429: Maximum rate limit reached for the model",
    }
    (directory / "segment-01.json").write_text(
        json.dumps(legacy_failed, ensure_ascii=False), encoding="utf-8"
    )
    fake = FakeTranscribe()
    ctx.transcribe = fake

    await stage.run(ctx)

    assert fake.calls == [str(audio)]
    record = json.loads((directory / "segment-01.json").read_text(encoding="utf-8"))
    assert record["status"] == "ok"
    assert record["transcription_text"] == "[ta] seg-01 का पाठ"
    assert len(chunk_files(ctx)) == 1  # short audio -> one chunk checkpoint
