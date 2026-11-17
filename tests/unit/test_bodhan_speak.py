"""Contract tests for the Bodhan ``POST /v1/audio/speech`` adapter + policy.

Covers the documented request (``instructions`` as a JSON string), WAV
handling (chunking + concatenation), and the single-engine retry policy:
bounded retries for transient failures (honouring ``retry-after``), terminal
auth/validation errors, and the guarantee that no fallback TTS provider
exists anywhere in the package (assignment: Indic-Speak only).
"""

from __future__ import annotations

import io
import json
import wave
from pathlib import Path

import httpx
import pytest

from bodhan.adapters.tts import (
    BodhanSpeakAdapter,
    TTSProviderError,
    TTSResponseError,
    split_for_speech,
)
from bodhan.domain.models import AudioSegment

BASE_URL = "https://api.test/v1/audio/speech"


def make_wav(frames: int = 2400, *, rate: int = 24000, channels: int = 1) -> bytes:
    """Build a small PCM16 mono 24 kHz WAV (documented format)."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * frames * channels)
    return buffer.getvalue()


def make_adapter(**kwargs: object) -> BodhanSpeakAdapter:
    defaults: dict[str, object] = {
        "base_url": BASE_URL,
        "api_key": "test-key",
        "voice": "Kavya",
        "lang": "hi",
        "timeout_s": 5.0,
        "max_retries": 2,
        # Large chunk budget keeps most tests at a single request.
        "chunk_seconds": 30.0,
        "chars_per_second": 13.0,
    }
    defaults.update(kwargs)
    return BodhanSpeakAdapter(**defaults)  # type: ignore[arg-type]


# --- payload --------------------------------------------------------------


def test_payload_matches_documented_contract() -> None:
    adapter = make_adapter()
    payload = adapter._payload("नमस्ते दुनिया")  # type: ignore[attr-defined]

    assert set(payload) == {"model", "input", "voice", "instructions"}
    assert payload["model"] == "indic-speak"
    assert payload["input"] == "नमस्ते दुनिया"
    assert payload["voice"] == "Kavya"
    # `instructions` is a JSON *string*, not an object.
    assert isinstance(payload["instructions"], str)
    assert json.loads(payload["instructions"]) == {"lang": "hi"}


def test_payload_includes_style_when_configured() -> None:
    adapter = make_adapter(style="cheerful")
    payload = adapter._payload("hi")  # type: ignore[attr-defined]

    assert json.loads(payload["instructions"]) == {"lang": "hi", "style": "cheerful"}


def test_validation_rejects_missing_voice_or_lang() -> None:
    with pytest.raises(ValueError, match="voice"):
        make_adapter(voice="")
    with pytest.raises(ValueError, match="lang"):
        make_adapter(lang="")


# --- chunking -------------------------------------------------------------


def test_split_for_speech_respects_budget() -> None:
    text = " ".join(f"यह एक वाक्य है {i}।" for i in range(20))
    chunks = split_for_speech(text, 100)

    assert chunks
    assert all(0 < len(chunk) <= 100 for chunk in chunks)
    # Nothing is lost or duplicated.
    assert "".join(chunks).replace(" ", "") == text.replace(" ", "")


def test_split_for_speech_hard_splits_oversized_sentence() -> None:
    sentence = "word " * 50  # single sentence, no boundary inside
    chunks = split_for_speech(sentence.strip(), 40)

    assert all(len(chunk) <= 40 for chunk in chunks)
    assert "".join(chunks).replace(" ", "") == sentence.strip().replace(" ", "")


def test_split_for_speech_empty_input() -> None:
    assert split_for_speech("   ", 100) == []
    with pytest.raises(ValueError):
        split_for_speech("x", 0)


# --- WAV handling ---------------------------------------------------------


async def test_single_chunk_writes_wav(tmp_path: Path) -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, content=make_wav(), headers={"content-type": "audio/wav"}
        )
    )
    adapter = make_adapter(client=httpx.AsyncClient(transport=transport))
    out = tmp_path / "seg-01.wav"

    segment = await adapter.synthesize(
        "नमस्ते।", segment_id="seg-01", lesson_id="lesson-01", output_path=out
    )
    await adapter.aclose()

    assert isinstance(segment, AudioSegment)
    assert Path(segment.audio_path) == out
    assert segment.duration_ms == 100  # 2400 frames / 24000 Hz
    with wave.open(str(out), "rb") as handle:
        assert handle.getframerate() == 24000
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
    assert adapter.extension == ".wav"


async def test_multiple_chunks_are_concatenated(tmp_path: Path) -> None:
    # Force 3 chunks by shrinking the budget.
    adapter = make_adapter(
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, content=make_wav(), headers={"content-type": "audio/wav"}
                )
            )
        ),
        chunk_seconds=1.0,
        chars_per_second=1.0,  # max_chars = 1 per chunk
    )
    text = "one two three"
    out = tmp_path / "seg-01.wav"

    segment = await adapter.synthesize(
        text, segment_id="seg-01", lesson_id="lesson-01", output_path=out
    )
    await adapter.aclose()

    # max_chars = int(1.0 * 1.0) = 1 -> many single-char chunks, each 2400 frames.
    expected_chunks = split_for_speech(text, 1)
    assert len(expected_chunks) >= 3
    with wave.open(str(out), "rb") as handle:
        assert handle.getnframes() == 2400 * len(expected_chunks)
        assert handle.getframerate() == 24000
    assert segment.duration_ms == pytest.approx(100 * len(expected_chunks), abs=5)


async def test_non_audio_response_raises(tmp_path: Path) -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"detail": "oops"}))
    adapter = make_adapter(client=httpx.AsyncClient(transport=transport))

    with pytest.raises(TTSResponseError, match="audio"):
        await adapter.synthesize(
            "नमस्ते।", segment_id="seg-01", lesson_id="lesson-01", output_path=tmp_path / "x.wav"
        )
    await adapter.aclose()


async def test_undecodable_audio_raises(tmp_path: Path) -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, content=b"not a wav", headers={"content-type": "audio/wav"}
        )
    )
    adapter = make_adapter(client=httpx.AsyncClient(transport=transport))

    with pytest.raises(TTSResponseError, match="WAV"):
        await adapter.synthesize(
            "नमस्ते।", segment_id="seg-01", lesson_id="lesson-01", output_path=tmp_path / "x.wav"
        )
    await adapter.aclose()


async def test_empty_text_is_rejected(tmp_path: Path) -> None:
    adapter = make_adapter()

    with pytest.raises(ValueError, match="no text"):
        await adapter.synthesize(
            "   ", segment_id="seg-01", lesson_id="lesson-01", output_path=tmp_path / "x.wav"
        )
    await adapter.aclose()


# --- error handling (no fallback inside the adapter itself) ---------------


async def test_auth_error_is_terminal(tmp_path: Path) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(401, json={"message": "invalid api key"})

    adapter = make_adapter(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    with pytest.raises(TTSProviderError) as exc_info:
        await adapter.synthesize(
            "नमस्ते।", segment_id="seg-01", lesson_id="lesson-01", output_path=tmp_path / "x.wav"
        )
    await adapter.aclose()

    assert exc_info.value.status_code == 401
    assert len(calls) == 1


async def test_validation_error_is_terminal(tmp_path: Path) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(422, json={"detail": "invalid voice"})

    adapter = make_adapter(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    with pytest.raises(TTSProviderError) as exc_info:
        await adapter.synthesize(
            "नमस्ते।", segment_id="seg-01", lesson_id="lesson-01", output_path=tmp_path / "x.wav"
        )
    await adapter.aclose()

    assert exc_info.value.status_code == 422
    assert len(calls) == 1


async def test_quota_429_without_retry_after_is_terminal(tmp_path: Path) -> None:
    """Out-of-credit 429 (no documented ``retry-after``) must not retry."""
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(429, json={"message": "quota exceeded"})

    adapter = make_adapter(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    with pytest.raises(TTSProviderError) as exc_info:
        await adapter.synthesize(
            "नमस्ते।", segment_id="seg-01", lesson_id="lesson-01", output_path=tmp_path / "x.wav"
        )
    await adapter.aclose()

    assert exc_info.value.status_code == 429
    assert exc_info.value.retry_after is None
    assert len(calls) == 1


async def test_transient_5xx_is_retried_then_raises(tmp_path: Path) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503, json={"message": "busy"})

    adapter = make_adapter(
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), max_retries=2
    )
    with pytest.raises(TTSProviderError) as exc_info:
        await adapter.synthesize(
            "नमस्ते।", segment_id="seg-01", lesson_id="lesson-01", output_path=tmp_path / "x.wav"
        )
    await adapter.aclose()

    assert exc_info.value.status_code == 503
    assert len(calls) == 2


# --- single-engine retry policy + no-fallback guarantees ------------------


async def test_rate_limit_429_with_retry_after_is_retried(tmp_path: Path) -> None:
    """Documented rate-limit 429 (with ``retry-after``) waits out the window."""
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) < 2:
            return httpx.Response(429, json={"message": "rate limit"}, headers={"retry-after": "0"})
        return httpx.Response(200, content=make_wav(), headers={"content-type": "audio/wav"})

    adapter = make_adapter(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    segment = await adapter.synthesize(
        "hello.", segment_id="seg-01", lesson_id="lesson-01", output_path=tmp_path / "x.wav"
    )
    await adapter.aclose()

    assert len(calls) == 2
    assert Path(segment.audio_path).is_file()


async def test_rate_limit_429_retries_are_bounded(tmp_path: Path) -> None:
    """Even a persistent rate limit cannot loop forever (bounded budget)."""
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(429, json={"message": "rate limit"}, headers={"retry-after": "0"})

    adapter = make_adapter(
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), max_retries=2
    )
    with pytest.raises(TTSProviderError) as exc_info:
        await adapter.synthesize(
            "hello.", segment_id="seg-01", lesson_id="lesson-01", output_path=tmp_path / "x.wav"
        )
    await adapter.aclose()

    assert exc_info.value.status_code == 429
    assert exc_info.value.retry_after == 0.0
    assert len(calls) == 2  # initial attempt + one bounded retry


@pytest.mark.parametrize("status", [400, 403, 413])
async def test_validation_and_forbidden_statuses_are_never_retried(
    tmp_path: Path, status: int
) -> None:
    """400/403/413 fail on the first attempt - no indiscriminate retries."""
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(status, json={"message": "nope"})

    adapter = make_adapter(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    with pytest.raises(TTSProviderError) as exc_info:
        await adapter.synthesize(
            "hello.", segment_id="seg-01", lesson_id="lesson-01", output_path=tmp_path / "x.wav"
        )
    await adapter.aclose()

    assert exc_info.value.status_code == status
    assert len(calls) == 1


def test_no_fallback_provider_module_exists() -> None:
    """edge-tts and the fallback wrapper were deleted - nothing can import them."""
    import importlib.util

    for module in ("bodhan.adapters.tts.edge_tts", "bodhan.adapters.tts.fallback"):
        assert importlib.util.find_spec(module) is None, module


def test_tts_package_exports_no_fallback_symbols() -> None:
    import bodhan.adapters.tts as tts_package

    for symbol in ("FallbackTTS", "EdgeTTSAdapter", "should_fallback"):
        assert not hasattr(tts_package, symbol), symbol


def test_tts_port_has_no_fallback_api() -> None:
    from bodhan.ports.tts import TTSPort

    assert not hasattr(TTSPort, "fallback_count")
