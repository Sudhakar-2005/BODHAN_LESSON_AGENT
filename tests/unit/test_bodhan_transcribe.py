"""Contract tests for the Bodhan ``POST /v1/audio/transcriptions`` adapter.

Covers the verified official contract (multipart upload with the documented
``model``/``language``/``file`` fields, ``{"text": "..."}`` response parsing,
the <= 30 s WAV splitting rule, captured latency/request-id/usage), the
single-provider retry policy (bounded retries honouring ``retry-after``,
terminal auth/validation/quota errors) and the guarantee that no alternative
speech-to-text provider can ever be imported or invoked (assignment: Bodhan
Indic-Transcribe only).

All HTTP is mocked with ``httpx.MockTransport`` - no network, no keys.
"""

from __future__ import annotations

import io
import re
import wave
from pathlib import Path

import httpx
import pytest

import bodhan
from bodhan.adapters.transcribe import (
    BodhanTranscribeAdapter,
    EmptyTranscriptionError,
    TranscriptionProviderError,
    TranscriptionResponseError,
    UnsupportedLanguageError,
    split_wav,
)
from bodhan.ports.transcribe import SUPPORTED_TRANSCRIBE_LANGUAGES

BASE_URL = "https://api.test/v1/audio/transcriptions"


def make_wav(seconds: float, *, rate: int = 24000, channels: int = 1) -> bytes:
    """Build a PCM16 WAV (the documented Indic-Speak output format)."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * int(seconds * rate) * channels)
    return buffer.getvalue()


def make_adapter(**kwargs: object) -> BodhanTranscribeAdapter:
    defaults: dict[str, object] = {
        "base_url": BASE_URL,
        "api_key": "test-key",
        "timeout_s": 5.0,
        "max_retries": 2,
    }
    defaults.update(kwargs)
    return BodhanTranscribeAdapter(**defaults)  # type: ignore[arg-type]


def text_response(text: str, *, request_id: str | None = None, **extra: object) -> httpx.Response:
    headers = {"content-type": "application/json"}
    if request_id is not None:
        headers["x-request-id"] = request_id
    return httpx.Response(200, json={"text": text, **extra}, headers=headers)


def multipart_field(body: bytes, name: str) -> bytes | None:
    """Value of a scalar multipart form field, or ``None`` when absent."""
    match = re.search(
        rb'name="' + name.encode() + rb'"(?:;[^)]*)?\r\n\r\n([^\r]+)\r\n',
        body,
    )
    return match.group(1) if match else None


# --- documented request contract -----------------------------------------


async def test_success_parses_documented_response(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return text_response("नमस्ते दुनिया", request_id="req-123")

    adapter = make_adapter(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    wav = tmp_path / "seg-01.wav"
    wav.write_bytes(make_wav(1.0))

    result = await adapter.transcribe(wav, language="hi")
    await adapter.aclose()

    assert result.text == "नमस्ते दुनिया"  # verbatim, in the spoken language
    assert result.model == "indic-transcribe"
    assert result.provider == "bodhan-transcribe"
    assert result.language == "hi"
    assert result.status_code == 200
    assert result.request_id == "req-123"
    assert result.latency_ms >= 0
    assert result.usage == {}  # documented responses carry no usage
    assert result.raw == [{"text": "नमस्ते दुनिया"}]  # raw provider body preserved


async def test_request_is_multipart_with_documented_fields(tmp_path: Path) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return text_response("पाठ")

    adapter = make_adapter(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    wav = tmp_path / "seg-01.wav"
    wav.write_bytes(make_wav(1.0))

    await adapter.transcribe(wav)  # no language -> documented auto-detect
    await adapter.aclose()

    assert len(seen) == 1
    request = seen[0]
    assert request.headers["content-type"].startswith("multipart/form-data")
    assert request.headers["authorization"] == "Bearer test-key"
    body = request.content
    assert multipart_field(body, "model") == b"indic-transcribe"
    assert multipart_field(body, "language") is None  # omitted, not guessed
    assert b'name="file"' in body
    assert b'filename="seg-01.wav"' in body
    assert b"RIFF" in body  # the actual WAV payload travels in the body


@pytest.mark.parametrize("language", ["hi", "ta", "bn"])
async def test_documented_language_is_sent(tmp_path: Path, language: str) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return text_response("...")

    adapter = make_adapter(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    wav = tmp_path / "seg-01.wav"
    wav.write_bytes(make_wav(1.0))

    result = await adapter.transcribe(wav, language=language)
    await adapter.aclose()

    assert language in SUPPORTED_TRANSCRIBE_LANGUAGES
    assert multipart_field(seen[0].content, "language") == language.encode()
    assert result.language == language


async def test_undocumented_language_fails_without_any_request(tmp_path: Path) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return text_response("...")

    adapter = make_adapter(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    wav = tmp_path / "seg-01.wav"
    wav.write_bytes(make_wav(1.0))

    # ks-Deva exists in the translation enum, not in the STT enum.
    with pytest.raises(UnsupportedLanguageError):
        await adapter.transcribe(wav, language="ks-Deva")
    await adapter.aclose()

    assert calls == []


async def test_missing_audio_file_stays_a_local_failure(tmp_path: Path) -> None:
    adapter = make_adapter()

    with pytest.raises(FileNotFoundError):
        await adapter.transcribe(tmp_path / "nope.wav")
    await adapter.aclose()


# --- documented 30 s ceiling ----------------------------------------------


def test_split_wav_respects_documented_ceiling() -> None:
    original = make_wav(65.0)
    pieces = split_wav(original, 30.0)

    assert len(pieces) == 3  # 30 + 30 + 5
    durations = []
    total_frames = 0
    for piece in pieces:
        with wave.open(io.BytesIO(piece), "rb") as handle:
            durations.append(handle.getnframes() / handle.getframerate())
            total_frames += handle.getnframes()
    assert all(duration <= 30.0 for duration in durations)
    with wave.open(io.BytesIO(original), "rb") as handle:
        assert total_frames == handle.getnframes()  # no audio lost or duplicated


def test_split_wav_passes_short_and_non_wav_payloads_through() -> None:
    short = make_wav(12.5)
    assert split_wav(short, 30.0) == [short]
    exact = make_wav(30.0)
    assert split_wav(exact, 30.0) == [exact]  # "up to 30 seconds"
    not_wav = b"ID3\x03\x00fake-mp3-bytes"
    assert split_wav(not_wav, 30.0) == [not_wav]  # cannot measure -> provider decides
    with pytest.raises(ValueError):
        split_wav(short, 0)


async def test_audio_within_ceiling_is_uploaded_unchanged(tmp_path: Path) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return text_response("छोटा ऑडियो", request_id="req-1")

    adapter = make_adapter(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    wav = tmp_path / "seg-02.wav"
    wav.write_bytes(make_wav(20.0))

    result = await adapter.transcribe(wav, language="hi")
    await adapter.aclose()

    assert len(seen) == 1
    assert b'filename="seg-02.wav"' in seen[0].content
    assert result.text == "छोटा ऑडियो"


async def test_long_audio_is_split_joined_and_metadata_accumulated(tmp_path: Path) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        index = len(seen) + 1
        seen.append(request)
        return text_response(f"टुकड़ा{index}", request_id=f"req-{index}")

    adapter = make_adapter(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    wav = tmp_path / "seg-03.wav"
    wav.write_bytes(make_wav(65.0))

    result = await adapter.transcribe(wav, language="hi")
    await adapter.aclose()

    assert len(seen) == 3  # <= 30 s per documented request
    assert b'filename="seg-03.part1.wav"' in seen[0].content
    assert b'filename="seg-03.part2.wav"' in seen[1].content
    assert b'filename="seg-03.part3.wav"' in seen[2].content
    # Provider texts joined verbatim, never rewritten.
    assert result.text == "टुकड़ा1 टुकड़ा2 टुकड़ा3"
    assert result.request_id == "req-1,req-2,req-3"
    assert result.raw == [{"text": "टुकड़ा1"}, {"text": "टुकड़ा2"}, {"text": "टुकड़ा3"}]
    assert result.latency_ms >= 0


# --- error handling and retry policy --------------------------------------


async def test_auth_failure_is_terminal(tmp_path: Path) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(401, json={"error": {"message": "invalid api key"}})

    adapter = make_adapter(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    wav = tmp_path / "seg-01.wav"
    wav.write_bytes(make_wav(1.0))

    with pytest.raises(TranscriptionProviderError) as exc_info:
        await adapter.transcribe(wav, language="hi")
    await adapter.aclose()

    assert exc_info.value.status_code == 401
    assert len(calls) == 1  # auth never retries, never falls back


@pytest.mark.parametrize("status", [400, 413, 415, 422])
async def test_validation_and_unsupported_media_are_terminal(tmp_path: Path, status: int) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(status, json={"detail": "rejected"})

    adapter = make_adapter(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    wav = tmp_path / "seg-01.wav"
    wav.write_bytes(make_wav(1.0))

    with pytest.raises(TranscriptionProviderError) as exc_info:
        await adapter.transcribe(wav)
    await adapter.aclose()

    assert exc_info.value.status_code == status
    assert len(calls) == 1


async def test_rate_limit_429_with_retry_after_is_retried(tmp_path: Path) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(
                429,
                json={"message": "rate limit"},
                headers={"retry-after": "0", "x-ratelimit-limit-requests": "8"},
            )
        return text_response("अब ठीक है", request_id="req-2")

    adapter = make_adapter(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    wav = tmp_path / "seg-01.wav"
    wav.write_bytes(make_wav(1.0))

    result = await adapter.transcribe(wav, language="hi")
    await adapter.aclose()

    assert len(calls) == 2  # waited out the documented window
    assert result.text == "अब ठीक है"
    # Published limits are read from the response, never hardcoded.
    assert adapter.rate_limit is not None
    assert adapter.rate_limit.limit_requests == 8


async def test_quota_429_without_retry_after_is_terminal(tmp_path: Path) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(429, json={"message": "out of credit"})

    adapter = make_adapter(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    wav = tmp_path / "seg-01.wav"
    wav.write_bytes(make_wav(1.0))

    with pytest.raises(TranscriptionProviderError) as exc_info:
        await adapter.transcribe(wav)
    await adapter.aclose()

    assert exc_info.value.status_code == 429
    assert exc_info.value.retry_after is None
    assert len(calls) == 1  # a permanent quota failure must not loop


async def test_transient_5xx_is_retried_then_raises(tmp_path: Path) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503, json={"detail": "busy"})

    adapter = make_adapter(
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), max_retries=2
    )
    wav = tmp_path / "seg-01.wav"
    wav.write_bytes(make_wav(1.0))

    with pytest.raises(TranscriptionProviderError) as exc_info:
        await adapter.transcribe(wav)
    await adapter.aclose()

    assert exc_info.value.status_code == 503
    assert len(calls) == 2  # bounded: initial attempt + one retry


@pytest.mark.parametrize("failure", [httpx.ReadTimeout, httpx.ConnectError])
async def test_timeout_and_network_failures_are_retried_then_raise(
    tmp_path: Path, failure: type[httpx.TransportError]
) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        raise failure("simulated transport failure")

    adapter = make_adapter(
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), max_retries=2
    )
    wav = tmp_path / "seg-01.wav"
    wav.write_bytes(make_wav(1.0))

    with pytest.raises(TranscriptionProviderError, match="request failed"):
        await adapter.transcribe(wav)
    await adapter.aclose()

    assert len(calls) == 2


# --- malformed / empty responses never become transcriptions ---------------


async def test_non_json_response_is_rejected(tmp_path: Path) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200, text="<html>oops</html>")

    adapter = make_adapter(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    wav = tmp_path / "seg-01.wav"
    wav.write_bytes(make_wav(1.0))

    with pytest.raises(TranscriptionResponseError, match="non-JSON"):
        await adapter.transcribe(wav)
    await adapter.aclose()

    assert len(calls) == 1  # never retried blindly


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({}, "missing 'text'"),
        ({"detail": "nope"}, "missing 'text'"),
        ({"error": {"message": "boom"}}, "error payload"),
    ],
)
async def test_malformed_success_bodies_are_rejected(
    tmp_path: Path, payload: dict[str, object], match: str
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    adapter = make_adapter(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    wav = tmp_path / "seg-01.wav"
    wav.write_bytes(make_wav(1.0))

    with pytest.raises(TranscriptionResponseError, match=match):
        await adapter.transcribe(wav)
    await adapter.aclose()


async def test_empty_transcription_is_rejected(tmp_path: Path) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return text_response("   ")

    adapter = make_adapter(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    wav = tmp_path / "seg-01.wav"
    wav.write_bytes(make_wav(1.0))

    with pytest.raises(EmptyTranscriptionError):
        await adapter.transcribe(wav)
    await adapter.aclose()

    assert len(calls) == 1


async def test_usage_is_captured_only_when_the_provider_sends_it(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return text_response("पाठ", usage={"audio_seconds": 4.5, "bad": "x"})

    adapter = make_adapter(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    wav = tmp_path / "seg-01.wav"
    wav.write_bytes(make_wav(1.0))

    result = await adapter.transcribe(wav, language="hi")
    await adapter.aclose()

    assert result.usage == {"audio_seconds": 4.5}  # numbers only, nothing invented


async def test_non_wav_audio_is_uploaded_as_is(tmp_path: Path) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return text_response("mp3 ऑडियो")

    adapter = make_adapter(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    mp3 = tmp_path / "seg-09.mp3"
    mp3.write_bytes(b"ID3\x03\x00\x00\x00fake-mp3")

    result = await adapter.transcribe(mp3, language="hi")
    await adapter.aclose()

    assert len(seen) == 1
    assert b'filename="seg-09.mp3"' in seen[0].content
    assert multipart_field(seen[0].content, "language") == b"hi"
    assert result.text == "mp3 ऑडियो"


def test_adapter_configuration_is_validated() -> None:
    with pytest.raises(ValueError, match="base_url"):
        make_adapter(base_url="")
    with pytest.raises(ValueError, match="model"):
        make_adapter(model="")
    with pytest.raises(ValueError, match="30"):
        make_adapter(max_audio_seconds=31.0)  # beyond the documented ceiling


# --- no alternative STT provider, ever (assignment rule) -------------------

_FORBIDDEN_IMPORT_PREFIXES = (
    "whisper",
    "deepgram",
    "vosk",
    "assemblyai",
    "speech_recognition",
    "rev_ai",
    "ibm_watson",
    "google.cloud.speech",
    "azure.cognitiveservices.speech",
    "azure.ai.speech",
)
_IMPORT_LINE = re.compile(r"^\s*(?:from|import)\s+([A-Za-z0-9_.]+)", re.MULTILINE)


def test_source_contains_no_alternative_stt_imports() -> None:
    """No module anywhere under src/bodhan imports another STT provider."""
    root = Path(bodhan.__file__).parent
    offenders: list[tuple[Path, str]] = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for match in _IMPORT_LINE.finditer(text):
            module = match.group(1)
            if module.startswith(_FORBIDDEN_IMPORT_PREFIXES):
                offenders.append((path, module))
    assert offenders == []


def test_no_alternative_stt_modules_exist() -> None:
    import importlib.util

    for module in (
        "bodhan.adapters.transcribe.whisper",
        "bodhan.adapters.transcribe.deepgram",
        "bodhan.adapters.transcribe.google_stt",
        "bodhan.adapters.transcribe.azure_stt",
        "bodhan.adapters.transcribe.fallback",
    ):
        assert importlib.util.find_spec(module) is None, module


def test_transcribe_package_exports_no_fallback_symbols() -> None:
    import bodhan.adapters.transcribe as transcribe_package

    for symbol in (
        "FallbackTranscribe",
        "WhisperAdapter",
        "should_fallback",
        "fallback_adapter",
        "alternative_provider",
    ):
        assert not hasattr(transcribe_package, symbol), symbol


def test_transcribe_port_has_no_fallback_api() -> None:
    from bodhan.ports.transcribe import TranscribePort

    for symbol in ("fallback", "fallback_count", "switch_provider"):
        assert not hasattr(TranscribePort, symbol), symbol
