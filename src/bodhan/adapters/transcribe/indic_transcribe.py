"""Bodhan ``POST /v1/audio/transcriptions`` adapter (indic-transcribe).

Verified official contract (``https://console.bodhan.ai/api-docs``, inspected
2026-11-17) - the implementation notes of Phase 2A Task 1:

- endpoint  : ``POST https://api.bodhan.ai/v1/audio/transcriptions``
- transport : ``multipart/form-data``
- auth      : ``Authorization: Bearer <BODHAN_TRANSCRIBE_KEY>`` (repr=False
              in settings, never logged, never printed)
- request   : ``file`` (required; WAV documented, FLAC/OGG/MP3 also accepted),
              ``model`` (required) = ``indic-transcribe`` - the ONLY documented
              gateway id (Core/Flex are the Hugging Face repositories
              ``bodhan-ai/indic-transcribe-core`` / ``-flex`` served behind
              it, not gateway model ids),
              ``language`` (optional; documented enum of 26 codes; omitted
              means the provider auto-detects the spoken language)
- response  : ``{"text": "..."}`` ONLY - no timestamps, no segment-level
              text, no usage, no model echo. Fields the contract does not
              return are never fabricated; anything we do record was read
              off an actual response or header.
- limits    : max 30 s of audio per request - longer audio is rejected. A
              narration segment concatenates <= 30 s speak chunks and can
              exceed the ceiling, so WAV payloads are split into <= 30 s
              pieces client-side, each piece is transcribed, and the piece
              texts are joined verbatim with a space.
- errors    : ``401`` auth, ``400``/``413`` validation, ``415`` unsupported
              or undecodable container, ``429`` out-of-credit (terminal, no
              header) vs rate-limit ``429`` + ``retry-after`` (documented
              8 RPM sliding window - retried), ``5xx`` transient (bounded
              exponential backoff). Auth/validation failures stay terminal.
- headers   : ``x-request-id`` captured when present; published limits read
              from ``x-ratelimit-limit-*`` (response-driven, never hardcoded).
- billing   : ₹0.10 per audio minute; the API does not echo usage, so
              ``usage`` stays empty unless a response actually carries it.

Bodhan Indic-Transcribe is the ONLY speech-to-text provider: no Whisper,
Google Speech-to-Text, Azure Speech, Deepgram or any other STT engine exists
in this package, and this adapter never silently switches providers.
"""

from __future__ import annotations

import io
import mimetypes
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from bodhan.adapters.rate_limit import RateLimitInfo, read_rate_limit
from bodhan.adapters.retry import RetryAfterWait, read_retry_after
from bodhan.adapters.transcribe.errors import (
    EmptyTranscriptionError,
    TranscriptionProviderError,
    TranscriptionResponseError,
    UnsupportedLanguageError,
)
from bodhan.ports.transcribe import (
    SUPPORTED_TRANSCRIBE_LANGUAGES,
    TranscribePort,
    TranscriptionResult,
)

#: Retryable statuses: transient transport/5xx plus the rate-limit 429 that
#: carries ``retry-after`` (that distinction is made in ``_is_retryable``).
_RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}
#: Documented per-request ceiling (the gateway rejects anything longer).
MAX_AUDIO_SECONDS = 30.0


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
        return True
    if not isinstance(exc, TranscriptionProviderError):
        return False
    if exc.status_code == 429:
        # Documented: the rate-limit 429 carries `retry-after` (8 RPM
        # sliding window); an out-of-credit 429 does not - that one is
        # terminal and must not be retried.
        return exc.retry_after is not None
    return exc.status_code in _RETRYABLE_STATUS


def split_wav(data: bytes, max_seconds: float = MAX_AUDIO_SECONDS) -> list[bytes]:
    """Split a WAV payload into pieces of at most ``max_seconds`` each.

    The documented gateway limit is 30 s of audio per request, while a
    narrated segment may be arbitrarily long. Splitting is done with the
    stdlib ``wave`` module (no ffmpeg dependency) on exact frame boundaries,
    so every piece is a complete, decodable WAV.

    Non-WAV or undecodable payloads are returned unchanged: formats we
    cannot measure locally are sent as-is and the provider's documented
    ``415``/``400`` surfaces the limit instead.
    """
    if max_seconds <= 0:
        raise ValueError("max_seconds must be positive")
    try:
        with wave.open(io.BytesIO(data), "rb") as handle:
            rate = handle.getframerate()
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            frames = handle.readframes(handle.getnframes())
    except (wave.Error, EOFError):
        return [data]
    if rate <= 0 or channels <= 0 or width <= 0:
        return [data]

    frame_bytes = channels * width
    piece_bytes = int(max_seconds * rate) * frame_bytes
    if len(frames) <= piece_bytes:
        return [data]

    pieces: list[bytes] = []
    for start in range(0, len(frames), piece_bytes):
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as out:
            out.setnchannels(channels)
            out.setsampwidth(width)
            out.setframerate(rate)
            out.writeframes(frames[start : start + piece_bytes])
        pieces.append(buffer.getvalue())
    return pieces


@dataclass(frozen=True)
class _PieceOutcome:
    """Everything one successful HTTP request contributed to the result."""

    text: str
    latency_ms: int
    request_id: str | None
    usage: dict[str, int | float]
    raw: dict[str, Any]
    status_code: int


class BodhanTranscribeAdapter(TranscribePort):
    """Async client for the Bodhan speech-to-text endpoint (the only STT)."""

    provider = "bodhan-transcribe"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str = "indic-transcribe",
        timeout_s: float = 60.0,
        max_retries: int = 3,
        max_audio_seconds: float = MAX_AUDIO_SECONDS,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("base_url is required")
        if not model:
            raise ValueError("model is required")
        if not 0 < max_audio_seconds <= MAX_AUDIO_SECONDS:
            raise ValueError(
                f"max_audio_seconds must be within the documented "
                f"{MAX_AUDIO_SECONDS:g} s request limit"
            )
        self._base_url = base_url
        self._model = model
        self._max_audio_seconds = max_audio_seconds
        self._max_retries = max(1, max_retries)
        self._rate_limit: RateLimitInfo | None = None
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(timeout_s))
        if api_key:  # auth applies to injected clients too
            self._client.headers["Authorization"] = f"Bearer {api_key}"

    @property
    def rate_limit(self) -> RateLimitInfo | None:
        """Latest published rate limits read from a response (no secrets)."""
        return self._rate_limit

    # -- TranscribePort -----------------------------------------------------
    async def transcribe(
        self, audio_path: Path, *, language: str | None = None
    ) -> TranscriptionResult:
        """Transcribe one audio file; provider text is returned verbatim.

        Raises:
            UnsupportedLanguageError: language outside the documented enum.
            FileNotFoundError: the audio file does not exist.
            TranscriptionProviderError: auth/validation/rate-limit/provider
                failure (terminal errors are not retried).
            TranscriptionResponseError / EmptyTranscriptionError: malformed
                or empty provider response (never retried, never accepted).
        """
        if language is not None and language not in SUPPORTED_TRANSCRIBE_LANGUAGES:
            raise UnsupportedLanguageError(
                f"language {language!r} is not in the documented transcription enum",
                code="unsupported_language",
            )

        path = Path(audio_path)
        data = path.read_bytes()  # FileNotFoundError stays a local failure
        pieces = split_wav(data, self._max_audio_seconds)

        texts: list[str] = []
        request_ids: list[str] = []
        usage: dict[str, int | float] = {}
        raw_bodies: list[dict[str, Any]] = []
        total_latency_ms = 0
        status_code = 200

        for index, piece in enumerate(pieces, start=1):
            filename = (
                path.name if len(pieces) == 1 else f"{path.stem}.part{index}{path.suffix or '.wav'}"
            )
            outcome = await self._request_text(piece, filename=filename, language=language)
            texts.append(outcome.text)
            if outcome.request_id is not None:
                request_ids.append(outcome.request_id)
            for key, value in outcome.usage.items():
                usage[key] = usage.get(key, 0) + value
            raw_bodies.append(outcome.raw)
            total_latency_ms += outcome.latency_ms
            status_code = outcome.status_code

        return TranscriptionResult(
            # Joined verbatim: each piece is the provider's own output, never
            # translated or rewritten here.
            text=" ".join(texts),
            model=self._model,
            language=language,
            provider=self.provider,
            status_code=status_code,
            latency_ms=total_latency_ms,
            request_id=",".join(request_ids) if request_ids else None,
            usage=usage,
            raw=raw_bodies,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # -- request ------------------------------------------------------------
    async def _request_text(
        self, data: bytes, *, filename: str, language: str | None
    ) -> _PieceOutcome:
        fields = {"model": self._model}
        if language is not None:
            fields["language"] = language
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        files = {"file": (filename, data, content_type)}

        started = time.perf_counter()
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self._max_retries),
                wait=RetryAfterWait(wait_exponential(multiplier=1.0, max=30.0)),
                retry=retry_if_exception(_is_retryable),
                reraise=True,
            ):
                with attempt:
                    response = await self._client.post(self._base_url, data=fields, files=files)
                    info = read_rate_limit(response)
                    if info is not None:
                        self._rate_limit = info
                    if response.status_code >= 400:
                        raise self._error_from(response)
        except TranscriptionProviderError:
            raise
        except httpx.HTTPError as exc:
            raise TranscriptionProviderError(f"transcription request failed: {exc}") from exc

        latency_ms = int((time.perf_counter() - started) * 1000)
        return self._parse(response, latency_ms=latency_ms)

    # -- response -----------------------------------------------------------
    @staticmethod
    def _error_from(response: httpx.Response) -> TranscriptionProviderError:
        """Parse the documented error envelope; fall back to a bare message."""
        code: str | None = None
        message = response.text[:300]
        try:
            body = response.json()
        except ValueError:
            body = None
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict):
                message = str(error.get("message") or message)
                code = str(error["code"]) if error.get("code") is not None else None
            elif isinstance(body.get("detail"), str):  # FastAPI-style envelope
                message = str(body["detail"])[:300]
            elif isinstance(body.get("message"), str):  # gateway-style envelope
                message = str(body["message"])[:300]
        return TranscriptionProviderError(
            f"transcribe HTTP {response.status_code}: {message}",
            status_code=response.status_code,
            code=code,
            retry_after=read_retry_after(response),
        )

    @staticmethod
    def _usage(body: dict[str, Any]) -> dict[str, int | float]:
        """Numbers from a ``usage`` object, only when the provider sends one."""
        usage: dict[str, int | float] = {}
        raw = body.get("usage")
        if isinstance(raw, dict):
            for key, value in raw.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                usage[str(key)] = value
        return usage

    @classmethod
    def _parse(cls, response: httpx.Response, *, latency_ms: int) -> _PieceOutcome:
        """Validate the documented ``{"text": "..."}`` body.

        Malformed and empty responses raise here, outside the retry loop:
        a response we cannot interpret must never be retried blindly or
        mistaken for a transcription.
        """
        try:
            body = response.json()
        except ValueError as exc:
            raise TranscriptionResponseError(
                "transcription returned non-JSON body", status_code=response.status_code
            ) from exc
        if not isinstance(body, dict):
            raise TranscriptionResponseError(
                "transcription returned a non-object body", status_code=response.status_code
            )
        if "error" in body:
            # Success responses never carry an `error` key (branch on presence).
            error = body.get("error")
            detail = error.get("message") if isinstance(error, dict) else error
            raise TranscriptionResponseError(
                f"transcription error payload: {detail}", status_code=response.status_code
            )
        text = body.get("text")
        if not isinstance(text, str):
            raise TranscriptionResponseError(
                "response is missing 'text'", status_code=response.status_code
            )
        if not text.strip():
            raise EmptyTranscriptionError(
                "provider returned an empty transcription",
                status_code=response.status_code,
                code="empty_transcription",
            )
        return _PieceOutcome(
            text=text,
            latency_ms=latency_ms,
            request_id=response.headers.get("x-request-id"),
            usage=cls._usage(body),
            raw=body,
            status_code=response.status_code,
        )
