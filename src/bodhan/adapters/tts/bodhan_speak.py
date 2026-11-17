"""Bodhan ``POST /v1/audio/speech`` adapter (indic-speak), documented contract.

Request (application/json, documented fields only)::

    {"model": "indic-speak",
     "input": "…text…",
     "voice": "Kavya",
     "instructions": "{\\"lang\\": \\"hi\\"[, \\"style\\": \\"…\\"]}"}

Response: raw ``audio/wav`` (PCM16, 24 kHz, mono). Errors: ``422`` when
input/voice/lang/style are invalid, ``401`` for auth, ``429`` when out of
credit (terminal) or when the per-key rate allowance is exceeded (the
documented ``429`` + ``retry-after`` case, retried until the window frees).

Documented operating limits honoured here:
- "keep each request to a sentence or two, about 30 seconds of speech;
  longer text is not split for you" -> we split locally, sentence-aware, and
  concatenate the WAV chunks into the single requested output file;
- any voice can read any language, but ``instructions.lang`` is required.

WAV handling uses the stdlib ``wave`` module (no ffmpeg dependency).
"""

from __future__ import annotations

import io
import json
import re
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
from bodhan.adapters.tts.errors import TTSProviderError, TTSResponseError
from bodhan.domain.models import AudioSegment
from bodhan.ports.tts import TTSPort

_RETRYABLE_STATUS = {408, 425, 500, 502, 503, 504}
_SENTENCE_BOUNDARY = re.compile(r"(?<=[।.!?])\s+")


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
        return True
    if not isinstance(exc, TTSProviderError):
        return False
    if exc.status_code == 429:
        # Documented: the rate-limit 429 carries `retry-after` (4 RPM
        # sliding window); an out-of-credit 429 does not - that one is
        # terminal and must not be retried.
        return exc.retry_after is not None
    return exc.status_code in _RETRYABLE_STATUS


def split_for_speech(text: str, max_chars: int) -> list[str]:
    """Sentence-aware chunking so each request stays within ~30 s of speech.

    A single sentence longer than ``max_chars`` is hard-split on word
    boundaries. Never returns an empty list for non-empty input.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    current = ""
    for sentence in _SENTENCE_BOUNDARY.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        while len(sentence) > max_chars:  # oversized sentence -> word split
            if current:
                chunks.append(current)
                current = ""
            cut = sentence.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            chunks.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if not sentence:
            continue
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks


@dataclass(frozen=True)
class _WavChunk:
    channels: int
    sampwidth: int
    framerate: int
    frames: bytes
    duration_s: float


def _read_wav(data: bytes) -> _WavChunk:
    """Decode a WAV payload (PCM16 24 kHz mono per docs)."""
    try:
        with wave.open(io.BytesIO(data), "rb") as handle:
            channels = handle.getnchannels()
            sampwidth = handle.getsampwidth()
            framerate = handle.getframerate()
            nframes = handle.getnframes()
            frames = handle.readframes(nframes)
    except (wave.Error, EOFError) as exc:
        raise TTSResponseError(f"response is not a decodable WAV: {exc}") from exc
    duration = nframes / framerate if framerate else 0.0
    return _WavChunk(
        channels=channels,
        sampwidth=sampwidth,
        framerate=framerate,
        frames=frames,
        duration_s=duration,
    )


class BodhanSpeakAdapter(TTSPort):
    """Async client for the Bodhan text-to-speech endpoint (returns WAV)."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        voice: str = "Kavya",
        lang: str = "hi",
        style: str | None = None,
        chunk_seconds: float = 30.0,
        chars_per_second: float = 13.0,
        timeout_s: float = 60.0,
        max_retries: int = 3,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("base_url is required")
        if not voice:
            raise ValueError("voice is required")
        if not lang:
            raise ValueError("lang is required")
        self._base_url = base_url
        self._voice = voice
        self._lang = lang
        self._style = style
        self._max_chars = max(1, int(chunk_seconds * chars_per_second))
        self._max_retries = max(1, max_retries)
        self._rate_limit: RateLimitInfo | None = None
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(timeout_s))
        if api_key:  # auth applies to injected clients too
            self._client.headers["Authorization"] = f"Bearer {api_key}"

    @property
    def extension(self) -> str:
        return ".wav"

    @property
    def rate_limit(self) -> RateLimitInfo | None:
        """Latest published rate limits read from a response (no secrets)."""
        return self._rate_limit

    # -- TTSPort ------------------------------------------------------------
    async def synthesize(
        self,
        text: str,
        *,
        segment_id: str,
        lesson_id: str,
        output_path: Path,
    ) -> AudioSegment:
        chunks = split_for_speech(text, self._max_chars)
        if not chunks:
            raise ValueError(f"segment '{segment_id}' has no text to narrate")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        parts: list[_WavChunk] = []
        for index, chunk in enumerate(chunks, start=1):
            data = await self._request_wav(chunk, label=f"{segment_id}#{index}/{len(chunks)}")
            parts.append(_read_wav(data))

        first = parts[0]
        mismatched = next(
            (
                part
                for part in parts
                if (part.channels, part.sampwidth, part.framerate)
                != (first.channels, first.sampwidth, first.framerate)
            ),
            None,
        )
        if mismatched is not None:
            raise TTSResponseError("WAV chunk format changed between requests")

        # Concatenate identical-format PCM chunks into one file.
        with wave.open(str(output_path), "wb") as out:
            out.setnchannels(first.channels)
            out.setsampwidth(first.sampwidth)
            out.setframerate(first.framerate)
            out.writeframes(b"".join(part.frames for part in parts))

        duration_ms = int(sum(part.duration_s for part in parts) * 1000)
        return AudioSegment(
            segment_id=segment_id,
            lesson_id=lesson_id,
            text=text,
            audio_path=str(output_path),
            duration_ms=duration_ms,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # -- internals ----------------------------------------------------------
    def _payload(self, text: str) -> dict[str, Any]:
        instructions: dict[str, str] = {"lang": self._lang}
        if self._style:
            instructions["style"] = self._style
        return {
            "model": "indic-speak",
            "input": text,
            "voice": self._voice,
            "instructions": json.dumps(instructions, ensure_ascii=False),
        }

    async def _request_wav(self, text: str, *, label: str) -> bytes:
        payload = self._payload(text)
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self._max_retries),
                wait=RetryAfterWait(wait_exponential(multiplier=1.0, max=30.0)),
                retry=retry_if_exception(_is_retryable),
                reraise=True,
            ):
                with attempt:
                    response = await self._client.post(self._base_url, json=payload)
                    info = read_rate_limit(response)
                    if info is not None:
                        self._rate_limit = info
                    if response.status_code >= 400:
                        raise self._error_from(response)
                    content_type = response.headers.get("content-type", "")
                    if not content_type.startswith("audio/"):
                        raise TTSResponseError(
                            f"expected audio/* response, got {content_type[:60]} for {label}",
                            status_code=response.status_code,
                        )
        except TTSProviderError:
            raise
        except httpx.HTTPError as exc:
            raise TTSProviderError(f"speech request failed: {exc}") from exc
        return response.content

    @staticmethod
    def _error_from(response: httpx.Response) -> TTSProviderError:
        message = response.text[:300]
        code: str | None = None
        try:
            body = response.json()
        except ValueError:
            body = None
        if isinstance(body, dict):
            for key in ("message", "detail", "error"):
                value = body.get(key)
                if isinstance(value, str) and value:
                    message = value[:300]
                    break
                if isinstance(value, dict) and isinstance(value.get("message"), str):
                    message = value["message"][:300]
                    break
            if body.get("code") is not None:
                code = str(body["code"])
        return TTSProviderError(
            f"speech HTTP {response.status_code}: {message}",
            status_code=response.status_code,
            code=code,
            retry_after=read_retry_after(response),
        )
