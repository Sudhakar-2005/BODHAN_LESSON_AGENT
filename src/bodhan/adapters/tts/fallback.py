"""Explicit primary -> fallback narration policy.

Fallback is allowed ONLY for transient transport failures:
network errors, timeouts and 5xx responses.

It is NEVER taken for:
- auth errors (401/403),
- quota / rate limits (429),
- validation errors (400/413/422, ``ValueError`` for empty/invalid input),
- any unrecognised exception.

Every fallback event is logged (reason only, never credentials or payloads).
"""

from __future__ import annotations

import logging
from pathlib import Path

import httpx

from bodhan.adapters.tts.errors import TTSProviderError
from bodhan.domain.models import AudioSegment
from bodhan.ports.tts import TTSPort

#: Transient statuses for which switching engines is acceptable.
_FALLBACK_SAFE_STATUS = frozenset({408, 425, 500, 502, 503, 504})


def should_fallback(exc: BaseException) -> bool:
    """Decide whether a failure may be recovered by the fallback engine."""
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
        return True
    if isinstance(exc, TTSProviderError):
        return exc.status_code in _FALLBACK_SAFE_STATUS
    return False  # auth/quota/validation/unknown -> never fall back silently


class FallbackTTS(TTSPort):
    """Wrap a primary engine with a fallback engine under a strict policy."""

    def __init__(
        self,
        primary: TTSPort,
        fallback: TTSPort,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._logger = logger or logging.getLogger(__name__)
        self._fallback_count = 0

    @property
    def fallback_count(self) -> int:
        """How many segments were served by the fallback engine."""
        return self._fallback_count

    @property
    def extension(self) -> str:
        # The primary engine decides the on-disk format of a segment.
        return self._primary.extension

    async def synthesize(
        self,
        text: str,
        *,
        segment_id: str,
        lesson_id: str,
        output_path: Path,
    ) -> AudioSegment:
        try:
            return await self._primary.synthesize(
                text, segment_id=segment_id, lesson_id=lesson_id, output_path=output_path
            )
        except Exception as exc:  # policy decides what propagates (see should_fallback)
            if not should_fallback(exc):
                raise
            self._logger.warning(
                "narration fallback for %s: primary failed (%s: %s)",
                segment_id,
                type(exc).__name__,
                str(exc)[:200],
            )
            self._fallback_count += 1
            return await self._fallback.synthesize(
                text, segment_id=segment_id, lesson_id=lesson_id, output_path=output_path
            )

    async def aclose(self) -> None:
        await self._primary.aclose()
        await self._fallback.aclose()
