"""Error taxonomy for TTS adapters."""

from __future__ import annotations


class TTSProviderError(Exception):
    """Base class for narration failures (never contains credentials)."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code


class TTSResponseError(TTSProviderError):
    """The provider returned something that is not the documented WAV/MP3."""


class TTSTooLongError(TTSProviderError):
    """Input exceeded the documented per-request speech budget."""
