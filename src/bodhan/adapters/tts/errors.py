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
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        #: Seconds the provider asked us to wait (documented ``retry-after``
        #: header on a rate-limit ``429``); ``None`` when absent/unparseable.
        self.retry_after = retry_after


class TTSResponseError(TTSProviderError):
    """The provider returned something that is not the documented WAV."""


class TTSTooLongError(TTSProviderError):
    """Input exceeded the documented per-request speech budget."""
