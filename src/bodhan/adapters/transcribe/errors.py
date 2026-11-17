"""Error taxonomy for speech-to-text adapters.

Never contains credentials. ``retry_after`` mirrors the shared retry helper:
it carries the documented ``retry-after`` header of a rate-limit ``429`` so
:class:`~bodhan.adapters.retry.RetryAfterWait` can ride out the window.
"""

from __future__ import annotations


class TranscriptionProviderError(Exception):
    """Base class for transcription failures (never contains credentials)."""

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


class TranscriptionResponseError(TranscriptionProviderError):
    """The provider returned a payload we could not interpret (malformed)."""


class EmptyTranscriptionError(TranscriptionResponseError):
    """The provider answered 200 with empty/whitespace transcription text."""


class UnsupportedLanguageError(TranscriptionProviderError):
    """Target language is not in the documented enum (rejected locally)."""
