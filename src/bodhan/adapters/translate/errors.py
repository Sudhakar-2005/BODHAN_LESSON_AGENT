"""Error taxonomy for translation adapters."""

from __future__ import annotations


class TranslationProviderError(Exception):
    """Base class for translation failures (never contains credentials)."""

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
        #: header on ``429``); ``None`` when absent or unparseable.
        self.retry_after = retry_after


class TranslationResponseError(TranslationProviderError):
    """The provider returned a payload we could not interpret."""


class TranslationTruncatedError(TranslationProviderError):
    """The provider reported ``truncated: true`` - output is incomplete."""


class UnsupportedLanguageError(TranslationProviderError):
    """Target language is not in the documented enum (rejected locally)."""
