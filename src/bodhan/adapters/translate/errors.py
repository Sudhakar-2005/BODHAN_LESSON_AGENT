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
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code


class TranslationResponseError(TranslationProviderError):
    """The provider returned a payload we could not interpret."""


class TranslationTruncatedError(TranslationProviderError):
    """The provider reported ``truncated: true`` - output is incomplete."""


class UnsupportedLanguageError(TranslationProviderError):
    """Target language is not in the documented enum (rejected locally)."""
