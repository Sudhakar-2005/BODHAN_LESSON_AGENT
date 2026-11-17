"""Bodhan translation adapter (POST /translate)."""

from bodhan.adapters.translate.bodhan import BodhanTranslateAdapter
from bodhan.adapters.translate.errors import (
    TranslationProviderError,
    TranslationResponseError,
    TranslationTruncatedError,
    UnsupportedLanguageError,
)

__all__ = [
    "BodhanTranslateAdapter",
    "TranslationProviderError",
    "TranslationResponseError",
    "TranslationTruncatedError",
    "UnsupportedLanguageError",
]
