"""Translation port.

Target-language codes are the documented ``target_language`` enum of the
Bodhan ``POST /translate`` endpoint (26 codes over 23 languages; ``bhb`` and
``bho`` are explicitly refused by the API).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

#: Documented target_language enum of the translation endpoint.
SUPPORTED_TARGET_LANGUAGES: frozenset[str] = frozenset(
    {
        "as",
        "bn",
        "brx",
        "doi",
        "en",
        "gu",
        "hi",
        "kn",
        "kok",
        "ks",
        "ks-Deva",
        "mai",
        "ml",
        "mni",
        "mni-Beng",
        "mr",
        "ne",
        "or",
        "pa",
        "sa",
        "sat",
        "sd",
        "sd-Arab",
        "ta",
        "te",
        "ur",
    }
)


class TranslationResult(BaseModel):
    """One validated translation response (no secrets, usage preserved)."""

    text: str
    target_language: str
    script: str | None = None
    usage: dict[str, int] = Field(default_factory=dict)
    provider: str = "bodhan-translate"


class TranslatePort(ABC):
    """Translate one text unit into a target language."""

    @abstractmethod
    async def translate(self, text: str, *, target_language: str) -> TranslationResult:
        """Translate ``text``; raises on validation/auth/quota/truncation."""
        raise NotImplementedError

    async def aclose(self) -> None:
        """Release underlying resources (optional)."""
        return None
