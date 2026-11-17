"""Speech-to-text port (back-transcription of narrated audio).

The assignment restricts speech-to-text to Bodhan Indic-Transcribe, so this
port has exactly one production implementation and no fallback provider.

Target-language codes are the documented ``language`` enum of the Bodhan
``POST /v1/audio/transcriptions`` endpoint (26 codes). Script variants such
as ``ks-Deva`` / ``mni-Beng`` / ``sd-Arab`` belong to the *translation*
enum and are deliberately absent here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field

#: Documented ``language`` enum of the speech-to-text endpoint.
SUPPORTED_TRANSCRIBE_LANGUAGES: frozenset[str] = frozenset(
    {
        "as",
        "bhb",
        "bho",
        "bn",
        "brx",
        "doi",
        "en",
        "gu",
        "hi",
        "kn",
        "kok",
        "ks",
        "mai",
        "ml",
        "mni",
        "mr",
        "ne",
        "or",
        "pa",
        "sa",
        "sat",
        "sd",
        "ta",
        "te",
        "ur",
    }
)


class TranscriptionResult(BaseModel):
    """One validated transcription response (no secrets, provider metadata preserved).

    The documented 200 body contains ``text`` only - timestamps, segment
    boundaries, usage and a model echo are NOT part of the contract, so they
    stay ``None``/empty unless the provider actually sends them. Nothing here
    is invented: every populated field was read off a real response.
    """

    #: Verbatim provider output - never translated, never rewritten.
    text: str
    model: str
    #: Language code we requested (``None`` = auto-detection was requested).
    language: str | None = None
    provider: str = "bodhan-transcribe"
    status_code: int = 200
    #: Wall-clock time for the request(s), including bounded retries.
    latency_ms: int = 0
    #: ``x-request-id`` of the response(s); comma-joined for a file that was
    #: split into several requests (documented 30 s ceiling).
    request_id: str | None = None
    #: Numbers only, from a ``usage`` object when the provider sends one.
    usage: dict[str, int | float] = Field(default_factory=dict)
    #: Verbatim provider bodies - one entry per HTTP request.
    raw: list[dict[str, Any]] = Field(default_factory=list)


class TranscribePort(ABC):
    """Back-transcribe one narrated audio file into text."""

    #: Provider identifier recorded in artifacts (never a credential).
    provider: ClassVar[str] = "unknown"

    @abstractmethod
    async def transcribe(
        self, audio_path: Path, *, language: str | None = None
    ) -> TranscriptionResult:
        """Transcribe ``audio_path``; raises on auth/validation/provider failure."""
        raise NotImplementedError

    async def aclose(self) -> None:
        """Release underlying resources (optional)."""
        return None
