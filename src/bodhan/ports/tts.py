"""Text-to-speech port."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from bodhan.domain.models import AudioSegment


class TTSPort(ABC):
    """Synthesize one block of lesson text into an audio file."""

    @property
    def extension(self) -> str:
        """File extension (with dot) produced by this implementation."""
        return ".mp3"

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        *,
        segment_id: str,
        lesson_id: str,
        output_path: Path,
    ) -> AudioSegment:
        """Write narration for ``text`` to ``output_path`` and describe it."""
        raise NotImplementedError

    async def aclose(self) -> None:
        """Release underlying resources (optional)."""
        return None
