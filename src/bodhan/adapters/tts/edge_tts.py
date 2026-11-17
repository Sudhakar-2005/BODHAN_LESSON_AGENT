"""edge-tts narration adapter (free neural voices, no API key)."""

from __future__ import annotations

from pathlib import Path

import edge_tts

from bodhan.domain.models import AudioSegment
from bodhan.ports.tts import TTSPort


class EdgeTTSAdapter(TTSPort):
    """Synthesize lesson text to MP3 files with Microsoft Edge neural voices."""

    def __init__(
        self,
        *,
        voice: str = "hi-IN-SwaraNeural",
        rate: str = "+0%",
        pitch: str = "+0Hz",
    ) -> None:
        self._voice = voice
        self._rate = rate
        self._pitch = pitch

    async def synthesize(
        self,
        text: str,
        *,
        segment_id: str,
        lesson_id: str,
        output_path: Path,
    ) -> AudioSegment:
        if not text.strip():
            raise ValueError(f"segment '{segment_id}' has no text to narrate")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        communicate = edge_tts.Communicate(
            text, voice=self._voice, rate=self._rate, pitch=self._pitch
        )
        await communicate.save(str(output_path))

        return AudioSegment(
            segment_id=segment_id,
            lesson_id=lesson_id,
            text=text,
            audio_path=str(output_path),
        )
