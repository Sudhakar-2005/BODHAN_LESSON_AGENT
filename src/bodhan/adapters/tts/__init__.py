"""TTS adapters: Bodhan Indic-Speak (the only narration engine)."""

from bodhan.adapters.tts.bodhan_speak import BodhanSpeakAdapter, split_for_speech
from bodhan.adapters.tts.errors import TTSProviderError, TTSResponseError, TTSTooLongError

__all__ = [
    "BodhanSpeakAdapter",
    "TTSProviderError",
    "TTSResponseError",
    "TTSTooLongError",
    "split_for_speech",
]
