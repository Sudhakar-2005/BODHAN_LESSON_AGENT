"""TTS adapters: Bodhan Indic-Speak (primary) + edge-tts (fallback)."""

from bodhan.adapters.tts.bodhan_speak import BodhanSpeakAdapter, split_for_speech
from bodhan.adapters.tts.edge_tts import EdgeTTSAdapter
from bodhan.adapters.tts.errors import TTSProviderError, TTSResponseError, TTSTooLongError
from bodhan.adapters.tts.fallback import FallbackTTS, should_fallback

__all__ = [
    "BodhanSpeakAdapter",
    "EdgeTTSAdapter",
    "FallbackTTS",
    "TTSProviderError",
    "TTSResponseError",
    "TTSTooLongError",
    "should_fallback",
    "split_for_speech",
]
