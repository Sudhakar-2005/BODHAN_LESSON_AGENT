"""Speech-to-text adapters: Bodhan Indic-Transcribe (the only STT provider)."""

from bodhan.adapters.transcribe.errors import (
    EmptyTranscriptionError,
    TranscriptionProviderError,
    TranscriptionResponseError,
    UnsupportedLanguageError,
)
from bodhan.adapters.transcribe.indic_transcribe import (
    MAX_AUDIO_SECONDS,
    BodhanTranscribeAdapter,
    split_wav,
)

__all__ = [
    "MAX_AUDIO_SECONDS",
    "BodhanTranscribeAdapter",
    "EmptyTranscriptionError",
    "TranscriptionProviderError",
    "TranscriptionResponseError",
    "UnsupportedLanguageError",
    "split_wav",
]
