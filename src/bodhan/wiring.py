"""Composition root: builds concrete providers from settings.

Keeps provider wiring (which adapter, which credentials) out of the CLI and
the pipeline, so tests can substitute fakes in one place.

Provider matrix (approved):
- OCR      : Bodhan Indic-OCR only (no alternative engine, ever)
- Translate: Bodhan Indic-Translate (all target-language text originates here)
- Narrate  : Bodhan indic-speak only (the assignment restricts alternative
             TTS providers - there is no fallback engine, ever)
- LLM      : Gemini for structure (ambiguous cases), quality and English-only
             lesson planning - never extraction, translation or Indian-language
             authoring
- Transcribe: Bodhan Indic-Transcribe only (back-transcription of narrated
              audio - no alternative STT provider, ever)
"""

from __future__ import annotations

from dataclasses import dataclass

from bodhan.adapters.llm.gemini import GeminiLLMAdapter
from bodhan.adapters.ocr.indic_ocr import IndicOCRAdapter
from bodhan.adapters.transcribe.indic_transcribe import BodhanTranscribeAdapter
from bodhan.adapters.translate.bodhan import BodhanTranslateAdapter
from bodhan.adapters.tts.bodhan_speak import BodhanSpeakAdapter
from bodhan.config import Settings
from bodhan.ports.llm import LLMPort
from bodhan.ports.ocr import OCRPort
from bodhan.ports.transcribe import TranscribePort
from bodhan.ports.translate import TranslatePort
from bodhan.ports.tts import TTSPort


@dataclass
class Providers:
    ocr: OCRPort
    llm: LLMPort
    translate: TranslatePort
    tts: TTSPort
    transcribe: TranscribePort

    async def aclose(self) -> None:
        for provider in (self.ocr, self.llm, self.translate, self.tts, self.transcribe):
            await provider.aclose()


def build_providers(settings: Settings, *, target_language: str | None = None) -> Providers:
    """Instantiate the production provider set.

    NOTE: Indic-OCR is the only OCR engine and Indic-Speak is the only
    narration engine - no alternative provider is ever constructed
    (the assignment restricts alternative TTS providers). Transient
    speak failures are handled by bounded retries inside the adapter.
    """
    speak_lang = target_language or settings.default_target_language
    return Providers(
        ocr=IndicOCRAdapter(
            base_url=settings.bodhan_ocr_endpoint,
            api_key=settings.bodhan_ocr_key,
            timeout_s=settings.bodhan_timeout_s,
            max_retries=settings.bodhan_max_retries,
            low_confidence_threshold=settings.ocr_low_confidence_threshold,
            table_format=settings.ocr_table_format,
            max_tokens=settings.ocr_max_tokens,
        ),
        llm=GeminiLLMAdapter(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            timeout_s=settings.llm_timeout_s,
            max_retries=settings.llm_max_retries,
        ),
        translate=BodhanTranslateAdapter(
            base_url=(
                settings.bodhan_translate_chat_endpoint
                if settings.translate_transport == "chat"
                else settings.bodhan_translate_endpoint
            ),
            api_key=settings.bodhan_translate_key,
            transport=settings.translate_transport,
            model=settings.translate_model,
            script=settings.translate_script,
            timeout_s=settings.bodhan_timeout_s,
            max_retries=settings.bodhan_max_retries,
        ),
        tts=BodhanSpeakAdapter(
            base_url=settings.bodhan_speak_endpoint,
            api_key=settings.bodhan_speak_key,
            voice=settings.bodhan_speak_voice,
            lang=speak_lang,
            style=settings.bodhan_speak_style,
            chunk_seconds=settings.bodhan_speak_chunk_seconds,
            chars_per_second=settings.speak_chars_per_second,
            timeout_s=settings.bodhan_timeout_s,
            max_retries=settings.bodhan_max_retries,
        ),
        transcribe=BodhanTranscribeAdapter(
            base_url=settings.bodhan_transcribe_endpoint,
            api_key=settings.bodhan_transcribe_key,
            model=settings.transcribe_model,
            timeout_s=settings.bodhan_timeout_s,
            max_retries=settings.bodhan_max_retries,
            max_audio_seconds=settings.transcribe_max_audio_s,
            pace_s=settings.transcribe_pacing_s,
        ),
    )
