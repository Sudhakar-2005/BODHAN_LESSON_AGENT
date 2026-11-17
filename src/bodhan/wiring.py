"""Composition root: builds concrete providers from settings.

Keeps provider wiring (which adapter, which credentials) out of the CLI and
the pipeline, so tests can substitute fakes in one place.

Provider matrix (approved):
- OCR      : Bodhan Indic-OCR only (no alternative engine, ever)
- Translate: Bodhan POST /translate
- Narrate  : Bodhan indic-speak (primary) -> edge-tts (transient failures only)
- LLM      : Gemini for structure (ambiguous cases), quality and composition
- Transcribe: reserved for Phase 2 (no stage, no adapter in this DAG)
"""

from __future__ import annotations

from dataclasses import dataclass

from bodhan.adapters.llm.gemini import GeminiLLMAdapter
from bodhan.adapters.ocr.indic_ocr import IndicOCRAdapter
from bodhan.adapters.translate.bodhan import BodhanTranslateAdapter
from bodhan.adapters.tts.bodhan_speak import BodhanSpeakAdapter
from bodhan.adapters.tts.edge_tts import EdgeTTSAdapter
from bodhan.adapters.tts.fallback import FallbackTTS
from bodhan.config import Settings
from bodhan.ports.llm import LLMPort
from bodhan.ports.ocr import OCRPort
from bodhan.ports.translate import TranslatePort
from bodhan.ports.tts import TTSPort


@dataclass
class Providers:
    ocr: OCRPort
    llm: LLMPort
    translate: TranslatePort
    tts: TTSPort

    async def aclose(self) -> None:
        for provider in (self.ocr, self.llm, self.translate, self.tts):
            await provider.aclose()


def build_providers(settings: Settings, *, target_language: str | None = None) -> Providers:
    """Instantiate the production provider set.

    NOTE: Indic-OCR is the only OCR engine -- no fallback is ever constructed.
    Narration falls back to edge-tts only for transient transport failures;
    auth/quota/validation errors propagate (see ``FallbackTTS``).
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
        tts=FallbackTTS(
            primary=BodhanSpeakAdapter(
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
            fallback=EdgeTTSAdapter(
                voice=settings.edge_tts_voice,
                rate=settings.edge_tts_rate,
                pitch=settings.edge_tts_pitch,
            ),
        ),
    )
