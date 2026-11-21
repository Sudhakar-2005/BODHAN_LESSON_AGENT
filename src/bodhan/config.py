"""Application configuration.

All tunables live here so that stages and adapters receive typed, validated
settings instead of reading environment variables ad hoc.
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application settings, sourced from `.env` and environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    # --- Paths -------------------------------------------------------------
    project_root: Path = Field(default_factory=Path.cwd)
    data_dir: Path = Path("data")

    # --- Bodhan.AI gateway (four models, one key per model) ----------------
    # Credential fields are repr=False so keys can never appear in logs,
    # tracebacks or terminal output.
    bodhan_api_base_url: str = "https://api.bodhan.ai"
    bodhan_ocr_key: str = Field(default="", repr=False)
    bodhan_translate_key: str = Field(default="", repr=False)
    bodhan_speak_key: str = Field(default="", repr=False)
    bodhan_transcribe_key: str = Field(default="", repr=False)
    bodhan_timeout_s: float = 60.0
    bodhan_max_retries: int = Field(default=3, ge=1)
    # DOCUMENTATION ONLY - NOT a dynamic concurrency controller.
    # All provider calls are intentionally serialized (sequential awaits in
    # every stage); there is no semaphore/gather and this value is not read
    # by any code path beyond documentation. Serialization keeps every Bodhan
    # request within the documented per-key RPM limits by construction.
    # Do NOT interpret this setting as active concurrency enforcement.
    provider_max_concurrency: int = Field(default=1, ge=1, le=8)

    # Indic-OCR request tuning (documented options of the OCR endpoint)
    ocr_table_format: Literal["html", "markdown"] = "html"
    ocr_max_tokens: int = Field(default=2048, ge=1)  # per block, not per page
    ocr_low_confidence_threshold: float = Field(default=0.85, ge=0.0, le=1.0)

    # --- Documented endpoint paths (single source of truth) ----------------
    @property
    def bodhan_ocr_endpoint(self) -> str:
        """POST /v1/chat/completions (model ``indic-ocr``)."""
        return f"{self.bodhan_api_base_url}/v1/chat/completions"

    @property
    def bodhan_translate_endpoint(self) -> str:
        """POST /translate (documented product surface).

        NOTE: live probe 2026-11-17 returns HTTP 404 for this route while the
        documented equivalent chat transport (below) serves requests; keep
        this endpoint available for when the gateway route is live.
        """
        return f"{self.bodhan_api_base_url}/translate"

    @property
    def bodhan_translate_chat_endpoint(self) -> str:
        """POST /v1/chat/completions (documented equivalent transport)."""
        return f"{self.bodhan_api_base_url}/v1/chat/completions"

    @property
    def bodhan_speak_endpoint(self) -> str:
        """POST /v1/audio/speech (model ``indic-speak``)."""
        return f"{self.bodhan_api_base_url}/v1/audio/speech"

    @property
    def bodhan_transcribe_endpoint(self) -> str:
        """POST /v1/audio/transcriptions (model ``indic-transcribe``)."""
        return f"{self.bodhan_api_base_url}/v1/audio/transcriptions"

    # --- LLM (structure / quality / English lesson planning only) ----------
    # The LLM never extracts text, never translates, and never writes
    # Hindi/Tamil/Bengali or any other Indian-language content (Req 15).
    llm_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    llm_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("LLM_API_KEY", "GEMINI_API_KEY"),
        repr=False,
    )
    # Probed live (2026-11-17): gemini-2.5-* returns 404 (retired for new
    # users), gemini-3.8-flash returns persistent 503 (high demand), while
    # gemini-3.6-flash serves requests reliably (HTTP 200).
    llm_model: str = "gemini-3.6-flash"
    llm_timeout_s: float = 120.0
    llm_max_retries: int = 5

    # --- Narration ---------------------------------------------------------
    # Bodhan indic-speak (WAV) is the ONLY narration engine - the assignment
    # restricts alternative TTS providers, so no fallback engine exists.
    bodhan_speak_voice: str = "Kavya"
    bodhan_speak_style: str | None = None
    bodhan_speak_chunk_seconds: float = Field(default=30.0, gt=0, le=30.0)
    speak_chars_per_second: float = Field(default=13.0, gt=0)  # chunking estimate

    # --- Back-transcription ------------------------------------------------
    # Bodhan Indic-Transcribe is the ONLY speech-to-text engine - the
    # assignment restricts STT providers, so no fallback engine exists.
    # Documented gateway model id (the Core/Flex variants are the Hugging
    # Face repositories bodhan-ai/indic-transcribe-core / -flex served
    # behind this id, not separate gateway ids).
    transcribe_model: str = "indic-transcribe"
    # Documented request ceiling: 30 s of audio; longer narration segments
    # are split client-side (WAV) and the piece texts joined verbatim.
    transcribe_max_audio_s: float = Field(default=30.0, gt=0, le=30.0)
    # Delay before each Indic-Transcribe request (seconds; 0 disables).
    # Requests are sequential; pacing spreads them out so a burst cannot
    # exhaust the provider's RPM window. The window itself is never hardcoded.
    transcribe_pacing_s: float = Field(default=0.0, ge=0.0)

    # --- CER evaluation (Phase 2B) -----------------------------------------
    # Configurable quality bands, expressed as CER percentages. These are
    # PROJECT evaluation thresholds, not official Bodhan quality standards.
    # Excellent: <=5%; Good: >5-10%; Needs Review: >10-20%; Poor: >20%.
    cer_excellent_max_pct: float = Field(default=5.0, ge=0.0)
    cer_good_max_pct: float = Field(default=10.0, ge=0.0)
    cer_review_max_pct: float = Field(default=20.0, ge=0.0)

    @model_validator(mode="after")
    def _validate_page_range(self) -> Settings:
        if self.min_input_pages > self.max_input_pages:
            raise ValueError("min_input_pages must not exceed max_input_pages")
        return self

    @model_validator(mode="after")
    def _validate_cer_thresholds(self) -> Settings:
        if not (
            0 <= self.cer_excellent_max_pct <= self.cer_good_max_pct <= self.cer_review_max_pct
        ):
            raise ValueError(
                "CER thresholds must satisfy 0 <= excellent <= good <= review "
                f"(got {self.cer_excellent_max_pct}, {self.cer_good_max_pct}, "
                f"{self.cer_review_max_pct})"
            )
        return self

    # --- Translation -------------------------------------------------------
    # None = omit the optional `script` field (server default: native).
    translate_script: Literal["native", "roman", "codemix"] | None = None
    # Documented transports (identical output per the official docs):
    # "chat"    -> POST /v1/chat/completions, model `indic-translate`
    #               (verified live 200; the /translate product route 404s).
    # "product" -> POST /translate (product surface, when it is deployed).
    translate_transport: Literal["chat", "product"] = "chat"
    translate_model: str = "indic-translate"

    # --- Pipeline ----------------------------------------------------------
    render_dpi: int = Field(default=200, ge=72, le=600)
    # Assignment target range. The maximum is always enforced; the minimum is
    # opt-in to keep legacy 2-page smoke fixtures runnable (see ingest.py).
    min_input_pages: int = Field(default=5, ge=1)
    max_input_pages: int = Field(default=30, ge=1)
    enforce_page_count_limits: bool = False
    default_target_language: str = "hi"

    # --- Derived directories ----------------------------------------------
    @cached_property
    def inbox_dir(self) -> Path:
        return self.project_root / self.data_dir / "inbox"

    @cached_property
    def work_dir(self) -> Path:
        return self.project_root / self.data_dir / "work"

    @cached_property
    def exports_dir(self) -> Path:
        return self.project_root / self.data_dir / "exports"


def load_settings() -> Settings:
    """Build :class:`Settings` from the process environment."""
    return Settings()
