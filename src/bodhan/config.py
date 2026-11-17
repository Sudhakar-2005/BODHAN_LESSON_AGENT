"""Application configuration.

All tunables live here so that stages and adapters receive typed, validated
settings instead of reading environment variables ad hoc.
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
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

    # --- LLM (structure / quality / translation / composition only) --------
    llm_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    llm_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("LLM_API_KEY", "GEMINI_API_KEY"),
    )
    # Probed live (2026-11-17): gemini-2.5-* returns 404 (retired for new
    # users), gemini-3.8-flash returns persistent 503 (high demand), while
    # gemini-3.6-flash serves requests reliably (HTTP 200).
    llm_model: str = "gemini-3.6-flash"
    llm_timeout_s: float = 120.0
    llm_max_retries: int = 5

    # --- Narration ---------------------------------------------------------
    # Primary: Bodhan indic-speak (WAV). Fallback: edge-tts (MP3), used only
    # for transient transport failures - never for auth/quota/validation.
    bodhan_speak_voice: str = "Kavya"
    bodhan_speak_style: str | None = None
    bodhan_speak_chunk_seconds: float = Field(default=30.0, gt=0, le=30.0)
    speak_chars_per_second: float = Field(default=13.0, gt=0)  # chunking estimate
    edge_tts_voice: str = "hi-IN-SwaraNeural"
    edge_tts_rate: str = "+0%"
    edge_tts_pitch: str = "+0Hz"

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
    pipeline_stage_timeout_s: float = 900.0
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
