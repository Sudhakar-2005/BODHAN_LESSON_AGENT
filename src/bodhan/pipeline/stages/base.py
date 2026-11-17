"""Shared stage contract: the context every stage receives.

Stages are pure orchestration units: they read/write through the context's
artifact store and depend on providers only via ports, never concrete classes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from bodhan.config import Settings
from bodhan.domain.models import Chapter
from bodhan.ports.artifact_store import ArtifactStore
from bodhan.ports.llm import LLMPort
from bodhan.ports.ocr import OCRPort
from bodhan.ports.transcribe import TranscribePort
from bodhan.ports.translate import TranslatePort
from bodhan.ports.tts import TTSPort


@dataclass
class StageContext:
    """Everything a stage is allowed to touch."""

    settings: Settings
    store: ArtifactStore
    chapter: Chapter
    ocr: OCRPort
    llm: LLMPort
    translate: TranslatePort
    tts: TTSPort
    transcribe: TranscribePort
    logger: logging.Logger
    # Free-form handoff between stages (e.g. OCR results -> structure).
    state: dict[str, Any] = field(default_factory=dict)
