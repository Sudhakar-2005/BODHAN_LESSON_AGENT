"""OCR port.

EXCLUSIVE OCR ENGINE: Bodhan AI Indic-OCR. This interface exists so the engine
can be exercised and contract-tested in isolation; there is intentionally no
secondary/fallback OCR port implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from bodhan.domain.models import OCRResult, Page


class OCRPort(ABC):
    """Extract layout-aware text from a single page image."""

    @abstractmethod
    async def extract(self, page: Page) -> OCRResult:
        """Run OCR on ``page`` and return blocks in reading order.

        Implementations must:
        - populate per-block ``confidence`` when the provider reports it;
        - raise :class:`bodhan.adapters.ocr.errors.OCRProviderError` on failure;
        - never invent or "correct" text (correction belongs to later stages).
        """
        raise NotImplementedError

    async def aclose(self) -> None:
        """Release underlying resources (optional)."""
        return None
