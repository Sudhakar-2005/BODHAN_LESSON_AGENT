"""LLM port.

The LLM is used ONLY for structural analysis, reading-order correction,
quality evaluation, translation and lesson composition -- never for OCR.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from pydantic import BaseModel


class LLMMessage(BaseModel):
    role: str
    content: str


class LLMResponse(BaseModel):
    text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    finish_reason: str | None = None


class LLMPort(ABC):
    """Minimal chat-completion contract shared by all LLM adapters."""

    @abstractmethod
    async def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        response_schema: type[BaseModel] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Run a single completion, optionally enforcing a pydantic schema."""
        raise NotImplementedError

    async def aclose(self) -> None:
        """Release underlying resources (optional)."""
        return None
