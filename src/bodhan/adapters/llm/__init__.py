"""LLM adapters.

Used only for structural analysis, reading-order correction, quality
evaluation, translation and lesson composition -- never for OCR.
"""

from bodhan.adapters.llm.gemini import GeminiLLMAdapter

__all__ = ["GeminiLLMAdapter"]
