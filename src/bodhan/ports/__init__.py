"""Provider interfaces (ports).

Adapters implement these; stages depend only on the interfaces, so any provider
can be swapped or faked in tests without touching pipeline logic.
"""

from bodhan.ports.artifact_store import ArtifactStore
from bodhan.ports.llm import LLMPort
from bodhan.ports.ocr import OCRPort
from bodhan.ports.tts import TTSPort

__all__ = ["ArtifactStore", "LLMPort", "OCRPort", "TTSPort"]
