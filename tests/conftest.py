"""Shared test fakes: offline OCR/LLM/TTS providers and a context factory."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from bodhan.config import Settings
from bodhan.domain.models import AudioSegment, Chapter, OCRBlock, OCRResult, Page, QualityFlag
from bodhan.pipeline.prompts import parse_json_response
from bodhan.pipeline.stages.base import StageContext
from bodhan.ports.llm import LLMMessage, LLMResponse
from bodhan.ports.translate import TranslatePort, TranslationResult
from bodhan.ports.tts import TTSPort
from bodhan.services.artifact_store import FileArtifactStore


def make_pdf(path: Path, pages: int = 2) -> Path:
    """Create a small multi-page PDF for ingest/e2e tests."""
    import pymupdf

    doc = pymupdf.open()
    for index in range(pages):
        page = doc.new_page(width=595, height=842)  # A4 points
        page.insert_text((72, 100), f"Chapter test page {index + 1}", fontsize=14)
        page.insert_text((72, 140), "The quick brown fox jumps over the lazy dog.")
    doc.save(path)
    doc.close()
    return path


class FakeOCR:
    """Deterministic OCR: two clean blocks + one low-confidence block/page."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    async def extract(self, page: Page) -> OCRResult:
        self.calls.append(page.page_index)
        blocks = [
            OCRBlock(
                block_id=f"p{page.page_index}-b0",
                page_index=page.page_index,
                text=f"अध्याय 1 पृष्ठ {page.page_index}",
                kind="heading",  # type: ignore[arg-type]
                confidence=0.99,
            ),
            OCRBlock(
                block_id=f"p{page.page_index}-b1",
                page_index=page.page_index,
                text=f"यह एक परीक्षण अनुच्छेद है {page.page_index}.",
                kind="paragraph",  # type: ignore[arg-type]
                confidence=0.95,
            ),
            OCRBlock(
                block_id=f"p{page.page_index}-b2",
                page_index=page.page_index,
                text="t0tally garbl3d",
                kind="paragraph",  # type: ignore[arg-type]
                confidence=0.40,
                flags=[QualityFlag.LOW_CONFIDENCE],
            ),
        ]
        return OCRResult(page_index=page.page_index, blocks=blocks, provider="fake-ocr")

    async def aclose(self) -> None:
        return None


class FakeLLM:
    """Returns stage-appropriate canned JSON based on the system prompt."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def complete(
        self,
        messages: list[LLMMessage] | tuple[LLMMessage, ...],
        *,
        response_schema: type | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        system, user = messages[0].content, messages[1].content
        payload = parse_json_response(user.split("\n", 1)[1])

        if "document structure analyst" in system:
            self.calls.append("structure")
            blocks = [
                {
                    "id": b["id"],
                    "kind": "heading" if b["text"].startswith("अध्याय") else "paragraph",
                    "reading_order": index,
                }
                for index, b in enumerate(payload["blocks"])
            ]
            text = json.dumps({"blocks": blocks}, ensure_ascii=False)
        elif "OCR quality auditor" in system:
            self.calls.append("quality")
            results = [{"id": b["id"], "flags": [], "reason": "clean"} for b in payload["blocks"]]
            text = json.dumps({"results": results}, ensure_ascii=False)
        elif "textbook translator" in system:
            self.calls.append("translate")
            items = [{"id": b["id"], "text": b["text"]} for b in payload["blocks"]]
            text = json.dumps({"items": items}, ensure_ascii=False)
        elif "teacher authoring" in system:
            self.calls.append("compose")
            text = json.dumps(
                {
                    "title": payload["chapter_title"],
                    "markdown": (
                        "# पाठ\n"
                        "## अध्याय के उद्देश्य\n- उद्देश्य एक\n- उद्देश्य दो\n"
                        "## पाठ\n"
                        "यह मुख्य विषय है।\n"
                        "- पहला बिंदु\n- दूसरा बिंदु\n"
                        "## सारांश\nसंक्षेप में यह पाठ।\n"
                        "## मुख्य शब्दावली\n- शब्द: अर्थ\n"
                        "## अभ्यास प्रश्न\n- प्रश्न? उत्तर।\n"
                    ),
                },
                ensure_ascii=False,
            )
        else:
            raise AssertionError(f"unexpected system prompt: {system[:80]}")

        return LLMResponse(text=text, model="fake-llm", input_tokens=1, output_tokens=1)

    async def aclose(self) -> None:
        return None


class FakeTTS(TTSPort):
    """Writes a tiny placeholder MP3 file."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def synthesize(
        self,
        text: str,
        *,
        segment_id: str,
        lesson_id: str,
        output_path: Path,
    ) -> AudioSegment:
        self.calls.append(segment_id)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"ID3\x03\x00\x00\x00fake-mp3")
        return AudioSegment(
            segment_id=segment_id,
            lesson_id=lesson_id,
            text=text,
            audio_path=str(output_path),
        )


class FakeTranslate(TranslatePort):
    """Deterministic translate: prefixes text, records calls and usage."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def translate(self, text: str, *, target_language: str) -> TranslationResult:
        self.calls.append(target_language)
        return TranslationResult(
            text=f"[{target_language}] {text}",
            target_language=target_language,
            script="native",
            usage={
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        )

    async def aclose(self) -> None:
        return None


def make_context(tmp_path: Path, *, pages: int = 2, chapter_id: str = "chapter-01") -> StageContext:
    settings = Settings(project_root=tmp_path, data_dir="data")
    store = FileArtifactStore(settings.work_dir / chapter_id)
    chapter = Chapter(
        chapter_id=chapter_id,
        title="Chapter 1",
        source_path=str(tmp_path / f"{chapter_id}.pdf"),
        language_code="en",
        target_language_code="hi",
    )
    return StageContext(
        settings=settings,
        store=store,
        chapter=chapter,
        ocr=FakeOCR(),  # type: ignore[arg-type]
        llm=FakeLLM(),  # type: ignore[arg-type]
        translate=FakeTranslate(),  # type: ignore[arg-type]
        tts=FakeTTS(),  # type: ignore[arg-type]
        logger=logging.getLogger("test"),
    )


@pytest.fixture
def context(tmp_path: Path) -> StageContext:
    return make_context(tmp_path)
