"""Phase 1 language-compliance tests (assignment Requirements 15 and 19).

Proves that:
1. no composition prompt tells the LLM to write in the target language;
2. no LLM-generated Indian-language content can enter the final lesson;
3. every target-language string in the lesson traces to Indic-Translate;
4. hi/ta/bn runs emit no hardcoded Hindi scaffolding and use labels and
   narration voices resolved for their target language;
5. English source content and equations are preserved verbatim.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from bodhan.adapters.tts.bodhan_speak import BodhanSpeakAdapter
from bodhan.config import Settings
from bodhan.domain.models import BlockKind, Lesson, OCRBlock, OCRResult, Page
from bodhan.pipeline import PipelineError, run_pipeline
from bodhan.pipeline.stages import compose, quality, structure
from bodhan.pipeline.stages.base import StageContext
from bodhan.ports.llm import LLMMessage, LLMResponse
from bodhan.ports.translate import TranslatePort, TranslationResult
from bodhan.wiring import build_providers
from tests.conftest import FakeLLM, FakeOCR, make_context, make_pdf

#: Hindi text an "evil" LLM might try to author (must never get through).
_HINDI = "यह हिंदी में लिखा गया पाठ है।"


class RecordingLLM(FakeLLM):
    """FakeLLM that records every message list sent to the model."""

    def __init__(self) -> None:
        super().__init__()
        self.messages: list[list[LLMMessage]] = []

    async def complete(
        self,
        messages: list[LLMMessage] | tuple[LLMMessage, ...],
        *,
        response_schema: type | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        self.messages.append(list(messages))
        return await super().complete(
            messages, response_schema=response_schema, temperature=temperature
        )


class MaliciousLLM(FakeLLM):
    """Injects Indian-language text into one field of the English plan."""

    def __init__(self, plant_path: str) -> None:
        super().__init__()
        self._plant_path = plant_path

    async def complete(
        self,
        messages: list[LLMMessage] | tuple[LLMMessage, ...],
        *,
        response_schema: type | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        response = await super().complete(
            messages, response_schema=response_schema, temperature=temperature
        )
        if "pedagogical lesson planner" in messages[0].content:
            plan = json.loads(response.text)
            _plant(plan, self._plant_path, _HINDI)
            response = LLMResponse(
                text=json.dumps(plan, ensure_ascii=False),
                model="malicious-llm",
                input_tokens=1,
                output_tokens=1,
            )
        return response


class EnglishOCR(FakeOCR):
    """English text only: any Indic script in the output is a hardcoding leak."""

    async def extract(self, page: Page) -> OCRResult:
        result = await super().extract(page)
        texts = [
            f"Chapter 1 overview (page {page.page_index + 1})",
            f"English body paragraph {page.page_index + 1}.",
            "t0tally garbl3d",
        ]
        blocks = [
            block.model_copy(update={"text": texts[index]})
            for index, block in enumerate(result.blocks)
        ]
        return OCRResult(page_index=page.page_index, blocks=blocks, provider="english-fake-ocr")


class EquationOCR(EnglishOCR):
    """EnglishOCR plus an equation block that must stay verbatim."""

    async def extract(self, page: Page) -> OCRResult:
        result = await super().extract(page)
        blocks = [
            *result.blocks,
            OCRBlock(
                block_id=f"p{page.page_index}-beq",
                page_index=page.page_index,
                text="E = mc^2",
                kind=BlockKind.EQUATION,
                confidence=0.99,
            ),
        ]
        return OCRResult(page_index=page.page_index, blocks=blocks, provider=result.provider)


class PassThroughTranslate(TranslatePort):
    """Identity translation, for English -> English preservation checks."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def translate(self, text: str, *, target_language: str) -> TranslationResult:
        self.calls.append(text)
        return TranslationResult(
            text=text,
            target_language=target_language,
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )

    async def aclose(self) -> None:
        return None


def _plant(plan: dict[str, Any], path: str, value: str) -> None:
    """Set a dotted path (list indices as int segments) inside the plan dict."""
    node: Any = plan
    *parents, last = path.split(".")
    for part in parents:
        node = node[int(part)] if isinstance(node, list) else node[part]
    if isinstance(node, list):
        node[int(last)] = value
    else:
        node[last] = value


def _load(ctx: StageContext, stage: str, name: str) -> Any:
    return json.loads((ctx.store.stage_dir(stage) / name).read_text(encoding="utf-8"))


# --- Proof 1: no prompt asks the LLM to write in the target language -------


async def test_no_prompt_instructs_writing_in_the_target_language(tmp_path: Path) -> None:
    make_pdf(tmp_path / "chapter-01.pdf", pages=1)
    ctx = make_context(tmp_path, target="ta")
    recorder = RecordingLLM()
    ctx.llm = recorder  # type: ignore[assignment]

    manifest = await run_pipeline(ctx)
    assert manifest["status"] == "done"

    # (a) static instruction prompts are English-only, never ask for
    # target-language authoring, and carry no hardcoded Hindi headings.
    for system_prompt in (structure._SYSTEM, quality._SYSTEM, compose._SYSTEM):
        lowered = system_prompt.lower()
        assert "target language" not in lowered, system_prompt[:120]
        assert "write in the target language" not in lowered
        assert not compose.contains_indian_script(system_prompt)
    # ...while the English-only requirement is explicit.
    assert "english only" in compose._SYSTEM.lower()
    assert "hindi" in compose._SYSTEM.lower()

    # (b) the messages actually sent for composition
    compose_calls = [
        call for call in recorder.messages if "pedagogical lesson planner" in call[0].content
    ]
    assert compose_calls, "compose prompt was never sent"
    for call in compose_calls:
        for message in call:
            assert "target language" not in message.content.lower()
            assert not re.search(r"(?i)write in the target language", message.content)
        user_message = call[1]
        # the payload does not even name the language the lesson targets
        assert user_message.content.startswith("Plan the lesson")
        assert '"target_language"' not in user_message.content


# --- Proof 2: LLM-generated Indian-language content is rejected ------------


@pytest.mark.parametrize(
    "plant_path",
    [
        "title_en",
        "sections.0.heading_en",
        "objectives_en.0",
        "summary_en",
        "glossary_en.0.term_en",
        "glossary_en.0.description_en",
        "exercises_en.0.question_en",
        "exercises_en.0.answer_en",
        "narration_en.0",
    ],
)
async def test_llm_indian_language_output_is_rejected(tmp_path: Path, plant_path: str) -> None:
    make_pdf(tmp_path / "chapter-01.pdf", pages=1)
    ctx = make_context(tmp_path)
    ctx.llm = MaliciousLLM(plant_path)  # type: ignore[assignment]

    with pytest.raises(PipelineError, match="Indian-language text"):
        await run_pipeline(ctx)

    lesson_path = ctx.store.stage_dir("compose") / "lesson.json"
    assert not lesson_path.is_file(), "a contaminated plan must not produce a lesson"


# --- Proof 3: every translated output traces to Indic-Translate ------------


async def test_every_lesson_line_is_traceable_to_indic_translate(tmp_path: Path) -> None:
    make_pdf(tmp_path / "chapter-01.pdf", pages=2)
    ctx = make_context(tmp_path, target="hi")

    manifest = await run_pipeline(ctx)
    assert manifest["status"] == "done"

    # FakeTranslate marks every translation with "[<target>] ".
    marker = "[hi] "
    export_dir = ctx.settings.exports_dir / "chapter-01"
    lesson_md = (export_dir / "lesson.md").read_text(encoding="utf-8")

    blocks = _load(ctx, "quality", "blocks.json")
    verbatim = {b["text"] for b in blocks if b["kind"] == "equation" and b["text"].strip()}
    for line in lesson_md.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lstrip("#*- \t").startswith(marker):
            continue  # routed through Indic-Translate
        assert any(text in stripped for text in verbatim), (
            f"line traces neither to Indic-Translate nor a verbatim block: {stripped!r}"
        )

    # scaffold provenance: English source -> Indic-Translate -> lesson text
    lesson = Lesson.model_validate(_load(ctx, "compose", "lesson.json"))
    assert lesson.scaffold
    for item in lesson.scaffold:
        assert item.provider == "bodhan-translate"
        assert item.source_en.strip()
        assert not compose.contains_indian_script(item.source_en)
        assert item.text.startswith(marker), item.item_id
        assert item.usage.get("total_tokens", 0) >= 1

    # UI labels are translated units as well
    for key in ("audio", "exercises", "glossary", "introduction", "objectives", "summary"):
        assert lesson.label(key).startswith(marker), key
    assert lesson.label("quality_warning").startswith(marker)

    # body blocks: translations keyed by block id, equations verbatim
    translations = _load(ctx, "translate", "translations.json")
    assert set(translations["items"]) == {b["block_id"] for b in blocks}
    for block in blocks:
        text = translations["items"][block["block_id"]]
        if block["kind"] == "equation" or not text:
            continue
        assert text.startswith(marker), block["block_id"]


# --- Proof 4: hi/ta/bn emit no Hindi scaffolding; labels are target-aware --


@pytest.mark.parametrize("target", ["hi", "ta", "bn"])
async def test_no_hindi_scaffolding_for_any_target(tmp_path: Path, target: str) -> None:
    make_pdf(tmp_path / "chapter-01.pdf", pages=1)
    ctx = make_context(tmp_path, target=target)
    ctx.ocr = EnglishOCR()  # type: ignore[assignment]

    manifest = await run_pipeline(ctx)
    assert manifest["status"] == "done"

    export_dir = ctx.settings.exports_dir / "chapter-01"
    lesson_md = (export_dir / "lesson.md").read_text(encoding="utf-8")
    lesson_html = (export_dir / "lesson.html").read_text(encoding="utf-8")

    # The exact Hindi strings that used to be hardcoded, gone everywhere.
    hardcoded = (
        "परिचय",
        "अध्याय के उद्देश्य",
        "सारांश",
        "मुख्य शब्दावली",
        "अभ्यास प्रश्न",
        "पाठ का ऑडियो",
    )
    for leaked in hardcoded:
        assert leaked not in lesson_md, f"hardcoded Hindi {leaked!r} in lesson.md"
        assert leaked not in lesson_html, f"hardcoded Hindi {leaked!r} in lesson.html"

    # English source: no Indian script may appear in any output artifact.
    for text in (lesson_md, lesson_html):
        assert not compose.contains_indian_script(text), "Indic script leaked into output"

    # ...yet labels are target-language-aware (routed through Indic-Translate).
    lesson = Lesson.model_validate(_load(ctx, "compose", "lesson.json"))
    marker = f"[{target}] "
    for key in ("audio", "exercises", "glossary", "introduction", "objectives", "summary"):
        label = lesson.label(key)
        assert label.startswith(marker), f"label {key} for {target}: {label!r}"

    # narration is wired to Indic-Speak for the target language (only engine)
    providers = build_providers(ctx.settings, target_language=target)
    try:
        assert isinstance(providers.tts, BodhanSpeakAdapter)
        assert providers.tts._lang == target
    finally:
        await providers.aclose()

    assert f'lang="{target}"' in lesson_html


# --- Proof 5: English source content and equations are preserved -----------


async def test_english_source_and_equations_are_preserved(tmp_path: Path) -> None:
    make_pdf(tmp_path / "chapter-01.pdf", pages=1)
    ctx = make_context(tmp_path, target="en")
    ctx.ocr = EquationOCR()  # type: ignore[assignment]
    ctx.translate = PassThroughTranslate()  # type: ignore[assignment]

    manifest = await run_pipeline(ctx)
    assert manifest["status"] == "done"

    translations = _load(ctx, "translate", "translations.json")
    # equations never reach the translation API; every other block does
    assert translations["verbatim_blocks"] == 1
    assert translations["items"]["p0-beq"] == "E = mc^2"
    block_texts = {b["text"] for b in _load(ctx, "quality", "blocks.json")}
    sent_block_texts = [text for text in ctx.translate.calls if text in block_texts]
    expected_sent = {text for text in block_texts if text and text != "E = mc^2"}
    assert set(sent_block_texts) == expected_sent
    assert "E = mc^2" not in ctx.translate.calls

    lesson_md = (ctx.settings.exports_dir / "chapter-01" / "lesson.md").read_text(encoding="utf-8")
    assert "E = mc^2" in lesson_md
    assert "Chapter 1 overview (page 1)" in lesson_md
    assert "English body paragraph 1." in lesson_md

    # the pipeline artifacts still carry the untouched English source text
    by_id = {b["block_id"]: b["text"] for b in _load(ctx, "quality", "blocks.json")}
    assert by_id["p0-b0"] == "Chapter 1 overview (page 1)"
    assert by_id["p0-beq"] == "E = mc^2"


# --- Single-engine narration wiring (Indic-Speak only) --------------------


def test_settings_have_no_edge_tts_configuration() -> None:
    """Obsolete edge-tts settings were removed with the fallback engine."""
    settings = Settings()
    assert not hasattr(settings, "edge_tts_voice")
    assert not hasattr(settings, "edge_tts_rate")
    assert not hasattr(settings, "edge_tts_pitch")
    assert not hasattr(settings, "edge_tts_voice_for")


async def test_build_providers_uses_indic_speak_only() -> None:
    """The wiring constructs the Indic-Speak adapter directly - no wrapper."""
    settings = Settings()
    providers = build_providers(settings, target_language="ta")
    try:
        assert isinstance(providers.tts, BodhanSpeakAdapter)
        assert providers.tts._lang == "ta"  # resolved for the target language
        # a fallback wrapper would carry primary/fallback slots - it must not
        assert not hasattr(providers.tts, "fallback")
        assert not hasattr(providers.tts, "primary")
    finally:
        await providers.aclose()
