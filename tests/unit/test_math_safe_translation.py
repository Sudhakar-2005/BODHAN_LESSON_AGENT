"""Phase 4I: math-safe translation, stage status and checkpoint compatibility.

Deterministic fixtures only; no live API calls. The provider double records
exactly what text it is asked to translate so the tests can prove that math
spans never leave the pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from bodhan.document import (
    TextSegment,
    preserve_math_only,
    prose_to_translate,
    reassemble,
    segment_mixed_text,
    validate_math_spans,
)
from bodhan.domain.models import BlockKind, OCRBlock, StageName
from bodhan.pipeline.errors import PipelineError
from bodhan.pipeline.stages import compose
from bodhan.pipeline.stages import translate as translate_stage
from bodhan.pipeline.state import load_json, save_json
from tests.conftest import FakeLLM, make_context


class RecordingTranslate:
    """Echo provider that records every prose string it is asked to translate."""

    provider = "recording-translate"

    def __init__(self, handler=None) -> None:
        self.calls: list[str] = []
        self.handler = handler

    async def translate(self, text: str, *, target_language: str) -> Any:
        self.calls.append(text)
        if self.handler is not None:
            return self.handler(text)
        return SimpleNamespace(
            text=f"[{target_language}] {text}",
            provider="bodhan-translate",
            model="indic-translate",
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )

    async def aclose(self) -> None:
        return None


def block(block_id: str, text: str, kind: BlockKind = BlockKind.PARAGRAPH) -> OCRBlock:
    return OCRBlock(block_id=block_id, page_index=0, reading_order=0, text=text, kind=kind)


def seed(tmp_path: Path, blocks: list[OCRBlock], *, tables: dict[str, Any] | None = None) -> Any:
    ctx = make_context(tmp_path, chapter_id="math-safe", target="hi")
    pages = [{"page_index": 0, "source_path": "chapter.pdf"}]
    save_json(ctx.store, StageName.QUALITY, "pages.json", pages)
    save_json(ctx.store, StageName.QUALITY, "blocks.json", [b.model_dump() for b in blocks])
    save_json(ctx.store, StageName.QUALITY, "quality_report.json", {"flagged": 0, "entries": []})
    save_json(ctx.store, StageName.STRUCTURE, "pages.json", pages)
    save_json(ctx.store, StageName.STRUCTURE, "tables.json", tables or {})
    save_json(ctx.store, StageName.TRANSLATE, "pages.json", pages)
    return ctx


def checkpoint(ctx, block_id: str) -> dict[str, Any]:
    safe = "".join(c if c.isalnum() or c in ".-_" else "_" for c in block_id)
    path = ctx.store.stage_dir(StageName.TRANSLATE.value) / "blocks" / f"{safe}.json"
    return json.loads(path.read_text(encoding="utf-8"))


# --- 1-6: math span handling ------------------------------------------------


async def test_mixed_english_and_inline_latex() -> None:
    text = "The value of $x^2$ is positive."
    segments, ambiguous = segment_mixed_text(text)
    assert ambiguous is False
    assert [segment.kind for segment in segments] == ["prose", "math", "prose"]
    assert prose_to_translate(segments) == ["The value of ", " is positive."]
    rebuilt = reassemble(segments, ["मान ", " सकारात्मक है।"])
    assert rebuilt == "मान $x^2$ सकारात्मक है।"
    error, count = validate_math_spans(segments, rebuilt)
    assert error is None and count == 1


async def test_mixed_english_and_display_latex() -> None:
    for text in (
        "Consider $$\\frac{a}{b} = c$$ below.",
        "Consider \\[a + b = c\\] below.",
    ):
        segments, ambiguous = segment_mixed_text(text)
        assert ambiguous is False
        assert any(segment.kind == "math" for segment in segments)
        rebuilt = reassemble(segments, ["विचार करें ", " नीचे।"])
        error, _ = validate_math_spans(segments, rebuilt)
        assert error is None
        assert "$" in rebuilt or "\\[" in rebuilt


async def test_multiple_formulas_in_one_paragraph() -> None:
    text = "Given $a = 1$ and $b = 2$ then $c = 3$ holds."
    segments, ambiguous = segment_mixed_text(text)
    assert ambiguous is False
    math = [s.text for s in segments if s.kind == "math"]
    assert math == ["$a = 1$", "$b = 2$", "$c = 3$"]
    rebuilt = reassemble(segments, ["दिया गया ", " और ", " तब ", " सत्य है।"])
    error, count = validate_math_spans(segments, rebuilt)
    assert error is None and count == 3
    assert rebuilt == "दिया गया $a = 1$ और $b = 2$ तब $c = 3$ सत्य है।"


async def test_fractions_and_mathematical_operators() -> None:
    text = "Use $\\frac{1}{2} + \\frac{3}{4} \\leq 1$ here."
    segments, ambiguous = segment_mixed_text(text)
    assert ambiguous is False
    rebuilt = reassemble(segments, ["यहाँ ", " लागू करें।"])
    error, count = validate_math_spans(segments, rebuilt)
    assert error is None and count == 1
    assert "\\frac{1}{2}" in rebuilt


async def test_subscripts_and_superscripts() -> None:
    text = "See $x_1^{2} + y^{10}$ now."
    segments, ambiguous = segment_mixed_text(text)
    assert ambiguous is False
    rebuilt = reassemble(segments, ["अभी ", " देखें।"])
    error, _ = validate_math_spans(segments, rebuilt)
    assert error is None
    assert "x_1^{2}" in rebuilt and "y^{10}" in rebuilt


async def test_greek_symbols() -> None:
    # Greek literals are the subject of this test; escapes keep ruff's RUF001 quiet.
    alpha_beta_theta = "\u03b1 + \u03b2 = \u03b8"
    delta = "\u0394"
    text = f"Angles ${alpha_beta_theta}$ and ${delta} \\theta$ are shown."
    segments, ambiguous = segment_mixed_text(text)
    assert ambiguous is False
    rebuilt = reassemble(segments, ["कोण ", " और ", " दिखाए गए हैं।"])
    error, count = validate_math_spans(segments, rebuilt)
    assert error is None and count == 2
    assert alpha_beta_theta in rebuilt and f"{delta} \\theta" in rebuilt


async def test_plain_arithmetic_without_dollars_is_protected() -> None:
    text = "The sum 2 + 3 = 5 is correct."
    segments, ambiguous = segment_mixed_text(text)
    assert ambiguous is False
    assert any(segment.kind == "math" and "2 + 3 = 5" in segment.text for segment in segments)


# --- 7-8, 10: stage-level math handling -------------------------------------


@pytest.mark.asyncio
async def test_standalone_equation_block_is_verbatim_without_api(tmp_path: Path) -> None:
    ctx = seed(tmp_path, [block("p0-b0", "$x^2 + y^2 = z^2$", BlockKind.EQUATION)])
    provider = RecordingTranslate()
    ctx.translate = provider  # type: ignore[assignment]

    result = await translate_stage.run(ctx)

    assert provider.calls == []
    assert result["verbatim_blocks"] == 1
    assert checkpoint(ctx, "p0-b0")["translation"] == "$x^2 + y^2 = z^2$"


@pytest.mark.asyncio
async def test_math_only_block_is_preserved_without_api(tmp_path: Path) -> None:
    ctx = seed(tmp_path, [block("p0-b0", "$x^2 + y^2 = z^2$")])
    provider = RecordingTranslate()
    ctx.translate = provider  # type: ignore[assignment]

    await translate_stage.run(ctx)

    assert provider.calls == []
    assert checkpoint(ctx, "p0-b0")["translation"] == "$x^2 + y^2 = z^2$"


@pytest.mark.asyncio
async def test_provider_cannot_remove_math_during_mixed_translation(tmp_path: Path) -> None:
    ctx = seed(tmp_path, [block("p0-b0", "Let $\\frac{a}{b}$ be given.")])
    provider = RecordingTranslate(lambda text: SimpleNamespace(text="बिल्कुल बदल दिया", usage={}))
    ctx.translate = provider  # type: ignore[assignment]

    result = await translate_stage.run(ctx)

    assert result["status"] == "completed"
    translated = checkpoint(ctx, "p0-b0")["translation"]
    assert "$\\frac{a}{b}$" in translated
    assert all("$" not in call for call in provider.calls)  # math never sent


# --- 9: ambiguous boundaries -------------------------------------------------


@pytest.mark.asyncio
async def test_ambiguous_math_boundaries_fail_safely(tmp_path: Path) -> None:
    ctx = seed(tmp_path, [block("p0-b0", "An unbalanced $formula here")])
    provider = RecordingTranslate()
    ctx.translate = provider  # type: ignore[assignment]

    with pytest.raises(PipelineError, match="translation failed for every block"):
        await translate_stage.run(ctx)

    record = checkpoint(ctx, "p0-b0")
    assert record["status"] == "failed"
    assert "ambiguous mathematical boundaries" in record["error"]
    assert provider.calls == []


# --- 11-12: stage status and compose guard ----------------------------------


@pytest.mark.asyncio
async def test_partial_status_when_one_block_fails(tmp_path: Path) -> None:
    ctx = seed(
        tmp_path,
        [block("p0-b0", "good"), block("p0-b1", "An unbalanced $formula")],
    )
    ctx.translate = RecordingTranslate()  # type: ignore[assignment]

    result = await translate_stage.run(ctx)

    assert result["status"] == "partial"
    payload = load_json(ctx.store, StageName.TRANSLATE, "translations.json")
    assert payload["status"] == "partial"
    assert [item["block_id"] for item in payload["failed_blocks"]] == ["p0-b1"]


@pytest.mark.asyncio
async def test_compose_blocked_by_failed_translations(tmp_path: Path) -> None:
    ctx = seed(
        tmp_path,
        [block("p0-b0", "good"), block("p0-b1", "An unbalanced $formula")],
    )
    ctx.translate = RecordingTranslate()  # type: ignore[assignment]
    await translate_stage.run(ctx)

    ctx.llm = FakeLLM()  # type: ignore[assignment]
    with pytest.raises(PipelineError, match="translation stage is 'partial'"):
        await compose.run(ctx)


# --- 13-14: checkpoint compatibility ----------------------------------------


@pytest.mark.asyncio
async def test_successful_checkpoint_is_reused(tmp_path: Path) -> None:
    ctx = seed(tmp_path, [block("p0-b0", "plain prose")])
    provider = RecordingTranslate()
    ctx.translate = provider  # type: ignore[assignment]

    await translate_stage.run(ctx)
    first = checkpoint(ctx, "p0-b0")["translation"]
    result = await translate_stage.run(ctx)

    assert result["reused_blocks"] == 1
    assert len(provider.calls) == 1  # second run made no provider call
    assert checkpoint(ctx, "p0-b0")["translation"] == first


@pytest.mark.asyncio
async def test_failed_checkpoint_is_retried(tmp_path: Path) -> None:
    ctx = seed(tmp_path, [block("p0-b0", "An unbalanced $formula")])
    ctx.translate = RecordingTranslate()  # type: ignore[assignment]

    with pytest.raises(PipelineError):
        await translate_stage.run(ctx)
    assert checkpoint(ctx, "p0-b0")["status"] == "failed"

    fixed = RecordingTranslate()  # type: ignore[assignment]
    ctx.translate = fixed
    ctx.chapter = ctx.chapter.model_copy(update={"target_language_code": "ta"})
    # changing target changes the source hash; re-seed blocks for the new target
    save_json(
        ctx.store,
        StageName.QUALITY,
        "blocks.json",
        [block("p0-b0", "a valid $formula$ here").model_dump()],
    )
    result = await translate_stage.run(ctx)

    assert result["status"] == "completed"
    assert checkpoint(ctx, "p0-b0")["status"] == "ok"
    assert fixed.calls == ["a valid ", " here"]


# --- 15: table-cell math preservation ---------------------------------------


@pytest.mark.asyncio
async def test_table_cell_math_is_preserved(tmp_path: Path) -> None:
    tables = {
        "p0-b0": {
            "block_id": "p0-b0",
            "page_index": 0,
            "rows": 1,
            "columns": 2,
            "incomplete": False,
            "review_reason": None,
            "source_block_ids": ["p0-b0"],
            "cells": [
                {
                    "row": 0,
                    "column": 0,
                    "text": "Value of $x^2$",
                    "is_header": True,
                    "source_block_id": "p0-b0",
                },
                {
                    "row": 0,
                    "column": 1,
                    "text": "$$x^2 + y^2$$",
                    "is_header": True,
                    "source_block_id": "p0-b0",
                },
            ],
        }
    }
    ctx = seed(tmp_path, [block("p0-b0", "<table></table>", BlockKind.TABLE)], tables=tables)
    provider = RecordingTranslate()
    ctx.translate = provider  # type: ignore[assignment]

    result = await translate_stage.run(ctx)

    assert result["status"] == "completed"
    markdown = checkpoint(ctx, "p0-b0")["translation"]
    assert "$x^2$" in markdown
    assert "$$x^2 + y^2$$" in markdown
    assert all("$" not in call for call in provider.calls)


# --- pure helper edge cases --------------------------------------------------


def test_preserve_math_only_and_segment_contract() -> None:
    assert preserve_math_only("$x^2$") is True
    assert preserve_math_only("plain prose") is False
    assert preserve_math_only("unbalanced $x") is False
    segments, _ = segment_mixed_text("a $x$ b")
    assert "".join(segment.text for segment in segments) == "a $x$ b"
    assert all(isinstance(segment, TextSegment) for segment in segments)


def _neutral_rebuild(segments: list[TextSegment]) -> str:
    """Reassemble with neutral prose so only math-span handling is under test."""
    chunks = [
        f"word{index} "
        for index, segment in enumerate(segments)
        if segment.kind == "prose" and segment.text.strip()
    ]
    return reassemble(segments, chunks)


def test_single_math_span_is_validated_once() -> None:
    segments, _ = segment_mixed_text("Value $x$ here")
    rebuilt = _neutral_rebuild(segments)

    error, count = validate_math_spans(segments, rebuilt)

    assert error is None
    assert count == 1


def test_same_math_span_twice_is_accepted() -> None:
    segments, _ = segment_mixed_text("Lines $l$ and $m$; are $l$ and $m$ equal?")

    error, count = validate_math_spans(segments, _neutral_rebuild(segments))

    assert error is None
    assert count == 4  # two '$l$' plus two '$m$'


def test_multiple_distinct_spans_each_repeated_are_accepted() -> None:
    segments, _ = segment_mixed_text("$a$ then $b$ then $a$ then $b$")

    error, count = validate_math_spans(segments, _neutral_rebuild(segments))

    assert error is None
    assert count == 4


@pytest.mark.parametrize(
    ("text", "span"),
    [
        ("Two lines $l$ and $m$; are $l$ and $m$ perpendicular?", "$l$"),
        ("$x = y$ and $a = b$; hence $x = y$ again", "$x = y$"),
        ("BA $\\parallel$ ED and BC $\\parallel$ EF", "$\\parallel$"),
        ("bisectors of $\\angle$ EAB and $\\angle$ RBA", "$\\angle$"),
    ],
)
def test_tamil_repeated_math_examples_are_accepted(text: str, span: str) -> None:
    """Spans that failed the Tamil live run are legitimate repeated source math."""
    segments, _ = segment_mixed_text(text)
    rebuilt = _neutral_rebuild(segments)

    assert rebuilt.count(span) == 2
    error, count = validate_math_spans(segments, rebuilt)

    assert error is None
    assert count >= 2


def test_missing_one_occurrence_of_repeated_span_is_rejected() -> None:
    segments, _ = segment_mixed_text("$l$ and $l$ again")
    rebuilt = _neutral_rebuild(segments).replace("$l$", "L", 1)

    error, _count = validate_math_spans(segments, rebuilt)

    assert error is not None
    assert "missing or altered" in error


def test_extra_occurrence_introduced_during_reconstruction_is_rejected() -> None:
    segments, _ = segment_mixed_text("$x$ here")
    rebuilt = _neutral_rebuild(segments) + " plus an extra $x$"

    error, _count = validate_math_spans(segments, rebuilt)

    assert error is not None
    assert "extra math span" in error


def test_reordered_math_spans_are_rejected() -> None:
    segments = [
        TextSegment(kind="math", text="$a$"),
        TextSegment(kind="prose", text=" and "),
        TextSegment(kind="math", text="$b$"),
    ]

    error, _count = validate_math_spans(segments, "$b$ and $a$")

    assert error is not None
    assert "reordered" in error


def test_modified_math_expression_is_rejected() -> None:
    segments, _ = segment_mixed_text("$x = y$ and $z$")
    rebuilt = _neutral_rebuild(segments).replace("$x = y$", "$x = z$")

    error, _count = validate_math_spans(segments, rebuilt)

    assert error is not None
    assert "missing or altered" in error or "extra math span" in error


def test_inline_and_display_math_are_accepted() -> None:
    segments, _ = segment_mixed_text("Inline $a^2$ and display $$x + y$$ end")

    error, count = validate_math_spans(segments, _neutral_rebuild(segments))

    assert error is None
    assert count == 2


def test_math_only_block_is_accepted() -> None:
    text = "$x^2 + y^2 = z^2$"
    segments, _ = segment_mixed_text(text)

    error, count = validate_math_spans(segments, text)

    assert error is None
    assert count == 1


def test_span_split_keeps_repeated_math_out_of_provider_text() -> None:
    text = "Lines $l$ and $m$; are $l$ and $m$ equal?"
    segments, _ = segment_mixed_text(text)

    assert all("$" not in chunk for chunk in prose_to_translate(segments))
    error, count = validate_math_spans(segments, _neutral_rebuild(segments))
    assert error is None
    assert count == 4


def test_bengali_textbf_failure_example_is_preserved() -> None:
    """The Phase 6M Bengali block p2-b8 must reconstruct without a false failure."""
    text = (
        "\\textbf{Solution :} AOC is not a line, because "
        "$\\angle AOB + \\angle COB = 100^\\circ + 82^\\circ = 182^\\circ$, "
        "which is not equal to $180^\\circ$."
    )
    segments, ambiguous = segment_mixed_text(text)
    provider_text = prose_to_translate(segments)

    assert ambiguous is False
    assert not any("{" in chunk or "}" in chunk for chunk in provider_text)
    assert not any("\\textbf" in chunk for chunk in provider_text)
    rebuilt = _neutral_rebuild(segments)
    assert rebuilt.startswith("\\textbf{")
    assert "$180^\\circ$" in rebuilt
    error, count = validate_math_spans(segments, rebuilt)
    assert error is None
    assert count == 4  # '\textbf{', '}', two inline math spans


@pytest.mark.parametrize(
    "command",
    ["\\textit{Important}", "\\text{some text}", "\\mathrm{note}", "\\mbox{label}"],
)
def test_other_text_commands_keep_arguments_translatable(command: str) -> None:
    text = f"Before {command} after $x$ end"
    segments, _ = segment_mixed_text(text)
    provider_text = prose_to_translate(segments)

    assert command.split("{")[0] + "{" not in "".join(provider_text)
    assert any(
        part.strip("{} ") in " ".join(provider_text)
        for part in [command[command.find("{") + 1 : -1]]
    )
    rebuilt = _neutral_rebuild(segments)
    assert command.split("{")[0] + "{" in rebuilt
    assert "}" in rebuilt
    error, count = validate_math_spans(segments, rebuilt)
    assert error is None
    assert count >= 3


def test_multiple_text_commands_in_one_block() -> None:
    text = "\\textbf{First} then \\textit{Second} and $a$ end"
    segments, _ = segment_mixed_text(text)
    provider_text = prose_to_translate(segments)

    assert not any("{" in chunk or "}" in chunk for chunk in provider_text)
    rebuilt = _neutral_rebuild(segments)
    assert rebuilt.count("\\textbf{") == 1
    assert rebuilt.count("\\textit{") == 1
    error, count = validate_math_spans(segments, rebuilt)
    assert error is None
    assert count == 5


def test_text_command_adjacent_to_math_expression() -> None:
    text = "\\textbf{Solution :}$x = y$ next"
    segments, _ = segment_mixed_text(text)
    rebuilt = _neutral_rebuild(segments)

    assert "\\textbf{" in rebuilt and "$x = y$" in rebuilt
    error, count = validate_math_spans(segments, rebuilt)
    assert error is None
    assert count == 3


def test_repeated_math_spans_alongside_text_commands() -> None:
    text = "\\textbf{Given} $l$ and $m$; are $l$ and $m$ parallel?"
    segments, _ = segment_mixed_text(text)
    rebuilt = _neutral_rebuild(segments)

    assert rebuilt.count("$l$") == 2
    error, count = validate_math_spans(segments, rebuilt)
    assert error is None
    assert count == 6  # '\textbf{', '}', '$l$', '$m$', '$l$', '$m$'


def test_corrupted_math_near_text_command_still_fails() -> None:
    text = "\\textbf{Solution :} $x = y$ end"
    segments, _ = segment_mixed_text(text)
    rebuilt = _neutral_rebuild(segments).replace("$x = y$", "$x = z$")

    error, _count = validate_math_spans(segments, rebuilt)

    assert error is not None
    assert "missing or altered" in error or "extra math span" in error
