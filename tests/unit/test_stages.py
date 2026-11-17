"""Unit tests for ingest/preprocess and the prompt helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from bodhan.pipeline import PipelineError
from bodhan.pipeline.prompts import chunked, parse_json_response
from bodhan.pipeline.stages import ingest, preprocess
from bodhan.pipeline.stages.base import StageContext
from bodhan.pipeline.state import load_pages
from tests.conftest import make_pdf


async def test_ingest_renders_pdf_pages(context: StageContext) -> None:
    make_pdf(Path(context.chapter.source_path), pages=3)

    result = await ingest.run(context)

    assert result["pages"] == 3
    pages = load_pages(context.store, ingest.NAME)
    assert len(pages) == 3
    for page in pages:
        assert page.image_path and Path(page.image_path).is_file()
        assert page.width and page.width > 100


async def test_ingest_rejects_missing_source(context: StageContext) -> None:
    with pytest.raises(PipelineError, match="not found"):
        await ingest.run(context)


async def test_ingest_rejects_unsupported_type(tmp_path: Path, context: StageContext) -> None:
    source = Path(context.chapter.source_path)
    source.unlink(missing_ok=True)
    source.with_suffix(".txt").write_text("plain text is not ingestible", encoding="utf-8")
    context.chapter.source_path = str(source.with_suffix(".txt"))

    with pytest.raises(PipelineError, match="unsupported source type"):
        await ingest.run(context)


async def test_preprocess_cleans_every_page(context: StageContext, tmp_path: Path) -> None:
    make_pdf(Path(context.chapter.source_path), pages=2)
    await ingest.run(context)

    result = await preprocess.run(context)

    assert result["pages"] == 2
    pages = load_pages(context.store, preprocess.NAME)
    for page in pages:
        image = cv2.imread(page.image_path)  # type: ignore[arg-type]
        assert image is not None
        # Output must be a binary (2-tone) image after adaptive threshold.
        assert len(np.unique(image)) <= 3


def test_estimate_skew_detects_rotated_text() -> None:
    image = np.full((400, 400), 255, dtype=np.uint8)
    for y in range(60, 360, 40):
        cv2.line(image, (30, y), (370, y - 20), 0, 3)  # tilted "text lines"
    angle = preprocess.estimate_skew_angle(image)
    assert abs(angle) > 1.0  # clearly tilted text must register a real angle


def test_deskew_noop_for_small_angle() -> None:
    image = np.full((100, 100), 128, dtype=np.uint8)
    out, applied = preprocess.deskew(image, 0.1)
    assert applied == 0.0
    assert out is image


def test_parse_json_response_handles_fences_and_prose() -> None:
    assert parse_json_response('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_response('Here you go: {"a": 2} done') == {"a": 2}
    with pytest.raises(PipelineError):
        parse_json_response("no json here")


def test_chunked_splits_evenly_and_remainder() -> None:
    assert chunked([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
    with pytest.raises(ValueError):
        chunked([1], 0)
