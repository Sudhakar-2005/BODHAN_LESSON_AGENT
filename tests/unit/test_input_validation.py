"""Phase 3B-2 Task 4: actionable input validation, no provider calls."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from bodhan.pipeline.errors import PipelineError
from bodhan.pipeline.stages import ingest
from tests.conftest import make_context, make_pdf


def test_corrupted_pdf_fails_with_actionable_error(tmp_path: Path) -> None:
    source = tmp_path / "broken.pdf"
    source.write_bytes(b"%PDF-1.4 not really a pdf")
    ctx = make_context(tmp_path)
    ctx.chapter = ctx.chapter.model_copy(update={"source_path": str(source)})

    with pytest.raises(PipelineError, match="corrupted or unreadable PDF"):
        asyncio_run(ingest.run(ctx))


def test_empty_pdf_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class EmptyPDF:
        page_count = 0
        needs_pass = False

        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            return None

    source = tmp_path / "empty.pdf"
    source.write_bytes(b"%PDF-1.4")
    monkeypatch.setattr("pymupdf.open", lambda path: EmptyPDF())
    ctx = make_context(tmp_path)
    ctx.chapter = ctx.chapter.model_copy(update={"source_path": str(source)})
    ctx.chapter = ctx.chapter.model_copy(update={"source_path": str(source)})

    with pytest.raises(PipelineError, match="PDF has no pages"):
        asyncio_run(ingest.run(ctx))


def test_password_protected_pdf_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class LockedPDF:
        needs_pass = True

        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            return None

    source = tmp_path / "locked.pdf"
    source.write_bytes(b"%PDF-1.4")
    monkeypatch.setattr("pymupdf.open", lambda path: LockedPDF())
    ctx = make_context(tmp_path)
    ctx.chapter = ctx.chapter.model_copy(update={"source_path": str(source)})
    with pytest.raises(PipelineError, match="password-protected"):
        asyncio_run(ingest.run(ctx))


def test_unsupported_image_format_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "picture.gif"
    source.write_bytes(b"GIF89a")
    ctx = make_context(tmp_path)
    ctx.chapter = ctx.chapter.model_copy(update={"source_path": str(source)})

    with pytest.raises(PipelineError, match="unsupported source type"):
        asyncio_run(ingest.run(ctx))


def test_corrupted_image_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "broken.png"
    source.write_bytes(b"not a png")
    ctx = make_context(tmp_path)
    ctx.chapter = ctx.chapter.model_copy(update={"source_path": str(source)})

    with pytest.raises(PipelineError, match="corrupted or unreadable image"):
        asyncio_run(ingest.run(ctx))


def test_empty_image_directory_is_rejected(tmp_path: Path) -> None:
    directory = tmp_path / "pages"
    directory.mkdir()
    ctx = make_context(tmp_path)
    ctx.chapter = ctx.chapter.model_copy(update={"source_path": str(directory)})

    with pytest.raises(PipelineError, match="no supported page images"):
        asyncio_run(ingest.run(ctx))


def test_mixed_folder_keeps_valid_pages_and_skips_corrupt_images(tmp_path: Path) -> None:
    directory = tmp_path / "pages"
    directory.mkdir()
    Image.new("RGB", (40, 60), "white").save(directory / "01.png")
    (directory / "02.png").write_bytes(b"broken")
    Image.new("RGB", (40, 60), "white").save(directory / "03.png")
    ctx = make_context(tmp_path)
    ctx.chapter = ctx.chapter.model_copy(update={"source_path": str(directory)})

    result = asyncio_run(ingest.run(ctx))

    assert result["pages"] == 2


@pytest.mark.parametrize(
    ("pages", "should_fail"),
    [(4, True), (5, False), (30, False), (31, True)],
)
def test_page_count_enforcement_boundaries(tmp_path: Path, pages: int, should_fail: bool) -> None:
    """Assignment range is 5-30 pages, enforced before any provider call."""
    source = make_pdf(tmp_path / f"doc-{pages}.pdf", pages=pages)
    ctx = make_context(tmp_path)
    ctx.chapter = ctx.chapter.model_copy(update={"source_path": str(source)})
    ctx.settings = ctx.settings.model_copy(update={"enforce_page_count_limits": True})

    if should_fail:
        with pytest.raises(PipelineError, match="supported range is 5-30"):
            asyncio_run(ingest.run(ctx))
        assert ctx.ocr.calls == []  # no provider call for an invalid document
    else:
        result = asyncio_run(ingest.run(ctx))
        assert result["pages"] == pages


def test_assignment_cli_enforces_page_range_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The assignment CLI passes enforce_page_count_limits=True to settings."""
    import bodhan_lesson.cli as assignment_cli

    captured: list[object] = []

    def fake_load_settings():
        from bodhan.config import Settings

        return Settings(project_root=tmp_path, data_dir="data")

    def fake_build(settings, *, target_language=None):
        captured.append(settings)
        from tests.unit.test_bodhan_lesson_cli import _FakeProviders

        return _FakeProviders()

    async def fake_run(ctx):
        return {"status": "done"}

    monkeypatch.setattr(assignment_cli, "load_settings", fake_load_settings)
    monkeypatch.setattr(assignment_cli, "build_providers", fake_build)
    monkeypatch.setattr(assignment_cli, "run_pipeline", fake_run)
    monkeypatch.delenv("BODHAN_ALLOW_SHORT_DOCUMENT", raising=False)

    source = tmp_path / "chapter.pdf"
    source.write_bytes(b"%PDF fake")
    out = tmp_path / "out"
    out.mkdir()
    (tmp_path / "data/exports/chapter-hi").mkdir(parents=True)
    (tmp_path / "data/exports/chapter-hi/lesson.md").write_text("# hi", encoding="utf-8")

    result = assignment_cli.run_lesson(
        input_path=source,
        langs="hi",
        out=out,
        source_lang="en",
        glossary=None,
        voices=None,
        code_mixed=False,
    )
    assert result is None
    assert all(getattr(s, "enforce_page_count_limits", False) for s in captured)


def test_pages_above_supported_range_are_rejected(tmp_path: Path) -> None:
    source = make_pdf(tmp_path / "many.pdf", pages=31)
    ctx = make_context(tmp_path)
    ctx.chapter = ctx.chapter.model_copy(update={"source_path": str(source)})

    with pytest.raises(PipelineError, match="supported range is 5-30"):
        asyncio_run(ingest.run(ctx))


def test_pages_below_range_can_be_enforced(tmp_path: Path) -> None:
    source = make_pdf(tmp_path / "short.pdf", pages=2)
    ctx = make_context(tmp_path)
    ctx.chapter = ctx.chapter.model_copy(update={"source_path": str(source)})
    ctx.settings = ctx.settings.model_copy(update={"enforce_page_count_limits": True})

    with pytest.raises(PipelineError, match="supported range is 5-30"):
        asyncio_run(ingest.run(ctx))


def asyncio_run(coro):  # tiny helper keeps the tests readable
    import asyncio

    return asyncio.run(coro)


def test_no_secret_material_in_validation_errors(tmp_path: Path) -> None:
    source = tmp_path / "broken.pdf"
    source.write_bytes(b"broken")
    ctx = make_context(tmp_path)
    ctx.chapter = ctx.chapter.model_copy(update={"source_path": str(source)})

    with pytest.raises(PipelineError) as exc_info:
        asyncio_run(ingest.run(ctx))
    assert "KEY" not in str(exc_info.value)
    assert "Bearer" not in str(exc_info.value)
