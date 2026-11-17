"""End-to-end pipeline test with offline fake providers.

Covers the full DAG: ingest -> preprocess -> extract -> structure -> quality
-> translate -> compose -> narrate -> export, including checkpointing,
OCR caching and the flagged-block audit trail.
"""

from __future__ import annotations

import json
from pathlib import Path

from bodhan.domain.models import StageName
from bodhan.pipeline import run_pipeline
from bodhan.pipeline.stages import narrate
from tests.conftest import make_context, make_pdf


async def test_full_pipeline_produces_exports(tmp_path: Path) -> None:
    make_pdf(tmp_path / "chapter-01.pdf", pages=2)
    ctx = make_context(tmp_path)

    manifest = await run_pipeline(ctx)

    statuses = {name: entry["status"] for name, entry in manifest["stages"].items()}
    assert set(statuses) == {s.value for s in StageName}
    assert all(status == "done" for status in statuses.values())
    assert manifest["status"] == "done"

    export_dir = ctx.settings.exports_dir / "chapter-01"
    assert (export_dir / "lesson.md").read_text(encoding="utf-8").startswith("#")
    assert (export_dir / "lesson.html").is_file()
    assert (export_dir / "manifest.json").is_file()
    assert (export_dir / "quality_report.json").is_file()

    audio = sorted((export_dir / "audio").glob("*.mp3"))
    assert len(audio) >= 4  # objectives, body, summary, vocabulary, exercises

    # Low-confidence OCR output must survive all the way to the report.
    report = json.loads((export_dir / "quality_report.json").read_text(encoding="utf-8"))
    flagged_ids = {entry["block_id"] for entry in report["entries"]}
    assert flagged_ids == {"p0-b2", "p1-b2"}
    assert "low_confidence" in report["entries"][0]["flags"]


async def test_extract_caches_provider_responses(tmp_path: Path) -> None:
    make_pdf(tmp_path / "chapter-01.pdf", pages=2)
    ctx = make_context(tmp_path)
    ocr = ctx.ocr

    for stage in (StageName.INGEST, StageName.PREPROCESS, StageName.EXTRACT):
        await run_pipeline(ctx, only=stage)
    assert ocr.calls == [0, 1]

    # Forced re-run must be served entirely from the extract cache.
    await run_pipeline(ctx, only=StageName.EXTRACT, force=True)
    assert ocr.calls == [0, 1]


async def test_completed_stages_are_skipped_on_rerun(tmp_path: Path) -> None:
    make_pdf(tmp_path / "chapter-01.pdf", pages=1)
    ctx = make_context(tmp_path)

    await run_pipeline(ctx)
    manifest = await run_pipeline(ctx)  # no-op re-run

    assert all(entry["status"] == "done" for entry in manifest["stages"].values())
    assert isinstance(ctx.llm, object)


def test_split_sections_and_narration_text() -> None:
    markdown = "# पाठ\n## उद्देश्य\n- एक\n- दो\n## पाठ\nमुख्य विषय।\n"
    sections = narrate.split_sections(markdown)
    headings = [heading for heading, _ in sections]
    # The h1 has no body of its own, so only bodied sections become segments.
    assert headings == ["उद्देश्य", "पाठ"]

    spoken = narrate.for_narration("**महत्वपूर्ण** [शब्द](x) `code`\n- बिंदु\n")
    assert spoken == "महत्वपूर्ण शब्द code बिंदु"
