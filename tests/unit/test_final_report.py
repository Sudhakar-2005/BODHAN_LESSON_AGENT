"""Deterministic offline tests for report.json aggregation (Phase 2C)."""

from __future__ import annotations

import json
import wave
from pathlib import Path
from typing import Any

import pytest

from bodhan.reporting import ReportBuilder, write_report


def _write(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _wav(path: Path, frames: int = 8000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        handle.writeframes(b"\x00" * frames * 2)
    return path


def seed(root: Path, *, target: str = "hi", broken: bool = False) -> dict[str, Path]:
    """Build a minimal but complete artifact tree with known values."""
    work = root / "work"
    export = root / "export"
    p2a = root / "phase2a"
    audio = root / "audio"
    paths = {"work": work, "export": export, "p2a": p2a, "audio": audio}

    def stage(name: str, status: str = "done", **artifacts: Any) -> dict[str, Any]:
        return {
            "status": status,
            "completed_at": "2026-11-17T05:00:00+00:00",
            "duration_s": 1.0,
            "artifacts": artifacts,
        }

    _write(
        work / "manifest.json",
        {
            "chapter_id": "unit-chapter",
            "status": "done",
            "stages": {
                "ingest": stage("ingest", pages=2, source="sample.pdf"),
                "extract": stage("extract", provider="indic-ocr"),
                "narrate": stage("narrate", segments=2, skipped_empty=0, voice="Kavya"),
                "compose": stage("compose", sections=2),
            },
        },
    )
    _write(
        work / "ingest/pages.json",
        [
            {"page_index": 0, "source_path": "/inbox/sample.pdf"},
            {"page_index": 1, "source_path": "/inbox/sample.pdf"},
        ],
    )
    blocks = [
        {
            "block_id": "b1",
            "page_index": 0,
            "text": "Source one",
            "reading_order": 0,
            "confidence": 0.9,
            "flags": [],
        },
        {
            "block_id": "b2",
            "page_index": 0,
            "text": "",
            "reading_order": 1,
            "confidence": 0.6,
            "flags": ["low_confidence", "untranslatable"],
        },
    ]
    _write(
        work / "extract/results.json",
        [
            {
                "page_index": 0,
                "blocks": blocks,
                "provider": "indic-ocr",
                "provider_request_id": "ocr-req-0",
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            },
            {
                "page_index": 1,
                "blocks": [],
                "provider": "indic-ocr",
                "provider_request_id": "ocr-req-1",
                "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            },
        ],
    )
    _write(work / "structure/blocks.json", blocks)
    _write(work / "quality/blocks.json", blocks)
    _write(
        work / "quality/quality_report.json",
        {
            "flagged": 1,
            "entries": [
                {
                    "block_id": "b2",
                    "page_index": 0,
                    "confidence": 0.6,
                    "flags": ["low_confidence", "untranslatable"],
                    "reason": "empty low-confidence block",
                }
            ],
        },
    )
    _write(
        work / "translate/translations.json",
        {
            "target_language": target,
            "items": {"b1": "अनुवाद एक", "b2": ""},
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            "verbatim_blocks": 0,
            "empty_blocks": 1,
        },
    )
    _write(
        work / "compose/plan.json",
        {"sections": [{"heading_en": "First", "block_ids": ["b1", "b2"]}]},
    )
    _write(
        work / "compose/lesson.json",
        {
            "lesson_id": "unit-lesson",
            "source_language_code": "en",
            "target_language_code": target,
            "markdown": "# Title\n## Section One\n## Section Two\n",
            "scaffold": [
                {
                    "item_id": "title",
                    "kind": "title",
                    "source_en": "Title",
                    "text": "शीर्षक",
                    "provider": "bodhan-translate",
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
                }
            ],
        },
    )
    wave_a = _wav(work / "narrate/seg-01.wav")
    wave_b = _wav(work / "narrate/seg-02.wav", frames=16000)
    _write(
        work / "narrate/segments.json",
        [
            {
                "segment_id": "seg-01",
                "text": "Section One. first",
                "audio_path": str(wave_a),
                "duration_ms": 1000,
            },
            {
                "segment_id": "seg-02",
                "text": "Section Two. second",
                "audio_path": str(wave_b),
                "duration_ms": 2000,
            },
        ],
    )
    # Full artifact paths in the manifest aggregate are deliberately inflated:
    # the report must sum only the per-page/per-block evidence.
    _write(
        export / "manifest.json",
        {
            "status": "done",
            "stages": {
                "extract": {
                    "status": "done",
                    "artifacts": {"usage": {"total_tokens": 999}},
                }
            },
        },
    )
    _write(
        p2a / "transcribe/manifest.json",
        {
            "status": "done",
            "target_language": target,
            "resumed_cached": 2,
            "failed_segments": [],
            "total_latency_ms": 100,
        },
    )
    # Records arrive reversed: pairing must use section_id.
    _write(
        p2a / "transcribe/segment-01.json",
        {
            "section_id": "seg-01",
            "target_language": target,
            "status": "ok",
            "transcription_text": "abc",
            "provider": "bodhan-transcribe",
            "model": "indic-transcribe",
            "latency_ms": 60,
            "request_id": None,
            "usage": {},
            "raw": [{"text": "abc", "usage": None}],
        },
    )
    _write(
        p2a / "transcribe/segment-02.json",
        {
            "section_id": "seg-02",
            "target_language": target,
            "status": "ok",
            "transcription_text": "de",
            "provider": "bodhan-transcribe",
            "model": "indic-transcribe",
            "latency_ms": 40,
            "request_id": None,
            "usage": {},
            "raw": [{"text": "de", "usage": None}],
        },
    )
    cer_manifest = {
        "status": "done",
        "macro_average_cer": 0.1234,
        "macro_average_cer_percentage": 12.34,
        "micro_average_cer": 0.1,
        "micro_average_cer_percentage": 10.0,
        "normalization_policy": {"id": "nfc+ws-collapse-v1", "description": "policy"},
        "thresholds": {"excellent_max_cer_percentage": 5.0},
        "formula": "(S+D+I)/reference_character_count",
        "total_segments": 2,
        "successful_evaluations": 2,
        "failed_evaluations": 0,
        "segments_with_unavailable_reference": 0,
        "languages": {target: {"segments": 2}},
    }
    _write(p2a / "evaluation/cer/manifest.json", cer_manifest)
    _write(
        p2a / "evaluation/cer/segment-01.json",
        {
            "section_id": "seg-01",
            "status": "ok",
            "reference_character_count": 17,
            "transcription_character_count": 19,
            "substitutions": 1,
            "deletions": 2,
            "insertions": 3,
            "edit_distance": 6,
            "cer": 0.125,
            "cer_percentage": 12.5,
            "quality": "Needs Review",
            "normalization_policy": "nfc+ws-collapse-v1",
            "reference_source": "narrate/segments.json:AudioSegment.text",
        },
    )
    _write(
        p2a / "evaluation/cer/segment-02.json",
        {
            "section_id": "seg-02",
            "status": "ok",
            "reference_character_count": 10,
            "transcription_character_count": 10,
            "substitutions": 0,
            "deletions": 0,
            "insertions": 0,
            "edit_distance": 0,
            "cer": 0.0,
            "cer_percentage": 0.0,
            "quality": "Excellent",
            "normalization_policy": "nfc+ws-collapse-v1",
        },
    )
    timeline = {
        "sections": [
            {
                "section_id": "seg-01",
                "section_title": "Section One",
                "audio_filename": "seg-01.wav",
                "start_time_seconds": 0.0,
                "end_time_seconds": 1.0,
                "duration_seconds": 1.0,
                "sequence_number": 1,
            },
            {
                "section_id": "seg-02",
                "section_title": "Section Two",
                "audio_filename": "seg-02.wav",
                "start_time_seconds": 1.0,
                "end_time_seconds": 3.0,
                "duration_seconds": 2.0,
                "sequence_number": 2,
            },
        ],
        "total_duration_seconds": 3.0,
        "marker_source": "sidecar only",
    }
    _write(audio / "timeline.json", timeline)
    _write(
        audio / "assembly_manifest.json",
        {
            "status": "done",
            "output": {
                "filename": "full_lesson.wav",
                "duration_seconds": 3.0,
                "format": {"container": "WAV", "channels": 1, "sample_rate_hz": 8000},
            },
            "section_count": 2,
            "missing_sections": [],
            "failed_sections": [],
            "embedded_chapters": False,
            "embedded_chapters_reason": "WAV sidecar only",
            "reused_on_rerun": False,
        },
    )
    paths.update(
        {
            "cer_manifest": p2a / "evaluation/cer/manifest.json",
            "narration_a": wave_a,
        }
    )
    return paths


def builder(paths: dict[str, Path], **kwargs: Any) -> ReportBuilder:
    return ReportBuilder(
        work_dir=paths["work"],
        export_dir=paths["export"],
        transcribe_dir=paths["p2a"] / "transcribe",
        evaluation_dir=paths["p2a"] / "evaluation/cer",
        assembly_dir=paths["audio"],
        commit="6f58961",
        **kwargs,
    )


# --- aggregation ------------------------------------------------------------


def test_aggregates_run_metadata_across_stages(tmp_path: Path) -> None:
    report = builder(seed(tmp_path)).build()
    metadata = report["run_metadata"]

    assert metadata["run_id"] == "unit-chapter-hi"
    assert metadata["source_filename"] == "sample.pdf"
    assert metadata["source_language"] == "en"
    assert metadata["target_language"] == "hi"
    assert metadata["page_count"] == 2
    assert metadata["total_pipeline_duration_s"] == 4.0
    assert metadata["pipeline_status"] == "done"
    assert metadata["application"]["commit"] == "6f58961"
    assert metadata["application"]["working_tree_clean"] is None
    assert "unknown" in metadata["application"]["commit_basis"]
    # generated_at is true wall-clock provenance, not a manifest timestamp.
    assert metadata["generated_at"] != metadata["completion_time"]
    assert metadata["evidence_as_of"] == metadata["completion_time"]
    assert metadata["generated_at_basis"].startswith("UTC wall-clock")


def test_per_page_ocr_information_uses_actual_provider_evidence(tmp_path: Path) -> None:
    report = builder(seed(tmp_path)).build()
    pages = report["per_page_ocr"]

    assert [page["page_number"] for page in pages] == [1, 2]
    assert pages[0]["extracted_block_count"] == 2
    assert pages[0]["low_confidence_block_count"] == 1
    assert pages[0]["confidence"]["min"] == 0.6
    assert pages[0]["confidence"]["max"] == 0.9
    assert pages[0]["provider_request_id"] == "ocr-req-0"
    assert pages[0]["ocr_status"] == "done"
    assert pages[0]["retry_count"] is None
    assert pages[0]["latency_ms"] is None
    assert "per_page_ocr[].retry_count" in report["unavailable_fields"]


def test_structure_quality_reports_actual_flags_only(tmp_path: Path) -> None:
    report = builder(seed(tmp_path)).build()
    structure = report["structure_and_quality"]

    assert structure["block_count"] == 2
    assert structure["unresolved_or_rejected_blocks"] == ["b2"]
    assert structure["empty_text_blocks"] == ["b2"]
    assert structure["quality_flags"] == {"low_confidence": 1, "untranslatable": 1}
    assert structure["narrated_section_count"] == 2
    assert structure["reading_order_issues"] == []
    assert structure["cross_page_continuity_issues"] is None
    assert "cross_page_continuity_issues" in "".join(report["unavailable_fields"])


def test_translation_and_scaffold_are_separated(tmp_path: Path) -> None:
    report = builder(seed(tmp_path)).build()
    translations = report["translations"]

    assert [item["category"] for item in translations["body"]] == ["body", "body"]
    assert translations["body"][0]["section_id"] == "section-1"
    assert translations["body"][0]["source_character_count"] == len("Source one")
    assert translations["body"][1]["translation_status"] == "empty"
    assert [item["category"] for item in translations["scaffold"]] == ["scaffold"]
    assert translations["scaffold"][0]["provider"] == "bodhan-translate"
    assert translations["scaffold"][0]["translated_character_count"] == len("शीर्षक")
    assert translations["body_usage"]["total_tokens"] == 4
    assert (
        translations["body_usage"]["total_tokens"]
        != translations["scaffold"][0]["usage"]["total_tokens"]
    )


def test_narration_and_transcription_are_paired_by_ids(tmp_path: Path) -> None:
    report = builder(seed(tmp_path)).build()

    assert [item["section_id"] for item in report["narration"]] == ["seg-01", "seg-02"]
    assert [item["section_id"] for item in report["transcription"]] == ["seg-01", "seg-02"]
    assert [item["audio_filename"] for item in report["narration"]] == [
        "seg-01.wav",
        "seg-02.wav",
    ]
    assert report["transcription"][0]["transcript_character_count"] == 3
    assert report["transcription"][1]["transcript_character_count"] == 2
    assert report["narration"][0]["provider"] is None
    assert report["narration"][0]["model"] == "indic-speak"


def test_cer_values_are_reused_verbatim_with_macro_and_micro(tmp_path: Path) -> None:
    report = builder(seed(tmp_path)).build()
    cer = report["cer_evaluation"]
    first = cer["per_section"][0]

    assert first["reference_character_count"] == 17  # deliberately not len("Section One. first")
    assert first["cer_percentage"] == 12.5
    assert first["quality_band"] == "Needs Review"
    assert first["normalization_policy"] == "nfc+ws-collapse-v1"
    assert cer["macro_average_cer_percentage"] == 12.34
    assert cer["micro_average_cer_percentage"] == 10.0
    assert cer["macro_average_cer_percentage"] != cer["micro_average_cer_percentage"]
    assert cer["source"].startswith("Phase 2B")


def test_hindi_tamil_bengali_results_remain_separate(tmp_path: Path) -> None:
    paths = seed(tmp_path)
    hi = builder(paths).build()
    ta_paths = seed(tmp_path / "ta", target="ta")
    ta = builder(ta_paths).build()
    bn_paths = seed(tmp_path / "bn", target="bn")
    bn = builder(bn_paths).build()

    assert hi["run_metadata"]["target_language"] == "hi"
    assert ta["run_metadata"]["target_language"] == "ta"
    assert bn["run_metadata"]["target_language"] == "bn"
    assert hi["run_metadata"]["run_id"] != ta["run_metadata"]["run_id"]
    assert set(hi["cer_evaluation"]["languages"]) == {"hi"}
    assert set(ta["cer_evaluation"]["languages"]) == {"ta"}
    assert set(bn["cer_evaluation"]["languages"]) == {"bn"}


def test_audio_assembly_section_mirrors_timeline_without_embedded_chapters(
    tmp_path: Path,
) -> None:
    report = builder(seed(tmp_path)).build()
    assembly = report["audio_assembly"]

    assert assembly["assembly_status"] == "done"
    assert assembly["final_audio_filename"] == "full_lesson.wav"
    assert assembly["total_duration_seconds"] == 3.0
    assert [section["sequence_number"] for section in assembly["section_timeline"]] == [1, 2]
    assert assembly["missing_sections"] == []
    assert assembly["embedded_chapters"] is False
    assert "sidecar" in assembly["embedded_chapters_note"]


# --- cost, usage and retries ------------------------------------------------


def test_cost_usage_separates_actual_reported_and_estimated(tmp_path: Path) -> None:
    report = builder(seed(tmp_path)).build()
    usage = report["cost_and_usage"]

    assert usage["provider_reported_usage"]["ocr"]["totals"]["total_tokens"] == 18
    assert usage["provider_reported_usage"]["translation_body"]["totals"]["total_tokens"] == 4
    assert usage["provider_reported_usage"]["translation_scaffold"]["totals"]["total_tokens"] == 3
    assert usage["provider_reported_usage"]["narration"] is None
    assert usage["provider_reported_usage"]["transcription"] is None
    assert usage["provider_reported_cost"] is None
    estimate = usage["estimated_cost"]["transcription"]
    assert estimate["label"].startswith("ESTIMATE")
    assert estimate["audio_minutes"] == pytest.approx(3 / 60)
    assert estimate["amount"] == pytest.approx(round((3 / 60) * 0.10, 4))
    assert "indic_transcribe.py" in estimate["rate_source"]
    assert usage["estimated_cost"]["narration"] is None
    assert usage["request_ids"]["ocr"] == ["ocr-req-0", "ocr-req-1"]
    assert usage["request_ids"]["transcription"] is None
    assert usage["successful_request_counts"]["transcription"] == 2


def test_retry_summary_has_null_counts_and_no_double_counting(tmp_path: Path) -> None:
    report = builder(seed(tmp_path)).build()
    retries = report["retry_and_failure_summary"]

    assert retries["total_retries"] is None
    assert retries["recovered_failures"] is None
    assert all(value is None for value in retries["retries_by_stage"].values())
    assert retries["terminal_failures"] == []
    assert retries["skipped_or_resumed"]["transcribe_resumed_cached"] == 2
    assert retries["skipped_or_resumed"]["compose_skipped_empty"] is None
    assert "null retry fields are not zero" in retries["note"]


# --- integrity, atomicity and idempotence ----------------------------------


def test_missing_artifacts_and_failed_stages_are_not_reported_as_success(
    tmp_path: Path,
) -> None:
    paths = seed(tmp_path)
    (paths["p2a"] / "transcribe/segment-02.json").unlink()
    (paths["p2a"] / "transcribe/manifest.json").write_text(
        json.dumps({"status": "failed", "failed_segments": [{"section_id": "seg-02"}]}),
        encoding="utf-8",
    )
    (paths["work"] / "narrate/seg-01.wav").unlink()
    report = builder(paths).build()

    assert report["run_metadata"]["pipeline_status"] == "partial"
    assert report["transcription"][0]["section_id"] == "seg-01"
    assert len(report["transcription"]) == 1
    assert report["retry_and_failure_summary"]["terminal_failures"]
    assert report["narration"][0]["narration_status"] == "failed"
    assert "narrate/status.json" in "".join(report["unavailable_fields"].values())


def test_atomic_report_write_leaves_no_temporary_files(tmp_path: Path) -> None:
    paths = seed(tmp_path)
    report = builder(paths).build()
    output = paths["work"] / "report.json"
    output.write_text("old content", encoding="utf-8")

    write_report(output, report)

    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert list(output.parent.glob(".*.tmp")) == []


def test_idempotent_reruns_produce_identical_report_bytes(tmp_path: Path) -> None:
    paths = seed(tmp_path)
    first = builder(paths).build()
    second = builder(paths).build()
    assert (
        first["run_metadata"]["generated_at"] != second["run_metadata"]["generated_at"]
    )  # true wall-clock provenance

    def canonical(report: dict[str, object]) -> str:
        copy = json.loads(json.dumps(report, ensure_ascii=False))
        copy["run_metadata"].pop("generated_at", None)
        return json.dumps(copy, sort_keys=True, ensure_ascii=False)

    assert canonical(first) == canonical(second)

    out = paths["work"] / "report.json"
    write_report(out, first)
    first_bytes = out.read_bytes()
    write_report(out, second)
    assert out.read_bytes() == first_bytes  # no-op rerun preserves bytes
    assert (
        json.loads(out.read_text(encoding="utf-8"))["run_metadata"]["generated_at"]
        == first["run_metadata"]["generated_at"]
    )


def test_every_unavailable_null_has_an_exact_path_reason(tmp_path: Path) -> None:
    report = builder(seed(tmp_path)).build()
    nulls: list[str] = []

    def walk(value: object, path: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                walk(item, f"{path}.{key}" if path else str(key))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")
        elif value is None:
            nulls.append(path)

    walk(report, "")
    assert nulls
    for path in nulls:
        leaf = path.rsplit(".", 1)[-1].split("[", 1)[0]
        if leaf in {"error", "cer_unavailable_reason"}:
            continue  # null means "no error", not "unavailable"
        assert path in report["null_value_reasons"], path
        assert report["null_value_reasons"][path]


def test_commit_provenance_reflects_uncommitted_working_tree(tmp_path: Path) -> None:
    report = ReportBuilder(
        work_dir=seed(tmp_path)["work"],
        export_dir=tmp_path / "export",
        transcribe_dir=tmp_path / "phase2a/transcribe",
        evaluation_dir=tmp_path / "phase2a/evaluation/cer",
        assembly_dir=tmp_path / "audio",
        commit="abc123",
        working_tree_clean=False,
    ).build()
    application = report["run_metadata"]["application"]
    assert application["commit"] == "abc123"
    assert application["working_tree_clean"] is False
    assert "uncommitted" in application["commit_basis"]


def test_unavailable_fields_are_null_with_reasons(tmp_path: Path) -> None:
    report = builder(seed(tmp_path)).build()

    assert report["audio_assembly"]["embedded_chapters"] is False
    assert all(
        isinstance(reason, str) and reason for reason in report["unavailable_fields"].values()
    )
    assert report["unavailable_fields"]["narrate/status.json"].startswith("artifact not found")
