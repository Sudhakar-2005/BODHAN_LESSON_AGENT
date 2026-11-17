"""Deterministic tests for per-section CER evaluation and aggregation.

All fixtures are inline and written to tmp_path as JSON; no API and no LLM.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from bodhan.config import Settings
from bodhan.evaluation import (
    NORMALIZATION_POLICY,
    REFERENCE_SOURCE,
    Evaluation,
    Thresholds,
    evaluate_sections,
    quality_label,
    write_evaluation,
)


def reference(section_id: str, text: str, *, audio: str | None = None) -> dict[str, Any]:
    return {
        "segment_id": section_id,
        "lesson_id": "lesson-01",
        "text": text,
        "audio_path": f"/narrate/{audio or section_id}.wav",
    }


def transcription(
    section_id: str | None,
    text: str,
    *,
    language: str = "hi",
    audio: str | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "target_language": language,
        "transcription_text": text,
        "status": "ok",
    }
    if section_id is not None:
        record["section_id"] = section_id
    if audio is not None:
        record["audio_file"] = audio
    return record


def by_id(records: list[dict[str, Any]]) -> dict[str | None, dict[str, Any]]:
    return {record["section_id"]: record for record in records}


# --- per-section records ---------------------------------------------------


def test_exact_pair_is_successful_and_recorded() -> None:
    evaluation = evaluate_sections(
        [reference("seg-01", "नमस्ते दुनिया", audio="seg-01.wav")],
        [transcription("seg-01", "नमस्ते दुनिया", audio="seg-01.wav")],
        expected_language="hi",
    )

    assert len(evaluation.records) == 1
    record = evaluation.records[0]
    assert record["section_id"] == "seg-01"
    assert record["target_language"] == "hi"
    assert record["audio_file"] == "seg-01.wav"
    assert record["reference_source"] == REFERENCE_SOURCE
    assert record["status"] == "ok"
    assert record["cer"] == 0.0
    assert record["cer_percentage"] == 0.0
    assert record["quality"] == "Excellent"
    assert record["normalization_policy"] == NORMALIZATION_POLICY["id"]
    assert "error" not in record


def test_pairing_uses_section_id_not_list_position() -> None:
    narrate = [reference("seg-01", "alpha"), reference("seg-02", "beta")]
    # Provider records arrive in the opposite order; their ids are authoritative.
    records = [transcription("seg-02", "betaX"), transcription("seg-01", "alpha")]

    evaluation = evaluate_sections(narrate, records, expected_language="hi")
    paired = by_id(evaluation.records)

    assert paired["seg-01"]["section_id"] == "seg-01"
    assert paired["seg-01"]["reference_source"] == REFERENCE_SOURCE
    assert paired["seg-01"]["edit_distance"] == 0  # its own reference "alpha"
    assert paired["seg-02"]["edit_distance"] == 1  # "beta" vs "betaX"
    # Both pairs must still be evaluated exactly once.
    assert evaluation.manifest["total_segments"] == 2


def test_missing_section_id_is_a_failure_without_inventing_an_id() -> None:
    evaluation = evaluate_sections(
        [reference("seg-01", "abc")],
        [transcription(None, "abc")],
        expected_language="hi",
    )

    assert len(evaluation.records) == 2
    unpaired = next(record for record in evaluation.records if record["section_id"] is None)
    assert unpaired["status"] == "failed"
    assert "section_id" in (unpaired["error"] or "")
    assert unpaired["cer"] is None
    assert unpaired["reference_character_count"] is None
    assert unpaired["edit_distance"] is None


def test_language_mismatch_is_rejected() -> None:
    evaluation = evaluate_sections(
        [reference("seg-01", "abc")],
        [transcription("seg-01", "abc", language="ta")],
        expected_language="hi",
    )

    record = evaluation.records[0]
    assert record["status"] == "failed"
    assert "language mismatch" in (record["error"] or "")
    assert "ta" in (record["error"] or "")
    assert "hi" in (record["error"] or "")
    assert record["cer"] is None  # no CER for mismatched languages


def test_missing_transcription_is_a_failure() -> None:
    evaluation = evaluate_sections([reference("seg-01", "abc")], [], expected_language="hi")

    record = evaluation.records[0]
    assert record["status"] == "failed"
    assert "transcription missing" in (record["error"] or "")
    assert record["transcription_character_count"] is None
    assert record["cer"] is None


def test_missing_reference_is_unavailable_and_never_fabricated() -> None:
    evaluation = evaluate_sections(
        [],
        [transcription("seg-99", "abc")],
        expected_language="hi",
    )

    record = evaluation.records[0]
    assert record["section_id"] == "seg-99"
    assert record["status"] == "unavailable"
    assert "reference unavailable" in (record["error"] or "")
    assert record["reference_source"] is None
    assert record["reference_character_count"] is None
    assert record["cer"] is None
    assert record["quality"] is None
    assert evaluation.manifest["segments_with_unavailable_reference"] == 1


def test_empty_reference_is_unavailable_with_reason() -> None:
    evaluation = evaluate_sections(
        [reference("seg-01", " \t\n ")],
        [transcription("seg-01", "abc")],
        expected_language="hi",
    )

    record = evaluation.records[0]
    assert record["status"] == "unavailable"
    assert "empty" in (record["error"] or "")
    assert "empty" in (record["cer_unavailable_reason"] or "")
    assert record["reference_character_count"] == 0
    assert record["transcription_character_count"] == 3
    assert record["edit_distance"] == 3
    assert record["cer"] is None
    assert record["cer_percentage"] is None


def test_empty_transcription_is_a_valid_full_deletion_result() -> None:
    evaluation = evaluate_sections(
        [reference("seg-01", "abc")],
        [transcription("seg-01", "")],
        expected_language="hi",
    )

    record = evaluation.records[0]
    assert record["status"] == "ok"
    assert record["deletions"] == 3
    assert record["cer"] == 1.0
    assert record["cer_percentage"] == 100.0
    assert record["quality"] == "Poor"


# --- artifact writing and manifest -----------------------------------------


def test_writes_independent_records_and_manifest(tmp_path: Path) -> None:
    narrate = [reference("seg-01", "नमस्ते"), reference("seg-02", "विश्व")]
    records = [transcription("seg-01", "नमस्ते"), transcription("seg-02", "विश्व")]
    output = tmp_path / "workspace" / "evaluation" / "cer"
    output.mkdir(parents=True)
    sentinel = output / "keep.txt"
    sentinel.write_text("untouched", encoding="utf-8")
    before = sentinel.read_bytes()

    evaluation = evaluate_sections(narrate, records, expected_language="hi")
    paths = write_evaluation(output, evaluation)

    assert [path.name for path in paths] == [
        "segment-01.json",
        "segment-02.json",
        "manifest.json",
    ]
    assert all(path.is_file() for path in paths)
    assert sentinel.read_bytes() == before  # unrelated files are left untouched
    segment = json.loads((output / "segment-01.json").read_text(encoding="utf-8"))
    assert segment["section_id"] == "seg-01"
    assert segment["status"] == "ok"
    assert segment["cer"] == 0.0

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "done"
    assert manifest["reference_source"] == REFERENCE_SOURCE
    assert manifest["normalization_policy"] == dict(NORMALIZATION_POLICY)
    assert manifest["thresholds"]["basis"].startswith("project evaluation thresholds")
    assert manifest["thresholds"]["excellent_max_cer_percentage"] == 5.0
    assert manifest["macro_average_cer"] == 0.0
    assert manifest["micro_average_cer"] == 0.0
    assert manifest["total_segments"] == 2
    assert manifest["successful_evaluations"] == 2
    assert manifest["languages"]["hi"]["segments"] == 2


def test_manifest_counts_success_failure_and_unavailable_separately() -> None:
    evaluation = evaluate_sections(
        [
            reference("seg-01", "abc"),
            reference("seg-02", "abc"),
            reference("seg-03", "abc"),
        ],
        [
            transcription("seg-01", "abc"),
            transcription("seg-02", "abc", language="ta"),
            transcription("seg-04", "abc"),
        ],
        expected_language="hi",
    )

    manifest = evaluation.manifest
    assert manifest["total_segments"] == 4
    assert manifest["successful_evaluations"] == 1
    assert manifest["failed_evaluations"] == 2  # language mismatch + missing transcription
    assert manifest["segments_with_unavailable_reference"] == 1


# --- aggregation -----------------------------------------------------------


def test_macro_and_micro_averages_are_distinct_and_correct() -> None:
    long_reference = "a" * 100
    evaluation = evaluate_sections(
        [reference("seg-01", "abcdefghij"), reference("seg-02", long_reference)],
        [
            transcription("seg-01", "abcdefghXj"),
            transcription("seg-02", long_reference[:-1] + "b"),
        ],
        expected_language="hi",
    )

    assert all(record["edit_distance"] == 1 for record in evaluation.records)
    assert evaluation.manifest["macro_average_cer"] == pytest.approx((0.1 + 0.01) / 2)
    assert evaluation.manifest["micro_average_cer"] == pytest.approx(2 / 110)
    assert evaluation.manifest["macro_average_cer"] != pytest.approx(
        evaluation.manifest["micro_average_cer"]
    )
    assert evaluation.manifest["macro_average_cer_percentage"] == pytest.approx(5.5)
    assert evaluation.manifest["micro_average_cer_percentage"] == pytest.approx(200 / 110)


def test_language_metrics_are_kept_separate() -> None:
    evaluation = evaluate_sections(
        [
            reference("seg-01", "abcdefghij", audio="hi.wav"),
            reference("seg-02", "abcd", audio="ta.wav"),
            reference("seg-03", "ab", audio="bn.wav"),
        ],
        [
            transcription("seg-01", "abcdefghXj", language="hi"),
            transcription("seg-02", "abcd", language="ta"),
            transcription("seg-03", "abX", language="bn"),
        ],
        expected_language=None,  # mixed-language batch; each language validated separately
    )

    languages = evaluation.manifest["languages"]
    assert set(languages) == {"hi", "ta", "bn"}
    assert languages["hi"]["macro_average_cer"] == pytest.approx(0.1)
    assert languages["ta"]["macro_average_cer"] == 0.0
    assert languages["bn"]["macro_average_cer"] == pytest.approx(1 / 2)
    for language in languages.values():
        assert language["segments"] == 1
        assert language["total_segments"] == 1


# --- thresholds ------------------------------------------------------------


@pytest.mark.parametrize(
    ("percentage", "expected"),
    [
        (0.0, "Excellent"),
        (5.0, "Excellent"),
        (5.01, "Good"),
        (10.0, "Good"),
        (10.01, "Needs Review"),
        (20.0, "Needs Review"),
        (20.01, "Poor"),
        (150.0, "Poor"),
    ],
)
def test_default_quality_thresholds(percentage: float, expected: str) -> None:
    assert quality_label(percentage) == expected


def test_thresholds_are_configurable_and_recorded() -> None:
    limits = Thresholds(excellent_max=1.0, good_max=2.0, review_max=3.0)
    evaluation = evaluate_sections(
        [reference("seg-01", "abcdefghijklmnopqrst")],
        [transcription("seg-01", "abcdefghijklmnopqrstX")],  # 1/20 = 5%
        expected_language="hi",
        thresholds=limits,
    )

    assert evaluation.records[0]["quality"] == "Poor"
    assert evaluation.manifest["thresholds"]["excellent_max_cer_percentage"] == 1.0
    assert evaluation.manifest["thresholds"]["poor"] == "CER > 3%"


def test_invalid_threshold_ordering_is_rejected() -> None:
    with pytest.raises(ValueError, match="0 <= excellent"):
        Thresholds(excellent_max=10.0, good_max=5.0)
    with pytest.raises(ValidationError):
        Settings(cer_excellent_max_pct=10.0, cer_good_max_pct=5.0)


def test_evaluation_dataclass_carries_records_and_manifest() -> None:
    evaluation = Evaluation(records=[], manifest={"status": "done"})
    assert evaluation.records == []
    assert evaluation.manifest["status"] == "done"
