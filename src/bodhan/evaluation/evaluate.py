"""Per-section CER evaluation, aggregation and artifact writing (Phase 2B).

Reference text is ``AudioSegment.text`` from ``narrate/segments.json`` - the
exact string passed to Indic-Speak by the narrate stage - and the hypothesis
is ``transcription_text`` from ``transcribe/segment-NN.json``. Pairs are
matched by ``segment_id`` <-> ``section_id`` (never by list position or file
name). Evaluation results are written to a separate ``evaluation/cer/`` tree
so the original transcription artifacts remain untouched.

Nothing here calls an API, an LLM, a translator or a speech engine.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bodhan.evaluation.cer import NORMALIZATION_POLICY, CERResult, compute_cer

#: Provenance recorded in every evaluated record. The narrate stage stores
#: ``for_narration(...)`` verbatim in AudioSegment.text and passes that same
#: string to ``tts.synthesize`` - it is the exact Indic-Speak input.
REFERENCE_SOURCE = "narrate/segments.json:AudioSegment.text (exact Indic-Speak input)"

STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_UNAVAILABLE = "unavailable"  # no usable reference (never invented)


@dataclass(frozen=True)
class Thresholds:
    """Configurable CER quality bands, expressed as percentages.

    These are project evaluation thresholds, NOT official Bodhan standards.
    """

    excellent_max: float = 5.0
    good_max: float = 10.0
    review_max: float = 20.0

    def __post_init__(self) -> None:
        if not 0 <= self.excellent_max <= self.good_max <= self.review_max:
            raise ValueError(
                "CER thresholds must satisfy 0 <= excellent <= good <= review "
                f"(got {self.excellent_max}, {self.good_max}, {self.review_max})"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "excellent_max_cer_percentage": self.excellent_max,
            "good_max_cer_percentage": self.good_max,
            "review_max_cer_percentage": self.review_max,
            "excellent": f"CER <= {self.excellent_max:g}%",
            "good": f"{self.excellent_max:g}% < CER <= {self.good_max:g}%",
            "needs_review": f"{self.good_max:g}% < CER <= {self.review_max:g}%",
            "poor": f"CER > {self.review_max:g}%",
            "basis": "project evaluation thresholds; not official Bodhan quality standards",
        }


def quality_label(cer_percentage: float, thresholds: Thresholds | None = None) -> str:
    """Map a CER percentage to a configured quality band."""
    limits = thresholds or Thresholds()
    if cer_percentage <= limits.excellent_max:
        return "Excellent"
    if cer_percentage <= limits.good_max:
        return "Good"
    if cer_percentage <= limits.review_max:
        return "Needs Review"
    return "Poor"


def _base_record(section_id: str | None) -> dict[str, Any]:
    """Build a record whose unknown fields are explicitly null, never invented."""
    return {
        "section_id": section_id,
        "target_language": None,
        "audio_file": None,
        "reference_source": None,
        "reference_character_count": None,
        "transcription_character_count": None,
        "substitutions": None,
        "deletions": None,
        "insertions": None,
        "edit_distance": None,
        "cer": None,
        "cer_percentage": None,
        "cer_unavailable_reason": None,
        "quality": None,
        "normalization_policy": NORMALIZATION_POLICY["id"],
        "status": STATUS_FAILED,
    }


def _reference_of(item: dict[str, Any] | None) -> tuple[str | None, str | None]:
    """Return (reference text, error) for a narration segment."""
    if item is None:
        return None, "reference unavailable: no narration segment with this section_id"
    text = item.get("text")
    if not isinstance(text, str):
        return None, "reference unavailable: narration segment has no text"
    return text, None


def _unpaired_transcription(item: dict[str, Any]) -> dict[str, Any]:
    """A transcription without a section_id cannot be paired - record the failure."""
    record = _base_record(None)
    for field in ("target_language", "audio_file"):
        value = item.get(field)
        if isinstance(value, str):
            record[field] = value
    record["error"] = "transcription record has no section_id; cannot pair by section_id"
    return record


def _evaluate_one(
    section_id: str | None,
    reference: dict[str, Any] | None,
    transcription: dict[str, Any] | None,
    *,
    expected_language: str | None,
    thresholds: Thresholds,
) -> dict[str, Any]:
    record = _base_record(section_id)
    if transcription is not None:
        language = transcription.get("target_language")
        if isinstance(language, str):
            record["target_language"] = language
        audio_file = transcription.get("audio_file")
        if isinstance(audio_file, str):
            record["audio_file"] = audio_file
    if reference is not None and record["audio_file"] is None:
        audio_path = reference.get("audio_path")
        if isinstance(audio_path, str):
            record["audio_file"] = Path(audio_path).name

    reference_text, reference_error = _reference_of(reference)
    if reference_error is not None:
        record["status"] = STATUS_UNAVAILABLE
        record["cer_unavailable_reason"] = reference_error
        record["error"] = reference_error
        return record

    record["reference_source"] = REFERENCE_SOURCE

    if transcription is None:
        message = f"transcription missing for section_id {section_id!r}"
        record["error"] = message
        return record

    transcription_text = transcription.get("transcription_text")
    if not isinstance(transcription_text, str):
        message = f"transcription text missing for section_id {section_id!r}"
        record["error"] = message
        return record

    # Language validation: every transcription must match the workspace
    # language from the transcribe manifest, and - when a reference record
    # carries a language field - that language too.
    transcription_language = transcription.get("target_language")
    # `_reference_of` guaranteed a usable reference record by this point.
    assert reference is not None
    reference_language = reference.get("target_language") or reference.get("language_code")
    for origin, language in (
        ("transcribe manifest", expected_language),
        ("reference segment", reference_language if isinstance(reference_language, str) else None),
    ):
        if language is not None and transcription_language != language:
            message = (
                f"language mismatch for section_id {section_id!r}: "
                f"transcription={transcription_language!r}, {origin}={language!r}"
            )
            record["error"] = message
            return record

    result: CERResult = compute_cer(reference_text or "", transcription_text)
    for field in (
        "reference_character_count",
        "transcription_character_count",
        "substitutions",
        "deletions",
        "insertions",
        "edit_distance",
        "cer",
        "cer_percentage",
        "cer_unavailable_reason",
    ):
        record[field] = getattr(result, field)

    if result.cer_percentage is None:
        # Empty reference: CER is undefined. Counts above are still factual.
        record["status"] = STATUS_UNAVAILABLE
        record["error"] = result.cer_unavailable_reason
    else:
        record["status"] = STATUS_OK
        record["quality"] = quality_label(result.cer_percentage, thresholds)
    return record


def _summarize(records: list[dict[str, Any]], thresholds: Thresholds) -> dict[str, Any]:
    """Compute macro/micro averages and per-language metrics."""
    total = len(records)
    successful = [record for record in records if record["status"] == STATUS_OK]
    failed = sum(1 for record in records if record["status"] == STATUS_FAILED)
    unavailable = sum(1 for record in records if record["status"] == STATUS_UNAVAILABLE)

    def averages(group: list[dict[str, Any]]) -> dict[str, Any]:
        sum_distance = sum(int(record["edit_distance"]) for record in group)
        sum_reference = sum(int(record["reference_character_count"]) for record in group)
        macro = sum(float(record["cer"]) for record in group) / len(group) if group else None
        micro = sum_distance / sum_reference if sum_reference else None
        return {
            "segments": len(group),
            "sum_edit_distance": sum_distance,
            "sum_reference_character_count": sum_reference,
            "macro_average_cer": macro,
            "macro_average_cer_percentage": macro * 100 if macro is not None else None,
            "micro_average_cer": micro,
            "micro_average_cer_percentage": micro * 100 if micro is not None else None,
        }

    languages: dict[str, Any] = {}
    language_values = sorted(
        {
            str(record["target_language"])
            for record in records
            if record["target_language"] is not None
        }
    )
    for language in language_values:
        group = [record for record in records if record["target_language"] == language]
        language_summary = averages([record for record in group if record["status"] == STATUS_OK])
        language_summary["total_segments"] = len(group)
        language_summary["failed_evaluations"] = sum(
            1 for record in group if record["status"] == STATUS_FAILED
        )
        language_summary["segments_with_unavailable_reference"] = sum(
            1 for record in group if record["status"] == STATUS_UNAVAILABLE
        )
        languages[language] = language_summary

    distribution: dict[str, int] = {}
    for record in successful:
        label = str(record["quality"])
        distribution[label] = distribution.get(label, 0) + 1

    return {
        "total_segments": total,
        "successful_evaluations": len(successful),
        "failed_evaluations": failed,
        "segments_with_unavailable_reference": unavailable,
        "overall": averages(successful),
        "quality_distribution": distribution,
        "languages": languages,
    }


@dataclass
class Evaluation:
    """Per-section records plus the aggregated manifest payload."""

    records: list[dict[str, Any]]
    manifest: dict[str, Any]


def evaluate_sections(
    narrate_segments: list[dict[str, Any]],
    transcribe_records: list[dict[str, Any]],
    *,
    expected_language: str | None = None,
    thresholds: Thresholds | None = None,
) -> Evaluation:
    """Pair by section id and evaluate every narration/transcription section.

    ``expected_language`` should come from ``transcribe/manifest.json``; pass
    ``None`` when evaluating a mixed-language batch (per-language metrics stay
    separate either way).
    """
    limits = thresholds or Thresholds()

    references: dict[str, dict[str, Any]] = {}
    for item in narrate_segments:
        if not isinstance(item, dict):
            continue
        section_id = item.get("segment_id") or item.get("section_id")
        if isinstance(section_id, str) and section_id:
            references[section_id] = item

    transcriptions: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for item in transcribe_records:
        if not isinstance(item, dict):
            continue
        section_id = item.get("section_id")
        if not isinstance(section_id, str) or not section_id:
            # Never guess an id: an unpaired record is an explicit failure.
            records.append(_unpaired_transcription(item))
            continue
        transcriptions[section_id] = item

    for section_id in sorted(set(references) | set(transcriptions)):
        records.append(
            _evaluate_one(
                section_id,
                references.get(section_id),
                transcriptions.get(section_id),
                expected_language=expected_language,
                thresholds=limits,
            )
        )
    # Unpaired (id-less) records first, then deterministic section_id order.
    records.sort(key=lambda record: (record["section_id"] is not None, record["section_id"] or ""))

    summary = _summarize(records, limits)
    manifest: dict[str, Any] = {
        "status": "done",
        "evaluation": "character_error_rate",
        "formula": "(substitutions + deletions + insertions) / reference_character_count",
        "reference_source": REFERENCE_SOURCE,
        "normalization_policy": dict(NORMALIZATION_POLICY),
        "macro_average_definition": (
            "arithmetic mean of per-segment CER over successful evaluations"
        ),
        "micro_average_definition": (
            "sum(edit_distance) / sum(reference_character_count) over successful evaluations"
        ),
        "thresholds": limits.as_dict(),
        "expected_language": expected_language,
        "total_segments": summary["total_segments"],
        "successful_evaluations": summary["successful_evaluations"],
        "failed_evaluations": summary["failed_evaluations"],
        "segments_with_unavailable_reference": summary["segments_with_unavailable_reference"],
        "macro_average_cer": summary["overall"]["macro_average_cer"],
        "macro_average_cer_percentage": summary["overall"]["macro_average_cer_percentage"],
        "micro_average_cer": summary["overall"]["micro_average_cer"],
        "micro_average_cer_percentage": summary["overall"]["micro_average_cer_percentage"],
        "sum_edit_distance": summary["overall"]["sum_edit_distance"],
        "sum_reference_character_count": summary["overall"]["sum_reference_character_count"],
        "quality_distribution": summary["quality_distribution"],
        "languages": summary["languages"],
        "segments": [
            {
                "section_id": record["section_id"],
                "status": record["status"],
                "target_language": record["target_language"],
                "cer": record["cer"],
                "cer_percentage": record["cer_percentage"],
                "quality": record["quality"],
                "error": record.get("error"),
            }
            for record in records
        ],
    }
    return Evaluation(records=records, manifest=manifest)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write UTF-8 JSON atomically so a crash cannot leave a partial record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_evaluation(directory: Path, evaluation: Evaluation) -> list[Path]:
    """Store ``segment-NN.json`` records + ``manifest.json`` in ``directory``."""
    paths: list[Path] = []
    for index, record in enumerate(evaluation.records, start=1):
        path = directory / f"segment-{index:02d}.json"
        _atomic_write_json(path, record)
        paths.append(path)
    manifest_path = directory / "manifest.json"
    _atomic_write_json(manifest_path, evaluation.manifest)
    paths.append(manifest_path)
    return paths
