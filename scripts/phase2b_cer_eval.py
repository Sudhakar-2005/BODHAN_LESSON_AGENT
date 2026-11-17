"""Phase 2B live (offline) evaluation: CER of the six Phase 2A Hindi segments.

Reads ONLY the persisted Phase 2A verification artifacts:

- ``narrate/segments.json``  -> ``text`` is the exact string sent to
  Indic-Speak (the CER reference; no English source, translation or Markdown
  is used);
- ``transcribe/segment-NN.json`` + ``manifest.json`` -> verbatim
  Indic-Transcribe output (the CER hypothesis), paired by ``section_id``.

Writes ``evaluation/cer/segment-NN.json`` + ``manifest.json`` next to those
inputs. NO API calls, NO audio regeneration, NO LLM and NO rewrite of any
transcription. Credentials from settings are never printed.

Usage::

    uv run python scripts/phase2b_cer_eval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from bodhan.config import load_settings
from bodhan.evaluation import Thresholds, evaluate_sections, write_evaluation


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise SystemExit(f"missing required artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    settings = load_settings()
    workspace = settings.work_dir / "phase2a-verify"
    narrate_segments_path = workspace / "narrate" / "segments.json"
    transcribe_dir = workspace / "transcribe"
    transcribe_manifest_path = transcribe_dir / "manifest.json"

    narrate_segments = _load_json(narrate_segments_path)
    transcribe_manifest = _load_json(transcribe_manifest_path)
    if not isinstance(narrate_segments, list):
        raise SystemExit(f"expected a JSON list in {narrate_segments_path}")
    if not isinstance(transcribe_manifest, dict):
        raise SystemExit(f"expected a JSON object in {transcribe_manifest_path}")

    transcription_records = []
    for path in sorted(transcribe_dir.glob("segment-*.json")):
        payload = _load_json(path)
        if isinstance(payload, dict):
            transcription_records.append(payload)

    expected_language = transcribe_manifest.get("target_language")
    if not isinstance(expected_language, str):
        expected_language = None

    evaluation = evaluate_sections(
        narrate_segments,
        transcription_records,
        expected_language=expected_language,
        thresholds=Thresholds(
            excellent_max=settings.cer_excellent_max_pct,
            good_max=settings.cer_good_max_pct,
            review_max=settings.cer_review_max_pct,
        ),
    )
    output_dir = workspace / "evaluation" / "cer"
    written = write_evaluation(output_dir, evaluation)

    summary = evaluation.manifest
    print(
        f"evaluated {summary['total_segments']} section(s): "
        f"{summary['successful_evaluations']} successful, "
        f"{summary['failed_evaluations']} failed, "
        f"{summary['segments_with_unavailable_reference']} unavailable-reference "
        f"(language={expected_language})"
    )
    print()
    print(
        f"{'section':<10} {'status':<12} {'CER%':>9}  {'quality':<13} reference <- exact Indic-Speak input"
    )
    for record in evaluation.records:
        cer_percentage = record["cer_percentage"]
        percentage = f"{cer_percentage:.2f}" if cer_percentage is not None else "n/a"
        error_text = record.get("error") or ""
        print(
            f"{record['section_id']!s:<10} {record['status']:<12} {percentage:>9}  "
            f"{record['quality']!s:<13} {error_text!s:<40}"
        )

    print()
    print(f"macro-average CER: {summary['macro_average_cer_percentage']}%")
    print(f"micro-average CER: {summary['micro_average_cer_percentage']}%")
    print(f"quality distribution: {summary['quality_distribution']}")
    for language, metrics in summary["languages"].items():
        print(
            f"language {language}: {metrics['segments']} evaluated, "
            f"macro={metrics['macro_average_cer_percentage']}%, "
            f"micro={metrics['micro_average_cer_percentage']}%"
        )
    print(f"\nartifacts written: {len(written)} files under {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
