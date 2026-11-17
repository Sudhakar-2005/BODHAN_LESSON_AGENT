"""Build a traceable, machine-readable ``report.json`` from persisted artifacts.

Every field comes from an existing artifact or is explicitly null with a
reason recorded in ``unavailable_fields``. Nothing is recalculated when an
artifact already carries the value (notably Phase 2B CER results). No API,
LLM, TTS or STT call is made; no prior artifact is mutated.
"""

from __future__ import annotations

import copy
import json
import re
import wave
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bodhan import __version__
from bodhan.services.json_io import write_json_atomic

REPORT_SCHEMA_VERSION = "1.0"
#: Documented in src/bodhan/adapters/transcribe/indic_transcribe.py (official
#: docs inspected 2026-11-17). Used ONLY for a clearly labelled estimate.
STT_RATE_INR_PER_AUDIO_MINUTE = 0.10
_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)


def _canonical(report: dict[str, Any]) -> str:
    """Stable representation used to detect no-op reruns.

    ``generated_at`` is true wall-clock provenance and therefore intentionally
    excluded: a rerun with otherwise identical evidence must not rewrite the
    artifact or change its bytes.
    """
    payload = copy.deepcopy(report)
    metadata = payload.get("run_metadata")
    if isinstance(metadata, dict):
        metadata.pop("generated_at", None)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def write_report(path: Path, report: dict[str, Any]) -> None:
    """Atomically write the report; skip a byte-identical no-op rerun."""
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(existing, dict) and _canonical(existing) == _canonical(report):
                return
        except (OSError, ValueError):
            pass  # unreadable prior report: regenerate atomically
    write_json_atomic(path, report)


#: A null ``error`` means "no error occurred", not "unavailable telemetry".
_NO_REASON_NEEDED = frozenset({"error", "cer_unavailable_reason"})


def _null_value_reasons(report: dict[str, Any], gaps: dict[str, str]) -> dict[str, str]:
    """Map every unavailable null to an exact-path reason (no fabrication)."""
    reasons: dict[str, str] = {}

    def walk(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                walk(item, f"{path}.{key}" if path else str(key))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")
        elif value is None:
            leaf = path.rsplit(".", 1)[-1].split("[", 1)[0]
            if leaf in _NO_REASON_NEEDED:
                return
            generic = re.sub(r"\[\d+\]", "[]", path)
            reason = gaps.get(generic) or gaps.get(path)
            if reason is None:
                reason = {
                    "model": "provider model is not recorded in the source artifacts",
                    "rate_limit": "provider rate-limit metadata was not persisted",
                }.get(str(leaf), "value is unavailable in the source artifacts; not fabricated")
            reasons[path] = reason

    walk(report, "")
    return reasons


class ReportBuilder:
    """Aggregate Phase 1/2A/2B/2C evidence for one target language."""

    def __init__(
        self,
        *,
        work_dir: Path,
        export_dir: Path | None,
        transcribe_dir: Path | None,
        evaluation_dir: Path | None,
        assembly_dir: Path,
        commit: str | None = None,
        working_tree_clean: bool | None = None,
    ) -> None:
        self.work_dir = Path(work_dir)
        self.export_dir = Path(export_dir) if export_dir else None
        self.transcribe_dir = Path(transcribe_dir) if transcribe_dir else None
        self.evaluation_dir = Path(evaluation_dir) if evaluation_dir else None
        self.assembly_dir = Path(assembly_dir)
        self.commit = commit
        self.working_tree_clean = working_tree_clean
        self.gaps: dict[str, str] = {}
        self.sources: dict[str, str] = {}
        self._cache: dict[str, Any] = {}

    # -- artifact loading / gap recording -----------------------------------
    def _unavailable(self, field: str, reason: str) -> None:
        self.gaps[field] = reason

    def _load(self, label: str, path: Path | None, *, required: bool = False) -> Any | None:
        if path is None:
            self.sources[label] = "not configured"
            if required:
                self._unavailable(label, "the artifact directory was not provided")
            return None
        if not path.is_file():
            self.sources[label] = "missing"
            self._unavailable(label, f"artifact not found: {path}")
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            self.sources[label] = "unreadable"
            self._unavailable(label, f"could not read {path}: {exc}")
            return None
        self.sources[label] = "loaded"
        self._cache[label] = payload
        return payload

    @staticmethod
    def _list(value: Any) -> list[Any]:
        return value if isinstance(value, list) else []

    @staticmethod
    def _dict(value: Any) -> dict[str, Any]:
        return value if isinstance(value, dict) else {}

    def _records(self, label: str, directory: Path | None) -> list[dict[str, Any]]:
        if directory is None or not directory.is_dir():
            self._unavailable(label, f"directory not found: {directory}")
            return []
        result: list[dict[str, Any]] = []
        for path in sorted(directory.glob("segment-*.json")):
            payload = self._load(f"{label}/{path.name}", path)
            if isinstance(payload, dict):
                result.append(payload)
        return result

    # -- A. run metadata -----------------------------------------------------
    def _run_metadata(self, *, pipeline_status: str) -> dict[str, Any]:
        manifest = self._dict(self._cache.get("work/manifest.json"))
        pages = self._list(self._cache.get("work/ingest/pages.json"))
        lesson = self._dict(self._cache.get("work/compose/lesson.json"))
        first_page = pages[0] if pages and isinstance(pages[0], dict) else {}
        source_path = first_page.get("source_path")
        source_filename = Path(source_path).name if isinstance(source_path, str) else None
        if source_filename is None:
            self._unavailable(
                "run_metadata.source_filename", "ingest/pages.json has no source_path"
            )

        from datetime import timedelta

        stages = self._dict(manifest.get("stages"))
        times: list[tuple[str, float]] = []
        for stage in stages.values():
            if isinstance(stage, dict) and isinstance(stage.get("completed_at"), str):
                times.append((str(stage["completed_at"]), float(stage.get("duration_s") or 0)))
        start_time: str | None
        completion_time: str | None
        wall_clock_s: float | None
        if times:
            parsed = [(datetime.fromisoformat(ts), duration) for ts, duration in times]
            start = min(ts - timedelta(seconds=duration) for ts, duration in parsed)
            end = max(ts for ts, _ in parsed)
            start_time = start.isoformat()
            completion_time = end.isoformat()
            wall_clock_s = round((end - start).total_seconds(), 3)
        else:
            start_time = completion_time = wall_clock_s = None
            self._unavailable(
                "run_metadata.start_time/completion_time/total_pipeline_duration",
                "manifest.json has no stage timestamps or durations",
            )
        total_duration = round(sum(duration for _, duration in times), 3) if times else None

        target_language = lesson.get("target_language_code")
        source_language = lesson.get("source_language_code")
        if not isinstance(target_language, str):
            target_language = None
            self._unavailable(
                "run_metadata.target_language", "compose/lesson.json has no target_language_code"
            )

        if self.commit is None:
            self._unavailable(
                "run_metadata.application.commit",
                "no commit identifier supplied to ReportBuilder",
            )
        if self.working_tree_clean is None:
            self._unavailable(
                "run_metadata.application.working_tree_clean",
                "working-tree cleanliness was not supplied to ReportBuilder",
            )

        if self.commit is None:
            commit_basis = "no commit identifier available"
        elif self.working_tree_clean is False:
            commit_basis = (
                "HEAD commit hash only; the working tree has uncommitted changes, "
                "so this commit does NOT describe the evaluated implementation"
            )
        elif self.working_tree_clean is True:
            commit_basis = "HEAD commit hash; working tree is clean"
        else:
            commit_basis = "HEAD commit hash; working-tree state unknown"

        return {
            "run_id": f"{(manifest.get('chapter_id') or 'chapter')}-{target_language or 'unknown'}",
            "source_filename": source_filename,
            "source_language": source_language if isinstance(source_language, str) else None,
            "target_language": target_language,
            "page_count": len(pages) if pages else None,
            "start_time": start_time,
            "completion_time": completion_time,
            "total_pipeline_duration_s": total_duration,
            "wall_clock_duration_s": wall_clock_s,
            "duration_basis": (
                "start_time = earliest stage completed_at minus its duration_s; "
                "completion_time = latest stage completed_at; total_pipeline_duration_s = "
                "sum of recorded stage durations; wall_clock_duration_s includes idle gaps"
            ),
            "pipeline_status": pipeline_status,
            "stage_statuses": {
                name: stage.get("status")
                for name, stage in stages.items()
                if isinstance(stage, dict)
            },
            "application": {
                "name": "BODHAN LESSON AGENT",
                "version": __version__,
                "commit": self.commit,
                "working_tree_clean": self.working_tree_clean,
                "commit_basis": commit_basis,
            },
            # True wall-clock generation time. write_report() skips a no-op
            # rerun by comparing the report without this field, so idempotence
            # is preserved without mislabelling a manifest timestamp.
            "generated_at": datetime.now(UTC).isoformat(),
            "generated_at_basis": "UTC wall-clock time when report.json was generated",
            "evidence_as_of": completion_time,
            "evidence_as_of_basis": "latest stage completed_at from manifest.json",
        }

    # -- B. OCR --------------------------------------------------------------
    def _ocr(self, extract_manifest: dict[str, Any]) -> list[dict[str, Any]]:
        results = self._list(self._cache.get("work/extract/results.json"))
        if not results:
            self._unavailable("per_page_ocr", "extract/results.json is missing or empty")
        output: list[dict[str, Any]] = []
        for entry in results:
            if not isinstance(entry, dict):
                continue
            blocks = [b for b in self._list(entry.get("blocks")) if isinstance(b, dict)]
            confidences = [
                float(b["confidence"])
                for b in blocks
                if isinstance(b.get("confidence"), (int, float))
                and not isinstance(b.get("confidence"), bool)
            ]
            low = sum(1 for b in blocks if "low_confidence" in self._list(b.get("flags")))
            status = (
                "done" if blocks or isinstance(entry.get("page_markdown"), str) else "unavailable"
            )
            page_number = entry.get("page_index")
            output.append(
                {
                    "page_number": page_number + 1 if isinstance(page_number, int) else None,
                    "ocr_status": status,
                    "extracted_block_count": len(blocks),
                    "confidence": (
                        {
                            "min": min(confidences),
                            "max": max(confidences),
                            "mean": round(sum(confidences) / len(confidences), 6),
                            "per_block": {
                                str(b.get("block_id")): b["confidence"]
                                for b in blocks
                                if isinstance(b.get("confidence"), (int, float))
                            },
                            "source": "block.confidence from Indic-OCR response",
                        }
                        if confidences
                        else None
                    ),
                    "low_confidence_block_count": low,
                    "retry_count": None,
                    "latency_ms": None,
                    "provider": entry.get("provider"),
                    "provider_request_id": entry.get("provider_request_id"),
                    "provider_usage": self._dict(entry.get("usage")) or None,
                    "rate_limit": self._dict(entry.get("rate_limit")) or None,
                    "errors": self._list(extract_manifest.get("error")) or [],
                }
            )
        for field, reason in (
            ("per_page_ocr[].retry_count", "retry attempts are not persisted per OCR page"),
            (
                "per_page_ocr[].latency_ms",
                "only the stage-level duration_s is persisted in manifest.json",
            ),
            (
                "per_page_ocr[].rate_limit",
                "rate-limit headers were not persisted for individual OCR pages",
            ),
        ):
            self._unavailable(field, reason)
        return output

    # -- C. structure / quality ----------------------------------------------
    def _structure_quality(self) -> dict[str, Any]:
        blocks = [
            b
            for b in self._list(self._cache.get("work/quality/blocks.json"))
            if isinstance(b, dict)
        ]
        quality_report = self._dict(self._cache.get("work/quality/quality_report.json"))
        lesson = self._dict(self._cache.get("work/compose/lesson.json"))
        plan = self._dict(self._cache.get("work/compose/plan.json"))
        headings = _HEADING.findall(str(lesson.get("markdown") or ""))
        flags = Counter(
            str(flag)
            for entry in self._list(quality_report.get("entries"))
            if isinstance(entry, dict)
            for flag in self._list(entry.get("flags"))
        )
        by_page: dict[int, list[int]] = {}
        for block in blocks:
            page = block.get("page_index")
            order = block.get("reading_order")
            if isinstance(page, int) and isinstance(order, int):
                by_page.setdefault(page, []).append(order)
        order_issues = [
            {"page_index": page, "reading_orders": orders}
            for page, orders in sorted(by_page.items())
            if len(orders) != len(set(orders)) or sorted(orders) != list(range(len(orders)))
        ]
        rejected = [
            b.get("block_id")
            for b in blocks
            if any(flag in {"untranslatable", "rejected"} for flag in self._list(b.get("flags")))
        ]
        empty = [b.get("block_id") for b in blocks if not str(b.get("text") or "")]
        self._unavailable(
            "structure_and_quality.cross_page_continuity_issues",
            "no cross-page continuity check is implemented or persisted in the artifacts",
        )
        return {
            "section_count": len(headings) - 1 if headings else None,
            "section_count_basis": "Markdown headings in compose/lesson.json minus the document title",
            "narrated_section_count": len(self._narration_segments()),
            "plan_section_count": len(self._list(plan.get("sections"))),
            "block_count": len(blocks),
            "unresolved_or_rejected_blocks": rejected,
            "empty_text_blocks": empty,
            "reading_order_issues": order_issues,
            "quality_flags": dict(flags),
            "flagged_block_count": quality_report.get("flagged"),
            "cross_page_continuity_issues": None,
        }

    # -- D. translation ------------------------------------------------------
    def _translations(self) -> dict[str, Any]:
        translations = self._dict(self._cache.get("work/translate/translations.json"))
        lesson = self._dict(self._cache.get("work/compose/lesson.json"))
        plan = self._dict(self._cache.get("work/compose/plan.json"))
        blocks = {
            str(b.get("block_id")): b
            for b in self._list(self._cache.get("work/structure/blocks.json"))
            if isinstance(b, dict)
        }
        block_section: dict[str, str] = {}
        for index, section in enumerate(self._list(plan.get("sections")), start=1):
            if isinstance(section, dict):
                for block_id in self._list(section.get("block_ids")):
                    block_section[str(block_id)] = f"section-{index}"

        body: list[dict[str, Any]] = []
        for block_id, text in self._dict(translations.get("items")).items():
            source = blocks.get(str(block_id), {})
            source_text = str(source.get("text") or "")
            translated = str(text)
            if translated:
                body_errors: list[str] = []
            elif source_text:
                body_errors = ["empty translation output"]
            else:
                body_errors = ["empty translation (source block text is empty)"]
            body.append(
                {
                    "category": "body",
                    "section_id": block_section.get(str(block_id)),
                    "section_id_basis": "derived from compose/plan.json block_ids",
                    "block_id": block_id,
                    "translation_status": "ok" if translated else "empty",
                    "source_language": lesson.get("source_language_code"),
                    "target_language": lesson.get("target_language_code"),
                    "provider": None,
                    "model": None,
                    "source_character_count": len(source_text),
                    "translated_character_count": len(translated),
                    "retry_count": None,
                    "latency_ms": None,
                    "errors": body_errors,
                }
            )

        scaffold: list[dict[str, Any]] = []
        for item in self._list(lesson.get("scaffold")):
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "")
            source_en = str(item.get("source_en") or "")
            scaffold.append(
                {
                    "category": "scaffold",
                    "section_id": item.get("item_id"),
                    "item_kind": item.get("kind"),
                    "translation_status": "ok" if text else "empty",
                    "source_language": lesson.get("source_language_code"),
                    "target_language": lesson.get("target_language_code"),
                    "provider": item.get("provider"),
                    "model": None,
                    "source_character_count": len(source_en),
                    "translated_character_count": len(text),
                    "usage": self._dict(item.get("usage")) or None,
                    "related_block_ids": self._list(item.get("related_block_ids")),
                    "retry_count": None,
                    "latency_ms": None,
                    "errors": [],
                    "distinction": "scaffold label from lesson output, not a body-block translation",
                }
            )
        for field, reason in (
            (
                "translations.body[].provider",
                "translations.json records no per-block provider/model",
            ),
            ("translations.body[].model", "translations.json records no per-block model"),
            ("translations.scaffold[].model", "scaffold items record a provider but no model"),
            ("translations.body[].latency_ms", "no per-block latency is persisted"),
            ("translations.body[].retry_count", "no per-block retry count is persisted"),
            ("translations.scaffold[].retry_count", "no per-unit retry count is persisted"),
            ("translations.scaffold[].latency_ms", "no per-unit latency is persisted"),
        ):
            self._unavailable(field, reason)
        return {
            "body": body,
            "scaffold": scaffold,
            "body_usage": self._dict(translations.get("usage")) or None,
            "verbatim_blocks": translations.get("verbatim_blocks"),
            "empty_blocks": translations.get("empty_blocks"),
            "separation_note": "body translations are keyed by block_id; scaffold units are keyed by item_id",
        }

    # -- E/F/G. narration, transcription, CER --------------------------------
    def _narration_segments(self) -> list[dict[str, Any]]:
        cached = self._cache.get("narrate/segments.json")
        return [s for s in self._list(cached) if isinstance(s, dict)]

    def _narration(self, manifest: dict[str, Any]) -> list[dict[str, Any]]:
        stages = self._dict(manifest.get("stages"))
        stage_status = self._dict(stages.get("narrate")).get("status")
        voice = self._dict(self._dict(stages.get("narrate")).get("artifacts")).get("voice")

        output: list[dict[str, Any]] = []
        for segment in self._narration_segments():
            audio_path = segment.get("audio_path")
            path = Path(audio_path) if isinstance(audio_path, str) else None
            exists = bool(path and path.is_file())
            measured: float | None = None
            if exists and path is not None:
                try:
                    with wave.open(str(path), "rb") as handle:
                        measured = round(handle.getnframes() / handle.getframerate(), 6)
                except (wave.Error, OSError, EOFError):  # report, never silently accept
                    measured = None
            output.append(
                {
                    "section_id": segment.get("segment_id"),
                    "audio_filename": path.name if path else None,
                    "narration_status": "ok" if exists and stage_status == "done" else "failed",
                    "duration_ms": segment.get("duration_ms"),
                    "measured_duration_seconds": measured,
                    "provider": None,
                    "model": "indic-speak",
                    "voice": voice,
                    "retry_count": None,
                    "latency_ms": None,
                    "errors": [] if exists else [f"audio file missing: {audio_path}"],
                }
            )
        for field, reason in (
            (
                "narration[].retry_count",
                "no narration retry count is persisted; narrate/status.json is absent from these artifacts",
            ),
            ("narration[].latency_ms", "no per-segment narration latency is persisted"),
            ("narration[].provider", "narrate/status.json is absent from these artifacts"),
        ):
            self._unavailable(field, reason)
        return output

    def _transcription(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for record in records:
            text = record.get("transcription_text")
            usage = record.get("usage")
            raw = self._list(record.get("raw"))
            output.append(
                {
                    "section_id": record.get("section_id"),
                    "transcription_status": record.get("status"),
                    "transcript_character_count": len(str(text)) if isinstance(text, str) else None,
                    "provider": record.get("provider"),
                    "model": record.get("model"),
                    "target_language": record.get("target_language"),
                    "latency_ms": record.get("latency_ms"),
                    "request_id": record.get("request_id"),
                    "successful_request_count": len(raw),
                    "successful_request_count_basis": "number of raw provider bodies (retries are not persisted)",
                    "usage": self._dict(usage) or None,
                    "retry_count": None,
                    "errors": [record["error"]] if record.get("error") else [],
                }
            )
        for field, reason in (
            ("transcription[].retry_count", "transcription adapter does not persist retry counts"),
            (
                "transcription[].usage",
                "provider returned usage: null; empty usage was not converted to zero",
            ),
        ):
            self._unavailable(field, reason)
        return output

    def _cer(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        manifest_path = (
            self.evaluation_dir / "manifest.json" if self.evaluation_dir is not None else None
        )
        manifest = self._dict(self._load("evaluation/cer/manifest.json", manifest_path))
        per_section: list[dict[str, Any]] = []
        for record in records:
            per_section.append(
                {
                    "section_id": record.get("section_id"),
                    "status": record.get("status"),
                    "reference_character_count": record.get("reference_character_count"),
                    "transcription_character_count": record.get("transcription_character_count"),
                    "substitutions": record.get("substitutions"),
                    "deletions": record.get("deletions"),
                    "insertions": record.get("insertions"),
                    "edit_distance": record.get("edit_distance"),
                    "cer": record.get("cer"),
                    "cer_percentage": record.get("cer_percentage"),
                    "quality_band": record.get("quality"),
                    "normalization_policy": record.get("normalization_policy"),
                    "reference_source": record.get("reference_source"),
                    "error": record.get("error"),
                }
            )
        return {
            "source": "Phase 2B evaluation/cer (values copied verbatim; not recalculated)",
            "status": manifest.get("status"),
            "macro_average_cer": manifest.get("macro_average_cer"),
            "macro_average_cer_percentage": manifest.get("macro_average_cer_percentage"),
            "micro_average_cer": manifest.get("micro_average_cer"),
            "micro_average_cer_percentage": manifest.get("micro_average_cer_percentage"),
            "normalization_policy": manifest.get("normalization_policy"),
            "thresholds_used": manifest.get("thresholds"),
            "formula": manifest.get("formula"),
            "totals": {
                "evaluated": manifest.get("total_segments"),
                "successful": manifest.get("successful_evaluations"),
                "failed": manifest.get("failed_evaluations"),
                "unavailable_reference": manifest.get("segments_with_unavailable_reference"),
            },
            "languages": manifest.get("languages"),
            "per_section": per_section,
        }

    # -- H. audio assembly ---------------------------------------------------
    def _assembly(self) -> dict[str, Any]:
        manifest = self._dict(
            self._load("assembly/manifest.json", self.assembly_dir / "assembly_manifest.json")
        )
        timeline = self._dict(
            self._load("assembly/timeline.json", self.assembly_dir / "timeline.json")
        )
        return {
            "assembly_status": manifest.get("status"),
            "final_audio_filename": self._dict(manifest.get("output")).get("filename"),
            "total_duration_seconds": self._dict(manifest.get("output")).get("duration_seconds"),
            "section_count": manifest.get("section_count"),
            "section_timeline": self._list(timeline.get("sections")),
            "missing_sections": self._list(manifest.get("missing_sections")),
            "failed_sections": self._list(manifest.get("failed_sections")),
            "output_format": self._dict(manifest.get("output")),
            "embedded_chapters": manifest.get("embedded_chapters"),
            "embedded_chapters_note": manifest.get("embedded_chapters_reason"),
            "ffmpeg_available": manifest.get("ffmpeg_available"),
            "transcoding": manifest.get("transcoding"),
            "marker_source": timeline.get("marker_source"),
        }

    # -- I/J. cost, usage, retries -------------------------------------------
    def _cost_usage(
        self, transcription: list[dict[str, Any]], assembly: dict[str, Any]
    ) -> dict[str, Any]:
        extract = self._list(self._cache.get("work/extract/results.json"))
        translations = self._dict(self._cache.get("work/translate/translations.json"))
        lesson = self._dict(self._cache.get("work/compose/lesson.json"))
        ocr_entries = [
            {"page": e.get("page_index"), "usage": self._dict(e.get("usage"))}
            for e in self._list(extract)
            if isinstance(e, dict)
        ]
        ocr_usage: dict[str, int] = {}
        for entry in ocr_entries:
            for key, value in self._dict(entry.get("usage")).items():
                if isinstance(value, int) and not isinstance(value, bool):
                    ocr_usage[key] = ocr_usage.get(key, 0) + value
        scaffold_usage: dict[str, int] = {}
        for item in self._list(lesson.get("scaffold")):
            if isinstance(item, dict):
                for key, value in self._dict(item.get("usage")).items():
                    if isinstance(value, int) and not isinstance(value, bool):
                        scaffold_usage[key] = scaffold_usage.get(key, 0) + value

        durations = [
            float(section.get("duration_seconds") or 0)
            for section in self._list(
                self._dict(
                    self._load("assembly/timeline.json", self.assembly_dir / "timeline.json")
                ).get("sections")
            )
            if isinstance(section, dict)
        ]
        audio_minutes = sum(durations) / 60 if durations else None
        estimate = (
            round(audio_minutes * STT_RATE_INR_PER_AUDIO_MINUTE, 4)
            if audio_minutes is not None
            else None
        )
        for field, reason in (
            (
                "cost_and_usage.provider_reported_usage.narration",
                "Indic-Speak usage/cost is not persisted in narrate artifacts",
            ),
            (
                "cost_and_usage.provider_reported_usage.transcription",
                "provider body carried usage: null; null was not converted to zero",
            ),
            (
                "cost_and_usage.provider_reported_cost",
                "no provider artifact reports a monetary cost",
            ),
            (
                "cost_and_usage.estimated_cost.narration",
                "no Indic-Speak rate is documented in the repository artifacts",
            ),
            (
                "cost_and_usage.request_ids.transcription",
                "gateway sent no x-request-id header on the live run",
            ),
            (
                "cost_and_usage.request_ids.narration",
                "narrate/status.json is absent from these artifacts",
            ),
        ):
            self._unavailable(field, reason)

        return {
            "provider_reported_usage": {
                "ocr": {
                    "totals": ocr_usage or None,
                    "per_page": ocr_entries,
                    "source": "extract/results.json (token counts from Indic-OCR responses)",
                },
                "translation_body": {
                    "totals": self._dict(translations.get("usage")) or None,
                    "source": "translate/translations.json",
                },
                "translation_scaffold": {
                    "totals": scaffold_usage or None,
                    "source": "compose/lesson.json scaffold[].usage",
                },
                "narration": None,
                "transcription": None,
                "deduplication_note": (
                    "token totals are summed from per-page/per-block artifacts only; "
                    "manifest.json aggregates are not added again to avoid double counting"
                ),
            },
            "provider_reported_cost": None,
            "estimated_cost": {
                "transcription": {
                    "currency": "INR",
                    "amount": estimate,
                    "audio_minutes": round(audio_minutes, 6) if audio_minutes is not None else None,
                    "rate_inr_per_audio_minute": STT_RATE_INR_PER_AUDIO_MINUTE,
                    "label": "ESTIMATE - not provider-reported",
                    "rate_source": (
                        "docstring of src/bodhan/adapters/transcribe/indic_transcribe.py "
                        "(official Bodhan API docs inspected 2026-11-17)"
                    ),
                    "audio_source": "actual section durations in assembly timeline.json",
                    "method": (
                        "sum of measured section WAV durations / 60 * documented rate; "
                        "excludes retries and per-request overhead; split requests are not double-counted"
                    ),
                },
                "narration": None,
            },
            "request_ids": {
                "ocr": [self._dict(e).get("provider_request_id") for e in self._list(extract)],
                "transcription": None,
                "narration": None,
            },
            "successful_request_counts": {
                "ocr": len(
                    [
                        e
                        for e in self._list(extract)
                        if isinstance(e, dict) and e.get("provider_request_id")
                    ]
                ),
                "transcription": sum(
                    int(r.get("successful_request_count") or 0) for r in transcription
                ),
                "basis": "provider request ids / raw response bodies; retry attempts are not persisted",
            },
            "assembly_audio_metadata": assembly.get("output_format"),
        }

    def _retry_failure_summary(self, *, pipeline_status: str) -> dict[str, Any]:
        manifest = self._dict(self._cache.get("work/manifest.json"))
        transcribe_manifest = self._dict(self._cache.get("transcribe/manifest.json"))
        stages = self._dict(manifest.get("stages"))
        terminal: list[dict[str, Any]] = [
            {"stage": name, "status": stage.get("status"), "error": stage.get("error")}
            for name, stage in stages.items()
            if isinstance(stage, dict) and stage.get("status") not in {None, "done"}
        ]
        for section in self._list(transcribe_manifest.get("failed_segments")):
            terminal.append({"stage": "transcribe", "section": section})
        skipped = {
            "compose_skipped_empty": None,
            "narrate_skipped_empty": self._dict(
                self._dict(stages.get("narrate")).get("artifacts")
            ).get("skipped_empty"),
            "transcribe_resumed_cached": transcribe_manifest.get("resumed_cached"),
        }
        self._unavailable(
            "retry_and_failure_summary.skipped_or_resumed.compose_skipped_empty",
            "compose stage does not persist a skipped-empty count",
        )
        self._unavailable(
            "retry_and_failure_summary.total_retries",
            "no stage or adapter artifact persists per-stage retry counters",
        )
        self._unavailable(
            "retry_and_failure_summary.recovered_failures",
            "retry outcomes are not persisted, so recovered failures cannot be reported",
        )
        return {
            "total_retries": None,
            "retries_by_stage": {
                name: None
                for name in (
                    "ocr",
                    "translation",
                    "narration",
                    "transcription",
                    "composition",
                )
            },
            "terminal_failures": terminal,
            "recovered_failures": None,
            "skipped_or_resumed": skipped,
            "partial_success": [
                section
                for section in self._list(transcribe_manifest.get("failed_segments"))
                if isinstance(section, dict)
            ],
            "pipeline_status": pipeline_status,
            "note": "null retry fields are not zero; they mean no retry metadata was persisted",
        }

    # -- public build --------------------------------------------------------
    def build(self) -> dict[str, Any]:
        self.gaps = {}
        self.sources = {}
        self._cache = {}

        work = self.work_dir
        self._load("work/manifest.json", work / "manifest.json")
        self._load("work/ingest/pages.json", work / "ingest/pages.json")
        self._load("work/extract/results.json", work / "extract/results.json")
        self._load("work/structure/blocks.json", work / "structure/blocks.json")
        self._load("work/quality/blocks.json", work / "quality/blocks.json")
        self._load("work/quality/quality_report.json", work / "quality/quality_report.json")
        self._load("work/translate/translations.json", work / "translate/translations.json")
        self._load("work/compose/lesson.json", work / "compose/lesson.json")
        self._load("work/compose/plan.json", work / "compose/plan.json")
        narrate_path = work / "narrate/segments.json"
        if not narrate_path.is_file() and self.transcribe_dir is not None:
            narrate_path = self.transcribe_dir.parent / "narrate/segments.json"
        self._load("narrate/segments.json", narrate_path)
        self._load("narrate/status.json", work / "narrate/status.json")
        self._load(
            "export/manifest.json",
            self.export_dir / "manifest.json" if self.export_dir else None,
        )
        transcribe_manifest = self._dict(
            self._load(
                "transcribe/manifest.json",
                self.transcribe_dir / "manifest.json" if self.transcribe_dir else None,
            )
        )
        transcribe_records = self._records("transcribe", self.transcribe_dir)
        cer_records = self._records("evaluation/cer", self.evaluation_dir)

        manifest = self._dict(self._cache.get("work/manifest.json"))
        stages = self._dict(manifest.get("stages"))
        assembly = self._assembly()
        cer_manifest = self._dict(
            self._load(
                "evaluation/cer/manifest.json",
                self.evaluation_dir / "manifest.json" if self.evaluation_dir else None,
            )
        )
        statuses = [stage.get("status") for stage in stages.values() if isinstance(stage, dict)]
        statuses += [
            transcribe_manifest.get("status"),
            cer_manifest.get("status"),
            assembly.get("assembly_status"),
        ]
        pipeline_status = (
            "done" if statuses and all(status == "done" for status in statuses) else "partial"
        )

        structure = self._structure_quality()
        translations = self._translations()
        narration = self._narration(manifest)
        transcription = self._transcription(transcribe_records)
        cer = self._cer(cer_records)
        cost = self._cost_usage(transcription, assembly)

        report: dict[str, Any] = {
            "report_schema_version": REPORT_SCHEMA_VERSION,
            "run_metadata": self._run_metadata(pipeline_status=pipeline_status),
            "per_page_ocr": self._ocr(self._dict(self._dict(stages).get("extract"))),
            "structure_and_quality": structure,
            "translations": translations,
            "narration": narration,
            "transcription": transcription,
            "cer_evaluation": cer,
            "audio_assembly": assembly,
            "cost_and_usage": cost,
            "retry_and_failure_summary": self._retry_failure_summary(
                pipeline_status=pipeline_status
            ),
            "unavailable_fields": self.gaps,
            "evidence_sources": self.sources,
        }
        report["null_value_reasons"] = _null_value_reasons(report, self.gaps)
        return report
