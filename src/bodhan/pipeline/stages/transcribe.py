"""Stage 9 - transcribe: back-transcribe narrated audio (Indic-Transcribe only).

Runs after narrate: every produced narration segment is sent to Bodhan
Indic-Transcribe - the only speech-to-text provider the assignment allows -
and stored under ``transcribe/`` next to (never merged with) the narration.

Artifacts::

    transcribe/
        segment-01.json   one record per narrated section: section id, audio
                          file, target language, provider/model metadata and
                          the verbatim transcription text
        manifest.json     stage summary: status, provider/model, per-segment
                          outcome, aggregated usage and latency

Checkpoint / resume:
- a record with ``status: "ok"`` for the same audio file (name, size, mtime)
  and target language is reused - completed segments are never retranscribed;
- every record is written atomically as soon as its segment completes, so a
  later failure cannot discard earlier successes;
- the stage raises (and its manifest says ``failed``) unless ALL narrated
  segments have a successful record - it is never marked done with gaps.

Integrity rules (assignment): the provider text is stored verbatim - never
translated, never rewritten by the LLM, never replaced by the English source
- and the narration script stays in ``narrate/segments.json``, a separate
field from ``transcription_text`` here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bodhan.domain.models import AudioSegment, StageName
from bodhan.observability import emit_event
from bodhan.pipeline.errors import PipelineError
from bodhan.pipeline.stages.base import StageContext
from bodhan.pipeline.state import load_input_pages, load_json, save_json, save_pages
from bodhan.ports.transcribe import SUPPORTED_TRANSCRIBE_LANGUAGES, TranscriptionResult

NAME = StageName.TRANSCRIBE


def _load_segments(ctx: StageContext) -> list[AudioSegment]:
    """Narrated segments (the required set - every one must end up transcribed)."""
    payload = load_json(ctx.store, StageName.NARRATE, "segments.json")
    if not isinstance(payload, list) or not payload:
        raise PipelineError("narrate produced no segments - run narrate first")
    try:
        return [AudioSegment.model_validate(item) for item in payload]
    except ValueError as exc:
        raise PipelineError("corrupt narrate/segments.json - re-run narrate") from exc


def _load_records(ctx: StageContext) -> dict[str, dict[str, Any]]:
    """Existing per-segment records keyed by section id (resume input).

    A missing or corrupt record simply means "not transcribed yet".
    """
    records: dict[str, dict[str, Any]] = {}
    directory = ctx.store.stage_dir(NAME.value)
    if not directory.is_dir():
        return records
    for path in sorted(directory.glob("segment-*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(record, dict) and isinstance(record.get("section_id"), str):
            records[record["section_id"]] = record
    return records


def _reusable(record: dict[str, Any], segment: AudioSegment, *, language: str) -> bool:
    """True when an existing record still describes this exact audio.

    Re-transcription happens only when something actually changed: a failed
    record, a different target language, a renamed/replaced/edited audio
    file (size or mtime) or a missing file.
    """
    if record.get("status") != "ok" or record.get("target_language") != language:
        return False
    path = Path(segment.audio_path)
    if record.get("audio_file") != path.name or not path.is_file():
        return False
    try:
        stat = path.stat()
    except OSError:
        return False
    return (
        record.get("audio_size_bytes") == stat.st_size
        and record.get("audio_mtime_ns") == stat.st_mtime_ns
    )


def _ok_record(
    segment: AudioSegment,
    path: Path,
    *,
    language: str,
    result: TranscriptionResult,
) -> dict[str, Any]:
    stat = path.stat()
    return {
        "section_id": segment.segment_id,
        "audio_file": path.name,
        "audio_size_bytes": stat.st_size,
        "audio_mtime_ns": stat.st_mtime_ns,
        "target_language": language,
        "provider": result.provider,
        "model": result.model,
        # Verbatim provider output - the stage never mutates this string.
        "transcription_text": result.text,
        "latency_ms": result.latency_ms,
        "usage": dict(result.usage),
        "request_id": result.request_id,
        "status": "ok",
        "raw": result.raw,
    }


def _failed_record(
    segment: AudioSegment,
    path: Path,
    *,
    language: str,
    provider: str,
    model: str,
    error: str,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "section_id": segment.segment_id,
        "audio_file": path.name,
        "target_language": language,
        "provider": provider,
        "model": model,
        "status": "failed",
        "error": error,
    }
    try:
        stat = path.stat()
    except OSError:
        return record  # missing audio: the reason is already in `error`
    record["audio_size_bytes"] = stat.st_size
    record["audio_mtime_ns"] = stat.st_mtime_ns
    return record


async def run(ctx: StageContext) -> dict[str, Any]:
    """Back-transcribe every narrated segment, checkpointing each result."""
    pages = load_input_pages(ctx.store, NAME)
    save_pages(ctx.store, NAME, pages)
    segments = _load_segments(ctx)

    language = ctx.chapter.target_language_code
    if language not in SUPPORTED_TRANSCRIBE_LANGUAGES:
        raise PipelineError(
            f"target language {language!r} is not in the documented transcription enum "
            f"({len(SUPPORTED_TRANSCRIBE_LANGUAGES)} codes)"
        )

    out_dir = ctx.store.stage_dir(NAME.value)
    prior = _load_records(ctx)
    # Records for sections narrate no longer produces are stale positions.
    expected_files = {f"segment-{index:02d}.json" for index in range(1, len(segments) + 1)}
    for path in out_dir.glob("segment-*.json"):
        if path.name not in expected_files:
            path.unlink(missing_ok=True)

    ok: dict[str, dict[str, Any]] = {}
    failed: list[dict[str, str]] = []
    reused = 0

    for index, segment in enumerate(segments, start=1):
        path = Path(segment.audio_path)
        existing = prior.get(segment.segment_id)
        if existing is not None and _reusable(existing, segment, language=language):
            ok[segment.segment_id] = existing
            reused += 1
            ctx.logger.info("reusing transcription of %s (already done)", segment.segment_id)
            continue

        try:
            if not path.is_file():
                # Fail this segment cleanly instead of asking a provider
                # about audio that is not there.
                raise FileNotFoundError(f"narrated audio is missing: {path}")
            result = await ctx.transcribe.transcribe(path, language=language)
            record = _ok_record(segment, path, language=language, result=result)
        except Exception as exc:  # record the failure, keep every earlier success
            message = f"{type(exc).__name__}: {exc}"
            record = _failed_record(
                segment,
                path,
                language=language,
                provider=ctx.transcribe.provider,
                model=ctx.settings.transcribe_model,
                error=message,
            )
            failed.append({"section_id": segment.segment_id, "error": message})
            ctx.logger.warning("transcription failed for %s: %s", segment.segment_id, message)

        # Atomic per-segment checkpoint: completed work survives any later failure.
        save_json(ctx.store, NAME, f"segment-{index:02d}.json", record)
        if record["status"] == "ok":
            ok[segment.segment_id] = record
            emit_event(
                ctx.logger,
                "transcription_completed",
                run_id=ctx.chapter.chapter_id,
                stage=NAME.value,
                target_language=language,
                section_id=segment.segment_id,
                chars=len(record["transcription_text"]),
                latency_ms=int(record.get("latency_ms", 0) or 0),
            )

    # A segment with no successful record at all (e.g. audio file missing
    # before the call) also counts as failed - success requires every segment.
    missing = [segment.segment_id for segment in segments if segment.segment_id not in ok]
    if missing and not failed:
        failed = [
            {"section_id": section_id, "error": "no successful transcription recorded"}
            for section_id in missing
        ]

    entries: list[dict[str, Any]] = []
    usage_totals: dict[str, int | float] = {}
    total_latency_ms = 0
    for segment in segments:
        stored = ok.get(segment.segment_id)
        entry: dict[str, Any] = {
            "section_id": segment.segment_id,
            "audio_file": Path(segment.audio_path).name,
            "status": stored["status"] if stored is not None else "failed",
        }
        if stored is None:
            reason = next(
                (item["error"] for item in failed if item["section_id"] == segment.segment_id),
                "no successful transcription recorded",
            )
            entry["error"] = reason
            entries.append(entry)
            continue
        entry["latency_ms"] = stored.get("latency_ms", 0)
        entry["request_id"] = stored.get("request_id")
        entries.append(entry)
        total_latency_ms += int(stored.get("latency_ms", 0) or 0)
        usage = stored.get("usage", {})
        if isinstance(usage, dict):
            for key, value in usage.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                usage_totals[key] = usage_totals.get(key, 0) + value

    save_json(
        ctx.store,
        NAME,
        "manifest.json",
        {
            "status": "done" if not missing else "failed",
            "target_language": language,
            "provider": ctx.transcribe.provider,
            "model": ctx.settings.transcribe_model,
            "total_segments": len(segments),
            "resumed_cached": reused,
            "completed_segments": [
                segment.segment_id for segment in segments if segment.segment_id in ok
            ],
            "failed_segments": failed,
            "segments": entries,
            "usage": usage_totals,
            "total_latency_ms": total_latency_ms,
        },
    )

    if missing:
        raise PipelineError(
            f"transcription incomplete: {len(ok)}/{len(segments)} segment(s) succeeded; "
            f"missing: {', '.join(missing)} ({len(ok)} successful transcription(s) preserved; "
            "re-run to resume)"
        )

    ctx.state["transcriptions"] = {
        segment.segment_id: ok[segment.segment_id]["transcription_text"] for segment in segments
    }
    return {
        "segments": len(segments),
        "transcribed": len(ok),
        "resumed_cached": reused,
        "target_language": language,
        "provider": ctx.transcribe.provider,
        "model": ctx.settings.transcribe_model,
        "usage": usage_totals,
        "total_latency_ms": total_latency_ms,
    }
