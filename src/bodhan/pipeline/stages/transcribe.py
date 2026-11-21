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

import hashlib
import io
import json
import tempfile
import wave
from pathlib import Path
from typing import Any

from bodhan.adapters.transcribe.errors import TranscriptionProviderError
from bodhan.adapters.transcribe.indic_transcribe import split_wav
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


# -- per-chunk checkpoints ---------------------------------------------------
# A segment is split into <=30 s pieces by the same helper the adapter uses.
# Each successful piece is persisted on its own so a 429 on a later piece can
# never discard earlier work, and a resume reuses every completed piece.


def _wav_meta(data: bytes) -> dict[str, int] | None:
    """Rate/channels/width/frames for a WAV piece, or None if not measurable."""
    try:
        with wave.open(io.BytesIO(data), "rb") as handle:
            return {
                "sample_rate": handle.getframerate(),
                "channels": handle.getnchannels(),
                "sample_width": handle.getsampwidth(),
                "frames": handle.getnframes(),
            }
    except (wave.Error, EOFError):
        return None


def _chunk_identity(
    path: Path,
    metas: list[dict[str, int] | None],
    pieces: list[bytes],
    index: int,
    language: str,
) -> dict[str, Any]:
    """Full identity of one chunk: audio, index, boundaries and language."""
    stat = path.stat()
    meta = metas[index - 1]
    identity: dict[str, Any] = {
        "audio_file": path.name,
        "audio_size_bytes": stat.st_size,
        "audio_mtime_ns": stat.st_mtime_ns,
        "chunk_index": index,
        "chunk_count": len(pieces),
        "chunk_sha256": hashlib.sha256(pieces[index - 1]).hexdigest(),
        "target_language": language,
    }
    if meta is not None:
        identity.update(
            {
                "sample_rate": meta["sample_rate"],
                "channels": meta["channels"],
                "sample_width": meta["sample_width"],
                "start_frame": sum(m["frames"] for m in metas[: index - 1] if m),
                "end_frame": sum(m["frames"] for m in metas[:index] if m),
            }
        )
    return identity


def _chunk_path(ctx: StageContext, section_id: str, index: int) -> Path:
    return ctx.store.stage_dir(NAME.value) / "chunks" / section_id / f"chunk-{index:02d}.json"


def _load_chunk(ctx: StageContext, section_id: str, index: int) -> dict[str, Any] | None:
    path = _chunk_path(ctx, section_id, index)
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return record if isinstance(record, dict) else None


def _chunk_reusable(record: dict[str, Any] | None, identity: dict[str, Any]) -> bool:
    """A chunk is reusable only on status ok AND an exact identity match."""
    return (
        record is not None
        and record.get("status") == "ok"
        and record.get("identity") == identity
        and isinstance(record.get("text"), str)
    )


def _ok_record_from_chunks(
    segment: AudioSegment,
    path: Path,
    *,
    language: str,
    chunks: list[dict[str, Any]],
    text: str,
    reused: int,
) -> dict[str, Any]:
    """Rebuild the documented segment record from verbatim chunk results."""
    stat = path.stat()
    usage: dict[str, int | float] = {}
    raw: list[dict[str, Any]] = []
    request_ids: list[str] = []
    providers: set[str] = set()
    models: set[str] = set()
    latency = 0
    for chunk in chunks:
        latency += int(chunk.get("latency_ms", 0) or 0)
        for key, value in (chunk.get("usage") or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                usage[key] = usage.get(key, 0) + value
        raw.extend(chunk.get("raw") or [])
        if chunk.get("request_id"):
            request_ids.append(str(chunk["request_id"]))
        if chunk.get("provider"):
            providers.add(str(chunk["provider"]))
        if chunk.get("model"):
            models.add(str(chunk["model"]))
    return {
        "section_id": segment.segment_id,
        "audio_file": path.name,
        "audio_size_bytes": stat.st_size,
        "audio_mtime_ns": stat.st_mtime_ns,
        "target_language": language,
        "provider": next(iter(providers)) if len(providers) == 1 else ",".join(sorted(providers)),
        "model": next(iter(models)) if len(models) == 1 else ",".join(sorted(models)),
        # Exact chunk order, joined verbatim - never rewritten or omitted.
        "transcription_text": text,
        "latency_ms": latency,
        "usage": usage,
        "request_id": ",".join(request_ids) if request_ids else None,
        "status": "ok",
        "raw": raw,
        "chunk_count": len(chunks),
        "chunks_reused": reused,
    }


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
    chunks_reused = 0

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
            data = path.read_bytes()
            pieces = split_wav(data, ctx.settings.transcribe_max_audio_s)
            metas = [_wav_meta(piece) for piece in pieces]
            texts: list[str] = []
            chunk_records: list[dict[str, Any]] = []
            failed_chunks: list[int] = []
            reused_here = 0
            # Only temporary COPIES of a piece are written; the source audio is
            # never modified. They are deleted as soon as the segment ends.
            with tempfile.TemporaryDirectory(prefix="bodhan-transcribe-") as tmp:
                tmp_dir = Path(tmp)
                for chunk_index, piece in enumerate(pieces, start=1):
                    identity = _chunk_identity(path, metas, pieces, chunk_index, language)
                    prior_chunk = _load_chunk(ctx, segment.segment_id, chunk_index)
                    if _chunk_reusable(prior_chunk, identity):
                        texts.append(prior_chunk["text"])  # type: ignore[index]
                        chunk_records.append(prior_chunk)  # type: ignore[arg-type]
                        reused_here += 1
                        chunks_reused += 1
                        ctx.logger.info(
                            "reusing chunk %s/%s of %s",
                            chunk_index,
                            len(pieces),
                            segment.segment_id,
                        )
                        continue

                    if len(pieces) == 1:
                        # Keep the original path for single-chunk segments:
                        # provider-facing naming/metadata stay unchanged.
                        piece_path = path
                    else:
                        piece_path = (
                            tmp_dir / f"{path.stem}.part{chunk_index:02d}{path.suffix or '.wav'}"
                        )
                        piece_path.write_bytes(piece)
                    try:
                        result = await ctx.transcribe.transcribe(piece_path, language=language)
                    except Exception as chunk_exc:  # persisted as a chunk failure
                        chunk_message = f"{type(chunk_exc).__name__}: {chunk_exc}"
                        chunk_record: dict[str, Any] = {
                            "status": "failed",
                            "identity": identity,
                            "error": chunk_message,
                        }
                        save_json(
                            ctx.store,
                            NAME,
                            f"chunks/{segment.segment_id}/chunk-{chunk_index:02d}.json",
                            chunk_record,
                        )
                        chunk_records.append(chunk_record)
                        failed_chunks.append(chunk_index)
                        ctx.logger.warning(
                            "transcription failed for %s chunk %s: %s",
                            segment.segment_id,
                            chunk_index,
                            chunk_message,
                        )
                        continue

                    ok_chunk: dict[str, Any] = {
                        "status": "ok",
                        "identity": identity,
                        "text": result.text,
                        "latency_ms": result.latency_ms,
                        "usage": dict(result.usage),
                        "request_id": result.request_id,
                        "provider": result.provider,
                        "model": result.model,
                        "raw": result.raw,
                    }
                    # Atomic per-chunk checkpoint: survives any later failure.
                    save_json(
                        ctx.store,
                        NAME,
                        f"chunks/{segment.segment_id}/chunk-{chunk_index:02d}.json",
                        ok_chunk,
                    )
                    texts.append(result.text)
                    chunk_records.append(ok_chunk)

            if failed_chunks:
                # A gap is never silently omitted - the whole segment fails and
                # only the missing chunks are retried on the next run. Keep the
                # underlying provider error in the segment message for evidence.
                details = "; ".join(
                    f"chunk {chunk['identity']['chunk_index']}: {chunk.get('error', '')}"
                    for chunk in chunk_records
                    if chunk.get("status") == "failed"
                )
                raise TranscriptionProviderError(
                    f"chunk(s) failed for {segment.segment_id}: {failed_chunks} ({details})"
                )
            record = _ok_record_from_chunks(
                segment,
                path,
                language=language,
                chunks=chunk_records,
                text=" ".join(texts),
                reused=reused_here,
            )
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
            "chunks_reused": chunks_reused,
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
        "chunks_reused": chunks_reused,
        "target_language": language,
        "provider": ctx.transcribe.provider,
        "model": ctx.settings.transcribe_model,
        "usage": usage_totals,
        "total_latency_ms": total_latency_ms,
    }
