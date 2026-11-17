"""Assemble narrated section WAV files into one lesson WAV + sidecar timeline.

No TTS call is made here: the existing Indic-Speak WAV files are only read
(and copied for the archive). PCM frames are concatenated byte-for-byte with
the stdlib ``wave`` module, so there is no transcoding or quality loss and no
new dependency. Files with different sample rate, width, channels or
compression are rejected explicitly instead of being silently resampled.

WAV has no embedded chapter-track format, so section markers are exposed in
``timeline.json`` only; ``embedded_chapters`` is always false and this is
stated in both the timeline and the assembly manifest (never claimed present).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import unicodedata
import wave
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bodhan.services.json_io import write_json_atomic

TIMELINE_VERSION = "1.0"
_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)


class AudioAssemblyError(Exception):
    """An explicit, recoverable audio assembly failure (never silently ignored)."""


@dataclass
class AssemblyResult:
    """Generated manifest, timeline and reuse flag for one assembly run."""

    manifest: dict[str, Any]
    timeline: dict[str, Any]
    reused: bool


def require_ffmpeg(*, reason: str) -> str:
    """Return the ffmpeg path or fail with an actionable, explicit error.

    The current WAV path does not need ffmpeg; this helper exists for any
    future input container that genuinely requires demuxing/conversion.
    """
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise AudioAssemblyError(
            f"ffmpeg is required ({reason}) but was not found on PATH. "
            "Install ffmpeg and add it to PATH, or supply PCM WAV sections "
            "so the stdlib 'wave' assembler can be used without conversion."
        )
    return executable


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def _lesson_headings(markdown: str) -> list[str]:
    """Section titles in lesson order (Markdown headings of the composed lesson)."""
    return [match.group(1).strip() for match in _HEADING.finditer(markdown)]


def plan_sections(segments: list[dict[str, Any]], markdown: str) -> list[dict[str, Any]]:
    """Order narration segments by lesson headings and attach their titles.

    Each narrate segment's ``text`` starts with the translated heading that
    produced it, so matching that prefix is stronger evidence of lesson order
    than the segment id or the filesystem order. Unmatched or ambiguous
    sections are errors - no order is guessed.
    """
    if not segments:
        raise AudioAssemblyError("no narration segments available for assembly")
    headings = _lesson_headings(markdown)
    if not headings:
        raise AudioAssemblyError(
            "compose/lesson.json contains no Markdown headings; cannot establish lesson order"
        )

    planned: list[dict[str, Any]] = []
    used: set[str] = set()
    for segment in segments:
        section_id = segment.get("segment_id")
        text = segment.get("text")
        if not isinstance(section_id, str) or not section_id:
            raise AudioAssemblyError("a narration segment has no segment_id")
        if not isinstance(text, str):
            raise AudioAssemblyError(f"narration segment {section_id} has no text")
        normalized_text = _normalize(text)
        matches = [
            heading for heading in headings if normalized_text.startswith(_normalize(heading))
        ]
        if len(matches) != 1:
            raise AudioAssemblyError(
                f"cannot map section {section_id} to exactly one lesson heading "
                f"(matches: {matches})"
            )
        title = matches[0]
        if title in used:
            raise AudioAssemblyError(f"two narration sections map to heading {title!r}")
        used.add(title)
        planned.append(
            {
                "section_id": section_id,
                "section_title": title,
                "text": text,
                "audio_path": segment.get("audio_path"),
            }
        )

    planned.sort(key=lambda item: headings.index(item["section_title"]))
    return planned


def _resolve_audio(segment: dict[str, Any], source_dir: Path | None) -> Path:
    audio_path = segment.get("audio_path")
    if not isinstance(audio_path, str) or not audio_path:
        raise AudioAssemblyError(f"section {segment['section_id']} has no audio_path")
    candidate = Path(audio_path)
    if not candidate.is_absolute() and source_dir is not None:
        candidate = source_dir / candidate
    if not candidate.is_file() and source_dir is not None:
        candidate = source_dir / Path(audio_path).name
    if not candidate.is_file():
        raise AudioAssemblyError(f"missing audio for section {segment['section_id']}: {audio_path}")
    return candidate


def _read_wav(path: Path, section_id: str) -> tuple[dict[str, Any], bytes]:
    """Read one WAV; reject empty, corrupt and inconsistent payloads explicitly."""
    if path.stat().st_size == 0:
        raise AudioAssemblyError(f"empty audio file for section {section_id}: {path.name}")
    try:
        with wave.open(str(path), "rb") as handle:
            info: dict[str, Any] = {
                "channels": handle.getnchannels(),
                "sample_width_bytes": handle.getsampwidth(),
                "sample_rate_hz": handle.getframerate(),
                "compression": handle.getcomptype() or "NONE",
                "frames": handle.getnframes(),
            }
            frames = handle.readframes(handle.getnframes())
    except (wave.Error, EOFError, OSError) as exc:
        raise AudioAssemblyError(
            f"corrupted/unsupported audio for section {section_id}: {path.name} ({exc})"
        ) from exc
    if info["frames"] == 0 or not frames:
        raise AudioAssemblyError(f"audio for section {section_id} has no samples: {path.name}")
    if info["sample_rate_hz"] <= 0 or info["channels"] <= 0 or info["sample_width_bytes"] <= 0:
        raise AudioAssemblyError(f"invalid WAV header for section {section_id}: {path.name}")
    info["duration_seconds"] = info["frames"] / info["sample_rate_hz"]
    return info, frames


def _write_wav_atomic(path: Path, params: dict[str, Any], frames: bytes) -> None:
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        with wave.open(str(temporary), "wb") as output:
            output.setnchannels(params["channels"])
            output.setsampwidth(params["sample_width_bytes"])
            output.setframerate(params["sample_rate_hz"])
            output.writeframes(frames)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _existing_reusable(
    output_dir: Path, inputs: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Return stored manifest+timeline only if the prior output is still valid."""
    manifest_path = output_dir / "assembly_manifest.json"
    timeline_path = output_dir / "timeline.json"
    audio_path = output_dir / "full_lesson.wav"
    if not all(path.is_file() for path in (manifest_path, timeline_path, audio_path)):
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    if manifest.get("status") != "done" or manifest.get("inputs") != inputs:
        return None
    if _sha256(audio_path) != manifest.get("output", {}).get("sha256"):
        return None
    return manifest, timeline


def assemble_audio(
    *,
    segments: list[dict[str, Any]],
    markdown: str,
    output_dir: Path,
    source_dir: Path | None = None,
) -> AssemblyResult:
    """Build ``full_lesson.wav``, ``timeline.json`` and ``assembly_manifest.json``.

    ``source_dir`` is only used to resolve relative audio paths; originals are
    never modified. A rerun reuses the existing output only when all input
    hashes and the output hash still match the stored manifest.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    segments_dir = output_dir / "segments"
    planned = plan_sections(segments, markdown)

    entries: list[dict[str, Any]] = []
    parts: list[bytes] = []
    reference: dict[str, Any] | None = None
    for sequence, section in enumerate(planned, start=1):
        source = _resolve_audio(section, source_dir)
        info, frames = _read_wav(source, section["section_id"])
        if reference is None:
            reference = info
        else:
            comparable = ("channels", "sample_width_bytes", "sample_rate_hz", "compression")
            if any(info[key] != reference[key] for key in comparable):
                raise AudioAssemblyError(
                    f"incompatible audio metadata for section {section['section_id']}: "
                    f"{info} != {reference}"
                )
        archived = segments_dir / source.name
        if not archived.is_file() or _sha256(archived) != _sha256(source):
            segments_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, archived)  # original stays byte-for-byte unchanged
        parts.append(frames)
        entries.append(
            {
                "section_id": section["section_id"],
                "section_title": section["section_title"],
                "audio_filename": archived.name,
                "sequence_number": sequence,
                "audio_info": info,
                "source_sha256": _sha256(source),
            }
        )

    assert reference is not None  # plan_sections rejects an empty segment list
    inputs = [
        {
            "section_id": entry["section_id"],
            "audio_filename": entry["audio_filename"],
            "sha256": entry["source_sha256"],
        }
        for entry in entries
    ]
    cached = _existing_reusable(output_dir, inputs)
    if cached is not None:
        # Strict idempotency: a valid prior assembly is returned unchanged,
        # with no artifact rewritten (checkpoint reuse is in AssemblyResult).
        cached_manifest, cached_timeline = cached
        return AssemblyResult(manifest=cached_manifest, timeline=cached_timeline, reused=True)

    timeline_sections: list[dict[str, Any]] = []
    cursor_frames = 0
    for entry in entries:
        frames_count = int(entry["audio_info"]["frames"])
        rate = int(entry["audio_info"]["sample_rate_hz"])
        start = cursor_frames / rate
        end = (cursor_frames + frames_count) / rate
        timeline_sections.append(
            {
                "section_id": entry["section_id"],
                "section_title": entry["section_title"],
                "audio_filename": entry["audio_filename"],
                "start_time_seconds": round(start, 6),
                "end_time_seconds": round(end, 6),
                "duration_seconds": round(end - start, 6),
                "sequence_number": entry["sequence_number"],
            }
        )
        cursor_frames += frames_count

    total_duration = cursor_frames / reference["sample_rate_hz"]
    audio_path = output_dir / "full_lesson.wav"
    _write_wav_atomic(audio_path, reference, b"".join(parts))

    timeline: dict[str, Any] = {
        "timeline_version": TIMELINE_VERSION,
        "audio_filename": audio_path.name,
        "total_duration_seconds": round(total_duration, 6),
        "format": {
            "container": "WAV",
            "encoding": "PCM (uncompressed)",
            "channels": reference["channels"],
            "sample_rate_hz": reference["sample_rate_hz"],
            "sample_width_bytes": reference["sample_width_bytes"],
        },
        "sections": timeline_sections,
        "embedded_chapters": False,
        "marker_source": "timeline.json sidecar only; WAV has no embedded chapter tracks",
    }
    manifest: dict[str, Any] = {
        "status": "done",
        "assembly": "full_lesson_audio",
        "ordering_basis": "compose/lesson.json Markdown section headings matched to narration text",
        "output": {
            "filename": audio_path.name,
            "sha256": _sha256(audio_path),
            "size_bytes": audio_path.stat().st_size,
            "duration_seconds": round(total_duration, 6),
            "format": dict(timeline["format"]),
        },
        "sections": entries,
        "inputs": inputs,
        "timeline_file": "timeline.json",
        "section_count": len(entries),
        "missing_sections": [],
        "failed_sections": [],
        "embedded_chapters": False,
        "embedded_chapters_reason": (
            "WAV container has no chapter-marker support; section markers are in the "
            "timeline.json sidecar only (no embedded chapters are claimed)."
        ),
        "ffmpeg_available": shutil.which("ffmpeg") is not None,
        "ffmpeg_required": False,
        "transcoding": "none; PCM frames concatenated byte-for-byte",
        "created_at": datetime.now(UTC).isoformat(),
    }
    write_json_atomic(output_dir / "timeline.json", timeline)
    write_json_atomic(output_dir / "assembly_manifest.json", manifest)
    return AssemblyResult(manifest=manifest, timeline=timeline, reused=False)
