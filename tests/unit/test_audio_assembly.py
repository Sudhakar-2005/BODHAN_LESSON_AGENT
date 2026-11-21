"""Deterministic offline tests for full-lesson audio assembly (Phase 2C)."""

from __future__ import annotations

import hashlib
import wave
from pathlib import Path
from typing import Any

import pytest

from bodhan.assembly import AudioAssemblyError, assemble_audio, require_ffmpeg
from bodhan.assembly.audio import plan_sections

MARKDOWN = "# Lesson Title\n## Section Alpha\n## Section Beta\n"


def make_wav(
    path: Path, *, frames: int = 8000, rate: int = 8000, channels: int = 1, width: int = 2
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(rate)
        handle.writeframes(b"\x00" * frames * channels * width)
    return path


def seed(tmp_path: Path) -> tuple[list[dict[str, Any]], Path]:
    source = tmp_path / "narrate"
    first = make_wav(source / "a.wav", frames=8000)
    second = make_wav(source / "b.wav", frames=16000)
    segments = [
        {
            "segment_id": "seg-02",
            "text": "Section Beta. Second section.",
            "audio_path": str(second),
        },
        {
            "segment_id": "seg-01",
            "text": "Section Alpha. First section.",
            "audio_path": str(first),
        },
    ]
    return segments, source


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- ordering, timeline and concatenation ----------------------------------


def test_sections_are_ordered_by_lesson_not_list_position(tmp_path: Path) -> None:
    segments, source = seed(tmp_path)
    result = assemble_audio(
        segments=segments, markdown=MARKDOWN, output_dir=tmp_path / "audio", source_dir=source
    )

    assert [section["section_id"] for section in result.timeline["sections"]] == [
        "seg-01",
        "seg-02",
    ]
    assert [section["section_title"] for section in result.timeline["sections"]] == [
        "Section Alpha",
        "Section Beta",
    ]
    assert result.manifest["ordering_basis"].startswith("compose/lesson.json")


def test_timeline_boundaries_durations_and_total(tmp_path: Path) -> None:
    segments, source = seed(tmp_path)
    result = assemble_audio(
        segments=segments, markdown=MARKDOWN, output_dir=tmp_path / "audio", source_dir=source
    )
    sections = result.timeline["sections"]

    assert [(s["start_time_seconds"], s["end_time_seconds"]) for s in sections] == [
        (0.0, 1.0),
        (1.0, 3.0),
    ]
    assert [s["duration_seconds"] for s in sections] == [1.0, 2.0]
    assert [s["sequence_number"] for s in sections] == [1, 2]
    assert result.timeline["total_duration_seconds"] == 3.0


def test_compatible_wav_frames_are_concatenated_without_transcoding(tmp_path: Path) -> None:
    segments, source = seed(tmp_path)
    before = [digest(Path(segment["audio_path"])) for segment in segments]

    result = assemble_audio(
        segments=segments, markdown=MARKDOWN, output_dir=tmp_path / "audio", source_dir=source
    )
    output = tmp_path / "audio/full_lesson.wav"
    with wave.open(str(output), "rb") as handle:
        assert handle.getnframes() == 24000
        assert (handle.getnchannels(), handle.getframerate(), handle.getsampwidth()) == (1, 8000, 2)
    assert result.manifest["transcoding"] == "none; PCM frames concatenated byte-for-byte"
    assert digest(Path(segments[0]["audio_path"])) == before[0]
    assert digest(Path(segments[1]["audio_path"])) == before[1]


def test_original_audio_and_archived_copies_are_byte_identical(tmp_path: Path) -> None:
    segments, source = seed(tmp_path)
    assemble_audio(
        segments=segments, markdown=MARKDOWN, output_dir=tmp_path / "audio", source_dir=source
    )
    for segment in segments:
        original = Path(segment["audio_path"])
        archive = tmp_path / "audio/segments" / original.name
        assert archive.is_file()
        assert digest(archive) == digest(original)


def test_no_embedded_chapters_are_claimed(tmp_path: Path) -> None:
    segments, source = seed(tmp_path)
    result = assemble_audio(
        segments=segments, markdown=MARKDOWN, output_dir=tmp_path / "audio", source_dir=source
    )
    assert result.timeline["embedded_chapters"] is False
    assert "sidecar only" in result.timeline["marker_source"]
    assert result.manifest["embedded_chapters"] is False
    assert "WAV" in result.manifest["embedded_chapters_reason"]


# --- overlapping headings ---------------------------------------------------


def test_short_heading_and_longer_heading_share_a_prefix() -> None:
    markdown = "# Summary and Questions\n## Summary\n"
    segments = [
        {"segment_id": "seg-01", "text": "Summary and Questions. Body.", "audio_path": "a.wav"},
        {"segment_id": "seg-02", "text": "Summary. Body.", "audio_path": "b.wav"},
    ]

    planned = plan_sections(segments, markdown)

    assert [section["section_id"] for section in planned] == ["seg-01", "seg-02"]
    assert [section["section_title"] for section in planned] == [
        "Summary and Questions",
        "Summary",
    ]


def test_longest_matching_heading_wins_regardless_of_segment_order() -> None:
    markdown = "# Summary and Questions\n## Summary\n"
    segments = [
        {"segment_id": "seg-02", "text": "Summary. Body.", "audio_path": "b.wav"},
        {"segment_id": "seg-01", "text": "Summary and Questions. Body.", "audio_path": "a.wav"},
    ]

    planned = plan_sections(segments, markdown)

    assert [section["section_title"] for section in planned] == [
        "Summary and Questions",
        "Summary",
    ]
    assert planned[0]["section_id"] == "seg-01"


def test_equal_length_ambiguous_matches_are_rejected() -> None:
    # Two raw headings normalize to the same string, so both are equally long
    # prefix matches: that is ambiguous and must never be guessed.
    markdown = "# Section  One\n## Section One\n"
    segments = [{"segment_id": "seg-01", "text": "Section One. Body.", "audio_path": "a.wav"}]

    with pytest.raises(AudioAssemblyError, match="ambiguous longest matches"):
        plan_sections(segments, markdown)


def test_prefix_free_headings_keep_their_behavior() -> None:
    markdown = "# Alpha\n## Beta\n"
    segments = [
        {"segment_id": "seg-02", "text": "Beta. Second.", "audio_path": "b.wav"},
        {"segment_id": "seg-01", "text": "Alpha. First.", "audio_path": "a.wav"},
    ]

    planned = plan_sections(segments, markdown)

    assert [section["section_title"] for section in planned] == ["Alpha", "Beta"]
    assert [section["section_id"] for section in planned] == ["seg-01", "seg-02"]


# --- explicit failure handling ---------------------------------------------


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "missing audio"),
        ("corrupt", "corrupted"),
        ("incompatible", "incompatible audio metadata"),
        ("empty", "no samples"),
    ],
)
def test_audio_problems_are_reported_explicitly(
    tmp_path: Path, mutation: str, message: str
) -> None:
    segments, source = seed(tmp_path)
    target = Path(segments[1]["audio_path"])
    if mutation == "missing":
        target.unlink()
    elif mutation == "corrupt":
        target.write_bytes(b"not a wav at all")
    elif mutation == "incompatible":
        make_wav(target, frames=16000, rate=16000)
    elif mutation == "empty":
        make_wav(target, frames=0)

    with pytest.raises(AudioAssemblyError, match=message):
        assemble_audio(
            segments=segments, markdown=MARKDOWN, output_dir=tmp_path / "audio", source_dir=source
        )


def test_unmapped_section_fails_instead_of_guessing_order(tmp_path: Path) -> None:
    segments, source = seed(tmp_path)
    segments[0]["text"] = "Unrelated heading. No lesson match."

    with pytest.raises(AudioAssemblyError, match="cannot map section"):
        assemble_audio(
            segments=segments, markdown=MARKDOWN, output_dir=tmp_path / "audio", source_dir=source
        )


def test_empty_segments_and_missing_headings_fail(tmp_path: Path) -> None:
    with pytest.raises(AudioAssemblyError, match="no narration segments"):
        assemble_audio(segments=[], markdown=MARKDOWN, output_dir=tmp_path / "audio")
    segments, source = seed(tmp_path)
    with pytest.raises(AudioAssemblyError, match="no Markdown headings"):
        assemble_audio(
            segments=segments,
            markdown="plain text",
            output_dir=tmp_path / "audio",
            source_dir=source,
        )


def test_ffmpeg_requirement_gives_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("bodhan.assembly.audio.shutil.which", lambda _: None)
    with pytest.raises(AudioAssemblyError, match="Install ffmpeg"):
        require_ffmpeg(reason="demux a non-WAV section")
    monkeypatch.setattr("bodhan.assembly.audio.shutil.which", lambda _: "/usr/bin/ffmpeg")
    assert require_ffmpeg(reason="test") == "/usr/bin/ffmpeg"


# --- reruns -----------------------------------------------------------------


def test_rerun_reuses_valid_output_and_rebuilds_after_change(tmp_path: Path) -> None:
    segments, source = seed(tmp_path)
    output_dir = tmp_path / "audio"
    first = assemble_audio(
        segments=segments, markdown=MARKDOWN, output_dir=output_dir, source_dir=source
    )
    wav_path = output_dir / "full_lesson.wav"
    first_digest = digest(wav_path)

    second = assemble_audio(
        segments=segments, markdown=MARKDOWN, output_dir=output_dir, source_dir=source
    )
    assert second.reused is True
    assert second.manifest == first.manifest  # strict idempotency: nothing rewritten
    assert digest(wav_path) == first_digest
    assert second.timeline == first.timeline

    seg_one = next(segment for segment in segments if segment["segment_id"] == "seg-01")
    make_wav(Path(seg_one["audio_path"]), frames=12000)
    third = assemble_audio(
        segments=segments, markdown=MARKDOWN, output_dir=output_dir, source_dir=source
    )
    assert third.reused is False
    assert digest(wav_path) != first_digest
    assert third.timeline["sections"][0]["duration_seconds"] == 1.5


def test_corrupt_output_is_rebuilt_on_rerun(tmp_path: Path) -> None:
    segments, source = seed(tmp_path)
    output_dir = tmp_path / "audio"
    assemble_audio(segments=segments, markdown=MARKDOWN, output_dir=output_dir, source_dir=source)
    (output_dir / "full_lesson.wav").write_bytes(b"damaged")

    result = assemble_audio(
        segments=segments, markdown=MARKDOWN, output_dir=output_dir, source_dir=source
    )
    assert result.reused is False
    with wave.open(str(output_dir / "full_lesson.wav"), "rb") as handle:
        assert handle.getnframes() == 24000
