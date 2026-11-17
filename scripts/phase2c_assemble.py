"""Phase 2C: assemble the six existing Indic-Speak WAVs into one lesson WAV.

Read-only on all Phase 1/2A/2B artifacts. No Indic-Speak call, no alternative
TTS, no transcoding; output is written under ``data/work/phase2c-verify/audio``.

Usage::

    uv run python scripts/phase2c_assemble.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from bodhan.assembly import AudioAssemblyError, assemble_audio

WORKSPACE = Path("data/work/phase1-verify")
PHASE2A = Path("data/work/phase2a-verify")
OUTPUT = Path("data/work/phase2c-verify/audio")


def _load(path: Path) -> Any:
    if not path.is_file():
        raise SystemExit(f"missing required artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    segments = _load(PHASE2A / "narrate/segments.json")
    lesson = _load(WORKSPACE / "compose/lesson.json")
    if not isinstance(segments, list) or not isinstance(lesson, dict):
        raise SystemExit("unexpected narrate/compose artifact schema")

    try:
        result = assemble_audio(
            segments=segments,
            markdown=str(lesson.get("markdown") or ""),
            output_dir=OUTPUT,
            source_dir=WORKSPACE / "narrate",
        )
    except AudioAssemblyError as exc:
        print(f"audio assembly failed: {exc}", file=sys.stderr)
        return 1

    print(
        f"assembly: status={result.manifest['status']} "
        f"reused={result.reused} sections={result.manifest['section_count']} "
        f"duration={result.timeline['total_duration_seconds']}s"
    )
    for section in result.timeline["sections"]:
        print(
            f"  {section['sequence_number']}. {section['section_id']} "
            f"[{section['start_time_seconds']:.3f} - {section['end_time_seconds']:.3f}] "
            f"{section['section_title']}"
        )
    print(f"embedded chapters: {result.manifest['embedded_chapters']} (sidecar timeline only)")
    print(f"artifacts: {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
