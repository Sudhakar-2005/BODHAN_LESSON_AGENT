"""Phase 2A controlled live validation: back-transcribe the Phase 1 Hindi WAVs.

Reads the six Indic-Speak WAV segments of the existing Hindi Phase 1
verification run (``data/work/phase1-verify/narrate``), executes the REAL
``transcribe`` pipeline stage against the REAL Bodhan Indic-Transcribe API
(no mocks, no alternative STT provider), and stores every result in a NEW
Phase 2A workspace (``data/work/phase2a-verify``).

- Phase 1 artifacts are only ever READ: the narrate manifests are copied
  into the new workspace, the WAV files are opened read-only.
- Resume-safe: a re-run transcribes only segments whose record is missing
  or whose audio changed (segment records are checkpointed after each call).
- Documented billing: about INR 0.10 per audio minute (Phase 1 total is
  ~74 s of speech -> 7 API requests because seg-06 is 30.55 s and is split
  at the documented 30 s per-request ceiling).
- API keys are never printed (settings fields are repr=False).

Usage::

    uv run python scripts/phase2a_live_transcribe.py
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import shutil
import sys
from pathlib import Path

from bodhan.config import load_settings
from bodhan.domain.models import Chapter, StageName
from bodhan.observability import get_logger
from bodhan.pipeline.errors import PipelineError
from bodhan.pipeline.stages import transcribe
from bodhan.pipeline.stages.base import StageContext
from bodhan.services.artifact_store import FileArtifactStore
from bodhan.wiring import build_providers

SOURCE_NARRATE = Path("data/work/phase1-verify/narrate")
TARGET_WORKSPACE = Path("data/work/phase2a-verify")
STAGE_INPUTS = ("pages.json", "segments.json")


def _setup_workspace(store: FileArtifactStore) -> None:
    """Copy the Phase 1 narrate manifests into the new Phase 2A workspace.

    The WAV files themselves are NOT copied: their ``audio_path`` entries
    keep pointing at the Phase 1 files, which are only ever read.
    """
    narrate_dir = store.stage_dir(StageName.NARRATE.value)
    narrate_dir.mkdir(parents=True, exist_ok=True)
    for name in STAGE_INPUTS:
        source = SOURCE_NARRATE / name
        if not source.is_file():
            raise SystemExit(f"missing Phase 1 artifact: {source}")
        shutil.copy2(source, narrate_dir / name)


def _report(ctx: StageContext) -> None:
    """Print provider metadata + the verbatim Hindi texts for manual review."""
    stage_dir = ctx.store.stage_dir(StageName.TRANSCRIBE.value)
    manifest_path = stage_dir / "manifest.json"
    if not manifest_path.is_file():
        print("no manifest yet: the stage failed before completing any segment")
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print()
    print(
        f"manifest: status={manifest['status']} "
        f"provider={manifest['provider']} model={manifest['model']} "
        f"language={manifest['target_language']} "
        f"segments={manifest['total_segments']} "
        f"resumed_cached={manifest['resumed_cached']} "
        f"total_latency_ms={manifest['total_latency_ms']} "
        f"usage={json.dumps(manifest['usage'], ensure_ascii=False)}"
    )
    for entry in manifest["segments"]:
        print(f"  {entry['section_id']} [{entry['status']}] {entry['audio_file']}", end="")
        if entry["status"] == "ok":
            print(f"  latency_ms={entry['latency_ms']}  request_id={entry['request_id']}")
        else:
            print(f"  error={entry['error']}")
    print()
    print("transcriptions (verbatim provider output):")
    for index in range(1, manifest["total_segments"] + 1):
        record_path = stage_dir / f"segment-{index:02d}.json"
        if not record_path.is_file():
            continue
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("status") != "ok":
            continue
        print(f"  {record['section_id']} ({record['audio_file']}, {record['target_language']}):")
        print(f"    {record['transcription_text']}")


async def _run() -> int:
    settings = load_settings()
    if not settings.bodhan_transcribe_key:
        print("BODHAN_TRANSCRIBE_KEY is not configured in .env - aborting.", file=sys.stderr)
        return 2
    if not SOURCE_NARRATE.is_dir():
        print(f"missing Phase 1 narrate output: {SOURCE_NARRATE}", file=sys.stderr)
        return 2

    store = FileArtifactStore(settings.work_dir / TARGET_WORKSPACE.name)
    store.root()
    _setup_workspace(store)

    chapter = Chapter(
        chapter_id=TARGET_WORKSPACE.name,
        title="Phase 2A live validation (Hindi)",
        source_path=str(SOURCE_NARRATE),
        language_code="en",
        target_language_code="hi",
    )
    providers = build_providers(settings, target_language=chapter.target_language_code)
    ctx = StageContext(
        settings=settings,
        store=store,
        chapter=chapter,
        ocr=providers.ocr,
        llm=providers.llm,
        translate=providers.translate,
        tts=providers.tts,
        transcribe=providers.transcribe,
        logger=get_logger("phase2a"),
    )

    try:
        summary = await transcribe.run(ctx)
    except PipelineError as exc:
        # Results for every successful segment are already checkpointed on
        # disk; a re-run resumes at exactly the failed segment(s).
        print(f"transcribe stage failed: {exc}", file=sys.stderr)
        _report(ctx)
        return 1
    finally:
        with contextlib.suppress(Exception):
            await providers.aclose()

    print("stage summary: " + json.dumps(summary, ensure_ascii=False))
    rate_limit = getattr(providers.transcribe, "rate_limit", None)
    if rate_limit is not None:
        print(f"published rate limits (from response headers): {rate_limit.as_dict()}")
    _report(ctx)
    print(f"\nresults: {store.stage_dir(StageName.TRANSCRIBE.value)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run()))
