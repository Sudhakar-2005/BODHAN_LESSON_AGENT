"""Pipeline orchestrator: ordered, checkpointed, resumable stage execution.

Stage order (BODHAN LESSON AGENT)::

    ingest -> preprocess -> extract -> structure -> quality
           -> translate -> compose -> narrate -> transcribe -> export

Checkpointing: each completed stage writes its name and timing into the
chapter manifest. A re-run skips stages already marked ``status == "done"``
unless ``force=True``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from bodhan.domain.models import StageName
from bodhan.observability import emit_event, get_logger
from bodhan.pipeline.checkpoint import (
    checkpoint_verification_level,
    stage_artifact_hashes,
    verify_stage_artifacts,
)
from bodhan.pipeline.errors import PipelineError, StageNotImplementedError
from bodhan.pipeline.stages import (
    compose,
    export,
    extract,
    ingest,
    narrate,
    preprocess,
    quality,
    structure,
    transcribe,
    translate,
)
from bodhan.pipeline.stages.base import StageContext

logger = get_logger(__name__)

StageFn = Callable[[StageContext], Awaitable[dict[str, Any]]]

#: Ordered (name, callable) pipeline definition.
STAGES: list[tuple[StageName, StageFn]] = [
    (ingest.NAME, ingest.run),
    (preprocess.NAME, preprocess.run),
    (extract.NAME, extract.run),
    (structure.NAME, structure.run),
    (quality.NAME, quality.run),
    (translate.NAME, translate.run),
    (compose.NAME, compose.run),
    (narrate.NAME, narrate.run),
    (transcribe.NAME, transcribe.run),
    (export.NAME, export.run),
]


def _is_done(manifest: dict[str, Any], stage: StageName) -> bool:
    entry = manifest.get("stages", {}).get(stage.value, {})
    return isinstance(entry, dict) and entry.get("status") == "done"


def _record(
    manifest: dict[str, Any],
    stage: StageName,
    *,
    status: str,
    duration_s: float,
    artifacts: dict[str, Any] | None = None,
    error: str | None = None,
    artifact_hashes: dict[str, str] | None = None,
) -> None:
    stages = manifest.setdefault("stages", {})
    entry: dict[str, Any] = {
        "status": status,
        "completed_at": datetime.now(UTC).isoformat(),
        "duration_s": round(duration_s, 3),
    }
    if artifacts:
        entry["artifacts"] = artifacts
    if error:
        entry["error"] = error
    if artifact_hashes:
        entry["artifact_hashes"] = artifact_hashes
    stages[stage.value] = entry


async def run_pipeline(
    ctx: StageContext,
    *,
    force: bool = False,
    only: StageName | None = None,
) -> dict[str, Any]:
    """Execute the pipeline over ``ctx.chapter``.

    Args:
        ctx: shared stage context (settings, store, providers, chapter).
        force: re-run stages even if the manifest marks them done.
        only: run a single stage (useful for development/debugging).

    Returns:
        The final manifest.

    Raises:
        StageNotImplementedError: when a stage has no implementation yet.
        PipelineError: on unexpected stage failure (manifest records it first).
    """
    manifest = ctx.store.read_manifest()
    manifest.setdefault("chapter_id", ctx.chapter.chapter_id)
    manifest.setdefault("stages", {})

    stages = [(only, dict(STAGES)[only])] if only else STAGES

    for name, fn in stages:
        if not force and _is_done(manifest, name):
            entry = manifest["stages"][name.value]
            recorded = entry.get("artifact_hashes") if isinstance(entry, dict) else None
            has_directory = (ctx.store.root() / name.value).is_dir()
            # Legacy manifests without any stage artifacts are trusted (the
            # original resume contract); new runs record hashes and are
            # validated, and an existing directory is always checked.
            valid, reasons = (
                verify_stage_artifacts(ctx.store.root(), name, legacy_ok=True)
                if has_directory or recorded
                else (True, [])
            )
            verification = (
                checkpoint_verification_level(
                    ctx.store.root(), name, recorded if isinstance(recorded, dict) else None
                )
                if has_directory or recorded
                else "legacy"  # no artifacts on disk at all: legacy, not invalid
            )
            if verification == "invalid":
                valid = False
                reasons.append("checkpoint not integrity-verified")
            if valid:
                emit_event(
                    ctx.logger,
                    "checkpoint_reused",
                    run_id=ctx.chapter.chapter_id,
                    stage=name.value,
                    target_language=ctx.chapter.target_language_code,
                    status="reused",
                    verification=verification,  # 'verified' or 'legacy'
                )
                continue
            emit_event(
                ctx.logger,
                "checkpoint_invalidated",
                run_id=ctx.chapter.chapter_id,
                stage=name.value,
                target_language=ctx.chapter.target_language_code,
                level=logging.WARNING,
                reasons=sorted(reasons),
            )

        emit_event(
            ctx.logger,
            "stage_started",
            run_id=ctx.chapter.chapter_id,
            stage=name.value,
            target_language=ctx.chapter.target_language_code,
        )
        started = time.perf_counter()
        try:
            artifacts = await fn(ctx)
        except NotImplementedError as exc:
            _record(manifest, name, status="not_implemented", duration_s=0.0)
            ctx.store.write_manifest(manifest)
            raise StageNotImplementedError(name) from exc
        except Exception as exc:
            duration = time.perf_counter() - started
            _record(manifest, name, status="failed", duration_s=duration, error=str(exc))
            ctx.store.write_manifest(manifest)
            emit_event(
                ctx.logger,
                "stage_failed",
                run_id=ctx.chapter.chapter_id,
                stage=name.value,
                target_language=ctx.chapter.target_language_code,
                level=logging.ERROR,
                status="failed",
                duration_s=round(duration, 3),
                error=str(exc)[:200],
            )
            raise PipelineError(f"stage '{name.value}' failed: {exc}") from exc

        duration = time.perf_counter() - started
        # A stage may explicitly report a non-success outcome (e.g. translate
        # reporting "partial"). Such a status must never become a reusable
        # "done" checkpoint - only "done" is skipped by _is_done.
        reported = artifacts.get("status") if isinstance(artifacts, dict) else None
        stage_status = reported if reported in {"partial", "failed"} else "done"
        _record(
            manifest,
            name,
            status=stage_status,
            duration_s=duration,
            artifacts=artifacts,
            artifact_hashes=stage_artifact_hashes(ctx.store.root(), name),
        )
        ctx.store.write_manifest(manifest)
        emit_event(
            ctx.logger,
            "stage_completed",
            run_id=ctx.chapter.chapter_id,
            stage=name.value,
            target_language=ctx.chapter.target_language_code,
            status=stage_status,
            duration_s=round(duration, 3),
        )
        if stage_status != "done":
            emit_event(
                ctx.logger,
                "pipeline_halted",
                run_id=ctx.chapter.chapter_id,
                stage=name.value,
                target_language=ctx.chapter.target_language_code,
                level=logging.WARNING,
                status=stage_status,
            )
            manifest["status"] = stage_status
            ctx.store.write_manifest(manifest)
            raise PipelineError(
                f"stage '{name.value}' reported status '{stage_status}'; "
                "downstream stages were not run and artifacts were preserved"
            )

    manifest["status"] = "done" if not only else manifest.get("status", "partial")
    ctx.store.write_manifest(manifest)
    return manifest
