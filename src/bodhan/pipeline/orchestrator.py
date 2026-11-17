"""Pipeline orchestrator: ordered, checkpointed, resumable stage execution.

Stage order (BODHAN LESSON AGENT)::

    ingest -> preprocess -> extract -> structure -> quality
           -> translate -> compose -> narrate -> export

Checkpointing: each completed stage writes its name and timing into the
chapter manifest. A re-run skips stages already marked ``status == "done"``
unless ``force=True``.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from bodhan.domain.models import StageName
from bodhan.observability import get_logger
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
            ctx.logger.info("stage %-10s skipped (already done)", name.value)
            continue

        ctx.logger.info("stage %-10s started", name.value)
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
            ctx.logger.exception("stage %s failed after %.2fs", name.value, duration)
            raise PipelineError(f"stage '{name.value}' failed: {exc}") from exc

        duration = time.perf_counter() - started
        _record(manifest, name, status="done", duration_s=duration, artifacts=artifacts)
        ctx.store.write_manifest(manifest)
        ctx.logger.info("stage %-10s done in %.2fs", name.value, duration)

    manifest["status"] = "done" if not only else manifest.get("status", "partial")
    ctx.store.write_manifest(manifest)
    return manifest
