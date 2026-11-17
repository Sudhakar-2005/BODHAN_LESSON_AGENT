"""BODHAN LESSON AGENT CLI (v1 entry point).

Examples::

    uv run bodhan stages
    uv run bodhan run data/inbox/chapter-01.pdf
    uv run bodhan run data/inbox/chapter-01.pdf --only extract --force
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import Any

import typer

from bodhan.config import load_settings
from bodhan.domain.models import Chapter, StageName
from bodhan.observability import configure_logging, get_logger
from bodhan.pipeline import StageNotImplementedError, run_pipeline
from bodhan.pipeline.orchestrator import STAGES
from bodhan.pipeline.stages.base import StageContext
from bodhan.services.artifact_store import FileArtifactStore
from bodhan.wiring import Providers, build_providers

app = typer.Typer(
    no_args_is_help=True,
    help="Convert scanned textbook chapters into translated, narrated lessons.",
)
logger = get_logger(__name__)


def _validate_chapter_id(chapter_id: str) -> str:
    """Reject ids that would escape the work directory or collide with files."""
    if not chapter_id or chapter_id in {".", ".."} or Path(chapter_id).name != chapter_id:
        typer.echo(f"Invalid chapter id: {chapter_id!r}", err=True)
        raise typer.Exit(code=2)
    return chapter_id


def _build_context(
    source: Path, chapter_id: str | None, language: str | None, target: str | None
) -> tuple[StageContext, Providers]:
    settings = load_settings()
    resolved_id = _validate_chapter_id(chapter_id or source.stem)
    chapter = Chapter(
        chapter_id=resolved_id,
        title=source.stem,
        source_path=str(source.resolve()),
        language_code=language or "unknown",
        target_language_code=target or settings.default_target_language,
    )
    store = FileArtifactStore(settings.work_dir / resolved_id)
    store.root()  # fail fast if the workspace path is unusable
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
        logger=get_logger("bodhan.pipeline"),
    )
    return ctx, providers


@app.command("stages")
def stages_command() -> None:
    """List pipeline stages in execution order."""
    for index, (name, _) in enumerate(STAGES, start=1):
        typer.echo(f"{index:>2}. {name.value}")


@app.command("run")
def run_command(
    source: Path = typer.Argument(..., exists=True, readable=True, dir_okay=False),
    chapter_id: str | None = typer.Option(
        None, help="Override the chapter id (default: file stem)."
    ),
    language: str | None = typer.Option(None, help="Source language tag sent to Indic-OCR."),
    target: str | None = typer.Option(
        None, "--target-lang", help="Target language for translation (default: from settings)."
    ),
    only: StageName | None = typer.Option(None, help="Run a single stage."),
    force: bool = typer.Option(False, help="Re-run stages already marked done."),
) -> None:
    """Run the full pipeline over a scanned chapter."""
    configure_logging()
    try:
        ctx, providers = _build_context(source, chapter_id, language, target)
    except (NotADirectoryError, ValueError) as exc:
        typer.echo(f"Cannot use this source: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    logger.info("chapter=%s source=%s work=%s", ctx.chapter.chapter_id, source, ctx.store.root())

    async def _run_and_close() -> dict[str, Any]:
        # Pipeline and provider shutdown must share one event loop: the
        # httpx pools are bound to the loop that opened them.
        try:
            return await run_pipeline(ctx, force=force, only=only)
        finally:
            with contextlib.suppress(Exception):
                await providers.aclose()

    try:
        manifest = asyncio.run(_run_and_close())
    except StageNotImplementedError as exc:
        typer.echo(f"Stopped: {exc}", err=True)
        raise typer.Exit(code=3) from exc
    except Exception as exc:  # pipeline already logged the traceback
        typer.echo(f"Pipeline failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Done. Manifest: {ctx.store.manifest_path}")
    summary = ", ".join(
        "{}={}".format(name, entry.get("status"))
        for name, entry in manifest.get("stages", {}).items()
        if isinstance(entry, dict)
    )
    typer.echo(f"Stages: {summary or '(none)'}")


if __name__ == "__main__":
    app()
