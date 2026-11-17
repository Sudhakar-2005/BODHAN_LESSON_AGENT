"""Assignment CLI: ``python -m bodhan_lesson``.

Required interface (assignment)::

    python -m bodhan_lesson --input path --langs hi,ta,bn --out ./out
    python -m bodhan_lesson ... [--source-lang en] [--glossary glossary.yaml] \\
        [--voices voices.yaml] [--code-mixed]

``--source-lang`` is OPTIONAL and defaults to ``en`` (the assignment's
example source language). Passing it explicitly remains backward compatible.

Behaviour:
- one full pipeline run per target language (each language gets its own
  work/ and exports/ directory, keyed ``<stem>-<lang>``), then the finished
  export is copied into ``<out>/<lang>/``;
- ``--input`` accepts a PDF, a single image, or a folder of page images
  (validated by the ingest stage before any API call);
- ``--glossary``: YAML file of English terms the lesson must cover - handed
  to the lesson planner and translated like every other scaffold unit;
- ``--voices``: YAML per-language Indic-Speak voice overrides (Indic-Speak
  is the only narration engine);
- ``--code-mixed``: sends the documented ``script: "codemix"`` field to
  Indic-Translate;
- ``--source-lang``: optional (default ``en``), recorded as the chapter/lesson
  source language (Indic-OCR and Indic-Translate take no source-language
  field per docs); invalid tags are rejected before any API call.

The legacy ``bodhan run`` CLI is untouched.

Page-count policy: the assignment target is 5-30 pages. This CLI ENFORCES
that range by default (ingest fails before any provider call for out-of-range
documents). For internal synthetic fixtures only, set the environment variable
``BODHAN_ALLOW_SHORT_DOCUMENT=1`` to relax the lower bound - this override is
intentionally not a public CLI flag and is never needed for real documents.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import shutil
from pathlib import Path
from typing import NoReturn

import typer
import yaml

from bodhan.config import Settings, load_settings
from bodhan.domain.models import Chapter
from bodhan.observability import configure_logging, get_logger
from bodhan.pipeline import StageNotImplementedError, run_pipeline
from bodhan.pipeline.errors import PipelineError
from bodhan.pipeline.stages.base import StageContext
from bodhan.pipeline.stages.compose import contains_indian_script
from bodhan.ports.translate import SUPPORTED_TARGET_LANGUAGES
from bodhan.services.artifact_store import FileArtifactStore
from bodhan.wiring import build_providers

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Convert scanned chapters into translated, narrated lessons (one per target language).",
)
logger = get_logger(__name__)


def _fail(message: str) -> NoReturn:
    """Report a validation problem and exit with the usage-error code."""
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(code=2)


def _parse_langs(raw: str) -> list[str]:
    """Parse ``--langs hi,ta,bn`` against the documented target enum."""
    langs = list(dict.fromkeys(part.strip().lower() for part in raw.split(",") if part.strip()))
    if not langs:
        _fail("--langs needs at least one target language (e.g. --langs hi,ta,bn)")
    unsupported = [lang for lang in langs if lang not in SUPPORTED_TARGET_LANGUAGES]
    if unsupported:
        supported = ", ".join(sorted(SUPPORTED_TARGET_LANGUAGES))
        _fail(f"unsupported target language(s): {', '.join(unsupported)}\nSupported: {supported}")
    return langs


#: Permissive BCP-47-style language tag (en, en-US, hi, bn-IN, ...). The
#: assignment passes no source-language field to Bodhan APIs, so this is a
#: metadata tag; validating its *shape* rejects typos such as "english" or
#: "!" before any provider call.
_SOURCE_LANG_RE = re.compile(r"^[A-Za-z]{2,3}(?:[-_][A-Za-z0-9]{2,8})*$")


def _validate_source_lang(value: str) -> str:
    """Normalise and validate ``--source-lang`` (default ``en``)."""
    language = value.strip()
    if not _SOURCE_LANG_RE.fullmatch(language):
        _fail(f"invalid --source-lang {value!r}: expected a language tag such as 'en' or 'en-US'")
    return language


def _read_yaml(path: Path, label: str) -> object:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        _fail(f"cannot read {label} file {path}: {exc}")
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        _fail(f"{label} file {path} is not valid YAML: {exc}")


def _load_glossary(path: Path) -> list[dict[str, str]]:
    """Load English glossary seeds from YAML.

    Accepted shapes::

        terms: [{term: ..., description: ...}, ...]
        [{term: ..., description: ...}, ...]
        {term: description, ...}
    """
    data = _read_yaml(path, "glossary")
    if data is None:
        _fail(f"glossary file {path} is empty")
    if isinstance(data, dict) and "terms" in data:
        items = data["terms"]
        if not isinstance(items, list):
            _fail("glossary 'terms' must be a list")
    elif isinstance(data, dict):
        items = [{"term": term, "description": desc} for term, desc in data.items()]
    elif isinstance(data, list):
        items = data
    else:
        _fail("glossary must be a mapping of terms or a list of entries")

    seeds: list[dict[str, str]] = []
    for index, item in enumerate(items, start=1):
        if isinstance(item, str):
            term, description = item, ""
        elif isinstance(item, dict) and isinstance(item.get("term"), str):
            term = item["term"]
            raw_desc = item.get("description")
            description = raw_desc if isinstance(raw_desc, str) else ""
        else:
            _fail(f"glossary entry #{index} must be a term or a mapping with 'term'")
        term, description = term.strip(), description.strip()
        if not term:
            _fail(f"glossary entry #{index} has an empty term")
        if contains_indian_script(term) or contains_indian_script(description):
            _fail(
                f"glossary entry {term!r} contains Indian-script text; write the "
                "English terms (they are translated per target language)"
            )
        seeds.append({"term": term, "description": description})
    return seeds


def _load_voices(path: Path) -> dict[str, tuple[str, str | None]]:
    """Load per-language Indic-Speak voice overrides from YAML.

    Accepted shapes::

        voices: {hi: Kavya, ta: {voice: Pallavi, style: cheerful}}
        {hi: Kavya, ...}
    """
    data = _read_yaml(path, "voices")
    if isinstance(data, dict) and isinstance(data.get("voices"), dict):
        raw_map = data["voices"]
    elif isinstance(data, dict):
        raw_map = data
    else:
        _fail("voices file must be a mapping of language -> voice")
    if not raw_map:
        _fail(f"voices file {path} contains no voices")

    voices: dict[str, tuple[str, str | None]] = {}
    for raw_lang, spec in raw_map.items():
        lang = str(raw_lang).strip().lower()
        if lang not in SUPPORTED_TARGET_LANGUAGES:
            _fail(f"voices file: unknown target language {raw_lang!r}")
        if isinstance(spec, str):
            voice, style = spec.strip(), None
        elif isinstance(spec, dict) and isinstance(spec.get("voice"), str):
            voice = spec["voice"].strip()
            raw_style = spec.get("style")
            style = raw_style.strip() if isinstance(raw_style, str) else None
        else:
            _fail(f"voices file: entry for {lang!r} must be a name or a mapping with 'voice'")
        if not voice:
            _fail(f"voices file: entry for {lang!r} has an empty voice name")
        voices[lang] = (voice, style)
    return voices


def _chapter_id(source: Path, lang: str) -> str:
    """Filesystem-safe chapter id, unique per target language."""
    stem = (source.stem if source.is_file() else source.name) or "chapter"
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in stem)
    safe = safe.strip("._") or "chapter"
    return f"{safe}-{lang}"


def _run_language(
    source: Path,
    lang: str,
    *,
    settings: Settings,
    source_lang: str,
    glossary_seeds: list[dict[str, str]],
    out_root: Path,
) -> Path:
    """Run the full pipeline for one target language; return the copied lesson."""
    resolved_id = _chapter_id(source, lang)
    chapter = Chapter(
        chapter_id=resolved_id,
        title=source.stem or source.name,
        source_path=str(source.resolve()),
        language_code=source_lang,
        target_language_code=lang,
        glossary_seeds=glossary_seeds,
    )
    store = FileArtifactStore(settings.work_dir / resolved_id)
    store.root()  # fail fast if the workspace path is unusable
    providers = build_providers(settings, target_language=lang)
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

    async def _run_and_close() -> dict[str, object]:
        # Pipeline and provider shutdown must share one event loop: the
        # httpx pools are bound to the loop that opened them.
        try:
            return await run_pipeline(ctx)
        finally:
            with contextlib.suppress(Exception):
                await providers.aclose()

    manifest = asyncio.run(_run_and_close())
    export_dir = settings.exports_dir / resolved_id
    if not export_dir.is_dir():
        raise PipelineError(f"export directory missing after run: {export_dir}")

    target_dir = out_root / lang
    shutil.copytree(export_dir, target_dir, dirs_exist_ok=True)
    logger.info("lang=%s status=%s out=%s", lang, manifest.get("status"), target_dir)
    return target_dir / "lesson.md"


@app.command()
def run_lesson(
    input_path: Path = typer.Option(
        ...,
        "--input",
        exists=True,
        readable=True,
        help="Scanned chapter: a PDF, an image, or a folder of page images.",
    ),
    langs: str = typer.Option(
        ...,
        "--langs",
        help="Comma-separated target languages, e.g. --langs hi,ta,bn.",
    ),
    out: Path = typer.Option(
        ...,
        "--out",
        help="Directory to copy each finished lesson into (one folder per language).",
    ),
    source_lang: str = typer.Option(
        "en",
        "--source-lang",
        help="Source language tag of the chapter (optional, default: en).",
    ),
    glossary: Path | None = typer.Option(
        None,
        "--glossary",
        help="YAML file of English terms the lesson must cover.",
    ),
    voices: Path | None = typer.Option(
        None,
        "--voices",
        help="YAML file of per-language Indic-Speak voice overrides.",
    ),
    code_mixed: bool = typer.Option(
        False,
        "--code-mixed",
        help="Send the documented script=codemix field to Indic-Translate.",
    ),
) -> None:
    """Run the pipeline once per target language and collect the lessons in --out."""
    configure_logging()
    # All language validation happens before provider construction/API calls.
    validated_source_lang = _validate_source_lang(source_lang)
    target_langs = _parse_langs(langs)

    # Optional files are validated up front, before any pipeline/API work.
    if glossary is not None and not glossary.is_file():
        _fail(f"--glossary path does not exist: {glossary}")
    if voices is not None and not voices.is_file():
        _fail(f"--voices path does not exist: {voices}")
    glossary_seeds = _load_glossary(glossary) if glossary is not None else []
    voice_map = _load_voices(voices) if voices is not None else {}

    settings = load_settings()
    # Assignment-facing default: enforce the 5-30 page range. Internal tests
    # using tiny synthetic PDFs set BODHAN_ALLOW_SHORT_DOCUMENT=1 explicitly.
    enforce_pages = os.environ.get("BODHAN_ALLOW_SHORT_DOCUMENT", "") != "1"
    settings = settings.model_copy(update={"enforce_page_count_limits": enforce_pages})
    if code_mixed:
        # Documented Indic-Translate option: `script: "codemix"`.
        settings = settings.model_copy(update={"translate_script": "codemix"})

    out_root = out
    out_root.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    for lang in target_langs:
        lang_settings = settings
        override = voice_map.get(lang)
        if override is not None:
            voice, style = override
            update: dict[str, object] = {"bodhan_speak_voice": voice}
            if style is not None:
                update["bodhan_speak_style"] = style
            lang_settings = settings.model_copy(update=update)

        typer.echo(f"[{lang}] running pipeline for {input_path.name} ...")
        try:
            lesson_path = _run_language(
                input_path,
                lang,
                settings=lang_settings,
                source_lang=validated_source_lang,
                glossary_seeds=glossary_seeds,
                out_root=out_root,
            )
        except StageNotImplementedError as exc:
            typer.echo(f"[{lang}] stopped: {exc}", err=True)
            failures.append(lang)
        except Exception as exc:
            typer.echo(f"[{lang}] failed: {exc}", err=True)
            failures.append(lang)
        else:
            typer.echo(f"[{lang}] done: {lesson_path}")

    if failures:
        typer.echo(f"Failed language(s): {', '.join(failures)}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
