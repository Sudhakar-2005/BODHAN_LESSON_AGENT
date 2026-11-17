"""Stage 7 - compose: structure the lesson from translated blocks.

The LLM is a *lesson planner*, not an author: it may order sections, group
blocks by id and draft pedagogical scaffolding **in English only**. It is
never asked - and not permitted - to write Hindi, Tamil, Bengali or any
other Indian-language text (assignment Requirement 15).

Every target-language string in the final lesson therefore comes from
Bodhan Indic-Translate:

- body text arrives from the translate stage (``translations.json``, keyed
  by block id; equations pass through verbatim);
- labels, the title, objectives, summary, glossary terms, exercise
  questions and narration notes are English units translated here and
  recorded in ``lesson.json`` (``scaffold``) with their English source,
  provider and token usage - so any line of the lesson traces back to
  either a block id or a translation call.

The validated English plan is kept as ``compose/plan.json`` for audit.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from bodhan.domain.models import Lesson, OCRBlock, ScaffoldItem, StageName
from bodhan.pipeline.errors import PipelineError
from bodhan.pipeline.prompts import complete_json, format_payload
from bodhan.pipeline.stages.base import StageContext
from bodhan.pipeline.state import load_input_pages, load_json, save_json, save_pages
from bodhan.services.json_io import write_json_atomic

NAME = StageName.COMPOSE

#: Fixed UI labels. English *source* strings only - each is translated via
#: Indic-Translate for the run's target language. Output labels are never
#: hardcoded in any language (assignment Requirement 19).
LABELS_EN: dict[str, str] = {
    "audio": "Lesson audio",
    "exercises": "Exercises",
    "glossary": "Key terms",
    "introduction": "Introduction",
    "objectives": "Chapter objectives",
    "summary": "Summary",
}

_SYSTEM = (
    "You are a pedagogical lesson planner. You receive textbook OCR blocks "
    "(stable id, kind, source text, existing translation) and return a "
    "lesson plan as STRICT JSON only:\n"
    '{"title_en":"...","sections":[{"heading_en":"...","block_ids":["..."]}],'
    '"objectives_en":["..."],"summary_en":"...",'
    '"glossary_en":[{"term_en":"...","description_en":"...","block_id":"..."}],'
    '"exercises_en":[{"question_en":"...","answer_en":"...","block_ids":["..."]}],'
    '"narration_en":["..."]}\n'
    "Rules:\n"
    "- Write every text field in English only. Never write Hindi, Tamil, "
    "Bengali or any other Indian-language text - a separate translation "
    "service produces all non-English output.\n"
    "- Organise the given blocks into sections: every input block id must "
    "appear exactly once across sections; never add, drop or merge ids.\n"
    "- Never rewrite, translate or complete any source text; refer to "
    "content by block id.\n"
    "- objectives_en: 3-5 short learning objectives; summary_en: a concise "
    "summary of the content; glossary_en: terms that need explanation "
    "(block_id links to where the term appears); exercises_en: 4-6 questions "
    "with answers grounded in the content; narration_en: optional speaking "
    "notes for the narrator.\n"
    "- Use ONLY facts present in the provided content; never invent material.\n"
    "- No commentary, no placeholders, no markdown code fences."
)

#: Unicode ranges of the scripts used by Indian languages. Used to reject
#: any Indian-language text the LLM might produce anyway. Covers the nine
#: assignment-required scripts (Devanagari, Bengali, Gurmukhi, Gujarati,
#: Oriya, Tamil, Telugu, Kannada, Malayalam), plus Devanagari Extended
#: (U+A8E0-U+A8FF, Vedic/Vedic Extensions used in Sanskrit text) and the
#: Arabic-script ranges because Urdu (``ur``, ``sd-Arab``) is a supported
#: target language whose output must also come from Indic-Translate.
_INDIAN_SCRIPT_RANGES: tuple[tuple[int, int], ...] = (
    (0x0900, 0x097F),  # Devanagari
    (0x0980, 0x09FF),  # Bengali
    (0x0A00, 0x0A7F),  # Gurmukhi
    (0x0A80, 0x0AFF),  # Gujarati
    (0x0B00, 0x0B7F),  # Oriya
    (0x0B80, 0x0BFF),  # Tamil
    (0x0C00, 0x0C7F),  # Telugu
    (0x0C80, 0x0CFF),  # Kannada
    (0x0D00, 0x0D7F),  # Malayalam
    (0xA8E0, 0xA8FF),  # Devanagari Extended (combining marks, Vedic)
    # Arabic script (Urdu, Sindhi in Arabic script) - target languages.
    (0x0600, 0x06FF),  # Arabic
    (0x0750, 0x077F),  # Arabic Supplement
    (0x0870, 0x089F),  # Arabic Extended-B
    (0x08A0, 0x08FF),  # Arabic Extended-A
    (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
    (0xFE70, 0xFEFE),  # Arabic Presentation Forms-B (U+FEFF=BOM, excluded)
)


def contains_indian_script(text: str) -> bool:
    """True when ``text`` contains characters of an Indian Brahmic script."""
    for char in text:
        code = ord(char)
        if any(low <= code <= high for low, high in _INDIAN_SCRIPT_RANGES):
            return True
    return False


def _english(value: Any, field_name: str, *, allow_empty: bool = False) -> str:
    """Validate one LLM-authored field: an English-only string."""
    if not isinstance(value, str):
        raise PipelineError(f"compose plan field {field_name!r} must be a string")
    text = value.strip()
    if not text:
        if allow_empty:
            return ""
        raise PipelineError(f"compose plan field {field_name!r} must not be empty")
    if contains_indian_script(text):
        raise PipelineError(
            f"compose plan field {field_name!r} contains Indian-language text; "
            "the planner must write English only (target-language text comes "
            "from Indic-Translate)"
        )
    return text


def _block_ids(value: Any, known_ids: set[str], where: str) -> tuple[str, ...]:
    """Validate an optional list of known block ids."""
    if value is None:
        return ()
    if not isinstance(value, list):
        raise PipelineError(f"compose plan {where} must be a list of block ids")
    ids = [str(item) for item in value]
    for block_id in ids:
        if block_id not in known_ids:
            raise PipelineError(f"compose plan {where} referenced unknown id {block_id!r}")
    return tuple(ids)


def _string_list(value: Any, field_name: str) -> tuple[str, ...]:
    """Validate an optional list of non-empty English strings."""
    if value is None:
        return ()
    if not isinstance(value, list):
        raise PipelineError(f"compose plan field {field_name!r} must be a list")
    return tuple(_english(item, f"{field_name}[{index}]") for index, item in enumerate(value))


@dataclass(frozen=True)
class _Section:
    heading_en: str
    block_ids: tuple[str, ...]


@dataclass(frozen=True)
class _GlossaryEntry:
    term_en: str
    description_en: str
    block_id: str | None


@dataclass(frozen=True)
class _Exercise:
    question_en: str
    answer_en: str
    block_ids: tuple[str, ...]


@dataclass(frozen=True)
class _Plan:
    title_en: str
    sections: tuple[_Section, ...]
    objectives_en: tuple[str, ...] = ()
    summary_en: str = ""
    glossary_en: tuple[_GlossaryEntry, ...] = ()
    exercises_en: tuple[_Exercise, ...] = ()
    narration_en: tuple[str, ...] = ()


def _validate_plan(raw: Any, known_ids: set[str]) -> _Plan:
    """Parse the LLM plan, enforcing English-only text and full id coverage."""
    if not isinstance(raw, dict):
        raise PipelineError("compose response must be a JSON object")

    title_en = _english(raw.get("title_en"), "title_en")

    raw_sections = raw.get("sections")
    if not isinstance(raw_sections, list) or not raw_sections:
        raise PipelineError("compose plan needs a non-empty 'sections' list")
    sections: list[_Section] = []
    placed: list[str] = []
    for index, item in enumerate(raw_sections):
        if not isinstance(item, dict):
            raise PipelineError(f"compose plan section {index} must be an object")
        heading_en = _english(item.get("heading_en"), f"sections[{index}].heading_en")
        ids = item.get("block_ids")
        if not isinstance(ids, list) or not ids:
            raise PipelineError(f"compose plan section {index} needs block_ids")
        block_ids = [str(block_id) for block_id in ids]
        for block_id in block_ids:
            if block_id not in known_ids:
                raise PipelineError(f"compose plan referenced unknown id {block_id!r}")
        sections.append(_Section(heading_en=heading_en, block_ids=tuple(block_ids)))
        placed.extend(block_ids)

    duplicated = sorted({block_id for block_id in placed if placed.count(block_id) > 1})
    if duplicated:
        raise PipelineError(f"compose plan duplicated block id {duplicated[0]!r}")
    omitted = sorted(known_ids - set(placed))
    if omitted:
        raise PipelineError(f"compose plan omitted ids: {omitted[:5]}")

    glossary: list[_GlossaryEntry] = []
    raw_glossary = raw.get("glossary_en", [])
    if raw_glossary is None:
        raw_glossary = []
    if not isinstance(raw_glossary, list):
        raise PipelineError("compose plan field 'glossary_en' must be a list")
    for index, item in enumerate(raw_glossary):
        if not isinstance(item, dict):
            raise PipelineError(f"compose plan glossary_en[{index}] must be an object")
        where = f"glossary_en[{index}]"
        raw_related = item.get("block_id")
        related: str | None = None
        if raw_related not in (None, ""):
            related = str(raw_related)
            if related not in known_ids:
                raise PipelineError(f"compose plan {where} referenced unknown id {related!r}")
        glossary.append(
            _GlossaryEntry(
                term_en=_english(item.get("term_en"), f"{where}.term_en"),
                description_en=_english(item.get("description_en"), f"{where}.description_en"),
                block_id=related,
            )
        )

    exercises: list[_Exercise] = []
    raw_exercises = raw.get("exercises_en", [])
    if raw_exercises is None:
        raw_exercises = []
    if not isinstance(raw_exercises, list):
        raise PipelineError("compose plan field 'exercises_en' must be a list")
    for index, item in enumerate(raw_exercises):
        if not isinstance(item, dict):
            raise PipelineError(f"compose plan exercises_en[{index}] must be an object")
        where = f"exercises_en[{index}]"
        exercises.append(
            _Exercise(
                question_en=_english(item.get("question_en"), f"{where}.question_en"),
                answer_en=_english(item.get("answer_en"), f"{where}.answer_en"),
                block_ids=_block_ids(item.get("block_ids"), known_ids, where),
            )
        )

    return _Plan(
        title_en=title_en,
        sections=tuple(sections),
        objectives_en=_string_list(raw.get("objectives_en"), "objectives_en"),
        summary_en=_english(raw.get("summary_en", ""), "summary_en", allow_empty=True),
        glossary_en=tuple(glossary),
        exercises_en=tuple(exercises),
        narration_en=_string_list(raw.get("narration_en"), "narration_en"),
    )


def _load_blocks(ctx: StageContext) -> list[OCRBlock]:
    return [
        OCRBlock.model_validate(item)
        for item in load_json(ctx.store, StageName.QUALITY, "blocks.json")
    ]


def _load_translations(ctx: StageContext) -> dict[str, str]:
    payload = load_json(ctx.store, StageName.TRANSLATE, "translations.json")
    items = payload.get("items", {}) if isinstance(payload, dict) else {}
    return {str(k): str(v) for k, v in items.items()}


def _suppressed_ids(ctx: StageContext) -> set[str]:
    try:
        payload = load_json(ctx.store, StageName.STRUCTURE, "suppressed.json")
    except PipelineError:
        return set()
    if isinstance(payload, list):
        return {
            str(entry.get("block_id"))
            for entry in payload
            if isinstance(entry, dict) and isinstance(entry.get("block_id"), str)
        }
    return set()


def _section_records(
    ctx: StageContext,
    plan: Any,
    headings: list[str],
    translations: dict[str, str],
    suppressed_ids: set[str],
) -> list[dict[str, Any]]:
    """Persist and validate each section independently before assembly."""
    directory = ctx.store.stage_dir(NAME.value) / "sections"
    directory.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    seen: dict[str, int] = {}

    for index, section in enumerate(plan.sections, start=1):
        block_ids = [str(block_id) for block_id in section.block_ids]
        suppressed = sorted(set(block_ids) & suppressed_ids)
        if suppressed:
            raise PipelineError(
                f"compose section {index} references suppressed block(s): {suppressed}"
            )
        for block_id in block_ids:
            if block_id in seen:
                raise PipelineError(
                    f"compose block {block_id!r} appears in sections {seen[block_id]} and {index}"
                )
            seen[block_id] = index

        body = [
            translations[block_id].strip()
            for block_id in block_ids
            if translations.get(block_id, "").strip()
        ]
        heading = headings[index - 1]
        body_hash = hashlib.sha256(
            json.dumps([heading, body], ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        record: dict[str, Any] = {
            "section_number": index,
            "heading": heading,
            "source_heading": section.heading_en,
            "block_ids": block_ids,
            "body": body,
            "body_hash": body_hash,
            "status": "ok",
        }
        path = directory / f"section-{index:02d}.json"
        if path.is_file():
            try:
                prior = json.loads(path.read_text(encoding="utf-8"))
                if (
                    isinstance(prior, dict)
                    and prior.get("body_hash") == body_hash
                    and prior.get("source_heading") == section.heading_en
                ):
                    record = prior
            except (OSError, ValueError):
                pass  # corrupt checkpoint: rewrite atomically below
        write_json_atomic(path, record)
        records.append(record)
    return records


async def run(ctx: StageContext) -> dict[str, Any]:
    """Plan the lesson (LLM, English only), translate every unit, assemble."""
    pages = load_input_pages(ctx.store, NAME)
    save_pages(ctx.store, NAME, pages)
    blocks = _load_blocks(ctx)
    translations = _load_translations(ctx)
    if not blocks or not translations:
        raise PipelineError("missing blocks/translations - run translate first")
    missing = sorted(b.block_id for b in blocks if b.block_id not in translations)
    if missing:
        raise PipelineError(
            f"{len(missing)} block(s) have no translation - re-run translate: {missing[:5]}"
        )

    content = [
        {
            "id": block.block_id,
            "kind": block.kind.value,
            "source_text": block.text,
            "text": translations[block.block_id],
        }
        for block in blocks
    ]
    payload: dict[str, Any] = {"chapter_title": ctx.chapter.title, "content": content}
    # User-provided glossary seeds (--glossary): English terms the lesson
    # must cover; they are demanded of the planner and translated like every
    # other scaffold unit.
    seeds = [seed for seed in ctx.chapter.glossary_seeds if seed.get("term")]
    if seeds:
        payload["glossary_terms"] = seeds
    user = f"Plan the lesson from these blocks:\n{format_payload(payload)}"
    if seeds:
        user += (
            "\nInclude EVERY entry of 'glossary_terms' in glossary_en "
            "(term_en/description_en in English)."
        )
    raw = await complete_json(
        ctx.llm,
        system=_SYSTEM,
        user=user,
        stage=NAME.value,
    )
    plan = _validate_plan(raw, {block.block_id for block in blocks})
    # Audit trail: the exact (validated) English plan the LLM produced.
    save_json(ctx.store, NAME, "plan.json", raw)

    target = ctx.chapter.target_language_code
    scaffold: list[ScaffoldItem] = []

    async def translate_unit(
        item_id: str,
        kind: str,
        source_en: str,
        related: tuple[str, ...] = (),
    ) -> str:
        """Route one English unit through Indic-Translate, with provenance."""
        result = await ctx.translate.translate(source_en, target_language=target)
        scaffold.append(
            ScaffoldItem(
                item_id=item_id,
                kind=kind,
                source_en=source_en,
                text=result.text,
                provider=result.provider,
                usage=result.usage,
                related_block_ids=list(related),
            )
        )
        return result.text

    title = await translate_unit("title", "title", plan.title_en)
    labels: dict[str, str] = {}
    for key in sorted(LABELS_EN):
        labels[key] = await translate_unit(f"label:{key}", "label", LABELS_EN[key])

    headings: list[str] = []
    for index, section in enumerate(plan.sections):
        headings.append(
            await translate_unit(
                f"section:{index}",
                "section_heading",
                section.heading_en,
                section.block_ids,
            )
        )

    objectives = [
        await translate_unit(f"objective:{index}", "objective", text)
        for index, text in enumerate(plan.objectives_en)
    ]
    summary = await translate_unit("summary", "summary", plan.summary_en) if plan.summary_en else ""

    glossary: list[tuple[str, str]] = []
    for index, entry in enumerate(plan.glossary_en):
        related = (entry.block_id,) if entry.block_id else ()
        term = await translate_unit(
            f"glossary_term:{index}", "glossary_term", entry.term_en, related
        )
        description = await translate_unit(
            f"glossary_description:{index}",
            "glossary_description",
            entry.description_en,
            related,
        )
        glossary.append((term, description))

    exercises: list[tuple[str, str]] = []
    for index, exercise in enumerate(plan.exercises_en):
        question = await translate_unit(
            f"exercise_question:{index}",
            "exercise_question",
            exercise.question_en,
            exercise.block_ids,
        )
        answer = await translate_unit(
            f"exercise_answer:{index}",
            "exercise_answer",
            exercise.answer_en,
            exercise.block_ids,
        )
        exercises.append((question, answer))

    for index, note in enumerate(plan.narration_en):
        await translate_unit(f"narration:{index}", "narration", note)

    flagged = [block.block_id for block in blocks if block.flags]
    if flagged:
        labels["quality_warning"] = await translate_unit(
            "label:quality_warning",
            "label",
            f"{len(flagged)} OCR blocks were flagged during quality review.",
        )

    # --- Assemble the lesson: only translations and verbatim blocks --------
    section_records = _section_records(ctx, plan, headings, translations, _suppressed_ids(ctx))
    parts: list[str] = [f"# {title}"]
    if objectives:
        parts.append(f"## {labels['objectives']}\n" + "\n".join(f"- {text}" for text in objectives))
    for record in section_records:
        chunk = f"## {record['heading']}"
        if record["body"]:
            chunk += "\n\n" + "\n\n".join(record["body"])
        parts.append(chunk)
    if summary:
        parts.append(f"## {labels['summary']}\n{summary}")
    if glossary:
        parts.append(
            f"## {labels['glossary']}\n"
            + "\n".join(f"- {term}: {description}" for term, description in glossary)
        )
    if exercises:
        parts.append(
            f"## {labels['exercises']}\n"
            + "\n".join(f"- {question} — {answer}" for question, answer in exercises)
        )
    markdown = "\n\n".join(parts) + "\n"
    if not markdown.startswith("# ") or "\n## " not in markdown:
        raise PipelineError("compose produced invalid Markdown (missing title or sections)")
    if "\x00" in markdown:
        raise PipelineError("compose produced invalid Markdown (null byte)")
    write_json_atomic(
        ctx.store.stage_dir(NAME.value) / "assembly.json",
        {
            "status": "validated",
            "section_count": len(section_records),
            "sections": [
                {
                    "section_number": record["section_number"],
                    "block_ids": record["block_ids"],
                    "body_hash": record["body_hash"],
                    "status": record["status"],
                }
                for record in section_records
            ],
            "markdown_sha256": hashlib.sha256(markdown.encode("utf-8")).hexdigest(),
        },
    )

    lesson = Lesson(
        lesson_id=f"{ctx.chapter.chapter_id}-lesson",
        chapter_id=ctx.chapter.chapter_id,
        title=title,
        markdown=markdown,
        source_language_code=ctx.chapter.language_code,
        target_language_code=target,
        flagged_block_ids=flagged,
        scaffold=scaffold,
    )
    save_json(ctx.store, NAME, "lesson.json", lesson)
    ctx.state["lesson"] = lesson
    return {
        "lesson_id": lesson.lesson_id,
        "chars": len(lesson.markdown),
        "flagged_blocks": len(flagged),
        "sections": len(plan.sections),
        "section_artifacts": len(section_records),
        "scaffold_units": len(scaffold),
    }
