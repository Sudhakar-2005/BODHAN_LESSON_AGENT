"""Stage 7 - compose: author the pedagogical lesson from translated blocks.

The lesson is built strictly from the translated content (plus the source for
numbers/equations); flagged blocks are carried into the lesson metadata so the
export can publish an audit trail.
"""

from __future__ import annotations

from typing import Any

from bodhan.domain.models import Lesson, OCRBlock, StageName
from bodhan.pipeline.errors import PipelineError
from bodhan.pipeline.prompts import complete_json, format_payload
from bodhan.pipeline.stages.base import StageContext
from bodhan.pipeline.state import load_input_pages, load_json, save_json, save_pages

NAME = StageName.COMPOSE

_SYSTEM = (
    "You are an experienced teacher authoring a lesson from translated "
    "textbook content. Return STRICT JSON only:\n"
    '{"title":"...","markdown":"..."}\n'
    "The markdown must contain, in order:\n"
    "1. `# <title>`\n"
    "2. `## अध्याय के उद्देश्य` section with 3-5 bullet learning objectives\n"
    "3. `## पाठ` - the body, preserving the original section structure "
    "(use `###` for sub-sections)\n"
    "4. `## सारांश` - a concise summary\n"
    "5. `## मुख्य शब्दावली` - key terms as a definition list\n"
    "6. `## अभ्यास प्रश्न` - 4-6 questions with answers\n"
    "Rules:\n"
    "- Use ONLY facts present in the provided content; never invent material.\n"
    "- Keep every equation, number and unit exactly as given.\n"
    "- Write in the target language; keep technical terms in original script "
    "when a translation would be misleading.\n"
    "- No commentary, no placeholders, no markdown code fences."
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


async def run(ctx: StageContext) -> dict[str, Any]:
    """Compose the lesson markdown and persist the Lesson record."""
    pages = load_input_pages(ctx.store, NAME)
    save_pages(ctx.store, NAME, pages)
    blocks = _load_blocks(ctx)
    translations = _load_translations(ctx)
    if not blocks or not translations:
        raise PipelineError("missing blocks/translations - run translate first")

    content = [
        {
            "id": b.block_id,
            "kind": b.kind.value,
            "source_text": b.text,
            "text": translations.get(b.block_id, b.text),
        }
        for b in blocks
    ]
    payload = {
        "chapter_title": ctx.chapter.title,
        "target_language": ctx.chapter.target_language_code,
        "content": content,
    }
    raw = await complete_json(
        ctx.llm,
        system=_SYSTEM,
        user=f"Author the lesson from this content:\n{format_payload(payload)}",
        stage=NAME.value,
    )
    if not isinstance(raw, dict) or not str(raw.get("markdown", "")).strip():
        raise PipelineError("compose response missing 'markdown'")

    flagged = [b.block_id for b in blocks if b.flags]
    lesson = Lesson(
        lesson_id=f"{ctx.chapter.chapter_id}-lesson",
        chapter_id=ctx.chapter.chapter_id,
        title=str(raw.get("title") or ctx.chapter.title),
        markdown=str(raw["markdown"]).strip(),
        source_language_code=ctx.chapter.language_code,
        target_language_code=ctx.chapter.target_language_code,
        flagged_block_ids=flagged,
    )
    save_json(ctx.store, NAME, "lesson.json", lesson)
    ctx.state["lesson"] = lesson
    return {
        "lesson_id": lesson.lesson_id,
        "chars": len(lesson.markdown),
        "flagged_blocks": len(flagged),
    }
