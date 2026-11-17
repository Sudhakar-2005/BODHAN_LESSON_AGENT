"""Stage 6 - translate: Bodhan POST /translate (per block, ID preserved).

Rules:
- target language is validated against the documented enum before any call;
- block ids are preserved locally: the output map is keyed by block id and
  the source text is never mutated;
- equations and empty blocks pass through verbatim (no API spend);
- truncated/incomplete provider output raises (adapter rejects it) and the
  stage fails loudly - nothing partial reaches composition;
- token usage from every response is aggregated into the stage artifacts for
  cost reporting.
"""

from __future__ import annotations

from typing import Any

from bodhan.domain.models import BlockKind, OCRBlock, StageName
from bodhan.pipeline.errors import PipelineError
from bodhan.pipeline.stages.base import StageContext
from bodhan.pipeline.state import load_input_pages, load_json, save_json, save_pages
from bodhan.ports.translate import SUPPORTED_TARGET_LANGUAGES

NAME = StageName.TRANSLATE

#: Never sent to the translation API - copied through unchanged.
_VERBATIM_KINDS = {BlockKind.EQUATION}


def _load_blocks(ctx: StageContext) -> list[OCRBlock]:
    return [
        OCRBlock.model_validate(item)
        for item in load_json(ctx.store, StageName.QUALITY, "blocks.json")
    ]


async def run(ctx: StageContext) -> dict[str, Any]:
    """Translate blocks to the target language, preserving ids and kinds."""
    pages = load_input_pages(ctx.store, NAME)
    save_pages(ctx.store, NAME, pages)
    blocks = _load_blocks(ctx)
    if not blocks:
        raise PipelineError("no blocks found - run quality first")

    target = ctx.chapter.target_language_code
    if target not in SUPPORTED_TARGET_LANGUAGES:
        raise PipelineError(
            f"target language {target!r} is not in the documented enum "
            f"({len(SUPPORTED_TARGET_LANGUAGES)} codes)"
        )

    translations: dict[str, str] = {}
    usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    verbatim = 0
    empty = 0

    for block in blocks:
        if not block.text.strip():
            translations[block.block_id] = ""  # id kept for a complete map
            empty += 1
            continue
        if block.kind in _VERBATIM_KINDS:
            translations[block.block_id] = block.text
            verbatim += 1
            continue
        result = await ctx.translate.translate(block.text, target_language=target)
        translations[block.block_id] = result.text
        for key, value in result.usage.items():
            if key in usage_totals:
                usage_totals[key] += value
        ctx.logger.debug("translated %s -> %d chars", block.block_id, len(result.text))

    save_json(
        ctx.store,
        NAME,
        "translations.json",
        {
            "target_language": target,
            "items": translations,
            "usage": usage_totals,
            "verbatim_blocks": verbatim,
            "empty_blocks": empty,
        },
    )
    ctx.state["translations"] = translations
    return {
        "blocks": len(translations),
        "target_language": target,
        "verbatim_blocks": verbatim,
        "empty_blocks": empty,
        "usage": usage_totals,
    }
