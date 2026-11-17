"""Stage 4 - structure: deterministic labels first, Gemini only if ambiguous.

Classification and reading order come from the OCR provider itself (the
37-class ``label`` mapping and the normalised ``order``), so most runs need
no LLM call at all. Blocks whose label maps to ``UNKNOWN`` are the ambiguous
cases handed to Gemini; the model only *labels* text that Indic-OCR returned
- it never rewrites, drops or invents blocks.
"""

from __future__ import annotations

from typing import Any

from bodhan.domain.models import BlockKind, OCRBlock, OCRResult, StageName
from bodhan.pipeline.errors import PipelineError
from bodhan.pipeline.prompts import chunked, complete_json, format_payload
from bodhan.pipeline.stages.base import StageContext
from bodhan.pipeline.state import load_input_pages, load_json, save_json, save_pages

NAME = StageName.STRUCTURE

_SYSTEM = (
    "You are a document structure analyst. You receive OCR blocks from a "
    "textbook page that could not be classified automatically, each with a "
    "stable id, its page index and its raw text. Return STRICT JSON only, "
    "with this shape:\n"
    '{"blocks":[{"id":"...","kind":"heading|paragraph|list_item|table|'
    'equation|figure|figure_caption|page_header|page_footer|unknown",'
    '"reading_order":0}]}\n'
    "Rules:\n"
    "- Every input id must appear exactly once; do not add, drop or merge ids.\n"
    "- reading_order restarts at 0 for each page and must follow natural "
    "reading order (columns left-to-right, top-to-bottom).\n"
    "- Never rewrite, translate or complete any text."
)

BLOCKS_PER_CALL = 40


def _load_blocks(ctx: StageContext) -> list[OCRBlock]:
    results = [
        OCRResult.model_validate(item)
        for item in load_json(ctx.store, StageName.EXTRACT, "results.json")
    ]
    return [block for result in results for block in result.blocks]


def _validate(payload: Any, blocks: list[OCRBlock]) -> dict[str, BlockKind]:
    if not isinstance(payload, dict) or not isinstance(payload.get("blocks"), list):
        raise PipelineError("structure response missing 'blocks' list")
    expected = {b.block_id for b in blocks}
    seen: dict[str, BlockKind] = {}
    for item in payload["blocks"]:
        if not isinstance(item, dict):
            continue
        block_id = str(item.get("id", ""))
        if block_id not in expected:
            raise PipelineError(f"structure response referenced unknown id {block_id!r}")
        if block_id in seen:
            raise PipelineError(f"structure response duplicated id {block_id!r}")
        try:
            kind = BlockKind(str(item.get("kind", "unknown")).lower())
        except ValueError:
            kind = BlockKind.UNKNOWN
        seen[block_id] = kind
    missing = expected - set(seen)
    if missing:
        raise PipelineError(f"structure response omitted ids: {sorted(missing)[:5]}")
    return seen


async def run(ctx: StageContext) -> dict[str, Any]:
    """Classify ambiguous blocks with the LLM; keep deterministic kinds."""
    pages = load_input_pages(ctx.store, NAME)
    save_pages(ctx.store, NAME, pages)
    blocks = _load_blocks(ctx)
    if not blocks:
        raise PipelineError("no OCR blocks found - run extract first")

    ambiguous = [b for b in blocks if b.kind is BlockKind.UNKNOWN]
    updates: dict[str, BlockKind] = {}

    if ambiguous:
        for batch in chunked(ambiguous, BLOCKS_PER_CALL):
            payload = {
                "blocks": [
                    {
                        "id": b.block_id,
                        "page_index": b.page_index,
                        "text": b.text,
                    }
                    for b in batch
                ]
            }
            raw = await complete_json(
                ctx.llm,
                system=_SYSTEM,
                user=f"Classify these unclassified OCR blocks:\n{format_payload(payload)}",
                stage=NAME.value,
            )
            updates.update(_validate(raw, batch))
    else:
        ctx.logger.info("structure: all %d blocks classified deterministically", len(blocks))

    structured: list[OCRBlock] = [
        block.model_copy(update={"kind": updates.get(block.block_id, block.kind)})
        for block in blocks
    ]
    # Reading order was normalised per page by the OCR stage; keep it stable.
    structured.sort(key=lambda b: (b.page_index, b.reading_order))

    save_json(ctx.store, NAME, "blocks.json", structured)
    ctx.state["blocks"] = structured

    by_kind: dict[str, int] = {}
    for block in structured:
        by_kind[block.kind.value] = by_kind.get(block.kind.value, 0) + 1
    return {
        "blocks": len(structured),
        "llm_classified": len(updates),
        "by_kind": by_kind,
    }
