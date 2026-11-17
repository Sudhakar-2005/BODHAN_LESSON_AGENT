"""Stage 5 - quality: LLM evaluation of OCR/structure output.

Evaluates every block (prioritising already-flagged ones), adds QualityFlags
for garbled/unreadable text, and writes a per-chapter quality report so no
suspect output reaches translation silently.
"""

from __future__ import annotations

from typing import Any

from bodhan.domain.models import OCRBlock, QualityFlag, StageName
from bodhan.pipeline.errors import PipelineError
from bodhan.pipeline.prompts import chunked, complete_json, format_payload
from bodhan.pipeline.stages.base import StageContext
from bodhan.pipeline.state import load_input_pages, load_json, save_json, save_pages

NAME = StageName.QUALITY

_SYSTEM = (
    "You are an OCR quality auditor for textbook content. You receive blocks "
    "with stable ids and a confidence score where available. Return STRICT "
    "JSON only:\n"
    '{"results":[{"id":"...","flags":["low_confidence"|"suspected_garbled"|'
    '"reading_order_ambiguous"|"untranslatable"],"reason":"short explanation"}]}\n'
    "Rules:\n"
    "- Every input id must appear exactly once.\n"
    "- Use suspected_garbled for text that is visibly corrupted/nonsensical.\n"
    "- Use reading_order_ambiguous only when the block placement is clearly wrong.\n"
    "- Use untranslatable only for placeholder/empty/garbage content.\n"
    "- Do not add flags for clean text (empty flags list).\n"
    "- Never rewrite any text."
)

BLOCKS_PER_CALL = 40


def _load_blocks(ctx: StageContext) -> list[OCRBlock]:
    return [
        OCRBlock.model_validate(item)
        for item in load_json(ctx.store, StageName.STRUCTURE, "blocks.json")
    ]


def _validate(payload: Any, blocks: list[OCRBlock]) -> dict[str, tuple[list[QualityFlag], str]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise PipelineError("quality response missing 'results' list")
    expected = {b.block_id for b in blocks}
    out: dict[str, tuple[list[QualityFlag], str]] = {}
    for item in payload["results"]:
        if not isinstance(item, dict):
            continue
        block_id = str(item.get("id", ""))
        if block_id not in expected:
            raise PipelineError(f"quality response referenced unknown id {block_id!r}")
        flags: list[QualityFlag] = []
        for raw_flag in item.get("flags", []) or []:
            try:
                flag = QualityFlag(str(raw_flag))
            except ValueError:
                continue  # ignore unknown flags rather than failing the run
            if flag not in flags:
                flags.append(flag)
        out[block_id] = (flags, str(item.get("reason", "")))
    missing = expected - set(out)
    if missing:
        raise PipelineError(f"quality response omitted ids: {sorted(missing)[:5]}")
    return out


async def run(ctx: StageContext) -> dict[str, Any]:
    """Evaluate blocks, apply flags, and persist the quality report."""
    pages = load_input_pages(ctx.store, NAME)
    save_pages(ctx.store, NAME, pages)
    blocks = _load_blocks(ctx)
    if not blocks:
        raise PipelineError("no blocks found - run structure first")

    evaluations: dict[str, tuple[list[QualityFlag], str]] = {}
    for batch in chunked(blocks, BLOCKS_PER_CALL):
        payload = {
            "blocks": [
                {
                    "id": b.block_id,
                    "page_index": b.page_index,
                    "kind": b.kind.value,
                    "confidence": b.confidence,
                    "flags": [f.value for f in b.flags],
                    "text": b.text,
                }
                for b in batch
            ]
        }
        raw = await complete_json(
            ctx.llm,
            system=_SYSTEM,
            user=f"Audit these OCR blocks:\n{format_payload(payload)}",
            stage=NAME.value,
        )
        evaluations.update(_validate(raw, batch))

    audited: list[OCRBlock] = []
    report: list[dict[str, Any]] = []
    for block in blocks:
        flags, reason = evaluations[block.block_id]
        updated = list(block.flags)
        for flag in flags:
            if flag not in updated:
                updated.append(flag)
        audited.append(block.model_copy(update={"flags": updated}))
        if updated:
            report.append(
                {
                    "block_id": block.block_id,
                    "page_index": block.page_index,
                    "confidence": block.confidence,
                    "flags": [f.value for f in updated],
                    "reason": reason,
                    "text": block.text[:160],
                }
            )

    save_json(ctx.store, NAME, "blocks.json", audited)
    save_json(
        ctx.store,
        NAME,
        "quality_report.json",
        {"flagged": len(report), "entries": report},
    )
    ctx.state["blocks"] = audited
    return {"blocks": len(audited), "flagged": len(report)}
