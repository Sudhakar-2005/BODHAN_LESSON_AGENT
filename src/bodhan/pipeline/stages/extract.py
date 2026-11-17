"""Stage 3 - extract: OCR every page through Indic-OCR (exclusive engine).

Safe per-page caching:
- each page's response is cached under ``extract/page-NNNN.json`` together
  with the SHA-256 of the page image it was produced from;
- a cache entry is reused only when the image hash still matches, so edited
  or re-preprocessed pages are never served stale results;
- entries are written atomically (temp + replace) and validated on read -
  a corrupt/truncated cache file simply triggers a re-run, never a crash.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from bodhan.domain.models import OCRResult, StageName
from bodhan.pipeline.errors import PipelineError
from bodhan.pipeline.stages.base import StageContext
from bodhan.pipeline.state import atomic_write_text, load_input_pages, save_json, save_pages
from bodhan.ports.artifact_store import ArtifactStore

NAME = StageName.EXTRACT


def _cache_path(store: ArtifactStore, page_index: int) -> Path:
    return store.stage_dir(NAME.value) / f"page-{page_index + 1:04d}.json"


def _image_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_cached(cache: Path, digest: str) -> OCRResult | None:
    """Return a cache hit only when the file is valid and the image matches."""
    if not cache.is_file():
        return None
    try:
        envelope = json.loads(cache.read_text(encoding="utf-8"))
        if envelope.get("image_sha256") != digest:
            return None
        return OCRResult.model_validate(envelope["result"])
    except (json.JSONDecodeError, ValueError, KeyError, TypeError):
        return None  # corrupt/partial entry -> recompute


async def run(ctx: StageContext) -> dict[str, Any]:
    """Run Indic-OCR on every preprocessed page, caching each result."""
    pages = load_input_pages(ctx.store, NAME)
    save_pages(ctx.store, NAME, pages)
    results: list[OCRResult] = []
    hits = 0

    for page in pages:
        if not page.image_path:
            raise PipelineError(f"page {page.page_index} has no image (run ingest)")
        image = Path(page.image_path)
        digest = _image_digest(image)
        cache = _cache_path(ctx.store, page.page_index)

        result = _load_cached(cache, digest)
        if result is not None:
            hits += 1
            ctx.logger.info("page %04d: cache hit (%d blocks)", page.page_index, len(result.blocks))
        else:
            result = await ctx.ocr.extract(page)
            envelope = {"image_sha256": digest, "result": json.loads(result.model_dump_json())}
            atomic_write_text(cache, json.dumps(envelope, ensure_ascii=False, indent=2))
            ctx.logger.info(
                "page %04d: %d blocks (%d flagged) usage=%s",
                page.page_index,
                len(result.blocks),
                len(result.flagged_blocks),
                result.usage or "-",
            )
        results.append(result)

    save_json(ctx.store, NAME, "results.json", results)
    ctx.state["ocr_results"] = results

    total_blocks = sum(len(r.blocks) for r in results)
    flagged = sum(len(r.flagged_blocks) for r in results)
    usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for result in results:
        for key, value in result.usage.items():
            if key in usage_totals:
                usage_totals[key] += value
    rate_limits = next((r.rate_limit for r in reversed(results) if r.rate_limit), {})
    return {
        "pages": len(results),
        "blocks": total_blocks,
        "flagged_blocks": flagged,
        "cache_hits": hits,
        "provider": "indic-ocr",
        "usage": usage_totals,
        "rate_limit": rate_limits,
    }
