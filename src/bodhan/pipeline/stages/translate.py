"""Stage 6 - translate: Bodhan POST /translate (per block, ID preserved).

Rules (Phase 3B-2):
- each block is translated independently and checkpointed under
  ``translate/blocks/<id>.json``; one failure cannot discard other blocks;
- a rerun reuses only records whose source hash, target language and protected
  placeholders still match - corrupted or missing records are retranslated;
- math placeholders are validated before restoration; a missing/duplicated/
  altered placeholder rejects the provider response (never silently repaired);
- table cells carry row/column provenance and are checkpointed with the block;
- terminal provider errors are recorded, not retried indefinitely (the
  adapter already handles bounded Retry-After retries);
- everything is offline-safe: no stage here chooses a provider.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

from bodhan.document import (
    TableGrid,
    protect_math,
    render_table_markdown,
    validate_placeholders,
)
from bodhan.domain.models import BlockKind, OCRBlock, StageName
from bodhan.observability import emit_event
from bodhan.pipeline.errors import PipelineError
from bodhan.pipeline.stages.base import StageContext
from bodhan.pipeline.state import load_input_pages, load_json, save_json, save_pages
from bodhan.ports.translate import SUPPORTED_TARGET_LANGUAGES
from bodhan.services.json_io import write_json_atomic

NAME = StageName.TRANSLATE

#: Never sent to the translation API - copied through unchanged.
_VERBATIM_KINDS = {BlockKind.EQUATION}
_PLACEHOLDER = re.compile(r"\ue000\d+\ue001")


def _safe_name(block_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", block_id)


def _source_hash(text: str, target: str) -> str:
    return hashlib.sha256(f"{target}\0{text}".encode()).hexdigest()


def _load_block_records(directory: Path) -> dict[str, dict[str, Any]]:
    """Load per-block checkpoints; unreadable records are treated as missing."""
    records: dict[str, dict[str, Any]] = {}
    if not directory.is_dir():
        return records
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("block_id"), str):
            records[str(payload["block_id"])] = payload
    return records


def _load_grids(ctx: StageContext) -> dict[str, TableGrid]:
    try:
        payload = load_json(ctx.store, StageName.STRUCTURE, "tables.json")
    except PipelineError:  # backward compatibility with pre-3B-1 artifacts
        return {}
    grids: dict[str, TableGrid] = {}
    if isinstance(payload, dict):
        for block_id, raw in payload.items():
            if isinstance(raw, dict):
                try:
                    grids[str(block_id)] = TableGrid.model_validate(raw)
                except ValueError:
                    continue  # malformed grid: fall back to plain block translation
    return grids


def _load_blocks(ctx: StageContext) -> list[OCRBlock]:
    return [
        OCRBlock.model_validate(item)
        for item in load_json(ctx.store, StageName.QUALITY, "blocks.json")
    ]


async def _translate_text(ctx: StageContext, text: str, target: str) -> dict[str, Any]:
    """Translate one string, returning usage and validated restored text."""
    protected, mapping, uncertain = protect_math(text)
    if uncertain:
        emit_event(
            ctx.logger,
            "math_delimiters_uncertain",
            run_id=ctx.chapter.chapter_id,
            stage=NAME.value,
            target_language=target,
            level=logging.WARNING,
            status="preserved",
        )
    placeholders = _PLACEHOLDER.findall(protected)
    residual = _PLACEHOLDER.sub("", protected).strip()
    if placeholders and not residual:
        return {"text": text, "usage": {}, "api_called": False}  # math-only
    result = await ctx.translate.translate(protected, target_language=target)
    if not hasattr(result, "text") or not isinstance(result.text, str):
        raise PipelineError("translation response missing text")
    restored, error = validate_placeholders(result.text, mapping)
    if error is not None:
        raise PipelineError(f"invalid translation response: {error}")
    return {
        "text": restored or "",
        "usage": dict(getattr(result, "usage", {}) or {}),
        "api_called": True,
    }


async def run(ctx: StageContext) -> dict[str, Any]:
    """Translate blocks independently, checkpointing every outcome."""
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

    directory = ctx.store.stage_dir(NAME.value) / "blocks"
    directory.mkdir(parents=True, exist_ok=True)
    prior = _load_block_records(directory)
    grids = _load_grids(ctx)

    translations: dict[str, str] = {}
    failed: list[dict[str, str]] = []
    table_records: list[dict[str, Any]] = []
    usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    verbatim = 0
    empty = 0
    reused = 0
    table_cells = 0

    def add_usage(usage: dict[str, Any]) -> None:
        for key, value in usage.items():
            if key in usage_totals and isinstance(value, int) and not isinstance(value, bool):
                usage_totals[key] += value

    for block in blocks:
        block_id = block.block_id
        source_hash = _source_hash(block.text, target)
        record = prior.get(block_id, {})
        if (
            record.get("status") == "ok"
            and record.get("source_hash") == source_hash
            and isinstance(record.get("translation"), str)
        ):
            translations[block_id] = str(record["translation"])
            reused += 1
            emit_event(
                ctx.logger,
                "checkpoint_reused",
                run_id=ctx.chapter.chapter_id,
                stage=NAME.value,
                target_language=target,
                block_id=block_id,
            )
            if record.get("table"):
                table_records.append(record["table"])
            continue

        outcome: dict[str, Any]
        try:
            if not block.text.strip():
                outcome = {"status": "ok", "translation": "", "usage": {}, "empty": True}
            elif block.kind in _VERBATIM_KINDS:
                outcome = {
                    "status": "ok",
                    "translation": block.text,
                    "usage": {},
                    "verbatim": True,
                }
            else:
                grid = grids.get(block_id) if block.kind is BlockKind.TABLE else None
                if grid is not None and grid.rows and grid.cells:
                    translated_cells: dict[tuple[int, int], str] = {}
                    cell_records: list[dict[str, Any]] = []
                    cell_errors: list[str] = []
                    for cell in grid.cells:
                        try:
                            cell_result: dict[str, Any] = (
                                await _translate_text(ctx, cell.text, target)
                                if cell.text.strip()
                                else {"text": "", "usage": {}, "api_called": False}
                            )
                            add_usage(cell_result["usage"])
                            translated_cells[(cell.row, cell.column)] = cell_result["text"]
                            cell_status, error, translated = "ok", None, cell_result["text"]
                            table_cells += 1
                        except Exception as exc:  # preserve sibling-cell successes
                            message = f"{type(exc).__name__}: {exc}"
                            cell_errors.append(f"r{cell.row}c{cell.column}: {message}")
                            cell_status, error, translated = "failed", message, ""
                        cell_records.append(
                            {
                                "row": cell.row,
                                "column": cell.column,
                                "source_text": cell.text,
                                "translated_text": translated,
                                "is_header": cell.is_header,
                                "source_block_id": cell.source_block_id,
                                "status": cell_status,
                                "error": error,
                            }
                        )
                    table_payload = {
                        "block_id": block_id,
                        "source_block_ids": grid.source_block_ids,
                        "rows": grid.rows,
                        "columns": grid.columns,
                        "incomplete": grid.incomplete,
                        "review_reason": grid.review_reason,
                        "cells": cell_records,
                        "failed_cells": cell_errors,
                    }
                    if cell_errors:
                        message = "table cell translation failed: " + "; ".join(cell_errors)
                        failed.append({"block_id": block_id, "error": message})
                        outcome = {
                            "status": "failed",
                            "error": message,
                            "translation": None,
                            "table": table_payload,
                        }
                    else:
                        outcome = {
                            "status": "ok",
                            "translation": render_table_markdown(grid, translated_cells),
                            "usage": {},
                            "table": table_payload,
                        }
                else:
                    text_result = await _translate_text(ctx, block.text, target)
                    add_usage(text_result["usage"])
                    outcome = {
                        "status": "ok",
                        "translation": text_result["text"],
                        "usage": text_result["usage"],
                    }
        except Exception as exc:  # isolate the failure to this block/table
            message = f"{type(exc).__name__}: {exc}"
            outcome = {"status": "failed", "error": message, "translation": None}
            failed.append({"block_id": block_id, "error": message})
            emit_event(
                ctx.logger,
                "block_translation_failed",
                run_id=ctx.chapter.chapter_id,
                stage=NAME.value,
                target_language=target,
                level=logging.WARNING,
                block_id=block_id,
                error=message[:200],
            )

        if outcome.get("empty"):
            empty += 1
        if outcome.get("verbatim"):
            verbatim += 1
        checkpoint = {
            "block_id": block_id,
            "target_language": target,
            "source_hash": source_hash,
            "status": outcome["status"],
            "translation": outcome.get("translation"),
            "error": outcome.get("error"),
            "usage": outcome.get("usage", {}),
            "provider": getattr(ctx.translate, "provider", "unknown"),
            "model": ctx.settings.translate_model,
        }
        if outcome.get("table"):
            checkpoint["table"] = outcome["table"]
            table_records.append(outcome["table"])
        write_json_atomic(directory / f"{_safe_name(block_id)}.json", checkpoint)

        if outcome["status"] == "ok":
            translations[block_id] = str(outcome["translation"])
            emit_event(
                ctx.logger,
                "block_translated",
                run_id=ctx.chapter.chapter_id,
                stage=NAME.value,
                target_language=target,
                block_id=block_id,
                chars=len(str(outcome["translation"])),
            )
        else:
            # A failed block must never leave a stale successful translation.
            translations.pop(block_id, None)

    if not translations and blocks:
        raise PipelineError(
            f"no block translated successfully ({len(failed)} failed); "
            "failed block ids are recorded under translate/blocks/"
        )

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
            "reused_blocks": reused,
            "failed_blocks": failed,
        },
    )
    save_json(
        ctx.store,
        NAME,
        "tables.json",
        {
            "target_language": target,
            "model": ctx.settings.translate_model,
            "tables": table_records,
        },
    )
    ctx.state["translations"] = translations
    return {
        "blocks": len(translations),
        "target_language": target,
        "verbatim_blocks": verbatim,
        "empty_blocks": empty,
        "reused_blocks": reused,
        "failed_blocks": failed,
        "table_cells_translated": table_cells,
        "usage": usage_totals,
    }
