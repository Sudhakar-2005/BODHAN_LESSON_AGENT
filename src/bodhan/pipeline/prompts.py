"""Shared prompt/response helpers for the LLM-backed stages.

The LLM is only ever asked to *annotate* OCR output (structure, quality,
translation, composition) -- never to produce source text that OCR did not
return.
"""

from __future__ import annotations

import json
import re
from typing import Any

from bodhan.pipeline.errors import PipelineError
from bodhan.ports.llm import LLMMessage, LLMPort

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def format_payload(payload: Any) -> str:
    """Serialise a prompt payload as strict JSON (never Python repr)."""
    return json.dumps(payload, ensure_ascii=False)


def parse_json_response(text: str) -> Any:
    """Parse a JSON payload from an LLM reply, tolerating code fences."""
    candidate = text.strip()
    fenced = _FENCE.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()
    if not candidate:
        raise PipelineError("LLM returned an empty response")
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    # Fall back to the outermost {...} or [...] span.
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = candidate.find(opener), candidate.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise PipelineError(f"LLM response is not valid JSON: {candidate[:200]}")


async def complete_json(
    llm: LLMPort,
    *,
    system: str,
    user: str,
    stage: str,
) -> Any:
    """Run one completion and parse the reply as JSON (errors name the stage)."""
    response = await llm.complete(
        [
            LLMMessage(role="user", content=system),
            LLMMessage(role="user", content=user),
        ]
    )
    try:
        return parse_json_response(response.text)
    except PipelineError as exc:
        raise PipelineError(f"[{stage}] {exc}") from exc


def chunked(items: list[Any], size: int) -> list[list[Any]]:
    """Split ``items`` into batches of at most ``size`` elements."""
    if size <= 0:
        raise ValueError("chunk size must be positive")
    return [items[i : i + size] for i in range(0, len(items), size)]
