"""Bodhan Document OCR adapter (``model: indic-ocr``), official contract.

Request (application/json, documented fields only)::

    {"model": "indic-ocr",
     "messages": [{"role": "user", "content": [
         {"type": "image_url",
          "image_url": {"url": "data:image/png;base64,<b64>"}}]}],
     "table_format": "html" | "markdown",
     "max_tokens": 2048}

Notes honoured from the docs:
- PNG/JPEG only, one whole-page image per request, **no prompt
  customisation**, **no language field**;
- response is a standard ``chat.completion`` plus a top-level ``blocks``
  array; ``choices[0].message.content`` holds the reading-ordered Markdown;
- block object: ``order``, ``label`` (37-class taxonomy), ``type``,
  ``bbox_xyxy``, ``conf`` (layout-detection confidence, 0..1), ``text``;
- ``max_tokens`` is a **per-block** transcription limit; a block exceeding it
  fails the page with ``502`` -> bounded recovery by raising the budget once;
- published rate limits come from ``x-ratelimit-*`` response headers;
- ``401`` is terminal; ``408/425/429/5xx`` retry with bounded exponential
  backoff, honouring the documented ``retry-after`` header on ``429``
  (4 requests/minute sliding window).

Reading order is normalised locally: blocks are sorted by the provider's
``order`` (which the docs describe as 0-based but examples show starting at
1) and renumbered 0..n-1 per page - the raw value is never assumed.
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from bodhan.adapters.ocr.errors import OCRProviderError, OCRResponseError
from bodhan.adapters.ocr.labels import map_label
from bodhan.adapters.rate_limit import RateLimitInfo, read_rate_limit
from bodhan.adapters.retry import RetryAfterWait, read_retry_after
from bodhan.domain.models import OCRBlock, OCRResult, Page, QualityFlag
from bodhan.observability import get_logger
from bodhan.ports.ocr import OCRPort

logger = get_logger(__name__)

#: Transient statuses retried with the same payload.
_RETRYABLE_STATUS = {408, 425, 429, 500, 503, 504}
#: 502 = "block exceeded max_tokens OR image undecodable" (bounded recovery).
_RECOVERY_STATUS = 502
_ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg"}
_MODEL = "indic-ocr"


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
        return True
    if isinstance(exc, OCRProviderError):
        status = exc.status_code
        return status in _RETRYABLE_STATUS
    return False


class IndicOCRAdapter(OCRPort):
    """Async client for the Bodhan Document OCR endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout_s: float = 60.0,
        max_retries: int = 3,
        low_confidence_threshold: float = 0.85,
        table_format: str = "html",
        max_tokens: int = 2048,
        max_tokens_ceiling: int = 8192,
        max_502_recoveries: int = 1,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("base_url is required")
        if table_format not in {"html", "markdown"}:
            raise ValueError("table_format must be 'html' or 'markdown'")
        self._base_url = base_url
        self._api_key = api_key
        self._max_retries = max(1, max_retries)
        self._threshold = low_confidence_threshold
        self._table_format = table_format
        self._max_tokens = max_tokens
        self._max_tokens_ceiling = max(max_tokens, max_tokens_ceiling)
        self._max_502_recoveries = max(0, max_502_recoveries)
        self._rate_limit: RateLimitInfo | None = None
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(timeout_s))
        if api_key:  # auth applies to injected clients too
            self._client.headers["Authorization"] = f"Bearer {api_key}"

    @property
    def rate_limit(self) -> RateLimitInfo | None:
        """Latest published rate limits read from a response (no secrets)."""
        return self._rate_limit

    # -- OCRPort ------------------------------------------------------------
    async def extract(self, page: Page) -> OCRResult:
        image_path = Path(page.image_path) if page.image_path else Path(page.source_path)
        payload = self._build_payload(image_path)

        response = await self._post_with_recovery(payload)
        return self._parse(response, page_index=page.page_index)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # -- request ------------------------------------------------------------
    def _build_payload(self, image_path: Path) -> dict[str, Any]:
        if not image_path.is_file():
            raise OCRProviderError(f"page image not found: {image_path}")
        suffix = image_path.suffix.lower()
        if suffix not in _ALLOWED_SUFFIXES:
            # Documented: PNG and JPEG only (PDFs are rasterised upstream).
            raise OCRProviderError(
                f"unsupported image type {suffix!r}; the API accepts PNG/JPEG only"
            )
        media_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
        image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return {
            "model": _MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{image_b64}"},
                        }
                    ],
                }
            ],
            "table_format": self._table_format,
            "max_tokens": self._max_tokens,
        }

    async def _post_with_recovery(self, payload: dict[str, Any]) -> httpx.Response:
        """POST with bounded retry + a bounded 502 max_tokens recovery."""
        current_max_tokens = int(payload["max_tokens"])
        recoveries = 0

        while True:
            attempt_payload = dict(payload, max_tokens=current_max_tokens)
            try:
                return await self._post_once(attempt_payload)
            except OCRProviderError as exc:
                if (
                    exc.status_code == _RECOVERY_STATUS
                    and recoveries < self._max_502_recoveries
                    and current_max_tokens < self._max_tokens_ceiling
                ):
                    recoveries += 1
                    previous = current_max_tokens
                    current_max_tokens = min(current_max_tokens * 2, self._max_tokens_ceiling)
                    logger.info(
                        "502 recovery: raising max_tokens %d -> %d (attempt %d/%d)",
                        previous,
                        current_max_tokens,
                        recoveries,
                        self._max_502_recoveries,
                    )
                    continue  # documented remedy: raise max_tokens and retry
                raise

    async def _post_once(self, payload: dict[str, Any]) -> httpx.Response:
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self._max_retries),
                wait=RetryAfterWait(wait_exponential(multiplier=0.5, max=20.0)),
                retry=retry_if_exception(_is_retryable),
                reraise=True,
            ):
                with attempt:
                    response = await self._client.post(self._base_url, json=payload)
                    info = read_rate_limit(response)
                    if info is not None:
                        self._rate_limit = info
                    if response.status_code == _RECOVERY_STATUS:
                        raise OCRProviderError(
                            f"Indic-OCR returned 502: {response.text[:300]}",
                            status_code=_RECOVERY_STATUS,
                        )
                    if response.status_code >= 400:
                        raise OCRProviderError(
                            f"Indic-OCR returned HTTP {response.status_code}: "
                            f"{response.text[:300]}",
                            status_code=response.status_code,
                            retry_after=read_retry_after(response),
                        )
        except OCRProviderError:
            raise
        except httpx.HTTPError as exc:
            raise OCRProviderError(f"Indic-OCR request failed: {exc}") from exc
        return response

    # -- response -----------------------------------------------------------
    def _parse(self, response: httpx.Response, *, page_index: int) -> OCRResult:
        try:
            body = response.json()
        except ValueError as exc:
            raise OCRResponseError(
                "Indic-OCR returned non-JSON body", raw_body=response.text[:500]
            ) from exc
        if not isinstance(body, dict):
            raise OCRResponseError("Indic-OCR returned a non-object body")

        # --- documented error envelope -----------------------------------
        if "error" in body:
            error = body.get("error")
            detail = error.get("message") if isinstance(error, dict) else error
            raise OCRProviderError(f"Indic-OCR error: {detail}")

        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise OCRResponseError("response is missing 'choices'", raw_body=str(body)[:300])
        message = choices[0].get("message") or {}
        page_markdown = message.get("content")

        raw_blocks = body.get("blocks")
        if not isinstance(raw_blocks, list):
            raise OCRResponseError(
                "response is missing the top-level 'blocks' array",
                raw_body=str(body)[:300],
            )

        # --- blocks: deterministic label mapping + normalised reading order
        staged: list[tuple[int, int, OCRBlock]] = []
        for index, raw in enumerate(raw_blocks):
            if not isinstance(raw, dict):
                continue
            label = str(raw.get("label") or "") or None
            order_raw = raw.get("order")
            sort_key = order_raw if isinstance(order_raw, int) else index
            text = str(raw.get("text") or "")
            conf = self._coerce_confidence(raw.get("conf"))
            block = OCRBlock(
                block_id=f"p{page_index}-b{index}",
                page_index=page_index,
                text=text,
                kind=map_label(label),
                reading_order=0,  # renumbered below
                confidence=conf,
                bbox=self._coerce_bbox(raw.get("bbox_xyxy")),
                provider_label=label,
                provider_type=str(raw["type"]) if raw.get("type") is not None else None,
            )
            if conf is not None and conf < self._threshold:
                block.flag(QualityFlag.LOW_CONFIDENCE)
            staged.append((sort_key, index, block))

        # Sort by provider order (falling back to array position) and renumber
        # 0..n-1 without assuming the provider starts at zero.
        staged.sort(key=lambda item: (item[0], item[1]))
        blocks = [block for _, _, block in staged]
        for position, block in enumerate(blocks):
            block.reading_order = position

        usage_raw = body.get("usage")
        usage: dict[str, int] = {}
        if isinstance(usage_raw, dict):
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = usage_raw.get(key)
                if isinstance(value, int):
                    usage[key] = value

        rate_limit = self._rate_limit.as_dict() if self._rate_limit else {}

        return OCRResult(
            page_index=page_index,
            blocks=blocks,
            provider="indic-ocr",
            provider_request_id=str(body["id"]) if body.get("id") else None,
            page_markdown=page_markdown if isinstance(page_markdown, str) else None,
            raw_blocks=[b for b in raw_blocks if isinstance(b, dict)],
            usage=usage,
            rate_limit=rate_limit,
        )

    @staticmethod
    def _coerce_confidence(value: Any) -> float | None:
        if value is None:
            return None
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return None
        if 1.0 < confidence <= 100.0:  # tolerate percentage-style values
            confidence /= 100.0
        return max(0.0, min(1.0, confidence))

    @staticmethod
    def _coerce_bbox(value: Any) -> tuple[float, float, float, float] | None:
        if isinstance(value, (list, tuple)) and len(value) == 4:
            try:
                coords = tuple(float(v) for v in value)
            except (TypeError, ValueError):
                return None
            return coords  # type: ignore[return-value]
        return None
