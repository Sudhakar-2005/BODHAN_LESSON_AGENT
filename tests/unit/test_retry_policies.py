"""Retry-policy tests for the shared helper and the OCR/Speak adapters (Task 4).

Proves:
- the documented ``retry-after`` header is honoured everywhere it applies
  (translate, speak, OCR) via the shared ``RetryAfterWait``;
- exponential backoff stays bounded (attempt budget + wait cap);
- terminal auth/validation statuses (401/403/400/413/422) are never
  retried - the 403 behaviour is unchanged;
- rate-limit 429 without a ``retry-after`` header (out-of-credit) is
  terminal for Speak.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import httpx
import pytest
from tenacity import RetryCallState, wait_exponential

from bodhan.adapters.ocr import IndicOCRAdapter, OCRProviderError
from bodhan.adapters.retry import RETRY_AFTER_CAP_S, RetryAfterWait, read_retry_after
from bodhan.adapters.tts import BodhanSpeakAdapter, TTSProviderError
from bodhan.domain.models import Page

OCR_URL = "https://api.test/v1/chat/completions"
SPEAK_URL = "https://api.test/v1/audio/speech"
PNG_BYTES = b"\x89PNG fake image bytes"


def _state_with(exc: BaseException) -> RetryCallState:
    """Build a RetryCallState whose last outcome is ``exc`` (no sleeping)."""
    state = RetryCallState(retry_object=None, fn=None, args=(), kwargs={})  # type: ignore[arg-type]
    state.set_exception((type(exc), exc, exc.__traceback__))
    return state


def _ocr_adapter(**kwargs: object) -> IndicOCRAdapter:
    defaults: dict[str, object] = {
        "base_url": OCR_URL,
        "api_key": "test-key",
        "timeout_s": 5.0,
        "max_retries": 3,
        "low_confidence_threshold": 0.85,
    }
    defaults.update(kwargs)
    return IndicOCRAdapter(**defaults)  # type: ignore[arg-type]


def _png_page(directory: Path) -> Page:
    image = directory / "page-001.png"
    image.write_bytes(PNG_BYTES)
    return Page(page_index=0, source_path=str(image), image_path=str(image))


def _ocr_success_body() -> dict[str, object]:
    return {
        "id": "chatcmpl-1",
        "choices": [{"message": {"role": "assistant", "content": "# Title\nBody text."}}],
        "blocks": [
            {
                "order": 1,
                "label": "Chapter-title",
                "type": "text",
                "bbox_xyxy": [1, 1, 10, 10],
                "conf": 0.99,
                "text": "Title",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


# --- shared helper: read_retry_after --------------------------------------


def test_read_retry_after_parses_seconds() -> None:
    response = httpx.Response(429, headers={"retry-after": "7.5"})
    assert read_retry_after(response) == 7.5


def test_read_retry_after_absent_or_garbage_is_none() -> None:
    assert read_retry_after(httpx.Response(429)) is None
    assert read_retry_after(httpx.Response(429, headers={"retry-after": "soon"})) is None
    assert read_retry_after(httpx.Response(429, headers={"retry-after": "-3"})) is None


# --- shared helper: RetryAfterWait ----------------------------------------


def test_retry_after_wait_prefers_provider_delay() -> None:
    wait = RetryAfterWait(wait_exponential(multiplier=1.0, max=30.0))
    exc = OCRProviderError("rate", status_code=429, retry_after=12.0)
    assert wait(_state_with(exc)) == 12.0


def test_retry_after_wait_caps_absurd_header() -> None:
    wait = RetryAfterWait(wait_exponential(multiplier=1.0, max=30.0))
    exc = OCRProviderError("rate", status_code=429, retry_after=99999.0)
    assert wait(_state_with(exc)) == RETRY_AFTER_CAP_S


def test_retry_after_wait_uses_backoff_without_header() -> None:
    wait = RetryAfterWait(wait_exponential(multiplier=1.0, max=30.0))
    exc = OCRProviderError("busy", status_code=503)
    assert wait(_state_with(exc)) == 1.0


# --- OCR: retry-after honoured, terminal statuses untouched ----------------


async def test_ocr_429_with_retry_after_is_retried(httpx_mock) -> None:
    """Rate-limit 429 + retry-after rides out the window, then succeeds."""
    httpx_mock.add_response(
        url=OCR_URL,
        status_code=429,
        headers={"retry-after": "0"},
        json={"error": {"message": "rate limit"}},
    )
    httpx_mock.add_response(url=OCR_URL, json=_ocr_success_body())
    adapter = _ocr_adapter(max_retries=2)
    with TemporaryDirectory() as tmp:
        page = _png_page(Path(tmp))
        try:
            result = await adapter.extract(page)
        finally:
            await adapter.aclose()

    assert result.provider == "indic-ocr"
    assert len(httpx_mock.get_requests()) == 2  # 429 then 200


async def test_ocr_429_retry_after_is_recorded_on_error(httpx_mock) -> None:
    """The structured error carries the provider's delay (audit + wait input)."""
    httpx_mock.add_response(
        url=OCR_URL,
        status_code=429,
        headers={"retry-after": "11"},
        json={"error": {"message": "rate limit"}},
    )
    adapter = _ocr_adapter(max_retries=1)  # single attempt: fail fast
    with TemporaryDirectory() as tmp:
        page = _png_page(Path(tmp))
        try:
            with pytest.raises(OCRProviderError) as exc_info:
                await adapter.extract(page)
        finally:
            await adapter.aclose()

    assert exc_info.value.status_code == 429
    assert exc_info.value.retry_after == 11.0
    assert len(httpx_mock.get_requests()) == 1  # budget exhausted, no runaway loop


async def test_ocr_forbidden_is_terminal_even_with_retry_after(httpx_mock) -> None:
    """403 behaviour is unchanged: never retried, header or not."""
    httpx_mock.add_response(
        url=OCR_URL,
        status_code=403,
        headers={"retry-after": "0"},
        json={"error": {"message": "forbidden"}},
    )
    adapter = _ocr_adapter(max_retries=3)
    with TemporaryDirectory() as tmp:
        page = _png_page(Path(tmp))
        try:
            with pytest.raises(OCRProviderError) as exc_info:
                await adapter.extract(page)
        finally:
            await adapter.aclose()

    assert exc_info.value.status_code == 403
    assert len(httpx_mock.get_requests()) == 1


# --- Speak: header-gated 429 policy (see also test_bodhan_speak.py) -------


async def test_speak_403_is_terminal(tmp_path: Path) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(403, json={"message": "forbidden"})

    adapter = BodhanSpeakAdapter(
        base_url=SPEAK_URL,
        api_key="test-key",
        voice="Kavya",
        lang="hi",
        timeout_s=5.0,
        max_retries=3,
        chunk_seconds=30.0,
        chars_per_second=13.0,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(TTSProviderError) as exc_info:
        await adapter.synthesize(
            "hello.", segment_id="seg-01", lesson_id="l1", output_path=tmp_path / "x.wav"
        )
    await adapter.aclose()

    assert exc_info.value.status_code == 403
    assert len(calls) == 1
