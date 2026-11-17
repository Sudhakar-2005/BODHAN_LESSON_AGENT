"""Contract tests for the Indic-OCR adapter (the exclusive OCR engine).

Exercises the official documented contract in isolation via a mocked HTTP
transport: chat-completion payload, 37-class label mapping, reading-order
normalisation, bounded 502 recovery, terminal 401, rate-limit headers and
preserved raw metadata.
"""

from __future__ import annotations

import base64
import json

import pytest

from bodhan.adapters.ocr import IndicOCRAdapter, OCRProviderError, OCRResponseError
from bodhan.adapters.ocr.labels import DOCUMENTED_LABELS, LABEL_TO_KIND, map_label
from bodhan.domain.models import BlockKind, Page, QualityFlag

BASE_URL = "https://api.test/v1/chat/completions"
PNG_BYTES = b"\x89PNG fake image bytes"


def make_adapter(**kwargs: object) -> IndicOCRAdapter:
    defaults: dict[str, object] = {
        "base_url": BASE_URL,
        "api_key": "test-key",
        "timeout_s": 5.0,
        "max_retries": 3,
        "low_confidence_threshold": 0.85,
    }
    defaults.update(kwargs)
    return IndicOCRAdapter(**defaults)  # type: ignore[arg-type]


def make_page(tmp_path: object) -> Page:
    from pathlib import Path

    image = Path(str(tmp_path)) / "page-001.png"
    image.write_bytes(PNG_BYTES)
    return Page(page_index=0, source_path=str(image), image_path=str(image))


def sample_response() -> dict[str, object]:
    """Documented shape: chat.completion + top-level ``blocks``."""
    return {
        "id": "chatcmpl-123",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "# अध्याय 1\n\nयह एक परीक्षण अनुच्छेद है।",
                }
            }
        ],
        "blocks": [
            {
                "order": 1,  # documented examples start at 1, not 0
                "label": "Chapter-title",
                "type": "text",
                "bbox_xyxy": [10, 10, 200, 60],
                "conf": 0.99,
                "text": "अध्याय 1",
            },
            {
                "order": 2,
                "label": "Paragraph",
                "type": "text",
                "bbox_xyxy": [10, 80, 580, 140],
                "conf": 0.42,
                "text": "Garbled t3xt",
            },
            {
                "order": 3,
                "label": "Equation",
                "type": "formula",
                "bbox_xyxy": [10, 160, 300, 200],
                "conf": 97,
                "text": "E = mc^2",
            },
            {
                "order": 4,
                "label": "Image",
                "type": "picture",
                "bbox_xyxy": [10, 220, 580, 400],
                "conf": 0.93,
                "text": "",
            },
        ],
        "usage": {"prompt_tokens": 500, "completion_tokens": 80, "total_tokens": 580},
    }


# --- label taxonomy -------------------------------------------------------


def test_label_map_covers_all_37_documented_labels() -> None:
    assert len(DOCUMENTED_LABELS) == 37
    assert set(LABEL_TO_KIND) == DOCUMENTED_LABELS


def test_map_label_is_deterministic() -> None:
    assert map_label("Chapter-title") is BlockKind.HEADING
    assert map_label("Paragraph") is BlockKind.PARAGRAPH
    assert map_label("List") is BlockKind.LIST_ITEM
    assert map_label("Table") is BlockKind.TABLE
    assert map_label("Equation") is BlockKind.EQUATION
    assert map_label("Image") is BlockKind.FIGURE
    assert map_label("Page-number") is BlockKind.PAGE_FOOTER
    assert map_label("Advertisement") is BlockKind.UNKNOWN
    # Unknown/blank labels degrade to UNKNOWN, never raise.
    assert map_label("Not-A-Label") is BlockKind.UNKNOWN
    assert map_label(None) is BlockKind.UNKNOWN
    assert map_label("") is BlockKind.UNKNOWN


# --- request payload ------------------------------------------------------


async def test_payload_is_documented_chat_completion(tmp_path: object, httpx_mock) -> None:
    httpx_mock.add_response(url=BASE_URL, json=sample_response())
    adapter = make_adapter()

    await adapter.extract(make_page(tmp_path))
    request = httpx_mock.get_requests()[0]
    await adapter.aclose()

    assert request.headers["authorization"] == "Bearer test-key"
    body = json.loads(request.content.decode("utf-8"))
    assert set(body) == {"model", "messages", "table_format", "max_tokens"}
    assert body["model"] == "indic-ocr"
    assert body["table_format"] == "html"
    assert body["max_tokens"] == 2048
    # Image is a data URI of the exact page bytes - no prompt, no language.
    image_url = body["messages"][0]["content"][0]["image_url"]["url"]
    assert image_url.startswith("data:image/png;base64,")
    assert base64.b64decode(image_url.split(",", 1)[1]) == PNG_BYTES


async def test_missing_image_file_fails_fast() -> None:
    adapter = make_adapter()
    page = Page(
        page_index=0, source_path="Z:/does/not/exist.png", image_path="Z:/does/not/exist.png"
    )

    with pytest.raises(OCRProviderError, match="not found"):
        await adapter.extract(page)
    await adapter.aclose()


async def test_non_png_jpeg_image_is_rejected(tmp_path: object) -> None:
    from pathlib import Path

    adapter = make_adapter()
    image = Path(str(tmp_path)) / "page-001.tiff"
    image.write_bytes(b"II*\x00tiff")
    page = Page(page_index=0, source_path=str(image), image_path=str(image))

    with pytest.raises(OCRProviderError, match="PNG/JPEG"):
        await adapter.extract(page)
    await adapter.aclose()


# --- response parsing -----------------------------------------------------


async def test_parse_blocks_maps_labels_and_normalises_order(tmp_path: object, httpx_mock) -> None:
    httpx_mock.add_response(url=BASE_URL, json=sample_response())
    adapter = make_adapter()

    result = await adapter.extract(make_page(tmp_path))
    await adapter.aclose()

    assert result.provider == "indic-ocr"
    assert result.provider_request_id == "chatcmpl-123"
    # Provider order started at 1 -> renumbered to 0..n-1, order preserved.
    assert [b.reading_order for b in result.blocks] == [0, 1, 2, 3]
    assert [b.kind for b in result.blocks] == [
        BlockKind.HEADING,
        BlockKind.PARAGRAPH,
        BlockKind.EQUATION,
        BlockKind.FIGURE,
    ]
    assert result.blocks[0].provider_label == "Chapter-title"
    assert result.blocks[3].provider_type == "picture"
    assert result.blocks[0].bbox == (10.0, 10.0, 200.0, 60.0)


async def test_reading_order_sorts_by_provider_order(tmp_path: object, httpx_mock) -> None:
    """Blocks arrive out of order in the array; sorting must follow `order`."""
    response = sample_response()
    blocks = response["blocks"]  # type: ignore[index]
    response["blocks"] = list(reversed(blocks))  # type: ignore[index]
    httpx_mock.add_response(url=BASE_URL, json=response)
    adapter = make_adapter()

    result = await adapter.extract(make_page(tmp_path))
    await adapter.aclose()

    assert [b.text for b in result.blocks] == ["अध्याय 1", "Garbled t3xt", "E = mc^2", ""]


async def test_raw_metadata_is_preserved(tmp_path: object, httpx_mock) -> None:
    httpx_mock.add_response(
        url=BASE_URL,
        json=sample_response(),
        headers={"x-ratelimit-limit-requests": "60", "x-ratelimit-limit-parallel-requests": "2"},
    )
    adapter = make_adapter()

    result = await adapter.extract(make_page(tmp_path))
    await adapter.aclose()

    assert result.page_markdown is not None and result.page_markdown.startswith("# अध्याय 1")
    assert len(result.raw_blocks) == 4
    assert result.usage == {"prompt_tokens": 500, "completion_tokens": 80, "total_tokens": 580}
    assert result.rate_limit == {"limit_requests": 60, "limit_parallel": 2}


async def test_low_confidence_blocks_are_flagged(tmp_path: object, httpx_mock) -> None:
    httpx_mock.add_response(url=BASE_URL, json=sample_response())
    adapter = make_adapter()

    result = await adapter.extract(make_page(tmp_path))
    await adapter.aclose()

    flagged = result.flagged_blocks
    assert [b.text for b in flagged] == ["Garbled t3xt"]
    assert QualityFlag.LOW_CONFIDENCE in flagged[0].flags
    assert flagged[0].confidence == pytest.approx(0.42)
    assert result.blocks[0].flags == []


async def test_percentage_confidence_is_normalised(tmp_path: object, httpx_mock) -> None:
    httpx_mock.add_response(url=BASE_URL, json=sample_response())
    adapter = make_adapter()

    result = await adapter.extract(make_page(tmp_path))
    await adapter.aclose()

    # Provider reported 97 (percent) -> 0.97, above threshold, not flagged.
    assert result.blocks[2].confidence == pytest.approx(0.97)
    assert result.blocks[2].flags == []


async def test_missing_blocks_array_raises_response_error(tmp_path: object, httpx_mock) -> None:
    httpx_mock.add_response(url=BASE_URL, json={"id": "x", "choices": [], "usage": {}})
    adapter = make_adapter()

    with pytest.raises(OCRResponseError):
        await adapter.extract(make_page(tmp_path))
    await adapter.aclose()


async def test_non_json_body_raises_response_error(tmp_path: object, httpx_mock) -> None:
    httpx_mock.add_response(url=BASE_URL, text="<html>gateway error</html>")
    adapter = make_adapter()

    with pytest.raises(OCRResponseError):
        await adapter.extract(make_page(tmp_path))
    await adapter.aclose()


async def test_error_envelope_raises_provider_error(tmp_path: object, httpx_mock) -> None:
    httpx_mock.add_response(
        url=BASE_URL, json={"error": {"message": "bad image", "code": "invalid_image"}}
    )
    adapter = make_adapter()

    with pytest.raises(OCRProviderError, match="bad image"):
        await adapter.extract(make_page(tmp_path))
    await adapter.aclose()


# --- retry / recovery policy ----------------------------------------------


async def test_auth_error_is_terminal(tmp_path: object, httpx_mock) -> None:
    httpx_mock.add_response(url=BASE_URL, status_code=401, json={"error": "unauthorized"})
    adapter = make_adapter()

    with pytest.raises(OCRProviderError) as exc_info:
        await adapter.extract(make_page(tmp_path))
    await adapter.aclose()

    assert exc_info.value.status_code == 401
    assert len(httpx_mock.get_requests()) == 1  # no retry


async def test_validation_error_is_terminal(tmp_path: object, httpx_mock) -> None:
    httpx_mock.add_response(url=BASE_URL, status_code=400, json={"error": "bad request"})
    adapter = make_adapter()

    with pytest.raises(OCRProviderError) as exc_info:
        await adapter.extract(make_page(tmp_path))
    await adapter.aclose()

    assert exc_info.value.status_code == 400
    assert len(httpx_mock.get_requests()) == 1


async def test_server_error_is_retried_then_fails(tmp_path: object, httpx_mock) -> None:
    for _ in range(3):
        httpx_mock.add_response(url=BASE_URL, status_code=503, json={"error": "busy"})
    adapter = make_adapter(max_retries=3)

    with pytest.raises(OCRProviderError) as exc_info:
        await adapter.extract(make_page(tmp_path))
    await adapter.aclose()

    assert exc_info.value.status_code == 503
    assert len(httpx_mock.get_requests()) == 3


async def test_502_recovery_doubles_max_tokens_once(tmp_path: object, httpx_mock) -> None:
    """Documented remedy: 502 = block exceeded max_tokens -> raise and retry."""
    httpx_mock.add_response(url=BASE_URL, status_code=502, text="block too long")
    httpx_mock.add_response(url=BASE_URL, json=sample_response())
    adapter = make_adapter(max_502_recoveries=1)

    result = await adapter.extract(make_page(tmp_path))
    await adapter.aclose()

    assert result.provider == "indic-ocr"
    requests = httpx_mock.get_requests()
    assert len(requests) == 2
    first = json.loads(requests[0].content.decode("utf-8"))
    second = json.loads(requests[1].content.decode("utf-8"))
    assert first["max_tokens"] == 2048
    assert second["max_tokens"] == 4096


async def test_502_recovery_is_bounded(tmp_path: object, httpx_mock) -> None:
    # Two 502s: the first triggers one recovery, the second must propagate.
    httpx_mock.add_response(url=BASE_URL, status_code=502, text="block too long")
    httpx_mock.add_response(url=BASE_URL, status_code=502, text="block too long")
    adapter = make_adapter(max_502_recoveries=1)

    with pytest.raises(OCRProviderError) as exc_info:
        await adapter.extract(make_page(tmp_path))
    await adapter.aclose()

    assert exc_info.value.status_code == 502
    # initial + one recovery = 2 requests, then the error propagates.
    assert len(httpx_mock.get_requests()) == 2
