"""Contract tests for the Bodhan translate adapter (both documented routes).

Default transport is ``POST /v1/chat/completions`` with ``model:
indic-translate`` (verified live 200); the ``POST /translate`` product
surface (currently 404 on the gateway) is covered under ``transport=
"product"``. Covers payload keys, the 26-code enum, truncated-output
rejection, the error envelope, and terminal vs retryable statuses.
"""

from __future__ import annotations

import json

import httpx
import pytest

from bodhan.adapters.translate import (
    BodhanTranslateAdapter,
    TranslationProviderError,
    TranslationResponseError,
    TranslationTruncatedError,
    UnsupportedLanguageError,
)
from bodhan.ports.translate import SUPPORTED_TARGET_LANGUAGES

CHAT_URL = "https://api.test/v1/chat/completions"
PRODUCT_URL = "https://api.test/translate"


def make_chat(**kwargs: object) -> BodhanTranslateAdapter:
    defaults: dict[str, object] = {
        "base_url": CHAT_URL,
        "api_key": "test-key",
        "transport": "chat",
        "timeout_s": 5.0,
        "max_retries": 3,
    }
    defaults.update(kwargs)
    return BodhanTranslateAdapter(**defaults)  # type: ignore[arg-type]


def make_product(**kwargs: object) -> BodhanTranslateAdapter:
    defaults: dict[str, object] = {
        "base_url": PRODUCT_URL,
        "api_key": "test-key",
        "transport": "product",
        "timeout_s": 5.0,
        "max_retries": 3,
    }
    defaults.update(kwargs)
    return BodhanTranslateAdapter(**defaults)  # type: ignore[arg-type]


def chat_body(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "id": "chatcmpl-1",
        "model": "indic-translate",
        "object": "chat.completion",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {"role": "assistant", "content": "यह एक परीक्षण वाक्य है।"},
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }
    body.update(overrides)
    return body


def product_body(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "translation": "यह एक परीक्षण वाक्य है।",
        "target_language": "hi",
        "script": "native",
        "truncated": False,
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }
    body.update(overrides)
    return body


def transport(handler: object) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]


def test_documented_enum_has_26_codes() -> None:
    assert len(SUPPORTED_TARGET_LANGUAGES) == 26


def test_invalid_transport_is_rejected() -> None:
    with pytest.raises(ValueError, match="transport"):
        make_chat(transport="grpc")  # type: ignore[arg-type]


def test_invalid_script_is_rejected_locally() -> None:
    with pytest.raises(ValueError, match="script"):
        make_chat(script="devanagari")


# --- chat transport (default, verified live) ------------------------------


async def test_chat_payload_has_documented_fields() -> None:
    recorded: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        recorded.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(200, json=chat_body())

    adapter = make_chat(client=transport(handler))
    await adapter.translate("hello world", target_language="hi")
    await adapter.aclose()

    assert len(recorded) == 1
    payload = recorded[0]
    assert set(payload) == {"model", "messages", "target_language"}
    assert payload["model"] == "indic-translate"
    assert payload["target_language"] == "hi"
    # Exactly one user turn (the gateway rejects two with 400).
    messages = payload["messages"]
    assert isinstance(messages, list) and len(messages) == 1
    assert messages[0] == {"role": "user", "content": "hello world"}


async def test_chat_script_is_sent_when_configured() -> None:
    recorded: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        recorded.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(200, json=chat_body())

    adapter = make_chat(client=transport(handler), script="roman")
    result = await adapter.translate("hello", target_language="hi")
    await adapter.aclose()

    assert recorded[0]["script"] == "roman"
    assert result.script == "roman"


async def test_chat_authorization_header_is_set() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer test-key"
        return httpx.Response(200, json=chat_body())

    adapter = make_chat(client=transport(handler))
    await adapter.translate("hello", target_language="hi")
    await adapter.aclose()


async def test_chat_success_response_is_parsed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=chat_body(),
            headers={"x-ratelimit-limit-requests": "60"},
        )

    adapter = make_chat(client=transport(handler))
    result = await adapter.translate("hello", target_language="hi")
    await adapter.aclose()

    assert result.text == "यह एक परीक्षण वाक्य है।"
    assert result.target_language == "hi"
    assert result.usage == {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    assert adapter.rate_limit is not None and adapter.rate_limit.limit_requests == 60


async def test_chat_finish_reason_length_is_rejected_as_truncated() -> None:
    """`finish_reason: length` is the chat form of the documented truncation."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = chat_body()
        choices = body["choices"]  # type: ignore[index]
        choices[0]["finish_reason"] = "length"  # type: ignore[index]
        return httpx.Response(200, json=body)

    adapter = make_chat(client=transport(handler))
    with pytest.raises(TranslationTruncatedError, match="incomplete"):
        await adapter.translate("hello", target_language="hi")
    await adapter.aclose()


async def test_chat_missing_choices_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "x"})

    adapter = make_chat(client=transport(handler))
    with pytest.raises(TranslationResponseError, match="choices"):
        await adapter.translate("hello", target_language="hi")
    await adapter.aclose()


async def test_chat_missing_content_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = chat_body()
        body["choices"] = [{"finish_reason": "stop", "index": 0, "message": {"role": "assistant"}}]
        return httpx.Response(200, json=body)

    adapter = make_chat(client=transport(handler))
    with pytest.raises(TranslationResponseError, match="content"):
        await adapter.translate("hello", target_language="hi")
    await adapter.aclose()


async def test_chat_error_payload_in_success_status_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"error": {"message": "quota exceeded", "code": "quota"}})

    adapter = make_chat(client=transport(handler))
    with pytest.raises(TranslationResponseError, match="quota exceeded"):
        await adapter.translate("hello", target_language="hi")
    await adapter.aclose()


# --- local validation (no network) ----------------------------------------


async def test_unsupported_target_language_is_rejected_locally() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("must not hit the network for an invalid language")

    adapter = make_chat(client=transport(handler))
    with pytest.raises(UnsupportedLanguageError, match="not supported"):
        await adapter.translate("hello", target_language="xx")
    await adapter.aclose()


async def test_empty_text_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("must not hit the network for empty text")

    adapter = make_chat(client=transport(handler))
    with pytest.raises(UnsupportedLanguageError, match="empty"):
        await adapter.translate("   ", target_language="hi")
    await adapter.aclose()


# --- error envelope & retry policy ----------------------------------------


async def test_error_envelope_is_parsed() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(
            400,
            json={
                "error": {
                    "message": "unsupported language 'xx'",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "400",
                }
            },
        )

    adapter = make_chat(client=transport(handler))
    with pytest.raises(TranslationProviderError) as exc_info:
        await adapter.translate("hello", target_language="hi")
    await adapter.aclose()

    assert exc_info.value.status_code == 400
    assert exc_info.value.code == "400"
    assert "unsupported language 'xx'" in str(exc_info.value)
    assert len(calls) == 1  # 4xx is terminal: exactly one attempt


async def test_auth_error_is_terminal() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(401, json={"error": {"message": "invalid key", "code": "401"}})

    adapter = make_chat(client=transport(handler))
    with pytest.raises(TranslationProviderError) as exc_info:
        await adapter.translate("hello", target_language="hi")
    await adapter.aclose()

    assert exc_info.value.status_code == 401
    assert len(calls) == 1


async def test_payload_too_large_is_terminal() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(413, json={"error": {"message": "too large", "code": "413"}})

    adapter = make_chat(client=transport(handler))
    with pytest.raises(TranslationProviderError) as exc_info:
        await adapter.translate("hello", target_language="hi")
    await adapter.aclose()

    assert exc_info.value.status_code == 413
    assert len(calls) == 1


async def test_server_error_is_retried() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(503, json={"error": {"message": "busy"}})
        return httpx.Response(200, json=chat_body())

    adapter = make_chat(client=transport(handler))
    result = await adapter.translate("hello", target_language="hi")
    await adapter.aclose()

    assert len(calls) == 3
    assert result.text


async def test_retries_are_bounded() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503, json={"error": {"message": "busy"}})

    adapter = make_chat(client=transport(handler), max_retries=2)
    with pytest.raises(TranslationProviderError):
        await adapter.translate("hello", target_language="hi")
    await adapter.aclose()

    assert len(calls) == 2


# --- product transport (POST /translate, currently 404 on the gateway) ----


async def test_product_payload_has_only_documented_fields() -> None:
    recorded: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        recorded.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(200, json=product_body())

    adapter = make_product(client=transport(handler))
    await adapter.translate("hello world", target_language="hi")
    await adapter.aclose()

    assert recorded == [{"text": "hello world", "target_language": "hi"}]


async def test_product_success_response_is_parsed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=product_body())

    adapter = make_product(client=transport(handler))
    result = await adapter.translate("hello", target_language="hi")
    await adapter.aclose()

    assert result.text == "यह एक परीक्षण वाक्य है।"
    assert result.target_language == "hi"
    assert result.script == "native"
    assert result.usage["total_tokens"] == 30


async def test_product_truncated_flag_is_rejected() -> None:
    """`truncated: true` reads fluently but is incomplete - never accepted."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=product_body(truncated=True))

    adapter = make_product(client=transport(handler))
    with pytest.raises(TranslationTruncatedError, match="incomplete"):
        await adapter.translate("hello", target_language="hi")
    await adapter.aclose()


async def test_product_missing_translation_field_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"truncated": False})

    adapter = make_product(client=transport(handler))
    with pytest.raises(TranslationResponseError, match="translation"):
        await adapter.translate("hello", target_language="hi")
    await adapter.aclose()


async def test_non_json_body_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>oops</html>")

    adapter = make_chat(client=transport(handler))
    with pytest.raises(TranslationResponseError):
        await adapter.translate("hello", target_language="hi")
    await adapter.aclose()
