"""Contract tests for the Bodhan translate adapter (both documented routes).

Default transport is ``POST /v1/chat/completions`` with ``model:
indic-translate`` (verified live 200); the ``POST /translate`` product
surface (currently 404 on the gateway) is covered under ``transport=
"product"``. Covers payload keys, the 26-code enum, truncated-output
rejection, the error envelope, and terminal vs retryable statuses.
"""

from __future__ import annotations

import json
import time

import httpx
import pytest
from tenacity import RetryCallState
from tenacity.wait import wait_exponential

from bodhan.adapters.retry import RetryAfterWait
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


# --- documented 429 retry-after (8 req/min sliding window) -----------------


def _state_with(exc: BaseException) -> RetryCallState:
    """Build a RetryCallState whose last outcome is ``exc`` (no sleeping)."""
    state = RetryCallState(retry_object=None, fn=None, args=(), kwargs={})  # type: ignore[arg-type]
    state.set_exception((type(exc), exc, exc.__traceback__))
    return state


def test_retry_after_header_is_parsed() -> None:
    response = httpx.Response(
        429,
        headers={"retry-after": "42"},
        json={"error": {"message": "Maximum rate limit reached for the model"}},
    )
    exc = BodhanTranslateAdapter._error_from(response)
    assert exc.status_code == 429
    assert exc.retry_after == 42.0


def test_retry_after_header_absent_or_garbage_is_none() -> None:
    absent = BodhanTranslateAdapter._error_from(
        httpx.Response(429, json={"error": {"message": "x"}})
    )
    assert absent.retry_after is None
    weird = BodhanTranslateAdapter._error_from(
        httpx.Response(429, headers={"retry-after": "soon"}, json={"error": {"message": "x"}})
    )
    assert weird.retry_after is None


def test_wait_honours_retry_after_over_exponential_backoff() -> None:
    """429 + retry-after must wait the provider's delay, not the 1s fallback."""
    wait = RetryAfterWait(wait_exponential(multiplier=1.0, max=30.0))
    exc = TranslationProviderError("translate HTTP 429: rate", status_code=429, retry_after=42.0)
    assert wait(_state_with(exc)) == 42.0


def test_wait_caps_absurd_retry_after() -> None:
    """A bogus huge header cannot stall the pipeline past the cap."""
    wait = RetryAfterWait(wait_exponential(multiplier=1.0, max=30.0))
    exc = TranslationProviderError("translate HTTP 429: rate", status_code=429, retry_after=99999.0)
    assert wait(_state_with(exc)) == 60.0


def test_wait_falls_back_to_exponential_without_header() -> None:
    """No header (or a non-retry-after error) keeps the exponential delay."""
    wait = RetryAfterWait(wait_exponential(multiplier=1.0, max=30.0))
    without = TranslationProviderError("translate HTTP 429: rate", status_code=429)
    assert wait(_state_with(without)) == 1.0
    other = TranslationProviderError("translate HTTP 503: busy", status_code=503, retry_after=None)
    assert wait(_state_with(other)) == 1.0


async def test_rate_limited_call_waits_out_retry_after_then_succeeds() -> None:
    """End to end: 429 + retry-after, then 200 - and the wait was honoured."""
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(
                429,
                headers={"retry-after": "1.5"},
                json={"error": {"message": "Maximum rate limit reached for the model"}},
            )
        return httpx.Response(200, json=chat_body())

    adapter = make_chat(client=transport(handler))
    started = time.monotonic()
    result = await adapter.translate("hello", target_language="hi")
    elapsed = time.monotonic() - started
    await adapter.aclose()

    assert len(calls) == 2
    assert result.text
    # exponential backoff alone would have waited only 1.0s
    assert elapsed >= 1.5


async def test_auth_error_stays_terminal_even_with_retry_after_header() -> None:
    """403 is terminal regardless of a retry-after header (documented policy)."""
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(
            403,
            headers={"retry-after": "5"},
            json={"error": {"message": "Invalid or missing authentication credentials"}},
        )

    adapter = make_chat(client=transport(handler))
    with pytest.raises(TranslationProviderError) as exc_info:
        await adapter.translate("hello", target_language="hi")
    await adapter.aclose()

    assert exc_info.value.status_code == 403
    assert len(calls) == 1


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
