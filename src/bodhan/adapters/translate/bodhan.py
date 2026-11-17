"""Bodhan translation adapter - both documented transports of the API.

The official docs publish two transports with identical output:

1. ``POST /translate`` (product surface)::

       {"text": "...", "target_language": "hi" [, "script": "..." ]}

   Response::

       {"translation": "...", "target_language": "hi", "script": "native",
        "truncated": false, "usage": {..}}

   Live probe (2026-11-17): this route answers ``404 {"detail":"Not Found"}``
   on the gateway, so it is selectable but not the default.

2. ``POST /v1/chat/completions`` with ``model: "indic-translate"``
   (documented equivalent transport; verified live ``200``)::

       {"model": "indic-translate",
        "messages": [{"role": "user", "content": "..."}],
        "target_language": "hi" [, "script": "..." ]}

   Response is a standard ``chat.completion``: the translation is
   ``choices[0].message.content``, token usage sits in ``usage``, and
   ``finish_reason == "length"`` marks the documented truncation case
   (the product surface's ``truncated: true``).

Error envelope (every failure, both transports)::

    {"error": {"message": "...", "code": "...", ...}}

Rules honoured here:
- no source-language field exists (the API infers it) - none is sent;
- exactly one user turn (two are rejected with 400 by the gateway);
- truncated/incomplete output raises, never passes downstream;
- 401/400/413/422 are terminal (no retry); 408/425/429/5xx retry with backoff,
  and a ``429`` waits out the documented ``retry-after`` header (the published
  requests-per-minute allowance is 8 for ``indic-translate``);
- published rate limits are read from response headers, never hardcoded.
"""

from __future__ import annotations

from typing import Any, Literal

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from bodhan.adapters.rate_limit import RateLimitInfo, read_rate_limit
from bodhan.adapters.retry import RetryAfterWait, read_retry_after
from bodhan.adapters.translate.errors import (
    TranslationProviderError,
    TranslationResponseError,
    TranslationTruncatedError,
    UnsupportedLanguageError,
)
from bodhan.ports.translate import SUPPORTED_TARGET_LANGUAGES, TranslatePort, TranslationResult

_RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}
_SCRIPTS = frozenset({"native", "roman", "codemix"})
_TRANSPORTS = frozenset({"chat", "product"})

TranslateTransport = Literal["chat", "product"]


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
        return True
    return isinstance(exc, TranslationProviderError) and exc.status_code in _RETRYABLE_STATUS


class BodhanTranslateAdapter(TranslatePort):
    """Async client for the Bodhan translation API (two documented routes)."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        transport: TranslateTransport = "chat",
        model: str = "indic-translate",
        script: str | None = None,
        timeout_s: float = 60.0,
        max_retries: int = 3,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("base_url is required")
        if transport not in _TRANSPORTS:
            raise ValueError(f"transport must be one of {sorted(_TRANSPORTS)}")
        if script is not None and script not in _SCRIPTS:
            raise ValueError(f"script must be one of {sorted(_SCRIPTS)}")
        self._base_url = base_url
        self._transport: TranslateTransport = transport
        self._model = model
        self._script = script
        self._max_retries = max(1, max_retries)
        self._rate_limit: RateLimitInfo | None = None
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(timeout_s))
        if api_key:  # auth applies to injected clients too
            self._client.headers["Authorization"] = f"Bearer {api_key}"

    @property
    def rate_limit(self) -> RateLimitInfo | None:
        """Latest published rate limits read from a response (no secrets)."""
        return self._rate_limit

    # -- TranslatePort ------------------------------------------------------
    async def translate(self, text: str, *, target_language: str) -> TranslationResult:
        if target_language not in SUPPORTED_TARGET_LANGUAGES:
            raise UnsupportedLanguageError(
                f"target language {target_language!r} is not supported by the API",
                code="unsupported_language",
            )
        if not text.strip():
            raise UnsupportedLanguageError("refusing to translate empty text")

        payload = self._build_payload(text, target_language)

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self._max_retries),
                wait=RetryAfterWait(wait_exponential(multiplier=1.0, max=30.0)),
                retry=retry_if_exception(_is_retryable),
                reraise=True,
            ):
                with attempt:
                    response = await self._client.post(self._base_url, json=payload)
                    self._update_rate_limit(response)
                    if response.status_code >= 400:
                        raise self._error_from(response)
        except TranslationProviderError:
            raise
        except httpx.HTTPError as exc:
            raise TranslationProviderError(f"translation request failed: {exc}") from exc

        return self._parse(response, target_language=target_language)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # -- request ------------------------------------------------------------
    def _build_payload(self, text: str, target_language: str) -> dict[str, Any]:
        """Build the documented payload for the configured transport."""
        if self._transport == "chat":
            payload: dict[str, Any] = {
                "model": self._model,
                "messages": [{"role": "user", "content": text}],
                "target_language": target_language,
            }
        else:
            payload = {"text": text, "target_language": target_language}
        if self._script is not None:
            payload["script"] = self._script
        return payload

    # -- response -----------------------------------------------------------
    def _update_rate_limit(self, response: httpx.Response) -> None:
        info = read_rate_limit(response)
        if info is not None:
            self._rate_limit = info

    @staticmethod
    def _error_from(response: httpx.Response) -> TranslationProviderError:
        """Parse the documented error envelope; fall back to a bare message."""
        code: str | None = None
        message = response.text[:300]
        try:
            body = response.json()
        except ValueError:
            body = None
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict):
                message = str(error.get("message") or message)
                code = str(error["code"]) if error.get("code") is not None else None
        return TranslationProviderError(
            f"translate HTTP {response.status_code}: {message}",
            status_code=response.status_code,
            code=code,
            retry_after=read_retry_after(response),
        )

    @staticmethod
    def _read_body(response: httpx.Response) -> dict[str, Any]:
        try:
            body = response.json()
        except ValueError as exc:
            raise TranslationResponseError(
                "translate returned non-JSON body", status_code=response.status_code
            ) from exc
        if not isinstance(body, dict):
            raise TranslationResponseError("translate returned a non-object body")
        if "error" in body:
            # Success responses never carry an `error` key (branch on presence).
            error = body.get("error")
            detail = error.get("message") if isinstance(error, dict) else error
            raise TranslationResponseError(f"translate error payload: {detail}")
        return body

    @staticmethod
    def _usage(body: dict[str, Any]) -> dict[str, int]:
        usage: dict[str, int] = {}
        usage_raw = body.get("usage")
        if isinstance(usage_raw, dict):
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = usage_raw.get(key)
                if isinstance(value, int):
                    usage[key] = value
        return usage

    def _parse(self, response: httpx.Response, *, target_language: str) -> TranslationResult:
        body = self._read_body(response)
        usage = self._usage(body)

        if self._transport == "chat":
            # The chat response does not echo script; report what we requested.
            return self._parse_chat(
                body, target_language=target_language, usage=usage, script=self._script
            )
        return self._parse_product(body, usage=usage)

    @staticmethod
    def _parse_chat(
        body: dict[str, Any],
        *,
        target_language: str,
        usage: dict[str, int],
        script: str | None,
    ) -> TranslationResult:
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise TranslationResponseError("chat response is missing 'choices'")
        choice = choices[0]
        message = choice.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content:
            raise TranslationResponseError("chat response is missing the translated content")

        # `truncated: true` of the product surface == finish_reason "length"
        # on the chat transport (both mean the output hit the token budget).
        if choice.get("finish_reason") == "length":
            raise TranslationTruncatedError(
                "translation hit the token budget and is incomplete",
                status_code=200,
                code="truncated",
            )

        return TranslationResult(
            text=content,
            target_language=target_language,  # the chat response does not echo it
            script=script,
            usage=usage,
        )

    @staticmethod
    def _parse_product(body: dict[str, Any], *, usage: dict[str, int]) -> TranslationResult:
        if body.get("truncated") is True:
            # Documented: a truncated translation reads fluently and simply
            # stops - only this flag distinguishes it. Reject, never downgrade.
            raise TranslationTruncatedError(
                "translation hit the token budget and is incomplete",
                status_code=200,
                code="truncated",
            )

        raw_text = body.get("translation")
        if not isinstance(raw_text, str) or not raw_text:
            raise TranslationResponseError("response is missing 'translation'")

        return TranslationResult(
            text=raw_text,
            target_language=str(body.get("target_language", "")),
            script=str(body["script"]) if body.get("script") is not None else None,
            usage=usage,
        )
