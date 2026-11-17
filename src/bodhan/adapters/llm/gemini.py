"""Gemini chat adapter (structure / quality / translation / composition only).

Minimal httpx client over the OpenAI-compatible style REST surface of the
configured ``LLM_BASE_URL``. Deliberately thin: prompt engineering and schema
validation live in the stages that use it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx
from pydantic import BaseModel
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from bodhan.ports.llm import LLMMessage, LLMPort, LLMResponse

_RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


class LLMProviderError(Exception):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
        return True
    return isinstance(exc, LLMProviderError) and exc.status_code in _RETRYABLE_STATUS


class GeminiLLMAdapter(LLMPort):
    """Async chat-completion client."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_s: float = 120.0,
        max_retries: int = 5,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not model:
            raise ValueError("model is required")
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._max_retries = max(1, max_retries)
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_s),
            headers={"x-goog-api-key": api_key} if api_key else {},
        )

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        response_schema: type[BaseModel] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        payload = self._build_payload(messages, temperature=temperature)
        url = f"{self._base_url}/models/{self._model}:generateContent"

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self._max_retries),
                wait=wait_exponential(multiplier=1.0, max=30.0),
                retry=retry_if_exception(_is_retryable),
                reraise=True,
            ):
                with attempt:
                    response = await self._client.post(url, json=payload)
                    if response.status_code >= 400:
                        raise LLMProviderError(
                            f"LLM returned HTTP {response.status_code}: {response.text[:300]}",
                            status_code=response.status_code,
                        )
        except LLMProviderError:
            raise
        except httpx.HTTPError as exc:
            raise LLMProviderError(f"LLM request failed: {exc}") from exc

        return self._parse(response, response_schema=response_schema)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # -- internals ----------------------------------------------------------
    @staticmethod
    def _build_payload(messages: Sequence[LLMMessage], *, temperature: float) -> dict[str, Any]:
        return {
            "contents": [{"role": m.role, "parts": [{"text": m.content}]} for m in messages],
            "generationConfig": {"temperature": temperature},
        }

    def _parse(
        self, response: httpx.Response, *, response_schema: type[BaseModel] | None
    ) -> LLMResponse:
        try:
            body = response.json()
        except ValueError as exc:
            raise LLMProviderError("LLM returned non-JSON body") from exc

        try:
            parts = body["candidates"][0]["content"]["parts"]
            text = "".join(str(p.get("text", "")) for p in parts)
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMProviderError(f"unexpected LLM response shape: {str(body)[:300]}") from exc

        usage = body.get("usageMetadata", {}) or {}
        if response_schema is not None and text.strip():
            # Structural validation is the caller's schema; parse eagerly so
            # stages receive typed data or a clear failure.
            response_schema.model_validate_json(text)

        return LLMResponse(
            text=text,
            model=self._model,
            input_tokens=usage.get("promptTokenCount"),
            output_tokens=usage.get("candidatesTokenCount"),
            finish_reason=(body.get("candidates") or [{}])[0].get("finishReason"),
        )
