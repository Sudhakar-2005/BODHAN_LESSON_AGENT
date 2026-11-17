"""Error taxonomy for OCR adapters."""

from __future__ import annotations


class OCRProviderError(Exception):
    """Base class for all OCR failures."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        #: Seconds the provider asked us to wait (documented ``retry-after``
        #: header on a rate-limit ``429``); ``None`` when absent/unparseable.
        self.retry_after = retry_after


class OCRResponseError(OCRProviderError):
    """The provider returned a payload we could not interpret."""

    def __init__(self, message: str, *, raw_body: str | None = None) -> None:
        super().__init__(message)
        self.raw_body = raw_body
