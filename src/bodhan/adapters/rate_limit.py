"""Rate-limit metadata read from provider response headers.

Only published limits are captured (never credentials, never tokens). Bodhan
returns these headers on every response::

    x-ratelimit-limit-requests             requests per minute for this key
    x-ratelimit-limit-parallel-requests    concurrent requests for this key
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx


def _to_int(raw: str | None) -> int | None:
    if raw is None:
        return None
    try:
        return int(raw.strip())
    except ValueError:
        return None


@dataclass(frozen=True)
class RateLimitInfo:
    """Published per-key limits (response-driven, not hardcoded)."""

    limit_requests: int | None = None
    limit_parallel: int | None = None

    def as_dict(self) -> dict[str, int]:
        out: dict[str, int] = {}
        if self.limit_requests is not None:
            out["limit_requests"] = self.limit_requests
        if self.limit_parallel is not None:
            out["limit_parallel"] = self.limit_parallel
        return out


def read_rate_limit(response: httpx.Response) -> RateLimitInfo | None:
    """Extract the published limits from a response, if present."""
    raw_requests = response.headers.get("x-ratelimit-limit-requests")
    raw_parallel = response.headers.get("x-ratelimit-limit-parallel-requests")
    if raw_requests is None and raw_parallel is None:
        return None
    return RateLimitInfo(
        limit_requests=_to_int(raw_requests),
        limit_parallel=_to_int(raw_parallel),
    )
