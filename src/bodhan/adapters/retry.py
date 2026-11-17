"""Shared retry helpers for the Bodhan.AI adapters.

The official docs publish two behaviours that every model respects:

- over the per-key requests-per-minute allowance the gateway answers
  ``429`` with a ``retry-after`` header (seconds until a slot frees up,
  sliding window: indic-translate 8 RPM, indic-ocr 4, indic-speak 4);
- transient transport/5xx failures are retried with bounded exponential
  backoff.

``RetryAfterWait`` combines the two: it waits the provider-specified
delay when the header is present and falls back to exponential backoff
otherwise, always bounded by ``cap`` so a bogus value cannot stall a
stage. Terminal statuses (auth ``401/403``, validation ``400/413/422``)
are never retried - that decision stays in each adapter's retry policy.
"""

from __future__ import annotations

import httpx
from tenacity import RetryCallState
from tenacity.wait import wait_base

#: Upper bound for any provider-specified wait (the sliding window is one
#: minute; anything larger is treated as a bogus/hostile header value).
RETRY_AFTER_CAP_S = 60.0


def read_retry_after(response: httpx.Response) -> float | None:
    """Seconds from the documented ``retry-after`` header, or ``None``.

    Absent or unparseable headers (and negative values) yield ``None`` so
    callers keep their bounded exponential backoff.
    """
    raw = response.headers.get("retry-after")
    if raw is None:
        return None
    try:
        seconds = float(raw.strip())
    except ValueError:
        return None
    return seconds if seconds >= 0 else None


class RetryAfterWait(wait_base):
    """Exponential backoff that honours a provider's ``retry-after``.

    Waiting the provider-specified delay is the only way to ride out a
    rate-limit window; the exponential delay remains the fallback when the
    header is absent and as a lower bound. The wait is capped (default
    60 s) so an absurd header cannot stall the pipeline - the bounded
    attempt budget then runs out and the stage fails loudly.
    """

    def __init__(self, fallback: wait_base, *, cap: float = RETRY_AFTER_CAP_S) -> None:
        self._fallback = fallback
        self._cap = cap

    def __call__(self, retry_state: RetryCallState) -> float:
        delay = self._fallback(retry_state)
        outcome = retry_state.outcome
        if outcome is not None and outcome.done() and not outcome.cancelled():
            exc = outcome.exception()
            retry_after = getattr(exc, "retry_after", None)
            if isinstance(retry_after, (int, float)) and not isinstance(retry_after, bool):
                delay = max(delay, min(float(retry_after), self._cap))
        return delay
