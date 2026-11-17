"""Central logging setup and structured operational events.

Keeps a single, consistent format across CLI, pipeline stages and adapters.
``emit_event`` writes one JSON object per line with explicit run/stage/target
context; it has allowlisted value types so credentials, authorization headers
and document contents can never be serialized by accident.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    """Install a single stream handler on the root logger (idempotent)."""
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


_EVENT_PREFIX = "bodhan.event"
_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "authorization",
        "auth",
        "token",
        "secret",
        "password",
        "key",
        "headers",
        "raw",
        "text",
        "content",
        "transcription_text",
    }
)


def emit_event(
    logger: logging.Logger,
    event: str,
    *,
    run_id: str | None = None,
    stage: str | None = None,
    target_language: str | None = None,
    level: int = logging.INFO,
    **fields: Any,
) -> dict[str, Any]:
    """Log a structured, content-free event and return the emitted payload.

    ``fields`` are filtered: strings/numbers/bools/None only, and keys that
    could carry secrets or document text are rejected. Latency/metric fields
    must be explicitly labelled by the caller (``estimated_*`` vs measured).
    """
    safe: dict[str, Any] = {}
    for key, value in fields.items():
        if key.lower() in _SENSITIVE_KEYS or key.lower().endswith("_text"):
            continue
        if (
            isinstance(value, (bool, int, float))
            or value is None
            or (isinstance(value, str) and len(value) <= 200 and "\n" not in value)
        ):
            safe[key] = value
        elif isinstance(value, (list, tuple)) and all(
            isinstance(item, (str, int, float)) for item in value
        ):
            safe[key] = list(value)
    payload = {
        "event": event,
        "run_id": run_id,
        "stage": stage,
        "target_language": target_language,
        "at": datetime.now(UTC).isoformat(),
        **safe,
    }
    logger.log(
        level, "%s %s", _EVENT_PREFIX, json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )
    return payload
