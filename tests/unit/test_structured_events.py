"""Phase 3B-2 Task 6: structured logging never leaks secrets or content."""

from __future__ import annotations

import json
import logging

from bodhan.observability import emit_event


def test_event_contains_run_stage_language_and_allowlisted_fields() -> None:
    logger = logging.getLogger("test.structured")
    records: list[str] = []
    handler = logging.Handler()

    def emit(record: logging.LogRecord) -> None:
        records.append(record.getMessage())

    handler.emit = emit  # type: ignore[method-assign]
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    payload = emit_event(
        logger,
        "stage_completed",
        run_id="chapter-01",
        stage="translate",
        target_language="hi",
        status="done",
        duration_s=1.25,
        block_ids=["p0-b0"],
        retries=2,
    )

    assert payload["run_id"] == "chapter-01"
    assert json.loads(records[0].split(" ", 1)[1]) == payload
    logger.removeHandler(handler)


def test_secrets_headers_and_document_text_are_filtered() -> None:
    logger = logging.getLogger("test.secrets")
    emitted: list[str] = []
    handler = logging.Handler()
    handler.emit = lambda record: emitted.append(record.getMessage())  # type: ignore[method-assign]
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)

    emit_event(
        logger,
        "provider_call",
        run_id="run-1",
        stage="translate",
        level=logging.WARNING,
        api_key="sk-super-secret",
        authorization="Bearer sk-super-secret",
        headers={"Authorization": "secret"},
        transcription_text="नमस्ते दुनिया",
        body_text="confidential document",
        retry_after=5,
        estimated_cost=0.12,
        page_index=3,
    )

    message = emitted[0]
    assert "sk-super-secret" not in message
    assert "Bearer" not in message
    assert "नमस्ते" not in message
    assert "confidential document" not in message
    payload = json.loads(message.split(" ", 1)[1])
    assert payload["retry_after"] == 5
    assert payload["estimated_cost"] == 0.12
    assert payload["page_index"] == 3
    logger.removeHandler(handler)


def test_multiline_and_long_strings_are_not_logged() -> None:
    logger = logging.getLogger("test.text")
    emitted: list[str] = []
    handler = logging.Handler()
    handler.emit = lambda record: emitted.append(record.getMessage())  # type: ignore[method-assign]
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    payload = emit_event(logger, "event", stage="x", note="a\nb", blob="z" * 500)
    assert "note" not in payload
    assert "blob" not in payload
    logger.removeHandler(handler)
