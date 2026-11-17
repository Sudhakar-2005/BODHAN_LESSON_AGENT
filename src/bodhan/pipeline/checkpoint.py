"""Stage checkpoint integrity checks (Phase 3B-2 Task 3).

A manifest entry saying ``status == "done"`` is necessary but not sufficient:
the stage's artifacts may be missing, empty, corrupt or altered. Callers use
these helpers before skipping a stage so a valid checkpoint is reused while a
damaged one is regenerated (without invalidating unrelated stages).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from bodhan.domain.models import StageName

#: Files whose absence makes a completed stage unusable.
_REQUIRED: dict[StageName, tuple[str, ...]] = {
    StageName.INGEST: ("pages.json",),
    StageName.PREPROCESS: ("pages.json",),
    StageName.EXTRACT: ("pages.json", "results.json"),
    StageName.STRUCTURE: (
        "pages.json",
        "blocks.json",
        "suppressed.json",
        "continuity.json",
        "tables.json",
        "figures.json",
        "document_intelligence.json",
    ),
    StageName.QUALITY: ("pages.json", "blocks.json", "quality_report.json"),
    StageName.TRANSLATE: ("pages.json", "translations.json", "tables.json"),
    StageName.COMPOSE: ("pages.json", "lesson.json", "plan.json"),
    StageName.NARRATE: ("pages.json", "segments.json", "status.json"),
    StageName.TRANSCRIBE: ("pages.json", "manifest.json"),
    StageName.EXPORT: ("manifest.json",),
}


def artifact_hash(path: Path) -> str:
    """SHA-256 of a file's bytes (stable integrity metadata)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_stage_artifacts(
    root: Path, stage: StageName, *, legacy_ok: bool = False
) -> tuple[bool, list[str]]:
    """Return ``(valid, reasons)`` for the artifacts of one completed stage.

    ``legacy_ok=True`` relaxes checking of Phase 3B-1-only files so pre-3B-1
    artifacts are not invalidated; callers should pass it for backward
    compatibility when validating older workspaces.
    """
    valid, reasons = _check_artifacts(root, stage, legacy_ok=legacy_ok)
    return valid, reasons


def _check_artifacts(root: Path, stage: StageName, *, legacy_ok: bool) -> tuple[bool, list[str]]:
    directory = root / stage.value
    if not directory.is_dir():
        return False, [f"stage directory missing: {directory}"]
    required = _REQUIRED.get(stage, ())
    reasons: list[str] = []
    for name in required:
        path = directory / name
        if not path.is_file():
            if legacy_ok and name in {
                "suppressed.json",
                "continuity.json",
                "tables.json",
                "figures.json",
                "document_intelligence.json",
                "status.json",
            }:
                continue
            reasons.append(f"missing artifact: {name}")
            continue
        if path.stat().st_size == 0:
            reasons.append(f"empty artifact: {name}")
            continue
        try:
            payload: Any = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            reasons.append(f"malformed JSON: {name}")
            continue
        if isinstance(payload, list) and not payload and name != "failed_segments.json":
            # Empty lists are legitimate for e.g. suppressed.json/figures.json.
            continue
        if isinstance(payload, dict) and not payload:
            reasons.append(f"empty JSON object: {name}")
    return (not reasons), reasons


def checkpoint_verification_level(
    root: Path,
    stage: StageName,
    recorded_hashes: dict[str, str] | None,
) -> str:
    """Return ``verified`` / ``legacy`` / ``invalid`` for a stage checkpoint.

    A checkpoint is ``verified`` only when recorded hashes exist and every
    recorded hash still matches the current artifact. A checkpoint with no
    hashes may be reused for backward compatibility but is explicitly
    labelled ``legacy`` - never integrity-verified.
    """
    valid, _reasons = _check_artifacts(root, stage, legacy_ok=True)
    if not valid:
        return "invalid"
    if not isinstance(recorded_hashes, dict) or not recorded_hashes:
        return "legacy"
    current = stage_artifact_hashes(root, stage)
    if any(current.get(key) != value for key, value in recorded_hashes.items()):
        return "invalid"
    return "verified"


def stage_artifact_hashes(root: Path, stage: StageName) -> dict[str, str]:
    """Record name -> sha256 for every artifact present in a stage directory."""
    directory = root / stage.value
    if not directory.is_dir():
        return {}
    return {
        path.name: artifact_hash(path) for path in sorted(directory.iterdir()) if path.is_file()
    }
