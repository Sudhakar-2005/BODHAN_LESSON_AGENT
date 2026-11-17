"""Phase 3B-2 Task 3: artifact integrity checks for stage checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

from bodhan.domain.models import StageName
from bodhan.pipeline.checkpoint import (
    artifact_hash,
    checkpoint_verification_level,
    stage_artifact_hashes,
    verify_stage_artifacts,
)
from bodhan.pipeline.state import save_json
from tests.conftest import make_context


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_valid_artifacts_are_reused(tmp_path: Path) -> None:
    ctx = make_context(tmp_path)
    save_json(ctx.store, StageName.EXTRACT, "pages.json", [{"page_index": 0}])
    save_json(ctx.store, StageName.EXTRACT, "results.json", [{"page_index": 0}])

    valid, reasons = verify_stage_artifacts(ctx.store.root(), StageName.EXTRACT)

    assert valid and reasons == []


def test_missing_artifact_invalidates_checkpoint(tmp_path: Path) -> None:
    ctx = make_context(tmp_path)
    save_json(ctx.store, StageName.EXTRACT, "pages.json", [{"page_index": 0}])

    valid, reasons = verify_stage_artifacts(ctx.store.root(), StageName.EXTRACT)

    assert not valid
    assert any("results.json" in reason for reason in reasons)


def test_corrupt_json_and_empty_artifact_are_detected(tmp_path: Path) -> None:
    ctx = make_context(tmp_path)
    directory = ctx.store.stage_dir(StageName.QUALITY.value)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "pages.json").write_text("{broken", encoding="utf-8")
    (directory / "blocks.json").write_text("", encoding="utf-8")
    write_json(directory / "quality_report.json", {"flagged": 0})

    valid, reasons = verify_stage_artifacts(ctx.store.root(), StageName.QUALITY)

    assert not valid
    assert any("malformed JSON" in reason for reason in reasons)
    assert any("empty artifact" in reason for reason in reasons)


def test_empty_object_is_not_treated_as_success(tmp_path: Path) -> None:
    ctx = make_context(tmp_path)
    directory = ctx.store.stage_dir(StageName.EXTRACT.value)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "pages.json", [])
    write_json(directory / "results.json", {})

    valid, reasons = verify_stage_artifacts(ctx.store.root(), StageName.EXTRACT)

    assert not valid
    assert any("empty JSON object" in reason for reason in reasons)


def test_modified_artifact_is_detected_by_hash(tmp_path: Path) -> None:
    ctx = make_context(tmp_path)
    save_json(ctx.store, StageName.EXTRACT, "pages.json", [{"page_index": 0}])
    save_json(ctx.store, StageName.EXTRACT, "results.json", [{"page_index": 0}])
    before = stage_artifact_hashes(ctx.store.root(), StageName.EXTRACT)

    write_json(ctx.store.stage_dir(StageName.EXTRACT.value) / "results.json", {"changed": True})
    after = stage_artifact_hashes(ctx.store.root(), StageName.EXTRACT)

    assert before["results.json"] != after["results.json"]
    assert (
        artifact_hash(ctx.store.stage_dir(StageName.EXTRACT.value) / "pages.json")
        == before["pages.json"]
    )


def test_legacy_workspace_without_optional_artifacts_can_be_accepted(tmp_path: Path) -> None:
    ctx = make_context(tmp_path)
    save_json(ctx.store, StageName.STRUCTURE, "pages.json", [])
    write_json(ctx.store.stage_dir(StageName.STRUCTURE.value) / "blocks.json", [])

    valid, reasons = verify_stage_artifacts(ctx.store.root(), StageName.STRUCTURE, legacy_ok=True)

    assert valid, reasons

    # Unhashed checkpoints may be reused but are NOT labelled verified.
    assert checkpoint_verification_level(ctx.store.root(), StageName.STRUCTURE, None) == "legacy"


def test_all_checkpoint_states_are_distinguished(tmp_path: Path) -> None:
    ctx = make_context(tmp_path)
    save_json(ctx.store, StageName.EXTRACT, "pages.json", [{"page_index": 0}])
    save_json(ctx.store, StageName.EXTRACT, "results.json", [{"page_index": 0}])
    hashes = stage_artifact_hashes(ctx.store.root(), StageName.EXTRACT)

    assert checkpoint_verification_level(ctx.store.root(), StageName.EXTRACT, hashes) == "verified"
    assert checkpoint_verification_level(ctx.store.root(), StageName.EXTRACT, None) == "legacy"

    write_json(ctx.store.stage_dir(StageName.EXTRACT.value) / "results.json", {"changed": True})
    assert checkpoint_verification_level(ctx.store.root(), StageName.EXTRACT, hashes) == "invalid"

    (ctx.store.stage_dir(StageName.EXTRACT.value) / "pages.json").unlink()
    assert checkpoint_verification_level(ctx.store.root(), StageName.EXTRACT, hashes) == "invalid"
