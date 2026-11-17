"""Tests for the filesystem artifact store and manifest checkpointing."""

from __future__ import annotations

import json
from pathlib import Path

from bodhan.services.artifact_store import FileArtifactStore


def test_stage_dir_is_created(tmp_path: Path) -> None:
    store = FileArtifactStore(tmp_path / "chapter-01")
    path = store.stage_dir("extract")
    assert path.is_dir()
    assert path == tmp_path / "chapter-01" / "extract"


def test_manifest_roundtrip(tmp_path: Path) -> None:
    store = FileArtifactStore(tmp_path / "chapter-01")
    assert store.read_manifest() == {}

    manifest = {"chapter_id": "chapter-01", "stages": {"ingest": {"status": "done"}}}
    store.write_manifest(manifest)

    assert store.read_manifest() == manifest
    assert json.loads(store.manifest_path.read_text(encoding="utf-8")) == manifest


def test_corrupt_manifest_returns_empty_dict(tmp_path: Path) -> None:
    store = FileArtifactStore(tmp_path / "chapter-01")
    store.root().joinpath("manifest.json").write_text("{ not json", encoding="utf-8")
    assert store.read_manifest() == {}


def test_no_temp_files_left_behind(tmp_path: Path) -> None:
    store = FileArtifactStore(tmp_path / "chapter-01")
    store.write_manifest({"a": 1})
    leftovers = [p for p in store.root().iterdir() if p.suffix == ".tmp"]
    assert leftovers == []
