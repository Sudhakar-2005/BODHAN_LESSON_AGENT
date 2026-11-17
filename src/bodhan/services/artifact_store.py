"""Artifact store implementation: a plain directory + JSON manifest."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from bodhan.ports.artifact_store import ArtifactStore


class FileArtifactStore(ArtifactStore):
    """Stores one chapter's stage outputs under ``<root>/<stage>/``.

    The manifest lives at ``<root>/manifest.json`` and is written atomically so
    a crash mid-write never corrupts checkpoint state.
    """

    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    def root(self) -> Path:
        return self._ensure_dir(self._root)

    @staticmethod
    def _ensure_dir(path: Path) -> Path:
        if path.exists() and not path.is_dir():
            raise NotADirectoryError(f"artifact path exists but is not a directory: {path}")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def stage_dir(self, stage: str) -> Path:
        if not stage or Path(stage).name != stage:
            raise ValueError(f"invalid stage name: {stage!r}")
        path = self.root() / stage
        return self._ensure_dir(path)

    @property
    def manifest_path(self) -> Path:
        return self.root() / "manifest.json"

    def read_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.is_file():
            return {}
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        return data if isinstance(data, dict) else {}

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        self.root()
        # Atomic write: temp file in the same directory, then replace.
        fd, tmp_name = tempfile.mkstemp(dir=self._root, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, ensure_ascii=False, indent=2)
            os.replace(tmp_name, self.manifest_path)
        except BaseException:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
            raise
