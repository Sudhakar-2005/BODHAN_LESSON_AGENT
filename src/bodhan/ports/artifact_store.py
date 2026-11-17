"""Artifact store port: where stage inputs/outputs and the manifest live."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class ArtifactStore(ABC):
    """Filesystem-backed persistence for a single chapter's work directory."""

    @abstractmethod
    def root(self) -> Path:
        """Root directory of this chapter's workspace."""
        raise NotImplementedError

    @abstractmethod
    def stage_dir(self, stage: str) -> Path:
        """Directory holding the artifacts of ``stage`` (created on demand)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def manifest_path(self) -> Path:
        """Path of the JSON manifest file for this workspace."""
        raise NotImplementedError

    @abstractmethod
    def read_manifest(self) -> dict[str, Any]:
        """Return the current manifest, or ``{}`` if none exists yet."""
        raise NotImplementedError

    @abstractmethod
    def write_manifest(self, manifest: dict[str, Any]) -> None:
        """Atomically persist the manifest."""
        raise NotImplementedError
