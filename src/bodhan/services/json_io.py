"""Shared atomic JSON writer for Phase 2C generated artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write UTF-8 JSON via a temp file + atomic replace (no partial artifacts)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
