"""Stage-to-stage data flow helpers.

Each stage reads its input from the previous stage's directory and writes its
own ``pages.json`` / artifact files, so any stage can be re-run in isolation
(`bodhan run <src> --only <stage>`) without keeping state in memory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from bodhan.domain.models import Page, StageName
from bodhan.pipeline.errors import PipelineError
from bodhan.ports.artifact_store import ArtifactStore

#: Which stage feeds each stage (None = reads the raw source file).
INPUT_OF: dict[StageName, StageName | None] = {
    StageName.INGEST: None,
    StageName.PREPROCESS: StageName.INGEST,
    StageName.EXTRACT: StageName.PREPROCESS,
    StageName.STRUCTURE: StageName.EXTRACT,
    StageName.QUALITY: StageName.STRUCTURE,
    StageName.TRANSLATE: StageName.QUALITY,
    StageName.COMPOSE: StageName.TRANSLATE,
    StageName.NARRATE: StageName.COMPOSE,
    StageName.TRANSCRIBE: StageName.NARRATE,
    StageName.EXPORT: StageName.NARRATE,
}


def save_pages(store: ArtifactStore, stage: StageName, pages: list[Page]) -> None:
    path = store.stage_dir(stage.value) / "pages.json"
    atomic_write_text(
        path, json.dumps([p.model_dump() for p in pages], ensure_ascii=False, indent=2)
    )


def atomic_write_text(path: Path, text: str) -> None:
    """Write a file atomically (temp file + replace) so readers never see partials."""
    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def load_pages(store: ArtifactStore, stage: StageName) -> list[Page]:
    path = store.stage_dir(stage.value) / "pages.json"
    if not path.is_file():
        raise PipelineError(f"missing {path} - run stage '{stage.value}' first")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return [Page.model_validate(item) for item in raw]
    except (json.JSONDecodeError, ValidationError) as exc:
        raise PipelineError(f"corrupt pages file: {path}") from exc


def load_input_pages(store: ArtifactStore, stage: StageName) -> list[Page]:
    """Load pages produced by the stage that feeds ``stage``."""
    source = INPUT_OF.get(stage)
    if source is None:
        raise PipelineError(f"stage '{stage.value}' has no upstream stage")
    return load_pages(store, source)


def save_json(store: ArtifactStore, stage: StageName, name: str, data: Any) -> None:
    path = store.stage_dir(stage.value) / name
    if isinstance(data, BaseModel):
        text = data.model_dump_json(indent=2)
    elif isinstance(data, list) and data and isinstance(data[0], BaseModel):
        text = json.dumps([m.model_dump() for m in data], ensure_ascii=False, indent=2)
    else:
        text = json.dumps(data, ensure_ascii=False, indent=2)
    atomic_write_text(path, text)


def load_json(store: ArtifactStore, stage: StageName, name: str) -> Any:
    path = store.stage_dir(stage.value) / name
    if not path.is_file():
        raise PipelineError(f"missing {path} - run stage '{stage.value}' first")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PipelineError(f"corrupt json file: {path}") from exc


def load_typed_list(
    store: ArtifactStore,
    stage: StageName,
    name: str,
    model: type[BaseModel],
) -> list[BaseModel]:
    """Load a JSON array and validate every element against ``model``."""
    raw = load_json(store, stage, name)
    if not isinstance(raw, list):
        raise PipelineError(f"{name} in stage '{stage.value}' is not a list")
    try:
        return [model.model_validate(item) for item in raw]
    except ValidationError as exc:
        raise PipelineError(f"invalid records in {stage.value}/{name}") from exc
