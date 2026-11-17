"""Pipeline error taxonomy.

Kept in its own module (not ``bodhan.pipeline.__init__``) so stages can import
errors without triggering a circular import through the orchestrator.
"""

from __future__ import annotations

from bodhan.domain.models import StageName


class PipelineError(Exception):
    """Base class for pipeline failures."""


class StageNotImplementedError(PipelineError):
    """A stage has no implementation yet."""

    def __init__(self, stage: StageName) -> None:
        super().__init__(f"stage '{stage}' is not implemented yet")
        self.stage = stage
