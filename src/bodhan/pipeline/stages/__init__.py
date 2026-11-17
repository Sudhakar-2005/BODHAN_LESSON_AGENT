"""Pipeline stages.

Each stage module exposes::

    NAME: StageName
    async def run(ctx: StageContext) -> dict[str, Any]

and must be idempotent: re-running with a populated manifest skips completed
work. Stages are intentionally small and independently testable.
"""

from bodhan.domain.models import StageName

__all__ = ["StageName"]
