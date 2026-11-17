"""Full-lesson audio assembly (Phase 2C)."""

from bodhan.assembly.audio import (
    AssemblyResult,
    AudioAssemblyError,
    assemble_audio,
    require_ffmpeg,
)

__all__ = [
    "AssemblyResult",
    "AudioAssemblyError",
    "assemble_audio",
    "require_ffmpeg",
]
