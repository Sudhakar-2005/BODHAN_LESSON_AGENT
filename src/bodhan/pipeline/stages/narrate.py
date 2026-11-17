"""Stage 8 - narrate: text-to-speech via edge-tts.

The lesson markdown is split into sections at `##` headings; each section
becomes one MP3 in ``narrate/``. v1 emits per-segment files (no ffmpeg needed).
"""

from __future__ import annotations

import re
from typing import Any

from bodhan.domain.models import AudioSegment, Lesson, StageName
from bodhan.pipeline.errors import PipelineError
from bodhan.pipeline.stages.base import StageContext
from bodhan.pipeline.state import load_input_pages, load_json, save_json, save_pages

NAME = StageName.NARRATE

_HEADING = re.compile(r"^(#{1,3})\s+(.*)$", re.MULTILINE)
_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_MARKUP = re.compile(r"[*_`>#|]+")
_HTML_TAG = re.compile(r"<[^>]+>")


def split_sections(markdown: str) -> list[tuple[str, str]]:
    """Split lesson markdown into (heading, body) sections at `##` boundaries."""
    matches = list(_HEADING.finditer(markdown))
    if not matches:
        return []
    # Text before the first heading is an intro section.
    intro = markdown[: matches[0].start()].strip()
    sections: list[tuple[str, str]] = []
    if intro:
        sections.append(("परिचय", intro))
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        body = markdown[start:end].strip()
        if body:
            sections.append((match.group(2).strip(), body))
    return sections


def for_narration(text: str) -> str:
    """Strip markup that would be read aloud awkwardly by the TTS engine."""
    text = _CODE_FENCE.sub(" ", text)
    text = _HTML_TAG.sub(" ", text)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)  # bullets
    text = _INLINE_MARKUP.sub("", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)  # links -> label
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _load_lesson(ctx: StageContext) -> Lesson:
    payload = load_json(ctx.store, StageName.COMPOSE, "lesson.json")
    return Lesson.model_validate(payload)


async def run(ctx: StageContext) -> dict[str, Any]:
    """Synthesize one MP3 per lesson section and record the segments."""
    pages = load_input_pages(ctx.store, NAME)
    save_pages(ctx.store, NAME, pages)
    lesson = _load_lesson(ctx)
    sections = split_sections(lesson.markdown)
    if not sections:
        raise PipelineError("lesson has no narratable sections")

    out_dir = ctx.store.stage_dir(NAME.value)
    extension = ctx.tts.extension  # .wav for Indic-Speak, .mp3 for edge-tts
    segments: list[AudioSegment] = []
    skipped = 0

    for index, (heading, body) in enumerate(sections, start=1):
        spoken = for_narration(f"{heading}. {body}")
        if not spoken:
            skipped += 1
            continue
        segment_id = f"seg-{index:02d}"
        audio_path = out_dir / f"{segment_id}{extension}"
        segment = await ctx.tts.synthesize(
            spoken,
            segment_id=segment_id,
            lesson_id=lesson.lesson_id,
            output_path=audio_path,
        )
        segments.append(segment)
        ctx.logger.info("narrated %s (%d chars)", segment_id, len(spoken))

    if not segments:
        raise PipelineError("no audio segments produced")

    save_json(ctx.store, NAME, "segments.json", [s.model_dump() for s in segments])
    ctx.state["segments"] = segments
    return {
        "segments": len(segments),
        "skipped_empty": skipped,
        # Audit trail: which engines were configured and how often the
        # fallback actually served a segment (0 = primary handled all).
        "voice": ctx.settings.bodhan_speak_voice,
        "fallback_voice": ctx.settings.edge_tts_voice,
        "fallback_segments": ctx.tts.fallback_count,
    }
