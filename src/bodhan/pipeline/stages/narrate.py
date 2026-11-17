"""Stage 8 - narrate: text-to-speech via Bodhan Indic-Speak (the only engine).

The lesson markdown is split into sections at `##` headings; each section
becomes one audio file in ``narrate/``. Indic-Speak (WAV) is the sole
narration engine - the assignment restricts alternative TTS providers, so
no fallback engine exists and none can ever be invoked.

Failure handling:
- transient failures are retried inside the adapter (bounded exponential
  backoff, honouring the documented ``retry-after`` window);
- when narration still fails, the segments completed so far are preserved
  (``segments.json`` + their audio files) and the failure is recorded in
  structured output (``status.json``) before the stage fails loudly;
- a re-run resumes at the first missing segment instead of re-speaking
  the whole lesson.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from bodhan.domain.models import AudioSegment, Lesson, StageName
from bodhan.observability import emit_event
from bodhan.pipeline.errors import PipelineError
from bodhan.pipeline.stages.base import StageContext
from bodhan.pipeline.state import load_input_pages, load_json, save_json, save_pages

NAME = StageName.NARRATE

_HEADING = re.compile(r"^(#{1,3})\s+(.*)$", re.MULTILINE)
_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_MARKUP = re.compile(r"[*_`>#|]+")
_HTML_TAG = re.compile(r"<[^>]+>")


def split_sections(markdown: str, *, intro_heading: str | None = None) -> list[tuple[str, str]]:
    """Split lesson markdown into (heading, body) sections at `##` boundaries.

    Text before the first heading becomes a section only when the caller
    supplies ``intro_heading`` (the translated "Introduction" label from the
    lesson scaffold). Without one the preamble is skipped - section labels
    are never hardcoded in any language.
    """
    matches = list(_HEADING.finditer(markdown))
    if not matches:
        return []
    # Text before the first heading is a preamble section.
    intro = markdown[: matches[0].start()].strip()
    sections: list[tuple[str, str]] = []
    if intro and intro_heading:
        sections.append((intro_heading, intro))
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        body = markdown[start:end].strip()
        if body:
            sections.append((match.group(2).strip(), body))
    return sections


def _table_rows_to_speech(text: str) -> str:
    """Convert Markdown table rows into ordered, comma-separated cell speech.

    The header row is spoken first, then each data row in order; separator
    rows are dropped. This preserves row/column relationships without adding
    any generated language (all words still come from Indic-Translate).
    """
    lines = text.splitlines()
    output: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if line.startswith("|") and line.endswith("|"):
            rows: list[str] = []
            while index < len(lines):
                candidate = lines[index].strip()
                if not (candidate.startswith("|") and candidate.endswith("|")):
                    break
                rows.append(candidate.replace("\\|", "\ue000"))
                index += 1
            for row in rows:
                cells = [cell.strip().replace("\ue000", "|") for cell in row[1:-1].split("|")]
                if all(re.fullmatch(r":?-{3,}:?", cell or "-") for cell in cells):
                    continue  # Markdown alignment row is not content
                if cells:
                    output.append(", ".join(cells))
            continue
        output.append(lines[index])
        index += 1
    return "\n".join(output)


def for_narration(text: str) -> str:
    """Strip markup that would be read aloud awkwardly by the TTS engine."""
    text = _CODE_FENCE.sub(" ", text)
    text = _HTML_TAG.sub(" ", text)
    text = _table_rows_to_speech(text)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)  # bullets
    text = _INLINE_MARKUP.sub("", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)  # links -> label
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _load_lesson(ctx: StageContext) -> Lesson:
    payload = load_json(ctx.store, StageName.COMPOSE, "lesson.json")
    return Lesson.model_validate(payload)


def _load_preserved(ctx: StageContext) -> dict[str, AudioSegment]:
    """Segments an earlier (failed) attempt completed successfully.

    Only entries whose audio file still exists are reused - a missing or
    deleted file forces that segment to be re-synthesized. A missing or
    corrupt ``segments.json`` simply means nothing is preserved.
    """
    segments_path = ctx.store.stage_dir(NAME.value) / "segments.json"
    if not segments_path.is_file():
        return {}
    try:
        payload = load_json(ctx.store, NAME, "segments.json")
    except PipelineError:
        return {}
    preserved: dict[str, AudioSegment] = {}
    for item in payload if isinstance(payload, list) else []:
        try:
            segment = AudioSegment.model_validate(item)
        except ValueError:
            continue
        if Path(segment.audio_path).is_file():
            preserved[segment.segment_id] = segment
    return preserved


async def run(ctx: StageContext) -> dict[str, Any]:
    """Synthesize one audio file per lesson section and record the segments."""
    pages = load_input_pages(ctx.store, NAME)
    save_pages(ctx.store, NAME, pages)
    lesson = _load_lesson(ctx)
    sections = split_sections(lesson.markdown, intro_heading=lesson.label("introduction") or None)
    if not sections:
        raise PipelineError("lesson has no narratable sections")

    out_dir = ctx.store.stage_dir(NAME.value)
    extension = ctx.tts.extension
    preserved = _load_preserved(ctx)

    segments: list[AudioSegment] = []
    skipped = 0
    failure: tuple[str, BaseException] | None = None

    for index, (heading, body) in enumerate(sections, start=1):
        segment_id = f"seg-{index:02d}"
        prior = preserved.get(segment_id)
        if prior is not None:
            segments.append(prior)
            ctx.logger.info("preserved %s from an earlier run", segment_id)
            continue
        spoken = for_narration(f"{heading}. {body}")
        if not spoken:
            skipped += 1
            continue
        audio_path = out_dir / f"{segment_id}{extension}"
        try:
            segment = await ctx.tts.synthesize(
                spoken,
                segment_id=segment_id,
                lesson_id=lesson.lesson_id,
                output_path=audio_path,
            )
        except Exception as exc:  # single engine: record where narration stopped
            failure = (segment_id, exc)
            break
        segments.append(segment)
        emit_event(
            ctx.logger,
            "narration_segment_generated",
            run_id=ctx.chapter.chapter_id,
            stage=NAME.value,
            target_language=ctx.chapter.target_language_code,
            section_id=segment_id,
            chars=len(spoken),
        )

    # Structured record: what completed, what was skipped, and - on
    # failure - exactly where narration stopped (audit + resume input).
    status: dict[str, Any] = {
        "status": "failed" if failure else "done",
        "tts_provider": type(ctx.tts).__name__,
        "voice": ctx.settings.bodhan_speak_voice,
        "total_sections": len(sections),
        "completed_segments": [s.segment_id for s in segments],
        "skipped_empty": skipped,
    }
    if failure is not None:
        failed_segment, failed_error = failure
        status["failed_segment"] = failed_segment
        status["error"] = f"{type(failed_error).__name__}: {failed_error}"

    if segments:
        # Preserve completed audio metadata before reporting the failure.
        save_json(ctx.store, NAME, "segments.json", [s.model_dump() for s in segments])
    save_json(ctx.store, NAME, "status.json", status)

    if failure is not None:
        failed_segment, failed_error = failure
        raise PipelineError(
            f"narration failed at {failed_segment}: {type(failed_error).__name__}: "
            f"{failed_error} ({len(segments)} completed segment(s) preserved; "
            "re-run to resume)"
        ) from failed_error

    if not segments:
        raise PipelineError("no audio segments produced")

    ctx.state["segments"] = segments
    return {
        "segments": len(segments),
        "skipped_empty": skipped,
        # Audit trail: the only engine configured for this run.
        "tts_provider": type(ctx.tts).__name__,
        "voice": ctx.settings.bodhan_speak_voice,
    }
