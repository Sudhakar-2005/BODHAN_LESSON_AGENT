"""Stage 10 - export: assemble the final deliverables under data/exports/.

Produces per chapter::

    data/exports/<chapter>/
        lesson.md            # translated lesson
        lesson.html          # rendered with embedded audio players
        audio/seg-NN.<ext>   # narration segments (.wav from Indic-Speak,
                             #  the only narration engine)
        quality_report.json  # every flagged OCR block + reason
        manifest.json        # copy of the run manifest (audit trail)

All visible labels in ``lesson.html`` come from the lesson scaffold
(translated via Indic-Translate) - nothing is hardcoded per language.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from jinja2 import Template

from bodhan.domain.models import AudioSegment, Lesson, StageName
from bodhan.pipeline.errors import PipelineError
from bodhan.pipeline.stages.base import StageContext
from bodhan.pipeline.state import load_input_pages, load_json, save_pages
from bodhan.services.json_io import write_json_atomic

NAME = StageName.EXPORT

_HTML_TEMPLATE = Template(
    """<!DOCTYPE html>
<html lang="{{ lesson.target_language_code }}">
<head>
  <meta charset="utf-8">
  <title>{{ lesson.title }}</title>
  <style>
    body { font-family: "Noto Sans", "Segoe UI", sans-serif; max-width: 46rem;
           margin: 2rem auto; padding: 0 1rem; line-height: 1.7; color: #1a1a1a; }
    h1 { border-bottom: 2px solid #2c5f8a; padding-bottom: .4rem; }
    .flag { background: #fff6e5; border-left: 4px solid #e8a33d; padding: .4rem .8rem;
            font-size: .85rem; margin: .2rem 0; }
    audio { width: 100%; margin: .4rem 0 1.2rem; }
    .segment h2 { margin-bottom: .2rem; }
  </style>
</head>
<body>
  <h1>{{ lesson.title }}</h1>
  {% if flagged %}
  <div class="flag">⚠ {{ lesson.label("quality_warning") }}
  <code>quality_report.json</code></div>
  {% endif %}
  <div class="markdown">{{ body_html | safe }}</div>
  {% if audio_label %}
  <h2>{{ audio_label }}</h2>
  {% endif %}
  {% for player in players %}
  <div class="segment">
    <audio controls preload="none" src="audio/{{ player.file }}"></audio>
  </div>
  {% endfor %}
</body>
</html>
"""
)


def _markdown_to_html(markdown: str) -> str:
    """Minimal, dependency-free markdown rendering (headings/lists/paragraphs)."""
    import html as html_mod
    import re

    lines = markdown.splitlines()
    out: list[str] = []
    in_list = False
    for line in lines:
        stripped = line.strip()
        heading = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if heading:
            if in_list:
                out.append("</ul>")
                in_list = False
            level = len(heading.group(1))
            out.append(f"<h{level}>{html_mod.escape(heading.group(2))}</h{level}>")
        elif re.match(r"^[-*+]\s+", stripped):
            if not in_list:
                out.append("<ul>")
                in_list = True
            item = re.sub(r"^[-*+]\s+", "", stripped)
            out.append(f"<li>{html_mod.escape(item)}</li>")
        elif not stripped:
            if in_list:
                out.append("</ul>")
                in_list = False
        else:
            if in_list:
                out.append("</ul>")
                in_list = False
            text = html_mod.escape(stripped)
            text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
            text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
            out.append(f"<p>{text}</p>")
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


async def run(ctx: StageContext) -> dict[str, Any]:
    """Render markdown/HTML, copy audio, and publish the quality report."""
    pages = load_input_pages(ctx.store, NAME)
    save_pages(ctx.store, NAME, pages)
    lesson = Lesson.model_validate(load_json(ctx.store, StageName.COMPOSE, "lesson.json"))
    segments = [
        AudioSegment.model_validate(item)
        for item in load_json(ctx.store, StageName.NARRATE, "segments.json")
    ]
    if not lesson.markdown or not segments:
        raise PipelineError("missing lesson or audio - run narrate first")

    export_dir = ctx.settings.exports_dir / ctx.chapter.chapter_id
    export_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = export_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for segment in segments:
        source = Path(segment.audio_path)
        if source.is_file():
            shutil.copy2(source, audio_dir / source.name)
            copied += 1

    (export_dir / "lesson.md").write_text(lesson.markdown, encoding="utf-8")
    players = [
        {"id": segment.segment_id, "file": Path(segment.audio_path).name}
        for segment in segments
        if Path(segment.audio_path).is_file()
    ]
    (export_dir / "lesson.html").write_text(
        _HTML_TEMPLATE.render(
            lesson=lesson,
            body_html=_markdown_to_html(lesson.markdown),
            players=players,
            flagged=len(lesson.flagged_block_ids),
            audio_label=lesson.label("audio"),
        ),
        encoding="utf-8",
    )

    quality_path = ctx.store.stage_dir(StageName.QUALITY.value) / "quality_report.json"
    if quality_path.is_file():
        shutil.copy2(quality_path, export_dir / "quality_report.json")

    return {
        "export_dir": str(export_dir),
        "audio_files": copied,
        "flagged_blocks": len(lesson.flagged_block_ids),
    }


def write_export_manifest(ctx: StageContext, manifest: dict[str, Any]) -> None:
    """Publish the finalized manifest after the orchestrator records export.

    Copying inside ``run()`` captured a snapshot that predated the export
    stage's own entry; writing here keeps the exported manifest consistent with
    the work manifest without hardcoding any stage status.
    """
    path = ctx.settings.exports_dir / ctx.chapter.chapter_id / "manifest.json"
    write_json_atomic(path, manifest)
