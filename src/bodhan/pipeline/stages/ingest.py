"""Stage 1 - ingest: register the source document and enumerate its pages.

PDFs are rasterised to PNG page images (text and scanned pages alike); plain
image files are treated as single-page chapters. Output: ``ingest/pages.json``
plus ``ingest/page-0001.png`` ...
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bodhan.domain.models import Page, StageName
from bodhan.pipeline.errors import PipelineError
from bodhan.pipeline.stages.base import StageContext
from bodhan.pipeline.state import save_pages

NAME = StageName.INGEST

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def render_pdf_pages(
    source: Path, out_dir: Path, *, dpi: int = 200
) -> list[tuple[int, str, int, int]]:
    """Rasterise every PDF page; returns (index, path, width, height) tuples."""
    import pymupdf

    rendered: list[tuple[int, str, int, int]] = []
    with pymupdf.open(source) as doc:
        if doc.page_count == 0:
            raise PipelineError(f"PDF has no pages: {source}")
        zoom = dpi / 72.0
        matrix = pymupdf.Matrix(zoom, zoom)
        for index, pdf_page in enumerate(doc):
            pixmap = pdf_page.get_pixmap(matrix=matrix, alpha=False)
            target = out_dir / f"page-{index + 1:04d}.png"
            pixmap.save(target)
            rendered.append((index, str(target), pixmap.width, pixmap.height))
    return rendered


async def run(ctx: StageContext) -> dict[str, Any]:
    """Open the source document and write page images + page metadata."""
    source = Path(ctx.chapter.source_path)
    if not source.is_file():
        raise PipelineError(f"source file not found: {source}")

    out_dir = ctx.store.stage_dir(NAME.value)
    suffix = source.suffix.lower()

    if suffix == ".pdf":
        rendered = render_pdf_pages(source, out_dir, dpi=ctx.settings.render_dpi)
    elif suffix in _IMAGE_SUFFIXES:
        target = out_dir / f"page-0001{suffix}"
        if target.resolve() != source.resolve():
            target.write_bytes(source.read_bytes())
        width, height = image_size(target)
        rendered = [(0, str(target), width, height)]
    else:
        raise PipelineError(f"unsupported source type '{suffix}' (expect PDF or image)")

    pages = [
        Page(
            page_index=index,
            source_path=str(source),
            image_path=path,
            width=width,
            height=height,
        )
        for index, path, width, height in rendered
    ]
    save_pages(ctx.store, NAME, pages)
    ctx.chapter.pages = pages
    ctx.state["pages"] = pages
    ctx.logger.info("ingested %d page(s) from %s", len(pages), source.name)
    return {"pages": len(pages), "source": source.name}


def image_size(path: Path) -> tuple[int, int]:
    """Return (width, height) of an image file."""
    from PIL import Image

    with Image.open(path) as img:
        return img.size
