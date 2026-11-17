"""Stage 1 - ingest: register the source document and enumerate its pages.

PDFs are rasterised to PNG page images (text and scanned pages alike); plain
image files are treated as single-page chapters; a folder of images becomes a
multi-page chapter (sorted by filename). Output: ``ingest/pages.json`` plus
``ingest/page-0001.png`` ...
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
    """Rasterise every PDF page; corrupt/encrypted files fail with clear errors."""
    import pymupdf

    out_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[tuple[int, str, int, int]] = []
    try:
        with pymupdf.open(source) as doc:
            if getattr(doc, "needs_pass", False):
                raise PipelineError(
                    f"PDF is password-protected and cannot be rasterised: {source.name} "
                    "(supply an unlocked copy)"
                )
            if doc.page_count == 0:
                raise PipelineError(f"PDF has no pages: {source.name}")
            zoom = dpi / 72.0
            matrix = pymupdf.Matrix(zoom, zoom)
            for index, pdf_page in enumerate(doc):
                try:
                    pixmap = pdf_page.get_pixmap(matrix=matrix, alpha=False)
                    target = out_dir / f"page-{index + 1:04d}.png"
                    pixmap.save(target)
                except Exception as exc:
                    raise PipelineError(
                        f"could not rasterise page {index + 1} of {source.name}: "
                        f"{type(exc).__name__}"
                    ) from exc
                rendered.append((index, str(target), pixmap.width, pixmap.height))
    except PipelineError:
        raise
    except Exception as exc:
        raise PipelineError(
            f"corrupted or unreadable PDF {source.name}: {type(exc).__name__}"
        ) from exc
    return rendered


async def run(ctx: StageContext) -> dict[str, Any]:
    """Open the source document and write page images + page metadata."""
    source = Path(ctx.chapter.source_path)
    if not source.exists():
        raise PipelineError(f"source file not found: {source}")

    out_dir = ctx.store.stage_dir(NAME.value)
    suffix = source.suffix.lower()

    if source.is_dir():
        # Image folder: one page per image, sorted by filename.
        images = sorted(
            (p for p in source.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES),
            key=lambda p: p.name.lower(),
        )
        if not images:
            raise PipelineError(
                f"folder contains no supported page images ({', '.join(sorted(_IMAGE_SUFFIXES))}): "
                f"{source}"
            )
        rendered: list[tuple[int, str, int, int]] = []
        skipped: list[str] = []
        for image in images:
            try:
                width, height = image_size(image)
                target = out_dir / f"page-{len(rendered) + 1:04d}{image.suffix.lower()}"
                if target.resolve() != image.resolve():
                    target.write_bytes(image.read_bytes())
            except (OSError, ValueError) as exc:
                skipped.append(f"{image.name} ({type(exc).__name__})")
                continue
            rendered.append((len(rendered), str(target), width, height))
        if skipped:
            ctx.logger.warning(
                "skipped %d unreadable image(s): %s", len(skipped), ", ".join(skipped)
            )
        if not rendered:
            raise PipelineError(
                "folder contains no readable page images; skipped: " + ", ".join(skipped)
            )
    elif suffix == ".pdf":
        rendered = render_pdf_pages(source, out_dir, dpi=ctx.settings.render_dpi)
    elif suffix in _IMAGE_SUFFIXES:
        try:
            width, height = image_size(source)
        except (OSError, ValueError) as exc:
            raise PipelineError(
                f"corrupted or unreadable image {source.name}: {type(exc).__name__}"
            ) from exc
        target = out_dir / f"page-0001{suffix}"
        if target.resolve() != source.resolve():
            target.write_bytes(source.read_bytes())
        rendered = [(0, str(target), width, height)]
    else:
        raise PipelineError(
            f"unsupported source type '{suffix}' (expect a PDF, an image, or a folder of images)"
        )

    if len(rendered) > ctx.settings.max_input_pages:
        raise PipelineError(
            f"document has {len(rendered)} pages; the supported range is "
            f"{ctx.settings.min_input_pages}-{ctx.settings.max_input_pages}"
        )
    if len(rendered) < ctx.settings.min_input_pages:
        message = (
            f"document has {len(rendered)} pages; the supported range is "
            f"{ctx.settings.min_input_pages}-{ctx.settings.max_input_pages}"
        )
        if ctx.settings.enforce_page_count_limits:
            raise PipelineError(message)
        ctx.logger.warning("page_count_below_supported_range status=warning detail=%s", message)

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
