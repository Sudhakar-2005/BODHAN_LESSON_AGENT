"""Stage 2 - preprocess: deskew, de-noise and normalise page images.

All image maths lives in module-level pure functions so they can be unit
tested without the pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from bodhan.domain.models import Page, StageName
from bodhan.pipeline.errors import PipelineError
from bodhan.pipeline.stages.base import StageContext
from bodhan.pipeline.state import load_input_pages, save_pages

NAME = StageName.PREPROCESS


def estimate_skew_angle(gray: np.ndarray) -> float:
    """Estimate page skew in degrees using the dominant text-line direction."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(binary > 0))
    if coords.size == 0:
        return 0.0
    angle = cv2.minAreaRect(coords)[-1]
    # cv2 returns angles in [-90, 0); normalise to a small signed skew.
    if angle < -45.0:
        angle = 90.0 + angle
    return float(angle)


def deskew(gray: np.ndarray, angle: float) -> tuple[np.ndarray, float]:
    """Rotate ``gray`` by ``angle`` degrees (counter-clockwise) about its centre."""
    if abs(angle) < 0.3:
        return gray, 0.0
    height, width = gray.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    rotated = cv2.warpAffine(
        gray,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, angle


def clean_page(image_bgr: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Full cleanup: grayscale -> denoise -> CLAHE -> deskew -> light threshold."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    denoised = cv2.medianBlur(gray, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrasted = clahe.apply(denoised)

    angle = estimate_skew_angle(contrasted)
    deskewed, applied = deskew(contrasted, angle)

    # Gentle adaptive threshold keeps scanned text crisp without destroying
    # equations/diagrams (pure global threshold tends to erase both).
    binary = cv2.adaptiveThreshold(
        deskewed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )
    return binary, {"skew_deg": round(applied, 3)}


async def run(ctx: StageContext) -> dict[str, Any]:
    """Clean every ingested page image and persist the cleaned copies."""
    pages = load_input_pages(ctx.store, NAME)
    out_dir = ctx.store.stage_dir(NAME.value)

    processed: list[Page] = []
    stats: dict[str, Any] = {"pages": len(pages), "skewed_pages": 0}

    for page in pages:
        if not page.image_path:
            raise PipelineError(f"page {page.page_index} has no image (run ingest)")
        source = Path(page.image_path)
        image = cv2.imread(str(source))
        if image is None:
            raise PipelineError(f"failed to read page image: {source}")

        cleaned, page_stats = clean_page(image)
        target = out_dir / f"page-{page.page_index + 1:04d}.png"
        if not cv2.imwrite(str(target), cleaned):
            raise PipelineError(f"failed to write page image: {target}")

        if page_stats["skew_deg"]:
            stats["skewed_pages"] += 1
        processed.append(
            page.model_copy(
                update={
                    "image_path": str(target),
                    "width": int(cleaned.shape[1]),
                    "height": int(cleaned.shape[0]),
                }
            )
        )

    save_pages(ctx.store, NAME, processed)
    ctx.state["pages"] = processed
    return stats
