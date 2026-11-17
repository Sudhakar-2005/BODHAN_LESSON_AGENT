"""OCR adapters.

Indic-OCR is the ONLY permitted OCR engine for this project.
"""

from bodhan.adapters.ocr.errors import OCRProviderError, OCRResponseError
from bodhan.adapters.ocr.indic_ocr import IndicOCRAdapter

__all__ = ["IndicOCRAdapter", "OCRProviderError", "OCRResponseError"]
