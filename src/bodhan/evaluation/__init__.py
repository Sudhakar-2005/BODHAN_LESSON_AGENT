"""Offline evaluation utilities (no API calls, no LLM, no provider access).

Phase 2B evaluates the exact narration text sent to Indic-Speak against the
verbatim Bodhan Indic-Transcribe output using a deterministic character-error
rate (CER) engine.
"""

from bodhan.evaluation.cer import (
    NORMALIZATION_POLICY,
    NORMALIZATION_POLICY_ID,
    CERResult,
    compute_cer,
    levenshtein_ops,
    normalize_text,
)
from bodhan.evaluation.evaluate import (
    REFERENCE_SOURCE,
    Evaluation,
    Thresholds,
    evaluate_sections,
    quality_label,
    write_evaluation,
)

__all__ = [
    "NORMALIZATION_POLICY",
    "NORMALIZATION_POLICY_ID",
    "REFERENCE_SOURCE",
    "CERResult",
    "Evaluation",
    "Thresholds",
    "compute_cer",
    "evaluate_sections",
    "levenshtein_ops",
    "normalize_text",
    "quality_label",
    "write_evaluation",
]
