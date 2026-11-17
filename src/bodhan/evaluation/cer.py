"""Deterministic Character Error Rate (CER) engine (Phase 2B, Task 2).

Reference and hypothesis are compared as *character sequences* only. No LLM,
translator, stemmer or any other semantic component is involved: this module
never rewrites, corrects, translates or removes content to improve a score.

    CER = (substitutions + deletions + insertions) / reference_character_count

Counts are measured on the normalized strings; see :data:`NORMALIZATION_POLICY`
for the exact, documented and deterministic normalization rules. CER is a
ratio (it may exceed 1.0 when insertions outnumber reference characters) and
``cer_percentage`` is simply ``cer * 100``.

Memory: the ordinary path stores the Levenshtein matrix as compact unsigned
shorts (2 bytes/cell, bounded by ``_FULL_MATRIX_CELL_LIMIT``); larger inputs
fall back to a Hirschberg divide-and-conquer alignment with linear memory.
Both paths return the same substitution/deletion/insertion breakdown.
"""

from __future__ import annotations

import unicodedata
from array import array
from dataclasses import dataclass
from typing import Any

#: Exact normalization policy applied to BOTH strings before comparison.
#: The id is recorded in every segment record; the full description is also
#: recorded in the evaluation manifest for auditability.
NORMALIZATION_POLICY: dict[str, str] = {
    "id": "nfc+ws-collapse-v1",
    "description": (
        "1. Unicode NFC normalization of both strings (canonical equivalence, "
        "e.g. a precomposed Devanagari vowel sign and its decomposed form compare equal); "
        "2. every run of Unicode whitespace (spaces, tabs, newlines, NBSP, ...) is collapsed "
        "to a single U+0020 space; 3. leading/trailing whitespace is stripped; "
        "4. comparison is case-sensitive - no case folding is applied; "
        "5. all other characters are preserved verbatim: punctuation (including the danda "
        "U+0964), digits, mathematical symbols, quotes, dashes and any remaining combining "
        "marks each count as characters; 6. no word or character is removed to improve the "
        "score, and no translation, transliteration or semantic rewriting is performed; "
        "7. counts are measured on the normalized strings; 8. Levenshtein ties are resolved "
        "deterministically (substitution, then deletion, then insertion)."
    ),
}
NORMALIZATION_POLICY_ID: str = NORMALIZATION_POLICY["id"]

#: Above this many matrix cells use the linear-memory Hirschberg path.
#: 10M cells * 2 bytes = ~20 MB upper bound for the ordinary path.
_FULL_MATRIX_CELL_LIMIT = 10_000_000
#: ``array("H")`` cannot hold distances above 65535; such extreme inputs also
#: take the Hirschberg path (which uses Python ints).
_MAX_USHORT_DISTANCE = 65_535


def normalize_text(text: str) -> str:
    """Apply the documented NFC + whitespace-collapse normalization policy."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def _distance_row(reference: str, hypothesis: str) -> list[int]:
    """Return ``dist(reference, hypothesis[:k])`` for every k (two-row DP)."""
    width = len(hypothesis)
    previous = list(range(width + 1))
    for i, reference_char in enumerate(reference, start=1):
        current = [i] + [0] * width
        for j, hypothesis_char in enumerate(hypothesis, start=1):
            cost = 0 if reference_char == hypothesis_char else 1
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
        previous = current
    return previous


def _ops_full(reference: str, hypothesis: str) -> tuple[int, int, int]:
    """Levenshtein S/D/I counts via a compact, fully backtrackable matrix."""
    rows = len(reference)
    cols = len(hypothesis)
    first = array("H", [0]) * (cols + 1)
    for j in range(1, cols + 1):
        first[j] = j
    matrix: list[array] = [first]

    for i, reference_char in enumerate(reference, start=1):
        previous = matrix[-1]
        current = array("H", [i]) * (cols + 1)
        for j, hypothesis_char in enumerate(hypothesis, start=1):
            cost = 0 if reference_char == hypothesis_char else 1
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
        matrix.append(current)

    i, j = rows, cols
    substitutions = deletions = insertions = 0
    while i > 0 and j > 0:
        diagonal = matrix[i - 1][j - 1]
        if reference[i - 1] == hypothesis[j - 1] and matrix[i][j] == diagonal:
            i -= 1
            j -= 1
        elif matrix[i][j] == diagonal + 1:
            substitutions += 1
            i -= 1
            j -= 1
        elif matrix[i][j] == matrix[i - 1][j] + 1:
            deletions += 1
            i -= 1
        else:
            insertions += 1
            j -= 1
    deletions += i
    insertions += j
    return substitutions, deletions, insertions


def _ops_hirschberg(reference: str, hypothesis: str) -> tuple[int, int, int]:
    """Levenshtein S/D/I counts with O(min(m, n)) memory (Hirschberg)."""
    rows = len(reference)
    cols = len(hypothesis)
    if rows == 0:
        return 0, 0, cols
    if cols == 0:
        return 0, rows, 0
    if rows == 1 or cols == 1:
        return _ops_full(reference, hypothesis)

    middle = rows // 2
    left = _distance_row(reference[:middle], hypothesis)
    # Reversed strings turn "distance to every suffix" into "distance to
    # every prefix", so only one more pair of rows is needed.
    right = _distance_row(reference[middle:][::-1], hypothesis[::-1])[::-1]
    split = min(range(cols + 1), key=lambda k: left[k] + right[k])

    first = _ops_hirschberg(reference[:middle], hypothesis[:split])
    second = _ops_hirschberg(reference[middle:], hypothesis[split:])
    return (
        first[0] + second[0],
        first[1] + second[1],
        first[2] + second[2],
    )


def levenshtein_ops(reference: str, hypothesis: str) -> tuple[int, int, int]:
    """Return ``(substitutions, deletions, insertions)`` for two raw strings.

    Deterministic tie-breaking is part of the normalization policy. The sum of
    the three counts always equals the Levenshtein edit distance.
    """
    rows = len(reference)
    cols = len(hypothesis)
    if rows == 0:
        return 0, 0, cols
    if cols == 0:
        return 0, rows, 0
    if reference == hypothesis:
        return 0, 0, 0  # avoids an O(m*n) pass over identical long segments
    if rows * cols <= _FULL_MATRIX_CELL_LIMIT and max(rows, cols) <= _MAX_USHORT_DISTANCE:
        return _ops_full(reference, hypothesis)
    return _ops_hirschberg(reference, hypothesis)


@dataclass(frozen=True)
class CERResult:
    """Complete, auditable CER calculation for one reference/hypothesis pair."""

    reference_character_count: int
    transcription_character_count: int
    substitutions: int
    deletions: int
    insertions: int
    edit_distance: int
    cer: float | None
    cer_percentage: float | None
    cer_unavailable_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return all eight required metrics plus an explicit reason field."""
        return {
            "reference_character_count": self.reference_character_count,
            "transcription_character_count": self.transcription_character_count,
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "edit_distance": self.edit_distance,
            "cer": self.cer,
            "cer_percentage": self.cer_percentage,
            "cer_unavailable_reason": self.cer_unavailable_reason,
        }


def compute_cer(reference: str, transcription: str) -> CERResult:
    """Compute CER for one pair; an empty reference yields ``cer=None``.

    No division by zero: when the normalized reference has no characters the
    ratio is undefined and an explicit reason is returned. The edit counts
    are still reported when they are well defined from the given strings.
    """
    reference_normalized = normalize_text(reference)
    transcription_normalized = normalize_text(transcription)
    reference_count = len(reference_normalized)
    transcription_count = len(transcription_normalized)

    if reference_count == 0:
        substitutions, deletions, insertions = 0, 0, transcription_count
        if transcription_count == 0:
            reason = (
                "CER unavailable: reference is empty after normalization "
                "(both reference and transcription are empty; division by zero)"
            )
        else:
            reason = (
                "CER unavailable: reference is empty after normalization "
                "(insertions cannot be normalized by a zero-length reference)"
            )
        return CERResult(
            reference_character_count=0,
            transcription_character_count=transcription_count,
            substitutions=substitutions,
            deletions=deletions,
            insertions=insertions,
            edit_distance=insertions,
            cer=None,
            cer_percentage=None,
            cer_unavailable_reason=reason,
        )

    substitutions, deletions, insertions = levenshtein_ops(
        reference_normalized, transcription_normalized
    )
    edit_distance = substitutions + deletions + insertions
    cer = edit_distance / reference_count
    return CERResult(
        reference_character_count=reference_count,
        transcription_character_count=transcription_count,
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        edit_distance=edit_distance,
        cer=cer,
        cer_percentage=cer * 100.0,
    )
