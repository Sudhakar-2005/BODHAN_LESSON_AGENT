"""Deterministic unit tests for the CER engine (Phase 2B Task 2).

No API, no network, no LLM, no live fixtures: every case is an inline string.
"""

from __future__ import annotations

import pytest

from bodhan.evaluation import cer as cer_module
from bodhan.evaluation.cer import (
    NORMALIZATION_POLICY,
    CERResult,
    compute_cer,
    levenshtein_ops,
    normalize_text,
)


def assert_counts(result: CERResult, *, reference: int, transcription: int, distance: int) -> None:
    """Shared invariant: S + D + I always equals the edit distance."""
    assert result.reference_character_count == reference
    assert result.transcription_character_count == transcription
    assert result.edit_distance == distance
    assert result.substitutions + result.deletions + result.insertions == result.edit_distance


# --- exact match and single edits -----------------------------------------


def test_exact_match_is_zero_cer() -> None:
    result = compute_cer("नमस्ते दुनिया", "नमस्ते दुनिया")

    assert_counts(result, reference=len("नमस्ते दुनिया"), transcription=len("नमस्ते दुनिया"), distance=0)
    assert (result.substitutions, result.deletions, result.insertions) == (0, 0, 0)
    assert result.cer == 0.0
    assert result.cer_percentage == 0.0
    assert result.cer_unavailable_reason is None


def test_one_substitution() -> None:
    result = compute_cer("abc", "axc")

    assert_counts(result, reference=3, transcription=3, distance=1)
    assert (result.substitutions, result.deletions, result.insertions) == (1, 0, 0)
    assert result.cer == pytest.approx(1 / 3)
    assert result.cer_percentage == pytest.approx(100 / 3)


def test_one_deletion() -> None:
    result = compute_cer("abc", "ac")

    assert_counts(result, reference=3, transcription=2, distance=1)
    assert (result.substitutions, result.deletions, result.insertions) == (0, 1, 0)
    assert result.cer == pytest.approx(1 / 3)


def test_one_insertion() -> None:
    result = compute_cer("abc", "abxc")

    assert_counts(result, reference=3, transcription=4, distance=1)
    assert (result.substitutions, result.deletions, result.insertions) == (0, 0, 1)
    assert result.cer == pytest.approx(1 / 3)


def test_multiple_edits_mixed_types() -> None:
    result = compute_cer("abcdef", "abXdf")

    assert_counts(result, reference=6, transcription=5, distance=2)
    assert (result.substitutions, result.deletions, result.insertions) == (1, 1, 0)
    assert result.cer == pytest.approx(2 / 6)


# --- empty / missing inputs ------------------------------------------------


def test_empty_transcription_is_all_deletions() -> None:
    result = compute_cer("abc", "")

    assert_counts(result, reference=3, transcription=0, distance=3)
    assert (result.substitutions, result.deletions, result.insertions) == (0, 3, 0)
    assert result.cer == 1.0
    assert result.cer_percentage == 100.0


def test_empty_reference_is_unavailable_not_zero_division() -> None:
    result = compute_cer("", "abc")

    assert_counts(result, reference=0, transcription=3, distance=3)
    assert (result.substitutions, result.deletions, result.insertions) == (0, 0, 3)
    assert result.cer is None
    assert result.cer_percentage is None
    assert result.cer_unavailable_reason is not None
    assert "empty" in result.cer_unavailable_reason


def test_both_empty_is_unavailable_with_explicit_reason() -> None:
    result = compute_cer("", "")

    assert_counts(result, reference=0, transcription=0, distance=0)
    assert result.cer is None
    assert result.cer_percentage is None
    assert "both reference and transcription" in (result.cer_unavailable_reason or "")


def test_whitespace_only_reference_is_unavailable_after_normalization() -> None:
    result = compute_cer(" \t\n ", "abc")

    assert result.reference_character_count == 0
    assert result.cer is None
    assert "empty after normalization" in (result.cer_unavailable_reason or "")


def test_as_dict_exposes_all_required_fields() -> None:
    payload = compute_cer("abc", "axc").as_dict()

    assert set(payload) == {
        "reference_character_count",
        "transcription_character_count",
        "substitutions",
        "deletions",
        "insertions",
        "edit_distance",
        "cer",
        "cer_percentage",
        "cer_unavailable_reason",
    }


# --- Unicode and languages -------------------------------------------------


def test_unicode_nfc_equivalence_is_zero_cer() -> None:
    decomposed = "क" + "\u093e"  # क + vowel sign aa
    composed = "\u0915\u093e"  # का
    assert normalize_text(decomposed) == normalize_text(composed)
    assert compute_cer(decomposed, composed).cer == 0.0

    assert compute_cer("e\u0301", "\u00e9").cer == 0.0  # e + combining acute == é


def test_combining_marks_are_counted_as_characters() -> None:
    result = compute_cer("\u0915\u093e", "\u0915\u093e\u0301")

    assert result.reference_character_count == 2
    assert result.transcription_character_count == 3
    assert (result.substitutions, result.deletions, result.insertions) == (0, 0, 1)
    assert result.cer == pytest.approx(1 / 2)


@pytest.mark.parametrize(
    ("reference", "changed"),
    [
        ("नमस्ते दुनिया", "नमस्ते दुनियाँ"),  # Hindi
        ("வணக்கம் உலகம்", "வணக்கம் உலகம்"),  # Tamil (exact)
        ("নমস্কার বিশ্ব", "নমস্কার বিশ্ব"),  # Bengali (exact)
    ],
)
def test_indic_language_texts(reference: str, changed: str) -> None:
    exact = compute_cer(reference, changed)
    assert exact.reference_character_count == len(reference)
    assert exact.cer is not None
    if reference == changed:
        assert exact.cer == 0.0
    else:
        assert exact.cer > 0

    edited = compute_cer(reference, reference[:-1] + "X")
    assert_counts(edited, reference=len(reference), transcription=len(reference), distance=1)
    assert edited.cer == pytest.approx(1 / len(reference))


def test_mixed_language_content() -> None:
    reference = "नमस्ते hello 123।"
    transcription = "नमस्ते hell0 123।"

    result = compute_cer(reference, transcription)

    assert result.cer is not None
    assert (result.substitutions, result.deletions, result.insertions) == (1, 0, 0)
    assert result.cer == pytest.approx(1 / len(reference))


def test_punctuation_is_preserved_and_counts() -> None:
    result = compute_cer("नमस्ते।", "नमस्ते,")

    assert (result.substitutions, result.deletions, result.insertions) == (1, 0, 0)
    assert result.cer == pytest.approx(1 / 7)


def test_digits_are_preserved_and_count() -> None:
    result = compute_cer("पृष्ठ 1", "पृष्ठ 2")

    assert result.reference_character_count == len("पृष्ठ 1")
    assert (result.substitutions, result.deletions, result.insertions) == (1, 0, 0)
    assert result.cer == pytest.approx(1 / len("पृष्ठ 1"))


def test_math_and_symbols_are_preserved() -> None:
    result = compute_cer("x = 2 + 3", "x = 2 - 3")

    assert (result.substitutions, result.deletions, result.insertions) == (1, 0, 0)
    assert result.cer == pytest.approx(1 / 9)


def test_whitespace_normalization() -> None:
    assert normalize_text("  a\t b\n\n c  ") == "a b c"
    assert normalize_text("a\u00a0b") == "a b"  # NBSP is Unicode whitespace
    assert compute_cer("  a\t b\n\n c  ", " a b c ").cer == 0.0


def test_case_is_sensitive() -> None:
    result = compute_cer("Hello", "hello")

    assert (result.substitutions, result.deletions, result.insertions) == (1, 0, 0)


# --- CER above 100% and long segments -------------------------------------


def test_cer_may_exceed_one_hundred_percent() -> None:
    result = compute_cer("ab", "abcdefgh")

    assert_counts(result, reference=2, transcription=8, distance=6)
    assert (result.substitutions, result.deletions, result.insertions) == (0, 0, 6)
    assert result.cer == 3.0
    assert result.cer_percentage == 300.0


def test_long_segment_completes_with_small_result() -> None:
    reference = "a" * 3000
    transcription = reference[:1500] + "b" + reference[1501:]

    result = compute_cer(reference, transcription)

    assert_counts(result, reference=3000, transcription=3000, distance=1)
    assert (result.substitutions, result.deletions, result.insertions) == (1, 0, 0)
    assert result.cer == pytest.approx(1 / 3000)

    # Identical long input takes the O(1) equality fast path.
    assert compute_cer(reference, reference).cer == 0.0


def test_edit_counts_sum_to_distance_for_many_pairs() -> None:
    pairs = [
        ("", "abc"),
        ("abc", ""),
        ("abc", "abc"),
        ("abc", "abxc"),
        ("abcdef", "abXdf"),
        ("पृष्ठ 1", "पृष्ठ वन"),
        ("हिन्दी and English 42!", "hindi और english 42।"),
        ("a" * 120, "b" * 97),
    ]
    for reference, transcription in pairs:
        substitutions, deletions, insertions = levenshtein_ops(reference, transcription)
        result = compute_cer(reference, transcription)
        assert substitutions + deletions + insertions == result.edit_distance
        if result.cer is not None:
            assert result.cer == pytest.approx(
                result.edit_distance / result.reference_character_count
            )


# --- normalization policy and the linear-memory fallback ------------------


def test_normalization_policy_is_documented_and_deterministic() -> None:
    assert NORMALIZATION_POLICY["id"] == "nfc+ws-collapse-v1"
    for keyword in ("NFC", "whitespace", "punctuation", "digits", "case-sensitive"):
        assert keyword in NORMALIZATION_POLICY["description"]
    assert normalize_text(normalize_text(" A\t B ")) == normalize_text(" A\t B ")


def test_hirschberg_fallback_matches_ordinary_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forcing the linear-memory path must not change the reported metrics."""
    fixtures = [
        ("abc", "axc"),
        ("abc", "ac"),
        ("abc", "abxc"),
        ("abcdef", "abXdf"),
        ("नमस्ते दुनिया", "नमस्ते दुनियाँ"),
        ("पृष्ठ 1", "पृष्ठ वन"),
        ("a" * 200, "a" * 100 + "b" + "a" * 99),
    ]
    ordinary = [compute_cer(reference, hypothesis) for reference, hypothesis in fixtures]

    monkeypatch.setattr(cer_module, "_FULL_MATRIX_CELL_LIMIT", 0)
    linear = [compute_cer(reference, hypothesis) for reference, hypothesis in fixtures]

    for expected, actual in zip(ordinary, linear, strict=True):
        assert expected.as_dict() == actual.as_dict()


def test_hirschberg_supports_inputs_beyond_ushort_distance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cer_module, "_FULL_MATRIX_CELL_LIMIT", 0)
    result = compute_cer("a" * 40, "b" * 70)

    assert result.edit_distance == 70
    assert (result.substitutions, result.deletions, result.insertions) == (40, 0, 30)
