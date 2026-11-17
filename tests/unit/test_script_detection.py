"""Script-detection tests (Task 3): Indian scripts + Devanagari Extended + Arabic.

Covers ``contains_indian_script`` (the gate that keeps the English-only
lesson planner honest) and its use inside plan validation:

- the nine required Brahmic scripts, char by char;
- Devanagari Extended (U+A8E0-U+A8FF);
- Arabic script (Urdu/``sd-Arab`` are supported target languages);
- mixed English/Indic text, nested dict/list LLM output, Unicode boundary
  cases, English punctuation and maths (must NOT be flagged), and valid
  English-only plans (must pass);
- unrelated Unicode scripts (Cyrillic, CJK, Hebrew, Thai, Greek) are never
  misclassified as Indian scripts.
"""

from __future__ import annotations

from typing import Any

import pytest

from bodhan.pipeline import PipelineError
from bodhan.pipeline.stages.compose import _validate_plan, contains_indian_script

# One representative character per required script.
REQUIRED_SCRIPT_CHARS = [
    ("Devanagari", "\u0915"),  # क
    ("Bengali", "\u0985"),  # অ
    ("Gurmukhi", "\u0a05"),  # ਅ
    ("Gujarati", "\u0a85"),  # અ
    ("Oriya", "\u0b05"),  # ଅ
    ("Tamil", "\u0b85"),  # அ
    ("Telugu", "\u0c05"),  # అ
    ("Kannada", "\u0c85"),  # ಅ
    ("Malayalam", "\u0d05"),  # അ
]

#: Text an "evil" LLM might plant; must be rejected at any nesting depth.
_HINDI = "यह हिंदी में लिखा गया पाठ है।"
_URDU = "یہ اردو ہے"  # Arabic script (Urdu is a supported target)
_DEVANAGARI_EXTENDED = "\ua8e0\ua8e1 x"  # U+A8E0-U+A8FF combining marks


def _plant(plan: dict[str, Any], path: str, value: str) -> None:
    """Set a dotted path (list indices as int segments) inside the plan dict."""
    node: Any = plan
    *parents, last = path.split(".")
    for part in parents:
        node = node[int(part)] if isinstance(node, list) else node[part]
    if isinstance(node, list):
        node[int(last)] = value
    else:
        node[last] = value


def _valid_plan() -> dict[str, Any]:
    """A fully valid English-only plan (must always pass validation)."""
    return {
        "title_en": "Photosynthesis in plants",
        "sections": [
            {"heading_en": "Overview", "block_ids": ["b1"]},
            {"heading_en": "Details", "block_ids": ["b2"]},
        ],
        "objectives_en": ["Define photosynthesis", "List the inputs"],
        "summary_en": "Plants convert light into chemical energy.",
        "glossary_en": [
            {"term_en": "Chlorophyll", "description_en": "The green pigment.", "block_id": "b1"}
        ],
        "exercises_en": [
            {
                "question_en": "What is chlorophyll?",
                "answer_en": "A green pigment in leaves.",
                "block_ids": ["b1", "b2"],
            }
        ],
        "narration_en": ["Speak slowly when introducing the term."],
    }


# --- individual characters ------------------------------------------------


@pytest.mark.parametrize(("script", "char"), REQUIRED_SCRIPT_CHARS)
def test_each_required_script_character_is_detected(script: str, char: str) -> None:
    assert contains_indian_script(char), script
    assert contains_indian_script(f"word {char} word"), script


def test_devanagari_extended_block_is_detected() -> None:
    # U+A8E0-U+A8FF: every code point in the block counts.
    for code in (0xA8E0, 0xA8E1, 0xA8F0, 0xA8FF):
        assert contains_indian_script(chr(code)), hex(code)
    # ...but the code points immediately outside do not.
    assert not contains_indian_script(chr(0xA8DF))
    assert not contains_indian_script(chr(0xA900))


def test_arabic_script_is_detected_for_urdu_support() -> None:
    # Basic Arabic letters (Urdu text) and presentation forms.
    assert contains_indian_script("یہ اردو ہے")
    assert contains_indian_script("\u0627\u0644\u0639\u0631\u0628\u064a\u0629")  # العربية
    assert contains_indian_script("\ufefb")  # presentation form (lam-alef)
    # Boundary of the Arabic ranges: Devanagari starts at U+0900, and the
    # unrelated block just below the Arabic ranges (Syriac Supplement) is not.
    assert contains_indian_script(chr(0x0600))
    assert contains_indian_script(chr(0x08FF))  # end of Arabic Extended-A
    assert contains_indian_script(chr(0x0900))  # ...immediately followed by Devanagari
    assert not contains_indian_script(chr(0x086F))  # Syriac Supplement (unrelated)


# --- mixed text, punctuation, maths ---------------------------------------


def test_mixed_english_and_indic_text_is_detected() -> None:
    assert contains_indian_script("The word अध्याय means chapter.")
    assert contains_indian_script("chapter का अर्थ है")
    assert not contains_indian_script("The quick brown fox jumps over the lazy dog.")


def test_english_punctuation_and_maths_are_not_detected() -> None:
    englishish = (
        "E = mc^2; 100% (a+b)^2 <= >= != +- sum sqrt infinity "
        "pi alpha beta \"quotes\" 'apostrophes' don't 3.14159 "
        "#hashtag @mention http://x.test/a?b=c&d=e [link](target) "
        "a,b;c:d | pipe ~ tilde `backtick` $5.00 42/7"
    )
    assert not contains_indian_script(englishish)
    assert not contains_indian_script(" Greek: \u03ba\u03b1\u03bb\u03b7\u03bc\u03ad\u03c1\u03b1")


def test_unrelated_unicode_scripts_are_never_flagged() -> None:
    unrelated = [
        "\u041f\u0440\u0438\u0432\u0435\u0442",  # Cyrillic (Russian)
        "\u4f60\u597d",  # CJK (Chinese)
        "\u05e9\u05dc\u05d5\u05dd",  # Hebrew
        "\u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35",  # Thai
        "\uac00\ud55c\uae00",  # Hangul (Korean)
        "\u1400\u1401",  # Canadian Aboriginal
        "\u1d400\U0001f600",  # math bold A + emoji (astral plane)
        "\u200b\u200d\ufeff",  # ZWSP / ZWJ / BOM (zero-width)
    ]
    for text in unrelated:
        assert not contains_indian_script(text), text


# --- plan validation: English-only plans pass ------------------------------


def test_valid_english_only_plan_passes() -> None:
    plan = _validate_plan(_valid_plan(), {"b1", "b2"})
    assert plan.title_en == "Photosynthesis in plants"
    assert [section.heading_en for section in plan.sections] == ["Overview", "Details"]
    assert plan.glossary_en[0].term_en == "Chlorophyll"


# --- plan validation: malicious nested LLM output is rejected -------------


@pytest.mark.parametrize(
    "plant_path",
    [
        "title_en",
        "sections.0.heading_en",
        "sections.1.heading_en",
        "objectives_en.0",
        "summary_en",
        "glossary_en.0.term_en",
        "glossary_en.0.description_en",
        "exercises_en.0.question_en",
        "exercises_en.0.answer_en",
        "narration_en.0",
    ],
)
def test_malicious_indic_text_in_nested_plan_is_rejected(plant_path: str) -> None:
    plan = _valid_plan()
    _plant(plan, plant_path, _HINDI)

    with pytest.raises(PipelineError, match="Indian-language text"):
        _validate_plan(plan, {"b1", "b2"})


def test_arabic_script_in_nested_plan_is_rejected() -> None:
    plan = _valid_plan()
    _plant(plan, "glossary_en.0.description_en", _URDU)

    with pytest.raises(PipelineError, match="Indian-language text"):
        _validate_plan(plan, {"b1", "b2"})


def test_devanagari_extended_char_in_nested_plan_is_rejected() -> None:
    plan = _valid_plan()
    _plant(plan, "sections.0.heading_en", _DEVANAGARI_EXTENDED)

    with pytest.raises(PipelineError, match="Indian-language text"):
        _validate_plan(plan, {"b1", "b2"})


def test_nested_dict_where_string_expected_is_rejected() -> None:
    """A dict hidden inside a text field can never slip past validation."""
    plan = _valid_plan()
    plan["title_en"] = {"deep": [_HINDI]}

    with pytest.raises(PipelineError, match="must be a string"):
        _validate_plan(plan, {"b1", "b2"})


def test_indic_text_in_any_list_item_is_rejected() -> None:
    for path in ("objectives_en.1", "narration_en.0"):
        plan = _valid_plan()
        _plant(plan, path, f"clean text {_HINDI}")
        with pytest.raises(PipelineError, match="Indian-language text"):
            _validate_plan(plan, {"b1", "b2"})
