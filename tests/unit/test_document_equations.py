"""Math-protection tests (Phase 3B-1 Task 5)."""

from __future__ import annotations

from bodhan.document import protect_math, restore_math


def test_inline_latex_is_protected_and_restored() -> None:
    text = "The value $x^2$ is positive."
    protected, mapping, uncertain = protect_math(text)

    assert uncertain is False
    assert "$x^2$" not in protected
    assert len(mapping) == 1
    assert restore_math(protected, mapping) == text


def test_display_equation_is_protected() -> None:
    text = "Consider $$\\frac{a}{b} = c$$ below."
    protected, mapping, uncertain = protect_math(text)

    assert uncertain is False
    assert "$$" not in protected
    assert restore_math(protected, mapping) == text


def test_fractions_subscripts_and_superscripts() -> None:
    text = r"Use \frac{1}{2} and H_2O and x^{10} here."
    protected, mapping, uncertain = protect_math(text)

    assert uncertain is False
    assert len(mapping) >= 3
    assert restore_math(protected, mapping) == text


def test_equation_mixed_with_natural_language() -> None:
    text = "Angle A = $90^\\circ$ and angle B is 30 degrees."
    protected, mapping, _ = protect_math(text)

    assert "angle B is 30 degrees." in protected  # natural text still translatable
    assert restore_math(protected, mapping) == text


def test_equation_inside_table_cell() -> None:
    cell = "<td>$x^2 + y^2 = z^2$</td>"
    protected, mapping, uncertain = protect_math(cell)

    assert uncertain is False
    assert restore_math(protected, mapping) == cell


def test_unbalanced_delimiters_are_reported() -> None:
    _, _, uncertain = protect_math("An unbalanced $formula")
    assert uncertain is True


def test_plain_text_is_translated_unchanged_by_protection() -> None:
    protected, mapping, uncertain = protect_math("ordinary sentence")
    assert protected == "ordinary sentence"
    assert mapping == {}
    assert uncertain is False
    assert restore_math(protected, mapping) == "ordinary sentence"
