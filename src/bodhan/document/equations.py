"""Protect mathematical tokens from machine translation (Phase 3B-1 Task 5).

Pure string handling: the protected spans are replaced by private-use
placeholders before a Bodhan Indic-Translate call and restored verbatim
afterwards. No mathematical token is ever sent to the translator, and no LLM
is involved. Unbalanced delimiters are reported instead of being silently
"fixed" by translation.
"""

from __future__ import annotations

import re

# Ordered longest-first: display math, inline math, LaTeX commands, script
# tokens. Bare numbers and ordinary words are deliberately NOT protected.
_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\$\$.+?\$\$", re.DOTALL),
    re.compile(r"\\\[.+?\\\]", re.DOTALL),
    re.compile(r"\$[^$\n]+\$"),
    re.compile(r"\\\(.+?\\\)", re.DOTALL),
    re.compile(r"\\[A-Za-z]+\{[^}]*\}(?:\{[^}]*\})?"),
    re.compile(r"\b[A-Za-z0-9]+(?:_\{[^}]*\}|_[A-Za-z0-9]|\^\{[^}]*\}|\^[A-Za-z0-9])"),
)
_MARKER = re.compile(r"\ue000(\d+)\ue001")


def _marker(index: int) -> str:
    return f"\ue000{index}\ue001"


def protect_math(text: str) -> tuple[str, dict[str, str], bool]:
    """Return ``(protected_text, token_map, uncertain)`` for one text string."""
    mapping: dict[str, str] = {}
    protected = text

    for pattern in _PATTERNS:

        def replace(match: re.Match[str]) -> str:
            token = match.group(0)
            key = _marker(len(mapping))
            mapping[key] = token
            return key

        protected = pattern.sub(replace, protected)

    uncertain = text.count("$$") % 2 == 1 or text.count("$") % 2 == 1
    uncertain = uncertain or protected.count("\ue000") != protected.count("\ue001")
    uncertain = uncertain or ("\\[" in protected) != ("\\]" in protected)
    return protected, mapping, uncertain


def restore_math(text: str, mapping: dict[str, str]) -> str:
    """Restore every private-use placeholder to its original math token."""

    def restore(match: re.Match[str]) -> str:
        return mapping.get(match.group(0), match.group(0))

    return _MARKER.sub(restore, text)


def has_math(text: str) -> bool:
    """True when at least one mathematical token would be protected."""
    _, mapping, _ = protect_math(text)
    return bool(mapping)


def validate_placeholders(
    translated: str, mapping: dict[str, str]
) -> tuple[str | None, str | None]:
    """Verify that every protected token survived exactly once.

    Returns ``(restored_text, error)``. A missing, duplicated, altered or
    extra placeholder means the provider response must be rejected - never
    silently "repaired" with possibly corrupted mathematics.
    """
    if not mapping:
        return translated, None
    found = [match.group(0) for match in _MARKER.finditer(translated)]
    expected = sorted(mapping)
    unexpected = sorted(set(found) - set(expected))
    missing = sorted(set(expected) - set(found))
    duplicated = sorted({token for token in found if found.count(token) != 1})
    if unexpected or missing or duplicated or len(found) != len(expected):
        return None, (
            "protected placeholders changed: "
            f"missing={missing}, duplicated={duplicated}, unexpected={unexpected}, "
            f"counts=({len(found)} found / {len(expected)} expected)"
        )
    return restore_math(translated, mapping), None
