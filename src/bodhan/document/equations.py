"""Protect mathematical tokens from machine translation (Phase 3B-1 Task 5).

Pure string handling: the protected spans are replaced by private-use
placeholders before a Bodhan Indic-Translate call and restored verbatim
afterwards. No mathematical token is ever sent to the translator, and no LLM
is involved. Unbalanced delimiters are reported instead of being silently
"fixed" by translation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

# Ordered longest-first: display math, inline math, LaTeX commands, script
# tokens. Bare numbers and ordinary words are deliberately NOT protected.
_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\$\$.+?\$\$", re.DOTALL),
    re.compile(r"\\\[.+?\\\]", re.DOTALL),
    re.compile(r"\$[^$\n]+\$"),
    re.compile(r"\\\(.+?\\\)", re.DOTALL),
    re.compile(r"\\[A-Za-z]+\{[^}]*\}(?:\{[^}]*\})?"),
    re.compile(r"\b[A-Za-z0-9]+(?:_\{[^}]*\}|_[A-Za-z0-9]|\^\{[^}]*\}|\^[A-Za-z0-9])"),
    # Greek letters, math operators and relation symbols (Unicode, written as
    # escapes so the source ASCII stays unambiguous).
    re.compile(
        "[\\u03b1-\\u03c9"
        "\\u0391-\\u03a9"
        "\\u2211\\u220f\\u222b\\u221a\\u221e\\u2264\\u2265\\u2260\\u2248\\u2261"
        "\\u00b1\\u00d7\\u00f7\\u2208\\u2209\\u2282\\u2283\\u222a\\u2229"
        "\\u2192\\u2190\\u2194\\u21d2\\u21d4\\u00b0]+"
    ),
    # Plain arithmetic without LaTeX, including chained expressions
    # (e.g. "2 + 3 = 5", "10-4 = 6"). Digit-leading only: avoid capturing
    # ordinary prose such as "A = B" as mathematics.
    re.compile(r"\d+(?:\.\d+)?(?:\s*[+\-=<>\u00d7\u00f7\u2264\u2265\u2260]\s*\d+(?:\.\d+)?)+"),
)
#: Commands whose brace argument is natural language and must be translated,
#: not protected as mathematics (only the command token itself is preserved).
_TEXT_COMMANDS = frozenset({"text", "textbf", "textit", "mathrm", "mathbf", "emph", "mbox"})
#: One combined scanner over the same patterns, used by span segmentation.
_MATH_SPAN_RE = re.compile("|".join(f"(?:{pattern.pattern})" for pattern in _PATTERNS), re.DOTALL)
_COMMAND_PREFIX = re.compile(r"^\\([A-Za-z]+)")
_MARKER = re.compile(r"\ue000(\d+)\ue001")


@dataclass(frozen=True)
class TextSegment:
    """One contiguous part of a mixed block: either math or prose."""

    kind: Literal["math", "prose"]
    text: str


def _is_ambiguous(text: str) -> bool:
    """True when math delimiters are unbalanced (fail-safe signal)."""
    if text.count("$") % 2 == 1:
        return True
    if ("\\[" in text) != ("\\]" in text):
        return True
    return ("\\(" in text) != ("\\)" in text)


def segment_mixed_text(text: str) -> tuple[list[TextSegment], bool]:
    """Split a block into alternating prose/math spans, in original order.

    Math spans are detected with the same patterns used historically for
    protection, extended for Greek/operator symbols and plain arithmetic.
    The concatenation of all returned segments is byte-for-byte equal to the
    input. ``ambiguous`` is True when delimiters are unbalanced - callers must
    fail safely rather than send a partially-understood string to translation.
    """
    ambiguous = _is_ambiguous(text)
    segments: list[TextSegment] = []
    cursor = 0
    for match in _MATH_SPAN_RE.finditer(text):
        start, end = match.span()
        span = match.group(0)
        prefix = _COMMAND_PREFIX.match(span)
        if (
            prefix is not None
            and prefix.group(1) in _TEXT_COMMANDS
            and len(span) > len(prefix.group(0))
        ):
            # Keep only the command token as math; its brace argument is prose.
            end = start + len(prefix.group(0))
            span = prefix.group(0)
        if start > cursor:
            segments.append(TextSegment(kind="prose", text=text[cursor:start]))
        segments.append(TextSegment(kind="math", text=span))
        cursor = end
    if cursor < len(text):
        segments.append(TextSegment(kind="prose", text=text[cursor:]))
    return segments, ambiguous


def prose_to_translate(segments: list[TextSegment]) -> list[str]:
    """Return only the non-empty prose chunks that need a translation call."""
    return [
        segment.text for segment in segments if segment.kind == "prose" and segment.text.strip()
    ]


def reassemble(segments: list[TextSegment], translated_prose: list[str]) -> str:
    """Rebuild a block from math spans + translated prose, deterministically."""
    output: list[str] = []
    index = 0
    for segment in segments:
        if segment.kind == "math":
            output.append(segment.text)
        elif not segment.text.strip():
            output.append(segment.text)  # preserve exact whitespace
        else:
            if index >= len(translated_prose):
                raise ValueError("not enough translated prose chunks for reassembly")
            output.append(translated_prose[index])
            index += 1
    if index != len(translated_prose):
        raise ValueError("too many translated prose chunks for reassembly")
    return "".join(output)


def validate_math_spans(segments: list[TextSegment], rebuilt: str) -> tuple[str | None, int]:
    """Verify every math span survived byte-for-byte, in order, exactly once.

    Returns ``(error, verified_count)``. This is the post-reconstruction
    counterpart of :func:`validate_placeholders`: the provider never sees math,
    so any alteration is a defect in our own reconstruction and must fail.
    """
    expected = [segment.text for segment in segments if segment.kind == "math"]
    cursor = 0
    verified = 0
    for span in expected:
        position = rebuilt.find(span, cursor)
        if position < 0:
            return f"math span missing or altered after reconstruction: {span[:80]!r}", verified
        if rebuilt.find(span, position + len(span)) >= 0:
            return f"math span duplicated after reconstruction: {span[:80]!r}", verified
        cursor = position + len(span)
        verified += 1
    return None, verified


def preserve_math_only(text: str) -> bool:
    """True when a block has math spans and no translatable prose at all."""
    segments, ambiguous = segment_mixed_text(text)
    if ambiguous or not text.strip():
        return False
    return all(segment.kind == "math" or not segment.text.strip() for segment in segments)


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
