"""Protect mathematical tokens from machine translation (Phase 3B-1 Task 5).

Pure string handling: the protected spans are replaced by private-use
placeholders before a Bodhan Indic-Translate call and restored verbatim
afterwards. No mathematical token is ever sent to the translator, and no LLM
is involved. Unbalanced delimiters are reported instead of being silently
"fixed" by translation.
"""

from __future__ import annotations

import re
from collections import Counter
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
            command = prefix.group(0)
            remainder = span[len(command) :]
            if remainder.startswith("{"):
                close = remainder.find("}")
                if close > 0:
                    # Protect the command delimiters; translate only its prose
                    # argument. The braces are restored deterministically, so
                    # the command stays intact regardless of provider output.
                    if start > cursor:
                        segments.append(TextSegment(kind="prose", text=text[cursor:start]))
                    segments.append(TextSegment(kind="math", text=f"{command}{{"))
                    argument = remainder[1:close]
                    if argument:
                        segments.append(TextSegment(kind="prose", text=argument))
                    segments.append(TextSegment(kind="math", text="}"))
                    cursor = start + len(command) + close + 2
                    continue
            # Fallback: command without a simple brace argument.
            end = start + len(command)
            span = command
        else:
            end = match.end()
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
    """Verify every math span survived byte-for-byte, in order, with its count.

    Repeated identical spans are legitimate in textbook text, so validation is
    a multiset check (each distinct span must occur exactly as many times as in
    the source) followed by an ordered-sequence check on the spans scanned
    from the rebuilt string. Missing, extra, reordered or altered spans all
    fail; the provider never sees math, so such changes are reconstruction
    defects.

    Returns ``(error, verified_count)``.
    """
    expected = [segment.text for segment in segments if segment.kind == "math"]
    expected_counts = Counter(expected)
    # Use the same segmentation (including text-command delimiter splitting)
    # on the rebuilt text, so validation compares like with like.
    rebuilt_segments, _ambiguous = segment_mixed_text(rebuilt)
    found = [segment.text for segment in rebuilt_segments if segment.kind == "math"]
    found_counts = Counter(found)

    for span, count in expected_counts.items():
        if found_counts.get(span, 0) < count:
            verified = sum(min(found_counts.get(item, 0), n) for item, n in expected_counts.items())
            return f"math span missing or altered after reconstruction: {span[:80]!r}", verified
    extra = found_counts - expected_counts
    if extra:
        span = sorted(extra)[0]
        verified = sum(min(found_counts.get(item, 0), n) for item, n in expected_counts.items())
        return f"extra math span introduced during reconstruction: {span[:80]!r}", verified
    if found != expected:
        return "math spans reordered after reconstruction", len(expected)
    return None, len(expected)


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
