"""Robust parsing of the models' constrained outputs."""

from __future__ import annotations

import re

_LETTER_RE = re.compile(r"\b([ABCD])\b")


def parse_stance(text: str, *, cot: bool = False) -> str | None:
    """Map a model response to 'correct' | 'incorrect' | None.

    'incorrect' must be tested before 'correct' because it contains it as a
    substring. For chain-of-thought responses we read the final non-empty line,
    which is where the verdict was requested, and fall back to a whole-text
    scan only if that line is unparseable.
    """
    if not text:
        return None

    def _scan(s: str) -> str | None:
        s = s.strip().strip(".*_`'\" \t").lower()
        if not s:
            return None
        # Exact match first, then bounded search.
        if s in ("incorrect", "correct"):
            return s
        has_inc = re.search(r"\bincorrect\b", s) is not None
        has_cor = re.search(r"(?<!in)\bcorrect\b", s) is not None
        if has_inc and not has_cor:
            return "incorrect"
        if has_cor and not has_inc:
            return "correct"
        return None

    if cot:
        lines = [ln for ln in text.splitlines() if ln.strip()]
        for ln in reversed(lines):
            got = _scan(ln)
            if got is not None:
                return got
        return None

    return _scan(text)


def parse_letter(text: str) -> str | None:
    """Extract an A/B/C/D answer from a knowledge probe response."""
    if not text:
        return None
    s = text.strip().strip(".*_`'\" \t")
    if len(s) == 1 and s.upper() in "ABCD":
        return s.upper()
    m = _LETTER_RE.search(s.upper())
    return m.group(1) if m else None
