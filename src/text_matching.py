"""Normalize and score French waste names for robust catalog matching."""

from __future__ import annotations

import re
import unicodedata

from rapidfuzz import fuzz

_PUNCTUATION: re.Pattern[str] = re.compile(r"[/+\-'’,.]+")
_WHITESPACE: re.Pattern[str] = re.compile(r"\s+")
_STOPWORDS: frozenset[str] = frozenset(
    {"de", "du", "des", "le", "la", "les", "l", "d", "en", "a", "au", "aux", "un", "une"}
)


def normalize(value: str) -> str:
    """Return an accent-free, case-folded and punctuation-normalized string."""
    decomposed = unicodedata.normalize("NFKD", value)
    unaccented = "".join(char for char in decomposed if not unicodedata.combining(char))
    cleaned = _PUNCTUATION.sub(" ", unaccented.casefold())
    return _WHITESPACE.sub(" ", cleaned).strip()


def _light_plural(token: str) -> str:
    """Apply conservative French plural normalization to one token."""
    if len(token) > 3 and token.endswith("aux"):
        return f"{token[:-3]}al"
    if len(token) > 3 and token.endswith(("s", "x")):
        return token[:-1]
    return token


def tokenize(value: str) -> tuple[str, ...]:
    """Normalize, remove French stopwords and lightly singularize tokens."""
    return tuple(
        normalized
        for token in normalize(value).split()
        if token not in _STOPWORDS
        for normalized in (_light_plural(token),)
        if normalized
    )


def score(query: str, candidate: str) -> float:
    """Return a combined token-set and whole-string similarity in [0, 1]."""
    query_tokens = tokenize(query)
    candidate_tokens = tokenize(candidate)
    if not query_tokens or not candidate_tokens:
        return 0.0
    token_score = fuzz.token_set_ratio(" ".join(query_tokens), " ".join(candidate_tokens))
    string_score = fuzz.ratio(normalize(query), normalize(candidate))
    return (0.7 * token_score + 0.3 * string_score) / 100.0
