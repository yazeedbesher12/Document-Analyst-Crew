"""Deterministic lexical tokenizer for GreenLoop company documents."""

from __future__ import annotations

import re
import unicodedata

_TOKEN_RE = re.compile(
    r"""
    \$\d+(?:\.\d+)?[a-z]* |
    \d+(?:\.\d+)?% |
    [a-z]+\d+(?:\.\d+)? |
    \d+(?:\.\d+)?[a-z]+ |
    \d+(?:\.\d+)? |
    [a-z0-9]+(?:-[a-z0-9]+)+ |
    [a-z]+
    """,
    re.VERBOSE,
)


def tokenize(text: str) -> list[str]:
    """Return deterministic lexical tokens without external language resources."""

    normalized = unicodedata.normalize("NFKC", text or "").lower()
    tokens: list[str] = []

    for match in _TOKEN_RE.finditer(normalized):
        token = match.group(0)
        _append_token_forms(tokens, token)

    return tokens


def _append_token_forms(tokens: list[str], token: str) -> None:
    tokens.append(token)

    if token.startswith("$"):
        without_dollar = token[1:]
        tokens.append(without_dollar)
        if without_dollar.endswith("m") and _is_number(without_dollar[:-1]):
            tokens.append(without_dollar[:-1])

    if token.endswith("%") and _is_number(token[:-1]):
        tokens.append(token[:-1])

    if "-" in token:
        parts = [part for part in token.split("-") if part]
        tokens.extend(parts)


def _is_number(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True
