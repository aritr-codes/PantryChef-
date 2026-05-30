"""Deterministic text normalization for ingredient strings."""

from __future__ import annotations

import re
import unicodedata

_PARENS = re.compile(r"\([^)]*\)")
_KEEP = re.compile(r"[^a-z0-9/.\-\s]")
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().strip()
    text = _PARENS.sub(" ", text)
    text = _KEEP.sub(" ", text)
    return _WS.sub(" ", text).strip()


def singularize(word: str) -> str:
    if len(word) <= 3:
        return word
    if word.endswith("ss"):
        return word
    if word.endswith("sses"):
        return word
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith(("ches", "shes", "ses", "xes", "zes")):
        return word[:-2]
    if word.endswith("oes"):
        return word[:-2]
    if word.endswith("s"):
        return word[:-1]
    return word
