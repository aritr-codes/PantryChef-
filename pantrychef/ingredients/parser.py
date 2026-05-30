"""Hybrid ingredient parser: rules for quantity/unit, vocabulary longest-match
for the canonical ingredient.

raw line --> ParsedIngredient(quantity, unit, canonical, modifier)
"""

from __future__ import annotations

from pantrychef.common.types import ParsedIngredient
from pantrychef.ingredients.normalize import canonicalize
from pantrychef.ingredients.units import parse_quantity, parse_unit


def match_canonical(phrase: str, vocab: list[str]) -> str | None:
    """Return the longest vocab phrase whose words are all present in `phrase`.

    Order-independent single pass (O(V)): among all subset matches, pick the one
    with the most words, tie-broken by longest string. Avoids re-sorting the
    vocab on every call — this runs per ingredient line across the whole corpus.

    `vocab` entries are compared as-is, so they MUST already be canonical
    (lowercase, singular, hyphen-free, space-separated) — i.e. produced by
    `build_vocabulary`. Only the input `phrase` is canonicalized here.
    """
    tokens = set(canonicalize(phrase).split())
    if not tokens:
        return None
    best: str | None = None
    best_key = (0, 0)
    for entry in vocab:
        words = entry.split()
        if words and all(w in tokens for w in words):
            key = (len(words), len(entry))
            if key > best_key:
                best_key = key
                best = entry
    return best


def parse(raw: str, vocab: list[str] | None = None) -> ParsedIngredient:
    vocab = vocab or []
    body = raw
    modifier: str | None = None
    if "," in raw:
        body, mod = raw.split(",", 1)
        modifier = mod.strip() or None
    qty, rest = parse_quantity(body)
    unit, rest = parse_unit(rest)
    canonical = match_canonical(rest, vocab) if vocab else None
    return ParsedIngredient(
        raw=raw.strip(),
        canonical=canonical,
        quantity=qty,
        unit=unit,
        modifier=modifier,
    )
