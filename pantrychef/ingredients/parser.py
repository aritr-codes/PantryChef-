"""Hybrid ingredient parser: rules for quantity/unit, vocabulary longest-match
for the canonical ingredient.

raw line --> ParsedIngredient(quantity, unit, canonical, modifier)
"""

from __future__ import annotations

from collections.abc import Iterable

from pantrychef.common.types import ParsedIngredient
from pantrychef.ingredients.normalize import canonicalize
from pantrychef.ingredients.units import parse_quantity, parse_unit

# Token -> vocab entries containing that token.
MatchIndex = dict[str, list[str]]


def build_match_index(vocab: list[str]) -> MatchIndex:
    """Map each token to the vocab entries containing it.

    Built once per vocab so `match_canonical` only inspects entries that share a
    word with the phrase, instead of scanning the whole vocab on every line.
    Turns per-line matching from O(V) into O(candidates) — the difference
    between a ~9-hour and a few-minute full-corpus run.
    """
    index: MatchIndex = {}
    for entry in vocab:
        for w in set(entry.split()):
            index.setdefault(w, []).append(entry)
    return index


def match_canonical(phrase: str, vocab: list[str], index: MatchIndex | None = None) -> str | None:
    """Return the longest vocab phrase whose words are all present in `phrase`.

    Among all subset matches, pick the one with the most words, tie-broken by
    longest string (order-independent). Pass a prebuilt `index` (from
    `build_match_index`) to prune candidates to entries sharing a token with the
    phrase; without it, falls back to a full O(V) scan. Both paths return the
    same result — any full match has all its words in `tokens`, so it is reached
    via every one of its tokens in the index.

    `vocab` entries are compared as-is, so they MUST already be canonical
    (lowercase, singular, hyphen-free, space-separated) — i.e. produced by
    `build_vocabulary`. Only the input `phrase` is canonicalized here.
    """
    tokens = set(canonicalize(phrase).split())
    if not tokens:
        return None
    if index is None:
        candidates: Iterable[str] = vocab
    else:
        seen: set[str] = set()
        candidates = [e for t in tokens for e in index.get(t, ()) if not (e in seen or seen.add(e))]
    best: str | None = None
    best_key = (0, 0)
    for entry in candidates:
        words = entry.split()
        if words and all(w in tokens for w in words):
            key = (len(words), len(entry))
            if key > best_key:
                best_key = key
                best = entry
    return best


def parse(
    raw: str, vocab: list[str] | None = None, index: MatchIndex | None = None
) -> ParsedIngredient:
    vocab = vocab or []
    body = raw
    modifier: str | None = None
    if "," in raw:
        body, mod = raw.split(",", 1)
        modifier = mod.strip() or None
    qty, rest = parse_quantity(body)
    unit, rest = parse_unit(rest)
    canonical = match_canonical(rest, vocab, index) if vocab else None
    return ParsedIngredient(
        raw=raw.strip(),
        canonical=canonical,
        quantity=qty,
        unit=unit,
        modifier=modifier,
    )
