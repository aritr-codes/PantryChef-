"""Hybrid ingredient parser: rules for quantity/unit, vocabulary longest-match
for the canonical ingredient.

raw line --> ParsedIngredient(quantity, unit, canonical, modifier)
"""

from __future__ import annotations

from collections.abc import Iterable

from pantrychef.common.types import ParsedIngredient
from pantrychef.ingredients.normalize import canonicalize
from pantrychef.ingredients.units import parse_quantity, parse_unit

# Rarest token -> vocab entries for which that word is the least-frequent token.
MatchIndex = dict[str, list[str]]


def build_match_index(vocab: list[str]) -> MatchIndex:
    """Map each entry's rarest word to the entries it is the rarest word of.

    Built once per vocab. An entry can only be a full match when ALL its words
    are present in the phrase, so it is reachable from any single one of its
    words — in particular its rarest. Registering each entry under only its
    rarest word (least document frequency) keeps `match_canonical` correct
    while preventing common words ("cheese", "sauce") from fanning out to
    hundreds of non-matching candidates per line. Tie broken by the word string
    for deterministic, reproducible buckets.
    """
    df: dict[str, int] = {}
    for entry in vocab:
        for w in set(entry.split()):
            df[w] = df.get(w, 0) + 1
    index: MatchIndex = {}
    for entry in vocab:
        words = set(entry.split())
        if not words:
            continue
        rarest = min(words, key=lambda w: (df[w], w))
        index.setdefault(rarest, []).append(entry)
    return index


def match_canonical(phrase: str, vocab: list[str], index: MatchIndex | None = None) -> str | None:
    """Return the longest vocab phrase whose words are all present in `phrase`.

    Among all subset matches, pick the one with the most words, tie-broken by
    longest string then by the entry string (lexicographically greatest),
    making the result deterministic and identical across both paths. Pass a prebuilt `index` (from
    `build_match_index`) to prune candidates to entries registered under their
    rarest word; a full match's rarest word is guaranteed to be in `tokens`, so
    it is reached via that bucket. Without an index, falls back to a full O(V)
    scan. Both paths return the same result.

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
    best_key: tuple[int, int, str] = (0, 0, "")
    for entry in candidates:
        words = entry.split()
        if words and all(w in tokens for w in words):
            key = (len(words), len(entry), entry)
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
