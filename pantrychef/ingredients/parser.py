"""Hybrid ingredient parser: rules for quantity/unit, vocabulary longest-match
for the canonical ingredient.

raw line --> ParsedIngredient(quantity, unit, canonical, modifier)
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from functools import cache

from pantrychef.common.types import ParsedIngredient
from pantrychef.ingredients.normalize import canonicalize
from pantrychef.ingredients.units import parse_quantity, parse_unit


def _is_subsequence(words: Sequence[str], phrase_tokens: Sequence[str]) -> bool:
    """True if ``words`` appears in order within ``phrase_tokens`` (gaps allowed).

    Multiplicity-aware: each word in ``words`` consumes a distinct later position
    in ``phrase_tokens``, so order-permuted vocab artifacts ("cream heavy") are
    distinguished from the source order ("heavy cream").
    """
    it = iter(phrase_tokens)
    return all(w in it for w in words)


@cache
def _entry_meta(entry: str) -> tuple[tuple[str, ...], tuple[tuple[str, int], ...], int]:
    """Cache per-vocab-entry derived data for the hot match loop.

    Returns ``(words, word_count_items, entry_len)``. Keyed by the entry string,
    so it is reused across the ~11M ``match_canonical`` calls a full-corpus clean
    makes (vocab is ~30k entries; the cache fills once). Read-only — callers must
    not mutate the returned tuples.
    """
    words = tuple(entry.split())
    return words, tuple(Counter(words).items()), len(entry)

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

    Coverage is **multiset-aware**: an entry matches only if each of its words
    occurs in the phrase at least as many times as in the entry. This stops a
    reduplicated noise entry ("butter butter") from matching a line that says
    "butter" once, while genuine reduplications ("cara cara orange") still match
    when the source repeats the word. Among matches, ranking is by word count,
    then by whether the entry's words appear in source order (so order-permuted
    artifacts like "cream heavy" lose to "heavy cream"), then by string length,
    then by entry string for full determinism.
    """
    phrase_tokens = canonicalize(phrase).split()
    if not phrase_tokens:
        return None
    counts = Counter(phrase_tokens)
    tokens = set(phrase_tokens)
    if index is None:
        candidates: Iterable[str] = vocab
    else:
        seen: set[str] = set()
        candidates = [e for t in tokens for e in index.get(t, ()) if not (e in seen or seen.add(e))]
    best: str | None = None
    best_key: tuple[int, int, int, str] = (0, 0, 0, "")
    for entry in candidates:
        words, word_counts, entry_len = _entry_meta(entry)
        if not words:
            continue
        if any(counts.get(w, 0) < n for w, n in word_counts):
            continue
        in_order = 1 if _is_subsequence(words, phrase_tokens) else 0
        key = (len(words), in_order, entry_len, entry)
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
