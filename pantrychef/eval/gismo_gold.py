"""Pure transform for the GISMo published substitution benchmark.

Decoded (source, target) ingredient-name pairs -> canonicalized gold mapping.
No I/O or network here so it is unit-testable in CI; the download + pickle
decode live in scripts/fetch_gismo_gold.py.

Source: Fatemi et al., "Learning to Substitute Ingredients in Recipes"
(arXiv:2302.07960). Benchmark data is CC BY-NC 4.0.
"""

from __future__ import annotations

from collections.abc import Iterable

from pantrychef.ingredients.normalize import canonicalize


def canonical_pairs(raw_pairs: Iterable[tuple[str, str]]) -> dict[str, set[str]]:
    """Canonicalize raw (source, target) pairs into a source -> {targets} gold map.

    Both sides are run through the project `canonicalize()`. Pairs are dropped
    when either side is empty after canonicalization or when source == target
    (a self-substitution carries no signal).
    """
    gold: dict[str, set[str]] = {}
    for src_raw, tgt_raw in raw_pairs:
        src = canonicalize(src_raw)
        tgt = canonicalize(tgt_raw)
        if not src or not tgt or src == tgt:
            continue
        gold.setdefault(src, set()).add(tgt)
    return gold
