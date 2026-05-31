"""Substitution gold sets: published pairs CSV, curated JSON, mined fallback.

All ingredient strings pass through our `canonicalize()` so gold and predictions
share one surface form. `mine_pairs` is the reproducible fallback when the
published set is unavailable: near-duplicate recipes differing by one ingredient
imply a swap.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from pantrychef.common.types import Recipe
from pantrychef.ingredients.normalize import canonicalize


def load_pairs_csv(path: str | Path) -> dict[str, set[str]]:
    """Load a two-column CSV (source, target) of known substitution pairs.

    Both columns are canonicalized before insertion so the returned dict shares
    the same surface form used by the rest of the pipeline.
    """
    gold: dict[str, set[str]] = defaultdict(set)
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            src = canonicalize(row["source"])
            tgt = canonicalize(row["target"])
            if src and tgt:
                gold[src].add(tgt)
    return dict(gold)


def load_curated(path: str | Path) -> dict[str, set[str]]:
    """Load a hand-curated JSON mapping ingredient → list[substitutes].

    Keys and values are all canonicalized on load. Empty strings (after
    canonicalization) are dropped to stay consistent with `load_pairs_csv`.
    Raises `TypeError` if the JSON value for a key is not a list of strings,
    so hand-authored data fails fast rather than silently corrupting evaluation.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    result: dict[str, set[str]] = {}
    for k, vs in raw.items():
        if not isinstance(vs, list):
            raise TypeError(
                f"curated entry for {k!r} must be a list of strings, got {type(vs).__name__}"
            )
        src = canonicalize(k)
        if not src:
            continue
        targets = {canonicalize(v) for v in vs if isinstance(v, str)}
        targets.discard("")
        if targets:
            result[src] = targets
    return result


def mine_pairs(recipes: Sequence[Recipe], min_overlap: int = 4) -> dict[str, set[str]]:
    """Group recipes by canonical title; within a group, pairs of recipes whose
    ingredient sets differ by exactly one element each imply a substitution.

    Complexity is O(n^2) over recipes sharing a title. In practice title groups
    are small (dozens at most), so this is bounded and acceptable (YAGNI).
    """
    by_title: dict[str, list[set[str]]] = defaultdict(list)
    for r in recipes:
        # Filter empty strings so they cannot become spurious gold labels.
        by_title[canonicalize(r.title)].append({x for x in r.canonical if x})
    gold: dict[str, set[str]] = defaultdict(set)
    for sets in by_title.values():
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                a, b = sets[i], sets[j]
                if len(a & b) < min_overlap:
                    continue
                only_a, only_b = a - b, b - a
                if len(only_a) == 1 and len(only_b) == 1:
                    x = next(iter(only_a))
                    y = next(iter(only_b))
                    gold[x].add(y)
                    gold[y].add(x)
    return dict(gold)
