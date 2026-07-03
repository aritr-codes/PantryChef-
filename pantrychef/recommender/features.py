"""Pure per-(query, candidate) feature extraction for the reranker.

No I/O, no model imports. The Phase-2 substitution signal enters via a
`sub_lookup` callable (missing_ingredient -> {substitute_name: score}) so this
module stays cheap to unit-test and free of circular label dependence.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from collections.abc import Set as AbstractSet

import numpy as np

from pantrychef.common.types import Recipe

FEATURE_NAMES: tuple[str, ...] = (
    "coverage",
    "match_frac_pantry",
    "n_matched",
    "n_missing",
    "recipe_len",
    "sub_fill_max",
    "sub_fill_mean",
    "dietary_ok",
)
SUB_FEATURES: frozenset[str] = frozenset({"sub_fill_max", "sub_fill_mean"})
FEATURE_INDEX: dict[str, int] = {name: i for i, name in enumerate(FEATURE_NAMES)}

SubLookup = Callable[[str], Mapping[str, float]] | None


def _feature_values(
    pantry: set[str],
    recipe: Recipe,
    sub_lookup: SubLookup = None,
    canon: AbstractSet[str] | None = None,
) -> tuple[float, ...]:
    """Compute feature values in FEATURE_NAMES order for one query/candidate pair.

    `dietary_ok` is a placeholder 1.0 here (no diet constraint passed in the
    recovery task); it is wired to DietTagger at the call site when a diet is
    supplied. Kept in the vector so the column exists for later phases.
    sub_lookup scores are treated as non-negative (clamped at 0).

    ``canon`` is an optional cached canonical-ingredient set for ``recipe``.
    When omitted, the set is reconstructed from ``recipe.canonical`` for
    backward compatibility.
    """
    canon_set = canon if canon is not None else set(recipe.canonical)
    recipe_len = len(canon_set)
    matched = pantry & canon_set
    missing = canon_set - pantry
    n_matched = len(matched)

    coverage = n_matched / recipe_len if recipe_len else 0.0
    match_frac_pantry = n_matched / len(pantry) if pantry else 0.0

    sub_max = 0.0
    per_missing_best: list[float] = []
    if sub_lookup is not None and missing:
        for m in missing:
            subs = sub_lookup(m)
            best = max(0.0, max((subs[p] for p in pantry if p in subs), default=0.0))
            per_missing_best.append(best)
            sub_max = max(sub_max, best)
    sub_mean = (sum(per_missing_best) / len(per_missing_best)) if per_missing_best else 0.0

    return (
        coverage,
        match_frac_pantry,
        float(n_matched),
        float(len(missing)),
        float(recipe_len),
        float(sub_max),
        float(sub_mean),
        1.0,
    )


def extract_feature_row(
    pantry: set[str],
    recipe: Recipe,
    sub_lookup: SubLookup = None,
    canon: AbstractSet[str] | None = None,
) -> tuple[float, ...]:
    """Return numeric feature values in the fixed FEATURE_NAMES order."""
    return _feature_values(pantry, recipe, sub_lookup, canon)


def extract_features(
    pantry: set[str],
    recipe: Recipe,
    sub_lookup: SubLookup = None,
    canon: AbstractSet[str] | None = None,
) -> dict[str, float]:
    """Compute the feature dict for one pantry/candidate-recipe pair."""
    values = _feature_values(pantry, recipe, sub_lookup, canon)
    return dict(zip(FEATURE_NAMES, values, strict=True))


def to_matrix(feats: list[dict[str, float]], columns: tuple[str, ...] | list[str]) -> np.ndarray:
    """Stack feature dicts into a (n, len(columns)) float array in column order."""
    if not feats:
        return np.empty((0, len(columns)), dtype=float)
    return np.array([[f[c] for c in columns] for f in feats], dtype=float)
