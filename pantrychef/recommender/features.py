"""Pure per-(query, candidate) feature extraction for the reranker.

No I/O, no model imports. The Phase-2 substitution signal enters via a
`sub_lookup` callable (missing_ingredient -> {substitute_name: score}) so this
module stays cheap to unit-test and free of circular label dependence.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

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

SubLookup = Callable[[str], Mapping[str, float]] | None


def extract_features(
    pantry: set[str],
    recipe: Recipe,
    sub_lookup: SubLookup = None,
) -> dict[str, float]:
    """Compute the feature dict for one pantry/candidate-recipe pair.

    `dietary_ok` is a placeholder 1.0 here (no diet constraint passed in the
    recovery task); it is wired to DietTagger at the call site when a diet is
    supplied. Kept in the vector so the column exists for later phases.
    """
    canon = set(recipe.canonical)
    recipe_len = len(canon)
    matched = pantry & canon
    missing = canon - pantry
    n_matched = len(matched)

    coverage = n_matched / recipe_len if recipe_len else 0.0
    match_frac_pantry = n_matched / len(pantry) if pantry else 0.0

    sub_max = 0.0
    per_missing_best: list[float] = []
    if sub_lookup is not None and missing:
        for m in missing:
            subs = sub_lookup(m)
            best = max((subs[p] for p in pantry if p in subs), default=0.0)
            per_missing_best.append(best)
            sub_max = max(sub_max, best)
    sub_mean = (sum(per_missing_best) / len(per_missing_best)) if per_missing_best else 0.0

    return {
        "coverage": coverage,
        "match_frac_pantry": match_frac_pantry,
        "n_matched": float(n_matched),
        "n_missing": float(len(missing)),
        "recipe_len": float(recipe_len),
        "sub_fill_max": float(sub_max),
        "sub_fill_mean": float(sub_mean),
        "dietary_ok": 1.0,
    }


def to_matrix(feats: list[dict[str, float]], columns: tuple[str, ...] | list[str]) -> np.ndarray:
    """Stack feature dicts into a (n, len(columns)) float array in column order."""
    if not feats:
        return np.empty((0, len(columns)), dtype=float)
    return np.array([[f[c] for c in columns] for f in feats], dtype=float)
