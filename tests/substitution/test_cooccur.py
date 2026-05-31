"""Tests for ingredient co-occurrence matrix and SPPMI weighting."""

from __future__ import annotations

import numpy as np

from pantrychef.common.types import Recipe
from pantrychef.substitution.cooccur import build_cooccurrence, l2norm_rows, sppmi

VOCAB = ["a", "b", "c", "d"]
RECIPES = [
    Recipe(recipe_id="r0", title="t", canonical=["a", "b", "c"]),
    Recipe(recipe_id="r1", title="t", canonical=["a", "b"]),
    Recipe(recipe_id="r2", title="t", canonical=["a", "d"]),
    Recipe(recipe_id="r3", title="t", canonical=["x"]),  # OOV dropped
]


def test_cooccurrence_counts_symmetric() -> None:
    C, word_df, n_recipes, idx = build_cooccurrence(RECIPES, VOCAB)
    A = C.toarray()
    assert A[idx["a"], idx["b"]] == 2  # co-occur in r0, r1
    assert A[idx["a"], idx["b"]] == A[idx["b"], idx["a"]]  # symmetric
    assert A[idx["a"], idx["a"]] == 0  # no self loops
    assert word_df["a"] == 3  # appears in r0,r1,r2
    assert n_recipes == 4


def test_sppmi_nonnegative_same_shape() -> None:
    C, *_ = build_cooccurrence(RECIPES, VOCAB)
    M = sppmi(C, shift=1.0)
    assert M.shape == C.shape
    assert (M.toarray() >= 0).all()


def test_l2norm_rows_unit_norm() -> None:
    C, *_ = build_cooccurrence(RECIPES, VOCAB)
    M = sppmi(C, shift=1.0)
    Mn = l2norm_rows(M).toarray()
    for row in Mn:
        n = np.linalg.norm(row)
        assert n == 0 or abs(n - 1.0) < 1e-9
