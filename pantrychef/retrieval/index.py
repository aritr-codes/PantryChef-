"""Inverted index: canonical ingredient -> set of recipe ids.

Carries two parallel views of the corpus:

* a string-keyed view (``postings`` / ``recipes`` / ``canon_sets``) used by the
  Phase-1 baseline retriever and the reranker's per-candidate feature/output
  lookups, and
* a row-int columnar view (``postings_rows`` / ``canon_len_by_row`` /
  ``id_rank_by_row`` / ``recipe_by_row``) built once in :meth:`build`, which lets
  :func:`pantrychef.recommender.recommend.candidate_pool` score candidates with
  NumPy instead of a Python dict loop. At the full 1.27M-recipe corpus the
  string-dict path costs ~3.3 s/pool (common ingredients touch hundreds of
  thousands of postings); the columnar path is ~50x faster and byte-identical.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from pantrychef.common.types import Recipe


class InvertedIndex:
    def __init__(self) -> None:
        self.postings: dict[str, set[str]] = {}
        self.recipes: dict[str, Recipe] = {}
        # Canonical ingredient sets, precomputed once per recipe. The reranker
        # scores a recipe across many queries; rebuilding set(recipe.canonical)
        # each time dominates candidate-pool cost, so cache it here.
        self.canon_sets: dict[str, frozenset[str]] = {}
        # Columnar view, populated by _finalize() (see module docstring).
        self.n: int = 0
        self.id_of_row: list[str] = []
        self.recipe_by_row: list[Recipe] = []
        self.canon_len_by_row: np.ndarray = np.empty(0, dtype=np.int64)
        self.id_rank_by_row: np.ndarray = np.empty(0, dtype=np.int64)
        self.postings_rows: dict[str, np.ndarray] = {}

    def add(self, recipe: Recipe) -> None:
        self.recipes[recipe.recipe_id] = recipe
        canon = frozenset(recipe.canonical)
        self.canon_sets[recipe.recipe_id] = canon
        for ing in canon:
            self.postings.setdefault(ing, set()).add(recipe.recipe_id)

    def _finalize(self) -> None:
        """Build the row-int columnar view from the string-keyed postings.

        Rows are assigned in recipe-insertion order. ``id_rank_by_row`` is the
        rank of each row's recipe_id in *string* order, so that lexsorting by it
        reproduces the reference tie-break (recipe_id ascending as a string,
        e.g. "r10" < "r2") exactly. Each ``postings_rows`` array holds each row
        at most once because it is derived from the per-ingredient ``set``.
        """
        ids = list(self.recipes.keys())  # dict preserves insertion order
        self.id_of_row = ids
        self.n = len(ids)
        row_of_id = {rid: i for i, rid in enumerate(ids)}
        self.recipe_by_row = [self.recipes[rid] for rid in ids]
        self.canon_len_by_row = np.fromiter(
            (len(self.canon_sets[rid]) for rid in ids), dtype=np.int64, count=self.n
        )
        self.postings_rows = {
            ing: np.fromiter((row_of_id[rid] for rid in rids), dtype=np.int64, count=len(rids))
            for ing, rids in self.postings.items()
        }
        id_rank = np.empty(self.n, dtype=np.int64)
        for rank, row in enumerate(sorted(range(self.n), key=lambda r: ids[r])):
            id_rank[row] = rank
        self.id_rank_by_row = id_rank

    @classmethod
    def build(cls, recipes: Iterable[Recipe]) -> InvertedIndex:
        idx = cls()
        for r in recipes:
            idx.add(r)
        idx._finalize()
        return idx

    def candidates(self, pantry: Iterable[str]) -> set[str]:
        out: set[str] = set()
        for ing in pantry:
            out |= self.postings.get(ing, set())
        return out
