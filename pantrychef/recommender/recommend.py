"""Candidate generation (shared pool) + learned rerank.

candidate_pool: all recipes sharing >=1 pantry ingredient, ordered by coverage
(the P1 baseline order), capped. Both the overlap baseline and every learned
model rank this SAME pool, so the comparison isolates the ordering function.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np

from pantrychef.common.types import Recipe, ScoredRecipe
from pantrychef.recommender.features import FEATURE_NAMES, to_matrix
from pantrychef.retrieval.index import InvertedIndex

FeatureFn = Callable[[set[str], Recipe], dict[str, float]]


class _Scorer(Protocol):
    """Anything with a score(X) -> ndarray method (LinearRanker, LambdaMARTRanker, OverlapModel)."""

    def score(self, X: np.ndarray) -> np.ndarray: ...


class CandidatePoolWorkspace:
    """Reusable runtime buffers for exact touched-row candidate counting.

    The workspace is owned by the caller and reused across queries so candidate
    generation avoids both the previous corpus-wide ``bincount`` array and a
    fresh concatenation buffer per call. It is runtime-only state, not
    serialized, and not thread-safe.

    Invariants:
    * ``rows[:m]`` contains the concatenated postings for the current query.
    * ``change[:m]`` marks the first occurrence of each touched row after sort.
    * The contents are scratch-only and may be reordered in place.
    """

    def __init__(self, capacity: int = 0) -> None:
        self.rows = np.empty(capacity, dtype=np.int32)
        self.change = np.empty(capacity, dtype=bool)

    def ensure_capacity(self, capacity: int) -> None:
        if capacity <= len(self.rows):
            return
        self.rows = np.empty(capacity, dtype=np.int32)
        self.change = np.empty(capacity, dtype=bool)


def _candidate_rows(
    index: InvertedIndex,
    pantry,
    cap: int,
    workspace: CandidatePoolWorkspace | None = None,
) -> np.ndarray:
    arrs = [index.postings_rows[ing] for ing in pantry if ing in index.postings_rows]
    if not arrs:
        return np.empty(0, dtype=np.int32)

    total_len = sum(len(arr) for arr in arrs)
    if total_len == 0:
        return np.empty(0, dtype=np.int32)
    workspace = workspace or CandidatePoolWorkspace(total_len)
    workspace.ensure_capacity(total_len)

    cursor = 0
    for arr in arrs:
        next_cursor = cursor + len(arr)
        workspace.rows[cursor:next_cursor] = arr
        cursor = next_cursor

    rows_all = workspace.rows[:total_len]
    rows_all.sort()
    change = workspace.change[:total_len]
    change[0] = True
    if total_len > 1:
        change[1:] = rows_all[1:] != rows_all[:-1]
    starts = np.flatnonzero(change)
    rows = rows_all[starts]
    ends = np.empty_like(starts)
    if len(starts) > 1:
        ends[:-1] = starts[1:]
    ends[-1] = total_len
    matched = ends - starts
    canon_len = index.canon_len_by_row[rows]
    keep = canon_len > 0
    rows, matched, canon_len = rows[keep], matched[keep], canon_len[keep]
    if len(rows) == 0:
        return np.empty(0, dtype=np.int32)
    coverage = matched / canon_len
    missing = canon_len - matched
    order = np.lexsort((index.id_rank_by_row[rows], missing, -coverage))
    return np.asarray(rows[order[:cap]], dtype=np.int32)


def candidate_pool(
    index: InvertedIndex,
    pantry: set[str],
    cap: int,
    workspace: CandidatePoolWorkspace | None = None,
) -> list[Recipe]:
    """Overlap candidates ordered by coverage desc, then fewer-missing, then id.

    This computes exactly the same overlap counts and ordering as the reference
    Python implementation, but it avoids the previous corpus-wide
    ``np.bincount(..., minlength=index.n)`` path by counting only touched rows
    from a reusable scratch buffer.
    """
    rows = _candidate_rows(index, pantry, cap, workspace=workspace)
    return [index.recipe_by_row[int(r)] for r in rows]


def rerank(
    index: InvertedIndex,
    pantry: set[str],
    model: _Scorer,
    feature_fn: FeatureFn,
    k: int = 10,
    cap: int = 200,
    columns: tuple[str, ...] = FEATURE_NAMES,
    pool: list[Recipe] | None = None,
) -> list[ScoredRecipe]:
    """Rerank the shared candidate pool by model score (tie-break recipe_id)."""
    if k <= 0 or cap <= 0:
        return []
    pool = pool if pool is not None else candidate_pool(index, pantry, cap)
    if not pool:
        return []
    feats = [feature_fn(pantry, r) for r in pool]
    scores = model.score(to_matrix(feats, columns))
    order = sorted(range(len(pool)), key=lambda i: (-float(scores[i]), pool[i].recipe_id))
    out: list[ScoredRecipe] = []
    for i in order[:k]:
        r = pool[i]
        canon = index.canon_sets[r.recipe_id]
        out.append(
            ScoredRecipe(
                recipe_id=r.recipe_id,
                score=float(scores[i]),
                title=r.title,
                matched=sorted(pantry & canon),
                missing=sorted(canon - pantry),
            )
        )
    return out
