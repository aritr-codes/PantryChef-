"""Substitution evaluation: precision@k / MRR / recall@k with coverage honesty.

Coverage is reported in full: pairs/queries dropping out of vocab are counted,
and the harness can report both covered-subset and pessimistic metrics
(uncovered query = miss). Dietary-validity is a separate guardrail metric.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Protocol

log = logging.getLogger(__name__)


def precision_at_k(preds: Sequence[str], gold: set[str], k: int) -> float:
    """Fraction of top-k predictions that appear in the gold set.

    The denominator is always *k* (not ``min(k, len(preds))``), so missing
    result slots are treated as wrong.  Duplicate predictions in *preds* are
    each counted separately; callers should deduplicate before scoring if the
    underlying system can emit duplicates.
    """
    if k <= 0:
        return 0.0
    top = preds[:k]
    return sum(1 for p in top if p in gold) / k


def recall_at_k(preds: Sequence[str], gold: set[str], k: int) -> float:
    """Fraction of gold items recovered in the top-k predictions.

    Duplicates in *preds* are collapsed (via set) before counting, so a
    system that repeats the same correct answer does not get extra credit.
    """
    if not gold or k <= 0:
        return 0.0
    top = set(preds[:k])
    return len(top & gold) / len(gold)


def mrr(preds: Sequence[str], gold: set[str]) -> float:
    """Mean reciprocal rank: reciprocal of the rank of the first gold hit."""
    for rank, p in enumerate(preds, 1):
        if p in gold:
            return 1.0 / rank
    return 0.0


def coverage_report(gold_pairs: Sequence[tuple[str, str]], vocab: set[str]) -> dict[str, float]:
    """Report how much of the gold benchmark is reachable via vocab.

    Args:
        gold_pairs: Sequence of (query_ingredient, gold_substitute) pairs.
        vocab: The set of known canonical ingredient tokens.

    Returns:
        Dictionary with coverage statistics:
        - n_pairs: total number of gold pairs
        - covered_pairs: pairs where both endpoints are in vocab
        - pair_coverage: fraction of pairs fully covered
        - endpoint_coverage: fraction of unique endpoints in vocab
        - query_coverage: fraction of unique query ingredients in vocab
    """
    queries = {a for a, _ in gold_pairs}
    endpoints = {x for pair in gold_pairs for x in pair}
    covered_pairs = sum(1 for a, b in gold_pairs if a in vocab and b in vocab)
    queryable = {a for a in queries if a in vocab}
    return {
        "n_pairs": float(len(gold_pairs)),
        "covered_pairs": float(covered_pairs),
        "pair_coverage": covered_pairs / len(gold_pairs) if gold_pairs else 0.0,
        "endpoint_coverage": len(endpoints & vocab) / len(endpoints) if endpoints else 0.0,
        "query_coverage": len(queryable) / len(queries) if queries else 0.0,
    }


class _Substitutor(Protocol):
    def substitutes(
        self, ingredient: str, diet: str | None = ..., recipe=..., k: int | None = ...
    ): ...


class _Tagger(Protocol):
    def is_valid(self, ingredient: str, diet: str | None) -> bool: ...


def evaluate_substitution(
    substitutor: _Substitutor,
    gold: dict[str, set[str]],
    vocab: set[str],
    k_list: Sequence[int] = (1, 5, 10),
    pessimistic: bool = False,
) -> dict[str, float]:
    """Aggregate metrics over gold {query -> set(valid subs)}.

    Args:
        substitutor: Object with a ``substitutes(ingredient, k=…)`` method.
        gold: Mapping from query ingredient to set of valid substitutes.
        vocab: Known canonical ingredient tokens (used for coverage filtering).
        k_list: Cut-off values for precision@k and recall@k.
        pessimistic: If False (default), skip queries whose endpoints are
            out-of-vocab (covered-subset evaluation). If True, count an
            uncovered query as a full miss (pessimistic bound).

    Returns:
        Dictionary containing ``mrr``, ``precision@k``, and ``recall@k`` for
        each k in *k_list*, plus ``n`` (number of evaluated queries).
    """
    if not k_list:
        raise ValueError("k_list must contain at least one value")
    per_k_p: dict[int, list[float]] = {k: [] for k in k_list}
    per_k_r: dict[int, list[float]] = {k: [] for k in k_list}
    mrrs: list[float] = []
    maxk = max(k_list)
    for query, golds in gold.items():
        covered_golds = {g for g in golds if g in vocab}
        if query not in vocab or not covered_golds:
            if pessimistic:
                for k in k_list:
                    per_k_p[k].append(0.0)
                    per_k_r[k].append(0.0)
                mrrs.append(0.0)
            continue
        preds = [s.ingredient for s in substitutor.substitutes(query, k=maxk)]
        for k in k_list:
            per_k_p[k].append(precision_at_k(preds, covered_golds, k))
            per_k_r[k].append(recall_at_k(preds, covered_golds, k))
        mrrs.append(mrr(preds, covered_golds))
    out: dict[str, float] = {"n": float(len(mrrs)), "mrr": _avg(mrrs)}
    for k in k_list:
        out[f"precision@{k}"] = _avg(per_k_p[k])
        out[f"recall@{k}"] = _avg(per_k_r[k])
    return out


def dietary_validity(
    substitutor: _Substitutor,
    tagger: _Tagger,
    queries_diets: Sequence[tuple[str, str]],
    k: int = 5,
) -> tuple[float, float]:
    """Fraction of returned subs that satisfy the requested diet, over all queries.

    Args:
        substitutor: Object with a ``substitutes(ingredient, diet=…, k=…)`` method.
        tagger: Object with an ``is_valid(ingredient, diet)`` method.
        queries_diets: Sequence of (ingredient, diet) pairs to evaluate.
        k: Number of substitutes to fetch per query.

    Returns:
        Tuple of (validity_rate, total_substitutes_evaluated).
    """
    total = valid = 0
    for query, diet in queries_diets:
        for s in substitutor.substitutes(query, diet=diet, k=k):
            total += 1
            if tagger.is_valid(s.ingredient, diet):
                valid += 1
    # When the substitutor returns nothing for all queries, treat as perfect
    # validity for this metric (the caller can detect total==0 and warn).
    rate = valid / total if total else 1.0
    if total == 0:
        log.warning("dietary_validity: no substitutes produced; rate=1.0 is vacuous")
    return rate, float(total)


def _avg(xs: Sequence[float]) -> float:
    """Return the mean of *xs*, or 0.0 for an empty sequence."""
    return sum(xs) / len(xs) if xs else 0.0


__all__ = [
    "precision_at_k",
    "recall_at_k",
    "mrr",
    "coverage_report",
    "evaluate_substitution",
    "dietary_validity",
]
