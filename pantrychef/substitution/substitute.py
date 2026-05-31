"""Hybrid substitution: fuse the embedding + graph arms, then hard-mask by diet.

Fusion uses blended-rank (weighted average of ranks), not score-blending — the
two arms live on different scales and per-query min-max amplifies noise.
``alpha`` blends the arms: 1.0 = emb-only, 0.0 = graph-only. Ties are broken by
rank-disagreement (lower = more consistent across arms), then by best individual
rank, then by name for full determinism. The dietary mask is applied LAST so the
guardrail is 100% by construction. ``rrf`` provides a Reciprocal Rank Fusion
alternative when called directly.

Gated recipe-context re-rank (``SubConfig.context_weight``) is OFF by default
(``context_weight=0``). When enabled it boosts candidates that fit the rest of
the recipe using the embedding arm's pairwise similarity, operating on the
already-fused, already-masked list so the dietary guardrail is never bypassed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from pantrychef.common.types import Substitute
from pantrychef.ingredients.normalize import canonicalize
from pantrychef.substitution.config import SubConfig


class Arm(Protocol):
    """Nearest-neighbour arm interface shared by embedding and graph backends."""

    def neighbors(self, ingredient: str, k: int = 5) -> list[tuple[str, float]]: ...


class Tagger(Protocol):
    """Dietary validity checker."""

    def is_valid(self, ingredient: str, diet: str | None) -> bool: ...


def rank_blend(
    emb: Sequence[tuple[str, float]],
    graph: Sequence[tuple[str, float]],
    alpha: float,
    k: int,
) -> list[tuple[str, float]]:
    """Fuse two ranked lists by blended rank position.

    Blended rank = alpha * rank_emb + (1 - alpha) * rank_graph.  A lower blended
    rank is better, so the score returned is 1 / blended_rank.  Ties are broken
    by rank-disagreement (|rank_emb - rank_graph|, lower = more consistent), then
    by the best individual rank, then alphabetically for full determinism.

    At the extremes alpha=1.0 returns ``emb[:k]`` verbatim and alpha=0.0 returns
    ``graph[:k]`` verbatim — no cross-contamination from the absent arm.
    """
    if alpha >= 1.0:
        return list(emb[:k])
    if alpha <= 0.0:
        return list(graph[:k])

    er: dict[str, int] = {}
    for r, (it, _) in enumerate(emb, 1):
        er.setdefault(it, r)

    gr: dict[str, int] = {}
    for r, (it, _) in enumerate(graph, 1):
        gr.setdefault(it, r)

    miss_e = len(emb) + 1
    miss_g = len(graph) + 1

    fused: list[tuple[str, float, int, int]] = []
    for it in set(er) | set(gr):
        re = er.get(it, miss_e)
        rg = gr.get(it, miss_g)
        blended_rank = alpha * re + (1 - alpha) * rg
        score = 1.0 / blended_rank
        disagreement = abs(re - rg)
        fused.append((it, score, disagreement, min(re, rg)))

    fused.sort(key=lambda x: (-x[1], x[2], x[3], x[0]))
    return [(it, score) for it, score, _, _ in fused[:k]]


def rrf(
    ranked_lists: Sequence[Sequence[tuple[str, float]]],
    k: int,
    c: int = 60,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion across an arbitrary number of ranked lists.

    RRF score for item i = sum_over_lists( 1 / (c + rank_i) ).  ``c`` is the
    smoothing constant (Cormack et al. 2009 recommend c=60).
    """
    scores: dict[str, float] = {}
    for lst in ranked_lists:
        for rank, (it, _) in enumerate(lst, 1):
            scores[it] = scores.get(it, 0.0) + 1.0 / (c + rank)
    out = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return out[:k]


class Substitutor:
    """Hybrid substitute ranker: fuses emb + graph arms, applies diet mask."""

    def __init__(
        self,
        emb: Arm | None,
        graph: Arm | None,
        tagger: Tagger,
        cfg: SubConfig,
    ) -> None:
        self.emb = emb
        self.graph = graph
        self.tagger = tagger
        self.cfg = cfg

    def substitutes(
        self,
        ingredient: str,
        diet: str | None = None,
        recipe: Sequence[str] | None = None,
        k: int | None = None,
    ) -> list[Substitute]:
        """Return up to ``k`` substitutes for ``ingredient`` respecting ``diet``.

        Pipeline:
          1. Canonicalize query.
          2. Fetch pool = max(k*4, 20) candidates from each arm.
          3. Rank-blend the two lists.
          4. Strip query ingredient itself from candidates.
          5. Hard-mask by dietary validity.
          6. Return top-k as ``Substitute`` objects (arm="hybrid").
        """
        if k is not None and k <= 0:
            return []
        k = self.cfg.k if k is None else k
        ing = canonicalize(ingredient)
        pool = max(k * 4, 20)

        emb_res = self.emb.neighbors(ing, pool) if self.emb else []
        graph_res = self.graph.neighbors(ing, pool) if self.graph else []

        fused = rank_blend(emb_res, graph_res, self.cfg.alpha, k=pool)
        # Strip the query ingredient itself before masking and slicing.
        fused = [(it, s) for it, s in fused if canonicalize(it) != ing]
        fused = _mask(fused, self.tagger, diet)

        # Gated recipe-context re-rank: only active when context_weight > 0.
        if (
            recipe
            and self.cfg.context_weight > 0
            and self.emb is not None
            and hasattr(self.emb, "similarity")
        ):
            # Exclude the queried ingredient from context ("rest of the recipe").
            ctx = [canonicalize(c) for c in recipe if canonicalize(c) != ing]
            fused = _context_rerank(fused, ctx, self.emb, self.cfg.context_weight)

        arm = "emb" if self.cfg.alpha >= 1.0 else "graph" if self.cfg.alpha <= 0.0 else "hybrid"
        return [
            Substitute(ingredient=it, score=float(s), dietary_valid=True, arm=arm)
            for it, s in fused[:k]
        ]


def _mask(
    candidates: Sequence[tuple[str, float]],
    tagger: Tagger,
    diet: str | None,
) -> list[tuple[str, float]]:
    """Remove candidates that fail the dietary validity check."""
    return [(it, s) for it, s in candidates if tagger.is_valid(it, diet)]


def _context_rerank(
    candidates: Sequence[tuple[str, float]],
    context: Sequence[str],
    emb: Arm,
    weight: float,
) -> list[tuple[str, float]]:
    """Re-rank ``candidates`` by blending fused rank position with recipe context fit.

    Uses rank reciprocal (1 / (rank + 1)) as the base signal so that the
    original fused order anchors the result.  The context fit is the mean
    pairwise similarity between the candidate and every *in-vocabulary* context
    ingredient — OOV entries are excluded from both numerator and denominator
    to avoid diluting the signal with uninformative zeros.  Duplicate context
    tokens are also deduplicated before scoring.

    The gate (``context_weight > 0``) and presence of ``similarity`` are
    already checked by the caller; this function assumes both hold.
    """
    if not context:
        return list(candidates)
    ctx_dedup = list(dict.fromkeys(context))  # preserve order, remove duplicates
    rescored = []
    for rank, (it, _) in enumerate(candidates):
        base = 1.0 / (rank + 1)  # preserve fused order as the base signal
        # Compute similarity once per pair; exclude OOV pairs (similarity == 0.0)
        # so that missing-vocab terms don't dilute the mean.
        pair_sims = [s for c in ctx_dedup if (s := emb.similarity(it, c)) != 0.0]  # type: ignore[attr-defined]
        fit = sum(pair_sims) / len(pair_sims) if pair_sims else 0.0
        rescored.append((it, base + weight * fit))
    rescored.sort(key=lambda x: (-x[1], x[0]))
    return rescored
