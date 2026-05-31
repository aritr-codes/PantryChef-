"""Tests for hybrid rank-fusion, RRF, and Substitutor."""

from __future__ import annotations

from pantrychef.substitution.substitute import Substitutor, rank_blend, rrf


def test_rank_blend_alpha_extremes() -> None:
    emb = [("a", 0.9), ("b", 0.8), ("c", 0.1)]
    graph = [("c", 0.9), ("b", 0.5), ("a", 0.4)]
    # alpha=1 => emb order
    assert [x for x, _ in rank_blend(emb, graph, alpha=1.0, k=3)] == ["a", "b", "c"]
    # alpha=0 => graph order
    assert [x for x, _ in rank_blend(emb, graph, alpha=0.0, k=3)] == ["c", "b", "a"]
    # blend puts b (good in both) on top
    assert rank_blend(emb, graph, alpha=0.5, k=1)[0][0] == "b"


def test_rrf_combines_ranks() -> None:
    emb = [("a", 0.9), ("b", 0.8)]
    graph = [("b", 0.9), ("a", 0.8)]
    fused = [x for x, _ in rrf([emb, graph], k=2)]
    assert set(fused) == {"a", "b"}


class _FakeArm:
    def __init__(self, ranked: list[tuple[str, float]]) -> None:
        self._r = ranked

    def neighbors(self, ingredient: str, k: int = 5) -> list[tuple[str, float]]:
        return self._r[:k]


class _FakeTagger:
    def is_valid(self, ingredient: str, diet: str | None) -> bool:
        return not (diet == "vegan" and ingredient == "butter")


def test_substitutor_masks_and_returns_types() -> None:
    from pantrychef.substitution.config import SubConfig

    emb = _FakeArm([("butter", 0.9), ("oil", 0.8), ("margarine", 0.7)])
    graph = _FakeArm([("oil", 0.9), ("margarine", 0.85), ("butter", 0.2)])
    s = Substitutor(emb=emb, graph=graph, tagger=_FakeTagger(), cfg=SubConfig(alpha=0.5, k=2))
    out = s.substitutes("butter", diet="vegan", k=2)
    names = [o.ingredient for o in out]
    assert "butter" not in names  # vegan mask drops it
    assert all(o.dietary_valid for o in out)
    assert all(o.arm == "hybrid" for o in out)
    assert len(out) == 2
