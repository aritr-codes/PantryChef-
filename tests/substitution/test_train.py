"""Tests for the substitution training pipeline (train_artifacts + Artifacts)."""

from __future__ import annotations

from pantrychef.common.types import Recipe
from pantrychef.substitution.config import SubConfig
from pantrychef.substitution.train import Artifacts, train_artifacts

VOCAB = ["butter", "oil", "flour", "egg", "milk"]
RECIPES = [
    Recipe(recipe_id=f"b{i}", title="t", canonical=["butter", "flour", "egg", "milk"])
    for i in range(15)
] + [
    Recipe(recipe_id=f"o{i}", title="t", canonical=["oil", "flour", "egg", "milk"])
    for i in range(15)
]


def test_train_artifacts_builds_both_arms() -> None:
    art = train_artifacts(RECIPES, VOCAB, SubConfig(dims=16, window=10, epochs=2))
    assert isinstance(art, Artifacts)
    assert art.embeddings.neighbors("butter", k=2)
    assert art.graph.neighbors("butter", k=2)[0][0] == "oil"


def test_build_substitutor() -> None:
    art = train_artifacts(RECIPES, VOCAB, SubConfig(dims=16, window=10, epochs=2))
    sub = art.substitutor(VOCAB)
    out = sub.substitutes("butter", k=2)
    assert all(o.arm == "hybrid" for o in out)
