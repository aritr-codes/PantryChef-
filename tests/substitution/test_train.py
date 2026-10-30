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


def test_bundle_roundtrip_preserves_inference(tmp_path) -> None:
    art = train_artifacts(RECIPES, VOCAB, SubConfig(dims=16, window=10, epochs=2))
    bundle = tmp_path / "substitution_bundle.zip"

    art.save_bundle(bundle)
    loaded = Artifacts.load_bundle(bundle)

    assert loaded.vocab == tuple(VOCAB)
    assert loaded.metadata["recipe_count"] == len(RECIPES)
    assert loaded.metadata["vocab_size"] == len(VOCAB)
    assert loaded.metadata["corpus_fingerprint"] == art.metadata["corpus_fingerprint"]
    assert loaded.substitutor().substitutes("butter", k=3) == art.substitutor().substitutes(
        "butter", k=3
    )
