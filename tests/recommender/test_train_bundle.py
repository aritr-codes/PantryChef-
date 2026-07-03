from __future__ import annotations

import pytest

from pantrychef.common.types import Recipe
from pantrychef.recommender.config import RecConfig
from pantrychef.recommender.features import SUB_FEATURES
from pantrychef.recommender.train import NO_SUB_COLUMNS, train_bundle, train_production_bundle
from pantrychef.retrieval.index import InvertedIndex


def _corpus() -> list[Recipe]:
    return [
        Recipe(
            recipe_id=str(i),
            title=str(i),
            canonical=["egg", "flour", "milk", "sugar", "butter", "salt"][: 4 + (i % 3)],
        )
        for i in range(80)
    ]


def test_train_bundle_builds_linear_artifact() -> None:
    corpus = _corpus()
    idx = InvertedIndex.build(corpus)

    bundle, stats = train_bundle(
        corpus, idx, sub_lookup=None, cfg=RecConfig(seed=3), use_lambdamart=False
    )

    assert set(bundle.models) == {"linear"}
    assert bundle.metadata["recipe_count"] == len(corpus)
    assert "corpus_fingerprint" in bundle.metadata
    assert stats["n_queries_kept"] > 0


def test_train_bundle_builds_full_model_set() -> None:
    pytest.importorskip("lightgbm")
    corpus = _corpus()
    idx = InvertedIndex.build(corpus)

    bundle, _ = train_bundle(
        corpus, idx, sub_lookup=None, cfg=RecConfig(seed=3), use_lambdamart=True
    )

    assert set(bundle.models) == {"linear", "lambdamart", "lambdamart-nosub"}


def test_train_production_bundle_trains_only_nosub_model() -> None:
    pytest.importorskip("lightgbm")
    corpus = _corpus()
    idx = InvertedIndex.build(corpus)

    bundle, stats = train_production_bundle(corpus, idx, cfg=RecConfig(seed=3))

    assert set(bundle.models) == {"lambdamart-nosub"}
    model = bundle.models["lambdamart-nosub"]
    assert tuple(model.columns) == NO_SUB_COLUMNS
    assert not SUB_FEATURES.intersection(model.columns)
    assert bundle.metadata["recipe_count"] == len(corpus)
    assert stats["n_queries_kept"] > 0
