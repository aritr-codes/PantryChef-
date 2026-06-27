from __future__ import annotations

import json
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest

from pantrychef.common.types import Recipe
from pantrychef.recommender.config import RecConfig
from pantrychef.recommender.features import FEATURE_NAMES, SUB_FEATURES, to_matrix
from pantrychef.recommender.rank import LambdaMARTRanker, LinearRanker
from pantrychef.recommender.train import build_examples
from pantrychef.retrieval.index import InvertedIndex

NO_SUB_COLUMNS = tuple(c for c in FEATURE_NAMES if c not in SUB_FEATURES)


def _corpus() -> list[Recipe]:
    return [
        Recipe(
            recipe_id=str(i),
            title=str(i),
            canonical=["egg", "flour", "milk", "sugar", "butter", "salt"][: 4 + (i % 3)],
        )
        for i in range(80)
    ]


def test_bundle_roundtrip_preserves_models_and_index(tmp_path) -> None:
    pytest.importorskip("lightgbm")
    from pantrychef.recommender.bundle import BundledRanker, RecommenderBundle

    corpus = _corpus()
    idx = InvertedIndex.build(corpus)
    cfg = RecConfig(seed=5)
    feats, labels, groups, _ = build_examples(
        corpus, idx, sub_lookup=None, cfg=cfg, train_only=True
    )
    x_full = to_matrix(feats, FEATURE_NAMES)
    x_nosub = to_matrix(feats, NO_SUB_COLUMNS)
    y = np.array(labels, dtype=int)

    bundle = RecommenderBundle(
        index=idx,
        cfg=cfg,
        models={
            "linear": BundledRanker(
                name="linear",
                kind="linear",
                columns=FEATURE_NAMES,
                model=LinearRanker(seed=cfg.seed).fit(x_full, y, groups),
            ),
            "lambdamart-nosub": BundledRanker(
                name="lambdamart-nosub",
                kind="lambdamart",
                columns=NO_SUB_COLUMNS,
                model=LambdaMARTRanker(seed=cfg.seed).fit(x_nosub, y, groups),
            ),
        },
        metadata={"recipe_count": len(corpus), "corpus_fingerprint": "abc123"},
    )
    out = tmp_path / "recommender_bundle.zip"

    bundle.save(out)
    restored = RecommenderBundle.load(out)

    assert restored.cfg == cfg
    assert restored.metadata["recipe_count"] == len(corpus)
    assert restored.metadata["corpus_fingerprint"] == "abc123"
    assert np.allclose(
        restored.models["linear"].model.score(x_full),
        bundle.models["linear"].model.score(x_full),
    )
    assert np.allclose(
        restored.models["lambdamart-nosub"].model.score(x_nosub),
        bundle.models["lambdamart-nosub"].model.score(x_nosub),
    )
    assert restored.models["lambdamart-nosub"].columns == NO_SUB_COLUMNS
    assert list(restored.index.recipes) == list(bundle.index.recipes)


def test_bundle_rejects_unsupported_version(tmp_path) -> None:
    from pantrychef.recommender.bundle import RecommenderBundle

    path = tmp_path / "bad_bundle.zip"
    with ZipFile(path, mode="w", compression=ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps({"bundle_version": 999}))

    with pytest.raises(ValueError, match="unsupported recommender bundle version"):
        RecommenderBundle.load(path)
