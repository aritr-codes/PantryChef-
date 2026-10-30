from __future__ import annotations

import pytest

from pantrychef.common.types import Recipe
from pantrychef.recommender.config import RecConfig
from pantrychef.recommender.features import extract_features
from pantrychef.recommender.recommend import rerank
from pantrychef.recommender.train import train_bundle, train_production_bundle
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


def test_runtime_roundtrip_preserves_recommendations(tmp_path) -> None:
    from pantrychef.recommender.runtime import RecommenderRuntime

    corpus = _corpus()
    idx = InvertedIndex.build(corpus)
    cfg = RecConfig(seed=4)
    bundle, _ = train_bundle(corpus, idx, sub_lookup=None, cfg=cfg, use_lambdamart=False)
    bundle_path = tmp_path / "recommender_bundle.zip"
    bundle.save(bundle_path)

    pantry = {"egg", "flour", "milk"}
    expected = rerank(
        idx,
        pantry,
        bundle.models["linear"].model,
        lambda p, r: extract_features(p, r, None),
        k=5,
        cap=cfg.candidate_cap,
        columns=bundle.models["linear"].columns,
    )
    runtime = RecommenderRuntime.load(bundle_path, sub_lookup=None)
    got = runtime.recommend(sorted(pantry), k=5, model_name="linear")

    assert got == expected


def test_runtime_inference_does_not_train(tmp_path, monkeypatch) -> None:
    from pantrychef.recommender.rank import LambdaMARTRanker, LinearRanker
    from pantrychef.recommender.runtime import RecommenderRuntime

    corpus = _corpus()
    idx = InvertedIndex.build(corpus)
    cfg = RecConfig(seed=4)
    bundle, _ = train_bundle(corpus, idx, sub_lookup=None, cfg=cfg, use_lambdamart=False)
    bundle_path = tmp_path / "recommender_bundle.zip"
    bundle.save(bundle_path)

    def explode(*_args, **_kwargs):
        raise AssertionError("runtime inference must not call model.fit()")

    monkeypatch.setattr(LinearRanker, "fit", explode)
    monkeypatch.setattr(LambdaMARTRanker, "fit", explode)

    runtime = RecommenderRuntime.load(bundle_path, sub_lookup=None)
    out = runtime.recommend(["egg", "flour", "milk"], k=3, model_name="linear")

    assert len(out) == 3


def test_runtime_loading_is_deterministic(tmp_path) -> None:
    from pantrychef.recommender.runtime import RecommenderRuntime

    corpus = _corpus()
    idx = InvertedIndex.build(corpus)
    cfg = RecConfig(seed=4)
    bundle, _ = train_bundle(corpus, idx, sub_lookup=None, cfg=cfg, use_lambdamart=False)
    bundle_path = tmp_path / "recommender_bundle.zip"
    bundle.save(bundle_path)

    a = RecommenderRuntime.load(bundle_path, sub_lookup=None)
    b = RecommenderRuntime.load(bundle_path, sub_lookup=None)

    got_a = a.recommend(["egg", "flour", "milk"], k=5, model_name="linear")
    got_b = b.recommend(["egg", "flour", "milk"], k=5, model_name="linear")
    assert got_a == got_b


def test_runtime_production_bundle_skips_sub_loading(tmp_path, monkeypatch) -> None:
    pytest.importorskip("lightgbm")
    import pantrychef.recommender.runtime as runtime_mod
    from pantrychef.recommender.runtime import RecommenderRuntime

    corpus = _corpus()
    idx = InvertedIndex.build(corpus)
    bundle, _ = train_production_bundle(corpus, idx, cfg=RecConfig(seed=4))
    bundle_path = tmp_path / "recommender_bundle.zip"
    bundle.save(bundle_path)

    def explode(*_args, **_kwargs):
        raise AssertionError("production bundle must not load substitution artifacts")

    monkeypatch.setattr(runtime_mod, "load_sub_lookup", explode)

    runtime = RecommenderRuntime.load(bundle_path)

    assert runtime.sub_lookup is None
    # default model resolves to lambdamart-nosub, the only bundled model
    out = runtime.recommend(["egg", "flour", "milk"], k=3)
    assert len(out) == 3


def test_runtime_default_model_prefers_production_nosub_in_full_bundle(tmp_path) -> None:
    """RFC behavior change, pinned deliberately: even for a full ablation bundle
    (linear + lambdamart + lambdamart-nosub), the unnamed-model default serves
    the production no-sub model, not the sub-feature lambdamart arm."""
    pytest.importorskip("lightgbm")
    from pantrychef.recommender.runtime import RecommenderRuntime

    corpus = _corpus()
    idx = InvertedIndex.build(corpus)
    cfg = RecConfig(seed=4)
    bundle, _ = train_bundle(corpus, idx, sub_lookup=None, cfg=cfg, use_lambdamart=True)
    bundle_path = tmp_path / "recommender_bundle.zip"
    bundle.save(bundle_path)

    runtime = RecommenderRuntime.load(bundle_path, sub_lookup=None)

    default = runtime.recommend(["egg", "flour", "milk"], k=5)
    explicit = runtime.recommend(["egg", "flour", "milk"], k=5, model_name="lambdamart-nosub")
    assert default == explicit


def test_runtime_sub_columns_bundle_loads_sub_lookup(tmp_path, monkeypatch) -> None:
    import pantrychef.recommender.runtime as runtime_mod
    from pantrychef.recommender.runtime import RecommenderRuntime

    corpus = _corpus()
    idx = InvertedIndex.build(corpus)
    cfg = RecConfig(seed=4)
    # linear trains on all FEATURE_NAMES, including sub_fill columns
    bundle, _ = train_bundle(corpus, idx, sub_lookup=None, cfg=cfg, use_lambdamart=False)
    bundle_path = tmp_path / "recommender_bundle.zip"
    bundle.save(bundle_path)

    calls: list[object] = []

    def fake_load(cfg_arg):
        calls.append(cfg_arg)
        return None

    monkeypatch.setattr(runtime_mod, "load_sub_lookup", fake_load)

    RecommenderRuntime.load(bundle_path)

    assert len(calls) == 1
