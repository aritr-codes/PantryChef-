from types import SimpleNamespace

import numpy as np
import pytest

from pantrychef.common.types import Recipe
from pantrychef.eval.recommend_main import (
    OverlapModel,
    main,
    run_bundle_leaderboard,
    run_leaderboard,
)
from pantrychef.recommender.bundle import DEFAULT_BUNDLE_NAME
from pantrychef.recommender.config import RecConfig
from pantrychef.recommender.train import train_bundle
from pantrychef.retrieval.index import InvertedIndex


def _corpus():
    return [
        Recipe(
            recipe_id=str(i),
            title=str(i),
            canonical=["egg", "flour", "milk", "sugar", "butter", "salt", "oil"][: 4 + (i % 4)],
        )
        for i in range(80)
    ]


def _write_bundle(tmp_path, use_lambdamart=False):
    corpus = _corpus()
    idx = InvertedIndex.build(corpus)
    bundle, _ = train_bundle(
        corpus,
        idx,
        sub_lookup=None,
        cfg=RecConfig(seed=3),
        use_lambdamart=use_lambdamart,
    )
    bundle_dir = tmp_path / "models" / "recommender"
    bundle = bundle_dir / DEFAULT_BUNDLE_NAME
    bundle.parent.mkdir(parents=True, exist_ok=True)
    train_bundle_obj, _ = train_bundle(
        corpus,
        idx,
        sub_lookup=None,
        cfg=RecConfig(seed=3),
        use_lambdamart=use_lambdamart,
    )
    train_bundle_obj.save(bundle)
    return corpus, idx, train_bundle_obj, bundle


def test_overlap_model_preserves_pool_order():
    m = OverlapModel()
    s = m.score(np.zeros((3, 2)))
    assert s[0] > s[1] > s[2]


def test_run_leaderboard_smoke():
    corpus = _corpus()
    idx = InvertedIndex.build(corpus)
    rows = run_leaderboard(
        corpus, idx, sub_lookup=None, cfg=RecConfig(seed=3), use_lambdamart=False
    )
    names = {r["model"] for r in rows}
    assert "overlap" in names and "linear" in names
    for r in rows:
        assert "recall@10" in r and "mrr@10" in r


def test_run_leaderboard_lambdamart_smoke():
    pytest.importorskip("lightgbm")  # optional `recommend` extra; skip if absent
    corpus = _corpus()
    idx = InvertedIndex.build(corpus)
    rows = run_leaderboard(corpus, idx, sub_lookup=None, cfg=RecConfig(seed=3), use_lambdamart=True)
    names = {r["model"] for r in rows}
    assert names == {"overlap", "linear", "lambdamart", "lambdamart-nosub"}
    for r in rows:
        assert "recall@10" in r and "mrr@10" in r


def test_run_bundle_leaderboard_matches_training_path() -> None:
    corpus = _corpus()
    idx = InvertedIndex.build(corpus)
    cfg = RecConfig(seed=3)
    expected = run_leaderboard(corpus, idx, sub_lookup=None, cfg=cfg, use_lambdamart=False)
    bundle, _ = train_bundle(corpus, idx, sub_lookup=None, cfg=cfg, use_lambdamart=False)

    got = run_bundle_leaderboard(bundle, sub_lookup=None, cfg=cfg, use_lambdamart=False)

    assert got == expected


def test_main_uses_bundle_without_training(tmp_path, monkeypatch) -> None:
    _, _, _, bundle_path = _write_bundle(tmp_path, use_lambdamart=False)

    import pantrychef.data.store as store
    import pantrychef.eval.recommend_main as rec_main
    import pantrychef.recommender.train as train_mod

    def explode(*_args, **_kwargs):
        raise AssertionError("evaluation must not retrain the recommender")

    monkeypatch.setattr(store, "load_recipes", explode)
    monkeypatch.setattr(train_mod, "build_examples", explode)
    monkeypatch.setattr(
        rec_main,
        "get_settings",
        lambda: SimpleNamespace(models_dir=bundle_path.parent.parent),
    )

    assert main(["--no-subs"]) == 0
