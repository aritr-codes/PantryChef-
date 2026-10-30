import numpy as np
import pytest

from pantrychef.common.types import Recipe
from pantrychef.recommender.config import RecConfig
from pantrychef.recommender.features import FEATURE_NAMES, extract_features, to_matrix
from pantrychef.recommender.rank import LambdaMARTRanker, LinearRanker
from pantrychef.recommender.recommend import rerank
from pantrychef.recommender.train import NO_SUB_COLUMNS, build_dataset, build_examples, train_bundle
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


def _reference_extract_features(pantry, recipe, sub_lookup=None, canon=None):
    canon_set = set(recipe.canonical)
    recipe_len = len(canon_set)
    matched = pantry & canon_set
    missing = canon_set - pantry
    n_matched = len(matched)

    coverage = n_matched / recipe_len if recipe_len else 0.0
    match_frac_pantry = n_matched / len(pantry) if pantry else 0.0

    sub_max = 0.0
    per_missing_best: list[float] = []
    if sub_lookup is not None and missing:
        for m in missing:
            subs = sub_lookup(m)
            best = max(0.0, max((subs[p] for p in pantry if p in subs), default=0.0))
            per_missing_best.append(best)
            sub_max = max(sub_max, best)
    sub_mean = (sum(per_missing_best) / len(per_missing_best)) if per_missing_best else 0.0

    return {
        "coverage": coverage,
        "match_frac_pantry": match_frac_pantry,
        "n_matched": float(n_matched),
        "n_missing": float(len(missing)),
        "recipe_len": float(recipe_len),
        "sub_fill_max": float(sub_max),
        "sub_fill_mean": float(sub_mean),
        "dietary_ok": 1.0,
    }


def _reference_dataset(corpus, idx, cfg, columns=FEATURE_NAMES):
    feats, labels, groups, stats = build_examples(
        corpus, idx, sub_lookup=None, cfg=cfg, observer=None
    )
    X = to_matrix(feats, columns)
    y = np.array(labels, dtype=int)
    return X, y, groups, stats


def test_extract_features_cached_canon_matches_reference() -> None:
    recipe = Recipe(recipe_id="r", title="r", canonical=["flour", "egg", "butter", "sugar"])
    pantry = {"flour", "egg", "milk"}
    cached = frozenset(recipe.canonical)

    got = extract_features(pantry, recipe, sub_lookup=None, canon=cached)
    expected = _reference_extract_features(pantry, recipe, sub_lookup=None)

    assert got == expected


def test_build_dataset_matches_build_examples_reference() -> None:
    corpus = _corpus()
    idx = InvertedIndex.build(corpus)
    cfg = RecConfig(seed=4)

    X_ref, y_ref, groups_ref, stats_ref = _reference_dataset(corpus, idx, cfg)
    X_new, y_new, groups_new, stats_new = build_dataset(
        corpus, idx, sub_lookup=None, cfg=cfg, observer=None
    )

    assert np.array_equal(X_new, X_ref)
    assert np.array_equal(y_new, y_ref)
    assert groups_new == groups_ref
    assert stats_new == stats_ref


def test_build_dataset_no_sub_columns_match_build_examples_reference() -> None:
    corpus = _corpus()
    idx = InvertedIndex.build(corpus)
    cfg = RecConfig(seed=4)

    X_ref, y_ref, groups_ref, stats_ref = _reference_dataset(corpus, idx, cfg, NO_SUB_COLUMNS)
    X_new, y_new, groups_new, stats_new = build_dataset(
        corpus,
        idx,
        sub_lookup=None,
        cfg=cfg,
        columns=NO_SUB_COLUMNS,
        observer=None,
    )

    assert np.array_equal(X_new, X_ref)
    assert np.array_equal(y_new, y_ref)
    assert groups_new == groups_ref
    assert stats_new == stats_ref


def test_build_dataset_repeated_runs_are_identical() -> None:
    corpus = _corpus()
    idx = InvertedIndex.build(corpus)
    cfg = RecConfig(seed=4)

    a = build_dataset(corpus, idx, sub_lookup=None, cfg=cfg, observer=None)
    b = build_dataset(corpus, idx, sub_lookup=None, cfg=cfg, observer=None)

    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])
    assert a[2] == b[2]
    assert a[3] == b[3]


def test_rerank_matches_reference_feature_extraction() -> None:
    corpus = _corpus()
    idx = InvertedIndex.build(corpus)
    cfg = RecConfig(seed=4)
    pantry = {"egg", "flour", "milk"}

    bundle, _ = train_bundle(corpus, idx, sub_lookup=None, cfg=cfg, use_lambdamart=False)
    model = bundle.models["linear"].model

    def ref_feat_fn(p, recipe):
        return _reference_extract_features(p, recipe, None)

    def cached_feat_fn(p, recipe):
        return extract_features(p, recipe, None, canon=idx.canon_sets[recipe.recipe_id])

    expected = rerank(
        idx,
        pantry,
        model,
        ref_feat_fn,
        k=5,
        cap=cfg.candidate_cap,
        columns=bundle.models["linear"].columns,
    )
    got = rerank(
        idx,
        pantry,
        model,
        cached_feat_fn,
        k=5,
        cap=cfg.candidate_cap,
        columns=bundle.models["linear"].columns,
    )

    assert got == expected


def test_train_bundle_linear_matches_build_examples_reference() -> None:
    corpus = _corpus()
    idx = InvertedIndex.build(corpus)
    cfg = RecConfig(seed=4)

    X_ref, y_ref, groups_ref, stats_ref = _reference_dataset(corpus, idx, cfg)
    reference_model = LinearRanker(seed=cfg.seed).fit(X_ref, y_ref, groups_ref)

    bundle, stats = train_bundle(corpus, idx, sub_lookup=None, cfg=cfg, use_lambdamart=False)

    assert stats == stats_ref
    model = bundle.models["linear"].model
    assert np.allclose(model.w_, reference_model.w_)
    assert np.isclose(model.b_, reference_model.b_)
    assert np.allclose(model.mean_, reference_model.mean_)
    assert np.allclose(model.std_, reference_model.std_)


def test_train_bundle_linear_preserves_rankings() -> None:
    corpus = _corpus()
    idx = InvertedIndex.build(corpus)
    cfg = RecConfig(seed=4)
    pantry = {"egg", "flour", "milk"}

    X_ref, y_ref, groups_ref, _ = _reference_dataset(corpus, idx, cfg)
    reference_model = LinearRanker(seed=cfg.seed).fit(X_ref, y_ref, groups_ref)
    bundle, _ = train_bundle(corpus, idx, sub_lookup=None, cfg=cfg, use_lambdamart=False)

    def cached_feat_fn(p, recipe):
        return extract_features(p, recipe, None, canon=idx.canon_sets[recipe.recipe_id])

    expected = rerank(
        idx,
        pantry,
        reference_model,
        cached_feat_fn,
        k=5,
        cap=cfg.candidate_cap,
        columns=FEATURE_NAMES,
    )
    got = rerank(
        idx,
        pantry,
        bundle.models["linear"].model,
        cached_feat_fn,
        k=5,
        cap=cfg.candidate_cap,
        columns=bundle.models["linear"].columns,
    )

    assert got == expected


def test_train_bundle_lambdamart_matches_build_examples_reference() -> None:
    pytest.importorskip("lightgbm")
    corpus = _corpus()
    idx = InvertedIndex.build(corpus)
    cfg = RecConfig(seed=4)

    X_full, y_ref, groups_ref, stats_ref = _reference_dataset(corpus, idx, cfg)
    X_nosub, _, _, _ = _reference_dataset(corpus, idx, cfg, NO_SUB_COLUMNS)
    reference_full = LambdaMARTRanker(seed=cfg.seed).fit(X_full, y_ref, groups_ref)
    reference_nosub = LambdaMARTRanker(seed=cfg.seed).fit(X_nosub, y_ref, groups_ref)

    bundle, stats = train_bundle(corpus, idx, sub_lookup=None, cfg=cfg, use_lambdamart=True)

    assert stats == stats_ref
    assert np.allclose(
        bundle.models["lambdamart"].model.score(X_full),
        reference_full.score(X_full),
    )
    assert np.allclose(
        bundle.models["lambdamart-nosub"].model.score(X_nosub),
        reference_nosub.score(X_nosub),
    )
