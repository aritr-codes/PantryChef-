"""Capped query sampling: seed-hash order, nested prefixes, buffer equivalence."""

from __future__ import annotations

import numpy as np

from pantrychef.common.types import Recipe
from pantrychef.eval.recommend_eval import build_eval_queries
from pantrychef.recommender.config import RecConfig
from pantrychef.recommender.query_sim import is_train, sample_order
from pantrychef.recommender.train import (
    NO_SUB_COLUMNS,
    _build_numeric_rows,
    _collect_training_examples,
    _matrix_from_rows,
)
from pantrychef.retrieval.index import InvertedIndex


def _corpus(n: int = 80) -> list[Recipe]:
    return [
        Recipe(
            recipe_id=str(i),
            title=str(i),
            canonical=["egg", "flour", "milk", "sugar", "butter", "salt"][: 4 + (i % 3)],
        )
        for i in range(n)
    ]


def _collect_gold_ids(corpus: list[Recipe], cap: int | None, seed: int = 13) -> list[str]:
    idx = InvertedIndex.build(corpus)
    cfg = RecConfig(seed=seed, max_train_queries=cap)
    golds: list[str] = []
    _collect_training_examples(
        corpus,
        idx,
        None,
        cfg,
        True,
        None,
        lambda pantry, pool, gold_id: golds.append(gold_id),
    )
    return golds


def test_sample_order_deterministic_and_input_order_independent() -> None:
    corpus = _corpus()
    a = [r.recipe_id for r in sample_order(corpus, seed=13)]
    b = [r.recipe_id for r in sample_order(list(reversed(corpus)), seed=13)]
    assert a == b
    assert sorted(a) == sorted(r.recipe_id for r in corpus)  # permutation
    assert a != [r.recipe_id for r in corpus]  # not corpus file order
    assert a != [r.recipe_id for r in sample_order(corpus, seed=14)]  # seed matters


def test_capped_train_sampling_nested_prefix_and_file_order_independent() -> None:
    corpus = _corpus()
    small = _collect_gold_ids(corpus, cap=6)
    large = _collect_gold_ids(corpus, cap=12)
    assert 0 < len(small) <= len(large)
    assert small == large[: len(small)]  # smaller cap = true prefix of larger
    assert _collect_gold_ids(list(reversed(corpus)), cap=6) == small


def test_uncapped_train_iteration_keeps_corpus_order() -> None:
    # uncapped runs must stay byte-identical to historical ones: file order
    corpus = _corpus(30)
    golds = _collect_gold_ids(corpus, cap=None)
    eligible = [r.recipe_id for r in corpus if is_train(r.recipe_id)]
    assert golds == [g for g in eligible if g in set(golds)]


def test_capped_eval_sampling_nested_prefix_and_file_order_independent() -> None:
    corpus = _corpus()
    idx = InvertedIndex.build(corpus)

    def golds(recipes: list[Recipe], cap: int) -> list[str]:
        cfg = RecConfig(seed=2, max_eval_queries=cap)
        return [gold for _, gold, _, _ in build_eval_queries(recipes, idx, cfg)]

    small = golds(corpus, 4)
    large = golds(corpus, 8)
    assert len(small) == 4 and len(large) == 8  # eval keeps every attempted query
    assert small == large[:4]
    assert golds(list(reversed(corpus)), 4) == small


def test_numeric_rows_column_subset_matches_full_buffer_slice() -> None:
    corpus = _corpus()
    idx = InvertedIndex.build(corpus)
    cfg = RecConfig(seed=3)
    full_vals, full_labels, full_groups, _ = _build_numeric_rows(corpus, idx, None, cfg)
    sub_vals, sub_labels, sub_groups, _ = _build_numeric_rows(
        corpus, idx, None, cfg, columns=NO_SUB_COLUMNS
    )
    assert len(sub_vals) == len(NO_SUB_COLUMNS) * len(sub_labels)
    x_from_full = _matrix_from_rows(full_vals, len(full_labels), NO_SUB_COLUMNS)
    x_from_sub = _matrix_from_rows(
        sub_vals, len(sub_labels), NO_SUB_COLUMNS, buffer_columns=NO_SUB_COLUMNS
    )
    assert np.array_equal(x_from_full, x_from_sub)
    assert list(full_labels) == list(sub_labels)
    assert full_groups == sub_groups
