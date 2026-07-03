"""Build LightGBM-style (X, y, groups) training data from masked queries.

Only train-split recipes are used. Queries whose candidate pool is empty or
lacks the gold recipe are dropped (counted) - they carry no positive label.
"""

from __future__ import annotations

import random
from array import array
from collections.abc import Callable
from time import perf_counter

import numpy as np

from pantrychef.common import get_logger
from pantrychef.common.types import Recipe
from pantrychef.recommender.benchmark import ExampleProgress, TrainObserver
from pantrychef.recommender.bundle import BundledRanker, RecommenderBundle, corpus_fingerprint
from pantrychef.recommender.config import RecConfig
from pantrychef.recommender.features import (
    FEATURE_INDEX,
    FEATURE_NAMES,
    SUB_FEATURES,
    SubLookup,
    extract_feature_row,
    extract_features,
)
from pantrychef.recommender.query_sim import is_train, make_query, sample_order
from pantrychef.recommender.rank import LambdaMARTRanker, LinearRanker, Ranker
from pantrychef.recommender.recommend import CandidatePoolWorkspace, candidate_pool
from pantrychef.retrieval.index import InvertedIndex

log = get_logger(__name__)
NO_SUB_COLUMNS = tuple(c for c in FEATURE_NAMES if c not in SUB_FEATURES)
_PROGRESS_RECIPE_INTERVAL = 2048
_PROGRESS_TIME_INTERVAL_SECONDS = 30.0


def _column_indices(columns: tuple[str, ...] | list[str]) -> tuple[int, ...]:
    return tuple(FEATURE_INDEX[column] for column in columns)


def _matrix_from_rows(
    flat_values: array,
    row_count: int,
    columns: tuple[str, ...] | list[str] = FEATURE_NAMES,
    buffer_columns: tuple[str, ...] | list[str] = FEATURE_NAMES,
) -> np.ndarray:
    columns_tuple = tuple(columns)
    buffer_tuple = tuple(buffer_columns)
    if row_count == 0:
        return np.empty((0, len(columns_tuple)), dtype=float)
    x_buf = np.frombuffer(flat_values, dtype=np.float64).reshape(row_count, len(buffer_tuple))
    if columns_tuple == buffer_tuple:
        return x_buf
    return x_buf[:, tuple(buffer_tuple.index(c) for c in columns_tuple)]


def _maybe_report_progress(
    observer: TrainObserver | None,
    *,
    recipes_processed: int,
    recipes_total: int,
    queries_total: int,
    queries_kept: int,
    candidates_total: int,
    progress_started: float,
    last_progress: float,
    force: bool = False,
) -> float:
    if observer is None:
        return last_progress
    if not force and recipes_processed % _PROGRESS_RECIPE_INTERVAL != 0:
        return last_progress
    now = perf_counter()
    if force or now - last_progress >= _PROGRESS_TIME_INTERVAL_SECONDS:
        observer.record_examples_progress(
            ExampleProgress(
                recipes_processed=recipes_processed,
                recipes_total=recipes_total,
                queries_total=queries_total,
                queries_kept=queries_kept,
                candidates_total=candidates_total,
                elapsed_seconds=now - progress_started,
            )
        )
        return now
    return last_progress


def _collect_training_examples(
    recipes: list[Recipe],
    index: InvertedIndex,
    sub_lookup: SubLookup,
    cfg: RecConfig,
    train_only: bool,
    observer: TrainObserver | None,
    on_group: Callable[[set[str], list[Recipe], str], None],
) -> tuple[list[int], dict[str, int]]:
    rng = random.Random(cfg.seed)
    groups: list[int] = []
    n_total = n_no_pool = n_no_gold = 0
    candidates_total = 0
    recipes_processed = 0
    progress_started = perf_counter()
    last_progress = progress_started
    pool_workspace = CandidatePoolWorkspace(index.n)

    cap = cfg.max_train_queries
    if cap is not None:
        # capped runs sample recipes in seed-hash order, not corpus file order
        recipes = sample_order(recipes, cfg.seed)
    for recipe in recipes:
        recipes_processed += 1
        if cap is not None and n_total >= cap:
            break
        if train_only and not is_train(recipe.recipe_id):
            last_progress = _maybe_report_progress(
                observer,
                recipes_processed=recipes_processed,
                recipes_total=len(recipes),
                queries_total=n_total,
                queries_kept=len(groups),
                candidates_total=candidates_total,
                progress_started=progress_started,
                last_progress=last_progress,
            )
            continue
        for _ in range(cfg.queries_per_recipe):
            if cap is not None and n_total >= cap:
                break
            q = make_query(recipe, cfg, rng)
            if q is None:
                continue
            n_total += 1
            pantry = set(q.pantry)
            pool = candidate_pool(index, pantry, cfg.candidate_cap, workspace=pool_workspace)
            if not pool:
                n_no_pool += 1
                continue
            candidates_total += len(pool)
            ids = {r.recipe_id for r in pool}
            if q.gold_id not in ids:
                n_no_gold += 1
                continue
            on_group(pantry, pool, q.gold_id)
            groups.append(len(pool))
        last_progress = _maybe_report_progress(
            observer,
            recipes_processed=recipes_processed,
            recipes_total=len(recipes),
            queries_total=n_total,
            queries_kept=len(groups),
            candidates_total=candidates_total,
            progress_started=progress_started,
            last_progress=last_progress,
        )

    stats = {
        "n_queries_total": n_total,
        "n_queries_kept": len(groups),
        "n_dropped_empty_pool": n_no_pool,
        "n_dropped_no_gold": n_no_gold,
    }
    _maybe_report_progress(
        observer,
        recipes_processed=recipes_processed,
        recipes_total=len(recipes),
        queries_total=n_total,
        queries_kept=len(groups),
        candidates_total=candidates_total,
        progress_started=progress_started,
        last_progress=last_progress,
        force=True,
    )
    log.info("build_examples: %s", stats)
    return groups, stats


def _build_numeric_rows(
    recipes: list[Recipe],
    index: InvertedIndex,
    sub_lookup: SubLookup,
    cfg: RecConfig,
    train_only: bool = True,
    observer: TrainObserver | None = None,
    columns: tuple[str, ...] = FEATURE_NAMES,
) -> tuple[array, array, list[int], dict[str, int]]:
    # buffering only `columns` keeps the flat buffer at len(columns)/8 of the
    # all-features footprint on column-subset (production no-sub) builds
    indices = None if tuple(columns) == FEATURE_NAMES else _column_indices(columns)
    flat_values = array("d")
    labels = array("b")

    def on_group(pantry: set[str], pool: list[Recipe], gold_id: str) -> None:
        for recipe in pool:
            row = extract_feature_row(
                pantry,
                recipe,
                sub_lookup,
                canon=index.canon_sets[recipe.recipe_id],
            )
            flat_values.extend(row if indices is None else [row[i] for i in indices])
            labels.append(1 if recipe.recipe_id == gold_id else 0)

    groups, stats = _collect_training_examples(
        recipes,
        index,
        sub_lookup,
        cfg,
        train_only,
        observer,
        on_group,
    )
    return flat_values, labels, groups, stats


def build_examples(
    recipes: list[Recipe],
    index: InvertedIndex,
    sub_lookup: SubLookup,
    cfg: RecConfig,
    train_only: bool = True,
    observer: TrainObserver | None = None,
) -> tuple[list[dict[str, float]], list[int], list[int], dict[str, int]]:
    """Masked-query training examples as raw feature dicts (column-agnostic).

    Returns (feats, labels, group_sizes, stats). Keeping the per-row feature
    dicts (rather than a fixed matrix) lets one build feed several models and
    column subsets - e.g. the LambdaMART vs LambdaMART-nosub ablation - without
    re-running the expensive candidate-pool pass per arm.

    When train_only is False, all recipes are used (no is_train filter). stats
    reports n_queries_total, n_queries_kept, n_dropped_empty_pool,
    n_dropped_no_gold (the last two feed the candidate-recall ceiling).
    """
    feats: list[dict[str, float]] = []
    labels: list[int] = []

    def on_group(pantry: set[str], pool: list[Recipe], gold_id: str) -> None:
        for recipe in pool:
            feats.append(
                extract_features(
                    pantry,
                    recipe,
                    sub_lookup,
                    canon=index.canon_sets[recipe.recipe_id],
                )
            )
            labels.append(1 if recipe.recipe_id == gold_id else 0)

    groups, stats = _collect_training_examples(
        recipes,
        index,
        sub_lookup,
        cfg,
        train_only,
        observer,
        on_group,
    )
    return feats, labels, groups, stats


def build_dataset(
    recipes: list[Recipe],
    index: InvertedIndex,
    sub_lookup: SubLookup,
    cfg: RecConfig,
    columns: tuple[str, ...] = FEATURE_NAMES,
    train_only: bool = True,
    observer: TrainObserver | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int], dict[str, int]]:
    """(X, y, group_sizes, stats) over masked queries in fixed column order."""
    columns_tuple = tuple(columns)
    flat_values, labels, groups, stats = _build_numeric_rows(
        recipes,
        index,
        sub_lookup,
        cfg,
        train_only,
        observer=observer,
        columns=columns_tuple,
    )
    return (
        _matrix_from_rows(flat_values, len(labels), columns_tuple, buffer_columns=columns_tuple),
        np.array(labels, dtype=int),
        groups,
        stats,
    )


def train_bundle(
    recipes: list[Recipe],
    index: InvertedIndex,
    sub_lookup: SubLookup,
    cfg: RecConfig,
    use_lambdamart: bool = True,
    metadata: dict[str, object] | None = None,
    observer: TrainObserver | None = None,
) -> tuple[RecommenderBundle, dict[str, int]]:
    """Train the recommender models once and package them into a bundle."""
    started = perf_counter()
    flat_values, labels, groups, stats = _build_numeric_rows(
        recipes,
        index,
        sub_lookup,
        cfg,
        train_only=True,
        observer=observer,
    )
    if observer is not None:
        observer.record_stage("generate_examples", "Generate examples", perf_counter() - started)
    if not groups:
        raise ValueError("build_examples produced 0 training groups; check corpus/config")
    y = np.array(labels, dtype=int)
    started = perf_counter()
    x_full = _matrix_from_rows(flat_values, len(labels), FEATURE_NAMES)
    if observer is not None:
        observer.record_stage("build_matrix_full", "Build matrix (full)", perf_counter() - started)
    started = perf_counter()
    linear = LinearRanker(seed=cfg.seed).fit(x_full, y, groups)
    if observer is not None:
        observer.record_stage("train_linear", "Train LinearRanker", perf_counter() - started)
    models = {
        "linear": BundledRanker(
            name="linear",
            kind="linear",
            columns=FEATURE_NAMES,
            model=linear,
        )
    }
    if use_lambdamart:
        started = perf_counter()
        lambdamart = LambdaMARTRanker(seed=cfg.seed).fit(x_full, y, groups)
        if observer is not None:
            observer.record_stage("train_lambdamart", "Train LambdaMART", perf_counter() - started)
        models["lambdamart"] = BundledRanker(
            name="lambdamart",
            kind="lambdamart",
            columns=FEATURE_NAMES,
            model=lambdamart,
        )
        started = perf_counter()
        x_nosub = x_full[:, _column_indices(NO_SUB_COLUMNS)]
        if observer is not None:
            observer.record_stage(
                "build_matrix_nosub",
                "Build matrix (no-sub)",
                perf_counter() - started,
            )
        started = perf_counter()
        lambdamart_nosub = LambdaMARTRanker(seed=cfg.seed).fit(x_nosub, y, groups)
        if observer is not None:
            observer.record_stage(
                "train_lambdamart_nosub",
                "Train LambdaMART (no-sub)",
                perf_counter() - started,
            )
        models["lambdamart-nosub"] = BundledRanker(
            name="lambdamart-nosub",
            kind="lambdamart",
            columns=NO_SUB_COLUMNS,
            model=lambdamart_nosub,
        )
    bundle_metadata = {
        "recipe_count": len(recipes),
        "corpus_fingerprint": corpus_fingerprint(recipes),
        **(metadata or {}),
    }
    return RecommenderBundle(index=index, cfg=cfg, models=models, metadata=bundle_metadata), stats


def train_production_bundle(
    recipes: list[Recipe],
    index: InvertedIndex,
    cfg: RecConfig,
    metadata: dict[str, object] | None = None,
    observer: TrainObserver | None = None,
) -> tuple[RecommenderBundle, dict[str, int]]:
    """Train only the production no-substitution LambdaMART model.

    This is the default production training path: no substitution lookup is
    consulted, sub_fill columns never reach the model, and the bundle contains
    a single `lambdamart-nosub` ranker. The full linear/lambdamart/
    lambdamart-nosub ablation set stays in train_bundle() for evaluation and
    research use.
    """
    started = perf_counter()
    flat_values, labels, groups, stats = _build_numeric_rows(
        recipes,
        index,
        None,
        cfg,
        train_only=True,
        observer=observer,
        columns=NO_SUB_COLUMNS,
    )
    if observer is not None:
        observer.record_stage("generate_examples", "Generate examples", perf_counter() - started)
    if not groups:
        raise ValueError("build_examples produced 0 training groups; check corpus/config")
    y = np.array(labels, dtype=int)
    started = perf_counter()
    x_nosub = _matrix_from_rows(
        flat_values, len(labels), NO_SUB_COLUMNS, buffer_columns=NO_SUB_COLUMNS
    )
    if observer is not None:
        observer.record_stage(
            "build_matrix_nosub",
            "Build matrix (no-sub)",
            perf_counter() - started,
        )
    started = perf_counter()
    lambdamart_nosub = LambdaMARTRanker(seed=cfg.seed).fit(x_nosub, y, groups)
    if observer is not None:
        observer.record_stage(
            "train_lambdamart_nosub",
            "Train LambdaMART (no-sub)",
            perf_counter() - started,
        )
    models = {
        "lambdamart-nosub": BundledRanker(
            name="lambdamart-nosub",
            kind="lambdamart",
            columns=NO_SUB_COLUMNS,
            model=lambdamart_nosub,
        )
    }
    bundle_metadata = {
        "recipe_count": len(recipes),
        "corpus_fingerprint": corpus_fingerprint(recipes),
        **(metadata or {}),
    }
    return RecommenderBundle(index=index, cfg=cfg, models=models, metadata=bundle_metadata), stats


def train_ranker(
    model: Ranker,
    recipes,
    index,
    sub_lookup,
    cfg,
    columns=FEATURE_NAMES,
    observer: TrainObserver | None = None,
):
    """Build the dataset and fit `model` in place; returns (model, stats)."""
    X, y, groups, stats = build_dataset(recipes, index, sub_lookup, cfg, columns, observer=observer)
    if not groups:
        raise ValueError("build_dataset produced 0 training groups; check corpus/config")
    model.fit(X, y, groups)
    return model, stats
