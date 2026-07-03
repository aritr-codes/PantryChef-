"""`python -m pantrychef.eval.recommend_main` — Phase 3 reranker leaderboard.

Compares overlap (P1) < linear < LambdaMART, plus a LambdaMART-without-subs
ablation, on the masked recipe-recovery task. Prints recall@10 / MRR@10, the
candidate ceiling, and in-pool conditional metrics.
"""

from __future__ import annotations

import argparse
import dataclasses
from zipfile import BadZipFile

import numpy as np

from pantrychef.common import get_logger
from pantrychef.common.types import Recipe
from pantrychef.config import get_settings
from pantrychef.eval.recommend_eval import build_eval_queries, score_queries
from pantrychef.recommender.bundle import DEFAULT_BUNDLE_NAME, RecommenderBundle
from pantrychef.recommender.config import RecConfig
from pantrychef.recommender.features import FEATURE_NAMES, SUB_FEATURES, to_matrix
from pantrychef.recommender.rank import LambdaMARTRanker, LinearRanker
from pantrychef.recommender.sub_lookup import load_sub_lookup
from pantrychef.recommender.train import build_examples
from pantrychef.retrieval.index import InvertedIndex

log = get_logger(__name__)

NO_SUB_COLUMNS = tuple(c for c in FEATURE_NAMES if c not in SUB_FEATURES)


class OverlapModel:
    """Identity ranker: keeps the candidate-pool (coverage) order. score = -row_index."""

    def fit(self, X, y, groups):
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return -np.arange(len(X), dtype=float)


def run_leaderboard(
    recipes: list[Recipe],
    index: InvertedIndex,
    sub_lookup,
    cfg: RecConfig,
    use_lambdamart: bool = True,
) -> list[dict]:
    """Train + evaluate each model; return a list of metric rows.

    Parameters
    ----------
    recipes:
        Corpus of recipes.  Must be a reusable sequence (list), not a
        one-shot iterator — it is traversed once per arm.
    index:
        Pre-built inverted index over *recipes*.
    sub_lookup:
        Phase-2 substitutor callable, or ``None`` to disable sub_fill features.
    cfg:
        Recommender configuration (mask fraction, seed, …).
    use_lambdamart:
        When ``True`` (default) train and evaluate the LambdaMART arm and its
        nosub ablation; set ``False`` for a fast dev/smoke run.

    Returns
    -------
    List of metric dicts, one per arm, each containing ``model`` plus all keys
    returned by :func:`score_queries`.

    The expensive candidate-pool passes are paid once: train examples are built
    a single time (and column-sliced per arm), and the eval pools are built once
    and scored by every arm — instead of rebuilding identical pools per model.
    """
    feats, labels, groups, _ = build_examples(recipes, index, sub_lookup, cfg, train_only=True)
    if not groups:
        raise ValueError("build_examples produced 0 training groups; check corpus/config")
    y = np.array(labels, dtype=int)
    x_full = to_matrix(feats, FEATURE_NAMES)
    eval_queries = build_eval_queries(recipes, index, cfg, test_only=True)

    def row(name: str, model, columns: tuple[str, ...]) -> dict:
        return {
            "model": name,
            **score_queries(model, eval_queries, index, sub_lookup, k=10, columns=columns),
        }

    rows = [row("overlap", OverlapModel(), FEATURE_NAMES)]
    rows.append(row("linear", LinearRanker(seed=cfg.seed).fit(x_full, y, groups), FEATURE_NAMES))

    if use_lambdamart:
        rows.append(
            row("lambdamart", LambdaMARTRanker(seed=cfg.seed).fit(x_full, y, groups), FEATURE_NAMES)
        )
        x_nosub = to_matrix(feats, NO_SUB_COLUMNS)
        rows.append(
            row(
                "lambdamart-nosub",
                LambdaMARTRanker(seed=cfg.seed).fit(x_nosub, y, groups),
                NO_SUB_COLUMNS,
            )
        )
    return rows


def run_bundle_leaderboard(
    bundle: RecommenderBundle,
    sub_lookup,
    cfg: RecConfig,
    use_lambdamart: bool = True,
) -> list[dict]:
    """Evaluate a persisted recommender bundle without retraining any models."""
    recipes = list(bundle.index.recipe_by_row)
    queries = build_eval_queries(recipes, bundle.index, cfg, test_only=True)

    def row(name: str, model, columns: tuple[str, ...]) -> dict:
        return {
            "model": name,
            **score_queries(model, queries, bundle.index, sub_lookup, k=10, columns=columns),
        }

    rows = [row("overlap", OverlapModel(), FEATURE_NAMES)]
    if "linear" in bundle.models:
        linear = bundle.models["linear"]
        rows.append(row("linear", linear.model, linear.columns))
    if use_lambdamart and "lambdamart" in bundle.models:
        lm = bundle.models["lambdamart"]
        rows.append(row("lambdamart", lm.model, lm.columns))
    if use_lambdamart and "lambdamart-nosub" in bundle.models:
        lm_nosub = bundle.models["lambdamart-nosub"]
        rows.append(row("lambdamart-nosub", lm_nosub.model, lm_nosub.columns))
    return rows


def _format_table(rows: list[dict]) -> str:
    """Aligned text table of leaderboard rows (model + recall@10/mrr@10/ceiling)."""
    cols = ["model", "recall@10", "mrr@10", "ceiling", "recall@10|in_pool", "n_queries"]
    header = "  ".join(c.ljust(18) for c in cols)
    lines = [header]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            cells.append((f"{v:.4f}" if isinstance(v, float) else str(v)).ljust(18))
        lines.append("  ".join(cells))
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI args for the reranker leaderboard."""
    ap = argparse.ArgumentParser(description="Phase 3 reranker leaderboard.")
    ap.add_argument("--mask-fraction", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--no-subs", action="store_true", help="skip Phase-2 sub_fill features")
    ap.add_argument(
        "--max-train-queries", type=int, default=None, help="cap attempted train queries"
    )
    ap.add_argument("--max-eval-queries", type=int, default=None, help="cap attempted eval queries")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: load the persisted bundle, score the leaderboard, print rows."""
    args = parse_args(argv)

    s = get_settings()
    bundle_path = s.models_dir / "recommender" / DEFAULT_BUNDLE_NAME
    if not bundle_path.exists():
        log.error("Need %s. Train the recommender bundle first.", bundle_path)
        return 1
    try:
        bundle = RecommenderBundle.load(bundle_path)
    except (BadZipFile, OSError, ValueError) as exc:
        log.error("Failed to load recommender bundle %s: %s", bundle_path, exc)
        return 1

    cfg = dataclasses.replace(
        bundle.cfg,
        mask_fraction=args.mask_fraction,
        seed=args.seed,
        max_train_queries=args.max_train_queries,
        max_eval_queries=args.max_eval_queries,
    )
    sub_lookup = None if args.no_subs else load_sub_lookup(cfg)
    if sub_lookup is None and not args.no_subs:
        log.warning(
            "Substitution artifacts absent; sub_fill features will be 0 and "
            "lambdamart vs lambdamart-nosub will be identical. Pass --no-subs to silence."
        )

    rows = run_bundle_leaderboard(bundle, sub_lookup, cfg, use_lambdamart=True)
    for r in rows:
        log.info("%s", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print(_format_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
