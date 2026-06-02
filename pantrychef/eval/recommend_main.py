"""`python -m pantrychef.eval.recommend_main` — Phase 3 reranker leaderboard.

Compares overlap (P1) < linear < LambdaMART, plus a LambdaMART-without-subs
ablation, on the masked recipe-recovery task. Prints recall@10 / MRR@10, the
candidate ceiling, and in-pool conditional metrics.
"""

from __future__ import annotations

import argparse

import numpy as np

from pantrychef.common import get_logger
from pantrychef.common.types import Recipe
from pantrychef.eval.recommend_eval import evaluate
from pantrychef.recommender.config import RecConfig
from pantrychef.recommender.features import FEATURE_NAMES, SUB_FEATURES
from pantrychef.recommender.rank import LambdaMARTRanker, LinearRanker
from pantrychef.recommender.train import train_ranker
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
    returned by :func:`evaluate`.
    """
    rows = []

    overlap = OverlapModel()
    rows.append({"model": "overlap", **evaluate(overlap, recipes, index, sub_lookup, cfg, k=10)})

    linear, _ = train_ranker(LinearRanker(seed=cfg.seed), recipes, index, sub_lookup, cfg)
    rows.append({"model": "linear", **evaluate(linear, recipes, index, sub_lookup, cfg, k=10)})

    if use_lambdamart:
        lm, _ = train_ranker(LambdaMARTRanker(seed=cfg.seed), recipes, index, sub_lookup, cfg)
        rows.append({"model": "lambdamart", **evaluate(lm, recipes, index, sub_lookup, cfg, k=10)})

        lm_ns, _ = train_ranker(
            LambdaMARTRanker(seed=cfg.seed), recipes, index, sub_lookup, cfg, columns=NO_SUB_COLUMNS
        )
        rows.append(
            {
                "model": "lambdamart-nosub",
                **evaluate(lm_ns, recipes, index, sub_lookup, cfg, k=10, columns=NO_SUB_COLUMNS),
            }
        )
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


def _build_substitutor():
    """Load Phase-2 artifacts.

    Returns (sub_lookup, recipes, vocab) or (None, None, None) if absent.
    """
    from pantrychef.config import get_settings
    from pantrychef.data.store import load_recipes
    from pantrychef.ingredients.vocab import load_vocabulary
    from pantrychef.substitution.config import SubConfig
    from pantrychef.substitution.cooccur import build_cooccurrence, sppmi
    from pantrychef.substitution.dietary import DietTagger
    from pantrychef.substitution.embeddings import EmbeddingModel
    from pantrychef.substitution.graph import ContextGraph
    from pantrychef.substitution.substitute import Substitutor

    s = get_settings()
    model = s.models_dir / "substitution" / "word2vec.kv"
    if not model.exists():
        return None, None, None
    recipes = load_recipes(s.processed_dir / "recipes.jsonl")
    vocab = load_vocabulary(s.processed_dir / "vocab.json")
    cmat, _, _, _ = build_cooccurrence(recipes, vocab)
    cfg = SubConfig()
    sub = Substitutor(
        emb=EmbeddingModel.load(model),
        graph=ContextGraph(
            sppmi(cmat, cfg.sppmi_shift),
            vocab,
            cmat,
            lam=cfg.lam,
            overlap_shrink=cfg.overlap_shrink,
        ),
        tagger=DietTagger(known=vocab),
        cfg=cfg,
    )

    def lookup(missing: str) -> dict[str, float]:
        return {s_.ingredient: s_.score for s_ in sub.substitutes(missing, k=20)}

    return lookup, recipes, vocab


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI args for the reranker leaderboard."""
    ap = argparse.ArgumentParser(description="Phase 3 reranker leaderboard.")
    ap.add_argument("--max-rows", type=int, default=None, help="cap corpus size for fast dev")
    ap.add_argument("--mask-fraction", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--no-subs", action="store_true", help="skip Phase-2 sub_fill features")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: load artifacts (or fall back), run the leaderboard, log + print rows."""
    from pantrychef.config import get_settings
    from pantrychef.data.store import load_recipes
    from pantrychef.retrieval.index import InvertedIndex

    args = parse_args(argv)

    s = get_settings()
    sub_lookup, recipes, _ = (None, None, None) if args.no_subs else _build_substitutor()
    if recipes is None:
        recipes = load_recipes(s.processed_dir / "recipes.jsonl")
    if args.max_rows is not None:
        recipes = recipes[: args.max_rows]

    if sub_lookup is None and not args.no_subs:
        log.warning(
            "Substitution artifacts absent; sub_fill features will be 0 and "
            "lambdamart vs lambdamart-nosub will be identical. Pass --no-subs to silence."
        )

    index = InvertedIndex.build(recipes)
    cfg = RecConfig(mask_fraction=args.mask_fraction, seed=args.seed)
    rows = run_leaderboard(recipes, index, sub_lookup, cfg)
    for r in rows:
        log.info("%s", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print(_format_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
