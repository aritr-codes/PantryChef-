"""`python -m pantrychef.eval.substitution_main` — substitution ablation leaderboard.

Loads trained artifacts + a gold set, runs emb-only / graph-only / hybrid, and
prints metrics with coverage. Also reports dietary-validity on the curated set.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

from pantrychef.common import get_logger
from pantrychef.config import get_settings
from pantrychef.eval.subs_gold import load_curated, load_pairs_csv
from pantrychef.eval.substitution_eval import (
    coverage_report,
    dietary_validity,
    evaluate_substitution,
)
from pantrychef.substitution.dietary import DietTagger
from pantrychef.substitution.substitute import Substitutor
from pantrychef.substitution.train import Artifacts

log = get_logger(__name__)


def run_ablation(
    artifacts: Artifacts,
    vocab: Sequence[str],
    gold: dict[str, set[str]],
    alphas: tuple[float, float, float] = (1.0, 0.0, 0.5),
) -> list[dict[str, float | str]]:
    """Run emb-only / graph-only / hybrid ablation over a gold set.

    Parameters
    ----------
    artifacts:
        Trained substitution artifacts (embeddings + graph + config).
    vocab:
        Full ingredient vocabulary for the dietary tagger.
    gold:
        Gold mapping from query ingredient to set of valid substitutes.
    alphas:
        Three alpha values corresponding to (emb-only, graph-only, hybrid).
        Default: (1.0, 0.0, 0.5).

    Returns
    -------
    List of metric dicts, one per arm, each containing ``arm`` plus all keys
    returned by :func:`evaluate_substitution`.
    """
    vocab_set = set(vocab)
    names = ("emb-only", "graph-only", "hybrid")
    rows: list[dict[str, float | str]] = []
    for alpha, name in zip(alphas, names, strict=True):
        cfg = dataclasses.replace(artifacts.cfg, alpha=alpha)
        sub = Substitutor(
            emb=artifacts.embeddings,
            graph=artifacts.graph,
            tagger=DietTagger(known=vocab),
            cfg=cfg,
        )
        metrics = evaluate_substitution(sub, gold, vocab_set, k_list=(1, 5, 10))
        rows.append({"arm": name, **metrics})
    return rows


def main() -> int:
    """CLI entrypoint: load artifacts + gold, run ablation, report metrics."""
    s = get_settings()
    from pantrychef.data.store import load_recipes
    from pantrychef.ingredients.vocab import load_vocabulary
    from pantrychef.substitution.config import SubConfig
    from pantrychef.substitution.cooccur import build_cooccurrence, sppmi
    from pantrychef.substitution.embeddings import EmbeddingModel
    from pantrychef.substitution.graph import ContextGraph

    model = s.models_dir / "substitution" / "word2vec.kv"
    gold_csv = s.data_dir / "eval" / "subs_gold.csv"
    if not model.exists() or not gold_csv.exists():
        log.error("Need %s and %s. Train + fetch gold first.", model, gold_csv)
        return 1

    recipes = load_recipes(s.processed_dir / "recipes.jsonl")
    vocab = load_vocabulary(s.processed_dir / "vocab.json")
    cmat, _, _, _ = build_cooccurrence(recipes, vocab)
    cfg = SubConfig()
    art = Artifacts(
        embeddings=EmbeddingModel.load(model),
        graph=ContextGraph(
            sppmi(cmat, cfg.sppmi_shift),
            vocab,
            cmat,
            lam=cfg.lam,
            overlap_shrink=cfg.overlap_shrink,
        ),
        cfg=cfg,
    )
    gold = load_pairs_csv(gold_csv)

    cov = coverage_report([(a, b) for a, bs in gold.items() for b in bs], set(vocab))
    log.info("Coverage: %s", {k: round(v, 3) for k, v in cov.items()})
    for row in run_ablation(art, vocab, gold):
        log.info("%s", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})

    curated = load_curated(s.data_dir / "eval" / "curated_subs.json")
    tagger = DietTagger(known=vocab)
    rate, n = dietary_validity(art.substitutor(vocab), tagger, [(q, "vegan") for q in curated], k=5)
    log.info(
        "Dietary-validity (vegan) = %.3f over %d subs; tag-coverage=%.3f",
        rate,
        int(n),
        tagger.coverage(vocab),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
