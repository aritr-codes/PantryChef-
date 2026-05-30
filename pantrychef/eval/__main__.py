"""`python -m pantrychef.eval` — run Phase 1 eval on processed artifacts."""

from __future__ import annotations

from pantrychef.common import get_logger
from pantrychef.config import get_settings
from pantrychef.data.store import load_recipes
from pantrychef.eval.retrieval_eval import evaluate_retrieval
from pantrychef.retrieval.index import InvertedIndex

log = get_logger(__name__)


def main() -> int:
    s = get_settings()
    recipes_path = s.processed_dir / "recipes.jsonl"
    vocab_path = s.processed_dir / "vocab.json"
    for artifact in (recipes_path, vocab_path):
        if not artifact.exists():
            log.error("Missing artifact %s. Run `make data` first.", artifact)
            return 1
    recipes = load_recipes(recipes_path)
    index = InvertedIndex.build(recipes)
    metrics = evaluate_retrieval(index, recipes, k=10)
    log.info(
        "Retrieval recall@%d = %.3f (n=%d)",
        int(metrics["k"]),
        metrics["recall_at_k"],
        int(metrics["n"]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
