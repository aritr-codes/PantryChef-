"""`python -m pantrychef.eval.nutrition_main` — Phase 4 coverage + dietary report.

Prints match/mass coverage, nutrition completeness, dietary-class accuracy, and
the imputation-gate decision. Reads raw recipes (with quantities) from the raw
dataset sample — NOT the canonical-only processed store."""

from __future__ import annotations

import argparse
import json

from pantrychef.common import get_logger
from pantrychef.config import get_settings
from pantrychef.dietary import DietTagger
from pantrychef.eval.nutrition_eval import coverage_report, dietary_accuracy
from pantrychef.ingredients.parser import build_match_index
from pantrychef.ingredients.vocab import load_vocabulary
from pantrychef.nutrition.config import NutritionConfig
from pantrychef.nutrition.match import IngredientMatcher
from pantrychef.nutrition.usda import load_artifact

log = get_logger(__name__)


def impute_gate_tripped(report: dict[str, float], cfg: NutritionConfig) -> bool:
    """Return True when either coverage gate says the macro imputer is warranted."""
    return (
        report["match_coverage"] < cfg.impute_match_cov_gate
        or report["median_unresolved_mass"] > cfg.impute_unresolved_mass_gate
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Phase 4 nutrition coverage report.")
    ap.add_argument("--max-rows", type=int, default=None)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from pantrychef.data.loaders import load_raw_recipes

    args = parse_args(argv)
    s = get_settings()
    cfg = NutritionConfig()

    table = load_artifact(s.processed_dir / "usda.json")
    aliases = json.loads((s.repo_root / "data/nutrition/aliases.json").read_text("utf-8"))
    aliases = {k: int(v) for k, v in aliases.items()}
    matcher = IngredientMatcher(table, aliases=aliases, jaccard_threshold=cfg.jaccard_threshold)

    vocab = load_vocabulary(s.processed_dir / "vocab.json")
    index = build_match_index(vocab)

    raws = load_raw_recipes(limit=args.max_rows or cfg.sample_size)
    recipes = [r.ingredients for r in raws]

    rep = coverage_report(recipes, vocab, index, matcher, cfg.usable_mass_fraction)

    labels = json.loads((s.repo_root / "data/nutrition/dietary_labels.json").read_text("utf-8"))
    acc = dietary_accuracy(DietTagger(known=vocab), labels)

    # Diagnostic only: signals whether the macro imputer is worth building. The
    # imputer (pantrychef/nutrition/impute.py) is a standalone gated component;
    # its predictions are NOT wired into the coverage/macros reported above.
    gate = impute_gate_tripped(rep, cfg)
    log.info("coverage: %s", rep)
    log.info("dietary_accuracy: %s", acc)
    log.info("imputation_gate_tripped (advisory): %s", gate)
    print(json.dumps({"coverage": rep, "dietary": acc, "impute_gate_advisory": gate}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
