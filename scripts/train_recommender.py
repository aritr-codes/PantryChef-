"""Train recommender artifacts on the processed corpus.

    uv run python scripts/train_recommender.py

Reads the processed recipe corpus, trains the Phase 3 reranker once, validates
the persisted artifact, and saves one recommender bundle to models/recommender/.

By default this trains only the production no-substitution LambdaMART model
and never loads Phase-2 substitution artifacts. Pass --experimental-subs to
train the full ablation set (linear, lambdamart, lambdamart-nosub) with
substitution-backed sub_fill features; that path is research-only and orders
of magnitude slower at scale.
"""

from __future__ import annotations

import argparse
import dataclasses
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from pantrychef.common import get_logger
from pantrychef.config import get_settings
from pantrychef.data.store import load_recipes
from pantrychef.recommender.benchmark import (
    BenchmarkRecorder,
    ConsoleObserver,
    TrainObserver,
    render_summary,
)
from pantrychef.recommender.bundle import DEFAULT_BUNDLE_NAME
from pantrychef.recommender.config import RecConfig
from pantrychef.recommender.sub_lookup import load_sub_lookup
from pantrychef.recommender.train import train_bundle, train_production_bundle
from pantrychef.retrieval.index import InvertedIndex

log = get_logger(__name__)


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=int, default=None, help="cap corpus size for fast dev")
    ap.add_argument("--mask-fraction", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument(
        "--experimental-subs",
        action="store_true",
        help="EXPERIMENTAL: load Phase-2 substitutions and train the full "
        "ablation model set (linear, lambdamart, lambdamart-nosub)",
    )
    ap.add_argument(
        "--max-train-queries", type=int, default=None, help="cap attempted train queries"
    )
    ap.add_argument("--output", default=None, help="override recommender bundle path")
    args = ap.parse_args(argv)

    s = get_settings()
    recipes_path = s.processed_dir / "recipes.jsonl"
    if not recipes_path.exists():
        log.error("Missing %s. Run `make data` first.", recipes_path)
        return 1

    git_sha = _git_commit()
    cfg = RecConfig(
        mask_fraction=args.mask_fraction,
        seed=args.seed,
        max_train_queries=args.max_train_queries,
    )
    recorder = BenchmarkRecorder(
        git_sha=git_sha,
        dataset_size=0,
        config={
            "cli": {
                "max_rows": args.max_rows,
                "experimental_subs": args.experimental_subs,
                "output": args.output,
            },
            "recommender": dataclasses.asdict(cfg),
            "mode": "experimental-ablation" if args.experimental_subs else "production",
        },
    )
    observer: TrainObserver = ConsoleObserver(recorder)

    with recorder.stage("load_recipes", "Load recipes"):
        recipes = load_recipes(recipes_path, limit=args.max_rows)
    recorder.dataset_size = len(recipes)
    with recorder.stage("build_index", "Build index"):
        index = InvertedIndex.build(recipes)
    metadata = {
        "trained_at": datetime.now(UTC).isoformat(),
        "git_commit": git_sha,
        "training_mode": "experimental-ablation" if args.experimental_subs else "production",
    }
    if args.experimental_subs:
        with recorder.stage("load_substitutions", "Load substitutions"):
            sub_lookup = load_sub_lookup(cfg)
        bundle, stats = train_bundle(
            recipes,
            index,
            sub_lookup,
            cfg,
            use_lambdamart=True,
            metadata=metadata,
            observer=observer,
        )
    else:
        bundle, stats = train_production_bundle(
            recipes,
            index,
            cfg,
            metadata=metadata,
            observer=observer,
        )
    bundle.metadata["train_stats"] = stats

    out = Path(args.output) if args.output else (s.models_dir / "recommender" / DEFAULT_BUNDLE_NAME)
    with recorder.stage("save_bundle", "Save bundle"):
        bundle.save(out)
    with recorder.stage("validation_reload", "Validation reload"):
        restored = bundle.load(out)
    if set(restored.models) != set(bundle.models):
        log.error(
            "Artifact integrity check failed: restored models %s != %s",
            restored.models,
            bundle.models,
        )
        return 1
    log.info("Saved recommender bundle to %s", out)
    log.info("Training stats: %s", stats)
    log.info("Benchmark summary:\n%s", render_summary(recorder.build_result()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
