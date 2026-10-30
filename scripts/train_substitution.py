"""Train substitution artifacts on the processed corpus + log to MLflow.

    uv run python scripts/train_substitution.py --epochs 5 --dims 100

Reads data/processed/{recipes.jsonl,vocab.json}. Saves one substitution bundle to
models/substitution/. Logs params to MLflow. CPU-only; full 2.23M corpus trains
in minutes.
"""

from __future__ import annotations

import argparse

import mlflow

from pantrychef.common import get_logger
from pantrychef.config import get_settings
from pantrychef.data.store import load_recipes
from pantrychef.ingredients.vocab import load_vocabulary
from pantrychef.substitution.bundle import DEFAULT_BUNDLE_NAME
from pantrychef.substitution.config import SubConfig
from pantrychef.substitution.train import train_artifacts

log = get_logger(__name__)


def main() -> int:
    """Parse CLI args, run training, persist artifacts, log to MLflow."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, default=100)
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--sppmi-shift", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.5)
    args = ap.parse_args()

    s = get_settings()
    recipes_path = s.processed_dir / "recipes.jsonl"
    vocab_path = s.processed_dir / "vocab.json"
    for art in (recipes_path, vocab_path):
        if not art.exists():
            log.error("Missing %s. Run `python scripts/download_data.py` first.", art)
            return 1

    cfg = SubConfig(
        dims=args.dims,
        window=args.window,
        epochs=args.epochs,
        sppmi_shift=args.sppmi_shift,
        lam=args.lam,
        alpha=args.alpha,
    )
    recipes = load_recipes(recipes_path)
    vocab = load_vocabulary(vocab_path)
    log.info("Training on %d recipes, vocab=%d", len(recipes), len(vocab))

    mlflow.set_tracking_uri(s.mlflow_tracking_uri)
    mlflow.set_experiment("substitution")
    with mlflow.start_run():
        mlflow.log_params(
            {
                "dims": cfg.dims,
                "window": cfg.window,
                "epochs": cfg.epochs,
                "sppmi_shift": cfg.sppmi_shift,
                "lam": cfg.lam,
                "alpha": cfg.alpha,
                "n_recipes": len(recipes),
                "vocab": len(vocab),
            }
        )
        artifacts = train_artifacts(recipes, vocab, cfg)
        out = s.models_dir / "substitution"
        bundle = out / DEFAULT_BUNDLE_NAME
        artifacts.save_bundle(bundle)
        log.info("Saved substitution bundle to %s", bundle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
