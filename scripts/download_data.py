"""Download RecipeNLG and build the Phase-1 processed artifacts.

Manual step (RecipeNLG requires accepting terms): download
`full_dataset.csv` from https://recipenlg.cs.put.poznan.pl/ and place it at
data/raw/full_dataset.csv  (see docs/DATASET.md). Then run:

    uv run python scripts/download_data.py --max-rows 50000

Outputs:
    data/processed/vocab.json     canonical ingredient vocabulary
    data/processed/recipes.jsonl  cleaned recipes with canonical sets
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pantrychef.common import get_logger
from pantrychef.config import get_settings
from pantrychef.data.clean import clean_recipes
from pantrychef.data.loaders import load_recipenlg
from pantrychef.data.store import save_recipes
from pantrychef.ingredients.vocab import build_vocabulary, save_vocabulary

log = get_logger(__name__)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--min-count", type=int, default=5)
    args = ap.parse_args()

    s = get_settings()
    raw_csv = s.raw_dir / "full_dataset.csv"
    if not raw_csv.exists():
        log.error("Missing %s — see docs/DATASET.md for the manual download.", raw_csv)
        return 1

    log.info("Building vocabulary (pass 1)...")
    vocab = build_vocabulary(
        load_recipenlg(raw_csv, max_rows=args.max_rows), min_count=args.min_count
    )
    save_vocabulary(vocab, s.processed_dir / "vocab.json")
    log.info("Vocabulary size: %d", len(vocab))

    log.info("Cleaning recipes (pass 2)...")
    recipes = clean_recipes(load_recipenlg(raw_csv, max_rows=args.max_rows), vocab)
    out = Path(s.processed_dir / "recipes.jsonl")
    save_recipes(recipes, out)
    log.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
