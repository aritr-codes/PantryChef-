"""Re-clean the processed corpus with the current parser, reusing the existing
vocabulary.

Used after the match_canonical multiset-coverage fix: vocab.json is unchanged
(``build_vocabulary`` was not touched), so only the clean pass needs to re-run.
Writes to a temp path; the caller validates and swaps it into place so a crash
mid-stream never corrupts the live corpus.

    uv run python scripts/reclean.py --out data/processed/recipes.new.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pantrychef.common import get_logger
from pantrychef.config import get_settings
from pantrychef.data.clean import clean_recipes
from pantrychef.data.loaders import load_recipenlg
from pantrychef.data.store import save_recipes
from pantrychef.ingredients.vocab import load_vocabulary

log = get_logger(__name__)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output path for the re-cleaned jsonl.")
    ap.add_argument("--max-rows", type=int, default=None)
    args = ap.parse_args()

    s = get_settings()
    raw_csv = s.raw_dir / "full_dataset.csv"
    vocab_path = s.processed_dir / "vocab.json"
    for art in (raw_csv, vocab_path):
        if not art.exists():
            log.error("Missing %s.", art)
            return 1

    vocab = load_vocabulary(vocab_path)
    log.info("Loaded vocab=%d; re-cleaning (max_rows=%s)...", len(vocab), args.max_rows)
    recipes = clean_recipes(load_recipenlg(raw_csv, max_rows=args.max_rows), vocab)
    out = Path(args.out)
    save_recipes(recipes, out)
    log.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
