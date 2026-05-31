"""Convert a published ingredient-substitution gold set to our pairs CSV.

The published set (GISMo / FoodBERT 'Exploiting Food Embeddings' lineage)
requires a manual download — see docs/DATASET.md. Point --in at the downloaded
file and --src-col/--tgt-col at its columns; this writes
data/eval/subs_gold.csv with canonicalized `source,target` rows.

Fallback (no published set): --mine builds reproducible pairs from
near-duplicate recipes in our own corpus.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from pathlib import Path

from pantrychef.common.types import Recipe
from pantrychef.data.store import load_recipes
from pantrychef.eval.subs_gold import mine_pairs
from pantrychef.ingredients.normalize import canonicalize


def convert_rows(
    rows: Sequence[dict[str, str]], src_col: str, tgt_col: str
) -> list[tuple[str, str]]:
    """Canonicalize and deduplicate substitution rows from a published gold CSV.

    Rows where the source and target canonicalize to the same string (or either
    is empty) are dropped to avoid trivially self-referential gold pairs.
    Uses `or ""` guards so ragged/None DictReader cells are handled safely.
    Output is sorted for reproducibility across Python versions and runs.
    """
    seen: set[tuple[str, str]] = set()
    for row in rows:
        s = canonicalize(row.get(src_col) or "")
        t = canonicalize(row.get(tgt_col) or "")
        if s and t and s != t:
            seen.add((s, t))
    return sorted(seen)


def _write_pairs(pairs: Sequence[tuple[str, str]], path: str | Path) -> int:
    """Write canonicalized (source, target) pairs to a CSV file.

    Creates parent directories as needed and returns the number of rows written.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "target"])
        w.writerows(pairs)
    return len(pairs)


def write_mined(recipes: Sequence[Recipe], path: str | Path, min_overlap: int = 4) -> int:
    """Mine substitution pairs from near-duplicate recipes and write to CSV.

    Delegates to `mine_pairs` for pair extraction, then flattens the resulting
    dict into rows and writes them via `_write_pairs`. Returns the pair count.
    """
    gold = mine_pairs(recipes, min_overlap=min_overlap)
    pairs = sorted((a, b) for a, bs in gold.items() for b in bs)
    return _write_pairs(pairs, path)


def main() -> int:
    """Entry point for CLI usage; see module docstring for workflow."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default=None, help="Published gold file (CSV).")
    ap.add_argument("--src-col", default="ingredient")
    ap.add_argument("--tgt-col", default="substitution")
    ap.add_argument("--out", default="data/eval/subs_gold.csv")
    ap.add_argument("--mine", action="store_true", help="Build mined fallback from our corpus.")
    ap.add_argument("--min-overlap", type=int, default=4)
    args = ap.parse_args()

    if args.mine:
        from pantrychef.config import get_settings

        recipes = load_recipes(get_settings().processed_dir / "recipes.jsonl")
        n = write_mined(recipes, args.out, min_overlap=args.min_overlap)
        print(f"Wrote {n} mined pairs to {args.out}")
        return 0

    if not args.in_path:
        print("Provide --in <published.csv> or --mine. See docs/DATASET.md.")
        return 1
    with Path(args.in_path).open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    pairs = convert_rows(rows, args.src_col, args.tgt_col)
    n = _write_pairs(pairs, args.out)
    print(f"Wrote {n} pairs to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
