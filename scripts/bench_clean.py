"""Verified throughput benchmark for the full-vocab clean.

Times clean_recipes over the first N raw rows using the real 30k-term vocab,
then projects the full-corpus (2.23M) wall-clock. Print-only; writes nothing.

    uv run python scripts/bench_clean.py --rows 5000
"""

from __future__ import annotations

import argparse
import json
import time

from pantrychef.config import get_settings
from pantrychef.data.clean import clean_recipes
from pantrychef.data.loaders import load_recipenlg
from pantrychef.ingredients.parser import build_match_index

FULL_CORPUS_ROWS = 2_231_142  # RecipeNLG full_dataset.csv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=5000)
    args = ap.parse_args()

    s = get_settings()
    with open(s.processed_dir / "vocab.json", encoding="utf-8") as f:
        vocab = json.load(f)
    print(f"vocab size: {len(vocab)}")

    # Report standalone index-build time so fixed vs per-row cost is visible.
    t_idx0 = time.perf_counter()
    _ = build_match_index(vocab)
    t_idx = time.perf_counter() - t_idx0
    print(f"index build (one-time): {t_idx:.2f}s")

    raw_csv = s.raw_dir / "full_dataset.csv"
    raws = load_recipenlg(raw_csv, max_rows=args.rows)

    t0 = time.perf_counter()
    n = sum(1 for _ in clean_recipes(raws, vocab))
    dt = time.perf_counter() - t0

    rpm = n / dt * 60
    eta_min = FULL_CORPUS_ROWS / rpm
    print(f"cleaned {n} recipes in {dt:.1f}s -> {rpm:,.0f} recipes/min")
    print(
        f"projected full corpus ({FULL_CORPUS_ROWS:,}): {eta_min:.1f} min "
        f"({eta_min / 60:.1f} hr)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
