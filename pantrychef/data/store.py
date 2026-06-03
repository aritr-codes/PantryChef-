"""Persist cleaned recipes as JSONL (one Recipe per line)."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from pantrychef.common.types import Recipe


def save_recipes(recipes: Iterable[Recipe], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in recipes:
            f.write(r.model_dump_json() + "\n")


def load_recipes(path: str | Path, limit: int | None = None) -> list[Recipe]:
    """Load recipes from a JSONL file, lean enough for the full 1.27M corpus.

    When ``limit`` is given, stop after parsing that many recipes — avoids
    parsing the whole corpus when only a capped sample is needed.

    Two memory optimizations make the full corpus fit in RAM alongside the
    inverted index and the substitution co-occurrence graph (the reranker reads
    only ``recipe_id``/``title``/``canonical`` — never ``ingredients_raw``):

    * ``ingredients_raw`` (the bulk of the on-disk bytes) is dropped — it is
      written by the cleaning pass but read by nothing downstream.
    * Canonical ingredient strings are deduplicated to one object each via a
      local intern table: ~30k distinct ingredients back 1.27M recipes, so the
      sets/lists holding them share storage instead of carrying 1.27M copies.

    Output is identical to the naive loader for every field the pipeline reads.
    """
    out: list[Recipe] = []
    interned: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = Recipe.model_validate_json(line)
                r.canonical = [interned.setdefault(c, c) for c in r.canonical]
                r.ingredients_raw = []
                out.append(r)
                if limit is not None and len(out) >= limit:
                    break
    return out
