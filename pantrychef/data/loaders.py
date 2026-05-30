"""Read RecipeNLG (CSV) into RawRecipe records.

RecipeNLG stores `ingredients`, `directions`, and `NER` as JSON-encoded lists
inside CSV cells. We parse those defensively (bad/empty cells -> []).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from pantrychef.data.schemas import RawRecipe


def _parse_list(cell: object) -> list[str]:
    if isinstance(cell, list):
        return [str(x) for x in cell]
    if not isinstance(cell, str):
        return []
    try:
        val = json.loads(cell)
    except (json.JSONDecodeError, TypeError):
        return []
    return [str(x) for x in val] if isinstance(val, list) else []


def load_recipenlg(
    path: str | Path, max_rows: int | None = None, chunksize: int = 10_000
) -> Iterator[RawRecipe]:
    """Stream RecipeNLG rows lazily.

    Read in chunks so the full (multi-GB) corpus never loads into memory at
    once. ``max_rows`` bounds the total yielded for quick local/Colab runs.
    """
    seen = 0
    for chunk in pd.read_csv(path, keep_default_na=False, chunksize=chunksize):
        for _, row in chunk.iterrows():
            if max_rows is not None and seen >= max_rows:
                return
            seen += 1
            yield RawRecipe(
                title=str(row.get("title", "")).strip(),
                ingredients=_parse_list(row.get("ingredients")),
                directions=_parse_list(row.get("directions")),
                ner=_parse_list(row.get("NER")),
                link=str(row.get("link", "")),
                source=str(row.get("source", "")),
            )
