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


def load_recipes(path: str | Path) -> list[Recipe]:
    out: list[Recipe] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(Recipe.model_validate_json(line))
    return out
