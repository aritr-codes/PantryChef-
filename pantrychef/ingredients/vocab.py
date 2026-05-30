"""Canonical ingredient vocabulary, built from RecipeNLG's NER entities.

The vocabulary is the bridge between messy free text and a controlled set of
canonical ingredients. Sorted longest-phrase-first so the parser can do
longest-match.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from pantrychef.data.schemas import RawRecipe
from pantrychef.ingredients.normalize import canonicalize


def build_vocabulary(recipes: Iterable[RawRecipe], min_count: int = 2) -> list[str]:
    counts: Counter[str] = Counter()
    for r in recipes:
        for ent in r.ner:
            c = canonicalize(ent)
            if c:
                counts[c] += 1
    vocab = [w for w, n in counts.items() if n >= min_count]
    vocab.sort(key=lambda w: (-len(w.split()), -len(w), w))
    return vocab


def save_vocabulary(vocab: list[str], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")


def load_vocabulary(path: str | Path) -> list[str]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
