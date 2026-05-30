"""Parser evaluation: predicted canonical set vs NER ground truth.

Micro-averaged precision/recall/F1 over the recipe corpus (set-level per
recipe, summed).
"""

from __future__ import annotations

from collections.abc import Iterable

from pantrychef.data.schemas import RawRecipe
from pantrychef.ingredients.parser import parse
from pantrychef.ingredients.vocab import _canon_token


def evaluate_parser(recipes: Iterable[RawRecipe], vocab: list[str]) -> dict[str, float]:
    tp = fp = fn = 0
    for r in recipes:
        gold = {_canon_token(e) for e in r.ner if _canon_token(e)}
        pred = {pi.canonical for line in r.ingredients if (pi := parse(line, vocab)).canonical}
        tp += len(pred & gold)
        fp += len(pred - gold)
        fn += len(gold - pred)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }
