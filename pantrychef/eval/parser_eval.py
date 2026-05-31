"""Parser evaluation: predicted canonical set vs NER ground truth.

Micro-averaged precision/recall/F1 over the recipe corpus (set-level per
recipe, summed).
"""

from __future__ import annotations

from collections.abc import Iterable

from pantrychef.data.schemas import RawRecipe
from pantrychef.ingredients.normalize import canonicalize
from pantrychef.ingredients.parser import build_match_index, parse


def evaluate_parser(recipes: Iterable[RawRecipe], vocab: list[str]) -> dict[str, float]:
    index = build_match_index(vocab)  # built once; reused per ingredient line
    tp = fp = fn = 0
    for r in recipes:
        gold = {canonicalize(e) for e in r.ner if canonicalize(e)}
        pred = {
            pi.canonical for line in r.ingredients if (pi := parse(line, vocab, index)).canonical
        }
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
