"""Tiered, deterministic canonical-ingredient -> USDA fdc_id resolver.

Tier 0 curated alias -> Tier 1 exact normalized (full description and the part
before the first comma) -> Tier 2 best token-Jaccard over description tokens.
Below threshold returns no match (a reported gap, never a guess — a wrong macro
is worse than a missing one)."""

from __future__ import annotations

from dataclasses import dataclass

from pantrychef.ingredients.normalize import canonicalize


@dataclass(frozen=True)
class Match:
    fdc_id: int | None
    method: str  # "alias" | "exact" | "jaccard" | "none"
    score: float


class IngredientMatcher:
    def __init__(
        self,
        table: dict[int, dict],
        aliases: dict[str, int] | None = None,
        jaccard_threshold: float = 0.34,
    ) -> None:
        self.table = table  # retained for callers (aggregate.py looks up matcher.table[fdc_id])
        self.threshold = jaccard_threshold
        self._aliases = {canonicalize(k): v for k, v in (aliases or {}).items()}
        self._exact: dict[str, int] = {}
        self._tokens: list[tuple[int, set[str]]] = []
        for fdc, food in table.items():
            desc = food["description"]
            full = canonicalize(desc)
            head = canonicalize(desc.split(",", 1)[0])
            self._exact.setdefault(full, fdc)
            self._exact.setdefault(head, fdc)
            self._tokens.append((fdc, set(full.split())))

    def match(self, ingredient: str) -> Match:
        c = canonicalize(ingredient)
        if c in self._aliases:
            return Match(self._aliases[c], "alias", 1.0)
        if c in self._exact:
            return Match(self._exact[c], "exact", 1.0)
        q = set(c.split())
        if q:
            best_fdc, best_score = None, 0.0
            for fdc, toks in self._tokens:
                union = q | toks
                if not union:
                    continue
                j = len(q & toks) / len(union)
                # deterministic tie-break: higher score, then smaller fdc_id
                if j > best_score or (j == best_score and best_fdc is not None and fdc < best_fdc):
                    best_fdc, best_score = fdc, j
            if best_fdc is not None and best_score >= self.threshold:
                return Match(best_fdc, "jaccard", best_score)
        return Match(None, "none", 0.0)
