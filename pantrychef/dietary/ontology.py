"""Dietary tagging via an ontology with category hierarchy + inheritance.

Ingredients are tagged by matching canonical tokens against a leaf-word map
(_LEAF_WORDS), then expanding each leaf to its ancestor categories via _PARENT.
Token matching (not substring) avoids false-positives like "eggplant" → "egg".
Multi-word / special-case overrides live in _CURATED (takes precedence).

Applied as a HARD mask so the guardrail is 100% by construction for *tagged*
constraints. Unknowns default to conservative EXCLUSION under a constrained diet
(an untagged ingredient is not assumed safe). `coverage()` reports how much of
the vocab we can actually judge.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from pantrychef.ingredients.normalize import canonicalize

# Hierarchy: child category -> parent.
# Diets forbidding a parent automatically forbid all descendants.
_PARENT: dict[str, str] = {
    "meat": "animal_product",
    "fish": "animal_product",
    "dairy": "animal_product",
    "egg": "animal_product",
    "honey": "animal_product",
}

# Leaf canonical TOKEN -> category.
# Token match (not substring): "eggplant" never hits "egg".
_LEAF_WORDS: dict[str, str] = {
    "beef": "meat",
    "pork": "meat",
    "chicken": "meat",
    "bacon": "meat",
    "ham": "meat",
    "sausage": "meat",
    "lamb": "meat",
    "turkey": "meat",
    "veal": "meat",
    "lard": "meat",
    "gelatin": "meat",
    "fish": "fish",
    "salmon": "fish",
    "tuna": "fish",
    "shrimp": "fish",
    "anchovy": "fish",
    "cod": "fish",
    "crab": "fish",
    "prawn": "fish",
    "milk": "dairy",
    "butter": "dairy",
    "cheese": "dairy",
    "cream": "dairy",
    "yogurt": "dairy",
    "ghee": "dairy",
    "casein": "dairy",
    "whey": "dairy",
    "egg": "egg",
    "honey": "honey",
    "wheat": "gluten",
    "barley": "gluten",
    "rye": "gluten",
    "bread": "gluten",
    "pasta": "gluten",
    "flour": "gluten",
    "couscous": "gluten",
    "semolina": "gluten",
}

# Multi-word / phrase overrides (canonical form -> categories).
# Takes precedence over token matching. Empty set = explicitly untagged (safe).
_CURATED: dict[str, set[str]] = {
    "worcestershire sauce": {"fish"},
    "fish sauce": {"fish"},
    "soy sauce": {"gluten"},
    "oat": set(),  # oats are GF unless cross-contaminated; don't flag gluten
    "almond flour": set(),
    "coconut flour": set(),
    "rice flour": set(),
    "eggplant": set(),
}

# Diet -> forbidden categories (uses ancestor expansion, so "animal_product" covers
# meat/fish/dairy/egg/honey transitively).
_DIETS: dict[str, set[str]] = {
    "vegan": {"animal_product"},
    "vegetarian": {"meat", "fish"},
    "gluten_free": {"gluten"},
    "dairy_free": {"dairy"},
}

# Kept for the shim's re-export; ontology no longer uses substrings.
_KEYWORDS: dict[str, tuple[str, ...]] = {}


def _ancestors(cat: str) -> set[str]:
    """Return *cat* plus all its ancestor categories."""
    out = {cat}
    while cat in _PARENT:
        cat = _PARENT[cat]
        out.add(cat)
    return out


class DietTagger:
    """Tag ingredients with dietary categories and apply hard-mask filtering."""

    def __init__(self, known: Iterable[str]) -> None:
        self._known = {canonicalize(w) for w in known}

    def tags(self, ingredient: str) -> set[str]:
        """Return the set of dietary categories for *ingredient*.

        Curated overrides take precedence over token-leaf matching.
        Each matched leaf is expanded to include all ancestor categories.
        """
        ing = canonicalize(ingredient)
        if ing in _CURATED:
            leaves = set(_CURATED[ing])
        else:
            leaves = {_LEAF_WORDS[w] for w in ing.split() if w in _LEAF_WORDS}
        out: set[str] = set()
        for leaf in leaves:
            out |= _ancestors(leaf)
        return out

    def _is_known(self, ingredient: str) -> bool:
        """Return True if the ingredient is in the vocab, curated table, or has leaf-token tags."""
        ing = canonicalize(ingredient)
        if ing in self._known or ing in _CURATED:
            return True
        # Known if any token maps to a leaf word
        return any(w in _LEAF_WORDS for w in ing.split())

    def is_valid(self, ingredient: str, diet: str | None) -> bool:
        """Return True if *ingredient* is allowed under *diet*.

        Unknown ingredients are conservatively excluded when a constrained diet
        is specified, since we cannot assert they are safe.
        """
        if not diet:
            return True
        if diet not in _DIETS:
            raise ValueError(f"unknown diet {diet!r}; expected one of {sorted(_DIETS)}")
        if not self._is_known(ingredient):
            return False  # conservative: don't assume an untagged ingredient is safe
        return not (self.tags(ingredient) & _DIETS[diet])

    def mask(
        self, candidates: Sequence[tuple[str, float]], diet: str | None
    ) -> list[tuple[str, float]]:
        """Filter *candidates* to those valid under *diet*."""
        return [(ing, s) for ing, s in candidates if self.is_valid(ing, diet)]

    def coverage(self, vocab: Sequence[str]) -> float:
        """Fraction of *vocab* items we can make a definitive dietary judgment about."""
        if not vocab:
            return 0.0
        judged = sum(1 for w in vocab if self._is_known(w))
        return judged / len(vocab)
