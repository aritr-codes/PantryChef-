"""Minimal dietary tagging for the Phase 2 guardrail (full ontology -> Phase 4).

Keyword rules over canonical ingredient strings + a curated override table for
cases keywords miss (honey, gelatin, broth, ...). Applied as a HARD mask so the
guardrail is 100% by construction for *tagged* constraints. Unknowns default to
conservative EXCLUSION under a constrained diet (an untagged ingredient is not
assumed safe). `coverage()` reports how much of the vocab we can actually judge.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from pantrychef.ingredients.normalize import canonicalize

# category -> substrings that imply it (matched on canonicalized text)
_KEYWORDS: dict[str, tuple[str, ...]] = {
    "meat": (
        "beef",
        "pork",
        "chicken",
        "bacon",
        "ham",
        "sausage",
        "lamb",
        "turkey",
        "veal",
        "lard",
    ),
    "fish": ("fish", "salmon", "tuna", "shrimp", "anchovy", "cod", "crab", "prawn"),
    "dairy": ("milk", "butter", "cheese", "cream", "yogurt", "ghee", "casein", "whey"),
    "egg": ("egg",),
    "honey": ("honey",),
    "gluten": (
        "wheat",
        "barley",
        "rye",
        "bread",
        "pasta",
        "flour",
        "couscous",
        "semolina",
    ),
}

# explicit overrides (canonical form -> categories) for cases keywords miss/over-match
_CURATED: dict[str, set[str]] = {
    "gelatin": {"meat"},
    "worcestershire sauce": {"fish"},
    "fish sauce": {"fish"},
    "soy sauce": {"gluten"},
    "oat": set(),  # oats are GF unless cross-contaminated; don't flag gluten
    "almond flour": set(),
    "coconut flour": set(),
    "rice flour": set(),
}

# diet -> forbidden categories
_DIETS: dict[str, set[str]] = {
    "vegan": {"meat", "fish", "dairy", "egg", "honey"},
    "vegetarian": {"meat", "fish"},
    "gluten_free": {"gluten"},
    "dairy_free": {"dairy"},
}


class DietTagger:
    """Tag ingredients with dietary categories and apply hard-mask filtering."""

    def __init__(self, known: Iterable[str]) -> None:
        self._known = {canonicalize(w) for w in known}

    def tags(self, ingredient: str) -> set[str]:
        """Return the set of dietary categories for *ingredient*.

        Curated overrides take precedence over keyword matching.
        """
        ing = canonicalize(ingredient)
        if ing in _CURATED:
            return set(_CURATED[ing])
        found: set[str] = set()
        for cat, kws in _KEYWORDS.items():
            if any(kw in ing for kw in kws):
                found.add(cat)
        return found

    def _is_known(self, ingredient: str) -> bool:
        """Return True if the ingredient is in the vocab, curated table, or has keyword tags."""
        ing = canonicalize(ingredient)
        return ing in self._known or ing in _CURATED or bool(self.tags(ing))

    def is_valid(self, ingredient: str, diet: str | None) -> bool:
        """Return True if *ingredient* is allowed under *diet*.

        Unknown ingredients are conservatively excluded when a constrained diet
        is specified, since we cannot assert they are safe.
        """
        if not diet or diet not in _DIETS:
            return True
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
