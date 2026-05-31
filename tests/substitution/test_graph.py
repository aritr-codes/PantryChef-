from __future__ import annotations

from pantrychef.common.types import Recipe
from pantrychef.substitution.cooccur import build_cooccurrence, sppmi
from pantrychef.substitution.graph import ContextGraph

VOCAB = ["butter", "oil", "flour", "egg", "milk"]
# butter-recipes and oil-recipes share {flour,egg,milk}; butter & oil never co-occur.
_BUTTER_RECIPES = [
    Recipe(recipe_id=f"b{i}", title="t", canonical=["butter", "flour", "egg", "milk"])
    for i in range(20)
]
_OIL_RECIPES = [
    Recipe(recipe_id=f"o{i}", title="t", canonical=["oil", "flour", "egg", "milk"])
    for i in range(20)
]
RECIPES = _BUTTER_RECIPES + _OIL_RECIPES


def _graph(lam: float = 0.5) -> ContextGraph:
    cmat, _, _, _ = build_cooccurrence(RECIPES, VOCAB)
    m = sppmi(cmat, shift=1.0)
    return ContextGraph(m, list(VOCAB), cmat, lam=lam)


def test_substitute_beats_complement() -> None:
    nbrs = _graph().neighbors("butter", k=4)
    names = [n for n, _ in nbrs]
    assert names[0] == "oil"  # shares context, never co-occurs => top sub
    assert names.index("oil") < names.index("flour")  # sub ranked above complement


def test_excludes_self_and_unknown() -> None:
    g = _graph()
    assert all(n != "butter" for n, _ in g.neighbors("butter", k=4))
    assert g.neighbors("dragonfruit", k=3) == []


def test_lambda_zero_is_pure_context() -> None:
    # With lam=0 the score is pure cosine of SPPMI rows (no co-occ penalty).
    nbrs = _graph(lam=0.0).neighbors("butter", k=1)
    assert nbrs[0][0] == "oil"
