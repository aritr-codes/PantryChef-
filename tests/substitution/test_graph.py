from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse import eye

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


# ---------------------------------------------------------------------------
# Existing behavioural tests (kept verbatim)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Fix 1 — k<=0 guard
# ---------------------------------------------------------------------------

def test_k_zero_or_negative() -> None:
    g = _graph()
    assert g.neighbors("butter", k=0) == []
    assert g.neighbors("butter", k=-1) == []


# ---------------------------------------------------------------------------
# Fix 2 — zero-context guard
# ---------------------------------------------------------------------------

def test_zero_context_returns_empty() -> None:
    """An ingredient that appears only in singleton recipes has an all-zero
    SPPMI row (no co-occurrence partners, so no positive PMI).  neighbors()
    must return [] rather than meaningless zero-scored results."""
    vocab_solo = ["butter", "flour", "egg", "milk", "lard"]
    recipes_solo = _BUTTER_RECIPES + [
        Recipe(recipe_id="solo0", title="t", canonical=["lard"])
    ]
    cmat, _, _, _ = build_cooccurrence(recipes_solo, vocab_solo)
    m = sppmi(cmat, shift=1.0)

    # Confirm the fixture is actually zero (fail loudly if the data builder changes)
    lard_idx = vocab_solo.index("lard")
    assert m[lard_idx].nnz == 0, (
        "Fixture broken: lard's SPPMI row should be all-zero (singleton recipe)"
    )

    g = ContextGraph(m, vocab_solo, cmat, lam=0.5)
    assert g.neighbors("lard", k=3) == []


# ---------------------------------------------------------------------------
# Fix 3 — shape validation in __init__
# ---------------------------------------------------------------------------

def test_shape_mismatch_raises() -> None:
    """Passing matrices whose shape disagrees with vocab length must raise."""
    m3 = eye(3, format="csr")
    vocab2 = ["a", "b"]
    with pytest.raises(ValueError):
        ContextGraph(m3, vocab2, m3)


def test_rectangular_matrix_raises() -> None:
    """Non-square (rectangular) matrices must also be rejected."""
    from scipy.sparse import csr_matrix
    m_rect = csr_matrix((3, 4))
    vocab3 = ["a", "b", "c"]
    m_sq3 = eye(3, format="csr")
    with pytest.raises(ValueError):
        ContextGraph(m_rect, vocab3, m_sq3)


# ---------------------------------------------------------------------------
# Fix 4 — penalty-isolating test (flagship claim)
# ---------------------------------------------------------------------------

VOCAB2 = ["butter", "oil", "shortening", "flour", "egg", "milk"]
# butter and oil: share flour/egg/milk context, never co-occur directly.
# shortening: same flour/egg/milk context AS oil, but ALSO co-occurs with butter
#   directly in 10 extra recipes — so its direct co-occurrence penalty > 0.
RECIPES2 = (
    [Recipe(recipe_id=f"b{i}", title="t", canonical=["butter", "flour", "egg", "milk"])
     for i in range(20)]
    + [Recipe(recipe_id=f"o{i}", title="t", canonical=["oil", "flour", "egg", "milk"])
       for i in range(20)]
    + [Recipe(recipe_id=f"s{i}", title="t", canonical=["shortening", "flour", "egg", "milk"])
       for i in range(20)]
    # shortening co-occurs with butter directly:
    + [Recipe(recipe_id=f"bs{i}", title="t",
              canonical=["butter", "shortening", "flour", "egg", "milk"])
       for i in range(10)]
)


def _graph2(lam: float = 0.5) -> ContextGraph:
    cmat, _, _, _ = build_cooccurrence(RECIPES2, VOCAB2)
    m = sppmi(cmat, shift=1.0)
    return ContextGraph(m, list(VOCAB2), cmat, lam=lam)


def test_penalty_demotes_cooccurring_rival() -> None:
    """The co-occurrence penalty must meaningfully change scores.

    Setup guarantees:
    - oil and shortening have identical second-order context (flour/egg/milk).
    - shortening directly co-occurs with butter; oil never does.
    - So penalty(shortening) > penalty(oil) == 0.

    Assertions:
    1. Under lam=0.5: oil scores strictly above shortening (penalty demotes rival).
    2. shortening's score is strictly lower at lam=0.5 than at lam=0.0 (penalty
       reduces it).
    3. oil's score is unchanged between lam=0.5 and lam=0.0 (zero penalty, so no
       change — proves the effect is localised to the penalised ingredient).
    """
    nbrs_lam0 = dict(_graph2(lam=0.0).neighbors("butter", k=5))
    nbrs_lam05 = dict(_graph2(lam=0.5).neighbors("butter", k=5))

    assert "oil" in nbrs_lam05, "oil should be a butter neighbor under lam=0.5"
    assert "shortening" in nbrs_lam05, "shortening should be a butter neighbor under lam=0.5"

    # Core claim: penalty demotes shortening below oil.
    assert nbrs_lam05["oil"] > nbrs_lam05["shortening"], (
        "Under lam=0.5, oil (zero direct co-occurrence) must score above "
        "shortening (non-zero direct co-occurrence with butter)"
    )

    # Penalty reduces shortening's score vs lam=0.
    assert nbrs_lam05["shortening"] < nbrs_lam0["shortening"], (
        "shortening's score must drop when lam increases from 0 to 0.5"
    )

    # Oil is unaffected (penalty == 0 for oil since it never co-occurs with butter).
    assert nbrs_lam05["oil"] == nbrs_lam0["oil"], (
        "oil's score must not change with lam since its co-occurrence penalty is 0"
    )


def test_svd_model_neighbors() -> None:
    from pantrychef.substitution.cooccur import build_cooccurrence, sppmi
    from pantrychef.substitution.graph import SvdContextModel

    cmat, _, _, _ = build_cooccurrence(RECIPES, VOCAB)
    m = sppmi(cmat, shift=1.0)
    model = SvdContextModel.fit(m, list(VOCAB), dims=3)
    nbrs = model.neighbors("butter", k=2)
    assert all(n != "butter" for n, _ in nbrs)
    assert "oil" in [n for n, _ in nbrs]


# ---------------------------------------------------------------------------
# Overlap-shrinkage tests
# ---------------------------------------------------------------------------

def _make_overlap_matrices():
    """Build a deterministic 4x4 SPPMI + cooccur by hand.

    Vocab: ["A", "B", "C", "D"]
    A is the query.
    B shares 3 context columns with A  -> high overlap candidate.
    C shares 0 context columns with A  -> low overlap, but has high raw cosine
       (we engineer this by giving C an identical normalised row via SPPMI
       values that happen to align when normalised — easier: just use a dense
       hand-crafted matrix).
    D shares 1 context column with A   -> mid overlap.

    We craft the SPPMI so that under legacy (no shrinkage):
      cos(A, C) > cos(A, B)  [C outranks B]
    And under shrinkage beta=10:
      score(A, B) > score(A, C)  [B recovers lead]

    Cosine is computed on l2-normalised rows.  We craft raw values so:
      - A row: [3, 3, 3, 3] (all 4 context dims)
      - B row: [3, 3, 3, 0] (shares cols 0,1,2 with A → ov=3)
      - C row: [0, 0, 0, 9] (shares col 3 with A → ov=1, but large value
                              → high raw cosine after normalisation)
      - D row: [0, 0, 0, 3] (shares col 3 with A → ov=1, moderate cosine)

    cos(A, B): A_norm=[1/2,1/2,1/2,1/2], B_norm=[1/√3,1/√3,1/√3,0]
               dot = 3/(2√3) ≈ 0.866
    cos(A, C): A_norm=[1/2]*4, C_norm=[0,0,0,1]
               dot = 1/2 = 0.500
    cos(A, D): A_norm=[1/2]*4, D_norm=[0,0,0,1]
               dot = 1/2 = 0.500

    Wait — that makes B rank above C even before shrinkage. Adjust so C has
    higher raw cosine than B:

      - A row: [2, 0, 0, 2]  (context cols 0 and 3)
      - B row: [2, 2, 2, 2]  (all cols; shares col 0 and 3 with A → ov=2)
      - C row: [0, 0, 0, 9]  (only col 3 → ov=1 with A)
      - D row: [0, 0, 0, 0]  (no context — padding row so matrix is 4x4)

    A_norm = [1/√2, 0, 0, 1/√2]
    B_norm = [1/2, 1/2, 1/2, 1/2]  cos(A,B) = (1/√2)(1/2) + (1/√2)(1/2) = 1/√2 ≈ 0.707
    C_norm = [0, 0, 0, 1]          cos(A,C) = 1/√2 ≈ 0.707  (same — need to break tie)

    Adjust C to give strictly higher cosine:
      - A row: [1, 0, 0, 2]   A_norm = [1/√5, 0, 0, 2/√5]
      - B row: [2, 2, 2, 2]   B_norm = [1/2, 1/2, 1/2, 1/2]
               cos(A,B) = (1/√5)(1/2) + (2/√5)(1/2) = 3/(2√5) ≈ 0.671   ov=2 (cols 0,3)
      - C row: [0, 0, 0, 4]   C_norm = [0,0,0,1]
               cos(A,C) = (2/√5)(1) = 2/√5 ≈ 0.894                        ov=1 (col 3 only)
      - D row: [0, 0, 0, 0]   (zero row, padding)

    Legacy: C (0.894) > B (0.671)  → C ranks above B.
    With beta=10:
      score(A,B) = 0.671 * 2/(2+10) = 0.671 * 0.167 ≈ 0.112
      score(A,C) = 0.894 * 1/(1+10) = 0.894 * 0.091 ≈ 0.081
    So B > C after shrinkage. ✓
    """
    vocab = ["A", "B", "C", "D"]
    # 4x4 SPPMI matrix (row = ingredient, col = context dimension = same vocab here)
    data = np.array([
        # A: cols 0,3
        [1.0, 0.0, 0.0, 2.0],
        # B: cols 0,1,2,3
        [2.0, 2.0, 2.0, 2.0],
        # C: col 3 only
        [0.0, 0.0, 0.0, 4.0],
        # D: zero row
        [0.0, 0.0, 0.0, 0.0],
    ], dtype=np.float64)
    sppmi = sp.csr_matrix(data)
    # cooccur: all zeros (no penalty)
    cooccur = sp.csr_matrix((4, 4), dtype=np.float64)
    return sppmi, cooccur, vocab


def test_overlap_shrink_zero_is_legacy() -> None:
    """ContextGraph(overlap_shrink=0.0) must produce byte-identical results to
    the default (no overlap_shrink kwarg) — backward-compatibility lock."""
    sppmi_mat, cooccur, vocab = _make_overlap_matrices()

    g_default = ContextGraph(sppmi_mat, vocab, cooccur, lam=0.0)
    g_explicit = ContextGraph(sppmi_mat, vocab, cooccur, lam=0.0, overlap_shrink=0.0)

    nbrs_default = g_default.neighbors("A", k=3)
    nbrs_explicit = g_explicit.neighbors("A", k=3)

    assert nbrs_default == nbrs_explicit, (
        f"overlap_shrink=0.0 must be identical to default: "
        f"default={nbrs_default} explicit={nbrs_explicit}"
    )


def test_overlap_shrink_downweights_low_overlap() -> None:
    """With beta=0 (legacy), low-overlap candidate C outranks high-overlap B.
    With overlap_shrink=10 (beta=10), B must recover and rank above C.

    See _make_overlap_matrices() docstring for the exact construction.
    """
    sppmi_mat, cooccur, vocab = _make_overlap_matrices()

    g_legacy = ContextGraph(sppmi_mat, vocab, cooccur, lam=0.0, overlap_shrink=0.0)
    g_shrink = ContextGraph(sppmi_mat, vocab, cooccur, lam=0.0, overlap_shrink=10.0)

    nbrs_legacy = dict(g_legacy.neighbors("A", k=3))
    nbrs_shrink = dict(g_shrink.neighbors("A", k=3))

    # Under legacy: C must rank above B (C has higher raw cosine)
    assert "B" in nbrs_legacy and "C" in nbrs_legacy, (
        f"Legacy neighbors must include B and C: {nbrs_legacy}"
    )
    assert nbrs_legacy["C"] > nbrs_legacy["B"], (
        f"Legacy: C (low-overlap, high cosine) must outscore B: "
        f"C={nbrs_legacy['C']:.4f} B={nbrs_legacy['B']:.4f}"
    )

    # Under shrinkage: B (ov=2) must rank above C (ov=1)
    assert "B" in nbrs_shrink and "C" in nbrs_shrink, (
        f"Shrink neighbors must include B and C: {nbrs_shrink}"
    )
    assert nbrs_shrink["B"] > nbrs_shrink["C"], (
        f"Shrinkage: B (high-overlap) must outscore C (low-overlap): "
        f"B={nbrs_shrink['B']:.4f} C={nbrs_shrink['C']:.4f}"
    )
