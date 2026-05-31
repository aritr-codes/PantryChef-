"""Ingredient co-occurrence matrix and SPPMI weighting.

Co-occurrence within a recipe is *complementarity*; SPPMI rows are the context
fingerprints the second-order graph compares. SPPMI = shifted PPMI
(max(PMI - log(shift), 0)) — smoothing/shift tame rare-ingredient instability.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
from scipy import sparse

from pantrychef.common.types import Recipe


def build_cooccurrence(
    recipes: Sequence[Recipe], vocab: Sequence[str]
) -> tuple[sparse.csr_matrix, Counter[str], int, dict[str, int]]:
    """Build a symmetric ingredient co-occurrence matrix from a recipe corpus.

    Parameters
    ----------
    recipes:
        Collection of Recipe objects. Ingredients not in *vocab* are silently
        dropped (OOV handling).
    vocab:
        Ordered vocabulary; position determines the matrix row/column index.

    Returns
    -------
    cmat:
        Symmetric CSR matrix of shape (|vocab|, |vocab|) with raw co-occurrence
        counts. Diagonal is always zero (no self-loops).
    word_df:
        Document-frequency counter — number of recipes each vocab term appears in.
    n_recipes:
        Total number of recipes processed (including those with no in-vocab items).
    idx:
        Mapping from vocab term to matrix index.
    """
    idx = {w: i for i, w in enumerate(vocab)}
    v = len(vocab)
    pair: Counter[tuple[int, int]] = Counter()
    word_df: Counter[str] = Counter()
    n_recipes = 0
    for r in recipes:
        n_recipes += 1
        items = sorted({idx[c] for c in r.canonical if c in idx})
        for i in items:
            word_df[vocab[i]] += 1
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                pair[(items[a], items[b])] += 1
    rows: list[int] = []
    cols: list[int] = []
    data: list[int] = []
    for (i, j), c in pair.items():
        rows += [i, j]
        cols += [j, i]
        data += [c, c]
    cmat = sparse.csr_matrix((data, (rows, cols)), shape=(v, v))
    return cmat, word_df, n_recipes, idx


def sppmi(cmat: sparse.csr_matrix, shift: float = 1.0) -> sparse.csr_matrix:
    """Compute Shifted Positive PMI from a raw co-occurrence matrix.

    SPPMI(i, j) = max(PMI(i, j) - log(shift), 0). The shift parameter (default
    1.0, i.e. standard PPMI) down-weights low-PMI pairs and zeroes out negatives,
    making the matrix sparse and non-negative — suitable as embedding input.

    Parameters
    ----------
    cmat:
        Symmetric CSR co-occurrence matrix produced by :func:`build_cooccurrence`.
    shift:
        Smoothing shift applied as ``log(shift)`` before clipping. Must be >= 1.

    Returns
    -------
    Sparse CSR matrix of the same shape with SPPMI values.
    """
    if shift < 1.0:
        raise ValueError(f"shift must be >= 1.0, got {shift}")
    coo = cmat.tocoo()
    if coo.nnz == 0:
        return cmat.tocsr()
    total = float(coo.data.sum())
    row_sums = np.asarray(cmat.sum(axis=1)).ravel().astype(float)
    pmi = np.log((coo.data * total) / (row_sums[coo.row] * row_sums[coo.col]))
    vals = np.maximum(pmi - np.log(shift), 0.0)
    m = sparse.csr_matrix((vals, (coo.row, coo.col)), shape=cmat.shape)
    m.eliminate_zeros()
    return m


def l2norm_rows(m: sparse.csr_matrix) -> sparse.csr_matrix:
    """Row-normalise a sparse matrix so each row has unit L2 norm.

    Zero rows (all-zero ingredients) are left as zero vectors rather than
    producing NaN. This makes cosine similarity well-defined for non-zero rows.

    Parameters
    ----------
    m:
        Input sparse matrix (e.g. SPPMI output).

    Returns
    -------
    Row-normalised sparse matrix of the same shape.
    """
    norms = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    return sparse.diags(1.0 / norms) @ m
