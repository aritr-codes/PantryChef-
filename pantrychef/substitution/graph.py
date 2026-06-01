"""Second-order context graph — the flagship substitution arm.

score(a, b) = cos(SPPMI_row_a, SPPMI_row_b) - lam * soft_penalty(cooccur(a, b))

Substitutes share co-occurrence neighborhoods (similar SPPMI rows) but rarely
co-occur directly. The penalty is *soft* (normalized co-occurrence, not a hard
cut) so pairs that sometimes appear together (butter+oil) aren't zeroed.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds

from pantrychef.substitution.cooccur import l2norm_rows


class ContextGraph:
    """Second-order ingredient graph backed by SPPMI cosine similarity.

    Parameters
    ----------
    sppmi_matrix:
        Sparse SPPMI matrix of shape (|vocab|, |vocab|), e.g. from
        :func:`pantrychef.substitution.cooccur.sppmi`.
    vocab:
        Ordered vocabulary matching the matrix row/column indices.
    cooccur:
        Raw co-occurrence matrix (same shape) used to penalise direct
        co-occurrence (complements).
    lam:
        Co-occurrence penalty weight in [0, 1]. ``lam=0`` gives pure
        second-order cosine similarity; ``lam=1`` gives maximum penalty.
    """

    def __init__(
        self,
        sppmi_matrix: sparse.csr_matrix,
        vocab: Sequence[str],
        cooccur: sparse.csr_matrix,
        lam: float = 0.5,
        overlap_shrink: float = 0.0,
    ) -> None:
        if overlap_shrink < 0 or not np.isfinite(overlap_shrink):
            raise ValueError(f"overlap_shrink must be >= 0 and finite; got {overlap_shrink}")
        n = len(vocab)
        if sppmi_matrix.shape != (n, n) or cooccur.shape != (n, n):
            raise ValueError(
                "sppmi_matrix, cooccur, and vocab must agree in size: "
                f"expected ({n}, {n}), got sppmi={sppmi_matrix.shape} cooccur={cooccur.shape}"
            )
        self.vocab = list(vocab)
        self.idx = {w: i for i, w in enumerate(self.vocab)}
        self._mn = l2norm_rows(sppmi_matrix).tocsr()
        self._c = cooccur.tocsr()
        self.lam = lam
        self._beta = overlap_shrink
        # Binary nonzero pattern for overlap counting (precomputed once; only
        # needed when shrinkage is active). `!= 0` drops any explicit zeros.
        if overlap_shrink > 0:
            spb = (sppmi_matrix != 0).astype(np.int32).tocsr()
            spb.eliminate_zeros()
            self._spb: sparse.csr_matrix | None = spb
        else:
            self._spb = None

    def neighbors(
        self, ingredient: str, k: int = 5, lam: float | None = None
    ) -> list[tuple[str, float]]:
        """Return the top-k substitution candidates for *ingredient*.

        Parameters
        ----------
        ingredient:
            Query ingredient (must be in vocab). Returns ``[]`` if unknown.
        k:
            Number of neighbors to return.
        lam:
            Override the instance-level lambda for this call only.

        Returns
        -------
        List of ``(name, score)`` tuples sorted by descending score. The query
        ingredient itself is excluded. Returns ``[]`` for unknown ingredients.
        """
        i = self.idx.get(ingredient)
        if i is None:
            return []
        if k <= 0:
            return []
        if self._mn[i].nnz == 0:
            return []
        lam = self.lam if lam is None else lam
        sim = np.asarray((self._mn @ self._mn[i].T).toarray()).ravel()
        if self._beta > 0 and self._spb is not None:
            ov = np.asarray((self._spb @ self._spb[i].T).toarray()).ravel().astype(float)
            sim = sim * (ov / (ov + self._beta))
        crow = np.asarray(self._c[i].toarray()).ravel().astype(float)
        cmax = crow.max()
        penalty = crow / cmax if cmax > 0 else crow
        score = sim - lam * penalty
        score[i] = -np.inf
        order = np.argsort(-score, kind="stable")[:k]
        return [(self.vocab[j], float(score[j])) for j in order if np.isfinite(score[j])]


class SvdContextModel:
    """Dense SPPMI+SVD embeddings; cosine kNN. The SPPMI+SVD baseline arm."""

    def __init__(self, vectors: np.ndarray, vocab: Sequence[str]) -> None:
        self.vocab = list(vocab)
        self.idx = {w: i for i, w in enumerate(self.vocab)}
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._v = vectors / norms

    @classmethod
    def fit(
        cls, sppmi_matrix: sparse.csr_matrix, vocab: Sequence[str], dims: int
    ) -> SvdContextModel:
        """Fit SVD on an SPPMI matrix and return a normalised embedding model."""
        d = min(dims, min(sppmi_matrix.shape) - 1)
        u, s, _ = svds(sppmi_matrix.asfptype(), k=d)
        return cls(u * s, vocab)

    def neighbors(self, ingredient: str, k: int = 5) -> list[tuple[str, float]]:
        """Return top-k cosine-similar ingredients by embedding, excluding the query."""
        if k <= 0:
            return []
        i = self.idx.get(ingredient)
        if i is None:
            return []
        sim = self._v @ self._v[i]
        sim[i] = -np.inf
        order = np.argsort(-sim, kind="stable")[:k]
        return [(self.vocab[j], float(sim[j])) for j in order if np.isfinite(sim[j])]
