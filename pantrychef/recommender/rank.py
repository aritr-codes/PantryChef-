"""Reranker model wrappers behind one interface.

LinearRanker: numpy logistic regression (pointwise baseline), standardized with
TRAIN-only mean/std. LambdaMARTRanker: LightGBM lambdarank flagship. Both expose
fit(X, y, groups) / score(X); `groups` is the per-query candidate counts (used by
lambdarank; ignored by the linear model).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Ranker(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray, groups: list[int]) -> Ranker: ...

    def score(self, X: np.ndarray) -> np.ndarray: ...


class LinearRanker:
    """Logistic regression via gradient descent. Ordering uses the raw logit."""

    def __init__(self, lr: float = 0.1, epochs: int = 500, seed: int = 13) -> None:
        self.lr = lr
        self.epochs = epochs
        self.seed = seed
        self.w_: np.ndarray | None = None
        self.b_: float = 0.0
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit(self, X: np.ndarray, y: np.ndarray, groups: list[int]) -> LinearRanker:
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0  # guard constant columns
        Xs = self._standardize(X)
        rng = np.random.default_rng(self.seed)
        n, d = Xs.shape
        self.w_ = rng.normal(0, 0.01, d)
        self.b_ = 0.0
        for _ in range(self.epochs):
            z = Xs @ self.w_ + self.b_
            p = 1.0 / (1.0 + np.exp(-z))
            err = p - y
            self.w_ -= self.lr * (Xs.T @ err) / n
            self.b_ -= self.lr * err.mean()
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.w_ is None or self.mean_ is None:
            raise RuntimeError("LinearRanker not fitted")
        return self._standardize(X) @ self.w_ + self.b_
