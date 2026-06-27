"""Reranker model wrappers behind one interface.

LinearRanker: numpy logistic regression (pointwise baseline), standardized with
TRAIN-only mean/std. LambdaMARTRanker: LightGBM lambdarank flagship. Both expose
fit(X, y, groups) / score(X); `groups` is the per-query candidate counts (used by
lambdarank; ignored by the linear model).
"""

from __future__ import annotations

from typing import Any, Protocol

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
        if X.ndim != 2 or y.ndim != 1 or len(y) != len(X):
            raise ValueError(f"shape mismatch: X={X.shape}, y={y.shape}")
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


    def to_state(self) -> dict[str, Any]:
        """Return the fitted linear model state for artifact persistence."""
        if self.w_ is None or self.mean_ is None or self.std_ is None:
            raise RuntimeError("LinearRanker not fitted")
        return {
            "lr": self.lr,
            "epochs": self.epochs,
            "seed": self.seed,
            "w": np.asarray(self.w_, dtype=float),
            "b": float(self.b_),
            "mean": np.asarray(self.mean_, dtype=float),
            "std": np.asarray(self.std_, dtype=float),
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> LinearRanker:
        """Restore a fitted linear model from persisted state."""
        model = cls(
            lr=float(state["lr"]),
            epochs=int(state["epochs"]),
            seed=int(state["seed"]),
        )
        model.w_ = np.asarray(state["w"], dtype=float)
        model.b_ = float(state["b"])
        model.mean_ = np.asarray(state["mean"], dtype=float)
        model.std_ = np.asarray(state["std"], dtype=float)
        return model

class LambdaMARTRanker:
    """LightGBM LambdaMART (objective='lambdarank'). Lazy import keeps core clean."""

    def __init__(
        self,
        num_leaves: int = 31,
        n_estimators: int = 200,
        min_child_samples: int = 5,
        seed: int = 13,
    ) -> None:
        self.num_leaves = num_leaves
        self.n_estimators = n_estimators
        self.min_child_samples = min_child_samples
        self.seed = seed
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray, groups: list[int]) -> LambdaMARTRanker:
        """Fit the LambdaMART ranker.

        Labels y are cast to int (relevance grades); this task uses {0, 1}.
        """
        if sum(groups) != len(X):
            raise ValueError(f"sum(groups)={sum(groups)} must equal len(X)={len(X)}")
        try:
            import lightgbm as lgb
            from sklearn.base import BaseEstimator as _unused  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "LambdaMARTRanker needs the [recommend] extra "
                "(lightgbm + scikit-learn): uv sync --extra recommend"
            ) from e
        self._model = lgb.LGBMRanker(
            objective="lambdarank",
            num_leaves=self.num_leaves,
            n_estimators=self.n_estimators,
            random_state=self.seed,
            min_child_samples=self.min_child_samples,
            verbose=-1,
        )
        self._model.fit(X, y.astype(int), group=groups)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("LambdaMARTRanker not fitted")
        return np.asarray(self._model.predict(X))


    def to_state(self) -> dict[str, Any]:
        """Return the fitted LambdaMART state for artifact persistence."""
        if self._model is None:
            raise RuntimeError("LambdaMARTRanker not fitted")
        booster = self._model.booster_ if hasattr(self._model, "booster_") else self._model
        return {
            "num_leaves": self.num_leaves,
            "n_estimators": self.n_estimators,
            "min_child_samples": self.min_child_samples,
            "seed": self.seed,
            "booster": booster.model_to_string(),
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> LambdaMARTRanker:
        """Restore a fitted LambdaMART model from persisted state."""
        try:
            import lightgbm as lgb
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "LambdaMARTRanker needs the [recommend] extra "
                "(lightgbm + scikit-learn): uv sync --extra recommend"
            ) from e
        model = cls(
            num_leaves=int(state["num_leaves"]),
            n_estimators=int(state["n_estimators"]),
            min_child_samples=int(state["min_child_samples"]),
            seed=int(state["seed"]),
        )
        model._model = lgb.Booster(model_str=str(state["booster"]))
        return model

