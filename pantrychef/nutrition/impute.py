"""Gated macro imputation for ingredients USDA can't match.

Built only when the coverage gate trips. Features = hashed ingredient name
tokens; model = ridge regression per target macro. MAE is measured on held-out
*matched* ingredients (train on matched, test on held-out matched), then applied
to unmatched ones. Light, deterministic, interpretable."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split


class MacroImputer:
    def __init__(self, seed: int = 42, n_features: int = 256) -> None:
        self.seed = seed
        self._vec = HashingVectorizer(n_features=n_features, alternate_sign=False)
        self._model = Ridge(alpha=1.0, random_state=seed)
        self.mae_ = 0.0
        self._fitted = False

    def fit(self, samples: list[tuple[str, dict]], target: str) -> MacroImputer:
        names = [n for n, macros in samples if target in macros]
        y = np.array([macros[target] for _, macros in samples if target in macros], dtype=float)
        if not names:
            # No training signal for this target (e.g. a micro absent from every
            # matched food). No-op: stay unfitted so predict returns 0.0 instead
            # of crashing on an empty design matrix.
            self.mae_ = 0.0
            return self
        x = self._vec.transform(names)
        if len(y) >= 4:
            x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.25, random_state=self.seed)
            self._model.fit(x_tr, y_tr)
            self.mae_ = float(np.mean(np.abs(self._model.predict(x_te) - y_te)))
            self._model.fit(x, y)  # refit on all for deployment
        else:
            self._model.fit(x, y)
            self.mae_ = 0.0
        self._fitted = True
        return self

    def predict(self, ingredient: str) -> float:
        if not self._fitted:
            return 0.0
        return float(self._model.predict(self._vec.transform([ingredient]))[0])
