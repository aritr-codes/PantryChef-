"""word2vec skip-gram over ingredient sets — the food2vec baseline arm.

Learns distributional *relatedness* (ingredients used in similar recipes), NOT
substitutability directly (see spec). kNN by cosine gives candidate subs.
CPU-only; workers=1 + seed for reproducibility.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from gensim.models import KeyedVectors, Word2Vec

from pantrychef.substitution.config import SubConfig
from pantrychef.substitution.corpus import IngredientCorpus


def train_word2vec(corpus: IngredientCorpus, cfg: SubConfig) -> KeyedVectors:
    """Train a skip-gram Word2Vec model on ingredient co-occurrence sets.

    Returns the trained KeyedVectors (weights only; model state discarded).
    """
    model = Word2Vec(
        sentences=corpus,
        sg=1,
        vector_size=cfg.dims,
        window=cfg.window,
        min_count=cfg.min_count,
        negative=cfg.negative,
        epochs=cfg.epochs,
        workers=1,
        seed=cfg.seed,
    )
    return model.wv


class EmbeddingModel:
    """kNN wrapper over trained ingredient vectors."""

    def __init__(self, wv: KeyedVectors) -> None:
        self._wv = wv

    @property
    def vocab(self) -> list[str]:
        """Return all tokens in the trained vocabulary."""
        return list(self._wv.index_to_key)

    def neighbors(self, ingredient: str, k: int = 5) -> list[tuple[str, float]]:
        """Return the k nearest neighbors by cosine similarity.

        Returns an empty list if the ingredient is not in the vocabulary or k <= 0.
        """
        if ingredient not in self._wv or k <= 0:
            return []
        return [(w, float(s)) for w, s in self._wv.most_similar(ingredient, topn=k)]

    def similarity(self, a: str, b: str) -> float:
        """Return cosine similarity between ``a`` and ``b``.

        Returns 0.0 if either token is absent from the vocabulary.
        """
        if a not in self._wv or b not in self._wv:
            return 0.0
        return float(self._wv.similarity(a, b))

    def to_state(self) -> dict[str, object]:
        """Return a pickle-free serialization state for bundle persistence."""
        return {
            "vocab": self.vocab,
            "vectors": np.asarray(self._wv.vectors, dtype=np.float32),
        }

    @classmethod
    def from_state(cls, vocab: Sequence[str], vectors: np.ndarray) -> EmbeddingModel:
        """Rebuild an EmbeddingModel from a persisted vocab + vector matrix."""
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] == 0:
            raise ValueError("vectors must be a non-empty 2D array")
        if len(vocab) != arr.shape[0]:
            raise ValueError(
                f"vocab and vectors must align: {len(vocab)} != {arr.shape[0]}"
            )
        wv = KeyedVectors(vector_size=int(arr.shape[1]))
        wv.add_vectors(list(vocab), arr)
        return cls(wv)

    def save(self, path: str | Path) -> None:
        """Persist the KeyedVectors to *path* (gensim native format)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._wv.save(str(p))

    @classmethod
    def load(cls, path: str | Path) -> EmbeddingModel:
        """Load a previously saved EmbeddingModel from *path*.

        .. warning::
            Uses gensim's native (pickle-based) format. Only load files from
            trusted sources — never from user-supplied or externally-sourced paths.
        """
        return cls(KeyedVectors.load(str(Path(path))))
