"""Recipe ingredient sets as word2vec 'sentences'.

Each recipe's canonical set is one sentence. Tokens are re-shuffled on every
iteration pass (= every training epoch) so gensim's reduced-window sampling
sees all in-recipe pairs as near-context over the run. Seeded for repro: a
fresh corpus with the same seed replays the same sequence of shuffles.

Design notes:
- Single-consumer only: the shared RNG is not thread-safe across concurrent
  iterators over the same instance.
- Any pass over the corpus (e.g. for vocab building) advances the RNG before
  training. Use a separate IngredientCorpus instance for vocab vs. training
  if epoch-0 ordering must be reproducible end-to-end.
"""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence

from pantrychef.common.types import Recipe


class IngredientCorpus:
    def __init__(self, recipes: Sequence[Recipe], seed: int = 42) -> None:
        # Materialize non-empty token lists once; shuffle copies per pass.
        self._sentences = [list(r.canonical) for r in recipes if r.canonical]
        self._rng = random.Random(seed)

    def __iter__(self) -> Iterator[list[str]]:
        for sent in self._sentences:
            tokens = list(sent)
            self._rng.shuffle(tokens)
            yield tokens
