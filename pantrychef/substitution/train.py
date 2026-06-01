"""Build all substitution artifacts from a cleaned recipe corpus.

Pure, testable orchestration: corpus -> word2vec; co-occurrence -> SPPMI ->
context graph. `Artifacts.substitutor()` wires the arms + dietary tagger into a
ready Substitutor. The script layer adds argparse + MLflow.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pantrychef.common.types import Recipe
from pantrychef.substitution.config import SubConfig
from pantrychef.substitution.cooccur import build_cooccurrence, sppmi
from pantrychef.substitution.corpus import IngredientCorpus
from pantrychef.substitution.dietary import DietTagger
from pantrychef.substitution.embeddings import EmbeddingModel, train_word2vec
from pantrychef.substitution.graph import ContextGraph
from pantrychef.substitution.substitute import Substitutor


@dataclass
class Artifacts:
    """Container for all trained substitution artifacts."""

    embeddings: EmbeddingModel
    graph: ContextGraph
    cfg: SubConfig

    def substitutor(self, known_vocab: Sequence[str]) -> Substitutor:
        """Wire trained artifacts + dietary tagger into a ready Substitutor."""
        return Substitutor(
            emb=self.embeddings,
            graph=self.graph,
            tagger=DietTagger(known=known_vocab),
            cfg=self.cfg,
        )


def train_artifacts(
    recipes: Sequence[Recipe], vocab: Sequence[str], cfg: SubConfig
) -> Artifacts:
    """Train both substitution arms and return a ready-to-use Artifacts container.

    Parameters
    ----------
    recipes:
        Cleaned recipe corpus with canonical ingredient lists.
    vocab:
        Ordered vocabulary for the co-occurrence matrix index.
    cfg:
        Hyperparameters controlling both word2vec and the SPPMI graph.

    Returns
    -------
    Artifacts containing the trained EmbeddingModel and ContextGraph.
    """
    wv = train_word2vec(IngredientCorpus(recipes, seed=cfg.seed), cfg)
    cmat, _, _, _ = build_cooccurrence(recipes, vocab)
    m = sppmi(cmat, shift=cfg.sppmi_shift)
    graph = ContextGraph(m, list(vocab), cmat, lam=cfg.lam, overlap_shrink=cfg.overlap_shrink)
    return Artifacts(embeddings=EmbeddingModel(wv), graph=graph, cfg=cfg)
