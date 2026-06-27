"""Build all substitution artifacts from a cleaned recipe corpus.

Pure, testable orchestration: corpus -> word2vec; co-occurrence -> SPPMI ->
context graph. `Artifacts.substitutor()` wires the arms + dietary tagger into a
ready Substitutor. The script layer adds argparse + MLflow.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from pantrychef.common.types import Recipe
from pantrychef.substitution.bundle import build_bundle_metadata, load_bundle, save_bundle
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
    vocab: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def substitutor(
        self,
        known_vocab: Sequence[str] | None = None,
        cfg: SubConfig | None = None,
    ) -> Substitutor:
        """Wire trained artifacts + dietary tagger into a ready Substitutor."""
        vocab = (
            list(known_vocab)
            if known_vocab is not None
            else list(self.vocab or self.graph.vocab)
        )
        return Substitutor(
            emb=self.embeddings,
            graph=self.graph,
            tagger=DietTagger(known=vocab),
            cfg=self.cfg if cfg is None else cfg,
        )

    def save_bundle(self, path: str | Path) -> None:
        """Persist this artifact set as a single substitution bundle."""
        save_bundle(
            path,
            embeddings=self.embeddings,
            graph=self.graph,
            cfg=self.cfg,
            vocab=self.vocab or tuple(self.graph.vocab),
            metadata=self.metadata,
        )

    @classmethod
    def load_bundle(cls, path: str | Path) -> Artifacts:
        """Load a previously persisted substitution bundle."""
        state = load_bundle(path)
        return cls(
            embeddings=state["embeddings"],
            graph=state["graph"],
            cfg=state["cfg"],
            vocab=state["vocab"],
            metadata=state["metadata"],
        )


def train_artifacts(recipes: Sequence[Recipe], vocab: Sequence[str], cfg: SubConfig) -> Artifacts:
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
    return Artifacts(
        embeddings=EmbeddingModel(wv),
        graph=graph,
        cfg=cfg,
        vocab=tuple(vocab),
        metadata=build_bundle_metadata(recipes, vocab),
    )
