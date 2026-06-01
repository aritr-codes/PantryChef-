"""Hyperparameters for the substitution arms and fusion.

Frozen dataclass (not Hydra — Phase 1 uses pydantic-settings + argparse). The
training script overrides these via argparse; the ablation sweeps `alpha`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SubConfig:
    # word2vec (food2vec baseline)
    dims: int = 100
    window: int = 50  # >= max recipe length so the whole set is in-window
    epochs: int = 5
    negative: int = 10
    min_count: int = 1  # vocab already frequency-filtered upstream
    # SPPMI context graph (flagship)
    sppmi_shift: float = 1.0  # shift k; 1.0 == plain PPMI
    svd_dims: int = 100
    lam: float = 0.5  # soft direct-co-occurrence penalty weight
    # support-shrinkage beta; downweights low-overlap candidates; 0 = off/legacy
    overlap_shrink: float = 100.0
    # fusion
    alpha: float = 0.5  # rank blend: 1.0 = emb-only, 0.0 = graph-only
    context_weight: float = 0.0  # gated recipe-context re-rank; 0 = off
    k: int = 5
    seed: int = 42
