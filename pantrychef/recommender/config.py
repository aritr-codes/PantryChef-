"""Phase 3 reranker knobs. Defaults locked in the design spec (2026-06-02)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RecConfig:
    mask_fraction: float = 0.3  # fraction of a recipe's ingredients hidden to form the pantry
    min_recipe_len: int = 4  # skip recipes shorter than this (need >=1 kept after masking)
    candidate_cap: int = 200  # max candidates per query, top by coverage (shared pool)
    seed: int = 13  # deterministic masking
    sub_pool: int = 20  # P2 substitutes fetched per missing ingredient for sub_fill
    queries_per_recipe: int = 1  # masked variants generated per train recipe
