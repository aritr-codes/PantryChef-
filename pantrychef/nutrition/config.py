"""Frozen config for the nutrition eval + imputation gate."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NutritionConfig:
    seed: int = 42
    sample_size: int = 5000  # recipes drawn for the coverage eval
    jaccard_threshold: float = 0.34
    usable_mass_fraction: float = 0.50  # recipe "usable" if >= this mass resolved
    # imputation gate (build impute.py iff EITHER trips):
    impute_match_cov_gate: float = 0.80  # build if match coverage < this
    impute_unresolved_mass_gate: float = 0.20  # or median unresolved-mass > this
