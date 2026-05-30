"""Typed runtime configuration (pydantic-settings).

Runtime/app config lives here. Experiment configs (sweeps, model
hyperparameters) live in `configs/` and are loaded via Hydra.
"""

from pantrychef.config.settings import Settings, get_settings

__all__ = ["Settings", "get_settings"]
