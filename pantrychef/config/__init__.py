"""Typed runtime configuration (pydantic-settings).

Runtime/app config lives here. Model hyperparameters are dataclasses next to
each model (for example `pantrychef/substitution/config.py`).
"""

from pantrychef.config.settings import Settings, get_settings

__all__ = ["Settings", "get_settings"]
