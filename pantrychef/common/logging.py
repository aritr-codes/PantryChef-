"""Minimal structured logging setup.

One helper so modules don't each reconfigure logging. Level driven by Settings.
"""

from __future__ import annotations

import logging

_CONFIGURED = False


def _configure() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    # Imported lazily to avoid a config import at module load.
    from pantrychef.config import get_settings

    logging.basicConfig(
        level=get_settings().log_level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for ``name`` (usually ``__name__``)."""
    _configure()
    return logging.getLogger(name)
