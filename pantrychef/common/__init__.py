"""Shared utilities: logging, IO helpers, common types.

Kept dependency-light so every other module can import it freely.
"""

from pantrychef.common.logging import get_logger

__all__ = ["get_logger"]
