"""Backwards-compatible shim. The dietary engine now lives in
`pantrychef.dietary`; this re-export keeps existing Phase-2 imports working."""

from __future__ import annotations

from pantrychef.dietary import DietTagger
from pantrychef.dietary.ontology import _CURATED, _DIETS, _KEYWORDS

__all__ = ["DietTagger", "_KEYWORDS", "_CURATED", "_DIETS"]
