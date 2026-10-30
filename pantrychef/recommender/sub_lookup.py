"""Load substitution-derived lookup features from the persisted substitution bundle."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from zipfile import BadZipFile

from pantrychef.config import get_settings
from pantrychef.recommender.config import RecConfig
from pantrychef.recommender.features import SubLookup
from pantrychef.substitution.bundle import DEFAULT_BUNDLE_NAME as DEFAULT_SUB_BUNDLE_NAME
from pantrychef.substitution.train import Artifacts as SubstitutionArtifacts


def load_sub_lookup(
    cfg: RecConfig,
    bundle_path: str | Path | None = None,
) -> SubLookup:
    """Load the persisted substitution bundle and expose a memoized lookup."""
    s = get_settings()
    path = Path(bundle_path or (s.models_dir / "substitution" / DEFAULT_SUB_BUNDLE_NAME))
    if not path.exists():
        return None
    try:
        art = SubstitutionArtifacts.load_bundle(path)
    except (BadZipFile, OSError, ValueError):
        return None
    sub = art.substitutor(cfg=replace(art.cfg, k=cfg.sub_pool))
    cache: dict[str, dict[str, float]] = {}

    def lookup(missing: str) -> dict[str, float]:
        hit = cache.get(missing)
        if hit is None:
            hit = {s_.ingredient: s_.score for s_ in sub.substitutes(missing, k=cfg.sub_pool)}
            cache[missing] = hit
        return hit

    return lookup
