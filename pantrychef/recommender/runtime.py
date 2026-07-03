"""Runtime loading and inference path for the persisted recommender bundle."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from pantrychef.common.types import ScoredRecipe
from pantrychef.ingredients.normalize import canonicalize
from pantrychef.recommender.bundle import DEFAULT_BUNDLE_NAME, RecommenderBundle
from pantrychef.recommender.features import SubLookup, extract_features
from pantrychef.recommender.recommend import rerank
from pantrychef.recommender.sub_lookup import load_sub_lookup


class RecommenderRuntime:
    """Load-once runtime wrapper over a persisted recommender bundle."""

    def __init__(self, bundle: RecommenderBundle, sub_lookup: SubLookup = None) -> None:
        self.bundle = bundle
        self.index = bundle.index
        self.cfg = bundle.cfg
        self.models = bundle.models
        self.sub_lookup = sub_lookup

    @classmethod
    def load(
        cls,
        path: str | Path,
        sub_lookup: SubLookup = None,
    ) -> RecommenderRuntime:
        bundle = RecommenderBundle.load(path)
        lookup = sub_lookup if sub_lookup is not None else load_sub_lookup(bundle.cfg)
        return cls(bundle=bundle, sub_lookup=lookup)

    @classmethod
    def load_default(cls, sub_lookup: SubLookup = None) -> RecommenderRuntime:
        from pantrychef.config import get_settings

        s = get_settings()
        return cls.load(s.models_dir / "recommender" / DEFAULT_BUNDLE_NAME, sub_lookup=sub_lookup)

    def recommend(
        self,
        pantry: Iterable[str],
        k: int = 10,
        model_name: str | None = None,
    ) -> list[ScoredRecipe]:
        """Score pantry candidates with a persisted recommender model."""
        if k <= 0:
            return []
        name = model_name or ("lambdamart" if "lambdamart" in self.models else "linear")
        bundled = self.models.get(name)
        if bundled is None:
            raise KeyError(f"unknown recommender model: {name}")
        pantry_set = {c for c in (canonicalize(p) for p in pantry) if c}

        def feat_fn(current_pantry: set[str], recipe):
            return extract_features(
                current_pantry,
                recipe,
                self.sub_lookup,
                canon=self.index.canon_sets[recipe.recipe_id],
            )

        return rerank(
            self.index,
            pantry_set,
            bundled.model,
            feat_fn,
            k=k,
            cap=self.cfg.candidate_cap,
            columns=bundled.columns,
        )
