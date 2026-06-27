"""Persist recommender inference artifacts as a single runtime bundle."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from pantrychef import __version__
from pantrychef.recommender.config import RecConfig
from pantrychef.recommender.rank import LambdaMARTRanker, LinearRanker
from pantrychef.retrieval.index import InvertedIndex

DEFAULT_BUNDLE_NAME = "recommender_bundle.zip"
BUNDLE_VERSION = 1


@dataclasses.dataclass
class BundledRanker:
    """One persisted recommender model plus its feature schema."""

    name: str
    kind: str
    columns: tuple[str, ...]
    model: LinearRanker | LambdaMARTRanker

    def to_manifest(self) -> dict[str, object]:
        return {"kind": self.kind, "columns": list(self.columns)}

    def to_state(self) -> dict[str, Any]:
        if self.kind == "linear":
            return self.model.to_state()
        if self.kind == "lambdamart":
            return self.model.to_state()
        raise ValueError(f"unsupported ranker kind: {self.kind}")

    @classmethod
    def from_state(
        cls,
        name: str,
        kind: str,
        columns: Sequence[str],
        state: dict[str, Any],
    ) -> BundledRanker:
        if kind == "linear":
            model = LinearRanker.from_state(state)
        elif kind == "lambdamart":
            model = LambdaMARTRanker.from_state(state)
        else:
            raise ValueError(f"unsupported ranker kind: {kind}")
        return cls(name=name, kind=kind, columns=tuple(columns), model=model)


@dataclasses.dataclass
class RecommenderBundle:
    """Single-file persisted runtime state for recommender inference."""

    index: InvertedIndex
    cfg: RecConfig
    models: dict[str, BundledRanker]
    metadata: dict[str, object] = dataclasses.field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        bundle_path = Path(path)
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "bundle_version": BUNDLE_VERSION,
            "pantrychef_version": __version__,
            "config": dataclasses.asdict(self.cfg),
            "metadata": dict(self.metadata),
            "models": {name: model.to_manifest() for name, model in self.models.items()},
        }
        with ZipFile(bundle_path, mode="w", compression=ZIP_DEFLATED) as zf:
            _write_json(zf, "manifest.json", manifest)
            _write_json(zf, "index.json", self.index.to_state())
            for name, model in self.models.items():
                _write_json(zf, f"models/{name}.json", _jsonify(model.to_state()))

    @classmethod
    def load(cls, path: str | Path) -> RecommenderBundle:
        bundle_path = Path(path)
        with ZipFile(bundle_path, mode="r") as zf:
            manifest = _read_json(zf, "manifest.json")
            version = manifest.get("bundle_version")
            if version != BUNDLE_VERSION:
                raise ValueError(f"unsupported recommender bundle version: {version}")
            index = InvertedIndex.from_state(_read_json(zf, "index.json"))
            cfg = RecConfig(**manifest["config"])
            models: dict[str, BundledRanker] = {}
            for name, spec in manifest["models"].items():
                models[name] = BundledRanker.from_state(
                    name=name,
                    kind=str(spec["kind"]),
                    columns=tuple(spec["columns"]),
                    state=_read_json(zf, f"models/{name}.json"),
                )
        return cls(index=index, cfg=cfg, models=models, metadata=dict(manifest.get("metadata", {})))


def _jsonify(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value


def _write_json(zf: ZipFile, name: str, value: object) -> None:
    zf.writestr(name, json.dumps(value, indent=2, sort_keys=True))


def _read_json(zf: ZipFile, name: str):
    return json.loads(zf.read(name).decode("utf-8"))
