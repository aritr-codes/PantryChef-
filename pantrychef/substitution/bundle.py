"""Persist substitution inference artifacts as a single runtime bundle."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from scipy import sparse

from pantrychef import __version__
from pantrychef.common.types import Recipe
from pantrychef.substitution.config import SubConfig
from pantrychef.substitution.embeddings import EmbeddingModel
from pantrychef.substitution.graph import ContextGraph

DEFAULT_BUNDLE_NAME = "substitution_bundle.zip"
BUNDLE_VERSION = 1


def corpus_fingerprint(recipes: Sequence[Recipe]) -> str:
    """Return a deterministic fingerprint for the training corpus."""
    h = hashlib.sha256()
    for recipe in recipes:
        h.update(recipe.recipe_id.encode("utf-8"))
        h.update(b"\x1f")
        for ingredient in recipe.canonical:
            h.update(ingredient.encode("utf-8"))
            h.update(b"\x1f")
        h.update(b"\n")
    return h.hexdigest()


def build_bundle_metadata(recipes: Sequence[Recipe], vocab: Sequence[str]) -> dict[str, object]:
    """Build deterministic training metadata for the persisted bundle."""
    return {
        "recipe_count": len(recipes),
        "vocab_size": len(vocab),
        "corpus_fingerprint": corpus_fingerprint(recipes),
    }


def save_bundle(
    path: str | Path,
    *,
    embeddings: EmbeddingModel,
    graph: ContextGraph,
    cfg: SubConfig,
    vocab: Sequence[str],
    metadata: dict[str, object] | None = None,
) -> None:
    """Persist all substitution inference state to a single zip bundle."""
    bundle_path = Path(path)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)

    emb_state = embeddings.to_state()
    graph_state = graph.to_state()
    payload = {
        "bundle_version": BUNDLE_VERSION,
        "pantrychef_version": __version__,
        "config": dataclasses.asdict(cfg),
        "graph": {
            "lam": graph_state["lam"],
            "overlap_shrink": graph_state["overlap_shrink"],
        },
        "metadata": dict(metadata or {}),
    }

    with ZipFile(bundle_path, mode="w", compression=ZIP_DEFLATED) as zf:
        _write_json(zf, "metadata.json", payload)
        _write_json(zf, "graph_vocab.json", list(vocab))
        _write_json(zf, "embedding_vocab.json", emb_state["vocab"])
        _write_npy(zf, "embedding_vectors.npy", emb_state["vectors"])
        _write_sparse(zf, "graph_norm.npz", graph_state["normalized_sppmi"])
        _write_sparse(zf, "graph_cooccur.npz", graph_state["cooccur"])


def load_bundle(path: str | Path) -> dict[str, object]:
    """Load a persisted substitution bundle into runtime-ready components."""
    bundle_path = Path(path)
    with ZipFile(bundle_path, mode="r") as zf:
        payload = _read_json(zf, "metadata.json")
        if payload.get("bundle_version") != BUNDLE_VERSION:
            raise ValueError(
                f"unsupported substitution bundle version: {payload.get('bundle_version')}"
            )
        graph_vocab = _read_json(zf, "graph_vocab.json")
        embedding_vocab = _read_json(zf, "embedding_vocab.json")
        vectors = _read_npy(zf, "embedding_vectors.npy")
        graph_norm = _read_sparse(zf, "graph_norm.npz")
        graph_cooccur = _read_sparse(zf, "graph_cooccur.npz")

    cfg = SubConfig(**payload["config"])
    return {
        "embeddings": EmbeddingModel.from_state(embedding_vocab, vectors),
        "graph": ContextGraph.from_state(
            graph_norm,
            graph_vocab,
            graph_cooccur,
            lam=float(payload["graph"]["lam"]),
            overlap_shrink=float(payload["graph"]["overlap_shrink"]),
        ),
        "cfg": cfg,
        "vocab": tuple(graph_vocab),
        "metadata": dict(payload.get("metadata", {})),
    }


def _write_json(zf: ZipFile, name: str, value: object) -> None:
    zf.writestr(name, json.dumps(value, indent=2, sort_keys=True))


def _read_json(zf: ZipFile, name: str):
    return json.loads(zf.read(name).decode("utf-8"))


def _write_npy(zf: ZipFile, name: str, array: np.ndarray) -> None:
    buf = BytesIO()
    np.save(buf, np.asarray(array), allow_pickle=False)
    zf.writestr(name, buf.getvalue())


def _read_npy(zf: ZipFile, name: str) -> np.ndarray:
    return np.load(BytesIO(zf.read(name)), allow_pickle=False)


def _write_sparse(zf: ZipFile, name: str, matrix: sparse.csr_matrix) -> None:
    buf = BytesIO()
    sparse.save_npz(buf, matrix.tocsr())
    zf.writestr(name, buf.getvalue())


def _read_sparse(zf: ZipFile, name: str) -> sparse.csr_matrix:
    return sparse.load_npz(BytesIO(zf.read(name))).tocsr()
