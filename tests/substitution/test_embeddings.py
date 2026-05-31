from pantrychef.common.types import Recipe
from pantrychef.substitution.config import SubConfig
from pantrychef.substitution.corpus import IngredientCorpus
from pantrychef.substitution.embeddings import EmbeddingModel, train_word2vec

RECIPES = [
    Recipe(recipe_id=f"r{i}", title="t", canonical=["flour", "egg", "milk", "butter"])
    for i in range(50)
] + [
    Recipe(recipe_id=f"s{i}", title="t", canonical=["flour", "egg", "milk", "oil"])
    for i in range(50)
]


def _model() -> EmbeddingModel:
    cfg = SubConfig(dims=16, window=10, epochs=3, seed=42)
    wv = train_word2vec(IngredientCorpus(RECIPES, seed=cfg.seed), cfg)
    return EmbeddingModel(wv)


def test_trains_and_has_vocab() -> None:
    m = _model()
    assert "flour" in m.vocab
    assert "butter" in m.vocab


def test_neighbors_excludes_self_and_bounds() -> None:
    m = _model()
    nbrs = m.neighbors("butter", k=3)
    assert len(nbrs) == 3
    assert all(ing != "butter" for ing, _ in nbrs)
    assert all(-1.0 <= score <= 1.0 for _, score in nbrs)


def test_unknown_returns_empty() -> None:
    assert _model().neighbors("dragonfruit", k=3) == []


def test_save_load_roundtrip(tmp_path) -> None:
    m = _model()
    p = tmp_path / "wv.kv"
    m.save(p)
    loaded = EmbeddingModel.load(p)
    assert loaded.neighbors("butter", k=2) == m.neighbors("butter", k=2)
