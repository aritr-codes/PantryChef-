from pantrychef.ingredients.vocab import (
    build_vocabulary,
    load_vocabulary,
    save_vocabulary,
)


def test_build_vocabulary_min_count(sample_raws) -> None:
    vocab = build_vocabulary(sample_raws, min_count=2)
    # flour, egg, milk, sugar each appear >= 2 times across NER columns
    assert "flour" in vocab and "egg" in vocab and "milk" in vocab and "sugar" in vocab
    # salt/butter/tomato/cucumber appear once -> excluded
    assert "salt" not in vocab and "tomato" not in vocab


def test_build_vocabulary_keeps_singletons_when_min_one(sample_raws) -> None:
    vocab = build_vocabulary(sample_raws, min_count=1)
    assert {"flour", "egg", "milk", "sugar", "salt", "butter", "tomato", "cucumber"} <= set(vocab)


def test_build_vocabulary_sorted_longest_first(sample_raws) -> None:
    vocab = build_vocabulary(sample_raws, min_count=1)
    word_counts = [len(v.split()) for v in vocab]
    assert word_counts == sorted(word_counts, reverse=True)


def test_save_and_load_roundtrip(tmp_path, sample_raws) -> None:
    vocab = build_vocabulary(sample_raws, min_count=1)
    p = tmp_path / "vocab.json"
    save_vocabulary(vocab, p)
    assert load_vocabulary(p) == vocab
