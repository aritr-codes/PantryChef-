from __future__ import annotations

import json

import pandas as pd

from pantrychef.data.loaders import load_recipenlg


def test_load_recipenlg_parses_json_columns(tmp_path) -> None:
    csv = tmp_path / "mini.csv"
    pd.DataFrame(
        [
            {
                "title": "Pancakes",
                "ingredients": json.dumps(["2 cups flour", "1 egg"]),
                "directions": json.dumps(["Mix.", "Cook."]),
                "link": "http://x",
                "source": "Gathered",
                "NER": json.dumps(["flour", "egg"]),
            }
        ]
    ).to_csv(csv, index=False)

    rows = list(load_recipenlg(csv))
    assert len(rows) == 1
    r = rows[0]
    assert r.title == "Pancakes"
    assert r.ingredients == ["2 cups flour", "1 egg"]
    assert r.directions == ["Mix.", "Cook."]
    assert r.ner == ["flour", "egg"]


def test_load_recipenlg_max_rows(tmp_path) -> None:
    csv = tmp_path / "mini.csv"
    pd.DataFrame(
        [{"title": f"R{i}", "ingredients": "[]", "directions": "[]", "NER": "[]"} for i in range(5)]
    ).to_csv(csv, index=False)
    assert len(list(load_recipenlg(csv, max_rows=2))) == 2


def test_load_recipenlg_defensive_parsing(tmp_path) -> None:
    # Documented contract: malformed JSON, empty cells, and non-list JSON -> [].
    csv = tmp_path / "bad.csv"
    pd.DataFrame(
        [
            {"title": "Bad JSON", "ingredients": "[not valid", "directions": "", "NER": '"egg"'},
            {"title": "Empty", "ingredients": "", "directions": "[]", "NER": ""},
        ]
    ).to_csv(csv, index=False)

    rows = list(load_recipenlg(csv))
    assert rows[0].ingredients == []  # malformed JSON
    assert rows[0].directions == []  # empty cell
    assert rows[0].ner == []  # valid JSON but not a list
    assert rows[1].ingredients == []
    assert rows[1].ner == []


def test_load_recipenlg_chunksize_spans_chunks(tmp_path) -> None:
    # Streaming across multiple chunks must yield every row.
    csv = tmp_path / "many.csv"
    pd.DataFrame(
        [
            {"title": f"R{i}", "ingredients": "[]", "directions": "[]", "NER": "[]"}
            for i in range(25)
        ]
    ).to_csv(csv, index=False)
    assert len(list(load_recipenlg(csv, chunksize=10))) == 25
