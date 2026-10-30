"""Wikibooks data quality: damaged characters, category mix, title sample, instruction length."""

import collections
import json
import random
import sys

wb = json.load(open(sys.argv[1], encoding="utf8"))
rng = random.Random(7)
ing = bad_line = recipes = steps_len = 0
cats = collections.Counter()
sample = []
lens = []
for item in wb:
    d = item["recipe_data"]
    lines = [
        x["text"]
        for x in d["text_lines"]
        if (x.get("section") or "").lower().startswith("ingredient") and x["line_type"] in ("ul", "li")
    ]
    if len(lines) < 2:
        continue
    recipes += 1
    ing += len(lines)
    bad_line += sum("�" in x for x in lines)
    cats[((d.get("infobox") or {}).get("category") or "none").rsplit(":", 1)[-1]] += 1
    body = [
        x["text"]
        for x in d["text_lines"]
        if any(k in (x.get("section") or "").lower() for k in ("procedure", "direction", "instruction", "method"))
    ]
    lens.append(sum(len(x) for x in body))
    sample.append((d["title"], lines[:3]))
lens.sort()
print("recipes", recipes, "ingredient lines", ing, "lines with replacement char", bad_line, round(bad_line / ing, 3))
print("median instruction chars", lens[len(lens) // 2], "share under 80", round(sum(x < 80 for x in lens) / recipes, 3))
print(cats.most_common(25))
print("keys sample", list(wb[0]["recipe_data"]))
for t, ls in rng.sample(sample, 14):
    print(repr(t), [x[:45] for x in ls])
