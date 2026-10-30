"""Fallback-corpus fitness: parse rate, field coverage, duplicates, pantry yield, link rot.

Usage: python measure_corpora.py <corpora_dir> <out.json>
Pantries are hand-written by the auditor; matching is a loose word-subset rule that
approximates a collapsed vocabulary. Results are indicative, not an evaluation.
"""

import gzip
import json
import random
import re
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pantrychef.ingredients.parser import build_match_index, parse

base, out = Path(sys.argv[1]), Path(sys.argv[2])
vocab = json.load(open("data/processed/vocab.json", encoding="utf8"))
index = build_match_index(vocab)
JUNK = {"large", "small", "whole", "garnish", "optional", "medium", "fresh"}
STAPLES = {"water", "salt", "pepper", "oil", "black pepper", "olive oil", "vegetable oil"}
PANTRIES = {
    "student": "egg rice soy sauce onion garlic butter pasta tomato cheese".split(" "),
    "weeknight": ["chicken", "onion", "garlic", "tomato", "rice", "bell pepper", "cumin", "lemon"],
    "baker": ["flour", "sugar", "butter", "egg", "milk", "baking powder", "vanilla", "chocolate"],
    "vegetarian": ["lentil", "spinach", "onion", "garlic", "carrot", "tomato", "cumin", "rice"],
    "indian": ["chicken", "onion", "garlic", "ginger", "tomato", "turmeric", "cumin", "yogurt"],
    "breakfast": ["egg", "milk", "bread", "butter", "bacon", "cheese", "banana"],
    "pasta": ["pasta", "tomato", "garlic", "basil", "parmesan", "onion", "olive oil"],
    "fish": ["salmon", "lemon", "potato", "dill", "butter", "garlic", "broccoli"],
    "tiny": ["egg", "cheese", "bread"],
    "beef": ["ground beef", "onion", "potato", "carrot", "tomato", "garlic", "beans"],
}


def canon_set(lines):
    got = set()
    n = hit = 0
    for line in lines:
        line = line.strip()
        if not line or len(line) > 200:
            continue
        n += 1
        try:
            c = parse(line, vocab, index).canonical
        except Exception:
            c = None
        if c:
            hit += 1
            if c not in JUNK:
                got.add(c)
    return got, n, hit


def covered(canon, pantry):
    words = set(canon.split())
    return any(set(p.split()) <= words for p in pantry)


def is_staple(canon):
    return canon in STAPLES or canon.split()[-1] in {"salt", "water"}


def pantry_yield(recipes, scale):
    rows = {}
    for name, pantry in PANTRIES.items():
        counts = [0, 0, 0]
        for canon in recipes:
            if len(canon) < 3:
                continue
            missing = sum(1 for c in canon if not is_staple(c) and not covered(c, pantry))
            for m in range(3):
                if missing <= m:
                    counts[m] += 1
        rows[name] = [round(c * scale) for c in counts]
    return rows


rng = random.Random(42)
res = {}

# Open Recipes
t = time.time()
all_or = []
for line in gzip.open(base / "openrecipes/r.json.gz", "rt", encoding="utf8", errors="replace"):
    try:
        r = json.loads(line)
    except ValueError:
        continue
    if r.get("ingredients"):
        all_or.append(
            {k: r.get(k) for k in ("name", "ingredients", "url", "source", "recipeYield", "cookTime", "prepTime")}
        )
exact = {(re.sub(r"\W+", " ", r["name"]).strip().lower(), r["ingredients"].strip().lower()) for r in all_or}
titles = {re.sub(r"\W+", " ", r["name"]).strip().lower() for r in all_or}
sample = rng.sample(all_or, 20000)
sets, tot, hit, lens = [], 0, 0, []
for r in sample:
    s, n, h = canon_set(r["ingredients"].split("\n"))
    sets.append(s)
    tot += n
    hit += h
    lens.append(len(s))
canon_keys = {frozenset(s) for s in sets if len(s) >= 3}
res["openrecipes"] = {
    "n_with_ingredients": len(all_or),
    "exact_unique_title_and_ingredients": len(exact),
    "unique_titles": len(titles),
    "parse_rate": round(hit / tot, 3),
    "median_canonical": statistics.median(lens),
    "share_ge3": round(sum(1 for x in lens if x >= 3) / len(lens), 3),
    "sample_unique_canon_sets_ge3": len(canon_keys),
    "field_rate": {
        k: round(sum(1 for r in all_or if r.get(k)) / len(all_or), 3)
        for k in ("recipeYield", "cookTime", "prepTime", "url")
    },
    "pantry_yield_scaled_le0_le1_le2": pantry_yield(sets, len(all_or) / len(sample)),
    "seconds": round(time.time() - t),
}

# Wikibooks
wb = json.load(open(base / "wikibooks/recipes.json", encoding="utf8"))
sets, tot, hit, lens, n_rec, bad, has = [], 0, 0, [], 0, 0, {"servings": 0, "time": 0, "steps": 0}
for item in wb:
    d = item["recipe_data"]
    lines = [x["text"] for x in d["text_lines"] if (x.get("section") or "").lower().startswith("ingredient") and x["line_type"] in ("ul", "li")]
    steps = [x for x in d["text_lines"] if "procedure" in (x.get("section") or "").lower() or "direction" in (x.get("section") or "").lower() or "instruction" in (x.get("section") or "").lower()]
    if len(lines) < 2:
        bad += 1
        continue
    n_rec += 1
    s, n, h = canon_set(lines)
    sets.append(s)
    tot += n
    hit += h
    lens.append(len(s))
    info = d.get("infobox") or {}
    has["servings"] += bool(info.get("servings"))
    has["time"] += bool(info.get("time"))
    has["steps"] += bool(steps)
res["wikibooks"] = {
    "n_pages": len(wb),
    "n_with_ingredient_list": n_rec,
    "parse_rate": round(hit / max(tot, 1), 3),
    "median_canonical": statistics.median(lens),
    "field_rate": {k: round(v / n_rec, 3) for k, v in has.items()},
    "pantry_yield_le0_le1_le2": pantry_yield(sets, 1.0),
}

# RecipeNLG reference: every 8th kept recipe, canonical already computed
sets = []
for i, line in enumerate(open("data/processed/recipes.jsonl", encoding="utf8")):
    if i % 8 == 0:
        sets.append({c for c in json.loads(line)["canonical"] if c not in JUNK})
res["recipenlg_ref"] = {
    "n_full": 1274290,
    "pantry_yield_scaled_le0_le1_le2": pantry_yield(sets, 8.0),
}
del sets

# Link rot: small polite sample
def alive(url):
    u = url if url.startswith("http") else "http://" + url
    try:
        req = urllib.request.Request(u, method="GET", headers={"User-Agent": "Mozilla/5.0 pantrychef-audit"})
        with urllib.request.urlopen(req, timeout=12) as h:
            return h.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return 0


links = rng.sample([r for r in all_or if r.get("url")], 150)
with ThreadPoolExecutor(6) as ex:
    codes = list(ex.map(lambda r: (r["source"], alive(r["url"])), links))
ok = sum(1 for _, c in codes if 200 <= c < 400)
per = {}
for s, c in codes:
    a = per.setdefault(s, [0, 0])
    a[1] += 1
    a[0] += 200 <= c < 400
res["openrecipes_link_sample"] = {
    "n": len(codes),
    "ok": ok,
    "blocked_403": sum(1 for _, c in codes if c == 403),
    "not_found_404": sum(1 for _, c in codes if c == 404),
    "dead_other": sum(1 for _, c in codes if c == 0),
    "by_source": dict(sorted(per.items(), key=lambda kv: -kv[1][1])),
}

out.write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
