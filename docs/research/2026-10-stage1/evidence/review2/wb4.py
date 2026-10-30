import sys, json, re, collections
sys.stdout.reconfigure(encoding="utf8")
wb = json.load(open(sys.argv[1], encoding="utf8"))
by = {i["recipe_data"]["title"]: i["recipe_data"] for i in wb}
for t in ("Spanish Potato Omelette", "Saffron Rice"):
    d = by[t]
    print(t, d["url"], d.get("infobox"))
    print([x["text"][:40] for x in d["text_lines"] if (x.get("section") or "").lower().startswith("ingredient")])
L = [(d["title"], x["text"]) for d in by.values() for x in d["text_lines"] if (x.get("section") or "").lower().startswith("ingredient") and x["line_type"] in ("ul", "li")]
bare = [l for _, l in L if re.fullmatch(r"(?i)\W*(pepper|salt and pepper|salt & pepper|pepper to taste)\W*", l.strip())]
print("bare pepper lines", len(bare))
print("lines with salt and pepper together", sum(bool(re.search(r"(?i)salt.*pepper", l)) for _, l in L))
print("class/alt lines needing a set of ids (or /):", sum(bool(re.search(r"(?i)\bor\b|/", l)) for _, l in L))
r = collections.Counter(t for t, l in L if re.search(r"(?i)\bor\b", l))
print("recipes with at least one 'or' line", len(r))
print("recipes with 1-2 non-staple heads trivial (<=3 lines)", sum(1 for d in by.values() if 2 <= sum(1 for x in d["text_lines"] if (x.get("section") or "").lower().startswith("ingredient") and x["line_type"] in ("ul", "li")) <= 3))
