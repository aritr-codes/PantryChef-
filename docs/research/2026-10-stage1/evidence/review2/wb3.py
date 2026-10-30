import sys, json, re, random, collections
sys.stdout.reconfigure(encoding="utf8")
REPO = r"G:\Books\Python\ACTUAL CODES\PROJECTS\pantrychef"
sys.dont_write_bytecode = True
sys.path.insert(0, REPO)
from pantrychef.ingredients.parser import build_match_index, parse

vocab = json.load(open(REPO + r"\data\processed\vocab.json", encoding="utf8"))
index = build_match_index(vocab)
print("vocab", len(vocab))
wb = json.load(open(sys.argv[1], encoding="utf8"))
recs = []
for it in wb:
    d = it["recipe_data"]
    L = [x["text"] for x in d["text_lines"] if (x.get("section") or "").lower().startswith("ingredient") and x["line_type"] in ("ul", "li")]
    if len(L) >= 2:
        recs.append((d, L))

print("== 50 random pages")
rng = random.Random(2026)
for d, L in rng.sample(recs, 50):
    print("-", d["title"], "|", ((d.get("infobox") or {}).get("category") or "").rsplit(":", 1)[-1][:18], "|", len(L), "|", [l[:30] for l in L[:3]])

comp = [d["title"] for d, L in recs if re.search(r"(?i)spice|stock|broth|sauce|dressing|marinade|season|syrup|beverage|condiment|dip", (d.get("infobox") or {}).get("category") or "")]
print("component/beverage categories", len(comp), "of", len(recs))
print("recipes with no amounts on any line", sum(all(not re.match(r"\s*[\d½¼¾⅓⅔⅛]", l) for l in L) for d, L in recs))
t = collections.Counter()
for it in wb:
    for x in it["recipe_data"]["text_lines"]:
        if re.search(r"Template:|\[edit\]|\{\{", x["text"]):
            t[x["text"][:50]] += 1
print("template lines", sum(t.values()), t.most_common(5))

rng = random.Random(99)
all_lines = [l for d, L in recs for l in L]
print("== parse sample")
for l in rng.sample(all_lines, 60):
    print(repr(l[:60]), "=>", parse(l, vocab, index).canonical)

parsed = []
cc = collections.Counter()
for d, L in recs:
    row = []
    for l in L:
        l2 = l.strip()
        if not l2 or len(l2) > 200:
            continue
        c = parse(l2, vocab, index).canonical
        row.append((l2, c))
        if c:
            cc[c] += 1
    parsed.append((d["title"], row))
tot = sum(len(r) for _, r in parsed)
hit = sum(1 for _, r in parsed for _, c in r if c)
print("parse rate", round(hit / tot, 3), "distinct canonical", len(cc))
print("top30", cc.most_common(30))

ranked = [k for k, _ in cc.most_common()]
for N in (300, 500, 1000, 2000):
    T = set(ranked[:N])
    m = sum(n for k, n in cc.items() if k in T) / tot
    full = sum(all(c in T for _, c in r) for _, r in parsed)
    print("topN", N, "mentions", round(m, 3), "recipes fully covered", full, round(full / len(parsed), 3))
cum = 0
for i, (k, n) in enumerate(cc.most_common(), 1):
    cum += n
    if cum / tot >= 0.90:
        print("rows for 90pct mentions", i)
        break

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
JUNK = {"large", "small", "whole", "garnish", "optional", "medium", "fresh"}
STAPLES = {"water", "salt", "pepper", "oil", "black pepper", "olive oil", "vegetable oil"}
V_ST = {"water", "salt", "black pepper", "pepper", "oil", "vegetable oil", "canola oil", "sunflower oil", "cooking oil"}
MOD = {"fresh", "large", "small", "medium", "whole", "chopped", "dried", "ground", "unsalted", "salted", "plain", "white", "ripe", "red", "yellow", "all", "purpose", "granulated", "cooked", "boiled", "long", "grain", "clove", "sliced", "diced", "minced", "beaten", "peeled", "can", "canned", "cup"}


def loose(c, p):
    w = set(c.split())
    return any(set(x.split()) <= w for x in p)


def strict(c, p):
    w = set(c.split())
    return any(set(x.split()) <= w and (w - set(x.split())) <= MOD for x in p)


def st_loose(c):
    return c in STAPLES or c.split()[-1] in {"salt", "water"}


def st_v(c):
    return c in V_ST or c.split()[-1] in {"salt", "water"}


def run(rule, st, unread):
    out, det = {}, {}
    for name, p in PANTRIES.items():
        c0 = c1 = 0
        lst, one = [], []
        for t, r in parsed:
            cs = {c for _, c in r if c and c not in JUNK}
            if len(cs) < 3:
                continue
            miss = [c for c in cs if not st(c) and not rule(c, p)]
            m = len(miss) + (sum(1 for _, c in r if not c) if unread else 0)
            c0 += m == 0
            c1 += m <= 1
            if m == 0:
                lst.append(t)
            if m == 1:
                one.append((t, miss[0] if miss else "UNREAD"))
        out[name] = (c0, c1 - c0)
        det[name] = (lst, one)
    return out, det


y, dl = run(loose, st_loose, False)
print("LOOSE repro (make now, exactly one)", y)
y2, d2 = run(strict, st_v, True)
print("STRICT, vision staples, unread counts missing (make now, exactly one)", y2)
rows = {t: r for t, r in parsed}
rng = random.Random(5)
print("== audit make-now loose")
for name in ("student", "breakfast", "baker", "tiny"):
    lst = dl[name][0]
    for t in rng.sample(lst, min(5, len(lst))):
        print(name, "|", t, "|", [l[:30] for l, c in rows[t]][:9])
print("== audit one-away loose (fish, vegetarian)")
for name in ("fish", "vegetarian"):
    one = dl[name][1]
    for t, m in rng.sample(one, min(6, len(one))):
        print(name, "|", t, "| missing:", m, "|", [l[:26] for l, c in rows[t]][:10])
print("strict make-now", {k: d2[k][0][:8] for k in ("student", "breakfast", "tiny")})
fp = collections.Counter()
for t, r in parsed:
    for l, c in r:
        if c:
            for p in ["egg", "milk", "butter", "cheese", "chicken", "tomato", "rice", "flour", "sugar", "bread", "onion", "beans", "potato", "lemon", "garlic"]:
                if p in c.split() and c != p:
                    fp[(p, c)] += 1
print("modified forms covered by word rule", fp.most_common(40))

# Best item to add check: for a pantry, greedy item by exactly-one-missing
for name in ("student", "tiny"):
    print("best add", name, collections.Counter(m for _, m in dl[name][1]).most_common(8))
