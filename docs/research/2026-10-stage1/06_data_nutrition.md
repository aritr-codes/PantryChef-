# 06 Data engineering, parsing, nutrition, dietary (audit)

Scripts and raw outputs: `...\scratchpad\research\{a_corpus,b_parser,c_diet,d_raw,e_nut,f_nut2}.py`, `occ.json`, `sample20k.json`. All numbers below were produced by streaming/sampling (no whole-file loads). Sample = every 64th line of `recipes.jsonl` (19,910 recipes).

## Headline
The pantry-to-recipe path works for plain English staples but is brittle for real user input. The nutrition and dietary layers are NOT product-ready: a hand-checked end-to-end run gives implausible totals, and the dietary tagger judges only 18.7% of the vocabulary (so the conservative default blocks almost everything). The licence situation blocks a public launch on its own.

## 1. Provenance and licence

**F1.1 RecipeNLG is non-commercial research only; the corpus also contains Recipe1M-derived rows. Critical. VERIFIED (web + file).**
- HF card for `recipe_nlg`: licence field "unknown", but the embedded agreement limits use to "non-commercial research and educational purposes", binds the researcher's employer, and has an indemnity clause (https://huggingface.co/datasets/recipe_nlg). Official page (https://recipenlg.cs.put.poznan.pl/) returned only a title to my fetch; read the agreement there before any decision.
- `data/raw/full_dataset.csv` has `source` = `Gathered` for the first ~55% of the file and `Recipes1M` for the rest (seeked at 10/30/50% = Gathered; 70/90/99% = Recipes1M). Recipe1M+ has its own academic-only terms.
- `Recipe` (`pantrychef/common/types.py`) and `recipes.jsonl` drop `link` and `source`, so provenance is unrecoverable from the processed store. `docs/DATASET.md:21` says only "check dataset card before redistribution (research use)", and `README.md:10` badges the repo MIT. Code is MIT; the data cannot be.
- Fix: do not ship RecipeNLG text (titles, raw lines, directions) in a public app. Ship only (a) your own derived statistics (ingredient co-occurrence, vocab) after legal review, and (b) recipes from a licence-clean source (own authored, CC0/CC-BY open recipe sets, or user-submitted). Worth it: mandatory. Get written permission from PUT Poznan if keeping any derived artefact.

**F1.2 Download is not reproducible. High. VERIFIED.**
`scripts/download_data.py:3` says "manual download"; no checksum, no row count, no release date in `docs/DATASET.md` (it even says "Record N, vocab size and recipe count of each run here" and the table is empty). `data/raw/dataset.zip` (592 MB) exists with no recorded sha256. Fix: record sha256 of the zip and CSV plus row count (2,231,142 expected; confirm) in DATASET.md and have the script verify. Cost: 10 minutes.

**F1.3 USDA FDC. Low. VERIFIED (docs) / external.**
SR Legacy 2018-04 and Foundation 2025-04-24 are recorded (`DATASET.md:35-41`). FDC is public domain/CC0 and asks for citation (https://fdc.nal.usda.gov/). Fine to redistribute `usda.json`. Note SR Legacy is frozen since 2018 and Foundation has only 411 foods; branded items and FNDDS recipes are not included.

## 2. Cleaning and duplicates

**F2.1 "Dedup" is exact lowercase-title only and discards about 43% of the corpus as false duplicates. High. VERIFIED.**
`pantrychef/data/clean.py:38-41` keys on `raw.title.strip().lower()`. 2.23M to 1.27M (1,274,290 kept) therefore means ~960k rows removed. In the first 300k raw rows 174,987 are title repeats, and only 1 of them has an identical ingredient list (`d_raw.py`). Examples: "chicken casserole" x1363, "banana bread" x588 are different recipes. The docs claim RecipeNLG is "largely pre-deduplicated; cheap safeguard" (`clean.py:3`), which contradicts this. Consequence: first-seen wins, so the kept recipe is arbitrary, the long tail of real variants is lost, and popular dish names have exactly one representative. Fix: dedup on (normalised title, canonical set) or MinHash on ingredient set, keep variants. Worth it, but needs a full reclean (~30 min on Kaggle).

**F2.2 Residual near-duplicates after dedup. Medium. VERIFIED (exact) / HYPOTHESIS (fuzzy).**
Full-corpus stream: 1,242,494 distinct canonical sets, so 31,796 recipes (2.5%) are exact-set repeats (48,696 recipes sit in repeated sets). Alphanumeric-normalised title gives 39,512 excess (3.1%) (punctuation/case variants the key misses). Only 632 are same title AND same set. So after the title key, true duplicates are modest; the damage is the other direction (F2.1). Jaccard>=0.8 near-dups need MinHash-LSH: Kaggle experiment `minhash_dups` (1.27M sets, datasketch, ~20 min CPU). Matters for eval leakage (same recipe in query and target) and for the product (top-5 lists showing 3 copies of the same recipe).

**F2.3 Splits and leakage. High. VERIFIED.**
No split file exists. `DATASET.md:27` says splits are "assigned at training time"; the Phase-1 retrieval eval is leave-one-out over the full set. `vocab.json` is built by `build_vocabulary` from RecipeNLG `NER` over ALL rows (`download_data.py:42`, pre-dedup, `min_count=5`), then used to canonicalise the same recipes later evaluated. So vocab, co-occurrence statistics and eval items share data. Mostly benign for vocab (min_count 5), but real for the parser score (F3.5).

## 3. Parser and canonicalisation

**F3.1 `pantrychef cook` does NOT use the parser or vocab at all. Critical for pantry entry. VERIFIED.**
`cli.py` splits on commas and `retrieval/baseline.py:15` does `canonicalize(p)` and an exact set intersection. Thus "2 cups flour", "a pinch of salt", "evoo", "scallions" (when recipes say "green onion"), "tomatos", "garlic cloves" all just miss. `canonicalize` also (`normalize.py:14`) strips non-ASCII after NFKC without accent folding: "jalapeño" becomes "jalape o" (vocab holds "jalape o chile"). Fix: a pantry-entry resolver = autocomplete over a curated list + alias table + fuzzy fallback (rapidfuzz, edit distance <=2), run before `recommend`. High value, small code.

**F3.2 Hit rate on 166 realistic inputs (`b_parser.py`). VERIFIED.**
My acceptance sets are lenient (accept either a synonym or the head noun).
| Path | Hit | Rate |
|---|---|---|
| cook path (`canonicalize` and must be in vocab) | 136/166 | 0.82 |
| `parse(line, vocab)` | 143/166 | 0.86 |
Hit rate is inflated because most inputs are single common words. Failures: "evoo", "oj", "pb", "s&p" (becomes "s p"), "half and half" (no vocab entry, returns None), "cilantro/coriander" (None), "garbanzo beans/chickpeas" ("garbanzo"), "ground beef/mince" ("ground"), "minced beef" ("beef"), "plain flour" ("flour plain", order-permuted artefact), "garlic cloves" ("clove garlic"), "red pepper flakes" ("pepper flake"), "a pinch of salt" ("pinch of salt", a junk vocab entry), "1 (14 oz) can diced tomatoes" ("tomato diced"). Misspellings: "tomatoe", "brocoli", "zuchini", "tumeric", "parmesean" are IN vocab as separate entries (so they hit but match zero recipes that spell it correctly); "tomatos", "cheeze", "pb" are not.
`/` handling: `_KEEP` keeps "/" so "cilantro/coriander" is one token and fails.

**F3.3 Synonyms are not clustered. High. VERIFIED.**
All of scallion / green onion / spring onion, eggplant / aubergine, zucchini / courgette, cilantro / coriander / fresh coriander, shrimp / prawn, cornstarch / cornflour, powdered / icing / confectioner sugar, garbanzo bean / chickpea, rocket / arugula, beet / beetroot exist as SEPARATE vocab entries. A user holding "scallions" misses recipes that say "green onion" (37,722 recipes). Only 34 aliases exist and they serve nutrition only (`aliases.json`). Worse: "garlic" (132,753) and "clove garlic" (116,079) are separate canonicals; pantry "garlic" misses 47% of garlic recipes. Likewise "pepper" vs "black pepper" vs "freshly ground black pepper"; "vanilla" vs "vanilla extract". Fix: collapse by rules (strip prep/size/state words, drop container words) plus a 300-entry synonym map, then re-index. Highest-leverage data fix for retrieval quality.

**F3.4 Vocab quality (`c_*`, manual 200-sample plus programmatic). VERIFIED.**
- Manual read of 200 random entries: roughly 55 (27%) are not clean ingredient names (quantities/containers: "can applesauce", "liter beef stock", "tb garlic", "gallon", "tbsp", "handful of coriander"; prep/state: "hot cooked rice", "crisply cooked bacon", "peeled ginger"; fragments: "toasted white", "fill margarita", "zero", "electric", "oval", "alway", "s angel"; slash/or artefacts "pecans/walnut", "ginger ale or") plus about 20 (10%) brands ("butterfinger", "hormel chili", "jell o ...", "progresso artichoke heart"). So ~35% junk or brand; the user-facing clean fraction is perhaps 60%.
- Programmatic over 30,481 entries: 995 contain unit/container tokens; 2,677 have a prep word at an edge; 79 contain "/", 38 contain " or "; 55 are <=2 chars; 1,252 order-permutation groups cover 2,528 entries (8.3%) (e.g. "cream of chicken soup or cream of mushroom soup" and its reverse); 267 groups of stem-near-duplicates ("chily"/"chile", "crunch style peanut butter"/"crunchy style ...").
- Canonicals that are junk but extremely frequent: "large" occurs 67,381 times (more than "lemon"), "whole" 9,207, "small" 5,228, "garnish" 3,726, "optional" 2,753.
- Singulariser bug (`normalize.py:27`): asparagus to "asparagu" (in the top-400 canonicals), couscous to "couscou", hummus to "hummu", citrus to "citru", octopus to "octopu", chilies to "chily". "bay leave", "basil leave", "knive", "halve" are consistent but ugly in a UI.

**F3.5 Parser F1 0.864 is circular. High. VERIFIED.**
`eval/parser_eval.py`: gold = `canonicalize(ner)` of the same RecipeNLG row; vocab = the same NER strings with count>=5. The parser can only emit vocab entries, which are NER entities, so the metric measures consistency with NER, not correctness. It also ignores wrong-granularity matches ("clove garlic" vs "garlic" both count as plausible) and the NER column is itself noisy. Fix: 300-line hand-labelled gold of lines from real user-style input, scored for (canonical, qty, unit). Cost: 2 hours of labelling. Not currently measured, so the "target >= 0.85 met" claim (README:52, EVALUATION.md:28) is unsupported as a quality claim.

**F3.6 Doubled-token fix incomplete. Medium. VERIFIED.**
The matcher is fixed (multiset coverage), but artefacts remain in the artefacts: vocab has 108 entries with repeated tokens; 46 are adjacent doubles ("tomato tomato" 1,628 uses, "chicken chicken" 906, "onion onion" 683, "sugar sugar" 599, "lemon juice juice" 580, "garlic garlic" 523, "extra virgin extra virgin olive oil" 645) and 107 distinct such canonicals (13,275 occurrences) still appear in `recipes.jsonl` (they occur when the source line truly repeats the word). `aliases.json` still lists "cucumber cucumber", "ginger ginger". Analogous classes: order-permuted duplicates (8.3% of vocab), prep/container words kept as part of the name, trailing "of" ("lemon juice of"), one-word adjectives ("large"), and accent stripping. Fix: vocab post-filter (drop adjacent repeats by collapsing, drop single-word stoplist, drop container/prep leading tokens, sort-key dedup).

**F3.7 Quantity parsing and source corruption. High. VERIFIED.**
In the Recipes1M-sourced tail, fractions are mangled in the source text: "13 cup soy sauce" (was 1/3), "14 cup vegetable oil" (1/4). In the sample, lines matching `^(12|13|14|34|...) (cup|tsp|tbsp|oz|lb)` are 0.2% in the Gathered half and 6.3% (4,707 of 74,688) in the tail; some are legitimate "14 ounce" cans, so treat 6.3% as an upper bound. `parse_quantity` reads these as 13 or 14 cups. Also: ranges take the first number; `" 2 -4 lbs"` and `"9\t2/3"` (crashed the first full clean, `clean_run.log`) show brittle regexes; units lack "head", "bunch", "package size in parens" ("1 (14 oz) can") so can-size is dropped by `_PARENS` in `normalize`.

## 4. Autocomplete feasibility
Using 11.33M canonical occurrences across 29,548 distinct canonicals actually used:
| Coverage of occurrences | Top-N canonicals |
|---|---|
| 50% | 121 |
| 80% | 1,042 |
| 90% | 2,904 |
| 95% | 6,178 |
| 99% | 16,593 |
So a curated 3,000-entry list covers about 90% of occurrences, 6k covers 95%. Because the junk and duplicates (F3.3, F3.4) inflate these counts, a cleaned list would cover 95% with perhaps 2,000-3,000 entries. Recommended: top ~3k after rule-based collapse plus manual review of the top 500, with synonyms mapped, display names singular-friendly, and a "other (type your own)" fallback.

## 5. Nutrition

**F5.1 End-to-end totals are implausible; several mass conversions are wrong. Critical. VERIFIED (`f_nut2.py`, 20 recipes).**
- A spareribs recipe (title withheld): total 11,005 g, 43,568 kcal. Cause: source line "13 cup soy sauce" read as 13 cups (F3.7).
- "3 cloves garlic" resolves to 408 g (3 x 136 g): `mass.py:~95` falls through to USDA `each` (a whole head, 136 g) because `clove` is not in `unit_grams`. Real value about 9 g. Garlic's mean mass in the 2000-recipe run was 319 g per line.
- "2 eggs" = 56.7 g (matched to "Eggs, scrambled, frozen mixture", each = 28.35 g); real ~100 g.
- Three recipes (titles withheld): a brownie at 2 g total and 0 kcal, a macaroni and cheese with no mass at all, and a pie at 94 g.
- Mass-resolved lines get totals as if complete; per-100g is then computed over the resolved fraction only.

**F5.2 Wrong USDA matches on the most frequent ingredients. Critical. VERIFIED (`e_nut.py`, 2000 recipes, 18,371 lines).**
First-hit "exact" on head-before-comma picks the first dict entry, not a sensible default:
sugar to "Sugars, brown"; butter to "Butter, Clarified butter (ghee)"; egg to "Eggs, scrambled, frozen mixture"; water to "Water, non-carbonated, bottles, natural fruit flavors, sweetened"; milk to "Milk, buttermilk ... reduced fat"; whole milk to "Cheese, mozzarella, whole milk"; vegetable oil to "Vegetable oil, palm kernel"; oil to "Oil, bearded seal (Oogruk) (Alaska Native)"; tomato to "Tomatoes, sun-dried"; carrot to "Carrots, canned"; potato to "Potatoes, hash brown"; mushroom to "Chanterelle"; chicken breast to "fat-free mesquite sliced"; chicken broth to "Chicken, canned, no broth"; dry white wine to "Beans, Dry, Small White"; shallot to "freeze-dried" (mean 2.4 g); yellow onion to "sauteed". These are deterministic mismatches for core staples; "flour" and "cream" happen to land on reasonable entries (all-purpose enriched; heavy/whipping). Share of kcal by match method: 56% exact, 38% Jaccard, 5.5% alias, so about 94% of computed calories come from un-curated matches. The "never a guess" claim (`match.py:5`) does not hold: Jaccard threshold 0.34 is a guess.

**F5.3 Coverage numbers are biased and measured on lines, not on calories. Medium. VERIFIED.**
`nutrition_main` uses `load_raw_recipes(limit=5000)`, i.e. the first 5,000 raw rows (Gathered, head of file), not a random sample. On a random 2,000 recipes (`e_nut.py`, 18,371 lines) I get match 0.710 and mass 0.439 versus the card's 0.800 and 0.487. Both are line counts; no calorie- or gram-weighted coverage is reported. Unresolved lines are disproportionately big-ticket (cans, packages, "1 lb chicken" is resolved but "1 package pasta" is not), so calorie-weighted coverage is likely lower than 0.44. How much of true calories is accounted for is HYPOTHESIS: Kaggle experiment `nutr_gold` on Nutrition5k/recipe subset below.

**F5.4 Partial totals are exposed as totals. High. VERIFIED.**
`aggregate.py` returns `facts_total` with unresolved lines silently absent; `match_coverage`/`mass_coverage` are attributes that no CLI path shows. No serving count exists (`DATASET.md:49`, model card limit 2). An app showing "Calories: 1,450" is both undercounting and not per serving.

**F5.5 Imputer is not wired. Low. VERIFIED.** `grep` finds `MacroImputer` referenced nowhere outside `impute.py` (only its gate in `nutrition_main.py:25`). The model card says so (limit 7), so honest. But hashed 256-feature ridge regression on a name is unlikely to help; skip.

**F5.6 Model card overclaims. Medium. VERIFIED.**
(a) "never a guess" (match tier 3 is similarity-based); (b) "Curating the 34 aliases lifted coverage 0.736 to 0.800" - coverage, not correctness, and measured on a non-random head sample; aliases themselves contain duplicates ("cucumber cucumber", "stalk celery stalk"); (c) "Dietary accuracy 1.0 on n=10" (F6.1) is stated honestly as a smoke test, but the table still reads as a metric; (d) "unmatched lines ... a recipe with low resolution reads as low-coverage rather than as a falsely-small total" - true only if the UI displays coverage; no code path does.

**F5.7 Proposed honest presentation.** Show a range or band only when mass coverage >= 0.8 (calorie-weighted) AND no flagged line; show "estimate for N of M ingredients" otherwise; never show a single calorie figure. Present per-recipe and "per 100 g of counted ingredients", and offer user-entered servings (default 4, clearly labelled) rather than guessing. Fix first: matcher defaults (prefer SR "raw"/generic entries by a curated priority; blocklist cooked/canned/dried/branded descriptors unless the ingredient says so), `clove` to 3 g, `egg` to 50 g, and bad-fraction repair (F3.7).

**F5.8 Validation benchmarks.** (1) Nutrition5k (Google, ~5k plated dishes with measured per-dish calories/macros; CC BY 4.0), good for ingredient-level mass+macro sanity but ingredient names are US cafeteria items; (2) USDA FNDDS (public domain) has recipe-level ingredient lists with computed nutrients per portion - the best fit for an apples-to-apples test of parse to match to mass to macros; (3) Recipe1M+ nutrition is not provided. Kaggle experiment: run the pipeline on 500 FNDDS recipes and report MAPE for kcal and the share within +-25%. Cost: 1 day.

## 6. Dietary ontology

**F6.1 The tagger judges 18.7% of the vocabulary; everything else is blocked. Critical. VERIFIED.**
`dietary_labels.json` has 10 entries (312 bytes). On the 30,481-entry vocab, only 5,701 (18.7%) are "known" (`ontology.py:_is_known`: must contain a leaf token or be in `_CURATED`). Weighted by occurrence, 75.5% of ingredient mentions are unjudged. Because `is_valid` rejects unjudged items under a diet, "salt", "onion", "garlic", "rice", "tomato", "sugar", "water", "pepper", "spinach", "olive oil" (via curated only), "lentil", "tofu"(curated) etc. are blocked. Applying the filter across the 19,910-recipe sample: vegan passes 0.01% of recipes, vegetarian 0.16%, gluten-free 0.20%, dairy-free 0.08%. As a recipe filter this is unusable. (Currently wired only to substitution; a public app will want it for recipes.)

**F6.2 Token matching produces unsafe-looking false positives and some real misses. High. VERIFIED (`c_diet.py`, 156 ingredients x 4 diets).**
- False positives (safe food blocked as dairy): butter bean, peanut butter, coconut milk, almond milk, soy milk, cocoa butter, shea butter, coconut cream, cream of tartar. These are in the vegan, dairy_free lists wrongly (20 wrong blocks), the same class of bug as the "eggplant" one that was fixed by hand for one item.
- Miss: egg noodle is allowed under gluten_free (the "noodle" token is unknown, so wait: it is tagged egg only; `is_valid` passes because `egg` is not forbidden by gluten_free). Real gluten exposure allowed. Same pattern for "noodle", "tortilla", "cracker", "malt vinegar", "seitan", "couscous" (ok), "beer": no gluten leaf, so they are "unknown" and blocked in GF only because unknown - correct by accident.
- False positive: oat flour as gluten (token "flour"). Honey tagged correctly. Gelatin, lard, worcestershire (fish), fish sauce correct. Anchovy paste, chicken stock, beef bouillon, marshmallow (gelatin) judged correct only by token/unknown. Parmesan alone and caesar dressing: unjudged, blocked (safe), but Parmesan is not recognised as dairy, so e.g. for a future "contains" display it would be wrong.
- Per diet confusion (156 items): vegan 73 TP / 0 FN / 67 FP (58 of which just unjudged); vegetarian 41/0/69; gluten_free 26/1/74; dairy_free 27/0/87. Recall is high (the one FN is egg noodle under GF), precision is poor, mostly via unjudged items.
- Diets supported: vegan, vegetarian, gluten_free, dairy_free only. No nut/peanut, shellfish, soy, sesame, egg-free, halal, kosher, pescatarian. Vegetarian forbids only meat/fish: note "gelatin" and "lard" map to meat, ok; but "vegetarian" ignores rennet-parmesan etc. (acceptable convention, should be disclosed).
- Fix: replace with a curated ~3,000-row ingredient to attributes table (allergen flags per EU-14 plus diet flags), derived once from the autocomplete list (F4), with explicit "contains / may contain / safe / unknown" tri-state and an "unknown" UI state instead of silent blocking. Worth it: required before any "vegan filter" button exists. Allergen claims need a disclaimer.

## 7. Pipeline efficiency

- **Storage (F7.1, Medium, VERIFIED):** `recipes.jsonl` is 610 MB, and the raw lines are stored but dropped on load (`store.py:35`: "ingredients_raw ... read by nothing downstream"). Parquet/Arrow with a dictionary-encoded canonical list column would be ~10x smaller and loads in seconds; a web app needs a SQLite/DuckDB index, not a JSONL scan (`cook_command` builds the full `InvertedIndex` each call from the 610 MB file).
- **Redundant work (F7.2, Medium):** vocab pass reads the 2.2 GB CSV once (about 8 to 30 min), then the clean pass reads it again (27 to 37 min; `clean_run*.log`, `reclean.log`, `regen_full.log`). Both use `iterrows` over pandas chunks. A single streaming pass with `csv` plus multiprocessing would cut this substantially. Nutrition re-parses raw lines from the CSV every time (`aggregate.py` docstring), and the Jaccard matcher is a linear 8,204-set scan per call (my 2,000-recipe test took about 4 minutes); cache `canonical -> fdc_id` (30k keys) offline once.
- **`recipes.buggy.jsonl` (F7.3, Low, VERIFIED):** 625 MB, gitignored by `data/processed/*`, referenced by no source file, test, Makefile or doc (only possibly CODEX_HANDOFF which grep did not match). The newer `recipes.jsonl` supersedes it. Safe to delete after one confirmation that no experiment pins it; frees 625 MB, which matters given the 3 GB RAM block.
- **Regenerate on Kaggle:** after fixes in F2.1, F3.3, F3.4, F3.7, F5.2: vocab, `recipes.parquet`, canonical to FDC map, alias table, and the MinHash duplicate report. Retrain recommender/substitution afterwards (their vocab changes).
- Reproducibility (F7.4): vocab `min_count` default differs between the script (5) and `build_vocabulary` (2); doc says 5. Record the command line in the artefact metadata.

## Contradictions with docs
1. `DATASET.md`/`clean.py`: "largely pre-deduplicated" vs 43% of rows removed by title dedup, with false collisions.
2. `match.py`/model card: "never a guess" vs 0.34 Jaccard fallback providing 38% of computed calories, and bad exact-head picks.
3. Model card nutrition coverage 0.800/0.487 comes from the first 5,000 rows, not a random sample (0.710/0.439 on random).
4. README "Parser F1 0.864 target met" measured against NER from the same data as the vocab.
5. README MIT badge vs data licence non-commercial; DATASET.md "record versions" table is empty.
6. `ontology.py` header: "100% by construction for tagged constraints" is true but coverage 18.7% is not stressed in the model card; dietary accuracy 1.0 on n=10.

## Ranked recommendations
1. Resolve data licensing before any public deploy (F1.1): drop RecipeNLG/Recipe1M text from the app; plan a licence-clean recipe source. (Critical, mandatory)
2. Build a pantry-entry resolver: curated autocomplete (~3k), synonym map, accent folding, fuzzy fallback; do not pass raw strings to `recommend` (F3.1, F3.3, F4). (Critical, 2 to 3 days)
3. Collapse vocab/canonicals by rules (garlic vs clove garlic, drop "large", container/prep words, permutations, adjacent doubles, singulariser exceptions) and reclean on Kaggle (F3.3, F3.4, F3.6). (High)
4. Replace the dietary tagger with a curated attribute table and a tri-state output; expand diets/allergens (F6). (Critical for any diet feature)
5. Fix nutrition before showing any numbers: priority-ordered USDA defaults, clove/egg portions, repair "13 cup"/"14 cup" fractions, display coverage and ranges, user-entered servings (F5.1 to F5.7). Hide nutrition in v1 if unfixed.
6. Redo dedup using title plus ingredient-set MinHash (F2.1, F2.2) and create a stored deterministic split file (F2.3).
7. Build a small hand-labelled parser gold set (300 lines) and a 500-recipe FNDDS nutrition validation (F3.5, F5.8).
8. Record checksums, row counts, and release metadata (F1.2); move to Parquet/DuckDB for serving; delete `recipes.buggy.jsonl` (F7).

## Sources
- RecipeNLG HF card and research-use agreement: https://huggingface.co/datasets/recipe_nlg
- RecipeNLG project page (fetch returned title only): https://recipenlg.cs.put.poznan.pl/
- USDA FoodData Central: https://fdc.nal.usda.gov/ (download page https://fdc.nal.usda.gov/download-datasets/ not fully retrieved; public-domain/CC0 statement and citation request taken from the repo's own DATASET.md and general FDC policy, not independently verified here)
- Local evidence: `data/raw/full_dataset.csv`, `data/processed/{recipes.jsonl,vocab.json,usda.json}`, `pantrychef/{ingredients,nutrition,dietary,data,retrieval}/*.py`, `docs/{DATASET,MODEL_CARD_nutrition}.md`, `data/processed/*.log`
