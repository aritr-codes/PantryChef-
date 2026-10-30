# PantryChef research findings

Status: consolidation of the Stage 1 research. Date: 2026-10-30.
Readers: the project owner, and any developer who must decide between options.

This document joins the seven Stage 1 reports into one set of decisions. It does not
copy the reports. Follow the links for detail. It describes research and decisions. It does
not describe built software. The web app and the TypeScript engine do not exist yet.

Order of authority when sources differ:

1. [ADR 0003](decisions/0003-licence-clean-recipe-corpus.md), [0004](decisions/0004-static-first-browser-engine.md), [0005](decisions/0005-honest-claims-and-eval-reset.md). All accepted on 2026-10-30.
2. The [Stage 1 synthesis](research/2026-10-stage1/README.md) and its [review](research/2026-10-stage1/REVIEW.md). The review overrides the numbered reports where they differ.
3. The numbered reports [01](research/2026-10-stage1/01_recommender.md), [02](research/2026-10-stage1/02_substitution.md), [03](research/2026-10-stage1/03_vision.md), [04](research/2026-10-stage1/04_product.md), [05](research/2026-10-stage1/05_deployment.md), [06](research/2026-10-stage1/06_data_nutrition.md), [07](research/2026-10-stage1/07_engineering_claims.md).
4. The evidence scripts and files in [evidence/](research/2026-10-stage1/evidence/).

## 0. How to read this document

### Labels

| Label | Meaning |
| --- | --- |
| MEASURED | A script, log or run produced the number. The source is named. |
| DOCUMENTED | A vendor page, paper or dataset card says it. The source and date are named. |
| ESTIMATE | A calculation or a secondary source. Nobody measured it. |
| HYPOTHESIS | An experiment must confirm it. |

Dates are 2026-10-30 unless the text says otherwise. The Stage 1 reports are dated 2026-10-30.
Their web sources were fetched on that day. "MEASURED (this task)" means the data checks made
for [PRODUCT_VISION.md](PRODUCT_VISION.md) on the Wikibooks snapshot on 2026-10-30.

### Terms

- **Recall@10**: the share of test queries where the right recipe is in the top 10.
- **MRR@10** (mean reciprocal rank): the average of 1 divided by the rank of the right recipe, counting the top 10 only.
- **IDF** (inverse document frequency): a weight that makes rare ingredients count more than common ones.
- **BM25**: a standard text-search score from the IDF family.
- **LambdaMART**: a learned ranking model made of decision trees.
- **ANN** (approximate nearest neighbour): a fast, inexact vector search.
- **ADR**: architecture decision record.
- **SC**: a WCAG success criterion.
- **CC BY-SA 4.0**: a licence that requires credit and requires adaptations to use the same licence.
- **AGPL**: a copyleft licence that also covers use over a network.
- **Pack**: the one versioned file of recipes that the app loads.

### Criteria for each option

Each table scores the options on these points. The columns are shortened.

| Column | Covers |
| --- | --- |
| Gain | Expected quality gain for users |
| Complexity | Build effort and moving parts |
| Cost | Compute (C), memory (M) and inference cost (I) |
| Deploy | Deployment difficulty |
| Upkeep | Maintenance and reproducibility |
| Solo | Fit for one person with no budget |

## 1. Summary of decisions

| Area | Decision | Basis |
| --- | --- | --- |
| Display corpus | Wikibooks Cookbook, about 3,800 recipes, CC BY-SA 4.0 | ADR 0003 |
| Training corpus | Open Recipes for statistics only. Statistics ship only if mined swaps pass the M5 blind rating. If they fail, nothing from Open Recipes ships | ADR 0003 |
| RecipeNLG | Research and comparison only | ADR 0003 |
| Hosting | Static site on GitHub Pages | ADR 0004, report 05 |
| Engine | TypeScript in a Web Worker, with a Python reference and parity tests | ADR 0004 |
| First ranker | IDF-weighted overlap. A learned ranker ships only if it wins on hand-judged pantries | ADR 0005, synthesis 4.3 |
| Substitution | Curated swaps ship in v1 (about 200 to 300 rows). Mined swaps, limited to the same role, ship in v1 only if they pass the M5 blind rating. Otherwise they do not ship | Report 02, ADR 0003, DEVELOPMENT_PLAN.md M5 |
| Dietary | Three-state preference filter. No allergen claim | ADR 0005 |
| Ingredient data | One curated ingredient table, built from Wikibooks lines | Synthesis 4.2 |
| Nutrition | Not in the first release | ADR 0005 |
| Vision | Not in the first release. Browser tile matching is a later hypothesis | Report 03 |
| Accounts and tracking | None | Report 04, ADR 0004 |
| Retired claim | "Graph beats food2vec" | ADR 0005 |

## 2. Recommendation and retrieval

Sources: [report 01](research/2026-10-stage1/01_recommender.md), [synthesis 2.1 and 4.3](research/2026-10-stage1/README.md), [review](research/2026-10-stage1/REVIEW.md), [report 07 F1](research/2026-10-stage1/07_engineering_claims.md).

### 2.1 What we know

- The old recovery test builds each pantry by removing 30% of one recipe's ingredients. The pantry is always a subset of the right recipe. A real pantry is not (MEASURED, reports 01 and 07, two independent runs).
- A rule with no machine learning reached recall@10 0.975 against overlap at 0.740 (50k recipes, 1,404 queries). LambdaMART scored about 0.980 in the project log on a different query set of 9,354 (MEASURED, report 01).
- On 20k recipes and 1,500 queries the rule scored MRR@10 0.952, LambdaMART 0.958 and overlap 0.544 (MEASURED, report 07).
- The review added pantries with unrelated extra items (recall@10, 20k spread sample, MEASURED, no paired intervals):

  | Extra items | Overlap | Fraction rule | Model, clean training | Model, retrained | IDF cosine |
  | --- | --- | --- | --- | --- | --- |
  | 10 | 0.322 | 0.853 | 0.601 | 0.943 | 0.958 |
  | 20 | 0.113 | 0.693 | 0.361 | 0.830 | 0.914 |

  At 3 extra items the retrained model scored 0.985 and the IDF rule 0.980.
- The extras came from a frequency sampler. That favours IDF. Other samplers are untested (the review says so).
- "Saturates at 10k training queries" is not supported. Different models gave the same gold rank on 2,000 of 2,000 queries, so the metric cannot see the difference (MEASURED, review).
- The full-corpus lead of 7.7 times predates the parser fix. No post-fix full-corpus result exists (`p3_full_fixed.log` ends in `MemoryError`) (MEASURED, report 07).
- The size of the corpus changes the problem. The old work chose among 1.27 million recipes. The display corpus now holds about 3,800. At that size the engine can score every recipe for every query. The old candidate ceiling problem (0.983 at 50k, 0.706 at 1.27M, pre-fix) does not apply (HYPOTHESIS: the claim that a full scan is fast enough in a browser needs a latency test).

### 2.2 Options

| Option | Gain | Complexity | Cost | Deploy | Upkeep | Solo | Decision and reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Coverage overlap (current baseline) | Low. MRR 0.544; prefers tiny recipes | Low | C, M, I: tiny | Easy | Easy | Good | **Reject as the ranker.** Keep as a named baseline |
| IDF-weighted overlap or cosine | High on pantries with extras (0.914 to 0.958 at 10 to 20 extras, MEASURED) | Low: one weight vector | C, M: tiny. I: a scan of about 3,800 recipes (ESTIMATE) | Easy: same maths in TypeScript | Easy; a parity test covers it | Excellent | **Adopt for v1.** ADR 0005 names it the rule to beat |
| BM25 over ingredient ids | Same family as IDF. Gain over IDF unknown | Low | Same as IDF | Easy | Easy | Excellent | **Test beside IDF.** Choose by hand-judged data. Do not ship both |
| Report 01's normalised coverage form | Poor: 0.194 at 20 extras against 0.914 | Low | Tiny | Easy | Easy | Good | **Reject** (MEASURED, synthesis 4.3) |
| LambdaMART with features that do not depend on overlap | Unknown. No demonstrated lead over IDF | Medium. Training, export, a tree walker in about 30 to 60 lines | C: about 25 min training (DOCUMENTED, repo notes, no completed run). M: 431 KB model. I: low ms | Medium: parity needed | Medium: retrain on each pack change | Fair | **Defer.** Ship only if it beats IDF on 100 hand-judged pantries (ADR 0005). If it loses, report the loss |
| Graded "makeability" labels | None. The label is a function of `n_missing`, which is a feature | Low | n/a | n/a | n/a | n/a | **Reject** (synthesis section 3) |
| Approximate nearest neighbour, FAISS, two-tower, SPLADE | Negative: blurs exact set matching | High | C: training. M: vectors. I: extra | Hard in a browser | Hard | Poor | **Reject.** Neither FAISS nor two-tower code exists in the repo (MEASURED, report 07 ledger 2 and 3) |
| WAND or MaxScore | None at this size | Medium | n/a | Medium | Medium | Poor | **Reject.** Add only if latency tests fail |
| Compressed sparse row (CSR) and memory-mapped index | Needed only on the Python side for the full corpus | Medium | M: 100 to 150 MB in Python (ESTIMATE, report 05) | n/a | Medium | Fair | **Skip for v1.** The pack is small. Keep as a note for a "more ideas" pack |
| Dedup and quality filter | Low at 3,800: no duplicate titles (MEASURED, this task) | Low | Tiny | Easy | Easy | Good | **Do a light check.** Compare ingredient sets for near-duplicates before release |
| Quality proxies (cluster size, a classifier from rated data) | Unknown. No ratings exist in the pack | High | Medium | Medium | Medium | Poor | **Reject for v1.** Wikibooks gives only a 1 to 4 difficulty number |
| LLM reranker | Unknown | High | High | Needs a server | High | Poor | **Reject.** The project excludes LLMs |
| Graph neural network | Unknown | High | High | Hard | Hard | Poor | **Reject.** No interaction data |
| ONNX Runtime Web for trees | None over a tree walker | Medium | M: about 10 MB wasm (ESTIMATE, report 05) | Medium | Medium | Poor | **Reject.** Use a small tree walker if a model ships |

### 2.3 Evaluation plan (from ADR 0005 and the synthesis)

- Keep masked recovery as a sanity test only. Put the no-ML rule on its leaderboard.
- Add pantries with distractors. Use more than one sampler for the extras.
- Report "makeable at k" with staples removed as a product measure only. It cannot select a ranker, because sorting by missing count wins it by construction.
- Build a hand-judged set: 100 pantries, top 5 results from 3 systems, bootstrap intervals. It can select a ranker. It is too small for LambdaMART. It may suit a small pairwise linear model (HYPOTHESIS). The set is built in M4 on the post-M3 pack.
- Write every result as JSON with commit, dirty flag, config, corpus fingerprint and metrics.

### 2.4 Source list for this section

- KERL, LLM-based recipe work: <https://arxiv.org/pdf/2505.14629>
- ONNX Runtime Web: <https://opensource.microsoft.com/blog/2021/09/02/onnx-runtime-web-running-your-machine-learning-model-in-browser/>
- LightGBM: <https://lightgbm.readthedocs.io>
- Burges, "From RankNet to LambdaRank to LambdaMART", MSR-TR-2008-109.

## 3. Substitution and the dietary guardrail

Sources: [report 02](research/2026-10-stage1/02_substitution.md), [report 06 section 6](research/2026-10-stage1/06_data_nutrition.md), synthesis 2.2 and 2.4.

### 3.1 What we know

- The mined model finds name variants more than substitutes. Hand check of the top result for 50 common ingredients (one rater, not blind, indicative): 39 variants, 5 real substitutes, 4 usable with a caveat, 2 wrong (MEASURED, review; replaces the 12/24/6/8 count in report 02).
- Graph minus embedding MRR is +0.030 with a 95% interval of -0.048 to +0.111 at n=82. The data cannot tell them apart. The shrinkage parameter was tuned on the same 82 queries (MEASURED, report 02).
- The gold set is mined from the training corpus. 36% of its pairs are name variants (MEASURED, report 02; hand label not rechecked by the review).
- The GISMo row predates the parser fix and was never rerun (MEASURED, report 02).
- The benchmark protocols differ. GISMo reports MRR 0.3151. Its own lookup-table plus frequency baseline reaches 0.2766. Food2Vec reaches 0.0965 (DOCUMENTED, arXiv 2302.07960, Table 1, read by the report author). Our figure of about 0.08 is Food2Vec class on a different protocol.
- Speed: 0.13 s per query, one thread. The bundle is 123 MB and needs scipy and gensim (MEASURED, report 02).
- Dietary: "100% validity" re-checks results with the same function that filtered them, so it cannot fail (MEASURED, report 02). The tagger judges 18.7% of the old vocabulary. It let egg noodle, cream of mushroom soup and chicken noodle soup pass "gluten free". It blocked peanut butter, coconut milk, salt, rice and onion under some diets (MEASURED, synthesis 2.2).
- Under the old tagger, a vegan filter would keep about 0.01% of sampled recipes (MEASURED, report 06).

### 3.2 Options

| Option | Gain | Complexity | Cost | Deploy | Upkeep | Solo | Decision and reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Current mined hybrid model | Low for users: variants and complements | Done | C: 0.13 s/query. M: 123 MB bundle | Hard in a browser | Medium | Fair | **Reject for v1.** Its quality on the new vocabulary is unknown |
| Curated functional table: about 200 to 300 rows with role, swaps, ratios, caveats, diet flags | Highest. Fixes buttermilk, egg, baking powder and soda, cornstarch, vegan and gluten-free swaps (HYPOTHESIS, test E1) | Low. Mostly manual: 2 to 3 days (ESTIMATE) | Tiny. KB scale | Easy | Easy; a human reviews each row | Excellent | **Adopt.** Ships in v1. Wikibooks "Cookbook:Substitutions" can be one source. If a row draws on that page, the file is CC BY-SA and the manifest says so |
| Mined candidates limited to the same role, collapsed to the head noun, then "no confident swap" | Medium (HYPOTHESIS) | Medium | Precomputed top-k table, 1 to 2 MB (ESTIMATE, report 02 arithmetic) | Easy once precomputed | Medium | Fair | **Ships in v1 only if it passes the M5 blind rating** (at least 70% rated "I would use this"). Otherwise it does not ship |
| Precomputed top-k table in place of the 123 MB bundle | Same quality; removes the slow path | Low | Build in 10 to 15 minutes on a CPU session (derived estimate) | Easy | Easy | Good | **Adopt if any mined fallback ships** |
| GISMo model or data | Better on its own protocol (MRR 0.315) | High: registration and GPU time | High | Hard | Medium | Poor | **Reject for shipping.** CC BY-NC. Use for evaluation only, with attribution (DOCUMENTED, GISMo repo) |
| Lookup table plus frequency (GISMo baseline) | MRR 0.277 on its protocol | Low | Tiny | Easy | Easy | Good | **Use as an evaluation prior only.** Its training data is non-commercial |
| Food2Vec, FoodKG | No better than our embedding arm | Low | Low | Medium | Medium | Fair | **Reject.** No gain |
| FlavorGraph | Flavour pairing, not substitution | Medium | Small embeddings | Medium | Medium | Fair | **Reject for now.** Licence of the repo is unverified |
| Epicure (arXiv 2605.22391, May 2026) | Unknown for substitution; its abstract has no substitution evaluation | Low to inspect | Small CSVs | Easy | Easy | Good | **Look at it** for the ingredient table. See section 8. Require source-corpus provenance first, because its abstract does not name its 11 sources |
| FoodOn, Wikidata or USDA categories to gate candidates | Medium: removes complements and gives diet hints | Medium | Tens of MB offline | Precompute | Medium | Fair | **Optional.** USDA is public domain. FoodOn licence is unverified |
| Role constraint (leavener, fat, binder, acid, sweetener, thickener) on top of a mined model | High ROI over the mined model alone (HYPOTHESIS) | Low: tag about 300 heads by hand | Tiny | Easy | Easy | Excellent | **Adopt with the mined fallback** |
| Old tagger (token matching) for diets | Unsafe | Done | Tiny | Easy | Hard to fix | Fair | **Reject** |
| Curated three-state ingredient table (suits, does not suit, not sure) with "any unsafe token blocks" for composites | High for trust | Medium: about 500 rows first | Tiny | Easy | Medium | Good | **Adopt.** Ship the diet filter only after it passes a check on independent labels |
| Allergen model (nuts, shellfish, sesame and so on) | Real need, but high harm if wrong | High | Low | Easy | Hard | Poor | **Reject for v1.** No allergen claim (ADR 0005). If added later, use "may contain" wording |
| Independent diet labels (Open Food Facts or FoodOn flags) to validate the table | Gives a metric that can fail | Medium | Low | n/a | Medium | Fair | **Plan.** Licences are not verified |

### 3.3 Source list for this section

- GISMo paper: <https://arxiv.org/abs/2302.07960> (HTML: <https://arxiv.org/html/2302.07960>)
- GISMo code, majority CC BY-NC: <https://github.com/facebookresearch/gismo>
- Epicure: <https://arxiv.org/abs/2605.22391>
- FlavorGraph: <https://doi.org/10.1038/s41598-020-79422-8>

## 4. Computer vision

Source: [report 03](research/2026-10-stage1/03_vision.md). The decision for v1 is simple: do not build it.

### 4.1 What we know

- No public dataset covers real fridge clutter with enough classes (report 03 finding; not an exhaustive survey).
- The best fridge set found is Roboflow "AICook": 516 images, 30 classes, MIT per a search snippet. The page returned 403 (unverified).
- No measured fridge benchmark exists for any approach. This is a gap, not a result.
- YOLOv8, 11 and 26 from Ultralytics are AGPL-3.0 or paid (DOCUMENTED, ultralytics.com/license). Shipping weights in a public MIT repo clashes with that. This is an engineering reading, not legal advice.
- SigLIP2 and OpenCLIP are permissive (Apache-2.0, MIT). MobileCLIP2 uses the `apple-amlr` licence (DOCUMENTED, model cards). Apache-licensed RF-DETR Nano and YOLOX are possible detectors.
- All browser latency and size numbers for vision are ESTIMATES. No low-end-phone figure was found.
- The old vocabulary has 30,481 phrases from recipe text. No detector can target it. A curated visual list of 150 to 200 items is needed whichever model is chosen (MEASURED, report 03).

### 4.2 Options

| Option | Gain | Complexity | Cost | Deploy | Upkeep | Solo | Decision and reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Skip vision in v1 | Avoids errors that damage trust | None | None | None | None | Excellent | **Adopt.** Typing and autocomplete come first (report 04) |
| Tile matching with SigLIP2 or OpenCLIP in the browser, against about 150 to 200 text embeddings | Per-photo F1 0.4 to 0.65 (ESTIMATE). No training | Medium | M: about 25 to 100 MB quantised (ESTIMATE). I: 100 to 500 ms per tile (ESTIMATE). Nothing is uploaded | Medium: host weights on Hugging Face and lazy-load | Medium | Fair | **Later beta (HYPOTHESIS E2).** Ship only at F1 of 0.5 or more and 5 edits or fewer per photo |
| Fine-tuned closed-set detector (RF-DETR Nano or YOLOX) | F1 0.5 to 0.7 on about 50 to 60 classes (ESTIMATE) | High: weeks of data cleaning | C: 3 to 6 hours on a GPU. M: 10 to 100 MB | Medium | Hard | Poor | **Fallback only** |
| YOLO from Ultralytics | Good, known tools | Low | Medium | Easy | Easy | Good | **Reject.** AGPL |
| Open-vocabulary detectors (OWLv2, Grounding DINO) | Any text label; packaged goods unmeasured | Medium | M: 150 to 700 MB. I: seconds | Server only | Medium | Poor | **Reject.** Needs a server |
| Whole-image multi-label classifier | Hard even in-domain | Medium | Small | Easy | Medium | Fair | **Reject.** No fridge data |
| Server-side CPU inference | Simple | Low | I: 1 to 3 s (ESTIMATE) | Needs an always-on host | Medium | Fair | **Reject.** No free always-on host (section 6) |
| MobileCLIP2 | Small and fast | Low | Low | Easy | Easy | Good | **Reject for now.** Licence `apple-amlr` needs a careful read |
| Cloud vision API | High | Low | Per-call cost, privacy cost | Easy | Easy | Fair | **Reject.** Breaks the no-upload promise |

### 4.3 Test plan if vision returns

100 to 150 photos with per-photo ingredient sets. Metrics: precision, recall, F1, edits needed, recall@K. No boxes and no mAP. Report produce and packaged goods apart (report 03 section 6).

### 4.4 Source list for this section

- AICook: <https://universe.roboflow.com/objectdetectionlearning-4empo/aicook-lcv4d-zzbgs>
- Other Roboflow sets: <https://universe.roboflow.com/objectdetectionlearning-4empo/fridgefooddetection-cxnl3> and <https://universe.roboflow.com/michael-ringer/fridge-vzg6p/dataset/>
- FoodSeg103 card: <https://huggingface.co/datasets/EduardoPacheco/FoodSeg103/raw/main/README.md>
- Fruits-360: <https://github.com/Horea94/Fruit-Images-Dataset>
- Freiburg Groceries: <https://www.arxiv.org/pdf/1611.05799>
- Recipes5k and Ingredients101: <https://arxiv.org/pdf/1707.08816>
- Ultralytics licence: <https://www.ultralytics.com/license> and <https://docs.ultralytics.com/>
- RF-DETR: <https://github.com/roboflow/rf-detr>
- YOLO-World README: <https://huggingface.co/spaces/stevengrove/YOLO-World/blame/a8dc1acccb3b3677f13a447d2e466dcba37417cb/README.md>
- Grounding DINO: <https://playground.roboflow.com/models/idea-research/grounding-dino>
- MobileCLIP2: <https://huggingface.co/apple/MobileCLIP2-B/blob/main/README.md>
- SigLIP2 ONNX: <https://huggingface.co/upshift/siglip2-base-patch16-384-onnx/discussions> and <https://huggingface.co/SkyStar-tech/siglip2-large-patch16-384-ONNX>
- Free GPU summary (secondary): <https://gpuperhour.com/blog/free-cloud-gpus-and-credits>

## 5. User experience and competitors

Source: [report 04](research/2026-10-stage1/04_product.md). The product design is in [PRODUCT_VISION.md](PRODUCT_VISION.md).

### 5.1 Competitor facts

| Product | Fact | Status | Source |
| --- | --- | --- | --- |
| SuperCook | Chips, "can make now", saved pantry. A reviewer reports a pantry lost | DOCUMENTED (App Store listing; review via appshunter.io, secondary) | <https://apps.apple.com/us/app/supercook/id1033049405> |
| MyFridgeFood | Quick-pick, shallow | Secondary summary, unverified at source | report 04 |
| Cooklist | Pantry plus cook now. Reviews cite update glitches and unstable barcode scans | Secondary | report 04 |
| Yummly | Closed 18 to 20 December 2024. Saved libraries were lost unless exported | DOCUMENTED | <https://www.plantoeat.com/blog/2024/12/yummly-is-closing-discover-the-best-meal-planning-alternative> |
| Mealime | A complaint that you cannot filter by ingredients on hand. Pantry features behind Pro | Secondary | <https://unstar.app/blog/allrecipes-yummly-nyt-cooking-paprika-mealime-recipe-apps-ranked-2026> |
| Paprika | Recipe manager, not a pantry search. Complaints on import accuracy | Secondary | report 04 |
| Samsung Food, Plant Jammer, Tasty, Allrecipes, Spoonacular | No sourced detail | **Gap** | n/a |
| Pantry (app) | Anecdote: entering "two bananas" feels like more work than a note | Anecdote | <https://www.pcworld.com/article/524936/pantry.html> |

There is no measured dropout figure. "Pantry entry is the top dropout point" is plausible and unquantified.

### 5.2 Options

| Option | Gain | Complexity | Cost | Deploy | Upkeep | Solo | Decision and reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Chips, autocomplete, quick-pick grid, example pantries | High. A common pattern in competitors | Medium: needs the ingredient table | Tiny | Easy | Medium | Good | **Adopt** |
| Staples assumed and editable | High: salt, pepper and water lift matches | Low | Tiny | Easy | Easy | Excellent | **Adopt.** Apply before ranking, not only in display |
| "Make now", "One item away" sections, with top `unlocks` (items that move the most recipes to make now), and on zero results the `blockers` change | High. The core promise | Low | Tiny | Easy | Easy | Excellent | **Adopt** |
| Pantry in local storage, export and import, URL hash | Prevents the Yummly and SuperCook loss | Low | Tiny | Easy | Easy | Excellent | **Adopt** |
| Accounts and cloud sync | Low in v1; adds a server | High | Needs a host | Hard | Hard | Poor | **Reject.** Violates ADR 0004 |
| Barcode scanning | Medium | High | Medium | Medium | Hard | Poor | **Reject.** Competitors report instability |
| Tracking quantities in the pantry | Medium | High | Low | Easy | Medium | Fair | **Reject.** The entry burden is the known problem |
| Typographic cards, no images | Neutral | Low | Tiny | Easy | Easy | Excellent | **Adopt.** The pack has no images |
| Stock photos or Wikimedia images | Medium | Medium | I: extra downloads | Medium | Medium | Fair | **Reject.** Licence is per file; hotlinks leak visitors' addresses |
| Nutrition view | Medium | High | Medium | Easy | Hard | Poor | **Reject for v1** (ADR 0005; section 9) |
| Ratings, reviews, popularity | Medium | n/a | n/a | n/a | n/a | n/a | **Reject.** The data has none |
| Meal planning, chat, "AI chef" | Unknown | High | High | Hard | Hard | Poor | **Reject** |
| Analytics for success measures | Medium | Low | Low | Easy | Easy | Good | **Reject.** Use the no-tracking measures in PRODUCT_VISION section 16 |

### 5.3 Report 04 revised for the Wikibooks corpus

Report 04 assumed RecipeNLG. Its data findings change as follows. All "now" values are MEASURED (this task) unless marked.

| Topic | Report 04 (RecipeNLG) | Now (Wikibooks) |
| --- | --- | --- |
| Stored fields | Title, raw ingredient lines, canonical set. No directions, no link | `url`, `title`, `infobox` (category, servings, time, difficulty), `text_lines` with section names |
| Directions | Dropped in cleaning; median 216 characters; 11% one step or fewer | Present. Median about 500 characters. 4 recipes have none. 75 are under 80 characters |
| Encoding damage | Seen, for example a lost degree sign | 0 ingredient lines with the replacement character |
| Titles | About 5% with artefacts; family names; the apostrophe bug | No duplicate titles. Titles often hold a native name and a gloss, such as "Kuddi (Spiced Yogurt Sauce)" |
| Images | None | None. No image reference in the file |
| Time and servings | Not usable | About 33% each, as free text. 223 of 1,319 time strings are long or joined (for example "Prep: 20 minutesBaking: 60 minutes"). 428 of 1,296 servings strings are not whole numbers |
| Cuisine and category | Absent | Present as a category, mixing dish type and cuisine. 118 pages have none; 33 point to a missing page |
| Difficulty | Absent | A 1 to 4 number on 3,769 of 3,895 pages. Reliability unknown |
| Ratings | Absent | Absent |
| Link-out for steps | Needed, because steps could not be shown | Not needed. The pack holds the steps. A link to the page is still required for credit |
| Source label | "from food.com" | "Wikibooks Cookbook" |
| Ingredients per recipe | Mean 7.0 | Median 8. 10th to 90th percentile 4 to 14 |
| Amounts | 31% of rows have a line with no leading number | 27% of lines have no leading number |
| Canonical tokens | Machine tokens, junk, brand names | Not built yet. The ingredient table must be built from Wikibooks lines |
| Licence | Non-commercial research | CC BY-SA 4.0. Credit and ShareAlike apply |
| Page leftovers | Not applicable | 57 lines hold `Template:` text. The build must remove them |

Report 04's advice that still holds: staples assumed, sections for Make now and One item away, never drop an input silently, "Looks vegan (judged from ingredient names)" wording, local storage plus export, accessibility notes.

## 6. Deployment and hosting

Source: [report 05](research/2026-10-stage1/05_deployment.md). Vendor figures were fetched on 2026-10-30 unless marked secondary.

### 6.1 What we know

- Hugging Face Docker and Gradio Spaces need a paid plan to create. Static Spaces stay free (DOCUMENTED, <https://huggingface.co/docs/hub/spaces-overview>, 2026-10-30).
- GitHub Pages: 1 GB site, 100 GB per month soft bandwidth, 10 minute deploy timeout (DOCUMENTED, <https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits>).
- The old Python runtime peaks at 2.8 GB on the full corpus and takes about 54 s to load (MEASURED, report 05). That no longer matters for a 3,800-recipe pack.
- The Wikibooks source file is 10.0 MB. A trimmed copy (title, URL, category, ingredient and step lines) is 4.1 MB raw and 1.2 MB gzipped (MEASURED, this task, rough). ADR 0003 estimates under 10 MB for the pack. The two figures agree on order of size. The engine's index is extra.
- Report 05's size table is for RecipeNLG: about 4.5 MB core for 100k recipes (ESTIMATE from a 20k sample). It does not apply to the new pack.
- Browser latency is not measured anywhere.

### 6.2 Hosting options

| Option | Gain | Complexity | Cost | Deploy | Upkeep | Solo | Decision and reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GitHub Pages, static | Always on, no secrets, no cost | Low | Free | Easy: Actions deploy | Low risk | Excellent | **Adopt** |
| Hugging Face Static Space | Same shape as Pages | Low | Free | Easy | Low to medium | Good | **Backup host** |
| Hugging Face Docker or Gradio Space | Python runtime | Low | Paid plan to create | Easy | Policy risk | Poor | **Reject** (DOCUMENTED) |
| Hugging Face Hub as a file CDN | Large shards | Low | Free "best-effort" | Needs a CORS and Range test | Medium | Fair | **Needed only for a vision model.** Test Range and CORS first |
| Render free web | Python runtime | Low | Free | Easy | Sleeps after 15 minutes; about 1 minute wake (DOCUMENTED, report 05) | Fair | **Reject** |
| Fly.io | n/a | n/a | No free tier for new accounts (secondary) | n/a | n/a | n/a | **Reject** |
| Koyeb free | 512 MB, scale to zero | Low | Free | Easy | Medium to high risk (secondary) | Fair | **Reject** |
| Railway | Trial only | Low | $5 trial, then $1/month credit (DOCUMENTED, docs.railway.com) | Easy | High risk | Poor | **Reject** |
| Google Cloud Run | Always scalable API | Medium | Free quota, card on file | Medium | Low risk | Fair | **Later option** for a hybrid API |
| Cloudflare Workers free | n/a | Medium | 128 MB, 10 ms CPU per request (DOCUMENTED) | Easy | Low | Poor | **Reject.** Too small for an index |
| Cloudflare R2 | Free egress for big shards | Medium | 10 GB-month free (DOCUMENTED) | Medium | Low | Fair | **Not needed for the 3,800-recipe pack** |
| Oracle Always Free | Large VM | High | Free, with idle reclamation and conflicting limits | Hard | High risk | Poor | **Reject** |
| AWS Lambda, Vercel, Netlify functions, jsDelivr | Various | Medium | Various | Medium | Medium | Fair | **Not evaluated in depth.** Figures unverified (see gaps) |
| Static site plus TypeScript engine in a Web Worker | No server | Medium: two implementations | C: client. M: small. I: free | Easy | Parity tests needed | Good | **Adopt** (ADR 0004) |
| Pyodide in the browser | Reuses Python | Medium | Tens of MB, seconds to boot on a phone | Medium | Low | Poor | **Reject for the app.** Possible as a CI oracle |
| SQLite-WASM or DuckDB-WASM | Query large files by range | High | 1 to 4 MB of wasm | Needs Range and CORS | Medium | Poor | **Reject.** The pack is small |
| Static frontend plus Python API | One implementation | Medium | Needs a host | Hard | Hard | Poor | **Reject for v1.** Keep the `Engine` interface so it stays possible (ADR 0004) |

### 6.3 Build and operations options

| Option | Gain | Complexity | Cost | Deploy | Upkeep | Solo | Decision and reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Build the Wikibooks pack on a laptop | Fast, reproducible | Low | Seconds to minutes (ESTIMATE) | Easy | Easy | Excellent | **Adopt** |
| Kaggle CPU session for Open Recipes statistics | Handles 173k recipes | Medium | CPU only. Quotas from secondary sources | Medium | Medium | Fair | **Use if needed.** Credentials and quota are unverified |
| Release manifest with SHA-256, commit, corpus fingerprint | Reproducible artifacts | Low | Tiny | Easy | Easy | Excellent | **Adopt** |
| DVC | Data versioning | Medium | Extra remote | Medium | Medium | Poor | **Reject.** The manifest does the job |
| MLflow | Run tracking | Medium | Extra dependency; 4 logged runs had no metrics | Medium | Medium | Poor | **Reject.** Delete it (synthesis section 5) |
| Hydra | Config | Low | n/a | n/a | n/a | n/a | **Reject.** It is not used in the code (MEASURED, report 05) |
| Golden-file parity test, Python against TypeScript (about 200 fixed pantries) | Prevents drift between two engines | Medium | Tiny | Easy in CI | Medium | Good | **Adopt** (ADR 0004) |
| CI quality gate: lint, tests, small-fixture metric floor, pip-audit | Catches regressions | Low | Tiny | Easy | Easy | Good | **Adopt** |

### 6.4 Source list for this section

- HF Spaces: <https://huggingface.co/docs/hub/spaces-overview>
- HF storage: <https://huggingface.co/docs/hub/storage-limits>
- GitHub Pages limits: <https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits>
- Render free: <https://render.com/docs/free>
- Railway trial: <https://docs.railway.com/pricing/free-trial>
- Cloudflare Workers limits: <https://developers.cloudflare.com/workers/platform/limits>
- Cloudflare R2 pricing: <https://developers.cloudflare.com/r2/pricing>
- Kaggle kernel metadata: <https://github.com/Kaggle/kaggle-api> (docs/kernels_metadata.md)

## 7. Fallback and alternative recipe corpora

Sources: [ADR 0003](decisions/0003-licence-clean-recipe-corpus.md), [corpora_fitness.json](research/2026-10-stage1/evidence/corpora_fitness.json), [report 06 section 1](research/2026-10-stage1/06_data_nutrition.md), [synthesis 2.5 and D1](research/2026-10-stage1/README.md).

### 7.1 ADR 0003 table (MEASURED, 2026-10-30)

| Corpus | Licence | Recipes | Instructions | Vocabulary hit rate | Up to one item away (cumulative), 10 test pantries |
| --- | --- | --- | --- | --- | --- |
| Wikibooks Cookbook snapshot, 2024-07-31 | CC BY-SA 4.0 | 3,800 with an ingredient list | 98.3% | 0.985 | 69 to 330 |
| Open Recipes snapshot, 2017-01-07 | CC BY 3.0 (database) | 173,276 | None, by design | 0.867 | 277 to 4,635 (scaled from a 20k sample) |
| RecipeNLG (reference only) | Research and education | 1,274,290 kept | Yes | Not comparable | 4,448 to 57,392 (scaled from a 1-in-8 sample) |

The pantries are hand-written and the match rule is loose. The numbers are indicative. They are not an evaluation.
The "one item away" column is cumulative: it counts recipes with up to one item missing, so it includes make-now recipes.
The 0.985 figure is a vocabulary hit rate, not parse accuracy. A hand read of 60 lines found at least 4 clearly wrong (one rater, not blind).
The evidence file also shows make-now counts (0 missing): Wikibooks 1 to 73, Open Recipes 17 to 901 (scaled), RecipeNLG 168 to 11,856 (scaled).
The Wikibooks "69 to 330" matches the evidence file.

A stricter rule (recipe staples only, unread lines count as missing) gave Wikibooks make-now 0 to 28 and exactly-one-away 50 to 134, across 10 hand-written pantries. Some strict make-now results are trivial, such as "Steamed Rice".

The measured snapshots are a third-party Wikibooks scrape with no revision ids, and the Open Recipes copy on Hugging Face. Hashes are in ADR 0003. For the product we build our own Wikibooks snapshot from the official MediaWiki API, with a revision id per page.

### 7.2 Options

| Option | Gain | Complexity | Cost | Deploy | Upkeep | Solo | Decision and reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Wikibooks Cookbook (display) | Clear licence, full steps, wide cuisines. Small | Low to medium: parse lines, clean leftovers | Pack under 10 MB (ESTIMATE, ADR) | Easy | Snapshot is dated 2024-07-31 | Excellent | **Adopt for display** |
| Open Recipes, statistics only | Large. Has servings (0.955) and times (0.68 to 0.75) | Medium | Kaggle-scale (62 s on this machine for the fitness script) | n/a: only derived statistics ship | Medium | Good | **Adopt for statistics.** Do not ship its titles, ingredient text, descriptions or images |
| Open Recipes as a display "more ideas" pack | Many more recipes; no steps | Medium | Larger pack | Medium | Many links dead: 33 of 150 sampled returned success (403: 19, 404: 42, other: 56). Tasty Kitchen, the largest source, gave 0 of 56 | Fair | **Defer.** Needs its own live-link test. The licence covers the database, not each publisher's rights |
| RecipeNLG subset with a source link | Largest. Has steps | Low | Large | Easy | Terms: non-commercial research and education; no public grant found. 31.9% of kept rows are tagged `Recipes1M`, which has separate terms (MEASURED, review) | Poor | **Reject for the public app.** Revisit only with written permission from the authors |
| tastyR | About 14k recipes with ratings; CC0 label | Low | Small | Easy | Scraped from Allrecipes. The label cannot clear the source's rights | Poor | **Reject for display** |
| USDA MyPlate Kitchen | Clear attribution | High: no bulk download | n/a | n/a | Some recipes come from partners | Poor | **Reject for now** |
| A demo with no recipe text, linking to sources | Safe | Low | Tiny | Easy | Weak product | Good | **Not needed** now that Wikibooks is chosen |
| RecipeNLG `Gathered`-only filter (867,187 kept recipes) | Avoids the Recipes1M terms | Low | Large | Easy | Does not clear RecipeNLG terms | Poor | **Reject** |

### 7.3 Licence notes

- RecipeNLG terms: non-commercial research and educational use. The Hugging Face licence field says "unknown" (DOCUMENTED, <https://huggingface.co/datasets/recipe_nlg>). The official page did not render for the report authors, so they read the terms through dataset card excerpts (<https://recipenlg.cs.put.poznan.pl/>).
- Copyright and contract are two questions. US guidance says bare ingredient lists are not protected, but descriptive text can be (<https://www.copyright.gov/circs/circ33.pdf>).
- Open Recipes statistics are a derived use of scraped data under a database licence. ADR 0003 calls this "an engineering judgement, not legal advice". This document makes the same statement.
- CC BY-SA 4.0 requires credit, a link to the licence, a note of changes, and the same licence on adaptations. The pack must follow it ([PRODUCT_VISION.md](PRODUCT_VISION.md) section 13).
- USDA FoodData Central is public domain with a request for citation (<https://fdc.nal.usda.gov/>). The statement comes from the repo's `DATASET.md`, not from an independent check.

### 7.4 Wikibooks data checks (MEASURED, this task)

These checks were made on 2026-10-30. Details are in [PRODUCT_VISION.md](PRODUCT_VISION.md) section 6.

- 3,895 pages, 3,800 with at least 2 ingredient list items.
- 0 image references. 0 damaged-character lines. 0 duplicate titles.
- Median 8 ingredient lines. Median about 500 characters of steps.
- 4 recipes with no steps. 75 with under 80 characters.
- 57 lines with `Template:` leftovers.
- 95 pages fall outside the 3,800. Some are real recipes with unusual headings.
- Category on 97% of pages. Difficulty (1 to 4) on 97%. Servings and time on about 33%, as free text.
- `evidence/wb_quality.py` printed 3,800 recipes, 33,546 ingredient lines and a 0.037 share of recipes with under 80 characters of steps (this task reran it).

## 8. The ingredient table and Epicure

The synthesis proposes one curated table. Each row holds the display name, synonyms, functional role, three-state diet flags, a USDA food id and a visual label. One table could serve autocomplete, retrieval, substitution gating, diet filters, nutrition matching and vision mapping (HYPOTHESIS, synthesis 4.2).

### 8.1 Evidence for the need

On the old RecipeNLG vocabulary (30,481 entries, MEASURED, report 06):

- About 35% of a 200-entry sample was junk or brand names.
- "garlic" and "clove garlic" were separate entries, so a pantry with garlic missed about 45% to 47% of garlic recipes (two sample measurements).
- Scallion and green onion, eggplant and aubergine were separate.
- The singulariser made "asparagu".
- Mention coverage hides recipe coverage:

  | Top N entries | Mentions covered | Recipes fully covered |
  | --- | --- | --- |
  | 300 | 63.6% | 5.4% |
  | 500 | 70.8% | 9.1% |
  | 1,000 | 79.8% | 19.3% |
  | 3,000 | 90.9% | 48.4% |

These figures are for the old corpus. On Wikibooks, before collapsing, a review measured coverage with a top-N proxy (MEASURED in review; not a built table):

| Top N entries | Mentions covered | Recipes fully covered |
| --- | --- | --- |
| 500 | 75.4% | 16.1% |
| 1,000 | 84.2% | 30.5% |
| 2,000 | 91.6% | 53.4% |

Do not shrink the pack to fully read recipes before you measure the cost. Collapse of size, container and preparation words may lower the needed size. The M3 plan may need about 1,700 rows or more.

### 8.2 Options

| Option | Gain | Complexity | Cost | Deploy | Upkeep | Solo | Decision and reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Rule-based collapse plus a first table of about 500 rows, built from Wikibooks lines, then measure | High (HYPOTHESIS) | Medium: days of manual review | Tiny | Easy | Medium | Good | **Adopt** (synthesis 4.2). Do not start with 3,000 hand-written rows |
| Reuse the old `vocab.json` | Poor | Low | n/a | n/a | n/a | n/a | **Reject.** It comes from RecipeNLG, so it cannot ship (ADR 0003) |
| Epicure's released 1,790-ingredient vocabulary as a seed or cross-check | Might save days. Unknown fit to Wikibooks lines | Low to inspect | Small CSVs | Easy | Easy | Good | **Look at it, time-boxed.** Do not adopt until the data licence is verified. The paper is CC BY 4.0 (secondary note in report 02). The data licence is **unverified**. Require source-corpus provenance: the abstract does not name its 11 sources |
| FoodOn or Wikidata as the base | Taxonomy for gating | High | Tens of MB | Precompute | Medium | Poor | **Optional later.** FoodOn licence unverified |
| USDA FDC foods for names | Public domain; names are not shopper friendly | Medium | Small | Easy | Easy | Fair | **Use for the USDA id column only** |
| Mine names from Open Recipes statistics | Frequency data | Medium | Kaggle | n/a | Medium | Fair | **Use for frequency ranking.** Ship only derived counts |

Epicure facts (DOCUMENTED, report 02): arXiv 2605.22391, May 2026; Chem, Cooc and Core metapath2vec embeddings on 4.14M multilingual recipes; 1,790 canonical ingredients; the abstract has no substitution evaluation. Source: <https://arxiv.org/abs/2605.22391>.

## 9. Nutrition (out of the first release)

Source: [report 06 section 5](research/2026-10-stage1/06_data_nutrition.md), ADR 0005.

- Matches on staples were wrong: "oil" matched bearded seal oil; "sugar" matched brown sugar; "whole milk" matched mozzarella; "egg" matched frozen scrambled egg (MEASURED, report 06).
- "3 cloves garlic" became 408 g. A mangled fraction ("13 cup soy sauce", from 1/3) produced a 43,568 kcal recipe (MEASURED, report 06). The fraction bug came from the Recipes1M-tagged rows; the Wikibooks data has Unicode fractions instead, and the effect of that on parsing is untested.
- Random-sample coverage is 0.710 (match) and 0.439 (mass), against 0.800 and 0.487 published from the first 5,000 rows. The review reproduced 0.700 and 0.443 (MEASURED).
- The Wikibooks snapshot has only 24 text lines under headings named "nutrition" (MEASURED, this task). That is too little to serve as a source. Quality is unchecked.

| Option | Gain | Complexity | Cost | Deploy | Upkeep | Solo | Decision and reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| No nutrition in v1 | Avoids false numbers | None | None | None | None | Excellent | **Adopt** (ADR 0005) |
| Existing USDA matcher | Wrong on staples | Done | Medium | Easy | Hard | Poor | **Reject** |
| Rebuilt matcher with curated defaults, clove and egg portions, and a coverage display | Medium | High | Low | Easy | Medium | Fair | **Later.** Enter only after a check on 500 USDA FNDDS recipes passes the target in `docs/DEVELOPMENT_PLAN.md` |
| Show ranges only when mass coverage is 0.8 or more | Honest display | Medium | Low | Easy | Medium | Fair | **Later** |
| Per-serving numbers | Common request | n/a | n/a | n/a | n/a | n/a | **Reject.** Servings are free text on 33% of pages |

Validation sources: USDA FNDDS recipes (public domain) and Nutrition5k (CC BY 4.0), both from report 06 (not re-fetched).

## 10. Evaluation, claims and engineering

Source: [report 07](research/2026-10-stage1/07_engineering_claims.md).

- The claims ledger checked 43 claims: 14 false, 11 unsupported, 11 stale, 7 true (MEASURED, report 07).
- 246 tests pass with 92% coverage. A fresh clone fails the README quickstart with 7 collection errors (MEASURED, report 07; not rechecked by the review).
- No pantry-to-answer function exists. `pantrychef cook` uses the baseline and not the parser, vocabulary or ranker (MEASURED, report 07).
- The "production" bundle holds a 1,000-recipe fixture index and is 9 commits old (MEASURED, report 05 and review).
- `pip-audit` flagged 16 packages. Reachability was not assessed (report 07).

| Option | Gain | Complexity | Cost | Deploy | Upkeep | Solo | Decision and reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Retract headlines and rebuild evaluation | High: all claims survive review | Medium | Low | n/a | Easy | Good | **Adopt** (ADR 0005) |
| Keep headlines with caveats | Low | Low | n/a | n/a | n/a | n/a | **Reject** (ADR 0005 option 1) |
| Hand-labelled gold sets: 300 parser lines, 100 judged pantries, 150 swap pairs | High | Medium: hours of labelling | Tiny | n/a | Easy | Good | **Adopt** |
| Results JSON with commit, dirty flag, config, corpus fingerprint, metrics | High | Low | Tiny | n/a | Easy | Excellent | **Adopt** |
| Tag-and-cut of unused code (Hydra, MLflow, placeholders, `rrf`, `SvdContextModel`) | Cleaner repo | Low | Tiny | n/a | Easy | Excellent | **Adopt** (report 07 section 3) |

Interview framing: a metric that looked excellent, the test that falsified it, and the better protocol. This replaces the retired headline. It does not replace a built deliverable (synthesis 4.3).

## 11. Recommended sequence

From the synthesis (section 5), adjusted for ADR 0003 and 0004:

1. Reset the repo: merge PR #5, fix the quickstart, make the README honest.
2. Build a thin slice: one `answer()` function, a static page, the Wikibooks pack, an IDF score.
3. Run baseline evaluation on the new data.
4. Build the pack cleaner and the first ingredient table of about 500 rows. Measure coverage on Wikibooks lines.
5. Run the retrieval and ranker test against hand-judged pantries.
6. Build the trust layer: three-state diet flags, curated swaps, explanations.
7. Build the full app. Deploy on GitHub Pages with parity tests.
8. Vision beta only if its own test passes.

## 12. Findings that contradict or refine ADR 0003 and the synthesis

Found while grounding the product in the real Wikibooks snapshot. None changes a decision. Some change a number or a sentence.

1. **Synthesis 4.1 says the data has no cooking time, servings or cuisine, and that filters on those fields do not exist.** That held for RecipeNLG. The Wikibooks pack has a category (mixing cuisine and dish type), a 1 to 4 difficulty on 97% of pages, and servings and time on about 33% as free text. No filter should rely on servings or time. A category filter is not in v1. It comes later, after cleaning.
2. **ADR 0003 says categories "allow simple filters".** Partly true. 118 pages have no category, 33 point to a missing page, and cuisine and dish type share one field.
3. **An earlier draft of ADR 0003 listed RecipeNLG "5 to 57k scale".** The evidence file shows 4,448 to 57,392 for one item away (scaled from a 1-in-8 sample). ADR 0003 now states this range. The "5" was wrong.
4. **"3,800 with an ingredient list" depends on the rule.** A strict rule (two or more list items under "Ingredients") gives 3,800. 95 pages fall outside it. Some are real recipes with ingredients under other headings or in numbered lists (for example "Apple Crumb Cake"). Others have no ingredient list ("Apple Potato Bread"). The true usable count may be higher. It is unmeasured.
5. **"Instructions 98.3%" counts presence, not quality.** 4 recipes have none. 75 have under 80 characters.
6. **Page leftovers.** 57 lines hold `Template:Nutritionsummary` or navigation text. ADR 0003 does not mention this. The pack build must clean it.
7. **Servings and time are messy.** ADR 0003 says they "exist on only about 33%". That is right. Of those, 428 of 1,296 servings and 223 of 1,319 time strings are not simple (examples: "~75", "Prep: 20 minutesBaking: 60 minutes").
8. **Cuisine mix.** ADR 0003 says "wide cuisine range". True. The mix is also uneven: Nigerian is the 5th largest category (139 pages), and many of those pages list ingredients with no amounts.
9. **Not rechecked in this task:** the vocabulary hit rate 0.985 (not parse accuracy), the 98.3% figure and the pantry yield counts for Wikibooks. They come from the evidence file. The "make now 1 to 73" and "one item away 69 to 330" values match the file.
10. **Not rechecked:** the review also did not recheck the fresh-clone failure, the full-corpus results, the hand labels behind the 36% and 35% figures, or most hosting facts.

## 13. Unverified items and gaps

### 13.1 Gaps in coverage

- No sourced detail on Samsung Food, Plant Jammer, Tasty, Allrecipes or Spoonacular.
- MyFridgeFood, Cooklist and Paprika rest on secondary summaries.
- No measured dropout rate for pantry entry.
- No user research of any kind. Target groups are reasoned (HYPOTHESIS).
- No fridge-clutter benchmark for any vision approach.
- Browser latency and memory are unmeasured, for the engine and for vision.
- The coverage of a built ingredient table on Wikibooks lines is unmeasured. Only a top-N proxy exists (section 8.1).
- Quality of mined substitutions on Open Recipes statistics (168k unique recipes) is unknown (ADR 0003).
- The post-parser-fix full-corpus recommender result does not exist.

### 13.2 Unverified facts

- Kaggle quotas (about 30 GPU hours a week, 9 to 12 hour sessions, about 29 GB RAM, 4 CPUs) come from secondary sources. The official page returned no values. The credential format is unverified.
- Fly.io, Koyeb, Oracle Always Free and Vercel or Netlify limits come from secondary sources or recall.
- Range request and CORS behaviour on GitHub Pages and Hugging Face are untested. Test with `curl -I -H "Range: bytes=0-99" URL`.
- Licences not verified: Epicure data, FoodOn, FlavorGraph repo, Roboflow AICook (page returned 403), Open Food Facts, Freiburg Groceries, KTH Grocery Store, and several vision datasets marked ESTIMATE in report 03.
- The official RecipeNLG terms page did not render. The terms were read through dataset card excerpts.
- The MediaWiki page-history link pattern (`?action=history`) used for author credit is unverified. Verify it in M1 before it ships.
- The font and icon licences named in PRODUCT_VISION section 14 need a check at build time.
- The "pip-audit" result (16 packages) has no reachability analysis.
- All licence conclusions are an engineering reading, not legal advice.

### 13.3 Reviewer caveats

- The 39/5/4/2 hand check of substitution results had one rater and was not blind.
- The extra-item pantry tests have no paired intervals, and the frequency sampler favours IDF.
- The 7.7 times full-corpus lead is from before the parser fix.

## 14. Rejected options at a glance

| Rejected | Reason in one line |
| --- | --- |
| RecipeNLG text in the public app | No public grant; mixed Recipes1M terms (ADR 0003) |
| tastyR, USDA MyPlate Kitchen | Source rights not cleared; no bulk download |
| Open Recipes text in the app | Licence covers the database, not each publisher; links often dead |
| Learned reranker as a headline | A rule scores almost the same; must win on hand-judged data first |
| Graded makeability labels | The label is a function of a feature |
| ANN, FAISS, two-tower, SPLADE | Exact set matching is lexical; no such code exists |
| LLM reranker, GNN | Project excludes LLMs; no interaction data |
| ONNX Runtime Web for trees | A small tree walker is enough |
| Current mined substitution model for users | Variants and complements; 123 MB; 0.13 s per query |
| GISMo, Recipe1MSubs, lookup-table training data | Non-commercial licences |
| Old dietary tagger | Unsafe and judges 18.7% of the vocabulary |
| Allergen claims | High harm if wrong (ADR 0005) |
| Ultralytics YOLO | AGPL |
| Open-vocabulary detectors, server-side vision, cloud vision API | Need a server or break the no-upload promise |
| Docker or Gradio Spaces, Render, Fly, Koyeb, Railway, Workers, Oracle | Paid, sleeping, tiny or unreliable |
| Pyodide, SQLite-WASM, DuckDB-WASM | Too heavy for a pack of this size |
| DVC, MLflow, Hydra | A manifest and results JSON do the job |
| Accounts, barcode scanning, quantity tracking | Server cost, instability, entry burden |
| Nutrition in v1 | Wrong matches and masses (ADR 0005) |
| Images from Wikimedia or stock photos | Per-file licences; privacy leak by hotlinking; none in the snapshot |
| "Graph beats food2vec" | Interval includes zero; tuned on the same queries (ADR 0005) |
