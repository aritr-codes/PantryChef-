# 02 - Substitution audit (PantryChef, branch fix/parser-doubled-token-coverage)

Method: ran the shipped bundle (`models/substitution/substitution_bundle.zip`, extracted to `research/bundle/`, ~340 MB free RAM, fit) through the real code path (rank_blend alpha=0.5, overlap_shrink=100, lam=0.5). Scripts in this folder: `run_subs.py`, `run_eval_boot.py`, `diet_attack.py`, `diet_pipeline.py`, `time_nb.py`; outputs `subs_out.json`, `perquery.json`.
Reproduction check: my per-query rerun gives emb 0.2587 / graph 0.2889 / hybrid 0.3173 MRR and R@10 0.4815/0.4659/0.5319, identical to `p2_eval_fixed.log`. The numbers are real; the question is what they mean.

## Headline verdict
The model is a decent finder of ingredient-name variants and related items, not a substitution engine. The eval gold is noisy and mined from the training corpus; the 100% dietary figure is tautological; the dietary guardrail returns an empty list for the most common vegan/dairy-free requests and lets wheat products through "gluten-free". Not ready to be a public flagship as-is.

## 1. Eval validity

**F1.1 Gold is mined from the training corpus and is itself noisy. High. VERIFIED.**
`pantrychef/eval/subs_gold.py:mine_pairs` groups recipes by title and treats two recipes whose ingredient sets differ by exactly one element as a swap. The 116 pairs (`data/eval/subs_gold.csv`, 82 query keys) were mined from the 28k sample (EVALUATION.md:44-46); the scored model is trained on 1.27M recipes (bundle metadata) that include those recipes. Circular twice: (a) the label evidence (x and y share context, never co-occur) is exactly what SPPMI/word2vec encode; (b) the recipes are in training. Label quality is poor (I read the CSV): `butter->salt`, `butter->unbaked pie shell`, `cinnamon->lemon juice`, `chocolate chip->vanilla`, `baking powder->soda`, `cornstarch->cream of tartar`. 42/116 pairs (36%) are pure name variants (`all purpose flour->flour`, `egg yolk->egg`, `bisquick mix->bisquick baking mix`) that the model finds trivially.

**F1.2 Graph vs embedding is not distinguishable at n=82. High. VERIFIED** (paired bootstrap, 10k resamples, per-query RR from the real bundle):
| contrast | dMRR | 95% CI | P(d<=0) |
|---|---|---|---|
| graph - emb | +0.030 | [-0.048, +0.111] | 0.23 |
| hybrid - emb | +0.059 | [+0.006, +0.112] | 0.014 |
| hybrid - graph | +0.028 | [-0.028, +0.087] | 0.17 |

Only hybrid>emb clears zero, barely, uncorrected, on a gold that was also used for tuning (F1.3). "Flagship graph beats food2vec" is unsupported. Contradicting docs: MODEL_CARD_substitution.md "Core hypothesis result: CONFIRMED" (graph 0.339 vs 0.290 on the 28k sample, hybrid worse than graph there; full-corpus ordering is hybrid>graph>emb) and EVALUATION.md "Flagship arm-ordering holds".

**F1.3 Hyperparameter tuned on the reporting gold. High. VERIFIED.**
`data/processed/diag_graph_fix_grid.log`: grid over overlap-floor / overlap-shrink / df-shrink / sqrt-overlap on "n=82", best picked by "mrr+r@10", marked `<-- beats emb (both)`. Same 82 queries are the reported eval. Graph-only was 0.155 < emb 0.169 before shrinkage in that same log; beta=100 was selected because it flipped the ordering. Alpha=0.5 and lam=0.5 are untuned defaults (lower concern, but never swept held-out). Fix: split queries into dev/test, or re-mine from a disjoint slice.

**F1.4 GISMo row is pre-parser-fix. High. VERIFIED.**
`eval_gismo.log` is stamped 2026-06-01 22:35 and uses the 06-01 `word2vec.kv`; the mined row moved emb 0.169 -> 0.259 after the fix (`eval_full_fixed.log` vs `p2_eval_fixed.log`). The GISMo row (0.057/0.076/0.083) was never re-run (EVALUATION.md:119-121 still says 2026-06-01), so "ordering holds on an independent benchmark" (EVALUATION.md:124) refers to the stale model.

**F1.5 Protocol mismatch with GISMo. High. VERIFIED** (arXiv 2302.07960 Table 1, read from the HTML version).
GISMo test MRR = 0.3151 (Hit@1 .2056, @3 .3589, @10 .5427), Recipe1MSubs test 10,747 samples, ranking the whole 6,653-ingredient vocabulary minus the source, per (recipe, source) query, with the recipe's ingredient set as context, all valid targets for that recipe accepted. Ours: context-free (`recipe` arg unused; `context_weight=0`), 30,481-token noisy vocab, queries aggregated per source name with a union of targets across all recipes, 21.5% of pairs dropped for no coverage (covered-subset scoring; the pessimistic number would be ~0.065), nothing trained on their data. The paper's trivial LT+Freq baseline (lookup table + frequency) gets MRR 0.2766; Food2Vec gets 0.0965. So ~0.08 here is Food2Vec-class and a table of observed swaps is ~3x better than anything we have. The docs caveat this (EVALUATION.md:112-115,128-131) but README.md:20,63 ("benchmarked against published research") and the flagship framing imply more comparability than exists. GISMo code/data are CC-BY-NC: fine for portfolio eval, not for shipping derived artifacts in a public product.

**F1.6 Other doc/log contradictions. Medium. VERIFIED.**
- Eval logs print `tag-coverage=1.000`; current `DietTagger.coverage(vocab)` is **0.187** (5,701/30,481 judged) because `_is_known` ignores the vocab (`dietary/ontology.py:139-146`). The stale 1.000 persists in MODEL_CARD:120 and EVALUATION.md.
- MODEL_CARD:18,76,146 says tags are "substring-matched"; code is token-matched.
- `dietary_labels.json` (312 B) is `data/nutrition/dietary_labels.json`, a 10-item accuracy gold used only by `nutrition_main.py:59`. It is not the guardrail's data; the guardrail is the hard-coded `_LEAF_WORDS`/`_CURATED` dicts.

## 2. Real-world quality (50 common ingredients, hybrid, hand-judged; full lists in `subs_out.json`)
Top-1 of hybrid, 50 queries:
| class | n | examples |
|---|---|---|
| Good real substitute | 12 | cream->whipping cream; honey->maple syrup; vegetable oil->canola oil; ground beef->lean ground beef / ground turkey; heavy cream->heavy whipping cream; chicken broth->chicken stock; cocoa powder->unsweetened cocoa; tomato paste->tomato sauce; mayonnaise->miracle whip (rank 3) |
| Spelling/modifier variant of the same thing (true, useless) | 24 | flour->all purpose flour; yeast->active dry yeast; garlic->clove garlic; onion->onion small; shallot->shallot small; sour cream->container sour cream; cilantro->fresh cilantro |
| Plausible with caveat | 6 | milk->evaporated milk; baking soda->baking powder (needs ~3x amount, and acid); butter->butter margarine |
| Wrong (complement/junk) | 8 | egg->"large"; buttermilk->granulated sugar; cornstarch->granulated sugar; bacon->garlic salt; vanilla extract->powdered sugar; salt->whole wheat flour |

Across top-5 lists: powdered sugar appears for butter, milk, egg, buttermilk, vanilla; margarine for milk; bay leaf for potato and parsley.

**Failure patterns (VERIFIED):**
1. Variants dominate. The vocab keeps modifiers/quantities ("onion small", "container sour cream", "lemon juice of", "bacon bacon", "onion onion", "egg egg", "baking power"). Upstream normalizer issue; the model learns it faithfully.
2. Complements from the graph arm (powdered sugar, nut, vanilla, bay leaf, celery, garlic salt). The lam penalty only subtracts pairs that co-occur in the same recipe, not same-cuisine companions.
3. Functional misses: buttermilk->milk+acid/yogurt, egg->applesauce/flax, baking powder<->soda ratio, cornstarch->arrowroot, lime->lemon juice, shallot->onion, soy sauce->tamari/coconut aminos (tamari is not in the vocab) are absent.
4. No quantity/ratio and no recipe context.
5. Safety: omnivore output is not harmful; baking soda<->powder 1:1 ruins bakes. The unsafe class is dietary (section 3).

## 3. Dietary guardrail

**F3.1 "100% validity" is tautological. Critical for any public claim. VERIFIED.**
`substitution_eval.dietary_validity` calls `tagger.is_valid` on outputs already filtered by `_mask` using the same `is_valid`; it cannot fall below 100%. `Substitute.dietary_valid=True` is hard-coded in `substitute.py` (final return). The 45 subs come from 9 hand-picked queries in `curated_subs.json`, vegan only, under a mask that rejects 81% of the vocab as unjudged, so survivors are the easy ones.

**F3.2 Pipeline returns nothing for the main dietary asks. High. VERIFIED** (`diet_pipeline.py`, pool 20, k 5):
vegan butter 0; vegan milk 0; vegan egg 0; vegan cream 0; vegan sour cream 0; dairy_free heavy cream 1 (`egg yolk`); dairy_free butter 1 (`egg yolk`); GF bread crumb 0; GF pasta 1 (`parmesan`); GF soy sauce 1 (`fish sauce`); vegetarian chicken broth 1 (`cream of mushroom soup`). The mask filters the top-20 pool after ranking; the pool is never widened. It strips `coconut milk`, `almond milk`, `oat milk`, `coconut oil`, `margarine`, `tahini`, `nutritional yeast`, `tamari`, `rice` (unjudged or token false-positives), i.e. exactly the answers a vegan wants. GF `flour` returns `cream light`, `butter margarine`: "valid" but not flour substitutes.

**F3.3 Concrete pass/fail on adversarial strings (`diet_attack.py`, `diet_pipeline.py`):**
- Vegan false positives (wrongly rejected): `butter beans`, `peanut butter`, `cocoa butter`, `almond milk`, `coconut milk`, `oat milk`, `coconut cream`, `cream of tartar` (tokens butter/milk/cream). Correct rejections: `worcestershire sauce` (curated fish), `gelatin`, `chicken stock`, `fish sauce`, `honey`, `buttermilk`. `eggplant`, `tofu`, `olive oil` correctly pass. `mayonnaise`, `sugar`, `margarine`, `shortening` rejected only because unjudged.
- **False negatives (unsafe items allowed), gluten_free: `cream of mushroom soup`, `cream of chicken soup` (wheat-thickened), `egg noodle`, `egg roll wrapper`, `chicken noodle soup` (wheat) all PASS.** `oat` passes by curated override; `worcestershire sauce` passes GF (malt vinegar versions exist). Cause: tagging by the first/any known leaf token, so composite foods are classified by the wrong word. This is a coeliac safety defect.
- `rice`, `beer`, `tamari`, `malt vinegar`, `panko` are all rejected under GF merely as unjudged (rice rejected is absurd).
- No allergen model: `_DIETS` has 4 entries (vegan, vegetarian, gluten_free, dairy_free). No nut/peanut/shellfish/soy/egg/sesame. Nuts are unjudged so they are excluded from constrained diets, but nothing screens `peanut butter` for a nut-free user. MODEL_CARD:17 "Not intended for allergen safety" is right; the UI must say it.

Fix: (a) tri-state curated table of ~300-500 canonical heads (safe / unsafe / unknown), with any-token-unsafe semantics for composites, and "unknown" shown to the user instead of silently dropped; (b) replace the metric with validity against an independent label set (Open Food Facts / FoodOn flags) or drop it; (c) widen the pool until k survivors, else say "no diet-safe substitute known"; (d) label as preference filter, not allergen/medical. 1-2 days; worth it, this is the product-defining feature.

## 4. Explainability
Not available today. Medium. VERIFIED. `Substitute` has only `ingredient, score, dietary_valid, arm` (`common/types.py:23-29`). Data that already exists: `graph_cooccur.npz` counts, normalized SPPMI rows, and the overlap count `ov` computed (then discarded) in `ContextGraph.neighbors` (`graph.py:148-150`), plus the rank disagreement in `rank_blend`. Cheap explanation fields: (1) top shared contexts = argmax of elementwise `mn[i] * mn[j]` ("both used with vanilla, flour, eggs"); (2) shared-context count `ov` as confidence (shrink factor `ov/(ov+100)`); (3) `cooccur[i,j]` = "appear together in N recipes" (high = complement, low = true substitute); (4) emb/graph agreement. About 100 LOC. Caveat: "used in similar recipes" is not functional equivalence.

## 5. Alternatives
| approach | expected gain | data / compute / size | licence | solo suitability |
|---|---|---|---|---|
| Hand-curated functional table (~200-300 rows: ingredient, role, 2-5 substitutes with ratios, caveats, diet flags) | Highest for users: fixes buttermilk, egg, baking powder/soda, cornstarch, vegan/GF swaps, ratios | 2-3 days manual; KBs | own (Wikibooks Cookbook:Substitutions is CC BY-SA, use as a source) | Excellent |
| GISMo (arXiv 2302.07960) | MRR 0.315 vs ours ~0.08 on its protocol; recipe-context aware | Recipe1M registration + GPU hours | CC-BY-NC | Eval only |
| LT+Freq / mined swap table | GISMo's own baseline: MRR 0.277. Counts of observed swaps | Recipe1MSubs or reviewer-comment mining; MBs | Recipe1MSubs NC | Good as prior/eval |
| Food2Vec / FoodKG | ~ our emb arm (0.0965 on GISMo protocol) | exists | n/a | no gain |
| FlavorGraph (doi 10.1038/s41598-020-79422-8) | flavour-compound pairing, not substitution | pretrained embeddings, small | not verified | optional re-ranker |
| Epicure (arXiv 2605.22391, May 2026) | Chem/Cooc/Core metapath2vec on 4.14M multilingual recipes, 1,790 canonical ingredients; abstract has no substitution eval | released CSVs, small | paper CC BY 4.0; data licence not verified | Cheap to try; its 1,790-head vocab is the clean vocab we lack |
| FoodOn / Wikidata / USDA FDC categories | class taxonomy to gate candidates (kill complements, diet flags) | OWL tens of MB offline; `data/processed/usda.json` already in repo | FoodOn CC BY (from memory, unverified); USDA public domain | Good for gating |
| Functional-role constraint (leavener/fat/binder/acid/sweetener/thickener) | Candidate must share role with query; removes complements by construction | tag ~300 heads by hand | own | Best ROI on top of mined model |

Would curated table + mined model beat the current model for users? Almost certainly yes, **HYPOTHESIS**. Evidence: current top-1 is a real substitute ~12/50, a spelling variant 24/50, wrong 8/50, and the curated table directly covers all failed functional cases. Experiment E1: 30-50 queries x {current hybrid, curated only, curated then role-filtered mined fallback}, blind-rated by 2 raters ("would I use this"); success >=70% vs current ~25-35%; no compute.
Recommended blend: exact curated hit (with ratio and caveat) -> else mined candidates restricted to same role/class and collapsed to head noun -> else "no confident substitute". Never display the query's own variants.

**F5.1 Serving artifact. VERIFIED arithmetic.** Bundle is 123 MB (131 MB raw), needs scipy+gensim and several hundred MB RAM (matrix has 10.1M nnz). Top-10 per vocab item = 30,481 x 10 x (4 B id + 2 B fp16) = 1.8 MB (+0.6 MB vocab; ~1 MB gzipped JSON). Top ~5k common heads only: ~0.3 MB. Replace with a 1-2 MB JSON/SQLite incl. explanation fields; gensim/scipy drop out of the web image.

## 6. Performance
`ContextGraph.neighbors` (`graph.py:120-157`): two sparse mat-vec products over a 30,481 x 30,481 matrix (10.1M nnz) + dense `toarray()` + argsort of 30k per call. **Measured 0.13 s/query**, single thread (`time_nb.py`, bundle). One user fine, concurrent users not. Batched 200-query matmul took 0.83 s (~4 ms/query for the similarity step alone). Precompute: naive loop 30,481 x 0.13 s ~ 66 min one thread; top-5k heads ~11 min (derived, VERIFIED arithmetic); blocked matmul + argpartition likely <10 min (HYPOTHESIS, needs ~1 GB for dense 200x30k blocks). Viable and removes the slow path from serving.
Kaggle experiment K1: CPU, 8-16 GB RAM, ~15-30 min. Inputs: bundle zip, `subs_gold.csv`, `subs_gold_gismo.csv` (166 KB). Outputs: `subs_topk.json.gz`, per-query RR CSV with bootstrap CIs, GISMo re-run on the post-fix model with pessimistic scoring.

## Ranked recommendations
1. Fix/relabel the dietary guardrail (F3.1-3.3): tri-state, any-token-unsafe, fix GF wheat composites, drop tautological metric, widen pool, no allergen claims. Critical. 1-2 days.
2. Ship a curated functional substitution table first (roles, ratios, caveats, diet flags), mined model as role-constrained fallback. Biggest user gain. 2-3 days.
3. Collapse substitution vocab to canonical heads upstream ("onion small", "container sour cream", "bacon bacon"). Removes about half the visible junk. 1-2 days.
4. Rebuild the eval: dev/test split of queries, tune beta/alpha on dev only, paired bootstrap in the report, hand-judged 150-pair functional gold, GISMo re-run post-fix + LT+Freq baseline beside it. Retire "graph beats food2vec" unless a CI excludes zero. 2 days.
5. Serve precomputed top-k + explanations (K1) instead of the 123 MB bundle. 1 day.
6. Add explanation fields (shared contexts, overlap, co-occurrence count). 0.5 day.
7. Optional: test Epicure vocab/embeddings (E2) as clean vocab / re-ranker.
8. Update stale docs: MODEL_CARD results are the 28k sample and say CONFIRMED; tag-coverage 1.000; "substring" tags; EVALUATION GISMo row date.

## Sources
- GISMo paper (Table 1, protocol): https://arxiv.org/html/2302.07960 (pdf https://arxiv.org/pdf/2302.07960)
- GISMo repo, CC-BY-NC majority licence: https://github.com/facebookresearch/gismo
- Epicure: https://arxiv.org/abs/2605.22391
- FlavorGraph: https://doi.org/10.1038/s41598-020-79422-8
- Not verified this run: FoodOn licence, FlavorGraph repo licence, Epicure data licence.
