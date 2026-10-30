# PantryChef: technical audit

Date: 2026-10-30. Audited commit: `d49f02c`. Companion: [PROJECT_STATE.md](PROJECT_STATE.md).
This document merges the [Stage 1 synthesis](research/2026-10-stage1/README.md), its [review](research/2026-10-stage1/REVIEW.md) and [reports 01 to 07](research/2026-10-stage1/). Where they differ, the synthesis and review win. I re-checked each file and line reference below against the repo.

Labels: VERIFIED (code, log or run shows it). HYPOTHESIS (an experiment must confirm it). MEASURED and ESTIMATE as in PROJECT_STATE.

Terms: IDF = inverse document frequency (rare items weigh more). MRR = mean reciprocal rank. recall@10 = share of queries whose right answer is in the top 10. LambdaMART = a learned ranking model. ADR = architecture decision record. CI = continuous integration. ANN = approximate nearest neighbour search.

## 1. Verdict

The code is clean and well tested. The product does not exist yet, and the headline results do not hold up. A no-ML rule matches the learned ranker on the main benchmark. The graph-versus-embedding win is within noise. The dietary check cannot fail. The nutrition output is wrong for staples. The README describes planned features as built. ADRs 0003 to 0005 already choose a sound reset. This audit asks that the reset be carried out and the claims be corrected.

## 2. Method and limits

- Method: five research agents read code and ran small scripts. A coordinator reconciled them. An independent reviewer reran key numbers and returned GO WITH FIXES. I rechecked line references and read the repo state. Scripts are in `docs/research/2026-10-stage1/evidence/`. Some contain absolute paths from the audit session.
- Limits: no full-corpus experiment ran. Vision accuracy, browser latency and quality signals are unmeasured. Hosting facts are partly from secondary sources. The RecipeNLG terms page did not render, so the terms come from dataset card excerpts. Licence points are an engineering reading, not legal advice. `pip-audit` flagged 16 packages and nobody checked reachability.
- The reviewer did not recheck: the fresh-clone failure, the 246 test count, any full-corpus result, browser latency, the 36% and 35% hand labels, most hosting facts.
- I did not run the CLI or any training script: free RAM is about 1 GB.
- Rater caveat: the 39/5/4/2 substitution check (finding H1) had one rater and was not blind.

## 3. Critical

### C1. The recommender headline measures the query generator. VERIFIED
- What: `make_query` removes 30% of one recipe's ingredients and uses the rest as the pantry (`pantrychef/recommender/query_sim.py:47`). The pantry is always a subset of the target recipe. It has no unrelated items. A real pantry is the reverse.
- Evidence: a no-ML rule `match_frac_pantry*1000 + coverage` scored recall@10 0.975 on 50k recipes and 1,404 queries, against 0.740 for overlap. LambdaMART scored 0.980 on a different set of 9,354 queries. On 20k recipes and 1,500 queries the MRR@10 values were 0.544 (overlap), 0.952 (rule) and 0.958 (model). The feature `match_frac_pantry` is at `pantrychef/recommender/features.py:57`. In the bundle its first split has gain 10138 and the next splits are under 17.
- With extra items in the pantry (MRR@10, 20k spread sample; corrected on 2026-10-30, these were mislabelled as recall@10; the IDF rule output is not committed): at 10 extras, overlap 0.322, fraction rule 0.853, model trained the current way 0.601, model retrained on extras 0.943, IDF cosine rule 0.958. At 20 extras the figures are 0.113, 0.693, 0.361, 0.830 and 0.914. At 3 extras the retrained model scored 0.985 and the IDF rule 0.980. No paired intervals exist, and frequency-sampled extras favour IDF.
- Why it matters: the "learned reranker crushes baseline" claim (`docs/EVALUATION.md:163`) shows how well a rule can read the generator. It is not evidence of ranking skill on real pantries.
- Fix: keep recovery as a sanity test. Add the IDF rule to the leaderboard. Add pantries with distractors using more than one sampler. Build a 100-pantry hand-judged set with bootstrap intervals.
- Worth the cost: yes. About an hour for the rule baseline. It becomes the best interview story.
- Under ADR 0005 this is the accepted plan.

### C2. The dietary filter is unsafe and its metric cannot fail. VERIFIED
- What: "100% dietary validity" re-checks results with the same tagger that filtered them (`pantrychef/eval/substitution_main.py:80` builds the tagger; `:114-115` validates with a new copy of it). The check cannot find a tagger error.
- Direct checks at this commit: egg noodle, cream of mushroom soup and chicken noodle soup pass "gluten free" (wrong, they contain wheat). Peanut butter, coconut milk and salt are blocked as "vegan" (wrong). Rice is blocked as gluten free. Onion is blocked as vegetarian. The tagger judges 18.7% of the vocabulary and blocks the rest. Used as a vegan recipe filter it would keep about 0.01% of sampled recipes (0 of 4,978).
- Why it matters: a wrong "safe" label on gluten is a health risk. A claim of "valid" invites trust it has not earned.
- Fix: three states (safe, unsafe, unknown) from a curated ingredient table. Evaluate against hand labels. Never claim allergen safety. ADR 0005 already requires this wording.
- Worth the cost: yes.

### C3. Nutrition output is not fit to show. VERIFIED
- What: the USDA matcher takes the first hit. `oil` matches bearded seal oil. `sugar` matches brown sugar. `whole milk` matches mozzarella. `egg` matches frozen scrambled egg. "3 cloves garlic" becomes 408 g. A mangled line "13 cup soy sauce" (from 1/3) gave a recipe of 43,568 kcal. Random-sample coverage is 0.710 match and 0.439 mass. The published 0.800 and 0.487 use the first 5,000 rows only.
- Fix: keep nutrition out of release 1 (ADR 0005). Validate against 500 USDA FNDDS recipes before it returns.
- Worth the cost: removing it is free. Rebuilding it is a later project.

### C4. Recipe data rights are unresolved. VERIFIED in part
- What: RecipeNLG terms allow non-commercial research and education. We found no grant of a public right. 26.4% of raw rows and 31.9% of kept rows are tagged `Recipes1M`, which has separate terms. Contract and copyright are separate questions. Bare ingredient lists are likely not protected. Instructions are the most exposed.
- Status: ADR 0003 settles the product path with Wikibooks and Open Recipes. The risk stays for any RecipeNLG-derived artifact. Open Recipes statistics are a derived use of scraped data under a database licence. That is a judgement, not legal advice.
- Fix: do not ship RecipeNLG text, vocabulary or trained artifacts. Optional: ask the RecipeNLG authors in writing.

## 4. High

### H1. The flagship graph model does not yet beat its baseline. VERIFIED
- What: graph MRR 0.289 against food2vec 0.259 at n=82. The difference is +0.030 with a 95% interval of -0.048 to +0.111. The shrinkage parameter was tuned on the same 82 queries. The gold set is mined from the training corpus, and 36% of its pairs are name variants. The "food2vec" baseline is the project's own word2vec. GISMo's model was never run. The GISMo table is pre-fix (`data/processed/eval_gismo.log`).
- Hand check of the top result for 50 common ingredients: 39 variants, 5 real substitutes, 4 usable with a caveat, 2 wrong (one rater, not blind). This replaces the 12/24/6/8 tally in report 02.
- Where: README rows at `README.md:20, 40, 54`.
- Fix: retire the headline (ADR 0005). Use a dev and test split. Report paired intervals.
- Worth the cost: yes.

### H2. No pantry-to-answer path exists. VERIFIED
- `cook` uses the P1 baseline and rebuilds the index per call (`pantrychef/cli.py:32-33`). The parser, vocabulary and ranker are not used.
- Cleaning drops directions and link. `load_recipes` blanks `ingredients_raw` (`pantrychef/data/store.py:47`). The schema keeps the fields (`pantrychef/data/schemas.py:11,13`).
- Pantry text goes through `canonicalize`, not the vocabulary. "2 cups all-purpose flour" becomes `2 cup all purpose flour`.
- `recipe_id` is sequential at clean time, so ids change on reclean.
- Deduplication uses the title only (`pantrychef/data/clean.py:36-42`). It removes about 42.9% of rows. Most removed rows are different recipes with the same name.
- Fix: one `answer(pantry, diet, k)` function, stable hash ids, directions kept. ADR 0004 and the "thin slice" step plan this.
- Worth the cost: required.

### H3. The vocabulary is a shared defect. Examples VERIFIED; the claim that fixing it repairs most failures is a HYPOTHESIS
- `garlic` and `clove garlic` are separate entries, so a pantry with garlic misses 45% to 47% of garlic recipes. Synonyms stay split (scallion, green onion). About 35% of a 200-entry sample is junk or brand names. `large` occurs 67,381 times. The singulariser makes `asparagu`.
- Other failures have separate causes: first-hit USDA matching, the `each` mass fallback, mangled fractions, tagger token logic.
- Table coverage before collapsing: on the old RecipeNLG vocabulary, the top 3,000 entries fully cover 48.4% of recipes. On Wikibooks, a top-N proxy (MEASURED in review) shows 500 rows cover 75.4% of mentions and 16.1% of recipes in full, 1,000 rows 84.2% and 30.5%, and 2,000 rows 91.6% and 53.4%. Weigh this when you size the table.
- Fix: rule-based collapse, then about 500 curated rows, then measure. ADR 0003 already requires a rebuild.

### H4. The "production" bundle does not exist. VERIFIED
- Docs say the default bundle holds `lambdamart-nosub` and the runtime serves it. The disk bundle has three models (`linear`, `lambdamart`, `lambdamart-nosub`), 1,000 recipes, 767 groups, commit `a74280c`, 9 commits old.
- The full-corpus rerun after the parser fix ended in `MemoryError` (`data/processed/p3_full_fixed.log`). "Production train about 25 minutes" has no completed run behind it. Measured parts: generation 14.1 min, fit 1.2 min.
- `scripts/train_recommender.py:139` reloads the bundle while the original is still in memory. That may add to the RAM block (HYPOTHESIS).
- Under ADR 0003 the full-corpus bundle is no longer a product need. See section 9.

### H5. README, ARCHITECTURE and ROADMAP describe a different project. VERIFIED
- Photo input (`README.md:4`), two-tower (`README.md:21,34`), ONNX CV (`README.md:22`), "Ingredient NER" (`README.md:32`; `pantrychef/ingredients/parser.py:1` is rules plus vocabulary match), FAISS (`ARCHITECTURE.md:27,48`), Hydra (`ARCHITECTURE.md:12,74`). None exists in code or `pyproject.toml`.
- `README.md:65` says all numbers reflect the 2026-06-05 parser fix. P1 (05-31) and P4 (06-03) were not rerun.
- Fix: rewrite with Built and Planned columns. Link numbers to `docs/EVALUATION.md` only. Highest return on this list. About half a day.

### H6. A fresh clone fails the quickstart. VERIFIED (report 07 clone test; not rechecked by the reviewer)
- `README.md:73` says `uv sync --extra dev`. Seven test files import scipy at module level and fail to collect. CI hides this because it installs all extras (`.github/workflows/ci.yml:28`). `CONTRIBUTING.md:11-12` (`uv sync`, `uv run pytest`) fails the same way: pytest sits in the `dev` extra.
- Fix: say `uv sync --all-extras`, or use dependency groups. Five minutes.

## 5. Medium

### M1. Evidence is not public or pinned. VERIFIED
- `*.log`, `models/*` and `*.csv` are ignored by git. Results cannot be traced off this laptop. No log records a git SHA. `scripts/recommender_learning_curve.py:105` sets `git_sha=None`. `_git_commit` (`scripts/train_recommender.py:41-50`) runs only `git rev-parse HEAD` and ignores a dirty tree. The raw CSV has no checksum. No documented command produces the P3 table.
- Good: bundles carry corpus fingerprints and seeds are recorded.
- Fix: one helper writes `results/*.json` (commit, dirty flag, config, corpus fingerprint, metrics). ADR 0005 requires it. About 30 lines.

### M2. The learning-curve claim is unsupported. VERIFIED
- The identical metrics at 10k and 20k caps (0.8275) are explained, not a bug. Models at 4k and 8k caps gave the same gold rank on 2,000 of 2,000 queries, and only 378 of 2,000 top-10 lists matched. The metric sees only the gold slot. "Saturates at 10k" (`docs/EVALUATION.md:273`) is not supported. Only two points exist, and none is below 10k.

### M3. Stale or wrong documents. VERIFIED
- `docs/MODEL_CARD_substitution.md` (28k numbers, "pending"). ROADMAP P2 and P3 numbers. `docs/EXPERIMENTS.md` ("No experiments yet"). `ci.yml:39` comment about an eval-gate job that does not exist. Makefile `train` and `eval` targets. The package version `0.0.0` against tag `v0.4.0` (`pantrychef/__init__.py:12`). The `Dockerfile` (`docker/Dockerfile:9-10`): a replica of its `uv sync --no-dev` step failed because the project readme is missing. No `docker build` was run.
- Fix: update or delete in the follow-up docs PR.

### M4. Dependency weight and advisories. Counts VERIFIED; attribution HYPOTHESIS
- `pip-audit` flagged 16 of 96 locked runtime packages, including `pydantic-settings` (direct; fixed in 2.14.2), mlflow, pillow, aiohttp, cryptography, urllib3. Reachability not assessed. `mlflow` is in the `substitution` extra but only 4 runs logged params, 0 metrics.
- Fix: delete mlflow, `uv lock --upgrade`, add `pip-audit` to CI.

### M5. Test gaps. VERIFIED
- Coverage 92%. `eval/__main__` is 0%. `nutrition_main` is 42%. `scripts/` is not measured. No end-to-end test exists because no pipeline exists. No metric-regression gate. CI has no Windows job though development is on Windows. The `dietary_validity` tests cannot catch a bad tagger.
- Fix: one end-to-end test on a tiny committed fixture after `answer()` lands. A small metric floor in CI.

## 6. Low

- L1. Input limits: pantry length, string length and `k` are unbounded. `k` is unbounded in `substitute_command`. Add caps before any API. VERIFIED.
- L2. Pickle loaders: `EmbeddingModel.load` (`pantrychef/substitution/embeddings.py:101`) has no production caller. `scripts/fetch_gismo_gold.py:48,98` unpickles a download (dev only). Delete the first, note the second. VERIFIED.
- L3. LightGBM threads may oversubscribe on a server. Set `n_jobs=1`. HYPOTHESIS.
- L4. Line endings: two committed files show mixed endings on Windows only, because `core.autocrlf=true`. CI is clean. Pre-commit pins ruff v0.4.10 while the lock holds 0.15.15. Add `.gitattributes` with `* text=auto eol=lf`. Bump the pre-commit rev. VERIFIED.
- L5. Dead and duplicate code: `rrf`, `SvdContextModel`, `run_leaderboard`, `build_examples`, `impute.py`, the `substitution/dietary.py` shim, `benchmark.py` (206 lines), duplicated bundle helpers. The `dietary_ok` feature is a constant. VERIFIED. Report 07 section 3 lists a cut. Do not cut LightGBM until the IDF comparison exists.
- L6. Local clutter: `recipes.buggy.jsonl` (625 MB), `models/substitution/buggy_backup/`, empty `.agents`, `.codex`, `tmp1hp3_sy0`, `.pytest-tmp`. Delete locally. Ignore `.codegraph/` and `temp_pytest/`.

## 7. Claims ledger summary

Report 07 checked 43 claims: 14 FALSE, 11 UNSUPPORTED, 11 STALE, 7 TRUE. Full table with evidence: [report 07, section 1](research/2026-10-stage1/07_engineering_claims.md).

| Status | Count | Examples |
| --- | --- | --- |
| FALSE | 14 | photo input; two-tower; FAISS; ONNX CV; NER; Hydra; MLflow "runs"; quickstart; `dietary_ok`; production bundle; per-serving nutrition; Docker path |
| UNSUPPORTED | 11 | "runs on Colab and Kaggle"; "benchmarked against food2vec and GISMo"; graph "beats" food2vec; 100% validity; "reranker crushes baseline"; "saturates at 10k"; "about 25 minutes"; P1 and P4 numbers with no log |
| STALE | 11 | model card; ROADMAP numbers; GISMo table; EXPERIMENTS; version `0.0.0` |
| TRUE | 7 | P2 full, P3 50k, P3 full pre-fix, sub_fill null, learning-curve values, imputer not wired, 246 tests |

Line numbers in report 07 are off by one or two in places. Examples: it cites `README:3` (actual line 4), `README:23` (22), `README:33` (32), `ARCHITECTURE:13` (12), `ARCHITECTURE:29` (27). This audit uses the corrected lines.

## 8. Metric validity

| Published result | Reproducible? | Correctly calculated? | Fair protocol? | Comparable to baselines? | Representative of real use? |
| --- | --- | --- | --- | --- | --- |
| P3 recall@10 0.980 vs 0.746 (50k) | Yes, log matches | Yes | No. Pantry is a subset of the target | No. Rule baseline missing; it scores 0.975 | No |
| P3 full corpus (7.7x, 0.706) | Pre-fix log only; post-fix run failed | Not rechecked | No. Prefix sampling was biased; unbiased ceiling 0.854 | No | No |
| Learning curve (0.8275, in-pool 0.9689) | Yes, JSON matches | Yes | No. Metric blind to model change | No | No |
| P2 MRR 0.289 vs 0.259 (n=82) | Yes, log matches | Yes | No. Beta tuned on the same queries; gold has name variants | No. Baseline is the project's word2vec | Not shown |
| GISMo comparison | No. Pre-fix | Not rechecked | Partial. 78.5% covered subset | No. GISMo model never run | Not shown |
| 100% dietary validity | Yes | Yes, but it is a tautology | No. Tagger checks itself, n=45 | Not applicable | No |
| P4 nutrition coverage 0.800 / 0.487 | No log on disk | Yes for first 5,000 rows | No. Not a random sample | Not applicable | No. Random sample gives 0.710 / 0.439 |
| P1 F1 0.864, recall@10 0.873 | No log | Unknown | Unknown | Unknown | Unknown |
| sub_fill null 0.9349 vs 0.9347 | Yes | Yes | Same leak as P3 | Within the same arms | Not shown |
| 246 tests, 92% coverage | Yes | Yes | Not applicable | Not applicable | Not applicable |

## 9. Checked and found fine

- Tests are real: 246 pass, 92% coverage, `ruff check` is clean, `uv lock --check` passes (VERIFIED, reports 07 and coordinator).
- No secrets in git history. Nothing large is tracked.
- Serving code avoids pickle: bundles use JSON, `.npy` with `allow_pickle=False`, and `.npz`.
- No shared mutable state on the runtime path. `CandidatePoolWorkspace` is not thread-safe, but the runtime never shares one.
- Import boundaries are clean. `recommender` reaches `substitution` only through a lazy hook.
- PR #5 code review found no bugs in: parser multiset logic, hash sampling, `limit<=0`, the conservative `is_known`. It is MERGEABLE with CI green (CI logs expired, so CI test count is unconfirmed).
- `docs/EVALUATION.md` is mostly honest. The imputer, per-serving and macro-gold limits are stated correctly.
- Published numbers that reproduce exactly: 0.975, 0.740, 0.952, 0.958, 0.544, 0.9804, the bootstrap interval, 18.7% coverage, 0.01% vegan, 42.9% dedup, 92% coverage.
- The reviewer reproduced random-sample nutrition at 0.700 and 0.443 against 0.710 and 0.439 in report 06, and the garlic gap at 45% against 47%. Small gaps from sampling; no conflict.
- Candidate pool is covered by persistence and recommend tests. Report 07 did not rerun the byte-identity claim.
- The sub_fill null result and the "no per-serving" statement are honest.

## 10. What ADR 0003 makes moot for the public product

| Finding | Product blocker before ADR 0003 | Now | Research |
| --- | --- | --- | --- |
| Full-corpus RAM block (2.8 GB peak; `MemoryError`) | Yes | No. The display corpus is about 3,800 recipes | Still relevant |
| Candidate ceiling collapse 0.983 to 0.706 at 1.27M | Yes | No. Corpus is small | Still relevant to any retrieval research |
| 123 MB substitution bundle and hosting memory tier | Yes | Mostly. Plan is a 2 MB table (HYPOTHESIS) | Relevant |
| Production bundle missing, "25 minute" run | Yes | No for the public app | Relevant to research |
| RecipeNLG licence (C4) | Yes | Moot for display. Applies to any RecipeNLG-derived artifact | Relevant |
| Current `vocab.json` | Used | Cannot ship. Rebuild (H3) | Relevant |
| Titles-only dedup removing 43% (H2) | Yes | Changes: Wikibooks and Open Recipes need their own cleaning | Relevant |

Not moot: C1, C2, C3, H1, H2, H5, H6. These apply to any corpus.

## 11. Source contradictions

- Report 02's 12/24/6/8 tally conflicts with the review's 39/5/4/2. Resolved: use the review.
- Report 06's 45% Recipes1M share conflicts with 26.4% of raw rows and 31.9% of kept rows. Resolved: use the full count. I found no "45%" string in report 06 by grep, so I cannot show where the error sits.
- Report 01's IDF formula is the wrong form. The synthesis says "length-normalised IDF cosine" and states report 01's coverage form scored 0.194 at 20 extras against 0.914.
- ADR 0003 now gives RecipeNLG "4,448 to 57,392 (scaled from a 1-in-8 sample)". An earlier draft said "5 to 57k". The "5" was wrong. Resolved: use the ADR figure. The Open Recipes counts are "scaled from a 20k sample", so the two scaled ranges use different samples. I did not re-derive either.
- Pantry yields in ADR 0003 are indicative, not an evaluation. The "up to one item away" column is cumulative: it includes make-now recipes. Under a stricter rule (recipe staples only, unread lines count as missing), Wikibooks gives make-now 0 to 28 and exactly-one-away 50 to 134, across 10 hand-written pantries. The 0.985 figure is a vocabulary hit rate, not parse accuracy. A hand read of 60 lines found at least 4 clearly wrong (one rater, not blind).
- The "246 tests" count is from the coordinator's run. The reviewer did not see it, and CI logs have expired.
