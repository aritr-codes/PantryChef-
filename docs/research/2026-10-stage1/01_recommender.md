# 01 Recommender audit (retrieval + ranking)

Evidence: code read via CodeGraph; logs `data/processed/p3_full.log`, `p3_50k_*.log`, `learning_curve.json`; one cheap script `01_leak_check.py` (same folder; first 50k recipes only, 1404 test queries).

## F1. The eval task is leaked by construction; the ML lead is largely an artifact. CRITICAL. VERIFIED.
- `query_sim.make_query` (query_sim.py:47-63): pantry = canonical set minus a random 30% (`mask_fraction=0.3`, config.py). The pantry is a strict subset of the gold recipe, with no distractors, and the hidden count is a deterministic function of recipe length (`round(0.3*L)`, min 1).
- Features (features.py:17-26) expose exactly that. `match_frac_pantry` is 1.0 for gold by definition. `n_missing`/`recipe_len`/`coverage` jointly encode "missing = round(0.3 L)". Three columns suffice. `sub_fill_*` and `dietary_ok` (constant 1.0) add nothing (consistent with the recorded sub null finding).
- Falsification run (01_leak_check.py, 50k recipes, same 200-cap pool): a hand rule with no ML ("pantry is subset of recipe, then smallest |missing - round(0.3L)|") gets **recall@10 = 0.975**. Overlap = 0.740, ceiling 0.989. The reported LambdaMART at 50k is ~0.98 (0.965 in the p3_50k_nosub log). So LambdaMART is roughly equal to a 3-line rule that encodes the generator. Only ~3.4 pool candidates per query are pantry-supersets.
- Consequence: "LambdaMART 0.980 vs overlap 0.746" measures "learned the query generator", not recommendation quality. Not defensible as a headline.
- Overlap is unfair in the other direction too. It sorts by coverage desc (recommend.py:95), so it prefers small recipes fully inside the pantry, whereas the gold has coverage ~0.7 by construction. At 1.27M overlap hits 0.089 because many tiny recipes have coverage 1.0.
- Metric code itself is correct (recall_at_k / mrr_at_k, recommend_eval.py:25-37; deterministic string-id tie-break). The flaw is task design, not an implementation bug.
- Train/eval split: md5(recipe_id) 80/20 (query_sim.py:25), disjoint for queries. No near-duplicate handling: exact-duplicate canonical sets are 2.45% in the first 50k (VERIFIED); near-dups HYPOTHESIS. A duplicate of the gold counts as a miss (understates), and duplicates of train recipes may sit in test (mild; the model has only 8 generic features so memorisation risk is low).
- Fix: keep recovery only as a unit-test sanity check; headline with F2 protocols. Cost low, value high.

## F2. The task does not match the product. High. VERIFIED (by reading); protocols are HYPOTHESIS until run.
A real pantry has 10-30 items, mostly irrelevant to any one recipe, lacks 1-3 recipe items, and the user wants good makeable recipes. No-user-data protocols, cheapest first:
1. Distractor pantries: keep 70-100% of a recipe plus k extras (k=0,5,10,20), extras uniform vs sampled by co-occurrence (basket-like). Reveals whether the leak signal (`match_frac_pantry`) collapses.
2. Makeability@k: share of top-k with missing <= m (m=0,1,2), staples excluded. This is the stated product goal; compare overlap, IDF-overlap, BM25, LTR.
3. Graded labels from the (pantry, recipe) pair (0 missing=3, 1=2, 2=1) on pantries not derived from any one gold; train LambdaMART on these. Label is not a function of a generator.
4. Dedup first (MinHash/Jaccard >= 0.9), count a hit if any cluster member ranks.
5. Quality/popularity proxies (F5).
6. Hand-judged set: 100 pantries x top-5 from 3 systems, pairwise preference, bootstrap CI (~2 h).
Worth it: 1, 2, 4, 6 yes; 3 yes if LTR stays.

## F3. Candidate bottleneck is retrieval ordering. High. VERIFIED (logic + log).
- `_candidate_rows` (recommend.py:53-96) unions postings of all pantry items, counts matches, orders by coverage = matched/recipe_len desc, then fewer missing, then id, cap 200. No IDF: "salt" (huge posting list) counts as much as "saffron". At 1.27M, many short recipes reach coverage 1.0 and fill the 200 slots; the ceiling falls 0.983 (50k) -> 0.706 (p3_full.log; older pre-parser-fix run, 5000 queries; the learning curve with hash-order sampling gives 0.854, so do not quote 0.706 as current).
- Retrieval and ranker pull opposite ways: the pool prefers high coverage, the model rewards pantry-subset with ~30% missing.
- Alternatives:
  - Bigger pool (1-5k): ceiling up; cost is linear because `extract_features` is a Python dict per candidate (recommend.py:132). Needs vectorised features. Medium value.
  - IDF-weighted overlap (sum idf(matched)/sum idf(recipe)): numpy over existing postings, only an idf vector extra. Best cost/benefit. HYPOTHESIS ceiling >0.9 at full scale. Experiment: Kaggle CPU, 8 GB, candidate-only (no training), 2k test queries, compare coverage vs IDF vs BM25 at pool 200/1000, <30 min.
  - BM25 over ingredient tokens (`bm25s`): same family, add to the same experiment.
  - Rarest-first filter: good for "must use X", weak in general.
  - WAND/MaxScore: correct algorithm for exact top-k of a weighted-overlap score, but only worth it if latency is measured as a problem.
  - SPLADE / static embeddings + FAISS/HNSW: reject. Set containment on exact ingredient identity is lexical; ANN blurs it, needs embedding training, and still needs the reranker.
  - Dedup/curate: strongest lever. Cuts RAM 4-10x, removes junk, helps the ceiling. Near-dup rate is a HYPOTHESIS (MinHash on Kaggle). Recommendation: ship a curated deduped subset (~100-300k), not 1.27M.
- Docs vs code: README.md:21,34 and recommender/__init__.py:4 say "two-tower"; retrieval/__init__.py:1-3 says "Phase 3 FAISS". **Neither exists** (grep: no faiss dependency or two-tower code). The shipped system is inverted index + LightGBM LambdaMART.

## F4. Serving feasibility. High (RAM), Medium (latency). Partly VERIFIED.
- Memory: `InvertedIndex` (index.py:26-73) keeps a pydantic `Recipe` per row, a frozenset per recipe, string-set `postings`, plus int64 `postings_rows`. `from_state` rebuilds from model_dump dicts. At 1.27M this is multi-GB; p3_full_fixed.log died with an ArrayMemoryError for 4.6 MB. The bundle on disk (`models/recommender/recommender_bundle.zip`, 306 KB) can only hold a tiny corpus; no full-corpus production bundle exists.
- Lower-memory design (HYPOTHESIS, simple): int32 CSR postings (indptr/indices), int16 lengths, ingredient-id vocab, titles in SQLite/mmap, no per-row objects. 300k recipes x ~9 items ~ 2.7M postings ~ 11 MB plus titles. Fits 512 MB.
- Latency: each query sorts the concatenated postings (recommend.py:76); common items have 1e5-1e6 rows, so tens of ms at full scale (HYPOTHESIS, measure; the "~50x faster" is in the docstring only). 200 Python feature dicts + LightGBM predict is low ms. Fine for a web app.
- Thread safety: `CandidatePoolWorkspace` is documented not thread-safe (recommend.py:28-40), but `RecommenderRuntime.recommend` (runtime.py) calls `rerank` without a workspace, so each request allocates its own buffer. The serving path is safe; the workspace matters only in train/eval. LightGBM Booster predict is read-safe.
- Browser: 8 features (3 useful), 200 trees x 31 leaves. Export via `model_to_string()` / `dump_model()` and a ~30-line JS tree walker. ONNX runtime-web is heavy for this and rejected. The real browser blocker is the corpus index: ~100k recipes as compact binary (5-10 MB) is feasible, 1.27M is not.
- Startup: `from_state` rebuilds all indices on load (minutes at 1.27M). Persist CSR arrays as .npy and mmap.

## F5. No quality signal. High for product. VERIFIED.
`Recipe` has only id/title/ingredients_raw/canonical (types.py:32-38); RecipeNLG carries no ratings. Excluding popularity is consistent with the recovery task but wrong for a product: junk and classics tie. Proxies without user data: (a) near-duplicate cluster size as crowd endorsement (HYPOTHESIS; check against the hand-judged set); (b) structural filters (>=4 ingredients, has steps, length bounds, parse rate, title sanity); (c) ingredient-combination typicality; (d) train a quality classifier on a rated corpus (Food.com reviews) and transfer on ingredient/structure features (Kaggle, AUC on held-out, then eyeball). Do (b) now, (a) next, (d) optional portfolio piece.

## F6. Other verified discrepancies
- `learning_curve.json`: the 10k and 20k rows are identical to 16 digits (mrr 0.82748095..., recall 0.8485, in-pool 0.96894725...) despite kept=8694 vs 17472. Either the fitted function is the same simple deterministic rule (consistent with F1) or there is a reuse bug. HYPOTHESIS; experiment: train 1k/2k/5k/10k, diff model dumps (cheap).
- Eval "ceiling 0.854" (curve) vs 0.706 (p3_full) differ by parser fix and sampling order.
- I did not parse the 3.9 MB `p3_50k_fixed.log`; confirm the final 50k number before quoting 0.980.

## F7. Research (kept what suits a solo project)
- Keep: IDF/BM25 weighted overlap; LightGBM LambdaMART as the reranker (Burges 2010, "From RankNet to LambdaRank to LambdaMART"; LightGBM lambdarank docs), trained on graded makeability labels with richer features (idf-coverage, rare-missing count, staple-excluded coverage, cluster size).
- Recent recipe work found is LLM-based (KERL, https://arxiv.org/pdf/2505.14629, include/exclude ingredient constraints) or menu planning. I found no public benchmark for pantry set-containment; publishing the distractor/makeability protocol is itself a contribution.
- Rejected: LLM rerankers (no-LLM constraint), two-tower/SPLADE/FAISS (F3), GNN recommenders (no interaction data), ONNX runtime-web (F4).
- Web search did not confirm LightGBM->ONNX/treelite specifics; the JS tree-walker does not depend on them.
- Sources: https://arxiv.org/pdf/2505.14629 ; https://opensource.microsoft.com/blog/2021/09/02/onnx-runtime-web-running-your-machine-learning-model-in-browser/ ; https://lightgbm.readthedocs.io ; Burges, MSR-TR-2008-109.

## Ranked recommendations
1. (Critical, 1-2 d) Retire recovery numbers as headline; add distractor + graded makeability eval; compare against the hand rule, overlap, IDF-overlap, BM25.
2. (High, 1 d) IDF/BM25 candidate scoring; measure full-corpus ceiling on Kaggle.
3. (High, 2-3 d) Dedup + quality-filter to a curated 100-300k corpus.
4. (High, 2 d) CSR/mmap index with persisted arrays; target <300 MB.
5. (Medium, 1 d) Quality/popularity-proxy feature.
6. (Medium, 2 h) Hand-judged pairwise set.
7. (Low) Fix FAISS/two-tower doc claims; investigate identical 10k/20k curve rows.
8. (Low) Browser JS tree walker once the index is small.
