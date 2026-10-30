# Independent review of the Stage 1 synthesis

Date: 2026-10-30. Reviewer: an independent agent that did not write the synthesis.
Verdict: **GO WITH FIXES**. Scripts and outputs are in `evidence/review/`.

## Required fixes (all applied to README.md)

1. The 12/24/6/8 count was not a top-result tally. Recounted as 39 variants, 5 real,
   4 with a caveat and 2 wrong (one rater, not blind).
2. "One data defect causes most failures" was not measured. Relabelled as a hypothesis.
3. The licence section overreached. "Blocks" became "unresolved risk". Contract and
   copyright questions are now separate. Derived artifacts are included in D1.
4. The sequence contradicted itself. D1 now has a time limit. Baseline evaluation now
   comes before data work. A thin end-to-end slice now follows the reset.
5. Allergen flags contradicted reports 02 and 04. Removed from the table definition.
6. The Recipes1M share was resolved by a full count: 26.4% of raw rows and 31.9% of
   kept rows.

## Other results applied

- The identical learning-curve metrics are explained: different models produced the
  same gold rank on 2,000 of 2,000 queries, so the metric cannot see the difference.
- On pantries with extra items, the model trained the current way falls below a
  one-line rule. Retrained, it has no demonstrated lead over an IDF cosine rule. At
  3 extra items the retrained model scored 0.985 and the IDF rule 0.980. At 10 and 20
  extras the IDF rule led. No paired intervals were run.
- "Makeable at k" cannot select a ranker, because sorting by missing count wins it by
  construction.
- A table of 3,000 entries fully covers only 48.4% of recipes before collapsing.

## Reproduced by the reviewer

0.975, 0.740 and 1,404 queries; 0.952, 0.958 and 0.544; 0.9804 from the log; the
bootstrap interval (+0.030, -0.051 to +0.109); 18.7% tagger coverage; 0.01% vegan
(0 of 4,978); 42.9% deduplication; 92% test coverage; the 1,000-recipe bundle at 9
commits old. Random-sample nutrition coverage came to 0.700 and 0.443, against
0.710 and 0.439 in report 06. The garlic gap came to 45%, against 47%.

## Not checked by the reviewer

The fresh-clone failure, the 246 test count (seen separately by the coordinator), any
full-corpus result, the official RecipeNLG terms page, browser latency, the 36% and
35% hand labels, and most hosting facts.
