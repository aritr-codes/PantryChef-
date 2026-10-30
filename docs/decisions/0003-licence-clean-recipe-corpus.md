# 0003. Licence-clean recipe corpus for the public app

- **Status:** Accepted
- **Date:** 2026-10-30
- **Deciders:** Project owner, Tech Lead

## Context

RecipeNLG access terms allow non-commercial research and educational use. They
allow sharing with colleagues who accept the same terms. We found no grant of a
public right. Also, 31.9% of the kept rows are tagged `Recipes1M`, which has
separate terms. The owner chose to build with an alternative corpus first, and not
to wait for permission from the RecipeNLG authors.

The public app needs recipes that it can display and that visitors can download.

Measurements on 2026-10-30 (`docs/research/2026-10-stage1/evidence/`):

| Corpus | Licence | Recipes | Instructions | Vocabulary hit rate | Up to one item away, 10 test pantries |
| --- | --- | --- | --- | --- | --- |
| Wikibooks Cookbook snapshot, 2024-07-31 | CC BY-SA 4.0 | 3,800 with an ingredient list | 98.3% | 0.985 | 69 to 330 |
| Open Recipes snapshot, 2017-01-07 | CC BY 3.0 (database) | 173,276 | None, by design | 0.867 | 277 to 4,635 (scaled from a 20k sample) |
| RecipeNLG (reference only) | Research and education | 1,274,290 kept | Yes | Not comparable | 4,448 to 57,392 (scaled from a 1-in-8 sample) |

The pantries are hand-written and the match rule is loose. It counts peanut butter
as butter, for example. The hit rate uses the RecipeNLG vocabulary and means only that
some vocabulary entry matched. A hand read of 60 lines found at least 4 clearly wrong.
These numbers are indicative and are not an evaluation.

A stricter review rule (recipe staples only, unread lines count as missing) gave
Wikibooks 0 to 28 make-now recipes and 50 to 134 at exactly one item away, across the
10 pantries. Some strict make-now results are trivial, such as "Steamed Rice".

Snapshots used for these measurements (sha256):

- Wikibooks, `gossminn/wikibooks-cookbook` on Hugging Face, third-party scrape with no
  revision ids: `67fd704a6d9280742d2f217ad9f9e4f50921b84fcdefdff9bac749a518c4d46b`.
- Open Recipes, `napsternxg/openrecipes-20170107-061401-recipeitems` on Hugging Face:
  `6e40a82e0bd83468f8020b461924d9d6cf8bdba822642c5276868104acb6d682`. The original S3
  URLs now return 403.

## Options considered

1. **RecipeNLG subset with a link to the source.** Largest choice, but it has no
   public grant, and the app would ship a derived bulk data file.
2. **Wikibooks Cookbook only.** Clean provenance, full instructions, wide cuisine
   range. Only 3,800 recipes. Too small to train ingredient embeddings well.
3. **Open Recipes only.** Large, with servings and times. It has no instructions,
   and only 33 of 150 sampled links returned success. Automated checks can be
   blocked, so the true rate is unknown. Tasty Kitchen is the largest source and its
   site did not respond. The licence covers the database. It does not clear the
   rights of each publisher.
4. **tastyR.** CC0 label, about 14k recipes with ratings, but scraped from Allrecipes.
   The label cannot clear the source's rights. Rejected for display.
5. **USDA MyPlate Kitchen.** Attribution rules are clear, but no bulk download exists
   and some recipes come from partners. Rejected for now.
6. **Two layers (chosen).** Wikibooks for display. Open Recipes for statistics.

## Decision

- **Display corpus:** Wikibooks Cookbook. The app shows full recipes with a link to
  the page, the authors' credit through that page, and the licence. We build our own
  snapshot from the official MediaWiki API or dump, with a revision id per page.
- **Training corpus:** Open Recipes supplies ingredient statistics and the mined
  substitution model. Derived statistics ship only if the mined swaps pass their test
  in M5. If they do not, nothing from Open Recipes ships. Titles, ingredient text,
  descriptions and images from Open Recipes never ship. The snapshot stays in a private
  store, and the About page credits the database if its statistics ship.
- **RecipeNLG:** research and comparison only. No RecipeNLG text, vocabulary or
  trained artifact ships in the public app.
- **Vocabulary:** rebuild from Wikibooks, Open Recipes and hand curation. The current
  `vocab.json` comes from RecipeNLG, so it cannot ship.
- **Recipe pack:** the app loads recipes from one replaceable versioned pack, so a
  later corpus needs no engine change.
- **Licences:** code stays MIT. The recipe pack is an adaptation (ingredient ids are
  added to the text), so it stays CC BY-SA 4.0. It is a separate file with its own
  licence notice, and it records changes. Any committed recipe text, such as a test
  fixture, sits under a path with a CC BY-SA notice. `LICENSES.md` lists the licence of
  each path and artifact.
- **Enforcement:** the pack builder reads only input files on a hash allow-list. Each
  table row records its provenance. A test fails if a build opens another data file.

## Consequences

- Positive: a clear public right for displayed text. A small pack (estimate: under
  10 MB) suits browser use. Wikibooks has categories, which allow simple filters, and
  more varied cuisines.
- Negative: far fewer recipes. Make-now counts are low for some pantries (1 to 73 in
  the test), so "one item away" carries the experience. Servings and time exist on
  only about 33% of recipes. Old claims about the 1.27M corpus are no longer product
  claims. Mined substitution quality on 168k unique recipes is unknown.
- Risk: Open Recipes statistics are a derived use of scraped data under a database
  licence. This is an engineering judgement, not legal advice.
- Revisit when: RecipeNLG authors grant permission, a better corpus appears, or user
  testing shows that 3,800 recipes are too few. A "more ideas" pack from Open Recipes
  needs its own live-link test first.
