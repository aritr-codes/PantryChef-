# 04 Product and UX report (PantryChef web)

Labels: VERIFIED = seen in repo/data/cited source this session. HYPOTHESIS = my inference, untested.

## 1. Data reality check

**What the served store has (VERIFIED, head -n 200 of recipes.jsonl and pantrychef/common/types.py):** only `recipe_id` ("r0"), `title`, `ingredients_raw` (strings with quantities), `canonical` (ingredient set). Mean 7.0 ingredients per recipe. The processed store has NO directions, NO source URL, NO source name. The raw CSV (`data/raw/full_dataset.csv`, columns `title, ingredients, directions, link, source, NER`) has directions and link, but `clean.py`/`schemas.py` drop them. A recipe detail page therefore needs a preprocessing re-run (or a join back to the CSV) first.

**Absent from the source entirely (VERIFIED, 3000 raw rows + schema):** image, time, servings, rating, cuisine, course, author, nutrition. docs/DATASET.md says "no per-serving - RecipeNLG has no servings field".
- Time: 65% of 3000 early rows mention "N minutes/hours" in steps, but that is a step duration, not total time. Not usable as a card badge.
- Servings: 23% mention serves/yields/makes in title or steps (e.g. "Quick Coffee Cake(6 Servings)" leaks into titles). Not reliable.
- 31% of rows have at least one ingredient line without a leading number ("salt to taste" etc.).

**Sources (VERIFIED):** the first 3000 rows are all `Gathered` from www.cookbooks.com. Sampling the CSV at 30/60/90% offsets (7.5k rows, regex): 69% `Gathered`, 31% `Recipes1M`; domains food.com 44%, tasteofhome.com 36%, then cookeatshare, cookpad, foodnetwork, kraftrecipes, epicurious, tastykitchen. Link-out is possible for most; whether old cookbooks.com ids still resolve is unchecked (HYPOTHESIS).

**Title quality (VERIFIED, ~35 titles read + regex on 3000):** about 5% carry artefacts (digits, parentheses, ALL-CAPS runs). Seen (RecipeNLG titles withheld from this public record): a family name with the `'S` bug, a place name run into the title, a parenthetical suffix, and joke titles. The title-case apostrophe bug ('S) is systematic. Many titles are family-named or jokey (low browse value). Roughly 85-90% usable after cleanup (HYPOTHESIS on the exact rate).
**Instruction quality (VERIFIED):** median 216 characters in total, mean 5 steps; 11% have one step or fewer; encoding damage ("Preheat oven to 425�" lost the degree sign). Terse ("Mix all together and beat into batter"). Fine as a reminder for experienced cooks, poor for novices. Do not present as a tutorial.

**What `canonical` looks like to a human (VERIFIED):** machine tokens, not display names: "jell o gelatin", "level tsp.", "field", "carton cream", "bite size shredded", "butter margarine", "tomato diced", "chicken breast boned", "pear halve", "english pea", "wesson oil". Brand names, unit leaks, stray words ("field" from field peas), merged alternatives ("butter margarine"). Never show it raw. Show the `ingredients_raw` line in the recipe and a curated label on chips. Autocomplete needs a human-checked display layer over the 30,481-token vocab (HYPOTHESIS: ~2-5k popular names, long tail hidden).

**Honest gap-filling (HYPOTHESIS):**
- Images: none. Use typographic cards with a generated monogram tile or the top 3 ingredients. No stock photos implying the dish.
- Dish type: rule from title keywords (cake, soup, salad, casserole, cookies, dip, sauce, bread) plus ingredient signals, labelled "looks like: soup".
- Diet tags: `DietTagger` exists (`pantrychef/dietary`); wording in section 3.
- Time: omit. Show "N steps". Servings: only if parsed from text.
- Source: always "from food.com" link-out.

**Licence (VERIFIED, with caveat):**
- RecipeNLG access terms (HF dataset card https://huggingface.co/datasets/recipe_nlg ; project https://recipenlg.cs.put.poznan.pl): non-commercial research and educational use only; a for-profit employer is also bound; no warranty. HF/TFDS licence metadata says "unknown", which is unreliable. A public web app serving this data is a redistribution and public-use risk even if free, and a monetised one is outside the terms. I could not fetch the official terms page; confirm directly with PUT.
- Copyright (US Copyright Office Circular 33 https://www.copyright.gov/circs/circ33.pdf ; FL-122): bare ingredient lists and simple directions are not copyrightable; descriptive prose and compilations can be. Ingredient facts are low-risk; verbatim instruction text is the exposed part. The dataset contract is a separate risk from copyright.
- Recommendation (HYPOTHESIS, not legal advice): display title, ingredient lines, our own derived tags, and link out for steps. At most a short first-step preview. Get a legal read before public launch. USDA FDC is public domain (DATASET.md); GISMo is CC BY-NC.

## 2. Competitors

- **SuperCook** (https://apps.apple.com/us/app/supercook/id1033049405): ingredient chips, "can make now", saved pantry. An App Store reviewer reports "lost all of my data... pantry was empty" (via appshunter.io/ios/app/1477747816). Lesson: pantry persistence must survive; offer export.
- **MyFridgeFood:** visual quick-pick, shallow ("less depth for customizing or planning", secondary summary, unverified at source).
- **Cooklist:** pantry check plus "cook now"; reviews cite pantry-update glitches after cooking and unstable barcode scanning.
- **Yummly:** shut down 18-20 Dec 2024; saved libraries inaccessible unless exported (https://www.plantoeat.com/blog/2024/12/yummly-is-closing-discover-the-best-meal-planning-alternative). Lesson: no server-held accounts as the only copy; let users export.
- **Mealime:** complaint "cannot filter by ingredients I have on hand"; pantry features behind Pro (https://unstar.app/blog/allrecipes-yummly-nyt-cooking-paprika-mealime-recipe-apps-ranked-2026). Demand exists.
- **Paprika:** recipe manager, not pantry search; complaints on import accuracy. Samsung Food (Whisk), Plant Jammer, Tasty, Allrecipes, Spoonacular: no primary-source detail obtained this session; any claim about them is HYPOTHESIS.
- **Pantry-entry burden (anecdotal, not measured):** PCWorld on Pantry: telling it you are down to two bananas "feels like more work than scribbling bananas on a notepad" (https://www.pcworld.com/article/524936/pantry.html). Reddit r/NoCodeSaaS and r/homemaking commenters call inventory tracking "too much life admin"; one thread suggests quick yes/no for common items. I found no dropout statistic, so "top dropout point" is plausible but unquantified.
- **Patterns that work (HYPOTHESIS from the above):** chips; live result count; "have X of Y"; quick-pick grid of the ~40 most common items; remembered pantry; no login wall.

## 3. Core UX decisions

- **Staples:** assume water, salt, black pepper, cooking oil by default (visible toggle, editable; optional "also assume" sugar, flour, butter). Show assumed staples greyed in the recipe so nothing is hidden. Salt, pepper, oil, water, sugar are very frequent in the 200-row sample, so ignoring them in the match raises hits a lot. Must be applied before ranking, not only in display: `ScoredRecipe.missing` today lists every canonical token.
- **Minimum usable pantry:** 3 ingredients incl. one non-staple protein/veg/carb; at 1-2, still show results with "add more for better matches". Never block.
- **Result reading:** sections "You can make now" (0 missing after staples), "One item away" (names the item, plus a swap chip if available), "Two away". Show "You have 6 of 8", never a percentage from the model `score`.
- **Empty / zero results:** empty pantry shows the quick-pick grid and example pantries. Zero results: offer the nearest by relaxing one ingredient ("Remove 'saffron' to see 40 recipes"). Never a bare "No matching recipes."
- **Synonyms / unknown:** autocomplete maps aubergine->eggplant, coriander->cilantro. Unknown input stays as a visible "Not recognised: 'xyz'. Closest: ..." chip the user can keep as ignored. Never silently drop.
- **Confidence without fake precision:** `Substitute.score` is uncalibrated; show rank order or three words ("good swap / works in some dishes / risky") only after validation. Nutrition: "about", ranges, and "based on 5 of 7 ingredients" from `match_coverage`/`mass_coverage`. Hide it below a coverage threshold (~60%, HYPOTHESIS). Serves is unknown, so show whole recipe or per 100 g (`facts_per100g`), not per serving.
- **Explanations:** one line on the card ("Uses 4 of your 5 items, incl. chicken"); fuller "Why this?" disclosure on detail. Derive from matched/missing, not model internals (LambdaMART is not per-recipe explainable; don't invent reasons).
- **Dietary claims (safety):** the ontology excludes unknowns under a constrained diet (VERIFIED in `dietary/ontology.py` docstring), which is the right default. Wording: "Looks vegan (judged from ingredient names)", not "Vegan". Allergens: "may contain, check labels"; never "safe for X". Diet filter does not imply allergy filter. Persistent footer plus per-card micro-copy. Vocab tag coverage is below 100% (EVALUATION.md ~102-103), so diet-filtered results will be thin.
- **Accounts:** CONFIRM no accounts in v1. localStorage for pantry, staples, diet, saved items, plus export/import JSON and a shareable URL-encoded pantry (answers the Yummly/SuperCook data-loss lessons). Warn that browsers can clear localStorage.

## 4. Value proposition

- **Who:** home cooks with a half-full fridge and no plan; students; budget cooks; waste reducers (HYPOTHESIS, no user data).
- **One thing better than Google:** "tell me what I have; get what I can cook now and the single item that unlocks more", no ads, no life story, no sign-in, on a phone.
- **Return reason:** weak by default (no images, thin steps). Saved pantry plus "one shop away" is the hook; be honest that v1 retention will be low.
- **Must-have v1:** chips + autocomplete + quick-pick grid; staples toggle; Make-now / One-away sections; detail with raw ingredient lines and link-out; localStorage; careful diet exclusion; mobile first.
- **Later:** inline substitutions, nutrition estimate, photo input, saved list, scaling.
- **Harmful if built early:** photo input (CV unbuilt, errors poison trust, adds a correction screen); per-serving nutrition (no servings, fake precision); ratings (none exist); chat/"AI chef"; accounts; time/cuisine filters (no data); allergen "safe" badges; meal planning.

## 5. Design direction

(The ui-ux-pro-max skill was not invoked; this direction is my own.)
- **Feel:** printed cookbook index / grocery list, not a SaaS dashboard. Warm off-white #FAF6EF, ink #1F1B16, one accent tomato #C8442B for actions, olive #4A6B3A for "have", amber #B7791F for "missing", never colour alone.
- **Type:** serif for recipe titles (Fraunces or Source Serif 4), clean sans for UI (Inter or similar); body 16-18 px, ingredient lines 17 px, line height 1.5.
- **Density:** single column on mobile; cards 88-120 px tall, no image slot; a thin have/need fill bar; generated monogram tile (title initial on a colour from the main ingredient) for visual rhythm without fake photos.
- **Card anatomy:** title; "You have 6 of 8 (staples assumed)"; missing chips with swap icon if a swap exists; diet pill only if computed; source domain. No score number.
- **Micro-interactions:** chip add scale ~120 ms; live result count in `aria-live="polite"` with no layout jump; undo toast on removal; skeleton cards not spinners; honour reduced-motion.
- **WCAG 2.2 AA:** chips are buttons with `aria-label="Remove eggs"`, target at least 24x24 CSS px (2.5.8), aim 44 px on mobile; autocomplete follows the ARIA combobox pattern (`role=combobox`, `aria-expanded`, `aria-controls`, `aria-activedescendant`, listbox options; Esc closes; Enter adds; announce "3 suggestions"); filters are real checkboxes with visible focus not obscured (2.4.11); contrast 4.5:1 text and 3:1 for icons/focus; have/missing by text and icon too (1.4.1); reflow at 320 px and 200% zoom (1.4.10); no drag-only interactions (2.5.7); don't ask to re-enter data (3.3.7).

**Wireframes**
```
S1 Pantry (mobile)
[PantryChef]                  [staples: on v]
What's in your kitchen?
[ type an ingredient...        ]
  suggestions: egg | eggplant | egg noodle
Your pantry (4): (eggs x)(rice x)(spinach x)(soy sauce x)
Quick add: [onion][garlic][chicken][pasta][tomato][cheese]
Diet: [ ]vegetarian [ ]vegan [ ]gluten-free [ ]dairy-free
[ See 38 recipes ]   Not recognised: "xyz" (ignore)

S2 Results
< Pantry | 4 items | Diet: vegetarian (judged from names)
YOU CAN MAKE NOW (3)
 Spinach Rice Bowl     You have 5 of 5 (+salt, oil assumed)
ONE ITEM AWAY (12)
 Egg Drop Soup         You have 4 of 5  missing: scallion [swap: onion]
 Show more
Footer: Diet tags are estimates from ingredient names. Check labels.

S3 Recipe detail
< Results
Egg Fried Rice                        from food.com ->
You have 4 of 5. Missing: scallion   [swap ideas v]
INGREDIENTS  (have / missing / assumed)
 [v] 2 c. cooked rice
 [x] 2 scallions  -> try onion, chives (rough suggestion)
 [.] salt, oil (assumed)
STEPS (n steps): full instructions on food.com ->
Rough nutrition: ~420 kcal whole recipe (4 of 5 ingredients counted) [how?]
Why this recipe? v
```

## 6. Frontend contract needs vs today

| Screen | Needs per request | Today (VERIFIED from code) |
|---|---|---|
| Autocomplete | prefix/fuzzy search on display names, canonical id, synonyms, popularity | Nothing. `vocab.json` is machine tokens; `canonicalize` exists. No display layer or synonym table (GAP) |
| Results | per recipe: id, clean title, matched[], missing[], have/total, staple-adjusted flags, diet flags, step count, source domain, top swap per missing; total count; unrecognised echo; sectioning | `ScoredRecipe(recipe_id, score, title, matched, missing)` only. No staples, totals, diet, source, swaps. `RecommenderRuntime.recommend(pantry,k,model_name)` returns a plain list, no pagination or count |
| Detail | `ingredients_raw` with per-line have/missing/assumed mapping, directions, link, source, nutrition with coverage | `ingredients_raw` is stored but not returned; directions and link are not in the processed store (GAP); no line-to-canonical mapping |
| Substitutions | ingredient + recipe context + diet -> ranked names, reason, coarse confidence, amount ratio | `Substitutor.substitutes(ingredient, diet, recipe, k)` -> `Substitute(ingredient, score, dietary_valid, arm)`. `dietary_valid` always True post-mask; score uncalibrated; no explanation or quantity conversion (GAP) |
| Nutrition | totals, per100g, coverage, which lines counted | `aggregate(raw_lines, vocab, index, matcher)` -> `RecipeNutrition` with coverage (good). Re-parses raw lines at request time; needs `usda.json` (2.2 MB) and matcher. `NutritionFacts` docstring says "per-serving" but there is no serving data; no per-line breakdown |
| Photo review | detections with confidence and canonical mapping | `vision/` and `serving/` are empty placeholders |

Other gaps: CLI `cook` uses the baseline `recommend` over a freshly built `InvertedIndex` (loads all recipes), not the production LambdaMART runtime. The backend must keep one index in memory (RAM is the binding constraint); a hosted demo probably needs a curated subset (100-300k recipes, HYPOTHESIS), which also helps title cleanup.

## What the brief assumes that the data cannot support
1. Images on cards (none exist in the dataset).
2. Nutrition per serving (no servings field).
3. Cooking time, cuisine, course, rating filters (no such fields).
4. Recipe detail with steps and source link from the current processed store (dropped in preprocessing; must be rebuilt from raw CSV).
5. Photo review/correction (vision module is a placeholder).
6. Certified dietary claims (names-only inference).
7. Public or commercial display of RecipeNLG content (terms are non-commercial research).
