import json,random
from pantrychef.ingredients.vocab import load_vocabulary
from pantrychef.ingredients.parser import parse,build_match_index
from pantrychef.nutrition.match import IngredientMatcher
from pantrychef.nutrition.aggregate import aggregate
from pantrychef.nutrition.mass import to_grams
S='C:/Users/HP/AppData/Local/Temp/claude/g--Books-Python-ACTUAL-CODES-PROJECTS-pantrychef/1127416b-d672-4b96-83c2-8570c428ce9d/scratchpad/research/'
v=load_vocabulary('data/processed/vocab.json'); idx=build_match_index(v)
u={int(k):x for k,x in json.load(open('data/processed/usda.json',encoding='utf8')).items()}
m=IngredientMatcher(u,{k:int(x) for k,x in json.load(open('data/nutrition/aliases.json')).items()})
s=json.load(open(S+'sample20k.json')); random.seed(11)
for title,c,raw in random.sample(s,20):
    r=aggregate(raw,v,idx,m)
    print(f'{title[:40]:40} lines={r.n_lines} match={r.match_coverage:.2f} mass={r.mass_coverage:.2f} grams={r.total_grams:.0f} kcal={r.facts_total.calories} kcal/100g={r.facts_per100g.calories}')
# garlic check
for ln in ["3 cloves garlic","2 garlic cloves, minced","1 head garlic","2 potatoes","3 carrots","1 lb potatoes","2 tablespoons vegetable oil","1 cup milk","1 cup water","2 eggs","1 lemon"]:
    pi=parse(ln,v,idx); res=m.match(pi.canonical) if pi.canonical else None
    if res and res.fdc_id:
        f=u[res.fdc_id]; g=to_grams(pi.quantity,pi.unit,pi.canonical,f); print(ln,'|',pi.canonical,pi.quantity,pi.unit,'|',f['description'][:50],f.get('unit_grams'),'->',g, 'kcal100',f['per100g'].get('kcal'))
    else: print(ln,'|',pi.canonical,pi.quantity,pi.unit,'| no match')
