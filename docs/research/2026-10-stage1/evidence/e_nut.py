import json,random,collections
from pantrychef.ingredients.vocab import load_vocabulary
from pantrychef.ingredients.parser import parse,build_match_index
from pantrychef.nutrition.match import IngredientMatcher
from pantrychef.nutrition.mass import to_grams
from pantrychef.nutrition.aggregate import aggregate
S='C:/Users/HP/AppData/Local/Temp/claude/g--Books-Python-ACTUAL-CODES-PROJECTS-pantrychef/1127416b-d672-4b96-83c2-8570c428ce9d/scratchpad/research/'
v=load_vocabulary('data/processed/vocab.json'); idx=build_match_index(v)
u={int(k):x for k,x in json.load(open('data/processed/usda.json',encoding='utf8')).items()}
al=json.load(open('data/nutrition/aliases.json'))
m=IngredientMatcher(u,al)
sample=json.load(open(S+'sample20k.json'))
random.seed(3); rs=random.sample(sample,2000)
# per-canonical match & kcal share
agg=collections.defaultdict(lambda:[0,0.0,None,None,0.0]) # n, kcal, desc, method
kc_matched=0.0; kc_unmatched_est=0
lines=matched=massed=0
unm=collections.Counter(); 
for title,c,raw in rs:
    for ln in raw:
        lines+=1
        pi=parse(ln,v,idx)
        if not pi.canonical: unm['<nocanon>']+=1; continue
        r=m.match(pi.canonical)
        if r.fdc_id is None: unm[pi.canonical]+=1; continue
        matched+=1
        food=u[r.fdc_id]; g,ok=to_grams(pi.quantity,pi.unit,pi.canonical,food)
        if ok and g is not None:
            massed+=1
            k=food['per100g'].get('kcal'); k=0 if k is None else k*g/100
            a=agg[pi.canonical]; a[0]+=1; a[1]+=k; a[2]=food['description']; a[3]=r.method; a[4]+=g
print('lines',lines,'match cov',matched/lines,'mass cov',massed/lines)
print('top unmatched canonical:',unm.most_common(25))
print('--- most frequent massed canon -> USDA desc (method)')
for c,a in sorted(agg.items(),key=lambda x:-x[1][0])[:70]:
    print(f'{c!r:35} n={a[0]:4} avg_g={a[4]/a[0]:6.1f} -> {a[2][:70]} [{a[3]}]')
# kcal share by method
tot=sum(a[1] for a in agg.values()); 
bym=collections.defaultdict(float)
for a in agg.values(): bym[a[3]]+=a[1]
print('kcal share by method',{k:round(x/tot,3) for k,x in bym.items()})
