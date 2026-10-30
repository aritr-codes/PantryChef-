import json, random, collections, re
from pantrychef.dietary import DietTagger
from pantrychef.ingredients.vocab import load_vocabulary
from pantrychef.ingredients.parser import parse, build_match_index
from pantrychef.ingredients.normalize import canonicalize
from pantrychef.nutrition.match import IngredientMatcher
from pantrychef.nutrition.usda import load_artifact
from pantrychef.nutrition.mass import to_grams
from pantrychef.nutrition.aggregate import aggregate
v=load_vocabulary('data/processed/vocab.json'); idx=build_match_index(v)
T=DietTagger(known=v)
print('vocab',len(v),'coverage',T.coverage(v))
for w,d in [('egg noodle','gluten_free'),('cream of mushroom soup','gluten_free'),('chicken noodle soup','gluten_free'),('peanut butter','vegan'),('coconut milk','vegan'),('salt','vegan'),('rice','gluten_free'),('onion','vegetarian')]:
    print(w,d,'in_vocab',canonicalize(w) in set(v),'valid',T.is_valid(w,d))
# production table + aliases exactly as eval/nutrition_main
table=load_artifact('data/processed/usda.json'); al={k:int(x) for k,x in json.load(open('data/nutrition/aliases.json')).items()}
m=IngredientMatcher(table,aliases=al)
for w in ['oil','sugar','whole milk','egg','butter','flour','salt','vegetable oil','milk','onion','garlic','chicken breast','water']:
    r=m.match(w); print(w,'->',table[r.fdc_id]['description'][:60] if r.fdc_id else None,r.method)
for ln in ['3 cloves garlic','2 eggs','13 cup soy sauce','1 cup milk','1 cup sugar']:
    pi=parse(ln,v,idx); r=m.match(pi.canonical) if pi.canonical else None
    if r and r.fdc_id: print(ln,'|',pi.canonical,pi.quantity,pi.unit,'|',table[r.fdc_id]['description'][:50],to_grams(pi.quantity,pi.unit,pi.canonical,table[r.fdc_id]))
    else: print(ln,'|',pi.canonical,'nomatch')
# sample: every 200th recipes (6.4k), then 1500 random; plus first 2000 lines
head=[];spread=[]
ctr=collections.Counter(); n=0; rec_any=0; gar=cg=gor=both=0; mentions=0
with open('data/processed/recipes.jsonl',encoding='utf8') as f:
    for i,l in enumerate(f):
        if i<1500: head.append(json.loads(l)['ingredients_raw'])
        if i%64: continue
        d=json.loads(l); n+=1; c=set(d['canonical'])
        if i%256==0: spread.append((d['title'],d['canonical'],d['ingredients_raw']))
        ctr.update(c); mentions+=len(c)
        g='garlic' in c; k='clove garlic' in c
        anyg=any('garlic' in x.split() for x in c)
        gar+=g; cg+=k; both+=g and k; gor+=anyg
print('recipes',n,'garlic',gar,'clove garlic',cg,'both',both,'any-token-garlic',gor,'miss if pantry=garlic',1-gar/gor,'naive',cg/(gar+cg))
print('large',ctr['large'],'lemon',ctr['lemon'],'x64')
def cov(lines,name):
    tl=tm=tg=0
    cache={}
    for raw in lines:
        for ln in raw:
            tl+=1; pi=parse(ln,v,idx)
            if not pi.canonical: continue
            r=cache.get(pi.canonical) or cache.setdefault(pi.canonical,m.match(pi.canonical))
            if r.fdc_id is None: continue
            tm+=1; g,ok=to_grams(pi.quantity,pi.unit,pi.canonical,table[r.fdc_id])
            if ok and g is not None: tg+=1
    print(name,'lines',tl,'match',round(tm/tl,3),'mass',round(tg/tl,3),flush=True)
random.seed(5); s=random.sample(spread,min(1500,len(spread)))
cov([x[2] for x in s],'spread'); cov(head,'first1500')
# diet recipe-level on spread (mine)
for d in ['vegan','vegetarian','gluten_free','dairy_free']:
    ok=0;tot=0
    for _,c,_r in spread:
        tot+=1; ok+=all(T.is_valid(x,d) for x in c)
    print(d,'recipe pass',round(ok/tot,5),tot)
# coverage at recipe level for top-N canonicals (mention vs whole recipe)
order=[c for c,_ in ctr.most_common()]; 
for N in (300,500,1000,3000,6000):
    top=set(order[:N]); m_c=sum(ctr[c] for c in order[:N])/mentions
    ok=sum(all(x in top for x in c) for _,c,_r in spread)/len(spread)
    print('top',N,'mention cov',round(m_c,3),'recipes fully covered',round(ok,3))
# GF false negative scan: vocab known + allowed under GF + contains wheat-derived tokens
bad=re.compile(r'\b(noodle|cracker|tortilla|biscuit|crouton|pretzel|ravioli|tortellini|lasagna|spaghetti|macaroni|cake|cookie|gravy|breaded|crust|pastry|pie|dumpling|stuffing|roll|bun|soup)\b')
hits=[w for w in v if T.is_valid(w,'gluten_free') and bad.search(w)]
print('GF-allowed with wheat-suspect token',len(hits),hits[:25])
print('GF allowed total',sum(T.is_valid(w,'gluten_free') for w in v))
