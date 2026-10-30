import json, random, sys, collections
from pantrychef.common.types import Recipe
from pantrychef.retrieval.index import InvertedIndex
from pantrychef.recommender.config import RecConfig
from pantrychef.recommender.query_sim import make_query, is_train
from pantrychef.recommender.recommend import candidate_pool
N=int(sys.argv[1]); rs=[Recipe(**json.loads(l)) for l in open('data/processed/recipes.jsonl') .readlines()[:0]] if False else []
with open('data/processed/recipes.jsonl') as f:
    for i,l in enumerate(f):
        if i>=N: break
        rs.append(Recipe(**json.loads(l)))
sets=collections.Counter(frozenset(r.canonical) for r in rs)
print("n",len(rs),"distinct canon sets",len(sets),"dup-rate",1-len(sets)/len(rs))
idx=InvertedIndex.build(rs); cfg=RecConfig(); rng=random.Random(1)
tests=[r for r in rs if not is_train(r.recipe_id)][:1500]
hit=ov=n=inpool=sup=0
for r in tests:
    q=make_query(r,cfg,rng)
    if not q: continue
    n+=1; P=set(q.pantry); pool=candidate_pool(idx,P,200)
    ids=[x.recipe_id for x in pool]
    if q.gold_id in ids: inpool+=1
    ov+= q.gold_id in ids[:10]
    # leak rule: pantry subset of recipe, rank by |missing - round(.3 L)| 
    def key(x):
        c=idx.canon_sets[x.recipe_id]; L=len(c)
        return (not P<=c, abs((L-len(P&c))-max(1,round(.3*L))), x.recipe_id)
    top=sorted(pool,key=key)[:10]
    hit+= q.gold_id in [x.recipe_id for x in top]
    sup+= sum(P<=idx.canon_sets[x.recipe_id] for x in pool)
print("n",n,"ceiling",inpool/n,"overlap r@10",ov/n,"handrule r@10 (no ML)",hit/n,"avg superset cands in pool",sup/n)
