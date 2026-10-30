import random,math,collections,numpy as np
from pantrychef.common.types import Recipe
from pantrychef.retrieval.index import InvertedIndex
from pantrychef.recommender.config import RecConfig
from pantrychef.recommender.query_sim import make_query,is_train,sample_order
from pantrychef.recommender.recommend import candidate_pool
rs=[];it={}
with open('data/processed/recipes.jsonl',encoding='utf8') as f:
    for i,l in enumerate(f):
        if i%64: continue
        r=Recipe.model_validate_json(l); r.canonical=[it.setdefault(c,c) for c in r.canonical]; r.ingredients_raw=[]; rs.append(r)
idx=InvertedIndex.build(rs); df=collections.Counter(c for r in rs for c in set(r.canonical)); N=len(rs)
vocab=list(df); w=np.array([df[c] for c in vocab],float); w/=w.sum(); idf=lambda c:math.log(1+N/df[c]) if c in df else 0
def dq(d,recipes,cap,seed):
    g=random.Random(seed); npg=np.random.default_rng(seed); out=[]
    for r in sample_order(recipes,seed):
        if len(out)>=cap: break
        q=make_query(r,RecConfig(seed=13),g)
        if not q: continue
        P=set(q.pantry); c=set(r.canonical)
        if d:
            for x in npg.choice(len(vocab),d*3,p=w):
                if len(P)>=len(q.pantry)+d: break
                if vocab[x] not in c: P.add(vocab[x])
        out.append((P,q.gold_id,candidate_pool(idx,P,200)))
    return out
R=dict(
 matched_then_missing=lambda P,c:(-len(P&c),len(c-P)),
 cosine=lambda P,c:(-len(P&c)/math.sqrt(len(P)*len(c)),),
 jaccard=lambda P,c:(-len(P&c)/len(P|c),),
 idf_cos=lambda P,c:(-sum(idf(x) for x in P&c)/math.sqrt(sum(idf(x) for x in c)+1e-9),),
 gen_aware=lambda P,c:(-len(P&c),abs(len(c-P)-max(1,round(.3*len(c))))),
)
test=[r for r in rs if not is_train(r.recipe_id)]
for d in (3,10,20):
    Q=dq(d,test,1000,101+d); o={}
    for k,f in R.items():
        m=0
        for P,g,pool in Q:
            ids=[r.recipe_id for r in sorted(pool,key=lambda r:(f(P,idx.canon_sets[r.recipe_id]),r.recipe_id))[:10]]
            if g in ids: m+=1/(ids.index(g)+1)
        o[k]=round(m/len(Q),4)
    print('d',d,o,flush=True)
