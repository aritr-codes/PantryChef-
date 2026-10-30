# usage: rec_test.py STEP CAP1,CAP2,...   (from repo root). Independent reviewer test.
import sys, json, math, random, collections, hashlib, time
import numpy as np
from pantrychef.common.types import Recipe
from pantrychef.retrieval.index import InvertedIndex
from pantrychef.recommender.config import RecConfig
from pantrychef.recommender.query_sim import make_query, is_train, sample_order
from pantrychef.recommender.recommend import candidate_pool, rerank
from pantrychef.recommender.features import extract_features, extract_feature_row
from pantrychef.recommender.train import train_production_bundle, NO_SUB_COLUMNS, _column_indices
from pantrychef.recommender.rank import LambdaMARTRanker
from pantrychef.eval.recommend_eval import build_eval_queries, mrr_at_k, recall_at_k
STEP=int(sys.argv[1]); CAPS=[int(x) for x in sys.argv[2].split(',')]; PREFIX=len(sys.argv)>3 and sys.argv[3]=='prefix'
rs=[]; it={}
with open('data/processed/recipes.jsonl',encoding='utf8') as f:
    for i,l in enumerate(f):
        if PREFIX:
            if i>=STEP: break
        elif i%STEP: continue
        r=Recipe.model_validate_json(l); r.canonical=[it.setdefault(c,c) for c in r.canonical]; r.ingredients_raw=[]; rs.append(r)
idx=InvertedIndex.build(rs); print('recipes',len(rs),flush=True)
df=collections.Counter(c for r in rs for c in set(r.canonical)); N=len(rs)
idf=lambda c: math.log(1+N/df[c]) if df.get(c) else 0
vocab=list(df); w=np.array([df[c] for c in vocab],float); w/=w.sum()
cfg=RecConfig(seed=13,max_eval_queries=2000)
EQ=build_eval_queries(rs,idx,cfg); print('eval',len(EQ),'ceiling',sum(q[3] for q in EQ)/len(EQ),flush=True)
# no-ML rules (mine). Each returns sort key (ascending) given P, canon.
def rA(P,c): return (-len(P&c)/len(P), len(c-P))               # mine: pantry fully used, then fewest missing
def rB(P,c):                                                   # 01's generator-aware rule
    L=len(c); return (not P<=c, abs(L-len(P&c)-max(1,round(.3*L))))
def rC(P,c): return (-len(P&c)/len(P), -len(P&c)/len(c))        # 07's rule as a tuple
def rI(P,c):                                                   # IDF overlap, no leak use
    m=sum(idf(x) for x in P&c); t=sum(idf(x) for x in c); return (-m/t if t else 0,)
def rP(P,c): return (-len(P&c)/len(P),)                         # only the leak feature (tie -> id)
RULES=dict(ruleA=rA,ruleB=rB,ruleC=rC,idfcov=rI,leakonly=rP)
def ranks_rule(rule,Q):
    out=[]
    for P,g,pool,_ in Q:
        o=sorted(pool,key=lambda r:(rule(P,idx.canon_sets[r.recipe_id]),r.recipe_id))
        ids=[r.recipe_id for r in o[:10]]; out.append(ids.index(g)+1 if g in ids else 0)
    return np.array(out)
def ranks_pool(Q):
    out=[]
    for P,g,pool,_ in Q:
        ids=[r.recipe_id for r in pool[:10]]; out.append(ids.index(g)+1 if g in ids else 0)
    return np.array(out)
def ranks_model(m,cols,Q):
    out=[];tops=[]
    fn=lambda p,r:extract_features(p,r,None,canon=idx.canon_sets[r.recipe_id])
    for P,g,pool,_ in Q:
        ids=[x.recipe_id for x in rerank(idx,P,m,fn,k=10,columns=cols,pool=pool)]
        out.append(ids.index(g)+1 if g in ids else 0); tops.append(tuple(ids))
    return np.array(out),tops
def met(r): return dict(mrr=round(float((1/np.where(r>0,r,np.inf)).mean()),4),rec=round(float((r>0).mean()),4))
def rr(r): return 1/np.where(r>0,r,np.inf)
def boot(a,b,n=5000,s=7):
    d=rr(a)-rr(b); g=np.random.default_rng(s); m=g.choice(d,(n,len(d))).mean(1)
    return round(float(d.mean()),4),round(float(np.quantile(m,.025)),4),round(float(np.quantile(m,.975)),4)
base=ranks_pool(EQ); print('overlap',met(base))
RR={k:ranks_rule(v,EQ) for k,v in RULES.items()}
for k,v in RR.items(): print(k,met(v))
print('supersets in pool per query',np.mean([sum(P<=idx.canon_sets[r.recipe_id] for r in pool) for P,_,pool,_ in EQ]))
models={};prev=None
for cap in CAPS:
    t=time.time(); c=RecConfig(seed=13,max_train_queries=cap)
    b,st=train_production_bundle(rs,idx,c); ra=b.models['lambdamart-nosub']; m=ra.model
    r,tops=ranks_model(m,ra.columns,EQ); s=m._model.booster_.model_to_string()
    imp=dict(zip(ra.columns,m._model.booster_.feature_importance('gain').round(0).tolist()))
    print('cap',cap,'kept',st['n_queries_kept'],met(r),'mrr_exact',repr(float(rr(r).mean())),'sec',round(time.time()-t),'gain',imp,flush=True)
    if prev:
        print('  vs prev: model-strings-equal',s==prev[2],'gold ranks equal',int((r==prev[0]).sum()),'/',len(r),'top10 lists equal',sum(x==y for x,y in zip(tops,prev[1])))
    prev=(r,tops,s); models[cap]=ra
last=models[CAPS[-1]]; rl=prev[0]
for k in RR: print('LM-',k,'dMRR, CI',boot(rl,RR[k]),'gold-rank-diff queries',int((rl!=RR[k]).sum()))
print('LM-overlap',boot(rl,base))
# Distractors: pantry + d extras sampled by corpus frequency (basket-like)
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
        pool=candidate_pool(idx,P,200); out.append((P,q.gold_id,pool,q.gold_id in {x.recipe_id for x in pool}))
    return out
test=[r for r in rs if not is_train(r.recipe_id)]; train=[r for r in rs if is_train(r.recipe_id)]
for d in (0,3,5,10,20):
    Q=dq(d,test,1000,101+d); rm,_=ranks_model(last.model,last.columns,Q)
    # retrain on the same distractor distribution
    TQ=dq(d,train,1500,202+d); X=[];Y=[];G=[]
    for P,g,pool,ip in TQ:
        if not ip: continue
        for r in pool:
            X.append(extract_feature_row(P,r,None,canon=idx.canon_sets[r.recipe_id])); Y.append(r.recipe_id==g)
        G.append(len(pool))
    Xa=np.array(X)[:,_column_indices(NO_SUB_COLUMNS)]; m2=LambdaMARTRanker(seed=13).fit(Xa,np.array(Y,int),G); del X,Y,Xa
    r2,_=ranks_model(m2,NO_SUB_COLUMNS,Q)
    print('d',d,'ceil',round(sum(q[3] for q in Q)/len(Q),3),'overlap',met(ranks_pool(Q)),'ruleA',met(ranks_rule(rA,Q)),'idf',met(ranks_rule(rI,Q)),'LM-clean',met(rm),'LM-retrained',met(r2),'retrain-idf',boot(r2,ranks_rule(rI,Q)),flush=True)
