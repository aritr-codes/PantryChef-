import json,zipfile,numpy as np
from pantrychef.data.store import load_recipes
from pantrychef.retrieval.index import InvertedIndex
from pantrychef.recommender.config import RecConfig
from pantrychef.recommender.rank import LambdaMARTRanker
from pantrychef.recommender.features import extract_features,FEATURE_NAMES
from pantrychef.eval.recommend_eval import build_eval_queries,score_queries
rs=load_recipes('data/processed/recipes.jsonl',limit=20000); idx=InvertedIndex.build(rs)
EQ=build_eval_queries(rs,idx,RecConfig(seed=13,max_eval_queries=1500))
z=zipfile.ZipFile('models/recommender/recommender_bundle.zip')
class Rule:
    def score(self,X): return X[:,1]*1000+X[:,0]
class Ov:
    def score(self,X): return X[:,0]*1000-X[:,3]
print('rule',{k:round(v,4) for k,v in score_queries(Rule(),EQ,idx,None,columns=FEATURE_NAMES).items()})
for nm in ('lambdamart-nosub',):
    st=json.loads(z.read(f'models/{nm}.json')); print(list(st)[:8])
    s=st.get('state',st); m=LambdaMARTRanker.from_state(s if 'booster' in s else st['model'])
    cols=tuple(c for c in FEATURE_NAMES if not c.startswith('sub_fill'))
    print(nm,{k:round(v,4) for k,v in score_queries(m,EQ,idx,None,columns=cols).items()})
ov=mr=n=0
for P,g,pool,ip in EQ:
    ids=[r.recipe_id for r in pool[:10]]; n+=1
    if g in ids: ov+=1; mr+=1/(ids.index(g)+1)
print('overlap rec',ov/n,'mrr',mr/n,n)
