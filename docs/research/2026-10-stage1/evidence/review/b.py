import json,numpy as np
d=json.load(open('docs/research/2026-10-stage1/evidence/perquery.json'))
g=np.random.default_rng(1)
for a,b in (('graph','emb'),('hybrid','emb'),('hybrid','graph')):
    x=np.array(d[a])-np.array(d[b]); m=g.choice(x,(10000,len(x))).mean(1)
    print(a,b,round(x.mean(),3),np.quantile(m,[.025,.975]).round(3),round((m<=0).mean(),3))
for k in ('emb','graph','hybrid'): print(k,round(float(np.mean(d[k])),4))
