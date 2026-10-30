import json, sys, gc, numpy as np
from scipy import sparse
from pathlib import Path
from pantrychef.ingredients.normalize import canonicalize
from pantrychef.substitution.substitute import rank_blend
from pantrychef.dietary import DietTagger
from pantrychef.eval.subs_gold import load_pairs_csv
B=Path(sys.argv[1]); gold=load_pairs_csv(sys.argv[2])
vocab=json.loads((B/"graph_vocab.json").read_text()); evoc=json.loads((B/"embedding_vocab.json").read_text())
idx={w:i for i,w in enumerate(vocab)}; eidx={w:i for i,w in enumerate(evoc)}
V=np.load(B/"embedding_vectors.npy").astype(np.float32); V/=np.maximum(np.linalg.norm(V,axis=1,keepdims=True),1e-9)
vs=set(vocab)
qs=[q for q,g in gold.items() if q in vs and any(x in vs for x in g)]
print("n queries",len(qs))
C=sparse.load_npz(B/"graph_cooccur.npz").tocsr()
crow={idx[q]:np.asarray(C[idx[q]].toarray()).ravel().astype(np.float32) for q in qs}
del C; gc.collect()
M=sparse.load_npz(B/"graph_norm.npz").tocsr(); M.data=M.data.astype(np.float32)
P=sparse.csr_matrix((np.ones(M.nnz,dtype=np.float32),M.indices,M.indptr),shape=M.shape)
def g(i,k):
    s=np.asarray((M@M[i].T).toarray()).ravel(); ov=np.asarray((P@P[i].T).toarray()).ravel()
    s=s*(ov/(ov+100.0)); cr=crow[i]; cm=cr.max(); s=s-0.5*(cr/cm if cm>0 else cr); s[i]=-np.inf
    o=np.argsort(-s,kind="stable")[:k]; return [(vocab[j],float(s[j])) for j in o if np.isfinite(s[j])]
def e(q,k):
    if q not in eidx: return []
    i=eidx[q]; s=V@V[i]; s[i]=-np.inf; o=np.argsort(-s,kind="stable")[:k]; return [(evoc[j],float(s[j])) for j in o]
def rr(p,G):
    for r,x in enumerate(p,1):
        if x in G: return 1/r
    return 0.0
rows={"emb":[],"graph":[],"hybrid":[]}; rec={"emb":[],"graph":[],"hybrid":[]}
for q in qs:
    G={x for x in gold[q] if x in vs}; i=idx[q]; pool=40
    E=e(q,pool); Gr=g(i,pool)
    lists={"emb":E[:10],"graph":Gr[:10]}
    H=[(a,b) for a,b in rank_blend(E,Gr,0.5,pool) if canonicalize(a)!=q][:10]
    lists["hybrid"]=H
    for a,l in lists.items():
        p=[w for w,_ in l]; rows[a].append(rr(p,G)); rec[a].append(len(set(p)&G)/len(G))
R={a:np.array(v) for a,v in rows.items()}
for a in R: print(a,"MRR %.4f"%R[a].mean(),"R@10 %.4f"%np.mean(rec[a]))
rng=np.random.default_rng(0); n=len(qs); ix=rng.integers(0,n,(10000,n))
def ci(a,b):
    d=(R[a]-R[b])[ix].mean(1); return d.mean(), np.percentile(d,2.5), np.percentile(d,97.5), (d<=0).mean()
for a,b in [("graph","emb"),("hybrid","emb"),("hybrid","graph")]:
    print(a,"-",b,"diff %.3f CI [%.3f, %.3f] P(diff<=0)=%.3f"%ci(a,b))
json.dump({"queries":qs,**{a:list(map(float,v)) for a,v in rows.items()}},open(sys.argv[3],"w"))
# which gold targets are variants?
var=0;tot=0
for q in qs:
    for x in gold[q]:
        tot+=1; var+= set(q.split())<=set(x.split()) or set(x.split())<=set(q.split())
print("gold pairs where one name token-contains the other:",var,"/",tot)
