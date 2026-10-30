import json, sys, gc, numpy as np
from scipy import sparse
from pathlib import Path
from pantrychef.ingredients.normalize import canonicalize
from pantrychef.substitution.substitute import rank_blend
from pantrychef.dietary import DietTagger
B = Path(sys.argv[1]); out = sys.argv[2]
QS = ["butter","milk","egg","flour","soy sauce","onion","chicken breast","lemon juice","yeast","baking soda","baking powder","cream","tomato paste","rice","sugar","brown sugar","honey","olive oil","vegetable oil","sour cream","buttermilk","heavy cream","cornstarch","garlic","parsley","basil","cilantro","white wine","vinegar","mayonnaise","cheddar cheese","parmesan cheese","bacon","ground beef","salt","black pepper","vanilla extract","cocoa powder","lime juice","shallot","chicken broth","coconut milk","cream cheese","bread crumb","pasta","potato","spinach","maple syrup","tofu","worcestershire sauce"]
vocab = json.loads((B/"graph_vocab.json").read_text())
evoc = json.loads((B/"embedding_vocab.json").read_text())
eidx={w:i for i,w in enumerate(evoc)}
idx = {w:i for i,w in enumerate(vocab)}
V = np.load(B/"embedding_vectors.npy").astype(np.float32)
V /= np.maximum(np.linalg.norm(V,axis=1,keepdims=True),1e-9)
qs = [canonicalize(q) for q in QS]
qi = [idx[q] for q in qs if q in idx]
miss = [q for q in qs if q not in idx]
C = sparse.load_npz(B/"graph_cooccur.npz").tocsr()
crow = {i: np.asarray(C[i].toarray()).ravel().astype(np.float32) for i in qi}
del C; gc.collect()
M = sparse.load_npz(B/"graph_norm.npz").tocsr()
M.data = M.data.astype(np.float32)
P = sparse.csr_matrix((np.ones(M.nnz,dtype=np.float32), M.indices, M.indptr), shape=M.shape)
def gneigh(i,k):
    sim = np.asarray((M @ M[i].T).toarray()).ravel()
    ov = np.asarray((P @ P[i].T).toarray()).ravel()
    sim = sim*(ov/(ov+100.0))
    cr = crow[i]; cm = cr.max(); pen = cr/cm if cm>0 else cr
    s = sim-0.5*pen; s[i]=-np.inf
    o = np.argsort(-s,kind="stable")[:k]
    return [(vocab[j],float(s[j])) for j in o if np.isfinite(s[j])], cr
def eneigh(q,k):
    if q not in eidx: return []
    i=eidx[q]; s = V@V[i]; s[i]=-np.inf
    o = np.argsort(-s,kind="stable")[:k]
    return [(evoc[j],float(s[j])) for j in o]
res = {}
for q in qs:
    if q not in idx: continue
    i = idx[q]
    g,cr = gneigh(i,20); e = eneigh(q,20)
    h = rank_blend(e,g,0.5,5)
    res[q] = {"emb":[w for w,_ in e[:5]],"graph":[w for w,_ in g[:5]],"hybrid":[w for w,_ in h],
              "hybrid_scores":[round(s,3) for _,s in h],
              "n_cooccur_partners":int((cr>0).sum()),"self_cooccur_max":float(cr.max()),
              "top_cooccur":[vocab[j] for j in np.argsort(-cr)[:5]]}
res["_missing"]=miss
json.dump(res, open(out,"w"), indent=1)
print("missing",miss)
for q,v in res.items():
    if q!="_missing": print(f"{q:18s} H: {v['hybrid']}\n{'':18s} E: {v['emb']}\n{'':18s} G: {v['graph']}")
