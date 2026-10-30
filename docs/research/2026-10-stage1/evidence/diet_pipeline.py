import json, sys, gc, numpy as np
from scipy import sparse
from pathlib import Path
from pantrychef.ingredients.normalize import canonicalize
from pantrychef.substitution.substitute import rank_blend
from pantrychef.dietary import DietTagger
B=Path(sys.argv[1])
vocab=json.loads((B/"graph_vocab.json").read_text()); evoc=json.loads((B/"embedding_vocab.json").read_text())
idx={w:i for i,w in enumerate(vocab)}; eidx={w:i for i,w in enumerate(evoc)}
T=DietTagger(known=vocab)
for w in ["soy milk","coconut cream","cream of tartar","peanut butter","cocoa butter","egg noodle","egg roll wrapper","chicken noodle soup","cream of mushroom soup","beer","rice","coconut oil","margarine","shortening","vegetable shortening","nutritional yeast","oat milk","cashew","almond","peanut","walnut","sesame seed","tahini","soy","shellfish","mayonnaise","soy sauce","tamari"]:
    print(f"{w:24s} tags={sorted(T.tags(w))} known={T._is_known(w)} vegan_ok={T.is_valid(w,'vegan')} gf_ok={T.is_valid(w,'gluten_free')} df_ok={T.is_valid(w,'dairy_free')}")
Q=[("butter","vegan"),("milk","vegan"),("egg","vegan"),("heavy cream","dairy_free"),("butter","dairy_free"),("flour","gluten_free"),("soy sauce","gluten_free"),("bread crumb","gluten_free"),("pasta","gluten_free"),("cream","vegan"),("chicken broth","vegetarian"),("sour cream","vegan")]
qs=[canonicalize(q) for q,_ in Q]
C=sparse.load_npz(B/"graph_cooccur.npz").tocsr()
crow={idx[q]:np.asarray(C[idx[q]].toarray()).ravel().astype(np.float32) for q in qs}
del C; gc.collect()
M=sparse.load_npz(B/"graph_norm.npz").tocsr(); M.data=M.data.astype(np.float32)
P=sparse.csr_matrix((np.ones(M.nnz,dtype=np.float32),M.indices,M.indptr),shape=M.shape)
V=np.load(B/"embedding_vectors.npy").astype(np.float32); V/=np.maximum(np.linalg.norm(V,axis=1,keepdims=True),1e-9)
def g(i,k):
    s=np.asarray((M@M[i].T).toarray()).ravel(); ov=np.asarray((P@P[i].T).toarray()).ravel()
    s=s*(ov/(ov+100.0)); cr=crow[i]; cm=cr.max(); s=s-0.5*(cr/cm if cm>0 else cr); s[i]=-np.inf
    o=np.argsort(-s,kind="stable")[:k]; return [(vocab[j],float(s[j])) for j in o]
def e(q,k):
    i=eidx[q]; s=V@V[i]; s[i]=-np.inf; o=np.argsort(-s,kind="stable")[:k]; return [(evoc[j],float(s[j])) for j in o]
print()
for (q,d),cq in zip(Q,qs):
    pool=20; E=e(cq,pool); G=g(idx[cq],pool)
    H=[(a,b) for a,b in rank_blend(E,G,0.5,pool) if canonicalize(a)!=cq]
    kept=[a for a,_ in H if T.is_valid(a,d)]
    print(f"{q} [{d}] pool={len(H)} kept={len(kept)} -> {kept[:5]}\n    rejected-sample: {[a for a,_ in H if not T.is_valid(a,d)][:6]}")
