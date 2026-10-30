import json,time,sys,numpy as np
from scipy import sparse
from pathlib import Path
B=Path(sys.argv[1]); M=sparse.load_npz(B/"graph_norm.npz").tocsr(); M.data=M.data.astype(np.float32)
P=sparse.csr_matrix((np.ones(M.nnz,dtype=np.float32),M.indices,M.indptr),shape=M.shape)
print("nnz",M.nnz, "mean nnz/row",M.nnz/M.shape[0])
rng=np.random.default_rng(1); ids=rng.integers(0,3000,20)  # common ingredients first in vocab? sample low idx
t=time.time()
for i in ids:
    s=np.asarray((M@M[i].T).toarray()).ravel(); ov=np.asarray((P@P[i].T).toarray()).ravel()
print("per query s", (time.time()-t)/20)
t=time.time(); X=(M@M[:200].T); print("batch200 matmul s", time.time()-t, "nnz",X.nnz)
