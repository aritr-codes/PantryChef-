import json,numpy as np,csv,sys,ast
d=json.load(open('docs/research/2026-10-stage1/evidence/perquery.json',encoding='utf8'))
print(type(d), list(d)[:5] if isinstance(d,dict) else d[:2])
def arr(x): return np.array([x[k] for k in sorted(x)],float) if isinstance(x,dict) else np.array(x,float)
if isinstance(d,dict):
    keys=list(d)
    if all(isinstance(d[k],(list,dict)) for k in keys):
        for k in keys: print(k,len(d[k]))
s=json.load(open('docs/research/2026-10-stage1/evidence/subs_out.json',encoding='utf8'))
print('subs',type(s),len(s), list(s)[:50] if isinstance(s,dict) else s[:1])
csv.field_size_limit(10**9)
f=open('data/raw/full_dataset.csv',encoding='utf8',newline=''); r=csv.reader(f); h=next(r); print(h)
seen={}; dup=same=j8=j5=0; N=200000
for i,row in enumerate(r):
    if i>=N: break
    t=row[1].strip().lower(); ner=frozenset(ast.literal_eval(row[-1]))
    if t in seen:
        dup+=1; a=seen[t]; u=len(a|ner); jac=len(a&ner)/u if u else 1
        same+=a==ner; j8+=jac>=.8; j5+=jac>=.5
    else: seen[t]=ner
print('rows',N,'title dups',dup,'same NER set',same,'jac>=.8',j8,'jac>=.5',j5)
