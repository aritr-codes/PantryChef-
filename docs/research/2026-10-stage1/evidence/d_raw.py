import csv,sys,json,collections
csv.field_size_limit(10**9)
f=open('data/raw/full_dataset.csv',encoding='utf8',newline='')
r=csv.reader(f); h=next(r); print(h)
N=300000; seen={}; dup=0; same_ing=0; same_dir=0; ex=[]; tc=collections.Counter(); srcs=collections.Counter()
for i,row in enumerate(r):
    if i>=N: break
    t=row[1].strip().lower(); srcs[row[5]]+=1
    tc[t]+=1
    if t in seen:
        dup+=1
        if seen[t][0]==row[2]: same_ing+=1
        elif len(ex)<6: ex.append((t,seen[t][0][:90],row[2][:90]))
    else: seen[t]=(row[2],row[3])
print('rows',N,'title-dups',dup,'of which identical ingredient list',same_ing)
print(srcs); print(tc.most_common(8)); 
for e in ex: print(e)
