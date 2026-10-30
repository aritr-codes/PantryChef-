import csv,collections,hashlib
csv.field_size_limit(10**9)
c=collections.Counter(); seen=set(); kept=collections.Counter(); blank=0
for i,row in enumerate(csv.reader(open('data/raw/full_dataset.csv',encoding='utf8',newline=''))):
    if i==0: continue
    c[row[5]]+=1
    t=row[1].strip().lower()
    if not t: blank+=1; continue
    h=hashlib.md5(t.encode()).digest()[:8]
    if h not in seen: seen.add(h); kept[row[5]]+=1
print('raw',sum(c.values()),dict(c),'blank',blank,'kept',sum(kept.values()),dict(kept))
