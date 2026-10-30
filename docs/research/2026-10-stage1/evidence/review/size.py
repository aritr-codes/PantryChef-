import json,gzip,collections
post=collections.defaultdict(list); fw=[]; titles=[]; raw=[]; n=0; ids={}
with open('data/processed/recipes.jsonl',encoding='utf8') as f:
    for i,l in enumerate(f):
        if i%13: continue
        d=json.loads(l); titles.append(d['title']); raw.append('|'.join(d['ingredients_raw']))
        c=[ids.setdefault(x,len(ids)) for x in set(d['canonical'])]; fw.append(c)
        for x in c: post[x].append(n)
        n+=1
def vb(xs):
    b=bytearray()
    for x in xs:
        while x>=128: b.append(x&127|128); x>>=7
        b.append(x)
    return bytes(b)
P=b''.join(vb([a-b for a,b in zip(p,[0]+p[:-1])]) for p in post.values())
F=b''.join(vb([len(c)]+sorted(c)) for c in fw)
z=lambda b:len(gzip.compress(b,6))
tp,ff,tt,rr=z(P),z(F),z('\n'.join(titles).encode()),z('\n'.join(raw).encode())
print('recipes',n,'mean canon',sum(map(len,fw))/n,'postings',len(P),tp,'forward',ff,'titles',tt,'raw',rr)
print('core MB post+fwd+titles',(tp+ff+tt)/1e6*100000/n,'raw MB per100k',rr/1e6*100000/n,'B/recipe',tp/n,ff/n,tt/n)
