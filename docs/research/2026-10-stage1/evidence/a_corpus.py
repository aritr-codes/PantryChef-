import json,re,collections,hashlib,random,sys
P='data/processed/recipes.jsonl'
random.seed(0)
n=0; occ=collections.Counter(); setc=collections.Counter(); titc=collections.Counter(); ttc=collections.Counter()
empty=0; sizes=[]; sample=[]
for line in open(P,encoding='utf8'):
    r=json.loads(line); n+=1
    c=r['canonical']; 
    if not c: empty+=1
    sizes.append(len(c))
    occ.update(c)
    k=hash(tuple(sorted(c))); setc[k]+=1
    t=re.sub(r'[^a-z0-9]','',r['title'].lower()); titc[hash(t)]+=1
    ttc[hash((t,tuple(sorted(c))))]+=1
    if n%64==0: sample.append((r['title'],c,r['ingredients_raw']))
print('n',n,'empty canon',empty)
import statistics
print('median canon size',statistics.median(sizes))
print('distinct canonical sets',len(setc),'recipes in repeated sets',sum(v for v in setc.values() if v>1), 'excess',n-len(setc))
print('distinct alnum titles',len(titc),'excess',n-len(titc))
print('distinct (title,set)',len(ttc),'excess',n-len(ttc))
tot=sum(occ.values()); cum=0
cuts={0.5:0,0.8:0,0.9:0,0.95:0,0.99:0}
for i,(w,v) in enumerate(occ.most_common(),1):
    cum+=v
    for q in list(cuts):
        if cuts[q]==0 and cum/tot>=q: cuts[q]=i
print('distinct canon used',len(occ),'total occ',tot,'cover cutoffs',cuts)
json.dump(occ.most_common(),open('C:/Users/HP/AppData/Local/Temp/claude/g--Books-Python-ACTUAL-CODES-PROJECTS-pantrychef/1127416b-d672-4b96-83c2-8570c428ce9d/scratchpad/research/occ.json','w'))
json.dump(sample,open('C:/Users/HP/AppData/Local/Temp/claude/g--Books-Python-ACTUAL-CODES-PROJECTS-pantrychef/1127416b-d672-4b96-83c2-8570c428ce9d/scratchpad/research/sample20k.json','w'))
print('sample',len(sample))
print(occ.most_common(60))
