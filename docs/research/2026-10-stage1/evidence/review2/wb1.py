import json,re,random,collections,sys
wb=json.load(open(sys.argv[1],encoding='utf8'))
print(len(wb), list(wb[0].keys()), list(wb[0]['recipe_data'].keys()))
print({k:wb[0][k] for k in wb[0] if k!='recipe_data'})
secs=collections.Counter(); lt=collections.Counter()
rec=[];
for it in wb:
    d=it['recipe_data']
    L=[x['text'] for x in d['text_lines'] if (x.get('section') or '').lower().startswith('ingredient') and x['line_type'] in('ul','li')]
    for x in d['text_lines']:
        secs[(x.get('section') or 'NONE').lower()[:30]]+=1; lt[x['line_type']]+=1
    rec.append((d,L))
print(secs.most_common(25)); print(lt)
ok=[r for r in rec if len(r[1])>=2]
print('n',len(ok))
# step line types
st=collections.Counter(); numbered=0; para=0
for d,L in ok:
    S=[x for x in d['text_lines'] if any(k in (x.get('section') or '').lower() for k in('procedure','direction','instruction','method','preparation'))]
    ty={x['line_type'] for x in S}
    st[tuple(sorted(ty))]+=1
print('step types',st.most_common(8))
titles={d['title'] for d,_ in ok}
for t in ['Cookbook:Substitutions','Substitutions','Spanish Potato Omelette','Saffron Rice','Black Forest Cake','Kuddi (Spiced Yogurt Sauce)']:
    print(t, t in titles, [x for x in titles if t.lower() in x.lower()][:4])
# nonrecipe-ish
pat=re.compile(r'(?i)^(how to|cooking|technique|equipment|substitut|history|about|list of|table|measure)')
print('titlelike nonrecipe',[d['title'] for d,_ in ok if pat.search(d['title'])][:30])
cats=collections.Counter(((d.get('infobox') or {}).get('category') or 'none').rsplit(':',1)[-1] for d,_ in ok)
print(cats.most_common(40), len(cats))
# junk in ingredient lines
junk=[l for d,L in ok for l in L if re.search(r'(?i)template:|see |cookbook|\|\||click|http|wikipedia',l)]
print('junk-ish ingredient lines',len(junk),random.Random(1).sample(junk,min(15,len(junk))))
L_all=[l for d,L in ok for l in L]
print('lines',len(L_all))
print('len>80',sum(len(l)>80 for l in L_all),'optional',sum('optional' in l.lower() for l in L_all),'to taste',sum('to taste' in l.lower() for l in L_all),'serving|garnish',sum(bool(re.search(r'(?i)for (serving|garnish)|to serve|garnish',l)) for l in L_all))
print('or',sum(bool(re.search(r'(?i)\bor\b|/',l)) for l in L_all)/len(L_all),'bracket',sum('(' in l for l in L_all)/len(L_all),'nonum',sum(not re.match(r'\s*[\d½¼¾⅓⅔⅛]',l) for l in L_all)/len(L_all))
print('section-header-like (ends colon)',sum(l.strip().endswith(':') for l in L_all))
# non-ascii/non-English signal: title gloss
print('nonascii titles',sum(any(ord(c)>127 for c in d['title']) for d,_ in ok))
# sample 40
rng=random.Random(2026)
for d,L in rng.sample(ok,30):
    S=[x['text'] for x in d['text_lines'] if any(k in (x.get('section') or '').lower() for k in('procedure','direction','instruction','method','preparation'))]
    print('---',d['title'],'|',(d.get('infobox') or {}).get('category','').rsplit(':',1)[-1],'|',len(L),'lines; steps chars',sum(map(len,S)))
    print('   ',[l[:40] for l in L[:5]])
