import json
from pantrychef.dietary import DietTagger
from pantrychef.ingredients.vocab import load_vocabulary
v=load_vocabulary('data/processed/vocab.json'); t=DietTagger(known=v)
# truth: set of diets it VIOLATES among vegan, vegetarian, gluten_free, dairy_free
V,VG,GF,DF='vegan','vegetarian','gluten_free','dairy_free'
truth={
"butter bean":set(),"peanut butter":{"x"} and set(),"coconut milk":set(),"cream of tartar":set(),"eggplant":set(),"egg noodle":{V,GF,'egg'},"worcestershire sauce":{V,VG},
"gelatin":{V,VG},"honey":{V},"fish sauce":{V,VG},"anchovy paste":{V,VG},"chicken stock":{V,VG},"soy sauce":{GF},"seitan":{GF},"oat flour":set(),"malt vinegar":{GF},
"marshmallow":{V,VG},"parmesan":{V,DF},"parmesan cheese":{V,DF},"lard":{V,VG},"beef bouillon":{V,VG},"caesar dressing":{V,VG,DF},
"almond milk":set(),"soy milk":set(),"cocoa butter":set(),"shea butter":set(),"butternut squash":set(),"buttermilk":{V,DF},"mayonnaise":{V},"ice cream":{V,DF},"milk chocolate":{V,DF},
"salt":set(),"onion":set(),"garlic":set(),"olive oil":set(),"rice":set(),"tomato":set(),"pepper":set(),"sugar":set(),"water":set(),"flour":{GF},"all purpose flour":{GF},"bread crumb":{GF},"spaghetti":{GF},"pasta":{GF},
"chicken breast":{V,VG},"ground beef":{V,VG},"bacon":{V,VG},"pork chop":{V,VG},"shrimp":{V,VG},"tuna":{V,VG},"salmon fillet":{V,VG},"egg":{V},"eggs":{V},"yogurt":{V,DF},"cheddar cheese":{V,DF},"cream cheese":{V,DF},"sour cream":{V,DF},"heavy cream":{V,DF},
"whipped cream":{V,DF},"ghee":{V,DF},"margarine":set(),"mozzarella":{V,DF},"feta":{V,DF},"ricotta":{V,DF},"cheddar":{V,DF},"cottage cheese":{V,DF},"half and half":{V,DF},
"tofu":set(),"tempeh":set(),"lentil":set(),"chickpea":set(),"quinoa":set(),"oat":set(),"rolled oat":set(),"wheat flour":{GF},"whole wheat bread":{GF},"couscous":{GF},"barley":{GF},"rye bread":{GF},"corn tortilla":set(),"flour tortilla":{GF},"tortilla":set(),
"beer":{GF},"soy sauce low sodium":{GF},"tamari":set(),"hoisin sauce":{GF},"teriyaki sauce":{GF},"miso":set(),"bouillon cube":set(),"vegetable broth":set(),"chicken broth":{V,VG},"fish":{V,VG},"crab meat":{V,VG},"oyster sauce":{V,VG},"clam juice":{V,VG},
"turkey":{V,VG},"ham":{V,VG},"sausage":{V,VG},"pepperoni":{V,VG},"prosciutto":{V,VG},"salami":{V,VG},"chorizo":{V,VG},"lamb":{V,VG},"venison":{V,VG},"duck":{V,VG},"hot dog":{V,VG},"steak":{V,VG},"mince":{V,VG},
"whey protein":{V,DF},"casein":{V,DF},"cake mix":{GF},"graham cracker":{GF},"croutons":{GF},"cornstarch":set(),"cornmeal":set(),"semolina":{GF},"farro":{GF},"bulgur":{GF},"noodle":{GF},"ramen noodle":{GF},"rice noodle":set(),
"mushroom":set(),"spinach":set(),"broccoli":set(),"carrot":set(),"lemon juice":set(),"vanilla extract":set(),"baking powder":set(),"yeast":set(),"vinegar":set(),"mustard":set(),"ketchup":set(),"hot sauce":set(),"maple syrup":set(),"agave":set(),"coconut oil":set(),"coconut cream":set(),"cashew":set(),"almond":set(),"walnut":set(),
"rennet":{V,VG},"isinglass":{V,VG},"cochineal":{V,VG},"shellac":{V},"bone broth":{V,VG},"beef stock":{V,VG},"suet":{V,VG},"tallow":{V,VG},"fat free milk":{V,DF},"skim milk":{V,DF},"evaporated milk":{V,DF},"condensed milk":{V,DF},"buttercream":{V,DF},"custard":{V,DF},"pudding":set(),
}
diets=[V,VG,GF,DF]
fn=fp=ok=blocked_safe=0; rows=[]
# only evaluate non-"x" entries
agg={d:{'tp':0,'fp':0,'fn':0,'unjudged_safe':0,'tn':0,'unjudged_viol':0} for d in diets}
for ing,viol in truth.items():
    for d in diets:
        allowed=t.is_valid(ing,d)
        should_block=d in viol
        a=agg[d]
        if should_block and allowed: a['fn']+=1; rows.append(('UNSAFE-ALLOWED',ing,d))
        elif should_block and not allowed: a['tp']+=1
        elif not should_block and allowed: a['tn']+=1
        else:
            a['fp']+=1
            if not t._is_known(ing): a['unjudged_safe']+=1
            else: rows.append(('WRONG-BLOCK',ing,d))
print(len(truth),'ingredients')
for d,a in agg.items(): print(d,a)
for r in rows: print(r)
print('unjudged-and-safe-blocked:',[(i) for i in truth if not truth[i] and not t._is_known(i)][:60])
# vocab-level
unj=[w for w in v if not t._is_known(w)]
print('vocab',len(v),'judged',len(v)-len(unj),'coverage',round(1-len(unj)/len(v),4))
import collections
occ=json.load(open('occ.json')) if False else None
