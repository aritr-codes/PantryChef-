import json, sys
from pantrychef.dietary import DietTagger
B=sys.argv[1]
vocab=json.load(open(B+"/graph_vocab.json"))
t=DietTagger(known=vocab)
print("coverage over 30481 vocab:", round(t.coverage(vocab),4))
unj=[w for w in vocab if not t._is_known(w)]
print("judged:", len(vocab)-len(unj), "unjudged:", len(unj))
cases={
 "vegan":["butter beans","peanut butter","almond milk","coconut milk","buttermilk","eggplant","worcestershire sauce","gelatin","chicken stock","chicken broth","fish sauce","oyster sauce","parmesan cheese","mayonnaise","honey","sugar","beef","lard","rennet","anchovy paste","cheese","egg noodle","mayo","miracle whip","caesar dressing","shrimp","clam juice","bone broth","tuna","salmon fillet","milk chocolate","ice cream","yoghurt","paneer","tofu","olive oil","margarine","shortening","oyster","scallop","squid","mussel","duck","goose","venison","prosciutto","pepperoni","salami","chorizo","bouillon cube","pancetta","hot dog"],
 "gluten_free":["soy sauce","tamari","flour","almond flour","oat","rolled oat","bread crumb","breadcrumb","panko","beer","malt vinegar","soy sauce light","teriyaki sauce","hoisin sauce","worcestershire sauce","cornstarch","spaghetti","penne","macaroni","noodle","tortilla","cracker","cracker crumb","biscuit","bulgur","farro","spelt","seitan","orzo","udon","ramen noodle","lasagna","fettuccine","cake flour","self rising flour","bisquick mix","gravy mix","cream of mushroom soup","cream of chicken soup","pita","bun","couscous","rice","corn tortilla"],
 "dairy_free":["butter","peanut butter","cocoa butter","buttermilk","almond milk","coconut milk","cream cheese","sour cream","evaporated milk","condensed milk","sweetened condensed milk","chocolate chip","milk chocolate","margarine","half and half","whipped topping","cool whip","ice cream","custard","cheese","yoghurt","yogurt","paneer","lactose free milk","kefir","creme fraiche","mascarpone","ricotta","parmesan","feta","mozzarella","brie","cottage cheese","evaporated milk","dulce de leche","white chocolate","butterscotch chip","milk powder","nonfat dry milk","cheddar","provolone","queso fresco","gruyere"],
 "vegetarian":["chicken stock","gelatin","lard","worcestershire sauce","fish sauce","oyster sauce","anchovy paste","bacon bit","pepperoni","prosciutto","salami","chorizo","bouillon cube","beef bouillon","shrimp paste","clam juice","caesar dressing","rennet","bone broth","duck","venison","pork rind","suet","tallow","hot dog","pancetta","oyster","scallop","squid","mussel","lobster","crab","crabmeat","tuna","sardine","mackerel","trout","haddock","tilapia"],
}
for diet,L in cases.items():
    fails=[];ok=[]
    for w in L:
        v=t.is_valid(w,diet)
        ok.append((w,v))
    print(f"\n== {diet}: ALLOWED by guardrail (potential violations to hand-check):")
    print([w for w,v in ok if v])
    print("== rejected:",[w for w,v in ok if not v])
