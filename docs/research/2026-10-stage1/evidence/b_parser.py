import json,sys
sys.path.insert(0,'.')
from pantrychef.ingredients.vocab import load_vocabulary
from pantrychef.ingredients.parser import parse,build_match_index
from pantrychef.ingredients.normalize import canonicalize
v=load_vocabulary('data/processed/vocab.json'); vs=set(v); idx=build_match_index(v)
# (input, acceptable canonical outputs set)  -- expected = what a user would consider right
T=[
("2 cups all-purpose flour",{"all purpose flour","flour"}),("evoo",{"olive oil","extra virgin olive oil"}),("scallions",{"scallion","green onion"}),
("green onions",{"green onion","scallion"}),("tomatos",{"tomato"}),("chicken thighs, boneless",{"chicken thigh"}),("chicken thighs",{"chicken thigh"}),
("1 (14 oz) can diced tomatoes",{"diced tomato","tomato"}),("a pinch of salt",{"salt"}),("half and half",{"half and half","half half"}),("s&p",{"salt pepper"}),
("garbanzo beans/chickpeas",{"garbanzo bean","chickpea"}),("cilantro/coriander",{"cilantro","coriander"}),("heavy cream",{"heavy cream"}),
("powdered sugar",{"powdered sugar"}),("icing sugar",{"powdered sugar","icing sugar"}),("aubergine",{"eggplant","aubergine"}),("ground beef",{"ground beef"}),("mince",{"ground beef","mince"}),
("courgette",{"zucchini","courgette"}),("coriander leaves",{"cilantro","coriander"}),("rocket",{"arugula","rocket"}),("capsicum",{"bell pepper","capsicum"}),
("caster sugar",{"superfine sugar","caster sugar","sugar"}),("plain flour",{"flour","plain flour","all purpose flour"}),("self-raising flour",{"self rising flour","self raising flour"}),
("bicarbonate of soda",{"baking soda","bicarbonate of soda"}),("double cream",{"heavy cream","double cream"}),("single cream",{"light cream","single cream"}),
("spring onions",{"green onion","scallion","spring onion"}),("swede",{"rutabaga","swede"}),("prawns",{"shrimp","prawn"}),("minced beef",{"ground beef","minced beef"}),
("cornflour",{"cornstarch","cornflour"}),("coriander",{"coriander","cilantro"}),("beetroot",{"beet","beetroot"}),("chips",{"french fry","chip"}),
("eggs",{"egg"}),("egg",{"egg"}),("milk",{"milk"}),("whole milk",{"whole milk","milk"}),("butter",{"butter"}),("unsalted butter",{"unsalted butter","butter"}),
("garlic",{"garlic"}),("garlic cloves",{"garlic","garlic clove"}),("onion",{"onion"}),("onions",{"onion"}),("red onion",{"red onion","onion"}),("yellow onion",{"yellow onion","onion"}),
("potatoes",{"potato"}),("sweet potatoes",{"sweet potato"}),("carrots",{"carrot"}),("bell pepper",{"bell pepper"}),("red bell pepper",{"red bell pepper","bell pepper"}),
("olive oil",{"olive oil"}),("extra virgin olive oil",{"extra virgin olive oil","olive oil"}),("vegetable oil",{"vegetable oil","oil"}),("canola oil",{"canola oil","oil"}),
("soy sauce",{"soy sauce"}),("rice",{"rice"}),("white rice",{"white rice","rice"}),("basmati rice",{"basmati rice","rice"}),("pasta",{"pasta"}),("spaghetti",{"spaghetti"}),
("parmesan",{"parmesan","parmesan cheese"}),("parmigiano reggiano",{"parmesan cheese","parmigiano reggiano"}),("mozzarella",{"mozzarella","mozzarella cheese"}),("cheddar",{"cheddar","cheddar cheese"}),
("feta",{"feta cheese","feta"}),("greek yogurt",{"greek yogurt","yogurt"}),("yoghurt",{"yogurt","yoghurt"}),("sour cream",{"sour cream"}),("cream cheese",{"cream cheese"}),
("baking powder",{"baking powder"}),("baking soda",{"baking soda"}),("vanilla extract",{"vanilla extract","vanilla"}),("brown sugar",{"brown sugar"}),("white sugar",{"white sugar","sugar"}),("sugar",{"sugar"}),
("salt",{"salt"}),("black pepper",{"black pepper","pepper"}),("freshly ground black pepper",{"black pepper","pepper"}),("kosher salt",{"kosher salt","salt"}),("sea salt",{"sea salt","salt"}),
("paprika",{"paprika"}),("smoked paprika",{"smoked paprika","paprika"}),("cumin",{"cumin"}),("ground cumin",{"ground cumin","cumin"}),("chili powder",{"chili powder"}),("chilli flakes",{"chili flake","red pepper flake","chilli flake"}),
("red pepper flakes",{"red pepper flake"}),("oregano",{"oregano"}),("basil",{"basil"}),("fresh basil",{"fresh basil","basil"}),("parsley",{"parsley"}),("thyme",{"thyme"}),("rosemary",{"rosemary"}),
("lemon",{"lemon"}),("lemon juice",{"lemon juice"}),("lime",{"lime"}),("chicken breast",{"chicken breast"}),("chicken breasts",{"chicken breast"}),("boneless skinless chicken breast",{"chicken breast","boneless skinless chicken breast"}),
("bacon",{"bacon"}),("ground turkey",{"ground turkey"}),("pork chops",{"pork chop"}),("salmon",{"salmon"}),("tuna",{"tuna"}),("canned tuna",{"tuna","canned tuna"}),("shrimp",{"shrimp"}),
("tomato paste",{"tomato paste"}),("tomato sauce",{"tomato sauce"}),("crushed tomatoes",{"crushed tomato","tomato"}),("canned tomatoes",{"tomato","canned tomato"}),("cherry tomatoes",{"cherry tomato","tomato"}),
("mushrooms",{"mushroom"}),("spinach",{"spinach"}),("broccoli",{"broccoli"}),("cauliflower",{"cauliflower"}),("zucchini",{"zucchini"}),("cucumber",{"cucumber"}),("avocado",{"avocado"}),("avocados",{"avocado"}),
("peanut butter",{"peanut butter"}),("honey",{"honey"}),("maple syrup",{"maple syrup"}),("oats",{"oat","rolled oat"}),("rolled oats",{"rolled oat","oat"}),("bread",{"bread"}),("tortillas",{"tortilla"}),
("black beans",{"black bean"}),("kidney beans",{"kidney bean"}),("chickpeas",{"chickpea","garbanzo bean"}),("lentils",{"lentil"}),("coconut milk",{"coconut milk"}),("chicken stock",{"chicken stock"}),("chicken broth",{"chicken broth"}),
("vegetable broth",{"vegetable broth"}),("water",{"water"}),("white wine",{"white wine"}),("red wine",{"red wine"}),("dijon mustard",{"dijon mustard","mustard"}),("mayo",{"mayonnaise","mayo"}),("mayonnaise",{"mayonnaise"}),
("ketchup",{"ketchup"}),("ginger",{"ginger"}),("fresh ginger",{"ginger","fresh ginger"}),("cinnamon",{"cinnamon"}),("nutmeg",{"nutmeg"}),("cocoa powder",{"cocoa powder","cocoa"}),("chocolate chips",{"chocolate chip"}),
("worcestershire",{"worcestershire sauce","worcestershire"}),("tabasco",{"hot sauce","tabasco"}),("sriracha",{"sriracha","hot sauce"}),("pb",{"peanut butter"}),("OJ",{"orange juice"}),("tomatoe",{"tomato"}),("potatos",{"potato"}),("cheeze",{"cheese"}),
("brocoli",{"broccoli"}),("zuchini",{"zucchini"}),("parmesean",{"parmesan"}),("tumeric",{"turmeric"}),("cilantro",{"cilantro"}),("2 tbsp EVOO",{"olive oil"}),("1 lb ground beef/mince",{"ground beef","mince"}),
]
def cook_path(s): return canonicalize(s)
def p(s):
    pi=parse(s,v,idx); return pi.canonical
rows=[];ca=pa=0
for s,exp in T:
    c=cook_path(s); pc=p(s)
    cok = c in exp and c in vs
    pok = pc in exp
    ca+=cok; pa+=pok
    rows.append((s,c,c in vs,cok,pc,pok))
print(len(T),'inputs; cook-path(canonicalize only) hit',ca,round(ca/len(T),3),' parse() hit',pa,round(pa/len(T),3))
print('in-vocab-but-wrong/unmatched examples:')
for r in rows:
    if not r[3] or not r[5]: print(r)
