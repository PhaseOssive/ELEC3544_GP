import re,unicodedata
from .rules import normalize_name
def base_name(name):
    name=re.sub(r'\s+\d{4}$','',str(name).strip())
    name=re.sub(r'^(?:and|&)\s+','',name,flags=re.I)
    name=re.split(r'\s+n[ée]e?\s+',name,flags=re.I)[0]
    return name.translate(str.maketrans({'ł':'l','Ł':'L','ø':'o','Ø':'O','đ':'d','Đ':'D','ß':'ss','ı':'i'}))

def tokens(name):
    n=unicodedata.normalize('NFKD',base_name(name)).casefold()
    return re.findall(r'[^\W_]+',''.join(c for c in n if not unicodedata.combining(c)),flags=re.UNICODE)

def compatible(a,b):
    a,b=base_name(a),base_name(b)
    if normalize_name(a)==normalize_name(b):return 'FULL_NAME'
    x,y=tokens(a),tokens(b)
    if len(x)<2 or len(y)<2:return None
    if sorted(x)==sorted(y):return 'NAME_ORDER'
    if x[-1]!=y[-1]:return None
    def match(l,r):return l==r or (len(l)==1 and r.startswith(l)) or (len(r)==1 and l.startswith(r))
    if not match(x[0],y[0]):return None
    short,long=(x[:-1],y[:-1]) if len(x)<=len(y) else (y[:-1],x[:-1]);j=0
    for t in long:
        if j<len(short) and match(short[j],t):j+=1
    return 'GIVEN_NAME_OR_INITIAL_EXPANSION' if j==len(short) else None

def edit_distance(a,b,limit=2):
    if abs(len(a)-len(b))>limit:return limit+1
    prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        cur=[i]
        for j,y in enumerate(b,1):cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
        if min(cur)>limit:return limit+1
        prev=cur
    return prev[-1]

