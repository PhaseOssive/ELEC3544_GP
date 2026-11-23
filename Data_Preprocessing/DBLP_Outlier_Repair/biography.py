"""Conservative primary-homepage evidence. No LLM guesses, no CAPTCHA bypass."""
import ipaddress,json,re,socket,time,urllib.request,urllib.parse,urllib.error
from lxml import html
from snapshot import orcid
from vendor.names import compatible
from vendor.rules import normalize_title

SKIP={'orcid.org','dblp.org','dblp.uni-trier.de','dblp.dagstuhl.de','wikipedia.org','wikidata.org','scholar.google.com','semanticscholar.org','researchgate.net'}

def public_url(url):
    u=urllib.parse.urlsplit(url)
    if u.scheme not in ('https','http') or not u.hostname or u.username or u.password:raise ValueError('Not a public web URL')
    if u.port not in (None,80,443):raise ValueError('Nonstandard port')
    for item in socket.getaddrinfo(u.hostname,u.port or (443 if u.scheme=='https' else 80),type=socket.SOCK_STREAM):
        if not ipaddress.ip_address(item[4][0]).is_global:raise ValueError('Non-public destination')
    return url

class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,request,fp,code,msg,headers,newurl):
        public_url(newurl)
        return super().redirect_request(request,fp,code,msg,headers,newurl)


def parse_page(raw,url):
    tree=html.fromstring(raw,base_url=url)
    title=' '.join(tree.xpath('//title/text()'))
    people=[]
    def collect(x):
        if isinstance(x,list):
            for item in x:collect(item)
        elif isinstance(x,dict):
            typ=x.get('@type',[]);typ=[typ] if isinstance(typ,str) else typ
            if 'Person' in typ:people.append(x)
            if '@graph' in x:collect(x['@graph'])
    for text in tree.xpath('//script[@type="application/ld+json"]/text()'):
        try:collect(json.loads(text))
        except ValueError:pass
    links=[urllib.parse.urljoin(url,h) for h in tree.xpath('//a/@href')]
    for node in tree.xpath('//script|//style'):node.drop_tree()
    text=' '.join(tree.text_content().split())
    if any(x in text.casefold() for x in ('making sure you','verify you are human','enable javascript and cookies')):return {'status':'CHALLENGE_NOT_BYPASSED','url':url}
    return {'status':'OK','url':url,'title':title,'text':text[:250000],'people':people,'links':links}

class Biography:
    def __init__(self,db,cfg):
        self.db=db;self.cfg=cfg;self.requests=0;self.last=0
        db.execute('CREATE TABLE IF NOT EXISTS pages(url TEXT PRIMARY KEY,payload TEXT,observed_at REAL)')
    def page(self,url):
        r=self.db.execute('SELECT payload,observed_at FROM pages WHERE url=?',(url,)).fetchone()
        if r:
            old=json.loads(r[0])
            if old['status']=='OK' or time.time()-r[1]<3600*self.cfg.get('http_error_retry_hours',12):return old
        if not self.cfg.get('biography_enabled',True):return {'status':'DISABLED','url':url}
        time.sleep(max(0,.5-(time.monotonic()-self.last)));self.last=time.monotonic();self.requests+=1
        try:
            public_url(url)
            request=urllib.request.Request(url,headers={'User-Agent':'GP-DBLP-Outlier-Audit/2.0 (public academic biography verification)'})
            with urllib.request.build_opener(SafeRedirect()).open(request,timeout=self.cfg.get('http_timeout_seconds',30)) as r:
                raw=r.read(2000001)
                if len(raw)>2000000:raise ValueError('Page exceeds bounded read; not parsed')
                contenttype=r.headers.get('Content-Type','').lower()
                if 'html' not in contenttype:raise ValueError('Not HTML; no inference')
                finalhost=(urllib.parse.urlsplit(r.geturl()).hostname or '').lower()
                if any(finalhost==s or finalhost.endswith('.'+s) for s in SKIP):raise ValueError('Redirected to non-primary authority page')
                result=parse_page(raw,r.geturl())
        except Exception as exc:
            # Unavailable/blocked pages are not evidence of a valid or invalid author.
            result={'status':'UNAVAILABLE','url':url,'error':str(exc)[:200]}
        self.db.execute('INSERT OR REPLACE INTO pages VALUES (?,?,?)',(url,json.dumps(result,ensure_ascii=False),time.time()));self.db.commit()
        return result
    def check(self,profile,name,paper,anchor):
        urls=[]
        for u in profile['urls']:
            host=(urllib.parse.urlsplit(u).hostname or '').lower()
            if host and not any(host==s or host.endswith('.'+s) for s in SKIP):urls.append(u)
        facts=[];checked=[];limit=self.cfg.get('biography_max_pages_per_author',4)
        queue=list(dict.fromkeys(urls))[:limit];i=0
        while i<len(queue) and i<limit:
            url=queue[i];i+=1;p=self.page(url);checked.append({'url':url,'status':p['status']})
            if p['status']!='OK':continue
            facts.extend(birth_facts(p,name,paper,anchor))
            host=urllib.parse.urlsplit(p['url']).hostname
            for link in p['links']:
                if len(queue)>=limit:break
                if urllib.parse.urlsplit(link).hostname==host and re.search(r'/(?:publications?|research|about|bio|cv|AI)(?:[/.?#]|$)',link,re.I) and link not in queue:queue.append(link)
        years={f['birth_year'] for f in facts}
        return {'status':'BIRTH_YEAR_SUPPORTED' if len(years)==1 else 'CONFLICTING_BIRTH_YEARS' if len(years)>1 else 'NO_VERIFIABLE_BIRTH_YEAR','birth_year':next(iter(years)) if len(years)==1 else None,'facts':facts,'pages_checked':checked}

def birth_facts(page,name,paper,anchor):
    """Require explicit identity binding; merely seeing a birth year is not enough."""
    facts=[];target_id=paper['paper_id'];target_title=normalize_title(paper['title_v1'])
    target_link=any(re.search(r'(?:arxiv\.org/(?:abs|pdf)/|doi\.org/10\.48550/arxiv\.)'+re.escape(target_id)+r'(?:v\d+|\.pdf)?(?:$|[?#])',u,re.I) for u in page['links'])
    title_match=bool(target_title and target_title in normalize_title(page['text']))
    for person in page['people']:
        if not compatible(person.get('name',''),name):continue
        same=person.get('sameAs',[]);same=[same] if isinstance(same,str) else same
        identity=(anchor and anchor in {orcid(v) for v in same}) or (target_link and title_match)
        birth=person.get('birthDate','')
        if identity and isinstance(birth,str) and re.fullmatch(r'(18|19|20)\d{2}(?:-\d{2}-\d{2})?',birth):
            facts.append({'birth_year':int(birth[:4]),'source':page['url'],'rule':'NAMED_PERSON_JSONLD_BIRTH_WITH_IDENTITY_BINDING','value':birth})
    # First-person biography only, named page title + exact target paper linkage.
    if target_link and title_match and normalize_title(name) in normalize_title(page['title']):
        for match in re.finditer(r'\bI was born\b[^.!?]{0,100}?\b((?:18|19|20)\d{2})\b',page['text'],re.I):
            facts.append({'birth_year':int(match[1]),'source':page['url'],'rule':'FIRST_PERSON_BIRTH_WITH_NAMED_PAGE_AND_TARGET_PAPER_LINK','excerpt':match[0]})
    return facts
