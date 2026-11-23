"""Filtered, resumable DBLP snapshot index. Never modifies recovery caches."""
import gzip,json,re,sqlite3
from functools import lru_cache
from collections import defaultdict
from pathlib import Path
from lxml import etree
from vendor.rules import (child_texts,normalize_name,name_signature,normalize_title,
    extract_dois,extract_arxiv_ids,get_year,make_history_fallback_key)
from vendor.names import base_name
from vendor.dedup import DiskDeduplicator

TAGS={'article','inproceedings','proceedings','book','incollection','phdthesis','mastersthesis'}
EXCLUDED={'withdrawn','edited','data','software','encyclopedia'}

def orcid(value):
    m=re.search(r'\b(\d{4}-\d{4}-\d{4}-\d{3}[\dX])\b',str(value),re.I)
    if not m:return ''
    value=m[1].upper();digits=value.replace('-','');total=0
    for d in digits[:-1]:total=(total+int(d))*2
    check=(12-total%11)%11
    return value if digits[-1]==('X' if check==10 else str(check)) else ''

class Resolver(etree.Resolver):
    def __init__(self,path):self.path=str(Path(path).resolve())
    def resolve(self,url,pubid,context):
        if url.rsplit('/',1)[-1]=='dblp.dtd' or url==self.path:return self.resolve_filename(self.path,context)
        raise ValueError('External XML entity forbidden: '+url)

def records(xml,dtd,progress=None):
    opener=gzip.open if str(xml).endswith('.gz') else open
    with open(xml,'rb') as rawfile:
        f=gzip.GzipFile(fileobj=rawfile) if str(xml).endswith('.gz') else rawfile
        it=etree.iterparse(f,events=('end',),tag=tuple(TAGS|{'www','person','data'}),load_dtd=True,resolve_entities=True,no_network=True,huge_tree=True,recover=False)
        it.resolvers.add(Resolver(dtd))
        for index,(_,e) in enumerate(it):
            parent=e.getparent()
            if parent is None or parent.tag!='dblp':continue
            if progress and index%4096==0:progress.update(rawfile.tell())
            if e.tag in TAGS|{'www'}:yield e
            e.clear()
            while e.getprevious() is not None:del parent[0]

class Snapshot:
    def __init__(self,path):
        self.db=sqlite3.connect(path)
        self.db.executescript('''PRAGMA journal_mode=DELETE; PRAGMA temp_store=FILE;
        CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY,v TEXT);
        CREATE TABLE IF NOT EXISTS profiles(pid TEXT PRIMARY KEY,redirect TEXT,valid INTEGER,orcids TEXT,urls TEXT);
        CREATE TABLE IF NOT EXISTS aliases(pid TEXT,raw TEXT,PRIMARY KEY(pid,raw));
        CREATE INDEX IF NOT EXISTS alias_raw ON aliases(raw);
        CREATE TABLE IF NOT EXISTS pubs(key TEXT PRIMARY KEY,payload TEXT,ordinal INTEGER);
        CREATE TABLE IF NOT EXISTS links(pid TEXT,key TEXT,position INTEGER,PRIMARY KEY(pid,key,position));
        CREATE INDEX IF NOT EXISTS link_key ON links(key);
        CREATE TABLE IF NOT EXISTS targets(aid TEXT,key TEXT,PRIMARY KEY(aid,key));''')
    def get(self,k):
        r=self.db.execute('SELECT v FROM meta WHERE k=?',(k,)).fetchone();return r[0] if r else None
    def put(self,k,v):self.db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',(k,str(v)))
    @lru_cache(maxsize=200000)
    def canonical(self,pid):
        seen=set()
        while pid not in seen:
            seen.add(pid);r=self.db.execute('SELECT redirect,valid FROM profiles WHERE pid=?',(pid,)).fetchone()
            if not r or not r[1]:return ''
            if not r[0]:return pid
            pid=r[0]
        return ''
    @lru_cache(maxsize=100000)
    def profile(self,pid):
        pid=self.canonical(pid)
        r=self.db.execute('SELECT orcids,urls FROM profiles WHERE pid=?',(pid,)).fetchone()
        return {'pid':pid,'orcids':set(json.loads(r[0])),'urls':json.loads(r[1])} if r else {'pid':'','orcids':set(),'urls':[]}
    def alias_map(self):
        canon={p:self.canonical(p) for p, in self.db.execute('SELECT pid FROM profiles')};raw=defaultdict(set)
        for pid,name in self.db.execute('SELECT pid,raw FROM aliases'):
            if canon.get(pid):raw[name].add(canon[pid])
        return raw
    def orcid_map(self):
        result=defaultdict(set)
        for pid,js in self.db.execute('SELECT pid,orcids FROM profiles WHERE valid=1'):
            cp=self.canonical(pid)
            if cp:
                for o in json.loads(js):result[o].add(cp)
        return result
    def build(self,xml,dtd,authors,papers,signature,checkpoint=lambda:None,legacy=None,progress=None):
        if self.get('signature') not in (None,signature):raise ValueError('Snapshot cache signature mismatch')
        self.put('signature',signature);self.db.commit()
        names=set(authors.original_name)|set(authors.repaired_name)
        norm={normalize_name(base_name(n)) for n in names};sigs={name_signature(n) for n in names};sigs.discard(None)
        pids=set(authors.dblp_person_id)
        if self.get('profiles_done')!='1' and self.get('legacy_profiles_seeded')!='1' and legacy is not None:
            legacy.seed(self,authors)
        if self.get('profiles_done')!='1' and self.get('legacy_profiles_seeded')!='1':
            if progress:progress.begin('XML profile scan',Path(xml).stat().st_size,'Compressed bytes for gz; phase ETA only.')
            done=int(self.get('profile_checkpoint') or 0)
            for n,e in enumerate(records(xml,dtd,progress),1):
                if n<=done:continue
                key=e.get('key','')
                if e.tag=='www' and key.startswith('homepages/'):
                    aliases=child_texts(e,'author');red=child_texts(e,'crossref')
                    if key in pids or any(normalize_name(base_name(a)) in norm or name_signature(a) in sigs for a in aliases):
                        valid=int(not bool(set(e.get('publtype','').lower().split())&{'disambiguation','group','noshow'}))
                        urls=child_texts(e,'url');oids=sorted({orcid(u) for u in urls if orcid(u)})
                        self.db.execute('INSERT OR REPLACE INTO profiles VALUES (?,?,?,?,?)',(key,red[0] if red else '',valid,json.dumps(oids),json.dumps(urls)))
                        self.db.executemany('INSERT OR IGNORE INTO aliases VALUES (?,?)',[(key,a) for a in aliases])
                if n%500000==0:
                    self.put('profile_checkpoint',n);self.db.commit();print(f'Profile scan: {n:,} records',flush=True)
                    if n%2000000==0:checkpoint()
            self.put('profiles_done',1);self.db.commit();checkpoint()
        raw=self.alias_map();paperids=set(papers.paper_id)
        if not raw:raise ValueError('No candidate aliases found in XML; check snapshot')
        if self.get('records_done')!='1':
            if progress:progress.begin('XML publications + identity enrichment',Path(xml).stat().st_size,'One full sequential pass, including cached records; metadata missing from old cache is filled here.')
            targetkeys=defaultdict(set)
            for a in authors.to_dict('records'):
                for key in json.loads(a.get('focal_record_keys_json') or '[]'):targetkeys[key].add(a['arxiv_id'])
            titles=defaultdict(set)
            for p in papers.to_dict('records'):titles[normalize_title(p['title_v1'])].add(p['paper_id'])
            enrich_profiles=self.get('legacy_profiles_seeded')=='1'
            raw_profile_pids={r[0] for r in self.db.execute('SELECT pid FROM profiles')}
            done=int(self.get('record_checkpoint') or 0)
            for n,e in enumerate(records(xml,dtd,progress),1):
                if n<=done:continue
                if enrich_profiles and e.tag=='www':
                    key=e.get('key','')
                    if key in raw_profile_pids:
                        urls=child_texts(e,'url');oids=sorted({orcid(u) for u in urls if orcid(u)})
                        self.db.execute('UPDATE profiles SET orcids=?,urls=? WHERE pid=?',(json.dumps(oids),json.dumps(urls),key))
                if e.tag in TAGS and not set(e.get('publtype','').lower().split())&EXCLUDED:
                    key=e.get('key','');ns=child_texts(e,'author')
                    relevant=any(a in raw for a in ns) or key in targetkeys
                    if relevant:
                        payload=legacy.payload(key,ns) if legacy else None
                        if payload is None:
                            title=next(iter(child_texts(e,'title')),'');year=get_year(e)
                            urls=' '.join(t for tag in ('ee','url','eprint') for t in child_texts(e,tag))
                            dois=extract_dois(urls);aids=extract_arxiv_ids(urls)
                            for doi in dois:
                                if doi.startswith('10.48550/arxiv.'):aids.add(re.sub(r'v\d+$','',doi.split('arxiv.',1)[1]))
                            payload=dict(key=key,title=title,year=year,authors=ns,dois=sorted(dois),arxiv_ids=sorted(aids),venue=' | '.join(child_texts(e,'journal')+child_texts(e,'booktitle')))
                        # Old missing cache discarded signature ORCIDs; always obtain these from XML.
                        payload['signatures']=[{'name':''.join(a.itertext()).strip(),'orcid':orcid(a.get('orcid',''))} for a in e.findall('author')]
                        title=payload['title'];aids=set(payload['arxiv_ids'])
                        self.db.execute('INSERT OR REPLACE INTO pubs VALUES (?,?,?)',(key,json.dumps(payload,ensure_ascii=False),n))
                        for j,name in enumerate(ns):
                            # Alias ambiguity is never resolved by selecting first PID.
                            if len(raw.get(name,()))==1:self.db.execute('INSERT OR IGNORE INTO links VALUES (?,?,?)',(next(iter(raw[name])),key,j))
                        hits=set(targetkeys.get(key,()))|set(aids)&paperids|titles.get(normalize_title(title),set())
                        self.db.executemany('INSERT OR IGNORE INTO targets VALUES (?,?)',[(a,key) for a in hits])
                if n%500000==0:
                    self.put('record_checkpoint',n);self.db.commit();print(f'Publication scan: {n:,} records',flush=True)
                    if n%2000000==0:checkpoint()
            self.put('profiles_done',1);self.put('records_done',1);self.db.commit();checkpoint()
        d=DiskDeduplicator(self.db)
        if self.get('dedup_done')!='1':
            for t in ('works','work_years','strong_keys','fallback_keys','record_works'):self.db.execute('DELETE FROM '+t)
            if progress:progress.begin('Deduplicate selected histories',self.db.execute('SELECT count(*) FROM pubs').fetchone()[0])
            for count,(key,payload) in enumerate(self.db.execute('SELECT key,payload FROM pubs ORDER BY ordinal'),1):
                if progress:progress.update(count)
                r=json.loads(payload);ids=[x[0] for x in self.db.execute('SELECT DISTINCT pid FROM links WHERE key=?',(key,))] if r['year'] is not None else []
                d.add_record({'doi:'+v for v in r['dois']}|{'arxiv:'+v for v in r['arxiv_ids']},make_history_fallback_key(r['title'],r['authors']),ids,r['year'],key)
            d.finalize();self.put('dedup_done',1);self.db.commit();checkpoint()
        self.raw=raw;self.oids=self.orcid_map()
    def target_records(self,aid):
        return [json.loads(r[0]) for r in self.db.execute('SELECT p.payload FROM pubs p JOIN targets t USING(key) WHERE t.aid=?',(aid,))]
    def history(self,pid,aid,year):
        focal={r[0] for r in self.db.execute('SELECT r.work FROM targets t JOIN record_works r USING(key) WHERE t.aid=?',(aid,))}
        result=[]
        for work,y in self.db.execute('SELECT work,year FROM work_years WHERE pid=? AND year<? ORDER BY year,work',(pid,year)):
            if work in focal:continue
            recs=[]
            for payload,pos in self.db.execute('SELECT p.payload,l.position FROM pubs p JOIN links l USING(key) JOIN record_works w USING(key) WHERE w.work=? AND l.pid=?',(work,pid)):
                r=json.loads(payload);r['position']=pos;recs.append(r)
            result.append({'work':work,'year':y,'records':recs})
        return result
    @lru_cache(maxsize=100000)
    def candidate_usable(self,pid):
        # An ambiguous alias or unknown-year history may hide works; never certify zero from that.
        aliases=[r[0] for r in self.db.execute('SELECT raw FROM aliases WHERE pid=?',(pid,))]
        if not aliases or any(len(self.raw.get(a,()))!=1 for a in aliases):return False
        for payload, in self.db.execute('SELECT p.payload FROM pubs p JOIN links l USING(key) WHERE l.pid=?',(pid,)):
            if json.loads(payload)['year'] is None:return False
        return True
