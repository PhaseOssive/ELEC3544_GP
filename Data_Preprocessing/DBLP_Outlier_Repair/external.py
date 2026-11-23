"""Crossref publisher-deposited evidence; durable cache; failures are UNKNOWN."""
import hashlib,json,sqlite3,time,urllib.request,urllib.error,urllib.parse
from datetime import datetime,timezone
from email.utils import parsedate_to_datetime
from difflib import SequenceMatcher
from snapshot import orcid
from vendor.names import compatible
from vendor.rules import normalize_title,normalize_doi

class Crossref:
    def __init__(self,path,cfg):
        self.db=sqlite3.connect(path);self.cfg=cfg;self.last=0;self.requests=0;self.hits=0
        self.db.execute('CREATE TABLE IF NOT EXISTS cache(doi TEXT PRIMARY KEY,status TEXT,payload TEXT,observed_at REAL)')
    def work(self,doi):
        doi=normalize_doi(doi)
        if not doi:return {'status':'NO_DOI'}
        old=self.db.execute('SELECT status,payload,observed_at FROM cache WHERE doi=?',(doi,)).fetchone()
        if old and (old[0] in ('OK','NOT_FOUND') or time.time()-old[2]<3600*self.cfg.get('http_error_retry_hours',12)):
            self.hits+=1;return json.loads(old[1])
        if not self.cfg.get('crossref_enabled',True):return {'status':'OFFLINE_NOT_CHECKED'}
        endpoint='https://api.crossref.org/works/'+urllib.parse.quote(doi,safe='')
        if self.cfg.get('crossref_mailto'):endpoint+='?mailto='+urllib.parse.quote(self.cfg['crossref_mailto'])
        result={'status':'NETWORK_ERROR'}
        for attempt in range(self.cfg.get('http_retries',3)):
            time.sleep(max(0,self.cfg.get('crossref_interval_seconds',1)-(time.monotonic()-self.last)))
            self.last=time.monotonic();self.requests+=1
            try:
                req=urllib.request.Request(endpoint,headers={'User-Agent':'GP-DBLP-Outlier-Audit/2.0','Accept':'application/json'})
                with urllib.request.urlopen(req,timeout=self.cfg.get('http_timeout_seconds',30)) as response:
                    content=response.read();data=json.loads(content)
                m=data.get('message',{})
                if data.get('status')!='ok' or normalize_doi(m.get('DOI'))!=doi:raise ValueError('Not the requested DOI record')
                # Keep only fields actually used, never treat HTML/challenge content as metadata.
                result={'status':'OK','doi':doi,'title':m.get('title',[]),'author':m.get('author',[]),'source':endpoint,'retrieved_at':datetime.now(timezone.utc).isoformat()}
                break
            except urllib.error.HTTPError as exc:
                result={'status':'NOT_FOUND' if exc.code==404 else 'HTTP_'+str(exc.code),'doi':doi,'source':endpoint}
                if exc.code==404:break
                if exc.code in (401,403):break
                if attempt+1<self.cfg.get('http_retries',3):
                    value=exc.headers.get('Retry-After','')
                    try:wait=float(value)
                    except ValueError:
                        try:wait=max(0,parsedate_to_datetime(value).timestamp()-time.time())
                        except (TypeError,ValueError):wait=2**(attempt+1)
                    # Never retry earlier than Retry-After. This is runtime code, not an interactive tool wait.
                    time.sleep(max(1,wait))
            except (ValueError,UnicodeError,urllib.error.URLError,TimeoutError,OSError) as exc:
                result={'status':'NETWORK_OR_PARSE_ERROR','doi':doi,'error':str(exc)[:250],'source':endpoint}
                if attempt+1<self.cfg.get('http_retries',3):time.sleep(2**(attempt+1))
        self.db.execute('INSERT OR REPLACE INTO cache VALUES (?,?,?,?)',(doi,result['status'],json.dumps(result,ensure_ascii=False),time.time()));self.db.commit()
        if self.requests%50==0:print(f'Publisher API: {self.requests:,} new requests; {self.hits:,} cache hits; last={result["status"]}',flush=True)
        return result

def align(metadata,title,names):
    """Whole roster alignment; never infer identity from same-name hit alone."""
    if metadata.get('status')!='OK':return None,'METADATA_'+metadata.get('status','UNKNOWN')
    titles=metadata.get('title') or []
    if not titles or max(SequenceMatcher(None,normalize_title(title),normalize_title(t)).ratio() for t in titles)<.92:return None,'TITLE_CONFLICT'
    people=metadata.get('author') or []
    other=[' '.join([p.get('given',''),p.get('family','')]).strip() or p.get('name','') for p in people]
    if len(names)!=len(other) or not names:return None,'ROSTER_SIZE_CONFLICT'
    if not all(compatible(a,b) for a,b in zip(names,other)):return None,'ORDERED_ROSTER_CONFLICT'
    if any(sum(bool(compatible(a,b)) for b in other)!=1 for a in names):return None,'AMBIGUOUS_NAME_ALIGNMENT'
    aligned=[]
    for p in people:
        aligned.append({'orcid':orcid(p.get('ORCID','')),'authenticated':p.get('authenticated-orcid') is True,'affiliations':[a.get('name','') for a in p.get('affiliation',[])],'name':' '.join([p.get('given',''),p.get('family','')]).strip()})
    return aligned,'ALIGNED_PUBLISHER_ROSTER'
