"""One streaming local XML pass for all target papers and candidate histories."""
import gzip,json,logging,re
from collections import defaultdict,Counter
from difflib import SequenceMatcher
from functools import lru_cache
from lxml import etree
from .progress import tqdm
from .rules import *
from .names import compatible

TAGS={'article','inproceedings','proceedings','book','incollection','phdthesis','mastersthesis'}
EXCLUDED={'withdrawn','edited','data','software','encyclopedia'}
class Resolver(etree.Resolver):
    def __init__(self,path):self.path=str(path)
    def resolve(self,url,pubid,context):
        if url.rsplit('/',1)[-1]=='dblp.dtd' or url==self.path:return self.resolve_filename(self.path,context)
        raise ValueError('External XML entity forbidden: '+url)

def dtd_audit(path):
    d=etree.DTD(str(path));attrs=[]
    for e in d.iterelements():
        if e.name=='author':attrs=[a.name for a in e.iterattributes()]
    return {'author_attributes':attrs,'has_direct_pid_attribute':'pid' in attrs,'publication_link_rule':'RAW_ALIAS_UNIQUE_VALID_CANONICAL_PROFILE','unrecognized_pid_attributes_ignored':True}

def records(xml,dtd):
    opener=gzip.open if str(xml).endswith('.gz') else open
    with opener(xml,'rb') as f:
        it=etree.iterparse(f,events=('end',),load_dtd=True,resolve_entities=True,no_network=True,huge_tree=True,recover=False);it.resolvers.add(Resolver(dtd))
        for _,e in it:
            parent=e.getparent()
            if parent is None or parent.tag!='dblp':continue
            if e.tag in TAGS|{'www'}:yield e
            e.clear()
            while e.getprevious() is not None:del parent[0]

def publication(e):
    authors=child_texts(e,'author');title=next(iter(child_texts(e,'title')),'');year=get_year(e)
    links=' '.join(x for tag in ('ee','url','eprint') for x in child_texts(e,tag));dois=extract_dois(links);aids=extract_arxiv_ids(links)
    for doi in dois:
        if doi.startswith('10.48550/arxiv.'):aids.add(re.sub(r'v\d+$','',doi.split('10.48550/arxiv.',1)[1]))
    return dict(key=make_dblp_record_key(e,e.tag,title,year,authors),type=e.tag,title=title,title_key=normalize_title(title),year=year,authors=authors,dois=sorted(dois),arxiv_ids=sorted(aids),venue=' | '.join(child_texts(e,'journal')+child_texts(e,'booktitle')+child_texts(e,'series')),publtype=e.get('publtype',''))

class TargetSearch:
    def __init__(self,papers,rows,cfg,index):
        self.cfg=cfg;self.index=index;self.papers={p['arxiv_id']:p for p in papers};self.ids=set(self.papers);self.title=defaultdict(set);self.doi=defaultdict(set);self.postings=defaultdict(set);self.words={};self.anchors=defaultdict(set);self.names=defaultdict(list);self.anchor_names=defaultdict(set);self.fuzzy_terms={}
        for r in rows:
            if r['identity_status']=='VERIFIED':
                self.anchors[r['arxiv_id']].add(r['dblp_person_id'])
                for alias in index.aliases(r['dblp_person_id']):
                    if index.raw_ids(alias)=={r['dblp_person_id']}:self.anchor_names[alias].add(r['arxiv_id'])
            self.names[r['arxiv_id']].append(r.get('repaired_name') or r['original_name'])
        frequency=Counter()
        for aid,p in self.papers.items():
            title=normalize_title(p['title_v1']);self.title[title].add(aid);d=normalize_doi(p.get('doi_v1',''))
            if d:self.doi[d].add(aid)
            self.words[aid]={w for w in title.split() if len(w)>2};frequency.update(self.words[aid])
        for aid,words in self.words.items():
            rare=[w for w in words if frequency[w]<=max(1,len(self.papers)*.02)]
            self.fuzzy_terms[aid]=sorted(rare,key=lambda w:(frequency[w],w))[:3]
            for w in self.fuzzy_terms[aid]:self.postings[w].add(aid)
    def find(self,r):
        hits={aid:'ARXIV_ID' for aid in set(r['arxiv_ids'])&self.ids}
        for doi in r['dois']:
            for aid in self.doi.get(doi,()):hits.setdefault(aid,'DOI')
        for aid in self.title.get(r['title_key'],()):hits.setdefault(aid,'EXACT_TITLE')
        counts=Counter(aid for w in set(r['title_key'].split()) for aid in self.postings.get(w,()))
        anchored=set(aid for name in r['authors'] for aid in self.anchor_names.get(name,()))
        for aid in anchored:counts[aid]=max(counts[aid],2)
        for aid,n in counts.items():
            if aid in hits or n<min(2,max(1,len(self.fuzzy_terms[aid]))):continue
            if self.anchors[aid] and aid not in anchored:continue
            p=self.papers[aid]
            if r['year'] is None or abs(int(p['t0_year'])-r['year'])>self.cfg.title_year_window:continue
            overlap=len(self.words[aid]&set(r['title_key'].split()))/max(1,len(self.words[aid]))
            if overlap>=self.cfg.title_keyword_coverage and SequenceMatcher(None,normalize_title(p['title_v1']),r['title_key']).ratio()>=self.cfg.title_similarity:hits[aid]='FUZZY_TITLE_KEYWORDS_YEAR'
        results=[]
        for aid,method in hits.items():
            pids={self.index.unique(n) for n in r['authors']};pids.discard(None);anchors=len(pids&self.anchors[aid]);allanchors=self.anchors[aid]<=pids
            coverage=sum(any(compatible(n,m) for m in r['authors']) for n in self.names[aid])/max(1,len(self.names[aid]))
            score=SequenceMatcher(None,normalize_title(self.papers[aid]['title_v1']),r['title_key']).ratio()
            raw_source=Counter(normalize_name(n) for n in self.names[aid]);raw_target=Counter(normalize_name(n) for n in r['authors'])
            overlap=sum(1 for n in raw_source if raw_source[n]==raw_target[n]==1)
            full_name_corroboration=overlap>0 and overlap/max(1,len(self.names[aid]))>=.5 and overlap/max(1,len(r['authors']))>=.5
            reliable=method in ('ARXIV_ID','DOI') or (method=='EXACT_TITLE' and ((anchors>=2 and allanchors and coverage>=.5) or full_name_corroboration))
            # Fuzzy title alone is never accepted as focal identity evidence.
            results.append((aid,r['key'],method,score,anchors,int(reliable)))
        return results

def scan(cfg,store,index,papers,rows,phase='primary'):
    db=store.db;key='scan_'+phase
    if store.get(key+'_complete')=='1':
        logging.info('Complete local XML scan cache reused: %s',phase)
        if phase=='primary' and store.get('discovered_pending')=='1':
            store.aliases(index);scan(cfg,store,index,papers,rows,phase='discovered_profiles')
        return
    search=TargetSearch(papers,rows,cfg,index);checkpoint=int(store.get(key+'_checkpoint',0));logging.info('XML scan %s: replay to %d committed records',phase,checkpoint)
    # Only candidate aliases, not the full DBLP profile index. Avoid a SQL miss
    # for every non-candidate author in the entire XML snapshot.
    wanted_aliases=dict(db.execute('SELECT raw,pid FROM wanted_alias'))
    logging.info('Candidate alias membership loaded: %d aliases',len(wanted_aliases))
    def wanted(name):return wanted_aliases.get(name)
    @lru_cache(maxsize=100000)
    def actual(name):
        ids=index.raw_ids(name);return next(iter(ids)) if len(ids)==1 else None
    targets=history=relationships=0
    with tqdm(desc='Complete local XML '+phase,unit='records') as bar:
        for n,e in enumerate(records(cfg.DBLP_XML_PATH,cfg.DTD_PATH),1):
            bar.update(1)
            if n<=checkpoint:continue
            if e.tag in TAGS and not set(e.get('publtype','').lower().split())&EXCLUDED:
                r=publication(e);found=search.find(r) if phase=='primary' else [];hits={wanted(a) for a in r['authors']};hits.discard(None)
                if found or hits:
                    db.execute('INSERT OR REPLACE INTO publications VALUES (?,?,?,?,?)',(r['key'],r['year'],r['title'],json.dumps(r),n))
                    for kind,values in [('doi',r['dois']),('arxiv',r['arxiv_ids'])]:db.executemany('INSERT OR IGNORE INTO identifiers VALUES (?,?,?)',((kind,v,r['key']) for v in values))
                    for pos,name in enumerate(r['authors'],1):
                        pid=actual(name)
                        db.execute('INSERT OR REPLACE INTO publication_authors VALUES (?,?,?,?,?)',(r['key'],pos,name,pid or '', 'UNIQUE_RAW_PROFILE_ALIAS' if pid else 'AMBIGUOUS_OR_ABSENT_RAW_ALIAS'))
                        if pid:relationships+=1
                    for pid in hits:
                        db.execute('INSERT OR IGNORE INTO candidate_publications VALUES (?,?,?)',(pid,r['key'],'UNIQUE_RAW_PROFILE_ALIAS'))
                        db.execute('INSERT OR REPLACE INTO candidate_publication_years VALUES (?,?,?)',(pid,r['year'],r['key']));history+=1
                    for result in found:
                        db.execute('INSERT OR REPLACE INTO targets VALUES (?,?,?,?,?,?)',result);targets+=1
                        # P0: candidate exclusion is independent of author-identity reliability.
                        db.execute('INSERT OR IGNORE INTO focal VALUES (?,?)',result[:2])
                        db.execute('INSERT OR REPLACE INTO focal_audit VALUES (?,?,?,?,?)',(result[0],result[1],'LOCAL_XML_'+result[2],'POTENTIAL_TARGET_EXCLUDED_FROM_ALL_HISTORY_EVIDENCE',result[-1]))
                if n%1000==0:bar.set_postfix(targets=targets,history=history,person_links=relationships,refresh=False)
            if n%cfg.checkpoint_records==0:
                store.set(key+'_checkpoint',n);db.commit();logging.info('XML %s committed=%d target_hits=%d candidate_records=%d person_links=%d',phase,n,targets,history,relationships)
        store.set(key+'_checkpoint',n if 'n' in locals() else 0);db.commit()
    # New PIDs found on reliable target papers may require a second batched history pass.
    old=db.execute('SELECT COUNT(*) FROM wanted').fetchone()[0]
    db.execute("INSERT OR IGNORE INTO wanted(pid) SELECT DISTINCT a.pid FROM publication_authors a JOIN targets t USING(key) WHERE t.reliable=1 AND a.pid<>''")
    added=db.execute('SELECT COUNT(*) FROM wanted').fetchone()[0]-old
    if added and phase=='primary':store.set('discovered_pending',1)
    db.commit()
    if phase=='primary' and store.get('discovered_pending')=='1':store.aliases(index);scan(cfg,store,index,papers,rows,phase='discovered_profiles')
    if phase=='discovered_profiles':store.set('discovered_pending',0)
    db.execute('DELETE FROM candidate_history_status')
    db.execute('''INSERT INTO candidate_history_status SELECT w.pid,1,COUNT(p.key),SUM(CASE WHEN p.key IS NOT NULL AND p.year IS NULL THEN 1 ELSE 0 END),w.ambiguous_alias,
    CASE WHEN w.ambiguous_alias=1 THEN 'AMBIGUOUS_ALIAS_COVERAGE' WHEN COUNT(p.key)=0 THEN 'NO_OBSERVED_RECORDS' WHEN SUM(CASE WHEN p.key IS NOT NULL AND p.year IS NULL THEN 1 ELSE 0 END)>0 THEN 'MISSING_YEAR' ELSE 'COMPLETE_OBSERVED_SNAPSHOT' END
    FROM wanted w LEFT JOIN candidate_publications cp ON cp.pid=w.pid LEFT JOIN publications p ON p.key=cp.key GROUP BY w.pid''');store.set(key+'_complete',1);db.commit()
