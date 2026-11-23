"""Offline triage and source-bounded publisher verification for the v2 repair engine."""
import json
import sqlite3
import time
from collections import Counter
from contextlib import closing
from functools import lru_cache
from pathlib import Path

from common import AGE, PUB, integer, same
from snapshot import Snapshot
from vendor.names import compatible
from vendor.rules import normalize_doi

class FastSnapshot(Snapshot):
    """Retrieve a PID's deduplicated works and all record positions in one query."""
    @lru_cache(maxsize=16)
    def history(self, pid, aid, year):
        rows = self.db.execute("""
            SELECT y.work, y.year, p.payload, l.position
            FROM links l JOIN pubs p USING(key) JOIN record_works r USING(key)
            JOIN work_years y ON y.work=r.work AND y.pid=l.pid
            WHERE l.pid=? AND y.year<? AND y.work NOT IN (
              SELECT f.work FROM targets t JOIN record_works f USING(key) WHERE t.aid=?
            ) ORDER BY y.year, y.work, p.key,l.position""", (pid,year,aid))
        result = {}
        for work, date, payload, position in rows:
            record = json.loads(payload)
            record['position'] = position
            record['history_pid'] = pid
            result.setdefault(work, {'work':work,'year':date,'records':[]})['records'].append(record)
        return list(result.values())

def reusable_snapshot(root, binding):
    fields = ('xml_sha256','dtd_sha256','author_subset_sha256','paper_subset_sha256')
    for path in sorted(Path(root).glob('*/snapshot.sqlite')):
        try:
            with closing(sqlite3.connect(path.resolve().as_uri()+'?mode=ro', uri=True)) as db:
                meta = dict(db.execute('SELECT k,v FROM meta'))
                old = json.loads(meta['signature'])
                if old.get('version') not in ('2.0.0','2.1.0','3.0.0','3.1.0'):continue
                if all(meta.get(k)=='1' for k in ('profiles_done','records_done','dedup_done')) and all(old.get(k)==binding[k] for k in fields):
                    return path
        except (sqlite3.Error,KeyError,ValueError):
            continue
    return None

class TargetedPublisher:
    def __init__(self, base, cfg):
        self.base = base
        self.db = base.db
        self.cfg = cfg
        self.allowed = set()
        self.seconds_left = float(cfg.get('external_budget_seconds',600))
        self.limit = int(cfg.get('max_external_requests',180))
        self.deferred = set()
        self.seen = set()
        # One sequential index read avoids a Google Drive query for each uncached historical DOI.
        self.known = {r[0] for r in self.db.execute('SELECT doi FROM cache')}

    @property
    def requests(self):return self.base.requests
    @property
    def hits(self):return self.base.hits

    @lru_cache(maxsize=16384)
    def cached(self, doi):
        if doi not in self.known:return None
        r = self.db.execute('SELECT status,payload,observed_at FROM cache WHERE doi=?',(doi,)).fetchone()
        if r and (r[0] in ('OK','NOT_FOUND') or time.time()-r[2] < self.cfg.get('http_error_retry_hours',12)*3600):
            return json.loads(r[1])
        return None

    def work(self, value):
        doi = normalize_doi(value)
        cached = self.cached(doi)
        if cached is not None:
            self.base.hits += 1
            return cached
        if doi not in self.allowed:return {'status':'OFFLINE_NO_REQUEST'}
        if not self.cfg.get('crossref_enabled',True):return {'status':'OFFLINE_NOT_CHECKED'}
        if self.requests>=self.limit or self.seconds_left<=0:
            self.deferred.add(doi)
            return {'status':'DEFERRED_BY_BUDGET'}
        self.seen.add(doi)
        started=time.monotonic()
        try:result = self.base.work(doi)
        finally:self.seconds_left-=time.monotonic()-started
        self.known.add(doi)
        self.cached.cache_clear()
        return result

def triage(author, target, snap, rules, client, duplicate):
    """Signals select evidence to inspect. They cannot directly change a feature."""
    pid = snap.canonical(author['dblp_person_id'])
    year = integer(author['t0_year'])
    works = snap.history(pid,author['arxiv_id'],year) if pid else []
    profile = snap.profile(pid)['orcids']
    reasons = set()
    allowed = set()
    if duplicate:reasons.add('PID_REPEATED_IN_TEAM')
    if not pid or author['identity_status']!='VERIFIED':reasons.add('IDENTITY_NOT_VERIFIED')
    aliases = [a for a, in snap.db.execute('SELECT raw FROM aliases WHERE pid=?',(pid,))]
    if pid and not any(compatible(author['repaired_name'] or author['original_name'],a) for a in aliases):reasons.add('NAME_PID_MISMATCH')
    anchor = target['local_orcids'] | target['orcids']
    if target['conflicts'] or len(anchor)>1 or (anchor and profile and not anchor<=profile):
        reasons.add('TARGET_PROFILE_IDENTITY_CONFLICT')
    years = sorted({w['year'] for w in works})
    age = year-years[0] if years else 0
    if not same(len(works),author[PUB]) or not same(age,author[AGE]):reasons.add('HISTORY_BASELINE_MISMATCH')
    # Isolated earliest *record* can be checked, including where the XML omits ORCID.
    isolated = len(years)>=2 and years[1]-years[0]>=15
    signature_ids = set()
    for w in works:
        for r in w['records']:
            sig = r['signatures'][r['position']]['orcid']
            if sig:signature_ids.add(sig)
    for w in works:
        for r in w['records']:
            sig = r['signatures'][r['position']]['orcid']
            record_reasons = set()
            if sig and ((profile and sig not in profile) or (anchor and sig not in anchor) or (not profile and not anchor and len(signature_ids)>1)):
                record_reasons.add('HISTORICAL_SIGNATURE_CONFLICT')
            if (pid,r['key']) in rules:record_reasons.add('REVIEWED_RECORD_KEY')
            if isolated and w['year']==years[0]:record_reasons.add('ISOLATED_EARLIEST_YEAR')
            # Cached publisher metadata can reveal errors absent from XML.
            for doi in r['dois']:
                meta = client.cached(doi)
                if meta and meta.get('status')=='OK':
                    from external import align
                    people,alignment = align(meta,r['title'],r['authors'])
                    if 'CONFLICT' in alignment or 'AMBIGUOUS' in alignment:record_reasons.add('CACHED_'+alignment)
                    if people:
                        oid = people[r['position']]['orcid']
                        if oid and (profile or anchor) and oid not in (profile | anchor):
                            record_reasons.add('CACHED_PUBLISHER_SIGNATURE_CONFLICT')
            if record_reasons:
                reasons.update(record_reasons)
                allowed.update(r['dois'])
    return sorted(reasons),allowed,works

def offline_summary(a, works):
    return dict(occurrence_id=a['occurrence_id'],paper_id=a['arxiv_id'],old_pid=a['dblp_person_id'],
                new_pid=a['dblp_person_id'],status='Unresolved',new_count=a[PUB],new_age=a[AGE],
                works_examined=len(works),works_identity_supported=0,works_removed=0,
                unresolved_retained_works=len(works),unresolved_transfer_works=0,
                doi_requests_logical=0,doi_aligned=0,reasons_json='["NO_INDEPENDENT_IDENTITY_PROOF"]')

def history_csv_check(cfg, authors, progress, ledger=None):
    """Compare recovery's exported INCLUDED rows against the per-occurrence features."""
    import pandas as pd
    option=cfg.get('recovery_history_csv','auto')
    columns=['occurrence_id','included_rows','first_year','status']
    if not option:return pd.DataFrame(columns=columns)
    path=(Path(cfg['recovery_root'])/'history/final_author_publication_history.csv'
          if option=='auto' else Path(option))
    if not path.exists() and option=='auto':return pd.DataFrame(columns=columns)
    expected=authors.set_index('occurrence_id')
    ids=set(expected.index)
    counts=Counter()
    earliest={}
    invalid=set()
    progress.begin('Stream recovery history CSV',path.stat().st_size,'INCLUDED rows only; byte-based ETA')
    from contextlib import ExitStack
    with ExitStack() as stack:
        stream=stack.enter_context(path.open('rb'))
        log=stack.enter_context(ledger.open('w',encoding='utf-8-sig',newline='')) if ledger else None
        fields=['arxiv_id','occurrence_id','dblp_person_id','publication_year','t0_year','decision']
        header=pd.read_csv(path,nrows=0).columns
        fields.extend(k for k in ('work_id','author_position','source') if k in header)
        if log:pd.DataFrame(columns=['source_csv_row',*fields]).to_csv(log,index=False)
        for chunk in pd.read_csv(stream,dtype=str,keep_default_na=False,usecols=fields,chunksize=150000):
            scoped=chunk.loc[chunk.occurrence_id.isin(ids),fields]
            if log and not scoped.empty:
                scoped=scoped.copy()
                scoped.insert(0,'source_csv_row',scoped.index+2)
                scoped.to_csv(log,index=False,header=False)
            rows=scoped.loc[scoped.decision.eq('INCLUDED')].copy()
            if not rows.empty:
                rows['year']=pd.to_numeric(rows.publication_year,errors='coerce')
                cutoff=pd.to_numeric(rows.occurrence_id.map(expected.t0_year),errors='coerce')
                aligned=(rows.year.notna() & rows.year.lt(cutoff) & rows.year.mod(1).eq(0) &
                         pd.to_numeric(rows.t0_year,errors='coerce').eq(cutoff) &
                         rows.dblp_person_id.eq(rows.occurrence_id.map(expected.dblp_person_id)) &
                         rows.arxiv_id.eq(rows.occurrence_id.map(expected.arxiv_id)))
                invalid.update(rows.loc[~aligned,'occurrence_id'])
                for oid,g in rows.groupby('occurrence_id').year.agg(['size','min']).iterrows():
                    counts[oid]+=int(g['size'])
                    if pd.notna(g['min']):earliest[oid]=min(earliest.get(oid,g['min']),g['min'])
            progress.update(stream.tell())
    output=[]
    for oid,a in expected.iterrows():
        age=int(a.t0_year)-earliest[oid] if oid in earliest else 0
        status='MATCH' if oid not in invalid and same(counts[oid],a[PUB]) and same(age,a[AGE]) else 'CONFLICT'
        output.append((oid,counts[oid],earliest.get(oid),status))
    return pd.DataFrame(output,columns=columns)
