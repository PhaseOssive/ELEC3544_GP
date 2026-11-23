"""Original temporal and zero/missing semantics, disk-backed evidence queries."""
import csv,json
from .progress import tqdm
PUB='dblp_author_publication_count_pre_t0'
AGE='dblp_author_academic_age_pre_t0'
FIRST='dblp_author_first_publication_year_pre_t0'

def recover(cfg,store,rows):
    db=store.db;path=cfg.out('history/final_author_publication_history.csv');tmp=path.with_suffix('.csv.tmp');result=[];included_bad=0
    with tmp.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['arxiv_id','author_position','occurrence_id','dblp_person_id','work_id','publication_year','t0_year','decision','source'])
        for r in tqdm(rows,desc='Strict pre-t0 author statistics',unit='authors'):
            r=dict(r);r[PUB]=r[AGE]=r[FIRST]=None;r['history_status']='UNRESOLVED_IDENTITY';r['dblp_focal_work_excluded_from_history']=False
            pid=r.get('dblp_person_id','');year=int(r['t0_year']);usable=0;first=None
            if r['identity_status']=='VERIFIED' and pid:
                status=db.execute('SELECT status FROM candidate_history_status WHERE pid=?',(pid,)).fetchone();status=status[0] if status else 'NOT_SCANNED'
                focal={x[0] for x in db.execute('SELECT rw.work FROM focal f JOIN record_works rw ON rw.key=f.key WHERE f.aid=?',(r['arxiv_id'],))}
                for wid,y in db.execute('SELECT work,year FROM work_years WHERE pid=? ORDER BY year,work',(pid,)):
                    decision='FOCAL_WORK_EXCLUDED' if wid in focal else 'SAME_OR_POST_T0_YEAR' if y>=year else 'INCLUDED'
                    if decision=='INCLUDED':usable+=1;first=y if first is None else min(first,y);included_bad+=int(y>=year or wid in focal)
                    if decision=='FOCAL_WORK_EXCLUDED' and y<year:r['dblp_focal_work_excluded_from_history']=True
                    w.writerow([r['arxiv_id'],r['author_position'],r['occurrence_id'],pid,wid,y,year,decision,'COMPLETE_LOCAL_DBLP_XML'])
                if status=='COMPLETE_OBSERVED_SNAPSHOT':
                    r[PUB]=usable;r[FIRST]=first;r[AGE]=year-first if first is not None else None;r['history_status']='COMPLETE_OBSERVED_DBLP_HISTORY' if usable else 'ZERO_PRIOR_WORKS_IN_DBLP_SNAPSHOT'
                else:r['history_status']=status
            result.append(r)
    tmp.replace(path)
    if included_bad:raise AssertionError('Temporal/focal boundary violation')
    return result,dict(included_same_or_future_year=0,included_focal_works=0,publication_year_boundary='publication_year < t0_year',unresolved_zero_fill=False)
