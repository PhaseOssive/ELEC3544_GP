"""Synthetic end-to-end benchmark. Does not represent the real DBLP distribution."""
import argparse,json,time,sqlite3,resource,sys
from pathlib import Path
from fixtures import create
from recovery.workflow import run

def benchmark(root,n,fresh_index=False):
    start=time.time();papers=[];authors=[];profiles={};pubs=[]
    for i in range(n):
        aid=f'{2000+i//10000:04d}.{i%10000:05d}';names=[f'Alice{i:06d} Family{i:06d}',f'Bob{i:06d} Other{i:06d}'];pids=[f'homepages/a{i}',f'homepages/b{i}']
        papers.append(dict(arxiv_id=aid,title_v1=f'Synthetic distinct topic {i} learning study'))
        for j in range(2):
            profiles[pids[j]]=[names[j]];authors.append(dict(arxiv_id=aid,author_position=j+1,original_name=names[j],identity_status='VERIFIED' if j==0 else 'UNRESOLVED',dblp_person_id=pids[j] if j==0 else ''))
        pubs.extend([dict(key=f'past/{i}',title=f'Historical topic {i}',year=2015,authors=names,links=[f'https://doi.org/10.9999/h{i}']),dict(key=f'focal/{i}',title=papers[-1]['title_v1'],year=2020,authors=names,links=['https://arxiv.org/abs/'+aid])])
    cfg=create(root,papers,authors,profiles,pubs)
    if fresh_index:Path(cfg.PROFILE_INDEX_PATH).unlink()
    # Release generator structures before measuring core run peak independently is not
    # possible via ru_maxrss; report whole-process peak, including fixture construction.
    del papers,authors,profiles,pubs
    setup=time.time()-start;s=run(cfg)
    db=sqlite3.connect(Path(cfg.DERIVED_CACHE_DIR)/'candidate_history.sqlite')
    plans={name:[list(r) for r in db.execute('EXPLAIN QUERY PLAN '+query)] for name,query in {
      'pid_history':"SELECT key FROM candidate_publications WHERE pid='homepages/a0'",
      'pid_year':"SELECT key FROM candidate_publication_years WHERE pid='homepages/a0' AND year<2020",
      'coauthors':"SELECT work FROM candidate_coauthors WHERE oid='x' AND pid='p'",
      'work_years':"SELECT work,year FROM work_years WHERE pid='homepages/a0'"}.items()}
    q=time.perf_counter()
    for i in range(10000):db.execute('SELECT key FROM candidate_publications WHERE pid=?',(f'homepages/a{i%n}',)).fetchall()
    query_time=time.perf_counter()-q;db.close()
    report=dict(dataset='SYNTHETIC_UNIQUE_NAMES_TWO_AUTHORS_ONE_HISTORY_AND_ONE_TARGET_PER_PAPER',paper_count=n,xml_publications=2*n,xml_profiles=2*n,xml_total_records=4*n,author_occurrences=2*n,fixture_setup_seconds=round(setup,3),workflow=s,ten_thousand_indexed_queries_seconds=round(query_time,3),query_plans=plans,sqlite_bytes=(Path(cfg.DERIVED_CACHE_DIR)/'candidate_history.sqlite').stat().st_size,whole_process_peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,limitations=['Not a full DBLP snapshot benchmark','Unique names do not model thousands of ambiguous candidates','Includes real XML streaming, SQL history/dedup, alignment, scoring/backtest, features and exports'])
    assert s['paper_count']==n and s['verified_author_count']==2*n and s['unresolved_author_count']==0,s
    Path(root,'benchmark.json').write_text(json.dumps(report,indent=2));return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--papers',type=int,default=30000);p.add_argument('--root',required=True);p.add_argument('--fresh-index',action='store_true');a=p.parse_args();print(json.dumps(benchmark(a.root,a.papers,a.fresh_index),indent=2))
