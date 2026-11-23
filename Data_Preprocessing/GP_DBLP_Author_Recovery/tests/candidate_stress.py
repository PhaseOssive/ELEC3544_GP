"""Many-candidate stress test, kept separate from the 30k unique-name cohort."""
import sys,json,time,resource,sqlite3
from pathlib import Path
from fixtures import create
from recovery.workflow import run
root=Path(sys.argv[1]);count=int(sys.argv[2]);profiles={'a':['Alice Jones'],'b':['Bob Green']};pubs=[]
for i in range(count):
    name=f'John Smith {i+1:04d}';profiles['p'+str(i)]=['John Smith',name];pubs.append(dict(key='hist/'+str(i),title='Candidate historical record '+str(i),year=2010,authors=[name],links=[]))
pubs.append(dict(key='target',title='Target title',year=2020,authors=['Alice Jones','John Smith 0001','Bob Green'],links=['https://arxiv.org/abs/2001.00001']))
rows=[dict(arxiv_id='2001.00001',author_position=i+1,original_name=n,identity_status='UNRESOLVED' if i==1 else 'VERIFIED',dblp_person_id='' if i==1 else 'a' if i==0 else 'b') for i,n in enumerate(['Alice Jones','John Smith','Bob Green'])]
cfg=create(root,[dict(arxiv_id='2001.00001',title_v1='Target title')],rows,profiles,pubs);start=time.time();summary=run(cfg)
db=sqlite3.connect(Path(cfg.DERIVED_CACHE_DIR)/'candidate_history.sqlite');actual=db.execute("SELECT COUNT(DISTINCT pid) FROM candidates WHERE oid='2001.00001:2' AND eligible=1").fetchone()[0]
assert actual==count,(actual,count)
r=dict(synthetic_candidates=count,eligible_candidates_retained=actual,elapsed_seconds=time.time()-start,peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,full_workflow=summary)
(root/'candidate_benchmark.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
