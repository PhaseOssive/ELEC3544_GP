"""Short reproducible synthetic benchmark. No Drive data or network is used."""
import hashlib
import json
import sqlite3
import tempfile
import time
from pathlib import Path
import pandas as pd
from common import FIELDS, PUB, AGE, FIRST
from main import run

def fixture(root, papers=64, authors_per_paper=3, works=96):
    recovery=root/'recovery'
    for name in ('features','report'):(recovery/name).mkdir(parents=True)
    main_rows=[];author_rows=[];teams=[];xml=['<dblp>']
    for p in range(papers):
        aid=f'2001.{p:05}';title=f'Synthetic focal {p}';names=[]
        for pos in range(1,authors_per_paper+1):
            i=p*authors_per_paper+pos;name=f'Author Person{i}';names.append(name)
            pid=f'homepages/synthetic/{i}'
            xml.append(f'<www key="{pid}"><author>{name}</author></www>')
            for w in range(works):
                xml.append(f'<article key="r{i}/{w}"><author>{name}</author><year>{2000+w%19}</year><title>Work {i} part {w}</title><ee>https://doi.org/10.9999/r{i}.{w}</ee></article>')
            author_rows.append(dict(arxiv_id=aid,author_position=pos,occurrence_id=aid+':'+str(pos),original_name=name,repaired_name=name,dblp_person_id=pid,identity_status='VERIFIED',history_status='COMPLETE_OBSERVED_DBLP_HISTORY',t0_year=2020,focal_record_keys_json=json.dumps([f'f{p}']),**{PUB:works,AGE:20,FIRST:2000}))
        xml.append(f'<article key="f{p}">'+''.join(f'<author>{n}</author>' for n in names)+f'<year>2020</year><title>{title}</title></article>')
        vals=dict(zip(FIELDS,[works,works,20,20]))
        main_rows.append(dict(paper_id=aid,title_v1=title,submission_year=2020,authors_v1_json=json.dumps([{'name':n} for n in names]),author_count_v1=authors_per_paper,**vals))
        teams.append(dict(arxiv_id=aid,expected_author_count=authors_per_paper,**vals))
    for i in range(4*papers):
        main_rows.append(dict(paper_id=f'filler{i}',title_v1='Synthetic filler',submission_year=2020,authors_v1_json='[]',author_count_v1=0,**dict.fromkeys(FIELDS,0)))
    xml.append('</dblp>');(root/'dblp.xml').write_text(''.join(xml));(root/'dblp.dtd').write_text('')
    (root/'rules.json').write_text('[]')
    pd.DataFrame(main_rows).to_csv(root/'master.csv',index=False)
    pd.DataFrame(author_rows).to_csv(recovery/'features/treated_author_features.csv',index=False)
    pd.DataFrame(teams).to_csv(recovery/'features/treated_team_features.csv',index=False)
    (recovery/'report/final_recovery_summary.json').write_text(json.dumps(dict(full_xml_scan_complete=True,final_author_occurrences=len(author_rows))))
    return dict(input_csv=str(root/'master.csv'),xml_path=str(root/'dblp.xml'),dtd_path=str(root/'dblp.dtd'),reviewed_rules=str(root/'rules.json'),recovery_root=str(recovery),local_work_dir=str(root/'work'),cache_root=str(root/'cache'),output_root=str(root/'out'),crossref_enabled=False,biography_enabled=False,stage_xml_locally=False,author_chunk_size=1000)

def measure():
    with tempfile.TemporaryDirectory() as d:
        root=Path(d);cfg=fixture(root);results={}
        for name in ('cold','warm'):
            if name=='warm':
                # Remove only this temporary fixture's author decisions. The benchmark measures fresh screening.
                next(root.glob('cache/*/audit_state.sqlite')).unlink()
            start=time.monotonic();out=run(cfg);elapsed=time.monotonic()-start
            s=json.loads((out/'summary.json').read_text())
            timings=json.loads((out/'progress.json').read_text())['completed_phases']
            assert s['outlier_papers']==64 and s['changed_cells']==s['external_http_requests']==0
            results[name]=dict(seconds=round(elapsed,3),phases=timings,papers=64,occurrences=192,pre_t0_work_occurrences=18432,external_requests=0)
        return results

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='benchmark_results.json')
    args=parser.parse_args();Path(args.output).write_text(json.dumps(measure(),indent=2))
