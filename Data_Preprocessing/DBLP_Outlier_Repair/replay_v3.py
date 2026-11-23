"""Read-only forensic comparison over v3 exports. This does not regenerate absent DBLP works."""
import argparse,json
from pathlib import Path
import pandas as pd
from common import read,sha,write,dump
from trace31 import resolve

def compare(baseline,master_file,team_file,revalidation,out,alignment=None):
    out=Path(out);out.mkdir(exist_ok=True,parents=True)
    baseline=Path(baseline)
    summary=json.loads((baseline/'summary.json').read_text())
    if sha(master_file)!=summary['input_sha256']:raise ValueError('Wrong master for v3 run')
    master=read(master_file,usecols=['paper_id','authors_v1_json','author_count_v1']).set_index('paper_id',drop=False).to_dict('index')
    teams=read(team_file).set_index('arxiv_id',drop=False).to_dict('index')
    rec=read(revalidation,usecols=['arxiv_id','author_position','canonical_pid','original_name'])
    lookup={}
    for row in rec.itertuples(index=False):
        key=(row.arxiv_id,row.author_position,row.canonical_pid)
        if key in lookup:raise ValueError('Duplicate recovery tuple')
        lookup[key]=row.original_name
    a=read(baseline/'author_verdicts.csv');papers=read(baseline/'paper_verdicts.csv')
    changes=read(baseline/'changes.csv')
    detail=read(baseline/'work_evidence.csv',usecols=['paper_id','occurrence_id','work_id','record_key'])
    if set(detail.occurrence_id)-set(a.occurrence_id) or a.occurrence_id.duplicated().any():raise ValueError('Invalid occurrence keys')
    dedup=detail.drop_duplicates(['occurrence_id','work_id']).groupby('occurrence_id').size()
    aligned={}
    if alignment:
        for e in read(alignment).itertuples(index=False):
            if e.resolution_status=='ACCEPTED' and e.candidate_pid:
                aligned.setdefault((e.arxiv_id,e.candidate_pid),set()).add((str(int(float(e.dblp_author_position))),e.recovered_author_name))
    rows=[]
    for row in a.itertuples(index=False):
        aid=row.paper_id
        entry=aligned.get((aid,row.old_pid),set())
        alignment_name=''
        if len(entry)==1:position,alignment_name=next(iter(entry))
        elif row.occurrence_id.startswith(aid+':') and row.occurrence_id[len(aid)+1:].isdigit():
            position=row.occurrence_id[len(aid)+1:]
        else:raise ValueError('Explicit aligned position required for '+row.occurrence_id)
        old=row.author_name
        exact=lookup.get((aid,position,row.old_pid),'') if row.old_pid else ''
        name,source,flags,ev=resolve(dict(arxiv_id=aid,author_position=position,dblp_person_id=row.old_pid,original_name=exact,repaired_name=old or alignment_name),master[aid],teams.get(aid))
        if source=='recovery_repaired':source='v3_existing_name' if old else 'accepted_alignment_exact_pid_position'
        if source=='recovery_original':source='revalidation_exact_tuple'
        rows.append(dict(paper_id=aid,occurrence_id=row.occurrence_id,author_position=position,pid=row.old_pid,v3_name=old,display_name=name,name_source=source,flags_json=json.dumps(flags),v3_works_examined=row.works_examined,v3_distinct_logged=dedup.get(row.occurrence_id,0)))
    df=pd.DataFrame(rows);write(df,out/'observed_name_audit.csv')
    result=dict(mode='READ_ONLY_REPLAY_OF_REAL_V3_EXPORTS',paper_count=len(papers),author_count=len(a),v3_missing_names=int(a.author_name.eq('').sum()),display_names_missing_after_replay=int(df.display_name.eq('').sum()),v3_papers_all_names_empty=int(a.groupby('paper_id').author_name.apply(lambda x:x.eq('').all()).sum()),papers_all_names_empty_after_replay=int(df.groupby('paper_id').display_name.apply(lambda x:x.eq('').all()).sum()),name_sources=df.name_source.value_counts().to_dict(),flags=df.flags_json.value_counts().to_dict(),v3_works_examined=int(a.works_examined.astype(int).sum()),v3_work_detail_rows=len(detail),v3_distinct_logged=int(dedup.sum()),v3_unlogged_distinct_work_occurrences=int(a.works_examined.astype(int).sum()-dedup.sum()),v3_changed_papers=changes.paper_id.nunique(),v3_changed_cells=len(changes),new_repairs_executed=0)
    dump(result,out/'observed_comparison.json');print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('baseline','master','team','revalidation','out'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--alignment');a=parser.parse_args();compare(a.baseline,a.master,a.team,a.revalidation,a.out,a.alignment)
