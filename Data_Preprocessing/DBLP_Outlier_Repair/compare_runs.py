"""Compare two completed runs only if their master and candidate sets match exactly."""
import json
from pathlib import Path
from common import read,write,dump
def comparison(old_dir,out):
    if not old_dir:return
    old=Path(old_dir)
    if not (old/'summary.json').is_file():return
    prior=json.loads((old/'summary.json').read_text());current=json.loads((out/'summary.json').read_text())
    if prior.get('input_sha256')!=current.get('input_sha256'):return
    a=read(old/'paper_verdicts.csv')
    b=read(out/'paper_verdicts.csv')
    if set(a.paper_id)!=set(b.paper_id):return
    joined=a.merge(b,on='paper_id',validate='one_to_one',suffixes=('_v3','_v31'))
    write(joined,out/'paper_comparison_v3.csv')
    old_auth=read(old/'author_verdicts.csv');new_auth=read(out/'author_verdicts.csv')
    old_rows=sum(len(chunk) for chunk in read(old/'work_evidence.csv',usecols=['occurrence_id'],chunksize=100000))
    dump(dict(baseline=str(old),mode=current['mode'],papers=len(joined),
              missing_author_names_before=int(old_auth.author_name.eq('').sum()),
              missing_author_names_after=int(new_auth.author_name.eq('').sum()),
              v3_work_detail_rows=old_rows,v31_work_detail_rows=current['evidence_record_rows'],
              v31_covered_works=current['distinct_occurrence_works'],v3_applied_cells=prior['changed_cells'],
              v31_proposed_cells=current['proposed_cells'],v31_applied_cells=current['changed_cells']),out/'comparison_v3.json')
