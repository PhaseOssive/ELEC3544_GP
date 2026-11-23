"""Explicitly synthetic 599-row preservation/treatment integration validation."""
import sys,json,random,time
from pathlib import Path
from fixtures import create
from recovery.workflow import run as recover
from recovery.features import FEATURES
from common import sha256
from missing_treatment import run


def benchmark(root):
    root=Path(root);papers=[];authors=[];profiles={};pubs=[]
    for i in range(599):
        aid=f'2001.{i:05d}';name=f'Alice{i:06d} Family{i:06d}';pid=f'homepages/test{i}';kind=i%4
        papers.append(dict(arxiv_id=aid,title_v1=f'Target synthetic paper {i}',Y1='001',Y2='NA',Y3='label, with\nnewline',original_extra_field='keep: '+str(i),**{f:'' for f in FEATURES}))
        authors.append(dict(arxiv_id=aid,author_position=1,original_name=name,identity_status='UNRESOLVED' if kind==2 else 'VERIFIED',dblp_person_id='' if kind==2 else pid))
        if kind!=2:profiles[pid]=[name]
        pubs.append(dict(key=f'focal/{i}',title=papers[-1]['title_v1'],year=2020,authors=[name],links=['https://arxiv.org/abs/'+aid]))
        if kind in (1,3):pubs.append(dict(key=f'past/{i}',title=f'Independent prior publication {i}',year=2015 if kind==1 else None,authors=[name],links=[f'https://doi.org/10.1234/past{i}']))
    random.Random(17).shuffle(papers)
    cfg=create(root,papers,authors,profiles,pubs);recover(cfg)
    master=Path(cfg.FETCH_DATA_ROOT)/cfg.master_file
    protected=[master,Path(cfg.PROFILE_INDEX_PATH),Path(cfg.DERIVED_CACHE_DIR)/'candidate_history.sqlite',Path(cfg.OUTPUT_DIR)/'report/temporal_integrity_audit.json']
    hashes={str(p):sha256(p) for p in protected}
    # Prove no dependency on XML for the final treatment step.
    Path(cfg.DBLP_XML_PATH).rename(root/'xml_not_available_during_treatment.gz')
    start=time.perf_counter();result=run(cfg);elapsed=time.perf_counter()-start
    assert hashes=={str(p):sha256(p) for p in protected}
    assert result['retained_paper_count']==599 and result['excluded_paper_count']==0
    assert result['zero_history_author_count']==150 and result['unresolved_author_count']==150
    assert result['history_missing_year_author_count']==149
    for f in FEATURES:
        assert result['final_features'][f]['valid_count']==300
        assert result['final_features'][f]['missing_count']==299
    for f in FEATURES[2:]:assert result['before_after_coverage'][f]['before']['valid_count']==150
    report={'dataset':'SYNTHETIC_NOT_REAL_599_PAPERS','treatment_seconds':elapsed,'protected_files_unchanged':True,'summary':result}
    (root/'treatment_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    return report
if __name__=='__main__':print(json.dumps(benchmark(sys.argv[1]),ensure_ascii=False,indent=2))
