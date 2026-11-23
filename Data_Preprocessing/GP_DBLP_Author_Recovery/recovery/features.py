"""Strict complete-team mean/max; partial statistics never enter official fields."""
import pandas as pd
from .progress import tqdm
from .history import PUB,AGE
from .io import write_csv

FEATURES=['dblp_team_'+op+'_'+metric+'_pre_t0' for metric in ['publication_count','academic_age'] for op in ['mean','max']]

def aggregate(papers,authors,legacy=False):
    if not isinstance(authors,pd.DataFrame):authors=pd.DataFrame(authors)
    rows=[]
    groups={aid:g for aid,g in authors.groupby("arxiv_id",sort=False)}
    for p in tqdm(papers.to_dict('records'),desc='Recalculate four team features',unit='papers'):
        g=groups.get(p['arxiv_id'],authors.iloc[:0]);expected=int(p['expected_author_count'])
        mask=g.dblp_author_matched_to_paper.astype(str).str.lower().eq('true')
        if not legacy and 'identity_status' in g:mask&=g.identity_status.eq('VERIFIED')
        matched=int(mask.sum());positions=sorted(g.author_position.astype(int))
        complete=expected>0 and len(g)==expected and matched==expected and positions==list(range(1,expected+1))
        if not legacy:complete=complete and g.dblp_person_id.nunique()==expected
        r={'arxiv_id':p['arxiv_id'],'matched_author_count':matched,'expected_author_count':expected,'author_coverage_ratio':matched/expected if expected else None,'partial_team_status':'COMPLETE_IDENTITY_SET' if complete else 'INCOMPLETE_IDENTITY_SET'}
        for column,label in [(PUB,'publication_count'),(AGE,'academic_age')]:
            val=pd.to_numeric(g[column],errors='coerce');valid=complete and val.notna().all()
            for op in ['mean','max']:
                r[f'dblp_team_{op}_{label}_pre_t0']=float(getattr(val,op)()) if valid else None
                partial=val[mask].dropna();r[f'partial_team_{op}_{label}_pre_t0']=float(getattr(partial,op)()) if len(partial) else None
            r[f'{label}_valid_author_count']=int(val[mask].notna().sum())
            r[f'{label}_status']='COMPLETE' if valid else 'INCOMPLETE_AUTHOR_OR_HISTORY_COVERAGE'
            r['dblp_total_author_publication_count_pre_t0' if column==PUB else 'dblp_total_academic_age_pre_t0']=float(val.sum()) if valid else None
        rows.append(r)
    return pd.DataFrame(rows)
