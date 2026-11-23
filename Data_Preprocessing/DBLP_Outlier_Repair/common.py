"""Evidence-gated DBLP-only repair. Python 3.10+; no network needed at runtime."""
import argparse, hashlib, json, math, shutil, sqlite3, tempfile
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd

FIELDS=[f'dblp_team_{op}_{metric}_pre_t0' for metric in ('publication_count','academic_age') for op in ('mean','max')]
PUB='dblp_author_publication_count_pre_t0'
AGE='dblp_author_academic_age_pre_t0'
FIRST='dblp_author_first_publication_year_pre_t0'
COMPLETE='COMPLETE_OBSERVED_DBLP_HISTORY'
ZERO='ZERO_PRIOR_WORKS_IN_DBLP_SNAPSHOT'

def read(path, **kw):
    return pd.read_csv(path,dtype=str,keep_default_na=False,**kw)

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def number(v):
    if str(v).strip()=='': return float('nan')
    return float(v)

def same(a,b):
    try: return bool(np.isclose(number(a),number(b),equal_nan=True,rtol=1e-12,atol=1e-10))
    except ValueError: return False

def integer(v):
    f=float(v)
    if not math.isfinite(f) or f!=int(f): raise ValueError('Expected integer: '+str(v))
    return int(f)

def write(df,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');df.to_csv(tmp,index=False,encoding='utf-8-sig');tmp.replace(path)

def dump(obj,path):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def unique(df,key,label):
    if key not in df or df[key].eq('').any() or df[key].duplicated().any():
        raise ValueError('Missing/duplicate '+label+' IDs')

def screen(master):
    flags=[];stats=[]
    for f in FIELDS:
        x=pd.to_numeric(master[f],errors='coerce');finite=x[np.isfinite(x)]
        q1,q3=finite.quantile([.25,.75]);lo=q1-1.5*(q3-q1);hi=q3+1.5*(q3-q1)
        bad=(master[f].ne('')&(~np.isfinite(x)))|(x<0)
        if '_max_' in f: bad |= x.notna() & (x%1!=0)
        mask=(x<lo)|(x>hi)|bad
        for i in master.index[mask]: flags.append({'paper_id':master.at[i,'paper_id'],'field':f,'value':master.at[i,f],'reason':'INVALID_VALUE' if bad[i] else 'IQR_SCREEN_ONLY'})
        stats.append({'field':f,'observed':int(x.notna().sum()),'missing':int(master[f].eq('').sum()),'q1':float(q1),'q3':float(q3),'upper_fence':float(hi),'flags':int(mask.sum())})
    for metric in ('publication_count','academic_age'):
        a=pd.to_numeric(master[f'dblp_team_mean_{metric}_pre_t0'],errors='coerce');b=pd.to_numeric(master[f'dblp_team_max_{metric}_pre_t0'],errors='coerce')
        for i in master.index[a>b+1e-10]:flags.append({'paper_id':master.at[i,'paper_id'],'field':metric,'value':'','reason':'MEAN_EXCEEDS_MAX'})
    return pd.DataFrame(flags,columns=['paper_id','field','value','reason']),stats

def aggregate(authors):
    counts=[a[0] for a in authors];ages=[a[1] for a in authors]
    return [float(np.mean(counts)),float(max(counts)),float(np.mean(ages)),float(max(ages))]
