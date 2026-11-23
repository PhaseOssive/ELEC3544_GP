"""Read-only input discovery, schema validation and atomic output."""
from pathlib import Path
import hashlib,json,logging
import pandas as pd


FILES = ['papers_master_raw_stage.csv','authors_arxiv_v1.csv','dblp_author_features.csv',
 'dblp_author_features_by_author.csv','dblp_candidate_match_audit.csv',
 'dblp_identity_conflicts.csv','dblp_identity_priority_audit.csv','dblp_source_metadata.json']
KEY = ['arxiv_id','author_position']

def write_csv(cfg, name, rows, columns=None):
    p=cfg.out(name); tmp=p.with_name(p.name+'.tmp')
    df=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows,columns=columns)
    df.to_csv(tmp,index=False,encoding='utf-8-sig'); tmp.replace(p)

def write_json(cfg,name,data):
    p=cfg.out(name); t=p.with_name(p.name+'.tmp')
    t.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str),encoding='utf-8');t.replace(p)

