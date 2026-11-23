#!/usr/bin/env python3
"""Offline Fetch bootstrap and DBLP recovery; initialize a missing local profile index."""
import argparse,json
from config import Config
from recovery.workflow import run
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='config.json');p.add_argument('--stage',choices=['prepare','scan','all','treatment'],default='all');args=p.parse_args()
    cfg=Config.load(args.config)
    from missing_treatment import run as treat
    if args.stage=='treatment':result=treat(cfg)
    else:
        result=run(cfg,args.stage)
        if args.stage=='all':result['missing_treatment']=treat(cfg)
    print(json.dumps(result,ensure_ascii=False,indent=2))
