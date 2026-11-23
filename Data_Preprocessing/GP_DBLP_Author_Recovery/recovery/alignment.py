"""Order-preserving top-two DP with fixed verified anchors and bounded windows."""
import json
from difflib import SequenceMatcher
from .rules import normalize_name
from .names import tokens,compatible

def block_similarity(source,target):
    a=''.join(normalize_name(x) for x in source);b=''.join(normalize_name(x) for x in target)
    if a==b:return 1.0,'EXACT_TOKEN_FLOW'
    if len(source)==len(target)==1 and compatible(source[0],target[0]):return .96,'NAME_VARIANT'
    # Token overlap tolerates missing middle names; must retain strong block similarity.
    x=' '.join(t for n in source for t in tokens(n));y=' '.join(t for n in target for t in tokens(n))
    char=SequenceMatcher(None,a,b).ratio();token=SequenceMatcher(None,x,y).ratio()
    return max(char,token),'TOKEN_REALLOCATION_OR_SPELLING'

def align(rows,target_names,target_pids,cfg,reliable):
    n,m=len(rows),len(target_names);states={(0,0):[(0.,[])]};anchors=sum(r['identity_status']=='VERIFIED' for r in rows)
    for i in range(n+1):
        for j in range(m+1):
            paths=states.get((i,j),[])
            if not paths:continue
            for a in range(1,min(cfg.alignment_window,n-i)+1):
                src=rows[i:i+a]
                for b in range(1,min(cfg.alignment_window,m-j)+1):
                    dst=target_names[j:j+b];pids=target_pids[j:j+b]
                    # A verified occurrence is an immutable 1:1 anchor.
                    if any(r['identity_status']=='VERIFIED' for r in src):
                        if a!=1 or b!=1 or pids[0]!=src[0]['dblp_person_id']:continue
                        sim,reason=1.,'FIXED_VERIFIED_ANCHOR'
                    else:
                        sim,reason=block_similarity([r.get('repaired_name') or r['original_name'] for r in src],dst)
                        if sim<cfg.alignment_min_similarity:continue
                    if a==b and a>1 and all(compatible(x.get('repaired_name') or x['original_name'],y) for x,y in zip(src,dst)):continue
                    # Positive cost; avoid artificial many-to-many wins on exact 1:1 chains.
                    cost=(1-sim)*max(a,b)+.025*(a+b-2)
                    edge=dict(source_start=i,source_count=a,target_start=j,target_count=b,similarity=sim,operation='ONE_TO_ONE' if a==b==1 else 'SPLIT' if a==1 else 'MERGE' if b==1 else 'TOKEN_REALLOCATION',reason=reason)
                    opts=states.setdefault((i+a,j+b),[])
                    opts.extend((score+cost,path+[edge]) for score,path in paths)
                    unique={json.dumps(path,sort_keys=True):(score,path) for score,path in opts}
                    states[i+a,j+b]=sorted(unique.values(),key=lambda x:x[0])[:2]
    best=states.get((n,m),[])
    if not best:return dict(accepted=False,reason='NO_ANCHOR_CONSISTENT_ALIGNMENT',evidence=[])
    score,path=best[0];second=best[1][0] if len(best)>1 else None;gap=second-score if second is not None else None
    # Exact unchanged edges can have alternate block grouping with same PID sequence;
    # only distinct source-to-target assignments compete, not serialization choices.
    def assignment(edges):return tuple((e['source_start'],e['source_count'],e['target_start'],e['target_count']) for e in edges if e['operation']!='ONE_TO_ONE')
    changed=[e for e in path if e['reason']!='FIXED_VERIFIED_ANCHOR']
    accepted=bool(reliable and all(target_pids) and len(set(target_pids))==m and (gap is None or gap>=cfg.alignment_margin) and (anchors>=2 or all(e['similarity']==1 and e['operation']=='ONE_TO_ONE' for e in changed)))
    evidence=[]
    for e in path:
        orig=rows[e['source_start']:e['source_start']+e['source_count']]
        for j in range(e['target_start'],e['target_start']+e['target_count']):
            evidence.append(dict(original_author_names_json=json.dumps([r['original_name'] for r in orig]),original_positions_json=json.dumps([r['author_position'] for r in orig]),source_occurrences_json=json.dumps([r['occurrence_id'] for r in orig]),recovered_author_name=target_names[j],dblp_author_position=j+1,candidate_pid=target_pids[j],alignment_operation=e['operation'],name_similarity=e['similarity'],anchor_consistency=True,alignment_score=score,second_best_score=second,score_margin=gap,confidence='HIGH' if accepted else 'MEDIUM',resolution_status='ACCEPTED' if accepted else 'REVIEW',source_start=e['source_start'],source_count=e['source_count']))
    return dict(accepted=accepted,reason='UNIQUE_ANCHORED_ALIGNMENT' if accepted else 'INSUFFICIENT_ALIGNMENT_MARGIN_OR_ANCHORS',evidence=evidence,best_score=score,second_best_score=second)
