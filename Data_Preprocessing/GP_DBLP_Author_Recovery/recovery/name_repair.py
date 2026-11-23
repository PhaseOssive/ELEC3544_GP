"""Existing evidence-based Alias segmentation, integrated with deep alignment."""
import re,json
from .rules import normalize_name
from .names import tokens
def segment_by_alias(name,index,max_parts=3):
    ts=re.findall(r'[^\s,;]+',re.sub(r'(?<=[a-z])(?=[A-Z])',' ',str(name)).replace('. ',' '));memo={}
    def visit(start,remaining):
        if (start,remaining) in memo:return memo[start,remaining]
        if start==len(ts):return [()]
        if remaining==0:return []
        out=[]
        for end in range(start+1,min(len(ts),start+7)+1):
            text=' '.join(ts[start:end]);ids,method,_=index.tier(text)
            if method=='UNIQUE_INITIAL_SURNAME_ALIAS' or len(ids)!=1:continue
            for tail in visit(end,remaining-1):out.append(((text,next(iter(ids))),)+tail)
        memo[start,remaining]=out;return out
    return [s for s in visit(0,max_parts) if len(s)>1 and len({pid for _,pid in s})==len(s)]

def split_merged(rows,selected,index):
    options={};audit=[]
    for r in rows:
        if len(tokens(re.sub(r'(?<=[a-z])(?=[A-Z])',' ',r['original_name'])))<4 and r['identity_status']!='MERGED_AUTHOR_RECORD_REQUIRES_REVIEW':continue
        for parts in segment_by_alias(r['original_name'],index):
            pids=[pid for _,pid in parts]
            for c in selected:
                names=c['record']['authors'];ids=c['pids'];ev=c['evidence'];pos=int(r['author_position'])-1
                for j in range(len(ids)-len(pids)+1):
                    if ids[j:j+len(pids)]!=pids or normalize_name(''.join(names[j:j+len(pids)]))!=normalize_name(r['original_name']):continue
                    expected=len(rows)+len(parts)-1;anchors=[];bad=[]
                    for other in rows:
                        if other['identity_status']!='VERIFIED':continue
                        op=int(other['author_position'])-1;mapped=op+(len(parts)-1 if op>pos else 0)
                        (anchors if mapped<len(ids) and index.canonical(other['dblp_person_id'])==ids[mapped] else bad).append(op)
                    accepted=expected==len(names) and j==pos and not bad and len(anchors)>=min(2,len(rows)-1) and ev['reliable']
                    record=dict(arxiv_id=r['arxiv_id'],original_author_position=r['author_position'],original_name=r['original_name'],split_names_json=json.dumps([n for n,_ in parts],ensure_ascii=False),split_pids_json=json.dumps(pids),record_key=c['record']['key'],corrected_count=expected,aligned_anchor_count=len(anchors),evidence_json=json.dumps(ev),accepted=accepted)
                    audit.append(record)
                    if accepted:options.setdefault(int(r['author_position']),{})[tuple(pids)]=(parts,record)
    final=[];verified=[]
    for r in rows:
        oldpos=int(r['author_position']);variants=options.get(oldpos,{})
        if len(variants)==1:
            parts,evidence=next(iter(variants.values()));verified.append(evidence)
            for seg,(name,pid) in enumerate(parts,1):final.append({**r,'original_author_position':oldpos,'segment_index':seg,'repaired_name':name,'resolved_pid':pid,'resolution_evidence':evidence,'split_verified':True})
        else:final.append({**r,'original_author_position':oldpos,'segment_index':1,'repaired_name':r['original_name'],'split_verified':False})
    for position,r in enumerate(final,1):r['author_position']=position;r['occurrence_id']=f"{r['arxiv_id']}:{r['original_author_position']}:{r['segment_index']}"
    return final,audit,verified

