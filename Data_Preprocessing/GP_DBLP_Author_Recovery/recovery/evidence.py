"""Existing paper-level identity and split rules, before whole-list alignment."""
import json
from collections import defaultdict,Counter
from .names import compatible,edit_distance,base_name
from .rules import normalize_name
from .name_repair import split_merged
from .io import write_csv
from .progress import tqdm

def paper_recovery(cfg,store,index,papers,rows):
    groups=defaultdict(list);out=[];audit=[];splitaudit=[];splitverified=[];pidconflicts=[]
    for r in rows:groups[r['arxiv_id']].append(r)
    for p in tqdm(papers,desc='Foundational paper identity and name recovery',unit='papers'):
        aid=p['arxiv_id'];g=sorted(groups[aid],key=lambda r:r['author_position']);selected=[]
        candidates=list(store.db.execute('SELECT t.key,t.method,p.payload FROM targets t JOIN publications p USING(key) WHERE t.aid=? AND t.reliable=1',(aid,)))
        ranks={'ARXIV_ID':3,'DOI':2,'EXACT_TITLE':1}
        if candidates:
            rank=max(ranks.get(c[1],0) for c in candidates);candidates=[c for c in candidates if ranks.get(c[1],0)==rank]
        for key,method,payload in candidates:
            rec=json.loads(payload);selected.append(dict(record=rec,pids=[index.unique_full(n) for n in rec['authors']],evidence={'reliable':True,'method':method,'strong':method in ('ARXIV_ID','DOI')}))
        # Independent local merged-name repair can succeed even where global sequence has other defects.
        expanded,sa,sv=split_merged(g,selected,index);splitaudit+=sa
        split_map={int(r['original_author_position']):r for r in sv}
        splitverified+=sv;originals={int(r['author_position']):r for r in g}
        for r in expanded:
            parent=originals[int(r['original_author_position'])]
            r['source_positions_json']=parent['source_positions_json'];r['source_occurrences_json']=parent['source_occurrences_json']
            if not r['split_verified']:
                r['occurrence_id']=parent['occurrence_id']
                if not sv:r['author_position']=parent['author_position']
            else:
                r.update(identity_status='VERIFIED',dblp_person_id=r['resolved_pid'],dblp_author_matched_to_paper=True,dblp_identity_method='EVIDENCE_SUPPORTED_MERGED_NAME_SPLIT',dblp_identity_confidence='HIGH',evidence_json=json.dumps(r['resolution_evidence']))
        proposals={}
        for r in expanded:
            if r['identity_status']=='VERIFIED':continue
            options=[]
            for c in selected:
                names=c['record']['authors'];ids=c['pids']
                for j,(name,pid) in enumerate(zip(names,ids)):
                    if not pid:continue
                    relation=compatible(r['repaired_name'],name)
                    if not relation and index.unique_full(r['repaired_name'])==pid:relation='CANONICAL_PROFILE_ALIAS'
                    if not relation:continue
                    if sum(bool(compatible(x['repaired_name'],name)) for x in expanded)!=1 or sum(bool(compatible(r['repaired_name'],n)) for n in names)!=1:continue
                    options.append(dict(pid=pid,name=name,key=c['record']['key'],method=c['evidence']['method'],relation=relation))
            # Restore bounded spelling repair supported by two nearby verified anchors.
            if not options:
                for c in selected:
                    names=c['record']['authors'];ids=c['pids'];j=int(r['author_position'])-1
                    if not c['evidence']['strong'] or len(names)!=len(expanded) or not ids[j]:continue
                    if c['record']['title_key']!=p.get('_normalized_title'):continue
                    anchors=[k for k,x in enumerate(expanded) if k!=j and abs(k-j)<=2 and x['identity_status']=='VERIFIED' and x['dblp_person_id']==ids[k] and compatible(x['repaired_name'],names[k])]
                    key=normalize_name(base_name(r['repaired_name']))
                    near=[k for k,n in enumerate(names) if edit_distance(key,normalize_name(base_name(n)),2)<=2]
                    if len(anchors)>=2 and len(key)>=6 and near==[j]:options.append(dict(pid=ids[j],name=names[j],key=c['record']['key'],method=c['evidence']['method'],relation='BOUNDED_SPELLING_WITH_ORDERED_ANCHORS',anchors=anchors))
            unique={o['pid'] for o in options};proposals[r['occurrence_id']]=(unique,options)
        claimed=Counter(next(iter(ids)) for ids,options in proposals.values() if len(ids)==1)
        used={r['dblp_person_id'] for r in expanded if r['identity_status']=='VERIFIED'}
        for r in expanded:
            ids,options=proposals.get(r['occurrence_id'],(set(),[]));accepted=len(ids)==1 and claimed[next(iter(ids))]==1 and next(iter(ids)) not in used
            if accepted:
                pid=next(iter(ids));r.update(identity_status='VERIFIED',dblp_person_id=pid,dblp_author_matched_to_paper=True,dblp_identity_method='TARGET_PAPER_UNIQUE_AUTHOR',dblp_identity_confidence='HIGH',repaired_name=next(o['name'] for o in options if o['pid']==pid),evidence_json=json.dumps({'rule':'PAPER_CORROBORATED_PID','target_options':options,'source_pid':r.get('source_pid','')}))
                used.add(pid)
            if options:audit.append(dict(arxiv_id=aid,author_position=r['author_position'],occurrence_id=r['occurrence_id'],original_name=r['original_name'],accepted=accepted,source_pid=r.get('source_pid',''),final_pid=r['dblp_person_id'],options_json=json.dumps(options)))
            out.append(r)
    write_csv(cfg,'matching/target_identity_recovery_audit.csv',audit,['arxiv_id','author_position','occurrence_id','original_name','accepted','source_pid','final_pid','options_json'])
    cols=['arxiv_id','original_author_position','original_name','split_names_json','split_pids_json','record_key','corrected_count','aligned_anchor_count','evidence_json','accepted']
    write_csv(cfg,'diagnosis/merged_author_split_candidates.csv',splitaudit,cols);write_csv(cfg,'matching/merged_author_split_verified.csv',splitverified,cols)
    return out
