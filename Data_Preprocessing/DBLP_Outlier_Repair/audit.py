"""Occurrence-level identity and work-level contamination audit for EVERY screened paper."""
from trace31 import record_meta
import json,re
from collections import Counter
from difflib import SequenceMatcher
from snapshot import orcid
from external import align
from vendor.names import compatible
from vendor.rules import normalize_title,normalize_doi
from common import PUB,AGE,FIELDS,number,same,integer,aggregate


def stats(works,year):return (len(works),year-min(w['year'] for w in works) if works else 0)

def target_evidence(paper,team,snapshot,client):
    names=[a['repaired_name'] or a['original_name'] for a in team]
    result=[{'orcids':set(),'authenticated':set(),'local_orcids':set(),'roster_supported':False,'sources':[],'conflicts':[]} for _ in names]
    dois=set();d=normalize_doi(paper.get('doi_v1',''))
    if d:dois.add(d)
    for rec in snapshot.target_records(paper['paper_id']):
        if len(rec['authors'])!=len(names) or not all(compatible(a,b) for a,b in zip(names,rec['authors'])):continue
        if SequenceMatcher(None,normalize_title(rec['title']),normalize_title(paper['title_v1'])).ratio()<.92:continue
        if any(sum(bool(compatible(a,b)) for b in rec['authors'])!=1 for a in names):continue
        # A whole-roster correspondence is context, not independent proof of person identity.
        dois.update(rec['dois'])
        for i,sig in enumerate(rec['signatures']):
            result[i]['roster_supported']=True
            if sig['orcid']:result[i]['local_orcids'].add(sig['orcid'])
    for doi in sorted(dois):
        metadata=client.work(doi);people,status=align(metadata,paper['title_v1'],names)
        for i,r in enumerate(result):
            r['sources'].append({'doi':doi,'status':status,'source':metadata.get('source',''),'aligned_author':people[i] if people else None})
            if people:
                r['roster_supported']=True;o=people[i]['orcid']
                if o:r['orcids'].add(o)
                if o and people[i]['authenticated']:r['authenticated'].add(o)
            elif 'CONFLICT' in status:r['conflicts'].append(status)
    for r in result:
        if len(r['orcids'])>1:r['conflicts'].append('CONFLICTING_TARGET_ORCIDS')
        if r['orcids'] and r['local_orcids'] and not r['orcids']&r['local_orcids']:r['conflicts'].append('TARGET_PUBLISHER_VS_XML_ORCID_CONFLICT')
    return result


def identity_decision(author,target,snapshot):
    old=snapshot.canonical(author['dblp_person_id']);profile=snapshot.profile(old)
    anchor=next(iter(target['orcids'])) if len(target['orcids'])==1 else ''
    result={'old_pid':old,'pid':old,'target_orcid':anchor,'identity_status':'NO_INDEPENDENT_IDENTITY_ANCHOR','replaced':False,'reasons':list(target['conflicts'])}
    if target['conflicts'] or not old:return result
    if anchor:
        candidates=snapshot.oids.get(anchor,set())
        if anchor in profile['orcids'] and candidates=={old}:
            result['identity_status']='TARGET_PUBLISHER_AND_VERIFIED_PROFILE_AGREE'
        elif len(candidates)==1 and anchor in target['authenticated'] and profile['orcids'] and not profile['orcids']&{anchor}:
            new=next(iter(candidates));aliases=[x[0] for x in snapshot.db.execute('SELECT raw FROM aliases WHERE pid=?',(new,))]
            if any(compatible(author['repaired_name'] or author['original_name'],a) for a in aliases) and snapshot.candidate_usable(new):
                result.update(pid=new,replaced=True,identity_status='CORRECTED_PID_BY_AUTHENTICATED_TARGET_ORCID')
            else:result['reasons'].append('CORRECT_PID_HISTORY_OR_NAME_NOT_CERTIFIED')
        else:result['reasons'].append('TARGET_ORCID_PROFILE_CONFLICT_OR_AMBIGUITY')
    return result


def record_evidence(rec,pid,anchor,snapshot,client,rules,bio=None):
    pos=rec['position'];evidence=[];positive=False;negative=False;risks=[];attempted=0;successful=0;foreign=set()
    local_oid=rec['signatures'][pos]['orcid']
    if bio and bio['status']=='BIRTH_YEAR_SUPPORTED' and rec['year'] is not None and rec['year']<=bio['birth_year']:
        negative=True;evidence.append({'rule':'PUBLICATION_AT_OR_BEFORE_VERIFIED_BIRTH_YEAR','kind':'BIOGRAPHY_YEAR_CONTRADICTION','facts':bio['facts']})
    for rule in rules:
        if rule['pid']==pid and rule['record_key']==rec['key'] and rule['year']==rec['year'] and rec['authors'][pos]==rule['author_alias']:
            if len(rule.get('evidence_urls',[]))<2 or not rule.get('reviewer'):raise ValueError('Incomplete reviewed work rule')
            negative=True;evidence.append({'rule':rule['rule_id'],'sources':rule['evidence_urls'],'kind':'REVIEWED_RECORD_EXCLUSION'})
    for doi in sorted(set(rec['dois'])):
        attempted+=1;meta=client.work(doi);people,status=align(meta,rec['title'],rec['authors'])
        entry={'doi':doi,'source':meta.get('source',''),'status':status}
        if people:
            successful+=1;person=people[pos];o=person['orcid'];entry.update(person)
            if o and not anchor and snapshot.profile(pid)['orcids'] and o not in snapshot.profile(pid)['orcids']:risks.append('UNANCHORED_PROFILE_PUBLISHER_ORCID_CONFLICT')
            if anchor and o==anchor:positive=True
            if anchor and o and o!=anchor:
                foreign.add(o);otherpids=snapshot.oids.get(o,set())
                # Different ORCID alone is a warning. Automatic removal additionally needs
                # authenticated publisher assertion + distinct, uniquely verified DBLP profile.
                if person['authenticated'] and len(otherpids)==1 and pid not in otherpids and anchor in snapshot.profile(pid)['orcids'] and any(compatible(rec['authors'][pos],v[0]) for v in snapshot.db.execute('SELECT raw FROM aliases WHERE pid=?',(next(iter(otherpids)),))):
                    negative=True;entry['rule']='AUTHENTICATED_FOREIGN_ORCID_AND_DISTINCT_VERIFIED_PROFILE'
                else:risks.append('UNCONFIRMED_ORCID_CONFLICT')
            if local_oid and o and local_oid!=o:risks.append('XML_PUBLISHER_SIGNATURE_CONFLICT')
        elif 'CONFLICT' in status:risks.append(status)
        evidence.append(entry)
    if anchor and local_oid and local_oid!=anchor:risks.append('LOCAL_SIGNATURE_FOREIGN_ORCID')
    if (positive and negative) or 'XML_PUBLISHER_SIGNATURE_CONFLICT' in risks:state='CONFLICT';risks.append('CONTRADICTORY_SIGNATURE_EVIDENCE')
    elif negative:state='WRONG_PERSON_CONFIRMED'
    elif positive:state='SAME_PERSON_SUPPORTED'
    else:state='UNRESOLVED'
    return {'state':state,'evidence':evidence,'risks':sorted(set(risks)),'doi_attempted':attempted,'doi_successful':successful,'foreign_orcids':sorted(foreign),'local_orcid':local_oid,'publisher_orcids':sorted({x.get('orcid') for x in evidence if x.get('orcid')})}


def work_decision(records):
    if not records:return 'UNRESOLVED'
    states={r['result']['state'] for r in records}
    if 'CONFLICT' in states or {'WRONG_PERSON_CONFIRMED','SAME_PERSON_SUPPORTED'}<=states:return 'CONFLICT'
    negatives=[r for r in records if r['result']['state']=='WRONG_PERSON_CONFIRMED']
    if negatives:
        identifiers=set()
        for r in negatives:identifiers.update('doi:'+d for d in r['record']['dois']);identifiers.update('arxiv:'+a for a in r['record']['arxiv_ids'])
        # Do not remove an unverified fallback-only version just because another version is wrong.
        for r in records:
            if r['result']['state']=='WRONG_PERSON_CONFIRMED':continue
            ids={'doi:'+d for d in r['record']['dois']}|{'arxiv:'+a for a in r['record']['arxiv_ids']}
            if not ids&identifiers:return 'UNRESOLVED_MIXED_WORK_COMPONENT'
        return 'REMOVE_WRONG_PERSON_WORK'
    if states=={'SAME_PERSON_SUPPORTED'}:return 'RETAIN_IDENTITY_SUPPORTED'
    return 'RETAIN_UNRESOLVED'


def structural_signals(works,year,target_names,self_aliases):
    if not works:return []
    years=sorted({w['year'] for w in works});flags=[]
    if year-years[0]>55:flags.append('ACADEMIC_AGE_OVER_55_REVIEW_ONLY')
    if any(b-a>=15 for a,b in zip(years,years[1:])):flags.append('HISTORY_GAP_AT_LEAST_15_YEARS_REVIEW_ONLY')
    counts=Counter(w['year'] for w in works)
    if max(counts.values())>150:flags.append('OVER_150_WORKS_IN_ONE_YEAR_REVIEW_ONLY')
    # Coauthor connectivity is a diagnostic, never grounds for removing a disconnected cluster.
    peers={n for n in target_names if n not in self_aliases};connected=0
    for w in works:
        if any(any(compatible(a,p) for a in r['authors'] if a not in self_aliases for p in peers) for r in w['records']):connected+=1
    if peers and len(works)>=5 and connected==0:flags.append('NO_TARGET_COAUTHOR_IN_PRIOR_HISTORY_REVIEW_ONLY')
    return flags


def audit_author(paper,author,target,snapshot,client,rules,biography=None):
    aid=paper['paper_id'];year=integer(paper['submission_year']);oid=author['occurrence_id'];decision=identity_decision(author,target,snapshot)
    old=decision['old_pid'];chosen=decision['pid'];oldworks=snapshot.history(old,aid,year) if old else []
    original=stats(oldworks,year);baseline_ok=bool(old and integer(author['t0_year'])==year and author['identity_status']=='VERIFIED' and author['history_status'] in ('COMPLETE_OBSERVED_DBLP_HISTORY','ZERO_PRIOR_WORKS_IN_DBLP_SNAPSHOT') and same(original[0],author[PUB]) and same(original[1],author[AGE]))
    works=list(oldworks) if chosen==old else list(snapshot.history(chosen,aid,year)) if chosen else []
    bio=biography.check(snapshot.profile(chosen),author['repaired_name'] or author['original_name'],paper,decision['target_orcid']) if biography and chosen else {'status':'NOT_CHECKED','birth_year':None,'facts':[],'pages_checked':[]}
    if bio.get('birth_year') is not None and bio['birth_year']>=year:bio['status']='BIOGRAPHY_CONTRADICTS_TARGET_YEAR'
    if decision['replaced']:
        current={w['work'] for w in works}
        works += [dict(w,transfer_candidate=True) for w in oldworks if w['work'] not in current]
    aliases={r[0] for r in snapshot.db.execute('SELECT raw FROM aliases WHERE pid=?',(chosen,))}
    signals=structural_signals(works,year,[a['name'] if isinstance(a,dict) else a for a in json.loads(paper['authors_v1_json'])],aliases)
    rows=[];kept=[];removed=[];supported=0;conflicts=0;attempted=successful=0;observed_orcids=set();unresolved_transfer=0
    if len(works)>=500:print(f'Auditing {oid}: {len(works):,} historical works (publisher requests are cached)',flush=True)
    for work in works:
        results=[]
        for rec in work['records']:
            ev=record_evidence(rec,chosen,decision['target_orcid'],snapshot,client,rules,bio)
            observed_orcids.update(ev['publisher_orcids'])
            attempted+=ev['doi_attempted'];successful+=ev['doi_successful'];results.append({'record':rec,'result':ev})
        state=work_decision(results)
        if work.get('transfer_candidate'):
            strong=any(e.get('authenticated') and e.get('orcid')==decision['target_orcid'] for r in results for e in r['result']['evidence'])
            if state=='RETAIN_IDENTITY_SUPPORTED' and strong:
                kept.append(work);supported+=1;state='TRANSFER_VERIFIED_WORK_TO_CORRECT_PID'
            elif state=='REMOVE_WRONG_PERSON_WORK':removed.append(work)
            else:
                unresolved_transfer+=1
                if state!='CONFLICT':state='OLD_PID_WORK_NOT_TRANSFERRED_WITHOUT_PROOF'
        else:
            if state=='REMOVE_WRONG_PERSON_WORK':removed.append(work)
            else:kept.append(work)
            supported+=int(state=='RETAIN_IDENTITY_SUPPORTED')
        conflicts+=int(state=='CONFLICT')
        for rec in results:
            r=rec['record'];e=rec['result']
            rows.append({'paper_id':aid,'occurrence_id':oid,'pid':chosen,'work_id':work['work'],'record_key':r['key'],'publication_year':work['year'],'record_title':r['title'],'target_orcid':decision['target_orcid'],'record_state':e['state'],'work_decision':state,'evidence_json':json.dumps(e,ensure_ascii=False),**record_meta(r)})
    # No observed metadata is never interpreted as a positive normal verdict.
    reasons=list(decision['reasons'])+signals
    if bio['status'] in ('CONFLICTING_BIRTH_YEARS','BIOGRAPHY_CONTRADICTS_TARGET_YEAR'):reasons.append(bio['status'])
    if len(observed_orcids)>1 and not removed:reasons.append('MULTIPLE_PUBLISHER_ORCIDS_IN_ONE_PROFILE_REQUIRES_DISAMBIGUATION')
    if unresolved_transfer:reasons.append('UNCERTAIN_OLD_PID_WORKS_NOT_TRANSFERRED')
    if not baseline_ok:reasons.append('ORIGINAL_SNAPSHOT_COUNTS_NOT_REPRODUCED')
    if chosen and not snapshot.candidate_usable(chosen):reasons.append('AMBIGUOUS_ALIAS_OR_UNKNOWN_YEAR')
    if conflicts:reasons.append('CONTRADICTORY_WORK_EVIDENCE')
    correction=bool(removed or decision['replaced'])
    can_repair=baseline_ok and chosen and snapshot.candidate_usable(chosen) and not decision['reasons'] and not conflicts and bio['status'] not in ('CONFLICTING_BIRTH_YEARS','BIOGRAPHY_CONTRADICTS_TARGET_YEAR')
    normal=(not correction and baseline_ok and not [r for r in reasons if r not in signals] and bool(decision['target_orcid']) and decision['identity_status']=='TARGET_PUBLISHER_AND_VERIFIED_PROFILE_AGREE' and supported==len(works))
    if correction and can_repair:status='CONFIRMED_ERROR_RECALCULATED'
    elif correction:status='CONFIRMED_ERROR_REPAIR_BLOCKED'
    elif normal:status='SUPPORTED_NORMAL_OUTLIER'
    else:status='REVIEW_REQUIRED'
    summary={'paper_id':aid,'occurrence_id':oid,'author_name':author['repaired_name'] or author['original_name'],'old_pid':old,'new_pid':chosen,'target_orcid':decision['target_orcid'],'identity_status':decision['identity_status'],'status':status,'baseline_reproduced':baseline_ok,'old_count':author[PUB],'old_age':author[AGE],'new_count':stats(kept,year)[0] if correction and can_repair else author[PUB],'new_age':stats(kept,year)[1] if correction and can_repair else author[AGE],'publisher_orcids_json':json.dumps(sorted(observed_orcids)),'works_examined':len(works),'works_identity_supported':supported,'works_removed':len(removed) if can_repair else 0,'unresolved_transfer_works':unresolved_transfer,'unresolved_retained_works':len(kept)-supported,'doi_requests_logical':attempted,'doi_aligned':successful,'reasons_json':json.dumps(reasons),'biography_evidence_json':json.dumps(bio,ensure_ascii=False),'target_evidence_json':json.dumps(target,default=lambda v:sorted(v),ensure_ascii=False),'removed_work_ids_json':json.dumps([w['work'] for w in removed])}
    return summary,rows
