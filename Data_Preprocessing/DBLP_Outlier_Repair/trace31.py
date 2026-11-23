"""Exact-key display names, reason codes and one auditable row per historical record."""
import json
from collections import Counter
from common import integer
from vendor.rules import normalize_title

def text(value):
    return value.strip() if isinstance(value,str) else ''

def slot(row,position,pid):
    """Only use an ordered roster in the same paper. Reject explicit mismatched keys."""
    if row is None:return '',False
    for key in ('authors_v1_json','authors_json','author_roster_json'):
        if not row.get(key):continue
        try:
            roster=json.loads(row[key])
            count=row.get('author_count_v1',row.get('expected_author_count',len(roster)))
            if not isinstance(roster,list) or len(roster)!=integer(count) or not 1<=position<=len(roster):return '',True
            person=roster[position-1]
            if isinstance(person,str):return text(person),False
            if not isinstance(person,dict):return '',True
            if 'author_position' in person and integer(person['author_position'])!=position:return '',True
            rid=person.get('dblp_person_id') or person.get('pid')
            if rid and rid!=pid:return '',True
            return text(person.get('name') or person.get('author_name')),False
        except (ValueError,TypeError,IndexError):return '',True
    return '',False

def resolve(a,paper,team,snap=None):
    pid=a['dblp_person_id'];pos=integer(a['author_position'])
    if paper.get('paper_id')!=a['arxiv_id']:raise ValueError('Paper key mismatch')
    if team and team.get('arxiv_id',a['arxiv_id'])!=a['arxiv_id']:raise ValueError('Team key mismatch')
    master,bad_master=slot(paper,pos,pid);team_name,bad_team=slot(team,pos,pid)
    if team is not None and str(paper.get('author_count_v1',''))!=str(team.get('expected_author_count','')):
        master,bad_master='',True
    original=text(a.get('original_name'));repaired=text(a.get('repaired_name'))
    values={'recovery_original':original,'recovery_repaired':repaired,'master_position':master,'team_position':team_name}
    flags=set()
    if bad_master or bad_team:flags.add('ROSTER_KEY_CONFLICT')
    if not pid:flags.add('MISSING_PID')
    # Repaired recovery names may legitimately include DBLP disambiguation suffixes.
    if master and original and original.casefold()!=master.casefold():flags.add('SOURCE_NAME_CONFLICT')
    if master and team_name and team_name.casefold()!=master.casefold():flags.add('SOURCE_NAME_CONFLICT')
    aliases=[]
    if snap is not None and pid and snap.canonical(pid):
        aliases=[n for n, in snap.db.execute('SELECT raw FROM aliases WHERE pid=? ORDER BY raw',(snap.canonical(pid),))]
    values['profile_pid_aliases']=aliases
    choices=[('recovery_repaired',repaired),('recovery_original',original),('master_position',master),('team_position',team_name)]
    if len(aliases)==1:choices.append(('profile_exact_pid',aliases[0]))
    source,name=next(((k,v) for k,v in choices if v),('missing',''))
    if not name:flags.add('NAME_MISSING')
    if source=='profile_exact_pid':flags.add('PROFILE_ONLY_NAME')
    return name,source,sorted(flags),values

def focal_conflicts(a,paper,group,snap):
    from vendor.names import compatible
    keys=set(json.loads(a.get('focal_record_keys_json') or '[]'))
    flags=set()
    for r in snap.target_records(paper['paper_id']):
        if r['key'] not in keys:continue
        if normalize_title(r['title'])!=normalize_title(paper['title_v1']):flags.add('FOCAL_TITLE_DIFFERENCE')
        names=[x['audit_name'] for x in group]
        if len(names)!=len(r['authors']) or not all(compatible(n,s) for n,s in zip(names,r['authors'])):
            flags.add('FOCAL_ROSTER_CONFLICT')
    return flags

def priority(signals):
    if any('CONFLICT' in s or 'ORCID' in s or 'TITLE_DIFFERENCE' in s for s in signals):return 0
    if any('HISTORY' in s or 'EARLIEST' in s or 'REVIEWED' in s for s in signals):return 1
    return 2 if signals else 3

def flags_from(status):
    s=str(status)
    flags=set()
    if 'DEFERRED' in s:flags.add('BUDGET_DEFERRED')
    if 'TITLE_CONFLICT' in s:flags.add('TITLE_CONFLICT')
    if 'ROSTER' in s and ('CONFLICT' in s or 'AMBIGUOUS' in s):flags.add('ROSTER_CONFLICT')
    if 'ORCID' in s and ('CONFLICT' in s or 'FOREIGN' in s):flags.add('ORCID_CONFLICT')
    if 'CONFLICT' in s or 'AMBIGUOUS_NAME_ALIGNMENT' in s:flags.add('EVIDENCE_CONFLICT')
    if s.startswith(('METADATA_HTTP','METADATA_NETWORK')):flags.add('EXTERNAL_ERROR')
    return flags

def target_flags(target):
    flags=set()
    if not target.get('sources'):flags.add('NO_TARGET_DOI')
    if not target.get('orcids'):flags.add('NO_IDENTITY_ANCHOR')
    for s in target.get('sources',[]):flags.update(flags_from(s.get('status')))
    for c in target.get('conflicts',[]):flags.update(flags_from(c))
    return flags

def record_meta(r):
    return dict(history_pid=r['history_pid'],record_position=int(r['position'])+1,
                dois_json=json.dumps(r['dois']),record_title=r['title'])

def light_rows(a,works):
    for w in works:
        for r in w['records']:
            yield dict(paper_id=a['arxiv_id'],occurrence_id=a['occurrence_id'],pid=r['history_pid'],work_id=w['work'],
                       record_key=r['key'],publication_year=w['year'],target_orcid='',record_state='OFFLINE_SCREENED',
                       work_decision='RETAIN_UNRESOLVED',evidence_json='{}',**record_meta(r))

def trace(a,target,result,rows):
    """Aggregate independent flags. Missing publisher evidence remains explicitly unresolved."""
    author_flags=set(a['name_flags'])|target_flags(target)
    if result.get('target_orcid'):author_flags.discard('NO_IDENTITY_ANCHOR')
    reasons=json.loads(result.get('reasons_json','[]'))
    for v in reasons:author_flags.update(flags_from(v))
    work_ids=set();counts=Counter()
    for row in rows:
        work_ids.add(row['work_id'])
        flags=set(a['name_flags'])|target_flags(target)
        if not json.loads(row['dois_json']):flags.add('NO_WORK_DOI')
        ev=json.loads(row['evidence_json'])
        for reason in ev.get('risks',[]):flags.update(flags_from(reason))
        for item in ev.get('evidence',[]):flags.update(flags_from(item.get('status','')))
        if row['record_state']=='CONFLICT':flags.add('EVIDENCE_CONFLICT')
        counts.update(flags);author_flags.update(flags)
        row.update(author_position=a['author_position'],author_name=a['audit_name'],audit_flags_json=json.dumps(sorted(flags)))
    if len(work_ids)!=result['works_examined']:raise ValueError(f"Work coverage mismatch for {a['occurrence_id']}")
    result.update(author_name=a['audit_name'],author_position=a['author_position'],name_source=a['name_source'],
                  name_evidence_json=json.dumps(a['name_evidence'],ensure_ascii=False),audit_flags_json=json.dumps(sorted(author_flags)),
                  historical_flags_count_json=json.dumps(counts),distinct_works_logged=len(work_ids),record_rows_logged=len(rows))

def reconcile_versions(a, original_works, chosen_works, rows):
    """Log old-PID versions sharing a work with the replacement PID, without changing its decision."""
    def key(row):
        return row['work_id'],row['record_key'],row['history_pid'],int(row['record_position'])
    seen={key(row) for row in rows}
    if len(seen)!=len(rows):raise ValueError('Duplicate work-record ledger key')
    expected=set()
    for works in (chosen_works,original_works):
        for w in works:
            for r in w['records']:
                expected.add((w['work'],r['key'],r['history_pid'],int(r['position'])+1))
    shared={w['work'] for w in original_works}&{w['work'] for w in chosen_works}
    if seen!=expected and original_works is not chosen_works:
        for row in light_rows(a,original_works):
            if key(row) not in seen and row['work_id'] in shared:
                row.update(record_state='ORIGINAL_PID_VERSION',work_decision='SHARED_DEDUPLICATED_WORK')
                rows.append(row);seen.add(key(row))
    if seen!=expected:raise ValueError('Missing or extraneous historical record versions')

def previous_hints(directory, binding, source_hash, authors):
    """Carry forward investigation leads, never a previous adjudication."""
    from pathlib import Path
    from common import read
    if not directory:return {}
    root=Path(directory)
    if not (root/'summary.json').exists():return {}
    prior=json.loads((root/'summary.json').read_text())
    if prior.get('version')!='3.0.0' or prior.get('input_sha256')!=source_hash:return {}
    previous=prior.get('snapshot_binding',{})
    if any(previous.get(k)!=binding[k] for k in ('xml_sha256','dtd_sha256','author_subset_sha256','paper_subset_sha256')):return {}
    keys={r.occurrence_id:(r.arxiv_id,r.dblp_person_id) for r in authors.itertuples()}
    hints={}
    def add(oid,signals,record_key=''):
        h=hints.setdefault(oid,{'signals':set(),'keys':set()})
        h['signals'].update(signals)
        if record_key:h['keys'].add(record_key)
    for r in read(root/'author_verdicts.csv').itertuples():
        if keys.get(r.occurrence_id)!=(r.paper_id,r.old_pid):continue
        signals=set(json.loads(r.offline_signals_json or '[]'))|set(json.loads(r.reasons_json or '[]'))
        t=json.loads(r.target_evidence_json or '{}')
        signals.update(t.get('conflicts',[]))
        flags={'PRIOR_'+s for s in signals if any(x in s for x in ('CONFLICT','MISMATCH','EARLIEST','GAP','ORCID','REVIEWED_RECORD'))}
        if flags:add(r.occurrence_id,flags)
    for chunk in read(root/'work_evidence.csv',chunksize=20000):
        for r in chunk.itertuples():
            if keys.get(r.occurrence_id)!=(r.paper_id,r.pid):continue
            ev=json.loads(r.evidence_json or '{}')
            signals=set(ev.get('risks',[]))|{str(x.get('status','')) for x in ev.get('evidence',[])}
            flags={'PRIOR_'+s for s in signals if 'CONFLICT' in s or 'FOREIGN_ORCID' in s or 'AMBIGUOUS' in s}
            if r.record_state=='WRONG_PERSON_CONFIRMED':flags.add('PRIOR_REVIEWED_RECORD')
            if flags:add(r.occurrence_id,flags,r.record_key)
    return {oid:{k:sorted(v) for k,v in h.items()} for oid,h in hints.items()}
