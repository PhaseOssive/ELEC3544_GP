"""All-outlier audit and four-column repair; Colab / local entry point."""
import argparse,csv,hashlib,json,os,shutil,sqlite3,time
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
from common import read,sha,screen,FIELDS,PUB,AGE,FIRST,unique,integer,number,same,aggregate,write,dump
from fast import FastSnapshot, TargetedPublisher, triage, reusable_snapshot, offline_summary, history_csv_check
from vendor.rules import normalize_doi, normalize_name
from vendor.names import base_name, compatible
from collections import defaultdict
from legacy_cache import Legacy
from progress import Progress,digest
from external import Crossref
from biography import Biography
from audit import target_evidence,audit_author

from compare_runs import comparison
from trace31 import resolve, priority, focal_conflicts, light_rows, trace, reconcile_versions, previous_hints

VERSION='3.1.0'
AUTHOR_COLUMNS=['arxiv_id','author_position','occurrence_id','original_name','repaired_name','dblp_person_id','identity_status','history_status','t0_year',PUB,AGE,FIRST,'focal_record_keys_json']
WORK_COLUMNS=['paper_id','occurrence_id','pid','work_id','record_key','publication_year','record_title','target_orcid','record_state','work_decision','evidence_json','history_pid','record_position','dois_json','author_position','author_name','audit_flags_json']

def copy_atomic(source,destination):
    source=Path(source);destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    if source.resolve()==destination.resolve():return
    temp=destination.with_suffix(destination.suffix+'.tmp');shutil.copy2(source,temp);os.replace(temp,destination)

def backup_db(db,destination):
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    temp=destination.with_suffix('.tmp');dest=sqlite3.connect(temp)
    try:db.commit();db.backup(dest)
    finally:dest.close()
    os.replace(temp,destination)

def load_authors(cfg,ids):
    path=Path(cfg['recovery_root'])/'features/treated_author_features.csv';parts=[];total=0
    header=read(path,nrows=0).columns
    missing=set(AUTHOR_COLUMNS)-set(header)
    if missing:raise ValueError('Author export missing columns: '+str(missing))
    for chunk in read(path,usecols=AUTHOR_COLUMNS,chunksize=cfg['author_chunk_size']):
        total+=len(chunk);parts.append(chunk[chunk.arxiv_id.isin(ids)])
        if total%50000<len(chunk):print(f'Author rows read: {total:,}',flush=True)
    authors=pd.concat(parts,ignore_index=True);unique(authors,'occurrence_id','author occurrence')
    if authors.duplicated(['arxiv_id','author_position']).any():raise ValueError('Duplicate author position')
    summary=json.loads((Path(cfg['recovery_root'])/'report/final_recovery_summary.json').read_text())
    if summary.get('full_xml_scan_complete') is not True:raise ValueError('Previous snapshot recovery incomplete')
    if summary.get('final_author_occurrences')!=total:raise ValueError('Author file row count does not match recovery summary')
    return authors

def _run(cfg,refresh_reviews=False,audit_only=False,client_factory=Crossref,progress=None):
    start=time.time();source=Path(cfg['input_csv']);master=read(source);unique(master,'paper_id','paper')
    required=set(FIELDS+['paper_id','title_v1','submission_year','authors_v1_json','author_count_v1'])
    if not required<=set(master):raise ValueError('Missing master columns: '+str(required-set(master)))
    root=Path(cfg['output_root']);root.mkdir(parents=True,exist_ok=True)
    out=root/datetime.now(timezone.utc).strftime('run_%Y%m%dT%H%M%S_%fZ');out.mkdir()
    progress.path=out/'progress.json'
    progress.begin('Screen master CSV',None)
    flags,distribution=screen(master);ids=set(flags.paper_id)
    if cfg.get('screen_previous_input_csv'):
        previous=read(cfg['screen_previous_input_csv']);unique(previous,'paper_id','previous paper');oldflags,_=screen(previous)
        ids.update(set(oldflags.paper_id)&set(master.paper_id))
    if cfg.get('candidate_csv'):
        supplied=set(read(cfg['candidate_csv'],usecols=['paper_id']).paper_id)
        unknown=supplied-set(master.paper_id)
        if unknown:raise ValueError(f'{len(unknown)} supplied candidate IDs missing from master')
        ids.update(supplied)
    write(flags,out/'outlier_flags.csv');selected=master[master.paper_id.isin(ids)]
    print(f'Audit scope: {len(ids):,} papers, {len(flags):,} field flags. No paper limit / top-N sampling.',flush=True)
    if not ids:
        write(master,out/'papers_features_dblp_repaired.csv')
        write(pd.DataFrame(columns=['paper_id','status']),out/'paper_verdicts.csv')
        write(pd.DataFrame(columns=['paper_id','field','old','new','reason']),out/'changes.csv')
        dump(dict(version=VERSION,mode='AUDIT_ONLY' if audit_only else 'REPAIR',all_outlier_papers_visited=True,outlier_papers=0,papers_with_verdict=0,verdict_counts={},changed_papers=0,changed_cells=0,rows=len(master),columns=len(master.columns),non_dblp_unchanged=True,input=str(source.resolve()),input_sha256=sha(source),output_sha256=sha(out/'papers_features_dblp_repaired.csv')),out/'summary.json')
        print('No outliers under configured screening; unchanged copy:',out)
        return out
    rules=json.loads(Path(cfg['reviewed_rules']).read_text())
    progress.begin('Read recovery author and team CSVs',None)
    authors=load_authors(cfg,ids)
    teams=read(Path(cfg['recovery_root'])/'features/treated_team_features.csv');unique(teams,'arxiv_id','team');teams=teams.set_index('arxiv_id')
    history_check=history_csv_check(cfg,authors,progress,out/'recovery_history_ledger.csv')
    write(history_check,out/'history_csv_check.csv')
    history_conflicts=set(history_check.loc[history_check.status.eq('CONFLICT'),'occurrence_id'])
    before=sha(source)
    # Cache bindings include actual inputs, rules, and code. No cache from different data is trusted.
    print('Hashing source XML and DTD for reproducible cache binding...',flush=True)
    xml_hash=digest(cfg['xml_path'],progress,'Verify original XML/gz');dtd_hash=digest(cfg['dtd_path'],progress,'Verify DTD')
    recovery_summary=json.loads((Path(cfg['recovery_root'])/'report/final_recovery_summary.json').read_text())
    expected_xml_hash=recovery_summary.get('xml_snapshot',{}).get('sha256')
    if expected_xml_hash and xml_hash!=expected_xml_hash:raise ValueError('XML does not match the snapshot used by recovery; use the original XML')
    code_hash=hashlib.sha256(b''.join(p.read_bytes() for p in sorted(Path(__file__).parent.rglob('*.py')))).hexdigest()
    binding=dict(version=VERSION,xml_sha256=xml_hash,dtd_sha256=dtd_hash,author_subset_sha256=hashlib.sha256(authors.to_csv(index=False).encode()).hexdigest(),paper_subset_sha256=hashlib.sha256(selected.to_csv(index=False).encode()).hexdigest(),rules_sha256=sha(cfg['reviewed_rules']),code_sha256=code_hash,crossref_enabled=cfg['crossref_enabled'],biography_enabled=cfg.get('biography_enabled',True),biography_max_pages_per_author=cfg.get('biography_max_pages_per_author',4))
    binding['team_subset_sha256']=hashlib.sha256(teams.loc[teams.index.isin(ids)].to_csv().encode()).hexdigest()
    binding['history_csv_check_sha256']=sha(out/'history_csv_check.csv')
    binding['verification_limits']={k:cfg.get(k) for k in ('max_external_requests','external_budget_seconds','http_error_retry_hours')}
    previous=previous_hints(cfg.get('baseline_v3_run'),binding,before,authors)
    binding['priority_hints_sha256']=hashlib.sha256(json.dumps(previous,sort_keys=True).encode()).hexdigest()
    signature=json.dumps(binding,sort_keys=True);job=hashlib.sha256(signature.encode()).hexdigest()[:24]
    persistent=Path(cfg['cache_root'])/job;persistent.mkdir(parents=True,exist_ok=True)
    local=Path(cfg['local_work_dir'])/job;local.mkdir(parents=True,exist_ok=True)
    xml=Path(cfg['xml_path']);dtd=Path(cfg['dtd_path'])
    progress.begin('Locate compatible filtered index',None)
    localdb=local/'snapshot.sqlite'
    saved=persistent/'snapshot.sqlite'
    source_snapshot=None
    if not localdb.exists():
        source_snapshot=saved if saved.exists() else reusable_snapshot(cfg['cache_root'],binding)
        if source_snapshot:
            progress.begin('Stage verified filtered snapshot',None)
            copy_atomic(source_snapshot,localdb)
    snap=FastSnapshot(localdb)
    if source_snapshot and source_snapshot!=saved:
        # Only our private copy is rebound. XML, DTD, paper and author digests matched.
        snap.put('signature',signature)
        snap.db.commit()
    complete=snap.get('dedup_done')=='1'
    workload=float(pd.to_numeric(authors[PUB],errors='coerce').clip(lower=0).fillna(0).sum())
    # Scenario assumptions are deliberately explicit. The measured ETA supersedes them during audit.
    estimate=dict(papers=len(ids),author_occurrences=len(authors),pre_t0_work_occurrences_estimated=workload,
                  filtered_snapshot_complete=complete,assumed_work_occurrences_per_second=[1000,5000],
                  planning_minutes_excluding_index_build=[round(3+workload/5000/60,1),round(10+workload/1000/60+cfg.get('external_budget_seconds',600)/60,1)],
                  assumptions='3–10 min other I/O and export, 0 to configured Crossref seconds. Cold index scans/dedup and optional biography are additional.')
    dump(estimate,out/'runtime_estimate.json');print('Runtime estimate:',json.dumps(estimate,ensure_ascii=False),flush=True)
    if not complete and cfg.get('stage_xml_locally',True):
        progress.begin('Stage compressed XML',None)
        localxml=Path(cfg['local_work_dir'])/('source_'+xml_hash+('.xml.gz' if str(xml).endswith('.gz') else '.xml'))
        if not localxml.exists() or sha(localxml)!=xml_hash:copy_atomic(xml,localxml)
        if sha(localxml)!=xml_hash:raise ValueError('Staged XML digest mismatch')
        xml=localxml
    checkpoint=lambda:backup_db(snap.db,saved)
    legacy=Legacy(cfg,xml_hash,dtd_hash,progress)
    legacy.report.update(filtered_snapshot_already_complete=complete,imported_from=str(source_snapshot or ''))
    try:snap.build(xml,dtd,authors,selected,signature,checkpoint,legacy=legacy,progress=progress)
    except BaseException:
        checkpoint();snap.db.close();raise
    finally:
        legacy.close();dump(legacy.report,out/'cache_reuse_report.json')
    state=sqlite3.connect(persistent/'audit_state.sqlite')
    state.executescript('CREATE TABLE IF NOT EXISTS authors(oid TEXT PRIMARY KEY,payload TEXT); CREATE TABLE IF NOT EXISTS works(oid TEXT,seq INTEGER,payload TEXT,PRIMARY KEY(oid,seq));')
    # One attempt per call keeps the request cap applicable to actual HTTP attempts.
    client=TargetedPublisher(client_factory(Path(cfg['cache_root'])/'crossref_cache.sqlite',dict(cfg,http_retries=1)),cfg)
    reviewed={(r['pid'],r['record_key']) for r in rules}
    biography=Biography(client.db,cfg) if cfg.get('biography_enabled',False) else None
    groups={aid:g.sort_values('author_position',key=lambda x:pd.to_numeric(x)).to_dict('records') for aid,g in authors.groupby('arxiv_id',sort=False)}
    paper_map={p['paper_id']:p for p in selected.to_dict('records')}
    for aid,group in groups.items():
        team=teams.loc[aid].to_dict() if aid in teams.index else None
        for a in group:
            name,origin,flags,values=resolve(a,paper_map[aid],team,snap)
            a.update(audit_name=name,name_source=origin,name_flags=flags,name_evidence=values)
            a['repaired_name']=name
    name_pids=defaultdict(set)
    pid_names=defaultdict(set)
    for a in authors.itertuples(index=False):
        name=a.repaired_name or a.original_name
        if a.dblp_person_id:
            name_pids[normalize_name(base_name(name))].add(a.dblp_person_id)
            pid_names[a.dblp_person_id].add(name)
    # Build a global queue without network access so the request budget serves conflicts first.
    progress.begin('Rank offline conflicts',len(selected))
    queued=[]
    for ix,(aid,paper) in enumerate(paper_map.items(),1):
        group=groups.get(aid,[])
        cached=not refresh_reviews and all(state.execute('SELECT 1 FROM authors WHERE oid=?',(a['occurrence_id'],)).fetchone() for a in group)
        signals=set()
        if not cached:
            client.allowed=set()
            targets=target_evidence(paper,group,snap,client)
            for a,t in zip(group,targets):
                local,_,_=triage(a,t,snap,reviewed,client,False)
                signals.update(local);signals.update(t['conflicts']);signals.update(previous.get(a['occurrence_id'],{}).get('signals',[]))
                signals.update(focal_conflicts(a,paper,group,snap));signals.update(a['name_flags'])
        queued.append((priority(signals),ix,aid,sorted(signals)))
        progress.update(ix)
    queued.sort()
    write(pd.DataFrame(queued,columns=['priority','original_order','paper_id','signals']),out/'audit_queue.csv')
    paper_rows=[];changes=[];edited=master.copy();index={aid:i for i,aid in enumerate(master.paper_id)};visited=0
    progress.begin('Audit all outlier papers',len(selected),'Rolling paper-level estimate includes HTTP, cache hits and biographies; skewed author histories can change ETA substantially. Later export time excluded.')
    audit_started=time.monotonic();work_done=0
    try:
        for n,(_,_,queued_id,_) in enumerate(queued,1):
            paper=paper_map[queued_id]
            aid=paper['paper_id'];group=groups.get(aid,[]);summaries=[];block=[]
            try:
                expected=integer(teams.loc[aid,'expected_author_count'])
                if expected!=len(group) or sorted(integer(a['author_position']) for a in group)!=list(range(1,expected+1)):block.append('INCOMPLETE_AUTHOR_ROSTER')
                if expected!=integer(paper['author_count_v1']):block.append('MASTER_AUTHOR_COUNT_CONFLICT')
                if len({a['dblp_person_id'] for a in group})!=len(group):block.append('DUPLICATE_PID_IN_TEAM')
            except (KeyError,ValueError):block.append('MISSING_TEAM_OR_INVALID_ROSTER')
            for a in group:
                block.extend(x for x in a['name_flags'] if x in ('NAME_MISSING','SOURCE_NAME_CONFLICT','ROSTER_KEY_CONFLICT','PROFILE_ONLY_NAME','MISSING_PID'))
            if any(a['occurrence_id'] in history_conflicts for a in group):block.append('RECOVERY_HISTORY_CSV_CONFLICT')
            # Every occurrence still gets an audit even when the complete-team repair is blocked.
            cached_team = not refresh_reviews and all(
                state.execute('SELECT 1 FROM authors WHERE oid=?',(a['occurrence_id'],)).fetchone() for a in group)
            if cached_team:
                target=[{'orcids':set()} for _ in group]
                plans=[([],set(),[]) for _ in group]
            else:
                client.allowed=set()
                target=target_evidence(paper,group,snap,client)
                plans=[]
                for author,t in zip(group,target):
                    signals,dois,works=triage(author,t,snap,reviewed,client,'DUPLICATE_PID_IN_TEAM' in block)
                    hint=previous.get(author['occurrence_id'],{})
                    signals.extend(hint.get('signals',[]))
                    if hint.get('keys'):
                        flagged_keys=set(hint['keys'])
                        dois.update(d for w in works for r in w['records'] if r['key'] in flagged_keys for d in r['dois'])
                    local_focal=focal_conflicts(author,paper,group,snap)
                    signals=sorted(set(signals)|local_focal|set(author['name_flags']))
                    if local_focal or any(x.startswith('CACHED_') and ('TITLE' in x or 'ROSTER' in x or 'AMBIGUOUS' in x) for x in signals):
                        block.append('UNRESOLVED_SOURCE_CONFLICT')
                    name=author['repaired_name'] or author['original_name']
                    if len(name_pids[normalize_name(base_name(name))])>1:signals.append('NAME_ASSIGNED_TO_MULTIPLE_PIDS')
                    if any(not compatible(name,alias) for alias in pid_names[author['dblp_person_id']]):
                        signals.append('PID_ASSIGNED_TO_INCOMPATIBLE_NAMES')
                    plans.append((signals,dois,works))
                # Only the focal publisher record(s) for a signalled paper may acquire an identity anchor.
                if any(signals for signals,_,_ in plans):
                    client.allowed={normalize_doi(source['doi']) for t in target for source in t['sources']}
                    target=target_evidence(paper,group,snap,client)
            for a,t,(signals,dois,works) in zip(group,target,plans):
                if t['orcids'] and signals:
                    extra,new_dois,_=triage(a,t,snap,reviewed,client,'DUPLICATE_PID_IN_TEAM' in block)
                    signals=sorted(set(signals)|set(extra))
                    dois.update(new_dois)
                oid=a['occurrence_id']
                cached=state.execute('SELECT payload FROM authors WHERE oid=?',(oid,)).fetchone()
                result=json.loads(cached[0]) if cached else None
                if result is None or (refresh_reviews and (result['status'] not in ('SUPPORTED_NORMAL_OUTLIER','CONFIRMED_ERROR_RECALCULATED') or result.get('unresolved_retained_works') or result.get('unresolved_transfer_works'))):
                    client.allowed={normalize_doi(d) for d in dois}
                    if signals or t['orcids']:
                        result,work_rows=audit_author(paper,a,t,snap,client,rules,biography if signals else None)
                        if integer(a['t0_year'])!=integer(paper['submission_year']):
                            works=snap.history(result['old_pid'],aid,integer(paper['submission_year'])) if result['old_pid'] else []
                    else:
                        result,work_rows=offline_summary(a,works),list(light_rows(a,works))
                    if result.get('new_pid')!=result.get('old_pid'):
                        chosen_works=snap.history(result['new_pid'],aid,integer(paper['submission_year'])) if result['new_pid'] else []
                    else:chosen_works=works
                    reconcile_versions(a,works,chosen_works,work_rows)
                    trace(a,t,result,work_rows)
                    result['source_conflict_block']=bool(focal_conflicts(a,paper,group,snap) or any(x.startswith('CACHED_') and ('TITLE' in x or 'ROSTER' in x or 'AMBIGUOUS' in x) for x in signals))
                    result.update(offline_signals_json=json.dumps(signals),targeted_history_dois_json=json.dumps(sorted(dois)))
                    if result['status']=='REVIEW_REQUIRED':result['status']='Unresolved'
                    if result.get('unresolved_transfer_works') and result['status']=='CONFIRMED_ERROR_RECALCULATED':
                        result['status']='CONFIRMED_ERROR_REPAIR_BLOCKED'
                    state.execute('INSERT OR REPLACE INTO authors VALUES (?,?)',(oid,json.dumps(result,ensure_ascii=False)))
                    state.execute('DELETE FROM works WHERE oid=?',(oid,))
                    state.executemany('INSERT INTO works VALUES (?,?,?)',[(oid,i,json.dumps(r,ensure_ascii=False)) for i,r in enumerate(work_rows)])
                if result.get('source_conflict_block'):block.append('UNRESOLVED_SOURCE_CONFLICT')
                summaries.append(result);visited+=1
            if n%20==0:state.commit()
            newvalues=None
            if group and not block:
                try:
                    oldvalues=aggregate([(number(a[PUB]),number(a[AGE])) for a in group])
                    for a in group:
                        if not all(pd.notna(number(a[c])) for c in (PUB,AGE)):raise ValueError('Missing author baseline')
                    if any(not same(v,teams.loc[aid,f]) for f,v in zip(FIELDS,oldvalues)):block.append('AUTHOR_TEAM_BASELINE_CONFLICT')
                    newvalues=aggregate([(number(a['new_count']),number(a['new_age'])) for a in summaries])
                    if len({a['new_pid'] for a in summaries})!=len(group):block.append('CORRECTED_PID_COLLISION')
                    current=[paper[f] for f in FIELDS]
                    if not (all(same(x,y) for x,y in zip(current,oldvalues)) or all(same(x,y) for x,y in zip(current,newvalues))):block.append('STALE_MASTER_BASELINE')
                except (ValueError,KeyError):block.append('INVALID_AUTHOR_BASELINE')
            confirmed=any(a['status']=='CONFIRMED_ERROR_RECALCULATED' for a in summaries)
            pending=any(a['status']=='Unresolved' or a['unresolved_retained_works']>0 or a.get('unresolved_transfer_works',0)>0 for a in summaries)
            if any(a['status']=='CONFIRMED_ERROR_REPAIR_BLOCKED' for a in summaries):block.append('CONFIRMED_AUTHOR_ERROR_BLOCKED')
            if confirmed and not block and newvalues is not None:
                paperstatus='REPAIRED_WITH_REMAINING_UNCERTAINTY' if pending else 'REPAIRED_CONFIRMED_ERROR'
                if all(same(paper[f],v) for f,v in zip(FIELDS,newvalues)):paperstatus='ALREADY_REPAIRED'
                elif audit_only:paperstatus='PROPOSED_REPAIR_WITH_UNCERTAINTY' if pending else 'PROPOSED_REPAIR'
                for f,v in zip(FIELDS,newvalues):
                    if not same(paper[f],v):
                        changes.append({'paper_id':aid,'field':f,'old':paper[f],'new':str(float(v)),'reason':'AUTHOR_IDENTITY_OR_WRONG_PERSON_WORK_EVIDENCE'})
                        if not audit_only:edited.at[index[aid],f]=str(float(v))
            elif not block and summaries and all(a['status']=='SUPPORTED_NORMAL_OUTLIER' for a in summaries):paperstatus='SUPPORTED_NORMAL_OUTLIER'
            else:paperstatus='Unresolved'
            paper_rows.append(dict(paper_id=aid,status=paperstatus,authors_expected=teams.loc[aid,'expected_author_count'] if aid in teams.index else '',authors_audited=len(summaries),works_examined=sum(a['works_examined'] for a in summaries),identity_supported_works=sum(a['works_identity_supported'] for a in summaries),wrong_person_works_removed=sum(a['works_removed'] for a in summaries) if not block else 0,doi_attempts=sum(a['doi_requests_logical'] for a in summaries),doi_aligned=sum(a['doi_aligned'] for a in summaries),blocked_reasons_json=json.dumps(block),author_statuses_json=json.dumps(Counter(a['status'] for a in summaries))))
            work_done+=paper_rows[-1]['works_examined']
            observed=time.monotonic()-audit_started
            estimate_left=None if not work_done or observed<5 else round(max(0,workload-work_done)*observed/work_done+120)
            progress.update(n,authors_visited=visited,http_requests=client.requests,http_cache_hits=client.hits,
                            estimated_run_remaining_seconds=estimate_left,work_occurrences_done=work_done,
                            run_eta_basis='Observed seconds per work occurrence plus 120 seconds estimated for export')
            if n%100==0 or n==len(selected):
                print(f'Checked {n:,}/{len(selected):,} papers; {visited:,} author occurrences; HTTP new={client.requests:,}, cache={client.hits:,}',flush=True)
        state.commit()
    finally:
        state.rollback();client.db.commit()
    progress.begin('Validate and export results',None)
    non=[c for c in master if c not in FIELDS]
    if not master[non].equals(edited[non]) or master.shape!=edited.shape:raise AssertionError('Forbidden non-DBLP mutation')
    if sha(source)!=before:raise RuntimeError('Input changed during run')
    verdicts=pd.DataFrame(paper_rows)
    if set(verdicts.paper_id)!=ids or len(verdicts)!=len(ids):raise AssertionError('Not every outlier received a verdict')
    write(verdicts,out/'paper_verdicts.csv')
    author_log=pd.DataFrame([json.loads(x[0]) for x in state.execute('SELECT payload FROM authors ORDER BY oid')])
    if set(author_log.get('occurrence_id',[]))!=set(authors.occurrence_id):raise ValueError('Incomplete author ledger')
    distinct_works=int(author_log.get('distinct_works_logged',pd.Series(dtype=int)).sum())
    record_rows=int(author_log.get('record_rows_logged',pd.Series(dtype=int)).sum())
    if distinct_works!=int(verdicts.works_examined.sum()) or state.execute('SELECT count(*) FROM works').fetchone()[0]!=record_rows:
        raise ValueError('Incomplete work ledger')
    write(author_log,out/'author_verdicts.csv')
    with (out/'work_evidence.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=WORK_COLUMNS);writer.writeheader()
        for payload, in state.execute('SELECT payload FROM works ORDER BY oid,seq'):writer.writerow(json.loads(payload))
    write(pd.DataFrame(changes,columns=['paper_id','field','old','new','reason']),out/'changes.csv')
    result=out/'papers_features_dblp_repaired.csv';write(edited,result)
    if not read(result).equals(edited):raise AssertionError('Output round-trip mismatch')
    remaining,_=screen(edited);write(remaining,out/'outlier_flags_after.csv')
    summary=dict(distinct_occurrence_works=distinct_works,evidence_record_rows=record_rows,authors_without_name=int(author_log.author_name.eq('').sum()) if len(author_log) else 0,version=VERSION,mode='AUDIT_ONLY' if audit_only else 'REPAIR',all_outlier_papers_visited=True,outlier_papers=len(ids),papers_with_verdict=len(verdicts),author_occurrences_visited=visited,verdict_counts=verdicts.status.value_counts().to_dict(),changed_papers=0 if audit_only else len({r['paper_id'] for r in changes}),changed_cells=0 if audit_only else len(changes),proposed_cells=len(changes),rows=len(master),columns=len(master.columns),non_dblp_unchanged=True,input=str(source.resolve()),input_sha256=before,output_sha256=sha(result),snapshot_binding=binding,cache_job=job,legacy_cache=legacy.report,biography_http_requests=biography.requests if biography else 0,targeted_uncached_dois=len(client.seen),deferred_dois=sorted(client.deferred),external_dois_requested=sorted(client.seen),external_http_requests=client.requests,external_cache_hits=client.hits,elapsed_seconds=round(time.time()-start,1),screening=distribution,limitation='Visited means attempted record-level audit; missing publisher identity evidence is REVIEW_REQUIRED, not proof of a normal outlier. Source XML coverage is not real-world bibliographic completeness.')
    dump(summary,out/'summary.json');dump(cfg,out/'config_used.json');dump(rules,out/'reviewed_rules_used.json')
    comparison(cfg.get('baseline_v3_run'),out)
    (out/'paper_verdicts.partial.csv').unlink(missing_ok=True)
    snap.history.cache_clear();snap.db.close();state.close();client.db.close();print(json.dumps(summary,ensure_ascii=False,indent=2));print('OUTPUT:',out)
    return out

def run(cfg,refresh_reviews=False,audit_only=True,client_factory=Crossref):
    root=Path(cfg['output_root']);root.mkdir(parents=True,exist_ok=True)
    cache=Path(cfg['cache_root']);cache.mkdir(parents=True,exist_ok=True)
    lock=sqlite3.connect(cache/'v3_writer_lock.sqlite',timeout=0)
    try:lock.execute('BEGIN IMMEDIATE')
    except sqlite3.OperationalError:
        lock.close()
        raise RuntimeError('Another v3 run is using this cache, or the cache is not writable')
    progress=Progress(root/'progress.json',cfg.get('progress_interval_seconds',15))
    try:
        out=_run(cfg,refresh_reviews,audit_only,client_factory,progress)
    except BaseException:
        progress.close('interrupted_or_failed');raise
    finally:lock.close()
    progress.close('finished');return out

def rollback(run_directory,output):
    root=Path(run_directory);summary=json.loads((root/'summary.json').read_text());src=root/'papers_features_dblp_repaired.csv'
    if sha(src)!=summary['output_sha256']:raise ValueError('Output modified since audit; refusing rollback')
    df=read(src);lookup={p:i for i,p in enumerate(df.paper_id)}
    if summary['mode']!='AUDIT_ONLY':
        for r in read(root/'changes.csv').to_dict('records'):
            if r['field'] not in FIELDS or df.at[lookup[r['paper_id']],r['field']]!=r['new']:raise ValueError('Rollback guard failed')
            df.at[lookup[r['paper_id']],r['field']]=r['old']
    dst=Path(output)
    if dst.exists() or dst.resolve()==Path(summary['input']).resolve():raise ValueError('Use a new output path')
    write(df,dst)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',default=str(Path(__file__).with_name('config.json')));parser.add_argument('--refresh-reviews',action='store_true');mode=parser.add_mutually_exclusive_group();mode.add_argument('--audit-only',action='store_true');mode.add_argument('--apply',action='store_true');parser.add_argument('--rollback');parser.add_argument('--output')
    args=parser.parse_args()
    if args.rollback:rollback(args.rollback,args.output)
    else:
        path=Path(args.config).resolve();cfg=json.loads(path.read_text())
        for k in ('input_csv','recovery_root','xml_path','dtd_path','output_root','cache_root','local_work_dir','reviewed_rules','screen_previous_input_csv','legacy_cache_root','candidate_csv','baseline_v3_run'):
            if cfg.get(k) and not Path(cfg[k]).is_absolute():cfg[k]=str(path.parent/cfg[k])
        if cfg.get('recovery_history_csv') not in (None,'','auto'):
            cfg['recovery_history_csv']=str(path.parent/Path(cfg['recovery_history_csv']))
        run(cfg,args.refresh_reviews,not args.apply)
