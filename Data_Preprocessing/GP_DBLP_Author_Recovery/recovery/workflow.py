"""Unified offline workflow from immutable Fetch inputs and durable derived cache."""
import json,logging,time,fcntl,resource,csv
from pathlib import Path
from collections import defaultdict,Counter
import pandas as pd
from .progress import tqdm
from common import offline,sha256,fingerprint,export_query
from .io import write_csv,write_json
from .profile import ProfileIndex
from .index_init import ensure_index
from .bootstrap import load,revalidate
from .evidence import paper_recovery
from .focal import audit as audit_focal
from .database import Database
from .search import scan,dtd_audit
from .dedup import build_works
from .alignment import align
from .network import score_candidates,calibrate,winner
from .history import recover,PUB,AGE,FIRST
from .features import aggregate,FEATURES



def apply_identity(cfg,store,index,papers,rows,policy):
    db=store.db;groups=defaultdict(list);alignment_rows=[];final=[];conflicts=[]
    for r in rows:groups[r['arxiv_id']].append(r)
    for p in tqdm(papers,desc='Anchored sequence and identity resolution',unit='papers'):
        aid=p['arxiv_id'];g=sorted(groups[aid],key=lambda x:x['author_position']);proposals=[]
        targets=list(db.execute('SELECT t.key,t.method,p.payload FROM targets t JOIN publications p USING(key) WHERE t.aid=? AND t.reliable=1',(aid,)))
        rank={'ARXIV_ID':3,'DOI':2,'EXACT_TITLE':1}
        if targets:
            best=max(rank.get(t[1],0) for t in targets);targets=[t for t in targets if rank.get(t[1],0)==best]
        inconsistent=False
        for key,method,payload in targets:
            rec=json.loads(payload);pids=[index.unique(n) for n in rec['authors']]
            known={r['dblp_person_id'] for r in g if r['identity_status']=='VERIFIED'}
            if not known<=set(pids):
                inconsistent=True;conflicts.append(dict(arxiv_id=aid,reason='TARGET_CONTRADICTS_VERIFIED_ANCHOR',record_key=key,verified_pids_json=json.dumps(sorted(known)),target_pids_json=json.dumps(pids)));continue
            positions=[pids.index(r['dblp_person_id']) for r in g if r['identity_status']=='VERIFIED']
            if positions!=sorted(positions):
                inconsistent=True;conflicts.append(dict(arxiv_id=aid,reason='ANCHOR_ORDER_CONFLICT',record_key=key,verified_pids_json=json.dumps([r['dblp_person_id'] for r in g if r['identity_status']=='VERIFIED']),target_pids_json=json.dumps(pids)))
                continue
            result=align(g,rec['authors'],pids,cfg,True)
            if not result['evidence']:
                alignment_rows.append(dict(arxiv_id=aid,record_key=key,paper_match_method=method,original_author_names_json=json.dumps([r['original_name'] for r in g]),original_positions_json=json.dumps([r['author_position'] for r in g]),resolution_status='UNRESOLVED',failure_reason=result['reason']))
            alignment_rows.extend(dict(arxiv_id=aid,record_key=key,paper_match_method=method,**r) for r in result['evidence'])
            if result['accepted']:proposals.append((key,result,pids))
        signatures={tuple(x[2]) for x in proposals}
        if len(signatures)>1:
            inconsistent=True;conflicts.append(dict(arxiv_id=aid,reason='COMPETING_TARGET_AUTHOR_SEQUENCES',record_key='',verified_pids_json='',target_pids_json=json.dumps(list(signatures))))
        if proposals and not inconsistent and len(signatures)==1:
            key,result,pids=proposals[0];new=[]
            for evidence in result['evidence']:
                start,count=evidence['source_start'],evidence['source_count'];parents=g[start:start+count];targetpos=evidence['dblp_author_position']
                if len(parents)==1 and parents[0]['identity_status']=='VERIFIED':r=dict(parents[0])
                else:
                    r=dict(parents[0]);r.update(identity_status='VERIFIED',dblp_person_id=evidence['candidate_pid'],dblp_identity_method='ANCHORED_PAPER_SEQUENCE',dblp_identity_confidence='HIGH',repaired_name=evidence['recovered_author_name'],dblp_author_matched_to_paper=True,evidence_json=json.dumps({'rule':'ANCHORED_PAPER_SEQUENCE','record_key':key,'alignment':evidence}),source_occurrences_json=json.dumps(sorted(set(v for x in parents for v in json.loads(x['source_occurrences_json'])))),source_positions_json=json.dumps(sorted(set(v for x in parents for v in json.loads(x['source_positions_json'])))))
                    if evidence['alignment_operation']!='ONE_TO_ONE':r['occurrence_id']=aid+':aligned:'+str(targetpos)
                r['author_position']=targetpos;new.append(r)
            g=new
        for r in g:
            if r['identity_status']=='VERIFIED':continue
            rule=policy.get('selected_rule');prediction,evidence=winner(db,r['occurrence_id'],rule['minimum_joint'] if rule else 5,rule['margin'] if rule else 8.)
            if policy['automatic_network_confirmation_enabled'] and prediction and not inconsistent:
                if prediction in {x['dblp_person_id'] for x in g if x['identity_status']=='VERIFIED'}:
                    conflicts.append(dict(arxiv_id=aid,reason='NETWORK_PID_ALREADY_IN_TEAM',record_key='',verified_pids_json=prediction,target_pids_json=json.dumps(evidence)))
                else:r.update(identity_status='VERIFIED',dblp_person_id=prediction,dblp_author_matched_to_paper=True,dblp_identity_method='CALIBRATED_PRE_T0_COAUTHOR_RULE',dblp_identity_confidence='HIGH',evidence_json=json.dumps(evidence))
            else:r.update(identity_status='UNRESOLVED',dblp_person_id='',dblp_author_matched_to_paper=False,final_failure_reason='CONFLICT_REQUIRES_REVIEW' if inconsistent else 'INSUFFICIENT_VALIDATED_IDENTITY_EVIDENCE',coauthor_evidence_json=json.dumps(evidence))
        # Preserve every previous confirmed occurrence, even if new contradictory evidence appears.
        for old in groups[aid]:
            if old['identity_status']=='VERIFIED':
                matches=[r for r in g if old['occurrence_id']==r['occurrence_id'] or old['occurrence_id'] in json.loads(r['source_occurrences_json'])]
                if len(matches)!=1 or matches[0]['dblp_person_id']!=old['dblp_person_id']:raise AssertionError('Verified anchor modified')
        counts=Counter(r['dblp_person_id'] for r in g if r['identity_status']=='VERIFIED')
        if any(n>1 for n in counts.values()):raise ValueError('Duplicate confirmed person in final team: '+aid)
        keys=[x[0] for x in db.execute('SELECT key FROM focal WHERE aid=?',(aid,))]
        for r in g:r['focal_record_keys_json']=json.dumps(keys);final.append(r)
    write_csv(cfg,'matching/sequence_alignment_results.csv',alignment_rows,['arxiv_id','record_key','paper_match_method','original_author_names_json','original_positions_json','source_occurrences_json','recovered_author_name','dblp_author_position','candidate_pid','alignment_operation','name_similarity','anchor_consistency','alignment_score','second_best_score','score_margin','confidence','resolution_status','source_start','source_count','failure_reason'])
    inherited=[]
    for r in final:
        if r.get('fetch_identity_conflict') or (r.get('fetch_claimed_verified') and r.get('revalidation_status') not in ('CANONICAL_PID_AND_ALIAS_SUPPORT_EXISTING_FETCH_MATCH','')):
            inherited.append(dict(arxiv_id=r['arxiv_id'],reason='FETCH_REVALIDATION_'+(r.get('fetch_identity_conflict') or r.get('revalidation_status','')),record_key='',verified_pids_json=r.get('source_pid',''),target_pids_json=r.get('dblp_person_id',''),resolution_status='RESOLVED_WITH_NEW_EVIDENCE' if r['identity_status']=='VERIFIED' else 'UNRESOLVED',evidence_json=r.get('evidence_json','')))
    write_csv(cfg,'matching/identity_conflicts.csv',[{**c,'resolution_status':'REQUIRES_REVIEW'} for c in conflicts]+inherited,['arxiv_id','reason','record_key','verified_pids_json','target_pids_json','resolution_status','evidence_json'])
    return final,conflicts


def output_features(cfg,papers,rows,master,old,conflicts=()):
    data=pd.DataFrame(rows);groups={aid:g for aid,g in data.groupby('arxiv_id',sort=False)}
    adjusted=[]
    for p in papers:
        g=groups[p['arxiv_id']];lineage=set(v for raw in g.source_occurrences_json for v in json.loads(raw));delta=len(g)-len(lineage)
        adjusted.append({**p,'expected_author_count':int(p['expected_author_count'])+delta})
    papers=adjusted;team=aggregate(pd.DataFrame(papers),data)
    states=[];conflict_counts=Counter(r['arxiv_id'] for r in conflicts)
    for r in team.to_dict('records'):
        g=groups[r['arxiv_id']];confirmed=g.identity_status.eq('VERIFIED');complete=r['partial_team_status']=='COMPLETE_IDENTITY_SET';pub=r['publication_count_status']=='COMPLETE';age=r['academic_age_status']=='COMPLETE'
        r.update(dblp_author_identity_complete=complete,dblp_author_match_count=int(confirmed.sum()),dblp_team_author_matched_count=int(confirmed.sum()),dblp_team_author_row_count=len(g),dblp_team_all_authors_matched=complete,dblp_publication_count_available=pub,dblp_academic_age_available=age,dblp_team_publication_features_status='COMPLETE' if pub else 'INCOMPLETE',dblp_team_academic_age_features_status='COMPLETE' if age else 'INCOMPLETE',dblp_status='COMPLETE' if pub and age else 'PARTIAL',dblp_match_status='FULL' if complete else 'PARTIAL' if confirmed.any() else 'NO_VERIFIED',dblp_match_method='LOCAL_DBLP_EVIDENCE',dblp_unresolved_author_count=int((~confirmed).sum()),dblp_academic_age_missing_author_count=int(g[AGE].isna().sum()),dblp_identity_conflict_author_count=0)
        for level in ('HIGH','MEDIUM','LOW'):
            count=int((confirmed & g.dblp_identity_confidence.eq(level)).sum());r['dblp_'+level.lower()+'_confidence_author_count']=count;r['dblp_team_'+level.lower()+'_confidence_author_count']=count
        r['dblp_all_authors_high_confidence']=bool((confirmed & g.dblp_identity_confidence.eq('HIGH')).all());r['dblp_all_authors_high_or_medium_confidence']=bool((confirmed & g.dblp_identity_confidence.isin(['HIGH','MEDIUM'])).all())
        r['dblp_focal_work_excluded_from_author_history']=bool(g.dblp_focal_work_excluded_from_history.any());r['dblp_focal_work_excluded_author_count']=int(g.dblp_focal_work_excluded_from_history.sum())
        r['dblp_identity_conflict_count']=conflict_counts[r['arxiv_id']]
        if conflict_counts[r['arxiv_id']]:r['dblp_status']='REVIEW_REQUIRED';r['dblp_identity_conflict_author_count']=None
        states.append(r)
    team=pd.DataFrame(states);preview=master.copy();indexed=team.set_index('arxiv_id');changes=[]
    for col in master:
        if col.startswith('dblp_') and col not in team:preview['source_'+col]=master[col];preview[col]=''
    for col in team:
        if col!='arxiv_id':preview[col]=preview.arxiv_id.map(indexed[col])
    preview['source_author_count_v1']=master.get('author_count_v1','');preview['source_authors_v1_json']=master.get('authors_v1_json','')
    preview['author_count_v1']=preview.arxiv_id.map(indexed.expected_author_count);preview['arxiv_author_count']=preview['author_count_v1']
    preview['recovered_author_names_json']=preview.arxiv_id.map(lambda aid:json.dumps(groups[aid].repaired_name.tolist(),ensure_ascii=False))
    # Preserve source author objects where possible; retain every parent object for merges.
    if 'authors_v1_json' in master:
        for i,r in master.iterrows():
            original=json.loads(r.authors_v1_json) if r.authors_v1_json else [];new=[]
            source_positions={v for value in groups[r.arxiv_id].source_positions_json for v in json.loads(value)}
            if original and source_positions!=set(range(1,len(original)+1)):continue
            for a in groups[r.arxiv_id].to_dict('records'):
                positions=json.loads(a['source_positions_json']);pos=int(positions[0])-1;obj=original[pos] if pos<len(original) else {'name':a['original_name']}
                if isinstance(obj,dict):obj={**obj,'name':a['repaired_name']}
                else:obj=a['repaired_name']
                new.append(obj)
            preview.loc[i,'authors_v1_json']=json.dumps(new,ensure_ascii=False)
    for col in FEATURES:
        if col not in master:continue
        a=pd.to_numeric(master[col],errors='coerce');b=pd.to_numeric(preview[col],errors='coerce')
        for i in master.index[a.notna() & (b.isna() | ((a-b).abs()>1e-9))]:changes.append(dict(arxiv_id=master.loc[i,'arxiv_id'],feature=col,previous=a[i],current=b[i],reason='COMPLETE_SNAPSHOT_RECALCULATION_REQUIRES_EVIDENCE_REVIEW'))
    allowed={'author_count_v1','authors_v1_json','arxiv_author_count'}|set(team.columns)
    for col in master:
        if not col.startswith('dblp_') and col not in allowed:assert master[col].tolist()==preview[col].tolist(),col
    assert not ((preview.dblp_status=='COMPLETE') & preview[FEATURES].isna().any(axis=1)).any()
    write_csv(cfg,'features/final_team_features.csv',team);write_csv(cfg,'features/final_author_features.csv',data);write_csv(cfg,'features/final_papers_master_preview.csv',preview)
    write_csv(cfg,'report/feature_regression_changes.csv',changes,['arxiv_id','feature','previous','current','reason'])
    comparison=[dict(feature=c,previous_available=int(pd.to_numeric(master.get(c,pd.Series(dtype=float)),errors='coerce').notna().sum()),final_available=int(team[c].notna().sum())) for c in FEATURES]
    write_csv(cfg,'report/feature_comparison.csv',comparison)
    return comparison,changes


def run(cfg,stage='all'):
    cfg.validate();start=time.time();summary={'status':'RUNNING','requested_stage':stage,'network_policy':'OFFLINE_SOCKET_AUDIT_AND_NO_HTTP_CLIENT','profile_index_rebuilt':False,'profile_index_copied':False};store=index=None;locks=[]
    logging.basicConfig(level=cfg.log_level,format='%(asctime)s %(levelname)s %(message)s',handlers=[logging.StreamHandler(),logging.FileHandler(cfg.out('logs/recovery.log'),encoding='utf8')],force=True)
    write_json(cfg,'report/final_recovery_summary.json',summary)
    sources={}
    with offline() as network:
        try:
            for path in (cfg.out('run.lock'),cfg.derived('run.lock')):
                lock=path.open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
            papers,rows,master,old,sources=load(cfg)
            index,created=ensure_index(cfg)
            summary['profile_index_created']=created
            summary['profile_index_reused']=not created
            rows=revalidate(cfg,index,rows)
            for p in (cfg.PROFILE_INDEX_PATH,cfg.DTD_PATH):sources[str(p)]=sha256(p)
            audit=dtd_audit(cfg.DTD_PATH);write_json(cfg,'report/dtd_association_audit.json',audit)
            summary.update(paper_count=len(papers),input_author_occurrences=len(rows),fetch_claimed_verified=sum(r.get('fetch_claimed_verified',False) for r in rows),previous_verified=sum(r['identity_status']=='VERIFIED' for r in rows),previous_unresolved=sum(r['identity_status']!='VERIFIED' for r in rows),xml_available=Path(cfg.DBLP_XML_PATH).is_file())
            derived_path=cfg.derived('candidate_history.sqlite')
            if not cfg.resume_enabled and derived_path.exists():raise ValueError('resume_enabled=false requires an empty DERIVED_CACHE_DIR; existing caches are not deleted')
            store=Database(derived_path,cfg.sqlite_cache_mb)
            # Bind to input identity, original profile snapshot and semantics, not machine paths.
            semantic={k:v for k,v in cfg.snapshot().items() if k in ('alignment_window','alignment_margin','alignment_min_similarity','title_similarity','title_keyword_coverage','title_year_window')}
            signature=json.dumps(dict(sources=sorted(sources.values()),snapshot=index.signature[:3],rules=semantic,algorithm_sha256={p.name:sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}),sort_keys=True)
            store.bind(signature)
            logging.info('Stage: candidate expansion/export');store.candidates(index,rows);store.export_candidates(cfg)
            summary['expanded_candidate_rows']=store.db.execute('SELECT COUNT(*) FROM candidates').fetchone()[0];summary['eligible_distinct_candidate_pids']=store.db.execute('SELECT COUNT(*) FROM wanted').fetchone()[0]
            write_json(cfg,'report/effective_config.json',cfg.snapshot());write_json(cfg,'report/source_manifest.json',sources)
            if stage=='prepare':summary['status']='PREPARED' if summary['xml_available'] else 'PREPARED_XML_REQUIRED';return summary
            if not summary['xml_available']:raise FileNotFoundError('DBLP_XML_PATH missing. Full XML is mandatory; downloading and partial-cache substitution are disabled.')
            fp=fingerprint(cfg.DBLP_XML_PATH);expected=index.signature[1]
            if any(fp[k]!=expected[k] for k in ('size','sample_sha256')):raise ValueError('XML differs from existing profile snapshot; no silent rebuild')
            sources[str(cfg.DBLP_XML_PATH)]=sha256(cfg.DBLP_XML_PATH);summary['xml_snapshot']={**fp,'sha256':sources[str(cfg.DBLP_XML_PATH)]};write_json(cfg,'report/source_manifest.json',sources)
            if store.get('xml_sha256') and store.get('xml_sha256')!=sources[str(cfg.DBLP_XML_PATH)]:raise ValueError('Derived XML fingerprint changed')
            store.set('xml_sha256',sources[str(cfg.DBLP_XML_PATH)])
            for r in rows:
                for key in json.loads(r.get('focal_record_keys_json') or '[]'):
                    store.db.execute('INSERT OR IGNORE INTO focal VALUES (?,?)',(r['arxiv_id'],key))
                    store.db.execute('INSERT OR REPLACE INTO focal_audit VALUES (?,?,?,?,?)',(r['arxiv_id'],key,'FETCH_INPUT','FETCH_TARGET_OR_CANDIDATE_RECORD',0))
            store.db.commit();logging.info('Stage: alias preparation and XML scans');store.aliases(index);scan(cfg,store,index,papers,rows)
            logging.info('Stage: disk deduplication');build_works(store)
            summary.update(xml_records_processed=int(store.get('scan_primary_checkpoint',0)),target_paper_hits=store.db.execute('SELECT COUNT(*) FROM targets').fetchone()[0],candidate_history_records=store.db.execute('SELECT COUNT(*) FROM candidate_publications').fetchone()[0])
            for path,query in [
                ('report/focal_work_exclusion_audit.csv','SELECT f.*,t.reliable AS current_identity_reliable,r.work AS excluded_work_id FROM focal_audit f LEFT JOIN targets t ON t.aid=f.aid AND t.key=f.key LEFT JOIN record_works r ON r.key=f.key'),
                ('matching/target_paper_search_results.csv','SELECT t.*,p.title,p.year,p.payload FROM targets t JOIN publications p USING(key)'),
                ('matching/candidate_history_evidence.csv','SELECT c.pid,c.evidence,p.*,r.work FROM candidate_publications c JOIN publications p USING(key) LEFT JOIN record_works r USING(key)'),
                ('matching/candidate_history_status.csv','SELECT * FROM candidate_history_status')]:export_query(cfg,path,store.db,query)
            if stage=='scan':summary['status']='LOCAL_SCAN_COMPLETE';return summary
            logging.info('Stage: network scores and calibration')
            score_candidates(cfg,store,papers);summary['coauthor_relations']=store.db.execute('SELECT COUNT(*) FROM candidate_coauthors').fetchone()[0];policy=calibrate(cfg,store);summary['backtest']=policy
            logging.info('Stage: identity recovery, history and features')
            recovered_rows=paper_recovery(cfg,store,index,papers,rows)
            final,conflicts=apply_identity(cfg,store,index,papers,recovered_rows,policy);final,temporal=recover(cfg,store,final)
            comparisons,changes=output_features(cfg,papers,final,master,old,conflicts)
            write_csv(cfg,'matching/final_author_identity_mapping.csv',final)
            pending=[r for r in final if r['identity_status']!='VERIFIED']
            columns=sorted(set().union(*(r.keys() for r in final))) if final else ['arxiv_id']
            write_csv(cfg,'matching/final_unresolved_authors.csv',pending,columns);write_csv(cfg,'fallback/semantic_scholar_fallback_queue.csv',pending,columns)
            temporal.update(audit_focal(store));write_json(cfg,'report/temporal_integrity_audit.json',temporal)
            summary.update(status='COMPLETE_WITH_REVIEW_ITEMS' if conflicts or changes else 'COMPLETE',final_author_occurrences=len(final),verified_author_count=len(final)-len(pending),unresolved_author_count=len(pending),identity_conflicts=len(conflicts),feature_regression_review_count=len(changes),feature_comparison=comparisons,temporal_validation=temporal,full_xml_scan_complete=True,history_status_counts=dict(Counter(r['history_status'] for r in final)))
        except BaseException as exc:
            summary.update(status='BLOCKED_MISSING_INPUT' if isinstance(exc,FileNotFoundError) else 'FAILED',error=repr(exc));logging.exception('Local recovery stopped without declaring completion');raise
        finally:
            logging.info('Stage: final source integrity verification')
            summary['source_files_unchanged']=all(Path(p).is_file() and sha256(p)==sha for p,sha in sources.items())
            summary['network_attempts_blocked']=network['network_attempts_blocked'];summary['subprocess_attempts_blocked']=network['subprocess_attempts_blocked'];summary['elapsed_seconds']=round(time.time()-start,3);summary['peak_process_rss_mb']=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,2)
            if store:store.db.close()
            if index:index.db.close()
            for lock in locks:lock.close()
            write_json(cfg,'report/final_recovery_summary.json',summary)
            cfg.out('report/final_recovery_report.md').write_text('# GP_DBLP_Author_Recovery\n\n状态如实区分准备、全量扫描和完成；PREPARED / BLOCKED 不代表完成身份或特征恢复。\n\n```json\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n```\n',encoding='utf8')
            if not summary['source_files_unchanged']:raise RuntimeError('Source mutated during execution')
    return summary
