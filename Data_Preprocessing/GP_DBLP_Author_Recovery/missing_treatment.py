"""Final-only zero-history treatment. Reads completed exports; never opens DBLP/SQLite.
The recovery algorithms and their cache signatures remain byte-for-byte unchanged.
"""
import argparse,fcntl,json,logging,math
from collections import Counter
from pathlib import Path
import pandas as pd
from common import sha256
from config import Config
from recovery.features import aggregate,FEATURES
from recovery.history import PUB,AGE,FIRST
from recovery.io import write_csv,write_json

ZERO='ZERO_PRIOR_WORKS_IN_DBLP_SNAPSHOT'
COMPLETE='COMPLETE_OBSERVED_DBLP_HISTORY'
POLICY='CONFIRMED_COMPLETE_ZERO_HISTORY_COUNT_AND_AGE_ZERO_V1'


def apply_treatment(authors):
    """Only certified zero-history records receive zeros; preserve evidence/status fields."""
    out=authors.copy();reasons=[]
    out['pre_treatment_publication_count']=out[PUB]
    out['pre_treatment_academic_age']=out[AGE]
    out[PUB]=pd.to_numeric(out[PUB],errors='coerce')
    out[AGE]=pd.to_numeric(out[AGE],errors='coerce')
    for i,r in out.iterrows():
        verified=r['identity_status']=='VERIFIED' and bool(str(r['dblp_person_id']).strip())
        status=r['history_status'];count=r[PUB];age=r[AGE]
        first=pd.to_numeric(pd.Series([r[FIRST]]),errors='coerce').iloc[0]
        first_is_missing=pd.isna(r[FIRST]) or str(r[FIRST]).strip()==''
        if not verified:
            out.loc[i,[PUB,AGE]]=float('nan');reason='UNRESOLVED_IDENTITY_REMAINS_MISSING'
        elif status==ZERO and pd.notna(count) and count==0 and first_is_missing:
            out.loc[i,[PUB,AGE]]=0.0;reason='CONFIRMED_COMPLETE_ZERO_HISTORY_COUNT_AND_AGE_ZERO'
        elif status==COMPLETE and pd.notna(count) and math.isfinite(count) and count>0 and count%1==0 and pd.notna(first):
            # Use existing history values; do not recalculate dates or works here.
            if pd.isna(age) or not math.isfinite(age) or age<0:
                out.loc[i,AGE]=float('nan');reason='EXISTING_HISTORY_ACADEMIC_AGE_UNAVAILABLE'
            else:reason='EXISTING_VALID_HISTORY_VALUES_RETAINED'
        else:
            out.loc[i,[PUB,AGE]]=float('nan')
            reason='HISTORY_NOT_CERTIFIED_ZERO_OR_VALID:'+str(status)
        reasons.append(reason)
    out['missing_treatment_reason']=reasons
    out['missing_treatment_policy']=POLICY
    return out


def coverage(table,ids):
    if 'arxiv_id' in table:table=table.set_index('arxiv_id').reindex(ids)
    n=len(ids);result={}
    for col in FEATURES:
        values=pd.to_numeric(table[col],errors='coerce') if col in table else pd.Series(float('nan'),index=range(n))
        valid=int(values.notna().sum())
        result[col]={'valid_count':valid,'missing_count':n-valid,'coverage_ratio':valid/n if n else None}
    return result


def run(cfg):
    cfg.validate();lock=cfg.out('run.lock').open('a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _run_locked(cfg)
    finally:lock.close()


def _run_locked(cfg):
    logging.info('Final Missing Treatment only: no XML scan, no SQLite access')
    root=Path(cfg.OUTPUT_DIR)
    paths={'summary':root/'report/final_recovery_summary.json',
           'manifest':root/'report/source_manifest.json',
           'temporal':root/'report/temporal_integrity_audit.json',
           'authors':root/'features/final_author_features.csv',
           'teams':root/'features/final_team_features.csv',
           'master':Path(cfg.FETCH_DATA_ROOT)/cfg.master_file}
    for path in paths.values():
        if not path.is_file():raise FileNotFoundError('Treatment requires a completed recovery output: '+str(path))
    summary=json.loads(paths['summary'].read_text());audit=json.loads(paths['temporal'].read_text())
    if summary.get('status') not in ('COMPLETE','COMPLETE_WITH_REVIEW_ITEMS') or summary.get('full_xml_scan_complete') is not True or summary.get('source_files_unchanged') is not True:
        raise ValueError('Recovery is not a completed, source-verified full XML run; no zero filling is allowed')
    for key in ('included_same_or_future_year','included_focal_works','target_candidates_missing_exclusion','coauthor_focal_leakage_count','coauthor_same_or_future_year_count'):
        if audit.get(key)!=0:raise ValueError('Existing temporal/Focal audit is incomplete or failed: '+key)
    hashes={name:sha256(path) for name,path in paths.items()}
    manifest=json.loads(paths['manifest'].read_text())
    if manifest.get(str(paths['master'].resolve()))!=hashes['master']:
        raise ValueError('Original master differs from the master used by completed recovery')
    # Also check the source roster. No XML/index hashing or access is necessary.
    roster=Path(cfg.FETCH_DATA_ROOT)/cfg.authors_file
    if not roster.is_file() or manifest.get(str(roster.resolve()))!=sha256(roster):
        raise ValueError('Source author roster differs from completed recovery')
    read=lambda path:pd.read_csv(path,dtype=str,keep_default_na=False)
    master=read(paths['master']);authors=read(paths['authors']);teams=read(paths['teams'])
    for label,frame in [('master',master),('teams',teams)]:
        if 'arxiv_id' not in frame or frame.arxiv_id.eq('').any() or frame.arxiv_id.duplicated().any():raise ValueError('Missing/duplicate paper IDs in '+label)
    needed={'arxiv_id','author_position','identity_status','dblp_person_id','dblp_author_matched_to_paper','history_status',PUB,AGE,FIRST}
    if not needed<=set(authors):raise ValueError('Incomplete author feature schema: '+str(sorted(needed-set(authors))))
    if not set(FEATURES+['expected_author_count'])<=set(teams):raise ValueError('Incomplete team feature schema')
    ids=master.arxiv_id.tolist()
    if set(teams.arxiv_id)!=set(ids) or set(authors.arxiv_id)!=set(ids):raise ValueError('Master/team/author cohorts differ')
    if summary.get('paper_count')!=len(master) or summary.get('final_author_occurrences')!=len(authors):raise ValueError('Summary/export row counts differ')
    positions=pd.to_numeric(authors.author_position,errors='raise')
    if (positions<1).any() or (positions%1!=0).any():raise ValueError('Invalid author positions')
    authors['author_position']=positions.astype(int)
    if authors.duplicated(['arxiv_id','author_position']).any():raise ValueError('Duplicate author occurrence')
    papers=teams.set_index('arxiv_id').loc[ids,['expected_author_count']].reset_index()
    expected=pd.to_numeric(papers.expected_author_count,errors='raise')
    if (expected<1).any() or (expected%1!=0).any():raise ValueError('Invalid expected author count')
    papers['expected_author_count']=expected.astype(int)
    # Establish before-coverage from the same strict aggregation, not stale Fetch values.
    before=aggregate(papers,authors)
    indexed=teams.set_index('arxiv_id').loc[ids]
    for col in FEATURES:
        actual=pd.to_numeric(before[col],errors='coerce').reset_index(drop=True)
        recorded=pd.to_numeric(indexed[col],errors='coerce').reset_index(drop=True)
        same=(actual.isna() & recorded.isna()) | ((actual-recorded).abs()<=1e-9)
        if not same.all():raise ValueError('Completed team/author exports disagree: '+col)
    treated=apply_treatment(authors);after=aggregate(papers,treated)
    newmaster=master.copy();after_index=after.set_index('arxiv_id')
    for col in FEATURES:newmaster[col]=master.arxiv_id.map(after_index[col])
    untouched=[c for c in master.columns if c not in FEATURES]
    if not newmaster[untouched].equals(master[untouched]):raise AssertionError('Non-feature source columns changed')
    if len(newmaster)!=len(master) or newmaster.arxiv_id.tolist()!=ids:raise AssertionError('Master rows/order changed')
    if list(newmaster.columns)!=list(master.columns)+[c for c in FEATURES if c not in master]:raise AssertionError('Master column ordering changed')
    zero=treated.missing_treatment_reason.eq('CONFIRMED_COMPLETE_ZERO_HISTORY_COUNT_AND_AGE_ZERO')
    unresolved=~((treated.identity_status=='VERIFIED') & treated.dblp_person_id.str.strip().ne(''))
    missing_age_before=pd.to_numeric(authors[AGE],errors='coerce').isna()
    before_coverage=coverage(before,ids);after_coverage=coverage(after,ids)
    change={c:{'before':before_coverage[c],'after':after_coverage[c],
               'valid_count_change':after_coverage[c]['valid_count']-before_coverage[c]['valid_count'],
               'coverage_percentage_point_change':100*(after_coverage[c]['coverage_ratio']-before_coverage[c]['coverage_ratio']) if ids else None} for c in FEATURES}
    result={'status':'COMPLETE','policy':POLICY,'author_count_unit':'AUTHOR_OCCURRENCES_NOT_UNIQUE_PERSONS',
            'zero_history_author_count':int(zero.sum()),'academic_age_newly_zero_filled_count':int((zero & missing_age_before).sum()),
            'zero_history_unique_pid_count':int(treated.loc[zero,'dblp_person_id'].nunique()),
            'unresolved_author_count':int(unresolved.sum()),
            'confirmed_authors_with_incomplete_history_count':int(((~unresolved)&treated[PUB].isna()).sum()),
            'history_missing_year_author_count':int((treated.history_status=='MISSING_YEAR').sum()),
            'reason_counts':dict(Counter(treated.missing_treatment_reason)),
            'input_paper_count':len(master),'retained_paper_count':len(newmaster),'excluded_paper_count':0,
            'before_after_coverage':change,'original_master_coverage':coverage(master,ids),
            'final_features':after_coverage,'input_sha256':hashes,
            'original_order_preserved':True,'all_non_feature_fields_preserved':True,
            'xml_scanned_by_treatment':False,'sqlite_opened_by_treatment':False,
            'master_output':'features/papers_master_dblp_imputed.csv',
            'note':'Zero reflects qualified history in the complete DBLP snapshot, not proof about publications outside DBLP. Original metadata columns remain unchanged.'}
    # Check inputs again before publishing derived files; never write original files.
    if any(sha256(paths[k])!=h for k,h in hashes.items()):raise RuntimeError('Treatment inputs changed during execution')
    write_csv(cfg,'features/treated_author_features.csv',treated)
    write_csv(cfg,'features/treated_team_features.csv',after)
    write_csv(cfg,'features/papers_master_dblp_imputed.csv',newmaster)
    # Verify the actual CSV round trip for all protected fields and paper order.
    saved=read(root/'features/papers_master_dblp_imputed.csv')
    if not saved[untouched].equals(master[untouched]) or saved.arxiv_id.tolist()!=ids:raise AssertionError('Written master preservation check failed')
    for col in FEATURES:
        a=pd.to_numeric(saved[col],errors='coerce');b=pd.to_numeric(newmaster[col],errors='coerce')
        if not ((a.isna()&b.isna())|((a-b).abs()<=1e-9)).all():raise AssertionError('Written feature round trip failed: '+col)
    write_json(cfg,'report/missing_treatment_summary.json',result)
    lines=['# Missing Treatment Summary','',f'保留论文：{len(newmaster)}/{len(master)}；删除：0。',
           f'确认零历史作者出现记录：{int(zero.sum())}；Academic Age 新填 0：{int((zero & missing_age_before).sum())}。',
           f'身份未确认作者出现记录：{int(unresolved.sum())}。','',
           '| 特征 | Treatment 前有效 | Treatment 后有效 | 最终 Missing | Coverage 前→后 |',
           '|---|---:|---:|---:|---|']
    for col in FEATURES:
        a=before_coverage[col];b=after_coverage[col]
        lines.append(f"| {col} | {a['valid_count']} | {b['valid_count']} | {b['missing_count']} | {a['coverage_ratio']:.2%} → {b['coverage_ratio']:.2%} |")
    lines+=['','仅更新原始主表四个 DBLP 特征；所有行、顺序、其他字段保留。原有 dblp_status 等字段为源数据元数据，故不改写；最新完整性状态见 treated_team_features.csv。',
            '未访问 XML/SQLite；Temporal/Focal 审计原件不修改；年份不明、扫描不完整或未观察到记录但不能认证零历史的作者保持 Missing。',
            '首次发表年份不填 0。完整零历史的 Academic Age=0 是最终 Treatment 约定。']
    cfg.out('report/missing_treatment_summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    logging.info('Treatment complete: %s',root/'features/papers_master_dblp_imputed.csv')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='config.json');args=p.parse_args()
    print(json.dumps(run(Config.load(args.config)),ensure_ascii=False,indent=2))
