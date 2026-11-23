"""输入论文主表；输出覆盖率、主键/标签约束检查与磁盘报告，不改写标签。"""
from pathlib import Path
import shutil
import pandas as pd
from fetch.control.tools import atomic_json,atomic_csv,utcnow

def collect(config,master,steps):
    root,raw,processed=config.paths
    problems=[]
    if master.arxiv_id.isna().any() or not master.arxiv_id.is_unique:problems.append('Master 主键为空或重复')
    c='s2_citation_count_3y';i='s2_influential_citation_count_3y'
    if {c,i}<=set(master):
        c3=pd.to_numeric(master[c],errors='coerce');ic3=pd.to_numeric(master[i],errors='coerce')
        if ((c3<0)|(ic3<0)|(c3<ic3)).any():problems.append('违反 C3 >= IC3 >= 0')
        valid=['COMPLETE','ZERO_CONFIRMED_BY_CUMULATIVE_COUNT']
        for col,status in [(c,'s2_c3_status'),(i,'s2_ic3_status')]:
            if status in master and (master[status].isin(valid)&master[col].isna()).any():problems.append(f'{col} 标记完成但值缺失')
    # 用唯一引用边核对完整标签；部分数据只报告观察量，不补成 0。
    edge_path=processed/'semantic_scholar_public/semantic_scholar_citation_edges_3y.csv'
    if edge_path.exists() and c in master and steps.get('semantic_scholar',{}).get('status') in ('SUCCESS','PARTIAL'):
        edges=pd.read_csv(edge_path,dtype={'arxiv_id':str,'citing_paper_id':str})
        if edges.duplicated(['arxiv_id','citing_paper_id']).any():problems.append('Citation edges 主键重复')
        window=edges[edges.window_status.eq('IN_WINDOW')]
        counts=window.groupby('arxiv_id').citing_paper_id.nunique()
        flags=window.is_influential.astype(str).str.lower().eq('true')
        influential=window[flags].groupby('arxiv_id').citing_paper_id.nunique()
        for label,status,count in [(c,'s2_c3_status',counts),(i,'s2_ic3_status',influential)]:
            mask=master[status].isin(['COMPLETE','ZERO_CONFIRMED_BY_CUMULATIVE_COUNT'])
            expected=master.arxiv_id.map(count).fillna(0)
            if (pd.to_numeric(master.loc[mask,label],errors='coerce')!=expected[mask]).any():
                problems.append(f'{label} 与 citation edges 不一致')
    coverage=[]
    for col in master:
        coverage.append(dict(field=col,total=len(master),non_missing=int(master[col].notna().sum()),coverage=float(master[col].notna().mean()) if len(master) else 0))
    files=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size) for p in root.rglob('*') if p.is_file()]
    usage=shutil.disk_usage(root)
    report=dict(created_at=utcnow(),rows=len(master),unique_ids=int(master.arxiv_id.nunique()),
                integrity_ok=not problems,problems=problems,
                incomplete_steps=[k for k,v in steps.items() if v['status'] not in ('SUCCESS','COMPLETE')],
                bytes=sum(f['bytes'] for f in files),free_bytes=usage.free,files=files,
                target_count=config.target_per_year*(config.end_year-config.start_year+1))
    for s in ['dblp','semantic_scholar','openreview']:
        col=s+'_status'
        report[s+'_status_counts']=master[col].value_counts(dropna=False).to_dict() if col in master else {}
    atomic_csv(pd.DataFrame(coverage),processed/'coverage.csv',index=False)
    atomic_json(processed/'quality_report.json',report)
    atomic_json(processed/'disk_usage.json',dict(bytes=report['bytes'],free_bytes=usage.free,files=files))
    return report
