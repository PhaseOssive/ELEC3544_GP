"""运行状态、阶段 checkpoint、表缓存与失败恢复记录。所有状态原子落盘。"""
from pathlib import Path
from uuid import uuid4
import json
import platform
import pandas as pd
from fetch.control.tools import atomic_json,atomic_csv,fingerprint,checksum,utcnow,SCHEMA_VERSION,ALGORITHM_VERSION

class State:
    def __init__(self,config):
        self.config=config;self.root=config.paths[0]/'state'
        self.run_id='GP_'+utcnow().replace(':','').replace('-','').split('.')[0]+'_'+uuid4().hex[:8]
        self.doc=dict(run_id=self.run_id,status='RUNNING',started_at=utcnow(),steps={},config=config.snapshot(),
                      python=platform.python_version(),schema_version=SCHEMA_VERSION,algorithm_version=ALGORITHM_VERSION)
        self.save()

    def save(self):
        atomic_json(self.root/'runs'/f'{self.run_id}.json',self.doc)
        atomic_json(self.root/'latest.json',self.doc)

    def step(self,name,result):
        self.doc['current_step']=name;self.doc['steps'][name]=result;self.save()

    def error(self,step,error,paper_id=None,retry_count=0):
        row=dict(timestamp=utcnow(),run_id=self.run_id,step=step,paper_id=paper_id,
                 error_type=type(error).__name__,message=str(error),retry_count=retry_count,recovery_status='PENDING')
        with (self.root.parent/'logs/errors.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        return row

    def recovered(self,step):
        # 追加恢复事件，保留历史错误而不改写审计轨迹。
        row=dict(timestamp=utcnow(),run_id=self.run_id,step=step,paper_id=None,error_type=None,
                 message='本阶段已完成；此前该阶段失败任务已恢复',retry_count=0,recovery_status='RECOVERED')
        with (self.root.parent/'logs/errors.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')

    def key(self,name,tables):
        # 路径、日志与重试参数不影响研究结果；采样/窗口/分页参数影响缓存。
        cfg=self.config.snapshot()
        for k in ('cache_policy','refresh_sources','resume_enabled','max_retries','request_timeout','wait_for_reconnection','reconnection_delay','max_retry_delay','log_level','source_intervals','minimum_free_gb','sources'):
            cfg.pop(k,None)
        inputs={k:fingerprint(v.to_json(orient='split',date_format='iso')) for k,v in tables.items()}
        if self.config.seed_csv:cfg['seed_checksum']=checksum(self.config.seed_csv)
        if self.config.dblp_xml_path:
            p=Path(self.config.dblp_xml_path);cfg['dblp_stat']=[p.stat().st_size,p.stat().st_mtime_ns]
        return fingerprint([name,cfg,inputs,SCHEMA_VERSION,ALGORITHM_VERSION])

    def load_tables(self,name,key):
        if not self.config.resume_enabled or self.config.refresh('arxiv' if name.startswith('arxiv') else name):return None
        directory=self.root/'processed_cache'/name
        try:
            m=json.loads((directory/'manifest.json').read_text())
            if m['key']!=key or not m['complete']:return None
            for entry in m['outputs']:
                if checksum(entry['path'])!=entry['sha256']:return None
            result={}
            for k,v in m['tables'].items():
                p=directory/v['file']
                if checksum(p)!=v['sha256']:return None
                df=pd.read_json(p,orient='table')
                for col in v.get('object_columns',[]): df[col]=df[col].astype(object)
                for col,positions in v.get('none_cells',{}).items():
                    for pos in positions:df.at[pos,col]=None
                if list(df.columns)!=v['columns'] or len(df)!=v['rows']:return None
                result[k]=df
            return result
        except (OSError,ValueError,KeyError,TypeError):return None

    def save_tables(self,name,key,tables,outputs):
        directory=self.root/'processed_cache'/name;directory.mkdir(parents=True,exist_ok=True)
        meta={}
        from fetch.control.tools import atomic_bytes
        for k,df in tables.items():
            p=directory/(k+'.json')
            atomic_bytes(p,df.to_json(orient='table',index=False,date_format='iso').encode())
            meta[k]=dict(file=p.name,sha256=checksum(p),columns=list(df.columns),rows=len(df),
                         object_columns=[col for col in df if df[col].dtype == object],
                         none_cells={col:[i for i,x in enumerate(df[col]) if x is None] for col in df if df[col].dtype == object})
        atomic_json(directory/'manifest.json',dict(key=key,complete=True,source=name,schema_version=SCHEMA_VERSION,
                    algorithm_version=ALGORITHM_VERSION,created_at=utcnow(),tables=meta,
                    outputs=[dict(path=str(p),sha256=checksum(p)) for p in outputs]))


def row_status(name,tables):
    """科学标签详细状态保持原值，另加统一运行状态供恢复和检查使用。"""
    import pandas as pd
    if name=='semantic_scholar':
        df=tables['s2_citation_3y_df'];out={}
        for r in df.to_dict('records'):
            statuses=[r['s2_c3_status'],r['s2_ic3_status']]
            out[r['arxiv_id']]=('REQUEST_ERROR' if 'REQUEST_ERROR' in statuses or any('ERROR' in s for s in statuses) else
                'ZERO_CONFIRMED' if all(s=='ZERO_CONFIRMED_BY_CUMULATIVE_COUNT' for s in statuses) else
                'COMPLETE' if all(s=='COMPLETE' for s in statuses) else
                'NO_MATCH' if all(s=='NO_S2_MATCH' for s in statuses) else 'PARTIAL')
        return out
    if name=='openreview':
        df=tables['openreview_audit_df']
        mapping={'RATING_FOUND':'COMPLETE','RATING_FOUND_PARTIAL':'PARTIAL','FETCH_ERROR':'REQUEST_ERROR',
                 'NO_MATCH_FOUND':'NO_MATCH','NO_INDEXED_PUBLIC_RATING':'NO_MATCH'}
        return {r['arxiv_id']:mapping.get(r['collection_status'],'PARTIAL') for r in df.to_dict('records')}
    if name=='dblp':
        return {str(a):'COMPLETE' for a in tables['dblp_author_features_df'].arxiv_id}
    return {}


from dataclasses import dataclass, field

@dataclass
class SourceResult:
    """各来源统一返回：summary 可直接序列化，tables 只在进程内传递。"""
    source: str
    status: str
    input_papers: int
    processed_papers: int
    failed_papers: int
    cache_hits: int
    network_requests: int
    output_files: list[str]
    errors: dict
    tables: dict = field(repr=False)

    def summary(self) -> dict:
        return {k:v for k,v in vars(self).items() if k != 'tables'}


def source_result(source, tables, http, input_papers=0):
    statuses=row_status(source,tables)
    errors={a:s for a,s in statuses.items() if s in ('REQUEST_ERROR','PARTIAL','PENDING')}
    root=http.config.paths[2]
    folder={'dblp':root/'dblp_author_features_v3_1_1',
            'semantic_scholar':root/'semantic_scholar_public',
            'openreview':root/'openreview_ratings'}.get(source,root)
    return SourceResult(source,'PARTIAL' if errors else 'SUCCESS',input_papers,
                        len(statuses) if statuses else len(next(iter(tables.values()))),
                        len(errors),http.cache.hits,http.network_requests,
                        [str(p) for p in folder.glob('*.csv')],errors,tables)
