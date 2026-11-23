"""唯一中控入口：python -m fetch.control.fetch；也支持 python fetch/control/fetch.py。"""
from pathlib import Path
import sys
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
import importlib
import json
import logging
import os
from fetch.control.config import Config




def configure_logging(root,level):
    logger=logging.getLogger('fetch');logger.setLevel(logging.DEBUG);logger.handlers.clear()
    f=logging.FileHandler(root/'logs/pipeline.log',encoding='utf-8');f.setLevel(logging.DEBUG)
    class BriefFormatter(logging.Formatter):
        def format(self,record):
            import copy
            record=copy.copy(record);record.exc_info=None;record.exc_text=None
            return super().format(record)
    console=logging.StreamHandler();console.setLevel(level)
    fmt=logging.Formatter('%(asctime)s %(levelname)s %(message)s');f.setFormatter(fmt);console.setFormatter(BriefFormatter(fmt._fmt))
    logger.addHandler(f);logger.addHandler(console)
    return logger


def preflight(config,network=False):
    import platform
    import importlib.metadata
    if sys.version_info < (3,10):raise RuntimeError('需要 Python >= 3.10')
    deps={n:importlib.metadata.version(n) for n in ['requests','pandas','numpy','tqdm','lxml']}
    config.validate();config.prepare_storage()
    from fetch.control.tools import HTTPClient,atomic_json
    result=dict(python=platform.python_version(),dependencies=deps,storage=str(config.paths[0]),network={})
    if network:
        urls={'arxiv':'https://export.arxiv.org/api/query?id_list=2010.11929v1',
              'dblp':'https://dblp.org/xml/dblp.dtd',
              'semantic_scholar':'https://api.semanticscholar.org/graph/v1/paper/ARXIV:2010.11929?fields=title',
              'openreview':'https://api2.openreview.net/notes?id=YicbFdNTTy'}
        import requests
        for source in ['arxiv']+config.sources:
            try:
                headers={'User-Agent':config.user_agent}
                r=requests.get(urls[source],timeout=min(config.request_timeout,15),headers=headers)
                result['network'][source]={'status_code':r.status_code,'reachable':r.status_code<400};r.close()
            except requests.RequestException as exc:result['network'][source]={'reachable':False,'error_type':type(exc).__name__}
    atomic_json(config.paths[0]/'state/preflight.json',result)
    return result


def run(config,client_factory=None):
    import pandas as pd
    from fetch.control.tools import HTTPClient,atomic_json,atomic_csv,utcnow
    from fetch.control.state import State, row_status
    from fetch.control import combine,check
    preflight(config)
    root,raw,processed=config.paths
    logger=configure_logging(root,config.log_level)
    # OS 文件锁自动随进程退出释放，不留下阻挡恢复的过期锁文件。
    lock=(root/'state/pipeline.lock').open('a+b')
    try:
        if os.name=='nt':
            import msvcrt
            lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError:
        lock.close();raise RuntimeError('同一 data_root 已有 Pipeline 正在运行')
    state=State(config);all_tables={};paper_status={};fatal=False
    factory=client_factory or HTTPClient
    modules={'arxiv_seed':'fetch.arxiv.paper_id','arxiv_metadata':'fetch.arxiv.paper_metadata',
             'dblp':'fetch.dblp.author_info','semantic_scholar':'fetch.semantic_scholar.citation',
             'openreview':'fetch.openreview.review_score'}
    try:
        for name,path in modules.items():
            if name in ('dblp','semantic_scholar','openreview') and name not in config.sources:
                state.step(name,dict(source=name,status='NOT_COLLECTED'));continue
            if fatal:
                state.step(name,dict(source=name,status='PENDING'));continue
            source='arxiv' if name.startswith('arxiv') else name
            client=factory(config,source)
            inputs={}
            if name=='arxiv_metadata':inputs={'seed_df':all_tables['seed_df']}
            elif name in config.sources:inputs={'papers_df':all_tables['papers_df']}
            key=state.key(name,inputs)
            state.step(name,dict(source=source,status='RUNNING',started_at=utcnow()))
            logger.info('开始 %s',name)
            try:
                tables=state.load_tables(name,key)
                cached=tables is not None
                if tables is None:
                    if name=='arxiv_seed' and config.seed_csv:
                        df=pd.read_csv(config.seed_csv,dtype={'arxiv_id':str})
                        if not {'arxiv_id','seed_query_year'}<=set(df) or df.empty:raise ValueError('seed_csv 必须包含 arxiv_id、seed_query_year 且非空')
                        if df.arxiv_id.isna().any() or not df.arxiv_id.is_unique:raise ValueError('seed_csv ID 为空或重复')
                        df['seed_query_year']=pd.to_numeric(df.seed_query_year,errors='raise').astype(int)
                        if not df.seed_query_year.between(config.start_year,config.end_year).all():raise ValueError('seed_csv 年份在配置范围之外')
                        tables={'seed_df':df}
                    else:tables=importlib.import_module(path).run(config,client,**inputs).tables
                if name=='arxiv_metadata':
                    p=tables['papers_df']
                    if p.empty or p.arxiv_id.isna().any() or not p.arxiv_id.is_unique:raise ValueError('arXiv v1 主表为空或主键无效')
                    if pd.to_datetime(p.submission_date_v1,errors='coerce',utc=True).isna().any():raise ValueError('arXiv 首次提交时间缺失')
                rows=row_status(name,tables);paper_status[name]=rows
                failed={k:v for k,v in rows.items() if v in ('PARTIAL','REQUEST_ERROR','PENDING')}
                status='PARTIAL' if failed else 'SUCCESS'
                if failed:
                    for aid,reason in failed.items():state.error(name,RuntimeError(reason),aid,client.retries)
                all_tables.update(tables)
                # 所有来源输出完整性都纳入 processed manifest，删除输出会触发重建。
                folder={'dblp':processed/'dblp_author_features_v3_1_1','semantic_scholar':processed/'semantic_scholar_public','openreview':processed/'openreview_ratings'}.get(name)
                outputs=list(folder.glob('*.csv')) if folder else list(processed.glob('*.csv')) if name.startswith('arxiv') else []
                if not cached and not failed:state.save_tables(name,key,tables,outputs)
                summary=dict(source=source,status=status,input_papers=len(next(iter(inputs.values()))) if inputs else 0,
                             processed_papers=len(rows) if rows else len(next(iter(tables.values()))),failed_papers=len(failed),
                             cache_hits=client.cache.hits+(1 if cached else 0),network_requests=client.network_requests,
                             output_files=list(map(str,outputs)),errors=failed)
                state.step(name,summary)
                atomic_json(root/'state/tasks'/f'{name}.json',dict(run_id=state.run_id,papers=rows,pending=failed))
                if not failed:state.recovered(name)
                logger.info('%s %s：处理 %s，失败/未完成 %s，网络请求 %s',name,status,summary['processed_papers'],len(failed),client.network_requests)
            except Exception as exc:
                logger.exception('%s 失败（其他独立来源会继续）',name)
                state.error(name,exc,retry_count=client.retries)
                state.step(name,dict(source=source,status='REQUEST_ERROR',errors=[str(exc)],network_requests=client.network_requests))
                ids=list(inputs.get('papers_df',pd.DataFrame()).get('arxiv_id',[]))
                paper_status[name]={str(a):'REQUEST_ERROR' for a in ids}
                atomic_json(root/'state/tasks'/f'{name}.json',dict(run_id=state.run_id,pending=paper_status[name],stage_pending=True))
                if name.startswith('arxiv'):fatal=True
            finally:client.close()
        if not fatal:
            try:
                state.step('combine',dict(source='combine',status='RUNNING'))
                master=combine.collect(config,None,papers_df=all_tables)['master_df']
                for source in ('dblp','semantic_scholar','openreview'):
                    default='NOT_COLLECTED' if source not in config.sources else 'REQUEST_ERROR'
                    master[source+'_status']=master.arxiv_id.map(paper_status.get(source,{})).fillna(default)
                # 来源整体失败/未启用时，不能让原合并逻辑的无匹配兜底掩盖失败。
                if 's2_all_df' not in all_tables:
                    status='NOT_COLLECTED' if 'semantic_scholar' not in config.sources else 'REQUEST_ERROR'
                    master['s2_match_status']=status
                    master['s2_c3_status']=status;master['s2_ic3_status']=status
                    master['s2_citation_count_3y']=float('nan');master['s2_influential_citation_count_3y']=float('nan')
                    master['s2_c3_label_valid']=False;master['s2_ic3_label_valid']=False
                if 'openreview_paper_ratings_df' not in all_tables:
                    master['openreview_score_status']='NOT_COLLECTED' if 'openreview' not in config.sources else 'REQUEST_ERROR'
                    master['openreview_mean_rating']=float('nan')
                if 'dblp_author_features_df' not in all_tables:
                    master['dblp_match_status']='NOT_COLLECTED' if 'dblp' not in config.sources else 'REQUEST_ERROR'
                if len(master)!=all_tables['papers_df'].arxiv_id.nunique():raise ValueError('主表行数与 arXiv 不一致')
                atomic_csv(master,processed/'papers_master_raw_stage.csv',index=False)
                from fetch.control.tools import atomic_bytes
                atomic_bytes(processed/'papers_master_raw_stage.jsonl',master.to_json(orient='records',lines=True,force_ascii=False).encode())
                metadata_path=processed/'papers_master_metadata.json'
                metadata=json.loads(metadata_path.read_text())
                metadata.update(run_id=state.run_id,column_count=len(master.columns),columns=list(master.columns),citation_window_years=config.citation_window_years)
                metadata['label_definitions']['C3']=f'Unique citing papers in [t0, t0 + {config.citation_window_years} calendar years).'
                metadata['label_definitions']['IC3']='Unique influential citing papers in the same window; incomplete labels remain missing.'
                atomic_json(metadata_path,metadata)
                state.step('combine',dict(source='combine',status='SUCCESS',rows=len(master)))
                report=check.collect(config,master,state.doc['steps'])
                state.step('check',dict(source='check',status='SUCCESS' if report['integrity_ok'] else 'PARTIAL',problems=report['problems']))
                complete=all(v['status']=='SUCCESS' for v in state.doc['steps'].values())
                state.doc.update(status='COMPLETE' if complete and report['integrity_ok'] else 'PARTIAL',master_rows=len(master),master_integrity_ok=report['integrity_ok'],
                                 all_sources_collected=all(s in config.sources for s in ('dblp','semantic_scholar','openreview')),
                                 master_complete=complete and report['integrity_ok'],
                                 all_labels_available=all(col in master and master[col].notna().all() for col in ('s2_citation_count_3y','s2_influential_citation_count_3y','openreview_mean_rating')))
            except Exception as exc:
                logger.exception('合并/检查失败');state.error('combine_or_check',exc)
                state.doc['status']='FAILED';state.step('combine_or_check',dict(status='REQUEST_ERROR',errors=[str(exc)]))
        else:
            state.doc['status']='FAILED'
            state.step('combine',dict(status='PENDING'));state.step('check',dict(status='PENDING',reason='arXiv 必要输入不完整'))
        state.doc['finished_at']=utcnow();state.save()
        atomic_json(root/'run_summary.json',state.doc)
        logger.info('最终状态 %s；摘要 %s',state.doc['status'],root/'run_summary.json')
        return state.doc
    except KeyboardInterrupt:
        state.doc['status']='INTERRUPTED';state.doc['finished_at']=utcnow();state.save()
        atomic_json(root/'run_summary.json',state.doc);raise
    finally:lock.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description='GP 论文数据采集；所有可调整参数见 control/config.py')
    parser.add_argument('--config',help='JSON 配置文件')
    parser.add_argument('--data-root',help='覆盖数据根目录')
    parser.add_argument('--preflight-only',action='store_true',help='仅检查环境、存储和服务可达性')
    parser.add_argument('--no-network',action='store_true',help='preflight 不发网络请求')
    args=parser.parse_args(argv)
    try:
        config=Config.load(args.config)
        if args.data_root:config.data_root=args.data_root
        if args.preflight_only:
            result=preflight(config,not args.no_network);print(json.dumps(result,ensure_ascii=False,indent=2))
            return 0 if all(v.get('reachable') for v in result['network'].values()) else 2
        result=run(config)
        return 0 if result['status']=='COMPLETE' else 2
    except KeyboardInterrupt:
        print('已中断。成功缓存已保存；重新执行相同命令恢复。',file=sys.stderr);return 130
    except Exception as exc:
        print(f'启动失败：{type(exc).__name__}: {exc}',file=sys.stderr);return 1

if __name__=='__main__':raise SystemExit(main())
