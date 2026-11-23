"""显式调用的 Notebook 迁移模块。输入输出与主键见 README 和 DATA_DICTIONARY。"""
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from fetch.control.config import Config
    from fetch.control.tools import HTTPClient
    import pandas as pd
from fetch.control.tools import atomic_csv, log_preview, log_print as print

def collect(config: Config, http: HTTPClient, papers_df=None, seed_df=None) -> dict[str, pd.DataFrame]:
    ROOT, RAW, PROCESSED = config.paths
    # 第 6 节：公开论文评分。输入 papers_df（arxiv_id, title_v1）。
    # 单元格可以直接替换；不覆盖 ROOT / RAW / PROCESSED 等上游配置。
    import concurrent.futures as cf
    import hashlib, json, re, threading, time, unicodedata
    import urllib.request, urllib.parse, urllib.error
    from pathlib import Path
    import pandas as pd

    OR_HOSTS = ('https://api2.openreview.net', 'https://api.openreview.net')
    # 仅这些字段作为论文总体分数；confidence/soundness 不混入总体分数。
    OR_SCORE_FIELDS = ('rating', 'overall_rating', 'overall_score',
                       'overall_recommendation', 'recommendation',
                       'recommendation_rating', 'score')

    def or_value(content, key):
        v = (content or {}).get(key)
        return v.get('value') if isinstance(v, dict) else v

    def or_title(s):
        s = unicodedata.normalize('NFKC', str(s or '')).casefold()
        return ''.join(c for c in s if c.isalnum())

    def or_invitations(n):
        inv = n.get('invitations') or n.get('invitation') or []
        return [inv] if isinstance(inv, str) else inv

    def or_submission(n):
        inv = ' '.join(or_invitations(n)).lower()
        return (bool(n.get('id')) and not n.get('replyto')
                and n.get('forum', n['id']) == n['id']
                and 'submission' in inv
                and not any(x in inv for x in ('dblp.org', 'public_article')))

    def or_review(n):
        # 只检查 invitation 的动作名，避免路径中的 Reviewer 误判。
        actions = [x.rsplit('/', 1)[-1].lower().replace('-', '_')
                   for x in or_invitations(n)]
        return any(x in ('review', 'official_review', 'full_review', 'paper_review')
                   for x in actions) and not n.get('ddate')

    def or_number(v):
        # 支持 6: Weak Accept / 6 / 6.5；不把任意文本内的数字当成分数。
        if isinstance(v, bool): return None
        m = re.match(r'^\s*([+-]?\d+(?:\.\d+)?)\s*(?::|/|\(|$)', str(v))
        return float(m.group(1)) if m else None

    def or_arxiv_ids(content):
        # 只看元数据链接，不能把摘要中引用的 arXiv ID 当成本篇 ID。
        text = ' '.join(str(or_value(content, k) or '') for k in
                        ('html', 'pdf', 'arxiv', 'arxiv_id', 'arxiv_url', 'url'))
        return set(re.findall(r'(?:arxiv\.org/(?:abs|pdf)/|arxiv:)(\d{4}\.\d{4,5}|[a-z.-]+/\d{7})', text, re.I))

    class ORClient:
        def __init__(self, cache_dir, config):
            self.cfg = config
            self.cache = Path(cache_dir); self.cache.mkdir(parents=True, exist_ok=True)
            self.lock = threading.Lock(); self.next_at = {}; self.blocked = {}
            self.stats = dict(requests=0, cache_hits=0, retries=0)

        def request(self, host, endpoint, payload, method):
            data = http.openreview(host, endpoint, payload, method, self.cache)
            self.stats.update(requests=http.network_requests, cache_hits=http.cache.hits)
            return data


        def pages(self, host, endpoint, payload, method='GET'):
            notes = {}; size = self.cfg['page_size']; offset = 0
            for _ in range(self.cfg['max_pages']):
                data = self.request(host, endpoint, dict(payload, limit=size, offset=offset), method)
                page = data['notes']; before = len(notes)
                for n in page:
                    if n.get('id'): notes[n['id']] = n
                count = data.get('count')
                offset += len(page)
                if page and len(notes)==before: raise RuntimeError('PAGINATION_STALLED')
                if not page or (isinstance(count,int) and offset >= count) or (count is None and len(page)<size):
                    if isinstance(count,int) and len(notes)<count:
                        raise RuntimeError('INCOMPLETE_PAGINATION: count exceeds unique notes')
                    return list(notes.values())
            raise RuntimeError('PAGE_LIMIT_REACHED: increase max_pages; result is incomplete')

        def search(self, host, titles):
            return self.pages(host, '/notes/search', {'content':{'title':{
                'terms':titles, 'matchMethod':'match_phrase'}}, 'source':'forum', 'count':True}, 'POST')


    def collect_openreview(papers_df, root, config=None):
        cfg = dict(config or {}); start = time.perf_counter()
        if not {'arxiv_id','title_v1'} <= set(papers_df.columns):
            raise ValueError('papers_df must contain arxiv_id and title_v1')
        papers = papers_df.drop_duplicates('arxiv_id').copy()
        if papers.arxiv_id.isna().any(): raise ValueError('arxiv_id contains missing values')
        papers['arxiv_id'] = papers.arxiv_id.astype(str)
        records = papers.to_dict('records')
        out = Path(root)/'data/processed/openreview_ratings'; out.mkdir(parents=True, exist_ok=True)
        client = ORClient(Path(root)/'data/raw/openreview/rating_cache_v3_search', cfg)
        errors=[]; search_rows=[]; candidates={}; failed_ids=set()
        titles = sorted(set(str(r['title_v1']).strip() for r in records if pd.notna(r['title_v1']) and str(r['title_v1']).strip()))
        title_ids={}
        for r in records:
            if pd.notna(r['title_v1']): title_ids.setdefault(or_title(r['title_v1']),set()).add(r['arxiv_id'])
        jobs=[(host,titles[i:i+cfg['batch_size']]) for host in OR_HOSTS for i in range(0,len(titles),cfg['batch_size'])]
        print(f'OpenReview: {len(records)} papers, {len(jobs)} initial search batches, {cfg["workers"]} workers', flush=True)
        def search_job(job): return client.search(*job)
        with cf.ThreadPoolExecutor(cfg['workers']) as pool:
            futures={pool.submit(search_job,j):j for j in jobs}
            completed_search_jobs=0
            for f in cf.as_completed(futures):
                host,batch=futures[f]
                try:
                    notes=f.result()
                    for n in notes: candidates[(host,n['id'])]=n
                    search_rows.append(dict(host=host,titles=batch,status='OK',notes=len(notes)))
                except Exception as e:
                    for t in batch: failed_ids.update(title_ids.get(or_title(t),set()))
                    errors.append(dict(stage='search',host=host,inputs=batch,error=str(e)))
                    search_rows.append(dict(host=host,titles=batch,status='ERROR',error=str(e)))
                completed_search_jobs += 1
                if completed_search_jobs % 25 == 0 or completed_search_jobs == len(futures):
                    print(f"OpenReview title search: {completed_search_jobs}/{len(futures)} batches; "
                          f"candidate notes={len(candidates)}; errors={len(errors)}", flush=True)
        # 从 DBLP/公开文章的显式 arXiv 链接发现改名后的标题，再批量搜索。
        aids={re.sub(r'v\d+$','',r['arxiv_id']):r['arxiv_id'] for r in records}
        aliases={k:set(v) for k,v in title_ids.items()}; extra=set()
        for n in list(candidates.values()):
            c=n.get('content',{}); t=or_value(c,'title')
            linked={aids[a] for a in or_arxiv_ids(c) if a in aids}
            if linked and t:
                aliases.setdefault(or_title(t),set()).update(linked)
                if t not in titles: extra.add(t)
        if extra:
            extra=sorted(extra)
            with cf.ThreadPoolExecutor(cfg['workers']) as pool:
                fj={pool.submit(client.search,h,extra[i:i+cfg['batch_size']]):(h,extra[i:i+cfg['batch_size']])
                    for h in OR_HOSTS for i in range(0,len(extra),cfg['batch_size'])}
                completed_alias_jobs=0
                for f in cf.as_completed(fj):
                    host,batch=fj[f]
                    try:
                        for n in f.result(): candidates[(host,n['id'])]=n
                    except Exception as e:
                        for t in batch: failed_ids.update(aliases.get(or_title(t),set()))
                        errors.append(dict(stage='alias_search',host=host,inputs=batch,error=str(e)))
                    completed_alias_jobs += 1
                    if completed_alias_jobs % 25 == 0 or completed_alias_jobs == len(fj):
                        print(f"OpenReview alias search: {completed_alias_jobs}/{len(fj)} batches; "
                              f"candidate notes={len(candidates)}", flush=True)
        submissions=[]; ambiguous=set()
        for (host,nid),n in candidates.items():
            if not or_submission(n): continue
            c=n.get('content',{}); t=or_value(c,'title'); linked={aids[a] for a in or_arxiv_ids(c) if a in aids}
            matches=linked or aliases.get(or_title(t),set())
            if len(matches)>1:
                ambiguous.update(matches); continue
            for aid in matches:
                submissions.append(dict(arxiv_id=aid,host=host,forum_id=n.get('forum') or nid,
                    submission_note_id=nid,title=t,venue=or_value(c,'venue'),venueid=or_value(c,'venueid'),
                    match_method='ARXIV_LINK' if linked else 'NORMALIZED_EXACT_TITLE',
                    url='https://openreview.net/forum?id='+nid,
                    number=n.get('number'), domain=n.get('domain') or or_invitations(n)[0].split('/-/')[0]))
        # 同一个 forum 在多个 API 可出现；保留来源，输出评分时按 note ID 去重。
        forum_jobs=sorted(set((s['host'],s['forum_id']) for s in submissions))
        forum_data={}; forum_errors={}
        print(f'Matched {len(set(s["arxiv_id"] for s in submissions))} papers; fetching {len(forum_jobs)} API/forum pairs', flush=True)
        # 官方搜索端点支持 query；先按论坛批量搜索，再严格检查每条 Note 的 forum。
        # 这是公开搜索索引，不能宣称与数据库的全部可见 Notes 完全等价。
        batches=[]
        for host in ('https://api.openreview.net',):
            ids=[fid for h,fid in forum_jobs if h==host]
            batches.extend((host,ids[i:i+cfg['batch_size']]) for i in range(0,len(ids),cfg['batch_size']))
        def fetch_forums(host, ids):
            if any(not re.fullmatch(r'[A-Za-z0-9_-]+', x) for x in ids):
                raise ValueError('Invalid forum ID')
            query='forum:('+' OR '.join(ids)+')'
            data=client.pages(host,'/notes/search',{'query':query,'source':'all','count':True},'POST')
            if any(n.get('forum') not in ids for n in data):
                raise RuntimeError('FORUM_SEARCH_FILTER_MISMATCH')
            return data
        with cf.ThreadPoolExecutor(cfg['workers']) as pool:
            fj={pool.submit(fetch_forums,h,ids):(h,ids) for h,ids in batches}
            completed_forum_jobs=0
            for f in cf.as_completed(fj):
                host,ids=fj[f]
                try:
                    data=f.result()
                    for fid in ids: forum_data[(host,fid)]=[n for n in data if n.get('forum')==fid]
                except Exception as e:
                    for fid in ids: forum_errors[(host,fid)]=str(e)
                    errors.append(dict(stage='forum_search',host=host,forum_ids=ids,error=str(e)))
                completed_forum_jobs += 1
                if completed_forum_jobs % 10 == 0 or completed_forum_jobs == len(fj):
                    print(f"OpenReview forum retrieval: {completed_forum_jobs}/{len(fj)} batches; "
                          f"errors={len(errors)}", flush=True)
        # API v2 实测不支持 POST query。按投稿号构造标准 Review invitation，
        # 用常见英文词的 OR 检索正式评审；记录范围，绝不声称遍历了全部 Notes。
        # 此路径可能漏掉不含这些词/使用自定义 invitation 的评审。
        v2jobs={}
        for sub in submissions:
            if sub['host']!='https://api2.openreview.net': continue
            if sub.get('number') is None:
                key=(sub['host'],sub['forum_id']);forum_errors[key]='MISSING_SUBMISSION_NUMBER';continue
            prefix=sub['domain']+'/Submission'+str(sub['number'])+'/-/'
            for action in ('Official_Review','Review'):
                v2jobs[(sub['host'],sub['forum_id'],prefix+action)]=None
        def fetch_v2(job):
            host,fid,inv=job
            data=client.pages(host,'/notes/search',{
                'terms':['the','this','paper','review','a','an','and','is'],
                'type':'terms','invitation':inv,'source':'reply','count':True},'POST')
            if any(n.get('forum')!=fid or inv not in or_invitations(n) for n in data):
                raise RuntimeError('V2_REVIEW_FILTER_MISMATCH')
            return data
        with cf.ThreadPoolExecutor(cfg['workers']) as pool:
            fj={pool.submit(fetch_v2,job):job for job in v2jobs}
            for f in cf.as_completed(fj):
                host,fid,inv=fj[f];key=(host,fid)
                try: forum_data.setdefault(key,[]).extend(f.result())
                except Exception as e:
                    forum_errors[key]=str(e)
                    errors.append(dict(stage='v2_review_search',host=host,forum_id=fid,invitation=inv,error=str(e)))
        reviews=[]; ratings=[]; missing=[]
        seen=set()
        for s in submissions:
            for n in forum_data.get((s['host'],s['forum_id']),[]):
                if not or_review(n): continue
                key=(s['arxiv_id'],n['id'])
                if key in seen: continue
                seen.add(key); c=n.get('content',{})
                base=dict(arxiv_id=s['arxiv_id'],forum_id=s['forum_id'],venue=s['venue'],
                    review_note_id=n['id'],reviewer_id=';'.join(n.get('signatures',[])),
                    review_invitation=';'.join(or_invitations(n)),source_api=s['host'],
                    retrieval_scope='V2_KEYWORD_REVIEW_SEARCH' if 'api2.' in s['host'] else 'V1_FORUM_SEARCH',
                    review_url='https://openreview.net/forum?id='+s['forum_id']+'&noteId='+n['id'])
                reviews.append(dict(base,content_json=json.dumps(c,ensure_ascii=False)))
                found=False
                for field in OR_SCORE_FIELDS:
                    v=or_value(c,field)
                    if v is None: continue
                    found=True
                    ratings.append(dict(base,rating_field=field,rating_score=or_number(v),rating_raw=json.dumps(v,ensure_ascii=False),
                        confidence_raw=json.dumps(or_value(c,'confidence'),ensure_ascii=False)))
                if not found: missing.append(dict(base,reason='NO_KNOWN_OVERALL_SCORE_FIELD'))
        # 每个输入保留一行，失败绝不写成“没有评分”。
        audit=[]
        for r in records:
            aid=r['arxiv_id']; ss=[s for s in submissions if s['arxiv_id']==aid]
            rr=[v for v in ratings if v['arxiv_id']==aid]
            broken=aid in failed_ids or any((s['host'],s['forum_id']) in forum_errors for s in ss)
            status=('RATING_FOUND_PARTIAL' if broken else 'RATING_FOUND') if rr else (
                'FETCH_ERROR' if broken else 'AMBIGUOUS_MATCH' if aid in ambiguous else
                'NO_INDEXED_PUBLIC_RATING' if ss else 'NO_MATCH_FOUND')
            if pd.isna(r['title_v1']) or not str(r['title_v1']).strip(): status='INVALID_TITLE'
            audit.append(dict(arxiv_id=aid,title_v1=r['title_v1'],collection_status=status,
                retrieval_scope=';'.join(sorted(set('V2_KEYWORD_REVIEW_SEARCH' if 'api2.' in s['host'] else 'V1_FORUM_SEARCH' for s in ss))),
                forum_count=len(set(s['forum_id'] for s in ss)),rating_count=len(rr),
                reviewer_count=len(set(v['review_note_id'] for v in rr))))
        frames={
            'submissions':pd.DataFrame(submissions,columns=['arxiv_id','host','forum_id','submission_note_id','title','venue','venueid','match_method','url']),
            'reviews':pd.DataFrame(reviews,columns=['arxiv_id','forum_id','venue','review_note_id','reviewer_id','review_invitation','source_api','retrieval_scope','review_url','content_json']),
            'reviewer_ratings':pd.DataFrame(ratings,columns=['arxiv_id','forum_id','venue','review_note_id','reviewer_id','review_invitation','source_api','retrieval_scope','review_url','rating_field','rating_score','rating_raw','confidence_raw']),
            'missing_ratings':pd.DataFrame(missing,columns=['arxiv_id','forum_id','review_note_id','reason']),
            'rating_audit':pd.DataFrame(audit,columns=['arxiv_id','title_v1','collection_status','retrieval_scope','forum_count','rating_count','reviewer_count']),
            'search_audit':pd.DataFrame(search_rows),
        }
        for name,df in frames.items(): atomic_csv(df, out/f'openreview_{name}.csv',index=False)
        # 原始 Notes 及错误都保留。相同论文跨 venue 的分数不能直接求平均。
        with (out/'openreview_forum_notes.jsonl').open('w',encoding='utf-8') as f:
            for (host,fid),notes in forum_data.items():
                for n in notes: f.write(json.dumps(dict(host=host,forum_id=fid,note=n),ensure_ascii=False)+'\n')
        (out/'openreview_errors.json').write_text(json.dumps(errors,ensure_ascii=False,indent=2),encoding='utf-8')
        stats=dict(client.stats,elapsed_seconds=round(time.perf_counter()-start,2),
                   input_papers=len(records),candidate_notes=len(candidates),rating_rows=len(ratings),errors=len(errors))
        (out/'openreview_run_stats.json').write_text(json.dumps(stats,indent=2))
        print(stats,flush=True)
        print(frames['rating_audit'].collection_status.value_counts().to_string(),flush=True)
        if client.blocked: print('接口要求认证/验证：请按 OpenReview 官方流程处理后重跑；本次未成功获取的内容不是零分。',flush=True)
        return frames,stats

    OPENREVIEW_SCORE_PRIORITY = (
        'rating', 'overall_rating', 'overall_score',
        'overall_recommendation', 'recommendation',
        'recommendation_rating', 'score'
    )


    def aggregate_openreview_paper_ratings(papers, raw_ratings, rating_audit=None):
        """One row per arXiv paper; preserves non-rated papers as NaN.

        Safeguards:
        1. One numeric overall score per review Note, preferring canonical fields.
        2. Deduplicates the same Note returned from multiple OpenReview APIs.
        3. Does NOT average across different scored forums/venues.
        4. Does NOT combine different score fields within one forum without review.
        """
        for name in ('arxiv_id', 'title_v1'):
            if name not in papers.columns:
                raise ValueError(f'papers_df missing: {name}')
        base = papers[['arxiv_id', 'title_v1']].copy()
        if base['arxiv_id'].isna().any():
            raise ValueError('papers_df contains missing arxiv_id')
        base['arxiv_id'] = base['arxiv_id'].astype(str).str.strip()
        if base['arxiv_id'].duplicated().any():
            raise ValueError('papers_df contains duplicate arxiv_id; resolve duplicates first')

        if rating_audit is not None:
            audit_cols = ['arxiv_id', 'collection_status', 'forum_count',
                          'rating_count', 'reviewer_count']
            a = rating_audit[[c for c in audit_cols if c in rating_audit.columns]].copy()
            a['arxiv_id'] = a['arxiv_id'].astype(str).str.strip()
            if a['arxiv_id'].duplicated().any():
                raise ValueError('rating_audit contains multiple rows per arxiv_id')
            a = a.rename(columns={'collection_status': 'openreview_collection_status',
                                  'forum_count': 'openreview_matched_forum_count',
                                  'rating_count': 'openreview_raw_rating_rows',
                                  'reviewer_count': 'openreview_raw_review_notes'})
            base = base.merge(a, on='arxiv_id', how='left', validate='one_to_one')

        output_cols = ['openreview_mean_rating', 'openreview_n_scored_reviews',
                       'openreview_scored_forum_count', 'openreview_scored_forum_id',
                       'openreview_score_field', 'openreview_score_status']
        base['openreview_mean_rating'] = float('nan')
        base['openreview_n_scored_reviews'] = 0
        base['openreview_scored_forum_count'] = 0
        base['openreview_scored_forum_id'] = pd.NA
        base['openreview_score_field'] = pd.NA
        base['openreview_score_status'] = 'NO_NUMERIC_PUBLIC_RATING'

        mandatory = {'arxiv_id', 'forum_id', 'review_note_id', 'rating_field', 'rating_score'}
        missing = mandatory - set(raw_ratings.columns)
        if missing:
            raise ValueError(f'reviewer_ratings missing columns: {sorted(missing)}')
        if raw_ratings.empty:
            return base

        scores = raw_ratings[list(mandatory)].copy()
        scores['arxiv_id'] = scores['arxiv_id'].astype(str).str.strip()
        scores['rating_score'] = pd.to_numeric(scores['rating_score'], errors='coerce')
        scores = scores.dropna(subset=['arxiv_id', 'forum_id', 'review_note_id', 'rating_score'])
        scores = scores[scores['arxiv_id'].isin(set(base['arxiv_id']))]
        scores = scores[scores['forum_id'].astype(str).str.len().gt(0)]
        scores = scores[scores['review_note_id'].astype(str).str.len().gt(0)]
        priority = {field: i for i, field in enumerate(OPENREVIEW_SCORE_PRIORITY)}
        scores['_field_rank'] = scores['rating_field'].map(priority)
        scores = scores.dropna(subset=['_field_rank'])
        if scores.empty:
            return base

        # Select one score field per Review Note; detect conflicting duplicates.
        one_score_per_note = []
        conflicting_papers = set()
        for (aid, forum_id, note_id), group in scores.groupby(
                ['arxiv_id', 'forum_id', 'review_note_id'], sort=False):
            best_rank = group['_field_rank'].min()
            top = group[group['_field_rank'].eq(best_rank)]
            if top['rating_score'].nunique() > 1:
                conflicting_papers.add(aid)
                continue  # Same note / same selected field but inconsistent score.
            one_score_per_note.append(top.iloc[0])

        if not one_score_per_note:
            base.loc[base['arxiv_id'].isin(conflicting_papers),
                     'openreview_score_status'] = 'CONFLICTING_REVIEW_RECORDS'
            return base

        selected = pd.DataFrame(one_score_per_note)
        base.loc[base['arxiv_id'].isin(conflicting_papers),
                 'openreview_score_status'] = 'CONFLICTING_REVIEW_RECORDS'
        for aid, group in selected.groupby('arxiv_id', sort=False):
            mask = base['arxiv_id'].eq(aid)
            forums = group['forum_id'].dropna().unique()
            base.loc[mask, 'openreview_n_scored_reviews'] = len(group)
            base.loc[mask, 'openreview_scored_forum_count'] = len(forums)

            # Prefer an unresolved status to silently averaging uncertain records.
            if aid in conflicting_papers:
                base.loc[mask, 'openreview_score_status'] = 'CONFLICTING_REVIEW_RECORDS'
            elif len(forums) > 1:
                base.loc[mask, 'openreview_score_status'] = 'MULTIPLE_SCORED_FORUMS'
            elif group['rating_field'].nunique() != 1:
                base.loc[mask, 'openreview_scored_forum_id'] = forums[0]
                base.loc[mask, 'openreview_score_status'] = 'MIXED_SCORE_FIELDS'
            else:
                base.loc[mask, 'openreview_scored_forum_id'] = forums[0]
                base.loc[mask, 'openreview_score_field'] = group['rating_field'].iloc[0]
                base.loc[mask, 'openreview_mean_rating'] = group['rating_score'].mean()
                base.loc[mask, 'openreview_score_status'] = 'MEAN_AVAILABLE'

        assert len(base) == papers['arxiv_id'].nunique()
        assert base['arxiv_id'].is_unique
        return base



    frames, stats = collect_openreview(papers_df, ROOT, config.openreview_options)
    ratings = aggregate_openreview_paper_ratings(papers_df, frames["reviewer_ratings"], frames["rating_audit"])
    # 部分评审仍保留观察均值，显式标记未完成，避免当作完整标签。
    partial = ratings["openreview_collection_status"].isin(["FETCH_ERROR", "RATING_FOUND_PARTIAL"])
    ratings.loc[partial, "openreview_score_status"] = "PARTIAL"
    atomic_csv(ratings, PROCESSED / "openreview_ratings/openreview_paper_ratings.csv", index=False)
    return {"openreview_paper_ratings_df": ratings, "openreview_audit_df": frames["rating_audit"]}


def run(config: Config, http: HTTPClient, papers_df=None, seed_df=None):
    """返回统一 SourceResult；异常由中控记录，绝不构造成功或零值。"""
    from fetch.control.state import source_result
    tables = collect(config, http, papers_df=papers_df, seed_df=seed_df)
    count = len(papers_df) if papers_df is not None else len(seed_df) if seed_df is not None else 0
    return source_result('openreview', tables, http, count)
