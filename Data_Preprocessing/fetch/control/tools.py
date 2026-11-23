"""通用网络、原子写入与分层缓存；业务匹配算法不在本文件。"""
from pathlib import Path
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import gzip
import hashlib
import json
import logging
import math
import os
import random
import re
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
import requests

SCHEMA_VERSION = 'gp-cache-1'
ALGORITHM_VERSION = 'notebook-migration-1'

def utcnow():
    return datetime.now(timezone.utc).isoformat()

def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,default=str).encode()).hexdigest()

def checksum(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()

def atomic_bytes(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(dir=path.parent,prefix=path.name+'.',suffix='.tmp')
    try:
        with os.fdopen(fd,'wb') as f:
            f.write(data);f.flush();os.fsync(f.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)

def atomic_json(path, value):
    atomic_bytes(path,json.dumps(value,ensure_ascii=False,indent=2,default=str).encode())

def atomic_csv(df,path,*args,**kwargs):
    # 引用边增量追加仅写临时文件，完成后由调用方原子替换。
    if kwargs.get('mode')=='a':return df.to_csv(path,*args,**kwargs)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(dir=path.parent,suffix='.tmp');os.close(fd)
    try:
        df.to_csv(name,*args,**kwargs);os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)

def write_jsonl(path, rows):
    atomic_bytes(path,''.join(json.dumps(r,ensure_ascii=False,default=str)+'\n' for r in rows).encode())

def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding='utf-8').splitlines() if l.strip()]

def chunks(items,n):
    for i in range(0,len(items),n):yield items[i:i+n]

def clean_arxiv_id(value):
    return re.sub(r'v\d+$','',re.sub(r'^https?://(?:export\.)?arxiv\.org/abs/','',str(value).strip())).strip('/')

def parse_arxiv_feed(data):
    root=ET.fromstring(data)
    total=root.find('{http://a9.com/-/spec/opensearch/1.1/}totalResults')
    return int(total.text) if total is not None and total.text else 0,root.findall('{http://www.w3.org/2005/Atom}entry')

def log_print(*args,**kwargs):
    logging.getLogger('fetch.detail').debug(' '.join(map(str,args)))

def log_preview(value):
    logging.getLogger('fetch.detail').debug('%s',value)

def quiet_tqdm(*args,**kwargs):
    from tqdm import tqdm
    kwargs['disable']=True
    return tqdm(*args,**kwargs)

class Cache:
    """原始字节不改变旧路径。sidecar 校验请求、版本、时间、完整性和 SHA256。"""
    def __init__(self,source,config):
        self.source=source;self.config=config;self.hits=0;self.written=set()
        self.lock=threading.RLock()

    def meta(self,path):return Path(str(path)+'.gp-meta.json')

    def record(self,path,key):
        atomic_json(self.meta(path),dict(source=self.source,request_fingerprint=fingerprint(key),
                    schema_version=SCHEMA_VERSION,algorithm_version=ALGORITHM_VERSION,
                    created_at=utcnow(),complete=True,sha256=checksum(path)))

    def read_bytes(self,path,key,validator,legacy=False):
        path=Path(path)
        with self.lock:
            if self.config.refresh(self.source) and str(path) not in self.written:return None
            try:
                data=path.read_bytes()
                if not validator(data):return None
                meta=self.meta(path)
                if meta.exists():
                    m=json.loads(meta.read_text())
                    if not (m.get('source')==self.source and m.get('request_fingerprint')==fingerprint(key) and m.get('schema_version')==SCHEMA_VERSION and m.get('algorithm_version')==ALGORITHM_VERSION and m.get('complete') is True and m.get('sha256')==hashlib.sha256(data).hexdigest()):return None
                elif legacy:
                    self.record(path,key)
                else:return None
                self.hits+=1
                return data
            except (OSError,ValueError,TypeError,ET.ParseError,EOFError,gzip.BadGzipFile):return None

    def write_bytes(self,path,key,data,validator=lambda x:True):
        if not validator(data):raise ValueError(f'{self.source}: 响应结构无效，拒绝缓存')
        with self.lock:
            atomic_bytes(path,data);self.record(path,key);self.written.add(str(path))

    def read_json(self,path,key,validator=lambda x:True,legacy=False):
        def check(b):return validator(json.loads(b))
        data=self.read_bytes(path,key,check,legacy)
        return json.loads(data) if data is not None else None

    def read_json_fast(self,path,validator=lambda x:True):
        """Fast path for deterministic request-hash JSON caches.

        The cache filename is derived from the full request fingerprint, and
        callers validate the response shape (summary caches also validate their
        embedded algorithm fingerprint). This avoids an extra Drive sidecar
        read and SHA256 pass for every tiny cached JSON file.
        """
        path=Path(path)
        with self.lock:
            if self.config.refresh(self.source) and str(path) not in self.written:
                return None
            try:
                value=json.loads(path.read_bytes())
                if not validator(value):
                    return None
                self.hits+=1
                return value
            except (OSError,ValueError,TypeError,EOFError):
                return None

    def write_json_fast(self,path,value):
        """Atomically persist a validated JSON cache without a sidecar.

        Used only where the deterministic cache filename itself binds the
        request (S2/OpenReview) or the payload contains an explicit fingerprint.
        Existing cache files and sidecars are never deleted.
        """
        path=Path(path)
        atomic_bytes(path,json.dumps(value,ensure_ascii=False).encode())
        self.written.add(str(path))

    def write_json(self,path,key,value):
        self.write_bytes(path,key,json.dumps(value,ensure_ascii=False).encode(),lambda b:json.loads(b) is not None)

class RequestFailure(RuntimeError):
    def __init__(self,source,status,attempts):
        self.source=source;self.status=status;self.attempts=attempts
        super().__init__(f'{source}: {status}; attempts={attempts}')

class DeferredRequest(RequestFailure):
    """S2 429：当前分页延后处理，不在当前论文内原地重试。"""
    def __init__(self,source,status,attempts,wait_seconds=0.0,global_pause=False):
        self.wait_seconds=float(wait_seconds)
        self.global_pause=bool(global_pause)
        super().__init__(source,status,attempts)

class HTTPClient:
    """每来源独立限速；S2 使用全局 1.1 秒节奏与 429 延迟队列。"""
    RETRYABLE_STATUS = (429, 500, 502, 503, 504)

    def __init__(self,config,source,session=None,sleep=time.sleep,clock=time.monotonic):
        self.config=config
        self.source=source
        self.session=session or requests.Session()
        self.session.headers.update({'User-Agent':config.user_agent})
        self.cache=Cache(source,config)
        self.network_requests=0
        self.retries=0
        self.sleep=sleep
        self.clock=clock
        self.next_at=0.
        self.lock=threading.Lock()
        # S2 stays anonymous and uses the original Notebook's 1.1s default.
        # Other sources retain their own configured intervals.
        self.base_interval=float(config.source_intervals[source])
        self.interval=self.base_interval
        self.logger=logging.getLogger('fetch')
        self.detail_logger=logging.getLogger('fetch.detail')
        # S2 的 429 采用延迟队列：前两次孤立限流先跳过当前论文；
        # 连续限流达到阈值后，触发全局冷却并优先探测待重试队列。
        self.s2_consecutive_429=0
        self.s2_pause_requested=False
        self.s2_pause_level=0
        self.s2_deferred_total=0

    def close(self):
        self.session.close()

    @staticmethod
    def parse_retry_after(value):
        """返回有效的正 Retry-After 秒数；缺失、无效、零或过期时返回 None。"""
        if value is None:
            return None
        raw=str(value).strip()
        if not raw:
            return None
        try:
            seconds=float(raw)
            return seconds if math.isfinite(seconds) and seconds>0 else None
        except (TypeError,ValueError,OverflowError):
            try:
                retry_date=parsedate_to_datetime(raw)
                if retry_date.tzinfo is None:
                    retry_date=retry_date.replace(tzinfo=timezone.utc)
                seconds=(retry_date-datetime.now(timezone.utc)).total_seconds()
                return seconds if seconds>0 else None
            except (ValueError,TypeError,OverflowError):
                return None

    def retry_after(self,value,attempt,max_delay=None,jitter=True):
        """解析 Retry-After；无效值回退到有上限的指数退避。"""
        retry_seconds=self.parse_retry_after(value)
        if retry_seconds is not None:
            return retry_seconds
        cap=float(max_delay if max_delay is not None else self.config.max_retry_delay)
        base=min(cap,2**attempt)
        if not jitter:
            return base
        return min(cap,base+random.uniform(0,min(0.5,max(0.,cap-base))))

    def _request_semantic_scholar(self,method,url,params=None,**kwargs):
        """匿名 S2 调度：429 立即交给论文延迟队列，其他瞬时错误有限重试。

        所有 S2 请求共享一个 1.1 秒起始间隔和一个全局冷却时间。
        单次 429 不会在当前论文上原地重试；调用方可以先处理其他论文。
        连续 3 次无成功的 429 后，暂停主队列并冷却 3/5/10/20/40/60 秒，
        优先从待重试队列取一篇论文做探测。服务器 Retry-After 优先。
        """
        max_attempts=int(self.config.s2_max_attempts)
        timeout=float(self.config.s2_request_timeout)
        retry_cap=float(self.config.s2_max_retry_delay)
        interval=float(self.config.source_intervals['semantic_scholar'])
        last='REQUEST_ERROR'
        delay=0.0
        cooldown_steps=(3.0,5.0,10.0,20.0,40.0,60.0)

        # Keep the lock through retries: this client never sends parallel S2 requests.
        with self.lock:
            wait_seconds=max(0.0,self.next_at-self.clock())
            if wait_seconds:
                self.sleep(wait_seconds)
            first_request_started=self.clock()
            self.next_at=first_request_started+interval

            for attempt in range(max_attempts):
                self.network_requests+=1
                attempt_started=self.clock()
                try:
                    response=self.session.request(
                        method,url,params=params,timeout=timeout,**kwargs
                    )
                    elapsed=max(0.0,self.clock()-attempt_started)
                    status=response.status_code
                    self.detail_logger.debug(
                        'semantic_scholar %s HTTP %s，耗时 %.2fs，attempt=%s/%s，interval=%.1fs',
                        method,status,elapsed,attempt+1,max_attempts,interval
                    )

                    if status==429:
                        server_delay=self.parse_retry_after(
                            response.headers.get('Retry-After')
                        )
                        self.s2_consecutive_429+=1
                        self.s2_deferred_total+=1
                        self.retries+=1
                        global_pause=False

                        if server_delay is not None:
                            delay=server_delay
                            global_pause=True
                            self.s2_pause_requested=True
                            self.s2_pause_level=max(1,self.s2_pause_level)
                        elif self.s2_pause_requested or self.s2_consecutive_429>=3:
                            self.s2_pause_level=min(
                                self.s2_pause_level+1,len(cooldown_steps)
                            )
                            delay=cooldown_steps[self.s2_pause_level-1]
                            global_pause=True
                            self.s2_pause_requested=True
                        else:
                            # A transient 429 only defers this paper. The next paper
                            # may proceed at the normal shared 1.1-second cadence.
                            delay=0.0

                        response.close()
                        # Without a server/global cooldown, keep only the already
                        # reserved 1.1s request-start slot (do not add 1.1s after response).
                        if delay>0:
                            self.next_at=max(self.next_at,self.clock()+delay)
                        self.detail_logger.warning(
                            'Semantic Scholar 429：当前论文分页延后；连续429=%s，全局冷却=%.1fs，暂停主队列=%s',
                            self.s2_consecutive_429,delay,global_pause
                        )
                        raise DeferredRequest(
                            'semantic_scholar','HTTP_429',attempt+1,
                            wait_seconds=delay,global_pause=global_pause
                        )

                    if status in (500,502,503,504):
                        delay=self.retry_after(
                            response.headers.get('Retry-After'),attempt,
                            max_delay=retry_cap,jitter=False
                        )
                        last=f'HTTP_{status}'
                        response.close()
                        if attempt==max_attempts-1:
                            self.next_at=max(self.next_at,self.clock()+delay)
                            break
                        self.retries+=1
                        self.logger.warning(
                            'semantic_scholar %s，%.1fs 后重试 (%s/%s)',
                            last,delay,attempt+1,max_attempts-1
                        )
                        self.sleep(delay)
                        continue

                    if status>=400:
                        response.close()
                        raise RequestFailure(
                            'semantic_scholar',f'HTTP_{status}',attempt+1
                        )

                    # Any successful S2 response closes the consecutive-429 streak.
                    if self.s2_consecutive_429 or self.s2_pause_requested:
                        self.detail_logger.debug(
                            'Semantic Scholar 探测成功：清除全局429冷却状态'
                        )
                    self.s2_consecutive_429=0
                    self.s2_pause_requested=False
                    self.s2_pause_level=0
                    return response

                except DeferredRequest:
                    raise
                except (requests.ConnectionError,requests.Timeout) as exc:
                    last='CONNECTION_ERROR'
                    delay=self.retry_after(None,attempt,max_delay=retry_cap,jitter=False)
                    self.detail_logger.debug(
                        'semantic_scholar %s 网络异常：%s',method,type(exc).__name__
                    )
                    if attempt==max_attempts-1:
                        self.next_at=max(self.next_at,self.clock()+delay)
                        break
                    self.retries+=1
                    self.logger.warning(
                        'semantic_scholar %s，%.1fs 后重试 (%s/%s)',
                        last,delay,attempt+1,max_attempts-1
                    )
                    self.sleep(delay)

                except requests.RequestException as exc:
                    last='REQUEST_ERROR'
                    delay=self.retry_after(None,attempt,max_delay=retry_cap,jitter=False)
                    self.detail_logger.debug(
                        'semantic_scholar %s 请求异常：%s',method,type(exc).__name__
                    )
                    if attempt==max_attempts-1:
                        self.next_at=max(self.next_at,self.clock()+delay)
                        break
                    self.retries+=1
                    self.logger.warning(
                        'semantic_scholar %s，%.1fs 后重试 (%s/%s)',
                        last,delay,attempt+1,max_attempts-1
                    )
                    self.sleep(delay)

        raise RequestFailure('semantic_scholar',last,max_attempts)

    def request(self,method,url,params=None,json_body=None,**kwargs):
        # Module-supplied Notebook timeout/retry settings are governed by Config.
        kwargs.pop('timeout',None)
        kwargs.pop('max_retries',None)
        if json_body is not None:
            kwargs['json']=json_body

        if self.source == 'semantic_scholar':
            return self._request_semantic_scholar(method,url,params=params,**kwargs)

        max_attempts=int(self.config.max_retries)+1
        retry_cap=float(self.config.max_retry_delay)
        last='REQUEST_ERROR'

        for attempt in range(max_attempts):
            with self.lock:
                # Enforce a minimum interval between request starts.
                wait_seconds=max(0.,self.next_at-self.clock())
                if wait_seconds:
                    self.sleep(wait_seconds)

                request_started=self.clock()
                interval_used=self.interval
                self.next_at=request_started+interval_used
                self.network_requests+=1

                try:
                    response=self.session.request(
                        method,
                        url,
                        params=params,
                        timeout=self.config.request_timeout,
                        **kwargs
                    )
                    elapsed=max(0.,self.clock()-request_started)
                    status=response.status_code
                    self.detail_logger.debug(
                        '%s %s HTTP %s，耗时 %.2fs，attempt=%s/%s，interval=%.1fs',
                        self.source,method,status,elapsed,attempt+1,
                        max_attempts,interval_used
                    )

                    if status not in self.RETRYABLE_STATUS:
                        if status>=400:
                            response.close()
                            raise RequestFailure(self.source,f'HTTP_{status}',attempt+1)

                        # A successful response restores the adaptive interval.
                        if self.interval!=self.base_interval:
                            previous_interval=self.interval
                            self.interval=self.base_interval
                            self.next_at=request_started+self.base_interval
                            self.detail_logger.debug(
                                '%s 请求成功，限速间隔从 %.1fs 恢复至 %.1fs',
                                self.source,previous_interval,self.base_interval
                            )
                        return response

                    delay=self.retry_after(
                        response.headers.get('Retry-After'),
                        attempt,
                        max_delay=retry_cap
                    )
                    last=f'HTTP_{status}'
                    response.close()

                    if status==429:
                        previous_interval=self.interval
                        self.interval=max(
                            self.base_interval,
                            min(self.config.max_retry_delay,self.interval*2)
                        )
                        self.detail_logger.debug(
                            '%s 收到 HTTP 429，限速间隔从 %.1fs 调整至 %.1fs',
                            self.source,previous_interval,self.interval
                        )

                except (requests.ConnectionError,requests.Timeout) as exc:
                    elapsed=max(0.,self.clock()-request_started)
                    last='CONNECTION_ERROR'
                    self.detail_logger.debug(
                        '%s %s 网络连接异常，耗时 %.2fs：%s',
                        self.source,method,elapsed,type(exc).__name__
                    )
                    if not self.config.wait_for_reconnection:
                        raise RequestFailure(self.source,last,attempt+1)
                    delay=max(
                        self.config.reconnection_delay,
                        self.retry_after(None,attempt,max_delay=retry_cap)
                    )

                except requests.RequestException as exc:
                    elapsed=max(0.,self.clock()-request_started)
                    last='REQUEST_ERROR'
                    self.detail_logger.debug(
                        '%s %s 请求异常，耗时 %.2fs：%s',
                        self.source,method,elapsed,type(exc).__name__
                    )
                    delay=self.retry_after(None,attempt,max_delay=retry_cap)

            if attempt < max_attempts-1:
                self.retries+=1
                self.logger.warning(
                    '%s %s，%.1fs 后重试 (%s/%s)',
                    self.source,last,delay,attempt+1,max_attempts-1
                )
                self.sleep(delay)

        raise RequestFailure(self.source,last,max_attempts)

    def arxiv(self,params,cache_path):
        key={'url':'https://export.arxiv.org/api/query','params':params}
        def valid(z):
            root=ET.fromstring(gzip.decompress(z))
            if root.tag!='{http://www.w3.org/2005/Atom}feed':return False
            entries=root.findall('{http://www.w3.org/2005/Atom}entry')
            if 'id_list' in params:
                expected=set(params['id_list'].split(','))
                got={e.findtext('{http://www.w3.org/2005/Atom}id','').split('/abs/')[-1] for e in entries}
                return expected==got
            total_node=root.find('{http://a9.com/-/spec/opensearch/1.1/}totalResults')
            if total_node is None or total_node.text is None:return False
            total=int(total_node.text)
            expected=min(int(params.get('max_results',10)),max(0,total-int(params.get('start',0))))
            return len(entries)==expected
        # 旧 seed 文件名包含年月分页；v1 文件名包含确切 ID 指纹。
        data=self.cache.read_bytes(cache_path,key,valid,legacy=True)
        if data is None:
            response=self.request('GET',key['url'],params=params)
            data=gzip.compress(response.content);response.close()
            self.cache.write_bytes(cache_path,key,data,valid)
        return gzip.decompress(data)

    def s2(self,url,payload,legacy_urls,cache_dir):
        def valid(x):
            if payload is not None:
                return isinstance(x,list) and len(x)==len(payload['ids']) and all(v is None or isinstance(v,dict) and 'paperId' in v for v in x)
            return isinstance(x,dict) and isinstance(x.get('data'),list) and all(isinstance(v,dict) and 'citingPaper' in v for v in x['data'])
        for u in [url]+list(legacy_urls or []):
            # Preserve the exact SHA256 key format used by the original Notebook.
            name=hashlib.sha256(json.dumps([u,payload],sort_keys=True).encode()).hexdigest()+'.json'
            p=Path(cache_dir)/name
            data=self.cache.read_json_fast(p,valid)
            if data is not None:return data

        response=self.request(
            'POST' if payload is not None else 'GET',
            url,
            json_body=payload
        )
        data=response.json();response.close()
        if not valid(data):raise ValueError('S2 响应结构不完整')
        p=Path(cache_dir)/(hashlib.sha256(json.dumps([url,payload],sort_keys=True).encode()).hexdigest()+'.json')
        self.cache.write_json_fast(p,data)
        return data

    def openreview(self,host,endpoint,payload,method,cache_dir):
        key=[host,endpoint,method,payload]
        name=hashlib.sha256(json.dumps(key,sort_keys=True,ensure_ascii=False).encode()).hexdigest()+'.json'
        p=Path(cache_dir)/name
        valid=lambda x:isinstance(x,dict) and isinstance(x.get('notes'),list)
        data=self.cache.read_json_fast(p,valid)
        if data is None:
            response=self.request(method,host+endpoint,params=payload if method=='GET' else None,json_body=payload if method=='POST' else None)
            data=response.json();response.close()
            if not valid(data):raise ValueError('OpenReview 响应缺少 notes')
            self.cache.write_json_fast(p,data)
        return data
