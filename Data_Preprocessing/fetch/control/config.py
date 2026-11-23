"""统一配置。JSON 配置未知字段直接报错；导入不创建目录、不挂载 Drive。"""
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import tempfile

@dataclass
class Config:
    storage_mode: str = 'local'
    data_root: str = './GP_Research_Papers'
    start_year: int = 2017
    end_year: int = 2022
    target_per_year: int = 5000
    random_seed: int = 3544
    citation_window_years: int = 3
    cache_policy: str = 'reuse_valid'
    refresh_sources: list[str] = field(default_factory=list)
    resume_enabled: bool = True
    max_retries: int = 7
    request_timeout: float = 120
    wait_for_reconnection: bool = True
    reconnection_delay: float = 30
    max_retry_delay: float = 120
    log_level: str = 'INFO'
    source_intervals: dict = field(default_factory=lambda: {'arxiv':3.1, 'dblp':1.0, 'semantic_scholar':1.1, 'openreview':1.0})
    # Semantic Scholar anonymous requests: match the original Notebook.
    s2_max_attempts: int = 5
    s2_request_timeout: float = 40
    s2_max_retry_delay: float = 60
    arxiv_page_size: int = 2000
    arxiv_metadata_batch_size: int = 100
    s2_batch_size: int = 500
    s2_page_size: int = 1000
    s2_max_pages: int = 10
    s2_max_citations: int = 9999
    openreview_workers: int = 1
    openreview_batch_size: int = 20
    openreview_page_size: int = 1000
    openreview_max_pages: int = 30
    dblp_xml_path: str | None = None
    sources: list[str] = field(default_factory=lambda: ['dblp','semantic_scholar','openreview'])
    seed_csv: str | None = None
    minimum_free_gb: float = 1.0
    user_agent: str = 'GP-Research-Data-Collection/1.0 (academic research)'
    drive_mount_point: str = '/content/drive'

    @classmethod
    def load(cls, path=None):
        if not path:
            return cls()
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            raise ValueError('配置文件顶层必须是 JSON object')
        # Backward compatibility: ignore the v1.0.1 key/anonymous-interval
        # fields. This release intentionally runs Semantic Scholar anonymously.
        data.pop('semantic_scholar_api_key', None)
        data.pop('s2_unauthenticated_interval', None)
        return cls(**data)

    def refresh(self, source):
        return self.cache_policy == 'refresh_all' or (self.cache_policy == 'refresh_source' and source in self.refresh_sources)

    @property
    def paths(self):
        root = Path(self.data_root).expanduser().resolve()
        return root, root/'data/raw', root/'data/processed'

    @property
    def openreview_options(self):
        return dict(workers=self.openreview_workers, batch_size=self.openreview_batch_size,
                    page_size=self.openreview_page_size, max_pages=self.openreview_max_pages)

    def snapshot(self):
        return asdict(self)

    def validate(self):
        if self.storage_mode not in ('local','colab_drive'): raise ValueError('storage_mode 必须为 local / colab_drive')
        if self.cache_policy not in ('reuse_valid','refresh_source','refresh_all'): raise ValueError('不支持的 cache_policy')
        available = {'arxiv','dblp','semantic_scholar','openreview'}
        if not set(self.sources) <= available-{'arxiv'}: raise ValueError('sources 含未知数据源')
        if not set(self.refresh_sources) <= available: raise ValueError('refresh_sources 含未知数据源')
        if self.cache_policy == 'refresh_source' and not self.refresh_sources: raise ValueError('refresh_source 需要指定 refresh_sources')
        for key in ('start_year','end_year','target_per_year','citation_window_years','max_retries','arxiv_page_size','arxiv_metadata_batch_size','s2_batch_size','s2_page_size','s2_max_pages','s2_max_citations','s2_max_attempts','openreview_workers','openreview_batch_size','openreview_page_size','openreview_max_pages'):
            value=getattr(self,key)
            if type(value) is not int or value <= 0: raise ValueError(f'{key} 必须是正整数')
        if self.start_year>self.end_year: raise ValueError('start_year 不能大于 end_year')
        if type(self.random_seed) is not int: raise ValueError('random_seed 必须是整数')
        for key in ('resume_enabled','wait_for_reconnection'):
            if type(getattr(self,key)) is not bool: raise ValueError(f'{key} 必须为布尔值')
        for key in ('request_timeout','reconnection_delay','max_retry_delay','s2_request_timeout','s2_max_retry_delay'):
            if not isinstance(getattr(self,key),(int,float)) or getattr(self,key)<=0: raise ValueError(f'{key} 必须大于 0')
        if self.minimum_free_gb<0: raise ValueError('minimum_free_gb 不能为负数')
        if set(self.source_intervals)!=available or any(not isinstance(x,(int,float)) or x<0 for x in self.source_intervals.values()): raise ValueError('source_intervals 必须为四个来源分别提供非负间隔')
        if self.arxiv_page_size>2000 or self.s2_batch_size>500 or self.s2_page_size>1000: raise ValueError('分页大小超过当前支持上限')
        if self.log_level not in ('DEBUG','INFO','WARNING','ERROR'): raise ValueError('log_level 无效')
        if self.dblp_xml_path and not Path(self.dblp_xml_path).is_file(): raise ValueError('dblp_xml_path 不存在')
        if self.seed_csv and not Path(self.seed_csv).is_file(): raise ValueError('seed_csv 不存在')

    def prepare_storage(self):
        if self.storage_mode=='colab_drive':
            try: from google.colab import drive
            except ImportError as exc: raise RuntimeError('colab_drive 仅能在 Colab 使用；本地请选择 local') from exc
            drive.mount(self.drive_mount_point)
            if not self.paths[0].is_relative_to(Path(self.drive_mount_point).resolve()):
                raise ValueError('colab_drive 的 data_root 必须位于挂载目录下')
        root,raw,processed=self.paths
        for p in (root,raw,processed,root/'logs',root/'state',raw/'arxiv/seed_xml',raw/'arxiv/v1_xml',raw/'arxiv/v1_jsonl'):
            p.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryFile(dir=root) as f:
            f.write(b'GP');f.seek(0)
            if f.read()!=b'GP':raise OSError('存储读写校验失败')
        import shutil
        if shutil.disk_usage(root).free < self.minimum_free_gb*1024**3: raise OSError('剩余磁盘低于 minimum_free_gb')
