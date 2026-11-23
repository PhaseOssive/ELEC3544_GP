"""All paths and decision thresholds. Original caches are never writable."""
from dataclasses import dataclass,asdict
from pathlib import Path
import json

@dataclass
class Config:
    runtime_mode: str='local'
    FETCH_DATA_ROOT: str='./GP_Research_Papers'
    DBLP_XML_PATH: str='./GP_Research_Papers/data/raw/dblp/dblp.xml.gz'
    CACHE_DIR: str='./dblp_cache'
    PROFILE_INDEX_PATH: str|None=None
    DTD_PATH: str|None=None
    DERIVED_CACHE_DIR: str|None=None
    OUTPUT_DIR: str='./GP_DBLP_Author_Recovery/local_recovery_output'
    master_file: str='data/processed/papers_master_raw_stage.csv'
    authors_file: str='data/processed/authors_arxiv_v1.csv'
    dblp_directory: str='data/processed/dblp_author_features_v3_1_1'
    checkpoint_records: int=50000
    resume_enabled: bool=True
    log_level: str='INFO'
    alignment_window: int=2
    alignment_margin: float=0.12
    alignment_min_similarity: float=0.85
    title_similarity: float=0.80
    title_keyword_coverage: float=0.60
    title_year_window: int=4
    backtest_min_holdout: int=100
    precision_lower_bound: float=0.99
    network_auto_accept: bool=True
    sqlite_cache_mb: int=64

    @classmethod
    def load(cls,path):return cls(**json.loads(Path(path).read_text()))
    def validate(self):
        if self.runtime_mode not in ('local','colab'):raise ValueError('runtime_mode must be local or colab')
        for k in ('FETCH_DATA_ROOT','DBLP_XML_PATH','CACHE_DIR','OUTPUT_DIR'):
            setattr(self,k,str(Path(getattr(self,k)).expanduser().resolve()))
        for k,default in [('PROFILE_INDEX_PATH',Path(self.CACHE_DIR)/'index.sqlite'),('DTD_PATH',(Path(self.CACHE_DIR)/'dblp.dtd') if (Path(self.CACHE_DIR)/'dblp.dtd').is_file() else Path(self.DBLP_XML_PATH).parent/'dblp.dtd'),('DERIVED_CACHE_DIR',Path(self.CACHE_DIR)/'derived')]:
            setattr(self,k,str(Path(getattr(self,k) or default).expanduser().resolve()))
        for path in (self.PROFILE_INDEX_PATH,):
            if not Path(path).is_relative_to(Path(self.CACHE_DIR)):raise ValueError('Original cache files must be inside CACHE_DIR')
        if not Path(self.DERIVED_CACHE_DIR).is_relative_to(Path(self.CACHE_DIR)) or self.DERIVED_CACHE_DIR==self.CACHE_DIR:raise ValueError('DERIVED_CACHE_DIR must be a distinct child of CACHE_DIR')
        out=Path(self.OUTPUT_DIR);src=Path(self.FETCH_DATA_ROOT);cache=Path(self.CACHE_DIR)
        if out.is_relative_to(src) or src.is_relative_to(out) or out.is_relative_to(cache) or cache.is_relative_to(out):raise ValueError('Output, previous inputs and cache must be disjoint')
        for p in (Path(self.PROFILE_INDEX_PATH),Path(self.DTD_PATH),Path(self.DBLP_XML_PATH)):
            if p.is_relative_to(out) or p.is_relative_to(Path(self.DERIVED_CACHE_DIR)):raise ValueError('Protected source inside writable area')
        if Path(self.DERIVED_CACHE_DIR).is_relative_to(src):raise ValueError('Derived cache cannot be under previous inputs')
        if cache.is_relative_to(src) or src.is_relative_to(cache):raise ValueError('Fetch inputs and cache must be disjoint')
        for rel in (self.master_file,self.authors_file,self.dblp_directory):
            if not (src/rel).resolve().is_relative_to(src):raise ValueError('Input file escapes previous inputs')
        if self.checkpoint_records<1 or not 1<=self.alignment_window<=3:raise ValueError('Invalid checkpoint/window')
        if not 0<self.precision_lower_bound<=1 or not 0<self.alignment_min_similarity<=1:raise ValueError('Invalid threshold')
        return self
    @property
    def output_root(self):return self.OUTPUT_DIR
    def out(self,name):
        p=(Path(self.OUTPUT_DIR)/name).resolve()
        if not p.is_relative_to(Path(self.OUTPUT_DIR)) or p.is_relative_to(Path(self.FETCH_DATA_ROOT)) or p.is_relative_to(Path(self.CACHE_DIR)):raise ValueError('Unsafe output path')
        p.parent.mkdir(parents=True,exist_ok=True);return p
    def derived(self,name):
        p=(Path(self.DERIVED_CACHE_DIR)/name).resolve()
        if not p.is_relative_to(Path(self.DERIVED_CACHE_DIR)) or p in [Path(self.PROFILE_INDEX_PATH),Path(self.DTD_PATH),Path(self.DBLP_XML_PATH)]:raise ValueError('Unsafe derived path')
        p.parent.mkdir(parents=True,exist_ok=True);return p
    def snapshot(self):return asdict(self)
