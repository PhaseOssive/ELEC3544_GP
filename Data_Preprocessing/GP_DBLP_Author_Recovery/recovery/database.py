"""Persistent derived state, transactional checkpoints, normalized evidence tables."""
import sqlite3,json,logging
from common import sha256
from pathlib import Path
from .progress import tqdm
from common import export_query

SCHEMA='local-deep-search-2'
class Database:
    def __init__(self,path,cache_mb=64):
        self.db=sqlite3.connect(path)
        self.db.execute('PRAGMA journal_mode=DELETE') # Network-mounted Drive: avoid WAL locks.
        self.db.execute('PRAGMA synchronous=FULL');self.db.execute('PRAGMA temp_store=FILE');self.db.execute(f'PRAGMA cache_size={-cache_mb*1024}')
        self.db.executescript('''
CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY,v TEXT);
CREATE TABLE IF NOT EXISTS occurrences(oid TEXT PRIMARY KEY,aid TEXT,pos INTEGER,name TEXT,pid TEXT,t0 INTEGER,verified INTEGER,payload TEXT);
CREATE INDEX IF NOT EXISTS occurrence_paper ON occurrences(aid,pos);
CREATE TABLE IF NOT EXISTS candidates(oid TEXT,pid TEXT,alias TEXT,method TEXT,grade INTEGER,eligible INTEGER,rejection TEXT,PRIMARY KEY(oid,pid,alias,method));
CREATE INDEX IF NOT EXISTS candidate_pid ON candidates(pid,eligible);
CREATE INDEX IF NOT EXISTS candidate_occ ON candidates(oid,eligible,grade);
CREATE TABLE IF NOT EXISTS wanted(pid TEXT PRIMARY KEY,ambiguous_alias INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS wanted_alias(raw TEXT PRIMARY KEY,pid TEXT);
CREATE INDEX IF NOT EXISTS wanted_alias_pid ON wanted_alias(pid);
CREATE TABLE IF NOT EXISTS publications(key TEXT PRIMARY KEY,year INTEGER,title TEXT,payload TEXT,ordinal INTEGER);
CREATE INDEX IF NOT EXISTS publication_year ON publications(year);
CREATE TABLE IF NOT EXISTS identifiers(kind TEXT,value TEXT,key TEXT,PRIMARY KEY(kind,value,key));
CREATE INDEX IF NOT EXISTS identifier_record ON identifiers(key);
CREATE TABLE IF NOT EXISTS publication_authors(key TEXT,pos INTEGER,name TEXT,pid TEXT,evidence TEXT,PRIMARY KEY(key,pos));
CREATE INDEX IF NOT EXISTS publication_person ON publication_authors(pid,key);
CREATE TABLE IF NOT EXISTS candidate_publications(pid TEXT,key TEXT,evidence TEXT,PRIMARY KEY(pid,key));
CREATE INDEX IF NOT EXISTS candidate_record ON candidate_publications(key,pid);
CREATE TABLE IF NOT EXISTS candidate_publication_years(pid TEXT,year INTEGER,key TEXT,PRIMARY KEY(pid,key));
CREATE INDEX IF NOT EXISTS candidate_year ON candidate_publication_years(pid,year);
CREATE TABLE IF NOT EXISTS candidate_history_status(pid TEXT PRIMARY KEY,scan_complete INTEGER,record_count INTEGER,missing_year INTEGER,ambiguous_alias INTEGER,status TEXT);
CREATE TABLE IF NOT EXISTS targets(aid TEXT,key TEXT,method TEXT,similarity REAL,anchors INTEGER,reliable INTEGER,PRIMARY KEY(aid,key));
CREATE TABLE IF NOT EXISTS focal(aid TEXT,key TEXT,PRIMARY KEY(aid,key));
CREATE TABLE IF NOT EXISTS focal_audit(aid TEXT,key TEXT,source TEXT,reason TEXT,identity_reliable INTEGER,PRIMARY KEY(aid,key,source));
CREATE TABLE IF NOT EXISTS candidate_coauthors(oid TEXT,pid TEXT,peer TEXT,key TEXT,year INTEGER,work INTEGER,PRIMARY KEY(oid,pid,peer,key));
CREATE INDEX IF NOT EXISTS coauthor_candidate ON candidate_coauthors(oid,pid,work);
CREATE TABLE IF NOT EXISTS scores(oid TEXT,pid TEXT,grade INTEGER,common INTEGER,joint INTEGER,coverage REAL,rare REAL,topic REAL,score REAL,complete INTEGER,PRIMARY KEY(oid,pid));
CREATE TABLE IF NOT EXISTS conflicts(aid TEXT,oid TEXT,reason TEXT,evidence TEXT);
''')
        existing=self.get('schema')
        if existing and existing!=SCHEMA:raise ValueError('Incompatible derived schema; choose a new DERIVED_CACHE_DIR')
        self.set('schema',SCHEMA);self.db.commit()
    def get(self,k,default=None):
        r=self.db.execute('SELECT v FROM meta WHERE k=?',(k,)).fetchone();return r[0] if r else default
    def set(self,k,v):self.db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',(k,str(v)))
    def bind(self,signature):
        old=self.get('input_signature')
        if old and old!=signature:
            previous=json.loads(old);current=json.loads(signature)
            known=json.loads((Path(__file__).parent.parent/'cache_compatibility.json').read_text())['trusted_previous_algorithm_sha256']
            old_code=previous.pop('algorithm_sha256',None)
            current.pop('algorithm_sha256',None)
            if old_code!=known or previous!=current:
                raise ValueError('Derived cache belongs to different inputs/config/XML/code; preserve it and use a new derived subdirectory')
            logging.info('Verified exact Standalone predecessor; reuse candidate/XML/dedup checkpoints')
            self.set('performance_migration_from',old)
            # Previous versions had no transactional per-author score checkpoint.
            self.db.execute('DROP TABLE IF EXISTS score_done')
            self.set('scores_complete','0')
        self.set('input_signature',signature);self.db.commit()
    def candidates(self,index,rows):
        if self.get('candidates_complete')=='1':return
        for r in tqdm(rows,desc='Expand all profile candidates',unit='occurrences'):
            oid=r['occurrence_id'];name=r['original_name'] if r['identity_status']=='VERIFIED' else (r.get('repaired_name') or r['original_name'])
            self.db.execute('INSERT OR REPLACE INTO occurrences VALUES (?,?,?,?,?,?,?,?)',(oid,r['arxiv_id'],int(r['author_position']),name,r.get('dblp_person_id',''),int(r['t0_year']),int(r['identity_status']=='VERIFIED'),json.dumps(r)))
            # Include verified authors with hidden truth for subsequent independent backtest.
            for pid,alias,method,grade,eligible,rejection in index.expand(name):
                self.db.execute('INSERT OR IGNORE INTO candidates VALUES (?,?,?,?,?,?,?)',(oid,pid,alias,method,grade,eligible,rejection))
        self.db.execute('INSERT OR IGNORE INTO wanted(pid) SELECT DISTINCT pid FROM candidates WHERE eligible=1')
        self.db.execute("INSERT OR IGNORE INTO wanted(pid) SELECT DISTINCT pid FROM occurrences WHERE verified=1 AND pid<>''")
        self.set('candidates_complete',1);self.db.commit()
    def aliases(self,index):
        done=set(r[0] for r in self.db.execute('SELECT DISTINCT pid FROM wanted_alias'))
        for pid, in tqdm(self.db.execute('SELECT pid FROM wanted'),desc='Prepare unique publication aliases',unit='profiles'):
            if pid in done:continue
            for name in index.aliases(pid):
                if index.raw_ids(name)=={pid}:self.db.execute('INSERT OR IGNORE INTO wanted_alias VALUES (?,?)',(name,pid))
                else:self.db.execute('UPDATE wanted SET ambiguous_alias=1 WHERE pid=?',(pid,))
        self.db.commit()
    def export_candidates(self,cfg):
        path=cfg.out('matching/expanded_candidate_pids.csv')
        key='candidate_export:'+str(path)
        if path.exists() and self.get(key) and sha256(path)==self.get(key):
            logging.info('Validated completed candidate CSV reused: %s',path);return
        export_query(cfg,'matching/expanded_candidate_pids.csv',self.db,'SELECT o.aid AS arxiv_id,o.pos AS author_position,c.* FROM candidates c JOIN occurrences o USING(oid) ORDER BY o.aid,o.pos,c.eligible DESC,c.grade DESC,c.pid')
        self.set(key,sha256(path));self.db.commit()
