"""Disk-backed translation of unchanged HistoricalWorkDeduplicator semantics."""
import json
from .progress import tqdm
from .rules import make_history_fallback_key

class DiskDeduplicator:
    def __init__(self,db):
        self.db=db
        db.executescript('''
CREATE TABLE IF NOT EXISTS works(id INTEGER PRIMARY KEY,parent INTEGER,rank INTEGER,strong INTEGER);
CREATE TABLE IF NOT EXISTS work_years(work INTEGER,pid TEXT,year INTEGER,PRIMARY KEY(work,pid));
CREATE INDEX IF NOT EXISTS work_person ON work_years(pid,year,work);
CREATE TABLE IF NOT EXISTS strong_keys(key TEXT PRIMARY KEY,work INTEGER);
CREATE TABLE IF NOT EXISTS fallback_keys(key TEXT,work INTEGER,PRIMARY KEY(key,work));
CREATE TABLE IF NOT EXISTS record_works(key TEXT PRIMARY KEY,work INTEGER);
CREATE INDEX IF NOT EXISTS record_work_id ON record_works(work,key);
''')
    def find(self,w):
        path=[]
        while True:
            p=self.db.execute('SELECT parent FROM works WHERE id=?',(w,)).fetchone()[0]
            if p==w:break
            path.append(w);w=p
        self.db.executemany('UPDATE works SET parent=? WHERE id=?',((w,x) for x in path));return w
    def new(self):
        c=self.db.execute('INSERT INTO works(parent,rank,strong) VALUES (0,0,0)');w=c.lastrowid;self.db.execute('UPDATE works SET parent=? WHERE id=?',(w,w));return w
    def strong(self,w):return bool(self.db.execute('SELECT strong FROM works WHERE id=?',(w,)).fetchone()[0])
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a==b:return a
        ra=self.db.execute('SELECT rank FROM works WHERE id=?',(a,)).fetchone()[0];rb=self.db.execute('SELECT rank FROM works WHERE id=?',(b,)).fetchone()[0]
        if ra<rb:a,b=b,a;ra,rb=rb,ra
        self.db.execute('UPDATE works SET parent=? WHERE id=?',(a,b))
        if ra==rb:self.db.execute('UPDATE works SET rank=rank+1 WHERE id=?',(a,))
        self.db.execute('UPDATE works SET strong=? WHERE id=?',(int(self.strong(a) or self.strong(b)),a))
        self.db.execute('''INSERT INTO work_years SELECT ?,pid,year FROM work_years WHERE work=?
        ON CONFLICT(work,pid) DO UPDATE SET year=MIN(year,excluded.year)''',(a,b))
        self.db.execute('DELETE FROM work_years WHERE work=?',(b,));return a
    def add_record(self,strong_keys,fallback_key,person_ids,year,record_key):
        strong_keys=sorted(set(strong_keys));strongroots=set()
        for k in strong_keys:
            row=self.db.execute('SELECT work FROM strong_keys WHERE key=?',(k,)).fetchone()
            if row:strongroots.add(self.find(row[0]))
        fallroots={self.find(r[0]) for r in self.db.execute('SELECT work FROM fallback_keys WHERE key=?',(fallback_key,))} if fallback_key else set()
        if strongroots:
            root=min(strongroots)
            for other in sorted(strongroots):root=self.union(root,other)
            if len(fallroots)==1:
                f=self.find(next(iter(fallroots)));root=self.find(root)
                if f!=root and not self.strong(f):root=self.union(root,f)
        elif strong_keys:
            f=next(iter(fallroots)) if len(fallroots)==1 else None
            root=f if f is not None and not self.strong(f) else self.new()
        else:root=next(iter(fallroots)) if len(fallroots)==1 else self.new()
        root=self.find(root)
        if strong_keys:
            self.db.execute('UPDATE works SET strong=1 WHERE id=?',(root,));self.db.executemany('INSERT OR REPLACE INTO strong_keys VALUES (?,?)',((k,root) for k in strong_keys))
        if fallback_key:self.db.execute('INSERT OR IGNORE INTO fallback_keys VALUES (?,?)',(fallback_key,root))
        self.db.executemany('INSERT INTO work_years VALUES (?,?,?) ON CONFLICT(work,pid) DO UPDATE SET year=MIN(year,excluded.year)',((root,pid,year) for pid in set(person_ids) if year is not None))
        self.db.execute('INSERT OR REPLACE INTO record_works VALUES (?,?)',(record_key,root))
        return root
    def finalize(self):
        for key,w in self.db.execute('SELECT key,work FROM record_works'):
            self.db.execute('UPDATE record_works SET work=? WHERE key=?',(self.find(w),key))
        self.db.commit()

def build_works(store):
    db=store.db;d=DiskDeduplicator(db)
    if store.get('dedup_complete')=='1':return
    # Deterministic replay is safe; a half-built dedup graph is never published.
    for table in ('works','work_years','strong_keys','fallback_keys','record_works'):db.execute('DELETE FROM '+table)
    total=db.execute('SELECT COUNT(*) FROM publications').fetchone()[0]
    for key,payload in tqdm(db.execute('SELECT key,payload FROM publications ORDER BY ordinal'),total=total,desc='Disk historical work deduplication',unit='records'):
        pids=[r[0] for r in db.execute('SELECT pid FROM candidate_publications WHERE key=?',(key,))]
        r=json.loads(payload)
        # Identity-only records (including unknown-year focal candidates) still connect versions.
        if r['year'] is None:pids=[]
        d.add_record({'doi:'+v for v in r['dois']}|{'arxiv:'+v for v in r['arxiv_ids']},make_history_fallback_key(r['title'],r['authors']),pids,r['year'],key)
    d.finalize();store.set('dedup_complete',1);db.commit()
