"""One-time local profile construction. Existing indexes are never overwritten."""
import json,sqlite3,os,fcntl,logging
from pathlib import Path
from datetime import datetime,timezone
from common import fingerprint,sha256
from .search import records,dtd_audit
from .profile import ProfileIndex
from .rules import child_texts,normalize_name,name_signature
from .progress import tqdm

def ensure_index(cfg):
    path=Path(cfg.PROFILE_INDEX_PATH)
    if path.exists():
        index=ProfileIndex(path)
        if Path(cfg.DBLP_XML_PATH).is_file():
            actual=fingerprint(cfg.DBLP_XML_PATH);expected=index.signature[1]
            if any(actual[k]!=expected[k] for k in ('size','sample_sha256')):
                index.db.close();raise ValueError('Existing index is for a different XML snapshot; not overwritten')
        return index,False
    if not Path(cfg.DBLP_XML_PATH).is_file():raise FileNotFoundError('Profile index absent: local DBLP_XML_PATH required for first build')
    if not Path(cfg.DTD_PATH).is_file():raise FileNotFoundError('Local DTD_PATH required for first build')
    dtd_audit(cfg.DTD_PATH);path.parent.mkdir(parents=True,exist_ok=True)
    lockpath=cfg.derived('profile_build.lock')
    with lockpath.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if path.exists():return ProfileIndex(path),False
        temp=path.with_name(path.name+'.building')
        before=fingerprint(cfg.DBLP_XML_PATH);xmlsha=sha256(cfg.DBLP_XML_PATH);dtdsha=sha256(cfg.DTD_PATH)
        signature=json.dumps(['dblp-profile-index-1',before,fingerprint(cfg.DTD_PATH)],sort_keys=True)
        db=sqlite3.connect(temp)
        try:
            db.executescript('''PRAGMA journal_mode=DELETE; PRAGMA synchronous=FULL; PRAGMA temp_store=FILE;
CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY,v TEXT);
CREATE TABLE IF NOT EXISTS profiles(pid TEXT PRIMARY KEY,redirect TEXT,valid INTEGER);
CREATE TABLE IF NOT EXISTS aliases(pid TEXT,raw TEXT,norm TEXT,sig TEXT,PRIMARY KEY(pid,raw));
CREATE INDEX IF NOT EXISTS alias_raw ON aliases(raw);
CREATE INDEX IF NOT EXISTS alias_norm ON aliases(norm);
CREATE INDEX IF NOT EXISTS alias_sig ON aliases(sig);''')
            def get(k):
                row=db.execute('SELECT v FROM meta WHERE k=?',(k,)).fetchone();return row[0] if row else None
            def put(k,v):db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',(k,str(v)))
            if get('signature') not in (None,signature):raise ValueError('Unfinished build belongs to another snapshot; preserve it and choose a new PROFILE_INDEX_PATH')
            if get('xml_sha256') not in (None,xmlsha):raise ValueError('XML full digest differs from unfinished build')
            put('signature',signature);put('xml_sha256',xmlsha);put('dtd_sha256',dtdsha)
            checkpoint=int(get('checkpoint') or 0)
            logging.info('Build missing profile index locally; committed records=%d',checkpoint)
            with tqdm(desc='Initial local Profile/Alias index',unit='records') as bar:
                for n,e in enumerate(records(cfg.DBLP_XML_PATH,cfg.DTD_PATH),1):
                    bar.update(1)
                    if n<=checkpoint:continue
                    key=e.get('key','')
                    if e.tag=='www' and key.startswith('homepages/'):
                        redirect=child_texts(e,'crossref');titles=child_texts(e,'title')
                        if 'Home Page' in titles or redirect:
                            valid=not bool(set(e.get('publtype','').lower().split())&{'disambiguation','group','noshow'})
                            db.execute('INSERT OR REPLACE INTO profiles VALUES (?,?,?)',(key,redirect[0] if redirect else '',int(valid)))
                            db.executemany('INSERT OR REPLACE INTO aliases VALUES (?,?,?,?)',((key,a,normalize_name(a),name_signature(a)) for a in child_texts(e,'author')))
                    if n%cfg.checkpoint_records==0:put('checkpoint',n);db.commit()
                put('checkpoint',n if 'n' in locals() else 0)
            if db.execute('SELECT COUNT(*) FROM profiles').fetchone()[0]==0:raise ValueError('XML contains no DBLP profiles; cannot publish empty index')
            if db.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise ValueError('Built index integrity failure')
            if sha256(cfg.DBLP_XML_PATH)!=xmlsha or sha256(cfg.DTD_PATH)!=dtdsha:raise ValueError('XML/DTD changed while building')
            put('created_at_utc',datetime.now(timezone.utc).isoformat());put('complete',1);db.commit()
        except BaseException:
            db.rollback();raise
        finally:db.close()
        validated=ProfileIndex(temp);validated.db.close()
        with temp.open('rb') as f:os.fsync(f.fileno())
        # All project writers hold the same lock. A file created externally is never replaced.
        if path.exists():raise FileExistsError('Index appeared during build; preserving both files')
        os.replace(temp,path)
        return ProfileIndex(path),True
