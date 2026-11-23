"""Read-only Fetch recovery cache adapter. Mismatches fall back to XML."""
import json,sqlite3
from pathlib import Path
from contextlib import closing
from progress import digest
from vendor.rules import normalize_name,name_signature
from vendor.names import base_name

def readonly(path):
    # Use a completed snapshot only; never open a live WAL database.
    if Path(str(path)+'-wal').exists() or Path(str(path)+'-shm').exists():
        raise ValueError('Live WAL/SHM cache is not a completed snapshot')
    # mode=ro, never immutable: do not silently ignore a live WAL file.
    return sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True)

def discover(root):
    root=Path(root);choices=[]
    snapshots=root/'runtime_snapshots'
    latest=snapshots/'LATEST.json'
    if latest.exists():
        try:
            obj=json.loads(latest.read_text());values=[obj] if isinstance(obj,str) else list(obj.values())
            for v in values:
                if isinstance(v,str) and '/' not in v and '\\' not in v and not v.endswith('.partial'):choices.append(snapshots/v)
        except (ValueError,AttributeError):pass
    choices.extend(sorted((p for p in snapshots.glob('*') if p.is_dir() and not p.name.endswith('.partial')),reverse=True))
    choices.append(root)
    # Backup indexes may belong to a different XML; never auto-select them.
    return list(dict.fromkeys(p for p in choices if (p/'index.sqlite').is_file()))

class Legacy:
    def __init__(self,cfg,xml_hash,dtd_hash,progress):
        self.cfg=cfg;self.xml_hash=xml_hash;self.dtd_hash=dtd_hash;self.progress=progress
        self.history=None;self.report={'enabled':bool(cfg.get('reuse_missing_cache',True)), 'attempts':[], 'profile_rows':0,'history_payload_hits':0,'history_payload_misses':0,'xml_passes_saved':0}
    def seed(self,snap,authors):
        if not self.report['enabled'] or not self.cfg.get('legacy_cache_root'):return False
        for root in discover(self.cfg['legacy_cache_root']):
            attempt={'snapshot':str(root)};self.report['attempts'].append(attempt)
            try:
                with closing(readonly(root/'index.sqlite')) as db:
                    meta=dict(db.execute('SELECT k,v FROM meta'))
                    if meta.get('complete')!='1':raise ValueError('Profile index incomplete')
                    if meta.get('xml_sha256')!=self.xml_hash or meta.get('dtd_sha256')!=self.dtd_hash:raise ValueError('Missing or different full XML/DTD SHA256')
                    # Check the full index bytes against the saved immutable snapshot manifest.
                    manifest_path=root/'snapshot_manifest.json'
                    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
                    if manifest.get('index.sqlite'):
                        if digest(root/'index.sqlite',self.progress,'Verify old profile index')!=manifest['index.sqlite']:raise ValueError('Index manifest hash mismatch')
                    names=set(authors.original_name)|set(authors.repaired_name);pids=set(authors.dblp_person_id);pids.discard('')
                    norms={normalize_name(base_name(n)) for n in names}
                    sigs={name_signature(n) for n in names};sigs.discard(None)
                    total=db.execute('SELECT count(*) FROM aliases').fetchone()[0]
                    self.progress.begin('Select cached author profiles',total,'Streaming alias index preserves the exact XML candidate-selection rule.')
                    for i,(pid,name) in enumerate(db.execute('SELECT pid,raw FROM aliases'),1):
                        if normalize_name(base_name(name)) in norms or name_signature(name) in sigs:pids.add(pid)
                        if i%10000==0:self.progress.update(i)
                    self.progress.update(total)
                    pending=list(pids);seen=set();profiles=[];aliases=[]
                    while pending:
                        pid=pending.pop()
                        if pid in seen:continue
                        seen.add(pid);row=db.execute('SELECT pid,redirect,valid FROM profiles WHERE pid=?',(pid,)).fetchone()
                        if not row:continue
                        profiles.append((*row,'[]','[]'));aliases.extend(db.execute('SELECT pid,raw FROM aliases WHERE pid=?',(pid,)))
                        if row[1]:pending.append(row[1])
                    if not profiles:raise ValueError('No relevant profiles in old cache')
                    with snap.db:
                        snap.db.executemany('INSERT OR REPLACE INTO profiles VALUES (?,?,?,?,?)',profiles)
                        snap.db.executemany('INSERT OR IGNORE INTO aliases VALUES (?,?)',aliases)
                        snap.put('legacy_profiles_seeded',1)
                    self.report.update(profile_rows=len(profiles),xml_passes_saved=1,selected_snapshot=str(root))
                    attempt['status']='PROFILE_INDEX_REUSED'
                    try:self._history(root,manifest,attempt)
                    except (OSError,sqlite3.Error,ValueError,TypeError) as exc:attempt['history']='unavailable; XML payload fallback: '+str(exc)
                    return True
            except (OSError,sqlite3.Error,ValueError,KeyError,TypeError) as exc:
                attempt['status']='FALLBACK';attempt['reason']=str(exc)
        return False
    def _history(self,root,manifest,attempt):
        path=root/'derived/candidate_history.sqlite'
        if not path.is_file():attempt['history']='missing; XML payloads used';return
        if not self.cfg.get('reuse_history_payloads',True):attempt['history']='disabled';return
        # The manifest binds this DB to the exact verified index. Never trust a filename alone.
        expected=manifest.get('derived/candidate_history.sqlite')
        if not expected:attempt['history']='no paired snapshot manifest; XML payloads used';return
        if digest(path,self.progress,'Verify old history database')!=expected:
            attempt['history']='manifest mismatch; XML payloads used';return
        db=readonly(path)
        try:
            db.execute('SELECT key,payload FROM publications LIMIT 0')
            meta=dict(db.execute('SELECT k,v FROM meta'))
            # Incomplete cached payloads are still safe: every key is independently visited in XML;
            # omissions always fall back. No scan-complete flag is inferred from this database.
            self.history=db;self.report['history_schema']=meta.get('schema','unknown')
            attempt['history']='verified paired payload cache; missing keys fall back to XML'
        except sqlite3.Error:
            db.close();attempt['history']='unsupported schema; XML payloads used'
    def payload(self,key,names):
        if self.history is None:return None
        row=self.history.execute('SELECT payload FROM publications WHERE key=?',(key,)).fetchone()
        if row:
            try:
                r=json.loads(row[0])
                if r.get('key')==key and r.get('authors')==names and all(k in r for k in ('title','year','dois','arxiv_ids','venue')):
                    self.report['history_payload_hits']+=1
                    return {k:r[k] for k in ('key','title','year','authors','dois','arxiv_ids','venue')}
            except (ValueError,TypeError):pass
        self.report['history_payload_misses']+=1;return None
    def close(self):
        if self.history:self.history.close()
