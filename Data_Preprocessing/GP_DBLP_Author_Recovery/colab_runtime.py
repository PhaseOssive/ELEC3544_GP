"""Local staging for large read inputs and SQLite, with cold-cache snapshots.

Google Drive is persistent storage only. XML, input CSVs, and SQLite work copies
are staged to /content before the recovery process starts. Output files/logs stay
on Drive and are written as sequential exports. Never copy an active SQLite DB.
"""
import argparse,contextlib,fcntl,hashlib,json,os,shutil,signal,subprocess,sys,time,uuid
from pathlib import Path

OPTIONAL_FETCH_FILES = (
    'dblp_author_features.csv',
    'dblp_author_features_by_author.csv',
    'dblp_candidate_match_audit.csv',
    'dblp_identity_conflicts.csv',
    'dblp_identity_priority_audit.csv',
    'dblp_author_match_audit.csv',
    'dblp_temporal_exclusion_audit.csv',
    'dblp_source_metadata.json',
)


def announce(message):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), message, flush=True)


def save_json_atomic(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(temp,path)


def copy_verified(src,dst):
    src=Path(src);dst=Path(dst);dst.parent.mkdir(parents=True,exist_ok=True)
    size=src.stat().st_size;before=src.stat();free=shutil.disk_usage(dst.parent).free
    if free<size+64*1024*1024:raise OSError(f'Insufficient space copying {src}: {size} bytes needed')
    temp=dst.with_name(dst.name+'.copying');h=hashlib.sha256();done=0;start=last=time.monotonic()
    announce(f'Copy {src} -> {dst} ({size/1048576:.1f} MiB)')
    with src.open('rb') as a,temp.open('wb') as b:
        while True:
            chunk=a.read(8*1024*1024)
            if not chunk:break
            b.write(chunk);h.update(chunk);done+=len(chunk);now=time.monotonic()
            if now-last>=10:
                announce(f'Copy {done/1048576:.1f}/{size/1048576:.1f} MiB, {done/1048576/max(now-start,.001):.1f} MiB/s')
                last=now
        b.flush();os.fsync(b.fileno())
    after=src.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise RuntimeError('Source changed during copy: '+str(src))
    check=hashlib.sha256()
    with temp.open('rb') as f:
        while True:
            chunk=f.read(8*1024*1024)
            if not chunk:break
            check.update(chunk)
    if check.digest()!=h.digest():raise RuntimeError('Copy checksum mismatch')
    shutil.copystat(src,temp);os.replace(temp,dst)
    announce(f'Copy verified: {dst.name} ({size/1048576:.1f} MiB)')
    return h.hexdigest()


def cache_files(root):
    root=Path(root)
    for name in ('index.sqlite','index.sqlite.building','derived/candidate_history.sqlite'):
        for suffix in ('','-journal','-wal','-shm'):
            f=root/(name+suffix)
            if f.is_file():yield f


@contextlib.contextmanager
def cache_lock(root):
    root=Path(root);(root/'derived').mkdir(parents=True,exist_ok=True)
    handles=[]
    try:
        for name in ('run.lock','profile_build.lock'):
            f=(root/'derived'/name).open('a');handles.append(f)
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise RuntimeError('Recovery still running; stop the old main.py process before copying cache')
        yield
    finally:
        for f in handles:f.close()


def source_cache(persistent):
    pointer=persistent/'runtime_snapshots/LATEST.json'
    if not pointer.exists():return persistent
    name=json.loads(pointer.read_text())['snapshot']
    root=(persistent/'runtime_snapshots'/name).resolve()
    if root.parent!=(persistent/'runtime_snapshots').resolve() or not (root/'snapshot_manifest.json').is_file():
        raise ValueError('Invalid completed cache snapshot pointer')
    return root


def _relative(value, label):
    p=Path(value)
    if p.is_absolute() or '..' in p.parts:
        raise ValueError(f'{label} must be a safe relative path: {value}')
    return p


def _needs_copy(src,dst,state,key):
    src=Path(src);dst=Path(dst)
    st=src.stat();stamp=[str(src.resolve()),st.st_size,st.st_mtime_ns]
    localstamp=[dst.stat().st_size,dst.stat().st_mtime_ns] if dst.is_file() else None
    old=state.get(key,{})
    return old.get('source')!=stamp or old.get('local')!=localstamp


def _stage_file(src,dst,state,statepath,key):
    src=Path(src);dst=Path(dst)
    st=src.stat();stamp=[str(src.resolve()),st.st_size,st.st_mtime_ns]
    localstamp=[dst.stat().st_size,dst.stat().st_mtime_ns] if dst.is_file() else None
    old=state.get(key,{})
    if old.get('source')==stamp and old.get('local')==localstamp:
        announce(f'Reuse verified local copy: {dst.name}')
        return
    digest=copy_verified(src,dst)
    state[key]={'source':stamp,'local':[dst.stat().st_size,dst.stat().st_mtime_ns],'sha256':digest}
    save_json_atomic(statepath,state)


def prepare(config,local_root):
    from config import Config
    config=Config(**dict(config)).validate().snapshot()
    local_root=Path(local_root).resolve();persistent=Path(config['CACHE_DIR']).resolve()
    original_fetch=Path(config['FETCH_DATA_ROOT']).resolve()
    persistent_output=Path(config['OUTPUT_DIR']).resolve()
    for protected in (original_fetch,persistent_output):
        if local_root.is_relative_to(protected) or protected.is_relative_to(local_root):
            raise ValueError('Local work root must be disjoint from Fetch inputs and outputs')
    for protected in (Path(config['DBLP_XML_PATH']),Path(config['DTD_PATH'])):
        if protected.is_relative_to(local_root):raise ValueError('Original XML/DTD must be outside local work root')
    if str(local_root).startswith('/content/drive/'):
        raise ValueError('Local working root must not be on mounted Drive')
    if local_root==persistent or local_root.is_relative_to(persistent) or persistent.is_relative_to(local_root):
        raise ValueError('Working and persistent cache must be disjoint')
    for k,expected in [('PROFILE_INDEX_PATH',persistent/'index.sqlite'),('DERIVED_CACHE_DIR',persistent/'derived')]:
        if config.get(k) and Path(config[k]).resolve()!=expected:
            raise ValueError(k+' must use standard persistent layout for this adapter; core main.py supports custom paths')
    local_root.mkdir(parents=True,exist_ok=True)
    identity=dict(persistent=str(persistent),fetch=str(original_fetch),xml=str(Path(config['DBLP_XML_PATH']).resolve()),dtd=str(Path(config['DTD_PATH']).resolve()))
    owner=local_root/'owner.json'
    if owner.exists() and json.loads(owner.read_text())!=identity:
        raise ValueError('Local workspace belongs to other inputs; use another local_root')
    save_json_atomic(owner,identity)

    local_cache=local_root/'cache';ready=local_root/'cache_ready.json'
    statepath=local_root/'input_copies.json'
    state=json.loads(statepath.read_text()) if statepath.exists() else {}

    # Calculate all first-run copy requirements before starting, to avoid a half-restored VM.
    copy_jobs=[]
    cache_source=None
    if not ready.exists():
        if local_cache.exists():
            raise ValueError('Incomplete local restore; preserve it and choose another local_root')
        cache_source=source_cache(persistent)
        copy_jobs.extend(list(cache_files(cache_source)))
    elif not local_cache.exists():
        raise ValueError('Local cache marker exists but cache directory is missing; choose a fresh local_root')

    xml_sources={}
    for key in ('DBLP_XML_PATH','DTD_PATH'):
        src=Path(config[key]).resolve();dest=local_root/'xml'/src.name
        if not src.is_file():raise FileNotFoundError(f'Missing {key}: {src}')
        if _needs_copy(src,dest,state,key):copy_jobs.append(src)
        xml_sources[key]=(src,dest)

    fetch_root_local=local_root/'fetch'
    required_rel=[_relative(config['master_file'],'master_file'),_relative(config['authors_file'],'authors_file')]
    optional_rel=[_relative(Path(config['dblp_directory'])/name,'dblp_directory') for name in OPTIONAL_FETCH_FILES]
    fetch_sources=[]
    for rel,required in [(x,True) for x in required_rel]+[(x,False) for x in optional_rel]:
        src=original_fetch/rel;dest=fetch_root_local/rel
        key='FETCH:'+rel.as_posix()
        if src.is_file():
            if _needs_copy(src,dest,state,key):copy_jobs.append(src)
            fetch_sources.append((rel,src,dest,key,required))
        elif required:
            raise FileNotFoundError('Required Fetch file missing: '+str(src))
        else:
            if dest.exists():dest.unlink()
            if key in state:
                state.pop(key,None);save_json_atomic(statepath,state)
            announce('Optional Fetch file absent: '+str(src))

    required_bytes=sum(p.stat().st_size for p in copy_jobs)
    free=shutil.disk_usage(local_root).free
    reserve=2*1024**3
    announce(f'Local staging estimate: {required_bytes/1024**3:.2f} GiB to copy; /content free={free/1024**3:.2f} GiB; safety reserve={reserve/1024**3:.1f} GiB')
    if free<required_bytes+reserve:
        raise OSError(f'Insufficient /content disk space before staging: need {required_bytes/1024**3:.2f} GiB plus {reserve/1024**3:.1f} GiB reserve, have {free/1024**3:.2f} GiB')

    if not ready.exists():
        staging=local_root/('restore_'+uuid.uuid4().hex);staging.mkdir()
        recorded=json.loads((cache_source/'snapshot_manifest.json').read_text()) if (cache_source/'snapshot_manifest.json').exists() else None
        with cache_lock(cache_source):
            if recorded is not None and set(recorded)!={str(f.relative_to(cache_source)) for f in cache_files(cache_source)}:
                raise ValueError('Persistent snapshot files missing or unexpected')
            for f in cache_files(cache_source):
                digest=copy_verified(f,staging/f.relative_to(cache_source))
                if recorded is not None and recorded.get(str(f.relative_to(cache_source)))!=digest:
                    raise ValueError('Persistent snapshot checksum mismatch: '+str(f))
        staging.rename(local_cache);save_json_atomic(ready,{'restored_from':str(cache_source)})
        announce('Restored cache from '+str(cache_source))
    else:
        announce('Reusing current VM local cache; persistent copy will not overwrite it')

    for key,(src,dest) in xml_sources.items():
        _stage_file(src,dest,state,statepath,key)
        config[key]=str(dest)

    # Stage all inputs used by bootstrap. This avoids repeated multi-GB reads and hashes on Drive.
    for rel,src,dest,key,required in fetch_sources:
        _stage_file(src,dest,state,statepath,key)
    config['FETCH_DATA_ROOT']=str(fetch_root_local)

    config.update(CACHE_DIR=str(local_cache),PROFILE_INDEX_PATH=str(local_cache/'index.sqlite'),DERIVED_CACHE_DIR=str(local_cache/'derived'))
    path=local_root/'config.local_runtime.json';save_json_atomic(path,config)
    announce('Local input staging complete; main.py will not read CSV/XML/SQLite from mounted Drive.')
    announce('Outputs and logs remain on Drive and are exported sequentially.')
    return path,local_cache,persistent


def snapshot(local_cache,persistent):
    base=Path(persistent)/'runtime_snapshots';base.mkdir(parents=True,exist_ok=True)
    name=time.strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8];temp=base/(name+'.partial');temp.mkdir()
    manifest={}
    with cache_lock(local_cache):
        for f in cache_files(local_cache):
            rel=str(f.relative_to(local_cache));manifest[rel]=copy_verified(f,temp/rel)
    save_json_atomic(temp/'snapshot_manifest.json',manifest)
    temp.rename(base/name)
    ptr=base/('LATEST.'+uuid.uuid4().hex+'.tmp');ptr.write_text(json.dumps({'snapshot':name}));os.replace(ptr,base/'LATEST.json')
    announce('Persistent cache snapshot saved: '+str(base/name));return base/name


def launch(project,config_path,local_root,stage='all'):
    project=Path(project).resolve();config=json.loads(Path(config_path).read_text())
    effective,local,persistent=prepare(config,local_root)
    child=None;exitcode=1
    try:
        child=subprocess.Popen([sys.executable,'-u',str(project/'main.py'),'--config',str(effective),'--stage',stage],cwd=project,start_new_session=True)
        try:exitcode=child.wait()
        except KeyboardInterrupt:
            announce('Interrupt: requesting graceful main.py exit; integrity verification may take time')
            os.killpg(child.pid,signal.SIGINT)
            exitcode=child.wait()
    finally:
        if child is not None and child.poll() is not None:
            announce('main.py exited; saving closed SQLite cache to Drive')
            snapshot(local,persistent)
    if exitcode:raise subprocess.CalledProcessError(exitcode,['main.py','--stage',stage])
    return effective


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--local-root',default='/content/GP_DBLP_Recovery_Work');p.add_argument('--stage',choices=['all','prepare','scan'],default='all');a=p.parse_args()
    launch(Path(__file__).parent,a.config,a.local_root,a.stage)
