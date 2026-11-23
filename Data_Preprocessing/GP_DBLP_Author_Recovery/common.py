"""Offline enforcement and auditable atomic, streaming exports."""
import sys,json,csv,hashlib,logging,time
from pathlib import Path
from contextlib import contextmanager

@contextmanager
def offline():
    state={'active':True,'network_attempts_blocked':0,'subprocess_attempts_blocked':0}
    def hook(event,args):
        if not state['active']:return
        if event in ('socket.connect','socket.connect_ex','socket.getaddrinfo','socket.sendto','socket.sendmsg','socket.bind'):
            state['network_attempts_blocked']+=1;raise RuntimeError('OFFLINE_POLICY: '+event)
        if event in ('subprocess.Popen','os.system','os.posix_spawn'):
            state['subprocess_attempts_blocked']+=1;raise RuntimeError('OFFLINE_POLICY: subprocess forbidden')
    sys.addaudithook(hook)
    try:yield state
    finally:state['active']=False

def sha256(path):
    h=hashlib.sha256();size=Path(path).stat().st_size;done=0;start=last=time.monotonic()
    visible=size>=64*1024*1024
    if visible:logging.info('SHA256 start %s (%.1f MiB)',path,size/1048576)
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):
            h.update(b);done+=len(b);now=time.monotonic()
            if visible and now-last>=10:
                logging.info('SHA256 %s %.1f/%.1f MiB %.1f MiB/s',path,done/1048576,size/1048576,done/1048576/max(now-start,.001));last=now
    if visible:logging.info('SHA256 complete %s %.2fs',path,time.monotonic()-start)
    return h.hexdigest()

def fingerprint(path):
    p=Path(path);h=hashlib.sha256()
    with p.open('rb') as f:h.update(f.read(1048576));f.seek(max(0,p.stat().st_size-1048576));h.update(f.read())
    return dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,sample_sha256=h.hexdigest())

def export_query(cfg,name,db,query,args=()):
    logging.info('Export start: %s',name)
    p=cfg.out(name);tmp=p.with_suffix(p.suffix+'.tmp');cur=db.execute(query,args)
    with tmp.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow([d[0] for d in cur.description]);w.writerows(cur)
    tmp.replace(p)
    logging.info('Export complete: %s (%.1f MiB)',name,p.stat().st_size/1048576)
