"""Measured phase ETA. Unknown work is never reported as zero remaining."""
import json,time,threading
from datetime import datetime,timezone
from pathlib import Path

class Progress:
    def __init__(self,path,interval=15):
        self.path=Path(path);self.interval=max(1,float(interval));self.lock=threading.RLock()
        self.completed_phases=[];self.started=time.monotonic();self.phase='initializing';self.total=None;self.done=0
        self.phase_start=self.started;self.note='';self.extra={};self.stop=threading.Event()
        self.thread=threading.Thread(target=self._loop,daemon=True);self.thread.start()
    def begin(self,phase,total=None,note=''):
        with self.lock:
            self.completed_phases.append(dict(phase=self.phase,seconds=round(time.monotonic()-self.phase_start,3)))
            self.phase=phase;self.total=total;self.done=0;self.phase_start=time.monotonic();self.note=note;self.extra={}
        self.emit()
    def update(self,done,**extra):
        with self.lock:self.done=done;self.extra.update(extra)
    def snapshot(self):
        with self.lock:
            elapsed=max(0,time.monotonic()-self.phase_start)
            remaining=None
            if self.total is not None and self.done>=self.total:remaining=0
            elif self.total and self.done>0 and elapsed>=5:remaining=elapsed/self.done*(self.total-self.done)
            return dict(completed_phases=self.completed_phases.copy(),phase=self.phase,completed=self.done,total=self.total,phase_elapsed_seconds=round(elapsed,1),
                elapsed_seconds=round(time.monotonic()-self.started,1),estimated_phase_remaining_seconds=None if remaining is None else round(remaining),
                approximate_phase_finish_utc=None if remaining is None else datetime.fromtimestamp(time.time()+remaining,timezone.utc).isoformat(),
                estimate_scope='current phase only; later phases not included',note=self.note,**self.extra)
    def emit(self):
        with self.lock:self._emit_locked()
    def _emit_locked(self):
        s=self.snapshot();tmp=self.path.with_suffix('.tmp');tmp.write_text(json.dumps(s,ensure_ascii=False,indent=2));tmp.replace(self.path)
        eta=s['estimated_phase_remaining_seconds'];label='warming up / unknown' if eta is None else f'~{eta/60:.1f} min'
        if s.get('estimated_run_remaining_seconds') is not None:label+=f" | to export completion ~{s['estimated_run_remaining_seconds']/60:.1f} min"
        print(f"[ETA] {s['phase']}: {s['completed']:,}/{s['total'] if s['total'] is not None else '?'}; phase remaining {label}",flush=True)
    def _loop(self):
        while not self.stop.wait(self.interval):
            try:self.emit()
            except OSError:pass
    def close(self,status='finished'):
        self.stop.set();self.thread.join();self.begin(status,0);self.emit()

def digest(path,progress=None,label='Verify file'):
    import hashlib
    path=Path(path);h=hashlib.sha256();n=0
    if progress:progress.begin(label,path.stat().st_size,'Bytes read; includes Drive I/O.')
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):
            h.update(b);n+=len(b)
            if progress:progress.update(n)
    return h.hexdigest()
