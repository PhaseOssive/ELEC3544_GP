"""Use tqdm when installed; otherwise timestamped rates without dependencies."""
import time,logging
try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self,iterable=None,total=None,desc='',unit='items',**kwargs):
            self.iterable=iterable;self.total=total;self.desc=desc;self.unit=unit;self.n=0;self.start=time.monotonic();self.last=self.start;self.extra={}
            logging.info('%s started',desc)
        def __iter__(self):
            try:
                for item in self.iterable:yield item;self.update(1)
            finally:self.close()
        def update(self,n):
            self.n+=n
            if time.monotonic()-self.last>=5:self.report()
        def report(self):
            self.last=time.monotonic();elapsed=self.last-self.start
            logging.info('%s processed=%d %s speed=%.1f/s elapsed=%.1fs %s',self.desc,self.n,self.unit,self.n/max(.001,elapsed),elapsed,self.extra)
        def set_postfix(self,refresh=False,**kwargs):self.extra.update(kwargs)
        def close(self):self.report()
        def __enter__(self):return self
        def __exit__(self,*args):self.close()
