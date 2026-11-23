"""Read-only adapter to the existing profile index. No rebuild path."""
import sqlite3,json
from pathlib import Path
from functools import lru_cache
from .rules import normalize_name,name_signature
from .names import base_name,tokens,compatible

class ProfileIndex:
    def __init__(self,path):
        self.path=Path(path).resolve()
        if not self.path.is_file():raise FileNotFoundError('Existing PROFILE_INDEX_PATH missing: '+str(path))
        self.db=sqlite3.connect(self.path.as_uri()+'?mode=ro',uri=True);self.db.execute('PRAGMA query_only=ON');self.db.execute('PRAGMA cache_size=-65536')
        for table,cols in [('meta',{'k','v'}),('profiles',{'pid','redirect','valid'}),('aliases',{'pid','raw','norm','sig'})]:
            actual={r[1] for r in self.db.execute('PRAGMA table_info('+table+')')}
            if not cols<=actual:raise ValueError('Incompatible existing profile schema: '+table)
        if self.db.execute('PRAGMA quick_check').fetchone()[0]!='ok':raise ValueError('Profile index corrupt')
        if self.get('complete')!='1':raise ValueError('Profile scan incomplete; refusing rebuild')
        self.signature=json.loads(self.get('signature'))
        if self.signature[0] not in ('recovery-sqlite-1','dblp-profile-index-1'):raise ValueError('Unknown source schema')
        self.reverse={}
        for old,new in self.db.execute("SELECT pid,redirect FROM profiles WHERE redirect<>''"):self.reverse.setdefault(new,[]).append(old)
    def get(self,k):
        r=self.db.execute('SELECT v FROM meta WHERE k=?',(k,)).fetchone();return r[0] if r else None
    @lru_cache(maxsize=100000)
    def canonical(self,pid):
        seen=set()
        while pid and pid not in seen:
            seen.add(pid);r=self.db.execute('SELECT redirect,valid FROM profiles WHERE pid=?',(pid,)).fetchone()
            if not r or not r[1]:return None
            if not r[0]:return pid
            pid=r[0]
        return None
    @lru_cache(maxsize=50000)
    def raw_ids(self,name):
        return frozenset(p for (v,) in self.db.execute('SELECT pid FROM aliases WHERE raw=?',(name,)) if (p:=self.canonical(v)))
    def aliases(self,pid):
        todo=[pid];seen=set()
        while todo:
            x=todo.pop()
            if x in seen:continue
            seen.add(x)
            yield from (r[0] for r in self.db.execute('SELECT raw FROM aliases WHERE pid=?',(x,)))
            todo.extend(self.reverse.get(x,()))
    def unique(self,name):
        ids=self.raw_ids(name)
        if len(ids)==1:return next(iter(ids))
        if ids:return None
        ids={self.canonical(r[0]) for r in self.db.execute('SELECT pid FROM aliases WHERE norm=?',(normalize_name(name),))};ids.discard(None)
        return next(iter(ids)) if len(ids)==1 else None
    def expand(self,name):
        """Stream all candidates and their rejected/eligible evidence; never LIMIT."""
        original=base_name(name);variants={original}
        ts=tokens(original)
        if len(ts)>1:variants.update([' '.join(ts[1:]+ts[:1]),' '.join(ts[-1:]+ts[:-1])])
        for v in sorted(variants):
            for col,val,method,rank in [('raw',v,'RAW_ALIAS',4),('norm',normalize_name(v),'NORMALIZED_ALIAS' if v==original else 'NAME_ORDER',3),('sig',name_signature(v),'INITIAL_EXPANSION',1)]:
                if not val:continue
                for pid,alias in self.db.execute('SELECT pid,raw FROM aliases WHERE '+col+'=?',(val,)):
                    canonical=self.canonical(pid)
                    if not canonical:continue
                    rel=compatible(original,alias)
                    eligible=bool(rel)
                    grade=rank
                    if col=='sig' and rel and rel!='GIVEN_NAME_OR_INITIAL_EXPANSION':grade=3
                    # Full given name mismatch remains explicitly rejected, never hidden.
                    yield canonical,alias,method,grade,int(eligible),'' if eligible else 'FULL_GIVEN_NAME_OR_SURNAME_CONFLICT'

    def tier(self,name):
        tiers=[]
        for col,val in [('raw',name.strip()),('norm',normalize_name(name)),('sig',name_signature(name))]:
            ids={self.canonical(r[0]) for r in self.db.execute('SELECT pid FROM aliases WHERE '+col+'=?',(val,))};ids.discard(None);tiers.append(ids)
        allids=set().union(*tiers)
        for ids,method in zip(tiers,['RAW_EXACT_ALIAS','NORMALIZED_FULL_NAME_ALIAS','UNIQUE_INITIAL_SURNAME_ALIAS']):
            if ids:return ids,method,allids-ids
        return set(),'NO_DBLP_PROFILE_MATCH',set()
    def unique_full(self,name):
        ids,method,_=self.tier(name)
        return next(iter(ids)) if len(ids)==1 and method!='UNIQUE_INITIAL_SURNAME_ALIAS' else None
