"""Offline fixture builder; synthetic publications are labeled and isolated."""
import json,sqlite3,gzip,sys
from pathlib import Path
from xml.sax.saxutils import escape
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import Config
from common import fingerprint
from recovery.rules import normalize_name,name_signature

DTD='''<!ELEMENT dblp ANY><!ELEMENT article ANY><!ELEMENT www ANY><!ELEMENT author (#PCDATA)><!ATTLIST author orcid CDATA #IMPLIED><!ELEMENT title (#PCDATA)><!ELEMENT year (#PCDATA)><!ELEMENT ee (#PCDATA)>'''
def create(root,papers,authors,profiles,publications):
    root=Path(root);cache=root/'cache';cache.mkdir(parents=True,exist_ok=True);prior=root/'fetch';processed=prior/'data/processed';dbdir=processed/'dblp_author_features_v3_1_1';dbdir.mkdir(parents=True,exist_ok=True)
    (cache/'dblp.dtd').write_text(DTD)
    db=sqlite3.connect(cache/'index.sqlite');db.executescript('''CREATE TABLE meta(k TEXT PRIMARY KEY,v TEXT);CREATE TABLE profiles(pid TEXT PRIMARY KEY,redirect TEXT,valid INTEGER);CREATE TABLE aliases(pid TEXT,raw TEXT,norm TEXT,sig TEXT,PRIMARY KEY(pid,raw));CREATE INDEX raw_index ON aliases(raw);CREATE INDEX norm_index ON aliases(norm);CREATE INDEX sig_index ON aliases(sig);''')
    for pid,names in profiles.items():
        db.execute('INSERT INTO profiles VALUES (?,?,?)',(pid,'',1))
        for name in names:db.execute('INSERT INTO aliases VALUES (?,?,?,?)',(pid,name,normalize_name(name),name_signature(name)))
    xml=root/'dblp.xml.gz'
    with gzip.open(xml,'wt') as f:
        f.write('<?xml version="1.0"?><!DOCTYPE dblp SYSTEM "dblp.dtd"><dblp>')
        for pid,names in profiles.items():
            f.write('<www key="'+pid+'">'+''.join('<author>'+escape(n)+'</author>' for n in names)+'<title>Home Page</title></www>')
        for r in publications:
            f.write('<article key="'+r['key']+'">'+''.join('<author>'+escape(n)+'</author>' for n in r['authors'])+'<title>'+escape(r['title'])+'</title>'+('<year>'+str(r['year'])+'</year>' if r.get('year') else '')+''.join('<ee>'+escape(v)+'</ee>' for v in r.get('links',[]))+'</article>')
        f.write('</dblp>')
    db.executemany('INSERT INTO meta VALUES (?,?)',[('complete','1'),('signature',json.dumps(['recovery-sqlite-1',fingerprint(xml),fingerprint(cache/'dblp.dtd'),papers]))]);db.commit();db.close()
    import pandas as pd
    records=[]
    for a in authors:
        r=dict(a);r.setdefault('occurrence_id',r['arxiv_id']+':'+str(r['author_position']));r.setdefault('t0_year',2020);r.setdefault('repaired_name',r['original_name']);r.setdefault('focal_record_keys_json','[]');r.setdefault('dblp_identity_confidence','HIGH' if r['identity_status']=='VERIFIED' else 'UNRESOLVED');r.setdefault('dblp_identity_method','SOURCE_VERIFIED');r.setdefault('evidence_json','{}');r.setdefault('dblp_person_id','');records.append(r)
    from collections import defaultdict
    grouped=defaultdict(list)
    for r in records:grouped[r['arxiv_id']].append(r)
    for p in papers:
        p.setdefault('submission_date_v1','2020-01-01');p.setdefault('author_count_v1',len(grouped[p['arxiv_id']]));p.setdefault('authors_v1_json',json.dumps([{'name':r['original_name']} for r in grouped[p['arxiv_id']]]));p.setdefault('Y1','001');p.setdefault('Y2','KEEP');p.setdefault('Y3','')
    pd.DataFrame([dict(arxiv_id=r['arxiv_id'],author_position=r['author_position'],author_name_raw=r['original_name']) for r in records]).to_csv(processed/'authors_arxiv_v1.csv',index=False)
    details=[dict(arxiv_id=r['arxiv_id'],author_position=r['author_position'],arxiv_author_name_v1=r['original_name'],dblp_person_id=r['dblp_person_id'],dblp_author_matched_to_paper=r['identity_status']=='VERIFIED',dblp_identity_confidence=r['dblp_identity_confidence'],dblp_identity_method=r['dblp_identity_method'],t0_year=r['t0_year'],dblp_record_key=next(iter(json.loads(r['focal_record_keys_json'])),'') ) for r in records]
    pd.DataFrame(details).to_csv(dbdir/'dblp_author_features_by_author.csv',index=False)
    pd.DataFrame(papers).to_csv(processed/'papers_master_raw_stage.csv',index=False)
    return Config(FETCH_DATA_ROOT=str(prior),CACHE_DIR=str(cache),DBLP_XML_PATH=str(xml),OUTPUT_DIR=str(root/'out')).validate()

def small(root):
    profiles={'a':['Alice Jones'],'b':['Bob Green'],'c':['John Smith 0001','John Smith'],'d':['John Smith 0002','John Smith']}
    rows=[dict(arxiv_id='2001.00001',author_position=i+1,original_name=n,identity_status='VERIFIED' if i<2 else 'UNRESOLVED',dblp_person_id=['a','b',''][i]) for i,n in enumerate(['Alice Jones','Bob Green','John Smith'])]
    pubs=[dict(key='target',title='The target paper',year=2020,authors=['Alice Jones','Bob Green','John Smith 0001'],links=['https://arxiv.org/abs/2001.00001']),dict(key='past',title='Independent old work',year=2015,authors=['Alice Jones','Bob Green','John Smith 0001'],links=['https://doi.org/10.1234/past']),dict(key='duplicate',title='Another version',year=2016,authors=['Alice Jones','Bob Green','John Smith 0001'],links=['https://doi.org/10.1234/past']),dict(key='future',title='Future cooperation',year=2022,authors=['Alice Jones','John Smith 0002'],links=[])]
    return create(root,[dict(arxiv_id='2001.00001',title_v1='The target paper')],rows,profiles,pubs)
