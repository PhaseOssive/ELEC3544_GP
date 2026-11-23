import sys,json,tempfile,sqlite3,unittest,copy,contextlib,io
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from main import run,rollback
from common import FIELDS,PUB,AGE,FIRST,read
from external import align
from snapshot import orcid
from audit import work_decision


def oid(seed):
    digits=f'{seed:015d}';n=0
    for d in digits:n=(n+int(d))*2
    c=(12-n%11)%11;s=digits+('X' if c==10 else str(c))
    return '-'.join(s[i:i+4] for i in range(0,16,4))
A,B,C,D,E=[oid(i) for i in range(1,6)]

class FakeCrossref:
    fixtures={}
    def __init__(self,path,cfg):self.db=sqlite3.connect(path);self.requests=0;self.hits=0
    def work(self,doi):
        self.hits+=1
        return copy.deepcopy(self.fixtures.get(doi,{'status':'NOT_FOUND'}))


def dataset(root,authenticate=True,with_normal=False):
    root=Path(root);recovery=root/'recovery'
    for d in ['features','report']:(recovery/d).mkdir(parents=True)
    (root/'empty.dtd').write_text('')
    pubs=[];fixtures={}
    def pub(key,name,person,year,title,external,auth=True,arxiv=None):
        doi='10.9999/'+key
        ae=f'<ee>https://arxiv.org/abs/{arxiv}</ee>' if arxiv else ''
        pubs.append(f'<article key="{key}"><author orcid="{person}">{name}</author><title>{title}</title><year>{year}</year><ee>https://doi.org/{doi}</ee>{ae}</article>')
        plain=name.rsplit(' ',1)[0] if name[-4:].isdigit() else name
        given,family=plain.split(' ',1)
        fixtures[doi.lower()]={'status':'OK','doi':doi,'title':[title],'author':[{'given':given,'family':family,'ORCID':external,'authenticated-orcid':auth}], 'source':'https://example.org/SYNTHETIC/'+key}
    pub('badA','Alex Lee 0001',B,2000,'Old circuits example',B,authenticate)
    pub('a1','Alex Lee 0001',A,2010,'Data structures example',A)
    pub('a2','Alex Lee 0001',A,2012,'Algorithms example',A)
    pub('tA','Alex Lee 0001',A,2020,'Target Alpha',A,arxiv='2001.00001')
    pub('badC','Casey Chen 0001',D,1999,'Old unrelated example',D,authenticate)
    pub('c1','Casey Chen 0001',C,2011,'Learning example',C)
    pub('tC','Casey Chen 0001',C,2020,'Target Gamma',C,arxiv='2002.00002')
    profiles=[('pA','Alex Lee 0001',A),('pB','Alex Lee 0002',B),('pC','Casey Chen 0001',C),('pD','Casey Chen 0002',D)]
    paper_specs=[('2001.00001','Target Alpha','Alex Lee','pA',3,20,'tA'),('2002.00002','Target Gamma','Casey Chen','pC',2,21,'tC')]
    if with_normal:
        profiles.append(('pE','Dana Ray',E))
        for i,y in enumerate([2010,2012,2014]):pub('e'+str(i),'Dana Ray',E,y,'Normal work '+str(i),E)
        pub('tE','Dana Ray',E,2020,'Target Normal',E,arxiv='2003.00003')
        paper_specs.append(('2003.00003','Target Normal','Dana Ray','pE',3,10,'tE'))
    for pid,name,o in profiles:pubs.append(f'<www key="homepages/{pid}"><author>{name}</author><title>Home Page</title><url>https://orcid.org/{o}</url></www>')
    (root/'dblp.xml').write_text('<dblp>'+''.join(pubs)+'</dblp>')
    master=[];authors=[];teams=[]
    for aid,title,name,pid,count,age,key in paper_specs:
        vals=[count,count,age,age]
        master.append(dict(paper_id=aid,title_v1=title,submission_year='2020',author_count_v1='1',authors_v1_json=json.dumps([{'name':name}]),doi_v1='10.9999/'+key,unrelated_label='keep this '+aid,**dict(zip(FIELDS,map(str,vals)))))
        authors.append(dict(arxiv_id=aid,author_position='1',occurrence_id=aid+':1',original_name=name,repaired_name=name,dblp_person_id='homepages/'+pid,identity_status='VERIFIED',t0_year='2020',history_status='COMPLETE_OBSERVED_DBLP_HISTORY',focal_record_keys_json=json.dumps([key]),**{PUB:str(count),AGE:str(age),FIRST:str(2020-age)}))
        teams.append(dict(arxiv_id=aid,expected_author_count='1',**dict(zip(FIELDS,map(str,vals)))))
    for i in range(12):master.append(dict(paper_id=f'filler-{i}',title_v1='Filler',submission_year='2020',author_count_v1='1',authors_v1_json='[{"name":"Filler Person"}]',doi_v1='',unrelated_label='unchanged',**dict(zip(FIELDS,['0']*4))))
    pd.DataFrame(master).to_csv(root/'input.csv',index=False)
    pd.DataFrame(authors).to_csv(recovery/'features/treated_author_features.csv',index=False)
    pd.DataFrame(teams).to_csv(recovery/'features/treated_team_features.csv',index=False)
    (recovery/'report/final_recovery_summary.json').write_text(json.dumps({'full_xml_scan_complete':True,'final_author_occurrences':len(authors)}))
    (root/'rules.json').write_text('[]')
    cfg=dict(input_csv=str(root/'input.csv'),recovery_root=str(recovery),xml_path=str(root/'dblp.xml'),dtd_path=str(root/'empty.dtd'),output_root=str(root/'results'),cache_root=str(root/'cache'),local_work_dir=str(root/'work'),reviewed_rules=str(root/'rules.json'),crossref_enabled=True,author_chunk_size=1,stage_xml_locally=True)
    FakeCrossref.fixtures=fixtures
    return cfg

