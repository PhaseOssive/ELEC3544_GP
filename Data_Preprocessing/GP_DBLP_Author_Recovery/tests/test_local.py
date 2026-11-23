import unittest,tempfile,sqlite3,json,socket,random,gzip
from pathlib import Path
from unittest.mock import patch
from fixtures import small,create
from config import Config
from common import offline,sha256
from recovery.workflow import run,load
from recovery.profile import ProfileIndex
from recovery.database import Database
from recovery.alignment import align
from recovery.rules import HistoricalWorkDeduplicator
from recovery.dedup import DiskDeduplicator
from recovery.history import PUB,AGE
from recovery.network import wilson
import pandas as pd

class LocalTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def test_offline_enforcement(self):
        with offline() as state:
            with self.assertRaises(RuntimeError):socket.create_connection(('example.com',443))
        self.assertGreater(state['network_attempts_blocked'],0)
    def test_missing_cache_no_rebuild(self):
        with self.assertRaises(FileNotFoundError):ProfileIndex(self.root/'missing.sqlite')
        self.assertFalse((self.root/'missing.sqlite').exists())
    def test_full_run_and_idempotence(self):
        cfg=small(self.root);source=sha256(cfg.PROFILE_INDEX_PATH);summary=run(cfg)
        self.assertEqual(summary['verified_author_count'],3);self.assertEqual(summary['network_attempts_blocked'],0);self.assertEqual(source,sha256(cfg.PROFILE_INDEX_PATH))
        a=pd.read_csv(Path(cfg.OUTPUT_DIR)/'features/final_author_features.csv');self.assertEqual(a[PUB].iloc[0],1);self.assertEqual(a[AGE].iloc[0],5)
        # Shared ambiguous raw alias blocks complete historical counts, despite verified identity.
        self.assertTrue(pd.isna(a[PUB].iloc[2]));self.assertEqual(a.dblp_person_id.tolist(),['a','b','c'])
        db=sqlite3.connect(Path(cfg.DERIVED_CACHE_DIR)/'candidate_history.sqlite');before=db.execute('SELECT COUNT(*) FROM candidate_publications').fetchone()[0];db.close()
        run(cfg);db=sqlite3.connect(Path(cfg.DERIVED_CACHE_DIR)/'candidate_history.sqlite');self.assertEqual(before,db.execute('SELECT COUNT(*) FROM candidate_publications').fetchone()[0]);db.close()
        preview=pd.read_csv(Path(cfg.OUTPUT_DIR)/'features/final_papers_master_preview.csv',dtype=str);self.assertEqual(preview.Y1.iloc[0],'001')
    def test_no_xml_fails_explicitly(self):
        cfg=small(self.root);Path(cfg.DBLP_XML_PATH).unlink()
        with self.assertRaises(FileNotFoundError):run(cfg)
        report=json.loads((Path(cfg.OUTPUT_DIR)/'report/final_recovery_summary.json').read_text());self.assertEqual(report['status'],'BLOCKED_MISSING_INPUT')
    def test_not_top_ten_and_initial_rejection(self):
        names={str(i):['John Smith'] for i in range(25)};names['other']=['James Smith']
        cfg=create(self.root,[dict(arxiv_id='x',title_v1='X')],[dict(arxiv_id='x',author_position=1,original_name='John Smith',identity_status='UNRESOLVED')],names,[])
        p=ProfileIndex(cfg.PROFILE_INDEX_PATH);expanded=list(p.expand('John Smith'));self.assertEqual(len({r[0] for r in expanded if r[4]}),25);self.assertTrue(any(r[0]=='other' and not r[4] for r in expanded));p.db.close()
    def test_zero_vs_unobserved(self):
        profiles={'a':['Alice Jones'],'b':['Bob Green']};rows=[dict(arxiv_id='x',author_position=i+1,original_name=n,identity_status='VERIFIED',dblp_person_id=p) for i,(p,n) in enumerate([('a','Alice Jones'),('b','Bob Green')])]
        cfg=create(self.root,[dict(arxiv_id='x',title_v1='Target')],rows,profiles,[dict(key='focal',title='Target',year=2020,authors=['Alice Jones','Bob Green'],links=[])])
        run(cfg);a=pd.read_csv(Path(cfg.OUTPUT_DIR)/'features/final_author_features.csv');self.assertEqual(a[PUB].tolist(),[0,0]);self.assertTrue(a[AGE].isna().all())
    def test_future_coauthor_not_used(self):
        cfg=small(self.root);run(cfg);d=sqlite3.connect(Path(cfg.DERIVED_CACHE_DIR)/'candidate_history.sqlite');self.assertEqual(d.execute('SELECT COUNT(*) FROM candidate_coauthors WHERE year>=2020').fetchone()[0],0);self.assertEqual(d.execute("SELECT COUNT(*) FROM candidate_coauthors WHERE key='target'").fetchone()[0],0);d.close()
    def test_dtd_no_pid_assumption(self):
        cfg=small(self.root);run(cfg,'prepare');a=json.loads((Path(cfg.OUTPUT_DIR)/'report/dtd_association_audit.json').read_text());self.assertFalse(a['has_direct_pid_attribute'])
    def test_path_guards(self):
        cfg=small(self.root);cfg.OUTPUT_DIR=cfg.CACHE_DIR
        with self.assertRaises(ValueError):cfg.validate()
    def test_verified_anchor_conflict_preserved(self):
        cfg=small(self.root)
        # Reverse known anchor positions in a target: sequence cannot rewrite verified people.
        x=gzip.open(cfg.DBLP_XML_PATH,'rt').read().replace('<author>Alice Jones</author><author>Bob Green</author>','<author>Bob Green</author><author>Alice Jones</author>')
        with gzip.open(cfg.DBLP_XML_PATH,'wt') as f:f.write(x)
        from common import fingerprint
        db=sqlite3.connect(cfg.PROFILE_INDEX_PATH);s=json.loads(db.execute("SELECT v FROM meta WHERE k='signature'").fetchone()[0]);s[1]=fingerprint(cfg.DBLP_XML_PATH);db.execute("UPDATE meta SET v=? WHERE k='signature'",(json.dumps(s),));db.commit();db.close()
        run(cfg);a=pd.read_csv(Path(cfg.OUTPUT_DIR)/'matching/final_author_identity_mapping.csv');self.assertEqual(a.dblp_person_id.iloc[:2].tolist(),['a','b'])
    def test_precision_gate_not_faked(self):
        self.assertLess(wilson(100,100),.99);self.assertGreater(wilson(1000,1000),.99)
        cfg=small(self.root);s=run(cfg);self.assertFalse(s['backtest']['automatic_network_confirmation_enabled'])

class AlignmentTests(unittest.TestCase):
    def check(self,source,target,ops):
        rows=[dict(original_name=n,repaired_name=n,author_position=i+1,occurrence_id=str(i),identity_status='VERIFIED' if i in (0,len(source)-1) else 'UNRESOLVED',dblp_person_id='left' if i==0 else 'right' if i==len(source)-1 else '') for i,n in enumerate(source)]
        ids=['left']+[str(i) for i in range(len(target)-2)]+['right'];cfg=Config(alignment_margin=.01)
        r=align(rows,target,ids,cfg,True);self.assertTrue(r['accepted'],r);self.assertTrue(set(ops)<=set(x['alignment_operation'] for x in r['evidence']))
    def test_split(self):self.check(['Alice Jones','Keith Briggs Michael Fitch','Bob Green'],['Alice Jones','Keith Briggs','Michael Fitch','Bob Green'],['SPLIT'])
    def test_merge(self):self.check(['Alice Jones','John','Smith','Bob Green'],['Alice Jones','John Smith','Bob Green'],['MERGE'])
    def test_token_shift(self):self.check(['Alice Jones','Haeyoung Lee Klaus','Moessner','Bob Green'],['Alice Jones','Haeyoung Lee','Klaus Moessner','Bob Green'],['TOKEN_REALLOCATION'])
    def test_missing_middle(self):self.check(['Alice Jones','Kazi Saidul Huq','Bob Green'],['Alice Jones','Kazi Mohammed Saidul Huq','Bob Green'],['ONE_TO_ONE'])
    def test_name_order(self):self.check(['Alice Jones','Smith John','Bob Green'],['Alice Jones','John Smith','Bob Green'],['ONE_TO_ONE'])
    def test_spelling(self):self.check(['Alice Jones','Oscar Carasco','Bob Green'],['Alice Jones','Oscar Carrasco','Bob Green'],['ONE_TO_ONE'])
    def test_no_name_only_accept(self):
        r=align([dict(original_name='Oscar Carasco',identity_status='UNRESOLVED',dblp_person_id='',occurrence_id='x',author_position=1)],['Oscar Carrasco'],['p'],Config(),False);self.assertFalse(r['accepted'])

class DedupParityTests(unittest.TestCase):
    def test_disk_matches_original_randomized(self):
        for seed in range(10):
            rng=random.Random(seed);d=HistoricalWorkDeduplicator();db=sqlite3.connect(':memory:');s=DiskDeduplicator(db)
            for i in range(120):
                strong={f'doi:{rng.randrange(20)}'} if rng.random()<.7 else set();fallback=f'f:{rng.randrange(15)}';pids={str(rng.randrange(5))};year=rng.randrange(1990,2025)
                d.add_record(strong,fallback,pids,year,str(i));s.add_record(strong,fallback,pids,year,str(i))
            expected=d.finalize();s.finalize();actual={}
            for w,pid,y in db.execute('SELECT work,pid,year FROM work_years'):actual.setdefault(w,{})[pid]=y
            self.assertEqual(sorted(sorted(v.items()) for v in expected[4].values()),sorted(sorted(v.items()) for v in actual.values()))
            mapping=dict(db.execute('SELECT key,work FROM record_works'))
            for a in range(120):
                for b in range(a):self.assertEqual(expected[5][str(a)]==expected[5][str(b)],mapping[str(a)]==mapping[str(b)])
            db.close()
if __name__=='__main__':unittest.main()
