import unittest,tempfile,sqlite3,json
from pathlib import Path
from unittest.mock import patch
from fixtures import small,create
from config import Config
from recovery.workflow import run,load
from recovery.search import records as real_records
from recovery.network import calibrate
from recovery.database import Database
import pandas as pd

class AdditionalIntegrityTests(unittest.TestCase):
    def setUp(self):self.t=tempfile.TemporaryDirectory();self.root=Path(self.t.name)
    def tearDown(self):self.t.cleanup()
    def test_interrupted_scan_resumes(self):
        cfg=small(self.root);cfg.checkpoint_records=1
        def interrupted(*args):
            for i,e in enumerate(real_records(*args)):
                if i==2:raise RuntimeError('TEST_INTERRUPTION')
                yield e
        with patch('recovery.search.records',side_effect=interrupted):
            with self.assertRaises(RuntimeError):run(cfg)
        d=sqlite3.connect(Path(cfg.DERIVED_CACHE_DIR)/'candidate_history.sqlite');self.assertEqual(d.execute("SELECT v FROM meta WHERE k='scan_primary_checkpoint'").fetchone()[0],'2');self.assertIsNone(d.execute("SELECT v FROM meta WHERE k='scan_primary_complete'").fetchone());d.close()
        self.assertEqual(run(cfg)['verified_author_count'],3)
    def test_focal_before_t0_and_duplicate_excluded(self):
        rows=[dict(arxiv_id='2001.00001',author_position=1,original_name='Alice Jones',identity_status='VERIFIED',dblp_person_id='a')]
        pubs=[dict(key='focal',title='Target',year=2018,authors=['Alice Jones'],links=['https://arxiv.org/abs/2001.00001','https://doi.org/10.1234/f']),dict(key='focalcopy',title='Target version',year=2017,authors=['Alice Jones'],links=['https://doi.org/10.1234/f']),dict(key='sameyear',title='Other',year=2020,authors=['Alice Jones'],links=[])]
        cfg=create(self.root,[dict(arxiv_id='2001.00001',title_v1='Target')],rows,{'a':['Alice Jones']},pubs);run(cfg)
        a=pd.read_csv(Path(cfg.OUTPUT_DIR)/'features/final_author_features.csv');self.assertEqual(a.dblp_author_publication_count_pre_t0.iloc[0],0);self.assertTrue(a.dblp_author_academic_age_pre_t0.isna().all())
        hist=pd.read_csv(Path(cfg.OUTPUT_DIR)/'history/final_author_publication_history.csv');self.assertNotIn('INCLUDED',hist.decision.tolist())
    def test_ambiguous_publication_name_not_assigned(self):
        cfg=small(self.root);run(cfg);d=sqlite3.connect(Path(cfg.DERIVED_CACHE_DIR)/'candidate_history.sqlite');self.assertEqual(d.execute("SELECT COUNT(*) FROM publication_authors WHERE name='John Smith' AND pid<>''").fetchone()[0],0);d.close()
    def test_new_target_pid_history_second_pass(self):
        # Name absent from aliases, but same token stream can be aligned to two profiles.
        rows=[dict(arxiv_id='2001.00001',author_position=i+1,original_name=n,identity_status='VERIFIED' if i in (0,2) else 'UNRESOLVED',dblp_person_id='a' if i==0 else 'b' if i==2 else '') for i,n in enumerate(['Alice Jones','Keith Briggs Michael Fitch','Bob Green'])]
        names=['Alice Jones','Keith Briggs','Michael Fitch','Bob Green'];profiles={'a':[names[0]],'k':[names[1]],'m':[names[2]],'b':[names[3]]}
        pubs=[dict(key='early',title='Prior independent work',year=2010,authors=names[1:3],links=[]),dict(key='focal',title='Target',year=2020,authors=names,links=['https://arxiv.org/abs/2001.00001'])]
        cfg=create(self.root,[dict(arxiv_id='2001.00001',title_v1='Target')],rows,profiles,pubs);s=run(cfg);self.assertEqual(s['verified_author_count'],4)
        d=sqlite3.connect(Path(cfg.DERIVED_CACHE_DIR)/'candidate_history.sqlite');self.assertEqual(d.execute("SELECT v FROM meta WHERE k='scan_discovered_profiles_complete'").fetchone()[0],'1');self.assertEqual(d.execute("SELECT COUNT(*) FROM candidate_publications WHERE key='early'").fetchone()[0],2);d.close()
    def test_calibration_holdout_enables_only_supported_rule(self):
        cfg=Config(OUTPUT_DIR=str(self.root/'output'));cfg.OUTPUT_DIR=str(Path(cfg.OUTPUT_DIR).resolve());cfg.FETCH_DATA_ROOT=str(self.root/'prior');cfg.CACHE_DIR=str(self.root/'cache')
        store=Database(self.root/'test.sqlite')
        for i in range(1800):
            oid=str(i);pid='p'+oid;store.db.execute('INSERT INTO occurrences VALUES (?,?,?,?,?,?,?,?)',(oid,'paper'+oid,1,'Some Name',pid,2020,1,'{}'))
            store.db.execute('INSERT INTO scores VALUES (?,?,?,?,?,?,?,?,?,?)',(oid,pid,4,2,5,1.,1.,0.,10.,1))
            store.db.execute('INSERT INTO scores VALUES (?,?,?,?,?,?,?,?,?,?)',(oid,'wrong'+oid,3,0,0,0.,0.,0.,0.,1))
        store.db.commit();report=calibrate(cfg,store);self.assertTrue(report['automatic_network_confirmation_enabled']);self.assertGreaterEqual(report['holdout_wilson_lower_95'],.99)
        store.db.execute('UPDATE scores SET score=100,common=2,joint=5 WHERE pid LIKE "wrong%"');store.db.commit();self.assertFalse(calibrate(cfg,store)['automatic_network_confirmation_enabled']);store.db.close()
    def test_fallback_preserves_reason(self):
        cfg=create(self.root,[dict(arxiv_id='x',title_v1='No record')],[dict(arxiv_id='x',author_position=1,original_name='Unknown Person',identity_status='UNRESOLVED')],{'a':['Other Name']},[])
        run(cfg);queue=pd.read_csv(Path(cfg.OUTPUT_DIR)/'fallback/semantic_scholar_fallback_queue.csv')
        self.assertEqual(queue.final_failure_reason.iloc[0],'INSUFFICIENT_VALIDATED_IDENTITY_EVIDENCE')
    def test_missing_year_not_zero(self):
        cfg=create(self.root,[dict(arxiv_id='x',title_v1='Target')],[dict(arxiv_id='x',author_position=1,original_name='Alice Jones',identity_status='VERIFIED',dblp_person_id='a')],{'a':['Alice Jones']},[dict(key='unknown',title='Unknown year',year=None,authors=['Alice Jones'],links=[])])
        run(cfg);a=pd.read_csv(Path(cfg.OUTPUT_DIR)/'features/final_author_features.csv');self.assertTrue(a.dblp_author_publication_count_pre_t0.isna().all())
if __name__=='__main__':unittest.main()
