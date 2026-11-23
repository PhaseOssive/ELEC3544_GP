import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from fixtures import small,create
from recovery.workflow import run as recover
from recovery.features import aggregate,FEATURES
from recovery.history import PUB,AGE,FIRST
from missing_treatment import apply_treatment,run,ZERO,COMPLETE
from common import sha256


def row(pos=1,status=ZERO,verified=True,count=0,age='',first=''):
    return dict(arxiv_id='x',author_position=pos,identity_status='VERIFIED' if verified else 'UNRESOLVED',dblp_person_id=str(pos) if verified else '',dblp_author_matched_to_paper=verified,history_status=status,**{PUB:count,AGE:age,FIRST:first})

class MissingTreatmentTests(unittest.TestCase):
    def test_certified_zero_count_and_age(self):
        a=apply_treatment(pd.DataFrame([row()]));self.assertEqual(a[PUB].iloc[0],0);self.assertEqual(a[AGE].iloc[0],0);self.assertEqual(a[FIRST].iloc[0],'')
    def test_unresolved_not_imputed_even_if_numeric(self):
        a=apply_treatment(pd.DataFrame([row(verified=False,count=7,age=4)]));self.assertTrue(a[[PUB,AGE]].isna().all().all())
    def test_missing_year_and_unverified_history_not_zero(self):
        for status in ['MISSING_YEAR','NOT_SCANNED','NO_OBSERVED_RECORDS','AMBIGUOUS_ALIAS_COVERAGE']:
            with self.subTest(status=status):
                a=apply_treatment(pd.DataFrame([row(status=status)]));self.assertTrue(a[[PUB,AGE]].isna().all().all());self.assertIn(status,a.missing_treatment_reason.iloc[0])
    def test_inconsistent_zero_evidence_not_imputed(self):
        for r in [row(count=''),row(count=1),row(first=2015),row(first='unverifiable-year')]:
            a=apply_treatment(pd.DataFrame([r]));self.assertTrue(a[[PUB,AGE]].isna().all().all())
    def test_zero_and_nonzero_team_mean_max(self):
        a=apply_treatment(pd.DataFrame([row(),row(2,COMPLETE,count=3,age=8,first=2012)]))
        t=aggregate(pd.DataFrame([dict(arxiv_id='x',expected_author_count=2)]),a)
        self.assertEqual(t[FEATURES].iloc[0].tolist(),[1.5,3.,4.,8.])
    def test_all_zero_team_has_four_zero_features(self):
        a=apply_treatment(pd.DataFrame([row(),row(2)]));t=aggregate(pd.DataFrame([dict(arxiv_id='x',expected_author_count=2)]),a);self.assertEqual(t[FEATURES].iloc[0].tolist(),[0.]*4)
    def test_one_unresolved_keeps_all_full_team_features_missing(self):
        a=apply_treatment(pd.DataFrame([row(),row(2,verified=False)]));t=aggregate(pd.DataFrame([dict(arxiv_id='x',expected_author_count=2)]),a);self.assertTrue(t[FEATURES].isna().all().all())
    def test_no_xml_sqlite_access_and_master_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=small(Path(tmp));recover(cfg)
            master=Path(cfg.FETCH_DATA_ROOT)/cfg.master_file;before=pd.read_csv(master,dtype=str,keep_default_na=False)
            cache=[Path(cfg.PROFILE_INDEX_PATH),Path(cfg.DERIVED_CACHE_DIR)/'candidate_history.sqlite'];digest={str(p):sha256(p) for p in cache}
            Path(cfg.DBLP_XML_PATH).unlink()
            with patch('sqlite3.connect',side_effect=AssertionError('SQLite accessed')),patch('recovery.search.records',side_effect=AssertionError('XML scanned')):
                s=run(cfg)
            after=pd.read_csv(Path(cfg.OUTPUT_DIR)/'features/papers_master_dblp_imputed.csv',dtype=str,keep_default_na=False)
            self.assertTrue(after[before.columns].equals(before));self.assertEqual(s['retained_paper_count'],1)
            self.assertEqual(digest,{str(p):sha256(p) for p in cache})
    def test_repeat_treatment_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=small(Path(tmp));recover(cfg);a=run(cfg);p=Path(cfg.OUTPUT_DIR)/'features/papers_master_dblp_imputed.csv';h=sha256(p);b=run(cfg);self.assertEqual(a,b);self.assertEqual(h,sha256(p))
    def test_prepare_only_not_treated(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=small(Path(tmp));recover(cfg,'prepare')
            with self.assertRaises((FileNotFoundError,ValueError)):run(cfg)
            self.assertFalse((Path(cfg.OUTPUT_DIR)/'features/papers_master_dblp_imputed.csv').exists())
    def test_original_master_mismatch_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=small(Path(tmp));recover(cfg);p=Path(cfg.FETCH_DATA_ROOT)/cfg.master_file;p.write_text(p.read_text()+'\n')
            with self.assertRaises(ValueError):run(cfg)
    def test_failed_temporal_audit_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=small(Path(tmp));recover(cfg);p=Path(cfg.OUTPUT_DIR)/'report/temporal_integrity_audit.json';v=json.loads(p.read_text());v['included_focal_works']=1;p.write_text(json.dumps(v))
            with self.assertRaises(ValueError):run(cfg)
    def test_author_team_export_mismatch_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=small(Path(tmp));recover(cfg);p=Path(cfg.OUTPUT_DIR)/'features/final_team_features.csv';t=pd.read_csv(p);t[FEATURES[0]]=99;t.to_csv(p,index=False)
            with self.assertRaises(ValueError):run(cfg)
