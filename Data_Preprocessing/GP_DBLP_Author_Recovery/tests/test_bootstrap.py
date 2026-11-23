import unittest,tempfile,json,sqlite3,gzip
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from fixtures import create,small
from config import Config
from common import sha256
from recovery.workflow import run
from recovery.bootstrap import load
from recovery.index_init import ensure_index
from recovery.search import records as real_records

class ArchitectureTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def fresh(self):
        rows=[dict(arxiv_id='2001.00001',author_position=1,original_name='Alice Jones',identity_status='VERIFIED',dblp_person_id='homepages/a')]
        pubs=[dict(key='old',title='Previous work',year=2010,authors=['Alice Jones'],links=[]),dict(key='focal',title='Target',year=2020,authors=['Alice Jones'],links=['https://arxiv.org/abs/2001.00001'])]
        cfg=create(self.root,[dict(arxiv_id='2001.00001',title_v1='Target')],rows,{'homepages/a':['Alice Jones']},pubs);Path(cfg.PROFILE_INDEX_PATH).unlink();return cfg
    def test_first_run_builds_and_reuses_no_recovery_inputs(self):
        cfg=self.fresh();files=list(Path(cfg.FETCH_DATA_ROOT).rglob('*'));self.assertFalse(any('recovered_' in p.name or 'preview' in p.name for p in files))
        before={str(p):sha256(p) for p in files if p.is_file()};s=run(cfg);self.assertTrue(s['profile_index_created']);self.assertEqual(s['verified_author_count'],1)
        old=sha256(cfg.PROFILE_INDEX_PATH);s=run(cfg);self.assertFalse(s['profile_index_created']);self.assertEqual(old,sha256(cfg.PROFILE_INDEX_PATH));self.assertEqual(before,{p:sha256(p) for p in before})
    def test_missing_optional_files_still_bootstraps(self):
        cfg=self.fresh();detail=Path(cfg.FETCH_DATA_ROOT)/cfg.dblp_directory/'dblp_author_features_by_author.csv';detail.unlink()
        s=run(cfg);self.assertEqual(s['verified_author_count'],1);self.assertEqual(s['fetch_claimed_verified'],0)
    def test_missing_required_file_clear_error(self):
        cfg=self.fresh();(Path(cfg.FETCH_DATA_ROOT)/cfg.authors_file).unlink()
        with self.assertRaisesRegex(FileNotFoundError,'Required Fetch'):run(cfg)
    def test_existing_incompatible_index_not_overwritten(self):
        cfg=self.fresh();Path(cfg.PROFILE_INDEX_PATH).write_bytes(b'not sqlite');before=sha256(cfg.PROFILE_INDEX_PATH)
        with self.assertRaises(sqlite3.DatabaseError):ensure_index(cfg)
        self.assertEqual(before,sha256(cfg.PROFILE_INDEX_PATH))
    def test_interrupted_build_never_publishes(self):
        cfg=self.fresh();cfg.checkpoint_records=1
        def stop(*args):
            for i,e in enumerate(real_records(*args)):
                if i==1:raise RuntimeError('interrupted first index build')
                yield e
        with patch('recovery.index_init.records',side_effect=stop):
            with self.assertRaises(RuntimeError):ensure_index(cfg)
        self.assertFalse(Path(cfg.PROFILE_INDEX_PATH).exists());self.assertTrue(Path(cfg.PROFILE_INDEX_PATH+'.building').exists())
        index,new=ensure_index(cfg);self.assertTrue(new);self.assertEqual(index.unique('Alice Jones'),'homepages/a');index.db.close()
    def test_local_dtd_outside_cache(self):
        cfg=self.fresh();target=self.root/'provided.dtd';Path(cfg.DTD_PATH).rename(target);cfg.DTD_PATH=str(target);cfg.validate();self.assertTrue(run(cfg)['profile_index_created'])
    def test_duplicate_occurrence_rejected(self):
        cfg=self.fresh();p=Path(cfg.FETCH_DATA_ROOT)/cfg.authors_file;d=pd.read_csv(p);pd.concat([d,d]).to_csv(p,index=False)
        with self.assertRaisesRegex(ValueError,'Duplicate Fetch'):run(cfg)
    def test_wrong_t0_rejected(self):
        cfg=self.fresh();p=Path(cfg.FETCH_DATA_ROOT)/cfg.dblp_directory/'dblp_author_features_by_author.csv';d=pd.read_csv(p);d.t0_year=2022;d.to_csv(p,index=False)
        with self.assertRaisesRegex(ValueError,'t0 differs'):run(cfg)
    def test_incomplete_roster_cannot_be_complete_team(self):
        cfg=self.fresh();p=Path(cfg.FETCH_DATA_ROOT)/cfg.master_file;d=pd.read_csv(p);d.author_count_v1=2;d.authors_v1_json='[{"name":"Alice Jones"},{"name":"Missing Person"}]';d.to_csv(p,index=False)
        run(cfg);team=pd.read_csv(Path(cfg.OUTPUT_DIR)/'features/final_team_features.csv');self.assertTrue(team.dblp_team_mean_publication_count_pre_t0.isna().all());self.assertFalse(team.dblp_author_identity_complete.iloc[0]);self.assertEqual(team.expected_author_count.iloc[0],2)
    def test_invalid_prior_pid_has_explicit_reason(self):
        cfg=self.fresh();p=Path(cfg.FETCH_DATA_ROOT)/cfg.dblp_directory/'dblp_author_features_by_author.csv';d=pd.read_csv(p);d.dblp_person_id='homepages/absent';d.to_csv(p,index=False)
        run(cfg);audit=pd.read_csv(Path(cfg.OUTPUT_DIR)/'matching/fetch_identity_revalidation.csv');self.assertEqual(audit.reason.iloc[0],'INVALID_PROFILE')
    def test_no_recovery_config_fields(self):
        cfg=Config();self.assertFalse(hasattr(cfg,'ROUND2_OUTPUT_DIR'));self.assertFalse(hasattr(cfg,'previous_run_root'));self.assertFalse(hasattr(cfg,'mapping_file'))

class FocalRegressionTests(unittest.TestCase):
    def setUp(self):self.t=tempfile.TemporaryDirectory();self.root=Path(self.t.name)
    def tearDown(self):self.t.cleanup()
    def fixture(self,unknown_year=False):
        profiles={'a':['Alice Jones'],'c':['John Smith 0001','John Smith'],'d':['John Smith 0002','John Smith']}
        rows=[dict(arxiv_id='2001.00001',author_position=1,original_name='Alice Jones',identity_status='VERIFIED',dblp_person_id='a'),dict(arxiv_id='2001.00001',author_position=2,original_name='John Smith',identity_status='UNRESOLVED',dblp_person_id='')]
        pubs=[dict(key='potential',title='Uncertain target study',year=None if unknown_year else 2017,authors=['No mapped author'] if unknown_year else ['Alice Jones','John Smith 0001','No mapped author'],links=['https://doi.org/10.1234/self']),dict(key='duplicate',title='Older differently named version',year=2015,authors=['Alice Jones','John Smith 0001'],links=['https://doi.org/10.1234/self']),dict(key='independent',title='Independent history',year=2010,authors=['Alice Jones','John Smith 0001'],links=[])]
        return create(self.root,[dict(arxiv_id='2001.00001',title_v1='Uncertain target study')],rows,profiles,pubs)
    def check(self,cfg):
        run(cfg);db=sqlite3.connect(Path(cfg.DERIVED_CACHE_DIR)/'candidate_history.sqlite')
        self.assertEqual(db.execute("SELECT reliable FROM targets WHERE key='potential'").fetchone()[0],0)
        self.assertEqual(db.execute("SELECT COUNT(*) FROM focal WHERE key='potential'").fetchone()[0],1)
        self.assertEqual(db.execute("SELECT COUNT(*) FROM candidate_coauthors WHERE key IN ('potential','duplicate')").fetchone()[0],0)
        self.assertGreater(db.execute("SELECT COUNT(*) FROM candidate_coauthors WHERE key='independent'").fetchone()[0],0)
        db.close();audit=json.loads((Path(cfg.OUTPUT_DIR)/'report/temporal_integrity_audit.json').read_text());self.assertEqual(audit['coauthor_focal_leakage_count'],0);self.assertGreater(audit['unreliable_target_candidates_excluded'],0)
        a=pd.read_csv(Path(cfg.OUTPUT_DIR)/'features/final_author_features.csv');self.assertEqual(a.dblp_author_publication_count_pre_t0.iloc[0],1)
    def test_unreliable_target_cannot_prove_itself(self):self.check(self.fixture())
    def test_unknown_year_unmapped_target_excludes_duplicate(self):self.check(self.fixture(True))
    def test_fetch_candidate_audit_independent_of_selected(self):
        cfg=self.fixture();p=Path(cfg.FETCH_DATA_ROOT)/cfg.dblp_directory/'dblp_candidate_match_audit.csv';pd.DataFrame([dict(arxiv_id='2001.00001',dblp_record_key='independent',selected=False,candidate_status='REJECTED')]).to_csv(p,index=False)
        run(cfg);d=sqlite3.connect(Path(cfg.DERIVED_CACHE_DIR)/'candidate_history.sqlite');self.assertEqual(d.execute('SELECT COUNT(*) FROM candidate_coauthors').fetchone()[0],0);d.close()
if __name__=='__main__':unittest.main()

class IntegratedLegacyIdentityTests(unittest.TestCase):
    def setUp(self):self.t=tempfile.TemporaryDirectory();self.root=Path(self.t.name)
    def tearDown(self):self.t.cleanup()
    def run_case(self,name,profiles,target,conflict=''):
        cfg=create(self.root,[dict(arxiv_id='2001.00001',title_v1='Target')],[dict(arxiv_id='2001.00001',author_position=1,original_name=name,identity_status='UNRESOLVED')],profiles,[dict(key='focal',title='Target',year=2020,authors=[target],links=['https://arxiv.org/abs/2001.00001'])])
        if conflict:
            p=Path(cfg.FETCH_DATA_ROOT)/cfg.dblp_directory/'dblp_author_features_by_author.csv';d=pd.read_csv(p).fillna('');d['dblp_identity_conflict_status']=conflict;d.to_csv(p,index=False)
        run(cfg);return pd.read_csv(Path(cfg.OUTPUT_DIR)/'matching/final_author_identity_mapping.csv',keep_default_na=False)
    def test_pid_mismatch_resolved_by_target_evidence(self):
        a=self.run_case('Ian Goodfellow',{'a':['Ian Goodfellow'],'b':['Ian J. Goodfellow']},'Ian J. Goodfellow','SOURCE_TARGET_PERSON_ID_MISMATCH');self.assertEqual(a.dblp_person_id.iloc[0],'b')
    def test_ambiguous_source_resolved_by_numbered_target_alias(self):
        a=self.run_case('Li Zhang',{'a':['Li Zhang','Li Zhang 0001'],'b':['Li Zhang','Li Zhang 0002']},'Li Zhang 0002');self.assertEqual(a.dblp_person_id.iloc[0],'b')
    def test_broken_spacing_normalized(self):
        a=self.run_case('John S mith',{'a':['John Smith']},'John Smith');self.assertEqual(a.dblp_person_id.iloc[0],'a')
    def test_initial_alone_does_not_override_full_given_name(self):
        a=self.run_case('John Smith',{'a':['James Smith']},'James Smith');self.assertEqual(a.identity_status.iloc[0],'UNRESOLVED')
    def test_existing_pid_redirect_revalidated(self):
        cfg=small(self.root);d=sqlite3.connect(cfg.PROFILE_INDEX_PATH);d.execute('INSERT INTO profiles VALUES (?,?,?)',('old','a',1));d.commit();d.close()
        p=Path(cfg.FETCH_DATA_ROOT)/cfg.dblp_directory/'dblp_author_features_by_author.csv';a=pd.read_csv(p);a.loc[0,'dblp_person_id']='old';a.to_csv(p,index=False)
        run(cfg);a=pd.read_csv(Path(cfg.OUTPUT_DIR)/'matching/fetch_identity_revalidation.csv');self.assertEqual(a.canonical_pid.iloc[0],'a');self.assertEqual(a.final_status.iloc[0],'VERIFIED')
