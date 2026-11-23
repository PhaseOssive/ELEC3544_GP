import unittest,tempfile,json,sqlite3,hashlib
from pathlib import Path
from unittest.mock import patch
from fixtures import small
from recovery.workflow import run
from recovery.database import Database
from recovery.network import score_candidates,calibrate
from recovery.dedup import DiskDeduplicator
from common import sha256
import reference_network
from colab_runtime import prepare,snapshot,cache_lock,source_cache

class PerformanceTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def test_reference_scoring_evidence_and_backtest_parity(self):
        cfg=small(self.root);run(cfg,'scan');store=Database(Path(cfg.DERIVED_CACHE_DIR)/'candidate_history.sqlite')
        papers=[dict(arxiv_id='2001.00001',title_v1='Target title')]
        # Use actual occurrence IDs/cohort titles from prepared input.
        import pandas as pd
        papers=pd.read_csv(Path(cfg.FETCH_DATA_ROOT)/cfg.master_file,dtype=str).to_dict('records')
        reference_network.score_candidates(cfg,store,papers)
        old=list(store.db.execute('SELECT * FROM scores ORDER BY oid,pid'));evidence=list(store.db.execute('SELECT * FROM candidate_coauthors ORDER BY oid,pid,peer,key'))
        policy=reference_network.calibrate(cfg,store)
        score_candidates(cfg,store,papers)
        self.assertEqual(old,list(store.db.execute('SELECT * FROM scores ORDER BY oid,pid')))
        self.assertEqual(evidence,list(store.db.execute('SELECT * FROM candidate_coauthors ORDER BY oid,pid,peer,key')))
        self.assertEqual(policy,calibrate(cfg,store))
        with patch('recovery.network.export_query'):
            trace=[];store.db.set_trace_callback(trace.append);score_candidates(cfg,store,papers);store.db.set_trace_callback(None)
        self.assertFalse(any(q.startswith('DELETE FROM scores') for q in trace));store.db.close()
    def test_verified_predecessor_cache_migration(self):
        cfg=small(self.root);run(cfg,'scan');store=Database(Path(cfg.DERIVED_CACHE_DIR)/'candidate_history.sqlite')
        new=store.get('input_signature');old=json.loads(new)
        manifest=json.loads((Path(__file__).resolve().parents[1]/'cache_compatibility.json').read_text())
        old['algorithm_sha256']=manifest['trusted_previous_algorithm_sha256'];store.set('input_signature',json.dumps(old));store.db.commit()
        count=store.db.execute('SELECT COUNT(*) FROM publications').fetchone()[0]
        store.bind(new)
        self.assertEqual(count,store.db.execute('SELECT COUNT(*) FROM publications').fetchone()[0]);self.assertEqual(store.get('scan_primary_complete'),'1');self.assertTrue(store.get('performance_migration_from'));store.db.close()
        result=run(cfg);self.assertEqual(result['verified_author_count'],3)
    def test_changed_input_not_migrated(self):
        d=Database(self.root/'db');d.bind(json.dumps({'sources':['a'],'algorithm_sha256':{}}))
        with self.assertRaises(ValueError):d.bind(json.dumps({'sources':['b'],'algorithm_sha256':{}}))
        d.db.close()
    def test_score_checkpoint_transaction_rollback(self):
        cfg=small(self.root/'fixture');d=Database(self.root/'scores.sqlite');DiskDeduplicator(d.db)
        for i in range(30):
            oid=f'{i:03d}';d.db.execute('INSERT INTO occurrences VALUES (?,?,?,?,?,?,?,?)',(oid,'x',i,'Name','p',2020,0,'{}'));d.db.execute('INSERT INTO candidates VALUES (?,?,?,?,?,?,?)',(oid,'p','Name','RAW_ALIAS',4,1,''))
        d.db.execute("CREATE TRIGGER injected_failure BEFORE INSERT ON scores WHEN NEW.oid='025' BEGIN SELECT RAISE(ABORT,'test interruption'); END");d.db.commit()
        with self.assertRaises(sqlite3.IntegrityError):score_candidates(cfg,d,[{'arxiv_id':'x','title_v1':'X'}])
        d.db.rollback();self.assertEqual(d.db.execute('SELECT COUNT(*) FROM score_done').fetchone()[0],25)
        d.db.execute('DROP TRIGGER injected_failure');d.db.commit();score_candidates(cfg,d,[{'arxiv_id':'x','title_v1':'X'}]);self.assertEqual(d.db.execute('SELECT COUNT(*) FROM score_done').fetchone()[0],30);d.db.close()
    def test_runtime_snapshot_restore_bytes_and_no_source_changes(self):
        cfg=small(self.root/'source');run(cfg,'scan');before=sha256(cfg.PROFILE_INDEX_PATH)
        path,local,persistent=prepare(cfg.snapshot(),self.root/'vm1')
        self.assertEqual(before,sha256(local/'index.sqlite'));self.assertEqual(before,sha256(cfg.PROFILE_INDEX_PATH))
        saved=snapshot(local,persistent);self.assertEqual(source_cache(persistent),saved)
        second,l2,_=prepare(cfg.snapshot(),self.root/'vm2')
        self.assertEqual(sha256(local/'derived/candidate_history.sqlite'),sha256(l2/'derived/candidate_history.sqlite'))
        # Repeated staging must not overwrite newer local state.
        (l2/'marker').write_text('new');prepare(cfg.snapshot(),self.root/'vm2');self.assertTrue((l2/'marker').exists())
        run(__import__('config').Config.load(second))
    def test_live_cache_copy_refused(self):
        cfg=small(self.root/'source')
        with cache_lock(Path(cfg.CACHE_DIR)):
            with self.assertRaises(RuntimeError):prepare(cfg.snapshot(),self.root/'vm')
    def test_snapshot_corruption_rejected(self):
        cfg=small(self.root/'source');_,local,persistent=prepare(cfg.snapshot(),self.root/'vm');saved=snapshot(local,persistent)
        with (saved/'index.sqlite').open('ab') as f:f.write(b'corruption')
        with self.assertRaises(ValueError):prepare(cfg.snapshot(),self.root/'vm2')
    def test_export_reuse_and_modified_file_regenerated(self):
        cfg=small(self.root);run(cfg,'prepare');d=Database(Path(cfg.DERIVED_CACHE_DIR)/'candidate_history.sqlite');p=Path(cfg.OUTPUT_DIR)/'matching/expanded_candidate_pids.csv';old=sha256(p)
        with patch('recovery.database.export_query') as call:d.export_candidates(cfg);call.assert_not_called()
        p.write_text('bad');d.export_candidates(cfg);self.assertEqual(old,sha256(p));d.db.close()
    def test_launch_end_to_end_auto_snapshot(self):
        from colab_runtime import launch
        cfg=small(self.root/'source');path=self.root/'config.json';path.write_text(json.dumps(cfg.snapshot()))
        launch(Path(__file__).resolve().parents[1],path,self.root/'vm','all')
        self.assertTrue((Path(cfg.CACHE_DIR)/'runtime_snapshots/LATEST.json').exists())
        summary=json.loads((Path(cfg.OUTPUT_DIR)/'report/final_recovery_summary.json').read_text());self.assertEqual(summary['verified_author_count'],3)
    def test_interrupt_waits_then_snapshots(self):
        from colab_runtime import launch
        from unittest.mock import MagicMock
        cfg=small(self.root/'source');path=self.root/'config.json';path.write_text(json.dumps(cfg.snapshot()))
        child=MagicMock();child.wait.side_effect=[KeyboardInterrupt(),-2];child.poll.return_value=-2;child.pid=123
        import subprocess
        with patch('colab_runtime.subprocess.Popen',return_value=child),patch('colab_runtime.os.killpg') as kill:
            with self.assertRaises(subprocess.CalledProcessError):launch(Path(__file__).resolve().parents[1],path,self.root/'vm')
            kill.assert_called_once();self.assertEqual(child.wait.call_count,2)
        self.assertTrue((Path(cfg.CACHE_DIR)/'runtime_snapshots/LATEST.json').exists())
    def test_runtime_refuses_writes_inside_fetch(self):
        cfg=small(self.root/'source')
        with self.assertRaises(ValueError):prepare(cfg.snapshot(),Path(cfg.FETCH_DATA_ROOT)/'bad')
        self.assertFalse((Path(cfg.FETCH_DATA_ROOT)/'bad').exists())
    def test_snapshot_missing_database_rejected(self):
        cfg=small(self.root/'source');_,local,persistent=prepare(cfg.snapshot(),self.root/'vm');saved=snapshot(local,persistent)
        (saved/'index.sqlite').unlink()
        with self.assertRaises(ValueError):prepare(cfg.snapshot(),self.root/'vm2')
