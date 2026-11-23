import contextlib
import io
import json
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fixtures import dataset, FakeCrossref
from main import run, rollback
from common import read, sha, FIELDS
from external import Crossref
from fast import FastSnapshot

class Publisher(Crossref):
    calls=[]
    def work(self,doi):
        self.calls.append(doi)
        self.requests+=1
        result=FakeCrossref.fixtures.get(doi,{'status':'NOT_FOUND'})
        with self.db:self.db.execute('INSERT OR REPLACE INTO cache VALUES (?,?,?,?)',(doi,result['status'],json.dumps(result),time.time()))
        return result

class FastTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.cfg=dataset(self.root,with_normal=True)
        self.cfg.update(max_external_requests=180,biography_enabled=False)
        Publisher.calls=[]
    def execute(self,**kwargs):
        kwargs.setdefault('audit_only',False)  # isolated temporary regression fixtures only
        with contextlib.redirect_stdout(io.StringIO()):return run(self.cfg,client_factory=Publisher,**kwargs)
    def test_full_coverage_strict_repair_and_request_selection(self):
        before=sha(self.cfg['input_csv'])
        out=self.execute()
        verdicts=read(out/'paper_verdicts.csv').set_index('paper_id')
        self.assertEqual(len(verdicts),3)
        self.assertEqual(verdicts.loc['2003.00003','status'],'Unresolved')
        self.assertEqual(set(Publisher.calls),{'10.9999/ta','10.9999/tc','10.9999/bada','10.9999/badc'})
        changes=read(out/'changes.csv')
        self.assertEqual(set(changes.paper_id),{'2001.00001','2002.00002'})
        self.assertEqual(set(changes.field),set(FIELDS))
        self.assertEqual(before,sha(self.cfg['input_csv']))
        rollback(out,self.root/'restored.csv')
        self.assertTrue(read(self.root/'restored.csv').equals(read(self.cfg['input_csv'])))
    def test_missing_proof_preserves_extremes(self):
        FakeCrossref.fixtures={}
        out=self.execute()
        self.assertTrue(read(self.cfg['input_csv']).equals(read(out/'papers_features_dblp_repaired.csv')))
        self.assertEqual(set(read(out/'paper_verdicts.csv').status),{'Unresolved'})
    def test_cached_supported_extreme_unchanged(self):
        path=Path(self.cfg['cache_root']);path.mkdir()
        c=Crossref(path/'crossref_cache.sqlite',self.cfg)
        for doi,v in FakeCrossref.fixtures.items():
            if doi.endswith(('te','e0','e1','e2')):c.db.execute('INSERT INTO cache VALUES (?,?,?,?)',(doi,'OK',json.dumps(v),time.time()))
        c.db.commit();c.db.close()
        out=self.execute()
        self.assertEqual(read(out/'paper_verdicts.csv').set_index('paper_id').loc['2003.00003','status'],'SUPPORTED_NORMAL_OUTLIER')
        self.assertNotIn('2003.00003',set(read(out/'changes.csv').paper_id))
    def test_budget_zero_full_coverage_and_resume(self):
        self.cfg['max_external_requests']=0
        out=self.execute()
        self.assertEqual(Publisher.calls,[])
        self.assertEqual(len(read(out/'paper_verdicts.csv')),3)
        self.assertEqual(len(read(out/'changes.csv')),0)
        second=self.execute()
        self.assertTrue(json.loads((second/'cache_reuse_report.json').read_text())['filtered_snapshot_already_complete'])
        self.assertTrue(read(out/'paper_verdicts.csv').equals(read(second/'paper_verdicts.csv')))
    def test_uncorroborated_signature_cannot_repair(self):
        for v in FakeCrossref.fixtures.values():
            for a in v['author']:a['authenticated-orcid']=False
        out=self.execute()
        self.assertTrue(read(out/'changes.csv').empty)
    def test_candidate_union_includes_unflagged_paper(self):
        ids=self.root/'ids.csv';ids.write_text('paper_id\nfiller-1\n')
        self.cfg['candidate_csv']=str(ids)
        out=self.execute()
        self.assertEqual(len(read(out/'paper_verdicts.csv')),4)
        self.assertNotIn('filler-1',set(read(out/'changes.csv').paper_id))
    def test_stale_baseline_blocks(self):
        p=self.cfg['input_csv'];df=read(p);df.loc[df.paper_id=='2001.00001',FIELDS[0]]='9999';df.to_csv(p,index=False)
        out=self.execute()
        self.assertNotIn('2001.00001',set(read(out/'changes.csv').paper_id))
    def test_review_only_preserves_original(self):
        out=self.execute(audit_only=True)
        self.assertTrue(read(self.cfg['input_csv']).equals(read(out/'papers_features_dblp_repaired.csv')))
    def test_pid_reassignment_requires_target_and_transfer_evidence(self):
        from fixtures import A,B
        path=Path(self.cfg['xml_path'])
        xml=path.read_text().replace(f'<article key="tA"><author orcid="{A}">','<article key="tA"><author>')
        xml=xml.replace('</dblp>',f'<article key="bnew"><author orcid="{B}">Alex Lee 0002</author><title>Independent correct work</title><year>2008</year></article></dblp>')
        path.write_text(xml)
        FakeCrossref.fixtures['10.9999/ta']['author'][0]['ORCID']=B
        out=self.execute()
        row=read(out/'author_verdicts.csv').set_index('paper_id').loc['2001.00001']
        self.assertEqual(row['new_pid'],'homepages/pB')
        self.assertEqual(float(row['new_count']),2.)
        self.assertIn('2001.00001',set(read(out/'changes.csv').paper_id))
    def test_matching_v2_snapshot_skips_xml_and_incompatible_one_does_not(self):
        out=self.execute()
        summary=json.loads((out/'summary.json').read_text())
        root=Path(self.cfg['cache_root'])
        src=root/summary['cache_job']/'snapshot.sqlite'
        import shutil
        legacy=root/'legacy_v2';legacy.mkdir();path=legacy/'snapshot.sqlite';shutil.copy2(src,path)
        with sqlite3.connect(path) as db:
            old=dict(summary['snapshot_binding'],version='2.0.0')
            db.execute('UPDATE meta SET v=? WHERE k=?',(json.dumps(old),'signature'))
        snapshot_hash=sha(path)
        from fast import reusable_snapshot
        import tempfile
        with tempfile.TemporaryDirectory() as z:
            only=Path(z)/'v2';only.mkdir();shutil.copy2(path,only/'snapshot.sqlite')
            self.assertIsNotNone(reusable_snapshot(z,summary['snapshot_binding']))
            self.assertIsNone(reusable_snapshot(z,dict(summary['snapshot_binding'],xml_sha256='changed')))
        shutil.rmtree(root/summary['cache_job']);shutil.rmtree(self.cfg['local_work_dir'])
        with patch('snapshot.records',side_effect=AssertionError('Unneeded XML scan')):
            self.execute()
        self.assertEqual(sha(path),snapshot_hash)
    def test_resume_after_interruption(self):
        import main
        original=main.triage
        def interrupt(a,*args,**kw):
            if a['arxiv_id']=='2002.00002':raise KeyboardInterrupt()
            return original(a,*args,**kw)
        with patch('main.triage',side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):self.execute()
        out=self.execute()
        self.assertEqual(len(read(out/'changes.csv')),8)
        self.assertEqual(len(read(out/'author_verdicts.csv')),3)
    def test_corrupt_exported_history_blocks_change(self):
        path=self.root/'oldhistory.csv'
        path.write_text('arxiv_id,occurrence_id,dblp_person_id,publication_year,t0_year,decision\n2001.00001,2001.00001:1,homepages/pA,2021,2020,INCLUDED\n')
        self.cfg['recovery_history_csv']=str(path)
        out=self.execute()
        self.assertIn('RECOVERY_HISTORY_CSV_CONFLICT',read(out/'paper_verdicts.csv').set_index('paper_id').loc['2001.00001','blocked_reasons_json'])
        self.assertTrue(read(out/'changes.csv').empty)
    def test_reviewed_record_rule_repairs_offline(self):
        self.cfg['crossref_enabled']=False
        rule=dict(pid='homepages/pA',record_key='badA',year=2000,author_alias='Alex Lee 0001',
                  evidence_urls=['https://example.org/synthetic-1','https://example.org/synthetic-2'],
                  reviewer='Synthetic test',rule_id='TEST_ONLY')
        Path(self.cfg['reviewed_rules']).write_text(json.dumps([rule]))
        out=self.execute()
        self.assertEqual(set(read(out/'changes.csv').paper_id),{'2001.00001'})
        self.assertEqual(Publisher.calls,[])
    def test_fallback_only_other_version_blocks_removal(self):
        from audit import work_decision
        rows=[{'record':{'dois':['a'],'arxiv_ids':[]},'result':{'state':'WRONG_PERSON_CONFIRMED'}},
              {'record':{'dois':['b'],'arxiv_ids':[]},'result':{'state':'UNRESOLVED'}}]
        self.assertEqual(work_decision(rows),'UNRESOLVED_MIXED_WORK_COMPONENT')
    def test_pid_collision_preserves_team(self):
        # Duplicate a verified position under the same PID, and give the master the matching roster size.
        p=Path(self.cfg['recovery_root'])/'features/treated_author_features.csv';authors=read(p)
        r=authors.iloc[0].copy();r['author_position']='2';r['occurrence_id']='2001.00001:2'
        import pandas as pd
        pd.concat([authors,r.to_frame().T]).to_csv(p,index=False)
        summary=p.parent.parent/'report/final_recovery_summary.json'
        obj=json.loads(summary.read_text());obj['final_author_occurrences']+=1;summary.write_text(json.dumps(obj))
        out=self.execute();v=read(out/'paper_verdicts.csv').set_index('paper_id')
        self.assertIn('DUPLICATE_PID_IN_TEAM',v.loc['2001.00001','blocked_reasons_json'])
        self.assertNotIn('2001.00001',set(read(out/'changes.csv').paper_id))
