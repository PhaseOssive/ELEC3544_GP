import contextlib,io,json,sqlite3,tempfile,time,unittest
from pathlib import Path
from fixtures import dataset,FakeCrossref
from test_fast import Publisher
from main import run
from common import read,sha
from trace31 import resolve,priority
from external import Crossref
class TraceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.cfg=dataset(self.root,with_normal=True)
        self.cfg['biography_enabled']=False;Publisher.calls=[]
    def audit(self):
        with contextlib.redirect_stdout(io.StringIO()):return run(self.cfg,client_factory=Publisher)
    def test_offline_names_and_complete_versions_and_no_changes(self):
        self.cfg['max_external_requests']=0
        p=Path(self.cfg['xml_path']);p.write_text(p.read_text().replace('</dblp>','<article key="aliasversion"><author>Dana Ray</author><title>Normal work 0</title><year>2010</year><ee>https://doi.org/10.9999/e0</ee></article></dblp>'))
        orig=sha(self.cfg['input_csv']);out=self.audit()
        self.assertEqual(sha(self.cfg['input_csv']),orig)
        s=json.loads((out/'summary.json').read_text());self.assertEqual(s['mode'],'AUDIT_ONLY');self.assertEqual(s['changed_cells'],0)
        a=read(out/'author_verdicts.csv');w=read(out/'work_evidence.csv')
        self.assertFalse(a.author_name.eq('').any());self.assertEqual(len(w.drop_duplicates(['occurrence_id','work_id'])),sum(a.works_examined.astype(int)))
        self.assertGreater(len(w),s['distinct_occurrence_works'])
        self.assertIn('BUDGET_DEFERRED',' '.join(a.audit_flags_json));self.assertEqual(Publisher.calls,[])
    def test_exact_master_fallback_and_pid_unchanged(self):
        p=Path(self.cfg['recovery_root'])/'features/treated_author_features.csv';df=read(p);df.loc[2,['original_name','repaired_name']]='';df.to_csv(p,index=False)
        out=self.audit();a=read(out/'author_verdicts.csv').set_index('paper_id').loc['2003.00003']
        self.assertEqual((a.author_name,a.name_source,a.old_pid),('Dana Ray','master_position','homepages/pE'))
    def test_team_profile_missing_and_wrong_key(self):
        a=dict(arxiv_id='p',author_position='1',dblp_person_id='homepages/x',original_name='',repaired_name='')
        paper={'paper_id':'p','author_count_v1':'1'}
        team={'arxiv_id':'p','expected_author_count':'1','authors_json':'[{"name":"Max","pid":"homepages/x"}]'}
        self.assertEqual(resolve(a,paper,team)[:2],('Max','team_position'))
        team['authors_json']='[{"name":"Max","pid":"wrong"}]'
        self.assertIn('ROSTER_KEY_CONFLICT',resolve(a,paper,team)[2])
        self.assertIn('NAME_MISSING',resolve(a,paper,team)[2])
        from fast import FastSnapshot
        snap=FastSnapshot(':memory:')
        snap.db.execute("INSERT INTO profiles VALUES ('homepages/x','',1,'[]','[]')")
        snap.db.execute("INSERT INTO aliases VALUES ('homepages/x','Max')")
        self.assertEqual(resolve(a,paper,None,snap)[:2],('Max','profile_exact_pid'))
        snap.db.execute("INSERT INTO aliases VALUES ('homepages/x','Max A')")
        self.assertIn('NAME_MISSING',resolve(a,paper,None,snap)[2]);snap.db.close()
    def test_cached_title_conflict_enters_queue_and_no_repeat(self):
        root=Path(self.cfg['cache_root']);root.mkdir()
        c=Crossref(root/'crossref_cache.sqlite',self.cfg)
        meta=dict(FakeCrossref.fixtures['10.9999/e0'],title=['Totally unrelated item'])
        c.db.execute('INSERT INTO cache VALUES (?,?,?,?)',('10.9999/e0','OK',json.dumps(meta),time.time()));c.db.commit();c.db.close()
        out=self.audit()
        q=read(out/'audit_queue.csv').set_index('paper_id')
        self.assertIn('CACHED_TITLE_CONFLICT',q.loc['2003.00003','signals'])
        a=read(out/'author_verdicts.csv').set_index('paper_id');self.assertIn('TITLE_CONFLICT',a.loc['2003.00003','audit_flags_json'])
        self.assertNotIn('10.9999/e0',Publisher.calls)
        self.assertNotIn('2003.00003',set(read(out/'changes.csv').paper_id))
    def test_no_work_doi_is_separate_from_missing_name(self):
        p=Path(self.cfg['xml_path']);p.write_text(p.read_text().replace('<ee>https://doi.org/10.9999/e0</ee>',''))
        out=self.audit();row=read(out/'work_evidence.csv').set_index('record_key').loc['e0']
        flags=json.loads(row.audit_flags_json);self.assertIn('NO_WORK_DOI',flags);self.assertIn('NO_IDENTITY_ANCHOR',flags);self.assertNotIn('NAME_MISSING',flags)
    def test_resume_does_not_duplicate_and_reports_same_work_keys(self):
        first=self.audit();count=len(Publisher.calls);second=self.audit()
        self.assertEqual(count,len(Publisher.calls))
        self.assertTrue(read(first/'work_evidence.csv').equals(read(second/'work_evidence.csv')))
    def test_cache_only_reaudit_preserves_confirmed_proposals(self):
        first=self.audit();expected=read(first/'changes.csv')
        self.assertEqual(len(expected),8)
        (next(Path(self.cfg['cache_root']).glob('*/audit_state.sqlite'))).unlink()
        Publisher.calls=[]
        second=self.audit();got=read(second/'changes.csv')
        self.assertTrue(got.sort_values(['paper_id','field']).reset_index(drop=True).equals(expected.sort_values(['paper_id','field']).reset_index(drop=True)))
        self.assertEqual(Publisher.calls,[])
