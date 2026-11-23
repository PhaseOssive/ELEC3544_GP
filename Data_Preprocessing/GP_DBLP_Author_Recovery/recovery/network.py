"""Pre-t0, focal-excluded independent coauthor evidence and masked-PID calibration."""
import json,math,hashlib,logging
from functools import lru_cache
from .progress import tqdm
from .rules import normalize_title
from common import export_query
from .io import write_json,write_csv


def score_candidates(cfg,store,papers):
    db=store.db;titles={p['arxiv_id']:set(normalize_title(p['title_v1']).split()) for p in papers}
    db.execute('CREATE TABLE IF NOT EXISTS score_done(oid TEXT PRIMARY KEY)')
    db.execute('CREATE TEMP TABLE IF NOT EXISTS anchors(pid TEXT PRIMARY KEY)')
    done={x[0] for x in db.execute('SELECT oid FROM score_done')}
    logging.info('Network scoring: %d committed author occurrences reused',len(done))
    @lru_cache(maxsize=10000)
    def rarity(peer,t0,aid):
        # Degree measured only in the complete candidate history population, pre-t0.
        degree=db.execute('''SELECT COUNT(DISTINCT cp.pid) FROM publication_authors a JOIN candidate_publications cp USING(key) JOIN publications p USING(key) WHERE a.pid=? AND p.year<? AND cp.pid<>? AND NOT EXISTS (SELECT 1 FROM focal f JOIN record_works fr ON fr.key=f.key JOIN record_works pr ON pr.work=fr.work WHERE f.aid=? AND pr.key=p.key)''',(peer,t0,peer,aid)).fetchone()[0]
        return 1/math.log2(2+degree)
    total=db.execute('SELECT COUNT(*) FROM occurrences').fetchone()[0]
    for oid,aid,t0,truth in tqdm(db.execute('SELECT oid,aid,t0,pid FROM occurrences'),total=total,desc='Historical coauthor evidence / hidden PID',unit='authors'):
        if oid in done:continue
        db.execute('DELETE FROM scores WHERE oid=?',(oid,))
        db.execute('DELETE FROM candidate_coauthors WHERE oid=?',(oid,))
        db.execute('DELETE FROM anchors')
        # Hidden truth is excluded as anchor and is NEVER injected into candidate lists.
        db.execute("INSERT OR IGNORE INTO anchors SELECT pid FROM occurrences WHERE aid=? AND oid<>? AND verified=1 AND pid<>''",(aid,oid))
        nanchors=db.execute('SELECT COUNT(*) FROM anchors').fetchone()[0]
        db.execute('''INSERT OR IGNORE INTO candidate_coauthors
        SELECT ?,c.pid,a.pid,p.key,p.year,rw.work FROM
        (SELECT DISTINCT pid FROM candidates WHERE oid=? AND eligible=1) c
        JOIN candidate_publications cp ON cp.pid=c.pid
        JOIN publications p ON p.key=cp.key
        JOIN record_works rw ON rw.key=p.key
        JOIN publication_authors a ON a.key=p.key
        JOIN anchors an ON an.pid=a.pid
        WHERE p.year<? AND a.pid<>c.pid AND NOT EXISTS
        (SELECT 1 FROM focal f JOIN record_works fr ON fr.key=f.key WHERE f.aid=? AND fr.work=rw.work)''',(oid,oid,t0,aid))
        # Aggregate only this occurrence: bounded memory, a few queries instead of
        # multiple round trips for every expanded candidate (mostly empty evidence).
        peers_by={};works_by={};titles_by={}
        for pid,peer,work,title in db.execute('SELECT c.pid,c.peer,c.work,p.title FROM candidate_coauthors c JOIN publications p USING(key) WHERE c.oid=? ORDER BY c.pid,c.peer',(oid,)):
            peers_by.setdefault(pid,set()).add(peer);works_by.setdefault(pid,set()).add(work);titles_by.setdefault(pid,set()).add(title)
        for pid,grade,status in db.execute("SELECT c.pid,MAX(c.grade),h.status FROM candidates c LEFT JOIN candidate_history_status h ON h.pid=c.pid WHERE c.oid=? AND c.eligible=1 GROUP BY c.pid",(oid,)):
            peers=sorted(peers_by.get(pid,()));joint=len(works_by.get(pid,()))
            rare=sum(rarity(peer,t0,aid) for peer in peers);topic=[]
            for title in sorted(titles_by.get(pid,())):
                words=set(normalize_title(title).split());target=titles[aid];topic.append(len(words&target)/max(1,len(words|target)))
            complete=int(status=='COMPLETE_OBSERVED_SNAPSHOT')
            db.execute('INSERT OR REPLACE INTO scores VALUES (?,?,?,?,?,?,?,?,?,?)',(oid,pid,grade,len(peers),joint,len(peers)/max(1,nanchors),rare,sum(topic)/len(topic) if topic else 0,joint+2*len(peers)+rare,complete))
        db.execute('INSERT OR REPLACE INTO score_done VALUES (?)',(oid,));done.add(oid)
        if len(done)%25==0:
            db.commit();logging.info('Network scoring committed=%d/%d',len(done),total)
    store.set('scores_complete',1)
    db.commit()
    export_query(cfg,'matching/candidate_coauthor_evidence.csv',db,'''SELECT c.*,p.title,p.payload AS publication_evidence_json FROM candidate_coauthors c JOIN publications p USING(key) ORDER BY oid,pid,work,key''')
    export_query(cfg,'matching/candidate_scores.csv',db,'''SELECT *, 'CANDIDATE_HISTORY_POPULATION_PRE_T0' AS rare_weight_scope, 'JOINT_PUBLICATION_TITLE_JACCARD_REVIEW_ONLY' AS research_similarity_method FROM scores ORDER BY oid,score DESC,pid''')

def winner(db,oid,minimum_joint,margin,ranking=None):
    def read():
        cur=db.execute('SELECT pid,grade,common,joint,score,complete FROM scores WHERE oid=? ORDER BY score DESC,pid',(oid,))
        best=cur.fetchone();second=cur.fetchone()
        complete=db.execute('SELECT MIN(complete) FROM scores WHERE oid=?',(oid,)).fetchone()[0]==1
        return best,second,complete
    if ranking is None:best,second,complete=read()
    else:
        if oid not in ranking:ranking[oid]=read()
        best,second,complete=ranking[oid]
    if not best:return None,None
    gap=best[4]-(second[4] if second else 0)
    accepted=complete and best[1]>=3 and best[2]>=2 and best[3]>=minimum_joint and gap>=margin
    evidence=dict(pid=best[0],name_grade=best[1],common_coauthors=best[2],joint_works=best[3],score=best[4],second_score=second[4] if second else 0,candidate_margin=gap,all_competitor_histories_complete=complete)
    return best[0] if accepted else None,evidence

def wilson(correct,n):
    if not n:return 0.
    z=1.96;p=correct/n;den=1+z*z/n
    return (p+z*z/(2*n)-z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/den

def calibrate(cfg,store):
    db=store.db
    # PID-disjoint connected components: repeated identities never cross train/test.
    parent={}
    def find(x):
        parent.setdefault(x,x)
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    for aid,pid in db.execute("SELECT aid,pid FROM occurrences WHERE verified=1 AND pid<>''"):
        a,b=find('paper:'+aid),find('pid:'+pid)
        if a!=b:parent[max(a,b)]=min(a,b)
    samples=[(oid,truth,'holdout' if int(hashlib.sha256(find('paper:'+aid).encode()).hexdigest(),16)%3==0 else 'calibration') for oid,aid,truth in db.execute('SELECT oid,aid,pid FROM occurrences WHERE verified=1')]
    trials=[];ranking={}
    for joint in (2,3,5):
        for margin in (2.,4.,8.):
            correct=accepted=0
            for oid,truth,split in samples:
                if split!='calibration':continue
                predicted,_=winner(db,oid,joint,margin,ranking)
                if predicted:accepted+=1;correct+=int(predicted==truth)
            trials.append(dict(minimum_joint=joint,margin=margin,accepted=accepted,correct=correct,precision=correct/accepted if accepted else None,precision_lower_bound=wilson(correct,accepted)))
    eligible=[r for r in trials if r['precision_lower_bound']>=cfg.precision_lower_bound]
    selected=max(eligible,key=lambda r:r['accepted']) if eligible else None
    evaluation=[];correct=accepted=0
    for oid,truth,split in samples:
        predicted,evidence=winner(db,oid,selected['minimum_joint'] if selected else 5,selected['margin'] if selected else 8.,ranking)
        evaluation.append(dict(occurrence_id=oid,hidden_pid=truth,predicted_pid=predicted or '',split=split,correct=predicted==truth if predicted else None,evidence_json=json.dumps(evidence),policy_selected=bool(selected)))
        if split=='holdout' and predicted:accepted+=1;correct+=int(predicted==truth)
    lower=wilson(correct,accepted)
    enabled=bool(cfg.network_auto_accept and selected and accepted>=cfg.backtest_min_holdout and lower>=cfg.precision_lower_bound)
    report=dict(automatic_network_confirmation_enabled=enabled,selected_rule=selected,calibration_trials=trials,holdout_accepted=accepted,holdout_correct=correct,holdout_precision=correct/accepted if accepted else None,holdout_wilson_lower_95=lower,requested_lower_bound=cfg.precision_lower_bound,holdout_policy_evaluated='selected_calibration_rule' if selected else 'diagnostic_strict_rule_only',split='CONNECTED_PAPER_PID_COMPONENTS',ground_truth='PREVIOUS_VERIFIED_PSEUDO_LABELS_NOT_EXTERNAL_GOLD',future_collaborations_allowed=False,reason='VALIDATED_CONSERVATIVE_RULE' if enabled else 'INSUFFICIENT_CALIBRATION_OR_HOLDOUT_SUPPORT_NETWORK_AUTO_DISABLED')
    write_csv(cfg,'report/masked_pid_backtest.csv',evaluation);write_json(cfg,'report/backtest_summary.json',report);return report
