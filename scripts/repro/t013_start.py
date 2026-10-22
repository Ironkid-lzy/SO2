"""One-shot formal admission; no retry, no resume, no new offline training."""
import datetime, subprocess, time
from t013_common import *
from t013_run import online
from t010_run import interrupted
import signal

def main():
    sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    v=json.loads((VALIDATION/'validation.json').read_text());assert v['status']=='passed' and v['code_sha']==sha
    r=json.loads((WORK/'independent-review.json').read_text());assert r['status']=='passed' and r['code_sha']==sha
    d=json.loads((ROOT/'_so2_work/diagnostics/t013/diagnostics.json').read_text())
    history=Path('/home/lzy/Projects/SO2_t011/_so2_work/runs/EXP-029/EXP-029-s0-attempt1/metrics.jsonl')
    events=[json.loads(x) for x in history.read_text().splitlines()];final=next(x for x in reversed(events) if x['event']=='final')
    assert final['env_steps']==100000 and final['updates']=={'critic':950000,'actor':95000,'alpha':0}
    historical_full=11068.844
    ratio=max(1.,d['timing_ratio'])
    estimate=historical_full*ratio*1.30
    latest=datetime.datetime.fromisoformat('2026-10-23T08:00:00+08:00').timestamp()
    ledger=json.loads((WORK/'budget.json').read_text())
    gate={'code_sha':sha,'historical_full_seconds':historical_full,'historical_includes_eval_save':True,'baseline_metrics_sha256':digest(history),'K4_K1_measured_ratio':d['timing_ratio'],'safety_factor':1.30,'estimate_seconds':estimate,'estimated_finish_unix':time.time()+estimate,'formal_finish_boundary_unix':latest,'prior_and_current_used_seconds':ledger['used_seconds'],'total_budget_seconds':21600}
    gate['passed']=gate['estimated_finish_unix']<latest and ledger['used_seconds']+estimate<21600
    atomic_json(WORK/'launch-gate.json',gate);assert gate['passed'], 'insufficient time or budget; do not start'
    assert subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/so2-t013-actor-mask-average'],text=True).split()[0]==sha, 'code must be pushed'
    m=parent_manifest(0);atomic_json(WORK/'parent-s0.json',m)
    run=run_path(0);assert not run.exists(), 'formal run already exists; never relaunch'
    run.mkdir(parents=True)
    torch.set_num_threads(2)
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    online(run,Path(m['parent_path']),0,'EXP-034',m,18000)
if __name__=='__main__':main()
