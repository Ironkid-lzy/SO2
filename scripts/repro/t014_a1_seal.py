"""CPU-only terminal-job finalizer, no new training, restart or quota dependency."""
import hashlib
import json
import math
import statistics
import time
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
WORK=ROOT/'_so2_work/t014/a1'
RUN=ROOT/'_so2_work/runs/EXP-035/EXP-035-s0-attempt1'
MARKER=WORK/'sealed.json'


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def evaluate(path):
    events=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    points=[r for r in events if r['event']=='evaluation']
    assert [r['env_steps'] for r in points]==list(range(0,100001,2500))
    for r in points:
        assert len(r['returns'])==20 and all(math.isfinite(x) for x in r['returns'])
        score=100*(statistics.mean(r['returns'])+280.178953)/(12135+280.178953)
        assert abs(score-r['normalized_score_100'])<1e-8
    auc=sum((b['env_steps']-a['env_steps'])*(a['normalized_score_100']+b['normalized_score_100'])/2
            for a,b in zip(points[:-1],points[1:]))/100000
    return {'auc':auc,'final':points[-1]['normalized_score_100'],'points':points}


def main():
    if MARKER.exists():return
    deadline=time.time()+18200
    job=None
    while time.time()<deadline:
        state=json.loads((ROOT/'_so2_work/t014/budget.json').read_text())
        job=next((j for j in state['jobs'] if j['job']=='a1-online-s0-v1'),None)
        if job and job['status'] in ('completed','partial'):break
        time.sleep(5)
    else:raise RuntimeError('finite finalizer wait expired; no restart')
    output=ROOT/'_so2_work/deliveries/t014-a1-v1';output.mkdir(parents=True,exist_ok=False)
    report={'status':job['status'],'job':job,'T014_used_seconds':state['used_seconds'],
            'actual_training_GPU_job_seconds':job['wall_seconds'],'automatic_multiseed':False,
            'scope':'Single seed0 exploratory known-method baseline, no novelty/significance claim'}
    report.update(job_status=job['status'],worth_expanding=False,verification_errors=[])
    try:
        if job['status']=='completed':
            assert json.loads((RUN/'acceptance.json').read_text())['accepted']
            control=Path('/home/lzy/Projects/SO2_t011/_so2_work/runs/EXP-029/EXP-029-s0-attempt1/metrics.jsonl')
            protocol=json.loads((WORK/'protocol.json').read_text())
            assert digest(control)==protocol['formal']['control_metrics_sha256']
            treatment=evaluate(RUN/'metrics.jsonl');baseline=evaluate(control)
            assert treatment['points'][0]['returns']==baseline['points'][0]['returns']
            report.update(treatment=treatment,control=baseline,
                          delta_auc=treatment['auc']-baseline['auc'],delta_final=treatment['final']-baseline['final'])
            report['worth_expanding']=report['delta_auc']>=1 and report['delta_final']>=-1
    except Exception:
        report.update(status='verification_failed',worth_expanding=False)
        report['verification_errors'].append(traceback.format_exc())
    files=[]
    for path in sorted(RUN.rglob('*')):
        if not path.is_file():continue
        row={'path':str(path),'backup':'Linux only'}
        try:
            row.update(size=path.stat().st_size,sha256=digest(path))
            if path.suffix in ('.json','.jsonl','.py','.yaml') and path.stat().st_size<10*1024**2:
                relative=path.relative_to(RUN);dest=output/'run'/relative;dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_bytes(path.read_bytes());row['copied']=str(dest.relative_to(output));row['backup']='CPU sealed small evidence'
        except Exception:
            row['error']=traceback.format_exc()
            report.update(status='verification_failed',worth_expanding=False)
            report['verification_errors'].append(row['error'])
        files.append(row)
    save(output/'report.json',report)
    save(output/'seal.json',{'status':report['status'],'files':files})
    save(MARKER,{'status':report['status'],'output':str(output),'report_sha256':digest(output/'report.json'),
                 'no_new_training':True,'budget_job':job})


if __name__=='__main__':main()
