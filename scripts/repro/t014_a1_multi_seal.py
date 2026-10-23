"""Independent finite CPU finalizer for one extension seed, never starts another."""
import argparse
import traceback
from t014_a1_multi import *
from t014_a1_seal import evaluate

def main(seed):
    protocol,record,work,run=settings(seed);marker=work/'sealed.json'
    if marker.exists():return
    end=time.time()+15840;job=None
    while time.time()<end:
        state=read(ROOT/'_so2_work/t014/budget.json');job=next((j for j in state['jobs'] if j['job']==f'a1-ext-s{seed}-v1'),None)
        if job and job['status'] in ('completed','partial'):break
        dispatch=work/'dispatch.json'
        if dispatch.exists() and not Path('/proc/'+str(read(dispatch)['supervisor_pid'])).exists():
            job=dict(job or {},status='partial',error='Supervisor exited without terminal ledger; no restart',wall_seconds=(job or {}).get('wall_seconds',0))
            break
        time.sleep(5)
    else:
        save(marker,dict(status='verification_failed',error='No terminal job before finite sealer deadline; no restart'));return
    output=ROOT/f'_so2_work/deliveries/t014-a1-ext-s{seed}-v1';output.mkdir(parents=True,exist_ok=False)
    report=dict(status=job['status'],seed=seed,job=job,T014_used_seconds=state['used_seconds'],errors=[],no_next_job=True)
    try:
        if job['status']=='completed':
            assert read(run/'acceptance.json')['accepted']
            assert read(run/'actual-initial-eval.json')['initial_returns_exact_control']
            assert digest(Path(record['control_metrics_path']))==record['control_metrics_sha256']
            treatment=evaluate(run/'metrics.jsonl');control=evaluate(Path(record['control_metrics_path']))
            assert treatment['points'][0]['returns']==control['points'][0]['returns']
            report.update(treatment=treatment,control=control,delta_auc=treatment['auc']-control['auc'],delta_final=treatment['final']-control['final'])
    except Exception:
        report['status']='verification_failed';report['errors'].append(traceback.format_exc())
    files=[]
    for path in sorted(run.rglob('*')):
        if not path.is_file():continue
        row=dict(path=str(path),backup='Linux only')
        try:
            row.update(size=path.stat().st_size,sha256=digest(path))
            if path.suffix in ('.json','.jsonl','.py','.yaml') and row['size']<10*1024**2:
                dest=output/'run'/path.relative_to(run);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes());row['copied']=str(dest.relative_to(output))
        except Exception:
            row['error']=traceback.format_exc();report['status']='verification_failed';report['errors'].append(row['error'])
        files.append(row)
    save(output/'report.json',report);save(output/'seal.json',dict(status=report['status'],files=files));save(marker,dict(status=report['status'],output=str(output),report_sha256=digest(output/'report.json'),job=job))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[1,2,3,4],required=True);a=p.parse_args();main(a.seed)
