"""Independent finite CPU finalizer for one extension seed, never starts another."""
import argparse
import traceback
import signal
from t014_a1_multi import *
from t014_a1_seal import evaluate
from t014_supervise import group_alive


def quiesce(job,seed):
    groups={job['pgid']} if job and job.get('pgid') else set()
    # Recover a missing child PID only from this unique runner, seed and checkout.
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:
            args=(entry/'cmdline').read_bytes().split(b'\0')
            if b'scripts/repro/t014_a1_multi_run.py' not in args:continue
            if b'--seed' not in args or args[args.index(b'--seed')+1]!=str(seed).encode():continue
            if Path(os.readlink(entry/'cwd')).resolve()!=ROOT:continue
            groups.add(os.getpgid(int(entry.name)))
        except (FileNotFoundError,ProcessLookupError,PermissionError,IndexError):continue
    start=time.monotonic()
    for sig,seconds in ((signal.SIGTERM,5),(signal.SIGKILL,5)):
        for pgid in groups:
            if group_alive(pgid):
                try:os.killpg(pgid,sig)
                except ProcessLookupError:pass
        end=time.monotonic()+seconds
        while any(group_alive(pgid) for pgid in groups) and time.monotonic()<end:time.sleep(.1)
    return dict(pgroups=sorted(groups),live_after_cleanup=[g for g in groups if group_alive(g)],cleanup_seconds=time.monotonic()-start)


def main(seed):
    protocol,record,work,run=settings(seed);marker=work/'sealed.json'
    if marker.exists():return
    end=time.time()+15840;job=None;lost=False
    while time.time()<end:
        state=read(ROOT/'_so2_work/t014/budget.json');job=next((j for j in state['jobs'] if j['job']==f'a1-ext-s{seed}-v1'),None)
        if job and job['status'] in ('completed','partial'):break
        dispatch=work/'dispatch.json'
        dispatch_record=read(dispatch) if dispatch.exists() else {}
        start_hint=(job or {}).get('start_unix',dispatch_record.get('started_unix'))
        if start_hint is not None and time.time()>=start_hint+protocol['budget']['job_outer_seconds']:
            lost=True;break
        if dispatch_record and not Path('/proc/'+str(dispatch_record['supervisor_pid'])).exists():
            lost=True;break
        time.sleep(5)
    else:lost=True
    output=ROOT/f'_so2_work/deliveries/t014-a1-ext-s{seed}-v1';output.mkdir(parents=True,exist_ok=False)
    abnormal=lost or (job and job.get('pgid') and group_alive(job['pgid']))
    cleanup=quiesce(job,seed) if abnormal else dict(pgroups=[],live_after_cleanup=[],cleanup_seconds=0)
    report=dict(status='partial' if lost else job['status'],seed=seed,job=job,T014_used_seconds=state['used_seconds'],errors=[],no_next_job=True,cleanup=cleanup)
    if abnormal:
        dispatch_record=read(work/'dispatch.json') if (work/'dispatch.json').exists() else {}
        start_hint=(job or {}).get('start_unix',dispatch_record.get('started_unix'))
        charge=None if start_hint is None else max(protocol['budget']['job_outer_seconds'],time.time()-start_hint,(job or {}).get('wall_seconds',0))
        report.update(status='partial',accounting_uncertain=True,original_ledger_preserved=True,
                      conservative_full_job_charge_seconds=charge,missing_start_time=charge is None)
        report['errors'].append('Missing/invalid terminal supervisor record or deadline expired; no restart. Whole-job conservative charge=max(outer, elapsed to cleanup end, recorded wall); missing start remains unknown and blocks all later seeds.')
    if cleanup['live_after_cleanup']:report['status']='live_unsealed'
    if job is None:job={'status':'partial','error':'No ledger job; supervised startup failed'}
    try:
        if report['status']=='completed':
            assert read(run/'acceptance.json')['accepted']
            assert read(run/'actual-initial-eval.json')['initial_returns_exact_control']
            assert digest(Path(record['control_metrics_path']))==record['control_metrics_sha256']
            treatment=evaluate(run/'metrics.jsonl');control=evaluate(Path(record['control_metrics_path']))
            assert treatment['points'][0]['returns']==control['points'][0]['returns']
            report.update(treatment=treatment,control=control,delta_auc=treatment['auc']-control['auc'],delta_final=treatment['final']-control['final'])
    except Exception:
        report['status']='verification_failed';report['errors'].append(traceback.format_exc())
    files=[]
    # Active output is never called a static seal. Preserve server files in place.
    for path in ([] if cleanup['live_after_cleanup'] else sorted(run.rglob('*'))):
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
