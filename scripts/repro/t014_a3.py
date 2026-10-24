"""A3 fixed identities, fresh quota, single dispatch and bounded supervision."""
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT/'_so2_work/t014/a3'
PROTO = WORK/'protocol.json'
RUN = ROOT/'_so2_work/runs/EXP-036/EXP-036-s0-attempt1'
JOB = 'a3-online-s0-v1'
PYTHON = '/home/lzy/Projects/SO2/.venv/bin/python'


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576), b''): h.update(block)
    return h.hexdigest()


def source_hashes():
    old = read(ROOT/'_so2_work/t014/a1-extension-v1/freeze-receipt.json')['source_sha256']
    for name, value in old.items():
        assert digest(ROOT/name) == value, ('frozen A1 source changed', name)
    files = set(old) | {str(p.relative_to(ROOT)) for p in (ROOT/'scripts/repro').glob('t014_a3*.py')}
    files |= {'scripts/repro/t014_supervise.py','scripts/repro/t014_launch.sh'}
    return {name:digest(ROOT/name) for name in sorted(files)}


def budget_ok(state):
    p = read(PROTO)['budget']
    assert p['additional_seconds_cap'] == 18000
    assert p['stage_endpoint_seconds'] == p['start_T014_seconds'] + 18000 <= 86400
    assert state['used_seconds'] >= p['start_T014_seconds']
    assert all(j['status'] == 'completed' or j['job'] == 'd02-cpu-semantics-v1' for j in state['jobs'])
    return min(p['stage_endpoint_seconds'],86400) - state['used_seconds']


def identity():
    hashes = source_hashes()
    receipt = read(WORK/'freeze-receipt.json')
    assert receipt['source_sha256'] == hashes and receipt['protocol_sha256'] == digest(PROTO)
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip() == receipt['code_sha']
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    return hashes


def permit():
    value = read(WORK/'permit.json')
    assert value['seed'] == 0 and value['allowed'] is True
    assert 0 <= time.time()-value['checked_unix'] <= 1800
    assert value['primary_used'] < 90 and value['weekly_used'] < 60
    return value


def formal_gate():
    hashes = identity()
    for name in ['validation-cpu.json','validation-cuda.json','review-approved.json']:
        value = read(WORK/name)
        assert value['source_sha256'] == hashes and value['protocol_sha256'] == digest(PROTO)
        assert value.get('status') == 'passed' or value.get('approved') is True
    bench = read(WORK/'validation-cuda.json')['benchmark']
    assert bench['formal_admitted'] and bench['median_ratio'] <= 1.10
    p = read(PROTO); c = p['control']
    assert digest(Path(c['metrics_path'])) == c['metrics_sha256']
    assert digest(Path(p['parent']['path'])) == p['parent']['sha256']
    assert shutil.disk_usage(ROOT).free > 8*1024**3
    return hashes, bench['estimated_formal_seconds_with30percent']


def supervise(job, seconds, command):
    from t014_supervise import supervise as bounded
    limits = read(PROTO)['budget']
    maximum = {'a3-cpu-validation-v1':limits['cpu_validation_cap'],
               'a3-cuda-validation-v1':limits['cuda_validation_and_benchmark_cap'],
               JOB:limits['job_outer_seconds']}[job]
    assert 0 < seconds <= maximum
    identity()
    state = read(ROOT/'_so2_work/t014/budget.json')
    assert budget_ok(state) >= seconds + 30
    return bounded(job,command,seconds,total=limits['stage_endpoint_seconds'],
                   acknowledged_cpu_failures=['d02-cpu-semantics-v1'])


def dispatch():
    with (WORK/'dispatch.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        quota = permit(); hashes, estimate = formal_gate()
        state = read(ROOT/'_so2_work/t014/budget.json');remaining = budget_ok(state)
        p = read(PROTO)['budget']
        assert estimate + 60 <= min(remaining,p['job_inner_seconds'],p['job_outer_seconds'])
        assert not any(j['job'] == JOB for j in state['jobs'])
        assert not RUN.exists() and not (WORK/'dispatch.json').exists() and not (WORK/'formal-start.json').exists()
        assert not list(Path('/home/lzy/Projects').glob('SO2*/_so2_work/runs/EXP-036*'))
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
        launch = str(ROOT/'scripts/repro/t014_launch.sh')
        command = ['bash',launch,'scripts/repro/t014_a3.py','supervise','--job',JOB,'--seconds',str(p['job_outer_seconds']),'--','bash',launch,'scripts/repro/t014_a3_run.py','--wall-seconds',str(p['job_inner_seconds'])]
        with (WORK/'supervisor-console.log').open('x') as f:
            proc = subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        with (WORK/'sealer-console.log').open('x') as f:
            env = os.environ.copy();env['CUDA_VISIBLE_DEVICES'] = ''
            sealer = subprocess.Popen([PYTHON,'scripts/repro/t014_a3_seal.py'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=env)
        record = dict(supervisor_pid=proc.pid,sealer_pid=sealer.pid,command=command,quota_permit=quota,started_unix=time.time(),source_sha256=hashes)
        save(WORK/'dispatch.json',record)
        print(json.dumps(record))


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('action',choices=['dispatch','supervise'])
    p.add_argument('--job');p.add_argument('--seconds',type=float);p.add_argument('command',nargs=argparse.REMAINDER)
    a = p.parse_args()
    if a.action == 'dispatch': dispatch()
    else:
        # REMAINDER parsing uses explicit separator; split options first below.
        q = argparse.ArgumentParser();q.add_argument('--job',required=True);q.add_argument('--seconds',type=float,required=True);q.add_argument('command',nargs=argparse.REMAINDER)
        b = q.parse_args(sys.argv[2:]);cmd = b.command[1:] if b.command[0] == '--' else b.command
        sys.exit(supervise(b.job,b.seconds,cmd))
