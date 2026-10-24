"""Finite, non-retrying T017 diagnostic launch and quiescent artifact seal."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT/'_so2_work/t017'
LOCK = Path('/home/lzy/Projects/SO2/_so2_work/training_resource.lock')
PYTHON = '/home/lzy/Projects/SO2/.venv/bin/python'
JOB = 'd04-es-final-v1'


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def save(p, value):
    path = Path(p)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(tmp, path)


def gpu_pids():
    return subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                                    '--format=csv,noheader,nounits'],text=True).strip()


def supervise_and_seal():
    from t017_supervise_base import supervise, group_alive
    code = supervise(JOB,[PYTHON,'scripts/repro/t017_es_diagnostic.py'],900,
                     work=WORK,total=1800)
    budget = read(WORK/'budget.json')
    item = next(j for j in budget['jobs'] if j['job'] == JOB)
    with LOCK.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not group_alive(item['pgid']) and not gpu_pids()
        out = WORK/'d04-es-final-v1'
        files = {str(p.relative_to(out)):digest(p) for p in sorted(out.rglob('*')) if p.is_file()}
        status = 'completed' if code == 0 and read(out/'summary.json')['status'] == 'completed' else 'partial'
        assert not (WORK/'sealed.json').exists()
        save(WORK/'sealed.json',dict(status=status,job=JOB,budget_used_seconds=budget['used_seconds'],
             child_pgid_quiescent=True,gpu_compute_empty=True,files_sha256=files,
             protocol_sha256=digest(WORK/'diagnostic-protocol.json'),sealed_unix=time.time()))
    return code


def dispatch():
    with (WORK/'dispatch.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (WORK/'dispatch.json').exists()
        assert not (WORK/'d04-es-final-v1').exists() and not (WORK/'sealed.json').exists()
        assert not read(WORK/'budget.json')['jobs']
        permit = read(WORK/'permit.json')
        assert permit['allowed'] and 0 <= time.time()-permit['checked_unix'] <= 1800
        assert permit['primary_used'] < 90 and permit['weekly_used'] < 60
        receipt = read(WORK/'freeze-receipt.json')
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip() == receipt['code_sha']
        assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
        assert digest(WORK/'diagnostic-protocol.json') == receipt['protocol_sha256']
        assert all(digest(ROOT/p)==v for p,v in receipt['source_sha256'].items())
        assert not gpu_pids()
        with LOCK.open('a+') as resource:
            fcntl.flock(resource,fcntl.LOCK_EX|fcntl.LOCK_NB)
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MUJOCO_GL='osmesa',D4RL_SUPPRESS_IMPORT_ERROR='1',LD_LIBRARY_PATH='/home/lzy/.mujoco/mujoco210/bin:/usr/lib/nvidia:/usr/lib/x86_64-linux-gnu:/home/lzy/Projects/SO2/.venv/lib/python3.10/site-packages/torch/lib')
        with (WORK/'supervisor-console.log').open('x') as log:
            proc = subprocess.Popen([PYTHON,'scripts/repro/t017_dispatch.py','--supervise'],
                cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True)
        save(WORK/'dispatch.json',dict(job=JOB,supervisor_pid=proc.pid,started_unix=time.time(),
             freeze_receipt=receipt,quota_permit=permit))
        print(json.dumps(read(WORK/'dispatch.json')))


if __name__ == '__main__':
    if sys.argv[1:] == ['--supervise']:
        sys.exit(supervise_and_seal())
    assert sys.argv[1:] == ['dispatch']
    dispatch()
