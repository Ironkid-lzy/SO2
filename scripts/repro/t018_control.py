"""Single-phase dispatch, finite supervision and independent CPU terminal seal."""
import fcntl,hashlib,json,math,os,signal,subprocess,sys,time
from pathlib import Path
from t018_supervise_base import supervise,group_alive
ROOT=Path(__file__).resolve().parents[2];WORK=ROOT/'_so2_work/t018'
LOCK=Path('/home/lzy/Projects/SO2/_so2_work/training_resource.lock')
PY='/home/lzy/Projects/SO2/.venv/bin/python'
JOBS={'preflight':'cost-preflight-v1','offline':'offline-v1','online':'online-v1'}
RUNS={'preflight':WORK/'cost-preflight-v1','offline':ROOT/'_so2_work/runs/EXP-037/EXP-037-s0-attempt1','online':ROOT/'_so2_work/runs/EXP-038/EXP-038-s0-attempt1'}

def read(p):return json.loads(Path(p).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def gpu():return subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
def alive(pid):
    try:return Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False

def permit_gate(p,now):
    assert p['allowed'] and p['ordinaryUsageAllowed'] and 0<=now-p['checked_unix']<=1800
    assert p['primary_used']<90 and p['weekly_used']<100

def schedule(phase,ledger,cost=None,offline_sealed=False):
    job=JOBS[phase]
    assert all(j['status']=='completed' for j in ledger['jobs']), 'failed/running prior job'
    assert job not in [j['job'] for j in ledger['jobs']], 'duplicate'
    assert 0<=ledger['used_seconds']<50400
    assert any(j['job']=='semantic-validation-v1' for j in ledger['jobs'])
    if phase=='preflight':
        assert ledger['used_seconds']+1500+60<=1800
        return 1500
    assert cost and cost['passed']
    assert any(j['job']=='cost-preflight-v1' for j in ledger['jobs'])
    limits={k:math.ceil(cost[k+'_with30percent'])+180 for k in ('offline','online')}
    assert limits['online']<18000, 'frozen A1 finite inner guard'
    if phase=='offline':assert sum(limits.values())+ledger['used_seconds']+60<=50400
    else:
        assert offline_sealed and any(j['job']=='offline-v1' for j in ledger['jobs'])
        assert limits['online']+ledger['used_seconds']+60<=50400
    return limits[phase]

def fault_tests():
    b={'used_seconds':3,'jobs':[{'job':'semantic-validation-v1','status':'completed'}]}
    assert schedule('preflight',b)==1500
    checks=0
    cases=[lambda:schedule('preflight',{'used_seconds':3,'jobs':[{'job':'semantic-validation-v1','status':'partial'}]}),
      lambda:schedule('preflight',{'used_seconds':3,'jobs':b['jobs']+[{'job':'cost-preflight-v1','status':'completed'}]}),
      lambda:schedule('preflight',{'used_seconds':300,'jobs':b['jobs']}),
      lambda:permit_gate(dict(allowed=True,ordinaryUsageAllowed=True,checked_unix=0,primary_used=0,weekly_used=0),1801),
      lambda:permit_gate(dict(allowed=True,ordinaryUsageAllowed=True,checked_unix=0,primary_used=90,weekly_used=0),0),
      lambda:permit_gate(dict(allowed=True,ordinaryUsageAllowed=True,checked_unix=0,primary_used=0,weekly_used=100),0),
      lambda:schedule('online',{'used_seconds':3,'jobs':b['jobs']+[{'job':'cost-preflight-v1','status':'completed'}]},dict(passed=True,offline_with30percent=100,online_with30percent=100),False),
      lambda:schedule('offline',{'used_seconds':3,'jobs':b['jobs']+[{'job':'cost-preflight-v1','status':'completed'}]},dict(passed=True,offline_with30percent=49000,online_with30percent=10000))]
    for fn in cases:
        try:fn()
        except AssertionError:checks+=1
        else:raise AssertionError('fail-closed gate accepted invalid state')
    return checks

def dispatch(phase):
    job=JOBS[phase];WORK.mkdir(parents=True,exist_ok=True)
    with (WORK/'dispatch.lock').open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (WORK/(phase+'-dispatch.json')).exists() and not RUNS[phase].exists()
        receipt=read(WORK/'formal-freeze.json')
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==receipt['code_sha']
        assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
        assert all(digest(ROOT/p)==v for p,v in receipt['source_sha256'].items())
        assert digest(WORK/'protocol.json')==receipt['protocol_sha256']
        permit=read(WORK/('permit-'+job+'.json'));permit_gate(permit,time.time())
        assert permit['job']==job
        offline_seal=read(WORK/'offline-sealed.json') if (WORK/'offline-sealed.json').exists() else None
        cost=read(WORK/'cost-preflight-v1/result.json') if phase!='preflight' else None
        if phase!='preflight':assert read(WORK/'preflight-sealed.json')['status']=='completed'
        outer=schedule(phase,read(WORK/'budget.json'),cost,offline_seal and offline_seal['status']=='completed')
        assert not gpu()
        with LOCK.open('a+') as resource:fcntl.flock(resource,fcntl.LOCK_EX|fcntl.LOCK_NB)
        env=os.environ.copy();env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MUJOCO_GL='osmesa',D4RL_SUPPRESS_IMPORT_ERROR='1',LD_LIBRARY_PATH='/home/lzy/.mujoco/mujoco210/bin:/usr/lib/nvidia:/usr/lib/x86_64-linux-gnu:/home/lzy/Projects/SO2/.venv/lib/python3.10/site-packages/torch/lib')
        with (WORK/(phase+'-console.log')).open('x') as log:
            proc=subprocess.Popen([PY,'scripts/repro/t018_control.py','supervise',phase,str(outer)],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        d=dict(job=job,phase=phase,supervisor_pid=proc.pid,started_unix=time.time(),outer_seconds=outer,inner_seconds=outer-60,quota_permit=permit,freeze=receipt)
        save(WORK/(phase+'-dispatch.json'),d)
        env['CUDA_VISIBLE_DEVICES']=''
        with (WORK/(phase+'-sealer.log')).open('x') as log:
            seal=subprocess.Popen([PY,'scripts/repro/t018_control.py','seal',phase],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        d['sealer_pid']=seal.pid;save(WORK/(phase+'-dispatch.json'),d);print(json.dumps(d))

def supervise_phase(phase,outer):
    command=[PY,'scripts/repro/t018_preflight.py'] if phase=='preflight' else [PY,'scripts/repro/t018_phase.py',phase,str(outer-60)]
    return supervise(JOBS[phase],command,outer,work=WORK,total=1800 if phase=='preflight' else 50400)

def cleanup(pgid):
    assert pgid>1 and pgid!=os.getpgrp()
    for sig,wait in ((signal.SIGTERM,10),(signal.SIGKILL,10)):
        if not group_alive(pgid):return
        try:os.killpg(pgid,sig)
        except ProcessLookupError:return
        until=time.monotonic()+wait
        while group_alive(pgid) and time.monotonic()<until:time.sleep(.2)
    assert not group_alive(pgid),'bounded cleanup failed'

def seal_phase(phase):
    d=read(WORK/(phase+'-dispatch.json'));job=d['job'];deadline=d['started_unix']+d['outer_seconds']+30
    item=None;failure=None
    while True:
        b=read(WORK/'budget.json');item=next((j for j in b['jobs'] if j['job']==job),None)
        if item and item['status']!='running' and not group_alive(item.get('pgid',0)):break
        if time.time()>deadline or not alive(d['supervisor_pid']):
            failure='supervisor_lost_or_independent_deadline'
            if item and item.get('pgid'):cleanup(item['pgid'])
            if alive(d['supervisor_pid']):
                os.kill(d['supervisor_pid'],signal.SIGTERM)
            break
        time.sleep(.5)
    until=time.monotonic()+20
    while alive(d['supervisor_pid']) and time.monotonic()<until:time.sleep(.2)
    if alive(d['supervisor_pid']):os.kill(d['supervisor_pid'],signal.SIGKILL)
    with LOCK.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not item or not group_alive(item.get('pgid',0))
        assert not gpu()
        t=time.monotonic();result=dict(status='partial',job=job,quiescent=True,failure=failure)
        b=read(WORK/'budget.json');item=next((j for j in b['jobs'] if j['job']==job),None)
        if failure:
            # Unknown run elapsed is conservatively charged to the entire reservation.
            if item:
                known=item.get('wall_seconds',0);charge=max(known,d['outer_seconds'])
                b['used_seconds']+=charge-known;item.update(status='partial',wall_seconds=charge,accounting='conservative reservation after lost supervisor')
            else:
                item=dict(job=job,status='partial',wall_seconds=d['outer_seconds'],accounting='unknown missing entry charged full reservation')
                b['jobs'].append(item);b['used_seconds']+=d['outer_seconds']
        elif item and item['status']=='completed' and item.get('exit_code')==0:
            try:
                from t018_verify_terminal import verify
                result.update(verify(phase,RUNS[phase]));result['status']='completed'
            except BaseException as exc:result.update(status='verification_failed',failure=repr(exc))
        files={str(p.relative_to(RUNS[phase])):digest(p) for p in sorted(RUNS[phase].rglob('*')) if p.is_file()}
        result.update(files_sha256=files,sealed_unix=time.time(),code_sha=d['freeze']['code_sha'],protocol_sha256=d['freeze']['protocol_sha256'])
        seal_seconds=time.monotonic()-t;b['used_seconds']+=seal_seconds
        if item:item['CPU_terminal_seal_seconds']=seal_seconds
        save(WORK/'budget.json',b);result['budget_used_seconds']=b['used_seconds']
        target=WORK/(phase+'-sealed.json');assert not target.exists();save(target,result)
    print(json.dumps(result));return 0 if result['status']=='completed' else 1

if __name__=='__main__':
    mode,phase=sys.argv[1:3];assert phase in JOBS
    if mode=='dispatch':dispatch(phase)
    elif mode=='supervise':sys.exit(supervise_phase(phase,float(sys.argv[3])))
    elif mode=='seal':sys.exit(seal_phase(phase))
    else:raise ValueError(mode)
