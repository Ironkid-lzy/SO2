"""Fresh-permit single-stage dispatch; one global T019 ledger and CPU sealer."""
import fcntl,hashlib,json,math,os,signal,subprocess,sys,time
from pathlib import Path
from t018_supervise_base import supervise,group_alive
ROOT=Path(__file__).resolve().parents[2]
BASEWORK=ROOT/'_so2_work/t019'
WORK=BASEWORK
LOCK=Path('/home/lzy/Projects/SO2/_so2_work/training_resource.lock')
PY='/home/lzy/Projects/SO2/.venv/bin/python'
TOTAL=151200

def read(p):return json.loads(Path(p).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def gpu():return subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
def alive(pid):
    try:return Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False
def job_for(phase,seed):return 'identity-preflight-v1' if phase=='precheck' else f's{seed}-{phase}-v1'
def run_for(phase,seed):return BASEWORK/'precheck' if phase=='precheck' else ROOT/f'_so2_work/runs/EXP-{39 if phase=="offline" else 40:03d}/EXP-{39 if phase=="offline" else 40:03d}-s{seed}-attempt1'
def stage_for(phase,seed):return BASEWORK if phase=='precheck' else BASEWORK/f'seed{seed}'
def seal_for(phase,seed):return stage_for(phase,seed)/(phase+'-sealed.json')
def permit_gate(p,now,phase,seed):
    assert p['allowed'] and p['ordinaryUsageAllowed'] and 0<=now-p['checked_unix']<=1800
    assert p['primary_used']<90 and p['weekly_used']<100
    assert p['job']==job_for(phase,seed) and p['seed']==seed

def schedule(phase,seed,ledger,protocol=None):
    p=protocol or read(BASEWORK/'protocol.json')
    sequence=[dict(phase='precheck',seed=1,job='identity-preflight-v1')]+p['execution_order']
    assert seed in (1,2,3,4) and phase in ('precheck','offline','online')
    index=len(ledger['jobs']);assert index<len(sequence)
    assert (phase,seed)==(sequence[index]['phase'],sequence[index]['seed']),'not the frozen next stage'
    assert [j['job'] for j in ledger['jobs']]==[q['job'] for q in sequence[:index]],'duplicate/out-of-order ledger'
    assert all(j['status']=='completed' and j.get('exit_code')==0 for j in ledger['jobs'])
    assert 0<=ledger['used_seconds']<TOTAL
    outer=p['phase_limits_seconds'][phase]['outer']
    remaining=sequence[index:]
    reservation=sum(p['phase_limits_seconds'][q['phase']]['outer']+60 for q in remaining)
    assert ledger['used_seconds']+reservation<=TOTAL,'remaining complete queue including cleanup does not fit'
    for ph in ('offline','online'):
        observed=[j['wall_seconds'] for j in ledger['jobs'] if j['job'].endswith('-'+ph+'-v1')]
        reference=p['cost_admission']['reference_'+ph+'_full_seconds']
        estimate=1.3*max([reference]+observed)
        assert estimate<=p['phase_limits_seconds'][ph]['inner'],'30 percent estimate does not fit frozen inner timeout'
    pre=sum(j['wall_seconds']+j.get('CPU_terminal_seal_seconds',0) for j in ledger['jobs'] if j['job']=='identity-preflight-v1')
    assert pre<=3600
    return outer

def frozen():
    receipt=read(BASEWORK/'formal-freeze.json')
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==receipt['code_sha']
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    assert all(digest(ROOT/p)==v for p,v in receipt['source_sha256'].items())
    assert digest(BASEWORK/'protocol.json')==receipt['protocol_sha256']
    assert digest(BASEWORK/'queue-manifest.json')==receipt['queue_sha256']
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==receipt['GPU_uuid'], 'hardware changed: old measured rates not eligible'
    return receipt

def dispatch(phase,seed):
    job=job_for(phase,seed);work=stage_for(phase,seed);work.mkdir(parents=True,exist_ok=True)
    with (BASEWORK/'dispatch.lock').open('a+') as gate:
        fcntl.flock(gate,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (work/(phase+'-dispatch.json')).exists() and not run_for(phase,seed).exists(),'already started'
        receipt=frozen();permit=read(BASEWORK/('permit-'+job+'.json'));permit_gate(permit,time.time(),phase,seed)
        b=read(BASEWORK/'budget.json');outer=schedule(phase,seed,b)
        p=read(BASEWORK/'protocol.json');sequence=[dict(phase='precheck',seed=1)]+p['execution_order']
        for previous in sequence[:len(b['jobs'])]:
            sealed=read(seal_for(previous['phase'],previous['seed']))
            assert sealed['status']=='completed' and sealed['quiescent'],'previous stage not technically sealed'
        if phase!='precheck':assert read(BASEWORK/'precheck-sealed.json')['all_seeds_exact']
        subgroup=sum(j['wall_seconds']+j.get('CPU_terminal_seal_seconds',0) for j in b['jobs'] if j['job']=='identity-preflight-v1')
        subgroup+=sum(read(x)['seconds'] for x in BASEWORK.glob('seed*/online-parent-check/validation-cost.json'))
        assert subgroup+(300 if phase=='online' else 0)<=3600
        assert __import__('shutil').disk_usage(ROOT).free>12*1024**3
        assert not gpu()
        with LOCK.open('a+') as resource:fcntl.flock(resource,fcntl.LOCK_EX|fcntl.LOCK_NB)
        env=os.environ.copy();env.update(SO2_T019_SEED=str(seed),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MUJOCO_GL='osmesa',D4RL_SUPPRESS_IMPORT_ERROR='1',LD_LIBRARY_PATH='/home/lzy/.mujoco/mujoco210/bin:/usr/lib/nvidia:/usr/lib/x86_64-linux-gnu:/home/lzy/Projects/SO2/.venv/lib/python3.10/site-packages/torch/lib')
        if phase=='precheck':env['CUDA_VISIBLE_DEVICES']=''
        with (work/(phase+'-console.log')).open('x') as log:
            proc=subprocess.Popen([PY,'scripts/repro/t019_control.py','supervise',phase,str(seed),str(outer)],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        d=dict(job=job,phase=phase,seed=seed,supervisor_pid=proc.pid,started_unix=time.time(),outer_seconds=outer,inner_seconds=outer-60,quota_permit=permit,freeze=receipt)
        save(work/(phase+'-dispatch.json'),d);env['CUDA_VISIBLE_DEVICES']=''
        with (work/(phase+'-sealer.log')).open('x') as log:
            sealer=subprocess.Popen([PY,'scripts/repro/t019_control.py','seal',phase,str(seed)],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        d['sealer_pid']=sealer.pid;save(work/(phase+'-dispatch.json'),d);print(json.dumps(d))

def cleanup(pgid):
    assert pgid>1 and pgid!=os.getpgrp()
    for sig,wait in ((signal.SIGTERM,10),(signal.SIGKILL,10)):
        if not group_alive(pgid):return
        try:os.killpg(pgid,sig)
        except ProcessLookupError:return
        until=time.monotonic()+wait
        while group_alive(pgid) and time.monotonic()<until:time.sleep(.2)
    assert not group_alive(pgid),'bounded cleanup failed'

def seal_phase(phase,seed):
    work=stage_for(phase,seed);d=read(work/(phase+'-dispatch.json'));job=d['job'];deadline=d['started_unix']+d['outer_seconds']+30
    item=None;failure=None
    while True:
        b=read(BASEWORK/'budget.json');item=next((j for j in b['jobs'] if j['job']==job),None)
        if item and item['status']!='running' and not group_alive(item.get('pgid',0)):break
        if time.time()>deadline or not alive(d['supervisor_pid']):
            failure='supervisor_lost_or_independent_deadline'
            if item and item.get('pgid'):cleanup(item['pgid'])
            if alive(d['supervisor_pid']):os.kill(d['supervisor_pid'],signal.SIGTERM)
            break
        time.sleep(.5)
    until=time.monotonic()+20
    while alive(d['supervisor_pid']) and time.monotonic()<until:time.sleep(.2)
    if alive(d['supervisor_pid']):os.kill(d['supervisor_pid'],signal.SIGKILL)
    with LOCK.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not item or not group_alive(item.get('pgid',0));assert not gpu()
        started=time.monotonic();result=dict(status='partial',job=job,seed=seed,quiescent=True,failure=failure)
        b=read(BASEWORK/'budget.json');item=next((j for j in b['jobs'] if j['job']==job),None)
        if failure:
            if item:
                known=item.get('wall_seconds',0);charge=max(known,d['outer_seconds']);b['used_seconds']+=charge-known
                item.update(status='partial',wall_seconds=charge,accounting='conservative full reservation after supervisor loss')
            else:
                item=dict(job=job,status='partial',wall_seconds=d['outer_seconds'],accounting='missing entry, charged full reservation')
                b['jobs'].append(item);b['used_seconds']+=d['outer_seconds']
        elif item and item['status']=='completed' and item.get('exit_code')==0:
            try:
                from t019_verify_terminal import verify
                result.update(verify(phase,run_for(phase,seed)));result['status']='completed'
            except BaseException as exc:result.update(status='verification_failed',failure=repr(exc))
        result.update(files_sha256={str(p.relative_to(run_for(phase,seed))):digest(p) for p in sorted(run_for(phase,seed).rglob('*')) if p.is_file()},sealed_unix=time.time(),code_sha=d['freeze']['code_sha'],protocol_sha256=d['freeze']['protocol_sha256'])
        elapsed=time.monotonic()-started;b['used_seconds']+=elapsed
        if item:item['CPU_terminal_seal_seconds']=elapsed
        save(BASEWORK/'budget.json',b);result['budget_used_seconds']=b['used_seconds']
        target=seal_for(phase,seed);assert not target.exists();save(target,result)
    print(json.dumps(result));return 0 if result['status']=='completed' else 1

if __name__=='__main__':
    mode,phase=sys.argv[1:3];seed=int(sys.argv[3]);assert seed in (1,2,3,4)
    os.environ['SO2_T019_SEED']=str(seed)
    if mode=='dispatch':dispatch(phase,seed)
    elif mode=='supervise':
        outer=float(sys.argv[4]);cmd=[PY,'scripts/repro/t019_precheck.py'] if phase=='precheck' else [PY,'scripts/repro/t019_phase.py',phase,str(outer-60)]
        sys.exit(supervise(job_for(phase,seed),cmd,outer,work=BASEWORK,total=TOTAL))
    elif mode=='seal':sys.exit(seal_phase(phase,seed))
    else:raise ValueError(mode)
