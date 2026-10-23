"""Fail-closed task process supervisor; all task compute runs through this entry."""
import argparse, datetime, fcntl, json, os, signal, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
DEADLINE=None
LOCK=Path('/home/lzy/Projects/SO2/_so2_work/training_resource.lock')
WORK=ROOT/'_so2_work/t014'
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False));os.replace(t,p)
def group_alive(pgid):
    # Ignore zombies: they cannot compute or retain a CUDA context.
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit(): continue
        try:
            fields=(entry/'stat').read_text().rsplit(')',1)[1].split()
            if int(fields[2])==pgid and fields[0]!='Z': return True
        except (FileNotFoundError,ProcessLookupError,PermissionError): pass
    return False

def stop_group(pgid):
    if not group_alive(pgid): return
    try: os.killpg(pgid,signal.SIGTERM)
    except ProcessLookupError: return
    until=time.monotonic()+5
    while group_alive(pgid) and time.monotonic()<until: time.sleep(.1)
    if group_alive(pgid):
        try: os.killpg(pgid,signal.SIGKILL)
        except ProcessLookupError: pass
    # Keep the resource lock until no live member can retain GPU work.
    while group_alive(pgid): time.sleep(.1)

def supervise(job,command,seconds,work=WORK,deadline=DEADLINE,total=86400,lockpath=LOCK,check_gpu=True,acknowledged_cpu_failures=()):
    work.mkdir(parents=True,exist_ok=True)
    with lockpath.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=work/'budget.json'; state=json.loads(path.read_text()) if path.exists() else {'used_seconds':0,'jobs':[]}
        acks=set(acknowledged_cpu_failures)
        eligible={j['job'] for j in state['jobs'] if j['status']=='partial' and j['job'].startswith('d02-cpu-semantics-v')}
        assert acks <= eligible, 'only explicitly inspected CPU semantics failures may be acknowledged'
        assert all(j['status']=='completed' or (j['status']=='partial' and j['job'] in acks) for j in state['jobs']), 'prior failure/running job: no automatic restart'
        assert job not in [j['job'] for j in state['jobs']], 'duplicate job'
        assert (deadline is None or time.time()<deadline) and state['used_seconds']<total
        if check_gpu:
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip(), 'GPU occupied'
        # Reserve 30s inside the cumulative budget for TERM/KILL and bookkeeping.
        limit=min(seconds,total-state['used_seconds']-30,deadline-time.time() if deadline is not None else seconds)
        assert limit>0, 'insufficient remaining budget for safe cleanup'
        base=state['used_seconds']; start=time.monotonic()
        item={'job':job,'status':'running','command':command,'start_unix':time.time(),'limit_seconds':limit,'supervisor_pid':os.getpid()}
        item['explicitly_acknowledged_prior_CPU_failures']=sorted(acks)
        state['jobs'].append(item);write(path,state)
        proc=None; reason=None; stop=False
        def interrupted(sig,frame):
            nonlocal stop
            stop=True
        signal.signal(signal.SIGTERM,interrupted); signal.signal(signal.SIGINT,interrupted)
        try:
            with (work/(job+'.log')).open('x') as log:
                proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                item['pid']=item['pgid']=proc.pid
                while proc.poll() is None:
                    elapsed=time.monotonic()-start
                    state['used_seconds']=base+elapsed;item['wall_seconds']=elapsed;write(path,state)
                    if stop or elapsed>=limit or (deadline is not None and time.time()>=deadline):
                        reason='signal' if stop else 'deadline_or_budget'
                        os.killpg(proc.pid,signal.SIGTERM)
                        try:proc.wait(timeout=15)
                        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                        break
                    time.sleep(.25)
                # Remove any descendants left behind by a failed root.
                stop_group(proc.pid)
                code=proc.returncode
            item.update(status='completed' if code==0 and reason is None else 'partial',exit_code=code,stop_reason=reason)
        except BaseException as exc:
            item.update(status='partial',error=repr(exc))
            if proc:
                stop_group(proc.pid)
                if proc.poll() is None: proc.wait()
            raise
        finally:
            item.update(end_unix=time.time(),wall_seconds=time.monotonic()-start)
            state['used_seconds']=base+item['wall_seconds'];state['deadline_unix']=deadline;state['limit_seconds']=total
            write(path,state)
        return 0 if item['status']=='completed' else 1
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--seconds',type=float,required=True);p.add_argument('--acknowledge-cpu-failure',action='append',default=[]);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    command=a.command[1:] if a.command[0]=='--' else a.command
    sys.exit(supervise(a.job,command,a.seconds,acknowledged_cpu_failures=a.acknowledge_cpu_failure))
