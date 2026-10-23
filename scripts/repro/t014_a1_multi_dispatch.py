"""Dispatch exactly the next seed with fresh quota; never automatic under model hold."""
import argparse
import fcntl
from t014_a1_multi import *
def dispatch(seed):
    protocol,record,work,run=settings(seed)
    with (STAGE/'dispatch.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        permit=quota(seed);hashes=source_hashes();receipt=read(STAGE/'freeze-receipt.json')
        assert hashes==receipt['source_sha256'] and digest(PROTO)==receipt['protocol_sha256'] and digest(QUEUE)==receipt['queue_sha256']
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==receipt['code_sha']
        assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
        for value in (read(STAGE/'review-approved.json'),read(STAGE/'identity-validation.json')):
            assert value['source_sha256']==hashes and value['protocol_sha256']==digest(PROTO)
        assert read(STAGE/'review-approved.json')['approved'] and read(STAGE/'identity-validation.json')['status']=='passed'
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
        assert not run.exists() and not (work/'dispatch.json').exists()
        state=read(ROOT/'_so2_work/t014/budget.json')
        assert all(j['status']=='completed' or j['job']=='d02-cpu-semantics-v1' for j in state['jobs'])
        assert not any(j['job']==f'a1-ext-s{seed}-v1' for j in state['jobs'])
        for previous in range(1,seed):assert read(STAGE/f'seed{previous}/sealed.json')['status']=='completed'
        assert protocol['budget']['formal_estimate_with30percent']+60<=min(protocol['budget']['stage_endpoint_seconds'],86400)-state['used_seconds']
        assert shutil.disk_usage(ROOT).free>8*1024**3
        work.mkdir(parents=True,exist_ok=True)
        launch=str(ROOT/'scripts/repro/t014_launch.sh');py='/home/lzy/Projects/SO2/.venv/bin/python'
        command=['bash',launch,'scripts/repro/t014_a1_multi_supervise.py','--job',f'a1-ext-s{seed}-v1','--seconds',str(protocol['budget']['job_outer_seconds']),'--','bash',launch,'scripts/repro/t014_a1_multi_run.py','--seed',str(seed),'--wall-seconds',str(protocol['budget']['job_inner_seconds'])]
        with (work/'supervisor-console.log').open('x') as f:
            proc=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        with (work/'sealer-console.log').open('x') as f:
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
            sealer=subprocess.Popen([py,'scripts/repro/t014_a1_multi_seal.py','--seed',str(seed)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=env)
        save(work/'dispatch.json',dict(seed=seed,supervisor_pid=proc.pid,sealer_pid=sealer.pid,command=command,quota_permit=permit,started_unix=time.time()))
        print(json.dumps(read(work/'dispatch.json')))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[1,2,3,4],required=True);a=p.parse_args();dispatch(a.seed)
