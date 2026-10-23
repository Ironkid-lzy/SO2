"""Bounded multi-seed orchestration identities; frozen A1 algorithm is imported unchanged."""
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
STAGE=ROOT/'_so2_work/t014/a1-extension-v1'
PROTO=STAGE/'protocol.json'
QUEUE=STAGE/'queue-manifest.json'
NEW_FILES=['t014_a1_multi.py','t014_a1_multi_run.py','t014_a1_multi_supervise.py','t014_a1_multi_dispatch.py','t014_a1_multi_seal.py']

def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');q.replace(p)
def settings(seed):
    assert seed in (1,2,3,4)
    p=read(PROTO);record=next(x for x in p['parents'] if x['seed']==seed)
    return p,record,STAGE/f'seed{seed}',ROOT/f'_so2_work/runs/EXP-035/EXP-035-s{seed}-attempt1'
def source_hashes():
    protocol=read(PROTO)
    for name,expected in protocol['frozen_seed0_sources'].items():
        assert digest(ROOT/name)==expected, 'frozen seed0 source changed: '+name
    return {**protocol['frozen_seed0_sources'],**{f'scripts/repro/{name}':digest(ROOT/'scripts/repro'/name) for name in NEW_FILES}}
def quota(seed):
    permit=read(STAGE/f'permit-seed{seed}.json')
    assert permit['seed']==seed and permit['allowed']
    assert 0<=time.time()-permit['checked_unix']<=1800
    assert permit['primary_used']<90 and permit['weekly_used']<60
    assert not (STAGE/'quota-hold.json').exists()
    return permit

def verified_parent(record):
    from t014_a1_common import torch,tensor_hash,DATASET,DATA_SHA,CONFIG
    assert digest(DATASET)==DATA_SHA==record['dataset_sha256']
    assert digest(CONFIG)==record['config_sha256']
    p=Path(record['parent_path']);assert 'EXP-027' in str(p) and f"s{record['seed']}-" in str(p)
    assert digest(p)==record['parent_sha256']
    saved=torch.load(p,map_location='cpu',weights_only=False)
    assert saved['updates']==3000000 and saved['train_env_steps']==0
    assert saved['identity']['seed']==record['seed'] and saved['identity']['git_commit']==record['parent_code_sha']
    assert saved['identity']['dataset_sha256']==record['dataset_sha256']
    weights={k:saved[k] for k in ('model','target_model')}
    assert all(tensor_hash(weights[k])==record['network_hashes'][k] for k in weights)
    assert digest(Path(record['export_path']))==record['export_sha256']
    safe=torch.load(record['export_path'],map_location='cpu',weights_only=True)
    assert all(tensor_hash(safe[k])==record['network_hashes'][k] for k in weights)
    assert digest(Path(record['control_metrics_path']))==record['control_metrics_sha256']
    return weights,dict(record)

def make_seed_policy(device,seed):
    from t014_a1_common import install,setup_config,compile_config,create_policy
    install();cfg,create=setup_config(device,online=True)
    cfg=compile_config(cfg,seed=seed,env=None,auto=True,create_cfg=create,save_cfg=False)
    cfg.policy.type+='_command';cfg.policy.import_names=['ding.policy.edac','ding.policy.command_mode_policy_instance']
    policy=create_policy(cfg.policy,model=None,enable_field=['learn','collect','eval','command'])
    return policy,cfg

def admission(seed,seconds):
    protocol,record,work,run=settings(seed)
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    hashes=source_hashes();receipt=read(STAGE/'freeze-receipt.json')
    assert hashes==receipt['source_sha256'] and digest(PROTO)==receipt['protocol_sha256']
    assert digest(QUEUE)==receipt['queue_sha256']
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==receipt['code_sha']
    review=read(STAGE/'review-approved.json');identity=read(STAGE/'identity-validation.json')
    for value in (review,identity):
        assert value['source_sha256']==hashes and value['protocol_sha256']==digest(PROTO)
    assert review['approved'] and identity['status']=='passed'
    quota(seed)
    assert not run.exists() and not (work/'formal-start.json').exists()
    for previous in range(1,seed):
        result=read(STAGE/f'seed{previous}/sealed.json');assert result['status']=='completed'
    budget=read(ROOT/'_so2_work/t014/budget.json')
    job=next(j for j in budget['jobs'] if j['job']==f'a1-ext-s{seed}-v1')
    assert job['status']=='running' and job['pid']==os.getpid()
    remaining=min(protocol['budget']['stage_endpoint_seconds']-budget['used_seconds'],
                  protocol['budget']['T014_seconds_cap']-budget['used_seconds'],
                  job['limit_seconds']-(time.time()-job['start_unix']),seconds)
    assert seconds==protocol['budget']['job_inner_seconds'] and seconds+30<=job['limit_seconds']
    assert protocol['budget']['formal_estimate_with30percent']<=remaining-60
    assert shutil.disk_usage(ROOT).free>8*1024**3
    save(work/'formal-admission.json',dict(passed=True,remaining_seconds=remaining,estimate=protocol['budget']['formal_estimate_with30percent']))
    return hashes

def validate_identity():
    from t014_a1_common import torch,load,bind_original,fixed_batch,rng_state,restore_rng,equivalent
    checked=[];hashes=source_hashes()
    for seed in range(1,5):
        protocol,record,work,run=settings(seed);assert not run.exists()
        weights,manifest=verified_parent(record)
        actual,_=make_seed_policy('cpu',seed);reference,_=make_seed_policy('cpu',seed)
        load(actual,weights);load(reference,weights);bind_original(reference)
        actual._model.eval();reference._model.eval()
        obs=torch.linspace(-1,1,32*17).reshape(32,17)
        action=torch.linspace(-.8,.8,32*6).reshape(32,6)
        with torch.no_grad():
            equivalent(actual._model(obs,mode='compute_actor'),reference._model(obs,mode='compute_actor'))
            equivalent(actual._model({'obs':obs,'action':action},mode='compute_critic'),reference._model({'obs':obs,'action':action},mode='compute_critic'))
        assert actual._forward_learn_cnt==reference._forward_learn_cnt==0
        checked.append({'seed':seed,'parent_sha256':record['parent_sha256'],'strict_tensors_and_function_exact':True,'optimizer_updates':0,'environment_interactions':0})
        del actual,reference,weights
    assert not torch.cuda.is_initialized()
    save(STAGE/'identity-validation.json',dict(status='passed',source_sha256=hashes,protocol_sha256=digest(PROTO),seeds=checked,GPU_work=False))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--validate-identities',action='store_true');a=p.parse_args()
    assert a.validate_identities
    validate_identity()
