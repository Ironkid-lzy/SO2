"""Own-seed zero-update CPU identity checks and only new admission fault cases."""
import time,copy,importlib
from t019_common import *
from t018_edac import ParameterizedEDAC
from t019_control import schedule,permit_gate

def faults():
    p=read(PROTO);empty=dict(used_seconds=0,jobs=[])
    assert schedule('precheck',1,empty)==900
    done=dict(used_seconds=1,jobs=[dict(job='identity-preflight-v1',status='completed',exit_code=0,wall_seconds=1)])
    assert schedule('offline',1,done)==20935
    good=dict(allowed=True,ordinaryUsageAllowed=True,checked_unix=0,primary_used=0,weekly_used=0,job='s1-offline-v1',seed=1)
    permit_gate(good,0,'offline',1)
    cases=[lambda:schedule('offline',2,done),lambda:schedule('online',1,done),
      lambda:schedule('offline',1,dict(used_seconds=1,jobs=[dict(job='identity-preflight-v1',status='partial',exit_code=1,wall_seconds=1)])),
      lambda:schedule('offline',1,dict(used_seconds=150000,jobs=done['jobs'])),
      lambda:permit_gate(good,1801,'offline',1),
      lambda:permit_gate({**good,'primary_used':90},0,'offline',1),
      lambda:permit_gate({**good,'weekly_used':100},0,'offline',1),
      lambda:permit_gate({**good,'ordinaryUsageAllowed':False},0,'offline',1),
      lambda:permit_gate(good,0,'offline',2),
      lambda:schedule('precheck',1,done)]
    count=0
    for fn in cases:
        try:fn()
        except AssertionError:count+=1
        else:raise AssertionError('invalid new admission accepted')
    return count

if __name__=='__main__':
    start=time.monotonic();torch.set_num_threads(2);frozen()
    ledger=read(BASEWORK/'budget.json');j=next(j for j in ledger['jobs'] if j['job']=='identity-preflight-v1')
    assert j['status']=='running' and j['pgid']==os.getpgrp()
    assert not torch.cuda.is_available()
    assert digest(DATASET)==DATA_SHA
    records=[]
    for record in read(PROTO)['initializations']:
        seed=record['seed'];path=Path(record['archive_update0_path']);assert digest(path)==record['archive_update0_sha256']
        archive=torch.load(path,map_location='cpu',weights_only=False)
        assert archive['updates']==archive['train_env_steps']==0 and archive['identity']['seed']==seed
        assert archive['identity']['dataset_sha256']==DATA_SHA
        seed_all(seed,False);trainer=ParameterizedEDAC(make_model('cpu'),es_coefficient=0)
        for key,model in [('model',trainer.model),('target_model',trainer.target)]:
            assert tensor_hash(model.state_dict())==tensor_hash(archive[key])
            model.load_state_dict(archive[key],strict=True)
        assert torch.equal(trainer.log_alpha.detach().cpu(),archive['log_alpha'])
        assert not trainer.actor_optim.state and not trainer.critic_optim.state and not trainer.alpha_optim.state
        assert all(not archive[k]['state'] for k in ('optimizer_policy','optimizer_q','optimizer_alpha'))
        assert set(archive['rng'])=={'python','numpy','torch_cpu','torch_cuda'}
        events=[json.loads(x) for x in (path.parents[1]/'metrics.jsonl').read_text().splitlines() if x.strip()]
        initial=next(x['returns'] for x in events if x.get('event')=='evaluation' and x['gradient_updates']==0)
        assert initial==record['historical_initial_returns']
        os.environ['SO2_T019_SEED']=str(seed)
        common=importlib.reload(importlib.import_module('t019_common'))
        off=importlib.reload(importlib.import_module('t019_offline'))
        on=importlib.reload(importlib.import_module('t019_online'))
        assert common.SEED==off.SEED==on.SEED==seed
        assert common.ARCHIVE==path and common.ARCHIVE_SHA==record['archive_update0_sha256']
        assert off.OFFLINE==common.OFFLINE and on.RUN==common.ONLINE
        assert off.offline.__defaults__[0]==seed
        records.append(dict(seed=seed,archive_sha256=digest(path),fresh_model_exact=True,fresh_target_exact=True,strict_CPU_load_exact=True,empty_optimizers=True,log_alpha_exact=True,seed_identity_exact=True,data_hash_exact=True,RNG_restore='Own GPU archive0 RNG restored in formal entry; CPU precheck does not consume CUDA RNG',historical_initial_returns_record_exact=True,actual_module_seed_and_paths_exact=True))
    out=BASEWORK/'precheck';out.mkdir(exist_ok=False)
    atomic_json(out/'result.json',dict(passed=True,all_seeds_exact=True,source_sha256=source_hashes(),protocol_sha256=digest(PROTO),checks=records,new_fault_cases_passed=faults(),updates=0,environment_interactions=0,model_forwards=0,wall_seconds=time.monotonic()-start))
