"""Actual BaseLearner strict hook and train/eval continuity, zero rollout."""
from unittest.mock import patch
from t012_common import *
from t010_validate import check

def hook_check(saved,init,out,seed=0):
    from ding.config import compile_config
    from ding.worker import BaseLearner
    from ding.worker.learner import learner_hook
    out.mkdir(parents=True)
    policy=make_policy(seed=seed);baseline=make_policy(seed=seed)
    baseline.learn_mode.load_state_dict(saved)
    fresh=runpy.run_path(str(CONFIG));configure_load(fresh["main_config"],init)
    cfg=compile_config(fresh["main_config"],seed=seed,env=None,auto=True,create_cfg=fresh["create_config"],save_cfg=False)
    learner=BaseLearner(cfg.policy.learn.learner,policy.learn_mode,exp_name=str(out.relative_to(ROOT)))
    try:
        with patch.object(learner_hook,"read_file",wraps=learner_hook.read_file) as reader, patch.object(policy._learn_model,"load_state_dict",wraps=policy._learn_model.load_state_dict) as online, patch.object(policy._target_model,"load_state_dict",wraps=policy._target_model.load_state_dict) as target:
            learner.call_hook("before_run");reader.assert_called_once_with(str(init))
            for call in (online,target):assert call.call_count==1 and call.call_args.kwargs=={"strict":True}
        assert learner.train_iter==policy._forward_learn_cnt==0
        errors={};obs=torch.linspace(-1,1,256*17,device="cuda").reshape(256,17);actions=torch.linspace(-1,1,256*6,device="cuda").reshape(256,6)
        for name,actual in (("model",policy._model),("target_model",policy._target_model)):
            assert tensor_hash(actual.state_dict())==tensor_hash(saved[name])
        with torch.no_grad():
            for mode in ("eval","train"):
                for name,a,b in (("online",baseline._model,policy._model),("target",baseline._target_model,policy._target_model)):
                    a.train(mode=="train");b.train(mode=="train")
                    rng=rng_state();qa=q_values(a,obs,actions);restore_rng(rng);qb=q_values(b,obs,actions)
                    errors[f"{mode}_{name}_q"]=check(qa,qb,name,{"atol":0,"rtol":0})
                    for label,x,y in zip(("mu","std"),a.forward(obs,mode="compute_actor")["logit"],b.forward(obs,mode="compute_actor")["logit"]):errors[f"{mode}_{name}_{label}"]=check(x,y,label,{"atol":0,"rtol":0})
                    rng=rng_state();aa=sample_actor(a.forward,obs)[0];restore_rng(rng);ab=sample_actor(b.forward,obs)[0]
                    errors[f"{mode}_{name}_action"]=check(aa,ab,name,{"atol":0,"rtol":0})
        v={"status":"passed","group":GROUP,"seed":seed,"default_reader_calls":1,"strict_calls":2,"missing_keys":[],"unexpected_keys":[],"errors":errors,"fixed_batch_updates":0,"train_env_steps":0,"evaluation_env_steps":0,"export_sha256":digest(init),"network_hashes":{k:tensor_hash(saved[k]) for k in ("model","target_model")}}
        atomic_json(out/"validation.json",v);return v
    finally:learner.close()
