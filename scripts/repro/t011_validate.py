"""Bounded GPU checks: complete min update parity, mean analytic gradient, real hooks."""
import argparse
import ast
import copy
import fcntl
import math
import pickle
import runpy
import subprocess
import time
import types
from unittest.mock import patch
from t011_common import *
from t010_validate import capture,check

def equal_rng(a,b):
    assert pickle.dumps(a["python"])==pickle.dumps(b["python"])
    assert a["numpy"][0]==b["numpy"][0] and np.array_equal(a["numpy"][1],b["numpy"][1])
    assert a["numpy"][2:]==b["numpy"][2:]
    assert torch.equal(a["torch_cpu"],b["torch_cpu"])
    assert len(a["torch_cuda"])==len(b["torch_cuda"])
    assert all(torch.equal(x,y) for x,y in zip(a["torch_cuda"],b["torch_cuda"]))

def parity():
    from ding.policy.edac import EDACPolicy
    text=subprocess.check_output(["git","show",f"{BASE_SHA}:SO2/ding/policy/edac.py"],cwd=ROOT,text=True)
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name=="EDACPolicy")
    func=next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=="_forward_learn")
    ns=dict(vars(__import__("ding.policy.edac",fromlist=["EDACPolicy"])))
    exec(compile(ast.Module(body=[func],type_ignores=[]),"frozen_T010_edac.py","exec"),ns)
    reference_forward=ns["_forward_learn"]
    old=subprocess.check_output(["git","show",f"{SEED0_SHA}:SO2/ding/policy/edac.py"],cwd=ROOT,text=True)
    assert old==text, "seed0 math differs"
    seed_all(113,True)
    batch=[{"obs":torch.randn(17),"next_obs":torch.randn(17),"action":torch.rand(6)*2-1,
            "reward":torch.randn(1),"done":False} for _ in range(2560)]
    a=make_policy(seed=0,aggregation="min")
    b=make_policy(seed=0,aggregation="min")
    b.learn_mode.load_state_dict(copy.deepcopy(a.learn_mode.state_dict()))
    b._forward_learn=types.MethodType(reference_forward,b)
    assert a._actor_q_aggregation=="min"
    ga={k:capture(o) for k,o in (("q",a._optimizer_q),("actor",a._optimizer_policy))}
    gb={k:capture(o) for k,o in (("q",b._optimizer_q),("actor",b._optimizer_policy))}
    errors={}; steps=0
    # Exercise both critic-only and scheduled actor updates with PVU and masks.
    for count in (0,9):
        a._forward_learn_cnt=b._forward_learn_cnt=count
        rng=rng_state(); oa=a._forward_learn(copy.deepcopy(batch)); ra=rng_state();steps+=1
        restore_rng(rng); ob=b._forward_learn(copy.deepcopy(batch)); rb=rng_state();steps+=1
        equal_rng(ra,rb)
        for key in oa:
            assert type(oa[key])==type(ob[key])
            if isinstance(oa[key],list): assert np.array_equal(oa[key],ob[key])
            else: assert oa[key]==ob[key], (key,oa[key],ob[key])
        for label in ga:
            assert len(ga[label])==len(gb[label])
            for x,y in zip(ga[label],gb[label]):
                if x is None or y is None: assert x is None and y is None
                else: check(x,y,label,{"atol":0,"rtol":0})
        for name,ma,mb in (("model",a._model,b._model),("target",a._target_model,b._target_model)):
            for key,x in ma.state_dict().items(): errors[f"{count}_{name}_{key}"]=check(x,mb.state_dict()[key],key,{"atol":0,"rtol":0})
    # Missing config field follows the frozen min path; illegal values fail early.
    import ding.policy.edac as module
    defaults=copy.deepcopy(EDACPolicy.config)
    try:
        del EDACPolicy.config["learn"]["actor_q_aggregation"]
        missing=make_policy(aggregation=None)
        assert missing._actor_q_aggregation=="min"
    finally: EDACPolicy.config=defaults
    try: make_policy(aggregation="invalid")
    except ValueError: pass
    else: raise AssertionError("illegal aggregation accepted")
    # Exercise the actual mean actor path and compare its exact captured Q graph.
    mean=make_policy(seed=0,aggregation="mean");mean._forward_learn_cnt=9
    captured={};orig=mean._learn_model.forward
    def forward(*args,**kwargs):
        out=orig(*args,**kwargs)
        if kwargs.get("mode")=="compute_actor": captured["logit"]=out["logit"]
        if kwargs.get("mode")=="compute_critic" and args[0]["action"].requires_grad:
            q=out["q_value"];captured["q"]=q;captured["action"]=args[0]["action"]
        return out
    mean._learn_model.forward=forward
    actual=[]
    def actor_step(*args,**kwargs):
        actual[:]=[p.grad.detach().clone() for p in mean._model.actor.parameters()]
        # This runs during the original backward; retain_graph is used in test only.
        q=captured["q"]
        assert q.shape==(2,2560)
        return None
    # Suppress actor optimizer mutation only in the temporary analytic probe.
    original_backward=torch.Tensor.backward
    def backward(self,*args,**kwargs):
        return original_backward(self,*args,**{**kwargs,"retain_graph":True})
    mean._optimizer_policy.step=actor_step
    from torch.distributions import Independent,Normal
    old_sample=Independent.rsample
    def rsample(self,*args,**kwargs):
        value=old_sample(self,*args,**kwargs)
        if value.requires_grad: captured["pred"]=value
        return value
    with patch.object(torch.Tensor,"backward",backward), patch.object(Independent,"rsample",rsample):
        result=mean._forward_learn(copy.deepcopy(batch));steps+=1
    q=captured["q"];action=captured["action"]
    params=list(mean._model.actor.parameters())
    mu,sigma=captured["logit"];pred=captured["pred"]
    logp=Independent(Normal(mu,sigma),1).log_prob(pred).unsqueeze(-1)
    logp=logp-torch.log(1-action.square()+1e-6).sum(-1,keepdim=True)
    expected=(mean._alpha*logp-q.mean(0).unsqueeze(-1)).mean()
    assert result["policy_loss"]==expected.item()
    expected_grads=torch.autograd.grad(expected,params,retain_graph=True)
    totalerr=max(check(x,y,"mean total actor",{"atol":2e-6,"rtol":2e-5}) for x,y in zip(actual,expected_grads))
    qg=torch.autograd.grad(-q.mean(0).mean(),params,retain_graph=True)
    heads=[torch.autograd.grad(-q[i].mean(),params,retain_graph=True) for i in range(2)]
    maxerr=max(check(g,(x+y)/2,"head mean",{"atol":2e-6,"rtol":2e-5})
               for g,x,y in zip(qg,*heads))
    assert sum(float(g.abs().sum()) for g in qg)>0
    return {"fixed_batch_updates":steps,"batch_size":2560,"min_loss_grad_weight_rng_exact":True,
            "min_max_abs_error":max(errors.values()),"seed0_source_exact":True,
            "mean_q_shape":list(q.shape),"mean_actor_q_gradient_max_abs_error":maxerr,
            "mean_total_actor_gradient_max_abs_error":totalerr,"mean_actor_gradient_nonzero":True,"mean_losses_finite":all(bool(torch.isfinite(v).all()) if isinstance(v,torch.Tensor) else math.isfinite(v) for v in result.values() if isinstance(v,(int,float,torch.Tensor)))}

def main():
    import t011_loading as loading
    torch.set_num_threads(2)
    start=time.monotonic(); sha=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    assert not subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip()
    with LOCK.open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True).strip()
        ledger=VALIDATION/"preflight-ledger.json"
        history=json.loads(ledger.read_text()) if ledger.exists() else {"attempts":[]}
        assert sum(a["reserved_fixed_batch_calls"] for a in history["attempts"])+5<=100
        history["attempts"].append({"code_sha":sha,"start_unix":time.time(),"reserved_fixed_batch_calls":5})
        atomic_json(ledger,history)
        result=parity()
        assert result["fixed_batch_updates"]<=100
        hooks=[]
        for seed in range(5):
            m=parent_manifest(seed);out=VALIDATION/f"hook-s{seed}";assert not out.exists();out.mkdir()
            loading.validate(out,Path(m["parent_path"]),m,seed)
            v=json.loads((out/"validation.json").read_text())
            assert v["status"]=="passed" and v["fixed_batch_updates"]==0
            assert v["initialization"]["export_sha256"]==m["export_sha256"]
            hooks.append({"seed":seed,"parent_sha256":m["parent_sha256"],"export_sha256":m["export_sha256"],"actual_hook_passed":True})
        for source in ("scripts/repro/t010_edac.py","scripts/repro/configs","SO2/ding/entry","SO2/ding/model"):
            assert subprocess.check_output(["git","rev-parse",f"{BASE_SHA}:{source}"],cwd=ROOT)==subprocess.check_output(["git","rev-parse",f"HEAD:{source}"],cwd=ROOT)
        oldlock=Path("/home/lzy/Projects/rl_sample_efficiency_research_t010/reports/so2/SO2-T010/requirements-lock.txt")
        assert oldlock.read_bytes()==(VALIDATION/"requirements-lock.txt").read_bytes()
        result.update(preflight_reserved_calls_cumulative=sum(a["reserved_fixed_batch_calls"] for a in history["attempts"]),status="passed",code_sha=sha,git_dirty=False,parent_hooks=hooks,
                      train_env_steps=0,evaluation_env_steps=0,wall_seconds=time.monotonic()-start,
                      tolerance_min={"atol":0,"rtol":0},tolerance_mean={"atol":2e-6,"rtol":2e-5},
                      dependencies_exact=True,frozen_pipeline_model_config_objects_exact=True)
        assert result["wall_seconds"]<=1800
        atomic_json(VALIDATION/"validation.json",result)
        print(json.dumps(result))

if __name__=="__main__":main()
