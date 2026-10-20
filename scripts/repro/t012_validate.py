"""Bounded fixed batches, exact frozen combination parity and both component hooks."""
import copy,fcntl,pickle,time
from t012_common import *
from t010_validate import capture,check
from t011_validate import equal_rng
from t012_loading import hook_check
import t010_edac as frozen

def batch(n):
    return {"observations":torch.randn(n,17,device="cuda"),"next_observations":torch.randn(n,17,device="cuda"),"actions":torch.rand(n,6,device="cuda")*2-1,"rewards":torch.randn(n,1,device="cuda"),"terminals":torch.zeros(n,1,device="cuda")}
def online_batch(b):
    return [{"obs":b["observations"][i],"next_obs":b["next_observations"][i],"action":b["actions"][i],"reward":b["rewards"][i],"done":False} for i in range(len(b["rewards"]))]
def compare_models(a,b):
    for k,v in a.state_dict().items():check(v,b.state_dict()[k],k,{"atol":0,"rtol":0})
def compare_grads(a,b):
    for k in a:
      for x,y in zip(a[k],b[k]):
        if x is None or y is None:assert x is None and y is None
        else:check(x,y,k,{"atol":0,"rtol":0})

def parity():
    assert GROUP=="combined"
    seed_all(113,True);a=OfflineEDAC(make_model("cuda"))
    seed_all(113,True);b=OfflineEDAC(frozen.make_model("cuda"));compare_models(a.model,b.model)
    data=batch(256);ga={k:capture(o) for k,o in (("alpha",a.alpha_optim),("actor",a.actor_optim),("critic",a.critic_optim))};gb={k:capture(o) for k,o in (("alpha",b.alpha_optim),("actor",b.actor_optim),("critic",b.critic_optim))}
    rng=rng_state();la=a.losses(copy.deepcopy(data));restore_rng(rng);lb=b.losses(copy.deepcopy(data))
    for k in la:check(la[k],lb[k],k,{"atol":0,"rtol":0})
    rng=rng_state();ra=a.update(copy.deepcopy(data));after=rng_state();restore_rng(rng);rb=b.update(copy.deepcopy(data));equal_rng(after,rng_state());assert ra==rb
    compare_grads(ga,gb);compare_models(a.model,b.model);compare_models(a.target,b.target)
    from t011_common import make_policy as frozen_policy
    a=make_policy();b=frozen_policy(aggregation="mean");b.learn_mode.load_state_dict(copy.deepcopy(a.learn_mode.state_dict()))
    a._cfg.learn.online=b._cfg.learn.online=True;a._forward_learn_cnt=b._forward_learn_cnt=9
    ga={k:capture(o) for k,o in (("actor",a._optimizer_policy),("critic",a._optimizer_q))};gb={k:capture(o) for k,o in (("actor",b._optimizer_policy),("critic",b._optimizer_q))}
    data=online_batch(batch(2560));rng=rng_state();ra=a._forward_learn(copy.deepcopy(data));after=rng_state();restore_rng(rng);rb=b._forward_learn(copy.deepcopy(data));equal_rng(after,rng_state())
    for k in ra:
        if isinstance(ra[k],torch.Tensor):check(ra[k],rb[k],k,{"atol":0,"rtol":0})
        else:assert np.array_equal(ra[k],rb[k]),k
    compare_grads(ga,gb);compare_models(a._model,b._model);compare_models(a._target_model,b._target_model)
    return {"status":"passed","fixed_batch_calls":4,"optimizer_steps":10,"offline_min_loss_gradient_weight_target_rng_exact":True,"online_mean_loss_gradient_weight_target_rng_exact":True,"max_abs_error":0}

def component(out):
    seed_all(7,True);model=make_model("cuda")
    seed_all(7,True);combination=frozen.make_model("cuda")
    linears=[k for k in model.state_dict() if "layer_norm" not in k]
    for k in linears:check(model.state_dict()[k],combination.state_dict()[k],k,{"atol":0,"rtol":0})
    assert model.critic.use_layer_norm==LN and model.critic.dropout_rate==P
    ln=[k for k in model.critic.state_dict() if "layer_norm" in k];assert len(ln)==(6 if LN else 0)
    assert len(model.critic.fcs)==3 and model.critic.num_q==2
    assert not any(isinstance(m,(torch.nn.LayerNorm,torch.nn.Dropout)) for m in model.actor.modules())
    b=batch(256);model.train();rng=rng_state();actual=q_values(model,b["observations"],b["actions"])
    restore_rng(rng);h=torch.cat((b["observations"],b["actions"]),-1).unsqueeze(0).repeat(2,1,1)
    for i,fc in enumerate(model.critic.fcs):
        h=fc(h,2);h=torch.nn.functional.dropout(h,p=P,training=True)
        if LN:h=torch.nn.functional.layer_norm(h,(256,))*model.critic.layer_norm_weights[i]+model.critic.layer_norm_biases[i]
        h=torch.relu(h)
    expected=model.critic.last_fc(h,2);check(actual,expected,"layer order",{"atol":0,"rtol":0})
    model.eval();q1=q_values(model,b["observations"],b["actions"]);q2=q_values(model,b["observations"],b["actions"]);assert torch.equal(q1,q2)
    model.train();rng=rng_state();q1=q_values(model,b["observations"],b["actions"]);after=rng_state();q2=q_values(model,b["observations"],b["actions"])
    if P:assert not torch.equal(q1,q2)
    else:assert torch.equal(q1,q2);equal_rng(rng,after)
    trainer=OfflineEDAC(model);losses=trainer.losses(copy.deepcopy(b));norms={}
    for key,params in (("es",list(model.critic.parameters())),("td",list(model.critic.parameters())),("actor",list(model.actor.parameters()))):
        grads=torch.autograd.grad(losses[key],params,retain_graph=True,allow_unused=True)
        assert all(torch.isfinite(g).all() for g in grads if g is not None)
        norms[key]=sum(float(g.norm()) for g in grads if g is not None);assert norms[key]>0
    torch.cuda.reset_peak_memory_stats();times=[]
    for _ in range(8):
        disposable=copy.deepcopy(b)
        torch.cuda.synchronize();start=time.monotonic();result=trainer.update(disposable);torch.cuda.synchronize();times.append(time.monotonic()-start)
        assert all(np.isfinite(v) for v in result.values())
    off_sec=sum(times[2:])/6;off_vram=torch.cuda.max_memory_allocated()
    saved=trainer.snapshot({"purpose":"T012 preflight only","group":GROUP});init=out/"weights.pth"
    export_weights(saved,init,init,{"git_commit":"preflight","recipe_sha256":digest(RECIPE),"config_sha256":digest(CONFIG)}) if False else atomic_torch(init,{k:saved[k] for k in ("model","target_model")})
    hook=hook_check(saved,init,out/"hook")
    policy=make_policy();policy.learn_mode.load_state_dict(saved);policy._cfg.learn.online=True;policy._forward_learn_cnt=9
    assert policy._actor_q_aggregation=="mean" and not policy._auto_alpha and policy._actor_update_freq==10
    data=online_batch(batch(2560));torch.cuda.reset_peak_memory_stats();times=[]
    for _ in range(10):
        disposable=copy.deepcopy(data)
        torch.cuda.synchronize();start=time.monotonic();result=policy._forward_learn(disposable);torch.cuda.synchronize();times.append(time.monotonic()-start)
        assert all(bool(torch.isfinite(v).all()) if isinstance(v,torch.Tensor) else np.isfinite(v) for v in result.values() if isinstance(v,(int,float,torch.Tensor)))
    on_sec=sum(times[2:])/8;on_vram=torch.cuda.max_memory_allocated()
    # Use measured update throughput with 40% overhead plus 0.5h evaluation margin.
    off_hours=off_sec*3000000/3600*1.4+.5;on_hours=on_sec*950000/3600*1.4+.5
    atomic_json(out/"timing.json",{"offline_projected_hours_with_margin":off_hours,"online_projected_hours_with_margin":on_hours,"offline_seconds_per_update":off_sec,"online_seconds_per_update":on_sec})
    assert off_hours<9.6 and on_hours<4.8,(off_hours,on_hours)
    return {"status":"passed","group":GROUP,"fixed_batch_calls":18,"optimizer_steps":35,"ln_parameter_tensors":len(ln),"dropout_p":P,"shared_initial_tensor_count":len(linears),"shared_initial_tensors_exact":True,"layer_order_exact":True,"gradient_norms":norms,"hook":hook,"offline_seconds_per_update":off_sec,"online_seconds_per_update":on_sec,"offline_projected_hours_with_margin":off_hours,"online_projected_hours_with_margin":on_hours,"offline_peak_allocated_bytes":off_vram,"online_peak_allocated_bytes":on_vram}

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--probe",action="store_true");a=p.parse_args();torch.set_num_threads(2)
    if a.probe:
        out=VALIDATION/(GROUP+os.environ.get("T012_ATTEMPT_SUFFIX",""));out.mkdir();start=time.monotonic()
        result=parity() if GROUP=="combined" else component(out)
        result.update(wall_seconds=time.monotonic()-start,train_env_steps=0,evaluation_env_steps=0)
        atomic_json(out/"result.json",result);return
    start=time.monotonic();sha=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    assert not subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip()
    with LOCK.open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True).strip()
        ledger=VALIDATION/"preflight-ledger.json";history=json.loads(ledger.read_text()) if ledger.exists() else {"attempts":[]}
        assert sum(x["reserved_fixed_batch_calls"] for x in history["attempts"])+40<=200
        history["attempts"].append({"code_sha":sha,"start_unix":time.time(),"reserved_fixed_batch_calls":40,"reserved_optimizer_steps":80});atomic_json(ledger,history)
        suffix="-attempt{}".format(len(history["attempts"]))
        results={}
        for group in ("combined","dropout","ln"):
            with (VALIDATION/f"{group}{suffix}.log").open("x") as log:
                subprocess.run([PYTHON,__file__,"--probe"],cwd=ROOT,env={**os.environ,"T012_GROUP":group,"T012_ATTEMPT_SUFFIX":suffix},stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
            results[group]=json.loads((VALIDATION/(group+suffix)/"result.json").read_text())
        for source in ("SO2","scripts/repro/t010_edac.py","scripts/repro/configs/halfcheetah_medium_replay_t010.py","scripts/repro/configs/halfcheetah_medium_replay_t005.py"):
            assert subprocess.check_output(["git","rev-parse",f"{BASE_SHA}:{source}"],cwd=ROOT)==subprocess.check_output(["git","rev-parse",f"HEAD:{source}"],cwd=ROOT)
        old=Path("/home/lzy/Projects/rl_sample_efficiency_research_t011/reports/so2/SO2-T011/requirements-lock.txt")
        assert old.read_bytes()==(VALIDATION/"requirements-lock.txt").read_bytes()
        history["attempts"][-1].update(actual_fixed_batch_calls=40,actual_optimizer_steps=80,wall_seconds=time.monotonic()-start,status="passed");atomic_json(ledger,history)
        cumulative=sum(x["reserved_fixed_batch_calls"] for x in history["attempts"])
        result={"status":"passed","code_sha":sha,"components":results,"reserved_fixed_batch_calls":cumulative,"actual_fixed_batch_calls":sum(x["actual_fixed_batch_calls"] for x in history["attempts"]),"optimizer_steps":sum(x["actual_optimizer_steps"] for x in history["attempts"]),"train_env_steps":0,"evaluation_env_steps":0,"wall_seconds":sum(x["wall_seconds"] for x in history["attempts"]),"frozen_core_objects_exact":True,"dependencies_exact":True}
        assert result["wall_seconds"]<=3600
        atomic_json(VALIDATION/"validation.json",result)
if __name__=="__main__":main()
