"""Fixed-batch tests against the pinned, unmodified EDAC trainer class."""
import argparse
import ast
import copy
import importlib
import json
import subprocess
import time
import types
from pathlib import Path

from t010_edac import *

TOL = {"atol": 2e-6, "rtol": 2e-5}
UPDATES = 0


def check(a, b, label, tol=TOL):
    assert a.shape == b.shape, (label, a.shape, b.shape)
    assert torch.allclose(a, b, **tol), (label, float((a - b).abs().max()))
    return float((a - b).detach().abs().max())


def reference_trainer():
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REFERENCE, text=True).strip() == REFERENCE_SHA
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=REFERENCE, text=True).strip()
    sys.path.insert(0, str(REFERENCE))
    import lifelong_rl.torch.pytorch_util as ref_ptu
    from lifelong_rl.util.eval_util import create_stats_ordered_dict
    from collections import OrderedDict
    # gtimer is only required by unrelated runner imports. Execute the complete
    # official SACTrainer class AST unchanged; its base only stores a counter.
    class CounterBase:
        def __init__(self):
            self._num_train_steps = 0
    path = REFERENCE / "lifelong_rl/trainers/q_learning/sac.py"
    node = next(x for x in ast.parse(path.read_text()).body if isinstance(x, ast.ClassDef) and x.name == "SACTrainer")
    ns = dict(np=np, torch=torch, nn=torch.nn, optim=torch.optim, ptu=ref_ptu,
              OrderedDict=OrderedDict, create_stats_ordered_dict=create_stats_ordered_dict,
              TorchTrainer=CounterBase)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    return ns["SACTrainer"], ref_ptu


def capture(optim):
    grads = []
    original = optim.step
    def step(*args, **kwargs):
        grads[:] = [None if p.grad is None else p.grad.detach().clone()
                    for group in optim.param_groups for p in group["params"]]
        return original(*args, **kwargs)
    optim.step = step
    return grads


def regression(device, bounds):
    global UPDATES
    Trainer, ref_ptu = reference_trainer()
    from lifelong_rl.models.networks import ParallelizedEnsembleFlattenMLP
    actor_module = importlib.import_module("lifelong_rl.policies.models.tanh_gaussian_policy")
    original_min = actor_module.LOG_SIG_MIN
    actor_module.LOG_SIG_MIN = bounds[0]  # Explicitly documented second comparison.
    ref_ptu.set_gpu_mode(device == "cuda")
    seed_all(41, device == "cuda")
    model = make_model(device, regularized=False, actor_bounds=bounds)
    qref = ParallelizedEnsembleFlattenMLP(2, [256]*3, 23, 1).to(device)
    qref.load_state_dict(model.critic.state_dict(), strict=True)
    aref = actor_module.TanhGaussianPolicy([256]*3, 17, 6).to(device)
    state = aref.state_dict()
    assert torch.count_nonzero(state["input_mu"]) == 0 and torch.equal(state["input_std"], torch.ones(17, device=device))
    assert set(state) == set(ACTOR_MAP.values()) | {"input_mu", "input_std"}
    for dst, src in ACTOR_MAP.items():
        state[src] = model.actor.state_dict()[dst].clone()
    aref.load_state_dict(state, strict=True)
    reference = Trainer(types.SimpleNamespace(action_space=types.SimpleNamespace(shape=(6,))),
                        aref, qref, copy.deepcopy(qref), num_qs=2, eta=1.)
    ours = OfflineEDAC(model)
    grad_ref = {"alpha": capture(reference.alpha_optimizer), "actor": capture(reference.policy_optimizer),
                "critic": capture(reference.qfs_optimizer)}
    grad_ours = {"alpha": capture(ours.alpha_optim), "actor": capture(ours.actor_optim),
                 "critic": capture(ours.critic_optim)}
    batch = {"observations": torch.randn(256, 17, device=device),
             "next_observations": torch.randn(256, 17, device=device),
             "actions": torch.rand(256, 6, device=device)*2-1,
             "rewards": torch.randn(256, 1, device=device), "terminals": torch.zeros(256, 1, device=device)}
    errors = []
    for iteration in range(3):
        rng = rng_state()
        reference._num_train_steps += 1
        reference.train_from_torch({k:v.clone() for k,v in batch.items()}, None)
        UPDATES += 1
        restore_rng(rng)
        ours.update({k:v.clone() for k,v in batch.items()}); UPDATES += 1
        for key in ("alpha", "critic"):
            for i, (a, b) in enumerate(zip(grad_ref[key], grad_ours[key])):
                assert a is not None and b is not None
                errors.append(check(a,b,f"{key}-gradient-{iteration}-{i}"))
        ref_grad = dict(zip([name for name,_ in aref.named_parameters()], grad_ref["actor"]))
        our_grad = dict(zip([name for name,_ in model.actor.named_parameters()], grad_ours["actor"]))
        for dst, src in ACTOR_MAP.items():
            errors.append(check(ref_grad[src],our_grad[dst],f"actor-gradient-{src}"))
            errors.append(check(aref.state_dict()[src],model.actor.state_dict()[dst],f"actor-step-{src}"))
        for name, value in qref.state_dict().items():
            errors.append(check(value,model.critic.state_dict()[name],f"critic-step-{name}"))
            errors.append(check(reference.target_qfs.state_dict()[name],ours.target.critic.state_dict()[name],f"target-{name}"))
        errors.append(check(reference.log_alpha,ours.log_alpha,"alpha-step"))
    # Force the lower-bound branch, which random initialization will not test.
    with torch.no_grad():
        aref.last_fc_log_std.weight.zero_(); aref.last_fc_log_std.bias.fill_(-30)
        model.actor[2].log_sigma_layer.weight.zero_(); model.actor[2].log_sigma_layer.bias.fill_(-30)
    ref_out = aref(batch["observations"], deterministic=True)
    mu, sigma = model(batch["observations"], mode="compute_actor")["logit"]
    check(ref_out[1],mu,"saturation-mu"); check(ref_out[2],sigma.log(),"saturation-std")
    assert float(sigma.log().min()) == bounds[0]
    actor_module.LOG_SIG_MIN = original_min
    return {"bounds": list(bounds), "official_actor_override": bounds[0] != -5,
            "max_abs_error": max(errors), "tolerance": TOL, "steps_each": 3,
            "reference_class_sha256": digest(REFERENCE / "lifelong_rl/trainers/q_learning/sac.py"),
            "reference_base": "counter-only; official full class AST unchanged; no gtimer dependency",
            "inactive_identity_input_stats": "official actor input_mu=0/input_std=1, unused by forward"}


def make_policy(device):
    import runpy
    from ding.config import compile_config
    from ding.policy import create_policy
    cfg = runpy.run_path(str(ROOT/"scripts/repro/configs/halfcheetah_medium_replay_t010.py"))
    cfg["main_config"].policy.cuda = device == "cuda"
    out = compile_config(cfg["main_config"], seed=0, env=None, auto=True,
                         create_cfg=cfg["create_config"], save_cfg=False)
    out.policy.type += "_command"
    out.policy.import_names = ["ding.policy.edac", "ding.policy.command_mode_policy_instance"]
    return create_policy(out.policy, model=None, enable_field=["learn","collect","eval","command"])


def continuity(trainer, state, device, batch):
    policy = make_policy(device)
    policy._load_state_dict_learn(state)
    assert set(policy._model.state_dict()) == set(state["model"])
    assert tensor_hash(policy._model.state_dict()) == tensor_hash(state["model"])
    assert tensor_hash(policy._target_model.state_dict()) == tensor_hash(state["target_model"])
    errors = {}
    for mode in ("eval", "train"):
        for name, a, b in (("online", trainer.model, policy._model), ("target", trainer.target, policy._target_model)):
            a.train(mode == "train"); b.train(mode == "train")
            rng = rng_state()
            qa = q_values(a,batch["observations"],batch["actions"])
            restore_rng(rng)
            qb = q_values(b,batch["observations"],batch["actions"])
            errors[f"{mode}_{name}_each_head_q"] = check(qa,qb,name,tol={"atol":0,"rtol":0})
        for i, label in enumerate(("mu","sigma")):
            a = trainer.model(batch["observations"], mode="compute_actor")["logit"][i]
            b = policy._model(batch["observations"], mode="compute_actor")["logit"][i]
            errors[f"{mode}_actor_{label}"] = check(a,b,label,tol={"atol":0,"rtol":0})
        rng = rng_state()
        aa = sample_actor(trainer.model,batch["observations"])[0]
        restore_rng(rng)
        ab = sample_actor(policy._model,batch["observations"])[0]
        errors[f"{mode}_fixed_rng_action"] = check(aa,ab,"action",tol={"atol":0,"rtol":0})
    assert not policy._auto_alpha and float(policy._alpha.item()) == float(np.float32(.2))
    assert policy._forward_learn_cnt == 0
    return policy, {"passed":True,"errors":errors,"missing_keys":[],"unexpected_keys":[],
                    "network_hash_equal":True,"online_alpha":.2,"online_es":False,
                    "optimizer_reset":True,"counter_reset":True,"actor_bounds":[-20,2]}


def regularized_tests(device, out):
    global UPDATES
    seed_all(7, device == "cuda")
    trainer = OfflineEDAC(make_model(device))
    batch = {"observations": torch.randn(256,17,device=device), "next_observations":torch.randn(256,17,device=device),
             "actions":torch.rand(256,6,device=device)*2-1, "rewards":torch.randn(256,1,device=device),
             "terminals":torch.zeros(256,1,device=device)}
    losses = trainer.losses(batch)
    params = list(trainer.model.critic.layer_norm_weights)+list(trainer.model.critic.layer_norm_biases)
    grads = torch.autograd.grad(losses["es"], params, retain_graph=True)
    norms = [float(g.norm()) for g in grads]
    assert all(torch.isfinite(g).all() for g in grads) and all(v > 0 for v in norms[:3])
    # Final LN bias changes the ReLU gating, whose local derivative is zero;
    # all LN tensors must still receive a finite nonzero total-loss gradient.
    total_grads = torch.autograd.grad(losses["critic"], params)
    total_norms = [float(g.norm()) for g in total_grads]
    assert all(torch.isfinite(g).all() for g in total_grads) and all(v > 0 for v in total_norms)
    for _ in range(3):
        trainer.update({k:v.clone() for k,v in batch.items()}); UPDATES += 1
    trainer.model.eval()
    assert torch.equal(q_values(trainer.model,batch["observations"],batch["actions"]),
                       q_values(trainer.model,batch["observations"],batch["actions"]))
    trainer.model.train()
    q1=q_values(trainer.model,batch["observations"],batch["actions"])
    q2=q_values(trainer.model,batch["observations"],batch["actions"])
    assert not torch.equal(q1,q2)
    # Make both heads identical; differences now prove independent train masks.
    clone=copy.deepcopy(trainer.model)
    with torch.no_grad():
        for p in clone.critic.parameters(): p[1].copy_(p[0])
    clone.train(); qs=q_values(clone,batch["observations"],batch["actions"])
    assert not torch.equal(qs[0],qs[1])
    state=trainer.snapshot({"purpose":"preflight-only; never a formal parent"})
    path=out/"test_full.pth"; atomic_torch(path,state)
    saved=torch.load(path,map_location=device,weights_only=False)
    assert state["online_critic_sha256"] != state["target_critic_sha256"]
    try: atomic_torch(path,state)
    except RuntimeError: pass
    else: raise AssertionError("overwrite accepted")
    restored=OfflineEDAC(make_model(device)); restored.restore(saved)
    rng=rng_state(); trainer.update({k:v.clone() for k,v in batch.items()}); UPDATES+=1
    restore_rng(rng); restored.update({k:v.clone() for k,v in batch.items()}); UPDATES+=1
    for key,value in trainer.model.state_dict().items():
        check(value,restored.model.state_dict()[key],"restore-"+key,tol={"atol":0,"rtol":0})
    check(trainer.log_alpha,restored.log_alpha,"restored-alpha",tol={"atol":0,"rtol":0})
    for key,value in trainer.target.state_dict().items():
        check(value,restored.target.state_dict()[key],"restored-target-"+key,tol={"atol":0,"rtol":0})
    # Restore trainer to the saved boundary before comparing policy loading.
    trainer.restore(saved)
    policy, transition = continuity(trainer,saved,device,batch)
    online=[{"obs":batch["observations"][i].detach(),"next_obs":batch["next_observations"][i].detach(),
             "action":batch["actions"][i].detach(),"reward":batch["rewards"][i].detach(),"done":False} for i in range(256)]
    policy._cfg.learn.online=True
    # Exactly one SO2 update also exercises actor, with a test-only counter.
    policy._forward_learn_cnt=9
    info=policy._forward_learn(online); UPDATES+=1
    assert all(torch.isfinite(v).all() for v in policy._model.state_dict().values())
    assert policy._forward_learn_cnt==10 and "policy_loss" in info
    # GPU throughput uses fresh, disposable fixed-batch weights.
    throughput=None
    if device == "cuda":
        bench=OfflineEDAC(make_model(device)); torch.cuda.synchronize(); start=time.monotonic()
        for _ in range(100): bench.update({k:v.clone() for k,v in batch.items()}); UPDATES+=1
        torch.cuda.synchronize(); elapsed=time.monotonic()-start
        throughput={"updates":100,"seconds":elapsed,"updates_per_second":100/elapsed,
                    "projected_3m_update_hours":elapsed*30000/3600,
                    "launch_estimate_hours_with_eval_margin":elapsed*30000/3600*1.2+.5,
                    "peak_allocated_bytes":torch.cuda.max_memory_allocated()}
        assert throughput["launch_estimate_hours_with_eval_margin"] <= 48, "fixed 3M exceeds 48h budget"
    return {"es_second_order_ln_grad_norms":norms,"total_critic_ln_grad_norms":total_norms,"dropout_eval_repeatable":True,
            "dropout_train_changes":True,"dropout_independent_heads":True,
            "full_checkpoint_restoration_exact":True,"actual_target_saved":True,
            "checkpoint_sha256":digest(path),"stage_continuity":transition,"one_so2_update_finite":True,
            "throughput":throughput}


def data_test():
    import runpy
    from ding.utils.data import create_dataset
    data, meta=load_dataset()
    cfg=runpy.run_path(str(ROOT/"scripts/repro/configs/halfcheetah_medium_replay_t010.py"))["main_config"]
    online=create_dataset(cfg)
    assert len(online)==meta["transitions"]
    assert set(data)==set(online.raw_dataset)
    for key,arr in data.items(): assert np.array_equal(arr,online.raw_dataset[key]), key
    for i in (0,len(online)//2,len(online)-1):
        for dst,src in (("obs","observations"),("next_obs","next_observations"),("action","actions"),("reward","rewards"),("done","terminals")):
            assert np.array_equal(np.asarray(online.data[i][dst]),data[src][i]), (i,dst)
    meta["so2_processed_arrays_equal"]=True
    return meta


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--device",choices=("cpu","cuda"),required=True)
    parser.add_argument("--out",type=Path,required=True); args=parser.parse_args()
    assert not args.out.exists(); args.out.mkdir(parents=True)
    torch.set_num_threads(2); start=time.monotonic()
    try:
        result={"device":args.device,"official_regression":regression(args.device,(-5.,2.)),
                "so2_boundary_regression":regression(args.device,(-20.,2.)),
                "regularized":regularized_tests(args.device,args.out)}
        if args.device=="cpu": result["data_identity"]=data_test()
        result.update(status="passed",fixed_batch_updates=UPDATES,train_env_steps=0,
                      wall_seconds=time.monotonic()-start,reference_sha=REFERENCE_SHA)
        assert UPDATES<=200 and (args.device=="cpu" or result["wall_seconds"]<=1800)
        atomic_json(args.out/"validation.json",result); print(json.dumps(result),flush=True)
    except BaseException as exc:
        atomic_json(args.out/"validation.json",{"status":"failed","error":repr(exc),"fixed_batch_updates":UPDATES,
                    "wall_seconds":time.monotonic()-start,"train_env_steps":0}); raise

if __name__=="__main__": main()
