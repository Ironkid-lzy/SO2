"""Zero-environment-step compatibility and actor aggregation checks for T009."""
import hashlib, json, math, runpy, sys, time
from pathlib import Path
import h5py
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"SO2"))
import d4rl  # noqa: F401
import ding
from ding.config import compile_config
from ding.policy import create_policy
assert Path(ding.__file__).resolve().is_relative_to(ROOT/"SO2")
torch.set_num_threads(2)
torch.manual_seed(117); np.random.seed(117); torch.cuda.manual_seed_all(117)
CHECKPOINT=Path("/home/lzy/Projects/SO2/_so2_work/assets/checkpoints/ckpt/halfcheetah-medium-replay-v2.ckpt")
DATASET=Path("/home/lzy/.d4rl/datasets/halfcheetah_medium_replay-v2.hdf5")
indices=np.arange(2560,dtype=np.int64)
with h5py.File(DATASET,"r") as f:
 arrays={k:np.asarray(f[name][indices],dtype=np.float32) for k,name in (("obs","observations"),("action","actions"),("reward","rewards"),("next_obs","next_observations"),("done","terminals"))}
batch_hash=hashlib.sha256(b"".join(arrays[k].tobytes() for k in ("obs","action","reward","next_obs","done"))).hexdigest()
data=[{k:torch.as_tensor(v[i].copy()) for k,v in arrays.items()} for i in range(len(indices))]
obs=torch.from_numpy(arrays["obs"]).cuda(); act=torch.from_numpy(arrays["action"]).cuda(); inp={"obs":obs,"action":act}
checkpoint=torch.load(CHECKPOINT,map_location="cpu",weights_only=False)
def make(aggregation=None,p=.005):
 f=runpy.run_path(str(ROOT/"scripts/repro/configs/halfcheetah_medium_replay_t009.py")); main,create=f["main_config"],f["create_config"]
 main.policy.model.critic_ensemble_size=2; main.policy.model.critic_layer_norm=True; main.policy.model.critic_dropout_rate=p; main.policy.learn.online=True
 if aggregation is None:
  del main.policy.learn.actor_q_aggregation
 else: main.policy.learn.actor_q_aggregation=aggregation
 cfg=compile_config(main,seed=0,env=None,auto=True,create_cfg=create,save_cfg=False); cfg.policy.type+="_command"; cfg.policy.import_names=["ding.policy.edac","ding.policy.command_mode_policy_instance"]
 obj=create_policy(cfg.policy,model=None,enable_field=["learn","collect","eval","command"]); obj._load_state_dict_learn(checkpoint); return obj
def q(obj,target=False):
 with torch.no_grad(): return (obj._target_model if target else obj._learn_model).forward(inp,mode="compute_critic")["q_value"].detach().cpu()
def actor_loss_q(qv, mode): return EDAC._aggregate_actor_q(qv,mode)
from ding.policy.edac import EDACPolicy as EDAC
# Old config (field missing) and explicit min have identical parameters and actor Q/loss.
old, explicit=make(None,.005),make("min",.005)
for obj in (old,explicit): obj._learn_model.eval(); obj._target_model.eval()
q_old,q_exp=q(old),q(explicit); assert torch.equal(q_old,q_exp)
assert torch.equal(actor_loss_q(q_old,"min"),torch.min(q_old,dim=0)[0])
assert torch.equal(q(old,True),q(explicit,True))
# Synthetic two-head example proves mean aggregation and gradients are headwise mean.
x=torch.tensor([[1.,3.],[5.,7.]],requires_grad=True)
mean=EDAC._aggregate_actor_q(x,"mean"); mean.sum().backward()
assert torch.equal(mean,torch.tensor([3.,5.])) and torch.equal(x.grad,torch.full_like(x,.5))
y=torch.tensor([[1.,5.],[2.,4.]],requires_grad=True); m=EDAC._aggregate_actor_q(y,"min"); m.sum().backward()
assert torch.equal(m,torch.tensor([1.,4.])) and torch.equal(y.grad,torch.tensor([[1.,0.],[0.,1.]]))
ten_heads=torch.arange(20,dtype=torch.float32).reshape(10,2)
assert torch.equal(EDAC._aggregate_actor_q(ten_heads,"min"),torch.min(ten_heads,dim=0)[0])
# Actual mean actor gradient is finite, nonzero, and Q output stays [heads,batch].
mean_obj=make("mean",.005); mean_obj._learn_model.eval(); mean_q=q(mean_obj)
assert mean_q.shape==(2,2560)
action=act.clone().requires_grad_(); qv=mean_obj._learn_model.forward({"obs":obs,"action":action},mode="compute_critic")["q_value"]
score=EDAC._aggregate_actor_q(qv,"mean").mean(); grad=torch.autograd.grad(score,action)[0]
assert torch.isfinite(grad).all() and grad.abs().max()>0
# Dropout p=0 deterministic; p>0 stochastic in training mode.
p0=make("min",0.0); p5=make("min",.005)
p0._learn_model.train(); p5._learn_model.train(); q0a,q0b=q(p0),q(p0); q5a,q5b=q(p5),q(p5)
assert torch.equal(q0a,q0b) and not torch.equal(q5a,q5b)
# Bad values fail at policy initialization.
bad=False
try:
 f=runpy.run_path(str(ROOT/"scripts/repro/configs/halfcheetah_medium_replay_t009.py")); f["main_config"].policy.learn.actor_q_aggregation="median"
 cfg=compile_config(f["main_config"],seed=0,env=None,auto=True,create_cfg=f["create_config"],save_cfg=False); cfg.policy.type+="_command"; cfg.policy.import_names=["ding.policy.edac","ding.policy.command_mode_policy_instance"]
 create_policy(cfg.policy,model=None,enable_field=["learn","collect","eval","command"])
except ValueError: bad=True
assert bad
# Strict load/save-reload with config identity, including legacy no-key and explicit min.
state=explicit._state_dict_learn(); reloaded=make("min",.005); reloaded._load_state_dict_learn(state); reloaded._learn_model.eval()
assert torch.equal(q(explicit),q(reloaded)); assert torch.equal(q(old),q(reloaded))
# One matched fixed-batch update compares the historical min expression with the new default path.
legacy_update=make(None,.005); default_update=make("min",.005)
legacy_update._cfg.learn.online=True; default_update._cfg.learn.online=True
legacy_update._forward_learn_cnt=9; default_update._forward_learn_cnt=9
old_target={k:v.detach().cpu().clone() for k,v in legacy_update._target_model.state_dict().items()}
legacy_update._aggregate_actor_q=lambda values, aggregation: torch.min(values,dim=0)[0]
cpu_rng=torch.get_rng_state(); cuda_rng=torch.cuda.get_rng_state_all()
torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state_all(cuda_rng)
legacy_info=legacy_update._forward_learn(data)
torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state_all(cuda_rng)
default_info=default_update._forward_learn(data)
def module_equal(a,b):
 sa,sb=a.state_dict(),b.state_dict()
 return sa.keys()==sb.keys() and all(torch.equal(sa[k].detach().cpu(),sb[k].detach().cpu()) for k in sa)
assert module_equal(legacy_update._learn_model,default_update._learn_model)
assert module_equal(legacy_update._target_model,default_update._target_model)
assert all(v.grad is None for v in legacy_update._target_model.parameters())
assert any(not torch.equal(old_target[k],v.detach().cpu()) for k,v in legacy_update._target_model.state_dict().items())
for key in ("critic_loss","policy_loss"):
 assert math.isclose(float(legacy_info[key]),float(default_info[key]),rel_tol=0,abs_tol=1e-8)
# Repeat the same equivalence check at p=0 to cover the C source group.
legacy_c=make(None,0.0); default_c=make("min",0.0)
legacy_c._cfg.learn.online=True; default_c._cfg.learn.online=True
legacy_c._forward_learn_cnt=9; default_c._forward_learn_cnt=9
legacy_c._aggregate_actor_q=lambda values, aggregation: torch.min(values,dim=0)[0]
rng_cpu=torch.get_rng_state(); rng_cuda=torch.cuda.get_rng_state_all()
torch.set_rng_state(rng_cpu); torch.cuda.set_rng_state_all(rng_cuda)
info_c_old=legacy_c._forward_learn(data)
torch.set_rng_state(rng_cpu); torch.cuda.set_rng_state_all(rng_cuda)
info_c_new=default_c._forward_learn(data)
assert module_equal(legacy_c._learn_model,default_c._learn_model)
assert module_equal(legacy_c._target_model,default_c._target_model)
for key in ("critic_loss","policy_loss"):
 assert math.isclose(float(info_c_old[key]),float(info_c_new[key]),rel_tol=0,abs_tol=1e-8)
result={"status":"passed","checkpoint_sha256":hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),"dataset_sha256":hashlib.sha256(DATASET.read_bytes()).hexdigest(),"batch_sha256":batch_hash,"batch_size":2560,"online_environment_steps":0,"validation_updates":4,"checks":{"legacy_config_min_exact":True,"explicit_min_exact":True,"td_target_unchanged":True,"synthetic_min_gradient_exact":True,"synthetic_mean_gradient_half_per_head":True,"real_mean_action_gradient_finite_nonzero":True,"ensemble_batch_shape_2_by_2560":True,"p0_deterministic":True,"p_positive_stochastic":True,"invalid_config_rejected":True,"strict_checkpoint_load_and_state_reload":True,"historical_min_single_update_exact":True,"historical_C_and_D_min_update_exact":True,"n10_default_min_exact":True,"target_no_gradient":True,"polyak_target_updated":True},"mean_action_grad_max_abs":float(grad.abs().max()),"completed_unix":time.time()}
out=ROOT/"_so2_work/validation/EXP-024/t009_fixed_batch.json"; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); print(json.dumps(result,indent=2))
