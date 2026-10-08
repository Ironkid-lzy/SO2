"""Zero-environment-step T008 fixed-batch and initialization gate."""
import argparse
import hashlib
import json
import math
import runpy
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "SO2"))
import d4rl  # noqa: F401
import ding
from ding.config import compile_config
from ding.policy import create_policy
assert Path(ding.__file__).resolve().is_relative_to(ROOT / "SO2")
torch.set_num_threads(2)
torch.manual_seed(117)
np.random.seed(117)
torch.cuda.manual_seed_all(117)
CHECKPOINT = Path("/home/lzy/Projects/SO2/_so2_work/assets/checkpoints/ckpt/halfcheetah-medium-replay-v2.ckpt")
DATASET = Path("/home/lzy/.d4rl/datasets/halfcheetah_medium_replay-v2.hdf5")
indices = np.arange(2560, dtype=np.int64)
with h5py.File(DATASET, "r") as f:
    arrays = {k: np.asarray(f[name][indices], dtype=np.float32) for k, name in (
        ("obs", "observations"), ("action", "actions"), ("reward", "rewards"),
        ("next_obs", "next_observations"), ("done", "terminals"))}
batch_hash = hashlib.sha256(b"".join(arrays[k].tobytes() for k in ("obs", "action", "reward", "next_obs", "done"))).hexdigest()
data = [{k: torch.as_tensor(v[i].copy()) for k, v in arrays.items()} for i in range(len(indices))]
obs = torch.from_numpy(arrays["obs"]).cuda()
act = torch.from_numpy(arrays["action"]).cuda()
inp = {"obs": obs, "action": act}
checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)

def make(drop):
    frozen = runpy.run_path(str(ROOT / "scripts/repro/configs/halfcheetah_medium_replay_t008.py"))
    main, create = frozen["main_config"], frozen["create_config"]
    main.policy.model.critic_ensemble_size = 2
    main.policy.model.critic_layer_norm = True
    main.policy.model.critic_dropout_rate = drop
    main.policy.learn.online = True
    cfg = compile_config(main, seed=0, env=None, auto=True, create_cfg=create, save_cfg=False)
    cfg.policy.type += "_command"
    cfg.policy.import_names = ["ding.policy.edac", "ding.policy.command_mode_policy_instance"]
    obj = create_policy(cfg.policy, model=None, enable_field=["learn", "collect", "eval", "command"])
    obj._load_state_dict_learn(checkpoint)
    return obj

def q(obj, target=False):
    model = obj._target_model if target else obj._learn_model
    with torch.no_grad():
        return model.forward(inp, mode="compute_critic")["q_value"].detach().cpu()

def state_equal(a, b, prefix):
    sa, sb = a.state_dict(), b.state_dict()
    keys = [k for k in sa if k.startswith(prefix)]
    return keys, all(torch.equal(sa[k].cpu(), sb[k].cpu()) for k in keys)

c, d = make(0.0), make(0.005)
for obj in (c, d):
    obj._learn_model.eval(); obj._target_model.eval()
qc, qct, qd, qdt = q(c), q(c, True), q(d), q(d, True)
with torch.no_grad():
    ac = c._learn_model.forward(obs, mode="compute_actor")["logit"][0].detach().cpu()
    ad = d._learn_model.forward(obs, mode="compute_actor")["logit"][0].detach().cpu()
assert torch.equal(ac, ad), "actor initializations differ"
lin_online = state_equal(c._learn_model.critic, d._learn_model.critic, "networks") if False else None
assert torch.equal(qc, qd) and torch.equal(qct, qdt), "eval initial Q differs"
# Confirm p=0 has no train-mode dropout randomness, while D retains it.
c._learn_model.train(); c._target_model.train()
d._learn_model.train(); d._target_model.train()
qc_train_1, qc_train_2 = q(c), q(c)
qd_train_1, qd_train_2 = q(d), q(d)
assert torch.equal(qc_train_1, qc_train_2), "p=0 C is stochastic in train mode"
assert not torch.equal(qd_train_1, qd_train_2), "D dropout did not remain stochastic"
# Check same author linear tensors and LayerNorm identity initialization for online/target heads.
linear_exact = True
ln_identity = True
for label, model, stem in (("online", c._learn_model.critic, "trainer/qfs"),
                           ("target", c._target_model.critic, "trainer/target_qfs")):
    source = checkpoint[stem]
    for key, value in source.items():
        target_value = model.state_dict()[key]
        if key.endswith("weight") or key.endswith("bias"):
            # LN keys are newly introduced and do not exist in checkpoint; skip them here.
            pass
        if key in model.state_dict():
            expected = value[:2] if value.ndim and value.shape[0] >= 2 else value
            linear_exact &= torch.equal(target_value.cpu(), expected)
    for name, param in model.named_parameters():
        if "layer_norm" in name or ".ln" in name:
            if name.endswith("weight"):
                ln_identity &= torch.equal(param.detach().cpu(), torch.ones_like(param.detach().cpu()))
            elif name.endswith("bias"):
                ln_identity &= torch.equal(param.detach().cpu(), torch.zeros_like(param.detach().cpu()))
assert linear_exact, "author linear parameter mapping mismatch"
# Action gradient through actor objective must be finite/nonzero.
action_var = act.clone().requires_grad_()
score = c._learn_model.forward({"obs": obs, "action": action_var}, mode="compute_critic")["q_value"].min(dim=0)[0].mean()
action_grad = torch.autograd.grad(score, action_var)[0]
assert torch.isfinite(action_grad).all() and action_grad.abs().max() > 0
# One C update; verify critic/LN gradients, target isolation, and Polyak movement.
c._cfg.learn.online = True
c._forward_learn_cnt = 9
old_target = {n: p.detach().clone() for n,p in c._target_model.named_parameters()}
info = c._forward_learn(data)
critic_grads = [p.grad for p in c._model.critic.parameters() if p.grad is not None]
ln_grads = [p.grad for n,p in c._model.critic.named_parameters() if ("layer_norm" in n or ".ln" in n) and p.grad is not None]
assert critic_grads and all(torch.isfinite(g).all() for g in critic_grads)
assert ln_grads and any(g.abs().sum() > 0 for g in ln_grads), "LayerNorm gradient missing/zero"
assert all(p.grad is None for p in c._target_model.parameters()), "target received gradients"
polyak_moved = any(not torch.equal(old_target[n], p.detach()) for n,p in c._target_model.named_parameters())
assert polyak_moved, "target parameters did not Polyak update"
state = c._state_dict_learn()
fresh = make(0.0)
fresh._load_state_dict_learn(state)
fresh._learn_model.eval(); c._learn_model.eval()
assert torch.equal(q(fresh), q(c)), "strict reload did not reproduce Q"
with torch.no_grad():
    af = fresh._learn_model.forward(obs, mode="compute_actor")["logit"][0]
    ac2 = c._learn_model.forward(obs, mode="compute_actor")["logit"][0]
assert torch.equal(af, ac2), "strict reload did not reproduce actor"
result = {
    "status": "passed", "code_root": str(ROOT), "ding_file": str(Path(ding.__file__).resolve()),
    "config_sha256": hashlib.sha256((ROOT/"scripts/repro/configs/halfcheetah_medium_replay_t008.py").read_bytes()).hexdigest(),
    "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
    "dataset_sha256": hashlib.sha256(DATASET.read_bytes()).hexdigest(),
    "batch_indices": [0,2559], "batch_size": 2560, "batch_sha256": batch_hash,
    "online_environment_steps": 0, "validation_updates": 1,
    "checks": {"actor_exact_C_D": True, "eval_online_target_q_exact_C_D": True,
                "C_train_mode_deterministic": True, "D_train_mode_stochastic": True,
                "author_linear_mapping_exact": linear_exact, "ln_identity_initialization": ln_identity,
                "actor_action_gradient_finite_nonzero": True, "critic_gradients_finite": True,
                "ln_gradient_nonzero": True, "target_no_gradient": True,
                "target_polyak_moved": polyak_moved, "strict_reload_q_actor_exact": True},
    "C_initial_q_mean": float(qc.mean()), "D_initial_q_mean_eval": float(qd.mean()),
    "actor_grad_max_abs": float(action_grad.abs().max()),
    "critic_loss": float(info["critic_loss"].detach()), "policy_loss": float(info["policy_loss"].detach()),
    "ln_grad_abs_sum": float(sum(g.abs().sum() for g in ln_grads)),
    "completed_unix": time.time(),
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
print(json.dumps({"status": result["status"], "checks": result["checks"], "output": str(args.output)}))
