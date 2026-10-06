"""T006 fixed batch control, mapping, update, and reload checks."""
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

P = argparse.ArgumentParser()
P.add_argument("--code-root", type=Path, required=True)
P.add_argument("--output", type=Path, required=True)
P.add_argument("--baseline-only", action="store_true")
args = P.parse_args()
root = args.code_root.resolve()
sys.path.insert(0, str(root / "SO2"))
import d4rl  # noqa: F401
from ding.config import compile_config
from ding.policy import create_policy
import ding

assert Path(ding.__file__).resolve().is_relative_to(root / "SO2")
torch.set_num_threads(2)
torch.manual_seed(117)
np.random.seed(117)
checkpoint = torch.load(
    "/home/lzy/Projects/SO2/_so2_work/assets/checkpoints/ckpt/halfcheetah-medium-replay-v2.ckpt",
    map_location="cpu", weights_only=False
)
dataset = "/home/lzy/.d4rl/datasets/halfcheetah_medium_replay-v2.hdf5"
indices = np.arange(2560, dtype=np.int64)
with h5py.File(dataset) as f:
    arrays = {key: np.asarray(f[name][indices], dtype=np.float32)
              for key, name in (
                  ("obs", "observations"), ("action", "actions"), ("reward", "rewards"),
                  ("next_obs", "next_observations"), ("done", "terminals"))}
batch_hash = hashlib.sha256(b"".join(arrays[key].tobytes() for key in
                   ("obs", "action", "reward", "next_obs", "done"))).hexdigest()
data = [{key: torch.as_tensor(value[i].copy()) for key, value in arrays.items()}
        for i in range(len(indices))]
obs = torch.from_numpy(arrays["obs"]).cuda()
act = torch.from_numpy(arrays["action"]).cuda()
inp = {"obs": obs, "action": act}

def policy(n, ln, p):
    frozen = runpy.run_path(str(root / "scripts/repro/configs/halfcheetah_medium_replay_full.py"))
    main, create = frozen["main_config"], frozen["create_config"]
    main.policy.model.critic_ensemble_size = n
    if not args.baseline_only:
        main.policy.model.critic_layer_norm = ln
        main.policy.model.critic_dropout_rate = p
    main.policy.learn.online = True
    cfg = compile_config(main, seed=0, env=None, auto=True, create_cfg=create, save_cfg=False)
    cfg.policy.type += "_command"
    cfg.policy.import_names = ["ding.policy.edac", "ding.policy.command_mode_policy_instance"]
    obj = create_policy(cfg.policy, model=None, enable_field=["learn", "collect", "eval", "command"])
    obj._load_state_dict_learn(checkpoint)
    return obj

def q(obj, target=False):
    module = obj._target_model if target else obj._learn_model
    with torch.no_grad():
        return module.forward(inp, mode="compute_critic")["q_value"].detach().cpu()

def qstats(x):
    return {"mean": x.mean().item(), "std": x.std().item(),
            "min": x.min().item(), "max": x.max().item()}

def update(obj):
    obj._cfg.learn.online = True
    obj._forward_learn_cnt = 9
    torch.manual_seed(9181)
    torch.cuda.manual_seed_all(9181)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    info = obj._forward_learn(data)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return {
        "wall_sec": dt, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "critic_loss": float(info["critic_loss"].detach()),
        "policy_loss": float(info["policy_loss"].detach()),
        "td_error": float(info["td_error"]),
        "target_q_value": float(info["target_q_value"]),
        "q_value": float(info["q_value"]),
    }

result = {"code_root": str(root), "ding_file": str(Path(ding.__file__).resolve()),
          "batch_indices": [int(indices[0]), int(indices[-1])], "batch_sha256": batch_hash,
          "batch_size": len(indices), "validation_updates": 0, "groups": {}}
groups = {"A": (10, False, 0.0)} if args.baseline_only else {
    "A": (10, False, 0.0), "B": (2, False, 0.0)}
saved_q = {}
saved_actor = {}
for name, (n, ln, drop) in groups.items():
    obj = policy(n, ln, drop)
    obj._learn_model.eval()
    obj._target_model.eval()
    qs = q(obj)
    tqs = q(obj, target=True)
    with torch.no_grad():
        actor = obj._learn_model.forward(obs, mode="compute_actor")["logit"][0].detach().cpu()
    saved_q[name] = qs
    saved_actor[name] = actor
    row = {"critic_params": sum(v.numel() for v in obj._model.critic.parameters()),
           "actor_params": sum(v.numel() for v in obj._model.actor.parameters()),
           "q_eval": qstats(qs), "target_q_eval": qstats(tqs),
           "q_probe": qs[:, :4].tolist(), "target_q_probe": tqs[:, :4].tolist(),
           "actor_probe": actor[:4].tolist()}
    if name == "B":
        assert torch.equal(qs, saved_q["A"][:2])
        row["head_slice_exact"] = True
    if name != "A":
        assert torch.equal(actor, saved_actor["A"])
        row["actor_exact"] = True
        for stem in ("trainer/qfs", "trainer/target_qfs"):
            module = obj._model.critic if stem == "trainer/qfs" else obj._target_model.critic
            for key, val in checkpoint[stem].items():
                assert torch.equal(module.state_dict()[key].cpu(), val[:2])
        row["linear_mapping_exact"] = True
    if name == "B":
        action_var = act.clone().requires_grad_()
        score = obj._learn_model.forward(
            {"obs": obs, "action": action_var}, mode="compute_critic"
        )["q_value"].min(dim=0)[0].mean()
        action_grad = torch.autograd.grad(score, action_var)[0]
        assert torch.isfinite(action_grad).all() and action_grad.abs().max() > 0
        row["actor_action_gradient_finite_nonzero"] = True
    if name in ("A", "B"):
        old_target_params = [v.detach().clone() for v in obj._target_model.critic.parameters()]
        critic_before = [v.detach().clone() for v in obj._model.critic.parameters()]
        row["update"] = update(obj)
        result["validation_updates"] += 1
        row["critic_grad_l2"] = math.sqrt(sum(float(v.grad.square().sum()) for v in obj._model.critic.parameters() if v.grad is not None))
        row["actor_grad_l2"] = math.sqrt(sum(float(v.grad.square().sum()) for v in obj._model.actor.parameters() if v.grad is not None))
        assert row["critic_grad_l2"] > 0 and row["actor_grad_l2"] > 0
        assert all(math.isfinite(v) for k, v in row["update"].items()
                   if k not in ("peak_cuda_allocated_bytes",))
        if name == "B":
            head_metrics = []
            params = list(obj._model.critic.parameters())
            for head in range(2):
                grads = [param.grad[head] for param in params
                         if param.grad is not None and param.ndim > 0 and param.shape[0] == 2]
                deltas = [(param.detach() - old)[head] for param, old in
                          zip(params, critic_before)
                          if param.ndim > 0 and param.shape[0] == 2]
                head_metrics.append({
                    "head": head,
                    "critic_grad_l2": math.sqrt(sum(float(g.square().sum()) for g in grads)),
                    "critic_parameter_update_l2": math.sqrt(sum(float(d.square().sum()) for d in deltas)),
                })
            row["per_head_update_diagnostics"] = head_metrics
            assert all(item["critic_grad_l2"] > 0 and item["critic_parameter_update_l2"] > 0
                       for item in head_metrics)
            assert all(v.grad is None for v in obj._target_model.critic.parameters())
            row["target_no_gradient"] = True
            state = obj._state_dict_learn()
            fresh = policy(2, False, 0.0)
            fresh._load_state_dict_learn(state)
            fresh._learn_model.eval()
            obj._learn_model.eval()
            assert torch.equal(q(fresh), q(obj))
            with torch.no_grad():
                aa = obj._learn_model.forward(obs, mode="compute_actor")["logit"][0]
                bb = fresh._learn_model.forward(obs, mode="compute_actor")["logit"][0]
            assert torch.equal(aa, bb)
            row["fresh_reload_exact"] = True
    if not args.baseline_only and name in ("A", "B"):
            obj._cfg.learn.online = True
            obj._forward_learn_cnt = 0
            timings = []
            for iteration in range(14):
                torch.cuda.synchronize()
                start = time.perf_counter()
                obj._forward_learn(data)
                torch.cuda.synchronize()
                if iteration >= 4:
                    timings.append(time.perf_counter() - start)
            result["validation_updates"] += 14
            row["microbench"] = {"warmup_updates": 4, "timed_updates": 10,
                "update_seconds": timings,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()}
    result["groups"][name] = row
    del obj

if not args.baseline_only:
    assert torch.equal(saved_q["B"], saved_q["A"][:2])
    assert all(result["groups"]["B"][key] for key in
               ("head_slice_exact", "linear_mapping_exact", "actor_exact",
                "fresh_reload_exact", "target_no_gradient",
                "actor_action_gradient_finite_nonzero"))
    result["two_head_initialization_and_update_checks"] = True
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"output": str(args.output), "updates": result["validation_updates"],
                  "groups": list(result["groups"])}, indent=2))
