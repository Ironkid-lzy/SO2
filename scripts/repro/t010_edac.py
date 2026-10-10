"""EDAC offline update adapted from snu-mllab/EDAC 198d570 (MIT).

The network and checkpoint keys are shared with the SO2 policy. This module
deliberately does not use SO2's online loss as an offline SAC trainer.
"""
import copy
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Normal

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "SO2"))
from ding.model.template.ensemble_qac import ENSEMBLEQAC
from ding.torch_utils import pytorch_util as ptu

REFERENCE_SHA = "198d5708701b531fd97a918a33152e1914ea14d7"
REFERENCE = Path("/home/lzy/Projects/EDAC_t010_pinned")
DATASET = Path("/home/lzy/.d4rl/datasets/halfcheetah_medium_replay-v2.hdf5")
DATA_SHA = "48d494a4770c11f48260736dec090b78bdca8375647ed4d24bfa5b6610c7f683"
ACTOR_MAP = {
    "0.weight": "fc0.weight", "0.bias": "fc0.bias",
    "2.main.0.weight": "fc1.weight", "2.main.0.bias": "fc1.bias",
    "2.main.2.weight": "fc2.weight", "2.main.2.bias": "fc2.bias",
    "2.mu.weight": "last_fc.weight", "2.mu.bias": "last_fc.bias",
    "2.log_sigma_layer.weight": "last_fc_log_std.weight",
    "2.log_sigma_layer.bias": "last_fc_log_std.bias",
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def tensor_hash(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        arr = value.detach().cpu().contiguous().numpy()
        h.update(key.encode()); h.update(str(arr.dtype).encode())
        h.update(str(arr.shape).encode()); h.update(arr.tobytes())
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def atomic_torch(path, value):
    path = Path(path)
    if path.exists():
        raise RuntimeError(f"refuse checkpoint overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(value, tmp); os.replace(tmp, path)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def restore_rng(state):
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def seed_all(seed, cuda=False):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


def make_model(device="cpu", regularized=True, actor_bounds=(-20., 2.)):
    model = ENSEMBLEQAC(
        17, 6, "reparameterization", twin_critic=True,
        actor_head_hidden_size=256, actor_head_layer_num=2,
        critic_head_hidden_size=256, critic_ensemble_size=2,
        critic_layer_norm=regularized, critic_dropout_rate=.005 if regularized else 0.,
        actor_log_sigma_bounds=actor_bounds)
    # Match the official EDAC actor initializer, including its .001 output range.
    for layer in (model.actor[0], model.actor[2].main[0], model.actor[2].main[2]):
        ptu.fanin_init(layer.weight); layer.bias.data.fill_(.1)
    for layer in (model.actor[2].mu, model.actor[2].log_sigma_layer):
        layer.weight.data.uniform_(-.001, .001); layer.bias.data.uniform_(-.001, .001)
    return model.to(device)


def sample_actor(model, obs, reparameterize=True):
    mu, std = model(obs, mode="compute_actor")["logit"]
    dist = Normal(mu, std)
    z = mu + std * Normal(torch.zeros_like(mu), torch.ones_like(std)).sample() if reparameterize else dist.sample().detach()
    action = z.tanh()
    logp = (dist.log_prob(z) - torch.log(1 - action.square() + 1e-6)).sum(-1, keepdim=True)
    return action, mu, std.log(), logp


def q_values(model, obs, action):
    return model.critic(torch.cat((obs, action), dim=-1))["pred"].unsqueeze(-1)


class OfflineEDAC:
    def __init__(self, model):
        self.model = model
        self.target = copy.deepcopy(model)
        self.log_alpha = torch.zeros(1, device=next(model.parameters()).device, requires_grad=True)
        self.actor_optim = torch.optim.Adam(model.actor.parameters(), lr=3e-4)
        self.critic_optim = torch.optim.Adam(model.critic.parameters(), lr=3e-4)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=3e-4)
        self.updates = 0

    def losses(self, batch):
        obs, next_obs = batch["observations"], batch["next_observations"]
        actions = batch["actions"].requires_grad_(True)
        act, _, _, logp = sample_actor(self.model, obs)
        alpha_loss = -(self.log_alpha * (logp - 6).detach()).mean()
        alpha = self.log_alpha.exp()
        actor_loss = (alpha * logp - q_values(self.model, obs, act).min(0)[0]).mean()
        prediction = q_values(self.model, obs, actions)
        next_act, _, _, next_logp = sample_actor(self.model, next_obs, reparameterize=False)
        target_value = q_values(self.target, next_obs, next_act).min(0)[0] - alpha * next_logp
        target = batch["rewards"] + (1 - batch["terminals"]) * .99 * target_value
        td = (prediction - target.detach().unsqueeze(0)).square().mean(dim=(1, 2)).sum()
        obs_tile = obs.unsqueeze(0).repeat(2, 1, 1)
        action_tile = actions.unsqueeze(0).repeat(2, 1, 1).requires_grad_(True)
        # One train-mode forward: each head receives independent dropout masks.
        qtile = q_values(self.model, obs_tile, action_tile)
        grad, = torch.autograd.grad(qtile.sum(), action_tile, retain_graph=True, create_graph=True)
        grad = grad / (grad.norm(p=2, dim=2).unsqueeze(-1) + 1e-10)
        grad = grad.transpose(0, 1)
        similarities = torch.einsum("bik,bjk->bij", grad, grad)
        mask = torch.eye(2, device=obs.device).unsqueeze(0).repeat(obs.shape[0], 1, 1)
        es = ((1 - mask) * similarities).sum(dim=(1, 2)).mean() / (2 - 1)
        return {"alpha": alpha_loss, "actor": actor_loss, "td": td, "es": es, "critic": td + es}

    def update(self, batch):
        self.model.train(); self.target.train()
        losses = self.losses(batch)
        # Official order. Q gradients from actor loss are discarded before Q loss.
        for key, optim in (("alpha", self.alpha_optim), ("actor", self.actor_optim), ("critic", self.critic_optim)):
            optim.zero_grad(); losses[key].backward(); optim.step()
        with torch.no_grad():
            for src, dst in zip(self.model.critic.parameters(), self.target.critic.parameters()):
                dst.mul_(.995).add_(src, alpha=.005)
            # Target actor is unused by both backups; keep schema fully populated.
            self.target.actor.load_state_dict(self.model.actor.state_dict(), strict=True)
        self.updates += 1
        return {key: float(value.detach()) for key, value in losses.items()}

    def snapshot(self, identity):
        return {"schema": "t010-full-v1", "model": self.model.state_dict(),
                "target_model": self.target.state_dict(), "log_alpha": self.log_alpha.detach().clone(),
                "optimizer_policy": self.actor_optim.state_dict(), "optimizer_q": self.critic_optim.state_dict(),
                "optimizer_alpha": self.alpha_optim.state_dict(), "rng": rng_state(),
                "updates": self.updates, "train_env_steps": 0, "identity": identity,
                "online_critic_sha256": tensor_hash(self.model.critic.state_dict()),
                "target_critic_sha256": tensor_hash(self.target.critic.state_dict())}

    def restore(self, state):
        assert state["schema"] == "t010-full-v1"
        self.model.load_state_dict(state["model"], strict=True)
        self.target.load_state_dict(state["target_model"], strict=True)
        self.log_alpha.data.copy_(state["log_alpha"])
        self.actor_optim.load_state_dict(state["optimizer_policy"])
        self.critic_optim.load_state_dict(state["optimizer_q"])
        self.alpha_optim.load_state_dict(state["optimizer_alpha"])
        self.updates = state["updates"]; restore_rng(state["rng"])


def load_dataset():
    import gym
    import d4rl
    import inspect
    assert digest(DATASET) == DATA_SHA
    env = gym.make("halfcheetah-medium-replay-v2")
    try:
        raw = env.get_dataset()
        data = d4rl.qlearning_dataset(env, dataset=raw)
        meta = {"hdf5_sha256": DATA_SHA, "hdf5_keys": sorted(raw),
                "raw_count": len(raw["rewards"]), "raw_terminals": int(raw["terminals"].sum()),
                "raw_timeouts": int(raw["timeouts"].sum()), "transitions": len(data["rewards"]),
                "processed_terminals": int(data["terminals"].sum()),
                "qlearning_source_sha256": hashlib.sha256(inspect.getsource(d4rl.qlearning_dataset).encode()).hexdigest(),
                "normalization": "none", "terminate_on_end": False, "arrays": {}}
        for key, arr in data.items():
            meta["arrays"][key] = {"shape": list(arr.shape), "dtype": str(arr.dtype),
                                   "sha256": hashlib.sha256(arr.tobytes()).hexdigest()}
        assert meta["processed_terminals"] == 0, "SO2 ignore_done requires explicit terminal audit"
        return data, meta
    finally:
        env.close()


def batch_at(data, indices, device):
    return {key: torch.as_tensor(arr[indices].copy(), dtype=torch.float32, device=device).reshape(len(indices), -1)
            for key, arr in data.items()}
