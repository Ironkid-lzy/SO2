"""T001 step C5: reproducibility check on a fixed-input update.

The task asks to check whether a fixed-input update is repeatable and to record which
parts are reproducible and which are not, without promising bitwise cross-hardware
equality.

Method (independent of the smoke run): build the exact training config, load the
official checkpoint, take a FIXED batch of transitions from the D4RL dataset, then run
a fixed number of real EDAC updates in two fresh processes with the same seed and once
with a different seed. Compare the resulting critic weights bitwise and numerically.
"""

# NOTE (T001 layout): this helper lives in _so2_work/scripts/. Its outputs go to
#   _so2_work/logs/<name>/   and the official checkpoint lives at
#   _so2_work/assets/checkpoints/ckpt/halfcheetah-medium-replay-v2.ckpt
import argparse
import json
import os
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "SO2"))

CKPT = os.environ.get(
    "SO2_CKPT", os.path.join(_REPO, "_so2_work/assets/checkpoints/ckpt/halfcheetah-medium-replay-v2.ckpt")
)
ENV_ID = os.environ.get("SO2_ENV_ID", "halfcheetah-medium-replay-v2")

# T001: d4rl first (registers gym ids), then ding.
import d4rl  # noqa: F401,E402
import gym  # noqa: F401,E402


def build_cfg():
    from dizoo.mujoco.config.halfcheetah_sac_default_config import main_config, create_config

    main_config.policy.model.critic_ensemble_size = 10
    main_config.policy.learn.learner = dict()
    main_config.policy.learn.learner.hook = dict()
    main_config.policy.learn.learner.load_path = CKPT
    main_config.policy.learn.learner.hook.load_ckpt_before_run = CKPT
    main_config.policy.learn.learner.hook.save_ckpt_after_run = False
    main_config.exp_name = "t001_seedcheck"
    create_config.policy.type = "edac"
    create_config.policy.import_names = ["ding.policy.edac"]
    main_config.policy.learn.online = False
    main_config.policy.learn.offline_pretrain_iterations = 0
    main_config.policy.learn.online_pretrain_iterations = 0
    main_config.policy.learn.update_per_collect = 10
    main_config.policy.learn.actor_update_freq = 10
    main_config.policy.learn.concat_online_ratio = 0.1
    main_config.policy.learn.batch_size = 256
    main_config.policy.collect.n_sample = 1
    main_config.policy.learn.online_ratio = 0.5
    main_config.policy.learn.offline_data_ratio = 1
    main_config.policy.learn.without_timeouts_done = True
    main_config.policy.collect.data_type = "d4rl"
    main_config.policy.collect.data_path = None
    main_config.policy.random_collect_size = 5000
    main_config.env.env_id = ENV_ID
    main_config.policy.eval.evaluator = dict()
    main_config.policy.eval.evaluator.eval_freq = 2500
    main_config.policy.learn.noise = True
    main_config.policy.learn.noise_sigma = 0.3
    main_config.policy.learn.noise_range = dict(min=-0.6, max=0.6)
    return main_config, create_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_updates", type=int, default=10)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or f"seed{args.seed}"

    import numpy as np
    import torch
    from tensorboardX import SummaryWriter
    from ding.config import compile_config
    from ding.utils import set_pkg_seed
    from ding.utils.data import create_dataset
    from ding.policy import create_policy
    from ding.worker import BaseLearner

    main_config, create_config = build_cfg()
    cfg = compile_config(main_config, seed=args.seed, env=None, auto=True,
                         create_cfg=create_config, save_cfg=False)
    cfg.policy.type = cfg.policy.type + "_command"
    cfg.policy.import_names = ["ding.policy.edac", "ding.policy.command_mode_policy_instance"]

    set_pkg_seed(args.seed, use_cuda=cfg.policy.cuda)
    dataset = create_dataset(cfg)

    policy = create_policy(cfg.policy, model=None,
                           enable_field=["learn", "collect", "eval", "command"])
    tb = SummaryWriter(os.path.join("./t001_seedcheck", "log", tag))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb, exp_name=tag)
    learner.call_hook("before_run")

    # Fixed input batch: deterministic sampling independent of the global RNG used by
    # training, so both runs see byte-identical inputs.
    rng = np.random.RandomState(1234)
    idx = rng.choice(len(dataset), size=256, replace=False)
    data = [dataset[int(i)] for i in idx]
    obs0 = torch.stack([d["obs"] for d in data])
    act0 = torch.stack([d["action"] for d in data])
    rew0 = torch.stack([d["reward"] for d in data])

    em = policy._learn_model
    critic_before = em.critic.fcs[0].W.detach().clone()
    actor_before = em.actor[0].weight.detach().clone()

    policy._cfg.learn.online = True
    logs = []
    for it in range(args.num_updates):
        log = policy._forward_learn(data)
        logs.append({
            "iter": it,
            "critic_loss": float(log["critic_loss"].detach()),
            "q_value": float(log["q_value"]),
            "target_q_value": float(log["target_q_value"]),
            "td_error": float(log["td_error"]),
            "alpha": float(log["alpha"]),
        })

    critic_after = em.critic.fcs[0].W.detach().clone()
    actor_after = em.actor[0].weight.detach().clone()
    delta_c = (critic_after - critic_before)
    delta_a = (actor_after - actor_before)

    # NOTE: use a *stable* hash. Python's built-in hash() of bytes is randomised per
    # process by PYTHONHASHSEED, which would make two identical runs look different.
    import hashlib
    def sha(t):
        return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()

    # Save a compact digest of the resulting weights for cross-run comparison.
    digest = {
        "seed": args.seed,
        "tag": tag,
        "num_updates": args.num_updates,
        "critic_sum": float(critic_after.double().sum()),
        "critic_absmax": float(critic_after.abs().max()),
        "critic_sha256": sha(critic_after),
        "actor_sum": float(actor_after.double().sum()),
        "actor_sha256": sha(actor_after),
        "critic_delta_absmax": float(delta_c.abs().max()),
        "actor_delta_absmax": float(delta_a.abs().max()),
        "input_obs_sum": float(obs0.double().sum()),
        "input_act_sum": float(act0.double().sum()),
        "input_rew_sum": float(rew0.double().sum()),
        "logs": logs,
        "torch_deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
    }
    print("SEEDCHECK_JSON " + json.dumps(digest), flush=True)
    os.makedirs("_so2_work/logs/seedcheck", exist_ok=True)
    with open(f"_so2_work/logs/seedcheck/{tag}.json", "w") as f:
        json.dump(digest, f, indent=2)
    torch.save(critic_after.cpu(), f"_so2_work/logs/seedcheck/{tag}_critic.pt")
    torch.save(actor_after.cpu(), f"_so2_work/logs/seedcheck/{tag}_actor.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
