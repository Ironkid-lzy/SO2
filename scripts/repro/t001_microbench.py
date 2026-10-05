"""T001 step C6 helper: independent microbenchmark of one real EDAC training update.

The end-to-end timing run reports ~19.7 ms per `learner.train()` call at the exact
SO2 configuration (batch 2560, 10 critics). This script measures the same quantity
directly and additionally breaks the cost down by batch size, so the 100k-step
projection can be checked against a second, independent method.
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
    main_config.exp_name = "t001_microbench"
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
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--batch_sizes", default="256,2560")
    args = ap.parse_args()

    import numpy as np
    import torch
    from tensorboardX import SummaryWriter
    from ding.config import compile_config
    from ding.utils import set_pkg_seed
    from ding.utils.data import create_dataset
    from ding.policy import create_policy
    from ding.worker import BaseLearner

    main_config, create_config = build_cfg()
    cfg = compile_config(main_config, seed=0, env=None, auto=True,
                         create_cfg=create_config, save_cfg=False)
    cfg.policy.type = cfg.policy.type + "_command"
    cfg.policy.import_names = ["ding.policy.edac", "ding.policy.command_mode_policy_instance"]
    set_pkg_seed(0, use_cuda=cfg.policy.cuda)
    dataset = create_dataset(cfg)

    policy = create_policy(cfg.policy, model=None,
                           enable_field=["learn", "collect", "eval", "command"])
    tb = SummaryWriter("./t001_microbench/log")
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb, exp_name="microbench")
    learner.call_hook("before_run")
    policy._cfg.learn.online = True

    rng = np.random.RandomState(0)
    out = {"env_id": ENV_ID, "ckpt": CKPT, "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name(0), "results": []}

    for bs in [int(x) for x in args.batch_sizes.split(",")]:
        idx = rng.choice(len(dataset), size=bs, replace=False)
        data = [dataset[int(i)] for i in idx]
        for _ in range(args.warmup):
            policy._forward_learn(data)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(args.reps):
            policy._forward_learn(data)
        torch.cuda.synchronize()
        dt = (time.time() - t0) / args.reps
        rec = {"batch_size": bs, "reps": args.reps, "sec_per_update": round(dt, 6),
               "updates_per_sec": round(1.0 / dt, 3)}
        out["results"].append(rec)
        print(f"[microbench] batch={bs}: {dt * 1000:.2f} ms/update "
              f"({1.0 / dt:.1f} updates/s)", flush=True)

    os.makedirs("_so2_work/logs/microbench", exist_ok=True)
    with open("_so2_work/logs/microbench/microbench.json", "w") as f:
        json.dump(out, f, indent=2)
    print("MICROBENCH_JSON " + json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
