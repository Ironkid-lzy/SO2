"""T001 step C6 helper: attribute the per-env-step wall clock.

Three runs agree that a policy env step costs ~0.20-0.25 s even after subtracting the
measured gradient-update and evaluation time. This script times the two prime suspects
directly:

  (a) raw MuJoCo stepping through the DI-engine env wrapper (no policy, no learning)
  (b) `SampleSerialCollector.collect()` end-to-end (policy forward + env stepping +
      transition processing + train-sample construction) for one episode

together with the pytorch-only gradient update cost already measured by
`t001_microbench.py`.
"""

# NOTE (T001 layout): this helper lives in _so2_work/scripts/. Its outputs go to
#   _so2_work/logs/<name>/   and the official checkpoint lives at
#   _so2_work/assets/checkpoints/ckpt/halfcheetah-medium-replay-v2.ckpt
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
import numpy as np  # noqa: E402


def main():
    from functools import partial
    from tensorboardX import SummaryWriter
    from ding.config import compile_config
    from ding.envs import get_vec_env_setting, create_env_manager
    from ding.policy import create_policy
    from ding.utils import set_pkg_seed
    from ding.worker import BaseLearner
    from ding.worker import create_serial_collector

    sys.path.insert(0, os.path.join(_REPO, "_so2_work"))
    import t001_microbench as MB

    main_config, create_config = MB.build_cfg()
    cfg = compile_config(main_config, seed=0, env=None, auto=True,
                         create_cfg=create_config, save_cfg=False)
    cfg.policy.type = cfg.policy.type + "_command"
    cfg.policy.import_names = ["ding.policy.edac", "ding.policy.command_mode_policy_instance"]
    set_pkg_seed(0, use_cuda=cfg.policy.cuda)

    env_fn, collector_cfg, _ = get_vec_env_setting(cfg.env)
    out = {"env_id": ENV_ID}

    # (a) raw env stepping through the DI-engine wrapper
    env = env_fn(cfg=collector_cfg[0])
    env.seed(0)
    obs = env.reset()
    act = np.zeros(6, dtype=np.float32)
    for _ in range(50):          # warmup
        env.step(act)
    t0 = time.time()
    n = 1000
    for _ in range(n):
        env.step(act)
    dt = (time.time() - t0) / n
    out["raw_env_step_sec"] = round(dt, 6)
    out["raw_env_steps_per_sec"] = round(1.0 / dt, 1)
    print(f"[attr] raw env step: {dt * 1000:.3f} ms ({1.0 / dt:.0f} steps/s)", flush=True)
    env.close()

    # (b) collector.collect() end-to-end for one episode (n_sample=1 -> one episode)
    collector_env = create_env_manager(
        cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_cfg])
    collector_env.seed(0)
    policy = create_policy(cfg.policy, model=None,
                           enable_field=["learn", "collect", "eval", "command"])
    tb = SummaryWriter("./t001_attr/log")
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb, exp_name="attr")
    learner.call_hook("before_run")
    collector = create_serial_collector(cfg.policy.collect.collector, env=collector_env,
                                        policy=policy.collect_mode, tb_logger=tb,
                                        exp_name="attr")
    collector.reset_policy(policy.collect_mode)
    _ = collector.collect(n_sample=1)      # warmup
    t0 = time.time()
    data = collector.collect(n_sample=1)
    dt = time.time() - t0
    out["collect_one_episode_sec"] = round(dt, 4)
    out["collect_one_episode_samples"] = len(data)
    out["collect_one_episode_envsteps"] = int(collector.envstep)
    print(f"[attr] collect(1 episode): {dt:.3f} s, {len(data)} train samples, "
          f"collector.envstep={collector.envstep}", flush=True)

    # (c) gradient update, same config
    import torch
    from ding.utils.data import create_dataset
    dataset = create_dataset(cfg)
    policy._cfg.learn.online = True
    idx = np.random.RandomState(0).choice(len(dataset), size=2560, replace=False)
    batch = [dataset[int(i)] for i in idx]
    for _ in range(3):
        policy._forward_learn(batch)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(20):
        policy._forward_learn(batch)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / 20
    out["one_update_sec"] = round(dt, 6)
    print(f"[attr] one update (batch 2560): {dt * 1000:.2f} ms", flush=True)

    os.makedirs("_so2_work/logs/attribution", exist_ok=True)
    with open("_so2_work/logs/attribution/attribution.json", "w") as f:
        json.dump(out, f, indent=2)
    print("ATTRIBUTION_JSON " + json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
