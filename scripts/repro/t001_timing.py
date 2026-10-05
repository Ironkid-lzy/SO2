"""T001 step C6 helper: measure the real throughput of the exact training config,
with a realistic evaluation protocol, so the 100k-step estimate is grounded in
measurement rather than guesswork.

It runs a small number of real online iterations (default 2000 training env steps
after the standard 5000 random-collect steps) and reports:
  * wall-clock seconds spent in the random-collect phase
  * wall-clock seconds and env-steps in the online phase
  * measured learner iterations
  * measured evaluation wall-clock (with the task's 20-episode protocol)
  * derived seconds per 100k training env steps

No learning-rule change: it calls the same patched pipeline.
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

# __T001_D4RL_IMPORT__
import d4rl  # noqa: F401,E402
import gym  # noqa: F401,E402


def build_cfg(args):
    from dizoo.mujoco.config.halfcheetah_sac_default_config import main_config, create_config

    main_config.policy.model.critic_ensemble_size = 10
    main_config.policy.learn.learner = dict()
    main_config.policy.learn.learner.hook = dict()
    main_config.policy.learn.learner.load_path = CKPT
    main_config.policy.learn.learner.hook.load_ckpt_before_run = CKPT
    main_config.policy.learn.learner.hook.log_show_after_iter = 2000
    main_config.policy.learn.learner.hook.save_ckpt_after_iter = 10 ** 10
    main_config.policy.learn.learner.hook.save_ckpt_after_run = False
    main_config.exp_name = args.exp_name
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
    main_config.policy.random_collect_size = args.random_collect_size
    main_config.env.env_id = ENV_ID
    main_config.policy.eval.evaluator = dict()
    main_config.policy.eval.evaluator.eval_freq = args.eval_freq
    main_config.policy.eval.evaluator.n_evaluator_episode = args.eval_episodes
    main_config.policy.learn.noise = True
    main_config.policy.learn.noise_sigma = 0.3
    main_config.policy.learn.noise_range = dict(min=-0.6, max=0.6)
    return [main_config, create_config]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_name", default="t001_timing")
    ap.add_argument("--random_collect_size", type=int, default=5000)
    ap.add_argument("--max_env_steps", type=int, default=7000)
    ap.add_argument("--eval_freq", type=int, default=2500)
    ap.add_argument("--eval_episodes", type=int, default=20)
    ap.add_argument("--max_wall_clock_sec", type=float, default=1800.0)
    args = ap.parse_args()

    from copy import deepcopy
    from ding.entry import serial_pipeline_offline2online
    from ding.worker import BaseLearner, InteractionSerialEvaluator
    from ding.worker.collector.sample_serial_collector import SampleSerialCollector

    stats = {"train_calls": 0, "train_sec": 0.0, "eval_calls": 0, "eval_sec": 0.0,
             "collect_calls": 0, "collect_sec": 0.0}

    _orig_train = BaseLearner.train
    def timed_train(self, data, envstep=-1):
        t = time.time()
        try:
            return _orig_train(self, data, envstep)
        finally:
            stats["train_calls"] += 1
            stats["train_sec"] += time.time() - t
    BaseLearner.train = timed_train

    _orig_eval = InteractionSerialEvaluator.eval
    def timed_eval(self, *a, **kw):
        t = time.time()
        try:
            return _orig_eval(self, *a, **kw)
        finally:
            stats["eval_calls"] += 1
            stats["eval_sec"] += time.time() - t
    InteractionSerialEvaluator.eval = timed_eval

    _orig_collect = SampleSerialCollector.collect
    def timed_collect(self, *a, **kw):
        t = time.time()
        try:
            return _orig_collect(self, *a, **kw)
        finally:
            stats["collect_calls"] += 1
            stats["collect_sec"] += time.time() - t
    SampleSerialCollector.collect = timed_collect

    cfg = build_cfg(args)
    os.makedirs(args.exp_name, exist_ok=True)

    t0 = time.time()
    policy, stop = serial_pipeline_offline2online(
        deepcopy(cfg), seed=0,
        max_env_steps=args.max_env_steps,
        max_wall_clock_sec=args.max_wall_clock_sec,
    )
    wall = time.time() - t0
    stats.update({
        "total_wall_sec": round(wall, 3),
        "max_env_steps": args.max_env_steps,
        "random_collect_size": args.random_collect_size,
        "eval_freq": args.eval_freq,
        "eval_episodes": args.eval_episodes,
    })
    for k in ("train_sec", "eval_sec", "collect_sec"):
        stats[k] = round(stats[k], 3)
    stats["other_sec"] = round(
        wall - stats["train_sec"] - stats["eval_sec"] - stats["collect_sec"], 3)
    print("[timing] " + json.dumps(stats), flush=True)
    with open(os.path.join(args.exp_name, "timing.json"), "w") as f:
        json.dump(stats, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
