"""T001 engineering smoke test runner (does NOT modify learning rules).

Wraps the official SO2 entrypoint so that the pipeline can be stopped with an
explicit, logged budget instead of the upstream hard-coded `sys.exit(0)` at
150k collector env-steps.  It imports the *unmodified* upstream
`offline_train` configuration builder and only overrides budget/plumbing knobs
that the task allows:

  * seed / env_id / checkpoint path      (already CLI args upstream)
  * max_env_steps                        (new explicit stop budget)
  * max_wall_clock_sec                   (new explicit wall-clock stop)
  * eval_freq / n_evaluator_episode      (evaluation protocol, from the task)
  * exp_name / log dir                   (so the smoke run is not confused
                                          with a formal run)

Everything else (loss, target computation, update frequencies, batch sizes,
random collect size, offline/online mixing) is inherited from the upstream
configuration, so this script cannot silently change the learning rule.

Usage:
    python _so2_work/t001_smoke.py --seed 0 \
        --env_id halfcheetah-medium-replay-v2 \
        --ckpt_path _so2_work/assets/checkpoints/ckpt/so2_halfcheetah_medium_replay.ckpt \
        --max_env_steps 6000 --max_wall_clock_sec 900 \
        --eval_freq 2500 --eval_episodes 5
"""

# NOTE (T001 layout): this helper lives in _so2_work/scripts/. Its outputs go to
#   _so2_work/logs/<name>/   and the official checkpoint lives at
#   _so2_work/assets/checkpoints/ckpt/halfcheetah-medium-replay-v2.ckpt
import argparse
import json
import os
import sys
import time
from copy import deepcopy

# Make the repository root importable regardless of CWD.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "SO2"))


# T001: import d4rl BEFORE anything from ding. d4rl is what registers the D4RL gym ids
# (e.g. 'halfcheetah-medium-replay-v2'); DI-engine's mujoco env only calls plain gym.make().
import d4rl  # noqa: F401,E402
import gym  # noqa: F401,E402


def build_config(args):
    """Reuse upstream `train.offline_train` config assembly verbatim."""
    env = args.env_id.split("-")[0]
    if env == "halfcheetah":
        from dizoo.mujoco.config.halfcheetah_sac_default_config import main_config, create_config
        main_config.policy.model.critic_ensemble_size = 10
    elif env == "hopper":
        from dizoo.mujoco.config.hopper_sac_default_config import main_config, create_config
        main_config.policy.model.critic_ensemble_size = 50
    elif env == "walker2d":
        from dizoo.mujoco.config.walker2d_sac_default_config import main_config, create_config
        main_config.policy.model.critic_ensemble_size = 10
    else:
        raise ValueError(f"unsupported env_id prefix: {env}")

    main_config.policy.learn.learner = dict()
    main_config.policy.learn.learner.hook = dict()

    if args.ckpt_path is not None:
        main_config.policy.learn.learner.load_path = args.ckpt_path
    else:
        main_config.policy.learn.learner.load_path = f"ckpt/{args.env_id}.ckpt"

    main_config.policy.learn.learner.hook.load_ckpt_before_run = main_config.policy.learn.learner.load_path
    main_config.policy.learn.learner.hook.log_show_after_iter = args.log_show_after_iter
    main_config.policy.learn.learner.hook.save_ckpt_after_iter = 10 ** 10
    main_config.policy.learn.learner.hook.save_ckpt_after_run = True

    # ---- T001 stop-budget plumbing (read by the patched pipeline) ----
    main_config.policy.learn.max_env_steps = args.max_env_steps
    main_config.policy.learn.max_wall_clock_sec = args.max_wall_clock_sec

    create_config.policy.type = "edac"
    create_config.policy.import_names = ["ding.policy.edac"]

    main_config.exp_name = args.exp_name

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
    main_config.policy.collect.data_path = args.data_path
    # Only override when the CLI actually provided a value; otherwise keep the upstream
    # config default (10000, or 5000 as set by train.py). Setting None would crash the
    # pipeline's `random_collect_size > 0` check.
    if args.random_collect_size is not None and args.random_collect_size >= 0:
        main_config.policy.random_collect_size = args.random_collect_size
    main_config.env.env_id = args.env_id
    main_config.policy.eval.evaluator = dict()
    main_config.policy.eval.evaluator.eval_freq = args.eval_freq
    main_config.policy.eval.evaluator.n_evaluator_episode = args.eval_episodes

    # smooth target policy (upstream default)
    main_config.policy.learn.noise = True
    main_config.policy.learn.noise_sigma = 0.3
    main_config.policy.learn.noise_range = dict(min=-0.6, max=0.6)
    return deepcopy([main_config, create_config])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", "-s", type=int, default=0)
    ap.add_argument("--env_id", "-e", type=str, default="halfcheetah-medium-replay-v2")
    ap.add_argument("--ckpt_path", "-c", type=str, default=None)
    ap.add_argument("--data_path", type=str, default=None)
    ap.add_argument("--max_env_steps", type=int, default=6000,
                    help="total training env steps INCLUDING random collection")
    ap.add_argument("--max_wall_clock_sec", type=float, default=900.0)
    ap.add_argument("--eval_freq", type=int, default=2500)
    ap.add_argument("--eval_episodes", type=int, default=5)
    ap.add_argument("--random_collect_size", type=int, default=5000,
                    help="mirrors train.py's 5000; pass -1 to keep the config file default")
    ap.add_argument("--log_show_after_iter", type=int, default=200,
                    help="learner log interval (upstream uses 2000)")
    ap.add_argument("--exp_name", type=str, default=None)
    args = ap.parse_args()

    if args.exp_name is None:
        args.exp_name = f"t001_smoke_{args.env_id}_seed{args.seed}"

    t0 = time.time()
    cfg = build_config(args)
    print(f"[T001] config built in {time.time() - t0:.2f}s; exp_name={args.exp_name}", flush=True)
    print(f"[T001] budget: max_env_steps={args.max_env_steps} "
          f"max_wall_clock_sec={args.max_wall_clock_sec}", flush=True)

    from ding.entry import serial_pipeline_offline2online

    meta = {
        "seed": args.seed,
        "env_id": args.env_id,
        "ckpt_path": args.ckpt_path,
        "max_env_steps": args.max_env_steps,
        "max_wall_clock_sec": args.max_wall_clock_sec,
        "eval_freq": args.eval_freq,
        "eval_episodes": args.eval_episodes,
        "exp_name": args.exp_name,
        "cwd": os.getcwd(),
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    os.makedirs(args.exp_name, exist_ok=True)
    with open(os.path.join(args.exp_name, "smoke_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    policy, stop = serial_pipeline_offline2online(
        cfg,
        seed=args.seed,
        max_env_steps=args.max_env_steps,
        max_wall_clock_sec=args.max_wall_clock_sec,
    )
    wall = time.time() - t0
    print(f"[T001] pipeline returned after {wall:.1f}s; stop={stop}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
