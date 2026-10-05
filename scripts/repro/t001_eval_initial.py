"""T001 step C3: load the official checkpoint and run an initial 5-episode evaluation.

No training. Reports raw return and D4RL normalized return. Also prints the actual
number of evaluation episodes executed so the report does not have to guess.
"""

# NOTE (T001 layout): this helper lives in _so2_work/scripts/. Its outputs go to
#   _so2_work/logs/<name>/   and the official checkpoint lives at
#   _so2_work/assets/checkpoints/ckpt/halfcheetah-medium-replay-v2.ckpt
import json
import os
import sys
import time
import traceback
from functools import partial

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "SO2"))

CKPT = os.environ.get(
    "SO2_CKPT", os.path.join(_REPO, "_so2_work/assets/checkpoints/ckpt/halfcheetah-medium-replay-v2.ckpt")
)
ENV_ID = os.environ.get("SO2_ENV_ID", "halfcheetah-medium-replay-v2")
SEED = int(os.environ.get("SO2_SEED", "0"))
N_EPISODE = int(os.environ.get("SO2_EVAL_EPISODES", "5"))
EXP_NAME = os.environ.get("SO2_EXP_NAME", "t001_eval_initial")


# T001: import d4rl BEFORE anything from ding. d4rl is what registers the D4RL gym ids
# (e.g. 'halfcheetah-medium-replay-v2'); DI-engine's mujoco env only calls plain gym.make().
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
    main_config.exp_name = EXP_NAME
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
    t0 = time.time()
    import torch
    from tensorboardX import SummaryWriter
    from ding.config import compile_config
    from ding.envs import get_vec_env_setting, create_env_manager
    from ding.policy import create_policy
    from ding.worker import BaseLearner, InteractionSerialEvaluator
    from ding.utils import set_pkg_seed

    main_config, create_config = build_cfg()
    cfg = compile_config(main_config, seed=SEED, env=None, auto=True,
                         create_cfg=create_config, save_cfg=True)
    cfg.policy.type = cfg.policy.type + "_command"
    cfg.policy.import_names = ["ding.policy.edac", "ding.policy.command_mode_policy_instance"]

    set_pkg_seed(SEED, use_cuda=cfg.policy.cuda)
    env_fn, collector_cfg, evaluator_cfg = get_vec_env_setting(cfg.env)
    print("evaluator env_num:", len(evaluator_cfg),
          "| cfg n_evaluator_episode:", evaluator_cfg[0].get("n_evaluator_episode"), flush=True)

    evaluator_env = create_env_manager(cfg.env.manager,
                                       [partial(env_fn, cfg=c) for c in evaluator_cfg])
    evaluator_env.seed(SEED, dynamic_seed=False)

    policy = create_policy(cfg.policy, model=None,
                           enable_field=["learn", "collect", "eval", "command"])
    tb = SummaryWriter(os.path.join("./{}/log/".format(cfg.exp_name), "eval"))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb, exp_name=cfg.exp_name)
    learner.call_hook("before_run")
    print("checkpoint loaded:", CKPT, flush=True)

    evaluator = InteractionSerialEvaluator(
        cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb, exp_name=cfg.exp_name
    )
    n_env = evaluator_env.env_num
    # VectorEvalMonitor asserts n_episode >= env_num and distributes n_episode episodes
    # across env_num envs (floor, with the remainder given to the first envs).
    n_episode_req = N_EPISODE
    if n_episode_req < n_env:
        print(f"note: requested n_episode={n_episode_req} < env_num={n_env}; "
              f"bumping to {n_env} (VectorEvalMonitor assertion)", flush=True)
        n_episode_req = n_env
    print(f"requested n_episode={N_EPISODE}; effective n_episode={n_episode_req}; "
          f"env_num={n_env} -> total scored episodes = {n_episode_req}", flush=True)

    t1 = time.time()
    stop, reward = evaluator.eval(None, 0, 0, n_episode=n_episode_req)
    eval_sec = time.time() - t1

    raw = float(reward)
    try:
        import d4rl
        import gym
        env = gym.make(ENV_ID)
        normalized = float(d4rl.get_normalized_score(ENV_ID, raw))
        env.close()
    except Exception as e:  # noqa: BLE001
        normalized = None
        print("normalization failed:", repr(e), flush=True)

    rec = {
        "step": "C3_initial_eval",
        "env_id": ENV_ID,
        "seed": SEED,
        "checkpoint": CKPT,
        "n_episode_requested": N_EPISODE,
        "n_episode_effective": n_episode_req,
        "n_evaluator_env": n_env,
        "raw_return": raw,
        "normalized_return": normalized,
        "stop_flag": bool(stop),
        "eval_wall_sec": round(eval_sec, 3),
        "total_wall_sec": round(time.time() - t0, 3),
        "deterministic_eval": True,
        "note": "VectorEvalMonitor distributes n_episode across env_num envs; total scored "
                "episodes equals n_episode. n_episode >= env_num is enforced by assertion.",
    }
    print("RESULT_JSON " + json.dumps(rec), flush=True)
    out = os.path.join(cfg.exp_name, "C3_initial_eval.json")
    os.makedirs(cfg.exp_name, exist_ok=True)
    with open(out, "w") as f:
        json.dump(rec, f, indent=2)
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
