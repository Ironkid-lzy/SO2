"""T001 preflight: build config / dataset / env / learner, verify checkpoint load,
then run ONE real EDAC update and check for finite values.

This is a build-time shakedown before the budgeted smoke run. It does not train to
any budget; it surfaces import / dataset / env-id / checkpoint-loading errors quickly.
Config assembly mirrors upstream `train.py`.
"""

# NOTE (T001 layout): this helper lives in _so2_work/scripts/. Its outputs go to
#   _so2_work/logs/<name>/   and the official checkpoint lives at
#   _so2_work/assets/checkpoints/ckpt/halfcheetah-medium-replay-v2.ckpt
import os
import sys
import traceback

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "SO2"))

CKPT = os.environ.get("SO2_CKPT", os.path.join(_REPO, "_so2_work/assets/checkpoints/ckpt/halfcheetah-medium-replay-v2.ckpt"))
ENV_ID = os.environ.get("SO2_ENV_ID", "halfcheetah-medium-replay-v2")


def step(name):
    def deco(fn):
        def wrapped(*a, **kw):
            print(f"\n[preflight] === {name} ===", flush=True)
            r = fn(*a, **kw)
            print(f"[preflight] --- {name}: OK", flush=True)
            return r
        return wrapped
    return deco


def build_cfg():
    from dizoo.mujoco.config.halfcheetah_sac_default_config import main_config, create_config

    main_config.policy.model.critic_ensemble_size = 10
    main_config.policy.learn.learner = dict()
    main_config.policy.learn.learner.hook = dict()
    main_config.policy.learn.learner.load_path = CKPT
    main_config.policy.learn.learner.hook.load_ckpt_before_run = CKPT
    main_config.policy.learn.learner.hook.log_show_after_iter = 20
    main_config.policy.learn.learner.hook.save_ckpt_after_iter = 10 ** 10
    main_config.policy.learn.learner.hook.save_ckpt_after_run = True
    main_config.exp_name = "t001_preflight"
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


@step("1. import train.py")
def s1():
    import importlib.util
    spec = importlib.util.spec_from_file_location("so2_train", os.path.join(_REPO, "train.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print("train.py imported; has offline_train:", hasattr(mod, "offline_train"))
    return mod


@step("2. gym / d4rl env id")
def s2():
    import gym
    import d4rl  # noqa: F401
    env = gym.make(ENV_ID)
    print("env type:", type(env))
    print("obs space:", env.observation_space, "act space:", env.action_space)
    obs = env.reset()
    print("reset obs:", obs.shape, obs.dtype)
    env.close()
    return True


@step("3. compile_config + dataset + env settings")
def s3():
    from ding.config import compile_config
    from ding.utils.data import create_dataset
    from ding.envs import get_vec_env_setting

    main_config, create_config = build_cfg()
    cfg = compile_config(main_config, seed=0, env=None, auto=True, create_cfg=create_config, save_cfg=True)
    # Mirror serial_pipeline_offline2online's own line: it appends '_command' after compiling,
    # and imports the command-mode policy instances so the 'command' field can be enabled.
    cfg.policy.type = cfg.policy.type + "_command"
    cfg.policy.import_names = ["ding.policy.edac", "ding.policy.command_mode_policy_instance"]
    print("compiled: env_id =", cfg.env.env_id, "| env.type =", cfg.env.type,
          "| import_names =", cfg.env.import_names)
    print("policy.type =", cfg.policy.type, "| random_collect_size =", cfg.policy.random_collect_size,
          "| update_per_collect =", cfg.policy.learn.update_per_collect,
          "| actor_update_freq =", cfg.policy.learn.actor_update_freq,
          "| concat_online_ratio =", cfg.policy.learn.concat_online_ratio)
    print("eval_freq =", cfg.policy.eval.evaluator.eval_freq,
          "| n_evaluator_episode =", cfg.policy.eval.evaluator.get("n_evaluator_episode"))

    dataset = create_dataset(cfg)
    print("dataset len =", len(dataset), "| keys:", sorted(dataset[0].keys()))
    d0 = dataset[0]
    print("obs shape", tuple(d0["obs"].shape), "action shape", tuple(d0["action"].shape))
    print("done dtype/unique:", d0["done"].dtype if hasattr(d0["done"], "dtype") else type(d0["done"]))

    env_fn, collector_cfg, evaluator_cfg = get_vec_env_setting(cfg.env)
    print("env_fn =", env_fn)
    print("collector env_id:", collector_cfg[0].get("env_id"),
          "| evaluator n_episode:", evaluator_cfg[0].get("n_evaluator_episode"))
    return cfg, dataset, env_fn, collector_cfg


@step("4. build env manager + policy + learner; LOAD CHECKPOINT")
def s4(cfg, dataset, env_fn, collector_cfg):
    import torch
    from functools import partial
    from tensorboardX import SummaryWriter
    from ding.envs import create_env_manager
    from ding.policy import create_policy
    from ding.worker import BaseLearner

    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_cfg])
    info = collector_env.env_info()
    print("env_info obs", info.obs_space.shape, "act", info.act_space.shape, info.act_space.value)

    policy = create_policy(cfg.policy, model=None, enable_field=["learn", "collect", "eval", "command"])
    # For EDAC the registry returns the concrete EDACPolicy itself (not a mode-accessor
    # namedtuple), because `edac` is registered without a `default_model` wrapper.
    print("policy class:", type(policy).__name__)
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode,
                          SummaryWriter(os.path.join("./{}/log/".format(cfg.exp_name), "preflight")),
                          exp_name=cfg.exp_name)

    em = policy._learn_model
    actor0_before = em.actor[0].weight.detach().clone()
    critic0_before = em.critic.fcs[0].W.detach().clone()

    print("loaded ckpt path:", cfg.policy.learn.learner.hook.load_ckpt_before_run)
    learner.call_hook("before_run")

    raw = torch.load(CKPT, map_location="cpu", weights_only=False)
    ref_actor0 = raw["trainer/policy"]["fc0.weight"].float()
    ref_critic0 = raw["trainer/qfs"]["fc0.W"].float()
    got_actor0 = em.actor[0].weight.detach().cpu().float()
    got_critic0 = em.critic.fcs[0].W.detach().cpu().float()
    print("raw ckpt actor fc0.weight mean:", float(ref_actor0.mean()))
    print("policy  actor fc0.weight mean:", float(got_actor0.mean()))
    print("raw ckpt critic fc0.W mean:", float(ref_critic0.mean()))
    print("policy  critic fc0.W mean:", float(got_critic0.mean()))
    print("actor  exact match:", bool(torch.equal(ref_actor0, got_actor0)))
    print("critic exact match:", bool(torch.equal(ref_critic0, got_critic0)))
    print("actor changed vs pre-load init:", not bool(torch.equal(actor0_before.cpu(), got_actor0)))
    print("critic changed vs pre-load init:", not bool(torch.equal(critic0_before.cpu(), got_critic0)))
    return policy, learner, collector_env, raw


@step("5. one real EDAC update, finite / weight-change check")
def s5(policy, learner, dataset):
    import torch
    import random
    batch_size = 256
    data = random.sample(dataset.data, batch_size)
    em = policy._learn_model
    before = em.critic.fcs[0].W.detach().clone()
    before_actor = em.actor[0].weight.detach().clone()
    policy._cfg.learn.online = True
    log = policy._forward_learn(data)
    delta_c = float((em.critic.fcs[0].W.detach() - before).abs().max())
    delta_a = float((em.actor[0].weight.detach() - before_actor).abs().max())
    print("learn output keys:", sorted(log.keys()))
    print("critic_loss =", log.get("critic_loss"))
    print("q_value mean =", log.get("q_value"), "| target_q_value mean =", log.get("target_q_value"))
    print("alpha =", log.get("alpha"), "| td_error =", log.get("td_error"))
    print("max |d critic.fcs[0].W| =", delta_c)
    print("max |d actor[0].weight|  =", delta_a, "(0 is expected: actor updates only every 10 iters)")
    assert delta_c > 0, "critic weights did not change -> no real update happened"
    for k in ("critic_loss", "q_value", "target_q_value", "td_error", "alpha"):
        v = log.get(k)
        if v is not None:
            assert torch.isfinite(torch.tensor(float(v))), f"{k} not finite"
    return log


def main():
    try:
        s1()
        s2()
        cfg, dataset, env_fn, collector_cfg = s3()
        policy, learner, collector_env, raw = s4(cfg, dataset, env_fn, collector_cfg)
        s5(policy, learner, dataset)
    except Exception:
        print("\n[preflight] FAILED", flush=True)
        traceback.print_exc()
        return 1
    print("\n[preflight] ALL STEPS PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
