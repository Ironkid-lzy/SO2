"""Verified full-parent to weights-only bridge; zero-update real learner test."""
import argparse
import pickle
import subprocess
import time
from unittest.mock import patch

from t010_edac import *
from t010_validate import check
from t011_common import checked_parent, export_weights, make_policy

OFFLINE_SHA = "f5dacc442f73e94e085988600fa2ff8cb8ad95b0"
PARENT_SHA = "799d8c07aff11009790b00650276e4e39ec96cc6d63739bf8d80070c2e034267"
PARENT = ROOT / "_so2_work/runs/EXP-025/EXP-025-s0-attempt1/ckpt/update_3000000.pth"
PREVIOUS_QUEUE = ROOT / "_so2_work/t010-v2/queue_state.json"


def remaining_online_seconds():
    previous = json.loads(PREVIOUS_QUEUE.read_text())
    attempt = previous["stages"]["online"]
    assert attempt["exit_code"] == 1 and previous["status"] == "partial"
    used = attempt["wall_sec"]
    assert 10 <= used < 11
    return 28800 - used


def configure_load(cfg, path):
    cfg.policy.learn.learner.load_path = str(path)
    cfg.policy.learn.learner.hook.load_ckpt_before_run = str(path)


def validate(out, parent=PARENT, record=None, seed=0):
    import runpy
    from ding.config import compile_config
    from ding.worker import BaseLearner
    from ding.worker.learner import learner_hook
    from t011_common import CONFIG, RECIPE

    start = time.monotonic()
    torch.set_num_threads(2)
    ident = {"git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
             "recipe_sha256":digest(RECIPE), "config_sha256":digest(CONFIG),
             "dataset_sha256":DATA_SHA, "reference_sha":REFERENCE_SHA, "seed":seed, "actor_bounds":[-20,2],
             "git_dirty":bool(subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip()),
             "source_sha256":{name:digest(ROOT/"scripts/repro"/name) for name in
                              ("t010_loading.py","t010_run.py","t010_resume_online.py","t010_deliver.py")},
             "torch":torch.__version__, "cuda":torch.version.cuda}
    saved = checked_parent(parent, ident, record)
    try:
        learner_hook.read_file(str(parent))
    except pickle.UnpicklingError as exc:
        assert "numpy.core.multiarray._reconstruct" in str(exc)
        rejected = str(exc)
    else:
        raise AssertionError("full parent unexpectedly accepted by default loader")
    init = out / "online_init.pth"
    manifest = export_weights(saved, parent, init, ident)
    safe = torch.load(init, map_location="cpu", weights_only=True)
    assert set(safe) == {"model", "target_model"}
    policy = make_policy("cuda",seed=seed)
    baseline = make_policy("cuda",seed=seed)
    baseline.learn_mode.load_state_dict(saved)
    fresh = runpy.run_path(str(CONFIG))
    fresh["main_config"].policy.learn.actor_q_aggregation = "mean"
    configure_load(fresh["main_config"], init)
    cfg = compile_config(fresh["main_config"], seed=seed, env=None, auto=True,
                         create_cfg=fresh["create_config"], save_cfg=False)
    assert cfg.policy.learn.learner.load_path == cfg.policy.learn.learner.hook.load_ckpt_before_run == str(init)
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, exp_name=str(out.relative_to(ROOT)))
    try:
        with patch.object(learner_hook, "read_file", wraps=learner_hook.read_file) as reader, \
             patch.object(policy._learn_model, "load_state_dict", wraps=policy._learn_model.load_state_dict) as online_load, \
             patch.object(policy._target_model, "load_state_dict", wraps=policy._target_model.load_state_dict) as target_load:
            learner.call_hook("before_run")
            reader.assert_called_once_with(str(init))
            for call in (online_load, target_load):
                assert call.call_count == 1 and call.call_args.kwargs == {"strict": True}
        assert learner.train_iter == policy._forward_learn_cnt == 0
        hashes = {}
        for name, model in (("model", policy._model), ("target_model", policy._target_model)):
            assert set(model.state_dict()) == set(saved[name])
            for key, value in model.state_dict().items():
                assert torch.equal(value, saved[name][key]), (name, key)
            hashes[name] = tensor_hash(model.state_dict())
            assert hashes[name] == manifest["network_hashes"][name]
        obs = torch.linspace(-1, 1, 256*17, device="cuda").reshape(256,17)
        actions = torch.linspace(-1, 1, 256*6, device="cuda").reshape(256,6)
        errors = {}
        with torch.no_grad():
            for mode in ("eval", "train"):
                for name, ref, actual in (("online", baseline._model, policy._model),
                                          ("target", baseline._target_model, policy._target_model)):
                    ref.train(mode == "train"); actual.train(mode == "train")
                    rng = rng_state(); expected = q_values(ref, obs, actions)
                    restore_rng(rng); got = q_values(actual, obs, actions)
                    errors[f"{mode}_{name}_q"] = check(expected, got, name, tol={"atol":0,"rtol":0})
                    for label, a, b in zip(("mu", "sigma"), ref.forward(obs, mode="compute_actor")["logit"],
                                           actual.forward(obs, mode="compute_actor")["logit"]):
                        errors[f"{mode}_{name}_{label}"] = check(a, b, label, tol={"atol":0,"rtol":0})
                    rng = rng_state(); expected = sample_actor(ref.forward, obs)[0]
                    restore_rng(rng); got = sample_actor(actual.forward, obs)[0]
                    errors[f"{mode}_{name}_sample_action"] = check(expected, got, name, tol={"atol":0,"rtol":0})
        assert policy._forward_learn_cnt == 0
        result = {"status":"passed", "identity":ident, "initialization":manifest,
                  "actual_entry":"BaseLearner.call_hook(before_run) -> LoadCkptHook -> read_file -> policy.learn_mode.load_state_dict",
                  "full_parent_rejected":rejected, "default_reader_calls":1, "strict_calls":2,
                  "missing_keys":[], "unexpected_keys":[], "network_hashes":hashes, "errors":errors,
                  "fixed_batch_updates":0, "cumulative_preflight_updates":0, "train_env_steps":0,
                  "evaluation_env_steps":0, "wall_seconds":time.monotonic()-start,
                  "remaining_online_seconds":18000}
        atomic_json(out / "validation.json", result)
        print(json.dumps(result))
    finally:
        learner.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parent",type=Path,default=PARENT)
    parser.add_argument("--parent-record",type=Path)
    parser.add_argument("--seed",type=int,default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    validate(args.out.resolve(),args.parent,json.loads(args.parent_record.read_text()) if args.parent_record else None,args.seed)
