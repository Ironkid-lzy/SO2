"""Frozen SO2 actor-aggregation/dropout T009 seed runner."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "SO2"))
import d4rl  # noqa: F401
import torch


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-env-steps", type=int, required=True)
    parser.add_argument("--max-wall-clock-sec", type=float, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--aggregation", choices=("min", "mean"), required=True)
    parser.add_argument("--dropout-p", type=float, required=True)
    args = parser.parse_args()
    assert args.seed in (0, 1, 2, 3, 4)
    assert args.dropout_p in (0.0, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1)
    assert args.max_env_steps == 100000
    assert 0 < args.max_wall_clock_sec <= 28800
    assert Path.cwd().resolve() == ROOT
    assert not args.run_dir.exists(), f"run directory exists: {args.run_dir}"
    args.run_dir.mkdir(parents=True)
    assert Path(__import__("ding").__file__).resolve().is_relative_to(ROOT / "SO2")
    from ding.entry import serial_pipeline_offline2online
    from ding.config import compile_config
    from ding.policy import create_policy
    import ding.entry.serial_entry_offline2online as pipeline_module
    assert Path(pipeline_module.__file__).resolve().is_relative_to(ROOT / "SO2")
    assert Path(sys.modules["ding"].__file__).resolve().is_relative_to(ROOT / "SO2")
    frozen = runpy.run_path(str(args.config.resolve()))
    main_cfg, create_cfg = frozen["main_config"], frozen["create_config"]
    checkpoint = Path("/home/lzy/Projects/SO2/_so2_work/assets/checkpoints/ckpt/halfcheetah-medium-replay-v2.ckpt")
    assert checkpoint.exists()
    main_cfg.exp_name = str(args.run_dir.resolve().relative_to(ROOT))
    main_cfg.policy.learn.max_env_steps = args.max_env_steps
    main_cfg.policy.learn.max_wall_clock_sec = args.max_wall_clock_sec
    main_cfg.policy.learn.learner.load_path = str(checkpoint)
    main_cfg.policy.learn.learner.hook.load_ckpt_before_run = str(checkpoint)
    p, e = main_cfg.policy, main_cfg.env
    assert (e.env_id, p.model.critic_ensemble_size, p.learn.update_per_collect,
            p.learn.actor_update_freq, p.learn.batch_size, p.random_collect_size,
            p.learn.concat_online_ratio, p.learn.alpha, p.learn.auto_alpha) == (
            "halfcheetah-medium-replay-v2", 2, 10, 10, 256, 5000, 0.1, 0.2, False)
    p.model.critic_dropout_rate = args.dropout_p
    p.learn.actor_q_aggregation = args.aggregation
    assert p.model.critic_layer_norm is True and p.model.critic_dropout_rate == args.dropout_p
    assert p.learn.actor_q_aggregation == args.aggregation
    assert p.eval.evaluator.n_evaluator_episode == 20 and p.eval.evaluator.eval_freq == 2500
    assert p.other.replay_buffer.replay_buffer_size == 1000000
    assert p.learn.ignore_done is True and p.learn.noise is True
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    meta = dict(status="running", seed=args.seed, env_id=e.env_id, aggregation=args.aggregation,
                dropout_p=args.dropout_p, git_commit=commit,
                dataset_path="/home/lzy/.d4rl/datasets/halfcheetah_medium_replay-v2.hdf5",
                dataset_sha256=digest("/home/lzy/.d4rl/datasets/halfcheetah_medium_replay-v2.hdf5"),
                config_path=str(args.config.resolve()), config_sha256=digest(args.config),
                checkpoint_path=str(checkpoint.resolve()), checkpoint_sha256=digest(checkpoint),
                cwd=str(ROOT), ding_file=str(Path(sys.modules["ding"].__file__).resolve()),
                pipeline_file=str(Path(pipeline_module.__file__).resolve()),
                command=sys.argv, start_unix=time.time(), python=sys.version,
                torch=torch.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    (args.run_dir / "effective_config.json").write_text(json.dumps(main_cfg, indent=2, sort_keys=True, default=str) + "\n")
    write(args.run_dir / "run_meta.json", meta)
    try:
        policy, _ = serial_pipeline_offline2online(
            [main_cfg, create_cfg], seed=args.seed, max_env_steps=args.max_env_steps,
            max_wall_clock_sec=args.max_wall_clock_sec, eval_env_step_interval=2500,
            eval_n_episode=20, metrics_path=str(args.run_dir / "metrics.jsonl"),
            policy_probe_path=str(args.run_dir / "live_policy_probes.jsonl"))
        records = [json.loads(s) for s in (args.run_dir / "metrics.jsonl").read_text().splitlines()]
        final = next(r for r in reversed(records) if r["event"] == "final")
        step = final["env_steps"]
        probes = [json.loads(s) for s in (args.run_dir / "live_policy_probes.jsonl").read_text().splitlines()]
        probe = next(r for r in reversed(probes) if r["env_steps"] == step)
        saved = args.run_dir / "ckpt" / f"envstep_{step}.pth.tar"
        assert saved.exists()
        fresh = runpy.run_path(str(args.config.resolve()))
        cfg = compile_config(fresh["main_config"], seed=args.seed, env=None, auto=True,
                             create_cfg=fresh["create_config"], save_cfg=False)
        cfg.policy.model.critic_dropout_rate = args.dropout_p
        cfg.policy.learn.actor_q_aggregation = args.aggregation
        cfg.policy.type += "_command"
        cfg.policy.import_names = ["ding.policy.edac", "ding.policy.command_mode_policy_instance"]
        loaded = create_policy(cfg.policy, model=None, enable_field=["learn", "collect", "eval", "command"])
        loaded._load_state_dict_learn(torch.load(saved, map_location="cpu", weights_only=False))
        obs = torch.tensor(probe["observation"])
        before = torch.tensor(probe["live_action_before_save"])
        live = torch.as_tensor(policy._forward_eval({0: obs})[0]["action"]).cpu()
        reloaded = torch.as_tensor(loaded._forward_eval({0: obs})[0]["action"]).cpu()
        check = dict(step=step, live_before_save=before.tolist(), live_after_save=live.tolist(),
                     freshly_loaded=reloaded.tolist(),
                     equal_live_after=bool(torch.allclose(before, live, atol=1e-6, rtol=0)),
                     equal_reloaded=bool(torch.allclose(before, reloaded, atol=1e-6, rtol=0)),
                     checkpoint_path=str(saved.resolve()), checkpoint_size=saved.stat().st_size,
                     checkpoint_sha256=digest(saved))
        write(args.run_dir / "policy_reload.json", check)
        assert check["equal_live_after"] and check["equal_reloaded"]
        meta.update(status="completed" if step == args.max_env_steps else "partial",
                    end_unix=time.time(), final_env_steps=step,
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None)
        write(args.run_dir / "run_meta.json", meta)
        assert step == args.max_env_steps, f"partial at {step}"
    except BaseException as exc:
        meta.update(status="failed", end_unix=time.time(), error=repr(exc), traceback=traceback.format_exc())
        write(args.run_dir / "run_meta.json", meta)
        raise

if __name__ == "__main__":
    main()
