"""SO2-T002 bounded protocol run; learning settings come from T001 build_config."""
import argparse
import json
import os
import time
from t001_smoke import build_config

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_env_steps", type=int, default=10000)
    ap.add_argument("--max_wall_clock_sec", type=float, default=1800.0)
    ap.add_argument("--exp_name", default="_so2_work/runs/t002_seed0")
    args = ap.parse_args()
    assert 5000 <= args.max_env_steps <= 10000
    assert 0 < args.max_wall_clock_sec <= 1800
    cfg_args = argparse.Namespace(
        seed=0, env_id="halfcheetah-medium-replay-v2",
        ckpt_path="_so2_work/assets/checkpoints/ckpt/halfcheetah-medium-replay-v2.ckpt",
        data_path=None, max_env_steps=args.max_env_steps,
        max_wall_clock_sec=args.max_wall_clock_sec, eval_freq=2500,
        eval_episodes=20, random_collect_size=5000,
        log_show_after_iter=1000, exp_name=args.exp_name)
    cfg = build_config(cfg_args)
    assert cfg[0].policy.learn.update_per_collect == 10
    assert cfg[0].policy.learn.actor_update_freq == 10
    assert cfg[0].policy.learn.concat_online_ratio == 0.1
    assert cfg[0].policy.learn.batch_size == 256
    assert cfg[0].policy.random_collect_size == 5000
    os.makedirs(args.exp_name, exist_ok=True)
    with open(os.path.join(args.exp_name, "run_config.json"), "w") as f:
        json.dump(vars(cfg_args), f, indent=2)
    from ding.entry import serial_pipeline_offline2online
    start = time.time()
    policy, stop = serial_pipeline_offline2online(
        cfg, seed=0, max_env_steps=args.max_env_steps,
        max_wall_clock_sec=args.max_wall_clock_sec,
        eval_env_step_interval=2500, eval_n_episode=20,
        metrics_path=os.path.join(args.exp_name, "metrics.jsonl"))
    print("[T002_DONE] " + json.dumps({"wall_total_sec": time.time()-start,
          "stop_value_reached": bool(stop)}), flush=True)

if __name__ == "__main__":
    main()
