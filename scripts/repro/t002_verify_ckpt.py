"""Verify that T002 saved checkpoints reload to the same deterministic policy output."""
import json
from pathlib import Path

import torch

from t001_smoke import build_config
from ding.config import compile_config
from ding.policy import create_policy
from argparse import Namespace

root = Path("_so2_work/runs/t002_seed0/ckpt")
a = torch.load(root / "envstep_10000.pth.tar", map_location="cpu", weights_only=False)
b = torch.load(root / "iteration_50000.pth.tar", map_location="cpu", weights_only=False)
keys = sorted(a["model"])
equal_weights = all(torch.equal(a["model"][k], b["model"][k]) for k in keys)
cfg_args = Namespace(seed=0, env_id="halfcheetah-medium-replay-v2",
    ckpt_path="_so2_work/assets/checkpoints/ckpt/halfcheetah-medium-replay-v2.ckpt",
    data_path=None, max_env_steps=10000, max_wall_clock_sec=1800,
    eval_freq=2500, eval_episodes=20, random_collect_size=5000,
    log_show_after_iter=1000, exp_name="_so2_work/runs/t002_reload")
main_cfg, create_cfg = build_config(cfg_args)
cfg = compile_config(main_cfg, seed=0, env=None, auto=True,
                     create_cfg=create_cfg, save_cfg=False)
cfg.policy.type += "_command"
cfg.policy.import_names = ["ding.policy.edac", "ding.policy.command_mode_policy_instance"]
outputs = []
for state in (a, b):
    policy = create_policy(cfg.policy, model=None,
        enable_field=["learn", "collect", "eval", "command"])
    policy._load_state_dict_learn(state)
    obs = torch.linspace(-1, 1, 17)
    result = policy._forward_eval({0: obs})[0]["action"]
    outputs.append(torch.as_tensor(result).cpu())
equal_output = torch.equal(outputs[0], outputs[1])
record = {"checkpoint_a": str(root / "envstep_10000.pth.tar"),
          "checkpoint_b": str(root / "iteration_50000.pth.tar"),
          "model_tensor_count": len(keys), "equal_weights": equal_weights,
          "equal_deterministic_action": equal_output,
          "action": outputs[0].tolist(),
          "note": "Checks two independently reloaded snapshots at the same final step; does not prove exact training resume."}
print(json.dumps(record, indent=2))
assert equal_weights and equal_output
