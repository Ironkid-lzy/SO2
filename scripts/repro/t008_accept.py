"""Machine-readable technical acceptance for one SO2 run."""
import argparse
import json
import math
import re
from pathlib import Path
import sys


def check(run_dir, target, log_path=None):
    errors = []
    def expect(ok, message):
        if not ok:
            errors.append(message)
    meta = json.loads((run_dir / "run_meta.json").read_text())
    rows = [json.loads(s) for s in (run_dir / "metrics.jsonl").read_text().splitlines()]
    evaluations = [r for r in rows if r.get("event") == "evaluation"]
    states = [r for r in rows if r.get("event") == "final"]
    reload = json.loads((run_dir / "policy_reload.json").read_text())
    steps = list(range(0, target + 1, 2500))
    expect(meta["status"] == "completed", "runner_status")
    expect(len(states) == 1, "final_state_count")
    expect(len(evaluations) == len(steps), "evaluation_count")
    expect([e["env_steps"] for e in evaluations] == steps, "evaluation_steps")
    for e in evaluations:
        expect(e.get("evaluation_env_steps", 0) > 0, f"eval_interactions_{e['env_steps']}")
        expect(e["episodes_requested"] == e["episodes_completed"] == len(e["returns"]) == 20,
               f"episodes_{e['env_steps']}")
        expect(all(math.isfinite(float(x)) for x in e["returns"]), f"finite_returns_{e['env_steps']}")
        expect(all(math.isfinite(float(e[k])) for k in ("return_mean", "normalized_score", "normalized_score_100")),
               f"finite_scores_{e['env_steps']}")
        expect(abs(e["normalized_score_100"] - 100 * e["normalized_score"]) < 1e-8,
               f"score_scale_{e['env_steps']}")
    if states:
        s = states[0]
        online = target - 5000
        expect(s["env_steps"] == s["collected_samples"] == target, "training_steps")
        expect(s["updates"] == {"critic": 10 * online, "actor": online, "alpha": 0}, "updates")
        expect(s["learner_train_iter"] == 10 * online, "learner_train_iter")
        expect(s["online_buffer"]["push_count"] == target, "online_push")
        expect(s["mixed_buffer"]["push_count"] == 201798 + online, "mixed_push")
        expect(s["mixed_buffer"]["evictions"] == online, "mixed_evictions")
        expect(s["batch_size"] == 2560 and s["online_batch"] == 256 and s["mixed_buffer_batch"] == 2304,
               "batch_shape")
        expect(s["auto_alpha"] is False, "auto_alpha")
        expect(s.get("evaluation_interactions_total") == sum(e.get("evaluation_env_steps", 0) for e in evaluations),
               "evaluation_interactions_total")
    if log_path is not None:
        log = log_path.read_text(errors="replace")
        value_lines = [line for line in log.splitlines() if line.startswith("| Value |")]
        expect(not any(re.search(r"\b(?:nan|inf|infinity)\b", line, re.I) for line in value_lines), "nonfinite_training_log")
        expect("out of memory" not in log.lower(), "oom_log")
    expect(reload["step"] == target and reload["equal_live_after"] and reload["equal_reloaded"], "policy_reload")
    expect(Path(reload["checkpoint_path"]).exists(), "checkpoint_missing")
    out = {"accepted": not errors, "errors": errors, "target_env_steps": target,
           "seed": meta["seed"], "git_commit": meta["git_commit"],
           "config_sha256": meta["config_sha256"], "final_state": states[0] if states else None,
           "evaluation_count": len(evaluations)}
    (run_dir / "acceptance.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--target", type=int, required=True)
    ap.add_argument("--log", type=Path)
    args = ap.parse_args()
    result = check(args.run_dir, args.target, args.log)
    print(json.dumps(result))
    sys.exit(0 if result["accepted"] else 1)
