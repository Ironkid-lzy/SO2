"""Durable sequential five-seed queue; stops at the first technical failure."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
RESEARCH = Path("/home/lzy/Projects/rl_sample_efficiency_research_t003")
CONFIG = ROOT / "scripts/repro/configs/halfcheetah_medium_replay_full.py"
WORK = ROOT / "_so2_work/runs/EXP-015"
STATE = WORK / "queue_state.json"
PYTHON = Path("/home/lzy/Projects/SO2/.venv/bin/python")
TARGET = 100000
LIMIT = 45 * 3600


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, obj):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def evidence(run_dir, seed):
    dest = RESEARCH / "results/raw/EXP-015" / f"EXP-015-s{seed}"
    dest.mkdir(parents=True, exist_ok=False)
    small = ["metrics.jsonl", "run_meta.json", "policy_reload.json", "acceptance.json",
             "live_policy_probes.jsonl", "formatted_total_config.py", "total_config.py"]
    for name in small:
        source = run_dir / name
        if source.exists():
            shutil.copy2(source, dest / name)
    artifacts = []
    for file in [WORK / "logs" / f"seed{seed}.log", *sorted((run_dir / "ckpt").glob("*.pth.tar"))]:
        artifacts.append({"path": str(file.resolve()), "size_bytes": file.stat().st_size,
                          "sha256": digest(file)})
    save(dest / "artifact_manifest.json", artifacts)


def summarize():
    curves = []
    for seed in range(5):
        path = RESEARCH / "results/raw/EXP-015" / f"EXP-015-s{seed}" / "metrics.jsonl"
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        evals = [r for r in rows if r.get("event") == "evaluation"]
        curves.append([r["normalized_score_100"] for r in evals])
    xs = list(range(0, 100001, 2500))
    auc = [sum((y[i] + y[i+1]) / 2 * 2500 for i in range(40)) / 100000 for y in curves]
    endpoint = [y[-1] for y in curves]
    out = {"steps": xs, "per_seed_curves": curves, "per_seed_endpoint": endpoint,
           "endpoint_mean": statistics.mean(endpoint), "endpoint_sample_std": statistics.stdev(endpoint),
           "per_seed_auc": auc, "auc_mean": statistics.mean(auc),
           "auc_sample_std": statistics.stdev(auc)}
    processed = RESEARCH / "results/processed"
    processed.mkdir(parents=True, exist_ok=True)
    save(processed / "EXP-015.json", out)
    import csv
    with open(processed / "EXP-015.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["env_steps", *[f"seed_{s}" for s in range(5)], "mean", "sample_std"])
        for i, x in enumerate(xs):
            vals = [y[i] for y in curves]
            w.writerow([x, *vals, statistics.mean(vals), statistics.stdev(vals)])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        for seed, y in enumerate(curves):
            ax.plot(xs, y, alpha=.5, label=f"seed {seed}")
        mean = [statistics.mean([y[i] for y in curves]) for i in range(41)]
        sd = [statistics.stdev([y[i] for y in curves]) for i in range(41)]
        ax.plot(xs, mean, color="black", linewidth=2, label="mean")
        ax.fill_between(xs, [a-b for a,b in zip(mean,sd)], [a+b for a,b in zip(mean,sd)],
                        color="black", alpha=.13, label="sample std")
        ax.axvspan(0, 5000, color="orange", alpha=.1, label="warmup")
        ax.set(xlabel="Total training environment steps", ylabel="D4RL normalized score × 100")
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        figures = RESEARCH / "results/figures"
        figures.mkdir(parents=True, exist_ok=True)
        fig.savefig(figures / "EXP-015.png", dpi=180)
    except Exception as exc:
        out["plot_error"] = repr(exc)
        save(processed / "EXP-015.json", out)


def main():
    WORK.mkdir(parents=True, exist_ok=False)
    (WORK / "logs").mkdir()
    state = {"status": "running", "start_unix": time.time(), "pid": os.getpid(),
             "code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
             "config_sha256": digest(CONFIG), "seeds": {}, "attempts": [],
             "validation_wall_sec": float(os.environ.get("T003_VALIDATION_WALL_SEC", "0"))}
    save(STATE, state)
    try:
        for seed in range(5):
            used = state["validation_wall_sec"] + sum(a["wall_sec"] for a in state["attempts"])
            remaining = LIMIT - used
            if remaining <= 0:
                raise RuntimeError("45-hour cumulative process wall-clock limit reached")
            run_dir = WORK / f"EXP-015-s{seed}-attempt1"
            wall_limit = min(28800, remaining)
            command = [str(PYTHON), "scripts/repro/t003_run.py", "--config", str(CONFIG),
                       "--seed", str(seed), "--max-env-steps", str(TARGET),
                       "--max-wall-clock-sec", str(wall_limit), "--run-dir", str(run_dir)]
            log_path = WORK / "logs" / f"seed{seed}.log"
            state["seeds"][str(seed)] = {"status": "running", "run_dir": str(run_dir),
                                         "command": command, "start_unix": time.time(),
                                         "stdout_stderr": str(log_path)}
            save(STATE, state)
            started = time.monotonic()
            with open(log_path, "w") as log:
                proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                        env=os.environ.copy())
                state["seeds"][str(seed)]["pid"] = proc.pid
                save(STATE, state)
                exit_code = proc.wait()
            attempt = {"seed": seed, "run_dir": str(run_dir), "exit_code": exit_code,
                       "wall_sec": time.monotonic()-started, "end_unix": time.time()}
            state["attempts"].append(attempt)
            state["seeds"][str(seed)].update(attempt)
            if exit_code:
                state["seeds"][str(seed)]["status"] = "failed"
                save(STATE, state)
                raise RuntimeError(f"seed {seed} exited {exit_code}")
            accepted = subprocess.run([str(PYTHON), "scripts/repro/t003_accept.py", str(run_dir),
                                       "--target", str(TARGET)], cwd=ROOT,
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            (WORK / "logs" / f"seed{seed}_accept.log").write_text(accepted.stdout)
            if accepted.returncode:
                state["seeds"][str(seed)]["status"] = "failed_acceptance"
                save(STATE, state)
                raise RuntimeError(f"seed {seed} failed technical acceptance")
            evidence(run_dir, seed)
            state["seeds"][str(seed)]["status"] = "completed"
            save(STATE, state)
        summarize()
        state["status"] = "completed"
    except BaseException as exc:
        state["status"] = "blocked"
        state["error"] = repr(exc)
        state["traceback"] = traceback.format_exc()
        raise
    finally:
        state["end_unix"] = time.time()
        save(STATE, state)


if __name__ == "__main__":
    main()
