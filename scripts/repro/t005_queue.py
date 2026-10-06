"""One-attempt durable EXP-016 queue and post-run evidence delivery."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
RESEARCH = Path("/home/lzy/Projects/rl_sample_efficiency_research_t005")
WORK = ROOT / "_so2_work/runs/EXP-016"
RUN = WORK / "EXP-016-s0-attempt1"
CONFIG = ROOT / "scripts/repro/configs/halfcheetah_medium_replay_t005.py"
PYTHON = Path("/home/lzy/Projects/SO2/.venv/bin/python")
STATE = WORK / "queue_state.json"
LOG = WORK / "train.log"

def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()

def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)

def git(*args, cwd):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()

def deliver(state):
    raw = RESEARCH / "results/raw/EXP-016/EXP-016-s0-attempt1"
    raw.mkdir(parents=True, exist_ok=False)
    for name in ("metrics.jsonl", "run_meta.json", "policy_reload.json",
                 "acceptance.json", "live_policy_probes.jsonl",
                 "formatted_total_config.py", "total_config.py"):
        src = RUN / name
        if src.exists():
            shutil.copy2(src, raw / name)
    artifacts = [LOG, *sorted((RUN / "ckpt").glob("*.pth.tar"))]
    save(raw / "artifact_manifest.json", [
        {"path": str(p.resolve()), "size_bytes": p.stat().st_size,
         "sha256": digest(p)} for p in artifacts])
    save(raw / "queue_state.json", state)
    rows = [json.loads(s) for s in (raw / "metrics.jsonl").read_text().splitlines()]
    ev = [r for r in rows if r.get("event") == "evaluation"]
    x = [r["env_steps"] for r in ev]
    y = [r["normalized_score_100"] for r in ev]
    assert x == list(range(0, 100001, 2500)) and all(math.isfinite(v) for v in y)
    auc = sum((y[i] + y[i + 1]) * 1250 for i in range(40)) / 100000
    base_rows = [json.loads(s) for s in (
        RESEARCH / "results/raw/EXP-015/EXP-015-s0/metrics.jsonl").read_text().splitlines()]
    base_ev = [r for r in base_rows if r.get("event") == "evaluation"]
    assert [r["env_steps"] for r in base_ev] == x
    a = [r["normalized_score_100"] for r in base_ev]
    baseline_auc = sum((a[i] + a[i + 1]) * 1250 for i in range(40)) / 100000
    summary = {"exp_id": "EXP-016", "seed": 0, "steps": x,
               "baseline_seed0": a, "droq_style_seed0": y,
               "baseline_endpoint": a[-1], "droq_style_endpoint": y[-1],
               "baseline_auc": baseline_auc, "droq_style_auc": auc,
               "endpoint_delta": y[-1] - a[-1], "auc_delta": auc - baseline_auc,
               "code_sha": state["code_sha"], "config_sha256": state["config_sha256"],
               "wall_sec": state["wall_sec"]}
    save(RESEARCH / "results/processed/EXP-016.json", summary)
    with open(RESEARCH / "results/processed/EXP-016.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["env_steps", "EXP-015_seed0", "EXP-016_seed0"])
        w.writerows(zip(x, a, y))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, a, label="A: SO2 EXP-015 seed 0")
    ax.plot(x, y, label="D: SO2 + DroQ-style critics EXP-016 seed 0")
    ax.axvspan(0, 5000, color="orange", alpha=0.1, label="warmup")
    ax.set(xlabel="Total training environment steps",
           ylabel="D4RL normalized score × 100")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESEARCH / "results/figures/EXP-016.png", dpi=180)
    plt.close(fig)
    report = RESEARCH / "reports/so2/SO2-T005.md"
    report.write_text(report.read_text().replace("状态：running", "状态：completed", 1))
    with open(report, "a") as f:
        f.write(f"""

## 正式结果（EXP-016 seed 0）

状态：技术验收通过，单次连续100k进程结束；科研解释待指挥端审查。
SO2执行SHA：`{state['code_sha']}`；冻结配置SHA256：`{state['config_sha256']}`。
实际进程墙钟：{state['wall_sec'] / 3600:.3f}小时。

| 指标 | A：EXP-015 seed0 | D：EXP-016 seed0 | D−A |
|---|---:|---:|---:|
| 100k normalized score ×100 | {a[-1]:.6f} | {y[-1]:.6f} | {y[-1]-a[-1]:+.6f} |
| 0–100k AUC | {baseline_auc:.6f} | {auc:.6f} | {auc-baseline_auc:+.6f} |

逐回合评估回报、实际计数和时间见 `results/raw/EXP-016/EXP-016-s0-attempt1/`；曲线为 `results/figures/EXP-016.png`，数值表为 `results/processed/EXP-016.csv`。大日志和checkpoint保留于 `{WORK}`，准确路径、大小、SHA256见原始证据的 `artifact_manifest.json`。D只有一个训练seed，A的五seed均值仅是背景；本次组合干预无法区分N、LN、Dropout的贡献。
""")
    tracker = RESEARCH / "docs/EXPERIMENT_TRACKER.md"
    with open(tracker, "a") as f:
        f.write(f"\nEXP-016 完成记录：seed0终点 {y[-1]:.6f}，AUC {auc:.6f}，技术验收通过；详见 reports/so2/SO2-T005.md。\n")
    log = RESEARCH / "docs/RESEARCH_LOG.md"
    with open(log, "a") as f:
        f.write(f"\n### 2026-10-06 EXP-016 单seed探索结果\nSO2 + DroQ-style critics seed0：100k终点 {y[-1]:.6f}，AUC {auc:.6f}；相对EXP-015 seed0分别为 {y[-1]-a[-1]:+.6f} 和 {auc-baseline_auc:+.6f}。这是组合干预的一次观察，不能作统计改善或单因素因果结论。见 reports/so2/SO2-T005.md。\n")
    git("add", "-f", "results/raw/EXP-016", cwd=RESEARCH)
    git("add", "reports/so2/SO2-T005.md",
        "results/processed/EXP-016.json", "results/processed/EXP-016.csv",
        "results/figures/EXP-016.png", "docs/EXPERIMENT_TRACKER.md",
        "docs/RESEARCH_LOG.md", cwd=RESEARCH)
    subprocess.check_call(["git", "commit", "-m", "EXP-016: deliver T005 single-seed result"],
                          cwd=RESEARCH)
    subprocess.check_call(["git", "push", "origin", "codex/so2-t005-delivery"],
                          cwd=RESEARCH)
    state["research_sha"] = git("rev-parse", "HEAD", cwd=RESEARCH)
    save(STATE, state)

def main():
    WORK.mkdir(parents=True, exist_ok=False)
    code_sha = git("rev-parse", "HEAD", cwd=ROOT)
    assert not git("status", "--porcelain", "--untracked-files=no", cwd=ROOT)
    assert not git("status", "--porcelain", "--untracked-files=no", cwd=RESEARCH)
    assert RUN.is_relative_to(ROOT)
    cmd = [str(PYTHON), "scripts/repro/t005_run.py", "--config", str(CONFIG),
           "--seed", "0", "--max-env-steps", "100000",
           "--max-wall-clock-sec", "28800", "--run-dir", str(RUN)]
    state = {"status": "running", "pid": os.getpid(), "start_unix": time.time(),
             "code_sha": code_sha, "config_sha256": digest(CONFIG),
             "command": cmd, "run_dir": str(RUN), "log": str(LOG),
             "research_branch": "codex/so2-t005-delivery"}
    save(STATE, state)
    try:
        with open(LOG, "w") as log:
            proc = subprocess.Popen(cmd, cwd=ROOT, env=os.environ.copy(),
                                    stdout=log, stderr=subprocess.STDOUT)
            state["train_pid"] = proc.pid
            save(STATE, state)
            deadline = time.monotonic() + 28830
            checked = False
            while proc.poll() is None:
                if not checked and (RUN / "metrics.jsonl").exists():
                    rows = []
                    for line in (RUN / "metrics.jsonl").read_text().splitlines():
                        try:
                            rows.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
                    sample = next((r for r in rows if r.get("event") == "evaluation_state"
                                   and r.get("env_steps") in (7500, 10000)), None)
                    if sample:
                        step = sample["env_steps"]
                        expected = 10 * (step - 5000)
                        eval_row = next((r for r in rows if r.get("event") == "evaluation" and r.get("env_steps") == step), None)
                        finite_eval = eval_row is not None and len(eval_row.get("returns", [])) == 20 and all(math.isfinite(float(v)) for v in eval_row["returns"])
                        ok = finite_eval and (sample["updates"] == {"critic": expected,
                            "actor": step - 5000, "alpha": 0}
                            and (RUN / "ckpt" / f"envstep_{step}.pth.tar").exists())
                        check = {"step": step, "updates": sample["updates"],
                                 "expected_critic_updates": expected,
                                 "checkpoint_exists": (RUN / "ckpt" / f"envstep_{step}.pth.tar").exists(),
                                 "finite_20_episode_eval": finite_eval,
                                 "normalized_score_100": eval_row["normalized_score_100"] if eval_row else None,
                                 "passed": ok, "checked_unix": time.time()}
                        save(WORK / "first_learning_check.json", check)
                        state["first_learning_check"] = check
                        save(STATE, state)
                        if not ok:
                            proc.terminate()
                            raise RuntimeError("first learning check failed")
                        checked = True
                if time.monotonic() > deadline:
                    proc.terminate()
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        proc.kill(); proc.wait()
                    raise RuntimeError("8-hour training limit reached")
                time.sleep(5)
            code = proc.wait()
        state["train_exit_code"] = code
        state["wall_sec"] = time.time() - state["start_unix"]
        save(STATE, state)
        if code:
            raise RuntimeError(f"training exited {code}")
        accept = subprocess.run([str(PYTHON), "scripts/repro/t003_accept.py",
            str(RUN), "--target", "100000", "--log", str(LOG)],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (WORK / "acceptance.log").write_text(accept.stdout)
        if accept.returncode:
            raise RuntimeError("technical acceptance failed")
        meta = json.loads((RUN / "run_meta.json").read_text())
        if meta["git_commit"] != code_sha or meta["config_sha256"] != state["config_sha256"]:
            raise RuntimeError("frozen code or config mismatch")
        state["status"] = "delivering"
        save(STATE, state)
        deliver(state)
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
