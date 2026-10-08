"""Durable, serial EXP-019 queue for T008 seeds 1–4; no retries."""
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
RESEARCH = Path("/home/lzy/Projects/rl_sample_efficiency_research_t008")
WORK = ROOT / "_so2_work/runs/EXP-019"
CONFIG = ROOT / "scripts/repro/configs/halfcheetah_medium_replay_t008.py"
PREFLIGHT = ROOT / "_so2_work/validation/EXP-019/preflight.json"
PYTHON = Path("/home/lzy/Projects/SO2/.venv/bin/python")
STATE = WORK / "queue_state.json"
GLOBAL_LOG = ROOT / "_so2_work/t008_queue.log"
CHECKPOINT = Path("/home/lzy/Projects/SO2/_so2_work/assets/checkpoints/ckpt/halfcheetah-medium-replay-v2.ckpt")
DATASET = Path("/home/lzy/.d4rl/datasets/halfcheetah_medium_replay-v2.hdf5")
EXPECTED_CHECKPOINT = "02d74b7860620f1f9863a2f217aff0abfaae37f4a69ba6d0e1a0ea10f98d94f7"
EXPECTED_DATASET = "48d494a4770c11f48260736dec090b78bdca8375647ed4d24bfa5b6610c7f683"
SEEDS = (0, 1, 2, 3, 4)
MAX_SEED_SEC = 8 * 3600
MAX_TOTAL_SEC = 40 * 3600
LOCK = ROOT / "_so2_work/EXP-019.queue.lock"
TARGET_STEPS = 100000


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def load_rows(path):
    rows = []
    if path.exists():
        for line in path.read_text(errors="replace").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def deliver(state, partial=False):
    cmd = [str(PYTHON), str(ROOT / "scripts/repro/t008_deliver.py"),
           "--state", str(STATE)]
    if partial:
        cmd.append("--partial")
    proc = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)
    (ROOT / "_so2_work/t008_delivery.log").write_text(proc.stdout)
    if proc.returncode:
        state["delivery_error"] = proc.stdout[-4000:]
        save(STATE, state)
        raise RuntimeError("research delivery failed")
    lines = [line for line in proc.stdout.splitlines() if line.startswith("{")]
    if lines:
        state["research_sha"] = json.loads(lines[-1]).get("research_sha")


def run_seed(state, seed, total_deadline):
    run = WORK / f"EXP-019-s{seed}-attempt1"
    assert not run.exists(), f"refusing to overwrite {run}"
    log_path = WORK / f"seed{seed}.log"
    cmd = [str(PYTHON), str(ROOT / "scripts/repro/t008_run.py"),
           "--config", str(CONFIG), "--seed", str(seed),
           "--max-env-steps", str(TARGET_STEPS),
           "--max-wall-clock-sec", str(MAX_SEED_SEC),
           "--run-dir", str(run)]
    entry = {"seed": seed, "status": "running", "run_dir": str(run),
             "log": str(log_path), "command": cmd, "start_unix": time.time()}
    state["current_seed"] = seed
    state.setdefault("seeds", []).append(entry)
    save(STATE, state)
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=os.environ.copy(),
                                stdout=log, stderr=subprocess.STDOUT)
        entry["pid"] = proc.pid
        save(STATE, state)
        start = time.monotonic()
        checked = False
        while proc.poll() is None:
            rows = load_rows(run / "metrics.jsonl")
            first = next((r for r in rows if r.get("event") == "evaluation_state"
                          and r.get("env_steps") in (7500, 10000)), None)
            if first and not checked:
                step = int(first["env_steps"])
                eval_row = next((r for r in rows if r.get("event") == "evaluation"
                                 and r.get("env_steps") == step), None)
                checkpoint = run / "ckpt" / f"envstep_{step}.pth.tar"
                updates = first.get("updates", {})
                expected = {"critic": 10 * (step - 5000),
                            "actor": step - 5000, "alpha": 0}
                finite_eval = (eval_row is not None
                               and len(eval_row.get("returns", [])) == 20
                               and all(math.isfinite(float(x)) for x in eval_row["returns"]))
                passed = updates == expected and finite_eval and checkpoint.exists()
                check = {"seed": seed, "step": step, "updates": updates,
                         "expected_updates": expected, "finite_20_episode_eval": finite_eval,
                         "checkpoint_exists": checkpoint.exists(),
                         "normalized_score_100": eval_row.get("normalized_score_100") if eval_row else None,
                         "passed": passed, "checked_unix": time.time()}
                entry["first_learning_check"] = check
                save(WORK / f"seed{seed}_first_learning_check.json", check)
                save(STATE, state)
                if not passed:
                    proc.terminate()
                    proc.wait(timeout=30)
                    entry["status"] = "failed"
                    entry["exit_code"] = proc.returncode
                    entry["error"] = "first learning window technical check failed"
                    entry["end_unix"] = time.time()
                    state["status"] = "partial"
                    state["error"] = entry["error"]
                    state["current_seed"] = seed
                    save(STATE, state)
                    raise RuntimeError(entry["error"])
                checked = True
            progress = next((r for r in reversed(rows) if r.get("event") == "evaluation_state"), None)
            if progress:
                state["progress"] = {"seed": seed, "env_steps": progress["env_steps"],
                                     "updates": progress.get("updates"), "checked_unix": time.time()}
                save(STATE, state)
            if time.monotonic() > min(start + MAX_SEED_SEC, total_deadline):
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                entry["status"] = "partial"
                entry["error"] = "8-hour per-seed wall-clock limit reached"
                entry["end_unix"] = time.time()
                state["status"] = "partial"
                state["error"] = entry["error"]
                save(STATE, state)
                raise RuntimeError(entry["error"])
            time.sleep(5)
        code = proc.wait()
    elapsed = time.monotonic() - start
    entry["exit_code"] = code
    entry["wall_sec"] = elapsed
    entry["end_unix"] = time.time()
    entry["status"] = "completed" if code == 0 else "failed"
    save(STATE, state)
    if code != 0:
        state["status"] = "partial"
        state["error"] = f"seed {seed} training exited {code}"
        save(STATE, state)
        raise RuntimeError(state["error"])
    if not checked:
        state["status"] = "partial"
        state["error"] = f"seed {seed} ended before first learning-window check"
        save(STATE, state)
        raise RuntimeError(state["error"])
    accept_cmd = [str(PYTHON), str(ROOT / "scripts/repro/t008_accept.py"),
                  str(run), "--target", str(TARGET_STEPS), "--log", str(log_path)]
    accept = subprocess.run(accept_cmd, cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    (WORK / f"seed{seed}_acceptance.log").write_text(accept.stdout)
    if accept.returncode:
        state["status"] = "partial"
        state["error"] = f"seed {seed} acceptance failed"
        entry["status"] = "failed"
        save(STATE, state)
        raise RuntimeError(state["error"])
    meta = json.loads((run / "run_meta.json").read_text())
    state_file = json.loads((run / "acceptance.json").read_text())
    if (meta.get("git_commit") != state["code_sha"]
            or meta.get("config_sha256") != state["config_sha256"]
            or meta.get("checkpoint_sha256") != EXPECTED_CHECKPOINT
            or meta.get("final_env_steps") != TARGET_STEPS
            or not state_file.get("accepted")):
        state["status"] = "partial"
        state["error"] = f"seed {seed} final identity/count audit failed"
        entry["status"] = "failed"
        save(STATE, state)
        raise RuntimeError(state["error"])
    entry["status"] = "accepted"
    entry["acceptance"] = str(run / "acceptance.json")
    state["completed_env_steps"] = sum(
        int(json.loads((WORK / f"EXP-019-s{s}-attempt1/acceptance.json").read_text())["final_state"]["env_steps"])
        for s in SEEDS if (WORK / f"EXP-019-s{s}-attempt1/acceptance.json").exists())
    state["total_train_wall_sec"] = sum(float(item.get("wall_sec", 0))
                                        for item in state["seeds"])
    save(STATE, state)


def main():
    if STATE.exists() or LOCK.exists():
        raise RuntimeError(f"queue state/lock already exists; refusing duplicate start: {STATE} {LOCK}")
    if not (ROOT / "scripts/repro/t008_deliver.py").exists():
        raise RuntimeError("delivery script missing")
    preflight = json.loads(PREFLIGHT.read_text())
    if preflight.get("status") != "passed":
        raise RuntimeError("preflight did not pass")
    if digest(CHECKPOINT) != EXPECTED_CHECKPOINT or digest(DATASET) != EXPECTED_DATASET:
        raise RuntimeError("frozen checkpoint or D4RL dataset hash mismatch")
    code_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    if dirty:
        raise RuntimeError("algorithm worktree is dirty; code/config must be frozen and pushed")
    if not subprocess.check_output(["git", "status", "--porcelain"], cwd=RESEARCH, text=True).strip() == "":
        raise RuntimeError("delivery worktree is dirty; frozen registration must be committed")
    WORK.parent.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=False)
    fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    state = {"status": "running", "exp_id": "EXP-019", "seeds_expected": list(SEEDS),
             "seeds_completed": [], "code_sha": code_sha,
             "config_path": str(CONFIG), "config_sha256": digest(CONFIG),
             "preflight_path": str(PREFLIGHT), "preflight_sha256": digest(PREFLIGHT),
             "checkpoint_path": str(CHECKPOINT), "checkpoint_sha256": digest(CHECKPOINT),
             "dataset_path": str(DATASET), "dataset_sha256": digest(DATASET),
             "started_unix": time.time(), "host": os.uname().nodename,
             "research_branch": "codex/so2-t008-delivery"}
    save(STATE, state)
    try:
        train_start = time.monotonic()
        total_deadline = train_start + MAX_TOTAL_SEC
        for seed in SEEDS:
            if time.monotonic() > total_deadline:
                raise RuntimeError("40-hour total training wall-clock limit reached")
            run_seed(state, seed, total_deadline)
            state["seeds_completed"].append(seed)
            state["current_seed"] = None
            state["total_train_wall_sec"] = sum(float(item.get("wall_sec", 0))
                                                for item in state["seeds"])
            save(STATE, state)
        state["status"] = "completed"
        state["finished_unix"] = time.time()
        save(STATE, state)
        deliver(state)
        state["research_sha"] = state.get("research_sha")
        save(STATE, state)
    except BaseException as exc:
        if state.get("status") == "running":
            state["status"] = "partial"
        state["error"] = repr(exc)
        state["traceback"] = traceback.format_exc()
        state["finished_unix"] = time.time()
        save(STATE, state)
        try:
            deliver(state, partial=True)
        except BaseException as deliver_exc:
            state["delivery_error"] = repr(deliver_exc)
            save(STATE, state)
        raise


if __name__ == "__main__":
    main()
