"""Unvalidated single-attempt T020 dispatch, finite supervisor and CPU sealer.

No model imports. All science runs as a bounded child only after the T019 barrier.
Canonical wall accounts the whole interval including loading/failures/final seal.
"""
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT/"_so2_work/t020"
RUN = WORK/"d0-v1"
PRIOR = Path("/home/lzy/Projects/SO2_t019/_so2_work/t019")
LOCK = Path("/home/lzy/Projects/SO2/_so2_work/training_resource.lock")
PY = "/home/lzy/Projects/SO2/.venv/bin/python"
JOB = "d0-v1"
LIMIT = 600


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    os.replace(temporary, path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def alive(pid):
    try:
        return Path("/proc/%d/stat" % pid).read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def group_alive(pgid):
    assert pgid > 1
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            try:
                fields = (path/"stat").read_text().rsplit(")", 1)[1].split()
            except (FileNotFoundError, ProcessLookupError):
                continue
            if fields[0] != "Z" and int(fields[2]) == pgid:
                return True
    return False


def gpu():
    return subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
                                    "--format=csv,noheader"], text=True).strip()


def terminate_own_group(pgid):
    assert pgid > 1 and pgid != os.getpgrp()
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if not group_alive(pgid):
            return
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        deadline = time.monotonic()+5
        while group_alive(pgid) and time.monotonic() < deadline:
            time.sleep(.1)
    assert not group_alive(pgid), "Own worker group did not become quiescent"


def prior_barrier():
    b = read(PRIOR/"budget.json")
    expected = ["identity-preflight-v1"]+[
        "s%d-%s-v1" % (seed, phase) for seed in range(1, 5)
        for phase in ("offline", "online")]
    assert [x["job"] for x in b["jobs"]] == expected
    assert all(x["status"] == "completed" and x["exit_code"] == 0 for x in b["jobs"])
    for seed in range(1, 5):
        for phase in ("offline", "online"):
            seal = read(PRIOR/("seed%d" % seed)/(phase+"-sealed.json"))
            dispatch = read(PRIOR/("seed%d" % seed)/(phase+"-dispatch.json"))
            assert seal["status"] == "completed" and seal["quiescent"]
            assert not alive(dispatch["supervisor_pid"]) and not alive(dispatch["sealer_pid"])
    assert all(not group_alive(x["pgid"]) for x in b["jobs"])
    assert not gpu()
    with LOCK.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)


def frozen():
    receipt = read(WORK/"formal-freeze.json")
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                   text=True).strip() == receipt["code_sha"]
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT,
                                       text=True).strip()
    for path, sha in receipt["source_sha256"].items():
        assert digest(ROOT/path) == sha, path
    assert digest(WORK/"protocol.json") == receipt["protocol_sha256"]
    assert read(WORK/"protocol.json")["frozen"] is True
    assert subprocess.check_output(["nvidia-smi", "--query-gpu=uuid",
        "--format=csv,noheader"], text=True).strip() == receipt["GPU_uuid"]
    return receipt


def dispatch():
    # This command itself is single-use. Existing state is never resumed/replaced.
    assert WORK.is_dir()
    with (WORK/"dispatch.lock").open("a+") as gate:
        fcntl.flock(gate, fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not RUN.exists() and not (WORK/"dispatch.json").exists()
        assert not (WORK/"budget.json").exists()
        started = time.time()
        receipt = frozen(); prior_barrier()
        permit = read(WORK/"permit-d0-v1.json")
        assert permit["job"] == JOB and permit["allowed"] and permit["ordinaryUsageAllowed"]
        assert 0 <= time.time()-permit["checked_unix"] <= 1800
        assert permit["primary_used"] < 90 and permit["weekly_used"] < 100
        assert shutil.disk_usage(ROOT).free > 12*1024**3
        assert time.time()-started < 10
        RUN.mkdir()
        budget = {"limit_seconds": LIMIT, "used_seconds": time.time()-started,
                  "job": JOB, "status": "admitted", "start_unix": started,
                  "automatic_retry": False, "environment_interactions": 0}
        save(WORK/"budget.json", budget)
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MUJOCO_GL="osmesa",
                   D4RL_SUPPRESS_IMPORT_ERROR="1",
                   LD_LIBRARY_PATH="/home/lzy/.mujoco/mujoco210/bin:/usr/lib/nvidia:/usr/lib/x86_64-linux-gnu:/home/lzy/Projects/SO2/.venv/lib/python3.10/site-packages/torch/lib")
        with (WORK/"supervisor.log").open("x") as log:
            process = subprocess.Popen([PY, "scripts/repro/t020_control.py", "supervise"],
                cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
        d = {"job": JOB, "started_unix": started, "supervisor_pid": process.pid,
             "quota_permit": permit, "freeze": receipt,
             "worker_deadline_unix": started+540, "final_seal_deadline_unix": started+590,
             "hard_deadline_unix": started+600}
        save(WORK/"dispatch.json", d)
        env["CUDA_VISIBLE_DEVICES"] = ""
        with (WORK/"sealer.log").open("x") as log:
            sealer = subprocess.Popen(["timeout", "--signal=KILL", "595", PY,
                "scripts/repro/t020_control.py", "seal"], cwd=ROOT, env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
        d["sealer_pid"] = sealer.pid; save(WORK/"dispatch.json", d)
        print(json.dumps(d))


def supervise():
    b = read(WORK/"budget.json"); end = b["start_unix"]+540
    assert 0 < end-time.time() < 540
    with LOCK.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not gpu()
        environment = os.environ.copy()
        environment["SO2_T020_MONOTONIC_DEADLINE"] = str(
            time.monotonic()+end-time.time())
        with (WORK/"worker.log").open("x") as log:
            child = subprocess.Popen(["timeout", "--signal=TERM", "--kill-after=5",
                str(end-time.time()), PY, "scripts/repro/t020_d0.py",
                "--protocol", str(WORK/"protocol.json"), "--output", str(RUN)],
                cwd=ROOT, env=environment, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
        b.update(status="running", worker_pid=child.pid, worker_pgid=child.pid,
                 supervisor_pid=os.getpid())
        save(WORK/"budget.json", b)
        try:
            while child.poll() is None:
                if time.time() > end+6:
                    terminate_own_group(child.pid)
                    break
                b["used_seconds"] = time.time()-b["start_unix"]
                save(WORK/"budget.json", b)
                time.sleep(.2)
        finally:
            terminate_own_group(child.pid)
            b.update(status="worker_exited", exit_code=child.poll(),
                     used_seconds=time.time()-b["start_unix"], quiescent=True)
            save(WORK/"budget.json", b)


def seal_inner():
    b = read(WORK/"budget.json")
    assert not group_alive(b["worker_pgid"])
    with LOCK.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not gpu()
        result = read(RUN/"result.json")
        assert result["environment_interactions"] == 0
        counts = result["optimizer_counts"]
        assert counts["critic"] <= 1000 and counts["actor"] <= 100
        status = "completed" if b.get("exit_code") == 0 and result["status"] in (
            "completed", "completed_inconclusive") else "partial"
        if status == "completed":
            assert read(RUN/"correctness.json")["passed"]
            assert read(RUN/"input-identity.json")["generated_before_gradient_collection"]
            assert read(RUN/"parameter-order.json")["written_before_gradients"]
        files = [p for p in sorted(RUN.rglob("*")) if p.is_file()]
        assert len(files) <= 100 and not any(p.is_symlink() for p in files)
        assert sum(p.stat().st_size for p in files) <= 2*1024**3
        hashes = {str(p.relative_to(RUN)): digest(p) for p in files}
        assert not group_alive(b["worker_pgid"])
        return {"status": status, "quiescent": True, "files_sha256": hashes,
                "interpretation": result["interpretation"], "optimizer_counts": counts,
                "environment_interactions": 0, "reusable_training_artifacts": False}


def seal():
    d = read(WORK/"dispatch.json"); failure = None
    while True:
        b = read(WORK/"budget.json")
        pgid = b.get("worker_pgid")
        if b["status"] == "worker_exited" and pgid and not group_alive(pgid):
            break
        if time.time() > d["started_unix"]+555 or not alive(d["supervisor_pid"]):
            failure = "supervisor_lost_or_deadline"
            if pgid:
                terminate_own_group(pgid)
            break
        time.sleep(.2)
    if alive(d["supervisor_pid"]):
        # Once the worker is gone, the supervisor may only be finishing its ledger.
        deadline = time.monotonic()+3
        while alive(d["supervisor_pid"]) and time.monotonic() < deadline:
            time.sleep(.1)
        if alive(d["supervisor_pid"]):
            os.kill(d["supervisor_pid"], signal.SIGKILL)
    b = read(WORK/"budget.json")
    result = {"status": "partial", "failure": failure,
              "quiescent": bool(b.get("worker_pgid")) and not group_alive(b["worker_pgid"])}
    if not failure and result["quiescent"]:
        try:
            remaining = min(20, d["started_unix"]+585-time.time())
            assert remaining > 0
            child = subprocess.run([PY, "scripts/repro/t020_control.py", "seal-inner"],
                cwd=ROOT, capture_output=True, text=True, timeout=remaining)
            assert child.returncode == 0, child.stderr
            result.update(json.loads(child.stdout))
        except BaseException as error:
            result.update(status="partial", failure=repr(error))
    used = time.time()-d["started_unix"]
    if used >= 590:
        result.update(status="partial", failure="closure_time_margin_exhausted")
    b.update(status=result["status"], used_seconds=used,
             accounting="Full admission-through-seal elapsed wall including failures",
             independent_sealer=True)
    result.update(job=JOB, sealed_unix=time.time(), budget_used_seconds=used,
                  code_sha=d["freeze"]["code_sha"],
                  protocol_sha256=d["freeze"]["protocol_sha256"])
    assert not (WORK/"sealed.json").exists()
    save(WORK/"budget.json", b); save(WORK/"sealed.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "dispatch":
        dispatch()
    elif mode == "supervise":
        supervise()
    elif mode == "seal":
        seal()
    elif mode == "seal-inner":
        print(json.dumps(seal_inner()))
    else:
        raise ValueError(mode)
