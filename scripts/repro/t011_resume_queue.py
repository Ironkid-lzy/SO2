"""User-authorized continuation of T011 seeds1-4; never rerun seed0."""
import copy
import fcntl
import os
import signal
import subprocess
import time
import traceback
from t011_common import *
import t011_queue as queue
import t011_deliver as delivery

HISTORY=ROOT/"_so2_work/t011-v1/queue_state.json"
WORK=ROOT/"_so2_work/t011-resume-v1"
STATE=WORK/"queue_state.json"
CHECK=ROOT/"_so2_work/validation/t011-resume-v1"
PLAN=(1,2,3,4)
TRAINING_SHA="87f0ba67db5abbf71529614bdabba70dab950074"

def checked_history():
    old=json.loads(HISTORY.read_text())
    assert old["status"]=="partial" and old["error"]=="KeyError('updates')"
    assert old["code_sha"]==TRAINING_SHA and list(old["stages"])==["s0"]
    try:os.kill(old["queue_pid"],0)
    except ProcessLookupError:pass
    else:raise AssertionError("old queue PID exists; investigate before continuing")
    run=run_path(0);raw=RESEARCH/"results/raw/EXP-029"/run.name
    audit=json.loads((RESEARCH/"reports/so2/SO2-T011/failure-audit.json").read_text())
    for name,h in audit["sealed_raw_hashes"].items():assert digest(raw/name)==h
    a=json.loads((run/"acceptance.json").read_text())
    assert a["accepted"] and not a["errors"] and a["final_state"]["env_steps"]==100000
    assert a["final_state"]["updates"]=={"critic":950000,"actor":95000,"alpha":0}
    assert old["stages"]["s0"]["exit_code"]==0
    assert 11068<old["stages"]["s0"]["wall_seconds"]<11069
    assert all(not run_path(seed).exists() for seed in PLAN)
    assert all(not (RESEARCH/"results/raw/EXP-029"/run_path(seed).name).exists() for seed in PLAN)
    for seed in range(5):parent_manifest(seed)
    return old,a

def build_state(sha):
    old,a=checked_history()
    stage=copy.deepcopy(old["stages"]["s0"])
    stage.update(status="completed",acceptance=a,reused_without_training=True,
                 historical_queue_status="partial",training_code_sha=TRAINING_SHA,
                 progress={k:a["final_state"][k] for k in ("env_steps","updates","evaluation_interactions_total")})
    return {"status":"running","code_sha":sha,"thread_id":THREAD,"queue_pid":os.getpid(),
            "started_unix":time.time(),"plan":list(PLAN),"stages":{"s0":stage},"auto_retry":False,
            "fixed_batch_updates":15,"reused_seed0_sha":TRAINING_SHA,"history_queue_path":str(HISTORY),
            "history_queue_sha256":digest(HISTORY),"user_authorization":"2026-10-15: continue seeds1-4",
            "training_limits":{"per_run_seconds":18000,"total_seconds":86400,"total_env_steps":500000,
                               "new_env_steps":400000},"queue_workdir":str(WORK)}

def run_remaining(state):
    for seed in PLAN:
        queue.watch(state,seed)
        delivery.seal(seed)
        delivery.deliver(state)
    state.update(status="completed",finished_unix=time.time(),current_seed=None)
    delivery.deliver(state)

def main():
    assert not WORK.exists(),"continuation queue exists; no automatic restart"
    sha=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    assert not subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip()
    v=json.loads((CHECK/"validation.json").read_text())
    assert v["status"]=="passed" and v["code_sha"]==sha
    state=build_state(sha)
    assert queue.used(state)==state["stages"]["s0"]["wall_seconds"]
    with LOCK.open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True).strip()
        assert shutil.disk_usage(ROOT).free>20*1024**3
        WORK.mkdir()
        lock.seek(0);lock.truncate();lock.write(str(os.getpid()));lock.flush()
        queue.WORK=WORK;queue.STATE=STATE;delivery.WORK=WORK
        atomic_json(STATE,state)
        def interrupt(signum,frame):raise RuntimeError(f"continuation signal {signum}; stop remaining")
        signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
        try:
            delivery.deliver(state)
            run_remaining(state)
        except BaseException as exc:
            state.update(status="partial",error=repr(exc),traceback=traceback.format_exc(),finished_unix=time.time())
            atomic_json(STATE,state)
            for seed in PLAN:
                if run_path(seed).exists() and not (RESEARCH/"results/raw/EXP-029"/run_path(seed).name).exists():delivery.seal(seed)
            try:delivery.deliver(state)
            except BaseException as error:state["delivery_error"]=repr(error);atomic_json(STATE,state)
            raise
        finally:
            lock.seek(0);lock.truncate();lock.flush()

if __name__=="__main__":main()
