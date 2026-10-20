"""One locked, fail-closed T011 queue; bounded and non-retrying."""
import fcntl
import math
import os
import shutil
import signal
import subprocess
import time
import traceback
from t011_common import *
from t011_deliver import deliver,seal,rows,control_events
STATE=WORK/"queue_state.json"

def used(state):
    return sum(v["wall_seconds"] for v in state["stages"].values())

def budget_for(state):
    remaining=86400-used(state)
    assert remaining>60,"cumulative budget exhausted"
    return min(18000.,remaining)

def belongs(pid,root_pid):
    for _ in range(12):
        if pid==root_pid:return True
        try:
            status=Path(f"/proc/{pid}/status").read_text()
            pid=int(next(v.split()[1] for v in status.splitlines() if v.startswith("PPid:")))
        except (FileNotFoundError,StopIteration):return False
    return False

def watch(state,seed):
    run=run_path(seed); budget=budget_for(state);m=parent_manifest(seed)
    cmd=["/usr/bin/timeout","--signal=TERM","--kill-after=5s",f"{budget-10:.3f}s",
         PYTHON,str(ROOT/"scripts/repro/t011_run.py"),"--seed",str(seed),"--run-dir",str(run),
         "--parent",m["parent_path"],"--parent-record",str(VALIDATION/f"parent-s{seed}.json"),
         "--wall-seconds",str(budget-30)]
    entry={"seed":seed,"command":cmd,"start_unix":time.time(),"budget_seconds":budget,"status":"running"}
    state.update(current_seed=seed,active=entry);atomic_json(STATE,state)
    start=time.monotonic();child=None;first_seen=False
    try:
        with (WORK/f"s{seed}.log").open("x") as log:
            child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            entry["pid"]=child.pid;atomic_json(STATE,state)
            while child.poll() is None:
                assert shutil.disk_usage(ROOT).free>20*1024**3,"disk reserve exhausted"
                assert subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()==state["code_sha"]
                assert not subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip(),"running code dirty"
                pids=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True).strip()
                assert all(belongs(int(pid),child.pid) for pid in pids.splitlines()), "another compute process detected"
                entry["wall_seconds"]=time.monotonic()-start
                if (run/"run_meta.json").exists():
                    meta=json.loads((run/"run_meta.json").read_text())
                    assert meta["git_commit"]==state["code_sha"] and meta["aggregation"]=="mean"
                    assert meta["checkpoint_sha256"]==m["parent_sha256"] and meta["seed"]==seed
                records=rows(run/"metrics.jsonl")
                evaluations=[e for e in records if e.get("event")=="evaluation"]
                states=[e for e in records if e.get("event")=="evaluation_state"]
                for e in evaluations:
                    assert len(e["returns"])==20 and all(math.isfinite(v) for v in e["returns"])
                for e in states:
                    c=max(0,e["env_steps"]-5000)*10
                    assert e["updates"]=={"critic":c,"actor":c//10,"alpha":0}
                if states:
                    entry["progress"]={k:states[-1][k] for k in ("env_steps","updates","evaluation_interactions_total")}
                if not first_seen and any(e["env_steps"]==7500 for e in states):
                    e=next(e for e in states if e["env_steps"]==7500)
                    returns=next(e for e in evaluations if e["env_steps"]==7500)["returns"]
                    ckpt=run/"ckpt/envstep_7500.pth.tar";assert ckpt.exists()
                    initial=evaluations[0];old=control_events(seed)[0]
                    # Initial score equality is a loading/RNG diagnostic, never a reward-selection gate.
                    first={"passed":True,"env_steps":7500,"updates":e["updates"],"returns_count":len(returns),
                           "returns_finite":True,"checkpoint_sha256":digest(ckpt),
                           "initial_returns_equal_control":initial["returns"]==old["returns"],
                           "initial_score_delta":initial["normalized_score_100"]-old["normalized_score_100"]}
                    atomic_json(run/"first_window.json",first);first_seen=True
                atomic_json(STATE,state)
                time.sleep(5)
            assert child.returncode==0,f"seed{seed} exited {child.returncode}; no retry"
            a=json.loads((run/"acceptance.json").read_text())
            assert a["accepted"] and first_seen
            assert a["final_state"]["updates"]=={"critic":950000,"actor":95000,"alpha":0}
            entry.update(status="completed",acceptance=a)
    except BaseException:
        entry["status"]="partial"
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        raise
    finally:
        entry.update(wall_seconds=time.monotonic()-start,end_unix=time.time(),exit_code=None if child is None else child.returncode)
        state["stages"][f"s{seed}"]=entry;state.pop("active",None);atomic_json(STATE,state)

def main():
    assert not WORK.exists(),"existing queue; never restart automatically"
    assert all(not run_path(s).exists() for s in range(5)),"formal attempt already exists"
    sha=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    assert not subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip()
    v=json.loads((VALIDATION/"validation.json").read_text())
    assert v["status"]=="passed" and v["code_sha"]==sha and v["fixed_batch_updates"]<=100 and v["wall_seconds"]<=1800
    for seed in range(5):
        m=parent_manifest(seed);atomic_json(VALIDATION/f"parent-s{seed}.json",m)
        hook=json.loads((VALIDATION/f"hook-s{seed}/validation.json").read_text())
        assert hook["status"]=="passed" and hook["initialization"]["parent_sha256"]==m["parent_sha256"]
    with LOCK.open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True).strip()
        WORK.mkdir();lock.seek(0);lock.truncate();lock.write(str(os.getpid()));lock.flush()
        state={"status":"running","code_sha":sha,"thread_id":THREAD,"queue_pid":os.getpid(),"started_unix":time.time(),
               "plan":[0,1,2,3,4],"stages":{},"auto_retry":False,"fixed_batch_updates":v["fixed_batch_updates"],
               "training_limits":{"per_run_seconds":18000,"total_seconds":86400,"total_env_steps":500000}}
        atomic_json(STATE,state)
        def interrupt(signum,frame):raise RuntimeError(f"queue signal {signum}; stop remaining")
        signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
        try:
            deliver(state)
            for seed in range(5):
                watch(state,seed);seal(seed);deliver(state)
            state.update(status="completed",finished_unix=time.time(),current_seed=None);deliver(state)
        except BaseException as exc:
            state.update(status="partial",error=repr(exc),traceback=traceback.format_exc(),finished_unix=time.time())
            atomic_json(STATE,state)
            for seed in range(5):
                if run_path(seed).exists() and not (RESEARCH/"results/raw/EXP-029"/run_path(seed).name).exists():seal(seed)
            try:deliver(state)
            except BaseException as error:state["delivery_error"]=repr(error);atomic_json(STATE,state)
            raise
        finally:
            lock.seek(0);lock.truncate();lock.flush()

if __name__=="__main__":main()
