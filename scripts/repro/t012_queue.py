"""Single-lock twenty-stage queue, separate cumulative budgets, no retry."""
import fcntl,math,signal,time,traceback
from t012_common import *
from t010_run import rows
from t011_queue import belongs
STATE=WORK/"queue_state.json"
PLAN=[(g,s,p) for g in ("dropout","ln") for s in range(5) for p in ("offline","online")]

def stage_key(group,seed,phase):return f"{group}-s{seed}-{phase}"
def budget_for(state,group,phase):
    limit=172800 if phase=="offline" else 86400
    used=sum(e["wall_seconds"] for e in state["stages"].values() if e["group"]==group and e["phase"]==phase)
    left=limit-used;assert left>60,"cumulative stage budget exhausted"
    return min(36000 if phase=="offline" else 18000,left)

def accept_stage(a,phase):
    assert a["accepted"] and not a["errors"],"stage rejected"
    if phase=="offline":
        assert a["updates"]=={"critic":3000000,"actor":3000000,"alpha":3000000}
        assert a["train_env_steps"]==0 and a["evaluation_count"]==31
    else:
        assert a["final_state"]["updates"]=={"critic":950000,"actor":95000,"alpha":0}
        assert a["final_state"]["env_steps"]==100000 and a["evaluation_count"]==41

def check_returns(events):
    for e in events:
        assert len(e["returns"])==20 and all(math.isfinite(v) for v in e["returns"])
        assert math.isfinite(e["normalized_score_100"])

def watch(state,group,seed,phase):
    run=run_path(group,seed,phase);assert not run.exists(),"duplicate formal output"
    budget=budget_for(state,group,phase)
    cmd=["/usr/bin/timeout","--signal=TERM","--kill-after=5s",f"{budget-10:.3f}s",PYTHON,str(ROOT/"scripts/repro/t012_run.py"),"--phase",phase,"--seed",str(seed),"--wall-seconds",str(budget-30)]
    key=stage_key(group,seed,phase);entry={"group":group,"seed":seed,"phase":phase,"command":cmd,"budget_seconds":budget,"start_unix":time.time(),"status":"running"}
    state.update(current=key,active=entry);atomic_json(STATE,state)
    start=time.monotonic();child=None;first=False
    try:
        with (WORK/f"{key}.log").open("x") as log:
            child=subprocess.Popen(cmd,cwd=ROOT,env={**os.environ,"T012_GROUP":group},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            entry["pid"]=child.pid
            while child.poll() is None:
                assert time.monotonic()-start<budget,"supervisor timeout"
                assert shutil.disk_usage(ROOT).free>20*1024**3,"disk reserve exhausted"
                assert subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()==state["code_sha"]
                assert not subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip(),"running code dirty"
                pids=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True).strip()
                assert all(belongs(int(pid),child.pid) for pid in pids.splitlines()),"another compute process"
                entry["wall_seconds"]=time.monotonic()-start
                if (run/"run_meta.json").exists():
                    m=json.loads((run/"run_meta.json").read_text())
                    assert m["git_commit"]==state["code_sha"] and m["seed"]==seed and m["group"]==group
                    assert m["layer_norm"]==SETTINGS[group][0] and m["dropout_p"]==SETTINGS[group][1]
                    assert m["offline_actor_aggregation"]=="min" and m["online_actor_aggregation"]=="mean"
                records=rows(run/"metrics.jsonl");ev=[e for e in records if e.get("event")=="evaluation"];check_returns(ev)
                if phase=="offline":
                    assert [e["gradient_updates"] for e in ev]==list(range(0,len(ev)*100000,100000))
                    if (run/"state.json").exists():
                        progress=json.loads((run/"state.json").read_text());entry["progress"]=progress
                        if "last_losses" in progress:assert all(math.isfinite(v) for v in progress["last_losses"].values())
                    checkpoint=run/"ckpt/update_100000.pth";ready=len(ev)>=2 and checkpoint.exists()
                else:
                    assert [e["env_steps"] for e in ev]==list(range(0,len(ev)*2500,2500))
                    states=[e for e in records if e.get("event")=="evaluation_state"]
                    for e in states:
                        c=max(0,e["env_steps"]-5000)*10
                        assert e["updates"]=={"critic":c,"actor":c//10,"alpha":0}
                    if states:entry["progress"]={k:states[-1][k] for k in ("env_steps","updates","evaluation_interactions_total")}
                    checkpoint=run/"ckpt/envstep_7500.pth.tar";ready=any(e["env_steps"]==7500 for e in states) and checkpoint.exists()
                if ready and not first:
                    atomic_json(run/"first_window.json",{"passed":True,"group":group,"seed":seed,"phase":phase,"checkpoint_sha256":digest(checkpoint),"finite_returns":True,"evaluation_count":len(ev),"progress":entry.get("progress")});first=True
                atomic_json(STATE,state);time.sleep(5)
            assert child.returncode==0,f"stage exited {child.returncode}; no retry"
            a=json.loads((run/"acceptance.json").read_text());accept_stage(a,phase);assert first
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
        state["stages"][key]=entry;state.pop("active",None);atomic_json(STATE,state)

def main():
    from t012_deliver import seal,deliver
    assert not WORK.exists(),"existing queue; no automatic restart"
    assert all(not run_path(*item).exists() for item in PLAN)
    sha=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    assert not subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip()
    v=json.loads((VALIDATION/"validation.json").read_text())
    assert v["status"]=="passed" and v["code_sha"]==sha and v["reserved_fixed_batch_calls"]<=200 and v["wall_seconds"]<=3600
    assert json.loads((VALIDATION/"queue-validation.json").read_text())["status"]=="passed"
    with LOCK.open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True).strip()
        assert shutil.disk_usage(ROOT).free>20*1024**3
        WORK.mkdir();lock.seek(0);lock.truncate();lock.write(str(os.getpid()));lock.flush()
        state={"status":"running","code_sha":sha,"thread_id":THREAD,"queue_pid":os.getpid(),"started_unix":time.time(),"plan":PLAN,"stages":{},"auto_retry":False,"preflight":v,"group_limits":{"offline":172800,"online":86400},"per_run_limits":{"offline":36000,"online":18000}}
        atomic_json(STATE,state)
        def interrupted(signum,frame):raise RuntimeError(f"queue signal {signum}")
        signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
        try:
            deliver(state)
            for item in PLAN:
                watch(state,*item);seal(*item);deliver(state)
            state.update(status="completed",finished_unix=time.time(),current=None);deliver(state)
        except BaseException as exc:
            state.update(status="partial",error=repr(exc),traceback=traceback.format_exc(),finished_unix=time.time());atomic_json(STATE,state)
            for item in PLAN:
                run=run_path(*item);raw=RESEARCH/"results/raw"/run.parent.name/run.name
                if run.exists() and not raw.exists():seal(*item)
            try:deliver(state)
            except BaseException as error:state["delivery_error"]=repr(error);atomic_json(STATE,state)
            raise
        finally:lock.seek(0);lock.truncate();lock.flush()
if __name__=="__main__":main()
