"""Persistent fail-closed T010 queue; awaits T009 handoff before touching CUDA."""
import fcntl
import math
import shutil
import signal
import subprocess
import time
import traceback

from t010_edac import *

PYTHON = "/home/lzy/Projects/SO2/.venv/bin/python"
WORK = ROOT / "_so2_work/t010-v2"
STATE = WORK / "queue_state.json"
RESEARCH = Path("/home/lzy/Projects/rl_sample_efficiency_research_t010")
T009 = Path("/home/lzy/Projects/SO2_t009")
T009_STATE = T009 / "_so2_work/runs/EXP-024/queue_state.json"
GLOBAL_LOCK = Path("/home/lzy/Projects/SO2/_so2_work/training_resource.lock")
RUNS = {"offline":ROOT/"_so2_work/runs/EXP-025/EXP-025-s0-attempt1",
        "online":ROOT/"_so2_work/runs/EXP-026/EXP-026-s0-attempt1"}
START = time.time()


def process_alive(pid):
    stat=Path(f"/proc/{pid}/stat")
    if not stat.exists():return False
    return stat.read_text().rsplit(")",1)[1].split()[0] != "Z"


def resource_gate():
    old=json.loads(T009_STATE.read_text())
    if old.get("status") != "partial": return False,"T009 not sealed/stopped",None
    if old.get("current_cell") is not None: return False,"T009 current run not sealed",None
    if (T009/"_so2_work/EXP-024.queue.lock").exists(): return False,"T009 queue lock retained",None
    if process_alive(21225) or process_alive(65524): return False,"original T009 writer/train PID active",None
    if old.get("delivery_error"):return False,"T009 delivery error",None
    assert "protocol" in old.get("error","").lower() or "研究协议" in old.get("error",""), "unexpected T009 failure; manual review required"
    for entry in old.get("runs",[]):
        if entry.get("status")=="running" or (entry.get("pid") and process_alive(entry["pid"])):
            return False,"T009 writer or training still running",None
    repo=Path("/home/lzy/Projects/rl_sample_efficiency_research_t009")
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip()
    remote=subprocess.check_output(["git","ls-remote","origin","refs/heads/codex/so2-t009-delivery"],cwd=repo,text=True).split()[0]
    if head!=remote:return False,"T009 latest evidence not pushed",None
    last=old["accepted_runs"][-1]; run=Path(last["run_dir"])
    assert (run/"artifact_manifest.json").exists() and json.loads((run/"acceptance.json").read_text())["accepted"]
    assert json.loads((run/"run_meta.json").read_text())["status"]=="completed"
    gpu=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True)
    if gpu.strip(): return False,"GPU compute process remains",None
    # Include any subsequent T009 queue process, not only the original PID.
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():continue
        try:
            cmd=(proc/"cmdline").read_bytes().replace(b"\0",b" ")
            if b"t009_queue.py" in cmd or b"t009_run.py" in cmd:
                if process_alive(int(proc.name)):return False,"another T009 process remains",None
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    return True,"resource handoff confirmed",{"research_sha":head,"accepted_runs":old["accepted_new_runs"],
                "train_env_steps":old["train_env_steps"],"stop_reason":old.get("error"),
                "last_run":last["cell_id"],"t009_code_sha":old["code_sha"],"original_pids_inactive":True,
                "lock_released":True,"gpu_compute_empty":True,"confirmed_unix":time.time()}


def load_rows(path):
    out=[]
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            try:out.append(json.loads(line))
            except json.JSONDecodeError:pass
    return out


def archive_checkpoints(phase):
    run=RUNS[phase]; path=run/"checkpoint_manifest.json"
    old=json.loads(path.read_text()) if path.exists() else []
    names={e["path"] for e in old}
    for file in sorted((run/"ckpt").glob("*")):
        rel=str(file.relative_to(run))
        if not file.is_file() or rel in names or file.suffix==".tmp":continue
        old.append({"path":rel,"size":file.stat().st_size,"sha256":digest(file),"backup_status":"Linux only"})
    atomic_json(path,old)


def deliver(state):
    atomic_json(STATE,state)
    proc=subprocess.run([PYTHON,str(ROOT/"scripts/repro/t010_deliver.py"),"--state",str(STATE)],cwd=ROOT,capture_output=True,text=True)
    with (WORK/"delivery.log").open("a") as f:f.write(proc.stdout+proc.stderr)
    if proc.returncode:
        state["delivery_error"]=(proc.stdout+proc.stderr)[-3000:];atomic_json(STATE,state)
        raise RuntimeError("research delivery failed; later stages stopped")
    state["research_sha"]=subprocess.check_output(["git","rev-parse","HEAD"],cwd=RESEARCH,text=True).strip()
    atomic_json(STATE,state)


def spawn_and_watch(state,phase,cmd,budget):
    assert shutil.disk_usage(ROOT).free>20*1024**3,"free disk below 20GiB"
    if phase in RUNS:assert not RUNS[phase].exists(),"formal run overwrite/restart refused"
    log=WORK/f"{phase}.log"; started=time.monotonic()
    state.update(status="running",phase=phase,command=cmd,phase_start_unix=time.time())
    with log.open("x") as f:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=os.environ.copy())
        state["child_pid"]=proc.pid;atomic_json(STATE,state)
        last_delivery=time.monotonic();first_checked=False
        try:
            while proc.poll() is None:
                assert time.monotonic()-started<=budget,"phase process wall-clock budget exhausted"
                assert shutil.disk_usage(ROOT).free>20*1024**3,"disk reserve exhausted"
                if phase=="offline":
                    rp=RUNS[phase]/"state.json"
                    if rp.exists():state["progress"]=json.loads(rp.read_text())
                elif phase=="online":
                    records=load_rows(RUNS[phase]/"metrics.jsonl")
                    states=[r for r in records if r.get("event")=="evaluation_state"]
                    if states:state["progress"]={k:states[-1].get(k) for k in ("env_steps","updates","evaluation_interactions_total")}
                    first=next((r for r in states if r["env_steps"]==7500),None)
                    if first and not first_checked:
                        ev=next((r for r in records if r.get("event")=="evaluation" and r["env_steps"]==7500),None)
                        assert first["updates"]=={"critic":25000,"actor":2500,"alpha":0}
                        assert ev and len(ev["returns"])==20 and all(math.isfinite(x) for x in ev["returns"])
                        assert (RUNS[phase]/"ckpt/envstep_7500.pth.tar").exists()
                        state["first_window"]={"passed":True,"env_steps":7500,"updates":first["updates"]}
                        atomic_json(RUNS[phase]/"first_window.json",state["first_window"]);first_checked=True
                state["phase_wall_sec"]=time.monotonic()-started;atomic_json(STATE,state)
                if time.monotonic()-last_delivery>=21600:
                    if phase in RUNS:archive_checkpoints(phase)
                    deliver(state);last_delivery=time.monotonic()
                time.sleep(min(10, max(.01, budget-(time.monotonic()-started))))
        except BaseException:
            proc.terminate()
            try:proc.wait(timeout=45)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
            state["child_exit_code"]=proc.returncode;atomic_json(STATE,state);raise
        code=proc.wait()
    state["child_exit_code"]=code;state["phase_wall_sec"]=time.monotonic()-started
    state.setdefault("stages",{})[phase]={"exit_code":code,"wall_sec":state["phase_wall_sec"],"log":str(log),"command":cmd}
    atomic_json(STATE,state)
    assert code==0,f"{phase} exited {code}; no automatic retry"
    assert state["phase_wall_sec"]<=budget,"phase finished after its wall-clock limit"


def main():
    assert not STATE.exists() and not WORK.exists(),"queue already exists; restart refused"
    WORK.mkdir(parents=True);state={"status":"waiting","phase":"handoff","started_unix":START,
       "thread_id":"01a12511-0d56-7db1-bd81-483c7c20d105","plan_sha":"3b6896c2e81b63c37fa2d0780130ec610fbf45b1",
       "code_sha":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
       "recipe_sha256":digest(ROOT/"scripts/repro/configs/t010_recipe.json"),"runs":{k:str(v) for k,v in RUNS.items()},
       "budgets_sec":{"gpu_preflight":1800,"offline":172800,"online":28800},"stages":{},
       "cpu_test_updates_actual":63,"fixed_batch_total_limit":200,"auto_retry":False,
       "engineering_started_unix":1791623273,"queue_workdir":str(WORK),
       "previous_preflight_updates":15,"previous_cpu_updates":48,"previous_gpu_preflight_wall_sec":10.016560283009312}
    atomic_json(STATE,state)
    def interrupt(signum, frame):
        raise RuntimeError(f"queue signal {signum}; retain partial and stop subsequent stages")
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    assert not subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip()
    try:
        deliver(state)
        while True:
            ready,reason,handoff=resource_gate();state["handoff_status"]=reason;atomic_json(STATE,state)
            if ready:break
            assert time.time()-state["engineering_started_unix"]<8*3600,"handoff not resolved within engineering budget"
            time.sleep(30)
        state["handoff"]=handoff;atomic_json(WORK/"handoff.json",handoff)
        with GLOBAL_LOCK.open("a+") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            lock.seek(0);lock.truncate();lock.write(str(os.getpid()));lock.flush()
            gpu_dir=ROOT/"_so2_work/validation/t010-gpu-v2"
            spawn_and_watch(state,"gpu_preflight",[PYTHON,str(ROOT/"scripts/repro/t010_validate.py"),"--device","cuda","--out",str(gpu_dir)],1800-state["previous_gpu_preflight_wall_sec"])
            result=json.loads((gpu_dir/"validation.json").read_text());assert result["status"]=="passed"
            assert result["fixed_batch_updates"]+state["cpu_test_updates_actual"]<=200
            state["gpu_preflight"]=result;deliver(state)
            for phase in ("offline","online"):
                cmd=[PYTHON,str(ROOT/"scripts/repro/t010_run.py"),"--phase",phase,"--run-dir",str(RUNS[phase])]
                if phase=="online":cmd += ["--parent",str(RUNS["offline"]/"ckpt/update_3000000.pth")]
                spawn_and_watch(state,phase,cmd,172800 if phase=="offline" else 28800)
                acceptance=json.loads((RUNS[phase]/"acceptance.json").read_text());assert acceptance["accepted"]
                state["stages"][phase]["acceptance"]=acceptance;archive_checkpoints(phase);deliver(state)
            state.update(status="completed",phase="completed",finished_unix=time.time());deliver(state)
    except BaseException as exc:
        state.update(status="partial",error=repr(exc),traceback=traceback.format_exc(),finished_unix=time.time())
        atomic_json(STATE,state)
        for phase in RUNS:
            if RUNS[phase].exists():archive_checkpoints(phase)
        try:deliver(state)
        except BaseException as e:state["delivery_error"]=repr(e);atomic_json(STATE,state)
        raise

if __name__=="__main__":main()
