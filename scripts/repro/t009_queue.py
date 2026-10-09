"""Durable serial EXP-024 factorial queue, staged and fail-closed."""
import hashlib, json, math, os, random, re, subprocess, sys, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RESEARCH=Path("/home/lzy/Projects/rl_sample_efficiency_research_t009")
WORK=ROOT/"_so2_work/runs/EXP-024"; STATE=WORK/"queue_state.json"; LOCK=ROOT/"_so2_work/EXP-024.queue.lock"
CONFIG=ROOT/"scripts/repro/configs/halfcheetah_medium_replay_t009.py"
PREFLIGHT=ROOT/"_so2_work/validation/EXP-024/t009_fixed_batch.json"
PYTHON=Path("/home/lzy/Projects/SO2/.venv/bin/python")
CHECKPOINT=Path("/home/lzy/Projects/SO2/_so2_work/assets/checkpoints/ckpt/halfcheetah-medium-replay-v2.ckpt")
DATASET=Path("/home/lzy/.d4rl/datasets/halfcheetah_medium_replay-v2.hdf5")
CKPT_SHA="02d74b7860620f1f9863a2f217aff0abfaae37f4a69ba6d0e1a0ea10f98d94f7"; DATA_SHA="48d494a4770c11f48260736dec090b78bdca8375647ed4d24bfa5b6610c7f683"
P_GRID=(0.0,0.0001,0.001,0.005,0.01,0.05,0.1); AGGS=("min","mean"); TARGET=100000; MAX_SEED=28800; MAX_TOTAL=168*3600

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix(p.suffix+".tmp"); t.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n"); os.replace(t,p)
def read_rows(p):
 out=[]
 if Path(p).exists():
  for line in Path(p).read_text(errors="replace").splitlines():
   try: out.append(json.loads(line))
   except Exception: pass
 return out
def cell_id(c): return f"{c['aggregation']}-p{c['p']:g}-s{c['seed']}"
def make_schedule():
 cells=[{"aggregation":a,"p":p,"seed":seed} for a in AGGS for p in P_GRID for seed in range(3)]
 first=[]
 for seed in range(3):
  first.extend([{"aggregation":"mean","p":0.0,"seed":seed},{"aggregation":"mean","p":0.005,"seed":seed}])
 reuse={("min",0.0,seed):"EXP-019" for seed in range(3)}
 reuse.update({("min",0.005,0):"EXP-016",("min",0.005,1):"EXP-018",("min",0.005,2):"EXP-018"})
 first_keys={(c["aggregation"],c["p"],c["seed"]) for c in first}
 rng=random.Random(20261009); rest=[]
 for seed in range(3):
  block=sorted([c for c in cells if c["seed"]==seed and
   (c["aggregation"],c["p"],c["seed"]) not in first_keys and
   (c["aggregation"],c["p"],c["seed"]) not in reuse],key=lambda c:(c["aggregation"],c["p"]))
  rng.shuffle(block); rest.extend(block)
 reused=[c for c in cells if (c["aggregation"],c["p"],c["seed"]) in reuse]
 return first+reused+rest,reuse
def score_cell(path):
 rows=read_rows(path/"metrics.jsonl"); ev=[r for r in rows if r.get("event")=="evaluation"]
 x=[float(r["env_steps"]) for r in ev]; y=[float(r["normalized_score_100"]) for r in ev]
 if len(ev)!=41 or x!=list(range(0,100001,2500)) or any(not math.isfinite(v) for v in y): raise RuntimeError(f"invalid evaluation series: {path}")
 auc=sum((y[i]+y[i+1])*.5*(x[i+1]-x[i]) for i in range(40))/100000
 return {"auc":auc,"endpoint":y[-1],"curve_steps":x,"curve_score":y}
def run_cell(state,c,total_deadline):
 cid=cell_id(c); run=WORK/cid/"attempt1"
 if run.exists(): raise RuntimeError(f"refuse overwrite {run}")
 run.parent.mkdir(parents=True,exist_ok=True); log=run.parent/"train.log"
 cmd=[str(PYTHON),str(ROOT/"scripts/repro/t009_run.py"),"--config",str(CONFIG),"--seed",str(c["seed"]),"--aggregation",c["aggregation"],"--dropout-p",str(c["p"]),"--max-env-steps",str(TARGET),"--max-wall-clock-sec",str(MAX_SEED),"--run-dir",str(run)]
 ent={**c,"cell_id":cid,"status":"running","run_dir":str(run),"log":str(log),"command":cmd,"start_unix":time.time()}; state["current_cell"]=cid; state.setdefault("runs",[]).append(ent); save(STATE,state)
 with log.open("w") as f:
  proc=subprocess.Popen(cmd,cwd=ROOT,env=os.environ.copy(),stdout=f,stderr=subprocess.STDOUT); ent["pid"]=proc.pid; save(STATE,state); start=time.monotonic(); checked=False
  while proc.poll() is None:
   rows=read_rows(run/"metrics.jsonl"); first=next((r for r in rows if r.get("event")=="evaluation_state" and r.get("env_steps") in (7500,10000)),None)
   if first and not checked:
    step=int(first["env_steps"]); ev=next((r for r in rows if r.get("event")=="evaluation" and r.get("env_steps")==step),None); ck=run/"ckpt"/f"envstep_{step}.pth.tar"; expected={"critic":10*(step-5000),"actor":step-5000,"alpha":0}
    valid=ev is not None and len(ev.get("returns",[]))==20 and all(math.isfinite(float(v)) for v in ev["returns"])
    check={"passed":first.get("updates")==expected and valid and ck.exists(),"seed":c["seed"],"step":step,"updates":first.get("updates"),"expected_updates":expected,"finite_20_returns":valid,"checkpoint_exists":ck.exists(),"time":time.time()}; ent["first_window"]=check; save(run.parent/f"first_window_s{c['seed']}.json",check); save(STATE,state)
    if not check["passed"]:
     proc.terminate(); proc.wait(timeout=30); ent.update(status="failed",exit_code=proc.returncode,error="first-window validation failed"); state.update(status="partial",error=ent["error"]); save(STATE,state); raise RuntimeError(ent["error"])
    checked=True
   progress=next((r for r in reversed(rows) if r.get("event")=="evaluation_state"),None)
   if progress: state["progress"]={"cell":cid,"env_steps":progress.get("env_steps"),"updates":progress.get("updates"),"time":time.time()}; save(STATE,state)
   if time.monotonic()>min(start+MAX_SEED,total_deadline):
    proc.terminate();
    try: proc.wait(timeout=30)
    except subprocess.TimeoutExpired: proc.kill(); proc.wait()
    ent.update(status="partial",error="per-run or total wall-clock budget exhausted",end_unix=time.time()); state.update(status="partial",error=ent["error"]); save(STATE,state); raise RuntimeError(ent["error"])
   time.sleep(5)
  code=proc.wait()
 ent.update(exit_code=code,wall_sec=time.monotonic()-start,end_unix=time.time(),status="completed" if code==0 else "failed"); save(STATE,state)
 if code or not checked: state.update(status="partial",error=f"run failed/ended before validation: {cid}"); save(STATE,state); raise RuntimeError(state["error"])
 ac=subprocess.run([str(PYTHON),str(ROOT/"scripts/repro/t009_accept.py"),str(run),"--target",str(TARGET),"--log",str(log)],cwd=ROOT,capture_output=True,text=True)
 (run.parent/f"acceptance_s{c['seed']}.log").write_text(ac.stdout+ac.stderr)
 if ac.returncode: state.update(status="partial",error=f"acceptance failed: {cid}"); save(STATE,state); raise RuntimeError(state["error"])
 a=json.loads((run/"acceptance.json").read_text()); meta=json.loads((run/"run_meta.json").read_text())
 if meta.get("git_commit")!=state["code_sha"] or meta.get("checkpoint_sha256")!=CKPT_SHA or meta.get("dataset_sha256")!=DATA_SHA or meta.get("aggregation")!=c["aggregation"] or meta.get("dropout_p")!=c["p"] or not a.get("accepted"): raise RuntimeError(f"identity audit failed {cid}")
 ent["status"]="accepted"; ent["metrics"]=score_cell(run); state["accepted_new_runs"]+=1; state["train_env_steps"]+=TARGET; state["train_wall_sec"]+=ent["wall_sec"]; state["current_cell"]=None; save(STATE,state)
 seal(run,ent,state); deliver(state,partial=True)
def seal(run,ent,state):
 files=[]
 for p in sorted(run.rglob("*")):
  if p.is_file() and p.name!="artifact_manifest.json": files.append({"path":str(p.relative_to(run)),"size":p.stat().st_size,"sha256":sha(p)})
 save(run/"artifact_manifest.json",{"run_id":ent["cell_id"],"files":files})
 import shutil
 dest=RESEARCH/"results/raw/EXP-024"/ent["cell_id"]; dest.mkdir(parents=True,exist_ok=True)
 for name in ("metrics.jsonl","run_meta.json","acceptance.json","policy_reload.json","effective_config.json","artifact_manifest.json"):
  src=run/name
  if src.exists(): shutil.copy2(src,dest/name)
 state.setdefault("accepted_runs",[]).append({k:ent[k] for k in ("cell_id","aggregation","p","seed","run_dir","wall_sec","metrics")}); save(STATE,state)
def write_processed(state):
 processed=RESEARCH/"results/processed/EXP-024"; processed.mkdir(parents=True,exist_ok=True)
 rows=[]
 for item in state.get("reused",[])+state.get("accepted_runs",[]):
  metrics=item.get("metrics",{})
  rows.append({"cell_id":item.get("cell_id",f"{item.get('aggregation')}-p{item.get('p')}-s{item.get('seed')}"),
   "aggregation":item.get("aggregation"),"p":item.get("p"),"seed":item.get("seed"),
   "source_run":item.get("source_run"),"source_sha":item.get("source_sha"),
   "auc":metrics.get("auc"),"endpoint":metrics.get("endpoint"),
   "status":"reused" if item in state.get("reused",[]) else "accepted",
   "run_dir":item.get("run_dir")})
 save(processed/"stage1_ledger.json",{"exp_id":"EXP-024","runs":rows,"accepted_new_runs":state.get("accepted_new_runs",0),"train_env_steps":state.get("train_env_steps",0)})
 save(processed/"queue_state.json",state)
 report=RESEARCH/"reports/so2/SO2-T009.md"
 if report.exists():
  content=report.read_text()
  status=f"执行状态：{state.get('status')} / {state.get('stage')}；新run {state.get('accepted_new_runs',0)} accepted；训练环境步 {state.get('train_env_steps',0)}。"
  content=re.sub(r"^执行状态：.*$",status,content,count=1,flags=re.M)
  report.write_text(content)
 tracker=RESEARCH/"docs/EXPERIMENT_TRACKER.md"
 if tracker.exists():
  lines=tracker.read_text().splitlines()
  for i,line in enumerate(lines):
   if line.startswith("| EXP-024 |"):
    cells=line.split("|")
    cells[3]=f" {state.get('status')} / {state.get('stage')}，新run {state.get('accepted_new_runs',0)}，环境步 {state.get('train_env_steps',0)} "
    lines[i]="|".join(cells); break
  tracker.write_text("\n".join(lines)+"\n")
 manifest={"sources":[],"new_runs":[]}
 for x in state.get("reused",[]): manifest["sources"].append({"run_id":Path(x["source_run"]).name,"source_run":x["source_run"],"source_sha":x["source_sha"],"aggregation":x["aggregation"],"p":x["p"],"seed":x["seed"]})
 for x in state.get("accepted_runs",[]): manifest["new_runs"].append({"run_id":x["cell_id"],"linux_path":x["run_dir"],"aggregation":x["aggregation"],"p":x["p"],"seed":x["seed"]})
 save(RESEARCH/"results/raw/EXP-024/manifest_index.json",manifest)
def deliver(state,partial=False):
 write_processed(state)
 subprocess.run(["git","add","reports/so2/SO2-T009.md","reports/so2/SO2-T009","docs/EXPERIMENT_TRACKER.md","results/processed/EXP-024","results/raw/EXP-024"],cwd=RESEARCH,check=True)
 diff=subprocess.run(["git","diff","--cached","--quiet"],cwd=RESEARCH)
 if diff.returncode:
  subprocess.run(["git","commit","-m",f"SO2-T009: record {state.get('accepted_new_runs',0)} accepted runs"],cwd=RESEARCH,check=True,stdout=subprocess.DEVNULL)
 subprocess.run(["git","push","origin","codex/so2-t009-delivery"],cwd=RESEARCH,check=True,stdout=subprocess.DEVNULL)
 state["research_sha"]=subprocess.check_output(["git","rev-parse","HEAD"],cwd=RESEARCH,text=True).strip(); save(STATE,state)
def main():
 if STATE.exists() or LOCK.exists(): raise RuntimeError("queue state/lock exists; duplicate start refused")
 if not PREFLIGHT.exists() or json.loads(PREFLIGHT.read_text()).get("status")!="passed": raise RuntimeError("preflight not passed")
 if sha(CHECKPOINT)!=CKPT_SHA or sha(DATASET)!=DATA_SHA: raise RuntimeError("input hash mismatch")
 if subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip(): raise RuntimeError("algorithm worktree dirty")
 if subprocess.check_output(["git","status","--porcelain"],cwd=RESEARCH,text=True).strip(): raise RuntimeError("research worktree dirty")
 schedule,reuse=make_schedule()
 schedule_path=ROOT/"scripts/repro/configs/t009_schedule.json"
 if not schedule_path.exists(): raise RuntimeError("frozen schedule missing")
 schedule_obj=json.loads(schedule_path.read_text())
 expected=[{**c,"cell_id":cell_id(c),"reuse":reuse.get((c["aggregation"],c["p"],c["seed"]))} for c in schedule]
 if schedule_obj.get("seed")!=20261009 or schedule_obj.get("schedule")!=expected: raise RuntimeError("frozen schedule mismatch")
 state={"status":"running","exp_id":"EXP-024","stage":"stage1","code_sha":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"config_sha256":sha(CONFIG),"checkpoint_sha256":CKPT_SHA,"dataset_sha256":DATA_SHA,"started_unix":time.time(),"host":os.uname().nodename,"accepted_new_runs":0,"train_env_steps":0,"train_wall_sec":0.0,"total_budget_sec":MAX_TOTAL,"schedule_sha256":sha(ROOT/"scripts/repro/configs/t009_schedule.json"),"reused":[]}
 WORK.mkdir(parents=True,exist_ok=False); fd=os.open(LOCK,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600); os.write(fd,str(os.getpid()).encode()); os.close(fd); save(STATE,state)
 try:
  deliver(state,partial=True)
  # Phase one, with mandated mean anchors first.
  for c in schedule:
   key=(c["aggregation"],c["p"],c["seed"])
   if key in reuse:
    source="/home/lzy/Projects/rl_sample_efficiency_research_t008/results/raw/EXP-019" if c["p"]==0 else ("/home/lzy/Projects/rl_sample_efficiency_research_t008/results/raw/EXP-016" if c["seed"]==0 else "/home/lzy/Projects/rl_sample_efficiency_research_t008/results/raw/EXP-018")
    exp_id=19 if c["p"]==0 else (16 if c["seed"]==0 else 18)
    run=Path(source)/f"EXP-{exp_id:03d}-s{c['seed']}-attempt1"
    expected_code="bb69fad91aff905e78c4535e0e88562d16dfa383" if c["p"]==0 else ("c1ed167618ce210dea7c974ffee28aba02b730cc" if c["seed"]==0 else "5417aedcf770278ff0a97519518546ab91be1310")
    expected_config="e5b3f813997d752c2bb05defc61937cd3f7cc78e973ca1faa096adf5ba11600c" if c["p"]==0 else "b314b70c161db8359b5d24753244776ee5a065591a4246d03f5f9374129d552f"
    srcmeta=json.loads((run/"run_meta.json").read_text()); acc=json.loads((run/"acceptance.json").read_text()); manifest=json.loads((run/"artifact_manifest.json").read_text()); metrics=score_cell(run)
    ev=[r for r in read_rows(run/"metrics.jsonl") if r.get("event")=="evaluation"]
    if not acc.get("accepted") or acc.get("errors") or srcmeta.get("git_commit")!=expected_code or srcmeta.get("config_sha256")!=expected_config or srcmeta.get("seed")!=c["seed"] or len(ev)!=41 or any(len(r.get("returns",[]))!=20 or not all(math.isfinite(float(v)) for v in r["returns"]) for r in ev) or acc.get("final_state",{}).get("updates")!={"critic":950000,"actor":95000,"alpha":0} or len(manifest)!=45 or any(not (run/e["path"]).exists() or sha(run/e["path"])!=e["sha256"] for e in manifest): raise RuntimeError(f"reused source validation failed: {run}")
    state["reused"].append({**c,"source_run":str(run),"source_sha":srcmeta["git_commit"],"source_manifest_sha256":sha(run/"artifact_manifest.json"),"accepted":acc["accepted"],"metrics":metrics}); save(STATE,state)
   else:
    if state["train_wall_sec"]>=MAX_TOTAL: raise RuntimeError("168-hour budget exhausted")
    run_cell(state,c,time.monotonic()+MAX_TOTAL-state["train_wall_sec"])
  state["stage1_complete"]=True; state["stage"]="selection"; save(STATE,state); select_and_stage2(state)
 except BaseException as exc:
  state.update(status="partial",error=repr(exc),traceback=traceback.format_exc(),finished_unix=time.time()); save(STATE,state)
  try: deliver(state,partial=True)
  except Exception as e: state["delivery_error"]=repr(e); save(STATE,state)
  raise
 finally:
  if STATE.exists() and json.loads(STATE.read_text()).get("status") in ("completed","partial","failed"):
   try: LOCK.unlink()
   except FileNotFoundError: pass

def select_and_stage2(state):
 rows=state.get("reused",[])+state.get("accepted_runs",[])
 lookup={(x["aggregation"],float(x["p"]),int(x["seed"])):x for x in rows}
 if len(lookup)!=42: raise RuntimeError(f"stage-one matrix incomplete: {len(lookup)}/42")
 selection={"metric":"mean seed AUC; ties: endpoint desc, AUC sample SD asc, p asc","selected":{},"candidates":[]}
 for agg in AGGS:
  candidates=[]
  for p in P_GRID:
   vals=[lookup[(agg,p,s)]["metrics"]["auc"] for s in range(3)]
   ends=[lookup[(agg,p,s)]["metrics"]["endpoint"] for s in range(3)]
   mean=sum(vals)/3; sd=(sum((v-mean)**2 for v in vals)/2)**0.5
   candidates.append({"aggregation":agg,"p":p,"auc_by_seed":vals,"endpoint_by_seed":ends,"mean_auc":mean,"mean_endpoint":sum(ends)/3,"auc_sd":sd})
  winner=sorted(candidates,key=lambda x:(-x["mean_auc"],-x["mean_endpoint"],x["auc_sd"],x["p"]))[0]
  selection["selected"][agg]=winner["p"]; selection["candidates"].extend(candidates)
 stage2=[]
 for agg in AGGS:
  for p in sorted({0.0,float(selection["selected"][agg])}):
   for seed in range(5):
    if (agg,p,seed) not in lookup: stage2.append({"aggregation":agg,"p":p,"seed":seed})
 selection["stage2_schedule"]=stage2; selection["stage2_new_runs"]=len(stage2)
 if len(stage2)>6: raise RuntimeError("stage-two run count exceeds six")
 save(RESEARCH/"results/processed/EXP-024/selection.json",selection); state["selection"]=selection; state["stage"]="stage2"; save(STATE,state); deliver(state,partial=True)
 deadline=time.monotonic()+max(0,MAX_TOTAL-state["train_wall_sec"])
 for c in stage2: run_cell(state,c,deadline)
 state["status"]="completed"; state["stage"]="completed"; state["finished_unix"]=time.time(); save(STATE,state); deliver(state)
if __name__=="__main__": main()
