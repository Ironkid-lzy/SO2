"""Fail-closed remaining T010 paired seeds; retain all attempts and budgets."""
import csv
import fcntl
import math
import os
import shutil
import signal
import statistics
import subprocess
import time
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo

from t010_edac import ROOT, Path, json, atomic_json, digest
from t010_deliver import curves, rows

PYTHON="/home/lzy/Projects/SO2/.venv/bin/python"
RESEARCH=Path("/home/lzy/Projects/rl_sample_efficiency_research_t010")
WORK=ROOT/"_so2_work/t010-multiseed-v1"
STATE=WORK/"queue_state.json"
VALIDATION=ROOT/"_so2_work/validation/t010-multiseed"
LOCK=Path("/home/lzy/Projects/SO2/_so2_work/training_resource.lock")
PLAN=[("offline",seed) for seed in (2,3,4)]+[("online",seed) for seed in (1,2,3,4)]
LIMITS={"offline":172800.,"online":115200.}
THREAD="01a12b0e-1d51-7730-9b01-98ffac956424"


def run_path(phase,seed):
    exp=("EXP-025" if seed==0 else "EXP-027") if phase=="offline" else ("EXP-026" if seed==0 else "EXP-028")
    attempt=2 if phase=="online" and seed==0 else 1
    return ROOT/f"_so2_work/runs/{exp}/{exp}-s{seed}-attempt{attempt}"


def used_seconds(state,phase):
    return state["prior_offline_wall_seconds"]*(phase=="offline")+sum(v["wall_seconds"] for v in state["stages"].values() if v["phase"]==phase)


def budget_for(state,phase):
    remaining=LIMITS[phase]-used_seconds(state,phase)
    budget=min(28800.,remaining) if phase=="online" else remaining
    assert budget>60,"cumulative budget exhausted"
    return budget


def freeze_parent(seed,sha):
    run=run_path("offline",seed);a=json.loads((run/"acceptance.json").read_text());meta=json.loads((run/"run_meta.json").read_text())
    assert a["accepted"] and meta["status"]=="completed" and meta["seed"]==seed
    assert a["updates"]=={"critic":3000000,"actor":3000000,"alpha":3000000}
    parent=run/"ckpt/update_3000000.pth";assert digest(parent)==a["parent_sha256"]
    expected="c7e263cbf09a5b659d3d66e060274cc525c10cd2" if seed==1 else sha
    assert meta["git_commit"]==expected
    record={"seed":seed,"parent_path":str(parent),"parent_sha256":a["parent_sha256"],"parent_code_sha":expected,
            "acceptance_sha256":digest(run/"acceptance.json"),"selection":"fixed 3M final; no reward gate"}
    path=WORK/f"parent-s{seed}.json";assert not path.exists();atomic_json(path,record)
    return parent,path


def archive(run):
    manifest=[]
    for folder in ("ckpt","partial"):
        for path in sorted((run/folder).glob("*")):
            if path.is_file() and path.suffix!=".tmp":manifest.append({"path":str(path.relative_to(run)),"size":path.stat().st_size,"sha256":digest(path),"backup_status":"Linux only"})
    atomic_json(run/"checkpoint_manifest.json",manifest)
    raw=RESEARCH/"results/raw"/run.parent.name/run.name
    assert not raw.exists(),"raw already sealed"
    raw.mkdir(parents=True)
    names=("run_meta.json","state.json","metrics.jsonl","effective_recipe.json","acceptance.json","stage_continuity.json","initialization_manifest.json","policy_reload.json","checkpoint_manifest.json","first_window.json")
    for name in names:
        if (run/name).exists():shutil.copy2(run/name,raw/name)
    files=[{"path":str(p.relative_to(run)),"size":p.stat().st_size,"sha256":digest(p)} for p in sorted(run.rglob("*")) if p.is_file()]
    atomic_json(raw/"artifact_manifest.json",{"linux_path":str(run),"files":files,"large_checkpoint_backup":"Linux only"})


def deliver(state):
    evidence=RESEARCH/"reports/so2/SO2-T010-multiseed";evidence.mkdir(parents=True,exist_ok=True)
    atomic_json(evidence/"queue_state.json",state)
    for name in ("validation.json","resources.json"):
        if (VALIDATION/name).exists():shutil.copy2(VALIDATION/name,evidence/name)
    for path in WORK.glob("parent-s*.json"):shutil.copy2(path,evidence/path.name)
    for path in WORK.glob("hook-s*/validation.json"):
        shutil.copy2(path,evidence/f"{path.parent.name}.json")
    table=[];aggregate={}
    for seed in range(5):
        row={"seed":seed}
        for phase in ("offline","online"):
            run=run_path(phase,seed);meta=json.loads((run/"run_meta.json").read_text()) if (run/"run_meta.json").exists() else {}
            a=json.loads((run/"acceptance.json").read_text()) if (run/"acceptance.json").exists() else {}
            row[phase+"_status"]="completed" if a.get("accepted") else meta.get("status","pending")
            stage=state["stages"].get(f"{phase}-s{seed}",{})
            if stage.get("status")=="partial":row[phase+"_status"]="partial"
            elif meta.get("status")=="completed" and not a.get("accepted"):row[phase+"_status"]="needs_validation"
            events=[v for v in rows(run/"metrics.jsonl") if v.get("event")=="evaluation"]
            row[phase+"_final_score"]=events[-1]["normalized_score_100"] if a.get("accepted") and events else None
            if phase=="online":
                row["online_auc"]=sum((x["normalized_score_100"]+y["normalized_score_100"])*.5*(y["env_steps"]-x["env_steps"]) for x,y in zip(events,events[1:]))/100000 if a.get("accepted") else None
            if (phase,seed) in PLAN and run.exists():
                label=f"{run.parent.name}/{run.name}";curves(run,phase,label,seed)
                processed=RESEARCH/"results/processed"/label
                for name in ("run_meta.json","state.json","acceptance.json","initialization_manifest.json","policy_reload.json","effective_recipe.json","checkpoint_manifest.json"):
                    if (run/name).exists():shutil.copy2(run/name,processed/name)
        table.append(row)
    for key in ("offline_final_score","online_final_score","online_auc"):
        values=[row[key] for row in table if row[key] is not None]
        aggregate[key]={"completed_n":len(values),"planned_n":5,"mean":statistics.mean(values) if len(values)==5 else None,
                        "sample_sd":statistics.stdev(values) if len(values)==5 else None}
    atomic_json(evidence/"summary.json",{"rows":table,"five_seed_statistics":aggregate})
    with (evidence/"summary.csv").open("w") as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    budget={p:{"limit_seconds":LIMITS[p],"used_seconds":used_seconds(state,p),"remaining_seconds":LIMITS[p]-used_seconds(state,p)} for p in LIMITS}
    atomic_json(evidence/"budget.json",budget)
    def fmt(value):return "—" if value is None else f"{value:.3f}"
    lines=[f"| {r['seed']} | {r['offline_status']} | {fmt(r['offline_final_score'])} | {r['online_status']} | {fmt(r['online_final_score'])} | {fmt(r['online_auc'])} |" for r in table]
    report=RESEARCH/"reports/so2/SO2-T010-multiseed.md"
    report.write_text(f"""# SO2-T010：五条独立两阶段重复

## 最新状态

执行状态：`{state['status']}`，当前 `{state.get('current_stage')}`。更新时间：{datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()}。技术验收逐run记录；研究验收待定。用户2026-10-12明确新的服务器窗口没有时间限制，取代此前9:00截止，仅保留任务实验预算。

已有seed0两阶段与seed1离线不重跑。本队列只执行offline2–4、online1–4；失败保留partial并停止后续，无自动重试。配对(0,0)..(4,4)，所有parent固定3M final，绝不按得分挑选。

## 结果

| Seed | 离线状态 | 离线final score | 在线状态 | 在线100k score | 在线AUC/100k |
|---|---|---:|---|---:|---:|
"""+"\n".join(lines)+f"""

表中score为D4RL normalized score×100，AUC横轴为在线训练env steps。缺失标为—，未满5条不报告五种子均值。完整均值/样本标准差(ddof=1)及单条数据见[summary.json](SO2-T010-multiseed/summary.json)和[CSV](SO2-T010-multiseed/summary.csv)。这五条重复同时改变offline/online随机性，不能分解两者方差；20评价episodes不是20次训练重复。

## 配方与预算

N2、3×256、LN、Dropout .005、actor bounds[-20,2]、actor min；离线EDAC eta1/lr3e-4/3M，在线SO2 100k。数学实现、数据和配方保持既有版本；只扩展seed、parent身份和执行/交付。每个新parent在线前必须通过实际BaseLearner→LoadCkptHook→read_file→policy strict加载和固定RNG train/eval连续性零更新验证。full parent先校验冻结哈希、acceptance、代码/数据/recipe身份，再导出纯张量；两个加载路径使用导出，不关闭weights_only。

新增离线总上限48h（含seed1已用{state['prior_offline_wall_seconds']:.3f}秒），在线每次8h/合计32h。已结束阶段耗时（含失败）见[budget.json](SO2-T010-multiseed/budget.json)，运行中当前阶段耗时另见queue_state，不把未结算消耗当成免费预算。预检累计181/200，本次及逐parent验证0更新。预算到期保存partial并停止，不缩短3M/100k的正式完成定义。

## 身份与证据

代码 `{state['code_sha']}`，SO2分支codex/so2-t010-pretrain；核心数学实现固定f5dacc442f73e94e085988600fa2ff8cb8ad95b0（SO2 tree和t010_edac.py对象相同）。科研分支codex/so2-t010-delivery，不合并main。源任务见[扩展任务单](../../docs/tasks/SO2-T010-multiseed-extension.md)。

精确命令、阶段PID/退出码/墙钟、资源、冻结parent记录、验证在[queue_state.json](SO2-T010-multiseed/queue_state.json)及同名目录。Linux完整日志：`{WORK}`；run目录：`{ROOT}/_so2_work/runs/EXP-027`和EXP-028。raw阶段终止后只封存一次，processed/figures每run单独目录。大checkpoint仅Linux一份，哈希不等于异地备份。旧EXP-026 attempt1失败保留，不重跑seed0。异常：`{state.get('error','无')}`。
""")
    tracker=RESEARCH/"docs/EXPERIMENT_TRACKER.md";text=tracker.read_text();lines=text.splitlines()
    for exp,phase in (("EXP-027","offline"),("EXP-028","online")):
        count=sum(r[phase+"_status"]=="completed" for r in table[1:])
        row=f"| {exp} | [T010扩展](tasks/SO2-T010-multiseed-extension.md) | {state['status']}；{count}/4完成 | 独立配对seeds1–4；{phase}；固定final，无筛选 | {'各3M；合计48h' if phase=='offline' else '各100k；各8h/合计32h'} |"
        found=False
        for i,line in enumerate(lines):
            if line.startswith(f"| {exp} |"):lines[i]=row;found=True
        if not found:lines.extend(["",row])
    tracker.write_text("\n".join(lines)+"\n")
    paths=["reports/so2/SO2-T010-multiseed.md","reports/so2/SO2-T010-multiseed","docs/EXPERIMENT_TRACKER.md"]
    for tier in ("raw","processed","figures"):
        for exp in ("EXP-027","EXP-028"):
            path=f"results/{tier}/{exp}"
            if (RESEARCH/path).exists():paths.append(path)
    subprocess.run(["git","add",*paths],cwd=RESEARCH,check=True,timeout=60)
    if subprocess.run(["git","diff","--cached","--quiet"],cwd=RESEARCH).returncode:
        subprocess.run(["git","commit","-m",f"SO2-T010: multiseed {state['status']} {state.get('current_stage')}"],cwd=RESEARCH,check=True,timeout=60)
    subprocess.run(["git","push","origin","codex/so2-t010-delivery"],cwd=RESEARCH,check=True,timeout=90)
    state["research_sha"]=subprocess.check_output(["git","rev-parse","HEAD"],cwd=RESEARCH,text=True).strip();atomic_json(STATE,state)


def watch(state,key,phase,seed,command,budget):
    start=time.monotonic();state["current_stage"]=key
    cmd=["/usr/bin/timeout","--signal=TERM","--kill-after=5s",f"{budget-10:.3f}s",*command]
    entry={"phase":phase,"seed":seed,"command":cmd,"start_unix":time.time(),"budget_seconds":budget,"status":"running"}
    state["active"]=entry;atomic_json(STATE,state)
    child=None;last_delivery=time.monotonic()
    try:
        with (WORK/f"{key}.log").open("x") as log:
            child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            entry["pid"]=child.pid;atomic_json(STATE,state)
            while child.poll() is None:
                assert shutil.disk_usage(ROOT).free>20*1024**3,"disk reserve exhausted"
                entry["wall_seconds"]=time.monotonic()-start
                if phase=="offline" and (run_path(phase,seed)/"state.json").exists():entry["progress"]=json.loads((run_path(phase,seed)/"state.json").read_text())
                if phase=="online":
                    records=rows(run_path(phase,seed)/"metrics.jsonl");states=[r for r in records if r.get("event")=="evaluation_state"]
                    if states:entry["progress"]={k:states[-1][k] for k in ("env_steps","updates")}
                    first=next((r for r in states if r["env_steps"]==7500),None)
                    if first:
                        assert first["updates"]=={"critic":25000,"actor":2500,"alpha":0}
                        atomic_json(run_path(phase,seed)/"first_window.json",{"passed":True,"env_steps":7500,"updates":first["updates"]})
                atomic_json(STATE,state)
                if time.monotonic()-last_delivery>=21600:deliver(state);last_delivery=time.monotonic()
                time.sleep(5)
            assert child.returncode==0,f"{key} exited {child.returncode}; no retry"
            entry["status"]="completed"
    except BaseException:
        entry["status"]="partial"
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        raise
    finally:
        entry.update(wall_seconds=time.monotonic()-start,exit_code=None if child is None else child.returncode,end_unix=time.time())
        state["stages"][key]=entry;state.pop("active",None);atomic_json(STATE,state)


def main():
    assert not WORK.exists(),"queue exists; no automatic restart"
    for phase,seed in PLAN:assert not run_path(phase,seed).exists(),"formal run exists"
    sha=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    assert not subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip()
    assert subprocess.check_output(["git","ls-remote","origin","refs/heads/codex/so2-t010-pretrain"],cwd=ROOT,text=True).split()[0]==sha
    validation=json.loads((VALIDATION/"validation.json").read_text())
    assert validation["status"]=="passed" and validation["code_sha"]==sha and not validation["git_dirty"] and validation["updates"]==0
    prior=json.loads((ROOT/"_so2_work/t010-seed1-20261012/queue_state.json").read_text());assert prior["status"]=="completed"
    for phase,seed in (("offline",0),("online",0),("offline",1)):
        assert json.loads((run_path(phase,seed)/"acceptance.json").read_text())["accepted"]
    with LOCK.open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True).strip()
        WORK.mkdir();lock.seek(0);lock.truncate();lock.write(str(os.getpid()));lock.flush()
        state={"status":"running","code_sha":sha,"thread_id":THREAD,"queue_pid":os.getpid(),"started_unix":time.time(),
               "prior_offline_wall_seconds":prior["wall_seconds"],"stages":{},"plan":PLAN,"auto_retry":False,
               "user_time_window":"2026-10-12 user confirmed no time limit; experiment budgets still apply","preflight_updates_cumulative":181}
        atomic_json(STATE,state)
        def interrupt(signum,frame):raise RuntimeError(f"queue signal {signum}; stop remaining stages")
        signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
        try:
            deliver(state)
            for phase,seed in PLAN:
                key=f"{phase}-s{seed}";budget=budget_for(state,phase);run=run_path(phase,seed)
                cmd=[PYTHON,str(ROOT/"scripts/repro/t010_run.py"),"--phase",phase,"--seed",str(seed),"--exp-id",run.parent.name,"--run-dir",str(run)]
                if phase=="offline":cmd += ["--stop-at",str(time.time()+budget-30)]
                else:
                    parent,record=freeze_parent(seed,sha)
                    hook_out=WORK/f"hook-s{seed}"
                    hook=[PYTHON,str(ROOT/"scripts/repro/t010_loading.py"),"--out",str(hook_out),"--parent",str(parent),"--parent-record",str(record),"--seed",str(seed)]
                    watch(state,f"hook-s{seed}","validation",seed,hook,120)
                    result=json.loads((hook_out/"validation.json").read_text());assert result["status"]=="passed" and result["fixed_batch_updates"]==0
                    cmd += ["--parent",str(parent),"--parent-record",str(record),"--wall-seconds",str(budget-30)]
                watch(state,key,phase,seed,cmd,budget)
                acceptance=json.loads((run/"acceptance.json").read_text());assert acceptance["accepted"]
                state["stages"][key]["acceptance"]=acceptance;archive(run);deliver(state)
            state.update(status="completed",finished_unix=time.time(),current_stage="completed");deliver(state)
        except BaseException as exc:
            state.update(status="partial",error=repr(exc),traceback=traceback.format_exc(),finished_unix=time.time())
            atomic_json(STATE,state)
            for phase,seed in PLAN:
                run=run_path(phase,seed);raw=RESEARCH/"results/raw"/run.parent.name/run.name
                if run.exists() and not raw.exists():archive(run)
            try:deliver(state)
            except BaseException as error:state["delivery_error"]=repr(error);atomic_json(STATE,state)
            raise


if __name__=="__main__":main()
