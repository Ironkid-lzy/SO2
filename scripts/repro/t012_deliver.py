"""Immutable small raw, complete curves and paired comparisons to frozen controls."""
import csv,math,statistics,time
from datetime import datetime
from zoneinfo import ZoneInfo
from t012_common import *
from t010_run import rows

def csv_write(path,records):
    if not records:return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w") as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)

def control(seed,phase):
    if phase=="online":sha=CONTROL_SHA;exp=29;attempt=1
    else:sha=OFF_CONTROL_SHA;exp=25 if seed==0 else 27;attempt=1
    rel=f"results/raw/EXP-{exp:03}/EXP-{exp:03}-s{seed}-attempt{attempt}/metrics.jsonl"
    text=subprocess.check_output(["git","show",f"{sha}:{rel}"],cwd=RESEARCH,text=True)
    return [e for e in map(json.loads,text.splitlines()) if e.get("event")=="evaluation"]

def events(run):return [e for e in rows(run/"metrics.jsonl") if e.get("event")=="evaluation"]
def metrics(ev,phase):
    key="gradient_updates" if phase=="offline" else "env_steps";n=31 if phase=="offline" else 41
    end=3000000 if phase=="offline" else 100000;interval=100000 if phase=="offline" else 2500
    assert len(ev)==n and [e[key] for e in ev]==list(range(0,end+1,interval))
    assert all(len(e["returns"])==20 and all(math.isfinite(v) for v in e["returns"]) for e in ev)
    return {"final":ev[-1]["normalized_score_100"],"auc":sum((a["normalized_score_100"]+b["normalized_score_100"])*.5*(b[key]-a[key]) for a,b in zip(ev,ev[1:]))/end}

def seal(group,seed,phase):
    run=run_path(group,seed,phase);raw=RESEARCH/"results/raw"/run.parent.name/run.name
    assert not raw.exists(),"raw already sealed"
    files=[{"path":str(p.relative_to(run)),"size":p.stat().st_size,"sha256":digest(p)} for p in sorted(run.rglob("*")) if p.is_file() and p.suffix!=".tmp"]
    raw.mkdir(parents=True)
    for p in sorted(run.iterdir()):
        if p.is_file() and (p.suffix in (".json",".jsonl") or p.name in ("total_config.py","formatted_total_config.py")):
            shutil.copy2(p,raw/p.name);assert digest(p)==digest(raw/p.name)
    if (run/"hook/validation.json").exists():shutil.copy2(run/"hook/validation.json",raw/"hook-validation.json")
    atomic_json(raw/"artifact_manifest.json",{"linux_path":str(run),"files":files,"large_artifact_backup":"Linux only; no verified offsite backup"})

def deliver(state):
    evidence=RESEARCH/"reports/so2/SO2-T012";evidence.mkdir(parents=True,exist_ok=True)
    atomic_json(evidence/"queue_state.json",state)
    for p in VALIDATION.iterdir():
        if p.is_file() and (p.suffix in (".json",".md") or p.name=="requirements-lock.txt"):shutil.copy2(p,evidence/p.name)
    summaries=[];aggregates={};completed=0;allcurves=[]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for group in ("dropout","ln"):
      for phase in ("offline","online"):
        rows_summary=[];curves=[];full={};key="gradient_updates" if phase=="offline" else "env_steps"
        exp=SETTINGS[group][2 if phase=="offline" else 3];processed=RESEARCH/f"results/processed/EXP-{exp:03}";figures=RESEARCH/f"results/figures/EXP-{exp:03}";figures.mkdir(parents=True,exist_ok=True)
        for seed in range(5):
            run=run_path(group,seed,phase);ev=events(run);base=control(seed,phase);b=metrics(base,phase)
            a=json.loads((run/"acceptance.json").read_text()) if (run/"acceptance.json").exists() else {}
            m=metrics(ev,phase) if a.get("accepted") else None
            completed+=int(m is not None)
            stage=state["stages"].get(f"{group}-s{seed}-{phase}",{})
            r={"group":group,"phase":phase,"seed":seed,"status":stage.get("status","pending"),"control_final":b["final"],"final":m["final"] if m else None,"final_difference":m["final"]-b["final"] if m else None,"control_auc":b["auc"] if phase=="online" else None,"auc":m["auc"] if m and phase=="online" else None,"auc_difference":m["auc"]-b["auc"] if m and phase=="online" else None,"wall_seconds":stage.get("wall_seconds")}
            rows_summary.append(r);summaries.append(r);full[seed]=(base,ev)
            for label,es in (("combined",base),(group,ev)):
                for e in es:curves.append({"seed":seed,"group":label,"phase":phase,"step":e[key],"score":e["normalized_score_100"],"return_mean":e["return_mean"],"evaluation_env_steps":e["evaluation_env_steps"]})
            if ev:csv_write(processed/run.name/"curves.csv",[c for c in curves if c["seed"]==seed])
            if ev:
                fig,ax=plt.subplots(figsize=(6.5,3.8))
                for label,es in (("LN + Dropout",base),(group,ev)):ax.plot([e[key] for e in es],[e["normalized_score_100"] for e in es],label=label)
                ax.set(xlabel="Offline gradient updates" if phase=="offline" else "Online training environment steps",ylabel="D4RL normalized score (x100)",title=f"T012 {group}, {phase}, seed {seed}");ax.legend();ax.grid(alpha=.2);fig.tight_layout()
                for ext in ("png","pdf"):fig.savefig(figures/f"seed-{seed}.{ext}")
                plt.close(fig)
        csv_write(processed/"summary.csv",rows_summary);csv_write(processed/"curves.csv",curves);allcurves+=curves
        if all(r["final"] is not None for r in rows_summary):
            agg={}
            for k in ("control_final","final","final_difference",*(("control_auc","auc","auc_difference") if phase=="online" else ())):
                values=[r[k] for r in rows_summary];agg[k]={"mean":statistics.mean(values),"sample_sd":statistics.stdev(values),"n":5}
            agg["final_improved_seeds"]=sum(r["final_difference"]>0 for r in rows_summary)
            if phase=="online":agg["auc_improved_seeds"]=sum(r["auc_difference"]>0 for r in rows_summary)
            aggregates[f"{group}-{phase}"]=agg
            fig,axes=plt.subplots(1,2,figsize=(11,4));gc=[]
            for seed,(_,ev) in full.items():axes[0].plot([e[key] for e in ev],[e["normalized_score_100"] for e in ev],label=f"seed {seed}")
            for label,idx in (("combined",0),(group,1)):
                es=full[0][idx];x=[e[key] for e in es]
                mu=[statistics.mean(full[s][idx][i]["normalized_score_100"] for s in range(5)) for i in range(len(es))]
                sd=[statistics.stdev(full[s][idx][i]["normalized_score_100"] for s in range(5)) for i in range(len(es))]
                axes[1].plot(x,mu,label=label);axes[1].fill_between(x,[a-b for a,b in zip(mu,sd)],[a+b for a,b in zip(mu,sd)],alpha=.15)
                gc += [{"group":label,"step":step,"mean":m,"sample_sd":s,"n":5} for step,m,s in zip(x,mu,sd)]
            for ax in axes:ax.legend();ax.grid(alpha=.2);ax.set(xlabel="Offline gradient updates" if phase=="offline" else "Online training environment steps",ylabel="D4RL score (x100)")
            fig.tight_layout()
            for ext in ("png","pdf"):fig.savefig(figures/f"group.{ext}")
            plt.close(fig);csv_write(processed/"group-curves.csv",gc)
    atomic_json(evidence/"summary.json",{"rows":summaries,"aggregate":aggregates,"completed_runs":completed,"planned_runs":20})
    csv_write(evidence/"paired-summary.csv",summaries)
    budget={}
    for group in ("dropout","ln"):
      for phase in ("offline","online"):
        budget[f"{group}-{phase}"]={"limit_seconds":172800 if phase=="offline" else 86400,"used_seconds":sum(s["wall_seconds"] for s in state["stages"].values() if s["group"]==group and s["phase"]==phase),"active_wall_seconds":state.get("active",{}).get("wall_seconds",0) if state.get("active",{}).get("group")==group and state["active"]["phase"]==phase else 0,"failed_attempts_counted":True}
    atomic_json(evidence/"budget.json",budget)
    def fmt(v):return "—" if v is None else f"{v:.3f}"
    table="\n".join(f"| {r['group']} | {r['phase']} | {r['seed']} | {r['status']} | {fmt(r['control_final'])} | {fmt(r['final'])} | {fmt(r['final_difference'])} | {fmt(r['auc'])} | {fmt(r['auc_difference'])} |" for r in summaries)
    aggregate_text="\n".join(f"- {label}：final差 {a['final_difference']['mean']:.3f} ± {a['final_difference']['sample_sd']:.3f}（sample SD）；5 seeds。"+(f"AUC差 {a['auc_difference']['mean']:.3f} ± {a['auc_difference']['sample_sd']:.3f}。" if 'auc_difference' in a else '') for label,a in aggregates.items())
    report=f"""# SO2-T012：仅Dropout → 仅LayerNorm两阶段消融

状态：**{state['status']}**，{completed}/20个正式run完成；研究验收待定。更新：{datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()}。

{aggregate_text or '尚未完成五seed整组，暂无组级研究结论。'}

| 组 | 阶段 | seed | 状态 | 组合final | 本组final | 差值 | 本组在线AUC | AUC差 |
|---|---|---:|---|---:|---:|---:|---:|---:|
{table}

主要指标为在线100k final，次要0–100k AUC/100000，离线3M final为补充。均为D4RL normalized score×100；所有五seed原始returns保留，不取best checkpoint，不平滑、不删低分。每点评价20episodes不是20条训练重复。完整CSV/PNG/PDF见results/processed及results/figures/EXP-030…033；[汇总及配对差](SO2-T012/summary.json)、[预算](SO2-T012/budget.json)、[队列](SO2-T012/queue_state.json)。

两组分别从随机初始化产生5个新parent；按同seed标签与组合对照配对，未共用parent，不能分解两阶段方差。本轮只估计LN在Dropout背景、Dropout在LN背景的增量；缺少双关闭同协议组，不能完整识别交互、唯一机制或把新增两组直接差值叫单因素效应。EXP-015/017不补此缺口，退役迁移组不参与。组序由用户固定，执行日期/负载可能不同，墙钟不作严格加速比。

冻结计划 {PLAN_SHA}；算法 Ironkid-lzy/SO2，codex/so2-t012-component-ablation@{state['code_sha']}（基准{BASE_SHA}）；科研分支codex/so2-t012-delivery。组合对照T010科研{OFF_CONTROL_SHA} + T011科研{CONTROL_SHA}；T011 seed0训练87f0ba67db5abbf71529614bdabba70dab950074，seeds1–4训练90a67de1c0ca94deb983211425f4f819d111dc63，身份不重记。

固定2Q、3×256、同线性初始化/ReLU、actor[-20,2]。offline min/EDAC eta1/auto-alpha/batch256/lr3e-4/target min/3M；online mean/SO2逐head target/PVU .3±.6/UTD10/固定alpha.2/batch2560/lr.001/100k含5k预热。每online run 950k critic/95k actor/0alpha。每offline31点、online41点、每点20episodes；评价与训练RNG隔离，不回流buffer。完整配置/输入数组hash/parent及export hash见各raw run_meta和initialization manifest。

资源和预检证据在同名目录；依赖复用SO2/.venv，未升级。每组offline≤48h且单run≤10h，online≤24h且单run≤5h；失败计入原预算，无自动重试/调参。单全局锁监督非有限、身份、计数、退出码、磁盘和重复计算进程。当前异常：{state.get('error','无')}。

执行聊天{THREAD}。大checkpoint和完整日志仅Linux {ROOT}/_so2_work，尚无已验证异地备份；raw的artifact_manifest记录路径/大小/hash。实际命令、时间、PID、退出码及计数见队列和逐run metadata。正式源码冻结，不热改；完成或失败后暂停本聊天heartbeat，不跨聊天发消息、不启动后续任务。
"""
    (RESEARCH/"reports/so2/SO2-T012.md").write_text(report)
    index=RESEARCH/"reports/so2/README.md"
    if "SO2-T012.md" not in index.read_text():
        with index.open("a") as f:f.write("\n- [T012：仅Dropout → 仅LayerNorm](SO2-T012.md)：EXP-030…033；状态见报告。\n")
    paths=["reports/so2/SO2-T012.md","reports/so2/SO2-T012","reports/so2/README.md"]
    for tier in ("raw","processed","figures"):
      for exp in range(30,34):
        rel=f"results/{tier}/EXP-{exp:03}"
        if (RESEARCH/rel).exists():paths.append(rel)
    subprocess.run(["git","add",*paths],cwd=RESEARCH,check=True)
    if subprocess.run(["git","diff","--cached","--quiet"],cwd=RESEARCH).returncode:subprocess.run(["git","commit","-m",f"SO2-T012: {state['status']}, {completed}/20 runs sealed"],cwd=RESEARCH,check=True)
    state["research_sha"]=subprocess.check_output(["git","rev-parse","HEAD"],cwd=RESEARCH,text=True).strip()
    try:
        subprocess.run(["git","push","origin","codex/so2-t012-delivery"],cwd=RESEARCH,check=True,timeout=60);state["delivery_status"]="pushed"
    except Exception as exc:state["delivery_status"]="local_committed";state["delivery_error"]=repr(exc)
    atomic_json(WORK/"queue_state.json",state)
