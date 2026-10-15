"""Immutable per-seed sealing and paired results; independent training seeds."""
import csv
import re
import math
import shutil
import statistics
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo
from t011_common import *
from t010_deliver import rows

def control_events(seed):
    run=control_run(seed)
    path=f"results/raw/{run.parent.name}/{run.name}/metrics.jsonl"
    text=subprocess.check_output(["git","show",f"{CONTROL_SHA}:{path}"],cwd=RESEARCH,text=True)
    return [v for v in map(json.loads,text.splitlines()) if v.get("event")=="evaluation"]

def events(seed):
    return [v for v in rows(run_path(seed)/"metrics.jsonl") if v.get("event")=="evaluation"]

def stats(ev):
    if len(ev)!=41 or ev[-1]["env_steps"]!=100000:return None
    assert [e["env_steps"] for e in ev]==list(range(0,100001,2500))
    assert all(len(e["returns"])==20 and all(math.isfinite(v) for v in e["returns"]) for e in ev)
    return {"final":ev[-1]["normalized_score_100"],
            "auc":sum((x["normalized_score_100"]+y["normalized_score_100"])*.5*(y["env_steps"]-x["env_steps"]) for x,y in zip(ev,ev[1:]))/100000}

def seal(seed):
    run=run_path(seed);raw=RESEARCH/"results/raw/EXP-029"/run.name
    assert not raw.exists(),"raw already sealed"
    checkpoint=[{"path":str(p.relative_to(run)),"size":p.stat().st_size,"sha256":digest(p),"backup_status":"Linux only"}
                for folder in ("ckpt","partial") for p in sorted((run/folder).glob("*")) if p.is_file() and p.suffix!=".tmp"]
    atomic_json(run/"checkpoint_manifest.json",checkpoint)
    raw.mkdir(parents=True)
    for name in ("run_meta.json","metrics.jsonl","initialization_manifest.json","stage_continuity.json","policy_reload.json","acceptance.json","effective_recipe.json","checkpoint_manifest.json","first_window.json","learner_guard.json","total_config.py","formatted_total_config.py"):
        if (run/name).exists():shutil.copy2(run/name,raw/name)
    files=[{"path":str(p.relative_to(run)),"size":p.stat().st_size,"sha256":digest(p)} for p in sorted(run.rglob("*")) if p.is_file() and p.suffix!=".tmp"]
    atomic_json(raw/"artifact_manifest.json",{"linux_path":str(run),"files":files,"large_artifact_backup":"Linux only"})

def csv_write(path,records):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w") as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)

def deliver(state):
    evidence=RESEARCH/"reports/so2/SO2-T011";evidence.mkdir(parents=True,exist_ok=True)
    atomic_json(evidence/"queue_state.json",state)
    for name in ("validation.json","resources.json","requirements-lock.txt","execution-manifest.json","modification-analysis.md","preflight-ledger.json","queue-validation.json","runner-contract.json"):
        src=VALIDATION/name
        if src.exists():shutil.copy2(src,evidence/name)
    for seed in range(5):
        src=VALIDATION/f"hook-s{seed}/validation.json"
        if src.exists():shutil.copy2(src,evidence/f"hook-s{seed}.json")
    summary=[];curve=[];full={}
    for seed in range(5):
        control=control_events(seed); treatment=events(seed)
        base=stats(control);assert base is not None
        run=run_path(seed);a=json.loads((run/"acceptance.json").read_text()) if (run/"acceptance.json").exists() else {}
        current=stats(treatment) if a.get("accepted") else None
        status=state["stages"].get(f"s{seed}",{}).get("status","pending")
        if state.get("current_seed")==seed and state["status"]=="running":status="running"
        row={"seed":seed,"status":status,"min_final":base["final"],"mean_final":current["final"] if current else None,
             "final_difference":current["final"]-base["final"] if current else None,
             "min_auc":base["auc"],"mean_auc":current["auc"] if current else None,"auc_difference":current["auc"]-base["auc"] if current else None}
        summary.append(row)
        processed=RESEARCH/f"results/processed/EXP-029/{run.name}";processed.mkdir(parents=True,exist_ok=True)
        atomic_json(processed/"evaluation_rows.json",treatment)
        for name in ("run_meta.json","acceptance.json","initialization_manifest.json","effective_recipe.json","checkpoint_manifest.json","policy_reload.json"):
            if (run/name).exists():shutil.copy2(run/name,processed/name)
        for label,ev in (("min",control),("mean",treatment)):
            for e in ev:curve.append({"seed":seed,"aggregation":label,"env_steps":e["env_steps"],"return_mean":e["return_mean"],"score":e["normalized_score_100"],"evaluation_env_steps":e["evaluation_env_steps"]})
        full[seed]=(control,treatment)
    n=sum(r["mean_final"] is not None for r in summary)
    aggregate={}
    if n==5:
        for key in ("min_final","mean_final","final_difference","min_auc","mean_auc","auc_difference"):
            values=[r[key] for r in summary]
            aggregate[key]={"mean":statistics.mean(values),"sample_sd":statistics.stdev(values),"n":5}
        aggregate["final_improved_seeds"]=sum(r["final_difference"]>0 for r in summary)
        aggregate["auc_improved_seeds"]=sum(r["auc_difference"]>0 for r in summary)
    atomic_json(evidence/"summary.json",{"rows":summary,"aggregate":aggregate,"completed_n":n,"planned_n":5})
    processed=RESEARCH/"results/processed/EXP-029"
    csv_write(processed/"paired-summary.csv",summary);csv_write(processed/"curves.csv",curve)
    budget={"limit_seconds":86400,"per_run_limit_seconds":18000,"used_seconds":sum(v["wall_seconds"] for v in state["stages"].values()),
            "active_wall_seconds":state.get("active",{}).get("wall_seconds",0),"failed_attempts_counted":True}
    atomic_json(evidence/"budget.json",budget)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures=RESEARCH/"results/figures/EXP-029";figures.mkdir(parents=True,exist_ok=True)
    for seed,(control,treatment) in full.items():
        fig,ax=plt.subplots(figsize=(6.5,3.8))
        for label,ev in (("min control",control),("mean",treatment)):
            if ev:ax.plot([e["env_steps"] for e in ev],[e["normalized_score_100"] for e in ev],label=label)
        ax.axvspan(0,5000,color="gold",alpha=.1);ax.legend()
        ax.set(xlabel="Training environment steps",ylabel="D4RL normalized score (x100)",title=f"T011 paired seed {seed}")
        ax.grid(alpha=.2);fig.tight_layout()
        fig.savefig(figures/f"seed-{seed}.png",dpi=160);fig.savefig(figures/f"seed-{seed}.pdf");plt.close(fig)
    if n==5:
        fig,ax=plt.subplots(figsize=(7,4))
        group=[]
        for label,idx in (("min",0),("mean",1)):
            x=[e["env_steps"] for e in full[0][idx]]
            mu=[statistics.mean(full[s][idx][i]["normalized_score_100"] for s in range(5)) for i in range(41)]
            sd=[statistics.stdev(full[s][idx][i]["normalized_score_100"] for s in range(5)) for i in range(41)]
            ax.plot(x,mu,label=label);ax.fill_between(x,[a-b for a,b in zip(mu,sd)],[a+b for a,b in zip(mu,sd)],alpha=.15)
            group += [{"aggregation":label,"env_steps":step,"mean":m,"sample_sd":d,"n":5} for step,m,d in zip(x,mu,sd)]
        ax.set(xlabel="Training environment steps",ylabel="D4RL normalized score (x100)",title="T011: 5 paired seeds, mean +/- sample SD")
        ax.legend();ax.grid(alpha=.2);fig.tight_layout()
        fig.savefig(figures/"group.png",dpi=180);fig.savefig(figures/"group.pdf");plt.close(fig)
        csv_write(processed/"group-curves.csv",group)
    def fmt(v):return "—" if v is None else f"{v:.3f}"
    table="\n".join(f"| {r['seed']} | {r['status']} | {fmt(r['min_final'])} | {fmt(r['mean_final'])} | {fmt(r['final_difference'])} | {fmt(r['min_auc'])} | {fmt(r['mean_auc'])} | {fmt(r['auc_difference'])} |" for r in summary)
    conclusion="未满五seed，尚不能回答本组研究问题。"
    if n==5:
        f=aggregate["final_difference"];auc=aggregate["auc_difference"]
        conclusion=f"100k final的mean−min配对差为{f['mean']:.3f} ± {f['sample_sd']:.3f}（sample SD），{aggregate['final_improved_seeds']}/5个seed改善；AUC差为{auc['mean']:.3f} ± {auc['sample_sd']:.3f}，{aggregate['auc_improved_seeds']}/5改善。"
        if f["mean"]>0:conclusion+="在固定五个parent与当前SO2协议下，mean的平均终点更高，支持actor的min约束可能限制策略改善。"
        else:conclusion+="本组平均终点没有改善，不支持取消min约束提高100k终点的假设。"
        conclusion+=" n=5未作显著性声明；不单独证明真实Q低估或Dropout机制。"
    report=f"""# SO2-T011：固定五个parent的actor min→mean

执行状态：{state['status']}；{n}/5正式run完成；技术验收逐run记录，研究验收待定。更新：{datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()}。

{conclusion}

| Seed | 状态 | min 100k | mean 100k | mean−min | min AUC | mean AUC | AUC差 |
|---:|---|---:|---:|---:|---:|---:|---:|
{table}

分数为D4RL normalized score×100；AUC为0–100k原始曲线梯形积分/100000（含5k预热），不取best checkpoint。统计单位为五个(pretrain_seed,online_seed)配对重复，每点20episodes不是20次训练重复。数据见[配对CSV](../../results/processed/EXP-029/paired-summary.csv)、[曲线CSV](../../results/processed/EXP-029/curves.csv)、[逐seed与组图PNG/PDF](../../results/figures/EXP-029/)。

唯一数学改动：actor对同一次Q forward沿ensemble轴dim=0取mean；默认min保持。2Q、3×256、LN、p=.005、actor/TD/target的Dropout、逐head target、PVU、UTD10、critic:actor10:1、固定alpha.2、lr.001、2560batch与两buffer规则、评价RNG和episode终止语义均沿用T010。完整配置和生效recipe随每个run保存。

每run100k训练步含5k预热，950k critic/95k actor/0alpha；五run合计500k训练步，评价交互另计。评价0及每2500步20episodes共41点。每run≤5h，训练累计≤24h（失败计入）；预检≤100固定batch更新/30分钟GPU/0环境交互。真实计数与预算见[queue_state](SO2-T011/queue_state.json)、[budget](SO2-T011/budget.json)、各raw acceptance。

计划v1：{PLAN_SHA}；历史min科研交付：{CONTROL_SHA}。算法：Ironkid-lzy/SO2，codex/so2-t011-actor-mean@{state['code_sha']}；冻结基准{BASE_SHA}。科研：codex/so2-t011-delivery。实际learner hook、parent/export哈希、train/eval连续性、min loss/梯度/更新权重/RNG、mean解析梯度见[validation](SO2-T011/validation.json)与hook-s0…s4；依赖逐行一致，未升级。

执行聊天：{THREAD}。远端队列持全局训练锁，实时检查有限性/身份/更新计数/磁盘/超时；异常封存partial并停止后续，不自动重试、不按得分淘汰。当前异常：{state.get('error','无')}。大checkpoint与完整日志仅Linux：{ROOT}/_so2_work；小型raw不可变，每seed结束封存一次并提交。代码启动前固定，不热改，不合并main，不推官方upstream。
"""
    if state.get("reused_seed0_sha"):
        report += f"\n续接记录：seed0实际训练SHA为{state['reused_seed0_sha']}，本次seeds1–4为{state['code_sha']}；只改执行器和报告元数据，训练/加载/配置对象与seed0版一致。旧失败队列保留于{state['history_queue_path']}，本续接状态目录为{state['queue_workdir']}。seed0不重跑、raw不改，累计预算含seed0墙钟。见[历史故障与修复](SO2-T011/report-before-schema-repair.md.txt)、[续接验证](SO2-T011/resume-v1/validation.json)。\n"
    (RESEARCH/"reports/so2/SO2-T011.md").write_text(report)
    tracker=RESEARCH/"docs/EXPERIMENT_TRACKER.md";text=tracker.read_text()
    if "## EXP-029" in text:
        start=text.index("## EXP-029"); end=text.find("\n## ",start+1)
        if end<0:end=len(text)
        section=re.sub(r"\*\*状态：[^\n]*?\*\*",f"**状态：{state['status']}，{n}/5完成，待研究验收。**",text[start:end],count=1)
        text=text[:start]+section+text[end:]
    tracker.write_text(text)
    index=RESEARCH/"reports/so2/README.md"
    if "SO2-T011.md" not in index.read_text():
        with index.open("a") as f:f.write("\n- [T011：固定parent的actor min→mean](SO2-T011.md)：EXP-029，状态见报告。\n")
    manifest=[{"path":str(p.relative_to(RESEARCH)),"size":p.stat().st_size,"sha256":digest(p)}
              for base in (evidence,processed,figures) for p in sorted(base.rglob("*")) if p.is_file() and p.name!="delivery-manifest.json"]
    atomic_json(evidence/"delivery-manifest.json",manifest)
    paths=["reports/so2/SO2-T011.md","reports/so2/SO2-T011","reports/so2/README.md","docs/EXPERIMENT_TRACKER.md","results/processed/EXP-029","results/figures/EXP-029"]
    if (RESEARCH/"results/raw/EXP-029").exists():paths.append("results/raw/EXP-029")
    subprocess.run(["git","add",*paths],cwd=RESEARCH,check=True)
    if subprocess.run(["git","diff","--cached","--quiet"],cwd=RESEARCH).returncode:
        subprocess.run(["git","commit","-m",f"SO2-T011: {state['status']}, {n}/5 paired runs sealed"],cwd=RESEARCH,check=True)
    state["research_sha"]=subprocess.check_output(["git","rev-parse","HEAD"],cwd=RESEARCH,text=True).strip()
    # Upload only this user-authorized personal task branch. Failure does not rerun training.
    try:
        subprocess.run(["git","push","origin","codex/so2-t011-delivery"],cwd=RESEARCH,check=True,timeout=60)
        state["delivery_status"]="pushed"
    except Exception as exc:state["delivery_status"]="local_committed";state["delivery_error"]=repr(exc)
    atomic_json(WORK/"queue_state.json",state)
