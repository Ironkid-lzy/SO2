"""Compact progress delivery every six hours/stage; raw is sealed once only."""
import argparse
import csv
import shutil
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo

from t010_edac import *

RESEARCH=Path("/home/lzy/Projects/rl_sample_efficiency_research_t010")
EVIDENCE=RESEARCH/"reports/so2/SO2-T010"
MARKER="\n## 自动执行证据\n"


def rows(path):
    if not path.exists():return []
    out=[]
    for line in path.read_text().splitlines():
        try:out.append(json.loads(line))
        except json.JSONDecodeError:pass
    return out


def curves(run,phase,exp):
    events=[r for r in rows(run/"metrics.jsonl") if r.get("event")=="evaluation"]
    axis="gradient_updates" if phase=="offline" else "env_steps"
    processed=RESEARCH/"results/processed"/exp;processed.mkdir(parents=True,exist_ok=True)
    atomic_json(processed/"evaluation_rows.json",events)
    if not events:return {}
    keys=[axis,"return_mean","normalized_score_100","evaluation_env_steps","evaluation_interactions_total"]
    with (processed/"curve.csv").open("w") as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        writer.writerows([{k:r[k] for k in keys} for r in events])
    scores=[r["normalized_score_100"] for r in events];x=[r[axis] for r in events]
    result={"points":len(events),"axis":axis,"start":scores[0],"last_x":x[-1],"last_score":scores[-1]}
    if phase=="online" and x[-1]==100000:
        result["endpoint_100k"]=scores[-1]
        result["auc_0_100k"]=sum((scores[i]+scores[i+1])*.5*(x[i+1]-x[i]) for i in range(len(x)-1))/100000
    atomic_json(processed/"summary.json",result)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(6.4,3.7));ax.plot(x,scores)
        ax.set(xlabel="Offline gradient updates" if phase=="offline" else "Online training environment steps",
               ylabel="D4RL normalized score (×100)",title=f"{exp}: seed0, same network EDAC → SO2")
        ax.grid(alpha=.25);fig.tight_layout()
        path=RESEARCH/"results/figures"/exp;path.mkdir(parents=True,exist_ok=True)
        fig.savefig(path/"curve.png",dpi=160);fig.savefig(path/"curve.pdf");plt.close(fig)
    except ImportError:
        result["figure_limitation"]="matplotlib unavailable; replayable CSV retained"
        atomic_json(processed/"summary.json",result)
    return result


def deliver(state):
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    atomic_json(EVIDENCE/"queue_state.json",state)
    recipe=ROOT/"scripts/repro/configs/t010_recipe.json"
    shutil.copy2(recipe,EVIDENCE/"recipe.json")
    for label in ("cpu-v1","cpu-v2","cpu-v3","gpu-v1","gpu-v2"):
        src=ROOT/f"_so2_work/validation/t010-{label}/validation.json"
        if src.exists():shutil.copy2(src,EVIDENCE/f"validation-{label}.json")
    src=ROOT/"_so2_work/validation/t010-queue-cpu/validation.json"
    if src.exists():shutil.copy2(src,EVIDENCE/"validation-queue-cpu.json")
    handoff=Path(state.get("queue_workdir", str(ROOT/"_so2_work/t010")))/"handoff.json"
    if handoff.exists():shutil.copy2(handoff,EVIDENCE/"handoff.json")
    summaries={}
    for phase,exp in (("offline","EXP-025"),("online","EXP-026")):
        run=Path(state["runs"][phase])
        if not run.exists():continue
        summaries[phase]=curves(run,phase,exp)
        processed=RESEARCH/"results/processed"/exp
        for name in ("run_meta.json","state.json","stage_continuity.json","acceptance.json","checkpoint_manifest.json","first_window.json","policy_reload.json"):
            src=run/name
            if src.exists():shutil.copy2(src,processed/name)
        meta=json.loads((run/"run_meta.json").read_text()) if (run/"run_meta.json").exists() else {}
        # No live raw copies: a terminal run is copied once, then never rewritten.
        if meta.get("status") in ("completed","partial","failed"):
            raw=RESEARCH/"results/raw"/exp/run.name
            if not raw.exists():
                raw.mkdir(parents=True)
                for name in ("run_meta.json","metrics.jsonl","stage_continuity.json","acceptance.json","checkpoint_manifest.json","first_window.json","policy_reload.json"):
                    src=run/name
                    if src.exists():shutil.copy2(src,raw/name)
                files=[{"path":str(f.relative_to(run)),"size":f.stat().st_size,"sha256":digest(f),"backup_status":"Linux only"}
                       for f in sorted(run.rglob("*")) if f.is_file() and f.parent.name!="ckpt"]
                checks=json.loads((run/"checkpoint_manifest.json").read_text()) if (run/"checkpoint_manifest.json").exists() else []
                atomic_json(raw/"artifact_manifest.json",{"linux_path":str(run),"files":files,"checkpoints":checks,
                            "large_artifact_backup":"Linux only; no second independent checkpoint backup"})
    stamp=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    report=RESEARCH/"reports/so2/SO2-T010.md"
    text=report.read_text().split(MARKER)[0]
    text=text.replace("状态：partial / engineering；未启动正式训练，尚未研究验收。",
                      "状态以文末自动执行证据为准；尚未研究验收。")
    table="\n".join(f"| {name} | {json.dumps(value,ensure_ascii=False)} |" for name,value in summaries.items())
    text+=MARKER+f"\n更新时间（Asia/Shanghai）：{stamp}。状态：`{state['status']} / {state['phase']}`。\n\n"
    text+=f"算法执行 SHA：`{state['code_sha']}`，个人分支 `codex/so2-t010-pretrain`；科研交付 `codex/so2-t010-delivery`。配置 SHA256：`{state['recipe_sha256']}`。\n\n"
    text+=f"阶段实际退出码、命令、墙钟：见 [queue_state.json](SO2-T010/queue_state.json)。资源与锁、版本依赖、CPU/CUDA测试、父checkpoint和逐点数据均保存于同名证据目录/processed；大checkpoint和完整日志保留Linux。\n\n"
    text+=f"GPU正式阶段：{list(state.get('stages',{}))}。当前进度：`{json.dumps(state.get('progress',{}),ensure_ascii=False)}`。阻塞/失败：`{state.get('error','无')}`。推送错误：`{state.get('delivery_error','无')}`。\n\n"
    text+="| 阶段 | 已观测统计（未完成时不是终点） |\n|---|---|\n"+table+"\n\n"
    text+="仅单seed工程可行性探索，不能说明优越性、Dropout增量或已恢复作者训练轨迹。两阶段横轴独立，评价交互不计入训练预算。alpha自动→固定.2、ES开启→关闭、TD sum→mean、离线min target→在线逐head+PVU、在线optimizer/计数重建均是明确的阶段算法切换；网络所有已学参数保留。\n"
    report.write_text(text)
    tracker=RESEARCH/"docs/EXPERIMENT_TRACKER.md";content=tracker.read_text()
    for phase,exp in (("offline","EXP-025"),("online","EXP-026")):
        status="completed（待研究验收）" if phase in state.get("stages",{}) and "acceptance" in state["stages"][phase] else (state["status"] if state["phase"]==phase else "pending")
        row=f"| {exp} | [SO2-T010](tasks/SO2-T010.md) | {status} | 同结构2Q+LN+Dropout(.005)，seed0；{phase}；actor边界[-20,2] | {'3M梯度更新/48h/0训练交互' if phase=='offline' else '100k训练环境步/8h'} |"
        lines=content.splitlines();found=False
        for i,line in enumerate(lines):
            if line.startswith(f"| {exp} |"):lines[i]=row;found=True
        if not found:lines += ["",row]
        content="\n".join(lines)+"\n"
    tracker.write_text(content)
    log=RESEARCH/"docs/RESEARCH_LOG.md";text=log.read_text()
    if "SO2-T010：同结构衔接工程探索" not in text:
        text+="\n## 2026-10-10 — SO2-T010：同结构衔接工程探索\n\n研究问题：H-SO2-010（工程可行性）从随机初始化进行2Q+LN+Dropout EDAC预训练，是否能保持所有已学网络函数接入SO2并完成100k？支持证据是精确参数/函数连续性、有限性和固定预算计数通过；加载/算法不对齐或预算失败则交付partial。不是性能提升假设。用户在训练前选择两阶段SO2 actor边界[-20,2]，与官方EDAC[-5,2]差异已登记。详见 [T010报告](../reports/so2/SO2-T010.md)，单seed不做因果或优越性结论。\n"
        log.write_text(text)
    index=RESEARCH/"reports/so2/README.md";text=index.read_text()
    if "SO2-T010.md" not in text:index.write_text(text+"\n- [SO2-T010：同结构 EDAC预训练→SO2](SO2-T010.md)：seed0工程探索；状态以报告为准，尚待验收。\n")
    # Record T009's stopping ledger without writing to its checkout.
    if state.get("handoff"):
        old_repo=Path("/home/lzy/Projects/rl_sample_efficiency_research_t009")
        old_row=next((line for line in (old_repo/"docs/EXPERIMENT_TRACKER.md").read_text().splitlines() if line.startswith("| EXP-024 |")),None)
        if old_row:
            tracker.write_text("\n".join(old_row if line.startswith("| EXP-024 |") else line for line in tracker.read_text().splitlines())+"\n")
    subprocess.run(["git","add","reports/so2/SO2-T010.md","reports/so2/SO2-T010","reports/so2/README.md",
                    "docs/EXPERIMENT_TRACKER.md","docs/RESEARCH_LOG.md","results/processed","results/figures","results/raw"],cwd=RESEARCH,check=True)
    if subprocess.run(["git","diff","--cached","--quiet"],cwd=RESEARCH).returncode:
        subprocess.run(["git","commit","-m",f"SO2-T010: record {state['status']} {state['phase']} evidence"],cwd=RESEARCH,check=True)
    subprocess.run(["git","push","origin","codex/so2-t010-delivery"],cwd=RESEARCH,check=True)
    print(json.dumps({"research_sha":subprocess.check_output(["git","rev-parse","HEAD"],cwd=RESEARCH,text=True).strip()}))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--state",type=Path,required=True);a=p.parse_args()
    deliver(json.loads(a.state.read_text()))
