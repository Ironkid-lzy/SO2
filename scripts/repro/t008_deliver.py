import argparse,csv,json,math,statistics,subprocess,shutil
from pathlib import Path
import numpy as np
R=Path("/home/lzy/Projects/rl_sample_efficiency_research_t008"); A=Path("/home/lzy/Projects/SO2_t008"); W=A/"_so2_work/runs/EXP-019"; steps=list(range(0,100001,2500))
def ev(p):
 z=[json.loads(x) for x in p.read_text().splitlines()]; z=[x for x in z if x.get("event")=="evaluation"]; assert [x["env_steps"] for x in z]==steps
 for x in z: assert len(x["returns"])==20 and all(math.isfinite(float(y)) for y in x["returns"])
 return [float(x["normalized_score_100"]) for x in z]
def auc(x): return float(np.trapz(x,steps)/100000)
def stats(x): return [statistics.mean(x),statistics.stdev(x) if len(x)>1 else None]
def main():
 p=argparse.ArgumentParser(); p.add_argument("--state",required=True); p.add_argument("--partial",action="store_true"); a=p.parse_args(); state=json.loads(Path(a.state).read_text()); done=sorted(state.get("seeds_completed",[])); complete=not a.partial and done==list(range(5)); c={}
 for s in range(5):
  c["A",s]=ev(R/f"results/raw/EXP-015/EXP-015-s{s}/metrics.jsonl"); c["B",s]=ev(R/f"results/raw/EXP-017/EXP-017-s{s}-attempt1/metrics.jsonl")
  d=R/("results/raw/EXP-016/EXP-016-s0-attempt1" if s==0 else f"results/raw/EXP-018/EXP-018-s{s}-attempt1"); c["D",s]=ev(d/"metrics.jsonl")
 for s in done: c["C",s]=ev(W/f"EXP-019-s{s}-attempt1/metrics.jsonl")
 rows=[]
 for s in range(5):
  x=c.get(("C",s)); z={"seed":s,"C_status":"accepted" if x else "missing","C_endpoint":x[-1] if x else None,"C_auc":auc(x) if x else None}
  for g in "ABD": z[g+"_endpoint"]=c[g,s][-1]; z[g+"_auc"]=auc(c[g,s])
  for g in "DB": z["C_minus_"+g+"_endpoint"]=z["C_endpoint"]-z[g+"_endpoint"] if x else None; z["C_minus_"+g+"_auc"]=z["C_auc"]-z[g+"_auc"] if x else None
  rows.append(z)
 P=R/"results/processed"; P.mkdir(exist_ok=True)
 with (P/"EXP-019-per-seed.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
 agg={k:stats([z[k] for z in rows if z["C_status"]=="accepted"]) for k in ["C_endpoint","C_auc","C_minus_D_endpoint","C_minus_B_endpoint","C_minus_D_auc","C_minus_B_auc"]}
 (P/"EXP-019-summary.json").write_text(json.dumps({"status":"completed" if complete else "partial","accepted":done,"missing":sorted(set(range(5))-set(done)),"per_seed":rows,"aggregate":agg,"code_sha":state["code_sha"]},indent=2)+"\n")
 import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(9,5))
 for g in "ABCD":
  ss=[s for s in range(5) if (g,s) in c]; m=np.asarray([c[g,s] for s in ss]); ax.plot(steps,m.mean(0),label=g)
  if len(ss)>1: ax.fill_between(steps,m.mean(0)-m.std(0,ddof=1),m.mean(0)+m.std(0,ddof=1),alpha=.15)
 ax.set_xlabel("training environment steps"); ax.set_ylabel("normalized score x 100"); ax.legend(); fig.tight_layout(); F=R/"results/figures"; F.mkdir(exist_ok=True); fig.savefig(F/"EXP-019.png",dpi=180); plt.close(fig)
 raw=R/"results/raw/EXP-019"; raw.mkdir(parents=True,exist_ok=True)
 for s in done:
  src=W/f"EXP-019-s{s}-attempt1"; dst=raw/src.name
  if not dst.exists(): shutil.copytree(src,dst,ignore=shutil.ignore_patterns("ckpt"))
  (dst/"large_artifact_location.txt").write_text(str(src)+"\n")
 (raw/"queue_state.json").write_text(json.dumps(state,indent=2)+"\n")
 report=f"""# SO2-T008：仅 LayerNorm 的 2Q 对照

状态：{"completed" if complete else "partial"}；EXP-019；accepted seeds {done}。

C=2Q+LayerNorm、Dropout=0；B=EXP-017 2Q无LN/Dropout；D=EXP-016 seed0和EXP-018 seeds1–4，2Q+LN+Dropout(.005)；A=EXP-015原版10Q。主要比较C-D，次要C-B。C-B还包含LN插入造成的预训练Q函数变化，不能仅解释为训练期正则化。没有Dropout-only组，无法识别交互。范围限于halfcheetah-medium-replay-v2、单一作者checkpoint和heads[0,1]。seed是训练重复单位；20 episodes不是独立训练。回报/TD error不代表真实Q误差。

## 版本与检查

计划SHA 88775b36a7d68b3ecbcc1e8395624d77827bcad9；科研main 3a7aba9caa25756d92f3397e8bb082daee9d9dc2；SO2 main 055a7f15e7ac87ab0bf8f6a21e74986d70ac34e。执行分支codex/so2-t008-ln-only，SHA {state["code_sha"]}；科研分支codex/so2-t008-delivery。Python 3.10.12、PyTorch 2.9.1+cu128、CUDA 12.8、RTX 5060 Laptop。checkpoint SHA {state["checkpoint_sha256"]}；数据SHA {state["dataset_sha256"]}。C/D actor和eval Q相同；p=0确定、D dropout随机；有限更新、action梯度、target无梯度、LN梯度/Polyak、严格重载通过。固定batch 1次更新、0环境步。

## 每seed结果

{(R/"results/processed/EXP-019-per-seed.csv").read_text()}

AUC为归一化分数梯形积分除以100000；汇总mean和样本SD见summary JSON。未预设显著性或非劣效检验。

![A/B/C/D曲线](../../results/figures/EXP-019.png)

原始returns、配置、日志位于results/raw/EXP-019，大checkpoint保留Linux {W}。A/B/D旧raw只读复用。Q诊断为确定性Q函数代理量，不代表校准不确定性或真实高估/低估。
"""
 (R/"reports/so2/SO2-T008.md").write_text(report)
 with (R/"docs/EXPERIMENT_TRACKER.md").open("a") as f: f.write("\n\n## EXP-019 — SO2-T008 C group\n\n| Run | Seed | Status | Endpoint | AUC |\n|---|---:|---|---:|---:|\n"+"".join(f"| EXP-019-s{x['seed']}-attempt1 | {x['seed']} | {x['C_status']} | {x['C_endpoint']} | {x['C_auc']} |\n" for x in rows))
 with (R/"docs/RESEARCH_LOG.md").open("a") as f: f.write(f"\n\n### 2026-10-08 EXP-019 / SO2-T008: {'completed' if complete else 'partial'}\n\nC accepted {done}; C-D tests dropout increment with LN; C-B also reflects changed pretrained Q initialization. No true-Q-error or interaction claim. See reports/so2/SO2-T008.md.\n")
 for x in ["docs/EXPERIMENT_TRACKER.md","docs/RESEARCH_LOG.md","reports/so2/SO2-T008.md","results/processed/EXP-019-per-seed.csv","results/processed/EXP-019-summary.json","results/figures/EXP-019.png"]: subprocess.check_call(["git","add",x],cwd=R)
 subprocess.check_call(["git","add","-f","results/raw/EXP-019"],cwd=R); subprocess.check_call(["git","commit","-m","EXP-019: deliver T008"],cwd=R); subprocess.check_call(["git","push","origin","codex/so2-t008-delivery"],cwd=R)
 print(json.dumps({"status":"completed" if complete else "partial","accepted":done,"research_sha":subprocess.check_output(["git","rev-parse","HEAD"],cwd=R,text=True).strip()}))
if __name__=="__main__": main()
