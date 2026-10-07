"""Aggregate immutable T006 evidence and push the research delivery branch."""
import argparse, csv, hashlib, json, math, statistics, subprocess, shutil
from pathlib import Path
import numpy as np
ROOT=Path("/home/lzy/Projects/SO2_t006")
RESEARCH=Path("/home/lzy/Projects/rl_sample_efficiency_research_t006")
WORK=ROOT/"_so2_work/runs/EXP-017"
SEEDS=range(5); STEPS=list(range(0,100001,2500))
def savej(p,x):
 p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n")
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def readm(p): return [json.loads(x) for x in p.read_text().splitlines()]
def evs(p):
 r=[x for x in readm(p) if x.get("event")=="evaluation"]
 assert [x["env_steps"] for x in r]==STEPS
 assert all(x["episodes_completed"]==20 and len(x["returns"])==20 for x in r)
 assert all(math.isfinite(float(y)) for x in r for y in x["returns"])
 return r,[float(x["normalized_score_100"]) for x in r]
def auc(y): return sum((y[i]+y[i+1])*1250 for i in range(40))/100000
def ms(x): return {"mean":statistics.mean(x),"std_sample":statistics.stdev(x)}
def git(*x): return subprocess.check_output(["git",*x],cwd=RESEARCH,text=True).strip()
def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--state",type=Path,required=True); ap.add_argument("--partial",action="store_true"); args=ap.parse_args()
 state=json.loads(args.state.read_text()); completed=sorted(state.get("seeds_completed",[]))
 rawbase=RESEARCH/"results/raw/EXP-017"
 if rawbase.exists(): raise RuntimeError("EXP-017 raw path exists; refusing overwrite")
 A={}; B={}; per=[]
 for seed in completed:
  run=WORK/f"EXP-017-s{seed}-attempt1"; acc=json.loads((run/"acceptance.json").read_text()); meta=json.loads((run/"run_meta.json").read_text())
  if not acc.get("accepted") or meta.get("status")!="completed":
   if not args.partial: raise RuntimeError(f"seed {seed} lacks acceptance")
   continue
  _,B[seed]=evs(run/"metrics.jsonl"); _,A[seed]=evs(RESEARCH/f"results/raw/EXP-015/EXP-015-s{seed}/metrics.jsonl")
  ae,be=A[seed][-1],B[seed][-1]; aa,ba=auc(A[seed]),auc(B[seed])
  per.append({"seed":seed,"A_endpoint":ae,"B_endpoint":be,"endpoint_delta":be-ae,"A_auc":aa,"B_auc":ba,"auc_delta":ba-aa,"wall_hours":(meta["end_unix"]-meta["start_unix"])/3600})
  raw=rawbase/f"EXP-017-s{seed}-attempt1"; raw.mkdir(parents=True,exist_ok=False)
  for name in ("metrics.jsonl","run_meta.json","policy_reload.json","acceptance.json","live_policy_probes.jsonl","formatted_total_config.py","total_config.py"):
   if (run/name).exists(): shutil.copy2(run/name,raw/name)
  if (run/"rng").exists(): shutil.copytree(run/"rng",raw/"rng")
  for name in (f"seed{seed}.log",f"seed{seed}_acceptance.log",f"seed{seed}_first_learning_check.json"):
   if (WORK/name).exists(): shutil.copy2(WORK/name,raw/name)
  files=[WORK/f"seed{seed}.log",*sorted((run/"ckpt").glob("*.pth.tar")),*sorted((run/"rng").glob("*.json"))]
  savej(raw/"artifact_manifest.json",[{"path":str(p.resolve()),"size_bytes":p.stat().st_size,"sha256":sha(p)} for p in files])
  savej(raw/"queue_state_snapshot.json",state)
 failed_seed=state.get("current_seed")
 if failed_seed is not None and failed_seed not in [r["seed"] for r in per]:
  run=WORK/f"EXP-017-s{failed_seed}-attempt1"
  if run.exists():
   raw=rawbase/f"EXP-017-s{failed_seed}-attempt1"
   raw.mkdir(parents=True,exist_ok=False)
   for name in ("metrics.jsonl","run_meta.json","policy_reload.json","acceptance.json","live_policy_probes.jsonl","formatted_total_config.py","total_config.py"):
    if (run/name).exists(): shutil.copy2(run/name,raw/name)
   if (run/"rng").exists(): shutil.copytree(run/"rng",raw/"rng")
   for name in (f"seed{failed_seed}.log",f"seed{failed_seed}_acceptance.log",f"seed{failed_seed}_first_learning_check.json"):
    if (WORK/name).exists(): shutil.copy2(WORK/name,raw/name)
   arts=[WORK/f"seed{failed_seed}.log",*sorted((run/"ckpt").glob("*.pth.tar")),*sorted((run/"rng").glob("*.json"))]
   arts=[p for p in arts if p.exists()]
   savej(raw/"artifact_manifest.json",[{"path":str(p.resolve()),"size_bytes":p.stat().st_size,"sha256":sha(p)} for p in arts])
   savej(raw/"queue_state_snapshot.json",state)
 missing=sorted(set(SEEDS)-{r["seed"] for r in per}); partial=args.partial or len(per)!=5
 pre=json.loads((ROOT/"_so2_work/validation/EXP-017/preflight.json").read_text())
 evdir=RESEARCH/"SO2-T006"; evdir.mkdir(parents=True,exist_ok=True)
 for src,dst in ((ROOT/"_so2_work/validation/EXP-017/preflight.json",evdir/"preflight.json"),(ROOT/"_so2_work/validation/EXP-017/A_reference.json",evdir/"A_reference.json"),(ROOT/"_so2_work/validation/EXP-017/T006_AB.json",evdir/"T006_AB.json"),(ROOT/"scripts/repro/configs/t006_effective.yaml",evdir/"t006_effective.yaml"),(Path("/tmp/SO2-T006-resources.json"),evdir/"resources_linux.json")):
  if src.exists(): shutil.copy2(src,dst)
 if per:
  ids=[x["seed"] for x in per]; ma=np.asarray([A[s] for s in ids],float); mb=np.asarray([B[s] for s in ids],float)
  col={"env_steps":STEPS}
  for i,s in enumerate(ids): col[f"A_seed{s}"]=ma[i].tolist(); col[f"B_seed{s}"]=mb[i].tolist()
  col["A_mean"]=ma.mean(0).tolist(); col["B_mean"]=mb.mean(0).tolist()
  col["A_std_sample"]=ma.std(0,ddof=1).tolist() if len(ids)>1 else [None]*len(STEPS)
  col["B_std_sample"]=mb.std(0,ddof=1).tolist() if len(ids)>1 else [None]*len(STEPS)
  out=RESEARCH/"results/processed"; out.mkdir(parents=True,exist_ok=True)
  with open(out/"EXP-017-curves.csv","w",newline="") as f:
   w=csv.writer(f); w.writerow(col.keys()); [w.writerow([col[k][i] for k in col]) for i in range(len(STEPS))]
  with open(out/"EXP-017-per-seed.csv","w",newline="") as f:
   fields=list(per[0]); w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(per)
  endpointsA=[A[s][-1] for s in ids]; endpointsB=[B[s][-1] for s in ids]
  auA=[x["A_auc"] for x in per]; auB=[x["B_auc"] for x in per]
  de=[x["endpoint_delta"] for x in per]; du=[x["auc_delta"] for x in per]
  summary={"exp_id":"EXP-017","status":"partial" if partial else "completed","available_seeds":ids,"expected_seeds":list(SEEDS),
   "endpoint":{"A":ms(endpointsA),"B":ms(endpointsB)},"A_auc":ms(auA),"B_auc":ms(auB),
   "paired_delta":{"endpoint_mean":statistics.mean(de),"endpoint_std_sample":statistics.stdev(de) if len(ids)>1 else None,"auc_mean":statistics.mean(du),"auc_std_sample":statistics.stdev(du) if len(ids)>1 else None},
   "per_seed":per,"steps":STEPS,"A_curves_by_seed":{str(s):A[s] for s in ids},"B_curves_by_seed":{str(s):B[s] for s in ids},
   "A_curve_mean":col["A_mean"],"B_curve_mean":col["B_mean"],"code_sha":state.get("code_sha"),"config_sha256":state.get("config_sha256")}
  savej(out/"EXP-017-summary.json",summary)
  import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
  fig,(a1,a2)=plt.subplots(2,1,figsize=(9,8),sharex=True)
  a1.plot(STEPS,col["A_mean"],label="A: EXP-015 10Q",color="#3465a4"); a1.plot(STEPS,col["B_mean"],label="B: EXP-017 2Q",color="#c43c35")
  if len(ids)>1:
   a1.fill_between(STEPS,ma.mean(0)-ma.std(0,ddof=1),ma.mean(0)+ma.std(0,ddof=1),alpha=.2,color="#3465a4")
   a1.fill_between(STEPS,mb.mean(0)-mb.std(0,ddof=1),mb.mean(0)+mb.std(0,ddof=1),alpha=.2,color="#c43c35")
  dfile=RESEARCH/"results/raw/EXP-016/EXP-016-s0-attempt1/metrics.jsonl"; dr=[r for r in readm(dfile) if r.get("event")=="evaluation"]
  if dr: a1.plot([r["env_steps"] for r in dr],[r["normalized_score_100"] for r in dr],"--",color="#777",label="D: EXP-016 seed0 exploratory")
  a1.axvspan(0,5000,color="orange",alpha=.1,label="5000-step warmup"); a1.set_ylabel("D4RL normalized score × 100"); a1.legend(fontsize=8,ncol=2)
  for s in ids: a2.plot(STEPS,np.asarray(B[s])-np.asarray(A[s]),label=f"seed {s}")
  a2.axhline(0,color="black",linewidth=.8); a2.set_xlabel("Total training environment steps"); a2.set_ylabel("Paired B − A"); a2.legend(fontsize=8,ncol=5)
  fig.tight_layout(); fd=RESEARCH/"results/figures"; fd.mkdir(parents=True,exist_ok=True); fig.savefig(fd/"EXP-017.png",dpi=180); plt.close(fig)
  dsummary={"endpoint":dr[-1]["normalized_score_100"],"auc":auc([r["normalized_score_100"] for r in dr])} if dr else {}
 else:
  summary={"exp_id":"EXP-017","status":"partial","available_seeds":[]}; dsummary={}
  savej(RESEARCH/"results/processed/EXP-017-summary.json",summary)
 if rawbase.exists(): savej(rawbase/"queue_state.json",state)
 stat="partial" if partial else "completed"
 seedtable="\n".join(f"| {r['seed']} | {r['A_endpoint']:.6f} | {r['B_endpoint']:.6f} | {r['endpoint_delta']:+.6f} | {r['A_auc']:.6f} | {r['B_auc']:.6f} | {r['auc_delta']:+.6f} | {r['wall_hours']:.3f} |" for r in per)
 if per:
  e=summary["endpoint"]; p=summary["paired_delta"]
  stattable=f"""| Metric | A: 10Q | B: 2Q | Paired B−A |
|---|---:|---:|---:|
| 100k endpoint | {e['A']['mean']:.6f} ± {e['A']['std_sample']:.6f} | {e['B']['mean']:.6f} ± {e['B']['std_sample']:.6f} | {p['endpoint_mean']:+.6f} ± {p['endpoint_std_sample']:.6f} |
| 0–100k AUC | {summary['A_auc']['mean']:.6f} ± {summary['A_auc']['std_sample']:.6f} | {summary['B_auc']['mean']:.6f} ± {summary['B_auc']['std_sample']:.6f} | {p['auc_mean']:+.6f} ± {p['auc_std_sample']:.6f} |"""
 else: stattable="No complete seed is available for aggregate estimates."
 report=f"""# SO2-T006：2Q 与原版 10Q 的 100k × 5 seeds 对照实验

> 状态：{stat}。EXP-017。任务单：docs/tasks/SO2-T006.md（2026-10-06-v2）。执行端交付，科研结论待指挥端审核。

## 设计与边界

A 为已验收 EXP-015 原版 10 critic、无 LayerNorm/Dropout；B 从原作者 checkpoint 的 online/target head [0,1] 初始化，actor 完整保留，SO2 三层 256 critic，ensemble size=2、LN=False、Dropout=0。每组 seeds 0–4，各训练 100000 总环境步（含 5000 步预热），每 2500 步用确定性 actor 评估 20 episodes，共 41 点。B batch 为 2560（online 256 + mixed 2304），UTD10，actor 每 10 次 critic 更新，alpha=0.2；其余规则遵照 EXP-015。

A/B 是固定 checkpoint、任务和 head 子集条件下的总处理效应，包含 actor min 与 loss reduction 随 N 变化的影响，不能称作纯网络容量效应。Episodes 不是训练重复；配对 seed 不保证后续 RNG 轨迹相同。回报差或 TD error 不等于真实 Q 估计误差变化。

## 冻结身份与开跑前检查

- Algorithm branch codex/so2-t006-b；code SHA {state.get('code_sha','')}.
- Research branch codex/so2-t006-delivery；plan base e407167383c7055a2ff4635ed2aee06cb165bc0b.
- Python config SHA256 {state.get('config_sha256','')}; preflight SHA256 {state.get('preflight_sha256','')}.
- Checkpoint SHA256 {state.get('checkpoint_sha256','')}; D4RL data SHA256 {state.get('dataset_sha256','')}.
- Fixed-batch used {state.get('validation_updates',31)} updates and 0 environment steps. A outputs/gradients matched the EXP-015 training path; B online/target head mapping, actor, finite action gradient, target no-gradient, finite update, and strict reload passed.
- Effective configuration diff: {json.dumps(pre.get('effective_config_diff',{}),sort_keys=True)}. LN=False and Dropout=0 are explicit locks at original defaults. Full records are in SO2-T006/.
- Queue ran seeds serially in distinct processes. Each seed loaded the same author checkpoint with a fresh online buffer. Evaluation checkpoints have global Python, NumPy, Torch CPU, and CUDA RNG sidecars; these are diagnostic snapshots, not full continuation checkpoints.

## Results

{('All five seeds completed and passed technical acceptance.' if not missing else 'Missing or incomplete seeds: '+', '.join(map(str,missing))+'. Available seeds only are shown; no complete five-seed conclusion is claimed.')}

| Seed | A 100k | B 100k | B−A | A AUC | B AUC | B−A AUC | B wall hours |
|---:|---:|---:|---:|---:|---:|---:|---:|
{seedtable}

{stattable}

AUC is trapezoidal 0–100k score integral divided by 100000. Means use sample SD (ddof=1) across training seeds. No formal hypothesis test or non-inferiority claim was pre-specified; five seeds leave substantial uncertainty. Figure results/figures/EXP-017.png shows A/B mean ± sample SD, paired trajectories, and D seed0 as an exploratory reference. Exact 41-point curves are in results/processed/EXP-017-curves.csv.

EXP-016 D seed0 supplementary comparison: endpoint {dsummary.get('endpoint',float('nan')):.6f}, AUC {dsummary.get('auc',float('nan')):.6f}. D is 2Q+LN+Dropout(0.005), one seed only, and cannot isolate LN or Dropout.

## Technical acceptance and artifacts

{('Every seed passed 100000 env steps, 950000 critic / 95000 actor / 0 alpha updates, 41×20 finite evaluation returns, checkpoint reload, and 41 RNG snapshots.' if not missing else 'See per-run acceptance files for passed seeds; missing/failed seeds and reasons remain in queue state.')}
Per-seed times, software versions, effective configs, original episode returns, and checkpoint/RNG hashes are in results/raw/EXP-017/. Large checkpoints remain on Linux under /home/lzy/Projects/SO2_t006/_so2_work/runs/EXP-017/. Resource, preflight, config, and fixed-batch records are in SO2-T006/. No best-checkpoint selection, rerun, C treatment, or hyperparameter search was performed.
"""
 (RESEARCH/"reports/so2/SO2-T006.md").write_text(report)
 with open(RESEARCH/"docs/EXPERIMENT_TRACKER.md","a") as f:
  f.write("\n\n## EXP-017 — SO2-T006 固定 2Q 五种子对照\n\n| EXP-ID | Run ID | Seed | Task | Control | Treatment | Status | 100k score | AUC |\n|---|---|---:|---|---|---|---|---:|---:|\n")
  for r in per: f.write(f"| EXP-017 | EXP-017-s{r['seed']}-attempt1 | {r['seed']} | halfcheetah-medium-replay-v2 | EXP-015 seed{r['seed']} | 2Q, LN off, Dropout 0 | {'partial' if partial else 'done'} | {r['B_endpoint']:.6f} | {r['B_auc']:.6f} |\n")
  if state.get("current_seed") is not None and state["current_seed"] in missing:
   f.write(f"| EXP-017 | EXP-017-s{state['current_seed']}-attempt1 | {state['current_seed']} | halfcheetah-medium-replay-v2 | EXP-015 seed{state['current_seed']} | 2Q, LN off, Dropout 0 | failed/partial | — | — |\n")
  f.write(f"\nStatus: {stat}; completed seeds {sorted(x['seed'] for x in per)}; missing {missing}. See reports/so2/SO2-T006.md.\n")
 with open(RESEARCH/"docs/RESEARCH_LOG.md","a") as f:
  f.write("\n\n### 2026-10-06 EXP-017 / SO2-T006 固定 2Q 对照\n")
  if partial: f.write(f"**状态：partial。** 可用 seeds {sorted(x['seed'] for x in per)}，缺失 {missing}。不作完整五种子推断；见 reports/so2/SO2-T006.md。\n")
  elif per: f.write(f"**观测：** A 100k {summary['endpoint']['A']['mean']:.6f} ± {summary['endpoint']['A']['std_sample']:.6f}；B {summary['endpoint']['B']['mean']:.6f} ± {summary['endpoint']['B']['std_sample']:.6f}。配对 B−A endpoint {summary['paired_delta']['endpoint_mean']:+.6f} ± {summary['paired_delta']['endpoint_std_sample']:.6f}，AUC {summary['paired_delta']['auc_mean']:+.6f} ± {summary['paired_delta']['auc_std_sample']:.6f}。\n**解释边界：** 固定初始化/任务/head[0,1]，包含 ensemble min 和 loss reduction 随 N 变化的总效应；5 seeds 不作非劣效或显著性声称，回报差不等于真实 Q 误差。结论待指挥端审核。见 reports/so2/SO2-T006.md。\n")
 git("add","docs/EXPERIMENT_TRACKER.md","docs/RESEARCH_LOG.md","reports/so2/SO2-T006.md","SO2-T006","results/processed/EXP-017-summary.json")
 for p in ("results/processed/EXP-017-per-seed.csv","results/processed/EXP-017-curves.csv","results/figures/EXP-017.png"):
  if (RESEARCH/p).exists(): git("add",p)
 git("add","-f","results/raw/EXP-017")
 subprocess.check_call(["git","commit","-m",("EXP-017: deliver T006 partial seed results" if partial else "EXP-017: deliver T006 five-seed SO2 ensemble comparison")],cwd=RESEARCH)
 subprocess.check_call(["git","push","origin","codex/so2-t006-delivery"],cwd=RESEARCH)
 print(json.dumps({"status":stat,"research_sha":git("rev-parse","HEAD"),"seeds":sorted(x["seed"] for x in per)}))
if __name__=="__main__": main()
