import argparse, hashlib, json, math, runpy, subprocess
from pathlib import Path

ROOT = Path("/home/lzy/Projects/SO2_t006")
WORK = ROOT / "_so2_work/validation/EXP-017"
BASE = Path("/home/lzy/Projects/SO2_t003")
CKPT = Path("/home/lzy/Projects/SO2/_so2_work/assets/checkpoints/ckpt/halfcheetah-medium-replay-v2.ckpt")
DATA = Path("/home/lzy/.d4rl/datasets/halfcheetah_medium_replay-v2.hdf5")
EXPECTED = {
    "checkpoint": "02d74b7860620f1f9863a2f217aff0abfaae37f4a69ba6d0e1a0ea10f98d94f7",
    "dataset": "48d494a4770c11f48260736dec090b78bdca8375647ed4d24bfa5b6610c7f683",
}

def digest(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""): h.update(chunk)
    return h.hexdigest()

def flatten(obj,prefix=""):
    out={}
    for k,v in obj.items():
        key=f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v,dict): out.update(flatten(v,key))
        else: out[key]=v
    return out

a=json.loads((WORK/"A_reference.json").read_text())
ab=json.loads((WORK/"T006_AB.json").read_text())
checks={}
checks["same_fixed_batch"] = a["batch_sha256"] == ab["batch_sha256"] and a["batch_size"] == ab["batch_size"] == 2560
checks["checkpoint_hash"] = digest(CKPT) == EXPECTED["checkpoint"]
checks["dataset_hash"] = digest(DATA) == EXPECTED["dataset"]
a0, aref = ab["groups"]["A"], a["groups"]["A"]
compare = {}
for key in ("q_probe","target_q_probe","actor_probe"):
    compare[key] = a0[key] == aref[key]
for key in ("critic_loss","policy_loss","td_error","target_q_value","q_value"):
    compare[key] = math.isclose(a0["update"][key],aref["update"][key],rel_tol=0,abs_tol=1e-6)
for key in ("critic_grad_l2","actor_grad_l2"):
    compare[key] = math.isclose(a0[key],aref[key],rel_tol=0,abs_tol=1e-6)
checks["A_matches_EXP015_training_path"] = all(compare.values())
checks["B_complete"] = all(ab["groups"]["B"].get(k,False) for k in
    ("head_slice_exact","linear_mapping_exact","actor_exact","fresh_reload_exact",
     "target_no_gradient","actor_action_gradient_finite_nonzero"))
checks["B_updates_finite"] = all(math.isfinite(v) for k,v in ab["groups"]["B"]["update"].items()
    if k != "peak_cuda_allocated_bytes")
checks["validation_updates_under_budget"] = a["validation_updates"] + ab["validation_updates"] <= 200
baseline=runpy.run_path(str(ROOT/"scripts/repro/configs/halfcheetah_medium_replay_full.py"))
t006=runpy.run_path(str(ROOT/"scripts/repro/configs/halfcheetah_medium_replay_t006.py"))
left=flatten(baseline["main_config"].to_dict() if hasattr(baseline["main_config"],"to_dict") else dict(baseline["main_config"]))
right=flatten(t006["main_config"].to_dict() if hasattr(t006["main_config"],"to_dict") else dict(t006["main_config"]))
expected_paths={"policy.model.critic_ensemble_size","policy.model.critic_layer_norm","policy.model.critic_dropout_rate"}
for path in expected_paths:
    if path.endswith("critic_layer_norm"): left.setdefault(path,False)
    if path.endswith("critic_dropout_rate"): left.setdefault(path,0.0)
diff={key:{"A":left.get(key,"<missing>"),"B":right.get(key,"<missing>")}
      for key in sorted(set(left)|set(right)) if left.get(key,"<missing>") != right.get(key,"<missing>")}
checks["config_diff_only_expected_critic_fields"] = set(diff).issubset(expected_paths) and diff.get("policy.model.critic_ensemble_size",{}).get("A")==10 and diff.get("policy.model.critic_ensemble_size",{}).get("B")==2
out={
 "status":"passed" if all(checks.values()) else "failed",
 "checks":checks,"A_comparisons":compare,
 "effective_config_diff":diff,
 "locked_config":{"critic_ensemble_size":2,"critic_layer_norm":False,"critic_dropout_rate":0.0,
                  "actor_critic_layers":3,"hidden_width":256,"updates_per_collect":10,
                  "actor_update_freq":10,"batch_size":2560,"online_batch":256,
                  "mixed_batch":2304,"alpha":0.2,"eval_episodes":20,"eval_interval":2500,
                  "random_collect_size":5000},
 "baseline_checkout_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=BASE, text=True).strip(),
 "baseline_training_tag_object_sha": subprocess.check_output(["git", "rev-parse", "so2-baseline-exp015"], cwd=BASE, text=True).strip(),
 "baseline_training_tag_sha": subprocess.check_output(["git", "rev-parse", "so2-baseline-exp015^{commit}"], cwd=BASE, text=True).strip(),
 "baseline_validation_checkout_sha": a["code_root"],
 "checkpoint_path":str(CKPT),"checkpoint_sha256":digest(CKPT),
 "dataset_path":str(DATA),"dataset_sha256":digest(DATA),
 "fixed_batch_sha256":ab["batch_sha256"],"fixed_batch_indices":ab["batch_indices"],
 "fixed_batch_updates":a["validation_updates"]+ab["validation_updates"],
 "A_path":a["ding_file"],"B_path":ab["ding_file"],
 "B_diagnostics":ab["groups"]["B"]}
(WORK/"preflight.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
print(json.dumps({"status":out["status"],"checks":checks,"config_diff":diff,
                  "fixed_batch_updates":out["fixed_batch_updates"]},indent=2))
if out["status"] != "passed": raise SystemExit(1)
