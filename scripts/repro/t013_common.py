"""T013 identities; frozen T010 parent, T011 control; actor K4 only."""
import shutil
import subprocess
from t010_edac import *

BASE_SHA = "728dac3c785059c3c9822ddef9de3da930c03c7f"
SEED0_SHA = "be3690b4d907651dbe51205bb456cec5e519ee33"
PLAN_SHA = "09336a72a4648ce2cdcfa80cea1249b40e36997d"
CONTROL_SHA = "94f4190f3cc4e5df9dd335f32389e8782c08e86d"
THREAD = "01a1647e-01f6-7c30-b8c1-482722ad1773"
OLD = Path("/home/lzy/Projects/SO2_t010")
RESEARCH = Path("/home/lzy/Projects/rl_sample_efficiency_research_t011")
WORK = ROOT/"_so2_work/t013"
VALIDATION = ROOT/"_so2_work/validation/t013"
RECIPE = ROOT/"scripts/repro/configs/t010_recipe.json"
CONFIG = ROOT/"scripts/repro/configs/halfcheetah_medium_replay_t010.py"
PARENT_HASHES = [
"799d8c07aff11009790b00650276e4e39ec96cc6d63739bf8d80070c2e034267",
"68aa1dfed8500f6111b1b730dcc778a68d48e42e9e196bd6a41f97151b8c5b20",
"3a697aa32b18d8836e2668ac1a7f1dae883783a418cba0042347992264a3b807",
"0e83b81a5bb67b7583c1a4721403a8c6047ebe906a7b4f72d510e236a7ba33ff",
"584e3c0e4a8d7c3aaf97c2c9e2c273b91221e086fc5081f5fa61eae54494085c"]
PYTHON = "/home/lzy/Projects/SO2/.venv/bin/python"
LOCK = Path("/home/lzy/Projects/SO2/_so2_work/training_resource.lock")

def control_run(seed):
    exp = "EXP-026" if seed == 0 else "EXP-028"
    return OLD/f"_so2_work/runs/{exp}/{exp}-s{seed}-attempt{2 if seed==0 else 1}"

def run_path(seed):
    return ROOT/f"_so2_work/runs/EXP-034/EXP-034-s{seed}-attempt1"

def parent_manifest(seed):
    path = control_run(seed)/"initialization_manifest.json"
    frozen = subprocess.check_output(["git","show",f"{CONTROL_SHA}:results/raw/{control_run(seed).parent.name}/{control_run(seed).name}/initialization_manifest.json"],cwd=RESEARCH,text=True)
    assert json.loads(frozen)==json.loads(path.read_text()), "control manifest changed"
    m=json.loads(frozen); assert m["parent_sha256"]==PARENT_HASHES[seed]
    assert digest(m["parent_path"])==m["parent_sha256"]
    assert digest(m["export_path"])==m["export_sha256"]
    return m

def checked_parent(parent, ident, record=None):
    seed=ident["seed"]; assert seed in range(5)
    m=parent_manifest(seed)
    assert parent.resolve()==Path(m["parent_path"]).resolve()
    if record is not None:
        assert all(record[k]==m[k] for k in ("parent_path","parent_sha256","parent_code_sha","export_path","export_sha256"))
    a=json.loads((parent.parents[1]/"acceptance.json").read_text())
    assert a["accepted"] and a["parent_sha256"]==m["parent_sha256"]
    assert a["updates"]=={"critic":3000000,"actor":3000000,"alpha":3000000}
    saved=torch.load(parent,map_location="cuda",weights_only=False)
    assert saved["updates"]==3000000 and saved["train_env_steps"]==0
    assert saved["identity"]["git_commit"]==m["parent_code_sha"]
    for key in ("recipe_sha256","config_sha256","dataset_sha256","reference_sha","seed","actor_bounds"):
        assert saved["identity"][key]==ident[key], key
    cpu=json.loads((OLD/"_so2_work/validation/t010-cpu-v3/validation.json").read_text())
    assert saved["identity"]["data_identity"]["arrays"]==cpu["data_identity"]["arrays"]
    assert digest(DATASET)==DATA_SHA
    for name in ("model","target_model"):
        assert tensor_hash(saved[name])==m["network_hashes"][name]
    return saved

def export_weights(saved,parent,path,ident):
    m=parent_manifest(ident["seed"])
    assert not path.exists()
    shutil.copy2(m["export_path"],path)
    assert digest(path)==m["export_sha256"]
    safe=torch.load(path,map_location="cpu",weights_only=True)
    assert set(safe)=={"model","target_model"}
    for name in safe:
        assert all(type(v) is torch.Tensor for v in safe[name].values())
        assert tensor_hash(safe[name])==tensor_hash(saved[name])==m["network_hashes"][name]
    return {**m,"export_path":str(path),"historical_export_path":m["export_path"],
            "online_code_sha":ident["git_commit"],"aggregation":"mean","historical_control_sha":CONTROL_SHA}

def identity(seed):
    assert Path.cwd().resolve()==ROOT
    assert not subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip()
    v=json.loads((VALIDATION/"validation.json").read_text())
    sha=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    assert v["status"]=="passed" and v["code_sha"]==sha
    return {"git_commit":sha,"recipe_sha256":digest(RECIPE),"config_sha256":digest(CONFIG),
            "reference_sha":REFERENCE_SHA,"dataset_sha256":DATA_SHA,"seed":seed,"actor_bounds":[-20,2],
            "torch":torch.__version__,"cuda":torch.version.cuda,"python":sys.version,
            "gpu":torch.cuda.get_device_name(0),"command":sys.argv,"cwd":str(ROOT),
            "actor_dropout_samples":4,"actor_mask_seed":130013,"actor_q_aggregation":"mean","thread_id":THREAD,"plan_sha":PLAN_SHA}

def make_policy(device="cuda",seed=0,aggregation="mean",samples=4):
    import runpy
    from ding.config import compile_config
    from ding.policy import create_policy
    fresh=runpy.run_path(str(CONFIG))
    fresh["main_config"].policy.cuda=device=="cuda"
    if aggregation is not None:
        fresh["main_config"].policy.learn.actor_q_aggregation=aggregation
    fresh["main_config"].policy.learn.actor_dropout_samples=samples
    fresh["main_config"].policy.learn.actor_mask_seed=130013
    cfg=compile_config(fresh["main_config"],seed=seed,env=None,auto=True,create_cfg=fresh["create_config"],save_cfg=False)
    cfg.policy.type += "_command"
    cfg.policy.import_names=["ding.policy.edac","ding.policy.command_mode_policy_instance"]
    return create_policy(cfg.policy,model=None,enable_field=["learn","collect","eval","command"])
