"""T012 component identities and factories; frozen algorithm cores."""
import os, runpy, shutil, subprocess
from t010_edac import *
BASE_SHA="90a67de1c0ca94deb983211425f4f819d111dc63"
PLAN_SHA="109e4ee279fb3e174d5680bd8888112f9bdf19f2"
CONTROL_SHA="fa76b2b3d985f4f822034d639beb38243e57ef16"
OFF_CONTROL_SHA="94f4190f3cc4e5df9dd335f32389e8782c08e86d"
THREAD="01a14523-599e-7451-a8f0-40e78999b4eb"
RESEARCH=Path("/home/lzy/Projects/rl_sample_efficiency_research_t012")
WORK=ROOT/"_so2_work/t012-v1"
VALIDATION=ROOT/"_so2_work/validation/t012-v1"
PYTHON="/home/lzy/Projects/SO2/.venv/bin/python"
LOCK=Path("/home/lzy/Projects/SO2/_so2_work/training_resource.lock")
GROUP=os.environ.get("T012_GROUP","combined")
SETTINGS={"dropout":(False,.005,30,31),"ln":(True,0.,32,33),"combined":(True,.005,0,0)}
assert GROUP in SETTINGS
LN,P,OFF_EXP,ON_EXP=SETTINGS[GROUP]
CONFIG=ROOT/f"scripts/repro/configs/halfcheetah_medium_replay_t012_{GROUP}.py"
RECIPE=ROOT/f"scripts/repro/configs/t012_{GROUP}_recipe.json"

def run_path(group,seed,phase):
    exp=SETTINGS[group][2 if phase=="offline" else 3]
    return ROOT/f"_so2_work/runs/EXP-{exp:03}/EXP-{exp:03}-s{seed}-attempt1"

def identity(seed):
    assert Path.cwd().resolve()==ROOT and GROUP in ("dropout","ln")
    assert not subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip()
    sha=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    v=json.loads((VALIDATION/"validation.json").read_text())
    assert v["status"]=="passed" and v["code_sha"]==sha
    return {"git_commit":sha,"recipe_sha256":digest(RECIPE),"config_sha256":digest(CONFIG),
      "reference_sha":REFERENCE_SHA,"dataset_sha256":DATA_SHA,"seed":seed,"actor_bounds":[-20,2],
      "torch":torch.__version__,"cuda":torch.version.cuda,"python":sys.version,
      "gpu":torch.cuda.get_device_name(0),"command":sys.argv,"cwd":str(ROOT),
      "group":GROUP,"layer_norm":LN,"dropout_p":P,"offline_actor_aggregation":"min",
      "online_actor_aggregation":"mean","thread_id":THREAD,"plan_sha":PLAN_SHA}

def make_policy(device="cuda",seed=0,aggregation="mean"):
    from ding.config import compile_config
    from ding.policy import create_policy
    fresh=runpy.run_path(str(CONFIG));fresh["main_config"].policy.cuda=device=="cuda"
    fresh["main_config"].policy.learn.actor_q_aggregation=aggregation
    cfg=compile_config(fresh["main_config"],seed=seed,env=None,auto=True,create_cfg=fresh["create_config"],save_cfg=False)
    cfg.policy.type+="_command";cfg.policy.import_names=["ding.policy.edac","ding.policy.command_mode_policy_instance"]
    return create_policy(cfg.policy,model=None,enable_field=["learn","collect","eval","command"])

def checked_parent(parent,ident,record=None):
    expected=run_path(GROUP,ident["seed"],"offline")/"ckpt/update_3000000.pth"
    assert parent.resolve()==expected and record is not None
    assert record["parent_path"]==str(expected) and digest(parent)==record["parent_sha256"]
    a=json.loads((parent.parents[1]/"acceptance.json").read_text())
    assert a["accepted"] and not a["errors"] and a["parent_sha256"]==digest(parent)
    assert a["updates"]=={"critic":3000000,"actor":3000000,"alpha":3000000}
    saved=torch.load(parent,map_location="cuda",weights_only=False)
    assert saved["updates"]==3000000 and saved["train_env_steps"]==0
    for k in ("git_commit","recipe_sha256","config_sha256","dataset_sha256","reference_sha","seed","actor_bounds","group","layer_norm","dropout_p"):
        assert saved["identity"][k]==ident[k],k
    assert digest(DATASET)==DATA_SHA
    frozen_data=json.loads(subprocess.check_output(["git","show",f"{OFF_CONTROL_SHA}:results/raw/EXP-025/EXP-025-s0-attempt1/run_meta.json"],cwd=RESEARCH,text=True))["data_identity"]
    assert saved["identity"]["data_identity"]==frozen_data
    return saved

from t010_loading import export_weights,configure_load

def make_model(device="cpu", regularized=True, actor_bounds=(-20., 2.), layer_norm=None, dropout_p=None):
    model = ENSEMBLEQAC(
        17, 6, "reparameterization", twin_critic=True,
        actor_head_hidden_size=256, actor_head_layer_num=2,
        critic_head_hidden_size=256, critic_ensemble_size=2,
        critic_layer_norm=(LN if layer_norm is None else layer_norm), critic_dropout_rate=(P if dropout_p is None else dropout_p),
        actor_log_sigma_bounds=actor_bounds)
    # Match the official EDAC actor initializer, including its .001 output range.
    for layer in (model.actor[0], model.actor[2].main[0], model.actor[2].main[2]):
        ptu.fanin_init(layer.weight); layer.bias.data.fill_(.1)
    for layer in (model.actor[2].mu, model.actor[2].log_sigma_layer):
        layer.weight.data.uniform_(-.001, .001); layer.bias.data.uniform_(-.001, .001)
    return model.to(device)
