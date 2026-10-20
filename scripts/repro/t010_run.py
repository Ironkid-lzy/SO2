"""One offline seed or the original online seed0; no retries or parent selection."""
import argparse
import math
import runpy
import signal
import subprocess
import time
import traceback
from functools import partial

from t010_edac import *
from t010_validate import continuity, make_policy
from t010_loading import checked_parent, export_weights, configure_load, remaining_online_seconds

RECIPE = ROOT / "scripts/repro/configs/t010_recipe.json"
CONFIG = ROOT / "scripts/repro/configs/halfcheetah_medium_replay_t010.py"
GPU_CHECK = ROOT / "_so2_work/validation/t010-gpu-v2/validation.json"


def rows(path):
    if not Path(path).exists(): return []
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def append(path, value):
    with Path(path).open("a") as f:
        f.write(json.dumps(value, allow_nan=False) + "\n"); f.flush()


def identity(seed=0):
    assert Path.cwd().resolve() == ROOT
    assert not subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip()
    assert json.loads(GPU_CHECK.read_text())["status"] == "passed"
    return {"git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
            "recipe_sha256":digest(RECIPE),"config_sha256":digest(CONFIG),"reference_sha":REFERENCE_SHA,
            "dataset_sha256":DATA_SHA,"seed":seed,"actor_bounds":[-20,2],
            "torch":torch.__version__,"cuda":torch.version.cuda,"python":sys.version,
            "gpu":torch.cuda.get_device_name(0),"command":sys.argv,"cwd":str(ROOT)}


def finite_model(model):
    assert all(torch.isfinite(value).all() for value in model.state_dict().values()), "nonfinite network tensor"


def interrupted(signum, frame):
    raise RuntimeError(f"terminated by signal {signum}; no automatic restart")


def offline(run, seed=0, exp_id="EXP-025", stop_at=None):
    from ding.envs import get_vec_env_setting, create_env_manager
    from ding.config import compile_config
    from ding.worker import InteractionSerialEvaluator
    import d4rl
    ident=identity(seed); data, data_meta=load_dataset()
    implementation_sha="f5dacc442f73e94e085988600fa2ff8cb8ad95b0"
    for source in ("SO2", "scripts/repro/t010_edac.py"):
        old=subprocess.check_output(["git","rev-parse",f"{implementation_sha}:{source}"],text=True).strip()
        current=subprocess.check_output(["git","rev-parse",f"HEAD:{source}"],text=True).strip()
        assert old==current, f"training implementation changed: {source}"
    ident.update(training_implementation_sha=implementation_sha, stop_at_unix=stop_at)
    effective=json.loads(RECIPE.read_text());effective.update(seed=seed,offline_exp=exp_id)
    if seed!=0:effective["online_exp"]=None  # Online stage is not scheduled by this run.
    atomic_json(run/"effective_recipe.json",effective)
    ident["effective_recipe_sha256"]=digest(run/"effective_recipe.json")
    ident["data_identity"] = data_meta
    seed_all(seed,True); trainer=OfflineEDAC(make_model("cuda"))
    meta={**ident,"status":"running","phase":"offline","exp_id":exp_id,"start_unix":time.time(),
          "data_identity":data_meta,"random_initialization":True,"train_env_steps":0}
    atomic_json(run/"run_meta.json",meta)
    rng=rng_state()
    fresh=runpy.run_path(str(CONFIG)); fresh["main_config"].exp_name=str(run.relative_to(ROOT))
    cfg=compile_config(fresh["main_config"],seed=seed,env=None,auto=True,create_cfg=fresh["create_config"],save_cfg=False)
    env_fn, _, eval_cfg=get_vec_env_setting(cfg.env)
    envs=create_env_manager(cfg.env.manager,[partial(env_fn,cfg=c) for c in eval_cfg]); envs.seed(seed,dynamic_seed=True)
    eval_policy=make_policy("cuda",seed=seed)
    evaluator=InteractionSerialEvaluator(cfg.policy.eval.evaluator,envs,eval_policy.eval_mode,exp_name=cfg.exp_name)
    restore_rng(rng)
    eval_steps=0; manifest=[]; start=time.monotonic()
    def evaluate_save():
        nonlocal eval_steps
        finite_model(trainer.model);finite_model(trainer.target)
        rng=rng_state(); modes=(trainer.model.training,trainer.target.training)
        try:
            eval_policy._load_state_dict_learn({"model":trainer.model.state_dict(),"target_model":trainer.target.state_dict()})
            _, reward=evaluator.eval(None,train_iter=trainer.updates,envstep=0,n_episode=20)
            returns=evaluator.last_episode_reward
        finally:
            restore_rng(rng);trainer.model.train(modes[0]);trainer.target.train(modes[1])
        assert len(returns)==20 and all(math.isfinite(x) for x in returns)
        eval_steps+=evaluator.last_envstep_count
        score=100*float(d4rl.get_normalized_score("halfcheetah-medium-replay-v2",float(reward)))
        append(run/"metrics.jsonl",{"event":"evaluation","gradient_updates":trainer.updates,"train_env_steps":0,
                  "returns":returns,"return_mean":float(reward),"normalized_score_100":score,
                  "evaluation_env_steps":evaluator.last_envstep_count,"evaluation_interactions_total":eval_steps,
                  "training_rng_restored":True})
        path=run/"ckpt"/f"update_{trainer.updates}.pth"
        atomic_torch(path,trainer.snapshot(ident))
        manifest.append({"path":str(path.relative_to(run)),"size":path.stat().st_size,"sha256":digest(path),
                         "gradient_updates":trainer.updates,"backup_status":"Linux only"})
        atomic_json(run/"checkpoint_manifest.json",manifest)
    try:
        evaluate_save()
        while trainer.updates<3000000:
            assert time.monotonic()-start<172800, "48h offline process budget exhausted"
            assert stop_at is None or time.time()<stop_at, "scheduled deadline at update boundary"
            idx=np.random.randint(0,len(data["rewards"]),size=256)
            losses=trainer.update(batch_at(data,idx,"cuda"))
            assert all(math.isfinite(x) for x in losses.values()), "nonfinite EDAC loss"
            if trainer.updates%1000==0:
                finite_model(trainer.model);finite_model(trainer.target)
                state={"status":"running","gradient_updates":trainer.updates,"train_env_steps":0,
                       "alpha":float(trainer.log_alpha.detach().exp()),"last_losses":losses,
                       "wall_seconds":time.monotonic()-start,"evaluation_interactions_total":eval_steps}
                atomic_json(run/"state.json",state)
            if trainer.updates%100000==0: evaluate_save()
        events=rows(run/"metrics.jsonl")
        assert [x["gradient_updates"] for x in events]==list(range(0,3000001,100000))
        final=run/"ckpt/update_3000000.pth"
        saved=torch.load(final,map_location="cuda",weights_only=False)
        assert saved["updates"]==3000000 and saved["train_env_steps"]==0
        assert saved["online_critic_sha256"]==tensor_hash(trainer.model.critic.state_dict())
        assert saved["target_critic_sha256"]==tensor_hash(trainer.target.critic.state_dict())
        batch=batch_at(data,np.arange(256),"cuda")
        _,check=continuity(trainer,saved,"cuda",batch,seed=seed)
        check.update(parent_path=str(final),parent_sha256=digest(final),offline_updates=3000000)
        atomic_json(run/"stage_continuity.json",check)
        acceptance={"accepted":True,"errors":[],"updates":{"critic":3000000,"actor":3000000,"alpha":3000000},
                    "train_env_steps":0,"evaluation_count":31,"evaluation_interactions_total":eval_steps,
                    "parent_path":str(final),"parent_sha256":digest(final),"continuity":check,
                    "wall_seconds":time.monotonic()-start}
        atomic_json(run/"acceptance.json",acceptance)
        meta.update(status="completed",end_unix=time.time(),**acceptance)
        atomic_json(run/"run_meta.json",meta);atomic_json(run/"state.json",meta)
    except BaseException as exc:
        meta.update(status="partial",error=repr(exc),traceback=traceback.format_exc(),end_unix=time.time(),
                    gradient_updates=trainer.updates,evaluation_interactions_total=eval_steps)
        atomic_json(run/"run_meta.json",meta);atomic_json(run/"state.json",meta)
        atomic_torch(run/"partial"/f"update_{trainer.updates}.pth",trainer.snapshot(ident))
        raise
    finally: evaluator.close()


def online(run,parent,seed=0,exp_id="EXP-026",parent_record=None,wall_seconds=None):
    from ding.entry import serial_pipeline_offline2online
    from t003_accept import check as accept
    ident=identity(seed)
    budget=remaining_online_seconds() if wall_seconds is None else wall_seconds
    assert 0<budget<=28800
    saved=checked_parent(parent,ident,parent_record)
    init_path=run/"online_init.pth"
    manifest=export_weights(saved,parent,init_path,ident)
    atomic_json(run/"initialization_manifest.json",manifest)
    probe_trainer=OfflineEDAC(make_model("cuda"));probe_trainer.restore(saved)
    seed_all(seed,True)
    # Fixed synthetic observations/actions only: no interaction and no update.
    probe={"observations":torch.linspace(-1,1,256*17,device="cuda").reshape(256,17),
           "actions":torch.linspace(-1,1,256*6,device="cuda").reshape(256,6)}
    _,transition=continuity(probe_trainer,saved,"cuda",probe,seed=seed)
    atomic_json(run/"stage_continuity.json",{**transition,"parent_sha256":digest(parent)})
    del probe_trainer,saved
    fresh=runpy.run_path(str(CONFIG));cfg,create_cfg=fresh["main_config"],fresh["create_config"]
    cfg.exp_name=str(run.relative_to(ROOT));configure_load(cfg,init_path)
    meta={**ident,"status":"running","phase":"online","exp_id":exp_id,"start_unix":time.time(),
          "checkpoint_path":str(parent),"checkpoint_sha256":digest(parent),"parent_offline_exp":"EXP-025" if seed==0 else "EXP-027",
          "pretrain_seed":seed,"online_seed":seed,
          "aggregation":"min","dropout_p":.005,"new_online_buffer":True,
          "initialization":manifest,"max_wall_clock_sec":budget}
    atomic_json(run/"run_meta.json",meta)
    try:
        policy,_=serial_pipeline_offline2online([cfg,create_cfg],seed=seed,max_env_steps=100000,
                max_wall_clock_sec=budget,eval_env_step_interval=2500,eval_n_episode=20,
                metrics_path=str(run/"metrics.jsonl"),policy_probe_path=str(run/"live_policy_probes.jsonl"),
                rng_snapshot_dir=str(run/"rng"))
        final=next(x for x in reversed(rows(run/"metrics.jsonl")) if x["event"]=="final")
        assert final["env_steps"]==100000
        assert final["initial_buffer_counts"]=={"online":0,"mixed":201798}
        probe=rows(run/"live_policy_probes.jsonl")[-1];obs=torch.tensor(probe["observation"])
        ckpt=run/"ckpt/envstep_100000.pth.tar";loaded=make_policy("cuda",seed=seed)
        loaded._load_state_dict_learn(torch.load(ckpt,map_location="cpu",weights_only=False))
        before=torch.tensor(probe["live_action_before_save"])
        live=policy._forward_eval({0:obs})[0]["action"].cpu();fresh_action=loaded._forward_eval({0:obs})[0]["action"].cpu()
        reload={"step":100000,"equal_live_after":bool(torch.allclose(before,live,atol=1e-6,rtol=0)),
                "equal_reloaded":bool(torch.allclose(before,fresh_action,atol=1e-6,rtol=0)),
                "checkpoint_path":str(ckpt),"checkpoint_size":ckpt.stat().st_size,"checkpoint_sha256":digest(ckpt)}
        atomic_json(run/"policy_reload.json",reload)
        finite_model(policy._model);finite_model(policy._target_model)
        meta.update(status="completed",end_unix=time.time(),final_env_steps=100000)
        atomic_json(run/"run_meta.json",meta)
        result=accept(run,100000)
        assert result["accepted"],result["errors"]
    except BaseException as exc:
        meta.update(status="partial",error=repr(exc),traceback=traceback.format_exc(),end_unix=time.time())
        atomic_json(run/"run_meta.json",meta);raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--phase",choices=("offline","online"),required=True)
    parser.add_argument("--run-dir",type=Path,required=True);parser.add_argument("--parent",type=Path)
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--exp-id")
    parser.add_argument("--stop-at",type=float)
    parser.add_argument("--parent-record",type=Path)
    parser.add_argument("--wall-seconds",type=float)
    args=parser.parse_args()
    assert args.seed in range(5)
    if args.phase=="online":
        assert args.stop_at is None
        if args.seed!=0:assert args.exp_id=="EXP-028" and args.parent_record is not None and args.wall_seconds is not None
    else:assert args.parent_record is None and args.wall_seconds is None
    if args.phase=="offline" and args.seed!=0:assert args.exp_id=="EXP-027" and args.stop_at is not None
    assert not args.run_dir.exists();args.run_dir.mkdir(parents=True)
    assert args.run_dir.resolve().is_relative_to(ROOT/"_so2_work/runs")
    torch.set_num_threads(2)
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    if args.phase=="offline": assert args.parent is None;offline(args.run_dir,args.seed,args.exp_id or "EXP-025",args.stop_at)
    else:
        assert args.parent is not None
        online(args.run_dir,args.parent,args.seed,args.exp_id or "EXP-026",
               json.loads(args.parent_record.read_text()) if args.parent_record else None,args.wall_seconds)

if __name__=="__main__": main()
