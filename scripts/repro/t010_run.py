"""One seed0 offline run or one online run, with no retries or best-parent selection."""
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

RECIPE = ROOT / "scripts/repro/configs/t010_recipe.json"
CONFIG = ROOT / "scripts/repro/configs/halfcheetah_medium_replay_t010.py"
GPU_CHECK = ROOT / "_so2_work/validation/t010-gpu-v1/validation.json"


def rows(path):
    if not Path(path).exists(): return []
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def append(path, value):
    with Path(path).open("a") as f:
        f.write(json.dumps(value, allow_nan=False) + "\n"); f.flush()


def identity():
    assert Path.cwd().resolve() == ROOT
    assert not subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip()
    assert json.loads(GPU_CHECK.read_text())["status"] == "passed"
    return {"git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
            "recipe_sha256":digest(RECIPE),"config_sha256":digest(CONFIG),"reference_sha":REFERENCE_SHA,
            "dataset_sha256":DATA_SHA,"seed":0,"actor_bounds":[-20,2],
            "torch":torch.__version__,"cuda":torch.version.cuda,"python":sys.version,
            "gpu":torch.cuda.get_device_name(0),"command":sys.argv,"cwd":str(ROOT)}


def finite_model(model):
    assert all(torch.isfinite(value).all() for value in model.state_dict().values()), "nonfinite network tensor"


def interrupted(signum, frame):
    raise RuntimeError(f"terminated by signal {signum}; no automatic restart")


def offline(run):
    from ding.envs import get_vec_env_setting, create_env_manager
    from ding.config import compile_config
    from ding.worker import InteractionSerialEvaluator
    import d4rl
    ident=identity(); data, data_meta=load_dataset()
    ident["data_identity"] = data_meta
    seed_all(0,True); trainer=OfflineEDAC(make_model("cuda"))
    meta={**ident,"status":"running","phase":"offline","exp_id":"EXP-025","start_unix":time.time(),
          "data_identity":data_meta,"random_initialization":True,"train_env_steps":0}
    atomic_json(run/"run_meta.json",meta)
    rng=rng_state()
    fresh=runpy.run_path(str(CONFIG)); fresh["main_config"].exp_name=str(run.relative_to(ROOT))
    cfg=compile_config(fresh["main_config"],seed=0,env=None,auto=True,create_cfg=fresh["create_config"],save_cfg=False)
    env_fn, _, eval_cfg=get_vec_env_setting(cfg.env)
    envs=create_env_manager(cfg.env.manager,[partial(env_fn,cfg=c) for c in eval_cfg]); envs.seed(0,dynamic_seed=True)
    eval_policy=make_policy("cuda")
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
        _,check=continuity(trainer,saved,"cuda",batch)
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


def online(run,parent):
    from ding.entry import serial_pipeline_offline2online
    from t003_accept import check as accept
    ident=identity(); acceptance=json.loads((parent.parents[1]/"acceptance.json").read_text())
    assert acceptance["accepted"] and parent.name=="update_3000000.pth"
    assert digest(parent)==acceptance["parent_sha256"]
    saved=torch.load(parent,map_location="cuda",weights_only=False)
    assert saved["updates"]==3000000 and saved["train_env_steps"]==0 and saved["identity"]["git_commit"]==ident["git_commit"]
    assert saved["identity"]["recipe_sha256"]==ident["recipe_sha256"]
    cpu_data = json.loads((ROOT/"_so2_work/validation/t010-cpu-v3/validation.json").read_text())["data_identity"]
    assert saved["identity"]["data_identity"]["arrays"] == cpu_data["arrays"]
    probe_trainer=OfflineEDAC(make_model("cuda"));probe_trainer.restore(saved)
    seed_all(0,True)
    # Fixed synthetic observations/actions only: no interaction and no update.
    probe={"observations":torch.linspace(-1,1,256*17,device="cuda").reshape(256,17),
           "actions":torch.linspace(-1,1,256*6,device="cuda").reshape(256,6)}
    _,transition=continuity(probe_trainer,saved,"cuda",probe)
    atomic_json(run/"stage_continuity.json",{**transition,"parent_sha256":digest(parent)})
    del probe_trainer,saved
    fresh=runpy.run_path(str(CONFIG));cfg,create_cfg=fresh["main_config"],fresh["create_config"]
    cfg.exp_name=str(run.relative_to(ROOT));cfg.policy.learn.learner.load_path=str(parent)
    cfg.policy.learn.learner.hook.load_ckpt_before_run=str(parent)
    meta={**ident,"status":"running","phase":"online","exp_id":"EXP-026","start_unix":time.time(),
          "checkpoint_path":str(parent),"checkpoint_sha256":digest(parent),"parent_offline_exp":"EXP-025",
          "aggregation":"min","dropout_p":.005,"new_online_buffer":True}
    atomic_json(run/"run_meta.json",meta)
    try:
        policy,_=serial_pipeline_offline2online([cfg,create_cfg],seed=0,max_env_steps=100000,
                max_wall_clock_sec=28800,eval_env_step_interval=2500,eval_n_episode=20,
                metrics_path=str(run/"metrics.jsonl"),policy_probe_path=str(run/"live_policy_probes.jsonl"),
                rng_snapshot_dir=str(run/"rng"))
        final=next(x for x in reversed(rows(run/"metrics.jsonl")) if x["event"]=="final")
        assert final["env_steps"]==100000
        assert final["initial_buffer_counts"]=={"online":0,"mixed":201798}
        probe=rows(run/"live_policy_probes.jsonl")[-1];obs=torch.tensor(probe["observation"])
        ckpt=run/"ckpt/envstep_100000.pth.tar";loaded=make_policy("cuda")
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
    args=parser.parse_args();assert not args.run_dir.exists();args.run_dir.mkdir(parents=True)
    assert args.run_dir.resolve().is_relative_to(ROOT/"_so2_work/runs")
    torch.set_num_threads(2)
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    if args.phase=="offline": assert args.parent is None;offline(args.run_dir)
    else: assert args.parent is not None;online(args.run_dir,args.parent)

if __name__=="__main__": main()
