"""T012 two-stage entry; no changes to frozen update equations."""
import argparse, math, runpy, signal, subprocess, time, traceback
from functools import partial
from t012_common import *
from t010_run import rows,append,finite_model,interrupted
from t010_validate import continuity as old_continuity
import t010_validate
from t012_loading import hook_check
from t011_guards import install_guards
t010_validate.make_policy=make_policy
continuity=old_continuity

def offline(run, seed=0, exp_id="EXP-025", stop_at=None):
    from ding.envs import get_vec_env_setting, create_env_manager
    from ding.config import compile_config
    from ding.worker import InteractionSerialEvaluator
    import d4rl
    ident=identity(seed); data, data_meta=load_dataset()
    ident.update(training_implementation_sha=BASE_SHA, stop_at_unix=stop_at)
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
            assert time.monotonic()-start<36000, "48h offline process budget exhausted"
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
    budget=wall_seconds
    assert 0<budget<=18000
    saved=checked_parent(parent,ident,parent_record)
    init_path=run/"online_init.pth"
    manifest=export_weights(saved,parent,init_path,ident)
    atomic_json(run/"initialization_manifest.json",manifest)
    hook_check(saved,init_path,run/"hook",seed)
    probe_trainer=OfflineEDAC(make_model("cuda"));probe_trainer.restore(saved)
    seed_all(seed,True)
    # Fixed synthetic observations/actions only: no interaction and no update.
    probe={"observations":torch.linspace(-1,1,256*17,device="cuda").reshape(256,17),
           "actions":torch.linspace(-1,1,256*6,device="cuda").reshape(256,6)}
    _,transition=continuity(probe_trainer,saved,"cuda",probe,seed=seed)
    atomic_json(run/"stage_continuity.json",{**transition,"parent_sha256":digest(parent)})
    del probe_trainer,saved
    fresh=runpy.run_path(str(CONFIG));cfg,create_cfg=fresh["main_config"],fresh["create_config"]
    cfg.policy.learn.actor_q_aggregation="mean"
    cfg.exp_name=str(run.relative_to(ROOT));configure_load(cfg,init_path)
    effective=json.loads(RECIPE.read_text())
    effective.update(seed=seed,actor_aggregation="mean",online_exp=f"EXP-{ON_EXP:03}",online_wall_sec=budget,
                     preflight_update_limit=200,plan_sha=PLAN_SHA)
    atomic_json(run/"effective_recipe.json",effective)
    meta={**ident,"status":"running","phase":"online","exp_id":exp_id,"start_unix":time.time(),
          "checkpoint_path":str(parent),"checkpoint_sha256":digest(parent),"parent_offline_exp":f"EXP-{OFF_EXP:03}",
          "pretrain_seed":seed,"online_seed":seed,
          "aggregation":"mean","dropout_p":P,"layer_norm":LN,"group":GROUP,"new_online_buffer":True,
          "initialization":manifest,"max_wall_clock_sec":budget,"effective_recipe_sha256":digest(run/"effective_recipe.json"),
          "source_sha256":{str(p.relative_to(ROOT)):digest(p) for p in [ROOT/"SO2/ding/policy/edac.py", *sorted((ROOT/"scripts/repro").glob("t011_*.py"))]},
          "requirements_sha256":digest(VALIDATION/"requirements-lock.txt")}
    atomic_json(run/"run_meta.json",meta)
    guarded=install_guards(run)
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
        atomic_json(run/"run_meta.json",meta)
        partial=guarded.get("policy")
        if partial is not None:
            atomic_torch(run/"partial"/f"learner_{partial._forward_learn_cnt}.pth.tar",partial.learn_mode.state_dict())
        raise

def main():
    p=argparse.ArgumentParser();p.add_argument("--phase",choices=("offline","online"),required=True)
    p.add_argument("--seed",type=int,required=True);p.add_argument("--wall-seconds",type=float,required=True)
    a=p.parse_args();assert a.seed in range(5) and GROUP in ("dropout","ln")
    run=run_path(GROUP,a.seed,a.phase);assert not run.exists();run.mkdir(parents=True)
    torch.set_num_threads(2);signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    if a.phase=="offline":offline(run,a.seed,f"EXP-{OFF_EXP:03}",time.time()+a.wall_seconds)
    else:
        parent=run_path(GROUP,a.seed,"offline")/"ckpt/update_3000000.pth"
        record={"parent_path":str(parent),"parent_sha256":digest(parent)}
        online(run,parent,a.seed,f"EXP-{ON_EXP:03}",record,a.wall_seconds)
if __name__=="__main__":main()
