"""T010 offline loop with only eta0 trainer and task-local identity/budget."""
import argparse,runpy,signal,time,traceback,math
from functools import partial
from t019_common import *
from t018_edac import ParameterizedEDAC
from t010_run import rows,append,finite_model,interrupted
from t010_validate import continuity,make_policy

def offline(run, seed=SEED, exp_id="EXP-039", stop_at=None, seconds=None):
    from ding.envs import get_vec_env_setting, create_env_manager
    from ding.config import compile_config
    from ding.worker import InteractionSerialEvaluator
    import d4rl
    ident=identity(seed); data, data_meta=load_dataset()
    implementation_sha="f5dacc442f73e94e085988600fa2ff8cb8ad95b0"
    ident.update(training_implementation_sha=implementation_sha, stop_at_unix=stop_at)
    effective=json.loads(RECIPE.read_text());effective.update(seed=seed,offline_exp=exp_id,online_exp="EXP-040",eta=0,offline_wall_sec=seconds)
    atomic_json(run/"effective_recipe.json",effective)
    ident["effective_recipe_sha256"]=digest(run/"effective_recipe.json")
    ident["data_identity"] = data_meta
    seed_all(seed,True); trainer=ParameterizedEDAC(make_model("cuda"),es_coefficient=0)
    archived=initial_training_rng(trainer)
    assert data_meta["arrays"]==archived["identity"]["data_identity"]["arrays"]
    atomic_json(run/"initialization.json",dict(archive_update0_sha256=ARCHIVE_SHA,fresh_model_hash=tensor_hash(trainer.model.state_dict()),fresh_target_hash=tensor_hash(trainer.target.state_dict()),optimizers="fresh empty",training_RNG="restored archive0; historical batch order not independently logged"))
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
    eval_steps=0; manifest=[]; start=PROCESS_START
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
        if trainer.updates==0:
            assert list(returns)==historical_initial_returns(), "untrained initial evaluation differs from own historical seed"
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
            assert time.monotonic()-start<seconds, "T019 offline finite inner timeout"
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
        meta.update(status="completed",end_unix=time.time(),peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),**acceptance)
        atomic_json(run/"run_meta.json",meta);atomic_json(run/"state.json",meta)
    except BaseException as exc:
        meta.update(status="partial",error=repr(exc),traceback=traceback.format_exc(),end_unix=time.time(),
                    gradient_updates=trainer.updates,evaluation_interactions_total=eval_steps)
        atomic_json(run/"run_meta.json",meta);atomic_json(run/"state.json",meta)
        atomic_torch(run/"partial"/f"update_{trainer.updates}.pth",trainer.snapshot(ident))
        raise
    finally: evaluator.close()

if __name__=='__main__':
    PROCESS_START=time.monotonic()
    p=argparse.ArgumentParser();p.add_argument('--wall-seconds',type=float,required=True);a=p.parse_args()
    torch.set_num_threads(2);torch.cuda.reset_peak_memory_stats()
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    phase_admission('offline',a.wall_seconds)
    assert not OFFLINE.exists();OFFLINE.mkdir(parents=True)
    offline(OFFLINE,seed=SEED,exp_id="EXP-039",seconds=a.wall_seconds)
