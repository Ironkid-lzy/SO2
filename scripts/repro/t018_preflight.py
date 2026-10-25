"""One complete cost session: offline learner/eval/save and frozen A1 learner."""
import argparse,math,runpy,statistics,subprocess,time
from functools import partial
from t018_common import *
from t018_edac import ParameterizedEDAC
from t010_run import finite_model,rows
from t010_validate import make_policy as offline_policy
from t014_a1_common import make_policy as online_policy,fixed_batch
from ding.config import compile_config
from ding.envs import get_vec_env_setting,create_env_manager
from ding.worker import InteractionSerialEvaluator,BaseLearner


def preflight():
    frozen();torch.set_num_threads(2)
    from t018_control import fault_tests
    assert fault_tests()==8
    out=WORK/'cost-preflight-v1';assert not out.exists();out.mkdir()
    started=time.monotonic();data,dm=load_dataset()
    seed_all(0,True);trainer=ParameterizedEDAC(make_model('cuda'),es_coefficient=0)
    archived=initial_training_rng(trainer)
    assert dm['arrays']==archived['identity']['data_identity']['arrays']
    init_rng=rng_state()
    fresh=runpy.run_path(str(CONFIG));fresh['main_config'].exp_name=str(out.relative_to(ROOT))
    cfg=compile_config(fresh['main_config'],seed=0,env=None,auto=True,create_cfg=fresh['create_config'],save_cfg=False)
    env_fn,_,ec=get_vec_env_setting(cfg.env)
    envs=create_env_manager(cfg.env.manager,[partial(env_fn,cfg=c) for c in ec]);envs.seed(0,dynamic_seed=True)
    pol=offline_policy('cuda',seed=0)
    evaluator=InteractionSerialEvaluator(cfg.policy.eval.evaluator,envs,pol.eval_mode,exp_name=cfg.exp_name)
    restore_rng(init_rng);before_eval=rng_state()
    torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
    try:
        pol._load_state_dict_learn({'model':trainer.model.state_dict(),'target_model':trainer.target.state_dict()})
        evaluator.eval(None,train_iter=0,envstep=0,n_episode=20)
        returns=list(evaluator.last_episode_reward);eval_interactions=evaluator.last_envstep_count
        assert returns==historical_initial_returns(), 'initial offline evaluation mismatch'
    finally:
        restore_rng(before_eval);evaluator.close()
    eval_seconds=time.perf_counter()-t
    ident=identity();ident['data_identity']=dm
    blocks=[]
    for _ in range(20):
        idx=np.random.randint(0,len(data['rewards']),size=256)
        trainer.update(batch_at(data,idx,'cuda'))
    for _ in range(6):
        torch.cuda.synchronize();t=time.perf_counter()
        for __ in range(100):
            idx=np.random.randint(0,len(data['rewards']),size=256)
            losses=trainer.update(batch_at(data,idx,'cuda'))
            assert all(math.isfinite(x) for x in losses.values())
        torch.cuda.synchronize();blocks.append(time.perf_counter()-t)
    t=time.perf_counter();finite_model(trainer.model);finite_model(trainer.target)
    finite_seconds=time.perf_counter()-t
    t=time.perf_counter();parent=out/'benchmark_parent.pth'
    atomic_torch(parent,trainer.snapshot(ident));parent_hash=digest(parent)
    save_hash_seconds=time.perf_counter()-t
    offline_peak=torch.cuda.max_memory_allocated()
    # Native weights-only hook and independent eval0, with zero collection/update.
    t=time.perf_counter()
    subprocess.run([sys.executable,'scripts/repro/t018_online.py','--initial-only',str(out/'parent-load-probe'),'--probe-parent',str(parent)],check=True,timeout=300)
    parent_check_seconds=time.perf_counter()-t
    assert read(out/'parent-load-probe/validation.json')['passed']
    weights=torch.load(out/'parent-load-probe/online_init.pth',map_location='cpu',weights_only=True)
    policy,pcfg=online_policy('cuda',mode='matched')
    pcfg.policy.learn.learner.load_path=str(out/'parent-load-probe/online_init.pth')
    pcfg.policy.learn.learner.hook.load_ckpt_before_run=pcfg.policy.learn.learner.load_path
    learner=BaseLearner(pcfg.policy.learn.learner,policy.learn_mode,exp_name=str(out/'learner'))
    batch=fixed_batch(2560);online_blocks=[]
    try:
        learner.call_hook('before_run')
        assert policy._forward_learn_cnt==0
        assert tensor_hash(policy._model.state_dict())==tensor_hash(weights['model'])
        for _ in range(20):learner.train(batch,envstep=0)
        torch.cuda.reset_peak_memory_stats()
        for _ in range(6):
            torch.cuda.synchronize();t=time.perf_counter()
            for __ in range(20):learner.train(batch,envstep=0)
            torch.cuda.synchronize();online_blocks.append(time.perf_counter()-t)
        assert policy._a1_paired_updates==policy._forward_learn_cnt==140
        online_peak=torch.cuda.max_memory_allocated()
    finally:learner.close()
    hist=Path('/home/lzy/Projects/SO2_t014/_so2_work/runs/EXP-035/EXP-035-s0-attempt1')
    final=next(r for r in reversed(rows(hist/'metrics.jsonl')) if r['event']=='final')
    historic_wall=11048.167322450994
    nonlearner=max(0,historic_wall-final['timings_sec']['learn'])
    offline_est=statistics.median(blocks)/100*3000000+31*(eval_seconds+save_hash_seconds)+3000*finite_seconds+60
    online_est=max(historic_wall,statistics.median(online_blocks)/20*950000+nonlearner)+parent_check_seconds+60
    result=dict(passed=True,single_speed_session=True,fail_closed_fault_tests=8,offline_ES_coefficient=0,offline_blocks100=blocks,
        online_blocks20=online_blocks,offline_eval20_seconds=eval_seconds,offline_eval_interactions=eval_interactions,
        initial_offline_returns=returns,initial_offline_exact_historical=True,dataset_arrays_exact_historical=True,
        offline_save_full_optimizer_and_hash_seconds=save_hash_seconds,finite_model_check_seconds=finite_seconds,
        parent_independent_load_eval_seconds=parent_check_seconds,
        online_nonlearner_historical_seconds=nonlearner,online_historical_floor_seconds=historic_wall,
        offline_estimate_seconds=offline_est,online_estimate_seconds=online_est,
        offline_with30percent=1.3*offline_est,online_with30percent=1.3*online_est,
        total_with30percent=1.3*(offline_est+online_est),
        offline_peak_allocated_bytes=offline_peak,online_peak_allocated_bytes=online_peak,
        preflight_optimizer_updates=dict(offline=620,online_critic=140,online_actor=14,online_alpha=0),
        full_parent_probe_updates=620,formal_updates=0,training_environment_interactions=0,
        evaluation_interactions_total=eval_interactions+read(out/'parent-load-probe/validation.json')['evaluation_interactions'],
        cost_scope='Measured full offline updates incl sampling/scalar synchronization, native 20-episode eval and full optimizer checkpoint save/hash, finite guards; full A1 BaseLearner; historical complete online nonlearner floor plus new-parent probe and60s setup each; all multiplied1.30. Historical timing not causal speed comparison.',
        elapsed_seconds=time.monotonic()-started)
    assert result['elapsed_seconds']<1500
    atomic_json(out/'result.json',result)
    print(json.dumps(result))


if __name__=='__main__':preflight()
