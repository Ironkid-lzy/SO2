"""Frozen A1 online loop with T018 new-parent identity and admission only."""
import argparse,os,signal,traceback
from t014_a1_common import *
from t019_common import *
from t014_a1_mask import install
from t010_run import rows,finite_model,interrupted
from t011_guards import install_guards
RUN=ONLINE


def checked_weights():
    parent,saved=checked_new_parent()
    m=read(WORK/'online-parent-check/initialization_manifest.json')
    assert digest(parent)==m['parent_sha256']
    assert digest(m['export_path'])==m['export_sha256']
    weights=torch.load(m['export_path'],map_location='cpu',weights_only=True)
    assert set(weights)=={'model','target_model'}
    for k in weights:
        assert tensor_hash(weights[k])==tensor_hash(saved[k])==m['network_hashes'][k]
    return weights,m


def initial_only(out,probe_parent=None):
    from ding.entry import serial_pipeline_offline2online
    frozen()
    assert probe_parent is None, 'No new T019 benchmark branch authorized'
    phase_admission('online',30)
    weights,m=export_parent(out)
    install('matched')
    cfg,create=setup_config('cuda',str(out.relative_to(ROOT)))
    cfg.policy.learn.learner.load_path=m['export_path']
    cfg.policy.learn.learner.hook.load_ckpt_before_run=m['export_path']
    seed_all(SEED,True)
    policy,_=serial_pipeline_offline2online([cfg,create],seed=SEED,max_env_steps=0,
        max_wall_clock_sec=300,eval_env_step_interval=2500,eval_n_episode=20,
        metrics_path=str(out/'metrics.jsonl'))
    assert policy._forward_learn_cnt==0
    assert tensor_hash(policy._model.state_dict())==m['network_hashes']['model']
    assert tensor_hash(policy._target_model.state_dict())==m['network_hashes']['target_model']
    assert not policy._optimizer_policy.state and not policy._optimizer_q.state
    assert not policy._auto_alpha and float(policy._alpha.item())==float(np.float32(.2))
    e=[e for e in rows(out/'metrics.jsonl') if e['event']=='evaluation']
    assert len(e)==1 and e[0]['env_steps']==0 and len(e[0]['returns'])==20
    atomic_json(out/'validation.json',dict(passed=True,parent_sha256=m['parent_sha256'],
        initial_returns=e[0]['returns'],evaluation_interactions=e[0]['evaluation_env_steps'],
        optimizer_policy_fresh=True,optimizer_q_fresh=True,updates=0,training_interactions=0,
        network_hashes=m['network_hashes'],actual_entry='unchanged A1 pipeline with max_env_steps0; native hook; no collection/optimizer'))

def online(seconds):
    from ding.entry import serial_pipeline_offline2online
    from t003_accept import check
    hashes=phase_admission("online",seconds)
    assert 0<seconds<18000
    with (WORK/'online-formal-start.json').open('x') as f:
        json.dump({'seed':SEED,'EXP':'EXP-040','started':time.time()},f)
    RUN.mkdir(parents=True,exist_ok=False)
    install('matched')
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    meta={'status':'running','phase':'online','seed':SEED,'pretrain_seed':SEED,'online_seed':SEED,
          'exp_id':'EXP-040','git_commit':sha,'start_unix':time.time(),'command':sys.argv,
          'config_sha256':digest(CONFIG),'protocol_sha256':digest(PROTO),'source_sha256':hashes,
          'dataset_sha256':DATA_SHA,'checkpoint_path':str(PARENT),'checkpoint_sha256':PARENT_SHA,
          'mask_coupling':'matched','actor_aggregation':'mean','dropout_p':.005,'layer_norm':True,
          'actual_batch':2560,'new_online_buffer':True,'parent_offline_exp':'EXP-039',
          'max_wall_clock_sec':seconds,'cuda':torch.version.cuda,'torch':torch.__version__,
          'python':sys.version,'gpu':torch.cuda.get_device_name(0),'no_automatic_resume':True}
    atomic_json(RUN/'run_meta.json',meta)
    guarded=None
    try:
        torch.cuda.reset_peak_memory_stats()
        weights,manifest=checked_weights()
        init=RUN/'online_init.pth';shutil.copy2(manifest['export_path'],init)
        assert digest(init)==manifest['export_sha256']
        manifest.update(export_path=str(init),online_code_sha=sha,aggregation='mean',critic_mask_coupling='matched')
        atomic_json(RUN/'initialization_manifest.json',manifest)
        shutil.copy2(PROTO,RUN/'protocol.json')
        cfg,create=setup_config('cuda',str(RUN.relative_to(ROOT)))
        # Preserve frozen initial online=False; original pipeline sets True at learning.
        assert cfg.policy.learn.online is False
        cfg.policy.learn.learner.load_path=str(init)
        cfg.policy.learn.learner.hook.load_ckpt_before_run=str(init)
        recipe=json.loads((ROOT/'scripts/repro/configs/t010_recipe.json').read_text())
        recipe.update(seed=SEED,eta=0,offline_exp='EXP-039',online_exp='EXP-040',actor_aggregation='mean',critic_mask_coupling='matched',
                      online_wall_sec=seconds,protocol_sha256=digest(PROTO),actual_online_batch=2560)
        atomic_json(RUN/'effective_recipe.json',recipe)
        # Observe actual hook-loaded learner identity before its first update.
        from ding.policy.edac import EDACPolicy
        original=EDACPolicy._forward_learn
        def parent_guard(self,data):
            if self._forward_learn_cnt==0:
                assert tensor_hash(self._model.state_dict())==manifest['network_hashes']['model']
                assert tensor_hash(self._target_model.state_dict())==manifest['network_hashes']['target_model']
                atomic_json(RUN/'actual-parent-hook.json',{'strict_tensor_hashes_match':True,'critic_updates_before':0})
            return original(self,data)
        EDACPolicy._forward_learn=parent_guard
        guarded=install_guards(RUN)
        from ding.worker import InteractionSerialEvaluator
        base_eval=InteractionSerialEvaluator.eval
        expected=read(WORK/'online-parent-check/validation.json')
        assert expected['passed'] and expected['parent_sha256']==PARENT_SHA
        checked=[]
        def initial_eval_guard(self,*args,**kwargs):
            result=base_eval(self,*args,**kwargs)
            if not checked:
                assert list(self.last_episode_reward)==expected['initial_returns'], 'new-parent independent initial eval mismatch'
                atomic_json(RUN/'actual-initial-eval.json',dict(initial_returns_exact_new_parent=True,returns=list(self.last_episode_reward),parent_sha256=PARENT_SHA))
                checked.append(True)
            return result
        InteractionSerialEvaluator.eval=initial_eval_guard
        seed_all(SEED,True)
        policy,_=serial_pipeline_offline2online([cfg,create],seed=SEED,max_env_steps=100000,
                    max_wall_clock_sec=seconds,eval_env_step_interval=2500,eval_n_episode=20,
                    metrics_path=str(RUN/'metrics.jsonl'),policy_probe_path=str(RUN/'live_policy_probes.jsonl'),
                    rng_snapshot_dir=str(RUN/'rng'))
        final=next(r for r in reversed(rows(RUN/'metrics.jsonl')) if r['event']=='final')
        assert final['env_steps']==100000 and final['batch_size']==2560
        assert final['updates']=={'critic':950000,'actor':95000,'alpha':0}
        assert policy._a1_paired_updates==950000
        finite_model(policy._model);finite_model(policy._target_model)
        probe=rows(RUN/'live_policy_probes.jsonl')[-1]
        obs=torch.tensor(probe['observation']);before=torch.tensor(probe['live_action_before_save'])
        ckpt=RUN/'ckpt/envstep_100000.pth.tar'
        fresh,_=make_policy('cuda')
        fresh._load_state_dict_learn(torch.load(ckpt,map_location='cpu',weights_only=False))
        live=policy._forward_eval({0:obs})[0]['action'].cpu()
        reloaded=fresh._forward_eval({0:obs})[0]['action'].cpu()
        atomic_json(RUN/'policy_reload.json',{'step':100000,'equal_live_after':bool(torch.allclose(before,live,atol=1e-6,rtol=0)),
                    'equal_reloaded':bool(torch.allclose(before,reloaded,atol=1e-6,rtol=0)),
                    'checkpoint_path':str(ckpt),'checkpoint_sha256':digest(ckpt),'checkpoint_size':ckpt.stat().st_size})
        meta.update(status='completed',end_unix=time.time(),final_env_steps=100000,
                    paired_critic_updates=policy._a1_paired_updates,
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
        atomic_json(RUN/'run_meta.json',meta)
        result=check(RUN,100000)
        assert result['accepted'],result['errors']
    except BaseException as exc:
        meta.update(status='partial',error=repr(exc),traceback=traceback.format_exc(),end_unix=time.time())
        atomic_json(RUN/'run_meta.json',meta)
        policy=guarded.get('policy') if guarded else None
        if policy is not None:
            atomic_torch(RUN/'partial'/f'runner_learner_{policy._forward_learn_cnt}.pth.tar',policy.learn_mode.state_dict())
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wall-seconds',type=float);p.add_argument('--initial-only',type=Path);p.add_argument('--probe-parent',type=Path);a=p.parse_args()
    torch.set_num_threads(2)
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    if a.initial_only:
        initial_only(a.initial_only,a.probe_parent)
    else:
        assert a.probe_parent is None and a.wall_seconds
        PARENT=OFFLINE/'ckpt/update_3000000.pth';PARENT_SHA=read(WORK/'offline-sealed.json')['parent_sha256']
        online(a.wall_seconds)
