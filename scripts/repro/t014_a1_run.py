"""One admitted seed0 run only; frozen T011 pipeline with opt-in critic pairing."""
import argparse
import os
import signal
import traceback
from t014_a1_common import *
from t014_a1_mask import install
from t010_run import rows, finite_model, interrupted
from t011_guards import install_guards


def admission(seconds):
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    hashes=source_hashes()
    for name in ['validation-cpu-final.json','validation-cuda.json']:
        val=json.loads((WORK/name).read_text())
        assert val['status']=='passed' and val['source_sha256']==hashes
        assert val['protocol_sha256']==digest(PROTO)
    val=json.loads((WORK/'validation-cuda.json').read_text())
    assert val['benchmark']['formal_admitted']
    # Revalidate the measured 30%-margin estimate at actual dispatch time.
    ledger=json.loads((ROOT/'_so2_work/t014/budget.json').read_text())
    protocol=json.loads(PROTO.read_text())
    job=next(j for j in ledger['jobs'] if j['job']=='a1-online-s0-v1')
    assert job['status']=='running' and job['pid']==os.getpid()
    elapsed=max(0,time.time()-job['start_unix'])
    stage_remaining=protocol['stage_budget']['absolute_T014_job_ledger_limit_for_A1']-ledger['used_seconds']
    outer_remaining=job['limit_seconds']-elapsed
    estimated=val['benchmark']['estimated_formal_seconds_with30percent']
    assert 0<seconds and seconds+30<=job['limit_seconds']
    available=min(stage_remaining-60,outer_remaining-60,seconds-60)
    assert estimated<=available, (estimated,available)
    dispatch_budget=dict(estimated_seconds_with30percent=estimated,stage_remaining_seconds=stage_remaining,
                         outer_remaining_seconds=outer_remaining,inner_seconds=seconds,
                         outer_limit_seconds=job['limit_seconds'],available_after_reserve_seconds=available,
                         checked_unix=time.time(),passed=True)
    review=json.loads((WORK/'review-approved.json').read_text())
    assert review['approved'] and review['source_sha256']==hashes
    assert review['protocol_sha256']==digest(PROTO)
    assert not list(Path('/home/lzy/Projects').glob('SO2*/_so2_work/runs/EXP-035*'))
    assert shutil.disk_usage(ROOT).free>8*1024**3
    atomic_json(WORK/'formal-admission.json',dispatch_budget)
    return hashes


def online(seconds):
    from ding.entry import serial_pipeline_offline2online
    from t003_accept import check
    hashes=admission(seconds)
    assert 0<seconds<18000
    with (WORK/'formal-start.json').open('x') as f:
        json.dump({'seed':0,'EXP':'EXP-035','started':time.time()},f)
    RUN.mkdir(parents=True,exist_ok=False)
    install('matched')
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    meta={'status':'running','phase':'online','seed':0,'pretrain_seed':0,'online_seed':0,
          'exp_id':'EXP-035','git_commit':sha,'start_unix':time.time(),'command':sys.argv,
          'config_sha256':digest(CONFIG),'protocol_sha256':digest(PROTO),'source_sha256':hashes,
          'dataset_sha256':DATA_SHA,'checkpoint_path':str(PARENT),'checkpoint_sha256':PARENT_SHA,
          'mask_coupling':'matched','actor_aggregation':'mean','dropout_p':.005,'layer_norm':True,
          'actual_batch':2560,'new_online_buffer':True,'parent_offline_exp':'EXP-025',
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
        recipe.update(seed=0,online_exp='EXP-035',actor_aggregation='mean',critic_mask_coupling='matched',
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
        seed_all(0,True)
        policy,_=serial_pipeline_offline2online([cfg,create],seed=0,max_env_steps=100000,
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
    parser=argparse.ArgumentParser();parser.add_argument('--wall-seconds',type=float,required=True)
    args=parser.parse_args();torch.set_num_threads(2)
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    online(args.wall_seconds)
