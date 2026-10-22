"""T013 online-only entry; actor K4 mask average, frozen T010/T011 pipeline."""
import argparse
import math
import runpy
import signal
import subprocess
import time
import traceback
from t010_edac import *
from t010_validate import continuity
from t010_run import rows, finite_model, interrupted
from t013_common import *
from t010_loading import configure_load
from t011_guards import install_guards

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
    cfg.policy.learn.actor_dropout_samples=4
    cfg.policy.learn.actor_mask_seed=130013
    cfg.exp_name=str(run.relative_to(ROOT));configure_load(cfg,init_path)
    effective=json.loads(RECIPE.read_text())
    effective.update(seed=seed,actor_aggregation="mean",actor_dropout_samples=4,actor_mask_seed=130013,online_exp="EXP-034",online_wall_sec=budget,
                     preflight_update_limit=None,preflight_wall_sec=None,preflight_fixed_batch_calls=2869,plan_sha=PLAN_SHA)
    atomic_json(run/"effective_recipe.json",effective)
    meta={**ident,"status":"running","phase":"online","exp_id":exp_id,"start_unix":time.time(),
          "checkpoint_path":str(parent),"checkpoint_sha256":digest(parent),"parent_offline_exp":"EXP-025" if seed==0 else "EXP-027",
          "pretrain_seed":seed,"online_seed":seed,
          "aggregation":"mean","dropout_p":.005,"new_online_buffer":True,
          "initialization":manifest,"max_wall_clock_sec":budget,"effective_recipe_sha256":digest(run/"effective_recipe.json"),
          "source_sha256":{str(p.relative_to(ROOT)):digest(p) for p in [ROOT/"SO2/ding/policy/edac.py", *sorted((ROOT/"scripts/repro").glob("t013_*.py"))]},
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
    p=argparse.ArgumentParser()
    p.add_argument("--run-dir",type=Path,required=True)
    p.add_argument("--parent",type=Path,required=True)
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--parent-record",type=Path,required=True)
    p.add_argument("--wall-seconds",type=float,required=True)
    a=p.parse_args()
    assert a.seed == 0 and a.run_dir.resolve()==run_path(a.seed)
    assert not a.run_dir.exists(); a.run_dir.mkdir(parents=True)
    torch.set_num_threads(2)
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    online(a.run_dir,a.parent,a.seed,"EXP-034",json.loads(a.parent_record.read_text()),a.wall_seconds)

if __name__=="__main__": main()
