"""Zero-update seed and hard-deadline regression checks."""
import argparse
import os
import subprocess
import time
from unittest.mock import patch

from t010_edac import *
from t010_validate import make_policy, continuity, check
from t010_seed1_queue import timed_command


def main(out):
    from ding.worker import BaseLearner
    from ding.worker.learner import learner_hook
    import t010_seed1_queue as queue
    torch.set_num_threads(2);out=out.resolve();out.mkdir(parents=True,exist_ok=True)
    baseline="f5dacc442f73e94e085988600fa2ff8cb8ad95b0"
    for path in ("SO2","scripts/repro/t010_edac.py","scripts/repro/configs/t010_recipe.json","scripts/repro/configs/halfcheetah_medium_replay_t010.py"):
        assert subprocess.check_output(["git","show",f"{baseline}:{path}"],cwd=ROOT) == (ROOT/path).read_bytes() if (ROOT/path).is_file() else subprocess.check_output(["git","rev-parse",f"{baseline}:{path}"],cwd=ROOT)==subprocess.check_output(["git","rev-parse",f"HEAD:{path}"],cwd=ROOT)
    seed0=ROOT/"_so2_work/runs/EXP-025/EXP-025-s0-attempt1"
    entry=next(x for x in json.loads((seed0/"checkpoint_manifest.json").read_text()) if x["gradient_updates"]==0)
    checkpoint=seed0/entry["path"];assert digest(checkpoint)==entry["sha256"]
    saved0=torch.load(checkpoint,map_location="cuda",weights_only=False)
    seed_all(0,True);original=OfflineEDAC(make_model("cuda"))
    assert tensor_hash(original.model.state_dict())==tensor_hash(saved0["model"])
    assert tensor_hash(original.target.state_dict())==tensor_hash(saved0["target_model"])
    seed_all(1,True);first=OfflineEDAC(make_model("cuda"));first_hash=tensor_hash(first.model.state_dict())
    randoms=(random.random(),np.random.rand(),torch.rand(3),torch.rand(3,device="cuda"))
    seed_all(1,True);second=OfflineEDAC(make_model("cuda"))
    repeated=(random.random(),np.random.rand(),torch.rand(3),torch.rand(3,device="cuda"))
    assert randoms[:2]==repeated[:2] and all(torch.equal(a,b) for a,b in zip(randoms[2:],repeated[2:]))
    assert first_hash==tensor_hash(second.model.state_dict())!=tensor_hash(original.model.state_dict())
    saved=first.snapshot({"seed":1,"purpose":"zero-update-validation"})
    obs=torch.linspace(-1,1,256*17,device="cuda").reshape(256,17)
    actions=torch.linspace(-1,1,256*6,device="cuda").reshape(256,6)
    expected,transition=continuity(first,saved,"cuda",{"observations":obs,"actions":actions},seed=1)
    export=out/"online_init.pth";atomic_torch(export,{name:{k:v.cpu() for k,v in saved[name].items()} for name in ("model","target_model")})
    actual=make_policy("cuda",seed=1);cfg=actual._cfg.learn.learner
    cfg.load_path=str(export);cfg.hook.load_ckpt_before_run=str(export)
    learner=BaseLearner(cfg,actual.learn_mode,exp_name=str(out.relative_to(ROOT)))
    try:
        with patch.object(learner_hook,"read_file",wraps=learner_hook.read_file) as reader:
            learner.call_hook("before_run");reader.assert_called_once_with(str(export))
        assert learner.train_iter==actual._forward_learn_cnt==0
        for name,model in (("model",actual._model),("target_model",actual._target_model)):
            assert tensor_hash(model.state_dict())==tensor_hash(saved[name])
        errors={}
        with torch.no_grad():
            for train in (False,True):
                for label,a,b in (("online",expected._model,actual._model),("target",expected._target_model,actual._target_model)):
                    a.train(train);b.train(train)
                    rng=rng_state();x=q_values(a,obs,actions);restore_rng(rng);y=q_values(b,obs,actions)
                    errors[f"{train}_{label}_q"]=check(x,y,label,tol={"atol":0,"rtol":0})
                    rng=rng_state();x=sample_actor(a.forward,obs)[0];restore_rng(rng);y=sample_actor(b.forward,obs)[0]
                    errors[f"{train}_{label}_action"]=check(x,y,label,tol={"atol":0,"rtol":0})
    finally:learner.close()
    for stubborn in (False,True):
        code="import time; time.sleep(60)" if not stubborn else "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)"
        start=time.monotonic();process=subprocess.run(timed_command([sys.executable,"-c",code],time.time()+.2),timeout=7)
        assert process.returncode in (124,-9,137) and time.monotonic()-start<6
    child_code="import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)"
    parent_code="import signal,subprocess,sys,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); p=subprocess.Popen([sys.executable,'-c',"+repr(child_code)+"]); print(p.pid,flush=True); time.sleep(60)"
    group=subprocess.run(timed_command([sys.executable,"-c",parent_code],time.time()+.3),capture_output=True,text=True,timeout=7)
    descendant=int(group.stdout.strip());stat=Path(f"/proc/{descendant}/stat")
    assert not stat.exists() or stat.read_text().rsplit(")",1)[1].split()[0]=="Z", "descendant survived hard stop"
    with patch.object(queue.time,"time",return_value=queue.SOFT):
        try:queue.main()
        except AssertionError as exc:assert "insufficient forecast" in str(exc)
        else:raise AssertionError("late launch accepted")
    result={"status":"passed","code_sha":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
            "git_dirty":bool(subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip()),
            "updates":0,"cumulative_updates":181,"train_env_steps":0,"eval_env_steps":0,
            "seed0_initial_parameters_exact":True,"seed1_reproducible_and_distinct":True,"all_four_rng_sources_repeat":True,
            "math_config_data_code_unchanged":True,"seed1_actual_hook":True,"continuity":transition,"hook_output_errors":errors,
            "deadline_term_and_kill_test":True,"descendant_process_group_killed":True,"late_launch_refused":True,"seed1_initial_hash":first_hash,
            "source_sha256":{name:digest(ROOT/"scripts/repro"/name) for name in ("t010_run.py","t010_validate.py","t010_seed1_queue.py","t010_seed1_validate.py","t010_deliver.py")}}
    atomic_json(out/"validation.json",result);print(json.dumps(result))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--out",type=Path,required=True)
    main(parser.parse_args().out)
