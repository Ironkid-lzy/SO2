"""Zero-update multi-seed orchestration, parent, and real-hook validation."""
import argparse
import ast
import copy
import subprocess
import tempfile
from unittest.mock import patch

from t010_edac import *
import t010_loading as loading
import t010_run as runner
import t010_multi_queue as queue


def main(out):
    out=out.resolve();out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    sha=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    seed1=queue.run_path("offline",1)/"ckpt/update_3000000.pth"
    record={"seed":1,"parent_path":str(seed1),"parent_sha256":"68aa1dfed8500f6111b1b730dcc778a68d48e42e9e196bd6a41f97151b8c5b20","parent_code_sha":"c7e263cbf09a5b659d3d66e060274cc525c10cd2"}
    for seed,parent,parent_record in ((0,loading.PARENT,None),(1,seed1,record)):
        folder=out/f"hook-s{seed}";folder.mkdir();loading.validate(folder,parent,parent_record,seed)
    source=subprocess.check_output(["git","show","c7e263cbf09a5b659d3d66e060274cc525c10cd2:scripts/repro/t010_run.py"],cwd=ROOT,text=True)
    def offline_ast(text):return ast.dump(next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=="offline"))
    assert offline_ast(source)==offline_ast((ROOT/"scripts/repro/t010_run.py").read_text())
    for path in ("SO2","scripts/repro/t010_edac.py","scripts/repro/configs"):
        old=subprocess.check_output(["git","rev-parse",f"{loading.OFFLINE_SHA}:{path}"],cwd=ROOT)
        assert old==subprocess.check_output(["git","rev-parse",f"HEAD:{path}"],cwd=ROOT)
    hashes={}
    for seed in range(5):
        seed_all(seed,True);first=make_model("cuda");h=tensor_hash(first.state_dict())
        seed_all(seed,True);second=make_model("cuda");assert h==tensor_hash(second.state_dict());hashes[seed]=h
    assert len(set(hashes.values()))==5
    ident={"git_commit":sha,"recipe_sha256":digest(runner.RECIPE),"config_sha256":digest(runner.CONFIG),"dataset_sha256":DATA_SHA,"reference_sha":REFERENCE_SHA,"seed":1,"actor_bounds":[-20,2]}
    negative={}
    for label,change in (("wrong_seed",{"seed":2}),("wrong_hash",{"parent_sha256":"0"*64}),("wrong_code",{"parent_code_sha":"0"*40}),("wrong_path",{"parent_path":"/tmp/wrong"})):
        wrong={**record,**change}
        with patch.object(loading.torch,"load",side_effect=AssertionError("unsafe full load reached")) as load:
            try:loading.checked_parent(seed1,ident,wrong)
            except AssertionError as exc:assert str(exc)!="unsafe full load reached"
            else:raise AssertionError("bad parent accepted")
            load.assert_not_called();negative[label]=True
    contract=out/"online-contract";contract.mkdir()
    def pipeline(config,**kwargs):
        cfg,_=config
        assert kwargs["seed"]==1 and kwargs["max_env_steps"]==100000 and kwargs["max_wall_clock_sec"]==28770
        assert kwargs["eval_env_step_interval"]==2500 and kwargs["eval_n_episode"]==20
        assert cfg.policy.learn.learner.load_path==cfg.policy.learn.learner.hook.load_ckpt_before_run==str(contract/"online_init.pth")
        safe=torch.load(contract/"online_init.pth",weights_only=True,map_location="cpu")
        assert set(safe)=={"model","target_model"}
        raise RuntimeError("TEST_STOP_BEFORE_ANY_ENV_OR_UPDATE")
    with patch.object(runner,"identity",side_effect=lambda seed:{**ident,"seed":seed}),patch("ding.entry.serial_pipeline_offline2online",side_effect=pipeline) as entry:
        try:runner.online(contract,seed1,1,"EXP-028",record,28770)
        except RuntimeError as exc:assert str(exc)=="TEST_STOP_BEFORE_ANY_ENV_OR_UPDATE"
        else:raise AssertionError("pipeline not reached")
        entry.assert_called_once()
    meta=json.loads((contract/"run_meta.json").read_text());assert meta["seed"]==meta["pretrain_seed"]==meta["online_seed"]==1 and meta["exp_id"]=="EXP-028"
    assert meta["initialization"]["parent_sha256"]==record["parent_sha256"]
    state={"prior_offline_wall_seconds":21631.8783724308,"stages":{}}
    assert queue.budget_for(state,"offline")==172800-21631.8783724308 and queue.budget_for(state,"online")==28800
    state["stages"]["failed"]={"phase":"offline","wall_seconds":10};assert queue.used_seconds(state,"offline")==21641.8783724308
    state["stages"]["o"]={"phase":"online","wall_seconds":115180}
    try:queue.budget_for(state,"online")
    except AssertionError:pass
    else:raise AssertionError("budget exhaustion accepted")
    assert queue.PLAN==[("offline",2),("offline",3),("offline",4),("online",1),("online",2),("online",3),("online",4)]
    qout=out/"queue-tests";qout.mkdir()
    with patch.object(queue,"WORK",qout),patch.object(queue,"STATE",qout/"state.json"):
        for label,command,budget in (("nonzero",[sys.executable,"-c","raise SystemExit(7)"],15),("timeout",[sys.executable,"-c","import time;time.sleep(60)"],10.2)):
            qs={"stages":{}}
            try:queue.watch(qs,label,"validation",0,command,budget)
            except AssertionError:pass
            else:raise AssertionError("failure did not stop")
            assert qs["stages"][label]["status"]=="partial" and qs["stages"][label]["exit_code"]!=0
        try:queue.main()
        except AssertionError as exc:assert "queue exists" in str(exc)
        else:raise AssertionError("duplicate allowed")
    result={"status":"passed","code_sha":sha,"git_dirty":bool(subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip()),
            "updates":0,"preflight_updates_cumulative":181,"train_env_steps":0,"eval_env_steps":0,
            "offline_function_ast_unchanged":True,"math_config_git_objects_unchanged":True,"all_seed_initializations_distinct_reproducible":hashes,
            "real_hook_parent_seeds":[0,1],"wrong_parent_rejected_before_pickle":negative,"online_seed1_entry_contract":True,
            "cumulative_budget_and_failure_accounting":True,"queue_nonzero_timeout_duplicate_refusal":True,
            "future_parent_hook_validation_required":True,
            "source_sha256":{name:digest(ROOT/"scripts/repro"/name) for name in ("t010_run.py","t010_loading.py","t010_multi_queue.py","t010_multi_validate.py")}}
    atomic_json(out/"validation.json",result);print(json.dumps(result))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--out",type=Path,required=True)
    main(parser.parse_args().out)
