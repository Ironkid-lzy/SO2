"""Zero-training supervisor contracts, including all four stage schemas."""
import copy,fcntl,tempfile,time
from unittest.mock import patch
from t012_common import *
import t012_queue as q

def main():
    result={};sha=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    for group in ("dropout","ln"):
      for phase in ("offline","online"):
        state={"stages":{}}
        assert q.budget_for(state,group,phase)==(36000 if phase=="offline" else 18000)
        limit=172800 if phase=="offline" else 86400
        state["stages"]["failed"]={"group":group,"phase":phase,"wall_seconds":limit-1000,"status":"partial"}
        assert q.budget_for(state,group,phase)==1000
        state["stages"]["failed"]["wall_seconds"]=limit-20
        try:q.budget_for(state,group,phase)
        except AssertionError:pass
        else:raise AssertionError("budget exhaustion accepted")
        a={"accepted":True,"errors":[],"updates":{"critic":3000000,"actor":3000000,"alpha":3000000},"train_env_steps":0,"evaluation_count":31} if phase=="offline" else {"accepted":True,"errors":[],"final_state":{"updates":{"critic":950000,"actor":95000,"alpha":0},"env_steps":100000},"evaluation_count":41}
        q.accept_stage(a,phase)
        for label in ("rejected","missing","wrong_count"):
            bad=copy.deepcopy(a)
            if label=="rejected":bad["accepted"]=False
            elif label=="missing":bad.pop("updates" if phase=="offline" else "final_state")
            elif phase=="offline":bad["updates"]["critic"]-=1
            else:bad["final_state"]["updates"]["critic"]-=1
            try:q.accept_stage(bad,phase)
            except (AssertionError,KeyError):pass
            else:raise AssertionError(f"accepted {label}")
        result[f"{group}-{phase}"]={"schema_success":True,"rejection_missing_wrong_count":True,"failed_budget_continuation_exact":True}
    with tempfile.TemporaryDirectory(dir=VALIDATION) as tmp:
        root=Path(tmp)
        for phase in ("offline","online"):
          for label in ("success","nonzero","timeout","nonfinite"):
            work=root/f"{phase}-{label}";work.mkdir();run=work/"run";stub=work/"stub"
            body=f"import pathlib,json,time,sys\nrun=pathlib.Path({str(run)!r})\n"
            if label=="nonzero":body+="raise SystemExit(7)\n"
            elif label=="timeout":body+="time.sleep(60)\n"
            else:
                key="gradient_updates" if phase=="offline" else "env_steps";step=100000 if phase=="offline" else 7500
                ev=[{"event":"evaluation",key:x,"returns":[1.]*20,"normalized_score_100":1.} for x in (range(0,100001,100000) if phase=="offline" else range(0,7501,2500))]
                if label=="nonfinite":ev[0]["returns"][0]=float("nan")
                a={"accepted":True,"errors":[],"updates":{"critic":3000000,"actor":3000000,"alpha":3000000},"train_env_steps":0,"evaluation_count":31} if phase=="offline" else {"accepted":True,"errors":[],"final_state":{"updates":{"critic":950000,"actor":95000,"alpha":0},"env_steps":100000},"evaluation_count":41}
                if phase=="online":ev.append({"event":"evaluation_state","env_steps":7500,"updates":{"critic":25000,"actor":2500,"alpha":0},"evaluation_interactions_total":96000})
                cp="update_100000.pth" if phase=="offline" else "envstep_7500.pth.tar"
                body+=f"run.mkdir();(run/'ckpt').mkdir();(run/'ckpt'/{cp!r}).write_text('stub');(run/'metrics.jsonl').write_text({''.join(json.dumps(e)+chr(10) for e in ev)!r});(run/'acceptance.json').write_text({json.dumps(a)!r});time.sleep(.5)\n"
            stub.write_text("#!/usr/bin/python3\n"+body);stub.chmod(0o755)
            state={"stages":{},"code_sha":sha};original_sleep=time.sleep
            with patch.object(q,"WORK",work),patch.object(q,"STATE",work/"state.json"),patch.object(q,"PYTHON",str(stub)),patch.object(q,"run_path",return_value=run),patch.object(q,"budget_for",return_value=10.2 if label=="timeout" else 15.),patch.object(q.time,"sleep",side_effect=lambda _:original_sleep(.1)):
                try:q.watch(state,"dropout",0,phase)
                except AssertionError:
                    assert label!="success"
                else:assert label=="success"
            e=state["stages"][f"dropout-s0-{phase}"]
            assert e["status"]==("completed" if label=="success" else "partial")
            if label in ("nonzero","timeout"):assert e["exit_code"]==(7 if label=="nonzero" else 124)
            result[f"watch-{phase}-{label}"]={k:e[k] for k in ("status","exit_code","wall_seconds")}
        with patch.object(q,"WORK",root):
            try:q.main()
            except AssertionError as exc:assert "existing queue" in str(exc)
            else:raise AssertionError("duplicate queue accepted")
        lockpath=root/"lock"
        with lockpath.open("a+") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            child=subprocess.run(["/usr/bin/flock","-n",str(lockpath),"true"]);assert child.returncode==1
        assert subprocess.run(["/usr/bin/flock","-n",str(lockpath),"true"]).returncode==0
        result["exclusive_lock_and_release"]=True;result["duplicate_queue_rejected"]=True
    atomic_json(VALIDATION/"queue-validation.json",{"status":"passed","fixed_batch_updates":0,"train_env_steps":0,**result})
if __name__=="__main__":main()
