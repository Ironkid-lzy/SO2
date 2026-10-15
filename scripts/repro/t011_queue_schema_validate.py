"""Test actual queue finalization with frozen real-run fixtures; no training."""
import copy
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch
from t011_common import *
import t011_queue as queue

OUT=ROOT/"_so2_work/validation/t011-schema-repair-v1"

def exercise(label,changes=None):
    source=run_path(0)
    meta=json.loads((source/"run_meta.json").read_text())
    accepted=json.loads((source/"acceptance.json").read_text())
    parent=parent_manifest(0)
    controls=queue.control_events(0)
    if changes=="wrong_updates":accepted["final_state"]["updates"]["critic"]-=1
    elif changes=="not_accepted":accepted["accepted"]=False
    elif changes=="missing_final_state":del accepted["final_state"]
    sha=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    with tempfile.TemporaryDirectory(dir=OUT) as tmp:
        work=Path(tmp);run=work/"fixture";run.mkdir();(run/"ckpt").mkdir()
        meta["git_commit"]=sha
        (run/"run_meta.json").write_text(json.dumps(meta))
        (run/"acceptance.json").write_text(json.dumps(accepted))
        shutil.copy2(source/"metrics.jsonl",run/"metrics.jsonl")
        (run/"ckpt/envstep_7500.pth.tar").symlink_to(source/"ckpt/envstep_7500.pth.tar")
        state={"stages":{},"code_sha":sha}
        real_popen=subprocess.Popen
        real_sleep=time.sleep
        commands=[]
        def child(cmd,**kwargs):
            commands.append(cmd)
            # Replace only the subprocess invocation with an idle CPU child.
            return real_popen([sys.executable,"-c","import time;time.sleep(.2)"],**kwargs)
        def output(cmd,**kwargs):
            if cmd[0]=="nvidia-smi":return ""
            if cmd==["git","rev-parse","HEAD"]:return sha+"\n"
            if cmd==["git","status","--porcelain"]:return ""
            raise AssertionError(f"unexpected subprocess query {cmd}")
        error=None
        with patch.object(queue,"WORK",work),patch.object(queue,"STATE",work/"queue_state.json"), \
             patch.object(queue,"run_path",return_value=run), \
             patch.object(queue,"parent_manifest",return_value=parent), \
             patch.object(queue,"control_events",return_value=controls), \
             patch.object(queue.subprocess,"Popen",side_effect=child), \
             patch.object(queue.subprocess,"check_output",side_effect=output), \
             patch.object(queue.time,"sleep",side_effect=lambda seconds:real_sleep(.03)):
            try:queue.watch(state,0)
            except (AssertionError,KeyError) as exc:error=type(exc).__name__+":"+str(exc)
        assert len(commands)==1
        entry=state["stages"]["s0"]
        assert entry["exit_code"]==0
        assert (run/"first_window.json").exists()
        if changes is None:
            assert error is None and entry["status"]=="completed"
            assert entry["acceptance"]["final_state"]["updates"]=={"critic":950000,"actor":95000,"alpha":0}
        else:
            assert error is not None and entry["status"]=="partial"
        return {"case":label,"queue_stage_status":entry["status"],"child_exit_code":entry["exit_code"],
                "error":error,"child_is_idle_cpu_stub":True,"training_entry_invoked":False,
                "first_window_checked":True,"wall_seconds":entry["wall_seconds"]}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cases=[exercise("real_acceptance_success"),exercise("wrong_counts_rejected","wrong_updates"),
           exercise("unaccepted_run_rejected","not_accepted"),exercise("missing_state_rejected","missing_final_state")]
    original=json.loads((run_path(0)/"acceptance.json").read_text())
    try: original["updates"]
    except KeyError: reproduces=True
    else: raise AssertionError("old bug no longer reproduced by real frozen fixture")
    base="87f0ba67db5abbf71529614bdabba70dab950074"
    changed=subprocess.check_output(["git","diff",base,"HEAD","--name-only"],cwd=ROOT,text=True).splitlines()
    assert changed==["scripts/repro/t011_queue.py","scripts/repro/t011_queue_schema_validate.py"]
    diff=subprocess.check_output(["git","diff",base,"HEAD","--","scripts/repro/t011_queue.py"],cwd=ROOT,text=True)
    removed=[x[1:] for x in diff.splitlines() if x.startswith("-") and not x.startswith("---")]
    added=[x[1:] for x in diff.splitlines() if x.startswith("+") and not x.startswith("+++")]
    assert len(removed)==len(added)==1
    assert added[0]==removed[0].replace('a["updates"]','a["final_state"]["updates"]')
    result={"status":"passed","code_sha":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
            "base_sha":base,"old_bug_reproduced_with_real_seed0_schema":reproduces,"cases":cases,
            "only_production_change_is_schema_path":True,"training_math_model_config_unchanged":True,
            "fixed_batch_updates":0,"train_env_steps":0,"evaluation_env_steps":0,"training_started":False}
    atomic_json(OUT/"validation.json",result);print(json.dumps(result))

if __name__=="__main__":main()
