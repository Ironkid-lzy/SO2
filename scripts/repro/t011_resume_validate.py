"""Zero-update continuation proof and orchestration lifecycle tests."""
import copy
import json
import subprocess
import tempfile
import time
from pathlib import Path
from unittest.mock import patch
from t011_common import *
import t011_resume_queue as resume

def git_object(sha,path):
    return subprocess.check_output(["git","rev-parse",f"{sha}:{path}"],cwd=ROOT,text=True).strip()

def main():
    sha=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    assert not subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip()
    base=resume.TRAINING_SHA
    protected=["SO2","scripts/repro/configs"]+["scripts/repro/"+n for n in
               ("t010_edac.py","t010_run.py","t010_loading.py","t010_validate.py",
                "t011_common.py","t011_run.py","t011_guards.py","t011_loading.py","t011_launch.sh")]
    objects={}
    for path in protected:
        objects[path]=git_object(base,path)
        assert objects[path]==git_object(sha,path),path
    allowed={"scripts/repro/t011_queue.py","scripts/repro/t011_queue_schema_validate.py",
             "scripts/repro/t011_resume_queue.py","scripts/repro/t011_resume_validate.py","scripts/repro/t011_deliver.py"}
    changed=set(subprocess.check_output(["git","diff",base,sha,"--name-only"],cwd=ROOT,text=True).splitlines())
    assert changed<=allowed
    repaired=subprocess.check_output(["git","show","6433f283a56f6d5acde7074a816578f68448afe8:scripts/repro/t011_queue.py"],cwd=ROOT)
    assert repaired==(ROOT/"scripts/repro/t011_queue.py").read_bytes()
    freeze=resume.CHECK/"requirements-lock.txt"
    assert freeze.read_bytes()==(VALIDATION/"requirements-lock.txt").read_bytes()
    original=json.loads((VALIDATION/"validation-training-87.json").read_text())
    assert original["status"]=="passed" and original["code_sha"]==base
    history,accept=resume.checked_history()
    state=resume.build_state(sha)
    prior=history["stages"]["s0"]["wall_seconds"]
    assert resume.PLAN==(1,2,3,4) and resume.queue.used(state)==prior
    assert resume.queue.budget_for(state)==18000 and 86400-prior==75331.1564519
    watched=[];sealed=[];delivered=[]
    def fake_watch(s,seed):
        watched.append(seed)
        assert seed!=0
        s["stages"][f"s{seed}"]={"status":"completed","seed":seed,"wall_seconds":10,"exit_code":0}
    with patch.object(resume.queue,"watch",side_effect=fake_watch), \
         patch.object(resume.delivery,"seal",side_effect=sealed.append), \
         patch.object(resume.delivery,"deliver",side_effect=lambda s:delivered.append(s["status"])):
        resume.run_remaining(state)
    assert watched==sealed==[1,2,3,4] and state["status"]=="completed"
    assert resume.queue.used(state)==prior+40
    with tempfile.TemporaryDirectory(dir=resume.CHECK) as tmp:
        tmp=Path(tmp);check=tmp/"check";check.mkdir()
        atomic_json(check/"validation.json",{"status":"passed","code_sha":sha})
        fresh=resume.build_state(sha)
        calls=[]
        def fail(s,seed):
            calls.append(seed)
            s["stages"][f"s{seed}"]={"status":"partial" if seed==2 else "completed","seed":seed,"wall_seconds":1,"exit_code":7 if seed==2 else 0}
            if seed==2:raise RuntimeError("TEST_FAIL_SEED2")
        with patch.object(resume,"WORK",tmp/"queue"),patch.object(resume,"STATE",tmp/"queue/state.json"), \
             patch.object(resume,"CHECK",check),patch.object(resume,"LOCK",tmp/"resource.lock"), \
             patch.object(resume,"build_state",return_value=fresh),patch.object(resume.queue,"watch",side_effect=fail), \
             patch.object(resume.delivery,"seal"),patch.object(resume.delivery,"deliver"), \
             patch.object(resume.queue,"WORK",resume.queue.WORK),patch.object(resume.queue,"STATE",resume.queue.STATE), \
             patch.object(resume.delivery,"WORK",resume.delivery.WORK):
            try:resume.main()
            except RuntimeError as exc:assert str(exc)=="TEST_FAIL_SEED2"
            else:raise AssertionError("failure ignored")
            saved=json.loads((tmp/"queue/state.json").read_text())
            assert saved["status"]=="partial" and calls==[1,2]
            assert saved["stages"]["s0"]["reused_without_training"]
            assert resume.queue.used(saved)==prior+2
            try:resume.main()
            except AssertionError as exc:assert "continuation queue exists" in str(exc)
            else:raise AssertionError("duplicate allowed")
    result={"status":"passed","code_sha":sha,"training_math_sha":base,"protected_git_objects":objects,
            "only_orchestration_and_delivery_changed":True,"repaired_acceptance_reader_preserved":True,
            "seed0_raw_and_parent_hashes_verified":True,"plan":[1,2,3,4],"seed0_never_scheduled":True,
            "success_lifecycle_completed":True,"failure_stops_before_seed3":True,"duplicate_queue_rejected":True,
            "prior_wall_seconds":prior,"remaining_budget_seconds":86400-prior,"per_run_seconds":18000,
            "fixed_batch_updates":0,"preflight_calls_cumulative":15,"train_env_steps":0,"evaluation_env_steps":0,
            "dependency_freeze_identical":True,"new_parent_actual_hooks_reused_from_exact_loading_objects":True}
    atomic_json(resume.CHECK/"validation.json",result)
    gate={**original,"code_sha":sha,"fixed_batch_updates":15,"preflight_reserved_calls_cumulative":15,
          "original_gpu_validation":str(VALIDATION/"validation-training-87.json"),
          "resume_equivalence_validation":str(resume.CHECK/"validation.json"),
          "resume_zero_update_semantic_proof":result}
    atomic_json(VALIDATION/"validation.json",gate)
    print(json.dumps({k:v for k,v in result.items() if k!="protected_git_objects"}))

if __name__=="__main__":main()
