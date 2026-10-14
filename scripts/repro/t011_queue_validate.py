"""Exercise timeout/nonzero/duplicate/budget stop without training."""
import tempfile
import subprocess
from unittest.mock import patch
from t011_common import *
import t011_queue as q

def main():
    record={}
    with tempfile.TemporaryDirectory(dir=VALIDATION) as tmp:
        root=Path(tmp)
        for label,code,budget in (("nonzero","raise SystemExit(7)",15.),("timeout","import time;time.sleep(60)",10.2)):
            work=root/label;work.mkdir()
            stub=work/"stub";stub.write_text("#!/usr/bin/env python3\\n"+code+"\\n");stub.chmod(0o755)
            state={"stages":{},"code_sha":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()}
            with patch.object(q,"WORK",work),patch.object(q,"STATE",work/"state.json"),patch.object(q,"PYTHON",str(stub)),patch.object(q,"budget_for",return_value=budget):
                try:q.watch(state,0)
                except AssertionError:pass
                else:raise AssertionError("supervisor accepted failure")
            assert state["stages"]["s0"]["status"]=="partial" and state["stages"]["s0"]["exit_code"]!=0
            record[label]=state["stages"]["s0"]
        assert q.budget_for({"stages":{}})==18000
        assert q.budget_for({"stages":{"failed":{"wall_seconds":85000}}})==1400
        try:q.budget_for({"stages":{"failed":{"wall_seconds":86380}}})
        except AssertionError:record["budget_exhaustion_rejected"]=True
        else:raise AssertionError("budget not stopped")
        with patch.object(q,"WORK",root):
            try:q.main()
            except AssertionError as exc:assert "existing queue" in str(exc)
            else:raise AssertionError("duplicate not refused")
        record["duplicate_rejected"]=True
    atomic_json(VALIDATION/"queue-validation.json",{"status":"passed","train_env_steps":0,"fixed_batch_updates":0,**record})

if __name__=="__main__":main()
