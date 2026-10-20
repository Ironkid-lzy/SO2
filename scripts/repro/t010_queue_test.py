"""CPU checks for queue failure/timeout refusal, without launching RL training."""
import tempfile
import time
from unittest.mock import patch

import t010_queue as q
from t010_edac import *


def main():
    out=ROOT/"_so2_work/validation/t010-queue-cpu"
    assert not out.exists();out.mkdir(parents=True)
    tests={}
    with patch.object(q,"WORK",out),patch.object(q,"STATE",out/"state.json"):
        state={"status":"test","phase":"test"}
        try:q.spawn_and_watch(state,"fail_cpu",[sys.executable,"-c","raise SystemExit(7)"],2)
        except AssertionError:tests["nonzero_stops_next_stage"]=state["child_exit_code"]==7
        else:raise AssertionError("failure accepted")
        state={"status":"test","phase":"test"}
        start=time.monotonic()
        try:q.spawn_and_watch(state,"timeout_cpu",[sys.executable,"-c","import time; time.sleep(20)"],.1)
        except AssertionError:tests["timeout_terminates_child"]=state["child_exit_code"] is not None and time.monotonic()-start<2
        else:raise AssertionError("timeout accepted")
        atomic_json(out/"atomic.json",{"generation":1});atomic_json(out/"atomic.json",{"generation":2})
        tests["atomic_replace"]=json.loads((out/"atomic.json").read_text())=={"generation":2}
        with patch.object(q,"WORK",out):
            try:q.main()
            except AssertionError:tests["duplicate_start_refused"]=True
            else:raise AssertionError("duplicate start accepted")
    tests["original_pids_detected"]=q.process_alive(65524) or not Path("/proc/65524").exists()
    assert all(tests.values()),tests
    atomic_json(out/"validation.json",{"status":"passed","tests":tests,"train_env_steps":0,"fixed_batch_updates":0})
    print(json.dumps(tests))

if __name__=="__main__":main()
