"""One authorized online-only attempt; retain the original queue and failure."""
import argparse
import fcntl
import signal
import subprocess
import time
import traceback

import t010_queue as queue
from t010_edac import ROOT, atomic_json, digest, json, os
from t010_loading import PARENT, PREVIOUS_QUEUE, remaining_online_seconds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thread-id", required=True)
    args = parser.parse_args()
    assert args.thread_id == "01a12b0e-1d51-7730-9b01-98ffac956424"
    assert not subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip()
    sha = subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    validation = json.loads((ROOT/"_so2_work/validation/t010-loading-repair/validation.json").read_text())
    assert validation["status"] == "passed" and validation["identity"]["git_commit"] == sha
    assert validation["fixed_batch_updates"] == 0 and validation["cumulative_preflight_updates"] == 181
    remote = subprocess.check_output(["git","ls-remote","origin","refs/heads/codex/so2-t010-pretrain"],cwd=ROOT,text=True).split()[0]
    assert remote == sha, "online code must be pushed before launch"
    previous = json.loads(PREVIOUS_QUEUE.read_text())
    queue.WORK = ROOT/"_so2_work/t010-online-attempt2"
    queue.STATE = queue.WORK/"queue_state.json"
    queue.RUNS = {"offline":PARENT.parents[1], "online":ROOT/"_so2_work/runs/EXP-026/EXP-026-s0-attempt2"}
    budget = remaining_online_seconds()
    with queue.GLOBAL_LOCK.open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True).strip()
        assert not queue.WORK.exists() and not queue.RUNS["online"].exists()
        queue.WORK.mkdir()
        lock.seek(0);lock.truncate();lock.write(str(os.getpid()));lock.flush()
        state = {"status":"running", "phase":"online", "thread_id":args.thread_id,
                 "queue_pid":os.getpid(), "started_unix":time.time(), "code_sha":sha,
                 "plan_sha":previous["plan_sha"], "parent_code_sha":previous["code_sha"],
                 "recipe_sha256":digest(ROOT/"scripts/repro/configs/t010_recipe.json"),
                 "runs":{k:str(v) for k,v in queue.RUNS.items()}, "queue_workdir":str(queue.WORK),
                 "stages":{"offline":previous["stages"]["offline"]}, "auto_retry":False,
                 "previous_attempt":{"state":str(PREVIOUS_QUEUE), "online":previous["stages"]["online"]},
                 "budgets_sec":{"online":budget}, "cumulative_preflight_updates":181,
                 "loading_validation":validation, "handoff":previous["handoff"]}
        atomic_json(queue.STATE,state)
        def interrupt(signum, frame):
            raise RuntimeError(f"queue signal {signum}; no retry")
        signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
        try:
            queue.deliver(state)
            cmd = [queue.PYTHON,str(ROOT/"scripts/repro/t010_run.py"),"--phase","online",
                   "--run-dir",str(queue.RUNS["online"]),"--parent",str(PARENT)]
            queue.spawn_and_watch(state,"online",cmd,budget)
            acceptance = json.loads((queue.RUNS["online"]/"acceptance.json").read_text())
            assert acceptance["accepted"]
            state["stages"]["online"]["acceptance"] = acceptance
            queue.archive_checkpoints("online")
            state.update(status="completed",phase="completed",finished_unix=time.time())
            queue.deliver(state)
        except BaseException as exc:
            state.update(status="partial",error=repr(exc),traceback=traceback.format_exc(),finished_unix=time.time())
            atomic_json(queue.STATE,state)
            if queue.RUNS["online"].exists():queue.archive_checkpoints("online")
            try:queue.deliver(state)
            except BaseException as err:
                state["delivery_error"]=repr(err);atomic_json(queue.STATE,state)
            raise


if __name__ == "__main__":main()
