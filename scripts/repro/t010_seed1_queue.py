"""Single seed1 offline job with fixed October 12 deadline; no next stage."""
import fcntl
import os
import shutil
import signal
import subprocess
import time
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo

from t010_edac import ROOT, Path, json, atomic_json, digest
from t010_deliver import curves

PYTHON = "/home/lzy/Projects/SO2/.venv/bin/python"
RESEARCH = Path("/home/lzy/Projects/rl_sample_efficiency_research_t010")
WORK = ROOT / "_so2_work/t010-seed1-20261012"
RUN = ROOT / "_so2_work/runs/EXP-027/EXP-027-s1-attempt1"
LOCK = Path("/home/lzy/Projects/SO2/_so2_work/training_resource.lock")
SOFT = datetime(2026,10,12,8,40,tzinfo=ZoneInfo("Asia/Shanghai")).timestamp()
HARD = datetime(2026,10,12,8,45,tzinfo=ZoneInfo("Asia/Shanghai")).timestamp()
EXIT = datetime(2026,10,12,8,50,tzinfo=ZoneInfo("Asia/Shanghai")).timestamp()
THREAD = "01a12b0e-1d51-7730-9b01-98ffac956424"


def timed_command(command, deadline):
    remaining = deadline-time.time()
    assert remaining>0, "deadline already passed"
    return ["/usr/bin/timeout","--signal=TERM","--kill-after=5s",f"{remaining:.3f}s",*command]


def deliver(state):
    evidence=RESEARCH/"reports/so2/SO2-T010-seed1";evidence.mkdir(parents=True,exist_ok=True)
    atomic_json(evidence/"queue_state.json",state)
    for name in ("validation.json","resources.json"):
        shutil.copy2(ROOT/"_so2_work/validation/t010-seed1"/name,evidence/name)
    summary=curves(RUN,"offline","EXP-027/EXP-027-s1-attempt1",seed=1) if RUN.exists() else {}
    names=("run_meta.json","state.json","metrics.jsonl","effective_recipe.json","stage_continuity.json","acceptance.json","checkpoint_manifest.json")
    processed=RESEARCH/"results/processed/EXP-027/EXP-027-s1-attempt1";processed.mkdir(parents=True,exist_ok=True)
    for name in names:
        if (RUN/name).exists():shutil.copy2(RUN/name,processed/name)
    if state["status"] in ("completed","partial") and RUN.exists():
        raw=RESEARCH/"results/raw/EXP-027/EXP-027-s1-attempt1"
        if not raw.exists():
            raw.mkdir(parents=True)
            for name in names:
                if (RUN/name).exists():shutil.copy2(RUN/name,raw/name)
            files=[{"path":str(p.relative_to(RUN)),"size":p.stat().st_size,"sha256":digest(p)}
                   for p in sorted(RUN.rglob("*")) if p.is_file()]
            atomic_json(raw/"artifact_manifest.json",{"linux_path":str(RUN),"files":files,"large_checkpoint_backup":"Linux only"})
    stamp=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    report=RESEARCH/"reports/so2/SO2-T010-seed1.md"
    report.write_text(f"""# SO2-T010 多种子扩展：seed1 离线执行

状态：`{state['status']}`，更新于 {stamp}。今晚仅 EXP-027 seed1 attempt1，3M EDAC updates；没有online或下一seed调用。seed0 EXP-025/026已完成，不重跑。

用户2026-10-12重新授权连接后继续，但9:00前必须释放服务器。基于seed0离线实际约6h，本次只安排seed1离线。内部8:40在完整更新边界停止并保存full partial；独立GNU timeout于8:45终止训练进程组（5秒后KILL兜底）；外层8:50终止整个执行/交付进程。截止优先，未达3M只标partial，不以较早checkpoint冒充final、不自动重试。中断恢复尚未获验证，不保证partial可以保持全部评价环境轨迹后续接跑。

执行代码SHA `{state['code_sha']}`；训练数学实现固定f5dacc442f73e94e085988600fa2ff8cb8ad95b0，SO2 tree和t010_edac.py Git对象完全相同。recipe/config源文件不变；effective_recipe记录seed=1和offline_exp=EXP-027，online_exp置空（本次不运行在线），属于任务单授权的唯一随机性变化。额外identity信息和截止门禁不改变正常更新。测试累计181/200，本次0更新。seed0初始模型逐张量复现、seed1确定性与差异性、实际policy hook连续性和timeout终止测试见validation.json。

配方：N2、3×256、LN、Dropout=.005、EDAC eta=1、lr3e-4、actor log_std[-20,2]、batch256；每100k更新评价20episodes，评价交互单列，不回流训练。不调参、不挑parent。所有成功/失败均保留。后续seed1在线仍需新的可用窗口和耗时告知。

实际command：`{state.get('command')}`。queue PID `{state.get('queue_pid')}`、child PID `{state.get('child_pid')}`；全局锁 `{LOCK}`。原始状态和日志 `{WORK}`，运行数据 `{RUN}`。exit code `{state.get('child_exit_code')}`；wall秒 `{state.get('wall_seconds')}`；进度 `{state.get('progress')}`；异常 `{state.get('error')}`。资源与依赖沿用既有SO2/.venv；资源快照见同名目录。完整数据/版本/执行环境记于run_meta；大checkpoint仅Linux一份。

观测：`{summary}`。完成不等于研究验收，当前扩展仅一个seed，不报告五种子统计或因果结论。
""")
    tracker=RESEARCH/"docs/EXPERIMENT_TRACKER.md";content=tracker.read_text()
    row=f"| EXP-027 | [T010扩展](tasks/SO2-T010-multiseed-extension.md) | {state['status']} | 独立离线seed1；seeds2–4待运行；固定3M final | 今日8:40停止/8:45强制终止 |"
    lines=content.splitlines();found=False
    for i,line in enumerate(lines):
        if line.startswith("| EXP-027 |"):lines[i]=row;found=True
    if not found:lines += ["",row]
    tracker.write_text("\n".join(lines)+"\n")
    paths=["reports/so2/SO2-T010-seed1.md","reports/so2/SO2-T010-seed1","docs/EXPERIMENT_TRACKER.md","results/processed/EXP-027"]
    for path in ("results/raw/EXP-027","results/figures/EXP-027"):
        if (RESEARCH/path).exists():paths.append(path)
    subprocess.run(["git","add",*paths],cwd=RESEARCH,check=True,timeout=30)
    if subprocess.run(["git","diff","--cached","--quiet"],cwd=RESEARCH).returncode:
        subprocess.run(["git","commit","-m",f"SO2-T010: seed1 offline {state['status']} before hard deadline"],cwd=RESEARCH,check=True,timeout=30)
    subprocess.run(["git","push","origin","codex/so2-t010-delivery"],cwd=RESEARCH,check=True,timeout=60)
    state["research_sha"]=subprocess.check_output(["git","rev-parse","HEAD"],cwd=RESEARCH,text=True).strip()
    atomic_json(WORK/"queue_state.json",state)


def main():
    assert SOFT-time.time()>6.4*3600, "insufficient forecast window; do not launch"
    assert not WORK.exists() and not RUN.exists(), "no duplicate or overwrite"
    assert not subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip()
    sha=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    validation=json.loads((ROOT/"_so2_work/validation/t010-seed1/validation.json").read_text())
    assert validation["status"]=="passed" and validation["code_sha"]==sha and not validation["git_dirty"]
    assert validation["updates"]==0
    with LOCK.open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True).strip()
        assert shutil.disk_usage(ROOT).free>20*1024**3
        WORK.mkdir();lock.seek(0);lock.truncate();lock.write(str(os.getpid()));lock.flush()
        command=timed_command([PYTHON,str(ROOT/"scripts/repro/t010_run.py"),"--phase","offline","--seed","1",
                    "--exp-id","EXP-027","--run-dir",str(RUN),"--stop-at",str(SOFT)],HARD)
        state={"status":"running","seed":1,"exp_id":"EXP-027","code_sha":sha,"thread_id":THREAD,
               "queue_pid":os.getpid(),"started_unix":time.time(),"command":command,
               "soft_stop_unix":SOFT,"hard_stop_unix":HARD,"supervisor_exit_unix":EXIT,
               "followup_training":False,"preflight_updates_cumulative":181}
        atomic_json(WORK/"queue_state.json",state)
        child=None
        def interrupt(signum, frame):raise RuntimeError(f"queue signal {signum}")
        signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
        try:
            with (WORK/"training.log").open("x") as log:
                child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                state["child_pid"]=child.pid;atomic_json(WORK/"queue_state.json",state)
                while child.poll() is None:
                    if (RUN/"state.json").exists():state["progress"]=json.loads((RUN/"state.json").read_text())
                    state["wall_seconds"]=time.time()-state["started_unix"]
                    atomic_json(WORK/"queue_state.json",state)
                    assert shutil.disk_usage(ROOT).free>20*1024**3
                    time.sleep(10)
            state["child_exit_code"]=child.returncode
            assert child.returncode==0, "offline stopped; no automatic retry"
            acceptance=json.loads((RUN/"acceptance.json").read_text());assert acceptance["accepted"]
            state.update(status="completed",acceptance=acceptance)
        except BaseException as exc:
            if child is not None and child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            state.update(status="partial",error=repr(exc),traceback=traceback.format_exc(),
                         child_exit_code=None if child is None else child.returncode)
        state.update(finished_unix=time.time(),wall_seconds=time.time()-state["started_unix"])
        if (RUN/"state.json").exists():state["progress"]=json.loads((RUN/"state.json").read_text())
        atomic_json(WORK/"queue_state.json",state)
        try:deliver(state)
        except BaseException as exc:
            state["delivery_error"]=repr(exc);atomic_json(WORK/"queue_state.json",state)
            raise


if __name__=="__main__":main()
