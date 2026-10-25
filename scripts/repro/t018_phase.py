"""One bounded formal phase only. Online requires an independent parent probe first."""
import subprocess,sys,time
from t018_common import *
phase=sys.argv[1];inner=float(sys.argv[2]);start=time.monotonic()
assert phase in ('offline','online')
phase_admission(phase,inner)
if phase=='online':
    t=time.monotonic()
    subprocess.run([sys.executable,'scripts/repro/t018_online.py','--initial-only',str(WORK/'online-parent-check')],check=True,timeout=300)
    elapsed=time.monotonic()-t
    b=read(WORK/'budget.json');pre=sum(j['wall_seconds']+j.get('CPU_terminal_seal_seconds',0) for j in b['jobs'] if j['job'] in ('semantic-validation-v1','cost-preflight-v1'))
    assert pre+elapsed<=1800,'total tests/benchmark subbudget exhausted'
    atomic_json(WORK/'online-parent-check/validation-cost.json',dict(seconds=elapsed,prior_tests_seconds=pre,total_tests_seconds=pre+elapsed))
remaining=inner-(time.monotonic()-start)-10
assert remaining>0
subprocess.run([sys.executable,'scripts/repro/t018_'+phase+'.py','--wall-seconds',str(remaining)],check=True,timeout=remaining+15)
