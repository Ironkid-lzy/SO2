"""One bounded T019 phase, preserving the T018 phase loop and parent probe."""
import subprocess,sys,time
from t019_common import *
phase=sys.argv[1];inner=float(sys.argv[2]);start=time.monotonic()
assert phase in ('offline','online');phase_admission(phase,inner)
if phase=='online':
    t=time.monotonic()
    subprocess.run([sys.executable,'scripts/repro/t019_online.py','--initial-only',str(WORK/'online-parent-check')],check=True,timeout=300)
    elapsed=time.monotonic()-t
    b=read(BASEWORK/'budget.json')
    prior=sum(j['wall_seconds']+j.get('CPU_terminal_seal_seconds',0) for j in b['jobs'] if j['job']=='identity-preflight-v1')
    prior+=sum(read(p)['seconds'] for p in BASEWORK.glob('seed*/online-parent-check/validation-cost.json'))
    assert prior+elapsed<=3600,'T019 precheck/probe subgroup exhausted'
    atomic_json(WORK/'online-parent-check/validation-cost.json',dict(seconds=elapsed,prior_tests_seconds=prior,total_tests_seconds=prior+elapsed,already_in_formal_job_wall=True))
remaining=inner-(time.monotonic()-start)-10
assert remaining>0
subprocess.run([sys.executable,'scripts/repro/t019_'+phase+'.py','--wall-seconds',str(remaining)],check=True,timeout=remaining+15)
