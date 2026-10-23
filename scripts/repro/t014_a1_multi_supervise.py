"""Finite17h extension in existing24h ledger; original global lock and failure stop."""
import argparse
import sys
from t014_a1_multi import *
from t014_supervise import supervise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--seconds',type=float,required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    protocol=read(PROTO);budget=protocol['budget'];cap=budget['stage_endpoint_seconds']
    assert cap==budget['start_T014_seconds']+budget['additional_seconds_cap'] and budget['additional_seconds_cap']==61200 and cap<=86400
    assert a.job in ['a1-ext-identity-v1']+[f'a1-ext-s{s}-v1' for s in range(1,5)]
    maximum=budget['identity_check_seconds_cap'] if a.job=='a1-ext-identity-v1' else budget['job_outer_seconds']
    assert 0<a.seconds<=maximum
    command=a.command[1:] if a.command[0]=='--' else a.command
    sys.exit(supervise(a.job,command,a.seconds,total=cap,acknowledged_cpu_failures=['d02-cpu-semantics-v1']))
