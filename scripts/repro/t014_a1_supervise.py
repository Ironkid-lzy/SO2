"""Nested finite A1 cap in the original T014 global ledger; never resets usage."""
import argparse
import json
import sys
from pathlib import Path
from t014_supervise import supervise

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--job',required=True)
    parser.add_argument('--seconds',required=True,type=float)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    protocol=json.loads((root/'_so2_work/t014/a1/protocol.json').read_text())
    assert args.job.startswith('a1-')
    cap=protocol['stage_budget']['absolute_T014_job_ledger_limit_for_A1']
    assert cap<=86400 and cap==protocol['stage_budget']['start_T014_seconds']+18000
    command=args.command[1:] if args.command[0]=='--' else args.command
    sys.exit(supervise(args.job,command,args.seconds,total=cap,
                       acknowledged_cpu_failures=['d02-cpu-semantics-v1']))
