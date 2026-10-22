"""CPU-only contract tests using isolated locks/budgets; never touches real ledger."""
import json, subprocess, sys, tempfile, time
from pathlib import Path
from t013_supervise import supervise,write

def main():
    result={}
    with tempfile.TemporaryDirectory(prefix='so2-t013-guard-') as tmp:
        root=Path(tmp);lock=root/'lock'
        cmd=[sys.executable,'-c','import signal,time,pathlib,sys; signal.signal(signal.SIGTERM,lambda *_:(pathlib.Path(sys.argv[1]).write_text("partial"),sys.exit(2))); time.sleep(30)',str(root/'partial')]
        for label,seconds,deadline,total in [('deadline',20,time.time()+1,100),('budget',20,time.time()+100,31)]:
            work=root/label
            code=supervise(label,cmd,seconds,work,deadline,total,lock,False)
            state=json.loads((work/'budget.json').read_text())
            assert code==1 and state['jobs'][-1]['status']=='partial' and (root/'partial').exists()
            assert state['used_seconds']<3
            result[label+'_stops_and_preserves_partial']=True
        import fcntl
        with lock.open('a+') as f:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:supervise('locked',['true'],1,root/'locked',time.time()+10,100,lock,False)
            except BlockingIOError:result['global_exclusion']=True
            else:raise AssertionError('lock not enforced')
        # Root exits while a descendant ignores TERM; the lock must survive cleanup.
        childcode='import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)'
        rootcode='import subprocess,sys,time; p=subprocess.Popen([sys.executable,"-c",sys.argv[1]]); time.sleep(.3)'
        t=time.monotonic()
        assert supervise('descendant',[sys.executable,'-c',rootcode,childcode],10,root/'descendant',time.time()+20,100,lock,False)==0
        assert 4.9<time.monotonic()-t<9
        result['TERM_ignoring_descendant_killed_before_lock_release']=True
        work=root/'success'; assert supervise('one',['true'],2,work,time.time()+10,100,lock,False)==0
        try:supervise('one',['true'],2,work,time.time()+10,100,lock,False)
        except AssertionError:result['duplicate_rejected']=True
        else:raise AssertionError('duplicate accepted')
        try:supervise('retry',['true'],2,root/'budget',time.time()+10,100,lock,False)
        except AssertionError:result['failure_blocks_followups']=True
        else:raise AssertionError('failure continuation accepted')
    write(Path('_so2_work/t013/supervisor-validation.json'),result)
    print(json.dumps(result))
if __name__=='__main__':main()
