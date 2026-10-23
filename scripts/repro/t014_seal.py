"""CPU terminal evidence sealer; never starts/restarts compute or writes research Git."""
import argparse, hashlib, json, os, shutil, time
from pathlib import Path
from t014_supervise import group_alive, write
ROOT=Path(__file__).resolve().parents[2];WORK=ROOT/'_so2_work/t014'
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def main(job):
    while True:
        ledger=WORK/'budget.json'
        if ledger.exists():
            state=json.loads(ledger.read_text());items=[x for x in state['jobs'] if x['job']==job]
            if items and items[0]['status']!='running' and not group_alive(items[0].get('pgid',-1)):break
        time.sleep(2)
    dst=ROOT/'_so2_work/sealed/t014'/job;dst.mkdir(parents=True,exist_ok=False)
    files=[];source=ROOT/'_so2_work/diagnostics/t014-action-mask-v1'
    for p in sorted(source.glob('*')):
        if p.is_file():
            record={'remote_path':str(p),'size':p.stat().st_size,'sha256':digest(p)}
            if p.suffix=='.json':shutil.copy2(p,dst/p.name);record['copied']=p.name
            files.append(record)
    for name in ['budget.json','diagnostic-design.json','cpu-validation.json','supervisor-validation.json',job+'.log']:
        p=WORK/name
        if p.exists():
            shutil.copy2(p,dst/p.name);files.append({'remote_path':str(p),'copied':p.name,'size':p.stat().st_size,'sha256':digest(p)})
    write(dst/'seal.json',{'job':items[0],'files':files,'large_gradients':'remote only, not independently backed up','sealed_unix':time.time()})
    write(WORK/'diagnostic-sealed.json',{'path':str(dst),'job':job,'status':items[0]['status']})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);main(p.parse_args().job)
