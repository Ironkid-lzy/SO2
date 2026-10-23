"""CPU D02 terminal sealer; no queue launches or model/policy construction."""
import argparse, hashlib, json, shutil, time
from pathlib import Path
from t014_supervise import group_alive,write
ROOT=Path(__file__).resolve().parents[2];WORK=ROOT/'_so2_work/t014'
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main(job):
 while True:
  if (WORK/'budget.json').exists():
   ledger=json.loads((WORK/'budget.json').read_text());items=[j for j in ledger['jobs'] if j['job']==job]
   if items and items[0]['status']!='running' and not group_alive(items[0].get('pgid',-1)):break
  time.sleep(2)
 dst=ROOT/'_so2_work/sealed/t014'/job;dst.mkdir(parents=True,exist_ok=False)
 files=[]
 sources=list((ROOT/'_so2_work/diagnostics/t014-bias-v1').glob('*'))
 sources.extend(WORK/n for n in ('budget.json','d02-design.json',job+'.log'))
 for p in sorted(sources):
  if p.is_file():
   rec={'remote_path':str(p),'size':p.stat().st_size,'sha256':digest(p)}
   if p.suffix in ('.json','.npz','.pkl','.log') and p.stat().st_size<20000000:
    shutil.copy2(p,dst/p.name);rec['copied']=p.name
   files.append(rec)
 write(dst/'seal.json',{'job':items[0],'files':files,'sealed_unix':time.time()})
 write(WORK/'d02-sealed.json',{'job':job,'status':items[0]['status'],'path':str(dst)})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--job',required=True);main(p.parse_args().job)
