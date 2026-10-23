"""T014-D02 fixed-reference calibration with predeclared reference-futility prefix."""
import json, math, pickle, shutil, signal, statistics, subprocess, time
from pathlib import Path
from t014_bias_env import *
from t014_reference_math import paired_direction_summary
ROOT=Path(__file__).resolve().parents[2];WORK=ROOT/'_so2_work/t014'
OUT=ROOT/'_so2_work/diagnostics/t014-bias-v1'
TIMES=(16,128,256,384,640)


def interval_summary(x):
    s=paired_direction_summary(x[0],x[1],x[2]);half=3.182446305284263*s['mean_standard_error']
    s['descriptive_t95']=[s['mean']-half,s['mean']+half]
    s['sign']=1 if s['descriptive_t95'][0]>0 else (-1 if s['descriptive_t95'][1]<0 else 0)
    return s


def case(env,snap,model,ident,ci,step,direction_id):
    cellid=f'{ident["label"]}-s{step}-u{direction_id}'
    rng=np.random.default_rng(14003000+ci*10000+step*2+direction_id)
    obs=snap['obs']; _,_,z=transition(model,obs,rng.normal(size=6))
    u=rng.choice([-1.,1.],6)/math.sqrt(6)
    masks=project_masks(model,obs,z,u,14004000+ci*10000+step*2+direction_id)
    mean=statistics.mean(masks);sd=statistics.stdev(masks)
    cell={'id':cellid,'checkpoint':ident,'state_step':step,'direction_id':direction_id,
          'base_z':z.tolist(),'direction':u.tolist(),'mask_projections':masks,
          'mask_mean':mean,'mask_sd':sd,'mask_mean_MC_SE':sd/8,
          'mask_cv':sd/(abs(mean)+1e-8),'mask_sign_agreement':statistics.mean((x*mean)>0 for x in masks),'references':{}}
    arrays={};deltas=(.05,.025) if step in (16,640) else (.05,)
    # Same epsilon/PVU array seeds also across delta, not merely across signs.
    for delta in deltas:
        returns={h:[[],[]] for h in (500,1000)}
        rs=[];ls=[]
        for rep in range(4):
            gen=np.random.default_rng(14005000+ci*100000+step*10+direction_id*4+rep)
            eps=gen.normal(size=(1000,6)).astype(np.float32)
            nu=(gen.normal(size=(1000,6))*.3).astype(np.float32)
            sides_r=[];sides_l=[]
            for sign_id,sign in enumerate((1,-1)):
                first=torch.tanh(torch.tensor(z)+sign*delta*torch.tensor(u,dtype=torch.float32)).numpy()
                rewards,logps=rollout(env,snap,model,first,eps,nu,ident['alpha'],ident['PVU'])
                sides_r.append(rewards);sides_l.append(logps)
                for h in (500,1000):returns[h][sign_id].append(soft_q_return(rewards[:h],logps[:h],.99,ident['alpha']))
            rs.append(sides_r);ls.append(sides_l)
        key=str(delta);cell['references'][key]={str(h):interval_summary((vals[0],vals[1],delta)) for h,vals in returns.items()}
        arrays['rewards_'+key]=np.asarray(rs);arrays['logps_'+key]=np.asarray(ls)
    np.savez_compressed(OUT/(cellid+'.npz'),**arrays)
    all_signs=[h['sign'] for ds in cell['references'].values() for h in ds.values()]
    cell['reference_direction_stable']=all(s!=0 and s==all_signs[0] for s in all_signs)
    cell['delta_audited']=len(deltas)==2
    cell['reference_sign']=all_signs[0] if cell['reference_direction_stable'] else 0
    cell['low_mask_spread']=cell['mask_cv']<=.5 and cell['mask_sign_agreement']>=.9
    cell['audited_opposite_example']=bool(cell['delta_audited'] and cell['reference_direction_stable'] and cell['low_mask_spread'] and mean*cell['reference_sign']<0)
    cell['interpretation']='Finite-H target-surrogate direction only; does not identify causal bias source or an algorithm repair.'
    atomic_json(OUT/(cellid+'.json'),cell);env.flush()
    return cell


def collect_states(env,model,ci):
    obs=env.reset(14006000+ci);rng=np.random.default_rng(14007000+ci);states={}
    for step in range(1,641):
        action,_,_=transition(model,obs,rng.normal(size=6))
        obs,_=env.step(action)
        if step in TIMES:states[step]=env.snapshot()
    return states


def main():
    torch.set_num_threads(1);started=time.monotonic()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    design=json.loads((WORK/'d02-design.json').read_text())
    assert design['counts']['total_env_upper']==337920 and design['replicates']==4
    assert design['main_delta']==.05 and design['sensitivity_delta']==.025
    assert design['pilot']['checkpoint']=='online25000' and design['pilot']['state']==16 and design['pilot']['direction']==0
    assert shutil.disk_usage(ROOT).free>10*1024**3
    cpu=json.loads((WORK/'d02-cpu-semantics-v2/validation.json').read_text());assert cpu['status']=='passed'
    preflight=sum(json.loads(p.read_text())['total_env_steps_including_constructor'] for p in WORK.glob('d02-cpu-semantics-v*/counts.json'))
    assert preflight<=2048
    OUT.mkdir(parents=True,exist_ok=False)
    report={'status':'running','code_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'design_sha256':digest(WORK/'d02-design.json'),'CPU_preflight_env_steps':preflight,'cells':[],
            'optimizer_updates':0,'training_env_steps':0,'train_buffer_writes':0,'device':'CPU; all job wall conservatively debited to GPU ledger'}
    atomic_json(OUT/'report.json',report)
    env=ReferenceEnv(OUT/'counts.json',384000-preflight)
    def interrupted(*_):raise InterruptedError('supervisor stop signal')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        for ci,label in ((1,'online25000'),(0,'offline3M'),(2,'online100000')):
            model,ident=load_frozen(label);cpu_rng=torch.get_rng_state().clone()
            states=collect_states(env,model,ci)
            with (OUT/(label+'-simstates.pkl')).open('wb') as f:pickle.dump(states,f)
            report.setdefault('checkpoints',[]).append(ident)
            for step in TIMES:
                for di in (0,1):
                    cell=case(env,states[step],model,ident,ci,step,di)
                    report['cells'].append(cell);report['current_env_steps']=env.steps
                    atomic_json(OUT/'report.json',report)
                    if len(report['cells'])==1:
                        assert label=='online25000' and step==16 and di==0
                        report['reference_pilot_pass']=cell['reference_direction_stable']
                        if not report['reference_pilot_pass']:
                            report['stop_reason']='predeclared_reference_futility; no extra reference samples or alternative pilot'
                            break
                if report.get('reference_pilot_pass') is False:break
            assert tensor_hash(model.state_dict())==ident['model_hash']
            assert torch.equal(cpu_rng,torch.get_rng_state())
            report.setdefault('verified_unchanged',[]).append(label)
            if report.get('reference_pilot_pass') is False:break
        report.update(status='completed',env_steps=env.steps,total_D02_env_steps=env.steps+preflight,
                      wall_seconds=time.monotonic()-started,
                      audited_opposite_examples=sum(c['audited_opposite_example'] for c in report['cells']),
                      no_online_method_admitted=True,limitations=['finiteH/delta,4CRN uncertainty','offline regularized EDAC policy-value proxy differs from online PVU surrogate; no cross-phase bias ranking','frozen operator reference does not identify lag/nonstationarity/repr/dropout causes','not an independent training repeat or true gradient oracle'])
    except BaseException as exc:
        report.update(status='partial',error=repr(exc),env_steps=env.steps,total_D02_env_steps=env.steps+preflight)
        raise
    finally:
        env.close();atomic_json(OUT/'report.json',report)
    print(json.dumps({k:report[k] for k in ('status','reference_pilot_pass','env_steps','total_D02_env_steps','wall_seconds','audited_opposite_examples')}))
if __name__=='__main__':main()
