"""Bounded actual CPU environment semantics. No scientific calibration results."""
import argparse, inspect, json, math, pickle, signal, time
from pathlib import Path
from t014_bias_env import *
from t014_reference_math import paired_direction_summary
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'_so2_work/t014/d02-cpu-semantics-v1'

def main():
    global OUT
    ap=argparse.ArgumentParser();ap.add_argument('--attempt',type=int,default=1);args=ap.parse_args()
    OUT=ROOT/f'_so2_work/t014/d02-cpu-semantics-v{args.attempt}'
    torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=False)
    env=ReferenceEnv(OUT/'counts.json',2048); report={'status':'running','CUDA_initialized':False}
    try:
        obs=env.reset(140021);model,ident=load_frozen('online25000')
        gen=np.random.default_rng(140022)
        for _ in range(32):obs,_=env.step(gen.uniform(-.8,.8,6).astype(np.float32))
        snap=env.snapshot();pickle.dump(snap,open(OUT/'sim_snapshot.pkl','wb'))
        actions=gen.uniform(-.8,.8,(64,6)).astype(np.float32)
        def replay():
            env.restore(snap);return np.stack([np.r_[env.step(a)[0],env.base.sim.data.qpos,env.base.sim.data.qvel] for a in actions])
        a=replay();b=replay();assert np.array_equal(a,b)
        report['full_state_restore_64step_exact']=True
        # TimeLimit wrapper is preserved, while diagnostic deliberately steps base continuing dynamics.
        env.restore(snap)
        for w in env.chain:
            if '_elapsed_steps' in vars(w):w._elapsed_steps=w._max_episode_steps
        for _ in range(4):env.step(np.zeros(6,dtype=np.float32))
        env.restore(snap)
        assert all(vars(env.chain[i])['_elapsed_steps']==x for i,x in snap['elapsed'])
        report['time_limit_not_absorbing_and_elapsed_restored']=True
        # Direct native target-action/logpi formula; independent fixed eps and perturbation.
        obs=env.restore(snap);epsilon=gen.normal(size=6).astype(np.float32);noise=np.array([2,-2,.2,-.2,0,.1],np.float32)
        act,lp,z=transition(model,obs,epsilon,noise)
        with torch.no_grad():
            mu,sigma=model(torch.tensor(obs,dtype=torch.float32)[None],mode='compute_actor')['logit']
            zz=mu+sigma*torch.tensor(epsilon)[None];base=zz.tanh()
            native_lp=Independent(Normal(mu,sigma),1).log_prob(zz)-torch.log(1-base.square()+1e-6).sum(-1)
            native_act=(base+torch.tensor(noise)[None].clamp(-.6,.6)).clamp(-1,1)
        assert np.array_equal(act,native_act[0].numpy()) and lp==native_lp.item()
        report['native_PVU_latent_entropy_exact']=True
        # Paired noise, each path uses its own evolving observation/policy logits.
        eps=gen.normal(size=(128,6)).astype(np.float32);nu=(gen.normal(size=(128,6))*.3).astype(np.float32)
        first=np.tanh(z).astype(np.float32)
        start=time.monotonic();r1,l1=rollout(env,snap,model,first,eps,nu,.2,True)
        r2,l2=rollout(env,snap,model,first,eps,nu,.2,True)
        assert np.array_equal(r1,r2) and np.array_equal(l1,l2)
        elapsed=time.monotonic()-start; report['same_CRN_full_rollout_exact']=True
        report['seconds_per_env_step_two128_rollouts']=elapsed/256
        report['first_logpi_zero']=bool(l1[0]==0)
        assert l1[0]==0 and soft_q_return(r1,l1,.99,.2)==soft_q_return(r2,l2,.99,.2)
        report['online_identity']=ident
        off,oid=load_frozen('offline3M');assert abs(oid['alpha']-.08807972073554993)<1e-12
        report['offline_alpha_from_checkpoint']=oid['alpha']
        assert not torch.cuda.is_initialized()
        report.update(status='passed',env_steps=env.steps,optimizer_updates=0,
                      environment_source={'path':inspect.getfile(type(env.base)),'sha256':digest(inspect.getfile(type(env.base)))},
                      reference_kind='online PVU/latent-entropy surrogate; offline frozen-policy value with saved alpha, not exact regularized EDAC objective',
                      production_337920_seconds_estimate_with_safety=337920*report['seconds_per_env_step_two128_rollouts']*1.3)
        np.savez_compressed(OUT/'replay_128.npz',rewards=r1,logps=l1,eps=eps,pvu=nu)
    finally:
        env.close();atomic_json(OUT/'validation.json',report)
    print(json.dumps(report))
if __name__=='__main__':main()
