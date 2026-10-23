"""Frozen-policy continuing HalfCheetah reference; never updates training state."""
import copy, inspect, json, math, pickle, time
from pathlib import Path
import gym
import d4rl
import mujoco_py
import numpy as np
import torch
from torch.distributions import Independent, Normal
from t010_edac import make_model, digest, tensor_hash, atomic_json
from t014_diagnose import checkpoints
from t014_reference_math import soft_q_return


class ReferenceEnv:
    def __init__(self, count_path, cap):
        self.count_path, self.cap = Path(count_path), cap
        self.env = gym.make('halfcheetah-medium-replay-v2')
        self.chain=[]; obj=self.env
        while True:
            self.chain.append(obj)
            if '_wrapped_env' in vars(obj): obj=vars(obj)['_wrapped_env']
            elif 'env' in vars(obj): obj=vars(obj)['env']
            else: break
        self.base=obj
        assert type(self.base).__name__ == 'OfflineHalfCheetahEnv'
        assert np.all(self.base.action_space.low == -1) and np.all(self.base.action_space.high == 1)
        for w in self.chain:
            if type(w).__name__ == 'NormalizedBoxEnv':
                assert w._should_normalize is False and w._reward_scale == 1.
        # Gym MujocoEnv constructor performs one internal step to infer obs shape.
        assert abs(self.base.sim.data.time-self.base.dt)<1e-12
        self.steps=1; self.flush()

    def flush(self):
        atomic_json(self.count_path, {'total_env_steps_including_constructor':self.steps,
                    'cap':self.cap,'training_env_steps':0,'training_buffer_writes':0})

    def reset(self, seed):
        self.env.seed(seed)
        o=self.env.reset()
        assert np.array_equal(o,self.base._get_obs()) and o.shape==(17,)
        return o

    def step(self, action):
        assert self.steps < self.cap
        assert np.isfinite(action).all() and np.max(np.abs(action))<=1
        self.steps+=1
        obs,reward,done,info=self.base.step(action)
        assert done is False, 'No physical terminal expected in continuing HalfCheetah'
        assert np.isfinite(obs).all() and math.isfinite(reward)
        if self.steps%128==0:self.flush()
        return obs,float(reward)

    def snapshot(self):
        sim=self.base.sim; st=sim.get_state()
        arrays={name:getattr(sim.data,name).copy() for name in (
            'qacc_warmstart','qacc','ctrl','qfrc_applied','xfrc_applied','mocap_pos','mocap_quat','userdata')}
        return {'sim':{'time':st.time,'qpos':st.qpos.copy(),'qvel':st.qvel.copy(),
                       'act':st.act.copy(),'udd_state':copy.deepcopy(st.udd_state)},
                'arrays':arrays,'np_random':copy.deepcopy(self.base.np_random),
                'elapsed':[(i,copy.deepcopy(vars(w)['_elapsed_steps'])) for i,w in enumerate(self.chain) if '_elapsed_steps' in vars(w)],
                'obs':self.base._get_obs().copy()}

    def restore(self, snap):
        s=snap['sim']; sim=self.base.sim
        sim.set_state(mujoco_py.MjSimState(s['time'],s['qpos'].copy(),s['qvel'].copy(),s['act'].copy(),copy.deepcopy(s['udd_state'])))
        for key,value in snap['arrays'].items():getattr(sim.data,key)[:]=value
        sim.forward()
        # forward recomputes accelerations; restore integration solver caches too.
        for key,value in snap['arrays'].items():getattr(sim.data,key)[:]=value
        self.base.np_random=copy.deepcopy(snap['np_random'])
        for i,x in snap['elapsed']:self.chain[i]._elapsed_steps=copy.deepcopy(x)
        obs=self.base._get_obs().copy()
        assert np.array_equal(obs,snap['obs'])
        return obs

    def close(self):
        self.flush(); self.env.close()


def load_frozen(label):
    _,path,sha=next(x for x in checkpoints() if x[0]==label)
    assert digest(path)==sha
    saved=torch.load(path,map_location='cpu',weights_only=False)
    model=make_model('cpu');model.load_state_dict(saved['model'],strict=True);model.train()
    alpha=float(saved['log_alpha'].exp().item()) if label=='offline3M' else .2
    assert math.isfinite(alpha) and alpha>0
    ident={'label':label,'path':str(path),'sha256':sha,'alpha':alpha,
           'PVU':label!='offline3M','model_hash':tensor_hash(model.state_dict())}
    return model,ident


@torch.inference_mode()
def transition(model,obs,epsilon,pvu_noise=None):
    x=torch.as_tensor(np.asarray(obs),dtype=torch.float32).unsqueeze(0)
    mu,sigma=model(x,mode='compute_actor')['logit']
    eps=torch.as_tensor(epsilon,dtype=torch.float32).reshape(1,6)
    z=mu+sigma*eps; base=z.tanh()
    lp=Independent(Normal(mu,sigma),1).log_prob(z)-torch.log(1-base.square()+1e-6).sum(-1)
    action=base
    if pvu_noise is not None:
        noise=torch.as_tensor(pvu_noise,dtype=torch.float32).reshape(1,6).clamp(-.6,.6)
        action=(action+noise).clamp(-1,1)
    assert torch.isfinite(lp).all() and torch.isfinite(action).all()
    return action[0].numpy().copy(),float(lp.item()),z[0].numpy().copy()


def rollout(env,snapshot,model,first_action,eps,noise,alpha,pvu):
    obs=env.restore(snapshot); rewards=[]; logps=[]
    for t in range(len(eps)):
        if t==0:action=np.asarray(first_action,dtype=np.float32);lp=0.
        else:action,lp,_=transition(model,obs,eps[t],noise[t] if pvu else None)
        obs,reward=env.step(action);rewards.append(reward);logps.append(lp)
    return np.asarray(rewards),np.asarray(logps)


def project_masks(model,obs,z,direction,seed,n=64):
    before=torch.get_rng_state().clone(); values=[]
    x=torch.tensor(obs,dtype=torch.float32).unsqueeze(0)
    u=torch.tensor(direction,dtype=torch.float32)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for _ in range(n):
            zz=torch.tensor(z,dtype=torch.float32,requires_grad=True)
            q=model({'obs':x,'action':zz.tanh().unsqueeze(0)},mode='compute_critic')['q_value'].mean()
            grad,=torch.autograd.grad(q,zz)
            values.append(float((grad*u).sum().item()))
    assert torch.equal(before,torch.get_rng_state()) and np.isfinite(values).all()
    return values
