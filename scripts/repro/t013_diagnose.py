"""Frozen-parent mask variability, not ground-truth Q error; zero optimizer updates."""
import copy, json, subprocess, time
import h5py
from torch.distributions import Independent
from t013_common import *
from t011_validate import equal_rng

def summary(samples):
    x=torch.stack(samples).double();y=x.reshape(-1,4,*x.shape[1:]).mean(1)
    v1=x.var(0,unbiased=True).mean().item();v4=y.var(0,unbiased=True).mean().item()
    return {'K1_mean_conditional_variance':v1,'K4_mean_conditional_variance':v4,'variance_ratio_K4_K1':v4/v1,'K1_mask_rms_std':v1**.5,'K4_mask_rms_std':v4**.5,'pool_samples':len(x),'independent_K4_groups':len(y)}
def main():
    torch.set_num_threads(2);start=time.monotonic();out=ROOT/'_so2_work/diagnostics/t013';out.mkdir(parents=True)
    m=parent_manifest(0);parent=Path(m['parent_path'])
    assert digest(parent)==PARENT_HASHES[0]
    saved=torch.load(m['export_path'],map_location='cuda',weights_only=True)
    policy=make_policy(samples=4);policy.learn_mode.load_state_dict(saved)
    policy._learn_model.train();policy._target_model.train()
    with h5py.File(DATASET,'r') as f:
        # Fixed strided offline observations; no rollout or replay writes.
        idx=np.arange(256)*700
        obs=torch.tensor(f['observations'][idx],device='cuda',dtype=torch.float32)
        nxt=torch.tensor(f['next_observations'][idx],device='cuda',dtype=torch.float32)
        rew=torch.tensor(f['rewards'][idx],device='cuda',dtype=torch.float32)
        done=torch.tensor(f['terminals'][idx],device='cuda',dtype=torch.float32)
    before=rng_state();qvals=[];targets=[];grads=[]
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(314013);torch.cuda.manual_seed(314013)
        epsilon=torch.randn(256,6,device='cuda');next_eps=torch.randn_like(epsilon)
        noise=(torch.randn_like(epsilon)*.3).clamp(-.6,.6)
        with torch.no_grad():
            mu,sigma=policy._learn_model.forward(nxt,mode='compute_actor')['logit']
            pred=mu+sigma*next_eps;next_action=pred.tanh()
            logp=Independent(Normal(mu,sigma),1).log_prob(pred)-torch.log(1-next_action.square()+1e-6).sum(-1)
            next_action=(next_action+noise).clamp(-1,1)
        fixed_action=None
        for i in range(256):
            mu,sigma=policy._learn_model.forward(obs,mode='compute_actor')['logit']
            action=(mu+sigma*epsilon).tanh()
            if fixed_action is None:fixed_action=action.detach().clone()
            else:assert torch.equal(action.detach(),fixed_action)
            q=policy._learn_model.forward({'obs':obs,'action':action},mode='compute_critic')['q_value']
            g=torch.autograd.grad(-q.mean(),list(policy._model.actor.parameters()))
            grads.append(torch.cat([x.detach().flatten().cpu() for x in g]));qvals.append(q.detach().cpu())
            with torch.no_grad():
                tq=policy._target_model.forward({'obs':nxt,'action':next_action},mode='compute_critic')['q_value']
                target=rew[None,:]+.99*(1-done[None,:])*(tq-.2*logp[None,:])
                targets.append(target.cpu())
    equal_rng(before,rng_state())
    assert all(torch.isfinite(x).all() for seq in (qvals,targets,grads) for x in seq)
    report={'code_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'parent_sha256':digest(parent),'export_sha256':digest(m['export_path']),'dataset_sha256':digest(DATASET),'fixed_data_indices':idx.tolist(),'diagnostic_seed':314013,'same_reparameterized_action':True,'global_rng_preserved':True,'optimizer_updates':0,'train_env_steps':0,'eval_env_steps':0,'Q_perhead':summary(qvals),'Q_head_mean':summary([x.mean(0) for x in qvals]),'SO2_perhead_bootstrap':summary(targets),'actor_Q_gradient':summary(grads),'interpretation':'Conditional mask variability only; K4 target averaging is a diagnostic reference and is NOT applied in training. Entropy is fixed across masks. No ground-truth Q-error estimate.'}
    # Actual training throughput: no deepcopy overhead, same existing batch list.
    batch=[{'obs':torch.randn(17),'next_obs':torch.randn(17),'action':torch.rand(6)*2-1,'reward':torch.randn(1),'done':False} for _ in range(2560)]
    timing={}
    for k in (1,4):
        p=make_policy(samples=k);p._cfg.learn.online=True
        for _ in range(20):p._forward_learn(batch)
        torch.cuda.synchronize();t=time.monotonic()
        for _ in range(1000):p._forward_learn(batch)
        torch.cuda.synchronize();timing[str(k)]=(time.monotonic()-t)/1000
    report.update(timing_seconds_per_update_no_deepcopy=timing,timing_ratio=timing['4']/timing['1'],timing_optimizer_calls=2040,wall_seconds=time.monotonic()-start)
    atomic_json(out/'diagnostics.json',report);print(json.dumps(report))
if __name__=='__main__':main()
