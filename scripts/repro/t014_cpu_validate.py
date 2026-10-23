"""CPU correctness/identity check; does not initialize CUDA or update weights."""
import json
from t014_diagnose import *

def main():
    torch.set_num_threads(2);torch.manual_seed(140014)
    model=make_model('cpu');model.train();obs=torch.randn(8,17)
    gen=torch.Generator().manual_seed(1414)
    errors={}
    for a,k in ARMS:
        eps=torch.randn(a,8,6,generator=gen)
        before=torch.get_rng_state().clone()
        ours=actor_loss(model,obs,eps,k,1415)
        assert torch.equal(before,torch.get_rng_state())
        g1=torch.autograd.grad(ours,tuple(model.actor.parameters()))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1415);terms=[]
            for e in eps:
                mu,sig=model(obs,mode='compute_actor')['logit']
                z=mu+sig*e;act=z.tanh()
                lp=torch.distributions.Normal(mu,sig).log_prob(z).sum(-1)-torch.log(1-act.square()+1e-6).sum(-1)
                for _ in range(k):
                    q=model({'obs':obs,'action':act},mode='compute_critic')['q_value']
                    terms.append((.2*lp-q.mean(0)).mean())
            ref=torch.stack(terms).mean()
        g2=torch.autograd.grad(ref,tuple(model.actor.parameters()))
        err=max((x-y).abs().max().item() for x,y in zip(g1,g2))
        assert err<2e-6,(a,k,err)
        errors[f'A{a}K{k}']=err
    ids=[]
    for label,path,sha in checkpoints():
        assert digest(path)==sha
        state=torch.load(path,map_location='cpu',weights_only=False)
        model.load_state_dict(state['model'],strict=True)
        ids.append({'label':label,'sha256':sha,'model_hash':tensor_hash(model.state_dict())})
    assert not torch.cuda.is_initialized()
    atomic_json(WORK/'cpu-validation.json',{'status':'passed','explicit_entropy_and_Q_gradient_max_abs_errors':errors,'strict_checkpoint_loads':ids,'toy':toy_checks(),'optimizer_updates':0,'environment_interactions':0,'cuda_initialized':False})
    print(json.dumps(errors))
if __name__=='__main__':main()
