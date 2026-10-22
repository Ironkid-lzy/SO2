"""Bounded real-learner parity, analytic gradient, stream and loading contracts."""
import ast, copy, json, pickle, subprocess, time, types
from unittest.mock import patch
from torch.distributions import Independent
from t013_common import *
from t011_validate import equal_rng, parity
from t010_validate import check
BASE='5a826f88fba159f7a7c4c14be9b53d86bbadbb71'
def reference_forward():
    import ding.policy.edac as module
    text=subprocess.check_output(['git','show',BASE+':SO2/ding/policy/edac.py'],text=True)
    cls=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name=='EDACPolicy')
    fun=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_forward_learn')
    ns=dict(vars(module));exec(compile(ast.Module(body=[fun],type_ignores=[]),'frozen_main','exec'),ns)
    return ns['_forward_learn']
def main():
    torch.set_num_threads(2);start=time.monotonic();out=VALIDATION;out.mkdir(parents=True,exist_ok=True)
    result={'legacy_min_and_mean':parity()}
    seed_all(142,True)
    batch=[{'obs':torch.randn(17),'next_obs':torch.randn(17),'action':torch.rand(6)*2-1,'reward':torch.randn(1),'done':False} for _ in range(2560)]
    a=make_policy(samples=1); b=make_policy(samples=1)
    b.learn_mode.load_state_dict(copy.deepcopy(a.learn_mode.state_dict())); b._forward_learn=types.MethodType(reference_forward(),b)
    for policy in (a,b):policy._cfg.learn.online=True;policy._forward_learn_cnt=9
    state=rng_state();oa=a._forward_learn(copy.deepcopy(batch));ra=rng_state()
    restore_rng(state);ob=b._forward_learn(copy.deepcopy(batch));equal_rng(ra,rng_state())
    for k in oa:
        if isinstance(oa[k],torch.Tensor):assert torch.equal(oa[k],ob[k]),k
        else:assert oa[k]==ob[k],k
    for key,x in a._model.state_dict().items():assert torch.equal(x,b._model.state_dict()[key]),key
    for key,x in a._target_model.state_dict().items():assert torch.equal(x,b._target_model.state_dict()[key]),key
    result['K1_mean_frozen_main_loss_weights_rng_exact']=True
    a=make_policy(samples=1);b=make_policy(samples=4);b.learn_mode.load_state_dict(copy.deepcopy(a.learn_mode.state_dict()))
    captures=[]
    for policy in (a,b):
        policy._cfg.learn.online=True;policy._forward_learn_cnt=9
        capture={'qs':[],'actions':[],'actor_calls':0};captures.append(capture)
        orig=policy._learn_model.forward
        def wrapped(*args,_orig=orig,_capture=capture,**kw):
            value=_orig(*args,**kw)
            if kw.get('mode')=='compute_actor':_capture['logit']=value['logit'];_capture['actor_calls']+=1
            if kw.get('mode')=='compute_critic' and args[0]['action'].requires_grad:
                _capture['qs'].append(value['q_value']);_capture['actions'].append(args[0]['action'])
            return value
        policy._learn_model.forward=wrapped
    old_backward=torch.Tensor.backward
    def backward(self,*args,**kw):return old_backward(self,*args,**{**kw,'retain_graph':True})
    old_rsample=Independent.rsample
    def rsample(self,*args,**kw):
        value=old_rsample(self,*args,**kw)
        if value.requires_grad: captures[1]['pred']=value
        return value
    b._optimizer_policy.step=lambda:None
    state=rng_state();oa=a._forward_learn(copy.deepcopy(batch));ra=rng_state();restore_rng(state)
    with patch.object(torch.Tensor,'backward',backward),patch.object(Independent,'rsample',rsample):ob=b._forward_learn(copy.deepcopy(batch))
    equal_rng(ra,rng_state())
    c=captures[1];assert len(c['qs'])==4 and c['actor_calls']==2
    assert all(x is c['actions'][0] for x in c['actions'])
    assert torch.equal(captures[0]['actions'][0],c['actions'][0])
    assert torch.equal(captures[0]['qs'][0],c['qs'][0])
    assert oa['critic_loss']==ob['critic_loss'] and oa['target_q_value']==ob['target_q_value']
    assert a._forward_learn_cnt==b._forward_learn_cnt==10
    for key,x in a._model.critic.state_dict().items():assert torch.equal(x,b._model.critic.state_dict()[key])
    mu,sigma=c['logit'];action=c['actions'][0];pred=c['pred']
    logp=Independent(Normal(mu,sigma),1).log_prob(pred).unsqueeze(-1)-torch.log(1-action.square()+1e-6).sum(-1,keepdim=True)
    expected=(b._alpha*logp-torch.stack(c['qs']).mean(0).mean(0).unsqueeze(-1)).mean()
    params=list(b._model.actor.parameters());grads=torch.autograd.grad(expected,params)
    error=max(check(p.grad,g,'K4 actor',{'atol':2e-6,'rtol':2e-5}) for p,g in zip(params,grads))
    assert expected.item()==ob['policy_loss'].item()
    result.update(K4_explicit_loss_equal=True,K4_actor_gradient_max_abs_error=error,same_action_four_masks=True,entropy_once=True,global_rng_equal_K1_K4=True,first_mask_equal=True,critic_loss_target_weights_equal=True,actor_forward_calls=2)
    # Check next masks after state roundtrip, while restoring the external global stream.
    saved=copy.deepcopy(b.learn_mode.state_dict());reload=make_policy(samples=4);reload.learn_mode.load_state_dict(saved)
    b._learn_model.train();reload._learn_model.train();data={'obs':torch.randn(64,17,device='cuda'),'action':torch.randn(64,6,device='cuda')}
    state=rng_state();x=b._actor_mask_q(data);rx=rng_state();restore_rng(state);y=reload._actor_mask_q(data);equal_rng(rx,rng_state());assert torch.equal(x,y)
    result['mask_stream_roundtrip_exact']=True
    # CPU path also isolates additional draws.
    cpu=make_policy('cpu',samples=4);cpu._learn_model.train();d={'obs':torch.randn(8,17),'action':torch.randn(8,6)}
    state=rng_state();cpu._actor_mask_q(d);end=rng_state();restore_rng(state);cpu._learn_model.forward(d,mode='compute_critic');equal_rng(end,rng_state());result['cpu_rng_isolation']=True
    for bad in (0,-1,1.5,True):
        try:make_policy(samples=bad)
        except ValueError:pass
        else:raise AssertionError('invalid K accepted')
    # Frozen parent through the actual default before_run hook (zero updates).
    import t011_loading as loading
    loading.make_policy=make_policy
    m=parent_manifest(0); hook=out/'hook-s0';hook.mkdir()
    loading.validate(hook,Path(m['parent_path']),m,0)
    result['parent_hook']=json.loads((hook/'validation.json').read_text())
    # Measure full learner overhead, same batch size and actor update frequency.
    timing={}
    for k in (1,4):
        policy=make_policy(samples=k);policy._cfg.learn.online=True
        for _ in range(10):policy._forward_learn(copy.deepcopy(batch))
        torch.cuda.synchronize();t=time.monotonic()
        for _ in range(400):policy._forward_learn(copy.deepcopy(batch))
        torch.cuda.synchronize();timing[str(k)]=(time.monotonic()-t)/400
    result.update(timing_seconds_per_critic_update=timing,timing_ratio_K4_K1=timing['4']/timing['1'],fixed_batch_update_calls=829,status='passed',code_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),wall_seconds=time.monotonic()-start,train_env_steps=0,evaluation_env_steps=0)
    atomic_json(out/'validation.json',result);print(json.dumps({k:v for k,v in result.items() if k!='parent_hook'}))
if __name__=='__main__':main()
